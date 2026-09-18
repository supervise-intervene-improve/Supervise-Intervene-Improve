
import sys
import time
import math
from pathlib import Path
from dataclasses import dataclass

import glfw
import mujoco
import numpy as np

import record

#MAX_WINDOWS = 10
MAX_WINDOWS = 17
DEFAULT_XML = "franka_emika_panda/lab1.xml"

ACTIVE_VIEWER = None
ALL_VIEWERS = []
GLOBAL_PAUSE = False
MAIN_WINDOW = None
MAIN_CTX = None
MOUSE = None
FOCUS_MODE = False
FOCUSED_VIEWER = None
REPLAN_IN_PROGRESS = False
PENDING_REPLAN = False

VIEW_HZ = 60.0
REPLAN_LOG_HZ = 60.0


MIRROR_ROBOT = False
MIRROR_ROBOT_KEY = "p1"
MIRROR_MODE = record.ControlType.HYBRID_JOINT_IMPEDANCE_CONTROL
#MAX_DQ = 0.8
MAX_DQ = 0.25
GRIP_EPS = 5e-4
GRIPPER_TRACK_EPS_M = 0.002
GRIPPER_CMD_PERIOD_S = 0.02
GRIPPER_CMD_SPEED = 0.35
GRIPPER_CMD_FORCE = 0.10

MIRROR_HANDLE = None
MIRROR_Q_PREV = None
MIRROR_GRIP_PREV = None
MIRROR_REPLAN_GRIP_SEED = None
MIRROR_GRIP_CMD_STATE = None

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


@dataclass
class MouseState:
    left_down: bool = False
    right_down: bool = False
    middle_down: bool = False
    last_x: float = 0.0
    last_y: float = 0.0


def gripper_ctrl_uses_real_width(model) -> bool:
    if model is None or model.nu < 8:
        return False

    return float(model.actuator_ctrlrange[7][1]) > 0.041


def gripper_ctrl_to_width(model, ctrl_value):
    ctrl_value = np.asarray(ctrl_value, dtype=np.float64)
    if gripper_ctrl_uses_real_width(model):
        ctrlrange = model.actuator_ctrlrange[7]
        return np.clip(ctrl_value, float(ctrlrange[0]), float(ctrlrange[1]))

    return np.clip(2.0 * ctrl_value, 0.0, 0.08)


def gripper_width_to_ctrl(model, width):
    width = float(width)
    if gripper_ctrl_uses_real_width(model):
        ctrlrange = model.actuator_ctrlrange[7]
        return float(np.clip(width, float(ctrlrange[0]), float(ctrlrange[1])))

    return float(np.clip(width / 2.0, 0.0, 0.04))


def stitch_npz(original_path: str, suffix_path: str, cut_idx: int, out_path: str, model=None):
    orig = np.load(original_path)
    suf = np.load(suffix_path)

    out = {}
    suffix_start = 0

    sim_t_offset = 0.0
    if "sim_t" in orig.files and len(orig["sim_t"]) > 0:
        sim_t_offset = float(orig["sim_t"][cut_idx])

    for k in orig.files:
        if k in suf.files and orig[k].ndim >= 1:
            left = orig[k][: cut_idx + 1]
            right = suf[k][suffix_start:]

            if k == "sim_t":
                right = right + sim_t_offset

            out[k] = np.concatenate([left, right], axis=0)
        else:
            out[k] = orig[k]

    for k in suf.files:
        if k not in out:
            if k == "grip_real":
                right = np.asarray(suf[k], dtype=np.float64).reshape(-1)[suffix_start:]
                prefix_len = int(cut_idx) + 1
                if "ctrl_sim" in orig.files:
                    ctrl_prefix = np.asarray(orig["ctrl_sim"][:prefix_len], dtype=np.float64)
                    if ctrl_prefix.ndim == 2 and ctrl_prefix.shape[1] > 7:
                        left = gripper_ctrl_to_width(model, ctrl_prefix[:, 7])
                    else:
                        left = np.full((prefix_len,), np.nan, dtype=np.float64)
                else:
                    left = np.full((prefix_len,), np.nan, dtype=np.float64)
                out[k] = np.concatenate([left, right], axis=0)
            else:
                out[k] = suf[k]

    np.savez_compressed(out_path, **out)
    print(f"[INFO] Stitched saved: {out_path}")

def rebuild_main_context():
    global MAIN_CTX

    if MAIN_WINDOW is None or not ALL_VIEWERS:
        return

    glfw.make_context_current(MAIN_WINDOW)

    if MAIN_CTX is not None:
        try:
            MAIN_CTX.free()
        except Exception:
            pass

    MAIN_CTX = mujoco.MjrContext(
        ALL_VIEWERS[0].model,
        mujoco.mjtFontScale.mjFONTSCALE_100,
    )

def clamp_step(q_cmd, q_prev, max_dq, dt):
    dq = (q_cmd - q_prev) / dt
    dq = np.clip(dq, -max_dq, max_dq)
    return q_prev + dq * dt


def get_viewer_gripper_width(viewer):
    if viewer is None:
        return None

    grip_seq = getattr(viewer, "grip_real_seq", None)
    if grip_seq is not None and len(grip_seq) > 0:
        idx = int(clamp(viewer.frame_idx, 0, len(grip_seq) - 1))
        return float(grip_seq[idx])

    if viewer.model.nu >= 8:
        return float(gripper_ctrl_to_width(viewer.model, float(viewer.data.ctrl[7])))

    return None


def _new_gripper_cmd_state() -> dict:
    return {
        "last_width_cmd": None,
        "last_cmd_t": 0.0,
        "future": None,
        "last_error_t": 0.0,
    }


def apply_gripper_width_command(robot, width_m: float, cmd_state: dict) -> None:
    if robot is None:
        return

    hand = robot.robot_gripper
    min_width = float(getattr(hand, "min_width", 0.0))
    max_width = float(getattr(hand, "max_width", 0.08) or 0.08)
    target = float(np.clip(width_m, min_width, max_width))
    now = time.time()

    last_width_cmd = cmd_state.get("last_width_cmd", None)
    last_cmd_t = float(cmd_state.get("last_cmd_t", 0.0))
    if (
        last_width_cmd is not None
        and abs(target - float(last_width_cmd)) < float(GRIPPER_TRACK_EPS_M)
        and (now - last_cmd_t) < float(GRIPPER_CMD_PERIOD_S)
    ):
        return

    try:
        low_level = getattr(hand, "robot", None)
        if low_level is not None and hasattr(low_level, "goto"):
            pool = getattr(hand, "pool", None)
            if pool is not None:
                fut = cmd_state.get("future", None)
                if fut is not None and not fut.done():
                    return
                cmd_state["future"] = pool.submit(
                    low_level.goto,
                    target,
                    float(GRIPPER_CMD_SPEED),
                    float(GRIPPER_CMD_FORCE),
                )
            else:
                low_level.goto(
                    target,
                    float(GRIPPER_CMD_SPEED),
                    float(GRIPPER_CMD_FORCE),
                )
        else:
            current_width = float(hand.get_sensors().item())
            err = target - current_width
            if abs(err) < float(GRIPPER_TRACK_EPS_M):
                return
            cmd = 1.0 if err > 0.0 else -1.0
            hand.apply_commands(
                cmd,
                speed=float(GRIPPER_CMD_SPEED),
                force=float(GRIPPER_CMD_FORCE),
            )
    except Exception as e:
        last_error_t = float(cmd_state.get("last_error_t", 0.0))
        if (now - last_error_t) > 1.0:
            print(f"[MIRROR][WARN] Gripper command failed: {e}")
            cmd_state["last_error_t"] = now
        return

    cmd_state["last_width_cmd"] = target
    cmd_state["last_cmd_t"] = now


def restore_gripper_width_binary(
    robot,
    target_width: float,
    tol: float = 0.002,
    timeout: float = 1.5,
    hz: float = 40.0,
    cmd_state: dict | None = None,
):
    if robot is None:
        return

    min_width = float(getattr(robot.robot_gripper, "min_width", 0.0))
    max_width = float(getattr(robot.robot_gripper, "max_width", 0.08) or 0.08)
    target_width = float(np.clip(target_width, min_width, max_width))

    if cmd_state is None:
        cmd_state = _new_gripper_cmd_state()

    dt = 1.0 / hz
    t_end = time.time() + timeout

    while time.time() < t_end:
        current_width = float(robot.robot_gripper.get_sensors().item())
        err = target_width - current_width

        if abs(err) <= tol:
            break

        apply_gripper_width_command(robot, target_width, cmd_state)
        time.sleep(dt)

    final_width = float(robot.robot_gripper.get_sensors().item())
    print(
        f"[MIRROR] Gripper restored: target={target_width:.4f}, "
        f"final={final_width:.4f}"
    )


class TrajectoryPlayer:
    def __init__(self, xml_path: str, npz_path: str, title: str):
        self.xml_path = str(xml_path)
        self.npz_path = str(npz_path)
        self.title = title

        self.log = np.load(self.npz_path)

        required = ["sim_t"]
        for k in required:
            if k not in self.log.files:
                raise RuntimeError(f"{self.npz_path} is missing required key '{k}'")

        self.sim_t = np.asarray(self.log["sim_t"], dtype=np.float64)
        self.nframes = len(self.sim_t)
        if self.nframes == 0:
            raise RuntimeError(f"{self.npz_path} contains zero frames")

        self.has_qpos = "qpos_sim" in self.log.files
        self.has_qvel = "qvel_sim" in self.log.files
        self.has_ctrl = "ctrl_sim" in self.log.files

        self.has_grip_real = "grip_real" in self.log.files

        self.qpos_seq = self.log["qpos_sim"] if self.has_qpos else None
        self.qvel_seq = self.log["qvel_sim"] if self.has_qvel else None
        self.ctrl_seq = self.log["ctrl_sim"] if self.has_ctrl else None

        self.grip_real_seq = self.log["grip_real"] if self.has_grip_real else None
        
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.scn = mujoco.MjvScene(self.model, maxgeom=2000)

        mujoco.mjv_defaultCamera(self.cam)
        mujoco.mjv_defaultOption(self.opt)

        self.set_fast_visuals()

        self.viewport = mujoco.MjrRect(0, 0, 1, 1)

        self.frame_idx = 0
        self.is_paused = False
        self.closed = False

        self.play_start_wall = None
        self.play_start_sim = None

        self.reset_to_start()


    

    def reload_from_npz(self, npz_path: str, start_idx: int | None = None, play: bool = False):
        self.npz_path = str(npz_path)
        self.log = np.load(self.npz_path)

        required = ["sim_t"]
        for k in required:
            if k not in self.log.files:
                raise RuntimeError(f"{self.npz_path} is missing required key '{k}'")

        self.sim_t = np.asarray(self.log["sim_t"], dtype=np.float64)
        self.nframes = len(self.sim_t)
        if self.nframes == 0:
            raise RuntimeError(f"{self.npz_path} contains zero frames")

        self.has_qpos = "qpos_sim" in self.log.files
        self.has_qvel = "qvel_sim" in self.log.files
        self.has_ctrl = "ctrl_sim" in self.log.files
        self.has_grip_real = "grip_real" in self.log.files

        self.qpos_seq = self.log["qpos_sim"] if self.has_qpos else None
        self.qvel_seq = self.log["qvel_sim"] if self.has_qvel else None
        self.ctrl_seq = self.log["ctrl_sim"] if self.has_ctrl else None
        self.grip_real_seq = self.log["grip_real"] if self.has_grip_real else None

        if start_idx is None:
            self.reset_to_start(play=play)
        else:
            idx = int(clamp(start_idx, 0, self.nframes - 1))
            self._apply_frame(idx)
            self.play_start_wall = time.perf_counter()
            self.play_start_sim = float(self.sim_t[self.frame_idx])
            self.is_paused = not play

    def set_fast_visuals(self):
        self.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        self.scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        self.scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
        self.scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = 0


    def set_pretty_visuals(self):
        self.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
        self.scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 1
        self.scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1
        self.scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1

    def reset_to_start(self, play=True):
        self.frame_idx = 0
        self._apply_frame(self.frame_idx)
        self.play_start_wall = time.perf_counter()
        self.play_start_sim = float(self.sim_t[self.frame_idx])
        if play:
            self.is_paused = False

    def restart_clock_from_current_frame(self):
        self.play_start_wall = time.perf_counter()
        self.play_start_sim = float(self.sim_t[self.frame_idx])

    def _apply_frame(self, idx: int):
        idx = int(clamp(idx, 0, self.nframes - 1))
        self.frame_idx = idx

        if self.has_qpos:
            if self.qpos_seq[idx].shape != self.data.qpos.shape:
                raise RuntimeError(
                    f"{self.npz_path}: qpos_sim frame shape {self.qpos_seq[idx].shape} "
                    f"!= model qpos shape {self.data.qpos.shape}"
                )
            self.data.qpos[:] = self.qpos_seq[idx]

        if self.has_qvel:
            if self.qvel_seq[idx].shape != self.data.qvel.shape:
                raise RuntimeError(
                    f"{self.npz_path}: qvel_sim frame shape {self.qvel_seq[idx].shape} "
                    f"!= model qvel shape {self.data.qvel.shape}"
                )
            self.data.qvel[:] = self.qvel_seq[idx]
        else:
            self.data.qvel[:] = 0.0

        if self.has_ctrl:
            n = min(len(self.data.ctrl), self.ctrl_seq[idx].shape[0])
            self.data.ctrl[:n] = self.ctrl_seq[idx][:n]

        self.data.time = float(self.sim_t[idx])
        mujoco.mj_forward(self.model, self.data)

    def step_frame(self, delta: int):
        new_idx = clamp(self.frame_idx + delta, 0, self.nframes - 1)
        self._apply_frame(new_idx)
        self.restart_clock_from_current_frame()

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        if not self.is_paused:
            self.restart_clock_from_current_frame()

    def update(self):
        if self.closed:
            return

        if not self.is_paused and not GLOBAL_PAUSE:
            target_sim = self.play_start_sim + (time.perf_counter() - self.play_start_wall)

            idx = self.frame_idx
            while idx + 1 < self.nframes and self.sim_t[idx + 1] <= target_sim:
                idx += 1

            if idx != self.frame_idx:
                self._apply_frame(idx)

            if self.frame_idx >= self.nframes - 1:
                self.is_paused = True

    def render(self, ctx):
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.opt,
            None,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scn,
        )
        mujoco.mjr_render(self.viewport, self.scn, ctx)
        self._draw_overlay(ctx)

    def _draw_overlay(self, ctx):

        active_mark = "ACTIVE" if self is ACTIVE_VIEWER else ""
        ended_state = self.frame_idx >= self.nframes - 1

        mirror_state = "MIRROR ON" if (FOCUS_MODE and self is FOCUSED_VIEWER and MIRROR_ROBOT) else ""

        if ended_state:
            local_state = "SIM ENDED"
        else:
            local_state = "PAUSED" if self.is_paused else "PLAYING"

        global_state = "GLOBAL-PAUSE" if GLOBAL_PAUSE else ""

        txt1 = (
            f"{Path(self.npz_path).name}\n"
            f"frame {self.frame_idx + 1}/{self.nframes} | t={self.sim_t[self.frame_idx]:.3f}"
        )
        txt2 = " ".join(s for s in [local_state, global_state, mirror_state, active_mark] if s)

        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            self.viewport,
            txt1,
            txt2,
            ctx,
        )

        if FOCUS_MODE and self is FOCUSED_VIEWER:
            help_left = (
                "R: reset active\n"
                "Space: pause active\n"
                "Left/Right: step active\n"
                "M: mirror on/off\n"
                "P: replanning"
            )
            help_right = (
                "Z: exit focus\n"
                "Esc: quit"
            )
        else:
            help_left = (
                "R: reset active\n"
                "Space: pause active\n"
                "Shift+Space: pause all\n"
                "Left/Right: step active\n"
                "Shift+Left/Right: step all"
            )
            help_right = (
                "Shift+R: reset all\n"
                "Click tile: activate\n"
                "Z: focus/unfocus active\n"
                "Esc: quit"
            )


        if self is ACTIVE_VIEWER:
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                self.viewport,
                help_left,
                help_right,
                ctx,
            )


def set_active(viewer):
    global ACTIVE_VIEWER
    ACTIVE_VIEWER = viewer


def compute_grid(n):
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def compute_viewports(win_w, win_h, n, pad=8):
    rows, cols = compute_grid(n)
    cell_w = win_w // cols
    cell_h = win_h // rows

    rects = []
    for i in range(n):
        r = i // cols
        c = i % cols

        x = c * cell_w + pad
        y_top = r * cell_h + pad
        w = max(1, cell_w - 2 * pad)
        h = max(1, cell_h - 2 * pad)

        y = win_h - (y_top + h)
        rects.append(mujoco.MjrRect(x, y, w, h))
    return rects


def pick_viewer_at_cursor(x, y, win_w, win_h):
    rects = compute_viewports(win_w, win_h, len(ALL_VIEWERS))
    y_mj = win_h - y

    for viewer, rect in zip(ALL_VIEWERS, rects):
        if rect.left <= x <= rect.left + rect.width and rect.bottom <= y_mj <= rect.bottom + rect.height:
            return viewer
    return None


def enable_mirror():
    global MIRROR_ROBOT, MIRROR_HANDLE, MIRROR_Q_PREV, MIRROR_GRIP_PREV, MIRROR_REPLAN_GRIP_SEED, MIRROR_GRIP_CMD_STATE

    if MIRROR_ROBOT:
        print("[MIRROR] Already enabled.")
        return

    if not FOCUS_MODE or FOCUSED_VIEWER is None:
        print("[MIRROR] Enter focus mode first with Z.")
        return

    try:
        robot = record.ROBOTS[MIRROR_ROBOT_KEY]
        print(f"[MIRROR] Connecting real robot {MIRROR_ROBOT_KEY} in {MIRROR_MODE}...")
        robot.connect(MIRROR_MODE)
        MIRROR_Q_PREV = robot.robot_arm.get_state().joint_pos.detach().cpu().numpy().astype(np.float64)
        MIRROR_GRIP_CMD_STATE = _new_gripper_cmd_state()

        target_grip_width = MIRROR_REPLAN_GRIP_SEED
        if target_grip_width is None:
            target_grip_width = get_viewer_gripper_width(FOCUSED_VIEWER)
        MIRROR_GRIP_PREV = target_grip_width

        if target_grip_width is not None:
            try:
                restore_gripper_width_binary(robot, target_grip_width, cmd_state=MIRROR_GRIP_CMD_STATE)
            except Exception as e:
                print(f"[MIRROR] Gripper sync warning: {e}")

        MIRROR_REPLAN_GRIP_SEED = None

        MIRROR_HANDLE = robot
        MIRROR_ROBOT = True
        print("[MIRROR] Enabled.")
    except Exception as e:
        MIRROR_HANDLE = None
        MIRROR_Q_PREV = None
        MIRROR_GRIP_PREV = None
        MIRROR_REPLAN_GRIP_SEED = None
        MIRROR_GRIP_CMD_STATE = None
        MIRROR_ROBOT = False
        print(f"[MIRROR] Failed to connect: {e}")


def disable_mirror():
    global MIRROR_ROBOT, MIRROR_HANDLE, MIRROR_Q_PREV, MIRROR_GRIP_PREV, MIRROR_GRIP_CMD_STATE

    if MIRROR_HANDLE is not None:
        try:
            print("[MIRROR] Closing robot connection...")
            MIRROR_HANDLE.close()
        except Exception as e:
            print(f"[MIRROR] Close warning: {e}")

    MIRROR_HANDLE = None
    MIRROR_Q_PREV = None
    MIRROR_GRIP_PREV = None
    MIRROR_GRIP_CMD_STATE = None
    MIRROR_ROBOT = False
    print("[MIRROR] Disabled.")


def mirror_focused_viewer_once():
    global MIRROR_Q_PREV, MIRROR_GRIP_PREV, MIRROR_GRIP_CMD_STATE

    if not MIRROR_ROBOT or MIRROR_HANDLE is None or FOCUSED_VIEWER is None:
        return

    viewer = FOCUSED_VIEWER

    q_cmd = np.asarray(viewer.data.ctrl[:7], dtype=np.float64)
    if MAX_DQ is not None and MIRROR_Q_PREV is not None:
        q_cmd = clamp_step(q_cmd, MIRROR_Q_PREV, MAX_DQ, VIEW_HZ and (1.0 / VIEW_HZ))
    MIRROR_Q_PREV = q_cmd.copy()

    MIRROR_HANDLE.robot_arm.apply_commands(q_cmd)

    grip_width_cmd = get_viewer_gripper_width(viewer)

    if grip_width_cmd is not None:
        if MIRROR_GRIP_CMD_STATE is None:
            MIRROR_GRIP_CMD_STATE = _new_gripper_cmd_state()
        apply_gripper_width_command(MIRROR_HANDLE, grip_width_cmd, MIRROR_GRIP_CMD_STATE)
        MIRROR_GRIP_PREV = grip_width_cmd


def replan_active_viewer():
    global GLOBAL_PAUSE, FOCUS_MODE, FOCUSED_VIEWER, REPLAN_IN_PROGRESS, MIRROR_ROBOT, MIRROR_HANDLE, MIRROR_Q_PREV, MIRROR_GRIP_PREV, MIRROR_REPLAN_GRIP_SEED, MIRROR_GRIP_CMD_STATE
    if REPLAN_IN_PROGRESS:
        print("[REPLAN] Already in progress.")
        return
    viewer = ACTIVE_VIEWER
    if viewer is None:
        print("[REPLAN] No active viewer.")
        return

    if not FOCUS_MODE or viewer is not FOCUSED_VIEWER:
        print("[REPLAN] Press Z first and replan only in focus mode.")
        return

    if not (viewer.is_paused or GLOBAL_PAUSE):
        print("[REPLAN] Pause the focused replay first, then press P.")
        return

    reuse_live_robot = False
    robot_replan = record.ROBOTS[record.ROBOT_KEY]

    if MIRROR_ROBOT and MIRROR_HANDLE is not None and MIRROR_ROBOT_KEY == record.ROBOT_KEY:
        print("[REPLAN] Reusing active mirror robot connection for replanning.")
        reuse_live_robot = True
        robot_replan = MIRROR_HANDLE
        MIRROR_ROBOT = False
        MIRROR_Q_PREV = None
        MIRROR_GRIP_PREV = None
        MIRROR_GRIP_CMD_STATE = None
    elif MIRROR_ROBOT:
        print("[REPLAN] Disabling mirror before replanning.")
        disable_mirror()

    REPLAN_IN_PROGRESS = True

    cut_idx = int(viewer.frame_idx)
    qpos_seed = viewer.data.qpos.copy()
    qvel_seed = viewer.data.qvel.copy()
    q_seed = np.asarray(viewer.data.ctrl[:7], dtype=np.float64).copy()

    grip_width_seed = None
    if getattr(viewer, "grip_real_seq", None) is not None:
        grip_width_seed = float(viewer.grip_real_seq[cut_idx])
        finger_seed = gripper_width_to_ctrl(viewer.model, grip_width_seed)
    elif viewer.model.nu >= 8:
        finger_seed = float(viewer.data.ctrl[7])
        grip_width_seed = float(gripper_ctrl_to_width(viewer.model, finger_seed))
    else:
        finger_seed = 0.0

    MIRROR_REPLAN_GRIP_SEED = grip_width_seed

    log_path = str(viewer.npz_path)
    xml_path = str(viewer.xml_path)

    suffix_path = str(Path(log_path).with_name(f"suffix_{int(time.time())}.npz"))
    out_path = str(
        Path(log_path).with_name(
            Path(log_path).stem + f"_replanned_{int(time.time())}.npz"
        )
    )
    
    print(f"[REPLAN] Starting replanning from frame {cut_idx} of {Path(log_path).name}")

    old_global_pause = GLOBAL_PAUSE
    GLOBAL_PAUSE = True
    viewer.is_paused = True

    if MAIN_WINDOW is not None:
        glfw.make_context_current(None)

    try:
        record.hold_then_replan_record(
            robot=robot_replan,
            q_hold=q_seed,
            finger_hold=finger_seed,
            grip_width_hold=grip_width_seed,
            save_path=Path(suffix_path),
            mujoco_xml_path=xml_path,
            view_hz=VIEW_HZ,
            log_hz=REPLAN_LOG_HZ,
            alpha=record.ALPHA,
            qpos_seed=qpos_seed,
            qvel_seed=qvel_seed,
            already_connected=reuse_live_robot,
        )

        stitch_npz(log_path, suffix_path, cut_idx, out_path, model=viewer.model)
        print(f"[REPLAN] Reloading focused viewer with {Path(out_path).name}")

        resume_idx = cut_idx + 1
        viewer.reload_from_npz(out_path, start_idx=resume_idx, play=False)
        viewer.set_pretty_visuals()

        FOCUS_MODE = True
        FOCUSED_VIEWER = viewer
        set_active(viewer)

        if reuse_live_robot and grip_width_seed is not None:
            try:
                restore_gripper_width_binary(robot_replan, grip_width_seed)
                MIRROR_GRIP_PREV = grip_width_seed
                MIRROR_REPLAN_GRIP_SEED = None
            except Exception as e:
                print(f"[REPLAN] Gripper sync warning: {e}")

    except Exception as e:
        MIRROR_REPLAN_GRIP_SEED = None
        print(f"[REPLAN] Failed: {e}")
    finally:
        if MAIN_WINDOW is not None:
            glfw.make_context_current(MAIN_WINDOW)
            rebuild_main_context()
            glfw.show_window(MAIN_WINDOW)

            if MOUSE is not None:
                MOUSE.left_down = False
                MOUSE.right_down = False
                MOUSE.middle_down = False

            glfw.poll_events()

        
        GLOBAL_PAUSE = old_global_pause
        viewer.is_paused = True
        REPLAN_IN_PROGRESS = False

def key_callback(window, key, scancode, action, mods):
    global GLOBAL_PAUSE, FOCUS_MODE, FOCUSED_VIEWER

    if action not in (glfw.PRESS, glfw.REPEAT):
        return

    viewer = ACTIVE_VIEWER
    shift = bool(mods & glfw.MOD_SHIFT)

    if key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)
        return

    if viewer is None:
        return

    if key == glfw.KEY_P:
        global PENDING_REPLAN
        if action == glfw.PRESS and not REPLAN_IN_PROGRESS:
            PENDING_REPLAN = True
        return

    if key == glfw.KEY_Z:
        if not FOCUS_MODE:
            FOCUS_MODE = True
            FOCUSED_VIEWER = viewer
            set_active(viewer)

            for v in ALL_VIEWERS:
                if v is viewer:
                    v.set_pretty_visuals()
                    v.restart_clock_from_current_frame()
                else:
                    v.set_fast_visuals()
        else:
            if MIRROR_ROBOT:
                disable_mirror()

            if FOCUSED_VIEWER is not None:
                FOCUSED_VIEWER.set_fast_visuals()
            FOCUS_MODE = False
            FOCUSED_VIEWER = None
        return

    if key == glfw.KEY_M:
        if action == glfw.PRESS:
            if not FOCUS_MODE or viewer is not FOCUSED_VIEWER:
                print("[MIRROR] M only works in focus mode on the focused viewer.")
            else:
                if MIRROR_ROBOT:
                    disable_mirror()
                else:
                    enable_mirror()
        return


    if key == glfw.KEY_SPACE:
        if shift:
            GLOBAL_PAUSE = not GLOBAL_PAUSE
            if GLOBAL_PAUSE:
                for v in ALL_VIEWERS:
                    if not v.closed:
                        v.is_paused = True
            else:
                for v in ALL_VIEWERS:
                    if not v.closed:
                        v.is_paused = False
                        v.restart_clock_from_current_frame()
        else:
            viewer.toggle_pause()
        return

    if key == glfw.KEY_R:
        if shift:
            GLOBAL_PAUSE = False
            for v in ALL_VIEWERS:
                if not v.closed:
                    v.reset_to_start(play=True)
        else:
            GLOBAL_PAUSE = False
            viewer.reset_to_start(play=True)
        return

    if key == glfw.KEY_RIGHT:
        if shift:
            for v in ALL_VIEWERS:
                if not v.closed and (GLOBAL_PAUSE or v.is_paused):
                    v.step_frame(+1)
        else:
            if GLOBAL_PAUSE or viewer.is_paused:
                viewer.step_frame(+1)
        return

    if key == glfw.KEY_LEFT:
        if shift:
            for v in ALL_VIEWERS:
                if not v.closed and (GLOBAL_PAUSE or v.is_paused):
                    v.step_frame(-1)
        else:
            if GLOBAL_PAUSE or viewer.is_paused:
                viewer.step_frame(-1)
        return


def mouse_button_callback(window, button, action, mods):
    if MOUSE is None:
        return

    x, y = glfw.get_cursor_pos(window)
    win_w, win_h = glfw.get_framebuffer_size(window)

    if FOCUS_MODE and FOCUSED_VIEWER is not None:
        viewer = FOCUSED_VIEWER
    else:
        viewer = pick_viewer_at_cursor(x, y, win_w, win_h)

    if viewer is not None:
        set_active(viewer)

    if button == glfw.MOUSE_BUTTON_LEFT:
        MOUSE.left_down = (action == glfw.PRESS)
    elif button == glfw.MOUSE_BUTTON_RIGHT:
        MOUSE.right_down = (action == glfw.PRESS)
    elif button == glfw.MOUSE_BUTTON_MIDDLE:
        MOUSE.middle_down = (action == glfw.PRESS)

    MOUSE.last_x = x
    MOUSE.last_y = y


def cursor_pos_callback(window, xpos, ypos):
    if MOUSE is None or ACTIVE_VIEWER is None:
        return

    dx = xpos - MOUSE.last_x
    dy = ypos - MOUSE.last_y
    MOUSE.last_x = xpos
    MOUSE.last_y = ypos

    if not (MOUSE.left_down or MOUSE.right_down or MOUSE.middle_down):
        return

    shift = (
        glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
        or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    )

    _, h = glfw.get_window_size(window)
    if h <= 0:
        return

    if MOUSE.right_down:
        action = mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
    elif MOUSE.left_down:
        action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mujoco.mjtMouse.mjMOUSE_ZOOM

    mujoco.mjv_moveCamera(
        ACTIVE_VIEWER.model,
        action,
        dx / max(1, h),
        dy / max(1, h),
        ACTIVE_VIEWER.scn,
        ACTIVE_VIEWER.cam,
    )


def scroll_callback(window, xoffset, yoffset):
    if ACTIVE_VIEWER is None:
        return

    mujoco.mjv_moveCamera(
        ACTIVE_VIEWER.model,
        mujoco.mjtMouse.mjMOUSE_ZOOM,
        0.0,
        -0.05 * yoffset,
        ACTIVE_VIEWER.scn,
        ACTIVE_VIEWER.cam,
    )


def window_close_callback(window):
    glfw.set_window_should_close(window, True)


def create_main_window(title="MuJoCo Multi Trajectory Viewer", width=1600, height=900):
    global MAIN_WINDOW, MAIN_CTX, MOUSE

    glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
    MAIN_WINDOW = glfw.create_window(width, height, title, None, None)
    if not MAIN_WINDOW:
        raise RuntimeError("Failed to create main GLFW window")

    glfw.make_context_current(MAIN_WINDOW)
    glfw.swap_interval(0)

    MAIN_CTX = mujoco.MjrContext(ALL_VIEWERS[0].model, mujoco.mjtFontScale.mjFONTSCALE_100)
    MOUSE = MouseState()

    glfw.set_key_callback(MAIN_WINDOW, key_callback)
    glfw.set_cursor_pos_callback(MAIN_WINDOW, cursor_pos_callback)
    glfw.set_mouse_button_callback(MAIN_WINDOW, mouse_button_callback)
    glfw.set_scroll_callback(MAIN_WINDOW, scroll_callback)
    glfw.set_window_close_callback(MAIN_WINDOW, window_close_callback)


def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python single_window_multi_npz_viewer.py path/to/model.xml traj1.npz [traj2.npz ... traj17.npz]")
        sys.exit(1)

    xml_path = sys.argv[1]
    npz_paths = sys.argv[2:]

    if len(npz_paths) > MAX_WINDOWS:
        raise RuntimeError(f"At most {MAX_WINDOWS} trajectories are supported")

    for p in npz_paths:
        if not Path(p).exists():
            raise FileNotFoundError(p)

    if not Path(xml_path).exists():
        raise FileNotFoundError(xml_path)

    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    try:
        for i, npz_path in enumerate(npz_paths):
            title = f"[{i}] {Path(npz_path).name}"
            viewer = TrajectoryPlayer(xml_path, npz_path, title=title)
            ALL_VIEWERS.append(viewer)

        if not ALL_VIEWERS:
            raise RuntimeError("No trajectories loaded")

        set_active(ALL_VIEWERS[0])
        create_main_window()

        while not glfw.window_should_close(MAIN_WINDOW):
            glfw.poll_events()
            

            global PENDING_REPLAN

            if PENDING_REPLAN and not REPLAN_IN_PROGRESS:
                PENDING_REPLAN = False
                replan_active_viewer()

            if FOCUS_MODE and FOCUSED_VIEWER is not None:
                if not FOCUSED_VIEWER.closed:
                    FOCUSED_VIEWER.update()
            else:
                for v in ALL_VIEWERS:
                    if not v.closed:
                        v.update()

            if FOCUS_MODE and FOCUSED_VIEWER is not None and MIRROR_ROBOT:
                mirror_focused_viewer_once()

            glfw.make_context_current(MAIN_WINDOW)
            win_w, win_h = glfw.get_framebuffer_size(MAIN_WINDOW)

            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, MAIN_CTX)

            full_rect = mujoco.MjrRect(0, 0, win_w, win_h)
            mujoco.mjr_rectangle(full_rect, 0.08, 0.08, 0.08, 1.0)

            if FOCUS_MODE and FOCUSED_VIEWER is not None:
                FOCUSED_VIEWER.viewport = full_rect
                FOCUSED_VIEWER.render(MAIN_CTX)
            else:
                viewports = compute_viewports(win_w, win_h, len(ALL_VIEWERS), pad=8)
                for viewer, rect in zip(ALL_VIEWERS, viewports):
                    viewer.viewport = rect
                    viewer.render(MAIN_CTX)

            glfw.swap_buffers(MAIN_WINDOW)
            time.sleep(0.001)

    finally:
        if MIRROR_HANDLE is not None:
            disable_mirror()
        if MAIN_CTX is not None:
            try:
                MAIN_CTX.free()
            except Exception:
                pass
        if MAIN_WINDOW is not None:
            try:
                glfw.destroy_window(MAIN_WINDOW)
            except Exception:
                pass
        glfw.terminate()


if __name__ == "__main__":
    main()
