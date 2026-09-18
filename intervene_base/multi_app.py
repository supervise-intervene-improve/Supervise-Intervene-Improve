import math
import os
import time
from pathlib import Path

import glfw
import mujoco
import numpy as np

from playback.player import TrajectoryPlayer
from rendering.viewer import SceneViewer
from data_io.stitch import stitch_npz, make_replanned_output_path

try:
    from robot.live_replan_session import LiveReplanSession, TrajectoryRecorder
    from robot.mirror_controller import MirrorController
    from robot.record_robot_adapter import RecordRobotAdapter
    from robot.light_polymetis_adapter import LightPolymetisRobotAdapter
    ROBOT_IMPORT_ERROR = None
except Exception as e:
    LiveReplanSession = None
    TrajectoryRecorder = None
    MirrorController = None
    RecordRobotAdapter = None
    LightPolymetisRobotAdapter = None
    ROBOT_IMPORT_ERROR = e


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def compute_grid(n: int):
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def compute_viewports(win_w: int, win_h: int, n: int, pad: int = 8):
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


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_list(name: str, default: str) -> list[str]:
    value = os.environ.get(name, default)
    return [part.strip() for part in value.replace(",", " ").split() if part.strip()]


class MouseState:
    def __init__(self):
        self.left_down = False
        self.right_down = False
        self.middle_down = False
        self.last_x = 0.0
        self.last_y = 0.0


class DisabledMirrorController:
    enabled = False
    robot = None

    def toggle(self, grip_width_seed=None, initial_q_target=None):
        return False

    def mirror_from_player(self, player):
        return

    def disable(self):
        return


class MultiApp:
    def __init__(
        self,
        xml_path: str,
        npz_paths: list[str],
        width=1600,
        height=900,
        robot_key="p1",
        player_mode: str = "replay",
        policy_configs: list[object] | None = None,
        view_mode: str = "free",
    ):
        self.xml_path = xml_path
        self.npz_paths = list(npz_paths)
        self.width = width
        self.height = height
        self.robot_key = robot_key
        self.player_mode = player_mode
        self.policy_configs = policy_configs
        self.view_mode = view_mode

        self.window = None
        self.ctx = None
        self.mouse = MouseState()

        self.players = []
        self.viewers = []

        self.active_idx = 0
        self.focus_mode = False
        self.global_pause = False
        self.policy_update_idx = 0

        # multi-mode replan state
        self.mode = "replay"          # "replay" | "replan"
        self.replan_session = None
        self.replan_player_idx = None
        self.replan_cut_idx = None
        self.replan_original_path = None
        self.pre_replan_focus_mode = False
        self.replan_should_resume_mirror = False
        self.episode_recorders = []
        self.episode_saved = []
        self.episode_save_paths = []
        self.last_recorded_player_frame_idxs = []
        self.record_camera_names = []
        self.record_rgb_width = 224
        self.record_rgb_height = 224
        self.record_save_rgb = True
        self.record_save_depth = False

        if MirrorController is None:
            self.robot_features_available = False
            self.mirror_controller = DisabledMirrorController()
            print(f"[INFO] MultiApp robot features disabled: {ROBOT_IMPORT_ERROR}")
        else:
            self.robot_features_available = False
            self.mirror_controller = MirrorController(
                robot_adapter=None,
                view_hz=60.0,
            )

            try:
                robot_backend = os.environ.get("INTERVENE_ROBOT_BACKEND", "record").strip().lower()
                control_mode = os.environ.get(
                    "INTERVENE_ROBOT_CONTROL_MODE",
                    "HYBRID_JOINT_IMPEDANCE_CONTROL",
                )
                if robot_backend == "record":
                    robot_adapter = RecordRobotAdapter(
                        robot_key=self.robot_key,
                        control_mode=control_mode,
                    )
                else:
                    robot_adapter = LightPolymetisRobotAdapter(
                        robot_key=self.robot_key,
                        control_mode=control_mode,
                    )

                self.mirror_controller.set_robot(robot_adapter)
                self.robot_features_available = True
                print(f"[INFO] MultiApp robot features enabled for {self.robot_key} via {robot_backend} backend.")
            except Exception as e:
                print(f"[INFO] MultiApp robot features disabled: {e}")

    @property
    def active_player(self):
        if not self.players:
            return None
        self.active_idx = clamp(self.active_idx, 0, len(self.players) - 1)
        return self.players[self.active_idx]

    @property
    def active_viewer(self):
        if not self.viewers:
            return None
        self.active_idx = clamp(self.active_idx, 0, len(self.viewers) - 1)
        return self.viewers[self.active_idx]

    def init(self):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        self.window = glfw.create_window(
            self.width, self.height, "MuJoCo Multi Trajectory Viewer", None, None
        )
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        if self.player_mode == "policy":
            from playback.policy_player import PolicyPlayer

            if self.policy_configs is None:
                raise ValueError("policy_configs is required when player_mode='policy'")
            if len(self.policy_configs) != len(self.npz_paths):
                raise ValueError("policy_configs must match npz_paths length")
            self.players = []
            for cfg in self.policy_configs:
                self.players.append(PolicyPlayer(
                    self.xml_path,
                    cfg,
                    context_current_fn=lambda: glfw.make_context_current(self.window),
                ))
                glfw.make_context_current(self.window)
        elif self.player_mode == "replay":
            self.players = [TrajectoryPlayer(self.xml_path, p) for p in self.npz_paths]
        else:
            raise ValueError(f"Unknown player_mode: {self.player_mode}")

        if not self.players:
            raise RuntimeError("No trajectories loaded.")

        glfw.make_context_current(self.window)
        self.viewers = [SceneViewer(p.model, view_mode=self.view_mode) for p in self.players]

        glfw.make_context_current(self.window)
        self.ctx = mujoco.MjrContext(
            self.players[0].model,
            mujoco.mjtFontScale.mjFONTSCALE_100,
        )

        glfw.set_window_user_pointer(self.window, self)
        glfw.set_key_callback(self.window, MultiApp._key_callback)
        glfw.set_cursor_pos_callback(self.window, MultiApp._cursor_pos_callback)
        glfw.set_mouse_button_callback(self.window, MultiApp._mouse_button_callback)
        glfw.set_scroll_callback(self.window, MultiApp._scroll_callback)
        self._ensure_episode_recorders()
        self._record_player_frames(force=True)

    def _episode_output_dir(self):
        return Path(
            os.environ.get(
                "INTERVENE_EPISODE_DIR",
                os.environ.get("INTERVENE_OUTPUT_DIR", "INTERVENTION_DATA"),
            )
        )

    def _ensure_episode_recorders(self):
        if (
            self.player_mode != "policy"
            or TrajectoryRecorder is None
            or self.episode_recorders
            or not _env_bool("INTERVENE_MULTI_RECORD", False)
        ):
            return self.episode_recorders

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        save_dir = self._episode_output_dir()
        self.record_camera_names = _env_list("INTERVENE_RECORD_CAMERAS", "right,left,wrist")
        self.record_rgb_width = int(os.environ.get("INTERVENE_RECORD_RGB_WIDTH", "224"))
        self.record_rgb_height = int(os.environ.get("INTERVENE_RECORD_RGB_HEIGHT", "224"))
        self.record_save_rgb = _env_bool("INTERVENE_RECORD_RGB", True)
        self.record_save_depth = _env_bool("INTERVENE_RECORD_DEPTH", False)

        for idx, player in enumerate(self.players):
            cfg = getattr(player, "config", None)
            save_path = save_dir / f"{self.player_mode}_multi_tile{idx}_{timestamp}.npz"
            log_hz = float(getattr(cfg, "policy_hz", 60.0))
            recorder = TrajectoryRecorder(
                save_path=save_path,
                lab_id=f"{self.player_mode}_multi_tile{idx}_intervention",
                log_hz=log_hz,
                view_hz=60.0,
                camera_names=self.record_camera_names,
                save_rgb=self.record_save_rgb,
                save_depth=self.record_save_depth,
                rgb_width=self.record_rgb_width,
                rgb_height=self.record_rgb_height,
                metadata={
                    "player_mode": self.player_mode,
                    "tile_index": idx,
                    "mujoco_xml_path": self.xml_path,
                    "reset_npz_path": str(getattr(cfg, "reset_npz", player.npz_path)),
                    "checkpoint": str(getattr(cfg, "checkpoint", "")) if cfg is not None else None,
                    "robot_key": self.robot_key,
                },
            )
            self.episode_recorders.append(recorder)
            self.episode_saved.append(False)
            self.episode_save_paths.append(save_path)
            self.last_recorded_player_frame_idxs.append(None)
            print(f"[INFO] Multi tile {idx} recording will save to: {save_path}")

        return self.episode_recorders

    def _capture_recorder_images(self, player_idx: int, data):
        if (
            self.ctx is None
            or not self.record_camera_names
            or player_idx < 0
            or player_idx >= len(self.viewers)
        ):
            return {}, {}

        glfw.make_context_current(self.window)
        try:
            rgbd = self.viewers[player_idx].render_rgbd_from_cameras(
                data,
                self.ctx,
                self.record_camera_names,
                width=self.record_rgb_width,
                height=self.record_rgb_height,
            )
        finally:
            glfw.make_context_current(self.window)
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.ctx)

        rgb_by_cam = {}
        depth_by_cam = {}
        for cam_name, frame in rgbd.items():
            if self.record_save_rgb:
                rgb_by_cam[cam_name] = frame["rgb"]
            if self.record_save_depth:
                depth_by_cam[cam_name] = frame["depth"]
        return rgb_by_cam, depth_by_cam

    def _record_player_frame(self, idx: int, force=False):
        if idx < 0 or idx >= len(self.episode_recorders) or idx >= len(self.players):
            return

        player = self.players[idx]
        frame_idx = int(getattr(player, "frame_idx", 0))
        if not force and self.last_recorded_player_frame_idxs[idx] == frame_idx:
            return

        q_real = self._get_player_arm_q(player)
        qvel = np.asarray(player.data.qvel, dtype=np.float64)
        dq_real = qvel[:7] if qvel.shape[0] >= 7 else np.full(7, np.nan, dtype=np.float64)
        grip_width = player.get_gripper_width()
        if grip_width is None:
            grip_width = np.nan

        self.episode_recorders[idx].record(
            now_wall=time.time(),
            data=player.data,
            q_real=q_real,
            dq_real=dq_real,
            gripper_width=grip_width,
            intervention=False,
            force=True,
            image_capture_fn=lambda data: self._capture_recorder_images(idx, data),
        )
        self.last_recorded_player_frame_idxs[idx] = frame_idx

    def _record_player_frames(self, force=False):
        if not self.episode_recorders:
            return
        for idx in range(len(self.players)):
            self._record_player_frame(idx, force=force)

    def _save_episode_recordings(self):
        if not self.episode_recorders:
            return []

        saved_paths = []
        for idx, recorder in enumerate(self.episode_recorders):
            if self.episode_saved[idx] or not recorder.has_frames():
                continue
            recorder.save()
            self.episode_saved[idx] = True
            saved_paths.append(str(recorder.save_path))
        return saved_paths

    def run(self):
        self.init()
        try:
            while not glfw.window_should_close(self.window):
                glfw.poll_events()

                win_w, win_h = glfw.get_framebuffer_size(self.window)
                full_rect = mujoco.MjrRect(0, 0, win_w, win_h)

                glfw.make_context_current(self.window)
                mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.ctx)
                mujoco.mjr_rectangle(full_rect, 0.08, 0.08, 0.08, 1.0)

                if self.mode == "replan" and self.replan_session is not None:
                    self.replan_session.update()
                    if self.focus_mode:
                        self._render_focused(full_rect)
                        overlay_rect = full_rect
                    else:
                        self._render_grid(win_w, win_h)
                        rects = compute_viewports(win_w, win_h, len(self.players), pad=8)
                        overlay_idx = (
                            self.replan_player_idx
                            if self.replan_player_idx is not None
                            else self.active_idx
                        )
                        overlay_rect = rects[overlay_idx]

                    mujoco.mjr_overlay(
                        mujoco.mjtFont.mjFONT_NORMAL,
                        mujoco.mjtGridPos.mjGRID_TOPLEFT,
                        overlay_rect,
                        f"REPLAN MODE\n tile {self.replan_player_idx}: guide the robot",
                        "Press Enter to resume policy from this state",
                        self.ctx,
                    )
                else:
                    if self.focus_mode:
                        self._render_focused(full_rect)
                    else:
                        self._render_grid(win_w, win_h)

                glfw.swap_buffers(self.window)

                if self.mode != "replan":
                    self._update_players()
                    self._record_player_frames()
                    glfw.make_context_current(self.window)
                    mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.ctx)

                    if self.mirror_controller.enabled:
                        player = self.active_player
                        if player is not None:
                            self.mirror_controller.mirror_from_player(player)
        finally:
            self.close()

    def close(self):
        if self.replan_session is not None:
            if self.episode_recorders:
                try:
                    self.replan_session.finish_segment()
                except Exception:
                    pass
            try:
                self.replan_session.robot_adapter.close()
            except Exception:
                pass

        if self.mirror_controller.enabled:
            self.mirror_controller.disable()

        try:
            saved_paths = self._save_episode_recordings()
            for out_path in saved_paths:
                print(f"[INFO] Saved multi episode recording on close: {out_path}")
        except Exception as e:
            print(f"[WARN] Could not save multi episode recordings on close: {e}")

        for player in self.players:
            if hasattr(player, "close"):
                try:
                    player.close()
                except Exception:
                    pass
        self.players = []

        if self.ctx is not None:
            try:
                self.ctx.free()
            except Exception:
                pass
            self.ctx = None

        if self.window is not None:
            try:
                glfw.destroy_window(self.window)
            except Exception:
                pass
            self.window = None

        glfw.terminate()

    def _update_players(self):
        if not self.players:
            return

        if self.focus_mode:
            p = self.active_player
            if p is not None and not self.global_pause:
                p.update()
            return

        if self.global_pause:
            return

        if self.player_mode == "policy" and _env_bool("INTERVENE_MULTI_STAGGER_POLICY", True):
            n = len(self.players)
            for _ in range(n):
                idx = self.policy_update_idx % n
                self.policy_update_idx = (self.policy_update_idx + 1) % n
                p = self.players[idx]
                if not p.is_paused:
                    p.update()
                    if hasattr(p, "restart_clock_from_current_frame"):
                        p.restart_clock_from_current_frame()
                    return
            return

        for p in self.players:
            p.update()

    def _render_focused(self, viewport):
        player = self.active_player
        viewer = self.active_viewer
        if player is None or viewer is None:
            return

        viewer.render(player, viewport, self.ctx, draw_overlay=False)
        self._draw_focus_overlay(viewport)

    def _render_grid(self, win_w: int, win_h: int):
        rects = compute_viewports(win_w, win_h, len(self.players), pad=8)

        for idx, (player, viewer, rect) in enumerate(zip(self.players, self.viewers, rects)):
            viewer.render(player, rect, self.ctx, draw_overlay=False)
            self._draw_tile_overlay(idx, rect)

    def _draw_tile_overlay(self, idx: int, viewport):
        player = self.players[idx]

        active_mark = "ACTIVE" if idx == self.active_idx else ""
        state = self._player_state_text(player)
        if self.global_pause:
            state = f"{state} | GLOBAL-PAUSE"

        right_text = " ".join(s for s in [state, active_mark] if s)

        top_left = (
            f"{idx}: {Path(player.npz_path).name}\n"
            f"frame {player.frame_idx + 1}/{player.nframes}\n"
            f"t = {player.sim_t[player.frame_idx]:.3f}"
        )

        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            viewport,
            top_left,
            right_text,
            self.ctx,
        )

        if idx == self.active_idx:
            help_left = (
                "Space: pause active\n"
                "Shift+Space: pause all\n"
                "Left/Right: step active when paused\n"
                "Shift+Left/Right: step all when paused\n"
                "R: reset active\n"
                "Shift+R: reset all"
            )
            help_right = (
                "Click tile: activate\n"
                "Z: focus active\n"
                "M: mirror active tile\n"
                "P: replan active tile\n"
                "End: jump active to end\n"
                "I: print frame summary\n"
                "Esc: quit"
            )

            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                viewport,
                help_left,
                help_right,
                self.ctx,
            )

    def _draw_focus_overlay(self, viewport):
        player = self.active_player
        if player is None:
            return

        state = self._player_state_text(player)
        if self.global_pause:
            state = f"{state} | GLOBAL-PAUSE"

        top_left = (
            f"{self.active_idx}: {Path(player.npz_path).name}\n"
            f"frame {player.frame_idx + 1}/{player.nframes}\n"
            f"t = {player.sim_t[player.frame_idx]:.3f}"
        )

        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            viewport,
            top_left,
            state,
            self.ctx,
        )

        help_left = (
            "Space: pause/resume active\n"
            "Left/Right: step active when paused\n"
            "R: reset active\n"
            "M: mirror active tile\n"
            "P: replan active tile\n"
            "End: jump active to last frame\n"
            "I: print frame summary"
        )
        help_right = (
            "Z: exit focus\n"
            "Mouse: active camera only\n"
            "Esc: quit"
        )

        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
            viewport,
            help_left,
            help_right,
            self.ctx,
        )

    def _player_state_text(self, player):
        if player.is_paused:
            return "PAUSED"
        if player.__class__.__name__ == "PolicyPlayer":
            return "POLICY"
        return "PLAYING"

    def _pick_active_from_cursor(self, x, y):
        if not self.players:
            return None

        win_w, win_h = glfw.get_framebuffer_size(self.window)
        rects = compute_viewports(win_w, win_h, len(self.players), pad=8)
        y_mj = win_h - y

        for idx, rect in enumerate(rects):
            if (
                rect.left <= x <= rect.left + rect.width
                and rect.bottom <= y_mj <= rect.bottom + rect.height
            ):
                self.active_idx = idx
                return idx
        return None

    def _get_player_arm_q(self, player):
        qpos_indices = getattr(player, "qpos_indices", None)
        if qpos_indices is not None:
            return np.asarray(player.data.qpos[qpos_indices], dtype=np.float64)
        return np.asarray(player.data.ctrl[:7], dtype=np.float64)

    def toggle_mirror_active(self):
        if self.mode != "replay":
            print("[MIRROR] Mirror toggle only allowed in replay mode.")
            return

        player = self.active_player
        if player is None:
            print("[MIRROR] No active player.")
            return

        grip_seed = player.get_gripper_width()
        q_seed = self._get_player_arm_q(player)
        ok = self.mirror_controller.toggle(
            grip_width_seed=grip_seed,
            initial_q_target=q_seed,
        )

        if ok:
            print(f"[MIRROR] Mirror toggled for active tile {self.active_idx}.")
        else:
            if not self.mirror_controller.enabled:
                print("[MIRROR] Mirror remains OFF.")

    def _toggle_focus(self):
        if not self.players or self.mode != "replay":
            return
        self.focus_mode = not self.focus_mode

    def _toggle_pause_active(self):
        p = self.active_player
        if p is not None:
            p.toggle_pause()

    def _toggle_pause_all(self):
        self.global_pause = not self.global_pause

        if self.global_pause:
            for p in self.players:
                p.is_paused = True
        else:
            for p in self.players:
                p.is_paused = False
                p.restart_clock_from_current_frame()

    def _reset_active(self):
        p = self.active_player
        if p is None:
            return
        self.global_pause = False
        p.reset_to_start(play=True)

    def _reset_all(self):
        self.global_pause = False
        for p in self.players:
            p.reset_to_start(play=True)

    def _step_active(self, delta: int):
        p = self.active_player
        if p is not None and (p.is_paused or self.global_pause):
            p.step_frame(delta)

    def _step_all(self, delta: int):
        for p in self.players:
            if p.is_paused or self.global_pause:
                p.step_frame(delta)

    def start_replan_active(self):
        if self.mode != "replay":
            print("[REPLAN] Already in replan mode.")
            return

        player = self.active_player
        if player is None:
            print("[REPLAN] No active player.")
            return

        if not player.is_paused:
            if hasattr(player, "pause_at_current_state"):
                player.pause_at_current_state()
            else:
                player.toggle_pause()
            print("[REPLAN] Active policy paused. Moving real robot to current MuJoCo pose...")

        # if not self.mirror_controller.enabled:
        #     print("[REPLAN] Activate mirroring first with M, then press P.")
        #     return

        if not self.robot_features_available:
            print("[REPLAN] Robot features unavailable.")
            return

        cut_idx = int(player.frame_idx)
        qpos_seed = player.data.qpos.copy()
        qvel_seed = player.data.qvel.copy()
        q_seed = self._get_player_arm_q(player)

        grip_width_seed = player.get_gripper_width()
        finger_seed = 0.0
        if player.model.nu >= 8:
            finger_seed = float(player.data.ctrl[7])

        self.replan_should_resume_mirror = bool(self.mirror_controller.enabled)
        if self.mirror_controller.enabled:
            robot_adapter = self.mirror_controller.detach_for_reuse()
            already_connected = True
        else:
            robot_adapter = self.mirror_controller.robot
            robot_adapter.connect(already_connected=False)
            already_connected = True

        recorder = None
        image_capture_fn = None
        if self.episode_recorders and self.active_idx < len(self.episode_recorders):
            recorder = self.episode_recorders[self.active_idx]
            recorder.reset_stride()
            image_capture_fn = (
                lambda data, idx=self.active_idx: self._capture_recorder_images(idx, data)
            )

        suffix_path = str(
            Path(player.npz_path).with_name(f"suffix_{int(time.time())}.npz")
        )

        self.replan_session = LiveReplanSession(
            robot_adapter=robot_adapter,
            save_path=suffix_path,
            mujoco_lab_id="replan",
            mujoco_xml_path=player.xml_path,
            q_hold=q_seed,
            finger_hold=finger_seed,
            grip_width_hold=grip_width_seed,
            qpos_seed=qpos_seed,
            qvel_seed=qvel_seed,
            view_hz=60.0,
            log_hz=60.0,
            alpha=0.6,
            already_connected=already_connected,
            model=player.model,
            data=player.data,
            recorder=recorder,
            image_capture_fn=image_capture_fn,
        )

        self.replan_player_idx = self.active_idx
        self.replan_cut_idx = cut_idx
        self.replan_original_path = str(player.npz_path)
        self.pre_replan_focus_mode = self.focus_mode

        self.mode = "replan"

        print(f"[REPLAN] Entered replan mode for tile {self.replan_player_idx}.")

    def finish_replan(self):
        if self.mode != "replan" or self.replan_session is None:
            return

        robot_adapter = self.replan_session.robot_adapter
        idx = self.replan_player_idx
        player = self.players[idx]
        should_resume_mirror = self.replan_should_resume_mirror

        if self.player_mode == "replay":
            suffix_path = self.replan_session.save_and_finish()
            out_path = make_replanned_output_path(self.replan_original_path)
            stitch_npz(self.replan_original_path, suffix_path, self.replan_cut_idx, out_path)

            try:
                robot_adapter.close()
            except Exception as e:
                print(f"[REPLAN] Robot close warning: {e}")

            player.reload(out_path, play=False)
            message = f"[REPLAN] Reloaded replanned trajectory into tile {idx}: {out_path}"
        else:
            if self.episode_recorders:
                self.replan_session.finish_segment()
            else:
                suffix_path = self.replan_session.save_and_finish()
                print(f"[REPLAN] Saved intervention suffix for tile {idx}: {suffix_path}")

            if should_resume_mirror:
                try:
                    self.mirror_controller.attach_reused_robot(
                        robot_adapter,
                        grip_width_seed=player.get_gripper_width(),
                        force_reconnect=True,
                    )
                except Exception as e:
                    print(f"[REPLAN] Mirror resume warning: {e}")
            elif hasattr(robot_adapter, "switch_control_mode"):
                try:
                    control_mode = os.environ.get(
                        "INTERVENE_ROBOT_CONTROL_MODE",
                        "HYBRID_JOINT_IMPEDANCE_CONTROL",
                    )
                    robot_adapter.switch_control_mode(control_mode)
                except Exception as e:
                    print(f"[REPLAN] Robot hold-mode restore warning: {e}")

            if hasattr(player, "resume_from_current_state"):
                player.resume_from_current_state(reset_policy_queue=True)
            elif getattr(player, "is_paused", False):
                player.toggle_pause()

            if (
                self.last_recorded_player_frame_idxs
                and idx < len(self.last_recorded_player_frame_idxs)
            ):
                self.last_recorded_player_frame_idxs[idx] = None

            message = (
                f"[REPLAN] Intervention finished for tile {idx}. "
                "Policy resumed from current MuJoCo state."
            )

        self.replan_session = None
        self.replan_player_idx = None
        self.replan_cut_idx = None
        self.replan_original_path = None
        self.replan_should_resume_mirror = False
        self.mode = "replay"
        self.focus_mode = self.pre_replan_focus_mode

        print(message)

    @staticmethod
    def _key_callback(window, key, scancode, action, mods):
        app = glfw.get_window_user_pointer(window)
        if app is None or action not in (glfw.PRESS, glfw.REPEAT):
            return

        if app.mode == "replan":
            if key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(window, True)
            elif key in (glfw.KEY_ENTER, glfw.KEY_KP_ENTER) and action == glfw.PRESS:
                app.finish_replan()
            return

        shift = bool(mods & glfw.MOD_SHIFT)

        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
            return

        if key == glfw.KEY_Z and action == glfw.PRESS:
            app._toggle_focus()
            return

        if key == glfw.KEY_M and action == glfw.PRESS:
            app.toggle_mirror_active()
            return

        if key == glfw.KEY_P and action == glfw.PRESS:
            app.start_replan_active()
            return

        if key == glfw.KEY_I and action == glfw.PRESS:
            p = app.active_player
            if p is not None:
                p.print_frame_summary()
            return

        if key == glfw.KEY_SPACE:
            if shift and not app.focus_mode:
                app._toggle_pause_all()
            else:
                app._toggle_pause_active()
            return

        if key == glfw.KEY_R:
            if shift and not app.focus_mode:
                app._reset_all()
            else:
                app._reset_active()
            return

        if key == glfw.KEY_END:
            p = app.active_player
            if p is not None:
                p.jump_to_last(play=False)
            return

        if key == glfw.KEY_RIGHT:
            if shift and not app.focus_mode:
                app._step_all(+1)
            else:
                app._step_active(+1)
            return

        if key == glfw.KEY_LEFT:
            if shift and not app.focus_mode:
                app._step_all(-1)
            else:
                app._step_active(-1)
            return

    @staticmethod
    def _mouse_button_callback(window, button, action, mods):
        app = glfw.get_window_user_pointer(window)
        if app is None:
            return

        x, y = glfw.get_cursor_pos(window)
        app.mouse.last_x = x
        app.mouse.last_y = y

        if app.mode == "replay" and not app.focus_mode:
            app._pick_active_from_cursor(x, y)

        if button == glfw.MOUSE_BUTTON_LEFT:
            app.mouse.left_down = (action == glfw.PRESS)
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            app.mouse.right_down = (action == glfw.PRESS)
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            app.mouse.middle_down = (action == glfw.PRESS)

    @staticmethod
    def _cursor_pos_callback(window, xpos, ypos):
        app = glfw.get_window_user_pointer(window)
        if app is None:
            return

        dx = xpos - app.mouse.last_x
        dy = ypos - app.mouse.last_y
        app.mouse.last_x = xpos
        app.mouse.last_y = ypos

        if not (app.mouse.left_down or app.mouse.right_down or app.mouse.middle_down):
            return

        _, h = glfw.get_window_size(window)
        if h <= 0:
            return

        shift = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )

        if app.mouse.right_down:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif app.mouse.left_down:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM

        viewer = app.active_viewer
        player = app.active_player
        if viewer is None or player is None:
            return

        viewer.move_camera(
            player.model,
            action,
            dx / max(1, h),
            dy / max(1, h),
        )

    @staticmethod
    def _scroll_callback(window, xoffset, yoffset):
        app = glfw.get_window_user_pointer(window)
        if app is None:
            return

        viewer = app.active_viewer
        player = app.active_player
        if viewer is None or player is None:
            return

        viewer.move_camera(
            player.model,
            mujoco.mjtMouse.mjMOUSE_ZOOM,
            0.0,
            -0.05 * yoffset,
        )
