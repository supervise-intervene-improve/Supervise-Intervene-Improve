import time
from pathlib import Path

import mujoco
import numpy as np


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def gripper_ctrl_to_width(model: mujoco.MjModel, ctrl_value: float) -> float:
    ctrl_value = float(ctrl_value)
    if model.nu < 8:
        return None

    ctrlrange = model.actuator_ctrlrange[7]
    ctrl_hi = float(ctrlrange[1])
    if ctrl_hi > 0.041:
        return float(np.clip(ctrl_value, float(ctrlrange[0]), ctrl_hi))

    return float(np.clip(2.0 * ctrl_value, 0.0, 0.08))


def gripper_width_to_ctrl(model: mujoco.MjModel, width: float) -> float | None:
    """Inverse of gripper_ctrl_to_width for normal and soft gripper models."""
    if model.nu < 8:
        return None
    width = float(width)
    ctrlrange = model.actuator_ctrlrange[7]
    ctrl_lo = float(ctrlrange[0])
    ctrl_hi = float(ctrlrange[1])
    if ctrl_hi > 0.041:
        return float(np.clip(width, ctrl_lo, ctrl_hi))
    return float(np.clip(0.5 * width, 0.0, 0.04))


class TrajectoryLog:
    def __init__(self, npz_path: str):
        self.npz_path = str(npz_path)
        self.data = np.load(self.npz_path)

        if "sim_t" not in self.data.files:
            raise RuntimeError(f"{self.npz_path} is missing required key 'sim_t'")

        self.sim_t = np.asarray(self.data["sim_t"], dtype=np.float64)
        if len(self.sim_t) == 0:
            raise RuntimeError(f"{self.npz_path} contains zero frames")

        self.wall_t = self._get_optional_1d("wall_t")
        self.q_real = self._get_optional_2d("q_real")
        self.dq_real = self._get_optional_2d("dq_real")
        self.grip_real = self._get_optional_1d("grip_real")

        self.qpos_sim = self._get_optional_2d("qpos_sim")
        self.qvel_sim = self._get_optional_2d("qvel_sim")
        self.ctrl_sim = self._get_optional_2d("ctrl_sim")

        self.nframes = len(self.sim_t)

    def _get_optional_1d(self, key: str):
        return np.asarray(self.data[key], dtype=np.float64) if key in self.data.files else None

    def _get_optional_2d(self, key: str):
        return np.asarray(self.data[key], dtype=np.float64) if key in self.data.files else None

    def _shape_or_none(self, arr):
        return None if arr is None else arr.shape

    def print_summary(self, model: mujoco.MjModel):
        print(
            "[INFO] Loaded trajectory:\n"
            f"  file      : {self.npz_path}\n"
            f"  frames    : {self.nframes}\n"
            f"  model     : nq={model.nq}, nv={model.nv}, nu={model.nu}\n"
            f"  wall_t    : {self._shape_or_none(self.wall_t)}\n"
            f"  sim_t     : {self.sim_t.shape}\n"
            f"  q_real    : {self._shape_or_none(self.q_real)}\n"
            f"  dq_real   : {self._shape_or_none(self.dq_real)}\n"
            f"  grip_real : {self._shape_or_none(self.grip_real)}\n"
            f"  qpos_sim  : {self._shape_or_none(self.qpos_sim)}\n"
            f"  qvel_sim  : {self._shape_or_none(self.qvel_sim)}\n"
            f"  ctrl_sim  : {self._shape_or_none(self.ctrl_sim)}"
        )

    def validate_against_model(self, model: mujoco.MjModel):
        T = self.nframes
        self.print_summary(model)

        def check_time_axis(name, arr):
            if arr is not None and len(arr) != T:
                raise RuntimeError(
                    f"{name} has length {len(arr)} but sim_t has length {T}"
                )

        check_time_axis("wall_t", self.wall_t)
        check_time_axis("grip_real", self.grip_real)
        check_time_axis("q_real", self.q_real)
        check_time_axis("dq_real", self.dq_real)
        check_time_axis("qpos_sim", self.qpos_sim)
        check_time_axis("qvel_sim", self.qvel_sim)
        check_time_axis("ctrl_sim", self.ctrl_sim)

        if self.qpos_sim is not None and self.qpos_sim.shape[1] != model.nq:
            raise RuntimeError(
                f"qpos_sim dim {self.qpos_sim.shape[1]} != model.nq {model.nq}"
            )

        if self.qvel_sim is not None and self.qvel_sim.shape[1] != model.nv:
            raise RuntimeError(
                f"qvel_sim dim {self.qvel_sim.shape[1]} != model.nv {model.nv}"
            )

        if self.ctrl_sim is not None and self.ctrl_sim.shape[1] != model.nu:
            raise RuntimeError(
                f"ctrl_sim dim {self.ctrl_sim.shape[1]} != model.nu {model.nu}"
            )


class TrajectoryPlayer:
    def __init__(self, xml_path: str, npz_path: str):
        self.xml_path = str(xml_path)
        self.npz_path = str(npz_path)

        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        self.log = None
        self.frame_idx = 0
        self.is_paused = False
        self.play_start_wall = None
        self.play_start_sim = None

        self.reload(npz_path, play=True)

    @property
    def nframes(self):
        return self.log.nframes

    @property
    def sim_t(self):
        return self.log.sim_t

    def reload(self, npz_path: str, play: bool = True):
        self.npz_path = str(npz_path)
        self.log = TrajectoryLog(self.npz_path)
        self.log.validate_against_model(self.model)
        self.reset_to_start(play=play)

    def reset_to_start(self, play=True):
        self.frame_idx = 0
        self._apply_frame(self.frame_idx)
        self.play_start_wall = time.perf_counter()
        self.play_start_sim = float(self.sim_t[self.frame_idx])
        self.is_paused = not play

    def jump_to_last(self, play=False):
        self.frame_idx = self.nframes - 1
        self._apply_frame(self.frame_idx)
        self.play_start_wall = time.perf_counter()
        self.play_start_sim = float(self.sim_t[self.frame_idx])
        self.is_paused = not play

    def restart_clock_from_current_frame(self):
        self.play_start_wall = time.perf_counter()
        self.play_start_sim = float(self.sim_t[self.frame_idx])

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        if not self.is_paused:
            self.restart_clock_from_current_frame()

    def step_frame(self, delta: int):
        new_idx = clamp(self.frame_idx + delta, 0, self.nframes - 1)
        self._apply_frame(new_idx)
        self.restart_clock_from_current_frame()

    def _apply_frame(self, idx: int):
        idx = int(clamp(idx, 0, self.nframes - 1))
        self.frame_idx = idx

        if self.log.qpos_sim is not None:
            self.data.qpos[:] = self.log.qpos_sim[idx]

        if self.log.qvel_sim is not None:
            self.data.qvel[:] = self.log.qvel_sim[idx]
        else:
            self.data.qvel[:] = 0.0

        if self.log.ctrl_sim is not None and self.model.nu > 0:
            self.data.ctrl[:] = self.log.ctrl_sim[idx]

        self.data.time = float(self.sim_t[idx])
        mujoco.mj_forward(self.model, self.data)

    def update(self):
        if self.is_paused:
            return

        target_sim = self.play_start_sim + (time.perf_counter() - self.play_start_wall)

        idx = self.frame_idx
        while idx + 1 < self.nframes and self.sim_t[idx + 1] <= target_sim:
            idx += 1

        if idx != self.frame_idx:
            self._apply_frame(idx)

        if self.frame_idx >= self.nframes - 1:
            self.is_paused = True

    def get_gripper_width(self):
        if self.log.grip_real is not None:
            return float(self.log.grip_real[self.frame_idx])

        if self.model.nu >= 8:
            return gripper_ctrl_to_width(self.model, self.data.ctrl[7])

        return None

    def get_arm_ctrl(self):
        if self.model.nu <= 0:
            return None
        n = min(7, self.model.nu)
        return np.asarray(self.data.ctrl[:n], dtype=np.float64).copy()

    def get_frame_summary_text(self):
        lines = [
            "[FRAME SUMMARY]",
            f"  file      : {Path(self.npz_path).name}",
            f"  frame     : {self.frame_idx + 1}/{self.nframes}",
            f"  sim_t     : {self.sim_t[self.frame_idx]:.6f}",
        ]

        if self.log.wall_t is not None:
            lines.append(f"  wall_t    : {self.log.wall_t[self.frame_idx]:.6f}")

        if self.log.q_real is not None:
            lines.append(
                f"  q_real    : {np.array2string(self.log.q_real[self.frame_idx], precision=4, suppress_small=True)}"
            )

        if self.log.dq_real is not None:
            lines.append(
                f"  dq_real   : {np.array2string(self.log.dq_real[self.frame_idx], precision=4, suppress_small=True)}"
            )

        grip = self.get_gripper_width()
        if grip is not None:
            lines.append(f"  grip      : {grip:.6f}")

        arm_ctrl = self.get_arm_ctrl()
        if arm_ctrl is not None:
            lines.append(
                f"  arm_ctrl  : {np.array2string(arm_ctrl, precision=4, suppress_small=True)}"
            )

        return "\n".join(lines)

    def print_frame_summary(self):
        print(self.get_frame_summary_text())

    def get_overlay_text(self, extra_status: str = ""):
        state = "PAUSED" if self.is_paused else "PLAYING"
        title = Path(self.npz_path).name
        grip = self.get_gripper_width()
        grip_txt = f"{grip:.4f}" if grip is not None else "n/a"

        if extra_status:
            state = f"{state} | {extra_status}"

        info = (
            f"{title}\n"
            f"frame {self.frame_idx + 1}/{self.nframes}\n"
            f"sim_t = {self.sim_t[self.frame_idx]:.3f}\n"
            f"grip  = {grip_txt}"
        )
        return info, state
