import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import mujoco


@dataclass
class TrajectoryRecorder:
    save_path: Path
    metadata: object
    log_hz: float
    view_hz: float
    camera_names: list[str]
    save_rgb: bool = True
    save_depth: bool = True

    _stride: int = field(init=False)
    _k: int = field(default=0, init=False)

    wall_t: list = field(default_factory=list)
    sim_t: list = field(default_factory=list)
    q_real: list = field(default_factory=list)
    dq_real: list = field(default_factory=list)
    grip_real: list = field(default_factory=list)
    qpos_sim: list = field(default_factory=list)
    qvel_sim: list = field(default_factory=list)
    ctrl_sim: list = field(default_factory=list)

    rgb_frames: dict = field(default_factory=dict)
    depth_frames: dict = field(default_factory=dict)

    def __post_init__(self):
        self._stride = max(1, int(round(self.view_hz / self.log_hz)))
        self.rgb_frames = {cam: [] for cam in self.camera_names}
        self.depth_frames = {cam: [] for cam in self.camera_names}

    def should_log_this_step(self) -> bool:
        self._k += 1
        return (self._k % self._stride) == 0

    def record_state(
        self,
        now_wall: float,
        data: mujoco.MjData,
        q_real: np.ndarray,
        dq_real: Optional[np.ndarray],
        gripper_width: float,
    ):
        self.wall_t.append(float(now_wall))
        self.sim_t.append(float(data.time))
        self.q_real.append(np.asarray(q_real, dtype=np.float64).copy())

        if dq_real is None:
            self.dq_real.append(np.full_like(q_real, np.nan, dtype=np.float64))
        else:
            self.dq_real.append(np.asarray(dq_real, dtype=np.float64).copy())

        self.grip_real.append(float(gripper_width))
        self.qpos_sim.append(np.asarray(data.qpos, dtype=np.float64).copy())
        self.qvel_sim.append(np.asarray(data.qvel, dtype=np.float64).copy())
        self.ctrl_sim.append(np.asarray(data.ctrl, dtype=np.float64).copy())

    def record_images(self, rgb_by_cam=None, depth_by_cam=None):
        if self.save_rgb and rgb_by_cam is not None:
            for cam, img in rgb_by_cam.items():
                self.rgb_frames[cam].append(img.copy())

        if self.save_depth and depth_by_cam is not None:
            for cam, img in depth_by_cam.items():
                self.depth_frames[cam].append(img.copy())

    def num_frames(self) -> int:
        return len(self.wall_t)

    def save(self):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "wall_t": np.array(self.wall_t, dtype=np.float64),
            "sim_t": np.array(self.sim_t, dtype=np.float64),
            "q_real": np.array(self.q_real, dtype=np.float64),
            "dq_real": np.array(self.dq_real, dtype=np.float64),
            "grip_real": np.array(self.grip_real, dtype=np.float64),
            "qpos_sim": np.array(self.qpos_sim, dtype=np.float64),
            "qvel_sim": np.array(self.qvel_sim, dtype=np.float64),
            "ctrl_sim": np.array(self.ctrl_sim, dtype=np.float64),
        }

        if self.save_rgb:
            for cam in self.camera_names:
                save_dict[f"rgb_{cam}"] = np.array(self.rgb_frames[cam], dtype=np.uint8)

        if self.save_depth:
            for cam in self.camera_names:
                save_dict[f"depth_{cam}"] = np.array(self.depth_frames[cam], dtype=np.float32)

        np.savez_compressed(self.save_path, **save_dict)

        meta_path = self.save_path.with_suffix(".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.metadata), f, indent=2)

        print(f"[INFO] Saved: {self.save_path}")
        print(f"[INFO] Saved: {meta_path}")