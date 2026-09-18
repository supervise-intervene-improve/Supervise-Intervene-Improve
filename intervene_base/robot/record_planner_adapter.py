import time
from pathlib import Path

import numpy as np

from robot.record_robot_adapter import RecordRobotAdapter


_record_backend = None
_record_error = None


def _get_record_backend():
    global _record_backend, _record_error
    if _record_backend is not None:
        return _record_backend
    try:
        import record as record_module
    except Exception as exc:
        _record_backend = None
        _record_error = exc
        return None
    _record_backend = record_module
    _record_error = None
    return record_module


class RecordPlannerAdapter:
    def __init__(
        self,
        robot_key=None,
        view_hz: float = 60.0,
        log_hz: float = 60.0,
    ):
        self.robot_key = robot_key if robot_key is not None else "p4"
        self.view_hz = view_hz
        self.log_hz = log_hz

    def run_replan(self, player, robot_adapter=None, already_connected=False) -> str:
        record = _get_record_backend()
        if record is None:
            raise RuntimeError(f"Robot backend unavailable: {_record_error}")

        if robot_adapter is None:
            robot_adapter = RecordRobotAdapter(robot_key=self.robot_key)
            robot_adapter.connect(already_connected=False)
            close_after = True
        else:
            close_after = False

        try:
            cut_idx = int(player.frame_idx)
            qpos_seed = player.data.qpos.copy()
            qvel_seed = player.data.qvel.copy()
            q_seed = np.asarray(player.data.ctrl[:7], dtype=np.float64).copy()

            grip_width_seed = None
            finger_seed = 0.0

            if getattr(player.log, "grip_real", None) is not None:
                grip_width_seed = float(player.log.grip_real[cut_idx])
                finger_seed = float(np.clip(grip_width_seed / 2.0, 0.0, 0.04))
            elif player.model.nu >= 8:
                finger_seed = float(np.clip(player.data.ctrl[7], 0.0, 0.04))
                grip_width_seed = float(np.clip(2.0 * finger_seed, 0.0, 0.08))

            log_path = str(player.npz_path)
            xml_path = str(player.xml_path)

            suffix_path = str(
                Path(log_path).with_name(f"suffix_{int(time.time())}.npz")
            )

            print(
                f"[PLANNER-ADAPTER] Replanning from frame {cut_idx} "
                f"of {Path(log_path).name}"
            )

            record.hold_then_replan_record(
                robot=robot_adapter.robot,
                q_hold=q_seed,
                finger_hold=finger_seed,
                grip_width_hold=grip_width_seed,
                save_path=Path(suffix_path),
                mujoco_xml_path=xml_path,
                view_hz=self.view_hz,
                log_hz=self.log_hz,
                alpha=record.ALPHA,
                qpos_seed=qpos_seed,
                qvel_seed=qvel_seed,
                already_connected=already_connected,
            )

            return suffix_path
        finally:
            if close_after:
                robot_adapter.close()
