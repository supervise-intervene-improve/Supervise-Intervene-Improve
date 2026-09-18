import os
import time
import numpy as np
from robot.record_robot_adapter import RecordRobotAdapter


def _truthy(value):
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def clamp_step(q_target, q_current, max_dq, dt):
    if max_dq <= 0.0:
        return q_target.copy()
    dq = (q_target - q_current) / dt
    dq = np.clip(dq, -max_dq, max_dq)
    return q_current + dq * dt


class MirrorController:
    def __init__(
        self,
        robot_adapter: RecordRobotAdapter = None,
        view_hz=60.0,
    ):
        self.robot = robot_adapter
        self.view_hz = view_hz

        self.enabled = False
        self.q_prev = None
        self.last_grip_width = None
        self._ramp_notice_shown = False
        self._last_debug_t = 0.0
        self.q_target_filtered = None

    def set_robot(self, robot_adapter):
        self.robot = robot_adapter

    def is_connected(self):
        return self.enabled and self.robot is not None

    def current_robot_key(self):
        if self.robot is None:
            return None
        return getattr(self.robot, "robot_key", None)

    def enable(self, grip_width_seed=None, initial_q_target=None):
        if self.robot is None:
            print("[MIRROR] No robot adapter configured.")
            return False

        if self.enabled:
            print("[MIRROR] Already enabled.")
            return True

        try:
            self.robot.connect(already_connected=False)
            self.q_prev = np.asarray(self.robot.get_joint_positions(), dtype=np.float64)
            self.q_target_filtered = self.q_prev.copy()

        except Exception as e:
            print(f"[MIRROR] Failed to enable mirror: {e}")
            self.enabled = False
            self.q_prev = None
            self.last_grip_width = None
            self.q_target_filtered = None
            return False

        if grip_width_seed is not None:
            try:
                self.robot.restore_gripper_width(grip_width_seed)
                self.last_grip_width = float(grip_width_seed)
            except Exception as e:
                print(f"[MIRROR] Gripper sync warning: {e}")

        self._pre_align_to_target(initial_q_target)

        self.enabled = True
        self._ramp_notice_shown = False
        self._last_debug_t = 0.0
        print("[MIRROR] Enabled.")
        return True

    def disable(self, close_connection=True):
        if self.robot is not None and self.enabled and close_connection:
            try:
                self.robot.close()
            except Exception as e:
                print(f"[MIRROR] Close warning: {e}")

        self.enabled = False
        self.q_prev = None
        self.last_grip_width = None
        self._ramp_notice_shown = False
        self._last_debug_t = 0.0
        self.q_target_filtered = None
        print("[MIRROR] Disabled.")

    def detach_for_reuse(self):
        if not self.enabled:
            return None

        robot = self.robot
        self.enabled = False
        self.q_prev = None
        self.last_grip_width = None
        self._ramp_notice_shown = False
        self._last_debug_t = 0.0
        self.q_target_filtered = None
        print("[MIRROR] Detached live robot connection for reuse.")
        return robot

    def attach_reused_robot(self, robot_adapter, grip_width_seed=None, force_reconnect=True):
        self.robot = robot_adapter

        if force_reconnect:
            self.robot.connect(already_connected=False)
        else:
            self.robot.connect(already_connected=True)

        self.q_prev = np.asarray(self.robot.get_joint_positions(), dtype=np.float64)
        self.q_target_filtered = self.q_prev.copy()
        # self.q_prev = self.robot.robot.robot_arm.get_state().joint_pos.detach().cpu().numpy().astype(np.float64)

        if grip_width_seed is not None:
            try:
                self.robot.restore_gripper_width(grip_width_seed)
                self.last_grip_width = float(grip_width_seed)
            except Exception as e:
                print(f"[MIRROR] Gripper restore warning: {e}")

        self.enabled = True
        self._ramp_notice_shown = False
        self._last_debug_t = 0.0
        print("[MIRROR] Reattached reused robot connection.")

    def toggle(self, grip_width_seed=None, initial_q_target=None):
        if self.enabled:
            self.disable()
            return False
        return self.enable(
            grip_width_seed=grip_width_seed,
            initial_q_target=initial_q_target,
        )

    def get_status_text(self):
        if not self.enabled:
            return ""
        return "MIRROR ON"

    def _get_player_target(self, player):
        qpos_indices = getattr(player, "qpos_indices", None)
        if qpos_indices is not None:
            return np.asarray(player.data.qpos[qpos_indices], dtype=np.float64)
        return np.asarray(player.data.ctrl[:7], dtype=np.float64)

    def _max_joint_speed(self):
        return float(
            os.environ.get(
                "INTERVENE_MIRROR_MAX_DQ_RAD_S",
                os.environ.get("INTERVENE_MIRROR_MAX_DQ", "0.80"),
            )
        )

    def _lowpass_alpha(self):
        return float(os.environ.get("INTERVENE_MIRROR_LOWPASS_ALPHA", "0.25"))

    def _mirror_goto_seconds(self):
        return float(
            os.environ.get(
                "INTERVENE_MIRROR_GOTO_SECONDS",
                os.environ.get("INTERVENE_GOTO_SECONDS", "4.0"),
            )
        )

    def _mirror_goto_speed_factor(self):
        return float(os.environ.get("INTERVENE_MIRROR_GOTO_SPEED_FACTOR", "0.25"))

    def _pre_align_to_target(self, q_target):
        if q_target is None:
            return

        q_target = np.asarray(q_target, dtype=np.float64)
        try:
            q_robot = np.asarray(self.robot.get_joint_positions(), dtype=np.float64)
        except Exception as e:
            print(f"[MIRROR] Pre-align skipped; could not read robot pose: {e}")
            return

        max_error = float(np.max(np.abs(q_target - q_robot)))
        tol = float(os.environ.get("INTERVENE_MIRROR_GOTO_TOL_RAD", "0.03"))
        if max_error <= tol:
            self.q_prev = q_robot.copy()
            self.q_target_filtered = q_robot.copy()
            print(f"[MIRROR] Robot already close to MuJoCo pose (err={max_error:.3f} rad).")
            return

        seconds = self._mirror_goto_seconds()
        speed_factor = self._mirror_goto_speed_factor()
        print(
            f"[MIRROR] Moving slowly to current MuJoCo pose before mirroring "
            f"(err={max_error:.3f} rad, seconds={seconds:.1f}, speed_factor={speed_factor:.2f})."
        )

        try:
            if hasattr(self.robot, "go_to_joint_positions"):
                try:
                    self.robot.go_to_joint_positions(
                        q_target,
                        max_vel_norm_factor=speed_factor,
                        duration_s=seconds,
                    )
                except TypeError:
                    self.robot.go_to_joint_positions(q_target)
            else:
                self.robot.send_joint_positions(q_target)
        except Exception as e:
            print(f"[MIRROR] Pre-align warning: {e}")

        try:
            self.q_prev = np.asarray(self.robot.get_joint_positions(), dtype=np.float64)
            self.q_target_filtered = self.q_prev.copy()
        except Exception:
            self.q_prev = q_target.copy()
            self.q_target_filtered = q_target.copy()

    def _smooth_target(self, q_target):
        alpha = float(np.clip(self._lowpass_alpha(), 0.0, 1.0))
        if alpha >= 1.0:
            self.q_target_filtered = q_target.copy()
            return q_target
        if self.q_target_filtered is None:
            self.q_target_filtered = q_target.copy()
            return q_target
        self.q_target_filtered = (
            (1.0 - alpha) * self.q_target_filtered + alpha * q_target
        )
        return self.q_target_filtered.copy()

    def _send_gripper_from_player(self, player):
        grip = player.get_gripper_width()
        if grip is not None:
            try:
                self.robot.send_gripper_width(float(grip))
                self.last_grip_width = float(grip)
            except Exception as e:
                print(f"[MIRROR] Gripper warning: {e}")

    def hold_current_player_pose(self, player, hold_seconds=0.4):
        if not self.enabled or self.robot is None:
            return

        q_target = self._get_player_target(player)
        dt = 1.0 / self.view_hz
        n_steps = max(1, int(hold_seconds / dt))

        print("[MIRROR] Holding current replay pose before replanning...")
        for _ in range(n_steps):
            try:
                if hasattr(self.robot, "go_to_joint_positions"):
                    self.robot.go_to_joint_positions(q_target)
                else:
                    self.robot.send_joint_positions(q_target)
                self.q_prev = np.asarray(self.robot.get_joint_positions(), dtype=np.float64)

                self._send_gripper_from_player(player)
            except Exception as e:
                print(f"[MIRROR] Hold failed: {e}")
                self.disable(close_connection=True)
                return
            time.sleep(dt)

    def mirror_from_player(self, player):
        if not self.enabled or self.robot is None:
            return

        if player.model.nu < 7:
            return

        frame_dt = 1.0 / self.view_hz
        q_target_raw = self._get_player_target(player)
        q_target = self._smooth_target(q_target_raw)

        try:
            q_robot = np.asarray(self.robot.get_joint_positions(), dtype=np.float64)
        except Exception as e:
            print(f"[MIRROR] Failed to read robot state: {e}")
            self.disable(close_connection=True)
            return

        max_dq = self._max_joint_speed()
        q_cmd = clamp_step(q_target, q_robot, max_dq=max_dq, dt=frame_dt)
        qd_cmd = None
        if max_dq > 0.0:
            qd_cmd = (q_cmd - q_robot) / frame_dt
        max_error = float(np.max(np.abs(q_target_raw - q_robot)))
        filtered_error = float(np.max(np.abs(q_target - q_robot)))
        if max_dq > 0.0 and max_error > 0.05 and not self._ramp_notice_shown:
            print(
                f"[MIRROR] Ramping to replay pose at max {max_dq:.2f} rad/s "
                f"with low-pass alpha {self._lowpass_alpha():.2f} "
                f"(initial max joint error {max_error:.3f} rad)."
            )
            self._ramp_notice_shown = True
        if _truthy(os.environ.get("INTERVENE_MIRROR_DEBUG")):
            now = time.time()
            if now - self._last_debug_t >= 1.0:
                self._last_debug_t = now
                print(
                    f"[MIRROR] tracking raw_err={max_error:.3f} rad "
                    f"filtered_err={filtered_error:.3f} rad"
                )

        try:
            t0 = time.perf_counter()
            try:
                self.robot.send_joint_positions(q_cmd, qd_cmd=qd_cmd)
            except TypeError:
                self.robot.send_joint_positions(q_cmd)
            send_dt = time.perf_counter() - t0

            if _truthy(os.environ.get("INTERVENE_MIRROR_TIMING")):
                print(f"send_joint_positions took {send_dt * 1000:.3f} ms")
            self.q_prev = q_cmd.copy()

        except Exception as e:
            print(f"[MIRROR] Arm command failed: {e}")
            self.disable(close_connection=True)
            return

        self._send_gripper_from_player(player)
