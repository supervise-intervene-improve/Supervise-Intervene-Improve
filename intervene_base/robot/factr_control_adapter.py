import json
import os
import threading
import time
from pathlib import Path

import numpy as np

from factr import (
    FACTRGravityCompensation,
    FactrPDController,
    FactrSafetyError,
    move_factr_to_joint_pose,
)
from factr.validation import (
    validate_calibration_report,
    validate_gravity_report,
    validate_torque_direction_report,
)


DEFAULT_FACTR_KP = np.array([0.75, 3.2, 1.0, 3.5, 0.4, 0.15, 0.15], dtype=np.float64)
DEFAULT_FACTR_KD = np.array([0.01, 0.005, 0.01, 0.01, 0.01, 0.03, 0.01], dtype=np.float64)
DEFAULT_FACTR_DRIVE_TORQUE = np.array(
    [2.2, 2.2, 2.2, 2.2, 1.2, 0.7, 0.7], dtype=np.float64
)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_joint_array(name: str, default: np.ndarray) -> np.ndarray:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return np.asarray(default, dtype=np.float64).copy()
    parts = [part.strip() for part in value.replace(",", " ").split() if part.strip()]
    if len(parts) == 1:
        return np.full(7, float(parts[0]), dtype=np.float64)
    if len(parts) != 7:
        raise ValueError(f"{name} must be one value or 7 comma-separated values")
    return np.asarray([float(part) for part in parts], dtype=np.float64)


def _load_pose(path: Path) -> tuple[np.ndarray, float | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    q = np.asarray(payload["joint_positions"], dtype=np.float64).reshape(7)
    # FACTR controllers operate in calibrated joint coordinates.  Saved pose
    # files may also contain the raw Dynamixel motor angle for diagnostics; do
    # not feed that raw value back into the calibrated controller.
    gripper = payload.get("gripper_position", payload.get("gripper_motor_position"))
    return q, None if gripper is None else float(gripper)


class FactrControlAdapter:
    """
    FACTR intervention controller for MuJoCo.

    Lifecycle:
    - startup staging: verify the parked rest pose, then move to the initial pose
    - intervention start: move FACTR to the paused MuJoCo robot pose
    - intervention active: FACTR is gravity-compensated and MuJoCo follows FACTR
    - intervention end/cancel: return FACTR to the configured initial pose
    """

    control_label = "FACTR"

    def __init__(self, robot_key: str = "factr", control_mode=None):
        del control_mode
        self.robot_key = robot_key
        self.config_path = Path(
            os.environ.get(
                "FACTR_CONFIG",
                os.environ.get("INTERVENE_FACTR_CONFIG", "factr/leader.yaml"),
            )
        )
        self.calibration_report = Path(
            os.environ.get(
                "INTERVENE_FACTR_CALIBRATION_REPORT",
                "factr/validation/calibration.json",
            )
        )
        self.torque_direction_report = Path(
            os.environ.get(
                "INTERVENE_FACTR_TORQUE_DIRECTION_REPORT",
                "factr/validation/torque_direction.json",
            )
        )
        self.gravity_report = Path(
            os.environ.get(
                "INTERVENE_FACTR_GRAVITY_REPORT",
                "factr/validation/gravity.json",
            )
        )
        self.initial_pose_path = Path(
            os.environ.get(
                "FACTR_INIT_POSE",
                os.environ.get(
                    "INTERVENE_FACTR_INITIAL_POSE",
                    "factr/validation/init_pose.json",
                ),
            )
        )
        self.rest_pose_path = Path(
            os.environ.get(
                "FACTR_REST_POSE",
                "factr/validation/rest_pose.json",
            )
        )

        self.factr: FACTRGravityCompensation | None = None
        self.controller: FactrPDController | None = None
        self.connected = False
        self._last_gripper_target: float | None = None
        self._return_rest_on_close = False
        self._move_cancel_event: threading.Event | None = None
        self._motion_lock = threading.RLock()
        self.robot = None
        # Set when startup staging failed but the arm is still being held under torque.
        # The caller must NOT close the adapter while this is true: closing disables
        # torque, and a leader arm has no brakes, so it would simply fall.
        self.staging_hold_active = False

    def _require_connected(self) -> FACTRGravityCompensation:
        if self.factr is None or not self.connected:
            raise RuntimeError("FACTR is not connected.")
        return self.factr

    def _enable_torque(self) -> None:
        factr = self._require_connected()
        if factr.driver is None:
            raise RuntimeError("FACTR driver is not initialized.")
        if not factr.driver.torque_enabled:
            factr.driver.set_operating_mode(0)
            factr.driver.set_torque_mode(True)

    def _disable_torque(self) -> None:
        factr = self._require_connected()
        if factr.driver is None:
            return
        try:
            factr.set_leader_joint_torque(
                np.zeros(factr.num_arm_joints, dtype=np.float64),
                0.0,
            )
        except Exception:
            pass
        factr.driver.set_torque_mode(False)

    def _stop_controller(self, *, disable_torque: bool = False) -> None:
        had_controller = self.controller is not None
        if self.controller is not None:
            self.controller.stop(disable_torque=disable_torque)
            self.controller = None
        if disable_torque and not had_controller and self.connected:
            self._disable_torque()

    def _controller_kwargs(self) -> dict:
        allow_over_config = _env_bool("INTERVENE_FACTR_ALLOW_OVER_CONFIG_TORQUE", True)
        return {
            "kp": _env_joint_array("INTERVENE_FACTR_KP", DEFAULT_FACTR_KP),
            "kd": _env_joint_array("INTERVENE_FACTR_KD", DEFAULT_FACTR_KD),
            "gripper_kp": _env_float("INTERVENE_FACTR_GRIPPER_KP", 5.0),
            "gripper_kd": _env_float("INTERVENE_FACTR_GRIPPER_KD", 2.0),
            "max_gripper_torque": _env_float("INTERVENE_FACTR_MAX_GRIPPER_TORQUE", 0.08),
            "gripper_torque_sign": _env_float("INTERVENE_FACTR_GRIPPER_TORQUE_SIGN", 1.0),
            "use_unclipped_gripper_position": True,
            "enforce_gripper_limits": False,
            "max_target_speed": _env_float(
                "FACTR_MAX_VELOCITY",
                _env_float("INTERVENE_FACTR_ALIGN_SPEED", 0.5),
            ),
            "max_position_error": _env_float("INTERVENE_FACTR_HOLD_ERROR", 0.04),
            "max_torque": _env_float("INTERVENE_FACTR_MAX_TORQUE", 10.0),
            "gravity_ramp_duration": _env_float("INTERVENE_FACTR_RAMP_DURATION", 5.0),
            "max_joint_velocity": _env_float("INTERVENE_FACTR_MAX_VELOCITY_RAD_S", 8.0),
            "max_state_jump": _env_float("INTERVENE_FACTR_MAX_STATE_JUMP_RAD", 3.2),
            "hold_gravity_compensation": _env_bool(
                "INTERVENE_FACTR_HOLD_GRAVITY_COMP", False
            ),
            "hold_friction_compensation": _env_bool(
                "INTERVENE_FACTR_HOLD_FRICTION_COMP", False
            ),
            "enforce_joint_limits": not _env_bool("INTERVENE_FACTR_DISABLE_LIMIT_TORQUE", False),
            "respect_config_torque_limit": not allow_over_config,
            "drive_torque": _env_joint_array("INTERVENE_FACTR_DRIVE_TORQUE", DEFAULT_FACTR_DRIVE_TORQUE),
            "drive_deadband": _env_float("INTERVENE_FACTR_DRIVE_DEADBAND", 0.05),
            "drive_ramp": _env_float("INTERVENE_FACTR_DRIVE_RAMP", 0.50),
        }

    def _waiting_hold_kwargs(self) -> dict:
        kwargs = self._controller_kwargs()
        kwargs["max_position_error"] = _env_float(
            "INTERVENE_FACTR_WAIT_HOLD_ERROR",
            kwargs["max_position_error"],
        )
        kwargs["max_torque"] = _env_float(
            "INTERVENE_FACTR_WAIT_MAX_TORQUE",
            kwargs["max_torque"],
        )
        kwargs["hold_gravity_compensation"] = _env_bool(
            "INTERVENE_FACTR_WAIT_HOLD_GRAVITY_COMP",
            kwargs["hold_gravity_compensation"],
        )
        kwargs["hold_friction_compensation"] = _env_bool(
            "INTERVENE_FACTR_WAIT_HOLD_FRICTION_COMP",
            kwargs["hold_friction_compensation"],
        )
        kwargs["drive_torque"] = _env_joint_array(
            "INTERVENE_FACTR_WAIT_DRIVE_TORQUE",
            kwargs["drive_torque"],
        )
        return kwargs

    def connect(self, already_connected=False):
        if already_connected and self.connected:
            print("[FACTR] Reusing existing FACTR connection.")
            return
        if self.connected:
            return

        validate_calibration_report(self.config_path, self.calibration_report)
        validate_torque_direction_report(self.config_path, self.torque_direction_report)
        validate_gravity_report(
            self.config_path,
            self.gravity_report,
            gain_scale=1.0,
            max_torque_nm=_env_float("INTERVENE_FACTR_MAX_TORQUE", 10.0),
        )

        print(f"[FACTR] Connecting controller from {self.config_path}.")
        try:
            self.factr = FACTRGravityCompensation(self.config_path, read_only=False)
            self.robot = self.factr
            self.connected = True

            state = self.factr.read_state()
            self._last_gripper_target = float(state.gripper_position)
        except Exception:
            self.close()
            raise

    def close(self):
        try:
            if (
                self.connected
                and self.factr is not None
                and self._return_rest_on_close
            ):
                self.return_to_rest_pose(best_effort=True)
            self._stop_controller(disable_torque=True)
        finally:
            try:
                if self.factr is not None:
                    self.factr.shutdown()
            finally:
                self.factr = None
                self.robot = None
                self.connected = False
                self._return_rest_on_close = False

    def _latest_state(self):
        factr = self._require_connected()
        if self.controller is not None:
            self.controller.raise_if_failed()
            state = factr.latest_state(max_age=0.5)
            if state is not None:
                return state
            time.sleep(0.01)
            state = factr.latest_state(max_age=0.5)
            if state is not None:
                return state
        return factr.read_state()

    def get_joint_positions(self):
        return self._latest_state().joint_positions.astype(np.float64)

    def get_joint_state(self):
        state = self._latest_state()
        return (
            state.joint_positions.astype(np.float64),
            state.joint_velocities.astype(np.float64),
        )

    def _move_to_pose(
        self,
        target_q,
        *,
        gripper_target: float | None,
        label: str,
        duration_s: float | None = None,
        allowed_start_q: np.ndarray | None = None,
        allowed_start_tolerance: float | None = None,
        tolerance: float | None = None,
        validate_safety_limits: bool | None = True,
        enforce_joint_limits: bool | None = None,
        cancel_event=None,
    ):
        factr = self._require_connected()
        self._stop_controller(disable_torque=True)

        if duration_s is None:
            duration_s = _env_float(
                "FACTR_ALIGNMENT_TIMEOUT",
                _env_float("INTERVENE_FACTR_GOTO_SECONDS", 18.0),
            )
        if tolerance is None:
            tolerance = _env_float(
                "FACTR_POSITION_TOLERANCE",
                _env_float("INTERVENE_FACTR_GOTO_TOL_RAD", 0.04),
            )
        kwargs = self._controller_kwargs()
        if enforce_joint_limits is not None:
            kwargs["enforce_joint_limits"] = bool(enforce_joint_limits)
        target_q = np.asarray(target_q, dtype=np.float64).reshape(7)
        if validate_safety_limits is not None:
            factr.validate_arm_positions(
                target_q,
                safety_limits=bool(validate_safety_limits),
            )

        start_state = factr.read_state()
        allowed_start_error = None
        if allowed_start_q is not None:
            allowed_start_q = np.asarray(allowed_start_q, dtype=np.float64).reshape(7)
            if allowed_start_tolerance is None or allowed_start_tolerance <= 0.0:
                raise ValueError("FACTR rest-position tolerance must be > 0")
            allowed_start_error = np.abs(
                (start_state.joint_positions - allowed_start_q + np.pi)
                % (2.0 * np.pi)
                - np.pi
            )
        try:
            factr.validate_arm_positions(
                start_state.joint_positions,
                safety_limits=True,
            )
        except FactrSafetyError:
            if allowed_start_error is None:
                raise
            if np.any(allowed_start_error > allowed_start_tolerance):
                joint_index = int(np.argmax(allowed_start_error)) + 1
                raise FactrSafetyError(
                    "FACTR is outside normal limits and does not match the "
                    "configured rest pose: "
                    f"J{joint_index} "
                    f"error={allowed_start_error[joint_index - 1]:.3f} rad "
                    f"> {allowed_start_tolerance:.3f} rad"
                )
            if not kwargs["enforce_joint_limits"]:
                raise FactrSafetyError(
                    "FACTR rest-to-initial movement requires joint-limit "
                    "direction enforcement"
                )
            print(
                "[FACTR] Rest pose verified outside the normal joint envelope; "
                "allowing the bounded inward move to the initial pose."
            )
        else:
            if allowed_start_error is not None and np.any(
                allowed_start_error > allowed_start_tolerance
            ):
                print(
                    "[FACTR] Current pose is already inside the normal joint "
                    "envelope; resuming movement to the initial pose."
                )

        max_velocity = _env_float(
            "FACTR_MAX_VELOCITY",
            _env_float("INTERVENE_FACTR_ALIGN_SPEED", 0.5),
        )
        if max_velocity <= 0.0:
            raise ValueError("FACTR_MAX_VELOCITY must be > 0")
        # Use the same requested smoothstep duration as go_to_ref_pose.py. The
        # shared controller still bounds local position error, torque, and its
        # velocity safety stop; do not silently rewrite this motion profile.
        trajectory_duration = _env_float(
            "INTERVENE_FACTR_TRAJECTORY_DURATION", 3.0
        )
        if trajectory_duration < 0.0:
            raise ValueError("FACTR trajectory duration must be >= 0")

        print(
            f"[FACTR] Moving to {label}: duration={float(duration_s):.1f}s, "
            f"trajectory={trajectory_duration:.1f}s, tol={tolerance:.3f} rad"
        )
        print(
            "[FACTR] Hold compensation: "
            f"gravity={kwargs['hold_gravity_compensation']}, "
            f"friction={kwargs['hold_friction_compensation']}; "
            f"drive_deadband={kwargs['drive_deadband']:.3f} rad, "
            f"drive_ramp={kwargs['drive_ramp']:.3f} rad"
        )
        # All preflight pose checks happen before torque is enabled.
        try:
            self._enable_torque()
            result = move_factr_to_joint_pose(
                factr,
                target_q,
                kp=kwargs["kp"],
                kd=kwargs["kd"],
                duration=float(duration_s),
                tolerance=tolerance,
                max_target_speed=kwargs["max_target_speed"],
                max_position_error=kwargs["max_position_error"],
                max_torque=kwargs["max_torque"],
                max_gripper_torque=kwargs["max_gripper_torque"],
                gripper_target=gripper_target,
                gripper_kp=kwargs["gripper_kp"],
                gripper_kd=kwargs["gripper_kd"],
                gripper_torque_sign=kwargs["gripper_torque_sign"],
                use_unclipped_gripper_position=True,
                enforce_gripper_limits=False,
                gravity_ramp_duration=kwargs["gravity_ramp_duration"],
                max_joint_velocity=kwargs["max_joint_velocity"],
                max_state_jump=kwargs["max_state_jump"],
                hold_gravity_compensation=kwargs["hold_gravity_compensation"],
                hold_friction_compensation=kwargs["hold_friction_compensation"],
                validate_safety_limits=validate_safety_limits,
                allowed_start_q=allowed_start_q,
                allowed_start_tolerance=(
                    0.0
                    if allowed_start_tolerance is None
                    else float(allowed_start_tolerance)
                ),
                enforce_joint_limits=kwargs["enforce_joint_limits"],
                respect_config_torque_limit=kwargs["respect_config_torque_limit"],
                drive_torque=kwargs["drive_torque"],
                drive_deadband=kwargs["drive_deadband"],
                drive_ramp=kwargs["drive_ramp"],
                trajectory_duration=trajectory_duration,
                cancel_event=cancel_event,
                keep_torque_on_cancel=True,
            )
        except BaseException:
            self._disable_torque()
            raise
        print(
            f"[FACTR] {label} result: reached={result.reached}, "
            f"max_error={result.max_error:.4f} rad"
        )
        final_error = (
            target_q - result.final_state.joint_positions + np.pi
        ) % (2.0 * np.pi) - np.pi
        print(
            "[FACTR] Final joint error [rad]: "
            + np.array2string(final_error, precision=4, suppress_small=True)
        )
        if cancel_event is not None and cancel_event.is_set():
            return result
        if not result.reached:
            raise TimeoutError(
                f"FACTR did not reach {label} within {float(duration_s):.1f}s; "
                f"max joint error={result.max_error:.4f} rad, "
                f"tolerance={tolerance:.4f} rad"
            )
        self._last_gripper_target = gripper_target
        return result

    def _start_controller(
        self,
        q_target: np.ndarray | None,
        *,
        waiting_hold: bool = False,
    ) -> None:
        factr = self._require_connected()
        self._stop_controller(disable_torque=False)
        self._enable_torque()
        kwargs = self._waiting_hold_kwargs() if waiting_hold else self._controller_kwargs()
        if q_target is None and not waiting_hold:
            kwargs["gravity_ramp_duration"] = _env_float(
                "INTERVENE_FACTR_TAKEOVER_GRAVITY_RAMP_DURATION",
                0.0,
            )
            kwargs["leader_nullspace_scale"] = _env_float(
                "INTERVENE_FACTR_TAKEOVER_NULLSPACE_SCALE",
                0.0,
            )
            kwargs["leader_gravity_scale"] = _env_float(
                "INTERVENE_FACTR_TAKEOVER_GRAVITY_SCALE",
                0.90,
            )
            kwargs["leader_friction_scale"] = _env_float(
                "INTERVENE_FACTR_TAKEOVER_FRICTION_SCALE",
                1.0,
            )
        self.controller = FactrPDController(factr, **kwargs)
        self.controller.set_position(
            None if q_target is None else np.asarray(q_target, dtype=np.float64).reshape(7),
            gripper=float(self._last_gripper_target or 0.0),
        )
        self.controller.start()

    def _force_controller_gravity_ramp_complete(self) -> None:
        if self.controller is None:
            return
        ramp_duration = float(getattr(self.controller, "_gravity_ramp_duration", 0.0))
        self.controller._started_at = time.monotonic() - ramp_duration

    def _hold_waiting_pose(self, q_target: np.ndarray, *, label: str) -> None:
        self._start_controller(q_target, waiting_hold=True)
        print(f"[FACTR] Holding {label} while waiting.")

    def stage_initial_pose(self):
        self.connect(already_connected=self.connected)
        rest_q, _ = _load_pose(self.rest_pose_path)
        q, gripper = _load_pose(self.initial_pose_path)
        rest_tolerance = _env_float("FACTR_REST_POSITION_TOLERANCE", 0.08)
        print(
            f"[FACTR] Expecting parked rest pose {self.rest_pose_path} "
            f"within {rest_tolerance:.3f} rad."
        )
        try:
            result = self._move_to_pose(
                q,
                gripper_target=gripper,
                label=f"initial pose {self.initial_pose_path}",
                duration_s=_env_float("FACTR_INIT_TIMEOUT", 18.0),
                allowed_start_q=rest_q,
                allowed_start_tolerance=rest_tolerance,
            )
        except Exception:
            # Staging failed with torque still enabled (move_factr_to_joint_pose is
            # called with keep_torque_on_cancel=True). Letting this propagate into
            # close() would disable torque and DROP the arm, which is what happened on
            # 2026-08-03 when the move stalled 0.063 rad short of a 0.040 rad tolerance.
            # Hold wherever it actually got to and hand the decision to the operator.
            if _env_bool("INTERVENE_FACTR_HOLD_ON_STAGING_FAILURE", True):
                try:
                    current_q = self.get_joint_positions()
                    self._hold_waiting_pose(
                        current_q, label="pose held after failed staging"
                    )
                    self.staging_hold_active = True
                    # Whoever eventually stops the service — the operator, or the
                    # launcher's ready-file timeout — must get a CONTROLLED return to the
                    # rest pose, not a torque cut. rest_pose is the mechanically stable
                    # parked configuration, so this is the safe place to let go.
                    self._return_rest_on_close = True
                    print(
                        "[FACTR] Staging failed. The arm is HELD at its current pose, "
                        "not released.\n"
                        "[FACTR] Support the arm. On stop it will return to the rest "
                        "pose before torque is disabled.",
                        flush=True,
                    )
                except Exception as hold_exc:
                    # Holding is best-effort: never mask the real staging failure.
                    print(
                        f"[FACTR][WARN] Could not hold after staging failure: {hold_exc}",
                        flush=True,
                    )
            raise
        self._last_gripper_target = gripper
        self._hold_waiting_pose(q, label="initial pose")
        self._return_rest_on_close = True
        return result

    def return_to_rest_pose(self, *, best_effort: bool = False):
        if not _env_bool("INTERVENE_FACTR_RETURN_REST_ON_CLOSE", True):
            return None
        if not self.connected:
            self.connect(already_connected=False)
        q, gripper = _load_pose(self.rest_pose_path)
        timeout = _env_float("INTERVENE_FACTR_REST_TIMEOUT", 18.0)
        tolerance = _env_float("INTERVENE_FACTR_REST_GOAL_TOLERANCE", 0.10)
        enforce_limits = not _env_bool("INTERVENE_FACTR_REST_DISABLE_LIMIT_TORQUE", True)
        print(
            f"[FACTR] Returning to rest pose {self.rest_pose_path}: "
            f"duration={timeout:.1f}s, tol={tolerance:.3f} rad"
        )
        try:
            result = self._move_to_pose(
                q,
                gripper_target=gripper,
                label=f"rest pose {self.rest_pose_path}",
                duration_s=timeout,
                tolerance=tolerance,
                validate_safety_limits=None,
                enforce_joint_limits=enforce_limits,
            )
        except Exception as exc:
            if not best_effort:
                raise
            print(
                "[FACTR][WARN] Could not return to rest pose before shutdown: "
                f"{type(exc).__name__}: {exc}"
            )
            return None
        self._last_gripper_target = gripper
        return result

    def arm_move_cancel_event(self) -> threading.Event:
        """Publish the cancel event BEFORE a background move starts.

        The caller that spawns the move calls this synchronously, so an interrupt
        arriving during the move's own setup (connect + pose load) still has something
        to set. Without it there is a window where interrupt_return_to_initial() finds
        None, cancels nothing, and two things end up commanding the arm.
        """
        event = self._move_cancel_event
        if event is None:
            event = threading.Event()
            self._move_cancel_event = event
        return event

    def return_to_initial_pose(self):
        # Reuse a pre-armed event if the caller published one; only create a fresh event
        # when this is called directly (no pre-arm), which is the standalone path.
        cancel_event = self.arm_move_cancel_event()
        if cancel_event.is_set():
            # Cancelled before we even began — honour it rather than start a move that
            # is already unwanted.
            print("[FACTR] Return to initial pose cancelled before it started.")
            self._move_cancel_event = None
            return None
        if not self.connected:
            self.connect(already_connected=False)
        q, gripper = _load_pose(self.initial_pose_path)
        with self._motion_lock:
            try:
                result = self._move_to_pose(
                    q,
                    gripper_target=gripper,
                    label="initial pose",
                    cancel_event=cancel_event,
                )
            finally:
                if self._move_cancel_event is cancel_event:
                    self._move_cancel_event = None
        if cancel_event.is_set():
            print("[FACTR] Return to initial pose interrupted by new intervention.")
            return result
        self._last_gripper_target = gripper
        self._hold_waiting_pose(q, label="initial pose")
        return result

    def interrupt_return_to_initial(self) -> None:
        event = self._move_cancel_event
        if event is not None:
            event.set()
        with self._motion_lock:
            if not self.connected:
                return
            state = self._latest_state()
            self._last_gripper_target = float(state.gripper_position)
            self._hold_waiting_pose(
                state.joint_positions,
                label="interrupted return pose",
            )

    @staticmethod
    def mujoco_to_factr_joints(q_mujoco) -> np.ndarray:
        """Map Panda MuJoCo joints to calibrated FACTR logical joints.

        FACTR applies the configured Dynamixel order, signs, offsets, wrapping,
        and calibration report while reading/writing hardware.  At this layer
        both robots therefore use the same seven Panda logical joint coordinates.
        """
        q = np.asarray(q_mujoco, dtype=np.float64)
        if q.shape != (7,):
            raise ValueError(
                f"MuJoCo/FACTR joint dimension mismatch: expected 7, got {q.shape}"
            )
        return q.copy()

    @staticmethod
    def factr_to_mujoco_joints(q_factr) -> np.ndarray:
        q = np.asarray(q_factr, dtype=np.float64)
        if q.shape != (7,):
            raise ValueError(
                f"FACTR/MuJoCo joint dimension mismatch: expected 7, got {q.shape}"
            )
        return q.copy()

    def align_to_mujoco(
        self,
        q_mujoco,
        *,
        gripper_width: float | None = None,
        tolerance: float | None = None,
        timeout: float | None = None,
        max_velocity: float | None = None,
    ) -> bool:
        q_factr = self.mujoco_to_factr_joints(q_mujoco)
        factr = self._require_connected()
        factr.validate_arm_positions(q_factr, safety_limits=True)
        if tolerance is not None:
            os.environ["FACTR_POSITION_TOLERANCE"] = str(float(tolerance))
        if max_velocity is not None:
            os.environ["FACTR_MAX_VELOCITY"] = str(float(max_velocity))
        if gripper_width is not None:
            self._last_gripper_target = self._width_to_gripper_pos(float(gripper_width))
        result = self._move_to_pose(
            q_factr,
            gripper_target=self._last_gripper_target,
            label="paused MuJoCo pose",
            duration_s=timeout,
        )
        self._hold_waiting_pose(q_factr, label="paused MuJoCo pose")
        return bool(result.reached)

    def begin_takeover(self) -> None:
        state = self._latest_state()
        self._last_gripper_target = float(state.gripper_position)
        assist_duration = _env_float("INTERVENE_FACTR_TAKEOVER_HOLD_ASSIST_SECONDS", 1.0)
        blend_duration = _env_float("INTERVENE_FACTR_TAKEOVER_TORQUE_BLEND_SECONDS", 0.6)
        nullspace_scale = _env_float("INTERVENE_FACTR_TAKEOVER_NULLSPACE_SCALE", 0.0)
        gravity_scale = _env_float("INTERVENE_FACTR_TAKEOVER_GRAVITY_SCALE", 0.90)
        friction_scale = _env_float("INTERVENE_FACTR_TAKEOVER_FRICTION_SCALE", 1.0)
        if self.controller is not None:
            self._enable_torque()
            self.controller.raise_if_failed()
            if hasattr(self.controller, "start_leader_mode"):
                self.controller.start_leader_mode(
                    anchor_q=state.joint_positions,
                    gripper=float(state.gripper_position),
                    hold_assist_duration=assist_duration,
                    torque_blend_duration=blend_duration,
                    leader_nullspace_scale=nullspace_scale,
                    leader_gravity_scale=gravity_scale,
                    leader_friction_scale=friction_scale,
                )
            else:
                self.controller.set_position(None, gripper=float(state.gripper_position))
            self._force_controller_gravity_ramp_complete()
        else:
            self._start_controller(None)
            self._force_controller_gravity_ramp_complete()
        print(
            "[FACTR] Leader mode active; MuJoCo may now follow FACTR "
            f"(hold assist={assist_duration:.2f}s, "
            f"torque blend={blend_duration:.2f}s, "
            f"nullspace={nullspace_scale:.2f}, gravity={gravity_scale:.2f})."
        )

    def wait_for_takeover_touch(self, anchor_q) -> np.ndarray:
        threshold = _env_float("INTERVENE_FACTR_TOUCH_RELEASE_DELTA_RAD", 0.015)
        if threshold <= 0.0:
            return self._latest_state().joint_positions.astype(np.float64).copy()
        timeout = _env_float("INTERVENE_FACTR_TOUCH_RELEASE_TIMEOUT", 0.0)
        settle_s = _env_float("INTERVENE_FACTR_TOUCH_RELEASE_REANCHOR_SECONDS", 0.25)
        if settle_s > 0.0:
            time.sleep(settle_s)
        anchor = self._latest_state().joint_positions.astype(np.float64).copy()
        print(
            "[FACTR] Holding paused pose. Move FACTR slightly to begin teleoperation "
            f"(threshold={threshold:.3f} rad)."
        )
        started = time.monotonic()
        last_notice = started
        while True:
            if self.controller is not None:
                self.controller.raise_if_failed()
            state = self._latest_state()
            delta = (state.joint_positions - anchor + np.pi) % (2.0 * np.pi) - np.pi
            max_delta = float(np.max(np.abs(delta)))
            if max_delta >= threshold:
                print(
                    "[FACTR] Touch release detected; entering leader teleoperation "
                    f"(motion={max_delta:.3f} rad)."
                )
                return state.joint_positions.astype(np.float64).copy()
            now = time.monotonic()
            if timeout > 0.0 and now - started >= timeout:
                print(
                    "[FACTR] Touch release wait timed out; entering leader "
                    "teleoperation."
                )
                return state.joint_positions.astype(np.float64).copy()
            if now - last_notice >= 2.0:
                print(
                    "[FACTR] Still holding paused pose; move FACTR slightly to "
                    f"take over (motion={max_delta:.3f}/{threshold:.3f} rad)."
                )
                last_notice = now
            time.sleep(0.02)

    def end_takeover(self, *, return_to_initial: bool | None = None) -> None:
        if not self.connected:
            return
        state = self._latest_state()
        self._last_gripper_target = float(state.gripper_position)
        if return_to_initial is None:
            return_to_initial = _env_bool("INTERVENE_FACTR_RETURN_INIT_ON_RELEASE", True)
        if return_to_initial:
            print("[FACTR] Intervention released; returning FACTR to initial pose.")
            self.return_to_initial_pose()
        else:
            self._hold_waiting_pose(state.joint_positions, label="current FACTR pose")

    def go_to_joint_positions(self, q_cmd, max_vel_norm_factor=None, duration_s=None):
        del max_vel_norm_factor
        gripper = self._last_gripper_target
        if gripper is None:
            state = self._latest_state()
            factr = self._require_connected()
            gripper = float(getattr(factr, "gripper_pos_unclipped", state.gripper_position))
        q_factr = self.mujoco_to_factr_joints(q_cmd)
        result = self._move_to_pose(
            q_factr,
            gripper_target=gripper,
            label="paused MuJoCo robot pose",
            duration_s=duration_s,
        )
        return result

    def send_joint_positions(self, q_cmd, qd_cmd=None):
        del q_cmd, qd_cmd
        raise RuntimeError(
            "FACTR is the leader during intervention; MuJoCo follows FACTR state."
        )

    def switch_control_mode(self, control_mode):
        mode = str(control_mode or "").strip().upper()
        if mode in {"OFF", "DISABLE", "DISABLED", "TORQUE_DISABLED"}:
            self._stop_controller(disable_torque=True)
            print("[FACTR] Torque disabled.")
            return

        if mode in {"HUMAN_CONTROL", "LEADER", "TAKEOVER"}:
            self.begin_takeover()
        else:
            self.end_takeover()

    def _gripper_pos_to_width(self, position: float) -> float:
        factr = self._require_connected()
        low = float(factr.gripper_limit_min)
        high = float(factr.gripper_limit_max)
        max_width = _env_float("INTERVENE_FACTR_GRIPPER_MAX_WIDTH_M", 0.08)
        if high <= low:
            return 0.0
        normalized = (float(position) - low) / (high - low)
        return float(np.clip(normalized, 0.0, 1.0) * max_width)

    def _width_to_gripper_pos(self, width: float) -> float:
        factr = self._require_connected()
        low = float(factr.gripper_limit_min)
        high = float(factr.gripper_limit_max)
        max_width = _env_float("INTERVENE_FACTR_GRIPPER_MAX_WIDTH_M", 0.08)
        normalized = 0.0 if max_width <= 0.0 else float(width) / max_width
        return low + float(np.clip(normalized, 0.0, 1.0)) * (high - low)

    def get_gripper_width(self):
        state = self._latest_state()
        factr = self._require_connected()
        gripper_pos = float(getattr(factr, "gripper_pos_unclipped", state.gripper_position))
        return self._gripper_pos_to_width(gripper_pos)

    def get_gripper_pressed(self):
        state = self._latest_state()
        factr = self._require_connected()
        gripper_pos = float(getattr(factr, "gripper_pos_unclipped", state.gripper_position))
        threshold = _env_float(
            "INTERVENE_FACTR_GRIPPER_THRESHOLD",
            float(factr.gripper_close_threshold),
        )
        close_above = _env_bool("INTERVENE_FACTR_GRIPPER_CLOSE_ABOVE", False)
        if close_above:
            return gripper_pos >= threshold
        return gripper_pos < threshold

    def restore_gripper_width(self, target_width, tol=0.002, timeout=1.5, hz=40.0):
        del tol, timeout, hz
        self._last_gripper_target = self._width_to_gripper_pos(float(target_width))
