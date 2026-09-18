from dataclasses import dataclass
import threading
import time
from typing import Any, Callable, Optional

import numpy as np
import numpy.typing as npt

from factr.hardware.factr import (
    FACTRGravityCompensation,
    FactrError,
    FactrSafetyError,
    FactrState,
)


def _wrapped_joint_delta(
    current: npt.NDArray[np.float64],
    anchor: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    delta = np.asarray(current, dtype=np.float64) - np.asarray(anchor, dtype=np.float64)
    return (delta + np.pi) % (2.0 * np.pi) - np.pi


@dataclass(frozen=True)
class FactrMoveToPoseResult:
    start_state: FactrState
    final_state: FactrState
    reached: bool
    max_error: float
    elapsed_s: float


class FactrPDController:
    """
    Single-thread controller for FACTR that supports two modes:

    - Hold mode  (set_position called with an array): PD + gravity comp to hold a target.
    - Leader mode (set_position called with None):    YAML controller terms under
                                                      this wrapper's ramp and torque cap.

    This replaces calling factr.start() directly; instead call this controller's start().
    factr.get_state() continues to work in both modes because this controller writes the
    same _leader_state_* cache fields that get_state() reads.
    """

    def __init__(
        self,
        factr: FACTRGravityCompensation,
        kp: np.ndarray | None = None,
        kd: np.ndarray | None = None,
        gripper_kp: float = 10.0,
        gripper_kd: float = 1.0,
        max_gripper_torque: float = 0.15,
        gripper_torque_sign: float = 1.0,
        use_unclipped_gripper_position: bool = False,
        enforce_gripper_limits: bool = True,
        max_target_speed: float = 0.25,
        max_position_error: float = 0.08,
        max_torque: float = 0.05,
        gravity_ramp_duration: float = 1.0,
        max_joint_velocity: float = 6.0,
        max_state_jump: float = 0.75,
        hold_gravity_compensation: bool = True,
        hold_friction_compensation: bool = True,
        enforce_joint_limits: bool = True,
        respect_config_torque_limit: bool = True,
        drive_torque: npt.ArrayLike | float = 0.0,
        drive_deadband: float = 0.08,
        drive_ramp: float = 0.30,
        leader_nullspace_scale: float = 1.0,
        leader_gravity_scale: float = 1.0,
        leader_friction_scale: float = 1.0,
    ):
        self._factr = factr
        self._kp = (
            np.asarray(kp, dtype=np.float64)
            if kp is not None
            else np.array([0.75, 2.0, 1.0, 2.25, 0.5, 0.35, 0.25])
        )
        self._kd = (
            np.asarray(kd, dtype=np.float64)
            if kd is not None
            else np.array([0.01, 0.005, 0.01, 0.01, 0.01, 0.005, 0.01])
        )
        self._gripper_kp = gripper_kp
        self._gripper_kd = gripper_kd
        self._max_gripper_torque = abs(float(max_gripper_torque))
        self._gripper_torque_sign = 1.0 if float(gripper_torque_sign) >= 0.0 else -1.0
        self._use_unclipped_gripper_position = bool(use_unclipped_gripper_position)
        self._enforce_gripper_limits = bool(enforce_gripper_limits)
        self._max_target_speed = float(max_target_speed)
        self._max_position_error = float(max_position_error)
        self._max_torque = abs(float(max_torque))
        requested_torque = np.full(7, self._max_torque, dtype=np.float64)
        if respect_config_torque_limit:
            requested_torque = np.minimum(
                np.asarray(self._factr.max_arm_torque, dtype=np.float64).reshape(7),
                requested_torque,
            )
        self._arm_torque_limit = requested_torque
        self._gravity_ramp_duration = max(0.0, float(gravity_ramp_duration))
        self._max_joint_velocity = float(max_joint_velocity)
        self._max_state_jump = float(max_state_jump)
        self._hold_gravity_compensation = bool(hold_gravity_compensation)
        self._hold_friction_compensation = bool(hold_friction_compensation)
        self._enforce_joint_limits = bool(enforce_joint_limits)
        drive_torque_array = np.asarray(drive_torque, dtype=np.float64)
        if drive_torque_array.shape == ():
            drive_torque_array = np.full(7, float(drive_torque_array), dtype=np.float64)
        self._drive_torque = np.abs(drive_torque_array).reshape(7)
        self._drive_deadband = max(0.0, float(drive_deadband))
        self._drive_ramp = max(1e-6, float(drive_ramp))
        self._leader_nullspace_scale = float(leader_nullspace_scale)
        self._leader_gravity_scale = float(leader_gravity_scale)
        self._leader_friction_scale = float(leader_friction_scale)

        self._target_lock = threading.Lock()
        self._q_target: Optional[npt.NDArray[np.float64]] = None
        self._filtered_q_target: Optional[npt.NDArray[np.float64]] = None
        self._gripper_target: float = 0.0
        self._leader_hold_anchor: Optional[npt.NDArray[np.float64]] = None
        self._leader_hold_assist_started_at: float | None = None
        self._leader_hold_assist_duration = 0.0
        self._leader_torque_blend_start: Optional[npt.NDArray[np.float64]] = None
        self._leader_torque_blend_started_at: float | None = None
        self._leader_torque_blend_duration = 0.0

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._failed: BaseException | None = None
        self._previous_q: npt.NDArray[np.float64] | None = None
        self._started_at: float | None = None
        self.latest_arm_torque = np.zeros(7, dtype=np.float64)
        self.latest_gripper_torque = 0.0
        self.latest_filtered_target: npt.NDArray[np.float64] | None = None

    def set_position(
        self, q: Optional[npt.NDArray[np.float64]], gripper: float = 0.0
    ) -> None:
        """Switch to hold mode (q is an array) or leader/GC mode (q is None)."""
        with self._target_lock:
            self._q_target = (
                None if q is None else np.asarray(q, dtype=np.float64).copy()
            )
            if q is None:
                self._filtered_q_target = None
                self._leader_hold_anchor = None
                self._leader_hold_assist_started_at = None
                self._leader_torque_blend_start = None
                self._leader_torque_blend_started_at = None
            self._gripper_target = float(gripper)

    def start_leader_mode(
        self,
        *,
        anchor_q: Optional[npt.NDArray[np.float64]] = None,
        gripper: float = 0.0,
        hold_assist_duration: float = 0.0,
        torque_blend_duration: float = 0.0,
        leader_nullspace_scale: float | None = None,
        leader_gravity_scale: float | None = None,
        leader_friction_scale: float | None = None,
    ) -> None:
        """Switch to leader/GC mode with an optional decaying hold assist.

        The assist avoids a sudden torque drop when changing from PD hold to
        human-controlled gravity compensation.  The torque blend avoids an
        impulse if the leader gravity torque is above or below the previous
        hold torque.
        """
        with self._target_lock:
            self._q_target = None
            self._filtered_q_target = None
            self._gripper_target = float(gripper)
            if leader_nullspace_scale is not None:
                self._leader_nullspace_scale = float(leader_nullspace_scale)
            if leader_gravity_scale is not None:
                self._leader_gravity_scale = float(leader_gravity_scale)
            if leader_friction_scale is not None:
                self._leader_friction_scale = float(leader_friction_scale)
            duration = max(0.0, float(hold_assist_duration))
            if anchor_q is not None and duration > 0.0:
                self._leader_hold_anchor = np.asarray(
                    anchor_q, dtype=np.float64
                ).reshape(7).copy()
                self._leader_hold_assist_started_at = time.monotonic()
                self._leader_hold_assist_duration = duration
            else:
                self._leader_hold_anchor = None
                self._leader_hold_assist_started_at = None
                self._leader_hold_assist_duration = 0.0
            blend_duration = max(0.0, float(torque_blend_duration))
            if blend_duration > 0.0:
                self._leader_torque_blend_start = np.asarray(
                    self.latest_arm_torque, dtype=np.float64
                ).reshape(7).copy()
                self._leader_torque_blend_started_at = time.monotonic()
                self._leader_torque_blend_duration = blend_duration
            else:
                self._leader_torque_blend_start = None
                self._leader_torque_blend_started_at = None
                self._leader_torque_blend_duration = 0.0

    def _step_hold(
        self, q_target: npt.NDArray[np.float64], gripper_target: float
    ) -> None:
        factr = self._factr
        arm_pos, arm_vel, gripper_pos, gripper_vel = factr.get_leader_joint_states(
            clip_gripper=not self._use_unclipped_gripper_position
        )

        # scale the kp-gains for the current kp-values
        # kp=np.array([0.25, 1.0, 0.75, 1.0, 0.5, 0.5, 0.25]), 
        # kd=np.array([0.025, 0.1, 0.075, 0.1, 0.05, 0.05, 0.025]))
        # scale = np.array([1.5, 1.3, 1.1, 1.0, 0.9, 0.8, 0.7])
        # we would need the per-joint inertia but approximate it with the scale factor
        # kp = self._kp / (scale ** 2)
        # kd = self._kd / scale
        kp = self._kp
        kd = self._kd

        # Match the original FACTR helper: every control step creates a local
        # target near the current arm position instead of integrating a target
        # trajectory. This avoids target windup when the arm lags or is held.
        max_step = max(1e-6, self._max_position_error)
        target_error = _wrapped_joint_delta(q_target, arm_pos)
        q_command = arm_pos + np.clip(
            target_error,
            -max_step,
            max_step,
        )

        with factr._leader_state_lock:
            factr._leader_arm_pos_latest[:] = arm_pos
            factr._leader_arm_vel_latest[:] = arm_vel
            factr._leader_gripper_pos_latest = float(gripper_pos)
            factr._leader_gripper_vel_latest = float(gripper_vel)
            factr._leader_state_timestamp = time.monotonic()
            factr._leader_state_ready = True

        # tau = (
        #     self._kp * (q_target - arm_pos)
        #     - self._kd * arm_vel
        #     + factr.gravity_compensation(arm_pos, arm_vel)
        # )
        # strong global damping (critical)
        position_error = np.clip(
            q_command - arm_pos,
            -self._max_position_error,
            self._max_position_error,
        )
        tau = kp * position_error - kd * arm_vel
        tau_l_gripper = 0.0
        if np.any(self._drive_torque > 0.0):
            drive_alpha = np.clip(
                (np.abs(target_error) - self._drive_deadband) / self._drive_ramp,
                0.0,
                1.0,
            )
            tau += np.sign(target_error) * self._drive_torque * drive_alpha
        if self._enforce_joint_limits:
            tau_l, tau_l_gripper = factr.joint_limit_barrier(
                arm_pos,
                arm_vel,
                gripper_pos,
                gripper_vel,
            )
            tau += tau_l
            if not self._enforce_gripper_limits:
                tau_l_gripper = 0.0
        if self._hold_gravity_compensation:
            tau += factr.gravity_compensation(arm_pos, arm_vel)
        if self._hold_friction_compensation:
            tau += factr.friction_compensation(arm_vel)
        if self._enforce_joint_limits:
            tau = factr.enforce_joint_limit_direction(arm_pos, tau)
        tau = np.clip(tau, -self._arm_torque_limit, self._arm_torque_limit)
        self.latest_arm_torque = tau.copy()
        self.latest_filtered_target = q_command.copy()

        tau_gripper = (
            self._gripper_kp * (gripper_target - gripper_pos)
            - self._gripper_kd * gripper_vel
        )
        if self._enforce_joint_limits:
            tau_gripper += tau_l_gripper
        tau_gripper = float(
            np.clip(tau_gripper, -self._max_gripper_torque, self._max_gripper_torque)
        )
        tau_gripper *= self._gripper_torque_sign
        self.latest_gripper_torque = float(tau_gripper)

        # tau = np.clip(tau, -0.5, 0.5)
        factr.set_leader_joint_torque(tau, tau_gripper)

    def _cache_state(
        self,
        arm_pos: npt.NDArray[np.float64],
        arm_vel: npt.NDArray[np.float64],
        gripper_pos: float,
        gripper_vel: float,
    ) -> None:
        factr = self._factr
        with factr._leader_state_lock:
            factr._leader_arm_pos_latest[:] = arm_pos
            factr._leader_arm_vel_latest[:] = arm_vel
            factr._leader_gripper_pos_latest = float(gripper_pos)
            factr._leader_gripper_vel_latest = float(gripper_vel)
            factr._leader_state_timestamp = time.monotonic()
            factr._leader_state_ready = True

    def _gravity_ramp(self) -> float:
        if self._gravity_ramp_duration <= 0.0 or self._started_at is None:
            return 1.0
        elapsed = time.monotonic() - self._started_at
        return min(1.0, max(0.0, elapsed / self._gravity_ramp_duration))

    def _step_leader_gravity(self) -> None:
        factr = self._factr
        arm_pos, arm_vel, gripper_pos, gripper_vel = factr.get_leader_joint_states()
        self._cache_state(arm_pos, arm_vel, gripper_pos, gripper_vel)

        torque_arm = np.zeros(factr.num_arm_joints, dtype=np.float64)
        torque_l, torque_gripper = factr.joint_limit_barrier(
            arm_pos, arm_vel, gripper_pos, gripper_vel
        )
        torque_arm += torque_l
        if self._leader_nullspace_scale != 0.0:
            torque_arm += self._leader_nullspace_scale * factr.null_space_regulation(
                arm_pos, arm_vel
            )

        if factr.enable_gravity_comp:
            torque_arm += self._leader_gravity_scale * factr.gravity_compensation(
                arm_pos, arm_vel
            )
            torque_arm += self._leader_friction_scale * factr.friction_compensation(
                arm_vel
            )

        if factr.enable_torque_feedback:
            external_joint_torque = factr.get_leader_arm_external_joint_torque()
            torque_arm += factr.torque_feedback(external_joint_torque, arm_vel)

        with self._target_lock:
            assist_anchor = (
                None
                if self._leader_hold_anchor is None
                else self._leader_hold_anchor.copy()
            )
            assist_started = self._leader_hold_assist_started_at
            assist_duration = self._leader_hold_assist_duration
        if (
            assist_anchor is not None
            and assist_started is not None
            and assist_duration > 0.0
        ):
            assist_progress = np.clip(
                (time.monotonic() - assist_started) / assist_duration,
                0.0,
                1.0,
            )
            assist_alpha = float(1.0 - assist_progress)
            assist_error = np.clip(
                _wrapped_joint_delta(assist_anchor, arm_pos),
                -self._max_position_error,
                self._max_position_error,
            )
            torque_arm += assist_alpha * (
                self._kp * assist_error - self._kd * arm_vel
            )
            if assist_progress >= 1.0:
                with self._target_lock:
                    self._leader_hold_anchor = None
                    self._leader_hold_assist_started_at = None
                    self._leader_hold_assist_duration = 0.0

        if factr.enable_gripper_feedback:
            external_gripper_torque = factr.get_leader_gripper_external_torque()
            torque_gripper += factr.gripper_feedback(external_gripper_torque, gripper_vel)

        ramp = self._gravity_ramp()
        torque_arm = factr.enforce_joint_limit_direction(arm_pos, torque_arm)
        torque_arm = np.clip(
            torque_arm * ramp,
            -self._arm_torque_limit,
            self._arm_torque_limit,
        )
        with self._target_lock:
            blend_start = (
                None
                if self._leader_torque_blend_start is None
                else self._leader_torque_blend_start.copy()
            )
            blend_started = self._leader_torque_blend_started_at
            blend_duration = self._leader_torque_blend_duration
        if (
            blend_start is not None
            and blend_started is not None
            and blend_duration > 0.0
        ):
            blend_progress = float(
                np.clip((time.monotonic() - blend_started) / blend_duration, 0.0, 1.0)
            )
            blend_progress = blend_progress * blend_progress * (3.0 - 2.0 * blend_progress)
            torque_arm = blend_start + blend_progress * (torque_arm - blend_start)
            torque_arm = np.clip(
                torque_arm,
                -self._arm_torque_limit,
                self._arm_torque_limit,
            )
            if blend_progress >= 1.0:
                with self._target_lock:
                    self._leader_torque_blend_start = None
                    self._leader_torque_blend_started_at = None
                    self._leader_torque_blend_duration = 0.0
        self.latest_arm_torque = torque_arm.copy()
        self.latest_filtered_target = None
        torque_gripper = float(
            np.clip(torque_gripper * ramp, -self._max_torque, self._max_torque)
        )

        if factr.enable_gravity_comp:
            factr.set_leader_joint_torque(torque_arm, torque_gripper)

    def _step(self) -> None:
        with self._target_lock:
            q_target = self._q_target
            gripper_target = self._gripper_target

        if q_target is None:
            self._step_leader_gravity()
        else:
            self._step_hold(q_target, gripper_target)
        state = self._factr.latest_state()
        if state is not None:
            abs_velocity = np.abs(state.joint_velocities)
            max_velocity = float(np.max(abs_velocity))
            if max_velocity > self._max_joint_velocity:
                joint_index = int(np.argmax(abs_velocity)) + 1
                raise FactrError(
                    "FACTR controller velocity safety limit exceeded: "
                    f"J{joint_index}={max_velocity:.3f} rad/s > "
                    f"{self._max_joint_velocity:.3f} rad/s"
                )
            if self._previous_q is not None:
                state_delta = state.joint_positions - self._previous_q
                state_delta = (state_delta + np.pi) % (2.0 * np.pi) - np.pi
                abs_jump = np.abs(state_delta)
                jump = float(np.max(abs_jump))
                if jump > self._max_state_jump:
                    joint_index = int(np.argmax(abs_jump)) + 1
                    raise FactrError(
                        "FACTR controller state jump safety limit exceeded: "
                        f"J{joint_index}={jump:.3f} rad > "
                        f"{self._max_state_jump:.3f} rad"
                    )
            self._previous_q = state.joint_positions.copy()

    def start(self, env: Optional[Any] = None) -> None:
        """Start the control loop. Pass env to enable Franka torque feedback."""
        del env
        assert not self._running

        self._running = True
        self._failed = None
        self._started_at = time.monotonic()

        def loop() -> None:
            while self._running:
                t0 = time.time()
                try:
                    self._step()
                except BaseException as exc:
                    self._failed = exc
                    self._running = False
                    break
                sleep = max(0.0, factr.dt - (time.time() - t0))
                if sleep:
                    time.sleep(sleep)

        factr = self._factr
        self._thread = threading.Thread(target=loop, daemon=True, name="factr_pd")
        self._thread.start()

    def raise_if_failed(self) -> None:
        if self._failed is not None:
            raise FactrError(f"FACTR controller loop failed: {self._failed}") from self._failed

    def stop(self, *, disable_torque: bool = False) -> None:
        """Stop the control loop thread. Call factr.shutdown() separately to release hardware."""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if disable_torque and self._factr.driver is not None:
            try:
                self._factr.set_leader_joint_torque(
                    np.zeros(self._factr.num_arm_joints), 0.0
                )
            except Exception:
                pass
            self._factr.driver.set_torque_mode(False)


def move_factr_to_joint_pose(
    factr: FACTRGravityCompensation,
    target_q: npt.ArrayLike,
    *,
    kp: npt.ArrayLike | None = None,
    kd: npt.ArrayLike | None = None,
    duration: float = 8.0,
    tolerance: float = 0.05,
    max_target_speed: float = 2.5,
    max_position_error: float = 0.02,
    max_torque: float = 0.08,
    max_gripper_torque: float = 0.15,
    gripper_target: float | None = None,
    gripper_kp: float = 10.0,
    gripper_kd: float = 1.0,
    gripper_torque_sign: float = 1.0,
    use_unclipped_gripper_position: bool = False,
    enforce_gripper_limits: bool = True,
    gravity_ramp_duration: float = 5.0,
    max_joint_velocity: float = 0.8,
    max_state_jump: float = 0.20,
    hold_gravity_compensation: bool = False,
    hold_friction_compensation: bool = False,
    validate_safety_limits: bool | None = True,
    allowed_start_q: npt.ArrayLike | None = None,
    allowed_start_tolerance: float = 0.0,
    enforce_joint_limits: bool = True,
    respect_config_torque_limit: bool = True,
    drive_torque: npt.ArrayLike | float = 0.0,
    drive_deadband: float = 0.08,
    drive_ramp: float = 0.30,
    trajectory_duration: float = 0.0,
    poll_interval: float = 0.05,
    progress_period: float = 1.0,
    progress_callback: Callable[
        [float, FactrState, npt.NDArray[np.float64], FactrPDController], None
    ]
    | None = None,
    cancel_event: threading.Event | None = None,
    keep_torque_on_cancel: bool = False,
) -> FactrMoveToPoseResult:
    """Move FACTR toward a joint pose with bounded torque-PD control.

    The helper owns the controller thread and always disables torque before
    returning or raising. It uses the controller-cached state while running so
    callers do not read the serial port concurrently with the control loop.
    """
    target = np.asarray(target_q, dtype=np.float64).reshape(factr.num_arm_joints)
    if validate_safety_limits is not None:
        factr.validate_arm_positions(target, safety_limits=validate_safety_limits)
    start_state = factr.read_state()
    allowed_start = None
    if allowed_start_q is not None:
        allowed_start = np.asarray(allowed_start_q, dtype=np.float64).reshape(
            factr.num_arm_joints
        )
        allowed_start_tolerance = float(allowed_start_tolerance)
        if allowed_start_tolerance <= 0.0:
            raise ValueError("allowed_start_tolerance must be > 0")
    if validate_safety_limits is not None:
        try:
            factr.validate_arm_positions(
                start_state.joint_positions,
                safety_limits=validate_safety_limits,
            )
        except FactrSafetyError:
            if allowed_start is None:
                raise
            allowed_error = np.abs(
                _wrapped_joint_delta(start_state.joint_positions, allowed_start)
            )
            if np.any(allowed_error > allowed_start_tolerance):
                joint_index = int(np.argmax(allowed_error)) + 1
                raise FactrSafetyError(
                    "FACTR out-of-limit start pose does not match the explicitly "
                    f"allowed pose: J{joint_index} "
                    f"error={allowed_error[joint_index - 1]:.3f} rad > "
                    f"{allowed_start_tolerance:.3f} rad"
                )
            if not enforce_joint_limits:
                raise FactrSafetyError(
                    "FACTR out-of-limit start recovery requires joint-limit "
                    "direction enforcement"
                )
    start_q = start_state.joint_positions.copy()
    if gripper_target is None:
        gripper_target = float(start_state.gripper_position)
    trajectory_duration = max(0.0, float(trajectory_duration))

    controller = FactrPDController(
        factr,
        kp=None if kp is None else np.asarray(kp, dtype=np.float64),
        kd=None if kd is None else np.asarray(kd, dtype=np.float64),
        gripper_kp=gripper_kp,
        gripper_kd=gripper_kd,
        max_gripper_torque=max_gripper_torque,
        gripper_torque_sign=gripper_torque_sign,
        use_unclipped_gripper_position=use_unclipped_gripper_position,
        enforce_gripper_limits=enforce_gripper_limits,
        max_target_speed=max_target_speed,
        max_position_error=max_position_error,
        max_torque=max_torque,
        gravity_ramp_duration=gravity_ramp_duration,
        max_joint_velocity=max_joint_velocity,
        max_state_jump=max_state_jump,
        hold_gravity_compensation=hold_gravity_compensation,
        hold_friction_compensation=hold_friction_compensation,
        enforce_joint_limits=enforce_joint_limits,
        respect_config_torque_limit=respect_config_torque_limit,
        drive_torque=drive_torque,
        drive_deadband=drive_deadband,
        drive_ramp=drive_ramp,
    )

    reached = False
    started_at = time.monotonic()
    final_state = start_state
    next_progress = started_at
    cancelled = False
    try:
        controller.set_position(
            start_q if trajectory_duration > 0.0 else target,
            gripper=float(gripper_target),
        )
        controller.start()
        deadline = started_at + max(0.0, float(duration))
        while time.monotonic() < deadline:
            if cancel_event is not None and cancel_event.is_set():
                cancelled = True
                break
            controller.raise_if_failed()
            now = time.monotonic()
            if trajectory_duration > 0.0:
                alpha = min(1.0, max(0.0, (now - started_at) / trajectory_duration))
                alpha = alpha * alpha * (3.0 - 2.0 * alpha)
                target_delta = _wrapped_joint_delta(target, start_q)
                controller.set_position(
                    start_q + alpha * target_delta,
                    gripper=float(gripper_target),
                )
            state = factr.latest_state(max_age=0.5)
            if state is not None:
                final_state = state
                error = float(
                    np.max(np.abs(_wrapped_joint_delta(state.joint_positions, target)))
                )
                if progress_callback is not None and now >= next_progress:
                    progress_callback(now - started_at, state, target, controller)
                    next_progress = now + max(0.0, float(progress_period))
                if error <= tolerance:
                    reached = True
                    break
            time.sleep(max(0.001, float(poll_interval)))

        controller.raise_if_failed()
        cached_state = factr.latest_state(max_age=None)
        if cached_state is not None:
            final_state = cached_state
    finally:
        controller.stop(disable_torque=not (cancelled and keep_torque_on_cancel))

    max_error = float(
        np.max(np.abs(_wrapped_joint_delta(final_state.joint_positions, target)))
    )
    return FactrMoveToPoseResult(
        start_state=start_state,
        final_state=final_state,
        reached=reached,
        max_error=max_error,
        elapsed_s=time.monotonic() - started_at,
    )
