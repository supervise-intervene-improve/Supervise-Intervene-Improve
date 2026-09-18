#!/usr/bin/env python3
"""FACTR leader -> MuJoCo Franka follower teleoperation.

This script reuses the scene loading and MuJoCo Franka controller from
mq3_mc_mujoco.py, but follows FACTR's seven calibrated leader joints directly.
The handoff is relative/anchored so the first follower command cannot jump.
"""

from __future__ import annotations

import argparse
import os
import select
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator, Optional

import mujoco
import numpy as np

from factr import (
    DEFAULT_CONFIG_PATH,
    FACTRGravityCompensation,
    FactrError,
    FactrPDController,
    FactrSafetyError,
    FactrState,
)
from factr.validation import (
    validate_calibration_report,
    validate_gravity_report,
    validate_torque_direction_report,
)
from mq3_mc_mujoco import (
    ARM_GRAVITY_COMPENSATION,
    REPO_ROOT,
    SCENE_ALIASES,
    MujocoFrankaController,
    RepairedXmlLoader,
    resolve_scene_arg,
)


DEFAULT_RATE_HZ = 60.0
DEFAULT_MAX_JOINT_STEP_RAD = 0.04
DEFAULT_MAX_JOINT_VELOCITY_RAD_S = 2.4
DEFAULT_MAX_FACTR_VELOCITY_RAD_S = 56.0
DEFAULT_MAX_FACTR_JUMP_RAD = 10.75
DEFAULT_MAX_SIM_VELOCITY_RAD_S = 20.0
DEFAULT_STATE_TIMEOUT_S = 0.25
DEFAULT_ALIGN_TOLERANCE_RAD = 0.08
DEFAULT_ALIGN_TIMEOUT_S = 60.0
DEFAULT_INITIAL_POSE_TOLERANCE_RAD = 0.15
DEBUG_PERIOD_S = 2.0


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


class TeleoperationSafetyStop(RuntimeError):
    """Raised when teleoperation must stop without sending another command."""


class ViewerClosed(RuntimeError):
    """Internal signal for an operator-closing the MuJoCo viewer."""


def _joint_text(values: np.ndarray) -> str:
    return np.array2string(
        np.asarray(values, dtype=np.float64),
        precision=4,
        suppress_small=True,
        separator=", ",
    )


def _print_joint_state(
    label: str,
    positions: np.ndarray,
    velocities: Optional[np.ndarray] = None,
) -> None:
    print(f"[{label}] q  [rad]:   {_joint_text(positions)}")
    if velocities is not None:
        print(f"[{label}] qd [rad/s]: {_joint_text(velocities)}")


def _validate_cli(cfg: argparse.Namespace) -> None:
    positive_values = {
        "--rate": cfg.rate,
        "--max-joint-step": cfg.max_joint_step,
        "--max-joint-velocity": cfg.max_joint_velocity,
        "--max-factr-velocity": cfg.max_factr_velocity,
        "--max-factr-jump": cfg.max_factr_jump,
        "--max-sim-velocity": cfg.max_sim_velocity,
        "--sim-limit-tolerance": cfg.sim_limit_tolerance,
        "--state-timeout": cfg.state_timeout,
        "--align-tolerance": cfg.align_tolerance,
        "--initial-pose-tolerance": cfg.initial_pose_tolerance,
        "--factr-align-speed": cfg.factr_align_speed,
        "--factr-hold-error": cfg.factr_hold_error,
        "--factr-max-torque": cfg.factr_max_torque,
        "--factr-gravity-ramp-duration": cfg.factr_gravity_ramp_duration,
    }
    for option, value in positive_values.items():
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{option} must be positive; got {value!r}")
    if not np.isfinite(cfg.align_timeout) or cfg.align_timeout < 0.0:
        raise ValueError("--align-timeout must be zero (disabled) or positive")
    if cfg.max_factr_motion is not None and (
        not np.isfinite(cfg.max_factr_motion) or cfg.max_factr_motion <= 0.0
    ):
        raise ValueError("--max-factr-motion must be positive when supplied")
    active_mode_count = sum(
        bool(value)
        for value in (cfg.command_factr, cfg.gravity_comp, cfg.no_command_factr)
    )
    if active_mode_count > 1:
        raise ValueError(
            "--command-factr, --gravity-comp, and --no-command-factr are "
            "mutually exclusive"
        )
    if cfg.command_factr and cfg.manual_align:
        raise ValueError("--command-factr and --manual-align are mutually exclusive")
    if cfg.command_factr:
        raise ValueError(
            "--command-factr is disabled during FACTR commissioning. This device "
            "has no validated native position-command mode; use relative-anchor "
            "teleoperation instead."
        )
    if cfg.gravity_comp and cfg.manual_align:
        raise ValueError("--gravity-comp and --manual-align are mutually exclusive")
    if cfg.require_initial_pose and not cfg.gravity_comp:
        raise ValueError("--require-initial-pose is only valid with --gravity-comp")
    if (cfg.command_factr or cfg.gravity_comp) and cfg.factr_port is not None:
        raise ValueError(
            "Active FACTR modes forbid --factr-port overrides; use the validated "
            "device by-id name stored in the YAML."
        )
    if cfg.gripper_threshold is not None and not np.isfinite(cfg.gripper_threshold):
        raise ValueError("--gripper-threshold must be finite")
    if not np.isfinite(cfg.factr_gravity_scale) or not (
        0.0 < cfg.factr_gravity_scale <= 1.0
    ):
        raise ValueError("--factr-gravity-scale must be in (0, 1]")
    for option, value in {
        "--factr-nullspace-scale": cfg.factr_nullspace_scale,
        "--factr-friction-scale": cfg.factr_friction_scale,
    }.items():
        if not np.isfinite(value):
            raise ValueError(f"{option} must be finite; got {value!r}")


def _print_safety_settings(cfg: argparse.Namespace) -> None:
    motion_text = (
        "disabled"
        if cfg.max_factr_motion is None
        else f"{cfg.max_factr_motion:.3f} rad from the handoff anchor"
    )
    print("[SAFETY] Active teleoperation settings:")
    print(
        f"  rate={cfg.rate:.1f} Hz, max_joint_step={cfg.max_joint_step:.3f} rad, "
        f"max_joint_velocity={cfg.max_joint_velocity:.3f} rad/s"
    )
    print(
        f"  FACTR velocity stop={cfg.max_factr_velocity:.3f} rad/s, "
        f"state-jump stop={cfg.max_factr_jump:.3f} rad, "
        f"total-motion stop={motion_text}"
    )
    print(
        f"  MuJoCo velocity stop={cfg.max_sim_velocity:.3f} rad/s, "
        f"joint-limit tolerance={cfg.sim_limit_tolerance:.3f} rad"
    )
    print(
        "  mapping=direct 1:1 joint deltas, filtering=none; commands are "
        "limited only when a printed limit is exceeded"
    )
    if cfg.gravity_comp:
        print(
            "  FACTR teleop torque: "
            f"gravity_scale={cfg.factr_gravity_scale:.3f}, "
            f"gravity_ramp={cfg.factr_gravity_ramp_duration:.3f}s, "
            f"nullspace_scale={cfg.factr_nullspace_scale:.3f}, "
            f"friction_scale={cfg.factr_friction_scale:.3f}"
        )


def _prepare_validated_gravity_config(
    factr: FACTRGravityCompensation,
    *,
    gain_scale: float,
) -> None:
    """Apply commissioned YAML gravity settings under the controller torque cap."""
    unsafe_terms = []
    if factr.enable_torque_feedback:
        unsafe_terms.append("torque feedback")
    if factr.enable_gripper_feedback:
        unsafe_terms.append("gripper feedback")
    if unsafe_terms:
        raise FactrSafetyError(
            "Active MuJoCo gravity mode does not use follower/gripper feedback. "
            "Disable: " + ", ".join(unsafe_terms)
        )
    if not factr.enable_gravity_comp:
        raise FactrSafetyError(
            "controller.gravity_comp.enable must be true for --gravity-comp"
        )
    factr.gravity_comp_gain = factr.gravity_comp_gain * gain_scale
    factr.gravity_comp_modifier = factr.gravity_comp_gain
    print(
        "[FACTR] Using validated gravity scale "
        f"{gain_scale:.3f}; effective per-joint gains="
        f"{_joint_text(factr.gravity_comp_gain)}"
    )
    print(
        "[FACTR] Using leader.yaml joint-limit, null-space, and static-friction "
        "controller clauses under the configured FACTR torque cap."
    )
    if gain_scale < 1.0:
        print(
            "[FACTR] This is partial gravity assistance, not position hold. "
            "Continue supporting FACTR; it may sag when released."
        )


def _command_limited_target(
    controller: MujocoFrankaController,
    desired: np.ndarray,
    previous: np.ndarray,
    *,
    max_joint_step: float,
    max_joint_velocity: float,
    dt: float,
) -> tuple[np.ndarray, bool, bool]:
    desired = np.asarray(desired, dtype=np.float64).reshape(7)
    if not np.all(np.isfinite(desired)):
        raise TeleoperationSafetyStop("Non-finite MuJoCo joint target received.")

    limited_by_joint_range = controller.clip_joint_ranges(desired)
    range_limited = not np.allclose(limited_by_joint_range, desired, atol=1e-10)
    allowed_step = min(float(max_joint_step), float(max_joint_velocity) * float(dt))
    rate_limited = bool(
        np.any(np.abs(limited_by_joint_range - previous) > allowed_step + 1e-12)
    )
    delta = np.clip(
        limited_by_joint_range - previous,
        -allowed_step,
        allowed_step,
    )
    command = controller.clip_arm_ctrl(previous + delta)
    return command, range_limited, rate_limited


def _wrapped_joint_delta(current: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    delta = np.asarray(current, dtype=np.float64) - np.asarray(anchor, dtype=np.float64)
    return (delta + np.pi) % (2.0 * np.pi) - np.pi


def _factr_gripper_closed(
    cfg: argparse.Namespace,
    factr: FACTRGravityCompensation,
    gripper_position: float,
) -> bool:
    threshold = (
        factr.gripper_close_threshold
        if cfg.gripper_threshold is None
        else cfg.gripper_threshold
    )
    if cfg.gripper_close_above:
        return gripper_position >= threshold
    # Matches the supplied FACTR API: low -> -1 (Franka grasp), high -> +1 (open).
    return gripper_position < threshold


def _check_factr_state(
    cfg: argparse.Namespace,
    factr: FACTRGravityCompensation,
    state: FactrState,
    previous_source_q: np.ndarray | None,
    *,
    enforce_configured_limits: bool,
) -> None:
    if not np.all(np.isfinite(state.joint_positions)) or not np.all(
        np.isfinite(state.joint_velocities)
    ):
        raise TeleoperationSafetyStop("FACTR returned a non-finite joint state.")
    if not np.isfinite(state.gripper_position) or not np.isfinite(
        state.gripper_velocity
    ):
        raise TeleoperationSafetyStop("FACTR returned a non-finite gripper state.")

    # The copied FACTR calibration/limits are command-safety metadata.  An
    # out-of-range calibrated value must prevent torque activation, but it must
    # not prevent torque-disabled state polling: passive teleoperation uses only
    # motion relative to the Enter handoff and independently clips the follower.
    if enforce_configured_limits:
        factr.validate_gripper_position(state.gripper_position)
    max_velocity = float(np.max(np.abs(state.joint_velocities)))
    if max_velocity > cfg.max_factr_velocity:
        raise TeleoperationSafetyStop(
            f"FACTR velocity limit exceeded: {max_velocity:.3f} rad/s > "
            f"{cfg.max_factr_velocity:.3f} rad/s."
        )
    if previous_source_q is not None:
        state_delta = state.joint_positions - previous_source_q
        state_delta = (state_delta + np.pi) % (2.0 * np.pi) - np.pi
        abs_jump = np.abs(state_delta)
        max_jump = float(np.max(abs_jump))
        if max_jump > cfg.max_factr_jump:
            joint_index = int(np.argmax(abs_jump)) + 1
            raise TeleoperationSafetyStop(
                f"FACTR state jump detected: J{joint_index}={max_jump:.3f} rad > "
                f"{cfg.max_factr_jump:.3f} rad."
            )


def _check_mujoco_state(
    cfg: argparse.Namespace,
    controller: MujocoFrankaController,
) -> None:
    if not np.all(np.isfinite(controller.data.qpos)) or not np.all(
        np.isfinite(controller.data.qvel)
    ):
        raise TeleoperationSafetyStop("MuJoCo produced a non-finite state.")
    arm_q = controller.get_arm_qpos()
    arm_qd = np.asarray(
        controller.data.qvel[controller.dof_indices], dtype=np.float64
    )
    lower = controller.joint_ranges[:, 0] - cfg.sim_limit_tolerance
    upper = controller.joint_ranges[:, 1] + cfg.sim_limit_tolerance
    outside = np.flatnonzero((arm_q < lower) | (arm_q > upper))
    if outside.size:
        details = ", ".join(
            f"J{i + 1}={arm_q[i]:.3f}" for i in outside
        )
        raise TeleoperationSafetyStop(
            f"MuJoCo arm crossed a joint limit: {details}"
        )
    max_velocity = float(np.max(np.abs(arm_qd)))
    if max_velocity > cfg.max_sim_velocity:
        raise TeleoperationSafetyStop(
            f"MuJoCo arm velocity limit exceeded: {max_velocity:.3f} rad/s > "
            f"{cfg.max_sim_velocity:.3f} rad/s."
        )


def _step_waiting_sim(
    cfg: argparse.Namespace,
    controller: MujocoFrankaController,
    dt: float,
    viewer,
) -> None:
    controller.step_sim(dt)
    _check_mujoco_state(cfg, controller)
    if viewer is not None:
        if not viewer.is_running():
            raise ViewerClosed()
        viewer.sync()


def _latest_factr_state(
    factr: FACTRGravityCompensation,
    factr_controller: Optional[FactrPDController],
    max_age: float,
) -> FactrState:
    if factr_controller is None:
        return factr.read_state()
    factr_controller.raise_if_failed()
    state = factr.latest_state(max_age=max_age)
    if state is None:
        raise TeleoperationSafetyStop("FACTR controller has not produced a state yet.")
    return state


def _wait_for_first_state(
    cfg: argparse.Namespace,
    controller: MujocoFrankaController,
    factr: FACTRGravityCompensation,
    factr_controller: Optional[FactrPDController],
    viewer,
) -> FactrState:
    if factr_controller is None:
        return factr.read_state()
    deadline = time.monotonic() + max(2.0, cfg.state_timeout * 10.0)
    period = 1.0 / cfg.rate
    while time.monotonic() < deadline:
        factr_controller.raise_if_failed()
        state = factr.latest_state()
        if state is not None:
            return state
        _step_waiting_sim(cfg, controller, period, viewer)
        time.sleep(min(period, 0.01))
    raise TeleoperationSafetyStop("Timed out waiting for the FACTR control loop.")


def _wait_until_aligned(
    cfg: argparse.Namespace,
    controller: MujocoFrankaController,
    factr: FACTRGravityCompensation,
    factr_controller: Optional[FactrPDController],
    target_q: np.ndarray,
    viewer,
) -> FactrState:
    started = time.monotonic()
    last_report = 0.0
    stable_samples = 0
    required_stable_samples = max(3, int(round(0.25 * cfg.rate)))
    period = 1.0 / cfg.rate
    previous_q = None

    while True:
        now = time.monotonic()
        if cfg.align_timeout > 0.0 and now - started > cfg.align_timeout:
            raise TeleoperationSafetyStop(
                f"FACTR alignment timed out after {cfg.align_timeout:.1f}s. "
                "Check encoder calibration or retry with manual alignment."
            )

        state = _latest_factr_state(
            factr, factr_controller, max_age=cfg.state_timeout
        )
        _check_factr_state(
            cfg,
            factr,
            state,
            previous_q,
            enforce_configured_limits=factr_controller is not None,
        )
        previous_q = state.joint_positions.copy()

        error = target_q - state.joint_positions
        max_error = float(np.max(np.abs(error)))
        settled_velocity = float(np.max(np.abs(state.joint_velocities))) < 0.20
        if max_error <= cfg.align_tolerance and settled_velocity:
            stable_samples += 1
        else:
            stable_samples = 0

        if now - last_report >= 0.5:
            print(
                f"[ALIGN] max error={max_error:.3f} rad; "
                f"error={_joint_text(error)}"
            )
            last_report = now
        if stable_samples >= required_stable_samples:
            return state

        _step_waiting_sim(cfg, controller, period, viewer)
        elapsed = time.monotonic() - now
        if period - elapsed > 0.0:
            time.sleep(period - elapsed)


def _wait_for_start_confirmation(
    cfg: argparse.Namespace,
    controller: MujocoFrankaController,
    factr: FACTRGravityCompensation,
    factr_controller: Optional[FactrPDController],
    viewer,
) -> None:
    """Wait for Enter while continuing health checks, sim steps, and rendering."""
    if sys.stdin.isatty():
        # Discard an Enter pressed during alignment; confirmation must happen
        # after the aligned message is actually shown.
        try:
            import termios

            termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
        except (ImportError, OSError):
            pass
    print(
        "FACTR aligned with MuJoCo robot. Press Enter to start teleoperation.",
        flush=True,
    )
    period = 1.0 / cfg.rate
    previous_q = None
    while True:
        readable, _, _ = select.select([sys.stdin], [], [], 0.0)
        if readable:
            line = sys.stdin.readline()
            if line == "":
                raise TeleoperationSafetyStop(
                    "Interactive stdin closed before teleoperation was confirmed. "
                    "Use --smoke-test for a non-interactive check."
                )
            return

        state = _latest_factr_state(
            factr, factr_controller, max_age=cfg.state_timeout
        )
        _check_factr_state(
            cfg,
            factr,
            state,
            previous_q,
            enforce_configured_limits=factr_controller is not None,
        )
        previous_q = state.joint_positions.copy()
        started = time.monotonic()
        _step_waiting_sim(cfg, controller, period, viewer)
        remaining = period - (time.monotonic() - started)
        if remaining > 0.0:
            time.sleep(remaining)


def _check_configured_initial_pose(
    cfg: argparse.Namespace,
    factr: FACTRGravityCompensation,
    state: FactrState,
    *,
    required: bool,
) -> bool:
    """Compare FACTR with its own configured hand-placed startup pose."""
    reference = factr.initial_match_joint_pos
    error = state.joint_positions - reference
    max_error = float(np.max(np.abs(error)))
    max_velocity = float(np.max(np.abs(state.joint_velocities)))
    pose_ok = max_error <= cfg.initial_pose_tolerance
    still = max_velocity <= 0.20

    print(f"[INITIAL POSE] desired q [rad]:  {_joint_text(reference)}")
    print(f"[INITIAL POSE] measured q [rad]: {_joint_text(state.joint_positions)}")
    print(f"[INITIAL POSE] error [rad]:      {_joint_text(error)}")
    if pose_ok and still:
        print(
            f"[INITIAL POSE] PASS: max error={max_error:.3f} rad, "
            f"max velocity={max_velocity:.3f} rad/s."
        )
        return True

    reasons = []
    if not pose_ok:
        reasons.append(
            f"max error {max_error:.3f} rad exceeds "
            f"{cfg.initial_pose_tolerance:.3f} rad"
        )
    if not still:
        reasons.append(f"FACTR is still moving at {max_velocity:.3f} rad/s")
    message = "; ".join(reasons)
    if required:
        raise TeleoperationSafetyStop(
            f"Configured FACTR initial-pose check failed: {message}. Place "
            "FACTR in the pose stored as arm_teleop.initialization."
            "initial_match_joint_pos in factr/leader.yaml, hold it still, and rerun."
        )
    print(
        f"[INITIAL POSE] WARNING: {message}. Continuing because relative "
        "anchoring does not require an exact match; FACTR hard-limit checks "
        "remain enforced."
    )
    return False


def synchronize_factr(
    cfg: argparse.Namespace,
    controller: MujocoFrankaController,
    factr: FACTRGravityCompensation,
    factr_controller: Optional[FactrPDController],
    viewer,
) -> tuple[np.ndarray, FactrState]:
    sim_q = controller.get_arm_qpos()
    # Freeze the exact measured startup pose before alignment. This clears any
    # restored arm velocity and prevents a reset NPZ's stale ctrl target from
    # drifting the follower while the operator aligns FACTR.
    controller.set_arm_qpos(sim_q)
    controller.data.ctrl[:7] = controller.clip_arm_ctrl(sim_q)
    mujoco.mj_forward(controller.model, controller.data)
    initial_factr_state = factr.read_state()
    torque_control_requested = factr_controller is not None
    try:
        _check_factr_state(
            cfg,
            factr,
            initial_factr_state,
            previous_source_q=None,
            enforce_configured_limits=torque_control_requested,
        )
    except FactrSafetyError as exc:
        raise FactrSafetyError(
            f"{exc}. FACTR torque/hold was explicitly requested, so startup "
            "cannot continue with this calibration. Verify joint_offsets, "
            "gripper zero/range, and physical limits in factr/leader.yaml; or "
            "run without --command-factr for torque-disabled manual mode."
        ) from exc
    _print_joint_state("MuJoCo", sim_q)
    _print_joint_state(
        "FACTR",
        initial_factr_state.joint_positions,
        initial_factr_state.joint_velocities,
    )
    print(f"[FACTR] raw calibrated gripper: {initial_factr_state.gripper_position:.4f} rad")
    if not torque_control_requested:
        arm_limit_issue = None
        gripper_limit_issue = None
        try:
            factr.validate_arm_positions(
                initial_factr_state.joint_positions, safety_limits=False
            )
        except FactrSafetyError as exc:
            arm_limit_issue = str(exc)
        try:
            factr.validate_gripper_position(initial_factr_state.gripper_position)
        except FactrSafetyError as exc:
            gripper_limit_issue = str(exc)
        for issue in (arm_limit_issue, gripper_limit_issue):
            if issue is not None:
                print(f"[FACTR] CALIBRATION WARNING: {issue}")
        if gripper_limit_issue is not None and not cfg.no_gripper:
            cfg.no_gripper = True
            print(
                "[FACTR] MuJoCo gripper mirroring was disabled because the "
                "calibrated trigger is outside its configured range."
            )
    if not cfg.no_gripper:
        polarity = "above" if cfg.gripper_close_above else "below"
        threshold = (
            factr.gripper_close_threshold
            if cfg.gripper_threshold is None
            else cfg.gripper_threshold
        )
        print(
            f"[FACTR] MuJoCo gripper closes {polarity} "
            f"{threshold:.4f} rad."
        )

    if not cfg.command_factr and not cfg.manual_align:
        _check_configured_initial_pose(
            cfg,
            factr,
            initial_factr_state,
            required=cfg.require_initial_pose,
        )

    # In torque-disabled leader mode, absolute joint matching is unnecessary:
    # teleoperation is relative to the state sampled after Enter.  Requiring the
    # operator to reproduce all seven follower joints is difficult and provides
    # no additional no-jump safety.  Keep the old explicit joint-match workflow
    # behind --manual-align for calibration/debugging.
    if not torque_control_requested and not cfg.manual_align:
        print(
            "[ALIGN] Relative-anchor mode (default). FACTR does not need to "
            "match the MuJoCo joint values. Place it at a comfortable pose; "
            "MuJoCo will hold its current pose."
        )
        print(
            "[ALIGN] FACTR remains torque-disabled; the confirmation below "
            "will set its current pose as the zero-motion anchor."
        )
        _wait_for_start_confirmation(
            cfg, controller, factr, factr_controller, viewer
        )
        state_at_handoff = _latest_factr_state(
            factr, factr_controller, max_age=cfg.state_timeout
        )
        sim_anchor = controller.get_arm_ctrl_target()
        print(
            "[FACTR/MuJoCo] Teleoperation active. Move FACTR; press Ctrl+C to stop."
        )
        return sim_anchor, state_at_handoff

    if cfg.gravity_comp:
        soft_clearance = np.minimum(
            initial_factr_state.joint_positions - factr.arm_joint_limits_min,
            factr.arm_joint_limits_max - initial_factr_state.joint_positions,
        )
        if float(np.min(soft_clearance)) < 0.30:
            details = ", ".join(
                f"J{i + 1}={value:.3f}"
                for i, value in enumerate(soft_clearance)
                if value < 0.30
            )
            print(
                "[FACTR] LIMIT WARNING: FACTR is close to or beyond configured "
                f"software arm limits: {details}"
            )
        print(
            "[ALIGN] Relative-anchor + gravity-compensation mode. FACTR does "
            "not need to match the MuJoCo joint values, and no FACTR position "
            "target will be commanded."
        )
        print(
            "[FACTR] WARNING: motor torque will now be enabled. Hold FACTR "
            "lightly and press Ctrl+C immediately if any joint pushes strongly "
            "or moves unexpectedly."
        )
        factr_controller.set_position(None)
        factr_controller.start(env=None)
        _wait_for_first_state(cfg, controller, factr, factr_controller, viewer)
        print(
            "[ALIGN] FACTR gravity compensation is active; its current pose "
            "will become the zero-motion anchor after Enter."
        )
        _wait_for_start_confirmation(
            cfg, controller, factr, factr_controller, viewer
        )
        state_at_handoff = _latest_factr_state(
            factr, factr_controller, max_age=cfg.state_timeout
        )
        sim_anchor = controller.get_arm_ctrl_target()
        print(
            "[FACTR/MuJoCo] Teleoperation active. Move FACTR; press Ctrl+C to stop."
        )
        return sim_anchor, state_at_handoff

    # Refuse an alignment target that the leader cannot safely represent.
    factr.validate_arm_positions(sim_q, safety_limits=True)

    automatic_alignment = torque_control_requested and cfg.command_factr
    if automatic_alignment:
        print(
            "[ALIGN] EXPERIMENTAL automatic alignment selected. FACTR has no "
            "native position mode; a bounded gravity+PD torque ramp will move it."
        )
        factr_controller.set_position(sim_q)
    else:
        print(
            "[ALIGN] Manual alignment mode (--manual-align). Move FACTR until its seven "
            "joint values match the MuJoCo values above."
        )
        print(
            "[ALIGN] FACTR torque is disabled; no gravity compensation or hold "
            "torque will be sent."
        )

    if factr_controller is not None:
        factr_controller.start(env=None)
        _wait_for_first_state(cfg, controller, factr, factr_controller, viewer)
    aligned_state = _wait_until_aligned(
        cfg,
        controller,
        factr,
        factr_controller,
        sim_q,
        viewer,
    )

    if factr_controller is None:
        print(
            "[ALIGN] FACTR remains torque-disabled while waiting for Enter."
        )
    else:
        # The target is already within tolerance. Switching to bounded hold here
        # cannot request the large startup motion avoided by manual alignment.
        factr_controller.set_position(sim_q)
        print("[ALIGN] FACTR is holding the aligned pose with bounded gravity+PD torque.")

    current_sim_q = controller.get_arm_qpos()
    sim_drift = float(np.max(np.abs(current_sim_q - sim_q)))
    if sim_drift > cfg.align_tolerance:
        raise TeleoperationSafetyStop(
            f"MuJoCo drifted {sim_drift:.3f} rad during alignment; cannot "
            "confirm a synchronized handoff."
        )

    _wait_for_start_confirmation(
        cfg, controller, factr, factr_controller, viewer
    )

    # Re-anchor after Enter, not before it. This absorbs any small movement while
    # waiting and guarantees desired_q == sim_anchor on the first iteration.
    state_at_handoff = _latest_factr_state(
        factr, factr_controller, max_age=cfg.state_timeout
    )
    sim_anchor = controller.get_arm_ctrl_target()
    if factr_controller is not None:
        factr_controller.set_position(None)
    print("[FACTR/MuJoCo] Teleoperation active. Move FACTR; press Ctrl+C to stop.")
    return sim_anchor, state_at_handoff


def run_teleoperation_loop(
    cfg: argparse.Namespace,
    controller: MujocoFrankaController,
    factr: FACTRGravityCompensation,
    factr_controller: Optional[FactrPDController],
    sim_anchor: np.ndarray,
    factr_anchor_state: FactrState,
    viewer,
) -> None:
    period = 1.0 / cfg.rate
    factr_anchor = factr_anchor_state.joint_positions.copy()
    previous_source_q = factr_anchor.copy()
    previous_command = sim_anchor.copy()
    first_iteration = True
    last_debug = time.monotonic()
    loop_count = 0
    clip_notice_time = 0.0
    rate_notice_time = 0.0

    while viewer is None or viewer.is_running():
        loop_started = time.monotonic()
        state = _latest_factr_state(
            factr, factr_controller, max_age=cfg.state_timeout
        )
        _check_factr_state(
            cfg,
            factr,
            state,
            previous_source_q,
            enforce_configured_limits=factr_controller is not None,
        )
        previous_source_q = state.joint_positions.copy()
        if cfg.max_factr_motion is not None:
            total_source_motion = float(
                np.max(np.abs(_wrapped_joint_delta(state.joint_positions, factr_anchor)))
            )
            if total_source_motion > cfg.max_factr_motion:
                raise TeleoperationSafetyStop(
                    "FACTR motion envelope exceeded: "
                    f"{total_source_motion:.3f} rad > "
                    f"{cfg.max_factr_motion:.3f} rad."
                )

        # Always spend one complete loop holding the existing MuJoCo target.
        # The next sample begins relative tracking. This makes the zero-jump
        # guarantee independent of how quickly the operator moves after Enter.
        if first_iteration:
            desired = sim_anchor.copy()
        else:
            desired = sim_anchor + _wrapped_joint_delta(state.joint_positions, factr_anchor)
        command, range_limited, rate_limited = _command_limited_target(
            controller,
            desired,
            previous_command,
            max_joint_step=cfg.max_joint_step,
            max_joint_velocity=cfg.max_joint_velocity,
            dt=period,
        )
        if first_iteration and not np.allclose(command, sim_anchor, atol=1e-12):
            raise TeleoperationSafetyStop(
                "Zero-jump handoff invariant failed before the first MuJoCo command."
            )
        first_iteration = False

        controller.data.ctrl[:7] = command
        previous_command = command.copy()
        if not cfg.no_gripper:
            controller.set_gripper_pressed(
                _factr_gripper_closed(cfg, factr, state.gripper_position)
            )
        controller.step_sim(period)
        _check_mujoco_state(cfg, controller)

        if viewer is not None:
            viewer.sync()

        now = time.monotonic()
        if range_limited and now - clip_notice_time >= 1.0:
            print("[SAFETY] Desired follower pose clipped at a MuJoCo joint limit.")
            clip_notice_time = now
        if rate_limited and now - rate_notice_time >= 1.0:
            print(
                "[SAFETY] Desired follower motion exceeded the configured "
                "step/velocity limit; this command was rate-limited."
            )
            rate_notice_time = now
        loop_count += 1
        if now - last_debug >= DEBUG_PERIOD_S:
            elapsed = now - last_debug
            tracking_error = float(np.max(np.abs(desired - previous_command)))
            print(
                f"[FACTR/MuJoCo] {loop_count / elapsed:.1f} Hz; "
                f"max tracking ramp={tracking_error:.3f} rad; "
                f"q={_joint_text(previous_command)}"
            )
            loop_count = 0
            last_debug = now

        remaining = period - (time.monotonic() - loop_started)
        if remaining > 0.0:
            time.sleep(remaining)


def run_smoke_test(
    cfg: argparse.Namespace,
    controller: MujocoFrankaController,
) -> None:
    """Exercise anchoring, rate/limit checks, gripper, and simulation offline."""
    period = 1.0 / cfg.rate
    sim_anchor = controller.get_arm_ctrl_target()
    # The source deliberately has a different absolute zero. Relative anchoring
    # must still produce exactly sim_anchor on its first sample.
    factr_anchor = sim_anchor + np.array(
        [0.04, -0.03, 0.02, 0.03, -0.02, 0.01, -0.04], dtype=np.float64
    )
    source_motion = np.array(
        [0.12, -0.10, 0.08, -0.12, 0.09, -0.08, 0.10], dtype=np.float64
    )
    previous_command = sim_anchor.copy()
    maximum_step_seen = 0.0
    first_command = None

    for step_index in range(90):
        # Inject a source step after the anchored first frame. The follower must
        # ramp it instead of copying the discontinuity into actuator controls.
        source_q = (
            factr_anchor.copy()
            if step_index == 0
            else factr_anchor + source_motion
        )
        desired = sim_anchor + _wrapped_joint_delta(source_q, factr_anchor)
        command, _, _ = _command_limited_target(
            controller,
            desired,
            previous_command,
            max_joint_step=cfg.max_joint_step,
            max_joint_velocity=cfg.max_joint_velocity,
            dt=period,
        )
        if first_command is None:
            first_command = command.copy()
        maximum_step_seen = max(
            maximum_step_seen,
            float(np.max(np.abs(command - previous_command))),
        )
        controller.data.ctrl[:7] = command
        controller.set_gripper_pressed(step_index >= 45)
        controller.step_sim(period)
        _check_mujoco_state(cfg, controller)
        previous_command = command

    allowed_step = min(cfg.max_joint_step, cfg.max_joint_velocity * period)
    if first_command is None or not np.allclose(first_command, sim_anchor, atol=1e-12):
        raise RuntimeError("Smoke test failed the zero-jump handoff check.")
    if maximum_step_seen > allowed_step + 1e-12:
        raise RuntimeError(
            f"Smoke test exceeded joint step limit: {maximum_step_seen} > {allowed_step}"
        )
    if not np.all(np.isfinite(controller.data.qpos)) or not np.all(
        np.isfinite(controller.data.qvel)
    ):
        raise RuntimeError("Smoke test produced a non-finite MuJoCo state.")
    wrapped_delta = _wrapped_joint_delta(
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.06]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -6.22]),
    )
    if abs(wrapped_delta[6] + 0.003185307179586232) > 1e-9:
        raise RuntimeError(
            f"Smoke test failed FACTR J7 wrap mapping: {wrapped_delta[6]}"
        )

    class _RejectingFactrCalibration:
        @staticmethod
        def validate_arm_positions(joint_positions, *, safety_limits=True):
            del joint_positions, safety_limits
            raise FactrSafetyError("synthetic copied arm-limit mismatch")

        @staticmethod
        def validate_gripper_position(gripper_position):
            del gripper_position
            raise FactrSafetyError("synthetic copied gripper-limit mismatch")

    synthetic_state = FactrState(
        joint_positions=np.array(
            [-0.16, -1.93, -0.13, -3.20, -0.04, 1.25, 2.20],
            dtype=np.float64,
        ),
        joint_velocities=np.zeros(7, dtype=np.float64),
        gripper_position=-2.33,
        gripper_velocity=0.0,
        timestamp=time.monotonic(),
    )
    synthetic_factr = _RejectingFactrCalibration()
    _check_factr_state(
        cfg,
        synthetic_factr,  # type: ignore[arg-type]
        synthetic_state,
        previous_source_q=None,
        enforce_configured_limits=False,
    )
    try:
        _check_factr_state(
            cfg,
            synthetic_factr,  # type: ignore[arg-type]
            synthetic_state,
            previous_source_q=None,
            enforce_configured_limits=True,
        )
    except FactrSafetyError:
        pass
    else:
        raise RuntimeError(
            "Smoke test failed: active FACTR mode accepted an invalid calibration."
        )

    over_limit = controller.joint_ranges[:, 1] + 1.0
    clipped = controller.clip_joint_ranges(over_limit)
    if np.any(clipped > controller.joint_ranges[:, 1] + 1e-12):
        raise RuntimeError("Smoke test failed MuJoCo joint-range clipping.")

    print(f"[SMOKE] Scene control range check: PASS")
    print(f"[SMOKE] Zero-jump first command: PASS ({_joint_text(first_command)})")
    print(
        f"[SMOKE] Max command step: {maximum_step_seen:.6f} rad "
        f"(limit {allowed_step:.6f} rad)"
    )
    print(f"[SMOKE] Final arm ctrl: {_joint_text(controller.data.ctrl[:7])}")
    print("[SMOKE] Passive/active FACTR calibration gate: PASS")
    print("[SMOKE] PASS: no FACTR hardware was opened.")


@contextmanager
def _viewer_context(cfg: argparse.Namespace, controller: MujocoFrankaController) -> Iterator:
    if cfg.headless:
        yield None
        return
    import mujoco.viewer

    with mujoco.viewer.launch_passive(controller.model, controller.data) as viewer:
        viewer.cam.lookat[:] = np.array([0.55, 0.0, 0.45], dtype=np.float64)
        viewer.cam.distance = 1.45
        viewer.cam.azimuth = 180.0
        viewer.cam.elevation = -28.0
        viewer.sync()
        yield viewer


def _run_factr_test(cfg: argparse.Namespace) -> int:
    from test_factr_connection import run_connection_test

    return run_connection_test(
        SimpleNamespace(
            config=cfg.factr_config,
            port=cfg.factr_port,
            samples=3,
            rate=10.0,
            json=False,
        )
    )


def run(cfg: argparse.Namespace) -> int:
    _validate_cli(cfg)
    _print_safety_settings(cfg)
    if cfg.factr_test:
        return _run_factr_test(cfg)

    if cfg.command_factr or cfg.gravity_comp:
        validate_calibration_report(
            cfg.factr_config,
            cfg.factr_calibration_report,
        )
        validate_torque_direction_report(
            cfg.factr_config,
            cfg.factr_validation_report,
        )
    if cfg.gravity_comp:
        gravity_report = validate_gravity_report(
            cfg.factr_config,
            cfg.factr_gravity_report,
            gain_scale=cfg.factr_gravity_scale,
            max_torque_nm=cfg.factr_max_torque,
        )
        criteria = gravity_report["criteria"]
        cfg.max_factr_velocity = min(
            cfg.max_factr_velocity,
            float(criteria["max_peak_velocity_rad_s"]),
        )
        cfg.max_factr_jump = min(
            cfg.max_factr_jump,
            float(criteria["max_peak_displacement_rad"]),
        )
        print(
            "[FACTR] Active motion envelope capped by gravity validation: "
            f"velocity={cfg.max_factr_velocity:.3f} rad/s, "
            f"state jump={cfg.max_factr_jump:.3f} rad."
        )

    scene_path = resolve_scene_arg(
        cfg.scene_path if cfg.scene_path is not None else cfg.scene
    )
    print(f"[MuJoCo] Loading XML: {scene_path}")
    loader = RepairedXmlLoader(REPO_ROOT)
    factr = None
    factr_controller = None
    try:
        model, loaded_xml = loader.load(scene_path)
        data = mujoco.MjData(model)
        print(f"[MuJoCo] Loaded XML: {loaded_xml}")
        print(
            f"[MuJoCo] model nq={model.nq}, nv={model.nv}, nu={model.nu}, "
            f"timestep={model.opt.timestep:g}"
        )
        controller = MujocoFrankaController(
            model,
            data,
            site_name=cfg.site,
            reset_npz=cfg.reset_npz,
            arm_gravity_compensation=not cfg.no_arm_gravity_comp,
        )

        if cfg.smoke_test:
            run_smoke_test(cfg, controller)
            return 0

        print(f"[FACTR] Connecting with config: {Path(cfg.factr_config).resolve()}")
        factr = FACTRGravityCompensation(
            cfg.factr_config,
            port=cfg.factr_port,
            read_only=not (cfg.command_factr or cfg.gravity_comp),
        )
        if (
            cfg.gripper_threshold is not None
            and (
                cfg.gripper_threshold < factr.gripper_limit_min
                or cfg.gripper_threshold > factr.gripper_limit_max
            )
        ):
            raise ValueError(
                f"--gripper-threshold {cfg.gripper_threshold:.3f} is outside the "
                "configured FACTR gripper range "
                f"[{factr.gripper_limit_min:.3f}, {factr.gripper_limit_max:.3f}] rad"
            )
        if cfg.command_factr or cfg.gravity_comp:
            if cfg.gravity_comp:
                _prepare_validated_gravity_config(
                    factr,
                    gain_scale=cfg.factr_gravity_scale,
                )
            factr_controller = FactrPDController(
                factr,
                max_target_speed=cfg.factr_align_speed,
                max_position_error=cfg.factr_hold_error,
                max_torque=cfg.factr_max_torque,
                gravity_ramp_duration=cfg.factr_gravity_ramp_duration,
                max_joint_velocity=cfg.max_factr_velocity,
                max_state_jump=cfg.max_factr_jump,
                leader_nullspace_scale=cfg.factr_nullspace_scale,
                leader_gravity_scale=1.0,
                leader_friction_scale=cfg.factr_friction_scale,
            )
        else:
            print(
                "[FACTR] Torque-disabled polling mode (default). "
                "Use --gravity-comp or --command-factr only after checking "
                "the absolute calibration and physical limits."
            )

        with _viewer_context(cfg, controller) as viewer:
            sim_anchor, factr_anchor_state = synchronize_factr(
                cfg,
                controller,
                factr,
                factr_controller,
                viewer,
            )
            run_teleoperation_loop(
                cfg,
                controller,
                factr,
                factr_controller,
                sim_anchor,
                factr_anchor_state,
                viewer,
            )
    except ViewerClosed:
        print("[FACTR/MuJoCo] Viewer closed; stopping safely.")
    finally:
        if factr_controller is not None:
            print("[FACTR/MuJoCo] Stopping FACTR controller...")
            try:
                factr_controller.stop(disable_torque=True)
            except Exception as exc:
                print(f"[FACTR/MuJoCo] WARNING during controller stop: {exc}")
        try:
            if factr is not None:
                print("[FACTR/MuJoCo] Closing FACTR...")
                factr.shutdown()
        finally:
            loader.close()
    print("[FACTR/MuJoCo] Shutdown complete.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="FACTR leader -> MuJoCo Franka joint teleoperation"
    )
    parser.add_argument(
        "--scene",
        default="t_shape",
        help=(
            "Scene alias or XML path. Aliases: "
            + ", ".join(sorted(SCENE_ALIASES.keys()))
        ),
    )
    parser.add_argument(
        "--scene-path",
        default=None,
        help="Explicit MuJoCo XML path; overrides --scene",
    )
    parser.add_argument("--site", default="gripper")
    parser.add_argument("--reset-npz", default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--smoke-test",
        "--test",
        "--dry-run",
        dest="smoke_test",
        action="store_true",
        help="Run an offline anchored/rate-limited MuJoCo test without FACTR",
    )
    parser.add_argument(
        "--factr-test",
        action="store_true",
        help="Run the passive FACTR connection test and exit (does not load MuJoCo)",
    )
    parser.add_argument(
        "--factr-config",
        "--config",
        dest="factr_config",
        default=str(DEFAULT_CONFIG_PATH),
        help="FACTR YAML configuration",
    )
    parser.add_argument(
        "--factr-port",
        default=None,
        help="Optional serial path or by-id name override",
    )
    parser.add_argument(
        "--factr-calibration-report",
        default="factr/validation/calibration.json",
        help="Required passive calibration PASS report for active FACTR modes",
    )
    parser.add_argument(
        "--factr-validation-report",
        default="factr/validation/torque_direction.json",
        help=(
            "Required J1..J7 torque-direction report for active FACTR modes"
        ),
    )
    parser.add_argument(
        "--factr-gravity-report",
        default="factr/validation/gravity.json",
        help="Required three-pose gravity PASS report for --gravity-comp",
    )
    alignment = parser.add_mutually_exclusive_group()
    alignment.add_argument(
        "--manual-align",
        dest="manual_align",
        action="store_true",
        default=True,
        help=(
            "Manually match FACTR joints to MuJoCo before start (default)"
        ),
    )
    alignment.add_argument(
        "--relative-anchor",
        dest="manual_align",
        action="store_false",
        help=(
            "Skip absolute matching and map motion relative to the Enter "
            "handoff; useful while encoder calibration is still being commissioned"
        ),
    )
    parser.add_argument(
        "--command-factr",
        action="store_true",
        help=(
            "EXPERIMENTAL: move FACTR toward MuJoCo with bounded torque-PD; "
            "FACTR has no native position-command mode"
        ),
    )
    parser.add_argument(
        "--gravity-comp",
        action="store_true",
        help=(
            "Enable bounded FACTR gravity compensation at its current pose; "
            "uses relative anchoring and never commands a FACTR position target"
        ),
    )
    parser.add_argument(
        "--no-command-factr",
        action="store_true",
        help=(
            "Never enable FACTR gravity-compensation/hold torque "
            "(this is already the safe default)"
        ),
    )
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE_HZ)
    parser.add_argument(
        "--max-joint-step",
        type=float,
        default=DEFAULT_MAX_JOINT_STEP_RAD,
        help="Maximum MuJoCo actuator-target change per loop, radians",
    )
    parser.add_argument(
        "--max-joint-velocity",
        type=float,
        default=DEFAULT_MAX_JOINT_VELOCITY_RAD_S,
        help="Maximum MuJoCo target velocity, rad/s",
    )
    parser.add_argument(
        "--max-factr-velocity",
        type=float,
        default=DEFAULT_MAX_FACTR_VELOCITY_RAD_S,
        help="Emergency-stop threshold for measured FACTR velocity, rad/s",
    )
    parser.add_argument(
        "--max-factr-jump",
        type=float,
        default=DEFAULT_MAX_FACTR_JUMP_RAD,
        help="Emergency-stop threshold for one FACTR state jump, radians",
    )
    parser.add_argument(
        "--max-factr-motion",
        type=float,
        default=None,
        help=(
            "Optional maximum per-joint FACTR displacement from the handoff "
            "anchor; omitted by default"
        ),
    )
    parser.add_argument(
        "--max-sim-velocity",
        type=float,
        default=DEFAULT_MAX_SIM_VELOCITY_RAD_S,
        help="Emergency-stop threshold for measured MuJoCo arm velocity, rad/s",
    )
    parser.add_argument(
        "--sim-limit-tolerance",
        type=float,
        default=0.02,
        help="Allowed numerical tolerance beyond MuJoCo joint limits, radians",
    )
    parser.add_argument(
        "--state-timeout",
        type=float,
        default=DEFAULT_STATE_TIMEOUT_S,
        help="Emergency stop when cached FACTR state is older than this many seconds",
    )
    parser.add_argument(
        "--align-tolerance",
        type=float,
        default=DEFAULT_ALIGN_TOLERANCE_RAD,
        help="Per-joint FACTR/MuJoCo alignment tolerance, radians",
    )
    parser.add_argument(
        "--initial-pose-tolerance",
        type=float,
        default=DEFAULT_INITIAL_POSE_TOLERANCE_RAD,
        help=(
            "Allowed error from FACTR's configured initial_match_joint_pos "
            "before gravity compensation starts, radians"
        ),
    )
    parser.add_argument(
        "--require-initial-pose",
        action="store_true",
        help=(
            "Require the configured initial-pose check to pass before "
            "gravity compensation; default behavior only warns"
        ),
    )
    parser.add_argument(
        "--align-timeout",
        type=float,
        default=DEFAULT_ALIGN_TIMEOUT_S,
        help="Alignment timeout in seconds; zero disables it",
    )
    parser.add_argument(
        "--factr-align-speed",
        type=float,
        default=0.25,
        help="Experimental PD alignment reference speed limit, rad/s",
    )
    parser.add_argument(
        "--factr-hold-error",
        type=float,
        default=0.08,
        help="Maximum FACTR PD position error used for torque, radians",
    )
    parser.add_argument(
        "--factr-max-torque",
        type=float,
        default=15.0,
        help="FACTR torque ceiling; must exactly match the gravity PASS report",
    )
    parser.add_argument(
        "--factr-gravity-scale",
        type=float,
        default=_env_float("FACTR_TAKEOVER_GRAVITY_SCALE", 0.20),
        help="Scale applied to configured gravity gains; must match PASS report",
    )
    parser.add_argument(
        "--factr-gravity-ramp-duration",
        type=float,
        default=_env_float("FACTR_TAKEOVER_GRAVITY_RAMP_DURATION", 1.0),
        help="Seconds to ramp FACTR gravity/friction torque in standalone teleop.",
    )
    parser.add_argument(
        "--factr-nullspace-scale",
        type=float,
        default=_env_float("FACTR_TAKEOVER_NULLSPACE_SCALE", 0.0),
        help="Scale for FACTR null-space regulation during standalone teleop.",
    )
    parser.add_argument(
        "--factr-friction-scale",
        type=float,
        default=_env_float("FACTR_TAKEOVER_FRICTION_SCALE", 1.0),
        help="Scale for FACTR static-friction compensation during standalone teleop.",
    )
    parser.add_argument(
        "--no-gripper",
        action="store_true",
        help="Do not mirror the FACTR trigger to the MuJoCo gripper",
    )
    parser.add_argument(
        "--gripper-threshold",
        type=float,
        default=None,
        help="FACTR trigger threshold in radians (default: half configured range)",
    )
    parser.add_argument(
        "--gripper-close-above",
        action="store_true",
        help=(
            "Close above the trigger threshold instead of the copied FACTR "
            "convention (low=-1/close, high=+1/open)"
        ),
    )
    parser.add_argument(
        "--no-arm-gravity-comp",
        action="store_true",
        default=not ARM_GRAVITY_COMPENSATION,
        help="Disable MuJoCo Panda qfrc_bias gravity compensation",
    )
    return parser.parse_args()


def main() -> int:
    cfg = parse_args()
    try:
        return run(cfg)
    except KeyboardInterrupt:
        print("\n[FACTR/MuJoCo] Ctrl+C received: emergency stop and clean shutdown.")
        return 130
    except (FactrError, TeleoperationSafetyStop, OSError, ValueError) as exc:
        print(f"[FACTR/MuJoCo] STOPPED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
