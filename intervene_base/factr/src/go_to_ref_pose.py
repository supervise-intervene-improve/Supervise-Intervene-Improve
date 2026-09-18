#!/usr/bin/env python3
"""Move the MuJoCo Panda to a reference joint pose."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mq3_mc_mujoco import (  # noqa: E402
    DEFAULT_ARM_Q,
    DT,
    MujocoFrankaController,
    RepairedXmlLoader,
    resolve_scene_arg,
)
from factr import FACTRGravityCompensation, move_factr_to_joint_pose  # noqa: E402
from factr.validation import (  # noqa: E402
    validate_calibration_report,
    validate_torque_direction_report,
)


KNOWN_FACTR_REFERENCE_Q = np.array(
    [0.0, -0.7854, 0.0, -2.356, 0.0, 1.57, 0.0],
    dtype=np.float64,
)


# GO_TO_FACTR_KP = np.array([0.75, 2.0, 1.0, 2.25, 0.5, 0.35, 0.25], dtype=np.float64)
# GO_TO_FACTR_KD = np.array([0.01, 0.005, 0.01, 0.01, 0.01, 0.005, 0.01], dtype=np.float64)

GO_TO_FACTR_KP = np.array([0.75, 3.2, 1.0, 3.5, 0.4, 0.15, 0.15], dtype=np.float64)
GO_TO_FACTR_KD = np.array([0.01, 0.005, 0.01, 0.01, 0.01, 0.03, 0.01], dtype=np.float64)

# GO_TO_FACTR_KP = np.array([0.75, 3.0, 1.0, 3.5, 0.4, 0.15, 0.15], dtype=np.float64)
# GO_TO_FACTR_KD = np.array([0.01, 0.005, 0.01, 0.01, 0.01, 0.03, 0.01], dtype=np.float64)


def _parse_joint_list(text: str) -> np.ndarray:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if len(values) != 7:
        raise argparse.ArgumentTypeError(
            f"expected 7 comma-separated joint values, got {len(values)}"
        )
    return np.asarray(values, dtype=np.float64)


def _parse_scalar_or_joint_list(text: str) -> float | np.ndarray:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if len(values) == 1:
        return values[0]
    if len(values) != 7:
        raise argparse.ArgumentTypeError(
            f"expected 1 value or 7 comma-separated joint values, got {len(values)}"
        )
    return np.asarray(values, dtype=np.float64)


def _format_vector(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(value): .3f}" for value in values) + "]"


def _format_scalar_or_vector(value: float | np.ndarray) -> str:
    array = np.asarray(value, dtype=np.float64)
    if array.shape == ():
        return f"{float(array):.3f}"
    return _format_vector(array.reshape(7))


def _wrapped_joint_delta(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    delta = np.asarray(current, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    return (delta + np.pi) % (2.0 * np.pi) - np.pi


def _load_yaml_pose(config_path: Path, field: str) -> np.ndarray:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    pose = payload["arm_teleop"]["initialization"][field]
    return np.asarray(pose[:7], dtype=np.float64)


def _load_pose_payload(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_pose_file(path: Path) -> np.ndarray:
    payload = _load_pose_payload(path)
    if isinstance(payload, list):
        pose = payload
    elif isinstance(payload, dict):
        pose = payload.get("joint_positions", payload.get("arm_joints"))
    else:
        pose = None
    if pose is None:
        raise ValueError(
            f"{path} must contain a list or a dict with joint_positions"
        )
    q = np.asarray(pose, dtype=np.float64)
    if q.shape != (7,):
        raise ValueError(f"{path} must contain exactly 7 joint values")
    return q


def _gripper_target_from_args(args: argparse.Namespace) -> float | None:
    if args.factr_gripper_target is not None:
        return float(args.factr_gripper_target)
    if args.pose_file is None:
        return None
    payload = _load_pose_payload(Path(args.pose_file))
    if isinstance(payload, dict) and "gripper_position" in payload:
        return float(payload["gripper_position"])
    if isinstance(payload, dict) and "gripper_motor_position" in payload:
        return float(payload["gripper_motor_position"])
    return None


def _gripper_limits_from_config(config_path: Path) -> tuple[float, float]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return 0.0, float(payload["gripper_teleop"]["actuation_range"])


def _clamp_gripper_target_for_config(
    gripper_target: float | None,
    config_path: Path,
) -> float | None:
    if gripper_target is None:
        return None
    lower, upper = _gripper_limits_from_config(config_path)
    clamped = float(np.clip(float(gripper_target), lower, upper))
    if abs(clamped - float(gripper_target)) > 1e-9:
        print(
            "[FACTR] Gripper target clipped to configured physical range: "
            f"{float(gripper_target):.6f} -> {clamped:.6f} rad "
            f"[{lower:.3f}, {upper:.3f}]"
        )
    return clamped


def _target_from_args(args: argparse.Namespace) -> np.ndarray:
    if args.target_joints is not None:
        return args.target_joints
    if args.pose_file is not None:
        return _load_pose_file(Path(args.pose_file))
    if args.reference == "known-factr":
        return KNOWN_FACTR_REFERENCE_Q.copy()
    if args.reference == "default-mujoco":
        return DEFAULT_ARM_Q.copy()
    return _load_yaml_pose(Path(args.config), args.yaml_pose)


def _step_to_target(
    controller: MujocoFrankaController,
    target_q: np.ndarray,
    *,
    duration: float,
    rate: float,
    viewer=None,
) -> None:
    start_q = controller.get_arm_qpos()
    duration = max(0.0, float(duration))
    period = 1.0 / max(1e-6, float(rate))
    start_time = time.monotonic()

    while True:
        elapsed = time.monotonic() - start_time
        alpha = 1.0 if duration <= 0.0 else min(1.0, elapsed / duration)
        q_cmd = start_q + alpha * (target_q - start_q)
        controller.data.ctrl[:7] = controller.clip_arm_ctrl(q_cmd)
        controller.step_sim(period)
        if viewer is not None:
            viewer.sync()
        if alpha >= 1.0:
            break
        sleep_s = period - (time.monotonic() - start_time - elapsed)
        if sleep_s > 0.0:
            time.sleep(sleep_s)

    controller.data.ctrl[:7] = controller.clip_arm_ctrl(target_q)
    controller.step_sim(0.25)
    if viewer is not None:
        viewer.sync()


def _command_factr_to_target(
    args: argparse.Namespace,
    target_q: np.ndarray,
    gripper_target: float | None,
) -> None:
    validate_calibration_report(args.factr_config, args.factr_calibration_report)
    validate_torque_direction_report(args.factr_config, args.factr_validation_report)

    print("[FACTR] Active bounded reference-pose command requested.")
    print(f"[FACTR] Target joints [rad]: {target_q.tolist()}")
    if gripper_target is not None and not args.no_factr_hold_gripper:
        print(f"[FACTR] Target gripper [rad]: {gripper_target:.6f}")
        print(
            f"[FACTR] Gripper hold kp={args.factr_gripper_kp:.3f}, "
            f"kd={args.factr_gripper_kd:.3f}, "
            f"max_torque={args.factr_max_gripper_torque:.3f} Nm, "
            f"sign={args.factr_gripper_torque_sign:+.0f}"
        )
    print(
        f"[FACTR] duration={args.factr_duration:.2f}s, "
        f"max_torque={args.factr_max_torque:.3f} Nm, "
        f"max_target_speed={args.factr_align_speed:.3f} rad/s"
    )
    if args.factr_trajectory_duration > 0.0:
        print(
            "[FACTR] target trajectory: "
            f"{args.factr_trajectory_duration:.2f}s smooth start-to-goal ramp"
        )
    if np.any(np.asarray(args.factr_drive_torque, dtype=np.float64) > 0.0):
        print(
            "[FACTR] far-error drive torque="
            f"{_format_scalar_or_vector(args.factr_drive_torque)} Nm "
            f"after {args.factr_drive_deadband:.3f} rad"
        )
    if args.factr_allow_over_config_torque:
        print("[FACTR] Config torque cap override: enabled")
    print(f"[FACTR] PD kp={_format_vector(args.factr_kp)}")
    print(f"[FACTR] PD kd={_format_vector(args.factr_kd)}")
    print("[FACTR] Keep a hand on FACTR and be ready to cut power.")
    input("Press Enter to enable bounded FACTR torque toward the reference pose...")

    factr = FACTRGravityCompensation(args.factr_config, read_only=False)
    try:
        if gripper_target is not None and not args.no_factr_hold_gripper:
            factr.validate_gripper_position(gripper_target)
        print(f"[FACTR] Torque enabled: {bool(factr.driver and factr.driver.torque_enabled)}")
        start_state = factr.read_state()
        validate_safety_limits = None
        if args.factr_validate_arm_limits:
            validate_safety_limits = not args.factr_ignore_soft_limits
            factr.validate_arm_positions(
                target_q,
                safety_limits=validate_safety_limits,
            )
            factr.validate_arm_positions(
                start_state.joint_positions,
                safety_limits=validate_safety_limits,
            )
        else:
            print("[FACTR] Arm limit validation: skipped for reference-pose move.")
        print(f"[FACTR] Start joints [rad]: {start_state.joint_positions.tolist()}")
        print(
            "[FACTR] Start max error [rad]: "
            f"{float(np.max(np.abs(_wrapped_joint_delta(start_state.joint_positions, target_q)))):.6f}"
        )

        def on_progress(elapsed_s, state, target, controller):
            del elapsed_s
            joint_error = _wrapped_joint_delta(state.joint_positions, target)
            error = float(np.max(np.abs(joint_error)))
            filtered = controller.latest_filtered_target
            filtered_error = (
                float(np.max(np.abs(_wrapped_joint_delta(filtered, state.joint_positions))))
                if filtered is not None
                else 0.0
            )
            max_tau = float(np.max(np.abs(controller.latest_arm_torque)))
            print(
                f"[FACTR] max error={error:.6f} rad, "
                f"ramped-target error={filtered_error:.6f} rad, "
                f"max torque cmd={max_tau:.6f} Nm"
            )
            print(
                "[FACTR] error per joint [rad]: "
                f"{_format_vector(joint_error)}"
            )
            print(
                "[FACTR] torque per joint [Nm]: "
                f"{_format_vector(controller.latest_arm_torque)}"
            )
            if gripper_target is not None and not args.no_factr_hold_gripper:
                print(
                    "[FACTR] gripper pos/target-error/torque: "
                    f"{state.gripper_position:.6f} rad, "
                    f"{gripper_target - state.gripper_position:.6f} rad, "
                    f"{controller.latest_gripper_torque:.6f} Nm"
                )

        result = move_factr_to_joint_pose(
            factr,
            target_q,
            kp=args.factr_kp,
            kd=args.factr_kd,
            duration=args.factr_duration,
            tolerance=args.factr_tolerance,
            max_target_speed=args.factr_align_speed,
            max_position_error=args.factr_hold_error,
            max_torque=args.factr_max_torque,
            max_gripper_torque=args.factr_max_gripper_torque,
            gripper_target=None if args.no_factr_hold_gripper else gripper_target,
            gripper_kp=args.factr_gripper_kp,
            gripper_kd=args.factr_gripper_kd,
            gripper_torque_sign=args.factr_gripper_torque_sign,
            use_unclipped_gripper_position=True,
            enforce_gripper_limits=True,
            gravity_ramp_duration=args.factr_ramp_duration,
            max_joint_velocity=args.max_factr_velocity,
            max_state_jump=args.max_factr_jump,
            hold_gravity_compensation=not args.no_factr_hold_gravity_comp,
            hold_friction_compensation=not args.no_factr_hold_friction_comp,
            validate_safety_limits=validate_safety_limits,
            enforce_joint_limits=not args.factr_disable_limit_torque,
            respect_config_torque_limit=not args.factr_allow_over_config_torque,
            drive_torque=args.factr_drive_torque,
            drive_deadband=args.factr_drive_deadband,
            drive_ramp=args.factr_drive_ramp,
            trajectory_duration=args.factr_trajectory_duration,
            progress_callback=on_progress,
        )
        if result.reached:
            print(f"[FACTR] Reached tolerance: {result.max_error:.6f} rad")
        print(f"[FACTR] Final joints [rad]: {result.final_state.joint_positions.tolist()}")
        print(
            "[FACTR] Final max error [rad]: "
            f"{result.max_error:.6f}"
        )
    finally:
        factr.shutdown()


def _monitor_factr_reference(
    args: argparse.Namespace,
    target_q: np.ndarray,
    gripper_target: float | None,
) -> None:
    validate_calibration_report(args.factr_config, args.factr_calibration_report)

    period = 1.0 / max(1e-6, float(args.factr_monitor_rate))
    duration = max(0.0, float(args.factr_monitor_duration))
    deadline = None if duration <= 0.0 else time.monotonic() + duration

    print("[FACTR] Read-only reference monitor; no torque will be enabled.")
    print(f"[FACTR] Target joints [rad]: {target_q.tolist()}")
    if gripper_target is not None:
        print(f"[FACTR] Target gripper [rad]: {gripper_target:.6f}")
    print(
        "[FACTR] Move FACTR by hand until max error is below "
        f"{args.factr_tolerance:.3f} rad. Press Ctrl+C to stop."
    )

    factr = FACTRGravityCompensation(args.factr_config, read_only=True)
    try:
        while True:
            state = factr.read_state()
            error = state.joint_positions - target_q
            max_error = float(np.max(np.abs(error)))
            print(f"[FACTR] Current joints [rad]: {state.joint_positions.tolist()}")
            print(f"[FACTR] Error per joint [rad]: {_format_vector(error)}")
            if gripper_target is not None:
                print(
                    "[FACTR] Gripper target error [rad]: "
                    f"{gripper_target - state.gripper_position:.6f}"
                )
            print(f"[FACTR] Max error [rad]: {max_error:.6f}")
            if max_error <= args.factr_tolerance:
                print("[FACTR] Reference pose reached.")
                return
            if deadline is not None and time.monotonic() >= deadline:
                print("[FACTR] Monitor duration ended before tolerance was reached.")
                return
            time.sleep(period)
    except KeyboardInterrupt:
        print("\n[FACTR] Reference monitor stopped.")
    finally:
        factr.shutdown()


def _save_factr_pose(args: argparse.Namespace, pose_path: Path) -> None:
    validate_calibration_report(args.factr_config, args.factr_calibration_report)

    print("[FACTR] Saving current FACTR pose in read-only mode; no torque will be enabled.")
    factr = FACTRGravityCompensation(args.factr_config, read_only=True)
    try:
        state = factr.read_state()
    finally:
        factr.shutdown()

    pose_path.parent.mkdir(parents=True, exist_ok=True)
    gripper_motor_position = float(
        getattr(factr, "gripper_pos_unclipped", state.gripper_position)
    )
    payload = {
        "joint_positions": state.joint_positions.tolist(),
        "joint_velocities": state.joint_velocities.tolist(),
        "gripper_position": float(state.gripper_position),
        "gripper_motor_position": gripper_motor_position,
        "gripper_velocity": float(state.gripper_velocity),
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "factr_config": str(args.factr_config),
    }
    pose_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[FACTR] Saved pose: {pose_path}")
    print(f"[FACTR] Joint positions [rad]: {state.joint_positions.tolist()}")
    print(f"[FACTR] Gripper motor position [rad]: {gripper_motor_position:.6f}")


def run(args: argparse.Namespace) -> int:
    if args.save_factr_pose is not None:
        _save_factr_pose(args, Path(args.save_factr_pose))
        return 0

    scene_path = resolve_scene_arg(args.scene)
    factr_target_q = _target_from_args(args)
    gripper_target = _clamp_gripper_target_for_config(
        _gripper_target_from_args(args),
        Path(args.factr_config),
    )

    print(f"[REF] Target joints [rad]: {factr_target_q.tolist()}")
    if gripper_target is not None:
        print(f"[REF] Target gripper [rad]: {gripper_target:.6f}")
    if args.print_target_only:
        return 0
    if args.factr_only:
        if args.command_factr:
            _command_factr_to_target(args, factr_target_q, gripper_target)
        else:
            _monitor_factr_reference(args, factr_target_q, gripper_target)
        return 0

    loader = RepairedXmlLoader(REPO_ROOT)
    try:
        print(f"[MuJoCo] Loading XML: {scene_path}")
        model, loaded_xml = loader.load(scene_path)
        data = mujoco.MjData(model)
        print(f"[MuJoCo] Loaded XML: {loaded_xml}")

        controller = MujocoFrankaController(
            model,
            data,
            site_name=args.site,
            initial_q=DEFAULT_ARM_Q,
            arm_gravity_compensation=not args.no_arm_gravity_comp,
        )
        target_q = controller.clip_joint_ranges(factr_target_q)
        if float(np.max(np.abs(target_q - factr_target_q))) > 1e-9:
            print(
                "[MuJoCo] Target was clipped for MuJoCo joint ranges; "
                "FACTR will still use the unclipped target."
            )

        def drive(viewer=None):
            _step_to_target(
                controller,
                target_q,
                duration=args.duration,
                rate=args.rate,
                viewer=viewer,
            )
            final_q = controller.get_arm_qpos()
            error = final_q - target_q
            pos, quat = controller.get_ee_pose()
            print(f"[REF] Final joints [rad]: {final_q.tolist()}")
            print(f"[REF] Max joint error [rad]: {float(np.max(np.abs(error))):.6f}")
            print(f"[REF] EE position [m]: {pos.tolist()}")
            print(f"[REF] EE quat xyzw: {quat.tolist()}")
            if args.mujoco_only:
                return
            if args.command_factr:
                _command_factr_to_target(args, factr_target_q, gripper_target)
            else:
                _monitor_factr_reference(args, factr_target_q, gripper_target)

        if args.headless:
            drive()
            return 0

        from mujoco import viewer as mujoco_viewer

        with mujoco_viewer.launch_passive(model, data) as viewer:
            drive(viewer)
            print("[REF] Viewer open. Press Ctrl+C in this terminal to exit.")
            while viewer.is_running():
                controller.data.ctrl[:7] = controller.clip_arm_ctrl(target_q)
                controller.step_sim(DT)
                viewer.sync()
                time.sleep(DT)
    finally:
        loader.close()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="t_shape")
    parser.add_argument("--site", default="gripper")
    parser.add_argument("--config", default="factr/leader.yaml")
    parser.add_argument(
        "--reference",
        choices=("leader-yaml", "known-factr", "default-mujoco"),
        default="leader-yaml",
        help="Named reference pose to use when --target-joints is omitted.",
    )
    parser.add_argument(
        "--yaml-pose",
        choices=("initial_match_joint_pos", "calibration_joint_pos"),
        default="initial_match_joint_pos",
        help="leader.yaml initialization field used by --reference leader-yaml.",
    )
    parser.add_argument(
        "--target-joints",
        type=_parse_joint_list,
        default=None,
        help="Seven comma-separated Panda joint targets in radians.",
    )
    parser.add_argument(
        "--pose-file",
        default=None,
        help="JSON pose saved with --save-factr-pose.",
    )
    parser.add_argument(
        "--save-factr-pose",
        nargs="?",
        const="factr/validation/saved_factr_pose.json",
        default=None,
        help="Read FACTR once without torque and save current joints to this JSON path.",
    )
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--rate", type=float, default=120.0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--print-target-only", action="store_true")
    parser.add_argument("--no-arm-gravity-comp", action="store_true")
    parser.add_argument(
        "--command-factr",
        action="store_true",
        help="Move physical FACTR toward the reference using bounded torque.",
    )
    parser.add_argument(
        "--mujoco-only",
        action="store_true",
        help="Only move the MuJoCo Panda; do not connect to FACTR hardware.",
    )
    parser.add_argument(
        "--factr-only",
        action="store_true",
        help="Only command or monitor FACTR; do not load MuJoCo or clip the target.",
    )
    parser.add_argument("--factr-config", default="factr/leader.yaml")
    parser.add_argument(
        "--factr-calibration-report",
        default="factr/validation/calibration.json",
    )
    parser.add_argument(
        "--factr-validation-report",
        default="factr/validation/torque_direction.json",
    )
    parser.add_argument("--factr-duration", type=float, default=15.0)
    parser.add_argument("--factr-align-speed", type=float, default=0.5)
    parser.add_argument(
        "--factr-hold-error",
        type=float,
        default=0.08,
        help="Local PD error window in rad; larger asks for more torque but can oscillate.",
    )
    parser.add_argument("--factr-max-torque", type=float, default=0.15)
    parser.add_argument(
        "--factr-gripper-target",
        type=float,
        default=None,
        help="Override saved gripper target in radians; omitted uses pose-file gripper_position.",
    )
    parser.add_argument(
        "--no-factr-hold-gripper",
        action="store_true",
        help="Do not hold the 8th FACTR motor during active reference-pose moves.",
    )
    parser.add_argument("--factr-gripper-kp", type=float, default=5.0)
    parser.add_argument("--factr-gripper-kd", type=float, default=2.0)
    parser.add_argument("--factr-max-gripper-torque", type=float, default=0.08)
    parser.add_argument(
        "--factr-gripper-torque-sign",
        type=float,
        choices=(-1.0, 1.0),
        default=1.0,
        help="Flip only the 8th motor hold torque if the gripper moves away from target.",
    )
    parser.add_argument(
        "--factr-drive-torque",
        type=_parse_scalar_or_joint_list,
        default=0.0,
        help="Extra signed torque for joints far from target; one value or 7 joint values.",
    )
    parser.add_argument("--factr-drive-deadband", type=float, default=0.08)
    parser.add_argument("--factr-drive-ramp", type=float, default=0.30)
    parser.add_argument("--factr-ramp-duration", type=float, default=5.0)
    parser.add_argument(
        "--factr-trajectory-duration",
        type=float,
        default=8.0,
        help="Seconds for the commanded FACTR target to move smoothly from current pose to goal.",
    )
    parser.add_argument("--factr-tolerance", type=float, default=0.05)
    parser.add_argument("--factr-monitor-rate", type=float, default=2.0)
    parser.add_argument(
        "--factr-monitor-duration",
        type=float,
        default=60.0,
        help="Seconds to print read-only FACTR alignment errors; 0 runs until reached or Ctrl+C.",
    )
    parser.add_argument("--factr-kp", type=_parse_joint_list, default=GO_TO_FACTR_KP)
    parser.add_argument("--factr-kd", type=_parse_joint_list, default=GO_TO_FACTR_KD)
    parser.add_argument("--no-factr-hold-gravity-comp", action="store_true", default=True)
    parser.add_argument(
        "--factr-hold-gravity-comp",
        dest="no_factr_hold_gravity_comp",
        action="store_false",
    )
    parser.add_argument("--no-factr-hold-friction-comp", action="store_true", default=True)
    parser.add_argument(
        "--factr-hold-friction-comp",
        dest="no_factr_hold_friction_comp",
        action="store_false",
    )
    parser.add_argument(
        "--factr-ignore-soft-limits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate against hard YAML limits instead of narrowed soft limits.",
    )
    parser.add_argument(
        "--factr-validate-arm-limits",
        action="store_true",
        help="Validate start and target against YAML arm limits before moving.",
    )
    parser.add_argument(
        "--factr-disable-limit-torque",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Do not add soft joint-limit barrier torques during the active reference move.",
    )
    parser.add_argument(
        "--factr-allow-over-config-torque",
        action="store_true",
        help="Use --factr-max-torque directly instead of clipping by controller.max_arm_torque.",
    )
    parser.add_argument("--max-factr-velocity", type=float, default=8.0)
    parser.add_argument("--max-factr-jump", type=float, default=3.2)
    return parser.parse_args()


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
