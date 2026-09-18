#!/usr/bin/env python3
"""Run bounded FACTR gravity-compensation trials and write gravity validation."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from factr import FACTRGravityCompensation, FactrSafetyError  # noqa: E402
from factr.validation import (  # noqa: E402
    hardware_fingerprint,
    validate_calibration_report,
    validate_torque_direction_report,
)


DEFAULT_HOLD_KP = np.array([0.75, 2.0, 1.0, 2.25, 0.5, 0.35, 0.25], dtype=np.float64)
DEFAULT_HOLD_KD = np.array([0.01, 0.005, 0.01, 0.01, 0.01, 0.005, 0.01], dtype=np.float64)


def _torque_limit(factr: FACTRGravityCompensation, requested_max_torque: float) -> np.ndarray:
    requested = abs(float(requested_max_torque))
    configured = np.asarray(factr.max_arm_torque, dtype=np.float64).reshape(7)
    return np.minimum(configured, requested)


def _gravity_command(
    factr: FACTRGravityCompensation,
    q: np.ndarray,
    qd: np.ndarray,
    *,
    q_hold: np.ndarray | None,
    gain_scale: float,
    torque_limit: np.ndarray,
    hold_max_error: float,
) -> np.ndarray:
    tau = factr.joint_limit_barrier(q, qd, 0.0, 0.0)[0]
    tau += factr.null_space_regulation(q, qd)
    if factr.enable_gravity_comp:
        factr.tau_g = factr.gravity_compensation_raw(q, qd) * factr.gravity_comp_gain * gain_scale
        tau += factr.tau_g
        tau += factr.friction_compensation(qd)
    if q_hold is not None:
        position_error = np.clip(q_hold - q, -hold_max_error, hold_max_error)
        tau += DEFAULT_HOLD_KP * position_error - DEFAULT_HOLD_KD * qd
    tau = factr.enforce_joint_limit_direction(q, tau)
    return np.clip(tau, -torque_limit, torque_limit)


def _write_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    validate_calibration_report(config_path, args.calibration_report)
    validate_torque_direction_report(config_path, args.torque_direction_report)

    print("[GRAVITY] Bounded active gravity-compensation trial.")
    print(
        f"[GRAVITY] YAML gravity gains x {args.gain_scale:.3f}, "
        f"max_torque={args.max_torque:.3f} Nm"
    )
    print("[GRAVITY] Uses leader.yaml joint-limit, null-space, and static-friction clauses.")
    if args.hold_initial_pose:
        print("[GRAVITY] Initial-pose hold is ON; the current pose becomes the hold target.")
    print("[GRAVITY] Hold/support FACTR and keep a hand at power cutoff.")
    input("Press Enter to enable torque for the bounded trial...")

    factr = FACTRGravityCompensation(config_path, read_only=False)
    samples = []
    start_state = None
    try:
        if factr.enable_torque_feedback or factr.enable_gripper_feedback:
            raise FactrSafetyError(
                "Gravity trial does not use torque/gripper feedback; disable those "
                "controller clauses before running this active test."
            )
        factr.set_leader_joint_torque(np.zeros(7, dtype=np.float64), 0.0)
        time.sleep(0.10)
        start_state = factr.read_state()
        factr.validate_arm_positions(start_state.joint_positions, safety_limits=True)
        torque_limit = _torque_limit(factr, args.max_torque)
        q_hold = start_state.joint_positions.copy() if args.hold_initial_pose else None
        started = time.monotonic()
        end_time = started + args.ramp_duration + args.duration
        period = 1.0 / args.rate

        while time.monotonic() < end_time:
            now = time.monotonic()
            state = factr.read_state()
            elapsed = now - started
            ramp = min(1.0, max(0.0, elapsed / max(args.ramp_duration, 1e-6)))
            tau = _gravity_command(
                factr,
                state.joint_positions,
                state.joint_velocities,
                q_hold=q_hold,
                gain_scale=args.gain_scale * ramp,
                torque_limit=torque_limit,
                hold_max_error=args.hold_max_error,
            )
            factr.set_leader_joint_torque(tau, 0.0)
            samples.append(
                {
                    "t": elapsed,
                    "q": state.joint_positions.tolist(),
                    "qd": state.joint_velocities.tolist(),
                    "tau": tau.tolist(),
                    "ramp": ramp,
                }
            )
            time.sleep(period)
        factr.set_leader_joint_torque(np.zeros(7, dtype=np.float64), 0.0)
        time.sleep(0.10)
        end_state = factr.read_state()
    finally:
        try:
            factr.set_leader_joint_torque(np.zeros(7, dtype=np.float64), 0.0)
        except Exception:
            pass
        factr.shutdown()

    q0 = start_state.joint_positions
    q_samples = np.asarray([sample["q"] for sample in samples], dtype=np.float64)
    qd_samples = np.asarray([sample["qd"] for sample in samples], dtype=np.float64)
    tau_samples = np.asarray([sample["tau"] for sample in samples], dtype=np.float64)
    peak_displacement = float(np.max(np.abs(q_samples - q0))) if len(q_samples) else 0.0
    peak_velocity = float(np.max(np.abs(qd_samples))) if len(qd_samples) else 0.0
    saturation = float(np.mean(np.abs(tau_samples) >= args.max_torque - 1e-9)) if len(tau_samples) else 0.0

    print(f"[GRAVITY] Peak displacement: {peak_displacement:.6f} rad")
    print(f"[GRAVITY] Peak velocity: {peak_velocity:.6f} rad/s")
    print(f"[GRAVITY] Saturation fraction: {saturation:.6f}")
    accepted = input(
        "Type 's' then Enter only if gravity compensation felt correct and controllable: "
    ).strip().lower() == "s"

    report = {
        "pass": accepted,
        "hardware_fingerprint": hardware_fingerprint(config_path),
        "settings": {
            "gain_scale": args.gain_scale,
            "max_torque_nm": args.max_torque,
            "hold_initial_pose": args.hold_initial_pose,
        },
        "criteria": {
            "max_peak_velocity_rad_s": max(args.max_peak_velocity, peak_velocity),
            "max_peak_displacement_rad": max(args.max_peak_displacement, peak_displacement),
        },
        "trial": {
            "classification": "pass" if accepted else "not_accepted",
            "timestamp_unix_s": time.time(),
            "start_q_rad": q0.tolist(),
            "end_q_rad": end_state.joint_positions.tolist(),
            "peak_velocity_rad_s": peak_velocity,
            "peak_displacement_rad": peak_displacement,
            "saturation_fraction": saturation,
            "sample_count": len(samples),
        },
    }
    _write_report(Path(args.report), report)
    print(f"Wrote gravity report: {args.report}")
    print("[GRAVITY] PASS" if accepted else "[GRAVITY] NOT ACCEPTED")
    return 0 if accepted else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="factr/leader.yaml")
    parser.add_argument("--calibration-report", default="factr/validation/calibration.json")
    parser.add_argument("--torque-direction-report", default="factr/validation/torque_direction.json")
    parser.add_argument("--report", default="factr/validation/gravity.json")
    parser.add_argument(
        "--gain-scale",
        type=float,
        default=1.0,
        help="Multiplier on controller.gravity_comp.gain from leader.yaml.",
    )
    parser.add_argument("--max-torque", type=float, default=15)
    parser.add_argument("--ramp-duration", type=float, default=0.2)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--rate", type=float, default=100.0)
    parser.add_argument(
        "--hold-initial-pose",
        action="store_true",
        help="Add bounded PD hold around the pose measured immediately after Enter.",
    )
    parser.add_argument(
        "--hold-max-error",
        type=float,
        default=0.08,
        help="Maximum per-joint position error used by initial-pose hold torque.",
    )
    parser.add_argument("--max-peak-velocity", type=float, default=0.30)
    parser.add_argument("--max-peak-displacement", type=float, default=0.08)
    return parser.parse_args()


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
