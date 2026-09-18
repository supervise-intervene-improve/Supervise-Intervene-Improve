#!/usr/bin/env python3
"""Passive FACTR offset capture for the local MuJoCo gravity-comp path."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from factr import FACTRGravityCompensation  # noqa: E402
from factr.validation import hardware_fingerprint  # noqa: E402


DEFAULT_REFERENCE = [0.074, -2.014, -0.058, -3.21, -0.062, 1.09, -6.22]
DEFAULT_GRIPPER_REFERENCE = -0.55


def _parse_reference(text: str | None) -> np.ndarray:
    if text is None:
        return np.asarray(DEFAULT_REFERENCE, dtype=np.float64)
    values = [float(part) for part in text.replace(",", " ").split()]
    if len(values) != 7:
        raise ValueError("--reference-joints must contain exactly 7 numbers")
    return np.asarray(values, dtype=np.float64)


def _parse_gripper_reference(text: str | None) -> float:
    if text is None:
        return float(DEFAULT_GRIPPER_REFERENCE)
    return float(text)


def _write_yaml(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def run(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    report_path = Path(args.report)
    reference = _parse_reference(args.reference_joints)
    gripper_reference = _parse_gripper_reference(args.reference_gripper)
    if not config_path.is_file():
        raise FileNotFoundError(f"FACTR config not found: {config_path}")

    print("Passive FACTR calibration capture.")
    print("No motor torque/current command will be enabled.")
    print(f"Reference arm pose [rad]: {reference.tolist()}")
    print(f"Reference gripper pose [rad]: {gripper_reference:.6f}")
    print("Place FACTR in that physical pose and hold it still.")
    input("Press Enter when FACTR is stationary in the reference pose...")

    factr = FACTRGravityCompensation(config_path, read_only=True)
    try:
        raw_positions = []
        raw_velocities = []
        period = 1.0 / args.rate
        for _ in range(args.samples):
            raw_pos, raw_vel = factr.driver.get_positions_and_velocities()
            raw_positions.append(np.asarray(raw_pos, dtype=np.float64))
            raw_velocities.append(np.asarray(raw_vel, dtype=np.float64))
            time.sleep(period)
    finally:
        factr.shutdown()

    raw_positions_arr = np.vstack(raw_positions)
    raw_velocities_arr = np.vstack(raw_velocities)
    mean_raw = np.mean(raw_positions_arr, axis=0)
    spread = np.ptp(raw_positions_arr[:, :7], axis=0)
    max_spread = float(np.max(spread))
    max_velocity = float(np.max(np.abs(raw_velocities_arr[:, :7])))

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    signs = np.asarray(payload["dynamixel"]["joint_signs"], dtype=np.float64)
    offsets = np.zeros(8, dtype=np.float64)
    offsets[:7] = mean_raw[:7] - reference / signs[:7]
    offsets[7] = mean_raw[7] - gripper_reference / signs[7]
    if args.prepare_gravity:
        payload["controller"]["null_space_regulation"]["kp"] = 0.0
        payload["controller"]["null_space_regulation"]["kd"] = 0.0
        payload["controller"]["static_friction_comp"]["gain"] = 0.0
        payload["controller"]["torque_feedback"]["enable"] = False
        payload["controller"]["gripper_feedback"]["enable"] = False

    if args.write_config:
        payload["dynamixel"]["joint_offsets"] = offsets.tolist()
        payload["arm_teleop"]["initialization"]["calibration_joint_pos"] = reference.tolist()
        payload["arm_teleop"]["initialization"]["initial_match_joint_pos"] = reference.tolist()
        _write_yaml(config_path, payload)
    calibrated = (mean_raw[:7] - offsets[:7]) * signs[:7]
    calibrated_gripper = float((mean_raw[7] - offsets[7]) * signs[7])
    error = calibrated - reference
    gripper_error = calibrated_gripper - gripper_reference
    max_error = float(np.max(np.abs(error)))
    passed = (
        max_error <= args.reference_tolerance
        and max_velocity <= args.max_velocity
        and max_spread <= args.max_spread
    )
    report = {
        "pass": passed,
        "hardware_fingerprint": hardware_fingerprint(config_path),
        "sample_count": args.samples,
        "reference_joints_rad": reference.tolist(),
        "reference_gripper_rad": gripper_reference,
        "measured_joints_rad": calibrated.tolist(),
        "measured_gripper_rad": calibrated_gripper,
        "error_rad": error.tolist(),
        "gripper_error_rad": gripper_error,
        "joint_offsets": offsets.tolist(),
        "max_error_rad": max_error,
        "max_velocity_rad_s": max_velocity,
        "max_sample_spread_rad": max_spread,
        "criteria": {
            "reference_tolerance_rad": args.reference_tolerance,
            "max_velocity_rad_s": args.max_velocity,
            "max_sample_spread_rad": args.max_spread,
        },
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if args.write_config:
        print(f"Wrote calibrated config: {config_path}")
    else:
        print(f"Left config unchanged: {config_path}")
    print(f"Wrote calibration report: {report_path}")
    print(f"Max reference error: {max_error:.6f} rad")
    print(f"Gripper reference error: {gripper_error:.6f} rad")
    print(f"Max sample velocity: {max_velocity:.6f} rad/s")
    print(f"Max sample spread: {max_spread:.6f} rad")
    print("[CALIBRATION] PASS" if passed else "[CALIBRATION] FAIL")
    return 0 if passed else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="factr/leader.yaml")
    parser.add_argument(
        "--report",
        default="factr/validation/calibration.json",
        help="Calibration report path consumed by active FACTR modes.",
    )
    parser.add_argument(
        "--reference-joints",
        default=None,
        help="Seven known physical joint values, comma or space separated.",
    )
    parser.add_argument(
        "--reference-gripper",
        default=None,
        help="Known physical 8th motor/gripper value in radians.",
    )
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--reference-tolerance", type=float, default=0.05)
    parser.add_argument("--max-velocity", type=float, default=0.10)
    parser.add_argument("--max-spread", type=float, default=0.01)
    parser.add_argument(
        "--prepare-gravity",
        action="store_true",
        help=(
            "When used with --write-config, disable auxiliary torque terms so "
            "--gravity-comp uses gravity only."
        ),
    )
    parser.add_argument(
        "--write-config",
        action="store_true",
        help="Write joint_offsets and gravity-only settings into --config.",
    )
    return parser.parse_args()


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
