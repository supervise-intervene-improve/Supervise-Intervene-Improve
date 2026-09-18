#!/usr/bin/env python3
"""Validate FACTR torque directions with short bounded pulses."""

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

from factr import FACTRGravityCompensation  # noqa: E402
from factr.validation import hardware_fingerprint, validate_calibration_report  # noqa: E402


def _load_report(path: Path, config_path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "hardware_fingerprint": hardware_fingerprint(config_path),
        "joints": {},
    }


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def _direction_sign(direction: str) -> float:
    if direction == "positive":
        return 1.0
    if direction == "negative":
        return -1.0
    raise ValueError("--direction must be positive or negative")


def _preflight(config_path: Path, calibration_report: Path) -> None:
    validate_calibration_report(config_path, calibration_report)
    print("[TORQUE DIRECTION] Passive preflight read.")
    passive = FACTRGravityCompensation(config_path, read_only=True)
    try:
        state = passive.read_state()
        passive.validate_arm_positions(state.joint_positions, safety_limits=True)
        print(f"  Current arm q [rad]: {state.joint_positions.tolist()}")
        print(f"  Current arm qd [rad/s]: {state.joint_velocities.tolist()}")
    finally:
        passive.shutdown()


def _planned_checks(args: argparse.Namespace) -> list[tuple[int, str]]:
    if args.all:
        return [
            (joint, direction)
            for joint in range(1, 8)
            for direction in ("positive", "negative")
        ]
    if args.joint is None or args.direction is None:
        raise ValueError("Use --all or provide both --joint and --direction")
    if args.joint < 1 or args.joint > 7:
        raise ValueError("--joint must be in 1..7")
    return [(args.joint, args.direction)]


def _record_result(
    *,
    report_path: Path,
    config_path: Path,
    joint: int,
    direction: str,
    accepted: bool,
    torque_nm: float,
    duration_s: float,
    before,
    after,
    delta: np.ndarray,
) -> None:
    report = _load_report(report_path, config_path)
    report["hardware_fingerprint"] = hardware_fingerprint(config_path)
    joint_report = report.setdefault("joints", {}).setdefault(str(joint), {})
    directions = joint_report.setdefault("directions", {})
    directions[direction] = {
        "pass": accepted,
        "timestamp_unix_s": time.time(),
        "torque_nm": torque_nm,
        "duration_s": duration_s,
        "start_q_rad": before.joint_positions.tolist(),
        "end_q_rad": after.joint_positions.tolist(),
        "delta_rad": delta.tolist(),
    }
    joint_report["pass"] = all(
        isinstance(directions.get(candidate), dict)
        and directions[candidate].get("pass", False)
        for candidate in ("positive", "negative")
    )
    _write_report(report_path, report)


def run(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    calibration_report = Path(args.calibration_report)
    report_path = Path(args.report)
    checks = _planned_checks(args)
    _preflight(config_path, calibration_report)

    print("[TORQUE DIRECTION] Planned checks:")
    for joint, direction in checks:
        sign = _direction_sign(direction)
        print(f"  J{joint} {direction}: {sign * args.torque:+.4f} Nm for {args.duration:.3f}s")

    if not args.execute:
        print("Dry run only. Add --execute to apply the bounded pulse.")
        return 0

    input(
        "Support the whole FACTR arm and keep a hand at power cutoff. "
        "Press Enter to begin the bounded pulse sequence..."
    )

    active = FACTRGravityCompensation(config_path, read_only=False)
    accepted_count = 0
    try:
        for index, (joint, direction) in enumerate(checks, start=1):
            joint_index = joint - 1
            sign = _direction_sign(direction)
            command = np.zeros(7, dtype=np.float64)
            command[joint_index] = sign * args.torque
            print(
                f"\n[TORQUE DIRECTION] {index}/{len(checks)}: J{joint} {direction}"
            )
            print(
                "Expected observation: calibrated J"
                f"{joint} angle should move {direction}."
            )
            if not args.no_per_pulse_prompt:
                input("Press Enter for this one pulse...")
            active.set_leader_joint_torque(np.zeros(7, dtype=np.float64), 0.0)
            time.sleep(0.10)
            before = active.read_state()
            active.set_leader_joint_torque(command, 0.0)
            time.sleep(args.duration)
            active.set_leader_joint_torque(np.zeros(7, dtype=np.float64), 0.0)
            time.sleep(0.10)
            after = active.read_state()

            delta = after.joint_positions - before.joint_positions
            print(f"[TORQUE DIRECTION] Observed delta [rad]: {delta.tolist()}")
            print(
                f"[TORQUE DIRECTION] J{joint} delta: "
                f"{delta[joint_index]:+.6f} rad"
            )
            response = input(
                "Type 's' then Enter only if this movement was correct: "
            ).strip().lower()
            accepted = response == "s"
            if accepted:
                accepted_count += 1
            _record_result(
                report_path=report_path,
                config_path=config_path,
                joint=joint,
                direction=direction,
                accepted=accepted,
                torque_nm=command[joint_index],
                duration_s=args.duration,
                before=before,
                after=after,
                delta=delta,
            )
            print(
                "[TORQUE DIRECTION] ACCEPTED"
                if accepted
                else "[TORQUE DIRECTION] NOT ACCEPTED"
            )
    finally:
        try:
            active.set_leader_joint_torque(np.zeros(7, dtype=np.float64), 0.0)
        except Exception:
            pass
        active.shutdown()

    print(f"Wrote torque-direction report: {report_path}")
    print(f"[TORQUE DIRECTION] Accepted {accepted_count}/{len(checks)} checks.")
    return 0 if accepted_count == len(checks) else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="factr/leader.yaml")
    parser.add_argument(
        "--calibration-report",
        default="factr/validation/calibration.json",
    )
    parser.add_argument(
        "--report",
        default="factr/validation/torque_direction.json",
    )
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--joint", type=int, default=None)
    parser.add_argument(
        "--direction",
        choices=("positive", "negative"),
        default=None,
    )
    parser.add_argument("--torque", type=float, default=0.025)
    parser.add_argument("--duration", type=float, default=0.20)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--no-per-pulse-prompt",
        action="store_true",
        help="After the initial arming prompt, run each pulse without another Enter prompt.",
    )
    return parser.parse_args()


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
