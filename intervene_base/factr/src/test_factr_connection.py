#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from factr import DEFAULT_CONFIG_PATH, FACTRGravityCompensation


def _state_payload(factr: FACTRGravityCompensation, state) -> dict:
    ee_pos, ee_rot = factr.get_end_effector_pose(state.joint_positions)
    return {
        "serial_device": factr.dynamixel_port,
        "joint_positions_rad": state.joint_positions.tolist(),
        "joint_velocities_rad_s": state.joint_velocities.tolist(),
        "gripper_position_rad": state.gripper_position,
        "gripper_velocity_rad_s": state.gripper_velocity,
        "end_effector_position_m": np.asarray(ee_pos).tolist(),
        "end_effector_rotation_matrix": np.asarray(ee_rot).tolist(),
        "command_status": "read-only; torque/current control was not enabled",
    }


def run_connection_test(args: argparse.Namespace | SimpleNamespace | Path | str) -> int:
    if isinstance(args, (str, Path)):
        args = SimpleNamespace(config=Path(args), port=None, samples=1, rate=10.0, json=False)
    config_path = Path(getattr(args, "config", DEFAULT_CONFIG_PATH))
    port = getattr(args, "port", None)
    samples = int(getattr(args, "samples", 1))
    rate = float(getattr(args, "rate", 10.0))
    as_json = bool(getattr(args, "json", False))

    if not config_path.is_file():
        raise FileNotFoundError(f"FACTR config not found: {config_path}")
    if samples <= 0:
        raise ValueError("--samples must be positive")

    if not as_json:
        print("Connecting to FACTR in read-only mode (motor settings are unchanged)...")
    factr = FACTRGravityCompensation(config_path, port=port, read_only=True)
    try:
        payloads = []
        period = 1.0 / rate if rate > 0 else 0.0
        for index in range(samples):
            state = factr.read_state()
            payload = _state_payload(factr, state)
            payloads.append(payload)
            if as_json:
                print(json.dumps(payload))
            else:
                print(f"FACTR sample {index + 1}/{samples}:")
                print(f"  Serial device: {payload['serial_device']}")
                print(f"  Arm joints [rad]: {payload['joint_positions_rad']}")
                print(f"  Arm velocities [rad/s]: {payload['joint_velocities_rad_s']}")
                print(f"  Gripper position [rad]: {payload['gripper_position_rad']:.6f}")
                print(f"  Gripper velocity [rad/s]: {payload['gripper_velocity_rad_s']:.6f}")
                print(f"  End-effector position [m]: {payload['end_effector_position_m']}")
                print("  Command status: read-only; torque/current control was not enabled")
            if index + 1 < samples and period > 0.0:
                time.sleep(period)
        if not as_json:
            print("FACTR connected: synchronous state read succeeded")
        return 0
    finally:
        factr.shutdown()
        if not as_json:
            print("FACTR connection closed cleanly")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read FACTR state without commanding the motors."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--port", default=None)
    parser.add_argument("--samples", type=int, default=1)
    parser.add_argument("--rate", type=float, default=10.0)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    return run_connection_test(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
