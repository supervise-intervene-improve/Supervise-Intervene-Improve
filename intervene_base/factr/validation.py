from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from factr.hardware.factr import FactrSafetyError


def hardware_fingerprint(config_path: str | Path) -> str:
    payload = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    dynamixel = payload.get("dynamixel", {})
    arm = payload.get("arm_teleop", {})
    tracked = {
        "dynamixel": {
            "dynamixel_port": dynamixel.get("dynamixel_port"),
            "servo_types": dynamixel.get("servo_types"),
            "joint_signs": dynamixel.get("joint_signs"),
        },
        "arm_teleop": {
            "leader_urdf": arm.get("leader_urdf"),
            "num_arm_joints": arm.get("num_arm_joints"),
            "arm_joint_limits_max": arm.get("arm_joint_limits_max"),
            "arm_joint_limits_min": arm.get("arm_joint_limits_min"),
            "arm_joint_limits_safety_margin": arm.get(
                "arm_joint_limits_safety_margin"
            ),
        },
        "gripper_teleop": payload.get("gripper_teleop", {}),
    }
    encoded = json.dumps(tracked, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_report(path: str | Path, label: str) -> dict:
    report_path = Path(path)
    if not report_path.is_file():
        raise FactrSafetyError(f"Required FACTR report not found: {report_path}")
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FactrSafetyError(f"Invalid {label} report JSON: {report_path}") from exc


def _check_fingerprint(config_path: str | Path, report: dict, label: str) -> None:
    expected = hardware_fingerprint(config_path)
    actual = report.get("hardware_fingerprint")
    if actual is not None and actual != expected:
        raise FactrSafetyError(
            f"{label} report hardware fingerprint does not match {config_path}"
        )


def validate_calibration_report(config_path: str | Path, report_path: str | Path) -> dict:
    report = _load_report(report_path, "calibration")
    if not report.get("pass", False):
        raise FactrSafetyError(f"FACTR calibration report is not PASS: {report_path}")
    _check_fingerprint(config_path, report, "Calibration")
    return report


def validate_torque_direction_report(
    config_path: str | Path, report_path: str | Path
) -> dict:
    report = _load_report(report_path, "torque-direction")
    _check_fingerprint(config_path, report, "Torque-direction")
    joints = report.get("joints", {})
    missing_or_failed = []
    for joint in range(1, 8):
        joint_report = joints.get(str(joint))
        if not isinstance(joint_report, dict):
            missing_or_failed.append(str(joint))
            continue
        if joint_report.get("pass", False):
            continue
        directions = joint_report.get("directions", {})
        if not isinstance(directions, dict) or not all(
            isinstance(directions.get(direction), dict)
            and directions[direction].get("pass", False)
            for direction in ("positive", "negative")
        ):
            missing_or_failed.append(str(joint))
    if missing_or_failed:
        raise FactrSafetyError(
            "FACTR torque-direction report is incomplete or not PASS for J"
            + ", J".join(missing_or_failed)
        )
    return report


def validate_gravity_report(
    config_path: str | Path,
    report_path: str | Path,
    *,
    gain_scale: float,
    max_torque_nm: float,
) -> dict:
    report = _load_report(report_path, "gravity")
    if not report.get("pass", False):
        raise FactrSafetyError(f"FACTR gravity report is not PASS: {report_path}")
    _check_fingerprint(config_path, report, "Gravity")
    settings = report.get("settings", report)
    if "gain_scale" in settings and abs(float(settings["gain_scale"]) - gain_scale) > 1e-9:
        raise FactrSafetyError("FACTR gravity gain scale was not validated")
    if (
        "max_torque_nm" in settings
        and max_torque_nm > float(settings["max_torque_nm"]) + 1e-9
    ):
        raise FactrSafetyError(
            "FACTR gravity torque cap exceeds the validated value"
        )
    report.setdefault(
        "criteria",
        {
            "max_peak_velocity_rad_s": 0.30,
            "max_peak_displacement_rad": 0.08,
        },
    )
    return report
