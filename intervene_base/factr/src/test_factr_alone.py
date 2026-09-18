#!/usr/bin/env python3
"""MuJoCo entry point for running FACTR without the real Panda stack.

With no arguments this requests bounded FACTR gravity compensation while using
relative-anchor MuJoCo teleoperation. The underlying runner still enforces the
calibration, torque-direction, and gravity-validation gates before any motor
torque can be enabled. Pass normal ``mq3_mc_mujoco_factr.py`` arguments after
the script name to override this, for example ``--smoke-test --headless``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mq3_mc_mujoco_factr import main  # noqa: E402


def _validated_gravity_args() -> list[str]:
    report_path = REPO_ROOT / "factr" / "validation" / "gravity.json"
    if not report_path.is_file():
        return ["--factr-gravity-scale", "1.0"]
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ["--factr-gravity-scale", "1.0"]
    settings = report.get("settings", {})
    args = [
        "--factr-gravity-scale",
        str(settings.get("gain_scale", 1.0)),
    ]
    if report.get("pass") and "max_torque_nm" in settings:
        args.extend(["--factr-max-torque", str(settings["max_torque_nm"])])
    return args


def _has_option(argv: list[str], option: str) -> bool:
    return option in argv or any(arg.startswith(f"{option}=") for arg in argv)


def _append_missing_validated_gravity_args(argv: list[str]) -> None:
    if "--gravity-comp" not in argv:
        return
    validated = _validated_gravity_args()
    for option, value in zip(validated[0::2], validated[1::2]):
        if not _has_option(argv, option):
            argv.extend([option, value])


def _append_missing_takeover_teleop_args(argv: list[str]) -> None:
    if "--gravity-comp" not in argv:
        return
    defaults = {
        "--factr-gravity-ramp-duration": os.environ.get(
            "FACTR_TAKEOVER_GRAVITY_RAMP_DURATION", "0.01"
        ),
        "--factr-nullspace-scale": os.environ.get(
            "FACTR_TAKEOVER_NULLSPACE_SCALE", "0.0"
        ),
        "--factr-friction-scale": os.environ.get(
            "FACTR_TAKEOVER_FRICTION_SCALE", "1.0"
        ),
    }
    for option, value in defaults.items():
        if not _has_option(argv, option):
            argv.extend([option, value])


if __name__ == "__main__":
    if len(sys.argv) == 1:
        sys.argv.extend([
            "--relative-anchor",
            "--gravity-comp",
        ])
    _append_missing_validated_gravity_args(sys.argv)
    _append_missing_takeover_teleop_args(sys.argv)
    raise SystemExit(main())
