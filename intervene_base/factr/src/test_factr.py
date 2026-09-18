#!/usr/bin/env python3
"""Compatibility entry point for FACTR -> MuJoCo teleoperation.

In this checkout, the maintained follower path is the MuJoCo runner in
``mq3_mc_mujoco_factr.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mq3_mc_mujoco_factr import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
