#!/usr/bin/env python3
"""Own the FACTR serial device for all MuJoCo policy processes."""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robot.factr_rpc import (  # noqa: E402
    FactrRpcService,
    FactrTcpServer,
    write_ready_file,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18075)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("factr/leader.yaml"))
    parser.add_argument(
        "--init-pose",
        type=Path,
        default=Path("factr/validation/init_pose.json"),
    )
    parser.add_argument(
        "--rest-pose",
        type=Path,
        default=Path("factr/validation/rest_pose.json"),
    )
    parser.add_argument("--rest-position-tolerance", type=float, default=0.08)
    parser.add_argument("--position-tolerance", type=float, default=0.04)
    parser.add_argument("--init-timeout", type=float, default=18.0)
    parser.add_argument("--alignment-timeout", type=float, default=18.0)
    parser.add_argument("--max-velocity", type=float, default=0.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.position_tolerance <= 0.0:
        raise ValueError("--position-tolerance must be > 0")
    if args.rest_position_tolerance <= 0.0:
        raise ValueError("--rest-position-tolerance must be > 0")
    if args.init_timeout <= 0.0:
        raise ValueError("--init-timeout must be > 0")
    if args.alignment_timeout <= 0.0:
        raise ValueError("--alignment-timeout must be > 0")
    if args.max_velocity <= 0.0:
        raise ValueError("--max-velocity must be > 0")
    for label, path in (
        ("FACTR config", args.config),
        ("FACTR rest pose", args.rest_pose),
        ("FACTR initial pose", args.init_pose),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    os.environ["FACTR_CONFIG"] = str(args.config)
    os.environ["FACTR_INIT_POSE"] = str(args.init_pose)
    os.environ["FACTR_REST_POSE"] = str(args.rest_pose)
    os.environ["FACTR_REST_POSITION_TOLERANCE"] = str(
        args.rest_position_tolerance
    )
    os.environ["FACTR_POSITION_TOLERANCE"] = str(args.position_tolerance)
    os.environ["FACTR_INIT_TIMEOUT"] = str(args.init_timeout)
    os.environ["FACTR_ALIGNMENT_TIMEOUT"] = str(args.alignment_timeout)
    os.environ["FACTR_MAX_VELOCITY"] = str(args.max_velocity)

    try:
        args.ready_file.unlink(missing_ok=True)
    except OSError:
        pass

    # Keep hardware-only dependencies (Pinocchio and Dynamixel SDK) out of
    # argument parsing and the policy processes. They are required only here.
    from robot.factr_control_adapter import FactrControlAdapter

    adapter = FactrControlAdapter()
    service = FactrRpcService(adapter)
    server = None

    def _stop(_signum, _frame):
        raise KeyboardInterrupt

    # Installed BEFORE initialize() so Ctrl+C works during startup staging — in
    # particular while the arm is being held after a staging failure.
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        print(
            "[FACTR] State INITIALIZING: connecting, verifying the rest pose, "
            "and moving to the initial pose.",
            flush=True,
        )
        service.initialize()
        server = FactrTcpServer((args.host, args.port), service)
        write_ready_file(args.ready_file, host=args.host, port=args.port)
        print(
            f"[FACTR] State WAITING: service ready on {args.host}:{args.port}; "
            "policy rollout may start.",
            flush=True,
        )

        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("[FACTR] Service stopping.", flush=True)
    except Exception as exc:
        print(f"[FACTR] FATAL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        if getattr(adapter, "staging_hold_active", False):
            # The arm is still under torque at its current pose. Falling through to the
            # finally block would close the adapter, disable torque and drop it. Wait for
            # the operator instead: they support the arm, then Ctrl+C to park it safely.
            print(
                "[FACTR] Arm is HELD, not released. Waiting for you to stop the "
                "service (Ctrl+C) — support the arm first.",
                file=sys.stderr,
                flush=True,
            )
            try:
                while True:
                    time.sleep(0.2)
            except KeyboardInterrupt:
                print("[FACTR] Operator stop; parking and releasing.", flush=True)
        return 1
    finally:
        if server is not None:
            server.server_close()
        service.close()
        try:
            args.ready_file.unlink(missing_ok=True)
        except OSError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
