#!/usr/bin/env python3
"""Probe robot reachability and say exactly which layer is broken.

WHY THIS EXISTS
---------------
When the polymetis server is not reachable, telekinesis intervention fails with a gRPC
`StatusCode.UNAVAILABLE / "failed to connect to all addresses"` buried in the policy log,
several seconds after the operator presses X in the headset. That message does not
distinguish between the three very different causes:

  * the host is down or the IP is wrong          -> connect times out
  * the host is up but no server is listening    -> connect is REFUSED (TCP RST)
  * everything is fine                           -> connect succeeds

That distinction is the whole diagnosis. A refusal from a host that also answers SSH in
0.2 ms means the address is right and the *server* is not running (or is bound to
127.0.0.1 on its own machine, which looks perfectly healthy from over there while being
unreachable from everywhere else). A timeout means the address or the network is wrong.

Usage:
    python3 utils/robot_doctor.py                # probe every known robot
    python3 utils/robot_doctor.py --key p4       # probe just one
    python3 utils/robot_doctor.py --quiet        # one line per robot, for the launcher

Exit code is 0 if every probed robot is fully reachable, 1 otherwise — so a launcher can
branch on it, though run_main_policy.sh deliberately treats it as non-fatal (the sim grid
is still useful without an arm).
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from pathlib import Path
from typing import List, Optional, Tuple

# Allow running as a bare script from anywhere: intervene_base must be importable so
# `robot.robot_endpoints` resolves.
_INTERVENE_ROOT = Path(__file__).resolve().parent.parent
if str(_INTERVENE_ROOT) not in sys.path:
    sys.path.insert(0, str(_INTERVENE_ROOT))

from robot.robot_endpoints import ENDPOINTS, RobotEndpoint  # noqa: E402

STATUS_OPEN = "OPEN"
STATUS_REFUSED = "REFUSED"
STATUS_TIMEOUT = "TIMEOUT"
STATUS_ERROR = "ERROR"

_EXPLANATION = {
    STATUS_OPEN: "server is listening",
    STATUS_REFUSED: "host is up, but no server is listening on this port",
    STATUS_TIMEOUT: "host unreachable (wrong IP, host down, or blocked)",
    STATUS_ERROR: "probe failed",
}


def probe_port(ip: str, port: int, timeout: float) -> Tuple[str, str]:
    """Return (status, detail) for a single TCP endpoint."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((ip, port))
        return STATUS_OPEN, ""
    except ConnectionRefusedError:
        return STATUS_REFUSED, ""
    except (socket.timeout, TimeoutError):
        return STATUS_TIMEOUT, ""
    except OSError as exc:
        # e.g. EHOSTUNREACH / ENETUNREACH — treat as unreachable, keep the errno text.
        return STATUS_TIMEOUT, f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # pragma: no cover - defensive
        return STATUS_ERROR, f"{type(exc).__name__}: {exc}"
    finally:
        try:
            sock.close()
        except Exception:
            pass


def resolve_hostname(ip: str) -> str:
    try:
        return socket.gethostbyaddr(ip)[0]
    except Exception:
        return ""


def check_robot(endpoint: RobotEndpoint, timeout: float) -> dict:
    arm_status, arm_detail = probe_port(endpoint.ip_address, endpoint.arm_port, timeout)
    grip_status, grip_detail = probe_port(endpoint.ip_address, endpoint.gripper_port, timeout)
    return {
        "key": endpoint.key,
        "name": endpoint.name,
        "ip": endpoint.ip_address,
        "arm_port": endpoint.arm_port,
        "gripper_port": endpoint.gripper_port,
        "arm_status": arm_status,
        "arm_detail": arm_detail,
        "gripper_status": grip_status,
        "gripper_detail": grip_detail,
        "hostname": resolve_hostname(endpoint.ip_address),
        "ok": arm_status == STATUS_OPEN and grip_status == STATUS_OPEN,
    }


def advice_for(result: dict) -> str:
    """One actionable sentence for the dominant failure of this robot."""
    statuses = {result["arm_status"], result["gripper_status"]}
    if statuses == {STATUS_OPEN}:
        return "reachable"
    if STATUS_TIMEOUT in statuses:
        return (
            f"host {result['ip']} unreachable — check the IP (override with "
            f"INTERVENE_ROBOT_{result['key'].upper()}_IP=...) and the network."
        )
    if STATUS_REFUSED in statuses:
        return (
            f"host {result['ip']} is up but the polymetis server is not listening — "
            "start it on that machine, and make sure it binds 0.0.0.0 and not 127.0.0.1 "
            "(a localhost bind looks healthy there but is unreachable from here)."
        )
    return "probe failed; see detail above."


def format_report(results: List[dict], quiet: bool) -> str:
    lines: List[str] = []
    if quiet:
        for r in results:
            if r["ok"]:
                lines.append(f"[RobotDoctor] {r['key']} @{r['ip']} — OK (arm+gripper listening).")
            else:
                lines.append(
                    f"[RobotDoctor] {r['key']} @{r['ip']}:{r['arm_port']} — UNREACHABLE "
                    f"(arm={r['arm_status']}, gripper={r['gripper_status']}). {advice_for(r)}"
                )
        return "\n".join(lines)

    lines.append("=" * 78)
    lines.append("Robot reachability report")
    lines.append("=" * 78)
    for r in results:
        host = f" ({r['hostname']})" if r["hostname"] else ""
        lines.append("")
        lines.append(f"{r['key']}  {r['name']}  @ {r['ip']}{host}")
        for label, port, status, detail in (
            ("arm    ", r["arm_port"], r["arm_status"], r["arm_detail"]),
            ("gripper", r["gripper_port"], r["gripper_status"], r["gripper_detail"]),
        ):
            note = _EXPLANATION.get(status, "")
            extra = f" [{detail}]" if detail else ""
            lines.append(f"    {label} :{port:<6} {status:<8} {note}{extra}")
        lines.append(f"    -> {advice_for(r)}")

    lines.append("")
    lines.append("-" * 78)
    bad = [r["key"] for r in results if not r["ok"]]
    if bad:
        lines.append(f"NOT reachable: {', '.join(bad)}")
        lines.append(
            "Telekinesis intervention switches the REAL arm to HUMAN_CONTROL, so it "
            "cannot start until the arm above is reachable."
        )
    else:
        lines.append("All probed robots are reachable.")
    lines.append("-" * 78)
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--key",
        action="append",
        default=None,
        help="Robot key to probe (repeatable). Default: the key in INTERVENE_ROBOT_KEY, "
             "or every known robot if that is unset.",
    )
    parser.add_argument("--all", action="store_true", help="Probe every known robot.")
    parser.add_argument("--timeout", type=float, default=2.0, help="Per-port timeout in seconds.")
    parser.add_argument("--quiet", action="store_true", help="One line per robot (for launchers).")
    args = parser.parse_args(argv)

    if args.key:
        keys = list(args.key)
    elif args.all:
        keys = list(ENDPOINTS.keys())
    else:
        env_key = os.environ.get("INTERVENE_ROBOT_KEY", "").strip()
        keys = [env_key] if env_key else list(ENDPOINTS.keys())

    unknown = [k for k in keys if k not in ENDPOINTS]
    if unknown:
        print(
            f"[RobotDoctor] Unknown robot key(s): {', '.join(unknown)}. "
            f"Known: {', '.join(sorted(ENDPOINTS))}",
            file=sys.stderr,
        )
        return 2

    results = [check_robot(ENDPOINTS[k], args.timeout) for k in keys]
    print(format_report(results, args.quiet))
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
