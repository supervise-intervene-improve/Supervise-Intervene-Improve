#!/usr/bin/env python3
"""Send an existing runtime command to every multi-window session.

This is a PC-side helper, so it does not require rebuilding/redeploying Unity.
It talks to the same per-session cmd_port that the Quest already uses.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Iterable


def _load_ports_from_status(status_file: Path, *, include_stopped: bool) -> list[tuple[int, int, str]]:
    if not status_file.exists():
        return []
    try:
        rows = json.loads(status_file.read_text())
    except Exception as exc:
        raise RuntimeError(f"Could not read {status_file}: {exc}") from exc
    ports: list[tuple[int, int, str]] = []
    for row in rows:
        try:
            session_index = int(row["session_index"])
            cmd_port = int(row["cmd_port"])
            status = str(row.get("status", "unknown"))
        except Exception:
            continue
        if include_stopped or status in {"running", "restarting", "unknown"}:
            ports.append((session_index, cmd_port, status))
    return sorted(set(ports), key=lambda item: item[0])


def _derive_ports(windows: int, base_topic_port: int, port_step: int) -> list[tuple[int, int, str]]:
    ports: list[tuple[int, int, str]] = []
    for session_index in range(max(0, int(windows))):
        topic_port = int(base_topic_port) + session_index * int(port_step)
        ports.append((session_index, topic_port + 5, "derived"))
    return ports


def _send_command(host: str, port: int, command: str, timeout_ms: int) -> bool:
    try:
        import zmq
    except ModuleNotFoundError as exc:
        raise RuntimeError("pyzmq is required in this Python environment.") from exc

    context = zmq.Context.instance()
    socket = context.socket(zmq.PUSH)
    socket.setsockopt(zmq.LINGER, int(timeout_ms))
    socket.setsockopt(zmq.SNDTIMEO, int(timeout_ms))
    socket.setsockopt(zmq.IMMEDIATE, 1)
    socket.setsockopt(zmq.SNDHWM, 1)
    try:
        socket.connect(f"tcp://{host}:{int(port)}")
        socket.send_string(command)
        return True
    except zmq.Again:
        return False
    finally:
        socket.close(linger=int(timeout_ms))


def _parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send a runtime command to all multi-window sessions "
                    "(B = sim start/pause, Y = reset replay, X = intervene, CANCEL = cancel)."
    )
    parser.add_argument("--command", default="B", help="Command to send. Default B = sim start/pause "
                                                       "(Y = reset replay).")
    parser.add_argument("--host", default=os.environ.get("COMMAND_HOST", "127.0.0.1"))
    parser.add_argument("--status_file", default="multi_session_status.json")
    parser.add_argument("--running_only", action="store_true", help="Only target sessions marked running/restarting.")
    parser.add_argument("--windows", type=int, default=0, help="Fallback count if status file is missing.")
    parser.add_argument("--base_topic_port", type=int, default=7741)
    parser.add_argument("--port_step", type=int, default=10)
    parser.add_argument("--timeout_ms", type=int, default=350)
    parser.add_argument("--gap_ms", type=int, default=40, help="Small delay between session sends.")
    parser.add_argument(
        "--sessions", default="",
        help="Comma-separated session indices to target (e.g. '0' or '0,3'). Default: ALL. "
             "Needed to reproduce the study's topology for measurement: only ONE session is "
             "ever in single view, so 'ENTER_SINGLE to every session' measures a 9x "
             "point-cloud load that no participant ever saw.",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] = sys.argv[1:]) -> int:
    args = _parse_args(argv)
    status_file = Path(args.status_file)
    include_stopped = not bool(args.running_only)
    command = str(args.command).strip().upper()
    if not command:
        print("[MultiWindowCommand][ERROR] Empty command.")
        return 2

    try:
        ports = _load_ports_from_status(status_file, include_stopped=include_stopped)
    except RuntimeError as exc:
        print(f"[MultiWindowCommand][WARN] {exc}")
        ports = []
    if not ports and args.windows > 0:
        ports = _derive_ports(args.windows, args.base_topic_port, args.port_step)
    if not ports:
        print(
            "[MultiWindowCommand][ERROR] No sessions found. Start the launcher first, "
            "or pass --windows N."
        )
        return 1

    if args.sessions.strip():
        try:
            wanted = {int(tok) for tok in args.sessions.split(",") if tok.strip()}
        except ValueError:
            print(f"[MultiWindowCommand][ERROR] --sessions must be integers, got {args.sessions!r}")
            return 2
        filtered = [row for row in ports if row[0] in wanted]
        missing = wanted - {row[0] for row in ports}
        if missing:
            # Loud, because silently sending to 8 of the 9 you asked for produces a
            # plausible-looking result that measures the wrong thing.
            print(f"[MultiWindowCommand][ERROR] No such session(s): {sorted(missing)}. "
                  f"Known: {sorted(row[0] for row in ports)}")
            return 1
        ports = filtered

    print(
        f"[MultiWindowCommand] Sending '{command}' to {len(ports)} session(s) "
        f"at host={args.host} timeout={args.timeout_ms}ms."
    )
    ok_count = 0
    for session_index, cmd_port, status in ports:
        ok = _send_command(args.host, cmd_port, command, args.timeout_ms)
        ok_count += int(ok)
        result = "sent" if ok else "timeout/no-peer"
        print(f"  S{session_index:02d} cmd_port={cmd_port} status={status}: {result}")
        if args.gap_ms > 0:
            time.sleep(float(args.gap_ms) / 1000.0)

    print(f"[MultiWindowCommand] Done: {ok_count}/{len(ports)} sent.")
    return 0 if ok_count > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
