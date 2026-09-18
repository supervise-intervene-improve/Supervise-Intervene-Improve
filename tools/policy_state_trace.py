#!/usr/bin/env python3
"""Independent policy-state transport trace and recovery verdict.

The collector is deliberately outside both the policy producer and SimPublisher. It
subscribes to every policy PUB endpoint and records transport cadence separately from
content cadence, so a 30 Hz stream containing the same 6 Hz pose cannot look healthy by
accident.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import signal
import statistics
import struct
import sys
import time
from collections import defaultdict, deque
from pathlib import Path

import zmq


def _qpos_hash(values) -> str:
    try:
        packed = struct.pack(f"<{len(values)}d", *(float(v) for v in values))
    except (TypeError, ValueError, struct.error):
        packed = json.dumps(values, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.blake2b(packed, digest_size=8).hexdigest()


def _percentile(values, percentile):
    values = sorted(values)
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    rank = percentile / 100.0 * (len(values) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(values) - 1)
    fraction = rank - lo
    return values[lo] * (1.0 - fraction) + values[hi] * fraction


class SessionStats:
    def __init__(self):
        self.first_t = None
        self.last_t = None
        self.messages = 0
        self.distinct = 0
        self.repeated = 0
        self.seq_gaps = 0
        self.last_seq = None
        self.last_hash = None
        self.last_distinct_t = None
        self.distinct_gaps = []
        self.source_ages = []

    def add(self, now, seq, content_hash, source_wall):
        if self.first_t is None:
            self.first_t = now
        self.last_t = now
        self.messages += 1
        if self.last_seq is not None and seq is not None and seq > self.last_seq:
            self.seq_gaps += max(0, seq - self.last_seq - 1)
        if seq is not None:
            self.last_seq = seq

        changed = self.last_hash is None or content_hash != self.last_hash
        if changed:
            self.distinct += 1
            if self.last_distinct_t is not None:
                self.distinct_gaps.append(now - self.last_distinct_t)
            self.last_distinct_t = now
        else:
            self.repeated += 1
        self.last_hash = content_hash

        if source_wall is not None:
            age = now - source_wall
            if -1.0 <= age <= 60.0:
                self.source_ages.append(max(0.0, age))
        return changed

    def summary(self, now=None):
        now = time.time() if now is None else now
        duration = max(1e-9, (self.last_t or now) - (self.first_t or now))
        current_gap = None
        if self.last_distinct_t is not None:
            current_gap = max(0.0, now - self.last_distinct_t)
        gaps = list(self.distinct_gaps)
        if current_gap is not None:
            gaps.append(current_gap)
        return {
            "duration_s": duration,
            "messages": self.messages,
            "snapshot_hz": max(0, self.messages - 1) / duration,
            "distinct_states": self.distinct,
            "distinct_hz": max(0, self.distinct - 1) / duration,
            "repeated_states": self.repeated,
            "repeated_ratio": self.repeated / max(1, self.messages),
            "seq_gaps": self.seq_gaps,
            "distinct_gap_p99_s": _percentile(gaps, 99),
            "max_distinct_gap_s": max(gaps) if gaps else None,
            "source_age_p99_s": _percentile(self.source_ages, 99),
            "source_age_max_s": max(self.source_ages) if self.source_ages else None,
        }


def collect(args) -> int:
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ready_path = Path(args.ready_file) if args.ready_file else None
    stall_path = Path(args.stall_events) if args.stall_events else None
    if ready_path:
        ready_path.unlink(missing_ok=True)
    if stall_path:
        stall_path.parent.mkdir(parents=True, exist_ok=True)
        stall_path.unlink(missing_ok=True)

    context = zmq.Context()
    poller = zmq.Poller()
    sockets = {}
    for session in range(args.sessions):
        port = args.base_port + session * args.port_step
        sock = context.socket(zmq.SUB)
        sock.setsockopt(zmq.LINGER, 0)
        sock.setsockopt(zmq.RCVHWM, 1000)
        sock.setsockopt_string(zmq.SUBSCRIBE, "")
        sock.connect(f"tcp://{args.host}:{port}")
        poller.register(sock, zmq.POLLIN)
        sockets[sock] = (session, port)

    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    stats = {session: SessionStats() for session in range(args.sessions)}
    recent = {session: deque() for session in range(args.sessions)}
    last_stall_notice = {session: 0.0 for session in range(args.sessions)}
    started = time.time()
    deadline = started + args.duration if args.duration > 0 else math.inf
    next_summary = started + args.window_s

    try:
        with out_path.open("a", encoding="utf-8", buffering=1) as out:
            while not stopping and time.time() < deadline:
                events = dict(poller.poll(timeout=100))
                now = time.time()
                mono = time.monotonic()
                for sock, event in events.items():
                    if not event & zmq.POLLIN:
                        continue
                    session, port = sockets[sock]
                    while True:
                        try:
                            raw = sock.recv(flags=zmq.NOBLOCK)
                        except zmq.Again:
                            break
                        try:
                            message = json.loads(raw.decode("utf-8"))
                        except Exception as exc:
                            out.write(json.dumps({
                                "type": "decode_error", "wall_t": now,
                                "session_index": session, "bytes": len(raw),
                                "error": str(exc),
                            }, separators=(",", ":")) + "\n")
                            continue
                        try:
                            seq = int(message.get("seq"))
                        except (TypeError, ValueError):
                            seq = None
                        try:
                            source_wall = float(message.get("wall_t"))
                        except (TypeError, ValueError):
                            source_wall = None
                        content_hash = _qpos_hash(message.get("qpos", []))
                        changed = stats[session].add(now, seq, content_hash, source_wall)
                        recent[session].append((now, changed))
                        out.write(json.dumps({
                            "type": "state",
                            "wall_t": now,
                            "mono_t": mono,
                            "session_index": session,
                            "port": port,
                            "bytes": len(raw),
                            "seq": seq,
                            "sim_t": message.get("sim_t"),
                            "frame_idx": message.get("frame_idx"),
                            "episode_id": message.get("episode_id", ""),
                            "mode": message.get("mode", ""),
                            "paused": bool(message.get("paused", False)),
                            "intervention_phase": message.get("intervention_phase", ""),
                            "source_wall_t": source_wall,
                            "source_age_s": None if source_wall is None else max(0.0, now - source_wall),
                            "qpos_hash": content_hash,
                            "qpos_changed": changed,
                        }, separators=(",", ":")) + "\n")

                ready = all(item.distinct >= args.ready_distinct for item in stats.values())
                if ready and ready_path and not ready_path.exists():
                    ready_path.write_text(json.dumps({
                        "ready": True, "wall_t": now,
                        "sessions": {str(k): v.summary(now) for k, v in stats.items()},
                    }, indent=2) + "\n", encoding="utf-8")

                if stall_path:
                    for session, item in stats.items():
                        if item.last_distinct_t is None:
                            continue
                        age = now - item.last_distinct_t
                        if age >= args.stall_s and now - last_stall_notice[session] >= args.stall_s:
                            last_stall_notice[session] = now
                            with stall_path.open("a", encoding="utf-8", buffering=1) as stalls:
                                stalls.write(json.dumps({
                                    "wall_t": now, "session_index": session,
                                    "distinct_state_age_s": age,
                                    "last_seq": item.last_seq,
                                }, separators=(",", ":")) + "\n")

                if now >= next_summary:
                    for session in range(args.sessions):
                        while recent[session] and recent[session][0][0] < now - args.window_s:
                            recent[session].popleft()
                        window = recent[session]
                        out.write(json.dumps({
                            "type": "window",
                            "wall_t": now,
                            "session_index": session,
                            "window_s": args.window_s,
                            "snapshot_hz": len(window) / args.window_s,
                            "distinct_hz": sum(changed for _, changed in window) / args.window_s,
                            "lifetime": stats[session].summary(now),
                        }, separators=(",", ":")) + "\n")
                    next_summary = now + args.window_s
    finally:
        for sock in sockets:
            poller.unregister(sock)
            sock.close(linger=0)
        context.term()
    return 0


def _load_state_rows(paths):
    rows = []
    for path in paths:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if row.get("type") == "state":
                    rows.append(row)
    return rows


def report(args) -> int:
    rows = _load_state_rows(args.inputs)
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["session_index"])].append(row)

    sessions = {}
    passed = bool(grouped)
    for session, items in sorted(grouped.items()):
        stats = SessionStats()
        for row in sorted(items, key=lambda value: value["wall_t"]):
            stats.add(
                float(row["wall_t"]), row.get("seq"), row.get("qpos_hash"),
                row.get("source_wall_t"),
            )
        summary = stats.summary(float(items[-1]["wall_t"]))
        failures = []
        if summary["snapshot_hz"] < args.min_snapshot_hz:
            failures.append("snapshot_hz")
        if summary["distinct_hz"] < args.min_distinct_hz:
            failures.append("distinct_hz")
        if (summary["max_distinct_gap_s"] is None
                or summary["max_distinct_gap_s"] > args.max_distinct_gap_s):
            failures.append("max_distinct_gap_s")
        summary["failures"] = failures
        summary["passed"] = not failures
        sessions[str(session)] = summary
        passed = passed and not failures

    missing = sorted(set(range(args.expected_sessions)) - {int(k) for k in sessions})
    if missing:
        passed = False
    result = {
        "passed": passed,
        "expected_sessions": args.expected_sessions,
        "missing_sessions": missing,
        "thresholds": {
            "min_snapshot_hz": args.min_snapshot_hz,
            "min_distinct_hz": args.min_distinct_hz,
            "max_distinct_gap_s": args.max_distinct_gap_s,
        },
        "sessions": sessions,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        Path(args.out).write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if passed else 1


def compare(args) -> int:
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    failures = []
    comparisons = {}
    session_ids = sorted(set(baseline.get("sessions", {})) | set(candidate.get("sessions", {})))
    for session in session_ids:
        base = baseline.get("sessions", {}).get(session)
        test = candidate.get("sessions", {}).get(session)
        if base is None or test is None:
            failures.append(f"session_{session}_missing")
            continue
        base_p99 = base.get("distinct_gap_p99_s")
        test_p99 = test.get("distinct_gap_p99_s")
        delta = None if base_p99 is None or test_p99 is None else test_p99 - base_p99
        session_failures = []
        if delta is None or delta > args.max_p99_increase_s:
            session_failures.append("distinct_gap_p99_increase")
        if (test.get("max_distinct_gap_s") is None
                or test["max_distinct_gap_s"] > args.max_distinct_gap_s):
            session_failures.append("max_distinct_gap_s")
        if not test.get("passed", False):
            session_failures.append("candidate_verdict")
        if session_failures:
            failures.extend(f"session_{session}_{item}" for item in session_failures)
        comparisons[session] = {
            "baseline_p99_s": base_p99,
            "candidate_p99_s": test_p99,
            "p99_increase_s": delta,
            "failures": session_failures,
        }
    result = {"passed": not failures, "failures": failures, "sessions": comparisons}
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out:
        Path(args.out).write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["passed"] else 1


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--out", required=True)
    collect_parser.add_argument("--host", default="127.0.0.1")
    collect_parser.add_argument("--base-port", type=int, default=8066)
    collect_parser.add_argument("--port-step", type=int, default=10)
    collect_parser.add_argument("--sessions", type=int, default=3)
    collect_parser.add_argument("--duration", type=float, default=0.0)
    collect_parser.add_argument("--window-s", type=float, default=10.0)
    collect_parser.add_argument("--stall-s", type=float, default=0.5)
    collect_parser.add_argument("--ready-distinct", type=int, default=3)
    collect_parser.add_argument("--ready-file")
    collect_parser.add_argument("--stall-events")
    collect_parser.set_defaults(func=collect)

    report_parser = sub.add_parser("report")
    report_parser.add_argument("inputs", nargs="+")
    report_parser.add_argument("--out")
    report_parser.add_argument("--expected-sessions", type=int, default=3)
    report_parser.add_argument("--min-snapshot-hz", type=float, default=20.0)
    report_parser.add_argument("--min-distinct-hz", type=float, default=4.5)
    report_parser.add_argument("--max-distinct-gap-s", type=float, default=0.5)
    report_parser.set_defaults(func=report)

    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("--baseline", required=True)
    compare_parser.add_argument("--candidate", required=True)
    compare_parser.add_argument("--out")
    compare_parser.add_argument("--max-p99-increase-s", type=float, default=0.1)
    compare_parser.add_argument("--max-distinct-gap-s", type=float, default=0.5)
    compare_parser.set_defaults(func=compare)
    return parser


def main():
    args = build_parser().parse_args()
    raise SystemExit(args.func(args))


if __name__ == "__main__":
    main()
