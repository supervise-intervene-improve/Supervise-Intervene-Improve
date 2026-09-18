#!/usr/bin/env python3
"""Watch and grade Linux-to-Quest selection and streaming delivery."""

from __future__ import annotations

import argparse
import glob
import json
import re
import signal
import statistics
import sys
import time
from pathlib import Path


SESSION_RE = re.compile(r"\[S(?P<session>\d+)\]\s+(?P<body>.*)")
FIRST_PC_RE = re.compile(r"First PC publish \(async\): cam=(?P<cam>\w+)")
PC_VISIT_RE = re.compile(
    r"\[PointCloudVisit\]\s+\S+\s+duration_s=(?P<duration>[0-9.]+)\s+"
    r"counts=(?P<counts>\{.*?\})\s+rates_hz=(?P<rates>\{.*\})"
)


def _append(path, row):
    with path.open("a", encoding="utf-8", buffering=1) as handle:
        handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def _write_ready(path, payload):
    if path.exists():
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def watch(args):
    runtime_dir = Path(args.runtime_dir)
    quest_path = Path(args.quest)
    events_path = Path(args.events)
    events_path.parent.mkdir(parents=True, exist_ok=True)
    positions = {}
    backend = {name: set() for name in ("started", "cmd", "policy", "rgb")}
    panel_sessions = set()
    display_seen = False
    active = {}
    last_exited_visit = {}
    current_session = {"value": None}
    visit_counts = [0] * args.sessions
    stop = {"value": False}
    measuring = {"value": False}

    def halt(_signum, _frame):
        stop["value"] = True

    signal.signal(signal.SIGINT, halt)
    signal.signal(signal.SIGTERM, halt)

    def event(kind, session=None, **extra):
        if not measuring["value"]:
            return
        row = {"wall_t": time.time(), "event": kind}
        if session is not None:
            row["session_index"] = session
        row.update(extra)
        _append(events_path, row)

    def runtime_line(line):
        match = SESSION_RE.search(line)
        if not match:
            return
        session = int(match.group("session"))
        body = match.group("body")
        if session >= args.sessions:
            return
        if "[Integration] Runtime started" in body:
            backend["started"].add(session)
        if "[CmdListener] PULL bound" in body:
            backend["cmd"].add(session)
        if "[PolicyStateMirror] applied seq=" in body:
            backend["policy"].add(session)
        if "[Integration] First RGB publish" in body:
            backend["rgb"].add(session)
        if "[CmdListener] ENTER_SINGLE -> set_pending" in body and session not in active:
            previous = current_session["value"]
            if previous is not None and previous != session:
                event(
                    "selection_overlap", session, other_session=previous,
                    other_visit=active.get(previous),
                )
            visit_counts[session] += 1
            visit = visit_counts[session]
            active[session] = visit
            current_session["value"] = session
            event("enter", session, visit=visit)
        elif "[CmdListener] EXIT_SINGLE -> set_pending" in body and session in active:
            visit = active.pop(session)
            last_exited_visit[session] = visit
            event("exit", session, visit=visit)
            if current_session["value"] == session:
                current_session["value"] = None
        visit_summary = PC_VISIT_RE.search(body)
        if visit_summary and session in last_exited_visit:
            try:
                event(
                    "linux_pc_visit", session,
                    visit=last_exited_visit[session],
                    duration_s=float(visit_summary.group("duration")),
                    counts=json.loads(visit_summary.group("counts")),
                    rates_hz=json.loads(visit_summary.group("rates")),
                )
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        pc = FIRST_PC_RE.search(body)
        if pc and session in active:
            event("linux_pc_publish", session, visit=active[session], camera=pc.group("cam"))

    def quest_row(row):
        nonlocal display_seen
        source = row.get("source")
        if source == "display":
            display_seen = True
        elif source == "rgb_panel" and row.get("session_index") is not None:
            panel_sessions.add(int(row["session_index"]))
        elif source == "pointcloud" and current_session["value"] is not None:
            observed_session = row.get("session_index")
            # Exactly one session should be active. Associate even a stale heartbeat
            # with that expected visit so an old-session cloud cannot disappear from
            # the forensic record merely because its identity is wrong.
            expected_session = current_session["value"]
            if isinstance(expected_session, int) and expected_session in active:
                payload = dict(row)
                payload.pop("wall_t", None)
                payload.pop("source", None)
                payload.pop("session_index", None)
                event(
                    "quest_pc", expected_session, visit=active[expected_session],
                    observed_session_index=observed_session, **payload,
                )

    while not stop["value"]:
        if not measuring["value"] and Path(args.measure_file).exists():
            active.clear()
            current_session["value"] = None
            visit_counts[:] = [0] * args.sessions
            measuring["value"] = True
        for filename in glob.glob(str(runtime_dir / "multi_session_*.log")):
            path = Path(filename)
            offset = positions.get(filename, 0)
            try:
                with path.open("r", encoding="utf-8", errors="replace") as handle:
                    handle.seek(offset)
                    for line in handle:
                        runtime_line(line)
                    positions[filename] = handle.tell()
            except OSError:
                pass
        if quest_path.exists():
            key = str(quest_path)
            offset = positions.get(key, 0)
            try:
                with quest_path.open("r", encoding="utf-8", errors="replace") as handle:
                    handle.seek(offset)
                    for line in handle:
                        try:
                            quest_row(json.loads(line))
                        except json.JSONDecodeError:
                            continue
                    positions[key] = handle.tell()
            except OSError:
                pass
        if all(len(values) == args.sessions for values in backend.values()):
            _write_ready(Path(args.backend_ready), {key: sorted(value) for key, value in backend.items()})
        if display_seen and len(panel_sessions) >= args.sessions:
            _write_ready(Path(args.quest_ready), {
                "display": True, "panel_sessions": sorted(panel_sessions),
            })
        time.sleep(args.poll_s)
    return 0


def _load_jsonl(path):
    rows = []
    if not Path(path).exists():
        return rows
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def _metric_rates(runtime_dir, sessions):
    result = {index: {cam: [] for cam in ("top", "right", "left")} for index in range(sessions)}
    for filename in glob.glob(str(Path(runtime_dir) / "metrics_S*.jsonl")):
        for row in _load_jsonl(filename):
            session = row.get("session_index")
            if session not in result or not row.get("labels", {}).get("pc_topics"):
                continue
            rates = row.get("rates_hz", {})
            for cam in result[session]:
                value = rates.get(f"{cam}/pc")
                if isinstance(value, dict) and isinstance(value.get("hz"), (int, float)):
                    result[session][cam].append(float(value["hz"]))
    return result


def _max_numbers(rows, key):
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    return max(values) if values else None


def _median_numbers(rows, key):
    values = [float(row[key]) for row in rows if isinstance(row.get(key), (int, float))]
    return statistics.median(values) if values else None


def report(args):
    events = _load_jsonl(args.events)
    quest = _load_jsonl(args.quest)
    policy = json.loads(Path(args.policy_verdict).read_text(encoding="utf-8"))
    rates = _metric_rates(args.runtime_dir, args.sessions)
    sessions = {}
    all_failures = []
    overlaps = [row for row in events if row.get("event") == "selection_overlap"]
    if overlaps:
        all_failures.append(f"{len(overlaps)} selection overlap(s) before EXIT_SINGLE")
    for session in range(args.sessions):
        enters = [row for row in events if row.get("event") == "enter" and row.get("session_index") == session]
        exits = [row for row in events if row.get("event") == "exit" and row.get("session_index") == session]
        visits = []
        failures = []
        if len(enters) < args.visits_per_session:
            failures.append(f"selection coverage {len(enters)}/{args.visits_per_session}")
        for enter in enters:
            visit_id = enter["visit"]
            publish = [row for row in events if row.get("event") == "linux_pc_publish" and row.get("session_index") == session and row.get("visit") == visit_id]
            visit_rate_rows = [
                row for row in events
                if row.get("event") == "linux_pc_visit"
                and row.get("session_index") == session
                and row.get("visit") == visit_id
            ]
            pc_rows = [row for row in events if row.get("event") == "quest_pc" and row.get("session_index") == session and row.get("visit") == visit_id]
            cameras = sorted({row.get("camera") for row in publish if row.get("camera")})
            identities = [row for row in pc_rows if row.get("identity_confirmed") is True and row.get("expected_session_index") == session]
            visit_failures = []
            if not any(row.get("visit") == visit_id for row in exits):
                visit_failures.append("missing EXIT_SINGLE")
            if set(cameras) != {"left", "right", "top"}:
                visit_failures.append(f"Linux first-publish cameras={cameras}")
            visit_rates = visit_rate_rows[-1].get("rates_hz", {}) if visit_rate_rows else {}
            if visit_rate_rows:
                for cam in ("top", "right", "left"):
                    value = visit_rates.get(cam)
                    if not isinstance(value, (int, float)) or value < args.min_pc_hz:
                        visit_failures.append(
                            f"Linux {cam}/pc visit rate {value} < {args.min_pc_hz}Hz"
                        )
            if not args.skip_quest:
                if not identities:
                    visit_failures.append("no matching Quest identity heartbeat")
                if not any((row.get("uploads_total") or 0) > 0 and (row.get("rx_total") or 0) > 0 for row in identities):
                    visit_failures.append("no fresh Quest receive/upload")
                accepted_at = min(
                    (row["wall_t"] for row in identities), default=float("inf")
                )
                stale = [
                    row for row in pc_rows
                    if row.get("wall_t", 0) >= accepted_at
                    and (
                        row.get("observed_session_index") not in (None, session)
                        or row.get("identity_confirmed") is False
                    )
                ]
                if stale:
                    visit_failures.append(f"{len(stale)} stale/mismatched Quest heartbeat(s)")
                for label, key in (
                    ("receive", "rx_hz"), ("draw", "draw_hz"),
                    ("upload", "upload_hz"),
                ):
                    value = _median_numbers(identities, key)
                    if value is None or value < args.min_pc_hz:
                        visit_failures.append(
                            f"Quest {label} rate {value} < {args.min_pc_hz}Hz"
                        )
            latency = {
                "linux_first_publish_ms": min(
                    ((row["wall_t"] - enter["wall_t"]) * 1000.0 for row in publish),
                    default=None,
                ),
                "identity_ms": _max_numbers(identities, "identity_latency_ms"),
                "first_payload_ms": _max_numbers(identities, "first_payload_latency_ms"),
                "first_upload_ms": _max_numbers(identities, "first_upload_latency_ms"),
            }
            for name, value in latency.items():
                if value is not None and value > args.max_activation_ms:
                    visit_failures.append(f"{name}={value:.1f}ms")
            visits.append({
                "visit": visit_id, "cameras": cameras, "quest_heartbeats": len(pc_rows),
                "quest_hz_median": {
                    key: _median_numbers(identities, key)
                    for key in ("rx_hz", "draw_hz", "upload_hz")
                },
                "linux_hz": visit_rates,
                "latency": latency, "failures": visit_failures,
            })
            failures.extend(f"visit {visit_id}: {item}" for item in visit_failures)
        visit_rate_values = {
            cam: [
                float(row.get("rates_hz", {}).get(cam))
                for row in events
                if row.get("event") == "linux_pc_visit"
                and row.get("session_index") == session
                and isinstance(row.get("rates_hz", {}).get(cam), (int, float))
            ]
            for cam in ("top", "right", "left")
        }
        camera_rates = {}
        for cam in ("top", "right", "left"):
            values = visit_rate_values[cam] or rates[session][cam]
            camera_rates[cam] = statistics.median(values) if values else None
        for cam, value in camera_rates.items():
            if value is None or value < args.min_pc_hz:
                failures.append(f"Linux {cam}/pc rate {value} < {args.min_pc_hz}Hz")
        sessions[str(session)] = {
            "passed": not failures, "enters": len(enters), "exits": len(exits),
            "camera_hz_median": camera_rates, "visits": visits, "failures": failures,
        }
        all_failures.extend(f"S{session:02d}: {item}" for item in failures)

    # Grade point-cloud cadence only while a measured visit is active. Grid-time
    # heartbeats may legitimately describe an idle loader.
    pc_rows = [
        row for row in events
        if row.get("event") == "quest_pc" and row.get("identity_confirmed") is True
    ]
    display = [row for row in quest if row.get("source") == "display"]
    panels = [row for row in quest if row.get("source") == "rgb_panel"]
    quest_failures = []
    quest_rates = {}
    if not args.skip_quest:
        for label, key in (("receive", "rx_hz"), ("draw", "draw_hz"), ("upload", "upload_hz")):
            median = _median_numbers(pc_rows, key)
            quest_rates[label] = median
            if median is None or median < args.min_pc_hz:
                quest_failures.append(f"Quest {label} median {median} < {args.min_pc_hz}Hz")
    panel_rates = {}
    for session in range(args.sessions):
        values = [float(row["rx_hz"]) for row in panels if row.get("session_index") == session and isinstance(row.get("rx_hz"), (int, float))]
        panel_rates[str(session)] = statistics.median(values) if values else None
        if not args.skip_quest and (panel_rates[str(session)] is None or panel_rates[str(session)] < args.min_grid_hz):
            quest_failures.append(f"S{session:02d} grid rate {panel_rates[str(session)]} < {args.min_grid_hz}Hz")
    display_fps = [float(row["fps"]) for row in display if isinstance(row.get("fps"), (int, float))]
    display_median = statistics.median(display_fps) if display_fps else None
    display_stale_max = _max_numbers(display, "stale")
    if not args.skip_quest and (display_median is None or display_median < args.min_display_fps):
        quest_failures.append(f"display FPS median {display_median} < {args.min_display_fps}")
    source_age_max = _max_numbers(pc_rows, "last_payload_age_s")
    if not args.skip_quest and (source_age_max is None or source_age_max > args.max_source_age_s):
        quest_failures.append(f"point-cloud source age max {source_age_max} > {args.max_source_age_s}s")
    all_failures.extend(quest_failures)
    result = {
        "passed": bool(policy.get("passed")) and not all_failures,
        "policy_passed": bool(policy.get("passed")), "policy": policy,
        "sessions": sessions,
        "quest": {
            "measured": not args.skip_quest,
            "pointcloud_rows": len(pc_rows), "display_rows": len(display),
            "panel_rows": len(panels), "panel_hz_median": panel_rates,
            "pointcloud_hz_median": quest_rates,
            "display_fps_median": display_median,
            "display_stale_max": display_stale_max,
            "source_age_max_s": source_age_max,
            "failures": quest_failures,
        },
        "failures": all_failures,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    Path(args.out).write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    watcher = sub.add_parser("watch")
    watcher.add_argument("--runtime-dir", required=True)
    watcher.add_argument("--quest", required=True)
    watcher.add_argument("--events", required=True)
    watcher.add_argument("--backend-ready", required=True)
    watcher.add_argument("--quest-ready", required=True)
    watcher.add_argument("--measure-file", required=True)
    watcher.add_argument("--sessions", type=int, required=True)
    watcher.add_argument("--poll-s", type=float, default=0.1)
    grader = sub.add_parser("report")
    grader.add_argument("--events", required=True)
    grader.add_argument("--quest", required=True)
    grader.add_argument("--runtime-dir", required=True)
    grader.add_argument("--policy-verdict", required=True)
    grader.add_argument("--out", required=True)
    grader.add_argument("--sessions", type=int, required=True)
    grader.add_argument("--visits-per-session", type=int, default=2)
    grader.add_argument("--min-pc-hz", type=float, default=15.0)
    grader.add_argument("--min-grid-hz", type=float, default=8.0)
    grader.add_argument("--min-display-fps", type=float, default=70.0)
    grader.add_argument("--max-activation-ms", type=float, default=1000.0)
    grader.add_argument("--max-source-age-s", type=float, default=0.5)
    grader.add_argument("--skip-quest", action="store_true")
    args = parser.parse_args()
    return watch(args) if args.command == "watch" else report(args)


if __name__ == "__main__":
    sys.exit(main())
