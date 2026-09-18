#!/usr/bin/env python3
"""Sample a launcher's process tree and GPU state for policy-mirror forensics."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path


def _read(path, binary=False):
    try:
        return Path(path).read_bytes() if binary else Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeError):
        return b"" if binary else ""


def _processes():
    result = {}
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        stat = _read(entry / "stat")
        status = _read(entry / "status")
        if not stat or not status:
            continue
        # The command name can contain spaces and parentheses; fields after the final ')'
        # start with state then ppid.
        try:
            rest = stat[stat.rfind(")") + 2:].split()
            ppid = int(rest[1])
            utime = int(rest[11])
            stime = int(rest[12])
        except (ValueError, IndexError):
            continue
        fields = {}
        for line in status.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                fields[key] = value.strip()
        cmdline = _read(entry / "cmdline", binary=True).replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        ).strip()
        result[int(entry.name)] = {
            "pid": int(entry.name), "ppid": ppid, "state": rest[0],
            "cpu_ticks": utime + stime, "cmdline": cmdline,
            "rss_kb": int(fields.get("VmRSS", "0 kB").split()[0]),
            "threads": int(fields.get("Threads", "0")),
            "voluntary_ctxt": int(fields.get("voluntary_ctxt_switches", "0")),
            "involuntary_ctxt": int(fields.get("nonvoluntary_ctxt_switches", "0")),
            "wchan": _read(entry / "wchan").strip(),
        }
    return result


def _descendants(processes, roots):
    selected = set(int(pid) for pid in roots)
    changed = True
    while changed:
        changed = False
        for pid, process in processes.items():
            if process["ppid"] in selected and pid not in selected:
                selected.add(pid)
                changed = True
    return selected


def _gpu_state():
    command = [
        "nvidia-smi",
        "--query-gpu=index,memory.total,memory.used,utilization.gpu,utilization.memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=2)
    except (OSError, subprocess.SubprocessError):
        return None
    if completed.returncode:
        return {"error": completed.stderr.strip() or completed.stdout.strip()}
    rows = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 5:
            rows.append({
                "index": int(parts[0]), "memory_total_mb": int(parts[1]),
                "memory_used_mb": int(parts[2]), "gpu_util_pct": int(parts[3]),
                "memory_util_pct": int(parts[4]),
            })
    return rows


def _dump_stall(processes, selected, event, output_dir):
    stamp = f"{event.get('wall_t', time.time()):.3f}".replace(".", "_")
    target = output_dir / f"stall_S{int(event.get('session_index', -1)):02d}_{stamp}.json"
    dump = {"event": event, "processes": []}
    for pid in sorted(selected):
        process = processes.get(pid)
        if not process:
            continue
        if not any(token in process["cmdline"] for token in (
            "intervene_base/main.py", "intervention_vr_runtime.py", "policy_grid_viewer.py"
        )):
            continue
        stack = _read(Path("/proc") / str(pid) / "stack")
        dump["processes"].append({**process, "kernel_stack": stack})
    target.write_text(json.dumps(dump, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    parser.add_argument("--root-pid", action="append", type=int, required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--duration", type=float, default=0.0)
    parser.add_argument("--stall-events")
    parser.add_argument("--stall-dir")
    args = parser.parse_args()

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    stall_dir = Path(args.stall_dir or output.parent / "stalls")
    stall_dir.mkdir(parents=True, exist_ok=True)
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    deadline = time.time() + args.duration if args.duration > 0 else float("inf")
    stall_offset = 0
    clock_ticks = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
    previous = {}
    previous_t = None

    with output.open("a", encoding="utf-8", buffering=1) as handle:
        while not stopping and time.time() < deadline:
            now = time.time()
            processes = _processes()
            selected = _descendants(processes, args.root_pid)
            rows = []
            for pid in sorted(selected):
                process = processes.get(pid)
                if process is None:
                    continue
                cpu_pct = None
                if previous_t is not None and pid in previous and now > previous_t:
                    cpu_pct = 100.0 * (
                        process["cpu_ticks"] - previous[pid]
                    ) / clock_ticks / (now - previous_t)
                rows.append({**process, "cpu_pct": cpu_pct})
            handle.write(json.dumps({
                "type": "resources", "wall_t": now,
                "roots": args.root_pid, "processes": rows, "gpu": _gpu_state(),
            }, separators=(",", ":")) + "\n")
            previous = {pid: value["cpu_ticks"] for pid, value in processes.items()}
            previous_t = now

            if args.stall_events and Path(args.stall_events).exists():
                with Path(args.stall_events).open("r", encoding="utf-8") as stalls:
                    stalls.seek(stall_offset)
                    for line in stalls:
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        _dump_stall(processes, selected, event, stall_dir)
                    stall_offset = stalls.tell()
            time.sleep(max(0.05, args.interval))


if __name__ == "__main__":
    main()
