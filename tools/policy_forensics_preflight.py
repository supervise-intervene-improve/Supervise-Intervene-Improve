#!/usr/bin/env python3
"""Fail-fast checks for a policy-mirror forensic run."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path


PROCESS_MARKERS = (
    "intervene_base/main.py",
    "intervention_vr_runtime.py",
    "multi_session_launcher.py",
    "policy_grid_viewer.py",
)


def stale_processes():
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except OSError:
            continue
        if any(marker in command for marker in PROCESS_MARKERS):
            found.append({"pid": int(entry.name), "command": command.strip()})
    return found


def occupied_ports(args):
    ports = set()
    for session in range(args.sessions):
        offset = session * args.port_step
        ports.update((
            args.state_base_port + offset,
            args.command_base_port + offset,
            args.topic_base_port + offset,
            args.service_base_port + offset,
            args.runtime_command_base_port + offset,
        ))
    occupied = []
    for port in sorted(ports):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((args.bind_ip, port))
        except OSError as exc:
            occupied.append({"port": port, "error": str(exc)})
        finally:
            sock.close()
    return occupied


def gpu_status(min_free_mb):
    command = [
        "nvidia-smi", "--query-gpu=index,name,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError) as exc:
        return [], str(exc)
    if completed.returncode:
        return [], completed.stderr.strip() or completed.stdout.strip()
    devices = []
    for line in completed.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            continue
        total = int(parts[2])
        used = int(parts[3])
        devices.append({
            "index": int(parts[0]), "name": parts[1], "total_mb": total,
            "used_mb": used, "free_mb": total - used,
            "sufficient": total - used >= min_free_mb,
        })
    return devices, None


def gpu_backend_status(python_bin, open3d_path):
    environment = os.environ.copy()
    if open3d_path:
        environment["PYTHONPATH"] = open3d_path + (
            os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
        )
    command = [
        python_bin, "-c",
        "import open3d as o3d; raise SystemExit(0 if o3d.core.cuda.is_available() else 3)",
    ]
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=15, env=environment
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, str(exc)
    return completed.returncode == 0, completed.stderr.strip() or completed.stdout.strip() or None


def adb_status(explicit, package):
    if explicit:
        adb = explicit if os.path.sep in explicit and os.access(explicit, os.X_OK) else shutil.which(explicit)
    else:
        adb = shutil.which("adb")
    result = {
        "available": bool(adb), "path": adb, "devices": [],
        "selected_serial": None, "package": package, "package_installed": False,
        "error": None,
    }
    if not adb:
        result["error"] = (
            "adb executable not found; install package 'adb' or set "
            "FORENSICS_ADB=/absolute/path/to/adb"
        )
        return result
    try:
        completed = subprocess.run(
            [adb, "devices", "-l"], capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.SubprocessError) as exc:
        result["error"] = str(exc)
        return result
    if completed.returncode:
        result["error"] = completed.stderr.strip() or completed.stdout.strip()
        return result
    for line in completed.stdout.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2:
            result["devices"].append({"serial": parts[0], "state": parts[1]})
    authorized = [item for item in result["devices"] if item["state"] == "device"]
    if len(authorized) != 1:
        result["error"] = (
            f"expected exactly one authorized Quest, found {len(authorized)}; "
            "check USB authorization with 'adb devices -l'"
        )
        return result
    serial = authorized[0]["serial"]
    result["selected_serial"] = serial
    try:
        completed = subprocess.run(
            [adb, "-s", serial, "shell", "pm", "path", package],
            capture_output=True, text=True, timeout=15,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        result["error"] = str(exc)
        return result
    result["package_installed"] = completed.returncode == 0 and "package:" in completed.stdout
    if not result["package_installed"]:
        result["error"] = f"APK package {package!r} is not installed on {serial}"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=3)
    parser.add_argument("--bind-ip", default="127.0.0.1")
    parser.add_argument("--port-step", type=int, default=10)
    parser.add_argument("--state-base-port", type=int, default=8066)
    parser.add_argument("--command-base-port", type=int, default=8065)
    parser.add_argument("--topic-base-port", type=int, default=7741)
    parser.add_argument("--service-base-port", type=int, default=7740)
    parser.add_argument("--runtime-command-base-port", type=int, default=7746)
    parser.add_argument("--min-gpu-free-mb", type=int, default=2048)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--open3d-path", default="")
    parser.add_argument("--allow-running", action="store_true")
    parser.add_argument("--skip-gpu-check", action="store_true")
    parser.add_argument("--require-adb", action="store_true")
    parser.add_argument("--adb", default="")
    parser.add_argument(
        "--adb-package", default="com.anonymous.SIIMetaQuest3"
    )
    parser.add_argument("--out")
    args = parser.parse_args()

    running = [] if args.allow_running else stale_processes()
    ports = occupied_ports(args)
    devices, gpu_error = ([], None) if args.skip_gpu_check else gpu_status(args.min_gpu_free_mb)
    backend_ok, backend_error = (
        (True, None) if args.skip_gpu_check
        else gpu_backend_status(args.python_bin, args.open3d_path)
    )
    gpu_ok = args.skip_gpu_check or (
        gpu_error is None and devices and any(device["sufficient"] for device in devices)
        and backend_ok
    )
    adb = adb_status(args.adb, args.adb_package) if args.require_adb else {
        "required": False, "available": None, "error": None
    }
    adb_ok = not args.require_adb or adb.get("error") is None
    adb["required"] = args.require_adb
    result = {
        "passed": not running and not ports and gpu_ok and adb_ok,
        "running_processes": running,
        "occupied_ports": ports,
        "gpu": devices,
        "gpu_error": gpu_error,
        "gpu_pointcloud_backend": {"available": backend_ok, "error": backend_error},
        "gpu_check_skipped": args.skip_gpu_check,
        "min_gpu_free_mb": args.min_gpu_free_mb,
        "adb": adb,
    }
    rendered = json.dumps(result, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
