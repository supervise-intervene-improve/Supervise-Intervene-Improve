r"""
multi_session_launcher.py — Launch up to 15 parallel intervention sessions
for the VR multi-session grid selector.

Each session runs a separate intervention_vr_runtime.py process on an offset
port pair so that the Quest selector scene can subscribe to all of them
simultaneously:

  Session i: topic_port  = base_topic_port  + i * port_step   (default 7741 + i*10)
             service_port = base_service_port + i * port_step  (default 7740 + i*10)

Session 0 is intentionally backward-compatible with the standard single-session
launch (ports 7741/7740).

Usage (PowerShell):
  .\.venv\Scripts\python.exe .\SimPublisher\sii\integration_v1\multi_session_launcher.py `
    --trajectories `
      .\intervene_base\teleop_logs\p1_traj_1772714528.npz `
      .\intervene_base\teleop_logs\p1_traj_1772714528.npz `
    --xml .\intervene_base\mujoco_scenes\working_scenes\with_soft_gripper\sii_scene_table_T_shape_multiwindow_fast.xml `
    --host 192.168.0.208 `
    --unity_node MQ3-2 `
    --bind_ip 0.0.0.0 `
    --fps 10 `
    --w 480 --h 360 `
    --cams front `
    --rgb_cams front `
    --visible_geoms_groups 2 `
    --start_paused

Usage (Linux bash):
  .venv/bin/python SimPublisher/sii/integration_v1/multi_session_launcher.py \\
    --trajectories \\
      intervene_base/teleop_logs/p1_traj_1772714528.npz \\
      intervene_base/teleop_logs/p1_traj_1772714528.npz \\
    --xml intervene_base/mujoco_scenes/working_scenes/with_soft_gripper/sii_scene_table_T_shape_multiwindow_fast.xml \\
    --host 192.168.0.208 \\
    --unity_node MQ3-2 \\
    --bind_ip 0.0.0.0 \\
    --fps 10 --w 480 --h 360 \\
    --cams front --rgb_cams front \\
    --start_paused
"""

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict

THIS_FILE  = Path(__file__).resolve()
RUNTIME    = THIS_FILE.parent / "intervention_vr_runtime.py"
PYTHON_EXE = sys.executable


def build_launcher_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Launch multiple intervention sessions for the VR grid selector.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Sessions ---
    ap.add_argument(
        "--trajectories", nargs="+", required=True,
        help="Paths to trajectory .npz files (one per session, max 15).",
    )
    ap.add_argument(
        "--windows", type=int, default=None,
        help=(
            "Number of active session windows to launch (1, 3, 6, 9, 12, or 15). "
            "Replicates/truncates --trajectories to exactly this many sessions so each "
            "group of 3 fills one column in the VR selector grid. If omitted, the "
            "session count equals the number of --trajectories passed."
        ),
    )
    ap.add_argument(
        "--grid_capacity", type=int, default=0,
        help=(
            "Selector panel capacity advertised separately from active sessions. "
            "0 matches the active count; Study C uses 9 so inactive cells are visible "
            "but do not get runtime workers or sockets."
        ),
    )
    ap.add_argument(
        "--single", action="store_true",
        help=(
            "Standalone single-session mode: launch exactly ONE session and bypass "
            "the multi-window selector grid. The discovery beacon advertises "
            "n_sessions=1 so the Quest auto-enters single view, and the session is "
            "launched with --start_in_single_view (ACTIVE from frame 0). Overrides "
            "--windows. Use only the first --trajectories entry."
        ),
    )
    ap.add_argument(
        "--base_topic_port",   type=int, default=7741,
        help="Topic port for session 0 (subsequent sessions offset by --port_step).",
    )
    ap.add_argument(
        "--base_service_port", type=int, default=7740,
        help="Service port for session 0.",
    )
    ap.add_argument(
        "--port_step", type=int, default=10,
        help="Port increment between consecutive sessions.",
    )

    # --- Forwarded to runtime (required) ---
    ap.add_argument("--xml",        required=True, help="MuJoCo XML scene file.")
    ap.add_argument("--host",       required=True, help="PC IP (same for all sessions).")
    ap.add_argument("--unity_node", required=True, help="Quest headset node name.")

    # --- Forwarded to runtime (optional, sensible thumbnail defaults) ---
    ap.add_argument("--bind_ip",   default="0.0.0.0")
    ap.add_argument("--fps",       type=float, default=60.0,
                    help="Render FPS for all sensors (RGB + PC alike). Default 60 "
                         "matches Quest display refresh. Lower if Wi-Fi saturates "
                         "(symptoms: published < enqueued, pending_topics > 0 in Python "
                         "log; rx lagging in Unity heartbeat). Sensible fallbacks: 30 "
                         "(half-rate), 15 (light), 10 (pre-iteration default).")
    ap.add_argument("--w",         type=int,   default=480,
                    help="Camera render width for thumbnails.")
    ap.add_argument("--h",         type=int,   default=360,
                    help="Camera render height for thumbnails.")
    ap.add_argument("--cams",      nargs="*",  default=["front"],
                    help="Cameras to render (thumbnail mode: 'front' is sufficient).")
    ap.add_argument("--rgb_cams",  nargs="*",  default=["front"])
    ap.add_argument("--visible_geoms_groups", nargs="+", type=int, default=[2])
    ap.add_argument("--start_paused", action="store_true", default=False)
    ap.add_argument("--replay_speed", type=float, default=0.45)
    ap.add_argument(
        "--post_replan_mode",
        choices=["preview_replay", "policy_handoff"],
        default="preview_replay",
        help="Forwarded to runtime: preview stitched replay for testing, or stop at policy handoff.",
    )
    ap.add_argument("--jpg_quality",  type=int,   default=75,
                    help="JPEG quality for thumbnails (lower = smaller payload).")
    ap.add_argument("--transport",    default="wifi",  choices=["wifi", "usb"])
    ap.add_argument("--usb_auto_reverse", action="store_true")
    ap.add_argument(
        "--extra_runtime_args", type=str, default="",
        help="Extra flags forwarded verbatim to each intervention_vr_runtime.py call. "
             "Pass as a SINGLE QUOTED STRING (split on whitespace internally). "
             "Example: --extra_runtime_args \"--perf_log --render_shadows\". "
             "Single arg: --extra_runtime_args \"--perf_log\". "
             "Use this form because argparse nargs='*' treats values starting with -- as new flags.",
    )
    ap.add_argument(
        "--policy_state_base_port", type=int, default=0,
        help="Per-session INDEPENDENT policy mirroring: session i subscribes to the policy "
             "state stream at base + i*policy_port_step. 0 = disabled (legacy: any policy "
             "ports arrive identically for all sessions via --extra_runtime_args, which made "
             "every window mirror the SAME single policy).",
    )
    ap.add_argument(
        "--policy_cmd_base_port", type=int, default=0,
        help="Per-session policy command forwarding: session i pushes INTERVENE/CANCEL to "
             "base + i*policy_port_step. 0 = disabled.",
    )
    ap.add_argument("--policy_port_step", type=int, default=10,
                    help="Port stride between per-session policy instances.")
    ap.add_argument("--policy_host", default="127.0.0.1",
                    help="Host of the policy instances (they run on this PC).")
    ap.add_argument(
        "--with_mujoco_publisher", action="store_true",
        help="Enable XR scene mesh publishing for thumbnail sessions (default OFF). "
             "By default, thumbnail sessions skip the MuJoCo scene publisher to prevent "
             "the robot scene from loading in the VR selector scene.",
    )
    ap.add_argument(
        "--with_xr_device", action="store_true",
        help="Enable MetaQuest3 VR input (A/B/X/Y controls) for all sessions (default OFF). "
             "By default, thumbnail sessions skip MetaQuest3 to prevent "
             "15×3=45 simultaneous Quest subscriptions that cause frame snapping.",
    )
    ap.add_argument(
        "--with_pc", action="store_true",
        help="Enable point clouds and cameras (top/right/left; wrist disabled for 15-window stability). "
             "Use when you need point clouds in SIIScene_InterventionV1 "
             "after session selection. Overrides --cams and --rgb_cams. "
             "Increases bandwidth but gives full point cloud quality.",
    )
    ap.add_argument(
        "--pc_active_only",
        action="store_true",
        help=(
            "With --with_pc, keep idle grid sessions top-RGB-only and publish "
            "full point clouds only when a session is active/selected. This "
            "preserves PC quality in single view while making 15 windows stable."
        ),
    )
    ap.add_argument(
        "--mc_active", action="store_true",
        help=(
            "Declare that this run is the motion-controller condition (MC_ACTIVE=1). "
            "Advertised in the discovery beacon ONLY — the launcher itself does nothing "
            "else with it. The Quest uses it to reserve the RIGHT controller for the "
            "motion controller while an intervention is live, so a clutch pull cannot "
            "also select a grid panel or a risk bar and switch sessions mid-takeover. "
            "Left off, the headset keeps its normal right-controller navigation (which "
            "is correct for the KT and FACTR conditions)."
        ),
    )
    ap.add_argument(
        "--rgb_mode", action="store_true",
        help=(
            "Switch the single-session view (SIIScene_InterventionV1) from point "
            "clouds to a 4-camera diamond RGB panel layout (top/left/right/wrist), "
            "anchored to the same controllable scene anchor. Mutually exclusive "
            "with --with_pc — the point cloud pipeline is never engaged when this "
            "is set. Overrides --cams and --rgb_cams. Advertised in the discovery "
            "beacon so the Quest selector knows which mode to render."
        ),
    )
    ap.add_argument(
        "--selected_sensor_activation",
        action="store_true",
        help=(
            "Forward --selected_sensor_activation to each runtime so ENTER_SINGLE/"
            "EXIT_SINGLE, not subscriber count, controls ACTIVE/IDLE sensor mode."
        ),
    )
    ap.add_argument(
        "--pc_stride", type=int, default=2,
        help="Point cloud stride when --with_pc is set (default 2). "
             "Lower = denser PC. 2 gives ~4x more points than the old default 4.",
    )
    ap.add_argument(
        "--pc_sampling", default="grid", choices=["grid", "stable_random"],
        help="Point cloud sampling mode when --with_pc is set (default 'grid'). "
             "'grid' is a cached deterministic stride sample — SAME density/count as "
             "'stable_random' but ~4x fewer depth-candidate lookups per point in the PC "
             "build (stable_random evaluates 4 candidate pixels per sample). 'grid' is "
             "quality-neutral and frees the PC worker thread to keep up. Use "
             "'stable_random' only if you specifically want jittered sampling.",
    )
    ap.add_argument(
        "--pc_max_points", type=int, default=200000,
        help="Max point cloud points per source when --with_pc is set (default 200000). "
             "Bumped 2x from old 100000 since we're no longer running MujocoPublisher.",
    )
    ap.add_argument(
        "--pc_fps_cap", type=float, default=60.0,
        help="Maximum fps cap when --with_pc is set (default 60 — Quest display refresh). "
             "Only enforced as a CEILING on --fps. The actual publish rate is still "
             "min(--fps, --pc_fps_cap), so at default --fps=10 you get 10 Hz unless you "
             "also raise --fps. Bandwidth scales linearly: 60 Hz × 200k pts × 3 cams ≈ "
             "216 MB/s per active session over Wi-Fi (over saturation). If you push Hz "
             "high, dial down --pc_max_points or raise --pc_stride to keep the per-frame "
             "size sane (Quest drops frames silently when its NetMQ recv buffer overflows).",
    )
    ap.add_argument(
        "--pc_width", type=int, default=640,
        help="Camera render width when --with_pc is set (default 640). "
             "Bumped from old 480 for better depth resolution.",
    )
    ap.add_argument(
        "--pc_height", type=int, default=480,
        help="Camera render height when --with_pc is set (default 480). "
             "Bumped from old 360 for better depth resolution.",
    )
    ap.add_argument(
        "--fps_idle", type=float, default=10.0,
        help="IDLE publish rate per session when peer_count is below "
             "--active_peer_threshold. Drops the 14 non-selected sessions to a "
             "low Hz, freeing CPU for the 1 active session at --fps. Set to "
             "--fps to disable adaptive behavior. Default 10 Hz; pair with "
             "default --fps=60 for ~6x global CPU savings when 1-of-N is selected.",
    )
    ap.add_argument(
        "--active_peer_threshold", type=int, default=2,
        help="Peer count at which a session boosts from --fps_idle to --fps. "
             "Defaults to 2: selector grid contributes 1 peer per session; "
             "entering single view adds the PC + RGB subscribers, pushing past 2.",
    )

    # --- Status file ---
    ap.add_argument(
        "--status_file", default="multi_session_status.json",
        help="Path to write the running-sessions status JSON.",
    )
    ap.add_argument(
        "--log_dir", default="session_logs",
        help="Directory for per-session stdout/stderr log files (default: ./session_logs/).",
    )
    ap.add_argument(
        "--no_session_logs", action="store_true",
        help="Disable per-session log files (output goes to terminal only).",
    )
    ap.add_argument(
        "--unified_log", action="store_true",
        help=(
            "Write all session output to a single multi_session_<ts>.log with [S{i}] line "
            "prefixes instead of separate per-session files. Much easier to correlate "
            "events (e.g. which session got selected, when peers reconnect) across sessions."
        ),
    )
    # --- Performance metrics (PART-1) ---
    ap.add_argument(
        "--metrics", action="store_true",
        help=(
            "Enable windowed performance metrics per session. Each session writes "
            "<log_dir>/metrics_S<ii>_<ts>.jsonl with avg/p50/p95/p99 latencies and per-topic "
            "publish rates. Join with the Quest-side collector via tools/perf_report.py."
        ),
    )
    ap.add_argument(
        "--metrics_window_s", type=float, default=10.0,
        help="Seconds per metrics window (default 10).",
    )
    ap.add_argument(
        "--metrics_summary", action="store_true",
        help="Also print the [PerfSummary] block per window into the session log.",
    )
    ap.add_argument(
        "--max_restarts", type=int, default=3,
        help=(
            "When a session subprocess exits (crash or normal), the launcher respawns it "
            "with the same args up to this many times before giving up. Set 0 to disable "
            "auto-restart entirely. Default 3."
        ),
    )
    ap.add_argument(
        "--max_sessions", type=int, default=13,
        help=(
            "Cap on parallel MuJoCo sessions. Default 13 was the verified safe ceiling "
            "on the current laptop — session 14+ historically hit MemoryError during "
            "trimesh geometry import. With MujocoPublisher removed (this iteration's "
            "thumbnail-only mode), each session is lighter; you may be able to push to "
            "15+. Raise carefully and watch Task Manager: if RAM headroom drops under "
            "~1 GB, the next session's trimesh import will OOM. Excess trajectories "
            "are trimmed silently."
        ),
    )
    ap.add_argument(
        "--no_orphan_kill", action="store_true",
        help=(
            "Skip the startup scan that kills leftover intervention_vr_runtime.py "
            "processes. Use only if you're intentionally running another launcher "
            "or single-session Python process in parallel."
        ),
    )
    ap.add_argument(
        "--discovery_port", type=int, default=8720,
        help=(
            "UDP port the launcher broadcasts a discovery beacon on (default 8720; "
            "deliberately NOT SimPub's own 7720 multicast discovery port). The Quest "
            "selector (PublisherDiscoveryListener) listens here and auto-sets its "
            "publisher IP + ports from the beacon, so the IP is NOT hardcoded in the "
            "Unity scene — it comes from this --host. Must match the Unity discoveryPort."
        ),
    )
    ap.add_argument(
        "--no_discovery", action="store_true",
        help=(
            "Disable the UDP discovery beacon. The Quest selector then falls back to the "
            "IP serialized in the scene. Use only if UDP broadcast is blocked on your network."
        ),
    )

    return ap


def _check_host_ip(host: str) -> bool:
    """Return True if host IP is currently assigned to a local network interface."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind((host, 0))
        s.close()
        return True
    except OSError:
        return False


def _start_discovery_broadcaster(host: str, base_topic_port: int, port_step: int,
                                 n_sessions: int, discovery_port: int,
                                 rgb_mode: bool = False, mc_active: bool = False,
                                 grid_capacity: int = 0):
    """Broadcast a UDP discovery beacon ~once/second so the Quest selector can
    auto-discover this launcher's IP + port scheme instead of using a hardcoded
    value baked into the Unity scene.

    The IP therefore lives only in --host: whatever PC runs the launcher advertises
    itself, and the headset points at it automatically.

    rgb_mode is advertised so the Quest (a separate device that cannot read the
    shell env var that selected this mode) knows whether the selected session's
    single view should render the diamond RGB camera panels instead of point
    clouds.

    mc_active is advertised for the same reason: under MC the Quest's RIGHT controller
    is the operator's hand on the arm, so for the duration of an intervention the
    headset must reserve it for the motion controller alone and stop reading it as a
    UI input (panel select, risk-bar navigate, B/A). See
    MotionControllerModeManager.McControlsLocked. The lock is scoped to MC precisely
    because in the KT and FACTR conditions the right controller drives nothing, and
    switching sessions there should keep working (and keep auto-cancelling).

    Returns (thread, stop_event). The thread is a daemon, so it dies with the
    process even if stop_event is never set; set it for a clean shutdown.
    """
    stop_event = threading.Event()
    payload = json.dumps({
        "app":             "SII_MULTI",
        "ip":              host,
        "base_topic_port": base_topic_port,
        "port_step":       port_step,
        "n_sessions":      n_sessions,
        "grid_capacity":   max(int(n_sessions), int(grid_capacity or n_sessions)),
        "rgb_mode":        bool(rgb_mode),
        "mc_active":       bool(mc_active),
        # NOTE: ood_owner/ood_enabled retired with the OOD auto-pause supervisor. A
        # deployed APK that still reads them sees them absent, which JsonUtility maps to
        # null/false — i.e. "supervisor off", the safe direction. No rebuild required.
    }).encode("utf-8")

    # Global broadcast plus the subnet-directed broadcast derived from --host
    # (assumes /24). The subnet-directed target is important on multi-NIC PCs
    # (e.g. Ethernet + WiFi): it routes out the interface that owns --host.
    targets = ["255.255.255.255"]
    octets = host.split(".")
    if len(octets) == 4:
        targets.append(f"{octets[0]}.{octets[1]}.{octets[2]}.255")

    def _loop():
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        # Bind to --host so beacons egress the correct interface on multi-NIC hosts.
        try:
            sock.bind((host, 0))
        except OSError:
            pass
        try:
            while not stop_event.is_set():
                for tgt in targets:
                    try:
                        sock.sendto(payload, (tgt, discovery_port))
                    except OSError:
                        pass
                stop_event.wait(1.0)
        finally:
            try:
                sock.close()
            except Exception:
                pass

    t = threading.Thread(target=_loop, name="DiscoveryBeacon", daemon=True)
    t.start()
    print(f"[Launcher] Discovery beacon broadcasting on UDP {discovery_port} "
          f"(ip={host} base_topic={base_topic_port} step={port_step} "
          f"n={n_sessions} rgb_mode={bool(rgb_mode)} mc_active={bool(mc_active)} "
          f"grid_capacity={max(int(n_sessions), int(grid_capacity or n_sessions))} "
          f"targets={targets}).")
    return t, stop_event


def _managed_ports(base_topic_port: int, base_service_port: int, port_step: int, num_sessions: int) -> list:
    ports = []
    for session_index in range(num_sessions):
        topic_port = base_topic_port + session_index * port_step
        service_port = base_service_port + session_index * port_step
        cmd_port = topic_port + 5
        ports.extend([topic_port, service_port, cmd_port])
    return sorted(set(int(port) for port in ports))


def _read_proc_cmdline(pid: int) -> str:
    try:
        raw = Path(f"/proc/{int(pid)}/cmdline").read_bytes()
        return raw.replace(b"\x00", b" ").decode(errors="replace").strip()
    except Exception:
        return ""


def _pid_cmdline(pid: int) -> str:
    try:
        import psutil
        proc = psutil.Process(int(pid))
        return " ".join(str(part) for part in (proc.cmdline() or []))
    except Exception:
        return _read_proc_cmdline(int(pid))


def _is_runtime_pid(pid: int) -> bool:
    return "intervention_vr_runtime.py" in _pid_cmdline(int(pid))


def _is_repo_launcher_cmdline(cmdline: str) -> bool:
    if "multi_session_launcher.py" not in cmdline:
        return False
    # Accept absolute-path invocations from this checkout, relative-path invocations
    # (run_multi_window_robot.sh uses a relative path so the absolute path is absent),
    # and bare-filename invocations. Any multi_session_launcher.py on this machine
    # is ours; --no_orphan_kill opts out if that assumption ever breaks.
    return True


def _runtime_cmdline_ports(cmdline: str) -> set:
    ports = set()
    for port_text in re.findall(r"--(?:topic_port|service_port|cmd_port)\s+(\d+)", cmdline):
        try:
            ports.add(int(port_text))
        except ValueError:
            pass
    return ports


def _terminate_pid(pid: int, killed_pids: set, *, reason: str) -> None:
    pid = int(pid)
    if pid <= 0 or pid in killed_pids:
        return
    cmdline = _pid_cmdline(pid)
    short_cmd = cmdline if len(cmdline) <= 160 else cmdline[:160] + "..."
    print(f"[Launcher][OrphanKill] PID {pid}: {reason}: {short_cmd}")
    try:
        if sys.platform.startswith("win"):
            subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True, timeout=5)
        else:
            os.kill(pid, signal.SIGTERM)
            deadline = time.time() + 3.0
            while time.time() < deadline:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.05)
            else:
                os.kill(pid, signal.SIGKILL)
        killed_pids.add(pid)
    except ProcessLookupError:
        killed_pids.add(pid)
    except Exception as exc:
        print(f"[Launcher][OrphanKill] Could not terminate PID {pid}: {exc}")


def _port_owners_psutil(ports: set) -> Dict[int, set]:
    owners: Dict[int, set] = {}
    try:
        import psutil
        for conn in psutil.net_connections(kind="tcp"):
            if conn.status != psutil.CONN_LISTEN or not conn.laddr:
                continue
            port = int(conn.laddr.port)
            if port in ports and conn.pid:
                owners.setdefault(port, set()).add(int(conn.pid))
    except Exception:
        pass
    return owners


def _port_owners_ss(ports: set) -> Dict[int, set]:
    owners: Dict[int, set] = {}
    if sys.platform.startswith("win"):
        return owners
    try:
        result = subprocess.run(["ss", "-ltnp"], capture_output=True, text=True, timeout=8)
    except Exception:
        return owners
    if result.returncode != 0:
        return owners
    for line in result.stdout.splitlines():
        if "LISTEN" not in line:
            continue
        port_match = re.search(r":(\d+)\s", line)
        if not port_match:
            continue
        port = int(port_match.group(1))
        if port not in ports:
            continue
        for pid_text in re.findall(r"pid=(\d+)", line):
            owners.setdefault(port, set()).add(int(pid_text))
    return owners


def _port_owners_netstat_windows(ports: set) -> Dict[int, set]:
    owners: Dict[int, set] = {}
    if not sys.platform.startswith("win"):
        return owners
    try:
        result = subprocess.run(["netstat", "-ano", "-p", "TCP"], capture_output=True, text=True, timeout=10)
    except Exception:
        return owners
    if result.returncode != 0:
        return owners
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[0] != "TCP" or parts[3] != "LISTENING":
            continue
        try:
            port = int(parts[1].rsplit(":", 1)[-1])
            pid = int(parts[-1])
        except ValueError:
            continue
        if port in ports:
            owners.setdefault(port, set()).add(pid)
    return owners


def _find_port_owners(ports_to_check: list) -> Dict[int, set]:
    ports = set(int(port) for port in ports_to_check)
    owners = _port_owners_psutil(ports)
    for port, pids in _port_owners_ss(ports).items():
        owners.setdefault(port, set()).update(pids)
    for port, pids in _port_owners_netstat_windows(ports).items():
        owners.setdefault(port, set()).update(pids)
    return owners


def _kill_orphan_sessions(
    my_pid: int,
    base_topic_port: int,
    base_service_port: int,
    port_step: int,
    num_sessions: int,
) -> int:
    ports_to_check = _managed_ports(base_topic_port, base_service_port, port_step, num_sessions)
    managed_port_set = set(ports_to_check)
    killed_pids = set()

    try:
        import psutil
        for proc in psutil.process_iter(["pid", "cmdline"]):
            try:
                pid = int(proc.info["pid"])
                if pid == int(my_pid):
                    continue
                cmdline = " ".join(str(part) for part in (proc.info.get("cmdline") or []))
                if _is_repo_launcher_cmdline(cmdline):
                    _terminate_pid(pid, killed_pids, reason="stale multi_session_launcher from this checkout")
                    continue
                runtime_ports = _runtime_cmdline_ports(cmdline)
                blocked_ports = sorted(runtime_ports & managed_port_set)
                if "intervention_vr_runtime.py" in cmdline and blocked_ports:
                    _terminate_pid(pid, killed_pids, reason=f"runtime cmdline uses managed ports {blocked_ports}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except ImportError:
        print("[Launcher][OrphanKill] psutil not installed; using OS port scan fallback.")

    owners = _find_port_owners(ports_to_check)
    for port, pids in sorted(owners.items()):
        for pid in sorted(pids):
            if pid == int(my_pid) or pid in killed_pids:
                continue
            if _is_runtime_pid(pid):
                _terminate_pid(pid, killed_pids, reason=f"owns managed port {port}")
            else:
                print(f"[Launcher][PortPreflight] Port {port} is owned by non-runtime PID {pid}: {_pid_cmdline(pid)}")

    if killed_pids:
        print(f"[Launcher][OrphanKill] Killed {len(killed_pids)} orphan(s): {sorted(killed_pids)}. Waiting for sockets...")
        time.sleep(3.0)
    else:
        print("[Launcher][OrphanKill] No orphan sessions found.")
    return len(killed_pids)


def _assert_managed_ports_free(base_topic_port: int, base_service_port: int, port_step: int, num_sessions: int) -> None:
    ports_to_check = _managed_ports(base_topic_port, base_service_port, port_step, num_sessions)
    owners = _find_port_owners(ports_to_check)
    if not owners:
        print(f"[Launcher][PortPreflight] All managed ports are free ({len(ports_to_check)} ports).")
        return
    print("[Launcher][ERROR] Managed ports are still occupied; refusing to launch partial grid:")
    for port, pids in sorted(owners.items()):
        for pid in sorted(pids):
            print(f"  port={port} pid={pid} cmd={_pid_cmdline(pid)}")
    raise SystemExit(1)


def _assert_managed_ports_bindable(
    bind_ip: str,
    base_topic_port: int,
    base_service_port: int,
    port_step: int,
    num_sessions: int,
    *,
    retries: int = 5,
    retry_delay_s: float = 0.5,
) -> None:
    """Open and close bind-test sockets for every managed TCP port.

    `ss`/`psutil` can miss owners under permission or shutdown races. A direct
    bind test is the final authority before advertising sessions to the Quest.

    Retries up to `retries` times with `retry_delay_s` between attempts to handle
    the TOCTOU race where a dying process releases the port within milliseconds of
    the bind-test (port free in ss, fails bind, free again 100ms later).
    """
    ports_to_check = _managed_ports(base_topic_port, base_service_port, port_step, num_sessions)

    for attempt in range(retries + 1):
        sockets = []
        failed = []
        try:
            for port in ports_to_check:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                try:
                    sock.bind((bind_ip, int(port)))
                    sockets.append(sock)
                except OSError as exc:
                    failed.append((int(port), exc))
                    try:
                        sock.close()
                    except Exception:
                        pass
        finally:
            for sock in sockets:
                try:
                    sock.close()
                except Exception:
                    pass

        if not failed:
            print(f"[Launcher][PortPreflight] Direct bind-test passed for {len(ports_to_check)} managed ports.")
            return

        if attempt < retries:
            print(f"[Launcher][PortPreflight] Bind-test failed (attempt {attempt+1}/{retries+1}); "
                  f"retrying in {retry_delay_s:.1f}s (ports still releasing)...")
            time.sleep(retry_delay_s)
        else:
            print("[Launcher][ERROR] Managed ports failed direct bind-test; refusing to launch partial grid:")
            for port, exc in failed:
                print(f"  port={port} bind_ip={bind_ip} error={exc}")
            raise SystemExit(1)


def build_session_cmd(args, traj: str, session_idx: int) -> list:
    """Build the subprocess argv for one session."""
    topic_port   = args.base_topic_port   + session_idx * args.port_step
    service_port = args.base_service_port + session_idx * args.port_step
    cmd_port     = topic_port + 5

    # When --with_pc is set, use higher-quality defaults. MujocoPublisher is
    # no longer running per-session so we have budget for denser PCs +
    # higher resolution + smoother fps. Caps are CLI-overridable.
    if getattr(args, "with_pc", False):
        effective_fps = min(float(args.fps), float(args.pc_fps_cap))
        effective_w   = args.pc_width
        effective_h   = args.pc_height
    else:
        effective_fps = float(args.fps)
        effective_w   = args.w
        effective_h   = args.h

    cmd = [
        PYTHON_EXE, str(RUNTIME),
        "--xml",         args.xml,
        "--trajectory",  traj,
        "--host",        args.host,
        "--unity_node",  args.unity_node,
        "--bind_ip",     args.bind_ip,
        "--topic_port",  str(topic_port),
        "--service_port", str(service_port),
        "--cmd_port",    str(cmd_port),
        "--session_index", str(session_idx),
        "--fps",         str(effective_fps),
        "--fps_idle",    str(args.fps_idle),
        "--active_peer_threshold", str(args.active_peer_threshold),
        "--w",           str(effective_w),
        "--h",           str(effective_h),
        "--jpg_quality", str(args.jpg_quality),
        "--replay_speed", str(args.replay_speed),
        "--post_replan_mode", str(args.post_replan_mode),
        "--transport",   args.transport,
        "--visible_geoms_groups",
    ] + [str(g) for g in args.visible_geoms_groups]

    if getattr(args, "rgb_mode", False):
        # Diamond RGB panel single view: all 4 cameras render, but the point
        # cloud pipeline is never engaged (no --pc flag at all) — this keeps
        # the RGB-panel path fully separate from the point cloud code path.
        # front = always-on thumbnail feed (selector grid + diamond top panel);
        # wrist/left/right are gated to the ACTIVE/selected session only.
        cmd += ["--cams", "front", "top", "right", "left", "wrist"]
        cmd += ["--rgb_cams", "front"]
        cmd += [
            "--rgb_panel_mode",
            "--rgb_panel_cams", "wrist", "left", "right",
            "--rgb_panel_active_only",
        ]
        if getattr(args, "selected_sensor_activation", False):
            cmd.append("--selected_sensor_activation")
    elif getattr(args, "with_pc", False):
        # Full intervention mode: all cameras + point clouds.
        # front = thumbnail feed in the selector scene; right/left/top point clouds
        # become active in single view (PC source cameras unchanged). Wrist is
        # intentionally disabled here.
        cmd += ["--cams", "front", "top", "right", "left"]
        cmd += ["--rgb_cams", "front"]
        cmd += [
            "--pc", "--pc_cams", "top", "right", "left",
            "--pc_sampling", str(args.pc_sampling),
            "--pc_stride", str(args.pc_stride),
            "--pc_max_points", str(args.pc_max_points),
            "--pc_min_depth", "0.08", "--pc_max_depth", "2.8",
            "--pc_intrinsics_mode", "mujoco",
            "--pc_clip_below_table",
            "--pc_table_clearance", "0.005",
            "--pc_top_table_clearance", "0.005",
            "--pc_object_only",
            "--pc_object_bbox_min", "0.30", "-0.45", "-0.07",
            "--pc_object_bbox_max", "1.00", "0.45", "0.90",
            "--cam_extrinsics_profile", "lab_standard",
            "--pc_no_anchor_auto_translate",
        ]
        if getattr(args, "pc_active_only", False):
            cmd.append("--pc_active_only")
        if getattr(args, "selected_sensor_activation", False):
            cmd.append("--selected_sensor_activation")
    else:
        # Thumbnail-only mode: only top camera RGB, no point clouds
        if args.cams:
            cmd += ["--cams"] + args.cams
        if args.rgb_cams:
            cmd += ["--rgb_cams"] + args.rgb_cams

    if args.start_paused:
        cmd.append("--start_paused")
    if args.usb_auto_reverse:
        cmd.append("--usb_auto_reverse")
    # Thumbnail sessions always skip MujocoPublisher.
    # With 13 sessions × ~60 CreateVisual service calls each = ~780 concurrent calls to
    # the Quest ZMQ service handler → SubscribeRigidObjectsController timeouts (confirmed
    # in session_12_20260516_181437.log: "timed out after 3 attempts, 16812ms").
    # SimPubClient is disabled in InterventionSessionBootstrap when coming from the
    # selector, so no scene service is needed. Use --with_mujoco_publisher to override.
    if not args.with_mujoco_publisher:
        cmd.append("--no_mujoco_publisher")

    if not getattr(args, "with_xr_device", False):
        cmd.append("--no_xr_device")

    # Standalone single-session mode: boot the session straight into single-view
    # state so point clouds / ACTIVE sensors come up without waiting for an
    # ENTER_SINGLE command (the Quest still auto-enters; this is belt-and-suspenders).
    if getattr(args, "single", False):
        cmd.append("--start_in_single_view")

    # Windowed performance metrics: one JSONL per session alongside the session logs,
    # sharing the launcher's timestamp so a run's files group together.
    if getattr(args, "metrics", False):
        stamp = getattr(args, "_run_stamp", None) or time.strftime("%Y%m%d_%H%M%S")
        metrics_path = Path(args.log_dir) / f"metrics_S{session_idx:02d}_{stamp}.jsonl"
        cmd += ["--metrics_out", str(metrics_path),
                "--metrics_window_s", str(args.metrics_window_s)]
        if getattr(args, "metrics_summary", False):
            cmd.append("--metrics_summary")

    # extra_runtime_args is a single string — shell-split into tokens before
    # appending. Using shlex.split keeps quoted substrings intact in case the
    # user passes something like --extra_runtime_args "--foo \"with spaces\"".
    extras_str = (args.extra_runtime_args or "").strip()
    if extras_str:
        import shlex as _shlex
        cmd += _shlex.split(extras_str, posix=False)

    # Per-session INDEPENDENT policy wiring: session i mirrors policy instance i.
    # Appended AFTER the extras so these explicit per-session ports win over any
    # legacy shared ports a caller might still pass via --extra_runtime_args.
    if getattr(args, "policy_state_base_port", 0):
        state_port = args.policy_state_base_port + session_idx * args.policy_port_step
        cmd += ["--policy_state_host", str(args.policy_host),
                "--policy_state_port", str(state_port)]
    if getattr(args, "policy_cmd_base_port", 0):
        pcmd_port = args.policy_cmd_base_port + session_idx * args.policy_port_step
        cmd += ["--policy_cmd_host", str(args.policy_host),
                "--policy_cmd_port", str(pcmd_port)]

    return cmd


def write_status(status_path: str, sessions: list) -> None:
    try:
        with open(status_path, "w") as f:
            json.dump(sessions, f, indent=2)
    except Exception as e:
        print(f"[Launcher][WARN] Could not write status file: {e}")


def main() -> None:
    ap = build_launcher_parser()
    args = ap.parse_args()

    if getattr(args, "rgb_mode", False) and getattr(args, "with_pc", False):
        print("[Launcher][ERROR] --rgb_mode and --with_pc are mutually exclusive "
              "(the single-session view is either point clouds or RGB camera panels, "
              "never both). Pass only one.")
        sys.exit(1)

    trajs = [str(Path(t).resolve()) for t in args.trajectories]

    # --single: standalone single-session mode. Launch exactly one session and
    # bypass the multi-window selector grid. Skips the --windows validation; the
    # discovery beacon advertises n_sessions=1 so the Quest auto-enters single view.
    if getattr(args, "single", False):
        trajs = trajs[:1]
        print("[Launcher] --single: standalone single-session mode "
              "(1 session, grid bypassed, --start_in_single_view forwarded).")
    # --windows N: launch exactly N active sessions. Fleet Study C also permits one;
    # --grid_capacity keeps its selector on the same nine-cell board without turning
    # that one worker into standalone auto-enter mode.
    # cycling the provided trajectory list. This is the single "how many windows"
    # variable — each group of 3 fills one column of the VR selector grid.
    elif args.windows is not None:
        n_win = int(args.windows)
        if n_win not in (1, 3, 6, 9, 12, 15):
            print(f"[Launcher][ERROR] --windows must be one of 1, 3, 6, 9, 12, 15 (got {n_win}).")
            sys.exit(1)
        trajs = [trajs[k % len(trajs)] for k in range(n_win)]
        print(f"[Launcher] --windows {n_win}: launching {n_win} sessions (cycling "
              f"{len(args.trajectories)} provided trajectory file(s)).")

    max_sessions = max(1, int(args.max_sessions))
    if len(trajs) > max_sessions:
        print(
            f"[Launcher][WARN] {len(trajs)} trajectories requested but --max_sessions "
            f"is {max_sessions}. Trimming to {max_sessions}. "
            "(Historical safe ceiling was 13; raise --max_sessions carefully and watch "
            "RAM headroom — session N+1's trimesh import is the usual OOM point.)"
        )
        trajs = trajs[:max_sessions]
    # Soft warning when we're inside ~80% of the user-set cap.
    if len(trajs) > max(8, int(max_sessions * 0.8)):
        print(
            f"[Launcher][WARN] {len(trajs)}/{max_sessions} sessions: RAM usage may be near "
            "the limit. Monitor Task Manager for OOM."
        )

    n = len(trajs)
    grid_capacity = int(getattr(args, "grid_capacity", 0) or n)
    if grid_capacity < n or grid_capacity > 15:
        print(f"[Launcher][ERROR] --grid_capacity must be between active sessions ({n}) "
              f"and 15 (got {grid_capacity}).")
        sys.exit(1)

    # --- Kill orphan processes from previous launcher runs that still hold
    # our port range. Strong two-pass approach: psutil cmdline scan + netstat
    # port-owner scan. Prevents the 'zmq.error.ZMQError: Address in use'
    # startup crashes. Skipped if --no_orphan_kill is set.
    if not getattr(args, "no_orphan_kill", False):
        _kill_orphan_sessions(os.getpid(), args.base_topic_port, args.base_service_port, args.port_step, n)
    _assert_managed_ports_free(args.base_topic_port, args.base_service_port, args.port_step, n)
    _assert_managed_ports_bindable(
        args.bind_ip,
        args.base_topic_port,
        args.base_service_port,
        args.port_step,
        n,
    )

    # --- Validate host IP is active before starting any session ---
    if not _check_host_ip(args.host):
        print(
            f"[Launcher][ERROR] --host {args.host} is not assigned to any local interface. "
            "Check that WiFi is connected and on the same subnet as the Quest. "
            "Run 'ipconfig' to see active addresses."
        )
        sys.exit(1)
    print(f"[Launcher] Host IP {args.host} confirmed active.")

    # --- UDP discovery beacon is started only after all sessions pass startup
    # health. This prevents the Quest from discovering a partial/broken grid.
    discovery_thread = None
    discovery_stop = None

    # --- Set up logging ---
    log_dir = Path(args.log_dir)
    log_handles = []
    unified_fh = None
    unified_lock = threading.Lock()
    ts = time.strftime("%Y%m%d_%H%M%S")
    # Share this run's stamp with build_session_cmd so metrics JSONLs group with the logs.
    args._run_stamp = ts

    use_unified = getattr(args, "unified_log", False)
    if use_unified:
        log_dir.mkdir(parents=True, exist_ok=True)
        unified_path = log_dir / f"multi_session_{ts}.log"
        try:
            unified_fh = open(unified_path, "w", buffering=1)
            print(f"[Launcher] Unified log → {unified_path.resolve()}")
        except Exception as e:
            print(f"[Launcher][WARN] Could not open unified log: {e}")
            unified_fh = None
    elif not args.no_session_logs:
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"[Launcher] Session logs → {log_dir.resolve()}/session_XX_{ts}.log")

    print(f"[Launcher] Starting {n} session(s)...")
    print(f"[Launcher] Runtime: {RUNTIME}")
    print(f"[Launcher] Port scheme: topic={args.base_topic_port}+{args.port_step}*i  "
          f"service={args.base_service_port}+{args.port_step}*i")

    reader_threads = []
    procs = []
    status = []

    def _spawn_session(i, traj, is_restart=False):
        """Spawn (or respawn) session i. Returns (proc, log_fh).
        Handles unified-log, per-session-log, and no-log modes uniformly.
        On restart, opens a fresh log file with the same naming so output keeps
        landing in the expected place."""
        cmd  = build_session_cmd(args, traj, i)
        t_port = args.base_topic_port   + i * args.port_step
        s_port = args.base_service_port + i * args.port_step
        c_port = t_port + 5
        suffix = " (RESTART)" if is_restart else ""
        print(f"[Launcher] Session {i:02d}{suffix} → topic={t_port} service={s_port} cmd={c_port} traj={Path(traj).name}")

        log_fh_local = None

        if use_unified and unified_fh is not None:
            try:
                unified_fh.write(
                    f"# [S{i:02d}]{suffix} topic={t_port} cmd={c_port} traj={traj}\n"
                    f"# [S{i:02d}]{suffix} CMD: {' '.join(str(x) for x in cmd)}\n\n"
                )
            except Exception:
                pass
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=Path(RUNTIME).parents[3],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                prefix = f"[S{i:02d}] "
                fh_ref = unified_fh
                lock_ref = unified_lock

                def _reader(p, pfx, fh, lk):
                    try:
                        for raw in p.stdout:
                            line = pfx + raw.decode(errors="replace")
                            with lk:
                                fh.write(line)
                                fh.flush()
                    except Exception:
                        pass

                t = threading.Thread(target=_reader, args=(proc, prefix, fh_ref, lock_ref), daemon=True)
                t.start()
                reader_threads.append(t)
            except Exception as e:
                print(f"[Launcher][ERROR] Failed to start session {i}: {e}")
                proc = None

        elif not args.no_session_logs:
            log_path = log_dir / f"session_{i:02d}_{ts}.log"
            try:
                mode = "a" if is_restart else "w"
                log_fh_local = open(log_path, mode, buffering=1)
                log_fh_local.write(f"# Session {i}{suffix}  topic={t_port}  traj={traj}\n")
                log_fh_local.write(f"# CMD: {' '.join(str(x) for x in cmd)}\n\n")
            except Exception as e:
                print(f"[Launcher][WARN] Could not open log file {log_path}: {e}")
                log_fh_local = None
            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=Path(RUNTIME).parents[3],
                    stdout=log_fh_local if log_fh_local else None,
                    stderr=log_fh_local if log_fh_local else None,
                )
                if log_fh_local and not is_restart:
                    print(f"[Launcher] Session {i:02d} log → {log_path.name}")
            except Exception as e:
                print(f"[Launcher][ERROR] Failed to start session {i}: {e}")
                proc = None
        else:
            try:
                proc = subprocess.Popen(cmd, cwd=Path(RUNTIME).parents[3])
            except Exception as e:
                print(f"[Launcher][ERROR] Failed to start session {i}: {e}")
                proc = None
        return proc, log_fh_local

    for i, traj in enumerate(trajs):
        proc, log_fh = _spawn_session(i, traj)
        log_handles.append(log_fh)
        procs.append(proc)
        t_port = args.base_topic_port   + i * args.port_step
        s_port = args.base_service_port + i * args.port_step
        c_port = t_port + 5
        time.sleep(1.0)
        initial_status = "running" if proc is not None and proc.poll() is None else "failed"
        if proc is not None and initial_status == "failed":
            print(f"[Launcher][WARN] Session {i:02d} exited during startup with code={proc.returncode}.")
        status.append({
            "session_index":  i,
            "pid":            proc.pid if proc else None,
            "trajectory":     traj,
            "topic_port":     t_port,
            "service_port":   s_port,
            "cmd_port":       c_port,
            "status":         initial_status,
            "restart_count":  0,
        })

    write_status(args.status_file, status)
    print(f"[Launcher] Status written to: {args.status_file}")
    initial_failed = [item for item in status if item.get("status") != "running"]
    if initial_failed:
        print("[Launcher][ERROR] One or more sessions failed during startup; stopping all sessions before discovery:")
        for item in initial_failed:
            print(
                f"  session={item.get('session_index')} pid={item.get('pid')} "
                f"topic={item.get('topic_port')} service={item.get('service_port')} cmd={item.get('cmd_port')}"
            )
        for i, proc in enumerate(procs):
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                    print(f"[Launcher] Session {i:02d} (pid={proc.pid}) terminated after startup failure.")
                except Exception as exc:
                    print(f"[Launcher] Session {i:02d} terminate error after startup failure: {exc}")
        for item in status:
            if item.get("status") == "running":
                item["status"] = "stopped"
        write_status(args.status_file, status)
        raise SystemExit(1)

    if not getattr(args, "no_discovery", False):
        discovery_thread, discovery_stop = _start_discovery_broadcaster(
            args.host, args.base_topic_port, args.port_step, n, args.discovery_port,
            rgb_mode=getattr(args, "rgb_mode", False),
            mc_active=getattr(args, "mc_active", False),
            grid_capacity=grid_capacity,
        )
    print(f"[Launcher] All sessions started. Press Ctrl+C to stop all.")

    # --- Signal handler for clean shutdown ---
    def _shutdown(sig, frame):
        print("\n[Launcher] Shutting down all sessions...")
        if discovery_stop is not None:
            discovery_stop.set()
        for i, proc in enumerate(procs):
            if proc is not None and proc.poll() is None:
                try:
                    proc.terminate()
                    print(f"[Launcher] Session {i} (pid={proc.pid}) terminated.")
                except Exception as e:
                    print(f"[Launcher] Session {i} terminate error: {e}")
        # Close log file handles (per-session files)
        for fh in log_handles:
            if fh is not None:
                try:
                    fh.close()
                except Exception:
                    pass
        # Close unified log if used
        if unified_fh is not None:
            try:
                unified_fh.close()
            except Exception:
                pass
        # Update status file
        for i, proc in enumerate(procs):
            if proc is not None:
                status[i]["status"] = "stopped"
        write_status(args.status_file, status)
        sys.exit(0)

    signal.signal(signal.SIGINT,  _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    # --- Monitor loop with restart-on-crash ---
    # If a session's PID exits, respawn it with the same args up to
    # args.max_restarts attempts before giving up. Prevents the "ports stuck"
    # scenarios from killing a session for good.
    max_restarts = max(0, int(args.max_restarts))
    try:
        while True:
            time.sleep(5.0)
            any_change = False
            for i, proc in enumerate(procs):
                if proc is None:
                    continue
                if proc.poll() is None:
                    continue  # still alive
                # Process has exited.
                rc = proc.returncode
                rcnt = int(status[i].get("restart_count", 0))
                if rcnt >= max_restarts:
                    if status[i]["status"] != "failed":
                        print(f"[Launcher] Session {i:02d} exited code={rc}; "
                              f"gave up after {rcnt} restart(s).")
                        status[i]["status"] = "failed"
                        any_change = True
                    continue
                # Respawn — brief delay lets stale port bindings release before retry.
                rcnt += 1
                print(f"[Launcher] Session {i:02d} exited code={rc}; "
                      f"waiting 3s then restarting (attempt {rcnt}/{max_restarts}).")
                time.sleep(3.0)
                new_proc, new_log_fh = _spawn_session(i, trajs[i], is_restart=True)
                procs[i] = new_proc
                if new_log_fh is not None:
                    log_handles[i] = new_log_fh
                status[i]["restart_count"] = rcnt
                status[i]["pid"] = new_proc.pid if new_proc else None
                status[i]["status"] = "running" if new_proc else "failed"
                any_change = True

            alive = sum(1 for p in procs if p is not None and p.poll() is None)
            dead  = n - alive
            if dead > 0 or any_change:
                print(f"[Launcher] Status: {alive}/{n} sessions running, {dead} dead.")
                write_status(args.status_file, status)
    except KeyboardInterrupt:
        _shutdown(None, None)


if __name__ == "__main__":
    main()
