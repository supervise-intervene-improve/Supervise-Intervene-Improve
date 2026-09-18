#!/usr/bin/env python3
"""Collect Quest-side streaming telemetry from adb logcat — no APK rebuild needed.

Everything this parses is already being printed on the headset today:

  * `VrApi : FPS=72/72,Prd=20ms,Tear=0,Early=0,Stale=0,...`
        The Meta runtime's own 1 Hz report: display FPS, predicted display latency,
        torn/early/stale frame counts. This is the app framerate that actually
        reaches the panels.

  * `[GpuMergedPointCloudLoader] hb name=... rx=N draws=N sources=3/3 points=N
     lastPayloadAge_s=...`
        1 Hz heartbeat from the merged point cloud loader. `rx` and `draws` are
        CUMULATIVE counters, so differencing successive heartbeats gives the
        point-cloud receive rate and the draw-submit rate on device.

  * `[SimPubRgbdSubscriber] hb rx=N idle_s=... mode=...`
        1 Hz heartbeat from the wrist RGB/RGBD subscriber; `rx` is cumulative.

  * `[SessionThumbnailPanel] ...rx=N...`
        Only printed on reconnect/shutdown today, so RGB *panel* rates are sparse
        unless the optional 1 Hz heartbeat is added to that component.

Output is JSONL on the `time.time()` timebase so tools/perf_report.py can join it
against the runtime's own metrics rows.

Usage:
    python tools/quest_telemetry.py --out quest.jsonl
    python tools/quest_telemetry.py --out quest.jsonl --duration 120
    python tools/quest_telemetry.py --from-file captured_logcat.txt --out quest.jsonl

Note: capture must be LIVE and filtered — a post-hoc `adb logcat -d` usually misses
the window you care about because hand-tracking spam evicts it from the ring buffer.
"""

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

# `VrApi : FPS=72/72,Prd=20ms,Tear=0,Early=0,Stale=0,Stale2/5/10/max=0/0/0/0,...`
RE_VRAPI = re.compile(
    r"FPS=(?P<fps>[\d.]+)/(?P<fps_target>[\d.]+)"
    r"(?:,Prd=(?P<prd_ms>-?[\d.]+)ms)?"
    r"(?:,Tear=(?P<tear>-?\d+))?"
    r"(?:,Early=(?P<early>-?\d+))?"
    r"(?:,Stale=(?P<stale>-?\d+))?"
)
# Same line also carries the on-device render cost breakdown, which is the Quest-side
# answer to "average rendering latency": App = the app's own frame time, CPU&GPU = the
# combined per-frame cost, TW = asynchronous timewarp cost.
RE_VRAPI_APP = re.compile(r"\bApp=(?P<app_ms>[\d.]+)ms")
RE_VRAPI_TW = re.compile(r"\bTW=(?P<tw_ms>[\d.]+)ms")
RE_VRAPI_CPUGPU = re.compile(r"CPU&GPU=(?P<cpugpu_ms>[\d.]+)ms")
RE_VRAPI_UTIL = re.compile(r"GPU%=(?P<gpu_pct>[\d.]+),CPU%=(?P<cpu_pct>[\d.]+)")

# `[GpuMergedPointCloudLoader] hb name='X' rx=107 draws=72 sources=3/3 points=20883 ...`
RE_PC_HB = re.compile(
    r"GpuMergedPointCloudLoader\]\s+hb\s+"
    r"(?:name='(?P<name>[^']*)'\s+)?"
    r"rx=(?P<rx>\d+)\s+draws=(?P<draws>\d+)\s+"
    r"sources=(?P<sources_live>\d+)/(?P<sources_total>\d+)\s+"
    r"points=(?P<points>\d+)"
)
RE_PC_AGE = re.compile(r"lastPayloadAge_s=(?P<age>[\d.]+)")
RE_PC_IDENTITY = re.compile(r"identity=(?P<confirmed>[01])/(?P<expected>-?\d+)")
RE_PC_IDENTITY_DROPS = re.compile(r"identityDrops=(?P<drops>\d+)")
RE_PC_SESSION = re.compile(r"\bsession=(?P<session>-?\d+)")
RE_PC_EPISODE = re.compile(r"episode='(?P<episode>[^']*)'")
RE_PC_POLICY_SEQ = re.compile(r"policySeq=(?P<seq>-?\d+)")
RE_PC_FRAME = re.compile(r"\bframe=(?P<frame>-?\d+)")
RE_PC_MODE = re.compile(r"\bmode='(?P<mode>[^']*)'")
RE_PC_PHASE = re.compile(r"\bphase='(?P<phase>[^']*)'")
RE_PC_UPSTREAM_STALE = re.compile(r"upstreamStale=(?P<stale>True|False)")
RE_PC_SESSION_HB_AGE = re.compile(r"sessionHbAge_s=(?P<age>[\d.]+)")
RE_PC_SOURCE_RX = re.compile(r"sourceRx='(?P<counts>[^']*)'")
RE_PC_QPOS_DELTA = re.compile(r"qposDelta=(?P<value>[\d.]+)")
RE_PC_STATE_CHANGE_AGE = re.compile(r"stateChangeAge_s=(?P<value>[\d.]+)")
RE_PC_UPSTREAM_STALE_AGE = re.compile(r"upstreamStaleAge_s=(?P<value>[\d.]+)")
RE_PC_UPLOADS = re.compile(r"\buploads=(?P<value>\d+)")
RE_PC_UPLOAD_MS = re.compile(r"\buploadMs=(?P<value>-?[\d.]+)")
RE_PC_IDENTITY_MS = re.compile(r"\bidentityMs=(?P<value>-?[\d.]+)")
RE_PC_FIRST_PAYLOAD_MS = re.compile(r"\bfirstPayloadMs=(?P<value>-?[\d.]+)")
RE_PC_FIRST_UPLOAD_MS = re.compile(r"\bfirstUploadMs=(?P<value>-?[\d.]+)")
RE_PC_GC0 = re.compile(r"\bgc0=(?P<value>\d+)")

# `[SimPubRgbdSubscriber] hb rx=41 idle_s=0.12 mode=StandaloneRgb ...`
RE_RGBD_HB = re.compile(
    r"SimPubRgbdSubscriber\]\s+hb\s+rx=(?P<rx>\d+)(?:\s+idle_s=(?P<idle_s>[\d.]+))?"
)
RE_RGBD_MODE = re.compile(r"mode=(?P<mode>\w+)")

# `[SessionThumbnailPanel] S03 ... rx=128 tcp://...`
# `[SessionThumbnailPanel] S00 hb rx=128 drops=3 decode_ms=1.42 idle_s=0.03 revealed=True
#  topic=SimPub/Sensors/wrist/rgb endpoint=tcp://...`
# In RGB mode every diamond panel reports sessionIndex=0, so the topic (not the index)
# is what identifies the stream.
RE_PANEL_HB = re.compile(
    r"SessionThumbnailPanel\]\s+S(?P<idx>\d+)\s+hb\s+rx=(?P<rx>\d+)"
    r"(?:\s+drops=(?P<drops>\d+))?"
    r"(?:\s+decode_ms=(?P<decode_ms>[\d.]+))?"
    r"(?:\s+idle_s=(?P<idle_s>[\d.]+))?"
)
RE_PANEL_TOPIC = re.compile(r"topic=(?P<topic>\S+)")
# Order-independent key=value reader for heartbeat lines.
RE_KV = re.compile(r"\b([a-z_]+)=([-\d.]+)\b")
# Fallback for the reconnect/shutdown lines that predate the heartbeat.
RE_PANEL = re.compile(r"SessionThumbnailPanel\]\s+S(?P<idx>\d+).*?rx=(?P<rx>\d+)")

# `-v threadtime` line prefix: `07-13 15:17:03.123  1234  5678 I VrApi   : ...`
# The device clock is the correct timebase for rates (host receive time is jittered by
# adb buffering), and it is the ONLY timebase available when replaying a capture file.
RE_LOGCAT_TS = re.compile(
    r"^(?P<mon>\d{2})-(?P<day>\d{2})\s+(?P<h>\d{2}):(?P<m>\d{2}):(?P<s>\d{2})\.(?P<ms>\d{3})"
)


def parse_logcat_timestamp(line, year=None):
    """Device wall-clock for a threadtime line, or None. Logcat omits the year."""
    m = RE_LOGCAT_TS.match(line)
    if not m:
        return None
    import calendar
    year = year or time.localtime().tm_year
    try:
        struct = (
            year, int(m.group("mon")), int(m.group("day")),
            int(m.group("h")), int(m.group("m")), int(m.group("s")),
            0, 0, -1,
        )
        return time.mktime(struct) + int(m.group("ms")) / 1000.0
    except (ValueError, OverflowError):
        return None

DEFAULT_LOGCAT_FILTER = [
    "-v", "threadtime",
    "VrApi:V", "Unity:V", "VrApi-Metrics:V", "*:S",
]



def _open_text_sniffed(path):
    """Open a capture as text, detecting UTF-16/UTF-8 from the byte-order mark."""
    with open(path, "rb") as fh:
        bom = fh.read(4)
    if bom.startswith(b"\xff\xfe\x00\x00") or bom.startswith(b"\x00\x00\xfe\xff"):
        encoding = "utf-32"
    elif bom.startswith(b"\xff\xfe") or bom.startswith(b"\xfe\xff"):
        encoding = "utf-16"
    elif bom.startswith(b"\xef\xbb\xbf"):
        encoding = "utf-8-sig"
    else:
        encoding = "utf-8"
    return open(path, "r", encoding=encoding, errors="replace")


def find_adb(explicit=None):
    """adb is often not on PATH — Unity bundles it with the Android player."""
    if explicit:
        return explicit
    found = shutil.which("adb")
    if found:
        return found
    candidates = [
        "/opt/unity/Editor/Data/PlaybackEngines/AndroidPlayer/SDK/platform-tools/adb",
        os.path.expanduser("~/Android/Sdk/platform-tools/adb"),
        os.path.expanduser("~/Library/Android/sdk/platform-tools/adb"),
    ]
    for pattern in [
        "/usr/local/lib/android/sdk/platform-tools/adb",
        "C:/Program Files/Unity/Hub/Editor/*/Editor/Data/PlaybackEngines/AndroidPlayer/SDK/platform-tools/adb.exe",
    ]:
        candidates.append(pattern)
    for cand in candidates:
        if "*" in cand:
            import glob
            matches = sorted(glob.glob(cand))
            if matches:
                return matches[-1]
        elif os.path.isfile(cand):
            return cand
    return None


class CounterRate:
    """Hz from a cumulative counter, tolerant of restarts (counter going backwards)."""

    def __init__(self):
        self.last_value = None
        self.last_t = None

    def update(self, value, now):
        rate = None
        if self.last_value is not None and self.last_t is not None:
            dv = value - self.last_value
            dt = now - self.last_t
            if dv >= 0 and dt > 1e-6:
                rate = dv / dt
        self.last_value = value
        self.last_t = now
        return rate


class QuestTelemetryParser:
    """Turns logcat lines into JSONL rows. Stateless w.r.t. adb — also parses files."""

    def __init__(self):
        self.pc_rx = CounterRate()
        self.pc_draws = CounterRate()
        self.pc_uploads = CounterRate()
        self.pc_source_rx = {}
        self.rgbd_rx = CounterRate()
        self.panel_rx = {}
        self.counts = {"vrapi": 0, "pc_hb": 0, "rgbd_hb": 0, "panel": 0}

    def parse_line(self, line, now=None):
        now = time.time() if now is None else now

        m = RE_PC_HB.search(line)
        if m:
            self.counts["pc_hb"] += 1
            rx = int(m.group("rx"))
            draws = int(m.group("draws"))
            age = RE_PC_AGE.search(line)
            identity = RE_PC_IDENTITY.search(line)
            identity_drops = RE_PC_IDENTITY_DROPS.search(line)
            session = RE_PC_SESSION.search(line)
            episode = RE_PC_EPISODE.search(line)
            policy_seq = RE_PC_POLICY_SEQ.search(line)
            frame = RE_PC_FRAME.search(line)
            mode = RE_PC_MODE.search(line)
            phase = RE_PC_PHASE.search(line)
            upstream_stale = RE_PC_UPSTREAM_STALE.search(line)
            session_hb_age = RE_PC_SESSION_HB_AGE.search(line)
            source_rx = RE_PC_SOURCE_RX.search(line)
            qpos_delta = RE_PC_QPOS_DELTA.search(line)
            state_change_age = RE_PC_STATE_CHANGE_AGE.search(line)
            upstream_stale_age = RE_PC_UPSTREAM_STALE_AGE.search(line)
            uploads = RE_PC_UPLOADS.search(line)
            upload_ms = RE_PC_UPLOAD_MS.search(line)
            identity_ms = RE_PC_IDENTITY_MS.search(line)
            first_payload_ms = RE_PC_FIRST_PAYLOAD_MS.search(line)
            first_upload_ms = RE_PC_FIRST_UPLOAD_MS.search(line)
            gc0 = RE_PC_GC0.search(line)
            source_totals = {}
            source_rates = {}
            if source_rx:
                for item in source_rx.group("counts").split(","):
                    name, sep, value = item.rpartition(":")
                    if not sep or not value.isdigit():
                        continue
                    total = int(value)
                    source_totals[name] = total
                    meter = self.pc_source_rx.setdefault(name, CounterRate())
                    source_rates[name] = meter.update(total, now)
            return {
                "wall_t": now,
                "source": "pointcloud",
                "name": m.group("name"),
                "rx_total": rx,
                "draws_total": draws,
                "rx_hz": self.pc_rx.update(rx, now),
                "draw_hz": self.pc_draws.update(draws, now),
                "uploads_total": int(uploads.group("value")) if uploads else None,
                "upload_hz": self.pc_uploads.update(int(uploads.group("value")), now) if uploads else None,
                "upload_ms": float(upload_ms.group("value")) if upload_ms else None,
                "identity_latency_ms": float(identity_ms.group("value")) if identity_ms else None,
                "first_payload_latency_ms": float(first_payload_ms.group("value")) if first_payload_ms else None,
                "first_upload_latency_ms": float(first_upload_ms.group("value")) if first_upload_ms else None,
                "gc0_total": int(gc0.group("value")) if gc0 else None,
                "sources_live": int(m.group("sources_live")),
                "sources_total": int(m.group("sources_total")),
                "points": int(m.group("points")),
                "last_payload_age_s": float(age.group("age")) if age else None,
                "identity_confirmed": bool(int(identity.group("confirmed"))) if identity else None,
                "expected_session_index": int(identity.group("expected")) if identity else None,
                "identity_drops": int(identity_drops.group("drops")) if identity_drops else None,
                "session_index": int(session.group("session")) if session else None,
                "episode_id": episode.group("episode") if episode else None,
                "policy_seq": int(policy_seq.group("seq")) if policy_seq else None,
                "frame_idx": int(frame.group("frame")) if frame else None,
                "mode": mode.group("mode") if mode else None,
                "intervention_phase": phase.group("phase") if phase else None,
                "upstream_stale": upstream_stale.group("stale") == "True" if upstream_stale else None,
                "session_heartbeat_age_s": (
                    float(session_hb_age.group("age")) if session_hb_age else None
                ),
                "source_rx_total": source_totals,
                "source_rx_hz": source_rates,
                "qpos_delta_norm": float(qpos_delta.group("value")) if qpos_delta else None,
                "state_change_age_s": (
                    float(state_change_age.group("value")) if state_change_age else None
                ),
                "upstream_stale_age_s": (
                    float(upstream_stale_age.group("value")) if upstream_stale_age else None
                ),
            }

        m = RE_RGBD_HB.search(line)
        if m:
            self.counts["rgbd_hb"] += 1
            rx = int(m.group("rx"))
            mode = RE_RGBD_MODE.search(line)
            return {
                "wall_t": now,
                "source": "rgb_wrist",
                "rx_total": rx,
                "rx_hz": self.rgbd_rx.update(rx, now),
                "idle_s": float(m.group("idle_s")) if m.group("idle_s") else None,
                "mode": mode.group("mode") if mode else None,
            }

        # VrApi check comes after the Unity ones: its line is long and the Unity
        # regexes are anchored on distinctive class names, so ordering is cheap.
        m = RE_VRAPI.search(line)
        if m and "VrApi" in line:
            self.counts["vrapi"] += 1
            def _num(key, cast=float):
                val = m.group(key)
                return cast(val) if val is not None else None

            def _opt(regex, key, cast=float):
                hit = regex.search(line)
                return cast(hit.group(key)) if hit else None

            return {
                "wall_t": now,
                "source": "display",
                "fps": _num("fps"),
                "fps_target": _num("fps_target"),
                "predicted_latency_ms": _num("prd_ms"),
                "tear": _num("tear", int),
                "early": _num("early", int),
                "stale": _num("stale", int),
                "app_render_ms": _opt(RE_VRAPI_APP, "app_ms"),
                "timewarp_ms": _opt(RE_VRAPI_TW, "tw_ms"),
                "cpu_gpu_ms": _opt(RE_VRAPI_CPUGPU, "cpugpu_ms"),
                "gpu_pct": _opt(RE_VRAPI_UTIL, "gpu_pct"),
                "cpu_pct": _opt(RE_VRAPI_UTIL, "cpu_pct"),
            }

        m = RE_PANEL_HB.search(line)
        if m:
            self.counts["panel"] += 1
            idx = int(m.group("idx"))
            rx = int(m.group("rx"))
            topic_m = RE_PANEL_TOPIC.search(line)
            topic = topic_m.group("topic") if topic_m else None
            key = topic or f"S{idx}"
            meter = self.panel_rx.setdefault(key, CounterRate())
            row = {
                "wall_t": now,
                "source": "rgb_panel",
                "session_index": idx,
                "topic": topic,
                "cam": topic.split("/")[-2] if topic and "/" in topic else None,
                "rx_total": rx,
                "rx_hz": meter.update(rx, now),
            }
            # Fields are read by NAME from the whole line, not by position. The heartbeat
            # grew a `skipped=` field between `drops=` and `decode_ms=`, and the old
            # positional regex then silently returned None for decode_ms and idle_s on
            # every row -- the same failure shape as the UTF-16 bug: no error, just
            # missing data. `skipped` (decodes refused by the per-frame budget) is kept
            # too; with `decodes` it is what distinguishes receive rate from the rate
            # the panel is actually redrawn at.
            fields = dict(RE_KV.findall(line))
            for key_name, cast in (("drops", int), ("skipped", int),
                                   ("decodes", int), ("decode_ms", float),
                                   ("idle_s", float)):
                val = fields.get(key_name)
                try:
                    row[key_name] = cast(val) if val is not None else None
                except ValueError:
                    row[key_name] = None
            dec = row.get("decodes")
            if dec is not None:
                dmeter = self.panel_rx.setdefault(f"S{idx}:{topic}:decodes", CounterRate())
                row["decode_hz"] = dmeter.update(dec, now)
            return row

        m = RE_PANEL.search(line)
        if m:
            self.counts["panel"] += 1
            idx = int(m.group("idx"))
            rx = int(m.group("rx"))
            meter = self.panel_rx.setdefault(f"S{idx}", CounterRate())
            return {
                "wall_t": now,
                "source": "rgb_panel",
                "session_index": idx,
                "topic": None,
                "cam": None,
                "rx_total": rx,
                "rx_hz": meter.update(rx, now),
            }

        return None


def stream_logcat(adb, serial=None, clear=True, extra_filter=None):
    cmd = [adb]
    if serial:
        cmd += ["-s", serial]
    if clear:
        try:
            subprocess.run(cmd + ["logcat", "-c"], timeout=15,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as exc:
            print(f"[QuestTelemetry][WARN] logcat -c failed: {exc}", file=sys.stderr)
    cmd += ["logcat"] + (extra_filter or DEFAULT_LOGCAT_FILTER)
    print(f"[QuestTelemetry] $ {' '.join(cmd)}", file=sys.stderr)
    return subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        text=True, errors="replace", bufsize=1,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="quest_telemetry.jsonl",
                    help="JSONL output path (default: quest_telemetry.jsonl).")
    ap.add_argument("--duration", type=float, default=0.0,
                    help="Stop after N seconds. 0 = run until Ctrl+C.")
    ap.add_argument("--adb", default=None, help="Path to adb (auto-detected if omitted).")
    ap.add_argument("--serial", default=None, help="adb device serial (-s).")
    ap.add_argument("--no_clear", action="store_true",
                    help="Do not run 'adb logcat -c' before capturing.")
    ap.add_argument("--from-file", dest="from_file", default=None,
                    help="Parse an existing logcat capture instead of running adb. "
                         "Rates are derived from line order, not real time.")
    ap.add_argument("--print_every", type=float, default=5.0,
                    help="Seconds between progress lines on stderr. 0 disables.")
    args = ap.parse_args()

    parser = QuestTelemetryParser()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    t_start = time.time()
    next_print = t_start + args.print_every if args.print_every > 0 else float("inf")

    proc = None
    if args.from_file:
        # Encoding is sniffed from the BOM, not assumed. The captures this repo already
        # has (quest_crash.txt and friends) were written by PowerShell's `Out-File`, which
        # produced UTF-16LE despite `-Encoding utf8` in the documented capture recipe.
        # Read as UTF-8 they decode to text containing no matchable line at all, so every
        # parser here silently yielded ZERO rows from a 76 MB file full of usable data --
        # a failure that looks exactly like "the headset never logged anything".
        src = _open_text_sniffed(args.from_file)
        line_iter = src
        synthetic_t = t_start
        replaying = True
    else:
        replaying = False
        adb = find_adb(args.adb)
        if not adb:
            print("[QuestTelemetry][ERROR] adb not found. Pass --adb /path/to/adb.\n"
                  "Unity bundles one at "
                  "<UnityEditor>/Data/PlaybackEngines/AndroidPlayer/SDK/platform-tools/adb",
                  file=sys.stderr)
            return 2
        proc = stream_logcat(adb, serial=args.serial, clear=not args.no_clear)
        line_iter = proc.stdout
        src = None
        synthetic_t = None

    stop = {"flag": False}

    def _handle_sigint(signum, frame):
        stop["flag"] = True

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        with open(out_path, "a", encoding="utf-8", buffering=1) as fh:
            for line in line_iter:
                if stop["flag"]:
                    break
                # Prefer the device's own logcat timestamp: it is immune to adb buffering
                # jitter, and it is the only usable timebase when replaying a file.
                now = parse_logcat_timestamp(line)
                if now is None and replaying:
                    # Capture without a threadtime prefix — fall back to a synthetic 1 Hz
                    # clock advanced once per heartbeat group so rates stay ordinal.
                    if "hb " in line or "FPS=" in line:
                        synthetic_t += 1.0
                    now = synthetic_t
                row = parser.parse_line(line, now=now)
                if row is not None:
                    fh.write(json.dumps(row, separators=(",", ":")) + "\n")
                    n_rows += 1
                t_now = time.time()
                if t_now >= next_print:
                    print(f"[QuestTelemetry] rows={n_rows} "
                          f"vrapi={parser.counts['vrapi']} pc={parser.counts['pc_hb']} "
                          f"rgb={parser.counts['rgbd_hb']} panel={parser.counts['panel']} "
                          f"elapsed={t_now - t_start:.0f}s", file=sys.stderr)
                    next_print = t_now + args.print_every
                if args.duration > 0 and (t_now - t_start) >= args.duration:
                    break
    finally:
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        if src is not None:
            src.close()

    print(f"[QuestTelemetry] wrote {n_rows} rows to {out_path} "
          f"(vrapi={parser.counts['vrapi']} pc_hb={parser.counts['pc_hb']} "
          f"rgbd_hb={parser.counts['rgbd_hb']} panel={parser.counts['panel']})",
          file=sys.stderr)
    if n_rows == 0:
        print("[QuestTelemetry][WARN] nothing parsed. Is the headset connected "
              "(`adb devices`) and the app running?", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
