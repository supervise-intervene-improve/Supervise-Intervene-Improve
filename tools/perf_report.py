#!/usr/bin/env python3
"""Join PC-side and Quest-side telemetry into one readable performance report.

Answers the PART-1 checklist for whichever VR mode was running:

  * point-cloud / RGB-panel update rate  — published Hz (PC) vs received Hz (Quest)
                                            vs drawn Hz (Quest)
  * average rendering latency            — MuJoCo render + JPEG + PC build (PC side)
                                            and app frame time (Quest side)
  * end-to-end FPS                       — the rate new data reaches the headset, plus
                                            the display FPS the runtime actually hits

Inputs:
  --python  session_logs/metrics_S00_*.jsonl   (from --metrics / METRICS=1)
  --quest   quest.jsonl                        (from tools/quest_telemetry.py)

Either side may be omitted; the report renders what it has.

Usage:
    python tools/perf_report.py --python session_logs/metrics_S00_*.jsonl \\
                                --quest quest.jsonl
    python tools/perf_report.py --python 'session_logs/metrics_S*.jsonl' --md report.md
"""

import argparse
import glob
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def load_jsonl(patterns):
    rows = []
    for pattern in patterns or []:
        matches = glob.glob(pattern)
        if not matches and Path(pattern).exists():
            matches = [pattern]
        if not matches:
            print(f"[PerfReport][WARN] no files matched: {pattern}", file=sys.stderr)
        for path in sorted(matches):
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # partial final write
                    row["_file"] = path
                    rows.append(row)
    return rows


def summarize(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    vals_sorted = sorted(vals)
    n = len(vals_sorted)

    def pct(p):
        if n == 1:
            return vals_sorted[0]
        rank = p / 100.0 * (n - 1)
        lo = int(rank)
        hi = min(lo + 1, n - 1)
        frac = rank - lo
        return vals_sorted[lo] * (1 - frac) + vals_sorted[hi] * frac

    return {
        "n": n,
        "avg": statistics.fmean(vals_sorted),
        "p50": pct(50),
        "p95": pct(95),
        "p99": pct(99),
        "min": vals_sorted[0],
        "max": vals_sorted[-1],
    }


def fmt_stat(stat, unit=""):
    if stat is None:
        return "—"
    return (f"{stat['avg']:.2f} / {stat['p50']:.2f} / {stat['p95']:.2f} / "
            f"{stat['p99']:.2f} / {stat['max']:.2f}{unit}")


def table(headers, rows):
    """Markdown table that also reads fine as plain text."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    out = ["| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"]
    out.append("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for row in rows:
        out.append("| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)) + " |")
    return "\n".join(out)


def analyse_python(rows, active_only=True):
    """Fold windowed runtime rows into one report section per session."""
    by_session = defaultdict(list)
    for row in rows:
        by_session[row.get("session_index")].append(row)

    sections = []
    for session_index in sorted(by_session, key=lambda k: (k is None, k)):
        srows = by_session[session_index]
        if active_only:
            active = [r for r in srows
                      if (r.get("labels") or {}).get("adaptive_mode") == "ACTIVE"]
            if active:
                srows = active
        if not srows:
            continue

        modes = {r.get("mode") for r in srows if r.get("mode")}
        labels = {}
        for r in srows:
            labels.update(r.get("labels") or {})

        latency = defaultdict(list)
        for r in srows:
            for name, stat in (r.get("latency_ms") or {}).items():
                # Windows are equal-length, so weighting by sample count keeps the
                # aggregate honest when one window saw more frames than another.
                latency[name].extend([stat["avg"]] * max(1, stat["n"] // 10))
                latency[f"{name}__max"].append(stat["max"])
                latency[f"{name}__p95"].append(stat["p95"])

        rates = defaultdict(list)
        for r in srows:
            for name, stat in (r.get("rates_hz") or {}).items():
                rates[name].append(stat["hz"])

        counters = defaultdict(float)
        for r in srows:
            for name, val in (r.get("counters") or {}).items():
                counters[name] += val

        sections.append({
            "session_index": session_index,
            "mode": "+".join(sorted(modes)) or "unknown",
            "windows": len(srows),
            "duration_s": sum(r.get("window_s", 0.0) for r in srows),
            "labels": labels,
            "latency": latency,
            "rates": rates,
            "counters": dict(counters),
        })
    return sections


def analyse_quest(rows):
    by_source = defaultdict(list)
    for row in rows:
        by_source[row.get("source")].append(row)

    out = {}
    display = by_source.get("display") or []
    if display:
        out["display"] = {
            "fps": summarize([r.get("fps") for r in display]),
            "fps_target": max((r.get("fps_target") or 0) for r in display),
            "app_render_ms": summarize([r.get("app_render_ms") for r in display]),
            "timewarp_ms": summarize([r.get("timewarp_ms") for r in display]),
            "cpu_gpu_ms": summarize([r.get("cpu_gpu_ms") for r in display]),
            "predicted_latency_ms": summarize([r.get("predicted_latency_ms") for r in display]),
            "stale_total": sum(r.get("stale") or 0 for r in display),
            "tear_total": sum(r.get("tear") or 0 for r in display),
            "samples": len(display),
        }

    pc = by_source.get("pointcloud") or []
    if pc:
        # Heartbeats emitted before any data arrives report 0 Hz and would drag the
        # averages down; the interesting question is the rate while streaming.
        live = [r for r in pc if (r.get("rx_hz") or 0) > 0]
        per_source_rx = defaultdict(list)
        for row in pc:
            for source_name, rate in (row.get("source_rx_hz") or {}).items():
                if rate is not None and rate > 0:
                    per_source_rx[source_name].append(rate)
        out["pointcloud"] = {
            "rx_hz": summarize([r.get("rx_hz") for r in live]),
            "draw_hz": summarize([r.get("draw_hz") for r in live]),
            "upload_hz": summarize([r.get("upload_hz") for r in live]),
            "upload_ms": summarize([r.get("upload_ms") for r in live]),
            "identity_latency_ms": summarize([r.get("identity_latency_ms") for r in pc]),
            "first_payload_latency_ms": summarize([r.get("first_payload_latency_ms") for r in pc]),
            "first_upload_latency_ms": summarize([r.get("first_upload_latency_ms") for r in pc]),
            "points": summarize([r.get("points") for r in live]),
            "last_payload_age_s": summarize([r.get("last_payload_age_s") for r in live]),
            "sources_live_max": max((r.get("sources_live") or 0) for r in pc),
            "sources_total": max((r.get("sources_total") or 0) for r in pc),
            "samples": len(pc),
            "live_samples": len(live),
            "identity_mismatch_samples": sum(
                r.get("identity_confirmed") is False for r in pc
            ),
            "identity_drops": max((r.get("identity_drops") or 0) for r in pc),
            "upstream_stale_samples": sum(r.get("upstream_stale") is True for r in pc),
            "session_heartbeat_age_s": summarize(
                [r.get("session_heartbeat_age_s") for r in pc]
            ),
            "latest_session_index": pc[-1].get("session_index"),
            "latest_episode_id": pc[-1].get("episode_id"),
            "latest_policy_seq": pc[-1].get("policy_seq"),
            "latest_frame_idx": pc[-1].get("frame_idx"),
            "latest_mode": pc[-1].get("mode"),
            "latest_intervention_phase": pc[-1].get("intervention_phase"),
            "latest_qpos_delta_norm": pc[-1].get("qpos_delta_norm"),
            "latest_state_change_age_s": pc[-1].get("state_change_age_s"),
            "latest_upstream_stale_age_s": pc[-1].get("upstream_stale_age_s"),
            "source_rx_hz": {
                name: summarize(values) for name, values in sorted(per_source_rx.items())
            },
        }

    entries = by_source.get("rgb_wrist") or []
    if entries:
        live = [r for r in entries if (r.get("rx_hz") or 0) > 0]
        out["rgb_wrist"] = {
            "rx_hz": summarize([r.get("rx_hz") for r in live]),
            "samples": len(entries),
            "live_samples": len(live),
        }

    # RGB panels: split per camera. In RGB mode every diamond panel reports
    # sessionIndex=0, so the topic/cam is the only thing that separates them.
    panels = by_source.get("rgb_panel") or []
    if panels:
        per_cam = defaultdict(list)
        for row in panels:
            per_cam[row.get("cam") or f"S{row.get('session_index')}"].append(row)
        out["rgb_panel"] = {}
        for cam, rows_for_cam in sorted(per_cam.items()):
            live = [r for r in rows_for_cam if (r.get("rx_hz") or 0) > 0]
            out["rgb_panel"][cam] = {
                "rx_hz": summarize([r.get("rx_hz") for r in live]),
                "decode_ms": summarize([r.get("decode_ms") for r in rows_for_cam]),
                "drops_total": max((r.get("drops") or 0) for r in rows_for_cam),
                "samples": len(rows_for_cam),
                "live_samples": len(live),
            }
    return out


def render_report(py_sections, quest, active_only):
    lines = ["# VR streaming performance report", ""]
    scope = "ACTIVE windows only" if active_only else "all windows"
    lines.append(f"_PC-side scope: {scope}. Latency columns are avg / p50 / p95 / p99 / max._")
    lines.append("")

    if not py_sections:
        lines += ["## PC side", "", "_No runtime metrics rows supplied._", ""]
    for sec in py_sections:
        labels = sec["labels"]
        lines.append(f"## PC side — session {sec['session_index']} "
                     f"(interface mode: **{sec['mode']}**)")
        lines.append("")
        cfg = " · ".join(
            f"{k}={labels[k]}" for k in sorted(labels)
            if k in {"target_fps", "fps_idle", "resolution", "pc_stride",
                     "pc_max_points", "jpg_quality", "pc_round_robin"}
        )
        lines.append(f"- config: {cfg or 'n/a'}")
        lines.append(f"- windows: {sec['windows']} covering {sec['duration_s']:.0f}s")
        if labels.get("active_session_count", 0) > 1:
            lines.append(
                f"- **lifecycle error:** {labels['active_session_count']} point-cloud "
                "sessions were active concurrently"
            )
        if "policy_seq_gaps" in labels:
            lines.append(
                f"- policy progress: seq={labels.get('policy_seq')} · "
                f"cumulative gaps={labels.get('policy_seq_gaps')} · "
                f"state age={labels.get('policy_state_age_s')}s · peers={labels.get('peer_count')}"
            )
        ticks = sec["counters"].get("publish_ticks", 0)
        over = sec["counters"].get("over_budget_frames", 0)
        if ticks:
            lines.append(f"- publish ticks: {int(ticks)} · over budget: {int(over)} "
                         f"({100.0 * over / ticks:.2f}%)")
        distinct_renders = sec["counters"].get("render_distinct_states", 0)
        repeated_renders = sec["counters"].get("render_repeated_states", 0)
        if distinct_renders + repeated_renders:
            lines.append(
                f"- repeated-state render ratio: "
                f"{100.0 * repeated_renders / (distinct_renders + repeated_renders):.1f}% "
                f"({int(repeated_renders)}/{int(distinct_renders + repeated_renders)} ticks)"
            )
        lines.append("")

        rows = []
        pretty = {
            "render_ms": "MuJoCo render",
            "jpeg_ms": "JPEG encode",
            "pc_build_ms": "PC build (GPU worker)",
            "pc_submit_ms": "PC submit (enqueue)",
            "pc_drain_ms": "PC drain + publish",
            "publish_ms": "RGB publish (enqueue)",
            "tick_total_ms": "publish tick TOTAL",
            "budget_ms": "budget (1/fps)",
        }
        for key, label in pretty.items():
            vals = sec["latency"].get(key)
            if not vals:
                continue
            stat = summarize(vals)
            maxes = sec["latency"].get(f"{key}__max") or []
            p95s = sec["latency"].get(f"{key}__p95") or []
            rows.append([
                label,
                f"{stat['avg']:.2f}",
                f"{statistics.fmean(p95s):.2f}" if p95s else "—",
                f"{max(maxes):.2f}" if maxes else "—",
            ])
        freshness_prefixes = {
            "state_to_render_ms": "policy state -> render",
            "pc_worker_queue_ms.": "PC worker queue",
            "pc_submission_to_publish_ms.": "PC submit -> publish",
            "pc_build_to_publish_ms.": "PC build -> publish",
            "pc_state_to_publish_ms.": "policy state -> PC publish",
        }
        for key in sorted(sec["latency"]):
            if key.endswith("__max") or key.endswith("__p95") or key in pretty:
                continue
            prefix = next((p for p in freshness_prefixes if key.startswith(p)), None)
            if prefix is None:
                continue
            stat = summarize(sec["latency"][key])
            maxes = sec["latency"].get(f"{key}__max") or []
            p95s = sec["latency"].get(f"{key}__p95") or []
            suffix = key[len(prefix):].lstrip(".")
            label = freshness_prefixes[prefix] + (f" ({suffix})" if suffix else "")
            rows.append([
                label,
                f"{stat['avg']:.2f}",
                f"{statistics.fmean(p95s):.2f}" if p95s else "—",
                f"{max(maxes):.2f}" if maxes else "—",
            ])
        if rows:
            lines.append("### Rendering latency (ms)")
            lines.append("")
            lines.append(table(["stage", "avg", "p95", "max"], rows))
            lines.append("")

        rate_rows = []
        for name in sorted(sec["rates"]):
            if name.startswith("_"):
                continue
            stat = summarize(sec["rates"][name])
            rate_rows.append([name, f"{stat['avg']:.1f}", f"{stat['min']:.1f}",
                              f"{stat['max']:.1f}"])
        if rate_rows:
            lines.append("### Published update rate (Hz, per topic)")
            lines.append("")
            lines.append(table(["topic", "avg", "min", "max"], rate_rows))
            lines.append("")

    lines.append("## Quest side (from adb logcat)")
    lines.append("")
    if not quest:
        lines.append("_No Quest telemetry supplied. Run tools/quest_telemetry.py._")
        lines.append("")
        return "\n".join(lines)

    disp = quest.get("display")
    if disp:
        lines.append("### Display / app frame timing")
        lines.append("")
        rows = [
            ["display FPS", fmt_stat(disp["fps"]), f"target {disp['fps_target']:.0f}"],
            ["app render", fmt_stat(disp["app_render_ms"], " ms"), ""],
            ["timewarp", fmt_stat(disp["timewarp_ms"], " ms"), ""],
            ["CPU&GPU total", fmt_stat(disp["cpu_gpu_ms"], " ms"), ""],
            ["predicted latency", fmt_stat(disp["predicted_latency_ms"], " ms"), ""],
            ["stale frames", str(disp["stale_total"]), f"over {disp['samples']} samples"],
            ["torn frames", str(disp["tear_total"]), ""],
        ]
        lines.append(table(["metric", "avg / p50 / p95 / p99 / max", "note"], rows))
        lines.append("")

    pc = quest.get("pointcloud")
    if pc:
        lines.append("### Point cloud (received on device)")
        lines.append("")
        rows = [
            ["receive rate", fmt_stat(pc["rx_hz"], " Hz"), "END-TO-END: payloads arriving"],
            ["draw rate", fmt_stat(pc["draw_hz"], " Hz"), "GPU draw submissions"],
            ["fresh upload rate", fmt_stat(pc["upload_hz"], " Hz"), "new combined clouds uploaded"],
            ["combined upload", fmt_stat(pc["upload_ms"], " ms"), "CPU pack plus two GPU uploads"],
            ["selection to identity", fmt_stat(pc["identity_latency_ms"], " ms"), "matching heartbeat accepted"],
            ["selection to payload", fmt_stat(pc["first_payload_latency_ms"], " ms"), "first accepted PC payload"],
            ["selection to upload", fmt_stat(pc["first_upload_latency_ms"], " ms"), "first visible fresh cloud"],
            ["points merged", fmt_stat(pc["points"]), ""],
            ["last payload age", fmt_stat(pc["last_payload_age_s"], " s"), "staleness"],
            ["live sources", f"{pc['sources_live_max']}/{pc['sources_total']}", ""],
            ["session heartbeat age", fmt_stat(pc["session_heartbeat_age_s"], " s"),
             f"session {pc['latest_session_index']} · episode {pc['latest_episode_id'] or 'n/a'}"],
            ["policy progress", f"seq {pc['latest_policy_seq']} · frame {pc['latest_frame_idx']}",
             f"{pc['latest_mode'] or 'unknown'} / {pc['latest_intervention_phase'] or 'unknown'}"],
            ["state content", f"qpos delta {pc['latest_qpos_delta_norm']}",
             f"unchanged {pc['latest_state_change_age_s']} s · source age {pc['latest_upstream_stale_age_s']} s"],
            ["identity gate", f"{pc['identity_mismatch_samples']} mismatched samples",
             f"{pc['identity_drops']} payloads discarded before confirmation"],
        ]
        lines.append(table(["metric", "avg / p50 / p95 / p99 / max", "note"], rows))
        lines.append("")
        if pc["source_rx_hz"]:
            lines.append(table(
                ["PC source", "Quest receive rate (avg / p50 / p95 / p99 / max)"],
                [[name, fmt_stat(stat, " Hz")] for name, stat in pc["source_rx_hz"].items()],
            ))
            lines.append("")

        publish_rates = []
        upstream_stale = pc["upstream_stale_samples"] > 0
        for sec in py_sections:
            for topic, values in sec["rates"].items():
                if topic.endswith("/pc") and values:
                    publish_rates.extend(values)
        publish_hz = statistics.fmean(publish_rates) if publish_rates else None
        rx_hz = pc["rx_hz"]["avg"] if pc["rx_hz"] else None
        draw_hz = pc["draw_hz"]["avg"] if pc["draw_hz"] else None
        upload_hz = pc["upload_hz"]["avg"] if pc["upload_hz"] else None
        policy_paused = any(
            bool(sec.get("labels", {}).get("policy_paused")) for sec in py_sections
        )
        legitimate_transition = pc["latest_intervention_phase"] in {
            "aligning", "finishing", "returning_home"
        }

        if policy_paused:
            diagnosis = "Policy is intentionally paused; repeated scene content is expected while transport and display telemetry remain live."
        elif legitimate_transition:
            diagnosis = f"Scene motion is intentionally held during {pc['latest_intervention_phase']}; verify rates again after policy_resumed."
        elif upstream_stale:
            diagnosis = "Policy state froze upstream; publisher output may still look healthy because it repeats the last pose."
        elif publish_hz is not None and publish_hz < 15.0:
            diagnosis = f"Selected publisher missed the 15 Hz recovery floor ({publish_hz:.1f} Hz average per PC topic)."
        elif publish_hz is not None and any(
            stat and stat["avg"] < 15.0 for stat in pc["source_rx_hz"].values()
        ):
            slow = ", ".join(
                f"{name}={stat['avg']:.1f} Hz" for name, stat in pc["source_rx_hz"].items()
                if stat and stat["avg"] < 15.0
            )
            diagnosis = f"Publisher is healthy but Quest receive is low for {slow}: inspect network/receiver loss."
        elif rx_hz is not None and rx_hz >= 15.0 and (upload_hz is None or upload_hz < 15.0):
            diagnosis = f"Quest receives clouds but fresh upload is low ({rx_hz:.1f} → {upload_hz or 0.0:.1f} Hz)."
        elif pc["identity_mismatch_samples"]:
            diagnosis = "Session identity is not consistently confirmed; clouds are intentionally gated to prevent old-scene display."
        elif rx_hz is not None and draw_hz is not None and upload_hz is not None:
            diagnosis = "Streaming stages are healthy. A still scene is legitimate when paused, aligning, or returning_home."
        else:
            diagnosis = "Insufficient overlapping publisher/Quest telemetry for an automatic diagnosis."
        lines.append(f"**Pipeline diagnosis:** {diagnosis}")
        lines.append("")

    entry = quest.get("rgb_wrist")
    if entry:
        lines.append("### Wrist RGB / RGBD subscriber (received on device)")
        lines.append("")
        rows = [["receive rate", fmt_stat(entry["rx_hz"], " Hz"),
                 f"{entry['live_samples']}/{entry['samples']} live samples"]]
        lines.append(table(["metric", "avg / p50 / p95 / p99 / max", "note"], rows))
        lines.append("")

    panels = quest.get("rgb_panel")
    if panels:
        lines.append("### RGB panels (received on device, per camera)")
        lines.append("")
        rows = []
        stale = False
        for cam, entry in sorted(panels.items()):
            rows.append([
                cam,
                fmt_stat(entry["rx_hz"], " Hz"),
                fmt_stat(entry["decode_ms"], " ms") if entry["decode_ms"] else "—",
                str(entry["drops_total"]),
                f"{entry['live_samples']}/{entry['samples']}",
            ])
            if entry["decode_ms"] is None:
                stale = True
        lines.append(table(
            ["camera", "receive rate (avg/p50/p95/p99/max)", "JPEG decode", "drops",
             "live/total"], rows))
        lines.append("")
        if stale:
            lines.append("_No decode timings: this capture predates the "
                         "SessionThumbnailPanel heartbeat, so panel rates come from "
                         "sparse reconnect lines only. Rebuild the APK for continuous "
                         "per-panel rates._")
            lines.append("")

    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--python", nargs="*", default=[],
                    help="Runtime metrics JSONL path(s) or glob(s).")
    ap.add_argument("--quest", nargs="*", default=[],
                    help="Quest telemetry JSONL path(s) or glob(s).")
    ap.add_argument("--md", default=None, help="Also write the report to this markdown file.")
    ap.add_argument("--all_windows", action="store_true",
                    help="Include IDLE windows. Default reports ACTIVE only (single view), "
                         "since grid/idle sessions publish at --fps_idle and would skew averages.")
    args = ap.parse_args()

    if not args.python and not args.quest:
        ap.error("supply at least one of --python / --quest")

    py_rows = load_jsonl(args.python)
    quest_rows = load_jsonl(args.quest)
    py_sections = analyse_python(py_rows, active_only=not args.all_windows)
    quest = analyse_quest(quest_rows)

    report = render_report(py_sections, quest, active_only=not args.all_windows)
    print(report)
    if args.md:
        Path(args.md).parent.mkdir(parents=True, exist_ok=True)
        Path(args.md).write_text(report + "\n", encoding="utf-8")
        print(f"\n[PerfReport] wrote {args.md}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
