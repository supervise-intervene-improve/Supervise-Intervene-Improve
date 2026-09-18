#!/usr/bin/env python3
"""Validate a recorded user-study session and produce a data-quality report.

Nothing read back the recorded NPZs until now, so a corrupt episode was only
discovered when analysis failed — potentially weeks after the participant left. This
walks a session directory, checks every episode, cross-references the NPZ against its
JSON sidecar / session.json / events.jsonl, and writes `quality_report.json` + `.md`.

Checks (one per PART-2 requirement):
  missing frames        gaps in wall_t larger than a multiple of the median dt
  broken timestamps     non-monotonic / NaN / negative-delta wall_t or sim_t
  array lengths         every per-frame array shares axis 0; widths match the model
  corrupted images      wrong shape/dtype, all-zero frames (the blank-frame fallback
                        writes zeros when camera capture fails), constant frames
  intervention bounds   intervention / action_source / intervention_id agree, and each
                        segment has matching start/end events in events.jsonl
  labels                task + scenario labels agree across all four places they appear
  outliers              duration / length IQR outliers, reusing data_cleaning's method

Usage:
    python utils/validate_study_data.py <session_dir>
    python utils/validate_study_data.py <session_dir> --backup /media/backup/study
    python utils/validate_study_data.py <participant_dir> --recursive
"""

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

SEVERITY_ORDER = {"fail": 0, "warn": 1, "info": 2}

# Per-frame arrays that must all share axis 0. Widths are checked where known.
FRAME_ARRAYS = (
    "wall_t", "sim_t", "q_real", "dq_real", "grip_real",
    "qpos_sim", "qvel_sim", "ctrl_sim",
    "intervention", "action_source", "intervention_id",
    "policy_action", "human_action", "acc", "acc_valid",
    "queried_policy", "chunk_index", "acc_components",
)


class Findings:
    def __init__(self):
        self.items = []

    def add(self, severity, check, message, **extra):
        self.items.append({"severity": severity, "check": check,
                           "message": message, **extra})

    def fail(self, check, message, **extra):
        self.add("fail", check, message, **extra)

    def warn(self, check, message, **extra):
        self.add("warn", check, message, **extra)

    def info(self, check, message, **extra):
        self.add("info", check, message, **extra)

    @property
    def worst(self):
        if not self.items:
            return "pass"
        return min((i["severity"] for i in self.items), key=lambda s: SEVERITY_ORDER[s])

    def counts(self):
        out = {"fail": 0, "warn": 0, "info": 0}
        for item in self.items:
            out[item["severity"]] += 1
        return out


def _iqr_outliers(values, names, factor=1.5):
    """IQR outlier detection, same method as utils/data_cleaning.py's StatsTracker."""
    if len(values) < 4:
        return []
    arr = np.asarray(values, dtype=np.float64)
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    if iqr <= 0:
        return []
    lo, hi = q1 - factor * iqr, q3 + factor * iqr
    return [(names[i], float(arr[i])) for i in range(len(arr))
            if arr[i] < lo or arr[i] > hi]


def check_timestamps(npz, findings):
    for key in ("wall_t", "sim_t"):
        if key not in npz:
            findings.fail("broken_timestamps", f"missing {key}")
            continue
        t = np.asarray(npz[key], dtype=np.float64)
        if t.size == 0:
            findings.fail("broken_timestamps", f"{key} is empty")
            continue
        if not np.all(np.isfinite(t)):
            findings.fail("broken_timestamps",
                          f"{key} has {int((~np.isfinite(t)).sum())} non-finite values")
        if t.size > 1:
            d = np.diff(t)
            n_back = int((d < 0).sum())
            if n_back:
                findings.fail("broken_timestamps",
                              f"{key} goes backwards at {n_back} frame(s)")
            n_zero = int((d == 0).sum())
            if n_zero > max(1, int(0.02 * d.size)):
                findings.warn("broken_timestamps",
                              f"{key} has {n_zero} zero-length steps "
                              f"({100.0 * n_zero / d.size:.1f}%)")


def check_missing_frames(npz, meta, findings):
    if "wall_t" not in npz:
        return
    t = np.asarray(npz["wall_t"], dtype=np.float64)
    if t.size < 3:
        findings.warn("missing_frames", f"only {t.size} frame(s) recorded")
        return
    d = np.diff(t)
    d = d[np.isfinite(d) & (d > 0)]
    if d.size == 0:
        findings.fail("missing_frames", "no positive time deltas")
        return
    median_dt = float(np.median(d))
    gaps = d[d > 2.0 * median_dt]
    if gaps.size:
        # Each oversized gap is at least one frame that never made it into the file.
        est_missing = int(np.sum(np.round(gaps / median_dt) - 1))
        severity = findings.fail if gaps.size > 0.05 * d.size else findings.warn
        severity("missing_frames",
                 f"{gaps.size} gap(s) > 2x median dt ({median_dt * 1000:.1f} ms); "
                 f"~{est_missing} frame(s) missing; largest {gaps.max() * 1000:.0f} ms",
                 median_dt_ms=median_dt * 1000.0, gap_count=int(gaps.size),
                 estimated_missing=est_missing)

    log_hz = meta.get("log_hz")
    if log_hz:
        expected_dt = 1.0 / float(log_hz)
        if median_dt > 0 and abs(median_dt - expected_dt) / expected_dt > 0.5:
            findings.warn("missing_frames",
                          f"median dt {median_dt * 1000:.1f} ms differs from log_hz "
                          f"{log_hz} Hz ({expected_dt * 1000:.1f} ms)")


def check_array_lengths(npz, findings):
    lengths = {}
    for key in FRAME_ARRAYS:
        if key in npz:
            arr = np.asarray(npz[key])
            if arr.ndim >= 1:
                lengths[key] = int(arr.shape[0])
    if not lengths:
        findings.fail("array_lengths", "no per-frame arrays found")
        return 0
    counts = {}
    for key, n in lengths.items():
        counts.setdefault(n, []).append(key)
    if len(counts) > 1:
        detail = "; ".join(f"{n}: {sorted(keys)}" for n, keys in sorted(counts.items()))
        findings.fail("array_lengths",
                      f"per-frame arrays disagree on length -> {detail}",
                      lengths=lengths)
    n_frames = max(counts)

    for key, width in (("q_real", 7), ("dq_real", 7),
                       ("policy_action", 8), ("human_action", 8)):
        if key in npz:
            arr = np.asarray(npz[key])
            if arr.ndim != 2 or arr.shape[1] != width:
                findings.fail("array_lengths",
                              f"{key} has shape {arr.shape}, expected (N, {width})")
    return n_frames


def check_images(npz, meta, findings):
    # State-only recording (INTERVENE_RECORD_RGB=0) deliberately stores no frames; they are
    # reconstructed offline from qpos/qvel/ctrl instead. The sidecar still lists
    # camera_names (which cameras WOULD have been used, and what a re-render needs), so
    # without this branch every such episode collected one warn per camera and dragged the
    # whole session's quality report down to "warn".
    if meta.get("save_rgb") is False:
        cams = list(meta.get("camera_names") or [])
        findings.info(
            "corrupted_images",
            "state-only episode (save_rgb=false); frames are re-rendered offline from "
            f"qpos/qvel/ctrl for cameras {cams or '<unspecified>'}",
        )
        # Any frames that ARE present would be unexpected — fall through so they get checked.
        if not any(k.startswith("rgb_") for k in npz.files):
            return

    cams = list(meta.get("camera_names") or [])
    if not cams:
        cams = [k[4:] for k in npz.files if k.startswith("rgb_")]
    if not cams:
        findings.info("corrupted_images", "no camera frames recorded")
        return
    for cam in cams:
        key = f"rgb_{cam}"
        if key not in npz:
            findings.warn("corrupted_images", f"{key} missing for camera '{cam}'")
            continue
        arr = np.asarray(npz[key])
        if arr.ndim != 4 or arr.shape[-1] != 3:
            findings.fail("corrupted_images", f"{key} has shape {arr.shape}, expected (N,H,W,3)")
            continue
        if arr.dtype != np.uint8:
            findings.warn("corrupted_images", f"{key} dtype is {arr.dtype}, expected uint8")
        # The recorder writes an all-zero frame when camera capture raises, so a run of
        # black frames means lost observations, not a dark scene.
        per_frame_max = arr.reshape(arr.shape[0], -1).max(axis=1)
        n_black = int((per_frame_max == 0).sum())
        if n_black:
            severity = findings.fail if n_black > 0.05 * arr.shape[0] else findings.warn
            severity("corrupted_images",
                     f"{key} has {n_black}/{arr.shape[0]} all-zero (capture-failed) frames",
                     camera=cam, black_frames=n_black)
        if arr.shape[0] > 2:
            flat = arr.reshape(arr.shape[0], -1)
            n_static = int(np.sum(np.all(flat[1:] == flat[:-1], axis=1)))
            if n_static > 0.5 * (arr.shape[0] - 1):
                findings.warn("corrupted_images",
                              f"{key} is static for {n_static}/{arr.shape[0] - 1} "
                              f"consecutive frame pairs (camera may be frozen)",
                              camera=cam)


def check_interventions(npz, findings, events_by_episode=None, episode_id=None):
    if "intervention" not in npz:
        findings.warn("intervention_boundaries", "no intervention flag recorded")
        return []
    flag = np.asarray(npz["intervention"]).astype(bool)
    source = (np.asarray(npz["action_source"]).astype(int)
              if "action_source" in npz else None)
    ids = (np.asarray(npz["intervention_id"]).astype(int)
           if "intervention_id" in npz else None)

    if source is not None:
        mismatch = int(np.sum(flag != (source == 1)))
        if mismatch:
            findings.fail("intervention_boundaries",
                          f"intervention flag and action_source disagree on "
                          f"{mismatch} frame(s)")
    if ids is not None:
        bad_zero = int(np.sum(flag & (ids == 0)))
        bad_nonzero = int(np.sum(~flag & (ids != 0)))
        if bad_zero:
            findings.fail("intervention_boundaries",
                          f"{bad_zero} intervention frame(s) have intervention_id=0")
        if bad_nonzero:
            findings.fail("intervention_boundaries",
                          f"{bad_nonzero} autonomous frame(s) have a non-zero "
                          f"intervention_id")

    # Contiguous runs of the flag == the human segments.
    segments = []
    if flag.size:
        padded = np.concatenate(([False], flag, [False]))
        edges = np.diff(padded.astype(np.int8))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        for s, e in zip(starts, ends):
            seg_ids = set(ids[s:e].tolist()) if ids is not None else set()
            segments.append({"start": int(s), "end": int(e), "frames": int(e - s),
                             "intervention_ids": sorted(seg_ids)})
            if len(seg_ids) > 1:
                findings.fail("intervention_boundaries",
                              f"segment [{s}:{e}] mixes intervention_ids {sorted(seg_ids)}")

    if "policy_action" in npz and "human_action" in npz:
        pol = np.asarray(npz["policy_action"], dtype=np.float64)
        hum = np.asarray(npz["human_action"], dtype=np.float64)
        if pol.shape[0] == flag.shape[0]:
            pol_on_human = int(np.sum(np.isfinite(pol[flag]).any(axis=1))) if flag.any() else 0
            hum_on_policy = int(np.sum(np.isfinite(hum[~flag]).any(axis=1))) if (~flag).any() else 0
            if pol_on_human:
                findings.warn("intervention_boundaries",
                              f"{pol_on_human} human frame(s) carry a policy action")
            if hum_on_policy:
                findings.warn("intervention_boundaries",
                              f"{hum_on_policy} autonomous frame(s) carry a human action")

    # Cross-check against the event log.
    if events_by_episode is not None and episode_id:
        logged = events_by_episode.get(episode_id, {}).get("interventions", set())
        recorded = {i for seg in segments for i in seg["intervention_ids"]}
        only_events = logged - recorded
        only_frames = recorded - logged
        if only_events:
            findings.warn("intervention_boundaries",
                          f"intervention id(s) {sorted(only_events)} logged in "
                          f"events.jsonl but absent from the trajectory "
                          f"(cancelled interventions are expected here)")
        if only_frames:
            findings.fail("intervention_boundaries",
                          f"intervention id(s) {sorted(only_frames)} present in the "
                          f"trajectory but missing from events.jsonl")
    return segments


def check_labels(npz, meta, findings, session_meta=None, event_meta=None):
    def npz_str(key):
        if key not in npz:
            return None
        val = npz[key]
        try:
            return val.item() if getattr(val, "ndim", 1) == 0 else val
        except Exception:
            return None

    for key in ("participant_id", "task", "condition", "scenario_id"):
        npz_val = npz_str(key)
        meta_val = meta.get(key)
        if npz_val is None and meta_val is None:
            findings.warn("labels", f"{key} missing from both NPZ and sidecar")
            continue
        if npz_val is not None and meta_val is not None and str(npz_val) != str(meta_val):
            findings.fail("labels",
                          f"{key} disagrees: NPZ={npz_val!r} sidecar={meta_val!r}")
        if session_meta is not None:
            sess_val = session_meta.get(key)
            ref = npz_val if npz_val is not None else meta_val
            if sess_val is not None and ref is not None and str(sess_val) != str(ref):
                findings.fail("labels",
                              f"{key} disagrees with session.json: "
                              f"episode={ref!r} session={sess_val!r}")
    if event_meta:
        for key in ("scenario_id",):
            ev_val = event_meta.get(key)
            ref = meta.get(key) or npz_str(key)
            if ev_val and ref and str(ev_val) != str(ref):
                findings.fail("labels",
                              f"{key} disagrees with events.jsonl: "
                              f"episode={ref!r} events={ev_val!r}")

    # scenario_id encodes the seed, so it must reproduce from the recorded seed.
    seed = meta.get("episode_seed")
    scenario = meta.get("scenario_id") or npz_str("scenario_id")
    if seed is not None and scenario:
        try:
            if f"{int(seed):08x}" not in str(scenario):
                findings.fail("labels",
                              f"scenario_id {scenario!r} does not encode "
                              f"episode_seed {seed}")
        except (TypeError, ValueError):
            pass


def load_events(session_dir: Path):
    """Index events.jsonl by episode: outcome, interventions seen, scenario id."""
    path = session_dir / "events.jsonl"
    if not path.is_file():
        return None
    by_episode = {}
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            ep = ev.get("episode_id")
            if not ep:
                continue
            entry = by_episode.setdefault(ep, {"interventions": set(), "events": 0})
            entry["events"] += 1
            if ev.get("scenario_id"):
                entry["scenario_id"] = ev["scenario_id"]
            if ev.get("kind") == "episode_end":
                entry["outcome"] = ev.get("outcome")
                entry["reason"] = ev.get("reason")
            # Only interventions that actually released control produce trajectory
            # frames; requests that were cancelled before release do not.
            if ev.get("kind") == "intervention_released" and ev.get("intervention_id"):
                entry["interventions"].add(int(ev["intervention_id"]))
    return by_episode


def validate_episode(npz_path: Path, session_meta=None, events_by_episode=None):
    findings = Findings()
    result = {"file": str(npz_path), "name": npz_path.name}

    meta = {}
    sidecar = npz_path.with_suffix(".json")
    if sidecar.is_file():
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception as exc:
            findings.fail("labels", f"sidecar unreadable: {exc}")
    else:
        findings.warn("labels", "no .json sidecar next to the NPZ")

    try:
        npz = np.load(npz_path, allow_pickle=False)
    except Exception as exc:
        findings.fail("readable", f"cannot open NPZ: {exc}")
        result.update({"status": "fail", "findings": findings.items,
                       "counts": findings.counts()})
        return result

    episode_id = meta.get("episode_id") or npz_path.stem
    event_meta = (events_by_episode or {}).get(episode_id)

    with npz:
        n_frames = check_array_lengths(npz, findings)
        check_timestamps(npz, findings)
        check_missing_frames(npz, meta, findings)
        check_images(npz, meta, findings)
        segments = check_interventions(npz, findings, events_by_episode, episode_id)
        check_labels(npz, meta, findings, session_meta, event_meta)

        duration = None
        if "wall_t" in npz:
            t = np.asarray(npz["wall_t"], dtype=np.float64)
            if t.size > 1:
                duration = float(t[-1] - t[0])
        acc_present = "acc" in npz
        acc_nonconstant = None
        if acc_present:
            acc = np.asarray(npz["acc"], dtype=np.float64)
            finite = acc[np.isfinite(acc)]
            acc_nonconstant = bool(finite.size and float(np.nanstd(finite)) > 1e-9)
            if finite.size and not acc_nonconstant:
                findings.warn("acc", f"acc is constant at {finite[0]:.3f} "
                                     f"(scorer may not be wired for this run)")

    if event_meta and event_meta.get("outcome"):
        outcome_dir = npz_path.parent.name
        if outcome_dir in ("success", "failure", "incomplete") \
                and outcome_dir != event_meta["outcome"]:
            findings.fail("labels",
                          f"episode filed under {outcome_dir}/ but events.jsonl says "
                          f"outcome={event_meta['outcome']}")

    result.update({
        "episode_id": episode_id,
        "frames": n_frames,
        "duration_s": duration,
        "outcome": (event_meta or {}).get("outcome") or meta.get("outcome"),
        "scenario_id": meta.get("scenario_id"),
        "intervention_segments": segments,
        "acc_present": acc_present,
        "acc_nonconstant": acc_nonconstant,
        "status": findings.worst,
        "counts": findings.counts(),
        "findings": findings.items,
    })
    return result


def sha256(path: Path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def backup_session(session_dir: Path, backup_root: Path):
    """Copy the session tree and verify every file by SHA-256."""
    dest = Path(backup_root) / session_dir.parent.name / session_dir.name
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {"source": str(session_dir), "dest": str(dest),
                "created": time.strftime("%Y-%m-%d %H:%M:%S"), "files": []}
    ok = True
    for src in sorted(session_dir.rglob("*")):
        if not src.is_file() or src.name == "backup_manifest.json":
            continue
        rel = src.relative_to(session_dir)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        src_hash = sha256(src)
        dst_hash = sha256(target)
        verified = src_hash == dst_hash
        ok = ok and verified
        manifest["files"].append({
            "path": str(rel), "bytes": src.stat().st_size,
            "sha256": src_hash, "verified": verified,
        })
    manifest["verified"] = ok
    manifest["file_count"] = len(manifest["files"])
    (dest / "backup_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    (session_dir / "backup_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def validate_session(session_dir: Path):
    session_dir = Path(session_dir)
    session_meta = None
    session_path = session_dir / "session.json"
    if session_path.is_file():
        try:
            raw = json.loads(session_path.read_text(encoding="utf-8"))
            block = raw.get("block") or {}
            session_meta = {
                "participant_id": raw.get("participant_id"),
                "task": block.get("task"),
                "condition": block.get("condition"),
                "interface": block.get("interface"),
            }
        except Exception as exc:
            print(f"[Validate][WARN] session.json unreadable: {exc}")

    events_by_episode = load_events(session_dir)
    episodes_dir = session_dir / "episodes"
    npz_files = sorted(episodes_dir.rglob("*.npz")) if episodes_dir.is_dir() \
        else sorted(session_dir.rglob("*.npz"))

    results = [validate_episode(p, session_meta, events_by_episode) for p in npz_files]

    durations = [r["duration_s"] for r in results if r.get("duration_s")]
    dur_names = [r["name"] for r in results if r.get("duration_s")]
    frames = [r["frames"] for r in results if r.get("frames")]
    frame_names = [r["name"] for r in results if r.get("frames")]

    report = {
        "session_dir": str(session_dir),
        "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "episodes": len(results),
        "status_counts": {
            s: sum(1 for r in results if r["status"] == s)
            for s in ("pass", "info", "warn", "fail")
        },
        "outcome_counts": {},
        "duration_outliers": _iqr_outliers(durations, dur_names),
        "length_outliers": _iqr_outliers(frames, frame_names),
        "session_meta": session_meta,
        "events_indexed": len(events_by_episode or {}),
        "results": results,
    }
    for r in results:
        key = str(r.get("outcome") or "unknown")
        report["outcome_counts"][key] = report["outcome_counts"].get(key, 0) + 1
    if events_by_episode is None:
        report["status_counts"]["warn"] += 1
        report["missing_events_log"] = True
    return report


def render_markdown(report):
    lines = [f"# Data quality report", "",
             f"- session: `{report['session_dir']}`",
             f"- generated: {report['generated']}",
             f"- episodes: {report['episodes']}",
             f"- outcomes: " + ", ".join(f"{k}={v}" for k, v in
                                         sorted(report["outcome_counts"].items())),
             f"- status: " + ", ".join(f"{k}={v}" for k, v in
                                       report["status_counts"].items()),
             ""]
    if report.get("missing_events_log"):
        lines += ["> **events.jsonl missing** — intervention boundaries could not be "
                  "cross-checked against the event log.", ""]

    lines += ["## Episodes", "",
              "| episode | outcome | frames | duration | ACC | interventions | status |",
              "|---|---|---|---|---|---|---|"]
    for r in report["results"]:
        acc = "—"
        if r.get("acc_present"):
            acc = "varies" if r.get("acc_nonconstant") else "constant"
        dur = f"{r['duration_s']:.1f}s" if r.get("duration_s") else "—"
        lines.append(
            f"| {r['name']} | {r.get('outcome') or '—'} | {r.get('frames') or '—'} | "
            f"{dur} | {acc} | {len(r.get('intervention_segments') or [])} | "
            f"**{r['status']}** |")
    lines.append("")

    problems = [r for r in report["results"] if r["counts"]["fail"] or r["counts"]["warn"]]
    if problems:
        lines += ["## Findings", ""]
        for r in problems:
            lines.append(f"### {r['name']}")
            lines.append("")
            for item in r["findings"]:
                if item["severity"] == "info":
                    continue
                mark = "FAIL" if item["severity"] == "fail" else "WARN"
                lines.append(f"- **{mark}** [{item['check']}] {item['message']}")
            lines.append("")
    else:
        lines += ["## Findings", "", "No warnings or failures.", ""]

    if report["duration_outliers"] or report["length_outliers"]:
        lines += ["## Outliers (IQR)", ""]
        for name, value in report["duration_outliers"]:
            lines.append(f"- duration: `{name}` = {value:.1f}s")
        for name, value in report["length_outliers"]:
            lines.append(f"- length: `{name}` = {value:.0f} frames")
        lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_dir", help="Session directory (contains session.json).")
    ap.add_argument("--recursive", action="store_true",
                    help="Treat the path as a participant directory and validate every "
                         "session inside it.")
    ap.add_argument("--backup", default=None,
                    help="After validating, copy the session tree here and verify with "
                         "SHA-256. Run this at the end of every participant.")
    ap.add_argument("--quiet", action="store_true", help="Only print the summary line.")
    args = ap.parse_args()

    root = Path(args.session_dir)
    if not root.is_dir():
        print(f"[Validate][ERROR] not a directory: {root}", file=sys.stderr)
        return 2

    sessions = ([p for p in sorted(root.iterdir())
                 if p.is_dir() and (p / "session.json").is_file()]
                if args.recursive else [root])
    if not sessions:
        print(f"[Validate][ERROR] no sessions found under {root}", file=sys.stderr)
        return 2

    exit_code = 0
    for session in sessions:
        report = validate_session(session)
        md = render_markdown(report)
        (session / "quality_report.json").write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8")
        (session / "quality_report.md").write_text(md + "\n", encoding="utf-8")
        if not args.quiet:
            print(md)
        failed = report["status_counts"].get("fail", 0)
        print(f"[Validate] {session.name}: {report['episodes']} episode(s), "
              f"{failed} failing, {report['status_counts'].get('warn', 0)} with warnings "
              f"-> {session / 'quality_report.md'}")
        if failed:
            exit_code = 1

        if args.backup:
            manifest = backup_session(session, Path(args.backup))
            status = "verified" if manifest["verified"] else "MISMATCH"
            print(f"[Backup] {manifest['file_count']} file(s) -> {manifest['dest']} "
                  f"[{status}]")
            if not manifest["verified"]:
                print("[Backup][ERROR] checksum mismatch — do not wipe the source!",
                      file=sys.stderr)
                exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
