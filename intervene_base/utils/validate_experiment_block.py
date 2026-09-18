"""Integrity check for one recorded HRI experiment block.

This is deliberately NOT an analysis tool. It answers one question -- "is this block
readable, complete and internally consistent enough to analyse later?" -- and never
computes a study statistic. Counts appear only where they are needed to say whether
something is missing.

It complements `utils/validate_study_data.py`, which checks episode NPZs frame by
frame. This one checks the things that only exist at BLOCK level: the shared
`events.jsonl` written by nine policy processes plus the grid and the VR runtimes, the
protocol ordering inside it, and whether the scenario metadata is sufficient to
re-render observations that were deliberately never stored.

Exit codes: 0 = no FAILs (warnings are fine), 1 = at least one FAIL. Warnings never
fail a block: a study run must not be discarded over a non-critical inconsistency.

    python utils/validate_experiment_block.py <block_dir>
    python utils/validate_experiment_block.py <block_dir> --backup /media/backup
    python utils/validate_experiment_block.py experiment_data/P003 --recursive
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from data_io.experiment_block import (  # noqa: E402
    CONDITION_MATRIX,
    STUDIES,
    TASKS,
    summarize_events,
)

FAIL = "fail"
WARN = "warn"
INFO = "info"

# Events whose ordering the protocol constrains. Anything else is free-form.
PHASE_ORDER = (
    "INTERVENTION_REQUEST",
    "CONTROLLER_PREPARATION_START",
    "READY",
    "HUMAN_CONTROL_START",
    "RELEASE",
)


class Report:
    def __init__(self, block_dir: Path):
        self.block_dir = block_dir
        self.checks = []

    def add(self, level, check, message, **extra):
        self.checks.append({"level": level, "check": check, "message": message, **extra})

    def ok(self, check, message, **extra):
        self.add(INFO, check, message, **extra)

    @property
    def fails(self):
        return [c for c in self.checks if c["level"] == FAIL]

    @property
    def warns(self):
        return [c for c in self.checks if c["level"] == WARN]

    def to_dict(self, summary=None):
        return {
            "block_dir": str(self.block_dir),
            "fails": len(self.fails),
            "warnings": len(self.warns),
            "checks": self.checks,
            "summary": summary or {},
        }

    def to_markdown(self, summary=None):
        lines = [f"# Block quality report", "", f"`{self.block_dir}`", ""]
        verdict = "FAIL" if self.fails else ("WARN" if self.warns else "PASS")
        lines += [f"**Verdict: {verdict}**  "
                  f"({len(self.fails)} fail, {len(self.warns)} warn)", ""]
        if summary:
            lines += ["## Counts", ""]
            for key in ("episodes_seen", "interventions_total", "events_total"):
                if key in summary:
                    lines.append(f"- {key}: {summary[key]}")
            if summary.get("episode_outcomes"):
                lines.append(f"- outcomes: {summary['episode_outcomes']}")
            if summary.get("cells_seen") is not None:
                lines.append(f"- cells: {summary['cells_seen']}")
            lines.append("")
        for level, title in ((FAIL, "Failures"), (WARN, "Warnings"), (INFO, "Notes")):
            rows = [c for c in self.checks if c["level"] == level]
            if not rows:
                continue
            lines += [f"## {title}", ""]
            for c in rows:
                lines.append(f"- **{c['check']}**: {c['message']}")
            lines.append("")
        return "\n".join(lines)


# ------------------------------------------------------------------ event loading

def load_events(events_path: Path, report: Report):
    """Read events.jsonl, tolerating one torn final line (hard-kill artefact)."""
    events = []
    bad = 0
    if not events_path.is_file():
        report.add(FAIL, "events_present", f"events.jsonl missing at {events_path}")
        return events
    with open(events_path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
                # One torn line is the expected artefact of a hard kill mid-write;
                # more than one means something else corrupted the file.
                note = ("a single torn tail line is expected after a hard kill"
                        if bad == 1 else "multiple corrupt lines")
                report.add(WARN if bad == 1 else FAIL, "events_readable",
                           f"line {lineno} is not valid JSON ({note})")
    if events:
        report.ok("events_readable", f"{len(events)} event(s) parsed")
    else:
        report.add(FAIL, "events_readable", "events.jsonl contains no events")
    return events


# ---------------------------------------------------------------------- the checks

def check_metadata(block_dir: Path, report: Report):
    path = block_dir / "block_metadata.json"
    if not path.is_file():
        report.add(FAIL, "metadata_present", "block_metadata.json is missing")
        return None
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        report.add(FAIL, "metadata_present", f"block_metadata.json is not valid JSON: {exc}")
        return None

    block = meta.get("block") or {}
    required = ("participant_id", "study", "condition_id", "supervision_interface",
                "controller_interface", "task", "block_id")
    missing = [k for k in required if not block.get(k)]
    if missing:
        report.add(FAIL, "metadata_fields", f"block_metadata.json is missing {missing}")
    else:
        report.ok("metadata_fields", "all required identity fields present")

    study, cond = block.get("study"), block.get("condition_id")
    if study not in STUDIES:
        report.add(FAIL, "known_condition", f"unknown study {study!r}")
    elif (study, cond) not in CONDITION_MATRIX:
        report.add(FAIL, "known_condition",
                   f"unknown condition {cond!r} for study {study}")
    else:
        expected = CONDITION_MATRIX[(study, cond)]
        actual = (block.get("supervision_interface"), block.get("controller_interface"))
        if actual != expected:
            report.add(FAIL, "known_condition",
                       f"{study}/{cond} should be {expected} but the block records {actual}")
        else:
            report.ok("known_condition", f"{study}/{cond} = {expected[0]} + {expected[1]}")

    if block.get("task") not in TASKS:
        report.add(FAIL, "known_task", f"unknown task {block.get('task')!r}")
    else:
        report.ok("known_task", f"task = {block.get('task')}")

    if meta.get("block_start_timestamp") is None:
        report.add(FAIL, "block_start_recorded", "block_start_timestamp is missing")
    if meta.get("status") != "complete":
        report.add(WARN, "block_complete",
                   f"block status is {meta.get('status')!r} — it was interrupted. The data "
                   "is kept and usable; note it as incomplete rather than deleting it.")
    else:
        report.ok("block_complete", "block was stopped cleanly")

    env = meta.get("environment") or {}
    if not env.get("git_sha_umbrella"):
        report.add(WARN, "provenance", "no git commit recorded for the umbrella repo")
    if env.get("git_dirty_umbrella"):
        report.add(WARN, "provenance",
                   "the working tree had uncommitted changes; the git SHA alone does "
                   "not reproduce this software version")
    return meta


def check_block_events(events, report: Report):
    kinds = defaultdict(int)
    for ev in events:
        kinds[ev.get("event")] += 1
    if not kinds.get("BLOCK_START"):
        report.add(FAIL, "block_start_event", "no BLOCK_START event")
    if not kinds.get("BLOCK_END"):
        report.add(WARN, "block_end_event",
                   "no BLOCK_END event — the block was interrupted before it stopped")
    return kinds


def check_timestamps(events, report: Report):
    """Monotonic per stream. Streams are per (source, pid): `mono_t` is CLOCK_MONOTONIC
    and therefore shared across processes on one host, but interleaving means the FILE
    is not globally sorted, and that is expected rather than a defect."""
    streams = defaultdict(list)
    for ev in events:
        key = (ev.get("source"), ev.get("pid"))
        mono = ev.get("mono_t")
        if isinstance(mono, (int, float)):
            streams[key].append(mono)
    bad = []
    for key, values in streams.items():
        for a, b in zip(values, values[1:]):
            if b < a:
                bad.append(key)
                break
    if bad:
        report.add(FAIL, "timestamps_monotonic",
                   f"non-monotonic mono_t within stream(s): {bad}")
    else:
        report.ok("timestamps_monotonic",
                  f"{len(streams)} stream(s), all monotonic")

    missing = sum(1 for ev in events if ev.get("mono_t") is None)
    if missing:
        report.add(WARN, "timestamps_present", f"{missing} event(s) have no mono_t")


def check_cells(events, report: Report):
    cells = {ev.get("cell_id") for ev in events if ev.get("cell_id") is not None}
    if not cells:
        report.add(WARN, "cells_present", "no event carries a cell_id")
    else:
        report.ok("cells_present", f"cells seen: {sorted(cells)}")

    # Episode ids must be unique to one cell. With nine cells writing into one
    # directory this is the collision that would silently merge two participants'
    # episodes into one id.
    owner = {}
    collisions = set()
    for ev in events:
        eid, cid = ev.get("episode_id"), ev.get("cell_id")
        if not eid or cid is None:
            continue
        if eid in owner and owner[eid] != cid:
            collisions.add(eid)
        owner.setdefault(eid, cid)
    if collisions:
        report.add(FAIL, "episode_ids_unique",
                   f"episode_id(s) used by more than one cell: {sorted(collisions)}")
    elif owner:
        report.ok("episode_ids_unique", f"{len(owner)} episode id(s), each on one cell")


def check_selection(events, report: Report):
    selections = [ev for ev in events if ev.get("event") == "CELL_SELECTED"]
    if not selections:
        report.add(WARN, "cell_selected_present",
                   "no CELL_SELECTED events — inspection time cannot be derived. "
                   "Check that the grid (mouse) or the VR runtime (quest_ray) attached "
                   "to this block.")
        return
    methods = {ev.get("selection_method") for ev in selections}
    unknown = methods - {"mouse", "quest_ray"}
    if unknown:
        report.add(WARN, "cell_selected_method",
                   f"unexpected selection_method value(s): {sorted(unknown)}")
    missing_id = sum(1 for ev in selections if ev.get("selected_cell_id") is None)
    if missing_id:
        report.add(FAIL, "cell_selected_fields",
                   f"{missing_id} CELL_SELECTED event(s) carry no selected_cell_id")
    report.ok("cell_selected_present",
              f"{len(selections)} confirmed selection(s), methods={sorted(methods)}")


def check_intervention_protocol(events, report: Report):
    """Ordering rules the study depends on, checked per (cell, intervention_id).

    Three of these are explicit study requirements: READY must follow a REQUEST, no
    human-control interval may exist without a READY, and every RELEASE must be
    followed by a POST_RELEASE_CHECK.
    """
    by_iv = defaultdict(list)
    for ev in events:
        name = ev.get("event")
        if name in PHASE_ORDER or name in ("POST_RELEASE_CHECK", "AUTONOMY_RESUMED"):
            key = (ev.get("cell_id"), ev.get("intervention_id"))
            by_iv[key].append(ev)

    if not by_iv:
        report.add(WARN, "interventions_present",
                   "no intervention events in this block")
        return

    n_ok = 0
    for (cell, iv_id), evs in sorted(by_iv.items(), key=lambda kv: str(kv[0])):
        evs = sorted(evs, key=lambda e: (e.get("mono_t") or 0.0))
        names = [e.get("event") for e in evs]
        where = f"cell {cell} intervention {iv_id}"

        def index_of(name):
            return names.index(name) if name in names else None

        i_req = index_of("INTERVENTION_REQUEST")
        i_ready = index_of("READY")
        i_human = index_of("HUMAN_CONTROL_START")
        i_rel = index_of("RELEASE")
        i_post = index_of("POST_RELEASE_CHECK")

        if i_ready is not None and (i_req is None or i_ready < i_req):
            report.add(FAIL, "ready_after_request",
                       f"{where}: READY without a preceding INTERVENTION_REQUEST")
        if i_human is not None and (i_ready is None or i_human < i_ready):
            report.add(FAIL, "human_control_after_ready",
                       f"{where}: HUMAN_CONTROL_START without a preceding READY")
        if i_rel is not None and i_human is None:
            report.add(WARN, "release_after_human_control",
                       f"{where}: RELEASE without HUMAN_CONTROL_START — the takeover "
                       "ended before control was handed over (a failed or cancelled "
                       "start looks like this)")
        if i_rel is not None and i_human is not None and i_post is None:
            report.add(FAIL, "post_release_check_present",
                       f"{where}: RELEASE is not followed by a POST_RELEASE_CHECK")
        if i_post is not None and i_rel is None:
            report.add(FAIL, "post_release_check_present",
                       f"{where}: POST_RELEASE_CHECK without a RELEASE")
        if i_req is not None and i_ready is not None and i_human is not None:
            n_ok += 1
    report.ok("intervention_protocol",
              f"{len(by_iv)} intervention(s) checked, {n_ok} completed the full phase "
              "sequence")


def check_post_release_semantics(events, report: Report):
    """The recovery classification the study rests on must be decidable.

    A POST_RELEASE_CHECK with status `success` means the human finished the task and
    the policy must NOT have acted afterwards. That is checkable here because every
    event carries `policy_step`.
    """
    by_cell = defaultdict(list)
    for ev in events:
        by_cell[ev.get("cell_id")].append(ev)

    checked = 0
    for cell, evs in by_cell.items():
        evs = sorted(evs, key=lambda e: (e.get("mono_t") or 0.0))
        for i, ev in enumerate(evs):
            if ev.get("event") != "POST_RELEASE_CHECK":
                continue
            checked += 1
            if ev.get("check_status") != "success":
                continue
            step = ev.get("policy_step")
            # The very next terminal event should be EPISODE_SUCCESS at the same
            # policy_step: a higher one means the policy acted in between, which would
            # make the episode an autonomous completion mislabelled as a human recovery.
            for later in evs[i + 1:]:
                name = later.get("event")
                if name in ("EPISODE_SUCCESS", "EPISODE_FAILURE", "EPISODE_TIMEOUT"):
                    lstep = later.get("policy_step")
                    if (isinstance(step, int) and isinstance(lstep, int)
                            and lstep > step):
                        report.add(
                            WARN, "post_release_no_policy_action",
                            f"cell {cell}: POST_RELEASE_CHECK reported success at "
                            f"policy_step {step} but {name} is at {lstep} — the policy "
                            "appears to have acted in between",
                        )
                    break
    if checked:
        report.ok("post_release_semantics", f"{checked} post-release check(s) inspected")


def check_episode_files(block_dir: Path, events, report: Report):
    """Trajectory files readable, and scenario metadata sufficient to re-render."""
    episodes_dir = block_dir / "episodes"
    if not episodes_dir.is_dir():
        report.add(WARN, "episodes_dir", "no episodes/ directory")
        return
    npzs = sorted(p for p in episodes_dir.rglob("*.npz"))
    if not npzs:
        report.add(WARN, "episode_files", "no episode .npz files were written")
        return

    unreadable, no_render_cfg, no_seed, no_labels = [], [], [], []
    frame_arrays = ("wall_t", "qpos_sim", "qvel_sim", "ctrl_sim", "intervention",
                    "policy_step", "sim_step", "control_state")
    length_mismatch = []
    image_arrays = []
    for path in npzs:
        try:
            with np.load(path, allow_pickle=False) as data:
                keys = set(data.files)
                lengths = {k: int(data[k].shape[0]) for k in frame_arrays if k in keys}
                if len(set(lengths.values())) > 1:
                    length_mismatch.append((path.name, lengths))
                # Camera frames must NOT be here: this study records state only and
                # re-renders observations offline.
                imgs = sorted(k for k in keys
                              if k.startswith("rgb_") or k.startswith("depth_"))
                if imgs:
                    image_arrays.append((path.name, imgs))
                if "episode_seed" not in keys:
                    no_seed.append(path.name)
                if not {"participant_id", "block_id", "cell_id"} & keys:
                    no_labels.append(path.name)
        except Exception as exc:
            unreadable.append((path.name, str(exc)))
            continue
        sidecar = path.with_suffix(".json")
        try:
            meta = json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception:
            no_render_cfg.append(path.name)
            continue
        cfg = meta.get("render_config")
        if not isinstance(cfg, dict) or not cfg.get("cameras"):
            no_render_cfg.append(path.name)

    if unreadable:
        report.add(FAIL, "trajectories_readable",
                   f"{len(unreadable)} unreadable .npz: {unreadable[:3]}")
    else:
        report.ok("trajectories_readable", f"{len(npzs)} trajectory file(s) readable")
    if length_mismatch:
        report.add(FAIL, "frame_arrays_aligned",
                   f"per-frame arrays disagree on length: {length_mismatch[:3]}")
    if no_render_cfg:
        report.add(WARN, "rerender_metadata",
                   f"{len(no_render_cfg)} episode(s) have no render_config in their "
                   "sidecar — observations for those cannot be re-rendered from the "
                   f"recorded state alone: {no_render_cfg[:3]}")
    else:
        report.ok("rerender_metadata",
                  "every episode carries camera/timestep metadata for re-rendering")
    if no_seed:
        report.add(FAIL, "scenario_seed",
                   f"{len(no_seed)} episode(s) have no episode_seed: {no_seed[:3]}")
    if no_labels:
        report.add(WARN, "episode_labels",
                   f"{len(no_labels)} episode(s) carry no block identity scalars")

    if image_arrays:
        report.add(WARN, "no_camera_frames",
                   f"{len(image_arrays)} episode(s) contain stored camera frames "
                   f"({image_arrays[0][1][:3]}...). This block was expected to be "
                   "state-only; storage will be ~100x larger than planned.")
    else:
        report.ok("no_camera_frames",
                  "no RGB/depth/point-cloud frames stored (state-only, as intended)")

    # Every episode id in the events should have a file, and vice versa.
    event_ids = {ev.get("episode_id") for ev in events if ev.get("episode_id")}
    file_ids = {p.stem for p in npzs}
    missing_files = event_ids - file_ids
    if missing_files:
        report.add(WARN, "episode_files_match_events",
                   f"{len(missing_files)} episode(s) appear in events.jsonl with no "
                   f"trajectory file (an episode that was still open at shutdown looks "
                   f"like this): {sorted(missing_files)[:3]}")
    orphans = file_ids - event_ids
    if orphans:
        report.add(WARN, "episode_files_match_events",
                   f"{len(orphans)} trajectory file(s) have no events: "
                   f"{sorted(orphans)[:3]}")


# ---------------------------------------------------------------------- driver

def validate_block(block_dir: Path, *, quiet: bool = False) -> Report:
    block_dir = Path(block_dir)
    report = Report(block_dir)
    check_metadata(block_dir, report)
    events = load_events(block_dir / "events.jsonl", report)
    if events:
        check_block_events(events, report)
        check_timestamps(events, report)
        check_cells(events, report)
        check_selection(events, report)
        check_intervention_protocol(events, report)
        check_post_release_semantics(events, report)
    check_episode_files(block_dir, events, report)

    summary = summarize_events(block_dir / "events.jsonl")
    (block_dir / "block_quality_report.json").write_text(
        json.dumps(report.to_dict(summary), indent=2, default=str), encoding="utf-8")
    (block_dir / "block_quality_report.md").write_text(
        report.to_markdown(summary), encoding="utf-8")

    if not quiet:
        print(report.to_markdown(summary))
    verdict = "FAIL" if report.fails else ("WARN" if report.warns else "PASS")
    print(f"[Validate] {verdict}  {block_dir}  "
          f"({len(report.fails)} fail, {len(report.warns)} warn)")
    return report


def find_blocks(root: Path):
    if (root / "block_metadata.json").is_file():
        return [root]
    return sorted(p.parent for p in root.rglob("block_metadata.json"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("path", help="A block directory, or a parent to search.")
    ap.add_argument("--recursive", action="store_true",
                    help="Validate every block under `path`.")
    ap.add_argument("--quiet", action="store_true", help="Only print the verdict line.")
    ap.add_argument("--backup", default=None,
                    help="Copy the block there and verify every file by SHA-256.")
    args = ap.parse_args(argv)

    root = Path(args.path)
    if not root.exists():
        print(f"[Validate][ERROR] {root} does not exist", file=sys.stderr)
        return 2
    blocks = find_blocks(root) if (args.recursive or not
                                   (root / "block_metadata.json").is_file()) else [root]
    if not blocks:
        print(f"[Validate][ERROR] no block_metadata.json found under {root}",
              file=sys.stderr)
        return 2

    worst = 0
    for block_dir in blocks:
        report = validate_block(block_dir, quiet=args.quiet)
        if report.fails:
            worst = 1
        if args.backup and not backup_block(block_dir, Path(args.backup)):
            worst = 1
    return worst


def backup_block(block_dir: Path, backup_root: Path) -> bool:
    """Copy the block and verify every file by SHA-256. Never reports a false success.

    Reuses `validate_study_data.sha256` rather than a second hashing helper. The
    destination keeps `<participant>/<condition>/<task>/<block_id>` so backups of
    different conditions cannot collide on the block directory name alone.
    """
    from utils.validate_study_data import sha256

    import shutil
    import time

    rel_parts = block_dir.resolve().parts[-4:]  # participant/condition/task/block_id
    dest = Path(backup_root).joinpath(*rel_parts)
    dest.mkdir(parents=True, exist_ok=True)
    manifest = {"source": str(block_dir), "dest": str(dest),
                "created": time.strftime("%Y-%m-%d %H:%M:%S"), "files": []}
    ok = True
    for src in sorted(block_dir.rglob("*")):
        if not src.is_file() or src.name == "backup_manifest.json":
            continue
        rel = src.relative_to(block_dir)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        verified = sha256(src) == sha256(target)
        ok = ok and verified
        manifest["files"].append({"path": str(rel), "bytes": src.stat().st_size,
                                  "sha256": sha256(src), "verified": verified})
    manifest["all_verified"] = ok
    for side in (block_dir, dest):
        (side / "backup_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")
    if ok:
        print(f"[Backup] {len(manifest['files'])} file(s) verified -> {dest}")
    else:
        print("[Backup][ERROR] checksum mismatch — do not wipe the source!",
              file=sys.stderr)
    return ok


if __name__ == "__main__":
    raise SystemExit(main())
