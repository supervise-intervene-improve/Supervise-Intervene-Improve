#!/usr/bin/env python3
"""Archive a permanent set of LIVE-camera-frame episodes as a re-render regression fixture.

Why this exists
---------------
Switching the recorder to state-only (state-only recording decision) is irreversible per
episode: once an episode is recorded without images, no tool can put them back. The claim
that offline re-rendering reproduces the live frames (measured: mean 0.673/255) can then
never be re-verified against anything, because there is nothing left that contains live
frames.

This archives a stratified sample of the existing live-frame episodes, verified by SHA-256,
so `utils/verify_rerender_fidelity.py` keeps a ground truth to test against forever — after a
MuJoCo upgrade, a scene edit, or a renderer change.

Run this BEFORE enabling INTERVENE_RECORD_RGB=0.

Usage
-----
    python utils/archive_golden_corpus.py --dry-run
    python utils/archive_golden_corpus.py                     # ~40 episodes, stratified by date
    python utils/archive_golden_corpus.py --per-day 10 --dest /media/backup/golden

Verify an existing archive at any time:

    python utils/archive_golden_corpus.py --verify
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

DEFAULT_SRC = "INTERVENTION_DATA"
DEFAULT_DEST = "INTERVENTION_DATA/_golden_live_frames"
MANIFEST = "golden_manifest.json"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def has_live_frames(npz: Path) -> bool:
    """True if the NPZ contains rgb_* arrays, without decompressing them."""
    import zipfile

    try:
        with zipfile.ZipFile(npz) as z:
            return any(n.startswith("rgb_") for n in z.namelist())
    except Exception:
        return False


def episode_day(npz: Path) -> str:
    stem = npz.stem
    for part in stem.split("_"):
        if len(part) == 8 and part.isdigit():
            return part
    return "unknown"


def collect(src: Path, dest: Path, per_day: int):
    """Stratify by day so the fixture spans scene/behaviour drift, not one afternoon."""
    by_day = defaultdict(list)
    for npz in sorted(src.rglob("*.npz")):
        if dest in npz.parents:
            continue
        if not has_live_frames(npz):
            continue
        by_day[episode_day(npz)].append(npz)

    chosen = []
    for day in sorted(by_day):
        files = by_day[day]
        if len(files) <= per_day:
            chosen.extend(files)
            continue
        # Evenly spaced across the day rather than the first N, so the sample spans a
        # session's start, middle and end.
        step = len(files) / float(per_day)
        chosen.extend(files[int(i * step)] for i in range(per_day))
    return by_day, chosen


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Archive live-frame episodes as a permanent re-render regression fixture.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--dest", default=DEFAULT_DEST)
    ap.add_argument("--per-day", type=int, default=5,
                    help="Episodes to keep per recording day.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--verify", action="store_true",
                    help="Re-check an existing archive against its manifest and exit.")
    args = ap.parse_args()

    src, dest = Path(args.src), Path(args.dest)
    manifest_path = dest / MANIFEST

    if args.verify:
        if not manifest_path.is_file():
            print(f"[Golden][ERROR] no manifest at {manifest_path}", file=sys.stderr)
            return 2
        manifest = json.loads(manifest_path.read_text())
        bad = missing = 0
        for entry in manifest["files"]:
            f = dest / entry["name"]
            if not f.is_file():
                print(f"  MISSING  {entry['name']}")
                missing += 1
            elif sha256(f) != entry["sha256"]:
                print(f"  CORRUPT  {entry['name']}")
                bad += 1
        total = len(manifest["files"])
        print(f"\n  {total - bad - missing}/{total} files verified by SHA-256"
              f"  (corrupt={bad} missing={missing})")
        return 0 if (bad == 0 and missing == 0) else 1

    if not src.is_dir():
        print(f"[Golden][ERROR] source not found: {src}", file=sys.stderr)
        return 2

    by_day, chosen = collect(src, dest, args.per_day)
    if not chosen:
        print("[Golden][ERROR] no episodes with live camera frames found — "
              "has state-only recording already been enabled?", file=sys.stderr)
        return 2

    total_bytes = 0
    pairs = []
    for npz in chosen:
        side = npz.with_suffix(".json")
        total_bytes += npz.stat().st_size + (side.stat().st_size if side.is_file() else 0)
        pairs.append((npz, side if side.is_file() else None))

    print(f"  source            : {src}")
    print(f"  live-frame days   : {len(by_day)} "
          f"({', '.join(f'{d}:{len(v)}' for d, v in sorted(by_day.items()))})")
    print(f"  selected          : {len(chosen)} episodes ({args.per_day}/day)")
    print(f"  size              : {total_bytes / 1e9:.2f} GB")
    print(f"  destination       : {dest}")
    missing_side = sum(1 for _, s in pairs if s is None)
    if missing_side:
        print(f"  [WARN] {missing_side} episode(s) have no sidecar .json — those cannot be "
              f"re-rendered later (the scene XML lives only in the sidecar).")

    if args.dry_run:
        print("\n  --dry-run: nothing copied.")
        return 0

    dest.mkdir(parents=True, exist_ok=True)
    files = []
    for npz, side in pairs:
        for f in (npz, side):
            if f is None:
                continue
            target = dest / f.name
            shutil.copy2(f, target)
            digest = sha256(target)
            if digest != sha256(f):
                print(f"[Golden][ERROR] checksum mismatch after copying {f.name}",
                      file=sys.stderr)
                return 1
            files.append({"name": f.name, "sha256": digest, "bytes": target.stat().st_size})

    manifest = {
        "purpose": "Permanent live-camera-frame fixture for utils/verify_rerender_fidelity.py. "
                   "Recorded before state-only recording (the state-only recording decision) was "
                   "enabled. Do not delete: once all new episodes are state-only, this is the "
                   "only ground truth that can re-verify offline re-render fidelity.",
        "source": str(src),
        "per_day": args.per_day,
        "episodes": len(chosen),
        "files": files,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n  archived {len(files)} files, all SHA-256 verified")
    print(f"  manifest: {manifest_path}")
    print(f"\n  verify at any time:  python utils/archive_golden_corpus.py --verify")
    print(f"  use as fixture    :  MUJOCO_GL=egl python utils/verify_rerender_fidelity.py \\")
    print(f"                         --episodes_glob '{dest}/*.npz' --episodes 12")
    return 0


if __name__ == "__main__":
    sys.exit(main())
