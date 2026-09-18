"""Validate the OOD model-variant pipeline before a study run starts.

Replaces the blanket "cups OOD is refused" hard-fail in `run_main_policy.sh` with an
actual check: read every corpus descriptor, compile a sample of them, and prove they are
drop-in compatible with the live base scene. Exits non-zero if anything is wrong, so a
misconfigured run dies at launch rather than halfway through a participant session.

Also prints the numbers that decide the grid viewer's memory budget (compile ms, resident
MB, estimated VRAM per `MjrContext`), because those are machine-specific and worth seeing
before nine sessions start.

    python -m model_variants.preflight --xml <scene.xml> --corpus <ood_scenes> --task cups

Runs under both interpreters (conda `polymetis` MuJoCo 3.4.0 and repo `.venv` 3.3.7).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def _rss_mb() -> float:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    except Exception:
        return float("nan")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xml", required=True, help="the LIVE base scene (respects LAB_FAST)")
    parser.add_argument("--corpus", required=True, help="ood_scenes root, or the task subdir")
    parser.add_argument("--task", default="cups")
    parser.add_argument("--max-scenes", type=int, default=3,
                        help="how many variants to actually compile (0 = all)")
    parser.add_argument("--dump-fingerprint", action="store_true",
                        help="print the base fingerprint as JSON (cross-interpreter check)")
    parser.add_argument("--descriptor", default="",
                        help="with --dump-fingerprint: fingerprint THIS variant instead")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    import mujoco
    from data_io.ood_scene_variants import (
        MODEL_VARIANT_TASKS, list_variant_scenes, scene_variant, task_dir_for,
    )
    from model_variants.builder import (
        VariantIncompatible, build_model, describe_reference, validate_against_reference,
    )
    from model_variants.descriptor import scene_closure_sha256, variant_key

    def say(*a):
        if not args.quiet:
            print(*a)

    xml = Path(args.xml).resolve()
    if not xml.is_file():
        print(f"[Preflight][ERROR] scene not found: {xml}", file=sys.stderr)
        return 2

    task = args.task.strip().lower()
    base_sha = scene_closure_sha256(xml)

    t0 = time.perf_counter()
    base = build_model(xml, None)
    base_ms = (time.perf_counter() - t0) * 1e3
    ref = describe_reference(base)

    if args.dump_fingerprint:
        model = base
        if args.descriptor:
            desc = json.loads(Path(args.descriptor).read_text()
                              if Path(args.descriptor).is_file() else args.descriptor)
            model = build_model(xml, desc, reference_model=base)
        print(json.dumps(describe_reference(model), sort_keys=True))
        return 0

    say(f"[Preflight] scene           : {xml}")
    say(f"[Preflight] closure sha256  : {base_sha[:16]}...")
    say(f"[Preflight] base compile    : {base_ms:.0f} ms   "
        f"nq={ref['nq']} nv={ref['nv']} nu={ref['nu']} ncam={ref['ncam']} ngeom={ref['ngeom']}")
    say(f"[Preflight] mujoco          : {mujoco.__version__}   python={sys.executable}")
    est_vram = (base.ntexdata + base.nmeshvert * 32 + base.nmeshface * 12) / 1e6
    say(f"[Preflight] per-slot cost   : ~{_rss_mb():.0f} MB resident now, "
        f"~{est_vram:.0f} MB VRAM per MjrContext "
        f"(tex {base.ntexdata/1e6:.0f} MB + {base.nmeshvert:,} verts)")

    if task not in MODEL_VARIANT_TASKS:
        say(f"[Preflight] task {task!r} is pose-only OOD; no model variants needed. OK")
        return 0

    corpus = Path(args.corpus).resolve()
    task_dir = task_dir_for(task, corpus)
    if task_dir is None or not task_dir.is_dir():
        # Tolerate being handed the task subdir directly.
        task_dir = corpus if corpus.is_dir() else None
    if task_dir is None or not task_dir.is_dir():
        print(f"[Preflight][ERROR] corpus dir not found: {args.corpus}", file=sys.stderr)
        return 2

    scenes = sorted(p for p in task_dir.glob("*.xml") if p.is_file())
    if not scenes:
        print(f"[Preflight][ERROR] no corpus scenes in {task_dir}", file=sys.stderr)
        return 2

    descriptors, rejected = [], []
    for scene in scenes:
        desc, prov = scene_variant(scene, task, task_dir, base_scene_sha256=base_sha)
        if desc is None:
            rejected.append(scene.name)
        else:
            descriptors.append((scene, desc, prov))

    if not descriptors:
        print(f"[Preflight][ERROR] every corpus scene in {task_dir} was refused "
              f"({len(rejected)} scenes). OOD cannot run.", file=sys.stderr)
        return 1

    keys = {variant_key(d) for _, d, _ in descriptors}
    zs = [z for _, d, _ in descriptors for z in d["cup_scale_z"].values()]
    say(f"[Preflight] corpus          : {len(descriptors)}/{len(scenes)} scenes readable, "
        f"{len(keys)} distinct variants")
    say(f"[Preflight] cup Z-scale     : {min(zs):.6f} .. {max(zs):.6f} over {len(zs)} samples")
    if rejected:
        say(f"[Preflight][WARN] refused {len(rejected)} scene(s): {rejected[:5]}"
            f"{' ...' if len(rejected) > 5 else ''}")

    n = len(descriptors) if args.max_scenes <= 0 else min(args.max_scenes, len(descriptors))
    if n <= 0:
        say("[Preflight] compile check skipped (--max-scenes 0 means all; none available)")
        return 0
    # First, middle, last -- a contiguous head would miss a corpus that degrades later.
    if n >= len(descriptors):
        sample = descriptors
    else:
        idxs = sorted({0, len(descriptors) // 2, len(descriptors) - 1} |
                      set(range(1, max(1, n - 2) + 1)))[:n]
        sample = [descriptors[i] for i in idxs]

    failures = []
    total_ms = 0.0
    for scene, desc, _prov in sample:
        try:
            t0 = time.perf_counter()
            model = build_model(xml, desc, reference_model=base)
            dt = (time.perf_counter() - t0) * 1e3
            total_ms += dt
            validate_against_reference(model, ref)
            say(f"[Preflight]   {scene.name:26s} {variant_key(desc)}  {dt:7.0f} ms  OK")
        except VariantIncompatible as exc:
            failures.append((scene.name, str(exc)))
            print(f"[Preflight][ERROR] {scene.name}: {exc}", file=sys.stderr)

    if failures:
        print(f"[Preflight][ERROR] {len(failures)}/{len(sample)} variant(s) incompatible with "
              f"{xml.name}. Refusing to launch.", file=sys.stderr)
        return 1

    say(f"[Preflight] compiled        : {len(sample)} variant(s), "
        f"avg {total_ms/max(1,len(sample)):.0f} ms, peak RSS {_rss_mb():.0f} MB")
    say("[Preflight] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
