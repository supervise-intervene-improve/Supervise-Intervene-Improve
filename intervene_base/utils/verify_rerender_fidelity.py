#!/usr/bin/env python3
"""Measure how closely offline re-rendering reproduces the LIVE recorded camera frames.

Why this exists
---------------
The episode recorder captures 3 cameras every policy frame on every session. Measured at 9
windows that is 270 render+readback operations per second -- 62% of all GPU work in the
system -- and none of it is displayed to anyone; it exists only to build the training set.

`utils/convert_npz_to_lerobot.py --rerender_images` can rebuild those frames afterwards from
`qpos_sim/qvel_sim/ctrl_sim`, which would remove that cost from the live path entirely. This
tool answers the question that decision depends on: *are the reconstructed frames actually
the same images?*

The answer turned out to hinge on a single render flag. `mujoco.Renderer` enables shadows by
default; the live recorder does not. With shadows on the frames differ by mean 13-25/255 --
with them off, by 0.4-1.4/255. Run with `--shadows on` to see that for yourself.

Usage
-----
    MUJOCO_GL=egl python utils/verify_rerender_fidelity.py
    MUJOCO_GL=egl python utils/verify_rerender_fidelity.py --frames 30 --episodes 10
    MUJOCO_GL=egl python utils/verify_rerender_fidelity.py --shadows on   # show the failure

Exit code is 0 when the mean difference is within --tolerance, 1 otherwise, so this can gate
a pipeline change in CI.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np

# EGL keeps this runnable headless / over SSH. Set before importing mujoco.
os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco  # noqa: E402

DEFAULT_XML = "mujoco_scenes/working_scenes/with_soft_gripper/sii_scene_table_T_shape.xml"
DEFAULT_EPISODE_GLOB = "INTERVENTION_DATA/**/*.npz"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Compare offline MuJoCo re-render against live-recorded episode frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--xml", default=DEFAULT_XML,
                    help="Scene XML the episodes were recorded with.")
    ap.add_argument("--episodes_glob", default=DEFAULT_EPISODE_GLOB,
                    help="Glob for episode NPZs (recursive).")
    ap.add_argument("--episodes", type=int, default=3,
                    help="How many of the NEWEST matching episodes to check.")
    ap.add_argument("--frames", type=int, default=12,
                    help="Frames sampled evenly across each episode.")
    ap.add_argument("--shadows", choices=["off", "on"], default="off",
                    help="Render flag for the offline pass. 'on' reproduces the original "
                         "mismatch and is useful for demonstrating the root cause.")
    ap.add_argument("--tolerance", type=float, default=2.0,
                    help="Max acceptable mean |difference| on a 0-255 scale.")
    ap.add_argument("--quiet", action="store_true", help="Only print the summary.")
    return ap.parse_args()


def find_episodes(pattern: str, count: int) -> list[str]:
    files = [f for f in glob.glob(pattern, recursive=True) if f.endswith(".npz")]
    files.sort(key=os.path.getmtime)
    return files[-count:] if count > 0 else files


def render_frame(model, data, renderer, npz, t: int, cam: str, shadows: bool) -> np.ndarray:
    """Reproduce one recorded frame from state alone."""
    data.qpos[:] = npz["qpos_sim"][t]
    data.qvel[:] = npz["qvel_sim"][t]
    ctrl = npz["ctrl_sim"][t]
    n = min(data.ctrl.shape[0], ctrl.shape[0])
    data.ctrl[:n] = ctrl[:n]
    mujoco.mj_forward(model, data)

    renderer.disable_depth_rendering()
    renderer.update_scene(data, camera=cam)
    # Mirror the LIVE recorder exactly (rendering/viewer.py::set_fast_visuals clears all
    # four). SHADOW is the one that matters — mujoco.Renderer enables it by default and the
    # live capture does not, which alone accounts for a 13-25/255 mismatch. The other three
    # make no difference in the current scenes but are cleared so a future XML that adds a
    # skybox, fog or a reflective floor cannot silently reintroduce one.
    # Must be applied AFTER update_scene(), which repopulates the scene each call.
    flags = renderer.scene.flags
    flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1 if shadows else 0
    if not shadows:
        flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
        flags[mujoco.mjtRndFlag.mjRND_FOG] = 0
    return renderer.render().astype(np.int16)


def main() -> int:
    args = parse_args()

    if not Path(args.xml).is_file():
        print(f"[ERROR] scene XML not found: {args.xml}", file=sys.stderr)
        return 2

    episodes = find_episodes(args.episodes_glob, args.episodes)
    if not episodes:
        print(f"[ERROR] no episodes matched: {args.episodes_glob}", file=sys.stderr)
        return 2

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    shadows = args.shadows == "on"

    print(f"Offline re-render vs LIVE recorded frames   (shadows {args.shadows.upper()})")
    print(f"  xml      : {args.xml}")
    print(f"  episodes : {len(episodes)}  frames/episode: {args.frames}\n")
    if not args.quiet:
        print(f"  {'episode':<44} {'cam':>6} {'mean':>7} {'p99':>6} {'max':>5} {'identical':>10}")

    all_diffs: list[np.ndarray] = []
    renderer = None
    render_seconds = 0.0
    render_count = 0

    try:
        for path in episodes:
            npz = np.load(path, allow_pickle=True)
            cams = [k[4:] for k in npz.files if k.startswith("rgb_")]
            if not cams:
                print(f"  {Path(path).name:<44} (no rgb_* arrays — already state-only)")
                continue

            h, w = npz[f"rgb_{cams[0]}"].shape[1:3]
            if renderer is None:
                renderer = mujoco.Renderer(model, height=h, width=w)

            n_frames = len(npz["ctrl_sim"])
            step = max(1, n_frames // max(1, args.frames))
            idx = list(range(0, n_frames, step))[: args.frames]

            for cam in cams:
                diffs = []
                identical = []
                for t in idx:
                    t0 = time.perf_counter()
                    got = render_frame(model, data, renderer, npz, t, cam, shadows)
                    render_seconds += time.perf_counter() - t0
                    render_count += 1
                    want = npz[f"rgb_{cam}"][t].astype(np.int16)
                    d = np.abs(got - want)
                    diffs.append(d)
                    identical.append((d == 0).mean())

                flat = np.concatenate([d.ravel() for d in diffs])
                all_diffs.append(flat)
                if not args.quiet:
                    print(f"  {Path(path).name[:44]:<44} {cam:>6} {flat.mean():>7.3f} "
                          f"{np.percentile(flat, 99):>6.1f} {flat.max():>5.0f} "
                          f"{100 * np.mean(identical):>9.1f}%")
    finally:
        if renderer is not None:
            renderer.close()

    if not all_diffs:
        print("\n[ERROR] nothing compared — no episodes contained camera frames.", file=sys.stderr)
        return 2

    flat = np.concatenate(all_diffs)
    mean = float(flat.mean())
    per_render_ms = 1000.0 * render_seconds / max(1, render_count)

    print(f"\n  OVERALL mean|diff| = {mean:.3f}/255 ({100 * mean / 255:.2f}%)   "
          f"p99 = {np.percentile(flat, 99):.1f}   identical = {100 * (flat == 0).mean():.1f}%")
    print(f"  offline re-render cost = {per_render_ms:.2f} ms/frame/camera "
          f"(batched, no GPU contention)")

    ok = mean <= args.tolerance
    if ok:
        print(f"\n  PASS — within --tolerance {args.tolerance}. Offline reconstruction is "
              f"faithful; live image capture is redundant for these episodes.")
    else:
        print(f"\n  FAIL — exceeds --tolerance {args.tolerance}.")
        if shadows:
            print("  You ran with --shadows on. That is the known root cause; "
                  "re-run with the default --shadows off.")
        else:
            print("  Check that --xml matches the scene the episodes were recorded with, and "
                  "that no visual randomisation happens outside qpos/qvel/ctrl.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
