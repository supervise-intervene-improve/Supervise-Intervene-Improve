#!/usr/bin/env python3
"""Re-score ACC offline by replaying recorded episodes through the policy.

Why this exists
---------------
The live `ensemble_disagreement` estimator needs a fresh action chunk per step, and each
chunk needs an observation — which means rendering 3 policy cameras and swapping the GL
context. Measured on 2026-07-28 across 9 policy processes, that pushed MuJoCo render time
from ~15 ms to ~70 ms on EVERY session, selected or not. The live path is now capped at
one extra observation per chunk, which is cheap but only yields ~2 overlapping predictions.

For research-grade ACC there is a better option: the recorded episode already contains
exactly what the policy consumed. `TrajectoryRecorder` saves `right,left,wrist` at
224x224 — the policy's own camera set and input resolution — plus `qpos_sim`, `qvel_sim`
and `ctrl_sim`, which is everything `build_state()` needs. So the policy can be replayed
after the session with a full ensemble window and ZERO cost to the live simulation.

This is the "proxy sim that isn't shown": a faithful replay, not a second live process.

Usage
-----
    python utils/rescore_acc.py <session_dir>
    python utils/rescore_acc.py <session_dir> --method ensemble_disagreement --window 8
    python utils/rescore_acc.py <episode.npz> --dry-run

Writes `acc_offline`, `acc_offline_components`, `acc_offline_valid` into each NPZ and
records the config in the `.json` sidecar. Reports live-vs-offline agreement so a renderer
mismatch between the recorder and the policy shows up as a number rather than silently.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve()
INTERVENE_ROOT = THIS.parents[1]
if str(INTERVENE_ROOT) not in sys.path:
    sys.path.insert(0, str(INTERVENE_ROOT))


def _lazy_imports():
    """Imported late so --help works without torch/lerobot/mujoco present."""
    import torch
    import mujoco
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.policies.factory import make_pre_post_processors
    from playback.acc_scorer import AccScorer, COMPONENT_NAMES
    from utils.rollout_act_mujoco import (
        build_state,
        chw_float01_from_rgb,
        get_qpos_indices_for_joints,
        policy_image_camera_names,
        PANDA_JOINT_NAMES,
    )
    return dict(
        torch=torch, mujoco=mujoco, ACTPolicy=ACTPolicy,
        make_pre_post_processors=make_pre_post_processors,
        AccScorer=AccScorer, COMPONENT_NAMES=COMPONENT_NAMES,
        build_state=build_state, chw_float01_from_rgb=chw_float01_from_rgb,
        get_qpos_indices_for_joints=get_qpos_indices_for_joints,
        policy_image_camera_names=policy_image_camera_names,
        PANDA_JOINT_NAMES=PANDA_JOINT_NAMES,
    )


class _OfflineFrameRenderer:
    """Reconstruct the frames a state-only episode did not store.

    Mirrors the LIVE recorder's render flags (rendering/viewer.py::set_fast_visuals).
    mujoco.Renderer enables shadows by default and the live capture does not — that one flag
    is the entire difference between a 13-25/255 mismatch and the measured 0.673/255 match
    (utils/verify_rerender_fidelity.py). The flags must be re-applied after every
    update_scene(), which repopulates the scene.
    """

    def __init__(self, mujoco_mod, xml_path: str, width: int, height: int):
        self._mj = mujoco_mod
        self.model = mujoco_mod.MjModel.from_xml_path(xml_path)
        self.data = mujoco_mod.MjData(self.model)
        self.renderer = mujoco_mod.Renderer(self.model, height=height, width=width)
        self.xml_path = xml_path

    def render(self, qpos, qvel, ctrl, camera: str):
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        n_ctrl = min(self.data.ctrl.shape[0], ctrl.shape[0])
        self.data.ctrl[:n_ctrl] = ctrl[:n_ctrl]
        self._mj.mj_forward(self.model, self.data)
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(self.data, camera=camera)
        flags = self.renderer.scene.flags
        flags[self._mj.mjtRndFlag.mjRND_SHADOW] = 0
        flags[self._mj.mjtRndFlag.mjRND_REFLECTION] = 0
        flags[self._mj.mjtRndFlag.mjRND_SKYBOX] = 0
        flags[self._mj.mjtRndFlag.mjRND_FOG] = 0
        return self.renderer.render().astype(np.uint8)

    def close(self):
        try:
            self.renderer.close()
        except Exception:
            pass


def find_episodes(target: Path):
    if target.is_file() and target.suffix == ".npz":
        return [target]
    if not target.is_dir():
        return []
    episodes_dir = target / "episodes"
    root = episodes_dir if episodes_dir.is_dir() else target
    return sorted(p for p in root.rglob("*.npz"))


def sidecar_for(npz_path: Path):
    path = npz_path.with_suffix(".json")
    if not path.is_file():
        return {}, path
    try:
        return json.loads(path.read_text(encoding="utf-8")), path
    except Exception:
        return {}, path


def resolve_checkpoint(explicit, sidecars):
    if explicit:
        return Path(explicit)
    for meta in sidecars:
        ckpt = meta.get("checkpoint")
        if ckpt and Path(ckpt).is_dir():
            return Path(ckpt)
    return None


def rescore_episode(npz_path: Path, ctx, policy, pre, post, cfg, args):
    """Replay one episode and return the per-frame offline ACC arrays."""
    with np.load(npz_path, allow_pickle=False) as z:
        keys = set(z.files)
        required = {"qpos_sim", "qvel_sim", "ctrl_sim"}
        missing = required - keys
        if missing:
            return None, f"missing {sorted(missing)}"
        qpos = z["qpos_sim"]
        qvel = z["qvel_sim"]
        ctrl = z["ctrl_sim"]
        n = qpos.shape[0]
        # State-only episodes (INTERVENE_RECORD_RGB=0) carry no rgb_* arrays; their frames
        # are reconstructed from qpos/qvel/ctrl instead. Returning None here used to print a
        # bare SKIP and exit 0 — offline ACC silently produced nothing for a whole session.
        cam_arrays = {}
        renderer = cfg.get("renderer")
        missing_cams = [c for c in cfg["camera_names"] if f"rgb_{c}" not in keys]
        if missing_cams and renderer is None:
            return None, (
                f"missing camera arrays {sorted(f'rgb_{c}' for c in missing_cams)} and no "
                f"offline renderer available (recorded cameras: "
                f"{sorted(k for k in keys if k.startswith('rgb_'))})"
            )
        for cam in cfg["camera_names"]:
            key = f"rgb_{cam}"
            if key in keys:
                cam_arrays[cam] = z[key]
        live_acc = z["acc"] if "acc" in keys else None
        intervention = z["intervention"].astype(bool) if "intervention" in keys else \
            np.zeros(n, dtype=bool)
        # Replay the ACTUAL query pattern. Assuming every frame was a fresh policy query
        # would inflate `replan_jump` (it fires on query steps) and zero `chunk_age`, so
        # the offline components would not be comparable to the live ones.
        queried = z["queried_policy"].astype(bool) if "queried_policy" in keys else None

    torch = ctx["torch"]
    scorer = ctx["AccScorer"](
        method=args.method,
        arm_action_mode=cfg["arm_action_mode"],
        ensemble_window=args.window,
    )
    n_comp = len(ctx["COMPONENT_NAMES"])
    acc_out = np.full(n, np.nan, dtype=np.float32)
    comp_out = np.full((n, n_comp), np.nan, dtype=np.float32)
    valid_out = np.zeros(n, dtype=bool)

    reset_len = max(1, int(cfg.get("reset_episode_len", n)))
    with torch.inference_mode():
        for i in range(n):
            # Reconstruct exactly what the policy saw at this frame.
            phase = 0.0 if reset_len <= 1 else min(i / (reset_len - 1), 1.0)
            state = ctx["build_state"](
                qpos_sim=qpos[i].astype(np.float32),
                qvel_sim=qvel[i].astype(np.float32),
                ctrl_sim_prev=ctrl[i].astype(np.float32),
                qpos_indices=cfg["qpos_indices"],
                state_dim=cfg["state_dim"],
                phase=phase,
            )
            obs = {"observation.state": torch.from_numpy(state)}
            for cam in cfg["camera_names"]:
                if cam in cam_arrays:
                    frame_rgb = cam_arrays[cam][i]
                else:
                    # State-only episode: reconstruct the frame the policy would have seen.
                    # Validated at mean 0.673/255 against live captures
                    # (utils/verify_rerender_fidelity.py).
                    frame_rgb = cfg["renderer"].render(qpos[i], qvel[i], ctrl[i], cam)
                obs[f"observation.images.{cam}"] = ctx["chw_float01_from_rgb"](frame_rgb)
            # A chunk is predicted on EVERY frame — that is the advantage of scoring
            # offline, and it is what gives the ensemble a full overlap window.
            chunk = policy.predict_action_chunk(pre(obs))
            scorer.note_chunk(post(chunk), step=i)

            # The executed arm command for this frame is ctrl_sim[:7]; on human frames it
            # is the operator's, which is exactly what we want the residual measured
            # against.
            action = ctrl[i][:8].astype(np.float32)
            q_now = qpos[i][cfg["qpos_indices"]].astype(np.float32)
            step_queried = bool(queried[i]) if queried is not None else True
            acc_out[i] = scorer.update(q_now, action, queried_policy=step_queried)
            comp_out[i] = scorer.components_vector()
            valid_out[i] = scorer.last_valid

    stats = {
        "frames": int(n),
        "valid": int(valid_out.sum()),
        "acc_mean": float(np.nanmean(acc_out)),
        "acc_max": float(np.nanmax(acc_out)),
        "human_frames": int(intervention.sum()),
    }
    if live_acc is not None and live_acc.shape[0] == n:
        both = np.isfinite(live_acc) & np.isfinite(acc_out)
        if both.any():
            stats["live_vs_offline_mean_abs_diff"] = float(
                np.mean(np.abs(live_acc[both] - acc_out[both]))
            )
            stats["live_acc_mean"] = float(np.mean(live_acc[both]))
    return (acc_out, comp_out, valid_out, stats), None


def write_back(npz_path: Path, acc, comp, valid, args, ctx, cfg, stats):
    with np.load(npz_path, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    if "acc_offline" in data and not args.overwrite:
        return False, "already scored (use --overwrite)"
    data["acc_offline"] = acc
    data["acc_offline_components"] = comp
    data["acc_offline_valid"] = valid
    data["acc_offline_method"] = np.array(args.method)
    data["acc_offline_component_names"] = np.array(",".join(ctx["COMPONENT_NAMES"]))
    # The temp name MUST end in .npz: np.savez_compressed appends ".npz" to any path that
    # does not, so "x.npz.tmp" would silently become "x.npz.tmp.npz" and the rename fails.
    tmp = npz_path.with_name(npz_path.stem + ".rescore-tmp.npz")
    np.savez_compressed(tmp, **data)
    tmp.replace(npz_path)

    meta, meta_path = sidecar_for(npz_path)
    meta["acc_offline"] = {
        "method": args.method,
        "ensemble_window": args.window,
        "checkpoint": str(cfg["checkpoint"]),
        "component_names": list(ctx["COMPONENT_NAMES"]),
        "scored_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stats": stats,
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    return True, None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("target", help="Session directory, participant directory, or one .npz")
    ap.add_argument("--checkpoint", default=None,
                    help="Policy checkpoint. Default: read from the episode .json sidecar.")
    ap.add_argument("--method", default="ensemble_disagreement",
                    help="ACC method (default ensemble_disagreement — the whole point of "
                         "scoring offline).")
    ap.add_argument("--window", type=int, default=8, help="Ensemble window (chunks).")
    ap.add_argument("--device", default=None, help="torch device (default: cuda if available).")
    ap.add_argument("--overwrite", action="store_true", help="Re-score already-scored episodes.")
    ap.add_argument("--dry-run", action="store_true", help="Score but do not write back.")
    ap.add_argument("--limit", type=int, default=0, help="Only process the first N episodes.")
    args = ap.parse_args()

    target = Path(args.target)
    episodes = find_episodes(target)
    if not episodes:
        print(f"[RescoreACC][ERROR] no .npz episodes found under {target}", file=sys.stderr)
        return 2
    if args.limit > 0:
        episodes = episodes[: args.limit]

    sidecars = [sidecar_for(p)[0] for p in episodes]
    ckpt = resolve_checkpoint(args.checkpoint, sidecars)
    if ckpt is None or not Path(ckpt).is_dir():
        print("[RescoreACC][ERROR] no usable checkpoint. Pass --checkpoint /path/to/"
              "pretrained_model (the episode sidecars did not contain a valid one).",
              file=sys.stderr)
        return 2

    ctx = _lazy_imports()
    torch = ctx["torch"]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[RescoreACC] checkpoint: {ckpt}")
    print(f"[RescoreACC] device: {device}  method={args.method} window={args.window}")

    policy = ctx["ACTPolicy"].from_pretrained(str(ckpt))
    policy.eval()
    policy.to(device)
    pre, post = ctx["make_pre_post_processors"](
        policy.config,
        pretrained_path=str(ckpt),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )

    # The MuJoCo model is needed only to map joint names to qpos indices — the same
    # mapping PolicyPlayer uses, so offline and live states agree exactly.
    xml_path = None
    for meta in sidecars:
        if meta.get("mujoco_xml_path") and Path(meta["mujoco_xml_path"]).is_file():
            xml_path = meta["mujoco_xml_path"]
            break
    if xml_path is None:
        print("[RescoreACC][ERROR] no mujoco_xml_path in any sidecar; cannot resolve "
              "joint indices.", file=sys.stderr)
        return 2
    model = ctx["mujoco"].MjModel.from_xml_path(xml_path)

    # Offline renderer for state-only episodes. Built lazily-but-once: episodes that still
    # carry rgb_* arrays use them, so this costs nothing on legacy data.
    renderer = None
    if any(not m.get("save_rgb", True) for m in sidecars) or not sidecars:
        rgb_w = next((int(m["rgb_width"]) for m in sidecars if m.get("rgb_width")), 224)
        rgb_h = next((int(m["rgb_height"]) for m in sidecars if m.get("rgb_height")), 224)
        renderer = _OfflineFrameRenderer(ctx["mujoco"], xml_path, rgb_w, rgb_h)
        print(f"[RescoreACC] state-only episodes detected -> re-rendering frames offline "
              f"from {Path(xml_path).name} at {rgb_w}x{rgb_h}")

    cfg = {
        "camera_names": ctx["policy_image_camera_names"](policy),
        "qpos_indices": ctx["get_qpos_indices_for_joints"](model, ctx["PANDA_JOINT_NAMES"]),
        "state_dim": policy.config.input_features["observation.state"].shape[0],
        "arm_action_mode": os.environ.get("ARM_ACTION_MODE", "absolute"),
        "checkpoint": ckpt,
        "reset_episode_len": 0,
        "renderer": renderer,
    }
    print(f"[RescoreACC] policy cameras: {cfg['camera_names']}  state_dim={cfg['state_dim']}")
    print(f"[RescoreACC] {len(episodes)} episode(s)\n")

    n_ok = n_skip = 0
    diffs = []
    for path in episodes:
        meta, _ = sidecar_for(path)
        cfg["reset_episode_len"] = int(meta.get("n_frames") or 0) or 0
        t0 = time.time()
        result, err = rescore_episode(path, ctx, policy, pre, post, cfg, args)
        if result is None:
            print(f"  SKIP  {path.name}: {err}")
            n_skip += 1
            continue
        acc, comp, valid, stats = result
        note = ""
        if not args.dry_run:
            wrote, why = write_back(path, acc, comp, valid, args, ctx, cfg, stats)
            if not wrote:
                note = f"  (not written: {why})"
                n_skip += 1
            else:
                n_ok += 1
        else:
            n_ok += 1
            note = "  (dry-run)"
        agree = stats.get("live_vs_offline_mean_abs_diff")
        if agree is not None:
            diffs.append(agree)
        print(f"  OK    {path.name}: frames={stats['frames']} valid={stats['valid']} "
              f"acc_mean={stats['acc_mean']:.3f} max={stats['acc_max']:.3f} "
              + (f"live|offline_diff={agree:.3f} " if agree is not None else "")
              + f"({time.time() - t0:.1f}s){note}")

    print(f"\n[RescoreACC] scored={n_ok} skipped={n_skip}")
    if diffs:
        print(f"[RescoreACC] mean |live - offline| ACC = {np.mean(diffs):.4f}")
        print("[RescoreACC] Note: the live and offline scores use DIFFERENT estimators by "
              "default (live=chunk_residual, offline=ensemble), so a large difference here "
              "is expected. Compare like-for-like with --method chunk_residual to check "
              "for a recorder-vs-policy renderer mismatch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
