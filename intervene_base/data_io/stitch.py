from pathlib import Path

import numpy as np


def stitch_npz(original_path: str, suffix_path: str, cut_idx: int, out_path: str):
    orig = np.load(original_path)
    suf = np.load(suffix_path)

    out = {}
    suffix_start = 0

    sim_t_offset = 0.0
    if "sim_t" in orig.files and len(orig["sim_t"]) > 0:
        sim_t_offset = float(orig["sim_t"][cut_idx])

    for k in orig.files:
        if k in suf.files and orig[k].ndim >= 1:
            left = orig[k][: cut_idx + 1]
            right = suf[k][suffix_start:]

            if k == "sim_t":
                right = right + sim_t_offset

            out[k] = np.concatenate([left, right], axis=0)
        else:
            out[k] = orig[k]

    for k in suf.files:
        if k not in out:
            if k in {"intervention", "is_intervention"}:
                right = np.asarray(suf[k]).reshape(-1)[suffix_start:]
                left = np.zeros((int(cut_idx) + 1,), dtype=right.dtype)
                out[k] = np.concatenate([left, right], axis=0)
            elif k in {"action_source", "intervention_id"}:
                right = np.asarray(suf[k]).reshape(-1)[suffix_start:]
                left = np.zeros((int(cut_idx) + 1,), dtype=right.dtype)
                out[k] = np.concatenate([left, right], axis=0)
            elif k == "grip_real":
                right = np.asarray(suf[k], dtype=np.float64).reshape(-1)[suffix_start:]
                prefix_len = int(cut_idx) + 1

                if "ctrl_sim" in orig.files:
                    ctrl_prefix = np.asarray(orig["ctrl_sim"][:prefix_len], dtype=np.float64)
                    if ctrl_prefix.ndim == 2 and ctrl_prefix.shape[1] > 7:
                        left = np.clip(ctrl_prefix[:, 7] * 2.0, 0.0, 0.08)
                    else:
                        left = np.full((prefix_len,), np.nan, dtype=np.float64)
                else:
                    left = np.full((prefix_len,), np.nan, dtype=np.float64)

                out[k] = np.concatenate([left, right], axis=0)
            else:
                out[k] = suf[k]

    np.savez_compressed(out_path, **out)
    print(f"[INFO] Stitched saved: {out_path}")


def make_replanned_output_path(log_path: str) -> str:
    p = Path(log_path)
    return str(p.with_name(p.stem + "_replanned.npz"))
