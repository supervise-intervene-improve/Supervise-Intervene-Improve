import argparse
import csv
import shutil
from pathlib import Path

import numpy as np


DEFAULT_INPUT_DIR = Path("CUP_demonstrations/demo_pick_place_CUPS/boxes_cups")


def default_output_dir(input_dir: Path) -> Path:
    return input_dir.with_name(input_dir.name + "_gripper_fixed")


def fix_gripper_signal(
    signal: np.ndarray,
    *,
    closed_value: float,
    closed_threshold: float,
    already_closed_tolerance: float,
    mode: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    fixed = np.asarray(signal, dtype=np.float64).copy()
    before_min = float(np.min(fixed))
    before_max = float(np.max(fixed))

    closed_mask = fixed <= closed_threshold
    needs_fix = before_min > closed_value + already_closed_tolerance and bool(np.any(closed_mask))

    if needs_fix:
        if mode == "clamp":
            fixed[closed_mask] = closed_value
        elif mode == "rescale":
            # Preserve transition shape: old min -> closed_value, threshold -> threshold.
            denom = closed_threshold - before_min
            if denom <= 1e-9:
                fixed[closed_mask] = closed_value
            else:
                alpha = (fixed[closed_mask] - before_min) / denom
                fixed[closed_mask] = closed_value + alpha * (closed_threshold - closed_value)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        fixed = np.maximum(fixed, closed_value)

    stats = {
        "before_min": before_min,
        "before_max": before_max,
        "after_min": float(np.min(fixed)),
        "after_max": float(np.max(fixed)),
        "closed_frames": int(np.count_nonzero(closed_mask)),
        "changed_frames": int(np.count_nonzero(np.abs(fixed - signal) > 1e-9)),
        "needs_fix": bool(needs_fix),
    }
    return fixed.astype(signal.dtype, copy=False), closed_mask, stats


def load_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=False)
    return {key: data[key] for key in data.files}


def save_npz(path: Path, data: dict[str, np.ndarray], *, compressed: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        np.savez_compressed(path, **data)
    else:
        np.savez(path, **data)


def process_episode(
    src: Path,
    dst: Path,
    *,
    closed_value: float,
    closed_threshold: float,
    already_closed_tolerance: float,
    mode: str,
    fix_grip_real: bool,
    compressed: bool,
    dry_run: bool,
) -> dict:
    data = load_npz(src)
    if "ctrl_sim" not in data:
        raise KeyError(f"{src} missing ctrl_sim")

    ctrl_sim = np.asarray(data["ctrl_sim"]).copy()
    if ctrl_sim.ndim != 2 or ctrl_sim.shape[1] <= 7:
        raise ValueError(f"{src} ctrl_sim must have shape (T, >=8), got {ctrl_sim.shape}")

    fixed_ctrl, closed_mask, stats = fix_gripper_signal(
        ctrl_sim[:, 7],
        closed_value=closed_value,
        closed_threshold=closed_threshold,
        already_closed_tolerance=already_closed_tolerance,
        mode=mode,
    )
    ctrl_sim[:, 7] = fixed_ctrl
    data["ctrl_sim"] = ctrl_sim

    real_changed_frames = 0
    if fix_grip_real and "grip_real" in data and stats["needs_fix"]:
        grip_real = np.asarray(data["grip_real"]).copy()
        real_before = grip_real.copy()
        grip_real[closed_mask[: len(grip_real)]] = np.minimum(
            grip_real[closed_mask[: len(grip_real)]],
            closed_value,
        )
        data["grip_real"] = grip_real
        real_changed_frames = int(np.count_nonzero(np.abs(grip_real - real_before) > 1e-9))

    if not dry_run:
        save_npz(dst, data, compressed=compressed)

    return {
        "episode": src.name,
        "frames": int(ctrl_sim.shape[0]),
        **stats,
        "real_changed_frames": real_changed_frames,
        "output": str(dst),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fix inconsistent fully-closed cup gripper commands in demonstration NPZ files."
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--closed-value", type=float, default=0.008)
    parser.add_argument(
        "--closed-threshold",
        type=float,
        default=0.025,
        help="Frames with ctrl_sim[:,7] <= this are considered closed/closing.",
    )
    parser.add_argument(
        "--already-closed-tolerance",
        type=float,
        default=1e-6,
        help="Skip episodes whose minimum is already at --closed-value within this tolerance.",
    )
    parser.add_argument(
        "--mode",
        choices=["rescale", "clamp"],
        default="rescale",
        help="rescale preserves transition shape; clamp sets all closed frames exactly to --closed-value.",
    )
    parser.add_argument(
        "--fix-grip-real",
        action="store_true",
        help="Also lower grip_real on detected closed frames. Default fixes only ctrl_sim.",
    )
    parser.add_argument("--compressed", action="store_true", help="Write compressed .npz files.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite input files after creating .bak files. Otherwise writes to --output-dir.",
    )
    args = parser.parse_args()

    if args.closed_threshold <= args.closed_value:
        raise ValueError("--closed-threshold must be larger than --closed-value")
    if args.output_dir is not None and args.in_place:
        raise ValueError("Use either --output-dir or --in-place, not both")

    return args


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)

    episode_paths = sorted(input_dir.glob("*.npz"))
    if not episode_paths:
        raise FileNotFoundError(f"No .npz files found in {input_dir}")

    output_dir = args.output_dir or default_output_dir(input_dir)
    rows = []

    print(f"[INFO] Input:  {input_dir}")
    if args.in_place:
        print("[INFO] Output: in-place, with .bak backups")
    else:
        print(f"[INFO] Output: {output_dir}")
    print(
        "[INFO] Fix: ctrl_sim[:,7], "
        f"closed_value={args.closed_value:g}, closed_threshold={args.closed_threshold:g}, mode={args.mode}"
    )
    if args.dry_run:
        print("[INFO] Dry run: no files will be written")

    for src in episode_paths:
        if args.in_place:
            dst = src
            backup = src.with_suffix(src.suffix + ".bak")
            if not args.dry_run and not backup.exists():
                shutil.copy2(src, backup)
        else:
            dst = output_dir / src.name

        row = process_episode(
            src,
            dst,
            closed_value=args.closed_value,
            closed_threshold=args.closed_threshold,
            already_closed_tolerance=args.already_closed_tolerance,
            mode=args.mode,
            fix_grip_real=args.fix_grip_real,
            compressed=args.compressed,
            dry_run=args.dry_run,
        )
        rows.append(row)

    changed = [row for row in rows if row["changed_frames"] > 0]
    print(f"[SUMMARY] episodes={len(rows)} changed={len(changed)}")
    if changed:
        print("[SUMMARY] Changed episodes:")
        for row in changed:
            print(
                f"  {row['episode']}: min {row['before_min']:.5f} -> {row['after_min']:.5f}, "
                f"frames {row['changed_frames']}/{row['frames']}"
            )

    if not args.dry_run:
        summary_path = (input_dir if args.in_place else output_dir) / "gripper_fix_summary.csv"
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[SUMMARY] Wrote {summary_path}")


if __name__ == "__main__":
    main()
