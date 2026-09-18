# utils/data_cleaning.py

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np


PANDA_JOINT_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
]

class StatsTracker:
    def __init__(self):
        self.names = []
        self.original_lengths = []
        self.trimmed_lengths = []
        self.original_durations = []
        self.trimmed_durations = []
        self.start_times = []
        self.end_times = []
        self.kept_ratios = []

    def add(self, name: str, t: np.ndarray, start_idx: int, end_idx: int):
        original_len = len(t)
        trimmed_len = end_idx - start_idx + 1

        original_duration = t[-1] - t[0]
        trimmed_duration = t[end_idx] - t[start_idx]

        kept_ratio = trimmed_len / original_len

        self.names.append(name)
        self.original_lengths.append(original_len)
        self.trimmed_lengths.append(trimmed_len)
        self.original_durations.append(original_duration)
        self.trimmed_durations.append(trimmed_duration)
        self.start_times.append(t[start_idx])
        self.end_times.append(t[end_idx])
        self.kept_ratios.append(kept_ratio)

    def _find_outliers_iqr(self, values, factor=1.5):
        values = np.asarray(values, dtype=float)

        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1

        lower = q1 - factor * iqr
        upper = q3 + factor * iqr

        return np.where((values < lower) | (values > upper))[0], lower, upper

    def _report_outliers(self, label: str, values, unit: str = ""):
        idxs, lower, upper = self._find_outliers_iqr(values)

        if len(idxs) == 0:
            return []

        odd = []

        print(f"\n⚠️ Odd demos by {label}")
        print(f"Expected range: {lower:.3f} to {upper:.3f} {unit}")

        for idx in idxs:
            odd.append(idx)
            print(
                f"  - {self.names[idx]}: "
                f"{values[idx]:.3f} {unit}"
            )

        return odd

    def summary(self):
        if len(self.original_lengths) == 0:
            return

        def mean_std(values):
            values = np.asarray(values, dtype=float)
            return float(np.mean(values)), float(np.std(values))

        orig_len_m, orig_len_s = mean_std(self.original_lengths)
        trim_len_m, trim_len_s = mean_std(self.trimmed_lengths)

        orig_dur_m, orig_dur_s = mean_std(self.original_durations)
        trim_dur_m, trim_dur_s = mean_std(self.trimmed_durations)

        start_m, start_s = mean_std(self.start_times)
        end_m, end_s = mean_std(self.end_times)

        kept_m, kept_s = mean_std(self.kept_ratios)

        print("\n" + "=" * 60)
        print("DATA CLEANING SUMMARY")
        print("=" * 60)
        print(f"Episodes processed:       {len(self.names)}")
        print(f"Original samples:         {orig_len_m:.1f} ± {orig_len_s:.1f}")
        print(f"Trimmed samples:          {trim_len_m:.1f} ± {trim_len_s:.1f}")
        print(f"Original duration:        {orig_dur_m:.3f} ± {orig_dur_s:.3f} s")
        print(f"Trimmed duration:         {trim_dur_m:.3f} ± {trim_dur_s:.3f} s")
        print(f"Average trim start time:  {start_m:.3f} ± {start_s:.3f} s")
        print(f"Average trim end time:    {end_m:.3f} ± {end_s:.3f} s")
        print(f"Average kept ratio:       {kept_m:.3f} ± {kept_s:.3f}")
        print(f"Average removed ratio:    {1.0 - kept_m:.3f}")

        print("\n" + "-" * 60)
        print("ODD DEMONSTRATION CHECK")
        print("-" * 60)

        all_odd = set()

        checks = [
            ("trimmed duration", self.trimmed_durations, "s"),
            ("start time", self.start_times, "s"),
            ("end time", self.end_times, "s"),
            ("kept ratio", self.kept_ratios, ""),
            ("trimmed samples", self.trimmed_lengths, "frames"),
        ]

        for label, values, unit in checks:
            odd_idxs = self._report_outliers(label, values, unit)
            all_odd.update(odd_idxs)

        print("\n" + "-" * 60)

        if len(all_odd) == 0:
            print("✅ No statistically odd demonstrations found.")
        else:
            print("⚠️ Potentially odd demonstrations:")
            for idx in sorted(all_odd):
                print(
                    f"  - {self.names[idx]} | "
                    f"trimmed={self.trimmed_durations[idx]:.2f}s, "
                    f"start={self.start_times[idx]:.2f}s, "
                    f"end={self.end_times[idx]:.2f}s, "
                    f"kept={self.kept_ratios[idx]:.2f}"
                )

        print("=" * 60 + "\n")
        return [self.names[idx] for idx in sorted(all_odd)]


def load_episode(path: Path) -> dict:
    npz = np.load(path, allow_pickle=True)
    return {k: npz[k] for k in npz.files}


def save_episode(path: Path, episode: dict) -> None:
    np.savez_compressed(path, **episode)


def get_qpos_indices_for_joints(
    model: mujoco.MjModel,
    joint_names: list[str],
) -> list[int]:
    idxs = []

    for name in joint_names:
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            name,
        )

        if joint_id < 0:
            raise ValueError(f"Joint not found in model: {name}")

        qpos_adr = model.jnt_qposadr[joint_id]
        idxs.append(int(qpos_adr))

    return idxs


def get_robot_q_from_qpos_sim(
    model: mujoco.MjModel,
    qpos_sim: np.ndarray,
    joint_names: list[str],
) -> np.ndarray:
    qpos_idxs = get_qpos_indices_for_joints(model, joint_names)
    return qpos_sim[:, qpos_idxs]


def get_time_array(episode: dict, time_key: str) -> np.ndarray:
    if time_key not in episode:
        raise KeyError(f"Time key not found: {time_key}")

    t = episode[time_key].astype(np.float64)
    return t - t[0]


def extract_body_positions_from_qpos(
    model: mujoco.MjModel,
    qpos_sim: np.ndarray,
    body_name: str,
) -> np.ndarray:
    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_name,
    )

    if body_id < 0:
        raise ValueError(f"Body not found in model: {body_name}")

    data = mujoco.MjData(model)
    positions = []

    for i, qpos in enumerate(qpos_sim):
        if qpos.shape[0] != model.nq:
            raise ValueError(
                f"qpos_sim[{i}] has wrong size. "
                f"Expected model.nq={model.nq}, got {qpos.shape[0]}"
            )

        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        positions.append(data.xpos[body_id].copy())

    return np.asarray(positions)


def extract_many_body_positions_from_qpos(
    model: mujoco.MjModel,
    qpos_sim: np.ndarray,
    body_names: list[str],
) -> dict[str, np.ndarray]:
    body_ids = {}
    for body_name in body_names:
        body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            body_name,
        )

        if body_id < 0:
            raise ValueError(f"Body not found in model: {body_name}")

        body_ids[body_name] = body_id

    data = mujoco.MjData(model)
    positions = {body_name: [] for body_name in body_names}

    for i, qpos in enumerate(qpos_sim):
        if qpos.shape[0] != model.nq:
            raise ValueError(
                f"qpos_sim[{i}] has wrong size. "
                f"Expected model.nq={model.nq}, got {qpos.shape[0]}"
            )

        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)

        for body_name, body_id in body_ids.items():
            positions[body_name].append(data.xpos[body_id].copy())

    return {
        body_name: np.asarray(body_positions)
        for body_name, body_positions in positions.items()
    }


def get_body_geom_ids(model: mujoco.MjModel, body_name: str) -> set[int]:
    body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        body_name,
    )

    if body_id < 0:
        raise ValueError(f"Body not found in model: {body_name}")

    return set(np.where(model.geom_bodyid == body_id)[0].tolist())


def extract_body_contact_mask_from_qpos(
    model: mujoco.MjModel,
    qpos_sim: np.ndarray,
    body_name: str,
    target_body_name: str,
) -> np.ndarray:
    body_geom_ids = get_body_geom_ids(model, body_name)
    target_geom_ids = get_body_geom_ids(model, target_body_name)

    data = mujoco.MjData(model)
    contact_mask = []

    for i, qpos in enumerate(qpos_sim):
        if qpos.shape[0] != model.nq:
            raise ValueError(
                f"qpos_sim[{i}] has wrong size. "
                f"Expected model.nq={model.nq}, got {qpos.shape[0]}"
            )

        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)

        has_contact = False
        for contact_idx in range(data.ncon):
            contact = data.contact[contact_idx]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)

            if (
                geom1 in body_geom_ids
                and geom2 in target_geom_ids
            ) or (
                geom2 in body_geom_ids
                and geom1 in target_geom_ids
            ):
                has_contact = True
                break

        contact_mask.append(has_contact)

    return np.asarray(contact_mask, dtype=bool)


def select_most_moved_body(
    body_positions: dict[str, np.ndarray],
) -> tuple[str, np.ndarray]:
    displacements = {}
    for body_name, positions in body_positions.items():
        displacements[body_name] = float(
            np.max(np.linalg.norm(positions - positions[0], axis=1))
        )

    selected_body = max(displacements, key=displacements.get)
    print(
        "[INFO] Auto-selected body for trimming: "
        f"{selected_body} (max displacement={displacements[selected_body]:.3f} m)"
    )

    return selected_body, body_positions[selected_body]


def find_start_index_from_robot_motion(
    q_robot: np.ndarray,
    threshold: float,
    stable_initial_frames: int = 3,
) -> int:
    q_ref = np.mean(q_robot[:stable_initial_frames], axis=0)
    joint_delta = np.linalg.norm(q_robot - q_ref, axis=1)

    candidates = np.where(joint_delta > threshold)[0]

    if len(candidates) == 0:
        return 0

    return int(candidates[0])


def compute_speed(t: np.ndarray, pos: np.ndarray) -> np.ndarray:
    dt = np.diff(t)
    valid_dt = dt[dt > 1e-8]

    median_dt = float(np.median(valid_dt)) if len(valid_dt) > 0 else 0.1
    dt[dt <= 1e-8] = median_dt

    dp = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    speed = dp / dt

    return np.concatenate([[0.0], speed])


def find_end_index_from_body_stability(
    t: np.ndarray,
    body_pos: np.ndarray,
    start_idx: int,
    move_threshold: float,
    stable_speed_threshold: float,
    stable_duration_s: float,
) -> int:
    speed = compute_speed(t, body_pos)

    start_pos = body_pos[start_idx]
    displacement = np.linalg.norm(body_pos - start_pos, axis=1)

    moved_mask = displacement > move_threshold
    moved_mask[:start_idx] = False

    moved_indices = np.where(moved_mask)[0]

    if len(moved_indices) == 0:
        print("[WARN] Body did not move enough. Using final frame as end.")
        return len(t) - 1

    first_moved_idx = int(moved_indices[0])

    dt_values = np.diff(t)
    dt_values = dt_values[dt_values > 1e-8]
    median_dt = float(np.median(dt_values)) if len(dt_values) > 0 else 0.1

    stable_window = max(1, int(round(stable_duration_s / median_dt)))

    for i in range(first_moved_idx, len(t) - stable_window + 1):
        window_speed = speed[i : i + stable_window]

        if np.all(window_speed < stable_speed_threshold):
            return int(i)

    print("[WARN] Body did not become stable. Using final frame as end.")
    return len(t) - 1


def find_end_index_from_placement_stability(
    t: np.ndarray,
    body_pos: np.ndarray,
    target_pos: np.ndarray,
    start_idx: int,
    move_threshold: float,
    placement_xy_threshold: float,
    placement_z_abs_threshold: float,
    stable_speed_threshold: float,
    stable_duration_s: float,
) -> int:
    speed = compute_speed(t, body_pos)

    start_pos = body_pos[start_idx]
    displacement = np.linalg.norm(body_pos - start_pos, axis=1)

    moved_mask = displacement > move_threshold
    moved_mask[:start_idx] = False

    moved_indices = np.where(moved_mask)[0]

    if len(moved_indices) == 0:
        print("[WARN] Body did not move enough. Using final frame as end.")
        return len(t) - 1

    first_moved_idx = int(moved_indices[0])

    relative_pos = body_pos - target_pos
    xy_error = np.linalg.norm(relative_pos[:, :2], axis=1)
    z_error = np.abs(relative_pos[:, 2])

    placed_mask = (
        (xy_error <= placement_xy_threshold)
        & (z_error <= placement_z_abs_threshold)
    )
    placed_mask[:first_moved_idx] = False

    dt_values = np.diff(t)
    dt_values = dt_values[dt_values > 1e-8]
    median_dt = float(np.median(dt_values)) if len(dt_values) > 0 else 0.1

    stable_window = max(1, int(round(stable_duration_s / median_dt)))

    for i in range(first_moved_idx, len(t) - stable_window + 1):
        window_placed = placed_mask[i : i + stable_window]
        window_speed = speed[i : i + stable_window]

        if np.all(window_placed) and np.all(window_speed < stable_speed_threshold):
            return int(i)

    placed_indices = np.where(placed_mask)[0]
    if len(placed_indices) > 0:
        fallback_idx = int(placed_indices[0])
        print(
            "[WARN] Body reached placement target but did not become stable. "
            f"Using first placed frame: {fallback_idx}."
        )
        return fallback_idx

    print("[WARN] Body did not reach placement target. Using final frame as end.")
    return len(t) - 1


def find_end_index_from_contact_stability(
    t: np.ndarray,
    body_pos: np.ndarray,
    contact_mask: np.ndarray,
    start_idx: int,
    move_threshold: float,
    stable_speed_threshold: float,
    stable_duration_s: float,
) -> int:
    speed = compute_speed(t, body_pos)

    start_pos = body_pos[start_idx]
    displacement = np.linalg.norm(body_pos - start_pos, axis=1)

    moved_mask = displacement > move_threshold
    moved_mask[:start_idx] = False

    moved_indices = np.where(moved_mask)[0]

    if len(moved_indices) == 0:
        print("[WARN] Body did not move enough. Using final frame as end.")
        return len(t) - 1

    first_moved_idx = int(moved_indices[0])

    dt_values = np.diff(t)
    dt_values = dt_values[dt_values > 1e-8]
    median_dt = float(np.median(dt_values)) if len(dt_values) > 0 else 0.1

    stable_window = max(1, int(round(stable_duration_s / median_dt)))

    for i in range(first_moved_idx, len(t) - stable_window + 1):
        window_contact = contact_mask[i : i + stable_window]
        window_speed = speed[i : i + stable_window]

        if np.all(window_contact) and np.all(window_speed < stable_speed_threshold):
            return int(i + stable_window - 1)

    contact_indices = np.where(contact_mask)[0]
    contact_indices = contact_indices[contact_indices >= first_moved_idx]
    if len(contact_indices) > 0:
        fallback_idx = int(contact_indices[-1])
        print(
            "[WARN] Body contacted target but did not become stable. "
            f"Using last contact frame: {fallback_idx}."
        )
        return fallback_idx

    print("[WARN] Body did not contact target. Using final frame as end.")
    return len(t) - 1


def apply_time_margin(
    t: np.ndarray,
    start_idx: int,
    end_idx: int,
    pre_margin_s: float,
    post_margin_s: float,
) -> tuple[int, int]:
    start_time = max(0.0, t[start_idx] - pre_margin_s)
    end_time = min(t[-1], t[end_idx] + post_margin_s)

    trim_start_idx = int(np.searchsorted(t, start_time, side="left"))
    trim_end_idx = int(np.searchsorted(t, end_time, side="right") - 1)

    trim_start_idx = max(0, trim_start_idx)
    trim_end_idx = min(len(t) - 1, trim_end_idx)

    if trim_end_idx <= trim_start_idx:
        print("[WARN] Invalid trim range. Keeping full episode.")
        return 0, len(t) - 1

    return trim_start_idx, trim_end_idx


def trim_episode_arrays(
    episode: dict,
    start_idx: int,
    end_idx: int,
    reference_key: str,
) -> dict:
    n = len(episode[reference_key])
    trimmed = {}

    for key, value in episode.items():
        if isinstance(value, np.ndarray) and value.shape[:1] == (n,):
            trimmed[key] = value[start_idx : end_idx + 1]
        else:
            trimmed[key] = value

    return trimmed


def process_episode(
    episode_path: Path,
    output_path: Path,
    model: mujoco.MjModel,
    body_name: str,
    time_key: str,
    joint_names: list[str],
    joint_motion_threshold: float,
    body_move_threshold: float,
    stable_speed_threshold: float,
    stable_duration_s: float,
    pre_margin_s: float,
    post_margin_s: float,
    end_mode: str,
    target_body_name: str | None,
    placement_xy_threshold: float,
    placement_z_abs_threshold: float,
    body_candidates: list[str] | None = None,
    stats: StatsTracker | None = None,
) -> None:
    episode = load_episode(episode_path)

    if "qpos_sim" not in episode:
        raise KeyError(f"{episode_path.name} does not contain qpos_sim.")

    if time_key not in episode:
        raise KeyError(f"{episode_path.name} does not contain {time_key}.")

    t = get_time_array(episode, time_key)
    qpos_sim = episode["qpos_sim"]

    print(f"\nProcessing: {episode_path.name}")
    print(f"Samples: {len(t)}")
    print(f"Duration: {t[-1]:.3f} s")

    q_robot = get_robot_q_from_qpos_sim(
        model=model,
        qpos_sim=qpos_sim,
        joint_names=joint_names,
    )

    if body_name == "auto":
        if not body_candidates:
            raise ValueError("--body-candidates is required when --body-name auto")

        all_body_positions = extract_many_body_positions_from_qpos(
            model=model,
            qpos_sim=qpos_sim,
            body_names=body_candidates,
        )
        selected_body_name, body_pos = select_most_moved_body(all_body_positions)
    else:
        selected_body_name = body_name
        all_body_positions = {
            body_name: extract_body_positions_from_qpos(
                model=model,
                qpos_sim=qpos_sim,
                body_name=body_name,
            )
        }
        body_pos = all_body_positions[body_name]

    raw_start_idx = find_start_index_from_robot_motion(
        q_robot=q_robot,
        threshold=joint_motion_threshold,
    )

    if end_mode == "body-stability":
        raw_end_idx = find_end_index_from_body_stability(
            t=t,
            body_pos=body_pos,
            start_idx=raw_start_idx,
            move_threshold=body_move_threshold,
            stable_speed_threshold=stable_speed_threshold,
            stable_duration_s=stable_duration_s,
        )
    elif end_mode == "placement":
        if target_body_name is None:
            raise ValueError("--target-body-name is required when --end-mode placement")

        target_body_pos = extract_body_positions_from_qpos(
            model=model,
            qpos_sim=qpos_sim,
            body_name=target_body_name,
        )

        all_body_positions[target_body_name] = target_body_pos
        raw_end_idx = find_end_index_from_placement_stability(
            t=t,
            body_pos=body_pos,
            target_pos=target_body_pos,
            start_idx=raw_start_idx,
            move_threshold=body_move_threshold,
            placement_xy_threshold=placement_xy_threshold,
            placement_z_abs_threshold=placement_z_abs_threshold,
            stable_speed_threshold=stable_speed_threshold,
            stable_duration_s=stable_duration_s,
        )
    elif end_mode == "contact-stability":
        if target_body_name is None:
            raise ValueError(
                "--target-body-name is required when --end-mode contact-stability"
            )

        target_body_pos = extract_body_positions_from_qpos(
            model=model,
            qpos_sim=qpos_sim,
            body_name=target_body_name,
        )
        contact_mask = extract_body_contact_mask_from_qpos(
            model=model,
            qpos_sim=qpos_sim,
            body_name=selected_body_name,
            target_body_name=target_body_name,
        )

        all_body_positions[target_body_name] = target_body_pos
        episode[f"{selected_body_name}_contact_{target_body_name}"] = contact_mask

        raw_end_idx = find_end_index_from_contact_stability(
            t=t,
            body_pos=body_pos,
            contact_mask=contact_mask,
            start_idx=raw_start_idx,
            move_threshold=body_move_threshold,
            stable_speed_threshold=stable_speed_threshold,
            stable_duration_s=stable_duration_s,
        )
    else:
        raise ValueError(f"Unknown end mode: {end_mode}")

    start_idx, end_idx = apply_time_margin(
        t=t,
        start_idx=raw_start_idx,
        end_idx=raw_end_idx,
        pre_margin_s=pre_margin_s,
        post_margin_s=post_margin_s,
    )

    if stats is not None:
        stats.add(episode_path.name, t, start_idx, end_idx)

    for candidate_body_name, candidate_body_pos in all_body_positions.items():
        episode[f"{candidate_body_name}_pos"] = candidate_body_pos

    episode["trim_body_name"] = np.array(selected_body_name)
    episode["q_robot_from_qpos_sim"] = q_robot

    trimmed = trim_episode_arrays(
        episode=episode,
        start_idx=start_idx,
        end_idx=end_idx,
        reference_key=time_key,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_episode(output_path, trimmed)

    print(f"Raw start idx/time: {raw_start_idx} / {t[raw_start_idx]:.3f} s")
    print(f"Raw end idx/time:   {raw_end_idx} / {t[raw_end_idx]:.3f} s")
    print(f"Trim start:         {start_idx} / {t[start_idx]:.3f} s")
    print(f"Trim end:           {end_idx} / {t[end_idx]:.3f} s")
    print(f"Trimmed samples:    {end_idx - start_idx + 1}")
    print(f"Saved:              {output_path}")


def print_usage_examples():
    print("\n❌ Missing arguments.\n")

    print("You need to provide at least:")
    print("  --input   input .npz file or folder")
    print("  --output  output .npz file or folder")
    print("  --xml     MuJoCo XML scene\n")

    print("Example: single episode\n")
    print("  python utils/data_cleaning.py \\")
    print("    --input demonstrations/demo_pick_place/T_shape/p1_ep_0001_1777365954.npz \\")
    print("    --output data_clean/episode_000_trimmed.npz \\")
    print("    --xml mujoco_scenes/working_scenes/with_soft_gripper/sii_scene_table_T_shape.xml\n")

    print("Example: full folder\n")
    print("  python utils/data_cleaning.py \\")
    print("    --input demonstrations/demo_pick_place/T_shape \\")
    print("    --output data_clean/T_shape \\")
    print("    --xml mujoco_scenes/working_scenes/with_soft_gripper/sii_scene_table_T_shape.xml\n")

    print("Example: with tuning parameters\n")
    print("  python utils/data_cleaning.py \\")
    print("    --input demonstrations/demo_pick_place/T_shape \\")
    print("    --output data_clean/T_shape \\")
    print("    --xml mujoco_scenes/working_scenes/with_soft_gripper/sii_scene_table_T_shape.xml \\")
    print("    --joint-motion-threshold 0.03 \\")
    print("    --body-move-threshold 0.01 \\")
    print("    --stable-speed-threshold 0.002 \\")
    print("    --stable-duration 1.0 \\")
    print("    --pre-margin 0.5 \\")
    print("    --post-margin 0.5\n")

    print("Example: wire game, end after spoon contacts base and is stable\n")
    print("  python utils/data_cleaning.py \\")
    print("    --input WG_Test_demonstrations/WG/wire_base_and_spoon \\")
    print("    --output WG_Test_demonstrations/WG/wire_base_and_spoon_trimmed \\")
    print("    --xml mujoco_scenes/working_scenes/with_soft_gripper/sii_scene_table_wire_base_and_spoon.xml \\")
    print("    --body-name spoon1 \\")
    print("    --target-body-name object \\")
    print("    --end-mode contact-stability \\")
    print("    --stable-speed-threshold 0.005 \\")
    print("    --stable-duration 0.5 \\")
    print("    --pre-margin 0 \\")
    print("    --post-margin 0\n")

    print("For all options:")
    print("  python utils/data_cleaning.py -h\n")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--xml", required=True)

    parser.add_argument("--body-name", default="cup3")
    parser.add_argument("--body-candidates", nargs="+", default=None)
    parser.add_argument("--time-key", default="wall_t", choices=["wall_t", "sim_t"])

    parser.add_argument("--joint-motion-threshold", type=float, default=0.03)
    parser.add_argument("--body-move-threshold", type=float, default=0.01)
    parser.add_argument("--stable-speed-threshold", type=float, default=0.002)
    parser.add_argument("--stable-duration", type=float, default=1.0)

    parser.add_argument("--pre-margin", type=float, default=0.5)
    parser.add_argument("--post-margin", type=float, default=0.5)

    parser.add_argument(
        "--end-mode",
        choices=["body-stability", "placement", "contact-stability"],
        default="body-stability",
    )
    parser.add_argument("--target-body-name", default=None)
    parser.add_argument("--placement-xy-threshold", type=float, default=0.02)
    parser.add_argument("--placement-z-abs-threshold", type=float, default=0.04)

    parser.add_argument(
        "--joint-names",
        nargs="+",
        default=PANDA_JOINT_NAMES,
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    xml_path = Path(args.xml)

    model = mujoco.MjModel.from_xml_path(str(xml_path))

    stats = StatsTracker()

    if input_path.is_file():
        process_episode(
            episode_path=input_path,
            output_path=output_path,
            model=model,
            body_name=args.body_name,
            time_key=args.time_key,
            joint_names=args.joint_names,
            joint_motion_threshold=args.joint_motion_threshold,
            body_move_threshold=args.body_move_threshold,
            stable_speed_threshold=args.stable_speed_threshold,
            stable_duration_s=args.stable_duration,
            pre_margin_s=args.pre_margin,
            post_margin_s=args.post_margin,
            end_mode=args.end_mode,
            target_body_name=args.target_body_name,
            placement_xy_threshold=args.placement_xy_threshold,
            placement_z_abs_threshold=args.placement_z_abs_threshold,
            body_candidates=args.body_candidates,
            stats=stats,
        )

        odd_episodes = stats.summary()
        with open("odd_episodes.txt", "w") as f:
            for name in odd_episodes:
                f.write(name + "\n")

    elif input_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)

        episode_files = sorted(input_path.glob("*.npz"))

        if len(episode_files) == 0:
            raise FileNotFoundError(f"No .npz files found in {input_path}")

        for episode_file in episode_files:
            out_file = output_path / episode_file.name.replace(".npz", "_trimmed.npz")

            process_episode(
                episode_path=episode_file,
                output_path=out_file,
                model=model,
                body_name=args.body_name,
                time_key=args.time_key,
                joint_names=args.joint_names,
                joint_motion_threshold=args.joint_motion_threshold,
                body_move_threshold=args.body_move_threshold,
                stable_speed_threshold=args.stable_speed_threshold,
                stable_duration_s=args.stable_duration,
                pre_margin_s=args.pre_margin,
                post_margin_s=args.post_margin,
                end_mode=args.end_mode,
                target_body_name=args.target_body_name,
                placement_xy_threshold=args.placement_xy_threshold,
                placement_z_abs_threshold=args.placement_z_abs_threshold,
                body_candidates=args.body_candidates,
                stats=stats,
            )

        odd_episodes = stats.summary()
        with open("odd_episodes.txt", "w") as f:
            for name in odd_episodes:
                f.write(name + "\n")
    else:
        raise FileNotFoundError(f"Input path does not exist: {input_path}")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print_usage_examples()
        sys.exit(1)

    main()
