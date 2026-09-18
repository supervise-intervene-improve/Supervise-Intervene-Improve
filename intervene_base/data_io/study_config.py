"""Per-participant user-study configuration.

A study run must never start unlabelled: if we cannot say which participant, which
interface, which task and whether the scene is in- or out-of-distribution, the episodes
we record are unusable for analysis. So this module loads a per-participant JSON that
declares the full (ideally counterbalanced) block order, and resolves
`STUDY_PARTICIPANT` + `STUDY_BLOCK` into a frozen `StudyBlock`. Anything missing or
malformed raises `StudyConfigError` rather than silently defaulting.

File layout (default `intervene_base/study/participants/<PID>.json`):

    {
      "participant_id": "P07",
      "notes": "left-handed, wears glasses",
      "blocks": [
        {"block": 1, "interface": "rgb",        "task": "tshape", "condition": "id",  "seed": 10701},
        {"block": 2, "interface": "pointcloud", "task": "tshape", "condition": "ood", "seed": 10702}
      ]
    }

`seed` is the block's base seed. Per-episode seeds are derived from it (see
`StudyBlock.episode_seed`) so every scene is reproducible from
`(participant, block, episode_number)` alone — which is what makes `scenario_id` a real
identifier rather than a label.

Leaving `STUDY_PARTICIPANT` unset disables all study behaviour; the app records exactly
as it did before.
"""

import json
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional

INTERFACES = ("rgb", "pointcloud", "both", "none")
CONDITIONS = ("id", "ood")
# Task names line up with INTERVENE_TASK_MODE / the randomization presets so a block can
# drive the launcher without a second mapping table.
TASKS = ("tshape", "cups", "wiregame", "single")
# How the human actually drives the arm during an intervention. Selects the launcher's
# FACTR_ACTIVE / MC_ACTIVE / MC_SIM triple, exactly as `interface` selects RGB/WITH_PC —
# so the recorded condition and the device the participant holds cannot drift apart.
# Defaults to telekinesis, which is what every pre-existing config was implicitly using.
CONTROLS = ("telekinesis", "motion_controller", "sim_mc", "factr")
DEFAULT_CONTROL = "telekinesis"

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[1] / "study" / "participants"


class StudyConfigError(RuntimeError):
    """Raised when study metadata is missing or invalid. Never swallowed."""


@dataclass(frozen=True)
class StudyBlock:
    participant_id: str
    block: int
    interface: str
    task: str
    condition: str
    seed: int
    control: str = DEFAULT_CONTROL
    notes: str = ""
    extra: Dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        # `control` is deliberately NOT in the label: this string names the on-disk
        # session directory, and adding a field would make new sessions incomparable to
        # already-collected data. It is recorded in session.json and events.jsonl instead.
        return f"block{self.block}_{self.interface}_{self.task}_{self.condition}"

    def episode_seed(self, episode_number: int) -> int:
        """Deterministic per-episode seed.

        The app previously seeded one RNG once per process, so an episode's scene could
        not be reproduced without replaying every episode before it. Deriving each
        episode's seed from (block seed, episode number) makes any single episode
        reconstructable on its own.
        """
        import numpy as np

        seq = np.random.SeedSequence([int(self.seed), int(episode_number)])
        return int(seq.generate_state(1, dtype="uint32")[0])

    def scenario_id(self, episode_number: int) -> str:
        return f"{self.task}-{self.condition}-{self.episode_seed(episode_number):08x}"

    def to_dict(self) -> Dict:
        return asdict(self)


def _validate_choice(value, allowed, field_name, ctx):
    if value not in allowed:
        raise StudyConfigError(
            f"{ctx}: {field_name}={value!r} is not one of {allowed}"
        )


def load_participant_file(path) -> Dict:
    path = Path(path)
    if not path.is_file():
        raise StudyConfigError(f"participant config not found: {path}")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        raise StudyConfigError(f"participant config {path} is not valid JSON: {exc}") from exc


def parse_blocks(raw: Dict, path=None) -> List[StudyBlock]:
    ctx = str(path or "<config>")
    participant_id = str(raw.get("participant_id") or "").strip()
    if not participant_id:
        raise StudyConfigError(f"{ctx}: missing 'participant_id'")
    blocks_raw = raw.get("blocks")
    if not isinstance(blocks_raw, list) or not blocks_raw:
        raise StudyConfigError(f"{ctx}: 'blocks' must be a non-empty list")

    notes = str(raw.get("notes") or "")
    blocks: List[StudyBlock] = []
    seen = set()
    for i, entry in enumerate(blocks_raw):
        if not isinstance(entry, dict):
            raise StudyConfigError(f"{ctx}: blocks[{i}] must be an object")
        where = f"{ctx}: blocks[{i}]"
        try:
            block_no = int(entry.get("block", i + 1))
        except (TypeError, ValueError):
            raise StudyConfigError(f"{where}: 'block' must be an integer") from None
        if block_no in seen:
            raise StudyConfigError(f"{ctx}: duplicate block number {block_no}")
        seen.add(block_no)

        interface = str(entry.get("interface", "")).strip().lower()
        task = str(entry.get("task", "")).strip().lower()
        condition = str(entry.get("condition", "")).strip().lower()
        # Defaulted BEFORE validation so configs written before `control` existed still
        # load, and resolve to the mode they were actually run with.
        control = str(entry.get("control", DEFAULT_CONTROL)).strip().lower()
        _validate_choice(interface, INTERFACES, "interface", where)
        _validate_choice(task, TASKS, "task", where)
        _validate_choice(condition, CONDITIONS, "condition", where)
        _validate_choice(control, CONTROLS, "control", where)

        if "seed" in entry and entry["seed"] is not None:
            try:
                seed = int(entry["seed"])
            except (TypeError, ValueError):
                raise StudyConfigError(f"{where}: 'seed' must be an integer") from None
        else:
            # Stable fallback so a config without explicit seeds is still reproducible.
            seed = abs(hash((participant_id, block_no))) % (2**31)

        known = {"block", "interface", "task", "condition", "control", "seed", "notes"}
        extra = {k: v for k, v in entry.items() if k not in known}

        blocks.append(StudyBlock(
            participant_id=participant_id,
            block=block_no,
            interface=interface,
            task=task,
            condition=condition,
            seed=seed,
            control=control,
            notes=str(entry.get("notes") or notes),
            extra=extra,
        ))
    blocks.sort(key=lambda b: b.block)
    return blocks


def resolve_config_dir(explicit: Optional[str] = None) -> Path:
    if explicit:
        return Path(explicit)
    env = os.environ.get("STUDY_CONFIG_DIR")
    if env:
        return Path(env)
    return DEFAULT_CONFIG_DIR


def load_block(
    participant_id: Optional[str] = None,
    block_number: Optional[int] = None,
    config_dir: Optional[str] = None,
) -> Optional[StudyBlock]:
    """Resolve the active study block, or None when study mode is off.

    Study mode is off exactly when `STUDY_PARTICIPANT` is unset/empty and no
    participant is passed explicitly — in that case every caller keeps its previous
    non-study behaviour.
    """
    participant_id = participant_id or os.environ.get("STUDY_PARTICIPANT") or ""
    participant_id = participant_id.strip()
    if not participant_id:
        return None

    if block_number is None:
        raw_block = os.environ.get("STUDY_BLOCK", "1").strip()
        try:
            block_number = int(raw_block)
        except ValueError:
            raise StudyConfigError(
                f"STUDY_BLOCK={raw_block!r} is not an integer"
            ) from None

    cfg_dir = resolve_config_dir(config_dir)
    path = cfg_dir / f"{participant_id}.json"
    raw = load_participant_file(path)
    blocks = parse_blocks(raw, path=path)

    for block in blocks:
        if block.block == block_number:
            return block
    available = ", ".join(str(b.block) for b in blocks)
    raise StudyConfigError(
        f"{path}: block {block_number} not found (available: {available})"
    )


def study_block_from_experiment(exp_block, cell_id: int = 0) -> StudyBlock:
    """Adapt an `experiment_block.ExperimentBlock` into the `StudyBlock` app.py expects.

    Everything downstream of `StudyBlock` -- per-episode seeds, `scenario_id`, the
    RGB/point-cloud launch mode, the KT/MC/FACTR control mode -- already works. Rather
    than teaching all of it a second vocabulary, the HRI block description is projected
    onto this one, once, here.

    The block seed is folded with `cell_id` so the nine concurrent cells do not all
    replay the same scene sequence, while any single episode stays reproducible from
    (participant, study, condition, task, cell, episode number).

    `condition` stays `"id"`: it is a block-level label, but which cells run
    out-of-distribution scenes is decided per episode by the OOD ledger. The per-episode
    `episode_ood` flag in the NPZ is the authoritative ID/OOD label -- see
    `app.py::_study_metadata`.
    """
    import numpy as np

    cell_seed = int(
        np.random.SeedSequence([int(exp_block.seed), int(cell_id)])
        .generate_state(1, dtype="uint32")[0]
    )
    return StudyBlock(
        participant_id=exp_block.participant_id,
        block=1,
        interface=exp_block.study_config_interface,
        task=exp_block.task_mode,
        condition="id",
        seed=cell_seed,
        control=exp_block.study_config_control,
        notes=exp_block.notes,
        extra={
            "study": exp_block.study,
            "condition_id": exp_block.condition_id,
            "supervision_interface": exp_block.supervision_interface,
            "controller_interface": exp_block.controller_interface,
            "block_id": exp_block.block_id,
            "cell_id": int(cell_id),
            "block_seed": int(exp_block.seed),
        },
    )


def describe(block: Optional[StudyBlock]) -> str:
    if block is None:
        return "[Study] disabled (STUDY_PARTICIPANT not set)"
    return (
        f"[Study] participant={block.participant_id} block={block.block} "
        f"interface={block.interface} task={block.task} condition={block.condition} "
        f"control={block.control} seed={block.seed}"
    )


if __name__ == "__main__":  # tiny CLI: validate a config without launching anything
    import argparse

    ap = argparse.ArgumentParser(description="Validate a participant study config.")
    ap.add_argument("participant", help="Participant ID (file stem).")
    ap.add_argument("--block", type=int, default=None, help="Block number to resolve.")
    ap.add_argument("--config_dir", default=None)
    ap.add_argument("--episodes", type=int, default=3,
                    help="Show derived scenario IDs for the first N episodes.")
    args = ap.parse_args()

    cfg_dir = resolve_config_dir(args.config_dir)
    raw = load_participant_file(cfg_dir / f"{args.participant}.json")
    blocks = parse_blocks(raw, path=cfg_dir / f"{args.participant}.json")
    print(f"participant={raw.get('participant_id')} blocks={len(blocks)}")
    for block in blocks:
        marker = " <-- selected" if args.block == block.block else ""
        print(f"  {block.label} seed={block.seed}{marker}")
        for ep in range(1, args.episodes + 1):
            print(f"      episode {ep}: scenario_id={block.scenario_id(ep)}")
