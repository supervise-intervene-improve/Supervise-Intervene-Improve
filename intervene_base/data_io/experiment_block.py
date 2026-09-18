"""Block-level identity and the shared, multi-process event log for the HRI study.

A "block" is one participant running one condition on one task for a few minutes.
Everything below exists because that block is spread across **many OS processes**:

    run_main_policy.sh
      +- app.py            x9   (one policy process per cell; owns the sim + episodes)
      +- runtime_impl.py   x9   (one VR runtime per cell; receives the Quest selection)
      +- policy_grid_viewer.py  (one desktop grid; receives the mouse selection)

`data_io/study_logger.StudySession` already records episodes and intervention phases,
but it derives its own output directory from its own start time -- so nine policy
processes produce nine unrelated directories, and the two processes that actually know
which cell the human selected (the grid and the VR runtime) write nothing at all.

This module supplies the two things that were missing:

  * `ExperimentBlock` -- one identity, resolved from the environment, that every process
    agrees on. `supervision_interface` and `controller_interface` are DERIVED from
    (study, condition_id) via `CONDITION_MATRIX`, so an impossible combination such as
    "Control/C2 with KT" cannot be entered at all.
  * `BlockEventLog` -- an append-only `events.jsonl` that all of those processes write to
    concurrently. Each event is one `os.write()` on an `O_APPEND` fd, which the kernel
    serialises against the file's inode lock, so lines never interleave or tear.

Timebase. `mono_t` is `time.monotonic()`, which on Linux is `CLOCK_MONOTONIC` -- a
system-wide clock with a single origin (boot), so it is directly comparable **across
processes on one machine**, which is exactly this study's topology. `elapsed_time` is
`mono_t` minus the block's monotonic origin (recorded in `block_metadata.json`), giving
every process one shared, NTP-immune, high-resolution timeline. `wall_t` is carried too,
but only for joining against logs that have no monotonic column.

Nothing here computes a study statistic. `events.jsonl` is the raw record; block_summary
carries counts for integrity checking only.
"""

import json
import os
import platform
import socket
import subprocess
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, Optional

SCHEMA_VERSION = 1

# --------------------------------------------------------------------------- vocabulary

STUDIES = ("Supervision", "Control")
# The two studies were called 1A / 1B while the protocol was being written. Accepted as
# input so an older command line or note still resolves, but never stored: `study` is a
# directory name and two spellings of it would split one participant's data in two.
STUDY_ALIASES = {
    "1A": "Supervision", "SUPERVISION": "Supervision", "S": "Supervision",
    "1B": "Control", "CONTROL": "Control", "C": "Control",
}

SUPERVISION_DESKTOP_RGB = "desktop_rgb"
SUPERVISION_VR_RGB = "vr_rgb"
SUPERVISION_VR_POINTCLOUD = "vr_pointcloud"
SUPERVISION_INTERFACES = (
    SUPERVISION_DESKTOP_RGB,
    SUPERVISION_VR_RGB,
    SUPERVISION_VR_POINTCLOUD,
)

CONTROLLER_KT = "KT"
CONTROLLER_MC = "MC"
CONTROLLER_FACTR = "FACTR"
CONTROLLER_INTERFACES = (CONTROLLER_KT, CONTROLLER_MC, CONTROLLER_FACTR)

# The whole point of this table: the experimenter types a condition ID and nothing else.
# Deriving both interfaces from it makes "VR-RGB + MC" -- a cell that does not exist in
# either study design -- unrepresentable rather than merely discouraged.
CONDITION_MATRIX: Dict[tuple, tuple] = {
    ("Supervision", "S1"): (SUPERVISION_DESKTOP_RGB, CONTROLLER_KT),
    ("Supervision", "S2"): (SUPERVISION_VR_RGB, CONTROLLER_KT),
    ("Supervision", "S3"): (SUPERVISION_VR_POINTCLOUD, CONTROLLER_KT),
    ("Control", "C1"): (SUPERVISION_VR_POINTCLOUD, CONTROLLER_KT),
    ("Control", "C2"): (SUPERVISION_VR_POINTCLOUD, CONTROLLER_MC),
    ("Control", "C3"): (SUPERVISION_VR_POINTCLOUD, CONTROLLER_FACTR),
}

# The launcher env each condition corresponds to -- the SAME variables the operator
# types on the command line. This table exists so the two cannot disagree: the launcher
# fills in whatever was not given, then VERIFIES the final values against it and refuses
# to record a block whose flags contradict its own label.
#
# MC is `MC_ACTIVE=1 MC_SIM=1` (sim-native motion controller: local IK against the
# simulated arm, no real robot and no mq3_mc.py sidecar) because that is what the study
# actually runs. Real MC would be MC_SIM=0 and is not a study condition.
SUPERVISION_LAUNCH_FLAGS = {
    SUPERVISION_DESKTOP_RGB:   {"START_VR": "0", "RGB": "0", "WITH_PC": "0"},
    SUPERVISION_VR_RGB:        {"START_VR": "1", "RGB": "1", "WITH_PC": "0"},
    SUPERVISION_VR_POINTCLOUD: {"START_VR": "1", "RGB": "0", "WITH_PC": "1"},
}
CONTROLLER_LAUNCH_FLAGS = {
    CONTROLLER_KT:    {"MC_ACTIVE": "0", "MC_SIM": "0", "FACTR_ACTIVE": "0"},
    CONTROLLER_MC:    {"MC_ACTIVE": "1", "MC_SIM": "1", "FACTR_ACTIVE": "0"},
    CONTROLLER_FACTR: {"MC_ACTIVE": "0", "MC_SIM": "0", "FACTR_ACTIVE": "1"},
}

SUPERVISION_SLUG = {
    SUPERVISION_DESKTOP_RGB: "DesktopRGB",
    SUPERVISION_VR_RGB: "VRRGB",
    SUPERVISION_VR_POINTCLOUD: "VRPointCloud",
}

# Display name -> the launcher/evaluator's internal task key (INTERVENE_TASK_MODE, LAB).
TASKS = {"TShape": "tshape", "Cups": "cups"}
TASK_ALIASES = {
    "tshape": "TShape", "t_shape": "TShape", "t-shape": "TShape", "TShape": "TShape",
    "cups": "Cups", "Cups": "Cups",
}

# The controller vocabulary `data_io/study_config.CONTROLS` already uses. Kept as a
# mapping rather than a rename so nothing downstream of `study_config` has to change.
CONTROL_TO_STUDY_CONFIG = {
    CONTROLLER_KT: "telekinesis",
    CONTROLLER_MC: "motion_controller",
    CONTROLLER_FACTR: "factr",
}
# ... and the `interface` vocabulary it uses for the VR launch mode.
SUPERVISION_TO_STUDY_CONFIG_INTERFACE = {
    SUPERVISION_DESKTOP_RGB: "rgb",
    SUPERVISION_VR_RGB: "rgb",
    SUPERVISION_VR_POINTCLOUD: "pointcloud",
}

DEFAULT_ROOT = "experiment_data"

# Env keys. Deliberately a separate `EXP_` namespace from the pre-existing `STUDY_`
# variables: a block can be started with or without the older participant-config flow,
# and mixing the two namespaces is how they would silently disagree.
ENV_PARTICIPANT = "EXP_PARTICIPANT"
ENV_STUDY = "EXP_STUDY"
ENV_CONDITION = "EXP_CONDITION"
ENV_TASK = "EXP_TASK"
ENV_ROOT = "EXP_ROOT"
ENV_BLOCK_ID = "EXP_BLOCK_ID"
ENV_BLOCK_DIR = "EXP_BLOCK_DIR"
ENV_SEED = "EXP_SEED"
ENV_CELLS = "EXP_CELLS"
ENV_NOTES = "EXP_NOTES"


class ExperimentConfigError(RuntimeError):
    """Raised for an invalid or incomplete block description. Never swallowed.

    Recording a participant under the wrong condition label is worse than not
    recording at all, so every path that could produce an unlabelled or mislabelled
    block raises instead of defaulting.
    """


# --------------------------------------------------------------------------- helpers

def _now():
    return time.time(), time.monotonic()


def _git_sha(path) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def _git_dirty(path) -> Optional[bool]:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode == 0:
            return bool(out.stdout.strip())
    except Exception:
        pass
    return None


def normalize_task(value: str) -> str:
    """Accept 'tshape', 't_shape', 'TShape', ... and return the canonical display name."""
    key = str(value or "").strip()
    if not key:
        raise ExperimentConfigError(f"task is required (one of {tuple(TASKS)})")
    if key in TASK_ALIASES:
        return TASK_ALIASES[key]
    lowered = key.lower()
    if lowered in TASK_ALIASES:
        return TASK_ALIASES[lowered]
    raise ExperimentConfigError(f"task={value!r} is not one of {tuple(TASKS)}")


def normalize_study(value: str) -> str:
    """Accept 'Supervision', 'supervision', or the legacy '1A'; return the canonical name."""
    raw = str(value or "").strip()
    if not raw:
        raise ExperimentConfigError(f"study is required (one of {STUDIES})")
    if raw in STUDIES:
        return raw
    resolved = STUDY_ALIASES.get(raw.upper())
    if resolved is None:
        raise ExperimentConfigError(
            f"study={value!r} is not one of {STUDIES} (legacy 1A/1B also accepted)")
    return resolved


def _truthy(value, default: str = "0") -> bool:
    raw = value if value not in (None, "") else default
    return str(raw).strip().lower() in {"1", "true", "yes", "y", "on"}


def interfaces_from_flags(flags: Dict[str, str]) -> tuple:
    """(supervision, controller) implied by the launcher env the operator typed.

    The inverse of `launch_flags`. Defaults match the launcher's own: VR on, point
    cloud rather than RGB, KT rather than MC/FACTR -- i.e. "no flags at all" is the
    plain VR-pointcloud + KT launch, which is exactly Supervision/S3 and Control/C1.
    """
    if _truthy(flags.get("FACTR_ACTIVE")):
        controller = CONTROLLER_FACTR
    elif _truthy(flags.get("MC_ACTIVE")):
        controller = CONTROLLER_MC
    else:
        controller = CONTROLLER_KT

    if not _truthy(flags.get("START_VR"), default="1"):
        supervision = SUPERVISION_DESKTOP_RGB
    elif _truthy(flags.get("RGB")):
        supervision = SUPERVISION_VR_RGB
    else:
        supervision = SUPERVISION_VR_POINTCLOUD
    return supervision, controller


def infer_condition(study: str, flags: Dict[str, str]) -> str:
    """Which condition of `study` the given launcher flags mean.

    Unambiguous by construction: within one study no two conditions share a
    (supervision, controller) pair -- Supervision varies only the supervision
    interface, Control varies only the controller. `test_condition_is_inferable`
    asserts that, so a future condition that breaks it fails a test rather than
    silently mislabelling a participant.

    Raises with the study that DOES have the requested combination, because the
    realistic mistake is running the right command under the wrong study.
    """
    study = normalize_study(study)
    supervision, controller = interfaces_from_flags(flags)
    matches = [cond for (s, cond), pair in CONDITION_MATRIX.items()
               if s == study and pair == (supervision, controller)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ExperimentConfigError(
            f"{study}: {supervision} + {controller} is ambiguous between "
            f"{sorted(matches)}. Pass EXP_CONDITION explicitly."
        )
    elsewhere = sorted(f"{s}/{c}" for (s, c), pair in CONDITION_MATRIX.items()
                       if pair == (supervision, controller))
    hint = (f" That combination is {', '.join(elsewhere)}." if elsewhere
            else " No study condition uses that combination.")
    allowed = sorted(c for (s, c) in CONDITION_MATRIX if s == study)
    raise ExperimentConfigError(
        f"the launch flags mean {supervision} + {controller}, which is not a "
        f"condition of study {study} (it has {allowed}).{hint}"
    )


def resolve_condition(study: str, condition_id: str) -> tuple:
    study = normalize_study(study)
    condition_id = str(condition_id or "").strip().upper()
    key = (study, condition_id)
    if key not in CONDITION_MATRIX:
        allowed = sorted(c for (s, c) in CONDITION_MATRIX if s == study)
        raise ExperimentConfigError(
            f"condition={condition_id!r} is not valid for study {study} (allowed: {allowed})"
        )
    return CONDITION_MATRIX[key]


# --------------------------------------------------------------------------- the block

@dataclass(frozen=True)
class ExperimentBlock:
    """One participant x condition x task recording block.

    Frozen because every process resolves it independently from the same environment
    and they must agree; a mutable field would be a way for one process to drift.
    """

    participant_id: str
    study: str
    condition_id: str
    supervision_interface: str
    controller_interface: str
    task: str                 # display name, e.g. "TShape"
    task_mode: str            # internal key, e.g. "tshape"
    block_id: str
    root: str
    seed: int = 0
    notes: str = ""
    n_cells: int = 0

    # ---- derived paths ----

    @property
    def condition_label(self) -> str:
        """e.g. 'Supervision_VRRGB_KT' -- the directory grouping this condition's blocks.

        The S1/S2/S3/C1/C2/C3 code is deliberately NOT in the path: it is a shorthand
        for exactly this (study, supervision, controller) triple, so including both
        would write the same fact twice. `study` IS needed -- VRPointCloud_KT exists in
        both studies (Supervision/S3 and Control/C1) and is the only condition that
        does. `condition_id` remains a field in the metadata, the events and the NPZ
        scalars, where it appears once and is what the protocol document calls it.
        """
        slug = SUPERVISION_SLUG[self.supervision_interface]
        return f"{self.study}_{slug}_{self.controller_interface}"

    @property
    def block_dir(self) -> Path:
        return (Path(self.root) / self.participant_id / self.condition_label
                / self.task / self.block_id)

    @property
    def episodes_dir(self) -> Path:
        return self.block_dir / "episodes"

    @property
    def events_path(self) -> Path:
        return self.block_dir / "events.jsonl"

    @property
    def metadata_path(self) -> Path:
        return self.block_dir / "block_metadata.json"

    @property
    def summary_path(self) -> Path:
        return self.block_dir / "block_summary.json"

    # ---- launcher settings derived from the condition -------------------------------
    # These exist so the recorded condition and what the participant actually sees
    # cannot drift apart: the same field drives both the label and the launch mode.

    @property
    def vr_enabled(self) -> bool:
        return self.supervision_interface != SUPERVISION_DESKTOP_RGB

    @property
    def vr_rgb(self) -> bool:
        return self.supervision_interface == SUPERVISION_VR_RGB

    @property
    def vr_pointcloud(self) -> bool:
        return self.supervision_interface == SUPERVISION_VR_POINTCLOUD

    @property
    def launch_flags(self) -> Dict[str, str]:
        """The exact launcher env this condition means.

        Used twice by `run_main_policy.sh`: to fill in whatever the operator did not
        type, and then to VERIFY the resolved values. Verification is the important
        half -- the operator runs each condition as its own command line
        (`RGB=1 WINDOWS=9 LAB=cups ...`), so a mistyped condition label would otherwise
        record a block under a name that does not match what the participant saw.
        """
        flags = dict(SUPERVISION_LAUNCH_FLAGS[self.supervision_interface])
        flags.update(CONTROLLER_LAUNCH_FLAGS[self.controller_interface])
        return flags

    @property
    def study_config_interface(self) -> str:
        return SUPERVISION_TO_STUDY_CONFIG_INTERFACE[self.supervision_interface]

    @property
    def study_config_control(self) -> str:
        return CONTROL_TO_STUDY_CONFIG[self.controller_interface]

    def episode_seed(self, cell_id: int, episode_number: int) -> int:
        """Deterministic per (cell, episode) seed.

        Includes `cell_id` so nine cells running the same block do not all replay the
        same scene sequence -- and so any single episode stays reproducible from
        (participant, block, cell, episode number) alone.
        """
        import numpy as np

        seq = np.random.SeedSequence([int(self.seed), int(cell_id), int(episode_number)])
        return int(seq.generate_state(1, dtype="uint32")[0])

    def scenario_id(self, cell_id: int, episode_number: int) -> str:
        return (f"{self.task_mode}-c{int(cell_id):02d}-"
                f"{self.episode_seed(cell_id, episode_number):08x}")

    def labels(self) -> Dict[str, Any]:
        """The identity fields stamped onto every event, episode and trajectory."""
        return {
            "participant_id": self.participant_id,
            "study": self.study,
            "condition": self.condition_id,
            "supervision_interface": self.supervision_interface,
            "controller_interface": self.controller_interface,
            "task": self.task,
            "block_id": self.block_id,
        }

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out.update({
            "condition_label": self.condition_label,
            "block_dir": str(self.block_dir),
            "vr_enabled": self.vr_enabled,
            "vr_rgb": self.vr_rgb,
            "vr_pointcloud": self.vr_pointcloud,
        })
        return out


def make_block_id(when: Optional[float] = None) -> str:
    """The block id is just its start time.

    Participant, study, condition and task are already the three directory levels
    above it, and every event and NPZ carries them as fields -- repeating them in the
    leaf name says nothing new. `(participant_id, block_id)` is the globally unique
    join key, and both halves are on every record.
    """
    return time.strftime("%Y%m%d_%H%M%S", time.localtime(when or time.time()))


def build_block(
    participant_id: str,
    study: str,
    condition_id: str,
    task: str,
    *,
    root: Optional[str] = None,
    block_id: Optional[str] = None,
    seed: Optional[int] = None,
    notes: str = "",
    n_cells: int = 0,
) -> ExperimentBlock:
    """Validate a (participant, study, condition, task) combination into a block.

    This is the only constructor: `supervision_interface` / `controller_interface` are
    never passed in, always derived, so they cannot contradict the condition ID.
    """
    participant_id = str(participant_id or "").strip()
    if not participant_id:
        raise ExperimentConfigError("participant_id is required (e.g. P003)")
    if any(ch in participant_id for ch in "/\\ \t"):
        raise ExperimentConfigError(
            f"participant_id={participant_id!r} must not contain path separators or spaces"
        )

    study = normalize_study(study)
    condition_id = str(condition_id).strip().upper()
    supervision, controller = resolve_condition(study, condition_id)
    task_display = normalize_task(task)

    if seed is None:
        # Stable fallback: the same block description always yields the same base seed,
        # so a block re-run after a crash reproduces the same scene sequence.
        seq = f"{participant_id}|{study}|{condition_id}|{task_display}"
        seed = int.from_bytes(seq.encode("utf-8"), "little") % (2 ** 31)

    return ExperimentBlock(
        participant_id=participant_id,
        study=study,
        condition_id=condition_id,
        supervision_interface=supervision,
        controller_interface=controller,
        task=task_display,
        task_mode=TASKS[task_display],
        block_id=block_id or make_block_id(),
        root=str(root or os.environ.get(ENV_ROOT) or DEFAULT_ROOT),
        seed=int(seed),
        notes=str(notes or ""),
        n_cells=int(n_cells or 0),
    )


def resolve_block_from_env(env: Optional[Dict[str, str]] = None) -> Optional[ExperimentBlock]:
    """Resolve the active block, or None when block recording is off.

    Off exactly when `EXP_PARTICIPANT` is unset/empty -- in which case every caller keeps
    its previous behaviour. Anything else missing is an error, never a default.
    """
    env = os.environ if env is None else env
    participant = (env.get(ENV_PARTICIPANT) or "").strip()
    if not participant:
        return None

    block_id = (env.get(ENV_BLOCK_ID) or "").strip() or None
    seed_raw = (env.get(ENV_SEED) or "").strip()
    try:
        seed = int(seed_raw) if seed_raw else None
    except ValueError:
        raise ExperimentConfigError(f"{ENV_SEED}={seed_raw!r} is not an integer") from None
    cells_raw = (env.get(ENV_CELLS) or "").strip()
    try:
        n_cells = int(cells_raw) if cells_raw else 0
    except ValueError:
        n_cells = 0

    block = build_block(
        participant_id=participant,
        study=env.get(ENV_STUDY, ""),
        condition_id=env.get(ENV_CONDITION, ""),
        task=env.get(ENV_TASK, ""),
        root=env.get(ENV_ROOT) or None,
        block_id=block_id,
        seed=seed,
        notes=env.get(ENV_NOTES, ""),
        n_cells=n_cells,
    )

    # EXP_BLOCK_DIR is what the starter exports. If it is present it wins: a worker must
    # never invent a second directory just because it recomputed the timestamp.
    explicit_dir = (env.get(ENV_BLOCK_DIR) or "").strip()
    if explicit_dir and str(block.block_dir) != explicit_dir:
        raise ExperimentConfigError(
            f"{ENV_BLOCK_DIR}={explicit_dir!r} does not match the directory derived from "
            f"{ENV_PARTICIPANT}/{ENV_STUDY}/{ENV_CONDITION}/{ENV_TASK}/{ENV_BLOCK_ID} "
            f"({block.block_dir}). Refusing to write into an ambiguous location."
        )
    return block


def describe(block: Optional[ExperimentBlock]) -> str:
    if block is None:
        return f"[Block] recording disabled ({ENV_PARTICIPANT} not set)"
    return (
        f"[Block] participant={block.participant_id} study={block.study} "
        f"condition={block.condition_id} "
        f"({block.supervision_interface} + {block.controller_interface}) "
        f"task={block.task} block_id={block.block_id}"
    )


# --------------------------------------------------------------- multi-process events

class BlockEventLog:
    """Append-only `events.jsonl`, safe for concurrent writers.

    One `os.write()` per event on an `O_APPEND` file descriptor. Linux takes the inode
    lock for the duration of an append write, so concurrent writes from the nine policy
    processes, the grid and the VR runtimes are serialised whole -- lines never
    interleave. There is no buffering to lose: a crash costs at most an event that was
    never handed to the kernel.
    """

    def __init__(self, path, *, block: ExperimentBlock, cell_id: Optional[int] = None,
                 source: str = "", mono_origin: Optional[float] = None,
                 wall_origin: Optional[float] = None):
        self.path = Path(path)
        self.block = block
        self.cell_id = cell_id
        self.source = source or "unknown"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fd = os.open(str(self.path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        self._labels = block.labels()
        self._seq = 0
        self._pid = os.getpid()
        self.mono_origin = mono_origin
        self.wall_origin = wall_origin
        self._closed = False

    def set_origin(self, mono_origin: Optional[float], wall_origin: Optional[float]):
        self.mono_origin = mono_origin
        self.wall_origin = wall_origin

    def write(self, event: str, *, cell_id: Optional[int] = None,
              episode_id: Optional[str] = None, **fields) -> Dict[str, Any]:
        """Append one event. Never raises -- logging must not be able to stop a study."""
        wall_t, mono_t = _now()
        self._seq += 1
        cell = self.cell_id if cell_id is None else cell_id
        if self.mono_origin is not None:
            elapsed = mono_t - float(self.mono_origin)
        elif self.wall_origin is not None:
            elapsed = wall_t - float(self.wall_origin)
        else:
            elapsed = None
        record: Dict[str, Any] = {
            "schema": SCHEMA_VERSION,
            "event": str(event).upper(),
            "timestamp": wall_t,
            "wall_t": wall_t,
            "mono_t": mono_t,
            "elapsed_time": elapsed,
            "source": self.source,
            "pid": self._pid,
            "seq": self._seq,
        }
        record.update(self._labels)
        record["cell_id"] = cell
        record["episode_id"] = episode_id
        record.update(fields)
        try:
            line = json.dumps(record, default=str, separators=(",", ":")) + "\n"
            os.write(self._fd, line.encode("utf-8"))
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[Block][WARN] event write failed ({event}): {exc}")
        return record

    def close(self):
        if self._closed:
            return
        self._closed = True
        try:
            os.close(self._fd)
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


# ------------------------------------------------------------------ metadata / summary

def _atomic_write_json(path: Path, payload: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def collect_environment(repo_root=None, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    repo_root = Path(repo_root or Path(__file__).resolve().parents[2])
    env = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "git_sha_umbrella": _git_sha(repo_root),
        "git_dirty_umbrella": _git_dirty(repo_root),
        "git_sha_intervene_base": _git_sha(Path(__file__).resolve().parents[1]),
        "cwd": os.getcwd(),
    }
    try:
        import mujoco  # noqa: F401 -- version only; absent in some tooling envs
        env["mujoco_version"] = mujoco.__version__
    except Exception:
        env["mujoco_version"] = None
    if extra:
        env.update(extra)
    return env


def start_block(block: ExperimentBlock, *, env_info: Optional[Dict[str, Any]] = None,
                config: Optional[Dict[str, Any]] = None,
                overwrite: bool = False) -> Dict[str, Any]:
    """Create the block directory, write `block_metadata.json`, log BLOCK_START.

    Metadata is written HERE, not at block end, so an interrupted block is still fully
    identified. `status` starts as "incomplete" and only `stop_block` sets it to
    "complete" -- an interrupted block is therefore marked, never deleted.
    """
    block_dir = block.block_dir
    if block_dir.exists() and any(block_dir.iterdir()) and not overwrite:
        raise ExperimentConfigError(
            f"block directory already exists and is not empty: {block_dir}\n"
            f"Refusing to overwrite collected data. Start a new block (a fresh block_id) "
            f"or pass overwrite=True deliberately."
        )
    for sub in ("success", "failure", "incomplete", "pending"):
        (block.episodes_dir / sub).mkdir(parents=True, exist_ok=True)
    (block_dir / "cells").mkdir(parents=True, exist_ok=True)

    wall_t, mono_t = _now()
    metadata = {
        "schema": SCHEMA_VERSION,
        "status": "incomplete",
        "block": block.to_dict(),
        "block_start_timestamp": wall_t,
        "block_start_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(wall_t)),
        "block_start_mono": mono_t,
        "block_end_timestamp": None,
        "block_end_mono": None,
        "duration_s": None,
        "environment": collect_environment(extra=(env_info or {})),
        "config": dict(config or {}),
    }
    _atomic_write_json(block.metadata_path, metadata)

    with BlockEventLog(block.events_path, block=block, source="block_control",
                       mono_origin=mono_t, wall_origin=wall_t) as log:
        log.write("BLOCK_START", block_start_timestamp=wall_t,
                  n_cells=block.n_cells, seed=block.seed)
    return metadata


def load_block_metadata(block_dir) -> Dict[str, Any]:
    path = Path(block_dir) / "block_metadata.json"
    if not path.is_file():
        raise ExperimentConfigError(f"block_metadata.json not found in {block_dir}")
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def block_origin(block_dir) -> tuple:
    """(mono_origin, wall_origin) for this block, or (None, None) if unreadable.

    Workers call this to place their own events on the block's shared elapsed-time axis.
    Failure is non-fatal: an event with `elapsed_time = null` is still a usable event.
    """
    try:
        meta = load_block_metadata(block_dir)
        return meta.get("block_start_mono"), meta.get("block_start_timestamp")
    except Exception:
        return None, None


def _iter_events(events_path: Path):
    if not events_path.is_file():
        return
    with open(events_path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # A torn final line can only come from a hard kill mid-write; skip it
                # rather than refusing to summarise everything before it.
                continue


def summarize_events(events_path) -> Dict[str, Any]:
    """Counts only -- integrity/descriptive information, never a study statistic."""
    events_path = Path(events_path)
    counts: Dict[str, int] = {}
    episodes = set()
    cells = set()
    interventions = 0
    outcomes = {"success": 0, "failure": 0, "timeout": 0, "incomplete": 0}
    first_mono = None
    last_mono = None
    for ev in _iter_events(events_path):
        name = str(ev.get("event") or "")
        counts[name] = counts.get(name, 0) + 1
        if ev.get("episode_id"):
            episodes.add((ev.get("cell_id"), ev.get("episode_id")))
        if ev.get("cell_id") is not None:
            cells.add(ev.get("cell_id"))
        if name == "INTERVENTION_REQUEST":
            interventions += 1
        if name == "EPISODE_SUCCESS":
            outcomes["success"] += 1
        elif name == "EPISODE_FAILURE":
            outcomes["failure"] += 1
        elif name == "EPISODE_TIMEOUT":
            outcomes["timeout"] += 1
        mono = ev.get("mono_t")
        if isinstance(mono, (int, float)):
            first_mono = mono if first_mono is None else min(first_mono, mono)
            last_mono = mono if last_mono is None else max(last_mono, mono)
    incomplete = counts.get("EPISODE_END", 0) - sum(
        outcomes[k] for k in ("success", "failure", "timeout")
    )
    outcomes["incomplete"] = max(0, incomplete)
    return {
        "events_total": sum(counts.values()),
        "event_counts": counts,
        "episodes_seen": len(episodes),
        "cells_seen": sorted(c for c in cells if c is not None),
        "interventions_total": interventions,
        "episode_outcomes": outcomes,
        "events_span_s": (None if first_mono is None or last_mono is None
                          else max(0.0, last_mono - first_mono)),
    }


def stop_block(block_dir, *, reason: str = "operator_stop",
               status: str = "complete") -> Dict[str, Any]:
    """Log BLOCK_END, write `block_summary.json`, mark the metadata complete.

    Safe to call twice and safe to call on a block whose processes already died --
    which is the case that matters, because that is what a crash looks like.
    """
    block_dir = Path(block_dir)
    meta = load_block_metadata(block_dir)
    block = ExperimentBlock(**{
        k: v for k, v in meta["block"].items()
        if k in ExperimentBlock.__dataclass_fields__
    })

    wall_t, mono_t = _now()
    mono_origin = meta.get("block_start_mono")
    wall_origin = meta.get("block_start_timestamp")
    with BlockEventLog(block_dir / "events.jsonl", block=block, source="block_control",
                       mono_origin=mono_origin, wall_origin=wall_origin) as log:
        log.write("BLOCK_END", reason=reason, status=status,
                  block_end_timestamp=wall_t)

    summary = summarize_events(block_dir / "events.jsonl")
    summary.update({
        "schema": SCHEMA_VERSION,
        "block_id": block.block_id,
        "participant_id": block.participant_id,
        "study": block.study,
        "condition": block.condition_id,
        "task": block.task,
        "status": status,
        "stop_reason": reason,
        "block_start_timestamp": wall_origin,
        "block_end_timestamp": wall_t,
        "duration_s": (None if mono_origin is None else max(0.0, mono_t - float(mono_origin))),
    })
    _atomic_write_json(block_dir / "block_summary.json", summary)

    meta["status"] = status
    meta["stop_reason"] = reason
    meta["block_end_timestamp"] = wall_t
    meta["block_end_mono"] = mono_t
    if mono_origin is not None:
        meta["duration_s"] = max(0.0, mono_t - float(mono_origin))
    _atomic_write_json(block_dir / "block_metadata.json", meta)
    return summary


# --------------------------------------------------------------------------- tiny CLI

def _cmd_start(args):
    block = build_block(
        participant_id=args.participant,
        study=args.study,
        condition_id=args.condition,
        task=args.task,
        root=args.root,
        seed=args.seed,
        notes=args.notes or "",
        n_cells=args.cells or 0,
    )
    start_block(block, config={"cells": args.cells})
    print(describe(block))
    print(f"[Block] RECORDING ACTIVE -> {block.block_dir}")
    # Printed as shell assignments so `eval "$(... start ... --export)"` wires a session.
    if args.export:
        print(f"export {ENV_PARTICIPANT}={block.participant_id}")
        print(f"export {ENV_STUDY}={block.study}")
        print(f"export {ENV_CONDITION}={block.condition_id}")
        print(f"export {ENV_TASK}={block.task}")
        print(f"export {ENV_ROOT}={block.root}")
        print(f"export {ENV_BLOCK_ID}={block.block_id}")
        print(f"export {ENV_BLOCK_DIR}={block.block_dir}")
        print(f"export {ENV_SEED}={block.seed}")
    return 0


def _cmd_stop(args):
    summary = stop_block(args.block_dir, reason=args.reason, status=args.status)
    print(f"[Block] RECORDING STOPPED -> {args.block_dir}")
    print(f"[Block] {summary['episodes_seen']} episode(s), "
          f"{summary['interventions_total']} intervention(s), "
          f"{summary['events_total']} event(s), status={summary['status']}")
    return 0


def _cmd_status(args):
    meta = load_block_metadata(args.block_dir)
    summary = summarize_events(Path(args.block_dir) / "events.jsonl")
    state = "ACTIVE" if meta.get("status") != "complete" else "STOPPED"
    print(f"[Block] {state}  {meta['block']['block_id']}  ({meta.get('status')})")
    print(f"[Block] dir: {args.block_dir}")
    print(f"[Block] episodes={summary['episodes_seen']} "
          f"interventions={summary['interventions_total']} "
          f"cells={summary['cells_seen']}")
    return 0


FLAG_KEYS = ("START_VR", "RGB", "WITH_PC", "MC_ACTIVE", "MC_SIM", "FACTR_ACTIVE")


def _cmd_infer(args):
    """Print the condition implied by the launch flags. Nothing is recorded."""
    flags = {k: getattr(args, k.lower()) for k in FLAG_KEYS}
    cond = infer_condition(args.study, flags)
    supervision, controller = interfaces_from_flags(flags)
    if args.quiet:
        print(cond)
    else:
        print(f"{cond}\t{supervision}\t{controller}")
    return 0


def _cmd_validate(args):
    block = build_block(
        participant_id=args.participant, study=args.study,
        condition_id=args.condition, task=args.task, root=args.root,
    )
    print(describe(block))
    print(f"  supervision_interface = {block.supervision_interface}")
    print(f"  controller_interface  = {block.controller_interface}")
    print(f"  would write to        = {block.block_dir}")
    return 0


def _cmd_mark(args):
    """Append an arbitrary named marker (e.g. the measured 5-minute window)."""
    meta = load_block_metadata(args.block_dir)
    block = ExperimentBlock(**{
        k: v for k, v in meta["block"].items()
        if k in ExperimentBlock.__dataclass_fields__
    })
    with BlockEventLog(Path(args.block_dir) / "events.jsonl", block=block,
                       source="block_control",
                       mono_origin=meta.get("block_start_mono"),
                       wall_origin=meta.get("block_start_timestamp")) as log:
        ev = log.write(args.name, note=args.note or "")
    print(f"[Block] {ev['event']} at elapsed_time={ev['elapsed_time']}")
    return 0


def main(argv=None):
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="Experiment block control.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def _add_desc(p):
        p.add_argument("--participant", required=True, help="e.g. P003")
        p.add_argument("--study", required=True,
                       help="Supervision or Control (legacy 1A/1B accepted)")
        p.add_argument("--condition", required=True, help="S1|S2|S3 (Supervision) or C1|C2|C3 (Control)")
        p.add_argument("--task", required=True, help="TShape or Cups")
        p.add_argument("--root", default=None, help=f"default {DEFAULT_ROOT}")

    p = sub.add_parser("start", help="Create the block directory and log BLOCK_START.")
    _add_desc(p)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--cells", type=int, default=9)
    p.add_argument("--notes", default="")
    p.add_argument("--export", action="store_true",
                   help="Print `export EXP_*=...` lines for `eval`.")
    p.set_defaults(func=_cmd_start)

    p = sub.add_parser("stop", help="Log BLOCK_END and write block_summary.json.")
    p.add_argument("block_dir")
    p.add_argument("--reason", default="operator_stop")
    p.add_argument("--status", default="complete",
                   choices=["complete", "incomplete", "aborted"])
    p.set_defaults(func=_cmd_stop)

    p = sub.add_parser("status", help="Show whether a block is ACTIVE or STOPPED.")
    p.add_argument("block_dir")
    p.set_defaults(func=_cmd_status)

    p = sub.add_parser("validate", help="Check a combination without recording anything.")
    _add_desc(p)
    p.set_defaults(func=_cmd_validate)

    p = sub.add_parser(
        "infer",
        help="Print the condition implied by a set of launch flags. Records nothing.")
    p.add_argument("--study", required=True,
                   help="Supervision or Control (legacy 1A/1B accepted)")
    for key in FLAG_KEYS:
        p.add_argument(f"--{key.lower()}", default=None,
                       help=f"{key} as the launcher resolved it")
    p.add_argument("--quiet", action="store_true", help="Print the condition id only.")
    p.set_defaults(func=_cmd_infer)

    p = sub.add_parser("mark", help="Append a named marker event to a running block.")
    p.add_argument("block_dir")
    p.add_argument("name", help="e.g. MEASUREMENT_START / MEASUREMENT_END")
    p.add_argument("--note", default="")
    p.set_defaults(func=_cmd_mark)

    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except ExperimentConfigError as exc:
        print(f"[Block][ERROR] {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
