r"""Episode- and intervention-level logging for the user study.

The trajectory NPZ already captures per-frame state. What was missing is everything
*around* it: who the participant was, which condition they were in, whether the episode
succeeded, how many times they intervened, and — the part a study actually turns on —
when each phase of an intervention happened.

Two artefacts per session:

  * `events.jsonl` — append-only, flushed on every write. This is the source of truth.
    A session that crashes mid-episode still leaves a complete record up to that point,
    which matters when the participant is sitting in the headset and you cannot redo it.
  * `session.json` — the block config, environment provenance (git SHAs, XML,
    checkpoint), and a rolling summary rewritten after each episode.

Timebases: every event carries both `wall_t` (`time.time()`, joinable with the perf
metrics and the NPZ's `wall_t`) and `mono_t` (`time.monotonic()`). All *durations* are
computed from the monotonic pair, so an NTP step during a session can never produce a
negative synchronization or correction time.

Intervention phases, matching the runtime's actual sequence:

    request ──► pause ──► ready(aligned) ──► release(human control) ──► finish
             \________ synchronization ________/\____ correction ____/

Block mode (HRI study, 2026-08-11). When an `data_io.experiment_block.ExperimentBlock` is
passed in, nine of these sessions -- one per concurrent cell -- share ONE output
directory and ONE `events.jsonl`, written through `BlockEventLog` (an `O_APPEND` fd, so
concurrent appends from all nine processes plus the grid and the VR runtimes are
serialised by the kernel). Each cell keeps its own `cells/cell_NN.json` rather than
racing on a single `session.json`. Every event then also carries the canonical
uppercase `event` name (see `CANONICAL_EVENT`) alongside the historical `kind`, so
existing consumers -- `utils/validate_study_data.py` reads `kind` -- keep working.
"""

import json
import os
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

SCHEMA_VERSION = 1

# Historical `kind` -> the canonical study event name. Both are written on every line:
# `kind` is what already-shipped tooling parses, `event` is what the HRI analysis keys
# off. Adding a name here is how a new event becomes visible to the block-level tools.
CANONICAL_EVENT = {
    "session_start": "CELL_SESSION_START",
    "session_end": "CELL_SESSION_END",
    "episode_start": "EPISODE_START",
    "episode_end": "EPISODE_END",
    "intervention_requested": "INTERVENTION_REQUEST",
    # The policy is paused and the robot begins aligning: the controller is being
    # prepared for the human. This is the phase the study calls CONTROLLER_PREPARATION.
    "intervention_paused": "CONTROLLER_PREPARATION_START",
    "intervention_ready": "READY",
    "intervention_released": "HUMAN_CONTROL_START",
    # RELEASE = the human hands control back (finish/cancel press), NOT the moment they
    # received it. `intervention_released` above is when they GOT control.
    "intervention_finished": "RELEASE",
    "post_release_check": "POST_RELEASE_CHECK",
    "autonomy_resumed": "AUTONOMY_RESUMED",
    "episode_success": "EPISODE_SUCCESS",
    "episode_failure": "EPISODE_FAILURE",
    "episode_timeout": "EPISODE_TIMEOUT",
    "cell_selected": "CELL_SELECTED",
    "intervention_failed": "INTERVENTION_FAILED",
}

OUTCOME_SUCCESS = "success"
OUTCOME_FAILURE = "failure"
OUTCOME_INCOMPLETE = "incomplete"
OUTCOMES = (OUTCOME_SUCCESS, OUTCOME_FAILURE, OUTCOME_INCOMPLETE)

# The task detector reports a timeout as a failure with one of these reasons
# (`LiveTaskEvaluator.decide` in app.py). Splitting EPISODE_TIMEOUT out of
# EPISODE_FAILURE happens HERE, in the logger, so the detector itself is untouched.
TIMEOUT_REASONS = ("duration_exceeded", "out_of_time")


def _now():
    return time.time(), time.monotonic()


def _git_sha(path) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


@dataclass
class InterventionRecord:
    """One human takeover, from the request press to the finish/cancel press."""

    intervention_id: int
    episode_id: str
    # Phase timestamps (wall clock; None until the phase happens).
    requested_wall_t: Optional[float] = None
    paused_wall_t: Optional[float] = None
    ready_wall_t: Optional[float] = None
    released_wall_t: Optional[float] = None
    finished_wall_t: Optional[float] = None
    # Monotonic counterparts — durations are derived from these only.
    _requested_mono: Optional[float] = None
    _paused_mono: Optional[float] = None
    _ready_mono: Optional[float] = None
    _released_mono: Optional[float] = None
    _finished_mono: Optional[float] = None

    frame_idx_at_request: Optional[int] = None
    mode: str = ""                     # telekinesis | motion_controller | sim_mc | factr
    outcome: str = "pending"           # completed | cancelled | aborted | pending
    robot_state_at_takeover: Dict[str, Any] = field(default_factory=dict)
    robot_state_at_release: Dict[str, Any] = field(default_factory=dict)
    recovery_result: Optional[str] = None   # back-filled from the episode outcome
    notes: str = ""

    # --- derived durations -------------------------------------------------

    def _delta(self, a, b) -> Optional[float]:
        if a is None or b is None:
            return None
        return max(0.0, b - a)

    @property
    def synchronization_time_s(self) -> Optional[float]:
        """Request press → robot released to the human. The waiting cost of a takeover."""
        return self._delta(self._requested_mono, self._released_mono)

    @property
    def correction_time_s(self) -> Optional[float]:
        """Human in control → finish press. The actual corrective work."""
        return self._delta(self._released_mono, self._finished_mono)

    @property
    def pause_latency_s(self) -> Optional[float]:
        return self._delta(self._requested_mono, self._paused_mono)

    @property
    def alignment_time_s(self) -> Optional[float]:
        """Pause → robot aligned and READY. Usually the bulk of synchronization time."""
        return self._delta(self._paused_mono, self._ready_mono)

    @property
    def total_time_s(self) -> Optional[float]:
        return self._delta(self._requested_mono, self._finished_mono)

    def to_dict(self) -> Dict[str, Any]:
        out = {k: v for k, v in asdict(self).items() if not k.startswith("_")}
        out.update({
            "synchronization_time_s": self.synchronization_time_s,
            "correction_time_s": self.correction_time_s,
            "pause_latency_s": self.pause_latency_s,
            "alignment_time_s": self.alignment_time_s,
            "total_time_s": self.total_time_s,
        })
        return out


@dataclass
class EpisodeRecord:
    episode_id: str
    episode_number: int
    scenario_id: str
    episode_seed: int
    start_wall_t: float
    _start_mono: float
    end_wall_t: Optional[float] = None
    _end_mono: Optional[float] = None
    outcome: Optional[str] = None
    reason: str = ""
    decided_by: str = ""               # auto | manual | shutdown
    npz_path: Optional[str] = None
    frames: Optional[int] = None
    randomization: Dict[str, Any] = field(default_factory=dict)
    interventions: List[InterventionRecord] = field(default_factory=list)

    @property
    def duration_s(self) -> Optional[float]:
        if self._end_mono is None:
            return None
        return max(0.0, self._end_mono - self._start_mono)

    @property
    def n_interventions(self) -> int:
        return len(self.interventions)

    @property
    def success_without_intervention(self) -> bool:
        return self.outcome == OUTCOME_SUCCESS and self.n_interventions == 0

    @property
    def success_after_intervention(self) -> bool:
        return self.outcome == OUTCOME_SUCCESS and self.n_interventions > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "episode_number": self.episode_number,
            "scenario_id": self.scenario_id,
            "episode_seed": self.episode_seed,
            "start_wall_t": self.start_wall_t,
            "end_wall_t": self.end_wall_t,
            "duration_s": self.duration_s,
            "outcome": self.outcome,
            "reason": self.reason,
            "decided_by": self.decided_by,
            "npz_path": self.npz_path,
            "frames": self.frames,
            "randomization": self.randomization,
            "n_interventions": self.n_interventions,
            "success_without_intervention": self.success_without_intervention,
            "success_after_intervention": self.success_after_intervention,
            "interventions": [i.to_dict() for i in self.interventions],
        }


class StudySession:
    """Owns one participant-block's output directory, event log and summary."""

    def __init__(
        self,
        block,
        root_dir=None,
        env_info: Optional[Dict[str, Any]] = None,
        repo_root=None,
        exp_block=None,
        cell_id: Optional[int] = None,
    ):
        self.block = block
        self.exp_block = exp_block
        self.cell_id = cell_id
        self.started_wall_t, self._started_mono = _now()
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(self.started_wall_t))

        # Optional hook returning {"simulation_step": int, "policy_step": int}. app.py
        # installs it so every event carries the step counters that let offline analysis
        # prove whether the policy acted between RELEASE and SUCCESS. Never allowed to
        # raise (see `_step_fields`).
        self.step_provider: Optional[Callable[[], Dict[str, Any]]] = None
        # Set by whoever owns the selection signal; stamped onto per-cell events.
        self.cell_selected = False

        self._block_log = None
        if exp_block is not None:
            # Block mode: nine cells share one directory. The directory is created by
            # `experiment_block.start_block` before any process launches, so a worker
            # only ever attaches to it.
            from data_io.experiment_block import BlockEventLog, block_origin

            self.session_dir = exp_block.block_dir
            self.episodes_dir = exp_block.episodes_dir
            for outcome in OUTCOMES:
                (self.episodes_dir / outcome).mkdir(parents=True, exist_ok=True)
            (self.session_dir / "cells").mkdir(parents=True, exist_ok=True)
            self.events_path = exp_block.events_path
            self.session_path = (self.session_dir / "cells"
                                 / f"cell_{int(cell_id or 0):02d}.json")
            mono_origin, wall_origin = block_origin(self.session_dir)
            self._block_log = BlockEventLog(
                self.events_path, block=exp_block, cell_id=cell_id, source="policy",
                mono_origin=mono_origin, wall_origin=wall_origin,
            )
            self._events_fh = None
        else:
            root = Path(root_dir or os.environ.get("INTERVENE_EPISODE_DIR")
                        or os.environ.get("INTERVENE_OUTPUT_DIR") or "INTERVENTION_DATA")
            self.session_dir = root / block.participant_id / f"{block.label}_{stamp}"
            self.episodes_dir = self.session_dir / "episodes"
            for outcome in OUTCOMES:
                (self.episodes_dir / outcome).mkdir(parents=True, exist_ok=True)
            self.events_path = self.session_dir / "events.jsonl"
            self.session_path = self.session_dir / "session.json"
            self._events_fh = open(self.events_path, "a", encoding="utf-8", buffering=1)
        self._event_seq = 0

        self.episodes: List[EpisodeRecord] = []
        self.current_episode: Optional[EpisodeRecord] = None
        self.current_intervention: Optional[InterventionRecord] = None
        self._next_intervention_id = 1

        self.env_info = dict(env_info or {})
        repo_root = repo_root or Path(__file__).resolve().parents[2]
        self.env_info.setdefault("git_sha_umbrella", _git_sha(repo_root))
        self.env_info.setdefault("git_sha_intervene_base",
                                 _git_sha(Path(__file__).resolve().parents[1]))
        self.env_info.setdefault("host", os.uname().nodename if hasattr(os, "uname") else "")
        self.env_info.setdefault("pid", os.getpid())

        self.log_event("session_start", block=block.to_dict(), env=self.env_info)
        self._write_session_json()
        print(f"[Study] session dir: {self.session_dir}"
              + (f" (cell {int(cell_id or 0):02d})" if exp_block is not None else ""))

    # ------------------------------------------------------------- events

    def _step_fields(self) -> Dict[str, Any]:
        """simulation_step / policy_step for this instant, or nulls.

        Wrapped: a bookkeeping hook must never be able to abort an episode or a live
        takeover, which is the same rule `LiveReplanSession._notify` follows.
        """
        if self.step_provider is None:
            return {"simulation_step": None, "policy_step": None}
        try:
            got = self.step_provider() or {}
            return {
                "simulation_step": got.get("simulation_step"),
                "policy_step": got.get("policy_step"),
            }
        except Exception:
            return {"simulation_step": None, "policy_step": None}

    def log_event(self, kind: str, **fields) -> Dict[str, Any]:
        self._event_seq += 1
        episode_id = (self.current_episode.episode_id
                      if self.current_episode is not None else None)
        payload: Dict[str, Any] = {
            "kind": kind,
            "block": self.block.block,
            "interface": self.block.interface,
            "scene_distribution": self.block.condition,
            "control": getattr(self.block, "control", ""),
            "cell_selected": bool(self.cell_selected),
        }
        payload.update(self._step_fields())
        if self.current_episode is not None:
            payload["episode_number"] = self.current_episode.episode_number
            payload["scenario_id"] = self.current_episode.scenario_id
        payload.update(fields)

        if self._block_log is not None:
            return self._block_log.write(
                CANONICAL_EVENT.get(kind, kind.upper()),
                episode_id=episode_id, **payload,
            )

        wall_t, mono_t = _now()
        event = {
            "schema": SCHEMA_VERSION,
            "seq": self._event_seq,
            "event": CANONICAL_EVENT.get(kind, kind.upper()),
            "wall_t": wall_t,
            "mono_t": mono_t,
            "timestamp": wall_t,
            "participant_id": self.block.participant_id,
            "task": self.block.task,
            "condition": self.block.condition,
            "cell_id": self.cell_id,
        }
        event.update(payload)
        if episode_id is not None:
            event["episode_id"] = episode_id
        try:
            self._events_fh.write(json.dumps(event, default=str, separators=(",", ":")) + "\n")
        except Exception as exc:  # never take the session down over logging
            print(f"[Study][WARN] event write failed ({kind}): {exc}")
        return event

    # ----------------------------------------------------- study-specific events

    def log_cell_selected(self, cell_id: int, selection_method: str, **fields):
        """The operator confirmed a cell. Only the CONFIRMED selection is recorded.

        Continuous Quest-ray / cursor position is deliberately not logged: the analysis
        derives inspection time from (INTERVENTION_REQUEST - most recent CELL_SELECTED),
        which needs the confirmation instant and nothing else.
        """
        self.cell_selected = (int(cell_id) == int(self.cell_id or 0))
        return self.log_event("cell_selected", selected_cell_id=int(cell_id),
                              selection_method=str(selection_method), **fields)

    def log_post_release_check(self, status: str, reason: str = "", **fields):
        """The task state evaluated at the instant the human released control.

        `status` is one of `success` / `failure` / `none`, taken straight from the
        existing task detector -- this is instrumentation, not a second evaluator.
        """
        return self.log_event("post_release_check", check_status=str(status),
                              reason=str(reason or ""), **fields)

    def log_autonomy_resumed(self, **fields):
        return self.log_event("autonomy_resumed", **fields)

    def log_episode_outcome(self, outcome: str, reason: str = "", **fields):
        """EPISODE_SUCCESS / EPISODE_FAILURE / EPISODE_TIMEOUT as their own events.

        `episode_end` already carries the outcome, but the study's event vocabulary
        wants the three terminal states as first-class names so a reader never has to
        interpret a field to find them. Timeout is a *kind* of failure in the detector
        (`duration_exceeded` / `out_of_time`), so it is split out here rather than in
        the detector -- no task logic changes.
        """
        outcome = str(outcome)
        reason = str(reason or "")
        if outcome == OUTCOME_SUCCESS:
            kind = "episode_success"
        elif outcome == OUTCOME_FAILURE:
            kind = ("episode_timeout"
                    if reason in TIMEOUT_REASONS else "episode_failure")
        else:
            return None
        return self.log_event(kind, outcome=outcome, reason=reason, **fields)

    # ------------------------------------------------------------ episodes

    def start_episode(self, episode_number: int, episode_id: str,
                      randomization: Optional[Dict[str, Any]] = None) -> EpisodeRecord:
        if self.current_episode is not None and self.current_episode.outcome is None:
            # Defensive: a new episode starting while one is open means we lost an
            # outcome somewhere. Close it as incomplete rather than dropping it.
            self.end_episode(OUTCOME_INCOMPLETE, reason="superseded", decided_by="auto")

        wall_t, mono_t = _now()
        record = EpisodeRecord(
            episode_id=episode_id,
            episode_number=episode_number,
            scenario_id=self.block.scenario_id(episode_number),
            episode_seed=self.block.episode_seed(episode_number),
            start_wall_t=wall_t,
            _start_mono=mono_t,
            randomization=dict(randomization or {}),
        )
        self.current_episode = record
        self.episodes.append(record)
        self._next_intervention_id = 1
        self.log_event("episode_start", episode_seed=record.episode_seed,
                       randomization=record.randomization)
        return record

    def end_episode(self, outcome: str, reason: str = "", decided_by: str = "auto",
                    npz_path=None, frames: Optional[int] = None) -> Optional[EpisodeRecord]:
        record = self.current_episode
        if record is None:
            return None
        if outcome not in OUTCOMES:
            outcome = OUTCOME_INCOMPLETE
        wall_t, mono_t = _now()
        record.end_wall_t = wall_t
        record._end_mono = mono_t
        record.outcome = outcome
        record.reason = reason or ""
        record.decided_by = decided_by
        if npz_path is not None:
            record.npz_path = str(npz_path)
        if frames is not None:
            record.frames = int(frames)

        # An intervention still open at episode end never got its finish press.
        if self.current_intervention is not None:
            self.finish_intervention(outcome="aborted", reason="episode_ended")

        for iv in record.interventions:
            iv.recovery_result = outcome

        # Terminal state first, then the generic close. Emitting SUCCESS/FAILURE/TIMEOUT
        # while `current_episode` is still set is what keeps `episode_id` on those lines.
        self.log_episode_outcome(outcome, reason=record.reason, decided_by=decided_by,
                                 n_interventions=record.n_interventions)
        self.log_event(
            "episode_end", outcome=outcome, reason=record.reason,
            decided_by=decided_by, duration_s=record.duration_s,
            n_interventions=record.n_interventions,
            success_without_intervention=record.success_without_intervention,
            success_after_intervention=record.success_after_intervention,
            npz_path=record.npz_path, frames=record.frames,
        )
        self.current_episode = None
        self._write_session_json()
        return record

    def episode_subdir(self, outcome: str) -> Path:
        if outcome not in OUTCOMES:
            outcome = OUTCOME_INCOMPLETE
        path = self.episodes_dir / outcome
        path.mkdir(parents=True, exist_ok=True)
        return path

    # -------------------------------------------------------- interventions

    def start_intervention(self, mode: str = "", frame_idx: Optional[int] = None
                           ) -> InterventionRecord:
        """Phase 1: the operator pressed the intervene button."""
        wall_t, mono_t = _now()
        record = InterventionRecord(
            intervention_id=self._next_intervention_id,
            episode_id=self.current_episode.episode_id if self.current_episode else "",
            requested_wall_t=wall_t,
            _requested_mono=mono_t,
            frame_idx_at_request=frame_idx,
            mode=mode,
        )
        self._next_intervention_id += 1
        self.current_intervention = record
        if self.current_episode is not None:
            self.current_episode.interventions.append(record)
        self.log_event("intervention_requested",
                       intervention_id=record.intervention_id,
                       mode=mode, frame_idx=frame_idx)
        return record

    def mark_intervention_paused(self, **fields):
        """Phase 2: the policy is paused and the robot is holding."""
        record = self.current_intervention
        if record is None:
            return
        record.paused_wall_t, record._paused_mono = _now()
        self.log_event("intervention_paused",
                       intervention_id=record.intervention_id,
                       pause_latency_s=record.pause_latency_s, **fields)

    def mark_intervention_ready(self, **fields):
        """Phase 3: alignment finished — the robot is at the takeover pose (READY)."""
        record = self.current_intervention
        if record is None:
            return
        record.ready_wall_t, record._ready_mono = _now()
        self.log_event("intervention_ready",
                       intervention_id=record.intervention_id,
                       alignment_time_s=record.alignment_time_s, **fields)

    def mark_intervention_released(self, robot_state: Optional[Dict] = None, **fields):
        """Phase 4: control handed to the human (HUMAN_CONTROL / impedance hold)."""
        record = self.current_intervention
        if record is None:
            return
        record.released_wall_t, record._released_mono = _now()
        if robot_state:
            record.robot_state_at_takeover = dict(robot_state)
        self.log_event("intervention_released",
                       intervention_id=record.intervention_id,
                       synchronization_time_s=record.synchronization_time_s,
                       robot_state=record.robot_state_at_takeover, **fields)

    def finish_intervention(self, outcome: str = "completed",
                            robot_state: Optional[Dict] = None,
                            reason: str = "", **fields) -> Optional[InterventionRecord]:
        """Phase 5: finish or cancel press — the human is done."""
        record = self.current_intervention
        if record is None:
            return None
        record.finished_wall_t, record._finished_mono = _now()
        record.outcome = outcome
        if reason:
            record.notes = reason
        if robot_state:
            record.robot_state_at_release = dict(robot_state)
        self.log_event("intervention_finished",
                       intervention_id=record.intervention_id,
                       outcome=outcome, reason=reason,
                       correction_time_s=record.correction_time_s,
                       synchronization_time_s=record.synchronization_time_s,
                       total_time_s=record.total_time_s,
                       robot_state=record.robot_state_at_release, **fields)
        self.current_intervention = None
        return record

    # --------------------------------------------------------- summary/close

    def summary(self) -> Dict[str, Any]:
        done = [e for e in self.episodes if e.outcome is not None]
        successes = [e for e in done if e.outcome == OUTCOME_SUCCESS]
        failures = [e for e in done if e.outcome == OUTCOME_FAILURE]
        all_ivs = [iv for e in self.episodes for iv in e.interventions]
        sync = [iv.synchronization_time_s for iv in all_ivs
                if iv.synchronization_time_s is not None]
        corr = [iv.correction_time_s for iv in all_ivs
                if iv.correction_time_s is not None]

        def mean(vals):
            return (sum(vals) / len(vals)) if vals else None

        return {
            "episodes_total": len(self.episodes),
            "episodes_completed": len(done),
            "successes": len(successes),
            "failures": len(failures),
            "success_rate": (len(successes) / len(done)) if done else None,
            "success_without_intervention": sum(
                1 for e in successes if e.success_without_intervention),
            "success_after_intervention": sum(
                1 for e in successes if e.success_after_intervention),
            "interventions_total": len(all_ivs),
            "interventions_per_episode": (len(all_ivs) / len(done)) if done else None,
            "mean_synchronization_time_s": mean(sync),
            "mean_correction_time_s": mean(corr),
        }

    def _write_session_json(self):
        payload = {
            "schema": SCHEMA_VERSION,
            "participant_id": self.block.participant_id,
            "cell_id": self.cell_id,
            "experiment_block": (self.exp_block.to_dict()
                                 if self.exp_block is not None else None),
            "block": self.block.to_dict(),
            "session_dir": str(self.session_dir),
            "started_wall_t": self.started_wall_t,
            "ended_wall_t": getattr(self, "ended_wall_t", None),
            "env": self.env_info,
            "summary": self.summary(),
            "episodes": [e.to_dict() for e in self.episodes],
        }
        try:
            tmp = self.session_path.with_suffix(".json.tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, default=str)
            tmp.replace(self.session_path)  # atomic: never leave a half-written summary
        except Exception as exc:
            print(f"[Study][WARN] session.json write failed: {exc}")

    def close(self, reason: str = "shutdown"):
        if self.current_episode is not None:
            self.end_episode(OUTCOME_INCOMPLETE, reason=reason, decided_by="shutdown")
        self.ended_wall_t = time.time()
        summary = self.summary()
        self.log_event("session_end", reason=reason, summary=summary)
        self._write_session_json()
        try:
            if self._events_fh is not None:
                self._events_fh.flush()
                self._events_fh.close()
            if self._block_log is not None:
                self._block_log.close()
        except Exception:
            pass
        print(f"[Study] session closed: {self.session_dir}")
        print(f"[Study] {summary['episodes_completed']} episode(s), "
              f"{summary['successes']} success / {summary['failures']} failure, "
              f"{summary['interventions_total']} intervention(s)")
