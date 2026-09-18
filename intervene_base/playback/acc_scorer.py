"""ACC — per-step policy-uncertainty scoring for ACT.

`app.py` already publishes `acc_risk` (read from `player.last_acc_score`) all the way
through to the Quest risk bar, but `last_acc_score` was never assigned anywhere: the
value has always been a hard-wired 0.0. This module computes it.

Why not plain temporal-ensemble disagreement
--------------------------------------------
The textbook ACT uncertainty measure is the spread across overlapping action chunks that
each predict the same timestep. That needs overlapping chunks. In this project's
checkpoints:

    chunk_size == n_action_steps  (10 for tshape, 16 for cups)
    temporal_ensemble_coeff == null
    ACTION_MODE=queue             (run_main_policy.sh default)

so the policy consumes a chunk fully before predicting the next and chunks never
overlap. Real overlap requires `ACTION_MODE=replan`, which calls `predict_action_chunk`
every step — 10x the forward passes on a CPU-only torch build. That is a decision about
the study's compute profile, not something to switch on silently, so both estimators
ship and the default is the one that costs nothing.

Everything the default estimator uses is in **post-processed action space** (radians for
`arm_action_mode=absolute`), i.e. the same values the player writes to `data.ctrl`. The
raw chunk tensors inside lerobot are normalized and are NOT comparable to joint angles,
which is why the default path deliberately never touches them.

`chunk_residual` (default, free, per-step)
    tracking_residual  ||q_now - commanded arm target||
        How far the arm actually is from what the policy just asked for. Grows when the
        policy commands something the arm cannot reach — pushing into an object, a
        blocked grasp, a pose the controller is fighting.
    replan_jump        ||a_t - a_{t-1}|| at a fresh-chunk boundary
        The discontinuity when the policy re-plans. A confident policy re-plans onto
        roughly the trajectory it was already following; a jump means the new
        observation changed its mind. (This is the "prediction inconsistency" the
        runtime's existing `--acc_risk_scale` help text already refers to.)
    action_jerk        ||a_t - 2*a_{t-1} + a_{t-2}||
        Within-chunk smoothness; noisy commands indicate a policy that is not settled.

`ensemble_disagreement` (opt-in)
    True per-step spread across the last K chunks predicting the current step. Requires
    both `ACTION_MODE=replan` (or a non-null `temporal_ensemble_coeff`) AND a working
    de-normalization of the chunk; if either is unavailable the score is reported with
    `valid=False` rather than silently returning a meaningless number.

Every component is logged alongside the scalar, so ACC can be re-derived offline from a
recorded episode without re-running a participant.
"""

import os
from collections import deque
from typing import Dict, Optional

import numpy as np

METHOD_CHUNK_RESIDUAL = "chunk_residual"
METHOD_ENSEMBLE = "ensemble_disagreement"
METHOD_NONE = "none"
METHODS = (METHOD_CHUNK_RESIDUAL, METHOD_ENSEMBLE, METHOD_NONE)

# Fixed order — `acc_components` is a plain (N, k) array and readers index positionally.
COMPONENT_NAMES = (
    "tracking_residual_rad",
    "replan_jump_rad",
    "action_jerk_rad",
    "chunk_spread_rad",
    "chunk_age_steps",
)


def _arm(vec) -> np.ndarray:
    return np.asarray(vec, dtype=np.float64).reshape(-1)[:7]


class AccScorer:
    """Stateful per-step uncertainty estimator. One instance per policy player."""

    def __init__(
        self,
        method: Optional[str] = None,
        arm_action_mode: str = "absolute",
        scale: Optional[float] = None,
        ensemble_window: int = 8,
    ):
        method = (method or os.environ.get("STUDY_ACC_METHOD")
                  or METHOD_CHUNK_RESIDUAL).strip().lower()
        if method not in METHODS:
            print(f"[ACC][WARN] unknown STUDY_ACC_METHOD={method!r}; "
                  f"falling back to {METHOD_CHUNK_RESIDUAL}")
            method = METHOD_CHUNK_RESIDUAL
        self.method = method
        self.arm_action_mode = str(arm_action_mode)
        # Radians that map to ACC = 1.0. The two methods live on different scales, so
        # they get different defaults — both calibrated from measured runs of the tshape
        # checkpoint, not guessed:
        #   chunk_residual: tracking error + re-plan jump run ~0.03-0.15 rad
        #   ensemble:       chunk spread runs ~0.002-0.028 rad (min/median/max measured
        #                   0.0021 / 0.0060 / 0.0278 over 40 steps)
        # Override per task with STUDY_ACC_SCALE once you have baseline runs.
        default_scale = 0.03 if method == METHOD_ENSEMBLE else 0.15
        env_scale = os.environ.get("STUDY_ACC_SCALE")
        self.scale = float(
            scale if scale is not None
            else (env_scale if env_scale not in (None, "") else default_scale)
        )
        self.ensemble_window = int(ensemble_window)

        self._prev_action: Optional[np.ndarray] = None
        self._prev_prev_action: Optional[np.ndarray] = None
        self._replan_jump = 0.0
        self._chunk_age = 0
        self._history = deque(maxlen=self.ensemble_window)   # (step, chunk) if usable
        self._global_step = 0
        self._warned_ensemble = False

        self.last_score = 0.0
        self.last_components = {name: 0.0 for name in COMPONENT_NAMES}
        self.last_valid = False

    # ------------------------------------------------------------------ input

    def note_chunk(self, chunk_in_action_space, step: Optional[int] = None) -> None:
        """Optionally record a de-normalized action chunk for the ensemble estimator.

        Only the `ensemble_disagreement` method uses this. The caller must pass a chunk
        already in post-processed action space; a normalized chunk would silently
        corrupt the spread, so anything unusable is dropped instead of guessed at.

        `step` is the global step the chunk was PREDICTED at, which is not necessarily
        the step at which it arrives — the probe runs asynchronously, so a chunk can land
        one or two steps late. Indexing by the prediction step keeps the temporal
        alignment correct; using the arrival step would shear every offset and silently
        understate the spread.
        """
        if chunk_in_action_space is None:
            return
        try:
            arr = (chunk_in_action_space.detach().cpu().numpy()
                   if hasattr(chunk_in_action_space, "detach")
                   else np.asarray(chunk_in_action_space))
            arr = np.asarray(arr, dtype=np.float64)
        except Exception:
            return
        if arr.ndim == 3:
            arr = arr[0]
        if arr.ndim != 2 or arr.shape[0] == 0:
            return
        self._history.append((int(step) if step is not None else self._global_step, arr))

    # ----------------------------------------------------------------- update

    def update(self, q_now, pred_action, queried_policy: bool = False) -> float:
        """Advance one policy step and return the ACC score in [0, 1].

        `pred_action` must be the POST-processed action (what the player is about to
        turn into `data.ctrl`), and `queried_policy` True on steps where the policy ran
        a fresh forward pass — i.e. where a new chunk began.
        """
        q_now = _arm(q_now)
        action = _arm(pred_action) if pred_action is not None else None
        components = {name: 0.0 for name in COMPONENT_NAMES}

        if queried_policy:
            self._chunk_age = 0
            if action is not None and self._prev_action is not None:
                self._replan_jump = float(np.linalg.norm(action - self._prev_action))
        components["chunk_age_steps"] = float(self._chunk_age)
        components["replan_jump_rad"] = float(self._replan_jump)

        tracking = 0.0
        if action is not None:
            if self.arm_action_mode == "absolute":
                # Action IS the joint target, so this is commanded-vs-achieved error.
                tracking = float(np.linalg.norm(q_now - action))
            else:
                # Delta modes: the action is a step, not a target. Its magnitude is the
                # closest free analogue (large steps = the policy wants a big change).
                tracking = float(np.linalg.norm(action))
        components["tracking_residual_rad"] = tracking

        jerk = 0.0
        if action is not None and self._prev_action is not None \
                and self._prev_prev_action is not None:
            jerk = float(np.linalg.norm(
                action - 2.0 * self._prev_action + self._prev_prev_action))
        components["action_jerk_rad"] = jerk

        spread, spread_valid = self._ensemble_spread()
        components["chunk_spread_rad"] = spread

        if self.method == METHOD_NONE:
            score, valid = 0.0, False
        elif self.method == METHOD_ENSEMBLE:
            score, valid = spread / max(self.scale, 1e-9), spread_valid
            if not spread_valid and not self._warned_ensemble:
                print("[ACC][WARN] ensemble_disagreement selected but no overlapping "
                      "chunks are available (queue mode with chunk_size == "
                      "n_action_steps, or chunks not supplied in action space). Scores "
                      "are flagged invalid. Use ACTION_MODE=replan, or "
                      "STUDY_ACC_METHOD=chunk_residual.")
                self._warned_ensemble = True
        else:
            # Tracking error dominates; the re-plan jump corroborates, so it is weighted
            # down rather than summed equally.
            raw = tracking + 0.5 * self._replan_jump
            score, valid = raw / max(self.scale, 1e-9), action is not None

        score = float(np.clip(score, 0.0, 1.0)) if np.isfinite(score) else 0.0

        if action is not None:
            self._prev_prev_action = self._prev_action
            self._prev_action = action
        self._chunk_age += 1
        self._global_step += 1
        self.last_score = score
        self.last_components = components
        self.last_valid = bool(valid)
        return score

    def _ensemble_spread(self):
        """Std-dev across chunks that all predict the current global step."""
        if len(self._history) < 2:
            return 0.0, False
        preds = []
        for start_step, chunk in self._history:
            offset = self._global_step - start_step
            if 0 <= offset < chunk.shape[0]:
                preds.append(_arm(chunk[offset]))
        if len(preds) < 2:
            return 0.0, False
        return float(np.mean(np.std(np.stack(preds), axis=0))), True

    # ------------------------------------------------------------------ misc

    def reset(self) -> None:
        self._prev_action = None
        self._prev_prev_action = None
        self._replan_jump = 0.0
        self._chunk_age = 0
        self._history.clear()
        self._global_step = 0
        self.last_score = 0.0
        self.last_components = {name: 0.0 for name in COMPONENT_NAMES}
        self.last_valid = False

    def components_vector(self) -> np.ndarray:
        return np.array([self.last_components.get(n, 0.0) for n in COMPONENT_NAMES],
                        dtype=np.float32)

    def describe(self) -> str:
        return (f"[ACC] method={self.method} scale={self.scale} "
                f"arm_action_mode={self.arm_action_mode} "
                f"components={list(COMPONENT_NAMES)}")

    def snapshot(self) -> Dict[str, float]:
        out = {"acc": self.last_score, "acc_valid": self.last_valid,
               "acc_method": self.method}
        out.update(self.last_components)
        return out
