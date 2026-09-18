import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
import torch

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

from playback.acc_scorer import METHOD_ENSEMBLE as ACC_METHOD_ENSEMBLE, AccScorer
from playback.player import gripper_ctrl_to_width
from utils.rollout_act_mujoco import (
    PANDA_JOINT_NAMES,
    action_needs_observation,
    build_state,
    chw_float01_from_rgb,
    clip_to_actuator_ranges,
    get_qpos_indices_for_joints,
    policy_image_camera_names,
    reset_from_npz,
    select_processed_action,
    validate_camera_exists,
)


@dataclass
class PolicyPlayerConfig:
    checkpoint: Path
    reset_npz: Path
    policy_hz: float = 10.0
    action_mode: str = "queue"
    arm_action_mode: str = "absolute"
    gripper_action_mode: str = "absolute"
    arm_delta_clip: float | None = None
    realtime_factor: float = 2.0
    max_steps: int = 0


def _reset_component(component):
    if hasattr(component, "reset"):
        component.reset()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


class AccProbeWorker:
    """Runs ACC measurement forward passes off the control step.

    The probe is pure instrumentation, so it must never extend a policy step. This mirrors
    `PCWorkerThread` in `SimPublisher/sii/integration_v1/runtime_impl.py`: latest-wins
    submission (a stale probe is worthless), a single daemon thread, and results drained by
    the caller whenever they happen to be ready.

    Note the forward pass releases the GIL inside torch, so the main thread genuinely
    proceeds during it.
    """

    def __init__(self, policy, postprocessor):
        self._policy = policy
        self._postprocessor = postprocessor
        self._pending = None            # (step, batch) — latest only
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._out = queue.Queue()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="AccProbe")
        self._thread.start()

    def submit(self, step: int, batch) -> None:
        with self._lock:
            self._pending = (int(step), batch)
        self._wake.set()

    def drain(self):
        """Return [(step, chunk_in_action_space), ...] ready since the last call."""
        out = []
        while True:
            try:
                out.append(self._out.get_nowait())
            except queue.Empty:
                break
        return out

    def clear(self) -> None:
        with self._lock:
            self._pending = None
        while True:
            try:
                self._out.get_nowait()
            except queue.Empty:
                break

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        try:
            self._thread.join(timeout=2.0)
        except Exception:
            pass

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self._wake.wait(timeout=0.2):
                continue
            self._wake.clear()
            with self._lock:
                job = self._pending
                self._pending = None
            if job is None or self._stop.is_set():
                continue
            step, batch = job
            try:
                with torch.inference_mode():
                    chunk = self._policy.predict_action_chunk(batch)
                    self._out.put((step, self._postprocessor(chunk)))
            except Exception as exc:
                print(f"[ACC][WARN] probe worker forward failed: {exc}")


class PolicyPlayer:
    """MuJoCo player-compatible wrapper that drives the sim with an ACT policy."""

    def __init__(self, xml_path: str, config: PolicyPlayerConfig, context_current_fn=None):
        self.xml_path = str(xml_path)
        self.npz_path = str(config.reset_npz)
        self.config = config
        self.context_current_fn = context_current_fn

        if self.config.policy_hz <= 0:
            raise ValueError("policy_hz must be > 0")
        if self.config.realtime_factor < 0:
            raise ValueError("realtime_factor must be >= 0")
        if self.config.max_steps < 0:
            raise ValueError("max_steps must be >= 0")

        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        if self.model.nu < 8:
            raise ValueError(f"Policy mode expects at least 8 actuators, model has nu={self.model.nu}")
        self.policy_dt = 1.0 / self.config.policy_hz
        self.n_substeps = max(1, int(round(self.policy_dt / self.model.opt.timestep)))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Loading ACT checkpoint from: {self.config.checkpoint}")
        print(f"[INFO] Torch device: {self.device}")
        self.policy = ACTPolicy.from_pretrained(self.config.checkpoint)
        self.policy.eval()
        self.policy.to(self.device)
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config,
            pretrained_path=str(self.config.checkpoint),
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
            postprocessor_overrides={"device_processor": {"device": "cpu"}},
        )

        self.state_dim = self.policy.config.input_features["observation.state"].shape[0]
        self.camera_names = policy_image_camera_names(self.policy)
        for camera_name in self.camera_names:
            validate_camera_exists(self.model, camera_name)
        self.qpos_indices = get_qpos_indices_for_joints(self.model, PANDA_JOINT_NAMES)
        self.policy_renderer = mujoco.Renderer(self.model, height=224, width=224)

        self.reset_episode_len = self._get_reset_episode_len()
        self.frame_idx = 0
        self.policy_query_count = 0
        self.last_target_ctrl = np.zeros(min(8, self.model.nu), dtype=np.float32)
        self.last_pred_action = np.zeros(8, dtype=np.float32)
        self.last_queried_policy = False
        self.last_loop_ms = 0.0
        self.last_observation_ms = 0.0
        self.last_observation_renders = 0
        # ACC (policy uncertainty). app.py has always published `last_acc_score` as
        # acc_risk -> SimPub/Status/risk -> the Quest risk bar, but nothing ever
        # assigned it. AccScorer fills it from values this loop already computes.
        self.acc_scorer = AccScorer(arm_action_mode=self.config.arm_action_mode)
        # Probe on steps the policy already built an observation for, plus one mid-chunk
        # step (needed for chunk overlap). STUDY_ACC_PROBE_MIDCHUNK=0 drops even that,
        # leaving the probe entirely free but with no ensemble overlap.
        self.acc_probe_midchunk = _env_bool("STUDY_ACC_PROBE_MIDCHUNK", True)
        self.acc_probe_async = _env_bool("STUDY_ACC_PROBE_ASYNC", True)
        self._acc_probe_counter = 0
        self._acc_worker = None
        self.last_acc_score = 0.0
        self.last_acc_components = self.acc_scorer.components_vector()
        self.last_acc_valid = False
        print(self.acc_scorer.describe())
        if self.acc_scorer.method == ACC_METHOD_ENSEMBLE:
            if self.acc_probe_async:
                self._acc_worker = AccProbeWorker(self.policy, self.postprocessor)
            chunk_size = int(getattr(self.policy.config, "chunk_size", 0) or 0)
            per_chunk = 1 + (1 if self.acc_probe_midchunk and chunk_size >= 4 else 0)
            print(f"[ACC] ensemble probe: {per_chunk} forward(s) per {chunk_size}-step "
                  f"chunk, async={self.acc_probe_async}, "
                  f"extra camera renders per chunk="
                  f"{1 if self.acc_probe_midchunk and chunk_size >= 4 else 0} "
                  f"(was {max(0, chunk_size - 1)}); control behaviour is unchanged.")
        self.is_paused = False
        self.play_start_wall = None
        self.play_start_step = 0

        self.reset_to_start(play=True)

        print(
            f"[INFO] Starting policy player at {self.config.policy_hz:g} Hz "
            f"with action_mode={self.config.action_mode!r}, "
            f"arm_action_mode={self.config.arm_action_mode!r}, "
            f"gripper_action_mode={self.config.gripper_action_mode!r}"
        )
        print(f"[INFO] Policy cameras: {self.camera_names}")

    @property
    def nframes(self):
        if self.config.max_steps > 0:
            return self.config.max_steps
        return max(self.reset_episode_len, self.frame_idx + 1)

    @property
    def sim_t(self):
        length = max(self.nframes, self.frame_idx + 2)
        return np.arange(length, dtype=np.float64) * self.policy_dt

    def _get_reset_episode_len(self) -> int:
        episode = np.load(self.config.reset_npz, allow_pickle=False)
        if "ctrl_sim" not in episode:
            raise KeyError(f"{self.config.reset_npz} missing key: ctrl_sim")
        return int(episode["ctrl_sim"].shape[0])

    def reload(self, npz_path: str, play: bool = True):
        self.npz_path = str(npz_path)
        self.config.reset_npz = Path(npz_path)
        self.reset_episode_len = self._get_reset_episode_len()
        self.reset_to_start(play=play)

    def reset_to_start(self, play=True):
        reset_from_npz(self.data, self.config.reset_npz)
        mujoco.mj_forward(self.model, self.data)
        for component in (self.policy, self.preprocessor, self.postprocessor):
            _reset_component(component)
        self.frame_idx = 0
        self.policy_query_count = 0
        self.last_target_ctrl = np.asarray(self.data.ctrl[: min(8, self.model.nu)], dtype=np.float32).copy()
        self.last_pred_action = np.zeros(8, dtype=np.float32)
        self.last_queried_policy = False
        self.last_loop_ms = 0.0
        self.last_observation_ms = 0.0
        self.last_observation_renders = 0
        self._reset_acc()
        self.play_start_wall = time.perf_counter()
        self.play_start_step = self.frame_idx
        self.is_paused = not play

    def jump_to_last(self, play=False):
        if self.config.max_steps > 0:
            while self.frame_idx < self.config.max_steps - 1:
                self._step_policy()
        self.is_paused = not play
        self.restart_clock_from_current_frame()

    def restart_clock_from_current_frame(self):
        self.play_start_wall = time.perf_counter()
        self.play_start_step = self.frame_idx

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        if not self.is_paused:
            self.restart_clock_from_current_frame()

    def resume_from_current_state(self, reset_policy_queue=True):
        if reset_policy_queue:
            for component in (self.policy, self.preprocessor, self.postprocessor):
                _reset_component(component)
        self.last_target_ctrl = np.asarray(
            self.data.ctrl[: min(8, self.model.nu)], dtype=np.float32
        ).copy()
        self.last_pred_action = np.zeros(8, dtype=np.float32)
        self.last_queried_policy = False
        self.last_loop_ms = 0.0
        self.last_observation_ms = 0.0
        self.last_observation_renders = 0
        if reset_policy_queue:
            # Only clear ACC history when the policy's own queue was cleared; resuming
            # mid-chunk should keep the running action history intact.
            self._reset_acc()
        self.is_paused = False
        self.restart_clock_from_current_frame()

    def _reset_acc(self):
        scorer = getattr(self, "acc_scorer", None)
        if scorer is None:
            return
        scorer.reset()
        self._acc_probe_counter = 0
        # Drop in-flight probes: they belong to the previous episode's step numbering and
        # would be scored against the new one.
        if getattr(self, "_acc_worker", None) is not None:
            self._acc_worker.clear()
        self.last_acc_score = 0.0
        self.last_acc_components = scorer.components_vector()
        self.last_acc_valid = False

    def _probe_is_due(self, observation) -> bool:
        """Should this step run an ACC ensemble probe?

        MEASURED COST LESSON (2026-07-28): the probe's expense was never the 3.98 ms GPU
        forward pass — it was `_build_observation()`, which renders 3 offscreen policy
        cameras and then swaps the GL context back (`_restore_window_context()` in a
        `finally`). In `queue` mode the policy itself builds an observation only 1 step in
        `chunk_size`, so probing every step multiplied the camera rendering ~10x. Across 9
        policy processes sharing one GL driver that pushed render time from ~15 ms to
        ~70 ms on EVERY session, selected or not.

        So the probe now runs ONLY on steps where an observation already exists, plus one
        mid-chunk step. Probing solely on query steps would put chunks exactly `chunk_size`
        apart — zero overlap, hence no ensemble at all — so one extra observation per chunk
        is the minimum that still yields overlapping predictions. That is 1 extra render
        burst per chunk instead of `chunk_size - 1`.
        """
        scorer = getattr(self, "acc_scorer", None)
        if scorer is None or scorer.method != ACC_METHOD_ENSEMBLE:
            return False
        if observation is not None:
            return True  # free: the policy already paid for this observation
        if not self.acc_probe_midchunk:
            return False
        chunk_size = int(getattr(self.policy.config, "chunk_size", 0) or 0)
        if chunk_size < 4:
            return False
        # One probe halfway through the chunk -> chunks land chunk_size/2 apart, giving
        # 2 overlapping predictions per timestep.
        return (self._acc_probe_counter % chunk_size) == (chunk_size // 2)

    def _submit_probe(self, observation, global_step: int) -> None:
        """Queue a measurement-only chunk prediction; never blocks the control step.

        `observation` is None only on the single mid-chunk probe step, where we deliberately
        pay for ONE observation build per chunk (the minimum that produces overlapping
        chunks). On query steps the policy's own observation is reused for free.
        """
        try:
            if observation is None:
                obs_t0 = time.perf_counter()
                observation = self._build_observation()
                self.last_observation_ms += (time.perf_counter() - obs_t0) * 1000.0
                self.last_observation_renders += len(self.camera_names)
            batch = self.preprocessor(observation)
        except Exception as exc:
            self._log_probe_error(exc)
            return
        if self._acc_worker is not None:
            self._acc_worker.submit(global_step, batch)
            return
        try:
            chunk = self.policy.predict_action_chunk(batch)
            self.acc_scorer.note_chunk(self.postprocessor(chunk), step=global_step)
        except Exception as exc:
            self._log_probe_error(exc)

    def _log_probe_error(self, exc) -> None:
        if not getattr(self, "_acc_probe_error_logged", False):
            print(f"[ACC][WARN] ensemble probe failed, falling back to components "
                  f"only: {exc}")
            self._acc_probe_error_logged = True

    def pause_at_current_state(self):
        self.is_paused = True

    def step_frame(self, delta: int):
        if delta < 0:
            print("[POLICY] Cannot step policy rollout backward. Press R to reset.")
            return
        self._step_policy()
        self.restart_clock_from_current_frame()

    def _render_policy_rgb(self, camera_name: str) -> np.ndarray:
        self.policy_renderer.disable_depth_rendering()
        self.policy_renderer.update_scene(self.data, camera=camera_name)
        return self.policy_renderer.render().copy()

    def swap_model(self, model):
        """Point the player at a different compiled model (an OOD model variant).

        Only ever called at an episode boundary, by `App._swap_model`. Every check runs
        BEFORE anything is mutated, and the replacement renderer is built before the old one
        is closed, so a rejected swap leaves the player exactly as it was.

        The checkpoint constrains the model: `nu >= 8`, its own camera names must exist, and
        `joint1..joint7` must land on the same qpos addresses -- `build_state()` reads
        `qvel[:7]` and `ctrl[7]` positionally, so a reordered model would feed the policy
        wrong numbers with no error at all.
        """
        if model is self.model:
            return
        if model.nu < 8:
            raise ValueError(f"variant has nu={model.nu}, policy mode needs >= 8")
        qpos_indices = get_qpos_indices_for_joints(model, PANDA_JOINT_NAMES)
        if list(qpos_indices) != list(self.qpos_indices):
            raise ValueError(
                f"variant moved the arm joints in qpos ({list(self.qpos_indices)} -> "
                f"{list(qpos_indices)}); build_state() would silently misread the state"
            )
        for camera_name in self.camera_names:
            validate_camera_exists(model, camera_name)
        n_substeps = max(1, int(round(self.policy_dt / model.opt.timestep)))
        if n_substeps != self.n_substeps:
            raise ValueError(
                f"variant changed opt.timestep ({self.model.opt.timestep} -> "
                f"{model.opt.timestep}); physics per policy step would change"
            )

        new_data = mujoco.MjData(model)
        # Build the replacement first: `mujoco.Renderer` makes its own GL context current,
        # which is why `_restore_window_context()` follows (same reason as _render_policy_rgb).
        new_renderer = mujoco.Renderer(model, height=224, width=224)
        old_renderer = self.policy_renderer
        self.policy_renderer = new_renderer
        try:
            old_renderer.close()
        except Exception as exc:
            print(f"[Variant][WARN] old policy renderer close failed: {exc}")
        self._restore_window_context()

        self.model = model
        self.data = new_data
        # frame_idx / queue / last_target_ctrl are intentionally left alone: the caller
        # applies a scene snapshot immediately afterwards, which resets all of them.

    def _restore_window_context(self):
        if self.context_current_fn is not None:
            self.context_current_fn()

    def _build_observation(self):
        phase = 0.0 if self.reset_episode_len <= 1 else min(self.frame_idx / (self.reset_episode_len - 1), 1.0)
        state = build_state(
            qpos_sim=np.asarray(self.data.qpos, dtype=np.float32),
            qvel_sim=np.asarray(self.data.qvel, dtype=np.float32),
            ctrl_sim_prev=np.asarray(self.data.ctrl, dtype=np.float32),
            qpos_indices=self.qpos_indices,
            state_dim=self.state_dim,
            phase=phase,
        )
        observation = {"observation.state": torch.from_numpy(state)}
        try:
            for camera_name in self.camera_names:
                observation[f"observation.images.{camera_name}"] = chw_float01_from_rgb(
                    self._render_policy_rgb(camera_name)
                )
        finally:
            self._restore_window_context()
        return observation

    def _step_policy(self):
        if self.config.max_steps > 0 and self.frame_idx >= self.config.max_steps:
            self.is_paused = True
            return False

        t0 = time.perf_counter()
        needs_observation = action_needs_observation(self.policy, self.config.action_mode)
        # `_build_observation` renders 3 offscreen cameras and swaps the GL context back;
        # in queue mode it runs once per chunk, not once per step (lesson 67). Timed
        # separately so its cost is never confused with the forward pass.
        _obs_t0 = time.perf_counter()
        observation = self._build_observation() if needs_observation else None
        if needs_observation:
            self.last_observation_ms = (time.perf_counter() - _obs_t0) * 1000.0
            self.last_observation_renders = len(self.camera_names)
        else:
            self.last_observation_ms = 0.0
            self.last_observation_renders = 0

        probe_chunk = [None]
        acc_step = self.acc_scorer._global_step if self.acc_scorer is not None else 0
        with torch.inference_mode():
            pred_action, queried_policy = select_processed_action(
                self.policy,
                self.preprocessor,
                self.postprocessor,
                observation,
                self.config.action_mode,
                # In 'replan' mode a chunk is produced anyway — reuse it so the ensemble
                # costs literally nothing there.
                chunk_sink=(lambda c: probe_chunk.__setitem__(0, c)),
            )
        # ACC temporal ensemble: measurement-only. predict_action_chunk does not touch the
        # policy's action queue (verified), so control is bit-identical with the probe on
        # or off. Submitted OUTSIDE the inference_mode block so the worker owns its own
        # context, and only when an observation already exists (see _probe_is_due).
        if probe_chunk[0] is None and self._probe_is_due(observation):
            self._submit_probe(observation, acc_step)
        if queried_policy:
            self.policy_query_count += 1
        if self.config.arm_delta_clip is not None:
            pred_action[:7] = np.clip(
                pred_action[:7],
                -self.config.arm_delta_clip,
                self.config.arm_delta_clip,
            )

        previous_ctrl = np.asarray(self.data.ctrl[:8], dtype=np.float32).copy()
        target_ctrl = previous_ctrl.copy()
        q_now = np.asarray(self.data.qpos[self.qpos_indices], dtype=np.float32)

        if self.config.arm_action_mode == "absolute":
            target_ctrl[:7] = pred_action[:7]
        elif self.config.arm_action_mode == "qpos_error":
            target_ctrl[:7] = q_now + pred_action[:7]
        elif self.config.arm_action_mode == "ctrl_delta":
            target_ctrl[:7] = previous_ctrl[:7] + pred_action[:7]
        else:
            raise ValueError(f"Unknown arm_action_mode: {self.config.arm_action_mode}")

        if self.config.gripper_action_mode == "absolute":
            target_ctrl[7] = pred_action[7]
        elif self.config.gripper_action_mode == "delta":
            target_ctrl[7] = previous_ctrl[7] + pred_action[7]
        else:
            raise ValueError(f"Unknown gripper_action_mode: {self.config.gripper_action_mode}")

        target_ctrl = clip_to_actuator_ranges(self.model, target_ctrl)
        self.data.ctrl[:8] = target_ctrl
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self.frame_idx += 1
        self.last_pred_action = pred_action
        self.last_target_ctrl = target_ctrl
        self.last_queried_policy = queried_policy
        self.last_loop_ms = (time.perf_counter() - t0) * 1000.0

        # ACC is scored AFTER the substeps so `q_now_after` reflects where the arm
        # actually ended up under the command just issued — that difference is the
        # tracking residual. Never allowed to break the control loop.
        try:
            q_now_after = np.asarray(self.data.qpos[self.qpos_indices], dtype=np.float32)
            # Register before update(): the scorer indexes chunks by the step they were
            # PREDICTED at, so async results carry their originating step with them.
            if probe_chunk[0] is not None:
                self.acc_scorer.note_chunk(probe_chunk[0], step=acc_step)
            if self._acc_worker is not None:
                for done_step, done_chunk in self._acc_worker.drain():
                    self.acc_scorer.note_chunk(done_chunk, step=done_step)
            self._acc_probe_counter += 1
            self.last_acc_score = self.acc_scorer.update(
                q_now_after, pred_action, queried_policy=bool(queried_policy)
            )
            self.last_acc_components = self.acc_scorer.components_vector()
            self.last_acc_valid = self.acc_scorer.last_valid
        except Exception as exc:
            if not getattr(self, "_acc_error_logged", False):
                print(f"[ACC][WARN] scoring failed, continuing with 0.0: {exc}")
                self._acc_error_logged = True
            self.last_acc_score = 0.0
            self.last_acc_valid = False

        if self.config.max_steps > 0 and self.frame_idx >= self.config.max_steps:
            self.is_paused = True
        return True

    def update(self):
        if self.is_paused:
            return False

        if self.config.realtime_factor <= 0:
            stepped = self._step_policy()
            self.restart_clock_from_current_frame()
            return bool(stepped)

        target_step = self.play_start_step + int(
            (time.perf_counter() - self.play_start_wall)
            / (self.policy_dt / self.config.realtime_factor)
        )
        if self.frame_idx <= target_step:
            return bool(self._step_policy())
        return False

    def get_gripper_width(self):
        if self.model.nu >= 8:
            return gripper_ctrl_to_width(self.model, self.data.ctrl[7])
        return None

    def get_arm_ctrl(self):
        if self.model.nu <= 0:
            return None
        return np.asarray(self.data.ctrl[: min(7, self.model.nu)], dtype=np.float64).copy()

    def get_frame_summary_text(self):
        lines = [
            "[POLICY SUMMARY]",
            f"  checkpoint : {Path(self.config.checkpoint).name}",
            f"  reset_npz  : {Path(self.config.reset_npz).name}",
            f"  step       : {self.frame_idx}",
            f"  sim_t      : {self.data.time:.6f}",
            f"  calls      : {self.policy_query_count}",
            f"  loop_ms    : {self.last_loop_ms:.1f}",
        ]
        grip = self.get_gripper_width()
        if grip is not None:
            lines.append(f"  grip      : {grip:.6f}")
        arm_ctrl = self.get_arm_ctrl()
        if arm_ctrl is not None:
            lines.append(
                f"  arm_ctrl  : {np.array2string(arm_ctrl, precision=4, suppress_small=True)}"
            )
        lines.append(
            f"  action    : {np.array2string(self.last_pred_action, precision=4, suppress_small=True)}"
        )
        return "\n".join(lines)

    def print_frame_summary(self):
        print(self.get_frame_summary_text())

    def get_overlay_text(self, extra_status: str = ""):
        state = "PAUSED" if self.is_paused else "POLICY"
        if extra_status:
            state = f"{state} | {extra_status}"
        grip = self.get_gripper_width()
        grip_txt = f"{grip:.4f}" if grip is not None else "n/a"
        info = (
            f"ACT policy: {Path(self.config.checkpoint).name}\n"
            f"reset: {Path(self.config.reset_npz).name}\n"
            f"step {self.frame_idx}/{self.nframes if self.config.max_steps > 0 else 'open'}\n"
            f"sim_t = {self.data.time:.3f}\n"
            f"policy calls = {self.policy_query_count}\n"
            f"grip = {grip_txt}\n"
            f"loop = {self.last_loop_ms:.1f} ms"
        )
        return info, state

    def close(self):
        try:
            if self._acc_worker is not None:
                self._acc_worker.stop()
                self._acc_worker = None
        except Exception:
            pass
        try:
            self.policy_renderer.close()
        except Exception:
            pass
