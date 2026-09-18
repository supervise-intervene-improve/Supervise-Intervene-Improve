import json
import os
import threading
import time
from pathlib import Path

import mujoco
import numpy as np

from playback.player import gripper_ctrl_to_width, gripper_width_to_ctrl

# FACTR leader-arm follow tuning. The MC_MUJOCO_* names are shared with the sim-MC
# path on purpose: both drive the same simulated gripper/arm, so a single set of
# knobs tunes them together.
FACTR_STANDALONE_GRIPPER_CLOSE_TIME_S = float(
    os.environ.get("MC_MUJOCO_GRIPPER_CLOSE_TIME_S", "0.85")
)
FACTR_STANDALONE_GRIPPER_OPEN_TIME_S = float(
    os.environ.get("MC_MUJOCO_GRIPPER_OPEN_TIME_S", "0.85")
)
# Gravity compensation for the SIMULATED arm during a hand-guided takeover. Applies to
# telekinesis and FACTR alike; sim-MC does the same thing inside its own step_sim.
#
# WHY IT IS NOT OPTIONAL IN PRACTICE: the Panda's actuators are position servos
# (kp 300-1000, forcerange +/-12 on the wrist joints) with no gravity term, so qpos
# settles a droop BELOW ctrl. Measured on the T-shape scene at the lab home pose:
# 0.0346 rad = 1.99 deg on joint2, zero with compensation. During a takeover ctrl is
# driven to the real arm's measured q -- which already corresponds to the sagged sim
# pose -- so without compensation the servo sags a SECOND time and the operator sees
# the arm drop at handoff. MC and FACTR never showed it because both already compensate.
#
# INTERVENE_INTERVENTION_ARM_GRAVITY_COMP is the current name; the two older names are
# still honoured so existing launch scripts keep working.
def _env_flag(*names, default="1"):
    for name in names:
        raw = os.environ.get(name)
        if raw is not None:
            return raw.strip().lower() not in {"0", "false", "no", "off"}
    return default.strip().lower() not in {"0", "false", "no", "off"}


INTERVENTION_ARM_GRAVITY_COMP = _env_flag(
    "INTERVENE_INTERVENTION_ARM_GRAVITY_COMP",
    "FACTR_STANDALONE_ARM_GRAVITY_COMP",
    "MC_MUJOCO_ARM_GRAVITY_COMP",
)
# Back-compat alias: read by nothing here any more, kept because external tooling and
# the LAB contract tests refer to it by name.
FACTR_STANDALONE_ARM_GRAVITY_COMP = INTERVENTION_ARM_GRAVITY_COMP

# Both spellings: the merged LAB scenes name the arm joints `joint1..joint7`, while the
# upstream Franka menagerie models use `panda_joint1..panda_joint7`. Looking up only the
# latter meant `_resolve_panda_dof_indices` ALWAYS hit its arange(7) fallback -- correct
# for today's scenes by luck (the arm is the first 7 DOFs) and silently wrong for any
# scene that orders joints differently.
PANDA_JOINT_NAMES = (
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
)
PANDA_JOINT_NAME_SETS = (
    PANDA_JOINT_NAMES,
    ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"),
)


# Control state, in the study's vocabulary, as an int8 column plus a labels scalar.
# The runtime's own phase strings are richer than the study needs, so they are folded
# here rather than renamed at the source -- no control path changes.
CONTROL_STATES = (
    "AUTONOMOUS",       # 0 - the policy is driving
    "PREPARING",        # 1 - requested / robot aligning to the takeover pose
    "READY",            # 2 - aligned, not yet handed over
    "HUMAN_CONTROL",    # 3 - the human has the arm
    "FINISHING",        # 4 - finish pressed, wrapping up the segment
    "RETURNING_HOME",   # 5 - policy already back in control, arm still homing
    "FAILED",           # 6 - takeover could not start
    "UNKNOWN",          # 7
)
_CONTROL_STATE_INDEX = {name: i for i, name in enumerate(CONTROL_STATES)}
_PHASE_TO_CONTROL_STATE = {
    "": "AUTONOMOUS", "idle": "AUTONOMOUS", "policy_resumed": "AUTONOMOUS",
    "requested": "PREPARING", "aligning": "PREPARING",
    "ready": "READY",
    "human_control": "HUMAN_CONTROL",
    "finishing": "FINISHING",
    "returning_home": "RETURNING_HOME",
    "failed": "FAILED",
}


def _env_float(name: str, default: float) -> float:
    """Read a float env knob, warning and falling back rather than raising."""
    raw = os.environ.get(name)
    if raw in (None, ""):
        return float(default)
    try:
        return float(raw)
    except ValueError:
        print(f"[REPLAN][WARN] {name}={raw!r} is not a number; using {default}.")
        return float(default)


def control_state_code(phase) -> int:
    """Map a runtime phase string to a CONTROL_STATES index. Never raises."""
    if phase is None:
        return _CONTROL_STATE_INDEX["AUTONOMOUS"]
    if isinstance(phase, (int, np.integer)):
        code = int(phase)
        return code if 0 <= code < len(CONTROL_STATES) else _CONTROL_STATE_INDEX["UNKNOWN"]
    name = _PHASE_TO_CONTROL_STATE.get(str(phase).strip().lower())
    if name is None:
        return _CONTROL_STATE_INDEX["UNKNOWN"]
    return _CONTROL_STATE_INDEX[name]


class TrajectoryRecorder:
    def __init__(
        self,
        save_path: Path,
        lab_id: str,
        log_hz: float,
        view_hz: float,
        camera_names=None,
        save_rgb: bool = False,
        save_depth: bool = False,
        rgb_width: int = 224,
        rgb_height: int = 224,
        metadata: dict | None = None,
        ee_site_id: int | None = None,
        ee_body_id: int | None = None,
    ):
        self.save_path = Path(save_path)
        self.lab_id = str(lab_id)
        self.log_hz = float(log_hz)
        self.view_hz = float(view_hz)
        self._stride = max(1, int(round(self.view_hz / self.log_hz)))
        self._k = 0
        self.camera_names = list(camera_names or [])
        self.save_rgb = bool(save_rgb and self.camera_names)
        self.save_depth = bool(save_depth and self.camera_names)
        self.rgb_width = int(rgb_width)
        self.rgb_height = int(rgb_height)
        self.metadata = dict(metadata or {})
        self._image_warning_shown = False
        self.wall_t = []
        self.sim_t = []
        self.q_real = []
        self.dq_real = []
        self.grip_real = []
        self.qpos_sim = []
        self.qvel_sim = []
        self.ctrl_sim = []
        self.intervention = []
        self.action_source = []
        self.intervention_id = []
        # Study columns. `ctrl_sim` records the resulting actuator target, which is not
        # the same as the policy's raw action once clipping (or a delta action mode) is
        # involved — so policy and human commands get their own explicit columns and are
        # NaN on frames where they do not apply. `acc` and `acc_components` come from
        # AccScorer; `queried_policy`/`chunk_index` say where in an action chunk a frame
        # sits, which is what makes ACC interpretable after the fact.
        self.policy_action = []
        self.human_action = []
        self.acc = []
        self.acc_components = []
        self.acc_valid = []
        self.queried_policy = []
        self.chunk_index = []
        self.acc_component_names = []
        # HRI study columns. `qpos_sim` already contains every object pose (the sidecar's
        # render_config.free_joint_qpos_map says which slice is which body), so objects
        # are NOT duplicated into per-object columns. What it does not contain is the
        # end-effector pose in world frame (forward kinematics), which cameras and reach
        # metrics both want, and the system state around each frame.
        self.ee_site_id = ee_site_id
        self.ee_body_id = ee_body_id
        self.ee_pos = []
        self.ee_quat = []
        self.gripper_cmd = []
        self.sim_step = []
        self.policy_step = []
        self.control_state = []
        self.cell_selected = []
        self.paused = []
        self.rgb_frames = {cam: [] for cam in self.camera_names}
        self.depth_frames = {cam: [] for cam in self.camera_names}

    def has_frames(self):
        return len(self.ctrl_sim) > 0

    def reset_stride(self):
        self._k = 0

    def record(
        self,
        now_wall,
        data,
        q_real,
        dq_real,
        gripper_width,
        intervention=False,
        intervention_id: int = 0,
        force=False,
        image_capture_fn=None,
        policy_action=None,
        acc=None,
        acc_components=None,
        acc_valid=False,
        queried_policy=False,
        chunk_index=None,
        sim_step=None,
        policy_step=None,
        control_state=None,
        cell_selected=False,
        paused=False,
    ):
        self._k += 1
        if not force and (self._k % self._stride) != 0:
            return False
        intervention = bool(intervention)
        self.wall_t.append(float(now_wall))
        self.sim_t.append(float(data.time))
        self.q_real.append(np.asarray(q_real, dtype=np.float64).copy())
        if dq_real is None:
            self.dq_real.append(np.full_like(q_real, np.nan, dtype=np.float64))
        else:
            self.dq_real.append(np.asarray(dq_real, dtype=np.float64).copy())
        self.grip_real.append(float(gripper_width))
        self.qpos_sim.append(np.asarray(data.qpos, dtype=np.float64).copy())
        self.qvel_sim.append(np.asarray(data.qvel, dtype=np.float64).copy())
        self.ctrl_sim.append(np.asarray(data.ctrl, dtype=np.float64).copy())
        self.intervention.append(intervention)
        self.action_source.append(1 if intervention else 0)
        self.intervention_id.append(int(intervention_id) if intervention else 0)

        # Policy vs human commands, kept in separate columns so a segment can be
        # analysed without re-deriving who was driving. NaN = "not applicable on this
        # frame", which is distinguishable from a genuine zero command.
        ctrl_now = np.asarray(data.ctrl, dtype=np.float64)
        n_act = min(8, ctrl_now.shape[0])
        nan8 = np.full(8, np.nan, dtype=np.float64)
        if intervention:
            human_act = nan8.copy()
            human_act[:n_act] = ctrl_now[:n_act]
            self.human_action.append(human_act)
            self.policy_action.append(nan8.copy())
        else:
            self.human_action.append(nan8.copy())
            if policy_action is None:
                self.policy_action.append(nan8.copy())
            else:
                pol = np.asarray(policy_action, dtype=np.float64).reshape(-1)
                buf = nan8.copy()
                buf[:min(8, pol.shape[0])] = pol[:8]
                self.policy_action.append(buf)

        self.acc.append(float(acc) if acc is not None else np.nan)
        self.acc_valid.append(bool(acc_valid))
        self.queried_policy.append(bool(queried_policy))
        self.chunk_index.append(int(chunk_index) if chunk_index is not None else -1)
        if acc_components is None:
            self.acc_components.append(None)   # widened to NaN in save()
        else:
            self.acc_components.append(
                np.asarray(acc_components, dtype=np.float32).reshape(-1)
            )

        # End-effector pose in world frame. Read from `data`, which mj_forward has
        # already updated for this step, so this is a lookup rather than a computation.
        ee_pos, ee_quat = self._read_ee_pose(data)
        self.ee_pos.append(ee_pos)
        self.ee_quat.append(ee_quat)
        # The gripper actuator target, split out of ctrl_sim because analysis asks
        # "when did they command a close" far more often than it asks about ctrl[7].
        self.gripper_cmd.append(
            float(ctrl_now[7]) if ctrl_now.shape[0] > 7 else np.nan
        )
        self.sim_step.append(int(sim_step) if sim_step is not None else -1)
        self.policy_step.append(int(policy_step) if policy_step is not None else -1)
        self.control_state.append(
            control_state_code(
                control_state if control_state is not None
                else ("human_control" if intervention else "policy_resumed")
            )
        )
        self.cell_selected.append(bool(cell_selected))
        self.paused.append(bool(paused))

        self._record_images(data, image_capture_fn)
        return True

    def _read_ee_pose(self, data):
        """(pos, quat) of the configured EE site/body, or NaN when unavailable.

        A site has no orientation quaternion in MjData (only `site_xmat`), so the
        orientation comes from the parent body when a site is used for position.
        """
        nan3 = np.full(3, np.nan, dtype=np.float64)
        nan4 = np.full(4, np.nan, dtype=np.float64)
        pos, quat = nan3, nan4
        try:
            if self.ee_site_id is not None and self.ee_site_id >= 0:
                pos = np.asarray(data.site_xpos[self.ee_site_id], dtype=np.float64).copy()
            elif self.ee_body_id is not None and self.ee_body_id >= 0:
                pos = np.asarray(data.xpos[self.ee_body_id], dtype=np.float64).copy()
            if self.ee_body_id is not None and self.ee_body_id >= 0:
                quat = np.asarray(data.xquat[self.ee_body_id], dtype=np.float64).copy()
        except Exception:
            return nan3, nan4
        return pos, quat

    def frame_count(self):
        return len(self.ctrl_sim)

    # Every per-frame list, in one place. truncate() is the cancel-path rollback: a
    # column missing from this tuple would be left longer than the rest and silently
    # corrupt the saved arrays, so keep it in sync when adding columns.
    _FRAME_LISTS = (
        "wall_t", "sim_t", "q_real", "dq_real", "grip_real",
        "qpos_sim", "qvel_sim", "ctrl_sim",
        "intervention", "action_source", "intervention_id",
        "policy_action", "human_action",
        "acc", "acc_components", "acc_valid", "queried_policy", "chunk_index",
        "ee_pos", "ee_quat", "gripper_cmd",
        "sim_step", "policy_step", "control_state", "cell_selected", "paused",
    )

    def truncate(self, length: int):
        length = max(0, int(length))
        for name in self._FRAME_LISTS:
            setattr(self, name, getattr(self, name)[:length])
        for cam in self.camera_names:
            if cam in self.rgb_frames:
                self.rgb_frames[cam] = self.rgb_frames[cam][:length]
            if cam in self.depth_frames:
                self.depth_frames[cam] = self.depth_frames[cam][:length]

    def _record_images(self, data, image_capture_fn):
        if not (self.save_rgb or self.save_depth):
            return

        rgb_by_cam = {}
        depth_by_cam = {}
        if image_capture_fn is not None:
            try:
                rgb_by_cam, depth_by_cam = image_capture_fn(data)
            except Exception as exc:
                if not self._image_warning_shown:
                    print(f"[WARN] Camera recording failed; writing blank images: {exc}")
                    self._image_warning_shown = True

        for cam in self.camera_names:
            if self.save_rgb:
                rgb = rgb_by_cam.get(cam)
                if rgb is None:
                    rgb = np.zeros((self.rgb_height, self.rgb_width, 3), dtype=np.uint8)
                self.rgb_frames[cam].append(np.asarray(rgb, dtype=np.uint8).copy())
            if self.save_depth:
                depth = depth_by_cam.get(cam)
                if depth is None:
                    depth = np.zeros((self.rgb_height, self.rgb_width), dtype=np.float32)
                self.depth_frames[cam].append(np.asarray(depth, dtype=np.float32).copy())

    def _stack_acc_components(self):
        """(N, k) float32. Frames recorded before ACC was available become NaN rows."""
        width = 0
        for entry in self.acc_components:
            if entry is not None:
                width = max(width, int(entry.shape[0]))
        n = len(self.acc_components)
        if width == 0:
            return np.zeros((n, 0), dtype=np.float32)
        out = np.full((n, width), np.nan, dtype=np.float32)
        for i, entry in enumerate(self.acc_components):
            if entry is not None:
                out[i, : entry.shape[0]] = entry[:width]
        return out

    def save(self):
        if not self.has_frames():
            raise RuntimeError("No trajectory frames recorded; nothing to save.")

        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        save_dict = {
            "wall_t": np.array(self.wall_t, dtype=np.float64),
            "sim_t": np.array(self.sim_t, dtype=np.float64),
            "q_real": np.stack(self.q_real).astype(np.float64),
            "dq_real": np.stack(self.dq_real).astype(np.float64),
            "grip_real": np.array(self.grip_real, dtype=np.float64),
            "qpos_sim": np.stack(self.qpos_sim).astype(np.float64),
            "qvel_sim": np.stack(self.qvel_sim).astype(np.float64),
            "ctrl_sim": np.stack(self.ctrl_sim).astype(np.float64),
            "intervention": np.array(self.intervention, dtype=np.bool_),
            "action_source": np.array(self.action_source, dtype=np.int8),
            "intervention_id": np.array(self.intervention_id, dtype=np.int32),
            "action_source_labels": np.array(["policy", "human_intervention"]),
            "lab_id": np.array(self.lab_id),
            "log_hz": np.array(self.log_hz, dtype=np.float64),
            "view_hz": np.array(self.view_hz, dtype=np.float64),
            "policy_action": np.stack(self.policy_action).astype(np.float64),
            "human_action": np.stack(self.human_action).astype(np.float64),
            "acc": np.array(self.acc, dtype=np.float64),
            "acc_valid": np.array(self.acc_valid, dtype=np.bool_),
            "queried_policy": np.array(self.queried_policy, dtype=np.bool_),
            "chunk_index": np.array(self.chunk_index, dtype=np.int32),
            "acc_components": self._stack_acc_components(),
            # Comma-joined scalar, not an array: utils/data_cleaning.py trims any array
            # whose first dimension equals the frame count, so a k-element name array
            # would be silently sliced by a k-frame episode. (The same latent hazard
            # already exists for action_source_labels.) The JSON sidecar carries the
            # list form for convenience.
            "acc_component_names": np.array(",".join(self.acc_component_names)),
            # HRI study columns.
            "ee_pos": np.stack(self.ee_pos).astype(np.float64),
            "ee_quat": np.stack(self.ee_quat).astype(np.float64),
            "gripper_cmd": np.array(self.gripper_cmd, dtype=np.float64),
            "sim_step": np.array(self.sim_step, dtype=np.int64),
            "policy_step": np.array(self.policy_step, dtype=np.int64),
            "control_state": np.array(self.control_state, dtype=np.int8),
            "cell_selected": np.array(self.cell_selected, dtype=np.bool_),
            "paused": np.array(self.paused, dtype=np.bool_),
            # Comma-joined scalar, not an array -- same data_cleaning.py trimming hazard
            # as acc_component_names above.
            "control_state_labels": np.array(",".join(CONTROL_STATES)),
        }

        # Study labels are written into the NPZ itself as well as the JSON sidecar, so
        # a stray .npz can always be traced back to its participant and condition even
        # if it gets separated from the rest of the session directory.
        # `mujoco_xml_path` / `mujoco_xml_sha256` are scene PROVENANCE, and they matter most
        # for state-only episodes: with no stored frames, the images are whatever
        # re-rendering this exact scene produces. A path alone cannot detect that the XML was
        # edited afterwards, so the hash travels with it. Previously both lived only in the
        # sidecar .json, leaving a separated .npz unreconstructible.
        for key in (
            "participant_id", "block", "interface", "task", "condition",
            "scenario_id", "episode_seed", "episode_number", "outcome",
            "fail_reason", "session_dir",
            "mujoco_xml_path", "mujoco_xml_sha256",
            # Per-episode OOD ground truth. `condition` is a block label; the OOD band
            # migrates between sessions, so only this says what this episode actually ran.
            "episode_ood",
            # Scalar strings, not arrays: data_cleaning.py trims any array whose first
            # dimension equals the frame count (see the acc_component_names note above).
            "episode_ood_scene",
            "episode_ood_scene_sha256",
            # WHICH COMPILED MODEL ran. For a model-level task (cups) an OOD episode uses a
            # variant of the same scene file, so mujoco_xml_sha256 no longer identifies the
            # geometry on its own. Scalars only -- the full descriptor and asset-hash JSON
            # blobs stay in the sidecar.
            "episode_model_variant_key",
            "episode_model_variant_source",
            "episode_model_variant_source_sha256",
            "episode_scene_closure_sha256",
            "episode_model_epoch",
            "episode_ood_pose_only",
            # HRI block identity: which participant, which condition, which cell. Same
            # rationale as the study labels above -- a stray .npz stays self-describing.
            "study", "condition_id", "supervision_interface", "controller_interface",
            "block_id", "cell_id", "task_display",
        ):
            if key in self.metadata and self.metadata[key] is not None:
                save_dict[key] = np.array(self.metadata[key])

        if self.save_rgb:
            for cam in self.camera_names:
                save_dict[f"rgb_{cam}"] = np.array(self.rgb_frames[cam], dtype=np.uint8)

        if self.save_depth:
            for cam in self.camera_names:
                save_dict[f"depth_{cam}"] = np.array(self.depth_frames[cam], dtype=np.float32)

        np.savez_compressed(self.save_path, **save_dict)

        meta = {
            **self.metadata,
            "lab_id": self.lab_id,
            "log_hz": self.log_hz,
            "view_hz": self.view_hz,
            "camera_names": self.camera_names,
            "rgb_width": self.rgb_width,
            "rgb_height": self.rgb_height,
            "save_rgb": self.save_rgb,
            "save_depth": self.save_depth,
            "action_source_labels": ["policy", "human_intervention"],
            "acc_component_names": list(self.acc_component_names),
            "control_state_labels": list(CONTROL_STATES),
            "n_frames": self.frame_count(),
        }
        # `render_config` travels as a JSON string (long blobs must not become NPZ
        # arrays), but the sidecar is where a human reads it -- expand it back.
        if isinstance(meta.get("render_config"), str) and meta["render_config"]:
            try:
                meta["render_config"] = json.loads(meta["render_config"])
            except json.JSONDecodeError:
                pass
        meta_path = self.save_path.with_suffix(".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[INFO] Saved trajectory to: {self.save_path}")
        print(f"[INFO] Saved metadata to: {meta_path}")


class InterventionTransitionError(RuntimeError):
    """A robot takeover failure with a stable reason for the VR status channel."""

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = str(reason)


class LiveReplanSession:
    def __init__(
        self,
        *,
        robot_adapter,
        save_path: str,
        mujoco_lab_id: str,
        mujoco_xml_path: str,
        q_hold: np.ndarray,
        finger_hold: float,
        grip_width_hold: float | None,
        qpos_seed: np.ndarray | None,
        qvel_seed: np.ndarray | None,
        view_hz: float = 60.0,
        log_hz: float = 60.0,
        alpha: float = 0.6,
        already_connected: bool = False,
        model=None,
        data=None,
        recorder: TrajectoryRecorder | None = None,
        image_capture_fn=None,
        intervention_id: int = 1,
        mc_mode: bool = False,
        sim_mc_driver=None,
        on_ready=None,
        on_released=None,
        on_tick=None,
        frame_status_provider=None,
    ):
        self.robot_adapter = robot_adapter
        self.robot = getattr(robot_adapter, "robot", None)
        # FACTR is discovered from the adapter rather than passed in, so nothing upstream
        # has to know which backend it built. `strict_alignment` is the important one: a
        # leader arm that has not reached the paused pose must never be handed control.
        self.factr_mode = bool(getattr(robot_adapter, "is_factr_adapter", False))
        self.strict_alignment = bool(
            getattr(robot_adapter, "requires_strict_alignment", False)
        )
        self.mc_mode = bool(mc_mode)
        self.sim_mc_driver = sim_mc_driver
        # Study instrumentation hooks. `on_ready` fires when the robot has finished
        # aligning to the takeover pose; `on_released` when control actually passes to
        # the human. Both are optional and must never break the takeover if they throw.
        self._on_ready = on_ready
        self._on_released = on_released
        # `on_tick` is pumped from inside the BLOCKING alignment steps so the caller can
        # keep publishing state and feeding its watchdog. Without it, every consumer
        # (desktop grid tiles, Quest point clouds) ages past its stale timeout and blanks
        # for the whole alignment. Same never-throw discipline as the two hooks above.
        self._on_tick = on_tick
        self._last_tick_log_wall = 0.0
        # Returns {"policy_step": int, "cell_selected": bool} for the study columns on
        # human-control frames. A provider rather than a snapshot because selection can
        # change during a takeover. Same never-throw discipline as the hooks above.
        self._frame_status_provider = frame_status_provider

        self.save_path = Path(save_path)
        self.mujoco_lab_id = str(mujoco_lab_id)
        self.mujoco_xml_path = str(mujoco_xml_path)
        self.view_hz = float(view_hz)
        self.log_hz = float(log_hz)
        self.alpha = float(alpha)
        self.dt = 1.0 / self.view_hz
        self.already_connected = bool(already_connected)
        self.image_capture_fn = image_capture_fn
        self.intervention_id = int(intervention_id)

        self.q_hold = np.asarray(q_hold, dtype=np.float64).reshape(7)
        self.finger_hold = float(np.clip(finger_hold, 0.0, 0.04))
        self.grip_width_hold = grip_width_hold

        if model is None:
            self.model = mujoco.MjModel.from_xml_path(self.mujoco_xml_path)
        else:
            self.model = model

        if data is None:
            self.data = mujoco.MjData(self.model)
        else:
            self.data = data

        if qpos_seed is not None:
            qpos_seed = np.asarray(qpos_seed, dtype=np.float64)
            if qpos_seed.shape != self.data.qpos.shape:
                raise RuntimeError(
                    f"qpos_seed shape {qpos_seed.shape} != data.qpos shape {self.data.qpos.shape}"
                )
            self.data.qpos[:] = qpos_seed

        if qvel_seed is not None:
            qvel_seed = np.asarray(qvel_seed, dtype=np.float64)
            if qvel_seed.shape != self.data.qvel.shape:
                raise RuntimeError(
                    f"qvel_seed shape {qvel_seed.shape} != data.qvel shape {self.data.qvel.shape}"
                )
            self.data.qvel[:] = qvel_seed

        self.data.ctrl[:7] = self.q_hold.copy()
        if self.model.nu >= 8:
            self.data.ctrl[7] = self.finger_hold
        mujoco.mj_forward(self.model, self.data)

        if recorder is None:
            self.recorder = TrajectoryRecorder(
                save_path=self.save_path,
                lab_id=self.mujoco_lab_id,
                log_hz=self.log_hz,
                view_hz=self.view_hz,
            )
        else:
            self.recorder = recorder

        self.started = False
        self.finished = False
        self.phase = "requested"

        # FACTR follow state. Resolved unconditionally (the DOF lookup falls back safely
        # on any model) so a non-FACTR session is still introspectable by the validator.
        # When the human actually got the arm; drives the follow-gain ramp for every
        # hand-guided path. `factr_takeover_started_wall` is the older FACTR-only name,
        # kept because the LAB contract tests and the validator reference it.
        self.takeover_started_wall: float | None = None
        self.factr_takeover_started_wall: float | None = None
        self.factr_gripper_ctrl_filtered: float | None = None
        self.factr_gripper_target_ctrl = self.finger_hold
        self.factr_gripper_close_time = max(1e-6, FACTR_STANDALONE_GRIPPER_CLOSE_TIME_S)
        self.factr_gripper_open_time = max(1e-6, FACTR_STANDALONE_GRIPPER_OPEN_TIME_S)
        self.factr_arm_dof_indices = self._resolve_panda_dof_indices()

        # --- Recording gate ---------------------------------------------------------
        # An intervention segment should contain the human's CORRECTION, not the dead air
        # between "you have the arm" and "the operator actually moved it". Frames are
        # therefore withheld until the simulated arm (or the gripper) is commanded away
        # from the takeover pose; from that instant on the gate LATCHES, so a correction
        # that pauses mid-way is never chopped into pieces.
        #
        # The signal is `ctrl`, not `qpos`: ctrl is the pose MuJoCo was commanded to, and
        # it leads qpos by a step, so no frame of the actual motion is lost. (Same reason
        # MujocoFrankaController.get_arm_qpos prefers it -- qpos lags and sags.)
        #
        # The baseline is seeded at handover and then EMA-relaxed while the gate is shut,
        # because the arm does move on its own for ~0.8 s after handover: the real arm
        # settles at its own impedance residual (up to ~0.025 rad, see _move_robot_to_q_hold)
        # and the follow-gain ramp walks the sim onto it. A fixed reference would trip on
        # that transient alone. A slow drift is absorbed by the baseline; a deliberate
        # motion outruns it -- at tau=0.4 s the baseline lags a drift by rate*tau, so the
        # ~0.03 rad/s settling contributes ~0.012 rad (under the 0.02 threshold) while a
        # slow-by-human-standards 0.3 rad/s trips it within ~0.07 s.
        self.record_on_motion = _env_flag("INTERVENE_RECORD_ON_MOTION")
        self.record_motion_rad = _env_float("INTERVENE_RECORD_MOTION_RAD", 0.02)
        self.record_motion_gripper = _env_float("INTERVENE_RECORD_MOTION_GRIPPER", 0.002)
        self.record_motion_baseline_tau_s = _env_float(
            "INTERVENE_RECORD_MOTION_BASELINE_TAU_S", 0.4
        )
        self.recording_open = not self.record_on_motion
        self.recording_opened_wall: float | None = None
        self.record_gate_frames_skipped = 0
        self._record_gate_baseline = None
        # Adopted on the first update() tick rather than at handover: the sim holds the
        # POLICY's ctrl[7] until then, while the follow loop writes clip(width/2) from the
        # real gripper -- and those two disagree by a constant on a soft gripper (the
        # reason ctrl[7] is deliberately not routed through gripper_width_to_ctrl). A
        # handover-time baseline would read that conversion offset as an operator action.
        self._record_gate_baseline_grip = None
        self._record_gate_last_wall = None

    def _get_robot_q(self):
        try:
            return np.asarray(self.robot_adapter.get_joint_positions(), dtype=np.float64)
        except Exception as exc:
            raise InterventionTransitionError(
                "state_read_failed", f"could not read robot joint state: {exc}"
            ) from exc

    def _start_async_gripper_restore(self):
        """Begin restoring the gripper to the paused width, off the arm's critical path.

        Returns the thread, or None when there is nothing to do / the async path is
        disabled, in which case `_join_async_gripper_restore` runs it synchronously and
        the behaviour is byte-for-byte the previous one.

        Skipped for FACTR: `align_to_mujoco` already staged the leader's gripper, and the
        leader's trigger -- not the sim -- is authoritative from there on.
        """
        if self.grip_width_hold is None or self.factr_mode:
            return None
        if not _env_flag("INTERVENE_TAKEOVER_ASYNC_GRIPPER"):
            return None

        def _restore():
            try:
                self.robot_adapter.restore_gripper_width(self.grip_width_hold)
            except Exception as exc:
                # Recorded, not raised: a thread that raises would die silently, and a
                # gripper that did not reach its target has never been fatal here.
                self._gripper_restore_error = exc

        self._gripper_restore_error = None
        thread = threading.Thread(
            target=_restore, name="ReplanGripperRestore", daemon=True
        )
        thread.start()
        return thread

    def _join_async_gripper_restore(self, thread):
        """Wait for the concurrent gripper restore, or do it synchronously if disabled."""
        if thread is None:
            if self.grip_width_hold is not None and not self.factr_mode:
                try:
                    self.robot_adapter.restore_gripper_width(self.grip_width_hold)
                except Exception as exc:
                    print(f"[REPLAN] Gripper restore warning: {exc}")
                finally:
                    self._tick("restoring gripper")
            return

        # Bounded: the restore has its own internal timeout, so a join that outlasts it
        # means the call is wedged. Better to hand over with an unstaged gripper than to
        # leave the operator waiting on a stuck device.
        budget = 3.0
        try:
            budget = float(os.environ.get("INTERVENE_TAKEOVER_GRIPPER_JOIN_S", "3.0"))
        except ValueError:
            pass
        t0 = time.time()
        while thread.is_alive() and (time.time() - t0) < budget:
            thread.join(timeout=0.05)
            self._tick("restoring gripper")
        if thread.is_alive():
            print(f"[REPLAN][WARN] Gripper restore still running after {budget:.1f}s; "
                  "continuing into human control without waiting.")
        error = getattr(self, "_gripper_restore_error", None)
        if error is not None:
            print(f"[REPLAN] Gripper restore warning: {error}")
        self._tick("restoring gripper")

    def _alignment_duration_s(self, q_start, q_goal):
        """How long the real arm should take to reach the takeover pose.

        The adapters' default was a FLAT 4 s (`INTERVENE_REPLAN_GOTO_SECONDS`)
        regardless of distance, so a 3-degree alignment took exactly as long as a
        90-degree one -- which is what made a takeover feel slow to start, since most
        alignments are short. This scales the duration with the distance actually
        travelled instead, bounded at both ends.

        The motion profile in `record_robot_adapter._go_to_joint_positions_over_time`
        is a smoothstep, `q = q_start + u^2(3-2u)*delta`, whose peak velocity is
        `1.5*delta/T`. Solving for T at a chosen peak speed gives the formula below.

        SAFETY: the default peak of 1.0 rad/s is barely above what this arm already did.
        The old flat 4 s produced 0.94 rad/s on a 2.5 rad move and 1.12 rad/s on a 3 rad
        one, so 1.0 sits inside the range the system has always used -- the change is
        that SHORT moves no longer crawl. It is ~46% of the Franka's 2.175 rad/s joint
        limit. Tune with INTERVENE_TAKEOVER_ALIGN_SPEED_RAD_S.

        An explicit INTERVENE_REPLAN_GOTO_SECONDS still wins outright, so existing
        launch scripts (e.g. run_multi.sh, which sets 6) are unaffected.
        """
        explicit = os.environ.get("INTERVENE_REPLAN_GOTO_SECONDS")
        if explicit not in (None, ""):
            try:
                return float(explicit)
            except ValueError:
                print(f"[REPLAN][WARN] INTERVENE_REPLAN_GOTO_SECONDS={explicit!r} is "
                      "not a number; using the distance-based duration.")

        def _f(name, default):
            raw = os.environ.get(name)
            if raw in (None, ""):
                return float(default)
            try:
                return float(raw)
            except ValueError:
                print(f"[REPLAN][WARN] {name}={raw!r} is not a number; using {default}.")
                return float(default)

        peak = _f("INTERVENE_TAKEOVER_ALIGN_SPEED_RAD_S", 0.5)
        floor = _f("INTERVENE_TAKEOVER_ALIGN_MIN_SECONDS", 0.3)
        cap = _f("INTERVENE_TAKEOVER_ALIGN_MAX_SECONDS", 6.0)
        if peak <= 0.0:
            return cap

        try:
            travel = float(np.max(np.abs(np.asarray(q_goal, dtype=np.float64)
                                         - np.asarray(q_start, dtype=np.float64))))
        except Exception:
            return cap
        if not np.isfinite(travel):
            return cap

        duration = 1.5 * travel / peak
        duration = float(min(max(duration, floor), cap))
        print(f"[REPLAN] Aligning over {duration:.2f}s "
              f"(travel={travel:.3f} rad, peak<={peak:.2f} rad/s).")
        return duration

    def _move_robot_to_q_hold(self, tol=0.03, settle_timeout=3.0):
        q_target = self.q_hold.copy()

        # FACTR aligns inside the service process (it owns the serial device and runs the
        # bounded move under gravity/friction compensation), so there is no polling loop
        # here. Failures are converted to InterventionTransitionError with reasons from
        # app.py::_transition_reason's allow-list — anything outside it collapses to the
        # generic "transition_failed" and the operator loses the diagnosis.
        if hasattr(self.robot_adapter, "align_to_mujoco"):
            tolerance = float(os.environ.get("FACTR_POSITION_TOLERANCE", str(tol)))
            timeout = float(
                os.environ.get("FACTR_ALIGNMENT_TIMEOUT", str(settle_timeout))
            )
            max_velocity = float(os.environ.get("FACTR_MAX_VELOCITY", "0.5"))
            print(
                "[FACTR] State ALIGNING: policy is paused; moving FACTR to the "
                "current MuJoCo pose."
            )
            # align_to_mujoco is ONE blocking RPC (up to FACTR_ALIGNMENT_TIMEOUT, 18 s by
            # default) with no progress callback, so only its edges can be pumped. The
            # stall inside it remains; fixing that needs a poll/progress RPC on the FACTR
            # service side.
            self._tick("aligning FACTR")
            try:
                reached = self.robot_adapter.align_to_mujoco(
                    q_target,
                    gripper_width=self.grip_width_hold,
                    tolerance=tolerance,
                    timeout=timeout,
                    max_velocity=max_velocity,
                )
            except Exception as exc:
                # A dead/busy service surfaces here as FactrRpcError; untyped it would
                # reach the operator as a bare "transition_failed".
                raise InterventionTransitionError(
                    "robot_unreachable", f"could not command FACTR alignment: {exc}"
                ) from exc
            finally:
                self._tick("aligning FACTR")
            if not reached:
                raise InterventionTransitionError(
                    "alignment_command_failed",
                    "FACTR alignment timed out before reaching the MuJoCo target "
                    f"within {tolerance} rad",
                )
            print("[FACTR] Target pose reached. Preparing leader handoff.")
            return True

        q_robot = self._get_robot_q()
        err_norm = float(np.linalg.norm(q_target - q_robot))

        print(f"[REPLAN] Move robot to paused MuJoCo pose. Initial err={err_norm:.4f}")

        if err_norm < tol:
            print("[REPLAN] Robot already close to paused pose.")
            return True

        align_seconds = self._alignment_duration_s(q_robot, q_target)
        try:
            if hasattr(self.robot_adapter, "go_to_joint_positions"):
                self.robot_adapter.go_to_joint_positions(
                    q_target, duration_s=align_seconds
                )
            elif self.robot is not None:
                self.robot.robot_arm.go_to_within_limits(q_target)
            else:
                self.robot_adapter.send_joint_positions(q_target)
        except Exception as exc:
            raise InterventionTransitionError(
                "alignment_command_failed", f"could not command robot alignment: {exc}"
            ) from exc

        # Wait for the arm to settle -- but stop as soon as it STOPS IMPROVING, not only
        # when it reaches `tol`.
        #
        # WHY: the real Franka's impedance controller has its own steady-state error. It
        # reaches ~0.025 rad and stays there. A tolerance below that is unreachable, so
        # the loop burned the entire `settle_timeout` on every single takeover waiting
        # for a number the hardware cannot produce, then gave up at the error it had
        # already reached within the first second. Measured on p3: arm motion done at
        # 0.5 s, error 0.0248 rad, then 4.5 s of polling for nothing.
        #
        # Plateau detection makes the wait proportional to how long the arm actually
        # takes, and `tol` goes back to being an early exit for when it can be met.
        # NOT applied to the strict (FACTR) path: there, failing to reach tolerance must
        # remain a real failure rather than an early success.
        # 0.25 s of "no improvement" is enough to call it settled: the move that got us
        # here is BLOCKING and ends its smoothstep at zero commanded velocity, so the arm
        # is already at rest when this loop starts. The patience only has to outlast
        # sensor noise, not a deceleration.
        patience = _env_float("INTERVENE_TAKEOVER_ALIGN_PATIENCE_S", 0.25)
        min_gain = _env_float("INTERVENE_TAKEOVER_ALIGN_MIN_IMPROVEMENT_RAD", 0.002)

        t0 = time.time()
        best_err = float("inf")
        best_wall = t0
        err_norm = float("inf")
        while time.time() - t0 < settle_timeout:
            # Heartbeat every poll: this loop is the multi-second stall that used to
            # blank every consumer while the arm caught up.
            self._tick("aligning robot", t0=t0)
            q_robot = self._get_robot_q()
            err_norm = float(np.linalg.norm(q_target - q_robot))
            if err_norm < tol:
                print(f"[REPLAN] Robot aligned to paused pose (err={err_norm:.4f}) "
                      f"in {time.time() - t0:.2f}s.")
                return True

            now = time.time()
            if err_norm < best_err - min_gain:
                best_err = err_norm
                best_wall = now
            elif not self.strict_alignment and (now - best_wall) >= patience > 0.0:
                # Converged as far as this arm goes. Treat it as aligned: the residual
                # is handed to the takeover ramp, which is what absorbs it either way.
                print(f"[REPLAN] Robot settled at err={err_norm:.4f} rad "
                      f"(> tol {tol:.4f}) after {now - t0:.2f}s; "
                      "no further improvement, continuing.")
                return True
            time.sleep(self.dt)

        print(f"[REPLAN] Alignment timeout; continuing with err={err_norm:.4f}.")
        return False

    def _notify(self, callback, **fields):
        if callback is None:
            return
        try:
            callback(**fields)
        except Exception as exc:
            print(f"[REPLAN][WARN] study hook failed: {exc}")

    def _sim_step(self) -> int:
        """Simulation step derived from sim time, matching App._step_counters.

        getattr-free but exception-guarded: several LAB contract tests build this class
        via __new__ and never run __init__, so `self.model` may be absent.
        """
        try:
            timestep = float(self.model.opt.timestep)
            if timestep > 0.0:
                return int(round(float(self.data.time) / timestep))
        except Exception:
            pass
        return -1

    def _frame_status(self) -> dict:
        """{"policy_step", "cell_selected"} for the study columns. Never raises."""
        provider = getattr(self, "_frame_status_provider", None)
        if provider is None:
            return {"policy_step": -1, "cell_selected": False}
        try:
            got = provider() or {}
            return {
                "policy_step": int(got.get("policy_step", -1)),
                "cell_selected": bool(got.get("cell_selected", False)),
            }
        except Exception:
            return {"policy_step": -1, "cell_selected": False}

    def _gate_signal(self):
        """(arm ctrl (7,), gripper ctrl) as the recording gate sees them, or None.

        Exception-guarded rather than getattr-checked: several LAB contract tests build
        this class via __new__, and a gate that raises would be a gate that can abort a
        live takeover.
        """
        try:
            arm = np.asarray(self.data.ctrl[:7], dtype=np.float64).copy()
            grip = float(self.data.ctrl[7]) if self.model.nu >= 8 else 0.0
        except Exception:
            return None
        return arm, grip

    def _arm_recording_gate(self):
        """Seed the motion gate's baseline at handover. Never raises.

        Called at the moment the human gets the arm, so the baseline is the takeover pose
        itself -- which is what lets the FIRST commanded motion open the gate instead of
        being spent capturing a reference.
        """
        # getattr with a True default: several LAB contract tests build this class via
        # __new__ and never run __init__, and a session with no gate configured must
        # behave exactly as it did before the gate existed -- record everything.
        if getattr(self, "recording_open", True):
            return
        signal = self._gate_signal()
        if signal is None:
            # No readable sim state: record everything, exactly as before this gate.
            self.recording_open = True
            return
        self._record_gate_baseline = signal[0]
        self._record_gate_baseline_grip = None   # first tick, see __init__
        self._record_gate_last_wall = time.time()

    def _recording_gate_open(self) -> bool:
        """True once the operator has actually moved the simulated robot.

        Called once per update() tick immediately before the frame would be recorded.
        Latches: every tick after the first detected motion records.
        """
        # See _arm_recording_gate for why the default is True.
        if getattr(self, "recording_open", True):
            return True

        signal = self._gate_signal()
        if signal is None:
            self.recording_open = True
            return True
        arm, grip = signal

        if self._record_gate_baseline is None:
            # start() normally seeds this; a session driven without it starts here and
            # spends one frame on the reference.
            self._record_gate_baseline = arm
            self._record_gate_baseline_grip = grip
            self._record_gate_last_wall = time.time()
            self.record_gate_frames_skipped += 1
            return False

        if self._record_gate_baseline_grip is None:
            self._record_gate_baseline_grip = grip

        moved = float(np.max(np.abs(arm - self._record_gate_baseline)))
        moved_grip = abs(grip - self._record_gate_baseline_grip)
        if moved >= self.record_motion_rad or moved_grip >= self.record_motion_gripper:
            now = time.time()
            self.recording_open = True
            self.recording_opened_wall = now
            started = self.takeover_started_wall
            since = now - (now if started is None else started)
            print(
                f"[REPLAN] Motion detected {since:.2f}s after takeover "
                f"(arm {moved:.4f} rad, gripper {moved_grip:.4f}); recording "
                f"intervention {self.intervention_id} from here "
                f"({self.record_gate_frames_skipped} idle frame(s) not recorded)."
            )
            return True

        # Absorb the handover settling transient so it cannot accumulate into a trip.
        now = time.time()
        dt = max(0.0, now - (self._record_gate_last_wall or now))
        self._record_gate_last_wall = now
        tau = self.record_motion_baseline_tau_s
        if tau > 0.0 and dt > 0.0:
            beta = float(np.clip(dt / tau, 0.0, 1.0))
            self._record_gate_baseline += beta * (arm - self._record_gate_baseline)
            self._record_gate_baseline_grip += beta * (
                grip - self._record_gate_baseline_grip
            )
        self.record_gate_frames_skipped += 1
        return False

    def _tick(self, what: str = "aligning", *, t0: float | None = None):
        """Pump the caller's heartbeat during a blocking transition step.

        getattr, not a direct read: several LAB contract tests build this class via
        __new__ and never run __init__.
        """
        callback = getattr(self, "_on_tick", None)
        if callback is None:
            return
        try:
            callback()
        except Exception as exc:
            print(f"[REPLAN][WARN] transition heartbeat failed: {exc}")
        # ~1 Hz progress line so a long alignment is still visible in the log even though
        # the heartbeat keeps the main-loop watchdog quiet for its duration.
        now = time.time()
        if now - getattr(self, "_last_tick_log_wall", 0.0) >= 1.0:
            self._last_tick_log_wall = now
            elapsed = "" if t0 is None else f" t={now - t0:.1f}s"
            print(f"[REPLAN] {what}...{elapsed}")

    def start(self):
        if self.started:
            return

        self.phase = "aligning"

        if self.mc_mode and self.sim_mc_driver is not None:
            # Sim-native motion-controller mode: no real robot at all. The arm is
            # already at the paused pose (same live `data` the sim_mc_driver's IK
            # controller is bound to) — skip robot connect/align/gripper-restore/
            # switch_control_mode entirely.
            # resync (not bare reset_clutch_state): the driver is cached for the whole
            # session, so the gripper target and clutch offsets must be re-seeded from the
            # LIVE sim state at every takeover or the hand opens and drops the object.
            self.sim_mc_driver.resync_from_live_state()
            # Non-blocking. Quest discovery is a 20 s UDP recv loop; running it here froze
            # the main loop, blanked every grid tile, and then FAILED the whole takeover
            # when no headset was broadcasting (the keyboard-only MC_SIM + START_VR=0
            # case). The takeover no longer depends on it: the arm holds its pose and the
            # controller attaches from tick() if and when a headset shows up.
            try:
                connected = bool(self.sim_mc_driver.ensure_connected())
            except Exception as exc:
                connected = False
                print(f"[REPLAN][MC-SIM][WARN] Quest discovery could not be started: {exc}")
            # Sim-MC has no alignment phase — the arm is already at the paused pose —
            # so READY and RELEASED coincide with entering human control.
            self._notify(self._on_ready, path="sim_mc", aligned=True)
            self._notify(self._on_released, path="sim_mc", control_mode="sim_mc_ik")
            self.started = True
            self.phase = "human_control"
            self._arm_recording_gate()
            print("[REPLAN][MC-SIM] Sim-native motion-controller mode: driving simulated "
                  "arm via local IK, no real robot.")
            print("[REPLAN][MC-SIM] " + (
                "Quest attached." if connected
                else "Waiting for Quest in the background; arm holds pose."
            ))
            # A is the GRIPPER button here (mq3_mc_mujoco.GRIPPER_CONTROL_BUTTON), not a
            # handoff gate -- sim-MC has no handoff to confirm, the arm is already the
            # operator's. Naming A as the way to take over sent operators looking for a
            # confirmation that does not exist while the clutch sat unused.
            print("[REPLAN][MC-SIM] Hold the RIGHT index trigger to move the arm "
                  "(RIGHT grip = rotate, RIGHT A = gripper).")
            return

        # Phase timing. The takeover latency has now been chased through four separate
        # causes, every one of which was first misattributed by reading the launcher
        # console instead of this log. One summary line at the end of start() ends the
        # guessing: whatever is slowest next time names itself.
        phase_t0 = time.time()
        phase_marks = {}

        def _mark(name, since):
            phase_marks[name] = time.time() - since
            return time.time()

        if not self.already_connected:
            self._tick("connecting to robot")
            try:
                self.robot_adapter.connect(already_connected=False)
            except Exception as exc:
                raise InterventionTransitionError(
                    "robot_unreachable", f"could not connect to robot: {exc}"
                ) from exc
            finally:
                self._tick("connecting to robot")
        _phase_cursor = _mark("connect", phase_t0)

        # Stage the gripper CONCURRENTLY with the arm move. `robot_gripper` is a separate
        # polymetis client from `robot_arm`, so the two commands do not contend; running
        # them in series just added the gripper's settle time (up to `timeout`, 1.5 s) on
        # top of the arm's travel for no reason. Started before the arm move and joined
        # where the blocking call used to sit, so the READY/RELEASE phase boundaries the
        # study records keep their exact previous meaning.
        gripper_thread = self._start_async_gripper_restore()

        # First move arm to paused MuJoCo pose
        align_ok = True
        align_t0 = time.time()
        if self.strict_alignment:
            # FACTR: a failed or timed-out alignment must NEVER fall through to takeover.
            # Deliberately not wrapped — the typed error propagates to trigger_replan's
            # handler so _fail_replan_start aborts cleanly and the arm is never released.
            align_ok = self._move_robot_to_q_hold(
                tol=float(os.environ.get("FACTR_POSITION_TOLERANCE", "0.04")),
                settle_timeout=float(os.environ.get("FACTR_ALIGNMENT_TIMEOUT", "18.0")),
            )
        else:
            # Alignment error accepted here is error the SIM travels once the blend
            # starts tracking the real arm, so tighter is visually better -- BUT it must
            # stay REACHABLE. Measured on p3: the Franka's impedance controller settles
            # at ~0.025 rad and stops improving, so a tolerance below that is a number
            # the hardware cannot produce and the loop simply burned the whole timeout on
            # every takeover before giving up at the error it already had. The plateau
            # detector in _move_robot_to_q_hold is what bounds the wait now; this is an
            # early exit for arms that CAN meet it, not the gate.
            align_ok = self._move_robot_to_q_hold(
                tol=_env_float("INTERVENE_TAKEOVER_ALIGN_TOLERANCE", 0.02),
                settle_timeout=_env_float("INTERVENE_TAKEOVER_ALIGN_TIMEOUT", 3.0),
            )
            if not align_ok:
                print(
                    "[REPLAN][WARN] Robot remained responsive but did not reach the strict "
                    "alignment tolerance before timeout; continuing into human control. "
                    "The sim arm will settle onto the real arm's actual pose over the "
                    "takeover ramp."
                )
        _phase_cursor = _mark("align", _phase_cursor)

        # Phase 3 — READY: the arm has reached (or given up reaching) the takeover pose.
        self._notify(self._on_ready, path="factr" if self.factr_mode else "robot",
                     aligned=align_ok, align_seconds=time.time() - align_t0)

        # Collect the gripper restore started above. Usually already finished -- the arm
        # move is the longer of the two -- so this is normally instant.
        self._join_async_gripper_restore(gripper_thread)
        _phase_cursor = _mark("gripper_join", _phase_cursor)
        if self.mc_mode:
            # Motion-controller mode: do NOT enter HUMAN_CONTROL (freedrive). The external
            # mq3_mc.py process drives the arm via CARTESIAN_IMPEDANCE, streaming ee_pos_desired /
            # ee_quat_desired to the active server policy. Hold here with CARTESIAN_IMPEDANCE (NOT a
            # joint policy) so the running policy exposes those exact params — otherwise mq3_mc's
            # update_current_policy fails with `KeyError: ee_pos_desired` against a HYBRID_JOINT
            # policy. Cartesian hold also holds the arm firm at the aligned pose (no freedrive)
            # until mq3_mc's own cartesian policy takes over on Right-A.
            hold_mode = os.environ.get(
                "INTERVENE_MC_HOLD_MODE",
                "CARTESIAN_IMPEDANCE_CONTROL",
            )
            print(f"[REPLAN][MC] Motion-controller mode: arm held ({hold_mode}) for external controller.")
            try:
                if hasattr(self.robot_adapter, "switch_control_mode"):
                    self.robot_adapter.switch_control_mode(hold_mode)
                elif self.robot is not None:
                    import record as record_module

                    self.robot.connect(getattr(record_module.ControlType, hold_mode))
            except Exception as e:
                raise InterventionTransitionError(
                    "mode_switch_failed", f"could not set MC hold mode {hold_mode}: {e}"
                ) from e
            finally:
                self._tick("switching control mode")
            # Phase 4 — control passes to the external motion controller.
            self._notify(self._on_released, path="mc", control_mode=hold_mode)
        else:
            # Telekinesis, and FACTR: both switch to HUMAN_CONTROL. For FACTR that maps
            # onto the service's begin_takeover RPC, which enables the leader's
            # gravity-compensated teleop hold.
            print("[REPLAN] Switching to HUMAN_CONTROL...")
            try:
                if hasattr(self.robot_adapter, "switch_control_mode"):
                    self.robot_adapter.switch_control_mode("HUMAN_CONTROL")
                elif self.robot is not None:
                    import record as record_module

                    self.robot.connect(record_module.ControlType.HUMAN_CONTROL)
            except Exception as e:
                raise InterventionTransitionError(
                    "mode_switch_failed", f"could not switch to HUMAN_CONTROL: {e}"
                ) from e
            finally:
                self._tick("switching control mode")
            # Phase 4 — the arm is now free for hand-guiding: the human has control.
            self._notify(self._on_released,
                         path="factr" if self.factr_mode else "telekinesis",
                         control_mode="HUMAN_CONTROL")

        self.started = True

        # Seed the follow loop from the aligned pose so the first update() blends from
        # where the arm actually is, not from a stale ctrl target. Placed after the
        # release notify so the ramp clock measures real human control time.
        #
        # This used to be FACTR-only. On telekinesis ctrl still held the POLICY's last
        # commanded target (which sits one gravity droop above the sagged qpos), so the
        # first blend step yanked it down by alpha*droop -- the visible drop at handoff.
        # Seeding ctrl = q_hold = qpos, with gravity compensation now on, means the arm
        # holds exactly where it already is: zero motion at the handoff instant.
        # Skipped for mc_mode (real MC): the external mq3_mc process owns ctrl there.
        #
        # Wrapped, and guarded on `data` actually existing: this is a comfort step, and
        # it must never be the reason a takeover aborts with the operator already
        # reaching for the arm. (Several LAB contract tests also drive start() on an
        # object built with __new__ that has no model/data at all.)
        if not self.mc_mode and getattr(self, "data", None) is not None:
            try:
                self.data.ctrl[:7] = self.q_hold.copy()
                if self.data.qvel.shape[0] >= 7:
                    self.data.qvel[:7] = 0.0
                self._apply_arm_gravity_compensation()
                mujoco.mj_forward(self.model, self.data)
            except Exception as exc:
                print(f"[REPLAN][WARN] could not seed the sim arm at handoff: {exc}")

        # Starts the follow-gain ramp (see _follow_gains). Set for every hand-guided
        # path, not just FACTR: telekinesis has the same handoff transient, caused by
        # the alignment residual the robot was allowed to keep.
        self.takeover_started_wall = time.time()
        # Seeded here, on the pose the operator was handed, so the first commanded motion
        # is what opens the recording gate (see _recording_gate_open).
        self._arm_recording_gate()

        if self.factr_mode:
            self.factr_takeover_started_wall = self.takeover_started_wall
            if self.model.nu >= 8:
                self.factr_gripper_ctrl_filtered = float(self.data.ctrl[7])
                self.factr_gripper_target_ctrl = float(self.data.ctrl[7])
            else:
                self.factr_gripper_ctrl_filtered = None
                self.factr_gripper_target_ctrl = self.finger_hold

        _mark("handover", _phase_cursor)
        self.phase = "human_control"
        print("[REPLAN] Takeover ready in {:.2f}s ({}).".format(
            time.time() - phase_t0,
            " ".join(f"{k}={v:.2f}s" for k, v in phase_marks.items()),
        ))
        print("[REPLAN] Live replanning started in current window.")
        if self.mc_mode:
            print("[REPLAN][MC] Press Right-A in VR to hand the arm to the motion controller.")
        elif self.factr_mode:
            print("[FACTR] Target pose reached. You may take over now.")
            print("[REPLAN] Guide FACTR. Press the intervention button again to finish.")
        else:
            print("[REPLAN] Guide the robot. Press Enter in the app window to finish.")

    def update(self):
        if self.finished:
            return False

        if not self.started:
            self.start()

        t0 = time.time()

        if self.sim_mc_driver is not None:
            # Sim-native motion-controller mode: drive the live arm via local IK
            # from Quest input, then step physics (this also ramps the gripper
            # and applies gravity compensation — both live inside step_sim()).
            self.sim_mc_driver.tick()
            self.sim_mc_driver.controller.step_sim(self.dt)

            dof_indices = self.sim_mc_driver.controller.dof_indices
            q_real = self.sim_mc_driver.controller.get_arm_qpos()
            dq_real = np.asarray(self.data.qvel[dof_indices], dtype=np.float64).copy()

            gripper_width = 0.0
            if self.model.nu >= 8:
                gripper_width = gripper_ctrl_to_width(self.model, float(self.data.ctrl[7])) or 0.0

            # Withheld until the operator actually moves the arm; see _recording_gate_open.
            if self._recording_gate_open():
                _status = self._frame_status()
                self.recorder.record(
                    now_wall=time.time(),
                    data=self.data,
                    q_real=q_real,
                    dq_real=dq_real,
                    gripper_width=gripper_width,
                    intervention=True,
                    intervention_id=self.intervention_id,
                    image_capture_fn=self.image_capture_fn,
                    # Human frames carry the same system-state columns as autonomous
                    # ones, so a segment can be analysed without joining against
                    # events.jsonl. `policy_step` is deliberately the FROZEN frame_idx:
                    # the policy takes no action while the human has the arm, and that is
                    # exactly what makes "did ACT act after RELEASE?" answerable from the
                    # trajectory alone.
                    sim_step=self._sim_step(),
                    policy_step=_status["policy_step"],
                    control_state=self.phase,
                    cell_selected=_status["cell_selected"],
                    paused=True,
                )
        else:
            if hasattr(self.robot_adapter, "get_joint_state"):
                q_real, dq_real = self.robot_adapter.get_joint_state()
            else:
                state = self.robot.robot_arm.get_state()
                q_real = state.joint_pos.detach().cpu().numpy().astype(np.float64)
                dq_real = getattr(state, "joint_vel", None)
                if dq_real is not None:
                    dq_real = dq_real.detach().cpu().numpy().astype(np.float64)

            if self.factr_mode:
                # FACTR only. Telekinesis legitimately reports dq_real=None (the recorder
                # NaN-fills it), so this must never run for the polymetis paths.
                q_real = np.asarray(q_real, dtype=np.float64)
                dq_real = np.asarray(dq_real, dtype=np.float64)
                if q_real.shape != self.q_hold.shape or dq_real.shape != self.q_hold.shape:
                    raise InterventionTransitionError(
                        "state_read_failed",
                        "MuJoCo/leader joint dimension mismatch: "
                        f"expected {self.q_hold.shape}, got q={q_real.shape}, dq={dq_real.shape}",
                    )
                if not np.all(np.isfinite(q_real)) or not np.all(np.isfinite(dq_real)):
                    raise InterventionTransitionError(
                        "state_read_failed", "leader returned non-finite joint state"
                    )

            alpha, max_step = self._follow_gains()
            alpha = float(np.clip(alpha, 0.0, 1.0))
            q_current = np.asarray(self.data.ctrl[:7], dtype=np.float64)
            q_target = (1.0 - alpha) * q_current + alpha * q_real
            if max_step > 0.0:
                q_target = q_current + np.clip(q_target - q_current, -max_step, max_step)
            self.data.ctrl[:7] = q_target

            if hasattr(self.robot_adapter, "get_gripper_width"):
                gripper_width = float(self.robot_adapter.get_gripper_width())
            else:
                gripper_width = float(self.robot.robot_gripper.get_sensors().item())
            if self.model.nu >= 8:
                # The leader's gripper is a CONTINUOUS Dynamixel axis, and
                # FactrControlAdapter.get_gripper_width() already maps its calibrated
                # [gripper_limit_min, gripper_limit_max] travel linearly onto a real width.
                # get_gripper_pressed() throws that away, reducing the whole axis to one
                # comparison against gripper_close_threshold -- so the sim could only ever be
                # fully open or fully shut and the operator had no partial grasp at all.
                #
                # Continuous is therefore the DEFAULT. The threshold path is kept, opt-in via
                # FACTR_MUJOCO_GRIPPER_MODE=standalone|threshold|pressed|binary, because it is
                # the right behaviour for a leader whose gripper really is a trigger switch
                # (and it is what every FACTR recording before this change used).
                use_threshold_gripper = (
                    self.factr_mode
                    and os.environ.get(
                        "FACTR_MUJOCO_GRIPPER_MODE",
                        os.environ.get(
                            "INTERVENE_FACTR_MUJOCO_GRIPPER_MODE", "continuous"
                        ),
                    ).strip().lower()
                    in {"standalone", "threshold", "pressed", "binary"}
                    and hasattr(self.robot_adapter, "get_gripper_pressed")
                )
                if use_threshold_gripper:
                    ctrlrange = self.model.actuator_ctrlrange[7]
                    if bool(self.robot_adapter.get_gripper_pressed()):
                        self.factr_gripper_target_ctrl = float(ctrlrange[0])
                    else:
                        self.factr_gripper_target_ctrl = float(ctrlrange[1])
                    finger_pos = float(self.data.ctrl[7])
                    self.factr_gripper_ctrl_filtered = None
                elif self.factr_mode:
                    finger_pos = gripper_width_to_ctrl(self.model, gripper_width)
                    if finger_pos is None:
                        finger_pos = 0.0
                else:
                    # ctrl[7] units come from the MODEL, not a hardcoded convention.
                    #
                    # This used to hardcode `width / 2`, with a note that it deliberately
                    # avoided gripper_width_to_ctrl because "the two disagree on soft
                    # grippers". They do -- and on the soft gripper the hardcoded branch is
                    # the wrong one. The soft-gripper actuator (`actuator8`) is a position
                    # servo on finger_joint1 with ctrlrange [0.008, 0.0875] and
                    # gain/|bias| = 754.717/1500 = 0.50314, i.e. its ctrl is a FULL width
                    # and the joint settles at half of it. Feeding it width/2 therefore
                    # commanded a HALF-width gripper: the sim closed to ~w/2 while the
                    # operator's real hand was open at w, so every hand-guided grasp
                    # squeezed roughly twice as hard as the human was actually holding
                    # (measured on a cup grasp: |actuator8| 21.6 N -> 7.7 N after this fix).
                    # The policy path never had this bug -- ACT writes ctrl directly, which
                    # is why the same cup is easy to grasp under policy and fights back
                    # under human control.
                    #
                    # gripper_width_to_ctrl picks the branch from `ctrlrange[1]`, so the
                    # classic 0..0.04 gripper still gets width/2 exactly as before.
                    # INTERVENE_LEGACY_GRIPPER_WIDTH_CTRL=1 restores the old arithmetic for
                    # comparing against telekinesis/MC recordings made before this change.
                    if _env_flag("INTERVENE_LEGACY_GRIPPER_WIDTH_CTRL", default="0"):
                        finger_pos = float(np.clip(gripper_width / 2.0, 0.0, 0.04))
                    else:
                        finger_pos = gripper_width_to_ctrl(self.model, gripper_width)
                        if finger_pos is None:
                            finger_pos = float(np.clip(gripper_width / 2.0, 0.0, 0.04))
                if self.factr_mode and not use_threshold_gripper:
                    gripper_max_step = float(
                        os.environ.get(
                            "FACTR_MUJOCO_GRIPPER_MAX_STEP",
                            os.environ.get(
                                "INTERVENE_FACTR_MUJOCO_GRIPPER_MAX_STEP", "0.008"
                            ),
                        )
                    )
                    current_gripper = (
                        float(self.data.ctrl[7])
                        if self.factr_gripper_ctrl_filtered is None
                        else float(self.factr_gripper_ctrl_filtered)
                    )
                    if gripper_max_step > 0.0:
                        finger_pos = current_gripper + float(
                            np.clip(
                                finger_pos - current_gripper,
                                -gripper_max_step,
                                gripper_max_step,
                            )
                        )
                    self.factr_gripper_ctrl_filtered = float(finger_pos)
                # Apply directly — no EMA so the sim instantly reflects the real gripper state
                self.data.ctrl[7] = finger_pos
                # The finger qpos is NOT written here any more.
                #
                # It used to be, "to bypass actuator lag in mj_step" -- these were the only
                # two finger-qpos writes in the repo, and they ran on the hand-guided path
                # only. Teleporting a joint that is in contact overrides whatever the solver
                # resolved: with a cup between the fingers it re-imposed the commanded
                # opening every tick, so each tick the servo drove the fingers into the cup
                # and the next teleport yanked them back out -- a forced cycle at the tick
                # rate, injected straight into the grasp contact, with qvel left untouched
                # so the fingers kept their velocity across the jump. The policy path has
                # never done this; it writes ctrl and lets the servo move the fingers, and
                # that is now what happens here too. The "lag" being bypassed is one
                # control tick of a servo whose settling time is far shorter than that.
                #
                # INTERVENE_LEGACY_GRIPPER_QPOS_WRITE=1 restores the teleport.
                if (
                    not use_threshold_gripper
                    and self.data.qpos.shape[0] >= 9
                    and _env_flag("INTERVENE_LEGACY_GRIPPER_QPOS_WRITE", default="0")
                ):
                    self.data.qpos[7] = finger_pos
                    self.data.qpos[8] = finger_pos

            # Withheld until the operator actually moves the arm; see _recording_gate_open.
            if self._recording_gate_open():
                _status = self._frame_status()
                self.recorder.record(
                    now_wall=time.time(),
                    data=self.data,
                    q_real=q_real,
                    dq_real=dq_real,
                    gripper_width=gripper_width,
                    intervention=True,
                    intervention_id=self.intervention_id,
                    image_capture_fn=self.image_capture_fn,
                    # Human frames carry the same system-state columns as autonomous
                    # ones, so a segment can be analysed without joining against
                    # events.jsonl. `policy_step` is deliberately the FROZEN frame_idx:
                    # the policy takes no action while the human has the arm, and that is
                    # exactly what makes "did ACT act after RELEASE?" answerable from the
                    # trajectory alone.
                    sim_step=self._sim_step(),
                    policy_step=_status["policy_step"],
                    control_state=self.phase,
                    cell_selected=_status["cell_selected"],
                    paused=True,
                )

            steps = max(1, int(self.dt / self.model.opt.timestep))
            for _ in range(steps):
                # The gripper ramp stays FACTR-only, and within FACTR it is threshold-mode
                # only -- it returns immediately under the continuous default, which writes
                # the leader's measured width directly. Every other path already writes a
                # real width.
                if self.factr_mode:
                    self._advance_factr_standalone_gripper(self.model.opt.timestep)
                # Gravity compensation is NOT FACTR-only. It used to sit inside the
                # branch above, which is why a telekinesis takeover visibly dropped the
                # arm at handoff (~2 deg on joint2) while FACTR and sim-MC did not.
                self._apply_arm_gravity_compensation()
                mujoco.mj_step(self.model, self.data)

        sleep_time = self.dt - (time.time() - t0)
        if sleep_time > 0:
            time.sleep(sleep_time)
        return True

    def _resolve_panda_dof_indices(self) -> np.ndarray:
        """DOF indices of the 7 arm joints, by name, with a positional fallback.

        Tries every naming convention in PANDA_JOINT_NAME_SETS before falling back to
        "the first 7 DOFs" -- which is right for the current scenes but only because
        the arm happens to be declared first.
        """
        for names in PANDA_JOINT_NAME_SETS:
            indices = []
            for name in names:
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if joint_id < 0:
                    indices = None
                    break
                indices.append(int(self.model.jnt_dofadr[joint_id]))
            if indices:
                return np.asarray(indices, dtype=np.int64)
        return np.arange(min(7, self.data.qvel.shape[0]), dtype=np.int64)

    def _advance_factr_standalone_gripper(self, dt: float) -> None:
        """Ramp the sim gripper toward the leader's open/closed target at a physical rate.

        THRESHOLD MODE ONLY (FACTR_MUJOCO_GRIPPER_MODE=standalone|threshold|pressed|binary),
        and a no-op otherwise. When the leader's gripper is read as a trigger, ctrl[7] has
        only two possible targets, and stepping straight to one would slam the fingers shut
        in a single substep -- hence the ramp.

        The continuous default needs none of this: it writes the leader's measured width
        every tick and rate-limits it with FACTR_MUJOCO_GRIPPER_MAX_STEP instead.
        """
        if self.model.nu < 8:
            return
        mode = os.environ.get(
            "FACTR_MUJOCO_GRIPPER_MODE",
            os.environ.get("INTERVENE_FACTR_MUJOCO_GRIPPER_MODE", "continuous"),
        ).strip().lower()
        # Self-disabling: in continuous mode ctrl[7] is written directly from the leader's
        # measured width every tick, so there is no open/closed endpoint to ramp toward.
        # This default must stay in step with the one in update(), or the ramp would fight
        # the width writes.
        if mode not in {"standalone", "threshold", "pressed", "binary"}:
            return

        current = float(self.data.ctrl[7])
        target = float(self.factr_gripper_target_ctrl)
        delta = target - current
        if abs(delta) < 1e-9:
            self.data.ctrl[7] = target
            return

        ctrlrange = self.model.actuator_ctrlrange[7]
        full_span = max(1e-9, abs(float(ctrlrange[1]) - float(ctrlrange[0])))
        duration = (
            self.factr_gripper_close_time
            if target < current
            else self.factr_gripper_open_time
        )
        max_step = full_span * max(0.0, float(dt)) / max(1e-6, duration)
        if abs(delta) <= max_step:
            self.data.ctrl[7] = target
        else:
            self.data.ctrl[7] = current + float(np.sign(delta)) * max_step

    def _follow_gains(self):
        """(alpha, max_step) for this tick's blend toward the operator's arm.

        Steady state is per-mode and UNCHANGED: telekinesis keeps alpha=self.alpha with
        no clamp, FACTR keeps its own tuned pair. What is shared is the *takeover ramp*.

        Why telekinesis needs one. `_move_robot_to_q_hold` accepts an alignment error of
        up to `tol` (0.03 rad = 1.7 deg) and, on timeout, prints "continuing" and hands
        over anyway -- so at t=0 the real arm is generally NOT at `q_hold`, and gravity
        makes the residual point downward. With alpha=0.6 and no clamp the sim crossed
        that whole gap in ~3 ticks, which reads as the arm dropping. Ramping spreads it
        over `ramp_s` so it reads as settling into tracking, which is what it is.

        After the ramp the gains return exactly to the mode's steady values, so
        mid-segment hand-guiding speed is untouched.
        """
        factr = bool(self.factr_mode)

        def _f(name, default):
            # FACTR keeps its established FACTR_MUJOCO_* names; telekinesis gets the
            # INTERVENE_TAKEOVER_* namespace. Both accept the INTERVENE_FACTR_* alias
            # that already existed.
            keys = ((f"FACTR_MUJOCO_{name}", f"INTERVENE_FACTR_MUJOCO_{name}")
                    if factr else (f"INTERVENE_TAKEOVER_{name}",))
            for key in keys:
                raw = os.environ.get(key)
                if raw not in (None, ""):
                    try:
                        return float(raw)
                    except ValueError:
                        print(f"[REPLAN][WARN] {key}={raw!r} is not a number; using default.")
            return float(default)

        if factr:
            steady_alpha = _f("ALPHA", self.alpha)
            steady_max_step = _f("MAX_STEP_RAD", 0.018)
        else:
            steady_alpha = self.alpha
            steady_max_step = 0.0          # unclamped, as telekinesis has always been

        ramp_s = _f("TAKEOVER_RAMP_SECONDS", 1.2 if factr else 0.8)
        takeover_max_step = _f("TAKEOVER_MAX_STEP_RAD", 0.002 if factr else 0.004)
        alpha_start = _f("TAKEOVER_ALPHA_START", 0.05)

        # getattr: several LAB contract tests build this class via __new__ and never run
        # __init__. No ramp start recorded means no ramp, i.e. the steady gains.
        started = getattr(self, "takeover_started_wall", None)
        if ramp_s <= 0.0 or takeover_max_step <= 0.0 or started is None:
            return steady_alpha, steady_max_step

        progress = float(np.clip((time.time() - started) / ramp_s, 0.0, 1.0))
        if progress >= 1.0:
            return steady_alpha, steady_max_step

        alpha = alpha_start + progress * (steady_alpha - alpha_start)
        # An unclamped steady state (telekinesis) ramps toward a step large enough to be
        # inert, then drops the clamp entirely at progress == 1 above.
        target_step = steady_max_step if steady_max_step > 0.0 else 0.05
        max_step = takeover_max_step + progress * (target_step - takeover_max_step)
        return alpha, max_step

    def _apply_arm_gravity_compensation(self) -> None:
        """Hold the sim arm against gravity so it tracks the human instead of sagging.

        Called on every hand-guided path (telekinesis and FACTR); it owns qfrc_applied
        for the duration of the takeover, and `App._end_intervention_transport` zeroes
        it on every exit path. `qfrc_bias` is c(q, v) -- Coriolis/centrifugal/gravity --
        so applying it on the arm DOFs cancels gravity while leaving contacts and the
        XML's position actuators untouched.
        """
        if not INTERVENTION_ARM_GRAVITY_COMP:
            self.data.qfrc_applied[:] = 0.0
            return
        self.data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.data.qfrc_applied[self.factr_arm_dof_indices] = self.data.qfrc_bias[
            self.factr_arm_dof_indices
        ]

    # Kept so external callers and the LAB contract tests that reference the old name
    # keep working; the behaviour is identical and no longer FACTR-specific.
    _apply_factr_standalone_arm_gravity_compensation = _apply_arm_gravity_compensation

    def save_and_finish(self) -> str:
        if not self.finished:
            self.phase = "finishing"
            self.recorder.save()
            self.finished = True
            print(f"[REPLAN] Saved suffix to: {self.save_path}")
        return str(self.save_path)

    def finish_segment(self):
        self.phase = "finishing"
        self.finished = True
        return self.robot_adapter
