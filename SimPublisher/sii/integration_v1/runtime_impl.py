import argparse
import asyncio
import errno
import faulthandler
import gc
import json
import math
import os
import queue as _queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[3]
SIMPUBLISHER_ROOT = THIS_FILE.parents[2]
AG_ROOT = THIS_FILE.parents[1]
SIMPUB_SRC = SIMPUBLISHER_ROOT / "src"
INTERVENE_ROOT = REPO_ROOT / "intervene_base"

for path in [AG_ROOT, SIMPUB_SRC, INTERVENE_ROOT, REPO_ROOT]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import glfw
import mujoco
import numpy as np
import zmq

import franka_quest_unified_publisher as unified
from perf_metrics import MetricsCollector, default_metrics_path
# Shared with the policy processes and the desktop grid so all three compile the SAME model
# from the same descriptor. Importable because INTERVENE_ROOT is on sys.path above.
# End-to-end latency probe. Optional: the measurement rig is not required to run a
# session, so an incomplete checkout must not stop the runtime from starting.
try:
    from tools.latency_stamp import LatencyProbe, parse_echo, stamped as _latency_stamped
except Exception:  # pragma: no cover - measurement rig absent
    LatencyProbe = None
    parse_echo = lambda _msg: None  # noqa: E731
    _latency_stamped = None

from model_variants.builder import describe_reference, fingerprint_diff
from model_variants.descriptor import BASE_KEY as BASE_VARIANT_KEY, scene_closure_sha256
from simpub.sim.mj_publisher import MujocoPublisher
from simpub.xr_device.meta_quest3 import MetaQuest3


if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


INTEGRATION_VERSION = "2026-03-19-intervention-vr-v1"
STATE_REPLAY_RUNNING = "replay_running"
STATE_REPLAY_PAUSED = "replay_paused"
STATE_REPLAY_SYNC = "replay_sync"
STATE_REPLAN_HOLD = "replan_hold"
STATE_REPLAN_RECORD = "replan_record"
STATE_REPLAN_TRANSITION = "replan_transition"
DEFAULT_MAX_DQ = 0.80
ROBOT_SYNC_MAX_DQ = 0.90
REPLAN_HOLD_MAX_DQ = 0.90
REPLAN_SUFFIX_TIME_SCALE = 2.0
ROBOT_SOFTSTART_DURATION_S = 4.0
ROBOT_SYNC_JOINT_TOL_RAD = 0.03
ROBOT_SYNC_STABLE_CYCLES = 20
ROBOT_SYNC_TIMEOUT_S = 45.0
ROBOT_REPLAY_LAG_PAUSE_RAD = 0.35
ROBOT_REPLAY_LAG_PAUSE_CYCLES = 12
GRIPPER_OPEN_THRESHOLD_M = 0.03
GRIPPER_TRACK_EPS_M = 0.002
GRIPPER_CMD_PERIOD_S = 0.02
GRIPPER_CMD_SPEED = 0.35
GRIPPER_CMD_FORCE = 0.10
GRIPPER_FULL_WIDTH_CTRL_THRESHOLD_M = 0.06
GRIPPER_SIM_QPOS_SYNC_ALPHA = 0.75
JOINT7_MAX_DQ_SCALE = 2.0
SCENE_SETTLE_DURATION_S = 0.5
SCENE_QPOS_DIFF_EPS = 1e-6
SCENE_DIAGNOSTIC_BODIES = ("T1", "T2")
_INTERVENE_RECORD = None
_CONTROL_TYPE = None
# Cache for camera intrinsics — keyed by (cam_name, w, h, fovy_deg, mode).
# Populated on first publish of each camera; values are constant for the session.
# Set in main() when --latency_probe is on. Read by publish_sensor_frames and the
# CmdListener thread; both go through LatencyProbe's own lock.
_LATENCY_PROBE = None

_CAM_INTRINSICS_CACHE: Dict[tuple, tuple] = {}

# One-shot diagnostics to pinpoint where the PC publish chain breaks (submit → build →
# drain). Lists so the closures can mutate without `global`. [DIAG] lines are harmless.
_PC_DIAG = {"submit_logged": False, "drain_logged": False}


class PolicyStateMirror:
    def __init__(self, host: str, port: int, timeout_s: float):
        self.host = str(host)
        self.port = int(port or 0)
        self.timeout_s = float(timeout_s)
        self.sock = None
        self.last_receive_wall: Optional[float] = None
        self.last_applied_seq: Optional[int] = None
        self.last_episode_id: Optional[str] = None
        self.latest_frame_idx: int = 0
        self.latest_mode: str = ""
        self.latest_intervention_phase: str = "policy_resumed"
        self.last_qpos_delta_norm: float = 0.0
        self.last_state_change_wall: Optional[float] = None
        self.latest_state_wall: Optional[float] = None
        self.received_messages: int = 0
        self.received_seq_gaps: int = 0
        self.distinct_qpos_updates: int = 0
        self.repeated_qpos_updates: int = 0
        self._last_received_seq: Optional[int] = None
        self._stale_warned = False
        self._shape_warned = False
        self._next_stale_warn_t = 0.0
        self._next_shape_warn_t = 0.0
        # Readable staleness, so the rest of the runtime can distinguish "the sim is still"
        # from "the sim stopped being updated".
        self.stale = False
        self.stale_age_s = 0.0
        self.latest_acc_risk: float = 0.0
        # Authoritative user-pause state streamed by the policy runner (app.py sends
        # "paused": bool(player.is_paused) in every state message). Exposed here so the
        # main loop can publish SimPub/Status/paused in policy mode (where `state` is
        # force-set to STATE_REPLAY_PAUSED every frame and is therefore not a usable signal).
        self.latest_paused: bool = False
        self.latest_intervention_live: bool = False
        self._prev_mirror_mode: str = ""
        self._prev_intervention_live: bool = False
        # None until the first state message, so we adopt app.py's counter as a baseline
        # rather than firing the HUD for failures that predate this runtime.
        self._prev_intervention_failed_seq: Optional[int] = None
        self.latest_intervention_failed_reason: str = ""
        self._pending_status_event: Optional[str] = None
        # --- model variant ---------------------------------------------------------------
        # Which compiled model the upstream policy is publishing from. For a model-level OOD
        # task (cups) an OOD episode runs a variant of the SAME scene file with different cup
        # heights and swapped box textures. nq/nv/nu are identical across variants, so
        # without this the runtime would render correct numbers on the wrong geometry -- and
        # the point clouds streamed to the headset would be silently wrong.
        self.active_variant_key: str = "base"
        self.latest_variant_key: str = "base"
        self.latest_variant: Optional[dict] = None
        self.latest_next_variant: Optional[dict] = None
        self.latest_model_epoch: int = 0
        self.base_scene_sha256: str = ""
        self._scene_mismatch_warned = False

        if self.port <= 0:
            return

        endpoint = f"tcp://{self.host}:{self.port}"
        self.sock = zmq.Context.instance().socket(zmq.SUB)
        self.sock.setsockopt(zmq.LINGER, 0)
        self.sock.setsockopt(zmq.RCVHWM, 1)
        try:
            self.sock.setsockopt(zmq.CONFLATE, 1)
        except Exception:
            pass
        self.sock.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sock.connect(endpoint)
        print(f"[PolicyStateMirror] connected {endpoint}")

    @property
    def enabled(self) -> bool:
        return self.sock is not None

    def drain_latest(self) -> Optional[dict]:
        if self.sock is None:
            return None

        latest = None
        while True:
            try:
                raw = self.sock.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            try:
                latest = json.loads(raw.decode("utf-8"))
                self.last_receive_wall = time.time()
                self.received_messages += 1
                try:
                    received_seq = int(latest.get("seq", -1))
                except (TypeError, ValueError):
                    received_seq = -1
                if received_seq >= 0:
                    if self._last_received_seq is not None and received_seq > self._last_received_seq:
                        self.received_seq_gaps += max(0, received_seq - self._last_received_seq - 1)
                    self._last_received_seq = received_seq
                self._stale_warned = False
            except Exception as exc:
                print(f"[PolicyStateMirror][WARN] dropped malformed state: {exc}")
        return latest

    def warn_if_stale(self) -> None:
        """Report a frozen upstream sim, repeatedly.

        This used to print exactly once per silence period, which is how a 24 s policy-side
        stall produced a single log line while the point cloud kept streaming byte-identical
        frames at full rate — the scene looked live and was not. Re-warn on a throttle and
        keep `stale` readable so callers can surface it.
        """
        self.stale = False
        self.stale_age_s = 0.0
        if self.sock is None or self.last_receive_wall is None or self.timeout_s <= 0:
            return
        age = time.time() - self.last_receive_wall
        self.stale_age_s = age
        if age < self.timeout_s:
            return
        self.stale = True
        self.stale_age_s = age
        now = time.time()
        if now >= self._next_stale_warn_t:
            print(
                f"[PolicyStateMirror][WARN] stale state stream: "
                f"last_seq={self.last_applied_seq} age={age:.3f}s — the mirrored sim is "
                f"FROZEN; point clouds published now repeat the last pose."
            )
            self._next_stale_warn_t = now + 2.0
            self._stale_warned = True

    def apply_to(self, model: mujoco.MjModel, data: mujoco.MjData, message: dict) -> Optional[dict]:
        if not message:
            return None

        # Intervention-status detection runs BEFORE the shape check so it fires even if
        # qpos/qvel/ctrl shapes temporarily mismatch (e.g. during model reload).
        new_mode = str(message.get("mode", ""))
        self._prev_mirror_mode = new_mode
        self.latest_mode = new_mode
        self.latest_intervention_phase = str(
            message.get("intervention_phase", "human_control" if new_mode == "replan" else "policy_resumed")
        )
        try:
            self.latest_frame_idx = int(message.get("frame_idx", self.latest_frame_idx))
        except (TypeError, ValueError):
            pass
        # Track user-pause early (before the shape-check early-return below) so the PAUSED
        # badge stays current even during model-reload shape mismatches.
        self.latest_paused = bool(message.get("paused", False))
        # Authoritative LEVEL for "this session is being intervened on". The
        # SimPub/Status/intervention topic carries one-shot EDGES for the HUD; a supervisor
        # guard needs a level it can re-read at any time, and one that self-heals if an
        # edge is lost. Prefer app.py's explicit flag, fall back to the mode string.
        self.latest_intervention_live = bool(
            message.get("intervention_live", new_mode == "replan")
        )

        # An intervention the operator ASKED for but that could not start (robot
        # unreachable, or another session owns the arm). app.py bumps a monotonic counter;
        # any increase is a fresh failure to report. Checked before the started/finished
        # transitions below because a failed start never toggles intervention_live, so
        # without this the press produces no VR feedback whatsoever.
        try:
            failed_seq = int(message.get("intervention_failed_seq", 0))
        except (TypeError, ValueError):
            failed_seq = 0
        if self._prev_intervention_failed_seq is None:
            # First state message: adopt the baseline instead of firing for failures that
            # happened before this runtime attached.
            self._prev_intervention_failed_seq = failed_seq
        elif failed_seq > self._prev_intervention_failed_seq:
            self._prev_intervention_failed_seq = failed_seq
            self.latest_intervention_failed_reason = str(
                message.get("intervention_failed_reason", "")
            )
            failure_reason = self.latest_intervention_failed_reason or "transition_failed"
            self._pending_status_event = f"intervention_failed:{failure_reason}"
            print(
                "[PolicyMirror] intervention_failed queued "
                f"(seq={failed_seq}, reason={self.latest_intervention_failed_reason!r})"
            )

        if "intervention_live" in message:
            # Preferred path: fire on the precise "intervention_live" flag, which app.py sets True
            # only once the robot is aligned and handed to the operator. This makes the HUD appear
            # exactly when the intervention begins — not ~4s early at the replay→replan mode switch
            # (X press, before alignment).
            live = bool(message.get("intervention_live"))
            if live != self._prev_intervention_live:
                if live:
                    self._pending_status_event = "intervention_started"
                    print("[PolicyMirror] intervention_started queued (intervention_live: False → True)")
                else:
                    self._pending_status_event = "intervention_finished"
                    print("[PolicyMirror] intervention_finished queued (intervention_live: True → False)")
                self._prev_intervention_live = live
        elif new_mode and new_mode != getattr(self, "_legacy_prev_mode", ""):
            # Legacy fallback (older app.py without intervention_live): mode transition.
            if new_mode == "replan" and getattr(self, "_legacy_prev_mode", "") in ("", "replay"):
                self._pending_status_event = "intervention_started"
                print("[PolicyMirror] intervention_started queued (legacy mode → replan)")
            elif new_mode == "replay" and getattr(self, "_legacy_prev_mode", "") == "replan":
                self._pending_status_event = "intervention_finished"
                print("[PolicyMirror] intervention_finished queued (legacy mode → replay)")
            self._legacy_prev_mode = new_mode

        # --- model variant, BEFORE any state is written -----------------------------------
        # Deliberately ahead of the shape check for the same reason the mode/paused handling
        # above is: it must work even while a model reload is in flight.
        if int(message.get("version", 1)) >= 2:
            self.latest_variant_key = str(message.get("variant_key", "base") or "base")
            self.latest_variant = message.get("variant")
            self.latest_next_variant = message.get("next_variant")
            self.latest_model_epoch = int(message.get("model_epoch", 0) or 0)
            theirs = str(message.get("base_scene_sha256", "") or "")
            if theirs and self.base_scene_sha256 and theirs != self.base_scene_sha256:
                if not self._scene_mismatch_warned:
                    self._scene_mismatch_warned = True
                    print("[PolicyStateMirror][ERROR] upstream is running a DIFFERENT scene "
                          f"(base_scene_sha256 {theirs[:12]}... != "
                          f"{self.base_scene_sha256[:12]}...). Refusing its state; this "
                          "runtime and that policy were launched on different XMLs.")
                return None
        else:
            self.latest_variant_key = "base"
            self.latest_variant = None

        # A variant mismatch is REPORTED but never withholds state.
        #
        # qpos/qvel/ctrl are variant-independent: nq/nv/nu, joint order and actuator order
        # are identical by construction (enforced in model_variants/builder.py and
        # re-checked by fingerprint_diff before any swap), so writing them onto the CURRENT
        # model is dimensionally valid and semantically correct -- only the cup geometry is
        # a episode behind for the ~1.5 s the background compile takes.
        #
        # This used to return early with applied=False, which combined with a compile that
        # can take many ticks meant the mirrored sim froze while still publishing at full
        # rate -- exactly the failure the desktop grid's identical rule exists to prevent
        # (see policy_grid_viewer._sync_session_variant). Do not reintroduce it.
        variant_changed = self.latest_variant_key != self.active_variant_key

        qpos = np.asarray(message.get("qpos", []), dtype=np.float64)
        qvel = np.asarray(message.get("qvel", []), dtype=np.float64)
        ctrl = np.asarray(message.get("ctrl", []), dtype=np.float64)
        if qpos.shape != data.qpos.shape or qvel.shape != data.qvel.shape or ctrl.shape != data.ctrl.shape:
            # Re-armed on a throttle: this used to warn once and then freeze the sim
            # forever in silence while the HUD/paused flags kept updating.
            if not self._shape_warned or time.time() >= self._next_shape_warn_t:
                self._next_shape_warn_t = time.time() + 5.0
                print(
                    "[PolicyStateMirror][WARN] shape mismatch; ignoring live state "
                    f"qpos={qpos.shape}/{data.qpos.shape} "
                    f"qvel={qvel.shape}/{data.qvel.shape} "
                    f"ctrl={ctrl.shape}/{data.ctrl.shape}"
                )
                self._shape_warned = True
            return None

        seq = int(message.get("seq", -1))
        episode_id = str(message.get("episode_id", ""))
        first_apply = self.last_applied_seq is None
        episode_changed = self.last_episode_id is not None and episode_id != self.last_episode_id

        self.last_qpos_delta_norm = float(np.linalg.norm(qpos - np.asarray(data.qpos)))
        if self.last_state_change_wall is None or self.last_qpos_delta_norm > 1e-7:
            self.last_state_change_wall = time.time()
            self.distinct_qpos_updates += 1
        else:
            self.repeated_qpos_updates += 1
        try:
            self.latest_state_wall = float(message.get("wall_t"))
        except (TypeError, ValueError):
            self.latest_state_wall = None

        data.qpos[:] = qpos
        data.qvel[:] = qvel
        data.ctrl[:] = ctrl
        data.time = float(message.get("sim_t", data.time))
        mujoco.mj_forward(model, data)

        self.last_applied_seq = seq
        self.last_episode_id = episode_id
        self.latest_acc_risk = float(message.get("acc_risk", 0.0))

        if first_apply or episode_changed or seq % 30 == 0:
            print(
                "[PolicyStateMirror] applied "
                f"seq={seq} episode={episode_id} "
                f"mode={message.get('mode')} paused={message.get('paused')} "
                f"frame={message.get('frame_idx')} sim_t={float(data.time):.4f}"
            )

        return {
            "frame_idx": int(message.get("frame_idx", 0)),
            "first_apply": first_apply,
            "episode_changed": episode_changed,
            "seq": seq,
            "variant_changed": variant_changed,
            "variant": self.latest_variant,
            "variant_key": self.latest_variant_key,
            "model_epoch": self.latest_model_epoch,
            "applied": True,
        }

    def drain_status_event(self) -> Optional[str]:
        evt = self._pending_status_event
        self._pending_status_event = None
        return evt

    def close(self) -> None:
        sock = self.sock
        self.sock = None
        if sock is not None:
            try:
                sock.close(linger=0)
            except Exception:
                pass


class RobotOwnershipLock:
    def __init__(self, robot_key: str):
        safe_key = "".join(
            char if char.isalnum() or char in ("-", "_", ".") else "_"
            for char in str(robot_key)
        )
        self.robot_key = str(robot_key)
        self.path = Path("/tmp") / f"iilar_robot_{safe_key}.lock"
        self._fd: Optional[int] = None
        self._pid = os.getpid()

    @staticmethod
    def _pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True

    def _read_owner_pid(self) -> Optional[int]:
        try:
            text = self.path.read_text(errors="ignore")
        except FileNotFoundError:
            return None
        except Exception:
            return None
        for line in text.splitlines():
            if line.startswith("pid="):
                try:
                    return int(line.split("=", 1)[1].strip())
                except ValueError:
                    return None
        return None

    def acquire(self) -> Tuple[bool, str]:
        if self._fd is not None:
            return True, f"already owns {self.robot_key}"
        for _ in range(2):
            try:
                fd = os.open(str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                owner_pid = self._read_owner_pid()
                if owner_pid is not None and not self._pid_alive(owner_pid):
                    try:
                        self.path.unlink()
                        print(f"[RobotLock] Removed stale lock {self.path} from pid={owner_pid}.")
                    except FileNotFoundError:
                        pass
                    except Exception as exc:
                        return False, f"stale lock unlink failed: {exc}"
                    continue
                return False, f"locked by pid={owner_pid if owner_pid is not None else 'unknown'} at {self.path}"
            except Exception as exc:
                return False, f"could not create lock {self.path}: {exc}"

            payload = (
                f"pid={self._pid}\n"
                f"robot_key={self.robot_key}\n"
                f"created_unix={time.time():.3f}\n"
                f"argv={' '.join(sys.argv)}\n"
            )
            try:
                os.write(fd, payload.encode("utf-8", errors="replace"))
            except Exception:
                pass
            self._fd = fd
            return True, f"acquired {self.path}"
        return False, f"could not acquire {self.path}"

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            os.close(self._fd)
        except Exception:
            pass
        self._fd = None
        try:
            owner_pid = self._read_owner_pid()
            if owner_pid in (None, self._pid):
                self.path.unlink()
        except FileNotFoundError:
            pass
        except Exception as exc:
            print(f"[RobotLock][WARN] Could not release {self.path}: {exc}")


def load_robot_stack():
    global _INTERVENE_RECORD, _CONTROL_TYPE
    if _INTERVENE_RECORD is not None and _CONTROL_TYPE is not None:
        return _INTERVENE_RECORD, _CONTROL_TYPE
    try:
        import record as intervene_record
        from real_robot_env.robot.hardware_franka import ControlType
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Robot features require the intervene_base runtime dependencies in this Python environment "
            f"(missing module: {exc.name})."
        ) from exc
    _INTERVENE_RECORD = intervene_record
    _CONTROL_TYPE = ControlType
    return _INTERVENE_RECORD, _CONTROL_TYPE


@dataclass
class TrajectoryRecorder:
    save_path: Path
    log_hz: float
    view_hz: float
    sim_t_zero: float = 0.0
    _stride: int = field(init=False)
    _k: int = field(default=0, init=False)
    wall_t: List[float] = field(default_factory=list)
    sim_t: List[float] = field(default_factory=list)
    q_real: List[np.ndarray] = field(default_factory=list)
    dq_real: List[np.ndarray] = field(default_factory=list)
    grip_real: List[float] = field(default_factory=list)
    qpos_sim: List[np.ndarray] = field(default_factory=list)
    qvel_sim: List[np.ndarray] = field(default_factory=list)
    ctrl_sim: List[np.ndarray] = field(default_factory=list)

    def __post_init__(self):
        self._stride = max(1, int(round(self.view_hz / max(self.log_hz, 1e-6))))

    def record(
        self,
        *,
        now_wall: float,
        data: mujoco.MjData,
        q_real: np.ndarray,
        dq_real: Optional[np.ndarray],
        gripper_width: float,
    ) -> None:
        self._k += 1
        if (self._k % self._stride) != 0:
            return
        self.wall_t.append(float(now_wall))
        self.sim_t.append(float(data.time - self.sim_t_zero))
        self.q_real.append(np.asarray(q_real, dtype=np.float64).copy())
        if dq_real is None:
            self.dq_real.append(np.full_like(np.asarray(q_real, dtype=np.float64), np.nan))
        else:
            self.dq_real.append(np.asarray(dq_real, dtype=np.float64).copy())
        self.grip_real.append(float(gripper_width))
        self.qpos_sim.append(np.asarray(data.qpos, dtype=np.float64).copy())
        self.qvel_sim.append(np.asarray(data.qvel, dtype=np.float64).copy())
        self.ctrl_sim.append(np.asarray(data.ctrl, dtype=np.float64).copy())

    def save(self) -> Path:
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        if len(self.q_real) == 0:
            raise RuntimeError("Recorder has no samples to save.")
        np.savez_compressed(
            self.save_path,
            wall_t=np.asarray(self.wall_t, dtype=np.float64),
            sim_t=np.asarray(self.sim_t, dtype=np.float64),
            q_real=np.stack(self.q_real),
            dq_real=np.stack(self.dq_real),
            grip_real=np.asarray(self.grip_real, dtype=np.float64),
            qpos_sim=np.stack(self.qpos_sim),
            qvel_sim=np.stack(self.qvel_sim),
            ctrl_sim=np.stack(self.ctrl_sim),
        )
        print(f"[Integration] Saved suffix trajectory: {self.save_path}")
        return self.save_path


@dataclass
class ReplayRobotBindings:
    arm_qpos_idx: np.ndarray
    arm_dof_idx: np.ndarray
    arm_ctrl_idx: np.ndarray
    finger_qpos_idx: np.ndarray
    finger_dof_idx: np.ndarray
    finger_ctrl_idx: int
    finger_ctrl_range: np.ndarray
    finger_qpos_range: np.ndarray
    finger_ctrl_is_width: bool

    @property
    def robot_qpos_idx(self) -> np.ndarray:
        parts = [self.arm_qpos_idx.reshape(-1), self.finger_qpos_idx.reshape(-1)]
        valid = [part[part >= 0] for part in parts if part.size > 0]
        if not valid:
            return np.zeros((0,), dtype=np.int32)
        return np.unique(np.concatenate(valid, axis=0)).astype(np.int32)

    @property
    def has_finger_ctrl(self) -> bool:
        return int(self.finger_ctrl_idx) >= 0

    def grip_width_from_ctrl(self, ctrl_value: float) -> float:
        ctrl_value = float(ctrl_value)
        if self.finger_ctrl_is_width:
            ctrl_high = float(self.finger_ctrl_range[1]) if self.finger_ctrl_range.size >= 2 else 0.08
            return float(np.clip(ctrl_value, 0.0, max(ctrl_high, 0.08)))
        return float(np.clip(ctrl_value * 2.0, 0.0, self.max_grip_width()))

    def ctrl_from_grip_width(self, width_m: float) -> float:
        width_m = float(max(0.0, width_m))
        ctrl_value = width_m if self.finger_ctrl_is_width else width_m / 2.0
        if self.finger_ctrl_range.size >= 2:
            ctrl_low = float(self.finger_ctrl_range[0])
            ctrl_high = float(self.finger_ctrl_range[1])
            if ctrl_high > ctrl_low:
                ctrl_value = float(np.clip(ctrl_value, ctrl_low, ctrl_high))
        return float(ctrl_value)

    def finger_qpos_from_grip_width(self, width_m: float) -> float:
        finger_qpos = float(max(0.0, width_m) / 2.0)
        valid_ranges = self.finger_qpos_range[
            np.logical_and(self.finger_qpos_idx >= 0, self.finger_qpos_range[:, 1] > self.finger_qpos_range[:, 0])
        ]
        if valid_ranges.size:
            qpos_low = float(np.max(valid_ranges[:, 0]))
            qpos_high = float(np.min(valid_ranges[:, 1]))
            if qpos_high > qpos_low:
                finger_qpos = float(np.clip(finger_qpos, qpos_low, qpos_high))
        return finger_qpos

    def max_grip_width(self) -> float:
        valid_ranges = self.finger_qpos_range[
            np.logical_and(self.finger_qpos_idx >= 0, self.finger_qpos_range[:, 1] > self.finger_qpos_range[:, 0])
        ]
        if valid_ranges.size:
            return float(max(0.08, 2.0 * float(np.min(valid_ranges[:, 1]))))
        if self.finger_ctrl_is_width and self.finger_ctrl_range.size >= 2:
            return float(max(0.08, float(self.finger_ctrl_range[1])))
        return 0.08

    def current_grip_width(self, data: mujoco.MjData) -> float:
        for qpos_index in self.finger_qpos_idx.reshape(-1):
            qpos_index = int(qpos_index)
            if 0 <= qpos_index < data.qpos.shape[0]:
                return float(np.clip(2.0 * data.qpos[qpos_index], 0.0, self.max_grip_width()))
        if self.has_finger_ctrl and int(self.finger_ctrl_idx) < data.ctrl.shape[0]:
            return self.grip_width_from_ctrl(float(data.ctrl[int(self.finger_ctrl_idx)]))
        return 0.0

    def sync_finger_state_from_grip_width(
        self,
        data: mujoco.MjData,
        width_m: float,
        *,
        dt: float,
        alpha: float = 1.0,
    ) -> None:
        if self.has_finger_ctrl and int(self.finger_ctrl_idx) < data.ctrl.shape[0]:
            target_ctrl = self.ctrl_from_grip_width(width_m)
            ctrl_index = int(self.finger_ctrl_idx)
            data.ctrl[ctrl_index] = (1.0 - alpha) * data.ctrl[ctrl_index] + alpha * target_ctrl

        target_qpos = self.finger_qpos_from_grip_width(width_m)
        safe_dt = max(float(dt), 1e-6)
        for qpos_index, dof_index in zip(self.finger_qpos_idx.reshape(-1), self.finger_dof_idx.reshape(-1)):
            qpos_index = int(qpos_index)
            dof_index = int(dof_index)
            if 0 <= qpos_index < data.qpos.shape[0]:
                previous_qpos = float(data.qpos[qpos_index])
                next_qpos = (1.0 - alpha) * previous_qpos + alpha * target_qpos
                data.qpos[qpos_index] = next_qpos
                if 0 <= dof_index < data.qvel.shape[0]:
                    data.qvel[dof_index] = (next_qpos - previous_qpos) / safe_dt


@dataclass
class SceneSeedState:
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    table_plane: Optional[unified.Plane]
    table_top_z: Optional[float]
    body_world_pos: Dict[str, np.ndarray]
    body_world_quat: Dict[str, np.ndarray]
    settle_steps: int
    settle_duration_s: float

    def restore(self, data: mujoco.MjData) -> None:
        data.qpos[:] = self.qpos
        data.qvel[:] = self.qvel
        data.ctrl[:] = self.ctrl
        data.time = 0.0
        mujoco.mj_forward(data.model, data)


@dataclass
class ReplanStateSnapshot:
    qpos: np.ndarray
    qvel: np.ndarray
    ctrl: np.ndarray
    time: float
    act: Optional[np.ndarray] = None
    qacc_warmstart: Optional[np.ndarray] = None
    qfrc_applied: Optional[np.ndarray] = None
    xfrc_applied: Optional[np.ndarray] = None
    mocap_pos: Optional[np.ndarray] = None
    mocap_quat: Optional[np.ndarray] = None
    userdata: Optional[np.ndarray] = None

    @staticmethod
    def _copy_array(data: mujoco.MjData, name: str) -> Optional[np.ndarray]:
        value = getattr(data, name, None)
        if value is None:
            return None
        try:
            return np.asarray(value, dtype=np.float64).copy()
        except Exception:
            return None

    @classmethod
    def capture(cls, data: mujoco.MjData) -> "ReplanStateSnapshot":
        return cls(
            qpos=np.asarray(data.qpos, dtype=np.float64).copy(),
            qvel=np.asarray(data.qvel, dtype=np.float64).copy(),
            ctrl=np.asarray(data.ctrl, dtype=np.float64).copy(),
            time=float(data.time),
            act=cls._copy_array(data, "act"),
            qacc_warmstart=cls._copy_array(data, "qacc_warmstart"),
            qfrc_applied=cls._copy_array(data, "qfrc_applied"),
            xfrc_applied=cls._copy_array(data, "xfrc_applied"),
            mocap_pos=cls._copy_array(data, "mocap_pos"),
            mocap_quat=cls._copy_array(data, "mocap_quat"),
            userdata=cls._copy_array(data, "userdata"),
        )

    @staticmethod
    def _restore_array(data: mujoco.MjData, name: str, value: Optional[np.ndarray]) -> None:
        if value is None or not hasattr(data, name):
            return
        target = getattr(data, name)
        try:
            if target.shape == value.shape:
                target[...] = value
        except Exception:
            pass

    def restore(self, data: mujoco.MjData) -> None:
        data.qpos[:] = self.qpos
        data.qvel[:] = self.qvel
        data.ctrl[:] = self.ctrl
        self.restore_auxiliary(data)
        data.time = float(self.time)
        mujoco.mj_forward(data.model, data)

    def restore_auxiliary(self, data: mujoco.MjData) -> None:
        self._restore_array(data, "act", self.act)
        self._restore_array(data, "qacc_warmstart", self.qacc_warmstart)
        self._restore_array(data, "qfrc_applied", self.qfrc_applied)
        self._restore_array(data, "xfrc_applied", self.xfrc_applied)
        self._restore_array(data, "mocap_pos", self.mocap_pos)
        self._restore_array(data, "mocap_quat", self.mocap_quat)
        self._restore_array(data, "userdata", self.userdata)

    def summary(self) -> str:
        return (
            f"time={self.time:.4f} "
            f"qpos_norm={float(np.linalg.norm(self.qpos)):.4f} "
            f"qvel_norm={float(np.linalg.norm(self.qvel)):.4f}"
        )


class ReplayTrajectory:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._npz = np.load(self.path)
        required = ["sim_t", "ctrl_sim", "qpos_sim", "qvel_sim"]
        missing = [name for name in required if name not in self._npz.files]
        if missing:
            raise RuntimeError(f"Trajectory '{self.path}' is missing required arrays: {missing}")
        self.sim_t = np.asarray(self._npz["sim_t"], dtype=np.float64).reshape(-1)
        self.ctrl_sim = np.asarray(self._npz["ctrl_sim"], dtype=np.float64)
        self.qpos_sim = np.asarray(self._npz["qpos_sim"], dtype=np.float64)
        self.qvel_sim = np.asarray(self._npz["qvel_sim"], dtype=np.float64)
        self.grip_real = (
            np.asarray(self._npz["grip_real"], dtype=np.float64).reshape(-1)
            if "grip_real" in self._npz.files
            else None
        )
        if len(self.sim_t) == 0:
            raise RuntimeError(f"Trajectory '{self.path}' is empty.")

    def __len__(self) -> int:
        return int(self.sim_t.shape[0])

    def grip_width_at(
        self,
        idx: int,
        *,
        replay_bindings: Optional[ReplayRobotBindings] = None,
    ) -> Optional[float]:
        if self.grip_real is not None and idx < len(self.grip_real):
            return float(self.grip_real[idx])
        if self.ctrl_sim.ndim == 2 and idx < self.ctrl_sim.shape[0] and self.ctrl_sim.shape[1] > 7:
            ctrl_value = float(self.ctrl_sim[idx, 7])
            if replay_bindings is not None:
                return replay_bindings.grip_width_from_ctrl(ctrl_value)
            return float(np.clip(ctrl_value * 2.0, 0.0, 0.08))
        return None

    def apply_frame(
        self,
        idx: int,
        data: mujoco.MjData,
        *,
        replay_bindings: Optional[ReplayRobotBindings] = None,
        scene_seed: Optional[SceneSeedState] = None,
        restore_scene: bool = False,
    ) -> int:
        idx = int(np.clip(idx, 0, len(self) - 1))
        if replay_bindings is None:
            data.qpos[:] = self.qpos_sim[idx]
            data.qvel[:] = self.qvel_sim[idx]
            data.ctrl[:] = self.ctrl_sim[idx]
            data.time = float(self.sim_t[idx])
            mujoco.mj_forward(data.model, data)
            return idx

        if restore_scene:
            if scene_seed is None:
                raise ValueError("scene_seed is required when restore_scene=True")
            scene_seed.restore(data)

        qpos_frame = np.asarray(self.qpos_sim[idx], dtype=np.float64).reshape(-1)
        qvel_frame = np.asarray(self.qvel_sim[idx], dtype=np.float64).reshape(-1)
        ctrl_frame = np.asarray(self.ctrl_sim[idx], dtype=np.float64).reshape(-1)

        for qadr in replay_bindings.robot_qpos_idx:
            if 0 <= int(qadr) < data.qpos.shape[0] and int(qadr) < qpos_frame.shape[0]:
                data.qpos[int(qadr)] = qpos_frame[int(qadr)]

        robot_dof_idx = np.concatenate(
            [
                replay_bindings.arm_dof_idx.reshape(-1),
                replay_bindings.finger_dof_idx.reshape(-1),
            ],
            axis=0,
        )
        for dadr in robot_dof_idx:
            if 0 <= int(dadr) < data.qvel.shape[0] and int(dadr) < qvel_frame.shape[0]:
                data.qvel[int(dadr)] = qvel_frame[int(dadr)]

        for i, ctrl_idx in enumerate(replay_bindings.arm_ctrl_idx.reshape(-1)):
            if 0 <= int(ctrl_idx) < data.ctrl.shape[0] and i < ctrl_frame.shape[0]:
                data.ctrl[int(ctrl_idx)] = ctrl_frame[i]

        finger_ctrl_value = None
        if ctrl_frame.shape[0] > 7:
            finger_ctrl_value = float(ctrl_frame[7])
        elif self.grip_real is not None and idx < len(self.grip_real):
            finger_ctrl_value = replay_bindings.ctrl_from_grip_width(float(self.grip_real[idx]))
        elif replay_bindings.finger_qpos_idx.size > 0:
            qadr = int(replay_bindings.finger_qpos_idx[0])
            if 0 <= qadr < data.qpos.shape[0]:
                finger_ctrl_value = replay_bindings.ctrl_from_grip_width(float(data.qpos[qadr] * 2.0))
        if finger_ctrl_value is not None and 0 <= int(replay_bindings.finger_ctrl_idx) < data.ctrl.shape[0]:
            data.ctrl[int(replay_bindings.finger_ctrl_idx)] = finger_ctrl_value

        data.time = float(self.sim_t[idx])
        mujoco.mj_forward(data.model, data)
        return idx

    def apply_controls(
        self,
        idx: int,
        data: mujoco.MjData,
        *,
        replay_bindings: ReplayRobotBindings,
    ) -> int:
        idx = int(np.clip(idx, 0, len(self) - 1))
        ctrl_frame = np.asarray(self.ctrl_sim[idx], dtype=np.float64).reshape(-1)

        for i, ctrl_idx in enumerate(replay_bindings.arm_ctrl_idx.reshape(-1)):
            if 0 <= int(ctrl_idx) < data.ctrl.shape[0] and i < ctrl_frame.shape[0]:
                data.ctrl[int(ctrl_idx)] = ctrl_frame[i]

        finger_ctrl_value = None
        if ctrl_frame.shape[0] > 7:
            finger_ctrl_value = float(ctrl_frame[7])
        elif self.grip_real is not None and idx < len(self.grip_real):
            finger_ctrl_value = replay_bindings.ctrl_from_grip_width(float(self.grip_real[idx]))
        if finger_ctrl_value is not None and 0 <= int(replay_bindings.finger_ctrl_idx) < data.ctrl.shape[0]:
            data.ctrl[int(replay_bindings.finger_ctrl_idx)] = finger_ctrl_value
        return idx


@dataclass
class ReplanSession:
    cut_idx: int
    q_seed: np.ndarray
    finger_seed: float
    qpos_seed: np.ndarray
    qvel_seed: np.ndarray
    start_snapshot: Optional[ReplanStateSnapshot] = None
    final_snapshot: Optional[ReplanStateSnapshot] = None
    recorder: Optional[TrajectoryRecorder] = None
    suffix_path: Optional[Path] = None
    sim_time_zero: float = 0.0
    hold_confirmed: bool = False


@dataclass
class ReplaySyncState:
    target_q: np.ndarray
    grip_target: Optional[float]
    start_t: float
    after_align: str = "resume"
    stable_cycles: int = 0


@dataclass
class SensorPublishState:
    next_sensor_t: float
    frame_idx: int = 0
    anchor_corr_by_cam: Dict[str, np.ndarray] = field(default_factory=dict)
    anchor_corr_global: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=np.float64)
    )
    first_rgb_logged: Dict[str, bool] = field(default_factory=dict)
    first_rgbd_logged: Dict[str, bool] = field(default_factory=dict)
    first_pc_logged: Dict[str, bool] = field(default_factory=dict)
    pc_debug_visibility_saved: Dict[str, bool] = field(default_factory=dict)
    # Adaptive-rate state: tracks last reported mode so we only log on
    # transitions (idle ↔ active) rather than every frame.
    last_adaptive_mode: str = ""
    last_adaptive_peers: int = -1
    # Debounce for the peer-count fallback. Combined view (both Unity scenes loaded)
    # leaves every session with live subscribers and panels that reconnect on an 8 s
    # idle timer, so a raw peer>=threshold test promoted UNSELECTED sessions to full
    # point-cloud rendering (measured 2026-07-28: "-> ACTIVE (selected=False, peers=6)").
    # We now require the condition to hold continuously before trusting it.
    peer_fallback_since: float = -1.0
    peer_fallback_engaged: bool = False
    # A selected session normally has the grid subscriber plus one or more single-view PC
    # subscribers. If EXIT_SINGLE is lost, only the grid peer remains. After a bounded grace
    # period, demote that stale selection until its subscribers return.
    selected_peer_missing_since: float = -1.0
    selected_peer_lease_expired: bool = False
    # Last time the selector-thumbnail RGB topic was published. Used by
    # --rgb_thumbnail_fps to hold the grid feed at ~10 Hz while the point cloud runs at
    # full rate, which removes one 640x480 render per tick from the selected session.
    last_thumbnail_pub_t: float = 0.0
    # Last time each camera was actually rendered+submitted, used by the round-robin
    # scheduler. Fixed modulo cycling gave each camera a period of tick_dt * n_cams with
    # NO floor: at the measured 9-window worst case (tick 238 ms, 4 working cameras) that
    # is ~950 ms, and Unity deletes a point-cloud source after 1.0 s of silence — the
    # "cameras drop out one at a time then come back" loop. See --pc_max_source_age_s.
    last_cam_submit_t: Dict[str, float] = field(default_factory=dict)
    starvation_logged: bool = False
    # Growth-only per-camera high-water mark of the ACTUAL point count, used to declare an
    # honest capacity on the wire. Unity sizes its per-payload decode buffers from the
    # DECLARED capacity (GpuMergedPointCloudLoader.EnsureCapacity), so declaring the
    # 200k cap while sending ~5k points made it allocate and zero 3.2 MB per payload on its
    # NetMQ poller thread — ~576 MB/s at 3 cameras x 60 Hz, for ~11k live points. The wire
    # payload itself is sized by actual_count (GpuPointCloudContract), so an honest
    # declaration costs zero bandwidth.
    pc_declared_hwm: Dict[str, int] = field(default_factory=dict)
    pc_declared_logged: Dict[str, bool] = field(default_factory=dict)
    last_render_qpos: Optional[np.ndarray] = None
    pc_visit_started_t: Optional[float] = None
    pc_visit_publish_counts: Dict[str, int] = field(default_factory=dict)


class ActivePcSessionTracker:
    """Cross-process diagnostic for concurrently active point-cloud sessions."""

    def __init__(self, *, session_index: int, topic_port: int, port_step: int = 10):
        self.session_index = int(session_index)
        self.topic_port = int(topic_port)
        base_port = self.topic_port - self.session_index * int(port_step)
        self.directory = Path("/tmp/iilar_active_pc_sessions") / f"base_{base_port}"
        self.marker = self.directory / f"S{self.session_index:02d}_{os.getpid()}.json"
        self.active = False
        self.last_count = 0
        self._next_count_check = 0.0
        self._multi_active_since: Optional[float] = None
        self._multi_active_warned = False

    def set_active(self, active: bool, *, peers: int, reason: str) -> None:
        active = bool(active)
        if active == self.active:
            return
        self.active = active
        if active:
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                self.marker.write_text(
                    json.dumps({
                        "pid": os.getpid(),
                        "session_index": self.session_index,
                        "topic_port": self.topic_port,
                        "peers": int(peers),
                        "reason": str(reason),
                        "wall_t": time.time(),
                    }),
                    encoding="utf-8",
                )
            except Exception as exc:
                print(f"[ActivePcSessions][WARN] marker write failed: {exc}")
        else:
            try:
                self.marker.unlink(missing_ok=True)
            except Exception:
                pass
        print(
            f"[ActivePcSessions] S{self.session_index:02d} "
            f"{'ACTIVE' if active else 'IDLE'} peers={peers} reason={reason}"
        )
        self.poll(force=True)

    def poll(self, *, force: bool = False, warn_after_s: float = 2.0) -> int:
        now = time.time()
        if not force and now < self._next_count_check:
            return self.last_count
        self._next_count_check = now + 1.0
        count = 0
        try:
            for marker in self.directory.glob("S*.json"):
                try:
                    payload = json.loads(marker.read_text(encoding="utf-8"))
                    pid = int(payload.get("pid", -1))
                    os.kill(pid, 0)
                    count += 1
                except ProcessLookupError:
                    marker.unlink(missing_ok=True)
                except (OSError, ValueError, TypeError, json.JSONDecodeError):
                    marker.unlink(missing_ok=True)
        except OSError:
            count = int(self.active)

        self.last_count = count
        if count > 1:
            if self._multi_active_since is None:
                self._multi_active_since = now
            elif not self._multi_active_warned and now - self._multi_active_since >= warn_after_s:
                print(
                    f"[ActivePcSessions][ERROR] {count} sustained active point-cloud "
                    f"sessions in base group for {now - self._multi_active_since:.1f}s"
                )
                self._multi_active_warned = True
        else:
            self._multi_active_since = None
            self._multi_active_warned = False
        return count

    def count(self) -> int:
        return self.poll()

    def close(self) -> None:
        self.set_active(False, peers=0, reason="shutdown")


def resolve_sensor_activation(
    *,
    sensor_state: "SensorPublishState",
    selected: Optional[bool],
    peer_count: int,
    threshold: int,
    grace_s: float,
    selected_peer_idle_grace_s: float = 0.0,
    now: float,
    force_idle: bool = False,
    advance_state: bool = True,
) -> Tuple[bool, str]:
    """Decide whether this session's sensors should run at ACTIVE rate.

    Single source of truth for both `publish_sensor_frames()` (which sets the publish
    rate) and `point_cloud_publish_allowed()` (which gates the /pc topics). These used to
    carry separate copies of the rule; if they ever disagreed, point clouds would keep
    publishing on a session the rate limiter had already demoted.

    Rules, in order:
      * `force_idle` (policy handoff) always wins.
      * When selection is available (`--selected_sensor_activation`), ENTER_SINGLE /
        EXIT_SINGLE is AUTHORITATIVE.
      * The peer count is only a last-resort fallback for a dropped ENTER_SINGLE
        (lesson 41), and only after it has held for `grace_s`. Combined view keeps grid
        subscribers alive on every session and panels reconnect on an 8 s idle timer, so
        an instantaneous peer test promoted UNSELECTED sessions to full point-cloud
        rendering — the dominant cost measured on 2026-07-28.
      * `grace_s <= 0` disables the fallback entirely (strict selection).

    `advance_state=False` lets read-only callers evaluate the same rule without mutating
    the debounce timer.
    """
    peer_active = int(peer_count) >= int(threshold)

    if advance_state:
        if peer_active:
            if sensor_state.peer_fallback_since < 0.0:
                sensor_state.peer_fallback_since = now
        else:
            sensor_state.peer_fallback_since = -1.0

        if bool(selected) and not peer_active and selected_peer_idle_grace_s > 0.0:
            if sensor_state.selected_peer_missing_since < 0.0:
                sensor_state.selected_peer_missing_since = now
        else:
            sensor_state.selected_peer_missing_since = -1.0
            sensor_state.selected_peer_lease_expired = False

    since = sensor_state.peer_fallback_since
    peer_sustained = (
        peer_active and since >= 0.0 and (now - since) >= grace_s and grace_s > 0.0
    )

    if force_idle:
        return False, "force_idle"
    if selected is None:
        return peer_active, "peers"
    if bool(selected):
        missing_since = sensor_state.selected_peer_missing_since
        lease_expired = (
            selected_peer_idle_grace_s > 0.0
            and not peer_active
            and missing_since >= 0.0
            and (now - missing_since) >= selected_peer_idle_grace_s
        )
        if lease_expired:
            if advance_state:
                sensor_state.selected_peer_lease_expired = True
            return False, "selected_no_peers"
        return True, "selection"
    if peer_sustained:
        return True, "peer_fallback"
    return False, "not_selected"


# Set by main() so the module-level publish_sensor_frames() can label metrics with
# upstream staleness. One session per process.
_ACTIVE_POLICY_MIRROR = None
_ACTIVE_PC_SESSION_TRACKER = None

_PC_DECLARED_STEP = 4096          # round up to this, so the value settles instead of creeping
# Headroom matters for CORRECTNESS, not just churn: GpuPointCloudPipeline treats
# declared_capacity as the point BUDGET and subsamples anything above it
# (GpuPointCloudPipeline.py:556-559, :840-843), and encode_frame_into raises if
# actual > declared. 2x means the cloud has to double between frames before a single
# frame is thinned, and the high-water mark then grows on that very frame.
_PC_DECLARED_HEADROOM = 2.0
_PC_DECLARED_FLOOR = 4096
# Used only before any observation exists. Chosen well above the measured working range
# (2200-5500 points) so start-up frames are never thinned, while still being ~12x smaller
# than the 200000 cap that caused the 3.2 MB-per-payload allocation on the Quest.
_PC_DECLARED_INITIAL = 16384


def _resolve_declared_capacity(args, sensor_state, cam_name: str, cap: int) -> int:
    """Capacity to advertise for this camera's next point-cloud payload.

    Unity allocates its decode buffers from the DECLARED capacity, not from the actual point
    count (`GpuMergedPointCloudLoader.EnsureCapacity(header.DeclaredCapacity)`), and it
    throws the buffer away between payloads on the fast path. Declaring the flat
    `--pc_max_points` cap (200000) while sending 2000-5500 points therefore cost 3.2 MB of
    allocate-and-zero per payload on the Quest's poller thread — measured ~576 MB/s, which
    starved the receiver and produced a visibly frozen cloud at a stable frame rate.

    The wire payload is sized by `actual_count`, so a smaller declaration is free.

    Growth-only by construction: the value can rise when a cloud gets denser but never falls,
    so Unity's `EnsureCapacity` reallocates a handful of times during warm-up and then never
    again. Always clamped to `cap`, which keeps it under Unity's `maxPointsPerSource` — if it
    ever exceeded that, every payload would be silently dropped (lesson 21).
    """
    if str(getattr(args, "pc_declared_capacity_mode", "adaptive")) != "adaptive":
        return int(cap)

    hwm = int(sensor_state.pc_declared_hwm.get(cam_name, 0))
    if hwm <= 0:
        # Nothing observed yet: start well above the working range rather than at the cap,
        # so start-up frames are never thinned and no payload triggers the 3.2 MB
        # allocation on the headset.
        return min(int(cap), _PC_DECLARED_INITIAL)
    want = int(hwm * _PC_DECLARED_HEADROOM)
    want = ((want + _PC_DECLARED_STEP - 1) // _PC_DECLARED_STEP) * _PC_DECLARED_STEP
    return max(_PC_DECLARED_FLOOR, min(int(cap), want))


def _note_declared_capacity_observation(args, sensor_state, cam_name: str, actual_count: int,
                                        cap: int) -> None:
    """Feed the observed point count back into the high-water mark."""
    if str(getattr(args, "pc_declared_capacity_mode", "adaptive")) != "adaptive":
        return
    actual = int(actual_count)
    if actual <= 0:
        return
    prev = int(sensor_state.pc_declared_hwm.get(cam_name, 0))
    if actual <= prev:
        return
    sensor_state.pc_declared_hwm[cam_name] = actual
    settled = _resolve_declared_capacity(args, sensor_state, cam_name, cap)
    # Log each time the declaration actually moves, so the settled value is visible in the
    # unified log and can be compared against the old flat 200000.
    if sensor_state.pc_declared_logged.get(cam_name) != settled:
        sensor_state.pc_declared_logged[cam_name] = settled
        print(
            f"[PC][DeclaredCapacity] {cam_name}: observed max {actual} points -> declaring "
            f"{settled} (cap={cap}). Unity sizes its per-payload decode buffer from this; "
            f"the wire payload is unchanged.",
            flush=True,
        )


class IsolatedSensorPublisher:
    """Wraps SensorNode with a completely isolated zmq.Context for the PUB socket.

    Avoids ZMQ context singleton interference between:
    - zmq.asyncio.Context.instance() used by MujocoPublisher/XRNodeManager
    - zmq.Context.instance() used by the original SensorNode
    - Any zmq usage from the robot stack (intervene_base imports)

    The discovery (multicast UDP) and REP (service) functionality is delegated
    to the original SensorNode. Only the PUB socket is replaced with an isolated one.

    Sensor publishing is funneled through a dedicated TX thread with
    latest-per-topic coalescing. This keeps the control loop non-blocking while
    avoiding sustained drop storms when large point-cloud payloads are active.
    """

    def __init__(self, sensor_node: unified.SensorNode):
        self._inner = sensor_node
        # Create a completely isolated ZMQ context -- NOT the singleton.
        self._isolated_ctx = zmq.Context()

        # Close the original PUB socket to free the port.
        bind_addr = f"tcp://{self._inner.bind_ip}:{self._inner.topic_port}"
        self._inner._pub.close(linger=0)

        # Create new PUB socket from isolated context.
        self._pub = self._isolated_ctx.socket(zmq.PUB)
        self._pub.setsockopt(zmq.LINGER, 0)
        # Keep only a small, bounded transport cushion. The application queue already
        # coalesces latest-per-topic; a HWM of 2000 moved stale clouds into libzmq where
        # they could no longer be replaced and could represent seconds of old poses.
        self._pub.setsockopt(zmq.SNDHWM, 8)
        self._pub.setsockopt(zmq.SNDTIMEO, 2)
        # The close() above is asynchronous: ZMQ's I/O thread releases the
        # underlying fd, and under heavy system load (many sessions starting
        # at once) that thread can be scheduled late enough that this bind()
        # races its own predecessor and hits EADDRINUSE -- not an external
        # process, just this socket's own not-yet-finished teardown. Retry
        # with backoff instead of crashing the whole session over a
        # self-inflicted race that always clears within ~1s.
        _bind_attempts = 20
        _bind_delay = 0.1
        for _attempt in range(1, _bind_attempts + 1):
            try:
                self._pub.bind(bind_addr)
                break
            except zmq.error.ZMQError as exc:
                if exc.errno != errno.EADDRINUSE or _attempt == _bind_attempts:
                    raise
                print(
                    f"[IsolatedSensorPublisher] bind({bind_addr}) busy "
                    f"(own socket still releasing the port), retry "
                    f"{_attempt}/{_bind_attempts}..."
                )
                time.sleep(_bind_delay)

        # Replace inner node's reference so REP loop info stays consistent.
        self._inner._pub = self._pub

        self._publish_count = 0
        self._publish_errors = 0
        self._enqueue_count = 0
        self._coalesced_count = 0

        self._pending_lock = threading.Lock()
        self._pending_by_topic: Dict[str, bytes] = {}
        self._topic_enqueue_count: Dict[str, int] = {}
        self._topic_publish_count: Dict[str, int] = {}
        self._topic_last_enqueue_t: Dict[str, float] = {}
        self._topic_last_publish_t: Dict[str, float] = {}
        self._tx_wake = threading.Event()
        self._tx_stop = threading.Event()
        self._tx_thread = threading.Thread(
            target=self._tx_loop,
            daemon=True,
            name="SensorPubTX",
        )
        self._last_no_peer_warn_t = 0.0

        # PUB socket monitor telemetry to verify actual subscriber connections.
        self._monitor_lock = threading.Lock()
        self._monitor_stop = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_sock: Optional[zmq.Socket] = None
        self._monitor_endpoint = f"inproc://sensorpub-monitor-{id(self)}"
        self._monitor_event_counts: Dict[str, int] = {}
        self._monitor_last_event = ""
        self._monitor_last_endpoint = ""
        self._monitor_last_t = 0.0
        self._peer_count_est = 0
        self._setup_monitor()

        print(f"[IsolatedSensorPublisher] PUB socket bound to {bind_addr} on isolated zmq.Context")

    def start(self):
        self._inner.start()
        if self._monitor_thread is not None and not self._monitor_thread.is_alive():
            self._monitor_thread.start()
        self._tx_thread.start()

    def peer_count(self) -> int:
        """Return current estimated peer count (subscribers connected to this
        PUB socket). Tracked via ZMQ socket monitoring; +1 on EVENT_ACCEPTED,
        -1 on EVENT_DISCONNECTED. Cross-platform (Linux/Windows/macOS) — relies
        only on libzmq's monitor API."""
        with self._monitor_lock:
            return int(self._peer_count_est)

    @staticmethod
    def _event_name(event_code: int) -> str:
        event_names = {
            int(getattr(zmq, "EVENT_ACCEPTED", -1)): "ACCEPTED",
            int(getattr(zmq, "EVENT_DISCONNECTED", -1)): "DISCONNECTED",
            int(getattr(zmq, "EVENT_CONNECTED", -1)): "CONNECTED",
            int(getattr(zmq, "EVENT_CONNECT_DELAYED", -1)): "CONNECT_DELAYED",
            int(getattr(zmq, "EVENT_CONNECT_RETRIED", -1)): "CONNECT_RETRIED",
            int(getattr(zmq, "EVENT_LISTENING", -1)): "LISTENING",
            int(getattr(zmq, "EVENT_BIND_FAILED", -1)): "BIND_FAILED",
            int(getattr(zmq, "EVENT_CLOSED", -1)): "CLOSED",
            int(getattr(zmq, "EVENT_CLOSE_FAILED", -1)): "CLOSE_FAILED",
            int(getattr(zmq, "EVENT_MONITOR_STOPPED", -1)): "MONITOR_STOPPED",
        }
        return event_names.get(int(event_code), f"EVENT_{int(event_code)}")

    def _setup_monitor(self):
        try:
            self._pub.monitor(self._monitor_endpoint, zmq.EVENT_ALL)
            self._monitor_sock = self._isolated_ctx.socket(zmq.PAIR)
            self._monitor_sock.setsockopt(zmq.LINGER, 0)
            self._monitor_sock.connect(self._monitor_endpoint)
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True,
                name="SensorPubMon",
            )
        except Exception as e:
            self._monitor_sock = None
            self._monitor_thread = None
            print(f"[IsolatedSensorPublisher][WARN] Failed to start PUB monitor: {e}")

    def _monitor_loop(self):
        try:
            from zmq.utils.monitor import recv_monitor_message
        except Exception as e:
            print(f"[IsolatedSensorPublisher][WARN] ZMQ monitor API unavailable: {e}")
            return

        while not self._monitor_stop.is_set():
            sock = self._monitor_sock
            if sock is None:
                return
            try:
                evt = recv_monitor_message(sock, flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.02)
                continue
            except Exception:
                if not self._monitor_stop.is_set():
                    time.sleep(0.05)
                continue

            event_code = int(evt.get("event", 0))
            endpoint_raw = evt.get("endpoint", b"")
            if isinstance(endpoint_raw, (bytes, bytearray)):
                endpoint = endpoint_raw.decode("utf-8", errors="ignore")
            else:
                endpoint = str(endpoint_raw)
            event_name = self._event_name(event_code)
            now = time.time()

            with self._monitor_lock:
                self._monitor_last_event = event_name
                self._monitor_last_endpoint = endpoint
                self._monitor_last_t = now
                self._monitor_event_counts[event_name] = int(
                    self._monitor_event_counts.get(event_name, 0)
                ) + 1
                if event_code == int(getattr(zmq, "EVENT_ACCEPTED", -1)):
                    self._peer_count_est += 1
                    print(
                        "[IsolatedSensorPublisher] Subscriber accepted: "
                        f"endpoint={endpoint} peers={self._peer_count_est}"
                    )
                elif event_code == int(getattr(zmq, "EVENT_DISCONNECTED", -1)):
                    self._peer_count_est = max(0, self._peer_count_est - 1)
                    print(
                        "[IsolatedSensorPublisher] Subscriber disconnected: "
                        f"endpoint={endpoint} peers={self._peer_count_est}"
                    )

    @staticmethod
    def _freeze_payload(payload) -> bytes:
        # PC payloads can be memoryviews into reusable staging buffers.
        # Freeze to immutable bytes so queued data cannot be overwritten before TX.
        if isinstance(payload, bytes):
            return payload
        if isinstance(payload, memoryview):
            return payload.tobytes()
        return bytes(payload)

    def publish(self, topic: str, payload: bytes):
        payload_bytes = self._freeze_payload(payload)
        now = time.time()
        with self._pending_lock:
            self._enqueue_count += 1
            if topic in self._pending_by_topic:
                self._coalesced_count += 1
            self._pending_by_topic[topic] = payload_bytes
            self._topic_enqueue_count[topic] = int(self._topic_enqueue_count.get(topic, 0)) + 1
            self._topic_last_enqueue_t[topic] = now
        self._tx_wake.set()

    def _tx_loop(self):
        while not self._tx_stop.is_set():
            self._tx_wake.wait(timeout=0.01)
            self._tx_wake.clear()

            with self._pending_lock:
                if not self._pending_by_topic:
                    continue
                batch = list(self._pending_by_topic.items())
                self._pending_by_topic.clear()

            for topic, payload in batch:
                if self._tx_stop.is_set():
                    return
                try:
                    # copy=True avoids zero-copy lifetime hazards across threads.
                    self._pub.send_multipart([topic.encode("utf-8"), payload], copy=True)
                    self._publish_count += 1
                    now = time.time()
                    with self._pending_lock:
                        self._topic_publish_count[topic] = int(self._topic_publish_count.get(topic, 0)) + 1
                        self._topic_last_publish_t[topic] = now
                    if now - self._last_no_peer_warn_t > 5.0:
                        with self._monitor_lock:
                            peer_count = int(self._peer_count_est)
                        if peer_count <= 0:
                            self._last_no_peer_warn_t = now
                            print(
                                "[IsolatedSensorPublisher][WARN] No subscriber ACCEPTED events observed yet. "
                                "If Quest logs show repeated idle reconnects, verify publisherIp/route/firewall."
                            )
                except zmq.Again:
                    self._publish_errors += 1
                    if self._publish_errors % 100 == 1:
                        print(
                            "[IsolatedSensorPublisher] ZMQ send timeout/dropped. "
                            f"errors={self._publish_errors} sent={self._publish_count}"
                        )
                except zmq.ZMQError as e:
                    self._publish_errors += 1
                    if self._publish_errors % 50 == 1:
                        print(
                            f"[IsolatedSensorPublisher] ZMQ send error: {e} "
                            f"errors={self._publish_errors}"
                        )

    def drop_pending_matching(self, predicate) -> int:
        """Drop queued-but-not-sent payloads matching predicate(topic)."""
        dropped = 0
        with self._pending_lock:
            for topic in list(self._pending_by_topic.keys()):
                try:
                    should_drop = bool(predicate(topic))
                except Exception:
                    should_drop = False
                if should_drop:
                    self._pending_by_topic.pop(topic, None)
                    dropped += 1
        return dropped

    def drop_pending_pc(self) -> int:
        return self.drop_pending_matching(lambda topic: str(topic).endswith("/pc"))

    def stop(self):
        self._tx_stop.set()
        self._tx_wake.set()
        if self._tx_thread.is_alive():
            self._tx_thread.join(timeout=2.0)
        self._monitor_stop.set()
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=1.0)
        self._inner.stop()
        try:
            self._pub.disable_monitor()
        except Exception:
            pass
        try:
            if self._monitor_sock is not None:
                self._monitor_sock.close(linger=0)
        except Exception:
            pass
        try:
            self._pub.close(linger=0)
        except Exception:
            pass
        try:
            self._isolated_ctx.term()
        except Exception:
            pass

    def get_diagnostics(self) -> dict:
        with self._pending_lock:
            pending_topics = len(self._pending_by_topic)
            pending_topic_names = sorted(self._pending_by_topic.keys())
            now = time.time()
            if pending_topic_names:
                oldest_pending_enqueue_t = min(
                    float(self._topic_last_enqueue_t.get(topic, now)) for topic in pending_topic_names
                )
                oldest_pending_age_s = max(0.0, now - oldest_pending_enqueue_t)
            else:
                oldest_pending_age_s = 0.0
        with self._monitor_lock:
            monitor_last_event = self._monitor_last_event
            monitor_last_endpoint = self._monitor_last_endpoint
            monitor_event_counts = dict(self._monitor_event_counts)
            peer_count_est = int(self._peer_count_est)
            if self._monitor_last_t > 0.0:
                monitor_last_age_s = max(0.0, time.time() - float(self._monitor_last_t))
            else:
                monitor_last_age_s = 0.0
        return {
            "publish_count": self._publish_count,
            "publish_errors": self._publish_errors,
            "enqueue_count": self._enqueue_count,
            "coalesced_count": self._coalesced_count,
            "pending_topics": pending_topics,
            "oldest_pending_age_s": oldest_pending_age_s,
            "pending_topic_sample": pending_topic_names[:3],
            "topic_enqueue_counts": dict(self._topic_enqueue_count),
            "topic_publish_counts": dict(self._topic_publish_count),
            "peer_count_est": peer_count_est,
            "monitor_last_event": monitor_last_event,
            "monitor_last_endpoint": monitor_last_endpoint,
            "monitor_last_age_s": monitor_last_age_s,
            "monitor_event_counts": monitor_event_counts,
        }


class RobotBridge:
    def __init__(self, robot_key: str):
        intervene_record, control_type_enum = load_robot_stack()
        if robot_key not in intervene_record.ROBOTS:
            raise KeyError(
                f"Unknown robot_key '{robot_key}'. Available: {sorted(intervene_record.ROBOTS.keys())}"
            )
        self.robot_key = robot_key
        self.robot = intervene_record.ROBOTS[robot_key]
        self.control_type_enum = control_type_enum
        self.mode = None
        self.q_prev: Optional[np.ndarray] = None
        self._last_gripper_width_cmd: Optional[float] = None
        self._last_gripper_cmd_t: float = 0.0
        self._last_gripper_error_t: float = 0.0
        self._gripper_future = None

    def _swap_policy_inplace(self, new_mode) -> None:
        """Swap the robot control policy WITHOUT disconnecting.

        Calls FrankaArm.set_policy() which creates a new policy and sends it
        via send_torch_policy(). Polymetis atomically replaces the running
        policy on the real-time server. No close()/reset() is called,
        so the robot stays at its current pose.
        """
        arm = self.robot.robot_arm
        if arm.robot is None:
            raise RuntimeError("Robot arm not connected, cannot swap policy in-place")
        self._last_gripper_width_cmd = None
        self._gripper_future = None
        arm.control_type = new_mode
        arm.set_policy()
        self.mode = new_mode
        self.q_prev = None
        print(f"[Integration] Policy swapped in-place to {new_mode.name}")

    def ensure_mode(self, mode) -> None:
        if self.mode == mode and getattr(self.robot, "is_connected", False):
            return
        # Prefer in-place swap when already connected (avoids close→reset→home jump)
        if getattr(self.robot, "is_connected", False) and self.robot.robot_arm.robot is not None:
            self._swap_policy_inplace(mode)
            self._last_gripper_width_cmd = None
            self._gripper_future = None
            return
        # First connection: full connect cycle
        self.robot.connect(mode)
        self.mode = mode
        self.q_prev = None
        self._last_gripper_width_cmd = None
        self._gripper_future = None
        print(f"[Integration] Robot {self.robot_key} connected in {mode.name}")

    def read_state(self) -> Tuple[np.ndarray, Optional[np.ndarray], float]:
        state = self.robot.robot_arm.get_state()
        q_real = state.joint_pos.detach().cpu().numpy().astype(np.float64)
        dq_real = getattr(state, "joint_vel", None)
        if dq_real is not None:
            dq_real = dq_real.detach().cpu().numpy().astype(np.float64)
        gripper_width = float(self.robot.robot_gripper.get_sensors().item())
        return q_real, dq_real, gripper_width
    def hold_pose(self, q_hold: np.ndarray, finger_hold: float, *, dt: float, max_dq: float) -> None:
        self.ensure_mode(self.control_type_enum.HYBRID_JOINT_IMPEDANCE_CONTROL)
        q_hold = np.asarray(q_hold, dtype=np.float64).reshape(7)
        if self.q_prev is None:
            try:
                q_now, _, _ = self.read_state()
                self.q_prev = np.asarray(q_now, dtype=np.float64).reshape(7)
            except Exception as e:
                print(f"[Integration][WARN] Could not seed hold clamp from robot state: {e}")
                self.q_prev = q_hold.copy()
        q_cmd = clamp_step(q_hold, self.q_prev, max_dq=max_dq, dt=dt)
        self.q_prev = q_cmd.copy()
        self.robot.robot_arm.apply_commands(q_cmd)
        self._apply_gripper_width(float(np.clip(finger_hold * 2.0, 0.0, 0.08)))

    def start_human_guidance(self) -> Tuple[bool, str]:
        """Switch to HUMAN_CONTROL with settling and validation.

        Uses in-place policy swap to avoid the close()/reset() cycle that
        causes the robot to jump to its home pose.

        Returns (success, message).
        """
        SETTLE_CYCLES = 5
        SETTLE_DT = 0.02
        JUMP_THRESHOLD_RAD = 0.15
        RECOVERY_HOLD_CYCLES = 20
        RECOVERY_HOLD_DT = 0.008

        # Phase 1: Reinforce current pose
        q_anchor, _, _ = self.read_state()
        for _ in range(SETTLE_CYCLES):
            self.robot.robot_arm.apply_commands(
                np.asarray(q_anchor, dtype=np.float64).reshape(7)
            )
            time.sleep(SETTLE_DT)

        q_pre, _, _ = self.read_state()
        print(f"[Integration] Pre-switch delta from anchor: "
              f"{np.linalg.norm(q_pre - q_anchor):.4f} rad")

        # Phase 2: Swap to HUMAN_CONTROL in-place (no disconnect)
        try:
            self._swap_policy_inplace(self.control_type_enum.HUMAN_CONTROL)
        except Exception as e:
            print(f"[Integration][ERROR] HUMAN_CONTROL swap failed: {e}")
            try:
                self._swap_policy_inplace(self.control_type_enum.HYBRID_JOINT_IMPEDANCE_CONTROL)
                self.robot.robot_arm.apply_commands(q_anchor)
            except Exception:
                pass
            return False, f"Policy swap failed: {e}"

        # Phase 3: Post-settle
        time.sleep(SETTLE_DT * 3)

        # Phase 4: Validate
        q_after, _, _ = self.read_state()
        jump_norm = np.linalg.norm(q_after - q_anchor)
        print(f"[Integration] Post-switch delta: {jump_norm:.4f} rad")

        if jump_norm > JUMP_THRESHOLD_RAD:
            print(f"[Integration][WARN] HUMAN_CONTROL caused jump of {jump_norm:.4f} rad; "
                  f"recovering to HYBRID hold.")
            try:
                self._swap_policy_inplace(self.control_type_enum.HYBRID_JOINT_IMPEDANCE_CONTROL)
                for _ in range(RECOVERY_HOLD_CYCLES):
                    self.robot.robot_arm.apply_commands(
                        np.asarray(q_anchor, dtype=np.float64).reshape(7)
                    )
                    time.sleep(RECOVERY_HOLD_DT)
                q_recovered, _, _ = self.read_state()
                recovery_delta = np.linalg.norm(q_recovered - q_anchor)
                print(f"[Integration] Recovery complete. Delta: {recovery_delta:.4f} rad")
            except Exception as e:
                return False, f"Recovery failed: {e}"
            return False, f"Jump detected ({jump_norm:.4f} rad), recovered to HYBRID hold"

        print("[Integration] HUMAN_CONTROL active. Guide the robot for the suffix.")
        return True, "HUMAN_CONTROL active"

    def mirror_command(
        self,
        q_cmd: np.ndarray,
        *,
        dt: float,
        max_dq: float,
        grip_width: Optional[float],
    ) -> None:
        self.ensure_mode(self.control_type_enum.HYBRID_JOINT_IMPEDANCE_CONTROL)
        q_cmd = np.asarray(q_cmd, dtype=np.float64).reshape(7)
        if self.q_prev is None:
            try:
                q_now, _, _ = self.read_state()
                self.q_prev = np.asarray(q_now, dtype=np.float64).reshape(7)
            except Exception as e:
                print(f"[Integration][WARN] Could not seed mirror clamp from robot state: {e}")
                self.q_prev = q_cmd.copy()
        q_cmd = clamp_step(q_cmd, self.q_prev, max_dq=max_dq, dt=dt)
        self.q_prev = q_cmd.copy()
        self.robot.robot_arm.apply_commands(q_cmd)
        if grip_width is not None:
            self._apply_gripper_width(grip_width)

    def _apply_gripper_width(self, width_m: float) -> None:
        hand = self.robot.robot_gripper
        target = float(np.clip(width_m, hand.min_width, hand.max_width))
        now = time.time()

        if (
            self._last_gripper_width_cmd is not None
            and abs(target - self._last_gripper_width_cmd) < float(GRIPPER_TRACK_EPS_M)
            and (now - self._last_gripper_cmd_t) < float(GRIPPER_CMD_PERIOD_S)
        ):
            return

        try:
            low_level = getattr(hand, "robot", None)
            if low_level is not None and hasattr(low_level, "goto"):
                pool = getattr(hand, "pool", None)
                if pool is not None:
                    if self._gripper_future is not None and not self._gripper_future.done():
                        return
                    self._gripper_future = pool.submit(
                        low_level.goto,
                        target,
                        float(GRIPPER_CMD_SPEED),
                        float(GRIPPER_CMD_FORCE),
                    )
                else:
                    low_level.goto(
                        target,
                        float(GRIPPER_CMD_SPEED),
                        float(GRIPPER_CMD_FORCE),
                    )
            else:
                current_width = float(hand.get_sensors().item())
                err = target - current_width
                if abs(err) < float(GRIPPER_TRACK_EPS_M):
                    return
                cmd = 1.0 if err > 0.0 else -1.0
                hand.apply_commands(
                    cmd,
                    speed=float(GRIPPER_CMD_SPEED),
                    force=float(GRIPPER_CMD_FORCE),
                )
        except Exception as e:
            if (now - self._last_gripper_error_t) > 1.0:
                print(f"[Integration][WARN] Gripper command failed: {e}")
                self._last_gripper_error_t = now
            return

        self._last_gripper_width_cmd = target
        self._last_gripper_cmd_t = now

    def close(self) -> None:
        if getattr(self.robot, "is_connected", False):
            self.robot.close()
        self.mode = None
        self.q_prev = None
        self._last_gripper_width_cmd = None
        self._last_gripper_cmd_t = 0.0
        self._gripper_future = None


def clamp_step(q_cmd: np.ndarray, q_prev: np.ndarray, *, max_dq: float, dt: float) -> np.ndarray:
    q_cmd_arr = np.asarray(q_cmd, dtype=np.float64)
    q_prev_arr = np.asarray(q_prev, dtype=np.float64)
    dq = (q_cmd_arr - q_prev_arr) / max(float(dt), 1e-6)

    limit = np.full_like(dq, float(max_dq), dtype=np.float64)
    if limit.size >= 7:
        limit[6] = float(max_dq) * float(JOINT7_MAX_DQ_SCALE)

    dq = np.clip(dq, -limit, limit)
    return q_prev_arr + dq * float(dt)


class RobotThread:
    """Background thread for robot communication.

    Decouples blocking robot I/O (network calls to the Franka arm) from the
    main loop so that sensor publishing (wrist camera, point clouds to Quest)
    is never starved by network latency.

    The public methods mirror the RobotBridge API but are non-blocking
    (fire-and-forget) for high-frequency commands (mirror, hold) and
    blocking only for one-shot transitions (ensure_mode, start_human_guidance).
    """

    def __init__(self, robot_key: str):
        self.bridge = RobotBridge(robot_key)
        self._cmd_q: _queue.Queue = _queue.Queue()
        self._motion_lock = threading.Lock()
        self._latest_motion_cmd: Optional[Tuple] = None
        self._latest_motion_seq = 0
        self._last_applied_motion_seq = 0
        self._motion_coalesced = 0
        self._motion_applied = 0
        self._motion_period_s = 1.0 / 120.0
        self._next_motion_t = 0.0
        self._last_motion_age_s = 0.0
        self._last_motion_kind = ""
        self._state_lock = threading.Lock()
        self._latest_state: Dict = {"q": None, "dq": None, "grip": 0.0, "t": 0.0}
        self._guidance_lock = threading.Lock()
        self._guidance_pending = False
        self._guidance_done = False
        self._guidance_success = False
        self._guidance_message = ""
        self._reading = False
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="RobotComm"
        )
        self._thread.start()

    # --- Public API (called from main thread) ---

    @property
    def control_type_enum(self):
        return self.bridge.control_type_enum

    def mirror_command(self, q_cmd, *, dt, max_dq, grip_width):
        """Non-blocking: publish latest mirror target, replacing stale targets."""
        self._set_motion_target(("mirror", q_cmd.copy(), float(max_dq), grip_width, time.time()), dt)

    def hold_pose(self, q_hold, finger_hold, *, dt: float, max_dq: float):
        """Non-blocking: publish latest hold target, replacing stale targets."""
        self._set_motion_target(("hold", q_hold.copy(), float(finger_hold), float(max_dq), time.time()), dt)

    def _set_motion_target(self, cmd: Tuple, dt: float) -> None:
        safe_dt = max(float(dt), 1e-4)
        with self._motion_lock:
            if self._latest_motion_cmd is not None and self._latest_motion_seq != self._last_applied_motion_seq:
                self._motion_coalesced += 1
            self._latest_motion_seq += 1
            self._latest_motion_cmd = cmd
            self._motion_period_s = safe_dt

    def clear_motion(self) -> None:
        with self._motion_lock:
            self._latest_motion_cmd = None
            self._next_motion_t = 0.0

    def start_human_guidance(self):
        """Blocking: switch to HUMAN_CONTROL and wait for completion."""
        evt = threading.Event()
        result = {"success": False, "message": "Timed out waiting for HUMAN_CONTROL transition"}
        self._cmd_q.put(("human_blocking", evt, result))
        if not evt.wait(timeout=15.0):
            return False, "Timed out waiting for HUMAN_CONTROL transition"
        return bool(result.get("success", False)), str(result.get("message", "Unknown result"))

    def start_human_guidance_async(self) -> Tuple[bool, str]:
        """Non-blocking: begin HUMAN_CONTROL transition and return immediately."""
        with self._guidance_lock:
            if self._guidance_pending:
                return False, "Guidance transition already in progress"
            self._guidance_pending = True
            self._guidance_done = False
            self._guidance_success = False
            self._guidance_message = ""
        self._cmd_q.put(("human_async",))
        return True, "Guidance transition started"

    def is_guidance_transition_done(self) -> Tuple[bool, bool, str]:
        """Poll async guidance transition state.

        Returns:
            done: transition has completed and result is available
            success: transition result when done=True
            message: status/result text
        """
        with self._guidance_lock:
            if self._guidance_pending:
                return False, False, "Guidance transition in progress"
            if not self._guidance_done:
                return False, False, "No guidance transition result available"
            done_success = bool(self._guidance_success)
            done_message = str(self._guidance_message)
            self._guidance_done = False
            self._guidance_message = ""
            return True, done_success, done_message

    def ensure_mode(self, mode):
        """Blocking: switch robot mode and wait for completion."""
        evt = threading.Event()
        self._cmd_q.put(("mode", mode, evt))
        evt.wait(timeout=10.0)

    def set_reading(self, active: bool):
        """Enable/disable continuous state reading for replan recording."""
        self._reading = active
        if not active:
            self.clear_motion()

    def get_state(self):
        """Get latest robot state (thread-safe). Returns (q, dq, grip)."""
        with self._state_lock:
            s = self._latest_state
            q = s["q"].copy() if s["q"] is not None else None
            dq = s["dq"].copy() if s["dq"] is not None else None
            return q, dq, s["grip"]

    def motion_diagnostics(self) -> Dict[str, float]:
        with self._motion_lock:
            pending = self._latest_motion_cmd is not None
            latest_seq = int(self._latest_motion_seq)
            applied_seq = int(self._last_applied_motion_seq)
            return {
                "pending": float(1 if pending else 0),
                "latest_seq": float(latest_seq),
                "applied_seq": float(applied_seq),
                "coalesced": float(self._motion_coalesced),
                "applied": float(self._motion_applied),
                "last_age_s": float(self._last_motion_age_s),
                "period_s": float(self._motion_period_s),
            }

    def close(self):
        """Stop the background thread and close the robot connection."""
        self._reading = False
        self._stop.set()
        self._thread.join(timeout=5.0)
        try:
            self.bridge.close()
        except Exception:
            pass

    # --- Background thread loop ---

    def _loop(self):
        while not self._stop.is_set():
            try:
                cmd = self._cmd_q.get(timeout=0.001)
                self._dispatch(cmd)
            except _queue.Empty:
                pass
            self._dispatch_latest_motion_if_due()
            if self._reading:
                try:
                    q, dq, gw = self.bridge.read_state()
                    with self._state_lock:
                        self._latest_state["q"] = q
                        self._latest_state["dq"] = dq
                        self._latest_state["grip"] = gw
                        self._latest_state["t"] = time.time()
                except Exception as e:
                    print(f"[RobotThread] read_state error: {e}")

    def _dispatch_latest_motion_if_due(self) -> None:
        now = time.time()
        with self._motion_lock:
            cmd = self._latest_motion_cmd
            if cmd is None:
                return
            period_s = max(float(self._motion_period_s), 1e-4)
            if self._next_motion_t <= 0.0:
                self._next_motion_t = now
            if now < self._next_motion_t:
                return
            self._next_motion_t = now + period_s
            seq = int(self._latest_motion_seq)

        kind = cmd[0]
        try:
            if kind == "mirror":
                _, q_cmd, max_dq, grip_width, enqueue_t = cmd
                self.bridge.mirror_command(q_cmd, dt=period_s, max_dq=max_dq, grip_width=grip_width)
            elif kind == "hold":
                _, q_hold, finger_hold, max_dq, enqueue_t = cmd
                self.bridge.hold_pose(q_hold, finger_hold, dt=period_s, max_dq=max_dq)
            else:
                return
            with self._motion_lock:
                self._last_applied_motion_seq = seq
                self._motion_applied += 1
                self._last_motion_age_s = max(0.0, time.time() - float(enqueue_t))
                self._last_motion_kind = str(kind)
        except Exception as e:
            print(f"[RobotThread] {kind} latest-target error: {e}")

    def _dispatch(self, cmd):
        kind = cmd[0]
        try:
            if kind == "human_blocking":
                self.clear_motion()
                _, evt, result_holder = cmd
                success, message = self.bridge.start_human_guidance()
                result_holder["success"] = bool(success)
                result_holder["message"] = str(message)
                evt.set()
            elif kind == "human_async":
                self.clear_motion()
                success, message = self.bridge.start_human_guidance()
                with self._guidance_lock:
                    self._guidance_pending = False
                    self._guidance_done = True
                    self._guidance_success = bool(success)
                    self._guidance_message = str(message)
            elif kind == "mode":
                self.clear_motion()
                self.bridge.ensure_mode(cmd[1])
                cmd[2].set()
        except Exception as e:
            print(f"[RobotThread] {kind} error: {e}")
            # Signal done events even on error to avoid deadlocking main thread
            if kind == "human_blocking":
                cmd[2]["success"] = False
                cmd[2]["message"] = f"Thread dispatch error: {e}"
                cmd[1].set()
            elif kind == "human_async":
                with self._guidance_lock:
                    self._guidance_pending = False
                    self._guidance_done = True
                    self._guidance_success = False
                    self._guidance_message = f"Thread dispatch error: {e}"
            elif kind == "mode":
                cmd[2].set()


def resolve_replay_robot_bindings(model: mujoco.MjModel) -> ReplayRobotBindings:
    arm_qpos_idx, arm_dof_idx, arm_ctrl_idx = unified.get_robot_arm_indices(model)
    if np.any(arm_qpos_idx < 0) or np.any(arm_dof_idx < 0):
        print(
            "[Integration][WARN] Could not fully resolve joint1..joint7 indices. "
            "Robot replay may be incomplete."
        )
    if np.any(arm_ctrl_idx < 0):
        print(
            "[Integration][WARN] Could not map all arm actuators by joint. "
            "Falling back to ctrl[0:7] where needed."
        )

    finger_qpos_idx = np.full(2, -1, dtype=np.int32)
    finger_dof_idx = np.full(2, -1, dtype=np.int32)
    finger_qpos_range = np.zeros((2, 2), dtype=np.float64)
    for i, joint_name in enumerate(["finger_joint1", "finger_joint2"]):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            continue
        finger_qpos_idx[i] = int(model.jnt_qposadr[jid])
        finger_dof_idx[i] = int(model.jnt_dofadr[jid])
        finger_qpos_range[i, :] = np.asarray(model.jnt_range[jid], dtype=np.float64)

    finger_ctrl_idx = -1
    finger_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "finger_joint1")
    if finger_jid >= 0:
        for actuator_id in range(model.nu):
            if int(model.actuator_trnid[actuator_id, 0]) == finger_jid:
                finger_ctrl_idx = actuator_id
                break
    finger_ctrl_range = np.zeros(2, dtype=np.float64)
    if 0 <= int(finger_ctrl_idx) < model.nu:
        finger_ctrl_range[:] = np.asarray(model.actuator_ctrlrange[int(finger_ctrl_idx)], dtype=np.float64)
    finger_ctrl_is_width = bool(
        finger_ctrl_range.shape[0] >= 2
        and float(finger_ctrl_range[1]) >= float(GRIPPER_FULL_WIDTH_CTRL_THRESHOLD_M)
    )
    if np.any(finger_qpos_idx < 0):
        print(
            "[Integration][WARN] Could not fully resolve finger_joint1/finger_joint2 indices. "
            "Gripper replay may be incomplete."
        )
    if finger_ctrl_idx < 0:
        print(
            "[Integration][WARN] Could not resolve the finger actuator index. "
            "Gripper ctrl replay will be skipped."
        )
    else:
        print(
            "[Integration] Gripper binding: "
            f"finger_ctrl_idx={finger_ctrl_idx} "
            f"ctrlrange=({finger_ctrl_range[0]:.4f}, {finger_ctrl_range[1]:.4f}) "
            f"mode={'full-width' if finger_ctrl_is_width else 'half-width'}"
        )

    return ReplayRobotBindings(
        arm_qpos_idx=arm_qpos_idx,
        arm_dof_idx=arm_dof_idx,
        arm_ctrl_idx=arm_ctrl_idx,
        finger_qpos_idx=finger_qpos_idx,
        finger_dof_idx=finger_dof_idx,
        finger_ctrl_idx=int(finger_ctrl_idx),
        finger_ctrl_range=finger_ctrl_range,
        finger_qpos_range=finger_qpos_range,
        finger_ctrl_is_width=finger_ctrl_is_width,
    )


def settle_scene_seed(args, model: mujoco.MjModel, data: mujoco.MjData) -> SceneSeedState:
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    settle_steps = max(
        1,
        int(np.ceil(float(SCENE_SETTLE_DURATION_S) / max(float(model.opt.timestep), 1e-6))),
    )
    for _ in range(settle_steps):
        mujoco.mj_step(model, data)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    table_plane = unified.resolve_table_plane(
        model=model,
        data=data,
        geom_candidates=args.pc_table_geoms,
        body_name=args.pc_table_body,
        manual_plane_z=args.pc_table_plane_z,
        body_top_offset=args.pc_table_body_top_offset,
    )
    table_top_z = float(table_plane.point[2]) if table_plane is not None else None

    body_world_pos: Dict[str, np.ndarray] = {}
    body_world_quat: Dict[str, np.ndarray] = {}
    for body_name in SCENE_DIAGNOSTIC_BODIES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if bid < 0:
            continue
        body_world_pos[body_name] = np.asarray(data.xpos[bid], dtype=np.float64).copy()
        body_world_quat[body_name] = np.asarray(data.xquat[bid], dtype=np.float64).copy()

    return SceneSeedState(
        qpos=np.asarray(data.qpos, dtype=np.float64).copy(),
        qvel=np.asarray(data.qvel, dtype=np.float64).copy(),
        ctrl=np.asarray(data.ctrl, dtype=np.float64).copy(),
        table_plane=table_plane,
        table_top_z=table_top_z,
        body_world_pos=body_world_pos,
        body_world_quat=body_world_quat,
        settle_steps=settle_steps,
        settle_duration_s=float(settle_steps) * float(model.opt.timestep),
    )


def build_scene_seed_diagnostics(
    *,
    model: mujoco.MjModel,
    scene_seed: SceneSeedState,
    trajectory: ReplayTrajectory,
    replay_bindings: ReplayRobotBindings,
) -> List[str]:
    lines: List[str] = []
    if scene_seed.table_top_z is not None:
        plane_source = scene_seed.table_plane.source if scene_seed.table_plane is not None else "<none>"
        lines.append(
            f"  settled_scene: table_top_z={scene_seed.table_top_z:.4f} "
            f"source={plane_source} settle_steps={scene_seed.settle_steps}"
        )

    for body_name in SCENE_DIAGNOSTIC_BODIES:
        pos = scene_seed.body_world_pos.get(body_name)
        quat = scene_seed.body_world_quat.get(body_name)
        if pos is None:
            continue
        quat_txt = "" if quat is None else f" quat={np.round(quat, 4)}"
        lines.append(f"  settled_{body_name}=pos={np.round(pos, 4)}{quat_txt}")

    if trajectory.qpos_sim.ndim == 2 and trajectory.qpos_sim.shape[1] == model.nq:
        robot_qpos_mask = np.zeros((model.nq,), dtype=bool)
        valid_robot_qpos_idx = replay_bindings.robot_qpos_idx
        if valid_robot_qpos_idx.size > 0:
            robot_qpos_mask[valid_robot_qpos_idx] = True

        raw_qpos0 = np.asarray(trajectory.qpos_sim[0], dtype=np.float64).reshape(-1)
        delta = np.abs(raw_qpos0 - scene_seed.qpos)
        changed_idx = np.flatnonzero((~robot_qpos_mask) & (delta > float(SCENE_QPOS_DIFF_EPS)))
        max_abs = float(delta[changed_idx].max()) if changed_idx.size > 0 else 0.0
        lines.append(
            "  raw_frame0_nonrobot_qpos_delta="
            f"changed={int(changed_idx.size)} max_abs={max_abs:.4f} "
            f"sample_idx={changed_idx[:8].tolist()}"
        )

        for joint_name in ["T1_free", "T2_free"]:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if jid < 0:
                continue
            qadr = int(model.jnt_qposadr[jid])
            if (qadr + 3) > raw_qpos0.shape[0] or (qadr + 3) > scene_seed.qpos.shape[0]:
                continue
            raw_xyz = np.round(raw_qpos0[qadr : qadr + 3], 4)
            settled_xyz = np.round(scene_seed.qpos[qadr : qadr + 3], 4)
            lines.append(f"  {joint_name}: settled_xyz={settled_xyz} raw_frame0_xyz={raw_xyz}")

    return lines


def stitch_npz(
    *,
    original_path: Path,
    suffix_path: Path,
    cut_idx: int,
    out_path: Path,
    drop_suffix_first_frame: bool,
    suffix_time_scale: float = 1.0,
) -> Path:
    orig = np.load(original_path)
    suf = np.load(suffix_path)
    suffix_start = 1 if drop_suffix_first_frame else 0
    sim_t_offset = float(orig["sim_t"][int(cut_idx)]) if "sim_t" in orig.files else 0.0
    out: Dict[str, np.ndarray] = {}
    for key in orig.files:
        if key in suf.files and getattr(orig[key], "ndim", 0) >= 1:
            left = orig[key][: int(cut_idx) + 1]
            right = suf[key][suffix_start:]
            if key == "sim_t":
                right = right * float(suffix_time_scale) + sim_t_offset
            out[key] = np.concatenate([left, right], axis=0)
        else:
            out[key] = orig[key]
    for key in suf.files:
        if key not in out:
            if key in {"intervention", "is_intervention", "action_source", "intervention_id"}:
                right = np.asarray(suf[key]).reshape(-1)[suffix_start:]
                left = np.zeros((int(cut_idx) + 1,), dtype=right.dtype)
                out[key] = np.concatenate([left, right], axis=0)
            elif key == "grip_real":
                right = np.asarray(suf[key], dtype=np.float64).reshape(-1)[suffix_start:]
                prefix_len = int(cut_idx) + 1
                if "ctrl_sim" in orig.files:
                    ctrl_prefix = np.asarray(orig["ctrl_sim"][:prefix_len], dtype=np.float64)
                    if ctrl_prefix.ndim == 2 and ctrl_prefix.shape[1] > 7:
                        left = np.clip(ctrl_prefix[:, 7] * 2.0, 0.0, 0.08)
                    else:
                        left = np.full((prefix_len,), np.nan, dtype=np.float64)
                else:
                    left = np.full((prefix_len,), np.nan, dtype=np.float64)
                out[key] = np.concatenate([left, right], axis=0)
            else:
                out[key] = suf[key]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)
    print(f"[Integration] Stitched trajectory saved: {out_path}")
    return out_path


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--trajectory", required=True)
    ap.add_argument("--host", required=True)
    ap.add_argument("--host_ip", default=None, help=argparse.SUPPRESS)
    ap.add_argument(
        "--transport",
        choices=["wifi", "usb"],
        default=None,
        help=(
            "Optional network mode override. "
            "'usb' enables adb reverse helpers but keeps host/host_ip unchanged "
            "so XR discovery/control plane is not broken."
        ),
    )
    ap.add_argument(
        "--usb_auto_reverse",
        action="store_true",
        help=(
            "When --transport usb, run adb reverse for topic/service ports automatically."
        ),
    )
    ap.add_argument(
        "--usb_adb_serial",
        default=None,
        help="Optional adb serial for USB mode (defaults to current adb target).",
    )
    ap.add_argument(
        "--usb_publisher_ip",
        default="127.0.0.1",
        help="Publisher IP advertised to Quest in USB mode (default: 127.0.0.1).",
    )
    ap.add_argument(
        "--usb_extra_reverse_ports",
        nargs="*",
        type=int,
        default=None,
        help="Optional extra ports to adb reverse in USB mode (for example: 7721).",
    )
    ap.add_argument(
        "--usb_reverse_remove_all",
        action="store_true",
        help="In USB mode, clear existing adb reverse mappings before applying new ones.",
    )
    ap.add_argument("--unity_node", required=True)
    ap.add_argument("--bind_ip", default="0.0.0.0")
    ap.add_argument("--start_idx", type=int, default=0)
    ap.add_argument("--robot_key", default="p1")
    ap.add_argument("--mirror_robot", action="store_true")
    ap.add_argument(
        "--mirror_on_select_only",
        action="store_true",
        help=(
            "With --mirror_robot, keep selector/grid replays sim-only until "
            "ENTER_SINGLE arrives on --cmd_port. EXIT_SINGLE releases robot "
            "ownership and returns that session to sim-only replay."
        ),
    )
    ap.add_argument("--replan_output_dir", default=None)
    ap.add_argument(
        "--post_replan_mode",
        choices=["preview_replay", "policy_handoff"],
        default="preview_replay",
        help=(
            "What to do after finishing intervention recording. preview_replay "
            "stitches and replays for testing; policy_handoff stitches, keeps "
            "the live MuJoCo state, and waits for a future policy continuation."
        ),
    )
    ap.add_argument(
        "--policy_handoff_idle_after_ready",
        action="store_true",
        help=(
            "In policy_handoff mode, stop the selected/single-view point-cloud stream "
            "and release robot ownership after the stitched handoff NPZ is ready."
        ),
    )
    ap.add_argument("--start_paused", action="store_true")
    ap.add_argument("--visible_geoms_groups", nargs="+", type=int, default=[2])
    ap.add_argument("--control_hz", type=float, default=120.0)
    ap.add_argument("--replay_speed", type=float, default=0.45,
                    help="Replay speed scale for trajectory timing (1.0=original, <1 slower)")
    ap.add_argument("--show_mujoco_window", action="store_true")
    ap.add_argument("--mujoco_window_cam", default=None,
                    help="Camera to display in the MuJoCo window (default: first cam in --cams)")
    ap.add_argument("--object_control", action="store_true")
    ap.add_argument("--object_control_bodies", nargs="*", default=None)
    ap.add_argument("--object_step_xy", type=float, default=0.01)
    ap.add_argument("--object_step_z", type=float, default=0.008)
    ap.add_argument("--object_move_rate_hz", type=float, default=10.0)
    ap.add_argument("--robot_start_preset", default="reverse_u")
    ap.add_argument("--robot_start_q", nargs=7, type=float, default=None)
    ap.add_argument("--service_port", type=int, default=7740)
    ap.add_argument("--topic_port", type=int, default=7741)
    ap.add_argument(
        "--cmd_port",
        type=int,
        default=0,
        help="If > 0, bind a ZMQ PULL socket on bind_ip:cmd_port that accepts "
             "single-frame button names ('A'|'B'|'X'|'Y') and routes them to the "
             "same pending-action handler that MetaQuest3 button events use. "
             "Designed for multi-session selector → per-session button forwarding.",
    )
    ap.add_argument(
        "--policy_cmd_host",
        default="127.0.0.1",
        help="Host for optional policy-viewer command forwarding.",
    )
    ap.add_argument(
        "--policy_cmd_port",
        type=int,
        default=0,
        help=(
            "If >0, forward action commands (A/B/X/Y/INTERVENE/CANCEL) received "
            "on this runtime's cmd_port to the policy viewer instead of handling "
            "local VR intervention/replan actions."
        ),
    )
    ap.add_argument(
        "--policy_state_host",
        default="127.0.0.1",
        help="Host for optional live policy-state mirroring.",
    )
    ap.add_argument(
        "--policy_state_port",
        type=int,
        default=0,
        help="If >0, subscribe to live Intervene MuJoCo state and render sensors from it.",
    )
    ap.add_argument(
        "--policy_state_timeout_s",
        type=float,
        default=1.0,
        help="Warn if no live policy-state frame has arrived for this many seconds after first state.",
    )
    ap.add_argument(
        "--no_mujoco_publisher",
        action="store_true",
        help="Skip MujocoPublisher (XR scene mesh). Use for thumbnail-only sessions "
             "in the multi-session launcher to prevent robot scenes from loading in "
             "the selector scene.",
    )
    ap.add_argument(
        "--no_xr_device",
        action="store_true",
        help=(
            "Skip MetaQuest3 device connection (VR controller input). "
            "Use for thumbnail-only sessions that do not need A/B/X/Y controls. "
            "Eliminates cross-session Quest overload in multi-session mode "
            "(15 sessions × 3 topics = 45 simultaneous subscriptions without this flag)."
        ),
    )
    ap.add_argument(
        "--scene_push_delay",
        type=float,
        default=0.0,
        help=(
            "Seconds to sleep before initializing MujocoPublisher (scene mesh service). "
            "Use in multi-session mode to stagger SpawnSimScene pushes so they don't all "
            "hit the Quest Unity node simultaneously. "
            "Recommended: session_index * 2.0 (e.g. session 5 → 10.0s delay)."
        ),
    )
    ap.add_argument("--fps", type=float, default=30.0,
                    help="ACTIVE sensor publish rate when peer_count >= --active_peer_threshold. "
                         "Falls back to --fps_idle when no one is subscribed to the heavy "
                         "topics (selector grid only). See Adaptive Rate section.")
    ap.add_argument("--pc_fps_cap", type=float, default=0.0,
                    help="Max publish rate per PC camera (Hz). 0 = uncapped (default). "
                         "At stride 4 the 3-camera burst payload is ~93KB and Quest handles it "
                         "without stutter; the cap only adds latency. Only set >0 if you observe "
                         "Quest-side Unity spikes at stride <=2 with very large point counts.")
    ap.add_argument("--fps_idle", type=float, default=10.0,
                    help="IDLE sensor publish rate when peer_count < --active_peer_threshold. "
                         "Use when only the selector thumbnail panel is subscribed (1 peer "
                         "per session × 15 sessions × 60 Hz × 5 cams was saturating the CPU; "
                         "drop the 14 non-active sessions to 10 Hz to free that budget). "
                         "Set equal to --fps to disable adaptive rate.")
    ap.add_argument("--active_peer_threshold", type=int, default=2,
                    help="Peer count at or above which the session enters ACTIVE mode (--fps). "
                         "Default 2: selector grid contributes 1 peer per session; entering "
                         "single view adds the PC + RGB subscribers (2+ extra peers), pushing "
                         "the count past 2 and triggering the boost. Tune based on what your "
                         "Unity scene subscribes to.")
    ap.add_argument("--main_loop_watchdog_s", type=float, default=3.0,
                    help="If the main loop does not advance for this many seconds, dump ALL "
                         "thread stacks (faulthandler.dump_traceback) to stderr so a hang shows "
                         "exactly where it is stuck. The dump fires once per stall; the watchdog "
                         "re-arms after the loop recovers. Set 0 to disable. Diagnostic aid for "
                         "the single-view point-cloud main-loop hang.")
    ap.add_argument("--gc_interval", type=int, default=300,
                    help="Run Python gc.collect() every N main-loop frames during idle time to "
                         "prevent automatic GC pauses from causing periodic VR FPS drops. "
                         "Default 300 ≈ every 5s at 60fps. Set 0 to leave Python's auto-GC on.")
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument(
        "--render_shadows", action="store_true",
        help="Enable shadow rendering in MuJoCo (default OFF for performance). "
             "Shadows add ~20-40%% per-frame render cost on lit RGB passes. Enable "
             "if you want a pretty-looking MuJoCo window for demos/screenshots.",
    )
    ap.add_argument(
        "--render_reflections", action="store_true",
        help="Enable reflections (default OFF for performance).",
    )
    ap.add_argument(
        "--render_haze", action="store_true",
        help="Enable haze/atmospheric scattering (default OFF for performance).",
    )
    ap.add_argument(
        "--render_skybox", action="store_true",
        help="Enable skybox rendering (default OFF for performance).",
    )
    ap.add_argument("--cams", nargs="*", default=None)
    ap.add_argument("--rgb_cams", nargs="*", default=None)
    ap.add_argument("--rgbd_cams", nargs="*", default=None)
    ap.add_argument("--jpg_quality", type=int, default=85)
    ap.add_argument(
        "--latency_probe", action="store_true",
        help="Measure TRUE end-to-end latency (publisher -> photons). Draws a 64-bit "
             "sequence number into a 256x16 corner of each published RGB frame; the "
             "headset reads it back and echoes it twice (on decode, and after present) "
             "over the existing command socket. Both echoes are timestamped on THIS "
             "machine's clock, so (RTT_B - RTT_A) gives the on-headset cost with no "
             "clock synchronisation. Needs an APK carrying LatencyStampProbe.cs; without "
             "one you simply get zero samples (and a small chequered strip on the panel), "
             "never wrong numbers. Costs ~0.03 ms/frame and does not enlarge the JPEG.",
    )
    ap.add_argument("--no_rgb_topic", action="store_true")
    ap.add_argument(
        "--rgb_panel_mode",
        action="store_true",
        help=(
            "Publish per-camera RGB JPEG topics for a diamond-layout camera panel "
            "single view (Unity-side) instead of point clouds. Cameras listed in "
            "--rgb_panel_cams are auto-merged into --cams so they actually render. "
            "Independent of --pc / the point cloud pipeline — does not touch it."
        ),
    )
    ap.add_argument(
        "--rgb_panel_cams", nargs="*", default=["wrist", "left", "right"],
        help=(
            "Cameras published as gated RGB-panel topics when --rgb_panel_mode is "
            "set. 'front' is intentionally excluded — it reuses the always-on "
            "front/rgb thumbnail topic already published for the selector grid."
        ),
    )
    ap.add_argument(
        "--rgb_panel_active_only",
        action="store_true",
        help=(
            "When --rgb_panel_mode is enabled, publish --rgb_panel_cams topics "
            "only while the session is active (mirrors --pc_active_only)."
        ),
    )
    ap.add_argument("--pc", action="store_true")
    ap.add_argument(
        "--pc_active_only",
        action="store_true",
        help=(
            "When --pc is enabled, publish heavy point cloud topics only while "
            "the session is active. By default active is peer-count based; combine "
            "with --selected_sensor_activation in multi-window mode so only "
            "ENTER_SINGLE activates full point clouds."
        ),
    )
    ap.add_argument(
        "--peer_fallback_grace_s", type=float, default=None,
        help="With --selected_sensor_activation, how long peers must stay >= "
             "--active_peer_threshold before the peer count is trusted as a fallback for a "
             "dropped ENTER_SINGLE. 0 disables the fallback (selection strictly "
             "authoritative). Exists because combined view keeps grid subscribers alive on "
             "every session, which used to promote UNSELECTED sessions to full point-cloud "
             "rendering and was the dominant cost measured on 2026-07-28. "
             "DEFAULT: 0 (disabled) when --selected_sensor_activation is set, else 3.0. "
             "Measured 2026-07-29: with the old 3.0 default an UNSELECTED session still "
             "self-promoted ('[AdaptiveRate][Fallback] ... peers=4 ... selected=False'), "
             "doing full 200k-point work and stealing budget from the selected session.",
    )
    ap.add_argument(
        "--selected_peer_idle_grace_s", type=float, default=5.0,
        help="Demote a selected session when its PUB peer count stays below "
             "--active_peer_threshold for this many seconds. It reactivates immediately "
             "when peers return. Prevents a lost EXIT_SINGLE from leaving an old session "
             "on the full point-cloud path. 0 disables the fail-safe.",
    )
    ap.add_argument(
        "--rgb_thumbnail_fps", type=float, default=10.0,
        help="Cap the selector-thumbnail RGB topic (the first --rgb_cams entry, normally "
             "'front') to this rate even when the session is ACTIVE. The grid thumbnail is "
             "small and does not need the full point-cloud frame rate; capping it removes "
             "one camera render per tick from the selected session. 0 = uncapped.",
    )
    ap.add_argument(
        "--selected_sensor_activation",
        action="store_true",
        help=(
            "Use ENTER_SINGLE/EXIT_SINGLE commands, not PUB subscriber count, to "
            "choose ACTIVE vs IDLE sensor mode. This keeps selector/grid sessions "
            "top-RGB-only even when Unity maintains multiple subscribers."
        ),
    )
    ap.add_argument(
        "--start_in_single_view_session", type=int, default=-1,
        help="Like --start_in_single_view, but only for the session whose "
             "--session_index matches. -1 (default) disables. Every session in a "
             "multi-window launch receives IDENTICAL extra args, so plain "
             "--start_in_single_view activates all N at once -- which measures an Nx "
             "point-cloud load no operator ever sees, since exactly one session is in "
             "single view at a time. This reproduces the real topology (one ACTIVE, N-1 "
             "idle thumbnails) from a single env var, with no second terminal, no "
             "ENTER_SINGLE timing and no status-file lookup.",
    )
    ap.add_argument(
        "--start_in_single_view",
        action="store_true",
        help=(
            "Start the session with single_view_active=True (sensors ACTIVE / "
            "point clouds publishing from frame 0) instead of waiting for an "
            "ENTER_SINGLE command. Used by standalone single-session launches that "
            "bypass the multi-window selector grid. The robot is unaffected — "
            "arming still requires an A/X press (gated by --mirror_on_select_only)."
        ),
    )
    ap.add_argument("--pc_cams", nargs="*", default=None)
    ap.add_argument("--pc_backend", choices=["gpu", "legacy"], default="gpu")
    ap.add_argument("--pc_worker_threads", type=int, default=1,
                    help="Number of parallel GPU point-cloud build threads. "
                         "1 = sequential (safe default). 3 = one thread per camera; "
                         "all cameras build simultaneously, cutting PC latency to ~1 build "
                         "cycle instead of 3. Requires thread-safe GPU pipeline.")
    ap.add_argument("--pc_sampling", choices=["grid", "stable_random"], default="grid")
    ap.add_argument("--pc_stride", type=int, default=4)
    ap.add_argument("--pc_size", type=float, default=0.02)
    ap.add_argument("--pc_scale", type=float, default=1.0)
    ap.add_argument("--pc_intrinsics_mode", choices=["mujoco", "aspect"], default="mujoco")
    ap.add_argument("--pc_min_depth", type=float, default=0.05)
    ap.add_argument("--pc_max_depth", type=float, default=None)
    ap.add_argument("--pc_flip_x", action="store_true")
    ap.add_argument("--pc_flip_y", action="store_true")
    ap.add_argument("--pc_no_flip_z", action="store_true")
    ap.add_argument("--pc_flip_z", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--pc_max_points", type=int, default=0)
    ap.add_argument("--pc_max_points_top", type=int, default=0)
    ap.add_argument("--pc_object_only", action="store_true")
    ap.add_argument("--pc_object_bbox_min", nargs=3, type=float, default=[0.35, -0.35, 0.40])
    ap.add_argument("--pc_object_bbox_max", nargs=3, type=float, default=[0.90, 0.35, 0.85])
    ap.add_argument("--rgbd_include_pc", action="store_true")
    ap.add_argument(
        "--pc_debug_visibility",
        action="store_true",
        help="Save first-frame point-cloud visibility diagnostics for selected PC cameras",
    )
    ap.add_argument(
        "--pc_debug_visibility_cams",
        nargs="*",
        default=None,
        help="Optional subset of PC cameras for first-frame visibility diagnostics",
    )
    ap.add_argument(
        "--pc_debug_visibility_outdir",
        default=None,
        help="Optional output directory for point-cloud visibility PNG diagnostics",
    )
    ap.add_argument("--cam_extrinsics_json", default=None)
    ap.add_argument("--cam_extrinsics_profile", choices=["none", "lab_standard"], default="lab_standard")
    ap.add_argument("--pc_clip_below_table", action="store_true")
    ap.add_argument("--pc_table_geoms", nargs="*", default=["table_collision", "table_visual"])
    ap.add_argument("--pc_table_body", default="table")
    ap.add_argument("--pc_table_plane_z", type=float, default=None)
    ap.add_argument(
        "--pc_idle_release_s", type=float, default=2.0,
        help="seconds a session must stay unselected before its Open3D CUDA cache is "
             "released back to the driver (0 disables the hysteresis, not the reclaim).",
    )
    ap.add_argument(
        "--pc_build_error_escalate_n", type=int, default=10,
        help="consecutive point-cloud build failures before a diagnosed ERROR is logged.",
    )
    ap.add_argument("--pc_table_margin", type=float, default=0.005)
    ap.add_argument("--pc_table_clearance", type=float, default=0.015)
    ap.add_argument("--pc_top_table_clearance", type=float, default=None)
    ap.add_argument("--pc_table_body_top_offset", type=float, default=0.02)
    ap.add_argument("--pc_anchor_auto_translate", dest="pc_anchor_auto_translate", action="store_true", default=False)
    ap.add_argument("--pc_no_anchor_auto_translate", dest="pc_anchor_auto_translate", action="store_false")
    ap.add_argument("--pc_anchor_site", default=None)
    ap.add_argument("--pc_anchor_body", default="link0")
    ap.add_argument("--pc_anchor_alpha", type=float, default=0.05)
    ap.add_argument("--pc_anchor_patch", type=int, default=3)
    ap.add_argument("--pc_anchor_depth_tol", type=float, default=0.15)
    ap.add_argument("--pc_anchor_max_corr", type=float, default=0.08)
    ap.add_argument("--pc_anchor_source", choices=["auto", "robot", "table"], default="auto")
    ap.add_argument("--pc_anchor_world_xyz", nargs=3, type=float, default=None)
    ap.add_argument("--pc_anchor_update_mode", choices=["global", "per_camera"], default="global")
    ap.add_argument("--log_every", type=int, default=60)
    ap.add_argument(
        "--perf_log",
        action="store_true",
        help="Print per-publish-frame wall-time breakdown (render, PC submit, total) to stdout.",
    )
    # --- Windowed metrics aggregation (PART-1 measurements) -------------------
    # --perf_log prints one line per frame and aggregates nothing. These flags add
    # avg/p50/p95/p99 over a rolling window plus a JSONL sink that tools/perf_report.py
    # joins against the Quest-side logcat collector (tools/quest_telemetry.py).
    ap.add_argument(
        "--metrics_out",
        type=str,
        default=None,
        help="Write windowed performance rows (JSONL) to this path. Pass 'auto' for "
             "session_logs/metrics_S<ii>_<timestamp>.jsonl. Empty/omitted disables the sink.",
    )
    ap.add_argument(
        "--metrics_window_s",
        type=float,
        default=10.0,
        help="Seconds per metrics window (row + optional [PerfSummary] block). Default 10.",
    )
    ap.add_argument(
        "--metrics_summary",
        action="store_true",
        help="Also print the [PerfSummary] block to stdout each window.",
    )
    ap.add_argument(
        "--pc_round_robin",
        action="store_true",
        help="Render only ONE camera per publish tick, cycling through the cameras that "
             "have publish work (e.g. top->right->left). Caps the main-thread render stall "
             "at a single render so the main loop advances the trajectory + robot mirror "
             "much more often -> smoother real<->sim mirroring. Each camera still renders at "
             "full resolution/stride/points (spatial PC quality unchanged); only its per-camera "
             "refresh rate drops to fps/N. Unity's merged loader keeps the last payload per "
             "source only up to its SourceIdleClearSeconds — see --pc_max_source_age_s, which "
             "puts a hard floor under that. Best when render is the dominant per-tick cost.",
    )
    ap.add_argument(
        "--pc_declared_capacity_mode", choices=["adaptive", "cap"], default="adaptive",
        help="How much per-payload capacity to advertise to Unity. 'adaptive' (default) "
             "declares a growth-only high-water mark of the ACTUAL point count; 'cap' "
             "declares the flat --pc_max_points. Unity sizes its decode buffer from this "
             "value and discards it between payloads, so declaring 200000 while sending "
             "~5000 points made it allocate and zero 3.2 MB per payload on its receive "
             "thread (~576 MB/s at 3 cams x 60 Hz) — measured as a visibly frozen point "
             "cloud at a stable frame rate. The wire payload is sized by the actual count, "
             "so the honest declaration costs no bandwidth. Use 'cap' to A/B the difference.",
    )
    ap.add_argument(
        "--pc_max_source_age_s", type=float, default=0.4,
        help="Warn once when a camera goes this long without being rendered under "
             "--pc_round_robin. Diagnostic, not a control: scheduling cannot create render "
             "budget that does not exist, and force-rendering the starved cameras together "
             "measurably made overload WORSE. Unity deletes a point-cloud source after its "
             "SourceIdleClearSeconds, so seeing this warning is the early sign of clouds "
             "dropping out one at a time. Keep it well below the Unity value. 0 = no warning.",
    )
    ap.add_argument("--session_index", type=int, default=0,
        help="0-based session index; used to phase-offset the sinusoid fallback risk wave across sessions.")
    ap.add_argument("--acc_risk_scale", type=float, default=3.0,
        help="Scale factor applied to raw ACC score before clamping to 0-10. "
             "Higher = more sensitive to prediction inconsistency. Default 3.0.")
    ap.add_argument("--risk_dummy", action="store_true", default=True,
        help="Publish ACC-based risk values on SimPub/Status/risk at 2 Hz. "
             "Falls back to trajectory jerk or sinusoid when policy mirror is unavailable.")
    return ap


def publish_sensor_frames(
    *,
    args,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: unified.GLFWMjrRenderer,
    sensor_node: unified.SensorNode,
    cams: List[str],
    rgb_cam_set: set,
    rgbd_cam_set: set,
    pc_cam_set: set,
    rgb_panel_cam_set: set = frozenset(),
    publish_rgb_topic: bool,
    extrinsics: Dict[str, unified.CameraExtrinsic],
    pc_pipeline,
    pc_target_sample_budget: int,
    pc_max_depth: float,
    object_bbox_min: np.ndarray,
    object_bbox_max: np.ndarray,
    table_plane: Optional[unified.Plane],
    anchor_site_id: int,
    anchor_body_id: int,
    manual_anchor_world: Optional[np.ndarray],
    flip_z: bool,
    sensor_state: SensorPublishState,
    pc_declared_capacity_by_cam: Dict[str, int],
    pc_debug_visibility_cam_set: set,
    pc_debug_visibility_outdir: Optional[Path],
    pc_worker: Optional["PCWorkerThread"] = None,
    selected_sensor_active: Optional[bool] = None,
    force_sensor_idle: bool = False,
    metrics: Optional[MetricsCollector] = None,
) -> Tuple[Optional[unified.Plane], SensorPublishState]:
    now = time.time()
    if now < sensor_state.next_sensor_t:
        return table_plane, sensor_state

    # Content cadence is distinct from render cadence: a runtime can publish at 20 Hz while
    # rendering the same mirrored policy pose repeatedly. The state vector is small, so this
    # comparison is cheap and only runs on actual sensor ticks while metrics are enabled.
    if metrics is not None:
        qpos_now = np.asarray(data.qpos)
        changed = (
            sensor_state.last_render_qpos is None
            or qpos_now.shape != sensor_state.last_render_qpos.shape
            or not np.allclose(qpos_now, sensor_state.last_render_qpos, rtol=0.0, atol=1e-7)
        )
        metrics.incr("render_distinct_states" if changed else "render_repeated_states")
        metrics.observe("render_repeated_ratio", 0.0 if changed else 1.0)
        sensor_state.last_render_qpos = qpos_now.copy()
        mirror = globals().get("_ACTIVE_POLICY_MIRROR")
        state_wall = getattr(mirror, "latest_state_wall", None) if mirror is not None else None
        if state_wall is not None:
            metrics.observe("state_to_render_ms", max(0.0, now - float(state_wall)) * 1000.0)

    # --- Adaptive rate / selected-only throttling ---
    # In multi-window mode Unity may keep several subscribers connected per
    # thumbnail, so peer-count is not a reliable "selected" signal. When
    # selected_sensor_active is provided, ENTER_SINGLE/EXIT_SINGLE is canonical.
    fps_idle = float(getattr(args, "fps_idle", float(args.fps)))
    threshold = int(getattr(args, "active_peer_threshold", 2))
    try:
        peer_count = sensor_node.peer_count() if hasattr(sensor_node, "peer_count") else threshold
    except Exception:
        peer_count = threshold
    # Selection (ENTER_SINGLE/EXIT_SINGLE) is authoritative. The peer count is only a
    # LAST-RESORT fallback for a dropped ENTER_SINGLE (lesson 41), and it must be
    # debounced: in combined view every session keeps grid subscribers alive and panels
    # reconnect on an 8 s idle timer, so a bare peer>=threshold test promoted unselected
    # sessions to full point-cloud rendering. That was the dominant cost in the
    # 2026-07-28 measurements (render 51 ms vs a 17 ms budget, PC at 18.6 Hz).
    grace_s = float(getattr(args, "peer_fallback_grace_s", 0.0) or 0.0)
    is_active, activation_reason = resolve_sensor_activation(
        sensor_state=sensor_state,
        selected=selected_sensor_active,
        peer_count=peer_count,
        threshold=threshold,
        grace_s=grace_s,
        selected_peer_idle_grace_s=float(
            getattr(args, "selected_peer_idle_grace_s", 5.0) or 0.0
        ),
        now=now,
        force_idle=force_sensor_idle,
    )
    if activation_reason == "peer_fallback":
        if not sensor_state.peer_fallback_engaged:
            print(
                f"[AdaptiveRate][Fallback] Not selected, but peers={peer_count} >= "
                f"{threshold} held for {grace_s:.1f}s — assuming a dropped ENTER_SINGLE "
                f"and activating. If this fires without a selection, raise "
                f"--active_peer_threshold or set --peer_fallback_grace_s 0."
            )
            sensor_state.peer_fallback_engaged = True
    elif activation_reason == "selected_no_peers":
        sensor_state.peer_fallback_engaged = False
    else:
        sensor_state.peer_fallback_engaged = False
    effective_fps = float(args.fps) if is_active else fps_idle
    mode_label = "ACTIVE" if is_active else "IDLE"
    publish_pc_topics = bool(args.pc) and (
        is_active or not bool(getattr(args, "pc_active_only", False))
    )
    active_tracker = globals().get("_ACTIVE_PC_SESSION_TRACKER")
    if active_tracker is not None:
        active_tracker.set_active(
            publish_pc_topics, peers=peer_count, reason=activation_reason
        )
        active_tracker.poll()
    # RGB-panel mode: independent of publish_pc_topics above — never touches the
    # point cloud gating. Reuses the same is_active computed above.
    publish_rgb_panel_topics = bool(getattr(args, "rgb_panel_mode", False)) and (
        is_active or not bool(getattr(args, "rgb_panel_active_only", False))
    )
    if (mode_label != sensor_state.last_adaptive_mode
            or peer_count != sensor_state.last_adaptive_peers):
        # Only log on transitions (mode change) or every time peers count moves —
        # the latter is fine because peer changes are rare events.
        if mode_label != sensor_state.last_adaptive_mode:
            activation_source = activation_reason
            print(
                f"[AdaptiveRate] {sensor_state.last_adaptive_mode or 'INIT'} -> {mode_label} "
                f"(source={activation_source}, selected={bool(selected_sensor_active) if selected_sensor_active is not None else 'n/a'}, "
                f"peers={peer_count}, threshold={threshold}, fps={effective_fps:.0f}, pc={publish_pc_topics})"
            )
        sensor_state.last_adaptive_mode = mode_label
        sensor_state.last_adaptive_peers = peer_count

    if args.pc and args.pc_clip_below_table and table_plane is None:
        table_plane = unified.resolve_table_plane(
            model=model,
            data=data,
            geom_candidates=args.pc_table_geoms,
            body_name=args.pc_table_body,
            manual_plane_z=args.pc_table_plane_z,
            body_top_offset=args.pc_table_body_top_offset,
        )

    anchor_world_mj = None
    if args.pc and args.pc_anchor_auto_translate:
        if manual_anchor_world is not None:
            anchor_world_mj = manual_anchor_world
        else:
            if args.pc_anchor_source in ("auto", "robot"):
                anchor_world_mj = unified.get_anchor_world_mj(
                    data=data,
                    anchor_site_id=anchor_site_id,
                    anchor_body_id=anchor_body_id,
                )
            if anchor_world_mj is None and args.pc_anchor_source in ("auto", "table") and table_plane is not None:
                anchor_world_mj = np.asarray(table_plane.point, dtype=np.float64)

    timestamp = now
    _perf_render_t = 0.0
    _perf_pc_t = 0.0
    _perf_jpeg_t = 0.0
    _perf_publish_t = 0.0

    # Thumbnail rate cap: the selector-grid feed does not need the point-cloud frame rate.
    # Only cameras whose SOLE job is the thumbnail are capped — a camera that also feeds
    # the point cloud or an RGB panel gets rendered regardless, so capping it would save
    # nothing and only starve the grid. Skipping the publish also skips the RENDER below
    # (a camera with no publish work is `continue`d), which is the actual saving.
    # Computed before the round-robin selection so a capped camera is not chosen as this
    # tick's single working camera.
    thumbnail_cam_set = set(rgb_cam_set) - set(pc_cam_set) - set(rgb_panel_cam_set)
    thumb_fps = float(getattr(args, "rgb_thumbnail_fps", 0.0) or 0.0)
    thumb_due = True
    if thumb_fps > 0.0 and effective_fps > thumb_fps and thumbnail_cam_set:
        thumb_due = (now - sensor_state.last_thumbnail_pub_t) >= (1.0 / thumb_fps)

    def _rgb_due(cam: str) -> bool:
        """Is this camera's plain /rgb topic due to publish this tick?"""
        if not (publish_rgb_topic and cam in rgb_cam_set):
            return False
        if cam in thumbnail_cam_set and not thumb_due:
            return False
        return True

    # Round-robin: render only ONE working camera per tick (cycling) to cap the
    # main-thread render stall at a single render -> the main loop advances the
    # trajectory + robot mirror far more often (smoother real<->sim). Spatial PC
    # quality is unchanged; only each camera's refresh rate drops to fps/N. Unity's
    # merged PC loader keeps the last payload per source so all clouds stay visible.
    iter_cams = cams
    if getattr(args, "pc_round_robin", False):
        work_cams = [
            c for c in cams
            if _rgb_due(c)
            or (c in rgbd_cam_set)
            or (publish_pc_topics and c in pc_cam_set)
            or (publish_rgb_panel_topics and c in rgb_panel_cam_set)
        ]
        if len(work_cams) > 1:
            # STARVATION FLOOR (the fix for "point clouds drop one camera at a time").
            # Unity's GpuMergedPointCloudLoader deletes a source that has been silent for
            # SourceIdleClearSeconds. Fixed `frame_idx % n` cycling gives each camera a
            # period of tick_dt * n with no upper bound, so whenever the main loop is slow
            # (9 windows: p99 tick 57 ms, max 238 ms) a camera silently crosses that cliff
            # and visibly disappears until its next turn. Any camera older than
            # --pc_max_source_age_s is rendered THIS tick, even if that means rendering
            # more than one: a one-off longer tick is far cheaper than the cloud vanishing.
            # Exactly ONE camera per tick, always — the whole point of round-robin is to cap
            # the main-thread stall at a single render. Rendering the starved cameras
            # *together* was tried and rejected: simulated against the measured 9-window
            # tick distribution it made things WORSE under overload (8x load: 6091 ms worst
            # silence batched vs 2455 ms with one-per-tick), because the extra renders
            # lengthen exactly the tick that was already too long.
            #
            # The fix is therefore ordering, not volume: pick the LEAST-RECENTLY-SERVICED
            # camera instead of `frame_idx % n`. That is the camera closest to Unity's
            # source-delete cliff, and it is self-correcting — a camera skipped because the
            # thumbnail rate cap changed the working-set size is picked up on the next tick
            # instead of waiting a full extra cycle. Never worse than modulo at any load
            # (measured: 317/614/1228/2455 ms vs 351/700/1551/3065 ms at 1x/2x/4x/8x).
            iter_cams = [
                min(work_cams, key=lambda c: sensor_state.last_cam_submit_t.get(c, 0.0))
            ]

            # Diagnostic only. Sustained starvation cannot be scheduled away — it means the
            # tick rate is too low for this many cameras, and the honest levers are fewer
            # windows, lower --pc_max_points, or a lower --fps.
            max_age = float(getattr(args, "pc_max_source_age_s", 0.0) or 0.0)
            if max_age > 0.0 and not sensor_state.starvation_logged:
                oldest = max(
                    now - sensor_state.last_cam_submit_t.get(c, now) for c in work_cams
                )
                if oldest >= max_age:
                    sensor_state.starvation_logged = True
                    print(
                        f"[PC][Starvation] a camera went {oldest:.2f}s without a render "
                        f"(--pc_max_source_age_s={max_age:.2f}s, {len(work_cams)} working "
                        f"cameras). Unity deletes a point-cloud source after its "
                        f"SourceIdleClearSeconds, so this is what makes clouds drop out one "
                        f"at a time. Reduce --pc_max_points / --fps / the window count."
                    )

    for cam_name in iter_cams:
        publish_rgb = _rgb_due(cam_name)
        publish_rgbd = cam_name in rgbd_cam_set
        publish_pc = publish_pc_topics and (cam_name in pc_cam_set)
        publish_rgb_panel = publish_rgb_panel_topics and (cam_name in rgb_panel_cam_set)
        if not (publish_rgb or publish_rgbd or publish_pc or publish_rgb_panel):
            continue

        # Stamp before the render, not after: the scheduler asks "how long since this camera
        # was last serviced", and stamping on entry keeps the interval independent of how
        # long this particular render happens to take.
        sensor_state.last_cam_submit_t[cam_name] = now

        _t0 = time.time()
        rgb, depth_m = renderer.render(data, cam_name, blit_to_window=False)
        _cam_render_t = time.time() - _t0
        _perf_render_t += _cam_render_t
        if metrics is not None:
            # Per-camera attribution: the aggregate render_ms cannot tell you whether the
            # cost is one expensive camera or several cheap ones.
            metrics.observe(f"render_ms.{cam_name}", _cam_render_t * 1000.0)
        # JPEG encode was previously outside every timer, so its cost was invisible in
        # [Perf] (it only showed up inside `total`). It is the dominant per-tick cost in
        # RGB mode, where there is no PC build to compare against.
        _t0 = time.time()
        rgb_jpg = None
        if publish_rgb or publish_rgbd or publish_rgb_panel:
            # The latency stamp goes in for the encode and comes straight back out: the
            # same `rgb` buffer colourises the point cloud a few lines below, and a
            # stamped one would speckle a strip of black/white points into the operator's
            # view. `stamped` saves and restores just the 256x16 region.
            _probe = globals().get("_LATENCY_PROBE")
            # Only the ACTIVE (selected) session stamps. Every session receives the same
            # --latency_probe, and under combined view the headset keeps decoding all N
            # grid thumbnails -- so with every session stamping it read and echoed ~90
            # frames/s that nobody was measuring, and each echo is two sends on the
            # headset's main thread. Unstamped idle thumbnails fail the magic check and
            # cost the headset nothing but the read itself.
            if _probe is not None and _latency_stamped is not None and is_active:
                _seq = _probe.issue(cam_name)
                with _latency_stamped(
                    rgb, _seq,
                    cam_id=(hash(cam_name) & 0xFF),
                    session_index=int(getattr(args, "session_index", 0)) & 0xFF,
                ):
                    rgb_jpg = unified.encode_rgb_jpg(rgb, quality=args.jpg_quality)
            else:
                rgb_jpg = unified.encode_rgb_jpg(rgb, quality=args.jpg_quality)
        _perf_jpeg_t += time.time() - _t0

        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        fovy = float(model.cam_fovy[cam_id])
        _ik = (cam_name, args.w, args.h, fovy, args.pc_intrinsics_mode)
        if _ik not in _CAM_INTRINSICS_CACHE:
            _CAM_INTRINSICS_CACHE[_ik] = unified.compute_intrinsics(
                args.w, args.h, fovy, mode=args.pc_intrinsics_mode,
            )
        fx, fy, cx, cy = _CAM_INTRINSICS_CACHE[_ik]
        calib = extrinsics.get(cam_name)
        cam_t_mj, cam_R_mj = unified.get_calibrated_cam_pose(data, cam_id, calib)
        cam_q_mj = unified.rotmat_to_quat_wxyz(cam_R_mj)
        cam_t_unity = unified.mj_to_unity_pos(cam_t_mj.astype(np.float32))
        cam_q_unity = unified.mj_to_unity_quat_xyzw(cam_q_mj)

        removed_below_plane = 0
        removed_outside_object_box = 0
        pc7 = None
        if publish_pc:
            anchor_corr_updated = False
            if args.pc_anchor_update_mode == "global":
                cam_anchor_corr = sensor_state.anchor_corr_global.copy()
            else:
                cam_anchor_corr = sensor_state.anchor_corr_by_cam.get(
                    cam_name, sensor_state.anchor_corr_global.copy()
                )
            if args.pc_anchor_auto_translate and anchor_world_mj is not None:
                corr_obs = unified.estimate_anchor_translation_correction(
                    anchor_world_mj=anchor_world_mj,
                    depth_m=depth_m,
                    fx=fx,
                    fy=fy,
                    cx=cx,
                    cy=cy,
                    cam_t_mj=cam_t_mj,
                    cam_R_mj=cam_R_mj,
                    flip_x=args.pc_flip_x,
                    flip_y=args.pc_flip_y,
                    flip_z=flip_z,
                    pc_scale=float(args.pc_scale),
                    patch_radius=int(args.pc_anchor_patch),
                    depth_tol=float(args.pc_anchor_depth_tol),
                    max_corr_norm=float(args.pc_anchor_max_corr),
                )
                if corr_obs is not None:
                    alpha = float(np.clip(args.pc_anchor_alpha, 0.0, 1.0))
                    if args.pc_anchor_update_mode == "global":
                        sensor_state.anchor_corr_global = (
                            (1.0 - alpha) * sensor_state.anchor_corr_global
                        ) + (alpha * corr_obs)
                        cam_anchor_corr = sensor_state.anchor_corr_global.copy()
                    else:
                        cam_anchor_corr = ((1.0 - alpha) * cam_anchor_corr) + (alpha * corr_obs)
                        sensor_state.anchor_corr_by_cam[cam_name] = cam_anchor_corr
                    anchor_corr_updated = True

            cam_clearance = float(args.pc_table_clearance)
            if cam_name == "top" and args.pc_top_table_clearance is not None:
                cam_clearance = float(args.pc_top_table_clearance)

            debug_visibility_requested = (
                bool(args.pc_debug_visibility)
                and cam_name in pc_debug_visibility_cam_set
                and not sensor_state.pc_debug_visibility_saved.get(cam_name, False)
            )
            _pc_build_kwargs = dict(
                cam_name=cam_name,
                depth_m=depth_m.copy() if pc_worker is not None else depth_m,
                rgb_u8=rgb.copy() if pc_worker is not None else rgb,
                width=args.w,
                height=args.h,
                fovy_deg=fovy,
                intrinsics_mode=args.pc_intrinsics_mode,
                stride=max(1, int(args.pc_stride)),
                sampling_mode=args.pc_sampling,
                min_depth=float(args.pc_min_depth),
                max_depth=float(pc_max_depth) if pc_max_depth is not None else None,
                flip_x=args.pc_flip_x,
                flip_y=args.pc_flip_y,
                flip_z=flip_z,
                pc_scale=float(args.pc_scale),
                cam_t_mj=cam_t_mj.copy() if pc_worker is not None else cam_t_mj,
                cam_R_mj=cam_R_mj.copy() if pc_worker is not None else cam_R_mj,
                cam_anchor_corr=cam_anchor_corr.copy() if pc_worker is not None else cam_anchor_corr,
                clip_below_table=bool(args.pc_clip_below_table),
                table_plane=table_plane,
                table_margin=float(args.pc_table_margin),
                table_clearance=cam_clearance,
                object_only=bool(args.pc_object_only),
                object_bbox_min=object_bbox_min,
                object_bbox_max=object_bbox_max,
                declared_capacity=_resolve_declared_capacity(
                    args, sensor_state, cam_name, pc_declared_capacity_by_cam[cam_name]
                ),
                debug_visibility=debug_visibility_requested,
            )
            _mirror = globals().get("_ACTIVE_POLICY_MIRROR")
            _pc_metadata = {
                "submitted_wall": time.time(),
                "rendered_wall": time.time(),
                "state_wall": getattr(_mirror, "latest_state_wall", None),
                "policy_seq": getattr(_mirror, "last_applied_seq", None),
                "policy_frame_idx": getattr(_mirror, "latest_frame_idx", None),
                # Which compiled model this depth buffer was rendered from. A build in
                # flight when the model swaps would otherwise be drained and published as
                # current -- the point cloud would carry the previous episode's cup
                # geometry, and nothing on the wire could tell Unity apart.
                "variant_key": getattr(_mirror, "active_variant_key", BASE_VARIANT_KEY),
            }

            if pc_worker is not None:
                # Offload build_frame (includes CUDA→CPU sync) to background thread.
                # Result is drained and published in the next main-loop iteration.
                if not _PC_DIAG["submit_logged"]:
                    print(f"[Integration][DIAG] first PC submit to worker: cam={cam_name} "
                          f"publish_pc_topics={publish_pc_topics} pc_cam_set={sorted(pc_cam_set)}", flush=True)
                    _PC_DIAG["submit_logged"] = True
                _t0_pc = time.time()
                pc_worker.submit(cam_name, _pc_build_kwargs, metadata=_pc_metadata)
                _perf_pc_t += time.time() - _t0_pc
                pc7 = None
            else:
                # Synchronous path (fallback or --pc_backend legacy).
                pc_frame = pc_pipeline.build_frame(**_pc_build_kwargs)
                removed_below_plane = pc_frame.removed_below_plane
                removed_outside_object_box = pc_frame.removed_outside_object_box
                sensor_node.publish(f"SimPub/Sensors/{cam_name}/pc", pc_frame.payload)
                if sensor_state.pc_visit_started_t is not None:
                    sensor_state.pc_visit_publish_counts[cam_name] = (
                        sensor_state.pc_visit_publish_counts.get(cam_name, 0) + 1
                    )
                if metrics is not None:
                    _published_wall = time.time()
                    metrics.observe(
                        f"pc_render_to_publish_ms.{cam_name}",
                        max(0.0, _published_wall - _pc_metadata["rendered_wall"]) * 1000.0,
                    )
                    if _pc_metadata["state_wall"] is not None:
                        metrics.observe(
                            f"pc_state_to_publish_ms.{cam_name}",
                            max(0.0, _published_wall - float(_pc_metadata["state_wall"])) * 1000.0,
                        )
                _note_declared_capacity_observation(
                    args, sensor_state, cam_name, pc_frame.actual_count,
                    pc_declared_capacity_by_cam[cam_name],
                )
                if not sensor_state.first_pc_logged.get(cam_name, False):
                    print(
                        f"[Integration] First PC publish: cam={cam_name} "
                        f"topic=SimPub/Sensors/{cam_name}/pc mode={args.pc_sampling} "
                        f"stride={args.pc_stride} target_samples={pc_target_sample_budget} "
                        f"actual={pc_frame.actual_count}"
                    )
                    sensor_state.first_pc_logged[cam_name] = True
                if debug_visibility_requested and pc_frame.debug_visibility is not None:
                    artifact_info = unified.save_pc_visibility_debug_artifacts(
                        outdir=pc_debug_visibility_outdir,
                        cam_name=cam_name,
                        rgb_u8=rgb,
                        debug_visibility=pc_frame.debug_visibility,
                        object_bbox_min=object_bbox_min,
                        object_bbox_max=object_bbox_max,
                        cam_t_mj=cam_t_mj,
                        cam_R_mj=cam_R_mj,
                        fx=fx,
                        fy=fy,
                        cx=cx,
                        cy=cy,
                    )
                    sensor_state.pc_debug_visibility_saved[cam_name] = True
                    dbg = pc_frame.debug_visibility
                    kept_count = int(dbg.xyz_world_mj_kept.shape[0])
                    if kept_count > 0:
                        dbg_min = np.round(dbg.xyz_world_mj_kept.min(axis=0), 4)
                        dbg_max = np.round(dbg.xyz_world_mj_kept.max(axis=0), 4)
                        dbg_centroid = np.round(dbg.xyz_world_mj_kept.mean(axis=0), 4)
                    else:
                        dbg_min = np.array([], dtype=np.float32)
                        dbg_max = np.array([], dtype=np.float32)
                        dbg_centroid = np.array([], dtype=np.float32)
                    print(
                        "[Integration][PCDebug] "
                        f"{cam_name} kept={kept_count} "
                        f"aabb_in_image={artifact_info['aabb_in_image_count']} "
                        f"aabb_front={artifact_info['aabb_front_count']} "
                        f"world_min={dbg_min} world_max={dbg_max} world_centroid={dbg_centroid} "
                        f"artifacts=({artifact_info['rgb_path']}, {artifact_info['selected_path']}, {artifact_info['rejected_path']})"
                    )
                if sensor_state.frame_idx % max(1, args.log_every) == 0:
                    if pc_frame.actual_count > 0:
                        mn = np.round(pc_frame.xyz_unity_m.min(axis=0), 4)
                        mx = np.round(pc_frame.xyz_unity_m.max(axis=0), 4)
                        print(
                            f"[PC] {cam_name} N={pc_frame.actual_count} "
                            f"declared_capacity={pc_frame.declared_capacity} "
                            f"removed_below={removed_below_plane} "
                            f"removed_outside_bbox={removed_outside_object_box} "
                            f"anchor_corr={np.round(cam_anchor_corr, 4)} "
                            f"anchor_global={np.round(sensor_state.anchor_corr_global, 4)} "
                            f"updated={anchor_corr_updated} "
                            f"bbox_min={mn} bbox_max={mx}"
                        )
                    else:
                        print(
                            f"[PC] {cam_name} N=0 declared_capacity={pc_frame.declared_capacity} "
                            f"removed_below={removed_below_plane} "
                            f"removed_outside_bbox={removed_outside_object_box} "
                            f"anchor_corr={np.round(cam_anchor_corr, 4)} "
                            f"anchor_global={np.round(sensor_state.anchor_corr_global, 4)} "
                            f"updated={anchor_corr_updated} "
                            "(depth/plane/object filters)"
                        )
                if args.rgbd_include_pc:
                    pc7 = unified.pc7_array(pc_frame.xyz_unity_m, pc_frame.rgb_u8, size=float(args.pc_size))

        if (publish_rgb or publish_rgb_panel) and rgb_jpg is not None:
            _t0 = time.time()
            sensor_node.publish(f"SimPub/Sensors/{cam_name}/rgb", rgb_jpg)
            _perf_publish_t += time.time() - _t0
            if publish_rgb and cam_name in thumbnail_cam_set:
                sensor_state.last_thumbnail_pub_t = now
            if metrics is not None:
                metrics.observe("rgb_bytes_kb", len(rgb_jpg) / 1024.0)
            if not sensor_state.first_rgb_logged.get(cam_name, False):
                print(f"[Integration] First RGB publish: cam={cam_name}")
                sensor_state.first_rgb_logged[cam_name] = True

        if publish_rgbd and rgb_jpg is not None:
            meta = {
                "calibrated_extrinsic_applied": bool(calib is not None),
                "calibration_source": calib.source if calib else "",
                "pc_clip_below_table": bool(args.pc_clip_below_table),
                "pc_removed_below_plane": int(removed_below_plane),
                "pc_object_only": bool(args.pc_object_only),
                "pc_removed_outside_object_box": int(removed_outside_object_box),
                "plane_source": table_plane.source if table_plane else "",
                "anchor_source_mode": args.pc_anchor_source,
                "anchor_has_world_point": bool(anchor_world_mj is not None),
                "anchor_world_point_mj": [] if anchor_world_mj is None else [float(v) for v in anchor_world_mj],
                "pc_anchor_corr_mj": [float(v) for v in cam_anchor_corr] if args.pc and cam_name in pc_cam_set else [],
                "pc_anchor_corr_global_mj": [float(v) for v in sensor_state.anchor_corr_global],
                "pc_anchor_corr_updated": bool(anchor_corr_updated) if args.pc and cam_name in pc_cam_set else False,
            }
            blob = unified.build_rgbd_blob(
                cam_name=cam_name,
                width=args.w,
                height=args.h,
                fovy_deg=fovy,
                timestamp=timestamp,
                rgb_jpg=rgb_jpg,
                depth_f32_m=depth_m,
                intrinsics={"fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy)},
                cam_pose_mj={
                    "pos": [float(x) for x in cam_t_mj],
                    "quat_wxyz": [float(x) for x in cam_q_mj],
                    "R": [[float(v) for v in row] for row in cam_R_mj.tolist()],
                },
                cam_pose_unity={
                    "pos": [float(x) for x in cam_t_unity],
                    "quat_xyzw": [float(x) for x in cam_q_unity],
                },
                pc7=pc7 if args.rgbd_include_pc else None,
                extra_meta=meta,
            )
            sensor_node.publish(f"SimPub/Sensors/{cam_name}/rgbd", blob)
            if not sensor_state.first_rgbd_logged.get(cam_name, False):
                print(f"[Integration] First RGBD publish: cam={cam_name}")
                sensor_state.first_rgbd_logged[cam_name] = True

    total_t = time.time() - now
    budget_t = 1.0 / max(effective_fps, 1e-6)
    if getattr(args, "perf_log", False):
        print(
            f"[Perf] frame={sensor_state.frame_idx} mode={mode_label} fps={effective_fps:.0f} "
            f"render={_perf_render_t*1000:.1f}ms "
            f"jpeg={_perf_jpeg_t*1000:.1f}ms "
            f"pc_submit={_perf_pc_t*1000:.1f}ms "
            f"total={total_t*1000:.1f}ms "
            f"budget={budget_t*1000:.1f}ms"
        )
    if metrics is not None:
        metrics.observe("render_ms", _perf_render_t * 1000.0)
        metrics.observe("jpeg_ms", _perf_jpeg_t * 1000.0)
        metrics.observe("pc_submit_ms", _perf_pc_t * 1000.0)
        metrics.observe("publish_ms", _perf_publish_t * 1000.0)
        metrics.observe("tick_total_ms", total_t * 1000.0)
        metrics.observe("budget_ms", budget_t * 1000.0)
        if total_t > budget_t:
            metrics.incr("over_budget_frames")
        metrics.incr("publish_ticks")
        metrics.set_label("adaptive_mode", mode_label)
        metrics.set_label("effective_fps", round(effective_fps, 1))
        metrics.set_label("peers", int(peer_count))
        metrics.set_label("pc_topics", bool(publish_pc_topics))
        metrics.set_label("rgb_panel_topics", bool(publish_rgb_panel_topics))
        # Whether the sim these point clouds depict is still being updated. Without this a
        # frozen upstream is indistinguishable from a stationary scene in the metrics: the
        # publish rates stay perfect while the content never changes.
        _mirror = globals().get("_ACTIVE_POLICY_MIRROR")
        if _mirror is not None and getattr(_mirror, "enabled", False):
            metrics.set_label("upstream_stale", bool(getattr(_mirror, "stale", False)))
            metrics.set_label("upstream_stale_age_s",
                              round(float(getattr(_mirror, "stale_age_s", 0.0)), 2))
            metrics.set_label("policy_seq", getattr(_mirror, "last_applied_seq", None))
            metrics.set_label("policy_frame_idx", int(getattr(_mirror, "latest_frame_idx", 0)))
            metrics.set_label("policy_episode_id", str(getattr(_mirror, "last_episode_id", "") or ""))
            metrics.set_label("policy_mode", str(getattr(_mirror, "latest_mode", "")))
            metrics.set_label("policy_paused", bool(getattr(_mirror, "latest_paused", False)))
            metrics.set_label("intervention_phase",
                              str(getattr(_mirror, "latest_intervention_phase", "")))
            metrics.set_label("qpos_delta_norm",
                              round(float(getattr(_mirror, "last_qpos_delta_norm", 0.0)), 8))
            _last_change = getattr(_mirror, "last_state_change_wall", None)
            metrics.set_label(
                "state_change_age_s",
                None if _last_change is None else round(max(0.0, time.time() - _last_change), 2),
            )
    # Advance the cadence target past `now`. Guard against a stale/non-positive next_sensor_t
    # (e.g. reset to 0 or left far behind after a pause): if it is more than one second behind,
    # snap forward in O(1) instead of incrementing in a multi-billion-iteration loop that would
    # freeze the main thread. Normal small gaps fall through to the tight loop unchanged.
    period = 1.0 / max(effective_fps, 1e-6)
    if sensor_state.next_sensor_t < now - 1.0:
        sensor_state.next_sensor_t = now + period
    else:
        while sensor_state.next_sensor_t <= now:
            sensor_state.next_sensor_t += period
    sensor_state.frame_idx += 1
    return table_plane, sensor_state


def apply_object_control(
    *,
    args,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: unified.GLFWMjrRenderer,
    movable_objects: List[unified.MovableBody],
    selected_object_idx: int,
    key_latch: Dict[int, bool],
    next_object_move_time: float,
) -> Tuple[int, float]:
    if not args.object_control or len(movable_objects) == 0:
        return selected_object_idx, next_object_move_time

    if unified.key_pressed_once(renderer, key_latch, glfw.KEY_TAB):
        selected_object_idx = (selected_object_idx + 1) % len(movable_objects)
        print(f"[Integration][ObjectControl] Selected: {movable_objects[selected_object_idx].body_name}")

    move_delta = np.zeros(3, dtype=np.float64)
    if unified.key_down_any(renderer, [glfw.KEY_UP, glfw.KEY_W]):
        move_delta[0] -= float(args.object_step_xy)
    if unified.key_down_any(renderer, [glfw.KEY_DOWN, glfw.KEY_S]):
        move_delta[0] += float(args.object_step_xy)
    if unified.key_down_any(renderer, [glfw.KEY_LEFT, glfw.KEY_A]):
        move_delta[1] -= float(args.object_step_xy)
    if unified.key_down_any(renderer, [glfw.KEY_RIGHT, glfw.KEY_D]):
        move_delta[1] += float(args.object_step_xy)
    if unified.key_down_any(renderer, [glfw.KEY_PAGE_UP, glfw.KEY_Q, glfw.KEY_R]):
        move_delta[2] += float(args.object_step_z)
    if unified.key_down_any(renderer, [glfw.KEY_PAGE_DOWN, glfw.KEY_E, glfw.KEY_F]):
        move_delta[2] -= float(args.object_step_z)

    if np.any(np.abs(move_delta) > 0.0):
        now_move = time.time()
        if now_move >= next_object_move_time:
            selected = movable_objects[selected_object_idx]
            unified.apply_movable_translation(model, data, selected, move_delta)
            next_object_move_time = now_move + (1.0 / max(1e-6, float(args.object_move_rate_hz)))
            pos_now = np.round(data.qpos[selected.qpos_adr : selected.qpos_adr + 3], 4)
            print(f"[Integration][ObjectControl] {selected.body_name} -> pos={pos_now}")
    return selected_object_idx, next_object_move_time


def apply_render_flags(renderer, args, *, announce: bool = True) -> None:
    """Strip shadows/reflections/haze/skybox from a renderer's scene.

    Worth ~25-50% of per-frame render time, and it has to be re-applied whenever the
    MjvScene is rebuilt (a fresh MjvScene comes back with MuJoCo's defaults). These are
    `mjtRndFlag` on `scene.flags[]`, NOT `mjtVisFlag` on `opt.flags[]` -- the latter does not
    even contain these constants, so using it silently does nothing.
    """
    try:
        _flag = mujoco.mjtRndFlag
        if not getattr(args, "render_shadows", False):
            renderer.scene.flags[_flag.mjRND_SHADOW] = 0
        if not getattr(args, "render_reflections", False):
            renderer.scene.flags[_flag.mjRND_REFLECTION] = 0
        if not getattr(args, "render_haze", False):
            renderer.scene.flags[_flag.mjRND_HAZE] = 0
        if not getattr(args, "render_skybox", False):
            renderer.scene.flags[_flag.mjRND_SKYBOX] = 0
        if announce:
            print("[Integration] Render flags re-applied after model swap.")
    except Exception as exc:
        print(f"[Integration][WARN] could not apply render flags: {exc}")


def resolve_runtime_setup(args):
    if args.object_control:
        args.show_mujoco_window = True

    publish_rgb_topic = not args.no_rgb_topic
    object_bbox_min = np.asarray(args.pc_object_bbox_min, dtype=np.float64).reshape(3)
    object_bbox_max = np.asarray(args.pc_object_bbox_max, dtype=np.float64).reshape(3)
    if args.pc_object_only and np.any(object_bbox_min >= object_bbox_max):
        raise ValueError("--pc_object_bbox_min must be strictly smaller than --pc_object_bbox_max")

    extrinsics_path = args.cam_extrinsics_json
    if not extrinsics_path and args.cam_extrinsics_profile == "lab_standard":
        if unified.STANDARD_EXTRINSICS_PROFILE.exists():
            extrinsics_path = str(unified.STANDARD_EXTRINSICS_PROFILE)

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    trajectory = ReplayTrajectory(Path(args.trajectory))
    replay_bindings = resolve_replay_robot_bindings(model)
    scene_seed = settle_scene_seed(args, model, data)
    startup_scene_diagnostics = build_scene_seed_diagnostics(
        model=model,
        scene_seed=scene_seed,
        trajectory=trajectory,
        replay_bindings=replay_bindings,
    )
    current_idx = trajectory.apply_frame(
        int(np.clip(args.start_idx, 0, len(trajectory) - 1)),
        data,
        replay_bindings=replay_bindings,
        scene_seed=scene_seed,
        restore_scene=True,
    )

    if args.robot_start_q is not None:
        q_override = np.asarray(args.robot_start_q, dtype=np.float64)
        for i, qadr in enumerate(replay_bindings.arm_qpos_idx):
            if 0 <= int(qadr) < data.qpos.shape[0] and i < len(q_override):
                data.qpos[int(qadr)] = q_override[i]
        for dadr in replay_bindings.arm_dof_idx:
            if 0 <= int(dadr) < data.qvel.shape[0]:
                data.qvel[int(dadr)] = 0.0
        for i, cidx in enumerate(replay_bindings.arm_ctrl_idx):
            if 0 <= int(cidx) < data.ctrl.shape[0] and i < len(q_override):
                data.ctrl[int(cidx)] = q_override[i]
        mujoco.mj_forward(model, data)

    available_cams = unified.list_cameras(model)
    cams = unified._choose_cameras(args.cams, available_cams)
    if getattr(args, "rgb_panel_mode", False):
        # Auto-merge the RGB-panel cameras (e.g. "wrist") into the active camera
        # set so the renderer actually renders them. Independent of pc_cam_set /
        # the point cloud pipeline below.
        extra_panel_cams = [c for c in args.rgb_panel_cams if c in available_cams and c not in cams]
        if extra_panel_cams:
            cams = list(cams) + extra_panel_cams
    if not cams:
        raise RuntimeError(f"No valid cameras found. Available cameras: {available_cams}")

    if args.mujoco_window_cam and args.mujoco_window_cam not in cams:
        print(f"[Integration][WARN] --mujoco_window_cam '{args.mujoco_window_cam}' not in active cameras {cams}. "
              f"Falling back to '{cams[0]}'.")
        args.mujoco_window_cam = cams[0]

    rgb_requested = ["wrist"] if args.rgb_cams is None else list(args.rgb_cams)
    rgbd_requested = [] if args.rgbd_cams is None else list(args.rgbd_cams)
    rgb_cam_set = set(unified._choose_cameras(rgb_requested, cams))
    rgbd_cam_set = set([c for c in rgbd_requested if c in cams])

    # RGB-panel mode: independent gated camera set for the Unity diamond-layout
    # single view (wrist/left/right). Does not touch rgb_cam_set/pc_cam_set.
    rgb_panel_cam_set = set()
    if getattr(args, "rgb_panel_mode", False):
        rgb_panel_cam_set = set([c for c in args.rgb_panel_cams if c in cams])

    pc_max_depth = args.pc_max_depth
    if pc_max_depth is None:
        pc_max_depth = float(model.vis.map.zfar) * 0.95

    pc_cam_set = set()
    pc_declared_capacity_by_cam: Dict[str, int] = {}
    pc_target_sample_budget = unified.sampled_point_capacity(args.w, args.h, max(1, int(args.pc_stride)))
    pc_pipeline = None
    if args.pc:
        requested_pc = args.pc_cams if args.pc_cams else list(cams)
        pc_cam_set = set([c for c in requested_pc if c in cams])
        for cam_name in sorted(pc_cam_set):
            pc_declared_capacity_by_cam[cam_name] = unified.resolve_declared_pc_capacity(
                cam_name=cam_name,
                width=args.w,
                height=args.h,
                stride=max(1, int(args.pc_stride)),
                global_cap=int(args.pc_max_points),
                top_cap=int(args.pc_max_points_top),
            )
        pc_pipeline = (
            unified.Open3dCudaPointCloudPipeline()
            if args.pc_backend == "gpu"
            else unified.LegacyCompatPointCloudPipeline()
        )

    pc_debug_visibility_cam_set = set()
    pc_debug_visibility_outdir: Optional[Path] = None
    if args.pc and args.pc_debug_visibility:
        requested_debug_cams = args.pc_debug_visibility_cams if args.pc_debug_visibility_cams else sorted(pc_cam_set)
        pc_debug_visibility_cam_set = set([c for c in requested_debug_cams if c in pc_cam_set])
        missing_debug_cams = [c for c in requested_debug_cams if c not in pc_cam_set]
        if missing_debug_cams:
            print(
                "[Integration][WARN] PC visibility debug cameras not in active PC camera set "
                f"(ignored): {missing_debug_cams}"
            )
        if not pc_debug_visibility_cam_set:
            print("[Integration][WARN] PC visibility debug requested but no valid PC cameras were selected.")
        outdir_value = args.pc_debug_visibility_outdir
        if outdir_value:
            pc_debug_visibility_outdir = Path(outdir_value)
        else:
            pc_debug_visibility_outdir = Path.cwd() / f"pc_debug_{time.strftime('%Y%m%d_%H%M%S')}"

    extrinsics = unified.load_camera_extrinsics(extrinsics_path)

    if getattr(args, "scene_push_delay", 0.0) > 0.0:
        print(
            f"[Integration] --scene_push_delay={args.scene_push_delay:.1f}s: "
            "waiting before MujocoPublisher init to stagger SpawnSimScene across sessions."
        )
        time.sleep(args.scene_push_delay)

    # Always start in thumbnail mode (no MujocoPublisher). The PROMOTE
    # command from the multi-session selector constructs MujocoPublisher
    # in-process when needed (see _handle_promote below). This eliminates
    # the prior re-exec dance + its GLFW context / port-handoff races.
    # The legacy `--no_mujoco_publisher` flag is now effectively default-on;
    # passing it is still accepted (for back-compat) but a no-op.
    if getattr(args, "no_mujoco_publisher", False):
        print("[Integration] --no_mujoco_publisher (legacy): startup is always thumbnail mode now.")
    else:
        print("[Integration] startup is always thumbnail mode; send PROMOTE to enable MujocoPublisher.")
    scene_pub = None
    if getattr(args, "no_xr_device", False):
        mq3 = None
        print("[Integration] --no_xr_device: skipping XRNodeManager dashboard and MetaQuest3 VR device connection.")
    else:
        # XRNodeManager would normally be initialized by MujocoPublisher as a
        # side effect. Initialize it directly only when controller/XR input is
        # actually requested; thumbnail sessions use cmd_port forwarding and do
        # not need the 127.0.0.1:5000 dashboard in every process.
        from simpub.core.node_manager import init_xr_node_manager
        init_xr_node_manager(args.host_ip or args.host)
        mq3 = MetaQuest3(args.unity_node)
        print(
            f"[Integration] MetaQuest3 initialized (unity_node='{args.unity_node}'). "
            "A/B/X/Y buttons active once Quest is discovered. "
            "Watch for 'XRDevice Connected' or 'Auto-selecting' in logs."
        )

    topic_list = [f"SimPub/Sensors/{cam}/rgbd" for cam in sorted(rgbd_cam_set)]
    if publish_rgb_topic:
        topic_list.extend([f"SimPub/Sensors/{cam}/rgb" for cam in sorted(rgb_cam_set)])
    if args.pc:
        topic_list.extend([f"SimPub/Sensors/{cam}/pc" for cam in sorted(pc_cam_set)])
    if getattr(args, "rgb_panel_mode", False):
        topic_list.extend([f"SimPub/Sensors/{cam}/rgb" for cam in sorted(rgb_panel_cam_set)])
    topic_list = sorted(list(set(topic_list)))
    sensor_node_inner = unified.SensorNode(
        name="SimPub",
        bind_ip=args.bind_ip,
        host_ip=args.host_ip or args.host,
        service_port=args.service_port,
        topic_port=args.topic_port,
        topic_list=topic_list,
    )
    sensor_node = IsolatedSensorPublisher(sensor_node_inner)
    sensor_node.start()

    renderer = unified.GLFWMjrRenderer(
        model,
        args.w,
        args.h,
        visible=args.show_mujoco_window,
        window_title="intervention_vr_runtime",
    )

    # Strip "pretty but expensive" visual effects from the per-frame render
    # pass. At 60 Hz × 5 cameras × N sessions these add up. Shadows are the
    # biggest single win (~20-40% off lit RGB renders). Reflections + haze +
    # skybox each shave a few percent more. All are CLI-overridable; default
    # off for performance, on if you want the pretty MuJoCo window.
    #
    # CRITICAL: these are RENDER flags (mjtRndFlag, on scene.flags[]), NOT
    # visualization flags (mjtVisFlag, on opt.flags[]). Easy mistake — vis
    # flags don't include shadow/reflection/etc.
    try:
        apply_render_flags(renderer, args, announce=False)
        _flag = mujoco.mjtRndFlag
        print(
            f"[Integration] Render flags: "
            f"shadow={bool(renderer.scene.flags[_flag.mjRND_SHADOW])} "
            f"reflection={bool(renderer.scene.flags[_flag.mjRND_REFLECTION])} "
            f"haze={bool(renderer.scene.flags[_flag.mjRND_HAZE])} "
            f"skybox={bool(renderer.scene.flags[_flag.mjRND_SKYBOX])}"
        )
    except Exception as ex:
        print(f"[Integration][WARN] Could not toggle render flags: {ex}")

    movable_objects: List[unified.MovableBody] = []
    if args.object_control:
        movable_objects = unified.discover_movable_free_bodies(model, args.object_control_bodies)
        if len(movable_objects) == 0:
            print("[Integration][WARN] Object control enabled but no free-joint objects were found.")
            args.object_control = False

    flip_z = (not args.pc_no_flip_z) or args.pc_flip_z
    table_plane = scene_seed.table_plane if args.pc and args.pc_clip_below_table else None

    anchor_site_id = -1
    anchor_body_id = -1
    if args.pc and args.pc_anchor_auto_translate:
        if args.pc_anchor_site:
            anchor_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, args.pc_anchor_site)
        anchor_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, args.pc_anchor_body)

    manual_anchor_world = (
        np.asarray(args.pc_anchor_world_xyz, dtype=np.float64).reshape(3)
        if args.pc_anchor_world_xyz is not None
        else None
    )
    sensor_state = SensorPublishState(
        next_sensor_t=time.time(),
        anchor_corr_by_cam={cam: np.zeros(3, dtype=np.float64) for cam in pc_cam_set},
    )

    return {
        "model": model,
        "data": data,
        "trajectory": trajectory,
        "replay_bindings": replay_bindings,
        "scene_seed": scene_seed,
        "startup_scene_diagnostics": startup_scene_diagnostics,
        "current_idx": current_idx,
        "cams": cams,
        "rgb_cam_set": rgb_cam_set,
        "rgbd_cam_set": rgbd_cam_set,
        "pc_cam_set": pc_cam_set,
        "rgb_panel_cam_set": rgb_panel_cam_set,
        "publish_rgb_topic": publish_rgb_topic,
        "pc_pipeline": pc_pipeline,
        "pc_target_sample_budget": pc_target_sample_budget,
        "pc_max_depth": pc_max_depth,
        "pc_declared_capacity_by_cam": pc_declared_capacity_by_cam,
        "pc_debug_visibility_cam_set": pc_debug_visibility_cam_set,
        "pc_debug_visibility_outdir": pc_debug_visibility_outdir,
        "object_bbox_min": object_bbox_min,
        "object_bbox_max": object_bbox_max,
        "extrinsics": extrinsics,
        "scene_pub": scene_pub,
        "mq3": mq3,
        "sensor_node": sensor_node,
        "renderer": renderer,
        "movable_objects": movable_objects,
        "table_plane": table_plane,
        "anchor_site_id": anchor_site_id,
        "anchor_body_id": anchor_body_id,
        "manual_anchor_world": manual_anchor_world,
        "flip_z": flip_z,
        "sensor_state": sensor_state,
    }



def _run_cmd_capture(cmd: List[str]) -> Tuple[int, str, str]:
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
        return int(proc.returncode), (proc.stdout or ""), (proc.stderr or "")
    except FileNotFoundError as exc:
        return 127, "", str(exc)
    except Exception as exc:
        return 1, "", str(exc)


def _adb_cmd_base(serial: Optional[str]) -> List[str]:
    base = ["adb"]
    if serial:
        base.extend(["-s", str(serial)])
    return base


def _configure_transport(args) -> None:
    if args.transport != "usb":
        if args.transport == "wifi":
            effective = args.host_ip or args.host
            print(f"[Integration][Net] transport=wifi publisher_ip={effective} bind_ip={args.bind_ip}")
        return

    usb_ip = str(args.usb_publisher_ip).strip() or "127.0.0.1"
    effective_host = args.host_ip or args.host
    print(
        "[Integration][Net] transport=usb keeping control-plane host "
        f"'{effective_host}' (XR discovery/scenes use this)."
    )
    print(
        "[Integration][Net] Expected Quest sensor subscriber publisherIp="
        f"'{usb_ip}' when using adb reverse."
    )
    print(f"[Integration][Net] bind_ip={args.bind_ip}")

    if not args.usb_auto_reverse:
        print(
            "[Integration][Net] usb_auto_reverse disabled. "
            "Run manually: adb reverse tcp:<port> tcp:<port>."
        )
        return

    adb_base = _adb_cmd_base(args.usb_adb_serial)
    rc, out, err = _run_cmd_capture(adb_base + ["devices"])
    if rc != 0:
        print(
            "[Integration][WARN] adb devices failed. "
            f"rc={rc} err='{err.strip()}'"
        )
        return
    if "	device" not in out:
        print(
            "[Integration][WARN] adb reports no ready device. "
            "Connect Quest via USB-C and accept USB debugging prompt."
        )
        return

    ports: List[int] = [int(args.topic_port), int(args.service_port)]
    if args.usb_extra_reverse_ports:
        ports.extend([int(p) for p in args.usb_extra_reverse_ports])
    seen = set()
    ordered_ports: List[int] = []
    for p in ports:
        if p <= 0:
            continue
        if p in seen:
            continue
        seen.add(p)
        ordered_ports.append(p)

    if args.usb_reverse_remove_all:
        # Optional cleanup for stale mappings; disabled by default to avoid side effects.
        _run_cmd_capture(adb_base + ["reverse", "--remove-all"])

    for port in ordered_ports:
        cmd = adb_base + ["reverse", f"tcp:{port}", f"tcp:{port}"]
        rc, _, err = _run_cmd_capture(cmd)
        if rc == 0:
            print(f"[Integration][Net] adb reverse active: tcp:{port} -> tcp:{port}")
        else:
            print(
                "[Integration][WARN] adb reverse failed for "
                f"tcp:{port}. rc={rc} err='{err.strip()}'"
            )

    rc, out, err = _run_cmd_capture(adb_base + ["reverse", "--list"])
    if rc == 0:
        listed = out.strip() if out.strip() else "(none)"
        print("[Integration][Net] adb reverse list:")
        print(listed)
    else:
        print(f"[Integration][WARN] adb reverse --list failed: rc={rc} err='{err.strip()}'")


def _trajectory_velocity_diagnostic(
    trajectory: ReplayTrajectory,
    *,
    replay_speed: float,
    max_dq: float,
) -> Optional[Dict[str, float]]:
    try:
        ctrl = np.asarray(trajectory.ctrl_sim, dtype=np.float64)
        sim_t = np.asarray(trajectory.sim_t, dtype=np.float64).reshape(-1)
        if ctrl.ndim != 2 or ctrl.shape[0] < 2 or ctrl.shape[1] < 7 or sim_t.shape[0] < 2:
            return None
        dt = np.diff(sim_t)
        valid = dt > 1e-6
        if not np.any(valid):
            return None
        dq = np.diff(ctrl[:, :7], axis=0)[valid] / dt[valid, None]
        required_wall_dq = np.abs(dq) * float(replay_speed)
        per_joint = np.max(required_wall_dq, axis=0)
        limits = np.full(7, float(max_dq), dtype=np.float64)
        limits[6] = float(max_dq) * float(JOINT7_MAX_DQ_SCALE)
        ratio = per_joint / np.maximum(limits, 1e-6)
        return {
            "max_required_dq": float(np.max(per_joint)),
            "max_limit_ratio": float(np.max(ratio)),
            "worst_joint": float(int(np.argmax(ratio)) + 1),
        }
    except Exception:
        return None


class RiskPublisherThread:
    """Publishes ACC (Action Chunk Consistency) risk score on SimPub/Status/risk.

    Three-tier fallback:
      1. Live ACC from PolicyPlayer via PolicyStateMirror (primary — policy execution path).
      2. Trajectory-based velocity-normalized jerk of ctrl_sim (replay-only fallback).
      3. Sinusoidal dummy (bare sim, no policy, no trajectory).
    """

    TOPIC   = "SimPub/Status/risk"
    PERIOD  = 20.0   # seconds for dummy sine wave fallback

    def __init__(
        self,
        sensor_node,
        *,
        policy_mirror=None,
        trajectory=None,
        get_frame_idx=None,
        scale: float = 3.0,
        session_index: int = 0,
        hz: float = 2.0,
        metrics=None,
    ):
        self._node      = sensor_node
        self._mirror    = policy_mirror
        self._metrics   = metrics
        self._traj      = trajectory
        self._get_frame = get_frame_idx
        self._scale     = scale
        self._phase     = session_index * 2.0 * math.pi / 15.0
        self._hz        = hz
        self._stop      = threading.Event()
        self._thread    = threading.Thread(target=self._run, name="RiskPub", daemon=True)

        # Precompute velocity norms from trajectory ctrl_sim for the fallback path
        self._vel_norms: Optional[np.ndarray] = None
        if trajectory is not None and hasattr(trajectory, "ctrl_sim"):
            ctrl = np.asarray(trajectory.ctrl_sim, dtype=np.float64)
            if ctrl.ndim == 2 and ctrl.shape[0] > 2 and ctrl.shape[1] >= 7:
                self._vel_norms = np.mean(np.abs(np.diff(ctrl[:, :7], axis=0)), axis=0) + 1e-6

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _compute_risk(self) -> float:
        # 1. Primary: live ACC score from PolicyPlayer via state mirror
        if self._mirror is not None and self._mirror.enabled:
            return float(np.clip(self._mirror.latest_acc_risk, 0.0, 10.0))

        # 2. Fallback: trajectory-based velocity-normalized jerk of ctrl_sim
        if self._traj is not None and self._vel_norms is not None and self._get_frame is not None:
            idx = self._get_frame()
            n = len(self._traj)
            if idx >= 2 and idx < n:
                ctrl = np.asarray(self._traj.ctrl_sim[idx - 2 : idx + 1, :7], dtype=np.float64)
                v1 = ctrl[1] - ctrl[0]
                v2 = ctrl[2] - ctrl[1]
                jerk = v2 - v1
                acc_raw = float(np.mean(np.abs(jerk) / self._vel_norms))
                return float(np.clip(acc_raw * self._scale, 0.0, 10.0))

        # 3. Last resort: sinusoidal dummy
        return 5.0 + 5.0 * math.sin(2.0 * math.pi * time.time() / self.PERIOD + self._phase)

    def _run(self) -> None:
        interval = 1.0 / max(0.1, self._hz)
        while not self._stop.wait(interval):
            risk = self._compute_risk()
            try:
                self._node.publish(self.TOPIC, f"{risk:.3f}".encode("utf-8"))
            except Exception:
                pass
            # Record the RAW acc_risk (not `risk` above, which is scaled/clipped to [0,10]
            # for the Quest risk bar). The OOD supervisors compare against the raw
            # acc_risk from the policy state, so the threshold must be calibrated from the
            # same numbers. Only recorded when a live policy is actually driving it —
            # the jerk and sine fallbacks would poison the distribution.
            if self._metrics is not None and self._mirror is not None and self._mirror.enabled:
                try:
                    self._metrics.observe("acc_risk", float(self._mirror.latest_acc_risk))
                except Exception:
                    pass


class PCWorkerThread:
    """Background GPU point cloud build worker(s).

    Per-camera dict semantics: submit() always replaces the pending entry for a
    camera so each worker always processes the freshest submitted frame.

    With n_workers=1 (default): one thread processes cameras sequentially.
    With n_workers=3: three threads each grab one camera from the shared dict and
    build in parallel — all cameras build simultaneously, cutting latency to ~1
    build cycle instead of 3×. Requires a thread-safe GPU pipeline (Open3D CUDA
    operations release the GIL during .cpu().numpy() so threads overlap naturally).

    Timing: when timing=True each build logs
        [PCBuild] cam=top  build=42.3ms N=198441
    so you can measure real GPU build time and verify parallel overlap.
    """

    def __init__(self, pc_pipeline, n_workers: int = 1, timing: bool = False, metrics=None,
                 escalate_after: int = 10):
        self._pipeline = pc_pipeline
        self._timing = timing
        # Metrics collector is written from the worker threads; MetricsCollector.observe
        # is a deque append (GIL-atomic) and emit() holds a lock, so this is safe.
        self._metrics = metrics
        # Per-camera dict: latest pending frame per camera name.
        self._pending: dict = {}
        self._pending_lock = threading.Lock()
        # Semaphore counts available (un-grabbed) camera jobs.  submit() releases
        # one token per NEW camera added; workers acquire one token per job taken.
        # This lets N worker threads each grab one independent camera without
        # spinning and without a separate wakeup event per thread.
        self._work_sem = threading.Semaphore(0)
        # Completed builds are latest-per-camera too. An unbounded FIFO here made the main
        # loop faithfully publish obsolete builds after a worker-side slowdown.
        self._completed: dict = {}
        self._completed_lock = threading.Lock()
        self._stop = threading.Event()
        # Consecutive-failure tracking, so a permanently broken pipeline escalates once
        # instead of printing the same line hundreds of times.
        self._consec_errors = 0
        self.last_error = ""
        self._escalate_after = max(1, int(escalate_after))
        self._next_escalate_t = 0.0
        self._threads = [
            threading.Thread(target=self._run, daemon=True, name=f"PCWorker-{i}")
            for i in range(max(1, n_workers))
        ]
        for t in self._threads:
            t.start()

    def submit(self, cam_name: str, build_kwargs: dict, metadata: Optional[dict] = None) -> None:
        """Always-latest submit: replaces any prior pending frame for this camera.
        Only releases a semaphore token if this is a brand-new camera slot (not an
        overwrite), so the token count stays equal to the number of unique pending
        cameras."""
        with self._pending_lock:
            is_new = cam_name not in self._pending
            self._pending[cam_name] = (build_kwargs, dict(metadata or {}))
        if is_new:
            self._work_sem.release()

    def drain(self) -> list:
        """Return the freshest completed frame for each camera."""
        with self._completed_lock:
            results = list(self._completed.items())
            self._completed.clear()
        return results

    def clear_input(self) -> int:
        """Drop only pending (not-yet-built) jobs. Completed frames on the output
        queue are kept so a warm-up refresh does not throw away a freshly built
        cloud that is waiting to be drained/published."""
        with self._pending_lock:
            n = len(self._pending)
            self._pending.clear()
        # Drain excess semaphore tokens so workers don't spin on empty dict.
        drained = 0
        while drained < n:
            if self._work_sem.acquire(blocking=False):
                drained += 1
            else:
                break
        return n

    def clear_pending(self) -> int:
        """Drop pending PC jobs AND completed results. Use on EXIT_SINGLE."""
        n = self.clear_input()
        with self._completed_lock:
            n += len(self._completed)
            self._completed.clear()
        return n

    def stop(self) -> None:
        self._stop.set()
        # Unblock any workers waiting on the semaphore so they can see _stop.
        for _ in self._threads:
            self._work_sem.release()
        for t in self._threads:
            t.join(timeout=3.0)

    def _run(self) -> None:
        _diag_first = True
        while not self._stop.is_set():
            if not self._work_sem.acquire(timeout=0.1):
                continue
            if self._stop.is_set():
                break
            # Grab exactly ONE camera's job from the shared dict.
            job = None
            with self._pending_lock:
                if self._pending:
                    cam_name, pending = next(iter(self._pending.items()))
                    del self._pending[cam_name]
                    job = (cam_name, pending)
            if job is None:
                continue
            cam_name, (kwargs, metadata) = job
            t0 = time.time()
            try:
                submitted_wall = metadata.get("submitted_wall")
                if self._metrics is not None and submitted_wall is not None:
                    self._metrics.observe(
                        f"pc_worker_queue_ms.{cam_name}",
                        max(0.0, t0 - float(submitted_wall)) * 1000.0,
                    )
                if _diag_first:
                    print(f"[PCWorker][DIAG] first build_frame START cam={cam_name}", flush=True)
                pc_frame = self._pipeline.build_frame(**kwargs)
                elapsed_ms = (time.time() - t0) * 1000.0
                if _diag_first:
                    print(f"[PCWorker][DIAG] first build_frame DONE cam={cam_name} "
                          f"build={elapsed_ms:.1f}ms N={getattr(pc_frame, 'actual_count', '?')}",
                          flush=True)
                    _diag_first = False
                elif self._timing:
                    print(f"[PCBuild] cam={cam_name:<6} build={elapsed_ms:.1f}ms "
                          f"N={getattr(pc_frame, 'actual_count', '?')}", flush=True)
                if self._metrics is not None:
                    self._metrics.observe("pc_build_ms", elapsed_ms)
                    self._metrics.observe(f"pc_build_ms.{cam_name}", elapsed_ms)
                    self._metrics.incr(f"pc_points.{cam_name}",
                                       float(getattr(pc_frame, "actual_count", 0) or 0))
                with self._completed_lock:
                    metadata["build_done_wall"] = time.time()
                    self._completed[cam_name] = (pc_frame, metadata)
                self._consec_errors = 0
            except Exception as exc:
                self._note_build_error(cam_name, exc)


    # ------------------------------------------------------------------ error reporting

    # Substrings that identify a GPU out-of-memory failure. Open3D reports it as an
    # [Open3D Error] with a CUDA runtime message, not as a Python MemoryError.
    _OOM_MARKERS = ("out of memory", "cuda runtime error", "memorymanager")

    def _note_build_error(self, cam_name: str, exc: Exception) -> None:
        """Count, diagnose and rate-limit a build failure.

        This used to be a bare print. A permanently broken session then looked exactly like
        a slow one: in a 9-window run 702 identical CUDA-OOM lines scrolled past, six of the
        nine sessions produced ZERO points for the whole session, and nothing said so. The
        operator saw "point clouds only work on some windows" with no cause anywhere.
        """
        self._consec_errors += 1
        self.last_error = f"{type(exc).__name__}: {exc}"
        if self._metrics is not None:
            self._metrics.incr("pc_build_errors")

        n = self._consec_errors
        if n == 1:
            print(f"[PCWorker][WARN] build_frame failed for {cam_name}: {exc}", flush=True)
            return
        if n < self._escalate_after:
            return
        now = time.time()
        if n > self._escalate_after and now < self._next_escalate_t:
            return
        self._next_escalate_t = now + 5.0

        text = str(exc).lower()
        if any(marker in text for marker in self._OOM_MARKERS):
            print(
                f"[PCWorker][ERROR] point cloud build has failed {n} consecutive times on "
                f"cam={cam_name}.\n"
                "  Diagnosis: GPU OUT OF MEMORY. No point cloud will reach the headset from\n"
                "  this session until VRAM is freed. This GPU is shared by every VR runtime\n"
                "  and every policy process (each holds a CUDA context and an MjrContext).\n"
                "  Remedies, cheapest first:\n"
                "    * LAB_FAST=1        - 640x480 offscreen instead of 1920x1080 (6.75x\n"
                "                          fewer pixels); physics is IDENTICAL either way\n"
                "    * PC_WORKER_THREADS=1 - each parallel build holds its own GPU buffers\n"
                "    * fewer WINDOWS, or stop policy processes you are not driving\n"
                "    * lower PC_STRIDE / PC_MAX_POINTS\n"
                f"  Last error: {self.last_error}",
                flush=True,
            )
        else:
            print(
                f"[PCWorker][ERROR] point cloud build has failed {n} consecutive times on "
                f"cam={cam_name}; this session is publishing NO point clouds. "
                f"Last error: {self.last_error}",
                flush=True,
            )


def main() -> None:
    args = build_arg_parser().parse_args()
    if float(args.replay_speed) <= 0.0:
        raise ValueError("--replay_speed must be > 0")

    # --peer_fallback_grace_s defaults to 0 (strict selection) whenever ENTER_SINGLE/
    # EXIT_SINGLE is authoritative, and 3.0 only when there is no selection signal to fall
    # back FROM. The old flat 3.0 default let an unselected session promote itself to
    # ACTIVE on peer count alone; an explicit value on the command line still wins.
    if args.peer_fallback_grace_s is None:
        args.peer_fallback_grace_s = (
            0.0 if bool(getattr(args, "selected_sensor_activation", False)) else 3.0
        )

    # Raise OS timer resolution to 1 ms on Windows (default is 15 ms).
    # This prevents time.sleep() from overshooting by up to 14 ms per iteration,
    # which causes cumulative frame-rate jitter at 30+ fps.
    _win_timer_raised = False
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.winmm.timeBeginPeriod(1)
            _win_timer_raised = True
        except Exception:
            pass

    _configure_transport(args)
    # Pre-import robot dependencies so all `import zmq` happens before any
    # ZMQ context singletons are created by MujocoPublisher/SensorNode.
    if args.mirror_robot:
        load_robot_stack()
    setup = resolve_runtime_setup(args)
    model = setup["model"]
    data = setup["data"]
    trajectory: ReplayTrajectory = setup["trajectory"]
    replay_bindings: ReplayRobotBindings = setup["replay_bindings"]
    scene_seed: SceneSeedState = setup["scene_seed"]
    current_idx = int(setup["current_idx"])
    scene_pub = setup["scene_pub"]
    renderer = setup["renderer"]
    sensor_node = setup["sensor_node"]
    mq3: MetaQuest3 = setup["mq3"]
    table_plane = setup["table_plane"]
    sensor_state: SensorPublishState = setup["sensor_state"]
    pc_debug_visibility_cam_set = setup["pc_debug_visibility_cam_set"]
    pc_debug_visibility_outdir = setup["pc_debug_visibility_outdir"]

    # Windowed performance metrics (PART-1). Created before the PC worker so worker
    # builds are captured too. `interface_mode` tags every row with which of the two
    # VR modes produced it, since RGB mode publishes no /pc topics at all.
    metrics: Optional[MetricsCollector] = None
    _metrics_out = getattr(args, "metrics_out", None)
    if _metrics_out or getattr(args, "metrics_summary", False):
        if _metrics_out == "auto":
            _metrics_out = default_metrics_path(session_index=getattr(args, "session_index", 0))
        _interface_mode = (
            "pointcloud" if args.pc
            else ("rgb_panel" if getattr(args, "rgb_panel_mode", False) else "rgb")
        )
        metrics = MetricsCollector(
            out_path=_metrics_out or None,
            window_s=float(getattr(args, "metrics_window_s", 10.0)),
            print_summary=bool(getattr(args, "metrics_summary", False)),
            session_index=int(getattr(args, "session_index", 0)),
            mode=_interface_mode,
        )
        metrics.set_label("target_fps", float(args.fps))
        metrics.set_label("fps_idle", float(getattr(args, "fps_idle", args.fps)))
        metrics.set_label("pc_stride", getattr(args, "pc_stride", None))
        metrics.set_label("pc_max_points", getattr(args, "pc_max_points", None))
        metrics.set_label("jpg_quality", getattr(args, "jpg_quality", None))
        metrics.set_label("resolution", f"{args.w}x{args.h}")
        metrics.set_label("pc_round_robin", bool(getattr(args, "pc_round_robin", False)))
        metrics.set_label("latency_probe", bool(getattr(args, "latency_probe", False)))

    # End-to-end latency probe. Published frames carry a sequence number; the headset
    # echoes it back twice and `LatencyProbe` turns those into a per-camera budget.
    # Independent of `metrics` so the probe can be used on its own.
    global _LATENCY_PROBE
    _LATENCY_PROBE = None
    if bool(getattr(args, "latency_probe", False)):
        if LatencyProbe is None:
            print("[LatencyProbe] tools/latency_stamp.py not importable; probe DISABLED.",
                  flush=True)
        else:
            _LATENCY_PROBE = LatencyProbe(session_index=int(getattr(args, "session_index", 0)))
            print(
                "[LatencyProbe] ON. Stamping published RGB frames; expecting 'LAT|A|..' / "
                "'LAT|B|..' echoes on the command port. If the counts stay at zero the "
                "deployed APK predates LatencyStampProbe.cs -- rebuild before trusting "
                "any absence of samples.",
                flush=True,
            )

    # Background PC worker — decouples GPU build_frame + CUDA→CPU sync from
    # the main render loop. Enabled only when the PC pipeline is active.
    pc_worker: Optional[PCWorkerThread] = (
        PCWorkerThread(
            setup["pc_pipeline"],
            n_workers=getattr(args, "pc_worker_threads", 1),
            timing=getattr(args, "perf_log", False),
            metrics=metrics,
            escalate_after=max(1, int(getattr(args, "pc_build_error_escalate_n", 10) or 10)),
        )
        if args.pc and setup["pc_cam_set"]
        else None
    )
    policy_state_mirror = PolicyStateMirror(
        args.policy_state_host,
        int(args.policy_state_port or 0),
        float(args.policy_state_timeout_s),
    )
    # Identity of OUR base geometry, so a policy running a different XML is refused rather
    # than mirrored onto the wrong scene.
    try:
        policy_state_mirror.base_scene_sha256 = scene_closure_sha256(args.xml)
    except Exception as exc:
        print(f"[Variant][WARN] scene closure hash failed for {args.xml}: {exc}")

    # Counts point-cloud frames discarded because they were built on the previous model.
    _pc_stale_epoch_drops = [0]
    # GPU-memory reclaim state. Probed once: a future Open3D without release_cache disables
    # the whole path rather than raising once per idle transition.
    _pc_idle_since = [None]
    _pc_cache_released = [False]
    _pc_release_cache_available = [False]
    if bool(args.pc):
        try:
            import open3d as _o3d_probe
            _pc_release_cache_available[0] = callable(
                getattr(getattr(getattr(_o3d_probe, "core", None), "cuda", None),
                        "release_cache", None)
            )
        except Exception:
            _pc_release_cache_available[0] = False
        if not _pc_release_cache_available[0]:
            print("[PC][Lifecycle] open3d.core.cuda.release_cache unavailable; "
                  "idle GPU-memory reclaim is disabled.")
    # Which variant is compiling in the background, so the "not compiled yet" line is logged
    # once per variant rather than once per state message.
    _pending_variant = {"key": None, "since": 0.0, "warned": False}

    # Capacity 2: the pinned base plus the current variant. This process mirrors exactly one
    # session, so it never needs more. Disabled entirely when there is no upstream policy.
    variant_cache = None
    if int(args.policy_state_port or 0) > 0:
        try:
            from model_variants.cache import VariantCache
            variant_cache = VariantCache(
                # base (pinned) + the model being rendered + one background prefetch.
                args.xml, reference_model=model, capacity=3,
                label=f"Variant/VR{getattr(args, 'session_index', 0)}",
            )
            variant_cache.install_base(model)
        except Exception as exc:
            print(f"[Variant][WARN] variant cache unavailable ({exc}); this runtime will "
                  "render every episode with the base model.")
            variant_cache = None
    # publish_sensor_frames() is module-level and does not receive the mirror, but it needs
    # to label each metrics window with whether the upstream sim is still being updated.
    # One session per process, so a module-level handle is unambiguous.
    global _ACTIVE_POLICY_MIRROR, _ACTIVE_PC_SESSION_TRACKER
    _ACTIVE_POLICY_MIRROR = policy_state_mirror
    _ACTIVE_PC_SESSION_TRACKER = (
        ActivePcSessionTracker(
            session_index=int(getattr(args, "session_index", 0)),
            topic_port=int(args.topic_port),
        )
        if sys.platform.startswith("linux") and metrics is not None
        else None
    )

    replan_output_dir = (
        Path(args.replan_output_dir)
        if args.replan_output_dir
        else Path(args.trajectory).resolve().parent / "replanned"
    )
    replan_output_dir.mkdir(parents=True, exist_ok=True)

    control_dt = 1.0 / max(args.control_hz, 1e-6)
    robot_replay_max_dq = float(DEFAULT_MAX_DQ)
    replay_wall_zero = time.time()
    replay_sim_zero = float(trajectory.sim_t[current_idx])
    replay_step_wall_prev = time.time()
    replay_step_accumulator = 0.0
    state = (
        STATE_REPLAY_PAUSED
        if args.start_paused or policy_state_mirror.enabled
        else STATE_REPLAY_RUNNING
    )
    selected_object_idx = 0
    object_key_latch: Dict[int, bool] = {}
    workflow_key_latch: Dict[int, bool] = {}
    next_object_move_time = 0.0
    robot_lock = RobotOwnershipLock(args.robot_key) if args.mirror_robot else None
    robot_armed = False
    robot_thread: Optional[RobotThread] = None
    replan: Optional[ReplanSession] = None
    replay_sync: Optional[ReplaySyncState] = None
    replay_soft_start_t0: Optional[float] = None
    robot_replay_lag_cycles = 0
    pending = {
        "pause": False,
        "sim_toggle": False,
        "sim_pause": False,
        "sim_resume": False,
        "reset": False,
        "intervene": False,
        "cancel": False,
        "enter_single": False,
        "single_view_ready": False,
        "exit_single": False,
    }
    # Standalone single-session launches (--start_in_single_view) boot directly
    # into single-view state so point clouds / ACTIVE sensors come up immediately
    # without depending on an ENTER_SINGLE command arriving from the headset.
    single_view_active = bool(getattr(args, "start_in_single_view", False))
    _sv_session = int(getattr(args, "start_in_single_view_session", -1))
    _sv_reason = "start_in_single_view"
    if _sv_session >= 0:
        # Per-session selection. Every session gets the same extra args, so this is the
        # only way to bring exactly ONE of them up ACTIVE from the command line.
        if int(getattr(args, "session_index", 0)) == _sv_session:
            single_view_active = True
            _sv_reason = f"start_in_single_view_session={_sv_session}"
        else:
            single_view_active = False
    policy_handoff_sensor_idle = False
    if single_view_active:
        print(f"[Integration] {_sv_reason}: single_view_active=True at startup "
              "(robot still arms only on A/X).")
    elif _sv_session >= 0:
        print(f"[Integration] start_in_single_view_session={_sv_session}: this session "
              f"({int(getattr(args, 'session_index', 0))}) stays IDLE -- the idle-thumbnail "
              "background the selected session is measured against.")

    # One-shot status events pushed to Unity over a lightweight PUB topic
    # (SimPub/Status/intervention). Re-sent for a few consecutive frames so PUB/SUB
    # delivery is reliable even if the subscriber's pipe isn't ready at the exact instant.
    status_pub_pending = {"event": None, "repeats": 0}

    # Authoritative per-session paused flag published on SimPub/Status/paused (topic port).
    # Drives the "PAUSED" badge on the Unity thumbnail panels. Published immediately on change
    # plus a low-rate heartbeat so a just-revealed panel picks up current state within ~250 ms.
    paused_pub_state = {"last_sent": None, "next_pub_t": 0.0}

    def queue_status(event: str, repeats: int = 6):
        status_pub_pending["event"] = event
        status_pub_pending["repeats"] = int(repeats)

    def set_pending(name: str):
        pending[name] = True

    policy_cmd_ref = {"sock": None}
    policy_cmd_forwarding = int(getattr(args, "policy_cmd_port", 0) or 0) > 0
    # Commands forwarded verbatim to app.py in policy mode. app.py is then the ONLY
    # interpreter (the CmdListener `continue`s after forwarding), so every name here must
    # exist in app.py's _POLICY_CMD_ALIASES with the same meaning.
    policy_cmd_actions = {
        "A", "B", "X", "Y",                       # legacy letters (un-rebuilt APKs)
        "INTERVENE", "CANCEL",
        "SIM_TOGGLE", "RESET",                    # semantic replacements for B / Y
        "PAUSE", "RESUME",                        # explicit transport
        # PAUSE_AUTO / RESUME_AUTO retired with the OOD auto-pause supervisor. app.py no
        # longer understands them either, so a stale sender gets an unknown-command log
        # rather than a pause nobody asked for.
        "TASK_SUCCESS", "TASK_FAIL",              # destructive: explicit names only
        # Selection telemetry for the HRI study. NOT sent by Unity -- derived below from
        # ENTER_SINGLE / EXIT_SINGLE, which must keep reaching _CMD_MAP (they drive the
        # sensor ACTIVE/IDLE switch). app.py treats these as bookkeeping only.
        "SELECTED", "DESELECTED",
    }

    # --- HRI study block logging ------------------------------------------------------
    # This process owns the Quest selection signal: ENTER_SINGLE arrives here and is
    # never forwarded to app.py (it drives the sensor ACTIVE/IDLE switch, so it must
    # reach _CMD_MAP). So the VR-side CELL_SELECTED event is written here, and a
    # separate SELECTED/DESELECTED notification is forwarded to the policy purely so its
    # per-frame rows carry the selection state. No control path is affected.
    block_log_ref = {"log": None}

    def _init_block_event_log():
        try:
            import sys as _sys

            candidate = (
                Path(__file__).resolve().parents[3] / "intervene_base"
            )
            if candidate.is_dir() and str(candidate) not in _sys.path:
                _sys.path.insert(0, str(candidate))
            from data_io.experiment_block import (  # type: ignore
                BlockEventLog, block_origin, resolve_block_from_env,
            )

            block = resolve_block_from_env()
            if block is None:
                return
            mono_origin, wall_origin = block_origin(block.block_dir)
            block_log_ref["log"] = BlockEventLog(
                block.events_path, block=block,
                cell_id=int(getattr(args, "session_index", 0)),
                source="vr_runtime", mono_origin=mono_origin, wall_origin=wall_origin,
            )
            print(f"[Block] VR runtime event logging ACTIVE -> {block.events_path}")
        except Exception as exc:
            print(f"[Block][WARN] VR runtime event logging unavailable: {exc}")

    _init_block_event_log()

    def log_block_event(event: str, **fields):
        log = block_log_ref["log"]
        if log is None:
            return
        try:
            log.write(event, **fields)
        except Exception as exc:
            print(f"[Block][WARN] {event} write failed: {exc}")

    def forward_policy_command(command: str) -> bool:
        command = str(command).strip().upper()
        if not policy_cmd_forwarding or command not in policy_cmd_actions:
            return False
        try:
            import zmq as _zmq

            sock = policy_cmd_ref["sock"]
            if sock is None:
                sock = _zmq.Context.instance().socket(_zmq.PUSH)
                sock.setsockopt(_zmq.LINGER, 0)
                sock.setsockopt(_zmq.SNDTIMEO, 25)
                endpoint = f"tcp://{args.policy_cmd_host}:{int(args.policy_cmd_port)}"
                sock.connect(endpoint)
                policy_cmd_ref["sock"] = sock
                print(f"[PolicyCmdForward] PUSH connected {endpoint}")
            sock.send_string(command)
            print(f"[PolicyCmdForward] forwarded {command}")
            return True
        except Exception as exc:
            print(f"[PolicyCmdForward][WARN] failed to forward {command}: {exc}")
            return False

    if mq3 is not None:
        # Control scheme (B/Y swapped so sim start/stop is on the RIGHT controller, same hand
        # as the selector pointer): B = MuJoCo sim start/stop (no mirror), Y = reset,
        # X = intervention, left-grip CANCEL = cancel. (Legacy A=pause kept for the standalone
        # MetaQuest3 path but unused by the multi-session cmd_port forwarder.)
        if policy_cmd_forwarding:
            mq3.register_button_press_event("A", lambda: forward_policy_command("A"))
            mq3.register_button_press_event("B", lambda: forward_policy_command("B"))
            mq3.register_button_press_event("X", lambda: forward_policy_command("INTERVENE"))
            mq3.register_button_press_event("Y", lambda: forward_policy_command("Y"))
        else:
            mq3.register_button_press_event("A", lambda: set_pending("pause"))
            mq3.register_button_press_event("B", lambda: set_pending("sim_toggle"))
            mq3.register_button_press_event("X", lambda: set_pending("intervene"))
            mq3.register_button_press_event("Y", lambda: set_pending("reset"))

    # scene_pub_ref lets the listener thread mutate the publisher reference
    # in response to PROMOTE/DEMOTE without dancing around closure rules.
    scene_pub_ref = {"pub": scene_pub}

    cmd_listener_stop = threading.Event()
    cmd_listener_sock = None
    if args.cmd_port and int(args.cmd_port) > 0:
        try:
            import zmq as _zmq
            _cmd_ctx = _zmq.Context.instance()
            cmd_listener_sock = _cmd_ctx.socket(_zmq.PULL)
            cmd_listener_sock.setsockopt(_zmq.LINGER, 0)
            cmd_listener_sock.bind(f"tcp://{args.bind_ip}:{int(args.cmd_port)}")
            # Semantic names are authoritative; bare controller letters are legacy aliases
            # kept so an un-rebuilt Quest APK keeps working. Letters MUST mean the same
            # thing here as in app.py's _POLICY_CMD_ALIASES — they drifted apart on
            # 2026-07-26 and "B" silently became mark_task_success() on the app.py side.
            _CMD_MAP = {
                "SIM_TOGGLE": "sim_toggle",   # MuJoCo sim start/stop (no robot mirror)
                "B": "sim_toggle",            # legacy alias for SIM_TOGGLE
                "A": "pause",                 # legacy/no longer forwarded by single view
                "PAUSE": "sim_pause",         # explicit/idempotent pause
                "RESUME": "sim_resume",       # explicit/idempotent resume
                "INTERVENE": "intervene",
                "X": "intervene",             # legacy alias
                "RESET": "reset",             # reset replay to frame 0
                "Y": "reset",                 # legacy alias
                "CANCEL": "cancel",           # left grip trigger: cancel intervention/replan
                "ENTER_SINGLE": "enter_single",
                "SINGLE_VIEW_READY": "single_view_ready",
                "EXIT_SINGLE": "exit_single",
            }

            def _handle_promote():
                # Construct MujocoPublisher IN-PROCESS so scene mesh starts
                # flowing to the Quest without restarting. Thumbnail sessions
                # may skip XRNodeManager at startup, so initialize lazily here.
                if scene_pub_ref["pub"] is not None:
                    print("[Promote] already in full mode (MujocoPublisher active), ignoring.")
                    return
                try:
                    print("[Promote] constructing MujocoPublisher in-process...")
                    from simpub.core.node_manager import init_xr_node_manager
                    init_xr_node_manager(args.host_ip or args.host)
                    pub = MujocoPublisher(
                        model,
                        data,
                        host=args.host_ip or args.host,
                        visible_geoms_groups=args.visible_geoms_groups,
                        preferred_xr_name=args.unity_node,
                    )
                    scene_pub_ref["pub"] = pub
                    print("[Promote] MujocoPublisher initialized; scene push pending Quest discovery.")
                except Exception as ex:
                    print(f"[Promote] FAILED to construct MujocoPublisher: {ex}")

            def _handle_demote():
                # Stop ONLY the scene streamer (shutdown_scene_only) so the
                # shared XRNodeManager + sensor publisher keep working. Then
                # drop the reference so the next PROMOTE constructs fresh.
                pub = scene_pub_ref["pub"]
                if pub is None:
                    print("[Demote] already in thumbnail mode, ignoring.")
                    return
                try:
                    pub.shutdown_scene_only()
                    scene_pub_ref["pub"] = None
                    print("[Demote] MujocoPublisher scene streamer stopped; back to thumbnail mode.")
                except Exception as ex:
                    print(f"[Demote] FAILED to shutdown MujocoPublisher: {ex}")

            def _cmd_loop():
                while not cmd_listener_stop.is_set():
                    try:
                        if cmd_listener_sock.poll(timeout=200) == 0:
                            continue
                        msg = cmd_listener_sock.recv_string(flags=_zmq.NOBLOCK)
                    except _zmq.Again:
                        continue
                    except Exception as ex:
                        if not cmd_listener_stop.is_set():
                            print(f"[CmdListener] recv error: {ex}")
                        return
                    # Latency echoes are routed FIRST and never fall through. They
                    # arrive at the publish rate (tens per second per camera), so they
                    # must not reach the per-command print below -- that alone would add
                    # thousands of lines a minute to the unified log -- and they must
                    # never reach set_pending, where 'A' and 'B' are real commands.
                    _echo = parse_echo(msg)
                    if _echo is not None:
                        _probe = globals().get("_LATENCY_PROBE")
                        if _probe is not None:
                            _phase, _seq, _decode_ms = _echo
                            _rtt, _cam = _probe.on_echo_detailed(_phase, _seq, decode_ms=_decode_ms)
                            if _rtt is not None and metrics is not None:
                                metrics.observe(f"e2e_rtt_{_phase.lower()}_ms", _rtt)
                                # Per stream: in RGB mode the pooled number mixes the
                                # 60 Hz camera panels with the 10 Hz `front` thumbnail,
                                # which is decoded twice (grid + diamond) and waits on
                                # the headset's decode gate. Only the per-camera split
                                # says which one the operator was actually looking at.
                                if _cam:
                                    metrics.observe(f"e2e_rtt_{_phase.lower()}_ms.{_cam}", _rtt)
                                if _decode_ms is not None:
                                    metrics.observe(f"e2e_headset_decode_ms.{_cam or 'unknown'}", _decode_ms)
                        continue

                    key_upper = msg.strip().upper()
                    # Always log RAW receive so we can verify what arrived,
                    # independent of the handler's own logging.
                    print(f"[CmdListener] RX cmd='{key_upper}'")
                    if key_upper == "PROMOTE":
                        _handle_promote()
                        continue
                    if key_upper == "DEMOTE":
                        _handle_demote()
                        continue
                    if key_upper in policy_cmd_actions and policy_cmd_forwarding:
                        forward_policy_command(key_upper)
                        continue
                    key = _CMD_MAP.get(key_upper)
                    if key is None:
                        print(f"[CmdListener] unknown cmd: '{msg}' (expected A/B/X/Y/INTERVENE/CANCEL/ENTER_SINGLE/SINGLE_VIEW_READY/EXIT_SINGLE/PROMOTE/DEMOTE)")
                        continue
                    set_pending(key)
                    print(f"[CmdListener] {key_upper} -> set_pending({key})")

                    # Study telemetry, AFTER the real handling above so a logging
                    # problem can never delay or swallow a selection command.
                    if key_upper in ("ENTER_SINGLE", "EXIT_SINGLE"):
                        selected = key_upper == "ENTER_SINGLE"
                        if selected:
                            log_block_event(
                                "CELL_SELECTED",
                                selected_cell_id=int(getattr(args, "session_index", 0)),
                                selection_method="quest_ray",
                            )
                        forward_policy_command("SELECTED" if selected else "DESELECTED")

            threading.Thread(target=_cmd_loop, name="CmdListener", daemon=True).start()
            print(f"[CmdListener] PULL bound tcp://{args.bind_ip}:{int(args.cmd_port)} "
                  "(A/B/X/Y + ENTER_SINGLE/SINGLE_VIEW_READY/EXIT_SINGLE -> runtime controls; PROMOTE/DEMOTE -> in-process MujocoPublisher on/off)")
        except Exception as ex:
            cmd_listener_sock = None
            print(f"[CmdListener] FAILED to bind cmd_port {args.cmd_port}: {ex}")

    risk_pub_thread: Optional[RiskPublisherThread] = None
    if getattr(args, "risk_dummy", True):
        risk_pub_thread = RiskPublisherThread(
            sensor_node,
            policy_mirror=policy_state_mirror,
            trajectory=trajectory,
            get_frame_idx=lambda: current_idx,
            scale=getattr(args, "acc_risk_scale", 3.0),
            session_index=getattr(args, "session_index", 0),
            metrics=metrics,
        )
        risk_pub_thread.start()
        _risk_src = "policy-ACC" if (policy_state_mirror is not None and policy_state_mirror.enabled) \
            else ("traj-jerk" if trajectory is not None else "sinusoid-dummy")
        print(f"[RiskPub] ACC risk publisher started (session={getattr(args, 'session_index', 0)}, "
              f"source={_risk_src}, scale={getattr(args, 'acc_risk_scale', 3.0)}, "
              f"topic={RiskPublisherThread.TOPIC})")

    def apply_current_frame(*, restore_scene: bool = False):
        nonlocal current_idx
        current_idx = trajectory.apply_frame(
            current_idx,
            data,
            replay_bindings=replay_bindings,
            scene_seed=scene_seed,
            restore_scene=restore_scene,
        )

    def has_robot_control() -> bool:
        return bool(args.mirror_robot and robot_armed and robot_thread is not None)

    def selected_sensor_active_arg() -> Optional[bool]:
        if policy_handoff_sensor_idle:
            return False
        if bool(getattr(args, "selected_sensor_activation", False)):
            return bool(single_view_active)
        return None

    def point_cloud_publish_allowed() -> bool:
        if policy_handoff_sensor_idle:
            return False
        # Mirrors the activation decision in publish_sensor_frames(). Selection
        # (ENTER_SINGLE/EXIT_SINGLE) is authoritative; the peer count is only a debounced
        # last-resort fallback for a dropped ENTER_SINGLE. Reads the SAME debounce state
        # the publish path maintains, so the two can never disagree — an undebounced copy
        # here would let point clouds keep publishing on a session the rate limiter had
        # already demoted.
        selected_active = selected_sensor_active_arg()
        threshold = int(getattr(args, "active_peer_threshold", 2))
        try:
            peers = int(sensor_node.peer_count()) if hasattr(sensor_node, "peer_count") else threshold
        except Exception:
            peers = threshold
        # advance_state=False: this is a read-only check that can run several times per
        # tick, and it must not restart the debounce timer the publish path owns.
        is_active, _ = resolve_sensor_activation(
            sensor_state=sensor_state,
            selected=selected_active,
            peer_count=peers,
            threshold=threshold,
            grace_s=float(getattr(args, "peer_fallback_grace_s", 0.0) or 0.0),
            selected_peer_idle_grace_s=float(
                getattr(args, "selected_peer_idle_grace_s", 5.0) or 0.0
            ),
            now=time.time(),
            force_idle=False,
            advance_state=False,
        )
        return bool(args.pc) and (is_active or not bool(getattr(args, "pc_active_only", False)))

    def drop_point_cloud_backlog(reason: str, *, keep_output: bool = False) -> None:
        # keep_output=True: drop only stale INPUT jobs, keep already-built frames on the worker
        # output queue and the sensor publisher's pending PC. Used by warm-up refreshes so a fresh
        # PC built moments earlier (e.g. between ENTER_SINGLE and SINGLE_VIEW_READY) is not thrown
        # away before it can be drained/published. EXIT_SINGLE uses keep_output=False (full clear).
        dropped_worker = 0
        dropped_sensor = 0
        if pc_worker is not None:
            try:
                dropped_worker = int(
                    pc_worker.clear_input() if keep_output else pc_worker.clear_pending()
                )
            except Exception:
                dropped_worker = 0
        if not keep_output:
            try:
                dropped_sensor = int(sensor_node.drop_pending_pc())
            except Exception:
                dropped_sensor = 0
        if dropped_worker or dropped_sensor:
            print(
                f"[AdaptiveRate] {reason}: dropped stale point-cloud backlog "
                f"(worker={dropped_worker}, sensor={dropped_sensor}, keep_output={keep_output})."
            )

    def rebind_renderer_model(new_model) -> None:
        """Point the offscreen renderer at a different compiled model.

        Done from HERE rather than by adding a method to `GLFWMjrRenderer` for two reasons:
        this project's rule is that `franka_quest_unified_publisher.py` is not modified, and
        that class's `close()` calls `glfw.terminate()` -- which would destroy the very
        window we are rebinding on. So we rebuild only the model-bound pieces in place and
        keep the window, camera, options and pixel buffers.
        """
        glfw.make_context_current(renderer._win)
        # The publisher applies these to the model it was constructed with; a fresh model
        # needs them before its MjrContext is built or the offscreen buffer is too small.
        new_model.vis.global_.offwidth = renderer.width
        new_model.vis.global_.offheight = renderer.height
        new_model.vis.quality.offsamples = 0

        new_scene = mujoco.MjvScene(new_model, maxgeom=10000)
        new_con = mujoco.MjrContext(new_model, mujoco.mjtFontScale.mjFONTSCALE_150)
        old_con = renderer.con
        renderer.model = new_model
        renderer.scene = new_scene
        renderer.con = new_con
        try:
            old_con.free()
        except Exception:
            pass
        try:
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, renderer.con)
        except Exception:
            pass
        # A fresh MjvScene resets the render flags to their defaults. Miss this and every VR
        # session silently gets shadows/reflections back, which is a 25-50% render-cost
        # regression that would look like "the variant is slow".
        apply_render_flags(renderer, args)

    def rebuild_for_variant(descriptor, key):
        """Adopt a different compiled model at an episode boundary. Returns (model, data).

        Only the model-BOUND state is rebuilt. Everything resolved by NAME -- camera sets,
        topic list, replay bindings, movable objects, the anchor site/body -- stays valid
        because a variant is required to have identical body/joint/geom/camera names, and
        that is asserted below rather than assumed.

        The SensorNode is deliberately NOT rebuilt: it owns the ZMQ ports the headset is
        subscribed to, and rebinding them would drop Unity's subscriptions.
        """
        nonlocal model, data, table_plane
        # INVARIANT: every exit path must leave the mirror able to make progress. Either
        # advance `active_variant_key` (declaring "this is as close as we get") or rely on
        # apply_to writing state unconditionally. Never a stale key AND withheld state --
        # that combination freezes the session forever while it keeps publishing.
        if variant_cache is None:
            if policy_state_mirror.active_variant_key != key:
                print(f"[Variant][WARN] no variant cache in this runtime; adopting key {key} "
                      "and rendering the base geometry for this episode.")
                policy_state_mirror.active_variant_key = key
            return model, data
        t0 = time.perf_counter()
        # NON-BLOCKING. A compile is ~1.3 s and this runs on the main loop, so blocking here
        # stalls the mirror, the sensor publish and the point clouds all at once (measured:
        # rebuild=1372 ms with "the mirrored sim is FROZEN" on either side of it). Instead
        # queue the build and keep running on the current model; apply_to writes state
        # regardless, so nothing freezes, and a later tick swaps once the model is ready.
        new_model = variant_cache.get(descriptor)
        if new_model is None:
            reason = variant_cache.failure_reason(descriptor)
            if reason:
                print(f"[Variant][ERROR] cannot build variant {key} ({reason}); continuing "
                      "on the current model. The headset will show the previous episode's "
                      "cup geometry until the next episode.")
                policy_state_mirror.active_variant_key = key   # cached failure: stop retrying
                return model, data
            variant_cache.prefetch(descriptor)
            now = time.time()
            if _pending_variant.get("key") != key:
                _pending_variant.update(key=key, since=now, warned=False)
                print(f"[Variant] {key} is not compiled yet; building in the background and "
                      f"CONTINUING on {policy_state_mirror.active_variant_key}. The headset "
                      "keeps streaming; cup geometry is one episode behind for ~1.5 s.")
            elif now - _pending_variant.get("since", now) > 10.0 and not _pending_variant.get("warned"):
                _pending_variant["warned"] = True
                print(f"[Variant][WARN] variant {key} has been pending for "
                      f"{now - _pending_variant['since']:.0f} s; the background build is not "
                      "completing. Still streaming on the previous model.")
            # active_variant_key deliberately NOT advanced: a later tick retries the get().
            return model, data

        before = describe_reference(model)
        after = describe_reference(new_model)
        diff = fingerprint_diff(before, after)
        if diff is not None:
            field, expected, got = diff
            print(f"[Variant][ERROR] variant {key} is not a drop-in replacement "
                  f"({field}); refusing to swap.")
            policy_state_mirror.active_variant_key = key
            return model, data

        # Nothing must be published from the old model after this point.
        drop_point_cloud_backlog("MODEL_VARIANT_SWAP", keep_output=False)

        new_data = mujoco.MjData(new_model)
        new_data.qpos[:] = data.qpos
        new_data.qvel[:] = data.qvel
        new_data.ctrl[:] = data.ctrl
        new_data.time = data.time
        mujoco.mj_forward(new_model, new_data)

        rebind_renderer_model(new_model)
        # Keyed on (cam, w, h, fovy, mode) and fovy is enforced identical, so this is
        # belt-and-braces -- but it is one line and clearing it is never wrong.
        _CAM_INTRINSICS_CACHE.clear()

        model, data = new_model, new_data
        setup["model"] = new_model
        setup["data"] = new_data
        # Re-resolve rather than trust: resolved from the `table_top` geom, which is
        # variant-invariant, but cheap enough that assuming would be the worse trade.
        table_plane = None
        if args.pc_max_depth is None:
            setup["pc_max_depth"] = float(new_model.vis.map.zfar) * 0.95

        # These are integer ids cached at startup. Names are identical across variants, so
        # they must still resolve to the same ids -- assert it, because a silently shifted
        # anchor id would mis-place every point cloud in the headset.
        for label, obj_type, name in (
            ("anchor_site", mujoco.mjtObj.mjOBJ_SITE, args.pc_anchor_site),
            ("anchor_body", mujoco.mjtObj.mjOBJ_BODY, args.pc_anchor_body),
        ):
            if not name:
                continue
            old_id = int(setup.get(f"{label}_id", -1))
            if old_id < 0:
                # -1 is a DELIBERATE "not resolved": resolve_runtime_setup only resolves
                # these under `args.pc and args.pc_anchor_auto_translate`, and
                # --pc_anchor_body has a truthy default ("link0"). Re-resolving here would
                # silently promote an unresolved anchor to a real id on the first swap.
                continue
            new_id = mujoco.mj_name2id(new_model, obj_type, name)
            if old_id != new_id:
                print(f"[Variant][WARN] {label} id moved {old_id} -> {new_id} for {name!r}; "
                      "updating.")
                setup[f"{label}_id"] = new_id

        previous_key = policy_state_mirror.active_variant_key
        policy_state_mirror.active_variant_key = key
        # Pin the model we are about to render from, so a background prefetch of the NEXT
        # episode's variant cannot evict it out from under us, and release the old one.
        try:
            variant_cache.retain(key)
            if previous_key != BASE_VARIANT_KEY and previous_key != key:
                variant_cache.release(previous_key)
        except Exception as exc:
            print(f"[Variant][WARN] cache retain/release failed: {exc}")
        _pending_variant.update(key=None, since=0.0, warned=False)
        # Skip this iteration's sensor publish so no frame straddles the swap. Must be
        # time.time(), never 0.0 -- see refresh_point_cloud_visit for why.
        sensor_state.next_sensor_t = time.time()
        sensor_state.first_pc_logged.clear()
        print(f"[Variant] model -> {key} (epoch={policy_state_mirror.latest_model_epoch}, "
              f"rebuild={(time.perf_counter() - t0) * 1e3:.0f} ms)")
        return model, data

    def release_gpu_point_cloud_cache(reason: str) -> None:
        """Hand Open3D's cached CUDA blocks back to the driver.

        Open3D allocates through a CACHING memory manager: blocks freed by `build_frame` are
        retained by the process, so a session that was selected once keeps pinning VRAM for
        the rest of the run even though only ONE session is ever active at a time. With nine
        runtimes plus nine policy processes on one 16 GB card that is what tipped it over --
        702 `CUDA runtime error: out of memory` in a single run, six of nine sessions
        producing zero points.

        Only safe once no build is in flight, which is why this runs after the backlog drop.
        """
        if not _pc_release_cache_available[0]:
            return
        try:
            import open3d as _o3d
            _o3d.core.cuda.release_cache()
            print(f"[PC][Lifecycle] {reason}: released Open3D CUDA cache.")
        except Exception as exc:
            print(f"[PC][Lifecycle][WARN] release_cache failed ({exc}); disabling.")
            _pc_release_cache_available[0] = False

    def reconcile_pc_gpu_memory() -> None:
        """Release GPU memory a short while after this session stops publishing clouds.

        Read-only w.r.t. the activation state machine: `point_cloud_publish_allowed()` calls
        `resolve_sensor_activation(advance_state=False)` precisely so a reader cannot disturb
        the debounce timer it is reading. Hysteresis is on the falling edge only, so a brief
        `selected_no_peers` flap does not churn the allocator.
        """
        if not _pc_release_cache_available[0]:
            return
        active = point_cloud_publish_allowed()
        if active:
            _pc_idle_since[0] = None
            _pc_cache_released[0] = False
            return
        if _pc_cache_released[0]:
            return
        now = time.time()
        if _pc_idle_since[0] is None:
            _pc_idle_since[0] = now
            return
        if now - _pc_idle_since[0] < float(getattr(args, "pc_idle_release_s", 2.0)):
            return
        drop_point_cloud_backlog("PC_IDLE_RELEASE", keep_output=False)
        release_gpu_point_cloud_cache("session idle")
        _pc_cache_released[0] = True

    def refresh_point_cloud_visit(reason: str, *, keep_output: bool) -> None:
        # ENTER_SINGLE is an identity boundary and clears every pending layer. The later
        # SINGLE_VIEW_READY warm-up stays within the confirmed visit and may preserve a
        # freshly completed frame.
        drop_point_cloud_backlog(reason, keep_output=keep_output)
        sensor_state.first_pc_logged.clear()
        # Publish immediately on the next tick WITHOUT poisoning the cadence catch-up loop.
        # next_sensor_t must stay near time.time(); setting it to 0.0 made the catch-up loop in
        # publish_sensor_frames (`while next_sensor_t <= now: next_sensor_t += 1/fps`) try to count
        # from 0 up to the current epoch time in ~1/30 s steps (~50e9 iterations) → permanent
        # main-loop freeze on every ENTER_SINGLE. Use now so the next publish still fires
        # (now >= next_sensor_t) but the loop terminates in 0-1 iterations.
        sensor_state.next_sensor_t = time.time()
        if reason == "ENTER_SINGLE":
            sensor_state.pc_visit_started_t = sensor_state.next_sensor_t
            sensor_state.pc_visit_publish_counts = {
                cam_name: 0 for cam_name in setup["pc_cam_set"]
            }
        sensor_state.last_adaptive_mode = None
        print(
            f"[AdaptiveRate] {reason}: point-cloud visit refreshed; "
            "first publish diagnostics reset."
        )

    def finish_point_cloud_visit(reason: str) -> None:
        started_t = sensor_state.pc_visit_started_t
        if started_t is None:
            return
        duration_s = max(0.0, time.time() - started_t)
        counts = sensor_state.pc_visit_publish_counts
        rates = {
            cam_name: (float(count) / duration_s if duration_s > 0.0 else 0.0)
            for cam_name, count in counts.items()
        }
        print(
            "[PointCloudVisit] "
            f"{reason} duration_s={duration_s:.3f} "
            f"counts={json.dumps(counts, separators=(',', ':'))} "
            f"rates_hz={json.dumps(rates, separators=(',', ':'))}"
            # A visit that reports 0.0 Hz on every camera should say WHY in the same line,
            # rather than leaving the operator to correlate it with the error stream.
            + (f" build_errors={pc_worker._consec_errors} last_error={pc_worker.last_error!r}"
               if pc_worker is not None and getattr(pc_worker, "_consec_errors", 0) else "")
        )
        sensor_state.pc_visit_started_t = None
        sensor_state.pc_visit_publish_counts = {}

    def reset_replay_timing() -> None:
        nonlocal replay_wall_zero, replay_sim_zero, replay_step_wall_prev, replay_step_accumulator, replay_soft_start_t0, robot_replay_lag_cycles
        replay_wall_zero = time.time()
        replay_sim_zero = float(trajectory.sim_t[current_idx])
        replay_step_wall_prev = replay_wall_zero
        replay_step_accumulator = 0.0
        replay_soft_start_t0 = None
        robot_replay_lag_cycles = 0

    def ensure_robot_control(reason: str, *, acquire_lock: bool) -> bool:
        nonlocal robot_armed, robot_thread
        if not args.mirror_robot:
            return False
        if args.mirror_on_select_only and not robot_armed and not acquire_lock:
            print(
                f"[RobotOwnership][WARN] {reason}: robot is not armed for this session. "
                "Select the session first so ENTER_SINGLE can acquire ownership."
            )
            return False
        if not robot_armed:
            if robot_lock is not None:
                acquired, message = robot_lock.acquire()
                if not acquired:
                    print(f"[RobotOwnership][WARN] {reason}: could not acquire robot '{args.robot_key}' ({message}).")
                    return False
                print(f"[RobotOwnership] {reason}: acquired robot '{args.robot_key}' ({message}).")
            robot_armed = True
        if robot_thread is None:
            try:
                robot_thread = RobotThread(args.robot_key)
                print(f"[RobotOwnership] {reason}: robot thread armed for '{args.robot_key}'.")
            except Exception as exc:
                print(f"[RobotOwnership][WARN] {reason}: failed to arm robot thread: {exc}")
                robot_armed = False
                if robot_lock is not None:
                    robot_lock.release()
                return False
        return True

    def release_robot_control(reason: str, *, resume_sim: bool) -> None:
        nonlocal robot_armed, robot_thread, replay_sync, replay_soft_start_t0, state
        if robot_thread is not None:
            try:
                robot_thread.set_reading(False)
            except Exception:
                pass
            try:
                robot_thread.close()
            except Exception as exc:
                print(f"[RobotOwnership][WARN] {reason}: robot thread close failed: {exc}")
            robot_thread = None
        if robot_armed:
            print(f"[RobotOwnership] {reason}: released robot '{args.robot_key}'.")
        robot_armed = False
        if robot_lock is not None:
            robot_lock.release()
        replay_sync = None
        replay_soft_start_t0 = None
        if resume_sim:
            reset_replay_timing()
            state = STATE_REPLAY_RUNNING
            print(f"[Integration] {reason}: resumed sim-only grid replay from frame {current_idx}.")

    def enter_single_view() -> None:
        nonlocal single_view_active, policy_handoff_sensor_idle
        policy_handoff_sensor_idle = False
        if single_view_active:
            print(
                f"[Integration] ENTER_SINGLE duplicate ignored at frame {current_idx}; "
                "single-view stream remains active."
            )
            return
        single_view_active = True

        refresh_point_cloud_visit("ENTER_SINGLE", keep_output=False)
        print(
            f"[Integration] ENTER_SINGLE: single-view active at frame {current_idx}; "
            f"state={state}; replay continues sim-only until A/X requests robot control."
        )

    def single_view_ready() -> None:
        if not single_view_active:
            print("[Integration][WARN] SINGLE_VIEW_READY received before ENTER_SINGLE; ignoring PC refresh until session is active.")
            return
        refresh_point_cloud_visit("SINGLE_VIEW_READY", keep_output=True)
        print("[Integration] SINGLE_VIEW_READY: point-cloud warm-up requested.")

    def reset_replay():
        nonlocal current_idx, replay_wall_zero, replay_sim_zero, replay_step_wall_prev, replay_step_accumulator, state, replan, replay_sync, replay_soft_start_t0
        current_idx = int(np.clip(args.start_idx, 0, len(trajectory) - 1))
        apply_current_frame(restore_scene=True)
        reset_replay_timing()
        state = STATE_REPLAY_PAUSED
        replan = None
        replay_sync = None
        replay_soft_start_t0 = None
        if robot_thread is not None:
            robot_thread.set_reading(False)
        print(f"[Integration] Reset replay to frame {current_idx}")

    def pause_replay():
        nonlocal state, replay_sync, replay_soft_start_t0
        if state in (STATE_REPLAY_RUNNING, STATE_REPLAY_SYNC):
            if state == STATE_REPLAY_RUNNING:
                apply_current_frame()
            state = STATE_REPLAY_PAUSED
            replay_sync = None
            replay_soft_start_t0 = None
            if robot_thread is not None:
                robot_thread.set_reading(False)
            print(f"[Integration] Paused replay at frame {current_idx}")

    def sim_resume():
        # Resume MuJoCo replay WITHOUT engaging the real robot (it stays in its default pose).
        # This is the "play" half of the Y sim start/stop button; mirroring only happens during
        # an X intervention, never via Y.
        nonlocal state, replay_sync, replay_soft_start_t0
        if state == STATE_REPLAY_PAUSED:
            reset_replay_timing()
            replay_sync = None
            replay_soft_start_t0 = None
            if robot_thread is not None:
                robot_thread.set_reading(False)
            state = STATE_REPLAY_RUNNING
            print(f"[Integration] Sim resumed (no mirror) from frame {current_idx}")

    def sim_toggle():
        # Y button: toggle MuJoCo sim play/pause. No robot mirroring on this control.
        if state in (STATE_REPLAY_RUNNING, STATE_REPLAY_SYNC):
            pause_replay()
        elif state == STATE_REPLAY_PAUSED:
            sim_resume()
        else:
            print(f"[Integration] Y/sim_toggle ignored in state={state}.")

    def begin_robot_sync(
        reason: str,
        grip_target_override: Optional[float] = None,
        *,
        after_align: str = "resume",
    ):
        nonlocal state, replay_sync, replay_soft_start_t0, replay_wall_zero, replay_sim_zero, replay_step_wall_prev, replay_step_accumulator
        if not args.mirror_robot:
            reset_replay_timing()
            state = STATE_REPLAY_RUNNING
            print(f"[Integration] Resumed replay from frame {current_idx}")
            return
        if args.mirror_on_select_only and not single_view_active:
            state = STATE_REPLAY_PAUSED
            replay_sync = None
            replay_soft_start_t0 = None
            print(
                f"[Integration][WARN] {reason}: robot sync ignored outside single view; "
                f"replay paused at frame {current_idx}."
            )
            return
        if not ensure_robot_control(reason, acquire_lock=True):
            state = STATE_REPLAY_PAUSED
            replay_sync = None
            replay_soft_start_t0 = None
            print(f"[Integration][WARN] {reason}: robot unavailable; replay remains paused at frame {current_idx}.")
            return
        apply_current_frame()
        replay_sync = ReplaySyncState(
            target_q=np.asarray(data.ctrl[:7], dtype=np.float64).copy(),
            grip_target=(
                float(np.clip(grip_target_override, 0.0, 0.08))
                if grip_target_override is not None
                else trajectory.grip_width_at(current_idx, replay_bindings=replay_bindings)
            ),
            start_t=time.time(),
            after_align=after_align,
        )
        replay_soft_start_t0 = None
        replay_step_wall_prev = time.time()
        replay_step_accumulator = 0.0
        robot_thread.set_reading(True)
        state = STATE_REPLAY_SYNC
        print(
            f"[Integration] Robot sync started ({reason}). "
            f"Moving slowly to frame {current_idx} pose before replay resumes."
        )

    def resume_replay():
        if state == STATE_REPLAY_PAUSED:
            begin_robot_sync("resume")

    def mirror_align_resume() -> None:
        nonlocal state, replay_sync, replay_soft_start_t0
        if state == STATE_REPLAY_SYNC:
            print(
                "[InputGuard] A ignored during robot sync; "
                "wait for alignment, or press Y/B/back to stop safely."
            )
            return
        if state in (STATE_REPLAN_HOLD, STATE_REPLAN_TRANSITION, STATE_REPLAN_RECORD):
            print(
                f"[InputGuard] A ignored during {state}; "
                "X finishes recording, Y/back cancels safely."
            )
            return
        if state == STATE_REPLAY_RUNNING:
            apply_current_frame()
            state = STATE_REPLAY_PAUSED
            replay_sync = None
            replay_soft_start_t0 = None
            if robot_thread is not None:
                robot_thread.set_reading(False)
            print(f"[Integration] A/mirror: froze replay at frame {current_idx} for robot alignment.")
        if state == STATE_REPLAY_PAUSED:
            begin_robot_sync("mirror-resume")

    def prepare_replan_session() -> Optional[float]:
        nonlocal replan
        apply_current_frame()
        q_seed = np.asarray(data.ctrl[:7], dtype=np.float64).copy()
        grip_seed = trajectory.grip_width_at(current_idx, replay_bindings=replay_bindings)
        if grip_seed is not None:
            finger_seed = replay_bindings.finger_qpos_from_grip_width(grip_seed)
            grip_target = float(np.clip(grip_seed, 0.0, 0.08))
        else:
            finger_seed = replay_bindings.finger_qpos_from_grip_width(
                replay_bindings.current_grip_width(data)
            )
            grip_target = float(np.clip(finger_seed * 2.0, 0.0, 0.08))
        replan = ReplanSession(
            cut_idx=current_idx,
            q_seed=q_seed,
            finger_seed=finger_seed,
            qpos_seed=np.asarray(data.qpos, dtype=np.float64).copy(),
            qvel_seed=np.asarray(data.qvel, dtype=np.float64).copy(),
            start_snapshot=ReplanStateSnapshot.capture(data),
        )
        return grip_target

    def start_intervention_auto_align() -> None:
        nonlocal state, replay_sync, replay_soft_start_t0
        if state == STATE_REPLAN_RECORD:
            finish_replan_recording()
            return
        if state == STATE_REPLAN_TRANSITION:
            print("[Integration] HUMAN_CONTROL transition already in progress; wait before pressing X again.")
            return
        if state == STATE_REPLAY_SYNC:
            print("[Integration] Robot sync already in progress; wait for alignment or press A/Y to pause/cancel.")
            return
        if state == STATE_REPLAY_RUNNING:
            apply_current_frame()
            state = STATE_REPLAY_PAUSED
            replay_sync = None
            replay_soft_start_t0 = None
            if robot_thread is not None:
                robot_thread.set_reading(False)
            print(f"[Integration] INTERVENE: froze replay at frame {current_idx} for robot alignment.")
        if state == STATE_REPLAN_HOLD:
            print("[Integration] Legacy replan hold detected; starting HUMAN_CONTROL recording.")
            start_replan_recording()
            return
        if state != STATE_REPLAY_PAUSED:
            print(f"[Integration][WARN] Cannot start intervention from state={state}.")
            return
        if not args.mirror_robot:
            print("[Integration] Intervention requires --mirror_robot.")
            return
        if not ensure_robot_control("INTERVENE", acquire_lock=True):
            print("[Integration][WARN] INTERVENE ignored because this session does not own the robot.")
            return
        grip_target = prepare_replan_session()
        print("[Integration] INTERVENE: auto-aligning robot to current MuJoCo frame, then entering HUMAN_CONTROL.")
        begin_robot_sync("intervention", grip_target_override=grip_target, after_align="intervene")

    def exit_single_view() -> None:
        nonlocal state, replay_sync, replay_soft_start_t0, single_view_active
        single_view_active = False
        finish_point_cloud_visit("EXIT_SINGLE")
        drop_point_cloud_backlog("EXIT_SINGLE")
        if state in (STATE_REPLAN_HOLD, STATE_REPLAN_TRANSITION, STATE_REPLAN_RECORD) and replan is not None:
            cancel_replan()
        elif state == STATE_REPLAY_SYNC:
            if robot_thread is not None:
                robot_thread.set_reading(False)
            replay_sync = None
            replay_soft_start_t0 = None
            state = STATE_REPLAY_PAUSED
        # A mirrored policy frame index belongs to the live policy episode, not to the
        # fallback trajectory loaded by this runtime. Resuming local replay here used
        # that foreign index in reset_replay_timing(), killing the runtime on every
        # EXIT_SINGLE once the policy advanced beyond the fallback trajectory length.
        resume_after_release = (
            not policy_state_mirror.enabled
            and (has_robot_control() or state != STATE_REPLAY_RUNNING)
        )
        release_robot_control("EXIT_SINGLE", resume_sim=resume_after_release)


    def ensure_single_view_for_control(command_name: str) -> bool:
        if bool(getattr(args, "selected_sensor_activation", False)) and not single_view_active:
            print(
                f"[InputGuard][WARN] {command_name} received while session is not marked "
                "single-view active; ignoring robot-control command."
            )
            return False
        return True


    def enter_replan_hold():
        nonlocal state, replan, robot_thread
        if args.mirror_robot:
            ensure_robot_control("legacy_replan_hold", acquire_lock=not args.mirror_on_select_only)
        apply_current_frame()
        q_seed = np.asarray(data.ctrl[:7], dtype=np.float64).copy()
        grip_seed = trajectory.grip_width_at(current_idx, replay_bindings=replay_bindings)
        if grip_seed is not None:
            finger_seed = replay_bindings.finger_qpos_from_grip_width(grip_seed)
        else:
            finger_seed = replay_bindings.finger_qpos_from_grip_width(
                replay_bindings.current_grip_width(data)
            )
        replan = ReplanSession(
            cut_idx=current_idx,
            q_seed=q_seed,
            finger_seed=finger_seed,
            qpos_seed=np.asarray(data.qpos, dtype=np.float64).copy(),
            qvel_seed=np.asarray(data.qvel, dtype=np.float64).copy(),
            start_snapshot=ReplanStateSnapshot.capture(data),
        )
        if robot_thread is not None:
            robot_thread.hold_pose(
                replan.q_seed,
                replan.finger_seed,
                dt=control_dt,
                max_dq=REPLAN_HOLD_MAX_DQ,
            )
        state = STATE_REPLAN_HOLD
        print("[Integration] Replan hold active. Press X to confirm hold, then X again to start recording. Y to cancel.")
        if robot_thread is None:
            print("[Integration][WARN] No robot connected (--mirror_robot not set). "
                  "Replan recording will not be possible. Press Y to cancel.")

    def start_replan_recording():
        nonlocal state, replan
        if robot_thread is None:
            print("[Integration] Replan recording requires --mirror_robot. Press Y to cancel.")
            queue_status("intervention_started")  # fire HUD even in sim-only mode
            return
        if replan is None:
            return
        replan.suffix_path = replan_output_dir / f"{trajectory.path.stem}_suffix_{int(time.time())}.npz"
        replan.sim_time_zero = float(data.time)
        replan.recorder = TrajectoryRecorder(
            save_path=replan.suffix_path,
            log_hz=float(args.control_hz),
            view_hz=float(args.control_hz),
            sim_t_zero=replan.sim_time_zero,
        )
        started, msg = robot_thread.start_human_guidance_async()
        if not started:
            replan.recorder = None
            replan.suffix_path = None
            print(f"[Integration][WARN] Could not start guidance transition: {msg}")
            return
        state = STATE_REPLAN_TRANSITION
        print("[Integration] Switching to HUMAN_CONTROL... X/Y disabled until transition completes.")

    def cancel_replan():
        nonlocal current_idx, state, replan, robot_thread, replay_sync, replay_soft_start_t0
        if replan is None:
            return
        live_snapshot = ReplanStateSnapshot.capture(data)
        current_idx = int(replan.cut_idx)
        replan = None
        live_snapshot.restore(data)
        state = STATE_REPLAY_PAUSED
        replay_sync = None
        replay_soft_start_t0 = None
        if robot_thread is not None:
            robot_thread.set_reading(False)
            if args.mirror_robot:
                robot_thread.ensure_mode(robot_thread.control_type_enum.HYBRID_JOINT_IMPEDANCE_CONTROL)
            else:
                robot_thread.close()
                robot_thread = None
        # Intervention ended (cancelled / exited) — clear the Quest HUD + MC-mode gate.
        queue_status("intervention_finished")
        print(f"[Integration] Replan cancelled at frame {current_idx}; live MuJoCo object state preserved.")

    def finish_replan_recording():
        nonlocal trajectory, current_idx, state, replan, replay_wall_zero, replay_sim_zero, replay_step_wall_prev, replay_step_accumulator, robot_thread, replay_sync, replay_soft_start_t0, single_view_active, policy_handoff_sensor_idle
        if replan is None or replan.recorder is None or replan.suffix_path is None:
            return
        if len(replan.recorder.q_real) == 0:
            print("[Integration][WARN] No suffix samples were recorded.")
            cancel_replan()
            return

        # Intervention is ending (X pressed to finish + stitch) — fire the Quest "Intervention
        # over" HUD and clear the MC-mode gate (InterventionStatusHud.InterventionActive).
        queue_status("intervention_finished")

        final_snapshot = ReplanStateSnapshot.capture(data)
        replan.final_snapshot = final_snapshot
        cut_idx = int(replan.cut_idx)
        paused_grip_target = float(np.clip(replan.finger_seed * 2.0, 0.0, 0.08))
        suffix_path = replan.recorder.save()
        stitched_path = replan_output_dir / (
            f"{trajectory.path.stem}_replanned_{int(time.time())}.npz"
        )
        stitch_npz(
            original_path=trajectory.path,
            suffix_path=suffix_path,
            cut_idx=cut_idx,
            out_path=stitched_path,
            drop_suffix_first_frame=True,
            suffix_time_scale=REPLAN_SUFFIX_TIME_SCALE,
        )

        trajectory = ReplayTrajectory(stitched_path)
        if robot_thread is not None:
            robot_thread.set_reading(False)
            if args.mirror_robot:
                robot_thread.ensure_mode(robot_thread.control_type_enum.HYBRID_JOINT_IMPEDANCE_CONTROL)
            else:
                robot_thread.close()
                robot_thread = None

        replay_sync = None
        replay_soft_start_t0 = None

        if args.post_replan_mode == "policy_handoff":
            current_idx = len(trajectory) - 1
            final_snapshot.restore(data)
            replan = None
            reset_replay_timing()
            state = STATE_REPLAY_PAUSED
            queue_status("policy_handoff_ready")
            if bool(getattr(args, "policy_handoff_idle_after_ready", False)):
                single_view_active = False
                policy_handoff_sensor_idle = True
                drop_point_cloud_backlog("POLICY_HANDOFF_READY")
                release_robot_control("POLICY_HANDOFF_READY", resume_sim=False)
            print(
                "[Integration] POLICY_HANDOFF_READY "
                f"stitched_path={stitched_path} current_idx={current_idx} "
                f"final_state=({final_snapshot.summary()})"
            )
            if bool(getattr(args, "policy_handoff_idle_after_ready", False)):
                print(
                    "[Integration] POLICY_HANDOFF_READY: idled point-cloud publishing "
                    "and released robot ownership; close the policy launcher to finish."
                )
            return

        current_idx = int(np.clip(cut_idx + 1, 0, len(trajectory) - 1))
        preview_snapshot = replan.start_snapshot or final_snapshot
        preview_snapshot.restore(data)
        trajectory.apply_frame(
            current_idx,
            data,
            replay_bindings=replay_bindings,
            scene_seed=scene_seed,
            restore_scene=False,
        )
        preview_snapshot.restore_auxiliary(data)
        replan = None

        print(
            f"[Integration] Reloaded stitched trajectory: {stitched_path} "
            f"(cut_idx={cut_idx}, resume_idx={current_idx}, time_scale={REPLAN_SUFFIX_TIME_SCALE:.2f}x, "
            f"preview_start_state={preview_snapshot.summary()}, final_state={final_snapshot.summary()})."
        )
        print(f"[Integration] Waiting for robot sync at frame {current_idx} before replay resumes.")
        begin_robot_sync("post-replan", grip_target_override=paused_grip_target)

    print(
        "[Integration] Runtime started\n"
        f"  version={INTEGRATION_VERSION}\n"
        f"  xml={args.xml}\n"
        f"  trajectory={trajectory.path}\n"
        "  VR trigger teleop is disabled here for safety.\n"
        "  Controls: Space sim start/stop, Backspace/B reset, X intervene/finish, Y/Esc cancel."
    )
    print("  Replay mode: robot-only control playback with physics stepping for object interaction.")
    if policy_state_mirror.enabled:
        print("  Live policy-state mirror: ON (local trajectory stepping disabled; sensors follow Intervene state).")
    print(f"  Replay speed scale: {args.replay_speed:.2f}x")
    print(f"  Post-replan mode: {args.post_replan_mode}")
    if args.mirror_robot:
        print(
            f"  [MIRROR ROBOT] ENABLED — robot_key={args.robot_key} "
            f"control_hz={args.control_hz} max_dq={robot_replay_max_dq:.3f} rad/s (replay-matched), "
            f"replan_hold_max_dq={REPLAN_HOLD_MAX_DQ:.3f} rad/s."
        )
        velocity_diag = _trajectory_velocity_diagnostic(
            trajectory,
            replay_speed=float(args.replay_speed),
            max_dq=float(robot_replay_max_dq),
        )
        if velocity_diag is not None:
            print(
                "  [RobotTiming] trajectory_required="
                f"{velocity_diag['max_required_dq']:.3f} rad/s "
                f"max_limit_ratio={velocity_diag['max_limit_ratio']:.2f} "
                f"worst_joint=J{int(velocity_diag['worst_joint'])} "
                f"lag_pause={ROBOT_REPLAY_LAG_PAUSE_RAD:.2f} rad/{ROBOT_REPLAY_LAG_PAUSE_CYCLES} cycles."
            )
            if float(velocity_diag["max_limit_ratio"]) > 1.0:
                print(
                    "  [RobotTiming][WARN] Replay contains joint deltas faster than the "
                    "configured robot limit; runtime will clamp and may pause/resync on lag."
                )
        if args.mirror_on_select_only:
            print(
                "  [MIRROR ROBOT] select-only mode: grid stays sim-only; "
                "ENTER_SINGLE activates single-view PC, A/X starts robot sync, EXIT_SINGLE releases."
            )
        elif not args.start_paused:
            print(
                "  [MIRROR ROBOT][WARN] --start_paused was NOT set. "
                "The robot will begin mirroring immediately from frame 0 when the loop starts. "
                "Make sure the robot is near the trajectory start pose before running."
            )
    else:
        print("  [MIRROR ROBOT] disabled (pass --mirror_robot --robot_key <key> to enable).")
    for line in setup["startup_scene_diagnostics"]:
        print(line)
    if args.show_mujoco_window:
        print("  Note: --show_mujoco_window opens the offscreen helper window, not a full MuJoCo viewer.")
    if args.pc and args.pc_debug_visibility:
        print(
            "  PC visibility debug:"
            f" cams={sorted(list(pc_debug_visibility_cam_set))}"
            f" outdir={pc_debug_visibility_outdir}"
        )

    if state == STATE_REPLAY_RUNNING and args.mirror_robot and not args.mirror_on_select_only:
        begin_robot_sync("startup")

    # Disable Python's automatic GC to prevent periodic 50-100 ms pauses that show as precise
    # repeating FPS drops on the Quest FPS chart. We collect on a controlled schedule instead,
    # during the natural idle time at the bottom of each main-loop iteration. gc.collect() at
    # startup cleans up initialization allocations so the first controlled collect is cheap.
    _gc_interval = max(1, int(args.gc_interval)) if getattr(args, "gc_interval", 300) > 0 else None
    if _gc_interval is not None:
        gc.collect()
        gc.disable()
    _gc_frame_counter = [0]
    _pc_fps_cap = float(getattr(args, "pc_fps_cap", 0.0))
    _pc_last_publish_t: dict = {}   # cam_name -> last wall-clock publish time

    # Main-loop hang watchdog. The main loop publishes top/right/left cameras + point clouds
    # on this thread; a blocking call there (e.g. the first render of a non-thumbnail camera on
    # the PC-active path) freezes the whole session while the CmdListener / ZMQ-monitor daemon
    # threads keep running, so the process looks alive but stops publishing. This daemon watches a
    # heartbeat timestamp the loop bumps each iteration; if it goes stale it dumps ALL thread
    # stacks (faulthandler) so the hung frame is visible in the unified log. Dumps once per stall,
    # re-arms on recovery. Disabled with --main_loop_watchdog_s 0.
    _loop_heartbeat = [time.time()]
    _watchdog_timeout = float(getattr(args, "main_loop_watchdog_s", 0.0) or 0.0)
    if _watchdog_timeout > 0.0:
        def _main_loop_watchdog():
            stall_reported = False
            while True:
                time.sleep(min(1.0, _watchdog_timeout / 2.0))
                age = time.time() - _loop_heartbeat[0]
                if age >= _watchdog_timeout:
                    if not stall_reported:
                        print(
                            f"[Watchdog][STALL] main loop has not advanced for {age:.1f}s "
                            f"(threshold={_watchdog_timeout:.1f}s). Dumping all thread stacks:",
                            flush=True,
                        )
                        faulthandler.dump_traceback()
                        sys.stderr.flush()
                        stall_reported = True
                else:
                    stall_reported = False
        threading.Thread(target=_main_loop_watchdog, daemon=True, name="MainLoopWatchdog").start()

    try:
        while True:
            loop_t0 = time.time()
            _loop_heartbeat[0] = loop_t0
            renderer.poll_events()
            if args.show_mujoco_window and (not renderer.is_window_open()):
                print("[Integration] MuJoCo window was closed; stopping.")
                break

            selected_object_idx, next_object_move_time = apply_object_control(
                args=args,
                model=model,
                data=data,
                renderer=renderer,
                movable_objects=setup["movable_objects"],
                selected_object_idx=selected_object_idx,
                key_latch=object_key_latch,
                next_object_move_time=next_object_move_time,
            )

            if not policy_state_mirror.enabled:
                if unified.key_pressed_once(renderer, workflow_key_latch, glfw.KEY_SPACE):
                    pending["sim_toggle"] = True
                if unified.key_pressed_once(renderer, workflow_key_latch, glfw.KEY_BACKSPACE):
                    pending["reset"] = True
                if unified.key_pressed_once(renderer, workflow_key_latch, glfw.KEY_X) or unified.key_pressed_once(renderer, workflow_key_latch, glfw.KEY_ENTER):
                    pending["intervene"] = True
                if unified.key_pressed_once(renderer, workflow_key_latch, glfw.KEY_Y) or unified.key_pressed_once(renderer, workflow_key_latch, glfw.KEY_ESCAPE):
                    pending["cancel"] = True

            if pending["exit_single"]:
                pending["exit_single"] = False
                pending["enter_single"] = False
                pending["single_view_ready"] = False
                pending["pause"] = False
                pending["sim_toggle"] = False
                pending["sim_pause"] = False
                pending["sim_resume"] = False
                pending["reset"] = False
                pending["intervene"] = False
                pending["cancel"] = False
                exit_single_view()
            if pending["enter_single"]:
                pending["enter_single"] = False
                enter_single_view()
            if pending["single_view_ready"]:
                pending["single_view_ready"] = False
                single_view_ready()

            if policy_state_mirror.enabled:
                pending["pause"] = False
                pending["sim_toggle"] = False
                pending["sim_pause"] = False
                pending["sim_resume"] = False
                pending["reset"] = False
                pending["intervene"] = False
                pending["cancel"] = False
                latest_policy_state = policy_state_mirror.drain_latest()
                if latest_policy_state is not None:
                    applied = policy_state_mirror.apply_to(model, data, latest_policy_state)
                    if applied is not None and applied.get("variant_changed"):
                        # State is ALREADY applied on the current model (apply_to never
                        # withholds it). Try to adopt the new one; re-apply only if the swap
                        # actually landed this tick, so the state ends up on the new MjData.
                        wanted_key = applied.get("variant_key", BASE_VARIANT_KEY)
                        model, data = rebuild_for_variant(applied.get("variant"), wanted_key)
                        if policy_state_mirror.active_variant_key == wanted_key:
                            applied = policy_state_mirror.apply_to(
                                model, data, latest_policy_state
                            )
                    if applied is not None and variant_cache is not None:
                        nxt = policy_state_mirror.latest_next_variant
                        if isinstance(nxt, dict):
                            variant_cache.prefetch(nxt)
                    if applied is not None:
                        # The policy frame belongs to a different episode and timeline than
                        # the local fallback trajectory. PolicyStateMirror owns that identity;
                        # current_idx must remain a valid index into `trajectory`.
                        if applied["first_apply"] or applied["episode_changed"]:
                            drop_point_cloud_backlog("POLICY_STATE_MIRROR")
                policy_state_mirror.warn_if_stale()
                state = STATE_REPLAY_PAUSED
                replay_sync = None
                replay_soft_start_t0 = None
                replan = None

            if pending["reset"]:
                pending["reset"] = False
                reset_replay()
            if pending["cancel"]:
                pending["cancel"] = False
                if state in (STATE_REPLAN_HOLD, STATE_REPLAN_TRANSITION, STATE_REPLAN_RECORD):
                    cancel_replan()
                elif state in (STATE_REPLAY_RUNNING, STATE_REPLAY_SYNC):
                    pause_replay()
            if pending["sim_toggle"]:
                pending["sim_toggle"] = False
                sim_toggle()
            if pending["sim_pause"]:
                pending["sim_pause"] = False
                pause_replay()
            if pending["sim_resume"]:
                pending["sim_resume"] = False
                sim_resume()
            if pending["pause"]:
                pending["pause"] = False
                # Legacy A (standalone MetaQuest3 path only; not forwarded by the multi-session
                # single view, which now uses Y for sim start/stop).
                if ensure_single_view_for_control("A"):
                    mirror_align_resume()
            if pending["intervene"]:
                pending["intervene"] = False
                if ensure_single_view_for_control("X"):
                    start_intervention_auto_align()

            # Forward intervention state transitions from policy mirror → VR HUD.
            if policy_state_mirror is not None:
                _mirror_evt = policy_state_mirror.drain_status_event()
                if _mirror_evt:
                    queue_status(_mirror_evt)
                    print(f"[PolicyMirror] publishing '{_mirror_evt}' to Quest HUD × {status_pub_pending['repeats']} frames")

            # Re-send any queued one-shot status event to the Quest for a few frames.
            if status_pub_pending["repeats"] > 0 and status_pub_pending["event"]:
                try:
                    sensor_node.publish(
                        "SimPub/Status/intervention",
                        str(status_pub_pending["event"]).encode("utf-8"),
                    )
                except Exception:
                    pass
                status_pub_pending["repeats"] -= 1

            # Publish authoritative paused flag for the Unity thumbnail "PAUSED" badge.
            # Source of truth differs by mode:
            #   - policy mode: `state` is force-set to STATE_REPLAY_PAUSED every frame as
            #     bookkeeping, so it's useless here; use the real user-pause the policy runner
            #     streams (PolicyStateMirror.latest_paused, from app.py's player.is_paused).
            #   - replay/mirror mode: STATE_REPLAY_PAUSED is the genuine sim-pause signal.
            if policy_state_mirror.enabled:
                paused_now = bool(policy_state_mirror.latest_paused)
                intervening_now = bool(policy_state_mirror.latest_intervention_live)
            else:
                paused_now = (state == STATE_REPLAY_PAUSED)
                intervening_now = state in (
                    STATE_REPLAN_HOLD, STATE_REPLAN_TRANSITION, STATE_REPLAN_RECORD,
                )
            _now_paused_pub = time.time()
            if paused_now != paused_pub_state["last_sent"] or _now_paused_pub >= paused_pub_state["next_pub_t"]:
                try:
                    sensor_node.publish(
                        "SimPub/Status/paused",
                        b"1" if paused_now else b"0",
                    )
                    # Authoritative per-session intervention LEVEL. The OOD supervisor on the
                    # Quest needs to protect ANY intervening session, not just the selected
                    # one — its old guard tested the selected index, which the guard above
                    # already covers, so it never actually protected anything.
                    sensor_node.publish(
                        "SimPub/Status/intervening",
                        b"1" if intervening_now else b"0",
                    )
                    if policy_state_mirror.enabled:
                        _last_change = policy_state_mirror.last_state_change_wall
                        _session_state = {
                            "version": 1,
                            "session_index": int(getattr(args, "session_index", 0)),
                            "topic_port": int(args.topic_port),
                            "episode_id": str(policy_state_mirror.last_episode_id or ""),
                            "policy_seq": policy_state_mirror.last_applied_seq,
                            "frame_idx": int(policy_state_mirror.latest_frame_idx),
                            "mode": str(policy_state_mirror.latest_mode),
                            "paused": paused_now,
                            "intervention_live": intervening_now,
                            "intervention_phase": str(policy_state_mirror.latest_intervention_phase),
                            "upstream_stale": bool(policy_state_mirror.stale),
                            "upstream_stale_age_s": round(float(policy_state_mirror.stale_age_s), 3),
                            "qpos_delta_norm": round(float(policy_state_mirror.last_qpos_delta_norm), 8),
                            "state_change_age_s": (
                                None if _last_change is None
                                else round(max(0.0, _now_paused_pub - _last_change), 3)
                            ),
                            "wall_t": _now_paused_pub,
                        }
                    else:
                        _session_state = {
                            "version": 1,
                            "session_index": int(getattr(args, "session_index", 0)),
                            "topic_port": int(args.topic_port),
                            "episode_id": "",
                            "policy_seq": None,
                            "frame_idx": int(current_idx),
                            "mode": str(state),
                            "paused": paused_now,
                            "intervention_live": intervening_now,
                            "intervention_phase": (
                                "human_control" if intervening_now else "policy_resumed"
                            ),
                            "upstream_stale": False,
                            "upstream_stale_age_s": 0.0,
                            "qpos_delta_norm": 0.0,
                            "state_change_age_s": None,
                            "wall_t": _now_paused_pub,
                        }
                    sensor_node.publish(
                        "SimPub/Status/session_state",
                        json.dumps(_session_state, separators=(",", ":")).encode("utf-8"),
                    )
                except Exception:
                    pass
                paused_pub_state["last_sent"] = paused_now
                paused_pub_state["next_pub_t"] = _now_paused_pub + 0.25  # ~4 Hz heartbeat


            if state == STATE_REPLAY_RUNNING:
                sim_target = replay_sim_zero + (time.time() - replay_wall_zero) * float(args.replay_speed)
                while current_idx + 1 < len(trajectory) and trajectory.sim_t[current_idx + 1] <= sim_target:
                    current_idx += 1
                trajectory.apply_controls(current_idx, data, replay_bindings=replay_bindings)
                step_now = time.time()
                replay_step_accumulator += max(0.0, step_now - replay_step_wall_prev)
                replay_step_wall_prev = step_now
                substeps = int(replay_step_accumulator / max(float(model.opt.timestep), 1e-6))
                if substeps > 0:
                    replay_step_accumulator -= float(substeps) * float(model.opt.timestep)
                    for _ in range(substeps):
                        mujoco.mj_step(model, data)
                if has_robot_control():
                    robot_thread.set_reading(True)
                    active_max_dq = float(robot_replay_max_dq)
                    if replay_soft_start_t0 is not None:
                        ramp = np.clip(
                            (time.time() - replay_soft_start_t0)
                            / max(float(ROBOT_SOFTSTART_DURATION_S), 1e-6),
                            0.0,
                            1.0,
                        )
                        active_max_dq = float(ROBOT_SYNC_MAX_DQ + (robot_replay_max_dq - ROBOT_SYNC_MAX_DQ) * ramp)
                        if ramp >= 1.0:
                            replay_soft_start_t0 = None

                    q_real, _, _ = robot_thread.get_state()

                    # Overwrite sim arm qpos with measured real-robot positions so
                    # the point cloud reflects where the arm actually is.
                    # mj_kinematics (FK-only, no solver) updates body/geom/camera
                    # world transforms at ~0.1–0.3 ms vs mj_forward at 1–3 ms.
                    if q_real is not None and replay_bindings is not None:
                        q_arr = np.asarray(q_real, dtype=np.float64)
                        n = min(len(replay_bindings.arm_qpos_idx), q_arr.shape[0])
                        for i in range(n):
                            data.qpos[int(replay_bindings.arm_qpos_idx[i])] = float(q_arr[i])
                        mujoco.mj_kinematics(model, data)

                    robot_thread.mirror_command(
                        np.asarray(data.ctrl[:7], dtype=np.float64),
                        dt=control_dt,
                        max_dq=active_max_dq,
                        grip_width=trajectory.grip_width_at(current_idx, replay_bindings=replay_bindings),
                    )
                    if q_real is not None:
                        replay_err = np.asarray(data.ctrl[:7], dtype=np.float64) - np.asarray(q_real, dtype=np.float64)
                        replay_max_abs_err = float(np.max(np.abs(replay_err)))
                        if replay_max_abs_err > float(ROBOT_REPLAY_LAG_PAUSE_RAD):
                            robot_replay_lag_cycles += 1
                        else:
                            robot_replay_lag_cycles = 0
                        if sensor_state.frame_idx % max(1, args.log_every) == 0 and robot_replay_lag_cycles > 0:
                            print(
                                "[Integration][RobotLag] "
                                f"max_abs_err={replay_max_abs_err:.4f} rad "
                                f"cycles={robot_replay_lag_cycles}/{ROBOT_REPLAY_LAG_PAUSE_CYCLES}"
                            )
                        if robot_replay_lag_cycles >= int(ROBOT_REPLAY_LAG_PAUSE_CYCLES):
                            robot_thread.set_reading(False)
                            replay_sync = None
                            replay_soft_start_t0 = None
                            state = STATE_REPLAY_PAUSED
                            robot_replay_lag_cycles = 0
                            print(
                                "[Integration][RobotLag][WARN] Robot fell behind MuJoCo replay "
                                f"(max_abs_err={replay_max_abs_err:.4f} rad). "
                                "Paused replay; press A to resync/resume."
                            )
                            continue
                if current_idx >= len(trajectory) - 1:
                    if robot_thread is not None:
                        robot_thread.set_reading(False)
                    state = STATE_REPLAY_PAUSED
                    print("[Integration] Replay reached the last frame and is now paused.")

            elif state == STATE_REPLAY_SYNC:
                if robot_thread is None or replay_sync is None:
                    state = STATE_REPLAY_PAUSED
                    replay_sync = None
                    if robot_thread is not None:
                        robot_thread.set_reading(False)
                    print("[Integration][WARN] Robot sync aborted; replay is paused.")
                else:
                    robot_thread.mirror_command(
                        replay_sync.target_q,
                        dt=control_dt,
                        max_dq=float(ROBOT_SYNC_MAX_DQ),
                        grip_width=replay_sync.grip_target,
                    )
                    q_real, _, _ = robot_thread.get_state()
                    if q_real is not None:
                        err = np.asarray(replay_sync.target_q, dtype=np.float64) - np.asarray(q_real, dtype=np.float64)
                        max_abs_err = float(np.max(np.abs(err)))
                        if max_abs_err <= float(ROBOT_SYNC_JOINT_TOL_RAD):
                            replay_sync.stable_cycles += 1
                        else:
                            replay_sync.stable_cycles = 0
                        if sensor_state.frame_idx % max(1, args.log_every) == 0:
                            print(
                                "[Integration][RobotSync] "
                                f"max_abs_err={max_abs_err:.4f} rad "
                                f"stable={replay_sync.stable_cycles}/{ROBOT_SYNC_STABLE_CYCLES}"
                            )
                        if replay_sync.stable_cycles >= int(ROBOT_SYNC_STABLE_CYCLES):
                            robot_thread.set_reading(False)
                            robot_replay_lag_cycles = 0
                            after_align = replay_sync.after_align
                            replay_sync = None
                            if after_align == "intervene":
                                state = STATE_REPLAY_PAUSED
                                replay_soft_start_t0 = None
                                print("[Integration] Robot aligned with MuJoCo pose. Starting HUMAN_CONTROL recording.")
                                start_replan_recording()
                            else:
                                replay_soft_start_t0 = time.time()
                                replay_wall_zero = replay_soft_start_t0
                                replay_sim_zero = float(trajectory.sim_t[current_idx])
                                replay_step_wall_prev = replay_soft_start_t0
                                replay_step_accumulator = 0.0
                                state = STATE_REPLAY_RUNNING
                                print("[Integration] Robot aligned with MuJoCo pose. Resuming trajectory with slow ramp.")
                        elif (time.time() - replay_sync.start_t) > float(ROBOT_SYNC_TIMEOUT_S):
                            robot_thread.set_reading(False)
                            state = STATE_REPLAY_PAUSED
                            replay_sync = None
                            print(
                                "[Integration][WARN] Robot sync timed out; replay remains paused. "
                                "Move the robot closer to the MuJoCo pose and resume again."
                            )

            elif state == STATE_REPLAY_PAUSED:
                pass

            elif state == STATE_REPLAN_HOLD:
                if has_robot_control() and replan is not None:
                    robot_thread.hold_pose(
                        replan.q_seed,
                        replan.finger_seed,
                        dt=control_dt,
                        max_dq=REPLAN_HOLD_MAX_DQ,
                    )
            elif state == STATE_REPLAN_TRANSITION:
                if robot_thread is None or replan is None or replan.recorder is None:
                    raise RuntimeError("Replan transition state entered without a valid session.")
                done, success, message = robot_thread.is_guidance_transition_done()
                if done:
                    if success:
                        robot_thread.set_reading(True)
                        state = STATE_REPLAN_RECORD
                        # Robot has released into HUMAN_CONTROL — this is the exact moment the
                        # Quest "Intervention has begun" HUD should appear.
                        queue_status("intervention_started")
                        print("[Integration] Replan recording active. Press X to finish and stitch, or Y to cancel.")
                    else:
                        robot_thread.ensure_mode(robot_thread.control_type_enum.HYBRID_JOINT_IMPEDANCE_CONTROL)
                        replan.hold_confirmed = False
                        replan.recorder = None
                        replan.suffix_path = None
                        state = STATE_REPLAN_HOLD
                        print(f"[Integration][WARN] HUMAN_CONTROL transition failed: {message}")
                        print("[Integration] Stayed in replan hold. Press X to confirm hold, then X again to retry.")

            elif state == STATE_REPLAN_RECORD:
                if robot_thread is None or replan is None or replan.recorder is None:
                    raise RuntimeError("Replan recording state entered without a valid session.")
                q_real, dq_real, gripper_width = robot_thread.get_state()
                if q_real is not None:
                    data.ctrl[:7] = (1.0 - 0.6) * data.ctrl[:7] + 0.6 * q_real
                    replay_bindings.sync_finger_state_from_grip_width(
                        data,
                        float(gripper_width),
                        dt=control_dt,
                        alpha=GRIPPER_SIM_QPOS_SYNC_ALPHA,
                    )
                    mujoco.mj_forward(model, data)
                    for _ in range(max(1, int(control_dt / model.opt.timestep))):
                        mujoco.mj_step(model, data)
                    replan.recorder.record(
                        now_wall=time.time(),
                        data=data,
                        q_real=q_real,
                        dq_real=dq_real,
                        gripper_width=gripper_width,
                    )

            # Drain completed PC frames from the background worker and publish them.
            # Always drain the full output queue to prevent backlog; the rate cap below
            # controls which frames are actually published to ZMQ.
            _pc_published_this_tick = False
            if pc_worker is not None:
                _pc_drain_t0 = time.time()
                reconcile_pc_gpu_memory()
                _pc_results = pc_worker.drain()
                if _pc_results and not _PC_DIAG["drain_logged"]:
                    print(f"[Integration][DIAG] first PC drain: {len(_pc_results)} frame(s); "
                          f"gate(point_cloud_publish_allowed)={point_cloud_publish_allowed()}", flush=True)
                    _PC_DIAG["drain_logged"] = True
                _pc_now = time.time()
                _pc_min_dt = (1.0 / _pc_fps_cap) if _pc_fps_cap > 0 else 0.0
                for _pc_cam, (_pc_frame, _pc_metadata) in _pc_results:
                    if not point_cloud_publish_allowed():
                        continue
                    # Drop anything built against the previous model. At most one frame per
                    # camera can be stale (the worker keeps latest-per-camera), but without
                    # this it would be published as current.
                    _built_key = _pc_metadata.get("variant_key", BASE_VARIANT_KEY)
                    if _built_key != policy_state_mirror.active_variant_key:
                        _pc_stale_epoch_drops[0] += 1
                        if _pc_stale_epoch_drops[0] <= 3:
                            print(f"[Variant] dropped stale {_pc_cam}/pc built on "
                                  f"{_built_key} (now {policy_state_mirror.active_variant_key})")
                        if metrics is not None:
                            metrics.observe("pc_dropped_stale_epoch", 1.0)
                        continue
                    # Per-camera rate cap: stagger delivery so all 3 cameras don't publish
                    # simultaneously. When all workers complete at once, only cameras past
                    # their interval are published this tick; others are dropped (the worker
                    # will have a fresher build ready next tick anyway).
                    if _pc_min_dt > 0 and _pc_now - _pc_last_publish_t.get(_pc_cam, 0.0) < _pc_min_dt:
                        continue
                    _pc_last_publish_t[_pc_cam] = _pc_now
                    sensor_node.publish(f"SimPub/Sensors/{_pc_cam}/pc", _pc_frame.payload)
                    if sensor_state.pc_visit_started_t is not None:
                        sensor_state.pc_visit_publish_counts[_pc_cam] = (
                            sensor_state.pc_visit_publish_counts.get(_pc_cam, 0) + 1
                        )
                    _pc_published_this_tick = True
                    # Feed the real point count back into the declared-capacity high-water
                    # mark. This is the async path (the one actually used with
                    # --pc_worker_threads), so it must be here and not only in the
                    # synchronous fallback inside publish_sensor_frames().
                    _note_declared_capacity_observation(
                        args, sensor_state, _pc_cam, _pc_frame.actual_count,
                        setup["pc_declared_capacity_by_cam"].get(
                            _pc_cam, int(args.pc_max_points)
                        ),
                    )
                    if metrics is not None:
                        metrics.observe(
                            f"pc_bytes_kb.{_pc_cam}", len(_pc_frame.payload) / 1024.0
                        )
                        _published_wall = time.time()
                        for _metric_name, _timestamp_key in (
                            ("pc_submission_to_publish_ms", "submitted_wall"),
                            ("pc_render_to_publish_ms", "rendered_wall"),
                            ("pc_build_to_publish_ms", "build_done_wall"),
                            ("pc_state_to_publish_ms", "state_wall"),
                        ):
                            _timestamp = _pc_metadata.get(_timestamp_key)
                            if _timestamp is not None:
                                metrics.observe(
                                    f"{_metric_name}.{_pc_cam}",
                                    max(0.0, _published_wall - float(_timestamp)) * 1000.0,
                                )
                        if _pc_metadata.get("policy_seq") is not None:
                            metrics.set_label(
                                f"pc_policy_seq.{_pc_cam}", _pc_metadata["policy_seq"]
                            )
                    if not sensor_state.first_pc_logged.get(_pc_cam, False):
                        print(
                            f"[Integration] First PC publish (async): cam={_pc_cam} "
                            f"topic=SimPub/Sensors/{_pc_cam}/pc "
                            f"N={_pc_frame.actual_count}"
                        )
                        sensor_state.first_pc_logged[_pc_cam] = True
                if metrics is not None and _pc_results:
                    metrics.observe("pc_drain_ms", (time.time() - _pc_drain_t0) * 1000.0)

            table_plane, sensor_state = publish_sensor_frames(
                args=args,
                model=model,
                data=data,
                renderer=renderer,
                sensor_node=sensor_node,
                cams=setup["cams"],
                rgb_cam_set=setup["rgb_cam_set"],
                rgbd_cam_set=setup["rgbd_cam_set"],
                pc_cam_set=setup["pc_cam_set"],
                rgb_panel_cam_set=setup["rgb_panel_cam_set"],
                publish_rgb_topic=setup["publish_rgb_topic"],
                extrinsics=setup["extrinsics"],
                pc_pipeline=setup["pc_pipeline"],
                pc_target_sample_budget=setup["pc_target_sample_budget"],
                pc_max_depth=setup["pc_max_depth"],
                object_bbox_min=setup["object_bbox_min"],
                object_bbox_max=setup["object_bbox_max"],
                table_plane=table_plane,
                anchor_site_id=setup["anchor_site_id"],
                anchor_body_id=setup["anchor_body_id"],
                manual_anchor_world=setup["manual_anchor_world"],
                flip_z=setup["flip_z"],
                sensor_state=sensor_state,
                pc_debug_visibility_cam_set=setup["pc_debug_visibility_cam_set"],
                pc_debug_visibility_outdir=setup["pc_debug_visibility_outdir"],
                pc_declared_capacity_by_cam=setup["pc_declared_capacity_by_cam"],
                pc_worker=pc_worker,
                selected_sensor_active=selected_sensor_active_arg(),
                force_sensor_idle=policy_handoff_sensor_idle,
                metrics=metrics,
            )

            # Windowed metrics: derive per-topic publish rates from the cumulative
            # counters IsolatedSensorPublisher already keeps, then emit a row when the
            # window elapses. All the expensive work happens inside maybe_emit(), which
            # only fires once per --metrics_window_s.
            if metrics is not None and metrics.due():
                try:
                    if hasattr(sensor_node, "get_diagnostics"):
                        _mdiag = sensor_node.get_diagnostics()
                        _topic_counts = _mdiag.get("topic_publish_counts") or {}
                        for _topic, _count in _topic_counts.items():
                            # "SimPub/Sensors/top/pc" -> "top/pc"
                            _short = "/".join(str(_topic).split("/")[-2:])
                            metrics.update_rate(_short, float(_count))
                        metrics.update_rate(
                            "_all_topics", float(_mdiag.get("publish_count", 0) or 0)
                        )
                        metrics.set_label(
                            "coalesced", int(_mdiag.get("coalesced_count", 0) or 0)
                        )
                        metrics.set_label(
                            "publish_errors", int(_mdiag.get("publish_errors", 0) or 0)
                        )
                        metrics.set_label(
                            "pending_oldest_s",
                            round(float(_mdiag.get("oldest_pending_age_s", 0.0) or 0.0), 4),
                        )
                    if robot_thread is not None:
                        _rdiag = robot_thread.motion_diagnostics()
                        metrics.set_label(
                            "robot_cmd_age_s", round(float(_rdiag.get("last_age_s", 0.0)), 4)
                        )
                    if policy_state_mirror.enabled:
                        metrics.update_rate(
                            "policy_received", float(policy_state_mirror.received_messages)
                        )
                        metrics.update_rate(
                            "policy_distinct_qpos",
                            float(policy_state_mirror.distinct_qpos_updates),
                        )
                        metrics.update_rate(
                            "policy_repeated_qpos",
                            float(policy_state_mirror.repeated_qpos_updates),
                        )
                        metrics.set_label(
                            "policy_seq", policy_state_mirror.last_applied_seq
                        )
                        metrics.set_label(
                            "policy_seq_gaps", policy_state_mirror.received_seq_gaps
                        )
                        metrics.set_label(
                            "policy_state_age_s",
                            None if policy_state_mirror.last_receive_wall is None else round(
                                max(0.0, time.time() - policy_state_mirror.last_receive_wall), 4
                            ),
                        )
                    _active_tracker = globals().get("_ACTIVE_PC_SESSION_TRACKER")
                    if _active_tracker is not None:
                        metrics.set_label("active_session_count", _active_tracker.count())
                    try:
                        metrics.set_label("peer_count", int(sensor_node.peer_count()))
                    except Exception:
                        pass
                    metrics.maybe_emit()
                except Exception as _mexc:
                    print(f"[Metrics][WARN] window emit failed: {_mexc}")

            if sensor_state.frame_idx > 0 and sensor_state.frame_idx % max(1, args.log_every) == 0:
                if hasattr(sensor_node, 'get_diagnostics'):
                    diag = sensor_node.get_diagnostics()
                    msg = (
                        f"[SensorPub] frame={sensor_state.frame_idx} "
                        f"published={diag.get('publish_count', -1)} "
                        f"errors={diag.get('publish_errors', -1)}"
                    )
                    if 'enqueue_count' in diag:
                        msg += (
                            f" enqueued={diag.get('enqueue_count', -1)} "
                            f"coalesced={diag.get('coalesced_count', -1)} "
                            f"pending_topics={diag.get('pending_topics', -1)}"
                        )
                    if 'oldest_pending_age_s' in diag:
                        msg += f" pending_oldest_s={float(diag.get('oldest_pending_age_s', 0.0)):.3f}"
                    if 'pending_topic_sample' in diag and diag.get('pending_topic_sample'):
                        msg += f" pending_sample={diag.get('pending_topic_sample')}"
                    if 'topic_publish_counts' in diag and diag.get('topic_publish_counts'):
                        msg += f" topic_pub={diag.get('topic_publish_counts')}"
                    if 'peer_count_est' in diag:
                        msg += f" peers={diag.get('peer_count_est', -1)}"
                    if 'monitor_last_event' in diag and diag.get('monitor_last_event'):
                        msg += f" mon_last={diag.get('monitor_last_event')}"
                        if diag.get('monitor_last_endpoint'):
                            msg += f" mon_ep={diag.get('monitor_last_endpoint')}"
                    if robot_thread is not None:
                        try:
                            motion_diag = robot_thread.motion_diagnostics()
                            msg += (
                                f" robot_cmd_age={motion_diag.get('last_age_s', 0.0):.3f}s"
                                f" robot_cmd_applied={int(motion_diag.get('applied', 0))}"
                                f" robot_cmd_coalesced={int(motion_diag.get('coalesced', 0))}"
                            )
                        except Exception:
                            pass
                    print(msg)

            if args.show_mujoco_window:
                display_cam = args.mujoco_window_cam or setup["cams"][0]
                renderer.render(
                    data,
                    display_cam,
                    blit_to_window=True,
                    paused_overlay=(state == STATE_REPLAY_PAUSED) and not policy_state_mirror.enabled,
                )

            sleep_t = control_dt - (time.time() - loop_t0)
            _gc_frame_counter[0] += 1
            if _gc_interval is not None and _gc_frame_counter[0] >= _gc_interval:
                _gc_frame_counter[0] = 0
                # Skip GC on any tick that published PC frames: gc.collect() with large numpy
                # arrays from the PC pipeline can take 5-30ms and would stack with the burst
                # drain cost, compounding the exact latency spike we're trying to eliminate.
                if sleep_t > 0.003 and not _pc_published_this_tick:
                    _t_gc = time.time()
                    gc.collect()
                    sleep_t -= (time.time() - _t_gc)
            if sleep_t > 0:
                time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n[Integration] Stopping...")
    finally:
        try:
            if risk_pub_thread is not None:
                risk_pub_thread.close()
        except Exception:
            pass
        try:
            cmd_listener_stop.set()
        except Exception:
            pass
        try:
            if cmd_listener_sock is not None:
                cmd_listener_sock.close(linger=0)
        except Exception:
            pass
        try:
            policy_sock = policy_cmd_ref.get("sock")
            if policy_sock is not None:
                policy_sock.close(linger=0)
                policy_cmd_ref["sock"] = None
        except Exception:
            pass
        try:
            policy_state_mirror.close()
        except Exception:
            pass
        try:
            active_tracker = globals().get("_ACTIVE_PC_SESSION_TRACKER")
            if active_tracker is not None:
                active_tracker.close()
        except Exception:
            pass
        # Latency budget BEFORE metrics.close(), so the numbers land in the same
        # session log block as the lifetime perf summary they belong with.
        try:
            _probe = globals().get("_LATENCY_PROBE")
            if _probe is not None:
                import json as _json
                _summary = _probe.summary()
                _cams = _summary.get("cameras") or {}
                if not _cams:
                    print(
                        "[LatencyProbe] NO ECHOES RECEIVED. Either no headset entered "
                        "single view, or the deployed APK has no LatencyStampProbe.cs. "
                        "This is an absence of data, NOT a latency of zero.",
                        flush=True,
                    )
                else:
                    print("[LatencyProbe] end-to-end budget (publisher clock only, "
                          "no clock sync):", flush=True)
                    for _cam, _row in sorted(_cams.items()):
                        _oh = _row.get("on_headset_ms") or {}
                        _tr = _row.get("transport_rtt_ms") or {}
                        _a = _row.get("rtt_a_ms") or {}
                        print(
                            f"[LatencyProbe]   {_cam}: n={_a.get('n', 0)} "
                            f"decode+render+present(EXACT)={_oh.get('med', float('nan')):.2f} ms  "
                            f"network_rtt(down+up)={_tr.get('med', float('nan')):.2f} ms  "
                            f"rtt_a_med={_a.get('med', float('nan')):.2f} ms",
                            flush=True,
                        )
                    print(f"[LatencyProbe] json={_json.dumps(_summary)}", flush=True)
        except Exception as _ex:
            print(f"[LatencyProbe] summary failed: {_ex}", flush=True)
        try:
            if metrics is not None:
                metrics.close()
        except Exception:
            pass
        try:
            pc_worker.stop()
        except Exception:
            pass
        if robot_thread is not None:
            try:
                robot_thread.close()
            except Exception:
                pass
        try:
            if robot_lock is not None:
                robot_lock.release()
        except Exception:
            pass
        try:
            sensor_node.stop()
        except Exception:
            pass
        try:
            renderer.close()
        except Exception:
            pass
        try:
            pub_obj = scene_pub_ref["pub"]
            if pub_obj is not None:
                pub_obj.shutdown()
        except Exception:
            pass
        if _win_timer_raised:
            try:
                import ctypes
                ctypes.windll.winmm.timeEndPeriod(1)
            except Exception:
                pass
