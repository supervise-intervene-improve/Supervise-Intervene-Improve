import copy
import faulthandler
import hashlib
import json
import math
import os
import queue
import sys
import threading
import time
from pathlib import Path

import glfw
import mujoco

from playback.player import TrajectoryPlayer, gripper_ctrl_to_width
from playback.policy_player import PolicyPlayer, PolicyPlayerConfig
from rendering.viewer import SceneViewer
from input.callbacks import MouseState, install_callbacks
from data_io.stitch import stitch_npz, make_replanned_output_path
from data_io.ood_scene_poses import (
    list_scenes as list_ood_scenes,
    scene_poses as ood_scene_poses,
    TASK_DIRS as OOD_TASK_DIRS,
)
from data_io.ood_scene_variants import (
    MODEL_VARIANT_TASKS as OOD_MODEL_VARIANT_TASKS,
    scene_variant as ood_scene_variant,
)
from model_variants.descriptor import (
    BASE_KEY as BASE_VARIANT_KEY,
    scene_closure_sha256,
    variant_key,
)
from data_io.experiment_block import (
    describe as describe_experiment_block,
    resolve_block_from_env as resolve_experiment_block,
)
from data_io.study_config import (
    StudyConfigError,
    describe as describe_study,
    load_block,
    study_block_from_experiment,
)
from data_io.study_logger import (
    OUTCOME_FAILURE,
    OUTCOME_INCOMPLETE,
    OUTCOME_SUCCESS,
    StudySession,
)
from utils.evaluate_act_mujoco import (
    bodies_in_contact,
    cups_fail_reason,
    cups_success_status,
    fail_reason,
    placement_metrics,
    tshape_fail_reason,
    tshape_success_status,
)

from PIL import Image
import numpy as np


class PolicyCommandReceiver:
    def __init__(self, bind: str, port: int):
        self.bind = str(bind)
        self.port = int(port)
        self.commands = queue.Queue()
        self._stop = threading.Event()
        self._thread = None
        self._sock = None

    def start(self):
        if self.port <= 0 or self._thread is not None:
            return

        def _loop():
            try:
                import zmq

                ctx = zmq.Context.instance()
                sock = ctx.socket(zmq.PULL)
                sock.setsockopt(zmq.LINGER, 0)
                sock.bind(f"tcp://{self.bind}:{self.port}")
                self._sock = sock
                print(f"[PolicyCmd] PULL bound tcp://{self.bind}:{self.port}")
                while not self._stop.is_set():
                    try:
                        if sock.poll(timeout=100) == 0:
                            continue
                        cmd = sock.recv_string(flags=zmq.NOBLOCK).strip().upper()
                    except zmq.Again:
                        continue
                    except Exception as exc:
                        if not self._stop.is_set():
                            print(f"[PolicyCmd][WARN] receive failed: {exc}")
                        break
                    if cmd:
                        print(f"[PolicyCmd] RX {cmd}")
                        self.commands.put(cmd)
            except Exception as exc:
                print(f"[PolicyCmd][WARN] disabled: {exc}")
            finally:
                sock = self._sock
                self._sock = None
                if sock is not None:
                    try:
                        sock.close(0)
                    except Exception:
                        pass

        self._thread = threading.Thread(target=_loop, name="PolicyCommandReceiver", daemon=True)
        self._thread.start()

    def drain(self):
        while True:
            try:
                yield self.commands.get_nowait()
            except queue.Empty:
                return

    def stop(self):
        self._stop.set()
        sock = self._sock
        if sock is not None:
            try:
                sock.close(0)
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=0.5)
            self._thread = None


class PolicyStatePublisher:
    def __init__(self, bind: str, port: int, hz: float):
        self.bind = str(bind)
        self.port = int(port)
        self.hz = float(hz)
        self._sock = None
        self._last_publish_mono = 0.0

    @property
    def enabled(self) -> bool:
        return self.port > 0 and self.hz > 0

    def start(self):
        if not self.enabled or self._sock is not None:
            return
        try:
            import zmq

            sock = zmq.Context.instance().socket(zmq.PUB)
            sock.setsockopt(zmq.LINGER, 0)
            sock.setsockopt(zmq.SNDHWM, 1)
            sock.bind(f"tcp://{self.bind}:{self.port}")
            self._sock = sock
            print(f"[PolicyState] PUB bound tcp://{self.bind}:{self.port}")
        except Exception as exc:
            self._sock = None
            print(f"[PolicyState][WARN] disabled: {exc}")

    def should_publish(self, *, force: bool = False) -> bool:
        if self._sock is None:
            return False
        if force:
            return True
        interval = 1.0 / max(self.hz, 1e-6)
        return (time.monotonic() - self._last_publish_mono) >= interval

    def publish(self, message: dict):
        if self._sock is None:
            return False
        try:
            import zmq

            self._sock.send_json(message, flags=zmq.NOBLOCK)
            self._last_publish_mono = time.monotonic()
            return True
        except zmq.Again:
            return False
        except Exception as exc:
            print(f"[PolicyState][WARN] publish failed: {exc}")
            return False

    def stop(self):
        sock = self._sock
        self._sock = None
        if sock is not None:
            try:
                sock.close(0)
            except Exception:
                pass


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_list(name: str, default: str) -> list[str]:
    value = os.environ.get(name, default)
    return [part.strip() for part in value.replace(",", " ").split() if part.strip()]


def _file_sha256(path: str) -> str:
    """Content hash of a file, or "" if unreadable.

    Used for scene provenance. A recorded XML *path* cannot tell you whether the scene was
    edited after the episode was recorded — which matters enormously for state-only
    episodes, whose images are produced by re-rendering that scene later.
    """
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return ""


def _qpos_sha256(qpos) -> str:
    values = np.ascontiguousarray(np.asarray(qpos, dtype=np.float64))
    return hashlib.sha256(values.tobytes()).hexdigest()[:16]


def _policy_session_index(cmd_port) -> int:
    """Derive this instance's session index from its (unique) command port.

    Multi-window launches give session i cmd port POLICY_CMD_PORT + i*POLICY_PORT_STEP,
    so the port is the only per-instance identity app.py already receives. Falls back to 0.
    """
    try:
        base = int(os.environ.get("POLICY_CMD_PORT", "8065"))
        step = int(os.environ.get("POLICY_PORT_STEP", "10")) or 10
        idx = (int(cmd_port) - base) // step
        return idx if 0 <= idx < 64 else 0
    except (TypeError, ValueError, ZeroDivisionError):
        return 0


def _build_policy_metrics(*, session_index: int, window_s: float, print_summary: bool):
    """Reuse the VR runtime's MetricsCollector rather than growing a second one.

    It lives in the SimPublisher submodule, which is not on the policy env's path, so add
    it here. Returns None if unavailable — metrics must never be able to stop a run.
    """
    try:
        import sys as _sys

        candidate = (
            Path(__file__).resolve().parent.parent / "SimPublisher" / "sii" / "integration_v1"
        )
        if candidate.is_dir() and str(candidate) not in _sys.path:
            _sys.path.insert(0, str(candidate))
        from perf_metrics import MetricsCollector, default_metrics_path

        out = os.environ.get("INTERVENE_METRICS_OUT", "").strip()
        if not out or out == "auto":
            out = default_metrics_path(session_index=session_index).replace(
                "metrics_S", "policy_metrics_S"
            )
        collector = MetricsCollector(
            out_path=out,
            window_s=window_s,
            print_summary=print_summary,
            session_index=session_index,
            mode="policy",
        )
        print(f"[Metrics] policy-side metrics enabled (session {session_index:02d}) -> {out}")
        return collector
    except Exception as exc:
        print(f"[Metrics][WARN] policy metrics unavailable: {exc}")
        return None


class PolicySceneRandomizer:
    def __init__(self, model, base_qpos):
        self.model = model
        self.base_qpos = np.asarray(base_qpos, dtype=np.float64).copy()
        self.enabled = _env_bool("INTERVENE_RANDOMIZE_SCENE", True)
        self.rng = np.random.default_rng(self._seed())
        self.xy_names = self._env_names(
            "INTERVENE_RANDOMIZE_OBJECT_NAMES",
            "T1,T2,cracker_box,sugar_box,cup1,cup2,cup3,cup4",
        )
        self.euler_names = self._env_names(
            "INTERVENE_RANDOMIZE_EULER_NAMES",
            "T1,T2,cracker_box,sugar_box",
        )
        self.xy_range = float(os.environ.get("INTERVENE_OBJECT_XY_RANDOM_RANGE", "0.01"))
        self.x_bounds = self._bounds("INTERVENE_OBJECT_X_BOUNDS", (0.40, 0.80))
        self.y_bounds = self._bounds("INTERVENE_OBJECT_Y_BOUNDS", (-0.25, 0.25))
        self.euler_ranges_deg = self._euler_ranges()
        self.xy_qpos_idxs = self._free_joint_qpos_idxs(self.xy_names, "xy randomize")
        self.euler_qpos_idxs = self._free_joint_qpos_idxs(self.euler_names, "euler randomize")

        # --- Out-of-distribution draws -------------------------------------------
        # An OOD episode takes its object poses from the PRE-GENERATED, contact-validated
        # corpus in mujoco_scenes/ood_scenes/ — nothing is sampled at runtime. Only the
        # poses are read; the corpus XML is never compiled, so every session keeps sharing
        # the one MjModel the desktop grid and the VR mirror both rely on, and OOD-ness can
        # still migrate between sessions at episode boundaries.
        self.ood_task_key = (
            os.environ.get("INTERVENE_OOD_TASK")
            or os.environ.get("INTERVENE_TASK_MODE")
            or "tshape"
        ).strip().lower()
        _corpus = os.environ.get("INTERVENE_OOD_SCENE_DIR", "mujoco_scenes/ood_scenes")
        # Resolve against the repo root, not the CWD: policy processes are launched from
        # varying directories.
        _corpus_path = Path(_corpus)
        if not _corpus_path.is_absolute():
            _corpus_path = Path(__file__).resolve().parent / _corpus_path
        self.ood_corpus_root = _corpus_path
        self.ood_scenes = list_ood_scenes(self.ood_task_key, self.ood_corpus_root)
        # Resolved here rather than mid-episode so a missing corpus is reported at startup.
        self.ood_available = bool(self.ood_scenes)
        self.ood_unavailable_reason = "" if self.ood_available else "corpus_unavailable"

        # Some corpora encode OOD in the MODEL (cup mesh Z-scale, swapped box textures)
        # rather than in object poses. Their pose deltas are no larger than ordinary
        # in-distribution jitter, so a pose-only draw there would produce an episode
        # labelled OOD that is not. Such a task needs a compiled model variant per episode.
        self.ood_model_variants = (
            self.ood_task_key in OOD_MODEL_VARIANT_TASKS
            and _env_bool("INTERVENE_OOD_MODEL_VARIANTS", True)
        )
        self.ood_pose_only = False
        # Identity of the geometry a descriptor is valid against. Pinned into every
        # descriptor so a consumer can refuse one built for a different scene.
        self.base_scene_sha256 = ""
        if self.ood_task_key in OOD_MODEL_VARIANT_TASKS:
            xml_path = os.environ.get("INTERVENE_MUJOCO_XML_PATH", "")
            if self.ood_model_variants and xml_path:
                try:
                    self.base_scene_sha256 = scene_closure_sha256(xml_path)
                except Exception as exc:
                    print(f"[OOD][WARN] could not hash scene closure for {xml_path}: {exc}")
            if not self.ood_model_variants:
                # Explicit opt-out. Allowed, but it must be DETECTABLE afterwards: the
                # whole reason this used to be a hard refusal is that mislabelled data
                # cannot be spotted once written.
                self.ood_pose_only = True
                if self.ood_available:
                    print(f"[OOD][WARN] task {self.ood_task_key!r}: model variants are "
                          "disabled (INTERVENE_OOD_MODEL_VARIANTS=0). OOD episodes will "
                          "differ from in-distribution ones by pose only, which for this "
                          "task is INSIDE the ordinary jitter. Episodes are stamped "
                          "episode_ood_pose_only=1.")
            elif not xml_path:
                # Without the scene path we cannot pin a descriptor to its geometry, and an
                # unpinned descriptor is exactly what the wire check exists to reject.
                print("[OOD][WARN] INTERVENE_MUJOCO_XML_PATH is unset; cannot build model "
                      "variants. Staying in-distribution.")
                self.ood_available = False
                self.ood_unavailable_reason = "no_scene_path_for_variants"

        # The five knobs below are scene HYGIENE, not the OOD signal: an object has to be
        # re-seated on the table and given zero velocity or it intersects the surface /
        # flies off at t=0. They apply to ID and OOD draws alike — the authored T pose sits
        # 17.5 mm below the table top, so this lift is load-bearing for BOTH.
        self.position_from_xml = set(self._env_names("INTERVENE_OBJECT_POSITION_FROM_XML_NAMES", ""))
        self.support_z_names = set(self._env_names("INTERVENE_OBJECT_SUPPORT_Z_NAMES", ""))
        self.support_z = float(os.environ.get("INTERVENE_OBJECT_SUPPORT_Z", "0.0") or 0.0)
        self.support_margin = float(os.environ.get("INTERVENE_OBJECT_SUPPORT_MARGIN", "0.0") or 0.0)
        self.zero_qvel_names = set(self._env_names("INTERVENE_OBJECT_ZERO_QVEL_NAMES", ""))
        self.zero_qvel_dof_idxs = self._free_joint_dof_idxs(self.zero_qvel_names)

        self._announced_disabled = False
        # Realized offsets from the most recent apply(). Previously these were printed
        # and discarded, which made a recorded scene impossible to reconstruct.
        self.last_offsets = {}
        self.last_seed = None

    def reseed(self, seed):
        """Re-seed for a specific episode so its scene is reproducible on its own.

        Without this the randomizer draws from one process-lifetime stream, so episode
        N can only be recreated by replaying episodes 1..N-1 first.
        """
        if seed is None:
            return
        self.last_seed = int(seed)
        self.rng = np.random.default_rng(self.last_seed)

    @staticmethod
    def _seed():
        value = os.environ.get("INTERVENE_OBJECT_RANDOM_SEED")
        if value is None or value.strip() == "":
            return None
        return int(value)

    @staticmethod
    def _env_names(name: str, default: str):
        return [part.strip() for part in os.environ.get(name, default).replace(",", " ").split() if part.strip()]

    @staticmethod
    def _bounds(name: str, default):
        value = os.environ.get(name)
        if not value:
            return tuple(default)
        parts = [float(part) for part in value.replace(",", " ").split()]
        if len(parts) != 2:
            raise ValueError(f"{name} must contain two numbers, got {value!r}")
        return (parts[0], parts[1])

    @staticmethod
    def _euler_ranges():
        default = "0 0 5"
        parts = [float(part) for part in os.environ.get("INTERVENE_OBJECT_EULER_RANDOM_RANGE_DEG", default).replace(",", " ").split()]
        if len(parts) != 3:
            raise ValueError("INTERVENE_OBJECT_EULER_RANDOM_RANGE_DEG must contain roll pitch yaw degrees")
        return np.radians(parts)


    def _free_joint_dof_idxs(self, body_names):
        dof_idxs = {}
        for body_name in body_names:
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{body_name}_free"
            )
            if joint_id < 0:
                continue
            if self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
                continue
            dof_idxs[body_name] = int(self.model.jnt_dofadr[joint_id])
        return dof_idxs


    def _free_joint_qpos_idxs(self, body_names, label):
        qpos_idxs = {}
        for body_name in body_names:
            joint_name = f"{body_name}_free"
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                continue
            joint_type = self.model.jnt_type[joint_id]
            if joint_type != mujoco.mjtJoint.mjJNT_FREE:
                print(f"[RANDOMIZE][WARN] Cannot {label} {body_name}: joint {joint_name!r} is not free.")
                continue
            qpos_idxs[body_name] = int(self.model.jnt_qposadr[joint_id])
        return qpos_idxs

    @staticmethod
    def _mj_euler_to_quat(euler):
        """Euler -> quaternion in MuJoCo's OWN convention (`eulerseq="xyz"`, intrinsic).

        Required for anything read out of a scene XML. `_euler_xyz_to_quat` below is the
        aerospace RPY (ZYX) composition and is NOT the same rotation: for T1's authored
        `euler="1.5708 3.14159 0"` the compiler gives w=+0.707108 and that function gives
        w=-0.707108. They agree only when at most one Euler component is nonzero.
        """
        q = np.zeros(4, dtype=np.float64)
        mujoco.mju_euler2Quat(q, np.asarray(euler, dtype=np.float64), "xyz")
        return q

    @staticmethod
    def _euler_xyz_to_quat(roll, pitch, yaw):
        """RPY/ZYX composition. NOT MuJoCo's `eulerseq="xyz"` — see `_mj_euler_to_quat`.

        Kept, and kept in use by the in-distribution jitter path only, because the default
        `INTERVENE_OBJECT_EULER_RANDOM_RANGE_DEG` is `0 0 5` (yaw only), where the two
        conventions coincide. Changing it would alter every previously recorded ID scene.
        Do not use it for multi-axis angles.
        """
        cr, sr = np.cos(0.5 * roll), np.sin(0.5 * roll)
        cp, sp = np.cos(0.5 * pitch), np.sin(0.5 * pitch)
        cy, sy = np.cos(0.5 * yaw), np.sin(0.5 * yaw)
        return np.array(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _quat_mul(q1, q2):
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float64,
        )

    def _read_scene_variant(self, scene_path, task_dir):
        """`(descriptor, provenance)` for a corpus scene, or `(None, {})`. Never raises."""
        try:
            return ood_scene_variant(
                scene_path, self.ood_task_key, task_dir,
                base_scene_sha256=self.base_scene_sha256,
            )
        except Exception as exc:
            print(f"[OOD][WARN] variant read failed for {scene_path.name}: {exc}")
            return None, {}

    def variant_for_scene_index(self, idx):
        """The descriptor a given corpus index would produce, without drawing from `self.rng`.

        Used by the speculative precompiler so the next episode's variant can be built while
        the current one runs. MUST NOT touch `self.rng` or `self.last_offsets`.
        """
        if not self.ood_model_variants or not self.ood_available:
            return None
        if not (0 <= idx < len(self.ood_scenes)):
            return None
        task_dir = self.ood_corpus_root / OOD_TASK_DIRS.get(self.ood_task_key, "")
        descriptor, _ = self._read_scene_variant(self.ood_scenes[idx], task_dir)
        return descriptor

    def peek_next_ood_scene_indices(self, count: int = 1):
        """The next `count` OOD corpus indices this randomizer WOULD draw, WITHOUT consuming.

        Outside study mode `self.rng` is never re-seeded -- `reseed()` is called only from
        `_start_new_randomized_scene` under `if self.study_block is not None` -- so the
        stream is continuous and the next draw is a pure function of the CURRENT generator
        state. Clone that state into a throwaway generator and draw from the clone.

        This is what lets the speculative precompiler work outside study mode, which is the
        configuration the operator actually runs. Without it every OOD episode boundary pays
        a ~1.3 s blocking compile in the policy process and again in the VR runtime.

        MUST NOT touch `self.rng`. an internal regression test (not part of this release) guarded this: it pinned
        the in-distribution draw byte-for-byte over 1000 seeds.
        """
        if not self.ood_available or not self.ood_scenes:
            return []
        try:
            state = copy.deepcopy(self.rng.bit_generator.state)
            clone = np.random.default_rng()
            if clone.bit_generator.state.get("bit_generator") != state.get("bit_generator"):
                # A different default bit generator: refuse rather than mispredict.
                return []
            clone.bit_generator.state = state
            n = len(self.ood_scenes)
            return [int(clone.integers(n)) for _ in range(max(1, int(count)))]
        except Exception as exc:
            print(f"[Variant][WARN] could not peek the next OOD index: {exc}")
            return []

    def rebind_model(self, model):
        """Point the randomizer at a different compiled model.

        Only name->index maps are rebuilt. `self.rng`, `self.last_seed`, `self.base_qpos`
        and `self.enabled` are deliberately untouched: this is the single highest-risk
        function in the model-variant change for the in-distribution bit-identity
        guarantee, and an internal regression test checked that it holds.
        """
        self.model = model
        self.xy_qpos_idxs = self._free_joint_qpos_idxs(self.xy_names, "xy randomize")
        self.euler_qpos_idxs = self._free_joint_qpos_idxs(self.euler_names, "euler randomize")
        self.zero_qvel_dof_idxs = self._free_joint_dof_idxs(self.zero_qvel_names)

    def _apply_ood_scene(self, qpos) -> bool:
        """Write one pre-generated corpus scene's poses into `qpos`. True if applied.

        No jitter is layered on top: the corpus pose IS the scene, and it was
        contact-validated offline at exactly that pose. Perturbing it would invalidate
        that validation.
        """
        if not self.ood_available:
            self.last_offsets["ood_fallback_reason"] = (
                self.ood_unavailable_reason or "corpus_unavailable"
            )
            return False

        idx = int(self.rng.integers(len(self.ood_scenes)))
        scene_path = self.ood_scenes[idx]
        task_dir = self.ood_corpus_root / OOD_TASK_DIRS.get(self.ood_task_key, "")
        try:
            poses = ood_scene_poses(scene_path, task_dir)
        except Exception as exc:
            print(f"[OOD][WARN] could not read {scene_path.name}: {exc}; staying in-distribution.")
            self.last_offsets["ood_fallback_reason"] = "scene_unreadable"
            return False

        written = {}
        for body_name, qpos_idx in {**self.xy_qpos_idxs, **self.euler_qpos_idxs}.items():
            pose = poses.get(body_name)
            if pose is None:
                continue
            pos, euler = pose
            quat = self._mj_euler_to_quat(euler)
            norm = np.linalg.norm(quat)
            if norm <= 0.0:
                continue
            qpos[qpos_idx:qpos_idx + 3] = pos
            qpos[qpos_idx + 3:qpos_idx + 7] = quat / norm
            written[body_name] = [float(v) for v in pos]

        if not written:
            # The corpus scene named none of this model's randomizable bodies — a renamed
            # body or the wrong task's corpus. Do not silently label an ID scene as OOD.
            print(f"[OOD][WARN] {scene_path.name} matched no randomizable body "
                  f"({sorted({**self.xy_qpos_idxs, **self.euler_qpos_idxs})}); staying in-distribution.")
            self.last_offsets["ood_fallback_reason"] = "no_matching_bodies"
            return False

        self.last_offsets.update({
            "ood_scene_file": scene_path.name,
            "ood_scene_index": idx,
            "ood_scene_sha256": _file_sha256(scene_path),
            "ood_corpus_root": str(self.ood_corpus_root),
            "ood_task_key": self.ood_task_key,
            "ood_bodies": written,
        })

        # For a model-level task, the poses above are only half the scene. Read the mesh /
        # texture parameters too and hand them up as a descriptor; `App._swap_model` turns
        # that into a compiled variant at the episode boundary. Returns None for every
        # pose-only task, which is what keeps t_shape on exactly this code path unchanged.
        if self.ood_model_variants:
            descriptor, provenance = self._read_scene_variant(scene_path, task_dir)
            if descriptor is None:
                # A refused scene must not become a mislabelled ID episode.
                print(f"[OOD][WARN] {scene_path.name}: model variant unreadable; "
                      "staying in-distribution.")
                self.last_offsets["ood_fallback_reason"] = "variant_unreadable"
                return False
            self.last_offsets["ood_model_variant"] = descriptor
            self.last_offsets["ood_model_variant_key"] = variant_key(descriptor)
            self.last_offsets["ood_model_variant_provenance"] = provenance
        print(f"[RANDOMIZE][OOD] scene {scene_path.name} "
              f"({idx + 1}/{len(self.ood_scenes)}): " +
              ", ".join(f"{b}=({p[0]:.3f}, {p[1]:.3f})" for b, p in sorted(written.items())))
        return True

    def apply(self, qpos, *, ood: bool = False):
        """Draw a scene.

        `ood=False` jitters the authored pose slightly — the in-distribution condition.
        `ood=True` takes the pose wholesale from a pre-generated, contact-validated corpus
        scene; nothing is sampled. Either way only `qpos` changes, so the one MjModel the
        grid viewer and VR mirror share still matches every session.
        """
        qpos = np.asarray(qpos, dtype=np.float64).copy()
        ood = bool(ood)
        self.last_offsets = {"seed": self.last_seed, "enabled": bool(self.enabled),
                             "ood": ood, "xy": {}, "euler_deg": {}}
        if not self.enabled:
            return qpos
        if not self.xy_qpos_idxs and not self.euler_qpos_idxs:
            if not self._announced_disabled:
                print("[RANDOMIZE] No matching free joints found in this XML; scene randomization is inactive.")
                self._announced_disabled = True
            return qpos

        if ood:
            # Guarded so that when ood=False NOTHING extra is drawn from self.rng — that
            # is what keeps the in-distribution stream bit-identical to before.
            if self._apply_ood_scene(qpos):
                # Same hygiene pass the ID path gets: the corpus deliberately allows the
                # object to intersect `table_top`, so it must be reseated. Mutates in place.
                self._reseat_supported_geometry(qpos)
                return qpos
            # corpus unavailable -> fall through to the in-distribution draw below

        xy_offsets = []
        for body_name, qpos_idx in self.xy_qpos_idxs.items():
            xy_offset = self.rng.uniform(-self.xy_range, self.xy_range, size=2)
            base_xy = self._base_xy(body_name, qpos_idx)
            randomized_xy = np.array(
                [
                    np.clip(base_xy[0] + xy_offset[0], self.x_bounds[0], self.x_bounds[1]),
                    np.clip(base_xy[1] + xy_offset[1], self.y_bounds[0], self.y_bounds[1]),
                ],
                dtype=np.float64,
            )
            qpos[qpos_idx:qpos_idx + 2] = randomized_xy
            xy_offsets.append(f"{body_name}=({randomized_xy[0]:.3f}, {randomized_xy[1]:.3f})")
            self.last_offsets["xy"][body_name] = [float(randomized_xy[0]),
                                                  float(randomized_xy[1])]

        euler_offsets = []
        for body_name, qpos_idx in self.euler_qpos_idxs.items():
            euler_offset = self.rng.uniform(-self.euler_ranges_deg, self.euler_ranges_deg)
            delta_quat = self._euler_xyz_to_quat(*euler_offset)
            base_quat = self.base_qpos[qpos_idx + 3:qpos_idx + 7]
            randomized_quat = self._quat_mul(delta_quat, base_quat)
            euler_offsets.append(
                f"{body_name}=({np.degrees(euler_offset[0]):+.1f}, "
                f"{np.degrees(euler_offset[1]):+.1f}, {np.degrees(euler_offset[2]):+.1f})deg"
            )
            self.last_offsets["euler_deg"][body_name] = [
                float(np.degrees(v)) for v in euler_offset
            ]
            randomized_quat = randomized_quat / np.linalg.norm(randomized_quat)
            qpos[qpos_idx + 3:qpos_idx + 7] = randomized_quat

        # Scene hygiene, applied to ID and OOD alike. Reseat from the transformed
        # geometry rather than assuming a fixed root height; this is what keeps a
        # rotated T supported without burying one of its bars in the tabletop.
        self._reseat_supported_geometry(qpos)

        if xy_offsets:
            print("[RANDOMIZE] object xy:", ", ".join(xy_offsets))
        if euler_offsets:
            print("[RANDOMIZE] object euler:", ", ".join(euler_offsets))
        return qpos

    def _base_xy(self, body_name, qpos_idx):
        """Base XY for a body: from the XML body_pos when asked, else from base_qpos.

        The RESET_NPZ that base_qpos comes from is a recorded demo frame, so its object
        pose can be mid-manipulation rather than the scene's authored rest pose.
        """
        if body_name in self.position_from_xml:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id >= 0:
                return np.asarray(self.model.body_pos[body_id][:2], dtype=np.float64)
        return self.base_qpos[qpos_idx:qpos_idx + 2]

    def _body_qpos_idx(self, body_name):
        qpos_idx = self.xy_qpos_idxs.get(body_name)
        if qpos_idx is None:
            qpos_idx = self.euler_qpos_idxs.get(body_name)
        return qpos_idx

    def _body_geom_ids(self, body_name):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            return body_id, []
        return body_id, [
            int(geom_id)
            for geom_id, geom_body_id in enumerate(self.model.geom_bodyid)
            if int(geom_body_id) == body_id
        ]

    def _geom_vertical_extent(self, geom_id, rotation):
        """Return the world-Z half extent for the supported MuJoCo primitive."""
        geom_type = self.model.geom_type[geom_id]
        size = np.asarray(self.model.geom_size[geom_id], dtype=np.float64)
        z_axis = np.abs(np.asarray(rotation, dtype=np.float64)[2, :])
        if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            return float(np.dot(z_axis, size[:3]))
        if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            return float(size[0])
        if geom_type in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
            return float(z_axis[2] * size[1] + np.hypot(z_axis[0], z_axis[1]) * size[0])
        if geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            return float(np.dot(z_axis, size[:3]))
        # T-shape scenes use boxes. For an unsupported primitive, use its center as a
        # conservative fallback; validation below will still reject buried geometry.
        return 0.0

    def _geom_low_z(self, data, geom_id):
        rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        return float(data.geom_xpos[geom_id][2] - self._geom_vertical_extent(geom_id, rotation))

    def _reseat_supported_geometry(self, qpos):
        """Lift configured bodies until their actual transformed geometry is supported."""
        if not self.support_z_names:
            return
        floor = self.support_z + self.support_margin
        data = mujoco.MjData(self.model)
        for body_name in self.support_z_names:
            qpos_idx = self._body_qpos_idx(body_name)
            if qpos_idx is None:
                continue
            data.qpos[:] = qpos
            mujoco.mj_forward(self.model, data)
            _, geom_ids = self._body_geom_ids(body_name)
            if not geom_ids:
                continue
            lowest = min(self._geom_low_z(data, geom_id) for geom_id in geom_ids)
            if lowest < floor:
                qpos[qpos_idx + 2] += floor - lowest

    def validate_tshape_scene(self, qpos, *, tolerance=0.005):
        """Validate a candidate T-shape reset using transformed MuJoCo geometry.

        The reset must be valid but unsolved: both colored T objects remain present,
        tabletop-supported, flat, inside the table, and separate from one another.
        Models for other tasks are intentionally treated as not applicable.
        """
        if os.environ.get("INTERVENE_TASK_MODE", "tshape").strip().lower() != "tshape":
            return True, ()

        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.shape != (self.model.nq,) or not np.all(np.isfinite(qpos)):
            return False, ("qpos_not_finite_or_wrong_shape",)

        required = {"T1": ("T1_stem", "T1_bar"), "T2": ("T2_stem", "T2_bar")}
        body_geoms = {}
        reasons = []
        for body_name, geom_names in required.items():
            body_id, geom_ids = self._body_geom_ids(body_name)
            if body_id < 0:
                reasons.append(f"missing_body:{body_name}")
                continue
            named = {
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id): geom_id
                for geom_id in geom_ids
            }
            if any(name not in named for name in geom_names):
                reasons.append(f"missing_geom:{body_name}")
            body_geoms[body_name] = geom_ids
        if reasons:
            return False, tuple(reasons)

        table_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
        if table_id < 0:
            return False, ("missing_table_top",)

        data = mujoco.MjData(self.model)
        data.qpos[:] = qpos
        mujoco.mj_forward(self.model, data)
        if not np.all(np.isfinite(data.geom_xpos)) or not np.all(np.isfinite(data.geom_xmat)):
            return False, ("transformed_geometry_not_finite",)

        table_rotation = np.asarray(data.geom_xmat[table_id], dtype=np.float64).reshape(3, 3)
        table_center = np.asarray(data.geom_xpos[table_id], dtype=np.float64)
        table_extent = self._geom_vertical_extent(table_id, table_rotation)
        table_size = np.asarray(self.model.geom_size[table_id], dtype=np.float64)
        table_x_half = float(np.dot(np.abs(table_rotation[0, :]), table_size[:3]))
        table_y_half = float(np.dot(np.abs(table_rotation[1, :]), table_size[:3]))
        table_top_z = float(table_center[2] + table_extent)

        for body_name, geom_ids in body_geoms.items():
            for geom_id in geom_ids:
                rgba = np.asarray(self.model.geom_rgba[geom_id], dtype=np.float64)
                size = np.asarray(self.model.geom_size[geom_id], dtype=np.float64)
                if rgba[3] <= 0.0 or not np.any(size > 0.0):
                    reasons.append(f"invisible_geom:{body_name}")
                rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
                # In the authored T XML local Y is the thin/table-normal axis. A world-Z
                # rotation must leave that axis vertical; this catches accidental local
                # pitch/roll while allowing any OOD table-plane angle.
                if abs(float(rotation[2, 1])) < 0.95:
                    reasons.append(f"not_tabletop_flat:{body_name}")
                low = self._geom_low_z(data, geom_id)
                high = float(data.geom_xpos[geom_id][2] + self._geom_vertical_extent(geom_id, rotation))
                x = float(data.geom_xpos[geom_id][0])
                y = float(data.geom_xpos[geom_id][1])
                x_half = float(np.dot(np.abs(rotation[0, :]), size[:3]))
                y_half = float(np.dot(np.abs(rotation[1, :]), size[:3]))
                if x - x_half < table_center[0] - table_x_half - tolerance or \
                   x + x_half > table_center[0] + table_x_half + tolerance or \
                   y - y_half < table_center[1] - table_y_half - tolerance or \
                   y + y_half > table_center[1] + table_y_half + tolerance:
                    reasons.append(f"outside_table_workspace:{body_name}")
                if low < table_top_z - tolerance or high <= table_top_z - tolerance:
                    reasons.append(f"below_tabletop:{body_name}")

        t1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "T1")
        t2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "T2")
        for contact_idx in range(int(data.ncon)):
            contact = data.contact[contact_idx]
            body1 = int(self.model.geom_bodyid[int(contact.geom1)])
            body2 = int(self.model.geom_bodyid[int(contact.geom2)])
            if {body1, body2} == {t1_id, t2_id} and float(contact.dist) < -tolerance:
                reasons.append("t_objects_intersect")
                break

        return not reasons, tuple(dict.fromkeys(reasons))

    def apply_qvel(self, qvel):
        """Zero the listed bodies' free-joint velocities.

        base_qvel comes from a recorded demo frame too, so without this an object can
        start the episode already moving and drift out of the scene at t=0.
        """
        qvel = np.asarray(qvel, dtype=np.float64).copy()
        for dof_idx in self.zero_qvel_dof_idxs.values():
            qvel[dof_idx:dof_idx + 6] = 0.0
        return qvel


class LiveTaskEvaluator:
    def __init__(self, model, data, player):
        self.model = model
        self.data = data
        self.player = player
        self.enabled = _env_bool("INTERVENE_AUTO_TASK_EVAL", True)
        self.task_mode = os.environ.get("INTERVENE_TASK_MODE", "tshape").strip().lower()
        self.body_name = os.environ.get("INTERVENE_TASK_BODY_NAME", "T1")
        self.target_body_name = os.environ.get("INTERVENE_TASK_TARGET_BODY_NAME", "T2")
        self.tshape_lifted_bar_geom = os.environ.get("INTERVENE_TASK_TSHAPE_LIFTED_BAR_GEOM", "T1_bar")
        self.tshape_lifted_stem_geom = os.environ.get("INTERVENE_TASK_TSHAPE_LIFTED_STEM_GEOM", "T1_stem")
        self.tshape_target_bar_geom = os.environ.get("INTERVENE_TASK_TSHAPE_TARGET_BAR_GEOM", "T2_bar")
        self.tshape_target_stem_geom = os.environ.get("INTERVENE_TASK_TSHAPE_TARGET_STEM_GEOM", "T2_stem")
        # Loosened 2026-08-12 for human-in-the-loop placement (0.60 -> 0.40 overlap,
        # 15 -> 25 deg axis). The old values were calibrated on the policy, which was
        # trained to the demonstrated yaw; a human placing through the MC controller's
        # IK cannot see the bar's yaw error and lands 13-16 deg off. Measured on two real
        # takeovers (policy_episode_20260812_131931_240 / _132043_006): every other
        # criterion PASSED -- bar-bar contact, face_gap ~0, face and stem alignment 1.00,
        # stem height 0.058 -- and both were scored a failure on this axis alone
        # (overlap 0.559 vs 0.60; axis 15.68 deg vs 15.0). Keep in step with the offline
        # evaluator's argparse defaults (utils/evaluate_act_mujoco.py) or the same episode
        # scores differently live and in analysis.
        self.tshape_min_bar_overlap = float(os.environ.get("INTERVENE_TASK_TSHAPE_MIN_BAR_OVERLAP", "0.40"))
        self.tshape_max_bar_axis_error_deg = float(
            os.environ.get("INTERVENE_TASK_TSHAPE_MAX_BAR_AXIS_ERROR_DEG", "25.0")
        )
        self.tshape_min_face_alignment = float(
            os.environ.get("INTERVENE_TASK_TSHAPE_MIN_FACE_ALIGNMENT", "0.85")
        )
        self.tshape_min_stem_up_alignment = float(
            os.environ.get("INTERVENE_TASK_TSHAPE_MIN_STEM_UP_ALIGNMENT", "0.75")
        )
        self.tshape_min_stem_height = float(os.environ.get("INTERVENE_TASK_TSHAPE_MIN_STEM_HEIGHT", "0.03"))
        self.tshape_max_face_gap = float(os.environ.get("INTERVENE_TASK_TSHAPE_MAX_FACE_GAP", "0.025"))
        self.tshape_min_target_top_alignment = float(
            os.environ.get("INTERVENE_TASK_TSHAPE_MIN_TARGET_TOP_ALIGNMENT", "0.75")
        )
        self.tshape_require_bar_contact = _env_bool("INTERVENE_TASK_TSHAPE_REQUIRE_BAR_CONTACT", True)
        self.tshape_forbid_stem_contact = _env_bool("INTERVENE_TASK_TSHAPE_FORBID_STEM_CONTACT", True)
        self.close_threshold = float(os.environ.get("INTERVENE_TASK_CLOSE_THRESHOLD", "0.02"))
        self.lift_threshold = float(os.environ.get("INTERVENE_TASK_LIFT_THRESHOLD", "0.08"))
        self.placement_xy_threshold = float(os.environ.get("INTERVENE_TASK_PLACEMENT_XY_THRESHOLD", "0.04"))
        self.placement_target_local_offset = self._vec3(
            "INTERVENE_TASK_PLACEMENT_TARGET_LOCAL_OFFSET",
            (0.0, 0.0, 0.0),
        )
        self.placement_z_offset = float(os.environ.get("INTERVENE_TASK_PLACEMENT_Z_OFFSET", "0.165"))
        self.placement_z_tolerance = float(os.environ.get("INTERVENE_TASK_PLACEMENT_Z_TOLERANCE", "0.025"))
        self.placement_yaw_threshold_deg = float(os.environ.get("INTERVENE_TASK_PLACEMENT_YAW_THRESHOLD_DEG", "25.0"))
        self.placement_gripper_open_threshold = float(os.environ.get("INTERVENE_TASK_GRIPPER_OPEN_THRESHOLD", "0.025"))
        self.placement_stable_steps = int(os.environ.get("INTERVENE_TASK_PLACEMENT_STABLE_STEPS", "10"))
        self.fail_at_reset_len = _env_bool("INTERVENE_AUTO_FAIL_AT_RESET_LEN", True)
        # End the episode as soon as an object leaves the table. Before this, a dropped
        # object was not terminal at all: `decide()` could only return success, the
        # (disabled) wall-clock timeout, or the step limit -- so an unrecoverable scene ran
        # out its full 127/217 steps before failing. Set INTERVENE_TASK_FAIL_ON_DROP=0 to
        # restore that. Only ever consulted from `decide()`, which runs in replay mode
        # only, so a drop DURING a takeover is never terminal -- see _dropped_object.
        # --- Episode outcome timing -------------------------------------------------
        # THE RULE (both labs): the END STATE decides.
        #   * SUCCESS is declared the instant the correct configuration is reached -- the
        #     episode does not run on to the cutoff once the task is done.
        #   * FAILURE is declared ONLY at the cutoff (the step limit). Nothing ends an
        #     episode early as a failure, because a scene that looks unrecoverable now can
        #     still be recovered -- by the policy, or by a human takeover -- before the
        #     cutoff, and judging it early would record an outcome the end state
        #     contradicts.
        #
        # The drop and topple detectors below therefore run for their REASON only: they
        # make the eventual cutoff failure say `object_dropped:cup1` instead of a generic
        # geometric miss. The `fail_early_*` switches turn them into terminal conditions
        # and both default OFF; they exist because an operator doing throughput runs
        # rather than a study may not want to watch dead time.
        self.fail_early_on_drop = _env_bool("INTERVENE_TASK_FAIL_EARLY_ON_DROP", False)
        self.drop_margin_m = float(os.environ.get("INTERVENE_TASK_DROP_MARGIN_M", "0.10"))
        self.initial_object_z = {}
        # Persistence, unlike the drop test, because a cup can legitimately exceed 35 deg
        # for a few frames WHILE BEING CARRIED. Requiring N consecutive steps separates
        # "toppled on the table" from "tilted in the gripper".
        self.fail_early_on_topple = _env_bool("INTERVENE_TASK_FAIL_EARLY_ON_TOPPLE", False)
        self.scene_invalid_steps = int(os.environ.get("INTERVENE_TASK_SCENE_INVALID_STEPS", "10"))
        self._scene_invalid_streak = 0
        self._scene_invalid_reason = ""
        self.timeout_seconds = float(os.environ.get("INTERVENE_TASK_TIMEOUT_SECONDS", "0"))
        self.cup_pairs = self._cup_pairs(os.environ.get("INTERVENE_TASK_CUP_PAIRS", "cup1:cup2,cup3:cup4"))
        self.cup_body_names = self._names(os.environ.get("INTERVENE_TASK_CUP_BODY_NAMES", "cup1 cup2 cup3 cup4"))
        self.box_body_names = self._names(os.environ.get("INTERVENE_TASK_BOX_BODY_NAMES", "cracker_box sugar_box"))
        self.box_upright_angle_deg = float(os.environ.get("INTERVENE_TASK_BOX_UPRIGHT_ANGLE_DEG", "20"))
        self.cup_upright_angle_deg = float(os.environ.get("INTERVENE_TASK_CUP_UPRIGHT_ANGLE_DEG", "35"))
        self.cups_require_contact = _env_bool("INTERVENE_TASK_CUPS_REQUIRE_CONTACT", True)
        self.cups_use_advanced_metric = _env_bool("INTERVENE_TASK_CUPS_USE_ADVANCED_METRIC", True)
        self.cups_gripper_open_threshold = float(os.environ.get("INTERVENE_TASK_CUPS_GRIPPER_OPEN_THRESHOLD", "0"))
        self.cups_max_axis_error_deg = float(os.environ.get("INTERVENE_TASK_CUPS_MAX_AXIS_ERROR_DEG", "20"))
        self.cups_min_radial_margin = float(os.environ.get("INTERVENE_TASK_CUPS_MIN_RADIAL_MARGIN", "-0.005"))
        self.cups_forbid_upper_bad_contacts = _env_bool("INTERVENE_TASK_CUPS_FORBID_UPPER_BAD_CONTACTS", False)
        self.cups_forbid_upper_wrong_contacts = _env_bool("INTERVENE_TASK_CUPS_FORBID_UPPER_WRONG_CONTACTS", False)
        self.cups_bad_contact_body_names = self._names(
            os.environ.get("INTERVENE_TASK_CUPS_BAD_CONTACT_BODY_NAMES", "table cracker_box sugar_box")
        )
        self.cups_max_pair_linear_speed = float(os.environ.get("INTERVENE_TASK_CUPS_MAX_PAIR_LINEAR_SPEED", "0"))
        self.cups_max_pair_angular_speed = float(os.environ.get("INTERVENE_TASK_CUPS_MAX_PAIR_ANGULAR_SPEED", "0"))
        self.available = self.enabled and self._required_bodies_exist()
        self.reset()

    @staticmethod
    def _names(value):
        return [part.strip() for part in value.replace(",", " ").split() if part.strip()]

    @staticmethod
    def _vec3(name, default):
        value = os.environ.get(name)
        if not value:
            return tuple(default)
        parts = [float(part) for part in value.replace(",", " ").split()]
        if len(parts) != 3:
            raise ValueError(f"{name} must contain three numbers")
        return tuple(parts)

    @staticmethod
    def _cup_pairs(value):
        pairs = []
        for raw_pair in value.replace(";", ",").split(","):
            raw_pair = raw_pair.strip()
            if not raw_pair:
                continue
            cup, target = [part.strip() for part in raw_pair.split(":", 1)]
            pairs.append((cup, target))
        return pairs

    def _body_exists(self, body_name):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name) >= 0

    def _tracked_body_names(self):
        """Objects whose fall off the table should end the episode."""
        if self.task_mode == "cups":
            return [*self.cup_body_names, *self.box_body_names]
        if self.task_mode == "tshape":
            return [self.body_name, self.target_body_name]
        return [n for n in (self.body_name, self.target_body_name) if n]

    def _note_scene_invalid(self, reason):
        """Advance/clear the toppled-object streak. Returns a reason once it persists."""
        if not reason:
            self._scene_invalid_streak = 0
            self._scene_invalid_reason = ""
            return None
        self._scene_invalid_streak += 1
        self._scene_invalid_reason = reason
        return reason if self._scene_invalid_streak >= max(1, self.scene_invalid_steps) else None

    def _dropped_object(self):
        """Name of the first object that has fallen, or None.

        Height only, measured against each object's OWN rest height at reset. Measured on
        both labs: every object rests at z=0.225 (the table top) and every legitimate task
        motion is UPWARD -- T1 goes on top of T2, a cup goes into another cup -- so there
        is no downward displacement to confuse this with. The table top is 0.225 m above
        the floor, so the 0.10 m default cannot be reached without leaving the table.

        Deliberately NOT the existing `boxes_ok`/`cups_ok`: those are upright-ANGLE tests.
        Verified that a cup lowered 2 m below the floor while staying upright still reports
        cups_ok=True and max_cup_angle=0.0deg -- an object can leave the table entirely
        without tipping, and nothing in the evaluator noticed.
        """
        if not self.initial_object_z:
            return None
        for name, z0 in self.initial_object_z.items():
            if not self._body_exists(name):
                continue
            if float(self._body_pos(name)[2]) < z0 - self.drop_margin_m:
                return name
        return None

    def _geom_exists(self, geom_name):
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name) >= 0

    def _required_bodies_exist(self):
        if not self.enabled:
            return False
        if self.task_mode == "cups":
            names = [*self.cup_body_names, *self.box_body_names]
            for cup, target in self.cup_pairs:
                names.extend([cup, target])
            missing = sorted({name for name in names if not self._body_exists(name)})
            if missing:
                print(f"[TASK] Auto evaluator disabled; missing body/bodies: {', '.join(missing)}")
                return False
            return True
        if self.task_mode == "tshape":
            body_names = [self.body_name, self.target_body_name]
            geom_names = [
                self.tshape_lifted_bar_geom,
                self.tshape_lifted_stem_geom,
                self.tshape_target_bar_geom,
                self.tshape_target_stem_geom,
            ]
            missing_bodies = sorted({name for name in body_names if not self._body_exists(name)})
            missing_geoms = sorted({name for name in geom_names if not self._geom_exists(name)})
            if missing_bodies:
                print(f"[TASK] Auto evaluator disabled; missing body/bodies: {', '.join(missing_bodies)}")
                return False
            if missing_geoms:
                print(f"[TASK] Auto evaluator disabled; missing geom/geoms: {', '.join(missing_geoms)}")
                return False
            return True
        else:
            names = [self.body_name, self.target_body_name]
        missing = sorted({name for name in names if not self._body_exists(name)})
        if missing:
            print(f"[TASK] Auto evaluator disabled; missing body/bodies: {', '.join(missing)}")
            return False
        return True

    def rebind(self, model, data):
        """Point the evaluator at a different compiled model + its new MjData.

        Nothing else to rebuild: every body and geom is resolved by name at call time
        (`_body_exists`, `_geom_exists`, `_body_pos`), so there are no cached ids to
        invalidate. The caller calls `reset()` right after.
        """
        self.model = model
        self.data = data

    def reset(self):
        self.initial_body_pos = None
        if self.enabled and self._body_exists(self.body_name):
            self.initial_body_pos = self._body_pos(self.body_name)
        # Per-object rest height, captured AFTER the scene is randomized, so the drop test
        # is measured against where this episode actually started rather than a constant.
        self.initial_object_z = {}
        if self.enabled:
            for name in self._tracked_body_names():
                if self._body_exists(name):
                    self.initial_object_z[name] = float(self._body_pos(name)[2])
        self.first_close_step = None
        self.first_lift_step = None
        self.first_close_wall = None
        self.first_lift_wall = None
        self.latched_during_intervention = False
        self.contact_streak = 0
        self._scene_invalid_streak = 0
        self._scene_invalid_reason = ""
        self.last_reason = ""
        self.start_wall_t = time.time()
        self.paused_wall_total = 0.0
        self._pause_started_wall = None
        print("[TASK] Auto evaluator reset.")

    # --- task clock -------------------------------------------------------------
    # The timeout must measure how long the TASK has been running, not how long the
    # human has been holding the arm. An intervention can legitimately take minutes;
    # charging that against INTERVENE_TASK_TIMEOUT_SECONDS would fail every episode a
    # human touched. Both calls are idempotent so double-invocation is harmless.

    def pause_clock(self):
        if getattr(self, "_pause_started_wall", None) is None:
            self._pause_started_wall = time.time()

    def resume_clock(self):
        started = getattr(self, "_pause_started_wall", None)
        if started is None:
            return
        self.paused_wall_total = getattr(self, "paused_wall_total", 0.0) + (
            time.time() - float(started)
        )
        self._pause_started_wall = None

    def elapsed(self):
        now = time.time()
        paused = getattr(self, "paused_wall_total", 0.0)
        started = getattr(self, "_pause_started_wall", None)
        if started is not None:
            paused += now - float(started)
        return (now - self.start_wall_t) - paused

    def snapshot_latches(self):
        """Capture the latched progress so a CANCELLED takeover can undo it.

        cancel_replan rolls the world back to the pre-intervention snapshot, so latches
        set during that takeover would describe events that no longer happened — and a
        stale first_lift_step is enough to make a later marginal pose score a success.
        """
        initial = self.initial_body_pos
        return {
            "initial_body_pos": None if initial is None else np.array(initial, copy=True),
            "first_close_step": self.first_close_step,
            "first_lift_step": self.first_lift_step,
            "first_close_wall": getattr(self, "first_close_wall", None),
            "first_lift_wall": getattr(self, "first_lift_wall", None),
            "latched_during_intervention": getattr(self, "latched_during_intervention", False),
            "contact_streak": self.contact_streak,
            "last_reason": self.last_reason,
            "paused_wall_total": getattr(self, "paused_wall_total", 0.0),
        }

    def restore_latches(self, state):
        if not state:
            return
        self.initial_body_pos = state.get("initial_body_pos")
        self.first_close_step = state.get("first_close_step")
        self.first_lift_step = state.get("first_lift_step")
        self.first_close_wall = state.get("first_close_wall")
        self.first_lift_wall = state.get("first_lift_wall")
        self.latched_during_intervention = bool(state.get("latched_during_intervention", False))
        self.contact_streak = int(state.get("contact_streak", 0))
        self.last_reason = state.get("last_reason", "")
        self.paused_wall_total = float(state.get("paused_wall_total", 0.0))

    def _body_pos(self, body_name):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        return self.data.xpos[body_id].copy()

    def _reached_episode_limit(self):
        frame_idx = int(getattr(self.player, "frame_idx", 0))
        max_steps = int(getattr(getattr(self.player, "config", None), "max_steps", 0) or 0)
        if max_steps > 0 and frame_idx >= max_steps:
            return True
        reset_len = int(getattr(self.player, "reset_episode_len", 0) or 0)
        return self.fail_at_reset_len and reset_len > 0 and frame_idx >= reset_len

    def _latch_close(self, frame_idx, during_intervention):
        if self.first_close_step is not None:
            return
        self.first_close_step = frame_idx
        self.first_close_wall = time.time()
        if during_intervention:
            self.latched_during_intervention = True

    def _latch_lift(self, frame_idx, during_intervention):
        if self.first_lift_step is not None:
            return
        self.first_lift_step = frame_idx
        self.first_lift_wall = time.time()
        if during_intervention:
            self.latched_during_intervention = True

    def update(self):
        """Observe the world and decide. Kept as the single-call public surface."""
        return self.decide(self.observe())

    def observe(self, *, during_intervention: bool = False):
        """Advance the latches/streak from the current sim state. Safe in ANY mode.

        Split out of update() so it can run during an intervention too. `first_close_step`
        and `first_lift_step` are REQUIRED for tshape/generic success, and they were only
        ever set from the replay branch — so a human who grasped, lifted and placed the
        object entirely by hand left both unset and the episode could never succeed. The
        decision half deliberately still runs only in replay (see decide()).

        Note `frame_idx` is frozen while the policy is paused, so every latch a human sets
        records the frame the intervention began; `first_close_wall`/`first_lift_wall` and
        `latched_during_intervention` carry the honest timing.
        """
        if not self.available:
            return None

        frame_idx = int(getattr(self.player, "frame_idx", 0))
        final_gripper = float(self.data.ctrl[7]) if self.data.ctrl.shape[0] > 7 else 0.0
        # Set by the cups/tshape branches when a gate that success REQUIRES has been lost.
        # Initialised here so the generic branch (which has no such gate) leaves it empty.
        invalid = ""

        if self.task_mode == "cups":
            status = cups_success_status(
                self.model,
                self.data,
                cup_pairs=self.cup_pairs,
                cup_body_names=self.cup_body_names,
                box_body_names=self.box_body_names,
                box_upright_angle_deg=self.box_upright_angle_deg,
                cup_upright_angle_deg=self.cup_upright_angle_deg,
                placement_target_local_offset=self.placement_target_local_offset,
                placement_z_offset=self.placement_z_offset,
                placement_xy_threshold=self.placement_xy_threshold,
                placement_z_tolerance=self.placement_z_tolerance,
                placement_gripper_open_threshold=self.cups_gripper_open_threshold,
                require_contact=self.cups_require_contact,
                use_advanced_metric=self.cups_use_advanced_metric,
                max_axis_error_deg=self.cups_max_axis_error_deg,
                min_radial_margin=self.cups_min_radial_margin,
                forbid_upper_bad_contacts=self.cups_forbid_upper_bad_contacts,
                forbid_upper_wrong_contacts=self.cups_forbid_upper_wrong_contacts,
                bad_contact_body_names=self.cups_bad_contact_body_names,
                max_pair_linear_speed=self.cups_max_pair_linear_speed,
                max_pair_angular_speed=self.cups_max_pair_angular_speed,
            )
            pose_is_placed = bool(status["success_now"])
            contact = bool(status["all_pairs_contact"])
            self.contact_streak = self.contact_streak + 1 if pose_is_placed else 0
            success = pose_is_placed
            reason = "" if success else cups_fail_reason(status, self.placement_stable_steps, self.contact_streak)
            # boxes_ok / cups_ok are AND-ed into success_now, so once an object is over
            # its upright limit the episode cannot be won any more.
            if not status["boxes_ok"]:
                invalid = f"box_toppled:{status['worst_box']}:{status['max_box_angle_deg']:.0f}deg"
            elif not status["cups_ok"]:
                invalid = f"cup_toppled:{status['worst_cup']}:{status['max_cup_angle_deg']:.0f}deg"
            else:
                invalid = ""
        elif self.task_mode == "tshape":
            if self.initial_body_pos is None:
                self.initial_body_pos = self._body_pos(self.body_name)
            obj_pos = self._body_pos(self.body_name)
            lift = float(obj_pos[2] - self.initial_body_pos[2])
            if final_gripper < self.close_threshold:
                self._latch_close(frame_idx, during_intervention)
            if lift >= self.lift_threshold:
                self._latch_lift(frame_idx, during_intervention)

            status = tshape_success_status(
                self.model,
                self.data,
                lifted_bar_geom=self.tshape_lifted_bar_geom,
                lifted_stem_geom=self.tshape_lifted_stem_geom,
                target_bar_geom=self.tshape_target_bar_geom,
                target_stem_geom=self.tshape_target_stem_geom,
                placement_gripper_open_threshold=self.placement_gripper_open_threshold,
                min_bar_overlap=self.tshape_min_bar_overlap,
                max_bar_axis_error_deg=self.tshape_max_bar_axis_error_deg,
                min_face_alignment=self.tshape_min_face_alignment,
                min_stem_up_alignment=self.tshape_min_stem_up_alignment,
                min_stem_height=self.tshape_min_stem_height,
                max_face_gap=self.tshape_max_face_gap,
                min_target_top_alignment=self.tshape_min_target_top_alignment,
                require_bar_contact=self.tshape_require_bar_contact,
                forbid_stem_contact=self.tshape_forbid_stem_contact,
            )
            pose_is_placed = bool(status["success_now"])
            self.contact_streak = self.contact_streak + 1 if pose_is_placed else 0
            success = (
                self.first_close_step is not None
                and self.first_lift_step is not None
                and pose_is_placed
            )
            # The T2 target must still face up; if it has been knocked over, no placement
            # of T1 can satisfy success_now.
            invalid = ("" if status["target_top_world_dot"] >= self.tshape_min_target_top_alignment
                       else f"target_toppled:{self.target_body_name}:"
                            f"{status['target_top_world_dot']:.2f}")
            reason = "" if success else tshape_fail_reason(
                status,
                first_close_step=self.first_close_step,
                first_lift_step=self.first_lift_step,
                placement_stable_steps=self.placement_stable_steps,
                contact_streak=self.contact_streak,
                min_target_top_alignment=self.tshape_min_target_top_alignment,
                min_bar_overlap=self.tshape_min_bar_overlap,
                max_bar_axis_error_deg=self.tshape_max_bar_axis_error_deg,
                min_face_alignment=self.tshape_min_face_alignment,
                min_stem_up_alignment=self.tshape_min_stem_up_alignment,
                min_stem_height=self.tshape_min_stem_height,
                max_face_gap=self.tshape_max_face_gap,
                placement_gripper_open_threshold=self.placement_gripper_open_threshold,
                require_bar_contact=self.tshape_require_bar_contact,
                forbid_stem_contact=self.tshape_forbid_stem_contact,
            )
        else:
            if self.initial_body_pos is None:
                self.initial_body_pos = self._body_pos(self.body_name)
            obj_pos = self._body_pos(self.body_name)
            lift = float(obj_pos[2] - self.initial_body_pos[2])
            if final_gripper < self.close_threshold:
                self._latch_close(frame_idx, during_intervention)
            if lift >= self.lift_threshold:
                self._latch_lift(frame_idx, during_intervention)

            _, xy_error, z_error, yaw_error_deg = placement_metrics(
                self.model,
                self.data,
                body_name=self.body_name,
                target_body_name=self.target_body_name,
                target_local_offset=np.asarray(self.placement_target_local_offset, dtype=np.float64),
                placement_z_offset=self.placement_z_offset,
            )
            contact = bodies_in_contact(self.model, self.data, self.body_name, self.target_body_name)
            pose_is_placed = (
                xy_error <= self.placement_xy_threshold
                and abs(z_error) <= self.placement_z_tolerance
                and yaw_error_deg <= self.placement_yaw_threshold_deg
                and (self.placement_gripper_open_threshold <= 0 or final_gripper >= self.placement_gripper_open_threshold)
                and contact
            )
            self.contact_streak = self.contact_streak + 1 if pose_is_placed else 0
            success = (
                self.first_close_step is not None
                and self.first_lift_step is not None
                and self.contact_streak >= self.placement_stable_steps
            )
            reason = "" if success else fail_reason(
                first_close_step=self.first_close_step,
                first_lift_step=self.first_lift_step,
                target_xy_error=xy_error,
                max_target_xy_error=self.placement_xy_threshold,
                target_z_error=z_error,
                max_abs_target_z_error=self.placement_z_tolerance,
                target_yaw_error_deg=yaw_error_deg,
                max_target_yaw_error_deg=self.placement_yaw_threshold_deg,
                final_gripper=final_gripper,
                min_open_gripper=self.placement_gripper_open_threshold,
                t1_t2_contact=contact,
                contact_streak=self.contact_streak,
                min_contact_streak=self.placement_stable_steps,
            )

        # Observed, never decided here: `observe()` also runs during an intervention (see
        # _observe_task_evaluator), and a human who lifts an object off the table mid-
        # takeover must not fail their own episode. `decide()` is replay-only and is the
        # single place this becomes terminal.
        # Detection is unconditional -- it feeds the REASON. Whether either becomes
        # terminal is decided in `decide()`, which by default lets the episode run to the
        # cutoff and judges the end state there.
        dropped = self._dropped_object()
        if dropped and not success:
            reason = f"object_dropped:{dropped}"

        # `invalid` is set by the cups/tshape branches above; the generic branch has no
        # equivalent required gate, so it stays unset there.
        scene_invalid = self._note_scene_invalid(invalid) if not success else None
        if scene_invalid and not success and not dropped:
            reason = scene_invalid

        self.last_reason = reason
        return {"success": bool(success), "reason": reason, "frame_idx": frame_idx,
                "dropped": dropped, "scene_invalid": scene_invalid}

    def decide(self, obs):
        """Turn an observation into a success/failure decision, or None to keep going.

        THE END STATE DECIDES, and the two directions are deliberately asymmetric:

          * SUCCESS the moment the correct configuration exists. Waiting for the cutoff
            would be wrong twice over -- it wastes the rest of the budget, and it lets the
            policy disturb a configuration that was already correct.
          * FAILURE only at the cutoff. Nothing ends an episode early as a failure. A
            scene that looks lost at step 40 can still be recovered by the policy or by a
            human takeover before the limit, and calling it early records an outcome the
            end state contradicts.

        `dropped` / `scene_invalid` therefore normally supply only the REASON carried to
        the cutoff, so the failure reads `object_dropped:cup1` rather than a generic
        geometric miss. Set INTERVENE_TASK_FAIL_EARLY_ON_DROP / _ON_TOPPLE to make them
        terminal (default off).

        Pure with respect to the sim: it reads only what observe() latched. Callers run
        this ONLY in replay mode, so an intervention can neither end an episode nor
        consume its budget.
        """
        if obs is None:
            return None

        frame_idx = int(obs.get("frame_idx", 0))
        reason = obs.get("reason", "")
        if obs.get("success"):
            return {"status": "success", "step": frame_idx, "reason": ""}
        # Opt-in early exits. Off by default: see the asymmetry above.
        if obs.get("dropped") and self.fail_early_on_drop:
            return {"status": "failure", "step": frame_idx,
                    "reason": reason or f"object_dropped:{obs['dropped']}"}
        if obs.get("scene_invalid") and self.fail_early_on_topple:
            return {"status": "failure", "step": frame_idx,
                    "reason": reason or str(obs["scene_invalid"])}
        # elapsed(), not raw wall time: the human's correction time is excluded, so a
        # long takeover cannot fail the episode on the task timeout.
        if self.timeout_seconds > 0 and self.elapsed() >= self.timeout_seconds:
            reason = "duration_exceeded" if self.task_mode in {"cups", "tshape"} else "out_of_time"
            return {"status": "failure", "step": frame_idx, "reason": reason}
        # frame_idx does not advance while paused, so an intervention cannot consume
        # episode budget either. Intentional, and symmetric with the timeout above.
        if self._reached_episode_limit():
            return {"status": "failure", "step": frame_idx, "reason": reason or "episode_limit"}
        return None


try:
    from robot.mirror_controller import MirrorController
    from robot.replan_controller import ReplanController
    from robot.live_replan_session import (
        InterventionTransitionError,
        LiveReplanSession,
        TrajectoryRecorder,
    )
    from robot.record_robot_adapter import (
        RecordRobotAdapter,
        get_record_backend_error,
        is_record_backend_available,
    )
    from robot.record_planner_adapter import RecordPlannerAdapter
    try:
        from robot.light_polymetis_adapter import LightPolymetisRobotAdapter
        LIGHT_POLYMETIS_IMPORT_ERROR = None
    except Exception as e:
        LightPolymetisRobotAdapter = None
        LIGHT_POLYMETIS_IMPORT_ERROR = e
    ROBOT_IMPORT_ERROR = None
except Exception as e:
    MirrorController = None
    ReplanController = None
    LiveReplanSession = None
    InterventionTransitionError = RuntimeError
    TrajectoryRecorder = None
    RecordRobotAdapter = None
    get_record_backend_error = None
    is_record_backend_available = None
    RecordPlannerAdapter = None
    LightPolymetisRobotAdapter = None
    LIGHT_POLYMETIS_IMPORT_ERROR = e
    ROBOT_IMPORT_ERROR = e


class DisabledMirrorController:
    enabled = False
    robot = None

    def set_robot(self, robot_adapter):
        self.robot = robot_adapter

    def toggle(self, grip_width_seed=None, initial_q_target=None):
        return False

    def get_status_text(self):
        return ""

    def mirror_from_player(self, player):
        return

    def detach_for_reuse(self):
        return None

    def disable(self):
        return


class App:
    def __init__(
        self,
        xml_path: str,
        npz_path: str,
        width=1400,
        height=900,
        view_mode: str = "free",
        player_mode: str = "replay",
        policy_config: PolicyPlayerConfig | None = None,
        dump_camera_images: bool = False,
        cmd_bind: str = "127.0.0.1",
        cmd_port: int = 0,
        state_bind: str = "127.0.0.1",
        state_port: int = 0,
        state_hz: float = 30.0,
    ):
        self.xml_path = xml_path
        # Hashed once at startup, not per episode: the scene FILE cannot change mid-run, and
        # this reads the whole thing. Recorded alongside the path so a later re-render can
        # prove it is using the same scene the episode was captured with.
        #
        # NOTE: the compiled MODEL can now change mid-run -- an OOD episode on a model-level
        # task (cups) compiles a variant of this same scene. `_xml_sha256` still identifies
        # the file and keeps its original meaning for existing analysis; which model an
        # episode actually ran is recorded separately as `episode_model_variant_key`.
        self._xml_sha256 = _file_sha256(xml_path)
        # Identity of the geometry, i.e. the scene file AND its include closure. Editing
        # `generated_cups_via_stl_4.xml` would not move `_xml_sha256` but does move this,
        # which is what makes a variant descriptor safe to pin against.
        try:
            self._scene_closure_sha256 = scene_closure_sha256(xml_path)
        except Exception as exc:
            print(f"[Variant][WARN] scene closure hash failed for {xml_path}: {exc}")
            self._scene_closure_sha256 = ""
        # Let PolicySceneRandomizer pin descriptors to this scene without re-plumbing it
        # through several constructors.
        os.environ.setdefault("INTERVENE_MUJOCO_XML_PATH", str(xml_path))

        # --- model variants -------------------------------------------------------------
        self.model_variant = None                 # descriptor for the CURRENT scene
        self.model_variant_key = BASE_VARIANT_KEY
        self.model_variant_prov = {}
        self.model_epoch = 0                      # monotonic; +1 on every actual swap
        self.model_reloads_total = 0
        self.next_variant_hint = None             # published so consumers can prefetch
        self.variant_cache = None
        self.speculation_hit = 0
        self.speculation_miss = 0
        self._session_index = _policy_session_index(cmd_port)
        self.npz_path = npz_path
        self.width = width
        self.height = height
        self.view_mode = view_mode
        self.player_mode = player_mode
        self.policy_config = policy_config
        self.dump_camera_images = dump_camera_images
        self.cmd_receiver = PolicyCommandReceiver(cmd_bind, int(cmd_port)) if int(cmd_port) > 0 else None
        self.state_publisher = (
            PolicyStatePublisher(state_bind, int(state_port), float(state_hz))
            if int(state_port) > 0
            else None
        )
        self.state_seq = 0
        self.episode_id = f"{self.player_mode}_{int(time.time())}"
        self.episode_number = 1
        self.success_count = 0
        self.failure_count = 0
        self.headless = _env_bool("INTERVENE_HEADLESS", False)

        # FACTR mode: a Dynamixel force-feedback leader arm drives the SIMULATED robot
        # during an intervention (utils/run_main_policy.sh FACTR_ACTIVE=1, or a study
        # block with "control": "factr"). Mutually exclusive with MC_ACTIVE — the
        # launcher rejects the combination before anything starts.
        self.factr_active = _env_bool(
            "INTERVENE_FACTR_ACTIVE",
            _env_bool("FACTR_ACTIVE", False),
        )

        # --- Policy-side performance metrics --------------------------------
        # The VR runtimes have had windowed metrics for a while; the POLICY side had none,
        # which is exactly why the episode recorder's cost stayed invisible. It turned out
        # to be 62% of all render work in the system (3 cameras x every policy frame x 9
        # sessions = 270 render+readback ops/s) while nothing displays those frames.
        # Off unless INTERVENE_METRICS=1, and a no-op object when off so the hot path is
        # a single attribute lookup.
        self.metrics = None
        if _env_bool("INTERVENE_METRICS", False):
            self.metrics = _build_policy_metrics(
                session_index=_policy_session_index(cmd_port),
                window_s=float(os.environ.get("INTERVENE_METRICS_WINDOW_S", "10") or 10.0),
                print_summary=_env_bool("INTERVENE_METRICS_SUMMARY", False),
            )

        # --- User study logging (PART-2) -----------------------------------
        # Off unless STUDY_PARTICIPANT is set; everything below degrades to the
        # previous behaviour when study_session is None. A malformed config is fatal
        # on purpose: recording a participant under the wrong condition is worse than
        # not starting.
        # HRI experiment block (2026-08-11). When EXP_PARTICIPANT is set, all nine cells
        # share one block directory created by `experiment_block.start_block` before any
        # process launched, and the participant-config flow is bypassed: the block
        # description IS the condition. Otherwise the pre-existing STUDY_PARTICIPANT
        # path is untouched.
        self.exp_block = resolve_experiment_block()
        if self.exp_block is not None:
            print(describe_experiment_block(self.exp_block))
            self.study_block = study_block_from_experiment(
                self.exp_block, cell_id=self._session_index
            )
        else:
            self.study_block = load_block()
        self.study_session = None
        if self.study_block is not None:
            print(describe_study(self.study_block))
            self.study_session = StudySession(
                self.study_block,
                exp_block=self.exp_block,
                cell_id=self._session_index,
                env_info={
                    "mujoco_xml_path": self.xml_path,
                    "mujoco_xml_sha256": self._xml_sha256,
                    "reset_npz_path": self.npz_path,
                    "checkpoint": (str(policy_config.checkpoint)
                                   if policy_config is not None else None),
                    "player_mode": self.player_mode,
                    "task_mode": os.environ.get("INTERVENE_TASK_MODE", ""),
                    "randomization_preset": os.environ.get(
                        "INTERVENE_RANDOMIZATION_PRESET", ""),
                    "acc_method": os.environ.get("STUDY_ACC_METHOD", "chunk_residual"),
                    "policy_hz": (float(getattr(policy_config, "policy_hz", 0.0))
                                  if policy_config is not None else None),
                    "action_mode": (getattr(policy_config, "action_mode", None)
                                    if policy_config is not None else None),
                    "cell_id": self._session_index,
                    "cmd_port": cmd_port,
                },
            )
            # Every event carries the two counters that decide, offline, whether the
            # policy acted between RELEASE and SUCCESS. Reading them lazily (rather than
            # passing them at each call site) keeps the ~20 logging call sites unchanged.
            self.study_session.step_provider = self._step_counters
        # Failed episodes used to be discarded outright, which throws away exactly the
        # recoveries an intervention study is about. Keep them by default in study
        # mode; STUDY_KEEP_FAILURES=0 restores the old discard behaviour.
        self.keep_failed_episodes = _env_bool(
            "STUDY_KEEP_FAILURES", self.study_session is not None
        )

        self.window = None
        self.ctx = None
        self.mouse = MouseState()

        self.player = None
        self.viewer = None

        if MirrorController is None:
            self.mirror_controller = DisabledMirrorController()
            self.replan_controller = None
            self.robot_features_available = False
            print(f"[INFO] Robot features disabled: {ROBOT_IMPORT_ERROR}")
        else:
            self.mirror_controller = MirrorController(
                robot_adapter=None,
                view_hz=60.0,
            )

            self.replan_controller = ReplanController(planner_adapter=None)

            self.robot_features_available = False

            try:
                # FACTR flips the defaults: the "robot" is the leader arm, reached over
                # the local RPC service rather than polymetis. An explicit override to a
                # different backend is a launch mistake, not something to paper over.
                robot_key = os.environ.get(
                    "INTERVENE_ROBOT_KEY",
                    "factr" if self.factr_active else "p4",
                )
                robot_backend = os.environ.get(
                    "INTERVENE_ROBOT_BACKEND",
                    "factr_rpc" if self.factr_active else "record",
                ).strip().lower()
                control_mode = os.environ.get(
                    "INTERVENE_ROBOT_CONTROL_MODE",
                    "HYBRID_JOINT_IMPEDANCE_CONTROL",
                )
                if self.factr_active:
                    if robot_backend != "factr_rpc":
                        raise RuntimeError(
                            "FACTR_ACTIVE=1 requires INTERVENE_ROBOT_BACKEND=factr_rpc "
                            f"(got {robot_backend!r})"
                        )
                    # Imported here, not at module scope: the client is stdlib+numpy only,
                    # so the 9-15 policy processes never pull in pinocchio/dynamixel_sdk.
                    # Only the standalone FACTR service imports the hardware adapter.
                    from robot.factr_rpc import FactrRpcAdapter

                    robot_adapter = FactrRpcAdapter(
                        robot_key=robot_key,
                        control_mode=control_mode,
                    )
                elif robot_backend == "record":
                    if (
                        is_record_backend_available is not None
                        and not is_record_backend_available()
                    ):
                        raise RuntimeError(
                            f"record backend unavailable: {get_record_backend_error()}"
                        )
                    robot_adapter = RecordRobotAdapter(
                        robot_key=robot_key,
                        control_mode=control_mode,
                    )
                else:
                    if LightPolymetisRobotAdapter is None:
                        raise RuntimeError(
                            f"light polymetis backend unavailable: {LIGHT_POLYMETIS_IMPORT_ERROR}"
                        )
                    robot_adapter = LightPolymetisRobotAdapter(
                        robot_key=robot_key,
                        control_mode=control_mode,
                    )

                self.mirror_controller.set_robot(robot_adapter)
                self.replan_controller.set_planner(
                    RecordPlannerAdapter(robot_key=robot_key, view_hz=60.0, log_hz=60.0)
                )
                self.robot_features_available = True
                print(f"[INFO] Robot features enabled for {robot_key} via {robot_backend} backend.")
            except Exception as e:
                print(f"[INFO] Robot features disabled: {e}")

        # Motion-controller mode: when set (via MC_ACTIVE=1 in run_main_policy.sh), interventions
        # do NOT enter HUMAN_CONTROL/freedrive — the external mq3_mc.py process drives the arm via
        # CARTESIAN_IMPEDANCE. Keeps the two controllers from clashing.
        self.mc_active = os.environ.get("INTERVENE_MC_ACTIVE", "0") in ("1", "true", "True")

        # Sim-native motion-controller mode (MC_ACTIVE=1 MC_SIM=1 in run_main_policy.sh):
        # Quest MotionController input drives the LIVE simulated arm via local IK
        # (robot/sim_mc_driver.py) — no real robot connection, no mq3_mc.py sidecar.
        self.sim_mc_active = os.environ.get("INTERVENE_MC_SIM", "0") in ("1", "true", "True")
        self.sim_mc_driver = None

        self.mode = "replay"

        # Failed-intervention notification + backoff.
        # Telekinesis needs the real arm, so with the robot unreachable every X press cost
        # a ~2 s blocking gRPC connect AND a pause/resume of the policy, while the operator
        # in the headset saw nothing at all. The counter drives the VR HUD; the backoff
        # stops repeated presses from stalling the sim once we already know it will fail.
        self.intervention_failed_seq = 0
        self.intervention_failed_reason = ""
        self.last_intervention_failure_wall = 0.0

        # Exclusive ownership of the physical arm, shared with the VR runtime via a lock
        # file. A multi-window run has N policy processes and one robot. Held only for the
        # duration of an intervention, never for the whole session, so windows can take
        # turns. INTERVENE_ROBOT_LOCK=0 disables it (single-process debugging).
        self.robot_lock = None
        if os.environ.get("INTERVENE_ROBOT_LOCK", "1") not in ("0", "false", "False"):
            try:
                from robot.robot_ownership_lock import RobotOwnershipLock

                self.robot_lock = RobotOwnershipLock(
                    os.environ.get("INTERVENE_ROBOT_KEY", "p4")
                )
            except Exception as exc:
                print(f"[INFO] Robot ownership lock unavailable: {exc}")

        # --- Out-of-distribution scene band ---------------------------------------
        # Keeps between OOD_MIN_STATES and OOD_MAX_STATES of the grid's sessions running an
        # OOD scene. Each session decides only for itself, at its own episode boundary, so
        # the OOD cells migrate without any grid-wide reset. Off iff min <= 0.
        self.ood_ledger = None
        self.current_scene_ood = False
        try:
            from robot.ood_state_ledger import OodStateLedger

            ledger = OodStateLedger(
                session_index=_policy_session_index(cmd_port),
                # Keyed on the launch cohort's base command port, NOT the robot key: the
                # robot key is shared by every run on this machine, so two concurrent
                # launches would otherwise contend over one band.
                ledger_key=os.environ.get(
                    "INTERVENE_OOD_LEDGER_KEY", os.environ.get("POLICY_CMD_PORT", "8065")
                ),
                min_states=int(os.environ.get("INTERVENE_OOD_MIN_STATES", "0") or 0),
                max_states=int(os.environ.get("INTERVENE_OOD_MAX_STATES", "3") or 3),
                ttl_s=float(os.environ.get("INTERVENE_OOD_LEDGER_TTL_S", "1800") or 1800),
            )
            if ledger.enabled:
                self.ood_ledger = ledger
                # The launcher pre-assigns the starting set so grid position is not
                # correlated with condition; band-filling at startup would decide in the
                # staggered launch order and pin OOD to the first sessions every run.
                initial = _env_bool("INTERVENE_OOD_INITIAL_STATE", False)
                self.current_scene_ood = ledger.register_initial(initial)
                print(f"[OOD] band {ledger.min_states}..{ledger.max_states} "
                      f"session={ledger.session_index} initial="
                      f"{'OOD' if self.current_scene_ood else 'ID'} ledger={ledger.path}")
                if (
                    self.current_scene_ood
                    and self.study_block is None
                    and not _env_bool("INTERVENE_RANDOMIZE_FIRST_EPISODE", True)
                ):
                    # Only reachable with INTERVENE_RANDOMIZE_FIRST_EPISODE=0: the first
                    # episode then runs the plain reset pose on the base model, so a
                    # session the ledger just registered as OOD shows nothing visibly OOD.
                    print("[OOD]   NOTE: INTERVENE_RANDOMIZE_FIRST_EPISODE=0, so the first "
                          "episode runs the UNRANDOMIZED reset pose on the base model. "
                          "This session becomes genuinely OOD at the FIRST episode "
                          "boundary, not now.")
        except Exception as exc:
            print(f"[INFO] OOD scene band unavailable: {exc}")

        self.replan_session = None
        self.replan_cut_idx = None
        self.replan_original_path = None
        self.replan_should_resume_mirror = False
        self.intervention_phase = "policy_resumed"
        self.intervention_operation = "idle"
        # Post-release check state (see _run_post_release_check).
        self._post_release_check_pending = False
        self._intervention_reached_human_control = False
        self.last_finished_intervention_id = 0
        self.last_finished_intervention_outcome = ""
        # Whether the operator currently has THIS cell selected. Set by the SELECTED /
        # DESELECTED notifications the VR runtime and the desktop grid forward; it is
        # bookkeeping only and never gates any control path.
        self.cell_selected = False
        self.intervention_transition_started_wall = 0.0
        # What the OPERATOR waits for: request -> policy back in their hands. Ends at the
        # finish/cancel keypress. The arm's return home is measured separately because it
        # no longer blocks anything.
        self.last_intervention_transition_s = 0.0
        self.last_return_home_s = 0.0
        self._replan_paused_policy = False
        self.return_home_interrupted_by_replan = False
        self._return_home_interrupted = False
        self._return_home_thread = None
        self._return_home_done = threading.Event()
        self._return_home_result = False
        self._return_home_error = None
        self._return_home_context = None
        self._return_home_cancel = threading.Event()
        self._return_home_timeout_reported = False
        self._pending_reintervene = False
        self._loop_heartbeat = [time.time()]
        self._metrics_last_frame_idx = None
        self._metrics_last_content_mono = time.monotonic()
        self._metrics_content_updates = 0
        self.episode_recorder = None
        self.episode_saved = False
        self.episode_save_path = None
        self.last_recorded_player_frame_idx = None
        self.next_intervention_id = 1
        self.active_intervention_id = 0
        self.replan_recorder_start_len = None
        self.replan_start_snapshot = None
        self.record_camera_names = []
        self.record_rgb_width = 224
        self.record_rgb_height = 224
        self.record_save_rgb = True
        self.record_save_depth = False
        self.scene_randomizer = None
        self.base_scene_qpos = None
        self.base_scene_qvel = None
        self.base_scene_ctrl = None
        self.current_scene_qpos = None
        self.current_scene_qvel = None
        self.current_scene_ctrl = None
        self.scene_id = 0
        self.scene_attempt = 0
        self.scene_qpos_hash = ""
        try:
            self.ood_max_scene_attempts = max(
                1, int(os.environ.get("INTERVENE_OOD_MAX_SCENE_ATTEMPTS", "3") or 3)
            )
        except (TypeError, ValueError):
            self.ood_max_scene_attempts = 3
        self.task_evaluator = None

    def init(self):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.default_window_hints()
        if self.headless:
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
            glfw.window_hint(glfw.DOUBLEBUFFER, glfw.FALSE)
        else:
            glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
            glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)
        glfw.window_hint(glfw.DEPTH_BITS, 24)

        win_w = 32 if self.headless else self.width
        win_h = 32 if self.headless else self.height
        self.window = glfw.create_window(
            win_w, win_h, "MuJoCo Single Scene Player", None, None
        )
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)
        glfw.swap_interval(0 if self.headless else 1)

        if self.player_mode == "policy":
            if self.policy_config is None:
                raise ValueError("policy_config is required when player_mode='policy'")
            self.player = PolicyPlayer(
                self.xml_path,
                self.policy_config,
                context_current_fn=lambda: glfw.make_context_current(self.window),
            )
        elif self.player_mode == "replay":
            self.player = TrajectoryPlayer(self.xml_path, self.npz_path)
        else:
            raise ValueError(f"Unknown player_mode: {self.player_mode}")

        self.viewer = SceneViewer(self.player.model, view_mode=self.view_mode)
        glfw.make_context_current(self.window)
        self.ctx = mujoco.MjrContext(
            self.player.model, mujoco.mjtFontScale.mjFONTSCALE_100
        )

        install_callbacks(self)
        if self.cmd_receiver is not None:
            self.cmd_receiver.start()
        if self.state_publisher is not None:
            self.state_publisher.start()
        self._initialize_scene_lifecycle()
        # `_initialize_scene_lifecycle` does NOT randomize, so episode 1 used to run the
        # reset-NPZ pose verbatim outside study mode. With WINDOWS>1 that meant every
        # cell in the grid booted to the IDENTICAL scene and stayed identical until its
        # first episode boundary. In a study it also made episode 1 the odd one out — a
        # confound — which is why the study path has always randomized it. Both want the
        # same thing, so it is now the default for every mode.
        # Set INTERVENE_RANDOMIZE_FIRST_EPISODE=0 to get the pristine reset pose back;
        # study mode randomizes regardless, since there the confound is not optional.
        if self.study_block is not None or _env_bool(
            "INTERVENE_RANDOMIZE_FIRST_EPISODE", True
        ):
            self._start_new_randomized_scene()
        self._ensure_episode_recorder()
        self._record_player_frame(force=True)
        self._publish_policy_state(force=True)

    def _initialize_scene_lifecycle(self):
        self.base_scene_qpos = self.player.data.qpos.copy()
        self.base_scene_qvel = self.player.data.qvel.copy()
        self.base_scene_ctrl = self.player.data.ctrl.copy()
        self.scene_randomizer = PolicySceneRandomizer(self.player.model, self.base_scene_qpos)
        self.current_scene_qpos = self.base_scene_qpos.copy()
        self.current_scene_qvel = self.base_scene_qvel.copy()
        self.current_scene_ctrl = self.base_scene_ctrl.copy()
        self.scene_id = 1
        self.scene_attempt = 1
        self.scene_qpos_hash = _qpos_sha256(self.current_scene_qpos)
        self.task_evaluator = LiveTaskEvaluator(self.player.model, self.player.data, self.player)
        # Mirrors the "[SCENE] New scene ..." line so `grep '\[SCENE\]'` reads as a clean
        # timeline. The initial scene is deliberately NOT randomized outside study mode.
        _pending_ood = bool(self.current_scene_ood)
        print(f"[SCENE] Initial scene id={self.scene_id} attempt={self.scene_attempt} "
              f"ood=0{' (pending=1)' if _pending_ood else ''} "
              f"qpos_hash={self.scene_qpos_hash} model={self.model_variant_key}"
              + ("  - not randomized; OOD applies from the next episode."
                 if _pending_ood else ""))
        self._init_variant_cache()
        # Start the first background compile NOW. Without this the first prefetch cannot run
        # until after the first _start_new_randomized_scene, i.e. after the first swap has
        # already stalled -- which is exactly the measured "built variant in 1458 ms" that
        # preceded "episode 2: model base -> ... swap=603 ms".
        self._schedule_speculative_variants()

    def _init_variant_cache(self):
        """Set up the per-episode model-variant cache, if this task needs one.

        Capacity 2 -- the pinned base plus the current variant. A policy process only ever
        renders one session, so it never needs more; the grid viewer is the one that sizes
        its cache by `OOD_MAX_STATES`.
        """
        if self.variant_cache is not None:
            return
        randomizer = self.scene_randomizer
        if randomizer is None or not getattr(randomizer, "ood_model_variants", False):
            return
        try:
            from model_variants.cache import VariantCache
            self.variant_cache = VariantCache(
                self.xml_path, reference_model=self.player.model,
                capacity=2, label=f"Variant/S{self._session_index:02d}",
                enable_worker=_env_bool("INTERVENE_OOD_VARIANT_PRECOMPILE", True),
            )
            self.variant_cache.install_base(self.player.model)
        except Exception as exc:
            print(f"[Variant][WARN] variant cache unavailable ({exc}); "
                  "this session stays IN-distribution.")
            self.variant_cache = None
            randomizer.ood_available = False
            randomizer.ood_unavailable_reason = f"variant_cache_unavailable:{exc}"

    def _swap_model(self, descriptor, key):
        """Adopt a different compiled model. ONLY legal at an episode boundary.

        Rebuilds every borrower of `player.model`: the player (data + policy renderer), the
        viewer's MjvScene, the window MjrContext, the randomizer's name->index maps, and the
        task evaluator. Returns True if the swap happened.

        On any failure the session falls back to the base model rather than aborting -- a
        study session must not die because one corpus scene is bad.
        """
        if self.replan_session is not None or self.mode == "replan":
            # A live takeover holds model= and data= references (see the LiveReplanSession
            # construction below); swapping under it would be a use-after-free.
            print("[Variant][ERROR] refusing to swap the model during an intervention.")
            # Every OTHER early return here clears current_scene_ood and records why (see
            # the two branches below). This one did not, so the caller re-drew an
            # in-distribution pose while the episode stayed stamped episode_ood=True —
            # the NPZ, the study metadata and the published variant_key all disagreed.
            self.current_scene_ood = False
            if getattr(self, "scene_randomizer", None) is not None:
                self.scene_randomizer.last_offsets["ood_fallback_reason"] = "intervention_active"
            return False
        if self.variant_cache is None:
            return False

        try:
            model = self.variant_cache.get_or_build(descriptor)
        except Exception as exc:
            print(f"[Variant][ERROR] variant {key} unavailable ({exc}); "
                  "falling back to the base model for this episode.")
            if self.model_variant_key != BASE_VARIANT_KEY:
                self._swap_model(None, BASE_VARIANT_KEY)
            self.current_scene_ood = False
            self.scene_randomizer.last_offsets["ood_fallback_reason"] = f"variant_build_failed:{exc}"
            return False

        t0 = time.perf_counter()
        try:
            self.player.swap_model(model)
        except Exception as exc:
            print(f"[Variant][ERROR] player refused variant {key} ({exc}); staying on "
                  f"{self.model_variant_key}.")
            self.current_scene_ood = False
            self.scene_randomizer.last_offsets["ood_fallback_reason"] = f"player_refused:{exc}"
            return False

        self.viewer.rebind_model(model)
        # Rebuild the render context only if one actually exists. `MjrContext` uploads the
        # model's meshes and textures to GL, so constructing it without a current GL context
        # aborts the process (observed as a libc++ recursive-initialization crash).
        if self.window is not None and self.ctx is not None:
            glfw.make_context_current(self.window)
            old_ctx = self.ctx
            self.ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_100)
            try:
                old_ctx.free()
            except Exception:
                pass
        self.scene_randomizer.rebind_model(model)
        if self.task_evaluator is not None:
            self.task_evaluator.rebind(model, self.player.data)
        # The sim-MC driver is ANOTHER borrower of model/data, and it is cached for the
        # whole session. Left unrebound it keeps driving the MjData that swap_model just
        # replaced: the arm never moves during the next takeover and the recording is a
        # frozen pose, with no error anywhere. Rebind keeps the Quest connection.
        if getattr(self, "sim_mc_driver", None) is not None:
            try:
                self.sim_mc_driver.rebind(model, self.player.data)
            except Exception as exc:
                print(f"[Variant][WARN] sim-MC driver rebind failed ({exc}); "
                      "it will be rebuilt at the next intervention.")
                self.sim_mc_driver = None

        previous = self.model_variant_key
        self.model_variant = descriptor
        self.model_variant_key = key
        self.model_variant_prov = dict(
            self.scene_randomizer.last_offsets.get("ood_model_variant_provenance") or {}
        )
        self.model_epoch += 1
        self.model_reloads_total += 1
        self.variant_cache.retain(key)
        if previous != BASE_VARIANT_KEY:
            self.variant_cache.release(previous)
        print(f"[Variant] episode {self.episode_number}: model {previous} -> {key} "
              f"(epoch={self.model_epoch}, swap={(time.perf_counter() - t0) * 1e3:.0f} ms)")
        return True

    def _schedule_speculative_variants(self):
        """Compile the NEXT episode's variant while this one runs, so the boundary is free.

        The draw is a pure function of the seed: in study mode the randomizer is re-seeded
        from `study_block.episode_seed(episode_number)` and, for a model-level task,
        `validate_tshape_scene` short-circuits so the first candidate is always taken. So the
        next scene index is predictable exactly -- this is not a guess.

        Two candidates, because the failure path advances `episode_number` by 1 or 2
        depending on `keep_failed_episodes` and OOD rotation. The in-distribution case needs
        no compile at all: it IS the base model.

        MUST use its own RNG. Touching `self.scene_randomizer.rng` here would shift every
        in-distribution draw in the study.
        """
        if self.variant_cache is None:
            return
        randomizer = self.scene_randomizer
        if randomizer is None or not randomizer.ood_available or not randomizer.ood_scenes:
            return

        if self.study_block is not None:
            # Each episode is independently re-seeded, so both candidates are exact.
            indices = []
            for offset in (1, 2):
                try:
                    seed = self.study_block.episode_seed(self.episode_number + offset)
                    indices.append(
                        int(np.random.default_rng(seed).integers(len(randomizer.ood_scenes)))
                    )
                except Exception as exc:
                    print(f"[Variant][WARN] seed peek failed (+{offset}): {exc}")
        else:
            # No per-episode reseed, so the stream is continuous and only the IMMEDIATE next
            # draw is predictable. Two-ahead would additionally assume the following episode
            # is also OOD; an in-distribution episode consumes rng.uniform draws and shifts
            # the stream. One wrong prediction costs a background compile, never a block.
            indices = randomizer.peek_next_ood_scene_indices(1)

        for rank, idx in enumerate(indices):
            try:
                descriptor = randomizer.variant_for_scene_index(idx)
                if descriptor is None:
                    continue
                self.variant_cache.prefetch(descriptor)
                if rank == 0:
                    self.next_variant_hint = descriptor
            except Exception as exc:
                print(f"[Variant][WARN] speculative precompile failed (rank {rank}): {exc}")

    def _apply_scene_snapshot(self, qpos, qvel, ctrl, *, play=True, reset_policy_queue=True):
        self.player.data.qpos[:] = np.asarray(qpos, dtype=np.float64)
        self.player.data.qvel[:] = np.asarray(qvel, dtype=np.float64)
        self.player.data.ctrl[:] = np.asarray(ctrl, dtype=np.float64)
        self.player.data.time = 0.0
        mujoco.mj_forward(self.player.model, self.player.data)
        if hasattr(self.player, "frame_idx"):
            self.player.frame_idx = 0
        if hasattr(self.player, "policy_query_count"):
            self.player.policy_query_count = 0
        if play and hasattr(self.player, "resume_from_current_state"):
            self.player.resume_from_current_state(reset_policy_queue=reset_policy_queue)
        elif not play:
            self.player.is_paused = True
        elif getattr(self.player, "is_paused", False):
            self.player.toggle_pause()

    def _start_new_randomized_scene(self):
        # keep_frames=True: a new episode means this one is being kept, so the human's
        # frames must be kept with it.
        self._end_intervention_for_scene_change("new randomized scene", keep_frames=True)
        if self.scene_randomizer is None:
            self._initialize_scene_lifecycle()

        # Re-seed per episode so the scene is reproducible from
        # (participant, block, episode_number) alone — this is what makes scenario_id
        # an identifier rather than just a label.
        if self.study_block is not None:
            self.scene_randomizer.reseed(
                self.study_block.episode_seed(self.episode_number)
            )

        # Re-decide OOD-ness ONLY here, i.e. only when a genuinely new scene is drawn
        # (task success). _restart_current_scene replays current_scene_qpos verbatim, so a
        # failed episode retries the identical scene including its OOD-ness — a recovery
        # attempt has to be the same condition as the failure it is recovering from.
        if self.ood_ledger is not None:
            try:
                self.current_scene_ood, others = self.ood_ledger.decide_for_new_episode(
                    currently_ood=self.current_scene_ood,
                    episode=self.episode_number,
                    scenario_id=(
                        self.study_block.scenario_id(self.episode_number)
                        if self.study_block is not None else ""
                    ),
                )
                print(f"[OOD] episode {self.episode_number}: this session is "
                      f"{'OOD' if self.current_scene_ood else 'ID'} "
                      f"(others OOD={others}, band={self.ood_ledger.min_states}"
                      f"..{self.ood_ledger.max_states})")
            except Exception as exc:
                print(f"[OOD][WARN] band decision failed ({exc}); staying IN-distribution.")
                self.current_scene_ood = False

        validation_attempts = max(
            1, int(os.environ.get("INTERVENE_SCENE_VALIDATION_ATTEMPTS", "16") or 16)
        )
        qpos = None
        validation_reasons = ()
        for _ in range(validation_attempts):
            candidate = self.scene_randomizer.apply(
                self.base_scene_qpos, ood=self.current_scene_ood
            )
            valid, validation_reasons = self.scene_randomizer.validate_tshape_scene(candidate)
            if valid:
                qpos = candidate
                break
        if qpos is None:
            raise RuntimeError(
                "Could not produce a valid T-shape scene after "
                f"{validation_attempts} candidates: {', '.join(validation_reasons)}"
            )
        # The model may change ONLY here, and only after qpos is settled: the randomizer
        # needs name->index lookups, which are variant-invariant. An in-distribution episode
        # resolves to `None`, i.e. the base model, so it never compiles anything.
        if self.variant_cache is not None:
            wanted = self.scene_randomizer.last_offsets.get("ood_model_variant")
            wanted_key = variant_key(wanted)
            if wanted_key != self.model_variant_key:
                if not self._swap_model(wanted, wanted_key) and wanted is not None:
                    # The variant was refused; re-draw in-distribution rather than record a
                    # cups episode labelled OOD that is really the base model.
                    qpos = self.scene_randomizer.apply(self.base_scene_qpos, ood=False)

        qvel = self.scene_randomizer.apply_qvel(self.base_scene_qvel)
        ctrl = self.base_scene_ctrl.copy()
        self.current_scene_qpos = qpos.copy()
        self.current_scene_qvel = qvel.copy()
        self.current_scene_ctrl = ctrl.copy()
        self.scene_id += 1
        self.scene_attempt = 1
        self.scene_qpos_hash = _qpos_sha256(qpos)
        self._apply_scene_snapshot(qpos, qvel, ctrl, play=True, reset_policy_queue=True)
        self.last_recorded_player_frame_idx = None
        self._publish_policy_state(force=True)
        if self.task_evaluator is not None:
            self.task_evaluator.reset()
        print(f"[SCENE] New scene id={self.scene_id} attempt={self.scene_attempt} "
              f"ood={int(self.current_scene_ood)} qpos_hash={self.scene_qpos_hash} "
              f"model={self.model_variant_key}")
        self._schedule_speculative_variants()

    def _restart_current_scene(self):
        # keep_frames=False: the scene is being replayed from the top, so the takeover's
        # frames are rolled back with everything else.
        self._end_intervention_for_scene_change("scene restart", keep_frames=False)
        if self.current_scene_qpos is None:
            self._initialize_scene_lifecycle()
        # An exact retry must replay the same scene on the same model. The variant can only
        # change in _start_new_randomized_scene, so this holds by construction -- asserted
        # rather than commented because "by construction" is what silently stops being true.
        assert variant_key(self.model_variant) == self.model_variant_key, (
            f"model variant drifted outside a scene change: "
            f"{variant_key(self.model_variant)} != {self.model_variant_key}"
        )
        self._apply_scene_snapshot(
            self.current_scene_qpos,
            self.current_scene_qvel,
            self.current_scene_ctrl,
            play=True,
            reset_policy_queue=True,
        )
        self.last_recorded_player_frame_idx = None
        self._record_player_frame(force=True)
        self._publish_policy_state(force=True)
        if self.task_evaluator is not None:
            self.task_evaluator.reset()
        print(f"[SCENE] Restarted current scene id={self.scene_id} "
              f"attempt={self.scene_attempt}/{self.ood_max_scene_attempts} "
              f"ood={int(self.current_scene_ood)} qpos_hash={self.scene_qpos_hash} "
              "after task failure.")

    def _episode_output_dir(self):
        if self.study_session is not None:
            # Episodes land in a pending/ staging folder and are moved into
            # success/ | failure/ | incomplete/ once the outcome is known.
            path = self.study_session.session_dir / "episodes" / "pending"
            path.mkdir(parents=True, exist_ok=True)
            return path
        return Path(
            os.environ.get(
                "INTERVENE_EPISODE_DIR",
                os.environ.get("INTERVENE_OUTPUT_DIR", "INTERVENTION_DATA"),
            )
        )

    def _last_ood_offset(self, key, default=""):
        """Read a field from the randomizer's last draw. Never raises: metadata must not
        be able to abort an episode save."""
        try:
            return (self.scene_randomizer.last_offsets or {}).get(key, default)
        except Exception:
            return default

    def _study_metadata(self):
        """Study labels embedded in every episode NPZ + sidecar."""
        _prov = getattr(self, "model_variant_prov", None) or {}
        metadata = {
            "episode_ood": bool(self.current_scene_ood),
            # WHICH corpus scene this episode used, so an OOD episode is reconstructible.
            # The sha matters because the corpus is unversioned for t_shape/boxes_cups
            # (its manifest was overwritten by a later wire_spoon-only regeneration), so a
            # filename alone does not pin the content. reseat_z records the lift applied
            # after loading, since runtime z != corpus z.
            "episode_ood_scene": str(self._last_ood_offset("ood_scene_file", "")),
            "episode_ood_scene_sha256": str(self._last_ood_offset("ood_scene_sha256", "")),
            "episode_ood_reseat_z": float(
                os.environ.get("INTERVENE_OBJECT_SUPPORT_Z", "0.0") or 0.0
            ),
            "episode_ood_unavailable_reason": str(
                self._last_ood_offset("ood_fallback_reason", "")
            ),
            "current_scene_ood": bool(self.current_scene_ood),
            "scene_id": int(self.scene_id),
            "scene_attempt": int(self.scene_attempt),
            "scene_qpos_hash": str(self.scene_qpos_hash),
            "ood_max_scene_attempts": int(self.ood_max_scene_attempts),
            # WHICH COMPILED MODEL produced these frames. For a pose-only task this is
            # always "base"; for a model-level task (cups) an OOD episode runs a variant of
            # the same scene file, so `mujoco_xml_sha256` alone no longer identifies the
            # geometry. `episode_scene_closure_sha256` pins the base geometry (file +
            # includes) and the variant key pins what was done to it.
            # getattr defaults, like _last_ood_offset above: metadata must never be able
            # to abort an episode save.
            "episode_model_variant_key": str(getattr(self, "model_variant_key", BASE_VARIANT_KEY)),
            "episode_model_epoch": int(getattr(self, "model_epoch", 0)),
            "episode_scene_closure_sha256": str(getattr(self, "_scene_closure_sha256", "")),
            "episode_model_variant_source": str(_prov.get("corpus_scene", "")),
            "episode_model_variant_source_sha256": str(_prov.get("corpus_scene_sha256", "")),
            "model_reloads_total": int(getattr(self, "model_reloads_total", 0)),
            # Set when model variants were deliberately disabled for a model-level task.
            # An OOD episode recorded with this flag differs from in-distribution by pose
            # only, which for cups is inside the ordinary jitter -- i.e. it is mislabelled,
            # and this is what makes that detectable after the fact.
            "episode_ood_pose_only": bool(
                getattr(getattr(self, "scene_randomizer", None), "ood_pose_only", False)
            ),
        }
        # Long JSON blobs: sidecar only, never the NPZ scalar whitelist.
        try:
            _variant = getattr(self, "model_variant", None)
            metadata["episode_model_variant"] = (
                json.dumps(_variant, sort_keys=True) if _variant else ""
            )
            metadata["episode_model_asset_sha256"] = json.dumps(
                _prov.get("asset_sha256", {}), sort_keys=True
            )
        except Exception:
            metadata["episode_model_variant"] = ""
            metadata["episode_model_asset_sha256"] = "{}"
        if self.study_block is None:
            return metadata
        metadata.update({
            # Whether THIS episode ran an out-of-distribution scene. `condition` is a
            # block-level label; because the OOD band migrates between sessions, only this
            # per-episode flag says what actually ran. Analysis should key off it.
            "participant_id": self.study_block.participant_id,
            "block": self.study_block.block,
            "interface": self.study_block.interface,
            "task": self.study_block.task,
            "condition": self.study_block.condition,
            "episode_number": self.episode_number,
            "episode_seed": self.study_block.episode_seed(self.episode_number),
            "scenario_id": self.study_block.scenario_id(self.episode_number),
            "session_dir": (str(self.study_session.session_dir)
                            if self.study_session is not None else None),
        })
        if self.exp_block is not None:
            # HRI block identity. Scalars only (long blobs go to the sidecar) so a stray
            # .npz still names its participant, condition and cell.
            metadata.update({
                "study": self.exp_block.study,
                "condition_id": self.exp_block.condition_id,
                "supervision_interface": self.exp_block.supervision_interface,
                "controller_interface": self.exp_block.controller_interface,
                "block_id": self.exp_block.block_id,
                "cell_id": int(self._session_index),
                "task_display": self.exp_block.task,
            })
        # Everything needed to re-render this episode's cameras offline. We deliberately
        # store NO RGB/depth/point-cloud frames, so the scene
        # closure hash, the model variant and the camera intrinsics/extrinsics below are
        # what make the observations reconstructible. Sidecar only: these are JSON blobs
        # and `utils/data_cleaning.py` trims any ARRAY whose first axis matches the frame
        # count, which is how a k-element metadata array gets silently sliced.
        try:
            metadata["render_config"] = json.dumps(self._render_config(), sort_keys=True)
        except Exception as exc:
            metadata["render_config"] = ""
            print(f"[WARN] render config capture failed: {exc}")
        return metadata

    def _render_config(self):
        """Camera + timestep configuration sufficient to re-render this episode.

        Read from the LIVE model, not from the XML, so a model-variant episode (cups
        OOD compiles a different model) reports the geometry that actually produced the
        frames. Computed once per episode, on the episode-recorder path -- not per frame.
        """
        model = self.player.model
        cameras = []
        for cam_id in range(int(model.ncam)):
            try:
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_id) or ""
            except Exception:
                name = ""
            cameras.append({
                "id": cam_id,
                "name": name,
                "fovy_deg": float(model.cam_fovy[cam_id]),
                "pos": [float(v) for v in model.cam_pos[cam_id]],
                "quat": [float(v) for v in model.cam_quat[cam_id]],
                "mode": int(model.cam_mode[cam_id]),
                "bodyid": int(model.cam_bodyid[cam_id]),
            })
        # Same derivation `_ensure_episode_recorder` uses for the recorder's log_hz, so
        # the reported control timestep always matches the frame cadence actually
        # recorded -- including in replay mode, which has no policy config. Not read
        # back off `self.episode_recorder`: the first call happens while it is still
        # being constructed.
        policy_hz = float(getattr(getattr(self.player, "config", None), "policy_hz", 60.0)
                          or 60.0)
        return {
            "cameras": cameras,
            "offwidth": int(model.vis.global_.offwidth),
            "offheight": int(model.vis.global_.offheight),
            "offsamples": int(model.vis.quality.offsamples),
            "sim_timestep": float(model.opt.timestep),
            "policy_hz": policy_hz,
            "n_substeps": int(getattr(self.player, "n_substeps", 0)),
            "control_timestep": (1.0 / policy_hz) if policy_hz > 0.0 else None,
            "nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu),
            # Which qpos slice belongs to which free-joint object. This is what makes
            # `qpos_sim` a complete object-pose record without duplicating it into
            # per-object columns.
            "free_joint_qpos_map": self._free_joint_qpos_map(),
            "recorded_cameras": list(getattr(self, "record_camera_names", []) or []),
            "record_rgb": bool(getattr(self, "record_save_rgb", False)),
            "record_depth": bool(getattr(self, "record_save_depth", False)),
            # The four render flags the live recorder disables. An offline re-render that
            # does not match these produces visibly different images (shadows alone move
            # the mean pixel difference from 0.4 to 13.5).
            "render_flags_disabled": ["SHADOW", "REFLECTION", "HAZE", "SKYBOX"],
        }

    def _resolve_ee_ids(self):
        """Site/body ids for the end-effector pose column.

        The soft-gripper Panda carries a `gripper` site at the TCP and a `hand` body;
        the site is preferred for position and the body supplies the orientation (MjData
        stores no site quaternion). Both are overridable for a scene that names them
        differently. A scene with neither simply records NaN -- the EE pose is derivable
        from `qpos_sim` offline, so a missing site is not worth failing a run over.
        """
        model = self.player.model
        site_name = os.environ.get("INTERVENE_EE_SITE", "gripper")
        body_name = os.environ.get("INTERVENE_EE_BODY", "hand")

        def _lookup(objtype, name):
            try:
                found = mujoco.mj_name2id(model, objtype, name)
                return int(found) if found >= 0 else None
            except Exception:
                return None

        ids = {
            "ee_site_id": _lookup(mujoco.mjtObj.mjOBJ_SITE, site_name),
            "ee_body_id": _lookup(mujoco.mjtObj.mjOBJ_BODY, body_name),
        }
        if ids["ee_site_id"] is None and ids["ee_body_id"] is None:
            print(f"[WARN] No end-effector site '{site_name}' or body '{body_name}' in "
                  "this model; ee_pos/ee_quat will be NaN.")
        return ids

    def _intervention_frame_status(self):
        """Per-frame study columns for HUMAN-CONTROL frames (see LiveReplanSession).

        `policy_step` is the player's frame_idx, which is FROZEN for the whole takeover.
        That is the point: a run of identical policy_step values across human frames is
        the trajectory-level proof that ACT executed nothing while the human had the arm.
        """
        return {
            "policy_step": int(getattr(self.player, "frame_idx", -1)),
            "cell_selected": bool(self.cell_selected),
        }

    def _free_joint_qpos_map(self):
        model = self.player.model
        out = {}
        for jid in range(int(model.njnt)):
            if int(model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_FREE):
                continue
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid) or f"joint_{jid}"
            body_id = int(model.jnt_bodyid[jid])
            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            adr = int(model.jnt_qposadr[jid])
            out[jname] = {"body": bname, "qpos_pos": [adr, adr + 3],
                          "qpos_quat": [adr + 3, adr + 7]}
        return out

    def _finalize_episode(self, outcome, reason="", decided_by="auto", save=True):
        """Close the study episode: save the NPZ, label it, file it by outcome.

        Returns the final path (or None). Safe to call when study mode is off — it then
        only performs the plain save that the previous code did.
        """
        out_path = None
        frames = (self.episode_recorder.frame_count()
                  if self.episode_recorder is not None else 0)
        if save and self.episode_recorder is not None:
            # Stamp the outcome into the NPZ before writing so a stray file is
            # self-describing.
            self.episode_recorder.metadata.update(self._study_metadata())
            self.episode_recorder.metadata["outcome"] = outcome
            self.episode_recorder.metadata["fail_reason"] = reason or ""
            out_path = self._save_episode_recording(mark_saved=True)

        if self.study_session is not None:
            if out_path is not None:
                out_path = self._move_episode_to_outcome(Path(out_path), outcome)
            self.study_session.end_episode(
                outcome, reason=reason, decided_by=decided_by,
                npz_path=out_path, frames=frames,
            )
        return out_path

    def _move_episode_to_outcome(self, npz_path: Path, outcome: str):
        """Move a saved episode (and its .json sidecar) into episodes/<outcome>/."""
        try:
            dest_dir = self.study_session.episode_subdir(outcome)
            dest = dest_dir / npz_path.name
            npz_path.replace(dest)
            sidecar = npz_path.with_suffix(".json")
            if sidecar.exists():
                sidecar.replace(dest.with_suffix(".json"))
            print(f"[Study] episode -> {dest}")
            return str(dest)
        except Exception as exc:
            print(f"[Study][WARN] could not file episode by outcome: {exc}")
            return str(npz_path)

    def _ensure_episode_recorder(self):
        if TrajectoryRecorder is None or self.episode_recorder is not None:
            return self.episode_recorder

        timestamp = f"{time.strftime('%Y%m%d_%H%M%S')}_{int((time.time() % 1.0) * 1000):03d}"
        save_dir = self._episode_output_dir()
        if self.exp_block is not None:
            # Nine cells share one episodes/ directory, so the cell index and the episode
            # number must be in the name: two cells starting an episode inside the same
            # millisecond would otherwise produce the same file. `episode_id` is this
            # stem, which is what makes it globally unambiguous.
            save_path = (save_dir / f"{self.player_mode}_episode"
                                    f"_c{self._session_index:02d}"
                                    f"_e{self.episode_number:04d}_{timestamp}.npz")
        else:
            save_path = save_dir / f"{self.player_mode}_episode_{timestamp}.npz"
        log_hz = float(getattr(getattr(self.player, "config", None), "policy_hz", 60.0))
        self.record_camera_names = _env_list("INTERVENE_RECORD_CAMERAS", "right,left,wrist")
        self.record_rgb_width = int(os.environ.get("INTERVENE_RECORD_RGB_WIDTH", "224"))
        self.record_rgb_height = int(os.environ.get("INTERVENE_RECORD_RGB_HEIGHT", "224"))
        # State-only is the DEFAULT inside an experiment block, and is enforced here
        # rather than only in the launcher: a block started by any other route must
        # still not write frames. Nine cells x 3 cameras x every policy frame is ~17
        # GB/hour and 62% of all GPU render work, and observations are re-rendered
        # offline from the recorded state instead. The sidecar's
        # `render_config` is what makes that reconstruction exact.
        # INTERVENE_RECORD_RGB=1 still opts back in — used to extend the golden corpus.
        self.record_save_rgb = _env_bool("INTERVENE_RECORD_RGB",
                                         self.exp_block is None)
        self.record_save_depth = _env_bool("INTERVENE_RECORD_DEPTH", False)
        self.episode_save_path = save_path
        self.episode_id = save_path.stem
        self.episode_recorder = TrajectoryRecorder(
            save_path=save_path,
            lab_id=f"{self.player_mode}_intervention",
            log_hz=log_hz,
            view_hz=60.0,
            camera_names=self.record_camera_names,
            save_rgb=self.record_save_rgb,
            save_depth=self.record_save_depth,
            rgb_width=self.record_rgb_width,
            rgb_height=self.record_rgb_height,
            metadata={
                "player_mode": self.player_mode,
                "mujoco_xml_path": self.xml_path,
                "mujoco_xml_sha256": self._xml_sha256,
                "reset_npz_path": self.npz_path,
                "checkpoint": (
                    str(self.policy_config.checkpoint)
                    if self.policy_config is not None
                    else None
                ),
                "robot_key": os.environ.get("INTERVENE_ROBOT_KEY", "p4"),
                **self._study_metadata(),
            },
            **self._resolve_ee_ids(),
        )
        scorer = getattr(self.player, "acc_scorer", None)
        if scorer is not None:
            from playback.acc_scorer import COMPONENT_NAMES
            self.episode_recorder.acc_component_names = list(COMPONENT_NAMES)
        print(f"[INFO] Full episode recording will save to: {save_path}")

        if self.study_session is not None:
            randomization = dict(
                getattr(self.scene_randomizer, "last_offsets", {}) or {}
            )
            self.study_session.start_episode(
                episode_number=self.episode_number,
                episode_id=self.episode_id,
                randomization=randomization,
            )
        return self.episode_recorder

    def _capture_recorder_images(self, data):
        if self.viewer is None or self.ctx is None or not self.record_camera_names:
            return {}, {}

        # This is the single largest GPU cost in the whole system: one render+readback per
        # camera, every policy frame, on every session. Measured at 9 windows it is 62% of
        # all render work (270 of ~437 ops/s) and none of these frames are displayed.
        _t0 = time.perf_counter() if self.metrics is not None else 0.0
        glfw.make_context_current(self.window)
        rgbd = self.viewer.render_rgbd_from_cameras(
            data,
            self.ctx,
            self.record_camera_names,
            width=self.record_rgb_width,
            height=self.record_rgb_height,
        )
        if self.metrics is not None:
            self.metrics.observe("recorder_capture_ms", (time.perf_counter() - _t0) * 1000.0)
            self.metrics.incr("recorder_renders", len(self.record_camera_names))
        rgb_by_cam = {}
        depth_by_cam = {}
        for cam_name, frame in rgbd.items():
            if self.record_save_rgb:
                rgb_by_cam[cam_name] = frame["rgb"]
            if self.record_save_depth:
                depth_by_cam[cam_name] = frame["depth"]
        return rgb_by_cam, depth_by_cam

    def _observe_policy_metrics(self, update_t0: float, did_step: bool) -> None:
        """One windowed sample per policy step. Cheap: a few dict appends, no formatting.

        Attributes cost to the three things that actually render on this side, so the
        recorder's share is directly visible instead of inferred:
          recorder_capture_ms  — 3 cams x EVERY frame (observed in _capture_recorder_images)
          observation_ms       — 3 cams x once per chunk (the policy's own input)
          player_update_ms     — everything else in the step (forward pass + mj_step)
        """
        if self.metrics is None:
            return
        try:
            player = self.player
            # App.run() spins much faster than policy_hz. Only sample retained policy
            # timings when a real step occurred; labels and window heartbeats remain live
            # while paused, aligning, intervening, or returning home.
            if did_step:
                self.metrics.observe(
                    "player_update_ms", (time.perf_counter() - update_t0) * 1000.0
                )
                self.metrics.observe(
                    "policy_loop_ms", float(getattr(player, "last_loop_ms", 0.0))
                )
                obs_ms = float(getattr(player, "last_observation_ms", 0.0))
                if obs_ms > 0.0:
                    self.metrics.observe("observation_ms", obs_ms)
                    self.metrics.incr(
                        "observation_renders",
                        int(getattr(player, "last_observation_renders", 0)),
                    )
                self.metrics.incr("policy_steps", 1)

            now_mono = time.monotonic()
            frame_idx = int(getattr(player, "frame_idx", 0))
            if frame_idx != self._metrics_last_frame_idx:
                self._metrics_last_frame_idx = frame_idx
                self._metrics_last_content_mono = now_mono
                self._metrics_content_updates += 1
            self.metrics.update_rate("policy_content", self._metrics_content_updates)
            self.metrics.set_label("mode", str(self.mode))
            self.metrics.set_label("paused", bool(getattr(player, "is_paused", False)))
            self.metrics.set_label("policy_frame_idx", frame_idx)
            self.metrics.set_label(
                "state_change_age_s",
                round(max(0.0, now_mono - self._metrics_last_content_mono), 4),
            )
            self.metrics.set_label("intervention_phase", str(self.intervention_phase))
            self.metrics.set_label("robot_operation", str(self.intervention_operation))
            transition_s = self.last_intervention_transition_s
            if self.mode == "replan" and self.intervention_transition_started_wall > 0.0:
                transition_s = max(0.0, time.time() - self.intervention_transition_started_wall)
            self.metrics.set_label("transition_duration_s", round(transition_s, 4))
            self.metrics.set_label(
                "intervention_failure_reason", str(self.intervention_failed_reason)
            )
            self.metrics.set_label("record_cameras", len(self.record_camera_names))
            self.metrics.set_label("record_rgb", bool(self.record_save_rgb))
            self.metrics.set_label("policy_hz", float(getattr(player.config, "policy_hz", 0.0)))
            if self.metrics.due():
                self.metrics.incr("window_heartbeat", 0)
            row = self.metrics.maybe_emit()
            if row is not None:
                self._print_render_attribution(row)
        except Exception:
            # Instrumentation must never take down a run.
            pass

    def _print_render_attribution(self, row: dict) -> None:
        """Turn one metrics window into the number the recorder decision needs."""
        try:
            counters = row.get("counters") or {}
            elapsed = float(row.get("window_s") or row.get("elapsed_s") or 0.0)
            if elapsed <= 0.0:
                return
            rec = float(counters.get("recorder_renders", 0.0)) / elapsed
            obs = float(counters.get("observation_renders", 0.0)) / elapsed
            total = rec + obs
            if total <= 0.0:
                return
            stats = row.get("latency_ms") or {}
            rec_ms = (stats.get("recorder_capture_ms") or {}).get("avg", 0.0)
            print(
                f"[RenderAttribution] recorder={rec:.1f} renders/s ({100 * rec / total:.0f}% of "
                f"this process, avg {rec_ms:.2f} ms/capture)  observation={obs:.1f} renders/s"
            )
        except Exception:
            pass

    def _record_player_frame(self, force=False):
        if self.episode_recorder is None or self.player is None:
            return

        frame_idx = int(getattr(self.player, "frame_idx", 0))
        if not force and self.last_recorded_player_frame_idx == frame_idx:
            return

        q_real = self._get_paused_mujoco_arm_q()
        qvel = np.asarray(self.player.data.qvel, dtype=np.float64)
        dq_real = qvel[:7] if qvel.shape[0] >= 7 else np.full(7, np.nan, dtype=np.float64)
        grip_width = self.player.get_gripper_width()
        if grip_width is None:
            grip_width = np.nan

        # Policy action + ACC are recorded per autonomous frame. `last_pred_action` is
        # the policy's raw command (before clipping / action-mode mapping), which
        # ctrl_sim does not preserve.
        scorer = getattr(self.player, "acc_scorer", None)
        steps = self._step_counters()
        self.episode_recorder.record(
            now_wall=time.time(),
            data=self.player.data,
            q_real=q_real,
            dq_real=dq_real,
            gripper_width=grip_width,
            intervention=False,
            intervention_id=0,
            force=True,
            image_capture_fn=self._capture_recorder_images,
            policy_action=getattr(self.player, "last_pred_action", None),
            acc=getattr(self.player, "last_acc_score", None),
            acc_components=getattr(self.player, "last_acc_components", None),
            acc_valid=getattr(self.player, "last_acc_valid", False),
            queried_policy=getattr(self.player, "last_queried_policy", False),
            chunk_index=(int(scorer.last_components.get("chunk_age_steps", -1))
                         if scorer is not None else None),
            sim_step=steps.get("simulation_step"),
            policy_step=steps.get("policy_step"),
            control_state=self.intervention_phase,
            cell_selected=bool(self.cell_selected),
            paused=bool(getattr(self.player, "is_paused", False)),
        )
        self.last_recorded_player_frame_idx = frame_idx

    def _save_episode_recording(self, *, mark_saved: bool = True):
        if self.episode_recorder is None:
            return None
        if self.episode_saved and mark_saved:
            return None
        if not self.episode_recorder.has_frames():
            return None
        self.episode_recorder.save()
        if mark_saved:
            self.episode_saved = True
        return str(self.episode_recorder.save_path)

    def _flush_episode_recording(self, reason: str):
        out_path = self._save_episode_recording(mark_saved=False)
        if out_path is not None:
            print(f"[INFO] Flushed continuous episode after {reason}: {out_path}")
        return out_path

    def _save_reset_and_start_new_episode(self, reason: str = "", decided_by: str = "auto"):
        self._end_intervention_for_scene_change("episode success", keep_frames=True)
        self._finalize_episode(OUTCOME_SUCCESS, reason=reason, decided_by=decided_by)
        self.episode_recorder = None
        self.episode_saved = False
        self.episode_save_path = None
        self.last_recorded_player_frame_idx = None
        self.next_intervention_id = 1
        self.active_intervention_id = 0
        self.episode_number += 1
        self._start_new_randomized_scene()
        self._ensure_episode_recorder()
        self._record_player_frame(force=True)
        self._publish_policy_state(force=True)
        print("[SCENE] Task success: saved episode and started a new randomized scene.")

    def _discard_recording_and_restart_current_scene(self, reason: str = "",
                                                    decided_by: str = "auto"):
        self._end_intervention_for_scene_change("episode failure", keep_frames=False)
        # Failures used to be dropped on the floor. For an intervention study they are
        # the most informative episodes, so in study mode they are saved and filed
        # under episodes/failure/ with the evaluator's reason.
        if self.keep_failed_episodes:
            self._finalize_episode(OUTCOME_FAILURE, reason=reason, decided_by=decided_by)
        elif self.study_session is not None:
            self.study_session.end_episode(
                OUTCOME_FAILURE, reason=reason, decided_by=decided_by,
                frames=(self.episode_recorder.frame_count()
                        if self.episode_recorder is not None else 0),
            )
        self.episode_recorder = None
        self.episode_saved = False
        self.episode_save_path = None
        self.last_recorded_player_frame_idx = None
        self.next_intervention_id = 1
        self.active_intervention_id = 0
        if self.keep_failed_episodes:
            # A retried scene is a new episode: it needs its own file and its own
            # scenario/seed record, otherwise attempts overwrite one another.
            self.episode_number += 1

        rotate_ood_scene = (
            bool(self.current_scene_ood)
            and self.scene_attempt >= self.ood_max_scene_attempts
        )
        if rotate_ood_scene:
            # The failed OOD scene has exhausted its exact-retry budget. Start a genuine
            # new scene so the ledger gets another chance to rotate this session back to
            # ID (or assign a fresh OOD draw).
            if not self.keep_failed_episodes:
                self.episode_number += 1
            self._start_new_randomized_scene()
            self._ensure_episode_recorder()
            self._record_player_frame(force=True)
            self._publish_policy_state(force=True)
            kept = "saved" if self.keep_failed_episodes else "discarded"
            print(f"[SCENE] OOD failure: {kept} attempt at scene limit; "
                  f"started new scene id={self.scene_id}.")
            return

        if self.current_scene_ood:
            self.scene_attempt += 1
        self._restart_current_scene()
        self._ensure_episode_recorder()
        self._record_player_frame(force=True)
        kept = "saved" if self.keep_failed_episodes else "discarded"
        print(f"[SCENE] Task failure: {kept} current attempt and restarted the same scene.")

    def _note_task_verdict(self, outcome: str, reason: str, decided_by: str):
        """Remember the last episode outcome so it can be shown on the desktop grid.

        Recorded HERE rather than at either call site because both the auto path
        (`_maybe_auto_finish_task` / the post-release check) and the manual keyboard and
        command paths funnel through these two methods -- so a verdict cannot be reached
        without passing through this.
        """
        self.last_task_outcome = str(outcome)
        self.last_task_reason = str(reason or "")
        self.last_task_decided_by = str(decided_by)
        self.last_task_step = int(getattr(self.player, "frame_idx", 0))
        self.last_task_wall_t = time.time()
        self.last_task_episode = int(getattr(self, "episode_number", 0))

    def mark_task_success(self, reason: str = "", decided_by: str = "manual"):
        if self.mode == "replan":
            self.finish_replan()
        self.success_count += 1
        self._note_task_verdict("success", reason, decided_by)
        self._save_reset_and_start_new_episode(reason=reason, decided_by=decided_by)

    def mark_task_failure(self, reason: str = "", decided_by: str = "manual"):
        if self.mode == "replan":
            self.cancel_replan()
        self.failure_count += 1
        self._note_task_verdict("failure", reason, decided_by)
        self._discard_recording_and_restart_current_scene(
            reason=reason, decided_by=decided_by)

    def _episode_status_text(self):
        return f"Ep {self.episode_number} (S: {self.success_count}, F: {self.failure_count})"

    def _compose_status_text(self, *parts):
        tokens = [self._episode_status_text()]
        tokens.extend(str(part) for part in parts if part)
        return " | ".join(tokens)

    def _observe_task_evaluator(self):
        """Keep the evaluator's latches current during an intervention.

        Observation only — never a decision (see _maybe_auto_finish_task). Wrapped
        because an instrumentation bug must never abort a live takeover with the operator
        holding the arm.
        """
        evaluator = self.task_evaluator
        if evaluator is None or not hasattr(evaluator, "observe"):
            return
        try:
            evaluator.observe(during_intervention=True)
        except Exception as exc:
            print(f"[TASK][WARN] evaluator observation failed during intervention: {exc}")

    def _cancel_intervention_on_deselect(self):
        """Leaving this cell ends a takeover that is still running on it.

        In VR the operator can only drive the arm from single view, so going back to the
        selector means they have walked away -- and `MultiSessionGridManager` broadcasts
        EXIT_SINGLE to every non-selected cell when another is picked. Without this a
        takeover left behind stayed live forever: the episode never resumed, the arm was
        never released, and with sim-MC the simulated arm kept following the controller
        from a cell nobody was looking at. Observed as `RX DESELECTED` x3 with no
        response in `policy_S04_20260811_210618.log`.

        This is the VR counterpart of the desktop grid's selection auto-cancel, which is
        suppressed there under --observe_only precisely because the Quest owns selection
        in these conditions. CANCEL (not finish) matches the desktop, so an abandoned
        takeover does not write half a correction into the episode.

        Two guards, and the second is not optional:
          * nothing to do unless a takeover is actually live;
          * NEVER while a return-home worker is running -- cancel has a second meaning
            there (`_handle_policy_command`) and would strand the arm mid-motion, which
            is the exact bug the desktop version shipped with.
        Idempotent and silent otherwise: EXIT_SINGLE arrives repeatedly.
        """
        if not _env_bool("INTERVENE_CANCEL_ON_DESELECT", True):
            return
        if self.mode != "replan":
            return
        if self._return_home_thread is not None:
            return
        print("[REPLAN] Operator left this cell; cancelling the active intervention.")
        self.cancel_replan()

    def _step_counters(self):
        """Monotonic step counters, for events and per-frame rows.

        `policy_step` is `player.frame_idx`, which advances once per policy action and
        is frozen while a human has the arm. `simulation_step` is derived from
        `data.time` rather than counted, so it stays exact across a model swap (which
        allocates a fresh MjData) and across the intervention path, whose substep count
        differs from the policy path's.
        """
        player = getattr(self, "player", None)
        if player is None:
            return {"simulation_step": None, "policy_step": None}
        sim_step = None
        try:
            timestep = float(player.model.opt.timestep)
            if timestep > 0.0:
                sim_step = int(round(float(player.data.time) / timestep))
        except Exception:
            sim_step = None
        return {
            "simulation_step": sim_step,
            "policy_step": int(getattr(player, "frame_idx", 0)),
            "policy_query_count": int(getattr(player, "policy_query_count", 0)),
        }

    def _run_post_release_check(self) -> bool:
        """Evaluate the task the instant the human releases, BEFORE the policy acts again.

        Why this exists. `_end_intervention_transport` flips `mode` back to `"replay"`
        and unpauses the player on the release keypress; the main loop's replay branch
        then runs `player.update()` (a full forward pass + `n_substeps` of `mj_step`)
        *before* `_maybe_auto_finish_task()`. So the state the detector judged was the
        state after one autonomous action, not the state the human left -- and the
        episode NPZ always ended with at least one post-release policy frame. For a
        recovery study that is the difference between "the human completed it" and "the
        policy finished it", which is exactly the distinction the analysis has to make.

        This is the smallest change that fixes it: the SAME detector, called once, on
        the release tick, before `player.update()`. No task logic is modified.

        Returns True when the episode ended here (the caller must not step the policy
        this tick -- a new scene has already been built).
        """
        # observe() + decide() rather than update(), which is exactly `decide(observe())`.
        # Splitting them costs nothing (no second observation, so the contact streak is not
        # double-advanced) and keeps the OBSERVATION, whose `reason` carries the near-miss
        # detail — `decide` returns a bare None for "not finished yet" and throws that away.
        obs = None
        decision = None
        skipped = None
        if self.mode != "replay":
            skipped = f"mode={self.mode!r}"
        elif self.task_evaluator is None:
            skipped = "no task evaluator"
        elif not getattr(self.task_evaluator, "available", True):
            skipped = "evaluator unavailable (required bodies missing)"
        else:
            try:
                obs = self.task_evaluator.observe()
                decision = self.task_evaluator.decide(obs)
            except Exception as exc:
                skipped = f"raised {exc}"
                print(f"[TASK][WARN] post-release check failed: {exc}")
        status = (decision or {}).get("status") or "none"
        reason = (decision or {}).get("reason", "")

        if self.study_session is not None:
            self.study_session.log_post_release_check(
                status=status, reason=reason,
                intervention_id=int(self.last_finished_intervention_id),
                intervention_outcome=str(self.last_finished_intervention_outcome),
            )

        if status == "success":
            print("[TASK] Post-release check: already successful; not resuming the policy.")
            self.mark_task_success(reason=reason or "post_release", decided_by="auto")
            return True
        if status == "failure":
            print(f"[TASK] Post-release check: already failed ({reason}).")
            self.mark_task_failure(reason=reason or "post_release", decided_by="auto")
            return True

        # ALWAYS say what the check saw. Previously this path printed only on success or
        # failure, so a "not finished yet" verdict was indistinguishable in the log from the
        # check never having run at all — which is exactly how "the detector does not observe
        # the intervened scene" gets diagnosed, wrongly, from a silent log. The near-miss
        # reason comes from the OBSERVATION (e.g. "bar_overlap_low:0.56"), which names the one
        # criterion the human's placement missed and by how much.
        if skipped is not None:
            print(f"[TASK] Post-release check SKIPPED: {skipped}.")
        else:
            detail = (obs or {}).get("reason", "") or "no reason reported"
            print(f"[TASK] Post-release check: not finished ({detail}); policy resumes.")

        if self.study_session is not None:
            self.study_session.log_autonomy_resumed(
                intervention_id=int(self.last_finished_intervention_id),
            )
        return False

    def _maybe_auto_finish_task(self):
        if self.mode != "replay" or self.task_evaluator is None:
            return
        decision = self.task_evaluator.update()
        if decision is None:
            return
        step = decision.get("step", "?")
        reason = decision.get("reason", "")
        if decision["status"] == "success":
            print(f"[TASK] Auto success at step {step}. Starting a new randomized scene.")
            self.mark_task_success(reason=reason, decided_by="auto")
        else:
            reason_text = f" ({reason})" if reason else ""
            print(f"[TASK] Auto failure at step {step}{reason_text}. Restarting current scene.")
            self.mark_task_failure(reason=reason, decided_by="auto")

    def _start_main_loop_watchdog(self):
        """Report main-loop stalls with a full thread dump.

        The VR runtime has had this since 2026-06-16; the policy side never did. That is why
        a 24 s block inside a robot return-home produced no signal at all — the sim froze,
        the desktop cell went black and the point cloud kept streaming identical frames, and
        the only way to find it was to reconstruct the timeline from logs afterwards.
        Set INTERVENE_MAIN_LOOP_WATCHDOG_S=0 to disable.
        """
        try:
            timeout = float(os.environ.get("INTERVENE_MAIN_LOOP_WATCHDOG_S", "3.0"))
        except (TypeError, ValueError):
            timeout = 3.0
        if timeout <= 0.0:
            return

        def _watch():
            stall_reported = False
            while True:
                time.sleep(min(1.0, timeout / 2.0))
                age = time.time() - self._loop_heartbeat[0]
                if age >= timeout:
                    if not stall_reported:
                        print(
                            f"[Watchdog][STALL] policy main loop has not advanced for "
                            f"{age:.1f}s (threshold={timeout:.1f}s). The mirrored sim and "
                            f"the VR point cloud are frozen for this session. Thread stacks:",
                            flush=True,
                        )
                        faulthandler.dump_traceback()
                        sys.stderr.flush()
                        stall_reported = True
                else:
                    if stall_reported:
                        print(f"[Watchdog] policy main loop recovered after {age:.1f}s.",
                              flush=True)
                    stall_reported = False

        threading.Thread(target=_watch, daemon=True, name="PolicyMainLoopWatchdog").start()
        print(f"[Watchdog] policy main-loop watchdog armed ({timeout:.1f}s).")

    def _publish_policy_state(self, *, force: bool = False):
        if self.state_publisher is None or self.player is None:
            return
        if not self.state_publisher.should_publish(force=force):
            return

        self.state_seq += 1
        data = self.player.data
        # Real KT/MC keeps the precise handoff edge: the HUD and grid guards change only
        # after the robot has aligned and control has been handed over. Sim-native MC has no
        # physical alignment phase, however, and its Quest controller discovery can overlap
        # with the always-visible grid's button handling. Claim the intervention as soon as
        # the sim session is created so grid actions cannot consume the same button cluster.
        # A later startup failure still emits the normal false edge via _fail_replan_start.
        intervention_live = (
            self.mode == "replan"
            and self.replan_session is not None
            and (
                bool(getattr(self.replan_session, "started", False))
                or self.sim_mc_active
            )
        )
        message = {
            # v2 adds the model-variant fields below. A consumer that only understands v1
            # treats every session as the base model, which is exactly right for a
            # pose-only task -- so producer and consumers can be deployed separately.
            "version": 2,
            "seq": int(self.state_seq),
            "episode_id": str(self.episode_id),
            "mode": str(self.mode),
            "factr_active": bool(self.factr_active),
            "intervention_phase": str(
                getattr(self.replan_session, "phase", self.intervention_phase)
                if self.mode == "replan"
                else self.intervention_phase
            ),
            "intervention_live": bool(intervention_live),
            # Last episode verdict, for the desktop grid's per-tile banner. Cheap scalars,
            # published every tick like the rest of the mirror -- the grid decides how long
            # to keep showing it (see VERDICT_HOLD_S there).
            "task_outcome": str(getattr(self, "last_task_outcome", "")),
            "task_reason": str(getattr(self, "last_task_reason", "")),
            "task_decided_by": str(getattr(self, "last_task_decided_by", "")),
            "task_step": int(getattr(self, "last_task_step", 0)),
            "task_wall_t": float(getattr(self, "last_task_wall_t", 0.0)),
            "success_count": int(getattr(self, "success_count", 0)),
            "failure_count": int(getattr(self, "failure_count", 0)),
            "paused": bool(getattr(self.player, "is_paused", False)),
            "frame_idx": int(getattr(self.player, "frame_idx", 0)),
            "sim_t": float(data.time),
            "wall_t": time.time(),
            "qpos": np.asarray(data.qpos, dtype=np.float64).tolist(),
            "qvel": np.asarray(data.qvel, dtype=np.float64).tolist(),
            "ctrl": np.asarray(data.ctrl, dtype=np.float64).tolist(),
            # Policy uncertainty, 0..1. Still published: it drives the grid tile border
            # colour and the Quest risk bar. It no longer drives any auto-pause.
            "acc_risk": float(getattr(self.player, "last_acc_score", 0.0)),
            # Monotonic counter, not a flag: the VR runtime fires the "intervention
            # unavailable" HUD on any INCREASE. A counter means repeated failures each
            # notify the operator, and there is no edge-state that could get stuck on.
            "intervention_failed_seq": int(self.intervention_failed_seq),
            "intervention_failed_reason": str(self.intervention_failed_reason),
            # --- model variant (v2) ------------------------------------------------------
            # Which compiled model produced this qpos. Consumers render from their OWN
            # compiled copy, so without this they would render the right numbers on the
            # wrong geometry -- and nothing would report it, because nq/nv/nu are identical
            # across variants by construction.
            "model_epoch": int(self.model_epoch),
            "variant_key": str(self.model_variant_key),
            # The FULL descriptor, not just the key, so a consumer is self-contained: it
            # never needs OOD_SCENE_DIR, never has to agree with us about episode_seed, and
            # cannot desync if the launcher passed it different OOD env. ~350 bytes against
            # a message that already carries nq+nv+nu float64s as JSON.
            "variant": self.model_variant,
            # Next episode's likely variant, so a consumer can compile it during THIS
            # episode instead of stalling at the boundary.
            "next_variant": self.next_variant_hint,
            # Identity of the base geometry (scene file + include closure). A consumer
            # launched on a different scene must refuse loudly rather than render a
            # plausible-looking lie.
            "base_scene_sha256": str(self._scene_closure_sha256),
        }
        # Published on EVERY message, not on change: the state sockets are CONFLATE=1, so a
        # late-joining subscriber only ever sees the newest message. "Publish on change"
        # would leave a fresh subscriber with no descriptor at all.
        self.state_publisher.publish(message)

    def _get_paused_mujoco_arm_q(self):
        qpos_indices = getattr(self.player, "qpos_indices", None)
        if qpos_indices is not None:
            return np.asarray(
                [self.player.data.qpos[idx] for idx in qpos_indices],
                dtype=np.float64,
            )

        return np.asarray(self.player.data.ctrl[:7], dtype=np.float64)

    def current_player_grip_seed(self):
        return self.player.get_gripper_width()

    def toggle_mirror(self):
        if self.mode != "replay":
            print("[MIRROR] Mirror toggle only allowed in replay mode.")
            return

        # Mirroring streams sim joint targets to a follower via send_joint_positions.
        # FACTR is a LEADER — its adapter deliberately has no such method (MuJoCo follows
        # FACTR, not the reverse), so mirroring would raise AttributeError mid-stream.
        # Only short-circuits when an adapter actually exists but cannot be mirrored; a
        # missing adapter keeps its original path through MirrorController.toggle().
        _mirror_robot = getattr(self.mirror_controller, "robot", None)
        if self.factr_active or (
            _mirror_robot is not None
            and not hasattr(_mirror_robot, "send_joint_positions")
        ):
            print("[MIRROR] Mirroring is unavailable with this robot backend "
                  "(FACTR is a leader; the simulation follows it).")
            return

        grip_seed = self.current_player_grip_seed()
        q_seed = self._get_paused_mujoco_arm_q()
        ok = self.mirror_controller.toggle(
            grip_width_seed=grip_seed,
            initial_q_target=q_seed,
        )
        if not ok and not self.mirror_controller.enabled:
            print("[MIRROR] Mirror remains OFF.")

    def _robot_state_snapshot(self):
        """Arm/gripper state for the intervention record. Never raises."""
        try:
            q = self._get_paused_mujoco_arm_q()
            qvel = np.asarray(self.player.data.qvel, dtype=np.float64)
            return {
                "q": [float(v) for v in np.asarray(q, dtype=np.float64).reshape(-1)[:7]],
                "dq": [float(v) for v in qvel[:7]] if qvel.shape[0] >= 7 else [],
                "gripper_width": (float(self.player.get_gripper_width())
                                  if self.player.get_gripper_width() is not None else None),
                "ctrl": [float(v) for v in
                         np.asarray(self.player.data.ctrl, dtype=np.float64)[:8]],
                "frame_idx": int(getattr(self.player, "frame_idx", 0)),
                "sim_t": float(self.player.data.time),
            }
        except Exception as exc:
            return {"error": str(exc)}

    def _intervention_mode_label(self):
        # The only place the study's intervention mode string is produced.
        # FACTR is checked first: the launcher forbids FACTR_ACTIVE=1 with MC_ACTIVE=1,
        # so the two can never both be set on a correctly-launched run.
        if self.factr_active:
            return "factr"
        if self.sim_mc_active:
            return "sim_mc"
        if self.mc_active:
            return "motion_controller"
        return "telekinesis"

    def _acquire_robot_ownership(self):
        """Take exclusive ownership of the physical arm for this intervention.

        Contends on the same lock file as the VR runtime's RobotOwnershipLock, so a
        multi-window run cannot have two processes driving one arm. Never raises: if the
        lock machinery itself is broken we log and allow the attempt, because refusing all
        interventions would be a worse failure than the race it guards against.
        """
        if self.robot_lock is None:
            return True, "robot lock disabled"
        try:
            # Ask a busy holder to yield instead of refusing immediately. A holder that is
            # merely driving back to its initial pose stops early and hands the arm over; a
            # holder with a human on the arm ignores the request and we refuse exactly as
            # before, after the short wait.
            wait_s = float(os.environ.get("INTERVENE_ROBOT_YIELD_WAIT_S", "2.5"))
            return self.robot_lock.acquire_with_yield(wait_s=wait_s, reason="intervene")
        except Exception as exc:
            print(f"[REPLAN][WARN] Robot ownership lock unavailable ({exc}); proceeding without it.")
            return True, "lock unavailable"

    def _release_robot_ownership(self):
        if self.robot_lock is None:
            return
        try:
            self.robot_lock.release()
        except Exception as exc:
            print(f"[REPLAN][WARN] Could not release robot ownership: {exc}")

    def _note_intervention_failed(self, reason: str, detail: str = ""):
        """Record an intervention that could not start, so the headset can say so.

        The operator's only feedback used to be a gRPC traceback in a per-session desktop
        log they cannot see while wearing the headset. Bumping this counter makes the VR
        runtime publish SimPub/Status/intervention = "intervention_failed", which
        InterventionStatusHud turns into a visible message.
        """
        self.intervention_failed_seq += 1
        self.intervention_failed_reason = str(reason)
        self.last_intervention_failure_wall = time.time()
        if detail:
            print(f"[REPLAN] Intervention unavailable ({reason}): {detail}")

    @staticmethod
    def _transition_reason(exc, fallback="transition_failed"):
        reason = str(getattr(exc, "reason", "") or fallback)
        allowed = {
            "robot_unreachable", "robot_busy", "alignment_command_failed",
            "state_read_failed", "mode_switch_failed", "return_home_failed",
            "transition_failed",
        }
        return reason if reason in allowed else fallback

    @staticmethod
    def _close_robot_adapter(adapter):
        if adapter is None or not hasattr(adapter, "close"):
            return
        try:
            adapter.close()
        except Exception as exc:
            print(f"[REPLAN][WARN] Could not close failed robot connection: {exc}")

    @staticmethod
    def _release_robot_adapter_without_homing(adapter):
        """Release an already-returned adapter without issuing a second home."""
        if adapter is None:
            return
        release = getattr(adapter, "release_without_homing", None)
        if callable(release):
            try:
                release()
                print("[REPLAN] Released returned robot connection without implicit home.")
                return
            except Exception as exc:
                print(f"[REPLAN][WARN] No-home robot release failed: {exc}")
        # Non-record backends do not expose the same safe detach operation. Leave
        # their existing close semantics intact rather than guessing at ownership.

    def _clear_replan_state(self):
        self.replan_session = None
        self.replan_cut_idx = None
        self.replan_original_path = None
        self.replan_should_resume_mirror = False
        self.active_intervention_id = 0
        self.replan_recorder_start_len = None
        self.replan_start_snapshot = None
        self.replan_evaluator_snapshot = None

    def _end_intervention_for_scene_change(self, why: str, *, keep_frames: bool) -> None:
        """A scene change while a takeover is live implicitly ends that takeover.

        Nothing may replace the model/data under a LiveReplanSession that holds
        references to them (see _swap_model's guard). Previously the scene-change paths
        simply had no guard, so the swap was silently refused and the episode ran the
        base model while still stamped OOD.

        `keep_frames` is not cosmetic: cancel_replan truncates the recorder back to the
        pre-intervention length, which would delete the human's frames out of an
        otherwise SUCCESSFUL episode. Reads only mode/replan_session — several contract
        tests drive these paths on a partially-constructed App.
        """
        # getattr on BOTH: the scene-lifecycle contract tests drive these paths on an
        # App.__new__ that sets only the scene/variant fields it cares about.
        if getattr(self, "mode", None) != "replan" and getattr(self, "replan_session", None) is None:
            return
        verb = "finishing" if keep_frames else "cancelling"
        print(f"[SCENE] Auto-{verb} the active intervention ({why}).")
        if keep_frames:
            self.finish_replan()
        else:
            self.cancel_replan()
        # A re-intervention queued against the OLD episode must not fire on a new scene.
        self._pending_reintervene = False

    def _resume_task_clock(self):
        evaluator = getattr(self, "task_evaluator", None)
        if evaluator is not None and hasattr(evaluator, "resume_clock"):
            evaluator.resume_clock()

    def _fail_replan_start(self, reason, detail, *, adapter=None):
        """Emit one failure edge, then restore the exact pre-attempt transport state."""
        self.intervention_phase = "failed"
        self.intervention_operation = str(reason)
        self._note_intervention_failed(reason, detail=detail)
        self._publish_policy_state(force=True)
        if self.study_session is not None:
            try:
                self.study_session.finish_intervention(
                    outcome="aborted",
                    robot_state=self._robot_state_snapshot(),
                    reason=str(reason),
                    failure_detail=str(detail),
                )
            except Exception as exc:
                print(f"[REPLAN][WARN] Could not close failed study intervention: {exc}")
        self._close_robot_adapter(adapter)
        self._release_robot_ownership()
        self._clear_replan_state()
        self._resume_task_clock()
        self.mode = "replay"
        if self._replan_paused_policy and getattr(self.player, "is_paused", False):
            if hasattr(self.player, "resume_from_current_state"):
                self.player.resume_from_current_state(reset_policy_queue=True)
            else:
                self.player.toggle_pause()
        self.last_recorded_player_frame_idx = None
        self._replan_paused_policy = False
        self.last_intervention_transition_s = max(
            0.0, time.time() - self.intervention_transition_started_wall
        )
        self.intervention_phase = "policy_resumed"
        self.intervention_operation = "idle"
        self._publish_policy_state(force=True)
        print("[REPLAN] Intervention cleanup complete; policy transport state restored.")

    def _settle_replan_terminal_failure(self, reason, detail, *, adapter=None):
        """Leave a failed finish/cancel usable when return-home cannot be launched."""
        self.intervention_phase = "failed"
        self.intervention_operation = str(reason)
        self._note_intervention_failed(reason, detail=detail)
        self._publish_policy_state(force=True)
        self._close_robot_adapter(adapter)
        self._release_robot_ownership()
        self._return_home_thread = None
        self._return_home_context = None
        self._return_home_done.clear()
        self._return_home_cancel.clear()
        self._clear_replan_state()
        self._resume_task_clock()
        self.mode = "replay"
        if hasattr(self.player, "resume_from_current_state"):
            self.player.resume_from_current_state(reset_policy_queue=True)
        elif getattr(self.player, "is_paused", False):
            self.player.toggle_pause()
        self.last_recorded_player_frame_idx = None
        self._replan_paused_policy = False
        self.last_intervention_transition_s = max(
            0.0, time.time() - self.intervention_transition_started_wall
        )
        self.intervention_phase = "policy_resumed"
        self.intervention_operation = "idle"
        self._publish_policy_state(force=True)
        print("[REPLAN] Terminal transition cleanup complete; policy resumed.")

    def _start_return_home_or_settle(self, robot_adapter, *, outcome, should_resume_mirror):
        try:
            # MUST run first: it reads self.replan_session (to mark its phase) and
            # self.player.get_gripper_width(), both of which the transport end below
            # invalidates.
            self._begin_return_home(
                robot_adapter,
                outcome=outcome,
                should_resume_mirror=should_resume_mirror,
            )
        except Exception as exc:
            # Already restores transport and resumes the policy itself.
            self._settle_replan_terminal_failure(
                "transition_failed", str(exc), adapter=robot_adapter
            )
            return
        self._end_intervention_transport(outcome=outcome)

    def _alignment_heartbeat(self):
        """Keep the world alive during a blocking transition step.

        The intervention start is deliberately synchronous (it mutates the live MjData),
        so the only way consumers survive it is to publish from inside it. Main thread
        only — PolicyStatePublisher is single-threaded by contract.

        Feeding `_loop_heartbeat` here is intentional: a legitimate multi-second
        alignment is not a hang, and it used to produce a watchdog stack dump every time.
        LiveReplanSession._tick prints a ~1 Hz progress line so the window stays visible.
        """
        try:
            self._loop_heartbeat[0] = time.time()
            self._publish_policy_state(force=True)
        except Exception as exc:
            print(f"[REPLAN][WARN] transition heartbeat failed: {exc}")

    def trigger_replan(self):
        if self.mode != "replay":
            print("[REPLAN] Already in replanning mode.")
            return

        # The intervention is over the moment the operator presses finish/cancel, so
        # `mode` is already "replay" while the arm is still homing in the background.
        # That means this can now be reached DIRECTLY (SPACE in input/callbacks.py, which
        # only gates on mode) while the worker owns the adapter. Queue instead of racing
        # it — same semantics _handle_policy_command used to provide, moved to the source
        # so every caller gets it.
        if self._return_home_thread is not None:
            self._pending_reintervene = True
            self._return_home_cancel.set()
            print("[REPLAN] Queued a new intervention; stopping the current return-home transition.")
            return

        self.intervention_transition_started_wall = time.time()
        self.intervention_operation = "requested"

        # One physical robot, one owner. Acquire before pausing the policy so a busy arm
        # cannot perturb the selected session at all.
        # FACTR is exempt: it drives the simulated arm, and its single serial device is
        # already arbitrated across every policy process by FactrRpcService's own owner
        # field. Taking the polymetis lock as well could only invent a false conflict.
        if not self.sim_mc_active and not self.factr_active:
            acquired, lock_msg = self._acquire_robot_ownership()
            if not acquired:
                self._replan_paused_policy = False
                self._fail_replan_start("robot_busy", lock_msg)
                return
        # Checked outside the lock block because FACTR skips the lock but still needs a
        # constructed adapter (FactrRpcAdapter). Only sim_mc builds its driver on demand
        # and legitimately has none.
        if not self.sim_mc_active and not self.robot_features_available:
            self._replan_paused_policy = False
            self._fail_replan_start(
                "transition_failed", "robot features are unavailable"
            )
            return

        # Phase 1 — the operator asked to take over. Timed before any work happens so
        # synchronization time includes everything the human actually waits for.
        if self.study_session is not None:
            self.study_session.start_intervention(
                mode=self._intervention_mode_label(),
                frame_idx=int(getattr(self.player, "frame_idx", 0)),
            )

        paused_for_replan = False
        if not self.player.is_paused:
            if hasattr(self.player, "pause_at_current_state"):
                self.player.pause_at_current_state()
            else:
                self.player.toggle_pause()
            paused_for_replan = True
            print("[REPLAN] Policy paused. Moving real robot to current MuJoCo pose...")

        # Phase 2 — the policy is stopped and the robot is holding position.
        if self.study_session is not None:
            self.study_session.mark_intervention_paused(
                paused_here=paused_for_replan,
                frame_idx=int(getattr(self.player, "frame_idx", 0)),
            )
        self._replan_paused_policy = paused_for_replan

        # if not self.mirror_controller.enabled:
        #     print("[REPLAN] Activate mirroring first with M, then press P.")
        #     return

        cut_idx = int(self.player.frame_idx)
        qpos_seed = self.player.data.qpos.copy()
        qvel_seed = self.player.data.qvel.copy()

        # IMPORTANT: use actual paused MuJoCo arm state, not ctrl[:7]
        # q_seed = self.player.data.ctrl[:7].copy()
        q_seed = self._get_paused_mujoco_arm_q()


        grip_width_seed = self.player.get_gripper_width()
        finger_seed = 0.0
        if self.player.model.nu >= 8:
            finger_seed = float(self.player.data.ctrl[7])

        if self.sim_mc_active:
            # Sim-native MC: no real robot needed at all. Skip robot_features_available
            # entirely — it's only about REAL hardware adapter construction.
            # Identity check, not just `is None`: PolicyPlayer.swap_model allocates a
            # brand-new MjData for an OOD variant episode, and a cached driver bound to
            # the old one drives a dead sim — the arm never moves and the recording is a
            # frozen pose. _swap_model rebinds it too; this is the belt-and-braces catch
            # for any future path that replaces player.data without going through there.
            if (
                self.sim_mc_driver is None
                or getattr(self.sim_mc_driver, "model", None) is not self.player.model
                or getattr(self.sim_mc_driver, "data", None) is not self.player.data
            ):
                from robot.sim_mc_driver import SimMcDriver

                if self.sim_mc_driver is None:
                    self.sim_mc_driver = SimMcDriver(
                        model=self.player.model, data=self.player.data
                    )
                else:
                    print("[REPLAN][MC-SIM] Model/data changed; rebinding the sim-MC driver.")
                    self.sim_mc_driver.rebind(self.player.model, self.player.data)
            robot_adapter = None
            already_connected = True
            self.replan_should_resume_mirror = False
        else:
            # If mirror is active, reuse its robot connection.
            # If not, create/connect a fresh robot adapter.
            self.replan_should_resume_mirror = bool(self.mirror_controller.enabled)
            try:
                if self.mirror_controller.enabled:
                    robot_adapter = self.mirror_controller.detach_for_reuse()
                    already_connected = True
                else:
                    robot_adapter = self.mirror_controller.robot
                    if robot_adapter is None:
                        raise RuntimeError("no robot adapter is configured")
                    # Blocking gRPC connect. Pump the heartbeat around it so consumers
                    # do not age out while a slow/unreachable arm is dialled.
                    self._alignment_heartbeat()
                    try:
                        robot_adapter.connect(already_connected=False)
                    finally:
                        self._alignment_heartbeat()
                    already_connected = True
            except Exception as e:
                self.replan_should_resume_mirror = False
                print(f"[REPLAN] Could not start intervention: {e}")
                self._fail_replan_start(
                    "robot_unreachable", str(e), adapter=locals().get("robot_adapter")
                )
                return

        if self.episode_recorder is not None:
            self.episode_recorder.reset_stride()
            self.replan_recorder_start_len = self.episode_recorder.frame_count()
        else:
            self.replan_recorder_start_len = None

        self.replan_start_snapshot = {
            "qpos": self.player.data.qpos.copy(),
            "qvel": self.player.data.qvel.copy(),
            "ctrl": self.player.data.ctrl.copy(),
            "time": float(self.player.data.time),
            "frame_idx": int(getattr(self.player, "frame_idx", 0)),
        }
        # The evaluator now observes DURING the takeover, so a cancel — which rolls the
        # world back to the snapshot above — must roll its latches back too, or they
        # describe a grasp/lift that no longer happened.
        self.replan_evaluator_snapshot = None
        if self.task_evaluator is not None and hasattr(self.task_evaluator, "snapshot_latches"):
            try:
                self.replan_evaluator_snapshot = self.task_evaluator.snapshot_latches()
                self.task_evaluator.pause_clock()
            except Exception as exc:
                print(f"[TASK][WARN] could not snapshot evaluator state: {exc}")
        self.active_intervention_id = int(self.next_intervention_id)

        suffix_path = str(
            Path(self.player.npz_path).with_name(f"suffix_{int(time.time())}.npz")
        )

        self.replan_session = LiveReplanSession(
            robot_adapter=robot_adapter,
            save_path=str(self.episode_save_path or suffix_path),
            mujoco_lab_id="replan",
            mujoco_xml_path=self.player.xml_path,
            q_hold=q_seed,
            finger_hold=finger_seed,
            grip_width_hold=grip_width_seed,
            qpos_seed=qpos_seed,
            qvel_seed=qvel_seed,
            view_hz=60.0,
            log_hz=60.0,
            alpha=0.6,
            already_connected=already_connected,
            model=self.player.model,
            data=self.player.data,
            recorder=self.episode_recorder,
            image_capture_fn=self._capture_recorder_images,
            intervention_id=self.active_intervention_id,
            mc_mode=(True if self.sim_mc_active else self.mc_active),
            sim_mc_driver=(self.sim_mc_driver if self.sim_mc_active else None),
            on_ready=self._on_intervention_ready,
            on_released=self._on_intervention_released,
            on_tick=self._alignment_heartbeat,
            frame_status_provider=self._intervention_frame_status,
        )

        # The robot answered, so any earlier failure is stale — clear the backoff so a
        # later genuine failure is reported immediately rather than being swallowed.
        self.last_intervention_failure_wall = 0.0
        self.intervention_failed_reason = ""

        self.replan_cut_idx = cut_idx
        self.replan_original_path = str(self.player.npz_path)
        self.mode = "replan"
        self.intervention_phase = "requested"
        self._publish_policy_state(force=True)
        print(f"[REPLAN] Entered replanning mode in current window (intervention_id={self.active_intervention_id}).")

        # Canonical 1d24f5e behavior: connection/alignment/mode handoff completes on the
        # command thread before normal updates resume. Return-home remains asynchronous.
        try:
            self.intervention_operation = "aligning"
            self.replan_session.start()
            self.intervention_phase = "human_control"
            self.intervention_operation = "human_control"
            self.last_intervention_transition_s = max(
                0.0, time.time() - self.intervention_transition_started_wall
            )
            self._publish_policy_state(force=True)
        except Exception as exc:
            reason = self._transition_reason(exc)
            print(f"[REPLAN] Intervention startup failed: {exc}")
            self._fail_replan_start(reason, str(exc), adapter=robot_adapter)

    def _on_intervention_ready(self, **fields):
        """Phase 3 callback from LiveReplanSession: robot aligned at the takeover pose."""
        if self.study_session is not None:
            self.study_session.mark_intervention_ready(**fields)

    def _on_intervention_released(self, **fields):
        """Phase 4 callback: control handed to the human."""
        # Gates the post-release check: a takeover that failed during alignment never
        # changed the world, so the pre-existing resume behaviour is left exactly as it
        # was for that case.
        self._intervention_reached_human_control = True
        if self.study_session is not None:
            self.study_session.mark_intervention_released(
                robot_state=self._robot_state_snapshot(), **fields
            )

    def _return_home_interrupt_requested(self) -> bool:
        return self._return_home_cancel.is_set()

    def _resolve_return_home_q_target(self):
        """Derive the 7-joint return-home target from the scene's initial qpos.

        MUST be called on the main thread: it reads `self.base_scene_qpos` and
        `self.player.qpos_indices`, both of which a scene change or a model swap can
        replace (`PolicyPlayer.swap_model` allocates a whole new MjData). The return-home
        worker used to read them itself, which was safe only while the main loop was
        frozen for the duration. Returns None when there is nothing to return to.
        """
        qpos = self.base_scene_qpos
        if qpos is None:
            return None
        try:
            qpos = np.asarray(qpos, dtype=np.float64)
            qpos_indices = getattr(self.player, "qpos_indices", None)
            if qpos_indices is not None:
                q_target = qpos[qpos_indices]
            else:
                q_target = qpos[:7]
            return np.asarray(q_target, dtype=np.float64).reshape(7)
        except Exception as exc:
            print(f"[REPLAN] Franka return skipped; invalid initial pose: {exc}")
            return None

    def _resolve_return_home_grip_width(self):
        """Scene-initial gripper width for the return. Main thread only (reads player.model)."""
        if self.base_scene_ctrl is None or len(self.base_scene_ctrl) < 8:
            return None
        try:
            return gripper_ctrl_to_width(self.player.model, self.base_scene_ctrl[7])
        except Exception as exc:
            print(f"[REPLAN] Franka initial gripper width unavailable: {exc}")
            return None

    def _return_franka_to_initial_pose(
        self, robot_adapter, *, q_target=None, restore_grip_width=None
    ) -> bool:
        # `q_target` / `restore_grip_width` are resolved by the CALLER on the main thread
        # (see _begin_return_home). They stay optional so close()'s direct, main-thread
        # call keeps working unchanged; when omitted they are derived here as before.
        #
        # False now means "did not reach the initial pose", which covers BOTH an operator
        # interrupt and an outright failure. Only the interrupt should re-enter an
        # intervention (finish_replan), so the two are tracked separately — otherwise a
        # robot that cannot move would loop straight back into a new takeover.
        self._return_home_interrupted = False
        if robot_adapter is None:
            return True
        # FACTR has no scene pose to return to: the leader arm is not in the scene, and
        # its joints are not the Franka's. Releasing the service IS the return — the
        # service parks the leader at its init pose in its own process
        # (INTERVENE_FACTR_RETURN_INIT_ON_RELEASE), off the policy's main loop.
        # Without this the generic path below would fall through to go_to_joint_positions
        # -> align_to_mujoco on an already-released client, fail, and log every single
        # FACTR intervention as return_home_failed. getattr, never a direct attribute
        # access: this must not fire before the `robot_adapter is None` check above.
        if getattr(robot_adapter, "is_factr_adapter", False):
            try:
                robot_adapter.close()
            except Exception as exc:
                print(f"[FACTR] Release warning: {exc}")
            return True
        if q_target is None:
            q_target = self._resolve_return_home_q_target()
        if q_target is None:
            print("[REPLAN] Franka return skipped; no initial scene pose is available.")
            return True
        q_target = np.asarray(q_target, dtype=np.float64).reshape(7)

        # An intervention leaves the arm in HUMAN_CONTROL (freedrive), or in
        # CARTESIAN_IMPEDANCE_CONTROL under MC. NEITHER policy exposes a `q_desired`
        # parameter, so streaming joint positions into them fails on every single step with
        # `RuntimeError: KeyError: q_desired`. FrankaArm.apply_commands swallows that error
        # and calls reconnect(), which re-sends the same freedrive policy and blocks — so the
        # 4 s ramp became a ~24 s main-loop stall that froze the sim, blacked out the desktop
        # cell and left the arm exactly where the human let go. Measured 2026-07-30: 240
        # failures, one per step.
        #
        # Switch into a joint-position mode FIRST. This is also what makes the post-return
        # switch_control_mode in finish_replan/cancel_replan redundant rather than corrective.
        control_mode = os.environ.get(
            "INTERVENE_ROBOT_CONTROL_MODE", "HYBRID_JOINT_IMPEDANCE_CONTROL"
        )
        # The cancel flag used to be read only inside the ramp below, so a preempt arriving
        # during the (blocking) mode switch or the gripper restore was not honoured until
        # that call returned. Each of these checks bounds how long a yielding session can
        # keep the arm after another session has asked for it.
        if self._return_home_interrupt_requested():
            self._return_home_interrupted = True
            return False
        if hasattr(robot_adapter, "switch_control_mode"):
            try:
                print(f"[REPLAN] Leaving human control -> {control_mode} before returning home.")
                robot_adapter.switch_control_mode(control_mode)
            except Exception as exc:
                print(f"[REPLAN] Could not switch out of human control: {exc}. "
                      "Skipping return-home; the arm stays where it is.")
                return False

        try:
            seconds = float(os.environ.get("INTERVENE_RETURN_INITIAL_SECONDS", "4.0"))
            speed_factor = float(os.environ.get("INTERVENE_RETURN_INITIAL_SPEED_FACTOR", "0.25"))
            print(
                f"[REPLAN] Returning Franka to initial scene pose "
                f"(seconds={seconds:.1f}, speed_factor={speed_factor:.2f})."
            )
            if hasattr(robot_adapter, "get_joint_positions") and hasattr(robot_adapter, "send_joint_positions"):
                hz = float(os.environ.get("INTERVENE_RETURN_INITIAL_HZ", "60.0"))
                dt = 1.0 / max(1.0, hz)
                n_steps = max(1, int(round(max(0.0, seconds) * hz)))
                # Read the start pose AFTER the mode switch: switching re-seeds the
                # controller's q_desired at the current pose, so a pose captured before it
                # would make the ramp start from a stale point.
                q_start = np.asarray(robot_adapter.get_joint_positions(), dtype=np.float64).reshape(7)
                delta = q_target - q_start
                # Bound the damage if the robot is unhealthy: each failed send costs a
                # blocking reconnect, so running all 240 steps into a dead arm is what turned
                # a robot problem into a whole-session rendering freeze.
                deadline = time.time() + max(2.0, seconds * 3.0)
                for i in range(1, n_steps + 1):
                    if self._return_home_interrupt_requested():
                        self._return_home_interrupted = True
                        return False
                    if time.time() > deadline:
                        print(f"[REPLAN] Return-home exceeded {seconds * 3.0:.1f}s "
                              f"(step {i}/{n_steps}); aborting so the main loop keeps running.")
                        return False
                    u = i / n_steps
                    smooth = u * u * (3.0 - 2.0 * u)
                    q_cmd = q_start + smooth * delta
                    if i < n_steps:
                        smooth_dot = 6.0 * u * (1.0 - u) / max(seconds, dt)
                        qd_cmd = smooth_dot * delta
                    else:
                        qd_cmd = np.zeros_like(delta)
                    robot_adapter.send_joint_positions(q_cmd, qd_cmd=qd_cmd)
                    time.sleep(dt)
            elif hasattr(robot_adapter, "go_to_joint_positions"):
                # Single blocking call with no cancel hook, so the only place a preempt can
                # be honoured is before it starts. The polymetis adapter has
                # send_joint_positions, so in practice this is a fallback for other
                # backends; the check bounds the worst case rather than eliminating it.
                if self._return_home_interrupt_requested():
                    self._return_home_interrupted = True
                    return False
                robot_adapter.go_to_joint_positions(q_target, max_vel_norm_factor=speed_factor)
            else:
                if self._return_home_interrupt_requested():
                    self._return_home_interrupted = True
                    return False
                robot_adapter.send_joint_positions(q_target)
        except Exception as exc:
            print(f"[REPLAN] Franka return warning: {exc}")
            return False

        # Verify rather than assume. `send_joint_positions` cannot fail loudly — FrankaArm
        # swallows controller errors — so the only way to know the arm actually moved is to
        # look. Callers branch on this value to decide whether to re-attach the mirror or
        # restore hold mode; it previously returned True even when nothing moved at all.
        try:
            q_final = np.asarray(robot_adapter.get_joint_positions(), dtype=np.float64).reshape(7)
            err = float(np.linalg.norm(q_final - q_target))
            tol = float(os.environ.get("INTERVENE_RETURN_INITIAL_TOL", "0.10"))
            if err > tol:
                print(f"[REPLAN] Return-home did NOT reach the initial pose "
                      f"(err={err:.4f} rad > tol={tol:.2f}). The arm is not where the next "
                      f"episode expects it.")
                return False
            print(f"[REPLAN] Franka back at initial scene pose (err={err:.4f} rad).")
        except Exception as exc:
            print(f"[REPLAN] Could not verify return-home pose: {exc}")
            return False

        if self._return_home_interrupt_requested():
            # The arm is already at the initial pose; only the gripper restore remains.
            # Yield now rather than make the requesting session wait on it.
            self._return_home_interrupted = True
            return False

        grip_width = restore_grip_width
        if grip_width is None:
            grip_width = self._resolve_return_home_grip_width()
        if grip_width is not None and hasattr(robot_adapter, "restore_gripper_width"):
            try:
                robot_adapter.restore_gripper_width(grip_width)
            except Exception as exc:
                print(f"[REPLAN] Franka gripper return warning: {exc}")
        return True

    def _return_home_worker(self):
        context = self._return_home_context or {}
        robot_adapter = context.get("robot_adapter")
        try:
            # Everything the return needs was resolved on the main thread in
            # _begin_return_home. The worker must not read base_scene_qpos / player.* —
            # the main loop keeps running now and can swap the model underneath it.
            completed = self._return_franka_to_initial_pose(
                robot_adapter,
                q_target=context.get("q_target"),
                restore_grip_width=context.get("restore_grip_width"),
            )
            if completed and context.get("should_resume_mirror"):
                self.mirror_controller.attach_reused_robot(
                    robot_adapter,
                    grip_width_seed=context.get("grip_width_seed"),
                    force_reconnect=True,
                )
            elif (
                completed
                and robot_adapter is not None
                and not getattr(robot_adapter, "is_factr_adapter", False)
                and hasattr(robot_adapter, "switch_control_mode")
            ):
                # Skipped for FACTR: _return_franka_to_initial_pose already closed the
                # client, and FactrRpcAdapter.switch_control_mode maps any non-takeover
                # mode onto close() anyway, so this would be a no-op second release.
                control_mode = os.environ.get(
                    "INTERVENE_ROBOT_CONTROL_MODE",
                    "HYBRID_JOINT_IMPEDANCE_CONTROL",
                )
                robot_adapter.switch_control_mode(control_mode)
            self._return_home_result = bool(completed)
        except Exception as exc:
            self._return_home_error = exc
            self._return_home_result = False
        finally:
            self._return_home_done.set()

    def _begin_return_home(self, robot_adapter, *, outcome: str, should_resume_mirror: bool):
        if self._return_home_thread is not None:
            print("[REPLAN] Return-home transition is already active.")
            return
        self.intervention_phase = "returning_home"
        self.intervention_operation = "returning_home"
        if self.replan_session is not None:
            self.replan_session.phase = "returning_home"
        self._return_home_context = {
            "robot_adapter": robot_adapter,
            "outcome": str(outcome),
            "should_resume_mirror": bool(should_resume_mirror),
            "grip_width_seed": self.player.get_gripper_width(),
            # Resolved HERE, on the main thread, for the same reason grip_width_seed is:
            # the worker outlives the intervention now, and the main loop may swap the
            # model or start a new scene while it runs.
            "q_target": self._resolve_return_home_q_target(),
            "restore_grip_width": self._resolve_return_home_grip_width(),
            "is_factr": bool(getattr(robot_adapter, "is_factr_adapter", False)),
            "started_wall": time.time(),
            "failure_noted": False,
        }
        self._return_home_result = False
        self._return_home_error = None
        self._return_home_done.clear()
        self._return_home_cancel.clear()
        self._return_home_timeout_reported = False
        self._return_home_thread = threading.Thread(
            target=self._return_home_worker,
            daemon=True,
            name="InterventionReturnHome",
        )
        self._return_home_thread.start()
        self._publish_policy_state(force=True)

    def _poll_return_home(self) -> bool:
        if self._return_home_thread is None:
            return False
        if not self._return_home_done.is_set():
            context = self._return_home_context or {}
            # Another session wants the arm. Honoured ONLY here — i.e. only while we are
            # returning home. A human mid-takeover is never interrupted; that requester
            # simply times out and gets the usual robot_busy refusal.
            if not self._return_home_cancel.is_set() and self.robot_lock is not None:
                try:
                    ttl_s = float(os.environ.get("INTERVENE_ROBOT_YIELD_TTL_S", "10.0"))
                    req = self.robot_lock.yield_requested_of_me(ttl_s=ttl_s)
                except Exception:
                    req = None
                if req:
                    context["yielded"] = True
                    self._return_home_cancel.set()
                    print(f"[REPLAN] Yielding the arm to pid={req.get('requester_pid')}; "
                          "stopping return-home early.")
                    try:
                        self.robot_lock.clear_yield_request(only_mine=False)
                    except Exception:
                        pass
            try:
                timeout_s = float(os.environ.get("INTERVENE_RETURN_WORKER_TIMEOUT_S", "15.0"))
            except (TypeError, ValueError):
                timeout_s = 15.0
            age = time.time() - float(context.get("started_wall", time.time()))
            if timeout_s > 0.0 and age >= timeout_s and not self._return_home_timeout_reported:
                self._return_home_timeout_reported = True
                self._return_home_cancel.set()
                context["failure_noted"] = True
                self.intervention_operation = "return_home_failed"
                self._note_intervention_failed(
                    "return_home_failed",
                    detail=f"return-home exceeded {timeout_s:.1f}s",
                )
                print(
                    f"[REPLAN] Return-home exceeded {timeout_s:.1f}s; cancellation requested "
                    "while policy-state publishing remains active."
                )
                self._publish_policy_state(force=True)
            return False

        context = self._return_home_context or {}
        outcome = context.get("outcome", "completed")
        error = self._return_home_error
        completed = bool(self._return_home_result)
        self.return_home_interrupted_by_replan = bool(self._return_home_interrupted)

        self._return_home_thread = None
        self._return_home_context = None
        self._return_home_done.clear()
        self._return_home_cancel.clear()

        self._clear_replan_state()

        if (
            self.return_home_interrupted_by_replan
            and error is None
            and not context.get("failure_noted")
        ):
            # DELIBERATE stop, not a failure: either this session queued a new intervention
            # or another session asked for the arm. It reports `completed=False` exactly
            # like a real failure, so without this branch every preemption logged a bogus
            # return_home_failed and fired a false intervention-failed edge at the headset.
            # The watchdog path is excluded (it sets failure_noted) — that IS a failure.
            reason = "yield" if context.get("yielded") else "new intervention"
            print(f"[REPLAN] Return-home stopped early ({reason}); "
                  "holding position for the next alignment.")
            # NOT _close_robot_adapter: Robot.close() -> FrankaArm.close() -> reset(), which
            # drives the arm to POLYMETIS' default home. On an interrupt that is exactly
            # wrong — it throws away the pose the operator stopped at and sends the arm
            # somewhere unrelated, and the next intervention then has to drag it back from
            # there. Detach the handles instead and leave the arm where it stopped, so the
            # next intervention aligns to the paused MuJoCo pose from there. This mirrors
            # FACTR, whose interrupt holds the current pose and then aligns.
            self._release_robot_adapter_without_homing(context.get("robot_adapter"))
        elif error is not None or not completed:
            detail = str(error) if error is not None else "return-home did not reach target"
            print(f"[REPLAN] Return-home transition failed: {detail}")
            if not context.get("failure_noted"):
                self._note_intervention_failed("return_home_failed", detail=detail)
                self._publish_policy_state(force=True)
            self._close_robot_adapter(context.get("robot_adapter"))
        elif not context.get("should_resume_mirror"):
            # The app has already moved the arm to the scene pose above. Closing the
            # legacy backend after that would call FrankaArm.reset()->go_home() and
            # visibly move it a second time to a different default pose.
            self._release_robot_adapter_without_homing(context.get("robot_adapter"))

        self._release_robot_ownership()
        # Normally a no-op: the transport half already ran at the keypress. It still runs
        # here for the paths that reach return-home without one (and for the LAB stub
        # tests, which drive this function directly with mode="replan").
        self._end_intervention_transport(outcome=outcome)

        self.last_return_home_s = max(
            0.0,
            time.time() - float(context.get("started_wall", time.time())),
        )
        # NOT last_intervention_transition_s any more: that now measures what the OPERATOR
        # waited for, which ends at the keypress. Homing time is reported separately.
        self.intervention_phase = "policy_resumed"
        self.intervention_operation = "idle"
        self._publish_policy_state(force=True)
        verb = "finished" if outcome == "completed" else "cancelled"
        print(f"[REPLAN] Return-home for the {verb} intervention completed "
              f"({self.last_return_home_s:.1f}s).")

        reintervene = self._pending_reintervene
        self._pending_reintervene = False
        self.return_home_interrupted_by_replan = False
        if reintervene:
            print("[REPLAN] Starting queued intervention after return-home transition.")
            self.trigger_replan()
        return True

    def _end_intervention_transport(self, *, outcome: str = "completed") -> bool:
        """Hand control back to the policy NOW, without waiting for the arm to home.

        The intervention is over the instant the operator presses finish/cancel. It used
        to stay nominally live until the return-home worker finished — 4-12 s during
        which the sim was frozen, the policy paused, the Quest HUD still said
        "intervention active" and the grid still drew the red LIVE border.

        Returning the arm home is a ROBOT concern and stays on the worker thread; this is
        the TRANSPORT half and runs on the keypress. Idempotent: _poll_return_home calls
        it again on completion, which is a no-op in production and the only caller in the
        LAB stub tests.
        """
        if self.mode != "replan" and self.replan_session is None:
            return False

        # Captured before _clear_replan_state() drops it: POST_RELEASE_CHECK and
        # AUTONOMY_RESUMED have to name the intervention they follow.
        self.last_finished_intervention_id = int(getattr(self, "active_intervention_id", 0))

        # Both of these gate `intervention_live` (see _publish_policy_state) and MUST
        # land before the publish below, or the False edge that drives the Quest HUD and
        # the grid border is delayed by a tick.
        self._clear_replan_state()
        self.mode = "replay"
        self.last_recorded_player_frame_idx = None
        self._replan_paused_policy = False

        # MC-SIM and FACTR both hold the arm up with qfrc_applied = qfrc_bias every
        # substep and neither ever clears it, so the policy would keep running against a
        # stale constant force for the rest of the episode and the recorded frames would
        # not be reproducible from their own ctrl. One place covers finish, cancel, and
        # both failure paths.
        try:
            self.player.data.qfrc_applied[:] = 0.0
        except Exception:
            pass

        if hasattr(self.player, "resume_from_current_state"):
            self.player.resume_from_current_state(reset_policy_queue=True)
        elif getattr(self.player, "is_paused", False):
            self.player.toggle_pause()

        evaluator = getattr(self, "task_evaluator", None)
        if evaluator is not None and hasattr(evaluator, "resume_clock"):
            evaluator.resume_clock()

        # Arm the post-release check. Deliberately a flag consumed by the main loop
        # rather than an evaluation here: finish_replan() still has to flush the episode
        # NPZ after this returns, and ending the episode (which rebuilds the scene) in
        # the middle of that would save the wrong frames. The loop consumes the flag on
        # the very next tick, before player.update(), so no policy action can slip in.
        if getattr(self, "_intervention_reached_human_control", False):
            self._post_release_check_pending = True
            self.last_finished_intervention_outcome = str(outcome)
        self._intervention_reached_human_control = False

        self.last_intervention_transition_s = max(
            0.0, time.time() - getattr(self, "intervention_transition_started_wall", time.time())
        )
        if self._return_home_thread is not None:
            # `intervention_live` is already False; this is the remaining signal that the
            # arm is physically still moving.
            self.intervention_phase = "returning_home"
            self.intervention_operation = "returning_home"
        else:
            self.intervention_phase = "policy_resumed"
            self.intervention_operation = "idle"
        self._publish_policy_state(force=True)
        print(f"[REPLAN] Intervention {'finished' if outcome == 'completed' else 'cancelled'}; "
              "policy resumed immediately." + (
                  " Arm is returning home in the background."
                  if self._return_home_thread is not None else ""))
        return True

    def finish_replan(self):
        if self.mode != "replan":
            return
        if self.replan_session is None:
            if self.study_session is not None:
                self.study_session.finish_intervention(
                    outcome="completed", robot_state=self._robot_state_snapshot(),
                    reason="mc_no_session",
                )
            self._start_return_home_or_settle(
                getattr(self.mirror_controller, "robot", None),
                outcome="completed",
                should_resume_mirror=False,
            )
            return

        self.intervention_operation = "finishing"
        robot_adapter = self.replan_session.robot_adapter

        # Phase 5 — the human pressed finish. Timed before the save/stitch work so
        # correction time measures the human's effort, not our file IO.
        if self.study_session is not None:
            self.study_session.finish_intervention(
                outcome="completed", robot_state=self._robot_state_snapshot(),
                frames=(self.episode_recorder.frame_count()
                        if self.episode_recorder is not None else None),
            )

        # Capture everything the save needs BEFORE _clear_replan_state nulls it. The
        # heavy file IO (a full np.savez_compressed of the episode, RGB stacks included)
        # used to run here, on the keypress, BEFORE the arm even started moving. It now
        # runs after the policy is already back in control.
        session = self.replan_session
        intervention_id = int(self.active_intervention_id)
        original_path = self.replan_original_path
        cut_idx = self.replan_cut_idx
        should_resume_mirror = self.replan_should_resume_mirror
        use_episode_recorder = self.episode_recorder is not None

        try:
            if use_episode_recorder:
                session.finish_segment()
                self.next_intervention_id = max(
                    self.next_intervention_id, intervention_id + 1
                )
        except Exception as exc:
            print(f"[REPLAN] Intervention recording finalization failed: {exc}")
            self._note_intervention_failed("transition_failed", detail=str(exc))

        if getattr(robot_adapter, "is_factr_adapter", False):
            print("[FACTR] Intervention finished. Control returned to policy.")

        self._start_return_home_or_settle(
            robot_adapter,
            outcome="completed",
            should_resume_mirror=should_resume_mirror,
        )

        # ---- transport is restored and published from here on; IO is off the keypress ----
        try:
            if use_episode_recorder:
                self._flush_episode_recording(f"intervention_id={intervention_id}")
            else:
                suffix_path = session.save_and_finish()
                out_path = make_replanned_output_path(original_path)
                stitch_npz(original_path, suffix_path, cut_idx, out_path)
                print(f"[REPLAN] Saved replanned trajectory: {out_path}")
        except Exception as exc:
            print(f"[REPLAN] Intervention recording finalization failed: {exc}")
            self._note_intervention_failed("transition_failed", detail=str(exc))

    def cancel_replan(self):
        if self.mode != "replan":
            print("[REPLAN] No active intervention to cancel.")
            return
        if self.replan_session is None:
            if self.study_session is not None:
                self.study_session.finish_intervention(
                    outcome="cancelled", robot_state=self._robot_state_snapshot(),
                    reason="mc_no_session",
                )
            self._start_return_home_or_settle(
                getattr(self.mirror_controller, "robot", None),
                outcome="cancelled",
                should_resume_mirror=False,
            )
            return

        self.intervention_operation = "finishing"
        robot_adapter = self.replan_session.robot_adapter
        should_resume_mirror = self.replan_should_resume_mirror

        if self.study_session is not None:
            self.study_session.finish_intervention(
                outcome="cancelled", robot_state=self._robot_state_snapshot(),
                reason="operator_cancelled",
            )

        try:
            if self.episode_recorder is not None and self.replan_recorder_start_len is not None:
                # Rolls back every per-frame column (see TrajectoryRecorder._FRAME_LISTS);
                # the study columns must truncate with the rest or the arrays desync.
                self.episode_recorder.truncate(self.replan_recorder_start_len)

            snapshot = self.replan_start_snapshot or {}
            if snapshot:
                self.player.data.qpos[:] = snapshot["qpos"]
                self.player.data.qvel[:] = snapshot["qvel"]
                self.player.data.ctrl[:] = snapshot["ctrl"]
                self.player.data.time = float(snapshot["time"])
                if hasattr(self.player, "frame_idx"):
                    self.player.frame_idx = int(snapshot["frame_idx"])
                mujoco.mj_forward(self.player.model, self.player.data)

            # The world is back to its pre-intervention state, so the evaluator's view of
            # it must be too. Without this a rolled-back lift stays latched and a later
            # marginal pose scores a bogus success.
            if self.task_evaluator is not None and self.replan_evaluator_snapshot:
                self.task_evaluator.restore_latches(self.replan_evaluator_snapshot)
        except Exception as exc:
            print(f"[REPLAN] Cancel rollback failed: {exc}")
            self._note_intervention_failed("transition_failed", detail=str(exc))

        # A cancelled intervention is still an intervention that is OVER: the human has let
        # go and the arm is wherever they left it. finish_replan() has always returned the
        # arm to the scene's initial pose; cancel must do the same or the next episode
        # starts from an arbitrary human-placed pose. Done before the control-mode restore
        # below, matching finish_replan's ordering.
        self._start_return_home_or_settle(
            robot_adapter,
            outcome="cancelled",
            should_resume_mirror=should_resume_mirror,
        )

    # Command vocabulary. Semantic names are authoritative; the bare controller letters are
    # legacy aliases kept only so an un-rebuilt Quest APK keeps working.
    #
    # WHY THIS TABLE EXISTS: Unity used to send the raw letter "B" for the sim pause/resume
    # button, and this handler mapped "B" into the task-success set — so pressing pause
    # saved the episode as a SUCCESS and started a new randomized scene, silently writing
    # false labels into the dataset. "Y" was inverted the same way (runtime_impl called it
    # reset, this called it pause). The two sides drifted apart on 2026-07-26 and nothing
    # caught it because each looked self-consistent. Destructive actions are now reachable
    # ONLY by explicit name, never by a bare letter.
    _POLICY_CMD_ALIASES = {
        # intervention
        "INTERVENE": "intervene", "X": "intervene",
        "CANCEL": "cancel",
        # sim transport (human)
        "SIM_TOGGLE": "sim_toggle", "B": "sim_toggle",   # B = pause/resume, NOT success
        "PAUSE": "pause",
        "RESUME": "resume",
        "A": "sim_toggle",
        # NOTE: PAUSE_AUTO / RESUME_AUTO are deliberately absent. They existed only for the
        # OOD auto-pause supervisor, which has been removed. An old sender emitting them
        # now falls through to the unknown-command log rather than silently pausing a
        # session no operator asked to pause.
        # scene
        "RESET": "restart", "Y": "restart", "R": "restart", "RESTART": "restart",
        # destructive — explicit names only
        "TASK_SUCCESS": "task_success", "SUCCESS": "task_success",
        "SUCCEEDED": "task_success", "DONE": "task_success",
        "TASK_FAIL": "task_fail", "TASK_FAILURE": "task_fail",
        "FAIL": "task_fail", "FAILED": "task_fail", "FAILURE": "task_fail",
        # Selection telemetry. PURE BOOKKEEPING: these set a flag that is written into
        # events and per-frame rows and nothing else — no control path reads it. The
        # authoritative CELL_SELECTED event is written by whichever process owns the
        # selection (the desktop grid for a mouse click, the VR runtime for a Quest ray),
        # because that is where the confirmation actually happens.
        "SELECTED": "cell_selected", "DESELECTED": "cell_deselected",
    }

    def _handle_policy_command(self, cmd: str):
        cmd = cmd.strip().upper()
        if not cmd:
            return
        action = self._POLICY_CMD_ALIASES.get(cmd)
        if action is None:
            print(f"[PolicyCmd] Ignoring unknown command: {cmd}")
            return
        # Log the resolved action, not just the raw token, so a future vocabulary drift is
        # visible in the first log line instead of needing a code trace to find.
        print(f"[PolicyCmd] {cmd} -> {action}")

        # Return-home no longer holds the operator hostage: the intervention is already
        # over and `mode` is "replay", so pause/reset/success/fail are legal again and
        # fall through below. Only `cancel` keeps its return-home-specific meaning, and
        # `intervene` is handled by trigger_replan itself (which owns the queuing rule so
        # the direct keyboard path in input/callbacks.py gets it too).
        # Handled before every other branch, including the replan suppression below: the
        # operator can select or leave a cell at any time, and losing the edge because a
        # takeover was live would leave the recorded selection state wrong for the rest
        # of the episode. Touches one bool; changes no control path.
        if action in ("cell_selected", "cell_deselected"):
            self.cell_selected = (action == "cell_selected")
            if self.study_session is not None:
                self.study_session.cell_selected = self.cell_selected
            if action == "cell_deselected":
                self._cancel_intervention_on_deselect()
            return

        if self._return_home_thread is not None and action == "cancel":
            self._return_home_cancel.set()
            print("[REPLAN] Cancelling the current return-home transition.")
            return

        if action == "intervene":
            if self.mode == "replan":
                phase = getattr(self.replan_session, "phase", "")
                if phase != "human_control":
                    print(f"[REPLAN] Finish ignored while intervention phase is {phase or 'unknown'}.")
                    return
                self.finish_replan()
            else:
                self.trigger_replan()
            return
        if action == "cancel":
            self.cancel_replan()
            return

        # RESET is allowed to interrupt a takeover; it is the operator explicitly asking
        # to abandon this attempt. _restart_current_scene already calls
        # _end_intervention_for_scene_change(keep_frames=False), which cancels the
        # intervention, returns the arm home and clears any queued re-intervention.
        # Handled ABOVE the suppression below because that gate was silently dropping it:
        # RESET sent from the Quest or the desktop grid never reached the auto-cancel path
        # at all, so a scene switch during a takeover simply did nothing.
        if action == "restart":
            if self.mode == "replan":
                print("[PolicyCmd] RESET during an active intervention: cancelling the "
                      "takeover and restarting the scene.")
            self._restart_current_scene()
            return

        # Everything below is suppressed during an intervention: the operator has the robot
        # in hand and nothing automated may move the sim underneath them. task_success and
        # task_fail stay here deliberately — an episode OUTCOME must never be markable
        # while a human is holding the arm, or the label describes a scene the policy
        # never produced.
        if self.mode == "replan":
            print(f"[PolicyCmd] Ignoring {cmd} ({action}) while intervention is active.")
            return

        # Every pause/resume is now an operator action: the OOD auto-pause supervisor that
        # produced the PAUSE_AUTO/RESUME_AUTO variants (and the manual-override window that
        # protected humans from it) has been removed. OOD no longer pauses anything.
        if action == "pause":
            if not self.player.is_paused:
                self.player.toggle_pause()
                print("[PolicyCmd] Policy paused (operator).")
            return
        if action == "resume":
            if self.player.is_paused:
                self.player.toggle_pause()
                print("[PolicyCmd] Policy resumed (operator).")
            return
        if action == "sim_toggle":
            self.player.toggle_pause()
            print(f"[PolicyCmd] Policy {'paused' if self.player.is_paused else 'resumed'}.")
            return
        # NOTE: `restart` is handled above the suppression gate, not here.
        if action == "task_success":
            self.mark_task_success(reason=f"operator:{cmd}", decided_by="manual")
            return
        if action == "task_fail":
            self.mark_task_failure(reason=f"operator:{cmd}", decided_by="manual")
            return


    def _drain_policy_commands(self):
        if self.cmd_receiver is None:
            return
        for cmd in self.cmd_receiver.drain():
            self._handle_policy_command(cmd)

    def run(self):
        self.init()
        self._start_main_loop_watchdog()
        try:
            while not glfw.window_should_close(self.window):
                self._loop_heartbeat[0] = time.time()
                glfw.poll_events()
                self._drain_policy_commands()
                # Return-home now runs alongside a LIVE policy, so it is polled every
                # tick regardless of mode. After _drain_policy_commands, so a command
                # that queues a re-intervention on the same tick the worker finishes is
                # seen by this poll rather than left set for the next return-home.
                self._poll_return_home()

                if self.mode == "replan" and self.replan_session is not None:
                    try:
                        self.replan_session.update()
                    except Exception as e:
                        print(f"[REPLAN] Intervention runtime error: {e}")
                        self.cancel_replan()
                        # Publish before bailing out: without this the mirror is left on the
                        # pre-error frame and the VR sim silently freezes there.
                        self._publish_policy_state(force=True)
                        continue
                    # Keep the task evaluator watching during the takeover. It only
                    # OBSERVES here (latches close/lift, contact streak); the decision
                    # still fires exclusively from the replay branch below. Without this
                    # a human who grasps, lifts and places by hand sets none of the
                    # latches tshape/generic success requires, and the episode can never
                    # succeed afterwards.
                    self._observe_task_evaluator()
                    self._publish_policy_state()
                    self._observe_policy_metrics(0.0, False)

                    if not self.headless:
                        win_w, win_h = glfw.get_framebuffer_size(self.window)
                        viewport = mujoco.MjrRect(0, 0, win_w, win_h)
                        glfw.make_context_current(self.window)
                        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.ctx)
                        mujoco.mjr_rectangle(viewport, 0.08, 0.08, 0.08, 1.0)
                        self.viewer.render(
                            self.player,
                            viewport,
                            self.ctx,
                            draw_overlay=True,
                            extra_status=self._compose_status_text("REPLAN: guide robot", "Enter: finish"),
                        )
                else:
                    # The human has just released: judge the world they left BEFORE the
                    # policy touches it. See _run_post_release_check for why this cannot
                    # live inside finish_replan().
                    if self._post_release_check_pending:
                        self._post_release_check_pending = False
                        if self._run_post_release_check():
                            self._publish_policy_state(force=True)
                            continue

                    _upd_t0 = time.perf_counter() if self.metrics is not None else 0.0
                    did_policy_step = bool(self.player.update())
                    glfw.make_context_current(self.window)
                    self._record_player_frame()
                    self._maybe_auto_finish_task()
                    self._publish_policy_state()
                    self._observe_policy_metrics(_upd_t0, did_policy_step)

                    # The return-home worker may call attach_reused_robot at any instant,
                    # flipping mirror_controller.enabled True while it is still ramping
                    # the arm. Streaming here at the same time would give one arm two
                    # writers. _return_home_thread is only ever assigned and cleared on
                    # this thread, so reading it here is race-free.
                    if self.mirror_controller.enabled and self._return_home_thread is None:
                        self.mirror_controller.mirror_from_player(self.player)

                    mirror_status = self.mirror_controller.get_status_text()
                    
                    # 1) draw the normal visible scene to the window
                    if not self.headless:
                        win_w, win_h = glfw.get_framebuffer_size(self.window)
                        viewport = mujoco.MjrRect(0, 0, win_w, win_h)
                        glfw.make_context_current(self.window)
                        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.ctx)
                        mujoco.mjr_rectangle(viewport, 0.08, 0.08, 0.08, 1.0)
                        self.viewer.render(
                            self.player,
                            viewport,
                            self.ctx,
                            draw_overlay=True,
                            extra_status=self._compose_status_text(mirror_status),
                        )

                if self.dump_camera_images:
                    rgbd = self.viewer.render_rgbd_from_cameras(
                        self.player.data,
                        self.ctx,
                        ["top", "front", "right", "left"],
                        width=224,
                        height=224,
                    )
                    rgb = rgbd["top"]["rgb"]
                    Image.fromarray(rgb).save("top_rgb.png")

                    depth = rgbd["top"]["depth"]
                    depth_vis = depth - depth.min()
                    depth_vis = depth_vis / (depth_vis.max() + 1e-8)
                    depth_vis = (255 * depth_vis).astype(np.uint8)
                    Image.fromarray(depth_vis).save("top_depth.png")

                if self.headless:
                    time.sleep(0.001)
                else:
                    glfw.swap_buffers(self.window)
        finally:
            self.close()

    def close(self):
        if self.replan_session is not None and self.episode_recorder is not None:
            try:
                self.replan_session.finish_segment()
            except Exception:
                pass

        # Shutting down DURING an intervention is still an intervention ending, and it is the
        # one path that used to skip this: finish_segment() does not release the robot or leave
        # HUMAN_CONTROL, so closing the viewer mid-takeover left the arm in freedrive at the
        # human's pose. Return home and restore the hold mode before any adapter is closed.
        if self.mode == "replan":
            try:
                if self._return_home_thread is not None:
                    self._return_home_cancel.set()
                    self._return_home_thread.join(timeout=3.0)
                else:
                    adapter = (
                        self.replan_session.robot_adapter
                        if self.replan_session is not None
                        else getattr(self.mirror_controller, "robot", None)
                    )
                    if self._return_franka_to_initial_pose(adapter) and hasattr(
                        adapter, "switch_control_mode"
                    ):
                        adapter.switch_control_mode(
                            os.environ.get(
                                "INTERVENE_ROBOT_CONTROL_MODE",
                                "HYBRID_JOINT_IMPEDANCE_CONTROL",
                            )
                        )
            except Exception as e:
                print(f"[WARN] Could not return the arm home on shutdown: {e}")
            finally:
                self._release_robot_ownership()
                self.mode = "replay"

        try:
            if self.study_session is not None:
                # A session closed mid-episode still has usable data — file it as
                # incomplete rather than losing it.
                out_path = self._finalize_episode(
                    OUTCOME_INCOMPLETE, reason="viewer_closed", decided_by="shutdown"
                )
            else:
                out_path = self._save_episode_recording()
            if out_path is not None:
                print(f"[INFO] Saved full episode recording on close: {out_path}")
        except Exception as e:
            print(f"[WARN] Could not save full episode recording on close: {e}")

        if self.study_session is not None:
            try:
                self.study_session.close(reason="viewer_closed")
            except Exception as e:
                print(f"[WARN] Could not close study session cleanly: {e}")
            self.study_session = None

        if self.replan_session is not None:
            try:
                self.replan_session.robot_adapter.close()
            except Exception:
                pass

        if self.sim_mc_driver is not None:
            try:
                self.sim_mc_driver.close()
            except Exception:
                pass

        if self.mirror_controller.enabled:
            self.mirror_controller.disable()

        if self.cmd_receiver is not None:
            self.cmd_receiver.stop()
            self.cmd_receiver = None

        if self.state_publisher is not None:
            self.state_publisher.stop()
            self.state_publisher = None

        if self.metrics is not None:
            try:
                self.metrics.close()   # flushes the final window + lifetime summary
            except Exception:
                pass
            self.metrics = None

        if self.player is not None and hasattr(self.player, "close"):
            try:
                self.player.close()
            except Exception:
                pass
            self.player = None

        if self.ctx is not None:
            try:
                self.ctx.free()
            except Exception:
                pass
            self.ctx = None

        if self.window is not None:
            try:
                glfw.destroy_window(self.window)
            except Exception:
                pass
            self.window = None

        glfw.terminate()
