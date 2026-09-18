import argparse
import math
import time
import sys
from dataclasses import dataclass
from pathlib import Path

import glfw
import mujoco
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors

try:
    from demo_collector.config import (
        VIEWER_CAMERA_NAMES as CONFIG_VIEWER_CAMERA_NAMES,
        VIEWER_MODE as CONFIG_VIEWER_MODE,
    )
except Exception:
    CONFIG_VIEWER_MODE = "freecam"
    CONFIG_VIEWER_CAMERA_NAMES = {
        "front": "front",
        "right": "VIS_RIGHT",
        "left": "VIS_LEFT",
    }

DEFAULT_TSHAPE_CHECKPOINT = Path(
    "outputs/tshape_act_abs_shift1_rerender_20260508_095159/checkpoints/040000/pretrained_model"
)
DEFAULT_CUPS_CHECKPOINT = Path(
    "outputs/cups_act_abs_shift1_wrist_rerender_gripper_fixed_20260508_100930/"
    "checkpoints/0020000/pretrained_model"
)

DEFAULT_POLICY_HZ = 10.0
DEFAULT_ACTION_MODE = "queue"
DEFAULT_ARM_ACTION_MODE = "absolute"
DEFAULT_GRIPPER_ACTION_MODE = "absolute"
DEFAULT_REALTIME_FACTOR = 2.0
DEFAULT_RENDER_EVERY_N = 2
DEFAULT_DEBUG_EVERY_STEPS = 5
DEFAULT_DEBUG_UNTIL_STEP = 120
DEFAULT_EXPERT_ACTION_SHIFT_FOR_DEBUG = 1
DEFAULT_WINDOW_W = 1280
DEFAULT_WINDOW_H = 720
DEFAULT_VIEWER_MODE = CONFIG_VIEWER_MODE or "multicam"


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def parse_vec3(value: str) -> tuple[float, float, float]:
    parts = [part.strip() for part in value.replace(",", " ").split()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"Expected 3 floats, got {value!r}")
    try:
        return tuple(float(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected 3 floats, got {value!r}") from exc


def parse_body_list(value: str) -> list[str]:
    return [part.strip() for part in value.replace(",", " ").split() if part.strip()]


def parse_cup_pairs(value: str) -> list[tuple[str, str]]:
    pairs = []
    for raw_pair in value.split(","):
        raw_pair = raw_pair.strip()
        if not raw_pair:
            continue
        if ":" not in raw_pair:
            raise argparse.ArgumentTypeError(
                f"Expected cup pair formatted as 'cup:target', got {raw_pair!r}"
            )
        cup, target = [part.strip() for part in raw_pair.split(":", 1)]
        if not cup or not target:
            raise argparse.ArgumentTypeError(
                f"Expected cup pair formatted as 'cup:target', got {raw_pair!r}"
            )
        pairs.append((cup, target))
    if not pairs:
        raise argparse.ArgumentTypeError("Expected at least one cup pair")
    return pairs

TASK_PRESETS = {
    "tshape": {
        "checkpoint": DEFAULT_TSHAPE_CHECKPOINT,
        "xml": Path("mujoco_scenes/working_scenes/with_soft_gripper/sii_scene_table_T_shape.xml"),
        "reset_npz": Path("data_clean/T_shape_clean_good/p1_ep_0001_1777367378_trimmed.npz"),
        "body_name": "T1",
        "target_body_name": "T2",
        "hand_body_name": "hand",
        "free_joint_name": "T1_free",
        "viewer_cameras": {"front": "front", "left": "VIS_LEFT", "right": "VIS_RIGHT"},
        "placement_target_local_offset": (0.0, 0.0, 0.0),
        "placement_z_offset": 0.165,
        "placement_xy_threshold": 0.04,
        "placement_z_tolerance": 0.025,
        "placement_yaw_threshold_deg": 25.0,
        "placement_gripper_open_threshold": 0.025,
        "placement_stable_steps": 10,
        "close_threshold": 0.02,
        "lift_threshold": 0.08,
    },
    "cups": {
        "checkpoint": DEFAULT_CUPS_CHECKPOINT,
        "xml": Path("mujoco_scenes/working_scenes/with_soft_gripper/sii_scene_table_boxes_cups.xml"),
        "reset_npz": Path("data_clean/CUPS_clean_full_gripper_fixed/p1_ep_0001_1778052422_trimmed.npz"),
        "body_name": "cup3",
        "target_body_name": "cup4",
        "hand_body_name": "hand",
        "free_joint_name": "cup3_free",
        "viewer_cameras": {"front": "front", "left": "VIS_LEFT", "right": "VIS_RIGHT", "wrist": "wrist"},
        "cup_pairs": parse_cup_pairs("cup1:cup2,cup3:cup4"),
        "cup_body_names": parse_body_list("cup1 cup2 cup3 cup4"),
        "box_body_names": parse_body_list("cracker_box sugar_box"),
        "box_upright_angle_deg": 20.0,
        "cup_upright_angle_deg": 35.0,
        "cups_require_contact": True,
        "placement_target_local_offset": (0.0, 0.0, 0.0),
        "placement_z_offset": 0.165,
        "placement_xy_threshold": 0.04,
        "placement_z_tolerance": 0.025,
        "placement_yaw_threshold_deg": 180.0,
        "placement_gripper_open_threshold": 0.0,
        "placement_stable_steps": 10,
        "close_threshold": 0.02,
        "lift_threshold": 0.03,
    },
}

PANDA_JOINT_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
]


@dataclass(frozen=True)
class RolloutConfig:
    task_name: str
    checkpoint: Path
    xml: Path
    reset_npz: Path
    policy_hz: float
    action_mode: str
    arm_action_mode: str
    gripper_action_mode: str
    arm_delta_clip: float | None
    realtime_factor: float
    max_steps: int
    stop_on_success: bool
    viewer_mode: str
    viewer_cameras: dict[str, str]
    window_w: int
    window_h: int
    render_every: int
    debug_every: int
    debug_until: int
    expert_action_shift_for_debug: int
    body_name: str
    target_body_name: str
    hand_body_name: str
    cup_pairs: list[tuple[str, str]]
    cup_body_names: list[str]
    box_body_names: list[str]
    box_upright_angle_deg: float
    cup_upright_angle_deg: float
    cups_require_contact: bool
    close_threshold: float
    lift_threshold: float
    placement_target_local_offset: tuple[float, float, float]
    placement_z_offset: float
    placement_xy_threshold: float
    placement_z_tolerance: float
    placement_yaw_threshold_deg: float
    placement_gripper_open_threshold: float
    placement_stable_steps: int


# -----------------------------
# HELPERS
# -----------------------------
def parse_viewer_cameras(value: str | None, fallback: dict[str, str]) -> dict[str, str]:
    if value is None or not value.strip():
        return dict(fallback)

    cameras = {}
    for index, raw_part in enumerate(value.replace(";", ",").split(",")):
        part = raw_part.strip()
        if not part:
            continue
        if "=" in part:
            label, camera = [piece.strip() for piece in part.split("=", 1)]
        elif ":" in part:
            label, camera = [piece.strip() for piece in part.split(":", 1)]
        else:
            label = part
            camera = part
        if not label or not camera:
            raise argparse.ArgumentTypeError(
                f"Invalid viewer camera entry {raw_part!r}. Use camera or label=camera."
            )
        if label in cameras:
            label = f"{label}_{index}"
        cameras[label] = camera

    if not cameras:
        raise argparse.ArgumentTypeError("At least one viewer camera is required.")
    return cameras


def infer_task_name(task: str, checkpoint: Path | None, xml: Path | None, reset_npz: Path | None) -> str:
    if task != "auto":
        return task

    joined = " ".join(
        str(part).lower()
        for part in (checkpoint, xml or "", reset_npz or "")
    )
    if "cup" in joined or "boxes_cups" in joined:
        return "cups"
    return "tshape"


def resolve_path_arg(value: Path | None, default: Path) -> Path:
    return value if value is not None else default


def option_or_preset(args: argparse.Namespace, preset: dict, key: str, fallback=None):
    value = getattr(args, key)
    if value is not None:
        return value
    return preset.get(key, fallback)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive ACT rollout viewer for T-shape and cup-stacking MuJoCo tasks."
    )
    parser.add_argument("--task", choices=["auto", "tshape", "cups"], default="auto")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--xml", type=Path, default=None)
    parser.add_argument("--reset_npz", type=Path, default=None)
    parser.add_argument("--policy_hz", type=float, default=DEFAULT_POLICY_HZ)
    parser.add_argument("--action_mode", choices=["queue", "replan"], default=DEFAULT_ACTION_MODE)
    parser.add_argument("--arm_action_mode", choices=["absolute", "ctrl_delta", "qpos_error"], default=DEFAULT_ARM_ACTION_MODE)
    parser.add_argument("--gripper_action_mode", choices=["absolute", "delta"], default=DEFAULT_GRIPPER_ACTION_MODE)
    parser.add_argument("--arm_delta_clip", type=float, default=None)
    parser.add_argument("--realtime_factor", type=float, default=DEFAULT_REALTIME_FACTOR)
    parser.add_argument("--max_steps", type=int, default=0, help="0 means run until the viewer is closed.")
    parser.add_argument("--stop_on_success", type=parse_bool, default=False)
    parser.add_argument("--viewer_mode", choices=["freecam", "multicam"], default=DEFAULT_VIEWER_MODE)
    parser.add_argument(
        "--viewer_cameras",
        default=None,
        help="Comma-separated camera list for multicam. Use 'front,left,wrist' or 'label=camera'.",
    )
    parser.add_argument("--window_w", type=int, default=DEFAULT_WINDOW_W)
    parser.add_argument("--window_h", type=int, default=DEFAULT_WINDOW_H)
    parser.add_argument("--render_every", type=int, default=DEFAULT_RENDER_EVERY_N)
    parser.add_argument("--debug_every", type=int, default=DEFAULT_DEBUG_EVERY_STEPS)
    parser.add_argument("--debug_until", type=int, default=DEFAULT_DEBUG_UNTIL_STEP)
    parser.add_argument("--expert_action_shift_for_debug", type=int, default=DEFAULT_EXPERT_ACTION_SHIFT_FOR_DEBUG)

    parser.add_argument("--body_name", default=None)
    parser.add_argument("--target_body_name", default=None)
    parser.add_argument("--hand_body_name", default=None)
    parser.add_argument("--free_joint_name", default=None)
    parser.add_argument("--cup_pairs", type=parse_cup_pairs, default=None)
    parser.add_argument("--cup_body_names", type=parse_body_list, default=None)
    parser.add_argument("--box_body_names", type=parse_body_list, default=None)
    parser.add_argument("--box_upright_angle_deg", type=float, default=None)
    parser.add_argument("--cup_upright_angle_deg", type=float, default=None)
    parser.add_argument("--cups_require_contact", type=parse_bool, default=None)
    parser.add_argument("--close_threshold", type=float, default=None)
    parser.add_argument("--lift_threshold", type=float, default=None)
    parser.add_argument("--placement_xy_threshold", type=float, default=None)
    parser.add_argument("--placement_target_local_offset", type=parse_vec3, default=None)
    parser.add_argument("--placement_z_offset", type=float, default=None)
    parser.add_argument("--placement_z_tolerance", type=float, default=None)
    parser.add_argument("--placement_yaw_threshold_deg", type=float, default=None)
    parser.add_argument("--placement_gripper_open_threshold", type=float, default=None)
    parser.add_argument("--placement_stable_steps", type=int, default=None)
    return parser.parse_args()


def build_rollout_config(args: argparse.Namespace) -> RolloutConfig:
    task_name = infer_task_name(args.task, args.checkpoint, args.xml, args.reset_npz)
    preset = TASK_PRESETS[task_name]

    cfg = RolloutConfig(
        task_name=task_name,
        checkpoint=resolve_path_arg(args.checkpoint, preset["checkpoint"]),
        xml=resolve_path_arg(args.xml, preset["xml"]),
        reset_npz=resolve_path_arg(args.reset_npz, preset["reset_npz"]),
        policy_hz=args.policy_hz,
        action_mode=args.action_mode,
        arm_action_mode=args.arm_action_mode,
        gripper_action_mode=args.gripper_action_mode,
        arm_delta_clip=args.arm_delta_clip,
        realtime_factor=args.realtime_factor,
        max_steps=args.max_steps,
        stop_on_success=args.stop_on_success,
        viewer_mode=args.viewer_mode,
        viewer_cameras=parse_viewer_cameras(args.viewer_cameras, preset["viewer_cameras"]),
        window_w=args.window_w,
        window_h=args.window_h,
        render_every=args.render_every,
        debug_every=args.debug_every,
        debug_until=args.debug_until,
        expert_action_shift_for_debug=args.expert_action_shift_for_debug,
        body_name=option_or_preset(args, preset, "body_name"),
        target_body_name=option_or_preset(args, preset, "target_body_name"),
        hand_body_name=option_or_preset(args, preset, "hand_body_name"),
        cup_pairs=option_or_preset(args, preset, "cup_pairs", []),
        cup_body_names=option_or_preset(args, preset, "cup_body_names", []),
        box_body_names=option_or_preset(args, preset, "box_body_names", []),
        box_upright_angle_deg=option_or_preset(args, preset, "box_upright_angle_deg", 20.0),
        cup_upright_angle_deg=option_or_preset(args, preset, "cup_upright_angle_deg", 35.0),
        cups_require_contact=option_or_preset(args, preset, "cups_require_contact", True),
        close_threshold=option_or_preset(args, preset, "close_threshold"),
        lift_threshold=option_or_preset(args, preset, "lift_threshold"),
        placement_target_local_offset=option_or_preset(args, preset, "placement_target_local_offset"),
        placement_z_offset=option_or_preset(args, preset, "placement_z_offset"),
        placement_xy_threshold=option_or_preset(args, preset, "placement_xy_threshold"),
        placement_z_tolerance=option_or_preset(args, preset, "placement_z_tolerance"),
        placement_yaw_threshold_deg=option_or_preset(args, preset, "placement_yaw_threshold_deg"),
        placement_gripper_open_threshold=option_or_preset(args, preset, "placement_gripper_open_threshold"),
        placement_stable_steps=option_or_preset(args, preset, "placement_stable_steps"),
    )

    if cfg.policy_hz <= 0:
        raise ValueError("--policy_hz must be > 0")
    if cfg.render_every < 1:
        raise ValueError("--render_every must be >= 1")
    if cfg.max_steps < 0:
        raise ValueError("--max_steps must be >= 0")
    if cfg.placement_stable_steps < 1:
        raise ValueError("--placement_stable_steps must be >= 1")
    return cfg


def get_qpos_indices_for_joints(model: mujoco.MjModel, joint_names: list[str]) -> list[int]:
    """
    Return the qpos indices in MuJoCo corresponding to the given joint names.

    MuJoCo stores joint positions in the global vector `data.qpos`.
    Each joint has an index (address) inside that vector.
    This function maps joint names -> qpos indices.
    """
    idxs = []
    for name in joint_names:
        j = model.joint(name)              # Access joint object
        joint_id = j.id                    # Internal MuJoCo joint ID
        qpos_adr = model.jnt_qposadr[joint_id]  # Address inside qpos
        idxs.append(int(qpos_adr))
    return idxs


def get_camera_id(model: mujoco.MjModel, cam_name: str) -> int:
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cam_id < 0:
        raise ValueError(f"Camera '{cam_name}' not found in MuJoCo model.")
    return cam_id


def depth_buffer_to_meters(model: mujoco.MjModel, depth_buffer: np.ndarray) -> np.ndarray:
    znear = model.vis.map.znear * model.stat.extent
    zfar = model.vis.map.zfar * model.stat.extent
    depth = depth_buffer.astype(np.float32)
    depth_m = znear / (1.0 - depth * (1.0 - znear / zfar))
    depth_m[~np.isfinite(depth_m)] = 0.0
    return depth_m


def chw_float01_from_rgb(rgb_hwc: np.ndarray) -> torch.Tensor:
    x = rgb_hwc.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))  # HWC -> CHW
    return torch.from_numpy(x)


def build_state(
    qpos_sim: np.ndarray,
    qvel_sim: np.ndarray,
    ctrl_sim_prev: np.ndarray,
    qpos_indices:  list[int],
    state_dim: int,
    phase: float = 0.0,
) -> np.ndarray:
    q_arm = qpos_sim[qpos_indices].astype(np.float32)
    dq_arm = qvel_sim[:7].astype(np.float32)
    ctrl = ctrl_sim_prev.astype(np.float32)
    grip = np.array([ctrl[7]], dtype=np.float32)
    if state_dim == 15:
        state = np.concatenate([q_arm, dq_arm, grip], axis=0)
    elif state_dim == 22:
        state = np.concatenate([q_arm, dq_arm, ctrl[:7], grip], axis=0)
    elif state_dim == 23:
        state = np.concatenate([q_arm, dq_arm, ctrl[:7], grip, np.array([phase], dtype=np.float32)], axis=0)
    else:
        raise ValueError(f"Unsupported observation.state dim: {state_dim}")
    return state.astype(np.float32)


def reset_from_npz(data: mujoco.MjData, npz_path: Path) -> None:
    """
    Start rollout from the same full simulator state used in the demonstration.

    The LeRobot state stores only robot q/dq/gripper. For visual policies, the
    object free-joint poses matter too, so we restore the full MuJoCo qpos/qvel.
    """
    if not npz_path.exists():
        raise FileNotFoundError(f"Reset NPZ not found: {npz_path}")

    episode = np.load(npz_path, allow_pickle=False)
    for key in ("qpos_sim", "qvel_sim", "ctrl_sim"):
        if key not in episode:
            raise KeyError(f"{npz_path} missing key: {key}")

    data.qpos[:] = episode["qpos_sim"][0]
    data.qvel[:] = episode["qvel_sim"][0]
    n_ctrl = min(data.ctrl.shape[0], episode["ctrl_sim"].shape[1])
    data.ctrl[:n_ctrl] = episode["ctrl_sim"][0, :n_ctrl]


def get_episode_length(npz_path: Path) -> int:
    episode = np.load(npz_path, allow_pickle=False)
    if "ctrl_sim" not in episode:
        raise KeyError(f"{npz_path} missing key: ctrl_sim")
    return int(episode["ctrl_sim"].shape[0])


def load_expert_ctrl(npz_path: Path) -> np.ndarray:
    episode = np.load(npz_path, allow_pickle=False)
    if "ctrl_sim" not in episode:
        raise KeyError(f"{npz_path} missing key: ctrl_sim")
    return episode["ctrl_sim"].astype(np.float32)


def clip_to_actuator_ranges(model: mujoco.MjModel, ctrl: np.ndarray) -> np.ndarray:
    clipped = ctrl.copy()
    for i in range(min(model.nu, clipped.shape[0])):
        if model.actuator_ctrllimited[i]:
            lo, hi = model.actuator_ctrlrange[i]
            clipped[i] = np.clip(clipped[i], lo, hi)
    return clipped


def queue_mode_needs_observation(policy: ACTPolicy) -> bool:
    if getattr(policy.config, "temporal_ensemble_coeff", None) is not None:
        return True
    return len(policy._action_queue) == 0


def action_needs_observation(policy: ACTPolicy, action_mode: str) -> bool:
    if action_mode == "replan":
        return True
    if action_mode == "queue":
        return queue_mode_needs_observation(policy)
    raise ValueError(f"Unknown ACTION_MODE: {action_mode}")


def select_processed_action(
    policy: ACTPolicy,
    preprocessor,
    postprocessor,
    observation: dict[str, torch.Tensor] | None,
    action_mode: str,
    chunk_sink=None,
) -> tuple[np.ndarray, bool]:
    """Pick the next action.

    `chunk_sink`, when given, receives the post-processed action chunk on steps where one
    was computed. In 'replan' mode the chunk is produced anyway, so ACC's temporal
    ensemble can reuse it instead of paying for a second forward pass.
    """
    queried_policy = action_needs_observation(policy, action_mode)
    batch = preprocessor(observation) if queried_policy else {}

    if action_mode == "replan":
        if observation is None:
            raise ValueError("ACTION_MODE='replan' requires an observation every step.")
        raw_chunk = policy.predict_action_chunk(batch)
        if chunk_sink is not None:
            try:
                chunk_sink(postprocessor(raw_chunk))
            except Exception:
                pass  # ACC instrumentation must never break action selection
        raw_action = raw_chunk[:, 0]
    elif action_mode == "queue":
        raw_action = policy.select_action(batch)
    else:
        raise ValueError(f"Unknown ACTION_MODE: {action_mode}")

    action = postprocessor(raw_action)
    if isinstance(action, torch.Tensor):
        return action.squeeze(0).detach().cpu().numpy().astype(np.float32), queried_policy
    return np.asarray(action, dtype=np.float32).reshape(8), queried_policy


def policy_image_camera_names(policy: ACTPolicy) -> list[str]:
    return [
        key.removeprefix("observation.images.")
        for key in policy.config.input_features
        if key.startswith("observation.images.")
    ]


def print_debug_line(
    step: int,
    policy_query_count: int,
    queried_policy: bool,
    pred_action: np.ndarray,
    target_ctrl: np.ndarray,
    previous_ctrl: np.ndarray,
    expert_ctrl: np.ndarray,
    elapsed: float,
    policy_dt: float,
    debug_every_steps: int,
    debug_until_step: int,
    expert_action_shift: int,
) -> None:
    if debug_every_steps <= 0:
        return
    if step > debug_until_step or step % debug_every_steps != 0:
        return

    expert_idx = min(step + expert_action_shift, len(expert_ctrl) - 1)
    expert = expert_ctrl[expert_idx]
    arm_error = float(np.linalg.norm(target_ctrl[:7] - expert[:7]))
    grip_error = float(target_ctrl[7] - expert[7])
    arm_step = float(np.linalg.norm(target_ctrl[:7] - previous_ctrl[:7]))
    print(
        "[DBG] "
        f"step={step:04d} sim_t={step * policy_dt:05.2f}s "
        f"policy_calls={policy_query_count:03d}{'*' if queried_policy else ' '} "
        f"arm_step={arm_step:.4f} arm_err_vs_demo={arm_error:.4f} "
        f"grip_pred={pred_action[7]:.4f} grip_cmd={target_ctrl[7]:.4f} "
        f"grip_demo_next={expert[7]:.4f} grip_err={grip_error:+.4f} "
        f"loop={elapsed * 1000.0:.1f}ms"
    )


# -----------------------------
# RENDERER
# -----------------------------
class SimpleRenderer:
    def __init__(
        self,
        model: mujoco.MjModel,
        width: int,
        height: int,
        *,
        viewer_mode: str,
        viewer_cameras: dict[str, str],
    ):
        self.model = model
        self.width = width
        self.height = height
        self.viewer_mode = viewer_mode
        self.viewer_cameras = dict(viewer_cameras)

        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW.")

        self.window = glfw.create_window(width, height, "ACT rollout", None, None)
        if self.window is None:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window.")

        glfw.make_context_current(self.window)
        glfw.swap_interval(0)

        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.scn = mujoco.MjvScene(model, maxgeom=10000)
        self.ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

        mujoco.mjv_defaultCamera(self.cam)
        mujoco.mjv_defaultOption(self.opt)

        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = np.array([0.45, 0.0, 0.45], dtype=np.float64)
        self.cam.distance = 1.5968364916117135
        self.cam.azimuth = 179.578125
        self.cam.elevation = -28.0

        self.viewport = mujoco.MjrRect(0, 0, width, height)

        self.rgb = np.zeros((224, 224, 3), dtype=np.uint8)
        self.depth = np.zeros((224, 224), dtype=np.float32)
        self.policy_renderer = mujoco.Renderer(model, height=224, width=224)
        self.show_overlay = True

    def should_close(self) -> bool:
        return glfw.window_should_close(self.window)

    def poll(self):
        glfw.poll_events()

    def _save_viewer_cam_state(self):
        return {
            "type": self.cam.type,
            "fixedcamid": self.cam.fixedcamid,
            "lookat": self.cam.lookat.copy(),
            "distance": self.cam.distance,
            "azimuth": self.cam.azimuth,
            "elevation": self.cam.elevation,
        }

    def _restore_viewer_cam_state(self, state):
        self.cam.type = state["type"]
        self.cam.fixedcamid = state["fixedcamid"]
        self.cam.lookat[:] = state["lookat"]
        self.cam.distance = state["distance"]
        self.cam.azimuth = state["azimuth"]
        self.cam.elevation = state["elevation"]

    def _render_fixed_camera_to_viewport(
        self,
        data: mujoco.MjData,
        viewport: mujoco.MjrRect,
        cam_name: str,
    ) -> None:
        cam_id = get_camera_id(self.model, cam_name)
        saved = self._save_viewer_cam_state()

        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam.fixedcamid = cam_id
        mujoco.mjv_updateScene(
            self.model,
            data,
            self.opt,
            None,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scn,
        )
        mujoco.mjr_render(viewport, self.scn, self.ctx)

        self._restore_viewer_cam_state(saved)

    def _draw_label(self, viewport: mujoco.MjrRect, label: str) -> None:
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            viewport,
            label,
            "",
            self.ctx,
        )

    def _render_multicam_grid(self, data: mujoco.MjData, width: int, height: int) -> None:
        items = list(self.viewer_cameras.items())
        if not items:
            raise ValueError("viewer_cameras is empty")

        cols = int(math.ceil(math.sqrt(len(items))))
        rows = int(math.ceil(len(items) / cols))
        cell_w = max(1, width // cols)
        cell_h = max(1, height // rows)

        for index, (label, camera_name) in enumerate(items):
            row = index // cols
            col = index % cols
            x = col * cell_w
            y = height - (row + 1) * cell_h
            w = width - x if col == cols - 1 else cell_w
            h = height - row * cell_h if row == rows - 1 else cell_h
            viewport = mujoco.MjrRect(x, y, w, h)
            self._render_fixed_camera_to_viewport(data, viewport, camera_name)
            self._draw_label(viewport, f"{label}: {camera_name}")

    def render_viewer(self, data: mujoco.MjData, overlay_lines: list[str] | None = None):
        glfw.make_context_current(self.window)
        w, h = glfw.get_framebuffer_size(self.window)
        self.viewport = mujoco.MjrRect(0, 0, w, h)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.ctx)

        if self.viewer_mode == "freecam":
            mujoco.mjv_updateScene(
                self.model,
                data,
                self.opt,
                None,
                self.cam,
                mujoco.mjtCatBit.mjCAT_ALL,
                self.scn,
            )
            mujoco.mjr_render(self.viewport, self.scn, self.ctx)

        elif self.viewer_mode == "multicam":
            self._render_multicam_grid(data, w, h)

        else:
            raise ValueError(f"Unknown viewer mode: {self.viewer_mode}")

        if self.show_overlay and overlay_lines:
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                self.viewport,
                "\n".join(overlay_lines),
                f"viewer: {self.viewer_mode}",
                self.ctx,
            )

        glfw.swap_buffers(self.window)

    def render_rgb(self, data: mujoco.MjData, cam_name: str, width: int = 224, height: int = 224):
        self.policy_renderer.disable_depth_rendering()
        self.policy_renderer.update_scene(data, camera=cam_name)
        return self.policy_renderer.render().copy()

    def close(self):
        try:
            self.policy_renderer.close()
        except Exception:
            pass
        try:
            self.ctx.free()
        except Exception:
            pass
        try:
            glfw.destroy_window(self.window)
        except Exception:
            pass
        glfw.terminate()


def validate_body_exists(model: mujoco.MjModel, body_name: str) -> None:
    if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name) < 0:
        raise ValueError(f"Body not found in model: {body_name}")


def validate_camera_exists(model: mujoco.MjModel, camera_name: str) -> None:
    get_camera_id(model, camera_name)


def body_id(model: mujoco.MjModel, body_name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise ValueError(f"Body not found: {body_name}")
    return int(bid)


def body_pos(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> np.ndarray:
    return data.xpos[body_id(model, body_name)].copy()


def body_yaw(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> float:
    rot = data.xmat[body_id(model, body_name)].reshape(3, 3)
    for axis_index in (0, 2, 1):
        axis = rot[:, axis_index].copy()
        axis[2] = 0.0
        if np.linalg.norm(axis) > 1e-6:
            return float(np.arctan2(axis[1], axis[0]))
    return 0.0


def bodies_in_contact(model: mujoco.MjModel, data: mujoco.MjData, body_a: str, body_b: str) -> bool:
    body_a_id = body_id(model, body_a)
    body_b_id = body_id(model, body_b)
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        geom1_body = int(model.geom_bodyid[contact.geom1])
        geom2_body = int(model.geom_bodyid[contact.geom2])
        if (geom1_body == body_a_id and geom2_body == body_b_id) or (
            geom1_body == body_b_id and geom2_body == body_a_id
        ):
            return True
    return False


def upright_angle_deg(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> float:
    rot = data.xmat[body_id(model, body_name)].reshape(3, 3)
    local_z_in_world = rot[:, 2]
    cos_angle = float(np.clip(local_z_in_world[2], -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))


def bodies_upright_status(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_names: list[str],
    max_angle_deg: float,
) -> tuple[bool, float, str]:
    worst_angle = 0.0
    worst_body = ""
    for name in body_names:
        angle = upright_angle_deg(model, data, name)
        if angle > worst_angle:
            worst_angle = angle
            worst_body = name
    return worst_angle <= max_angle_deg, worst_angle, worst_body


def angle_error_deg(angle_a: float, angle_b: float) -> float:
    diff = (angle_a - angle_b + np.pi) % (2.0 * np.pi) - np.pi
    return float(abs(np.degrees(diff)))


def placement_metrics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    body_name: str,
    target_body_name: str,
    target_local_offset: np.ndarray,
    placement_z_offset: float,
) -> tuple[np.ndarray, float, float, float]:
    obj_pos = body_pos(model, data, body_name)
    target_id = body_id(model, target_body_name)
    target_body_pos = data.xpos[target_id].copy()
    target_rot = data.xmat[target_id].reshape(3, 3)
    target_pos = target_body_pos + target_rot @ target_local_offset
    xy_error = float(np.linalg.norm(obj_pos[:2] - target_pos[:2]))
    z_error = float(obj_pos[2] - (target_pos[2] + placement_z_offset))
    yaw_error = angle_error_deg(body_yaw(model, data, body_name), body_yaw(model, data, target_body_name))
    return target_pos, xy_error, z_error, yaw_error


def cup_pair_status(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    cup_body_name: str,
    target_body_name: str,
    target_local_offset: np.ndarray,
    placement_z_offset: float,
    placement_xy_threshold: float,
    placement_z_tolerance: float,
    require_contact: bool,
) -> dict:
    target_pos, xy_error, z_error, yaw_error_deg = placement_metrics(
        model,
        data,
        body_name=cup_body_name,
        target_body_name=target_body_name,
        target_local_offset=target_local_offset,
        placement_z_offset=placement_z_offset,
    )
    contact = bodies_in_contact(model, data, cup_body_name, target_body_name)
    placed = (
        xy_error <= placement_xy_threshold
        and abs(z_error) <= placement_z_tolerance
        and (contact or not require_contact)
    )
    return {
        "cup": cup_body_name,
        "target": target_body_name,
        "target_pos": target_pos,
        "xy_error": xy_error,
        "z_error": z_error,
        "yaw_error_deg": yaw_error_deg,
        "contact": contact,
        "placed": placed,
    }


def cups_success_status(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    cup_pairs: list[tuple[str, str]],
    cup_body_names: list[str],
    box_body_names: list[str],
    box_upright_angle_deg: float,
    cup_upright_angle_deg: float,
    placement_target_local_offset: tuple[float, float, float],
    placement_z_offset: float,
    placement_xy_threshold: float,
    placement_z_tolerance: float,
    placement_gripper_open_threshold: float,
    require_contact: bool,
) -> dict:
    target_local_offset = np.asarray(placement_target_local_offset, dtype=np.float64)
    pair_statuses = [
        cup_pair_status(
            model,
            data,
            cup_body_name=cup,
            target_body_name=target,
            target_local_offset=target_local_offset,
            placement_z_offset=placement_z_offset,
            placement_xy_threshold=placement_xy_threshold,
            placement_z_tolerance=placement_z_tolerance,
            require_contact=require_contact,
        )
        for cup, target in cup_pairs
    ]
    boxes_ok, max_box_angle, worst_box = bodies_upright_status(
        model,
        data,
        box_body_names,
        box_upright_angle_deg,
    )
    cups_ok, max_cup_angle, worst_cup = bodies_upright_status(
        model,
        data,
        cup_body_names,
        cup_upright_angle_deg,
    )
    gripper_open = (
        placement_gripper_open_threshold <= 0
        or float(data.ctrl[7]) >= placement_gripper_open_threshold
    )
    pairs_ok = all(status["placed"] for status in pair_statuses)
    return {
        "success_now": bool(boxes_ok and cups_ok and pairs_ok and gripper_open),
        "pairs": pair_statuses,
        "boxes_ok": boxes_ok,
        "max_box_angle_deg": max_box_angle,
        "worst_box": worst_box,
        "cups_ok": cups_ok,
        "max_cup_angle_deg": max_cup_angle,
        "worst_cup": worst_cup,
        "gripper_open": gripper_open,
        "all_pairs_contact": all(status["contact"] for status in pair_statuses),
    }


def cups_fail_reason(status: dict, placement_stable_steps: int, contact_streak: int) -> str:
    if not status["boxes_ok"]:
        return f"box_fell:{status['worst_box']}:{status['max_box_angle_deg']:.1f}deg"
    if not status["cups_ok"]:
        return f"cup_fell:{status['worst_cup']}:{status['max_cup_angle_deg']:.1f}deg"
    for pair in status["pairs"]:
        name = f"{pair['cup']}_on_{pair['target']}"
        if not pair["contact"]:
            return f"{name}_not_touching"
        if not pair["placed"]:
            if abs(pair["z_error"]) > pair["xy_error"]:
                return f"{name}_bad_z"
            return f"{name}_not_centered"
    if not status["gripper_open"]:
        return "gripper_not_open"
    if contact_streak < placement_stable_steps:
        return "not_stably_placed"
    return ""


def fail_reason(
    first_close_step: int | None,
    first_lift_step: int | None,
    target_xy_error: float,
    max_target_xy_error: float,
    target_z_error: float,
    max_abs_target_z_error: float,
    target_yaw_error_deg: float,
    max_target_yaw_error_deg: float,
    final_gripper: float,
    min_open_gripper: float,
    t1_t2_contact: bool,
    contact_streak: int,
    min_contact_streak: int,
) -> str:
    if first_close_step is None:
        return "no_gripper_close"
    if first_lift_step is None:
        return "no_lift"
    if target_xy_error > max_target_xy_error:
        return "not_centered_on_target"
    if abs(target_z_error) > max_abs_target_z_error:
        if target_z_error > 0:
            return "too_high_above_target"
        return "too_low_for_stack"
    if target_yaw_error_deg > max_target_yaw_error_deg:
        return "yaw_misaligned"
    if min_open_gripper > 0 and final_gripper < min_open_gripper:
        return "gripper_not_open"
    if not t1_t2_contact:
        return "not_touching_target"
    if contact_streak < min_contact_streak:
        return "not_stably_sitting"
    return ""


def make_task_status_lines(
    *,
    task_name: str,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    target_body_name: str,
    initial_body_pos: np.ndarray,
    cup_pairs: list[tuple[str, str]],
    cup_body_names: list[str],
    box_body_names: list[str],
    box_upright_angle_deg: float,
    cup_upright_angle_deg: float,
    cups_require_contact: bool,
    placement_target_local_offset: tuple[float, float, float],
    placement_z_offset: float,
    placement_xy_threshold: float,
    placement_z_tolerance: float,
    placement_yaw_threshold_deg: float,
    placement_gripper_open_threshold: float,
    placement_stable_steps: int,
    first_close_step: int | None,
    first_lift_step: int | None,
    contact_streak: int,
    step: int,
) -> tuple[bool, int, list[str], str]:
    final_gripper = float(data.ctrl[7])

    if task_name == "cups":
        status = cups_success_status(
            model,
            data,
            cup_pairs=cup_pairs,
            cup_body_names=cup_body_names,
            box_body_names=box_body_names,
            box_upright_angle_deg=box_upright_angle_deg,
            cup_upright_angle_deg=cup_upright_angle_deg,
            placement_target_local_offset=placement_target_local_offset,
            placement_z_offset=placement_z_offset,
            placement_xy_threshold=placement_xy_threshold,
            placement_z_tolerance=placement_z_tolerance,
            placement_gripper_open_threshold=placement_gripper_open_threshold,
            require_contact=cups_require_contact,
        )
        placed_now = bool(status["success_now"])
        next_streak = contact_streak + 1 if placed_now else 0
        success = next_streak >= placement_stable_steps
        reason = "" if success else cups_fail_reason(status, placement_stable_steps, next_streak)
        pair_text = " ".join(
            f"{pair['cup']}->{pair['target']}:{'OK' if pair['placed'] else 'NO'}"
            f"(xy={pair['xy_error']:.3f},z={pair['z_error']:+.3f},c={int(pair['contact'])})"
            for pair in status["pairs"]
        )
        lines = [
            f"task: cups {'PASS' if success else 'running'}",
            pair_text,
            f"upright boxes={int(status['boxes_ok'])} max_box={status['max_box_angle_deg']:.1f}deg "
            f"cups={int(status['cups_ok'])} max_cup={status['max_cup_angle_deg']:.1f}deg",
            f"stable: {next_streak}/{placement_stable_steps} grip={final_gripper:.4f}",
        ]
        if reason:
            lines.append(f"why not pass: {reason}")
        return success, next_streak, lines, reason

    obj_pos = body_pos(model, data, body_name)
    lift = float(obj_pos[2] - initial_body_pos[2])
    target_pos, xy_error, z_error, yaw_error_deg = placement_metrics(
        model,
        data,
        body_name=body_name,
        target_body_name=target_body_name,
        target_local_offset=np.asarray(placement_target_local_offset, dtype=np.float64),
        placement_z_offset=placement_z_offset,
    )
    contact = bodies_in_contact(model, data, body_name, target_body_name)
    placed_now = (
        xy_error <= placement_xy_threshold
        and abs(z_error) <= placement_z_tolerance
        and yaw_error_deg <= placement_yaw_threshold_deg
        and (placement_gripper_open_threshold <= 0 or final_gripper >= placement_gripper_open_threshold)
        and contact
    )
    next_streak = contact_streak + 1 if placed_now else 0
    success = (
        first_close_step is not None
        and first_lift_step is not None
        and next_streak >= placement_stable_steps
    )
    reason = "" if success else fail_reason(
        first_close_step=first_close_step,
        first_lift_step=first_lift_step,
        target_xy_error=xy_error,
        max_target_xy_error=placement_xy_threshold,
        target_z_error=z_error,
        max_abs_target_z_error=placement_z_tolerance,
        target_yaw_error_deg=yaw_error_deg,
        max_target_yaw_error_deg=placement_yaw_threshold_deg,
        final_gripper=final_gripper,
        min_open_gripper=placement_gripper_open_threshold,
        t1_t2_contact=contact,
        contact_streak=next_streak,
        min_contact_streak=placement_stable_steps,
    )
    lines = [
        f"task: tshape {'PASS' if success else 'running'}",
        f"close={first_close_step if first_close_step is not None else '-'} "
        f"lift={first_lift_step if first_lift_step is not None else '-'} "
        f"lift_now={lift:.3f}",
        f"place xy={xy_error:.3f}/{placement_xy_threshold:.3f} "
        f"zerr={z_error:+.3f}/+/-{placement_z_tolerance:.3f} "
        f"yaw={yaw_error_deg:.1f}/{placement_yaw_threshold_deg:.1f}",
        f"contact={int(contact)} stable={next_streak}/{placement_stable_steps} grip={final_gripper:.4f}",
    ]
    if reason:
        lines.append(f"why not pass: {reason}")
    return success, next_streak, lines, reason


# -----------------------------
# MAIN
# -----------------------------
def main():
    cfg = build_rollout_config(parse_args())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy_dt = 1.0 / cfg.policy_hz

    print(f"[INFO] Task: {cfg.task_name}")
    print(f"[INFO] Loading ACT checkpoint from: {cfg.checkpoint}")
    print(f"[INFO] Torch device: {device}")
    policy = ACTPolicy.from_pretrained(cfg.checkpoint)
    policy.eval()
    policy.to(device)

    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(cfg.checkpoint),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )

    for component in (policy, preprocessor, postprocessor):
        if hasattr(component, "reset"):
            component.reset()

    state_dim = policy.config.input_features["observation.state"].shape[0]
    camera_names = policy_image_camera_names(policy)

    print(f"[INFO] Loading MuJoCo model: {cfg.xml}")
    model = mujoco.MjModel.from_xml_path(str(cfg.xml))
    data = mujoco.MjData(model)

    qpos_indices = get_qpos_indices_for_joints(model, PANDA_JOINT_NAMES)
    for camera_name in camera_names:
        validate_camera_exists(model, camera_name)
    for camera_name in cfg.viewer_cameras.values():
        validate_camera_exists(model, camera_name)
    for name in [cfg.body_name, cfg.target_body_name, cfg.hand_body_name, *cfg.cup_body_names, *cfg.box_body_names]:
        validate_body_exists(model, name)

    print(f"[INFO] Resetting MuJoCo state from: {cfg.reset_npz}")
    reset_from_npz(data, cfg.reset_npz)
    episode_len = get_episode_length(cfg.reset_npz)
    expert_ctrl = load_expert_ctrl(cfg.reset_npz)
    expert_close_frames = np.flatnonzero(expert_ctrl[:, 7] < 0.02)
    expert_close_frame = int(expert_close_frames[0]) if len(expert_close_frames) else None
    rollout_step = 0

    mujoco.mj_forward(model, data)

    renderer = SimpleRenderer(
        model,
        cfg.window_w,
        cfg.window_h,
        viewer_mode=cfg.viewer_mode,
        viewer_cameras=cfg.viewer_cameras,
    )
    n_substeps = max(1, int(round(policy_dt / model.opt.timestep)))
    policy_query_count = 0
    initial_body_pos = body_pos(model, data, cfg.body_name)
    first_close_step = None
    first_lift_step = None
    contact_streak = 0
    success_step = None

    print(
        f"[INFO] Starting rollout at {cfg.policy_hz:g} Hz "
        f"with action_mode={cfg.action_mode!r}, arm_action_mode={cfg.arm_action_mode!r}, "
        f"gripper_action_mode={cfg.gripper_action_mode!r}, arm_delta_clip={cfg.arm_delta_clip!r}"
    )
    print(
        f"[INFO] ACT chunk_size={policy.config.chunk_size}, "
        f"n_action_steps={policy.config.n_action_steps}, "
        f"MuJoCo timestep={model.opt.timestep:g}, substeps/action={n_substeps}"
    )
    print(f"[INFO] Policy cameras: {camera_names}")
    print(f"[INFO] Viewer mode={cfg.viewer_mode!r}, cameras={cfg.viewer_cameras}")
    if cfg.max_steps > 0:
        print(f"[INFO] Max steps: {cfg.max_steps} ({cfg.max_steps / cfg.policy_hz:.1f}s)")
    if expert_close_frame is not None:
        expected_policy_close_frame = max(0, expert_close_frame - cfg.expert_action_shift_for_debug)
        print(
            "[INFO] Demo gripper first closes below 0.02 at "
            f"step {expert_close_frame} ({expert_close_frame * policy_dt:.2f}s)."
        )
        print(
            "[INFO] Shifted policy should start commanding that close around "
            f"step {expected_policy_close_frame} "
            f"({expected_policy_close_frame * policy_dt:.2f}s)."
        )
    print(
        f"[INFO] Viewer render every {cfg.render_every} policy steps, "
        f"realtime factor={cfg.realtime_factor}."
    )
    print("[INFO] Close the window to stop.")

    try:
        while not renderer.should_close():
            if cfg.max_steps > 0 and rollout_step >= cfg.max_steps:
                print(f"[INFO] Reached max steps: {cfg.max_steps}")
                break
            t0 = time.time()
            step = rollout_step

            renderer.poll()

            if action_needs_observation(policy, cfg.action_mode):
                state = build_state(
                    qpos_sim=np.array(data.qpos, dtype=np.float32),
                    qvel_sim=np.array(data.qvel, dtype=np.float32),
                    ctrl_sim_prev=np.array(data.ctrl, dtype=np.float32),
                    qpos_indices=qpos_indices,
                    state_dim=state_dim,
                    phase=0.0 if episode_len <= 1 else min(step / (episode_len - 1), 1.0),
                )

                observation = {
                    "observation.state": torch.from_numpy(state),
                }
                for camera_name in camera_names:
                    rgb = renderer.render_rgb(data, camera_name)
                    observation[f"observation.images.{camera_name}"] = chw_float01_from_rgb(rgb)
            else:
                observation = None

            with torch.inference_mode():
                pred_action, queried_policy = select_processed_action(
                    policy,
                    preprocessor,
                    postprocessor,
                    observation,
                    cfg.action_mode,
                )
            if queried_policy:
                policy_query_count += 1

            if cfg.arm_delta_clip is not None:
                pred_action[:7] = np.clip(pred_action[:7], -cfg.arm_delta_clip, cfg.arm_delta_clip)

            q_now = np.array(data.qpos[qpos_indices], dtype=np.float32)
            previous_ctrl = np.array(data.ctrl[:8], dtype=np.float32)

            target_ctrl = previous_ctrl.copy()
            if cfg.arm_action_mode == "absolute":
                target_ctrl[:7] = pred_action[:7]
            elif cfg.arm_action_mode == "qpos_error":
                target_ctrl[:7] = q_now + pred_action[:7]
            elif cfg.arm_action_mode == "ctrl_delta":
                target_ctrl[:7] = previous_ctrl[:7] + pred_action[:7]
            else:
                raise ValueError(f"Unknown arm_action_mode: {cfg.arm_action_mode}")

            if cfg.gripper_action_mode == "absolute":
                target_ctrl[7] = pred_action[7]
            elif cfg.gripper_action_mode == "delta":
                target_ctrl[7] = previous_ctrl[7] + pred_action[7]
            else:
                raise ValueError(f"Unknown gripper_action_mode: {cfg.gripper_action_mode}")
            target_ctrl = clip_to_actuator_ranges(model, target_ctrl)

            data.ctrl[:8] = target_ctrl
            if first_close_step is None and float(target_ctrl[7]) < cfg.close_threshold:
                first_close_step = step

            for _ in range(n_substeps):
                mujoco.mj_step(model, data)
            mujoco.mj_forward(model, data)
            rollout_step += 1

            if first_lift_step is None:
                lift_now = float(body_pos(model, data, cfg.body_name)[2] - initial_body_pos[2])
                if lift_now >= cfg.lift_threshold:
                    first_lift_step = step

            success_now, contact_streak, status_lines, reason = make_task_status_lines(
                task_name=cfg.task_name,
                model=model,
                data=data,
                body_name=cfg.body_name,
                target_body_name=cfg.target_body_name,
                initial_body_pos=initial_body_pos,
                cup_pairs=cfg.cup_pairs,
                cup_body_names=cfg.cup_body_names,
                box_body_names=cfg.box_body_names,
                box_upright_angle_deg=cfg.box_upright_angle_deg,
                cup_upright_angle_deg=cfg.cup_upright_angle_deg,
                cups_require_contact=cfg.cups_require_contact,
                placement_target_local_offset=cfg.placement_target_local_offset,
                placement_z_offset=cfg.placement_z_offset,
                placement_xy_threshold=cfg.placement_xy_threshold,
                placement_z_tolerance=cfg.placement_z_tolerance,
                placement_yaw_threshold_deg=cfg.placement_yaw_threshold_deg,
                placement_gripper_open_threshold=cfg.placement_gripper_open_threshold,
                placement_stable_steps=cfg.placement_stable_steps,
                first_close_step=first_close_step,
                first_lift_step=first_lift_step,
                contact_streak=contact_streak,
                step=step,
            )
            if success_now and success_step is None:
                success_step = step
                print(f"[PASS] {cfg.task_name} success at step {step} ({step * policy_dt:.2f}s)")

            if cfg.render_every <= 1 or step % cfg.render_every == 0:
                overlay_lines = [
                    f"task: {cfg.task_name}",
                    f"step: {step}",
                    f"policy calls: {policy_query_count}",
                    f"grip cmd: {target_ctrl[7]:.4f}",
                    f"arm step: {np.linalg.norm(target_ctrl[:7] - previous_ctrl[:7]):.4f}",
                    f"checkpoint: {cfg.checkpoint.name}",
                    *status_lines,
                ]
                renderer.render_viewer(data, overlay_lines=overlay_lines)

            elapsed = time.time() - t0
            print_debug_line(
                step=step,
                policy_query_count=policy_query_count,
                queried_policy=queried_policy,
                pred_action=pred_action,
                target_ctrl=target_ctrl,
                previous_ctrl=previous_ctrl,
                expert_ctrl=expert_ctrl,
                elapsed=elapsed,
                policy_dt=policy_dt,
                debug_every_steps=cfg.debug_every,
                debug_until_step=cfg.debug_until,
                expert_action_shift=cfg.expert_action_shift_for_debug,
            )
            if cfg.stop_on_success and success_now:
                break
            if cfg.realtime_factor > 0:
                sleep_time = policy_dt / cfg.realtime_factor - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    finally:
        renderer.close()


if __name__ == "__main__":
    main()
