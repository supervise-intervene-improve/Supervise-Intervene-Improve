import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import mujoco
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors


DEFAULT_CHECKPOINT_DIR = Path(
    "outputs/tshape_act_abs_shift1_rerender_20260504_095516/checkpoints/050000/pretrained_model"
)
DEFAULT_XML_PATH = Path("mujoco_scenes/working_scenes/with_soft_gripper/sii_scene_table_T_shape.xml")
DEFAULT_EPISODES_DIR = Path("data_clean/T_shape_clean_good")

PANDA_JOINT_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
]


@dataclass
class EpisodeResult:
    case_id: str
    episode: str
    repeat: int
    xy_jitter_x: float
    xy_jitter_y: float
    yaw_jitter_deg: float
    steps: int
    success: bool
    success_step: int | None
    first_close_step: int | None
    first_lift_step: int | None
    first_transport_step: int | None
    first_contact_step: int | None
    success_target_xy_error: float | None
    success_target_z_error: float | None
    success_target_yaw_error_deg: float | None
    success_gripper: float | None
    success_contact_steps: int | None
    target_x: float
    target_y: float
    target_z: float
    initial_x: float
    initial_y: float
    initial_z: float
    final_x: float
    final_y: float
    final_z: float
    max_z: float
    max_y: float
    displacement: float
    final_lift: float
    max_lift: float
    final_target_xy_error: float
    final_target_z_error: float
    final_target_yaw_error_deg: float
    final_gripper: float
    final_t1_t2_contact: bool
    final_contact_streak: int
    strict_bar_overlap: float | None
    strict_target_bar_overlap: float | None
    strict_lifted_bar_overlap: float | None
    strict_bar_axis_error_deg: float | None
    strict_face_alignment: float | None
    strict_stem_up_alignment: float | None
    strict_stem_height: float | None
    strict_face_gap: float | None
    strict_bar_bar_contact: bool | None
    strict_stem_target_contact: bool | None
    min_hand_object_dist: float
    mean_loop_ms: float
    policy_calls: int
    video_path: str
    fail_reason: str


def get_qpos_indices_for_joints(model: mujoco.MjModel, joint_names: list[str]) -> list[int]:
    idxs = []
    for name in joint_names:
        joint = model.joint(name)
        idxs.append(int(model.jnt_qposadr[joint.id]))
    return idxs


def chw_float01_from_rgb(rgb_hwc: np.ndarray) -> torch.Tensor:
    x = rgb_hwc.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x)


def build_state(
    qpos_sim: np.ndarray,
    qvel_sim: np.ndarray,
    ctrl_sim_prev: np.ndarray,
    qpos_indices: list[int],
    state_dim: int,
    phase: float,
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


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
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


def yaw_quat(yaw_rad: float) -> np.ndarray:
    half = 0.5 * yaw_rad
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


def reset_from_npz(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    npz_path: Path,
    free_joint_name: str,
    xy_jitter: tuple[float, float] = (0.0, 0.0),
    yaw_jitter_rad: float = 0.0,
) -> dict[str, np.ndarray]:
    episode = dict(np.load(npz_path, allow_pickle=False))
    for key in ("qpos_sim", "qvel_sim", "ctrl_sim"):
        if key not in episode:
            raise KeyError(f"{npz_path} missing key: {key}")

    data.qpos[:] = episode["qpos_sim"][0]
    data.qvel[:] = episode["qvel_sim"][0]
    n_ctrl = min(data.ctrl.shape[0], episode["ctrl_sim"].shape[1])
    data.ctrl[:n_ctrl] = episode["ctrl_sim"][0, :n_ctrl]

    if xy_jitter != (0.0, 0.0) or yaw_jitter_rad != 0.0:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, free_joint_name)
        if joint_id < 0:
            raise ValueError(f"Free joint not found: {free_joint_name}")
        qpos_adr = int(model.jnt_qposadr[joint_id])
        data.qpos[qpos_adr] += xy_jitter[0]
        data.qpos[qpos_adr + 1] += xy_jitter[1]
        if yaw_jitter_rad != 0.0:
            quat_adr = qpos_adr + 3
            base_quat = np.array(data.qpos[quat_adr : quat_adr + 4], dtype=np.float64)
            rotated_quat = quat_mul(yaw_quat(yaw_jitter_rad), base_quat)
            quat_norm = np.linalg.norm(rotated_quat)
            if quat_norm <= 0:
                raise ValueError(f"Invalid free-joint quaternion for {free_joint_name}")
            data.qpos[quat_adr : quat_adr + 4] = rotated_quat / quat_norm
    return episode


def clip_to_actuator_ranges(model: mujoco.MjModel, ctrl: np.ndarray) -> np.ndarray:
    clipped = ctrl.copy()
    for i in range(min(model.nu, clipped.shape[0])):
        if model.actuator_ctrllimited[i]:
            lo, hi = model.actuator_ctrlrange[i]
            clipped[i] = np.clip(clipped[i], lo, hi)
    return clipped


def reset_component(component) -> None:
    if hasattr(component, "reset"):
        component.reset()


class PolicyRenderer:
    def __init__(self, model: mujoco.MjModel, width: int = 224, height: int = 224):
        self.renderer = mujoco.Renderer(model, height=height, width=width)

    def render_rgb(self, data: mujoco.MjData, camera: str) -> np.ndarray:
        self.renderer.disable_depth_rendering()
        self.renderer.update_scene(data, camera=camera)
        return self.renderer.render().copy()

    def close(self) -> None:
        self.renderer.close()


class EpisodeVideoRecorder:
    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        camera: str,
        width: int,
        height: int,
        fps: float,
        every_n_steps: int,
    ):
        try:
            import cv2
        except ImportError as exc:
            raise RuntimeError("Video recording requires OpenCV/Python package 'cv2'.") from exc

        self.cv2 = cv2
        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.cameras = [part.strip() for part in re.split(r"[, ]+", camera) if part.strip()]
        if not self.cameras:
            raise ValueError("--video_camera must contain at least one camera name")
        self.width = width
        self.height = height
        self.fps = fps
        self.every_n_steps = every_n_steps
        self.frames: list[np.ndarray] = []

    def reset(self) -> None:
        self.frames = []

    def capture(self, data: mujoco.MjData) -> None:
        self.renderer.disable_depth_rendering()
        views = []
        for camera in self.cameras:
            self.renderer.update_scene(data, camera=camera)
            view = self.renderer.render().copy()
            self.cv2.putText(
                view,
                camera,
                (10, 24),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                self.cv2.LINE_AA,
            )
            views.append(view)
        self.frames.append(np.concatenate(views, axis=1))

    def maybe_capture_step(self, data: mujoco.MjData, step: int) -> None:
        if step % self.every_n_steps == 0:
            self.capture(data)

    def write(self, path: Path) -> None:
        if not self.frames:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = self.cv2.VideoWriter_fourcc(*"mp4v")
        writer_width = self.width * len(self.cameras)
        writer = self.cv2.VideoWriter(str(path), fourcc, self.fps, (writer_width, self.height))
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer for {path}")
        try:
            for frame_rgb in self.frames:
                writer.write(self.cv2.cvtColor(frame_rgb, self.cv2.COLOR_RGB2BGR))
        finally:
            writer.release()

    def close(self) -> None:
        self.renderer.close()


def policy_needs_observation(policy: ACTPolicy, action_mode: str) -> bool:
    if action_mode == "replan":
        return True
    if action_mode == "queue":
        if getattr(policy.config, "temporal_ensemble_coeff", None) is not None:
            return True
        return len(policy._action_queue) == 0
    raise ValueError(f"Unknown action mode: {action_mode}")


def select_action(
    policy: ACTPolicy,
    preprocessor,
    postprocessor,
    observation: dict[str, torch.Tensor] | None,
    action_mode: str,
) -> tuple[np.ndarray, bool]:
    queried_policy = policy_needs_observation(policy, action_mode)
    batch = preprocessor(observation) if queried_policy else {}

    if action_mode == "replan":
        if observation is None:
            raise ValueError("action_mode='replan' requires an observation every step")
        raw_chunk = policy.predict_action_chunk(batch)
        raw_action = raw_chunk[:, 0]
    elif action_mode == "queue":
        raw_action = policy.select_action(batch)
    else:
        raise ValueError(f"Unknown action mode: {action_mode}")

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


def body_pos(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body not found: {body_name}")
    return data.xpos[body_id].copy()


def body_yaw(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> float:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Body not found: {body_name}")
    rot = data.xmat[body_id].reshape(3, 3)
    for axis_index in (0, 2, 1):
        axis = rot[:, axis_index].copy()
        axis[2] = 0.0
        if np.linalg.norm(axis) > 1e-6:
            return float(np.arctan2(axis[1], axis[0]))
    return 0.0


def body_id(model: mujoco.MjModel, body_name: str) -> int:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise ValueError(f"Body not found: {body_name}")
    return int(bid)


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


def geom_id(model: mujoco.MjModel, geom_name: str) -> int:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    if gid < 0:
        raise ValueError(f"Geom not found: {geom_name}")
    return int(gid)


def geoms_in_contact(model: mujoco.MjModel, data: mujoco.MjData, geom_a: str, geom_b: str) -> bool:
    geom_a_id = geom_id(model, geom_a)
    geom_b_id = geom_id(model, geom_b)
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        if (contact.geom1 == geom_a_id and contact.geom2 == geom_b_id) or (
            contact.geom1 == geom_b_id and contact.geom2 == geom_a_id
        ):
            return True
    return False


def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _signed_polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    return 0.5 * float(
        np.dot(poly[:, 0], np.roll(poly[:, 1], -1))
        - np.dot(poly[:, 1], np.roll(poly[:, 0], -1))
    )


def _polygon_area(poly: np.ndarray) -> float:
    return abs(_signed_polygon_area(poly))


def _ensure_ccw(poly: np.ndarray) -> np.ndarray:
    if len(poly) >= 3 and _signed_polygon_area(poly) < 0:
        return poly[::-1].copy()
    return poly


def _line_intersection_2d(p1: np.ndarray, p2: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d1 = p2 - p1
    d2 = b - a
    denom = _cross2(d1, d2)
    if abs(denom) < 1e-12:
        return p2.copy()
    t = _cross2(a - p1, d2) / denom
    return p1 + t * d1


def _clip_polygon_against_convex(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    subject = _ensure_ccw(np.asarray(subject, dtype=np.float64))
    clip = _ensure_ccw(np.asarray(clip, dtype=np.float64))
    output = subject
    eps = 1e-10

    for i in range(len(clip)):
        edge_a = clip[i]
        edge_b = clip[(i + 1) % len(clip)]
        if len(output) == 0:
            break
        input_poly = output
        output_points = []

        prev = input_poly[-1]
        prev_inside = _cross2(edge_b - edge_a, prev - edge_a) >= -eps
        for cur in input_poly:
            cur_inside = _cross2(edge_b - edge_a, cur - edge_a) >= -eps
            if cur_inside:
                if not prev_inside:
                    output_points.append(_line_intersection_2d(prev, cur, edge_a, edge_b))
                output_points.append(cur)
            elif prev_inside:
                output_points.append(_line_intersection_2d(prev, cur, edge_a, edge_b))
            prev = cur
            prev_inside = cur_inside

        output = np.asarray(output_points, dtype=np.float64)

    return output


def _geom_axes(data: mujoco.MjData, gid: int) -> np.ndarray:
    return data.geom_xmat[gid].reshape(3, 3)


def _top_face_axis(axes: np.ndarray, up: np.ndarray) -> tuple[int, float, np.ndarray]:
    dots = axes.T @ up
    axis_index = int(np.argmax(np.abs(dots)))
    sign = 1.0 if dots[axis_index] >= 0 else -1.0
    normal = sign * axes[:, axis_index]
    return axis_index, abs(float(dots[axis_index])), normal


def _project_axis_to_plane(axis: np.ndarray, normal: np.ndarray) -> np.ndarray:
    projected = axis - normal * float(np.dot(axis, normal))
    norm = float(np.linalg.norm(projected))
    if norm <= 1e-9:
        return projected
    return projected / norm


def _face_polygon_2d(
    *,
    center: np.ndarray,
    axes: list[np.ndarray],
    half_extents: list[float],
    basis_u: np.ndarray,
    basis_v: np.ndarray,
) -> np.ndarray:
    corners = []
    for su, sv in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]:
        point = (
            center
            + su * float(half_extents[0]) * axes[0]
            + sv * float(half_extents[1]) * axes[1]
        )
        corners.append([float(np.dot(point, basis_u)), float(np.dot(point, basis_v))])
    return _ensure_ccw(np.asarray(corners, dtype=np.float64))


def upright_angle_deg(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> float:
    bid = body_id(model, body_name)
    rot = data.xmat[bid].reshape(3, 3)
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

    for body_name in body_names:
        angle = upright_angle_deg(model, data, body_name)
        if angle > worst_angle:
            worst_angle = angle
            worst_body = body_name

    return worst_angle <= max_angle_deg, worst_angle, worst_body


def boxes_upright_status(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    box_body_names: list[str],
    max_angle_deg: float,
) -> tuple[bool, float, str]:
    return bodies_upright_status(model, data, box_body_names, max_angle_deg)


_CUP_GEOM_CACHE: dict[tuple[int, str], dict] = {}


def _quat_to_mat(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in quat]
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _body_mesh_vertices_local(model: mujoco.MjModel, body_name: str) -> np.ndarray:
    bid = body_id(model, body_name)
    vertices = []
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) != bid:
            continue

        geom_type = int(model.geom_type[gid])
        geom_rot = _quat_to_mat(model.geom_quat[gid])
        geom_pos = model.geom_pos[gid].copy()
        if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
            mesh_id = int(model.geom_dataid[gid])
            if mesh_id < 0:
                continue

            start = int(model.mesh_vertadr[mesh_id])
            count = int(model.mesh_vertnum[mesh_id])
            if count <= 0:
                continue
            local_vertices = model.mesh_vert[start:start + count].copy()
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            sx, sy, sz = model.geom_size[gid]
            local_vertices = np.asarray(
                [[x, y, z] for x in (-sx, sx) for y in (-sy, sy) for z in (-sz, sz)],
                dtype=np.float64,
            )
        elif geom_type in {int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_CAPSULE)}:
            radius = float(model.geom_size[gid][0])
            half_length = float(model.geom_size[gid][1])
            points = []
            for z in (-half_length, 0.0, half_length):
                for theta in np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False):
                    points.append([radius * np.cos(theta), radius * np.sin(theta), z])
            local_vertices = np.asarray(points, dtype=np.float64)
        elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
            radius = float(model.geom_size[gid][0])
            local_vertices = np.asarray(
                [
                    [0.0, 0.0, radius],
                    [0.0, 0.0, -radius],
                    [radius, 0.0, 0.0],
                    [-radius, 0.0, 0.0],
                    [0.0, radius, 0.0],
                    [0.0, -radius, 0.0],
                ],
                dtype=np.float64,
            )
        else:
            continue

        vertices.append(local_vertices @ geom_rot.T + geom_pos)

    if not vertices:
        return np.zeros((0, 3), dtype=np.float64)
    return np.concatenate(vertices, axis=0)


def cup_geometry_descriptor(model: mujoco.MjModel, body_name: str) -> dict:
    cache_key = (id(model), body_name)
    cached = _CUP_GEOM_CACHE.get(cache_key)
    if cached is not None:
        return cached

    vertices = _body_mesh_vertices_local(model, body_name)
    if len(vertices) == 0:
        descriptor = {
            "bottom_center_local": np.zeros(3, dtype=np.float64),
            "rim_center_local": np.zeros(3, dtype=np.float64),
            "bottom_radius": 0.0,
            "rim_radius": 0.0,
            "height": 0.0,
        }
        _CUP_GEOM_CACHE[cache_key] = descriptor
        return descriptor

    z_min = float(np.percentile(vertices[:, 2], 1.0))
    z_max = float(np.percentile(vertices[:, 2], 99.0))
    height = max(z_max - z_min, 1e-6)
    band = max(0.006, 0.12 * height)

    bottom_vertices = vertices[vertices[:, 2] <= z_min + band]
    rim_vertices = vertices[vertices[:, 2] >= z_max - band]
    if len(bottom_vertices) == 0:
        bottom_vertices = vertices
    if len(rim_vertices) == 0:
        rim_vertices = vertices

    bottom_xy = np.median(bottom_vertices[:, :2], axis=0)
    rim_xy = np.median(rim_vertices[:, :2], axis=0)
    bottom_center = np.array([bottom_xy[0], bottom_xy[1], float(np.median(bottom_vertices[:, 2]))])
    rim_center = np.array([rim_xy[0], rim_xy[1], float(np.median(rim_vertices[:, 2]))])

    bottom_radii = np.linalg.norm(bottom_vertices[:, :2] - bottom_xy, axis=1)
    rim_radii = np.linalg.norm(rim_vertices[:, :2] - rim_xy, axis=1)
    descriptor = {
        "bottom_center_local": bottom_center,
        "rim_center_local": rim_center,
        "bottom_radius": float(np.percentile(bottom_radii, 70.0)) if len(bottom_radii) else 0.0,
        "rim_radius": float(np.percentile(rim_radii, 70.0)) if len(rim_radii) else 0.0,
        "height": height,
    }
    _CUP_GEOM_CACHE[cache_key] = descriptor
    return descriptor


def _body_local_to_world(model: mujoco.MjModel, data: mujoco.MjData, body_name: str, local_point: np.ndarray) -> np.ndarray:
    bid = body_id(model, body_name)
    rot = data.xmat[bid].reshape(3, 3)
    return data.xpos[bid].copy() + rot @ local_point


def _body_world_to_local(model: mujoco.MjModel, data: mujoco.MjData, body_name: str, world_point: np.ndarray) -> np.ndarray:
    bid = body_id(model, body_name)
    rot = data.xmat[bid].reshape(3, 3)
    return rot.T @ (world_point - data.xpos[bid])


def body_velocity_norms(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> tuple[float, float]:
    bid = body_id(model, body_name)
    cvel = data.cvel[bid]
    return float(np.linalg.norm(cvel[3:])), float(np.linalg.norm(cvel[:3]))


def first_contact_body(model: mujoco.MjModel, data: mujoco.MjData, body_name: str, candidates: list[str]) -> str:
    for candidate in candidates:
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, candidate) < 0:
            continue
        if bodies_in_contact(model, data, body_name, candidate):
            return candidate
    return ""


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
    target_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_body_name)
    if target_body_id < 0:
        raise ValueError(f"Body not found: {target_body_name}")
    target_body_pos = data.xpos[target_body_id].copy()
    target_rot = data.xmat[target_body_id].reshape(3, 3)
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
    all_cup_body_names: list[str],
    bad_contact_body_names: list[str],
    target_local_offset: np.ndarray,
    placement_z_offset: float,
    placement_xy_threshold: float,
    placement_z_tolerance: float,
    require_contact: bool,
    use_advanced_metric: bool,
    max_axis_error_deg: float,
    min_radial_margin: float,
    forbid_upper_bad_contacts: bool,
    forbid_upper_wrong_contacts: bool,
    max_pair_linear_speed: float,
    max_pair_angular_speed: float,
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
    legacy_placed = (
        xy_error <= placement_xy_threshold
        and abs(z_error) <= placement_z_tolerance
        and (contact or not require_contact)
    )

    cup_desc = cup_geometry_descriptor(model, cup_body_name)
    target_desc = cup_geometry_descriptor(model, target_body_name)
    cup_bottom_world = _body_local_to_world(model, data, cup_body_name, cup_desc["bottom_center_local"])
    target_bottom_world = _body_local_to_world(model, data, target_body_name, target_desc["bottom_center_local"])
    target_bottom_local = _body_world_to_local(model, data, target_body_name, target_bottom_world)
    cup_bottom_in_target = _body_world_to_local(model, data, target_body_name, cup_bottom_world)
    target_plane_delta = cup_bottom_in_target[:2] - target_bottom_local[:2]
    radial_error = float(np.linalg.norm(target_plane_delta))
    vertical_error = float(cup_bottom_in_target[2] - (target_bottom_local[2] + placement_z_offset))

    cup_bid = body_id(model, cup_body_name)
    target_bid = body_id(model, target_body_name)
    cup_up = data.xmat[cup_bid].reshape(3, 3)[:, 2]
    target_up = data.xmat[target_bid].reshape(3, 3)[:, 2]
    axis_alignment = float(np.clip(np.dot(cup_up, target_up), -1.0, 1.0))
    axis_error_deg = float(np.degrees(np.arccos(axis_alignment)))

    radial_margin = float(target_desc["rim_radius"] - (radial_error + cup_desc["bottom_radius"]))
    bad_contact_body = first_contact_body(model, data, cup_body_name, bad_contact_body_names)
    wrong_cup_candidates = [
        name for name in all_cup_body_names
        if name not in {cup_body_name, target_body_name}
    ]
    wrong_contact_body = first_contact_body(model, data, cup_body_name, wrong_cup_candidates)
    cup_linear_speed, cup_angular_speed = body_velocity_norms(model, data, cup_body_name)
    target_linear_speed, target_angular_speed = body_velocity_norms(model, data, target_body_name)
    pair_linear_speed = max(cup_linear_speed, target_linear_speed)
    pair_angular_speed = max(cup_angular_speed, target_angular_speed)

    advanced_checks_ok = (
        radial_error <= placement_xy_threshold
        and abs(vertical_error) <= placement_z_tolerance
        and axis_error_deg <= max_axis_error_deg
        and radial_margin >= min_radial_margin
        and (contact or not require_contact)
        and (not bad_contact_body or not forbid_upper_bad_contacts)
        and (not wrong_contact_body or not forbid_upper_wrong_contacts)
        and (max_pair_linear_speed <= 0 or pair_linear_speed <= max_pair_linear_speed)
        and (max_pair_angular_speed <= 0 or pair_angular_speed <= max_pair_angular_speed)
    )
    placed = bool(advanced_checks_ok if use_advanced_metric else legacy_placed)
    return {
        "cup": cup_body_name,
        "target": target_body_name,
        "target_pos": target_pos,
        "xy_error": xy_error,
        "z_error": z_error,
        "yaw_error_deg": yaw_error_deg,
        "contact": contact,
        "legacy_placed": legacy_placed,
        "advanced": use_advanced_metric,
        "radial_error": radial_error,
        "vertical_error": vertical_error,
        "axis_error_deg": axis_error_deg,
        "axis_alignment": axis_alignment,
        "radial_margin": radial_margin,
        "cup_bottom_radius": float(cup_desc["bottom_radius"]),
        "target_rim_radius": float(target_desc["rim_radius"]),
        "bad_contact_body": bad_contact_body,
        "wrong_contact_body": wrong_contact_body,
        "forbid_upper_bad_contacts": forbid_upper_bad_contacts,
        "forbid_upper_wrong_contacts": forbid_upper_wrong_contacts,
        "pair_linear_speed": pair_linear_speed,
        "pair_angular_speed": pair_angular_speed,
        "placement_xy_threshold": placement_xy_threshold,
        "placement_z_tolerance": placement_z_tolerance,
        "max_axis_error_deg": max_axis_error_deg,
        "min_radial_margin": min_radial_margin,
        "max_pair_linear_speed": max_pair_linear_speed,
        "max_pair_angular_speed": max_pair_angular_speed,
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
    use_advanced_metric: bool = True,
    max_axis_error_deg: float = 20.0,
    min_radial_margin: float = -0.005,
    forbid_upper_bad_contacts: bool = False,
    forbid_upper_wrong_contacts: bool = False,
    bad_contact_body_names: list[str] | None = None,
    max_pair_linear_speed: float = 0.0,
    max_pair_angular_speed: float = 0.0,
) -> dict:
    target_local_offset = np.asarray(placement_target_local_offset, dtype=np.float64)
    bad_contact_body_names = bad_contact_body_names or ["table", *box_body_names]
    pair_statuses = [
        cup_pair_status(
            model,
            data,
            cup_body_name=cup,
            target_body_name=target,
            all_cup_body_names=cup_body_names,
            bad_contact_body_names=bad_contact_body_names,
            target_local_offset=target_local_offset,
            placement_z_offset=placement_z_offset,
            placement_xy_threshold=placement_xy_threshold,
            placement_z_tolerance=placement_z_tolerance,
            require_contact=require_contact,
            use_advanced_metric=use_advanced_metric,
            max_axis_error_deg=max_axis_error_deg,
            min_radial_margin=min_radial_margin,
            forbid_upper_bad_contacts=forbid_upper_bad_contacts,
            forbid_upper_wrong_contacts=forbid_upper_wrong_contacts,
            max_pair_linear_speed=max_pair_linear_speed,
            max_pair_angular_speed=max_pair_angular_speed,
        )
        for cup, target in cup_pairs
    ]
    boxes_ok, max_box_angle, worst_box = boxes_upright_status(
        model,
        data,
        box_body_names=box_body_names,
        max_angle_deg=box_upright_angle_deg,
    )
    cups_ok, max_cup_angle, worst_cup = bodies_upright_status(
        model,
        data,
        body_names=cup_body_names,
        max_angle_deg=cup_upright_angle_deg,
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
        "max_pair_xy_error": max(status["xy_error"] for status in pair_statuses),
        "max_pair_abs_z_error": max(abs(status["z_error"]) for status in pair_statuses),
        "max_pair_radial_error": max(status["radial_error"] for status in pair_statuses),
        "max_pair_abs_vertical_error": max(abs(status["vertical_error"]) for status in pair_statuses),
        "min_pair_radial_margin": min(status["radial_margin"] for status in pair_statuses),
        "max_pair_axis_error_deg": max(status["axis_error_deg"] for status in pair_statuses),
        "max_pair_linear_speed": max(status["pair_linear_speed"] for status in pair_statuses),
        "max_pair_angular_speed": max(status["pair_angular_speed"] for status in pair_statuses),
        "all_pairs_contact": all(status["contact"] for status in pair_statuses),
        "use_advanced_metric": use_advanced_metric,
    }


def cups_fail_reason(status: dict, placement_stable_steps: int, contact_streak: int) -> str:
    if not status["boxes_ok"]:
        return f"box_fell:{status['worst_box']}:{status['max_box_angle_deg']:.1f}deg"
    if not status["cups_ok"]:
        return f"cup_fell:{status['worst_cup']}:{status['max_cup_angle_deg']:.1f}deg"

    for pair in status["pairs"]:
        name = f"{pair['cup']}_on_{pair['target']}"
        if pair.get("advanced"):
            if pair["bad_contact_body"] and pair.get("forbid_upper_bad_contacts", False):
                return f"{name}_bad_contact:{pair['bad_contact_body']}"
            if pair["wrong_contact_body"] and pair.get("forbid_upper_wrong_contacts", False):
                return f"{name}_touching_wrong_cup:{pair['wrong_contact_body']}"
        if not pair["contact"]:
            return f"{name}_not_touching"
        if pair.get("advanced"):
            if pair["axis_error_deg"] > pair["max_axis_error_deg"]:
                return f"{name}_tilted_relative:{pair['axis_error_deg']:.1f}deg"
            if pair["radial_margin"] < pair["min_radial_margin"]:
                return f"{name}_not_nested:{pair['radial_margin']:.3f}m"
            if abs(pair["vertical_error"]) > pair["placement_z_tolerance"]:
                return f"{name}_bad_z:{pair['vertical_error']:.3f}m"
            if (
                (
                    pair["max_pair_linear_speed"] > 0
                    and pair["pair_linear_speed"] > pair["max_pair_linear_speed"]
                )
                or (
                    pair["max_pair_angular_speed"] > 0
                    and pair["pair_angular_speed"] > pair["max_pair_angular_speed"]
                )
            ):
                return f"{name}_not_settled"
            if pair["radial_error"] > pair["placement_xy_threshold"]:
                return f"{name}_not_centered:{pair['radial_error']:.3f}m"
            if not pair["placed"]:
                return f"{name}_not_placed"
        if not pair["placed"]:
            if abs(pair["z_error"]) > pair["xy_error"]:
                return f"{name}_bad_z"
            return f"{name}_not_centered"

    if not status["gripper_open"]:
        return "gripper_not_open"
    return ""


def tshape_success_status(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    lifted_bar_geom: str,
    lifted_stem_geom: str,
    target_bar_geom: str,
    target_stem_geom: str,
    placement_gripper_open_threshold: float,
    min_bar_overlap: float,
    max_bar_axis_error_deg: float,
    min_face_alignment: float,
    min_stem_up_alignment: float,
    min_stem_height: float,
    max_face_gap: float,
    min_target_top_alignment: float,
    require_bar_contact: bool,
    forbid_stem_contact: bool,
) -> dict:
    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    lifted_bar_id = geom_id(model, lifted_bar_geom)
    lifted_stem_id = geom_id(model, lifted_stem_geom)
    target_bar_id = geom_id(model, target_bar_geom)
    target_stem_id = geom_id(model, target_stem_geom)

    lifted_bar_axes = _geom_axes(data, lifted_bar_id)
    target_bar_axes = _geom_axes(data, target_bar_id)
    target_top_axis, target_top_world_dot, target_top_normal = _top_face_axis(target_bar_axes, up)

    target_face_axis_indices = [idx for idx in range(3) if idx != target_top_axis]
    target_basis_u = _project_axis_to_plane(target_bar_axes[:, target_face_axis_indices[0]], target_top_normal)
    if np.linalg.norm(target_basis_u) <= 1e-9:
        target_basis_u = _project_axis_to_plane(target_bar_axes[:, target_face_axis_indices[1]], target_top_normal)
    target_basis_v = np.cross(target_top_normal, target_basis_u)
    target_basis_v /= max(np.linalg.norm(target_basis_v), 1e-9)

    target_top_center = (
        data.geom_xpos[target_bar_id].copy()
        + target_top_normal * float(model.geom_size[target_bar_id][target_top_axis])
    )
    target_face_poly = _face_polygon_2d(
        center=target_top_center,
        axes=[target_bar_axes[:, idx] for idx in target_face_axis_indices],
        half_extents=[float(model.geom_size[target_bar_id][idx]) for idx in target_face_axis_indices],
        basis_u=target_basis_u,
        basis_v=target_basis_v,
    )

    # For these T-shape assets, the bar's local +Z face is the face opposite the stem.
    # In the desired upside-down placement that +Z face points down into T2_bar.
    lifted_contact_axis = 2
    lifted_contact_normal = lifted_bar_axes[:, lifted_contact_axis]
    lifted_contact_center = (
        data.geom_xpos[lifted_bar_id].copy()
        + lifted_contact_normal * float(model.geom_size[lifted_bar_id][lifted_contact_axis])
    )
    lifted_face_axis_indices = [idx for idx in range(3) if idx != lifted_contact_axis]
    lifted_face_poly = _face_polygon_2d(
        center=lifted_contact_center,
        axes=[lifted_bar_axes[:, idx] for idx in lifted_face_axis_indices],
        half_extents=[float(model.geom_size[lifted_bar_id][idx]) for idx in lifted_face_axis_indices],
        basis_u=target_basis_u,
        basis_v=target_basis_v,
    )

    overlap_poly = _clip_polygon_against_convex(lifted_face_poly, target_face_poly)
    overlap_area = _polygon_area(overlap_poly)
    lifted_face_area = _polygon_area(lifted_face_poly)
    target_face_area = _polygon_area(target_face_poly)
    lifted_overlap = overlap_area / lifted_face_area if lifted_face_area > 1e-12 else 0.0
    target_overlap = overlap_area / target_face_area if target_face_area > 1e-12 else 0.0
    bar_overlap = min(lifted_overlap, target_overlap)

    lifted_long_axis = _project_axis_to_plane(lifted_bar_axes[:, 0], target_top_normal)
    target_long_axis = _project_axis_to_plane(target_bar_axes[:, 0], target_top_normal)
    if np.linalg.norm(lifted_long_axis) <= 1e-9 or np.linalg.norm(target_long_axis) <= 1e-9:
        bar_axis_error_deg = 180.0
    else:
        axis_dot = abs(float(np.clip(np.dot(lifted_long_axis, target_long_axis), -1.0, 1.0)))
        bar_axis_error_deg = float(np.degrees(np.arccos(axis_dot)))

    face_alignment = -float(np.dot(lifted_contact_normal, target_top_normal))
    face_gap = float(np.dot(lifted_contact_center - target_top_center, target_top_normal))

    lifted_bar_pos = data.geom_xpos[lifted_bar_id].copy()
    lifted_stem_pos = data.geom_xpos[lifted_stem_id].copy()
    stem_vec = lifted_stem_pos - lifted_bar_pos
    stem_norm = float(np.linalg.norm(stem_vec))
    stem_up_alignment = float(np.dot(stem_vec / stem_norm, target_top_normal)) if stem_norm > 1e-9 else -1.0
    stem_height = float(np.dot(stem_vec, target_top_normal))

    bar_bar_contact = geoms_in_contact(model, data, lifted_bar_geom, target_bar_geom)
    stem_target_contact = geoms_in_contact(model, data, lifted_stem_geom, target_bar_geom) or geoms_in_contact(
        model,
        data,
        lifted_stem_geom,
        target_stem_geom,
    )
    gripper_open = (
        placement_gripper_open_threshold <= 0
        or float(data.ctrl[7]) >= placement_gripper_open_threshold
    )

    center_delta = lifted_contact_center - target_top_center
    center_plane_delta = center_delta - target_top_normal * float(np.dot(center_delta, target_top_normal))
    bar_center_error = float(np.linalg.norm(center_plane_delta))

    success_now = (
        target_top_world_dot >= min_target_top_alignment
        and bar_overlap >= min_bar_overlap
        and bar_axis_error_deg <= max_bar_axis_error_deg
        and face_alignment >= min_face_alignment
        and stem_up_alignment >= min_stem_up_alignment
        and stem_height >= min_stem_height
        and abs(face_gap) <= max_face_gap
        and (bar_bar_contact or not require_bar_contact)
        and (not stem_target_contact or not forbid_stem_contact)
        and gripper_open
    )

    return {
        "success_now": bool(success_now),
        "lifted_bar_geom": lifted_bar_geom,
        "lifted_stem_geom": lifted_stem_geom,
        "target_bar_geom": target_bar_geom,
        "target_stem_geom": target_stem_geom,
        "target_top_world_dot": target_top_world_dot,
        "bar_overlap": bar_overlap,
        "target_bar_overlap": target_overlap,
        "lifted_bar_overlap": lifted_overlap,
        "bar_axis_error_deg": bar_axis_error_deg,
        "face_alignment": face_alignment,
        "stem_up_alignment": stem_up_alignment,
        "stem_height": stem_height,
        "face_gap": face_gap,
        "bar_center_error": bar_center_error,
        "bar_bar_contact": bar_bar_contact,
        "stem_target_contact": stem_target_contact,
        "gripper_open": gripper_open,
    }


def tshape_fail_reason(
    status: dict,
    *,
    first_close_step: int | None,
    first_lift_step: int | None,
    placement_stable_steps: int,
    contact_streak: int,
    min_target_top_alignment: float,
    min_bar_overlap: float,
    max_bar_axis_error_deg: float,
    min_face_alignment: float,
    min_stem_up_alignment: float,
    min_stem_height: float,
    max_face_gap: float,
    placement_gripper_open_threshold: float,
    require_bar_contact: bool,
    forbid_stem_contact: bool,
) -> str:
    if first_close_step is None:
        return "no_gripper_close"
    if first_lift_step is None:
        return "no_lift"
    if status["target_top_world_dot"] < min_target_top_alignment:
        return f"target_bar_not_flat:{status['target_top_world_dot']:.2f}"
    if require_bar_contact and not status["bar_bar_contact"]:
        return "T1_bar_not_touching_T2_bar"
    if forbid_stem_contact and status["stem_target_contact"]:
        return "T1_stem_touching_T2"
    if status["bar_overlap"] < min_bar_overlap:
        return f"bar_overlap_low:{status['bar_overlap']:.2f}"
    if status["bar_axis_error_deg"] > max_bar_axis_error_deg:
        return f"bar_axis_misaligned:{status['bar_axis_error_deg']:.1f}deg"
    if status["face_alignment"] < min_face_alignment:
        return f"wrong_T1_bar_face:{status['face_alignment']:.2f}"
    if status["stem_up_alignment"] < min_stem_up_alignment:
        return f"T1_stem_not_up:{status['stem_up_alignment']:.2f}"
    if status["stem_height"] < min_stem_height:
        return f"T1_stem_too_low:{status['stem_height']:.3f}m"
    if abs(status["face_gap"]) > max_face_gap:
        if status["face_gap"] > 0:
            return f"bar_face_gap_too_high:{status['face_gap']:.3f}m"
        return f"bar_face_gap_penetrating:{status['face_gap']:.3f}m"
    if placement_gripper_open_threshold > 0 and not status["gripper_open"]:
        return "gripper_not_open"
    return ""


def _normalized(vec: np.ndarray, *, fallback: tuple[float, float, float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-9:
        return np.asarray(fallback, dtype=np.float64)
    return arr / norm


def wiregame_reference_distance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    spoon_body_name: str,
    wire_body_name: str,
    spoon_ring_center_local: tuple[float, float, float],
    wire_base_target_local: tuple[float, float, float],
) -> float:
    ring_world = _body_local_to_world(
        model,
        data,
        spoon_body_name,
        np.asarray(spoon_ring_center_local, dtype=np.float64),
    )
    target_world = _body_local_to_world(
        model,
        data,
        wire_body_name,
        np.asarray(wire_base_target_local, dtype=np.float64),
    )
    return float(np.linalg.norm(ring_world - target_world))


def wiregame_success_status(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    spoon_body_name: str,
    wire_body_name: str,
    initial_spoon_pos: np.ndarray,
    initial_wire_pos: np.ndarray,
    initial_end_distance: float,
    spoon_ring_center_local: tuple[float, float, float],
    spoon_ring_normal_local: tuple[float, float, float],
    wire_base_target_local: tuple[float, float, float],
    max_end_distance: float,
    max_aperture_radius: float,
    max_aperture_plane_dist: float,
    min_progress: float,
    max_wire_tilt_deg: float,
    max_wire_xy_displacement: float,
    max_wire_z_lift: float,
    max_spoon_drop: float,
    require_base_contact: bool,
) -> dict:
    ring_center_local = np.asarray(spoon_ring_center_local, dtype=np.float64)
    ring_normal_local = _normalized(
        np.asarray(spoon_ring_normal_local, dtype=np.float64),
        fallback=(0.0, 0.0, 1.0),
    )
    target_local = np.asarray(wire_base_target_local, dtype=np.float64)

    ring_world = _body_local_to_world(model, data, spoon_body_name, ring_center_local)
    target_world = _body_local_to_world(model, data, wire_body_name, target_local)
    target_in_spoon = _body_world_to_local(model, data, spoon_body_name, target_world)
    aperture_delta = target_in_spoon - ring_center_local
    aperture_plane_signed = float(np.dot(aperture_delta, ring_normal_local))
    aperture_plane_error = abs(aperture_plane_signed)
    aperture_radial = aperture_delta - aperture_plane_signed * ring_normal_local
    aperture_radial_error = float(np.linalg.norm(aperture_radial))

    end_distance = float(np.linalg.norm(ring_world - target_world))
    progress = float(initial_end_distance - end_distance)
    base_contact = bodies_in_contact(model, data, spoon_body_name, wire_body_name)
    aperture_threaded = (
        aperture_radial_error <= max_aperture_radius
        and aperture_plane_error <= max_aperture_plane_dist
    )
    progress_ok = end_distance <= max_end_distance and progress >= min_progress

    wire_pos = body_pos(model, data, wire_body_name)
    spoon_pos = body_pos(model, data, spoon_body_name)
    wire_tilt_deg = upright_angle_deg(model, data, wire_body_name)
    wire_xy_displacement = float(np.linalg.norm(wire_pos[:2] - initial_wire_pos[:2]))
    wire_z_lift = float(max(0.0, wire_pos[2] - initial_wire_pos[2]))
    spoon_z_delta = float(spoon_pos[2] - initial_spoon_pos[2])

    wire_fell = wire_tilt_deg > max_wire_tilt_deg
    wire_displaced = wire_xy_displacement > max_wire_xy_displacement
    wire_lifted = wire_z_lift > max_wire_z_lift
    spoon_dropped = spoon_z_delta < -max_spoon_drop
    safety_ok = not (wire_fell or wire_displaced or wire_lifted or spoon_dropped)

    success_now = (
        safety_ok
        and (base_contact or not require_base_contact)
        and aperture_threaded
        and progress_ok
    )

    return {
        "success_now": bool(success_now),
        "spoon_body_name": spoon_body_name,
        "wire_body_name": wire_body_name,
        "base_contact": bool(base_contact),
        "require_base_contact": bool(require_base_contact),
        "aperture_threaded": bool(aperture_threaded),
        "aperture_radial_error": aperture_radial_error,
        "aperture_plane_error": aperture_plane_error,
        "max_aperture_radius": float(max_aperture_radius),
        "max_aperture_plane_dist": float(max_aperture_plane_dist),
        "end_distance": end_distance,
        "max_end_distance": float(max_end_distance),
        "progress": progress,
        "min_progress": float(min_progress),
        "wire_tilt_deg": float(wire_tilt_deg),
        "max_wire_tilt_deg": float(max_wire_tilt_deg),
        "wire_xy_displacement": wire_xy_displacement,
        "max_wire_xy_displacement": float(max_wire_xy_displacement),
        "wire_z_lift": wire_z_lift,
        "max_wire_z_lift": float(max_wire_z_lift),
        "spoon_z_delta": spoon_z_delta,
        "max_spoon_drop": float(max_spoon_drop),
        "wire_fell": bool(wire_fell),
        "wire_displaced": bool(wire_displaced),
        "wire_lifted": bool(wire_lifted),
        "spoon_dropped": bool(spoon_dropped),
        "safety_ok": bool(safety_ok),
    }


def wiregame_fail_reason(status: dict) -> str:
    if status["wire_fell"]:
        return f"wire_fell:{status['wire_tilt_deg']:.1f}deg"
    if status["wire_displaced"]:
        return f"wire_base_moved:{status['wire_xy_displacement']:.3f}m"
    if status["wire_lifted"]:
        return f"wire_base_lifted:{status['wire_z_lift']:.3f}m"
    if status["spoon_dropped"]:
        return f"spoon_dropped:{status['spoon_z_delta']:.3f}m"
    if status["require_base_contact"] and not status["base_contact"]:
        return "spoon_not_touching_wire_base"
    if not status["aperture_threaded"]:
        if status["aperture_radial_error"] > status["max_aperture_radius"]:
            return f"spoon_not_around_wire:{status['aperture_radial_error']:.3f}m"
        return f"wire_not_in_spoon_plane:{status['aperture_plane_error']:.3f}m"
    if status["end_distance"] > status["max_end_distance"]:
        return f"not_at_wire_base:{status['end_distance']:.3f}m"
    if status["progress"] < status["min_progress"]:
        return f"insufficient_progress:{status['progress']:.3f}m"
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


def safe_video_stem(case_id: str, status: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", case_id)
    return f"{stem}_{status}.mp4"


def parse_bool(value: str) -> bool:
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


def evaluate_episode(
    *,
    npz_path: Path,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    renderer: PolicyRenderer,
    policy: ACTPolicy,
    preprocessor,
    postprocessor,
    qpos_indices: list[int],
    state_dim: int,
    body_name: str,
    target_body_name: str,
    hand_body_name: str,
    free_joint_name: str,
    repeat: int,
    xy_jitter: tuple[float, float],
    yaw_jitter_rad: float,
    policy_hz: float,
    action_mode: str,
    arm_action_mode: str,
    gripper_action_mode: str,
    task_mode: str,
    tshape_lifted_bar_geom: str,
    tshape_lifted_stem_geom: str,
    tshape_target_bar_geom: str,
    tshape_target_stem_geom: str,
    tshape_min_bar_overlap: float,
    tshape_max_bar_axis_error_deg: float,
    tshape_min_face_alignment: float,
    tshape_min_stem_up_alignment: float,
    tshape_min_stem_height: float,
    tshape_max_face_gap: float,
    tshape_min_target_top_alignment: float,
    tshape_require_bar_contact: bool,
    tshape_forbid_stem_contact: bool,
    cup_pairs: list[tuple[str, str]],
    cup_body_names: list[str],
    box_body_names: list[str],
    box_upright_angle_deg: float,
    cup_upright_angle_deg: float,
    cups_require_contact: bool,
    cups_use_advanced_metric: bool,
    cups_max_axis_error_deg: float,
    cups_min_radial_margin: float,
    cups_forbid_upper_bad_contacts: bool,
    cups_forbid_upper_wrong_contacts: bool,
    cups_bad_contact_body_names: list[str],
    cups_max_pair_linear_speed: float,
    cups_max_pair_angular_speed: float,
    wire_spoon_body_name: str,
    wire_base_body_name: str,
    wire_ring_center_local: tuple[float, float, float],
    wire_ring_normal_local: tuple[float, float, float],
    wire_base_target_local: tuple[float, float, float],
    wire_max_end_distance: float,
    wire_max_aperture_radius: float,
    wire_max_aperture_plane_dist: float,
    wire_min_progress: float,
    wire_max_tilt_deg: float,
    wire_max_xy_displacement: float,
    wire_max_z_lift: float,
    wire_max_spoon_drop: float,
    wire_require_base_contact: bool,
    close_threshold: float,
    lift_threshold: float,
    transport_y_threshold: float,
    placement_xy_threshold: float,
    placement_target_local_offset: tuple[float, float, float],
    placement_z_offset: float,
    placement_z_tolerance: float,
    placement_yaw_threshold_deg: float,
    placement_gripper_open_threshold: float,
    placement_stable_steps: int,
    max_steps: int | None,
    timeout_seconds: float,
    stop_on_success: bool,
    video_recorder: EpisodeVideoRecorder | None = None,
    video_dir: Path | None = None,
    write_success_video: bool = False,
    write_failed_video: bool = False,
) -> EpisodeResult:
    episode = reset_from_npz(
        model=model,
        data=data,
        npz_path=npz_path,
        free_joint_name=free_joint_name,
        xy_jitter=xy_jitter,
        yaw_jitter_rad=yaw_jitter_rad,
    )
    mujoco.mj_forward(model, data)
    if video_recorder is not None:
        video_recorder.reset()
        video_recorder.capture(data)

    for component in (policy, preprocessor, postprocessor):
        reset_component(component)

    episode_len = int(episode["ctrl_sim"].shape[0])
    base_steps = episode_len if max_steps is None else max_steps
    timeout_steps = None
    if timeout_seconds > 0:
        timeout_steps = max(1, int(np.ceil(timeout_seconds * policy_hz)))
    steps = min(base_steps, timeout_steps) if timeout_steps is not None else base_steps
    n_substeps = max(1, int(round((1.0 / policy_hz) / model.opt.timestep)))

    if task_mode == "wiregame":
        initial_pos = body_pos(model, data, wire_spoon_body_name)
        initial_wire_pos = body_pos(model, data, wire_base_body_name)
        initial_end_distance = wiregame_reference_distance(
            model,
            data,
            spoon_body_name=wire_spoon_body_name,
            wire_body_name=wire_base_body_name,
            spoon_ring_center_local=wire_ring_center_local,
            wire_base_target_local=wire_base_target_local,
        )
        target_pos = _body_local_to_world(
            model,
            data,
            wire_base_body_name,
            np.asarray(wire_base_target_local, dtype=np.float64),
        )
        target_xy_error = float(initial_end_distance)
        target_z_error = 0.0
        target_yaw_error_deg = 0.0
    else:
        initial_wire_pos = None
        initial_end_distance = 0.0
        initial_pos = body_pos(model, data, body_name)
        target_pos, target_xy_error, target_z_error, target_yaw_error_deg = placement_metrics(
            model,
            data,
            body_name=body_name,
            target_body_name=target_body_name,
            target_local_offset=np.asarray(placement_target_local_offset, dtype=np.float64),
            placement_z_offset=placement_z_offset,
        )
    pos_history = [initial_pos.copy()]
    hand_pos = body_pos(model, data, hand_body_name)
    min_hand_object_dist = float(np.linalg.norm(hand_pos - initial_pos))

    first_close_step = None
    first_lift_step = None
    first_transport_step = None
    first_contact_step = None
    success_step = None
    success_target_xy_error = None
    success_target_z_error = None
    success_target_yaw_error_deg = None
    success_gripper = None
    success_contact_steps = None
    contact_streak = 0
    final_t1_t2_contact = False
    tshape_status = None
    wire_status = None
    wire_safety_status = None
    policy_calls = 0
    loop_times = []
    camera_names = policy_image_camera_names(policy)

    for step in range(steps):
        t0 = time.time()

        if policy_needs_observation(policy, action_mode):
            phase = 0.0 if episode_len <= 1 else min(step / (episode_len - 1), 1.0)
            state = build_state(
                qpos_sim=np.array(data.qpos, dtype=np.float32),
                qvel_sim=np.array(data.qvel, dtype=np.float32),
                ctrl_sim_prev=np.array(data.ctrl, dtype=np.float32),
                qpos_indices=qpos_indices,
                state_dim=state_dim,
                phase=phase,
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
            pred_action, queried_policy = select_action(
                policy=policy,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                observation=observation,
                action_mode=action_mode,
            )
        if queried_policy:
            policy_calls += 1

        previous_ctrl = np.array(data.ctrl[:8], dtype=np.float32)
        target_ctrl = previous_ctrl.copy()
        if arm_action_mode == "absolute":
            target_ctrl[:7] = pred_action[:7]
        elif arm_action_mode == "ctrl_delta":
            target_ctrl[:7] = previous_ctrl[:7] + pred_action[:7]
        elif arm_action_mode == "qpos_error":
            target_ctrl[:7] = np.array(data.qpos[qpos_indices], dtype=np.float32) + pred_action[:7]
        else:
            raise ValueError(f"Unknown arm action mode: {arm_action_mode}")

        if gripper_action_mode == "absolute":
            target_ctrl[7] = pred_action[7]
        elif gripper_action_mode == "delta":
            target_ctrl[7] = previous_ctrl[7] + pred_action[7]
        else:
            raise ValueError(f"Unknown gripper action mode: {gripper_action_mode}")

        target_ctrl = clip_to_actuator_ranges(model, target_ctrl)
        data.ctrl[:8] = target_ctrl

        if first_close_step is None and target_ctrl[7] < close_threshold:
            first_close_step = step

        for _ in range(n_substeps):
            mujoco.mj_step(model, data)
        mujoco.mj_forward(model, data)
        if video_recorder is not None:
            video_recorder.maybe_capture_step(data, step + 1)

        obj_pos = body_pos(model, data, wire_spoon_body_name if task_mode == "wiregame" else body_name)
        pos_history.append(obj_pos.copy())
        hand_pos = body_pos(model, data, hand_body_name)
        min_hand_object_dist = min(min_hand_object_dist, float(np.linalg.norm(hand_pos - obj_pos)))

        lift = float(obj_pos[2] - initial_pos[2])
        if first_lift_step is None and lift >= lift_threshold:
            first_lift_step = step
        if first_transport_step is None and obj_pos[1] >= transport_y_threshold:
            first_transport_step = step
        if task_mode == "wiregame":
            target_pos = _body_local_to_world(
                model,
                data,
                wire_base_body_name,
                np.asarray(wire_base_target_local, dtype=np.float64),
            )
            target_xy_error = float(np.linalg.norm(obj_pos - target_pos))
            target_z_error = 0.0
            target_yaw_error_deg = 0.0
        else:
            target_pos, target_xy_error, target_z_error, target_yaw_error_deg = placement_metrics(
                model,
                data,
                body_name=body_name,
                target_body_name=target_body_name,
                target_local_offset=np.asarray(placement_target_local_offset, dtype=np.float64),
                placement_z_offset=placement_z_offset,
            )
        final_gripper = float(data.ctrl[7])
        if task_mode == "cups":
            cups_status = cups_success_status(
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
                use_advanced_metric=cups_use_advanced_metric,
                max_axis_error_deg=cups_max_axis_error_deg,
                min_radial_margin=cups_min_radial_margin,
                forbid_upper_bad_contacts=cups_forbid_upper_bad_contacts,
                forbid_upper_wrong_contacts=cups_forbid_upper_wrong_contacts,
                bad_contact_body_names=cups_bad_contact_body_names,
                max_pair_linear_speed=cups_max_pair_linear_speed,
                max_pair_angular_speed=cups_max_pair_angular_speed,
            )
            final_t1_t2_contact = bool(cups_status["all_pairs_contact"])
            target_xy_error = float(cups_status["max_pair_radial_error"])
            target_z_error = float(cups_status["max_pair_abs_vertical_error"])
            target_yaw_error_deg = float(cups_status["max_pair_axis_error_deg"])
            pose_is_placed = bool(cups_status["success_now"])
        elif task_mode == "tshape":
            tshape_status = tshape_success_status(
                model,
                data,
                lifted_bar_geom=tshape_lifted_bar_geom,
                lifted_stem_geom=tshape_lifted_stem_geom,
                target_bar_geom=tshape_target_bar_geom,
                target_stem_geom=tshape_target_stem_geom,
                placement_gripper_open_threshold=placement_gripper_open_threshold,
                min_bar_overlap=tshape_min_bar_overlap,
                max_bar_axis_error_deg=tshape_max_bar_axis_error_deg,
                min_face_alignment=tshape_min_face_alignment,
                min_stem_up_alignment=tshape_min_stem_up_alignment,
                min_stem_height=tshape_min_stem_height,
                max_face_gap=tshape_max_face_gap,
                min_target_top_alignment=tshape_min_target_top_alignment,
                require_bar_contact=tshape_require_bar_contact,
                forbid_stem_contact=tshape_forbid_stem_contact,
            )
            final_t1_t2_contact = bool(tshape_status["bar_bar_contact"])
            target_xy_error = float(tshape_status["bar_center_error"])
            target_z_error = float(tshape_status["face_gap"])
            target_yaw_error_deg = float(tshape_status["bar_axis_error_deg"])
            pose_is_placed = bool(tshape_status["success_now"])
        elif task_mode == "wiregame":
            wire_status = wiregame_success_status(
                model,
                data,
                spoon_body_name=wire_spoon_body_name,
                wire_body_name=wire_base_body_name,
                initial_spoon_pos=initial_pos,
                initial_wire_pos=initial_wire_pos,
                initial_end_distance=initial_end_distance,
                spoon_ring_center_local=wire_ring_center_local,
                spoon_ring_normal_local=wire_ring_normal_local,
                wire_base_target_local=wire_base_target_local,
                max_end_distance=wire_max_end_distance,
                max_aperture_radius=wire_max_aperture_radius,
                max_aperture_plane_dist=wire_max_aperture_plane_dist,
                min_progress=wire_min_progress,
                max_wire_tilt_deg=wire_max_tilt_deg,
                max_wire_xy_displacement=wire_max_xy_displacement,
                max_wire_z_lift=wire_max_z_lift,
                max_spoon_drop=wire_max_spoon_drop,
                require_base_contact=wire_require_base_contact,
            )
            if wire_safety_status is None:
                wire_safety_status = dict(wire_status)
            else:
                wire_safety_status["wire_tilt_deg"] = max(
                    float(wire_safety_status["wire_tilt_deg"]),
                    float(wire_status["wire_tilt_deg"]),
                )
                wire_safety_status["wire_xy_displacement"] = max(
                    float(wire_safety_status["wire_xy_displacement"]),
                    float(wire_status["wire_xy_displacement"]),
                )
                wire_safety_status["wire_z_lift"] = max(
                    float(wire_safety_status["wire_z_lift"]),
                    float(wire_status["wire_z_lift"]),
                )
                wire_safety_status["spoon_z_delta"] = min(
                    float(wire_safety_status["spoon_z_delta"]),
                    float(wire_status["spoon_z_delta"]),
                )
                wire_safety_status["wire_fell"] = bool(
                    wire_safety_status["wire_fell"] or wire_status["wire_fell"]
                )
                wire_safety_status["wire_displaced"] = bool(
                    wire_safety_status["wire_displaced"] or wire_status["wire_displaced"]
                )
                wire_safety_status["wire_lifted"] = bool(
                    wire_safety_status["wire_lifted"] or wire_status["wire_lifted"]
                )
                wire_safety_status["spoon_dropped"] = bool(
                    wire_safety_status["spoon_dropped"] or wire_status["spoon_dropped"]
                )
            final_t1_t2_contact = bool(wire_status["base_contact"])
            target_xy_error = float(wire_status["end_distance"])
            target_z_error = float(wire_status["aperture_plane_error"])
            target_yaw_error_deg = float(wire_status["wire_tilt_deg"])
            pose_is_placed = bool(wire_status["success_now"]) and not any(
                bool(wire_safety_status[key])
                for key in ("wire_fell", "wire_displaced", "wire_lifted", "spoon_dropped")
            )
        else:
            final_t1_t2_contact = bodies_in_contact(model, data, body_name, target_body_name)
            pose_is_placed = (
                target_xy_error <= placement_xy_threshold
                and abs(target_z_error) <= placement_z_tolerance
                and target_yaw_error_deg <= placement_yaw_threshold_deg
                and (placement_gripper_open_threshold <= 0 or final_gripper >= placement_gripper_open_threshold)
                and final_t1_t2_contact
            )
        if final_t1_t2_contact and first_contact_step is None:
            first_contact_step = step
        if pose_is_placed:
            contact_streak += 1
        else:
            contact_streak = 0

        loop_times.append(time.time() - t0)

        if task_mode in {"cups", "tshape", "wiregame"}:
            success_gate = pose_is_placed
        else:
            success_gate = contact_streak >= placement_stable_steps

        if (
            success_step is None
            and (task_mode in {"cups", "wiregame"} or first_close_step is not None)
            and (task_mode in {"cups", "wiregame"} or first_lift_step is not None)
            and success_gate
        ):
            success_step = step
            success_target_xy_error = target_xy_error
            success_target_z_error = target_z_error
            success_target_yaw_error_deg = target_yaw_error_deg
            success_gripper = final_gripper
            success_contact_steps = contact_streak
            if stop_on_success:
                break

    positions = np.asarray(pos_history)
    final_pos = positions[-1]
    displacement = float(np.linalg.norm(final_pos - initial_pos))
    final_lift = float(final_pos[2] - initial_pos[2])
    max_lift = float(np.max(positions[:, 2]) - initial_pos[2])
    max_z = float(np.max(positions[:, 2]))
    max_y = float(np.max(positions[:, 1]))
    final_gripper = float(data.ctrl[7])
    if task_mode == "wiregame":
        target_pos = _body_local_to_world(
            model,
            data,
            wire_base_body_name,
            np.asarray(wire_base_target_local, dtype=np.float64),
        )
        target_xy_error = float(np.linalg.norm(final_pos - target_pos))
        target_z_error = 0.0
        target_yaw_error_deg = 0.0
        final_t1_t2_contact = bodies_in_contact(model, data, wire_spoon_body_name, wire_base_body_name)
    else:
        target_pos, target_xy_error, target_z_error, target_yaw_error_deg = placement_metrics(
            model,
            data,
            body_name=body_name,
            target_body_name=target_body_name,
            target_local_offset=np.asarray(placement_target_local_offset, dtype=np.float64),
            placement_z_offset=placement_z_offset,
        )
        final_t1_t2_contact = bodies_in_contact(model, data, body_name, target_body_name)
    cups_status = None
    if task_mode == "tshape":
        tshape_status = tshape_success_status(
            model,
            data,
            lifted_bar_geom=tshape_lifted_bar_geom,
            lifted_stem_geom=tshape_lifted_stem_geom,
            target_bar_geom=tshape_target_bar_geom,
            target_stem_geom=tshape_target_stem_geom,
            placement_gripper_open_threshold=placement_gripper_open_threshold,
            min_bar_overlap=tshape_min_bar_overlap,
            max_bar_axis_error_deg=tshape_max_bar_axis_error_deg,
            min_face_alignment=tshape_min_face_alignment,
            min_stem_up_alignment=tshape_min_stem_up_alignment,
            min_stem_height=tshape_min_stem_height,
            max_face_gap=tshape_max_face_gap,
            min_target_top_alignment=tshape_min_target_top_alignment,
            require_bar_contact=tshape_require_bar_contact,
            forbid_stem_contact=tshape_forbid_stem_contact,
        )
        target_xy_error = float(tshape_status["bar_center_error"])
        target_z_error = float(tshape_status["face_gap"])
        target_yaw_error_deg = float(tshape_status["bar_axis_error_deg"])
        final_t1_t2_contact = bool(tshape_status["bar_bar_contact"])
    if task_mode == "cups":
        cups_status = cups_success_status(
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
            use_advanced_metric=cups_use_advanced_metric,
            max_axis_error_deg=cups_max_axis_error_deg,
            min_radial_margin=cups_min_radial_margin,
            forbid_upper_bad_contacts=cups_forbid_upper_bad_contacts,
            forbid_upper_wrong_contacts=cups_forbid_upper_wrong_contacts,
            bad_contact_body_names=cups_bad_contact_body_names,
            max_pair_linear_speed=cups_max_pair_linear_speed,
            max_pair_angular_speed=cups_max_pair_angular_speed,
        )
        target_xy_error = float(cups_status["max_pair_radial_error"])
        target_z_error = float(cups_status["max_pair_abs_vertical_error"])
        target_yaw_error_deg = float(cups_status["max_pair_axis_error_deg"])
        final_t1_t2_contact = bool(cups_status["all_pairs_contact"])
    if task_mode == "wiregame":
        wire_status = wiregame_success_status(
            model,
            data,
            spoon_body_name=wire_spoon_body_name,
            wire_body_name=wire_base_body_name,
            initial_spoon_pos=initial_pos,
            initial_wire_pos=initial_wire_pos,
            initial_end_distance=initial_end_distance,
            spoon_ring_center_local=wire_ring_center_local,
            spoon_ring_normal_local=wire_ring_normal_local,
            wire_base_target_local=wire_base_target_local,
            max_end_distance=wire_max_end_distance,
            max_aperture_radius=wire_max_aperture_radius,
            max_aperture_plane_dist=wire_max_aperture_plane_dist,
            min_progress=wire_min_progress,
            max_wire_tilt_deg=wire_max_tilt_deg,
            max_wire_xy_displacement=wire_max_xy_displacement,
            max_wire_z_lift=wire_max_z_lift,
            max_spoon_drop=wire_max_spoon_drop,
            require_base_contact=wire_require_base_contact,
        )
        if wire_safety_status is not None:
            wire_status_for_reason = dict(wire_status)
            for key in ("wire_fell", "wire_displaced", "wire_lifted", "spoon_dropped"):
                wire_status_for_reason[key] = bool(wire_safety_status[key])
            for key in ("wire_tilt_deg", "wire_xy_displacement", "wire_z_lift"):
                wire_status_for_reason[key] = float(wire_safety_status[key])
            wire_status_for_reason["spoon_z_delta"] = float(wire_safety_status["spoon_z_delta"])
        else:
            wire_status_for_reason = wire_status
        target_xy_error = float(wire_status["end_distance"])
        target_z_error = float(wire_status["aperture_plane_error"])
        target_yaw_error_deg = float(wire_status["wire_tilt_deg"])
        final_t1_t2_contact = bool(wire_status["base_contact"])

    timed_out = (
        timeout_steps is not None
        and success_step is None
        and (len(pos_history) - 1) >= timeout_steps
        and timeout_steps <= base_steps
    )

    if success_step is not None:
        reason = ""
        success = True
    elif timed_out:
        reason = (
            f"duration_exceeded:>{timeout_seconds:.2f}s"
            if task_mode in {"cups", "tshape", "wiregame"}
            else "out_of_time"
        )
        success = False
    elif task_mode == "cups":
        reason = cups_fail_reason(
            cups_status,
            placement_stable_steps=placement_stable_steps,
            contact_streak=contact_streak,
        )
        success = False
    elif task_mode == "tshape":
        reason = tshape_fail_reason(
            tshape_status,
            first_close_step=first_close_step,
            first_lift_step=first_lift_step,
            placement_stable_steps=placement_stable_steps,
            contact_streak=contact_streak,
            min_target_top_alignment=tshape_min_target_top_alignment,
            min_bar_overlap=tshape_min_bar_overlap,
            max_bar_axis_error_deg=tshape_max_bar_axis_error_deg,
            min_face_alignment=tshape_min_face_alignment,
            min_stem_up_alignment=tshape_min_stem_up_alignment,
            min_stem_height=tshape_min_stem_height,
            max_face_gap=tshape_max_face_gap,
            placement_gripper_open_threshold=placement_gripper_open_threshold,
            require_bar_contact=tshape_require_bar_contact,
            forbid_stem_contact=tshape_forbid_stem_contact,
        )
        success = False
    elif task_mode == "wiregame":
        reason = wiregame_fail_reason(wire_status_for_reason)
        success = False
    else:
        reason = fail_reason(
            first_close_step=first_close_step,
            first_lift_step=first_lift_step,
            target_xy_error=target_xy_error,
            max_target_xy_error=placement_xy_threshold,
            target_z_error=target_z_error,
            max_abs_target_z_error=placement_z_tolerance,
            target_yaw_error_deg=target_yaw_error_deg,
            max_target_yaw_error_deg=placement_yaw_threshold_deg,
            final_gripper=final_gripper,
            min_open_gripper=placement_gripper_open_threshold,
            t1_t2_contact=final_t1_t2_contact,
            contact_streak=contact_streak,
            min_contact_streak=placement_stable_steps,
        )
        success = False
    case_id = f"{npz_path.stem}::repeat_{repeat:03d}"
    video_path = ""
    if video_recorder is not None and video_dir is not None:
        should_write = (success and write_success_video) or (not success and write_failed_video)
        if should_write:
            status = "PASS" if success else f"FAIL_{reason}"
            path = video_dir / safe_video_stem(case_id, status)
            video_recorder.write(path)
            video_path = str(path)

    return EpisodeResult(
        case_id=case_id,
        episode=npz_path.name,
        repeat=repeat,
        xy_jitter_x=float(xy_jitter[0]),
        xy_jitter_y=float(xy_jitter[1]),
        yaw_jitter_deg=float(np.degrees(yaw_jitter_rad)),
        steps=len(pos_history) - 1,
        success=success,
        success_step=success_step,
        first_close_step=first_close_step,
        first_lift_step=first_lift_step,
        first_transport_step=first_transport_step,
        first_contact_step=first_contact_step,
        success_target_xy_error=success_target_xy_error,
        success_target_z_error=success_target_z_error,
        success_target_yaw_error_deg=success_target_yaw_error_deg,
        success_gripper=success_gripper,
        success_contact_steps=success_contact_steps,
        target_x=float(target_pos[0]),
        target_y=float(target_pos[1]),
        target_z=float(target_pos[2]),
        initial_x=float(initial_pos[0]),
        initial_y=float(initial_pos[1]),
        initial_z=float(initial_pos[2]),
        final_x=float(final_pos[0]),
        final_y=float(final_pos[1]),
        final_z=float(final_pos[2]),
        max_z=max_z,
        max_y=max_y,
        displacement=displacement,
        final_lift=final_lift,
        max_lift=max_lift,
        final_target_xy_error=target_xy_error,
        final_target_z_error=target_z_error,
        final_target_yaw_error_deg=target_yaw_error_deg,
        final_gripper=final_gripper,
        final_t1_t2_contact=final_t1_t2_contact,
        final_contact_streak=contact_streak,
        strict_bar_overlap=None if tshape_status is None else float(tshape_status["bar_overlap"]),
        strict_target_bar_overlap=None if tshape_status is None else float(tshape_status["target_bar_overlap"]),
        strict_lifted_bar_overlap=None if tshape_status is None else float(tshape_status["lifted_bar_overlap"]),
        strict_bar_axis_error_deg=None if tshape_status is None else float(tshape_status["bar_axis_error_deg"]),
        strict_face_alignment=None if tshape_status is None else float(tshape_status["face_alignment"]),
        strict_stem_up_alignment=None if tshape_status is None else float(tshape_status["stem_up_alignment"]),
        strict_stem_height=None if tshape_status is None else float(tshape_status["stem_height"]),
        strict_face_gap=None if tshape_status is None else float(tshape_status["face_gap"]),
        strict_bar_bar_contact=None if tshape_status is None else bool(tshape_status["bar_bar_contact"]),
        strict_stem_target_contact=None if tshape_status is None else bool(tshape_status["stem_target_contact"]),
        min_hand_object_dist=min_hand_object_dist,
        mean_loop_ms=float(np.mean(loop_times) * 1000.0) if loop_times else 0.0,
        policy_calls=policy_calls,
        video_path=video_path,
        fail_reason=reason,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Headless ACT rollout evaluator for the MuJoCo T-shape task.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML_PATH)
    parser.add_argument("--episodes_dir", type=Path, default=DEFAULT_EPISODES_DIR)
    parser.add_argument("--episode_glob", default="*.npz")
    parser.add_argument("--num_episodes", type=int, default=10, help="Number of episodes to evaluate. Use <=0 for all.")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help="0 means each demo's trimmed length. Values > demo length extend the rollout.",
    )
    parser.add_argument(
        "--timeout_seconds",
        type=float,
        default=0.0,
        help="Stop unfinished rollouts after this many policy seconds. T-shape/cups/wiregame report duration_exceeded. Use <=0 to disable.",
    )
    parser.add_argument("--policy_hz", type=float, default=10.0)
    parser.add_argument("--action_mode", choices=["queue", "replan"], default="queue")
    parser.add_argument("--arm_action_mode", choices=["absolute", "ctrl_delta", "qpos_error"], default="absolute")
    parser.add_argument("--gripper_action_mode", choices=["absolute", "delta"], default="absolute")
    parser.add_argument("--task_mode", choices=["single", "tshape", "cups", "wiregame"], default="single")
    parser.add_argument("--body_name", default="T1")
    parser.add_argument("--target_body_name", default="T2")
    parser.add_argument("--hand_body_name", default="hand")
    parser.add_argument("--free_joint_name", default="T1_free")
    parser.add_argument("--tshape_lifted_bar_geom", default="T1_bar")
    parser.add_argument("--tshape_lifted_stem_geom", default="T1_stem")
    parser.add_argument("--tshape_target_bar_geom", default="T2_bar")
    parser.add_argument("--tshape_target_stem_geom", default="T2_stem")
    # Loosened 2026-08-12 (0.60 -> 0.40, 15 -> 25 deg) to match the live detector in
    # app.py's LiveTaskEvaluator. These two default sets are independent, so they must be
    # changed together: otherwise the same episode is a success live and a failure when
    # re-scored offline, and no analysis can reconcile the two. See the note at
    # app.py:tshape_min_bar_overlap for the measurements behind the new values.
    parser.add_argument(
        "--tshape_min_bar_overlap",
        type=float,
        default=0.40,
        help="Minimum min(lifted-face overlap, target-face overlap) for strict T-shape bar-on-bar success.",
    )
    parser.add_argument("--tshape_max_bar_axis_error_deg", type=float, default=25.0)
    parser.add_argument(
        "--tshape_min_face_alignment",
        type=float,
        default=0.85,
        help="Require T1_bar local +Z face to point down into T2_bar; 1.0 is perfectly face-on.",
    )
    parser.add_argument("--tshape_min_stem_up_alignment", type=float, default=0.75)
    parser.add_argument("--tshape_min_stem_height", type=float, default=0.03)
    parser.add_argument("--tshape_max_face_gap", type=float, default=0.025)
    parser.add_argument("--tshape_min_target_top_alignment", type=float, default=0.75)
    parser.add_argument("--tshape_require_bar_contact", type=parse_bool, default=True)
    parser.add_argument("--tshape_forbid_stem_contact", type=parse_bool, default=True)
    parser.add_argument(
        "--cup_pairs",
        type=parse_cup_pairs,
        default=parse_cup_pairs("cup1:cup2,cup3:cup4"),
        help="Comma-separated cup placements for --task_mode cups, e.g. 'cup1:cup2,cup3:cup4'.",
    )
    parser.add_argument(
        "--box_body_names",
        type=parse_body_list,
        default=parse_body_list("cracker_box sugar_box"),
        help="Box bodies that must stay upright for --task_mode cups.",
    )
    parser.add_argument(
        "--cup_body_names",
        type=parse_body_list,
        default=parse_body_list("cup1 cup2 cup3 cup4"),
        help="Cup bodies that must stay upright for --task_mode cups.",
    )
    parser.add_argument("--box_upright_angle_deg", type=float, default=20.0)
    parser.add_argument("--cup_upright_angle_deg", type=float, default=35.0)
    parser.add_argument("--cups_require_contact", type=parse_bool, default=True)
    parser.add_argument("--cups_use_advanced_metric", type=parse_bool, default=True)
    parser.add_argument("--cups_max_axis_error_deg", type=float, default=20.0)
    parser.add_argument(
        "--cups_min_radial_margin",
        type=float,
        default=-0.005,
        help="Require target rim radius - (radial error + upper cup bottom radius) to be at least this value.",
    )
    parser.add_argument("--cups_forbid_upper_bad_contacts", type=parse_bool, default=False)
    parser.add_argument("--cups_forbid_upper_wrong_contacts", type=parse_bool, default=False)
    parser.add_argument(
        "--cups_bad_contact_body_names",
        type=parse_body_list,
        default=parse_body_list("table cracker_box sugar_box"),
    )
    parser.add_argument("--cups_max_pair_linear_speed", type=float, default=0.0)
    parser.add_argument("--cups_max_pair_angular_speed", type=float, default=0.0)
    parser.add_argument("--wire_spoon_body_name", default="spoon1")
    parser.add_argument("--wire_base_body_name", default="object")
    parser.add_argument("--wire_ring_center_local", type=parse_vec3, default=(0.0, 0.0, 0.01))
    parser.add_argument("--wire_ring_normal_local", type=parse_vec3, default=(0.0, 0.0, 1.0))
    parser.add_argument("--wire_base_target_local", type=parse_vec3, default=(0.0, 0.0, 0.02))
    parser.add_argument("--wire_max_end_distance", type=float, default=0.06)
    parser.add_argument("--wire_max_aperture_radius", type=float, default=0.04)
    parser.add_argument("--wire_max_aperture_plane_dist", type=float, default=0.04)
    parser.add_argument("--wire_min_progress", type=float, default=0.08)
    parser.add_argument("--wire_max_tilt_deg", type=float, default=35.0)
    parser.add_argument("--wire_max_xy_displacement", type=float, default=0.05)
    parser.add_argument("--wire_max_z_lift", type=float, default=0.05)
    parser.add_argument("--wire_max_spoon_drop", type=float, default=0.18)
    parser.add_argument("--wire_require_base_contact", type=parse_bool, default=True)
    parser.add_argument("--repeats_per_episode", type=int, default=1)
    parser.add_argument("--xy_jitter", type=float, default=0.0, help="Uniform random +/- jitter in meters for T1 x/y.")
    parser.add_argument(
        "--yaw_jitter_deg",
        type=float,
        default=0.0,
        help="Uniform random +/- yaw rotation in degrees for the T-shape free joint.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--close_threshold", type=float, default=0.02)
    parser.add_argument("--lift_threshold", type=float, default=0.08)
    parser.add_argument("--transport_y_threshold", type=float, default=0.05)
    parser.add_argument("--final_lift_threshold", type=float, default=0.08, help="Deprecated; kept for old commands.")
    parser.add_argument("--placement_xy_threshold", type=float, default=0.04)
    parser.add_argument(
        "--placement_target_local_offset",
        type=parse_vec3,
        default=(0.0, 0.0, 0.0),
        help="Target point offset in target body local coordinates, e.g. '0 0 0.12' for the T2 bar.",
    )
    parser.add_argument("--placement_z_offset", type=float, default=0.165)
    parser.add_argument("--placement_z_tolerance", type=float, default=0.025)
    parser.add_argument("--placement_yaw_threshold_deg", type=float, default=25.0)
    parser.add_argument("--placement_gripper_open_threshold", type=float, default=0.025)
    parser.add_argument(
        "--placement_stable_steps",
        type=int,
        default=10,
        help="Require this many consecutive policy steps with T1 sitting on T2 before PASS.",
    )
    parser.add_argument(
        "--stop_on_success",
        type=parse_bool,
        default=False,
        help="Stop a rollout as soon as all success criteria are satisfied.",
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--record_videos", choices=["none", "failed", "success", "all"], default="none")
    parser.add_argument(
        "--success_video_limit",
        type=int,
        default=0,
        help="When >0, additionally save up to this many successful rollout videos.",
    )
    parser.add_argument(
        "--stop_after_success_videos",
        type=parse_bool,
        default=False,
        help="Stop evaluation once success_video_limit successful videos have been saved.",
    )
    parser.add_argument("--video_dir", type=Path, default=None)
    parser.add_argument("--video_camera", default="front")
    parser.add_argument("--video_width", type=int, default=640)
    parser.add_argument("--video_height", type=int, default=360)
    parser.add_argument("--video_fps", type=float, default=0.0, help="Use <=0 to derive from policy_hz/video_every_n_steps.")
    parser.add_argument("--video_every_n_steps", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    episodes = sorted(args.episodes_dir.glob(args.episode_glob))
    if not episodes:
        raise FileNotFoundError(f"No episodes found in {args.episodes_dir} matching {args.episode_glob!r}")
    episodes = episodes[args.start_index :]
    if args.num_episodes > 0:
        episodes = episodes[: args.num_episodes]

    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("outputs") / f"eval_act_tshape_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    video_dir = args.video_dir if args.video_dir is not None else output_dir / "videos"
    if args.video_every_n_steps < 1:
        raise ValueError("--video_every_n_steps must be >= 1")
    video_fps = args.video_fps if args.video_fps > 0 else args.policy_hz / args.video_every_n_steps

    print(f"[INFO] Checkpoint: {args.checkpoint}")
    print(f"[INFO] XML:        {args.xml}")
    if args.repeats_per_episode < 1:
        raise ValueError("--repeats_per_episode must be >= 1")
    if args.xy_jitter < 0:
        raise ValueError("--xy_jitter must be >= 0")
    if args.yaw_jitter_deg < 0:
        raise ValueError("--yaw_jitter_deg must be >= 0")
    if args.max_steps < 0:
        raise ValueError("--max_steps must be >= 0")
    if args.timeout_seconds < 0:
        raise ValueError("--timeout_seconds must be >= 0")
    if args.placement_xy_threshold < 0:
        raise ValueError("--placement_xy_threshold must be >= 0")
    if args.placement_z_tolerance < 0:
        raise ValueError("--placement_z_tolerance must be >= 0")
    if args.placement_yaw_threshold_deg < 0:
        raise ValueError("--placement_yaw_threshold_deg must be >= 0")
    if args.placement_stable_steps < 1:
        raise ValueError("--placement_stable_steps must be >= 1")
    if not 0 <= args.tshape_min_bar_overlap <= 1:
        raise ValueError("--tshape_min_bar_overlap must be in [0, 1]")
    if args.tshape_max_bar_axis_error_deg < 0:
        raise ValueError("--tshape_max_bar_axis_error_deg must be >= 0")
    if not 0 <= args.tshape_min_face_alignment <= 1:
        raise ValueError("--tshape_min_face_alignment must be in [0, 1]")
    if not -1 <= args.tshape_min_stem_up_alignment <= 1:
        raise ValueError("--tshape_min_stem_up_alignment must be in [-1, 1]")
    if args.tshape_min_stem_height < 0:
        raise ValueError("--tshape_min_stem_height must be >= 0")
    if args.tshape_max_face_gap < 0:
        raise ValueError("--tshape_max_face_gap must be >= 0")
    if not 0 <= args.tshape_min_target_top_alignment <= 1:
        raise ValueError("--tshape_min_target_top_alignment must be in [0, 1]")
    if args.box_upright_angle_deg < 0:
        raise ValueError("--box_upright_angle_deg must be >= 0")
    if args.cup_upright_angle_deg < 0:
        raise ValueError("--cup_upright_angle_deg must be >= 0")
    if args.cups_max_axis_error_deg < 0:
        raise ValueError("--cups_max_axis_error_deg must be >= 0")
    if args.cups_max_pair_linear_speed < 0:
        raise ValueError("--cups_max_pair_linear_speed must be >= 0; use 0 to disable")
    if args.cups_max_pair_angular_speed < 0:
        raise ValueError("--cups_max_pair_angular_speed must be >= 0; use 0 to disable")
    for name in (
        "wire_max_end_distance",
        "wire_max_aperture_radius",
        "wire_max_aperture_plane_dist",
        "wire_min_progress",
        "wire_max_tilt_deg",
        "wire_max_xy_displacement",
        "wire_max_z_lift",
        "wire_max_spoon_drop",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name} must be >= 0")
    if args.success_video_limit < 0:
        raise ValueError("--success_video_limit must be >= 0")
    if args.stop_after_success_videos and args.success_video_limit <= 0:
        raise ValueError("--stop_after_success_videos requires --success_video_limit > 0")
    if args.task_mode == "wiregame":
        if args.body_name == "T1":
            args.body_name = args.wire_spoon_body_name
        if args.target_body_name == "T2":
            args.target_body_name = args.wire_base_body_name
        if args.free_joint_name == "T1_free":
            args.free_joint_name = f"{args.wire_spoon_body_name}_free"
    rng = np.random.default_rng(args.seed)

    print(f"[INFO] Episodes:   {len(episodes)} from {args.episodes_dir}")
    print(
        f"[INFO] Cases:      {len(episodes) * args.repeats_per_episode} "
        f"({args.repeats_per_episode} repeat(s)/episode, "
        f"xy_jitter=+/-{args.xy_jitter:g} m, yaw_jitter=+/-{args.yaw_jitter_deg:g} deg)"
    )
    print(f"[INFO] Device:     {device}")
    print(f"[INFO] Output:     {output_dir}")
    if args.max_steps > 0:
        print(f"[INFO] Max steps:  {args.max_steps} ({args.max_steps / args.policy_hz:.1f}s at {args.policy_hz:g} Hz)")
    if args.timeout_seconds > 0:
        timeout_reason = "duration_exceeded" if args.task_mode in {"cups", "tshape", "wiregame"} else "out_of_time"
        print(f"[INFO] Timeout:    {args.timeout_seconds:g}s -> FAIL:{timeout_reason}")
    print(f"[INFO] Stop success: {args.stop_on_success}")
    record_candidates = args.record_videos != "none" or args.success_video_limit > 0
    if record_candidates:
        print(
            f"[INFO] Videos:     {args.record_videos} -> {video_dir} "
            f"({args.video_camera}, {args.video_width}x{args.video_height}, {video_fps:g} fps, "
            f"success_limit={args.success_video_limit})"
        )
        if args.stop_after_success_videos:
            print(f"[INFO] Stop eval:  after {args.success_video_limit} successful video(s)")
    if args.task_mode == "cups":
        pair_text = ", ".join(f"{cup}->{target}" for cup, target in args.cup_pairs)
        print(
            "[INFO] Success criterion: "
            f"cups placed ({pair_text}) with xy<={args.placement_xy_threshold:g} m, "
            f"z_offset={args.placement_z_offset:g}+/-{args.placement_z_tolerance:g} m, "
            f"contact={args.cups_require_contact}, final object-state check, "
            f"boxes upright angle<={args.box_upright_angle_deg:g} deg: {args.box_body_names}, "
            f"cups upright angle<={args.cup_upright_angle_deg:g} deg: {args.cup_body_names}"
            + (
                f", advanced nesting radial<={args.placement_xy_threshold:g} m, "
                f"axis<={args.cups_max_axis_error_deg:g} deg, "
                f"margin>={args.cups_min_radial_margin:g} m, "
                f"speed<={args.cups_max_pair_linear_speed:g} m/s"
                if args.cups_use_advanced_metric
                else ", legacy body-center metric"
            )
            + (
                f", released grip>={args.placement_gripper_open_threshold:g}"
                if args.placement_gripper_open_threshold > 0
                else ""
            )
        )
    elif args.task_mode == "tshape":
        print(
            "[INFO] Success criterion: "
            f"grasped grip<{args.close_threshold:g}, lifted>={args.lift_threshold:g} m, "
            f"{args.tshape_lifted_bar_geom} face-on {args.tshape_target_bar_geom}, "
            f"bar_overlap>={args.tshape_min_bar_overlap:g}, "
            f"bar_axis<={args.tshape_max_bar_axis_error_deg:g} deg, "
            f"face_align>={args.tshape_min_face_alignment:g}, "
            f"stem_up>={args.tshape_min_stem_up_alignment:g}, "
            f"stem_height>={args.tshape_min_stem_height:g} m, "
            f"face_gap<=+/-{args.tshape_max_face_gap:g} m, "
            f"bar_contact={args.tshape_require_bar_contact}, "
            f"forbid_stem_contact={args.tshape_forbid_stem_contact}, "
            "final strict pose check"
            + (
                f", released grip>={args.placement_gripper_open_threshold:g}"
                if args.placement_gripper_open_threshold > 0
                else ""
            )
        )
    elif args.task_mode == "wiregame":
        print(
            "[INFO] Success criterion: "
            f"{args.wire_spoon_body_name} reaches {args.wire_base_body_name} base target "
            f"{args.wire_base_target_local}, "
            f"ring/base distance<={args.wire_max_end_distance:g} m, "
            f"aperture radial<={args.wire_max_aperture_radius:g} m, "
            f"aperture plane<={args.wire_max_aperture_plane_dist:g} m, "
            f"progress>={args.wire_min_progress:g} m, "
            f"base_contact={args.wire_require_base_contact}, "
            f"wire tilt<={args.wire_max_tilt_deg:g} deg anytime, "
            f"wire xy move<={args.wire_max_xy_displacement:g} m, "
            f"wire z lift<={args.wire_max_z_lift:g} m, "
            f"spoon drop<={args.wire_max_spoon_drop:g} m, "
            "no duration limit label for wiregame"
        )
    else:
        print(
            "[INFO] Success criterion: "
            f"grasped grip<{args.close_threshold:g}, lifted>={args.lift_threshold:g} m, "
            f"T1 near {args.target_body_name}+{args.placement_target_local_offset}: "
            f"xy<={args.placement_xy_threshold:g} m, "
            f"z_offset={args.placement_z_offset:g}+/-{args.placement_z_tolerance:g} m, "
            f"yaw<={args.placement_yaw_threshold_deg:g} deg, "
            f"T1/T2 contact for {args.placement_stable_steps} step(s)"
            + (
                f", released grip>={args.placement_gripper_open_threshold:g}"
                if args.placement_gripper_open_threshold > 0
                else ""
            )
        )

    policy = ACTPolicy.from_pretrained(args.checkpoint)
    policy.eval()
    policy.to(device)
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=str(args.checkpoint),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )

    state_dim = policy.config.input_features["observation.state"].shape[0]
    model = mujoco.MjModel.from_xml_path(str(args.xml))
    data = mujoco.MjData(model)
    qpos_indices = get_qpos_indices_for_joints(model, PANDA_JOINT_NAMES)
    renderer = PolicyRenderer(model)
    video_recorder = None
    if record_candidates:
        video_recorder = EpisodeVideoRecorder(
            model,
            camera=args.video_camera,
            width=args.video_width,
            height=args.video_height,
            fps=video_fps,
            every_n_steps=args.video_every_n_steps,
        )

    results: list[EpisodeResult] = []
    success_videos_written = 0
    stop_requested = False
    try:
        total_cases = len(episodes) * args.repeats_per_episode
        case_idx = 0
        for episode_path in episodes:
            for repeat in range(args.repeats_per_episode):
                case_idx += 1
                if args.xy_jitter > 0:
                    xy_jitter = tuple(rng.uniform(-args.xy_jitter, args.xy_jitter, size=2).astype(float))
                else:
                    xy_jitter = (0.0, 0.0)
                if args.yaw_jitter_deg > 0:
                    yaw_jitter_rad = float(np.radians(rng.uniform(-args.yaw_jitter_deg, args.yaw_jitter_deg)))
                else:
                    yaw_jitter_rad = 0.0
                result = evaluate_episode(
                    npz_path=episode_path,
                    model=model,
                    data=data,
                    renderer=renderer,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    qpos_indices=qpos_indices,
                    state_dim=state_dim,
                    body_name=args.body_name,
                    target_body_name=args.target_body_name,
                    hand_body_name=args.hand_body_name,
                    free_joint_name=args.free_joint_name,
                    repeat=repeat,
                    xy_jitter=xy_jitter,
                    yaw_jitter_rad=yaw_jitter_rad,
                    policy_hz=args.policy_hz,
                    action_mode=args.action_mode,
                    arm_action_mode=args.arm_action_mode,
                    gripper_action_mode=args.gripper_action_mode,
                    task_mode=args.task_mode,
                    tshape_lifted_bar_geom=args.tshape_lifted_bar_geom,
                    tshape_lifted_stem_geom=args.tshape_lifted_stem_geom,
                    tshape_target_bar_geom=args.tshape_target_bar_geom,
                    tshape_target_stem_geom=args.tshape_target_stem_geom,
                    tshape_min_bar_overlap=args.tshape_min_bar_overlap,
                    tshape_max_bar_axis_error_deg=args.tshape_max_bar_axis_error_deg,
                    tshape_min_face_alignment=args.tshape_min_face_alignment,
                    tshape_min_stem_up_alignment=args.tshape_min_stem_up_alignment,
                    tshape_min_stem_height=args.tshape_min_stem_height,
                    tshape_max_face_gap=args.tshape_max_face_gap,
                    tshape_min_target_top_alignment=args.tshape_min_target_top_alignment,
                    tshape_require_bar_contact=args.tshape_require_bar_contact,
                    tshape_forbid_stem_contact=args.tshape_forbid_stem_contact,
                    cup_pairs=args.cup_pairs,
                    cup_body_names=args.cup_body_names,
                    box_body_names=args.box_body_names,
                    box_upright_angle_deg=args.box_upright_angle_deg,
                    cup_upright_angle_deg=args.cup_upright_angle_deg,
                    cups_require_contact=args.cups_require_contact,
                    cups_use_advanced_metric=args.cups_use_advanced_metric,
                    cups_max_axis_error_deg=args.cups_max_axis_error_deg,
                    cups_min_radial_margin=args.cups_min_radial_margin,
                    cups_forbid_upper_bad_contacts=args.cups_forbid_upper_bad_contacts,
                    cups_forbid_upper_wrong_contacts=args.cups_forbid_upper_wrong_contacts,
                    cups_bad_contact_body_names=args.cups_bad_contact_body_names,
                    cups_max_pair_linear_speed=args.cups_max_pair_linear_speed,
                    cups_max_pair_angular_speed=args.cups_max_pair_angular_speed,
                    wire_spoon_body_name=args.wire_spoon_body_name,
                    wire_base_body_name=args.wire_base_body_name,
                    wire_ring_center_local=args.wire_ring_center_local,
                    wire_ring_normal_local=args.wire_ring_normal_local,
                    wire_base_target_local=args.wire_base_target_local,
                    wire_max_end_distance=args.wire_max_end_distance,
                    wire_max_aperture_radius=args.wire_max_aperture_radius,
                    wire_max_aperture_plane_dist=args.wire_max_aperture_plane_dist,
                    wire_min_progress=args.wire_min_progress,
                    wire_max_tilt_deg=args.wire_max_tilt_deg,
                    wire_max_xy_displacement=args.wire_max_xy_displacement,
                    wire_max_z_lift=args.wire_max_z_lift,
                    wire_max_spoon_drop=args.wire_max_spoon_drop,
                    wire_require_base_contact=args.wire_require_base_contact,
                    close_threshold=args.close_threshold,
                    lift_threshold=args.lift_threshold,
                    transport_y_threshold=args.transport_y_threshold,
                    placement_xy_threshold=args.placement_xy_threshold,
                    placement_target_local_offset=args.placement_target_local_offset,
                    placement_z_offset=args.placement_z_offset,
                    placement_z_tolerance=args.placement_z_tolerance,
                    placement_yaw_threshold_deg=args.placement_yaw_threshold_deg,
                    placement_gripper_open_threshold=args.placement_gripper_open_threshold,
                    placement_stable_steps=args.placement_stable_steps,
                    max_steps=None if args.max_steps <= 0 else args.max_steps,
                    timeout_seconds=args.timeout_seconds,
                    stop_on_success=args.stop_on_success,
                    video_recorder=video_recorder,
                    video_dir=video_dir,
                    write_success_video=(
                        args.record_videos in {"success", "all"}
                        or (
                            args.success_video_limit > 0
                            and success_videos_written < args.success_video_limit
                        )
                    ),
                    write_failed_video=args.record_videos in {"failed", "all"},
                )
                results.append(result)
                if result.success and result.video_path:
                    success_videos_written += 1
                    if (
                        args.stop_after_success_videos
                        and args.success_video_limit > 0
                        and success_videos_written >= args.success_video_limit
                    ):
                        stop_requested = True
                status = "PASS" if result.success else f"FAIL:{result.fail_reason}"
                video = f" video={result.video_path}" if result.video_path else ""
                close = "None" if result.first_close_step is None else str(result.first_close_step)
                success_step = "None" if result.success_step is None else str(result.success_step)
                strict_text = ""
                if args.task_mode == "tshape" and result.strict_bar_overlap is not None:
                    strict_text = (
                        f" strict_overlap={result.strict_bar_overlap:.2f} "
                        f"face={result.strict_face_alignment:.2f} "
                        f"stem={result.strict_stem_up_alignment:.2f} "
                        f"gap={result.strict_face_gap:+.3f} "
                    )
                print(
                    f"[{case_idx:03d}/{total_cases:03d}] {status:29s} "
                    f"close={close:>4s} success={success_step:>4s} final_y={result.final_y:+.3f} "
                    f"place_xy={result.final_target_xy_error:.3f} "
                    f"place_zerr={result.final_target_z_error:+.3f} "
                    f"yawerr={result.final_target_yaw_error_deg:.1f} "
                    f"grip={result.final_gripper:.3f} "
                    f"contact={int(result.final_t1_t2_contact)}x{result.final_contact_streak} "
                    f"jitter=({result.xy_jitter_x:+.3f},{result.xy_jitter_y:+.3f},"
                    f"yaw={result.yaw_jitter_deg:+.1f}deg) "
                    f"{strict_text}"
                    f"calls={result.policy_calls:02d} loop={result.mean_loop_ms:.1f}ms "
                    f"{result.episode}{video}"
                )
                if stop_requested:
                    break
            if stop_requested:
                break
    finally:
        renderer.close()
        if video_recorder is not None:
            video_recorder.close()

    rows = [asdict(result) for result in results]
    csv_path = output_dir / "episodes.csv"
    json_path = output_dir / "summary.json"

    if rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    n_success = sum(result.success for result in results)
    summary = {
        "checkpoint": str(args.checkpoint),
        "xml": str(args.xml),
        "episodes_dir": str(args.episodes_dir),
        "task_mode": args.task_mode,
        "body_name": args.body_name,
        "target_body_name": args.target_body_name,
        "tshape_lifted_bar_geom": args.tshape_lifted_bar_geom,
        "tshape_lifted_stem_geom": args.tshape_lifted_stem_geom,
        "tshape_target_bar_geom": args.tshape_target_bar_geom,
        "tshape_target_stem_geom": args.tshape_target_stem_geom,
        "tshape_min_bar_overlap": args.tshape_min_bar_overlap,
        "tshape_max_bar_axis_error_deg": args.tshape_max_bar_axis_error_deg,
        "tshape_min_face_alignment": args.tshape_min_face_alignment,
        "tshape_min_stem_up_alignment": args.tshape_min_stem_up_alignment,
        "tshape_min_stem_height": args.tshape_min_stem_height,
        "tshape_max_face_gap": args.tshape_max_face_gap,
        "tshape_min_target_top_alignment": args.tshape_min_target_top_alignment,
        "tshape_require_bar_contact": args.tshape_require_bar_contact,
        "tshape_forbid_stem_contact": args.tshape_forbid_stem_contact,
        "cup_pairs": [list(pair) for pair in args.cup_pairs],
        "cup_body_names": args.cup_body_names,
        "box_body_names": args.box_body_names,
        "box_upright_angle_deg": args.box_upright_angle_deg,
        "cup_upright_angle_deg": args.cup_upright_angle_deg,
        "cups_require_contact": args.cups_require_contact,
        "cups_use_advanced_metric": args.cups_use_advanced_metric,
        "cups_max_axis_error_deg": args.cups_max_axis_error_deg,
        "cups_min_radial_margin": args.cups_min_radial_margin,
        "cups_forbid_upper_bad_contacts": args.cups_forbid_upper_bad_contacts,
        "cups_forbid_upper_wrong_contacts": args.cups_forbid_upper_wrong_contacts,
        "cups_bad_contact_body_names": args.cups_bad_contact_body_names,
        "cups_max_pair_linear_speed": args.cups_max_pair_linear_speed,
        "cups_max_pair_angular_speed": args.cups_max_pair_angular_speed,
        "wire_spoon_body_name": args.wire_spoon_body_name,
        "wire_base_body_name": args.wire_base_body_name,
        "wire_ring_center_local": list(args.wire_ring_center_local),
        "wire_ring_normal_local": list(args.wire_ring_normal_local),
        "wire_base_target_local": list(args.wire_base_target_local),
        "wire_max_end_distance": args.wire_max_end_distance,
        "wire_max_aperture_radius": args.wire_max_aperture_radius,
        "wire_max_aperture_plane_dist": args.wire_max_aperture_plane_dist,
        "wire_min_progress": args.wire_min_progress,
        "wire_max_tilt_deg": args.wire_max_tilt_deg,
        "wire_max_xy_displacement": args.wire_max_xy_displacement,
        "wire_max_z_lift": args.wire_max_z_lift,
        "wire_max_spoon_drop": args.wire_max_spoon_drop,
        "wire_require_base_contact": args.wire_require_base_contact,
        "placement_target_local_offset": list(args.placement_target_local_offset),
        "repeats_per_episode": args.repeats_per_episode,
        "xy_jitter": args.xy_jitter,
        "yaw_jitter_deg": args.yaw_jitter_deg,
        "max_steps": args.max_steps,
        "timeout_seconds": args.timeout_seconds,
        "stop_on_success": args.stop_on_success,
        "seed": args.seed,
        "record_videos": args.record_videos,
        "success_video_limit": args.success_video_limit,
        "stop_after_success_videos": args.stop_after_success_videos,
        "success_videos_written": success_videos_written,
        "video_dir": str(video_dir) if record_candidates else None,
        "num_episodes": len(results),
        "successes": n_success,
        "success_rate": n_success / len(results) if results else 0.0,
        "mean_first_close_step": float(
            np.mean([r.first_close_step for r in results if r.first_close_step is not None])
        )
        if any(r.first_close_step is not None for r in results)
        else None,
        "mean_success_step": float(np.mean([r.success_step for r in results if r.success_step is not None]))
        if any(r.success_step is not None for r in results)
        else None,
        "mean_success_target_xy_error": float(
            np.mean([r.success_target_xy_error for r in results if r.success_target_xy_error is not None])
        )
        if any(r.success_target_xy_error is not None for r in results)
        else None,
        "mean_success_target_z_error": float(
            np.mean([r.success_target_z_error for r in results if r.success_target_z_error is not None])
        )
        if any(r.success_target_z_error is not None for r in results)
        else None,
        "mean_success_target_yaw_error_deg": float(
            np.mean(
                [
                    r.success_target_yaw_error_deg
                    for r in results
                    if r.success_target_yaw_error_deg is not None
                ]
            )
        )
        if any(r.success_target_yaw_error_deg is not None for r in results)
        else None,
        "mean_success_contact_steps": float(
            np.mean([r.success_contact_steps for r in results if r.success_contact_steps is not None])
        )
        if any(r.success_contact_steps is not None for r in results)
        else None,
        "mean_strict_bar_overlap": float(
            np.mean([r.strict_bar_overlap for r in results if r.strict_bar_overlap is not None])
        )
        if any(r.strict_bar_overlap is not None for r in results)
        else None,
        "mean_strict_bar_axis_error_deg": float(
            np.mean([r.strict_bar_axis_error_deg for r in results if r.strict_bar_axis_error_deg is not None])
        )
        if any(r.strict_bar_axis_error_deg is not None for r in results)
        else None,
        "mean_strict_face_alignment": float(
            np.mean([r.strict_face_alignment for r in results if r.strict_face_alignment is not None])
        )
        if any(r.strict_face_alignment is not None for r in results)
        else None,
        "mean_strict_stem_up_alignment": float(
            np.mean([r.strict_stem_up_alignment for r in results if r.strict_stem_up_alignment is not None])
        )
        if any(r.strict_stem_up_alignment is not None for r in results)
        else None,
        "mean_final_y": float(np.mean([r.final_y for r in results])) if results else None,
        "mean_final_lift": float(np.mean([r.final_lift for r in results])) if results else None,
        "mean_max_lift": float(np.mean([r.max_lift for r in results])) if results else None,
        "mean_final_target_xy_error": float(np.mean([r.final_target_xy_error for r in results])) if results else None,
        "mean_final_target_z_error": float(np.mean([r.final_target_z_error for r in results])) if results else None,
        "mean_final_target_yaw_error_deg": float(np.mean([r.final_target_yaw_error_deg for r in results]))
        if results
        else None,
        "mean_loop_ms": float(np.mean([r.mean_loop_ms for r in results])) if results else None,
        "criteria": {
            "close_threshold": args.close_threshold,
            "lift_threshold": args.lift_threshold,
            "transport_y_threshold": args.transport_y_threshold,
            "placement_xy_threshold": args.placement_xy_threshold,
            "placement_target_local_offset": list(args.placement_target_local_offset),
            "placement_z_offset": args.placement_z_offset,
            "placement_z_tolerance": args.placement_z_tolerance,
            "placement_yaw_threshold_deg": args.placement_yaw_threshold_deg,
            "placement_gripper_open_threshold": args.placement_gripper_open_threshold,
            "placement_stable_steps": args.placement_stable_steps,
            "tshape_lifted_bar_geom": args.tshape_lifted_bar_geom,
            "tshape_lifted_stem_geom": args.tshape_lifted_stem_geom,
            "tshape_target_bar_geom": args.tshape_target_bar_geom,
            "tshape_target_stem_geom": args.tshape_target_stem_geom,
            "tshape_min_bar_overlap": args.tshape_min_bar_overlap,
            "tshape_max_bar_axis_error_deg": args.tshape_max_bar_axis_error_deg,
            "tshape_min_face_alignment": args.tshape_min_face_alignment,
            "tshape_min_stem_up_alignment": args.tshape_min_stem_up_alignment,
            "tshape_min_stem_height": args.tshape_min_stem_height,
            "tshape_max_face_gap": args.tshape_max_face_gap,
            "tshape_min_target_top_alignment": args.tshape_min_target_top_alignment,
            "tshape_require_bar_contact": args.tshape_require_bar_contact,
            "tshape_forbid_stem_contact": args.tshape_forbid_stem_contact,
            "cups_use_advanced_metric": args.cups_use_advanced_metric,
            "cups_max_axis_error_deg": args.cups_max_axis_error_deg,
            "cups_min_radial_margin": args.cups_min_radial_margin,
            "cups_forbid_upper_bad_contacts": args.cups_forbid_upper_bad_contacts,
            "cups_forbid_upper_wrong_contacts": args.cups_forbid_upper_wrong_contacts,
            "cups_bad_contact_body_names": args.cups_bad_contact_body_names,
            "cups_max_pair_linear_speed": args.cups_max_pair_linear_speed,
            "cups_max_pair_angular_speed": args.cups_max_pair_angular_speed,
            "wire_spoon_body_name": args.wire_spoon_body_name,
            "wire_base_body_name": args.wire_base_body_name,
            "wire_ring_center_local": args.wire_ring_center_local,
            "wire_ring_normal_local": args.wire_ring_normal_local,
            "wire_base_target_local": args.wire_base_target_local,
            "wire_max_end_distance": args.wire_max_end_distance,
            "wire_max_aperture_radius": args.wire_max_aperture_radius,
            "wire_max_aperture_plane_dist": args.wire_max_aperture_plane_dist,
            "wire_min_progress": args.wire_min_progress,
            "wire_max_tilt_deg": args.wire_max_tilt_deg,
            "wire_max_xy_displacement": args.wire_max_xy_displacement,
            "wire_max_z_lift": args.wire_max_z_lift,
            "wire_max_spoon_drop": args.wire_max_spoon_drop,
            "wire_require_base_contact": args.wire_require_base_contact,
        },
        "failure_counts": {},
    }
    for result in results:
        if result.success:
            continue
        summary["failure_counts"][result.fail_reason] = summary["failure_counts"].get(result.fail_reason, 0) + 1

    json_path.write_text(json.dumps(summary, indent=2) + "\n")

    print()
    print(
        f"[SUMMARY] success_rate={summary['success_rate']:.1%} "
        f"({summary['successes']}/{summary['num_episodes']}) "
        f"mean_close_step={summary['mean_first_close_step']} "
        f"mean_success_step={summary['mean_success_step']} "
        f"mean_success_xy={summary['mean_success_target_xy_error']} "
        f"mean_success_contact_steps={summary['mean_success_contact_steps']} "
        f"mean_place_xy={summary['mean_final_target_xy_error']:.3f} "
        f"mean_place_zerr={summary['mean_final_target_z_error']:.3f} "
        f"mean_yawerr={summary['mean_final_target_yaw_error_deg']:.1f}"
    )
    if summary["mean_strict_bar_overlap"] is not None:
        print(
            f"[SUMMARY] strict_overlap={summary['mean_strict_bar_overlap']:.3f} "
            f"strict_axis_err={summary['mean_strict_bar_axis_error_deg']:.1f} "
            f"strict_face={summary['mean_strict_face_alignment']:.3f} "
            f"strict_stem={summary['mean_strict_stem_up_alignment']:.3f}"
        )
    print(f"[SUMMARY] Wrote {csv_path}")
    print(f"[SUMMARY] Wrote {json_path}")


if __name__ == "__main__":
    main()
