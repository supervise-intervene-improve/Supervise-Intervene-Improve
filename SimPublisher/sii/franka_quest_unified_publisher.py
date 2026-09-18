import argparse
import asyncio
import json
import socket
import struct
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import glfw
import mujoco
import numpy as np
import zmq
from GPU.GpuPointCloudPipeline import (
    LegacyCompatPointCloudPipeline,
    Open3dCudaPointCloudPipeline,
    sampled_point_capacity,
)

from simpub.sim.mj_publisher import MujocoPublisher
from simpub.xr_device.meta_quest3 import MetaQuest3


if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


DISCOVERY_PORT = 7720
MULTICAST_GRP = "239.255.10.10"
REP_SUCCESS = b"SUCCESS"
UNIFIED_PUBLISHER_VERSION = "2026-03-17-gpu-pc-stable-random"
STANDARD_EXTRINSICS_PROFILE = (
    Path(__file__).resolve().parent / "camera_extrinsics_lab_standard.json"
)
REVERSE_U_JOINTS_RAD = np.array([0.0, -0.35, 0.0, -2.25, 0.0, 1.95, 0.78], dtype=np.float64)


def list_cameras(model: mujoco.MjModel) -> List[str]:
    cams: List[str] = []
    for i in range(model.ncam):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        if name:
            cams.append(name)
    return cams


def find_first_existing(model, names, objtype):
    for n in names:
        try:
            obj_id = mujoco.mj_name2id(model, objtype, n)
            if obj_id != -1:
                return n, obj_id
        except Exception:
            pass
    return None, -1


def quat_normalize(q):
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    return q / n if n > 1e-9 else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def quat_mul(a, b):
    # MuJoCo quaternion convention: wxyz
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def quat_to_rotvec(q):
    q = quat_normalize(q)
    w = np.clip(q[0], -1.0, 1.0)
    angle = 2.0 * np.arccos(w)
    s = np.sqrt(max(1e-12, 1.0 - w * w))
    axis = q[1:] / s
    if angle < 1e-6:
        return np.zeros(3, dtype=np.float64)
    return axis * angle


def unity_to_mj_pos(p_unity):
    # Inverse of mj->unity mapping (-y, z, x).
    x, y, z = p_unity
    return np.array([z, -x, y], dtype=np.float64)


def unity_to_mj_quat(q_unity_xyzw):
    # Inverse of mj2unity quat mapping: [mj_y, -mj_z, -mj_x, mj_w]
    x, y, z, w = q_unity_xyzw
    return quat_normalize(np.array([w, -z, x, -y], dtype=np.float64))


def mj_to_unity_pos(p_mj: np.ndarray) -> np.ndarray:
    x, y, z = p_mj
    return np.array([-y, z, x], dtype=np.float32)


def mj_to_unity_quat_xyzw(q_mj_wxyz: np.ndarray) -> np.ndarray:
    # Matches scene publisher conversion used by SimPub for rigid objects.
    q = np.asarray(q_mj_wxyz, dtype=np.float64)
    return np.array([q[2], -q[3], -q[1], q[0]], dtype=np.float64)


def damped_ls(J, e, lam=0.05):
    # dq = J^T (J J^T + lam^2 I)^-1 e
    JJt = J @ J.T
    A = JJt + (lam * lam) * np.eye(JJt.shape[0], dtype=np.float64)
    return J.T @ np.linalg.solve(A, e)


def _array_vec3(arr, idx: int) -> np.ndarray:
    raw = np.asarray(arr)
    if raw.ndim == 2:
        return np.asarray(raw[idx], dtype=np.float64).reshape(3,)
    return np.asarray(raw[idx * 3 : (idx + 1) * 3], dtype=np.float64).reshape(3,)


def _array_mat3(arr, idx: int) -> np.ndarray:
    raw = np.asarray(arr)
    if raw.ndim == 2:
        block = raw[idx]
    else:
        block = raw[idx * 9 : (idx + 1) * 9]
    return np.asarray(block, dtype=np.float64).reshape(3, 3)


def rotmat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    # Stable conversion for orthonormal rotation matrices.
    m = np.asarray(R, dtype=np.float64)
    tr = float(m[0, 0] + m[1, 1] + m[2, 2])
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return quat_normalize(np.array([w, x, y, z], dtype=np.float64))


def encode_rgb_jpg(rgb_u8: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(
        ".jpg",
        rgb_u8[:, :, ::-1],
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        raise RuntimeError("cv2.imencode failed")
    return buf.tobytes()


def compute_intrinsics(
    width: int,
    height: int,
    fovy_deg: float,
    mode: str = "mujoco",
) -> Tuple[float, float, float, float]:
    # MuJoCo cam_fovy is vertical FOV. For rendered depth, fx=fy is typically correct.
    fovy = np.deg2rad(float(fovy_deg))
    fy = 0.5 * float(height) / np.tan(0.5 * fovy)
    if mode == "aspect":
        fx = fy * (float(width) / float(height))
    else:
        fx = fy
    cx = (float(width) - 1.0) * 0.5
    cy = (float(height) - 1.0) * 0.5
    return float(fx), float(fy), float(cx), float(cy)


def depth_to_pointcloud_xyzrgb(
    depth_m: np.ndarray,
    rgb_u8: np.ndarray,
    width: int,
    height: int,
    fovy_deg: float,
    intrinsics_mode: str = "mujoco",
    stride: int = 4,
    min_depth: float = 0.02,
    max_depth: Optional[float] = None,
    flip_y: bool = False,
    flip_x: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    fx, fy, cx, cy = compute_intrinsics(width, height, fovy_deg, mode=intrinsics_mode)
    yy, xx = np.mgrid[0:height:stride, 0:width:stride]
    z = depth_m[yy, xx]

    valid = np.isfinite(z) & (z > float(min_depth))
    if max_depth is not None:
        valid &= z < float(max_depth)

    x = (xx - cx) * z / fx
    y = (yy - cy) * z / fy
    if flip_x:
        x = -x
    if flip_y:
        y = -y

    xyz = np.stack([x, y, z], axis=-1)[valid].reshape(-1, 3).astype(np.float32)
    rgb = rgb_u8[yy, xx, :][valid].reshape(-1, 3).astype(np.uint8)
    return xyz, rgb


def mujoco_to_unity_xyz(points_mj: np.ndarray) -> np.ndarray:
    if points_mj.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.stack(
        [-points_mj[:, 1], points_mj[:, 2], points_mj[:, 0]],
        axis=1,
    ).astype(np.float32)


def pc7_array(xyz_unity_f32: np.ndarray, rgb_u8: np.ndarray, size: float = 0.02) -> np.ndarray:
    if xyz_unity_f32.size == 0:
        return np.zeros((0, 7), dtype=np.float32)
    xyz = np.asarray(xyz_unity_f32, dtype=np.float32)
    rgb = np.asarray(rgb_u8, dtype=np.float32) / 255.0
    s = np.full((xyz.shape[0], 1), float(size), dtype=np.float32)
    return np.concatenate([xyz, rgb, s], axis=1).astype(np.float32)


def resolve_declared_pc_capacity(
    *,
    cam_name: str,
    width: int,
    height: int,
    stride: int,
    global_cap: int,
    top_cap: int,
) -> int:
    declared = int(global_cap)
    if cam_name == "top" and int(top_cap) > 0:
        declared = int(top_cap)
    if declared > 0:
        return declared
    return sampled_point_capacity(width, height, stride)


def build_rgbd_blob(
    *,
    cam_name: str,
    width: int,
    height: int,
    fovy_deg: float,
    timestamp: float,
    rgb_jpg: bytes,
    depth_f32_m: np.ndarray,
    intrinsics: Dict[str, float],
    cam_pose_mj: Dict[str, List[float]],
    cam_pose_unity: Dict[str, List[float]],
    pc7: Optional[np.ndarray] = None,
    extra_meta: Optional[Dict] = None,
) -> bytes:
    depth = np.asarray(depth_f32_m, dtype=np.float32)
    depth_bytes = depth.tobytes(order="C")
    pc_bytes = b"" if pc7 is None else np.asarray(pc7, dtype=np.float32).tobytes(order="C")

    header = {
        "cam_name": cam_name,
        "width": int(width),
        "height": int(height),
        "fovy_deg": float(fovy_deg),
        "timestamp": float(timestamp),
        "rgb_format": "jpg",
        "depth_format": "f32",
        "pc_format": "f32x7",
        "rgb_len": int(len(rgb_jpg)),
        "depth_len": int(len(depth_bytes)),
        "pc_len": int(len(pc_bytes)),
        "intrinsics": intrinsics,
        "cam_pose_mj": cam_pose_mj,
        "cam_pose_unity": cam_pose_unity,
    }
    if extra_meta:
        header["meta"] = extra_meta

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return struct.pack("<I", len(header_bytes)) + header_bytes + rgb_jpg + depth_bytes + pc_bytes


class GLFWMjrRenderer:
    """
    Windows-safe offscreen renderer:
      - Hidden GLFW window -> valid OpenGL context.
      - MuJoCo mjr_readPixels -> RGB + depth.
      - Depth converted to linear meters.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        width: int,
        height: int,
        visible: bool = False,
        window_title: str = "mujoco_offscreen",
    ):
        self.model = model
        self.width = int(width)
        self.height = int(height)
        self.visible = bool(visible)

        if not glfw.init():
            raise RuntimeError("glfw.init() failed")

        glfw.default_window_hints()
        glfw.window_hint(glfw.VISIBLE, glfw.TRUE if self.visible else glfw.FALSE)
        glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE if self.visible else glfw.FALSE)
        glfw.window_hint(glfw.DEPTH_BITS, 24)
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 2)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)

        win_w = self.width if self.visible else 32
        win_h = self.height if self.visible else 32
        self._win = glfw.create_window(win_w, win_h, window_title, None, None)
        if not self._win:
            glfw.terminate()
            raise RuntimeError("glfw.create_window failed")
        glfw.make_context_current(self._win)
        glfw.swap_interval(0)

        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        mujoco.mjv_defaultCamera(self.cam)
        mujoco.mjv_defaultOption(self.opt)

        self.model.vis.global_.offwidth = self.width
        self.model.vis.global_.offheight = self.height
        self.model.vis.quality.offsamples = 0

        self.con = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        try:
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.con)
        except Exception:
            pass

        self.viewport = mujoco.MjrRect(0, 0, self.width, self.height)
        self.rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.depth = np.zeros((self.height, self.width), dtype=np.float32)

    def poll_events(self):
        try:
            glfw.poll_events()
        except Exception:
            pass

    def is_window_open(self) -> bool:
        try:
            if self._win is None:
                return False
            return not bool(glfw.window_should_close(self._win))
        except Exception:
            return False

    def key_down(self, keycode: int) -> bool:
        try:
            state = glfw.get_key(self._win, keycode)
            return state in (glfw.PRESS, glfw.REPEAT)
        except Exception:
            return False

    def close(self):
        try:
            glfw.make_context_current(self._win)
        except Exception:
            pass
        try:
            glfw.destroy_window(self._win)
        except Exception:
            pass
        glfw.terminate()

    def render(
        self,
        data: mujoco.MjData,
        cam_name: str,
        *,
        blit_to_window: bool = True,
        paused_overlay: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        glfw.make_context_current(self._win)

        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cam_id < 0:
            raise RuntimeError(f"Camera not found: {cam_name}")

        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam.fixedcamid = cam_id

        mujoco.mjv_updateScene(
            self.model,
            data,
            self.opt,
            None,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scene,
        )
        mujoco.mjr_render(self.viewport, self.scene, self.con)
        mujoco.mjr_readPixels(self.rgb, self.depth, self.viewport, self.con)

        if self.visible and blit_to_window:
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.con)
            mujoco.mjr_render(self.viewport, self.scene, self.con)
            if paused_overlay:
                mujoco.mjr_overlay(
                    mujoco.mjtFont.mjFONT_BIG,
                    mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                    self.viewport,
                    "|| PAUSED",
                    "",
                    self.con,
                )
            glfw.swap_buffers(self._win)
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.con)

        depth_m = depth_buffer_to_meters(self.model, self.depth)
        return self.rgb.copy(), depth_m.astype(np.float32)


def depth_buffer_to_meters(model: mujoco.MjModel, depth_buf: np.ndarray) -> np.ndarray:
    # MuJoCo depth conversion uses clipping planes scaled by model extent.
    extent = float(getattr(model.stat, "extent", 1.0))
    znear = float(model.vis.map.znear) * extent
    zfar = float(model.vis.map.zfar) * extent
    d = np.clip(depth_buf, 1e-6, 1.0 - 1e-6).astype(np.float32)
    return (2.0 * znear * zfar) / (zfar + znear - (2.0 * d - 1.0) * (zfar - znear))


def quat_wxyz_to_mat(q_wxyz: np.ndarray) -> np.ndarray:
    q = quat_normalize(np.asarray(q_wxyz, dtype=np.float64))
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def euler_xyz_deg_to_mat(euler_deg: List[float]) -> np.ndarray:
    rx, ry, rz = np.deg2rad(np.asarray(euler_deg, dtype=np.float64))
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    return (Rz @ Ry @ Rx).astype(np.float64)


@dataclass
class CameraExtrinsic:
    R_offset: np.ndarray  # 3x3
    t_offset: np.ndarray  # 3,
    source: str


def _parse_rotation_matrix(cfg: Dict) -> np.ndarray:
    if "R" in cfg:
        return np.asarray(cfg["R"], dtype=np.float64).reshape(3, 3)
    if "rotation_matrix" in cfg:
        return np.asarray(cfg["rotation_matrix"], dtype=np.float64).reshape(3, 3)
    if "quat_wxyz" in cfg:
        return quat_wxyz_to_mat(np.asarray(cfg["quat_wxyz"], dtype=np.float64).reshape(4,))
    if "quat_xyzw" in cfg:
        q = np.asarray(cfg["quat_xyzw"], dtype=np.float64).reshape(4,)
        q_wxyz = np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)
        return quat_wxyz_to_mat(q_wxyz)
    if "rpy_deg" in cfg:
        return euler_xyz_deg_to_mat(cfg["rpy_deg"])
    if "euler_deg" in cfg:
        return euler_xyz_deg_to_mat(cfg["euler_deg"])
    return np.eye(3, dtype=np.float64)


def _parse_translation(cfg: Dict) -> np.ndarray:
    if "t" in cfg:
        return np.asarray(cfg["t"], dtype=np.float64).reshape(3,)
    if "translation" in cfg:
        return np.asarray(cfg["translation"], dtype=np.float64).reshape(3,)
    if "offset_m" in cfg:
        return np.asarray(cfg["offset_m"], dtype=np.float64).reshape(3,)
    return np.zeros(3, dtype=np.float64)


def load_camera_extrinsics(path: Optional[str]) -> Dict[str, CameraExtrinsic]:
    if not path:
        return {}

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Extrinsics JSON not found: {path}")

    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "cameras" in raw and isinstance(raw["cameras"], dict):
        raw = raw["cameras"]
    if not isinstance(raw, dict):
        raise ValueError("Extrinsics JSON must be a dict or {\"cameras\": {...}}.")

    out: Dict[str, CameraExtrinsic] = {}
    for cam_name, cfg in raw.items():
        if not isinstance(cfg, dict):
            continue
        R = _parse_rotation_matrix(cfg)
        t = _parse_translation(cfg)
        out[str(cam_name)] = CameraExtrinsic(
            R_offset=R,
            t_offset=t,
            source=str(path),
        )
    return out


def get_calibrated_cam_pose(
    data: mujoco.MjData,
    cam_id: int,
    calib: Optional[CameraExtrinsic],
) -> Tuple[np.ndarray, np.ndarray]:
    t_raw = _array_vec3(data.cam_xpos, cam_id)
    R_raw = _array_mat3(data.cam_xmat, cam_id)
    if calib is None:
        return t_raw, R_raw
    # Apply local camera-frame offset: T_world_cam_cal = T_world_cam_raw * T_cam_offset
    R = R_raw @ calib.R_offset
    t = t_raw + (R_raw @ calib.t_offset)
    return t, R


@dataclass
class Plane:
    point: np.ndarray   # 3,
    normal: np.ndarray  # 3, unit
    source: str


def resolve_table_plane(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom_candidates: List[str],
    body_name: str,
    manual_plane_z: Optional[float],
    body_top_offset: float,
) -> Optional[Plane]:
    if manual_plane_z is not None:
        return Plane(
            point=np.array([0.0, 0.0, float(manual_plane_z)], dtype=np.float64),
            normal=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            source=f"manual_z={manual_plane_z}",
        )

    for gname in geom_candidates:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, gname)
        if gid < 0:
            continue
        gpos = _array_vec3(data.geom_xpos, gid)
        gmat = _array_mat3(data.geom_xmat, gid)

        normal = np.asarray(gmat[:, 2], dtype=np.float64).reshape(3,)
        nrm = np.linalg.norm(normal)
        if nrm < 1e-9:
            continue
        normal = normal / nrm
        if normal[2] < 0:
            normal = -normal

        gtype = int(model.geom_type[gid])
        lift = 0.0
        if gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
            lift = float(model.geom_size[gid][2])
        elif gtype == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
            lift = float(model.geom_size[gid][1])
        elif gtype == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
            lift = float(model.geom_size[gid][1] + model.geom_size[gid][0])

        point = gpos + normal * lift
        return Plane(point=point, normal=normal, source=f"geom:{gname}")

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid >= 0:
        bpos = _array_vec3(data.xpos, bid)
        point = np.array(
            [float(bpos[0]), float(bpos[1]), float(bpos[2] + body_top_offset)],
            dtype=np.float64,
        )
        return Plane(
            point=point,
            normal=np.array([0.0, 0.0, 1.0], dtype=np.float64),
            source=f"body:{body_name}",
        )
    return None


def clip_points_below_plane(
    points_world_mj: np.ndarray,
    colors_u8: np.ndarray,
    plane: Optional[Plane],
    margin: float,
    clearance: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, int]:
    if plane is None or points_world_mj.size == 0:
        return points_world_mj, colors_u8, 0
    threshold = max(float(margin), float(clearance))
    signed = (points_world_mj - plane.point.reshape(1, 3)) @ plane.normal.reshape(3, 1)
    keep = signed.reshape(-1) > threshold
    removed = int(points_world_mj.shape[0] - np.count_nonzero(keep))
    return points_world_mj[keep], colors_u8[keep], removed


def clip_points_outside_aabb(
    points_world_mj: np.ndarray,
    colors_u8: np.ndarray,
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int]:
    if points_world_mj.size == 0:
        return points_world_mj, colors_u8, 0
    lo = np.asarray(aabb_min, dtype=np.float64).reshape(1, 3)
    hi = np.asarray(aabb_max, dtype=np.float64).reshape(1, 3)
    keep = np.all(points_world_mj >= lo, axis=1) & np.all(points_world_mj <= hi, axis=1)
    removed = int(points_world_mj.shape[0] - np.count_nonzero(keep))
    return points_world_mj[keep], colors_u8[keep], removed


def get_anchor_world_mj(
    data: mujoco.MjData,
    anchor_site_id: int = -1,
    anchor_body_id: int = -1,
) -> Optional[np.ndarray]:
    if anchor_site_id >= 0:
        return _array_vec3(data.site_xpos, anchor_site_id)
    if anchor_body_id >= 0:
        return _array_vec3(data.xpos, anchor_body_id)
    return None


def project_world_to_pixel(
    point_world_mj: np.ndarray,
    cam_t_mj: np.ndarray,
    cam_R_mj: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> Optional[Tuple[float, float, float]]:
    # p_world = p_cam_mj @ R.T + t  => p_cam_mj = (p_world - t) @ R
    # MuJoCo camera forward is -Z. Convert to depth-frame z>0 via z_depth = -z_mj.
    p_cam_mj = (point_world_mj - cam_t_mj) @ cam_R_mj
    z_depth = float(-p_cam_mj[2])
    if z_depth <= 1e-6:
        return None
    u = (float(p_cam_mj[0]) * float(fx) / z_depth) + float(cx)
    v = (float(p_cam_mj[1]) * float(fy) / z_depth) + float(cy)
    return float(u), float(v), z_depth


def sample_depth_at_pixel(
    depth_m: np.ndarray,
    u: float,
    v: float,
    radius: int = 3,
    preferred_depth: Optional[float] = None,
    max_abs_error: Optional[float] = None,
) -> Optional[float]:
    if depth_m.ndim != 2:
        return None
    h, w = depth_m.shape
    ui = int(round(u))
    vi = int(round(v))
    if ui < 0 or ui >= w or vi < 0 or vi >= h:
        return None

    r = max(0, int(radius))
    u0 = max(0, ui - r)
    u1 = min(w - 1, ui + r)
    v0 = max(0, vi - r)
    v1 = min(h - 1, vi + r)

    patch = depth_m[v0 : v1 + 1, u0 : u1 + 1]
    valid = patch[np.isfinite(patch) & (patch > 1e-6)]
    if valid.size == 0:
        return None
    if preferred_depth is not None:
        diffs = np.abs(valid - float(preferred_depth))
        best_idx = int(np.argmin(diffs))
        best = float(valid[best_idx])
        if max_abs_error is not None and float(diffs[best_idx]) > float(max_abs_error):
            return None
        return best
    return float(np.median(valid))


def estimate_anchor_translation_correction(
    *,
    anchor_world_mj: np.ndarray,
    depth_m: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    cam_t_mj: np.ndarray,
    cam_R_mj: np.ndarray,
    flip_x: bool,
    flip_y: bool,
    flip_z: bool,
    pc_scale: float,
    patch_radius: int,
    depth_tol: float,
    max_corr_norm: float,
) -> Optional[np.ndarray]:
    proj = project_world_to_pixel(
        point_world_mj=anchor_world_mj,
        cam_t_mj=cam_t_mj,
        cam_R_mj=cam_R_mj,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )
    if proj is None:
        return None
    u, v, z_expected = proj
    max_err = float(depth_tol) if float(depth_tol) > 0.0 else None
    z_measured = sample_depth_at_pixel(
        depth_m,
        u,
        v,
        radius=patch_radius,
        preferred_depth=float(z_expected),
        max_abs_error=max_err,
    )
    if z_measured is None:
        return None
    depth_err = abs(z_measured - z_expected)
    if float(depth_tol) > 0.0 and depth_err > float(depth_tol):
        return None
    atten = 1.0
    if float(depth_tol) > 1e-9:
        # Conservative weighting near tolerance boundary.
        atten = float(np.clip(1.0 - (depth_err / float(depth_tol)), 0.1, 1.0))

    x = (u - cx) * z_measured / fx
    y = (v - cy) * z_measured / fy
    z = z_measured

    if flip_x:
        x = -x
    if flip_y:
        y = -y
    if flip_z:
        z = -z

    p_cam = np.array([x, y, z], dtype=np.float64) * float(pc_scale)
    p_world_est = (p_cam @ cam_R_mj.T) + cam_t_mj
    corr = np.asarray(anchor_world_mj - p_world_est, dtype=np.float64)

    nrm = float(np.linalg.norm(corr))
    if nrm > float(max_corr_norm) and nrm > 1e-9:
        corr = corr * (float(max_corr_norm) / nrm)
    return corr * float(atten)


def _mark_pixels_rgb(
    image_rgb: np.ndarray,
    pixels_yx: np.ndarray,
    color_rgb: Tuple[int, int, int],
    *,
    radius: int = 1,
) -> None:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3 or pixels_yx.size == 0:
        return
    h, w = image_rgb.shape[:2]
    pts = np.asarray(pixels_yx, dtype=np.int32).reshape(-1, 2)
    r = max(0, int(radius))
    color = np.asarray(color_rgb, dtype=np.uint8).reshape(1, 1, 3)
    for y, x in pts:
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        y0 = max(0, y - r)
        y1 = min(h, y + r + 1)
        x0 = max(0, x - r)
        x1 = min(w, x + r + 1)
        image_rgb[y0:y1, x0:x1] = color


def _workspace_aabb_corners(aabb_min: np.ndarray, aabb_max: np.ndarray) -> np.ndarray:
    lo = np.asarray(aabb_min, dtype=np.float64).reshape(3,)
    hi = np.asarray(aabb_max, dtype=np.float64).reshape(3,)
    return np.asarray(
        [
            [lo[0], lo[1], lo[2]],
            [lo[0], lo[1], hi[2]],
            [lo[0], hi[1], lo[2]],
            [lo[0], hi[1], hi[2]],
            [hi[0], lo[1], lo[2]],
            [hi[0], lo[1], hi[2]],
            [hi[0], hi[1], lo[2]],
            [hi[0], hi[1], hi[2]],
        ],
        dtype=np.float64,
    )


def _draw_projected_aabb_corners_rgb(
    image_rgb: np.ndarray,
    *,
    aabb_min: np.ndarray,
    aabb_max: np.ndarray,
    cam_t_mj: np.ndarray,
    cam_R_mj: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> Tuple[int, int]:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        return 0, 0
    h, w = image_rgb.shape[:2]
    front_count = 0
    in_image_count = 0
    for idx, corner in enumerate(_workspace_aabb_corners(aabb_min, aabb_max)):
        proj = project_world_to_pixel(
            point_world_mj=np.asarray(corner, dtype=np.float64),
            cam_t_mj=np.asarray(cam_t_mj, dtype=np.float64),
            cam_R_mj=np.asarray(cam_R_mj, dtype=np.float64),
            fx=float(fx),
            fy=float(fy),
            cx=float(cx),
            cy=float(cy),
        )
        if proj is None:
            continue
        front_count += 1
        u, v, _ = proj
        if 0.0 <= u < float(w) and 0.0 <= v < float(h):
            in_image_count += 1
            center = (int(round(u)), int(round(v)))
            cv2.circle(image_rgb, center, 4, (255, 255, 255), 1)
            cv2.putText(
                image_rgb,
                str(idx),
                (center[0] + 3, center[1] - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA,
            )
    return front_count, in_image_count


def _world_x_histogram_counts(x_world_mj: np.ndarray, bins: int = 6) -> List[int]:
    x = np.asarray(x_world_mj, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return []
    lo = float(np.min(x))
    hi = float(np.max(x))
    if hi - lo < 1e-6:
        counts = [0] * int(max(1, bins))
        counts[0] = int(x.size)
        return counts
    hist, _ = np.histogram(x, bins=int(max(1, bins)), range=(lo, hi))
    return [int(v) for v in hist.tolist()]


def save_pc_visibility_debug_artifacts(
    *,
    outdir: Path,
    cam_name: str,
    rgb_u8: np.ndarray,
    debug_visibility,
    object_bbox_min: np.ndarray,
    object_bbox_max: np.ndarray,
    cam_t_mj: np.ndarray,
    cam_R_mj: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> Dict[str, object]:
    outdir.mkdir(parents=True, exist_ok=True)
    rgb_ref = np.ascontiguousarray(rgb_u8.astype(np.uint8, copy=True))
    selected = rgb_ref.copy()
    rejected = rgb_ref.copy()

    _mark_pixels_rgb(selected, debug_visibility.sampled_pixels_yx, (255, 255, 0), radius=1)
    _mark_pixels_rgb(selected, debug_visibility.valid_depth_pixels_yx, (0, 255, 255), radius=1)
    _mark_pixels_rgb(selected, debug_visibility.kept_pixels_yx, (0, 255, 0), radius=1)
    _mark_pixels_rgb(rejected, debug_visibility.rejected_bbox_pixels_yx, (255, 0, 0), radius=1)
    _mark_pixels_rgb(rejected, debug_visibility.rejected_table_pixels_yx, (255, 0, 255), radius=1)

    aabb_front, aabb_in_image = _draw_projected_aabb_corners_rgb(
        selected,
        aabb_min=object_bbox_min,
        aabb_max=object_bbox_max,
        cam_t_mj=cam_t_mj,
        cam_R_mj=cam_R_mj,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )
    _draw_projected_aabb_corners_rgb(
        rejected,
        aabb_min=object_bbox_min,
        aabb_max=object_bbox_max,
        cam_t_mj=cam_t_mj,
        cam_R_mj=cam_R_mj,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
    )

    rgb_path = outdir / f"{cam_name}_rgb.png"
    selected_path = outdir / f"{cam_name}_selected.png"
    rejected_path = outdir / f"{cam_name}_rejected.png"

    if not cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_ref, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"cv2.imwrite failed for {rgb_path}")
    if not cv2.imwrite(str(selected_path), cv2.cvtColor(selected, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"cv2.imwrite failed for {selected_path}")
    if not cv2.imwrite(str(rejected_path), cv2.cvtColor(rejected, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"cv2.imwrite failed for {rejected_path}")

    return {
        "rgb_path": str(rgb_path),
        "selected_path": str(selected_path),
        "rejected_path": str(rejected_path),
        "aabb_front_count": int(aabb_front),
        "aabb_in_image_count": int(aabb_in_image),
    }


class SensorNode:
    def __init__(
        self,
        *,
        name: str,
        bind_ip: str,
        host_ip: str,
        service_port: int,
        topic_port: int,
        topic_list: List[str],
    ):
        self.name = name
        self.bind_ip = bind_ip
        self.host_ip = host_ip
        self.service_port = int(service_port)
        self.topic_port = int(topic_port)
        self.topic_list = list(topic_list)

        self.node_id = str(uuid.uuid4())
        self.node_info_id = str(uuid.uuid4())

        self._stop_evt = threading.Event()
        self._ctx = zmq.Context.instance()

        self._pub = self._ctx.socket(zmq.PUB)
        self._pub.setsockopt(zmq.LINGER, 0)
        self._bind_socket_or_raise(
            self._pub,
            f"tcp://{self.bind_ip}:{self.topic_port}",
            label="Sensor PUB",
        )

        self._t_disc = threading.Thread(target=self._discovery_loop, daemon=True)
        self._t_rep = threading.Thread(target=self._rep_loop, daemon=True)

    def start(self):
        self._t_disc.start()
        self._t_rep.start()

    def stop(self):
        self._stop_evt.set()
        time.sleep(0.1)
        try:
            self._pub.close(0)
        except Exception:
            pass

    def publish(self, topic: str, payload: bytes):
        self._pub.send_multipart([topic.encode("utf-8"), payload])

    def _bind_socket_or_raise(self, sock: zmq.Socket, addr: str, *, label: str):
        try:
            sock.bind(addr)
        except zmq.ZMQError as e:
            if e.errno == zmq.EADDRINUSE:
                raise RuntimeError(
                    f"{label} could not bind to {addr} because the address is already in use. "
                    f"This usually means an older publisher process is still running. "
                    f"On Windows, check with 'netstat -ano | findstr {addr.rsplit(':', 1)[-1]}' "
                    f"and stop the owning PID, or rerun with different --service_port/--topic_port values."
                ) from e
            raise

    def _node_info_dict(self) -> Dict:
        return {
            "name": self.name,
            "type": "SimPub",
            "nodeInfoID": self.node_info_id,
            "servicePort": self.service_port,
            "topicPort": self.topic_port,
            "serviceList": [],
            "topicList": self.topic_list,
            "ip": self.host_ip,
            "port": 0,
        }

    def _discovery_loop(self):
        msg = f"{self.node_id}{self.node_info_id}{self.service_port}".encode("utf-8")
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 1)
        try:
            sock.setsockopt(
                socket.IPPROTO_IP,
                socket.IP_MULTICAST_IF,
                socket.inet_aton(self.host_ip),
            )
        except OSError as e:
            print(f"[Unified][WARN] Could not set IP_MULTICAST_IF={self.host_ip}: {e}")

        while not self._stop_evt.is_set():
            sock.sendto(msg, (MULTICAST_GRP, DISCOVERY_PORT))
            time.sleep(0.5)
        sock.close()

    def _rep_loop(self):
        rep = self._ctx.socket(zmq.REP)
        rep.setsockopt(zmq.LINGER, 0)
        self._bind_socket_or_raise(
            rep,
            f"tcp://{self.bind_ip}:{self.service_port}",
            label="Sensor REP",
        )

        while not self._stop_evt.is_set():
            try:
                if rep.poll(50) == 0:
                    continue
                parts = rep.recv_multipart()
                if not parts:
                    continue

                if len(parts) == 1:
                    raw = parts[0]
                    if b"|" in raw:
                        svc_b, _ = raw.split(b"|", 1)
                    else:
                        svc_b = raw
                else:
                    svc_b = parts[0]

                service = svc_b.decode("utf-8", errors="ignore").strip()
                if service in ("GetNodeInfo", "GetMasterInfo"):
                    payload = json.dumps(self._node_info_dict(), separators=(",", ":")).encode(
                        "utf-8"
                    )
                    rep.send(payload)
                else:
                    rep.send(REP_SUCCESS)
            except zmq.error.ZMQError:
                break
            except Exception:
                try:
                    rep.send(REP_SUCCESS)
                except Exception:
                    pass

        rep.close(0)


def _choose_cameras(requested: Optional[List[str]], available: List[str]) -> List[str]:
    if requested:
        chosen = [c for c in requested if c in available]
        missing = [c for c in requested if c not in available]
        if missing:
            print(f"[Unified][WARN] Missing cameras in XML (ignored): {missing}")
        return chosen

    preferred = ["top", "front", "left", "right", "wrist", "overhead"]
    chosen = [c for c in preferred if c in available]
    for c in available:
        if c not in chosen:
            chosen.append(c)
    return chosen


@dataclass
class MovableBody:
    body_name: str
    body_id: int
    joint_name: str
    joint_id: int
    qpos_adr: int
    dof_adr: int


def discover_movable_free_bodies(
    model: mujoco.MjModel,
    requested_bodies: Optional[List[str]] = None,
) -> List[MovableBody]:
    requested = None
    if requested_bodies:
        requested = set([str(x) for x in requested_bodies])

    out: List[MovableBody] = []
    free_type = int(mujoco.mjtJoint.mjJNT_FREE)
    for j in range(model.njnt):
        if int(model.jnt_type[j]) != free_type:
            continue
        body_id = int(model.jnt_bodyid[j])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
        if requested is not None and body_name not in requested:
            continue
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint_{j}"
        out.append(
            MovableBody(
                body_name=body_name,
                body_id=body_id,
                joint_name=joint_name,
                joint_id=j,
                qpos_adr=int(model.jnt_qposadr[j]),
                dof_adr=int(model.jnt_dofadr[j]),
            )
        )

    if requested is not None:
        missing = sorted(list(requested - set([o.body_name for o in out])))
        if missing:
            print(f"[Unified][WARN] Object control bodies not found or not free-joint: {missing}")
    return out


def key_pressed_once(
    renderer: GLFWMjrRenderer,
    key_latch: Dict[int, bool],
    keycode: int,
) -> bool:
    down = renderer.key_down(keycode)
    prev = key_latch.get(keycode, False)
    key_latch[keycode] = down
    return down and (not prev)


def key_down_any(renderer: GLFWMjrRenderer, keycodes: List[int]) -> bool:
    for k in keycodes:
        if renderer.key_down(k):
            return True
    return False


def apply_movable_translation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    movable: MovableBody,
    delta_xyz: np.ndarray,
):
    adr = int(movable.qpos_adr)
    data.qpos[adr : adr + 3] = data.qpos[adr : adr + 3] + np.asarray(delta_xyz, dtype=np.float64).reshape(3,)
    vadr = int(movable.dof_adr)
    if vadr >= 0 and (vadr + 6) <= data.qvel.shape[0]:
        data.qvel[vadr : vadr + 6] = 0.0
    mujoco.mj_forward(model, data)


def set_robot_joint_targets(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    q7: np.ndarray,
):
    """Set panda joint positions by joint names, avoiding qpos-order assumptions."""
    q7 = np.asarray(q7, dtype=np.float64).reshape(7,)
    joint_names = [f"joint{i}" for i in range(1, 8)]
    found = 0
    for i, jn in enumerate(joint_names):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            continue
        qadr = int(model.jnt_qposadr[jid])
        dadr = int(model.jnt_dofadr[jid])
        if 0 <= qadr < data.qpos.shape[0]:
            data.qpos[qadr] = q7[i]
            found += 1
        if 0 <= dadr < data.qvel.shape[0]:
            data.qvel[dadr] = 0.0

    if data.ctrl.shape[0] >= 7:
        data.ctrl[0:7] = q7
    if data.ctrl.shape[0] > 7:
        data.ctrl[7] = 0.04

    mujoco.mj_forward(model, data)
    if found < 7:
        print(f"[Unified][WARN] Robot preset applied partially ({found}/7 joints found).")


def get_robot_arm_indices(model: mujoco.MjModel) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Resolve arm joint qpos/dof indices and matching actuator ctrl indices by joint name.
    Returns:
      qpos_idx: shape (7,)
      dof_idx:  shape (7,)
      ctrl_idx: shape (7,)  (-1 where no matching actuator was found)
    """
    qpos_idx = np.full(7, -1, dtype=np.int32)
    dof_idx = np.full(7, -1, dtype=np.int32)
    ctrl_idx = np.full(7, -1, dtype=np.int32)

    for i in range(7):
        jname = f"joint{i + 1}"
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            continue
        qpos_idx[i] = int(model.jnt_qposadr[jid])
        dof_idx[i] = int(model.jnt_dofadr[jid])

        for a in range(model.nu):
            if int(model.actuator_trnid[a, 0]) == jid:
                ctrl_idx[i] = a
                break

    return qpos_idx, dof_idx, ctrl_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True, help="Path to MJCF scene")
    ap.add_argument("--host", required=True, help="PC LAN IP (scene + sensor discovery)")
    ap.add_argument("--host_ip", default=None, help=argparse.SUPPRESS)  # backward compat alias
    ap.add_argument("--unity_node", required=True, help="Quest node name from the SimPub dashboard")
    ap.add_argument("--bind_ip", default="0.0.0.0")

    # Scene publishing
    ap.add_argument(
        "--visible_geoms_groups",
        nargs="+",
        type=int,
        default=[2],
        help="MuJoCo geom groups to spawn in Unity scene, e.g. 2 or 0 2",
    )

    # Teleop/control loop
    ap.add_argument("--control_hz", type=float, default=120.0)
    ap.add_argument(
        "--show_mujoco_window",
        action="store_true",
        help="Show MuJoCo GLFW render window (needed for keyboard object control)",
    )
    ap.add_argument(
        "--object_control",
        action="store_true",
        help="Enable keyboard nudging for free-joint objects in the MuJoCo render window",
    )
    ap.add_argument(
        "--object_control_bodies",
        nargs="*",
        default=None,
        help="Optional body names for object control (defaults to all free-joint bodies)",
    )
    ap.add_argument(
        "--object_step_xy",
        type=float,
        default=0.01,
        help="Translation step in X/Y for object control (meters)",
    )
    ap.add_argument(
        "--object_step_z",
        type=float,
        default=0.008,
        help="Translation step in Z for object control (meters)",
    )
    ap.add_argument(
        "--object_move_rate_hz",
        type=float,
        default=10.0,
        help="Repeat rate while holding movement keys",
    )
    ap.add_argument(
        "--robot_start_preset",
        choices=["xml_default", "reverse_u"],
        default="reverse_u",
        help="Initial robot joint posture at startup",
    )
    ap.add_argument(
        "--robot_start_q",
        nargs=7,
        type=float,
        default=None,
        metavar=("Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"),
        help="Optional manual initial 7-joint target in radians (overrides --robot_start_preset)",
    )

    # Sensor transport
    ap.add_argument("--service_port", type=int, default=7740)
    ap.add_argument("--topic_port", type=int, default=7741)
    ap.add_argument("--fps", type=float, default=30.0, help="Sensor publish FPS")
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--cams", nargs="*", default=None)
    ap.add_argument(
        "--rgb_cams",
        nargs="*",
        default=None,
        help="Cameras that publish standalone SimPub/Sensors/<cam>/rgb topics (default: wrist)",
    )
    ap.add_argument(
        "--rgbd_cams",
        nargs="*",
        default=None,
        help="Cameras that publish SimPub/Sensors/<cam>/rgbd topics (default: none)",
    )
    ap.add_argument("--jpg_quality", type=int, default=85)
    ap.add_argument(
        "--no_rgb_topic",
        action="store_true",
        help="Disable standalone SimPub/Sensors/<cam>/rgb (RGB stays in /rgbd)",
    )

    # Point cloud
    ap.add_argument("--pc", action="store_true")
    ap.add_argument("--pc_cams", nargs="*", default=None)
    ap.add_argument(
        "--pc_backend",
        choices=["gpu", "legacy"],
        default="gpu",
        help="Point-cloud processing backend. 'gpu' requires Open3D with CUDA; 'legacy' keeps the NumPy processing path but still publishes the compact /pc contract.",
    )
    ap.add_argument(
        "--pc_sampling",
        choices=["grid", "stable_random"],
        default="grid",
        help="Primary pixel selection policy before filtering. 'grid' preserves compatibility; 'stable_random' picks one deterministic random pixel per stride cell and is the recommended validation mode for Quest side-camera coverage checks.",
    )
    ap.add_argument("--pc_stride", type=int, default=4)
    ap.add_argument("--pc_size", type=float, default=0.02)
    ap.add_argument("--pc_scale", type=float, default=1.0)
    ap.add_argument(
        "--pc_intrinsics_mode",
        choices=["mujoco", "aspect"],
        default="mujoco",
        help="Intrinsics model: 'mujoco' uses fx=fy, 'aspect' uses fx=fy*(w/h)",
    )
    ap.add_argument("--pc_min_depth", type=float, default=0.05)
    ap.add_argument("--pc_max_depth", type=float, default=None)
    ap.add_argument("--pc_flip_x", action="store_true")
    ap.add_argument("--pc_flip_y", action="store_true")
    ap.add_argument("--pc_no_flip_z", action="store_true")
    ap.add_argument("--pc_flip_z", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--pc_max_points", type=int, default=0, help="Hard cap after all filtering")
    ap.add_argument(
        "--pc_max_points_top",
        type=int,
        default=0,
        help="Optional hard cap for top-camera point cloud only; 0 uses --pc_max_points",
    )
    ap.add_argument(
        "--pc_object_only",
        action="store_true",
        help="Keep point cloud only inside a workspace AABB for object-focused VR rendering",
    )
    ap.add_argument(
        "--pc_object_bbox_min",
        nargs=3,
        type=float,
        default=[0.35, -0.35, 0.40],
        metavar=("X", "Y", "Z"),
        help="Workspace AABB min corner (MuJoCo world frame) used by --pc_object_only",
    )
    ap.add_argument(
        "--pc_object_bbox_max",
        nargs=3,
        type=float,
        default=[0.90, 0.35, 0.85],
        metavar=("X", "Y", "Z"),
        help="Workspace AABB max corner (MuJoCo world frame) used by --pc_object_only",
    )
    ap.add_argument(
        "--rgbd_include_pc",
        action="store_true",
        help="Also append point cloud bytes to /rgbd blobs (off by default for bandwidth)",
    )
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

    # Camera extrinsics
    ap.add_argument(
        "--cam_extrinsics_json",
        default=None,
        help=(
            "JSON file with per-camera extrinsic offsets. "
            "Format: {\"top\": {\"translation\": [x,y,z], \"rpy_deg\": [rx,ry,rz]}} "
            "or use quat_wxyz/quat_xyzw/rotation_matrix."
        ),
    )
    ap.add_argument(
        "--cam_extrinsics_profile",
        choices=["none", "lab_standard"],
        default="lab_standard",
        help="Built-in extrinsics profile used when --cam_extrinsics_json is not provided",
    )

    # Table-plane clipping (point cloud reduction for VR)
    ap.add_argument(
        "--pc_clip_below_table",
        action="store_true",
        help="Keep only points above table top plane (removes floor/table clutter)",
    )
    ap.add_argument(
        "--pc_table_geoms",
        nargs="*",
        default=["table_collision", "table_visual"],
        help="Geom names used to estimate table top plane",
    )
    ap.add_argument("--pc_table_body", default="table")
    ap.add_argument("--pc_table_plane_z", type=float, default=None, help="Manual horizontal table plane Z in MJ frame")
    ap.add_argument("--pc_table_margin", type=float, default=0.005, help="Keep points strictly above plane+margin")
    ap.add_argument(
        "--pc_table_clearance",
        type=float,
        default=0.015,
        help="Additional clearance above plane to suppress tabletop points/noise",
    )
    ap.add_argument(
        "--pc_top_table_clearance",
        type=float,
        default=None,
        help="Optional top-camera-specific clearance above table plane",
    )
    ap.add_argument("--pc_table_body_top_offset", type=float, default=0.02)
    ap.add_argument(
        "--pc_anchor_auto_translate",
        dest="pc_anchor_auto_translate",
        action="store_true",
        default=False,
        help="Auto-estimate per-camera translation correction from robot anchor point",
    )
    ap.add_argument(
        "--pc_no_anchor_auto_translate",
        dest="pc_anchor_auto_translate",
        action="store_false",
        help="Disable anchor-based translation auto calibration",
    )
    ap.add_argument(
        "--pc_anchor_site",
        default=None,
        help="Anchor site name (preferred) for auto translation",
    )
    ap.add_argument(
        "--pc_anchor_body",
        default="link0",
        help="Anchor body name fallback for auto translation",
    )
    ap.add_argument(
        "--pc_anchor_alpha",
        type=float,
        default=0.05,
        help="EMA gain [0..1] for anchor correction update",
    )
    ap.add_argument(
        "--pc_anchor_patch",
        type=int,
        default=3,
        help="Depth sampling patch radius around anchor projection (pixels)",
    )
    ap.add_argument(
        "--pc_anchor_depth_tol",
        type=float,
        default=0.15,
        help="Reject update if measured-vs-expected anchor depth differs more than this (m)",
    )
    ap.add_argument(
        "--pc_anchor_max_corr",
        type=float,
        default=0.08,
        help="Clamp correction magnitude to this max in meters",
    )
    ap.add_argument(
        "--pc_anchor_source",
        choices=["auto", "robot", "table"],
        default="auto",
        help="Anchor source for auto translation: robot base/site, table center, or auto fallback",
    )
    ap.add_argument(
        "--pc_anchor_world_xyz",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Manual anchor point in MuJoCo world coordinates (overrides anchor source lookup)",
    )
    ap.add_argument(
        "--pc_anchor_update_mode",
        choices=["global", "per_camera"],
        default="global",
        help="Apply anchor correction as one shared translation or separately per camera",
    )
    ap.add_argument("--log_every", type=int, default=60)

    args = ap.parse_args()

    if args.object_control:
        args.show_mujoco_window = True

    host_ip = args.host_ip or args.host
    publish_rgb_topic = not args.no_rgb_topic
    object_bbox_min = np.asarray(args.pc_object_bbox_min, dtype=np.float64).reshape(3,)
    object_bbox_max = np.asarray(args.pc_object_bbox_max, dtype=np.float64).reshape(3,)
    if args.pc_object_only and np.any(object_bbox_min >= object_bbox_max):
        raise ValueError("--pc_object_bbox_min must be strictly smaller than --pc_object_bbox_max")

    extrinsics_path = args.cam_extrinsics_json
    if not extrinsics_path and args.cam_extrinsics_profile == "lab_standard":
        if STANDARD_EXTRINSICS_PROFILE.exists():
            extrinsics_path = str(STANDARD_EXTRINSICS_PROFILE)
        else:
            print(
                "[Unified][WARN] Standard extrinsics profile requested but file is missing: "
                f"{STANDARD_EXTRINSICS_PROFILE}"
            )

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)
    arm_qpos_idx, arm_dof_idx, arm_ctrl_idx = get_robot_arm_indices(model)
    if np.any(arm_qpos_idx < 0) or np.any(arm_dof_idx < 0):
        print(
            "[Unified][WARN] Could not fully resolve joint1..joint7 indices. "
            "Teleop may not behave as expected."
        )
    if np.any(arm_ctrl_idx < 0):
        print(
            "[Unified][WARN] Could not map all arm actuators by joint. "
            "Falling back to ctrl[0:7] for missing entries."
        )
    if args.robot_start_q is not None:
        set_robot_joint_targets(model, data, np.asarray(args.robot_start_q, dtype=np.float64))
        print(f"[Unified] Robot start pose: manual q={np.round(np.asarray(args.robot_start_q), 4)}")
    elif args.robot_start_preset == "reverse_u":
        set_robot_joint_targets(model, data, REVERSE_U_JOINTS_RAD)
        print(f"[Unified] Robot start pose: reverse_u q={np.round(REVERSE_U_JOINTS_RAD, 4)}")
    else:
        print("[Unified] Robot start pose: XML default")
    mujoco.mj_forward(model, data)

    available_cams = list_cameras(model)
    cams = _choose_cameras(args.cams, available_cams)
    if not cams:
        raise RuntimeError(f"No valid cameras found. Available cameras: {available_cams}")
    rgb_requested = ["wrist"] if args.rgb_cams is None else list(args.rgb_cams)
    rgbd_requested = [] if args.rgbd_cams is None else list(args.rgbd_cams)
    rgb_cam_set = set(_choose_cameras(rgb_requested, cams))
    rgbd_cam_set = set([c for c in rgbd_requested if c in cams])
    missing_rgbd_cams = [c for c in rgbd_requested if c not in cams]
    if missing_rgbd_cams:
        print(f"[Unified][WARN] RGBD cameras not in selected cams (ignored): {missing_rgbd_cams}")

    pc_max_depth = args.pc_max_depth
    if pc_max_depth is None:
        pc_max_depth = float(model.vis.map.zfar) * 0.95

    pc_cam_set = set()
    pc_declared_capacity_by_cam: Dict[str, int] = {}
    pc_target_sample_budget = sampled_point_capacity(args.w, args.h, max(1, int(args.pc_stride)))
    pc_pipeline = None
    if args.pc:
        requested_pc = args.pc_cams if args.pc_cams else list(cams)
        pc_cam_set = set([c for c in requested_pc if c in cams])
        missing_pc = [c for c in requested_pc if c not in cams]
        if missing_pc:
            print(f"[Unified][WARN] PC cameras not in selected cams (ignored): {missing_pc}")
        for cam_name in sorted(pc_cam_set):
            pc_declared_capacity_by_cam[cam_name] = resolve_declared_pc_capacity(
                cam_name=cam_name,
                width=args.w,
                height=args.h,
                stride=max(1, int(args.pc_stride)),
                global_cap=int(args.pc_max_points),
                top_cap=int(args.pc_max_points_top),
            )
        if args.pc_backend == "gpu":
            pc_pipeline = Open3dCudaPointCloudPipeline()
        else:
            pc_pipeline = LegacyCompatPointCloudPipeline()

    pc_debug_visibility_cam_set = set()
    pc_debug_visibility_outdir: Optional[Path] = None
    pc_debug_visibility_saved: Dict[str, bool] = {}
    if args.pc and args.pc_debug_visibility:
        requested_debug_cams = args.pc_debug_visibility_cams if args.pc_debug_visibility_cams else sorted(pc_cam_set)
        pc_debug_visibility_cam_set = set([c for c in requested_debug_cams if c in pc_cam_set])
        missing_debug_cams = [c for c in requested_debug_cams if c not in pc_cam_set]
        if missing_debug_cams:
            print(f"[Unified][WARN] PC visibility debug cameras not in active PC camera set (ignored): {missing_debug_cams}")
        if not pc_debug_visibility_cam_set:
            print("[Unified][WARN] PC visibility debug requested but no valid PC cameras were selected.")
        outdir_value = args.pc_debug_visibility_outdir
        if outdir_value:
            pc_debug_visibility_outdir = Path(outdir_value)
        else:
            pc_debug_visibility_outdir = Path.cwd() / f"pc_debug_{time.strftime('%Y%m%d_%H%M%S')}"

    extrinsics = load_camera_extrinsics(extrinsics_path)
    for cam in extrinsics.keys():
        if cam not in cams:
            print(f"[Unified][WARN] Extrinsic provided for '{cam}' but camera is not active.")

    # Scene publisher (teleop visuals + rigid object updates)
    scene_pub = MujocoPublisher(
        model,
        data,
        host=host_ip,
        visible_geoms_groups=args.visible_geoms_groups,
        preferred_xr_name=args.unity_node,
    )

    # VR input
    mq3 = MetaQuest3(args.unity_node)

    # Sensor topic list for NodeInfo
    topic_list = [f"SimPub/Sensors/{cam}/rgbd" for cam in sorted(rgbd_cam_set)]
    if publish_rgb_topic:
        topic_list.extend([f"SimPub/Sensors/{cam}/rgb" for cam in sorted(rgb_cam_set)])
    if args.pc:
        topic_list.extend([f"SimPub/Sensors/{cam}/pc" for cam in sorted(pc_cam_set)])
    topic_list = sorted(list(set(topic_list)))

    sensor_node = SensorNode(
        name="SimPub",
        bind_ip=args.bind_ip,
        host_ip=host_ip,
        service_port=args.service_port,
        topic_port=args.topic_port,
        topic_list=topic_list,
    )
    sensor_node.start()

    renderer = GLFWMjrRenderer(
        model,
        args.w,
        args.h,
        visible=args.show_mujoco_window,
        window_title="mujoco_unified_publisher",
    )
    movable_objects: List[MovableBody] = []
    selected_object_idx = 0
    key_latch: Dict[int, bool] = {}
    next_object_move_time = 0.0
    if args.object_control:
        movable_objects = discover_movable_free_bodies(model, args.object_control_bodies)
        if len(movable_objects) == 0:
            print("[Unified][WARN] Object control enabled but no free-joint objects were found.")
            args.object_control = False
        else:
            print("[Unified] Object control enabled.")
            print(
                "[Unified] Keys: TAB=cycle object, arrows/WASD=XY move, "
                "PageUp/PageDown or Q/E or R/F=Z move"
            )
            print("[Unified] Controllable objects:", [o.body_name for o in movable_objects])
            print(
                f"[Unified] Selected object: {movable_objects[selected_object_idx].body_name} "
                f"(step_xy={args.object_step_xy}, step_z={args.object_step_z})"
            )

    # End-effector selection for teleop IK
    site_name, site_id = find_first_existing(
        model,
        ["ee_site", "grasp_site", "panda_hand_site", "tcp", "tool0"],
        mujoco.mjtObj.mjOBJ_SITE,
    )
    if site_id == -1:
        body_name, body_id = find_first_existing(
            model,
            ["panda_hand", "hand", "eef", "panda_link8"],
            mujoco.mjtObj.mjOBJ_BODY,
        )
        if body_id == -1:
            raise RuntimeError("Could not find EE site/body. Add a site named ee_site to your MJCF.")
        use_site = False
        ee_id = body_id
        print(f"[Unified] Using EE body: {body_name}")
    else:
        use_site = True
        ee_id = site_id
        print(f"[Unified] Using EE site: {site_name}")

    if use_site:
        target_pos = data.site_xpos[ee_id].copy()
        target_quat = data.site_xquat[ee_id].copy()
    else:
        target_pos = data.xpos[ee_id].copy()
        target_quat = data.xquat[ee_id].copy()

    clutch_prev = False
    pos_offset = np.zeros(3, dtype=np.float64)
    quat_offset = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    close_gripper = False

    flip_z = (not args.pc_no_flip_z) or args.pc_flip_z
    table_plane: Optional[Plane] = None
    if args.pc and args.pc_clip_below_table:
        table_plane = resolve_table_plane(
            model=model,
            data=data,
            geom_candidates=args.pc_table_geoms,
            body_name=args.pc_table_body,
            manual_plane_z=args.pc_table_plane_z,
            body_top_offset=args.pc_table_body_top_offset,
        )
        if table_plane is None:
            print(
                "[Unified][WARN] Table plane clipping requested, but plane could not be resolved. "
                "Use --pc_table_plane_z for manual fallback."
            )
        else:
            print(
                f"[Unified] Table clipping plane source={table_plane.source} "
                f"point={np.round(table_plane.point, 4)} normal={np.round(table_plane.normal, 4)}"
            )

    anchor_site_id = -1
    anchor_body_id = -1
    robot_anchor_available = False
    if args.pc and args.pc_anchor_auto_translate:
        if args.pc_anchor_site:
            anchor_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, args.pc_anchor_site)
            if anchor_site_id < 0:
                print(f"[Unified][WARN] Anchor site '{args.pc_anchor_site}' not found, using anchor body fallback.")
        if anchor_site_id < 0:
            anchor_candidates = [args.pc_anchor_body, "link0", "panda_link0", "panda", "base"]
            anchor_name, anchor_body_id = find_first_existing(
                model,
                anchor_candidates,
                mujoco.mjtObj.mjOBJ_BODY,
            )
            if anchor_body_id < 0:
                if args.pc_anchor_source == "robot":
                    print("[Unified][WARN] Robot anchor requested but not found; disabling auto translation correction.")
                    args.pc_anchor_auto_translate = False
                else:
                    print("[Unified][WARN] Robot anchor not found; table anchor fallback may be used.")
            else:
                print(f"[Unified] Anchor body for auto translation: {anchor_name}")
                robot_anchor_available = True
        else:
            print(f"[Unified] Anchor site for auto translation: {args.pc_anchor_site}")
            robot_anchor_available = True

    manual_anchor_world = None
    if args.pc_anchor_world_xyz is not None:
        manual_anchor_world = np.asarray(args.pc_anchor_world_xyz, dtype=np.float64).reshape(3,)
        print(f"[Unified] Manual anchor world point: {np.round(manual_anchor_world, 4)}")

    print("[Unified] XML:", args.xml)
    print("[Unified] Version:", UNIFIED_PUBLISHER_VERSION)
    print("[Unified] Scene host:", host_ip)
    print("[Unified] Visible geom groups:", args.visible_geoms_groups)
    print("[Unified] Cameras:", cams)
    print("[Unified] Sensor REP:", f"tcp://{args.bind_ip}:{args.service_port}")
    print("[Unified] Sensor PUB:", f"tcp://{args.bind_ip}:{args.topic_port}")
    print("[Unified] Standalone RGB topic:", "ON" if publish_rgb_topic else "OFF")
    if publish_rgb_topic:
        print("[Unified] RGB cams:", sorted(list(rgb_cam_set)))
    print("[Unified] RGBD cams:", sorted(list(rgbd_cam_set)))
    print("[Unified] MuJoCo window:", "VISIBLE" if args.show_mujoco_window else "HIDDEN")
    if args.object_control:
        print(
            f"[Unified] Object control: ON ({len(movable_objects)} objects, "
            f"selected={movable_objects[selected_object_idx].body_name})"
        )
    else:
        print("[Unified] Object control: OFF")
    if args.pc:
        print("[Unified] PC cams:", sorted(list(pc_cam_set)))
        print("[Unified] PC backend:", args.pc_backend)
        print(f"[Unified] PC depth range: {args.pc_min_depth} .. {pc_max_depth}")
        print(f"[Unified] PC intrinsics mode: {args.pc_intrinsics_mode}")
        print(
            f"[Unified] PC sampling mode: {args.pc_sampling} "
            f"(stride={args.pc_stride}, target_samples_per_cam={pc_target_sample_budget})"
        )
        print(f"[Unified] PC flips: x={args.pc_flip_x} y={args.pc_flip_y} z={flip_z}")
        print(f"[Unified] PC scale: {args.pc_scale} max_points={args.pc_max_points}")
        if args.pc_max_points_top:
            print(f"[Unified] PC top-camera max_points override: {args.pc_max_points_top}")
        print(f"[Unified] PC declared capacities: {pc_declared_capacity_by_cam}")
        if args.pc_object_only:
            print(
                "[Unified] PC object-only AABB:"
                f" min={np.round(object_bbox_min, 4)} max={np.round(object_bbox_max, 4)}"
            )
        if args.pc_clip_below_table:
            print(
                f"[Unified] PC table filter: margin={args.pc_table_margin} "
                f"clearance={args.pc_table_clearance}"
            )
        if args.pc_debug_visibility:
            print(
                "[Unified] PC visibility debug:"
                f" cams={sorted(list(pc_debug_visibility_cam_set))} "
                f"outdir={pc_debug_visibility_outdir}"
            )
        print(
            f"[Unified] Anchor auto translation: {args.pc_anchor_auto_translate} "
            f"(source={args.pc_anchor_source}, robot_available={robot_anchor_available})"
        )
        print(f"[Unified] Anchor update mode: {args.pc_anchor_update_mode}")
        if extrinsics_path:
            print(f"[Unified] Extrinsics file: {extrinsics_path}")
        else:
            print(f"[Unified] Extrinsics profile: {args.cam_extrinsics_profile} (none loaded)")
    else:
        print("[Unified] Point cloud: OFF")

    control_dt = 1.0 / max(args.control_hz, 1e-6)
    sensor_period = 1.0 / max(args.fps, 1e-6)
    next_sensor_t = time.time()
    frame_idx = 0
    anchor_corr_by_cam: Dict[str, np.ndarray] = {
        cam: np.zeros(3, dtype=np.float64) for cam in pc_cam_set
    }
    anchor_corr_global = np.zeros(3, dtype=np.float64)
    first_rgb_logged: Dict[str, bool] = {}
    first_rgbd_logged: Dict[str, bool] = {}
    first_pc_logged: Dict[str, bool] = {}

    try:
        while True:
            loop_t0 = time.time()
            renderer.poll_events()
            if args.show_mujoco_window and (not renderer.is_window_open()):
                print("[Unified] MuJoCo window was closed; stopping.")
                break

            if args.object_control and len(movable_objects) > 0:
                if key_pressed_once(renderer, key_latch, glfw.KEY_TAB):
                    selected_object_idx = (selected_object_idx + 1) % len(movable_objects)
                    print(
                        "[Unified][ObjectControl] Selected:",
                        movable_objects[selected_object_idx].body_name,
                    )

                move_delta = np.zeros(3, dtype=np.float64)
                if key_down_any(renderer, [glfw.KEY_UP, glfw.KEY_W]):
                    move_delta[0] -= float(args.object_step_xy)
                if key_down_any(renderer, [glfw.KEY_DOWN, glfw.KEY_S]):
                    move_delta[0] += float(args.object_step_xy)
                if key_down_any(renderer, [glfw.KEY_LEFT, glfw.KEY_A]):
                    move_delta[1] -= float(args.object_step_xy)
                if key_down_any(renderer, [glfw.KEY_RIGHT, glfw.KEY_D]):
                    move_delta[1] += float(args.object_step_xy)
                if key_down_any(renderer, [glfw.KEY_PAGE_UP, glfw.KEY_Q]):
                    move_delta[2] += float(args.object_step_z)
                if key_down_any(renderer, [glfw.KEY_PAGE_DOWN, glfw.KEY_E]):
                    move_delta[2] -= float(args.object_step_z)
                if key_down_any(renderer, [glfw.KEY_R]):
                    move_delta[2] += float(args.object_step_z)
                if key_down_any(renderer, [glfw.KEY_F]):
                    move_delta[2] -= float(args.object_step_z)

                if np.any(np.abs(move_delta) > 0.0):
                    now_move = time.time()
                    if now_move >= next_object_move_time:
                        selected = movable_objects[selected_object_idx]
                        apply_movable_translation(model, data, selected, move_delta)
                        next_object_move_time = now_move + (1.0 / max(1e-6, float(args.object_move_rate_hz)))
                        pos_now = np.round(data.qpos[selected.qpos_adr : selected.qpos_adr + 3], 4)
                        print(
                            f"[Unified][ObjectControl] {selected.body_name} moved by "
                            f"{np.round(move_delta, 4)} -> pos={pos_now}"
                        )

            # XR controller input
            inp = mq3.get_controller_data()
            if inp and "right" in inp:
                right = inp["right"]
                u_pos = np.array(right["pos"], dtype=np.float64)
                u_rot = np.array(right["rot"], dtype=np.float64)  # xyzw
                clutch = (
                    right.get("hand_trigger", 0.0) > 0.6
                    or right.get("grip", 0.0) > 0.6
                    or bool(right.get("thumbstick", False))
                )
                close_gripper = right.get("index_trigger", 0.0) > 0.6

                mj_pos = unity_to_mj_pos(u_pos)
                mj_quat = unity_to_mj_quat(u_rot)

                if clutch and not clutch_prev:
                    mujoco.mj_forward(model, data)
                    if use_site:
                        ee_pos = data.site_xpos[ee_id].copy()
                        ee_quat = data.site_xquat[ee_id].copy()
                    else:
                        ee_pos = data.xpos[ee_id].copy()
                        ee_quat = data.xquat[ee_id].copy()
                    pos_offset = ee_pos - mj_pos
                    quat_offset = quat_mul(ee_quat, quat_conj(mj_quat))

                clutch_prev = clutch
                if clutch:
                    target_pos = mj_pos + pos_offset
                    target_quat = quat_mul(quat_offset, mj_quat)

            # IK
            mujoco.mj_forward(model, data)
            if use_site:
                ee_pos = data.site_xpos[ee_id].copy()
                ee_quat = data.site_xquat[ee_id].copy()
                Jp = np.zeros((3, model.nv))
                Jr = np.zeros((3, model.nv))
                mujoco.mj_jacSite(model, data, Jp, Jr, ee_id)
            else:
                ee_pos = data.xpos[ee_id].copy()
                ee_quat = data.xquat[ee_id].copy()
                Jp = np.zeros((3, model.nv))
                Jr = np.zeros((3, model.nv))
                mujoco.mj_jacBody(model, data, Jp, Jr, ee_id)

            pos_err = target_pos - ee_pos
            q_err = quat_mul(target_quat, quat_conj(ee_quat))
            rot_err = quat_to_rotvec(q_err)
            e = np.hstack([pos_err, rot_err])

            J = np.vstack([Jp, Jr])
            dq = damped_ls(J, e, lam=0.08)

            # IMPORTANT: use named joint/dof mapping (not qpos[0:7]) because scenes can
            # have free-joint objects before robot joints in state ordering.
            for i in range(7):
                qadr = int(arm_qpos_idx[i])
                dadr = int(arm_dof_idx[i])
                if qadr < 0 or dadr < 0:
                    continue
                q_des = float(data.qpos[qadr] + (0.25 * dq[dadr]))
                cadr = int(arm_ctrl_idx[i])
                if 0 <= cadr < data.ctrl.shape[0]:
                    data.ctrl[cadr] = q_des
                elif i < data.ctrl.shape[0]:
                    data.ctrl[i] = q_des
            if data.ctrl.shape[0] > 7:
                data.ctrl[7] = 0.0 if close_gripper else 0.04

            mujoco.mj_step(model, data)

            # Sensor publishing
            now = time.time()
            if now >= next_sensor_t:
                timestamp = now
                if args.pc and args.pc_clip_below_table and table_plane is None:
                    table_plane = resolve_table_plane(
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
                            anchor_world_mj = get_anchor_world_mj(
                                data=data,
                                anchor_site_id=anchor_site_id,
                                anchor_body_id=anchor_body_id,
                            )
                        if anchor_world_mj is None and args.pc_anchor_source in ("auto", "table"):
                            if table_plane is not None:
                                anchor_world_mj = np.asarray(table_plane.point, dtype=np.float64)

                for cam_name in cams:
                    rgb, depth_m = renderer.render(data, cam_name)
                    publish_rgb = publish_rgb_topic and (cam_name in rgb_cam_set)
                    publish_rgbd = cam_name in rgbd_cam_set
                    needs_rgb_jpg = publish_rgb or publish_rgbd
                    rgb_jpg = encode_rgb_jpg(rgb, quality=args.jpg_quality) if needs_rgb_jpg else None

                    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
                    fovy = float(model.cam_fovy[cam_id])
                    fx, fy, cx, cy = compute_intrinsics(
                        args.w,
                        args.h,
                        fovy,
                        mode=args.pc_intrinsics_mode,
                    )

                    calib = extrinsics.get(cam_name)
                    cam_t_mj, cam_R_mj = get_calibrated_cam_pose(data, cam_id, calib)
                    cam_q_mj = rotmat_to_quat_wxyz(cam_R_mj)
                    cam_t_unity = mj_to_unity_pos(cam_t_mj.astype(np.float32))
                    cam_q_unity = mj_to_unity_quat_xyzw(cam_q_mj)

                    pc7: Optional[np.ndarray] = None
                    if args.pc_anchor_update_mode == "global":
                        cam_anchor_corr = anchor_corr_global.copy()
                    else:
                        cam_anchor_corr = anchor_corr_by_cam.get(cam_name, anchor_corr_global.copy())
                    anchor_corr_updated = False
                    removed_below_plane = 0
                    removed_outside_object_box = 0
                    if args.pc and cam_name in pc_cam_set:
                        if args.pc_anchor_auto_translate and anchor_world_mj is not None:
                            corr_obs = estimate_anchor_translation_correction(
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
                                    anchor_corr_global = ((1.0 - alpha) * anchor_corr_global) + (alpha * corr_obs)
                                    cam_anchor_corr = anchor_corr_global.copy()
                                else:
                                    cam_anchor_corr = ((1.0 - alpha) * cam_anchor_corr) + (alpha * corr_obs)
                                    anchor_corr_by_cam[cam_name] = cam_anchor_corr
                                    anchor_corr_global = ((1.0 - alpha) * anchor_corr_global) + (alpha * corr_obs)
                                anchor_corr_updated = True

                        cam_clearance = float(args.pc_table_clearance)
                        if cam_name == "top" and args.pc_top_table_clearance is not None:
                            cam_clearance = float(args.pc_top_table_clearance)

                        debug_visibility_requested = (
                            bool(args.pc_debug_visibility)
                            and cam_name in pc_debug_visibility_cam_set
                            and not pc_debug_visibility_saved.get(cam_name, False)
                        )
                        pc_frame = pc_pipeline.build_frame(
                            cam_name=cam_name,
                            depth_m=depth_m,
                            rgb_u8=rgb,
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
                            cam_t_mj=cam_t_mj,
                            cam_R_mj=cam_R_mj,
                            cam_anchor_corr=cam_anchor_corr,
                            clip_below_table=bool(args.pc_clip_below_table),
                            table_plane=table_plane,
                            table_margin=float(args.pc_table_margin),
                            table_clearance=cam_clearance,
                            object_only=bool(args.pc_object_only),
                            object_bbox_min=object_bbox_min,
                            object_bbox_max=object_bbox_max,
                            declared_capacity=pc_declared_capacity_by_cam[cam_name],
                            debug_visibility=debug_visibility_requested,
                        )
                        removed_below_plane = pc_frame.removed_below_plane
                        removed_outside_object_box = pc_frame.removed_outside_object_box

                        sensor_node.publish(f"SimPub/Sensors/{cam_name}/pc", pc_frame.payload)

                        if not first_pc_logged.get(cam_name, False):
                            print(
                                f"[Unified] First PC publish: cam={cam_name} "
                                f"topic=SimPub/Sensors/{cam_name}/pc mode={args.pc_sampling} "
                                f"stride={args.pc_stride} target_samples={pc_target_sample_budget} "
                                f"actual={pc_frame.actual_count} declared_capacity={pc_frame.declared_capacity}"
                            )
                            first_pc_logged[cam_name] = True

                        if debug_visibility_requested and pc_frame.debug_visibility is not None:
                            artifact_info = save_pc_visibility_debug_artifacts(
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
                            pc_debug_visibility_saved[cam_name] = True
                            dbg = pc_frame.debug_visibility
                            if dbg.xyz_world_mj_kept.shape[0] > 0:
                                dbg_min = np.round(dbg.xyz_world_mj_kept.min(axis=0), 4)
                                dbg_max = np.round(dbg.xyz_world_mj_kept.max(axis=0), 4)
                                dbg_centroid = np.round(dbg.xyz_world_mj_kept.mean(axis=0), 4)
                                dbg_hist = _world_x_histogram_counts(dbg.xyz_world_mj_kept[:, 0], bins=6)
                            else:
                                dbg_min = np.array([], dtype=np.float32)
                                dbg_max = np.array([], dtype=np.float32)
                                dbg_centroid = np.array([], dtype=np.float32)
                                dbg_hist = []
                            print(
                                f"[PCDBG] {cam_name} sampled={dbg.sampled_count} "
                                f"depth_valid={dbg.valid_depth_count} kept={dbg.kept_count} "
                                f"rejected_table={dbg.rejected_table_count} "
                                f"rejected_bbox={dbg.rejected_bbox_count}"
                            )
                            print(
                                f"[PCDBG] {cam_name} world_min={dbg_min} "
                                f"world_max={dbg_max} world_centroid={dbg_centroid} "
                                f"x_hist={dbg_hist}"
                            )
                            print(
                                f"[PCDBG] {cam_name} aabb_corners_front={artifact_info['aabb_front_count']}/8 "
                                f"aabb_corners_in_image={artifact_info['aabb_in_image_count']}/8 "
                                f"rgb={artifact_info['rgb_path']} "
                                f"selected={artifact_info['selected_path']} "
                                f"rejected={artifact_info['rejected_path']}"
                            )

                        if args.rgbd_include_pc:
                            pc7 = pc7_array(
                                pc_frame.xyz_unity_m,
                                pc_frame.rgb_u8,
                                size=float(args.pc_size),
                            )

                        if frame_idx % max(1, args.log_every) == 0:
                            if pc_frame.actual_count > 0:
                                mn = np.round(pc_frame.xyz_unity_m.min(axis=0), 4)
                                mx = np.round(pc_frame.xyz_unity_m.max(axis=0), 4)
                                print(
                                    f"[PC] {cam_name} N={pc_frame.actual_count} "
                                    f"declared_capacity={pc_frame.declared_capacity} "
                                    f"removed_below={removed_below_plane} "
                                    f"removed_outside_bbox={removed_outside_object_box} "
                                    f"anchor_corr={np.round(cam_anchor_corr, 4)} "
                                    f"anchor_global={np.round(anchor_corr_global, 4)} "
                                    f"updated={anchor_corr_updated} "
                                    f"bbox_min={mn} bbox_max={mx}"
                                )
                            else:
                                print(
                                    f"[PC] {cam_name} N=0 declared_capacity={pc_frame.declared_capacity} "
                                    f"removed_below={removed_below_plane} "
                                    f"removed_outside_bbox={removed_outside_object_box} "
                                    f"anchor_corr={np.round(cam_anchor_corr, 4)} "
                                    f"anchor_global={np.round(anchor_corr_global, 4)} "
                                    "(depth/plane filters)"
                                )

                    if publish_rgb and rgb_jpg is not None:
                        sensor_node.publish(f"SimPub/Sensors/{cam_name}/rgb", rgb_jpg)
                        if not first_rgb_logged.get(cam_name, False):
                            print(
                                f"[Unified] First RGB publish: cam={cam_name} "
                                f"topic=SimPub/Sensors/{cam_name}/rgb bytes={len(rgb_jpg)}"
                            )
                            first_rgb_logged[cam_name] = True

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
                            "pc_anchor_corr_mj": [float(v) for v in cam_anchor_corr],
                            "pc_anchor_corr_global_mj": [float(v) for v in anchor_corr_global],
                            "pc_anchor_corr_updated": bool(anchor_corr_updated),
                        }
                        blob = build_rgbd_blob(
                            cam_name=cam_name,
                            width=args.w,
                            height=args.h,
                            fovy_deg=fovy,
                            timestamp=timestamp,
                            rgb_jpg=rgb_jpg,
                            depth_f32_m=depth_m,
                            intrinsics={
                                "fx": float(fx),
                                "fy": float(fy),
                                "cx": float(cx),
                                "cy": float(cy),
                            },
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
                        if not first_rgbd_logged.get(cam_name, False):
                            print(
                                f"[Unified] First RGBD publish: cam={cam_name} "
                                f"topic=SimPub/Sensors/{cam_name}/rgbd bytes={len(blob)}"
                            )
                            first_rgbd_logged[cam_name] = True

                while next_sensor_t <= now:
                    next_sensor_t += sensor_period

            elapsed = time.time() - loop_t0
            sleep_t = control_dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)
            frame_idx += 1

    except KeyboardInterrupt:
        print("\n[Unified] Stopping...")
    finally:
        try:
            sensor_node.stop()
        except Exception:
            pass
        try:
            renderer.close()
        except Exception:
            pass
        try:
            scene_pub.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()


# $LAB_XML   = ".\LAB\lab2_boxes_cups_vizCollisionSeparated_v3_groups_named.xml"
# $HOST_IP   = "192.168.0.208"
# $UNITY_NODE = "MQ3-2"



# $LAB_XML   = ".\LAB\lab3_stick_maze_vizCollisionSeparated_v3_groups_named.xml"
# $HOST_IP   = "192.168.0.208"
# $UNITY_NODE = "MQ3-2"



# $LAB_XML   = ".\LAB\lab1_T_stack.xml"
# $HOST_IP   = "192.168.0.208"
# $UNITY_NODE = "MQ3-2"


# # "$LAB_XML | $HOST_IP | $UNITY_NODE"



# # ---------- Edit these 3 ----------
# # $LAB_XML    = ".\LAB\lab2_boxes_cups_vizCollisionSeparated_v3_groups_named.xml"
# # $HOST_IP    = "192.168.0.208"
# # $UNITY_NODE = "Unity-localhost"

# # ---------- Run (from repo root, with venv) ----------
# # .\.venv\Scripts\python.exe .\SimPublisher\sii\franka_quest_unified_publisher.py `
# #   --xml "$LAB_XML" `
# #   --host "$HOST_IP" `
# #   --unity_node "$UNITY_NODE" `
# #   --bind_ip 0.0.0.0 `
# #   --visible_geoms_groups 2 `
# #   --fps 15 --w 640 --h 480 `
# #   --cams wrist top right left `
# #   --pc --pc_cams top right left `
# #   --pc_stride 2 --pc_max_points 40000 `
# #   --pc_min_depth 0.08 --pc_max_depth 2.8 `
# #   --pc_intrinsics_mode mujoco `
# #   --pc_clip_below_table --pc_table_clearance 0.02 --pc_top_table_clearance 0.035 `
# #   --pc_object_only --pc_object_bbox_min 0.30 -0.45 0.35 --pc_object_bbox_max 1.00 0.45 0.92 `
# #   --cam_extrinsics_profile lab_standard `
# #   --pc_no_anchor_auto_translate `
# #   --robot_start_preset reverse_u `
# #   --show_mujoco_window `
# #   --object_control `
# #   --object_step_xy 0.01 --object_step_z 0.008 --object_move_rate_hz 10
  
  

# # $ADB="$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"
# # if (!(Test-Path $ADB)) { $ADB="C:\Program Files\Unity\Hub\Editor\6000.0.33f1\Editor\Data\PlaybackEngines\AndroidPlayer\SDK\platform-tools\adb.exe" }

# # $PKG="com.anonymous.SIIMetaQuest3"
# # $OUT=Join-Path (Get-Location) ("quest_log_{0}.txt" -f (Get-Date -Format yyyyMMdd_HHmmss))

# # & $ADB logcat -c
# # & $ADB shell am force-stop $PKG
# # & $ADB shell monkey -p $PKG -c android.intent.category.LAUNCHER 1 | Out-Null
# # Start-Sleep 3

# # # leave running 20-30s, then Ctrl+C
# # & $ADB logcat -v time | Tee-Object -FilePath $OUT






# # Get-Item $OUT
# # Select-String -Path $OUT -Pattern "Unity-localhost|offline for 5s|TimeoutError|SimPubRgbdSubscriber|Panel diagnostics|Panel rescued|Anchor-applied"


# Right-camera coverage validation baseline:
# .\.venv\Scripts\python.exe .\SimPublisher\sii\franka_quest_unified_publisher.py `
#   --xml "$LAB_XML" `
#   --host "$HOST_IP" `
#   --unity_node "$UNITY_NODE" `
#   --bind_ip 0.0.0.0 `
#   --visible_geoms_groups 2 `
#   --fps 30 --w 960 --h 720 `
#   --cams wrist top right left `
#   --pc --pc_cams top right left `
#   --pc_sampling stable_random `
#   --pc_stride  8 --pc_max_points 40000 `
#   --pc_min_depth 0.08 --pc_max_depth 2.8 `
#   --pc_intrinsics_mode mujoco `
#   --pc_clip_below_table --pc_table_clearance 0.02 --pc_top_table_clearance 0.035 `
#   --pc_object_only --pc_object_bbox_min 0.30 -0.45 0.35 --pc_object_bbox_max 1.00 0.45 0.92 `
#   --cam_extrinsics_profile lab_standard `
#   --pc_no_anchor_auto_translate `
#   --robot_start_preset reverse_u `
#   --show_mujoco_window `
#   --object_control `
#   --object_step_xy 0.01 --object_step_z 0.008 --object_move_rate_hz 10

# Manual latency check:
# - Use the current publisher command unchanged.
# - Solo pass first: 5-10 minutes, no logging, just feel in VR.
# - Hold the clutch and do 10 short wrist snaps plus 10 stop-start motions.
# - Do 10 quick gripper squeeze/release cycles.
# - Put a bright object in the wrist camera view and move the wrist past it quickly.
# - Use Tab to cycle object_control targets and W/A/S/D/Q/E for discrete step moves.
# - Pass if robot motion feels coupled and the wrist panel updates during motion.
# - See manual_latency_check.md for the full checklist.

# Proof run to isolate filtering from sampling/rendering:
# .\.venv\Scripts\python.exe .\SimPublisher\sii\franka_quest_unified_publisher.py `
#   --xml "$LAB_XML" `
#   --host "$HOST_IP" `
#   --unity_node "$UNITY_NODE" `
#   --bind_ip 0.0.0.0 `
#   --visible_geoms_groups 2 `
#   --fps 30 --w 960 --h 720 `
#   --cams wrist top right left `
#   --pc --pc_cams top right left `
#   --pc_sampling stable_random `
#   --pc_stride 8 --pc_max_points 40000 `
#   --pc_min_depth 0.08 --pc_max_depth 2.8 `
#   --pc_intrinsics_mode mujoco `
#   --pc_clip_below_table --pc_table_clearance 0.02 --pc_top_table_clearance 0.035 `
#   --pc_debug_visibility `
#   --pc_debug_visibility_cams right left top `
#   --cam_extrinsics_profile lab_standard `
#   --pc_no_anchor_auto_translate `
#   --robot_start_preset reverse_u `
#   --show_mujoco_window `
#   --object_control `
#   --object_step_xy 0.01 --object_step_z 0.008 --object_move_rate_hz 10


# $ADB="$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"
# if (!(Test-Path $ADB)) {
#   $ADB="C:\Program Files\Unity\Hub\Editor\6000.0.33f1\Editor\Data\PlaybackEngines\AndroidPlayer\SDK\platform-tools\adb.exe"
# }

# $PKG="com.anonymous.SIIMetaQuest3"
# $STAMP=Get-Date -Format yyyyMMdd_HHmmss
# $OUT=Join-Path (Get-Location) ("quest_pc_log_{0}.txt" -f $STAMP)

# $PATTERN="Unity-localhost|Unity|GpuMergedPointCloudBootstrap|PointCloudTest|GpuPointCloudSubscriber|SimPubClient|SimPubRgbdSubscriber|First frame|First draw|Connected tcp://|Waiting for SimScene|Dropped|blocked|renderer|pointcloud|PointCloudAnchor"

# & $ADB logcat -c
# & $ADB shell am force-stop $PKG
# Start-Sleep 1
# & $ADB shell monkey -p $PKG -c android.intent.category.LAUNCHER 1 | Out-Null
# Start-Sleep 4

# Write-Host "App launched. Put headset on, wait 20-30s while looking for the point cloud, then press Ctrl+C."
# & $ADB logcat -v time | Tee-Object -FilePath $OUT

# # After Ctrl+C:
# Get-Item $OUT
# Write-Host "`n==== Filtered point-cloud lines ===="
# Select-String -Path $OUT -Pattern $PATTERN | Set-Content ($OUT -replace "\.txt$","_filtered.txt")
# Get-Content ($OUT -replace "\.txt$","_filtered.txt")



# .\.venv\Scripts\python.exe .\SimPublisher\sii\franka_quest_unified_publisher.py `
#   --xml "$LAB_XML" `
#   --host "$HOST_IP" `
#   --unity_node "$UNITY_NODE" `
#   --bind_ip 0.0.0.0 `
#   --visible_geoms_groups 2 `
#   --fps 30 --w 960 --h 720 `
#   --cams wrist top right left `
#   --pc --pc_cams top right left `
#   --pc_sampling stable_random `
#   --pc_stride 8 --pc_max_points 40000 `
#   --pc_min_depth 0.08 --pc_max_depth 2.8 `
#   --pc_intrinsics_mode mujoco `
#   --pc_clip_below_table --pc_table_clearance 0.02 --pc_top_table_clearance 0.035 `
#   --pc_object_only --pc_object_bbox_min 0.30 -0.45 0.35 --pc_object_bbox_max 1.00 0.45 0.92 `
#   --cam_extrinsics_profile lab_standard `
#   --pc_no_anchor_auto_translate `
#   --robot_start_preset reverse_u `
#   --show_mujoco_window `
#   --object_control `
#   --object_step_xy 0.01 --object_step_z 0.008 --object_move_rate_hz 10

# .\.venv\Scripts\python.exe .\SimPublisher\sii\franka_quest_unified_publisher.py `
#   --xml "$LAB_XML" `
#   --host "$HOST_IP" `
#   --unity_node "$UNITY_NODE" `
#   --bind_ip 0.0.0.0 `
#   --visible_geoms_groups 2 `
#   --fps 30 --w 960 --h 720 `
#   --cams wrist top right left `
#   --rgb_cams wrist `
#   --pc --pc_cams top right left `
#   --pc_sampling stable_random `
#   --pc_stride 8 --pc_max_points 50000 `
#   --pc_min_depth 0.08 --pc_max_depth 2.8 `
#   --pc_intrinsics_mode mujoco `
#   --pc_clip_below_table --pc_table_clearance 0.02 --pc_top_table_clearance 0.035 `
#   --pc_object_only --pc_object_bbox_min 0.30 -0.45 0.35 --pc_object_bbox_max 1.00 0.45 0.92 `
#   --cam_extrinsics_profile lab_standard `
#   --pc_no_anchor_auto_translate `
#   --robot_start_preset reverse_u `
#   --show_mujoco_window `
#   --object_control `
#   --object_step_xy 0.01 --object_step_z 0.008 --object_move_rate_hz 10
