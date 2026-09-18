from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from GPU.GpuPointCloudBuffers import GpuPointCloudBufferPool
from GPU.GpuPointCloudContract import encode_frame_into, stage_frame_data

try:
    import open3d as _o3d
except Exception:
    _o3d = None


_SUPPORTED_SAMPLING_MODES = {"grid", "stable_random"}
_SAMPLED_PIXEL_GRID_CACHE: dict[tuple[str, int, int, int, str], tuple[np.ndarray, np.ndarray]] = {}
_SAMPLED_PIXEL_CANDIDATE_CACHE: dict[tuple[str, int, int, int, str, int], tuple[np.ndarray, np.ndarray]] = {}
_STABLE_RANDOM_CANDIDATE_COUNT = 4


def compute_intrinsics(
    width: int,
    height: int,
    fovy_deg: float,
    mode: str = "mujoco",
) -> tuple[float, float, float, float]:
    fovy = np.deg2rad(float(fovy_deg))
    fy = 0.5 * float(height) / np.tan(0.5 * fovy)
    fx = fy * (float(width) / float(height)) if mode == "aspect" else fy
    cx = (float(width) - 1.0) * 0.5
    cy = (float(height) - 1.0) * 0.5
    return float(fx), float(fy), float(cx), float(cy)


def sampled_point_capacity(width: int, height: int, stride: int) -> int:
    stride = max(1, int(stride))
    sx = (int(width) + stride - 1) // stride
    sy = (int(height) + stride - 1) // stride
    return int(sx * sy)


def normalize_sampling_mode(sampling_mode: str) -> str:
    mode = str(sampling_mode or "grid").strip().lower()
    if mode not in _SUPPORTED_SAMPLING_MODES:
        raise ValueError(
            f"Unsupported point-cloud sampling mode '{sampling_mode}'. "
            f"Expected one of {sorted(_SUPPORTED_SAMPLING_MODES)}."
        )
    return mode


def _stable_hash_u32(text: str) -> int:
    h = 2166136261
    for byte in text.encode("utf-8"):
        h ^= byte
        h = (h * 16777619) & 0xFFFFFFFF
    return h or 1


def build_sampled_pixel_grids_numpy(
    *,
    cam_name: str,
    width: int,
    height: int,
    stride: int,
    sampling_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    stride = max(1, int(stride))
    mode = normalize_sampling_mode(sampling_mode)
    key = (str(cam_name), int(width), int(height), int(stride), mode)
    cached = _SAMPLED_PIXEL_GRID_CACHE.get(key)
    if cached is not None:
        return cached

    if mode == "grid":
        yy, xx = np.mgrid[0:height:stride, 0:width:stride]
        cached = (
            np.ascontiguousarray(yy.astype(np.int32)),
            np.ascontiguousarray(xx.astype(np.int32)),
        )
        _SAMPLED_PIXEL_GRID_CACHE[key] = cached
        return cached

    ys = np.arange(0, int(height), stride, dtype=np.int32)
    xs = np.arange(0, int(width), stride, dtype=np.int32)
    yy = np.empty((ys.shape[0], xs.shape[0]), dtype=np.int32)
    xx = np.empty_like(yy)

    rng = np.random.default_rng(
        _stable_hash_u32(f"{cam_name}|{width}|{height}|{stride}|{mode}")
    )

    for iy, y0 in enumerate(ys):
        cell_h = min(stride, int(height) - int(y0))
        for ix, x0 in enumerate(xs):
            cell_w = min(stride, int(width) - int(x0))
            yy[iy, ix] = int(y0) + int(rng.integers(0, cell_h))
            xx[iy, ix] = int(x0) + int(rng.integers(0, cell_w))

    cached = (yy, xx)
    _SAMPLED_PIXEL_GRID_CACHE[key] = cached
    return cached


def build_sampled_pixel_candidate_grids_numpy(
    *,
    cam_name: str,
    width: int,
    height: int,
    stride: int,
    sampling_mode: str,
    candidate_count: int = _STABLE_RANDOM_CANDIDATE_COUNT,
) -> tuple[np.ndarray, np.ndarray]:
    stride = max(1, int(stride))
    mode = normalize_sampling_mode(sampling_mode)
    count = max(1, int(candidate_count))
    key = (str(cam_name), int(width), int(height), int(stride), mode, count)
    cached = _SAMPLED_PIXEL_CANDIDATE_CACHE.get(key)
    if cached is not None:
        return cached

    ys = np.arange(0, int(height), stride, dtype=np.int32)
    xs = np.arange(0, int(width), stride, dtype=np.int32)
    yy = np.empty((count, ys.shape[0], xs.shape[0]), dtype=np.int32)
    xx = np.empty_like(yy)

    rng = np.random.default_rng(
        _stable_hash_u32(f"{cam_name}|{width}|{height}|{stride}|{mode}|candidates={count}")
    )

    for iy, y0 in enumerate(ys):
        cell_h = min(stride, int(height) - int(y0))
        for ix, x0 in enumerate(xs):
            cell_w = min(stride, int(width) - int(x0))
            yy[:, iy, ix] = int(y0) + rng.integers(0, cell_h, size=count, dtype=np.int32)
            xx[:, iy, ix] = int(x0) + rng.integers(0, cell_w, size=count, dtype=np.int32)

    cached = (yy, xx)
    _SAMPLED_PIXEL_CANDIDATE_CACHE[key] = cached
    return cached


def select_sampled_pixel_grids_numpy(
    *,
    cam_name: str,
    depth_m: np.ndarray,
    width: int,
    height: int,
    stride: int,
    sampling_mode: str,
    min_depth: float,
    max_depth: Optional[float],
) -> tuple[np.ndarray, np.ndarray]:
    mode = normalize_sampling_mode(sampling_mode)
    if mode == "grid":
        return build_sampled_pixel_grids_numpy(
            cam_name=cam_name,
            width=width,
            height=height,
            stride=stride,
            sampling_mode=mode,
        )

    yy_candidates, xx_candidates = build_sampled_pixel_candidate_grids_numpy(
        cam_name=cam_name,
        width=width,
        height=height,
        stride=stride,
        sampling_mode=mode,
    )
    z_candidates = depth_m[yy_candidates, xx_candidates]
    valid = np.isfinite(z_candidates) & (z_candidates > float(min_depth))
    if max_depth is not None:
        valid &= z_candidates < float(max_depth)

    ranked_depth = np.where(valid, z_candidates, np.inf)
    best_idx = np.argmin(ranked_depth, axis=0)
    yy = np.take_along_axis(yy_candidates, best_idx[None, ...], axis=0)[0]
    xx = np.take_along_axis(xx_candidates, best_idx[None, ...], axis=0)[0]
    return yy, xx


@dataclass
class SampledDepthPointCloudNumpy:
    sampled_yy: np.ndarray
    sampled_xx: np.ndarray
    valid_mask: np.ndarray
    xyz_cam_valid: np.ndarray
    rgb_valid: np.ndarray


def sample_depth_to_pointcloud_numpy(
    *,
    cam_name: str,
    depth_m: np.ndarray,
    rgb_u8: np.ndarray,
    width: int,
    height: int,
    fovy_deg: float,
    intrinsics_mode: str,
    stride: int,
    min_depth: float,
    max_depth: Optional[float],
    flip_y: bool,
    flip_x: bool,
    sampling_mode: str = "grid",
) -> SampledDepthPointCloudNumpy:
    fx, fy, cx, cy = compute_intrinsics(width, height, fovy_deg, mode=intrinsics_mode)
    yy, xx = select_sampled_pixel_grids_numpy(
        cam_name=cam_name,
        depth_m=depth_m,
        width=width,
        height=height,
        stride=stride,
        sampling_mode=sampling_mode,
        min_depth=min_depth,
        max_depth=max_depth,
    )
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
    return SampledDepthPointCloudNumpy(
        sampled_yy=np.ascontiguousarray(yy.astype(np.int32)),
        sampled_xx=np.ascontiguousarray(xx.astype(np.int32)),
        valid_mask=np.ascontiguousarray(valid.astype(bool)),
        xyz_cam_valid=xyz,
        rgb_valid=rgb,
    )


def depth_to_pointcloud_xyzrgb_numpy(
    *,
    cam_name: str,
    depth_m: np.ndarray,
    rgb_u8: np.ndarray,
    width: int,
    height: int,
    fovy_deg: float,
    intrinsics_mode: str,
    stride: int,
    min_depth: float,
    max_depth: Optional[float],
    flip_y: bool,
    flip_x: bool,
    sampling_mode: str = "grid",
) -> tuple[np.ndarray, np.ndarray]:
    sampled = sample_depth_to_pointcloud_numpy(
        cam_name=cam_name,
        depth_m=depth_m,
        rgb_u8=rgb_u8,
        width=width,
        height=height,
        fovy_deg=fovy_deg,
        intrinsics_mode=intrinsics_mode,
        stride=stride,
        min_depth=min_depth,
        max_depth=max_depth,
        flip_y=flip_y,
        flip_x=flip_x,
        sampling_mode=sampling_mode,
    )
    return sampled.xyz_cam_valid, sampled.rgb_valid


def compute_keep_mask_above_plane_numpy(
    xyz_world_mj: np.ndarray,
    table_plane,
    margin: float,
    clearance: float,
) -> np.ndarray:
    if table_plane is None or xyz_world_mj.size == 0:
        return np.ones((xyz_world_mj.shape[0],), dtype=bool)
    plane_point = np.asarray(table_plane.point, dtype=np.float32).reshape(1, 3)
    plane_normal = np.asarray(table_plane.normal, dtype=np.float32).reshape(3,)
    threshold = float(margin) + float(clearance)
    signed = (xyz_world_mj - plane_point) @ plane_normal.reshape(3, 1)
    return signed.reshape(-1) > threshold


def compute_keep_mask_inside_aabb_numpy(
    xyz_world_mj: np.ndarray,
    bbox_min,
    bbox_max,
) -> np.ndarray:
    if xyz_world_mj.size == 0:
        return np.ones((xyz_world_mj.shape[0],), dtype=bool)
    lo = np.asarray(bbox_min, dtype=np.float32).reshape(1, 3)
    hi = np.asarray(bbox_max, dtype=np.float32).reshape(1, 3)
    return np.all((xyz_world_mj >= lo) & (xyz_world_mj <= hi), axis=1)


def clip_points_below_plane_numpy(
    xyz_world_mj: np.ndarray,
    rgb_u8: np.ndarray,
    table_plane,
    margin: float,
    clearance: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    if table_plane is None or xyz_world_mj.size == 0:
        return xyz_world_mj, rgb_u8, 0
    keep = compute_keep_mask_above_plane_numpy(
        xyz_world_mj,
        table_plane,
        margin=float(margin),
        clearance=float(clearance),
    )
    removed = int(keep.shape[0] - np.count_nonzero(keep))
    return xyz_world_mj[keep], rgb_u8[keep], removed


def clip_points_outside_aabb_numpy(
    xyz_world_mj: np.ndarray,
    rgb_u8: np.ndarray,
    bbox_min,
    bbox_max,
) -> tuple[np.ndarray, np.ndarray, int]:
    if xyz_world_mj.size == 0:
        return xyz_world_mj, rgb_u8, 0
    keep = compute_keep_mask_inside_aabb_numpy(
        xyz_world_mj,
        bbox_min,
        bbox_max,
    )
    removed = int(keep.shape[0] - np.count_nonzero(keep))
    return xyz_world_mj[keep], rgb_u8[keep], removed


def mujoco_to_unity_xyz(points_mj: np.ndarray) -> np.ndarray:
    if points_mj.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    return np.stack(
        [-points_mj[:, 1], points_mj[:, 2], points_mj[:, 0]],
        axis=1,
    ).astype(np.float32)


@dataclass
class PointCloudFrameOutput:
    payload: memoryview
    xyz_world_mj: np.ndarray
    xyz_unity_m: np.ndarray
    rgb_u8: np.ndarray
    declared_capacity: int
    actual_count: int
    removed_below_plane: int
    removed_outside_object_box: int
    debug_visibility: Optional["PointCloudVisibilityDebug"] = None


@dataclass
class PointCloudVisibilityDebug:
    sampled_pixels_yx: np.ndarray
    valid_depth_pixels_yx: np.ndarray
    kept_pixels_yx: np.ndarray
    rejected_table_pixels_yx: np.ndarray
    rejected_bbox_pixels_yx: np.ndarray
    xyz_world_mj_kept: np.ndarray
    sampled_count: int
    valid_depth_count: int
    kept_count: int
    rejected_table_count: int
    rejected_bbox_count: int


class _BasePointCloudPipeline:
    backend_name = "base"

    def __init__(self, buffer_pool: Optional[GpuPointCloudBufferPool] = None):
        self.buffer_pool = buffer_pool or GpuPointCloudBufferPool()

    def _pack_output(
        self,
        *,
        cam_name: str,
        declared_capacity: int,
        xyz_world_mj: np.ndarray,
        rgb_u8: np.ndarray,
        removed_below_plane: int,
        removed_outside_object_box: int,
        debug_visibility: Optional[PointCloudVisibilityDebug] = None,
    ) -> PointCloudFrameOutput:
        xyz_world = np.ascontiguousarray(xyz_world_mj, dtype=np.float32)
        rgb = np.ascontiguousarray(rgb_u8, dtype=np.uint8)
        xyz_unity = mujoco_to_unity_xyz(xyz_world)

        slot = self.buffer_pool.ensure_slot(cam_name, declared_capacity)
        actual_count = stage_frame_data(slot, xyz_unity, rgb)
        payload = encode_frame_into(slot, declared_capacity, actual_count)

        return PointCloudFrameOutput(
            payload=payload,
            xyz_world_mj=xyz_world,
            xyz_unity_m=xyz_unity,
            rgb_u8=rgb,
            declared_capacity=int(declared_capacity),
            actual_count=int(actual_count),
            removed_below_plane=int(removed_below_plane),
            removed_outside_object_box=int(removed_outside_object_box),
            debug_visibility=debug_visibility,
        )


class LegacyCompatPointCloudPipeline(_BasePointCloudPipeline):
    backend_name = "legacy"

    def build_frame(
        self,
        *,
        cam_name: str,
        depth_m: np.ndarray,
        rgb_u8: np.ndarray,
        width: int,
        height: int,
        fovy_deg: float,
        intrinsics_mode: str,
        stride: int,
        min_depth: float,
        max_depth: Optional[float],
        flip_x: bool,
        flip_y: bool,
        flip_z: bool,
        pc_scale: float,
        cam_t_mj: np.ndarray,
        cam_R_mj: np.ndarray,
        cam_anchor_corr: np.ndarray,
        clip_below_table: bool,
        table_plane,
        table_margin: float,
        table_clearance: float,
        object_only: bool,
        object_bbox_min,
        object_bbox_max,
        declared_capacity: int,
        sampling_mode: str = "grid",
        debug_visibility: bool = False,
    ) -> PointCloudFrameOutput:
        sampled = sample_depth_to_pointcloud_numpy(
            cam_name=cam_name,
            depth_m=depth_m,
            rgb_u8=rgb_u8,
            width=width,
            height=height,
            fovy_deg=fovy_deg,
            intrinsics_mode=intrinsics_mode,
            stride=max(1, int(stride)),
            sampling_mode=sampling_mode,
            min_depth=float(min_depth),
            max_depth=float(max_depth) if max_depth is not None else None,
            flip_y=flip_y,
            flip_x=flip_x,
        )
        xyz_cam = sampled.xyz_cam_valid.copy()
        rgb_pc = sampled.rgb_valid.copy()

        if flip_z and xyz_cam.shape[0] > 0:
            xyz_cam[:, 2] *= -1.0
        if pc_scale != 1.0 and xyz_cam.shape[0] > 0:
            xyz_cam *= float(pc_scale)

        if xyz_cam.shape[0] > 0:
            xyz_world_mj = (
                (xyz_cam @ np.asarray(cam_R_mj, dtype=np.float32).T)
                + np.asarray(cam_t_mj, dtype=np.float32).reshape(1, 3)
                + np.asarray(cam_anchor_corr, dtype=np.float32).reshape(1, 3)
            )
        else:
            xyz_world_mj = np.zeros((0, 3), dtype=np.float32)

        debug_visibility_meta: Optional[PointCloudVisibilityDebug] = None
        removed_below_plane = 0
        removed_outside_object_box = 0
        if debug_visibility:
            sampled_pixels_yx = np.stack(
                [sampled.sampled_yy.reshape(-1), sampled.sampled_xx.reshape(-1)],
                axis=1,
            ).astype(np.int32, copy=False)
            valid_depth_pixels_yx = np.stack(
                [sampled.sampled_yy[sampled.valid_mask], sampled.sampled_xx[sampled.valid_mask]],
                axis=1,
            ).astype(np.int32, copy=False)

            table_keep = np.ones((xyz_world_mj.shape[0],), dtype=bool)
            if clip_below_table:
                table_keep = compute_keep_mask_above_plane_numpy(
                    xyz_world_mj,
                    table_plane,
                    margin=float(table_margin),
                    clearance=float(table_clearance),
                )
            removed_below_plane = int(table_keep.shape[0] - np.count_nonzero(table_keep))
            rejected_table_pixels_yx = valid_depth_pixels_yx[~table_keep]

            xyz_after_table = xyz_world_mj[table_keep]
            rgb_after_table = rgb_pc[table_keep]
            pixels_after_table_yx = valid_depth_pixels_yx[table_keep]

            bbox_keep = np.ones((xyz_after_table.shape[0],), dtype=bool)
            if object_only:
                bbox_keep = compute_keep_mask_inside_aabb_numpy(
                    xyz_after_table,
                    object_bbox_min,
                    object_bbox_max,
                )
            removed_outside_object_box = int(bbox_keep.shape[0] - np.count_nonzero(bbox_keep))
            rejected_bbox_pixels_yx = pixels_after_table_yx[~bbox_keep]

            xyz_world_mj = xyz_after_table[bbox_keep]
            rgb_pc = rgb_after_table[bbox_keep]
            kept_pixels_yx = pixels_after_table_yx[bbox_keep]

            debug_visibility_meta = PointCloudVisibilityDebug(
                sampled_pixels_yx=np.ascontiguousarray(sampled_pixels_yx, dtype=np.int32),
                valid_depth_pixels_yx=np.ascontiguousarray(valid_depth_pixels_yx, dtype=np.int32),
                kept_pixels_yx=np.ascontiguousarray(kept_pixels_yx, dtype=np.int32),
                rejected_table_pixels_yx=np.ascontiguousarray(rejected_table_pixels_yx, dtype=np.int32),
                rejected_bbox_pixels_yx=np.ascontiguousarray(rejected_bbox_pixels_yx, dtype=np.int32),
                xyz_world_mj_kept=np.ascontiguousarray(xyz_world_mj, dtype=np.float32),
                sampled_count=int(sampled_pixels_yx.shape[0]),
                valid_depth_count=int(valid_depth_pixels_yx.shape[0]),
                kept_count=int(kept_pixels_yx.shape[0]),
                rejected_table_count=int(rejected_table_pixels_yx.shape[0]),
                rejected_bbox_count=int(rejected_bbox_pixels_yx.shape[0]),
            )
        else:
            if clip_below_table:
                xyz_world_mj, rgb_pc, removed_below_plane = clip_points_below_plane_numpy(
                    xyz_world_mj,
                    rgb_pc,
                    table_plane,
                    margin=float(table_margin),
                    clearance=float(table_clearance),
                )

            if object_only:
                xyz_world_mj, rgb_pc, removed_outside_object_box = clip_points_outside_aabb_numpy(
                    xyz_world_mj,
                    rgb_pc,
                    object_bbox_min,
                    object_bbox_max,
                )

        if declared_capacity and xyz_world_mj.shape[0] > declared_capacity:
            step = int(np.ceil(xyz_world_mj.shape[0] / float(declared_capacity)))
            xyz_world_mj = xyz_world_mj[::step][:declared_capacity]
            rgb_pc = rgb_pc[::step][:declared_capacity]

        return self._pack_output(
            cam_name=cam_name,
            declared_capacity=declared_capacity,
            xyz_world_mj=xyz_world_mj,
            rgb_u8=rgb_pc,
            removed_below_plane=removed_below_plane,
            removed_outside_object_box=removed_outside_object_box,
            debug_visibility=debug_visibility_meta,
        )


class Open3dCudaPointCloudPipeline(_BasePointCloudPipeline):
    backend_name = "gpu"

    def __init__(
        self,
        *,
        device: str = "CUDA:0",
        buffer_pool: Optional[GpuPointCloudBufferPool] = None,
    ):
        super().__init__(buffer_pool=buffer_pool)

        if _o3d is None:
            raise RuntimeError(
                "Open3D is not installed. Install an Open3D build with CUDA support to use --pc_backend gpu."
            )
        if not _o3d.core.cuda.is_available():
            raise RuntimeError(
                "Open3D CUDA support is unavailable. Install/configure an Open3D CUDA build for --pc_backend gpu."
            )

        self.o3d = _o3d
        self.device = self.o3d.core.Device(device)
    def _to_tensor(self, array, dtype):
        return self.o3d.core.Tensor(array, dtype=dtype, device=self.device)

    def _count_true(self, mask) -> int:
        # Open3D CUDA reductions do not support Bool directly on this build.
        return int(mask.to(self.o3d.core.Dtype.Int32).sum().item())

    def build_frame(
        self,
        *,
        cam_name: str,
        depth_m: np.ndarray,
        rgb_u8: np.ndarray,
        width: int,
        height: int,
        fovy_deg: float,
        intrinsics_mode: str,
        stride: int,
        min_depth: float,
        max_depth: Optional[float],
        flip_x: bool,
        flip_y: bool,
        flip_z: bool,
        pc_scale: float,
        cam_t_mj: np.ndarray,
        cam_R_mj: np.ndarray,
        cam_anchor_corr: np.ndarray,
        clip_below_table: bool,
        table_plane,
        table_margin: float,
        table_clearance: float,
        object_only: bool,
        object_bbox_min,
        object_bbox_max,
        declared_capacity: int,
        sampling_mode: str = "grid",
        debug_visibility: bool = False,
    ) -> PointCloudFrameOutput:
        stride = max(1, int(stride))
        fx, fy, cx, cy = compute_intrinsics(width, height, fovy_deg, mode=intrinsics_mode)

        yy_np, xx_np = select_sampled_pixel_grids_numpy(
            cam_name=cam_name,
            depth_m=depth_m,
            width=width,
            height=height,
            stride=stride,
            sampling_mode=sampling_mode,
            min_depth=min_depth,
            max_depth=max_depth,
        )
        depth_sampled = np.ascontiguousarray(depth_m[yy_np, xx_np], dtype=np.float32)
        rgb_sampled = np.ascontiguousarray(rgb_u8[yy_np, xx_np, :], dtype=np.uint8)
        valid_np = np.isfinite(depth_sampled) & (depth_sampled > float(min_depth))
        if max_depth is not None:
            valid_np &= depth_sampled < float(max_depth)

        z = self._to_tensor(depth_sampled, self.o3d.core.Dtype.Float32)
        xx = self._to_tensor(xx_np.astype(np.float32), self.o3d.core.Dtype.Float32)
        yy = self._to_tensor(yy_np.astype(np.float32), self.o3d.core.Dtype.Float32)
        rgb_t = self._to_tensor(rgb_sampled, self.o3d.core.Dtype.UInt8)

        valid = z > float(min_depth)
        if max_depth is not None:
            valid = valid & (z < float(max_depth))

        x = (xx - float(cx)) * z / float(fx)
        y = (yy - float(cy)) * z / float(fy)
        if flip_x:
            x = -x
        if flip_y:
            y = -y

        xyz = self.o3d.core.concatenate(
            (
                x.reshape((-1, 1)),
                y.reshape((-1, 1)),
                z.reshape((-1, 1)),
            ),
            1,
        )
        colors = rgb_t.reshape((-1, 3))
        mask = valid.reshape((-1,))
        xyz = xyz[mask]
        colors = colors[mask]

        debug_visibility_meta: Optional[PointCloudVisibilityDebug] = None

        if int(xyz.shape[0]) == 0:
            xyz_world_mj = np.zeros((0, 3), dtype=np.float32)
            rgb_cpu = np.zeros((0, 3), dtype=np.uint8)
            if debug_visibility:
                sampled_pixels_yx = np.stack(
                    [yy_np.reshape(-1), xx_np.reshape(-1)],
                    axis=1,
                ).astype(np.int32, copy=False)
                valid_depth_pixels_yx = np.stack(
                    [yy_np[valid_np], xx_np[valid_np]],
                    axis=1,
                ).astype(np.int32, copy=False)
                debug_visibility_meta = PointCloudVisibilityDebug(
                    sampled_pixels_yx=np.ascontiguousarray(sampled_pixels_yx, dtype=np.int32),
                    valid_depth_pixels_yx=np.ascontiguousarray(valid_depth_pixels_yx, dtype=np.int32),
                    kept_pixels_yx=np.zeros((0, 2), dtype=np.int32),
                    rejected_table_pixels_yx=np.zeros((0, 2), dtype=np.int32),
                    rejected_bbox_pixels_yx=np.zeros((0, 2), dtype=np.int32),
                    xyz_world_mj_kept=np.zeros((0, 3), dtype=np.float32),
                    sampled_count=int(sampled_pixels_yx.shape[0]),
                    valid_depth_count=int(valid_depth_pixels_yx.shape[0]),
                    kept_count=0,
                    rejected_table_count=0,
                    rejected_bbox_count=0,
                )
            return self._pack_output(
                cam_name=cam_name,
                declared_capacity=declared_capacity,
                xyz_world_mj=xyz_world_mj,
                rgb_u8=rgb_cpu,
                removed_below_plane=0,
                removed_outside_object_box=0,
                debug_visibility=debug_visibility_meta,
            )

        if flip_z:
            xyz = self.o3d.core.concatenate(
                (
                    xyz[:, 0].reshape((-1, 1)),
                    xyz[:, 1].reshape((-1, 1)),
                    (-xyz[:, 2]).reshape((-1, 1)),
                ),
                1,
            )
        if pc_scale != 1.0:
            xyz = xyz * float(pc_scale)

        if debug_visibility:
            sampled_pixels_yx = np.stack(
                [yy_np.reshape(-1), xx_np.reshape(-1)],
                axis=1,
            ).astype(np.int32, copy=False)
            valid_depth_pixels_yx = np.stack(
                [yy_np[valid_np], xx_np[valid_np]],
                axis=1,
            ).astype(np.int32, copy=False)
            x_np = ((xx_np.astype(np.float32) - float(cx)) * depth_sampled / float(fx))
            y_np = ((yy_np.astype(np.float32) - float(cy)) * depth_sampled / float(fy))
            if flip_x:
                x_np = -x_np
            if flip_y:
                y_np = -y_np
            xyz_cam_debug = np.stack([x_np, y_np, depth_sampled], axis=-1)[valid_np].reshape(-1, 3).astype(np.float32)
            if flip_z and xyz_cam_debug.shape[0] > 0:
                xyz_cam_debug[:, 2] *= -1.0
            if pc_scale != 1.0 and xyz_cam_debug.shape[0] > 0:
                xyz_cam_debug *= float(pc_scale)
            xyz_world_debug = (
                (xyz_cam_debug @ np.asarray(cam_R_mj, dtype=np.float32).T)
                + np.asarray(cam_t_mj, dtype=np.float32).reshape(1, 3)
                + np.asarray(cam_anchor_corr, dtype=np.float32).reshape(1, 3)
            )
            table_keep_debug = np.ones((xyz_world_debug.shape[0],), dtype=bool)
            if clip_below_table:
                table_keep_debug = compute_keep_mask_above_plane_numpy(
                    xyz_world_debug,
                    table_plane,
                    margin=float(table_margin),
                    clearance=float(table_clearance),
                )
            rejected_table_pixels_yx = valid_depth_pixels_yx[~table_keep_debug]
            xyz_after_table_debug = xyz_world_debug[table_keep_debug]
            pixels_after_table_yx = valid_depth_pixels_yx[table_keep_debug]

            bbox_keep_debug = np.ones((xyz_after_table_debug.shape[0],), dtype=bool)
            if object_only:
                bbox_keep_debug = compute_keep_mask_inside_aabb_numpy(
                    xyz_after_table_debug,
                    object_bbox_min,
                    object_bbox_max,
                )
            rejected_bbox_pixels_yx = pixels_after_table_yx[~bbox_keep_debug]
            kept_pixels_yx = pixels_after_table_yx[bbox_keep_debug]
            xyz_world_kept_debug = xyz_after_table_debug[bbox_keep_debug]

            debug_visibility_meta = PointCloudVisibilityDebug(
                sampled_pixels_yx=np.ascontiguousarray(sampled_pixels_yx, dtype=np.int32),
                valid_depth_pixels_yx=np.ascontiguousarray(valid_depth_pixels_yx, dtype=np.int32),
                kept_pixels_yx=np.ascontiguousarray(kept_pixels_yx, dtype=np.int32),
                rejected_table_pixels_yx=np.ascontiguousarray(rejected_table_pixels_yx, dtype=np.int32),
                rejected_bbox_pixels_yx=np.ascontiguousarray(rejected_bbox_pixels_yx, dtype=np.int32),
                xyz_world_mj_kept=np.ascontiguousarray(xyz_world_kept_debug, dtype=np.float32),
                sampled_count=int(sampled_pixels_yx.shape[0]),
                valid_depth_count=int(valid_depth_pixels_yx.shape[0]),
                kept_count=int(kept_pixels_yx.shape[0]),
                rejected_table_count=int(rejected_table_pixels_yx.shape[0]),
                rejected_bbox_count=int(rejected_bbox_pixels_yx.shape[0]),
            )

        cam_R_t = self._to_tensor(np.asarray(cam_R_mj, dtype=np.float32), self.o3d.core.Dtype.Float32)
        cam_t_t = self._to_tensor(
            np.asarray(cam_t_mj, dtype=np.float32).reshape(1, 3),
            self.o3d.core.Dtype.Float32,
        )
        corr_t = self._to_tensor(
            np.asarray(cam_anchor_corr, dtype=np.float32).reshape(1, 3),
            self.o3d.core.Dtype.Float32,
        )
        xyz_world = (xyz @ cam_R_t.T()) + cam_t_t + corr_t

        removed_below_plane = 0
        if clip_below_table and table_plane is not None and int(xyz_world.shape[0]) > 0:
            plane_point = self._to_tensor(
                np.asarray(table_plane.point, dtype=np.float32).reshape(1, 3),
                self.o3d.core.Dtype.Float32,
            )
            plane_normal = self._to_tensor(
                np.asarray(table_plane.normal, dtype=np.float32).reshape(3, 1),
                self.o3d.core.Dtype.Float32,
            )
            signed = ((xyz_world - plane_point) @ plane_normal).reshape((-1,))
            keep = signed > float(table_margin + table_clearance)
            removed_below_plane = int(xyz_world.shape[0]) - self._count_true(keep)
            xyz_world = xyz_world[keep]
            colors = colors[keep]

        removed_outside_object_box = 0
        if object_only and int(xyz_world.shape[0]) > 0:
            bbox_min = self._to_tensor(
                np.asarray(object_bbox_min, dtype=np.float32).reshape(1, 3),
                self.o3d.core.Dtype.Float32,
            )
            bbox_max = self._to_tensor(
                np.asarray(object_bbox_max, dtype=np.float32).reshape(1, 3),
                self.o3d.core.Dtype.Float32,
            )
            keep = (
                (xyz_world[:, 0] >= bbox_min[0, 0])
                & (xyz_world[:, 0] <= bbox_max[0, 0])
                & (xyz_world[:, 1] >= bbox_min[0, 1])
                & (xyz_world[:, 1] <= bbox_max[0, 1])
                & (xyz_world[:, 2] >= bbox_min[0, 2])
                & (xyz_world[:, 2] <= bbox_max[0, 2])
            )
            removed_outside_object_box = int(xyz_world.shape[0]) - self._count_true(keep)
            xyz_world = xyz_world[keep]
            colors = colors[keep]

        if declared_capacity and int(xyz_world.shape[0]) > declared_capacity:
            step = int(np.ceil(int(xyz_world.shape[0]) / float(declared_capacity)))
            xyz_world = xyz_world[::step][:declared_capacity]
            colors = colors[::step][:declared_capacity]

        xyz_world_mj = np.asarray(xyz_world.cpu().numpy(), dtype=np.float32)
        rgb_cpu = np.asarray(colors.cpu().numpy(), dtype=np.uint8)

        return self._pack_output(
            cam_name=cam_name,
            declared_capacity=declared_capacity,
            xyz_world_mj=xyz_world_mj,
            rgb_u8=rgb_cpu,
            removed_below_plane=removed_below_plane,
            removed_outside_object_box=removed_outside_object_box,
            debug_visibility=debug_visibility_meta,
        )
