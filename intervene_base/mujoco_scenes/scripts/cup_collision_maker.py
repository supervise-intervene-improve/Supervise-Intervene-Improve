import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import trimesh


@dataclass
class CupShellConfig:
    name: str
    mesh_file: str

    # Body pose in MuJoCo world
    body_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Visual mesh asset
    mesh_name: str = "cup_visual_mesh"
    material_name: str = "cup_blue"
    mesh_scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    rgba: str = "0.2 0.5 0.9 1"

    # Hollow-cylinder collision model
    num_wall_boxes: int = 16
    wall_thickness: float = 0.003
    wall_outward_offset: float = 0.0
    bottom_thickness: float = 0.004
    rim_margin: float = 0.002

    # Top boxes
    top_cap_enabled: bool = True
    top_cap_radial_thickness: float = 0.002
    top_cap_vertical_thickness: float = 0.002
    top_cap_tangential_scale: float = 1.0
    top_cap_outward_offset: float = 0.0
    top_cap_z_offset: float = 0.0

    # Estimate bottom radius / center from lower part of mesh
    bottom_region_fraction: float = 0.15
    bottom_radius_quantile: float = 0.95
    radius_padding: float = 0.001

    # Axis detection
    auto_detect_axis: bool = True
    axis_if_fixed: str = "z"  # "x", "y", or "z"

    # Manual correction for the WHOLE collision shell in LOCAL cup frame
    collision_frame_pos_offset_local: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    collision_frame_euler_deg_local: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # rx, ry, rz

    # Optional manual correction for visual geom inside body frame
    visual_pos_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    visual_euler_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # Debug / output
    show_collision: bool = True
    free_joint: bool = True
    print_debug: bool = True
    GRAVITY:bool = False


def fmt(vals) -> str:
    if isinstance(vals, (list, tuple, np.ndarray)):
        return " ".join(f"{float(v):.6g}" for v in vals)
    return f"{float(vals):.6g}"


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v if n < 1e-12 else v / n


def rotation_matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = R
    q = trimesh.transformations.quaternion_from_matrix(M)  # w x y z
    return np.asarray(q, dtype=float)


def euler_xyz_deg_to_rotmat(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)

    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)

    Rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cx, -sx],
        [0.0, sx, cx],
    ], dtype=float)

    Ry = np.array([
        [cy, 0.0, sy],
        [0.0, 1.0, 0.0],
        [-sy, 0.0, cy],
    ], dtype=float)

    Rz = np.array([
        [cz, -sz, 0.0],
        [sz, cz, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)

    return Rz @ Ry @ Rx


def euler_xyz_deg_to_quat_wxyz(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    R = euler_xyz_deg_to_rotmat(rx_deg, ry_deg, rz_deg)
    return rotation_matrix_to_quat_wxyz(R)


def load_mesh_scaled(mesh_file: str, scale: Tuple[float, float, float]) -> trimesh.Trimesh:
    mesh = trimesh.load(mesh_file, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Could not load mesh as Trimesh: {mesh_file}")

    verts = np.asarray(mesh.vertices).copy()
    verts[:, 0] *= scale[0]
    verts[:, 1] *= scale[1]
    verts[:, 2] *= scale[2]

    return trimesh.Trimesh(vertices=verts, faces=mesh.faces, process=False)


def estimate_frame(
    mesh: trimesh.Trimesh,
    auto_detect_axis: bool = True,
    axis_if_fixed: str = "z",
):
    """
    Returns:
        center_raw: origin of local cup frame in RAW mesh coordinates
        R_raw_from_local: local cup frame -> RAW mesh frame
        local_vertices: vertices in local cup frame
    """
    vertices = np.asarray(mesh.vertices)

    # More stable than plain vertex mean
    center_raw = np.asarray(mesh.bounding_box.centroid, dtype=float)
    centered = vertices - center_raw

    if auto_detect_axis:
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, order]

        best_extent = -1.0
        z_axis = None
        for i in range(3):
            axis = eigvecs[:, i]
            proj = centered @ axis
            extent = proj.max() - proj.min()
            if extent > best_extent:
                best_extent = extent
                z_axis = normalize(axis)
    else:
        axis_map = {
            "x": np.array([1.0, 0.0, 0.0]),
            "y": np.array([0.0, 1.0, 0.0]),
            "z": np.array([0.0, 0.0, 1.0]),
        }
        if axis_if_fixed not in axis_map:
            raise ValueError("axis_if_fixed must be one of: x, y, z")
        z_axis = axis_map[axis_if_fixed]

    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(tmp, z_axis)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])

    x0 = normalize(np.cross(tmp, z_axis))
    y0 = normalize(np.cross(z_axis, x0))
    R0 = np.column_stack([x0, y0, z_axis])

    local0 = (R0.T @ centered.T).T

    # Force +Z from bottom -> top
    z0 = local0[:, 2]
    low_mask = z0 < np.quantile(z0, 0.10)
    high_mask = z0 > np.quantile(z0, 0.90)

    r_low = np.linalg.norm(local0[low_mask, :2], axis=1).mean() if low_mask.any() else 0.0
    r_high = np.linalg.norm(local0[high_mask, :2], axis=1).mean() if high_mask.any() else 0.0

    if r_high < r_low:
        z_axis = -z_axis
        x0 = -x0
        y0 = normalize(np.cross(z_axis, x0))
        R0 = np.column_stack([x0, y0, z_axis])
        local0 = (R0.T @ centered.T).T

    # Stabilize yaw if bottom is anisotropic
    z = local0[:, 2]
    z_min = float(z.min())
    z_max = float(z.max())
    height = z_max - z_min

    bottom_cut = z_min + 0.20 * height
    bottom_pts = local0[z <= bottom_cut]

    if bottom_pts.shape[0] < 10:
        return center_raw, R0, local0

    xy = bottom_pts[:, :2]
    xy_centered = xy - xy.mean(axis=0, keepdims=True)

    cov2 = np.cov(xy_centered.T)
    eigvals2, eigvecs2 = np.linalg.eigh(cov2)
    order2 = np.argsort(eigvals2)[::-1]
    eigvals2 = eigvals2[order2]
    eigvecs2 = eigvecs2[:, order2]

    major2 = eigvecs2[:, 0]
    anisotropy = 0.0
    if abs(eigvals2[0]) > 1e-12:
        anisotropy = (eigvals2[0] - eigvals2[1]) / eigvals2[0]

    if anisotropy < 0.02:
        return center_raw, R0, local0

    x_local2 = np.array([major2[0], major2[1], 0.0])
    x_axis = normalize(R0 @ x_local2)
    y_axis = normalize(np.cross(z_axis, x_axis))

    R_raw_from_local = np.column_stack([x_axis, y_axis, z_axis])
    local_vertices = (R_raw_from_local.T @ centered.T).T

    return center_raw, R_raw_from_local, local_vertices


def estimate_height_bottom_radius_and_center(local_vertices: np.ndarray, cfg: CupShellConfig):
    z = local_vertices[:, 2]
    xy = local_vertices[:, :2]

    z_min = float(z.min())
    z_max = float(z.max())
    height = z_max - z_min
    if height <= 1e-8:
        raise ValueError("Degenerate mesh height")

    bottom_cut = z_min + cfg.bottom_region_fraction * height
    mask_bottom = z <= bottom_cut
    if mask_bottom.sum() < 10:
        raise ValueError("Too few bottom-region points for radius estimation")

    xy_bottom = xy[mask_bottom]
    local_xy_center = xy_bottom.mean(axis=0)

    r_bottom = np.linalg.norm(xy_bottom - local_xy_center[None, :], axis=1)
    outer_radius = float(np.quantile(r_bottom, cfg.bottom_radius_quantile)) + cfg.radius_padding

    return z_min, z_max, height, outer_radius, local_xy_center


def apply_local_frame_offset(
    local_pos: np.ndarray,
    R_local_geom: np.ndarray,
    frame_pos_offset_local: np.ndarray,
    frame_euler_deg_local: Tuple[float, float, float],
):
    """
    Apply one rigid correction to the whole collision shell in LOCAL cup frame.
    """
    R_offset = euler_xyz_deg_to_rotmat(*frame_euler_deg_local)
    local_pos_new = np.asarray(frame_pos_offset_local, dtype=float) + R_offset @ local_pos
    R_local_geom_new = R_offset @ R_local_geom
    return local_pos_new, R_local_geom_new


def local_to_body_geom(
    local_pos: np.ndarray,
    R_local_geom: np.ndarray,
    center_raw: np.ndarray,
    R_raw_from_local: np.ndarray,
    frame_pos_offset_local: np.ndarray = None,
    frame_euler_deg_local: Tuple[float, float, float] = (0.0, 0.0, 0.0),
):
    """
    Convert geom from LOCAL cup frame directly to BODY/source frame.

    IMPORTANT:
    The collision shell is built from the RAW/source mesh coordinates.
    Therefore it should be placed in the body/source frame directly.

    DO NOT apply MuJoCo mesh_pos/mesh_quat here.
    MuJoCo already applies those internally to the visual mesh geom.
    """
    if frame_pos_offset_local is None:
        frame_pos_offset_local = np.zeros(3, dtype=float)

    local_pos_corr, R_local_geom_corr = apply_local_frame_offset(
        local_pos=local_pos,
        R_local_geom=R_local_geom,
        frame_pos_offset_local=np.asarray(frame_pos_offset_local, dtype=float),
        frame_euler_deg_local=frame_euler_deg_local,
    )

    pos_body = center_raw + R_raw_from_local @ local_pos_corr
    R_body = R_raw_from_local @ R_local_geom_corr
    quat_body = rotation_matrix_to_quat_wxyz(R_body)

    return pos_body, quat_body


def make_bottom_geom(
    name: str,
    center_raw: np.ndarray,
    R_raw_from_local: np.ndarray,
    local_xy_center: np.ndarray,
    outer_radius: float,
    z_min: float,
    bottom_thickness: float,
    show: bool,
    frame_pos_offset_local: np.ndarray,
    frame_euler_deg_local: Tuple[float, float, float],
) -> str:
    rgba = "0 1 0 0.15" if show else "0 0 0 0"

    local_center = np.array([
        local_xy_center[0],
        local_xy_center[1],
        z_min + 0.5 * bottom_thickness,
    ], dtype=float)

    R_local_geom = np.eye(3)

    pos_body, quat_body = local_to_body_geom(
        local_pos=local_center,
        R_local_geom=R_local_geom,
        center_raw=center_raw,
        R_raw_from_local=R_raw_from_local,
        frame_pos_offset_local=frame_pos_offset_local,
        frame_euler_deg_local=frame_euler_deg_local,
    )

    return (
        f'      <geom name="{name}_bottom" type="cylinder" '
        f'pos="{fmt(pos_body)}" quat="{fmt(quat_body)}" '
        f'size="{fmt([outer_radius, 0.5 * bottom_thickness])}" '
        f'contype="1" conaffinity="1" rgba="{rgba}"/>'
    )


def make_wall_box_geoms(
    name: str,
    center_raw: np.ndarray,
    R_raw_from_local: np.ndarray,
    local_xy_center: np.ndarray,
    z_min: float,
    height: float,
    outer_radius: float,
    wall_thickness: float,
    wall_outward_offset: float,
    bottom_thickness: float,
    rim_margin: float,
    num_boxes: int,
    show: bool,
    frame_pos_offset_local: np.ndarray,
    frame_euler_deg_local: Tuple[float, float, float],
):
    rgba = "1 0 0 0.15" if show else "0 0 0 0"

    inner_radius = max(0.0, outer_radius - wall_thickness)
    mid_radius = 0.5 * (outer_radius + inner_radius) + wall_outward_offset

    wall_z_min = z_min + bottom_thickness
    wall_height = max(1e-6, height - bottom_thickness - rim_margin)
    wall_center_z = wall_z_min + 0.5 * wall_height

    dtheta = 2.0 * math.pi / num_boxes
    tangential_half = mid_radius * math.tan(math.pi / num_boxes)
    radial_half = 0.5 * wall_thickness
    vertical_half = 0.5 * wall_height

    geoms = []
    for i in range(num_boxes):
        theta = i * dtheta

        local_center = np.array([
            local_xy_center[0] + mid_radius * math.cos(theta),
            local_xy_center[1] + mid_radius * math.sin(theta),
            wall_center_z,
        ], dtype=float)

        radial = np.array([math.cos(theta), math.sin(theta), 0.0], dtype=float)
        tangent = np.array([-math.sin(theta), math.cos(theta), 0.0], dtype=float)
        vertical = np.array([0.0, 0.0, 1.0], dtype=float)
        R_local_geom = np.column_stack([radial, tangent, vertical])

        pos_body, quat_body = local_to_body_geom(
            local_pos=local_center,
            R_local_geom=R_local_geom,
            center_raw=center_raw,
            R_raw_from_local=R_raw_from_local,
            frame_pos_offset_local=frame_pos_offset_local,
            frame_euler_deg_local=frame_euler_deg_local,
        )

        geoms.append(
            f'      <geom name="{name}_wall_{i:02d}" type="box" '
            f'pos="{fmt(pos_body)}" quat="{fmt(quat_body)}" '
            f'size="{fmt([radial_half, tangential_half, vertical_half])}" '
            f'contype="1" conaffinity="1" rgba="{rgba}"/>'
        )

    return geoms, inner_radius


def make_top_cap_geoms(
    name: str,
    center_raw: np.ndarray,
    R_raw_from_local: np.ndarray,
    local_xy_center: np.ndarray,
    z_min: float,
    height: float,
    outer_radius: float,
    wall_thickness: float,
    wall_outward_offset: float,
    z_offset: float,
    rim_margin: float,
    num_boxes: int,
    cap_radial_thickness: float,
    cap_vertical_thickness: float,
    cap_tangential_scale: float,
    cap_outward_offset: float,
    show: bool,
    frame_pos_offset_local: np.ndarray,
    frame_euler_deg_local: Tuple[float, float, float],
):
    rgba = "0 0 1 0.20" if show else "0 0 0 0"

    inner_radius = max(0.0, outer_radius - wall_thickness)
    base_mid_radius = 0.5 * (outer_radius + inner_radius) + wall_outward_offset
    cap_mid_radius = base_mid_radius + cap_outward_offset

    dtheta = 2.0 * math.pi / num_boxes

    tangential_half = cap_tangential_scale * base_mid_radius * math.tan(math.pi / num_boxes)
    radial_half = 0.5 * cap_radial_thickness
    vertical_half = 0.5 * cap_vertical_thickness

    top_center_z = z_min + height - rim_margin - vertical_half + z_offset

    geoms = []
    for i in range(num_boxes):
        theta = i * dtheta

        local_center = np.array([
            local_xy_center[0] + cap_mid_radius * math.cos(theta),
            local_xy_center[1] + cap_mid_radius * math.sin(theta),
            top_center_z,
        ], dtype=float)

        radial = np.array([math.cos(theta), math.sin(theta), 0.0], dtype=float)
        tangent = np.array([-math.sin(theta), math.cos(theta), 0.0], dtype=float)
        vertical = np.array([0.0, 0.0, 1.0], dtype=float)
        R_local_geom = np.column_stack([radial, tangent, vertical])

        pos_body, quat_body = local_to_body_geom(
            local_pos=local_center,
            R_local_geom=R_local_geom,
            center_raw=center_raw,
            R_raw_from_local=R_raw_from_local,
            frame_pos_offset_local=frame_pos_offset_local,
            frame_euler_deg_local=frame_euler_deg_local,
        )

        geoms.append(
            f'      <geom name="{name}_topcap_{i:02d}" type="box" '
            f'pos="{fmt(pos_body)}" quat="{fmt(quat_body)}" '
            f'size="{fmt([radial_half, tangential_half, vertical_half])}" '
            f'contype="1" conaffinity="1" rgba="{rgba}"/>'
        )

    return geoms


def generate_cup_xml(cfg: CupShellConfig) -> str:
    raw_mesh = load_mesh_scaled(cfg.mesh_file, cfg.mesh_scale)

    center_raw, R_raw_from_local, local_vertices = estimate_frame(
        raw_mesh,
        auto_detect_axis=cfg.auto_detect_axis,
        axis_if_fixed=cfg.axis_if_fixed,
    )

    z_min, z_max, height, outer_radius, local_xy_center = \
        estimate_height_bottom_radius_and_center(local_vertices, cfg)

    frame_pos_offset_local = np.asarray(cfg.collision_frame_pos_offset_local, dtype=float)
    frame_euler_deg_local = cfg.collision_frame_euler_deg_local
    visual_quat = euler_xyz_deg_to_quat_wxyz(*cfg.visual_euler_deg)

    body_lines = [f'    <body name="{cfg.name}" pos="{fmt(cfg.body_pos)}">']

    if cfg.free_joint:
        body_lines.append(f'      <joint name="{cfg.name}_free" type="free"/>')

    # Visual mesh: MuJoCo applies mesh compiler offsets internally
    body_lines.append(
        f'      <geom name="{cfg.name}_visual" type="mesh" mesh="{cfg.mesh_name}" '
        f'material="{cfg.material_name}" contype="0" conaffinity="0" '
        f'pos="{fmt(cfg.visual_pos_offset)}" quat="{fmt(visual_quat)}"/>'
    )

    body_lines.append(
        make_bottom_geom(
            name=cfg.name,
            center_raw=center_raw,
            R_raw_from_local=R_raw_from_local,
            local_xy_center=local_xy_center,
            outer_radius=outer_radius,
            z_min=z_min,
            bottom_thickness=cfg.bottom_thickness,
            show=cfg.show_collision,
            frame_pos_offset_local=frame_pos_offset_local,
            frame_euler_deg_local=frame_euler_deg_local,
        )
    )

    wall_geoms, inner_radius = make_wall_box_geoms(
        name=cfg.name,
        center_raw=center_raw,
        R_raw_from_local=R_raw_from_local,
        local_xy_center=local_xy_center,
        z_min=z_min,
        height=height,
        outer_radius=outer_radius,
        wall_thickness=cfg.wall_thickness,
        wall_outward_offset=cfg.wall_outward_offset,
        bottom_thickness=cfg.bottom_thickness,
        rim_margin=cfg.rim_margin,
        num_boxes=cfg.num_wall_boxes,
        show=cfg.show_collision,
        frame_pos_offset_local=frame_pos_offset_local,
        frame_euler_deg_local=frame_euler_deg_local,
    )
    body_lines.extend(wall_geoms)

    if cfg.top_cap_enabled:
        top_cap_geoms = make_top_cap_geoms(
            name=cfg.name,
            center_raw=center_raw,
            R_raw_from_local=R_raw_from_local,
            local_xy_center=local_xy_center,
            z_min=z_min,
            height=height,
            outer_radius=outer_radius,
            wall_thickness=cfg.wall_thickness,
            wall_outward_offset=cfg.wall_outward_offset,
            z_offset=cfg.top_cap_z_offset,
            rim_margin=cfg.rim_margin,
            num_boxes=cfg.num_wall_boxes,
            cap_radial_thickness=cfg.top_cap_radial_thickness,
            cap_vertical_thickness=cfg.top_cap_vertical_thickness,
            cap_tangential_scale=cfg.top_cap_tangential_scale,
            cap_outward_offset=cfg.top_cap_outward_offset,
            show=cfg.show_collision,
            frame_pos_offset_local=frame_pos_offset_local,
            frame_euler_deg_local=frame_euler_deg_local,
        )
        body_lines.extend(top_cap_geoms)

    body_lines.append("    </body>")

    mesh_path = Path(cfg.mesh_file).expanduser().resolve()

    xml = f"""<mujoco model="{cfg.name}">
  <option gravity="0 0 0"/>

  <asset>
    <material name="{cfg.material_name}" rgba="0.2 0.5 0.9 1"/>
    <mesh name="{cfg.mesh_name}" file="{mesh_path.as_posix()}" scale="{fmt(cfg.mesh_scale)}"/>
  </asset>

  <worldbody>
{chr(10).join(body_lines)}
  </worldbody>
</mujoco>
"""

    if cfg.print_debug:
        print("Estimated cup parameters:")
        print(f"  height                      = {height:.6f}")
        print(f"  z_min                       = {z_min:.6f}")
        print(f"  z_max                       = {z_max:.6f}")
        print(f"  outer_radius                = {outer_radius:.6f}")
        print(f"  inner_radius                = {inner_radius:.6f}")
        print(f"  local_xy_center             = ({local_xy_center[0]:.6f}, {local_xy_center[1]:.6f})")
        print(f"  center_raw                  = ({center_raw[0]:.6f}, {center_raw[1]:.6f}, {center_raw[2]:.6f})")
        print("  R_raw_from_local =")
        print(R_raw_from_local)
        print(f"  collision_frame_pos_offset  = {cfg.collision_frame_pos_offset_local}")
        print(f"  collision_frame_euler_deg   = {cfg.collision_frame_euler_deg_local}")
        print(f"  visual_pos_offset           = {cfg.visual_pos_offset}")
        print(f"  visual_euler_deg            = {cfg.visual_euler_deg}")
        print(f"  wall_boxes                  = {cfg.num_wall_boxes}")

    return xml


def generate_multi_cup_xml(configs) -> str:
    asset_lines = []
    body_blocks = []

    for cfg in configs:
        raw_mesh = load_mesh_scaled(cfg.mesh_file, cfg.mesh_scale)

        center_raw, R_raw_from_local, local_vertices = estimate_frame(
            raw_mesh,
            auto_detect_axis=cfg.auto_detect_axis,
            axis_if_fixed=cfg.axis_if_fixed,
        )

        z_min, z_max, height, outer_radius, local_xy_center = \
            estimate_height_bottom_radius_and_center(local_vertices, cfg)

        frame_pos_offset_local = np.asarray(cfg.collision_frame_pos_offset_local, dtype=float)
        frame_euler_deg_local = cfg.collision_frame_euler_deg_local
        visual_quat = euler_xyz_deg_to_quat_wxyz(*cfg.visual_euler_deg)

        mesh_path = Path(cfg.mesh_file).expanduser().resolve()

        asset_lines.append(
            f'    <material name="{cfg.material_name}" rgba="{cfg.rgba}"/>'
        )
        asset_lines.append(
            f'    <mesh name="{cfg.mesh_name}" file="{mesh_path.as_posix()}" scale="{fmt(cfg.mesh_scale)}"/>'
        )

        body_lines = [f'    <body name="{cfg.name}" pos="{fmt(cfg.body_pos)}">']

        if cfg.free_joint:
            body_lines.append(f'      <joint name="{cfg.name}_free" type="free"/>')

        body_lines.append(
            f'      <geom name="{cfg.name}_visual" type="mesh" mesh="{cfg.mesh_name}" '
            f'material="{cfg.material_name}" contype="0" conaffinity="0" '
            f'pos="{fmt(cfg.visual_pos_offset)}" quat="{fmt(visual_quat)}"/>'
        )

        body_lines.append(
            make_bottom_geom(
                name=cfg.name,
                center_raw=center_raw,
                R_raw_from_local=R_raw_from_local,
                local_xy_center=local_xy_center,
                outer_radius=outer_radius,
                z_min=z_min,
                bottom_thickness=cfg.bottom_thickness,
                show=cfg.show_collision,
                frame_pos_offset_local=frame_pos_offset_local,
                frame_euler_deg_local=frame_euler_deg_local,
            )
        )

        wall_geoms, inner_radius = make_wall_box_geoms(
            name=cfg.name,
            center_raw=center_raw,
            R_raw_from_local=R_raw_from_local,
            local_xy_center=local_xy_center,
            z_min=z_min,
            height=height,
            outer_radius=outer_radius,
            wall_thickness=cfg.wall_thickness,
            wall_outward_offset=cfg.wall_outward_offset,
            bottom_thickness=cfg.bottom_thickness,
            rim_margin=cfg.rim_margin,
            num_boxes=cfg.num_wall_boxes,
            show=cfg.show_collision,
            frame_pos_offset_local=frame_pos_offset_local,
            frame_euler_deg_local=frame_euler_deg_local,
        )
        body_lines.extend(wall_geoms)

        if cfg.top_cap_enabled:
            top_cap_geoms = make_top_cap_geoms(
                name=cfg.name,
                center_raw=center_raw,
                R_raw_from_local=R_raw_from_local,
                local_xy_center=local_xy_center,
                z_min=z_min,
                height=height,
                outer_radius=outer_radius,
                wall_thickness=cfg.wall_thickness,
                wall_outward_offset=cfg.wall_outward_offset,
                z_offset=cfg.top_cap_z_offset,
                rim_margin=cfg.rim_margin,
                num_boxes=cfg.num_wall_boxes,
                cap_radial_thickness=cfg.top_cap_radial_thickness,
                cap_vertical_thickness=cfg.top_cap_vertical_thickness,
                cap_tangential_scale=cfg.top_cap_tangential_scale,
                cap_outward_offset=cfg.top_cap_outward_offset,
                show=cfg.show_collision,
                frame_pos_offset_local=frame_pos_offset_local,
                frame_euler_deg_local=frame_euler_deg_local,
            )
            body_lines.extend(top_cap_geoms)

        body_lines.append("    </body>")
        body_blocks.append("\n".join(body_lines))

        if cfg.print_debug:
            print(f"\n[{cfg.name}]")
            print(f"  height                    = {height:.6f}")
            print(f"  outer_radius              = {outer_radius:.6f}")
            print(f"  inner_radius              = {inner_radius:.6f}")
            print(f"  local_xy_center           = ({local_xy_center[0]:.6f}, {local_xy_center[1]:.6f})")
            print(f"  collision_frame_pos       = {cfg.collision_frame_pos_offset_local}")
            print(f"  collision_frame_euler_deg = {cfg.collision_frame_euler_deg_local}")
            print(f"  mesh_scale                = {cfg.mesh_scale}")
            print(f"  body_pos                  = {cfg.body_pos}")

    gravity_line = '<option gravity="0 0 0"/>' if cfg.GRAVITY else '<!-- <option gravity="0 0 0"/> -->'

    xml = f"""<mujoco model="multi_cup_scene">
    {gravity_line}

    <asset>
    {chr(10).join(asset_lines)}
    </asset>

    <worldbody>
    {chr(10).join(body_blocks)}
    </worldbody>
    </mujoco>
    """
    return xml

# if __name__ == "__main__":



#     cfg = CupShellConfig(
#         name="cup1",
#         mesh_file="cups/cup_nontextured.stl",
#         body_pos=(0.48, -0.20, 0.24),

#         mesh_name="cup_normal",
#         material_name="cup_blue",
#         mesh_scale=(1.0, 1.0, 1.0),

#         num_wall_boxes=25,
#         wall_thickness=0.0015,
#         wall_outward_offset=0.001,
#         bottom_thickness=0.004,
#         rim_margin=0.002,

#         top_cap_enabled=True,
#         top_cap_radial_thickness=0.0065,
#         top_cap_vertical_thickness=0.004,
#         top_cap_tangential_scale=1.0,
#         top_cap_outward_offset=0.00175,
#         top_cap_z_offset=0.002,

#         bottom_region_fraction=0.15,
#         bottom_radius_quantile=0.95,
#         radius_padding=0.001,

#         auto_detect_axis=True,
#         axis_if_fixed="z",

#         # Manual tuning for collision shell
#         collision_frame_pos_offset_local=(0.0, 0.0, 0.0),
#         collision_frame_euler_deg_local=(-2.0, 1.0, 0.0),

#         # Usually leave visual at zero first
#         visual_pos_offset=(0.0, 0.0, 0.0),
#         visual_euler_deg=(0.0, 0.0, 0.0),

#         show_collision=True,
#         free_joint=True,
#         print_debug=True,
#     )

#     xml = generate_cup_xml(cfg)
#     Path("generated_hollow_cup.xml").write_text(xml, encoding="utf-8")
#     print("Saved: generated_hollow_cup.xml")

if __name__ == "__main__":
    common_kwargs = dict(
        mesh_file="cups/cup_nontextured.stl",

        num_wall_boxes=25,
        wall_thickness=0.0015,
        wall_outward_offset=0.001,
        bottom_thickness=0.004,
        rim_margin=0.002,

        top_cap_enabled=True,
        top_cap_radial_thickness=0.0055,
        top_cap_vertical_thickness=0.004,
        top_cap_tangential_scale=1.0,
        top_cap_outward_offset=0.00125,
        top_cap_z_offset=0.002,

        bottom_region_fraction=0.15,
        bottom_radius_quantile=0.95,
        radius_padding=0.001,

        auto_detect_axis=True,
        axis_if_fixed="z",

        collision_frame_pos_offset_local=(0.0, 0.0, 0.0),
        collision_frame_euler_deg_local=(-2.0, 1.0, 0.0),

        visual_pos_offset=(0.0, 0.0, 0.0),
        visual_euler_deg=(0.0, 0.0, 0.0),

        free_joint=True,
        print_debug=True,

        show_collision=False,
        GRAVITY=False,
    )

    cups = [
        CupShellConfig(
            name="cup1",
            body_pos=(0.42, -0.22, 0.24),
            mesh_name="cup_mesh_1",
            material_name="cup_red",
            mesh_scale=(0.90, 0.90, 0.90),
            rgba="0.85 0.20 0.20 1",
            **common_kwargs,
        ),
        CupShellConfig(
            name="cup2",
            body_pos=(0.78, -0.22, 0.24),
            mesh_name="cup_mesh_2",
            material_name="cup_green",
            mesh_scale=(1.00, 1.00, 1.00),
            rgba="0.20 0.75 0.30 1",
            **common_kwargs,
        ),
        CupShellConfig(
            name="cup3",
            body_pos=(0.42, 0.22, 0.24),
            mesh_name="cup_mesh_3",
            material_name="cup_blue",
            mesh_scale=(1.10, 1.10, 1.10),
            rgba="0.20 0.45 0.90 1",
            **common_kwargs,
        ),
        CupShellConfig(
            name="cup4",
            body_pos=(0.78, 0.22, 0.24),
            mesh_name="cup_mesh_4",
            material_name="cup_yellow",
            mesh_scale=(0.80, 0.80, 1.15),
            rgba="0.95 0.80 0.20 1",
            **common_kwargs,
        ),
    ]

    xml = generate_multi_cup_xml(cups)
    xml_name = f"generated_{len(cups)}_cups"
    Path(xml_name + ".xml").write_text(xml, encoding="utf-8")
    print("Saved: " + xml_name + ".xml")