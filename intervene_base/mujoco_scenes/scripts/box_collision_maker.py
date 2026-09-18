import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import numpy as np
import trimesh


@dataclass
class TexturedBoxConfig:
    name: str

    # visual mesh should be textured.obj
    visual_mesh_file: str

    # collision shape can be estimated from same obj or from nontextured stl
    collision_mesh_file: str

    # texture image
    texture_file: str

    # world pose
    body_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # asset names
    mesh_name: str = "visual_mesh"
    texture_name: str = "visual_tex"
    material_name: str = "visual_mat"

    # scaling
    visual_mesh_scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    collision_mesh_scale: Tuple[float, float, float] = (1.0, 1.0, 1.0)

    # frame estimation
    auto_detect_axes: bool = True

    # visual mesh manual tuning inside body frame
    visual_pos_offset_local: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    visual_euler_deg_local: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # collision box tuning
    collision_size_scale: Tuple[float, float, float] = (0.98, 0.98, 0.98)
    collision_padding: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    collision_pos_offset_local: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    collision_euler_deg_local: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # rendering / dynamics
    show_collision: bool = True
    show_debug_collision_mesh: bool = False
    free_joint: bool = True
    mass: float = 0.3
    print_debug: bool = True

    # Contact friction "sliding torsional rolling", emitted explicitly on the collision
    # geom. Added 2026-08-13: these geoms previously carried no friction at all and
    # inherited the scene-level default (1.0 0.01 0.001), which is shared with the
    # tabletop and the walls -- so a box could not be made grippier without also
    # changing the table. Sliding is set above the tabletop's 1.0 because MuJoCo takes
    # the element-wise MAX of the two contacting geoms, so anything <= 1.0 here is
    # masked by the table and by the 3.0 soft finger pads.
    collision_friction: Tuple[float, float, float] = (1.5, 0.02, 0.002)

    # gravity toggle
    GRAVITY: bool = True


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
    return rotation_matrix_to_quat_wxyz(euler_xyz_deg_to_rotmat(rx_deg, ry_deg, rz_deg))


def load_mesh_scaled(mesh_file: str, scale: Tuple[float, float, float]) -> trimesh.Trimesh:
    mesh = trimesh.load(mesh_file, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"Could not load mesh as Trimesh: {mesh_file}")

    verts = np.asarray(mesh.vertices).copy()
    verts[:, 0] *= scale[0]
    verts[:, 1] *= scale[1]
    verts[:, 2] *= scale[2]

    return trimesh.Trimesh(vertices=verts, faces=mesh.faces, process=False)


def estimate_box_frame(mesh: trimesh.Trimesh, auto_detect_axes: bool = True):
    """
    Returns:
        center_raw
        R_raw_from_local
        local_vertices
    """
    vertices = np.asarray(mesh.vertices)
    center_raw = np.asarray(mesh.bounding_box.centroid, dtype=float)
    centered = vertices - center_raw

    if not auto_detect_axes:
        R_raw_from_local = np.eye(3)
        local_vertices = centered.copy()
        return center_raw, R_raw_from_local, local_vertices

    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    R_raw_from_local = np.asarray(eigvecs[:, order], dtype=float)

    if np.linalg.det(R_raw_from_local) < 0:
        R_raw_from_local[:, 2] *= -1.0

    local_vertices = (R_raw_from_local.T @ centered.T).T
    return center_raw, R_raw_from_local, local_vertices


def estimate_box_size_and_center(local_vertices: np.ndarray):
    mins = local_vertices.min(axis=0)
    maxs = local_vertices.max(axis=0)
    local_center = 0.5 * (mins + maxs)
    full_size = maxs - mins
    return mins, maxs, local_center, full_size


def apply_local_frame_offset(
    local_pos: np.ndarray,
    R_local_geom: np.ndarray,
    frame_pos_offset_local: np.ndarray,
    frame_euler_deg_local: Tuple[float, float, float],
):
    R_offset = euler_xyz_deg_to_rotmat(*frame_euler_deg_local)
    local_pos_new = np.asarray(frame_pos_offset_local, dtype=float) + R_offset @ local_pos
    R_local_geom_new = R_offset @ R_local_geom
    return local_pos_new, R_local_geom_new


def local_to_body_geom(
    local_pos: np.ndarray,
    R_local_geom: np.ndarray,
    center_raw: np.ndarray,
    R_raw_from_local: np.ndarray,
    frame_pos_offset_local: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    frame_euler_deg_local: Tuple[float, float, float] = (0.0, 0.0, 0.0),
):
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


def make_collision_box_geom(
    cfg: TexturedBoxConfig,
    center_raw: np.ndarray,
    R_raw_from_local: np.ndarray,
    local_box_center: np.ndarray,
    full_size: np.ndarray,
) -> str:
    rgba = "0 1 0 0.18" if cfg.show_collision else "0 0 0 0"

    scaled_full_size = np.array([
        full_size[0] * cfg.collision_size_scale[0] + 2.0 * cfg.collision_padding[0],
        full_size[1] * cfg.collision_size_scale[1] + 2.0 * cfg.collision_padding[1],
        full_size[2] * cfg.collision_size_scale[2] + 2.0 * cfg.collision_padding[2],
    ], dtype=float)

    half_size = 0.5 * scaled_full_size
    R_local_geom = np.eye(3)

    pos_body, quat_body = local_to_body_geom(
        local_pos=local_box_center,
        R_local_geom=R_local_geom,
        center_raw=center_raw,
        R_raw_from_local=R_raw_from_local,
        frame_pos_offset_local=cfg.collision_pos_offset_local,
        frame_euler_deg_local=cfg.collision_euler_deg_local,
    )

    return (
        f'      <geom name="{cfg.name}_collision" type="box" '
        f'pos="{fmt(pos_body)}" quat="{fmt(quat_body)}" '
        f'size="{fmt(half_size)}" '
        f'mass="{cfg.mass:.6g}" '
        f'contype="1" conaffinity="1" '
        f'friction="{fmt(cfg.collision_friction)}" '
        f'rgba="{rgba}"/>'
    )


def make_visual_mesh_geom(cfg: TexturedBoxConfig) -> str:
    visual_quat = euler_xyz_deg_to_quat_wxyz(*cfg.visual_euler_deg_local)
    return (
        f'      <geom name="{cfg.name}_visual" type="mesh" '
        f'mesh="{cfg.mesh_name}" material="{cfg.material_name}" '
        f'pos="{fmt(cfg.visual_pos_offset_local)}" '
        f'quat="{fmt(visual_quat)}" '
        f'contype="0" conaffinity="0"/>'
    )


def build_textured_box(cfg: TexturedBoxConfig):
    collision_mesh = load_mesh_scaled(cfg.collision_mesh_file, cfg.collision_mesh_scale)

    center_raw, R_raw_from_local, local_vertices = estimate_box_frame(
        collision_mesh,
        auto_detect_axes=cfg.auto_detect_axes,
    )
    mins, maxs, local_box_center, full_size = estimate_box_size_and_center(local_vertices)

    visual_mesh_path = Path(cfg.visual_mesh_file).expanduser().resolve()
    texture_path = Path(cfg.texture_file).expanduser().resolve()
    collision_mesh_path = Path(cfg.collision_mesh_file).expanduser().resolve()

    asset_lines = [
        f'    <texture name="{cfg.texture_name}" type="2d" file="{texture_path.as_posix()}"/>',
        f'    <material name="{cfg.material_name}" texture="{cfg.texture_name}" specular="0.15" shininess="0.2"/>',
        f'    <mesh name="{cfg.mesh_name}" file="{visual_mesh_path.as_posix()}" scale="{fmt(cfg.visual_mesh_scale)}"/>',
    ]

    body_lines = [f'    <body name="{cfg.name}" pos="{fmt(cfg.body_pos)}">']

    if cfg.free_joint:
        body_lines.append(f'      <joint name="{cfg.name}_free" type="free"/>')

    body_lines.append(make_visual_mesh_geom(cfg))
    body_lines.append(
        make_collision_box_geom(
            cfg=cfg,
            center_raw=center_raw,
            R_raw_from_local=R_raw_from_local,
            local_box_center=local_box_center,
            full_size=full_size,
        )
    )

    if cfg.show_debug_collision_mesh:
        body_lines.append(
            f'      <geom name="{cfg.name}_debug_collision_mesh" type="mesh" '
            f'mesh="{cfg.mesh_name}" rgba="1 1 1 0.15" contype="0" conaffinity="0"/>'
        )

    body_lines.append("    </body>")

    if cfg.print_debug:
        print(f"\n[{cfg.name}]")
        print(f"  visual_mesh_file            = {visual_mesh_path}")
        print(f"  collision_mesh_file         = {collision_mesh_path}")
        print(f"  texture_file                = {texture_path}")
        print(f"  center_raw                  = {center_raw}")
        print("  R_raw_from_local =")
        print(R_raw_from_local)
        print(f"  local mins                  = {mins}")
        print(f"  local maxs                  = {maxs}")
        print(f"  local_box_center            = {local_box_center}")
        print(f"  full_size                   = {full_size}")
        print(f"  collision_pos_offset_local  = {cfg.collision_pos_offset_local}")
        print(f"  collision_euler_deg_local   = {cfg.collision_euler_deg_local}")
        print(f"  visual_pos_offset_local     = {cfg.visual_pos_offset_local}")
        print(f"  visual_euler_deg_local      = {cfg.visual_euler_deg_local}")

    return asset_lines, "\n".join(body_lines)


def generate_scene_xml(configs: List[TexturedBoxConfig]) -> str:
    if len(configs) == 0:
        raise ValueError("configs cannot be empty")

    gravity_enabled = any(cfg.GRAVITY for cfg in configs)
    gravity_line = '<option gravity="0 0 0"/>' if gravity_enabled else '<!-- <option gravity="0 0 0"/> -->'

    asset_lines = []
    body_blocks = []

    for cfg in configs:
        a, b = build_textured_box(cfg)
        asset_lines.extend(a)
        body_blocks.append(b)

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


if __name__ == "__main__":
    cracker = TexturedBoxConfig(
        name="cracker_box",
        visual_mesh_file="ycb/Boxes/Craker_Box/assets/textured.obj",
        collision_mesh_file="ycb/Boxes/Craker_Box/assets/craker_nontextured.stl",
        texture_file="ycb/Boxes/Craker_Box/assets/texture_map.png",
        body_pos=(0.45, -0.18, 0.25),

        mesh_name="cracker_mesh",
        texture_name="cracker_tex",
        material_name="cracker_mat",

        visual_mesh_scale=(1.0, 1.0, 1.0),
        collision_mesh_scale=(1.0, 1.0, 1.0),
        auto_detect_axes=True,

        visual_pos_offset_local=(0.0, 0.0, 0.0),
        visual_euler_deg_local=(0.0, 0.0, 0.0),

        collision_size_scale=(0.99, 0.975, 0.94),
        collision_padding=(0.0, 0.0, 0.0),
        collision_pos_offset_local=(0.001, 0.0, 0.001),
        collision_euler_deg_local=(0.0, 0.0, -1.0),

        show_collision=False,
        show_debug_collision_mesh=False,
        mass=0.511,
        free_joint=True,
        print_debug=True,
        GRAVITY=False,
    )

    sugar = TexturedBoxConfig(
        name="sugar_box",
        visual_mesh_file="ycb/Boxes/Sugar_Box/assets/textured.obj",
        collision_mesh_file="ycb/Boxes/Sugar_Box/assets/sugar_nontextured.stl",
        texture_file="ycb/Boxes/Sugar_Box/assets/texture_map.png",
        body_pos=(0.60, -0.18, 0.25),

        mesh_name="sugar_mesh",
        texture_name="sugar_tex",
        material_name="sugar_mat",

        visual_mesh_scale=(1.0, 1.0, 1.0),
        collision_mesh_scale=(1.0, 1.0, 1.0),
        auto_detect_axes=True,

        visual_pos_offset_local=(0.0, 0.0, 0.0),
        visual_euler_deg_local=(0.0, 0.0, 0.0),

        collision_size_scale=(0.99, 0.95, 0.965),
        collision_padding=(0.0, 0.0, 0.0),
        collision_pos_offset_local=(0.001, -0.0002, 0.0002),
        collision_euler_deg_local=(0.0, 0.0, -2.0),

        show_collision=False,
        show_debug_collision_mesh=False,
        mass=0.314,
        free_joint=True,
        print_debug=True,
        GRAVITY=False,
    )

    xml = generate_scene_xml([cracker, sugar])
    Path("generated_ycb_textured_boxes.xml").write_text(xml, encoding="utf-8")
    print("Saved: generated_ycb_textured_boxes.xml")