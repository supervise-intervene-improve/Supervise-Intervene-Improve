import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import trimesh


@dataclass
class SpoonRingConfig:
    name: str
    mesh_file: str

    body_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    body_euler_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    mesh_name: str = "spoon_mesh"
    material_name: str = "spoon_green"
    mesh_scale: Tuple[float, float, float] = (0.001, 0.001, 0.001)
    rgba: str = "0.223 0.392 0.278 1"

    use_identity_frame: bool = True

    collision_frame_pos_offset_local: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    collision_frame_euler_deg_local: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    visual_pos_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    visual_euler_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    ring_center_local_xy: Tuple[float, float] = (0.0, 0.0)
    ring_radius: float = 0.0345
    ring_z_center: float = 0.010
    ring_start_deg: float = 0.0
    ring_end_deg: float = 360.0
    ring_num_segments: int = 16

    ring_capsule_radius: float = 0.003
    ring_capsule_half_length: float = 0.0055

    handle_pos_local: Tuple[float, float, float] = (0.08, 0.0, 0.010)
    handle_size: Tuple[float, float, float] = (0.05, 0.016, 0.010)
    handle_euler_deg_local: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    top_tab_enabled: bool = True
    top_tab_pos_local: Tuple[float, float, float] = (-0.042, 0.00, 0.010)
    top_tab_size: Tuple[float, float, float] = (0.0305, 0.004, 0.010)
    top_tab_euler_deg_local: Tuple[float, float, float] = (0.0, 0.0, 90.0)

    top_connector_enabled: bool = True
    top_connector_pos_local: Tuple[float, float, float] = (-0.035, 0.00, 0.010)
    top_connector_size: Tuple[float, float, float] = (0.01, 0.0045, 0.005)
    top_connector_euler_deg_local: Tuple[float, float, float] = (0.0, 0.0, 90.0)

    collision_class_name: str = "spoon_collision"
    collision_rgba: str = "0 0 1 1"
    collision_contype: int = 1
    collision_conaffinity: int = 1
    collision_friction: Tuple[float, float, float] = (0.03, 0.001, 0.00005)
    collision_solref: Tuple[float, float] = (0.01, 1.0)
    collision_solimp: Tuple[float, float, float, float, float] = (0.95, 0.99, 0.001, 0.5, 2.0)
    collision_condim: int = 3
    collision_priority: int = 1

    show_collision: bool = True
    free_joint: bool = True
    gravity_off: bool = False


def fmt(vals) -> str:
    if isinstance(vals, (list, tuple, np.ndarray)):
        return " ".join(f"{float(v):.6g}" for v in vals)
    return f"{float(vals):.6g}"


def rotation_matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = R
    q = trimesh.transformations.quaternion_from_matrix(M)
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
    frame_pos_offset_local: np.ndarray = None,
    frame_euler_deg_local: Tuple[float, float, float] = (0.0, 0.0, 0.0),
):
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


def make_box_geom(
    name: str,
    local_pos,
    local_size,
    local_euler_deg,
    center_raw,
    R_raw_from_local,
    frame_pos_offset_local,
    frame_euler_deg_local,
    collision_class_name: str,
):
    R_local_geom = euler_xyz_deg_to_rotmat(*local_euler_deg)
    pos_body, quat_body = local_to_body_geom(
        local_pos=np.array(local_pos, dtype=float),
        R_local_geom=R_local_geom,
        center_raw=center_raw,
        R_raw_from_local=R_raw_from_local,
        frame_pos_offset_local=frame_pos_offset_local,
        frame_euler_deg_local=frame_euler_deg_local,
    )

    return (
        f'      <geom name="{name}" type="box" '
        f'pos="{fmt(pos_body)}" quat="{fmt(quat_body)}" '
        f'size="{fmt(local_size)}" class="{collision_class_name}"/>'
    )


def make_capsule_geom(
    name: str,
    local_pos,
    radius: float,
    half_length: float,
    local_euler_deg,
    center_raw,
    R_raw_from_local,
    frame_pos_offset_local,
    frame_euler_deg_local,
    collision_class_name: str,
):
    R_local_geom = euler_xyz_deg_to_rotmat(*local_euler_deg)
    pos_body, quat_body = local_to_body_geom(
        local_pos=np.array(local_pos, dtype=float),
        R_local_geom=R_local_geom,
        center_raw=center_raw,
        R_raw_from_local=R_raw_from_local,
        frame_pos_offset_local=frame_pos_offset_local,
        frame_euler_deg_local=frame_euler_deg_local,
    )

    return (
        f'      <geom name="{name}" type="capsule" '
        f'pos="{fmt(pos_body)}" quat="{fmt(quat_body)}" '
        f'size="{fmt((radius, half_length))}" class="{collision_class_name}"/>'
    )


def make_ring_capsule_geoms(
    cfg: SpoonRingConfig,
    center_raw: np.ndarray,
    R_raw_from_local: np.ndarray,
    frame_pos_offset_local: np.ndarray,
    frame_euler_deg_local: Tuple[float, float, float],
):
    geoms = []

    cx, cy = cfg.ring_center_local_xy
    zc = cfg.ring_z_center

    if cfg.ring_num_segments < 3:
        raise ValueError("ring_num_segments should be at least 3")

    total_span = cfg.ring_end_deg - cfg.ring_start_deg
    angles_deg = [
        cfg.ring_start_deg + i * total_span / cfg.ring_num_segments
        for i in range(cfg.ring_num_segments)
    ]

    for i, ang_deg in enumerate(angles_deg):
        ang = math.radians(ang_deg)

        local_pos = (
            cx + cfg.ring_radius * math.cos(ang),
            cy + cfg.ring_radius * math.sin(ang),
            zc,
        )

        tangent_deg = ang_deg + 90.0

        geoms.append(
            make_capsule_geom(
                name=f"{cfg.name}_ring_{i:02d}",
                local_pos=local_pos,
                radius=cfg.ring_capsule_radius,
                half_length=cfg.ring_capsule_half_length,
                local_euler_deg=(0.0, 0.0, tangent_deg),
                center_raw=center_raw,
                R_raw_from_local=R_raw_from_local,
                frame_pos_offset_local=frame_pos_offset_local,
                frame_euler_deg_local=frame_euler_deg_local,
                collision_class_name=cfg.collision_class_name,
            )
        )

    return geoms


def generate_spoon_xml(cfg: SpoonRingConfig) -> str:
    if cfg.use_identity_frame:
        center_raw = np.zeros(3, dtype=float)
        R_raw_from_local = np.eye(3)
    else:
        center_raw = np.zeros(3, dtype=float)
        R_raw_from_local = np.eye(3)

    frame_pos_offset_local = np.asarray(cfg.collision_frame_pos_offset_local, dtype=float)
    frame_euler_deg_local = cfg.collision_frame_euler_deg_local
    visual_quat = euler_xyz_deg_to_quat_wxyz(*cfg.visual_euler_deg)

    gravity_line = '<option gravity="0 0 0"/>' if cfg.gravity_off else ""

    body_lines = [
    f'    <body name="{cfg.name}" pos="{fmt(cfg.body_pos)}" euler="{fmt([math.radians(v) for v in cfg.body_euler_deg])}">'
    ]

    if cfg.free_joint:
        body_lines.append(f'      <joint name="{cfg.name}_free" type="free"/>')

    body_lines.append(
        f'      <geom name="{cfg.name}_visual" type="mesh" mesh="{cfg.mesh_name}" '
        f'material="{cfg.material_name}" contype="0" conaffinity="0" '
        f'pos="{fmt(cfg.visual_pos_offset)}" quat="{fmt(visual_quat)}"/>'
    )

    body_lines.append(
        make_box_geom(
            name=f"{cfg.name}_handle",
            local_pos=cfg.handle_pos_local,
            local_size=cfg.handle_size,
            local_euler_deg=cfg.handle_euler_deg_local,
            center_raw=center_raw,
            R_raw_from_local=R_raw_from_local,
            frame_pos_offset_local=frame_pos_offset_local,
            frame_euler_deg_local=frame_euler_deg_local,
            collision_class_name=cfg.collision_class_name,
        )
    )

    body_lines.extend(
        make_ring_capsule_geoms(
            cfg=cfg,
            center_raw=center_raw,
            R_raw_from_local=R_raw_from_local,
            frame_pos_offset_local=frame_pos_offset_local,
            frame_euler_deg_local=frame_euler_deg_local,
        )
    )

    if cfg.top_connector_enabled:
        body_lines.append(
            make_box_geom(
                name=f"{cfg.name}_top_connector",
                local_pos=cfg.top_connector_pos_local,
                local_size=cfg.top_connector_size,
                local_euler_deg=cfg.top_connector_euler_deg_local,
                center_raw=center_raw,
                R_raw_from_local=R_raw_from_local,
                frame_pos_offset_local=frame_pos_offset_local,
                frame_euler_deg_local=frame_euler_deg_local,
                collision_class_name=cfg.collision_class_name,
            )
        )

    if cfg.top_tab_enabled:
        body_lines.append(
            make_box_geom(
                name=f"{cfg.name}_top_tab",
                local_pos=cfg.top_tab_pos_local,
                local_size=cfg.top_tab_size,
                local_euler_deg=cfg.top_tab_euler_deg_local,
                center_raw=center_raw,
                R_raw_from_local=R_raw_from_local,
                frame_pos_offset_local=frame_pos_offset_local,
                frame_euler_deg_local=frame_euler_deg_local,
                collision_class_name=cfg.collision_class_name,
            )
        )

    body_lines.append("    </body>")

    mesh_path = Path(cfg.mesh_file).expanduser().resolve()

    xml = f"""<mujoco model="{cfg.name}">
  {gravity_line}

  <asset>
    <material name="{cfg.material_name}" rgba="{cfg.rgba}"/>
    <mesh name="{cfg.mesh_name}" file="{mesh_path.as_posix()}" scale="{fmt(cfg.mesh_scale)}"/>
  </asset>

  <default>
    <default class="{cfg.collision_class_name}">
      <geom rgba="{cfg.collision_rgba}"
      contype="{cfg.collision_contype}" conaffinity="{cfg.collision_conaffinity}" friction="{fmt(cfg.collision_friction)}" solref="{fmt(cfg.collision_solref)}" solimp="{fmt(cfg.collision_solimp)}" condim="{cfg.collision_condim}" priority="{cfg.collision_priority}"/>
    </default>
  </default>

  <worldbody>
{chr(10).join(body_lines)}
  </worldbody>
</mujoco>
"""
    return xml


if __name__ == "__main__":
    cfg = SpoonRingConfig(
        name="spoon1",
        mesh_file="/path/to/Intervention_IL_AR/mujoco/franka_emika_panda/wire_and_loop_assets/Spoon_wider.stl",
        body_pos=(0.4, -0.10, 0.3),
        body_euler_deg=(90.0, 0, 90.0),

        mesh_name="spoon_mesh",
        material_name="spoon_green",
        mesh_scale=(0.001, 0.001, 0.001),
        rgba="0.223 0.392 0.278 1",

        use_identity_frame=True,

        collision_frame_pos_offset_local=(0.0, 0.0, 0.0),
        collision_frame_euler_deg_local=(0.0, 0.0, 0.0),

        visual_pos_offset=(0.0, 0.0, 0.0),
        visual_euler_deg=(0.0, 0.0, 0.0),

        ring_center_local_xy=(0.0, 0.0),
        ring_radius=0.0345,
        ring_z_center=0.010,
        ring_start_deg=0.0,
        ring_end_deg=360.0,
        ring_num_segments=16,
        ring_capsule_radius=0.005,
        ring_capsule_half_length=0.0055,

        handle_pos_local=(0.08, 0.0, 0.010),
        handle_size=(0.05, 0.016, 0.010),
        handle_euler_deg_local=(0.0, 0.0, 0.0),

        top_tab_enabled=True,
        top_tab_pos_local=(-0.042, 0.00, 0.010),
        top_tab_size=(0.0305, 0.004, 0.010),
        top_tab_euler_deg_local=(0.0, 0.0, 90.0),

        top_connector_enabled=True,
        top_connector_pos_local=(-0.035, 0.00, 0.010),
        top_connector_size=(0.01, 0.0045, 0.005),
        top_connector_euler_deg_local=(0.0, 0.0, 90.0),

        collision_class_name="spoon_collision",
        collision_rgba="0 0 1 1",
        collision_contype=1,
        collision_conaffinity=1,
        collision_friction=(0.03, 0.001, 0.00005),
        collision_solref=(0.01, 1.0),
        collision_solimp=(0.95, 0.99, 0.001, 0.5, 2.0),
        collision_condim=3,
        collision_priority=1,

        show_collision=True,
        free_joint=True,
        gravity_off=False,
    )

    xml = generate_spoon_xml(cfg)
    out_path = Path("generated_spoon_wider_capsule.xml")
    out_path.write_text(xml, encoding="utf-8")
    print(f"Saved: {out_path}")