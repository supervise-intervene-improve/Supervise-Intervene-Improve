#!/usr/bin/env python3
"""Generate validated MuJoCo OOD scenes for the table tasks.

Run with an interpreter that has the `mujoco` Python package installed, for
example:

    conda run -n polymetis python generate_ood_scenes.py --num-scenes 50 --seed 0
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = REPO_ROOT / "mujoco_scenes" / "ood_scenes"

BASE_SCENES = {
    "t_shape": REPO_ROOT
    / "mujoco_scenes"
    / "working_scenes"
    / "with_soft_gripper"
    / "sii_scene_table_T_shape.xml",
    "boxes_cups": REPO_ROOT
    / "mujoco_scenes"
    / "working_scenes"
    / "with_soft_gripper"
    / "sii_scene_table_boxes_cups.xml",
    "wire_spoon": REPO_ROOT
    / "mujoco_scenes"
    / "working_scenes"
    / "with_soft_gripper"
    / "sii_scene_table_wire_base_and_spoon.xml",
}

INCLUDE_SOURCES = {
    "boxes": REPO_ROOT / "mujoco_scenes" / "generated_boxes" / "generated_ycb_textured_boxes.xml",
    "cups": REPO_ROOT / "mujoco_scenes" / "generated_cups" / "generated_cups_via_stl_4.xml",
    "wire_base": REPO_ROOT / "mujoco_scenes" / "generated_wire_game" / "generated_base_via_stl.xml",
    "spoon": REPO_ROOT / "mujoco_scenes" / "generated_wire_game" / "generated_spoon_via_stl.xml",
}

PANDA_MODEL = REPO_ROOT / "mujoco_scenes" / "generated_panda" / "sii_panda_model_soft_gripper.xml"

TASK_BODY_NAMES = {
    "t_shape": ["T1", "T2"],
    "boxes_cups": ["cracker_box", "sugar_box", "cup1", "cup2", "cup3", "cup4"],
    "wire_spoon": ["object", "spoon1"],
}

SCENE_PREFIX = {
    "t_shape": "t_shape_ood",
    "boxes_cups": "boxes_cups_ood",
    "wire_spoon": "wire_spoon_ood",
}

YCB_TEXTURES = {
    "cracker_original": REPO_ROOT
    / "mujoco_scenes"
    / "ycb"
    / "Boxes"
    / "Craker_Box"
    / "assets"
    / "texture_map.png",
    "sugar_original": REPO_ROOT
    / "mujoco_scenes"
    / "ycb"
    / "Boxes"
    / "Sugar_Box"
    / "assets"
    / "texture_map.png",
}

# Purpose-built OOD box textures (high-contrast synthetic patterns), 1024x1024 to match
# the downscaled YCB maps -- a 4096 pair costs ~100 MB in EVERY MjrContext, and a
# nine-window run holds ~19 of them. Originals kept alongside as *.4096.orig.png.
OOD_BOX_TEXTURES = {
    "cracker": REPO_ROOT / "mujoco_scenes" / "ood_scenes" / "boxes_cups" / "assets"
    / "cracker_box_texture_map.png",
    "sugar": REPO_ROOT / "mujoco_scenes" / "ood_scenes" / "boxes_cups" / "assets"
    / "sugar_box_texture_map.png",
}

XY_RANDOM_RANGE = 0.01
XY_BOUNDS = {"x": (0.40, 0.80), "y": (-0.25, 0.25)}
BOX_YAW_RANDOM_RANGE_RAD = math.radians(5.0)
CUP_Z_SCALE_RANGES = [(0.00075, 0.00085), (0.00110, 0.00125)]


@dataclass
class Candidate:
    scene_type: str
    index: int
    scene_seed: int
    main_tree: ET.ElementTree
    include_trees: dict[str, ET.ElementTree]
    final_include_relpaths: dict[str, str]
    temp_include_relpaths: dict[str, str]
    final_panda_relpath: str
    temp_panda_relpath: str
    params: dict[str, Any]


@dataclass
class ValidationResult:
    valid: bool
    reason: str
    contact_count: int = 0
    invalid_contacts: list[dict[str, Any]] | None = None


def parse_vec(value: str | None, expected_len: int) -> list[float]:
    if value is None:
        return [0.0] * expected_len
    parts = [float(part) for part in value.split()]
    if len(parts) != expected_len:
        raise ValueError(f"Expected {expected_len} values, got {value!r}")
    return parts


def format_vec(values: Iterable[float]) -> str:
    return " ".join(f"{value:.8g}" for value in values)


def sample_uniform(rng: random.Random, lo: float, hi: float) -> float:
    return rng.uniform(lo, hi)


def sample_union(rng: random.Random, ranges: list[tuple[float, float]]) -> float:
    lo, hi = rng.choice(ranges)
    return sample_uniform(rng, lo, hi)


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def read_tree(path: Path) -> ET.ElementTree:
    return ET.parse(path)


def panda_include_tree() -> ET.ElementTree:
    tree = read_tree(PANDA_MODEL)
    root = tree.getroot()
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    compiler.set("meshdir", str(REPO_ROOT / "mujoco_scenes" / "assets"))
    return tree


def write_tree(tree: ET.ElementTree, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=False)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n")


def find_body(root: ET.Element, body_name: str) -> ET.Element:
    for body in root.iter("body"):
        if body.get("name") == body_name:
            return body
    raise ValueError(f"Body {body_name!r} not found")


def set_body_pose(
    root: ET.Element,
    body_name: str,
    pos: tuple[float, float, float],
    euler: tuple[float, float, float] | None = None,
) -> None:
    body = find_body(root, body_name)
    body.set("pos", format_vec(pos))
    if euler is not None:
        body.set("euler", format_vec(euler))


def set_include(root: ET.Element, matcher: str, new_file: str) -> None:
    for include in root.iter("include"):
        include_file = include.get("file", "")
        if matcher in include_file:
            include.set("file", new_file)
            return
    raise ValueError(f"Include containing {matcher!r} not found")


def normalize_panda_include(root: ET.Element, include_file: str) -> None:
    for matcher in ("generated_panda", "sii_panda_model_soft_gripper.xml"):
        try:
            set_include(root, matcher, include_file)
            return
        except ValueError:
            continue
    raise ValueError("Panda include not found")


def set_temp_includes(candidate: Candidate) -> None:
    root = candidate.main_tree.getroot()
    for key, relpath in candidate.temp_include_relpaths.items():
        if key == "boxes":
            set_include(root, "generated_ycb_textured_boxes.xml", relpath)
        elif key == "cups":
            set_include(root, "generated_cups_via_stl_4.xml", relpath)
        elif key == "wire_base":
            set_include(root, "generated_base_via_stl.xml", relpath)
        elif key == "spoon":
            set_include(root, "generated_spoon_via_stl.xml", relpath)


def set_final_includes(candidate: Candidate) -> None:
    root = candidate.main_tree.getroot()
    normalize_panda_include(root, candidate.final_panda_relpath)
    for key, relpath in candidate.final_include_relpaths.items():
        if key == "boxes":
            set_include(root, "boxes.xml", relpath)
        elif key == "cups":
            set_include(root, "cups.xml", relpath)
        elif key == "wire_base":
            set_include(root, "base.xml", relpath)
        elif key == "spoon":
            set_include(root, "spoon.xml", relpath)


def sample_t_shape(index: int, scene_seed: int, final_panda_relpath: str, temp_panda_relpath: str) -> Candidate:
    rng = random.Random(scene_seed)
    tree = read_tree(BASE_SCENES["t_shape"])
    root = tree.getroot()
    normalize_panda_include(root, temp_panda_relpath)

    t1 = {
        "pos": (
            sample_uniform(rng, 0.5, 0.7),
            # Tracks the base scene's authored T1 y. 2026-08-13 shifted it -0.225 ->
            # -0.190 (+0.035, toward the table centre); 2026-08-14 reverted that, so it
            # is back at -0.225 and this band moved back with it. These are ABSOLUTE,
            # not offsets from the base body_pos, so they must move with it or a
            # regeneration would silently change the ID<->OOD separation in y.
            sample_uniform(rng, -0.350, -0.250),
            0.225,
        ),
        "euler": (
            1.5708,
            sample_union(rng, [(3.49159, 4.14159), (2.64159, 2.79159)]),
            0.0,
        ),
    }
    t2 = {
        "pos": (
            sample_uniform(rng, 0.55, 0.65),
            # Keep the red T-shape 3 cm closer to the table centre than the previous
            # 0.050--0.080 m band. This is an absolute band, not an offset from the
            # base body position, so it must stay aligned with the generated corpus.
            sample_uniform(rng, 0.020, 0.050),
            0.225,
        ),
        "euler": (
            1.5708,
            sample_union(rng, [(3.34159, 3.49159), (2.80159, 2.94159)]),
            0.0,
        ),
    }

    set_body_pose(root, "T1", t1["pos"], t1["euler"])
    set_body_pose(root, "T2", t2["pos"], t2["euler"])

    return Candidate(
        scene_type="t_shape",
        index=index,
        scene_seed=scene_seed,
        main_tree=tree,
        include_trees={"panda": panda_include_tree()},
        final_include_relpaths={"panda": final_panda_relpath},
        temp_include_relpaths={"panda": temp_panda_relpath},
        final_panda_relpath=final_panda_relpath,
        temp_panda_relpath=temp_panda_relpath,
        params={"T1": t1, "T2": t2},
    )


def swap_box_textures(boxes_root: ET.Element) -> dict[str, str]:
    """Assign the OOD box textures. (Name kept: it is referenced by the manifest writer.)

    Was a SWAP of the two YCB photos -- each box wore the other's texture. Changed
    2026-08-12 to dedicated high-contrast textures under `ood_scenes/boxes_cups/assets/`,
    because the swap was a weak signal (two similar cereal boxes) and could not be
    verified downstream: both YCB files are named `texture_map.png` inside a directory
    called `assets`, so a consumer comparing paths could not tell them apart.

    This function is the ONLY place the assignment is defined. It had to change here, not
    just in the 50 generated files: regenerating the corpus rewrites those files, which is
    exactly how the first attempt at this change was silently undone.
    """
    assignments = {
        "cracker_tex": str(OOD_BOX_TEXTURES["cracker"]),
        "sugar_tex": str(OOD_BOX_TEXTURES["sugar"]),
    }
    for texture in boxes_root.iter("texture"):
        name = texture.get("name")
        if name in assignments:
            texture.set("file", assignments[name])
    return assignments


def randomize_xy(rng: random.Random, base_pos: list[float]) -> tuple[float, float, float]:
    x = clamp(
        base_pos[0] + sample_uniform(rng, -XY_RANDOM_RANGE, XY_RANDOM_RANGE),
        *XY_BOUNDS["x"],
    )
    y = clamp(
        base_pos[1] + sample_uniform(rng, -XY_RANDOM_RANGE, XY_RANDOM_RANGE),
        *XY_BOUNDS["y"],
    )
    return (x, y, base_pos[2])


def randomize_box_pose(
    rng: random.Random, root: ET.Element, body_name: str
) -> dict[str, tuple[float, float, float]]:
    body = find_body(root, body_name)
    base_pos = parse_vec(body.get("pos"), 3)
    base_euler = parse_vec(body.get("euler"), 3)
    pos = randomize_xy(rng, base_pos)
    euler = (base_euler[0], base_euler[1], base_euler[2] + sample_uniform(rng, -BOX_YAW_RANDOM_RANGE_RAD, BOX_YAW_RANDOM_RANGE_RAD))
    set_body_pose(root, body_name, pos, euler)
    return {"pos": pos, "euler": euler}


def randomize_cup_pose(rng: random.Random, root: ET.Element, body_name: str) -> dict[str, tuple[float, float, float]]:
    body = find_body(root, body_name)
    base_pos = parse_vec(body.get("pos"), 3)
    pos = randomize_xy(rng, base_pos)
    set_body_pose(root, body_name, pos, None)
    return {"pos": pos}


def set_cup_z_scales(rng: random.Random, cups_root: ET.Element) -> dict[str, float]:
    z_scales: dict[str, float] = {}
    for cup_name in ("cup1", "cup2", "cup3", "cup4"):
        z_scale = sample_union(rng, CUP_Z_SCALE_RANGES)
        z_scales[cup_name] = z_scale
        prefix = f"{cup_name}_"
        for mesh in cups_root.iter("mesh"):
            mesh_name = mesh.get("name", "")
            if not mesh_name.startswith(prefix):
                continue
            scale = parse_vec(mesh.get("scale"), 3)
            scale[2] = z_scale
            mesh.set("scale", format_vec(scale))
    return z_scales


def sample_boxes_cups(
    index: int,
    scene_seed: int,
    include_dir: str,
    temp_include_dir: str,
    final_panda_relpath: str,
    temp_panda_relpath: str,
) -> Candidate:
    rng = random.Random(scene_seed)
    main_tree = read_tree(BASE_SCENES["boxes_cups"])
    main_root = main_tree.getroot()
    normalize_panda_include(main_root, temp_panda_relpath)

    boxes_tree = read_tree(INCLUDE_SOURCES["boxes"])
    boxes_root = boxes_tree.getroot()
    texture_assignments = swap_box_textures(boxes_root)

    cups_tree = read_tree(INCLUDE_SOURCES["cups"])
    cups_root = cups_tree.getroot()

    params: dict[str, Any] = {
        "texture_strategy": "dedicated_ood_box_textures",
        "texture_assignments": texture_assignments,
        "xy_random_range": XY_RANDOM_RANGE,
        "xy_bounds": XY_BOUNDS,
        "box_yaw_random_range_deg": 5.0,
        "cup_z_scale_interpretation": "third component of each cup visual/collision mesh scale",
        "objects": {},
        "cup_z_scales": set_cup_z_scales(rng, cups_root),
    }

    for body_name in ("cracker_box", "sugar_box"):
        params["objects"][body_name] = randomize_box_pose(rng, boxes_root, body_name)
    for body_name in ("cup1", "cup2", "cup3", "cup4"):
        params["objects"][body_name] = randomize_cup_pose(rng, cups_root, body_name)

    final_relpaths = {
        "panda": final_panda_relpath,
        "boxes": f"{include_dir}/boxes_cups_ood_{index:03d}_boxes.xml",
        "cups": f"{include_dir}/boxes_cups_ood_{index:03d}_cups.xml",
    }
    temp_relpaths = {
        "panda": temp_panda_relpath,
        "boxes": f"{temp_include_dir}/boxes_cups_ood_{index:03d}_boxes.xml",
        "cups": f"{temp_include_dir}/boxes_cups_ood_{index:03d}_cups.xml",
    }
    candidate = Candidate(
        scene_type="boxes_cups",
        index=index,
        scene_seed=scene_seed,
        main_tree=main_tree,
        include_trees={"panda": panda_include_tree(), "boxes": boxes_tree, "cups": cups_tree},
        final_include_relpaths=final_relpaths,
        temp_include_relpaths=temp_relpaths,
        final_panda_relpath=final_panda_relpath,
        temp_panda_relpath=temp_panda_relpath,
        params=params,
    )
    set_temp_includes(candidate)
    return candidate


def sample_wire_spoon(
    index: int,
    scene_seed: int,
    include_dir: str,
    temp_include_dir: str,
    final_panda_relpath: str,
    temp_panda_relpath: str,
) -> Candidate:
    rng = random.Random(scene_seed)
    main_tree = read_tree(BASE_SCENES["wire_spoon"])
    main_root = main_tree.getroot()
    normalize_panda_include(main_root, temp_panda_relpath)

    base_tree = read_tree(INCLUDE_SOURCES["wire_base"])
    base_root = base_tree.getroot()
    spoon_tree = read_tree(INCLUDE_SOURCES["spoon"])
    spoon_root = spoon_tree.getroot()

    wire_base = {
        "pos": (
            sample_uniform(rng, 0.52, 0.72),
            sample_uniform(rng, -0.20, 0.00),
            0.22,
        ),
        "euler": (
            0.0,
            0.0,
            sample_union(rng, [(0.25, 0.75), (-0.75, -0.25)]),
        ),
    }
    spoon = {
        "pos": (
            sample_uniform(rng, 0.5, 0.7),
            sample_uniform(rng, 0.00, 0.2),
            0.27,
        ),
        "euler": (
            1.5708,
            sample_union(rng, [(1.1708, 1.3708), (1.7708, 1.9708)]),
            1.5708,
        ),
    }

    set_body_pose(base_root, "object", wire_base["pos"], wire_base["euler"])
    set_body_pose(spoon_root, "spoon1", spoon["pos"], spoon["euler"])

    final_relpaths = {
        "panda": final_panda_relpath,
        "wire_base": f"{include_dir}/wire_spoon_ood_{index:03d}_base.xml",
        "spoon": f"{include_dir}/wire_spoon_ood_{index:03d}_spoon.xml",
    }
    temp_relpaths = {
        "panda": temp_panda_relpath,
        "wire_base": f"{temp_include_dir}/wire_spoon_ood_{index:03d}_base.xml",
        "spoon": f"{temp_include_dir}/wire_spoon_ood_{index:03d}_spoon.xml",
    }
    candidate = Candidate(
        scene_type="wire_spoon",
        index=index,
        scene_seed=scene_seed,
        main_tree=main_tree,
        include_trees={"panda": panda_include_tree(), "wire_base": base_tree, "spoon": spoon_tree},
        final_include_relpaths=final_relpaths,
        temp_include_relpaths=temp_relpaths,
        final_panda_relpath=final_panda_relpath,
        temp_panda_relpath=temp_panda_relpath,
        params={"object": wire_base, "spoon1": spoon},
    )
    set_temp_includes(candidate)
    return candidate


def build_candidate(
    scene_type: str,
    index: int,
    scene_seed: int,
    include_dir: str,
    temp_include_dir: str,
    final_panda_relpath: str,
    temp_panda_relpath: str,
) -> Candidate:
    if scene_type == "t_shape":
        return sample_t_shape(index, scene_seed, final_panda_relpath, temp_panda_relpath)
    if scene_type == "boxes_cups":
        return sample_boxes_cups(
            index,
            scene_seed,
            include_dir,
            temp_include_dir,
            final_panda_relpath,
            temp_panda_relpath,
        )
    if scene_type == "wire_spoon":
        return sample_wire_spoon(
            index,
            scene_seed,
            include_dir,
            temp_include_dir,
            final_panda_relpath,
            temp_panda_relpath,
        )
    raise ValueError(f"Unsupported scene type: {scene_type}")


def import_mujoco():
    try:
        import mujoco  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "The mujoco Python package is required for generation/validation. "
            "Run with the project environment, e.g. `conda run -n polymetis python "
            "generate_ood_scenes.py --num-scenes 50 --seed 0`."
        ) from exc
    return mujoco


def body_names_for_model(mujoco, model) -> set[str]:
    names: set[str] = set()
    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if name:
            names.add(name)
    return names


def infer_scene_type(mujoco, model) -> str:
    names = body_names_for_model(mujoco, model)
    for scene_type, required_names in TASK_BODY_NAMES.items():
        if all(name in names for name in required_names):
            return scene_type
    raise ValueError("Could not infer scene type from model body names")


def contact_record(mujoco, model, contact) -> dict[str, Any]:
    geom1 = int(contact.geom1)
    geom2 = int(contact.geom2)
    body1 = int(model.geom_bodyid[geom1])
    body2 = int(model.geom_bodyid[geom2])
    return {
        "dist": float(contact.dist),
        "geom1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom1),
        "geom2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom2),
        "body1": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1),
        "body2": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2),
    }


def is_allowed_table_support(record: dict[str, Any]) -> bool:
    return (
        (record["geom1"] == "table_top" and record["body2"] in sum(TASK_BODY_NAMES.values(), []))
        or (record["geom2"] == "table_top" and record["body1"] in sum(TASK_BODY_NAMES.values(), []))
    )


def validate_xml(
    xml_path: Path,
    scene_type: str | None = None,
    contact_tolerance: float = 1e-7,
) -> ValidationResult:
    mujoco = import_mujoco()
    try:
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
    except Exception as exc:  # MuJoCo raises several native-backed exceptions.
        return ValidationResult(False, f"MuJoCo load/forward failed: {type(exc).__name__}: {exc}")

    try:
        effective_scene_type = scene_type or infer_scene_type(mujoco, model)
    except Exception as exc:
        return ValidationResult(False, str(exc), contact_count=int(data.ncon))

    task_body_ids = set()
    for body_name in TASK_BODY_NAMES[effective_scene_type]:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            return ValidationResult(False, f"Missing task body {body_name!r}", contact_count=int(data.ncon))
        task_body_ids.add(int(body_id))

    invalid_contacts: list[dict[str, Any]] = []
    for contact_index in range(int(data.ncon)):
        contact = data.contact[contact_index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        body1 = int(model.geom_bodyid[geom1])
        body2 = int(model.geom_bodyid[geom2])
        if body1 == body2:
            continue
        if body1 not in task_body_ids and body2 not in task_body_ids:
            continue
        if float(contact.dist) > contact_tolerance:
            continue
        record = contact_record(mujoco, model, contact)
        if is_allowed_table_support(record):
            continue
        invalid_contacts.append(record)

    if invalid_contacts:
        return ValidationResult(
            False,
            f"{len(invalid_contacts)} invalid task-object contact(s)",
            contact_count=int(data.ncon),
            invalid_contacts=invalid_contacts,
        )
    return ValidationResult(True, "ok", contact_count=int(data.ncon), invalid_contacts=[])


def candidate_paths(output_root: Path, scene_type: str, index: int) -> tuple[Path, Path, Path, str, str]:
    scene_dir = output_root / scene_type
    temp_dir = scene_dir / ".tmp_generate_ood"
    include_dir = "includes"
    temp_include_dir = ".tmp_generate_ood/includes"
    filename = f"{SCENE_PREFIX[scene_type]}_{index:03d}.xml"
    return scene_dir / filename, temp_dir / filename, scene_dir / include_dir, include_dir, temp_include_dir


def write_candidate(candidate: Candidate, main_path: Path, include_dir: Path, use_final_includes: bool) -> None:
    working_candidate = copy.deepcopy(candidate)
    if use_final_includes and working_candidate.final_include_relpaths:
        set_final_includes(working_candidate)
    for key, tree in working_candidate.include_trees.items():
        relpath = (
            working_candidate.final_include_relpaths[key]
            if use_final_includes
            else working_candidate.temp_include_relpaths[key]
        )
        write_tree(tree, main_path.parent / relpath)
    write_tree(working_candidate.main_tree, main_path)


def cleanup_temp(scene_dir: Path) -> None:
    temp_dir = scene_dir / ".tmp_generate_ood"
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


def flatten_for_csv(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            flatten_for_csv(f"{prefix}.{key}" if prefix else str(key), child, out)
    elif isinstance(value, (list, tuple)):
        out[prefix] = " ".join(f"{item:.8g}" if isinstance(item, float) else str(item) for item in value)
    else:
        out[prefix] = value


def manifest_row(record: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "scene_type": record["scene_type"],
        "index": record["index"],
        "scene": record["scene"],
        "global_seed": record["global_seed"],
        "scene_seed": record["scene_seed"],
        "attempts": record["attempts"],
        "contacts": record["validation"]["contact_count"],
    }
    flatten_for_csv("params", record["params"], row)
    return row


def write_manifest(output_root: Path, records: list[dict[str, Any]], report: dict[str, Any]) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_json = output_root / "manifest.json"
    manifest_csv = output_root / "manifest.csv"
    report_json = output_root / "validation_report.json"

    manifest_json.write_text(json.dumps(records, indent=2), encoding="utf-8")
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    rows = [manifest_row(record) for record in records]
    fieldnames = sorted({key for row in rows for key in row})
    with manifest_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_readme(output_root: Path) -> None:
    readme = output_root / "README.md"
    readme.write_text(
        """# MuJoCo OOD Scenes

Generated scenes are grouped by task:

- `t_shape/`
- `boxes_cups/`
- `wire_spoon/`

Commands:

```bash
conda run -n polymetis python generate_ood_scenes.py --num-scenes 30 --seed 0
conda run -n polymetis python generate_ood_scenes.py --num-scenes 50 --seed 0
conda run -n polymetis python generate_ood_scenes.py --validate-only mujoco_scenes/ood_scenes/t_shape/t_shape_ood_001.xml
```

The boxes/cups generator interprets the requested cup Z-scale intervals as the
third component of each cup visual and collision mesh `scale` attribute. Cup
X/Y mesh scales are preserved so the top-down cup footprint stays circular and
the red-to-green and blue-to-yellow radial fit relationships are unchanged.

`manifest.json` and `manifest.csv` record every accepted sample.
`validation_report.json` records request counts, rejection counts, and the
assumptions used by the generator.
""",
        encoding="utf-8",
    )


def generate(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    master_rng = random.Random(args.seed)
    scene_types = args.scene_type if args.scene_type else ["t_shape", "boxes_cups", "wire_spoon"]
    records: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "requested_per_scene_type": args.num_scenes,
        "global_seed": args.seed,
        "scene_types": scene_types,
        "generated": {scene_type: 0 for scene_type in scene_types},
        "rejected": {scene_type: 0 for scene_type in scene_types},
        "failures": [],
        "assumptions": [
            "Cup Z-scale intervals are applied to the third mesh scale component for visual and collision meshes.",
            "Cup X/Y scales are preserved, keeping the top-down circular footprint and pair fit relationships.",
            "Boxes reuse local YCB textures by swapping cracker and sugar texture maps; no external assets are downloaded.",
            "Object-table contacts involving table_top are allowed as support contacts; task contacts with other objects, robot, walls, floor, or table legs are rejected.",
            "Boxes/cups XY and box-yaw randomization use demo_collector/config.py ranges: +/-0.01 m XY and +/-5 degrees yaw for boxes.",
            "Each generated scene includes a generated copy of the Panda XML with absolute meshdir pointing at mujoco_scenes/assets; the original Panda XML is not modified.",
        ],
    }

    for scene_type in scene_types:
        generated = 0
        attempts_for_type = 0
        while generated < args.num_scenes:
            index = generated + 1
            final_path, temp_path, _include_path, include_dir, temp_include_dir = candidate_paths(
                output_root, scene_type, index
            )
            cleanup_temp(final_path.parent)

            accepted: tuple[Candidate, ValidationResult, int] | None = None
            last_failure: ValidationResult | None = None
            for attempt in range(1, args.max_attempts + 1):
                attempts_for_type += 1
                scene_seed = master_rng.randrange(0, 2**32)
                candidate = build_candidate(
                    scene_type,
                    index,
                    scene_seed,
                    include_dir,
                    temp_include_dir,
                    f"{include_dir}/sii_panda_model_soft_gripper.xml",
                    f"{temp_include_dir}/sii_panda_model_soft_gripper.xml",
                )
                write_candidate(candidate, temp_path, temp_path.parent / "includes", use_final_includes=False)
                validation = validate_xml(temp_path, scene_type=scene_type, contact_tolerance=args.contact_tolerance)
                if validation.valid:
                    accepted = (candidate, validation, attempt)
                    break
                last_failure = validation
                report["rejected"][scene_type] += 1
                cleanup_temp(final_path.parent)

            if accepted is None:
                reason = last_failure.reason if last_failure else "no candidate sampled"
                report["failures"].append(
                    {
                        "scene_type": scene_type,
                        "index": index,
                        "reason": reason,
                        "max_attempts": args.max_attempts,
                    }
                )
                write_manifest(output_root, records, report)
                write_readme(output_root)
                print(
                    f"Failed to generate {scene_type} scene {index:03d} after "
                    f"{args.max_attempts} attempts: {reason}",
                    file=sys.stderr,
                )
                return 1

            candidate, validation, attempts = accepted
            write_candidate(candidate, final_path, final_path.parent / include_dir, use_final_includes=True)
            final_validation = validate_xml(final_path, scene_type=scene_type, contact_tolerance=args.contact_tolerance)
            if not final_validation.valid:
                report["failures"].append(
                    {
                        "scene_type": scene_type,
                        "index": index,
                        "reason": f"Final saved XML failed validation: {final_validation.reason}",
                    }
                )
                write_manifest(output_root, records, report)
                write_readme(output_root)
                print(report["failures"][-1]["reason"], file=sys.stderr)
                return 1

            cleanup_temp(final_path.parent)
            generated += 1
            report["generated"][scene_type] = generated
            records.append(
                {
                    "scene_type": scene_type,
                    "index": index,
                    "scene": str(final_path.relative_to(REPO_ROOT)),
                    "global_seed": args.seed,
                    "scene_seed": candidate.scene_seed,
                    "attempts": attempts,
                    "params": candidate.params,
                    "validation": {
                        "reason": validation.reason,
                        "contact_count": final_validation.contact_count,
                    },
                }
            )

        print(
            f"{scene_type}: requested={args.num_scenes} generated={generated} "
            f"rejected={report['rejected'][scene_type]} attempts={attempts_for_type}"
        )

    write_manifest(output_root, records, report)
    write_readme(output_root)
    print(f"Wrote {len(records)} scenes under {output_root.relative_to(REPO_ROOT)}")
    print(f"Manifest: {(output_root / 'manifest.json').relative_to(REPO_ROOT)}")
    print(f"Validation report: {(output_root / 'validation_report.json').relative_to(REPO_ROOT)}")
    return 0


def validate_only(args: argparse.Namespace) -> int:
    result = validate_xml(args.validate_only.resolve(), scene_type=args.validate_scene_type)
    print(
        json.dumps(
            {
                "xml": str(args.validate_only),
                "valid": result.valid,
                "reason": result.reason,
                "contact_count": result.contact_count,
                "invalid_contacts": result.invalid_contacts or [],
            },
            indent=2,
        )
    )
    return 0 if result.valid else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-scenes", type=int, default=50, help="Number of scenes to generate per selected scene type.")
    parser.add_argument("--seed", type=int, default=0, help="Global random seed.")
    parser.add_argument(
        "--scene-type",
        action="append",
        choices=["t_shape", "boxes_cups", "wire_spoon"],
        help="Scene type to generate. Repeat for multiple types. Defaults to all.",
    )
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT, help="Directory for generated OOD scenes.")
    parser.add_argument("--max-attempts", type=int, default=200, help="Bounded sampling attempts per output scene.")
    parser.add_argument(
        "--contact-tolerance",
        type=float,
        default=1e-7,
        help="Contacts at or below this distance are treated as collisions unless explicitly allowed.",
    )
    parser.add_argument("--validate-only", type=Path, help="Load and validate one XML instead of generating scenes.")
    parser.add_argument(
        "--validate-scene-type",
        choices=["t_shape", "boxes_cups", "wire_spoon"],
        help="Optional scene type for --validate-only. Inferred from body names if omitted.",
    )
    args = parser.parse_args()
    if args.num_scenes <= 0:
        parser.error("--num-scenes must be positive")
    if args.max_attempts <= 0:
        parser.error("--max-attempts must be positive")
    return args


def main() -> int:
    args = parse_args()
    if args.validate_only:
        return validate_only(args)
    return generate(args)


if __name__ == "__main__":
    raise SystemExit(main())
