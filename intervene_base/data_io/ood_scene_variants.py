"""Read model-variant PARAMETERS out of the pre-generated OOD scene corpus.

Sibling of `ood_scene_poses.py`, which reads the same corpus for object poses. That module
covers `t_shape`, where the OOD signal is entirely free-joint pose. This one covers
`boxes_cups`, where it is not:

  * each cup's mesh `scale[2]` varies 0.000752-0.001247 (base 0.001), drawn from two bands
    with an empty gap between them (`generate_ood_scenes.py:85-88`);
  * the two box textures are swapped with each other -- deterministic and byte-identical in
    all 50 scenes (`generate_ood_scenes.py:275-284`);
  * poses vary by only +/-0.01 m and +/-5 deg, i.e. INSIDE the ordinary in-distribution
    jitter, so poses alone do NOT make a cups episode out of distribution.

So we extract the first two as a *descriptor* and let `model_variants/builder.py` apply
them to the LIVE base scene. We never compile a corpus XML -- see `ood_scene_poses.py`'s
docstring for the full argument, but the short version is that the corpus files carry
absolute `/home/user/...` paths, lack the base scene's `multiccd` flag, and carry a stale
180-degree-rolled `front` camera. Reading parameters sidesteps all three, and keeps
`LAB_FAST` working since the parameters are geometry-independent.

Only `scale[2]` is ever read. X and Y are deliberately ignored and taken from the base
spec, because they differ per cup AND between a cup's visual and collision meshes
(`cup1_visual` 0.00075 vs `cup1_col_*` 0.0007725). Anything that assumed one XY per cup
would silently deform the collision hull.

DELIBERATELY STDLIB-ONLY, for the same reasons as `ood_scene_poses.py`: importing `mujoco`
here would let a caller compile a corpus file by accident.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional, Tuple

from data_io.ood_scene_poses import TASK_DIRS, iter_include_targets, list_scenes
from model_variants.descriptor import (
    file_sha256,
    make_descriptor,
    map_corpus_asset_path,
    repo_relative,
)

# Tasks whose OOD signal is model-level. Every other task returns `None`, which means
# "pose-only OOD" and leaves t_shape/wire_spoon on exactly the code path they use today.
MODEL_VARIANT_TASKS = ("cups",)

# `cupN_visual` / `cupN_col_<i>` -- the naming the cups include uses for all 14 meshes of a
# cup. Anchored so a stray `cup1x_visual` cannot match.
_CUP_MESH_RE = re.compile(r"^(cup\d+)_(?:visual|col_\d+)$")

# The two box textures the corpus swaps. Named explicitly: a corpus that swapped some other
# texture is a corpus we do not understand and must refuse rather than guess at.
_SWAPPED_TEXTURES = ("cracker_tex", "sugar_tex")

_variant_cache: Dict[str, Tuple[Optional[dict], dict]] = {}
_warned: set = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        print(f"[OODVariant][WARN] {message}")


def _scene_files(xml_path: Path, task_dir: Path):
    """The scene plus the per-scene includes it owns (containment-checked upstream)."""
    yield xml_path
    yield from iter_include_targets(xml_path, task_dir)


def _collect_raw(xml_path: Path, task_dir: Path):
    """`({mesh_name: (sx, sy, sz)}, {texture_name: raw_file_string})` across the scene."""
    meshes: Dict[str, Tuple[float, float, float]] = {}
    textures: Dict[str, str] = {}
    for path in _scene_files(xml_path, task_dir):
        try:
            root = ET.parse(path).getroot()
        except Exception as exc:
            _warn_once(f"parse:{path}", f"could not parse {path}: {exc}")
            return None, None
        for mesh in root.iter("mesh"):
            name = mesh.get("name")
            if not name or not _CUP_MESH_RE.match(name):
                continue
            parts = (mesh.get("scale") or "").replace(",", " ").split()
            if len(parts) != 3:
                _warn_once(f"scale:{path}:{name}",
                           f"{path.name}: mesh {name!r} has no usable scale; refusing scene")
                return None, None
            try:
                meshes[name] = tuple(float(p) for p in parts)
            except ValueError:
                _warn_once(f"scaleval:{path}:{name}",
                           f"{path.name}: mesh {name!r} scale is not numeric; refusing scene")
                return None, None
        for tex in root.iter("texture"):
            name = tex.get("name")
            if name in _SWAPPED_TEXTURES and tex.get("file"):
                textures[name] = tex.get("file")
    return meshes, textures


def _cup_scale_z(meshes: Dict[str, Tuple[float, float, float]], label: str) -> Optional[Dict[str, float]]:
    """One Z per cup, refusing any cup whose 14 meshes disagree.

    The generator sets a single Z on every mesh of a cup, so disagreement means the corpus
    is not what we think it is. Averaging would produce a plausible-looking model with a
    collision hull that does not match its visual -- refuse instead.
    """
    per_cup: Dict[str, Dict[str, float]] = {}
    for name, scale in meshes.items():
        cup = _CUP_MESH_RE.match(name).group(1)
        per_cup.setdefault(cup, {})[name] = scale[2]
    if not per_cup:
        _warn_once(f"nocups:{label}", f"{label}: no cup meshes found; refusing scene")
        return None

    out: Dict[str, float] = {}
    for cup, by_mesh in sorted(per_cup.items()):
        values = set(by_mesh.values())
        if len(values) != 1:
            _warn_once(
                f"zdisagree:{label}:{cup}",
                f"{label}: {cup} meshes disagree on scale[2] ({sorted(values)}); refusing scene",
            )
            return None
        out[cup] = float(next(iter(values)))
    return out


def scene_variant(xml_path, task_key, task_dir=None, *, base_scene_sha256="",
                  mujoco_scenes_root=None):
    """`(descriptor_or_None, provenance)` for one corpus scene.

    Returns `(None, {})` for any task not in `MODEL_VARIANT_TASKS` -- that is what keeps
    t_shape on its existing pose-only path -- and also whenever the scene cannot be read
    safely. A refused scene is dropped from the pool, never silently degraded, because a
    degraded cups "OOD" episode is exactly the mislabelled data this whole design exists to
    prevent.
    """
    if str(task_key).strip().lower() not in MODEL_VARIANT_TASKS:
        return None, {}

    xml_path = Path(xml_path)
    cache_key = f"{xml_path.resolve()}|{base_scene_sha256}"
    cached = _variant_cache.get(cache_key)
    if cached is not None:
        return cached

    task_dir = Path(task_dir) if task_dir is not None else xml_path.parent
    if mujoco_scenes_root is None:
        # <intervene_base>/mujoco_scenes/ood_scenes/<task>/scene.xml -> mujoco_scenes/
        mujoco_scenes_root = task_dir.resolve().parent.parent
    mujoco_scenes_root = Path(mujoco_scenes_root).resolve()

    meshes, textures = _collect_raw(xml_path, task_dir)
    if meshes is None:
        _variant_cache[cache_key] = (None, {})
        return None, {}

    cup_scale_z = _cup_scale_z(meshes, xml_path.name)
    if cup_scale_z is None:
        _variant_cache[cache_key] = (None, {})
        return None, {}

    missing = [t for t in _SWAPPED_TEXTURES if t not in textures]
    if missing:
        _warn_once(f"tex:{xml_path.name}",
                   f"{xml_path.name}: missing texture(s) {missing}; refusing scene")
        _variant_cache[cache_key] = (None, {})
        return None, {}

    # Untrusted absolute paths -> re-rooted, containment-checked, existence-checked.
    texture_file: Dict[str, str] = {}
    asset_sha: Dict[str, str] = {}
    for name in _SWAPPED_TEXTURES:
        resolved = map_corpus_asset_path(textures[name], mujoco_scenes_root)
        if resolved is None:
            _warn_once(f"asset:{xml_path.name}:{name}",
                       f"{xml_path.name}: texture {name!r} path refused; refusing scene")
            _variant_cache[cache_key] = (None, {})
            return None, {}
        rel = repo_relative(resolved, mujoco_scenes_root)
        texture_file[name] = rel
        asset_sha[rel] = file_sha256(resolved)

    descriptor = make_descriptor(
        task="cups",
        base_scene_sha256=base_scene_sha256,
        cup_scale_z=cup_scale_z,
        texture_file=texture_file,
    )
    provenance = {
        "corpus_scene": xml_path.name,
        "corpus_scene_sha256": file_sha256(xml_path),
        "corpus_include_sha256": {
            p.name: file_sha256(p) for p in iter_include_targets(xml_path, task_dir)
        },
        "asset_sha256": asset_sha,
        "corpus_root": str(task_dir),
    }
    _variant_cache[cache_key] = (descriptor, provenance)
    return descriptor, provenance


def list_variant_scenes(task_key, corpus_root):
    """Corpus scenes for a task. Thin pass-through so callers need one import."""
    return list_scenes(task_key, corpus_root)


def task_dir_for(task_key, corpus_root) -> Optional[Path]:
    subdir = TASK_DIRS.get(str(task_key).strip().lower())
    return None if subdir is None else Path(corpus_root) / subdir


def clear_cache() -> None:
    _variant_cache.clear()
    _warned.clear()
