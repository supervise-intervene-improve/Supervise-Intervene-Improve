"""Read object poses out of the pre-generated OOD scene corpus.

WHY THIS EXISTS
---------------
`mujoco_scenes/ood_scenes/` holds 150 contact-validated out-of-distribution scenes produced
by `generate_ood_scenes.py`. Those scenes — not a runtime sampler — are the source of truth
for what "out of distribution" means.

We READ the poses out of them and write those into `qpos`. We never compile a corpus XML as
a model. That is deliberate, and it is what makes the whole approach work:

* 203 files under `ood_scenes/` reference `/home/user/...` (another user's home, mode 0750),
  so `t_shape_ood_001.xml` does not even load on this machine. The top-level t_shape files
  are clean, and the per-scene include files we do read carry only poses.
* The corpus carries a stale `front` camera that was rolled 180 degrees before the base
  scenes were fixed. Reading poses leaves the live scene's cameras alone.
* The corpus derives from the full-fidelity bases, so loading it would conflict with
  LAB_FAST. Poses are model-independent, so it does not.

Verified: across all 50 `t_shape` scenes the ONLY elements that differ from the base scene
are `<include>`, `<camera name="front">`, and the `pos`/`euler` of `<body name="T1">` and
`<body name="T2">`. The OOD signal there is entirely free-joint pose.

NOT EVERY TASK IS POSE-ONLY
---------------------------
The `boxes_cups` corpus is different: its real signal is cup mesh Z-scale and a swap of the
two box textures, both compile-time. Poses there vary by only +/-0.01 m / +/-5 deg, i.e.
INSIDE the in-distribution jitter, so reading poses alone would produce an episode labelled
OOD that is not. That task is handled by `data_io/ood_scene_variants.py` +
`model_variants/`, which read the same corpus files for their mesh/texture PARAMETERS and
compile a variant of the LIVE base scene. The reasons above still hold there — the corpus
XML is still never compiled — which is what preserves the base scene's `multiccd` flag and
its corrected `front` camera. This module stays the pose reader; `iter_include_targets` is
public so both readers share one copy of the containment rules.

DELIBERATELY STDLIB-ONLY. Importing `mujoco` here would let a caller compile a corpus file
by accident, and would make this untestable without a GL context.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple

# INTERVENE_TASK_MODE value -> corpus subdirectory. Keyed off the task rather than LAB
# because LAB has no `wiregame` case while INTERVENE_TASK_MODE carries all four values.
TASK_DIRS = {
    "tshape": "t_shape",
    "cups": "boxes_cups",
    "wiregame": "wire_spoon",
}

# The one include we must never follow: it is a per-corpus copy of the robot whose
# `meshdir` was absolutized to /home/user at generation time. It contains no object poses.
_POISONED_INCLUDE = "sii_panda_model_soft_gripper.xml"

Pose = Tuple[Tuple[float, float, float], Tuple[float, float, float]]

_scene_cache: Dict[Tuple[str, str], List[Path]] = {}
_pose_cache: Dict[str, Dict[str, Pose]] = {}
_warned: set = set()


def _warn_once(key: str, message: str) -> None:
    """A missing or broken corpus must degrade to in-distribution, not kill an episode,
    and must not spam once per episode across a long run."""
    if key not in _warned:
        _warned.add(key)
        print(f"[OOD][WARN] {message}")


def _floats(text: str | None, n: int = 3) -> Tuple[float, ...]:
    if not text:
        return (0.0,) * n
    parts = text.replace(",", " ").split()
    vals = [float(p) for p in parts[:n]]
    vals += [0.0] * (n - len(vals))
    return tuple(vals)


def list_scenes(task_key: str, corpus_root) -> List[Path]:
    """Top-level scene XMLs for a task, sorted. `includes/` is excluded by construction."""
    corpus_root = Path(corpus_root)
    cache_key = (str(task_key), str(corpus_root))
    if cache_key in _scene_cache:
        return _scene_cache[cache_key]

    subdir = TASK_DIRS.get(str(task_key))
    if subdir is None:
        _warn_once(f"task:{task_key}",
                   f"no OOD corpus is defined for task {task_key!r}; "
                   f"known tasks: {sorted(TASK_DIRS)}")
        _scene_cache[cache_key] = []
        return []

    task_dir = corpus_root / subdir
    if not task_dir.is_dir():
        _warn_once(f"dir:{task_dir}",
                   f"OOD corpus directory not found: {task_dir.resolve()} — "
                   "sessions will stay IN-distribution")
        _scene_cache[cache_key] = []
        return []

    scenes = sorted(p for p in task_dir.glob("*.xml") if p.is_file())
    if not scenes:
        _warn_once(f"empty:{task_dir}",
                   f"OOD corpus directory is empty: {task_dir.resolve()} — "
                   "sessions will stay IN-distribution")
    _scene_cache[cache_key] = scenes
    return scenes


def iter_include_targets(xml_path: Path, task_dir: Path):
    """Yield the per-scene include files this scene owns.

    Only `t_shape` keeps its object bodies in the top-level file. `boxes_cups` and
    `wire_spoon` top-levels hold just the table and delegate everything to per-scene
    includes, so a top-level-only parser would silently return zero bodies for them and
    quietly produce an in-distribution scene labelled OOD.

    Three hard rules bound what we will open:
      1. the path must be relative (never absolute — that is how /home/user gets in),
      2. it must resolve INSIDE the task directory,
      3. it must not be the poisoned shared robot include.
    """
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return
    task_dir = task_dir.resolve()
    for inc in root.iter("include"):
        raw = inc.get("file")
        if not raw or Path(raw).is_absolute():
            continue
        if Path(raw).name == _POISONED_INCLUDE:
            continue
        target = (xml_path.parent / raw).resolve()
        try:
            target.relative_to(task_dir)
        except ValueError:
            continue                      # escapes the task dir — refuse
        if target.is_file():
            yield target


# Private alias kept so existing callers/tests keep working. The public name exists because
# `data_io/ood_scene_variants.py` reads the SAME corpus files for mesh scales and textures
# and must not grow a second copy of the three containment rules above.
_iter_include_targets = iter_include_targets


def scene_poses(xml_path, task_dir=None) -> Dict[str, Pose]:
    """`{body_name: ((x, y, z), (roll, pitch, yaw))}` for one corpus scene.

    Euler values are returned raw, in the file's own units (the corpus is written with
    MuJoCo's default `angle="radian"`). Converting them to a quaternion is the caller's
    job and MUST use MuJoCo's `eulerseq="xyz"` convention.

    Bodies with no `pos`/`euler` attribute default to zeros, matching MuJoCo — the cups
    bodies carry no `euler` at all.
    """
    xml_path = Path(xml_path)
    key = str(xml_path.resolve())
    if key in _pose_cache:
        return _pose_cache[key]

    task_dir = Path(task_dir) if task_dir is not None else xml_path.parent
    poses: Dict[str, Pose] = {}

    def collect(path: Path) -> None:
        try:
            root = ET.parse(path).getroot()
        except Exception as exc:
            _warn_once(f"parse:{path}", f"could not parse OOD scene {path}: {exc}")
            return
        for body in root.iter("body"):
            name = body.get("name")
            if not name:
                continue
            poses[name] = (_floats(body.get("pos")), _floats(body.get("euler")))

    collect(xml_path)
    for inc in _iter_include_targets(xml_path, task_dir):
        collect(inc)

    _pose_cache[key] = poses
    return poses


def clear_cache() -> None:
    """For tests that point at different corpus roots in one process."""
    _scene_cache.clear()
    _pose_cache.clear()
    _warned.clear()
