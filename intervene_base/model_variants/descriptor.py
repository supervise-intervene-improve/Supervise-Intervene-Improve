"""Canonical form, identity, and path safety for a model variant.

WHAT A VARIANT IS
-----------------
For the cups task the out-of-distribution signal is not a pose — it is compile-time model
state: each cup's mesh Z-scale, and a swap of the two box textures. Neither can be written
into `qpos`, so an OOD cups episode needs a genuinely different compiled `MjModel`.

A *descriptor* is the small, portable value that says which one:

    {"schema": 1, "task": "cups",
     "base_scene_sha256": "<hash of the live scene XML + its include closure>",
     "cup_scale_z": {"cup1": 0.00116, "cup2": 0.00079, ...},
     "texture_file": {"cracker_tex": "<repo-relative>", "sugar_tex": "<repo-relative>"}}

`None` means "the base model, unmodified" and is never built — it *is* the scene as
authored, which is what keeps in-distribution episodes free.

The descriptor, not a compiled model, is what crosses a process boundary. That matters
because the policy/grid processes run MuJoCo 3.4.0 (conda `polymetis`) while the VR runtime
runs 3.3.7 (repo `.venv`): compiled `.mjb` files are version-specific, but both versions
were measured to produce a bit-identical model from the same descriptor.

WHY THE PATH RULES ARE HERE
---------------------------
Descriptors are read out of `mujoco_scenes/ood_scenes/`, whose files were generated on a
different machine and carry absolute `/home/user/...` asset paths (203 files; that home is
mode 0750 and unreadable here). Those strings are untrusted input. `map_corpus_asset_path`
is the only sanctioned way to turn one into a path we will open, and it re-roots rather
than following. This is the asset-side analogue of `_POISONED_INCLUDE` in
`data_io/ood_scene_poses.py`.

DELIBERATELY STDLIB-ONLY. This module must stay importable in a process that has no
`mujoco` — the descriptor tests run without a GL context, and `run_main_policy.sh` reads it
from shells that may not have the sim stack.
"""

from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional, Tuple

SCHEMA = 1

# The base model. Never built, never hashed — the absence of a descriptor.
BASE_KEY = "base"

# Marker used to re-root an absolute corpus asset path onto this machine.
_ASSET_ROOT_MARKER = "mujoco_scenes/"

# Defensive bound on include recursion while hashing a scene closure.
_MAX_INCLUDE_DEPTH = 16

_file_hash_cache: Dict[Tuple[str, int, int], str] = {}
_closure_cache: Dict[str, str] = {}
_warned: set = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _warned:
        _warned.add(key)
        print(f"[Variant][WARN] {message}")


# --------------------------------------------------------------------------- hashing


def file_sha256(path) -> str:
    """Hash a file, memoised on (path, mtime_ns, size).

    Asset hashes go into episode provenance, so this runs once per episode per asset; the
    stat-keyed cache keeps that free while still noticing an edited file.
    """
    path = Path(path)
    try:
        st = path.stat()
    except OSError as exc:
        _warn_once(f"stat:{path}", f"cannot stat {path}: {exc}")
        return ""
    key = (str(path), st.st_mtime_ns, st.st_size)
    cached = _file_hash_cache.get(key)
    if cached is not None:
        return cached
    h = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except OSError as exc:
        _warn_once(f"read:{path}", f"cannot read {path}: {exc}")
        return ""
    digest = h.hexdigest()
    _file_hash_cache[key] = digest
    return digest


def _iter_scene_includes(xml_path: Path, depth: int = 0):
    """Yield the include closure of a TRUSTED scene file.

    This is deliberately NOT the corpus reader's containment-checked traversal: the live
    base scene legitimately includes upward (`../../generated_panda/...`), and we authored
    it. The guards here are against accident, not attack: relative-only, existing, cycle-
    free, depth-capped.
    """
    if depth > _MAX_INCLUDE_DEPTH:
        _warn_once(f"depth:{xml_path}", f"include depth cap hit at {xml_path}")
        return
    try:
        root = ET.parse(xml_path).getroot()
    except Exception as exc:
        _warn_once(f"parse:{xml_path}", f"could not parse scene {xml_path}: {exc}")
        return
    for inc in root.iter("include"):
        raw = inc.get("file")
        if not raw or Path(raw).is_absolute():
            continue
        target = (xml_path.parent / raw).resolve()
        if not target.is_file():
            _warn_once(f"missing:{target}", f"include not found: {target}")
            continue
        yield target
        yield from _iter_scene_includes(target, depth + 1)


def scene_closure_sha256(xml_path) -> str:
    """Identity of the geometry a scene actually compiles to.

    `app.py` already records `mujoco_xml_sha256`, but that hashes the top-level file only —
    editing `generated_cups_via_stl_4.xml` would not change it. A variant descriptor is
    only meaningful relative to a specific base geometry, so the closure hash is what it
    pins, and what a consumer checks before trusting a descriptor off the wire.
    """
    xml_path = Path(xml_path).resolve()
    cached = _closure_cache.get(str(xml_path))
    if cached is not None:
        return cached

    seen = {}
    seen[xml_path.name] = file_sha256(xml_path)
    for inc in _iter_scene_includes(xml_path):
        # Key by name+hash rather than absolute path so the value is machine-independent.
        seen[inc.name] = file_sha256(inc)

    blob = json.dumps(sorted(seen.items()), separators=(",", ":"))
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    _closure_cache[str(xml_path)] = digest
    return digest


# ------------------------------------------------------------------- canonical form


def canonical_json(descriptor: Optional[dict]) -> str:
    """The exact bytes `variant_key` hashes. Sorted keys, no whitespace.

    `json.dumps` uses `repr()` for floats, which round-trips float64 exactly on CPython, so
    the key is stable across the 3.4.0 / 3.3.7 interpreter split.
    """
    if descriptor is None:
        return ""
    return json.dumps(descriptor, sort_keys=True, separators=(",", ":"))


def variant_key(descriptor: Optional[dict]) -> str:
    """Stable 16-hex identity. `None` -> "base"."""
    if descriptor is None:
        return BASE_KEY
    return hashlib.sha256(canonical_json(descriptor).encode("utf-8")).hexdigest()[:16]


def make_descriptor(*, task: str, base_scene_sha256: str,
                    cup_scale_z: Dict[str, float],
                    texture_file: Dict[str, str]) -> dict:
    """Build a descriptor in canonical form.

    Floats are coerced to `float` and names sorted so two readers of the same corpus scene
    can never produce different keys for the same variant.
    """
    return {
        "schema": SCHEMA,
        "task": str(task),
        "base_scene_sha256": str(base_scene_sha256),
        "cup_scale_z": {str(k): float(v) for k, v in sorted(cup_scale_z.items())},
        "texture_file": {str(k): str(v) for k, v in sorted(texture_file.items())},
    }


def is_valid_descriptor(descriptor) -> Tuple[bool, str]:
    """Shape check for a descriptor arriving off the wire. Never raises."""
    if descriptor is None:
        return True, ""
    if not isinstance(descriptor, dict):
        return False, "not_a_dict"
    if descriptor.get("schema") != SCHEMA:
        return False, f"bad_schema:{descriptor.get('schema')!r}"
    if not isinstance(descriptor.get("task"), str) or not descriptor["task"]:
        return False, "bad_task"
    if not isinstance(descriptor.get("base_scene_sha256"), str) or len(descriptor["base_scene_sha256"]) != 64:
        return False, "bad_base_scene_sha256"
    for field in ("cup_scale_z", "texture_file"):
        if not isinstance(descriptor.get(field), dict) or not descriptor[field]:
            return False, f"bad_{field}"
    for name, value in descriptor["cup_scale_z"].items():
        if not isinstance(value, (int, float)) or not (0.0 < float(value) < 1.0):
            return False, f"bad_scale:{name}={value!r}"
    return True, ""


# ------------------------------------------------------------------- path safety


def map_corpus_asset_path(raw: str, mujoco_scenes_root) -> Optional[Path]:
    """Re-root an UNTRUSTED corpus asset path onto this machine, or refuse.

    The corpus writes absolute paths from the generating machine
    (`/home/user/.../mujoco_scenes/ycb/Boxes/Sugar_Box/assets/texture_map.png`). We never
    open that string. Instead we keep the tail after the last `mujoco_scenes/` and resolve
    it under OUR `mujoco_scenes/`, then require that the result really is inside it.

    Returns `None` — never raises, never a partial path — if the string has no marker,
    escapes the root, or names a file that does not exist. A refused asset drops the whole
    scene from the variant pool rather than degrading it silently.
    """
    if not raw or not isinstance(raw, str):
        return None
    root = Path(mujoco_scenes_root).resolve()

    marker = _ASSET_ROOT_MARKER
    idx = raw.rfind(marker)
    tail = raw[idx + len(marker):] if idx >= 0 else raw
    tail = tail.lstrip("/")
    if not tail:
        return None

    candidate = (root / tail).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        _warn_once(f"escape:{raw}", f"refusing asset path that escapes mujoco_scenes/: {raw!r}")
        return None
    if not candidate.is_file():
        _warn_once(f"noasset:{candidate}", f"corpus asset not found on this machine: {candidate}")
        return None
    return candidate


def repo_relative(path, mujoco_scenes_root) -> str:
    """`mujoco_scenes/`-relative POSIX string, for putting in a descriptor."""
    root = Path(mujoco_scenes_root).resolve()
    return Path(path).resolve().relative_to(root).as_posix()


def clear_cache() -> None:
    """For tests that point at different roots or edit files in one process."""
    _file_hash_cache.clear()
    _closure_cache.clear()
    _warned.clear()
