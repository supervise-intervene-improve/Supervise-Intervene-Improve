"""Compile a model variant from the LIVE base scene plus a descriptor.

The corpus XMLs are never compiled. Instead we load the base scene as an `MjSpec`, write
the descriptor's cup Z-scales onto the base's own meshes, point the two box textures at the
files the descriptor names, and compile. Consequences that make this safe:

  * the mesh SET and every X/Y scale come from the base spec, so a corpus scene that added
    or renamed a mesh cannot leak into a compiled model;
  * the base scene's `<flag multiccd="enable"/>` and its corrected `front` camera survive,
    both of which a corpus XML would have reverted;
  * `LAB_FAST` keeps working, because the descriptor is geometry-independent.

`build_model(xml, None)` returns the base model by the ordinary `from_xml_path` route, so
in-distribution episodes are byte-for-byte what they were before this module existed.

MEASURED on this machine (cups scene, both interpreters): `MjSpec.from_file` 3 ms,
`spec.compile()` 1285 ms, model ~404 MB resident. MuJoCo 3.4.0 (conda `polymetis`, used by
the policy and grid) and 3.3.7 (repo `.venv`, used by the VR runtime) produce a
bit-identical model from the same descriptor -- which is why the descriptor, not a compiled
`.mjb`, is what crosses a process boundary.
"""

from __future__ import annotations

import math
import re
import threading
from pathlib import Path
from typing import Dict, Optional

import mujoco

from model_variants.descriptor import is_valid_descriptor, scene_closure_sha256, variant_key


class VariantIncompatible(RuntimeError):
    """A descriptor cannot be applied to this base scene, or the result is unusable.

    Always caught by callers: a bad variant must fall back to the base model and stamp a
    reason on the episode, never abort a running session.
    """


_CUP_MESH_RE = re.compile(r"^(cup\d+)_(?:visual|col_\d+)$")
_SWAPPED_TEXTURES = ("cracker_tex", "sugar_tex")

# Fields compared with a tolerance rather than exactly. All three feed downstream maths:
# `stat.extent`/`znear`/`zfar` drive `depth_buffer_to_meters` in the VR publisher, and
# `cam_fovy` is part of the `_CAM_INTRINSICS_CACHE` key.
_FLOAT_FIELDS = ("stat_extent", "vis_znear", "vis_zfar", "opt_timestep")

_spec_lock = threading.Lock()
_spec_cache: Dict[str, "mujoco.MjSpec"] = {}


def _names(model, objtype, count):
    out = []
    for i in range(count):
        out.append(mujoco.mj_id2name(model, objtype, i) or "")
    return out


def base_spec(xml_path) -> "mujoco.MjSpec":
    """Parsed base scene, cached per path. Callers must `.copy()` before mutating."""
    key = str(Path(xml_path).resolve())
    with _spec_lock:
        spec = _spec_cache.get(key)
        if spec is None:
            spec = mujoco.MjSpec.from_file(key)
            _spec_cache[key] = spec
        return spec


def describe_reference(model) -> dict:
    """A fingerprint of everything a consumer of this model depends on.

    Two variants must agree on all of it. The name lists matter because every consumer
    resolves bodies, joints, cameras and geoms by NAME and caches the resulting integer id
    (`anchor_site_id`, `replay_bindings`, `movable_objects`, the point-cloud camera set);
    `jnt_qposadr` and `actuator_ctrlrange` matter because `build_state()` reads `qvel[:7]`
    and `ctrl[7]` positionally.
    """
    return {
        "nq": int(model.nq), "nv": int(model.nv), "nu": int(model.nu),
        "na": int(model.na), "nbody": int(model.nbody), "njnt": int(model.njnt),
        "ngeom": int(model.ngeom), "ncam": int(model.ncam),
        "nsensor": int(model.nsensor), "nmocap": int(model.nmocap),
        "opt_timestep": float(model.opt.timestep),
        "opt_integrator": int(model.opt.integrator),
        "opt_cone": int(model.opt.cone),
        "opt_solver": int(model.opt.solver),
        "opt_enableflags": int(model.opt.enableflags),
        "opt_disableflags": int(model.opt.disableflags),
        "stat_extent": float(model.stat.extent),
        "vis_znear": float(model.vis.map.znear),
        "vis_zfar": float(model.vis.map.zfar),
        "body_names": _names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody),
        "joint_names": _names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt),
        "geom_names": _names(model, mujoco.mjtObj.mjOBJ_GEOM, model.ngeom),
        "camera_names": _names(model, mujoco.mjtObj.mjOBJ_CAMERA, model.ncam),
        "actuator_names": _names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu),
        "sensor_names": _names(model, mujoco.mjtObj.mjOBJ_SENSOR, model.nsensor),
        "jnt_type": [int(v) for v in model.jnt_type],
        "jnt_qposadr": [int(v) for v in model.jnt_qposadr],
        "jnt_dofadr": [int(v) for v in model.jnt_dofadr],
        "actuator_ctrlrange": [[float(a), float(b)] for a, b in model.actuator_ctrlrange],
        "camera_fovy": [float(v) for v in model.cam_fovy],
    }


def fingerprint_diff(reference: dict, candidate: dict):
    """First differing field as `(name, ref, cand)`, or `None`. Order is stable."""
    for field in reference:
        ref, cand = reference[field], candidate.get(field)
        if field in _FLOAT_FIELDS:
            if cand is None or not math.isclose(float(ref), float(cand), rel_tol=1e-9, abs_tol=1e-12):
                return (field, ref, cand)
        elif field == "camera_fovy":
            if cand is None or len(cand) != len(ref) or not all(
                math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12) for a, b in zip(ref, cand)
            ):
                return (field, ref, cand)
        elif ref != cand:
            return (field, ref, cand)
    return None


def validate_against_reference(model, reference_model) -> None:
    """Raise `VariantIncompatible` unless the variant is a drop-in for the reference."""
    ref = reference_model if isinstance(reference_model, dict) else describe_reference(reference_model)
    diff = fingerprint_diff(ref, describe_reference(model))
    if diff is not None:
        field, expected, got = diff
        if isinstance(expected, list) and isinstance(got, list) and len(expected) != len(got):
            detail = f"length {len(expected)} != {len(got)}"
        else:
            detail = f"{expected!r} != {got!r}"
            if len(detail) > 300:
                detail = detail[:300] + "..."
        raise VariantIncompatible(f"{field}: {detail}")


def _assert_descriptor_matches_base(spec, descriptor, xml_path) -> None:
    ok, reason = is_valid_descriptor(descriptor)
    if not ok:
        raise VariantIncompatible(f"malformed descriptor ({reason})")

    expected = descriptor.get("base_scene_sha256") or ""
    actual = scene_closure_sha256(xml_path)
    if expected and expected != actual:
        raise VariantIncompatible(
            f"descriptor was built for a different scene (base_scene_sha256 "
            f"{expected[:12]}... != {actual[:12]}...)"
        )

    by_cup: Dict[str, set] = {}
    for mesh in spec.meshes:
        m = _CUP_MESH_RE.match(mesh.name or "")
        if m:
            by_cup.setdefault(m.group(1), set()).add(float(mesh.scale[2]))
    for cup in descriptor["cup_scale_z"]:
        if cup not in by_cup:
            raise VariantIncompatible(f"base scene has no meshes for {cup!r}")
        if len(by_cup[cup]) != 1:
            raise VariantIncompatible(
                f"base scene's {cup} meshes disagree on scale[2] ({sorted(by_cup[cup])}); "
                "the base scene was edited in a way this builder does not understand"
            )

    names = {t.name for t in spec.textures}
    missing = [t for t in _SWAPPED_TEXTURES if t not in names]
    if missing:
        raise VariantIncompatible(f"base scene is missing texture(s) {missing}")


def apply_descriptor_to_spec(spec, descriptor: dict, *, mujoco_scenes_root=None) -> int:
    """Mutate `spec` in place to match `descriptor`. Returns the mesh count written.

    Separated from `build_model` so a test can assert the mutation exactly. That matters
    because the compiled model CANNOT be used to check it: MuJoCo stores `mesh_vert` in the
    mesh's principal-axis frame and rewrites `mesh_quat`, so changing Z permutes which
    stored axis is which and per-axis vertex extents are not comparable across a variant.
    (Frame-independent evidence lives in the tests: each cup's mass ratio equals its Z-scale
    ratio to within 0.07%, which would be ratio^2 or ratio^3 had X or Y also moved.)
    """
    # Cup Z-scale. Only index 2 is ever written: X and Y differ per cup AND between a cup's
    # visual and collision meshes, and come from the base by construction.
    wanted = descriptor["cup_scale_z"]
    written = 0
    for mesh in spec.meshes:
        m = _CUP_MESH_RE.match(mesh.name or "")
        if m and m.group(1) in wanted:
            scale = list(mesh.scale)
            scale[2] = float(wanted[m.group(1)])
            mesh.scale = scale
            written += 1
    if written == 0:
        raise VariantIncompatible("descriptor matched no base meshes")

    # Box textures. The descriptor's `texture_file` is now AUTHORITATIVE: each box texture
    # is SET to the file the corpus scene names, resolved under our own mujoco_scenes/.
    #
    # This replaced a swap (`cracker_tex` <-> `sugar_tex`, so each YCB box wore the other's
    # photo). Two reasons the swap is gone:
    #   * the corpus now names dedicated OOD textures that are not a permutation of the
    #     base's own two files, so there is nothing to swap TO -- exchanging the base's
    #     strings could never produce them;
    #   * the swap could only ever be CHECKED by filename, and both base textures are named
    #     `texture_map.png` under a directory named `assets`, so the check compared
    #     ("texture_map.png", "assets") against itself and could not actually tell the two
    #     apart (it needed the grandparent). Setting the path removes the ambiguity: what
    #     the descriptor names is what gets compiled.
    # The path is re-rooted rather than trusted: `texture_file` is mujoco_scenes/-relative
    # by construction (`repo_relative`), and a value that escapes the root or names a
    # missing file refuses the variant instead of compiling something unintended.
    textures = {t.name: t for t in spec.textures}
    root = _mujoco_scenes_root(spec, mujoco_scenes_root)
    for name in _SWAPPED_TEXTURES:
        claimed = str(descriptor["texture_file"][name])
        resolved = (root / claimed).resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            raise VariantIncompatible(
                f"texture path for {name!r} escapes mujoco_scenes/: {claimed!r}"
            ) from None
        if not resolved.is_file():
            raise VariantIncompatible(
                f"texture file for {name!r} not found on this machine: {resolved}"
            )
        textures[name].file = str(resolved)
    return written


def _mujoco_scenes_root_for_path(xml_path) -> Optional[Path]:
    """Walk up an ABSOLUTE base-scene path to its `mujoco_scenes/` directory.

    Returns None rather than raising so `_mujoco_scenes_root` can still try its own
    fallback; only if both fail does the variant refuse.
    """
    src = Path(xml_path).resolve()
    for candidate in src.parents:
        if candidate.name == "mujoco_scenes":
            return candidate
    return None


def _mujoco_scenes_root(spec, explicit=None) -> Path:
    """The `mujoco_scenes/` directory the descriptor's texture paths are relative to.

    `build_model` passes it explicitly, derived from the absolute base-scene path it
    already resolved. The fallback exists only for callers that hand us a bare spec (the
    tests do): `MjSpec.modelfiledir` is stored RELATIVE TO THE CWD THAT LOADED THE FILE
    ('mujoco_scenes/working_scenes/with_soft_gripper/'), so resolving it is correct only
    while the process is still in that directory -- which is why it is not the primary
    path. Walk up by NAME rather than a fixed depth: the base scene has siblings
    (`_multiwindow_fast`, `_local_assets`) and any of them may be passed.
    """
    if explicit is not None:
        return Path(explicit).resolve()
    src = Path(getattr(spec, "modelfiledir", "") or "").resolve()
    for candidate in (src, *src.parents):
        if candidate.name == "mujoco_scenes":
            return candidate
    raise VariantIncompatible(
        f"cannot locate mujoco_scenes/ above the base scene ({src}); "
        "descriptor texture paths are relative to it"
    )


def build_model(xml_path, descriptor: Optional[dict], *, reference_model=None):
    """Compile the variant. `descriptor is None` -> the base model, unmodified."""
    xml_path = str(Path(xml_path).resolve())
    if descriptor is None:
        return mujoco.MjModel.from_xml_path(xml_path)

    spec = base_spec(xml_path).copy()
    _assert_descriptor_matches_base(spec, descriptor, xml_path)
    apply_descriptor_to_spec(spec, descriptor,
                             mujoco_scenes_root=_mujoco_scenes_root_for_path(xml_path))

    try:
        model = spec.compile()
    except Exception as exc:                      # MuJoCo raises a bare ValueError here
        raise VariantIncompatible(f"compile failed: {exc}") from exc

    if reference_model is not None:
        validate_against_reference(model, reference_model)
    return model


def build_key(xml_path, descriptor) -> str:
    """Cache identity: the descriptor key, which already pins the base scene."""
    return variant_key(descriptor)


def clear_cache() -> None:
    with _spec_lock:
        _spec_cache.clear()
