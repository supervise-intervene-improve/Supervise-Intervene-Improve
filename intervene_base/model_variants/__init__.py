"""Compiled MuJoCo model variants for model-level out-of-distribution conditions.

`descriptor` is stdlib-only and must stay importable in a process with no `mujoco`; the
builder and cache are therefore exposed lazily (PEP 562) so `import model_variants` never
drags MuJoCo in by itself.
"""

from model_variants.descriptor import (  # noqa: F401
    BASE_KEY,
    SCHEMA,
    canonical_json,
    clear_cache,
    file_sha256,
    is_valid_descriptor,
    make_descriptor,
    map_corpus_asset_path,
    repo_relative,
    scene_closure_sha256,
    variant_key,
)

_LAZY = {
    "VariantIncompatible": "model_variants.builder",
    "base_spec": "model_variants.builder",
    "build_model": "model_variants.builder",
    "describe_reference": "model_variants.builder",
    "validate_against_reference": "model_variants.builder",
    "VariantCache": "model_variants.cache",
}

__all__ = sorted(set(_LAZY) | {
    "BASE_KEY", "SCHEMA", "canonical_json", "clear_cache", "file_sha256",
    "is_valid_descriptor", "make_descriptor", "map_corpus_asset_path",
    "repo_relative", "scene_closure_sha256", "variant_key",
})


def __getattr__(name):
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    return getattr(importlib.import_module(module_name), name)


def __dir__():
    return __all__
