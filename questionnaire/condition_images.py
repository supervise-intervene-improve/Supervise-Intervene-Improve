"""Condition reference-image paths and neutral missing-image placeholders."""

from __future__ import annotations

import html
from dataclasses import dataclass
from pathlib import Path

from config import BASE_DIR
from questionnaire_definitions import CONDITIONS


CONDITION_IMAGE_FILES = {
    "1A": {
        "S1": "study1a/desktop_rgb.png",
        "S2": "study1a/vr_rgb.png",
        "S3": "study1a/vr_pointcloud.png",
    },
    "1B": {
        "C1": "study1b/kinesthetic_teaching.png",
        "C2": "study1b/motion_controller.png",
        "C3": "study1b/factr.png",
    },
}


@dataclass(frozen=True)
class ConditionImageReference:
    study: str
    condition_code: str
    condition_name: str
    path: Path

    @property
    def exists(self) -> bool:
        return self.path.is_file()

    def placeholder_html(self) -> str:
        name = html.escape(self.condition_name)
        return (
            '<div class="condition-image-placeholder" role="img" '
            f'aria-label="Reference image missing for {name}">'
            '<span>Reference image not yet provided</span>'
            f'<strong>{name}</strong></div>'
        )


def resolve_condition_image(
    study: str,
    condition_code: str,
    assets_root: str | Path | None = None,
) -> ConditionImageReference:
    """Resolve a condition image without fabricating or downloading imagery."""

    if study not in CONDITION_IMAGE_FILES or condition_code not in CONDITION_IMAGE_FILES[study]:
        raise ValueError(f"Unknown condition {study}/{condition_code}.")
    root = Path(assets_root) if assets_root is not None else BASE_DIR / "assets" / "conditions"
    return ConditionImageReference(
        study=study,
        condition_code=condition_code,
        condition_name=CONDITIONS[study][condition_code],
        path=root / CONDITION_IMAGE_FILES[study][condition_code],
    )


def condition_image_references(
    study: str, assets_root: str | Path | None = None
) -> list[ConditionImageReference]:
    return [resolve_condition_image(study, code, assets_root) for code in CONDITIONS[study]]


def missing_condition_images(study: str, assets_root: str | Path | None = None) -> list[Path]:
    return [reference.path for reference in condition_image_references(study, assets_root) if not reference.exists]
