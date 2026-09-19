from __future__ import annotations

from condition_images import condition_image_references, missing_condition_images, resolve_condition_image


def test_condition_image_paths_match_required_structure(tmp_path):
    reference = resolve_condition_image("1A", "S1", tmp_path)
    assert reference.path == tmp_path / "study1a" / "desktop_rgb.png"
    assert not reference.exists
    assert "Desktop-RGB + Kinesthetic Teaching" in reference.placeholder_html()
    assert "Reference image not yet provided" in reference.placeholder_html()


def test_missing_image_fallback_disappears_when_all_files_exist(tmp_path):
    references = condition_image_references("1B", tmp_path)
    assert len(missing_condition_images("1B", tmp_path)) == 3
    for reference in references:
        reference.path.parent.mkdir(parents=True, exist_ok=True)
        reference.path.write_bytes(b"synthetic-test-image")
    assert missing_condition_images("1B", tmp_path) == []
