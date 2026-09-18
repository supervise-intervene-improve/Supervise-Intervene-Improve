#!/usr/bin/env python3
"""Browse MuJoCo XML scenes from a folder and remove bad scenes with D.

Examples:

    conda run -n polymetis python browse_mujoco_scene_folder.py mujoco_scenes/ood_scenes
    conda run -n polymetis python browse_mujoco_scene_folder.py mujoco_scenes/ood_scenes/boxes_cups
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_FOLDER = REPO_ROOT / "mujoco_scenes" / "ood_scenes"
SKIP_DIR_NAMES = {"includes", ".removed_scenes", ".tmp_generate_ood", "__pycache__"}


def import_mujoco():
    try:
        import mujoco  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "The mujoco Python package is required. Run with the project env, "
            "for example: conda run -n polymetis python browse_mujoco_scene_folder.py "
            "mujoco_scenes/ood_scenes"
        ) from exc
    return mujoco


def import_glfw():
    try:
        import glfw  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "The glfw Python package is required for the interactive browser."
        ) from exc
    return glfw


def is_scene_xml(path: Path) -> bool:
    if path.suffix.lower() != ".xml":
        return False
    if any(part in SKIP_DIR_NAMES for part in path.parts):
        return False
    return True


def find_xml_files(folder: Path, recursive: bool) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Folder does not exist: {folder}")
    if folder.is_file():
        return [folder] if is_scene_xml(folder) else []

    if not recursive:
        return sorted(path for path in folder.glob("*.xml") if is_scene_xml(path))

    xml_paths: list[Path] = []
    for path in folder.rglob("*.xml"):
        try:
            rel_parts = path.relative_to(folder).parts
        except ValueError:
            rel_parts = path.parts
        if any(part in SKIP_DIR_NAMES for part in rel_parts):
            continue
        if is_scene_xml(path):
            xml_paths.append(path)
    return sorted(xml_paths)


def rel_display(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def unique_target(path: Path) -> Path:
    if not path.exists():
        return path
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    candidate = path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
    counter = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{timestamp}_{counter}{path.suffix}")
        counter += 1
    return candidate


def included_scene_files(xml_path: Path) -> list[Path]:
    """Return per-scene include files that should move with this XML.

    Shared includes such as the generated Panda file are left in place. Includes
    whose basename starts with the scene XML stem are considered per-scene files.
    """

    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        return []

    include_paths: list[Path] = []
    for include in root.iter("include"):
        include_file = include.get("file")
        if not include_file:
            continue
        source = Path(include_file)
        if not source.is_absolute():
            source = (xml_path.parent / source).resolve()
        if source.exists() and source.name.startswith(xml_path.stem):
            include_paths.append(source)
    return include_paths


@dataclass
class RemovalRecord:
    scene: str
    moved_to: str
    moved_includes: list[str]
    timestamp: str


def move_scene_to_removed(xml_path: Path, scan_root: Path, removed_root: Path) -> RemovalRecord:
    xml_path = xml_path.resolve()
    scan_root = scan_root.resolve() if scan_root.is_dir() else scan_root.resolve().parent
    removed_root = removed_root.resolve()

    include_paths = included_scene_files(xml_path)

    try:
        scene_rel = xml_path.relative_to(scan_root)
    except ValueError:
        scene_rel = Path(xml_path.name)

    target_scene = unique_target(removed_root / scene_rel)
    target_scene.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(xml_path), str(target_scene))

    moved_includes: list[str] = []
    for include_path in include_paths:
        if not include_path.exists():
            continue
        try:
            include_rel = include_path.relative_to(scan_root)
        except ValueError:
            include_rel = Path("includes") / include_path.name
        target_include = unique_target(removed_root / include_rel)
        target_include.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(include_path), str(target_include))
        moved_includes.append(rel_display(target_include))

    record = RemovalRecord(
        scene=rel_display(xml_path),
        moved_to=rel_display(target_scene),
        moved_includes=moved_includes,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    )

    manifest_path = removed_root / "removed_manifest.jsonl"
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.__dict__) + "\n")

    return record


class SceneFolderBrowser:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.mujoco = import_mujoco()
        self.glfw = import_glfw()
        self.scan_root = args.folder.resolve()
        self.removed_root = args.removed_dir.resolve() if args.removed_dir else self._default_removed_root()
        self.xml_paths = find_xml_files(self.scan_root, recursive=args.recursive)
        self.index = 0
        self.model = None
        self.data = None
        self.cam = None
        self.opt = None
        self.scn = None
        self.ctx = None
        self.window = None
        self.show_overlay = True
        self.status = ""

        if args.limit is not None:
            self.xml_paths = self.xml_paths[: args.limit]
        if not self.xml_paths:
            raise RuntimeError(f"No scene XML files found in {self.scan_root}")

    def _default_removed_root(self) -> Path:
        if self.scan_root.is_dir():
            return self.scan_root / ".removed_scenes"
        return self.scan_root.parent / ".removed_scenes"

    def run(self) -> None:
        if not self.glfw.init():
            raise RuntimeError("Failed to initialize GLFW.")

        self.window = self.glfw.create_window(
            self.args.width,
            self.args.height,
            "MuJoCo Scene Folder Browser",
            None,
            None,
        )
        if self.window is None:
            self.glfw.terminate()
            raise RuntimeError("Failed to create GLFW window.")

        self.glfw.make_context_current(self.window)
        self.glfw.swap_interval(1)
        self.glfw.set_window_user_pointer(self.window, self)
        self.glfw.set_key_callback(self.window, key_callback)

        self.load_current()

        while not self.glfw.window_should_close(self.window):
            self.render()
            self.glfw.poll_events()

        self.close()

    def close(self) -> None:
        if self.ctx is not None:
            self.ctx.free()
            self.ctx = None
        if self.window is not None:
            self.glfw.destroy_window(self.window)
            self.window = None
        self.glfw.terminate()

    def load_current(self) -> None:
        if not self.xml_paths:
            self.status = "No scenes left."
            if self.window is not None:
                self.glfw.set_window_should_close(self.window, True)
            return

        if self.ctx is not None:
            self.ctx.free()
            self.ctx = None

        xml_path = self.xml_paths[self.index]
        self.model = self.mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = self.mujoco.MjData(self.model)
        self.mujoco.mj_forward(self.model, self.data)

        self.cam = self.mujoco.MjvCamera()
        self.opt = self.mujoco.MjvOption()
        self.scn = self.mujoco.MjvScene(self.model, maxgeom=self.args.maxgeom)
        self.ctx = self.mujoco.MjrContext(
            self.model,
            self.mujoco.mjtFontScale.mjFONTSCALE_150,
        )
        self.mujoco.mjv_defaultCamera(self.cam)
        self.mujoco.mjv_defaultOption(self.opt)
        self.status = f"Loaded {rel_display(xml_path)}"
        print(f"[LOAD] {self.index + 1}/{len(self.xml_paths)} {rel_display(xml_path)}")

    def next_scene(self) -> None:
        if not self.xml_paths:
            return
        self.index = (self.index + 1) % len(self.xml_paths)
        self.load_current()

    def previous_scene(self) -> None:
        if not self.xml_paths:
            return
        self.index = (self.index - 1) % len(self.xml_paths)
        self.load_current()

    def reload_scene(self) -> None:
        self.load_current()

    def remove_current_scene(self) -> None:
        if not self.xml_paths:
            return
        xml_path = self.xml_paths[self.index]
        record = move_scene_to_removed(xml_path, self.scan_root, self.removed_root)
        print(f"[REMOVE] {record.scene} -> {record.moved_to}")
        for include_path in record.moved_includes:
            print(f"[REMOVE] include -> {include_path}")
        self.xml_paths.pop(self.index)
        if self.index >= len(self.xml_paths):
            self.index = max(0, len(self.xml_paths) - 1)
        self.status = f"Removed {record.scene}"
        self.load_current()

    def render(self) -> None:
        if self.model is None or self.data is None or self.ctx is None:
            return
        width, height = self.glfw.get_framebuffer_size(self.window)
        viewport = self.mujoco.MjrRect(0, 0, width, height)
        self.mujoco.mjr_setBuffer(self.mujoco.mjtFramebuffer.mjFB_WINDOW, self.ctx)
        self.mujoco.mjr_rectangle(viewport, 0.06, 0.06, 0.06, 1.0)

        rects = self.camera_rects(width, height, len(self.args.cameras))
        for rect, camera_name in zip(rects, self.args.cameras):
            self.render_camera(rect, camera_name)

        if self.show_overlay:
            self.draw_overlay(viewport)

        self.glfw.swap_buffers(self.window)

    def camera_rects(self, width: int, height: int, count: int) -> list[Any]:
        if count == 1:
            return [self.mujoco.MjrRect(0, 0, width, height)]

        if count == 3:
            top_h = height // 2
            bottom_h = height - top_h
            half_w = width // 2
            return [
                self.mujoco.MjrRect(0, bottom_h, width, top_h),
                self.mujoco.MjrRect(0, 0, half_w, bottom_h),
                self.mujoco.MjrRect(half_w, 0, width - half_w, bottom_h),
            ]

        cols = math.ceil(math.sqrt(count))
        rows = math.ceil(count / cols)
        cell_w = max(1, width // cols)
        cell_h = max(1, height // rows)
        rects = []
        for idx in range(count):
            row = idx // cols
            col = idx % cols
            left = col * cell_w
            bottom = height - (row + 1) * cell_h
            rect_w = width - left if col == cols - 1 else cell_w
            rect_h = height - max(bottom, 0) if row == rows - 1 else cell_h
            rects.append(self.mujoco.MjrRect(left, max(bottom, 0), rect_w, rect_h))
        return rects

    def render_camera(self, rect: Any, camera_name: str) -> None:
        cam_id = self.mujoco.mj_name2id(
            self.model,
            self.mujoco.mjtObj.mjOBJ_CAMERA,
            camera_name,
        )
        if cam_id < 0:
            self.mujoco.mjr_rectangle(rect, 0.12, 0.02, 0.02, 1.0)
            self.mujoco.mjr_overlay(
                self.mujoco.mjtFont.mjFONT_NORMAL,
                self.mujoco.mjtGridPos.mjGRID_TOPLEFT,
                rect,
                f"missing camera: {camera_name}",
                "",
                self.ctx,
            )
            return

        self.cam.type = self.mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam.fixedcamid = cam_id
        self.mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.opt,
            None,
            self.cam,
            self.mujoco.mjtCatBit.mjCAT_ALL,
            self.scn,
        )
        self.mujoco.mjr_render(rect, self.scn, self.ctx)
        self.mujoco.mjr_overlay(
            self.mujoco.mjtFont.mjFONT_NORMAL,
            self.mujoco.mjtGridPos.mjGRID_TOPLEFT,
            rect,
            camera_name,
            "",
            self.ctx,
        )

    def draw_overlay(self, viewport: Any) -> None:
        current = self.xml_paths[self.index] if self.xml_paths else None
        title = (
            f"{self.index + 1}/{len(self.xml_paths)}  {rel_display(current)}"
            if current is not None
            else "No scenes"
        )
        help_text = (
            "Right/N/Space: next\n"
            "Left/P: previous\n"
            "D: move scene to .removed_scenes\n"
            "R: reload\n"
            "O: overlay\n"
            "Q/Esc: quit"
        )
        right = f"{self.status}\n{help_text}"
        self.mujoco.mjr_overlay(
            self.mujoco.mjtFont.mjFONT_NORMAL,
            self.mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
            viewport,
            title,
            right,
            self.ctx,
        )


def key_callback(window, key, scancode, action, mods):
    glfw = import_glfw()
    if action not in (glfw.PRESS, glfw.REPEAT):
        return
    browser = glfw.get_window_user_pointer(window)

    if key in (glfw.KEY_Q, glfw.KEY_ESCAPE):
        glfw.set_window_should_close(window, True)
    elif key in (glfw.KEY_RIGHT, glfw.KEY_N, glfw.KEY_SPACE):
        browser.next_scene()
    elif key in (glfw.KEY_LEFT, glfw.KEY_P):
        browser.previous_scene()
    elif key == glfw.KEY_R:
        browser.reload_scene()
    elif key == glfw.KEY_O:
        browser.show_overlay = not browser.show_overlay
    elif key == glfw.KEY_D and action == glfw.PRESS:
        browser.remove_current_scene()


def validate_loads(xml_paths: list[Path], limit: int | None) -> int:
    mujoco = import_mujoco()
    if limit is not None:
        xml_paths = xml_paths[:limit]
    failures = []
    for index, xml_path in enumerate(xml_paths, start=1):
        try:
            model = mujoco.MjModel.from_xml_path(str(xml_path))
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            print(f"[OK] {index}/{len(xml_paths)} {rel_display(xml_path)} ncon={data.ncon}")
        except Exception as exc:
            failures.append((xml_path, exc))
            print(f"[FAIL] {rel_display(xml_path)}: {type(exc).__name__}: {exc}")
    if failures:
        print(f"{len(failures)} failed out of {len(xml_paths)}", file=sys.stderr)
        return 1
    print(f"Validated {len(xml_paths)} XML scene(s).")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "folder",
        nargs="?",
        type=Path,
        default=DEFAULT_FOLDER,
        help="Folder or XML file to browse. Defaults to mujoco_scenes/ood_scenes.",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["front", "left", "right"],
        help="Fixed MuJoCo cameras to show.",
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--maxgeom", type=int, default=10000)
    parser.add_argument("--limit", type=int, help="Use only the first N discovered XML files.")
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Only load XML files directly inside the folder.",
    )
    parser.add_argument(
        "--removed-dir",
        type=Path,
        help="Where D moves removed scenes. Defaults to <folder>/.removed_scenes.",
    )
    parser.add_argument(
        "--validate-loads",
        action="store_true",
        help="Load XML files without opening a GUI window.",
    )
    parser.set_defaults(recursive=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    xml_paths = find_xml_files(args.folder.resolve(), recursive=args.recursive)
    if args.limit is not None:
        xml_paths = xml_paths[: args.limit]
    if not xml_paths:
        print(f"No XML scene files found in {args.folder}", file=sys.stderr)
        return 1

    if args.validate_loads:
        return validate_loads(xml_paths, args.limit)

    browser = SceneFolderBrowser(args)
    browser.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
