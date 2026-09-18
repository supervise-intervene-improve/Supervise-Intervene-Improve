import glfw
import mujoco
import numpy as np

from config import VIEWER_MODE, VIEWER_CAMERA_NAMES
from .utils import depth_buffer_to_meters, get_camera_id


class MujocoRenderer:
    def __init__(self, model: mujoco.MjModel, window_width: int, window_height: int, title: str):
        self.model = model
        self.window_width = int(window_width)
        self.window_height = int(window_height)
        self.title = title

        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW.")

        self.window = glfw.create_window(
            self.window_width,
            self.window_height,
            self.title,
            None,
            None,
        )
        if self.window is None:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window.")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.scn = mujoco.MjvScene(model, maxgeom=10000)
        self.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False

        self.ctx = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

        mujoco.mjv_defaultCamera(self.cam)
        mujoco.mjv_defaultOption(self.opt)

        # -------- Your exact FREECAM settings --------
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = np.array([0.45, 0.0, 0.45], dtype=np.float64)
        self.cam.distance = 1.5968364916117135
        self.cam.azimuth = 179.578125
        self.cam.elevation = -28.0
        # --------------------------------------------

        self.viewport = mujoco.MjrRect(0, 0, self.window_width, self.window_height)

        self.rgb = None
        self.depth = None

        self.show_overlay = True

        print("[Renderer] model.stat.center =", self.model.stat.center)
        print("[Renderer] model.stat.extent =", self.model.stat.extent)
        print("[Renderer] initial cam.lookat =", self.cam.lookat)
        print("[Renderer] initial cam.distance =", self.cam.distance)
        print("[Renderer] initial cam.azimuth =", self.cam.azimuth)
        print("[Renderer] initial cam.elevation =", self.cam.elevation)

    def window_should_close(self) -> bool:
        return glfw.window_should_close(self.window)

    def poll_events(self):
        glfw.poll_events()

    def _save_viewer_cam_state(self):
        return {
            "type": self.cam.type,
            "fixedcamid": self.cam.fixedcamid,
            "lookat": self.cam.lookat.copy(),
            "distance": self.cam.distance,
            "azimuth": self.cam.azimuth,
            "elevation": self.cam.elevation,
        }

    def _restore_viewer_cam_state(self, state):
        self.cam.type = state["type"]
        self.cam.fixedcamid = state["fixedcamid"]
        self.cam.lookat[:] = state["lookat"]
        self.cam.distance = state["distance"]
        self.cam.azimuth = state["azimuth"]
        self.cam.elevation = state["elevation"]

    def _render_fixed_camera_to_viewport(
        self,
        data: mujoco.MjData,
        viewport: mujoco.MjrRect,
        cam_name: str,
    ):
        saved = self._save_viewer_cam_state()

        cam_id = get_camera_id(self.model, cam_name)
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam.fixedcamid = cam_id

        mujoco.mjv_updateScene(
            self.model,
            data,
            self.opt,
            None,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scn,
        )
        mujoco.mjr_render(viewport, self.scn, self.ctx)

        self._restore_viewer_cam_state(saved)

    def render_viewer(self, data: mujoco.MjData, overlay_lines=None):
        glfw.make_context_current(self.window)

        width, height = glfw.get_framebuffer_size(self.window)
        self.viewport = mujoco.MjrRect(0, 0, width, height)

        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.ctx)

        if VIEWER_MODE == "freecam":
            mujoco.mjv_updateScene(
                self.model,
                data,
                self.opt,
                None,
                self.cam,
                mujoco.mjtCatBit.mjCAT_ALL,
                self.scn,
            )
            mujoco.mjr_render(self.viewport, self.scn, self.ctx)

        elif VIEWER_MODE == "multicam":
            top_h = height // 2
            bottom_h = height - top_h
            half_w = width // 2

            # Top full-width panel: front
            vp_front = mujoco.MjrRect(0, bottom_h, width, top_h)

            # Bottom row: left / right
            vp_left = mujoco.MjrRect(0, 0, half_w, bottom_h)
            vp_right = mujoco.MjrRect(half_w, 0, width - half_w, bottom_h)

            self._render_fixed_camera_to_viewport(
                data, vp_front, VIEWER_CAMERA_NAMES["front"]
            )
            self._render_fixed_camera_to_viewport(
                data, vp_left, VIEWER_CAMERA_NAMES["left"]
            )
            self._render_fixed_camera_to_viewport(
                data, vp_right, VIEWER_CAMERA_NAMES["right"]
            )

            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                vp_front,
                "front",
                "",
                self.ctx,
            )
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                vp_left,
                "left",
                "",
                self.ctx,
            )
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                vp_right,
                "right",
                "",
                self.ctx,
            )

        else:
            raise ValueError(f"Unknown VIEWER_MODE: {VIEWER_MODE}")

        if self.show_overlay and overlay_lines is not None and len(overlay_lines) > 0:
            left_text = "\n".join(overlay_lines)
            right_text = f"viewer: {VIEWER_MODE}"
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                self.viewport,
                left_text,
                right_text,
                self.ctx,
            )

        glfw.swap_buffers(self.window)

    def render_rgbd_from_camera(self, data: mujoco.MjData, cam_name: str, width: int, height: int):
        glfw.make_context_current(self.window)

        if self.rgb is None or self.rgb.shape[:2] != (height, width):
            self.rgb = np.zeros((height, width, 3), dtype=np.uint8)
            self.depth = np.zeros((height, width), dtype=np.float32)

        viewport = mujoco.MjrRect(0, 0, width, height)
        cam_id = get_camera_id(self.model, cam_name)

        saved = self._save_viewer_cam_state()

        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam.fixedcamid = cam_id

        mujoco.mjv_updateScene(
            self.model,
            data,
            self.opt,
            None,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scn,
        )

        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.ctx)
        mujoco.mjr_render(viewport, self.scn, self.ctx)
        mujoco.mjr_readPixels(self.rgb, self.depth, viewport, self.ctx)

        rgb = np.flipud(self.rgb).copy()
        depth = np.flipud(self.depth).copy()
        depth_m = depth_buffer_to_meters(self.model, depth)

        self._restore_viewer_cam_state(saved)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.ctx)

        return rgb, depth_m

    def toggle_overlay(self):
        self.show_overlay = not self.show_overlay
        print(f"[Renderer] show_overlay = {self.show_overlay}")

    def close(self):
        try:
            if self.ctx is not None:
                self.ctx.free()
        except Exception:
            pass

        try:
            if self.window is not None:
                glfw.destroy_window(self.window)
        except Exception:
            pass

        glfw.terminate()