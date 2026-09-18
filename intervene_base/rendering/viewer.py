import mujoco
import numpy as np


ROLLOUT_FREE_CAMERA = {
    "lookat": np.array([0.45, 0.0, 0.45], dtype=np.float64),
    "distance": 1.5968364916117135,
    "azimuth": 179.578125,
    "elevation": -28.0,
}

VIEW_MODES = ("free", "cameras")
FIXED_VIEW_CAMERAS = ("front", "right", "left")


class SceneViewer:
    def __init__(self, model: mujoco.MjModel, view_mode: str = "free"):
        self.model = model
        self.cam_default = mujoco.MjvCamera()
        self.cam_default.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam = mujoco.MjvCamera()
        self.view_mode = "free"

        self.opt = mujoco.MjvOption()
        self.scn = mujoco.MjvScene(self.model, maxgeom=2000)

        mujoco.mjv_defaultCamera(self.cam_default)
        mujoco.mjv_defaultCamera(self.cam)
        mujoco.mjv_defaultOption(self.opt)
        self.reset_free_camera()
        self.set_view_mode(view_mode, announce=False)

        self.set_fast_visuals()

    def rebind_model(self, model: mujoco.MjModel):
        """Point this viewer at a different compiled model (an OOD model variant).

        `MjvScene` is allocated from the model, so it has to be rebuilt. Deliberately does
        NOT touch `cam`, `cam_default`, `opt` or `view_mode`: those hold where the operator
        has dragged the free camera, and resetting them at every episode boundary would be
        an obvious regression in the grid viewer.
        """
        self.model = model
        self.scn = mujoco.MjvScene(self.model, maxgeom=2000)
        self.set_fast_visuals()

    def reset_free_camera(self):
        self.cam_default.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam_default.lookat[:] = ROLLOUT_FREE_CAMERA["lookat"]
        self.cam_default.distance = ROLLOUT_FREE_CAMERA["distance"]
        self.cam_default.azimuth = ROLLOUT_FREE_CAMERA["azimuth"]
        self.cam_default.elevation = ROLLOUT_FREE_CAMERA["elevation"]

    def set_view_mode(self, view_mode: str, announce: bool = True):
        if view_mode not in VIEW_MODES:
            raise ValueError(f"Unknown view mode {view_mode!r}. Expected one of {VIEW_MODES}.")

        self.view_mode = view_mode
        if announce:
            print(f"[VIEW] mode = {self.view_mode}")

    def toggle_view_mode(self):
        next_mode = "cameras" if self.view_mode == "free" else "free"
        self.set_view_mode(next_mode)

    def get_view_status_text(self):
        if self.view_mode == "free":
            return "view: free"
        return "view: front/right/left"

    def set_fast_visuals(self):
        self.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        self.scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        self.scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
        self.scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = 0

    def set_pretty_visuals(self):
        self.scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 1
        self.scn.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 1
        self.scn.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1
        self.scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1

    def render(self, player, viewport, ctx, draw_overlay=True, extra_status: str = ""):
        if self.view_mode == "cameras":
            self.render_fixed_camera_layout(player, viewport, ctx)
            if draw_overlay:
                self.draw_overlay(player, viewport, ctx, extra_status=extra_status)
            return

        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, ctx)
        mujoco.mjv_updateScene(
            player.model,
            player.data,
            self.opt,
            None,
            self.cam_default,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scn,
        )
        mujoco.mjr_render(viewport, self.scn, ctx)
        if draw_overlay:
            self.draw_overlay(player, viewport, ctx, extra_status=extra_status)

    def render_fixed_camera_layout(self, player, viewport, ctx):
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, ctx)
        mujoco.mjr_rectangle(viewport, 0.08, 0.08, 0.08, 1.0)

        width = int(viewport.width)
        height = int(viewport.height)
        left = int(viewport.left)
        bottom = int(viewport.bottom)
        top_h = max(1, height // 2)
        bottom_h = max(1, height - top_h)
        half_w = max(1, width // 2)

        camera_rects = {
            "front": mujoco.MjrRect(left, bottom + bottom_h, width, top_h),
            "left": mujoco.MjrRect(left, bottom, half_w, bottom_h),
            "right": mujoco.MjrRect(left + half_w, bottom, width - half_w, bottom_h),
        }

        for cam_name in FIXED_VIEW_CAMERAS:
            rect = camera_rects[cam_name]
            self._set_fixed_camera(cam_name)
            mujoco.mjv_updateScene(
                player.model,
                player.data,
                self.opt,
                None,
                self.cam,
                mujoco.mjtCatBit.mjCAT_ALL,
                self.scn,
            )
            mujoco.mjr_render(rect, self.scn, ctx)
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                rect,
                cam_name,
                "",
                ctx,
            )

    def render_model_data(self, model, data, viewport, ctx, title_text=None, status_text=None):
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, ctx)

        mujoco.mjv_updateScene(
            model,
            data,
            self.opt,
            None,
            self.cam_default,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scn,
        )
        mujoco.mjr_render(viewport, self.scn, ctx)

        if title_text is not None or status_text is not None:
            mujoco.mjr_overlay(
                mujoco.mjtFont.mjFONT_NORMAL,
                mujoco.mjtGridPos.mjGRID_TOPLEFT,
                viewport,
                "" if title_text is None else title_text,
                "" if status_text is None else status_text,
                ctx,
            )

    def _set_fixed_camera(self, cam_name: str):
        cam_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            cam_name,
        )
        if cam_id < 0:
            raise ValueError(f"Camera '{cam_name}' not found in model.")

        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam.fixedcamid = cam_id
        return cam_id

    def render_rgbd_from_camera(
        self,
        data: mujoco.MjData,
        ctx: mujoco.MjrContext,
        cam_name: str,
        width: int,
        height: int,
    ):
        self._set_fixed_camera(cam_name)

        viewport = mujoco.MjrRect(0, 0, width, height)

        # Render into the offscreen buffer
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, ctx)

        mujoco.mjv_updateScene(
            self.model,
            data,
            self.opt,
            None,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scn,
        )

        mujoco.mjr_render(viewport, self.scn, ctx)

        rgb = np.empty((height, width, 3), dtype=np.uint8)
        depth = np.empty((height, width), dtype=np.float32)

        mujoco.mjr_readPixels(rgb, depth, viewport, ctx)

        # OpenGL origin is bottom-left, flip to image coordinates
        rgb = np.flipud(rgb)
        depth_raw = np.flipud(depth)

        depth_m = depth_to_meters(depth_raw, self.model)

        depth_m = np.clip(depth_m, 0.0, 2.0)
        depth = depth_m / 2.0

        return rgb, depth

    def render_rgbd_from_cameras(
        self,
        data: mujoco.MjData,
        ctx: mujoco.MjrContext,
        camera_names: list[str],
        width: int,
        height: int,
    ):
        outputs = {}
        for cam_name in camera_names:
            rgb, depth = self.render_rgbd_from_camera(
                data=data,
                ctx=ctx,
                cam_name=cam_name,
                width=width,
                height=height,
            )
            outputs[cam_name] = {
                "rgb": rgb,
                "depth": depth,
            }
        return outputs

    def draw_overlay(self, player, viewport, ctx, extra_status: str = ""):
        info, state = player.get_overlay_text(extra_status=extra_status)
        view_status = self.get_view_status_text()
        state = f"{state} | {view_status}"

        help_left = (
            "Space: intervene/takeover\n"
            "P: pause/resume\n"
            "Left/Right: step when paused\n"
            "S: success + new random scene\n"
            "F: fail + restart scene\n"
            "R: restart scene\n"
            "End: jump to last frame\n"
            "I: print frame summary\n"
            "1: free view\n"
            "2: camera view\n"
            "Esc: quit"
        )
        help_right = (
            "Mouse left: rotate\n"
            "Mouse right: move\n"
            "Mouse wheel: zoom\n"
            "0: reset free view\n"
            "Shift + drag: alt axis"
        )

        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            viewport,
            info,
            state,
            ctx,
        )
        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
            viewport,
            help_left,
            help_right,
            ctx,
        )

    def move_camera(self, model, action, dx, dy):
        if self.view_mode != "free":
            return

        mujoco.mjv_moveCamera(
            model,
            action,
            dx,
            dy,
            self.scn,
            self.cam_default,
        )

def depth_to_meters(depth, model):
    near = model.vis.map.znear
    far = model.vis.map.zfar

    z = (2.0 * near * far) / (
        far + near - (2.0 * depth - 1.0) * (far - near)
    )
    return z
