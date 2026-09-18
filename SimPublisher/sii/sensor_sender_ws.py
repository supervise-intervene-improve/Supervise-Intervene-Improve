# sensor_sender_ws.py  (MuJoCo 3.2 compatible: GLFW hidden context + mjr_readPixels depth)
import argparse
import asyncio
import json
import time
import numpy as np
import cv2
import mujoco
import websockets

from mujoco.glfw import glfw  # comes with mujoco python package

# -------------------------
# Camera utilities
# -------------------------
def list_cameras(model):
    cams = []
    for i in range(model.ncam):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        if name:
            cams.append(name)
    return cams

# -------------------------
# Windows-safe GLFW context
# -------------------------
class GLFWHiddenContext:
    """Creates a hidden GLFW window and makes its OpenGL context current."""
    def __init__(self, width=640, height=480, title="mujoco_offscreen"):
        if not glfw.init():
            raise RuntimeError("glfw.init() failed. On Windows, ensure graphics drivers are installed.")
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.DOUBLEBUFFER, glfw.TRUE)
        # You can try forcing OpenGL version if needed:
        # glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        # glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        # glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

        self._win = glfw.create_window(width, height, title, None, None)
        if not self._win:
            glfw.terminate()
            raise RuntimeError("glfw.create_window() failed.")
        glfw.make_context_current(self._win)
        glfw.swap_interval(0)

    def make_current(self):
        glfw.make_context_current(self._win)

    def free(self):
        try:
            glfw.destroy_window(self._win)
        finally:
            glfw.terminate()

# -------------------------
# Offscreen renderer via mjr_*
# -------------------------
class OffscreenMujocoRenderer:
    """
    Renders RGB + depth using:
      mjv_updateScene -> mjr_render -> mjr_readPixels
    Depth returned is linearized to metric distance (approx) using znear/zfar.
    """
    def __init__(self, model: mujoco.MjModel, width: int, height: int):
        self.model = model
        self.width = int(width)
        self.height = int(height)

        # MUST have an active OpenGL context before creating MjrContext
        self.con = mujoco.MjrContext(model, mujoco.mjtFontScale.mjFONTSCALE_150)

        self.scn = mujoco.MjvScene(model, maxgeom=10000)
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()

        # Offscreen buffer selection (MuJoCo handles it internally)
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_OFFSCREEN, self.con)

        self.viewport = mujoco.MjrRect(0, 0, self.width, self.height)

        # Allocate read buffers
        self._rgb = np.empty((self.height, self.width, 3), dtype=np.uint8)
        self._depth = np.empty((self.height, self.width), dtype=np.float32)

    def render(self, data: mujoco.MjData, camera_name: str):
        # set fixed camera by name
        cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id < 0:
            raise RuntimeError(f"Camera '{camera_name}' not found in model.")
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam.fixedcamid = cam_id

        # Update + render
        mujoco.mjv_updateScene(
            self.model, data, self.opt, None, self.cam,
            mujoco.mjtCatBit.mjCAT_ALL, self.scn
        )
        mujoco.mjr_render(self.viewport, self.scn, self.con)

        # Read pixels
        mujoco.mjr_readPixels(self._rgb, self._depth, self.viewport, self.con)

        # MuJoCo/OpenGL usually returns RGB upside-down
        rgb = np.flipud(self._rgb).copy()
        depth = np.flipud(self._depth).copy()

        # depth currently is OpenGL depth buffer [0..1], non-linear.
        depth_m = self._linearize_depth(depth)

        fovy_deg = float(self.model.cam_fovy[cam_id])

        return rgb, depth_m, fovy_deg

    def _linearize_depth(self, depth01: np.ndarray) -> np.ndarray:
        """
        Convert OpenGL depth buffer (0..1) to linear depth in meters (approx).
        Uses model.vis.map.znear/zfar (scene near/far).
        """
        znear = float(self.model.vis.map.znear)
        zfar  = float(self.model.vis.map.zfar)
        # Avoid pathological config
        znear = max(znear, 1e-6)
        zfar  = max(zfar, znear + 1e-3)

        # Convert depth buffer to NDC z in [-1, 1]
        z_ndc = depth01 * 2.0 - 1.0

        # Standard OpenGL perspective projection inverse
        # linear = (2*n*f) / (f+n - z_ndc*(f-n))
        denom = (zfar + znear) - z_ndc * (zfar - znear)
        linear = (2.0 * znear * zfar) / np.maximum(denom, 1e-12)

        # Filter invalid
        linear[~np.isfinite(linear)] = 0.0
        linear[linear < 0] = 0.0
        return linear.astype(np.float32)

# -------------------------
# Depth -> point cloud
# -------------------------
def depth_to_pointcloud(depth_m: np.ndarray, fovy_deg: float, stride: int = 4):
    """
    depth_m: float32 [H,W] in meters (camera view depth)
    returns Nx3 float32 (camera coordinates)
    """
    H, W = depth_m.shape
    fovy = np.deg2rad(float(fovy_deg))
    fy = (H / 2.0) / np.tan(fovy / 2.0)
    fx = fy  # assume square pixels
    cx = (W - 1) / 2.0
    cy = (H - 1) / 2.0

    ys = np.arange(0, H, stride)
    xs = np.arange(0, W, stride)
    xx, yy = np.meshgrid(xs, ys)

    z = depth_m[yy, xx]
    valid = np.isfinite(z) & (z > 1e-6)
    xx = xx[valid].astype(np.float32)
    yy = yy[valid].astype(np.float32)
    z  = z[valid].astype(np.float32)

    x = (xx - cx) * z / fx
    y = (yy - cy) * z / fy
    pc = np.stack([x, y, z], axis=1).astype(np.float32)
    return pc

# -------------------------
# Streaming
# -------------------------
async def stream_mujoco(xml_path, ws_url, fps, width, height, pc_enabled, pc_stride, cams):
    print(f"[sensor_sender_ws] XML: {xml_path}")
    print(f"[sensor_sender_ws] WS : {ws_url}")
    print(f"[sensor_sender_ws] FPS: {fps}")
    print(f"[sensor_sender_ws] RES: {width}x{height}")
    print(f"[sensor_sender_ws] PC : {'ON' if pc_enabled else 'OFF'} (stride={pc_stride})")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)

    if not cams:
        cams = list_cameras(model)
    if not cams:
        raise RuntimeError("No cameras found in model. Add <camera name=...> to XML.")
    print(f"[sensor_sender_ws] Cams: {cams}")

    # Create a hidden GLFW context FIRST (fixes gladLoadGL)
    gl_ctx = GLFWHiddenContext(width, height)
    gl_ctx.make_current()

    # Now safe to create MjrContext
    renderer = OffscreenMujocoRenderer(model, width, height)

    dt = 1.0 / float(fps)
    last = time.time()

    try:
        while True:
            try:
                async with websockets.connect(ws_url, max_size=2**30) as ws:
                    print("[sensor_sender_ws] WebSocket connected.")

                    while True:
                        # Step physics once per frame (you can do more for stability)
                        mujoco.mj_step(model, data)

                        for cam_name in cams:
                            # Render RGB+depth
                            rgb, depth_m, fovy_deg = renderer.render(data, cam_name)

                            # JPEG encode RGB (receiver can decode)
                            ok, rgb_jpg = cv2.imencode(
                                ".jpg",
                                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                                [int(cv2.IMWRITE_JPEG_QUALITY), 80],
                            )
                            if not ok:
                                continue
                            rgb_bytes = rgb_jpg.tobytes()

                            depth_bytes = depth_m.tobytes()

                            pc = np.zeros((0, 3), dtype=np.float32)
                            if pc_enabled:
                                pc = depth_to_pointcloud(depth_m, fovy_deg, stride=pc_stride)

                            header = {
                                "camera": cam_name,
                                "w": int(width),
                                "h": int(height),
                                "fovy_deg": float(fovy_deg),
                                "pc_points": int(pc.shape[0]),
                                "rgb_size": int(len(rgb_bytes)),
                                "depth_size": int(len(depth_bytes)),
                            }

                            await ws.send(json.dumps(header))
                            await ws.send(rgb_bytes)
                            await ws.send(depth_bytes)
                            if pc_enabled:
                                await ws.send(pc.tobytes())

                        # FPS pacing
                        now = time.time()
                        elapsed = now - last
                        sleep_t = max(0.0, dt - elapsed)
                        await asyncio.sleep(sleep_t)
                        last = time.time()

            except Exception as e:
                print(f"[sensor_sender_ws] WS error: {e}")
                print("[sensor_sender_ws] Reconnecting in 1s...")
                await asyncio.sleep(1.0)
    finally:
        gl_ctx.free()
        
        
# --- Windows-safe offscreen renderer for MuJoCo 3.2 (GLFW + mjr_readPixels) ---

import numpy as np
import mujoco
from mujoco import glfw

class GLFWOffscreenRenderer:
    def __init__(self, model, width, height):
        self.model = model
        self.width = int(width)
        self.height = int(height)

        # Create a GLFW context (hidden window) on Windows
        if not glfw.init():
            raise RuntimeError("glfw.init() failed (no OpenGL context available).")

        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        glfw.window_hint(glfw.DOUBLEBUFFER, glfw.FALSE)

        self.window = glfw.create_window(self.width, self.height, "offscreen", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("glfw.create_window() failed.")

        glfw.make_context_current(self.window)

        # MuJoCo rendering structs
        self.con = mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_150)
        self.scn = mujoco.MjvScene(self.model, maxgeom=10000)
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()

        # Set framebuffer size
        self.viewport = mujoco.MjrRect(0, 0, self.width, self.height)

        # Buffers for readPixels
        self.rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.depth = np.zeros((self.height, self.width), dtype=np.float32)

    def render_camera(self, model, data, cam_name: str):
        # Setup camera by name
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cam_id < 0:
            raise RuntimeError(f"Camera '{cam_name}' not found")

        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam.fixedcamid = cam_id

        mujoco.mjv_updateScene(
            model, data, self.opt, None, self.cam,
            mujoco.mjtCatBit.mjCAT_ALL, self.scn
        )
        mujoco.mjr_render(self.viewport, self.scn, self.con)

        # Read pixels (RGB uint8, depth float32)
        mujoco.mjr_readPixels(self.rgb, self.depth, self.viewport, self.con)

        # IMPORTANT: mjr_readPixels returns image upside-down; flip vertically
        rgb = np.flipud(self.rgb).copy()
        depth = np.flipud(self.depth).copy()

        # fovy (degrees)
        fovy_deg = float(model.cam_fovy[cam_id])

        # camera world pose (MuJoCo provides cam_xpos, cam_xmat)
        cam_pos = data.cam_xpos[cam_id].copy()              # (3,)
        cam_xmat = data.cam_xmat[cam_id].reshape(3, 3).copy()  # world-from-camera rotation? see note below

        return rgb, depth, fovy_deg, cam_pos, cam_xmat

    def close(self):
        try:
            glfw.make_context_current(None)
            if self.window:
                glfw.destroy_window(self.window)
        finally:
            glfw.terminate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", required=True)
    ap.add_argument("--ws", default="ws://127.0.0.1:8765")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--pc", action="store_true", help="Send pointcloud")
    ap.add_argument("--pc_stride", type=int, default=4, help="Pointcloud subsampling stride")
    ap.add_argument("--cams", nargs="*", default=None, help="Optional list of camera names. If omitted, uses all.")
    args = ap.parse_args()

    asyncio.run(stream_mujoco(args.xml, args.ws, args.fps, args.w, args.h, args.pc, args.pc_stride, args.cams))

if __name__ == "__main__":
    main()
