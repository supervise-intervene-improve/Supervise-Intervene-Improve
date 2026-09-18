import sys
import time
from pathlib import Path
from dataclasses import dataclass

import glfw
import mujoco
import numpy as np


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


@dataclass
class MouseState:
    left_down: bool = False
    right_down: bool = False
    middle_down: bool = False
    last_x: float = 0.0
    last_y: float = 0.0


class TrajectoryPlayer:
    """
    Minimal single-scene trajectory player.

    Expected NPZ keys:
      - sim_t               (required) shape: [T]
      - qpos_sim            (optional) shape: [T, nq]
      - qvel_sim            (optional) shape: [T, nv]
      - ctrl_sim            (optional) shape: [T, nu]

    Recommended:
      store at least sim_t + qpos_sim
    """

    def __init__(self, xml_path: str, npz_path: str):
        self.xml_path = str(xml_path)
        self.npz_path = str(npz_path)

        # Load model and data
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        # Load trajectory
        self.log = np.load(self.npz_path)

        if "sim_t" not in self.log.files:
            raise RuntimeError(f"{self.npz_path} is missing required key 'sim_t'")

        self.sim_t = np.asarray(self.log["sim_t"], dtype=np.float64)
        if len(self.sim_t) == 0:
            raise RuntimeError(f"{self.npz_path} contains zero frames")

        self.qpos_seq = self.log["qpos_sim"] if "qpos_sim" in self.log.files else None
        self.qvel_seq = self.log["qvel_sim"] if "qvel_sim" in self.log.files else None
        self.ctrl_seq = self.log["ctrl_sim"] if "ctrl_sim" in self.log.files else None

        self.nframes = len(self.sim_t)
        self.frame_idx = 0
        self.is_paused = False

        # Viewer state
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.scn = mujoco.MjvScene(self.model, maxgeom=2000)
        mujoco.mjv_defaultCamera(self.cam)
        mujoco.mjv_defaultOption(self.opt)

        # Playback clock
        self.play_start_wall = None
        self.play_start_sim = None

        self.reset_to_start(play=True)

    def reset_to_start(self, play=True):
        self.frame_idx = 0
        self._apply_frame(self.frame_idx)
        self.play_start_wall = time.perf_counter()
        self.play_start_sim = float(self.sim_t[self.frame_idx])
        self.is_paused = not play

    def restart_clock_from_current_frame(self):
        self.play_start_wall = time.perf_counter()
        self.play_start_sim = float(self.sim_t[self.frame_idx])

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        if not self.is_paused:
            self.restart_clock_from_current_frame()

    def step_frame(self, delta: int):
        new_idx = clamp(self.frame_idx + delta, 0, self.nframes - 1)
        self._apply_frame(new_idx)
        self.restart_clock_from_current_frame()

    def _apply_frame(self, idx: int):
        idx = int(clamp(idx, 0, self.nframes - 1))
        self.frame_idx = idx

        if self.qpos_seq is not None:
            if self.qpos_seq[idx].shape != self.data.qpos.shape:
                raise RuntimeError(
                    f"qpos_sim frame shape {self.qpos_seq[idx].shape} != model qpos shape {self.data.qpos.shape}"
                )
            self.data.qpos[:] = self.qpos_seq[idx]

        if self.qvel_seq is not None:
            if self.qvel_seq[idx].shape != self.data.qvel.shape:
                raise RuntimeError(
                    f"qvel_sim frame shape {self.qvel_seq[idx].shape} != model qvel shape {self.data.qvel.shape}"
                )
            self.data.qvel[:] = self.qvel_seq[idx]
        else:
            self.data.qvel[:] = 0.0

        if self.ctrl_seq is not None and self.model.nu > 0:
            n = min(self.model.nu, self.ctrl_seq[idx].shape[0])
            self.data.ctrl[:n] = self.ctrl_seq[idx][:n]

        self.data.time = float(self.sim_t[idx])
        mujoco.mj_forward(self.model, self.data)

    def update(self):
        if self.is_paused:
            return

        target_sim = self.play_start_sim + (time.perf_counter() - self.play_start_wall)

        idx = self.frame_idx
        while idx + 1 < self.nframes and self.sim_t[idx + 1] <= target_sim:
            idx += 1

        if idx != self.frame_idx:
            self._apply_frame(idx)

        if self.frame_idx >= self.nframes - 1:
            self.is_paused = True

    def render(self, viewport, ctx):
        mujoco.mjv_updateScene(
            self.model,
            self.data,
            self.opt,
            None,
            self.cam,
            mujoco.mjtCatBit.mjCAT_ALL,
            self.scn,
        )
        mujoco.mjr_render(viewport, self.scn, ctx)
        self._draw_overlay(viewport, ctx)

    def _draw_overlay(self, viewport, ctx):
        state = "PAUSED" if self.is_paused else "PLAYING"

        txt1 = (
            f"{Path(self.npz_path).name}\n"
            f"frame {self.frame_idx + 1}/{self.nframes}\n"
            f"t = {self.sim_t[self.frame_idx]:.3f}"
        )
        txt2 = (
            "Space: pause/resume\n"
            "Left/Right: step when paused\n"
            "R: reset\n"
            "Esc: quit"
        )

        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_TOPLEFT,
            viewport,
            txt1,
            state,
            ctx,
        )

        mujoco.mjr_overlay(
            mujoco.mjtFont.mjFONT_NORMAL,
            mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
            viewport,
            txt2,
            "",
            ctx,
        )

class App:
    def __init__(self, xml_path: str, npz_path: str, width=1400, height=900):
        self.xml_path = xml_path
        self.npz_path = npz_path
        self.width = width
        self.height = height

        self.window = None
        self.ctx = None
        self.mouse = MouseState()
        self.player = None

    def init(self):
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        self.window = glfw.create_window(
            self.width, self.height, "MuJoCo Single Scene Player", None, None
        )
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        self.player = TrajectoryPlayer(self.xml_path, self.npz_path)
        self.ctx = mujoco.MjrContext(
            self.player.model, mujoco.mjtFontScale.mjFONTSCALE_100
        )

        glfw.set_window_user_pointer(self.window, self)
        glfw.set_key_callback(self.window, App._key_callback)
        glfw.set_cursor_pos_callback(self.window, App._cursor_pos_callback)
        glfw.set_mouse_button_callback(self.window, App._mouse_button_callback)
        glfw.set_scroll_callback(self.window, App._scroll_callback)

    def run(self):
        self.init()
        try:
            while not glfw.window_should_close(self.window):
                glfw.poll_events()
                self.player.update()

                win_w, win_h = glfw.get_framebuffer_size(self.window)
                viewport = mujoco.MjrRect(0, 0, win_w, win_h)

                mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, self.ctx)
                mujoco.mjr_rectangle(viewport, 0.08, 0.08, 0.08, 1.0)

                self.player.render(viewport, self.ctx)
                glfw.swap_buffers(self.window)
        finally:
            self.close()

    def close(self):
        if self.ctx is not None:
            try:
                self.ctx.free()
            except Exception:
                pass
            self.ctx = None

        if self.window is not None:
            try:
                glfw.destroy_window(self.window)
            except Exception:
                pass
            self.window = None

        glfw.terminate()

    @staticmethod
    def _key_callback(window, key, scancode, action, mods):
        app = glfw.get_window_user_pointer(window)
        if app is None or action not in (glfw.PRESS, glfw.REPEAT):
            return

        player = app.player

        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(window, True)
        elif key == glfw.KEY_SPACE:
            player.toggle_pause()
        elif key == glfw.KEY_R:
            player.reset_to_start(play=True)
        elif key == glfw.KEY_RIGHT:
            if player.is_paused:
                player.step_frame(+1)
        elif key == glfw.KEY_LEFT:
            if player.is_paused:
                player.step_frame(-1)

    @staticmethod
    def _mouse_button_callback(window, button, action, mods):
        app = glfw.get_window_user_pointer(window)
        if app is None:
            return

        x, y = glfw.get_cursor_pos(window)
        app.mouse.last_x = x
        app.mouse.last_y = y

        if button == glfw.MOUSE_BUTTON_LEFT:
            app.mouse.left_down = (action == glfw.PRESS)
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            app.mouse.right_down = (action == glfw.PRESS)
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            app.mouse.middle_down = (action == glfw.PRESS)

    @staticmethod
    def _cursor_pos_callback(window, xpos, ypos):
        app = glfw.get_window_user_pointer(window)
        if app is None:
            return

        dx = xpos - app.mouse.last_x
        dy = ypos - app.mouse.last_y
        app.mouse.last_x = xpos
        app.mouse.last_y = ypos

        if not (app.mouse.left_down or app.mouse.right_down or app.mouse.middle_down):
            return

        _, h = glfw.get_window_size(window)
        if h <= 0:
            return

        shift = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )

        if app.mouse.right_down:
            action = mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif app.mouse.left_down:
            action = mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM

        mujoco.mjv_moveCamera(
            app.player.model,
            action,
            dx / max(1, h),
            dy / max(1, h),
            app.player.scn,
            app.player.cam,
        )

    @staticmethod
    def _scroll_callback(window, xoffset, yoffset):
        app = glfw.get_window_user_pointer(window)
        if app is None:
            return

        mujoco.mjv_moveCamera(
            app.player.model,
            mujoco.mjtMouse.mjMOUSE_ZOOM,
            0.0,
            -0.05 * yoffset,
            app.player.scn,
            app.player.cam,
        )


def main():
    if len(sys.argv) != 3:
        print("Usage:")
        print("  python single_scene_player.py path/to/model.xml path/to/traj.npz")
        sys.exit(1)

    xml_path = sys.argv[1]
    npz_path = sys.argv[2]

    if not Path(xml_path).exists():
        raise FileNotFoundError(xml_path)
    if not Path(npz_path).exists():
        raise FileNotFoundError(npz_path)

    app = App(xml_path, npz_path)
    app.run()


if __name__ == "__main__":
    main()