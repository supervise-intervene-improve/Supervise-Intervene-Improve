import glfw
import mujoco


class MouseState:
    def __init__(self):
        self.left_down = False
        self.right_down = False
        self.middle_down = False
        self.last_x = 0.0
        self.last_y = 0.0


def install_callbacks(window, model, scene, cam, cmd_state, renderer):
    mouse_state = MouseState()
    glfw.set_window_user_pointer(window, {
        "mouse_state": mouse_state,
        "model": model,
        "scene": scene,
        "cam": cam,
        "cmd_state": cmd_state,
        "renderer": renderer,
    })

    glfw.set_key_callback(window, _key_callback)
    glfw.set_cursor_pos_callback(window, _cursor_pos_callback)
    glfw.set_mouse_button_callback(window, _mouse_button_callback)
    glfw.set_scroll_callback(window, _scroll_callback)


def _key_callback(window, key, scancode, action, mods):
    if action != glfw.PRESS:
        return

    user_data = glfw.get_window_user_pointer(window)
    cmd_state = user_data["cmd_state"]

    if key == glfw.KEY_S:
        cmd_state.request_start()
        print("[KEY] Start requested.")
    elif key == glfw.KEY_E:
        cmd_state.request_stop()
        print("[KEY] Stop requested.")
    elif key == glfw.KEY_Q or key == glfw.KEY_ESCAPE:
        cmd_state.request_quit()
        print("[KEY] Quit requested.")
    elif key == glfw.KEY_X:
        cmd_state.request_discard()
        print("[KEY] Discard requested.")
    elif key == glfw.KEY_O:
        renderer = user_data["renderer"]
        renderer.toggle_overlay()
    elif key == glfw.KEY_P:
        cam = user_data["cam"]
        print("cam.lookat =", cam.lookat.copy())
        print("cam.distance =", cam.distance)
        print("cam.azimuth =", cam.azimuth)
        print("cam.elevation =", cam.elevation)


def _mouse_button_callback(window, button, action, mods):
    user_data = glfw.get_window_user_pointer(window)
    mouse = user_data["mouse_state"]

    mouse.left_down = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
    mouse.right_down = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
    mouse.middle_down = glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS

    x, y = glfw.get_cursor_pos(window)
    mouse.last_x = x
    mouse.last_y = y


def _cursor_pos_callback(window, xpos, ypos):
    user_data = glfw.get_window_user_pointer(window)
    mouse = user_data["mouse_state"]
    model = user_data["model"]
    scene = user_data["scene"]
    cam = user_data["cam"]

    dx = xpos - mouse.last_x
    dy = ypos - mouse.last_y
    mouse.last_x = xpos
    mouse.last_y = ypos

    if not (mouse.left_down or mouse.right_down or mouse.middle_down):
        return

    width, height = glfw.get_window_size(window)
    if height == 0:
        return

    shift_pressed = (
        glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
        or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    )

    if mouse.right_down:
        action = (
            mujoco.mjtMouse.mjMOUSE_MOVE_H
            if shift_pressed
            else mujoco.mjtMouse.mjMOUSE_MOVE_V
        )
    elif mouse.left_down:
        action = (
            mujoco.mjtMouse.mjMOUSE_ROTATE_H
            if shift_pressed
            else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        )
    else:
        action = mujoco.mjtMouse.mjMOUSE_ZOOM

    mujoco.mjv_moveCamera(
        model,
        action,
        dx / max(width, 1),
        dy / max(height, 1),
        scene,
        cam,
    )


def _scroll_callback(window, xoffset, yoffset):
    user_data = glfw.get_window_user_pointer(window)
    model = user_data["model"]
    scene = user_data["scene"]
    cam = user_data["cam"]

    mujoco.mjv_moveCamera(
        model,
        mujoco.mjtMouse.mjMOUSE_ZOOM,
        0.0,
        -0.05 * yoffset,
        scene,
        cam,
    )