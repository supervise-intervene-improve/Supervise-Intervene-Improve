from dataclasses import dataclass

import glfw
import mujoco


@dataclass
class MouseState:
    left_down: bool = False
    right_down: bool = False
    middle_down: bool = False
    last_x: float = 0.0
    last_y: float = 0.0


def install_callbacks(app):
    glfw.set_window_user_pointer(app.window, app)
    glfw.set_key_callback(app.window, key_callback)
    glfw.set_cursor_pos_callback(app.window, cursor_pos_callback)
    glfw.set_mouse_button_callback(app.window, mouse_button_callback)
    glfw.set_scroll_callback(app.window, scroll_callback)


def key_callback(window, key, scancode, action, mods):
    app = glfw.get_window_user_pointer(window)
    if app is None or action not in (glfw.PRESS, glfw.REPEAT):
        return

    mode = getattr(app, "mode", "replay")

    if key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)
        return

    if key in (glfw.KEY_ENTER, glfw.KEY_KP_ENTER) and action == glfw.PRESS:
        if mode == "replan":
            app.finish_replan()
        return

    if action == glfw.PRESS and getattr(app, "viewer", None) is not None:
        if key == glfw.KEY_1:
            app.viewer.set_view_mode("free")
            return

        if key == glfw.KEY_2:
            app.viewer.set_view_mode("cameras")
            return

        if key == glfw.KEY_0:
            app.viewer.reset_free_camera()
            print("[VIEW] free camera reset")
            return

    # Same rule as the ZMQ command path (_handle_policy_command): an episode OUTCOME must
    # never be markable while a human is holding the arm. These two used to run with no
    # mode check at all, so a stray keypress could call finish_replan mid-grip and label
    # the episode from a scene the policy never produced.
    if key == glfw.KEY_S and action == glfw.PRESS:
        if mode == "replan":
            print("[INPUT] Ignoring S (task success) while an intervention is active.")
            return
        app.mark_task_success()
        return

    if key == glfw.KEY_F and action == glfw.PRESS:
        if mode == "replan":
            print("[INPUT] Ignoring F (task failure) while an intervention is active.")
            return
        app.mark_task_failure()
        return

    # R is allowed DURING a takeover, matching RESET on the command path: it is the
    # operator abandoning the attempt, and _restart_current_scene cancels the
    # intervention on the way. It sat below the replay-only guard, so a scene switch
    # during an intervention was impossible from the keyboard too.
    if key == glfw.KEY_R and action == glfw.PRESS:
        if mode == "replan":
            print("[INPUT] R during an active intervention: cancelling the takeover "
                  "and restarting the scene.")
        if hasattr(app, "_restart_current_scene"):
            app._restart_current_scene()
        else:
            app.player.reset_to_start(play=True)
        return

    if mode != "replay":
        return

    if key == glfw.KEY_SPACE and action == glfw.PRESS:
        if getattr(app, "mode", "replay") != "replay":
            return

        app.trigger_replan()
        return

    if key == glfw.KEY_P and action == glfw.PRESS:
        app.player.toggle_pause()
        return

    if key == glfw.KEY_M and action == glfw.PRESS:
        app.toggle_mirror()
        return

    # NOTE: R is handled above the replay-only guard, not here.

    if key == glfw.KEY_END:
        app.player.jump_to_last(play=False)
        return

    if key == glfw.KEY_I and action == glfw.PRESS:
        app.player.print_frame_summary()
        return

    if key == glfw.KEY_RIGHT and app.player.is_paused:
        app.player.step_frame(+1)
        return

    if key == glfw.KEY_LEFT and app.player.is_paused:
        app.player.step_frame(-1)
        return


def mouse_button_callback(window, button, action, mods):
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


def cursor_pos_callback(window, xpos, ypos):
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

    app.viewer.move_camera(
        app.player.model,
        action,
        dx / max(1, h),
        dy / max(1, h),
    )


def scroll_callback(window, xoffset, yoffset):
    app = glfw.get_window_user_pointer(window)
    if app is None:
        return

    app.viewer.move_camera(
        app.player.model,
        mujoco.mjtMouse.mjMOUSE_ZOOM,
        0.0,
        -0.05 * yoffset,
    )
