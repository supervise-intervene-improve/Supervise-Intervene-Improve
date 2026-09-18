import os
import time
import numpy as np


_record_backend = None
_record_error = None


def _get_record_backend():
    global _record_backend, _record_error
    if _record_backend is not None:
        return _record_backend
    try:
        import record as record_module
    except Exception as exc:
        _record_backend = None
        _record_error = exc
        return None
    _record_backend = record_module
    _record_error = None
    return record_module


def is_record_backend_available():
    return _get_record_backend() is not None


def get_record_backend_error():
    _get_record_backend()
    return _record_error


def _resolve_control_mode(record_module, control_mode):
    if control_mode is None:
        return record_module.ControlType.HYBRID_JOINT_IMPEDANCE_CONTROL
    if isinstance(control_mode, str):
        return getattr(record_module.ControlType, control_mode)
    return control_mode


def _fast_control_switch_enabled() -> bool:
    """Swap the arm policy in place instead of rebuilding both gRPC clients.

    Read per call rather than at import so it can be flipped without a relaunch while
    debugging a takeover on hardware. See RecordRobotAdapter.switch_control_mode.
    """
    return os.environ.get(
        "INTERVENE_FAST_CONTROL_SWITCH", "1"
    ).strip().lower() not in {"0", "false", "no", "off"}


class RecordRobotAdapter:
    def __init__(
        self,
        robot_key: str = "p1",
        control_mode=None,
        gripper_speed: float = 0.35,
        gripper_force: float = 0.10,
        gripper_track_eps_m: float = 0.002,
        gripper_cmd_period_s: float = 0.02,
    ):
        self.robot_key = robot_key
        self.control_mode = control_mode
        self.gripper_speed = gripper_speed
        self.gripper_force = gripper_force
        self.gripper_track_eps_m = gripper_track_eps_m
        self.gripper_cmd_period_s = gripper_cmd_period_s

        self.robot = None
        self.connected = False
        self._gripper_cmd_state = self._new_gripper_cmd_state()

    def _new_gripper_cmd_state(self):
        return {
            "last_width_cmd": None,
            "last_cmd_t": 0.0,
            "future": None,
            "last_error_t": 0.0,
        }

    def connect(self, already_connected=False):
        record_module = _get_record_backend()
        if record_module is None:
            raise RuntimeError(f"Robot backend unavailable: {_record_error}")

        self.robot = record_module.ROBOTS[self.robot_key]

        if already_connected:
            self.connected = True
            print(f"[ROBOT-ADAPTER] Reusing already-connected robot '{self.robot_key}'.")
            return

        control_mode = _resolve_control_mode(record_module, self.control_mode)

        print(
            f"[ROBOT-ADAPTER] Connecting robot '{self.robot_key}' "
            f"with mode {control_mode}..."
        )
        self.robot.connect(control_mode)
        self.connected = True

    def close(self):
        if self.robot is not None and self.connected:
            self.robot.close()
        self.connected = False
        self.robot = None
        self._gripper_cmd_state = self._new_gripper_cmd_state()

    def release_without_homing(self):
        """Drop the Polymetis handles without invoking the backend reset.

        ``Robot.close()`` calls ``FrankaArm.close()``, whose legacy shutdown path
        calls ``reset()`` and therefore sends the arm to Polymetis' default home.
        The intervention state machine already performs the deliberate return to
        the scene pose, so a second implicit home here is both surprising and
        unsafe.  This is only for terminal cleanup after that explicit move.
        """
        backend = self.robot
        if backend is not None:
            arm = getattr(backend, "robot_arm", None)
            hand = getattr(backend, "robot_gripper", None)
            # The hardware wrappers' destructors are allowed to run after these
            # objects become unreachable.  Detaching their RPC clients makes that
            # destructor a no-op instead of another reset/go_home.
            if arm is not None:
                arm.robot = None
            if hand is not None:
                hand.robot = None
            if hasattr(backend, "is_connected"):
                backend.is_connected = False
        self.connected = False
        self.robot = None
        self._gripper_cmd_state = self._new_gripper_cmd_state()

    def get_joint_positions(self):
        if self.robot is None or not self.connected:
            raise RuntimeError("Robot is not connected.")
        return (
            self.robot.robot_arm.get_state()
            .joint_pos.detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

    def get_joint_state(self):
        if self.robot is None or not self.connected:
            raise RuntimeError("Robot is not connected.")
        state = self.robot.robot_arm.get_state()
        q = state.joint_pos.detach().cpu().numpy().astype(np.float64)
        dq = state.joint_vel.detach().cpu().numpy().astype(np.float64)
        return q, dq

    def send_joint_positions(self, q_cmd, qd_cmd=None):
        if self.robot is None or not self.connected:
            raise RuntimeError("Robot is not connected.")
        q_cmd = np.asarray(q_cmd, dtype=np.float64)
        if qd_cmd is None:
            self.robot.robot_arm.apply_commands(q_desired=q_cmd)
        else:
            self.robot.robot_arm.apply_commands(
                q_desired=q_cmd,
                qd_desired=np.asarray(qd_cmd, dtype=np.float64),
            )

    def go_to_joint_positions(self, q_cmd, max_vel_norm_factor=None, duration_s=None):
        if self.robot is None or not self.connected:
            raise RuntimeError("Robot is not connected.")
        q_cmd = np.asarray(q_cmd, dtype=np.float64)
        if duration_s is None:
            duration_s = float(os.environ.get("INTERVENE_REPLAN_GOTO_SECONDS", "4.0"))
        duration_s = float(duration_s)
        if duration_s > 0.0:
            self._go_to_joint_positions_over_time(q_cmd, duration_s)
            return

        if max_vel_norm_factor is None:
            max_vel_norm_factor = float(
                os.environ.get("INTERVENE_REPLAN_GOTO_SPEED_FACTOR", "0.25")
            )
        print(f"[ROBOT-ADAPTER] Moving arm with speed factor {max_vel_norm_factor:.2f}.")
        self.robot.robot_arm.go_to_within_limits(
            q_cmd,
            max_vel_norm_factor=float(max_vel_norm_factor),
        )

    def _go_to_joint_positions_over_time(self, q_goal, duration_s):
        q_start = self.get_joint_positions()
        hz = float(getattr(self.robot.robot_arm, "hz", 60.0) or 60.0)
        dt = 1.0 / hz
        n_steps = max(1, int(round(float(duration_s) * hz)))
        delta = q_goal - q_start

        print(f"[ROBOT-ADAPTER] Moving arm over {duration_s:.1f}s.")
        for i in range(1, n_steps + 1):
            u = i / n_steps
            smooth = u * u * (3.0 - 2.0 * u)
            q_cmd = q_start + smooth * delta
            if i < n_steps:
                smooth_dot = 6.0 * u * (1.0 - u) / duration_s
                qd_cmd = smooth_dot * delta
            else:
                qd_cmd = np.zeros_like(delta)
            self.robot.robot_arm.apply_commands(
                q_desired=q_cmd,
                qd_desired=qd_cmd,
            )
            time.sleep(dt)

    def switch_control_mode(self, control_mode):
        """Change the ARM's control mode on the connection we already have.

        This is the ~1 s gap the operator sees between "arm aligned" and "you may take
        over". It was never the Franka withholding control -- it is our own teardown and
        rebuild. `Robot.connect(mode)` does three things:

            1. build a brand-new RobotInterface (fresh gRPC channel + robot-model fetch)
            2. build a brand-new GripperInterface
            3. send the new torch policy   <- the only step that changes the mode

        Step 2 has nothing to do with an arm mode switch, and step 1 is redundant on a
        live connection (it also leaked a channel and a robot model per takeover, since
        the old RobotInterface is never closed). `light_polymetis_adapter` has always
        switched modes by sending a policy on the existing connection; this does the same.

        Falls back to the full reconnect on ANY failure, so the worst case is the old
        behaviour rather than a broken takeover. INTERVENE_FAST_CONTROL_SWITCH=0 forces
        the old path -- use it if the arm ever fails to enter freedrive.
        """
        if self.robot is None or not self.connected:
            raise RuntimeError("Robot is not connected.")
        record_module = _get_record_backend()
        if record_module is None:
            raise RuntimeError(f"Robot backend unavailable: {_record_error}")
        mode = _resolve_control_mode(record_module, control_mode)

        t0 = time.time()
        if _fast_control_switch_enabled():
            arm = getattr(self.robot, "robot_arm", None)
            # `arm.robot` is the live RobotInterface; without one there is nothing to
            # reuse and the full connect is the only option.
            if arm is not None and getattr(arm, "robot", None) is not None:
                try:
                    arm.control_type = mode
                    arm.set_policy()
                    print(f"[ROBOT-ADAPTER] Control mode -> {mode} "
                          f"in {time.time() - t0:.2f}s (policy swap, no reconnect).")
                    return
                except Exception as exc:
                    print(f"[ROBOT-ADAPTER][WARN] fast control-mode switch failed "
                          f"({exc}); falling back to a full reconnect.")

        self.robot.connect(mode)
        print(f"[ROBOT-ADAPTER] Control mode -> {mode} "
              f"in {time.time() - t0:.2f}s (full reconnect).")

    def get_gripper_width(self):
        if self.robot is None or not self.connected:
            raise RuntimeError("Robot is not connected.")
        return float(self.robot.robot_gripper.get_sensors().item())

    def send_gripper_width(self, width):
        if self.robot is None or not self.connected:
            raise RuntimeError("Robot is not connected.")

        hand = self.robot.robot_gripper
        min_width = float(getattr(hand, "min_width", 0.0))
        max_width = float(getattr(hand, "max_width", 0.08) or 0.08)
        target = float(np.clip(width, min_width, max_width))
        now = time.time()

        last_width_cmd = self._gripper_cmd_state.get("last_width_cmd", None)
        last_cmd_t = float(self._gripper_cmd_state.get("last_cmd_t", 0.0))

        if (
            last_width_cmd is not None
            and abs(target - float(last_width_cmd)) < self.gripper_track_eps_m
            and (now - last_cmd_t) < self.gripper_cmd_period_s
        ):
            return

        try:
            low_level = getattr(hand, "robot", None)
            if low_level is not None and hasattr(low_level, "goto"):
                pool = getattr(hand, "pool", None)
                fut = self._gripper_cmd_state.get("future", None)

                if pool is not None:
                    if fut is not None and not fut.done():
                        return
                    self._gripper_cmd_state["future"] = pool.submit(
                        low_level.goto,
                        target,
                        float(self.gripper_speed),
                        float(self.gripper_force),
                    )
                else:
                    low_level.goto(
                        target,
                        float(self.gripper_speed),
                        float(self.gripper_force),
                    )
            else:
                current_width = float(hand.get_sensors().item())
                err = target - current_width
                if abs(err) < self.gripper_track_eps_m:
                    return

                cmd = 1.0 if err > 0.0 else -1.0
                hand.apply_commands(
                    cmd,
                    speed=float(self.gripper_speed),
                    force=float(self.gripper_force),
                )

        except Exception as e:
            last_error_t = float(self._gripper_cmd_state.get("last_error_t", 0.0))
            if (now - last_error_t) > 1.0:
                print(f"[ROBOT-ADAPTER] Gripper command failed: {e}")
                self._gripper_cmd_state["last_error_t"] = now
            return

        self._gripper_cmd_state["last_width_cmd"] = target
        self._gripper_cmd_state["last_cmd_t"] = now

    def restore_gripper_width(self, target_width, tol=0.002, timeout=1.5, hz=40.0):
        if self.robot is None or not self.connected:
            raise RuntimeError("Robot is not connected.")

        hand = self.robot.robot_gripper
        min_width = float(getattr(hand, "min_width", 0.0))
        max_width = float(getattr(hand, "max_width", 0.08) or 0.08)
        target_width = float(np.clip(target_width, min_width, max_width))

        dt = 1.0 / hz
        t_end = time.time() + timeout

        while time.time() < t_end:
            current_width = float(hand.get_sensors().item())
            err = target_width - current_width

            if abs(err) <= tol:
                break

            self.send_gripper_width(target_width)
            time.sleep(dt)

        final_width = float(hand.get_sensors().item())
        print(
            f"[ROBOT-ADAPTER] Gripper restored: "
            f"target={target_width:.4f}, final={final_width:.4f}"
        )
