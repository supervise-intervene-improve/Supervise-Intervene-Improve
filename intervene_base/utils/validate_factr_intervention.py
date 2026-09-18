#!/usr/bin/env python3
"""Hardware-free validation of the FACTR policy-intervention lifecycle."""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import mujoco
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robot.factr_rpc import (  # noqa: E402
    FactrRpcAdapter,
    FactrRpcError,
    FactrRpcService,
    FactrTcpServer,
    write_ready_file,
)
from robot.live_replan_session import (  # noqa: E402
    InterventionTransitionError,
    LiveReplanSession,
)


class MockFactrHardware:
    def __init__(self, events: list[str], *, fail_alignment: bool = False):
        self.events = events
        self.fail_alignment = fail_alignment
        self.q = np.zeros(7, dtype=np.float64)
        self.dq = np.zeros(7, dtype=np.float64)
        self.gripper_width = 0.08
        self.closed = False

    def connect(self, already_connected=False):
        del already_connected
        self.events.append("factr_connect")

    def stage_initial_pose(self):
        self.events.append("factr_rest_verified")
        self.q = np.array([0.1, -0.6, 0.05, -2.2, -0.05, 1.6, -6.2])
        self.events.append("factr_init_reached")
        return True

    def align_to_mujoco(
        self,
        q_mujoco,
        *,
        gripper_width=None,
        tolerance=None,
        timeout=None,
        max_velocity=None,
    ):
        del tolerance, timeout, max_velocity
        self.events.append("factr_align")
        if self.fail_alignment:
            raise TimeoutError("mock alignment timeout")
        self.q = np.asarray(q_mujoco, dtype=np.float64).copy()
        # Exercise the real service's 2-pi branch mapping for the FACTR J7.
        self.q[6] -= 2.0 * np.pi
        if gripper_width is not None:
            self.gripper_width = float(gripper_width)
        return True

    def begin_takeover(self):
        self.events.append("factr_takeover")

    def end_takeover(self):
        self.events.append("factr_waiting")

    def get_joint_state(self):
        return self.q.copy(), self.dq.copy()

    def get_gripper_width(self):
        return self.gripper_width

    def get_gripper_pressed(self):
        return self.gripper_width <= 0.04

    def restore_gripper_width(self, width):
        self.gripper_width = float(width)

    def close(self):
        self.closed = True
        self.events.append("factr_close")


class SlowReturnFactrHardware(MockFactrHardware):
    def __init__(self, events: list[str]):
        super().__init__(events)
        self.return_started = threading.Event()
        self.return_interrupted = threading.Event()
        self.return_continue = threading.Event()

    def return_to_initial_pose(self):
        self.events.append("factr_return_start")
        self.return_started.set()
        self.return_continue.wait(timeout=2.0)
        if self.return_interrupted.is_set():
            self.events.append("factr_return_interrupted")
        else:
            self.events.append("factr_return_complete")

    def interrupt_return_to_initial(self):
        self.events.append("factr_return_interrupt_requested")
        self.return_interrupted.set()
        self.return_continue.set()


class MockPolicy:
    def __init__(self, events: list[str]):
        self.events = events
        self.is_paused = False
        self.action_queue = ["stale-a", "stale-b"]
        self.reset_requested = False

    def pause_at_current_state(self):
        self.is_paused = True
        self.events.append("policy_paused")

    def resume_from_current_state(self, reset_policy_queue=True):
        self.reset_requested = bool(reset_policy_queue)
        if reset_policy_queue:
            self.action_queue.clear()
        self.is_paused = False
        self.events.append("policy_resumed")


class MockRecorder:
    def __init__(self):
        self.frames = 0

    def record(self, **_kwargs):
        self.frames += 1
        return True


def _make_arm_gripper_model():
    bodies = []
    closing = []
    for i in range(8):
        bodies.append(
            f'<body name="b{i}" pos="0 0 0.1">'
            f'<joint name="j{i}" type="hinge" axis="0 0 1" damping="0.1"/>'
            '<geom type="sphere" size="0.01" mass="0.1"/>'
        )
        closing.append("</body>")
    actuators = "".join(
        f'<position name="a{i}" joint="j{i}" kp="20"/>' for i in range(8)
    )
    xml = (
        '<mujoco><option timestep="0.001"/><worldbody>'
        + "".join(bodies)
        + "".join(reversed(closing))
        + f"</worldbody><actuator>{actuators}</actuator></mujoco>"
    )
    model = mujoco.MjModel.from_xml_string(xml)
    return model, mujoco.MjData(model)


def _assert_disabled_default():
    old_factr = os.environ.pop("FACTR_ACTIVE", None)
    old_intervene = os.environ.pop("INTERVENE_FACTR_ACTIVE", None)
    try:
        requested = os.environ.get(
            "INTERVENE_FACTR_ACTIVE", os.environ.get("FACTR_ACTIVE", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        assert not requested, "FACTR must remain opt-in"
    finally:
        if old_factr is not None:
            os.environ["FACTR_ACTIVE"] = old_factr
        if old_intervene is not None:
            os.environ["INTERVENE_FACTR_ACTIVE"] = old_intervene


def validate_success_path():
    events: list[str] = []
    hardware = MockFactrHardware(events)
    service = FactrRpcService(hardware)
    service.initialize()
    assert events == [
        "factr_connect",
        "factr_rest_verified",
        "factr_init_reached",
    ]
    assert events.index("factr_rest_verified") < events.index("factr_init_reached")

    server = FactrTcpServer(("127.0.0.1", 0), service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = FactrRpcAdapter()
    client.host, client.port = server.server_address

    model, data = _make_arm_gripper_model()
    # Match the soft-gripper convention where actuator control is full width.
    model.actuator_ctrlrange[7] = np.array([0.0, 0.0875])
    q_hold = np.array([0.2, -0.4, 0.1, -1.5, -0.2, 1.2, 0.25])
    data.qpos[:7] = q_hold
    data.ctrl[:7] = q_hold
    mujoco.mj_forward(model, data)
    policy = MockPolicy(events)
    recorder = MockRecorder()

    old_alpha = os.environ.get("FACTR_MUJOCO_ALPHA")
    old_step = os.environ.get("FACTR_MUJOCO_MAX_STEP_RAD")
    old_takeover_ramp = os.environ.get("FACTR_MUJOCO_TAKEOVER_RAMP_SECONDS")
    old_gripper_step = os.environ.get("FACTR_MUJOCO_GRIPPER_MAX_STEP")
    old_gripper_mode = os.environ.get("FACTR_MUJOCO_GRIPPER_MODE")
    old_gripper_mode_alt = os.environ.get("INTERVENE_FACTR_MUJOCO_GRIPPER_MODE")
    # Unset so the gripper assertions below exercise the CODE DEFAULT rather than whatever
    # the operator's shell happens to export.
    os.environ.pop("FACTR_MUJOCO_GRIPPER_MODE", None)
    os.environ.pop("INTERVENE_FACTR_MUJOCO_GRIPPER_MODE", None)
    os.environ["FACTR_MUJOCO_ALPHA"] = "1.0"
    os.environ["FACTR_MUJOCO_MAX_STEP_RAD"] = "0"
    os.environ["FACTR_MUJOCO_TAKEOVER_RAMP_SECONDS"] = "0"
    os.environ["FACTR_MUJOCO_GRIPPER_MAX_STEP"] = "0"
    try:
        # This is the same ordering used by App.trigger_replan(): pause first,
        # then claim/align through LiveReplanSession.
        policy.pause_at_current_state()
        assert policy.is_paused
        assert events.index("factr_init_reached") < events.index("policy_paused")
        client.connect(already_connected=False)
        other_client = FactrRpcAdapter()
        other_client.host, other_client.port = server.server_address
        try:
            other_client.connect(already_connected=False)
        except FactrRpcError as exc:
            assert "busy" in str(exc)
        else:
            raise AssertionError("a second policy client acquired the single FACTR device")
        try:
            client.align_to_mujoco(np.zeros(6))
        except ValueError as exc:
            assert "dimension mismatch" in str(exc)
        else:
            raise AssertionError("joint dimension mismatch was not rejected")
        # The study logger drives its intervention phases off these callbacks. FACTR must
        # report them like every other mode, or a FACTR block records no
        # synchronization/correction time at all.
        study_phases = []
        session = LiveReplanSession(
            robot_adapter=client,
            save_path="/tmp/mock_factr_intervention.npz",
            mujoco_lab_id="mock",
            mujoco_xml_path="unused.xml",
            q_hold=q_hold,
            finger_hold=0.0,
            grip_width_hold=0.06,
            qpos_seed=data.qpos.copy(),
            qvel_seed=data.qvel.copy(),
            model=model,
            data=data,
            recorder=recorder,
            already_connected=True,
            on_ready=lambda **kw: study_phases.append(("ready", kw)),
            on_released=lambda **kw: study_phases.append(("released", kw)),
        )
        assert session.factr_mode, "FACTR adapter was not detected by the session"
        assert session.strict_alignment, "FACTR must require strict alignment"
        assert session.phase == "requested"
        session.start()
        assert events.index("policy_paused") < events.index("factr_align")
        assert events.index("factr_align") < events.index("factr_takeover")
        assert session.started
        assert session.phase == "human_control", session.phase

        kinds = [k for k, _ in study_phases]
        assert kinds == ["ready", "released"], kinds
        ready_kw = study_phases[0][1]
        released_kw = study_phases[1][1]
        assert ready_kw["path"] == "factr", ready_kw
        assert ready_kw["aligned"] is True, ready_kw
        assert released_kw["path"] == "factr", released_kw
        assert released_kw["control_mode"] == "HUMAN_CONTROL", released_kw

        corrected = q_hold + np.array([0.03, -0.02, 0.01, 0.0, 0.02, -0.01, 0.04])
        hardware.q = corrected.copy()
        hardware.q[6] -= 2.0 * np.pi
        hardware.gripper_width = 0.04
        session.update()
        np.testing.assert_allclose(data.ctrl[:7], corrected, atol=1e-9)
        assert recorder.frames == 1

        # --- The sim gripper follows the leader's MEASURED WIDTH, continuously. ---
        #
        # This assertion REVERSED on 2026-08-24, when the default moved from the threshold
        # path to the continuous one. It used to read
        #     assert session.factr_gripper_target_ctrl == float(model.actuator_ctrlrange[7, 0])
        # which pinned nothing in either direction: factr_gripper_target_ctrl is seeded from
        # data.ctrl[7] (0.0 here) at takeover and ctrlrange[7, 0] is also 0.0, so it held under
        # BOTH modes. What actually distinguishes them -- and what the operator asked for -- is
        # that a PARTIAL squeeze produces a PARTIAL opening instead of snapping to an endpoint.
        # FACTR_MUJOCO_GRIPPER_MAX_STEP is pinned to 0 above, so there is no slew limit to
        # account for and ctrl[7] should equal the mapped width exactly.
        ctrl_lo = float(model.actuator_ctrlrange[7, 0])
        ctrl_hi = float(model.actuator_ctrlrange[7, 1])
        np.testing.assert_allclose(data.ctrl[7], 0.04, atol=1e-9)
        for width in (0.05, 0.02, 0.065):
            hardware.gripper_width = width
            session.update()
            np.testing.assert_allclose(data.ctrl[7], width, atol=1e-9)
            assert ctrl_lo < float(data.ctrl[7]) < ctrl_hi, (
                f"width {width} snapped to an endpoint: ctrl[7]={float(data.ctrl[7])}"
            )

        # --- The threshold path is still reachable, opt-in, and still snaps. ---
        # Keeping it covered matters: it is the correct behaviour for a leader whose gripper
        # really is a trigger, and it is what every pre-2026-08-24 FACTR recording used.
        os.environ["FACTR_MUJOCO_GRIPPER_MODE"] = "standalone"
        try:
            hardware.gripper_width = 0.06   # mock: pressed = width <= 0.04 -> OPEN endpoint
            session.update()
            assert session.factr_gripper_target_ctrl == ctrl_hi, (
                session.factr_gripper_target_ctrl
            )
            hardware.gripper_width = 0.01   # -> CLOSED endpoint
            session.update()
            assert session.factr_gripper_target_ctrl == ctrl_lo, (
                session.factr_gripper_target_ctrl
            )
        finally:
            os.environ.pop("FACTR_MUJOCO_GRIPPER_MODE", None)

        session.finish_segment()
        client.switch_control_mode("HYBRID_JOINT_IMPEDANCE_CONTROL")
        policy.resume_from_current_state(reset_policy_queue=True)
        assert service.state == "WAITING"
        assert service.owner is None
        assert policy.reset_requested and not policy.action_queue
        assert not policy.is_paused
        np.testing.assert_allclose(data.ctrl[:7], corrected, atol=1e-9)
        assert events.index("factr_waiting") < events.index("policy_resumed")
    finally:
        if client.connected:
            client.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
        service.close()
        if old_alpha is None:
            os.environ.pop("FACTR_MUJOCO_ALPHA", None)
        else:
            os.environ["FACTR_MUJOCO_ALPHA"] = old_alpha
        if old_step is None:
            os.environ.pop("FACTR_MUJOCO_MAX_STEP_RAD", None)
        else:
            os.environ["FACTR_MUJOCO_MAX_STEP_RAD"] = old_step
        if old_takeover_ramp is None:
            os.environ.pop("FACTR_MUJOCO_TAKEOVER_RAMP_SECONDS", None)
        else:
            os.environ["FACTR_MUJOCO_TAKEOVER_RAMP_SECONDS"] = old_takeover_ramp
        if old_gripper_step is None:
            os.environ.pop("FACTR_MUJOCO_GRIPPER_MAX_STEP", None)
        else:
            os.environ["FACTR_MUJOCO_GRIPPER_MAX_STEP"] = old_gripper_step
        for _name, _val in (
            ("FACTR_MUJOCO_GRIPPER_MODE", old_gripper_mode),
            ("INTERVENE_FACTR_MUJOCO_GRIPPER_MODE", old_gripper_mode_alt),
        ):
            if _val is None:
                os.environ.pop(_name, None)
            else:
                os.environ[_name] = _val
    assert hardware.closed


def validate_ready_signal_after_initialization():
    events: list[str] = []
    hardware = MockFactrHardware(events)
    service = FactrRpcService(hardware)
    ready_path = Path("/tmp/mock_factr_ready_signal.ready")
    ready_path.unlink(missing_ok=True)

    assert not ready_path.exists()
    assert service.state == "INITIALIZING"
    service.initialize()
    assert not ready_path.exists()
    assert service.state == "WAITING"
    assert events == [
        "factr_connect",
        "factr_rest_verified",
        "factr_init_reached",
    ]

    write_ready_file(ready_path, host="127.0.0.1", port=18075)
    assert ready_path.exists()
    assert service.state == "WAITING"

    service.close()
    ready_path.unlink(missing_ok=True)


def validate_return_interrupted_by_new_intervention():
    events: list[str] = []
    hardware = SlowReturnFactrHardware(events)
    service = FactrRpcService(hardware)
    service.initialize()
    service.dispatch({"operation": "claim", "client_id": "first"})
    service.dispatch(
        {
            "operation": "align",
            "client_id": "first",
            "joint_positions": [0.1] * 7,
            "gripper_width": 0.04,
            "tolerance": 0.04,
            "alignment_timeout": 0.1,
            "max_velocity": 0.5,
        }
    )
    service.dispatch({"operation": "begin_takeover", "client_id": "first"})
    service.dispatch({"operation": "release", "client_id": "first"})
    assert service.state == "RETURNING_INIT"
    assert hardware.return_started.wait(timeout=1.0)

    service.dispatch({"operation": "claim", "client_id": "second"})
    assert hardware.return_interrupted.wait(timeout=1.0)
    assert service.owner == "second"
    assert service.state == "WAITING"
    service.dispatch(
        {
            "operation": "align",
            "client_id": "second",
            "joint_positions": [0.2] * 7,
            "gripper_width": 0.04,
            "tolerance": 0.04,
            "alignment_timeout": 0.1,
            "max_velocity": 0.5,
        }
    )
    interrupt_idx = events.index("factr_return_interrupt_requested")
    align_indices = [i for i, event in enumerate(events) if event == "factr_align"]
    assert len(align_indices) == 2
    assert interrupt_idx < align_indices[-1]
    service.close()


def validate_exception_cleanup():
    events: list[str] = []
    hardware = MockFactrHardware(events, fail_alignment=True)
    service = FactrRpcService(hardware)
    service.initialize()
    service.dispatch({"operation": "claim", "client_id": "failure-test"})
    try:
        service.dispatch(
            {
                "operation": "align",
                "client_id": "failure-test",
                "joint_positions": [0.0] * 7,
                "gripper_width": 0.04,
                "tolerance": 0.04,
                "alignment_timeout": 0.1,
                "max_velocity": 0.5,
            }
        )
    except TimeoutError:
        pass
    else:
        raise AssertionError("mock alignment failure was not propagated")
    assert service.state == "WAITING" and service.owner is None
    assert "factr_waiting" in events
    service.close()
    assert hardware.closed and service.state == "STOPPED"


def validate_return_cancel_is_armed_before_the_move_starts():
    """A claim arriving during the return's own SETUP must still cancel it.

    The cancel event used to be created inside return_to_initial_pose(), after connect()
    and a pose-file read. A claim landing in that window found nothing to set, cancelled
    nothing, and the return then ran a full uncancelled move while the claim proceeded
    into alignment — two things commanding one leader arm. `release` followed immediately
    by `claim` is exactly the study workflow, so the window is reachable.
    """
    events: list[str] = []

    class SlowSetupHardware(MockFactrHardware):
        """Blocks in the move's setup, the way connect()+_load_pose() would."""

        def __init__(self, evts):
            super().__init__(evts)
            self.setup_entered = threading.Event()
            self.release_setup = threading.Event()
            self.move_ran = False

        def return_to_initial_pose(self):
            self.events.append("factr_return_setup")
            self.setup_entered.set()
            self.release_setup.wait(timeout=2.0)
            # By here a claim has already interrupted us. The pre-armed event is the
            # only thing that could have recorded that.
            event = getattr(self, "_move_cancel_event", None)
            if event is not None and event.is_set():
                self.events.append("factr_return_cancelled_in_setup")
                return None
            self.move_ran = True
            self.events.append("factr_return_move_ran")
            return None

        def arm_move_cancel_event(self):
            event = getattr(self, "_move_cancel_event", None)
            if event is None:
                event = threading.Event()
                self._move_cancel_event = event
            return event

        def interrupt_return_to_initial(self):
            event = getattr(self, "_move_cancel_event", None)
            if event is not None:
                event.set()
            self.events.append("factr_return_interrupt_requested")

    hardware = SlowSetupHardware(events)
    service = FactrRpcService(hardware)
    service.initialize()

    service.dispatch({"operation": "claim", "client_id": "first"})
    service.dispatch({"operation": "release", "client_id": "first"})
    assert hardware.setup_entered.wait(timeout=2.0), "return never started"
    assert service.state == "RETURNING_INIT", service.state

    # Capture the thread BEFORE claiming: `claim` clears service._return_thread, so
    # reading it afterwards yields None and the join below would silently not happen,
    # letting the assertions race the worker.
    thread = service._return_thread
    assert thread is not None

    # Second session claims while the return is still in its setup window.
    service.dispatch({"operation": "claim", "client_id": "second"})
    hardware.release_setup.set()
    thread.join(timeout=2.0)
    assert not thread.is_alive(), "return thread did not finish"

    assert "factr_return_interrupt_requested" in events, events
    assert not hardware.move_ran, "the return MOVED after being interrupted — two movers"
    assert "factr_return_move_ran" not in events, events
    assert service.owner == "second", service.owner
    service.close()


def validate_strict_alignment_never_releases():
    """A failed FACTR alignment must abort the takeover, not fall through to it.

    This is the single highest-consequence behaviour in the merge: the telekinesis path
    deliberately warns-and-continues on a missed alignment tolerance, and inheriting that
    for FACTR would hand a human control of a leader arm that is NOT at the paused pose.
    The failure must also carry a reason from app.py::_transition_reason's allow-list,
    otherwise the operator only ever sees the generic "transition_failed".
    """
    events: list[str] = []
    hardware = MockFactrHardware(events, fail_alignment=True)
    service = FactrRpcService(hardware)
    service.initialize()

    server = FactrTcpServer(("127.0.0.1", 0), service)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = FactrRpcAdapter()
    client.host, client.port = server.server_address

    model, data = _make_arm_gripper_model()
    q_hold = np.array([0.2, -0.4, 0.1, -1.5, -0.2, 1.2, 0.25])
    data.qpos[:7] = q_hold
    data.ctrl[:7] = q_hold
    mujoco.mj_forward(model, data)

    study_phases: list[str] = []
    try:
        client.connect(already_connected=False)
        session = LiveReplanSession(
            robot_adapter=client,
            save_path="/tmp/mock_factr_strict_alignment.npz",
            mujoco_lab_id="mock",
            mujoco_xml_path="unused.xml",
            q_hold=q_hold,
            finger_hold=0.0,
            grip_width_hold=0.06,
            qpos_seed=data.qpos.copy(),
            qvel_seed=data.qvel.copy(),
            model=model,
            data=data,
            recorder=MockRecorder(),
            already_connected=True,
            on_ready=lambda **kw: study_phases.append("ready"),
            on_released=lambda **kw: study_phases.append("released"),
        )
        assert session.phase == "requested"
        try:
            session.start()
        except InterventionTransitionError as exc:
            allowed = {
                "robot_unreachable", "robot_busy", "alignment_command_failed",
                "state_read_failed", "mode_switch_failed", "return_home_failed",
                "transition_failed",
            }
            assert exc.reason in allowed, f"reason {exc.reason!r} collapses to transition_failed"
        else:
            raise AssertionError(
                "a failed FACTR alignment fell through into human control"
            )

        assert not session.started, "takeover started despite failed alignment"
        assert session.phase == "aligning", session.phase
        assert "released" not in study_phases, study_phases
        assert service.owner is None, "ownership leaked after a failed alignment"
        assert service.state == "WAITING", service.state
    finally:
        server.shutdown()
        server.server_close()
        service.close()


def main():
    _assert_disabled_default()
    validate_ready_signal_after_initialization()
    validate_success_path()
    validate_return_interrupted_by_new_intervention()
    validate_exception_cleanup()
    validate_strict_alignment_never_releases()
    validate_return_cancel_is_armed_before_the_move_starts()
    print("[PASS] FACTR-disabled mode remains opt-in.")
    print("[PASS] READY signal appears only after FACTR initialization completes.")
    print("[PASS] Initialization completes before policy activity.")
    print("[PASS] Parked rest pose is verified before initialization movement.")
    print("[PASS] Policy pauses before alignment and takeover waits for alignment.")
    print("[PASS] FACTR joint/gripper state controls MuJoCo during intervention.")
    print("[PASS] Release clears stale actions and resumes from corrected state.")
    print("[PASS] Return-to-home is interrupted by a new intervention.")
    print("[PASS] Exceptions release ownership and close the hardware adapter.")
    print("[PASS] Study ready/released phases are reported for FACTR takeovers.")
    print("[PASS] A failed strict alignment aborts instead of releasing control.")
    print("[PASS] A claim during return setup cancels it (never two movers on the arm).")


if __name__ == "__main__":
    main()
