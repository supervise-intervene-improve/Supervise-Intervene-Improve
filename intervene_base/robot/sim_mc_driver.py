import os
import threading
import time

import mujoco
import numpy as np

from mq3_mc_mujoco import (
    CARTESIAN_ORI_DEADBAND_DEG,
    CARTESIAN_POS_DEADBAND_M,
    CLUTCH_ENGAGE_SETTLE_S,
    CONTROL_HAND,
    CONTROL_MODE,
    GRIPPER_CONTROL_BUTTON,
    JUMP_RESET_ORI_DEG,
    JUMP_RESET_POS_M,
    MC_SMOOTH_POS_ALPHA,
    MC_SMOOTH_ROT_ALPHA,
    ORIENTATION_STEP_CLAMP_DEG,
    ORIENTATION_TRIGGER_THRESHOLD,
    POSITION_STEP_CLAMP_M,
    POSITION_TRIGGER_THRESHOLD,
    POS_MIRROR_XY,
    QUAT_MIRROR_Z180,
    READ_SIM_POSE_ON_TRIGGER_EDGE,
    MetaQuest3ZmqReceiver,
    MujocoFrankaController,
    clamp_step_pos,
    clamp_step_quat,
    is_button_pressed,
    normalize_quat,
    quat_angle_deg_xyzw,
    quat_inv_xyzw,
    quat_mul_xyzw,
    quat_slerp_xyzw,
)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


class SimMcDriver:
    """Drives the LIVE policy's simulated Franka arm via local IK from Quest
    MotionController input. No real robot involved.

    Reuses mq3_mc_mujoco.MujocoFrankaController (imported, unmodified) bound to
    the SAME model/data the running policy already owns. The per-tick clutch
    logic below is a ported (not shared) copy of mq3_mc_mujoco.run_teleop_session's
    body (source lines ~910-1136 as of this change) adapted from a blocking loop
    to a single tick() call driven by LiveReplanSession.update().
    """

    def __init__(self, model, data, *, site_name="gripper", discovery_timeout_s=20.0):
        self.site_name = site_name
        self.mq3 = MetaQuest3ZmqReceiver(discovery_timeout_s=discovery_timeout_s)
        self._connected = False
        self._discovered = False
        self._discover_thread = None
        self._discovery_failed = False
        self._connect_error = None

        self._gripper_edge_latch = _env_bool("MC_SIM_GRIPPER_EDGE_LATCH", True)
        self._gripper_latched = False
        self._gripper_button_baseline = None

        # Link health. Every failure mode between "Quest attached" and "the arm moves"
        # was previously silent: tick() returns early both when no packet has arrived
        # and when the operator is simply not squeezing, and the two are
        # indistinguishable from the log. These counters make the next takeover say
        # which one it is instead of costing another session to guess.
        self._diag_period_s = float(os.environ.get("MC_SIM_LINK_LOG_S", "1.0"))
        self._reset_link_diagnostics()

        self._build_controller(model, data)
        self.reset_clutch_state()

    def _build_controller(self, model, data):
        """Construct the controller against `model`/`data` without disturbing the scene.

        MujocoFrankaController.__init__ -> initialize() calls mujoco.mj_resetData()
        unconditionally, wiping the ENTIRE live data (all qpos/qvel/ctrl/time), not
        just the 7 arm joints. Snapshot + restore around construction so this is
        correct regardless of what the live scene currently holds.
        """
        qpos_snap = np.asarray(data.qpos, dtype=np.float64).copy()
        qvel_snap = np.asarray(data.qvel, dtype=np.float64).copy()
        ctrl_snap = np.asarray(data.ctrl, dtype=np.float64).copy()
        time_snap = float(data.time)

        self.controller = MujocoFrankaController(model, data, site_name=self.site_name)

        data.qpos[:] = qpos_snap
        data.qvel[:] = qvel_snap
        data.ctrl[:] = ctrl_snap
        data.time = time_snap
        mujoco.mj_forward(model, data)

        self.model = model
        self.data = data
        # initialize() ends with set_gripper_open(immediate=True), so the controller's
        # gripper TARGET is open even though we just restored data.ctrl. step_sim() then
        # ramps ctrl[7] toward that target over gripper_open_time (0.85 s) on every
        # intervention tick — silently opening the hand and dropping whatever the policy
        # was holding. The ctrl restore above does not cover it; this does.
        self._seed_gripper_target_from_live_ctrl()

    def _seed_gripper_target_from_live_ctrl(self):
        idx = self.controller.gripper_ctrl_index
        if idx is None or idx >= self.data.ctrl.shape[0]:
            return
        self.controller.gripper_target_ctrl = float(self.data.ctrl[idx])

    def _reset_link_diagnostics(self):
        """Per-takeover counters. Reset by resync_from_live_state, not by rebind: a model
        swap keeps the same link, and zeroing there would hide a stream that died."""
        self._attached_wall = None
        self._rx_last_wall = None
        self._cmds_issued = 0
        self._diag_last_log = 0.0
        self._diag_last_rx = 0
        self._no_data_warned = False
        self._last_index_trigger = 0.0
        self._last_hand_trigger = 0.0

    def resync_from_live_state(self):
        """Adopt the live sim's current gripper/EE state as this takeover's baseline.

        The driver is cached for the whole session (App.trigger_replan), so __init__ runs
        at most once — every SUBSEQUENT takeover would otherwise inherit the previous
        one's gripper target and clutch offsets.
        """
        self._seed_gripper_target_from_live_ctrl()
        self._gripper_latched = False
        self._gripper_button_baseline = None
        self._discovery_failed = False      # re-arm discovery for this takeover
        self._reset_link_diagnostics()
        self.reset_clutch_state()

    def rebind(self, model, data):
        """Re-point at a swapped model/data (OOD variant episode).

        PolicyPlayer.swap_model allocates a brand-new MjData, so a cached driver would
        keep driving the dead one: the arm never moves and the recording is a frozen
        pose. Discovery state is deliberately preserved — a model swap must not cost a
        fresh 20 s Quest rediscovery.
        """
        self._build_controller(model, data)
        self.resync_from_live_state()

    def ensure_connected(self, *, blocking=False):
        """Attach to the Quest MotionController stream. Non-blocking by default.

        Returns True iff the SUB socket is live. Discovery (a UDP recv loop of up to
        discovery_timeout_s) runs on a worker thread; only `discover_meta_quest` runs
        there, and the SUB socket itself is always created on the calling thread — no
        ZMQ object ever crosses threads. `_discovered` is written LAST by the worker, so
        observing it True is a sufficient publication barrier for quest_ip/zmq_port/topic.
        """
        if self._connected:
            return True

        if blocking:
            print("[MC-SIM] Waiting for Quest MotionController discovery (up to "
                  f"{self.mq3.discovery_timeout_s:.0f}s)...")
            self.mq3.connect()
            self._connected = True
            self._attached_wall = time.time()
            return True

        if self._discovered:
            try:
                # connect() short-circuits discovery when quest_ip is already set.
                self.mq3.connect()
                self._connected = True
                self._attached_wall = time.time()
                print(f"[MC-SIM] Quest attached at {self.mq3.zmq_address}.")
            except Exception as exc:
                self._connect_error = exc
                print(f"[MC-SIM][WARN] Quest socket setup failed: {exc}")
            return self._connected

        # ONE attempt per takeover. tick() calls this every frame, so without the flag a
        # failed discovery would relaunch another 20 s UDP listen forever. The listen
        # window itself is what lets a late headset attach: anything that starts
        # broadcasting during it is picked up. After it expires the operator re-presses
        # the intervention button, and resync_from_live_state re-arms this.
        if self._discovery_failed:
            return False

        if self._discover_thread is None or not self._discover_thread.is_alive():
            self._connect_error = None
            self._discover_thread = threading.Thread(
                target=self._discover_worker,
                daemon=True,
                name="SimMcQuestDiscovery",
            )
            self._discover_thread.start()
        return False

    def _discover_worker(self):
        try:
            self.mq3.discover_meta_quest()
            self._discovered = True     # published LAST, after quest_ip/zmq_port/topic
        except Exception as exc:
            self._connect_error = exc
            self._discovery_failed = True
            print(f"[MC-SIM][WARN] Quest discovery failed: {exc} "
                  "The arm holds its pose; press the intervention button again to retry.")

    def reset_clutch_state(self):
        self.last_cmd_pos, self.last_cmd_quat = self.controller.get_ee_pose()

        self.pos_offset = None
        self.quat_engage_raw = None
        self.quat_engage_cmd = None

        self.prev_position_trigger = False
        self.prev_orientation_trigger = False

        self.prev_raw_pos = None
        self.prev_raw_quat = None

        self.position_settle_until = 0.0
        self.orientation_settle_until = 0.0

    def _reset_offsets(self):
        self.pos_offset = None
        self.quat_engage_raw = None
        self.quat_engage_cmd = None
        self.prev_position_trigger = False
        self.prev_orientation_trigger = False

    def tick(self):
        """One iteration of the clutch/IK command logic. Does NOT step physics
        or advance the gripper ramp — call self.controller.step_sim(dt) after
        this, mirroring mq3_mc_mujoco.py's run_teleop_session per-tick order."""
        # Re-checked every tick (cheap once connected) so a headset that appears mid
        # takeover attaches by itself, with no operator action and no restart.
        if not self._connected and not self.ensure_connected():
            return

        input_data = self.mq3.get_latest_controller_data()
        if not input_data or CONTROL_HAND not in input_data:
            self._log_link_health()
            return

        hand_data = input_data[CONTROL_HAND]

        position_trigger_value = float(hand_data.get("index_trigger", 0.0))
        orientation_trigger_value = float(hand_data.get("hand_trigger", 0.0))

        self._rx_last_wall = time.time()
        self._last_index_trigger = position_trigger_value
        self._last_hand_trigger = orientation_trigger_value
        self._log_link_health()

        position_trigger = position_trigger_value > POSITION_TRIGGER_THRESHOLD
        orientation_trigger = orientation_trigger_value > ORIENTATION_TRIGGER_THRESHOLD

        raw_pos = np.asarray(hand_data["pos"], dtype=np.float32)
        raw_quat = normalize_quat(np.asarray(hand_data["rot"], dtype=np.float32))

        if self.prev_raw_pos is not None:
            position_diff = float(np.linalg.norm(raw_pos - self.prev_raw_pos))
            if position_diff > JUMP_RESET_POS_M:
                self._reset_offsets()
                print(f"[MC-SIM][WARN] Controller position jump: {position_diff:.3f} m. Offsets reset.")

        if self.prev_raw_quat is not None:
            orientation_diff = quat_angle_deg_xyzw(self.prev_raw_quat, raw_quat)
            if orientation_diff > JUMP_RESET_ORI_DEG:
                self._reset_offsets()
                print(f"[MC-SIM][WARN] Controller orientation jump: {orientation_diff:.2f} deg. Offsets reset.")

        self.prev_raw_pos = raw_pos.copy()
        self.prev_raw_quat = raw_quat.copy()

        if CONTROL_MODE == "mirror":
            ctrl_pos = POS_MIRROR_XY @ raw_pos
        else:
            ctrl_pos = raw_pos

        position_trigger_edge = position_trigger and not self.prev_position_trigger
        orientation_trigger_edge = orientation_trigger and not self.prev_orientation_trigger
        now_wall = time.perf_counter()

        if READ_SIM_POSE_ON_TRIGGER_EDGE and (position_trigger_edge or orientation_trigger_edge):
            self.last_cmd_pos, self.last_cmd_quat = self.controller.get_ee_pose()

        if position_trigger_edge:
            self.pos_offset = self.last_cmd_pos - ctrl_pos
            self.position_settle_until = now_wall + max(0.0, float(CLUTCH_ENGAGE_SETTLE_S))

        if orientation_trigger_edge:
            self.quat_engage_raw = raw_quat.copy()
            self.quat_engage_cmd = self.last_cmd_quat.copy()
            self.orientation_settle_until = now_wall + max(0.0, float(CLUTCH_ENGAGE_SETTLE_S))

        self.prev_position_trigger = position_trigger
        self.prev_orientation_trigger = orientation_trigger

        position_settling = position_trigger and now_wall < self.position_settle_until
        orientation_settling = orientation_trigger and now_wall < self.orientation_settle_until

        if position_settling:
            self.pos_offset = self.last_cmd_pos - ctrl_pos
            pos_des = self.last_cmd_pos
        elif self.pos_offset is not None and position_trigger:
            pos_des = ctrl_pos + self.pos_offset
            pos_des = clamp_step_pos(
                desired=pos_des,
                current=self.last_cmd_pos,
                max_step=POSITION_STEP_CLAMP_M,
            )
        else:
            pos_des = self.last_cmd_pos

        if orientation_settling:
            self.quat_engage_raw = raw_quat.copy()
            self.quat_engage_cmd = self.last_cmd_quat.copy()
            quat_des = self.last_cmd_quat
        elif self.quat_engage_raw is not None and orientation_trigger:
            delta = quat_mul_xyzw(raw_quat, quat_inv_xyzw(self.quat_engage_raw))
            if CONTROL_MODE == "mirror":
                delta = quat_mul_xyzw(
                    QUAT_MIRROR_Z180,
                    quat_mul_xyzw(delta, quat_inv_xyzw(QUAT_MIRROR_Z180)),
                )
            quat_des = normalize_quat(quat_mul_xyzw(delta, self.quat_engage_cmd))
            quat_des = clamp_step_quat(
                current=self.last_cmd_quat,
                desired=quat_des,
                max_angle_deg=ORIENTATION_STEP_CLAMP_DEG,
            )
        else:
            quat_des = self.last_cmd_quat

        if MC_SMOOTH_POS_ALPHA < 1.0:
            pos_des = (
                MC_SMOOTH_POS_ALPHA * pos_des
                + (1.0 - MC_SMOOTH_POS_ALPHA) * self.last_cmd_pos
            ).astype(np.float32)
        if MC_SMOOTH_ROT_ALPHA < 1.0:
            quat_des = quat_slerp_xyzw(self.last_cmd_quat, quat_des, MC_SMOOTH_ROT_ALPHA)

        gripper_pressed = is_button_pressed(
            input_data=input_data,
            hand_name=CONTROL_HAND,
            button_name=GRIPPER_CONTROL_BUTTON,
        )
        # Edge latch. set_gripper_pressed(False) re-opens the hand, so forwarding the
        # button unconditionally would open the gripper on the FIRST packet after the
        # takeover — dropping whatever the policy was holding — purely because the
        # operator happened not to be squeezing yet.
        #
        # The baseline is the BUTTON's first observed value, not the sim's grip state:
        # those two legitimately disagree at takeover (that disagreement IS the bug), so
        # comparing against the grip would call the very first packet a change and open
        # the hand anyway. The button becomes authoritative — in both directions — the
        # moment the operator actually moves it.
        if not self._gripper_edge_latch or self._gripper_latched:
            self.controller.set_gripper_pressed(gripper_pressed)
        else:
            if self._gripper_button_baseline is None:
                self._gripper_button_baseline = gripper_pressed
            if gripper_pressed != self._gripper_button_baseline:
                self._gripper_latched = True
                self.controller.set_gripper_pressed(gripper_pressed)

        cartesian_active = position_trigger or orientation_trigger
        skip_cartesian_on_edge = position_trigger_edge or orientation_trigger_edge
        skip_cartesian_for_settle = position_settling or orientation_settling
        target_pos_delta = float(np.linalg.norm(np.asarray(pos_des) - np.asarray(self.last_cmd_pos)))
        target_ori_delta = float(quat_angle_deg_xyzw(self.last_cmd_quat, quat_des))
        target_changed = (
            target_pos_delta >= float(CARTESIAN_POS_DEADBAND_M)
            or target_ori_delta >= float(CARTESIAN_ORI_DEADBAND_DEG)
        )

        if (
            cartesian_active
            and target_changed
            and not skip_cartesian_on_edge
            and not skip_cartesian_for_settle
        ):
            self.controller.command_ee_pose(pos_des, quat_des)
            self._cmds_issued += 1
            self.last_cmd_pos = pos_des
            self.last_cmd_quat = quat_des

    def _log_link_health(self):
        """One line per _diag_period_s while a takeover is live. Distinguishes the three
        silent states: no packets arriving, packets arriving with the triggers released,
        and packets plus a squeezed trigger but no IK command (clutch/deadband).

        Called from tick() on both the data and the no-data path so a stream that stops
        mid-takeover is reported, not just one that never starts.
        """
        now = time.time()

        # Attached but nothing on the wire. Said once, with the address, because the
        # usual cause is having attached to a DIFFERENT broadcaster than the headset
        # being worn (two Quests, or one that changed IP after discovery).
        if (
            not self._no_data_warned
            and self._rx_last_wall is None
            and self._attached_wall is not None
            and now - self._attached_wall >= 2.0
        ):
            self._no_data_warned = True
            print(f"[MC-SIM][WARN] No controller packets from {self.mq3.zmq_address} "
                  f"{now - self._attached_wall:.1f}s after attaching. The arm cannot move. "
                  "Check that this is the headset you are wearing and that it is still "
                  "in an intervention (the Quest publishes only while one is active).")

        if self._diag_period_s <= 0.0:
            return
        if now - self._diag_last_log < self._diag_period_s:
            return

        # frames_received is the receiver's own count of REAL packets. Counting tick()
        # calls instead would understate it: get_latest_controller_data drains the whole
        # queue per tick and returns only the newest frame.
        rx_total = int(getattr(self.mq3, "frames_received", 0))
        elapsed = now - self._diag_last_log if self._diag_last_log else 0.0
        rx_hz = (rx_total - self._diag_last_rx) / elapsed if elapsed > 0.0 else 0.0
        self._diag_last_log = now
        self._diag_last_rx = rx_total

        age = "never" if self._rx_last_wall is None else f"{now - self._rx_last_wall:.1f}s"
        print(f"[MC-SIM][Link] rx={rx_total} ({rx_hz:.0f} Hz, last={age}) "
              f"index_trigger={self._last_index_trigger:.2f} "
              f"grip_trigger={self._last_hand_trigger:.2f} "
              f"ik_cmds={self._cmds_issued} src={self.mq3.zmq_address}")

    def close(self):
        try:
            self.mq3.close()
        except Exception:
            pass
