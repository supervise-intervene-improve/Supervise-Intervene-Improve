"""DEPRECATED: real-robot Quest motion-controller teleop.

Superseded by mq3_mc_mujoco.py. We no longer need to mirror the physical Franka arm
during motion-controller intervention — driving the MuJoCo sim robot directly is
sufficient, so mq3_mc_mujoco.py (same clutch/smoothing/packet-handling logic, local
IK instead of a polymetis CARTESIAN_IMPEDANCE connection) is now the script to launch.
This file is kept only for reference / in case real-robot mirroring is needed again.
"""

import argparse
import json
import os
import socket
import threading
import time

import numpy as np
import torch
import zmq

try:
    import grpc  # polymetis transport — used to detect "controller replaced under us" errors
except ImportError:  # pragma: no cover
    grpc = None

from real_robot_env.robot.hardware_franka import FrankaArm, ControlType
from real_robot_env.robot.hardware_frankahand import FrankaHand


# ============================================================
# Low-latency teleoperation settings
# ============================================================

# Main loop target. The Quest publisher samples at its frame rate (~72-90 Hz) and sends each
# sample once, so anything much above ~120 Hz is a pure busy-loop with no fresh data to act on.
FPS = 120
DT = 1.0 / FPS

CONTROL_HAND = "right"
CONTROL_MODE = "mirror"

GRIPPER_CONTROL_BUTTON = "A"  # can be "A", "B", "X", or "Y"

POSITION_TRIGGER_THRESHOLD = 0.3
ORIENTATION_TRIGGER_THRESHOLD = 0.3

# Set True if the robot should hold still when no new Quest packet arrived.
# False re-sends the latest command while waiting for fresh data.
COMMAND_ONLY_ON_NEW_PACKET = True

# Re-read the robot's live EE pose at every clutch engage so last_cmd can never be stale
# (e.g. after app.py re-aligned the arm for a new intervention) — kills engage-time jumps.
READ_ROBOT_POSE_ON_TRIGGER_EDGE = True

# Jump detection (tracking-glitch reset) vs per-command clamps are now SEPARATE:
# jump thresholds compare consecutive RAW packets and reset the clutch offsets — they must be
# large enough that fast-but-real hand motion never trips them (25 deg/packet did, dropping
# control mid-motion). Step clamps limit each COMMAND increment sent to the robot.
JUMP_RESET_POS_M = 2.0               # meters between consecutive packets → reset offsets
JUMP_RESET_ORI_DEG = 80.0            # degrees between consecutive packets → reset offsets
POSITION_STEP_CLAMP_M = 0.10         # max meters per command sent to the robot
ORIENTATION_STEP_CLAMP_DEG = 25.0    # max degrees per command sent to the robot

# One-pole low-pass on the outgoing command (1.0 = off). Smooths frame-quantized Quest
# samples so the impedance controller doesn't chase discrete jumps at speed.
MC_SMOOTH_POS_ALPHA = float(os.environ.get("MC_SMOOTH_POS_ALPHA", "0.35"))
MC_SMOOTH_ROT_ALPHA = float(os.environ.get("MC_SMOOTH_ROT_ALPHA", "0.35"))

# If no MC packet arrives for this long mid-session, the intervention ended without an X we
# could see (reject/CANCEL disables the Quest publisher) — tear down and re-enter discovery.
MC_PACKET_TIMEOUT_S = float(os.environ.get("MC_PACKET_TIMEOUT_S", "2.0"))

# Optional low-rate loop printout.
PRINT_LOOP_HZ = True
DEBUG_PERIOD_S = 2.0


# ============================================================
# Quest ZMQ discovery settings
# ============================================================

DISCOVERY_PORT = 6656
DISCOVERY_TIMEOUT_S = 20.0
EXPECTED_DISCOVERY_TYPE = "MQ3_ZMQ_DISCOVERY"


# ============================================================
# Robot settings
# ============================================================

FRANKA_IP = "192.0.2.153"
FRANKA_ARM_NAME = "p4"
FRANKA_HAND_NAME = "p4_hand"
FRANKA_ARM_PORT = 50053
FRANKA_HAND_PORT = 50054


POS_MIRROR_XY = np.array(
    [
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

# POS_MIRROR_XY is a PROPER rotation: Rz(180 deg). Its quaternion (xyzw) — used to conjugate
# world-frame orientation deltas so rotation direction matches the mirrored position feel.
QUAT_MIRROR_Z180 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)


# ============================================================
# Meta Quest ZMQ receiver
# ============================================================

class MetaQuest3ZmqReceiver:
    """
    Direct low-latency receiver for the Quest app.

    The important bit is get_latest_controller_data(): it drains the SUB
    socket and returns only the newest packet, preventing old controller poses
    from building a queue and showing up as robot lag.
    """

    def __init__(
        self,
        discovery_port=6656,
        discovery_timeout_s=20.0,
        expected_type="MQ3_ZMQ_DISCOVERY",
        default_topic="MotionController",
    ):
        self.discovery_port = discovery_port
        self.discovery_timeout_s = discovery_timeout_s
        self.expected_type = expected_type
        self.default_topic = default_topic

        self.context = None
        self.sub = None

        self.quest_ip = None
        self.zmq_port = None
        self.topic = None
        self.zmq_address = None

        self.frames_received = 0
        self.frames_dropped = 0

    def discover_meta_quest(self):
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        udp.bind(("", self.discovery_port))
        udp.settimeout(0.25)

        print(f"Listening for Quest UDP broadcast on port {self.discovery_port}...")

        start = time.perf_counter()

        try:
            while time.perf_counter() - start < self.discovery_timeout_s:
                try:
                    packet, addr = udp.recvfrom(4096)
                except socket.timeout:
                    continue

                quest_ip = addr[0]

                try:
                    msg = json.loads(packet.decode("utf-8"))
                except Exception:
                    continue

                if msg.get("type") != self.expected_type:
                    continue

                zmq_port = int(msg["zmq_port"])
                topic = msg.get("topic", self.default_topic)
                device_name = msg.get("device_name", "UnknownQuest")

                print("Found Quest:")
                print(f"  device: {device_name}")
                print(f"  ip:     {quest_ip}")
                print(f"  port:   {zmq_port}")
                print(f"  topic:  {topic}")

                self.quest_ip = quest_ip
                self.zmq_port = zmq_port
                self.topic = topic
                self.zmq_address = f"tcp://{quest_ip}:{zmq_port}"

                return self.quest_ip, self.zmq_port, self.topic
        finally:
            udp.close()

        raise RuntimeError("Could not find Quest broadcast.")

    def connect(self):
        if self.quest_ip is None:
            self.discover_meta_quest()

        self.context = zmq.Context.instance()
        self.sub = self.context.socket(zmq.SUB)

        self.sub.setsockopt(zmq.LINGER, 0)
        self.sub.setsockopt(zmq.RCVHWM, 1)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, self.topic)

        # Do not use CONFLATE here: with multipart PUB/SUB it can keep only
        # one message part. Draining is safer and still gets latest-frame
        # behavior.
        self._try_setsockopt("TCP_KEEPALIVE", 1)
        self._try_setsockopt("TCP_KEEPALIVE_IDLE", 1)
        self._try_setsockopt("TCP_KEEPALIVE_INTVL", 1)

        self.sub.connect(self.zmq_address)

        print(f"Connected to {self.zmq_address}, topic={self.topic}")
        print("Waiting for controller data...")

    def _try_setsockopt(self, name, value):
        option = getattr(zmq, name, None)
        if option is not None:
            try:
                self.sub.setsockopt(option, value)
            except zmq.ZMQError:
                pass

    def get_latest_controller_data(self):
        if self.sub is None:
            self.connect()

        latest_msg = None
        drained = 0

        while True:
            try:
                parts = self.sub.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break

            if len(parts) < 2:
                continue

            latest_msg = parts[-1]
            drained += 1

        if latest_msg is None:
            return None

        self.frames_received += drained
        if drained > 1:
            self.frames_dropped += drained - 1

        try:
            return json.loads(latest_msg.decode("utf-8"))
        except Exception as e:
            print(f"Failed to decode Quest controller data: {e}")
            return None

    def close(self):
        if self.sub is not None:
            self.sub.close(linger=0)
            self.sub = None


# ============================================================
# Fast NumPy quaternion helpers, xyzw format
# ============================================================

def normalize_quat(q):
    q = np.asarray(q, dtype=np.float32)
    return q / (np.linalg.norm(q) + 1e-9)


def quat_conj_xyzw(q):
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)


def quat_inv_xyzw(q):
    return quat_conj_xyzw(normalize_quat(q))


def quat_mul_xyzw(q1, q2):
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float32,
    )


def quat_angle_deg_xyzw(q1, q2):
    q1 = normalize_quat(q1)
    q2 = normalize_quat(q2)

    dot = abs(float(np.dot(q1, q2)))
    dot = np.clip(dot, -1.0, 1.0)

    return 2.0 * np.arccos(dot) * 180.0 / np.pi


def quat_slerp_xyzw(q1, q2, fraction):
    q1 = normalize_quat(q1)
    q2 = normalize_quat(q2)

    dot = float(np.dot(q1, q2))

    if dot < 0.0:
        q2 = -q2
        dot = -dot

    dot = np.clip(dot, -1.0, 1.0)

    if dot > 0.9995:
        q = q1 + fraction * (q2 - q1)
        return normalize_quat(q)

    theta_0 = np.arccos(dot)
    sin_0 = np.sin(theta_0)
    theta_t = theta_0 * fraction

    s1 = np.sin(theta_0 - theta_t) / sin_0
    s2 = np.sin(theta_t) / sin_0

    q = s1 * q1 + s2 * q2
    return normalize_quat(q)


def clamp_step_pos(desired, current, max_step):
    diff = desired - current
    n = np.linalg.norm(diff)

    if n < 1e-9:
        return desired

    if n > max_step:
        diff *= max_step / n

    return current + diff


def clamp_step_quat(current, desired, max_angle_deg):
    angle = quat_angle_deg_xyzw(current, desired)

    if angle <= max_angle_deg:
        return normalize_quat(desired)

    fraction = max_angle_deg / max(angle, 1e-6)
    return quat_slerp_xyzw(current, desired, fraction)


def _release_arm_without_homing(arm):
    """Drop our FrankaArm client handle WITHOUT letting it home the robot.

    hardware_franka.FrankaArm.__del__ -> close() -> reset() -> robot.go_home(3s): when the
    next session rebinds `robot_arm`, the previous session's object is finalized and would
    physically DRIVE THE ARM TO ITS DEFAULT POSE — right after app.py has just aligned it
    for the new intervention (the 2nd-intervention "goes to default" regression).

    app.py owns the controller after finish/cancel (it installs its own hold policy), so we
    only need to detach our gRPC handle: nulling `arm.robot` makes close()/__del__ a no-op.
    """
    if arm is not None:
        arm.robot = None


def get_robot_pose_np(robot_arm):
    ee_pos, ee_quat = robot_arm.robot.get_ee_pose()

    if isinstance(ee_pos, torch.Tensor):
        ee_pos = ee_pos.detach().cpu().numpy()

    if isinstance(ee_quat, torch.Tensor):
        ee_quat = ee_quat.detach().cpu().numpy()

    ee_pos = np.asarray(ee_pos, dtype=np.float32)
    ee_quat = normalize_quat(np.asarray(ee_quat, dtype=np.float32))

    return ee_pos, ee_quat


def is_button_pressed(input_data, hand_name, button_name):
    hand_data = input_data.get(hand_name, {})

    if button_name in hand_data:
        return bool(hand_data[button_name])

    return bool(input_data.get(button_name, False))


# ============================================================
# Non-blocking gripper worker
# ============================================================

class GripperWorker:
    def __init__(self, gripper):
        self.gripper = gripper
        self.lock = threading.Lock()
        self.desired_width = None
        self.last_sent_width = None
        self.running = True

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def set_width(self, width):
        with self.lock:
            self.desired_width = width

    def _loop(self):
        while self.running:
            with self.lock:
                width = self.desired_width

            if width is not None and width != self.last_sent_width:
                try:
                    self.gripper.apply_commands(width=width)
                    print(f"[GripperWorker] Gripper command sent: width={width}")
                    self.last_sent_width = width
                except Exception as e:
                    print(f"[GripperWorker] Gripper command failed: {e}")

            time.sleep(0.002)

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)


# ============================================================
# Main teleoperation loop
# ============================================================

def main(cfg=None):
    if cfg is None:
        cfg = argparse.Namespace(
            robot_key=FRANKA_ARM_NAME,
            cmd_host="127.0.0.1",
            cmd_port=0,
        )

    # Resolve robot_key → ip / ports from record.ROBOTS (single source of truth)
    try:
        import record
        robot_info = record.ROBOTS[cfg.robot_key]
        robot_ip   = robot_info.robot_arm.ip_address
        robot_name = robot_info.name
        arm_port   = robot_info.robot_arm.port
        hand_port  = robot_info.robot_gripper.port
        print(f"[MC] Robot key '{cfg.robot_key}' → {robot_name} @ {robot_ip} "
              f"(arm:{arm_port} hand:{hand_port})")
    except Exception as e:
        print(f"[MC] Could not resolve robot_key='{cfg.robot_key}' from record.ROBOTS ({e}); "
              f"falling back to module defaults")
        robot_ip   = FRANKA_IP
        robot_name = FRANKA_ARM_NAME
        arm_port   = FRANKA_ARM_PORT
        hand_port  = FRANKA_HAND_PORT

    # -----------------------------
    # Optional cmd forwarding to policy app.py
    # -----------------------------

    cmd_sock = None
    if cfg.cmd_port > 0:
        _ctx = zmq.Context.instance()
        cmd_sock = _ctx.socket(zmq.PUSH)
        cmd_sock.setsockopt(zmq.LINGER, 0)
        cmd_sock.connect(f"tcp://{cfg.cmd_host}:{cfg.cmd_port}")
        print(f"[MC] cmd forwarder connected tcp://{cfg.cmd_host}:{cfg.cmd_port}")

    def send_cmd(s):
        if cmd_sock is not None:
            try:
                cmd_sock.send_string(s, flags=zmq.NOBLOCK)
            except zmq.Again:
                pass

    # -----------------------------------------------------------------------
    # Outer restart loop — each iteration = one MC intervention session.
    # After the user presses X to finish, we close the robot/Quest connections
    # and restart from discovery so the next intervention is clean.
    # Ctrl+C breaks out of the outer loop.
    # -----------------------------------------------------------------------

    while True:
        # --------------------------
        # Quest ZMQ discovery
        # --------------------------
        # Robot setup is deferred until after Quest MC discovery.
        # X (intervention) → app.py aligns robot + enables HUMAN_CONTROL.
        # User then presses A → MC publisher starts → discovery succeeds.
        # By the time we connect via CARTESIAN_IMPEDANCE the robot is already
        # at the sim pose — no reset() needed.
        mq3 = None
        while mq3 is None:
            _attempt = MetaQuest3ZmqReceiver(
                discovery_port=DISCOVERY_PORT,
                discovery_timeout_s=DISCOVERY_TIMEOUT_S,
                expected_type=EXPECTED_DISCOVERY_TYPE,
            )
            try:
                _attempt.connect()
                mq3 = _attempt
            except RuntimeError:
                print("[MC] Quest MC publisher not found yet. "
                      "Enable Motion Controller mode (press A) in the Quest app, then mq3_mc.py will connect.")
                time.sleep(3)

        # --------------------------
        # Robot setup
        # --------------------------
        robot_arm = FrankaArm(
            name=robot_name,
            ip_address=robot_ip,
            port=arm_port,
            control_type=ControlType.CARTESIAN_IMPEDANCE_CONTROL,
        )

        # Establish the polymetis RobotInterface and start the CARTESIAN_IMPEDANCE policy at the
        # robot's CURRENT (app.py-aligned) pose. Without this, robot_arm.robot is None and every
        # get_ee_pose()/update_current_policy() call fails — the arm never moves.
        assert robot_arm.connect(), f"Connection to {robot_arm.name} failed"

        gripper = FrankaHand(
            name=robot_name + "_hand",
            ip_address=robot_ip,
            port=hand_port,
        )

        assert gripper.connect(), f"Connection to {gripper.name} failed"

        gripper_worker = GripperWorker(gripper)

        # --------------------------
        # Wait for first controller packet
        # --------------------------
        input_data = mq3.get_latest_controller_data()

        while not input_data or CONTROL_HAND not in input_data:
            input_data = mq3.get_latest_controller_data()
            time.sleep(0.001)

        last_input_data = input_data

        print("Controller connected.")
        print(f"Initial input data: {input_data}")

        # --------------------------
        # Initial robot pose
        # --------------------------
        last_cmd_pos, last_cmd_quat = get_robot_pose_np(robot_arm)

        pos_offset = None
        # Orientation clutch anchors: raw controller quat + command quat at engage. The delta
        # from quat_engage_raw is computed in the WORLD frame, mirror-conjugated, and applied
        # left of quat_engage_cmd — directionally consistent with the mirrored position.
        quat_engage_raw = None
        quat_engage_cmd = None

        prev_position_trigger = False
        prev_orientation_trigger = False

        prev_raw_pos = None
        prev_raw_quat = None

        loop_counter = 0
        command_counter = 0
        last_debug_time = time.perf_counter()
        last_packet_wall = time.perf_counter()
        # Seed X as pressed: if the X that started the intervention is somehow still held on
        # the first packet, it must not read as a rising edge and end the session instantly.
        _prev_btns: dict = {"X": True}
        _done_intervention = False

        print("Starting low-latency teleoperation loop...")

        try:
            while not _done_intervention:
                loop_start = time.perf_counter()

                input_data = mq3.get_latest_controller_data()

                if not input_data:
                    # Publisher went quiet: the intervention was finished/cancelled on a path
                    # we cannot see (e.g. grip=CANCEL) and Unity disabled the MC publisher.
                    if time.perf_counter() - last_packet_wall > MC_PACKET_TIMEOUT_S:
                        print(f"[MC] No MC packets for {MC_PACKET_TIMEOUT_S:.1f}s — "
                              "intervention ended externally. Ending MC session.")
                        _done_intervention = True
                        continue
                    if COMMAND_ONLY_ON_NEW_PACKET:
                        time.sleep(0.001)
                        continue
                    input_data = last_input_data

                if input_data is None or CONTROL_HAND not in input_data:
                    time.sleep(0.001)
                    continue

                last_input_data = input_data
                last_packet_wall = time.perf_counter()

                hand_data = input_data[CONTROL_HAND]

                # --------------------------
                # Read controller state
                # --------------------------
                position_trigger_value = float(hand_data.get("index_trigger", 0.0))
                orientation_trigger_value = float(hand_data.get("hand_trigger", 0.0))

                position_trigger = position_trigger_value > POSITION_TRIGGER_THRESHOLD
                orientation_trigger = orientation_trigger_value > ORIENTATION_TRIGGER_THRESHOLD

                raw_pos = np.asarray(hand_data["pos"], dtype=np.float32)
                raw_quat = normalize_quat(np.asarray(hand_data["rot"], dtype=np.float32))

                # --------------------------
                # Jump detection
                # --------------------------
                if prev_raw_pos is not None:
                    position_diff = float(np.linalg.norm(raw_pos - prev_raw_pos))

                    if position_diff > JUMP_RESET_POS_M:
                        pos_offset = None
                        quat_engage_raw = None
                        quat_engage_cmd = None
                        prev_position_trigger = False
                        prev_orientation_trigger = False

                        print(
                            f"[WARN] Controller position jump: {position_diff:.3f} m. "
                            "Offsets reset."
                        )

                if prev_raw_quat is not None:
                    orientation_diff = quat_angle_deg_xyzw(prev_raw_quat, raw_quat)

                    if orientation_diff > JUMP_RESET_ORI_DEG:
                        pos_offset = None
                        quat_engage_raw = None
                        quat_engage_cmd = None
                        prev_position_trigger = False
                        prev_orientation_trigger = False

                        print(
                            f"[WARN] Controller orientation jump: {orientation_diff:.2f} deg. "
                            "Offsets reset."
                        )

                prev_raw_pos = raw_pos.copy()
                prev_raw_quat = raw_quat.copy()

                # --------------------------
                # Controller transform
                # --------------------------
                if CONTROL_MODE == "mirror":
                    ctrl_pos = POS_MIRROR_XY @ raw_pos
                else:
                    ctrl_pos = raw_pos

                # --------------------------
                # Trigger edge handling
                # --------------------------
                position_trigger_edge = position_trigger and not prev_position_trigger
                orientation_trigger_edge = orientation_trigger and not prev_orientation_trigger

                if READ_ROBOT_POSE_ON_TRIGGER_EDGE and (
                    position_trigger_edge or orientation_trigger_edge
                ):
                    last_cmd_pos, last_cmd_quat = get_robot_pose_np(robot_arm)

                if position_trigger_edge:
                    pos_offset = last_cmd_pos - ctrl_pos
                    print("Position control engaged.")

                if orientation_trigger_edge:
                    quat_engage_raw = raw_quat.copy()
                    quat_engage_cmd = last_cmd_quat.copy()
                    print("Orientation control engaged.")

                prev_position_trigger = position_trigger
                prev_orientation_trigger = orientation_trigger

                # --------------------------
                # Desired position
                # --------------------------
                if pos_offset is not None and position_trigger:
                    pos_des = ctrl_pos + pos_offset
                    pos_des = clamp_step_pos(
                        desired=pos_des,
                        current=last_cmd_pos,
                        max_step=POSITION_STEP_CLAMP_M,
                    )
                else:
                    pos_des = last_cmd_pos

                # --------------------------
                # Desired orientation — world-frame delta since engage, mirror-conjugated so
                # rotation direction matches the mirrored position mapping.
                # --------------------------
                if quat_engage_raw is not None and orientation_trigger:
                    delta = quat_mul_xyzw(raw_quat, quat_inv_xyzw(quat_engage_raw))
                    if CONTROL_MODE == "mirror":
                        delta = quat_mul_xyzw(
                            QUAT_MIRROR_Z180,
                            quat_mul_xyzw(delta, quat_inv_xyzw(QUAT_MIRROR_Z180)),
                        )
                    quat_des = normalize_quat(quat_mul_xyzw(delta, quat_engage_cmd))

                    quat_des = clamp_step_quat(
                        current=last_cmd_quat,
                        desired=quat_des,
                        max_angle_deg=ORIENTATION_STEP_CLAMP_DEG,
                    )
                else:
                    quat_des = last_cmd_quat

                # --------------------------
                # Command smoothing (one-pole low-pass toward the target)
                # --------------------------
                if MC_SMOOTH_POS_ALPHA < 1.0:
                    pos_des = (
                        MC_SMOOTH_POS_ALPHA * pos_des
                        + (1.0 - MC_SMOOTH_POS_ALPHA) * last_cmd_pos
                    ).astype(np.float32)
                if MC_SMOOTH_ROT_ALPHA < 1.0:
                    quat_des = quat_slerp_xyzw(last_cmd_quat, quat_des, MC_SMOOTH_ROT_ALPHA)

                # --------------------------
                # Send robot command
                # --------------------------
                try:
                    robot_arm.robot.update_current_policy(
                        {
                            "ee_pos_desired": torch.as_tensor(pos_des, dtype=torch.float32),
                            "ee_quat_desired": torch.as_tensor(quat_des, dtype=torch.float32),
                        }
                    )
                except Exception as e:
                    # app.py's finish/cancel replaces the controller (HYBRID_JOINT hold) — our
                    # cartesian param update then fails (grpc.RpcError / KeyError). That means
                    # the intervention is over; end the session instead of crash-looping.
                    if grpc is not None and not isinstance(e, grpc.RpcError):
                        raise
                    print(f"[MC] Robot policy update rejected — intervention ended externally "
                          f"({type(e).__name__}). Ending MC session.")
                    _done_intervention = True
                    continue

                last_cmd_pos = pos_des
                last_cmd_quat = quat_des
                command_counter += 1

                # --------------------------
                # Non-blocking gripper control
                # --------------------------
                gripper_pressed = is_button_pressed(
                    input_data=input_data,
                    hand_name=CONTROL_HAND,
                    button_name=GRIPPER_CONTROL_BUTTON,
                )

                if gripper_pressed:
                    gripper_worker.set_width(-1.0)
                else:
                    gripper_worker.set_width(1.0)

                # --------------------------
                # Session end on X (accept). We do NOT forward B/X/Y to app.py anymore —
                # Unity's InterventionButtonForwarder is the single command path (forwarding
                # from here too caused a double-X race: finish + immediate re-trigger).
                # X in the MC stream is only used LOCALLY to tear this session down fast;
                # reject/CANCEL is covered by the packet timeout above.
                # --------------------------
                now_x = bool(input_data.get("X", False))
                if now_x and not _prev_btns.get("X", False):
                    print("[MC] X (accept) seen — ending MC session; "
                          "Unity forwarder delivers the finish command.")
                    _done_intervention = True
                _prev_btns = {"X": now_x}

                # --------------------------
                # Debug print
                # --------------------------
                loop_counter += 1
                now = time.perf_counter()

                if PRINT_LOOP_HZ and now - last_debug_time > DEBUG_PERIOD_S:
                    dt = now - last_debug_time
                    measured_hz = loop_counter / dt
                    command_hz = command_counter / dt
                    dropped = mq3.frames_dropped
                    received = mq3.frames_received
                    print(
                        f"Teleop loop: {measured_hz:.1f} Hz, "
                        f"commands: {command_hz:.1f} Hz, "
                        f"Quest frames dropped: {dropped}/{received}"
                    )
                    loop_counter = 0
                    command_counter = 0
                    mq3.frames_received = 0
                    mq3.frames_dropped = 0
                    last_debug_time = now

                # --------------------------
                # Rate limit
                # --------------------------
                elapsed = time.perf_counter() - loop_start
                sleep_time = DT - elapsed

                if sleep_time > 0.0:
                    time.sleep(sleep_time)

        except KeyboardInterrupt:
            print("Stopping teleoperation...")
            gripper_worker.stop()
            mq3.close()
            _release_arm_without_homing(robot_arm)
            break

        gripper_worker.stop()
        mq3.close()
        _release_arm_without_homing(robot_arm)
        robot_arm = None
        print("[MC] Ready for next intervention — waiting for MC mode to be activated...")


def parse_args():
    ap = argparse.ArgumentParser(description="Quest motion controller → Franka robot teleoperation")
    ap.add_argument("--robot_key",
                    default=os.environ.get("INTERVENE_ROBOT_KEY", os.environ.get("ROBOT_KEY", FRANKA_ARM_NAME)),
                    help="Robot key (p1/p2/p3/p4) — looks up IP and ports from record.ROBOTS")
    ap.add_argument("--cmd_host",  default=os.environ.get("MC_CMD_HOST", "127.0.0.1"),
                    help="Host to forward B/X/Y button commands to (policy app.py cmd socket)")
    ap.add_argument("--cmd_port",  type=int, default=int(os.environ.get("MC_CMD_PORT", 0)),
                    help="ZMQ PUSH port for forwarding B/X/Y to policy app.py (0 = disabled)")
    return ap.parse_args()


if __name__ == "__main__":
    main(parse_args())
