import argparse
import json
import os
import shutil
import socket
import tempfile
import time
from pathlib import Path

import mujoco
import numpy as np
import zmq


# ============================================================
# Low-latency teleoperation settings
# ============================================================

# Kept aligned with mq3_mc.py so Quest clutching, smoothing, and packet handling
# feel the same even though the command target is now a MuJoCo model.
FPS = 120
DT = 1.0 / FPS

CONTROL_HAND = "right"
CONTROL_MODE = "mirror"

GRIPPER_CONTROL_BUTTON = "A"

POSITION_TRIGGER_THRESHOLD = 0.3
ORIENTATION_TRIGGER_THRESHOLD = 0.3

COMMAND_ONLY_ON_NEW_PACKET = True
READ_SIM_POSE_ON_TRIGGER_EDGE = True

JUMP_RESET_POS_M = 2.0
JUMP_RESET_ORI_DEG = 80.0
POSITION_STEP_CLAMP_M = 0.10
ORIENTATION_STEP_CLAMP_DEG = 25.0

MC_SMOOTH_POS_ALPHA = float(os.environ.get("MC_SMOOTH_POS_ALPHA", "0.35"))
MC_SMOOTH_ROT_ALPHA = float(os.environ.get("MC_SMOOTH_ROT_ALPHA", "0.35"))
MC_PACKET_TIMEOUT_S = float(os.environ.get("MC_PACKET_TIMEOUT_S", "2.0"))

# MuJoCo actuator ctrl changes are instantaneous unless we rate-limit them here.
GRIPPER_CLOSE_TIME_S = float(os.environ.get("MC_MUJOCO_GRIPPER_CLOSE_TIME_S", "0.85"))
GRIPPER_OPEN_TIME_S = float(os.environ.get("MC_MUJOCO_GRIPPER_OPEN_TIME_S", "0.85"))

# Ignore the first few controller frames after clutch engagement. This absorbs
# the small hand/controller twitch caused by squeezing the trigger/button.
CLUTCH_ENGAGE_SETTLE_S = float(os.environ.get("MC_MUJOCO_CLUTCH_SETTLE_S", "0.12"))

# Do not run IK for an effectively unchanged Cartesian target. The first packet
# after clutch settle can otherwise ask IK to reproduce the same pose and rewrite
# arm ctrl, which looks like a delayed drop.
CARTESIAN_POS_DEADBAND_M = float(os.environ.get("MC_MUJOCO_CARTESIAN_POS_DEADBAND_M", "0.003"))
CARTESIAN_ORI_DEADBAND_DEG = float(os.environ.get("MC_MUJOCO_CARTESIAN_ORI_DEADBAND_DEG", "1.0"))

ARM_GRAVITY_COMPENSATION = os.environ.get("MC_MUJOCO_ARM_GRAVITY_COMP", "1").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

PRINT_LOOP_HZ = True
DEBUG_PERIOD_S = 2.0


# ============================================================
# Quest ZMQ discovery settings
# ============================================================

DISCOVERY_PORT = 6656
DISCOVERY_TIMEOUT_S = 20.0
EXPECTED_DISCOVERY_TYPE = "MQ3_ZMQ_DISCOVERY"


# ============================================================
# MuJoCo settings
# ============================================================

REPO_ROOT = Path(__file__).resolve().parent

SCENE_ALIASES = {
    "boxes_cups": "mujoco_scenes/working_scenes/with_soft_gripper/sii_scene_table_boxes_cups.xml",
    "cups": "mujoco_scenes/working_scenes/with_soft_gripper/sii_scene_table_boxes_cups.xml",
    "t_shape": "mujoco_scenes/working_scenes/with_soft_gripper/sii_scene_table_T_shape.xml",
    "tshape": "mujoco_scenes/working_scenes/with_soft_gripper/sii_scene_table_T_shape.xml",
    "wire": "mujoco_scenes/working_scenes/with_soft_gripper/sii_scene_table_wire_base_and_spoon.xml",
    "wire_base_and_spoon": "mujoco_scenes/working_scenes/with_soft_gripper/sii_scene_table_wire_base_and_spoon.xml",
    "wire_spoon": "mujoco_scenes/working_scenes/with_soft_gripper/sii_scene_table_wire_base_and_spoon.xml",
}

PANDA_JOINT_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
]

# A stable table-facing Panda posture. The XML default is all zeros, which puts
# joint4 outside its declared range. This replaces the real script's "read the
# live robot pose at startup" behavior.
DEFAULT_ARM_Q = np.array(
    [-0.10, 0.35, 0.02, -1.90, 0.02, 2.33, -0.87],
    dtype=np.float64,
)

POS_MIRROR_XY = np.array(
    [
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

QUAT_MIRROR_Z180 = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

OLD_INTERVENE_ROOTS = (
    "/path/to/Intervention_IL_AR/intervene_base",
    "/home/user/projects/intervene_base",
)


# ============================================================
# Meta Quest ZMQ receiver
# ============================================================

class MetaQuest3ZmqReceiver:
    """
    Direct low-latency receiver for the Quest app.

    This is intentionally the same interface/behavior as mq3_mc.py: UDP
    discovery, ZMQ SUB, and latest-frame draining to prevent stale controller
    packets from becoming robot/sim lag.
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


def mat_to_quat_xyzw(mat):
    mat = np.asarray(mat, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(mat))

    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (mat[2, 1] - mat[1, 2]) / s
        y = (mat[0, 2] - mat[2, 0]) / s
        z = (mat[1, 0] - mat[0, 1]) / s
    elif mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
        s = np.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2]) * 2.0
        w = (mat[2, 1] - mat[1, 2]) / s
        x = 0.25 * s
        y = (mat[0, 1] + mat[1, 0]) / s
        z = (mat[0, 2] + mat[2, 0]) / s
    elif mat[1, 1] > mat[2, 2]:
        s = np.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2]) * 2.0
        w = (mat[0, 2] - mat[2, 0]) / s
        x = (mat[0, 1] + mat[1, 0]) / s
        y = 0.25 * s
        z = (mat[1, 2] + mat[2, 1]) / s
    else:
        s = np.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1]) * 2.0
        w = (mat[1, 0] - mat[0, 1]) / s
        x = (mat[0, 2] + mat[2, 0]) / s
        y = (mat[1, 2] + mat[2, 1]) / s
        z = 0.25 * s

    return normalize_quat(np.array([x, y, z, w], dtype=np.float32))


def quat_to_rotvec_xyzw(q):
    q = normalize_quat(q).astype(np.float64)
    if q[3] < 0.0:
        q = -q

    xyz = q[:3]
    sin_half = float(np.linalg.norm(xyz))
    if sin_half < 1e-9:
        return 2.0 * xyz

    angle = 2.0 * np.arctan2(sin_half, float(q[3]))
    if angle > np.pi:
        angle -= 2.0 * np.pi

    return xyz / sin_half * angle


def is_button_pressed(input_data, hand_name, button_name):
    hand_data = input_data.get(hand_name, {})

    if button_name in hand_data:
        return bool(hand_data[button_name])

    return bool(input_data.get(button_name, False))


# ============================================================
# MuJoCo model loading and control
# ============================================================

class RepairedXmlLoader:
    """
    Loads the requested XML without editing the repository XMLs.

    Two target scenes contain absolute paths from another workstation, and some
    included XMLs do too. We create a temporary XML/asset mirror: XML files are
    rewritten to this checkout, non-XML assets are symlinked. This is the sim
    equivalent of resolving a robot key in mq3_mc.py, but it never touches
    hardware config or source files.
    """

    def __init__(self, repo_root):
        self.repo_root = Path(repo_root).resolve()
        self.tmp = None

    def close(self):
        if self.tmp is not None:
            self.tmp.cleanup()
            self.tmp = None

    def load(self, xml_path):
        xml_path = Path(xml_path).resolve()
        try:
            model = mujoco.MjModel.from_xml_path(str(xml_path))
            return model, xml_path
        except Exception as original_error:
            if "Error opening file" not in str(original_error) and OLD_INTERVENE_ROOTS[0] not in str(original_error):
                raise

            self.tmp = tempfile.TemporaryDirectory(prefix="mq3_mc_mujoco_xml_")
            repaired_base = Path(self.tmp.name)
            repaired_root = repaired_base / "mujoco_scenes"
            source_root = self.repo_root / "mujoco_scenes"

            self._mirror_tree(source_root, repaired_root, repaired_base)
            repaired_xml = repaired_root / xml_path.relative_to(source_root)

            print(f"[MuJoCo] Repaired scene paths in a temporary XML mirror: {repaired_xml}")
            model = mujoco.MjModel.from_xml_path(str(repaired_xml))
            return model, repaired_xml

    def _mirror_tree(self, source_root, repaired_root, repaired_base):
        for src in source_root.rglob("*"):
            rel = src.relative_to(source_root)
            dst = repaired_root / rel

            if src.is_dir():
                dst.mkdir(parents=True, exist_ok=True)
                continue

            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.suffix.lower() == ".xml":
                text = src.read_text()
                for old_root in OLD_INTERVENE_ROOTS:
                    text = text.replace(old_root, str(repaired_base))
                dst.write_text(text)
            else:
                try:
                    os.symlink(src, dst)
                except FileExistsError:
                    pass
                except OSError:
                    shutil.copy2(src, dst)


def get_qpos_indices_for_joints(model, joint_names):
    idxs = []
    for name in joint_names:
        joint = model.joint(name)
        idxs.append(int(model.jnt_qposadr[joint.id]))
    return idxs


def get_dof_indices_for_joints(model, joint_names):
    idxs = []
    for name in joint_names:
        joint = model.joint(name)
        idxs.append(int(model.jnt_dofadr[joint.id]))
    return idxs


def actuator_ctrl_range(model, actuator_idx):
    if actuator_idx >= model.nu:
        return None
    if not bool(model.actuator_ctrllimited[actuator_idx]):
        return None
    return np.asarray(model.actuator_ctrlrange[actuator_idx], dtype=np.float64)


class MujocoFrankaController:
    def __init__(
        self,
        model,
        data,
        *,
        site_name="gripper",
        initial_q=None,
        reset_npz=None,
        ik_pos_weight=1.0,
        ik_ori_weight=0.35,
        ik_damping=0.05,
        ik_iters=8,
        ik_max_joint_step=0.08,
        gripper_close_time=GRIPPER_CLOSE_TIME_S,
        gripper_open_time=GRIPPER_OPEN_TIME_S,
        arm_gravity_compensation=ARM_GRAVITY_COMPENSATION,
    ):
        self.model = model
        self.data = data
        self.site_name = site_name
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id < 0:
            raise RuntimeError(f"MuJoCo model has no site named {site_name!r}")

        self.qpos_indices = get_qpos_indices_for_joints(model, PANDA_JOINT_NAMES)
        self.dof_indices = get_dof_indices_for_joints(model, PANDA_JOINT_NAMES)
        self.joint_ids = [int(model.joint(name).id) for name in PANDA_JOINT_NAMES]
        self.joint_ranges = self._joint_ranges()

        self.ik_data = mujoco.MjData(model)
        self.ik_pos_weight = float(ik_pos_weight)
        self.ik_ori_weight = float(ik_ori_weight)
        self.ik_damping = float(ik_damping)
        self.ik_iters = int(ik_iters)
        self.ik_max_joint_step = float(ik_max_joint_step)

        self.gripper_ctrl_index = 7 if model.nu >= 8 else None
        self.gripper_open_ctrl = 0.04
        self.gripper_close_ctrl = 0.0
        self.gripper_target_ctrl = 0.04
        self.gripper_close_time = max(1e-6, float(gripper_close_time))
        self.gripper_open_time = max(1e-6, float(gripper_open_time))
        self.arm_gravity_compensation = bool(arm_gravity_compensation)
        self._configure_gripper_range()
        self.gripper_target_ctrl = self.gripper_open_ctrl

        self.initialize(initial_q=initial_q, reset_npz=reset_npz)

    def _joint_ranges(self):
        ranges = []
        for arm_idx, joint_id in enumerate(self.joint_ids):
            if bool(self.model.jnt_limited[joint_id]):
                lo, hi = self.model.jnt_range[joint_id]
            else:
                ctrlrange = actuator_ctrl_range(self.model, arm_idx)
                if ctrlrange is None:
                    lo, hi = -np.inf, np.inf
                else:
                    lo, hi = ctrlrange
            ranges.append((float(lo), float(hi)))
        return np.asarray(ranges, dtype=np.float64)

    def _configure_gripper_range(self):
        if self.gripper_ctrl_index is None:
            return

        ctrlrange = actuator_ctrl_range(self.model, self.gripper_ctrl_index)
        if ctrlrange is None:
            return

        self.gripper_close_ctrl = float(ctrlrange[0])
        self.gripper_open_ctrl = float(ctrlrange[1])

    def initialize(self, initial_q=None, reset_npz=None):
        mujoco.mj_resetData(self.model, self.data)

        if reset_npz is not None:
            self._reset_from_npz(reset_npz)
        else:
            q0 = DEFAULT_ARM_Q if initial_q is None else np.asarray(initial_q, dtype=np.float64)
            self.set_arm_qpos(q0)
            self.data.ctrl[:7] = self.clip_arm_ctrl(q0)
            self._set_finger_qpos_open()
            self.set_gripper_open(immediate=True)
            mujoco.mj_forward(self.model, self.data)

        print("[MuJoCo] Panda joint mapping:")
        for name, qpos_idx, dof_idx in zip(PANDA_JOINT_NAMES, self.qpos_indices, self.dof_indices):
            print(f"  {name:>8s}: qpos[{qpos_idx}], dof[{dof_idx}]")

        pos, quat = self.get_ee_pose()
        print(f"[MuJoCo] Initial {self.site_name} pos: {pos}")
        print(f"[MuJoCo] Initial {self.site_name} quat xyzw: {quat}")
        if self.gripper_ctrl_index is not None:
            print(
                "[MuJoCo] Gripper ctrl: "
                f"open={self.gripper_open_ctrl:.4f}, close={self.gripper_close_ctrl:.4f}"
            )
        print(f"[MuJoCo] Arm gravity compensation: {self.arm_gravity_compensation}")

    def _reset_from_npz(self, reset_npz):
        reset_npz = Path(reset_npz)
        episode = np.load(reset_npz, allow_pickle=False)

        if "qpos_sim" in episode:
            qpos = np.asarray(episode["qpos_sim"][0], dtype=np.float64)
            if qpos.shape != self.data.qpos.shape:
                raise RuntimeError(f"{reset_npz}: qpos_sim shape {qpos.shape} != model qpos {self.data.qpos.shape}")
            self.data.qpos[:] = qpos

        if "qvel_sim" in episode:
            qvel = np.asarray(episode["qvel_sim"][0], dtype=np.float64)
            if qvel.shape != self.data.qvel.shape:
                raise RuntimeError(f"{reset_npz}: qvel_sim shape {qvel.shape} != model qvel {self.data.qvel.shape}")
            self.data.qvel[:] = qvel

        if "ctrl_sim" in episode:
            ctrl = np.asarray(episode["ctrl_sim"][0], dtype=np.float64)
            n = min(self.model.nu, ctrl.shape[0])
            self.data.ctrl[:n] = ctrl[:n]
        else:
            self.data.ctrl[:7] = self.clip_arm_ctrl(self.get_arm_qpos())
            self.set_gripper_open(immediate=True)

        mujoco.mj_forward(self.model, self.data)
        if self.gripper_ctrl_index is not None:
            self.gripper_target_ctrl = float(self.data.ctrl[self.gripper_ctrl_index])
        print(f"[MuJoCo] Seeded simulation from reset NPZ: {reset_npz}")

    def _set_finger_qpos_open(self):
        for name in ("finger_joint1", "finger_joint2"):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                continue
            qpos_idx = int(self.model.jnt_qposadr[joint_id])
            if bool(self.model.jnt_limited[joint_id]):
                self.data.qpos[qpos_idx] = float(self.model.jnt_range[joint_id][1])

    def get_arm_qpos(self):
        return np.asarray(self.data.qpos[self.qpos_indices], dtype=np.float64).copy()

    def get_arm_ctrl_target(self):
        if self.model.nu < 7:
            return self.get_arm_qpos()
        return self.clip_arm_ctrl(np.asarray(self.data.ctrl[:7], dtype=np.float64))

    def set_arm_qpos(self, q):
        q = np.asarray(q, dtype=np.float64).reshape(7)
        q = self.clip_joint_ranges(q)
        for value, qpos_idx in zip(q, self.qpos_indices):
            self.data.qpos[qpos_idx] = value
        for dof_idx in self.dof_indices:
            self.data.qvel[dof_idx] = 0.0

    def clip_joint_ranges(self, q):
        q = np.asarray(q, dtype=np.float64).reshape(7)
        lo = self.joint_ranges[:, 0]
        hi = self.joint_ranges[:, 1]
        return np.minimum(np.maximum(q, lo), hi)

    def clip_arm_ctrl(self, q):
        q = np.asarray(q, dtype=np.float64).reshape(7).copy()
        for i in range(min(7, self.model.nu)):
            ctrlrange = actuator_ctrl_range(self.model, i)
            if ctrlrange is not None:
                q[i] = float(np.clip(q[i], ctrlrange[0], ctrlrange[1]))
        return q

    def get_ee_pose(self):
        mujoco.mj_forward(self.model, self.data)
        pos = np.asarray(self.data.site_xpos[self.site_id], dtype=np.float32).copy()
        mat = np.asarray(self.data.site_xmat[self.site_id], dtype=np.float64).reshape(3, 3)
        quat = mat_to_quat_xyzw(mat)
        return pos, quat

    def command_ee_pose(self, pos_des, quat_des):
        q_target = self.solve_ik(pos_des, quat_des)
        q_target = self.clip_arm_ctrl(q_target)
        self.data.ctrl[:7] = q_target
        return q_target

    def set_gripper_pressed(self, pressed):
        if pressed:
            self.set_gripper_close()
        else:
            self.set_gripper_open()

    def set_gripper_open(self, immediate=False):
        if self.gripper_ctrl_index is not None:
            self._set_gripper_target(self.gripper_open_ctrl, immediate=immediate)

    def set_gripper_close(self, immediate=False):
        if self.gripper_ctrl_index is not None:
            self._set_gripper_target(self.gripper_close_ctrl, immediate=immediate)

    def _set_gripper_target(self, target_ctrl, immediate=False):
        target_ctrl = float(target_ctrl)
        self.gripper_target_ctrl = target_ctrl
        if immediate and self.gripper_ctrl_index is not None:
            self.data.ctrl[self.gripper_ctrl_index] = target_ctrl

    def _advance_gripper_ctrl(self, dt):
        if self.gripper_ctrl_index is None:
            return

        current = float(self.data.ctrl[self.gripper_ctrl_index])
        target = float(self.gripper_target_ctrl)
        delta = target - current
        if abs(delta) < 1e-9:
            self.data.ctrl[self.gripper_ctrl_index] = target
            return

        full_span = max(1e-9, abs(self.gripper_open_ctrl - self.gripper_close_ctrl))
        duration = self.gripper_close_time if target < current else self.gripper_open_time
        max_step = full_span * max(0.0, float(dt)) / max(1e-6, duration)

        if abs(delta) <= max_step:
            self.data.ctrl[self.gripper_ctrl_index] = target
        else:
            self.data.ctrl[self.gripper_ctrl_index] = current + np.sign(delta) * max_step

    def _apply_arm_gravity_compensation(self):
        if not self.arm_gravity_compensation:
            self.data.qfrc_applied[:] = 0.0
            return

        # qfrc_bias is c(q, v): Coriolis/centrifugal/gravity. Applying it on
        # the Panda DOFs cancels gravity while preserving MuJoCo contacts and
        # the XML's position actuators. Without this, the model's small wrist
        # actuator force limits let the arm sag even when no MC command changes.
        self.data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.data.qfrc_applied[self.dof_indices] = self.data.qfrc_bias[self.dof_indices]

    def step_sim(self, target_dt):
        steps = max(1, int(round(float(target_dt) / float(self.model.opt.timestep))))
        for _ in range(steps):
            self._advance_gripper_ctrl(self.model.opt.timestep)
            self._apply_arm_gravity_compensation()
            mujoco.mj_step(self.model, self.data)

    def solve_ik(self, pos_des, quat_des):
        pos_des = np.asarray(pos_des, dtype=np.float64).reshape(3)
        quat_des = normalize_quat(np.asarray(quat_des, dtype=np.float32))

        self.ik_data.qpos[:] = self.data.qpos
        self.ik_data.qvel[:] = 0.0
        # Seed from the current actuator target, not measured qpos. The measured
        # qpos can lag/sag under gravity, while ctrl[:7] is the pose MuJoCo was
        # already trying to hold before this Cartesian update.
        q = self.get_arm_ctrl_target()

        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)

        for _ in range(self.ik_iters):
            for value, qpos_idx in zip(q, self.qpos_indices):
                self.ik_data.qpos[qpos_idx] = value

            mujoco.mj_forward(self.model, self.ik_data)

            cur_pos = np.asarray(self.ik_data.site_xpos[self.site_id], dtype=np.float64)
            cur_quat = mat_to_quat_xyzw(self.ik_data.site_xmat[self.site_id].reshape(3, 3))

            pos_err = pos_des - cur_pos
            quat_err = quat_mul_xyzw(quat_des, quat_inv_xyzw(cur_quat))
            rot_err = quat_to_rotvec_xyzw(quat_err)

            if np.linalg.norm(pos_err) < 1e-4 and np.linalg.norm(rot_err) < 2e-3:
                break

            mujoco.mj_jacSite(self.model, self.ik_data, jacp, jacr, self.site_id)
            j_arm = np.vstack(
                (
                    self.ik_pos_weight * jacp[:, self.dof_indices],
                    self.ik_ori_weight * jacr[:, self.dof_indices],
                )
            )
            err = np.concatenate(
                (
                    self.ik_pos_weight * pos_err,
                    self.ik_ori_weight * rot_err,
                )
            )

            lhs = j_arm @ j_arm.T + (self.ik_damping ** 2) * np.eye(6)
            dq = j_arm.T @ np.linalg.solve(lhs, err)
            max_abs = float(np.max(np.abs(dq)))
            if max_abs > self.ik_max_joint_step:
                dq *= self.ik_max_joint_step / max_abs

            q = self.clip_joint_ranges(q + dq)

        return q


# ============================================================
# Main teleoperation loop
# ============================================================

def resolve_scene_arg(scene_arg):
    if scene_arg in SCENE_ALIASES:
        return (REPO_ROOT / SCENE_ALIASES[scene_arg]).resolve()

    scene_path = Path(scene_arg)
    if not scene_path.is_absolute():
        scene_path = REPO_ROOT / scene_path
    return scene_path.resolve()


def connect_quest_loop(discovery_timeout_s):
    while True:
        attempt = MetaQuest3ZmqReceiver(
            discovery_port=DISCOVERY_PORT,
            discovery_timeout_s=discovery_timeout_s,
            expected_type=EXPECTED_DISCOVERY_TYPE,
        )
        try:
            attempt.connect()
            return attempt
        except RuntimeError:
            print(
                "[MC/MuJoCo] Quest MC publisher not found yet. "
                "Enable Motion Controller mode (press A) in the Quest app, then this script will connect."
            )
            time.sleep(3)


def run_smoke_test(controller):
    pos0, quat0 = controller.get_ee_pose()
    pos_des = pos0 + np.array([0.015, -0.010, 0.010], dtype=np.float32)
    quat_des = quat_slerp_xyzw(quat0, quat_mul_xyzw(np.array([0.0, 0.0, 0.02, 0.9998], dtype=np.float32), quat0), 1.0)
    q_target = controller.command_ee_pose(pos_des, quat_des)
    controller.set_gripper_pressed(True)
    controller.step_sim(0.05)
    pos1, quat1 = controller.get_ee_pose()

    print("[SMOKE] IK target q:", q_target)
    print("[SMOKE] EE before:", pos0, quat0)
    print("[SMOKE] EE after: ", pos1, quat1)
    print("[SMOKE] ctrl:", controller.data.ctrl[: min(8, controller.model.nu)])


def run_headless(cfg, controller):
    while True:
        mq3 = connect_quest_loop(cfg.discovery_timeout)
        try:
            keep_running = run_teleop_session(cfg, controller, mq3, viewer=None)
            if not keep_running:
                break
        finally:
            mq3.close()

        print("[MC/MuJoCo] Ready for next intervention - waiting for MC mode to be activated...")


def run_with_viewer(cfg, controller):
    import mujoco.viewer

    with mujoco.viewer.launch_passive(controller.model, controller.data) as viewer:
        viewer.cam.lookat[:] = np.array([0.55, 0.0, 0.45], dtype=np.float64)
        viewer.cam.distance = 1.45
        viewer.cam.azimuth = 180.0
        viewer.cam.elevation = -28.0

        while viewer.is_running():
            mq3 = connect_quest_loop(cfg.discovery_timeout)
            try:
                keep_running = run_teleop_session(cfg, controller, mq3, viewer=viewer)
                if not keep_running:
                    break
            finally:
                mq3.close()

            print("[MC/MuJoCo] Ready for next intervention - waiting for MC mode to be activated...")


def run_teleop_session(cfg, controller, mq3, viewer=None):
    input_data = mq3.get_latest_controller_data()
    while not input_data or CONTROL_HAND not in input_data:
        if viewer is not None and not viewer.is_running():
            return False
        input_data = mq3.get_latest_controller_data()
        controller.step_sim(DT)
        if viewer is not None:
            viewer.sync()
        time.sleep(0.001)

    last_input_data = input_data

    print("Controller connected.")
    print(f"Initial input data: {input_data}")

    last_cmd_pos, last_cmd_quat = controller.get_ee_pose()

    pos_offset = None
    quat_engage_raw = None
    quat_engage_cmd = None

    prev_position_trigger = False
    prev_orientation_trigger = False

    prev_raw_pos = None
    prev_raw_quat = None

    position_settle_until = 0.0
    orientation_settle_until = 0.0

    loop_counter = 0
    command_counter = 0
    last_debug_time = time.perf_counter()
    last_packet_wall = time.perf_counter()
    prev_btns = {"X": True}
    done_intervention = False

    print("Starting low-latency MuJoCo teleoperation loop...")

    try:
        while not done_intervention:
            loop_start = time.perf_counter()

            if viewer is not None and not viewer.is_running():
                return False

            input_data = mq3.get_latest_controller_data()

            if not input_data:
                if time.perf_counter() - last_packet_wall > MC_PACKET_TIMEOUT_S:
                    print(
                        f"[MC/MuJoCo] No MC packets for {MC_PACKET_TIMEOUT_S:.1f}s - "
                        "intervention ended externally. Ending MC session."
                    )
                    done_intervention = True
                    continue
                if COMMAND_ONLY_ON_NEW_PACKET:
                    controller.step_sim(DT)
                    if viewer is not None:
                        viewer.sync()
                    time.sleep(0.001)
                    continue
                input_data = last_input_data

            if input_data is None or CONTROL_HAND not in input_data:
                controller.step_sim(DT)
                if viewer is not None:
                    viewer.sync()
                time.sleep(0.001)
                continue

            last_input_data = input_data
            last_packet_wall = time.perf_counter()

            hand_data = input_data[CONTROL_HAND]

            position_trigger_value = float(hand_data.get("index_trigger", 0.0))
            orientation_trigger_value = float(hand_data.get("hand_trigger", 0.0))

            position_trigger = position_trigger_value > POSITION_TRIGGER_THRESHOLD
            orientation_trigger = orientation_trigger_value > ORIENTATION_TRIGGER_THRESHOLD

            raw_pos = np.asarray(hand_data["pos"], dtype=np.float32)
            raw_quat = normalize_quat(np.asarray(hand_data["rot"], dtype=np.float32))

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

            if CONTROL_MODE == "mirror":
                ctrl_pos = POS_MIRROR_XY @ raw_pos
            else:
                ctrl_pos = raw_pos

            position_trigger_edge = position_trigger and not prev_position_trigger
            orientation_trigger_edge = orientation_trigger and not prev_orientation_trigger
            now_wall = time.perf_counter()

            if READ_SIM_POSE_ON_TRIGGER_EDGE and (
                position_trigger_edge or orientation_trigger_edge
            ):
                last_cmd_pos, last_cmd_quat = controller.get_ee_pose()

            if position_trigger_edge:
                pos_offset = last_cmd_pos - ctrl_pos
                position_settle_until = now_wall + max(0.0, float(cfg.clutch_settle))
                print("Position control engaged.")

            if orientation_trigger_edge:
                quat_engage_raw = raw_quat.copy()
                quat_engage_cmd = last_cmd_quat.copy()
                orientation_settle_until = now_wall + max(0.0, float(cfg.clutch_settle))
                print("Orientation control engaged.")

            prev_position_trigger = position_trigger
            prev_orientation_trigger = orientation_trigger

            position_settling = position_trigger and now_wall < position_settle_until
            orientation_settling = orientation_trigger and now_wall < orientation_settle_until

            if position_settling:
                # Keep the end effector fixed while the trigger press settles,
                # but continuously re-anchor to the latest controller pose.
                pos_offset = last_cmd_pos - ctrl_pos
                pos_des = last_cmd_pos
            elif pos_offset is not None and position_trigger:
                pos_des = ctrl_pos + pos_offset
                pos_des = clamp_step_pos(
                    desired=pos_des,
                    current=last_cmd_pos,
                    max_step=POSITION_STEP_CLAMP_M,
                )
            else:
                pos_des = last_cmd_pos

            if orientation_settling:
                # Same idea as position settling: no rotation should be caused
                # by the physical act of engaging the clutch.
                quat_engage_raw = raw_quat.copy()
                quat_engage_cmd = last_cmd_quat.copy()
                quat_des = last_cmd_quat
            elif quat_engage_raw is not None and orientation_trigger:
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

            if MC_SMOOTH_POS_ALPHA < 1.0:
                pos_des = (
                    MC_SMOOTH_POS_ALPHA * pos_des
                    + (1.0 - MC_SMOOTH_POS_ALPHA) * last_cmd_pos
                ).astype(np.float32)
            if MC_SMOOTH_ROT_ALPHA < 1.0:
                quat_des = quat_slerp_xyzw(last_cmd_quat, quat_des, MC_SMOOTH_ROT_ALPHA)

            # Real robot difference: hardware accepts -1/1 gripper commands.
            # The MuJoCo soft gripper uses a scalar actuator ctrl range.
            gripper_pressed = is_button_pressed(
                input_data=input_data,
                hand_name=CONTROL_HAND,
                button_name=GRIPPER_CONTROL_BUTTON,
            )
            controller.set_gripper_pressed(gripper_pressed)

            cartesian_active = position_trigger or orientation_trigger
            skip_cartesian_on_edge = position_trigger_edge or orientation_trigger_edge
            skip_cartesian_for_settle = position_settling or orientation_settling
            target_pos_delta = float(np.linalg.norm(np.asarray(pos_des) - np.asarray(last_cmd_pos)))
            target_ori_delta = float(quat_angle_deg_xyzw(last_cmd_quat, quat_des))
            target_changed = (
                target_pos_delta >= float(cfg.cartesian_pos_deadband)
                or target_ori_delta >= float(cfg.cartesian_ori_deadband_deg)
            )

            # Real robot difference: mq3_mc.py sends this Cartesian target to
            # Polymetis' CARTESIAN_IMPEDANCE policy. Here we solve a local IK
            # target and write MuJoCo joint-position actuator controls. On the
            # clutch edge/settle window we leave the previous arm ctrl untouched:
            # setting ctrl to measured qpos removes gravity-balancing preload and
            # makes this position-actuated model sag.
            if (
                cartesian_active
                and target_changed
                and not skip_cartesian_on_edge
                and not skip_cartesian_for_settle
            ):
                controller.command_ee_pose(pos_des, quat_des)
            controller.step_sim(DT)

            if (
                cartesian_active
                and target_changed
                and not skip_cartesian_on_edge
                and not skip_cartesian_for_settle
            ):
                last_cmd_pos = pos_des
                last_cmd_quat = quat_des
                command_counter += 1

            now_x = bool(input_data.get("X", False))
            if now_x and not prev_btns.get("X", False):
                print("[MC/MuJoCo] X (accept) seen - ending MC session.")
                done_intervention = True
            prev_btns = {"X": now_x}

            loop_counter += 1
            now = time.perf_counter()

            if PRINT_LOOP_HZ and now - last_debug_time > DEBUG_PERIOD_S:
                dt = now - last_debug_time
                measured_hz = loop_counter / dt
                command_hz = command_counter / dt
                dropped = mq3.frames_dropped
                received = mq3.frames_received
                ee_pos, _ = controller.get_ee_pose()
                print(
                    f"MuJoCo teleop loop: {measured_hz:.1f} Hz, "
                    f"commands: {command_hz:.1f} Hz, "
                    f"Quest frames dropped: {dropped}/{received}, "
                    f"ee={ee_pos}"
                )
                loop_counter = 0
                command_counter = 0
                mq3.frames_received = 0
                mq3.frames_dropped = 0
                last_debug_time = now

            if viewer is not None:
                viewer.sync()

            elapsed = time.perf_counter() - loop_start
            sleep_time = DT - elapsed

            if sleep_time > 0.0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("Stopping MuJoCo teleoperation...")
        return False

    return True


def parse_args():
    ap = argparse.ArgumentParser(description="Quest motion controller -> MuJoCo Franka teleoperation")
    ap.add_argument(
        "--scene",
        default="t_shape",
        help=(
            "Scene alias or XML path. Aliases: "
            + ", ".join(sorted(SCENE_ALIASES.keys()))
        ),
    )
    ap.add_argument("--site", default="gripper", help="MuJoCo site to control as the end effector")
    ap.add_argument("--reset-npz", default=None, help="Optional NPZ whose first qpos/qvel/ctrl frame seeds the sim")
    ap.add_argument("--headless", action="store_true", help="Run without a MuJoCo viewer")
    ap.add_argument("--smoke-test", action="store_true", help="Load the scene, run one IK/gripper command, then exit")
    ap.add_argument("--discovery-timeout", type=float, default=DISCOVERY_TIMEOUT_S, help="Quest UDP discovery timeout in seconds")
    ap.add_argument("--ik-pos-weight", type=float, default=1.0)
    ap.add_argument("--ik-ori-weight", type=float, default=0.35)
    ap.add_argument("--ik-damping", type=float, default=0.05)
    ap.add_argument("--ik-iters", type=int, default=8)
    ap.add_argument("--ik-max-joint-step", type=float, default=0.08)
    ap.add_argument("--gripper-close-time", type=float, default=GRIPPER_CLOSE_TIME_S, help="Seconds for the MuJoCo gripper to close after A is pressed")
    ap.add_argument("--gripper-open-time", type=float, default=GRIPPER_OPEN_TIME_S, help="Seconds for the MuJoCo gripper to open after A is released")
    ap.add_argument("--clutch-settle", type=float, default=CLUTCH_ENGAGE_SETTLE_S, help="Seconds to hold/re-anchor after translation or rotation clutch engagement")
    ap.add_argument("--cartesian-pos-deadband", type=float, default=CARTESIAN_POS_DEADBAND_M, help="Meters of desired EE motion required before sending a post-clutch IK command")
    ap.add_argument("--cartesian-ori-deadband-deg", type=float, default=CARTESIAN_ORI_DEADBAND_DEG, help="Degrees of desired EE rotation required before sending a post-clutch IK command")
    ap.add_argument("--no-arm-gravity-comp", action="store_true", help="Disable MuJoCo qfrc_bias gravity compensation on the Panda arm")
    return ap.parse_args()


def main(cfg=None):
    if cfg is None:
        cfg = parse_args()

    scene_path = resolve_scene_arg(cfg.scene)
    print(f"[MuJoCo] Loading XML: {scene_path}")

    loader = RepairedXmlLoader(REPO_ROOT)
    try:
        model, loaded_xml = loader.load(scene_path)
        data = mujoco.MjData(model)
        print(f"[MuJoCo] Loaded XML: {loaded_xml}")
        print(f"[MuJoCo] model nq={model.nq}, nv={model.nv}, nu={model.nu}, timestep={model.opt.timestep:g}")

        controller = MujocoFrankaController(
            model,
            data,
            site_name=cfg.site,
            reset_npz=cfg.reset_npz,
            ik_pos_weight=cfg.ik_pos_weight,
            ik_ori_weight=cfg.ik_ori_weight,
            ik_damping=cfg.ik_damping,
            ik_iters=cfg.ik_iters,
            ik_max_joint_step=cfg.ik_max_joint_step,
            gripper_close_time=cfg.gripper_close_time,
            gripper_open_time=cfg.gripper_open_time,
            arm_gravity_compensation=not cfg.no_arm_gravity_comp,
        )

        if cfg.smoke_test:
            run_smoke_test(controller)
            return

        print("[MC/MuJoCo] Enable Motion Controller mode in the Quest app to start teleoperation.")
        if cfg.headless:
            run_headless(cfg, controller)
        else:
            run_with_viewer(cfg, controller)
    finally:
        loader.close()


if __name__ == "__main__":
    main()
