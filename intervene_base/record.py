import time
import numpy as np
import threading

import mujoco
import mujoco.viewer

from real_robot_env.robot.hardware_franka import ControlType
from collect_data import Robot
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

@dataclass
class TrajectoryRecorder:
    # Path where the trajectory file will be saved (.npz)
    save_path: Path
    
    # Identifier for the lab / experiment (useful for metadata)
    lab_id: str
    
    # Logging frequency (how often we WANT to record data)
    log_hz: float
    
    # Simulation / viewer frequency (how often record() gets called)
    view_hz: float

    # Internal: how many steps to skip between logs
    _stride: int = field(init=False)
    
    # Internal counter to track steps
    _k: int = field(default=0, init=False)

    # --- Logged data buffers (grow over time) ---
    
    # Wall-clock timestamps (real time)
    wall_t: list = field(default_factory=list)
    
    # Simulation timestamps (MuJoCo time)
    sim_t: list = field(default_factory=list)
    
    # Real robot joint positions
    q_real: list = field(default_factory=list)
    
    # Real robot joint velocities
    dq_real: list = field(default_factory=list)
    
    # Real gripper width / state
    grip_real: list = field(default_factory=list)
    
    # Simulated joint positions (MuJoCo qpos)
    qpos_sim: list = field(default_factory=list)
    
    # Simulated joint velocities (MuJoCo qvel)
    qvel_sim: list = field(default_factory=list)
    
    # Control inputs sent to simulation (actuators)
    ctrl_sim: list = field(default_factory=list)

    def __post_init__(self):
        # Compute how many simulation steps to skip between logs
        # Example: view_hz=1000, log_hz=100 → stride=10
        self._stride = max(1, int(round(self.view_hz / self.log_hz)))

    def record(self, now_wall, data, q_real, dq_real, gripper_width):
        # Increment step counter
        self._k += 1

        # Only log every `_stride` steps (downsampling)
        if (self._k % self._stride) != 0:
            return

        # --- Time ---
        self.wall_t.append(float(now_wall))       # real-world time
        self.sim_t.append(float(data.time))       # MuJoCo simulation time

        # --- Real robot state ---
        # Store a copy to avoid accidental mutation later
        self.q_real.append(np.asarray(q_real, dtype=np.float64).copy())

        # Handle missing velocity data
        if dq_real is None:
            # Fill with NaNs if unavailable
            self.dq_real.append(np.full_like(q_real, np.nan, dtype=np.float64))
        else:
            self.dq_real.append(np.asarray(dq_real, dtype=np.float64).copy())

        self.grip_real.append(float(gripper_width))

        # --- Simulation state (from MuJoCo `data`) ---
        self.qpos_sim.append(np.asarray(data.qpos, dtype=np.float64).copy())
        self.qvel_sim.append(np.asarray(data.qvel, dtype=np.float64).copy())

        # --- Control inputs ---
        self.ctrl_sim.append(np.asarray(data.ctrl, dtype=np.float64).copy())

    def save(self):
        # Ensure output directory exists
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save everything as a compressed NumPy archive
        np.savez_compressed(
            self.save_path,
            wall_t=np.array(self.wall_t),
            sim_t=np.array(self.sim_t),
            q_real=np.stack(self.q_real),
            dq_real=np.stack(self.dq_real),
            grip_real=np.array(self.grip_real),
            qpos_sim=np.stack(self.qpos_sim),
            qvel_sim=np.stack(self.qvel_sim),
            ctrl_sim=np.stack(self.ctrl_sim),
        )

        # Simple confirmation message
        print(f"[INFO] Saved trajectory to: {self.save_path}")

# Addresses come from robot.robot_endpoints so they can be corrected with an env var
# (INTERVENE_ROBOT_P4_IP etc.) instead of a source edit. The lab machines are on DHCP,
# so a baked-in address can drift and then surface only as a gRPC "failed to connect to
# all addresses" at the first intervention attempt. Defaults are unchanged.
# utils/robot_doctor.py reads the same table, so the doctor and the real connection can
# never disagree about where the robot is supposed to be.
from robot.robot_endpoints import ENDPOINTS as ROBOT_ENDPOINTS

ROBOTS = {
    key: Robot(
        name=endpoint.name,
        ip_address=endpoint.ip_address,
        arm_port=endpoint.arm_port,
        gripper_port=endpoint.gripper_port,
    )
    for key, endpoint in ROBOT_ENDPOINTS.items()
}


LAB_ID="T_shape"
ROBOT_KEY = "p1"
MUJOCO_XML_PATH = "mujoco_scenes/working_scenes/with_soft_gripper/sii_scene_table_" + LAB_ID + ".xml"
#MUJOCO_XML_PATH = "/path/to/workspace/franka_emika_panda/lab2.xml"
#MUJOCO_XML_PATH = "/path/to/workspace/franka_emika_panda/lab3.xml"

PANDA_JOINT_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
]

VIEW_HZ = 60.0
DT = 1.0 / VIEW_HZ

ALPHA = 0.6

def get_qpos_indices_for_joints(model: mujoco.MjModel, joint_names: list[str]) -> list[int]:
    """Return qpos indices (qpos addresses) for a list of joint names."""
    idxs = []
    for name in joint_names:
        j = model.joint(name)
        joint_id = j.id
        qpos_adr = model.jnt_qposadr[joint_id]
        idxs.append(int(qpos_adr))
    return idxs

def restore_gripper_width_binary(robot, target_width: float, tol: float = 0.002, timeout: float = 1.5, hz: float = 40.0):
    target_width = float(np.clip(
        target_width,
        robot.robot_gripper.min_width,
        robot.robot_gripper.max_width,
    ))

    dt = 1.0 / hz
    t_end = time.time() + timeout

    while time.time() < t_end:
        current_width = float(robot.robot_gripper.get_sensors().item())
        err = target_width - current_width

        if abs(err) <= tol:
            break

        cmd = 1.0 if err > 0.0 else -1.0
        robot.robot_gripper.apply_commands(cmd)
        time.sleep(dt)

    final_width = float(robot.robot_gripper.get_sensors().item())
    print(f"[HOLD] Gripper restore target={target_width:.4f}, final={final_width:.4f}")

def hold_then_replan_record(
    robot: Robot,
    q_hold: np.ndarray,
    finger_hold: float,
    grip_width_hold: float | None,
    save_path: Path,
    *,
    mujoco_lab_id: str,
    mujoco_xml_path: str,
    view_hz: float = 60.0,
    log_hz: float = 60.0,
    alpha: float = 0.6,
    qpos_seed: np.ndarray | None = None,
    qvel_seed: np.ndarray | None = None,
    already_connected: bool = False,
):

    q_hold = np.asarray(q_hold, dtype=np.float64).reshape(7)
    finger_hold = float(np.clip(finger_hold, 0.0, 0.04))
    dt = 1.0 / view_hz

    model = mujoco.MjModel.from_xml_path(mujoco_xml_path)
    data = mujoco.MjData(model)

    if qpos_seed is not None:
        qpos_seed = np.asarray(qpos_seed, dtype=np.float64)
        if qpos_seed.shape != data.qpos.shape:
            raise RuntimeError(f"qpos_seed shape {qpos_seed.shape} != data.qpos shape {data.qpos.shape}")
        data.qpos[:] = qpos_seed

    if qvel_seed is not None:
        qvel_seed = np.asarray(qvel_seed, dtype=np.float64)
        if qvel_seed.shape != data.qvel.shape:
            raise RuntimeError(f"qvel_seed shape {qvel_seed.shape} != data.qvel shape {data.qvel.shape}")
        data.qvel[:] = qvel_seed


    data.ctrl[:7] = q_hold.copy()
    if model.nu >= 8:
        data.ctrl[7] = finger_hold
    mujoco.mj_forward(model, data)

    if not already_connected:
        robot.connect(ControlType.HYBRID_JOINT_IMPEDANCE_CONTROL)

    if grip_width_hold is not None:
        restore_gripper_width_binary(robot, grip_width_hold)


    print("\n[HOLD] Moving/holding robot at paused pose. It should not fall back now.")
    print("[HOLD] Please put your hands on the robot and be ready to guide it.")
    print("[HOLD] When you're holding it steady at that pose, type 'y' and press Enter.")
    
    last_prompt = 0.0
    ready = False
    while not ready:
        # robot.robot_arm.apply_commands(q_hold)
        robot.robot_arm.go_to_within_limits(q_hold)

        time.sleep(dt)

        now = time.time()
        if now - last_prompt > 0.7:
            last_prompt = now
            ans = input("Ready to take control? Hold the robot there and type y (or n to keep holding): ").strip().lower()
            if ans == "y":
                ready = True

    print("[REPLAN] Starting replanning record NOW. Guide the robot. Close window / Ctrl+C to stop.")

    rec = TrajectoryRecorder(save_path=save_path, lab_id=mujoco_lab_id, log_hz=log_hz, view_hz=view_hz)

    try:
        st = robot.robot_arm.get_state()
        q_anchor = st.joint_pos.detach().cpu().numpy().astype(np.float64)

        robot.connect(ControlType.HUMAN_CONTROL)


        st2 = robot.robot_arm.get_state()
        q2 = st2.joint_pos.detach().cpu().numpy().astype(np.float64)

        if np.linalg.norm(q2 - q_anchor) > 0.2:
            print("[WARN] Switching to HUMAN_CONTROL caused a jump/reset. Falling back to HYBRID impedance.")
            robot.connect(ControlType.HYBRID_JOINT_IMPEDANCE_CONTROL)
           
            for _ in range(int(0.5 * view_hz)):
                robot.robot_arm.apply_commands(q_anchor)
                time.sleep(1.0 / view_hz)
        else:
            print("[INFO] HUMAN_CONTROL active. You can guide freely.")

    except Exception as e:
        print(f"[WARN] Could not switch to HUMAN_CONTROL ({e}). Staying in HYBRID impedance.")
        try:
            robot.connect(ControlType.HYBRID_JOINT_IMPEDANCE_CONTROL)
        except Exception:
            pass


    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                t0 = time.time()

                state = robot.robot_arm.get_state()
                q_real = state.joint_pos.detach().cpu().numpy().astype(np.float64)

                data.ctrl[:7] = (1.0 - alpha) * data.ctrl[:7] + alpha * q_real

                gripper_width = robot.robot_gripper.get_sensors().item()  # meters
                if model.nu >= 8:
                    finger_pos = np.clip(gripper_width / 2.0, 0.0, 0.04)
                    data.ctrl[7] = (1.0 - alpha) * data.ctrl[7] + alpha * finger_pos

                dq_real = getattr(state, "joint_vel", None)
                if dq_real is not None:
                    dq_real = dq_real.detach().cpu().numpy().astype(np.float64)

                rec.record(time.time(), data, q_real, dq_real, gripper_width)

                steps = max(1, int(dt / model.opt.timestep))
                for _ in range(steps):
                    mujoco.mj_step(model, data)

                viewer.sync()
                sleep_time = dt - (time.time() - t0)
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[REPLAN] Stopping replanning record.")
    finally:
        rec.save()

    return save_path

def main():
    print(f"[INFO] Loading MuJoCo XML: {MUJOCO_XML_PATH}")
    model = mujoco.MjModel.from_xml_path(MUJOCO_XML_PATH)
    data = mujoco.MjData(model)

    try:
        arm_qpos_idxs = get_qpos_indices_for_joints(model, PANDA_JOINT_NAMES)
    except Exception:
        print("\n[ERROR] Could not find expected Panda joint names in the MuJoCo model.")
        print("        Your XML likely uses different joint names.")
        print("\n[DEBUG] Available joint names in this model:")
        print(list(model.joint_names))
        raise

    print("[INFO] Joint mapping (MuJoCo qpos indices):")
    for name, idx in zip(PANDA_JOINT_NAMES, arm_qpos_idxs):
        print(f"  {name:>12s} -> qpos[{idx}]")

    robot = ROBOTS[ROBOT_KEY]
    print(f"\n[INFO] Connecting to real robot: {ROBOT_KEY} ({robot.name})")

    robot.connect(ControlType.HUMAN_CONTROL)
    robot.reset()
    print("[INFO] Connected.")

    state0 = robot.robot_arm.get_state()
    q0 = state0.joint_pos.detach().cpu().numpy().astype(np.float64)

    if q0.shape[0] != 7:
        raise RuntimeError(f"Expected 7 arm joints, got shape {q0.shape}")

    for i, qpos_idx in enumerate(arm_qpos_idxs):
        data.qpos[qpos_idx] = q0[i]

    mujoco.mj_forward(model, data)

    data.ctrl[:7] = q0.copy()

    rec = TrajectoryRecorder(
        save_path=Path("teleop_logs") / f"{ROBOT_KEY}_traj_{int(time.time())}.npz",
        lab_id=LAB_ID,
        log_hz=60.0,     
        view_hz=VIEW_HZ, 
    )

    print("[INFO] Initialized sim to real robot pose. Starting mirror loop.")
    print("[INFO] Close the MuJoCo window to stop, or press Ctrl+C in terminal.\n")

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                t0 = time.time()

                state = robot.robot_arm.get_state()
                q_real = state.joint_pos.detach().cpu().numpy().astype(np.float64)

                data.ctrl[:7] = (1.0 - ALPHA) * data.ctrl[:7] + ALPHA * q_real

                gripper_width = robot.robot_gripper.get_sensors().item()  
                finger_pos = np.clip(gripper_width / 2.0, 0.0, 0.04)

                data.ctrl[7] = (1.0 - ALPHA) * data.ctrl[7] + ALPHA * finger_pos

                if q_real.shape[0] != 7:
                    raise RuntimeError(f"Expected 7 arm joints, got shape {q_real.shape}")

                dq_real = getattr(state, "joint_vel", None)

                if dq_real is not None:
                    dq_real = dq_real.detach().cpu().numpy().astype(np.float64)

                rec.record(
                    now_wall=time.time(),
                    data=data,
                    q_real=q_real,
                    dq_real=dq_real,
                    gripper_width=gripper_width,
                )

                steps = max(1, int(DT / model.opt.timestep))
                for _ in range(steps):
                    mujoco.mj_step(model, data)

                viewer.sync()

                elapsed = time.time() - t0
                sleep_time = DT - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt: stopping.")
    finally:
        rec.save()
        print("[INFO] Closing robot connection...")
        robot.close()
        print("[INFO] Done.")

if __name__ == "__main__":
    main()
