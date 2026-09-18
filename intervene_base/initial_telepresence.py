import time
import numpy as np

import mujoco
import mujoco.viewer

# Control modes for the real Franka robot
from real_robot_env.robot.hardware_franka import ControlType

# Wrapper class used to connect to and control the real robot
from collect_data import Robot
from simpub.sim.mj_publisher import MujocoPublisher


# ---------------------------------------------------------
# Robot configuration dictionary
# Each entry defines a robot instance with its network
# configuration (IP + communication ports).
# ---------------------------------------------------------
ROBOTS = {
    "p1": Robot(
        name="p1 leader",
        ip_address="192.168.0.150",
        arm_port=1234,
        gripper_port=1235,
    ),
    "p2": Robot(
        name="p2 leader",
        ip_address="192.168.0.150",
        arm_port=4321,
        gripper_port=4322,
    ),
    "p3": Robot(
        name="p3 follower",
        ip_address="192.0.2.153",
        arm_port=50051,
        gripper_port=50052,
    ),
    "p4": Robot(
        name="p4 follower",
        ip_address="192.0.2.153",
        arm_port=50053,
        gripper_port=50054,
    ),
}

# Select which robot to control
ROBOT_KEY = "p1"

# MuJoCo simulation scene
# MUJOCO_XML_PATH = "/path/to/Intervention_IL_AR/intervene_base/mujoco_scenes/working_scenes/with_soft_gripper/sii_scene_table_boxes_cups.xml"
# MUJOCO_XML_PATH = "/path/to/Intervention_IL_AR/intervene_base/franka_emika_panda/working_scenes/with_soft_gripper/sii_scene_table_T_shape.xml"
MUJOCO_XML_PATH = "/path/to/Intervention_IL_AR/intervene_base/mujoco_scenes/working_scenes/with_soft_gripper/sii_scene_table_wire_base_and_spoon.xml"
# Alternative scenes (commented)
# MUJOCO_XML_PATH = "/home/user/Documents/intervene_base/franka_emika_panda/lab2.xml"
# MUJOCO_XML_PATH = "/home/user/Documents/intervene_base/franka_emika_panda/lab3.xml"


# ---------------------------------------------------------
# Names of the Panda arm joints inside the MuJoCo model
# These must match the joint names defined in the XML.
# ---------------------------------------------------------
PANDA_JOINT_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "joint7",
]


# Viewer refresh rate (Hz)
VIEW_HZ = 60.0

# Simulation control period
DT = 1.0 / VIEW_HZ

# Low-pass smoothing factor for control signals
# (helps reduce jitter when mirroring real robot states)
ALPHA = 0.9


def get_qpos_indices_for_joints(model: mujoco.MjModel, joint_names: list[str]) -> list[int]:
    """
    Return the qpos indices in MuJoCo corresponding to the given joint names.

    MuJoCo stores joint positions in the global vector `data.qpos`.
    Each joint has an index (address) inside that vector.
    This function maps joint names -> qpos indices.
    """
    idxs = []
    for name in joint_names:
        j = model.joint(name)              # Access joint object
        joint_id = j.id                    # Internal MuJoCo joint ID
        qpos_adr = model.jnt_qposadr[joint_id]  # Address inside qpos
        idxs.append(int(qpos_adr))
    return idxs


def main():
    # ---------------------------------------------------------
    # Load MuJoCo model and create simulation data structure
    # ---------------------------------------------------------
    print(f"[INFO] Loading MuJoCo XML: {MUJOCO_XML_PATH}")
    model = mujoco.MjModel.from_xml_path(MUJOCO_XML_PATH)
    data = mujoco.MjData(model)

    # ---------------------------------------------------------
    # Get qpos indices for the Panda joints
    # ---------------------------------------------------------
    try:
        arm_qpos_idxs = get_qpos_indices_for_joints(model, PANDA_JOINT_NAMES)

    except Exception:
        # Helpful debug output if joint names do not match
        print("\n[ERROR] Could not find expected Panda joint names in the MuJoCo model.")
        print("        Your XML likely uses different joint names.")

        print("\n[DEBUG] Available joint names in this model:")
        print(list(model.joint_names))
        raise

    # Print joint mapping for debugging
    print("[INFO] Joint mapping (MuJoCo qpos indices):")
    for name, idx in zip(PANDA_JOINT_NAMES, arm_qpos_idxs):
        print(f"  {name:>12s} -> qpos[{idx}]")

    # ---------------------------------------------------------
    # Connect to the real robot
    # ---------------------------------------------------------
    robot = ROBOTS[ROBOT_KEY]

    print(f"\n[INFO] Connecting to real robot: {ROBOT_KEY} ({robot.name})")

    # Start robot in human control mode
    robot.connect(ControlType.HUMAN_CONTROL)

    # Reset robot state
    robot.reset()

    print("[INFO] Connected.")

    # ---------------------------------------------------------
    # Read initial joint positions from the real robot
    # ---------------------------------------------------------
    state0 = robot.robot_arm.get_state()

    # Convert torch tensor -> numpy array
    q0 = state0.joint_pos.detach().cpu().numpy().astype(np.float64)

    if q0.shape[0] != 7:
        raise RuntimeError(f"Expected 7 arm joints, got shape {q0.shape}")

    # ---------------------------------------------------------
    # Initialize MuJoCo simulation to match the real robot pose
    # ---------------------------------------------------------
    for i, qpos_idx in enumerate(arm_qpos_idxs):
        data.qpos[qpos_idx] = q0[i]

    # Update internal MuJoCo state
    mujoco.mj_forward(model, data)

    # Initialize controller targets
    data.ctrl[:7] = q0.copy()

    print("[INFO] Initialized sim to real robot pose. Starting mirror loop.")
    print("[INFO] Close the MuJoCo window to stop, or press Ctrl+C in terminal.\n")
    publisher = MujocoPublisher(model, data, host="192.168.0.38", visible_geoms_groups=[0, 2])

    try:
        # ---------------------------------------------------------
        # Launch MuJoCo viewer in passive mode
        # (we manually step the simulation)
        # ---------------------------------------------------------
        with mujoco.viewer.launch_passive(model, data) as viewer:

            while viewer.is_running():

                t0 = time.time()

                # -------------------------------------------------
                # Read current joint state from real robot
                # -------------------------------------------------
                state = robot.robot_arm.get_state()

                q_real = state.joint_pos.detach().cpu().numpy().astype(np.float64)

                # Smooth control commands to reduce noise
                data.ctrl[:7] = (1.0 - ALPHA) * data.ctrl[:7] + ALPHA * q_real


                # -------------------------------------------------
                # GRIPPER CONTROL
                # Convert gripper width -> finger joint position
                # -------------------------------------------------
                gripper_width = robot.robot_gripper.get_sensors().item()  # meters

                finger_pos = np.clip(gripper_width / 2.0, 0.0, 0.04)

                data.ctrl[7] = (1.0 - ALPHA) * data.ctrl[7] + ALPHA * finger_pos

                if q_real.shape[0] != 7:
                    raise RuntimeError(f"Expected 7 arm joints, got shape {q_real.shape}")


                # -------------------------------------------------
                # Advance simulation
                # -------------------------------------------------
                steps = max(1, int(DT / model.opt.timestep))

                for _ in range(steps):
                    mujoco.mj_step(model, data)

                # Update viewer
                viewer.sync()


                # -------------------------------------------------
                # Maintain fixed control frequency
                # -------------------------------------------------
                elapsed = time.time() - t0
                sleep_time = DT - elapsed

                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt: stopping.")

    finally:
        # ---------------------------------------------------------
        # Clean shutdown
        # ---------------------------------------------------------
        print("[INFO] Closing robot connection...")
        robot.close()
        print("[INFO] Done.")


# ---------------------------------------------------------
# Script entry point
# ---------------------------------------------------------
if __name__ == "__main__":
    main()
    