import time
import numpy as np

import mujoco
import mujoco.viewer

from real_robot_env.robot.hardware_franka import ControlType
from collect_data import Robot


ROBOTS = {
    "p1": Robot(
        name="p1 leader",
        ip_address="172.16.1.1",
        arm_port=1234,
        gripper_port=1235,
    ),
    "p2": Robot(
        name="p2 leader",
        ip_address="172.16.2.2",
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

ROBOT_KEY = "p1"  

MUJOCO_XML_PATH = "/path/to/workspace/franka_emika_panda/mjx_panda.xml"  

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


def get_qpos_indices_for_joints(model: mujoco.MjModel, joint_names: list[str]) -> list[int]:
    """Return qpos indices (qpos addresses) for a list of joint names."""
    idxs = []
    for name in joint_names:
        j = model.joint(name)         
        joint_id = j.id
        qpos_adr = model.jnt_qposadr[joint_id]
        idxs.append(int(qpos_adr))
    return idxs


def main():
    
    print(f"[INFO] Loading MuJoCo XML: {MUJOCO_XML_PATH}")
    model = mujoco.MjModel.from_xml_path(MUJOCO_XML_PATH)
    data = mujoco.MjData(model)

    try:
        arm_qpos_idxs = get_qpos_indices_for_joints(model, PANDA_JOINT_NAMES)
    except Exception as e:
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
    print("[INFO] Connected. Starting mirror loop.")
    print("[INFO] Close the MuJoCo window to stop, or press Ctrl+C in terminal.\n")


    last_ok_state_time = time.time()

    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                t0 = time.time()

                state = robot.robot_arm.get_state()

                q_real = state.joint_pos.detach().cpu().numpy().astype(np.float64)

                if q_real.shape[0] != 7:
                    raise RuntimeError(f"Expected 7 arm joints, got shape {q_real.shape}")

                for i, qpos_idx in enumerate(arm_qpos_idxs):
                    data.qpos[qpos_idx] = q_real[i]

                mujoco.mj_forward(model, data)

                viewer.sync()

                last_ok_state_time = time.time()

                elapsed = time.time() - t0
                sleep_time = DT - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt: stopping.")
    finally:
        print("[INFO] Closing robot connection...")
        robot.close()
        print("[INFO] Done.")


if __name__ == "__main__":
    main()
