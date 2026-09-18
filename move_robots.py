from real_robot_env.robot.hardware_franka import ControlType
from collect_data import Robot
import argparse
from real_robot_env.robot.utils.keyboard import KeyManager

ROBOTS = {
    "p1": Robot(
        name="p1 leader",
                    ip_address="172.16.1.1",
            arm_port=1234,
            gripper_port=1235,
    ),
    "p2": Robot(
        name="p2 leader", ip_address="172.16.2.2", arm_port=4321,  gripper_port=4322
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("robots", nargs="+")
    args = parser.parse_args()

    for robot_name in args.robots:
        ROBOTS[robot_name].connect(ControlType.HUMAN_CONTROL)
        #ROBOTS[robot_name].connect(ControlType.CARTESIAN_IMPEDANCE_CONTROL)
        ROBOTS[robot_name].reset()

    km = KeyManager()
    print("Press 'q' to quit.")
    while km.key != "q":
        km.pool()

    for robot_name in args.robots:
        ROBOTS[robot_name].close()
