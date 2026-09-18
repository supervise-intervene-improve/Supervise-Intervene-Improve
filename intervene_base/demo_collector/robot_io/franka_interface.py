import numpy as np

from real_robot_env.robot.hardware_franka import ControlType
from collect_data import Robot
import time


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


class FrankaInterface:
    def __init__(self, robot_key: str):
        if robot_key not in ROBOTS:
            raise ValueError(f"Unknown robot key: {robot_key}")

        self.robot = ROBOTS[robot_key]
        self.robot_key = robot_key
        self.current_control_type = None
        
    @property
    def name(self):
        return self.robot.name

    def connect_human_control(self):
        self.robot.connect(ControlType.HUMAN_CONTROL)
        self.current_control_type = ControlType.HUMAN_CONTROL
        self.robot.reset()

    def switch_control(self, control_type: ControlType):
        if self.current_control_type == control_type:
            return

        self.robot.connect(control_type)
        self.current_control_type = control_type

    def close(self):
        self.robot.close()

    def get_state(self):
        st = self.robot.robot_arm.get_state()
        q = st.joint_pos.detach().cpu().numpy().astype(np.float64)

        dq = getattr(st, "joint_vel", None)
        if dq is not None:
            dq = dq.detach().cpu().numpy().astype(np.float64)

        grip = float(self.robot.robot_gripper.get_sensors().item())
        return q, dq, grip

    def go_to_joint_and_gripper_safe(
        self,
        q_target: np.ndarray,
        grip_target: float,
        max_vel_norm_factor: float = 1.0,
    ):
        q_target = np.asarray(q_target, dtype=np.float64).reshape(7)
        grip_target = float(grip_target)

        prev_mode = self.current_control_type
        if prev_mode is None:
            prev_mode = ControlType.HUMAN_CONTROL

        try:
            self.switch_control(ControlType.HYBRID_JOINT_IMPEDANCE_CONTROL)

            self.robot.robot_arm.go_to_within_limits(
                q_target,
                max_vel_norm_factor=max_vel_norm_factor,
            )

            self.set_gripper_width_safe(grip_target)

        finally:
            self.switch_control(prev_mode)
    
    def go_to_initial_pose(self, time_to_go=2.5):
        """Go to initial pose"""
        print("Going to initial pose")
        self.robot.robot_arm.robot.go_home(time_to_go=time_to_go)
        # self.robot.go_home(time_to_go=time_to_go)

    def set_gripper_width_safe(self, target_width: float, tol: float = 0.002, timeout: float = 2.0, hz: float = 20.0):
        target_width = float(target_width)
        dt = 1.0 / hz
        t_end = time.time() + timeout

        while time.time() < t_end:
            current_width = float(self.robot.robot_gripper.get_sensors().item())
            err = target_width - current_width

            if abs(err) <= tol:
                break

            self.robot.robot_gripper.apply_commands(target_width, blocking=False)
            time.sleep(dt)

        final_width = float(self.robot.robot_gripper.get_sensors().item())
        print(f"[GRIPPER] target={target_width:.4f}, final={final_width:.4f}")