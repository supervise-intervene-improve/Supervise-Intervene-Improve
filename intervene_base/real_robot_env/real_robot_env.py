import cv2
import gymnasium as gym

from real_robot_env.robot.hardware_azure import Azure
from real_robot_env.robot.hardware_franka import ArmState, ControlType, FrankaArm
from real_robot_env.robot.hardware_frankahand import FrankaHand
import numpy as np
from typing import Any, NamedTuple

class Observation(NamedTuple):
    arm_state: ArmState
    gripper_width: float
    cam0_img: np.ndarray
    cam1_img: np.ndarray

class RealRobotEnv(gym.Env):
    def __init__(
        self,
        robot_name: str,
        robot_ip_address: str,
        robot_arm_port: int,
        robot_gripper_port: int,
    ):
        self.robot_arm, self.robot_gripper = self.__setup_robot(
            name=robot_name,
            ip_address=robot_ip_address,
            arm_port=robot_arm_port,
            gripper_port=robot_gripper_port,
        )

        self.cam0 = Azure(device_id=0)
        self.cam1 = Azure(device_id=1)
        
        assert self.cam0.connect(), f"Connection to {self.cam0.name} failed"
        assert self.cam1.connect(), f"Connection to {self.cam1.name} failed"

        # TODO add gymasium attributes here (action_space, observation_space, ...)

    def step(self, action: np.ndarray) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        des_joint_pos = action[:7]
        des_gripper_state = 1 if action[-1] > 0 else -1
        
        self.robot_arm.go_to_within_limits(goal=des_joint_pos)
        self.robot_gripper.apply_commands(width=des_gripper_state)
        
        obs = self.__get_obs()
        info = self.__get_info()
        
        return obs, 0, False, False, info # TODO truncated can be helpful (time limit, robot constraint violation, ...)
                        
    def reset(self) -> tuple[Observation, dict[str, Any]]:
        self.robot_arm.reset()
        self.robot_gripper.reset()
        
        obs = self.__get_obs()
        info = self.__get_info()
        
        return obs, info

    def close(self):
        self.robot_arm.close()
        self.robot_gripper.close()

        self.cam0.close()
        self.cam1.close()
            
    def __get_obs(self) -> Observation:
        arm_state = self.robot_arm.get_state()
        gripper_width = self.robot_gripper.get_sensors().item()
        
        img0 = self.cam0._get_sensors()["rgb"][:, :, :3]  # remove depth
        img1 = self.cam1._get_sensors()["rgb"][:, :, :3]

        img0 = cv2.resize(img0, (512, 512))[:, 100:370]
        img1 = cv2.resize(img1, (512, 512))
        
        processed_img0 = cv2.resize(img0, (128, 256)).astype(np.float32).transpose((2, 0, 1)) / 255.0
        processed_img1 = cv2.resize(img1, (256, 256)).astype(np.float32).transpose((2, 0, 1)) / 255.0
        
        return Observation(arm_state, gripper_width, processed_img0, processed_img1)
    
    def __get_info(self) -> dict[str, Any]:
        return {}

    def __setup_robot(
        self,
        name: str,
        ip_address: str,
        arm_port: int,
        gripper_port: int,
    ) -> tuple[FrankaArm, FrankaHand]:

        robot_arm = FrankaArm(
            name=f"{name} arm",
            ip_address=ip_address,
            port=arm_port,
            control_type=ControlType.HYBRID_JOINT_IMPEDANCE_CONTROL,
            hz=100,  # TODO why 100?
        )
        assert robot_arm.connect(), f"Connection to {robot_arm.name} failed"

        robot_gripper = FrankaHand(
            name=f"{name} gripper", ip_address=ip_address, port=gripper_port
        )
        assert robot_gripper.connect(), f"Connection to {robot_gripper.name} failed"

        return robot_arm, robot_gripper