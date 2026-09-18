from real_robot_env.robot.hardware_devices import DiscreteDevice, ContinuousDevice
from real_robot_env.robot.hardware_franka import FrankaArm, ControlType
from real_robot_env.robot.hardware_frankahand import FrankaHand
from datetime import datetime
import torch
from pathlib import Path
import shutil
from enum import Enum, auto
import copy
from typing import NamedTuple, List
import time

import struct
import numpy as np

from zmq_server import start_sender_process, stop_sender_process


def _create_key_manager():
    from real_robot_env.robot.utils.keyboard import KeyManager

    return KeyManager()



class TeleoperationType(Enum):
    JOINT_SPACE = auto()
    TASK_SPACE = auto()


class Robot:
    def __init__(
        self,
        name: str,
        ip_address: str,
        arm_port: str,
        gripper_port: str,
        reset_pose: List[float] = None, # Nicht vorhanden bei P3P4
        policy_kwargs: dict = None,   # ✅ neu by IBA für Uebergabe von dof_mask (impedance control) oder anderen Parametern
    ):
        self.name = name

        self.robot_arm = FrankaArm(
            name=f"{name} arm",
            ip_address=ip_address,
            port=arm_port,
            default_reset_pose=reset_pose,
            policy_kwargs=policy_kwargs,   # ✅ um dof_mask durchreichen
        )

        self.robot_gripper = FrankaHand(
            name=f"{name} gripper",
            ip_address=ip_address,
            port=gripper_port,
        )

        self.is_connected = False

    def connect(self, control_type: ControlType):
        self.__connect_to_arm(control_type)
        self.__connect_to_gripper()

        self.is_connected = True

    def close(self):
        if not self.is_connected:
            return print("Robot is already not connected")

        self.robot_arm.close()
        self.robot_gripper.close()

        self.is_connected = False

    def reset(self):
        self.robot_arm.reset()
        self.robot_gripper.reset()

    def __connect_to_arm(self, control_type: ControlType):
        self.robot_arm.control_type = control_type
        assert self.robot_arm.connect(), f"Connection to {self.robot_arm.name} failed"

    def __connect_to_gripper(self):
        assert (
            self.robot_gripper.connect()
        ), f"Connection to {self.robot_gripper.name} failed"


class TeleoperationPair(NamedTuple):
    leader_robot: Robot
    follower_robot: Robot


class CollectionData:
    def __init__(self):
        self.joint_pos_list = []
        self.joint_vel_list = []
        self.ee_pos_list = []
        self.ee_vel_list = []
        self.gripper_state_list = []


class DataCollectionManager:
    def __init__(
        self,
        teleoperation_pairs: list[TeleoperationPair],
        data_dir: Path,
        teleoperation_type: TeleoperationType,
        discrete_devices: list[DiscreteDevice] = [],
        # continuous_devices: list[ContinuousDevice] = [],
        continuous_devices: bool = True,
        capture_interval: int = 0, # nicht vorhanden bei P3P4
        streaming: bool = False,
        save: bool = True,
        use_robot: bool = True,
        impedance_settings: dict = None
    ):
        self.teleoperation_type = teleoperation_type
        self.teleoperation_pairs = teleoperation_pairs
        self.impedance_settings = impedance_settings or {}

        self.use_robot = use_robot
        if self.use_robot:
            if self.teleoperation_type is TeleoperationType.JOINT_SPACE:
                follower_control_type = ControlType.IMITATION_CONTROL
            elif self.teleoperation_type is TeleoperationType.TASK_SPACE:
                follower_control_type = ControlType.CARTESIAN_IMPEDANCE_CONTROL
            else:
                raise ValueError("The given teleoperation type is invalid")

            for teleoperation_pair in self.teleoperation_pairs:
                teleoperation_pair.leader_robot.connect(ControlType.HUMAN_CONTROL)
                teleoperation_pair.follower_robot.connect(follower_control_type)

        self.discrete_cams = discrete_devices
        # self.continuous_cams = continuous_devices
        # self.__setup_cams()
        self.data_dir = data_dir
        self.data_dir.mkdir(exist_ok=True)

        self.timestamps = [] # nicht vorhanden bei P3P4
        self.cur_timestep = 0 # nicht vorhanden bei P3P4
        self.capture_interval = capture_interval # nicht vorhanden bei P3P4
        
        # Folgendes existiert bei P3P4 aber nicht bei P1P2
        """
        self.last_frame = time.time()
        self.dt = 0.07
        self.record_index = 0
        """

        self.streaming = streaming
        self.save = save
        if self.streaming == True:
            print("Starting ZMQ server for streaming data")
            start_sender_process()


    def start_key_listener(self):
        km = _create_key_manager()
        print("Press 'n' to collect new data or 'q' to quit data collection")

        while km.key != "q":
            if km.key == "n":
                print()
                print("Preparing for new data collection")

                self.__create_new_recording_dir()
                self.__create_empty_data()
                if self.use_robot:
                    self.__reset_robots()
                self.__start_continuous_recordings()

                print("Start! Press 's' to save collected data or 'd' to discard.")

                self.timestamps = []
                self.cur_timestep = 0
                while km.key not in ["s", "d"]:
                    if self.use_robot:
                        self.__collection_step()
                    km.pool()

                else:
                    self.__stop_continuous_recordings()
                    if km.key == "s":
                        print()
                        print("Saving data")
                        if self.use_robot:
                            self.__save_data()

                        print("Saved!")
                    elif km.key == "d":
                        print()
                        print("Discarding data")

                        shutil.rmtree(self.record_dir)
                        self.__discard_continuous_recordings()

                        print("Discarded!")

                    print(
                        "Press 'n' to collect new data or 'q' to quit data collection"
                    )

            km.pool()

        print()
        print("Ending data collection...")
        stop_sender_process()
        km.close()
        if self.use_robot:
            self.__close_hardware_connections()

    def __setup_cams(self):
        for cam in self.discrete_cams:
            #if not cam.connect(): self.discrete_cams.remove(cam)
            assert cam.connect(), f"Connection to {cam.name} failed"
        for cam in self.continuous_cams:
            #if not cam.connect(): self.continuous_cams.remove(cam)
            assert cam.connect(), f"Connection to {cam.name} failed"

    def __create_new_recording_dir(self):
        self.record_dir = self.data_dir / datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
        self.record_dir.mkdir()

        for teleoperation_pair in self.teleoperation_pairs:
            self.leader_robot_dir = (
                self.record_dir / teleoperation_pair.leader_robot.name
            )
            self.leader_robot_dir.mkdir()

            self.follower_robot_dir = (
                self.record_dir / teleoperation_pair.follower_robot.name
            )
            self.follower_robot_dir.mkdir()

        self.image_dir = self.record_dir / "images"
        self.image_dir.mkdir()
     
        # for cam in self.discrete_cams:
        #     cam_dir = self.image_dir / f"{cam.name}"
        #     cam_dir.mkdir()

        # for cam in self.continuous_cams:
        #     cam_dir = self.image_dir / f"{cam.name}"
        #     cam_dir.mkdir()

    def __create_empty_data(self):
        self.robot_to_data: dict[Robot, CollectionData] = {}

        for teleoperation_pair in self.teleoperation_pairs:
            self.robot_to_data[teleoperation_pair.leader_robot] = CollectionData()
            self.robot_to_data[teleoperation_pair.follower_robot] = CollectionData()

    def __reset_robots(self):
        for teleoperation_pair in self.teleoperation_pairs:
            teleoperation_pair.leader_robot.reset()
            teleoperation_pair.follower_robot.reset()

    def __start_continuous_recordings(self):
        # for cam in self.continuous_cams:
        #     cam.start_recording()
        pass

    def __stop_continuous_recordings(self):
        for cam in self.continuous_cams:
            cam.stop_recording()

    def __discard_continuous_recordings(self):
        # for device in self.continuous_cams:
        #     device.delete_recording()
        pass

    def __collection_step(self):
        for teleoperation_pair in self.teleoperation_pairs:
            self.leader_arm = teleoperation_pair.leader_robot.robot_arm
            # self.leader_gripper = teleoperation_pair.leader_robot.robot_gripper
            self.follower_arm = teleoperation_pair.follower_robot.robot_arm
            # self.follower_gripper = teleoperation_pair.follower_robot.robot_gripper

            leader_arm_state = self.leader_arm.get_state()
            # leader_gripper_width = self.leader_gripper.get_sensors().item()
            # leader_gripper_state = self.__get_follower_gripper_state(
            #     leader_gripper_width, self.leader_gripper.robot.metadata.max_width / 2
            # )

            if self.teleoperation_type is TeleoperationType.JOINT_SPACE:
                self.follower_arm.apply_commands(
                    q_desired=leader_arm_state.joint_pos,
                    qd_desired=leader_arm_state.joint_vel,
                )
            elif self.teleoperation_type is TeleoperationType.TASK_SPACE:
                self.follower_arm.robot.update_current_policy(
                    {
                        "ee_pos_desired": leader_arm_state.ee_pos[:3],
                        "ee_quat_desired": leader_arm_state.ee_pos[3:],
                        "ee_vel_desired": leader_arm_state.ee_vel[:3],
                        "ee_rvel_desired": leader_arm_state.ee_vel[3:],
                    }
                )
            else:
                raise ValueError("The given teleoperation type is invalid")

            # self.follower_gripper.apply_commands(leader_gripper_state)

            follower_arm_state = self.follower_arm.get_state()
            # follower_gripper_width = self.follower_gripper.get_sensors().item()
            # follower_gripper_state = self.__get_follower_gripper_state(
            #     follower_gripper_width,
            #     self.follower_gripper.robot.metadata.max_width / 2,
            # )

            cur_time = time.time()

            if not self.timestamps or cur_time - self.timestamps[-1] >= self.capture_interval:
                self.robot_to_data[teleoperation_pair.leader_robot].joint_pos_list.append(
                    leader_arm_state.joint_pos
                )
                self.robot_to_data[teleoperation_pair.leader_robot].joint_vel_list.append(
                    leader_arm_state.joint_vel
                )
                self.robot_to_data[teleoperation_pair.leader_robot].ee_pos_list.append(
                    leader_arm_state.ee_pos
                )
                self.robot_to_data[teleoperation_pair.leader_robot].ee_vel_list.append(
                    leader_arm_state.ee_vel
                )
                # self.robot_to_data[
                #     teleoperation_pair.leader_robot
                # ].gripper_state_list.append(leader_gripper_state)

                self.robot_to_data[teleoperation_pair.follower_robot].joint_pos_list.append(
                    follower_arm_state.joint_pos
                )
                self.robot_to_data[teleoperation_pair.follower_robot].joint_vel_list.append(
                    follower_arm_state.joint_vel
                )
                self.robot_to_data[teleoperation_pair.follower_robot].ee_pos_list.append(
                    follower_arm_state.ee_pos
                )
                self.robot_to_data[teleoperation_pair.follower_robot].ee_vel_list.append(
                    follower_arm_state.ee_vel
                )
                # self.robot_to_data[
                #     teleoperation_pair.follower_robot
                # ].gripper_state_list.append(follower_gripper_state)

                self.timestamps.append(cur_time)

                # for cam in self.discrete_cams:
                #     cam.store_last_frame(self.image_dir / cam.name, f"{self.cur_timestep + cam.start_frame_latency}")
            
                self.cur_timestep += 1
            #else: print(f"too quick: {self.cur_timestep}")
    


    # def __get_follower_gripper_state(self, leader_gripper_width: float, thresh: float):
        # if leader_gripper_width < thresh:
        #     return -1
        # else:
        #     return 1

    def __save_data(self):
        for teleoperation_pair in self.teleoperation_pairs:
            leader_data = self.robot_to_data[teleoperation_pair.leader_robot]
            follower_data = self.robot_to_data[teleoperation_pair.follower_robot]

            leader_joint_pos_list = torch.stack(leader_data.joint_pos_list)
            leader_joint_vel_list = torch.stack(leader_data.joint_vel_list)
            leader_ee_pos_list = torch.stack(leader_data.ee_pos_list)
            leader_ee_vel_list = torch.stack(leader_data.ee_vel_list)
            # leader_gripper_state_list = torch.Tensor(leader_data.gripper_state_list)

            follower_joint_pos_list = torch.stack(follower_data.joint_pos_list)
            follower_joint_vel_list = torch.stack(follower_data.joint_vel_list)
            follower_ee_pos_list = torch.stack(follower_data.ee_pos_list)
            follower_ee_vel_list = torch.stack(follower_data.ee_vel_list)
            # follower_gripper_state_list = torch.Tensor(follower_data.gripper_state_list)

            torch.save(
                leader_joint_pos_list,
                self.record_dir / teleoperation_pair.leader_robot.name / "joint_pos.pt",
            )
            torch.save(
                leader_joint_vel_list,
                self.record_dir / teleoperation_pair.leader_robot.name / "joint_vel.pt",
            )
            torch.save(
                leader_ee_pos_list,
                self.record_dir / teleoperation_pair.leader_robot.name / "ee_pos.pt",
            )
            torch.save(
                leader_ee_vel_list,
                self.record_dir / teleoperation_pair.leader_robot.name / "ee_vel.pt",
            )
            # torch.save(
            #     leader_gripper_state_list,
            #     self.record_dir
            #     / teleoperation_pair.leader_robot.name
            #     / "gripper_state.pt",
            # )

            torch.save(
                follower_joint_pos_list,
                self.record_dir
                / teleoperation_pair.follower_robot.name
                / "joint_pos.pt",
            )
            torch.save(
                follower_joint_vel_list,
                self.record_dir
                / teleoperation_pair.follower_robot.name
                / "joint_vel.pt",
            )
            torch.save(
                follower_ee_pos_list,
                self.record_dir / teleoperation_pair.follower_robot.name / "ee_pos.pt",
            )
            torch.save(
                follower_ee_vel_list,
                self.record_dir / teleoperation_pair.follower_robot.name / "ee_vel.pt",
            )
            # torch.save(
            #     follower_gripper_state_list,
            #     self.record_dir
            #     / teleoperation_pair.follower_robot.name
            #     / "gripper_state.pt",
            # )
      
        # for cam in self.continuous_cams:
        #     if self.save:
        #         cam.store_recording(self.image_dir / cam.name, "recording", self.timestamps)
        
        # for cam in self.discrete_cams:
        #     cam.timestamps = []

    def __close_hardware_connections(self):
        for teleoperation_pair in self.teleoperation_pairs:
            teleoperation_pair.leader_robot.close()
            teleoperation_pair.follower_robot.close()

        # for cam in self.discrete_cams:
        #     cam.close()

        # for cam in self.continuous_cams:
        #     cam.close()


if __name__ == "__main__":
    # from real_robot_env.robot.hardware_azure import Azure
    # from real_robot_env.robot.hardware_depthai import DepthAI, DAICameraType
    # from real_robot_env.robot.hardware_realsense import RealSense
    # from real_robot_env.robot.hardware_zed import Zed
    # from real_robot_env.robot.hardware_audio import AsyncAudioInterface
    # from real_robot_env.robot.hardware_cameras import AsynchronousCamera

    use_impedance_control = True

    teleoperation_pair = TeleoperationPair(
                leader_robot=Robot(
        name="p3 follower",
        ip_address="192.0.2.153",
        arm_port=50051,
        gripper_port=50052,
        ),
        follower_robot=Robot(
            name="p1 follower",
        ip_address="192.0.2.153",
        arm_port=50053,
        gripper_port=50054,
        ),
    )
    # print([dev.name for dev in DepthAI.get_devices()])


    # front_cam = AsynchronousCamera[RealSense](
    #    camera_class=RealSense,
    #    device_id='007522061936',
    #    name='front_cam', #add _orig here if you need itn
    #    height=512,
    #    width=512,
    # )

    # upper_cam = AsynchronousCamera[RealSense](
    #     camera_class=RealSense,
    #     device_id='007522061936',
    #     name='upper_cam', #add _orig here if you need itn
    #     fps=60,
    #     height=512,
    #     width=512,
    #     streaming=True, # Set to True if you want to stream the data
    #     save=False, # Set to True if you want to save the data
    #  )


    # wrist_cam = AsynchronousCamera[DepthAI](
    #     camera_class=DepthAI,
    #     device_id='14442C10113FE2D200',
    #     name='wrist_cam', #add _orig here if you need itnsq
    #     height=512,
    #     width=512,
    #     camera_type=DAICameraType.OAK_D_SR
    # )

    # mic = AsyncAudioInterface.get_specific_devices("Scarlett 4i4 4th Gen: USB Audnio")

    # continuous_devices = [upper_cam, down_cam, left_cam, right_cam]
    # continuous_devices = [upper_cam, left_cam, right_cam]

    data_collection_manager = DataCollectionManager(
        #teleoperation_pairs=[teleoperation_pair_1, teleoperation_pair_2],
        teleoperation_pairs=[teleoperation_pair],
        data_dir=Path("/home/user/Documents/data"),
        teleoperation_type=TeleoperationType.JOINT_SPACE,
        continuous_devices=None,
        # capture_interval=0.05, #20Hz robot state capture
        capture_interval=0.1, #10Hz robot state capture NEW
        streaming=True, # Set to True if you want to stream the data
        save=False, # Set to True if you want to save the data
        use_robot = True
    )

    data_collection_manager.start_key_listener()
