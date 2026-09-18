"""
adopted from gello repo new update for factr style gravity compensated teleop for YAM

Standalone FACTR Gravity Compensation Script (Non-ROS)

This script provides the similar gravity compensation functionality as the ROS-based
FACTR teleop system, but without ROS dependencies.
Usage:
    python3 gello/factr/gravity_compensation.py --config configs/yam_gello_factr_hw.yaml
"""

import argparse
import json
import logging
import math
import os
import signal
import subprocess
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt
import pinocchio as pin
import yaml

from factr.hardware.dynamixel import DynamixelDriver

log = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "factr" / "leader.yaml"


class FactrError(RuntimeError):
    """Base exception for repo-local FACTR hardware integration."""


class FactrSafetyError(FactrError):
    """Raised when a FACTR state or command violates a safety gate."""


@dataclass(frozen=True)
class FactrState:
    joint_positions: npt.NDArray[np.float64]
    joint_velocities: npt.NDArray[np.float64]
    gripper_position: float
    gripper_velocity: float
    timestamp: float


def find_ttyusb(port_name: str) -> str:
    """Locate the underlying ttyUSB device."""
    base_path = "/dev/serial/by-id/"
    full_path = os.path.join(base_path, port_name)
    if not os.path.exists(full_path):
        raise Exception(f"Port '{port_name}' does not exist in {base_path}.")
    try:
        resolved_path = os.readlink(full_path)
        actual_device = os.path.basename(resolved_path)
        if actual_device.startswith("ttyUSB"):
            return actual_device
        else:
            raise Exception(
                f"The port '{port_name}' does not correspond to a ttyUSB device. It links to {resolved_path}."
            )
    except Exception as e:
        raise Exception(
            f"Unable to resolve the symbolic link for '{port_name}'. {e}"
        ) from e


def _instantiate_from_dict(cfg: Dict[str, Any]) -> Any:
    """Lightweight instantiation from a dict with a _target_ path.

    Keeps this script self-contained without importing broader launch utilities.
    """
    assert isinstance(cfg, dict) and "_target_" in cfg, "Invalid instantiation config"
    module_path, class_name = cfg["_target_"].rsplit(".", 1)
    cls = getattr(import_module(module_path), class_name)
    kwargs = {k: v for k, v in cfg.items() if k != "_target_"}

    # Recurse into nested dicts/lists
    def _recurse(v):
        if isinstance(v, dict) and "_target_" in v:
            return _instantiate_from_dict(v)
        if isinstance(v, dict):
            return {kk: _recurse(vv) for kk, vv in v.items()}
        if isinstance(v, list):
            return [_recurse(x) for x in v]
        return v

    return cls(**{k: _recurse(v) for k, v in kwargs.items()})


class FACTRGravityCompensation:
    """
    Standalone FACTR gravity compensation system without ROS dependencies.

    This class implements the core functionality of FACTR teleop gravity compensation,
    including:
    - Gravity compensation using inverse dynamics
    - Null-space regulation
    - Joint limit barriers
    - Static friction compensation
    """

    CALIBRATION_RANGE_MULTIPLIER = 20  # Range: -20π to 20π
    CALIBRATION_STEP_COUNT = 81  # 20 * 4 + 1 steps

    def __init__(
        self,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        *,
        port: str | None = None,
        read_only: bool = False,
        command_enabled: bool | None = None,
    ):
        self.running = False
        self.ctrl_thread = None

        if command_enabled is not None:
            read_only = not command_enabled
        self.config_path = str(config_path)
        self.port_override = port
        self.read_only = bool(read_only)
        self.driver: Optional[DynamixelDriver] = None  # Initialize early for cleanup
        self._torque_enabled_by_us = False

        self.initial_joint_pos = np.array(
            [0.0, -1.882, 0.0, -3.157, 0.058, 1.266, 0.0, 0.0]
        )

        # Franka external torque feedback (non-blocking)
        self._franka_arm = None
        self._franka_query_executor: Optional[ThreadPoolExecutor] = None
        self._franka_query_future: Optional[Future] = None
        self._franka_query_timeout_s = 5e-4  # 0.0005s
        self._ext_tau_lock = threading.Lock()
        self._external_torque_latest = np.zeros(7, dtype=np.float64)

        # Real franka teleop runtime (collect_data style)
        self.real_teleop_enabled: bool = False
        self.real_teleop_rate_hz: float = 30.0
        self.real_teleop_thread: Optional[threading.Thread] = None

        # Use cached leader state from 500Hz loop (avoid extra dynamixel reads in 30Hz thread)
        self._leader_state_lock = threading.Lock()
        self._leader_state_ready = False
        self._leader_arm_pos_latest = np.zeros(7, dtype=np.float64)
        self._leader_arm_vel_latest = np.zeros(7, dtype=np.float64)
        self._leader_gripper_pos_latest: float = 0.0
        self._leader_gripper_vel_latest: float = 0.0
        self._leader_state_timestamp: float = 0.0

        # Real teleop targets
        self.teleop_arm = None  # will point to self._franka_arm
        self.teleop_gripper = None  # optional separate interface
        self._owns_teleop_gripper = False

        try:
            self._load_config()
            self._setup_parameters()
            self._prepare_dynamixel()
            self._prepare_inverse_dynamics()
            self._calibrate_system()
        except Exception as e:
            # Cleanup on initialization failure
            self.shutdown()
            raise FactrError(f"Failed to initialize FACTR system: {e}") from e

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        with open(self.config_path, "r") as config_file:
            self.config = yaml.safe_load(config_file)
        print(f"Loaded config: {self.config['name']}")

    def _setup_parameters(self) -> None:
        """Initialize parameters from config."""
        self.dt = 1 / self.config["controller"]["frequency"]

        # Leader arm parameters
        self.num_arm_joints = self.config["arm_teleop"]["num_arm_joints"]
        self.configured_safety_margin = float(
            self.config["arm_teleop"]["arm_joint_limits_safety_margin"]
        )
        self.safety_margin = float(
            os.environ.get("FACTR_ARM_SOFT_LIMIT_MARGIN", "0.0")
        )
        self.arm_joint_limits_max = (
            np.array(self.config["arm_teleop"]["arm_joint_limits_max"])
            - self.safety_margin
        )
        self.arm_joint_limits_min = (
            np.array(self.config["arm_teleop"]["arm_joint_limits_min"])
            + self.safety_margin
        )
        self.arm_joint_limits_max_hard = np.array(
            self.config["arm_teleop"]["arm_joint_limits_max"], dtype=np.float64
        )
        self.arm_joint_limits_min_hard = np.array(
            self.config["arm_teleop"]["arm_joint_limits_min"], dtype=np.float64
        )
        self.hard_limit_tolerance = float(
            self.config["arm_teleop"].get("hard_limit_tolerance", 0.03)
        )
        self.calibration_joint_pos = np.array(
            self.config["arm_teleop"]["initialization"]["calibration_joint_pos"]
        )
        self.initial_match_joint_pos = np.array(
            self.config["arm_teleop"]["initialization"]["initial_match_joint_pos"]
        )

        # Gripper parameters
        self.gripper_limit_min = 0.0
        self.gripper_limit_max = self.config["gripper_teleop"]["actuation_range"]
        self.gripper_close_threshold = 0.5 * (
            self.gripper_limit_min + self.gripper_limit_max
        )
        self.gripper_pos_prev = 0.0
        self.gripper_pos = 0.0
        self.gripper_pos_unclipped_prev = 0.0
        self.gripper_pos_unclipped = 0.0

        # gravity comp
        self.enable_gravity_comp = self.config["controller"]["gravity_comp"]["enable"]
        self.gravity_comp_gain = np.array(
            self.config["controller"]["gravity_comp"]["gain"],
            dtype=np.float64,
        )
        self.gravity_comp_modifier = self.gravity_comp_gain.copy()
        self.tau_g = np.zeros(self.num_arm_joints)
        self.max_arm_torque = np.array(
            self.config["controller"].get("max_arm_torque", [0.15] * 7),
            dtype=np.float64,
        )

        # Friction compensation
        self.stiction_comp_enable_speed = self.config["controller"][
            "static_friction_comp"
        ]["enable_speed"]
        self.stiction_comp_gain = self.config["controller"]["static_friction_comp"][
            "gain"
        ]
        self.stiction_comp_tau = np.array(self.config["controller"]["static_friction_comp"][
            "tau"
        ])
        self.stiction_dither_flag = np.ones((self.num_arm_joints), dtype=bool)

        # Joint limit barrier
        self.joint_limit_kp = self.config["controller"]["joint_limit_barrier"]["kp"]
        self.joint_limit_kd = self.config["controller"]["joint_limit_barrier"]["kd"]

        # Null space regulation
        self.null_space_joint_target = np.array(
            self.config["controller"]["null_space_regulation"][
                "null_space_joint_target"
            ]
        )
        self.null_space_kp = self.config["controller"]["null_space_regulation"]["kp"]
        self.null_space_kd = self.config["controller"]["null_space_regulation"]["kd"]

        # torque feedback
        self.enable_torque_feedback = self.config["controller"]["torque_feedback"][
            "enable"
        ]
        self.torque_feedback_gain = self.config["controller"]["torque_feedback"]["gain"]
        self.torque_feedback_motor_scalar = self.config["controller"][
            "torque_feedback"
        ]["motor_scalar"]
        self.torque_feedback_damping = self.config["controller"]["torque_feedback"][
            "damping"
        ]
        # gripper feedback
        self.enable_gripper_feedback = self.config["controller"]["gripper_feedback"][
            "enable"
        ]

        print(f"Control frequency: {1 / self.dt:.1f} Hz")
        print(
            f"Gravity compensation: {'enabled' if self.enable_gravity_comp else 'disabled'}"
        )

    def _prepare_dynamixel(self) -> None:
        """Initialize Dynamixel servo driver."""
        self.servo_types = self.config["dynamixel"]["servo_types"]
        self.num_motors = len(self.servo_types)
        self.joint_signs = np.array(
            self.config["dynamixel"]["joint_signs"], dtype=float
        )
        configured_port = self.config["dynamixel"]["dynamixel_port"]
        selected_port = self.port_override or configured_port
        if selected_port.startswith("/"):
            self.dynamixel_port = selected_port
        elif selected_port.startswith("usb-"):
            self.dynamixel_port = "/dev/serial/by-id/" + selected_port
        else:
            self.dynamixel_port = selected_port

        # Check latency timer
        # checks of the latency timer on ttyUSB of the corresponding port is 1
        # if it is not 1, the control loop cannot run at above 200 Hz, which will
        # cause extremely undesirable behaviour for the leader arm. If the latency
        # timer is not 1, one can set it to 1 as follows:
        # echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB{NUM}/latency_timer
        try:
            port_name = os.path.basename(self.dynamixel_port)
            ttyUSBx = (
                port_name
                if port_name.startswith("ttyUSB")
                else find_ttyusb(port_name)
            )
            latency_path = f"/sys/bus/usb-serial/devices/{ttyUSBx}/latency_timer"
            result = subprocess.run(
                ["cat", latency_path], capture_output=True, text=True, check=True
            )
            ttyUSB_latency_timer = int(result.stdout)
            if ttyUSB_latency_timer != 1:
                print(
                    f"Warning: Latency timer of {ttyUSBx} is {ttyUSB_latency_timer}, should be 1 for optimal performance."
                )
                print(
                    f"Run: echo 1 | sudo tee /sys/bus/usb-serial/devices/{ttyUSBx}/latency_timer"
                )
        except (subprocess.CalledProcessError, FileNotFoundError, PermissionError) as e:
            print(f"Could not check latency timer (file access issue): {e}")
        except (ValueError, IndexError) as e:
            print(f"Could not parse latency timer value: {e}")
        except Exception as e:
            print(f"Unexpected error checking latency timer: {e}")

        # Initialize driver
        joint_ids = (np.arange(self.num_motors) + 1).tolist()
        try:
            self.driver = DynamixelDriver(
                joint_ids,
                self.servo_types,
                self.dynamixel_port,
            )
            print(f"Connected to Dynamixel servos on {self.dynamixel_port}")
        except Exception as e:
            raise FactrError(f"Failed to connect to Dynamixel servos: {e}") from e

        # Configure servos
        self.driver.set_torque_mode(False)
        if self.read_only:
            print("FACTR read-only mode: torque/current control is disabled.")
        elif self.enable_gravity_comp:
            # Use current control with torque enabled when GC is active
            self.driver.set_operating_mode(0)  # Current control mode
            self.driver.set_torque_mode(True)
            self._torque_enabled_by_us = True
        else:
            # # When GC is disabled, keep motors free/backdrivable similar to baseline
            # # Try to switch to position mode (not strictly necessary) and keep torque disabled
            # try:
            #     self.driver.set_operating_mode(3)  # Position control mode
            # except Exception:
            #     pass
            # self.driver.set_torque_mode(False)
            pass

    def _prepare_inverse_dynamics(self) -> None:
        """Initialize Pinocchio model for inverse dynamics."""
        # Construct URDF path - make GELLO completely self-contained
        urdf_filename = self.config["arm_teleop"]["leader_urdf"]
        urdf_path = REPO_ROOT / "factr" / "assets" / urdf_filename

        print(f"Loading URDF: {urdf_path}")
        # Only the dynamics model is needed for gravity compensation.
        self.pin_model = pin.buildModelFromUrdf(str(urdf_path))
        self.pin_data = self.pin_model.createData()

    def _calibrate_system(self) -> None:
        """Calibrate Dynamixel offsets and match initial position."""
        print("Calibrating Dynamixel offsets...")
        # self._get_dynamixel_offsets()
        # print(self.joint_offsets)
        # self.joint_offsets = np.array([0., 3.14159265, 6.28318531, 1.57079633 , 3.14159265, 6.28318531, 1.57079633, 0.0])
        # self.joint_offsets = np.array([1.57079633, 3.14159265, 0.,         1.57079633, 3.14159265, 6.28318531,
        #  1.57079633, 0.01840777])
        # self.joint_offsets = np.array([1.57079633, 3.14159265, 6.28318531, 0.78539816, 3.14159265, 6.28318531, 1.57079633, 0.02607767])
        configured_offsets = self.config["dynamixel"].get("joint_offsets")
        self.joint_offsets = np.array(
            configured_offsets
            if configured_offsets is not None
            else
            # [
            #     0.0,
            #     3.14159265,
            #     6.28318531,
            #     1.57079633,
            #     3.14159265,
            #     6.28318531,
            #     1.57079633 + 1.57079633 / 2.0,
            #     0.0,
            # ]
            [
                -math.pi / 2,
                -0.75 * math.pi,
                0.0,#3.141592654,
                0.375 * math.pi,
                0.0,
                -0.0 * math.pi,
                -0.25 * math.pi,
                0.5 * math.pi,
            ],
            dtype=np.float64,
        )
        if configured_offsets is not None:
            print("Using configured Dynamixel joint offsets from YAML.")
            print("System calibrated and ready!")
            return
        calibration_report = (
            Path(self.config_path).parent / "validation" / "calibration.json"
        )
        if calibration_report.is_file():
            try:
                report = json.loads(calibration_report.read_text(encoding="utf-8"))
                report_offsets = report.get("joint_offsets")
                if report.get("pass") and report_offsets is not None:
                    self.joint_offsets = np.asarray(report_offsets, dtype=np.float64)
                    if self.joint_offsets.shape == (self.num_motors,):
                        print(
                            "Using calibrated Dynamixel joint offsets from "
                            f"{calibration_report}."
                        )
                        print("System calibrated and ready!")
                        return
            except (OSError, ValueError, TypeError):
                pass

        # loop through all joints and add +- 2pi to the joint offsets to get the closest to start joints
        new_joint_offsets = []
        curr_joints, _ = self.driver.get_positions_and_velocities()
        start_joints = self.initial_joint_pos
        assert curr_joints.shape == start_joints.shape

        for idx, (c_joint, s_joint, offset, joint_sign) in enumerate(
            zip(curr_joints, start_joints, self.joint_offsets, self.joint_signs)
        ):
            best_bias = 0.0
            best_error = np.inf
            for bias in [0.0, -2 * np.pi, 2 * np.pi]:
                joint_i = joint_sign * (c_joint - offset - bias)
                error = abs(joint_i - s_joint)
                if error < best_error:
                    best_error = error
                    best_bias = bias
            print("-> ", best_bias)
            new_joint_offsets.append(offset + best_bias)

        self.joint_offsets = np.array(new_joint_offsets)

        print("Skipping initial position match...")
        print("System calibrated and ready!")

    # def _get_dynamixel_offsets(self, verbose: bool = True) -> None:
    #     """Calibrate Dynamixel servos to match expected joint positions."""
    #     # Warm up
    #     if self.driver is None:
    #         raise RuntimeError("Driver not initialized")
    #     for _ in range(10):
    #         self.driver.get_positions_and_velocities()

    #     def get_error(calibration_joint_pos, offset, index, joint_state):
    #         joint_sign_i = self.joint_signs[index]
    #         joint_i = joint_sign_i * (joint_state[index] - offset)
    #         start_i = calibration_joint_pos[index]
    #         return np.abs(joint_i - start_i)

    #     # Get arm offsets
    #     self.joint_offsets = []
    #     curr_joints, _ = self.driver.get_positions_and_velocities()
    #     for i in range(self.num_arm_joints):
    #         best_offset = 0
    #         best_error = 1e9
    #         # Search over intervals of pi/2
    #         for offset in np.linspace(
    #             -self.CALIBRATION_RANGE_MULTIPLIER * np.pi,
    #             self.CALIBRATION_RANGE_MULTIPLIER * np.pi,
    #             self.CALIBRATION_STEP_COUNT,
    #         ):
    #             error = get_error(self.calibration_joint_pos, offset, i, curr_joints)
    #             if error < best_error:
    #                 best_error = error
    #                 best_offset = offset
    #         self.joint_offsets.append(best_offset)

    #     # Get gripper offset
    #     curr_gripper_joint = curr_joints[-1]
    #     self.joint_offsets.append(curr_gripper_joint)
    #     self.joint_offsets = np.asarray(self.joint_offsets)
    #     # TODO dump these offsets to a file for future runs so we don't have to recalibrate every time

    #     if verbose:
    #         print(f"Joint offsets: {[f'{x:.3f}' for x in self.joint_offsets]}")

    def _get_dynamixel_offsets(self, verbose: bool = True) -> None:
        """Calibrate Dynamixel servos to match expected joint positions."""
        # Warm up
        if self.driver is None:
            raise RuntimeError("Driver not initialized")
        for _ in range(10):
            self.driver.get_positions_and_velocities()

        def get_error(calibration_joint_pos, offset, index, joint_state):
            joint_sign_i = self.joint_signs[index]
            joint_i = joint_sign_i * (joint_state[index] - offset)
            start_i = calibration_joint_pos[index]
            return np.abs(joint_i - start_i)

        # Get arm offsets
        # self.joint_offsets = []
        # curr_joints, _ = self.driver.get_positions_and_velocities()
        # for i in range(self.num_arm_joints):
        #     best_offset = 0
        #     best_error = 1e9
        #     # Search over intervals of pi/2
        #     for offset in np.linspace(
        #         -self.CALIBRATION_RANGE_MULTIPLIER * np.pi,
        #         self.CALIBRATION_RANGE_MULTIPLIER * np.pi,
        #         self.CALIBRATION_STEP_COUNT,
        #     ):
        #         error = get_error(self.calibration_joint_pos, offset, i, curr_joints)
        #         if error < best_error:
        #             best_error = error
        #             best_offset = offset
        #     self.joint_offsets.append(best_offset)

        # get arm offsets
        self.joint_offsets = []
        curr_joints, _ = self.driver.get_positions_and_velocities()
        for i in range(self.num_arm_joints):
            best_offset = 0
            best_error = 1e9
            # intervals of pi/2
            if i == 1 or i == 3:
                for offset in np.linspace(-20 * np.pi, 20 * np.pi, 20 * 8 + 1):
                    error = get_error(
                        self.calibration_joint_pos, offset, i, curr_joints
                    )
                    if error < best_error:
                        best_error = error
                        best_offset = offset
                if i == 1:
                    print(best_offset, best_error)
            else:
                for offset in np.linspace(-20 * np.pi, 20 * np.pi, 20 * 4 + 1):
                    error = get_error(
                        self.calibration_joint_pos, offset, i, curr_joints
                    )
                    if error < best_error:
                        best_error = error
                        best_offset = offset
            self.joint_offsets.append(best_offset)

        # Get gripper offset
        curr_gripper_joint = curr_joints[-1]
        self.joint_offsets.append(curr_gripper_joint)
        self.joint_offsets = np.asarray(self.joint_offsets)
        # TODO dump these offsets to a file for future runs so we don't have to recalibrate every time

        # if verbose:
        #     print(f"Joint offsets: {[f'{x:.3f}' for x in self.joint_offsets]}")

    def _match_start_pos(self) -> None:
        """Wait for leader arm to be moved to initial position."""
        while True:
            curr_pos, _, _, _ = self.get_leader_joint_states()
            current_joint_error = np.linalg.norm(
                curr_pos - self.initial_match_joint_pos[0 : self.num_arm_joints]
            )
            if current_joint_error <= 0.6:
                break
            print(
                f"Please match starting joint position. Current error: {current_joint_error:.3f}"
            )
            time.sleep(0.5)

    def get_leader_joint_states(
        self,
        *,
        clip_gripper: bool = True,
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float, float]:
        """Get current joint positions and velocities."""
        if self.driver is None:
            raise RuntimeError("Driver not initialized")
        self.gripper_pos_prev = self.gripper_pos
        self.gripper_pos_unclipped_prev = self.gripper_pos_unclipped
        joint_pos, joint_vel = self.driver.get_positions_and_velocities()

        # Apply offsets and signs for arm joints
        joint_pos_arm = (
            joint_pos[0 : self.num_arm_joints]
            - self.joint_offsets[0 : self.num_arm_joints]
        ) * self.joint_signs[0 : self.num_arm_joints]
        wrap_center = self.initial_match_joint_pos[0 : self.num_arm_joints]
        joint_pos_arm = wrap_center + (
            (joint_pos_arm - wrap_center + np.pi) % (2.0 * np.pi) - np.pi
        )
        joint_vel_arm = (
            joint_vel[0 : self.num_arm_joints]
            * self.joint_signs[0 : self.num_arm_joints]
        )

        # Process gripper
        self.leader_gripper_raw_rad = float(joint_pos[-1])
        calibrated_gripper_pos = (joint_pos[-1] - self.joint_offsets[-1]) * self.joint_signs[
            -1
        ]
        self.gripper_pos_unclipped = float(calibrated_gripper_pos)
        self.gripper_pos = float(
            np.clip(
                calibrated_gripper_pos,
                self.gripper_limit_min,
                self.gripper_limit_max,
            )
        )
        if clip_gripper:
            gripper_pos = self.gripper_pos
            gripper_vel = (self.gripper_pos - self.gripper_pos_prev) / self.dt
        else:
            gripper_pos = self.gripper_pos_unclipped
            gripper_vel = (
                self.gripper_pos_unclipped - self.gripper_pos_unclipped_prev
            ) / self.dt

        return joint_pos_arm, joint_vel_arm, gripper_pos, gripper_vel

    def read_state(self) -> FactrState:
        """Read one synchronous FACTR state and cache it for teleop loops."""
        joint_pos, joint_vel, gripper_pos, gripper_vel = self.get_leader_joint_states()
        state = FactrState(
            joint_positions=np.asarray(joint_pos, dtype=np.float64).copy(),
            joint_velocities=np.asarray(joint_vel, dtype=np.float64).copy(),
            gripper_position=float(gripper_pos),
            gripper_velocity=float(gripper_vel),
            timestamp=time.monotonic(),
        )
        with self._leader_state_lock:
            self._leader_arm_pos_latest[:] = state.joint_positions
            self._leader_arm_vel_latest[:] = state.joint_velocities
            self._leader_gripper_pos_latest = state.gripper_position
            self._leader_gripper_vel_latest = state.gripper_velocity
            self._leader_state_timestamp = state.timestamp
            self._leader_state_ready = True
        return state

    def latest_state(self, max_age: float | None = None) -> FactrState | None:
        """Return the latest controller-cached state if present and fresh enough."""
        with self._leader_state_lock:
            if not self._leader_state_ready:
                return None
            state = FactrState(
                joint_positions=self._leader_arm_pos_latest.copy(),
                joint_velocities=self._leader_arm_vel_latest.copy(),
                gripper_position=float(self._leader_gripper_pos_latest),
                gripper_velocity=float(self._leader_gripper_vel_latest),
                timestamp=float(self._leader_state_timestamp),
            )
        if max_age is not None and time.monotonic() - state.timestamp > max_age:
            return None
        return state

    def validate_arm_positions(
        self,
        joint_positions: npt.NDArray[np.float64],
        *,
        safety_limits: bool = True,
    ) -> None:
        q = np.asarray(joint_positions, dtype=np.float64).reshape(self.num_arm_joints)
        if safety_limits:
            lower = self.arm_joint_limits_min
            upper = self.arm_joint_limits_max
            label = "soft"
        else:
            lower = self.arm_joint_limits_min_hard - self.hard_limit_tolerance
            upper = self.arm_joint_limits_max_hard + self.hard_limit_tolerance
            label = "hard"
        outside = np.flatnonzero((q < lower) | (q > upper))
        if outside.size:
            details = ", ".join(f"J{i + 1}={q[i]:.3f}" for i in outside)
            raise FactrSafetyError(
                f"FACTR arm position outside configured {label} limits: {details}"
            )

    def validate_gripper_position(self, gripper_position: float) -> None:
        value = float(gripper_position)
        if value < self.gripper_limit_min or value > self.gripper_limit_max:
            raise FactrSafetyError(
                "FACTR gripper position outside configured range: "
                f"{value:.3f} rad not in "
                f"[{self.gripper_limit_min:.3f}, {self.gripper_limit_max:.3f}]"
            )

    def get_end_effector_pose(
        self,
        joint_positions: npt.NDArray[np.float64],
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        q = np.asarray(joint_positions, dtype=np.float64).reshape(self.num_arm_joints)
        pin.forwardKinematics(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        frame_id = self.pin_model.getFrameId("link_7")
        if frame_id >= len(self.pin_model.frames):
            frame_id = len(self.pin_model.frames) - 1
        placement = self.pin_data.oMf[frame_id]
        return np.asarray(placement.translation).copy(), np.asarray(placement.rotation).copy()

    def set_leader_joint_torque(
        self, arm_torque: npt.NDArray[np.float64], gripper_torque: float
    ) -> None:
        """Apply torque to leader arm and gripper."""
        if self.driver is None:
            raise RuntimeError("Driver not initialized")
        arm_gripper_torque = np.append(arm_torque, gripper_torque)
        self.driver.set_torque((arm_gripper_torque * self.joint_signs).tolist())

    def joint_limit_barrier(
        self,
        arm_joint_pos: npt.NDArray[np.float64],
        arm_joint_vel: npt.NDArray[np.float64],
        gripper_joint_pos: float,
        gripper_joint_vel: float,
    ) -> Tuple[npt.NDArray[np.float64], float]:
        """Compute joint limit repulsive torques."""
        # Arm joint limits
        exceed_max_mask = arm_joint_pos > self.arm_joint_limits_max
        tau_l = (
            -self.joint_limit_kp * (arm_joint_pos - self.arm_joint_limits_max)
            - self.joint_limit_kd * arm_joint_vel
        ) * exceed_max_mask

        exceed_min_mask = arm_joint_pos < self.arm_joint_limits_min
        tau_l += (
            -self.joint_limit_kp * (arm_joint_pos - self.arm_joint_limits_min)
            - self.joint_limit_kd * arm_joint_vel
        ) * exceed_min_mask

        # Gripper limits
        if gripper_joint_pos > self.gripper_limit_max:
            tau_l_gripper = (
                -self.joint_limit_kp * (gripper_joint_pos - self.gripper_limit_max)
                - self.joint_limit_kd * gripper_joint_vel
            )
        elif gripper_joint_pos < self.gripper_limit_min:
            tau_l_gripper = (
                -self.joint_limit_kp * (gripper_joint_pos - self.gripper_limit_min)
                - self.joint_limit_kd * gripper_joint_vel
            )
        else:
            tau_l_gripper = 0.0

        return tau_l, tau_l_gripper

    def gravity_compensation(
        self,
        arm_joint_pos: npt.NDArray[np.float64],
        arm_joint_vel: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute gravity compensation torques using inverse dynamics."""
        self.tau_g = self.gravity_compensation_raw(arm_joint_pos, arm_joint_vel)
        self.tau_g *= self.gravity_comp_modifier
        return self.tau_g

    def gravity_compensation_raw(
        self,
        arm_joint_pos: npt.NDArray[np.float64],
        arm_joint_vel: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute unscaled generalized gravity for the FACTR arm."""
        del arm_joint_vel
        return pin.computeGeneralizedGravity(  # type: ignore[attr-defined]
            self.pin_model,
            self.pin_data,
            arm_joint_pos,
        )

    def enforce_joint_limit_direction(
        self,
        arm_joint_pos: npt.NDArray[np.float64],
        arm_torque: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Zero any torque that would push a joint farther beyond a soft limit."""
        q = np.asarray(arm_joint_pos, dtype=np.float64)
        tau = np.asarray(arm_torque, dtype=np.float64).copy()
        tau[(q > self.arm_joint_limits_max) & (tau > 0.0)] = 0.0
        tau[(q < self.arm_joint_limits_min) & (tau < 0.0)] = 0.0
        return tau

    def friction_compensation(
        self, arm_joint_vel: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Compute static friction compensation torques."""
        tau_ss = np.zeros(self.num_arm_joints)
        for i in range(self.num_arm_joints):
            if abs(arm_joint_vel[i]) < self.stiction_comp_enable_speed:
                if self.stiction_dither_flag[i]:
                    tau_ss[i] += self.stiction_comp_gain * self.stiction_comp_tau[i]
                    # tau_ss[i] += self.stiction_comp_gain * np.abs(self.tau_g[i])
                else:
                    tau_ss[i] -= self.stiction_comp_gain * self.stiction_comp_tau[i]
                    # tau_ss[i] -= self.stiction_comp_gain * np.abs(self.tau_g[i])
                self.stiction_dither_flag[i] = ~self.stiction_dither_flag[i]
        return tau_ss

    def null_space_regulation(
        self,
        arm_joint_pos: npt.NDArray[np.float64],
        arm_joint_vel: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Compute null-space regulation torques."""
        J = pin.computeJointJacobian(
            self.pin_model, self.pin_data, arm_joint_pos, self.num_arm_joints
        )  # type: ignore[attr-defined]
        J_dagger = np.linalg.pinv(J)
        null_space_projector = np.eye(self.num_arm_joints) - J_dagger @ J
        q_error = arm_joint_pos - self.null_space_joint_target[0 : self.num_arm_joints]
        tau_n = null_space_projector @ (
            -self.null_space_kp * q_error - self.null_space_kd * arm_joint_vel
        )
        return tau_n

    def shutdown(self) -> None:
        """Safely shutdown the system."""
        self.running = False

        if hasattr(self, "driver") and self.driver is not None:
            print("Disabling motor torques...")
            try:
                self.set_leader_joint_torque(np.zeros(self.num_arm_joints), 0.0)
            except Exception:
                pass
            self.driver.set_torque_mode(False)
            self.driver.close()

        if self._franka_query_executor is not None:
            try:
                if (
                    self._franka_query_future is not None
                    and not self._franka_query_future.done()
                ):
                    self._franka_query_future.cancel()
                self._franka_query_executor.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
            self._franka_query_executor = None
            self._franka_query_future = None

        if self.ctrl_thread is not None and self.ctrl_thread.is_alive():
            self.ctrl_thread.join(timeout=1.0)

        log.info("Shutdown complete")

    def control_loop_step(self) -> None:
        """Execute one step of the control loop."""
        # start = time.time()
        # Get current joint states
        leader_arm_pos, leader_arm_vel, leader_gripper_pos, leader_gripper_vel = (
            self.get_leader_joint_states()
        )
        # print(f"read time: {(time.time()-start)*1000:.2f} ms")

        # added for teleop
        with self._leader_state_lock:
            self._leader_arm_pos_latest[:] = leader_arm_pos
            self._leader_arm_vel_latest[:] = leader_arm_vel
            self._leader_gripper_pos_latest = float(leader_gripper_pos)
            self._leader_gripper_vel_latest = float(leader_gripper_vel)
            self._leader_state_timestamp = time.monotonic()
            self._leader_state_ready = True

        # start=time.time()
        # Initialize torque commands
        torque_arm = np.zeros(self.num_arm_joints)

        # Joint limit barriers
        torque_l, torque_gripper = self.joint_limit_barrier(
            leader_arm_pos, leader_arm_vel, leader_gripper_pos, leader_gripper_vel
        )
        torque_arm += torque_l

        # Null space regulation
        torque_arm += self.null_space_regulation(leader_arm_pos, leader_arm_vel)

        # Gravity compensation and friction compensation
        if self.enable_gravity_comp:
            torque_arm += self.gravity_compensation(leader_arm_pos, leader_arm_vel)
            torque_arm += self.friction_compensation(leader_arm_vel)

        if self.enable_torque_feedback:
            external_joint_torque = self.get_leader_arm_external_joint_torque()
            torque_arm += self.torque_feedback(external_joint_torque, leader_arm_vel)

        if self.enable_gripper_feedback:
            external_gripper_torque = self.get_leader_gripper_external_torque()
            torque_gripper += self.gripper_feedback(
                external_gripper_torque, leader_gripper_vel
            )

        # print(f"computation time: {(time.time()-start)*1000:.2f} ms")

        # start = time.time()
        # Apply torques only if GC is enabled (torque mode is off otherwise)
        if self.enable_gravity_comp:
            self.set_leader_joint_torque(torque_arm, torque_gripper)
        # print(f"write time: {(time.time()-start)*1000:.2f} ms")

    def torque_feedback(self, external_torque, arm_joint_vel):
        """
        Computes joint torque for the leader arm to achieve force-feedback based on
        the external joint torque from the follower arm.

        This method implements Equation 1 in Section III.A of the paper.
        """
        tau_ff = (
            -1.0
            * self.torque_feedback_gain
            / self.torque_feedback_motor_scalar
            * external_torque
        )
        tau_ff -= self.torque_feedback_damping * arm_joint_vel
        return tau_ff

    def gripper_feedback(
        self, leader_gripper_pos, leader_gripper_vel, gripper_feedback
    ):
        """
        Processes feedback data from the follower gripper. This method is intended to compute
        force-feedback for the leader gripper. This method is called at every iteration of the
        control loop if self.enable_gripper_feedback is set to True.

        Args:
            leader_gripper_pos (float): Leader gripper position. Can be used to provide force-
            feedback for the gripper.
            leader_gripper_vel (float): Leader gripper velocity. Can be used to provide force-
            feedback for the gripper.
            gripper_feedback (Any): Feedback data from the gripper. The format can vary depending
            on the implementation, such as a NumPy array, scalar, or custom object.

        Returns:
            float: The computed joint torque value to apply force-feedback to the leader gripper.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.
        """
        pass

    def get_leader_arm_external_joint_torque(self):
        # No interface yet: return latest cached value
        if self._franka_query_executor is None:
            with self._ext_tau_lock:
                return self._external_torque_latest.copy()

        # Step 1: ensure one query is in flight (start if none)
        if self._franka_query_future is None:
            self._franka_query_future = self._franka_query_executor.submit(
                self._query_external_torque_once
            )

        fut = self._franka_query_future

        # Step 2: wait up to 0.5 ms for current query to finish
        try:
            tau = fut.result(timeout=self._franka_query_timeout_s)
            # Finished within timeout -> update one-slot buffer
            with self._ext_tau_lock:
                self._external_torque_latest = tau
            # Immediately launch next query for pipelining
            self._franka_query_future = self._franka_query_executor.submit(
                self._query_external_torque_once
            )
        except TimeoutError:
            # Not finished within 0.5 ms -> keep running in background
            pass
        except Exception as e:
            print(f"[franka_feedback] query failed: {e}")
            # Drop failed future; next cycle will start a new one
            self._franka_query_future = None

        # Return latest available buffered value
        with self._ext_tau_lock:
            return self._external_torque_latest.copy()

    def _query_external_torque_once(self) -> np.ndarray:
        state = self._panda_env.arm.get_robot_state()
        assert state["motor_torques_external"].shape == (7,)

        tau = np.asarray(state["motor_torques_external"], dtype=np.float64)

        out = np.zeros(self.num_arm_joints, dtype=np.float64)
        n = min(len(tau), self.num_arm_joints)
        out[:n] = tau[:n]
        return out

    def get_leader_gripper_external_torque(self):
        pass

    def start(self, env: Any = None) -> None:
        # configure franka arm state thread
        self._panda_env = env
        if env is not None:
            self._franka_query_executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="franka_state"
            )
            self._franka_query_future = self._franka_query_executor.submit(
                self._query_external_torque_once
            )

        self.running = True

        def control_loop():
            while self.running:
                start_time = time.time()

                self.control_loop_step()

                # Maintain loop timing
                elapsed = time.time() - start_time
                sleep_time = max(0, self.dt - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)

        assert self.ctrl_thread is None
        self.ctrl_thread = threading.Thread(
            target=control_loop, daemon=True, name="factr_control"
        )
        self.ctrl_thread.start()

    def get_state(self):
        with self._leader_state_lock:
            if not self._leader_state_ready:
                return None

            leader_q = self._leader_arm_pos_latest.copy()
            leader_qd = self._leader_arm_vel_latest.copy()
            leader_grip = self._leader_gripper_pos_latest

        # TODO make gripper binarization optional
        if self.gripper_limit_max is not None:
            thresh = 0.5 * self.gripper_limit_max
            leader_grip = np.array(
                [-1.0 if leader_grip < thresh else 1.0], dtype=np.float32
            )

        return leader_q, leader_qd, leader_grip


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Standalone FACTR Gravity Compensation"
    )
    parser.add_argument(
        "--config",
        "-c",
        default=str(__import__("pathlib").Path(__file__).resolve().parents[1] / "leader.yaml"),
        help="Path to configuration YAML file",
    )

    args = parser.parse_args()

    # Verify config file exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return 1

    try:
        # Create and run gravity compensation system
        system = FACTRGravityCompensation(args.config)

        # Set up signal handler for clean shutdown
        def signal_handler(signum, frame):
            print("\nReceived shutdown signal")
            system.running = False

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Run the system
        system.run()

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0
