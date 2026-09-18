import io
import os
import time
from typing import Dict, Generator

import grpc
import numpy as np
import torch

from polymetis_pb2 import ControllerChunk, Empty, GripperCommand
from polymetis_pb2_grpc import GripperServerStub, PolymetisControllerServerStub


EMPTY = Empty()
MAX_BYTES_PER_MSG = 1024

ROBOTS = {
    "p1": ("192.168.0.150", 1234, 1235),
    "p2": ("192.168.0.176", 4321, 4322),
    "p3": ("192.0.2.153", 50051, 50052),
    "p4": ("192.0.2.153", 50053, 50054),
}


class ParamDictContainer(torch.nn.Module):
    param_dict: Dict[str, torch.Tensor]

    def __init__(self, param_dict: Dict[str, torch.Tensor]):
        super().__init__()
        self.param_dict = param_dict

    def forward(self) -> Dict[str, torch.Tensor]:
        return self.param_dict


class JointPDPolicy(torch.nn.Module):
    _terminated: bool

    def __init__(self, q_current: torch.Tensor, kp: torch.Tensor, kd: torch.Tensor):
        super().__init__()
        self._terminated = False
        self.q_desired = torch.nn.Parameter(q_current.clone())
        self.qd_desired = torch.nn.Parameter(torch.zeros_like(q_current))
        self.kp = torch.nn.Parameter(kp.clone())
        self.kd = torch.nn.Parameter(kd.clone())

    @torch.jit.export
    def update(self, update_dict: Dict[str, torch.Tensor]) -> None:
        if "q_desired" in update_dict:
            self.q_desired.data.copy_(update_dict["q_desired"])
        if "joint_pos_desired" in update_dict:
            self.q_desired.data.copy_(update_dict["joint_pos_desired"])
        if "qd_desired" in update_dict:
            self.qd_desired.data.copy_(update_dict["qd_desired"])
        if "joint_vel_desired" in update_dict:
            self.qd_desired.data.copy_(update_dict["joint_vel_desired"])
        if "kp" in update_dict:
            self.kp.data.copy_(update_dict["kp"])
        if "kd" in update_dict:
            self.kd.data.copy_(update_dict["kd"])

    @torch.jit.export
    def is_terminated(self) -> bool:
        return self._terminated

    @torch.jit.export
    def reset(self) -> None:
        self._terminated = False

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        q = state_dict["joint_positions"]
        dq = state_dict["joint_velocities"]
        torque = self.kp * (self.q_desired - q) + self.kd * (self.qd_desired - dq)
        return {"joint_torques": torque}


class HumanControlPolicy(torch.nn.Module):
    _terminated: bool

    def __init__(self):
        super().__init__()
        self._terminated = False
        self.gain = torch.nn.Parameter(
            torch.tensor([0.3, 0.12, 0.40, 1.11, 1.10, 0.6, 0.85], dtype=torch.float32)
        )

    @torch.jit.export
    def update(self, update_dict: Dict[str, torch.Tensor]) -> None:
        if "gain" in update_dict:
            self.gain.data.copy_(update_dict["gain"])

    @torch.jit.export
    def is_terminated(self) -> bool:
        return self._terminated

    @torch.jit.export
    def reset(self) -> None:
        self._terminated = False

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ext = state_dict["motor_torques_external"]
        return {"joint_torques": -self.gain * ext}


def _chunks(scripted_module: torch.nn.Module) -> Generator[ControllerChunk, None, None]:
    buffer = io.BytesIO()
    torch.jit.save(scripted_module, buffer)
    buffer.seek(0)
    while True:
        chunk = buffer.read(MAX_BYTES_PER_MSG)
        if not chunk:
            break
        yield ControllerChunk(torchscript_binary_chunk=chunk)


class LightPolymetisRobotAdapter:
    def __init__(
        self,
        robot_key: str = "p4",
        control_mode=None,
        gripper_speed: float = 0.35,
        gripper_force: float = 0.10,
    ):
        self.robot_key = robot_key
        self.control_mode = control_mode
        self.gripper_speed = gripper_speed
        self.gripper_force = gripper_force

        self.arm_channel = None
        self.gripper_channel = None
        self.arm = None
        self.gripper = None
        self.metadata = None
        self.gripper_metadata = None
        self.connected = False

    @staticmethod
    def _is_no_controller_running(exc):
        try:
            details = exc.details()
        except Exception:
            details = str(exc)
        details = str(details).lower()
        return "no controller running" in details or "perform a controller update" in details

    def _target(self):
        try:
            return ROBOTS[self.robot_key]
        except KeyError as exc:
            available = ", ".join(sorted(ROBOTS))
            raise KeyError(f"Unknown robot key {self.robot_key!r}. Available: {available}") from exc

    def _tensor(self, values):
        return torch.tensor(list(values), dtype=torch.float32)

    def _joint_speed_limit(self):
        return float(
            os.environ.get(
                "INTERVENE_LIGHT_GOTO_MAX_DQ_RAD_S",
                os.environ.get(
                    "INTERVENE_MIRROR_MAX_DQ_RAD_S",
                    os.environ.get("INTERVENE_MIRROR_MAX_DQ", "0.80"),
                ),
            )
        )

    def _send_torch_policy(self, policy: torch.nn.Module):
        scripted = torch.jit.script(policy)
        self.arm.SetController(_chunks(scripted))

    def _update_current_policy(self, params: Dict[str, torch.Tensor]):
        scripted = torch.jit.script(ParamDictContainer(params))
        self.arm.UpdateController(_chunks(scripted))

    def _default_gains(self, q_current: torch.Tensor):
        if self.metadata is not None and len(self.metadata.default_Kq) == len(q_current):
            kp = self._tensor(self.metadata.default_Kq)
            kd = self._tensor(self.metadata.default_Kqd)
        else:
            kp = torch.tensor([80.0, 80.0, 80.0, 60.0, 35.0, 25.0, 15.0], dtype=torch.float32)
            kd = torch.tensor([8.0, 8.0, 8.0, 6.0, 4.0, 3.0, 2.0], dtype=torch.float32)

        kp *= float(os.environ.get("INTERVENE_LIGHT_KP_SCALE", "0.5"))
        kd *= float(os.environ.get("INTERVENE_LIGHT_KD_SCALE", "0.5"))
        return kp, kd

    def _start_joint_pd(self):
        q_current = self._tensor(self.arm.GetRobotState(EMPTY).joint_positions)
        kp, kd = self._default_gains(q_current)
        self._send_torch_policy(JointPDPolicy(q_current, kp, kd))

    def connect(self, already_connected=False):
        if already_connected and self.connected:
            print(f"[ROBOT-LIGHT] Reusing robot '{self.robot_key}'.")
            return

        ip, arm_port, gripper_port = self._target()
        print(
            f"[ROBOT-LIGHT] Connecting robot '{self.robot_key}' "
            f"arm={ip}:{arm_port} gripper={ip}:{gripper_port}..."
        )
        self.arm_channel = grpc.insecure_channel(f"{ip}:{arm_port}")
        self.gripper_channel = grpc.insecure_channel(f"{ip}:{gripper_port}")
        self.arm = PolymetisControllerServerStub(self.arm_channel)
        self.gripper = GripperServerStub(self.gripper_channel)
        self.metadata = self.arm.GetRobotClientMetadata(EMPTY)
        try:
            self.gripper_metadata = self.gripper.GetRobotClientMetadata(EMPTY)
        except grpc.RpcError:
            self.gripper_metadata = None

        if str(self.control_mode) == "HUMAN_CONTROL":
            self._send_torch_policy(HumanControlPolicy())
        else:
            self._start_joint_pd()
        self.connected = True

    def close(self):
        if self.arm is not None:
            try:
                self.arm.TerminateController(EMPTY)
            except Exception:
                pass
        if self.arm_channel is not None:
            self.arm_channel.close()
        if self.gripper_channel is not None:
            self.gripper_channel.close()
        self.connected = False
        self.arm = None
        self.gripper = None

    def get_joint_state(self):
        state = self.arm.GetRobotState(EMPTY)
        q = np.asarray(state.joint_positions, dtype=np.float64)
        dq = np.asarray(state.joint_velocities, dtype=np.float64)
        return q, dq

    def get_joint_positions(self):
        q, _ = self.get_joint_state()
        return q

    def send_joint_positions(self, q_cmd, qd_cmd=None):
        q_cmd = self._tensor(np.asarray(q_cmd, dtype=np.float64))
        if qd_cmd is None:
            qd_cmd = torch.zeros_like(q_cmd)
        else:
            qd_cmd = self._tensor(np.asarray(qd_cmd, dtype=np.float64))
        params = {
            "q_desired": q_cmd,
            "qd_desired": qd_cmd,
        }
        try:
            self._update_current_policy(params)
        except grpc.RpcError as exc:
            if not self._is_no_controller_running(exc):
                raise
            print("[ROBOT-LIGHT] Joint controller stopped; restarting and retrying once.")
            self._start_joint_pd()
            self._update_current_policy(params)

    def go_to_joint_positions(self, q_cmd, max_vel_norm_factor=None, duration_s=None):
        q_goal = np.asarray(q_cmd, dtype=np.float64)
        hz = float(os.environ.get("INTERVENE_LIGHT_GOTO_HZ", "60.0"))
        dt = 1.0 / hz
        max_dq = self._joint_speed_limit()

        q_start = self.get_joint_positions()
        max_err = float(np.max(np.abs(q_goal - q_start)))
        if duration_s is None:
            duration_s = float(os.environ.get("INTERVENE_LIGHT_GOTO_SECONDS", "0.0"))
        duration_s = float(duration_s)
        if duration_s > 0.0 and max_err > 0.0:
            max_dq = min(max_dq, max_err / duration_s) if max_dq > 0.0 else max_err / duration_s
        if max_dq <= 0.0:
            self.send_joint_positions(q_goal)
            return

        expected = max_err / max_dq
        timeout = float(os.environ.get("INTERVENE_LIGHT_GOTO_TIMEOUT_S", str(max(5.0, expected + 3.0))))
        tol = float(os.environ.get("INTERVENE_LIGHT_GOTO_TOL_RAD", "0.02"))
        print(
            f"[ROBOT-LIGHT] Moving to target slowly: max_dq={max_dq:.2f} rad/s, "
            f"max_error={max_err:.3f} rad, expected={expected:.1f}s"
        )

        t_end = time.time() + timeout
        max_step = max_dq * dt
        final_err = max_err
        while time.time() < t_end:
            q_now = self.get_joint_positions()
            err = q_goal - q_now
            final_err = float(np.max(np.abs(err)))
            if final_err <= tol:
                print(f"[ROBOT-LIGHT] Slow move reached target (err={final_err:.3f} rad).")
                return
            q_next = q_now + np.clip(err, -max_step, max_step)
            self.send_joint_positions(q_next)
            time.sleep(dt)

        print(f"[ROBOT-LIGHT] Slow move timeout; continuing with err={final_err:.3f} rad.")

    def get_gripper_width(self):
        return float(self.gripper.GetState(EMPTY).width)

    def send_gripper_width(self, width):
        cmd = GripperCommand(
            width=float(width),
            speed=float(self.gripper_speed),
            force=float(self.gripper_force),
            grasp=False,
        )
        cmd.timestamp.GetCurrentTime()
        self.gripper.Goto(cmd)

    def restore_gripper_width(self, target_width, tol=0.002, timeout=1.5, hz=40.0):
        target_width = float(target_width)
        t_end = time.time() + float(timeout)
        dt = 1.0 / float(hz)
        final = self.get_gripper_width()
        while time.time() < t_end:
            final = self.get_gripper_width()
            if abs(target_width - final) <= float(tol):
                break
            self.send_gripper_width(target_width)
            time.sleep(dt)
        print(f"[ROBOT-LIGHT] Gripper restored: target={target_width:.4f}, final={final:.4f}")

    def switch_control_mode(self, control_mode):
        if str(control_mode) == "HUMAN_CONTROL":
            self._send_torch_policy(HumanControlPolicy())
        else:
            self._start_joint_pd()
