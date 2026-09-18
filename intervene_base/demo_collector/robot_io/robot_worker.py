import threading
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .franka_interface import FrankaInterface


@dataclass
class RobotSharedState:
    connected: bool = False
    connect_error: Optional[str] = None

    q: Optional[np.ndarray] = None
    dq: Optional[np.ndarray] = None
    grip: Optional[float] = None

    last_update_time: float = 0.0
    busy: bool = False

    move_seq: int = 0
    last_completed_move_seq: int = 0
    last_completed_move_time: float = 0.0


class RobotWorker:
    def __init__(self, robot_key: str, poll_hz: float = 100.0):
        self.robot = FrankaInterface(robot_key)
        self.poll_hz = float(poll_hz)

        self.state = RobotSharedState()
        self.lock = threading.Lock()

        self._thread = None
        self._stop_event = threading.Event()

        self._move_request = None
        self._move_lock = threading.Lock()
        self._next_move_seq = 1

    def start(self):
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def request_go_to_joint_and_gripper(self, q_target: np.ndarray, grip_target: float) -> int:
        with self._move_lock:
            move_seq = self._next_move_seq
            self._next_move_seq += 1
            self._move_request = {
                "seq": move_seq,
                "q_target": np.asarray(q_target, dtype=np.float64).copy(),
                "grip_target": float(grip_target),
            }
            return move_seq

    def _pop_move_request(self):
        with self._move_lock:
            req = self._move_request
            self._move_request = None
            return req

    def _set_busy(self, value: bool):
        with self.lock:
            self.state.busy = value

    def _mark_move_completed(self, move_seq: int):
        with self.lock:
            self.state.last_completed_move_seq = move_seq
            self.state.last_completed_move_time = time.time()

    def _run(self):
        try:
            self.robot.connect_human_control()
            with self.lock:
                self.state.connected = True
                self.state.connect_error = None
        except Exception as e:
            with self.lock:
                self.state.connected = False
                self.state.connect_error = str(e)
            return

        dt = 1.0 / self.poll_hz

        while not self._stop_event.is_set():
            try:
                move_req = self._pop_move_request()
                if move_req is not None:
                    move_seq = move_req["seq"]
                    q_target = move_req["q_target"]
                    grip_target = move_req["grip_target"]

                    with self.lock:
                        self.state.move_seq = move_seq

                    self._set_busy(True)
                    self.robot.go_to_joint_and_gripper_safe(q_target, grip_target)
                    self._set_busy(False)
                    self._mark_move_completed(move_seq)

                q, dq, grip = self.robot.get_state()
                with self.lock:
                    self.state.q = q.copy()
                    self.state.dq = None if dq is None else dq.copy()
                    self.state.grip = float(grip)
                    self.state.last_update_time = time.time()
                    self.state.connect_error = None

            except Exception as e:
                with self.lock:
                    self.state.connect_error = str(e)
                self._set_busy(False)

            time.sleep(dt)

    def get_latest_state(self):
        with self.lock:
            return {
                "connected": self.state.connected,
                "connect_error": self.state.connect_error,
                "q": None if self.state.q is None else self.state.q.copy(),
                "dq": None if self.state.dq is None else self.state.dq.copy(),
                "grip": self.state.grip,
                "last_update_time": self.state.last_update_time,
                "busy": self.state.busy,
                "move_seq": self.state.move_seq,
                "last_completed_move_seq": self.state.last_completed_move_seq,
                "last_completed_move_time": self.state.last_completed_move_time,
            }

    def close(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        try:
            self.robot.close()
        except Exception:
            pass