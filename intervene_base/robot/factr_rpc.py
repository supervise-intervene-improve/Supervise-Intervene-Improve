"""Local RPC boundary between policy processes and the single FACTR device."""

from __future__ import annotations

import json
import os
import socket
import socketserver
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np


MAX_MESSAGE_BYTES = 1024 * 1024


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return bool(default)
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


class FactrRpcError(RuntimeError):
    pass


class FactrRpcAdapter:
    """Robot-adapter-compatible client for the one FACTR hardware service."""

    control_label = "FACTR"
    is_factr_adapter = True
    requires_strict_alignment = True

    def __init__(self, robot_key: str = "factr", control_mode=None):
        del control_mode
        self.robot_key = robot_key
        self.robot = None
        self.host = os.environ.get("FACTR_SERVICE_HOST", "127.0.0.1")
        self.port = int(os.environ.get("FACTR_SERVICE_PORT", "18075"))
        self.client_id = os.environ.get(
            "INTERVENE_FACTR_CLIENT_ID",
            f"policy-{os.getpid()}-{uuid.uuid4().hex[:8]}",
        )
        self.connected = False

    def _request(
        self,
        operation: str,
        *,
        timeout: float | None = None,
        **payload: Any,
    ) -> dict[str, Any]:
        if timeout is None:
            timeout = _env_float("FACTR_RPC_TIMEOUT", 5.0)
        message = {
            "operation": operation,
            "client_id": self.client_id,
            **payload,
        }
        raw = (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")
        try:
            with socket.create_connection((self.host, self.port), timeout=timeout) as sock:
                sock.settimeout(timeout)
                sock.sendall(raw)
                received = bytearray()
                while b"\n" not in received:
                    chunk = sock.recv(65536)
                    if not chunk:
                        break
                    received.extend(chunk)
                    if len(received) > MAX_MESSAGE_BYTES:
                        raise FactrRpcError("FACTR RPC response exceeded size limit")
        except (OSError, TimeoutError) as exc:
            raise FactrRpcError(
                f"FACTR service request {operation!r} failed at "
                f"{self.host}:{self.port}: {exc}"
            ) from exc

        if not received:
            raise FactrRpcError(f"FACTR service returned no response for {operation!r}")
        try:
            response = json.loads(bytes(received).split(b"\n", 1)[0].decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FactrRpcError(f"Malformed FACTR service response: {exc}") from exc
        if not response.get("ok", False):
            error_type = response.get("error_type", "FactrRpcError")
            error = response.get("error", "unknown FACTR service error")
            raise FactrRpcError(f"{error_type}: {error}")
        return response

    def connect(self, already_connected=False):
        if self.connected and already_connected:
            return
        if self.connected:
            return
        self._request("claim")
        self.connected = True
        print(f"[FACTR] Policy client {self.client_id} acquired the FACTR service.")

    def close(self):
        if not self.connected:
            return
        try:
            self._request("release")
        finally:
            self.connected = False

    @staticmethod
    def _validate_joint_vector(q, *, direction: str) -> np.ndarray:
        q_array = np.asarray(q, dtype=np.float64)
        if q_array.shape != (7,):
            raise ValueError(
                f"{direction} joint dimension mismatch: expected 7, got {q_array.shape}"
            )
        return q_array

    def align_to_mujoco(
        self,
        q_mujoco,
        *,
        gripper_width: float | None = None,
        tolerance: float | None = None,
        timeout: float | None = None,
        max_velocity: float | None = None,
    ) -> bool:
        q = self._validate_joint_vector(
            q_mujoco,
            direction="MuJoCo/FACTR",
        )
        alignment_timeout = (
            _env_float("FACTR_ALIGNMENT_TIMEOUT", 18.0)
            if timeout is None
            else float(timeout)
        )
        response = self._request(
            "align",
            timeout=alignment_timeout + 5.0,
            joint_positions=q.tolist(),
            gripper_width=None if gripper_width is None else float(gripper_width),
            tolerance=(
                _env_float("FACTR_POSITION_TOLERANCE", 0.04)
                if tolerance is None
                else float(tolerance)
            ),
            alignment_timeout=alignment_timeout,
            max_velocity=(
                _env_float("FACTR_MAX_VELOCITY", 0.5)
                if max_velocity is None
                else float(max_velocity)
            ),
        )
        return bool(response.get("reached", False))

    def go_to_joint_positions(self, q_cmd, max_vel_norm_factor=None, duration_s=None):
        del max_vel_norm_factor
        return self.align_to_mujoco(q_cmd, timeout=duration_s)

    def switch_control_mode(self, control_mode):
        mode = str(control_mode or "").strip().upper()
        if mode in {"HUMAN_CONTROL", "LEADER", "TAKEOVER"}:
            self._request("begin_takeover")
            return
        if mode in {"OFF", "DISABLE", "DISABLED", "TORQUE_DISABLED"}:
            self.close()
            return
        self.close()

    def _state(self) -> dict[str, Any]:
        if not self.connected:
            raise FactrRpcError("FACTR client is not connected")
        return self._request("get_state")

    def get_joint_positions(self):
        state = self._state()
        return self._validate_joint_vector(
            state["joint_positions"],
            direction="FACTR/MuJoCo",
        )

    def get_joint_state(self):
        state = self._state()
        q = self._validate_joint_vector(
            state["joint_positions"],
            direction="FACTR/MuJoCo",
        )
        dq = self._validate_joint_vector(
            state["joint_velocities"],
            direction="FACTR velocity/MuJoCo",
        )
        return q, dq

    def get_gripper_width(self):
        return float(self._state()["gripper_width"])

    def get_gripper_pressed(self):
        return bool(self._state()["gripper_pressed"])

    def restore_gripper_width(self, target_width, tol=0.002, timeout=1.5, hz=40.0):
        del tol, timeout, hz
        self._request("set_gripper_width", gripper_width=float(target_width))


class FactrRpcService:
    """Single-owner state machine around a hardware FactrControlAdapter."""

    def __init__(self, hardware_adapter):
        self.adapter = hardware_adapter
        self.state = "INITIALIZING"
        self.owner: str | None = None
        self._mujoco_anchor: np.ndarray | None = None
        self._factr_anchor: np.ndarray | None = None
        self._return_thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self.closed = False

    def initialize(self) -> None:
        with self._lock:
            self.adapter.connect(already_connected=False)
            self.adapter.stage_initial_pose()
            self.state = "WAITING"

    def _require_owner(self, client_id: str) -> None:
        if self.owner != client_id:
            current = self.owner or "none"
            raise FactrRpcError(
                f"FACTR is not owned by client {client_id}; current owner is {current}"
            )

    def dispatch(self, request: dict[str, Any]) -> dict[str, Any]:
        operation = str(request.get("operation", ""))
        client_id = str(request.get("client_id", ""))
        if not operation:
            raise ValueError("FACTR RPC request is missing operation")
        if not client_id and operation != "health":
            raise ValueError("FACTR RPC request is missing client_id")

        with self._lock:
            if operation == "health":
                return {"state": self.state, "owner": self.owner}

            if operation == "claim":
                if self.state == "RETURNING_INIT":
                    # interrupt_return_to_initial() takes the adapter's motion lock, which
                    # the return move holds for its whole duration — so this call is what
                    # actually guarantees the mover has stopped, not the join below. The
                    # join is bookkeeping only; do not lengthen it, it is held under
                    # self._lock and stalls every other session's RPCs.
                    if hasattr(self.adapter, "interrupt_return_to_initial"):
                        self.adapter.interrupt_return_to_initial()
                    self.state = "WAITING"
                    thread = self._return_thread
                    if thread is not None and thread.is_alive():
                        thread.join(timeout=_env_float("FACTR_RETURN_INTERRUPT_JOIN_S", 0.5))
                        if thread.is_alive():
                            # The mover has released the motion lock (interrupt returned),
                            # so this is a slow teardown rather than continued motion. Say
                            # so instead of leaving a silent thread behind.
                            print(
                                "[FACTR][WARN] Return thread still finishing after the "
                                "interrupt; proceeding (the arm has already stopped).",
                                flush=True,
                            )
                    self._return_thread = None
                if self.state not in {"WAITING", "TAKEOVER", "ALIGNING"}:
                    raise FactrRpcError(f"FACTR service is not ready: state={self.state}")
                if self.owner not in {None, client_id}:
                    raise FactrRpcError(
                        f"FACTR is busy with another policy client: {self.owner}"
                    )
                self.owner = client_id
                return {"state": self.state}

            if operation == "release":
                if self.owner is None:
                    return {"state": self.state}
                self._require_owner(client_id)
                try:
                    self._end_takeover_without_parking()
                finally:
                    self.owner = None
                    self._mujoco_anchor = None
                    self._factr_anchor = None
                    self._start_return_to_initial_if_needed_locked()
                return {"state": self.state}

            self._require_owner(client_id)

            if operation == "align":
                self.state = "ALIGNING"
                try:
                    q_mujoco = np.asarray(
                        request.get("joint_positions", []), dtype=np.float64
                    )
                    if q_mujoco.shape != (7,):
                        raise ValueError(
                            "MuJoCo/FACTR joint dimension mismatch: "
                            f"expected (7,), got {q_mujoco.shape}"
                        )
                    reached = self.adapter.align_to_mujoco(
                        q_mujoco,
                        gripper_width=request.get("gripper_width"),
                        tolerance=float(request["tolerance"]),
                        timeout=float(request["alignment_timeout"]),
                        max_velocity=float(request["max_velocity"]),
                    )
                    if not reached:
                        raise TimeoutError("FACTR alignment returned without reaching target")
                    q_factr, _ = self.adapter.get_joint_state()
                    self._mujoco_anchor = q_mujoco.copy()
                    self._factr_anchor = np.asarray(q_factr, dtype=np.float64).copy()
                    self.state = "ALIGNED"
                    return {"state": self.state, "reached": True}
                except Exception:
                    self._safe_release_after_error()
                    raise

            if operation == "begin_takeover":
                if self.state != "ALIGNED":
                    raise FactrRpcError(
                        f"FACTR takeover requires successful alignment; state={self.state}"
                    )
                try:
                    aligned_factr_anchor = (
                        None if self._factr_anchor is None else self._factr_anchor.copy()
                    )
                    if _env_bool("INTERVENE_FACTR_TOUCH_RELEASE_ENABLED", True):
                        if aligned_factr_anchor is None:
                            raise FactrRpcError("Missing FACTR aligned anchor")
                        if hasattr(self.adapter, "wait_for_takeover_touch"):
                            aligned_factr_anchor = np.asarray(
                                self.adapter.wait_for_takeover_touch(
                                    aligned_factr_anchor
                                ),
                                dtype=np.float64,
                            ).copy()
                    self.adapter.begin_takeover()
                    settle_s = _env_float("INTERVENE_FACTR_TAKEOVER_SETTLE_SECONDS", 0.4)
                    if settle_s > 0.0:
                        print(
                            "[FACTR] Settling leader handoff before MuJoCo follows "
                            f"({settle_s:.2f}s).",
                            flush=True,
                        )
                        time.sleep(settle_s)
                    if _env_bool("INTERVENE_FACTR_PRESERVE_ALIGNED_ANCHOR", True):
                        self._factr_anchor = aligned_factr_anchor
                    else:
                        q_factr, _ = self.adapter.get_joint_state()
                        self._factr_anchor = np.asarray(q_factr, dtype=np.float64).copy()
                except Exception:
                    self._safe_release_after_error()
                    raise
                self.state = "TAKEOVER"
                return {"state": self.state}

            if operation == "get_state":
                if self.state != "TAKEOVER":
                    raise FactrRpcError(
                        f"FACTR state streaming requires TAKEOVER; state={self.state}"
                    )
                q, dq = self.adapter.get_joint_state()
                q = np.asarray(q, dtype=np.float64)
                dq = np.asarray(dq, dtype=np.float64)
                if q.shape != (7,) or dq.shape != (7,):
                    raise ValueError(
                        "FACTR hardware returned incompatible dimensions: "
                        f"q={q.shape}, dq={dq.shape}, expected (7,)"
                    )
                if self._mujoco_anchor is None or self._factr_anchor is None:
                    raise FactrRpcError("Missing FACTR/MuJoCo alignment anchor")
                # Map FACTR motion relative to its latest hardware anchor onto
                # MuJoCo motion relative to the paused simulation anchor.  The
                # anchors are refreshed when leader mode starts, so any small
                # physical settling during the FACTR controller handoff is not
                # interpreted as a human command.
                q_delta = (q - self._factr_anchor + np.pi) % (2.0 * np.pi) - np.pi
                q = self._mujoco_anchor + q_delta
                self._factr_anchor = (
                    self._factr_anchor + q_delta
                )
                self._mujoco_anchor = q.copy()
                return {
                    "state": self.state,
                    "joint_positions": q.tolist(),
                    "joint_velocities": dq.tolist(),
                    "gripper_width": float(self.adapter.get_gripper_width()),
                    "gripper_pressed": bool(self.adapter.get_gripper_pressed()),
                }

            if operation == "set_gripper_width":
                self.adapter.restore_gripper_width(float(request["gripper_width"]))
                return {"state": self.state}

            raise ValueError(f"Unknown FACTR RPC operation: {operation}")

    def _safe_release_after_error(self) -> None:
        try:
            self._end_takeover_without_parking()
        except Exception:
            pass
        self.owner = None
        self._mujoco_anchor = None
        self._factr_anchor = None
        self.state = "WAITING"

    def _end_takeover_without_parking(self) -> None:
        try:
            self.adapter.end_takeover(return_to_initial=False)
        except TypeError:
            self.adapter.end_takeover()

    def _start_return_to_initial_if_needed_locked(self) -> None:
        if (
            not _env_bool("INTERVENE_FACTR_RETURN_INIT_ON_RELEASE", True)
            or not hasattr(self.adapter, "return_to_initial_pose")
        ):
            self.state = "WAITING"
            return

        self.state = "RETURNING_INIT"

        # Arm the cancel event HERE, synchronously, before the thread exists.
        # return_to_initial_pose() used to create it itself — but only after connect()
        # and a pose file read, so a `claim` landing in that window found
        # _move_cancel_event still None, cancelled nothing, and the return then ran a
        # full uncancelled move while the new claim went on to align: two movers on one
        # leader arm. release -> immediate claim is exactly the study workflow.
        if hasattr(self.adapter, "arm_move_cancel_event"):
            try:
                self.adapter.arm_move_cancel_event()
            except Exception as exc:
                print(f"[FACTR][WARN] Could not pre-arm return cancel: {exc}", flush=True)

        def _return_to_initial() -> None:
            try:
                print(
                    "[FACTR] Returning to initial pose in background; "
                    "policy may continue.",
                    flush=True,
                )
                self.adapter.return_to_initial_pose()
            except Exception as exc:
                print(
                    "[FACTR][WARN] Background return to initial pose failed: "
                    f"{type(exc).__name__}: {exc}",
                    flush=True,
                )
            finally:
                with self._lock:
                    if self.state == "RETURNING_INIT":
                        self.state = "WAITING"

        self._return_thread = threading.Thread(
            target=_return_to_initial,
            name="factr-return-init",
            daemon=True,
        )
        self._return_thread.start()

    def close(self) -> None:
        with self._lock:
            if self.closed:
                return
            self.closed = True
            self.state = "STOPPING"
            self.owner = None
            self._mujoco_anchor = None
            self._factr_anchor = None
            try:
                self.adapter.close()
            finally:
                self.state = "STOPPED"


class _FactrRequestHandler(socketserver.StreamRequestHandler):
    def handle(self):
        raw = self.rfile.readline(MAX_MESSAGE_BYTES + 1)
        if len(raw) > MAX_MESSAGE_BYTES:
            self._respond({"ok": False, "error_type": "ValueError", "error": "request too large"})
            return
        try:
            request = json.loads(raw.decode("utf-8"))
            payload = self.server.factr_service.dispatch(request)
            self._respond({"ok": True, **payload})
        except Exception as exc:
            self._respond(
                {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )

    def _respond(self, payload: dict[str, Any]) -> None:
        self.wfile.write((json.dumps(payload, separators=(",", ":")) + "\n").encode("utf-8"))


class FactrTcpServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, service: FactrRpcService):
        self.factr_service = service
        super().__init__(address, _FactrRequestHandler)


def write_ready_file(path: str | Path, *, host: str, port: int) -> None:
    ready_path = Path(path)
    ready_path.parent.mkdir(parents=True, exist_ok=True)
    ready_path.write_text(
        json.dumps({"ready": True, "host": host, "port": int(port)}) + "\n",
        encoding="utf-8",
    )
