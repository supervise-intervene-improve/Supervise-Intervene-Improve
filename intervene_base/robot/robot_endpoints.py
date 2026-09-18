"""Robot network endpoints (host + arm/gripper ports), with environment overrides.

WHY THIS MODULE EXISTS
----------------------
`record.ROBOTS` used to hardcode every address, and `collect_data.Robot.__init__`
builds its gRPC clients immediately — so an address baked into that table cannot be
corrected without editing source. That matters here because the lab machines get their
Ethernet address from DHCP (`enp8s0` is `dynamic` on workstation), so an IP that is correct
today can silently drift tomorrow and surface only as a gRPC "failed to connect to all
addresses" at the first intervention attempt.

This module holds the endpoint table on its own and applies env overrides, so:

  * `record.py` imports it to build the real `Robot` objects, and
  * `utils/robot_doctor.py` imports it to probe reachability

...from a single source of truth. It deliberately has **no heavy imports** (no mujoco, no
polymetis, no real_robot_env) so the doctor can run in any interpreter, including one
where the robot stack is not installed — which is exactly the situation you are in when
you are trying to work out why the robot stack cannot connect.

Overrides (all optional, per robot key, case-insensitive key in the variable name):

    INTERVENE_ROBOT_P4_IP=192.0.2.153
    INTERVENE_ROBOT_P4_ARM_PORT=50053
    INTERVENE_ROBOT_P4_GRIPPER_PORT=50054

Defaults below are unchanged from the original `record.ROBOTS` table.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class RobotEndpoint:
    key: str
    name: str
    ip_address: str
    arm_port: int
    gripper_port: int

    @property
    def arm_addr(self) -> str:
        return f"{self.ip_address}:{self.arm_port}"

    @property
    def gripper_addr(self) -> str:
        return f"{self.ip_address}:{self.gripper_port}"


# Defaults — kept byte-identical to the historical record.ROBOTS values.
_DEFAULTS = {
    "p1": ("p1 leader", "192.168.0.150", 1234, 1235),
    "p2": ("p2 leader", "192.168.0.150", 4321, 4322),
    "p3": ("p3 follower", "192.0.2.153", 50051, 50052),
    "p4": ("p4 follower", "192.0.2.153", 50053, 50054),
}


def _env_str(key: str, suffix: str, default: str) -> str:
    value = os.environ.get(f"INTERVENE_ROBOT_{key.upper()}_{suffix}")
    if value is None:
        return default
    value = value.strip()
    return value or default


def _env_int(key: str, suffix: str, default: int) -> int:
    raw = os.environ.get(f"INTERVENE_ROBOT_{key.upper()}_{suffix}")
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw.strip())
    except ValueError:
        # A malformed port is a configuration mistake, not a reason to crash the whole
        # launch — fall back to the default loudly so it shows up in the log.
        print(
            f"[RobotEndpoints][WARN] INTERVENE_ROBOT_{key.upper()}_{suffix}="
            f"{raw!r} is not an integer; using default {default}."
        )
        return default


def load_endpoints() -> Dict[str, RobotEndpoint]:
    """Build the endpoint table, applying any environment overrides."""
    endpoints: Dict[str, RobotEndpoint] = {}
    for key, (name, ip, arm_port, gripper_port) in _DEFAULTS.items():
        resolved_ip = _env_str(key, "IP", ip)
        resolved_arm = _env_int(key, "ARM_PORT", arm_port)
        resolved_gripper = _env_int(key, "GRIPPER_PORT", gripper_port)
        if (resolved_ip, resolved_arm, resolved_gripper) != (ip, arm_port, gripper_port):
            print(
                f"[RobotEndpoints] {key}: overridden -> {resolved_ip}:"
                f"{resolved_arm}/{resolved_gripper} (default {ip}:{arm_port}/{gripper_port})"
            )
        endpoints[key] = RobotEndpoint(
            key=key,
            name=name,
            ip_address=resolved_ip,
            arm_port=resolved_arm,
            gripper_port=resolved_gripper,
        )
    return endpoints


ENDPOINTS: Dict[str, RobotEndpoint] = load_endpoints()
