#!/usr/bin/env python3
"""Commission a replacement FACTR Dynamixel without enabling motor torque.

This utility talks directly to one Protocol 2.0 servo. It can scan for a new
motor, configure the FACTR J3 communication settings, and verify the complete
1..8 chain. It never writes goal position/current, never enables torque, and
never clears the multi-turn counter.

For configuration, disconnect all other servos from the data bus. The script
also scans the current baud rate and refuses to continue unless exactly one
motor is visible.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml
from dynamixel_sdk import PacketHandler, PortHandler
from dynamixel_sdk.robotis_def import COMM_SUCCESS


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "factr" / "leader.yaml"
PROTOCOL_VERSION = 2.0
EXPECTED_XC330_T288_MODEL = 1220

ADDR_MODEL_NUMBER = 0
ADDR_ID = 7
ADDR_BAUD_RATE = 8
ADDR_RETURN_DELAY = 9
ADDR_OPERATING_MODE = 11
ADDR_SECONDARY_ID = 12
ADDR_TORQUE_ENABLE = 64
ADDR_STATUS_RETURN_LEVEL = 68
ADDR_HARDWARE_ERROR = 70
ADDR_PRESENT_INPUT_VOLTAGE = 144
ADDR_PRESENT_TEMPERATURE = 146

BAUD_CODES = {
    9_600: 0,
    57_600: 1,
    115_200: 2,
    1_000_000: 3,
    2_000_000: 4,
    3_000_000: 5,
    4_000_000: 6,
}
DEFAULT_SCAN_BAUDS = (
    57_600,
    4_000_000,
    1_000_000,
    2_000_000,
    3_000_000,
    115_200,
    9_600,
)


class CommissioningError(RuntimeError):
    """Raised when a commissioning precondition or verification fails."""


@dataclass(frozen=True)
class FoundMotor:
    dxl_id: int
    model_number: int


def _resolve_port(config_path: Path, override: str | None) -> Path:
    if override:
        configured = override
    else:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        configured = str(payload["dynamixel"]["dynamixel_port"])
    if configured.startswith("/"):
        port = Path(configured)
    elif configured.startswith("usb-"):
        port = Path("/dev/serial/by-id") / configured
    else:
        port = Path(configured)
    if not port.exists():
        raise CommissioningError(f"Serial port does not exist: {port}")
    return port


def _other_port_users(port: Path) -> list[int]:
    resolved = port.resolve()
    users: set[int] = set()
    own_pid = os.getpid()
    for fd_path in glob.glob("/proc/[0-9]*/fd/*"):
        try:
            pid = int(fd_path.split("/")[2])
            if pid == own_pid:
                continue
            if Path(fd_path).resolve() == resolved:
                users.add(pid)
        except (FileNotFoundError, PermissionError, OSError, ValueError):
            continue
    return sorted(users)


def _result_text(packet: PacketHandler, result: int, error: int = 0) -> str:
    details = packet.getTxRxResult(result)
    if error:
        details += f"; device={packet.getRxPacketError(error)}"
    return f"result={result} ({details}), device_error={error}"


class Bus:
    def __init__(self, port_path: Path, baud: int):
        self.port_path = port_path
        self.baud = baud
        self.port = PortHandler(str(port_path))
        self.packet = PacketHandler(PROTOCOL_VERSION)

    def __enter__(self) -> "Bus":
        if not self.port.setBaudRate(self.baud):
            raise CommissioningError(
                f"Could not open {self.port_path} at {self.baud} baud"
            )
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        try:
            self.port.closePort()
        except Exception:
            pass

    def ping(self, dxl_id: int) -> FoundMotor | None:
        model, result, error = self.packet.ping(self.port, dxl_id)
        if result != COMM_SUCCESS or error != 0:
            return None
        return FoundMotor(dxl_id=dxl_id, model_number=int(model))

    def scan(self, first_id: int, last_id: int) -> list[FoundMotor]:
        found = []
        for dxl_id in range(first_id, last_id + 1):
            motor = self.ping(dxl_id)
            if motor is not None:
                found.append(motor)
        return found

    def broadcast_scan(self) -> tuple[list[FoundMotor], int]:
        """Discover Protocol 2.0 motors using the SDK's long receive window."""
        data, result = self.packet.broadcastPing(self.port)
        found = [
            FoundMotor(dxl_id=int(dxl_id), model_number=int(values[0]))
            for dxl_id, values in sorted(data.items())
        ]
        return found, int(result)

    def read_u8(self, dxl_id: int, address: int, label: str) -> int:
        value, result, error = self.packet.read1ByteTxRx(
            self.port, dxl_id, address
        )
        if result != COMM_SUCCESS or error != 0:
            raise CommissioningError(
                f"Failed to read {label} from ID {dxl_id}: "
                f"{_result_text(self.packet, result, error)}"
            )
        return int(value)

    def read_u16(self, dxl_id: int, address: int, label: str) -> int:
        value, result, error = self.packet.read2ByteTxRx(
            self.port, dxl_id, address
        )
        if result != COMM_SUCCESS or error != 0:
            raise CommissioningError(
                f"Failed to read {label} from ID {dxl_id}: "
                f"{_result_text(self.packet, result, error)}"
            )
        return int(value)

    def write_u8_verified(
        self,
        dxl_id: int,
        address: int,
        value: int,
        label: str,
        *,
        verify_id: int | None = None,
    ) -> None:
        # TxOnly works even when Status Return Level was previously configured to
        # suppress write acknowledgements. Every write is verified with a read.
        result = self.packet.write1ByteTxOnly(
            self.port, dxl_id, address, int(value)
        )
        if result != COMM_SUCCESS:
            raise CommissioningError(
                f"Failed to transmit {label} to ID {dxl_id}: "
                f"{_result_text(self.packet, result)}"
            )
        time.sleep(0.10)
        read_id = dxl_id if verify_id is None else verify_id
        actual = self.read_u8(read_id, address, label)
        if actual != value:
            raise CommissioningError(
                f"{label} verification failed on ID {read_id}: "
                f"expected {value}, read {actual}"
            )

    def write_u8_tx_only(
        self, dxl_id: int, address: int, value: int, label: str
    ) -> None:
        result = self.packet.write1ByteTxOnly(
            self.port, dxl_id, address, int(value)
        )
        if result != COMM_SUCCESS:
            raise CommissioningError(
                f"Failed to transmit {label} to ID {dxl_id}: "
                f"{_result_text(self.packet, result)}"
            )
        time.sleep(0.10)

    def settings(self, dxl_id: int) -> dict[str, int]:
        return {
            "model_number": self.read_u16(
                dxl_id, ADDR_MODEL_NUMBER, "Model Number"
            ),
            "id": self.read_u8(dxl_id, ADDR_ID, "ID"),
            "baud_code": self.read_u8(dxl_id, ADDR_BAUD_RATE, "Baud Rate"),
            "return_delay": self.read_u8(
                dxl_id, ADDR_RETURN_DELAY, "Return Delay Time"
            ),
            "operating_mode": self.read_u8(
                dxl_id, ADDR_OPERATING_MODE, "Operating Mode"
            ),
            "secondary_id": self.read_u8(
                dxl_id, ADDR_SECONDARY_ID, "Secondary ID"
            ),
            "torque_enable": self.read_u8(
                dxl_id, ADDR_TORQUE_ENABLE, "Torque Enable"
            ),
            "status_return_level": self.read_u8(
                dxl_id, ADDR_STATUS_RETURN_LEVEL, "Status Return Level"
            ),
            "hardware_error": self.read_u8(
                dxl_id, ADDR_HARDWARE_ERROR, "Hardware Error Status"
            ),
        }


def _print_settings(settings: dict[str, int], *, baud: int) -> None:
    print(f"  model_number={settings['model_number']}")
    print(f"  id={settings['id']}")
    print(f"  baud={baud} (register={settings['baud_code']})")
    print(f"  return_delay={settings['return_delay']}")
    print(f"  operating_mode={settings['operating_mode']}")
    print(f"  secondary_id={settings['secondary_id']}")
    print(f"  torque_enable={settings['torque_enable']}")
    print(f"  status_return_level={settings['status_return_level']}")
    print(f"  hardware_error={settings['hardware_error']}")


HARDWARE_ERROR_BITS = {
    0x01: "input voltage",
    0x04: "overheating",
    0x10: "electrical shock / insufficient power",
    0x20: "overload",
}


def _decode_hardware_error(value: int) -> str:
    if value == 0:
        return "none"
    names = [name for bit, name in HARDWARE_ERROR_BITS.items() if value & bit]
    known_mask = sum(HARDWARE_ERROR_BITS)
    unknown = value & ~known_mask
    if unknown:
        names.append(f"unknown bits 0x{unknown:02x}")
    return ", ".join(names)


def _validate_id_range(first_id: int, last_id: int) -> None:
    if not 0 <= first_id <= last_id <= 252:
        raise CommissioningError("ID range must satisfy 0 <= first <= last <= 252")


def run_scan(args: argparse.Namespace, port_path: Path) -> int:
    _validate_id_range(args.first_id, args.last_id)
    bauds = args.baud or list(DEFAULT_SCAN_BAUDS)
    for baud in bauds:
        if baud not in BAUD_CODES:
            raise CommissioningError(f"Unsupported XC330 baud rate: {baud}")

    total = 0
    for baud in bauds:
        print(
            f"[SCAN] {port_path} at {baud} baud, "
            f"IDs {args.first_id}..{args.last_id}"
        )
        with Bus(port_path, baud) as bus:
            found = bus.scan(args.first_id, args.last_id)
        if not found:
            print("  no motors found")
            continue
        for motor in found:
            print(f"  ID {motor.dxl_id}: model {motor.model_number}")
        total += len(found)
    print(f"[SCAN] Total responses across baud rates: {total}")
    return 0 if total else 1


def run_broadcast_scan(args: argparse.Namespace, port_path: Path) -> int:
    """Probe all IDs with a receive window long enough for delayed replies."""
    bauds = args.baud or list(DEFAULT_SCAN_BAUDS)
    for baud in bauds:
        if baud not in BAUD_CODES:
            raise CommissioningError(f"Unsupported XC330 baud rate: {baud}")

    total = 0
    print(
        "[BROADCAST-SCAN] Read-only Protocol 2.0 ping; no register writes "
        "will be sent."
    )
    for baud in bauds:
        with Bus(port_path, baud) as bus:
            found, result = bus.broadcast_scan()
            result_text = bus.packet.getTxRxResult(result)
        print(f"[BROADCAST-SCAN] {baud} baud: result={result} ({result_text})")
        if not found:
            print("  no motors found")
            continue
        for motor in found:
            print(f"  ID {motor.dxl_id}: model {motor.model_number}")
        total += len(found)
    print(f"[BROADCAST-SCAN] Total responses across baud rates: {total}")
    return 0 if total else 1


def run_diagnose(args: argparse.Namespace, port_path: Path) -> int:
    """Read alarm telemetry without changing torque or EEPROM settings."""
    _validate_id_range(args.first_id, args.last_id)
    if args.baud not in BAUD_CODES:
        raise CommissioningError(f"Unsupported baud rate: {args.baud}")

    failures = 0
    with Bus(port_path, args.baud) as bus:
        found = bus.scan(args.first_id, args.last_id)
        if not found:
            print(f"[DIAGNOSE] No motors responded at {args.baud} baud")
            return 1

        print(
            "[DIAGNOSE] Read-only telemetry; no torque, goal, reboot, or "
            "EEPROM writes will be sent."
        )
        for motor in found:
            try:
                torque = bus.read_u8(
                    motor.dxl_id, ADDR_TORQUE_ENABLE, "Torque Enable"
                )
                error = bus.read_u8(
                    motor.dxl_id, ADDR_HARDWARE_ERROR, "Hardware Error Status"
                )
                voltage_raw = bus.read_u16(
                    motor.dxl_id,
                    ADDR_PRESENT_INPUT_VOLTAGE,
                    "Present Input Voltage",
                )
                temperature = bus.read_u8(
                    motor.dxl_id,
                    ADDR_PRESENT_TEMPERATURE,
                    "Present Temperature",
                )
                print(
                    f"  ID {motor.dxl_id}: model={motor.model_number} "
                    f"torque={torque} voltage={voltage_raw / 10.0:.1f}V "
                    f"temperature={temperature}C error=0x{error:02x} "
                    f"({_decode_hardware_error(error)})"
                )
            except CommissioningError as exc:
                failures += 1
                print(f"  ID {motor.dxl_id}: READ FAILED: {exc}")

    if args.warn_duplicate_id is not None:
        print(
            f"[DIAGNOSE] WARNING: telemetry for ID {args.warn_duplicate_id} "
            "is not trustworthy if two motors share that ID; their responses "
            "collide and software cannot address them separately."
        )
    return 1 if failures else 0


def _require_expected_motor(motor: FoundMotor, expected_model: int) -> None:
    if motor.model_number != expected_model:
        raise CommissioningError(
            f"ID {motor.dxl_id} is model {motor.model_number}; expected "
            f"XC330-T288-T model {expected_model}"
        )


def run_configure(args: argparse.Namespace, port_path: Path) -> int:
    if not args.yes:
        raise CommissioningError(
            "Configuration requires --yes after isolating and supporting the motor"
        )
    if args.current_baud not in BAUD_CODES:
        raise CommissioningError(
            f"Unsupported current baud rate: {args.current_baud}"
        )
    if args.target_baud not in BAUD_CODES:
        raise CommissioningError(
            f"Unsupported target baud rate: {args.target_baud}"
        )
    _validate_id_range(args.current_id, args.current_id)
    _validate_id_range(args.target_id, args.target_id)

    print(
        "[CONFIGURE] Torque will remain disabled. No goal position/current "
        "or multi-turn command will be written."
    )
    print(
        f"[CONFIGURE] Requiring exactly one motor at {args.current_baud} baud."
    )
    if (
        args.current_id != args.target_id
        or args.current_baud != args.target_baud
    ):
        with Bus(port_path, args.target_baud) as target_bus:
            occupant = target_bus.ping(args.target_id)
        if occupant is not None:
            raise CommissioningError(
                f"Destination ID {args.target_id} already responds at "
                f"{args.target_baud} baud (model {occupant.model_number})"
            )

    with Bus(port_path, args.current_baud) as bus:
        found = bus.scan(0, 252)
        if len(found) != 1:
            found_text = ", ".join(
                f"ID {motor.dxl_id}/model {motor.model_number}" for motor in found
            ) or "none"
            raise CommissioningError(
                "Expected exactly one isolated motor at the current baud; found "
                f"{len(found)} ({found_text})"
            )
        motor = found[0]
        if motor.dxl_id != args.current_id:
            raise CommissioningError(
                f"Isolated motor responded as ID {motor.dxl_id}, not requested "
                f"ID {args.current_id}"
            )
        _require_expected_motor(motor, args.expected_model)

        # A replacement may have Status Return Level 0 or 1. Use TxOnly to make
        # the bus readable first, then verify that torque is actually disabled.
        bus.write_u8_tx_only(
            args.current_id,
            ADDR_TORQUE_ENABLE,
            0,
            "Torque Enable",
        )
        bus.write_u8_tx_only(
            args.current_id,
            ADDR_STATUS_RETURN_LEVEL,
            2,
            "Status Return Level",
        )
        if bus.read_u8(
            args.current_id, ADDR_TORQUE_ENABLE, "Torque Enable"
        ) != 0:
            raise CommissioningError("Torque-disable verification failed")
        if bus.read_u8(
            args.current_id,
            ADDR_STATUS_RETURN_LEVEL,
            "Status Return Level",
        ) != 2:
            raise CommissioningError("Status Return Level verification failed")

        before = bus.settings(args.current_id)
        print("[CONFIGURE] Current settings:")
        _print_settings(before, baud=args.current_baud)

        bus.write_u8_verified(
            args.current_id,
            ADDR_RETURN_DELAY,
            0,
            "Return Delay Time",
        )
        bus.write_u8_verified(
            args.current_id,
            ADDR_SECONDARY_ID,
            255,
            "Secondary ID",
        )
        bus.write_u8_verified(
            args.current_id,
            ADDR_OPERATING_MODE,
            0,
            "Operating Mode",
        )
        active_id = args.current_id
        if args.target_id != active_id:
            if bus.ping(args.target_id) is not None:
                raise CommissioningError(
                    f"Destination ID {args.target_id} already responds at "
                    f"{args.current_baud} baud"
                )
            bus.write_u8_verified(
                active_id,
                ADDR_ID,
                args.target_id,
                "ID",
                verify_id=args.target_id,
            )
            active_id = args.target_id

        if args.target_baud != args.current_baud:
            bus.write_u8_tx_only(
                active_id,
                ADDR_BAUD_RATE,
                BAUD_CODES[args.target_baud],
                "Baud Rate",
            )
        else:
            bus.write_u8_verified(
                active_id,
                ADDR_BAUD_RATE,
                BAUD_CODES[args.target_baud],
                "Baud Rate",
            )

    with Bus(port_path, args.target_baud) as bus:
        motor = bus.ping(args.target_id)
        if motor is None:
            raise CommissioningError(
                f"Motor did not respond as ID {args.target_id} at "
                f"{args.target_baud} baud after configuration"
            )
        _require_expected_motor(motor, args.expected_model)
        after = bus.settings(args.target_id)

    expected = {
        "model_number": args.expected_model,
        "id": args.target_id,
        "baud_code": BAUD_CODES[args.target_baud],
        "return_delay": 0,
        "operating_mode": 0,
        "secondary_id": 255,
        "torque_enable": 0,
        "status_return_level": 2,
        "hardware_error": 0,
    }
    mismatches = {
        key: (expected_value, after[key])
        for key, expected_value in expected.items()
        if after[key] != expected_value
    }
    print("[CONFIGURE] Verified settings:")
    _print_settings(after, baud=args.target_baud)
    if mismatches:
        raise CommissioningError(f"Post-configuration mismatch: {mismatches}")
    print("[CONFIGURE] PASS: replacement motor configured with torque disabled.")
    return 0


def run_verify(args: argparse.Namespace, port_path: Path) -> int:
    if args.baud not in BAUD_CODES:
        raise CommissioningError(f"Unsupported baud rate: {args.baud}")
    with Bus(port_path, args.baud) as bus:
        found = bus.scan(args.first_id, args.last_id)
        print(f"[VERIFY] Responses at {args.baud} baud:")
        for motor in found:
            print(f"  ID {motor.dxl_id}: model {motor.model_number}")
        expected_ids = set(range(args.first_id, args.last_id + 1))
        found_ids = {motor.dxl_id for motor in found}
        if found_ids != expected_ids:
            raise CommissioningError(
                f"Expected IDs {sorted(expected_ids)}, found {sorted(found_ids)}"
            )
        target = next(motor for motor in found if motor.dxl_id == args.target_id)
        _require_expected_motor(target, args.expected_model)
        settings = bus.settings(args.target_id)

    print(f"[VERIFY] ID {args.target_id} settings:")
    _print_settings(settings, baud=args.baud)
    required = {
        "id": args.target_id,
        "baud_code": BAUD_CODES[args.baud],
        "return_delay": 0,
        "operating_mode": 0,
        "secondary_id": 255,
        "torque_enable": 0,
        "status_return_level": 2,
        "hardware_error": 0,
    }
    mismatches = {
        key: (expected, settings[key])
        for key, expected in required.items()
        if settings[key] != expected
    }
    if mismatches:
        raise CommissioningError(f"Verification mismatch: {mismatches}")
    print("[VERIFY] PASS: complete chain responds and J3 is configured safely.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--port", default=None)
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("scan", help="Scan without writing motor settings")
    scan.add_argument("--baud", type=int, action="append")
    scan.add_argument("--first-id", type=int, default=0)
    scan.add_argument("--last-id", type=int, default=20)

    broadcast_scan = subparsers.add_parser(
        "broadcast-scan",
        help="Read-only discovery with a longer response-delay window",
    )
    broadcast_scan.add_argument(
        "--baud",
        type=int,
        action="append",
        help="Baud rate to probe; repeat for multiple rates (default: all)",
    )

    diagnose = subparsers.add_parser(
        "diagnose", help="Read alarm, voltage, temperature, and torque state"
    )
    diagnose.add_argument("--baud", type=int, default=4_000_000)
    diagnose.add_argument("--first-id", type=int, default=1)
    diagnose.add_argument("--last-id", type=int, default=8)
    diagnose.add_argument(
        "--warn-duplicate-id",
        type=int,
        default=None,
        help="Print an explicit warning that reads at this duplicated ID are ambiguous",
    )

    configure = subparsers.add_parser(
        "configure", help="Configure one isolated replacement motor"
    )
    configure.add_argument("--current-id", type=int, required=True)
    configure.add_argument("--current-baud", type=int, required=True)
    configure.add_argument("--target-id", type=int, default=3)
    configure.add_argument("--target-baud", type=int, default=4_000_000)
    configure.add_argument(
        "--expected-model", type=int, default=EXPECTED_XC330_T288_MODEL
    )
    configure.add_argument(
        "--yes",
        action="store_true",
        help="Confirm the replacement is isolated and physically supported",
    )

    verify = subparsers.add_parser(
        "verify", help="Verify the complete FACTR chain without enabling torque"
    )
    verify.add_argument("--baud", type=int, default=4_000_000)
    verify.add_argument("--first-id", type=int, default=1)
    verify.add_argument("--last-id", type=int, default=8)
    verify.add_argument("--target-id", type=int, default=3)
    verify.add_argument(
        "--expected-model", type=int, default=EXPECTED_XC330_T288_MODEL
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config_path = args.config.resolve()
        port_path = _resolve_port(config_path, args.port)
        users = _other_port_users(port_path)
        if users:
            raise CommissioningError(
                f"Serial port is already open by PID(s): {', '.join(map(str, users))}"
            )
        print(f"[FACTR Commission] Port: {port_path} -> {port_path.resolve()}")
        if args.command == "scan":
            return run_scan(args, port_path)
        if args.command == "broadcast-scan":
            return run_broadcast_scan(args, port_path)
        if args.command == "diagnose":
            return run_diagnose(args, port_path)
        if args.command == "configure":
            return run_configure(args, port_path)
        if args.command == "verify":
            return run_verify(args, port_path)
        raise CommissioningError(f"Unknown command: {args.command}")
    except CommissioningError as exc:
        print(f"[FACTR Commission][ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
