from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Tuple

import numpy as np


FRAME_MAGIC = b"GPC1"
HEADER_STRUCT = struct.Struct("<4sII")
HEADER_SIZE = HEADER_STRUCT.size
XYZ_BYTES_PER_POINT = 3 * np.dtype(np.int16).itemsize
RGB_BYTES_PER_POINT = 3 * np.dtype(np.uint8).itemsize
POINT_STRIDE_BYTES = XYZ_BYTES_PER_POINT + RGB_BYTES_PER_POINT


@dataclass(frozen=True)
class GpuPcFrameHeader:
    declared_capacity: int
    actual_count: int


def payload_nbytes_for_capacity(capacity: int) -> int:
    capacity = max(0, int(capacity))
    return HEADER_SIZE + (capacity * POINT_STRIDE_BYTES)


def decode_frame_header(payload: bytes | bytearray | memoryview) -> GpuPcFrameHeader:
    if payload is None:
        raise ValueError("Point-cloud payload is null.")
    if len(payload) < HEADER_SIZE:
        raise ValueError(
            f"Point-cloud payload too small for header: got {len(payload)} bytes."
        )

    magic, declared_capacity, actual_count = HEADER_STRUCT.unpack_from(payload, 0)
    if magic != FRAME_MAGIC:
        raise ValueError(f"Unexpected point-cloud magic: {magic!r}")
    if declared_capacity < 0 or actual_count < 0:
        raise ValueError("Negative declared capacity or point count in payload header.")
    if actual_count > declared_capacity:
        raise ValueError(
            f"Point-cloud count {actual_count} exceeds declared capacity {declared_capacity}."
        )

    expected_len = HEADER_SIZE + (actual_count * POINT_STRIDE_BYTES)
    if len(payload) != expected_len:
        raise ValueError(
            f"Point-cloud payload length mismatch: got {len(payload)} bytes, expected {expected_len}."
        )

    return GpuPcFrameHeader(
        declared_capacity=int(declared_capacity),
        actual_count=int(actual_count),
    )


def stage_frame_data(slot, xyz_unity_m: np.ndarray, rgb_u8: np.ndarray) -> int:
    xyz = np.ascontiguousarray(xyz_unity_m, dtype=np.float32)
    rgb = np.ascontiguousarray(rgb_u8, dtype=np.uint8)

    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz_unity_m must have shape (N, 3), got {xyz.shape}")
    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(f"rgb_u8 must have shape (N, 3), got {rgb.shape}")
    if xyz.shape[0] != rgb.shape[0]:
        raise ValueError(
            f"xyz/rgb point-count mismatch: {xyz.shape[0]} != {rgb.shape[0]}"
        )

    actual_count = int(xyz.shape[0])
    if actual_count > int(slot.capacity):
        raise ValueError(
            f"Point-cloud count {actual_count} exceeds buffer slot capacity {slot.capacity}."
        )

    if actual_count == 0:
        return 0

    scratch = slot.xyz_mm_scratch[:actual_count]
    np.multiply(xyz, 1000.0, out=scratch)
    np.rint(scratch, out=scratch)

    mm_min = float(scratch.min())
    mm_max = float(scratch.max())
    int16_info = np.iinfo(np.int16)
    if mm_min < int16_info.min or mm_max > int16_info.max:
        raise ValueError(
            f"Point-cloud coordinates exceed int16 millimeter range: [{mm_min}, {mm_max}]"
        )

    slot.xyz_mm[:actual_count] = scratch
    slot.rgb_u8[:actual_count] = rgb
    return actual_count


def encode_frame_into(slot, declared_capacity: int, actual_count: int) -> memoryview:
    declared_capacity = int(declared_capacity)
    actual_count = int(actual_count)

    if declared_capacity < 0 or actual_count < 0:
        raise ValueError("Declared capacity and actual count must be non-negative.")
    if actual_count > declared_capacity:
        raise ValueError(
            f"Point-cloud count {actual_count} exceeds declared capacity {declared_capacity}."
        )
    if declared_capacity > int(slot.capacity):
        raise ValueError(
            f"Declared capacity {declared_capacity} exceeds allocated slot capacity {slot.capacity}."
        )

    HEADER_STRUCT.pack_into(slot.payload, 0, FRAME_MAGIC, declared_capacity, actual_count)

    payload_view = memoryview(slot.payload)
    xyz_len = actual_count * XYZ_BYTES_PER_POINT
    rgb_len = actual_count * RGB_BYTES_PER_POINT

    if actual_count > 0:
        xyz_bytes = memoryview(slot.xyz_mm[:actual_count].reshape(-1)).cast("B")
        rgb_bytes = memoryview(slot.rgb_u8[:actual_count].reshape(-1)).cast("B")

        payload_view[HEADER_SIZE : HEADER_SIZE + xyz_len] = xyz_bytes
        payload_view[
            HEADER_SIZE + xyz_len : HEADER_SIZE + xyz_len + rgb_len
        ] = rgb_bytes

    total_len = HEADER_SIZE + xyz_len + rgb_len
    return payload_view[:total_len]


def decode_frame(
    payload: bytes | bytearray | memoryview,
) -> Tuple[GpuPcFrameHeader, np.ndarray, np.ndarray]:
    header = decode_frame_header(payload)

    xyz_nbytes = header.actual_count * XYZ_BYTES_PER_POINT
    body = memoryview(payload)[HEADER_SIZE:]

    xyz_mm = np.frombuffer(
        body[:xyz_nbytes],
        dtype="<i2",
        count=header.actual_count * 3,
    ).reshape(header.actual_count, 3).copy()
    rgb_u8 = np.frombuffer(
        body[xyz_nbytes : xyz_nbytes + (header.actual_count * RGB_BYTES_PER_POINT)],
        dtype=np.uint8,
        count=header.actual_count * 3,
    ).reshape(header.actual_count, 3).copy()
    return header, xyz_mm, rgb_u8
