from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

from GPU.GpuPointCloudContract import payload_nbytes_for_capacity


@dataclass
class GpuPointCloudBufferSlot:
    camera_name: str
    capacity: int
    xyz_mm: np.ndarray
    rgb_u8: np.ndarray
    xyz_mm_scratch: np.ndarray
    payload: bytearray
    device_tensors: Dict[str, object] = field(default_factory=dict)


class GpuPointCloudBufferPool:
    """
    Reuses stable staging buffers per camera so point-count fluctuations do not
    constantly reallocate host-side packing memory.
    """

    def __init__(self):
        self._slots: Dict[str, GpuPointCloudBufferSlot] = {}

    def ensure_slot(self, camera_name: str, capacity: int) -> GpuPointCloudBufferSlot:
        capacity = max(0, int(capacity))
        slot = self._slots.get(camera_name)
        if slot is not None and slot.capacity >= capacity:
            return slot

        alloc_capacity = max(1, capacity)
        slot = GpuPointCloudBufferSlot(
            camera_name=str(camera_name),
            capacity=alloc_capacity,
            xyz_mm=np.zeros((alloc_capacity, 3), dtype=np.int16),
            rgb_u8=np.zeros((alloc_capacity, 3), dtype=np.uint8),
            xyz_mm_scratch=np.zeros((alloc_capacity, 3), dtype=np.float32),
            payload=bytearray(payload_nbytes_for_capacity(alloc_capacity)),
        )
        self._slots[camera_name] = slot
        return slot
