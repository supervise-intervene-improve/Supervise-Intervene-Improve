import sys
import unittest
from pathlib import Path

import numpy as np


SII_DIR = Path(__file__).resolve().parents[1] / "sii"
if str(SII_DIR) not in sys.path:
    sys.path.insert(0, str(SII_DIR))

from GPU.GpuPointCloudBuffers import GpuPointCloudBufferPool
from GPU.GpuPointCloudContract import (
    decode_frame,
    decode_frame_header,
    encode_frame_into,
    stage_frame_data,
)


class GpuPointCloudContractTests(unittest.TestCase):
    def test_round_trip_preserves_header_and_quantized_data(self):
        pool = GpuPointCloudBufferPool()
        slot = pool.ensure_slot("top", 8)

        xyz = np.array(
            [
                [0.1000, -0.2000, 0.3000],
                [0.0124, 0.0126, -0.0104],
            ],
            dtype=np.float32,
        )
        rgb = np.array([[255, 12, 34], [5, 6, 7]], dtype=np.uint8)

        count = stage_frame_data(slot, xyz, rgb)
        payload = bytes(encode_frame_into(slot, declared_capacity=8, actual_count=count))

        header, xyz_mm, rgb_u8 = decode_frame(payload)
        self.assertEqual(header.declared_capacity, 8)
        self.assertEqual(header.actual_count, 2)
        np.testing.assert_array_equal(
            xyz_mm,
            np.array([[100, -200, 300], [12, 13, -10]], dtype=np.int16),
        )
        np.testing.assert_array_equal(rgb_u8, rgb)

    def test_decode_rejects_length_mismatch(self):
        pool = GpuPointCloudBufferPool()
        slot = pool.ensure_slot("top", 1)
        xyz = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        rgb = np.array([[1, 2, 3]], dtype=np.uint8)
        count = stage_frame_data(slot, xyz, rgb)
        payload = bytearray(encode_frame_into(slot, declared_capacity=1, actual_count=count))
        payload.pop()

        with self.assertRaisesRegex(ValueError, "length mismatch"):
            decode_frame_header(payload)

    def test_encode_rejects_over_capacity(self):
        pool = GpuPointCloudBufferPool()
        slot = pool.ensure_slot("left", 2)
        xyz = np.zeros((3, 3), dtype=np.float32)
        rgb = np.zeros((3, 3), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "exceeds buffer slot capacity"):
            stage_frame_data(slot, xyz, rgb)

    def test_buffer_pool_reuses_existing_camera_slot(self):
        pool = GpuPointCloudBufferPool()
        first = pool.ensure_slot("right", 12)
        second = pool.ensure_slot("right", 6)
        self.assertIs(first, second)


if __name__ == "__main__":
    unittest.main()
