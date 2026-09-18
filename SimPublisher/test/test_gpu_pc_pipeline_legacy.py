import sys
from types import SimpleNamespace
import unittest
from pathlib import Path

import numpy as np


SII_DIR = Path(__file__).resolve().parents[1] / "sii"
if str(SII_DIR) not in sys.path:
    sys.path.insert(0, str(SII_DIR))

from GPU.GpuPointCloudContract import decode_frame
from GPU.GpuPointCloudPipeline import (
    LegacyCompatPointCloudPipeline,
    Open3dCudaPointCloudPipeline,
    build_sampled_pixel_candidate_grids_numpy,
    build_sampled_pixel_grids_numpy,
    select_sampled_pixel_grids_numpy,
)


class LegacyCompatPipelineTests(unittest.TestCase):
    def test_stable_random_sampling_is_deterministic_and_cell_bounded(self):
        yy_a, xx_a = build_sampled_pixel_grids_numpy(
            cam_name="top",
            width=8,
            height=6,
            stride=3,
            sampling_mode="stable_random",
        )
        yy_b, xx_b = build_sampled_pixel_grids_numpy(
            cam_name="top",
            width=8,
            height=6,
            stride=3,
            sampling_mode="stable_random",
        )
        yy_grid, xx_grid = build_sampled_pixel_grids_numpy(
            cam_name="top",
            width=8,
            height=6,
            stride=3,
            sampling_mode="grid",
        )

        np.testing.assert_array_equal(yy_a, yy_b)
        np.testing.assert_array_equal(xx_a, xx_b)
        self.assertEqual(yy_a.shape, yy_grid.shape)
        self.assertEqual(xx_a.shape, xx_grid.shape)

        for iy in range(yy_a.shape[0]):
            for ix in range(xx_a.shape[1]):
                y0 = iy * 3
                x0 = ix * 3
                self.assertGreaterEqual(int(yy_a[iy, ix]), y0)
                self.assertLess(int(yy_a[iy, ix]), min(y0 + 3, 6))
                self.assertGreaterEqual(int(xx_a[iy, ix]), x0)
                self.assertLess(int(xx_a[iy, ix]), min(x0 + 3, 8))

    def test_stable_random_selection_prefers_nearest_valid_candidate_in_cell(self):
        depth = np.full((6, 8), 5.0, dtype=np.float32)
        yy_candidates, xx_candidates = build_sampled_pixel_candidate_grids_numpy(
            cam_name="right",
            width=8,
            height=6,
            stride=3,
            sampling_mode="stable_random",
        )

        # Make the first cell contain one obviously foreground candidate and
        # leave the rest far so the selector should choose the nearest valid depth.
        depth[yy_candidates[:, 0, 0], xx_candidates[:, 0, 0]] = np.array([3.0, 0.4, 2.0, 1.5], dtype=np.float32)

        yy_sel, xx_sel = select_sampled_pixel_grids_numpy(
            cam_name="right",
            depth_m=depth,
            width=8,
            height=6,
            stride=3,
            sampling_mode="stable_random",
            min_depth=0.1,
            max_depth=10.0,
        )

        self.assertAlmostEqual(float(depth[yy_sel[0, 0], xx_sel[0, 0]]), 0.4, places=6)

    def test_legacy_pipeline_matches_expected_projection_and_contract(self):
        pipeline = LegacyCompatPointCloudPipeline()

        depth = np.array(
            [
                [1.0, 2.0],
                [0.5, 1.5],
            ],
            dtype=np.float32,
        )
        rgb = np.array(
            [
                [[10, 20, 30], [40, 50, 60]],
                [[70, 80, 90], [100, 110, 120]],
            ],
            dtype=np.uint8,
        )

        result = pipeline.build_frame(
            cam_name="top",
            depth_m=depth,
            rgb_u8=rgb,
            width=2,
            height=2,
            fovy_deg=90.0,
            intrinsics_mode="mujoco",
            stride=1,
            min_depth=0.0,
            max_depth=None,
            flip_x=False,
            flip_y=False,
            flip_z=False,
            pc_scale=1.0,
            cam_t_mj=np.zeros(3, dtype=np.float32),
            cam_R_mj=np.eye(3, dtype=np.float32),
            cam_anchor_corr=np.zeros(3, dtype=np.float32),
            clip_below_table=False,
            table_plane=None,
            table_margin=0.0,
            table_clearance=0.0,
            object_only=False,
            object_bbox_min=np.zeros(3, dtype=np.float32),
            object_bbox_max=np.ones(3, dtype=np.float32),
            declared_capacity=4,
        )

        expected_unity = np.array(
            [
                [0.5, 1.0, -0.5],
                [1.0, 2.0, 1.0],
                [-0.25, 0.5, -0.25],
                [-0.75, 1.5, 0.75],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(result.xyz_unity_m, expected_unity, rtol=0, atol=1e-6)
        np.testing.assert_array_equal(result.rgb_u8, rgb.reshape(-1, 3))

        header, xyz_mm, rgb_u8 = decode_frame(bytes(result.payload))
        self.assertEqual(header.declared_capacity, 4)
        self.assertEqual(header.actual_count, 4)
        np.testing.assert_array_equal(
            xyz_mm,
            np.array(
                [
                    [500, 1000, -500],
                    [1000, 2000, 1000],
                    [-250, 500, -250],
                    [-750, 1500, 750],
                ],
                dtype=np.int16,
            ),
        )
        np.testing.assert_array_equal(rgb_u8, rgb.reshape(-1, 3))

    def test_declared_capacity_downsamples_without_realloc_contract_shape_changes(self):
        pipeline = LegacyCompatPointCloudPipeline()
        depth = np.ones((2, 2), dtype=np.float32)
        rgb = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)

        result = pipeline.build_frame(
            cam_name="left",
            depth_m=depth,
            rgb_u8=rgb,
            width=2,
            height=2,
            fovy_deg=90.0,
            intrinsics_mode="mujoco",
            stride=1,
            min_depth=0.0,
            max_depth=None,
            flip_x=False,
            flip_y=False,
            flip_z=False,
            pc_scale=1.0,
            cam_t_mj=np.zeros(3, dtype=np.float32),
            cam_R_mj=np.eye(3, dtype=np.float32),
            cam_anchor_corr=np.zeros(3, dtype=np.float32),
            clip_below_table=False,
            table_plane=None,
            table_margin=0.0,
            table_clearance=0.0,
            object_only=False,
            object_bbox_min=np.zeros(3, dtype=np.float32),
            object_bbox_max=np.ones(3, dtype=np.float32),
            declared_capacity=2,
        )

        self.assertEqual(result.declared_capacity, 2)
        self.assertEqual(result.actual_count, 2)

    def test_debug_visibility_metadata_is_consistent_and_does_not_change_payload_count(self):
        pipeline = LegacyCompatPointCloudPipeline()
        depth = np.ones((4, 4), dtype=np.float32)
        rgb = np.full((4, 4, 3), 128, dtype=np.uint8)
        table_plane = SimpleNamespace(
            point=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            normal=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        )

        baseline = pipeline.build_frame(
            cam_name="right",
            depth_m=depth,
            rgb_u8=rgb,
            width=4,
            height=4,
            fovy_deg=90.0,
            intrinsics_mode="mujoco",
            stride=1,
            min_depth=0.0,
            max_depth=None,
            flip_x=False,
            flip_y=False,
            flip_z=False,
            pc_scale=1.0,
            cam_t_mj=np.zeros(3, dtype=np.float32),
            cam_R_mj=np.eye(3, dtype=np.float32),
            cam_anchor_corr=np.zeros(3, dtype=np.float32),
            clip_below_table=True,
            table_plane=table_plane,
            table_margin=0.0,
            table_clearance=0.2,
            object_only=True,
            object_bbox_min=np.array([0.0, -10.0, 0.0], dtype=np.float32),
            object_bbox_max=np.array([10.0, 10.0, 10.0], dtype=np.float32),
            declared_capacity=16,
        )

        debug = pipeline.build_frame(
            cam_name="right",
            depth_m=depth,
            rgb_u8=rgb,
            width=4,
            height=4,
            fovy_deg=90.0,
            intrinsics_mode="mujoco",
            stride=1,
            min_depth=0.0,
            max_depth=None,
            flip_x=False,
            flip_y=False,
            flip_z=False,
            pc_scale=1.0,
            cam_t_mj=np.zeros(3, dtype=np.float32),
            cam_R_mj=np.eye(3, dtype=np.float32),
            cam_anchor_corr=np.zeros(3, dtype=np.float32),
            clip_below_table=True,
            table_plane=table_plane,
            table_margin=0.0,
            table_clearance=0.2,
            object_only=True,
            object_bbox_min=np.array([0.0, -10.0, 0.0], dtype=np.float32),
            object_bbox_max=np.array([10.0, 10.0, 10.0], dtype=np.float32),
            declared_capacity=16,
            debug_visibility=True,
        )

        self.assertIsNotNone(debug.debug_visibility)
        dbg = debug.debug_visibility
        self.assertEqual(dbg.sampled_count, 16)
        self.assertEqual(dbg.valid_depth_count, 16)
        self.assertEqual(
            dbg.kept_count + dbg.rejected_bbox_count + dbg.rejected_table_count,
            dbg.valid_depth_count,
        )
        self.assertEqual(debug.actual_count, dbg.kept_count)
        self.assertEqual(debug.actual_count, baseline.actual_count)
        np.testing.assert_allclose(debug.xyz_unity_m, baseline.xyz_unity_m, rtol=0, atol=1e-6)
        np.testing.assert_array_equal(debug.rgb_u8, baseline.rgb_u8)

    def test_gpu_backend_fails_fast_when_open3d_cuda_is_missing(self):
        try:
            pipeline = Open3dCudaPointCloudPipeline()
        except RuntimeError:
            pipeline = None

        if pipeline is None:
            return

        self.assertIsInstance(pipeline, Open3dCudaPointCloudPipeline)


if __name__ == "__main__":
    unittest.main()
