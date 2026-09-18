#!/usr/bin/env bash
set -euo pipefail

LAB_XML="${LAB_XML:-intervene_base/mujoco_scenes/working_scenes/with_soft_gripper/sii_scene_table_T_shape_multiwindow_fast.xml}"
TRAJECTORY="${TRAJECTORY:-intervene_base/teleop_logs/p1_traj_1772714528.npz}"
HOST_IP="${HOST_IP:-192.168.0.38}"
UNITY_NODE="${UNITY_NODE:-MQ3-2}"
ROBOT_KEY="${ROBOT_KEY:-p1}"

if ! command -v adb >/dev/null 2>&1; then
  echo "[USB] adb not found in PATH" >&2
  exit 1
fi

echo "[USB] Checking Quest connection..."
adb devices

echo "[USB] Configuring adb reverse..."
adb reverse tcp:7741 tcp:7741
adb reverse tcp:7740 tcp:7740
adb reverse tcp:7721 tcp:7721
adb reverse --list

echo "[USB] Starting intervention runtime..."
.venv/bin/python ./SimPublisher/sii/integration_v1/intervention_vr_runtime.py \
  --xml "$LAB_XML" \
  --trajectory "$TRAJECTORY" \
  --host "$HOST_IP" \
  --unity_node "$UNITY_NODE" \
  --bind_ip 0.0.0.0 \
  --visible_geoms_groups 2 \
  --fps 30 --w 960 --h 720 \
  --cams wrist top right left \
  --rgb_cams wrist \
  --pc --pc_cams top right left \
  --pc_sampling stable_random \
  --pc_stride 8 --pc_max_points 50000 \
  --pc_min_depth 0.08 --pc_max_depth 2.8 \
  --pc_intrinsics_mode mujoco \
  --pc_clip_below_table --pc_table_clearance 0.02 --pc_top_table_clearance 0.035 \
  --pc_object_only --pc_object_bbox_min 0.30 -0.45 -0.07 --pc_object_bbox_max 1.00 0.45 0.50 \
  --cam_extrinsics_profile lab_standard \
  --pc_no_anchor_auto_translate \
  --log_every 30 \
  --show_mujoco_window \
  --object_control \
  --object_step_xy 0.01 --object_step_z 0.008 --object_move_rate_hz 10 \
  --mirror_robot \
  --transport usb \
  --usb_auto_reverse \
  --usb_extra_reverse_ports 7721 \
  --robot_key "$ROBOT_KEY" \
  --control_hz 120 \
  --start_paused
