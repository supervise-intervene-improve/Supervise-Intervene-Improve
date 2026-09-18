#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

set +u
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate base
source "/home/user/Downloads/Intervention_IL_AR/.venv/bin/activate"
set -u

export PYTHONPATH="/home/user/open3d_build/Open3D/build/lib/python_package:${PYTHONPATH:-}"
export POLYMETIS_LIB_DIR="${POLYMETIS_LIB_DIR:-/home/user/miniconda3/envs/polymetis/lib}"
export LD_LIBRARY_PATH="$POLYMETIS_LIB_DIR:${LD_LIBRARY_PATH:-}"

if [[ -f "$POLYMETIS_LIB_DIR/libtorchscript_pinocchio.so" ]]; then
  ln -sf "$POLYMETIS_LIB_DIR/libtorchscript_pinocchio.so" ./libtorchscript_pinocchio.so
fi
if [[ -f "$POLYMETIS_LIB_DIR/libtorchrot.so" ]]; then
  ln -sf "$POLYMETIS_LIB_DIR/libtorchrot.so" ./libtorchrot.so
fi

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
HOST_IP="${HOST_IP:-192.168.0.38}"
UNITY_NODE="${UNITY_NODE:-MQ3-2}"
ROBOT_KEY="${ROBOT_KEY:-p1}"

LAB_XML="${LAB_XML:-intervene_base/mujoco_scenes/working_scenes/with_soft_gripper/sii_scene_table_T_shape_multiwindow_fast.xml}"
TRAJECTORY="${TRAJECTORY:-intervene_base/teleop_logs/p1_traj_1772714528.npz}"
REPLAN_OUTPUT_DIR="${REPLAN_OUTPUT_DIR:-intervene_base/teleop_logs/replans}"

# Default: safest first test. Local MuJoCo window + keyboard controls, no Quest dependency.
# Set QUEST=1 to stream the full camera/point-cloud topics to the Quest too.
QUEST="${QUEST:-0}"

BASE_ARGS=(
  SimPublisher/sii/integration_v1/intervention_vr_runtime.py
  --xml "$LAB_XML"
  --trajectory "$TRAJECTORY"
  --host "$HOST_IP"
  --unity_node "$UNITY_NODE"
  --bind_ip 0.0.0.0
  --visible_geoms_groups 2
  --mirror_robot
  --robot_key "$ROBOT_KEY"
  --control_hz 120
  --start_paused
  --replan_output_dir "$REPLAN_OUTPUT_DIR"
  --show_mujoco_window
  --mujoco_window_cam top
  --log_every 30
)

LOCAL_ARGS=(
  --no_mujoco_publisher
  --no_xr_device
  --fps 15
  --fps_idle 5
  --w 640
  --h 480
  --cams top
  --rgb_cams top
)

QUEST_ARGS=(
  --fps 30
  --fps_idle 10
  --w 960
  --h 720
  --cams wrist top right left
  --rgb_cams wrist top
  --pc
  --pc_cams top right left
  --pc_sampling stable_random
  --pc_stride 4
  --pc_max_points 80000
  --pc_min_depth 0.08
  --pc_max_depth 2.8
  --pc_intrinsics_mode mujoco
  --pc_clip_below_table
  --pc_table_clearance 0.02
  --pc_top_table_clearance 0.035
  --pc_object_only
  --pc_object_bbox_min 0.30 -0.45 -0.07
  --pc_object_bbox_max 1.00 0.45 0.50
  --cam_extrinsics_profile lab_standard
  --pc_no_anchor_auto_translate
)

echo "[Launch] Python: $PYTHON_BIN"
echo "[Launch] Host IP: $HOST_IP"
echo "[Launch] Robot: $ROBOT_KEY"
echo "[Launch] XML: $LAB_XML"
echo "[Launch] Trajectory: $TRAJECTORY"
echo "[Launch] Output dir: $REPLAN_OUTPUT_DIR"
echo "[Launch] Starts PAUSED. Robot will not move until you resume."

if [[ "$QUEST" == "1" ]]; then
  echo "[Launch] Mode: Quest streaming + local MuJoCo window"
  exec "$PYTHON_BIN" "${BASE_ARGS[@]}" "${QUEST_ARGS[@]}"
else
  echo "[Launch] Mode: local MuJoCo window only"
  exec "$PYTHON_BIN" "${BASE_ARGS[@]}" "${LOCAL_ARGS[@]}"
fi
