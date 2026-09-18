#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

export PYTHONPATH="/home/user/open3d_build/Open3D/build/lib/python_package:${PYTHONPATH:-}"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
HOST_IP="${HOST_IP:-192.168.0.38}"
UNITY_NODE="${UNITY_NODE:-MQ3-2}"

WINDOWS="${WINDOWS:-3}"        # allowed: 3, 6, 9, 12, 15
MAX_SESSIONS="${MAX_SESSIONS:-$WINDOWS}"
TRAJECTORY="${TRAJECTORY:-intervene_base/teleop_logs/p1_traj_1772714528.npz}"
LAB_XML="${LAB_XML:-intervene_base/mujoco_scenes/working_scenes/with_soft_gripper/sii_scene_table_T_shape_multiwindow_fast.xml}"

FPS="${FPS:-30}"
FPS_IDLE="${FPS_IDLE:-10}"
PC_STRIDE="${PC_STRIDE:-4}"
PC_MAX_POINTS="${PC_MAX_POINTS:-80000}"
PC_FPS_CAP="${PC_FPS_CAP:-30}"
EXTRA_RUNTIME_ARGS="${EXTRA_RUNTIME_ARGS:-}"

echo "[MultiWindow] Python: $PYTHON_BIN"
echo "[MultiWindow] Host IP: $HOST_IP"
echo "[MultiWindow] Quest node: $UNITY_NODE"
echo "[MultiWindow] Windows: $WINDOWS"
echo "[MultiWindow] XML: $LAB_XML"
echo "[MultiWindow] Trajectory: $TRAJECTORY"
echo "[MultiWindow] This is multi-session sim replay; it does not connect the real robot."

CMD=(
  "$PYTHON_BIN" SimPublisher/sii/integration_v1/multi_session_launcher.py
  --trajectories "$TRAJECTORY"
  --windows "$WINDOWS"
  --max_sessions "$MAX_SESSIONS"
  --xml "$LAB_XML"
  --host "$HOST_IP"
  --unity_node "$UNITY_NODE"
  --bind_ip 0.0.0.0
  --visible_geoms_groups 2
  --fps "$FPS"
  --fps_idle "$FPS_IDLE"
  --with_pc
  --pc_stride "$PC_STRIDE"
  --pc_max_points "$PC_MAX_POINTS"
  --pc_fps_cap "$PC_FPS_CAP"
  --unified_log
)

if [[ -n "$EXTRA_RUNTIME_ARGS" ]]; then
  CMD+=(--extra_runtime_args "$EXTRA_RUNTIME_ARGS")
fi

exec "${CMD[@]}"
