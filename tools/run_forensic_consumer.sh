#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SIMPUBLISHER_ROOT="${SIMPUBLISHER_ROOT:?set SIMPUBLISHER_ROOT to the consumer checkout}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
WINDOWS="${WINDOWS:-3}"
HOST_IP="${HOST_IP:-}"
OPEN3D_PYTHONPATH="${FORENSICS_OPEN3D_PYTHONPATH:-$HOME/open3d_build/Open3D/build/lib/python_package}"
export PYTHONPATH="$OPEN3D_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "$HOST_IP" ]]; then
  HOST_IP="$(ip -4 -o addr show scope global 2>/dev/null | awk '$4 ~ /^192\.168\./ {sub(/\/.*/, "", $4); print $4; exit}')"
fi
if [[ -z "$HOST_IP" ]]; then
  echo "[ForensicConsumer][ERROR] HOST_IP is unset and no 192.168.x address was found." >&2
  exit 2
fi

EXTRAS=(
  --policy_state_timeout_s 1.0
  --pc_backend "${PC_BACKEND:-gpu}"
  --pc_worker_threads "${PC_WORKER_THREADS:-3}"
  --pc_fps_cap "${PC_FPS_CAP:-30}"
)

CMD=(
  "$PYTHON_BIN" "$SIMPUBLISHER_ROOT/sii/integration_v1/multi_session_launcher.py"
  --trajectories "${TRAJ:?set TRAJ}"
  --windows "$WINDOWS"
  --max_sessions "$WINDOWS"
  --xml "${LAB_XML:?set LAB_XML}"
  --host "$HOST_IP"
  --unity_node "${UNITY_NODE:-MQ3-2}"
  --bind_ip 0.0.0.0
  --visible_geoms_groups 2
  --fps "${FPS:-60}"
  --fps_idle "${FPS_IDLE:-10}"
  --jpg_quality "${JPG_QUALITY:-75}"
  --post_replan_mode preview_replay
  --max_restarts 2
  --base_topic_port "${BASE_TOPIC_PORT:-7741}"
  --base_service_port "${BASE_SERVICE_PORT:-7740}"
  --port_step "${POLICY_PORT_STEP:-10}"
  --unified_log
  --metrics
  --metrics_window_s "${METRICS_WINDOW_S:-10}"
  --log_dir "${FORENSICS_RUNTIME_LOG_DIR:?set FORENSICS_RUNTIME_LOG_DIR}"
  --with_pc
  --pc_max_points "${PC_MAX_POINTS:-80000}"
  --pc_sampling grid
  --pc_stride "${PC_STRIDE:-3}"
  --pc_width 640
  --pc_height 480
  --pc_active_only
  --selected_sensor_activation
  --policy_state_base_port "${POLICY_STATE_BASE_PORT:-8066}"
  --policy_cmd_base_port "${POLICY_CMD_BASE_PORT:-8065}"
  --policy_port_step "${POLICY_PORT_STEP:-10}"
  --policy_host 127.0.0.1
  --extra_runtime_args "${EXTRAS[*]}"
)

# NOTE: the --ood_enabled/--ood_owner passthrough was removed with the OOD auto-pause
# supervisor; multi_session_launcher.py no longer accepts those flags.

printf '[ForensicConsumer] CMD:'
printf ' %q' "${CMD[@]}"
printf '\n'
cd "$FORENSICS_RUNTIME_LOG_DIR"
exec "${CMD[@]}"
