#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

set +u
CONDA_SH=""
for candidate in "$HOME/miniforge3/etc/profile.d/conda.sh" "$HOME/miniconda3/etc/profile.d/conda.sh"; do
  if [[ -f "$candidate" ]]; then
    CONDA_SH="$candidate"
    break
  fi
done
if [[ -n "$CONDA_SH" ]]; then
  source "$CONDA_SH"
  conda activate base
else
  echo "[MultiWindow][WARN] No conda.sh found; using current shell environment."
fi
VR_VENV="${VR_VENV:-${SCRIPT_DIR}/.venv}"
if [[ -f "${VR_VENV}/bin/activate" ]]; then
  source "${VR_VENV}/bin/activate"
else
  echo "[MultiWindow][WARN] VR venv not found at ${VR_VENV}; using current Python environment."
fi
set -u

# OPEN3D_PYTHON_PACKAGE: optional path to a locally built CUDA Open3D python package
# (e.g. <open3d_build>/lib/python_package). Leave unset if Open3D is pip-installed in the venv.
if [[ -n "${OPEN3D_PYTHON_PACKAGE:-}" ]]; then
  export PYTHONPATH="${OPEN3D_PYTHON_PACKAGE}:${PYTHONPATH:-}"
fi
if [[ -z "${POLYMETIS_LIB_DIR:-}" ]]; then
  if [[ -d "$HOME/miniforge3/envs/polymetis/lib" ]]; then
    POLYMETIS_LIB_DIR="$HOME/miniforge3/envs/polymetis/lib"
  else
    POLYMETIS_LIB_DIR="$HOME/miniconda3/envs/polymetis/lib"
  fi
fi
export POLYMETIS_LIB_DIR
export LD_LIBRARY_PATH="$POLYMETIS_LIB_DIR:${LD_LIBRARY_PATH:-}"

if [[ -f "$POLYMETIS_LIB_DIR/libtorchscript_pinocchio.so" ]]; then
  ln -sf "$POLYMETIS_LIB_DIR/libtorchscript_pinocchio.so" ./libtorchscript_pinocchio.so
fi
if [[ -f "$POLYMETIS_LIB_DIR/libtorchrot.so" ]]; then
  ln -sf "$POLYMETIS_LIB_DIR/libtorchrot.so" ./libtorchrot.so
fi

# Auto-detect the WiFi IP the Quest can reach. `|| true` is REQUIRED: this script runs
# under `set -euo pipefail`, so when the WiFi is down grep matches nothing, returns 1,
# pipefail propagates it and set -e kills the script BEFORE any output — the operator
# just sees a bare "exited with status 1" and no reason.
_auto_ip=$(ip addr show | grep 'inet 192\.168\.' | awk '{print $2}' | cut -d'/' -f1 | head -1 || true)
export HOST_IP="${HOST_IP:-${_auto_ip}}"
if [[ -z "${HOST_IP}" ]]; then
  echo "[MultiWindow][ERROR] No 192.168.x.y address found — the PC's WiFi is not connected." >&2
  echo "[MultiWindow]         The Quest can only be reached over WiFi; the Ethernet subnet is unreachable from it." >&2
  _other_ips=$(ip -4 addr show scope global 2>/dev/null | grep -oP 'inet \K[\d.]+' | paste -sd' ' || true)
  echo "[MultiWindow]         Interfaces currently up: ${_other_ips:-<none>}" >&2
  echo "[MultiWindow]  FIX:   connect the PC to the Quest's WiFi router, then re-run." >&2
  echo "[MultiWindow]  OR:    override explicitly -> HOST_IP=192.168.0.x bash ./run_multi_window_robot.sh" >&2
  exit 1
fi
export UNITY_NODE="${UNITY_NODE:-MQ3-2}"
export WINDOWS="${WINDOWS:-3}"

# SINGLE=1 → standalone single-session mode: launch ONE session, bypass the
# multi-window selector grid (the Quest auto-enters single view). Overrides WINDOWS.
SINGLE="${SINGLE:-0}"
# ROBOT=1 → mirror the real Franka arm (arms only on A/X via --mirror_on_select_only).
# Default 0 = sim-only. Equivalent to run_multi_window_robot_mirror.sh, as an inline knob.
ROBOT="${ROBOT:-0}"
# Must match app.py's INTERVENE_ROBOT_KEY default (p4), otherwise the VR runtime and the
# policy process target different machines. run_main_policy.sh exports both names from one
# key; this default only applies when this script is run standalone.
export ROBOT_KEY="${ROBOT_KEY:-${INTERVENE_ROBOT_KEY:-p4}}"
export INTERVENE_ROBOT_KEY="${INTERVENE_ROBOT_KEY:-${ROBOT_KEY}}"
export CONTROL_HZ="${CONTROL_HZ:-120}"
export REPLAN_OUTPUT_DIR="${REPLAN_OUTPUT_DIR:-intervene_base/teleop_logs/replans}"

DEFAULT_TRAJ_DIR="${DEFAULT_TRAJ_DIR:-intervene_base/INTERVENTION_DATA}"
DEFAULT_TRAJ_GLOB="${DEFAULT_TRAJ_GLOB:-policy_episode_*.npz}"
TRAJ="${TRAJ:-}"
# Performance-tuned scene: offwidth/offheight 640x480, multiccd disabled,
# implicitfast integrator, timestep 0.002, castshadow off. Drop-in for the
# heavy sii_scene_table_T_shape.xml (same gripper, objects, cameras).
DEFAULT_LAB_XML="intervene_base/mujoco_scenes/working_scenes/with_soft_gripper/sii_scene_table_T_shape.xml"
if [[ ! -f "$DEFAULT_LAB_XML" ]]; then
  DEFAULT_LAB_XML="intervene_base/mujoco_scenes/working_scenes/with_soft_gripper/sii_scene_table_T_shape_multiwindow_fast.xml"
fi
export LAB_XML="${LAB_XML:-$DEFAULT_LAB_XML}"

PYTHON_BIN="${PYTHON_BIN:-$(which python)}"
if [[ -z "${FPS:-}" ]]; then
  # Selected-session sensor target. Keep grid sessions at FPS_IDLE=10; 60 Hz leaves
  # enough render budget for all three non-round-robin PC cameras on the nine-session run.
  FPS=60
fi
if [[ -z "${FPS_IDLE:-}" ]]; then
  FPS_IDLE=10
fi
if [[ -z "${JPG_QUALITY:-}" ]]; then
  JPG_QUALITY=75
fi
# RGB=1 → switch the single session view from point clouds to a 4-camera
# diamond RGB panel layout (top/left/right/wrist), anchored to the same
# controllable/persisted scene anchor. Mutually exclusive with WITH_PC: when
# RGB=1, point clouds are never engaged (default WITH_PC to 0 unless the
# caller explicitly set it, in which case we refuse to launch ambiguous state).
RGB="${RGB:-0}"
if [[ "$RGB" == "1" || "$RGB" == "true" || "$RGB" == "TRUE" ]]; then
  if [[ -n "${WITH_PC:-}" && ( "$WITH_PC" == "1" || "$WITH_PC" == "true" || "$WITH_PC" == "TRUE" ) ]]; then
    echo "[MultiWindow][ERROR] RGB=1 and WITH_PC=1 are mutually exclusive (single view is either point clouds or RGB camera panels, never both)." >&2
    exit 1
  fi
  WITH_PC=0
fi
if [[ -z "${WITH_PC:-}" ]]; then
  WITH_PC=1
fi
# MC_ACTIVE=1 → the motion-controller condition: the Quest's RIGHT controller drives the
# arm during an intervention. Forwarded to the launcher purely so it lands in the discovery
# beacon; the headset then reserves the right controller for MC while an intervention is
# live (no panel select / risk-bar navigate / A / B), releasing it on accept or reject.
# Inherited from run_main_policy.sh, which exports it alongside RGB/WITH_PC.
MC_ACTIVE="${MC_ACTIVE:-0}"
# PC_MAX_POINTS: points per camera per frame. 200k is right for a handful of windows; at
# 9+ concurrent sessions it is the dominant ACTIVE cost (measured 2026-07-29: 31.7% of
# ACTIVE frames over a 16.7 ms budget, max tick 238 ms). The project already used 80k for
# 15 windows. Scaled by window count unless the caller sets it explicitly.
if [[ -z "${PC_MAX_POINTS:-}" ]]; then
  if (( ${WINDOWS:-3} >= 9 )); then
    PC_MAX_POINTS=80000
  else
    PC_MAX_POINTS=200000
  fi
fi
# PC_STRIDE: sample every Nth depth pixel. 2=dense, 4=half, 8=sparse. Higher = faster + fewer points.
PC_STRIDE="${PC_STRIDE:-3}"
# PC_WIDTH / PC_HEIGHT: resolution of the depth render fed into the PC pipeline.
PC_WIDTH="${PC_WIDTH:-640}"
PC_HEIGHT="${PC_HEIGHT:-480}"
# PC_MAX_SOURCE_AGE_S: hard floor on per-camera refresh under PC_ROUND_ROBIN. Must stay
# well under Unity's GpuMergedPointCloudLoader.SourceIdleClearSeconds, or a slow tick
# lets a camera go silent long enough for Unity to DELETE that point-cloud source (the
# "cameras drop out one by one and come back" loop). 0 = no floor (pure cycling).
PC_MAX_SOURCE_AGE_S="${PC_MAX_SOURCE_AGE_S:-0.4}"
PC_ACTIVE_ONLY="${PC_ACTIVE_ONLY:-1}"
SELECTED_SENSOR_ACTIVATION="${SELECTED_SENSOR_ACTIVATION:-1}"
POST_REPLAN_MODE="${POST_REPLAN_MODE:-preview_replay}"
MAX_RESTARTS="${MAX_RESTARTS:-3}"
# BASE_TOPIC_PORT: first topic port (default 7741). Bump by +200 or similar to sidestep
# leftover port conflicts when a reboot is not practical.
BASE_TOPIC_PORT="${BASE_TOPIC_PORT:-7741}"
# Per-session INDEPENDENT policy wiring (set by run_main_policy.sh in WINDOWS mode):
# session i mirrors the policy instance at state port POLICY_STATE_BASE_PORT + i*STEP and
# forwards INTERVENE/CANCEL to cmd port POLICY_CMD_BASE_PORT + i*STEP. Empty = legacy
# behavior (any policy ports come via EXTRA_RUNTIME_ARGS, identical for all sessions).
POLICY_CMD_BASE_PORT="${POLICY_CMD_BASE_PORT:-}"
POLICY_STATE_BASE_PORT="${POLICY_STATE_BASE_PORT:-}"
POLICY_PORT_STEP="${POLICY_PORT_STEP:-10}"
POLICY_HOST="${POLICY_HOST:-127.0.0.1}"
# Quality-neutral PC sampling: 'grid' is ~4x cheaper than 'stable_random' at the
# same density/count. Keep 'grid' unless you specifically want jittered sampling.
PC_SAMPLING="${PC_SAMPLING:-grid}"
# PERF_LOG=1 prints per-frame [Perf] render/total/budget; use it to measure.
PERF_LOG="${PERF_LOG:-0}"
# PC_ROUND_ROBIN=1 renders one PC camera per tick (top->right->left) so the main
# loop advances the robot mirror ~3x more often -> smoother real<->sim. Spatial
# PC density unchanged; each camera just refreshes at fps/3. Try if robot lags.
PC_ROUND_ROBIN="${PC_ROUND_ROBIN:-0}"
# PC_WORKER_THREADS: parallel GPU build threads in PCWorkerThread.
# 1 = sequential (safe default). 3 = one thread per camera, all 3 build
# simultaneously — reduces PC latency from 3x to ~1x build cycle.
# Scale with the cohort: every parallel build holds its own GPU intermediates, and all
# WINDOWS runtimes share one card with the WINDOWS policy processes. Measured at 9 windows
# with 3 threads: 702 Open3D "CUDA runtime error: out of memory", six of nine sessions
# publishing ZERO points. Latency on a working cloud beats 0 Hz, so serialise when the
# cohort is large. An explicit PC_WORKER_THREADS= still wins.
if (( ${WINDOWS:-3} >= 4 )); then
  PC_WORKER_THREADS="${PC_WORKER_THREADS:-1}"
else
  PC_WORKER_THREADS="${PC_WORKER_THREADS:-3}"
fi
# Cap each selected camera at the end-to-end acceptance target. Publishing faster than
# the Quest can decode and upload only creates receiver pressure; the known-good build
# looked smooth at 15-22 Hz, while this keeps a measured 30 Hz target.
PC_FPS_CAP="${PC_FPS_CAP:-30}"
# METRICS=1 writes windowed perf rows (avg/p50/p95/p99 + per-topic publish Hz) to
# session_logs/metrics_S<ii>_<ts>.jsonl. Unlike PERF_LOG (one line per frame, no
# aggregation), these rows are what tools/perf_report.py consumes.
METRICS="${METRICS:-0}"
METRICS_WINDOW_S="${METRICS_WINDOW_S:-10}"
# METRICS_SUMMARY=1 additionally prints the [PerfSummary] block into the session log.
METRICS_SUMMARY="${METRICS_SUMMARY:-0}"
# NOTE: OOD_OWNER / OOD_ENABLED retired with the OOD auto-pause supervisor. OOD is now
# a per-session out-of-distribution SCENE, decided in app.py; nothing pauses anything, so
# there is no supervisor to own and nothing to advertise. See OOD_MIN_STATES/OOD_MAX_STATES.
# PEER_FALLBACK_GRACE_S: how long peers must stay >= threshold before the peer count is
# trusted as a fallback for a dropped ENTER_SINGLE. Combined view keeps a subscriber on
# every session, so an instant peer test used to promote UNSELECTED sessions to full
# point-cloud rendering. Set 0 for strict selection (ENTER/EXIT_SINGLE only).
# DEFAULT 0: measured 2026-07-29 at 9 windows, the old 3.0 let an unselected session
# self-promote ("[AdaptiveRate][Fallback] ... peers=4 ... selected=False"), so two sessions
# rendered full 200k-point clouds at once and the selected one missed its frame budget.
# With SELECTED_SENSOR_ACTIVATION=1 the ENTER_SINGLE command is authoritative and the
# fallback only masks a delivery bug that has since been fixed (socket pre-warm + resend).
PEER_FALLBACK_GRACE_S="${PEER_FALLBACK_GRACE_S:-0}"
# Fail-safe for a lost EXIT_SINGLE. A selected runtime may warm without PC subscribers for
# this grace period; after that it idles until the subscribers return. 0 disables the lease.
SELECTED_PEER_IDLE_GRACE_S="${SELECTED_PEER_IDLE_GRACE_S:-5}"
# RGB_THUMBNAIL_FPS: cap the selector-grid feed (front/rgb) even when ACTIVE. The
# thumbnail is small; holding it at 10 Hz removes a 640x480 render per tick from the
# selected session. 0 = uncapped (old behaviour).
RGB_THUMBNAIL_FPS="${RGB_THUMBNAIL_FPS:-10}"
EXTRA_RUNTIME_ARGS="${EXTRA_RUNTIME_ARGS:-}"

if [[ -z "${PC_BACKEND:-}" ]]; then
  if [[ "$WITH_PC" == "1" || "$WITH_PC" == "true" || "$WITH_PC" == "TRUE" ]]; then
    if "$PYTHON_BIN" -c 'import open3d as o3d; cuda = getattr(getattr(o3d, "core", None), "cuda", None); raise SystemExit(0 if cuda and cuda.is_available() else 1)' >/dev/null 2>&1; then
      PC_BACKEND="gpu"
    else
      PC_BACKEND="legacy"
    fi
  else
    PC_BACKEND="legacy"
  fi
fi

# ROBOT=1 → prepend the real-robot mirror args (same set as run_multi_window_robot_mirror.sh).
if [[ "$ROBOT" == "1" || "$ROBOT" == "true" || "$ROBOT" == "TRUE" ]]; then
  MIRROR_ARGS="--mirror_robot --mirror_on_select_only --robot_key $ROBOT_KEY --control_hz $CONTROL_HZ --replan_output_dir $REPLAN_OUTPUT_DIR"
  if [[ -n "$EXTRA_RUNTIME_ARGS" ]]; then
    EXTRA_RUNTIME_ARGS="$MIRROR_ARGS $EXTRA_RUNTIME_ARGS"
  else
    EXTRA_RUNTIME_ARGS="$MIRROR_ARGS"
  fi
fi

# Compose flags forwarded to each runtime: measurement/round-robin toggles plus
# any caller-provided EXTRA_RUNTIME_ARGS (e.g. the mirror script's robot args).
RUNTIME_EXTRAS=""
RUNTIME_EXTRAS="$RUNTIME_EXTRAS --peer_fallback_grace_s $PEER_FALLBACK_GRACE_S"
RUNTIME_EXTRAS="$RUNTIME_EXTRAS --selected_peer_idle_grace_s $SELECTED_PEER_IDLE_GRACE_S"
RUNTIME_EXTRAS="$RUNTIME_EXTRAS --rgb_thumbnail_fps $RGB_THUMBNAIL_FPS"
RUNTIME_EXTRAS="$RUNTIME_EXTRAS --pc_max_source_age_s $PC_MAX_SOURCE_AGE_S"
if [[ "$PERF_LOG" == "1" || "$PERF_LOG" == "true" || "$PERF_LOG" == "TRUE" ]]; then
  RUNTIME_EXTRAS="$RUNTIME_EXTRAS --perf_log"
fi
if [[ "$PC_ROUND_ROBIN" == "1" || "$PC_ROUND_ROBIN" == "true" || "$PC_ROUND_ROBIN" == "TRUE" ]]; then
  RUNTIME_EXTRAS="$RUNTIME_EXTRAS --pc_round_robin"
fi
if [[ "$WITH_PC" == "1" || "$WITH_PC" == "true" || "$WITH_PC" == "TRUE" ]]; then
  RUNTIME_EXTRAS="$RUNTIME_EXTRAS --pc_backend $PC_BACKEND"
  if [[ -n "${PC_WORKER_THREADS:-}" && "${PC_WORKER_THREADS}" != "1" ]]; then
    RUNTIME_EXTRAS="$RUNTIME_EXTRAS --pc_worker_threads $PC_WORKER_THREADS"
  fi
  if [[ -n "${PC_FPS_CAP:-}" ]]; then
    RUNTIME_EXTRAS="$RUNTIME_EXTRAS --pc_fps_cap $PC_FPS_CAP"
  fi
fi
if [[ -n "$EXTRA_RUNTIME_ARGS" ]]; then
  RUNTIME_EXTRAS="$RUNTIME_EXTRAS $EXTRA_RUNTIME_ARGS"
fi
RUNTIME_EXTRAS="$(echo "$RUNTIME_EXTRAS" | sed -e 's/^ *//' -e 's/ *$//')"

echo "[MultiWindow] Python: $PYTHON_BIN"
echo "[MultiWindow] Host IP: $HOST_IP"
echo "[MultiWindow] Quest node: $UNITY_NODE"
if [[ "$SINGLE" == "1" || "$SINGLE" == "true" || "$SINGLE" == "TRUE" ]]; then
  echo "[MultiWindow] Mode: SINGLE (1 session, grid bypassed → Quest auto-enters single view)"
else
  echo "[MultiWindow] Mode: MULTI ($WINDOWS windows)"
fi
if [[ "$ROBOT" == "1" || "$ROBOT" == "true" || "$ROBOT" == "TRUE" ]]; then
  echo "[MultiWindow] Robot: ON (key=$ROBOT_KEY control_hz=$CONTROL_HZ; arms only on A/X)"
else
  echo "[MultiWindow] Robot: OFF (sim-only)"
fi
echo "[MultiWindow] XML: $LAB_XML"
if [[ "$SINGLE" == "1" || "$SINGLE" == "true" || "$SINGLE" == "TRUE" ]]; then
  TRAJ_COUNT=1
else
  TRAJ_COUNT="$WINDOWS"
fi

TRAJ_ARGS=()
if [[ -n "$TRAJ" ]]; then
  # Preserve the existing override behavior: a caller-provided TRAJ may contain
  # one or more space-separated .npz paths.
  read -r -a TRAJ_ARGS <<< "$TRAJ"
else
  if [[ ! -d "$DEFAULT_TRAJ_DIR" ]]; then
    # Recorded policy episodes are not shipped with the repository; fall back to the
    # bundled T-shape reset trajectory (replicated across all windows by the launcher).
    FALLBACK_TRAJ="${FALLBACK_TRAJ:-intervene_base/RESET_NPZs/TSHAPE/p1_ep_0001_1778506313_trimmed.npz}"
    if [[ ! -f "$FALLBACK_TRAJ" ]]; then
      echo "[MultiWindow][ERROR] Default trajectory directory not found: $DEFAULT_TRAJ_DIR (and no $FALLBACK_TRAJ)" >&2
      exit 1
    fi
    echo "[MultiWindow][WARN] $DEFAULT_TRAJ_DIR not found; using $FALLBACK_TRAJ for all windows." >&2
    DEFAULT_TRAJ_DIR="$(dirname "$FALLBACK_TRAJ")"
    DEFAULT_TRAJ_GLOB="$(basename "$FALLBACK_TRAJ")"
    TRAJ_COUNT=1  # launcher --windows replicates the single trajectory
  fi
  mapfile -t DEFAULT_TRAJ_ARGS < <(find "$DEFAULT_TRAJ_DIR" -maxdepth 1 -type f -name "$DEFAULT_TRAJ_GLOB" | sort)
  TRAJ_ARGS=("${DEFAULT_TRAJ_ARGS[@]:0:TRAJ_COUNT}")
  if (( ${#TRAJ_ARGS[@]} < TRAJ_COUNT )); then
    echo "[MultiWindow][ERROR] Found only ${#TRAJ_ARGS[@]} trajectory file(s), need $TRAJ_COUNT." >&2
    echo "[MultiWindow]         dir=$DEFAULT_TRAJ_DIR pattern=$DEFAULT_TRAJ_GLOB" >&2
    exit 1
  fi
fi

echo "[MultiWindow] Trajectories (${#TRAJ_ARGS[@]}):"
printf '  %s\n' "${TRAJ_ARGS[@]}"
echo "[MultiWindow] FPS: $FPS (idle=$FPS_IDLE, jpg_quality=$JPG_QUALITY)"
echo "[MultiWindow] Grid PC streams: $WITH_PC"
echo "[MultiWindow] RGB diamond panel mode: $RGB"
echo "[MultiWindow] Motion controller (right-controller lock during intervention): $MC_ACTIVE"
echo "[MultiWindow] PC active-only: $PC_ACTIVE_ONLY"
echo "[MultiWindow] Selected-only sensor activation: $SELECTED_SENSOR_ACTIVATION"
echo "[MultiWindow] Post-replan mode: $POST_REPLAN_MODE"
echo "[MultiWindow] Max restarts: $MAX_RESTARTS"
echo "[MultiWindow] PC sampling: $PC_SAMPLING  backend: $PC_BACKEND  perf_log: $PERF_LOG  round_robin: $PC_ROUND_ROBIN"
if [[ -n "$RUNTIME_EXTRAS" ]]; then
  echo "[MultiWindow] Forwarded runtime args: $RUNTIME_EXTRAS"
else
  echo "[MultiWindow] Forwarded runtime args: <none> (sim-only, no real robot mirror)"
fi

CMD=(
  "$PYTHON_BIN" SimPublisher/sii/integration_v1/multi_session_launcher.py
  --trajectories "${TRAJ_ARGS[@]}"
)
if [[ "$SINGLE" == "1" || "$SINGLE" == "true" || "$SINGLE" == "TRUE" ]]; then
  CMD+=(--single --max_sessions 1)
else
  CMD+=(--windows "$WINDOWS" --max_sessions "$WINDOWS")
fi
CMD+=(
  --xml "$LAB_XML"
  --host "$HOST_IP"
  --unity_node "$UNITY_NODE"
  --bind_ip 0.0.0.0
  --visible_geoms_groups 2
  --fps "$FPS"
  --fps_idle "$FPS_IDLE"
  --jpg_quality "$JPG_QUALITY"
  --post_replan_mode "$POST_REPLAN_MODE"
  --max_restarts "$MAX_RESTARTS"
  --base_topic_port "$BASE_TOPIC_PORT"
  --unified_log
)

# GRID_CAPACITY: number of selector cells shown on the headset, independent of the number of
# active sessions (e.g. WINDOWS=3 on a nine-cell board -> GRID_CAPACITY=9).
GRID_CAPACITY="${GRID_CAPACITY:-}"
if [[ -n "$GRID_CAPACITY" ]]; then
  CMD+=(--grid_capacity "$GRID_CAPACITY")
  echo "[MultiWindow] Grid capacity: $GRID_CAPACITY cells"
fi

# Windowed performance metrics -> session_logs/metrics_S<ii>_<ts>.jsonl.
if [[ "$METRICS" == "1" || "$METRICS" == "true" || "$METRICS" == "TRUE" ]]; then
  CMD+=(--metrics --metrics_window_s "$METRICS_WINDOW_S")
  if [[ "$METRICS_SUMMARY" == "1" || "$METRICS_SUMMARY" == "true" || "$METRICS_SUMMARY" == "TRUE" ]]; then
    CMD+=(--metrics_summary)
  fi
  echo "[MultiWindow] Metrics ON: window=${METRICS_WINDOW_S}s -> session_logs/metrics_S*.jsonl"
fi

# Per-session independent policy ports (session i ↔ policy i).
if [[ -n "$POLICY_STATE_BASE_PORT" ]]; then
  CMD+=(--policy_state_base_port "$POLICY_STATE_BASE_PORT" --policy_port_step "$POLICY_PORT_STEP" --policy_host "$POLICY_HOST")
  if [[ -n "$POLICY_CMD_BASE_PORT" ]]; then
    CMD+=(--policy_cmd_base_port "$POLICY_CMD_BASE_PORT")
  fi
  echo "[MultiWindow] Independent policies: state base=$POLICY_STATE_BASE_PORT cmd base=${POLICY_CMD_BASE_PORT:-<none>} step=$POLICY_PORT_STEP host=$POLICY_HOST"
fi

if [[ "$MC_ACTIVE" == "1" || "$MC_ACTIVE" == "true" || "$MC_ACTIVE" == "TRUE" ]]; then
  CMD+=(--mc_active)
fi

if [[ "$RGB" == "1" || "$RGB" == "true" || "$RGB" == "TRUE" ]]; then
  CMD+=(--rgb_mode)
  if [[ "$SELECTED_SENSOR_ACTIVATION" == "1" || "$SELECTED_SENSOR_ACTIVATION" == "true" || "$SELECTED_SENSOR_ACTIVATION" == "TRUE" ]]; then
    CMD+=(--selected_sensor_activation)
  fi
elif [[ "$WITH_PC" == "1" || "$WITH_PC" == "true" || "$WITH_PC" == "TRUE" ]]; then
  CMD+=(--with_pc --pc_max_points "$PC_MAX_POINTS" --pc_sampling "$PC_SAMPLING" --pc_stride "$PC_STRIDE" --pc_width "$PC_WIDTH" --pc_height "$PC_HEIGHT")
  if [[ "$PC_ACTIVE_ONLY" == "1" || "$PC_ACTIVE_ONLY" == "true" || "$PC_ACTIVE_ONLY" == "TRUE" ]]; then
    CMD+=(--pc_active_only)
  fi
  if [[ "$SELECTED_SENSOR_ACTIVATION" == "1" || "$SELECTED_SENSOR_ACTIVATION" == "true" || "$SELECTED_SENSOR_ACTIVATION" == "TRUE" ]]; then
    CMD+=(--selected_sensor_activation)
  fi
fi

if [[ -n "$RUNTIME_EXTRAS" ]]; then
  CMD+=(--extra_runtime_args "$RUNTIME_EXTRAS")
fi

exec "${CMD[@]}"
