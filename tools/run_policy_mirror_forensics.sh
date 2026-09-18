#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CASE="${1:-}"
WINDOWS="${FORENSICS_WINDOWS:-3}"
DURATION_S="${FORENSICS_DURATION_S:-120}"
STARTUP_TIMEOUT_S="${FORENSICS_STARTUP_TIMEOUT_S:-120}"
GRID="${FORENSICS_GRID:-0}"
STATE_BASE_PORT="${POLICY_STATE_PORT:-8066}"
CMD_BASE_PORT="${POLICY_CMD_PORT:-8065}"
PORT_STEP="${POLICY_PORT_STEP:-10}"
TOPIC_BASE_PORT="${BASE_TOPIC_PORT:-7741}"
SERVICE_BASE_PORT="${BASE_SERVICE_PORT:-7740}"
RUNTIME_CMD_BASE_PORT="${BASE_CMD_PORT:-$((TOPIC_BASE_PORT + 5))}"
OLD_POLICY_COMMIT="${FORENSICS_OLD_POLICY_COMMIT:-1d24f5e}"
OLD_SIMPUB_COMMIT="${FORENSICS_OLD_SIMPUB_COMMIT:-b364bd8c}"
OLD_CONSUMER_COMPAT="${FORENSICS_OLD_CONSUMER_COMPAT:-1}"
OUT_ROOT="${FORENSICS_OUT_ROOT:-$ROOT/session_logs/policy_mirror_forensics}"
AB_TRAJ="${FORENSICS_TRAJ:-$ROOT/intervene_base/RESET_NPZs/TSHAPE/p1_ep_0001_1778506313_trimmed.npz}"
AB_XML="${FORENSICS_XML:-$ROOT/intervene_base/mujoco_scenes/working_scenes/with_soft_gripper/sii_scene_table_T_shape.xml}"
RECORD_IMAGES="${FORENSICS_RECORD_IMAGES:-0}"
ACC_METHOD="${FORENSICS_ACC_METHOD:-none}"
ACC_PROBE_ASYNC="${FORENSICS_ACC_PROBE_ASYNC:-1}"
ACC_PROBE_MIDCHUNK="${FORENSICS_ACC_PROBE_MIDCHUNK:-0}"
# FORENSICS_OOD_* retired with the OOD auto-pause supervisor: there is no longer a
# feature to isolate here. ACC remains isolatable via FORENSICS_ACC_METHOD.
QUEST_ENABLED="${FORENSICS_QUEST:-0}"
INTERACTIVE="${FORENSICS_INTERACTIVE:-$QUEST_ENABLED}"
VALIDATE_DELIVERY="${FORENSICS_VALIDATE_DELIVERY:-$INTERACTIVE}"
VISITS_PER_SESSION="${FORENSICS_VISITS_PER_SESSION:-2}"
METRICS_WINDOW_S="${FORENSICS_METRICS_WINDOW_S:-2}"
ADB_BIN="${FORENSICS_ADB:-adb}"
ADB_PACKAGE="${FORENSICS_ADB_PACKAGE:-com.anonymous.SIIMetaQuest3}"
ADB_ACTIVITY="${FORENSICS_ADB_ACTIVITY:-com.unity3d.player.UnityPlayerGameActivity}"
OPEN3D_PYTHONPATH="${FORENSICS_OPEN3D_PYTHONPATH:-$HOME/open3d_build/Open3D/build/lib/python_package}"

source_fingerprint() {
  sha256sum \
    "$ROOT/tools/run_policy_mirror_forensics.sh" \
    "$ROOT/tools/run_forensic_consumer.sh" \
    "$ROOT/tools/prepare_forensic_consumer.py" \
    "$ROOT/SimPublisher/sii/integration_v1/runtime_impl.py" \
    "$ROOT/intervene_base/app.py" \
    "$ROOT/intervene_base/playback/policy_player.py" \
    | sha256sum | awk '{print $1}'
}

require_source_fingerprint() {
  local expected="${FORENSICS_EXPECTED_SOURCE_FINGERPRINT:-}"
  [[ -z "$expected" ]] && return 0
  local actual
  actual="$(source_fingerprint)"
  if [[ "$actual" != "$expected" ]]; then
    echo "[PolicyForensics][ERROR] source changed during matrix: expected=$expected actual=$actual" >&2
    return 1
  fi
}

FORENSICS_PYTHON="${FORENSICS_PYTHON:-}"
if [[ -z "$FORENSICS_PYTHON" ]]; then
  for candidate in "$ROOT/.venv/bin/python" "$HOME/miniforge3/envs/polymetis/bin/python" python3; do
    if "$candidate" -c 'import zmq' >/dev/null 2>&1; then
      FORENSICS_PYTHON="$candidate"
      break
    fi
  done
fi
if [[ -z "$FORENSICS_PYTHON" ]]; then
  echo "[PolicyForensics][ERROR] no Python environment with pyzmq is available." >&2
  exit 2
fi

usage() {
  cat <<'EOF'
Usage: tools/run_policy_mirror_forensics.sh <old-old|current-old|old-current|current-current|matrix>

Defaults to a 3-session, 120-second, no-grid run. Examples:
  FORENSICS_DURATION_S=120 tools/run_policy_mirror_forensics.sh matrix
  FORENSICS_WINDOWS=9 FORENSICS_GRID=1 tools/run_policy_mirror_forensics.sh current-current
  FORENSICS_WINDOWS=9 FORENSICS_DURATION_S=180 FORENSICS_QUEST=1 tools/run_policy_mirror_forensics.sh matrix

The runner never resets the workspace. Historical components are detached worktrees in
/tmp and current components are read directly from the dirty workspace.
EOF
}

if [[ ! "$CASE" =~ ^(old-old|current-old|old-current|current-current|matrix)$ ]]; then
  usage >&2
  exit 2
fi

for numeric_bool in "$GRID" "$RECORD_IMAGES" "$QUEST_ENABLED" "$INTERACTIVE" "$VALIDATE_DELIVERY" "$OLD_CONSUMER_COMPAT"; do
  if [[ ! "$numeric_bool" =~ ^[01]$ ]]; then
    echo "[PolicyForensics][ERROR] grid and record-images flags must be 0 or 1." >&2
    exit 2
  fi
done
if [[ ! "$WINDOWS" =~ ^(3|6|9|12|15)$ || ! "$DURATION_S" =~ ^[1-9][0-9]*$ ]]; then
  echo "[PolicyForensics][ERROR] FORENSICS_WINDOWS must be 3, 6, 9, 12, or 15; duration must be a positive integer." >&2
  exit 2
fi
if [[ ! "$VISITS_PER_SESSION" =~ ^[1-9][0-9]*$ ]]; then
  echo "[PolicyForensics][ERROR] FORENSICS_VISITS_PER_SESSION must be positive." >&2
  exit 2
fi
if [[ ! "$ACC_METHOD" =~ ^(none|chunk_residual|ensemble_disagreement)$ ]]; then
  echo "[PolicyForensics][ERROR] unsupported FORENSICS_ACC_METHOD=$ACC_METHOD" >&2
  exit 2
fi
if [[ ! "$ACC_PROBE_ASYNC" =~ ^[01]$ || ! "$ACC_PROBE_MIDCHUNK" =~ ^[01]$ ]]; then
  echo "[PolicyForensics][ERROR] ACC probe flags must be 0 or 1." >&2
  exit 2
fi

if [[ "$CASE" == "matrix" ]]; then
  matrix_group="${FORENSICS_MATRIX_GROUP:-$(date +%Y%m%d_%H%M%S)}"
  export FORENSICS_MATRIX_GROUP="$matrix_group"
  export FORENSICS_EXPECTED_SOURCE_FINGERPRINT="$(source_fingerprint)"
  matrix_status=0
  for item in old-old current-old old-current current-current; do
    require_source_fingerprint || exit 1
    if ! "$0" "$item"; then
      matrix_status=1
    fi
    require_source_fingerprint || exit 1
  done
  "$FORENSICS_PYTHON" "$ROOT/tools/policy_matrix_report.py" \
    --root "$OUT_ROOT" --group "$matrix_group" \
    --out "$OUT_ROOT/matrix_${matrix_group}.json" || matrix_status=1
  exit "$matrix_status"
fi

require_source_fingerprint

if [[ "$CASE" == old-* ]]; then PRODUCER_KIND=old; else PRODUCER_KIND=current; fi
if [[ "$CASE" == *-old ]]; then CONSUMER_KIND=old; else CONSUMER_KIND=current; fi

STAMP="$(date +%Y%m%d_%H%M%S)"
GROUP_PART="${FORENSICS_MATRIX_GROUP:+${FORENSICS_MATRIX_GROUP}_}"
RUN_DIR="$OUT_ROOT/${STAMP}_${GROUP_PART}${CASE}_w${WINDOWS}_grid${GRID}"
TMP_ROOT="$(mktemp -d "/tmp/iilar-policy-forensics-${CASE}-XXXXXX")"
MATRIX_ROOT="$TMP_ROOT/workspace"
mkdir -p "$RUN_DIR/runtime" "$RUN_DIR/policy" "$RUN_DIR/episodes" "$MATRIX_ROOT"
ln -s "$RUN_DIR/policy" "$MATRIX_ROOT/session_logs"

POLICY_PID=""
CONSUMER_PID=""
TRACE_PID=""
RESOURCE_PID=""
QUEST_PID=""
DELIVERY_PID=""
OLD_POLICY_TREE=""
OLD_SIMPUB_TREE=""

require_runtime_children() {
  local status_file="$RUN_DIR/runtime/multi_session_status.json"
  "$FORENSICS_PYTHON" - "$status_file" "$WINDOWS" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected = int(sys.argv[2])
if not path.exists():
    raise SystemExit(1)
try:
    rows = json.loads(path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError):
    raise SystemExit(1)
failures = []
if len(rows) != expected:
    failures.append(f"status has {len(rows)}/{expected} sessions")
for row in rows:
    session = int(row.get("session_index", -1))
    status = row.get("status")
    restarts = int(row.get("restart_count", 0) or 0)
    pid = int(row.get("pid", -1) or -1)
    alive = pid > 0
    if alive:
        try:
            os.kill(pid, 0)
        except OSError:
            alive = False
    if status != "running" or restarts != 0 or not alive:
        failures.append(
            f"S{session:02d} status={status} pid={pid} alive={alive} restarts={restarts}"
        )
if failures:
    print("[PolicyForensics][ERROR] runtime child failure: " + "; ".join(failures), file=sys.stderr)
    raise SystemExit(1)
PY
}

stop_pid() {
  local pid="${1:-}"
  [[ -z "$pid" ]] && return 0
  if kill -0 "$pid" 2>/dev/null; then
    kill -INT -- "-$pid" 2>/dev/null || kill -INT "$pid" 2>/dev/null || true
    for _ in $(seq 1 50); do
      kill -0 "$pid" 2>/dev/null || return 0
      sleep 0.1
    done
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  fi
}

cleanup() {
  set +e
  stop_pid "$QUEST_PID"
  stop_pid "$DELIVERY_PID"
  stop_pid "$RESOURCE_PID"
  stop_pid "$TRACE_PID"
  stop_pid "$CONSUMER_PID"
  stop_pid "$POLICY_PID"
  if [[ -n "$OLD_POLICY_TREE" ]]; then
    git -C "$ROOT/intervene_base" worktree remove --force "$OLD_POLICY_TREE" >/dev/null 2>&1 || true
  fi
  if [[ -n "$OLD_SIMPUB_TREE" ]]; then
    git -C "$ROOT/SimPublisher" worktree remove --force "$OLD_SIMPUB_TREE" >/dev/null 2>&1 || true
  fi
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT INT TERM

PREFLIGHT_ARGS=(
  --sessions "$WINDOWS"
  --state-base-port "$STATE_BASE_PORT"
  --command-base-port "$CMD_BASE_PORT"
  --topic-base-port "$TOPIC_BASE_PORT"
  --service-base-port "$SERVICE_BASE_PORT"
  --runtime-command-base-port "$RUNTIME_CMD_BASE_PORT"
  --port-step "$PORT_STEP"
  --min-gpu-free-mb "${FORENSICS_MIN_GPU_FREE_MB:-2048}"
  --python-bin "$ROOT/.venv/bin/python"
  --open3d-path "$OPEN3D_PYTHONPATH"
  --out "$RUN_DIR/preflight.json"
)
if [[ "${FORENSICS_SKIP_GPU_CHECK:-0}" == "1" ]]; then
  PREFLIGHT_ARGS+=(--skip-gpu-check)
fi
if [[ "$QUEST_ENABLED" == "1" ]]; then
  PREFLIGHT_ARGS+=(--require-adb --adb "$ADB_BIN" --adb-package "$ADB_PACKAGE")
fi
"$FORENSICS_PYTHON" "$ROOT/tools/policy_forensics_preflight.py" "${PREFLIGHT_ARGS[@]}"

if [[ "$PRODUCER_KIND" == "old" ]]; then
  OLD_POLICY_TREE="$TMP_ROOT/intervene_base_old"
  git -C "$ROOT/intervene_base" worktree add --detach "$OLD_POLICY_TREE" "$OLD_POLICY_COMMIT"
  ln -s "$ROOT/intervene_base/MODEL_WEIGHTS" "$OLD_POLICY_TREE/MODEL_WEIGHTS"
  PRODUCER_ROOT="$OLD_POLICY_TREE"
else
  PRODUCER_ROOT="$ROOT/intervene_base"
fi

if [[ "$CONSUMER_KIND" == "old" ]]; then
  OLD_SIMPUB_TREE="$TMP_ROOT/SimPublisher_old"
  git -C "$ROOT/SimPublisher" worktree add --detach "$OLD_SIMPUB_TREE" "$OLD_SIMPUB_COMMIT"
  if [[ "$OLD_CONSUMER_COMPAT" == "1" ]]; then
    "$FORENSICS_PYTHON" "$ROOT/tools/prepare_forensic_consumer.py" \
      --runtime "$OLD_SIMPUB_TREE/sii/integration_v1/runtime_impl.py" \
      --report "$RUN_DIR/old_consumer_compat.json"
  else
    echo "[PolicyForensics][ERROR] the current APK requires the old-consumer identity shim." >&2
    echo "[PolicyForensics][ERROR] set FORENSICS_OLD_CONSUMER_COMPAT=1 or test with the historical APK." >&2
    exit 2
  fi
  ln -s "$OLD_SIMPUB_TREE" "$MATRIX_ROOT/SimPublisher"
else
  ln -s "$ROOT/SimPublisher" "$MATRIX_ROOT/SimPublisher"
fi
# run_main_policy.sh validates this path even when START_VR=0. It is never executed by
# the matrix: tools/run_forensic_consumer.sh supplies one common command to both versions.
cp "$ROOT/run_multi_window_robot.sh" "$MATRIX_ROOT/run_multi_window_robot.sh"
chmod +x "$MATRIX_ROOT/run_multi_window_robot.sh"
ln -s "$ROOT/.venv" "$MATRIX_ROOT/.venv"
[[ -d "$ROOT/src" ]] && ln -s "$ROOT/src" "$MATRIX_ROOT/src"

cat > "$RUN_DIR/manifest.json" <<EOF
{
  "case": "$CASE",
  "producer": "$PRODUCER_KIND",
  "consumer": "$CONSUMER_KIND",
  "old_policy_commit": "$OLD_POLICY_COMMIT",
  "old_simpublisher_commit": "$OLD_SIMPUB_COMMIT",
  "old_consumer_compat": $(if [[ "$CONSUMER_KIND" == "old" ]]; then echo true; else echo false; fi),
  "current_root_head": "$(git -C "$ROOT" rev-parse HEAD)",
  "current_policy_head": "$(git -C "$ROOT/intervene_base" rev-parse HEAD)",
  "current_simpublisher_head": "$(git -C "$ROOT/SimPublisher" rev-parse HEAD)",
  "current_root_dirty": $(if [[ -n "$(git -C "$ROOT" status --porcelain --untracked-files=no)" ]]; then echo true; else echo false; fi),
  "current_policy_dirty": $(if [[ -n "$(git -C "$ROOT/intervene_base" status --porcelain --untracked-files=no)" ]]; then echo true; else echo false; fi),
  "current_simpublisher_dirty": $(if [[ -n "$(git -C "$ROOT/SimPublisher" status --porcelain --untracked-files=no)" ]]; then echo true; else echo false; fi),
  "current_policy_app_sha256": "$(sha256sum "$ROOT/intervene_base/app.py" | awk '{print $1}')",
  "current_policy_player_sha256": "$(sha256sum "$ROOT/intervene_base/playback/policy_player.py" | awk '{print $1}')",
  "current_consumer_runtime_sha256": "$(sha256sum "$ROOT/SimPublisher/sii/integration_v1/runtime_impl.py" | awk '{print $1}')",
  "windows": $WINDOWS,
  "grid": $GRID,
  "duration_s": $DURATION_S,
  "realtime_factor": 0.6,
  "policy_hz": 10,
  "state_hz": 30,
  "metrics_window_s": $METRICS_WINDOW_S,
  "runtime_fps": 60,
  "pc_fps_cap": 30,
  "pc_max_points": 80000,
  "record_images": $RECORD_IMAGES,
  "acc_method": "$ACC_METHOD",
  "acc_probe_async": $ACC_PROBE_ASYNC,
  "acc_probe_midchunk": $ACC_PROBE_MIDCHUNK,
  ,"quest_enabled": $QUEST_ENABLED
  ,"interactive": $INTERACTIVE
  ,"validate_delivery": $VALIDATE_DELIVERY
  ,"visits_per_session": $VISITS_PER_SESSION
}
EOF

READY_FILE="$RUN_DIR/policies_ready.json"
STALL_EVENTS="$RUN_DIR/stall_events.jsonl"
setsid "$FORENSICS_PYTHON" "$ROOT/tools/policy_state_trace.py" collect \
  --out "$RUN_DIR/policy_state.jsonl" \
  --base-port "$STATE_BASE_PORT" --port-step "$PORT_STEP" --sessions "$WINDOWS" \
  --ready-file "$READY_FILE" --stall-events "$STALL_EVENTS" --stall-s 0.5 \
  >"$RUN_DIR/state_trace.log" 2>&1 &
TRACE_PID=$!

POLICY_ENV=(
  WINDOWS="$WINDOWS"
  START_VR=0
  START_GRID_UI="$GRID"
  ROBOT=0
  MC_ACTIVE=0
  MC_SIM=0
  POLICY_HZ=10
  REALTIME_FACTOR=0.6
  POLICY_STATE_HZ=30
  POLICY_STATE_PORT="$STATE_BASE_PORT"
  POLICY_CMD_PORT="$CMD_BASE_PORT"
  POLICY_PORT_STEP="$PORT_STEP"
  POLICY_LAUNCH_STAGGER="${FORENSICS_POLICY_STAGGER_S:-1}"
  PERF_LOG=0
  METRICS=1
  METRICS_WINDOW_S="$METRICS_WINDOW_S"
  METRICS_SUMMARY=0
  RECORD_IMAGES="$RECORD_IMAGES"
  INTERVENE_RECORD_RGB="$RECORD_IMAGES"
  STUDY_ACC_METHOD="$ACC_METHOD"
  STUDY_ACC_PROBE_ASYNC="$ACC_PROBE_ASYNC"
  STUDY_ACC_PROBE_MIDCHUNK="$ACC_PROBE_MIDCHUNK"
  INTERVENE_RANDOMIZE_SCENE=0
  INTERVENE_AUTO_TASK_EVAL=0
  INTERVENE_OUTPUT_DIR="$RUN_DIR/episodes"
  INTERVENE_EPISODE_DIR="$RUN_DIR/episodes"
  REPLAN_OUTPUT_DIR="$RUN_DIR/episodes/replans"
  XML="$AB_XML"
  RESET_NPZ="$AB_TRAJ"
  UMBRELLA_ROOT="$MATRIX_ROOT"
)
setsid env "${POLICY_ENV[@]}" bash "$PRODUCER_ROOT/utils/run_main_policy.sh" \
  >"$RUN_DIR/policy_launcher.log" 2>&1 &
POLICY_PID=$!

startup_deadline=$((SECONDS + STARTUP_TIMEOUT_S))
while [[ ! -s "$READY_FILE" ]]; do
  if ! kill -0 "$POLICY_PID" 2>/dev/null; then
    echo "[PolicyForensics][ERROR] policy launcher exited before every state stream advanced." >&2
    exit 1
  fi
  if (( SECONDS >= startup_deadline )); then
    echo "[PolicyForensics][ERROR] policies did not produce three distinct states each within ${STARTUP_TIMEOUT_S}s." >&2
    exit 1
  fi
  sleep 0.25
done
echo "[PolicyForensics] all $WINDOWS policy streams are advancing; starting VR consumer."

CONSUMER_ENV=(
  WINDOWS="$WINDOWS"
  ROBOT=0
  SINGLE=0
  START_VR=1
  TRAJ="$AB_TRAJ"
  LAB_XML="$AB_XML"
  FPS=60
  FPS_IDLE=10
  PC_MAX_POINTS=80000
  PC_STRIDE=3
  PC_FPS_CAP=30
  PC_ROUND_ROBIN=0
  PERF_LOG=0
  METRICS=1
  METRICS_WINDOW_S="$METRICS_WINDOW_S"
  METRICS_SUMMARY=0
  POLICY_STATE_BASE_PORT="$STATE_BASE_PORT"
  POLICY_CMD_BASE_PORT="$CMD_BASE_PORT"
  POLICY_PORT_STEP="$PORT_STEP"
  POLICY_HOST=127.0.0.1
  BASE_TOPIC_PORT="$TOPIC_BASE_PORT"
  BASE_SERVICE_PORT="$SERVICE_BASE_PORT"
  BASE_CMD_PORT="$RUNTIME_CMD_BASE_PORT"
  SELECTED_PEER_IDLE_GRACE_S=5
)
setsid env "${CONSUMER_ENV[@]}" \
  SIMPUBLISHER_ROOT="$MATRIX_ROOT/SimPublisher" \
  PYTHON_BIN="$ROOT/.venv/bin/python" \
  FORENSICS_RUNTIME_LOG_DIR="$RUN_DIR/runtime" \
  bash "$ROOT/tools/run_forensic_consumer.sh" \
  >"$RUN_DIR/consumer_launcher.log" 2>&1 &
CONSUMER_PID=$!

setsid "$FORENSICS_PYTHON" "$ROOT/tools/process_resource_trace.py" \
  --out "$RUN_DIR/resources.jsonl" --root-pid "$POLICY_PID" --root-pid "$CONSUMER_PID" \
  --stall-events "$STALL_EVENTS" --stall-dir "$RUN_DIR/stalls" \
  >"$RUN_DIR/resource_trace.log" 2>&1 &
RESOURCE_PID=$!

if [[ "$QUEST_ENABLED" == "1" ]]; then
  ADB_SERIAL="$($FORENSICS_PYTHON -c 'import json,sys; print(json.load(open(sys.argv[1]))["adb"]["selected_serial"])' "$RUN_DIR/preflight.json")"
fi

BACKEND_READY="$RUN_DIR/backend_ready.json"
QUEST_READY="$RUN_DIR/quest_ready.json"
MEASURE_FILE="$RUN_DIR/measurement_started"
setsid "$FORENSICS_PYTHON" "$ROOT/tools/forensic_delivery_trace.py" watch \
  --runtime-dir "$RUN_DIR/runtime" --quest "$RUN_DIR/quest.jsonl" \
  --events "$RUN_DIR/delivery_events.jsonl" --sessions "$WINDOWS" \
  --backend-ready "$BACKEND_READY" --quest-ready "$QUEST_READY" \
  --measure-file "$MEASURE_FILE" >"$RUN_DIR/delivery_trace.log" 2>&1 &
DELIVERY_PID=$!

readiness_deadline=$((SECONDS + STARTUP_TIMEOUT_S))
while [[ ! -s "$BACKEND_READY" ]]; do
  if ! kill -0 "$DELIVERY_PID" 2>/dev/null; then
    echo "[PolicyForensics][ERROR] delivery watcher exited during backend readiness." >&2
    exit 1
  fi
  if ! kill -0 "$CONSUMER_PID" 2>/dev/null; then
    echo "[PolicyForensics][ERROR] consumer exited before all runtimes became ready." >&2
    exit 1
  fi
  if (( SECONDS >= readiness_deadline )); then
    echo "[PolicyForensics][ERROR] runtimes did not reach command/RGB/policy readiness." >&2
    exit 1
  fi
  sleep 0.25
done
require_runtime_children

if [[ "$QUEST_ENABLED" == "1" ]]; then
  stop_pid "$DELIVERY_PID"; DELIVERY_PID=""
  timeout 30s "$ADB_BIN" -s "$ADB_SERIAL" shell am force-stop "$ADB_PACKAGE"
  sleep 0.5
  : >"$RUN_DIR/quest.jsonl"
  rm -f "$QUEST_READY" "$RUN_DIR/delivery_events.jsonl"
  setsid "$FORENSICS_PYTHON" "$ROOT/tools/quest_telemetry.py" \
    --adb "$ADB_BIN" --serial "$ADB_SERIAL" --out "$RUN_DIR/quest.jsonl" \
    >>"$RUN_DIR/quest_telemetry.log" 2>&1 &
  QUEST_PID=$!
  setsid "$FORENSICS_PYTHON" "$ROOT/tools/forensic_delivery_trace.py" watch \
    --runtime-dir "$RUN_DIR/runtime" --quest "$RUN_DIR/quest.jsonl" \
    --events "$RUN_DIR/delivery_events.jsonl" --sessions "$WINDOWS" \
    --backend-ready "$BACKEND_READY" --quest-ready "$QUEST_READY" \
    --measure-file "$MEASURE_FILE" >>"$RUN_DIR/delivery_trace.log" 2>&1 &
  DELIVERY_PID=$!
  timeout 30s "$ADB_BIN" -s "$ADB_SERIAL" shell am start -W -S \
    -n "$ADB_PACKAGE/$ADB_ACTIVITY" >"$RUN_DIR/apk_restart.log"
  readiness_deadline=$((SECONDS + STARTUP_TIMEOUT_S))
  while [[ ! -s "$QUEST_READY" ]]; do
    if ! kill -0 "$QUEST_PID" 2>/dev/null; then
      echo "[PolicyForensics][ERROR] Quest telemetry exited before headset readiness." >&2
      exit 1
    fi
    if ! kill -0 "$DELIVERY_PID" 2>/dev/null; then
      echo "[PolicyForensics][ERROR] delivery watcher exited during Quest readiness." >&2
      exit 1
    fi
    if (( SECONDS >= readiness_deadline )); then
      echo "[PolicyForensics][ERROR] Quest did not report VrApi plus all $WINDOWS grid panels." >&2
      exit 1
    fi
    sleep 0.25
  done
fi

if [[ "$INTERACTIVE" == "1" ]]; then
  echo
  echo "[PolicyForensics] CASE $CASE is backend-ready."
  if [[ "$QUEST_ENABLED" != "1" ]]; then
    echo "[PolicyForensics] ADB is disabled: manually restart the APK and wait for the nine-panel grid."
  fi
  echo "[PolicyForensics] After pressing Enter, select sessions 0..$((WINDOWS - 1)) in order,"
  echo "[PolicyForensics] stay in each scene for at least 7 seconds, return to the grid,"
  echo "[PolicyForensics] and repeat the full sequence $VISITS_PER_SESSION time(s)."
  echo "[PolicyForensics] Do not intervene during this matrix."
  read -r -p "Press Enter only when the fresh grid is visible: "
fi
date +%s.%N >"$MEASURE_FILE"
echo "[PolicyForensics] measurement started for ${DURATION_S}s; fresh ENTER_SINGLE events are now required."

run_deadline=$((SECONDS + DURATION_S))
next_runtime_check=$SECONDS
while (( SECONDS < run_deadline )); do
  if ! kill -0 "$POLICY_PID" 2>/dev/null; then
    echo "[PolicyForensics][ERROR] policy launcher died during the measured run." >&2
    exit 1
  fi
  if ! kill -0 "$CONSUMER_PID" 2>/dev/null; then
    echo "[PolicyForensics][ERROR] VR consumer died during the measured run." >&2
    exit 1
  fi
  if (( SECONDS >= next_runtime_check )); then
    require_runtime_children
    next_runtime_check=$((SECONDS + 2))
  fi
  if [[ "$QUEST_ENABLED" == "1" ]] && ! kill -0 "$QUEST_PID" 2>/dev/null; then
    echo "[PolicyForensics][ERROR] Quest telemetry exited during the measured run." >&2
    exit 1
  fi
  sleep 0.5
done

require_source_fingerprint

if [[ "$INTERACTIVE" == "1" ]]; then
  echo
  read -r -p "Return to the grid so the final EXIT_SINGLE is sent, then press Enter: "
  sleep 1
fi

stop_pid "$QUEST_PID"; QUEST_PID=""
sleep 0.5
stop_pid "$DELIVERY_PID"; DELIVERY_PID=""
stop_pid "$RESOURCE_PID"; RESOURCE_PID=""
stop_pid "$CONSUMER_PID"; CONSUMER_PID=""
stop_pid "$POLICY_PID"; POLICY_PID=""
stop_pid "$TRACE_PID"; TRACE_PID=""

set +e
"$FORENSICS_PYTHON" "$ROOT/tools/policy_state_trace.py" report \
  "$RUN_DIR/policy_state.jsonl" --out "$RUN_DIR/verdict.json" \
  --expected-sessions "$WINDOWS" --min-snapshot-hz 20 --min-distinct-hz 4.5 \
  --max-distinct-gap-s 0.5 | tee "$RUN_DIR/verdict.txt"
verdict_status=${PIPESTATUS[0]}
set -e

if [[ "$VALIDATE_DELIVERY" == "1" ]]; then
  DELIVERY_ARGS=(
    report --events "$RUN_DIR/delivery_events.jsonl" --quest "$RUN_DIR/quest.jsonl"
    --runtime-dir "$RUN_DIR/runtime" --policy-verdict "$RUN_DIR/verdict.json"
    --out "$RUN_DIR/delivery_verdict.json" --sessions "$WINDOWS"
    --visits-per-session "$VISITS_PER_SESSION" --min-pc-hz 15 --min-grid-hz 8
    --min-display-fps 70 --max-activation-ms 1000 --max-source-age-s 0.5
  )
  if [[ "$QUEST_ENABLED" != "1" ]]; then DELIVERY_ARGS+=(--skip-quest); fi
  set +e
  "$FORENSICS_PYTHON" "$ROOT/tools/forensic_delivery_trace.py" "${DELIVERY_ARGS[@]}" \
    | tee "$RUN_DIR/delivery_verdict.txt"
  delivery_status=${PIPESTATUS[0]}
  set -e
  if [[ "$delivery_status" -ne 0 ]]; then verdict_status="$delivery_status"; fi
fi

echo "[PolicyForensics] case=$CASE artifacts=$RUN_DIR verdict_status=$verdict_status"
if [[ -n "${FORENSICS_BASELINE_VERDICT:-}" ]]; then
  set +e
  "$FORENSICS_PYTHON" "$ROOT/tools/policy_state_trace.py" compare \
    --baseline "$FORENSICS_BASELINE_VERDICT" --candidate "$RUN_DIR/verdict.json" \
    --out "$RUN_DIR/baseline_comparison.json" | tee "$RUN_DIR/baseline_comparison.txt"
  compare_status=${PIPESTATUS[0]}
  set -e
  if [[ "$compare_status" -ne 0 ]]; then
    verdict_status="$compare_status"
  fi
fi
exit "$verdict_status"
