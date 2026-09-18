#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRIAL="${1:-}"
DURATION_S="${TRIAL_DURATION_S:-300}"
AB_TRAJ="${AB_TRAJ:-$ROOT/intervene_base/RESET_NPZs/TSHAPE/p1_ep_0001_1778506313_trimmed.npz}"
AB_XML="${AB_XML:-$ROOT/intervene_base/mujoco_scenes/working_scenes/with_soft_gripper/sii_scene_table_T_shape.xml}"

if [[ ! "$TRIAL" =~ ^[1-5]$ ]]; then
  echo "Usage: TRIAL_DURATION_S=300 $0 <1|2|3|4|5>" >&2
  exit 2
fi

COMMON_ENV=(
  WINDOWS=9
  START_VR=1
  ROBOT=0
  MC_ACTIVE=0
  MC_SIM=0
  REALTIME_FACTOR=0.6
  PC_ROUND_ROBIN=0
  METRICS=1
  METRICS_WINDOW_S="${METRICS_WINDOW_S:-10}"
  METRICS_SUMMARY="${METRICS_SUMMARY:-0}"
  PERF_LOG=0
  RECORD_IMAGES=0
  STUDY_ACC_METHOD=none
  INTERVENE_RANDOMIZE_SCENE=0
  INTERVENE_AUTO_TASK_EVAL=0
  SELECTED_PEER_IDLE_GRACE_S="${SELECTED_PEER_IDLE_GRACE_S:-5}"
  RESET_NPZ="$AB_TRAJ"
  TRAJ="$AB_TRAJ"
  XML="$AB_XML"
  LAB_XML="$AB_XML"
)

case "$TRIAL" in
  1)
    LABEL="mirrored_grid"
    RUNNER="$ROOT/intervene_base/utils/run_main_policy.sh"
    TRIAL_ENV=(VR_POLICY_MIRROR=1 START_GRID_UI=1)
    ;;
  2)
    LABEL="mirrored_no_grid"
    RUNNER="$ROOT/intervene_base/utils/run_main_policy.sh"
    TRIAL_ENV=(VR_POLICY_MIRROR=1 START_GRID_UI=0)
    ;;
  3)
    LABEL="local_replay_policy_load_grid"
    RUNNER="$ROOT/intervene_base/utils/run_main_policy.sh"
    TRIAL_ENV=(VR_POLICY_MIRROR=0 START_GRID_UI=1)
    ;;
  4)
    LABEL="local_replay_policy_load_no_grid"
    RUNNER="$ROOT/intervene_base/utils/run_main_policy.sh"
    TRIAL_ENV=(VR_POLICY_MIRROR=0 START_GRID_UI=0)
    ;;
  5)
    LABEL="direct_local_replay_no_policies"
    RUNNER="$ROOT/run_multi_window_robot.sh"
    TRIAL_ENV=(START_GRID_UI=0)
    ;;
esac

STAMP="$(date +%Y%m%d_%H%M%S)"
echo "[LinuxStreamingAB] trial=$TRIAL label=$LABEL duration_s=$DURATION_S stamp=$STAMP"
echo "[LinuxStreamingAB] Quest/APK/network/scene must remain unchanged between trials."

set +e
timeout --foreground --signal=INT --kill-after=30s "${DURATION_S}s" \
  env "${COMMON_ENV[@]}" "${TRIAL_ENV[@]}" \
  bash "$RUNNER"
status=$?
set -e

# timeout returns 124 after delivering INT; the launchers' traps perform normal cleanup.
if [[ "$status" -ne 0 && "$status" -ne 124 && "$status" -ne 130 ]]; then
  echo "[LinuxStreamingAB][ERROR] trial failed with status $status" >&2
  exit "$status"
fi

echo "[LinuxStreamingAB] trial complete: $LABEL"
echo "[LinuxStreamingAB] Analyze: python tools/perf_report.py --python 'session_logs/metrics_S*.jsonl'"
