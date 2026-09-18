#!/usr/bin/env bash
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export ROBOT_KEY="${ROBOT_KEY:-p1}"
export CONTROL_HZ="${CONTROL_HZ:-120}"
export REPLAN_OUTPUT_DIR="${REPLAN_OUTPUT_DIR:-intervene_base/teleop_logs/replans}"

MIRROR_ARGS=(
  --mirror_robot
  --mirror_on_select_only
  --robot_key "$ROBOT_KEY"
  --control_hz "$CONTROL_HZ"
  --replan_output_dir "$REPLAN_OUTPUT_DIR"
)

if [[ -n "${EXTRA_RUNTIME_ARGS:-}" ]]; then
  export EXTRA_RUNTIME_ARGS="${MIRROR_ARGS[*]} ${EXTRA_RUNTIME_ARGS}"
else
  export EXTRA_RUNTIME_ARGS="${MIRROR_ARGS[*]}"
fi

echo "[MultiWindowMirror] Robot key: $ROBOT_KEY"
echo "[MultiWindowMirror] Control Hz: $CONTROL_HZ"
echo "[MultiWindowMirror] Replan output: $REPLAN_OUTPUT_DIR"
echo "[MultiWindowMirror] Grid policies run continuously; robot activates only after selecting a session."
echo "[MultiWindowMirror] Single view starts paused via ENTER_SINGLE; A resumes mirror, X auto-aligns intervention."
echo "[MultiWindowMirror] Stop the current launcher first, then run this script."

exec bash ./run_multi_window_robot.sh
