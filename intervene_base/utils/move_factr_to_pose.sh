#!/usr/bin/env bash
# Safely stage FACTR at a saved JSON pose and hold it there.
#
# The same safety-gated controller used by run_main_policy.sh is used here:
# calibration/torque/gravity reports are validated, the target is checked against
# the configured limits, and an out-of-limit start is accepted only when it is
# sufficiently close to the saved rest pose. Ctrl+C returns FACTR to rest before
# disabling torque.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERVENE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  printf '%s\n' \
    "Usage: bash utils/move_factr_to_pose.sh [POSE.json]" \
    "" \
    "Default pose:" \
    "  ${INTERVENE_ROOT}/factr/validation/init_pose.json" \
    "" \
    "The target is held until Ctrl+C. Shutdown then returns FACTR to the" \
    "configured rest pose before disabling motor torque." \
    "" \
    "Useful environment overrides:" \
    "  FACTR_POSITION_TOLERANCE=0.08" \
    "  FACTR_REST_POSITION_TOLERANCE=1.5" \
    "  FACTR_INIT_TIMEOUT=25" \
    "  FACTR_MAX_VELOCITY=0.5" \
    "  FACTR_MOVE_ASSUME_YES=1  # skip the typed MOVE confirmation"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
if [[ "$#" -gt 1 ]]; then
  usage >&2
  exit 2
fi

if [[ -n "${FACTR_PYTHON_BIN:-}" ]]; then
  python_candidates=("${FACTR_PYTHON_BIN}")
else
  python_candidates=()
  if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
    python_candidates+=("${CONDA_PREFIX}/bin/python")
  fi
  python_candidates+=(
    "${HOME}/miniforge3/envs/polymetis/bin/python"
    "${HOME}/miniconda3/envs/polymetis/bin/python"
    "${PYTHON_BIN:-python}"
  )
fi

PYTHON_BIN=""
for python_candidate in "${python_candidates[@]}"; do
  if { command -v "${python_candidate}" >/dev/null 2>&1 || [[ -x "${python_candidate}" ]]; } \
      && "${python_candidate}" -c 'import dynamixel_sdk, mujoco, pinocchio, yaml' \
        >/dev/null 2>&1; then
    PYTHON_BIN="${python_candidate}"
    break
  fi
done

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "[FACTRMove][ERROR] No Python interpreter can import the FACTR stack." >&2
  echo "[FACTRMove] Activate polymetis or set FACTR_PYTHON_BIN explicitly." >&2
  exit 1
fi

TARGET_POSE="${1:-${INTERVENE_ROOT}/factr/validation/init_pose.json}"
FACTR_CONFIG="${FACTR_CONFIG:-${INTERVENE_ROOT}/factr/leader.yaml}"
FACTR_REST_POSE="${FACTR_REST_POSE:-${INTERVENE_ROOT}/factr/validation/rest_pose.json}"
CALIBRATION_REPORT="${INTERVENE_FACTR_CALIBRATION_REPORT:-${INTERVENE_ROOT}/factr/validation/calibration.json}"
TORQUE_DIRECTION_REPORT="${INTERVENE_FACTR_TORQUE_DIRECTION_REPORT:-${INTERVENE_ROOT}/factr/validation/torque_direction.json}"
GRAVITY_REPORT="${INTERVENE_FACTR_GRAVITY_REPORT:-${INTERVENE_ROOT}/factr/validation/gravity.json}"
FACTR_SERVICE_HOST="${FACTR_SERVICE_HOST:-127.0.0.1}"
FACTR_SERVICE_PORT="${FACTR_SERVICE_PORT:-18075}"

if [[ "${TARGET_POSE}" != /* ]]; then
  if [[ -f "${TARGET_POSE}" ]]; then
    TARGET_POSE="$(realpath "${TARGET_POSE}")"
  else
    TARGET_POSE="${INTERVENE_ROOT}/${TARGET_POSE}"
  fi
fi

for required_file in \
  "${TARGET_POSE}" \
  "${FACTR_CONFIG}" \
  "${FACTR_REST_POSE}" \
  "${CALIBRATION_REPORT}" \
  "${TORQUE_DIRECTION_REPORT}" \
  "${GRAVITY_REPORT}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "[FACTRMove][ERROR] Required file not found: ${required_file}" >&2
    exit 1
  fi
done

if pgrep -f '[f]actr/src/factr_intervention_server\.py' >/dev/null 2>&1; then
  echo "[FACTRMove][ERROR] A FACTR intervention service is already running." >&2
  echo "[FACTRMove] Stop the policy launcher/service cleanly before using this script." >&2
  exit 1
fi

echo "[FACTRMove] Validating and displaying target without touching the motors..."
echo "[FACTRMove] Python: ${PYTHON_BIN}"
(
  cd "${INTERVENE_ROOT}"
  "${PYTHON_BIN}" factr/src/go_to_ref_pose.py \
    --factr-only \
    --pose-file "${TARGET_POSE}" \
    --config "${FACTR_CONFIG}" \
    --print-target-only
)

echo
echo "[FACTRMove] Target: ${TARGET_POSE}"
echo "[FACTRMove] Rest:   ${FACTR_REST_POSE}"
echo "[FACTRMove] Keep a hand on FACTR and be ready to cut motor power."
echo "[FACTRMove] The arm will remain under torque at the target."
echo "[FACTRMove] Press Ctrl+C to return it to rest and disable torque."

if [[ "${FACTR_MOVE_ASSUME_YES:-0}" != "1" ]]; then
  read -r -p "Type MOVE to enable bounded FACTR position control: " confirmation
  if [[ "${confirmation}" != "MOVE" ]]; then
    echo "[FACTRMove] Cancelled; motor torque was not enabled."
    exit 0
  fi
fi

READY_FILE="$(mktemp "/tmp/intervene_factr_move_${UID}_XXXXXX.ready")"
rm -f "${READY_FILE}"
cleanup() {
  rm -f "${READY_FILE}"
}
trap cleanup EXIT

echo "[FACTRMove] Starting safety-gated move."
(
  cd "${INTERVENE_ROOT}"
  env \
    FACTR_CONFIG="${FACTR_CONFIG}" \
    INTERVENE_FACTR_CALIBRATION_REPORT="${CALIBRATION_REPORT}" \
    INTERVENE_FACTR_TORQUE_DIRECTION_REPORT="${TORQUE_DIRECTION_REPORT}" \
    INTERVENE_FACTR_GRAVITY_REPORT="${GRAVITY_REPORT}" \
    INTERVENE_FACTR_MAX_TORQUE="${FACTR_MAX_TORQUE:-0.15}" \
    INTERVENE_FACTR_ALLOW_OVER_CONFIG_TORQUE="${FACTR_ALLOW_OVER_CONFIG_TORQUE:-0}" \
    INTERVENE_FACTR_HOLD_ERROR="${FACTR_HOLD_ERROR:-0.08}" \
    INTERVENE_FACTR_DRIVE_TORQUE="${FACTR_DRIVE_TORQUE:-2.2,2.2,2.2,2.2,1.2,1.0,0.7}" \
    INTERVENE_FACTR_DRIVE_DEADBAND="${FACTR_DRIVE_DEADBAND:-0.01}" \
    INTERVENE_FACTR_DRIVE_RAMP="${FACTR_DRIVE_RAMP:-0.20}" \
    INTERVENE_FACTR_TRAJECTORY_DURATION="${FACTR_TRAJECTORY_DURATION:-3.0}" \
    INTERVENE_FACTR_RAMP_DURATION="${FACTR_GRAVITY_RAMP_DURATION:-1.0}" \
    INTERVENE_FACTR_SETTLE_SECONDS="${FACTR_SETTLE_SECONDS:-0.50}" \
    INTERVENE_FACTR_SETTLE_MAX_VELOCITY_RAD_S="${FACTR_SETTLE_MAX_VELOCITY:-0.10}" \
    INTERVENE_FACTR_MAX_VELOCITY_RAD_S="${FACTR_MAX_VELOCITY:-0.5}" \
    INTERVENE_FACTR_MAX_STATE_JUMP_RAD="${FACTR_MAX_STATE_JUMP:-0.20}" \
    INTERVENE_FACTR_HOLD_GRAVITY_COMP="${FACTR_HOLD_GRAVITY_COMP:-1}" \
    INTERVENE_FACTR_HOLD_FRICTION_COMP="${FACTR_HOLD_FRICTION_COMP:-0}" \
    INTERVENE_FACTR_WAIT_HOLD_GRAVITY_COMP="${FACTR_HOLD_GRAVITY_COMP:-1}" \
    INTERVENE_FACTR_WAIT_HOLD_FRICTION_COMP="${FACTR_HOLD_FRICTION_COMP:-0}" \
    INTERVENE_FACTR_RETURN_REST_ON_CLOSE="1" \
    INTERVENE_FACTR_REST_TIMEOUT="${FACTR_REST_TIMEOUT:-25.0}" \
    INTERVENE_FACTR_REST_GOAL_TOLERANCE="${FACTR_REST_GOAL_TOLERANCE:-0.12}" \
    INTERVENE_FACTR_REST_DISABLE_LIMIT_TORQUE="1" \
    FACTR_MAX_REST_TO_INIT_DELTA="${FACTR_MAX_REST_TO_INIT_DELTA:-2.0}" \
    "${PYTHON_BIN}" factr/src/factr_intervention_server.py \
      --host "${FACTR_SERVICE_HOST}" \
      --port "${FACTR_SERVICE_PORT}" \
      --ready-file "${READY_FILE}" \
      --config "${FACTR_CONFIG}" \
      --init-pose "${TARGET_POSE}" \
      --rest-pose "${FACTR_REST_POSE}" \
      --rest-position-tolerance "${FACTR_REST_POSITION_TOLERANCE:-1.5}" \
      --position-tolerance "${FACTR_POSITION_TOLERANCE:-0.08}" \
      --init-timeout "${FACTR_INIT_TIMEOUT:-25.0}" \
      --alignment-timeout "${FACTR_ALIGNMENT_TIMEOUT:-18.0}" \
      --max-velocity "${FACTR_MAX_VELOCITY:-0.5}"
)
