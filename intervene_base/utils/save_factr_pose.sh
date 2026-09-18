#!/usr/bin/env bash
# Capture the current FACTR joints and gripper without enabling motor torque.

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERVENE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

usage() {
  printf '%s\n' \
    "Usage: bash utils/save_factr_pose.sh [OUTPUT.json]" \
    "" \
    "The capture is read-only: it does not enable or command motor torque." \
    "If OUTPUT.json is omitted, a timestamped candidate is written under:" \
    "  ${INTERVENE_ROOT}/factr/validation/candidates/" \
    "" \
    "Active rest_pose.json and init_pose.json files are protected from" \
    "direct overwrite. Review a candidate before installing it."
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
      && "${python_candidate}" -c 'import dynamixel_sdk, pinocchio, yaml' \
        >/dev/null 2>&1; then
    PYTHON_BIN="${python_candidate}"
    break
  fi
done

if [[ -z "${PYTHON_BIN}" ]]; then
  echo "[FACTRSave][ERROR] No Python interpreter can import the FACTR stack." >&2
  echo "[FACTRSave] Activate polymetis or set FACTR_PYTHON_BIN explicitly." >&2
  exit 1
fi

capture_timestamp="$(date +%Y%m%d_%H%M%S)"
OUTPUT_PATH="${1:-${INTERVENE_ROOT}/factr/validation/candidates/factr_pose_${capture_timestamp}.json}"
if [[ "${OUTPUT_PATH}" != /* ]]; then
  OUTPUT_PATH="${INTERVENE_ROOT}/${OUTPUT_PATH}"
fi
OUTPUT_PATH="$(realpath -m "${OUTPUT_PATH}")"

ACTIVE_REST="$(realpath -m "${INTERVENE_ROOT}/factr/validation/rest_pose.json")"
ACTIVE_INIT="$(realpath -m "${INTERVENE_ROOT}/factr/validation/init_pose.json")"
if [[ "${OUTPUT_PATH}" == "${ACTIVE_REST}" || "${OUTPUT_PATH}" == "${ACTIVE_INIT}" ]]; then
  echo "[FACTRSave][ERROR] Refusing to overwrite active pose: ${OUTPUT_PATH}" >&2
  echo "[FACTRSave] Save a candidate and review it before updating the active file." >&2
  exit 1
fi

if pgrep -f '[f]actr/src/factr_intervention_server\.py|[g]o_to_ref_pose\.py' \
    >/dev/null 2>&1; then
  echo "[FACTRSave][ERROR] Another FACTR process is already using the serial device." >&2
  echo "[FACTRSave] Stop it cleanly before capturing a pose." >&2
  exit 1
fi

FACTR_CONFIG="${FACTR_CONFIG:-${INTERVENE_ROOT}/factr/leader.yaml}"
CALIBRATION_REPORT="${INTERVENE_FACTR_CALIBRATION_REPORT:-${INTERVENE_ROOT}/factr/validation/calibration.json}"

echo "[FACTRSave] Python: ${PYTHON_BIN}"
echo "[FACTRSave] Output: ${OUTPUT_PATH}"
echo "[FACTRSave] Reading one stationary sample; motor torque will not be enabled."

(
  cd "${INTERVENE_ROOT}"
  "${PYTHON_BIN}" factr/src/go_to_ref_pose.py \
    --save-factr-pose "${OUTPUT_PATH}" \
    --factr-config "${FACTR_CONFIG}" \
    --factr-calibration-report "${CALIBRATION_REPORT}"
)

echo "[FACTRSave] Candidate saved: ${OUTPUT_PATH}"
