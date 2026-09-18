#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/home/user/miniconda3/envs/polymetis/bin/python}

RAW_DIR=${RAW_DIR:-WG_Test_demonstrations/WG/wire_base_and_spoon}
CLEAN_DIR=${CLEAN_DIR:-WG_Test_demonstrations/WG/wire_base_and_spoon_trimmed}
XML_PATH=${XML_PATH:-mujoco_scenes/working_scenes/with_soft_gripper/sii_scene_table_wire_base_and_spoon.xml}

BODY_NAME=${BODY_NAME:-spoon1}
TARGET_BODY_NAME=${TARGET_BODY_NAME:-object}
JOINT_MOTION_THRESHOLD=${JOINT_MOTION_THRESHOLD:-0.03}
BODY_MOVE_THRESHOLD=${BODY_MOVE_THRESHOLD:-0.01}
STABLE_SPEED_THRESHOLD=${STABLE_SPEED_THRESHOLD:-0.005}
STABLE_DURATION=${STABLE_DURATION:-0.5}
PRE_MARGIN=${PRE_MARGIN:-0.5}
POST_MARGIN=${POST_MARGIN:-0.5}

if [[ ! -d "${RAW_DIR}" ]]; then
  echo "[ERROR] Missing raw Wire Game folder: ${RAW_DIR}" >&2
  exit 1
fi

if [[ ! -f "${XML_PATH}" ]]; then
  echo "[ERROR] Missing MuJoCo XML: ${XML_PATH}" >&2
  exit 1
fi

NUM_EPISODES=$(find -L "${RAW_DIR}" -maxdepth 1 -type f -name '*.npz' 2>/dev/null | wc -l)
if [[ "${NUM_EPISODES}" -eq 0 ]]; then
  echo "[ERROR] No .npz episodes found in: ${RAW_DIR}" >&2
  exit 1
fi

echo "[INFO] Cleaning ${NUM_EPISODES} Wire Game episodes"
echo "[INFO] Input:  ${RAW_DIR}"
echo "[INFO] Output: ${CLEAN_DIR}"
echo "[INFO] End condition: ${BODY_NAME} contacts ${TARGET_BODY_NAME} and is stable for ${STABLE_DURATION}s"

"${PYTHON_BIN}" utils/data_cleaning.py \
  --input "${RAW_DIR}" \
  --output "${CLEAN_DIR}" \
  --xml "${XML_PATH}" \
  --body-name "${BODY_NAME}" \
  --target-body-name "${TARGET_BODY_NAME}" \
  --end-mode contact-stability \
  --joint-motion-threshold "${JOINT_MOTION_THRESHOLD}" \
  --body-move-threshold "${BODY_MOVE_THRESHOLD}" \
  --stable-speed-threshold "${STABLE_SPEED_THRESHOLD}" \
  --stable-duration "${STABLE_DURATION}" \
  --pre-margin "${PRE_MARGIN}" \
  --post-margin "${POST_MARGIN}"
