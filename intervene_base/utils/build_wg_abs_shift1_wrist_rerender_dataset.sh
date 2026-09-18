#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/home/user/miniconda3/envs/polymetis/bin/python}

CLEAN_DIR=${CLEAN_DIR:-WG_Test_demonstrations/WG/wire_base_and_spoon_trimmed}
DATA_ROOT=${DATA_ROOT:-lerobot_local_data_wg_wire_base_and_spoon_abs_shift1_wrist_rerender_v1}
DATASET_REPO_ID=${DATASET_REPO_ID:-wg_wire_base_and_spoon_abs_shift1_wrist_rerender}
XML_PATH=${XML_PATH:-mujoco_scenes/working_scenes/with_soft_gripper/sii_scene_table_wire_base_and_spoon.xml}
CAMERA_NAMES=${CAMERA_NAMES:-right,left,wrist}

if [[ ! -d "${CLEAN_DIR}" ]]; then
  echo "[ERROR] Missing cleaned Wire Game folder: ${CLEAN_DIR}" >&2
  echo "[HINT] Use CLEAN_DIR=WG_Test_demonstrations/WG/wire_base_and_spoon if you want the untrimmed demos." >&2
  exit 1
fi

NUM_EPISODES=$(find -L "${CLEAN_DIR}" -maxdepth 1 -type f -name '*.npz' 2>/dev/null | wc -l)
if [[ "${NUM_EPISODES}" -eq 0 ]]; then
  echo "[ERROR] No .npz episodes found in: ${CLEAN_DIR}" >&2
  exit 1
fi

echo "[INFO] Building LeRobot dataset from ${NUM_EPISODES} Wire Game episodes: ${CLEAN_DIR}"
echo "[INFO] Cameras: ${CAMERA_NAMES}"

MUJOCO_GL=${MUJOCO_GL:-egl} "${PYTHON_BIN}" utils/convert_npz_to_lerobot.py \
  --xml "${XML_PATH}" \
  --episodes_dir "${CLEAN_DIR}" \
  --repo_id "${DATASET_REPO_ID}" \
  --root "${DATA_ROOT}" \
  --task "wire game spoon on wire base" \
  --fps 10 \
  --action_shift 1 \
  --camera_names "${CAMERA_NAMES}" \
  --rerender_images \
  --overwrite
