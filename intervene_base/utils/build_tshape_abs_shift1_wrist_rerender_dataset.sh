#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/home/user/miniconda3/envs/polymetis/bin/python}

CLEAN_GOOD_DIR=${CLEAN_GOOD_DIR:-data_clean/T_shape_clean_good}
DATA_ROOT=${DATA_ROOT:-lerobot_local_data_tshape_abs_shift1_wrist_rerender_v1}
DATASET_REPO_ID=${DATASET_REPO_ID:-tshape_sim_bc_abs_shift1_wrist_rerender}
XML_PATH=${XML_PATH:-mujoco_scenes/working_scenes/with_soft_gripper/sii_scene_table_T_shape.xml}
CAMERA_NAMES=${CAMERA_NAMES:-right,left,wrist}

if [[ ! -d "${CLEAN_GOOD_DIR}" ]]; then
  echo "[ERROR] Missing cleaned T-shape folder: ${CLEAN_GOOD_DIR}" >&2
  exit 1
fi

NUM_EPISODES=$(find -L "${CLEAN_GOOD_DIR}" -maxdepth 1 -type f -name '*.npz' 2>/dev/null | wc -l)
echo "[INFO] Building LeRobot dataset from ${NUM_EPISODES} T-shape episodes: ${CLEAN_GOOD_DIR}"
echo "[INFO] Cameras: ${CAMERA_NAMES}"

MUJOCO_GL=${MUJOCO_GL:-egl} "${PYTHON_BIN}" utils/convert_npz_to_lerobot.py \
  --xml "${XML_PATH}" \
  --episodes_dir "${CLEAN_GOOD_DIR}" \
  --repo_id "${DATASET_REPO_ID}" \
  --root "${DATA_ROOT}" \
  --task "T-shape pick and place" \
  --fps 10 \
  --action_shift 1 \
  --camera_names "${CAMERA_NAMES}" \
  --rerender_images \
  --overwrite
