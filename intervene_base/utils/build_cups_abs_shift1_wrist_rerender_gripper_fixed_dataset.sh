#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/home/user/miniconda3/envs/polymetis/bin/python}

CLEAN_DIR=${CLEAN_DIR:-data_clean/CUPS_clean_full_gripper_fixed}
DATA_ROOT=${DATA_ROOT:-lerobot_local_data_cups_abs_shift1_wrist_rerender_gripper_fixed_v1}
DATASET_REPO_ID=${DATASET_REPO_ID:-cups_sim_bc_abs_shift1_wrist_rerender_gripper_fixed}
XML_PATH=${XML_PATH:-mujoco_scenes/working_scenes/with_soft_gripper/sii_scene_table_boxes_cups.xml}

if [[ ! -d "${CLEAN_DIR}" ]]; then
  echo "[ERROR] Missing fixed cleaned cups folder: ${CLEAN_DIR}" >&2
  echo "[HINT] Run: python utils/fix_cup_gripper_closed_value.py --input-dir data_clean/CUPS_clean_full --output-dir ${CLEAN_DIR}" >&2
  exit 1
fi

NUM_EPISODES=$(find "${CLEAN_DIR}" -maxdepth 1 -type f -name '*.npz' | wc -l)
echo "[INFO] Building LeRobot dataset from ${NUM_EPISODES} fixed cleaned cup episodes: ${CLEAN_DIR}"

MUJOCO_GL=${MUJOCO_GL:-egl} "${PYTHON_BIN}" utils/convert_npz_to_lerobot.py \
  --xml "${XML_PATH}" \
  --episodes_dir "${CLEAN_DIR}" \
  --repo_id "${DATASET_REPO_ID}" \
  --root "${DATA_ROOT}" \
  --task "red cup into green cup and blue cup into yellow cup" \
  --fps 10 \
  --action_shift 1 \
  --camera_names "right,left,wrist" \
  --rerender_images \
  --overwrite
