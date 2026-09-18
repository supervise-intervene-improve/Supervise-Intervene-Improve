#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/home/user/miniforge3/envs/polymetis/bin/python}
# XML=${XML:-/path/to/Intervention_IL_AR/intervene_base/mujoco_scenes/working_scenes/with_soft_gripper/sii_scene_table_boxes_cups_multiwindow_fast.xml}
XML=${XML:-/path/to/Intervention_IL_AR/intervene_base/mujoco_scenes/working_scenes/with_soft_gripper/sii_scene_table_boxes_cups.xml}
RESET_NPZ=${RESET_NPZ:-/path/to/Intervention_IL_AR/intervene_base/RESET_NPZs/CUPS/p1_ep_0001_1778052422_trimmed.npz}
CHECKPOINT=${CHECKPOINT:-/path/to/Intervention_IL_AR/intervene_base/MODEL_WEIGHTS/cups_act_16_16_novae_d005_ff4096_20260622_174446/checkpoints/0600000/pretrained_model}

if [[ -z "${CHECKPOINT:-}" ]]; then
  if [[ -z "${MODEL_RUN:-}" ]]; then
    MODEL_RUN=$(
      find outputs -maxdepth 1 -type d \
        \( -name 'cups_act_abs_shift1_wrist_rerender_gripper_fixed_*' -o -name 'cups_act_abs_shift1_wrist_rerender_*' -o -name 'cups_act_abs_shift1_rerender_*' \) \
        -printf '%T@ %p\n' \
        | sort -n \
        | tail -1 \
        | cut -d' ' -f2-
    )
  fi

  if [[ -z "${MODEL_RUN:-}" || ! -d "${MODEL_RUN}" ]]; then
    echo "[ERROR] Could not find a cups training run. Set MODEL_RUN=outputs/cups_act_abs_shift1_..." >&2
    exit 1
  fi

  CHECKPOINT_STEP=${CHECKPOINT_STEP:-latest}
  if [[ "${CHECKPOINT_STEP}" == "latest" ]]; then
    CHECKPOINT_STEP=$(
      find "${MODEL_RUN}/checkpoints" -mindepth 1 -maxdepth 1 -type d ! -name last \
        -printf '%f\n' \
        | sort -V \
        | tail -1
    )
  fi
  CHECKPOINT="${MODEL_RUN}/checkpoints/${CHECKPOINT_STEP}/pretrained_model"
fi

if [[ ! -d "${CHECKPOINT}" ]]; then
  echo "[ERROR] Missing checkpoint: ${CHECKPOINT}" >&2
  echo "[HINT] Set CHECKPOINT=/path/to/pretrained_model or MODEL_RUN=outputs/your_run CHECKPOINT_STEP=latest." >&2
  exit 1
fi

"${PYTHON_BIN}" utils/rollout_act_mujoco.py \
  --task cups \
  --checkpoint "${CHECKPOINT}" \
  --xml "${XML}" \
  --reset_npz "${RESET_NPZ}" \
  --viewer_mode "${VIEWER_MODE:-multicam}" \
  --viewer_cameras "${VIEWER_CAMERAS:-front=front,left=VIS_LEFT,right=VIS_RIGHT,wrist=wrist}" \
  --realtime_factor "${REALTIME_FACTOR:-2.0}" \
  --max_steps "${MAX_STEPS:-0}" \
  --stop_on_success "${STOP_ON_SUCCESS:-false}"
