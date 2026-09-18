#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
export MUJOCO_GL=${MUJOCO_GL:-glfw}
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  TORCH_LIB="$("${PYTHON_BIN}" - <<'PY'
import pathlib
import torch
print(pathlib.Path(torch.__file__).resolve().parent / "lib")
PY
)"
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${TORCH_LIB}:${PWD}:${LD_LIBRARY_PATH:-}"
fi

XML=${XML:-mujoco_scenes/working_scenes/with_soft_gripper/sii_scene_table_T_shape.xml}
RESET_NPZ=${RESET_NPZ:-RESET_NPZs/TSHAPE/p1_ep_0001_1778506313_trimmed.npz}
NUM_TILES=${NUM_TILES:-15}
if [[ -z "${NPZ_PATHS:-}" ]]; then
  NPZ_PATHS=""
  for _ in $(seq 1 "${NUM_TILES}"); do
    NPZ_PATHS+="${RESET_NPZ} "
  done
  NPZ_PATHS="${NPZ_PATHS% }"
fi

CHECKPOINT=${CHECKPOINT:-MODEL_WEIGHTS/tshape_act_abs_10_10_novae_512_2048_20260622_171636/\
checkpoints/0950000/pretrained_model}
ROBOT_KEY=${ROBOT_KEY:-${INTERVENE_ROBOT_KEY:-p4}}
export INTERVENE_ROBOT_KEY=${INTERVENE_ROBOT_KEY:-${ROBOT_KEY}}

is_valid_checkpoint() {
  local checkpoint_dir="$1"
  [[ -s "${checkpoint_dir}/config.json" ]] \
    && [[ -s "${checkpoint_dir}/model.safetensors" ]] \
    && [[ -s "${checkpoint_dir}/policy_preprocessor_step_3_normalizer_processor.safetensors" ]] \
    && [[ -s "${checkpoint_dir}/policy_postprocessor_step_0_unnormalizer_processor.safetensors" ]]
}

if [[ ! -f "${XML}" ]]; then
  echo "[ERROR] Missing XML: ${XML}" >&2
  exit 1
fi

if [[ ! -d "${CHECKPOINT}" ]]; then
  echo "[ERROR] Missing checkpoint: ${CHECKPOINT}" >&2
  echo "[HINT] Set CHECKPOINT=/path/to/pretrained_model." >&2
  exit 1
fi

if ! is_valid_checkpoint "${CHECKPOINT}"; then
  echo "[ERROR] Incomplete checkpoint: ${CHECKPOINT}" >&2
  echo "[HINT] It must contain config.json, model.safetensors, and pre/postprocessor normalizer files." >&2
  exit 1
fi

read -r -a NPZ_ARRAY <<< "${NPZ_PATHS}"
if [[ "${#NPZ_ARRAY[@]}" -lt 1 ]]; then
  echo "[ERROR] NPZ_PATHS must contain at least one reset NPZ." >&2
  exit 1
fi

for npz_path in "${NPZ_ARRAY[@]}"; do
  if [[ ! -f "${npz_path}" ]]; then
    echo "[ERROR] Missing reset NPZ: ${npz_path}" >&2
    exit 1
  fi
done

EXTRA_ARGS=()
if [[ -n "${ARM_DELTA_CLIP:-}" ]]; then
  EXTRA_ARGS+=(--arm_delta_clip "${ARM_DELTA_CLIP}")
fi

"${PYTHON_BIN}" multi_main.py \
  "${XML}" \
  "${NPZ_ARRAY[@]}" \
  --mode policy \
  --checkpoint "${CHECKPOINT}" \
  --view "${VIEW:-cameras}" \
  --action_mode "${ACTION_MODE:-queue}" \
  --arm_action_mode "${ARM_ACTION_MODE:-absolute}" \
  --gripper_action_mode "${GRIPPER_ACTION_MODE:-absolute}" \
  --policy_hz "${POLICY_HZ:-10}" \
  --realtime_factor "${REALTIME_FACTOR:-2.0}" \
  --max_steps "${MAX_STEPS:-0}" \
  --width "${WINDOW_W:-1600}" \
  --height "${WINDOW_H:-900}" \
  --robot_key "${ROBOT_KEY}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
