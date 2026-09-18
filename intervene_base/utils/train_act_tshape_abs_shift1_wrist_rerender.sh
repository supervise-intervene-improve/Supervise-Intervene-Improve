#!/usr/bin/env bash
set -euo pipefail

LEROBOT_TRAIN=${LEROBOT_TRAIN:-/home/user/miniconda3/envs/polymetis/bin/lerobot-train}

export HF_HOME=${HF_HOME:-./.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-${HF_HOME}/datasets}
export HF_HUB_CACHE=${HF_HUB_CACHE:-${HF_HOME}/hub}

DATA_ROOT=${DATA_ROOT:-./lerobot_local_data_tshape_abs_shift1_wrist_rerender_v1}
DATASET_REPO_ID=${DATASET_REPO_ID:-tshape_sim_bc_abs_shift1_wrist_rerender}
DEVICE=${DEVICE:-cuda}
BATCH_SIZE=${BATCH_SIZE:-8}
STEPS=${STEPS:-200000}
SAVE_FREQ=${SAVE_FREQ:-20000}
EVAL_FREQ=${EVAL_FREQ:-20000}
NUM_WORKERS=${NUM_WORKERS:-4}
RESUME_CHECKPOINT=${RESUME_CHECKPOINT:-}
POLICY_LR=${POLICY_LR:-1e-4}
BACKBONE_LR=${BACKBONE_LR:-1e-5}

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "[ERROR] Missing LeRobot dataset root: ${DATA_ROOT}" >&2
  echo "[HINT] Run ./utils/build_tshape_abs_shift1_wrist_rerender_dataset.sh first." >&2
  exit 1
fi

if [[ ! -f "${DATA_ROOT}/meta/info.json" ]]; then
  echo "[ERROR] Dataset root does not look like a LeRobot dataset: ${DATA_ROOT}" >&2
  echo "[HINT] Expected ${DATA_ROOT}/meta/info.json" >&2
  exit 1
fi

if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  if [[ -f "${RESUME_CHECKPOINT}/pretrained_model/train_config.json" ]]; then
    CONFIG_PATH="${RESUME_CHECKPOINT}/pretrained_model/train_config.json"
  else
    CONFIG_PATH="${RESUME_CHECKPOINT}/train_config.json"
  fi

  if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "[ERROR] Missing resume config: ${CONFIG_PATH}" >&2
    exit 1
  fi

  echo "[INFO] Resume config: ${CONFIG_PATH}"
  echo "[INFO] Training until total steps: ${STEPS}"

  "${LEROBOT_TRAIN}" \
    --config_path="${CONFIG_PATH}" \
    --resume=true \
    --steps=${STEPS} \
    --save_freq=${SAVE_FREQ} \
    --eval_freq=${EVAL_FREQ} \
    --num_workers=${NUM_WORKERS} \
    --batch_size=${BATCH_SIZE}

  exit 0
fi

RUN_ID=$(date +%Y%m%d_%H%M%S)
OUTDIR=./outputs/tshape_act_abs_shift1_wrist_rerender_${RUN_ID}
MODEL_REPO_ID=anonymous/tshape_act_abs_shift1_wrist_rerender_${RUN_ID}

"${LEROBOT_TRAIN}" \
  --dataset.repo_id=${DATASET_REPO_ID} \
  --dataset.root=${DATA_ROOT} \
  --policy.type=act \
  --policy.chunk_size=10 \
  --policy.n_action_steps=10 \
  --policy.dim_model=512 \
  --policy.dim_feedforward=3200 \
  --policy.n_heads=8 \
  --policy.n_encoder_layers=4 \
  --policy.n_decoder_layers=1 \
  --policy.use_vae=false \
  --policy.dropout=0.0 \
  --policy.optimizer_lr=${POLICY_LR} \
  --policy.optimizer_lr_backbone=${BACKBONE_LR} \
  --output_dir=${OUTDIR} \
  --job_name=tshape_act_abs_shift1_wrist_rerender_${RUN_ID} \
  --policy.repo_id=${MODEL_REPO_ID} \
  --policy.push_to_hub=false \
  --policy.device=${DEVICE} \
  --batch_size=${BATCH_SIZE} \
  --num_workers=${NUM_WORKERS} \
  --steps=${STEPS} \
  --save_freq=${SAVE_FREQ} \
  --eval_freq=${EVAL_FREQ}
