#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/home/user/miniconda3/envs/polymetis/bin/python}
NUM_EPISODES=${NUM_EPISODES:-0}
REPEATS_PER_EPISODE=${REPEATS_PER_EPISODE:-1}
MAX_STEPS=${MAX_STEPS:-600}
TIMEOUT_SECONDS=${TIMEOUT_SECONDS:-15}
STOP_ON_SUCCESS=${STOP_ON_SUCCESS:-true}
XY_JITTER=${XY_JITTER:-0.0}
YAW_JITTER_DEG=${YAW_JITTER_DEG:-0.0}
PLACEMENT_XY_THRESHOLD=${PLACEMENT_XY_THRESHOLD:-0.04}
PLACEMENT_TARGET_LOCAL_OFFSET=${PLACEMENT_TARGET_LOCAL_OFFSET:-"0 0 0.12"}
PLACEMENT_Z_OFFSET=${PLACEMENT_Z_OFFSET:-0.165}
PLACEMENT_Z_TOLERANCE=${PLACEMENT_Z_TOLERANCE:-0.025}
PLACEMENT_YAW_THRESHOLD_DEG=${PLACEMENT_YAW_THRESHOLD_DEG:-25}
PLACEMENT_GRIPPER_OPEN_THRESHOLD=${PLACEMENT_GRIPPER_OPEN_THRESHOLD:-0.025}
PLACEMENT_STABLE_STEPS=${PLACEMENT_STABLE_STEPS:-10}
RECORD_VIDEOS=${RECORD_VIDEOS:-none}
SUCCESS_VIDEO_LIMIT=${SUCCESS_VIDEO_LIMIT:-0}
STOP_AFTER_SUCCESS_VIDEOS=${STOP_AFTER_SUCCESS_VIDEOS:-false}
VIDEO_CAMERA=${VIDEO_CAMERA:-"front,VIS_RIGHT,top"}
VIDEO_WIDTH=${VIDEO_WIDTH:-480}
VIDEO_HEIGHT=${VIDEO_HEIGHT:-270}
VIDEO_FPS=${VIDEO_FPS:-0}
VIDEO_EVERY_N_STEPS=${VIDEO_EVERY_N_STEPS:-1}

if [[ -z "${MODEL_RUN:-}" ]]; then
  MODEL_RUN=$(
    find outputs -maxdepth 1 -type d \
      \( -name 'tshape_act_abs_shift1_rerender*' -o -name 'tshape_act_abs_shift1_wrist_rerender*' \) \
      -printf '%T@ %p\n' \
      | sort -n \
      | tail -1 \
      | cut -d' ' -f2-
  )
fi

if [[ -z "${MODEL_RUN}" || ! -d "${MODEL_RUN}" ]]; then
  echo "[ERROR] Could not find a T-shape training run. Set MODEL_RUN=outputs/tshape_act_abs_shift1_rerender_..." >&2
  exit 1
fi

CHECKPOINT_STEP=${CHECKPOINT_STEP:-latest}
if [[ -z "${CHECKPOINT:-}" ]]; then
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

RUN_LABEL=${RUN_LABEL:-shift1_xy${XY_JITTER}_yaw${YAW_JITTER_DEG}_r${REPEATS_PER_EPISODE}}
"${PYTHON_BIN}" utils/evaluate_act_mujoco.py \
  --checkpoint "${CHECKPOINT}" \
  --output_dir "$MODEL_RUN/evals/${RUN_LABEL}_$(date +%Y%m%d_%H%M%S)" \
  --xml mujoco_scenes/working_scenes/with_soft_gripper/sii_scene_table_T_shape.xml \
  --episodes_dir data_clean/T_shape_clean_good \
  --num_episodes "${NUM_EPISODES}" \
  --repeats_per_episode "${REPEATS_PER_EPISODE}" \
  --max_steps "${MAX_STEPS}" \
  --timeout_seconds "${TIMEOUT_SECONDS}" \
  --stop_on_success "${STOP_ON_SUCCESS}" \
  --xy_jitter "${XY_JITTER}" \
  --yaw_jitter_deg "${YAW_JITTER_DEG}" \
  --action_mode queue \
  --arm_action_mode absolute \
  --gripper_action_mode absolute \
  --target_body_name T2 \
  --placement_xy_threshold "${PLACEMENT_XY_THRESHOLD}" \
  --placement_target_local_offset "${PLACEMENT_TARGET_LOCAL_OFFSET}" \
  --placement_z_offset "${PLACEMENT_Z_OFFSET}" \
  --placement_z_tolerance "${PLACEMENT_Z_TOLERANCE}" \
  --placement_yaw_threshold_deg "${PLACEMENT_YAW_THRESHOLD_DEG}" \
  --placement_gripper_open_threshold "${PLACEMENT_GRIPPER_OPEN_THRESHOLD}" \
  --placement_stable_steps "${PLACEMENT_STABLE_STEPS}" \
  --record_videos "${RECORD_VIDEOS}" \
  --success_video_limit "${SUCCESS_VIDEO_LIMIT}" \
  --stop_after_success_videos "${STOP_AFTER_SUCCESS_VIDEOS}" \
  --video_camera "${VIDEO_CAMERA}" \
  --video_width "${VIDEO_WIDTH}" \
  --video_height "${VIDEO_HEIGHT}" \
  --video_fps "${VIDEO_FPS}" \
  --video_every_n_steps "${VIDEO_EVERY_N_STEPS}" \
  "$@"
