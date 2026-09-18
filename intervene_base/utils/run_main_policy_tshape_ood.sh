#!/usr/bin/env bash
set -euo pipefail

# T-shape out-of-distribution policy launcher.
#
# This wraps utils/run_main_policy.sh and only changes scene randomization.
# The normal policy checkpoint, viewer, robot/mirror options, and command-line
# arguments are still handled by run_main_policy.sh.
#
# Default OOD setup:
#   - T1 position is randomized.
#   - T2 position is randomized.
#   - T1 pitch is sampled from one of two far-angle bands:
#       [-75, -45] degrees OR [45, 75] degrees.
#   - T2 pitch randomization is present as an easy switch, but disabled by default.
#
# The T-shape XML starts like:
#   <body name="T1" ... euler="1.5708 3.14159 0">
# This OOD launcher sets the free-joint quaternion from exactly:
#   euler = 1.5708, 3.14159 + sampled_delta, 0
# at reset time. It does not edit the XML file on disk.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export XML="${XML:-mujoco_scenes/working_scenes/with_soft_gripper/sii_scene_table_T_shape.xml}"
export RESET_NPZ="${RESET_NPZ:-RESET_NPZs/TSHAPE/p1_ep_0001_1778506313_trimmed.npz}"
export INTERVENE_RANDOMIZATION_PRESET="${INTERVENE_RANDOMIZATION_PRESET:-tshape}"
export INTERVENE_EPISODE_DIR="${INTERVENE_EPISODE_DIR:-INTERVENTION_DATA/tshape_ood_45_75_helped}"




# Position randomization for both T objects.
# T1/T2 positions are based on the XML body pos, not the reset NPZ, so the
# object z stays on the table instead of inheriting a lifted demo-frame z.
# Increase INTERVENE_OBJECT_XY_RANDOM_RANGE for more table-position variation.
export INTERVENE_RANDOMIZE_OBJECT_NAMES="${INTERVENE_RANDOMIZE_OBJECT_NAMES:-T1,T2}"
export INTERVENE_OBJECT_POSITION_FROM_XML_NAMES="${INTERVENE_OBJECT_POSITION_FROM_XML_NAMES:-T1,T2}"
export INTERVENE_OBJECT_SUPPORT_Z_NAMES="${INTERVENE_OBJECT_SUPPORT_Z_NAMES:-T1,T2}"
export INTERVENE_OBJECT_SUPPORT_Z="${INTERVENE_OBJECT_SUPPORT_Z:-0.22}"
export INTERVENE_OBJECT_SUPPORT_MARGIN="${INTERVENE_OBJECT_SUPPORT_MARGIN:-0.0005}"
export INTERVENE_OBJECT_ZERO_QVEL_NAMES="${INTERVENE_OBJECT_ZERO_QVEL_NAMES:-T1,T2}"
export INTERVENE_OBJECT_XY_RANDOM_RANGE="${INTERVENE_OBJECT_XY_RANDOM_RANGE:-0.01}"
export INTERVENE_OBJECT_X_BOUNDS="${INTERVENE_OBJECT_X_BOUNDS:-0.40 0.80}"
export INTERVENE_OBJECT_Y_BOUNDS="${INTERVENE_OBJECT_Y_BOUNDS:--0.25 0.25}"

# Orientation randomization.
#
# By default only T1 gets the OOD pitch. This is the main distribution shift.
# If you also want to randomize T2 orientation, run for example:
#   INTERVENE_RANDOMIZE_EULER_NAMES=T1,T2 \
#   INTERVENE_OBJECT_PITCH_RANDOM_BANDS_DEG='T1:-75:-45|45:75;T2:-15:15' \
#   bash utils/run_main_policy_tshape_ood.sh
export INTERVENE_RANDOMIZE_EULER_NAMES="${INTERVENE_RANDOMIZE_EULER_NAMES:-T1}"
export INTERVENE_OBJECT_EULER_RANDOM_RANGE_DEG="${INTERVENE_OBJECT_EULER_RANDOM_RANGE_DEG:-0 0 0}"
export INTERVENE_OBJECT_BASE_EULER_RAD="${INTERVENE_OBJECT_BASE_EULER_RAD:-T1:1.5708:3.14159:0}"
export INTERVENE_OBJECT_PITCH_RANDOM_BANDS_DEG="${INTERVENE_OBJECT_PITCH_RANDOM_BANDS_DEG:-T1:-45:-75|75:45}"

echo "[TShapeOOD] Base euler: ${INTERVENE_OBJECT_BASE_EULER_RAD}"
echo "[TShapeOOD] T1 pitch bands: ${INTERVENE_OBJECT_PITCH_RANDOM_BANDS_DEG}"
echo "[TShapeOOD] XML-position objects: ${INTERVENE_OBJECT_POSITION_FROM_XML_NAMES}"
echo "[TShapeOOD] Table support: ${INTERVENE_OBJECT_SUPPORT_Z_NAMES} min_z=${INTERVENE_OBJECT_SUPPORT_Z}+${INTERVENE_OBJECT_SUPPORT_MARGIN}"
echo "[TShapeOOD] Position objects: ${INTERVENE_RANDOMIZE_OBJECT_NAMES} (+/-${INTERVENE_OBJECT_XY_RANDOM_RANGE} m)"
echo "[TShapeOOD] Orientation objects: ${INTERVENE_RANDOMIZE_EULER_NAMES}"
echo "[TShapeOOD] Episode dir: ${INTERVENE_EPISODE_DIR}"

# temp
export INTERVENE_ENABLE_SIMPUBLISHER="${INTERVENE_ENABLE_SIMPUBLISHER:-1}"
# Temporary OOD collection mode: you decide the label manually.
# Press S for success or F for failure in the viewer.
export INTERVENE_AUTO_TASK_EVAL="${INTERVENE_AUTO_TASK_EVAL:-0}"
echo "[TShapeOOD] Auto task eval: ${INTERVENE_AUTO_TASK_EVAL} (manual S/F labels)"

exec bash "${SCRIPT_DIR}/run_main_policy.sh" "$@"
