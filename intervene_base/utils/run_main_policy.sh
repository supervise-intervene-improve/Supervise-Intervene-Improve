#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INTERVENE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
UMBRELLA_ROOT="${UMBRELLA_ROOT:-$(cd "${INTERVENE_ROOT}/.." && pwd)}"
cd "${INTERVENE_ROOT}"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  for candidate in \
    "$HOME/miniforge3/envs/polymetis/bin/python" \
    "$HOME/miniconda3/envs/polymetis/bin/python" \
    python
  do
    if command -v "$candidate" >/dev/null 2>&1 || [[ -x "$candidate" ]]; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi
PYTHON_BIN=${PYTHON_BIN:-python}
export MUJOCO_GL=${MUJOCO_GL:-glfw}
TORCH_ENV_INFO="$("${PYTHON_BIN}" - <<'PY' || true
import pathlib
import sys
import torch
print(sys.prefix)
print(pathlib.Path(torch.__file__).resolve().parent / "lib")
PY
)"
if [[ -z "${TORCH_ENV_INFO}" ]]; then
  echo "[ERROR] ${PYTHON_BIN} cannot import torch." >&2
  echo "[HINT] Use the policy env, e.g. PYTHON_BIN=$HOME/miniforge3/envs/polymetis/bin/python" >&2
  exit 1
fi
POLICY_PREFIX="$(printf '%s\n' "${TORCH_ENV_INFO}" | sed -n '1p')"
TORCH_LIB="$(printf '%s\n' "${TORCH_ENV_INFO}" | sed -n '2p')"
export LD_LIBRARY_PATH="${POLICY_PREFIX}/lib:${TORCH_LIB}:${PWD}:${LD_LIBRARY_PATH:-}"

if [[ -z "${POLYMETIS_PYTHON_DIR:-}" ]]; then
  POLYMETIS_CANDIDATE="${UMBRELLA_ROOT}/src/polymetis/polymetis/python"
  if [[ -d "${POLYMETIS_CANDIDATE}" ]]; then
    POLYMETIS_PYTHON_DIR="${POLYMETIS_CANDIDATE}"
  fi
fi
if [[ -n "${POLYMETIS_PYTHON_DIR:-}" ]]; then
  export PYTHONPATH="${POLYMETIS_PYTHON_DIR}:${PYTHONPATH:-}"
fi

# --- HRI experiment block (2026-08-11) -----------------------------------------
# One command starts a recorded participant block:
#
#   RGB=1 WINDOWS=9 LAB=tshape FPS=60 \
#   EXP_PARTICIPANT=P003 EXP_STUDY=Supervision EXP_TASK=TShape \
#     bash utils/run_main_policy.sh
#
# Three labels only: WHO, WHICH STUDY, WHICH TASK. The condition is INFERRED from the
# flags already on the command line, so it is never typed twice:
#
#   Supervision (KT throughout)   START_VR=0 -> S1 | RGB=1 -> S2 | neither -> S3
#   Control     (VR-PC throughout) MC_ACTIVE=1 MC_SIM=1 -> C2 | FACTR_ACTIVE=1 -> C3
#                                  | neither -> C1
#
# EXP_CONDITION may still be set explicitly; it is then cross-checked against the flags.
#
# `supervision_interface` and `controller_interface` are DERIVED from (study,
# condition) by data_io/experiment_block.py. Each condition is normally launched with
# its own explicit flags (RGB=1 / MC_ACTIVE=1 MC_SIM=1 / FACTR_ACTIVE=1 / START_VR=0),
# so those ALWAYS win — the condition only fills in what was not typed. The resolved
# flags are then VERIFIED against the condition and a contradiction aborts before
# anything launches, because a block labelled as something the participant never saw
# cannot be fixed after they leave.
#
# Leaving EXP_PARTICIPANT unset keeps every previous behaviour, including the older
# STUDY_PARTICIPANT flow.
EXP_PARTICIPANT="${EXP_PARTICIPANT:-}"
EXP_STUDY="${EXP_STUDY:-}"
EXP_CONDITION="${EXP_CONDITION:-}"
EXP_TASK="${EXP_TASK:-}"
EXP_ROOT="${EXP_ROOT:-${INTERVENE_ROOT}/experiment_data}"
EXP_SEED="${EXP_SEED:-}"
EXP_NOTES="${EXP_NOTES:-}"
EXP_CELLS="${EXP_CELLS:-${WINDOWS:-9}}"

if [[ -n "${EXP_PARTICIPANT}" ]]; then
  # EXP_CONDITION is OPTIONAL. Within a study the flags already determine it uniquely
  # (Supervision varies only the supervision interface, Control only the controller),
  # so it is inferred from what you typed rather than typed twice. Passing it
  # explicitly still works and is then cross-checked below.
  #   Supervision: START_VR=0 -> S1 | RGB=1 -> S2 | neither -> S3
  #   Control:     MC_ACTIVE=1 -> C2 | FACTR_ACTIVE=1 -> C3 | neither -> C1
  # Defaults here (START_VR=1, RGB=0, everything else 0) are the launcher's own, so
  # "no flags at all" infers the plain VR-pointcloud + KT condition.
  if [[ -z "${EXP_CONDITION}" ]]; then
    if ! _exp_inferred="$(PYTHONPATH="${INTERVENE_ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" \
          -m data_io.experiment_block infer --quiet --study "${EXP_STUDY}" \
          --start_vr "${START_VR:-1}" --rgb "${RGB:-0}" --with_pc "${WITH_PC:-1}" \
          --mc_active "${MC_ACTIVE:-0}" --mc_sim "${MC_SIM:-0}" \
          --factr_active "${FACTR_ACTIVE:-0}" 2>&1)"; then
      echo "${_exp_inferred}" >&2
      echo "[ERROR] Could not infer EXP_CONDITION from the launch flags. Nothing was launched." >&2
      echo "[HINT] Either fix the flags, switch EXP_STUDY, or set EXP_CONDITION explicitly." >&2
      exit 1
    fi
    EXP_CONDITION="${_exp_inferred}"
    echo "[PolicyLaunch] BLOCK: inferred EXP_CONDITION=${EXP_CONDITION} from the launch flags."
  fi

  _exp_start_args=(--participant "${EXP_PARTICIPANT}" --study "${EXP_STUDY}"
                   --condition "${EXP_CONDITION}" --task "${EXP_TASK}"
                   --root "${EXP_ROOT}" --cells "${EXP_CELLS}" --export)
  if [[ -n "${EXP_SEED}" ]]; then _exp_start_args+=(--seed "${EXP_SEED}"); fi
  if [[ -n "${EXP_NOTES}" ]]; then _exp_start_args+=(--notes "${EXP_NOTES}"); fi

  # Creates the block directory and writes block_metadata.json + BLOCK_START BEFORE any
  # process starts, so an interrupted block is still fully identified.
  if ! _exp_out="$(PYTHONPATH="${INTERVENE_ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" \
        -m data_io.experiment_block start "${_exp_start_args[@]}" 2>&1)"; then
    echo "${_exp_out}" >&2
    echo "[ERROR] Could not start the experiment block. Nothing was launched." >&2
    echo "[HINT] Check the combination first:" >&2
    echo "       ${PYTHON_BIN} -m data_io.experiment_block validate --participant ${EXP_PARTICIPANT} --study ${EXP_STUDY} --condition ${EXP_CONDITION} --task ${EXP_TASK}" >&2
    echo "[HINT] studies: Supervision (S1/S2/S3) or Control (C1/C2/C3)." >&2
    exit 1
  fi
  # Non-export lines are the human-readable banner; the export lines wire the children.
  printf '%s\n' "${_exp_out}" | grep -v '^export ' || true
  eval "$(printf '%s\n' "${_exp_out}" | grep '^export ')"
  export EXP_NOTES EXP_CELLS

  _exp_meta="$(PYTHONPATH="${INTERVENE_ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" - <<'PY'
import sys
from data_io.experiment_block import resolve_block_from_env
b = resolve_block_from_env()
f = b.launch_flags
print("\t".join([b.study, b.task_mode, b.supervision_interface, b.controller_interface,
                 f["START_VR"], f["RGB"], f["WITH_PC"],
                 f["MC_ACTIVE"], f["MC_SIM"], f["FACTR_ACTIVE"]]))
PY
)" || { echo "[ERROR] Could not resolve the started block." >&2; exit 1; }
  IFS=$'\t' read -r EXP_STUDY_NAME EXP_TASK_MODE EXP_SUPERVISION EXP_CONTROLLER \
                    EXP_F_START_VR EXP_F_RGB EXP_F_WITH_PC \
                    EXP_F_MC_ACTIVE EXP_F_MC_SIM EXP_F_FACTR <<<"${_exp_meta}"

  # Fill in only what the operator did NOT type. Each condition is normally launched as
  # its own command line (RGB=1 WINDOWS=9 LAB=cups ...), so an explicit value always
  # wins here — and is then checked against the condition below.
  LAB="${LAB:-${EXP_TASK_MODE}}"
  START_VR="${START_VR:-${EXP_F_START_VR}}"
  START_GRID_UI="${START_GRID_UI:-1}"
  RGB="${RGB:-${EXP_F_RGB}}"
  WITH_PC="${WITH_PC:-${EXP_F_WITH_PC}}"
  MC_ACTIVE="${MC_ACTIVE:-${EXP_F_MC_ACTIVE}}"
  MC_SIM="${MC_SIM:-${EXP_F_MC_SIM}}"
  FACTR_ACTIVE="${FACTR_ACTIVE:-${EXP_F_FACTR}}"
  WINDOWS="${WINDOWS:-${EXP_CELLS}}"
  # RGB / WITH_PC reach the VR runner by inheritance, so they must be exported —
  # a shell-local assignment would silently launch the wrong visualisation.
  export RGB WITH_PC

  # VERIFY. The block label and the flags the participant actually experiences must
  # agree; a mistyped EXP_CONDITION is otherwise unrecoverable after they leave.
  _exp_flag_mismatch=""
  _exp_check_flag() {   # name expected actual
    local truthy_e truthy_a
    case "${2}" in 1|true|TRUE|yes|YES|on|ON) truthy_e=1 ;; *) truthy_e=0 ;; esac
    case "${3}" in 1|true|TRUE|yes|YES|on|ON) truthy_a=1 ;; *) truthy_a=0 ;; esac
    if [[ "${truthy_e}" != "${truthy_a}" ]]; then
      _exp_flag_mismatch+="    ${1}: condition expects ${truthy_e}, launch has ${truthy_a}"$'\n'
    fi
  }
  _exp_check_flag START_VR     "${EXP_F_START_VR}"   "${START_VR}"
  _exp_check_flag RGB          "${EXP_F_RGB}"        "${RGB}"
  _exp_check_flag WITH_PC      "${EXP_F_WITH_PC}"    "${WITH_PC}"
  _exp_check_flag MC_ACTIVE    "${EXP_F_MC_ACTIVE}"  "${MC_ACTIVE}"
  _exp_check_flag MC_SIM       "${EXP_F_MC_SIM}"     "${MC_SIM}"
  _exp_check_flag FACTR_ACTIVE "${EXP_F_FACTR}"      "${FACTR_ACTIVE}"
  if [[ -n "${_exp_flag_mismatch}" ]]; then
    echo "[ERROR] The launch flags contradict EXP_CONDITION=${EXP_CONDITION}." >&2
    echo "        ${EXP_STUDY_NAME}/${EXP_CONDITION} = ${EXP_SUPERVISION} + ${EXP_CONTROLLER}, which means:" >&2
    echo "          START_VR=${EXP_F_START_VR} RGB=${EXP_F_RGB} WITH_PC=${EXP_F_WITH_PC} MC_ACTIVE=${EXP_F_MC_ACTIVE} MC_SIM=${EXP_F_MC_SIM} FACTR_ACTIVE=${EXP_F_FACTR}" >&2
    echo "        Disagreements:" >&2
    printf '%s' "${_exp_flag_mismatch}" >&2
    echo "        Recording a block whose label does not match what the participant sees is" >&2
    echo "        not recoverable afterwards, so nothing was launched. Fix the condition or" >&2
    echo "        the flags. Leaving a flag unset lets the condition supply it." >&2
    PYTHONPATH="${INTERVENE_ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" \
        -m data_io.experiment_block stop "${EXP_BLOCK_DIR}" \
        --reason flag_mismatch --status aborted >/dev/null 2>&1 || true
    exit 1
  fi
  echo "[PolicyLaunch] BLOCK: ${EXP_STUDY_NAME}/${EXP_CONDITION} = ${EXP_SUPERVISION} + ${EXP_CONTROLLER}" \
       "(START_VR=${START_VR} RGB=${RGB} WITH_PC=${WITH_PC} MC_ACTIVE=${MC_ACTIVE} MC_SIM=${MC_SIM} FACTR_ACTIVE=${FACTR_ACTIVE})"
  # Episodes land inside the block directory, not in flat INTERVENTION_DATA.
  INTERVENE_EPISODE_DIR="${INTERVENE_EPISODE_DIR:-${EXP_BLOCK_DIR}/episodes}"
  export INTERVENE_EPISODE_DIR

  # BLOCK_END on an EXIT trap, not at the bottom of the script: `on_interrupt` calls
  # `exit 130` itself, so anything after the main loop is skipped on Ctrl+C -- which is
  # precisely the exit path a participant session actually takes.
  _exp_wrapup_done=0
  exp_block_wrapup() {
    local rc="${1:-0}"
    if [[ "${_exp_wrapup_done}" == "1" ]]; then return 0; fi
    _exp_wrapup_done=1
    local status="complete"
    if [[ "${rc}" != "0" && "${rc}" != "130" ]]; then status="incomplete"; fi
    PYTHONPATH="${INTERVENE_ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" \
        -m data_io.experiment_block stop "${EXP_BLOCK_DIR}" \
        --reason "launcher_exit_${rc}" --status "${status}" \
      || echo "[PolicyLaunch][WARN] BLOCK: could not write block_summary.json" >&2
    echo "[PolicyLaunch] BLOCK: validating ${EXP_BLOCK_DIR}"
    local args=("${EXP_BLOCK_DIR}")
    if [[ -n "${STUDY_BACKUP_DIR:-}" ]]; then args+=(--backup "${STUDY_BACKUP_DIR}"); fi
    ( cd "${INTERVENE_ROOT}" && "${PYTHON_BIN}" utils/validate_experiment_block.py "${args[@]}" ) \
      || echo "[PolicyLaunch][WARN] BLOCK: validation reported problems — read ${EXP_BLOCK_DIR}/block_quality_report.md before the next participant." >&2
    echo "[PolicyLaunch] BLOCK: RECORDING STOPPED -> ${EXP_BLOCK_DIR}"
  }
  trap 'exp_block_wrapup "$?"' EXIT
fi

# --- Lab selection -------------------------------------------------------------
# LAB=tshape|cups switches the whole coherent set at once: scene XML, reset trajectory,
# ACT checkpoint, and the OOD band's body names. These four have to move together — a
# cups scene run against the tshape checkpoint produces confident nonsense — which is
# exactly why they are one switch and not four.
#
#   LAB=cups bash utils/run_main_policy.sh
#
# XML / RESET_NPZ / CHECKPOINT set explicitly still win; this only supplies defaults.
# The randomization preset and task evaluator are derived from the XML path further
# down, so they follow automatically.
#
# LAB_FAST=1 selects the *_multiwindow_fast scene variant: 640x480 offscreen
# instead of 1920x1080, timestep 0.002 with the implicitfast integrator, multiccd off,
# and no shadow maps. Measured on the cups scene: 2.0x less physics wall time and 6.8x
# fewer offscreen pixels per camera render — which is what makes N parallel windows
# viable. LAB_FAST=0 (default, used in the user studies) uses the full-fidelity scene.
# NOTE: inline truth test, not is_truthy() — that helper is defined further down.
LAB="${LAB:-tshape}"
LAB_FAST="${LAB_FAST:-0}"
_lab_scene_dir="mujoco_scenes/working_scenes/with_soft_gripper"
case "${LAB}" in
  tshape)
    _lab_scene="${_lab_scene_dir}/sii_scene_table_T_shape"
    _lab_reset="RESET_NPZs/TSHAPE/p1_ep_0001_1778506313_trimmed.npz"
    _lab_ckpt="MODEL_WEIGHTS/tshape/0800000/pretrained_model"
    _lab_ood_bodies="T1,T2"
    ;;
  cups)
    _lab_scene="${_lab_scene_dir}/sii_scene_table_boxes_cups"
    _lab_reset="RESET_NPZs/CUPS/p1_ep_0001_1778052422_trimmed.npz"
    _lab_ckpt="MODEL_WEIGHTS/cups_act_16_16_novae_d005_ff4096_20260622_174446/checkpoints/0800000/pretrained_model"
    # Bodies whose pose the OOD corpus overrides. For cups the corpus ALSO varies cup mesh
    # Z-scale and swaps the box textures, which are model-level; those are applied by
    # compiling a variant of this same scene (see the OOD_MODEL_VARIANTS preflight below).
    _lab_ood_bodies="cracker_box,sugar_box,cup1,cup2,cup3,cup4"
    ;;
  *)
    echo "[ERROR] LAB=${LAB} is not a known lab. Use: tshape | cups" >&2
    exit 1
    ;;
esac
case "${LAB_FAST}" in
  1|true|TRUE|yes|YES|on|ON) _lab_scene="${_lab_scene}_multiwindow_fast.xml" ;;
  *)                         _lab_scene="${_lab_scene}.xml" ;;
esac

XML=${XML:-${_lab_scene}}
RESET_NPZ=${RESET_NPZ:-${_lab_reset}}

DEFAULT_CHECKPOINT="${_lab_ckpt}"
CHECKPOINT=${CHECKPOINT:-${DEFAULT_CHECKPOINT}}

# Pick a randomization preset for the active scene.
#
# Usually you do not need to set this manually:
#   - T_shape XMLs use the tshape preset.
#   - boxes/cups XMLs use the cups preset.
#   - wiregame XMLs use the wiregame preset.
#
# If you want to force one explicitly:
#   INTERVENE_RANDOMIZATION_PRESET=tshape bash utils/run_main_policy.sh
#   INTERVENE_RANDOMIZATION_PRESET=cups bash utils/run_main_policy.sh
#   INTERVENE_RANDOMIZATION_PRESET=wiregame bash utils/run_main_policy.sh
INTERVENE_RANDOMIZATION_PRESET="${INTERVENE_RANDOMIZATION_PRESET:-}"
if [[ -z "${INTERVENE_RANDOMIZATION_PRESET}" ]]; then
  case "${XML}" in
    *T_shape*) INTERVENE_RANDOMIZATION_PRESET="tshape" ;;
    *boxes_cups*|*cups*) INTERVENE_RANDOMIZATION_PRESET="cups" ;;
    *wire_base*|*wire*) INTERVENE_RANDOMIZATION_PRESET="wiregame" ;;
    *) INTERVENE_RANDOMIZATION_PRESET="generic" ;;
  esac
fi
export INTERVENE_RANDOMIZATION_PRESET

# Common randomization knobs shared by every preset.
#
# INTERVENE_OBJECT_XY_RANDOM_RANGE is the +/- xy jitter in meters.
# INTERVENE_OBJECT_EULER_RANDOM_RANGE_DEG is roll pitch yaw jitter in degrees.
# The x/y bounds keep randomized objects on the useful table area.
export INTERVENE_RANDOMIZE_SCENE="${INTERVENE_RANDOMIZE_SCENE:-1}"
# Randomize EPISODE 1 too. Without this every cell of a WINDOWS>1 grid boots to the
# identical reset-NPZ pose and only diverges at its first episode boundary. Set 0 for
# the pristine reset pose; study mode randomizes episode 1 regardless.
export INTERVENE_RANDOMIZE_FIRST_EPISODE="${INTERVENE_RANDOMIZE_FIRST_EPISODE:-1}"
export INTERVENE_OBJECT_XY_RANDOM_RANGE="${INTERVENE_OBJECT_XY_RANDOM_RANGE:-0.01}"
export INTERVENE_OBJECT_EULER_RANDOM_RANGE_DEG="${INTERVENE_OBJECT_EULER_RANDOM_RANGE_DEG:-0 0 5}"
export INTERVENE_OBJECT_X_BOUNDS="${INTERVENE_OBJECT_X_BOUNDS:-0.40 0.80}"
export INTERVENE_OBJECT_Y_BOUNDS="${INTERVENE_OBJECT_Y_BOUNDS:--0.25 0.25}"

case "${INTERVENE_RANDOMIZATION_PRESET}" in
  tshape)
    # T-shape task: randomize both T objects and evaluate T1 placed on T2.
    export INTERVENE_RANDOMIZE_OBJECT_NAMES="${INTERVENE_RANDOMIZE_OBJECT_NAMES:-T1,T2}"
    export INTERVENE_RANDOMIZE_EULER_NAMES="${INTERVENE_RANDOMIZE_EULER_NAMES:-T1,T2}"
    export INTERVENE_TASK_MODE="${INTERVENE_TASK_MODE:-tshape}"
    export INTERVENE_TASK_BODY_NAME="${INTERVENE_TASK_BODY_NAME:-T1}"
    export INTERVENE_TASK_TARGET_BODY_NAME="${INTERVENE_TASK_TARGET_BODY_NAME:-T2}"
    export INTERVENE_AUTO_TASK_EVAL="${INTERVENE_AUTO_TASK_EVAL:-1}"
    ;;
  cups)
    # Cups task: randomize cups + boxes. Only boxes get yaw randomization by
    # default, because cup orientation can make the task unnecessarily chaotic.
    export INTERVENE_RANDOMIZE_OBJECT_NAMES="${INTERVENE_RANDOMIZE_OBJECT_NAMES:-cracker_box,sugar_box,cup1,cup2,cup3,cup4}"
    export INTERVENE_RANDOMIZE_EULER_NAMES="${INTERVENE_RANDOMIZE_EULER_NAMES:-cracker_box,sugar_box}"
    export INTERVENE_TASK_MODE="${INTERVENE_TASK_MODE:-cups}"
    export INTERVENE_TASK_CUP_BODY_NAMES="${INTERVENE_TASK_CUP_BODY_NAMES:-cup1,cup2,cup3,cup4}"
    export INTERVENE_TASK_BOX_BODY_NAMES="${INTERVENE_TASK_BOX_BODY_NAMES:-cracker_box,sugar_box}"
    export INTERVENE_TASK_CUP_PAIRS="${INTERVENE_TASK_CUP_PAIRS:-cup1:cup2,cup3:cup4}"
    export INTERVENE_AUTO_TASK_EVAL="${INTERVENE_AUTO_TASK_EVAL:-1}"
    # Nesting depth, NOT the generic stacking offset. Without this, cups inherited the
    # global default 0.165 m -- an offset meant for placing one object ON TOP of another --
    # so the check demanded cup1's bottom sit 165 mm above cup2's bottom while ALSO
    # requiring the two to be in contact. For a 97 mm cup those are mutually exclusive:
    # contact ends at ~80 mm. Measured, the pair could never be "placed" at ANY height, so
    # a cups episode could not succeed and every one ended at the 217-step limit
    # (session logs: 330x "cup1_on_cup2_bad_z:-0.158m", 18 recorded episodes, 0 successes).
    # 0.0035 is the value utils/evaluate_act_cups_shift1.sh has always used offline.
    export INTERVENE_TASK_PLACEMENT_Z_OFFSET="${INTERVENE_TASK_PLACEMENT_Z_OFFSET:-0.0035}"
    ;;
  wiregame)
    # Wiregame randomization is enabled, but automatic success/failure is off
    # by default because the current evaluator only knows T-shape and cups.
    # Note: the wire base body is named "object" in XML, but its free joint is
    # "wire_base_free"; the randomizer uses the free-joint name prefix.
    export INTERVENE_RANDOMIZE_OBJECT_NAMES="${INTERVENE_RANDOMIZE_OBJECT_NAMES:-spoon1,wire_base}"
    export INTERVENE_RANDOMIZE_EULER_NAMES="${INTERVENE_RANDOMIZE_EULER_NAMES:-spoon1,wire_base}"
    export INTERVENE_TASK_MODE="${INTERVENE_TASK_MODE:-wiregame}"
    export INTERVENE_AUTO_TASK_EVAL="${INTERVENE_AUTO_TASK_EVAL:-0}"
    ;;
  *)
    # Generic fallback: list all known movable objects. Missing objects are
    # ignored safely by the Python randomizer.
    export INTERVENE_RANDOMIZE_OBJECT_NAMES="${INTERVENE_RANDOMIZE_OBJECT_NAMES:-T1,T2,cracker_box,sugar_box,cup1,cup2,cup3,cup4,spoon1,wire_base}"
    export INTERVENE_RANDOMIZE_EULER_NAMES="${INTERVENE_RANDOMIZE_EULER_NAMES:-T1,T2,cracker_box,sugar_box,spoon1,wire_base}"
    export INTERVENE_TASK_MODE="${INTERVENE_TASK_MODE:-single}"
    export INTERVENE_AUTO_TASK_EVAL="${INTERVENE_AUTO_TASK_EVAL:-1}"
    ;;
esac

# Automatic task success/failure behavior.
#
# Success starts a new randomized scene.
# Failure restarts the current scene.
# Manual keys are still available in the viewer:
#   S = force success, F = force failure, R = restart current scene.
export INTERVENE_AUTO_FAIL_AT_RESET_LEN="${INTERVENE_AUTO_FAIL_AT_RESET_LEN:-1}"

# --- Episode camera frames: state-only recording (state-only recording decision) --------
# Capturing 3 camera frames on every policy frame, on every session, was 62% of ALL GPU
# render work in the system (270 of ~437 render+readback ops/sec at 9 windows) and 99% of
# episode file size — for images nothing displays. They are reconstructed offline from
# qpos_sim/qvel_sim/ctrl_sim instead, measured at mean 0.673/255 against live captures
# (bit-identical on pre-2026-07-13 data); see utils/verify_rerender_fidelity.py.
#
# RECORD_IMAGES=1 restores live capture — use it when extending the golden corpus
# (utils/archive_golden_corpus.py), which is the permanent fixture that fidelity is
# verified against. This is IRREVERSIBLE per episode: a state-only episode can never be
# turned back into one containing live frames.
# NOTE: inline test, not is_truthy() — that helper is defined further down this file and
# would not exist yet at this point in execution.
RECORD_IMAGES="${RECORD_IMAGES:-0}"
if [[ "${RECORD_IMAGES}" == "1" || "${RECORD_IMAGES}" == "true" || "${RECORD_IMAGES}" == "TRUE" || "${RECORD_IMAGES}" == "yes" ]]; then
  export INTERVENE_RECORD_RGB=1
  echo "[PolicyLaunch] Episode camera frames: LIVE CAPTURE (RECORD_IMAGES=1) — ~30 renders/s per session"
else
  export INTERVENE_RECORD_RGB=0
  echo "[PolicyLaunch] Episode camera frames: state-only; re-render offline with"
  echo "[PolicyLaunch]   utils/convert_npz_to_lerobot.py --rerender_images   (RECORD_IMAGES=1 to capture live)"
fi
export INTERVENE_TASK_PLACEMENT_STABLE_STEPS="${INTERVENE_TASK_PLACEMENT_STABLE_STEPS:-10}"
export INTERVENE_TASK_TIMEOUT_SECONDS="${INTERVENE_TASK_TIMEOUT_SECONDS:-0}"

START_VR=${START_VR:-1}
# Diagnostic A/B: keep all policy processes (and the optional desktop grid) running, but
# let the Linux VR runtimes advance their local replay instead of consuming policy snapshots.
# This isolates policy-state mirroring without changing GPU/process pressure.
VR_POLICY_MIRROR=${VR_POLICY_MIRROR:-1}
# Record which control flags the OPERATOR set explicitly, before any defaulting. A study
# block's "control" field sets these below, but an explicit value on the command line must
# still win — this is the only way to tell "unset" from "deliberately 0".
_MC_ACTIVE_EXPLICIT="${MC_ACTIVE+set}"
_MC_SIM_EXPLICIT="${MC_SIM+set}"
_FACTR_ACTIVE_EXPLICIT="${FACTR_ACTIVE+set}"

MC_MODE="${MC_MODE:-0}"   # 1 = motion controller only; suppresses VR runner (Quest uses SIIScene_MotionControllerOnly)
MC_ACTIVE="${MC_ACTIVE:-0}"     # 1 = launch mq3_mc.py sidecar (Quest right controller → real robot)
                                # Robot IP/ports resolved from INTERVENE_ROBOT_KEY via record.ROBOTS
MC_SIM="${MC_SIM:-0}"          # 1 = sim-native MC: Quest right controller drives the SIMULATED arm via
                                # local IK (robot/sim_mc_driver.py) — no real robot, no mq3_mc.py sidecar.
                                # Only meaningful when MC_ACTIVE=1.
# NOTE: INTERVENE_MC_ACTIVE / INTERVENE_MC_SIM / INTERVENE_FACTR_ACTIVE are exported LATER,
# after the study block's "control" field has been resolved. Exporting here would freeze the
# pre-study values and a block requesting sim_mc or factr would silently launch as
# telekinesis. See "Control mode -> intervention flags" below.
MC_MAX_RESTARTS="${MC_MAX_RESTARTS:-5}"   # bounded auto-restart of the mq3_mc.py sidecar

# --- FACTR leader arm ---------------------------------------------------------
# FACTR is a Dynamixel force-feedback leader arm. During an intervention the human moves
# FACTR and the SIMULATED robot follows it; no real Franka is involved. A single service
# process owns the serial device and arbitrates it across every policy window.
FACTR_ACTIVE="${FACTR_ACTIVE:-0}"
FACTR_CONFIG="${FACTR_CONFIG:-factr/leader.yaml}"
FACTR_REST_POSE="${FACTR_REST_POSE:-factr/validation/rest_pose.json}"
FACTR_INIT_POSE="${FACTR_INIT_POSE:-factr/validation/init_pose.json}"
FACTR_REST_POSITION_TOLERANCE="${FACTR_REST_POSITION_TOLERANCE:-1.5}"
# 0.08, not the fork launcher's 0.04. Measured on this arm 2026-08-03: startup staging
# converges to ~0.057 rad WITH gravity+friction compensation, so 0.04 can never be met and
# the service never reaches READY. 0.08 is also the value the FACTR fork's own README uses
# for policy intervention, and the one this arm was validated with.
FACTR_POSITION_TOLERANCE="${FACTR_POSITION_TOLERANCE:-0.08}"
FACTR_INIT_TIMEOUT="${FACTR_INIT_TIMEOUT:-18.0}"
FACTR_ALIGNMENT_TIMEOUT="${FACTR_ALIGNMENT_TIMEOUT:-18.0}"
FACTR_MAX_VELOCITY="${FACTR_MAX_VELOCITY:-0.5}"
FACTR_SERVICE_HOST="${FACTR_SERVICE_HOST:-127.0.0.1}"
FACTR_SERVICE_PORT="${FACTR_SERVICE_PORT:-18075}"
FACTR_STARTUP_TIMEOUT="${FACTR_STARTUP_TIMEOUT:-60}"
FACTR_SHUTDOWN_TIMEOUT="${FACTR_SHUTDOWN_TIMEOUT:-45}"
# Policy-client socket timeout. MUST exceed human reaction time: begin_takeover blocks in
# the service until the operator physically pushes the arm (FACTR_TOUCH_RELEASE_TIMEOUT=0
# = wait forever), and the client's default is only 5 s — which would abort every takeover
# with mode_switch_failed 5 s after the intervention key is pressed.
FACTR_RPC_TIMEOUT="${FACTR_RPC_TIMEOUT:-180.0}"
# Torque/hold profile. These reach the SERVICE only — never a policy process.
FACTR_MAX_TORQUE="${FACTR_MAX_TORQUE:-10}"
FACTR_ALLOW_OVER_CONFIG_TORQUE="${FACTR_ALLOW_OVER_CONFIG_TORQUE:-1}"
FACTR_HOLD_ERROR="${FACTR_HOLD_ERROR:-0.04}"
FACTR_DRIVE_TORQUE="${FACTR_DRIVE_TORQUE:-2.2,2.2,2.2,2.2,1.2,1.0,0.7}"
FACTR_DRIVE_DEADBAND="${FACTR_DRIVE_DEADBAND:-0.05}"
FACTR_DRIVE_RAMP="${FACTR_DRIVE_RAMP:-0.50}"
FACTR_TRAJECTORY_DURATION="${FACTR_TRAJECTORY_DURATION:-3}"
# DEVIATION from the reference config (which uses 0/0), kept deliberately: measured
# 2026-08-03 on this arm, staging converges to 0.0568 rad with comp ON vs 0.0629 with it
# OFF. The FACTR README also prescribes 1 ("set 0 only for hardware diagnostics") even
# though its launcher says 0. Do not "restore" these to 0 without re-measuring.
FACTR_HOLD_GRAVITY_COMP="${FACTR_HOLD_GRAVITY_COMP:-1}"
FACTR_HOLD_FRICTION_COMP="${FACTR_HOLD_FRICTION_COMP:-1}"
FACTR_WAIT_MAX_TORQUE="${FACTR_WAIT_MAX_TORQUE:-${FACTR_MAX_TORQUE}}"
FACTR_WAIT_HOLD_ERROR="${FACTR_WAIT_HOLD_ERROR:-${FACTR_HOLD_ERROR}}"
FACTR_WAIT_DRIVE_TORQUE="${FACTR_WAIT_DRIVE_TORQUE:-${FACTR_DRIVE_TORQUE}}"
FACTR_WAIT_HOLD_GRAVITY_COMP="${FACTR_WAIT_HOLD_GRAVITY_COMP:-${FACTR_HOLD_GRAVITY_COMP}}"
FACTR_WAIT_HOLD_FRICTION_COMP="${FACTR_WAIT_HOLD_FRICTION_COMP:-${FACTR_HOLD_FRICTION_COMP}}"
# --- The takeover handshake -----------------------------------------------------
# align -> PD-hold the aligned pose -> WAIT for the human to physically push the arm
# (>= DELTA_RAD, no timeout) -> blend torque over BLEND_SECONDS with a decaying
# HOLD_ASSIST -> release into gravity-comp leader mode.
# GRAVITY_SCALE is deliberately < 1: the arm settles into your hand instead of
# free-floating. These are the reference values; a previous transcription error had them
# at 1.0 / 0.0 / 0.0 / disabled, which dropped the arm's weight the instant you pressed
# the key. Do not "simplify" them.
FACTR_TAKEOVER_GRAVITY_SCALE="${FACTR_TAKEOVER_GRAVITY_SCALE:-0.75}"
FACTR_TAKEOVER_GRAVITY_RAMP_DURATION="${FACTR_TAKEOVER_GRAVITY_RAMP_DURATION:-0.0}"
FACTR_TAKEOVER_FRICTION_SCALE="${FACTR_TAKEOVER_FRICTION_SCALE:-1.0}"
FACTR_TAKEOVER_NULLSPACE_SCALE="${FACTR_TAKEOVER_NULLSPACE_SCALE:-0.1}"
FACTR_TAKEOVER_SETTLE_SECONDS="${FACTR_TAKEOVER_SETTLE_SECONDS:-0.4}"
FACTR_TAKEOVER_TORQUE_BLEND_SECONDS="${FACTR_TAKEOVER_TORQUE_BLEND_SECONDS:-0.6}"
FACTR_TAKEOVER_HOLD_ASSIST_SECONDS="${FACTR_TAKEOVER_HOLD_ASSIST_SECONDS:-1.0}"
FACTR_TOUCH_RELEASE_ENABLED="${FACTR_TOUCH_RELEASE_ENABLED:-1}"
FACTR_TOUCH_RELEASE_DELTA_RAD="${FACTR_TOUCH_RELEASE_DELTA_RAD:-0.015}"
# 0 = wait indefinitely for the human. A non-zero timeout releases the arm on a timer
# whether or not anyone is holding it, which defeats the point of the wait.
FACTR_TOUCH_RELEASE_TIMEOUT="${FACTR_TOUCH_RELEASE_TIMEOUT:-0.0}"
FACTR_RETURN_INIT_ON_RELEASE="${FACTR_RETURN_INIT_ON_RELEASE:-1}"
FACTR_RETURN_REST_ON_CLOSE="${FACTR_RETURN_REST_ON_CLOSE:-1}"
FACTR_REST_TIMEOUT="${FACTR_REST_TIMEOUT:-25.0}"
FACTR_REST_GOAL_TOLERANCE="${FACTR_REST_GOAL_TOLERANCE:-0.12}"
FACTR_REST_DISABLE_LIMIT_TORQUE="${FACTR_REST_DISABLE_LIMIT_TORQUE:-1}"
FACTR_PRESERVE_ALIGNED_ANCHOR="${FACTR_PRESERVE_ALIGNED_ANCHOR:-1}"
# How the SIM follows the leader. These reach the policy processes.
FACTR_MUJOCO_ALPHA="${FACTR_MUJOCO_ALPHA:-1.0}"
FACTR_MUJOCO_MAX_STEP_RAD="${FACTR_MUJOCO_MAX_STEP_RAD:-0.04}"
# 0.0: with the touch-release handshake restored, the sim anchor is re-taken at the
# moment of release, so there is no discontinuity left for a ramp to hide.
FACTR_MUJOCO_TAKEOVER_RAMP_SECONDS="${FACTR_MUJOCO_TAKEOVER_RAMP_SECONDS:-0.0}"
FACTR_MUJOCO_TAKEOVER_ALPHA_START="${FACTR_MUJOCO_TAKEOVER_ALPHA_START:-0.05}"
FACTR_MUJOCO_TAKEOVER_MAX_STEP_RAD="${FACTR_MUJOCO_TAKEOVER_MAX_STEP_RAD:-0.002}"
FACTR_MUJOCO_GRIPPER_MAX_STEP="${FACTR_MUJOCO_GRIPPER_MAX_STEP:-0.008}"
# continuous = sim ctrl[7] follows the leader gripper's measured width every tick, so
# the operator gets partial grasps. Set to "standalone" (or threshold/pressed/binary)
# to restore the old open/closed trigger behaviour used by pre-2026-08-24 recordings.
FACTR_MUJOCO_GRIPPER_MODE="${FACTR_MUJOCO_GRIPPER_MODE:-continuous}"

# The FACTR service needs pinocchio + dynamixel_sdk, which may live in a different
# environment from the policy interpreter. An explicit override always wins; otherwise
# the candidates are probed in the FACTR preflight below and the first one that can
# actually import the hardware stack is used. Guessing from CONDA_PREFIX alone is not
# enough: with the conda BASE env active it resolves to a python without pinocchio, and
# the failure only shows up after the service has been spawned.
FACTR_PYTHON_BIN_EXPLICIT="${FACTR_PYTHON_BIN+set}"
FACTR_PYTHON_BIN="${FACTR_PYTHON_BIN:-}"

# --- One robot key for BOTH halves --------------------------------------------
# app.py reads INTERVENE_ROBOT_KEY (default p4 -> 192.0.2.153) while the VR runner
# run_multi_window_robot.sh reads ROBOT_KEY (default p1 -> 192.168.0.150). This script
# used to export neither, so the policy process and the VR runtime targeted two
# DIFFERENT machines by default. Resolve one key here and export both names, so setting
# either variable moves the whole system together.
ROBOT_KEY="${INTERVENE_ROBOT_KEY:-${ROBOT_KEY:-p3}}"
export ROBOT_KEY
export INTERVENE_ROBOT_KEY="${ROBOT_KEY}"
WINDOWS="${WINDOWS:-}"   # empty = SINGLE mode (default, backward-compatible); set to 3/6/9/12/15 for multi-window grid
if [[ -z "${START_GRID_UI:-}" ]]; then
  if [[ -n "${WINDOWS:-}" ]]; then
    START_GRID_UI=1
  else
    START_GRID_UI=0
  fi
fi
BASE_TOPIC_PORT="${BASE_TOPIC_PORT:-}"  # override first ZMQ topic port (default 7741 in launcher)
POLICY_CMD_BIND=${POLICY_CMD_BIND:-127.0.0.1}
POLICY_CMD_PORT=${POLICY_CMD_PORT:-8065}
POLICY_STATE_BIND=${POLICY_STATE_BIND:-127.0.0.1}
POLICY_STATE_HOST=${POLICY_STATE_HOST:-127.0.0.1}
POLICY_STATE_PORT=${POLICY_STATE_PORT:-8066}
POLICY_STATE_HZ=${POLICY_STATE_HZ:-30}
POLICY_STATE_TIMEOUT_S=${POLICY_STATE_TIMEOUT_S:-1.0}
VR_RUNNER=${VR_RUNNER:-"${UMBRELLA_ROOT}/run_multi_window_robot.sh"}

# --- User study (PART-2) -----------------------------------------------------
# Set STUDY_PARTICIPANT to enable study logging. Everything else is read from
# study/participants/<ID>.json, so the operator only supplies who and which block:
#
#   STUDY_PARTICIPANT=P07 STUDY_BLOCK=2 bash utils/run_main_policy.sh
#
# Leaving STUDY_PARTICIPANT unset keeps the previous behaviour exactly: flat
# INTERVENTION_DATA output, no session directory, failures discarded as before.
STUDY_PARTICIPANT="${STUDY_PARTICIPANT:-}"
STUDY_BLOCK="${STUDY_BLOCK:-1}"
STUDY_CONFIG_DIR="${STUDY_CONFIG_DIR:-${INTERVENE_ROOT}/study/participants}"
STUDY_BACKUP_DIR="${STUDY_BACKUP_DIR:-}"
STUDY_ACC_METHOD="${STUDY_ACC_METHOD:-chunk_residual}"
# These controls matter for ensemble ACC, but exporting them here makes every launch
# reproducible and prevents a later feature-isolation run from inheriting shell-local
# values. chunk_residual does not create probe renders.
STUDY_ACC_PROBE_ASYNC="${STUDY_ACC_PROBE_ASYNC:-1}"
STUDY_ACC_PROBE_MIDCHUNK="${STUDY_ACC_PROBE_MIDCHUNK:-0}"
export STUDY_PARTICIPANT STUDY_BLOCK STUDY_CONFIG_DIR STUDY_ACC_METHOD \
  STUDY_ACC_PROBE_ASYNC STUDY_ACC_PROBE_MIDCHUNK

if [[ -n "${STUDY_PARTICIPANT}" ]]; then
  # Resolve the block up front so a bad config fails BEFORE the participant is in the
  # headset, and so the interface (rgb|pointcloud) can drive the VR launch mode.
  STUDY_INFO="$(
    cd "${INTERVENE_ROOT}" && "${PYTHON_BIN}" - <<'PYEOF'
import os, sys
sys.path.insert(0, os.getcwd())
try:
    from data_io.study_config import load_block
    b = load_block()
    if b is None:
        raise SystemExit("STUDY_PARTICIPANT resolved to nothing")
    print(f"{b.interface}\t{b.task}\t{b.condition}\t{b.seed}\t{b.label}\t{b.control}")
except Exception as exc:
    print(f"ERROR\t{exc}", file=sys.stderr)
    raise SystemExit(1)
PYEOF
  )" || {
    echo "[ERROR] Study config for participant '${STUDY_PARTICIPANT}' block ${STUDY_BLOCK} is invalid." >&2
    echo "[HINT]  Validate it with: (cd ${INTERVENE_ROOT} && python -m data_io.study_config ${STUDY_PARTICIPANT} --block ${STUDY_BLOCK})" >&2
    exit 1
  }
  STUDY_INTERFACE="$(printf '%s' "${STUDY_INFO}" | cut -f1)"
  STUDY_TASK="$(printf '%s' "${STUDY_INFO}" | cut -f2)"
  STUDY_CONDITION="$(printf '%s' "${STUDY_INFO}" | cut -f3)"
  STUDY_SEED="$(printf '%s' "${STUDY_INFO}" | cut -f4)"
  STUDY_LABEL="$(printf '%s' "${STUDY_INFO}" | cut -f5)"
  STUDY_CONTROL="$(printf '%s' "${STUDY_INFO}" | cut -f6)"

  # The block's task selects the evaluator; its condition selects the randomization
  # preset. Both stay overridable so a pilot run can deviate deliberately.
  export INTERVENE_TASK_MODE="${INTERVENE_TASK_MODE:-${STUDY_TASK}}"
  export INTERVENE_RANDOMIZATION_PRESET="${INTERVENE_RANDOMIZATION_PRESET:-${STUDY_TASK}}"

  # The block's interface picks the VR mode: RGB panels vs point clouds.
  case "${STUDY_INTERFACE}" in
    rgb)        STUDY_VR_RGB=1; STUDY_VR_PC=0 ;;
    pointcloud) STUDY_VR_RGB=0; STUDY_VR_PC=1 ;;
    both)       STUDY_VR_RGB=0; STUDY_VR_PC=1 ;;   # PC mode also publishes front/rgb
    none)       STUDY_VR_RGB=0; STUDY_VR_PC=0 ;;
    *)          STUDY_VR_RGB=0; STUDY_VR_PC=1 ;;
  esac

  # The block's control picks the intervention device, the same way interface picks the
  # VR mode — so the recorded condition and the hardware in the participant's hands
  # cannot drift apart. An explicit MC_ACTIVE/MC_SIM/FACTR_ACTIVE on the command line
  # still wins, for pilots and debugging.
  # Full if-blocks, not `[[ ... ]] && X=1`: under `set -e` a failing test as the last
  # command of the line would abort the launch.
  case "${STUDY_CONTROL}" in
    factr)
      if [[ -z "${_FACTR_ACTIVE_EXPLICIT}" ]]; then FACTR_ACTIVE=1; fi
      if [[ -z "${_MC_ACTIVE_EXPLICIT}" ]];    then MC_ACTIVE=0;    fi
      if [[ -z "${_MC_SIM_EXPLICIT}" ]];       then MC_SIM=0;       fi
      ;;
    motion_controller)
      if [[ -z "${_MC_ACTIVE_EXPLICIT}" ]];    then MC_ACTIVE=1;    fi
      if [[ -z "${_MC_SIM_EXPLICIT}" ]];       then MC_SIM=0;       fi
      if [[ -z "${_FACTR_ACTIVE_EXPLICIT}" ]]; then FACTR_ACTIVE=0; fi
      ;;
    sim_mc)
      if [[ -z "${_MC_ACTIVE_EXPLICIT}" ]];    then MC_ACTIVE=1;    fi
      if [[ -z "${_MC_SIM_EXPLICIT}" ]];       then MC_SIM=1;       fi
      if [[ -z "${_FACTR_ACTIVE_EXPLICIT}" ]]; then FACTR_ACTIVE=0; fi
      ;;
    telekinesis|*)
      : # leave whatever the operator/defaults already set
      ;;
  esac
fi

# --- Control mode -> intervention flags ---------------------------------------
# Exported HERE, after the study block has resolved, so a block requesting sim_mc or
# factr actually reaches app.py. Exporting these at declaration time would freeze the
# pre-study values and silently launch the wrong mode with no error.
export INTERVENE_MC_ACTIVE="${MC_ACTIVE}"
export INTERVENE_MC_SIM="${MC_SIM}"
export INTERVENE_FACTR_ACTIVE="${FACTR_ACTIVE}"

is_truthy() {
  case "${1:-}" in
    1|true|TRUE|yes|YES|on|ON) return 0 ;;
    *) return 1 ;;
  esac
}

# --- Retired OOD auto-pause knobs ---------------------------------------------
# The OOD auto-pause supervisor (rank sessions by acc_risk, force-pause all but the top N)
# has been removed. OOD now means "this session runs an out-of-distribution SCENE" and
# never pauses anything, so there is no supervisor and nothing to own. Saved command lines
# still carrying the old variables would otherwise do nothing at all, silently.
for _retired in OOD_ENABLED OOD_OWNER OOD_MANUAL_OVERRIDE_S \
                OOD_PITCH_BANDS_DEG OOD_BASE_EULER_RAD \
                GRID_UI_ACC_OOD_THRESHOLD GRID_UI_MAX_ACTIVE_OOD_CELLS \
                GRID_UI_OOD_EVALUATE_INTERVAL_S; do
  if [[ -n "${!_retired:-}" ]]; then
    echo "[PolicyLaunch][RETIRED] ${_retired} no longer does anything (OOD auto-pause was" \
         "removed). OOD is now scene-based: see OOD_MIN_STATES / OOD_MAX_STATES." >&2
  fi
done
unset _retired

# --- OOD scenes ---------------------------------------------------------------
# Between MIN and MAX of the grid's sessions run an out-of-distribution scene at any
# time. Each session re-decides only at its OWN episode boundary, so the OOD cells
# migrate around the grid without any synchronized reset (a grid-wide reshuffle would
# disturb the point-cloud/VR pipelines and break session spatial stability).
# MIN=0 disables the feature entirely — no ledger, no file I/O.
OOD_MIN_STATES="${OOD_MIN_STATES:-2}"
OOD_MAX_STATES="${OOD_MAX_STATES:-3}"
OOD_MAX_SCENE_ATTEMPTS="${OOD_MAX_SCENE_ATTEMPTS:-3}"
export OOD_MAX_SCENE_ATTEMPTS
OOD_LEDGER_TTL_S="${OOD_LEDGER_TTL_S:-1800}"
# Keyed on the launch cohort, so two concurrent runs keep separate bands.
OOD_LEDGER_KEY="${OOD_LEDGER_KEY:-${POLICY_CMD_PORT}}"
# What makes a scene OOD: an object rotation drawn from bands far from nominal, on the
# SAME XML (the desktop grid loads one shared MjModel for every session). The remaining
# knobs are scene hygiene — re-seat transformed geometry on the table and zero free-joint
# velocity — and apply to in-distribution scenes identically.
# Bodies and bands come from the LAB block above, so switching LAB switches these too.
# Hardcoding T1 here would silently no-op on the cups lab: the parser would look for a
# body that does not exist and every "OOD" episode would be in-distribution.
# Where the pre-generated, contact-validated OOD scenes live. An OOD episode takes its
# object poses verbatim from one of these; nothing is sampled at runtime. The XML itself is
# never loaded (see data_io/ood_scene_poses.py for why), only its poses are read.
OOD_SCENE_DIR="${OOD_SCENE_DIR:-mujoco_scenes/ood_scenes}"
# Model-level OOD (currently the cups task). 1 = compile a model variant per OOD episode.
OOD_MODEL_VARIANTS="${OOD_MODEL_VARIANTS:-1}"
# Grid-viewer slot budget, INCLUDING the pinned base slot. Measured ~404 MB RAM and
# ~137 MB VRAM per extra compiled model.
#
# Scaled down for large cohorts because the GPU, not the CPU, is the binding constraint
# there: each session costs ~298 MB of Open3D CUDA context plus ~326 MB of torch CUDA
# context in its policy process, and NEITHER can be freed in-process (release_cache
# reclaims working blocks, not the context). At 9 windows that is ~5.6 GB gone before any
# point cloud is computed, and the grid's render slots compete for what is left. Tiles that
# cannot get a slot fall back to the base model with an "(base render)" badge -- the
# per-tile degradation path -- so nothing breaks, the picture is just approximate.
if (( ${WINDOWS:-0} >= 9 )); then
  OOD_VARIANT_CACHE="${OOD_VARIANT_CACHE:-2}"
else
  OOD_VARIANT_CACHE="${OOD_VARIANT_CACHE:-$((OOD_MAX_STATES + 1))}"
fi
# Compile the NEXT episode's variant in the background while the current one runs. Off =>
# a ~1.3 s pause at each OOD episode boundary.
OOD_VARIANT_PRECOMPILE="${OOD_VARIANT_PRECOMPILE:-1}"
# How many corpus variants the preflight actually compiles (0 = all 50, ~65 s).
OOD_VARIANT_PREFLIGHT_SCENES="${OOD_VARIANT_PREFLIGHT_SCENES:-3}"
# Desktop grid: exact | base | off. `base` renders OOD tiles with the base model plus a
# badge (zero extra memory); `exact` compiles per-variant render slots.
GRID_UI_VARIANT_RENDER="${GRID_UI_VARIANT_RENDER:-exact}"
# Which corpus subdirectory. Keyed off the task, not LAB: LAB has no wiregame case.
OOD_TASK="${OOD_TASK:-${INTERVENE_TASK_MODE:-tshape}}"
OOD_POSITION_FROM_XML_NAMES="${OOD_POSITION_FROM_XML_NAMES:-${_lab_ood_bodies}}"
OOD_SUPPORT_Z_NAMES="${OOD_SUPPORT_Z_NAMES:-${_lab_ood_bodies}}"
# Both labs stand their objects on the same table (z=0.225 authored).
OOD_SUPPORT_Z="${OOD_SUPPORT_Z:-0.22}"
OOD_SUPPORT_MARGIN="${OOD_SUPPORT_MARGIN:-0.0}"
OOD_ZERO_QVEL_NAMES="${OOD_ZERO_QVEL_NAMES:-${_lab_ood_bodies}}"

# --- OOD corpus preflight -----------------------------------------------------
if [[ "${OOD_MIN_STATES}" -gt 0 ]]; then
  # For a task whose corpus encodes OOD in the MODEL (cup mesh Z-scale + swapped box
  # textures) rather than in object poses, a pose-only draw would be INSIDE the ordinary
  # in-distribution jitter -- i.e. episodes labelled OOD that are not. This used to be a
  # blanket refusal; it is now a real check that those variants build and are drop-in
  # compatible with the live scene. OOD_MODEL_VARIANTS=0 opts out, and stamps
  # episode_ood_pose_only=1 on every episode so the mislabelling stays detectable.
  if [[ "${OOD_TASK}" == "cups" ]] && is_truthy "${OOD_MODEL_VARIANTS}"; then
    # ${XML}, not ${XML_REAL}: this preflight runs long before XML_REAL is resolved, and
    # under `set -u` referencing it here aborts the launch. preflight.py resolves the path
    # itself and reports a missing scene with its own error.
    if ! "${PYTHON_BIN}" -m model_variants.preflight \
          --xml "${XML}" --corpus "${OOD_SCENE_DIR}" --task "${OOD_TASK}" \
          --max-scenes "${OOD_VARIANT_PREFLIGHT_SCENES}"; then
      echo "[ERROR] OOD model-variant preflight failed for task '${OOD_TASK}'." >&2
      echo "[HINT]  Pose-only (data WILL be mislabelled): OOD_MODEL_VARIANTS=0" >&2
      echo "[HINT]  Disable OOD entirely:                 OOD_MIN_STATES=0" >&2
      exit 1
    fi
  elif [[ "${OOD_TASK}" == "cups" ]]; then
    echo "[PolicyLaunch][WARN] OOD_MODEL_VARIANTS=0: cups OOD episodes will differ from" >&2
    echo "[PolicyLaunch][WARN] in-distribution ones by POSE ONLY, which for this task is" >&2
    echo "[PolicyLaunch][WARN] inside the ordinary jitter. Episodes are stamped" >&2
    echo "[PolicyLaunch][WARN] episode_ood_pose_only=1 so this stays detectable." >&2
  fi
  _ood_corpus_dir="${OOD_SCENE_DIR}/$(
    case "${OOD_TASK}" in
      tshape)   echo t_shape ;;
      cups)     echo boxes_cups ;;
      wiregame) echo wire_spoon ;;
      *)        echo "" ;;
    esac)"
  _ood_scene_count=0
  if [[ -d "${_ood_corpus_dir}" ]]; then
    _ood_scene_count=$(find "${_ood_corpus_dir}" -maxdepth 1 -name '*.xml' | wc -l)
  fi
  if [[ "${_ood_scene_count}" -eq 0 ]]; then
    echo "[ERROR] OOD is enabled (OOD_MIN_STATES=${OOD_MIN_STATES}) but no scenes were found in" >&2
    echo "[ERROR]   ${_ood_corpus_dir}" >&2
    echo "[HINT]  Set OOD_SCENE_DIR, or disable OOD with OOD_MIN_STATES=0." >&2
    exit 1
  fi
  echo "[PolicyLaunch] OOD corpus: ${OOD_TASK} -> ${_ood_corpus_dir} (${_ood_scene_count} scenes)"
fi

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

if [[ ! -f "${RESET_NPZ}" ]]; then
  echo "[ERROR] Missing reset NPZ: ${RESET_NPZ}" >&2
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

# --- FACTR preflight ----------------------------------------------------------
# Fail before anything starts: a study run must never be half-launched, and the FACTR
# service takes real torque-enabled control of the leader arm.
if is_truthy "${FACTR_ACTIVE}"; then
  if is_truthy "${MC_ACTIVE}"; then
    echo "[ERROR] FACTR_ACTIVE=1 and MC_ACTIVE=1 cannot be used together." >&2
    echo "[HINT] They are two different intervention devices. Pick one." >&2
    exit 1
  fi
  # app.py checks sim_mc_active independently of mc_active, so MC_SIM=1 alongside FACTR
  # would route the takeover through the sim-MC branch while the study record says
  # "factr". Reject it rather than record a run under the wrong mode.
  if is_truthy "${MC_SIM}"; then
    echo "[ERROR] FACTR_ACTIVE=1 and MC_SIM=1 cannot be used together." >&2
    echo "[HINT] MC_SIM routes interventions through the sim motion-controller path." >&2
    exit 1
  fi
  for _factr_file in "${FACTR_CONFIG}" "${FACTR_INIT_POSE}" "${FACTR_REST_POSE}"; do
    if [[ ! -f "${INTERVENE_ROOT}/${_factr_file}" ]] && [[ ! -f "${_factr_file}" ]]; then
      echo "[ERROR] Missing FACTR file: ${_factr_file}" >&2
      echo "[HINT] Commissioning reports live in factr/validation/ and are per-arm." >&2
      exit 1
    fi
  done
  # Pick an interpreter that can actually import the FACTR hardware stack, and prove it
  # BEFORE the service is spawned. Failing here costs a second; failing after the spawn
  # means the arm has already been connected and possibly moved.
  _factr_import_check='import pinocchio, dynamixel_sdk'
  if [[ -n "${FACTR_PYTHON_BIN_EXPLICIT}" ]]; then
    _factr_py_candidates=("${FACTR_PYTHON_BIN}")
  else
    _factr_py_candidates=()
    [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]] && \
      _factr_py_candidates+=("${CONDA_PREFIX}/bin/python")
    _factr_py_candidates+=("${PYTHON_BIN}")
  fi
  _factr_py_ok=""
  for _cand in "${_factr_py_candidates[@]}"; do
    [[ -x "${_cand}" ]] || continue
    if "${_cand}" -c "${_factr_import_check}" >/dev/null 2>&1; then
      _factr_py_ok="${_cand}"
      break
    fi
    echo "[PolicyLaunch] FACTR: ${_cand} cannot import pinocchio/dynamixel_sdk; trying next."
  done
  if [[ -z "${_factr_py_ok}" ]]; then
    echo "[ERROR] No Python found that can import the FACTR hardware stack (pinocchio, dynamixel_sdk)." >&2
    echo "[HINT] Tried: ${_factr_py_candidates[*]}" >&2
    echo "[HINT] Set FACTR_PYTHON_BIN to an interpreter that has both, e.g." >&2
    echo "[HINT]   FACTR_PYTHON_BIN=\$HOME/miniforge3/envs/polymetis/bin/python" >&2
    exit 1
  fi
  FACTR_PYTHON_BIN="${_factr_py_ok}"

  # Simulation speed relative to wall-clock time. All user-study conditions ran at 0.2.
  # An explicit REALTIME_FACTOR still wins.
  REALTIME_FACTOR="${REALTIME_FACTOR:-0.2}"

  # The touch-wait blocks the policy's main loop until the operator pushes the arm.
  # Without this the 3 s watchdog dumps a thread traceback on every takeover.
  if [[ -z "${INTERVENE_MAIN_LOOP_WATCHDOG_S:-}" ]] && is_truthy "${FACTR_TOUCH_RELEASE_ENABLED}"; then
    export INTERVENE_MAIN_LOOP_WATCHDOG_S=600
  fi

  echo "[PolicyLaunch] FACTR_ACTIVE=1: leader arm drives the SIMULATED robot during interventions."
  echo "[PolicyLaunch]   realtime_factor=${REALTIME_FACTOR} (default 0.2, as used in the user studies)"
  echo "[PolicyLaunch]   config=${FACTR_CONFIG} service=${FACTR_SERVICE_HOST}:${FACTR_SERVICE_PORT}"
  echo "[PolicyLaunch]   python=${FACTR_PYTHON_BIN}"
  if is_truthy "${FACTR_TOUCH_RELEASE_ENABLED}"; then
    echo "[PolicyLaunch]   takeover: align -> hold -> WAIT for your push (>=${FACTR_TOUCH_RELEASE_DELTA_RAD} rad$(
        [[ "${FACTR_TOUCH_RELEASE_TIMEOUT}" == "0.0" || "${FACTR_TOUCH_RELEASE_TIMEOUT}" == "0" ]] \
          && printf ', no timeout' || printf ', timeout %ss' "${FACTR_TOUCH_RELEASE_TIMEOUT}"
      )) -> blend ${FACTR_TAKEOVER_TORQUE_BLEND_SECONDS}s -> leader mode"
    echo "[PolicyLaunch]   rpc_timeout=${FACTR_RPC_TIMEOUT}s (must exceed human reaction time)"
  else
    echo "[PolicyLaunch]   takeover: IMMEDIATE release on keypress (touch-wait DISABLED)"
  fi
  echo "[PolicyLaunch]   Keep a hand on FACTR and be ready to cut motor power."
  # NOTE: the RPC service holds its global lock for the duration of the touch-wait, so
  # other sessions' FACTR RPCs block until the operator takes the arm. Accepted: there is
  # one human and one takeover at a time.
  # NEVER start factr/src/factr_intervention_server.py bare — its hold gravity/friction
  # comp default to False and the arm falls. Always launch through this script.
fi

if is_truthy "${MC_MODE}"; then
  START_VR=0
  echo "[PolicyLaunch] MC_MODE=1: VR runner suppressed. Launch Quest in SIIScene_MotionControllerOnly (ZMQ port 6090)."
fi

if is_truthy "${START_VR}"; then
  if [[ ! -f "${VR_RUNNER}" ]]; then
    echo "[ERROR] Missing VR runner: ${VR_RUNNER}" >&2
    exit 1
  fi
fi

XML_REAL="$(realpath "${XML}")"
RESET_REAL="$(realpath "${RESET_NPZ}")"

echo "[PolicyLaunch] Intervene root: ${INTERVENE_ROOT}"
echo "[PolicyLaunch] Umbrella root: ${UMBRELLA_ROOT}"
echo "[PolicyLaunch] Python: ${PYTHON_BIN}"
if [[ -n "${POLYMETIS_PYTHON_DIR:-}" ]]; then
  echo "[PolicyLaunch] Polymetis Python: ${POLYMETIS_PYTHON_DIR}"
fi
echo "[PolicyLaunch] LAB: ${LAB} (fast_scene=${LAB_FAST}); switch with LAB=tshape|cups"
echo "[PolicyLaunch] XML: ${XML}"
echo "[PolicyLaunch] Reset NPZ: ${RESET_NPZ}"
echo "[PolicyLaunch] Checkpoint: ${CHECKPOINT}"
echo "[PolicyLaunch] Start VR: ${START_VR}"
echo "[PolicyLaunch] Start grid UI: ${START_GRID_UI}"
echo "[PolicyLaunch] VR policy mirror: ${VR_POLICY_MIRROR}"
echo "[PolicyLaunch] Randomization preset: ${INTERVENE_RANDOMIZATION_PRESET}"
echo "[PolicyLaunch] Randomize scene: ${INTERVENE_RANDOMIZE_SCENE} (xy +/-${INTERVENE_OBJECT_XY_RANDOM_RANGE} m, euler ${INTERVENE_OBJECT_EULER_RANDOM_RANGE_DEG} deg)"
echo "[PolicyLaunch] Randomize first episode: ${INTERVENE_RANDOMIZE_FIRST_EPISODE} (0 = every cell starts on the identical reset pose)"
echo "[PolicyLaunch] Randomize objects: ${INTERVENE_RANDOMIZE_OBJECT_NAMES}"
echo "[PolicyLaunch] Task eval: ${INTERVENE_AUTO_TASK_EVAL} mode=${INTERVENE_TASK_MODE} stable_steps=${INTERVENE_TASK_PLACEMENT_STABLE_STEPS}"
# --- Robot reachability preflight ---------------------------------------------
# Telekinesis intervention (the default, MC_ACTIVE=0) switches the REAL arm to
# HUMAN_CONTROL, so it cannot start without a reachable polymetis server. Without this
# check the failure only appears as a gRPC "failed to connect to all addresses" buried
# in a per-session policy log, seconds AFTER the operator presses X in the headset.
# Non-fatal: the sim grid is still useful without an arm.
if is_truthy "${ROBOT_PREFLIGHT:-1}"; then
  _doctor="${INTERVENE_ROOT}/utils/robot_doctor.py"
  if [[ -f "${_doctor}" ]]; then
    echo "[PolicyLaunch] Robot key: ${ROBOT_KEY} (app.py + VR runner + mq3_mc all use this)"
    if ! "${PYTHON_BIN}" "${_doctor}" --quiet --timeout "${ROBOT_PREFLIGHT_TIMEOUT:-2.0}"; then
      echo "[PolicyLaunch]   Telekinesis intervention will NOT start until this is fixed."
      echo "[PolicyLaunch]   Run '${PYTHON_BIN} ${_doctor}' for the full report. Set ROBOT_PREFLIGHT=0 to skip."
    fi
  fi
fi

if [[ -n "${STUDY_PARTICIPANT}" ]]; then
  echo "[PolicyLaunch] STUDY: participant=${STUDY_PARTICIPANT} ${STUDY_LABEL} seed=${STUDY_SEED} acc_method=${STUDY_ACC_METHOD}"
  echo "[PolicyLaunch] STUDY: VR interface=${STUDY_INTERFACE} (RGB=${STUDY_VR_RGB} WITH_PC=${STUDY_VR_PC}); failures ARE saved"
  echo "[PolicyLaunch] STUDY: control=${STUDY_CONTROL:-telekinesis} (FACTR_ACTIVE=${FACTR_ACTIVE} MC_ACTIVE=${MC_ACTIVE} MC_SIM=${MC_SIM})"
else
  echo "[PolicyLaunch] STUDY: disabled (set STUDY_PARTICIPANT to enable study logging)"
fi
echo "[PolicyLaunch] Policy command: tcp://${POLICY_CMD_BIND}:${POLICY_CMD_PORT}"
echo "[PolicyLaunch] Policy state: tcp://${POLICY_STATE_BIND}:${POLICY_STATE_PORT} @ ${POLICY_STATE_HZ} Hz"
if is_truthy "${MC_ACTIVE}"; then
  if is_truthy "${MC_SIM}"; then
    echo "[PolicyLaunch] MC_ACTIVE=1 MC_SIM=1: sim-native motion controller (no real robot, no mq3_mc.py sidecar)"
  else
    echo "[PolicyLaunch] MC_ACTIVE=1 MC_SIM=0: real-robot motion controller (mq3_mc.py sidecar)"
  fi
fi
if [[ -n "${WINDOWS:-}" ]]; then
  echo "[PolicyLaunch] VR mode: MULTI (WINDOWS=${WINDOWS})"
else
  echo "[PolicyLaunch] VR mode: SINGLE (selector grid bypassed)"
fi

EXTRA_ARGS=()
if [[ -n "${ARM_DELTA_CLIP:-}" ]]; then
  EXTRA_ARGS+=(--arm_delta_clip "${ARM_DELTA_CLIP}")
fi
if [[ -n "${DUMP_CAMERA_IMAGES:-}" ]]; then
  EXTRA_ARGS+=(--dump_camera_images "${DUMP_CAMERA_IMAGES}")
fi

# ---------------------------------------------------------------------------
# Independent policies — one PER WINDOW.
# Each grid window is its OWN policy rollout (own randomized scene, own auto-eval,
# own recorder, own ACC risk, own intervention machinery). Instance i uses
#   cmd   port = POLICY_CMD_PORT   + i*POLICY_PORT_STEP
#   state port = POLICY_STATE_PORT + i*POLICY_PORT_STEP
# and VR session i mirrors policy i (per-session ports are computed by
# multi_session_launcher.py from the same base+step). Previously ONE policy's
# state was forwarded to ALL sessions — N identical mirrors ("watch party"),
# which defeated the multi-policy monitoring purpose of the project.
# ---------------------------------------------------------------------------
POLICY_PORT_STEP="${POLICY_PORT_STEP:-10}"
if [[ -n "${WINDOWS:-}" ]]; then
  N_POLICIES="${WINDOWS}"
else
  N_POLICIES=1
fi

# Base command WITHOUT cmd/state ports — those are appended per instance.
POLICY_CMD=(
  "${PYTHON_BIN}" main.py
  "${XML}" \
  "${RESET_NPZ}" \
  --mode policy \
  --checkpoint "${CHECKPOINT}" \
  --view "${VIEW:-cameras}" \
  --action_mode "${ACTION_MODE:-queue}" \
  --arm_action_mode "${ARM_ACTION_MODE:-absolute}" \
  --gripper_action_mode "${GRIPPER_ACTION_MODE:-absolute}" \
  --policy_hz "${POLICY_HZ:-10}" \
  --realtime_factor "${REALTIME_FACTOR:-0.2}" \
  --max_steps "${MAX_STEPS:-0}" \
  "${EXTRA_ARGS[@]}" \
  "$@"
)

_policy_instance_port_args() {
  # $1 = instance index → echoes the per-instance cmd/state port args.
  local i="$1"
  local cmd_port=$((POLICY_CMD_PORT + i * POLICY_PORT_STEP))
  local state_port=$((POLICY_STATE_PORT + i * POLICY_PORT_STEP))
  local out=()
  if [[ "${POLICY_CMD_PORT}" -gt 0 ]]; then
    out+=(--cmd_bind "${POLICY_CMD_BIND}" --cmd_port "${cmd_port}")
  fi
  if [[ "${POLICY_STATE_PORT}" -gt 0 ]]; then
    out+=(--state_bind "${POLICY_STATE_BIND}" --state_port "${state_port}" --state_hz "${POLICY_STATE_HZ}")
  fi
  echo "${out[@]:-}"
}

# FACTR is excluded from the exec fast-path on purpose: this shell has to stay alive to
# supervise the FACTR service and to stop it on the way out. exec'ing away would leave the
# leader arm torque-enabled with nothing owning it.
if ! is_truthy "${START_VR}" && [[ -z "${WINDOWS:-}" ]] && ! is_truthy "${FACTR_ACTIVE}"; then
  # Legacy policy-only mode: single visible instance 0 with its base ports.
  exec "${POLICY_CMD[@]}" $(_policy_instance_port_args 0)
fi

# Points per camera per frame for the VR runtimes. 200k is right for a few windows; at 9+
# concurrent sessions it is the dominant ACTIVE cost (measured 2026-07-29: 31.7% of ACTIVE
# frames over a 16.7 ms budget, max tick 238 ms). The project already used 80k for 15
# windows. Computed here because the VR_ENV array literals below take NAME=value only.
if [[ -n "${WINDOWS:-}" ]] && (( WINDOWS >= 9 )); then
  _PC_MAX_POINTS_DEFAULT=80000
else
  _PC_MAX_POINTS_DEFAULT=200000
fi

VR_EXTRA_RUNTIME_ARGS_VALUE="${VR_EXTRA_RUNTIME_ARGS:-${EXTRA_RUNTIME_ARGS:-}}"
VR_FORWARD_ARGS=""
if ! is_truthy "${VR_POLICY_MIRROR}"; then
  if [[ "${VR_EXTRA_RUNTIME_ARGS_VALUE}" == *"--policy_state_"* \
     || "${VR_EXTRA_RUNTIME_ARGS_VALUE}" == *"--policy_cmd_"* ]]; then
    echo "[PolicyLaunch][ERROR] VR_POLICY_MIRROR=0 conflicts with policy wiring in VR_EXTRA_RUNTIME_ARGS/EXTRA_RUNTIME_ARGS." >&2
    exit 2
  fi
  echo "[PolicyLaunch] DIAGNOSTIC: VR runtimes use local trajectory replay; policies remain running."
elif [[ -n "${WINDOWS:-}" ]]; then
  # MULTI mode: per-session policy ports are computed by multi_session_launcher.py from
  # base+step (session i mirrors policy i). Only the port-agnostic timeout is forwarded
  # via the shared extras string — the old fixed --policy_state_port here was what made
  # every window an identical mirror of policy 0.
  VR_FORWARD_ARGS="--policy_state_timeout_s ${POLICY_STATE_TIMEOUT_S}"
else
  # SINGLE mode: one policy, one session — keep the fixed forwarding (backward compat).
  VR_FORWARD_ARGS="--policy_cmd_host ${POLICY_CMD_BIND} --policy_cmd_port ${POLICY_CMD_PORT} --policy_state_host ${POLICY_STATE_HOST} --policy_state_port ${POLICY_STATE_PORT} --policy_state_timeout_s ${POLICY_STATE_TIMEOUT_S}"
fi
if [[ -n "${VR_EXTRA_RUNTIME_ARGS_VALUE}" && -n "${VR_FORWARD_ARGS}" ]]; then
  VR_EXTRA_RUNTIME_ARGS_VALUE="${VR_EXTRA_RUNTIME_ARGS_VALUE} ${VR_FORWARD_ARGS}"
elif [[ -n "${VR_FORWARD_ARGS}" ]]; then
  VR_EXTRA_RUNTIME_ARGS_VALUE="${VR_FORWARD_ARGS}"
fi

if [[ -n "${WINDOWS:-}" ]]; then
  VR_ENV=(
    WINDOWS="${WINDOWS}"
    SINGLE=0
    ROBOT=0
    TRAJ="${RESET_REAL}"
    LAB_XML="${XML_REAL}"
    FPS="${FPS:-60}"
    # Round-robin defaults ON in the policy path: with N policy processes and N VR
    # runtimes sharing one GL driver, rendering all PC cameras every tick was the
    # dominant cost (measured 2026-07-28: render 51 ms vs a 17 ms budget, point clouds
    # publishing at 18.6 Hz instead of 60). One camera per tick keeps spatial density
    # identical (lesson 37) and only lowers each camera's refresh to fps/3.
    # OFF. Round-robin renders ONE camera per publish tick, so each camera's refresh rate
    # becomes fps / n_working_cameras -- with front+top+right+left that is 60/4 = 15 Hz.
    # Measured 2026-07-30: round_robin=1 -> 14.5 Hz per camera; round_robin=0 -> 60.0 Hz.
    # That 4x cut is what made the point clouds visibly lag the robot. It was off during the
    # period this system worked well (2026-07-13 onward) and was re-enabled on 07-29 as a
    # render-cost saving -- a saving the recorder change now covers,
    # having removed 62% of all render work. If the ACTIVE session goes over budget again,
    # lower FPS (45 still gives 45 Hz/camera) rather than turning this back on.
    PC_ROUND_ROBIN="${PC_ROUND_ROBIN:-0}"
    PC_FPS_CAP="${PC_FPS_CAP:-30}"
    PC_MAX_POINTS="${PC_MAX_POINTS:-${_PC_MAX_POINTS_DEFAULT}}"
    PC_STRIDE="${PC_STRIDE:-2}"
    PC_WIDTH="${PC_WIDTH:-640}"
    PC_HEIGHT="${PC_HEIGHT:-480}"
    PERF_LOG="${PERF_LOG:-0}"
    MAX_RESTARTS="${VR_MAX_RESTARTS:-2}"
    METRICS="${METRICS:-0}"
    METRICS_WINDOW_S="${METRICS_WINDOW_S:-10}"
    METRICS_SUMMARY="${METRICS_SUMMARY:-0}"
    EXTRA_RUNTIME_ARGS="${VR_EXTRA_RUNTIME_ARGS_VALUE}"
    POLICY_CMD_BASE_PORT=""
    POLICY_STATE_BASE_PORT=""
    POLICY_PORT_STEP="${POLICY_PORT_STEP}"
    POLICY_HOST="${POLICY_STATE_HOST}"
  )
  if is_truthy "${VR_POLICY_MIRROR}"; then
    VR_ENV+=(POLICY_CMD_BASE_PORT="${POLICY_CMD_PORT}")
    VR_ENV+=(POLICY_STATE_BASE_PORT="${POLICY_STATE_PORT}")
  fi
  if [[ -n "${BASE_TOPIC_PORT:-}" ]]; then
    VR_ENV+=(BASE_TOPIC_PORT="${BASE_TOPIC_PORT}")
  fi
  # ALWAYS explicit, unlike RGB/WITH_PC below: MC_ACTIVE may have been DERIVED from the
  # condition matrix rather than typed on the command line, and a derived value is a plain
  # shell variable that `env` would not pass down. The VR runner forwards it to the launcher,
  # which advertises it in the discovery beacon so the headset knows to reserve the right
  # controller for the motion controller while an intervention is live.
  VR_ENV+=(MC_ACTIVE="${MC_ACTIVE}")
  if [[ -n "${STUDY_PARTICIPANT}" ]]; then
    VR_ENV+=(RGB="${STUDY_VR_RGB}" WITH_PC="${STUDY_VR_PC}")
  elif [[ -n "${EXP_PARTICIPANT}" ]]; then
    # Passed explicitly rather than relying on inheritance: the block has already
    # verified these against the condition, so they are the authoritative values.
    VR_ENV+=(RGB="${RGB}" WITH_PC="${WITH_PC}")
  fi

else
  VR_ENV=(
    SINGLE=1
    ROBOT=0
    TRAJ="${RESET_REAL}"
    LAB_XML="${XML_REAL}"
    FPS="${FPS:-60}"
    # Round-robin defaults ON in the policy path: with N policy processes and N VR
    # runtimes sharing one GL driver, rendering all PC cameras every tick was the
    # dominant cost (measured 2026-07-28: render 51 ms vs a 17 ms budget, point clouds
    # publishing at 18.6 Hz instead of 60). One camera per tick keeps spatial density
    # identical (lesson 37) and only lowers each camera's refresh to fps/3.
    # OFF. Round-robin renders ONE camera per publish tick, so each camera's refresh rate
    # becomes fps / n_working_cameras -- with front+top+right+left that is 60/4 = 15 Hz.
    # Measured 2026-07-30: round_robin=1 -> 14.5 Hz per camera; round_robin=0 -> 60.0 Hz.
    # That 4x cut is what made the point clouds visibly lag the robot. It was off during the
    # period this system worked well (2026-07-13 onward) and was re-enabled on 07-29 as a
    # render-cost saving -- a saving the recorder change now covers,
    # having removed 62% of all render work. If the ACTIVE session goes over budget again,
    # lower FPS (45 still gives 45 Hz/camera) rather than turning this back on.
    PC_ROUND_ROBIN="${PC_ROUND_ROBIN:-0}"
    PC_FPS_CAP="${PC_FPS_CAP:-30}"
    PC_MAX_POINTS="${PC_MAX_POINTS:-${_PC_MAX_POINTS_DEFAULT}}"
    PC_STRIDE="${PC_STRIDE:-3}"
    PC_WIDTH="${PC_WIDTH:-640}"
    PC_HEIGHT="${PC_HEIGHT:-480}"
    PERF_LOG="${PERF_LOG:-0}"
    MAX_RESTARTS="${VR_MAX_RESTARTS:-2}"
    METRICS="${METRICS:-0}"
    METRICS_WINDOW_S="${METRICS_WINDOW_S:-10}"
    METRICS_SUMMARY="${METRICS_SUMMARY:-0}"
    EXTRA_RUNTIME_ARGS="${VR_EXTRA_RUNTIME_ARGS_VALUE}"
    POLICY_CMD_BASE_PORT=""
    POLICY_STATE_BASE_PORT=""
  )
  if [[ -n "${BASE_TOPIC_PORT:-}" ]]; then
    VR_ENV+=(BASE_TOPIC_PORT="${BASE_TOPIC_PORT}")
  fi
  # See the MULTI branch above: always explicit, because a condition-derived MC_ACTIVE is
  # not exported and would otherwise never reach the launcher's discovery beacon.
  VR_ENV+=(MC_ACTIVE="${MC_ACTIVE}")
  if [[ -n "${STUDY_PARTICIPANT}" ]]; then
    VR_ENV+=(RGB="${STUDY_VR_RGB}" WITH_PC="${STUDY_VR_PC}")
  elif [[ -n "${EXP_PARTICIPANT}" ]]; then
    # Passed explicitly rather than relying on inheritance: the block has already
    # verified these against the condition, so they are the authoritative values.
    VR_ENV+=(RGB="${RGB}" WITH_PC="${WITH_PC}")
  fi
fi

policy_pid=""
policy_pids=()
vr_pid=""
mc_pid=""
grid_pid=""
factr_pid=""
factr_log=""
factr_ready_file=""
cleanup_started=0

is_running() {
  local pid="$1"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

cleanup_children() {
  if [[ "${cleanup_started}" -eq 1 ]]; then
    return
  fi
  cleanup_started=1

  # Send SIGINT to every policy instance so Python converts it to KeyboardInterrupt,
  # which propagates through try/finally blocks and triggers the robot reset + gripper
  # close before the process exits.  SIGTERM would bypass Python cleanup entirely.
  local p
  for p in "${policy_pids[@]:-}"; do
    if is_running "${p}"; then
      echo "[PolicyLaunch] Sending SIGINT to policy pid=${p} (robot reset)..."
      kill -INT "${p}" 2>/dev/null || true
    fi
  done

  # Kill the VR process GROUP immediately — no robot there, no need to wait.
  if [[ -n "${vr_pid:-}" ]]; then
    kill -TERM -- -"${vr_pid}" 2>/dev/null || true
    kill -TERM "${vr_pid}" 2>/dev/null || true
  fi

  if [[ -n "${grid_pid:-}" ]]; then
    kill -TERM "${grid_pid}" 2>/dev/null || true
  fi

  # Wait up to 10 s for the policies to finish their robot reset, then force-kill.
  local i any_alive
  for i in $(seq 1 10); do
    any_alive=0
    for p in "${policy_pids[@]:-}"; do
      if is_running "${p}"; then any_alive=1; break; fi
    done
    [[ "${any_alive}" -eq 0 ]] && break
    sleep 1
  done
  for p in "${policy_pids[@]:-}"; do
    if is_running "${p}"; then
      echo "[PolicyLaunch] Policy pid=${p} still running after 10s; force-killing."
      kill -9 "${p}" 2>/dev/null || true
    fi
    wait "${p}" 2>/dev/null || true
  done

  # Hard-kill VR group and wait
  if [[ -n "${vr_pid:-}" ]]; then
    kill -9 -- -"${vr_pid}" 2>/dev/null || true
    wait "${vr_pid}" 2>/dev/null || true
  fi

  if [[ -n "${grid_pid:-}" ]]; then
    kill -9 "${grid_pid}" 2>/dev/null || true
    wait "${grid_pid}" 2>/dev/null || true
  fi

  # Kill mq3_mc.py sidecar (SIGINT first so gripper/robot reset can run, then force)
  if [[ -n "${mc_pid:-}" ]]; then
    kill -INT "${mc_pid}" 2>/dev/null || true
    sleep 1
    kill -9 "${mc_pid}" 2>/dev/null || true
    wait "${mc_pid}" 2>/dev/null || true
  fi

  # FACTR service LAST, deliberately. It owns the serial port, so the policy clients must
  # have had their chance to release takeover first; and its own shutdown drives the
  # leader arm back to the rest pose, which is a real physical move — hence the generous
  # FACTR_SHUTDOWN_TIMEOUT before any force-kill.
  if [[ -n "${factr_pid:-}" ]]; then
    echo "[PolicyLaunch] Stopping FACTR service (parking the leader arm)..."
    kill -INT "${factr_pid}" 2>/dev/null || true
    local factr_wait
    for factr_wait in $(seq 1 "${FACTR_SHUTDOWN_TIMEOUT}"); do
      is_running "${factr_pid}" || break
      sleep 1
    done
    if is_running "${factr_pid}"; then
      echo "[PolicyLaunch] FACTR service did not stop in ${FACTR_SHUTDOWN_TIMEOUT}s; forcing." >&2
      kill -9 "${factr_pid}" 2>/dev/null || true
    fi
    wait "${factr_pid}" 2>/dev/null || true
  fi
  if [[ -n "${factr_ready_file:-}" ]]; then
    rm -f "${factr_ready_file}"
  fi
}

on_interrupt() {
  trap '' INT TERM   # prevent re-entrant Ctrl+C during cleanup
  echo "[PolicyLaunch] Stopping policy + VR..."
  cleanup_children
  exit 130
}
trap on_interrupt INT TERM

# Kill any stale policy or VR processes from a previous run before taking their ports.
# Two-pass: (1) cmdline scan via pgrep, (2) port-based scan for anything pgrep missed.
# IMPORTANT: stale policy processes (main.py) holding 8065/8066 must be killed here —
# if they survive, the new policy disables its cmd/state sockets and the VR runtime
# connects to the old session's frozen state instead of the new one, causing the Quest
# view and the MuJoCo window to show completely different content.

# Pass 1a: kill stale policy + mq3_mc processes
_stale_mc=$(pgrep -f "mq3_mc\.py" 2>/dev/null || true)
if [[ -n "${_stale_mc}" ]]; then
  echo "[PolicyLaunch] Killing stale mq3_mc processes: $(echo ${_stale_mc} | tr '\n' ' ')"
  echo "${_stale_mc}" | xargs -r kill -INT 2>/dev/null || true
  sleep 2
  echo "${_stale_mc}" | xargs -r kill -9 2>/dev/null || true
  sleep 1
fi

_stale_policy=$(pgrep -f "main\.py.*--mode.*policy\|main\.py.*mode policy" 2>/dev/null || true)
if [[ -n "${_stale_policy}" ]]; then
  echo "[PolicyLaunch] Killing stale policy processes (cmdline scan): $(echo ${_stale_policy} | tr '\n' ' ')"
  echo "${_stale_policy}" | xargs -r kill -INT 2>/dev/null || true
  sleep 3
  echo "${_stale_policy}" | xargs -r kill -9 2>/dev/null || true
  sleep 1
fi

# Pass 1b: kill anything still holding a policy cmd/state port (all N instances).
for _i in $(seq 0 $((N_POLICIES - 1))); do
  for _port in $((POLICY_CMD_PORT + _i * POLICY_PORT_STEP)) $((POLICY_STATE_PORT + _i * POLICY_PORT_STEP)); do
    _pid=$(ss -tlnp "sport = :${_port}" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1 || true)
    if [[ -n "${_pid}" && "${_pid}" != "$$" ]]; then
      echo "[PolicyLaunch] Killing process holding policy port ${_port}: pid=${_pid}"
      kill -9 "${_pid}" 2>/dev/null || true
    fi
  done
done

if is_truthy "${START_VR}"; then
  # Pass 2a: kill stale VR processes
  _stale=$(pgrep -f "multi_session_launcher.py" 2>/dev/null || true)
  _stale+=" "$(pgrep -f "intervention_vr_runtime.py" 2>/dev/null || true)
  _stale=$(echo "${_stale}" | tr ' ' '\n' | grep -v '^$' | sort -u || true)
  if [[ -n "${_stale}" ]]; then
    echo "[PolicyLaunch] Killing stale VR processes (cmdline scan): $(echo ${_stale} | tr '\n' ' ')"
    echo "${_stale}" | xargs -r kill -TERM 2>/dev/null || true
    sleep 2
    echo "${_stale}" | xargs -r kill -9 2>/dev/null || true
    sleep 1
  fi

  # Pass 2b: port-based pass for VR ports (7740–7750)
  _port_pids=""
  for _port in 7740 7741 7742 7743 7744 7745 7746 7747 7748 7749 7750; do
    _pid=$(ss -tlnp "sport = :${_port}" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | head -1 || true)
    if [[ -n "${_pid}" && "${_pid}" != "$$" ]]; then
      _port_pids+=" ${_pid}"
    fi
  done
  _port_pids=$(echo "${_port_pids}" | tr ' ' '\n' | grep -v '^$' | sort -u || true)
  if [[ -n "${_port_pids}" ]]; then
    echo "[PolicyLaunch] Killing processes still holding VR ports: $(echo ${_port_pids} | tr '\n' ' ')"
    echo "${_port_pids}" | xargs -r kill -9 2>/dev/null || true
    sleep 1
  fi
fi

# --- FACTR service ------------------------------------------------------------
# One process owns the serial device for ALL policy windows and arbitrates it via its own
# claim/release protocol. It must be initialized (rest pose verified, moved to the init
# pose, socket listening) BEFORE any policy starts, or the first intervention races a
# half-initialized arm.
if is_truthy "${FACTR_ACTIVE}"; then
  _stale_factr=$(pgrep -f "factr/src/factr_intervention_server\.py" 2>/dev/null || true)
  if [[ -n "${_stale_factr}" ]]; then
    echo "[PolicyLaunch] Stopping stale FACTR service: $(echo ${_stale_factr} | tr '\n' ' ')"
    echo "${_stale_factr}" | xargs -r kill -INT 2>/dev/null || true
    sleep 2
    echo "${_stale_factr}" | xargs -r kill -9 2>/dev/null || true
  fi

  mkdir -p "${UMBRELLA_ROOT}/session_logs" 2>/dev/null || true
  _factr_ts=$(date +%Y%m%d_%H%M%S)
  factr_log="${UMBRELLA_ROOT}/session_logs/factr_service_${_factr_ts}.log"
  factr_ready_file="/tmp/intervene_factr_${USER:-user}_${FACTR_SERVICE_PORT}.ready"
  rm -f "${factr_ready_file}"

  echo "[PolicyLaunch] Starting FACTR service; policy processes will wait for initialization."
  echo "[PolicyLaunch]   log -> ${factr_log}"
  ( cd "${INTERVENE_ROOT}" && env \
      FACTR_CONFIG="${FACTR_CONFIG}" \
      FACTR_MAX_TORQUE="${FACTR_MAX_TORQUE}" \
      INTERVENE_FACTR_MAX_TORQUE="${FACTR_MAX_TORQUE}" \
      INTERVENE_FACTR_ALLOW_OVER_CONFIG_TORQUE="${FACTR_ALLOW_OVER_CONFIG_TORQUE}" \
      INTERVENE_FACTR_HOLD_ERROR="${FACTR_HOLD_ERROR}" \
      INTERVENE_FACTR_DRIVE_TORQUE="${FACTR_DRIVE_TORQUE}" \
      INTERVENE_FACTR_DRIVE_DEADBAND="${FACTR_DRIVE_DEADBAND}" \
      INTERVENE_FACTR_DRIVE_RAMP="${FACTR_DRIVE_RAMP}" \
      INTERVENE_FACTR_TRAJECTORY_DURATION="${FACTR_TRAJECTORY_DURATION}" \
      INTERVENE_FACTR_HOLD_GRAVITY_COMP="${FACTR_HOLD_GRAVITY_COMP}" \
      INTERVENE_FACTR_HOLD_FRICTION_COMP="${FACTR_HOLD_FRICTION_COMP}" \
      INTERVENE_FACTR_WAIT_MAX_TORQUE="${FACTR_WAIT_MAX_TORQUE}" \
      INTERVENE_FACTR_WAIT_HOLD_ERROR="${FACTR_WAIT_HOLD_ERROR}" \
      INTERVENE_FACTR_WAIT_DRIVE_TORQUE="${FACTR_WAIT_DRIVE_TORQUE}" \
      INTERVENE_FACTR_WAIT_HOLD_GRAVITY_COMP="${FACTR_WAIT_HOLD_GRAVITY_COMP}" \
      INTERVENE_FACTR_WAIT_HOLD_FRICTION_COMP="${FACTR_WAIT_HOLD_FRICTION_COMP}" \
      INTERVENE_FACTR_TAKEOVER_GRAVITY_SCALE="${FACTR_TAKEOVER_GRAVITY_SCALE}" \
      INTERVENE_FACTR_TAKEOVER_GRAVITY_RAMP_DURATION="${FACTR_TAKEOVER_GRAVITY_RAMP_DURATION}" \
      INTERVENE_FACTR_TAKEOVER_FRICTION_SCALE="${FACTR_TAKEOVER_FRICTION_SCALE}" \
      INTERVENE_FACTR_TAKEOVER_NULLSPACE_SCALE="${FACTR_TAKEOVER_NULLSPACE_SCALE}" \
      INTERVENE_FACTR_TAKEOVER_SETTLE_SECONDS="${FACTR_TAKEOVER_SETTLE_SECONDS}" \
      INTERVENE_FACTR_TAKEOVER_TORQUE_BLEND_SECONDS="${FACTR_TAKEOVER_TORQUE_BLEND_SECONDS}" \
      INTERVENE_FACTR_TAKEOVER_HOLD_ASSIST_SECONDS="${FACTR_TAKEOVER_HOLD_ASSIST_SECONDS}" \
      INTERVENE_FACTR_TOUCH_RELEASE_ENABLED="${FACTR_TOUCH_RELEASE_ENABLED}" \
      INTERVENE_FACTR_TOUCH_RELEASE_DELTA_RAD="${FACTR_TOUCH_RELEASE_DELTA_RAD}" \
      INTERVENE_FACTR_TOUCH_RELEASE_TIMEOUT="${FACTR_TOUCH_RELEASE_TIMEOUT}" \
      INTERVENE_FACTR_RETURN_INIT_ON_RELEASE="${FACTR_RETURN_INIT_ON_RELEASE}" \
      INTERVENE_FACTR_RETURN_REST_ON_CLOSE="${FACTR_RETURN_REST_ON_CLOSE}" \
      INTERVENE_FACTR_REST_TIMEOUT="${FACTR_REST_TIMEOUT}" \
      INTERVENE_FACTR_REST_GOAL_TOLERANCE="${FACTR_REST_GOAL_TOLERANCE}" \
      INTERVENE_FACTR_REST_DISABLE_LIMIT_TORQUE="${FACTR_REST_DISABLE_LIMIT_TORQUE}" \
      INTERVENE_FACTR_PRESERVE_ALIGNED_ANCHOR="${FACTR_PRESERVE_ALIGNED_ANCHOR}" \
      "${FACTR_PYTHON_BIN}" factr/src/factr_intervention_server.py \
        --host "${FACTR_SERVICE_HOST}" \
        --port "${FACTR_SERVICE_PORT}" \
        --ready-file "${factr_ready_file}" \
        --config "${FACTR_CONFIG}" \
        --init-pose "${FACTR_INIT_POSE}" \
        --rest-pose "${FACTR_REST_POSE}" \
        --rest-position-tolerance "${FACTR_REST_POSITION_TOLERANCE}" \
        --position-tolerance "${FACTR_POSITION_TOLERANCE}" \
        --init-timeout "${FACTR_INIT_TIMEOUT}" \
        --alignment-timeout "${FACTR_ALIGNMENT_TIMEOUT}" \
        --max-velocity "${FACTR_MAX_VELOCITY}" \
  ) >"${factr_log}" 2>&1 &
  factr_pid=$!
  echo "[PolicyLaunch] FACTR service pid=${factr_pid}"

  # The ready file is the real gate: the service writes it only after the rest pose is
  # verified, the arm has reached the init pose, and the socket is listening.
  _factr_ready=0
  for _i in $(seq 1 "${FACTR_STARTUP_TIMEOUT}"); do
    if [[ -s "${factr_ready_file}" ]] && is_running "${factr_pid}"; then
      _factr_ready=1
      break
    fi
    if ! is_running "${factr_pid}"; then
      break
    fi
    sleep 1
  done
  if [[ "${_factr_ready}" -ne 1 ]]; then
    echo "[ERROR] FACTR failed to initialize before policy startup. Log: ${factr_log}" >&2
    tail -n 40 "${factr_log}" >&2 || true
    cleanup_children
    exit 1
  fi
  echo "[PolicyLaunch] FACTR initialized and waiting; starting MuJoCo policies now."
fi

policy_pids=()
policy_logs=()
_policy_ts=$(date +%Y%m%d_%H%M%S)
POLICY_ENV=()

# OOD scene band. Shared by every session; only INTERVENE_OOD_INITIAL_STATE differs per
# instance and is added in the launch loop below.
if [[ "${OOD_MIN_STATES}" -gt 0 ]]; then
  POLICY_ENV+=(
    INTERVENE_OOD_MIN_STATES="${OOD_MIN_STATES}"
    INTERVENE_OOD_MAX_STATES="${OOD_MAX_STATES}"
    INTERVENE_OOD_MAX_SCENE_ATTEMPTS="${OOD_MAX_SCENE_ATTEMPTS}"
    INTERVENE_OOD_LEDGER_KEY="${OOD_LEDGER_KEY}"
    INTERVENE_OOD_LEDGER_TTL_S="${OOD_LEDGER_TTL_S}"
    INTERVENE_OOD_SCENE_DIR="${OOD_SCENE_DIR}"
    INTERVENE_OOD_TASK="${OOD_TASK}"
    INTERVENE_OOD_MODEL_VARIANTS="${OOD_MODEL_VARIANTS}"
    INTERVENE_OOD_VARIANT_PRECOMPILE="${OOD_VARIANT_PRECOMPILE}"
    INTERVENE_OOD_VARIANT_CACHE="${OOD_VARIANT_CACHE}"
    INTERVENE_OBJECT_POSITION_FROM_XML_NAMES="${OOD_POSITION_FROM_XML_NAMES}"
    INTERVENE_OBJECT_SUPPORT_Z_NAMES="${OOD_SUPPORT_Z_NAMES}"
    INTERVENE_OBJECT_SUPPORT_Z="${OOD_SUPPORT_Z}"
    INTERVENE_OBJECT_SUPPORT_MARGIN="${OOD_SUPPORT_MARGIN}"
    INTERVENE_OBJECT_ZERO_QVEL_NAMES="${OOD_ZERO_QVEL_NAMES}"
  )
fi
if [[ -n "${WINDOWS:-}" ]]; then
  # Multi-policy A/B runs must stay headless whether the aggregate grid is enabled or
  # disabled. Otherwise START_GRID_UI=0 replaces one grid renderer with N individual
  # policy viewers and no longer isolates desktop-grid GPU contention.
  POLICY_ENV+=(INTERVENE_HEADLESS=1)
  if is_truthy "${START_GRID_UI}"; then
    echo "[PolicyLaunch] Policy MuJoCo windows hidden; grid UI will render policy front cameras."
  else
    echo "[PolicyLaunch] Policy MuJoCo windows hidden; desktop grid disabled for A/B isolation."
  fi
fi

# Policy-side performance metrics. The VR runtimes have had windowed metrics for a while;
# the policy processes had none, which is why the episode recorder's cost (3 camera
# render+readbacks per frame per session) stayed invisible. Same METRICS switch as the VR
# side so one env var instruments both halves of the system.
if is_truthy "${METRICS:-0}"; then
  POLICY_ENV+=(INTERVENE_METRICS=1)
  POLICY_ENV+=(INTERVENE_METRICS_WINDOW_S="${METRICS_WINDOW_S:-10}")
  if is_truthy "${METRICS_SUMMARY:-0}"; then
    POLICY_ENV+=(INTERVENE_METRICS_SUMMARY=1)
  fi
  echo "[PolicyLaunch] Policy metrics ON -> session_logs/policy_metrics_S*.jsonl"
  echo "[PolicyLaunch]   watch for [RenderAttribution] lines: recorder vs observation renders/s"
fi

# FACTR: policy processes get ONLY the client-side knobs — service endpoint, alignment
# limits, and how the sim follows the leader. Every torque/hold/hardware setting goes to
# the service process alone, so a policy window can never command the arm directly.
if is_truthy "${FACTR_ACTIVE}"; then
  POLICY_ENV+=(
    INTERVENE_FACTR_ACTIVE=1
    INTERVENE_ROBOT_BACKEND=factr_rpc
    INTERVENE_ROBOT_KEY=factr
    FACTR_SERVICE_HOST="${FACTR_SERVICE_HOST}"
    FACTR_SERVICE_PORT="${FACTR_SERVICE_PORT}"
    FACTR_RPC_TIMEOUT="${FACTR_RPC_TIMEOUT}"
    FACTR_POSITION_TOLERANCE="${FACTR_POSITION_TOLERANCE}"
    FACTR_ALIGNMENT_TIMEOUT="${FACTR_ALIGNMENT_TIMEOUT}"
    FACTR_MAX_VELOCITY="${FACTR_MAX_VELOCITY}"
    FACTR_MUJOCO_ALPHA="${FACTR_MUJOCO_ALPHA}"
    FACTR_MUJOCO_MAX_STEP_RAD="${FACTR_MUJOCO_MAX_STEP_RAD}"
    FACTR_MUJOCO_TAKEOVER_RAMP_SECONDS="${FACTR_MUJOCO_TAKEOVER_RAMP_SECONDS}"
    FACTR_MUJOCO_TAKEOVER_ALPHA_START="${FACTR_MUJOCO_TAKEOVER_ALPHA_START}"
    FACTR_MUJOCO_TAKEOVER_MAX_STEP_RAD="${FACTR_MUJOCO_TAKEOVER_MAX_STEP_RAD}"
    FACTR_MUJOCO_GRIPPER_MAX_STEP="${FACTR_MUJOCO_GRIPPER_MAX_STEP}"
    FACTR_MUJOCO_GRIPPER_MODE="${FACTR_MUJOCO_GRIPPER_MODE}"
  )
fi
if [[ "${N_POLICIES}" -eq 1 ]]; then
  echo "[PolicyLaunch] Starting policy viewer..."
  env "${POLICY_ENV[@]}" "${POLICY_CMD[@]}" $(_policy_instance_port_args 0) &
  policy_pids+=($!)
  policy_logs+=("<console>")
else
  mkdir -p "${UMBRELLA_ROOT}/session_logs" 2>/dev/null || true
  echo "[PolicyLaunch] Starting ${N_POLICIES} INDEPENDENT policy instances (one per window)..."
  # Pre-assign the initial OOD set. Doing it here rather than letting sessions band-fill
  # at startup avoids a confound: startup decisions would serialize in the staggered
  # launch order, so S00/S01 would be the OOD cells on every single run.
  _ood_initial=""
  if [[ "${OOD_MIN_STATES}" -gt 0 ]]; then
    _ood_initial="$(
      OOD_N="${N_POLICIES}" OOD_MIN="${OOD_MIN_STATES}" OOD_SEED="${STUDY_SEED:-}" \
      "${PYTHON_BIN}" -c '
import os, random
n = int(os.environ["OOD_N"]); k = min(int(os.environ["OOD_MIN"]), n)
seed = os.environ.get("OOD_SEED") or None
rng = random.Random(int(seed)) if seed else random.Random()
print(",".join(str(i) for i in sorted(rng.sample(range(n), k))))
'
    )"
    if [[ "${N_POLICIES}" -le "${OOD_MIN_STATES}" ]]; then
      echo "[PolicyLaunch][OOD] N=${N_POLICIES} <= OOD_MIN_STATES=${OOD_MIN_STATES}:" \
           "every session will be OOD. Set OOD_MIN_STATES=0 for a single-window debug run."
    fi
    echo "[PolicyLaunch] OOD states: min=${OOD_MIN_STATES} max=${OOD_MAX_STATES}" \
         "scene_attempts=${OOD_MAX_SCENE_ATTEMPTS}" \
         "initial=[${_ood_initial}] ledger=/tmp/iilar_ood_${OOD_LEDGER_KEY}.json"
  fi
  for _i in $(seq 0 $((N_POLICIES - 1))); do
    _plog="${UMBRELLA_ROOT}/session_logs/policy_S$(printf '%02d' "${_i}")_${_policy_ts}.log"
    _ood_this=0
    if [[ ",${_ood_initial}," == *",${_i},"* ]]; then _ood_this=1; fi
    echo "[PolicyLaunch]   policy S$(printf '%02d' "${_i}"): cmd=$((POLICY_CMD_PORT + _i * POLICY_PORT_STEP)) state=$((POLICY_STATE_PORT + _i * POLICY_PORT_STEP)) ood=${_ood_this} log → ${_plog}"
    env "${POLICY_ENV[@]}" INTERVENE_OOD_INITIAL_STATE="${_ood_this}" \
        "${POLICY_CMD[@]}" $(_policy_instance_port_args "${_i}") >"${_plog}" 2>&1 &
    policy_pids+=($!)
    policy_logs+=("${_plog}")
    # Stagger checkpoint loads so N CUDA initializations don't spike VRAM simultaneously.
    sleep "${POLICY_LAUNCH_STAGGER:-1}"
  done
  echo "[PolicyLaunch] Follow a policy log with: tail -f ${UMBRELLA_ROOT}/session_logs/policy_S00_${_policy_ts}.log"
fi
policy_pid="${policy_pids[0]}"

sleep "${POLICY_STARTUP_DELAY:-2}"

for _i in "${!policy_pids[@]}"; do
  if ! is_running "${policy_pids[$_i]}"; then
    set +e
    wait "${policy_pids[$_i]}"
    exit_status=$?
    set -e
    echo "[PolicyLaunch] Policy instance ${_i} exited during startup with status ${exit_status} (log: ${policy_logs[$_i]}); aborting."
    cleanup_children
    exit "${exit_status}"
  fi
done

if [[ -n "${WINDOWS:-}" ]] && is_truthy "${START_GRID_UI}"; then
  GRID_ARGS=(
    --xml "${XML}"
    --sessions "${N_POLICIES}"
    --state_host "${POLICY_STATE_HOST}"
    --state_base_port "${POLICY_STATE_PORT}"
    --cmd_host "${POLICY_CMD_BIND}"
    --cmd_base_port "${POLICY_CMD_PORT}"
    --port_step "${POLICY_PORT_STEP}"
    --width "${GRID_UI_WIDTH:-1600}"
    --height "${GRID_UI_HEIGHT:-900}"
    --cam "${GRID_UI_CAM:-front}"
    --detail_cameras "${GRID_UI_DETAIL_CAMS:-front,left,right,wrist}"
    --detail_aspect "${GRID_UI_DETAIL_ASPECT:-1.3333333}"
    --stale_timeout_s "${POLICY_STATE_TIMEOUT_S}"
    --variant_render "${GRID_UI_VARIANT_RENDER}"
    --variant_cache "${OOD_VARIANT_CACHE}"
  )
  if is_truthy "${GRID_UI_FLIP_FRONT:-1}"; then
    GRID_ARGS+=(--flip_front)
  else
    GRID_ARGS+=(--no_flip_front)
  fi
  # Default 0: the scene XMLs author left/right upright (up=+Z), so rolling them showed
  # the detail panes upside down. Only `front` is authored rolled and has its own flag
  # above. Set 1 for a scene whose side cameras ARE authored rolled.
  if is_truthy "${GRID_UI_ROLL_SIDE_CAMERAS:-0}"; then
    GRID_ARGS+=(--roll_side_cameras)
  else
    GRID_ARGS+=(--no_roll_side_cameras)
  fi
  if is_truthy "${START_VR}"; then
    GRID_ARGS+=(--observe_only)
    echo "[PolicyLaunch] Starting passive policy grid UI..."
  else
    echo "[PolicyLaunch] Starting interactive policy grid UI..."
  fi
  "${PYTHON_BIN}" "${INTERVENE_ROOT}/policy_grid_viewer.py" "${GRID_ARGS[@]}" &
  grid_pid=$!
  sleep "${GRID_UI_STARTUP_DELAY:-1}"
  if ! is_running "${grid_pid}"; then
    set +e
    wait "${grid_pid}"
    grid_status=$?
    set -e
    if is_truthy "${START_VR}"; then
      echo "[PolicyLaunch] Passive policy grid UI exited during startup with status ${grid_status}; policy+VR continuing."
      grid_pid=""
    else
      echo "[PolicyLaunch] Interactive policy grid UI exited during startup with status ${grid_status}; aborting."
      cleanup_children
      exit "${grid_status}"
    fi
  fi
fi

mc_restarts=0
mc_log=""
start_mc_sidecar() {
  _mc_robot_key="${INTERVENE_ROBOT_KEY:-${ROBOT_KEY:-p3}}"
  mkdir -p "${UMBRELLA_ROOT}/session_logs" 2>/dev/null || true
  mc_log="${UMBRELLA_ROOT}/session_logs/mq3_mc_$(date +%Y%m%d_%H%M%S).log"
  echo "[PolicyLaunch] MC_ACTIVE=1 — starting mq3_mc.py (robot_key=${_mc_robot_key}, cmd_port=${POLICY_CMD_PORT}); log → ${mc_log}"
  # mq3_mc.py lives at the intervene_base root (INTERVENE_ROOT), not utils/ (SCRIPT_DIR).
  "${PYTHON_BIN}" "${INTERVENE_ROOT}/mq3_mc.py" \
    --robot_key   "${_mc_robot_key}" \
    --cmd_host    "${POLICY_CMD_BIND}" \
    --cmd_port    "${POLICY_CMD_PORT}" >"${mc_log}" 2>&1 &
  mc_pid=$!
  echo "[PolicyLaunch] mq3_mc.py pid=${mc_pid}"
}

if is_truthy "${MC_ACTIVE}" && ! is_truthy "${MC_SIM}"; then
  start_mc_sidecar
elif is_truthy "${MC_ACTIVE}"; then
  echo "[PolicyLaunch] MC_SIM=1: skipping mq3_mc.py sidecar — sim-native IK drives the arm in-process."
fi

if is_truthy "${START_VR}"; then
  echo "[PolicyLaunch] Starting VR control shell via ${VR_RUNNER}..."
  # set -m gives the background subshell its own process group (PGID = vr_pid).
  # On cleanup we can then kill the whole group to reach launcher children (session
  # runtimes) that would otherwise become port-holding orphans after Ctrl+C.
  set -m
  (
    cd "${UMBRELLA_ROOT}"
    if [[ -n "${VR_PYTHON_BIN:-}" ]]; then
      env "${VR_ENV[@]}" PYTHON_BIN="${VR_PYTHON_BIN}" bash "${VR_RUNNER}"
    else
      env -u PYTHON_BIN "${VR_ENV[@]}" bash "${VR_RUNNER}"
    fi
  ) &
  vr_pid=$!
  set +m
fi

exit_status=0
while true; do
  sleep 1
  _policy_died=""
  for _i in "${!policy_pids[@]}"; do
    if ! is_running "${policy_pids[$_i]}"; then
      _policy_died="${_i}"
      break
    fi
  done
  if [[ -n "${_policy_died}" ]]; then
    set +e
    wait "${policy_pids[$_policy_died]}"
    exit_status=$?
    set -e
    echo "[PolicyLaunch] Policy instance ${_policy_died} exited with status ${exit_status} (log: ${policy_logs[$_policy_died]:-<console>}); stopping everything."
    cleanup_children
    break
  fi

  if is_truthy "${START_VR}" && ! is_running "${vr_pid}"; then
    set +e
    wait "${vr_pid}"
    exit_status=$?
    set -e
    echo "[PolicyLaunch] VR process exited with status ${exit_status}; stopping policy."
    cleanup_children
    break
  fi

  if [[ -n "${grid_pid}" ]] && ! is_running "${grid_pid}"; then
    set +e
    wait "${grid_pid}"
    grid_status=$?
    set -e
    if is_truthy "${START_VR}"; then
      echo "[PolicyLaunch] Passive policy grid UI exited with status ${grid_status}; policy+VR continuing."
      grid_pid=""
    else
      exit_status="${grid_status}"
      echo "[PolicyLaunch] Interactive policy grid UI exited with status ${grid_status}; stopping policies."
      cleanup_children
      break
    fi
  fi

  # mq3_mc.py exit is non-fatal for policy+VR, but auto-restart it (bounded) so a transient
  # crash doesn't silently kill robot teleop. Exit details are in ${mc_log}.
  if is_truthy "${MC_ACTIVE}" && ! is_truthy "${MC_SIM}" && [[ -n "${mc_pid}" ]] && ! is_running "${mc_pid}"; then
    set +e; wait "${mc_pid}"; mc_status=$?; set -e
    if [[ "${mc_restarts}" -lt "${MC_MAX_RESTARTS}" ]]; then
      mc_restarts=$((mc_restarts + 1))
      echo "[PolicyLaunch] mq3_mc.py exited (status ${mc_status}); restart ${mc_restarts}/${MC_MAX_RESTARTS} — see ${mc_log}"
      sleep 1
      start_mc_sidecar
    else
      echo "[PolicyLaunch] mq3_mc.py exited (status ${mc_status}); restart limit ${MC_MAX_RESTARTS} reached — robot teleop lost; policy+VR continuing. See ${mc_log}"
      mc_pid=""
    fi
  fi

  # FACTR service death IS fatal, unlike mq3_mc.py: it owns the serial device, so without
  # it every subsequent intervention would fail at claim time with the arm in an unknown
  # state. Not auto-restarted — re-initialization is a physical move that must not happen
  # unattended.
  if is_truthy "${FACTR_ACTIVE}" && [[ -n "${factr_pid}" ]] && ! is_running "${factr_pid}"; then
    set +e; wait "${factr_pid}"; factr_status=$?; set -e
    factr_pid=""
    exit_status="${factr_status}"
    echo "[PolicyLaunch] FACTR service exited with status ${factr_status}; stopping policies safely. Log: ${factr_log}"
    cleanup_children
    break
  fi
done

# --- Study wrap-up: validate, and back up if a destination was given -----------
# Runs on every exit path (normal or Ctrl+C), so a session is never left unverified.
if [[ -n "${STUDY_PARTICIPANT}" ]]; then
  STUDY_ROOT="${INTERVENE_EPISODE_DIR:-${INTERVENE_OUTPUT_DIR:-INTERVENTION_DATA}}"
  STUDY_PARTICIPANT_DIR="${INTERVENE_ROOT}/${STUDY_ROOT}/${STUDY_PARTICIPANT}"
  if [[ "${STUDY_ROOT}" = /* ]]; then
    STUDY_PARTICIPANT_DIR="${STUDY_ROOT}/${STUDY_PARTICIPANT}"
  fi
  # The session just written is the newest directory under the participant.
  STUDY_SESSION_DIR="$(ls -1dt "${STUDY_PARTICIPANT_DIR}"/*/ 2>/dev/null | head -1 || true)"
  if [[ -n "${STUDY_SESSION_DIR}" && -f "${STUDY_SESSION_DIR}/session.json" ]]; then
    echo "[PolicyLaunch] STUDY: validating ${STUDY_SESSION_DIR}"
    VALIDATE_ARGS=("${STUDY_SESSION_DIR}" --quiet)
    if [[ -n "${STUDY_BACKUP_DIR}" ]]; then
      VALIDATE_ARGS+=(--backup "${STUDY_BACKUP_DIR}")
    else
      echo "[PolicyLaunch] STUDY: no STUDY_BACKUP_DIR set — skipping backup. Back this participant up before the next one."
    fi
    ( cd "${INTERVENE_ROOT}" && "${PYTHON_BIN}" utils/validate_study_data.py "${VALIDATE_ARGS[@]}" ) || {
      echo "[PolicyLaunch][WARN] STUDY: data validation reported problems — read ${STUDY_SESSION_DIR}quality_report.md before running the next participant." >&2
    }
  else
    echo "[PolicyLaunch][WARN] STUDY: no session directory found under ${STUDY_PARTICIPANT_DIR}" >&2
  fi
fi

exit "${exit_status}"
