#!/usr/bin/env bash
set -euo pipefail

# Compare multiple cup-policy checkpoints with the same evaluation settings.
#
# Example:
#   bash utils/evaluate_cups_checkpoints.sh
#
# Robustness pass:
#   REPEATS_PER_EPISODE=3 XY_JITTER=0.005 YAW_JITTER_DEG=3 RECORD_VIDEOS=failed \
#     bash utils/evaluate_cups_checkpoints.sh

MODEL_RUN=${MODEL_RUN:-}
if [[ -z "${MODEL_RUN}" ]]; then
  MODEL_RUN=$(
    find outputs -maxdepth 1 -type d \
      \( -name 'cups_act_abs_shift1_wrist_rerender_gripper_fixed_*' -o -name 'cups_act_abs_shift1_wrist_rerender_*' -o -name 'cups_act_abs_shift1_rerender_*' \) \
      -printf '%T@ %p\n' \
      | sort -n \
      | tail -1 \
      | cut -d' ' -f2-
  )
fi

if [[ -z "${MODEL_RUN}" || ! -d "${MODEL_RUN}" ]]; then
  echo "[ERROR] Could not find a cups training run. Set MODEL_RUN=outputs/your_cups_run." >&2
  exit 1
fi

CHECKPOINT_STEPS=${CHECKPOINT_STEPS:-"0200000 0400000 0600000 0800000 1000000"}
BATCH_LABEL=${BATCH_LABEL:-cups_ckpt_sweep_$(date +%Y%m%d_%H%M%S)}

NUM_EPISODES=${NUM_EPISODES:-0}
MAX_STEPS=${MAX_STEPS:-600}
REPEATS_PER_EPISODE=${REPEATS_PER_EPISODE:-1}
XY_JITTER=${XY_JITTER:-0.0}
YAW_JITTER_DEG=${YAW_JITTER_DEG:-0.0}
RECORD_VIDEOS=${RECORD_VIDEOS:-none}

echo "[INFO] Model run: ${MODEL_RUN}"
echo "[INFO] Batch:     ${BATCH_LABEL}"
echo "[INFO] Steps:     ${CHECKPOINT_STEPS}"
echo "[INFO] Episodes:  ${NUM_EPISODES} (0 means all), repeats=${REPEATS_PER_EPISODE}"
echo "[INFO] Jitter:    xy=+/-${XY_JITTER} m, yaw=+/-${YAW_JITTER_DEG} deg"
echo

for STEP in ${CHECKPOINT_STEPS}; do
  CHECKPOINT_DIR="${MODEL_RUN}/checkpoints/${STEP}/pretrained_model"
  if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
    echo "[WARN] Skipping missing checkpoint: ${CHECKPOINT_DIR}" >&2
    continue
  fi

  echo "======================================================================"
  echo "[INFO] Evaluating checkpoint ${STEP}"
  echo "======================================================================"

  MODEL_RUN="${MODEL_RUN}" \
  CHECKPOINT_STEP="${STEP}" \
  NUM_EPISODES="${NUM_EPISODES}" \
  MAX_STEPS="${MAX_STEPS}" \
  REPEATS_PER_EPISODE="${REPEATS_PER_EPISODE}" \
  XY_JITTER="${XY_JITTER}" \
  YAW_JITTER_DEG="${YAW_JITTER_DEG}" \
  RECORD_VIDEOS="${RECORD_VIDEOS}" \
  RUN_LABEL="${BATCH_LABEL}_ckpt_${STEP}" \
    bash utils/evaluate_act_cups_shift1.sh
done

echo
echo "======================================================================"
echo "[SUMMARY] Checkpoint sweep results"
echo "======================================================================"

python - "${MODEL_RUN}" "${BATCH_LABEL}" <<'PY'
import json
import sys
from pathlib import Path

model_run = Path(sys.argv[1])
batch_label = sys.argv[2]
eval_root = model_run / "evals"

rows = []
for summary_path in sorted(eval_root.glob(f"{batch_label}_ckpt_*/summary.json")):
    summary = json.loads(summary_path.read_text())
    name = summary_path.parent.name
    step = name.split("_ckpt_", 1)[-1].split("_", 1)[0]
    rows.append(
        {
            "step": step,
            "success_rate": float(summary.get("success_rate", 0.0)),
            "successes": int(summary.get("successes", 0)),
            "num_episodes": int(summary.get("num_episodes", 0)),
            "mean_success_step": summary.get("mean_success_step"),
            "mean_place_xy": summary.get("mean_final_target_xy_error"),
            "mean_place_z": summary.get("mean_final_target_z_error"),
            "failure_counts": summary.get("failure_counts", {}),
            "path": summary_path.parent,
        }
    )

if not rows:
    print("[WARN] No summary.json files found for this batch.")
    raise SystemExit(0)

rows.sort(key=lambda row: (-row["success_rate"], row["step"]))

print(f"{'step':>8s}  {'success':>12s}  {'mean_success_step':>17s}  {'mean_xy':>8s}  {'mean_z':>8s}  failures")
for row in rows:
    mean_success_step = row["mean_success_step"]
    mean_success_text = "-" if mean_success_step is None else f"{mean_success_step:.1f}"
    mean_xy = row["mean_place_xy"]
    mean_z = row["mean_place_z"]
    mean_xy_text = "-" if mean_xy is None else f"{mean_xy:.3f}"
    mean_z_text = "-" if mean_z is None else f"{mean_z:+.3f}"
    failures = row["failure_counts"]
    print(
        f"{row['step']:>8s}  "
        f"{row['success_rate']:>7.1%} ({row['successes']:>2d}/{row['num_episodes']:<2d})  "
        f"{mean_success_text:>17s}  "
        f"{mean_xy_text:>8s}  "
        f"{mean_z_text:>8s}  "
        f"{failures}"
    )

best = rows[0]
print()
print("[BEST]")
print(f"  step:       {best['step']}")
print(f"  checkpoint: {model_run / 'checkpoints' / best['step'] / 'pretrained_model'}")
print(f"  eval dir:   {best['path']}")
PY
