#!/usr/bin/env bash
set -euo pipefail

# Configuration-only feature isolation for the recovered nine-session baseline.
# Each phase gets a separate forensic directory and can be rerun independently with
# PHASE=baseline|acc|soak.
#
# The former 'ood' phase (and the OOD half of 'soak') isolated the OOD AUTO-PAUSE
# supervisor, which has been removed. OOD is now an out-of-distribution SCENE chosen per
# session in app.py; it pauses nothing and therefore cannot perturb streaming, so there is
# no longer a feature to isolate. 'soak' is now a long ACC soak. The filename is kept so
# existing references and result directories still resolve.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PHASE="${PHASE:-all}"
WINDOWS="${RECOVERY_WINDOWS:-9}"
DURATION_S="${RECOVERY_DURATION_S:-180}"
OUT_ROOT="${RECOVERY_OUT_ROOT:-$ROOT/session_logs/acc_ood_recovery}"

case "$PHASE" in
  baseline)
    ACC_METHOD=none
    LABEL=baseline
    ;;
  acc)
    ACC_METHOD=chunk_residual
    LABEL=acc_chunk_residual
    ;;
  soak)
    ACC_METHOD=chunk_residual
    LABEL=acc_soak
    DURATION_S="${RECOVERY_SOAK_DURATION_S:-1200}"
    ;;
  all)
    for phase in baseline acc soak; do
      PHASE="$phase" RECOVERY_OUT_ROOT="$OUT_ROOT" "$0"
  done
  exit 0
  ;;
  *)
    echo "Usage: PHASE=baseline|acc|soak|all $0" >&2
    exit 2
    ;;
esac

RUN_ROOT="$OUT_ROOT/$LABEL"
mkdir -p "$RUN_ROOT"
echo "[AccOodRecovery] phase=$LABEL windows=$WINDOWS duration_s=$DURATION_S"
echo "[AccOodRecovery] output=$RUN_ROOT"

FORENSICS_WINDOWS="$WINDOWS" \
FORENSICS_DURATION_S="$DURATION_S" \
FORENSICS_OUT_ROOT="$RUN_ROOT" \
FORENSICS_ACC_METHOD="$ACC_METHOD" \
FORENSICS_ACC_PROBE_ASYNC=1 \
FORENSICS_ACC_PROBE_MIDCHUNK=0 \
FORENSICS_QUEST="${FORENSICS_QUEST:-0}" \
FORENSICS_INTERACTIVE="${FORENSICS_INTERACTIVE:-${FORENSICS_QUEST:-0}}" \
FORENSICS_VALIDATE_DELIVERY="${FORENSICS_VALIDATE_DELIVERY:-${FORENSICS_QUEST:-0}}" \
FORENSICS_RECORD_IMAGES=0 \
  bash "$ROOT/tools/run_policy_mirror_forensics.sh" current-current

echo "[AccOodRecovery] phase complete: $LABEL"
echo "[AccOodRecovery] compare against: $OUT_ROOT/baseline"
