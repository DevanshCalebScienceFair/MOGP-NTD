#!/bin/bash
# Two staged arms, sequential, with a WALL-CLOCK BUDGET.
#
#   1. artifact rejection  (6 runs, ~2.5 h)  -- the fix F19 pointed to
#   2. reference point     (6 runs, ~2.5 h)
#
# Both are metric-preserving, so both are directly comparable to
# asym_campaign/full_seed* with no re-scoring.
#
# DEADLINE: no NEW run is started once BUDGET_H has elapsed. A run already in
# flight is allowed to finish, so the true overrun is at most one run (~25 min).
# Both arms are resumable, so whatever is missed is picked up by re-running.
#
#   ./run_night2.sh            budget 5.5 h (default)
#   BUDGET_H=3 ./run_night2.sh
set -u
cd /Users/devansh/mogp-main-vscode/MOGP-NTD
BUDGET_H="${BUDGET_H:-5.5}"
START=$(date +%s)
STAMP=$(date '+%Y%m%d_%H%M')
LOG=/tmp/night2_${STAMP}.log
exec > >(tee -a "$LOG") 2>&1

deadline_passed () {
  local now elapsed
  now=$(date +%s); elapsed=$(( now - START ))
  awk -v e="$elapsed" -v b="$BUDGET_H" 'BEGIN{exit !(e > b*3600)}'
}

echo "=============================================================="
echo " night queue started $(date)   budget ${BUDGET_H} h"
echo " log: $LOG"
echo "=============================================================="

for STAGE in artifact refpoint; do
  if deadline_passed; then
    echo; echo "### BUDGET REACHED — not starting '$STAGE'. Re-run to finish it."
    break
  fi
  case "$STAGE" in
    artifact)
      echo; echo "### 1/2  artifact rejection (keep clashing poses off the front)"
      SEEDS="${SEEDS:-0 1 2 3 4 5}" BUDGET_START=$START BUDGET_H=$BUDGET_H \
        ./run_artifact_rejection_arm.sh || echo "!!! artifact arm stopped early" ;;
    refpoint)
      echo; echo "### 2/2  acquisition reference point (zeros vs nadir)"
      SEEDS="${SEEDS:-0 1 2 3 4 5}" BUDGET_START=$START BUDGET_H=$BUDGET_H \
        ./run_ref_point_arm.sh || echo "!!! reference-point arm stopped early" ;;
  esac
done

ELAPSED=$(( $(date +%s) - START ))
echo; echo "=============================================================="
printf " finished %s   elapsed %.2f h\n" "$(date)" "$(awk -v e=$ELAPSED 'BEGIN{print e/3600}')"
echo "=============================================================="
echo
echo "ANALYSE:"
echo "  python analysis_scripts/artifact_rejection_analysis.py"
echo "  python analysis_scripts/ref_point_analysis.py"
