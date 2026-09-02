#!/bin/bash
# Overnight queue. Runs the two pending experiments STRICTLY SEQUENTIALLY --
# never two arms at once on this machine (memory logging undercounts ~3x and it
# has run out of application memory before).
#
#   ./run_tonight.sh            both, in order   (~7.5 h)
#   ./run_tonight.sh compare    old vs new model only        (~5 h, 12 runs)
#   ./run_tonight.sh hdhfr      hDHFR ceiling arm only       (~2.5 h, 6 runs)
#
# Both are RESUMABLE: anything already finished is skipped, so Ctrl-C and
# restarting loses at most the run in flight.
set -u
cd /Users/devansh/mogp-main-vscode/MOGP-NTD
WHICH="${1:-both}"
STAMP=$(date '+%Y%m%d_%H%M')
LOG=/tmp/tonight_${STAMP}.log
exec > >(tee -a "$LOG") 2>&1

echo "=============================================================="
echo " overnight queue started $(date)"
echo " log: $LOG"
echo "=============================================================="

if [ "$WHICH" = "both" ] || [ "$WHICH" = "compare" ]; then
  echo; echo "### 1/2  OLD vs NEW model, complete data, 6 paired seeds"
  ./run_model_comparison.sh || { echo "!!! model comparison stopped early"; }
fi

if [ "$WHICH" = "both" ] || [ "$WHICH" = "hdhfr" ]; then
  echo; echo "### 2/2  hDHFR ceiling arm (alternative normalization frame)"
  if [ ! -f evaluation_bounds_hdhfr0.json ]; then
    echo "  building the alternative frame first..."
    /opt/anaconda3/envs/mogp-drug/bin/python make_alt_bounds.py || exit 1
  fi
  ./run_hdhfr_bound_arm.sh || { echo "!!! hDHFR arm stopped early"; }
fi

echo; echo "=============================================================="
echo " finished $(date)"
echo "=============================================================="
echo
echo "ANALYSE:"
echo "  python analysis_scripts/model_comparison_analysis.py"
echo "  python analysis_scripts/hdhfr_bound_analysis.py"
