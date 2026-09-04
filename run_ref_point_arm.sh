#!/bin/bash
# Reference-point arm: does a TIGHTER acquisition reference help?
#
# qNEHVI measures improvement against a reference point. Every published run used
# the all-zeros corner of the normalized cube -- safe and method-independent, but
# it means the dominated region includes a large constant block far below any real
# molecule, so improvement in the region that matters is a small share of the total.
#
#   published : --acquisition-ref-point zeros   (the default)
#   this arm  : --acquisition-ref-point nadir   (just under the worst observed)
#
# UNLIKE the hDHFR bound arm, this does NOT change the metric: the reported
# hypervolume always uses evaluation.FIXED_REFERENCE_POINT. So this arm IS
# directly comparable to asym_campaign/full_seed*, which is the baseline.
#
# There is a second reason to care. partitioning_alpha discards cells whose share
# of TOTAL volume falls below it, so a reference far beneath the data inflates the
# total and makes alpha discard more aggressively. alpha=1e-3 was measured to
# preserve the candidate ranking only weakly (Spearman 0.505, ALPHA_EXPLAINED.md);
# a tighter reference is one lever that might reduce that distortion.
#
# Usage:  ./run_ref_point_arm.sh          # 6 seeds, ~25 min each
set -u
cd /Users/devansh/mogp-main-vscode/MOGP-NTD
export PATH=/opt/anaconda3/envs/mogp-drug/bin:$PATH
export PYTHONUNBUFFERED=1
PY=/opt/anaconda3/envs/mogp-drug/bin/python
ROOT=ref_point_arm
LIMIT_GB=20
ITERS=50
SEEDS="${SEEDS:-0 1 2 3 4 5}"
mkdir -p "$ROOT/logs"

# Optional wall-clock budget, set by run_night2.sh. No NEW seed starts once the
# budget is spent; a run already in flight finishes. Both arms are resumable, so
# whatever is skipped is picked up by re-running.
BUDGET_START="${BUDGET_START:-}"
BUDGET_H="${BUDGET_H:-}"
budget_spent () {
  [ -z "$BUDGET_START" ] && return 1
  [ -z "$BUDGET_H" ] && return 1
  local e=$(( $(date +%s) - BUDGET_START ))
  awk -v e="$e" -v b="$BUDGET_H" 'BEGIN{exit !(e > b*3600)}'
}

for SEED in $SEEDS; do
  if budget_spent; then
    echo "### BUDGET REACHED — stopping before seed=$SEED. Re-run to finish."
    break
  fi
  OUT="$ROOT/nadir_seed${SEED}"; LOG="$ROOT/logs/nadir_seed${SEED}.log"
  if [ -f "$OUT/history.csv" ] && [ "$(wc -l < "$OUT/history.csv")" -ge $((ITERS+1)) ]; then
    echo "[skip] $OUT already complete"; continue
  fi
  echo "=== $(date '+%H:%M:%S')  START nadir seed=$SEED"
  mkdir -p "$OUT"
  $PY loop.py --model hadamard --posterior joint --acquisition-alpha 1e-3 \
       --n-init 40 --batch-size 5 --n-iterations "$ITERS" \
       --acquisition-pool-size 2000 --seed "$SEED" \
       --acquisition-ref-point nadir --output-dir "$OUT" > "$LOG" 2>&1 &
  PID=$!
  for _ in $(seq 1 60); do grep -q "^  acquisition_ref_point=" "$LOG" 2>/dev/null && break
    kill -0 $PID 2>/dev/null || break; sleep 2; done
  if ! grep -q "seed=$SEED  hdhfr_fraction=1.0" "$LOG" 2>/dev/null \
     || ! grep -q "acquisition_ref_point=nadir" "$LOG" 2>/dev/null; then
    echo "!!! ABORT: header does not confirm seed=$SEED and ref_point=nadir"
    grep -m1 "^  seed=" "$LOG"; grep -m1 "^  acquisition_ref_point=" "$LOG"
    kill -9 $PID 2>/dev/null; exit 1
  fi
  echo "    verified: $(grep -m1 '^  seed=' "$LOG" | sed 's/^  //')"
  echo "    verified: $(grep -m1 '^  acquisition_ref_point=' "$LOG" | sed 's/^  //')"
  while kill -0 $PID 2>/dev/null; do
    RSS_KB=$(ps -o rss= -p $PID 2>/dev/null | tr -d ' ')
    if [ -n "${RSS_KB:-}" ] && [ "$RSS_KB" -gt $((LIMIT_GB*1024*1024)) ]; then
      echo "!!! WATCHDOG: seed=$SEED hit $((RSS_KB/1048576))GB -- killing"
      kill -9 $PID 2>/dev/null; break
    fi
    sleep 20
  done
  wait $PID 2>/dev/null
  echo "=== $(date '+%H:%M:%S')  END   nadir seed=$SEED"
  grep -E "Final hypervolume|Total wall-clock|Traceback" "$LOG" | tail -2
done
echo "=== ALL DONE $(date '+%H:%M:%S') ==="
echo "Compare with:  python analysis_scripts/ref_point_analysis.py"
