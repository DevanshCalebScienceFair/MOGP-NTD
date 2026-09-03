#!/bin/bash
# Filter the GP TRAINING SET, not the front — the hypothesis F21 produced.
#
# ARTIFACT_REJECTION_RESULT.md: keeping clashing poses off the qNEHVI baseline
# engaged (9.6% of the front removed) and changed nothing; the rejecting arm even
# evaluated slightly MORE artifacts. Artifacts do not enter through the front,
# they enter through the MODEL. On seed 0, 31 of 290 evaluated molecules were
# artifacts with mean apparent selectivity +0.68 against +0.13 for physical ones,
# so the GP is fitted on labels saying "extremely selective", learns the
# fragments that produce them, and keeps proposing more.
#
#   baseline : asym_campaign/full_seed*
#   this arm : --reject-artifacts-training
#
# KNOWN RISK, stated in advance so the result cannot be spun: dropping those rows
# leaves the GP with NO data in that region and therefore high posterior variance
# there, which qNEHVI may read as worth exploring. This arm could make artifact
# chasing WORSE. That is the measurement.
#
# Metric unchanged, molecules not discarded -> directly comparable, no re-scoring.
#
# Usage:  ./run_artifact_training_arm.sh          # 6 seeds, ~25 min each
set -u
cd /Users/devansh/mogp-main-vscode/MOGP-NTD
export PATH=/opt/anaconda3/envs/mogp-drug/bin:$PATH
export PYTHONUNBUFFERED=1
PY=/opt/anaconda3/envs/mogp-drug/bin/python
ROOT=artifact_training_arm
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
  OUT="$ROOT/trainfilter_seed${SEED}"; LOG="$ROOT/logs/trainfilter_seed${SEED}.log"
  if [ -f "$OUT/history.csv" ] && [ "$(wc -l < "$OUT/history.csv")" -ge $((ITERS+1)) ]; then
    echo "[skip] $OUT already complete"; continue
  fi
  echo "=== $(date '+%H:%M:%S')  START trainfilter seed=$SEED"
  mkdir -p "$OUT"
  $PY loop.py --model hadamard --posterior joint --acquisition-alpha 1e-3 \
       --n-init 40 --batch-size 5 --n-iterations "$ITERS" \
       --acquisition-pool-size 2000 --seed "$SEED" \
       --reject-artifacts-training --output-dir "$OUT" > "$LOG" 2>&1 &
  PID=$!
  for _ in $(seq 1 60); do grep -q "^  reject_artifacts=" "$LOG" 2>/dev/null && break
    kill -0 $PID 2>/dev/null || break; sleep 2; done
  if ! grep -q "seed=$SEED  hdhfr_fraction=1.0" "$LOG" 2>/dev/null \
     || ! grep -q "reject_artifacts_training=True" "$LOG" 2>/dev/null; then
    echo "!!! ABORT: header does not confirm seed=$SEED and reject_artifacts_training=True"
    grep -m1 "^  seed=" "$LOG"; grep -m1 "^  reject_artifacts=" "$LOG"
    kill -9 $PID 2>/dev/null; exit 1
  fi
  echo "    verified: $(grep -m1 '^  seed=' "$LOG" | sed 's/^  //')"
  echo "    verified: $(grep -m1 '^  reject_artifacts=' "$LOG" | sed 's/^  //')"
  while kill -0 $PID 2>/dev/null; do
    RSS_KB=$(ps -o rss= -p $PID 2>/dev/null | tr -d ' ')
    if [ -n "${RSS_KB:-}" ] && [ "$RSS_KB" -gt $((LIMIT_GB*1024*1024)) ]; then
      echo "!!! WATCHDOG: seed=$SEED hit $((RSS_KB/1048576))GB -- killing"
      kill -9 $PID 2>/dev/null; break
    fi
    sleep 20
  done
  wait $PID 2>/dev/null
  # the rejection MUST have engaged, or this arm is just the baseline again
  N=$(grep -c "artifacts out of training" "$LOG" 2>/dev/null || echo 0)
  echo "    iterations that filtered training data: $N (must be > 0)"
  echo "=== $(date '+%H:%M:%S')  END   trainfilter seed=$SEED"
  grep -E "Final hypervolume|Total wall-clock|Traceback" "$LOG" | tail -2
done
echo "=== ALL DONE $(date '+%H:%M:%S') ==="
echo "Compare with:  python analysis_scripts/artifact_training_analysis.py"
