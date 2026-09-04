#!/bin/bash
# Artifact-rejection arm: the fix F19 pointed to.
#
# The hDHFR ceiling arm showed the selectivity axis IS truncated, but that
# un-truncating it mostly bought clashing poses (+0.5 useful molecules, +2.2
# artifacts, worse in 6/6 seeds). This attacks the same defect from the other
# side: keep the published frame, and stop letting non-physical poses define the
# front the acquisition optimizes against.
#
#   baseline : asym_campaign/full_seed*        (artifacts on the front)
#   this arm : --reject-artifacts              (PfDHFR > -7.0 or hDHFR > 0.0
#                                               excluded from the qNEHVI baseline)
#
# The metric is UNCHANGED and the molecules are NOT discarded -- they stay in
# Y_evaluated and hypervolume scores them exactly as before. So unlike the hDHFR
# arm, this one is DIRECTLY comparable to the baseline with no re-scoring.
#
# Usage:  ./run_artifact_rejection_arm.sh          # 6 seeds, ~25 min each
set -u
cd /Users/devansh/mogp-main-vscode/MOGP-NTD
export PATH=/opt/anaconda3/envs/mogp-drug/bin:$PATH
export PYTHONUNBUFFERED=1
PY=/opt/anaconda3/envs/mogp-drug/bin/python
ROOT=artifact_rejection_arm
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
  OUT="$ROOT/reject_seed${SEED}"; LOG="$ROOT/logs/reject_seed${SEED}.log"
  if [ -f "$OUT/history.csv" ] && [ "$(wc -l < "$OUT/history.csv")" -ge $((ITERS+1)) ]; then
    echo "[skip] $OUT already complete"; continue
  fi
  echo "=== $(date '+%H:%M:%S')  START reject seed=$SEED"
  mkdir -p "$OUT"
  $PY loop.py --model hadamard --posterior joint --acquisition-alpha 1e-3 \
       --n-init 40 --batch-size 5 --n-iterations "$ITERS" \
       --acquisition-pool-size 2000 --seed "$SEED" \
       --reject-artifacts --output-dir "$OUT" > "$LOG" 2>&1 &
  PID=$!
  for _ in $(seq 1 60); do grep -q "^  reject_artifacts=" "$LOG" 2>/dev/null && break
    kill -0 $PID 2>/dev/null || break; sleep 2; done
  if ! grep -q "seed=$SEED  hdhfr_fraction=1.0" "$LOG" 2>/dev/null \
     || ! grep -q "reject_artifacts=True" "$LOG" 2>/dev/null; then
    echo "!!! ABORT: header does not confirm seed=$SEED and reject_artifacts=True"
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
  N=$(grep -c "artifacts rejected" "$LOG" 2>/dev/null || echo 0)
  echo "    iterations that rejected at least one artifact: $N"
  echo "=== $(date '+%H:%M:%S')  END   reject seed=$SEED"
  grep -E "Final hypervolume|Total wall-clock|Traceback" "$LOG" | tail -2
done
echo "=== ALL DONE $(date '+%H:%M:%S') ==="
echo "Compare with:  python analysis_scripts/artifact_rejection_analysis.py"
