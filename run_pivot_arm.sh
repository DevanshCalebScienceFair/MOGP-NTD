#!/bin/bash
# THE 5-TO-2 PIVOT, plus the uncapped library search.
#
# Two changes, both structural, both motivated by measurements already in this
# repository rather than by tuning:
#
#  1. --admet-constraints
#     The three ADMET objectives become a hard pass/fail bar and only the two
#     docking objectives are optimized. At 5 objectives 62.8% of evaluated
#     molecules are non-dominated and an exact box decomposition needs 62,433
#     cells; at 2 it is 0.7% and 3. Dropping any ONE ADMET objective shrinks the
#     front by 17-22 points, so they are half that curse. All three are known
#     EXACTLY, so nothing estimated is discarded. The bar passes 5,197 of 26,660
#     molecules (19.5%) -- 18x more than a campaign evaluates.
#
#  2. NO --acquisition-pool-size
#     The 2,000-candidate cap existed for exactly one reason: the compute cost
#     that alpha + dedup cut by 17x. Capped, the search scores 7.5% of the
#     library each round and ignores the rest. It has never been re-tested since
#     the speedup.
#
# FRAME: unlike the hDHFR bound arm, this one needs NO re-scoring. Only the
# ACQUISITION sees two objectives; the reported metric still computes
# hypervolume over all five in the published frame, so history.csv is directly
# comparable to every existing run. Verified on a smoke run (0.0341 both ways).
#
# Uncapped scoring is ~6x slower per iteration, so budget ~2.5-3 h per run.
#
# Usage:  ./run_pivot_arm.sh            # 6 seeds
#         SEEDS="0 1 2" ./run_pivot_arm.sh
set -u
cd /Users/devansh/mogp-main-vscode/MOGP-NTD
export PATH=/opt/anaconda3/envs/mogp-drug/bin:$PATH
export PYTHONUNBUFFERED=1
PY=/opt/anaconda3/envs/mogp-drug/bin/python
ROOT=pivot_arm
LIMIT_GB=20
ITERS=50
SEEDS="${SEEDS:-0 1 2 3 4 5}"
BUDGET_START="${BUDGET_START:-}"; BUDGET_H="${BUDGET_H:-}"
budget_spent () {
  [ -z "$BUDGET_START" ] && return 1; [ -z "$BUDGET_H" ] && return 1
  local e=$(( $(date +%s) - BUDGET_START ))
  awk -v e="$e" -v b="$BUDGET_H" 'BEGIN{exit !(e > b*3600)}'
}
mkdir -p "$ROOT/logs"

for SEED in $SEEDS; do
  if budget_spent; then echo "### BUDGET REACHED — stopping before seed=$SEED."; break; fi
  OUT="$ROOT/pivot_seed${SEED}"; LOG="$ROOT/logs/pivot_seed${SEED}.log"
  if [ -f "$OUT/history.csv" ] && [ "$(wc -l < "$OUT/history.csv")" -ge $((ITERS+1)) ]; then
    echo "[skip] $OUT already complete"; continue
  fi
  echo "=== $(date '+%H:%M:%S')  START pivot seed=$SEED  (2 objectives, FULL library)"
  mkdir -p "$OUT"
  $PY loop.py --model hadamard --posterior joint --acquisition-alpha 1e-3 \
       --n-init 40 --batch-size 5 --n-iterations "$ITERS" --seed "$SEED" \
       --admet-constraints --output-dir "$OUT" > "$LOG" 2>&1 &
  PID=$!
  for _ in $(seq 1 90); do grep -q "^  admet_constraints=" "$LOG" 2>/dev/null && break
    kill -0 $PID 2>/dev/null || break; sleep 2; done
  if ! grep -q "seed=$SEED  hdhfr_fraction=1.0" "$LOG" 2>/dev/null \
     || ! grep -q "admet_constraints=True" "$LOG" 2>/dev/null \
     || ! grep -q "acquisition_pool_size=None" "$LOG" 2>/dev/null; then
    echo "!!! ABORT: header must confirm seed=$SEED, admet_constraints=True, pool=None"
    grep -m1 "^  seed=" "$LOG"; grep -m1 "^  admet_constraints=" "$LOG"
    kill -9 $PID 2>/dev/null; exit 1
  fi
  echo "    verified: $(grep -m1 '^  seed=' "$LOG" | sed 's/^  //')"
  echo "    verified: $(grep -m1 '^  admet_constraints=' "$LOG" | sed 's/^  //')"
  while kill -0 $PID 2>/dev/null; do
    RSS_KB=$(ps -o rss= -p $PID 2>/dev/null | tr -d ' ')
    if [ -n "${RSS_KB:-}" ] && [ "$RSS_KB" -gt $((LIMIT_GB*1024*1024)) ]; then
      echo "!!! WATCHDOG: seed=$SEED hit $((RSS_KB/1048576))GB -- killing"
      kill -9 $PID 2>/dev/null; break
    fi
    sleep 20
  done
  wait $PID 2>/dev/null
  echo "    candidates scored per iteration: $(grep -oE '\([0-9]+ candidates\)' "$LOG" | tail -1)"
  echo "=== $(date '+%H:%M:%S')  END   pivot seed=$SEED"
  grep -E "Final hypervolume|Total wall-clock|Traceback" "$LOG" | tail -2
done
echo "=== ALL DONE $(date '+%H:%M:%S') ==="
echo "Analyse with:  python analysis_scripts/pivot_analysis.py"
