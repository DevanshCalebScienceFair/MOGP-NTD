#!/bin/bash
# OLD vs NEW model, head to head, on COMPLETE data.
#
#   old = coregionalized  (Kronecker ICM, mogp_coregionalized.py) -- the model the
#                          whole project was built on. Cannot take missing labels.
#   new = hadamard        (stacked-index ICM, mogp_hadamard.py)   -- same model,
#                          written so gaps are expressible.
#
# Every molecule is docked against both targets here (hdhfr_fraction defaults to
# 1.0), so this asks ONE question: does the rewrite cost anything when the data is
# complete? If they tie, the new model strictly dominates -- same quality, plus it
# handles gaps. If the new one is worse, that is a real price for the flexibility
# and it belongs in the paper.
#
# The two differ deliberately in one place: the Kronecker form has ONE NOISE PER
# TASK (MultitaskGaussianLikelihood); the Hadamard form flattens to a vector so it
# gets ONE SHARED noise. Targets are standardized per task first, which puts both
# on unit variance, but it is a real difference -- state it with any result.
#
# Usage:   ./run_model_comparison.sh              # seeds 0-5
#          SEEDS="0 1 2" ./run_model_comparison.sh
# Analyse: python analysis_scripts/model_comparison_analysis.py
set -u
cd /Users/devansh/mogp-main-vscode/MOGP-NTD
export PATH=/opt/anaconda3/envs/mogp-drug/bin:$PATH
export PYTHONUNBUFFERED=1
PY=/opt/anaconda3/envs/mogp-drug/bin/python
ROOT=model_comparison
LIMIT_GB=20
ITERS=50
SEEDS="${SEEDS:-0 1 2 3 4 5}"
mkdir -p "$ROOT/logs"

run_one () {   # model seed
  local MODEL=$1 SEED=$2
  local OUT="$ROOT/${MODEL}_seed${SEED}" LOG="$ROOT/logs/${MODEL}_seed${SEED}.log"
  if [ -f "$OUT/history.csv" ] && [ "$(wc -l < "$OUT/history.csv")" -ge $((ITERS+1)) ]; then
    echo "[skip] $OUT already complete"; return 0
  fi
  echo "=== $(date '+%H:%M:%S')  START $MODEL seed=$SEED"
  mkdir -p "$OUT"
  cat > "$OUT/run_config.json" <<JSON
{"model":"$MODEL","hdhfr_fraction":1.0,"seed":$SEED,
 "n_init":40,"batch_size":5,"n_iterations":$ITERS,
 "posterior_mode":"joint","partitioning_alpha":0.001,"acquisition_pool_size":2000,
 "note":"old (Kronecker) vs new (Hadamard) ICM on COMPLETE data"}
JSON
  $PY loop.py --model "$MODEL" \
       --posterior joint --acquisition-alpha 1e-3 \
       --n-init 40 --batch-size 5 --n-iterations "$ITERS" \
       --acquisition-pool-size 2000 --seed "$SEED" \
       --output-dir "$OUT" > "$LOG" 2>&1 &
  local PID=$!

  # Gate on the values the run ECHOES, not on the ones we asked for. loop.py had
  # no --seed flag until 2026-09-01 and a whole sweep silently reused seed 42.
  for _ in $(seq 1 60); do grep -q "^  seed=" "$LOG" 2>/dev/null && break
    kill -0 $PID 2>/dev/null || break; sleep 2; done
  if ! grep -q "seed=$SEED  hdhfr_fraction=1.0" "$LOG" 2>/dev/null \
     || ! grep -q "'$MODEL' GP model" "$LOG" 2>/dev/null; then
    echo "!!! ABORT: header does not confirm model=$MODEL seed=$SEED"
    grep -m1 "^Running BO loop" "$LOG"; grep -m1 "^  seed=" "$LOG"
    kill -9 $PID 2>/dev/null; exit 1
  fi
  echo "    verified: $MODEL · $(grep -m1 '^  seed=' "$LOG" | sed 's/^  //')"

  while kill -0 $PID 2>/dev/null; do
    RSS_KB=$(ps -o rss= -p $PID 2>/dev/null | tr -d ' ')
    if [ -n "${RSS_KB:-}" ] && [ "$RSS_KB" -gt $((LIMIT_GB*1024*1024)) ]; then
      echo "!!! WATCHDOG: $MODEL seed=$SEED hit $((RSS_KB/1048576))GB -- killing"
      kill -9 $PID 2>/dev/null; echo WATCHDOG_KILLED >> "$LOG"; break
    fi
    sleep 20
  done
  wait $PID 2>/dev/null
  echo "=== $(date '+%H:%M:%S')  END   $MODEL seed=$SEED"
  grep -E "Final hypervolume|Total wall-clock|Traceback" "$LOG" | tail -2
}

for SEED in $SEEDS; do
  run_one coregionalized "$SEED"   # OLD
  run_one hadamard       "$SEED"   # NEW
done
echo "=== ALL DONE $(date '+%H:%M:%S') ==="
echo "Now run:  python analysis_scripts/model_comparison_analysis.py"
