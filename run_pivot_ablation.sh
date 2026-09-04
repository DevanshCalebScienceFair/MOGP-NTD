#!/bin/bash
# ARM B of the pivot attribution chain: THE PIVOT ALONE, at the baseline's draw.
#
# run_pivot_arm.sh changes TWO things at once versus the published baseline
# (ADMET-as-constraints AND uncapping). If it wins we could not say which change
# won, and the judge pitch credits the two separately. This arm supplies the
# missing middle so each change gets its own number.
#
#   A  model_comparison/hadamard_seed*   2,000 draw, 5 objectives   (already run)
#        |  --admet-constraints                    <- THE PIVOT
#   B  pivot_ablation/ablate_seed*        2,000 draw, 2 objectives   (this file)
#        |  drop --acquisition-pool-size           <- THE UNCAP
#   D  pivot_arm/pivot_seed*              full library, 2 objectives
#
# A and B differ in EXACTLY one flag, as do B and D. Same model (hadamard), same
# posterior (joint), same alpha (1e-3), same n_init/batch/iters, same seeds 0-5.
#
# ON THE POOL SIZE, deliberately: loop.py applies the cap BEFORE the ADMET filter
# (the subsample_candidates call precedes the passes_admet block), so this arm
# draws 2,000 and then scores only the ~19.5% that clear the safety bar, i.e.
# ~390. That is NOT a confound to correct -- removing unsafe molecules is the
# treatment, and the shrunken pool is its mechanism, not a nuisance. What is held
# fixed between A and B is the DRAW (2,000 molecules, same seed, same iteration
# reseeding), which is the thing a fair ablation must hold fixed.
#
# ~20 min/run measured on arm D; expect this arm to be FASTER (smaller pool).
#
# Usage:  ./run_pivot_ablation.sh
#         SEEDS="0 1 2" ./run_pivot_ablation.sh
set -u
cd /Users/devansh/mogp-main-vscode/MOGP-NTD
export PATH=/opt/anaconda3/envs/mogp-drug/bin:$PATH
export PYTHONUNBUFFERED=1
PY=/opt/anaconda3/envs/mogp-drug/bin/python
ROOT=pivot_ablation
LIMIT_GB=20
ITERS=50
POOL=2000
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
  OUT="$ROOT/ablate_seed${SEED}"; LOG="$ROOT/logs/ablate_seed${SEED}.log"
  if [ -f "$OUT/history.csv" ] && [ "$(wc -l < "$OUT/history.csv")" -ge $((ITERS+1)) ]; then
    echo "[skip] $OUT already complete"; continue
  fi
  echo "=== $(date '+%H:%M:%S')  START ablate seed=$SEED  (2 objectives, 2000 draw)"
  mkdir -p "$OUT"
  $PY loop.py --model hadamard --posterior joint --acquisition-alpha 1e-3 \
       --n-init 40 --batch-size 5 --n-iterations "$ITERS" --seed "$SEED" \
       --acquisition-pool-size "$POOL" \
       --admet-constraints --output-dir "$OUT" > "$LOG" 2>&1 &
  PID=$!
  for _ in $(seq 1 90); do grep -q "^  admet_constraints=" "$LOG" 2>/dev/null && break
    kill -0 $PID 2>/dev/null || break; sleep 2; done
  # Gate on BOTH flags. Arm B is defined by having the cap AND the constraints;
  # a run missing either one is arm A or arm D wearing this arm's directory name.
  if ! grep -q "seed=$SEED  hdhfr_fraction=1.0" "$LOG" 2>/dev/null \
     || ! grep -q "admet_constraints=True" "$LOG" 2>/dev/null \
     || ! grep -q "acquisition_pool_size=$POOL" "$LOG" 2>/dev/null; then
    echo "!!! ABORT: header must confirm seed=$SEED, admet_constraints=True, pool=$POOL"
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
  echo "=== $(date '+%H:%M:%S')  END   ablate seed=$SEED"
  grep -E "Final hypervolume|Total wall-clock|Traceback" "$LOG" | tail -2
done
echo "=== ALL DONE $(date '+%H:%M:%S') ==="
