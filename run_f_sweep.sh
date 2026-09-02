#!/bin/bash
# F-SWEEP: what fraction of molecules should get the expensive second assay?
#
# F14 tested two points (F=1.00, F=0.25) and found a decline. It did NOT locate an
# optimum. This fills in F=0.75 and F=0.50 at the SAME ~580-dock-call budget, so all
# four points are comparable and an interior optimum would be visible.
#
#   F      molecules  iters  hDHFR labels  calls   arm dir
#   1.00      290       50        290       580    asym_campaign/full_seed*   (done)
#   0.75      330       58        248       578    f_sweep/f075_seed*
#   0.50      385       69        192       578    f_sweep/f050_seed*
#   0.25      465       85        116       581    asym_campaign/asym_seed*   (done)
#
# Everything else is held fixed: Hadamard ICM, joint posterior, alpha=1e-3,
# pool=2000, n_init=40, batch=5, identical initial molecules per seed.
set -u
cd /Users/devansh/mogp-main-vscode/MOGP-NTD
export PATH=/opt/anaconda3/envs/mogp-drug/bin:$PATH
export PYTHONUNBUFFERED=1
PY=/opt/anaconda3/envs/mogp-drug/bin/python
ROOT=f_sweep
LIMIT_GB=20
SEEDS="${SEEDS:-0 1 2 3 4 5}"
mkdir -p "$ROOT/logs"

run_one () {   # tag frac iters seed
  local TAG=$1 FRAC=$2 ITERS=$3 SEED=$4
  local OUT="$ROOT/${TAG}_seed${SEED}" LOG="$ROOT/logs/${TAG}_seed${SEED}.log"
  if [ -f "$OUT/history.csv" ] && [ "$(wc -l < "$OUT/history.csv")" -ge $((ITERS+1)) ]; then
    echo "[skip] $OUT already complete"; return 0
  fi
  echo "=== $(date '+%H:%M:%S')  START $TAG seed=$SEED  (frac=$FRAC iters=$ITERS)"
  mkdir -p "$OUT"
  cat > "$OUT/run_config.json" <<JSON
{"arm":"$TAG","model":"hadamard","hdhfr_fraction":$FRAC,"seed":$SEED,
 "n_init":40,"batch_size":5,"n_iterations":$ITERS,
 "posterior_mode":"joint","partitioning_alpha":0.001,"acquisition_pool_size":2000,
 "budget_dock_calls":578,"note":"F-sweep, matched docking budget"}
JSON
  $PY loop.py --model hadamard --hdhfr-fraction "$FRAC" \
       --posterior joint --acquisition-alpha 1e-3 \
       --n-init 40 --batch-size 5 --n-iterations "$ITERS" \
       --acquisition-pool-size 2000 --seed "$SEED" \
       --output-dir "$OUT" > "$LOG" 2>&1 &
  local PID=$!

  for _ in $(seq 1 60); do grep -q "^  seed=" "$LOG" 2>/dev/null && break
    kill -0 $PID 2>/dev/null || break; sleep 2; done
  # Compare against Python's OWN float repr: the shell says 0.50, argparse echoes
  # 0.5, and a literal string match aborts a perfectly correct run.
  local FRAC_REPR; FRAC_REPR=$($PY -c "print(float('$FRAC'))")
  if ! grep -q "seed=$SEED  hdhfr_fraction=$FRAC_REPR" "$LOG" 2>/dev/null; then
    echo "!!! ABORT: expected seed=$SEED hdhfr_fraction=$FRAC_REPR; log says: $(grep -m1 '^  seed=' "$LOG")"
    kill -9 $PID 2>/dev/null; exit 1
  fi
  echo "    verified: $(grep -m1 '^  seed=' "$LOG" | sed 's/^  //')"

  while kill -0 $PID 2>/dev/null; do
    RSS_KB=$(ps -o rss= -p $PID 2>/dev/null | tr -d ' ')
    if [ -n "${RSS_KB:-}" ] && [ "$RSS_KB" -gt $((LIMIT_GB*1024*1024)) ]; then
      echo "!!! WATCHDOG: $TAG seed=$SEED hit $((RSS_KB/1048576))GB -- killing"
      kill -9 $PID 2>/dev/null; echo WATCHDOG_KILLED >> "$LOG"; break
    fi
    sleep 20
  done
  wait $PID 2>/dev/null
  echo "    partly-labelled training iterations: $(grep -c 'partly labelled' "$LOG" 2>/dev/null || echo 0)"
  echo "=== $(date '+%H:%M:%S')  END   $TAG seed=$SEED"
  grep -E "Final hypervolume|Total wall-clock" "$LOG" | tail -2
}

for SEED in $SEEDS; do
  run_one f075 0.75 58 "$SEED"
  run_one f050 0.50 69 "$SEED"
done
echo "=== ALL DONE $(date '+%H:%M:%S') ==="
