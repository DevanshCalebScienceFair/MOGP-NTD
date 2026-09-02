#!/bin/bash
# Closed-loop asymmetric campaign, at MATCHED DOCKING BUDGET (see CLOSED_LOOP_DESIGN.md).
#
# The comparison is NOT at equal molecule count -- that would measure the scoring
# rule, not the method, because hypervolume silently drops any molecule missing an
# objective and the asymmetric arm would lose by construction. We fix the number of
# DOCK CALLS instead, which is what a lab actually pays.
#
# --acquisition-pool-size 2000 matches every other ablation on this project.
# Leaving it unset scores the whole ~26,600-molecule library every iteration:
# measured at 98.2% of wall clock (178 of 181 min), making a run ~6x slower AND
# incomparable to every existing result. Docking was 0.2%.
#
# BOTH arms use --model hadamard, so the ONLY difference is --hdhfr-fraction.
# Running FULL on the Kronecker ICM instead would confound the design change
# (complete vs partial labels) with a model change.
#
#   FULL  F=1.00 : 40 + 50*5 = 290 molecules, 2.00 calls each = 580 calls
#   ASYM  F=0.25 : 40 + 85*5 = 465 molecules, 1.25 calls each = 581 calls
#
# So the asymmetric arm reaches 60% MORE distinct molecules for the same spend.
# That is the trade being tested.
set -u
cd /Users/devansh/mogp-main-vscode/MOGP-NTD
export PATH=/opt/anaconda3/envs/mogp-drug/bin:$PATH
export PYTHONUNBUFFERED=1
PY=/opt/anaconda3/envs/mogp-drug/bin/python
ROOT=asym_campaign
LIMIT_GB=20
SEEDS="${SEEDS:-0 1}"
mkdir -p "$ROOT/logs"

run_one () {   # arm model frac iters seed
  local ARM=$1 MODEL=$2 FRAC=$3 ITERS=$4 SEED=$5
  local OUT="$ROOT/${ARM}_seed${SEED}" LOG="$ROOT/logs/${ARM}_seed${SEED}.log"
  if [ -f "$OUT/history.csv" ] && [ "$(wc -l < "$OUT/history.csv")" -ge $((ITERS+1)) ]; then
    echo "[skip] $OUT already complete"; return 0
  fi
  echo "=== $(date '+%H:%M:%S')  START $ARM seed=$SEED  (model=$MODEL frac=$FRAC iters=$ITERS)"
  mkdir -p "$OUT"
  cat > "$OUT/run_config.json" <<JSON
{"arm":"$ARM","model":"$MODEL","hdhfr_fraction":$FRAC,"seed":$SEED,
 "n_init":40,"batch_size":5,"n_iterations":$ITERS,
 "posterior_mode":"joint","partitioning_alpha":0.001,
 "budget_dock_calls":580,"note":"matched docking budget, not matched molecule count"}
JSON
  $PY loop.py --model "$MODEL" --hdhfr-fraction "$FRAC" \
       --posterior joint --acquisition-alpha 1e-3 \
       --n-init 40 --batch-size 5 --n-iterations "$ITERS" \
       --acquisition-pool-size 2000 --seed "$SEED" \
       --output-dir "$OUT" > "$LOG" 2>&1 &
  local PID=$!

  # verification gate: the arm must actually be running the model+fraction asked for
  for _ in $(seq 1 90); do grep -q "Running BO loop" "$LOG" 2>/dev/null && break
    kill -0 $PID 2>/dev/null || break; sleep 2; done
  if ! grep -q "'$MODEL' GP model" "$LOG" 2>/dev/null; then
    echo "!!! ABORT: log does not confirm model=$MODEL"; tail -3 "$LOG"; kill -9 $PID 2>/dev/null; exit 1
  fi
  # The seed and fraction MUST be echoed back. loop.py had no --seed flag until
  # 2026-09-01, so a sweep silently produced N identical runs; only comparing two
  # seeds' outputs byte-for-byte caught it. Gate on the resolved values.
  for _ in $(seq 1 30); do grep -q "^  seed=" "$LOG" 2>/dev/null && break
    kill -0 $PID 2>/dev/null || break; sleep 2; done
  if ! grep -q "seed=$SEED  hdhfr_fraction=$FRAC" "$LOG" 2>/dev/null; then
    echo "!!! ABORT: expected seed=$SEED hdhfr_fraction=$FRAC; log says: $(grep -m1 '^  seed=' "$LOG")"
    kill -9 $PID 2>/dev/null; exit 1
  fi
  echo "    verified: $(grep -m1 '^  seed=' "$LOG" | sed 's/^  //')"
  if [ "$FRAC" != "1.0" ] && ! grep -q "partly labelled" <(tail -c 200000 "$LOG") 2>/dev/null; then
    : # the phrase only appears from iteration 1 onward; checked again after the run
  fi

  while kill -0 $PID 2>/dev/null; do
    RSS_KB=$(ps -o rss= -p $PID 2>/dev/null | tr -d ' ')
    if [ -n "${RSS_KB:-}" ] && [ "$RSS_KB" -gt $((LIMIT_GB*1024*1024)) ]; then
      echo "!!! WATCHDOG: $ARM seed=$SEED hit $((RSS_KB/1048576))GB -- killing"
      kill -9 $PID 2>/dev/null; echo WATCHDOG_KILLED >> "$LOG"; break
    fi
    sleep 20
  done
  wait $PID 2>/dev/null
  # post-hoc check: the asymmetric arm MUST have trained on partly-labelled rows,
  # otherwise it silently degenerated into a smaller symmetric arm.
  if [ "$FRAC" != "1.0" ]; then
    local N=$(grep -c "partly labelled" "$LOG" 2>/dev/null || echo 0)
    echo "    partly-labelled training iterations: $N (must be > 0)"
  fi
  echo "=== $(date '+%H:%M:%S')  END   $ARM seed=$SEED"
  grep -E "Final hypervolume|Total wall-clock|Traceback" "$LOG" | tail -2
}

for SEED in $SEEDS; do
  run_one full hadamard 1.0 50 "$SEED"
  run_one asym hadamard 0.25 85 "$SEED"
done
echo "=== ALL DONE $(date '+%H:%M:%S') ==="
