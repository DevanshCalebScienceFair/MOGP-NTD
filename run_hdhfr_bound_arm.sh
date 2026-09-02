#!/bin/bash
# The hDHFR-ceiling arm: does un-truncating the selectivity axis help the SEARCH?
#
# The shared -5.0 upper bound collapses the most selective molecules onto one
# normalized value, so qNEHVI gets no gradient on the axis the project is about.
# This runs the SAME configuration under a frame whose hDHFR ceiling is 0.0.
#
#   published frame : evaluation_bounds.json          (hDHFR max -5.0)
#   this arm        : evaluation_bounds_hdhfr0.json   (hDHFR max  0.0)
#
# *** HYPERVOLUMES FROM THE TWO FRAMES ARE NOT COMPARABLE. ***
# Changing a bound moves every number for reasons unrelated to the method. Judge
# this arm on frame-INDEPENDENT quantities -- artifact-filtered selectivity of
# the molecules it finds, and how many physical binders it turns up -- or
# re-score both arms in one frame. analysis_scripts/hdhfr_bound_analysis.py does
# the frame-independent comparison and refuses to mix fingerprints.
#
# Usage:  python make_alt_bounds.py && ./run_hdhfr_bound_arm.sh
set -u
cd /Users/devansh/mogp-main-vscode/MOGP-NTD
export PATH=/opt/anaconda3/envs/mogp-drug/bin:$PATH
export PYTHONUNBUFFERED=1
PY=/opt/anaconda3/envs/mogp-drug/bin/python
ROOT=hdhfr_bound_arm
ALT=evaluation_bounds_hdhfr0.json
LIMIT_GB=20
ITERS=50
SEEDS="${SEEDS:-0 1 2 3 4 5}"
[ -f "$ALT" ] || { echo "!!! $ALT missing -- run: python make_alt_bounds.py"; exit 1; }
mkdir -p "$ROOT/logs"

for SEED in $SEEDS; do
  OUT="$ROOT/hdhfr0_seed${SEED}"; LOG="$ROOT/logs/hdhfr0_seed${SEED}.log"
  if [ -f "$OUT/history.csv" ] && [ "$(wc -l < "$OUT/history.csv")" -ge $((ITERS+1)) ]; then
    echo "[skip] $OUT already complete"; continue
  fi
  echo "=== $(date '+%H:%M:%S')  START hdhfr0 seed=$SEED"
  mkdir -p "$OUT"
  $PY loop.py --model hadamard --posterior joint --acquisition-alpha 1e-3 \
       --n-init 40 --batch-size 5 --n-iterations "$ITERS" \
       --acquisition-pool-size 2000 --seed "$SEED" \
       --bounds-path "$ALT" --output-dir "$OUT" > "$LOG" 2>&1 &
  PID=$!
  for _ in $(seq 1 60); do grep -q "^  bounds_frame=" "$LOG" 2>/dev/null && break
    kill -0 $PID 2>/dev/null || break; sleep 2; done
  if ! grep -q "seed=$SEED  hdhfr_fraction=1.0" "$LOG" 2>/dev/null \
     || ! grep -q "bounds_frame=$ALT" "$LOG" 2>/dev/null; then
    echo "!!! ABORT: header does not confirm seed=$SEED and the ALTERNATIVE frame"
    grep -m1 "^  seed=" "$LOG"; grep -m1 "^  bounds_frame=" "$LOG"
    kill -9 $PID 2>/dev/null; exit 1
  fi
  echo "    verified: $(grep -m1 '^  seed=' "$LOG" | sed 's/^  //')"
  echo "    verified: $(grep -m1 '^  bounds_frame=' "$LOG" | sed 's/^  //')"
  while kill -0 $PID 2>/dev/null; do
    RSS_KB=$(ps -o rss= -p $PID 2>/dev/null | tr -d ' ')
    if [ -n "${RSS_KB:-}" ] && [ "$RSS_KB" -gt $((LIMIT_GB*1024*1024)) ]; then
      echo "!!! WATCHDOG: seed=$SEED hit $((RSS_KB/1048576))GB -- killing"
      kill -9 $PID 2>/dev/null; break
    fi
    sleep 20
  done
  wait $PID 2>/dev/null
  echo "=== $(date '+%H:%M:%S')  END   hdhfr0 seed=$SEED"
  grep -E "Final hypervolume|Total wall-clock|Traceback" "$LOG" | tail -2
done
echo "=== ALL DONE $(date '+%H:%M:%S') ==="
echo "Compare with:  python analysis_scripts/hdhfr_bound_analysis.py"
