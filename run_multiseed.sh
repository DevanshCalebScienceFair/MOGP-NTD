#!/bin/bash
# Multi-seed ICM vs independent at the settled acquisition config
# (posterior=joint, alpha=1e-3, pool=2000, rank=1, pool/diversity as seed 0).
#
# Seed 0 already exists for BOTH arms in ablation_joint_alpha/ and is merged at
# analysis time, so seeds 1-9 run here -> 10 seeds total, matching the campaign.
#
# WHY 10 AND NOT 5: a paired Wilcoxon over n=5 seeds has a minimum two-sided p of
# 0.0625 -- it cannot reach p<0.05 even if ICM wins every single seed. n=6 reaches
# 0.0312, n=10 reaches 0.0020. Analysis is valid at any checkpoint; completed runs
# are skipped, so this is safe to stop and resume.
#
# FOOTGUN: run_ablation.py's --seeds takes a COMMA LIST or a COUNT. A bare
# integer is a COUNT: '--seeds 1' means seeds [0], '--seeds 0' means NO seeds.
# A single seed N must be written 'N,' with a trailing comma. This has silently
# wasted two runs. The verification gate below aborts if the resolved seed in
# the log header does not match what was requested.
#
# Runs strictly SEQUENTIALLY, one process per (model, seed), with a hard RSS
# watchdog. One kill loses one run, not the sweep; completed runs are skipped.
set -u
cd /Users/devansh/mogp-main-vscode/MOGP-NTD
export PATH=/opt/anaconda3/envs/mogp-drug/bin:$PATH
export PYTHONUNBUFFERED=1
PY=/opt/anaconda3/envs/mogp-drug/bin/python
ROOT=ablation_multiseed
LIMIT_GB=20
mkdir -p "$ROOT/logs"

SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9}"
for SEED in $SEEDS; do
  for MODEL in coregionalized independent; do
    OUT="$ROOT/${MODEL}_seed${SEED}"
    if [ -f "$OUT/history.csv" ] && [ "$(wc -l < "$OUT/history.csv")" -ge 51 ]; then
      echo "[skip] $OUT already complete"; continue
    fi
    LOG="$ROOT/logs/${MODEL}_seed${SEED}.log"
    echo "=== $(date '+%H:%M:%S')  START $MODEL seed=$SEED  -> $LOG"
    $PY run_ablation.py \
      --seeds "${SEED}," --models "$MODEL" \
      --posterior joint --acquisition-alpha 1e-3 \
      --acquisition-pool-size 2000 \
      --n-init 40 --batch-size 5 --n-iterations 50 \
      --mogp-iters 200 --rank 1 --diversity-threshold 0.7 \
      --library-dir data/library --output-root "$ROOT" \
      > "$LOG" 2>&1 &
    RUN_PID=$!

    # --- verification gate: the header must say seeds=[$SEED] ---
    for _ in $(seq 1 60); do
      grep -q "^seeds=" "$LOG" 2>/dev/null && break
      kill -0 $RUN_PID 2>/dev/null || break
      sleep 2
    done
    HDR=$(grep -m1 "^seeds=" "$LOG" 2>/dev/null)
    if ! echo "$HDR" | grep -q "seeds=\[${SEED}\]"; then
      echo "!!! ABORT: expected seeds=[$SEED] but header says: ${HDR:-<none>}"
      kill -9 $RUN_PID 2>/dev/null; exit 1
    fi
    echo "    verified: $HDR"

    # --- watchdog: kill the run if its RSS exceeds LIMIT_GB ---
    while kill -0 $RUN_PID 2>/dev/null; do
      RSS_KB=$(ps -o rss= -p $RUN_PID 2>/dev/null | tr -d ' ')
      if [ -n "${RSS_KB:-}" ] && [ "$RSS_KB" -gt $((LIMIT_GB*1024*1024)) ]; then
        echo "!!! $(date '+%H:%M:%S') WATCHDOG: $MODEL seed=$SEED hit $((RSS_KB/1048576))GB > ${LIMIT_GB}GB -- killing"
        kill -9 $RUN_PID 2>/dev/null; echo "WATCHDOG_KILLED" >> "$LOG"; break
      fi
      sleep 20
    done
    wait $RUN_PID 2>/dev/null
    echo "=== $(date '+%H:%M:%S')  END   $MODEL seed=$SEED"
    grep -E "Final hypervolume|Total wall-clock" "$LOG" | tail -2
  done
done
echo "=== ALL DONE $(date '+%H:%M:%S') ==="
