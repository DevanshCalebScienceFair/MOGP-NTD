#!/usr/bin/env bash
# Launch the full-scale feature matrix, detached.
#
# Exists because the launch command is long enough that pasting it into a
# terminal wraps and splits it: the tail becomes a separate command, so the
# scale flags and the output redirect are silently dropped. Inside a file the
# backslash continuations below are safe.
#
#   bash go.sh
#
# Scale is the repo's Grand Campaign (loop.py:114) — 40 init + 50 x 5 = 290
# molecules per case — with production GP training.

set -euo pipefail

cd "$(dirname "$0")"

PYTHON=/opt/anaconda3/envs/mogp-drug/bin/python

# Put the env's bin/ on PATH. Calling the env's python by absolute path is NOT
# the same as activating the env: the env's bin/ stays off PATH, so torch
# cannot find `ninja` and botorch silently falls back to the pure-Python
# qLogEHVI ("Failed to compile fused qLogEHVI C++ extension") — roughly 3x
# slower on the acquisition hot path, which dominates a full-scale sweep.
export PATH="/opt/anaconda3/envs/mogp-drug/bin:$PATH"

# Archive any previous sweep rather than deleting it. The docking cache lives
# in data/docking_cache/, OUTSIDE matrix_results/, so every dock from previous
# runs carries over and this costs nothing in re-docking.
# Output directory: first positional argument, default matrix_results. Naming a
# fresh directory leaves any previous sweep untouched, which is what a
# corrected-vs-original comparison needs — only the DEFAULT target is archived.
OUTDIR="${1:-matrix_results}"

if [ "$OUTDIR" = "matrix_results" ] && [ -d matrix_results ]; then
  n=1
  while [ -e "matrix_results_run${n}" ]; do n=$((n + 1)); done
  mv matrix_results "matrix_results_run${n}"
  echo "Archived previous sweep -> matrix_results_run${n}"
elif [ -d "$OUTDIR" ]; then
  echo "ERROR: $OUTDIR already exists; refusing to overwrite it." >&2
  echo "       Remove it or choose another name." >&2
  exit 1
fi

mkdir -p "$OUTDIR"

# caffeinate -i keeps the Mac awake for what is a multi-hour run; without it a
# sleep stalls the sweep silently partway through.
nohup caffeinate -i "$PYTHON" -u run_matrix.py \
  --tier full \
  --lib-pull 6000 \
  --n-init 20 \
  --batch-size 5 \
  --n-iterations 10 \
  --mogp-iters 100 \
  --output-root "$OUTDIR" \
  > "$OUTDIR/console.log" 2>&1 &

echo "Started (pid $!)."
echo "Output dir:     $OUTDIR"
echo "Watch it with:  tail -f $OUTDIR/console.log"
