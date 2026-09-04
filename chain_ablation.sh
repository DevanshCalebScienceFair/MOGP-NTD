#!/bin/bash
# Wait for the pivot arm (D) to finish, then run the ablation arm (B).
# Sequential by design: guardrail #4 in CLAUDE.md — never two arms in parallel
# on this 48 GB machine.
set -u
cd /Users/devansh/mogp-main-vscode/MOGP-NTD
while pgrep -f "run_pivot_arm.sh" > /dev/null 2>&1; do sleep 60; done
echo "=== pivot arm finished at $(date '+%H:%M:%S'); starting ablation arm ==="
export BUDGET_START=$(date +%s) BUDGET_H=6
exec ./run_pivot_ablation.sh
