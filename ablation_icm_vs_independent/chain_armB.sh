#!/bin/bash
# Wait for arm A (PID passed in) to exit, then run arm B alone.
# Sequential because two concurrent arms peak at ~14 GB with the front still
# small, and the peak grows with front size -- see logs/LAUNCH.txt.
cd /Users/devansh/mogp-main-vscode/MOGP-NTD
export PATH="/opt/anaconda3/envs/mogp-drug/bin:$PATH"   # ninja -> fast qLogEHVI path
D=ablation_icm_vs_independent
A_PID=$1

while kill -0 "$A_PID" 2>/dev/null; do sleep 60; done
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) arm A exited; starting arm B" >> $D/logs/LAUNCH.txt

python -u $D/run_arm.py --model independent \
    --output-dir $D/armB_independent_seed0 > $D/logs/armB_independent.log 2>&1
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) arm B exited rc=$?" >> $D/logs/LAUNCH.txt
