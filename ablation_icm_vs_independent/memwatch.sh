#!/bin/bash
# Per-minute peak RSS of whichever arm is running, plus system memory. Tracks by
# process name so it survives arm A ending and arm B starting.
OUT=ablation_icm_vs_independent/logs/memwatch.csv
[ -f $OUT ] || echo "utc,arm,peak_gb,free_pct,swap_used_mb" > $OUT
while pgrep -f run_arm.py > /dev/null || pgrep -f chain_armB.sh > /dev/null; do
  peak=0; arm="none"
  for _ in $(seq 1 12); do
    line=$(ps -Ao rss=,command= | grep "[r]un_arm.py" | sort -rn | head -1)
    r=$(echo "$line" | awk '{print $1}')
    if [ -n "$r" ] && [ "$r" -gt "$peak" ] 2>/dev/null; then
      peak=$r
      arm=$(echo "$line" | grep -o "coregionalized\|independent" | head -1)
    fi
    sleep 5
  done
  free=$(memory_pressure 2>/dev/null | awk '/free percentage/{print $NF}' | tr -d '%')
  swap=$(sysctl -n vm.swapusage | awk '{print $6}' | tr -d 'M')
  printf "%s,%s,%.2f,%s,%s\n" "$(date -u +%H:%M:%S)" "$arm" \
    "$(echo "scale=2;$peak/1048576"|bc)" "$free" "$swap" >> $OUT
done
