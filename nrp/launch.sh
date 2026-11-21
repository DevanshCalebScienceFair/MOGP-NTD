#!/usr/bin/env bash
# Submit the rung-1 Chagas campaign.  ./nrp/launch.sh [--dry-run] [--only N]
#
# 12 cells: 2 arms x 6 seeds. `mogp` is the shared Hadamard ICM, `indep` is
# independent per-task GPs on the same molecules, budget and seed. The
# independent arm is what makes the claim falsifiable: without it there is no
# way to say afterwards how much sharing bought.
set -euo pipefail

NS=lahs-asi
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ARMS=(full asym indep)
SEEDS=(0 1 2 3 4 5)
DRY=""; ONLY=""
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;;
    --only)    ONLY="$2"; shift ;;
    *) echo "unknown arg $1"; exit 1 ;;
  esac; shift
done
[ -n "$ONLY" ] && SEEDS=("$ONLY")

for arm in "${ARMS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    r=$(ARM="$arm" SEED="$seed" envsubst '${ARM} ${SEED}' < "$HERE/nrp/job-cell.yaml")
    if [ -n "$DRY" ]; then echo "--- $arm seed $seed ---"; echo "$r"
    else echo "$r" | kubectl apply -n "$NS" -f -; fi
  done
done
[ -n "$DRY" ] || cat <<MSG

submitted $(( ${#ARMS[@]} * ${#SEEDS[@]} )) cells to $NS
  watch:  kubectl -n $NS get jobs -l app=rung1-campaign -w
  logs:   kubectl -n $NS logs -f -l arm=mogp --tail=40
MSG
