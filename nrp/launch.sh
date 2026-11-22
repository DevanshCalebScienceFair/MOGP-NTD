#!/usr/bin/env bash
# Submit the rung-1 Chagas campaign.  ./nrp/launch.sh [--dry-run] [--only N]
#
# 12 cells: 2 arms x 6 seeds. `mogp` is the shared Hadamard ICM, `indep` is
# independent per-task GPs on the same molecules, budget and seed. The
# independent arm is what makes the claim falsifiable: without it there is no
# way to say afterwards how much sharing bought.
set -euo pipefail

NS=lahs-asi
# A campaign is selected here, not hard-coded in the manifest. PREFIX keeps
# job names unique per campaign so rung 2 cannot collide with rung 1, and
# WORKDIR keeps their results in separate directories on the volume.
CAMPAIGN="${CAMPAIGN:-chagas}"
PREFIX="${PREFIX:-rung1}"
WORKDIR="${WORKDIR:-/data/${PREFIX}}"
export CAMPAIGN PREFIX WORKDIR
HERE="$(cd "$(dirname "$0")/.." && pwd)"
ARMS=(${ARMS_OVERRIDE:-full asym indep})
# SEEDS_OVERRIDE mirrors ARMS_OVERRIDE so a power top-up can add seeds
# 6..9 without editing this file. A bare --only still pins ONE seed.
SEEDS=(${SEEDS_OVERRIDE:-0 1 2 3 4 5})
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
    r=$(ARM="$arm" SEED="$seed" envsubst '${ARM} ${SEED} ${CAMPAIGN} ${PREFIX} ${WORKDIR}' < "$HERE/nrp/job-cell.yaml")
    if [ -n "$DRY" ]; then echo "--- $arm seed $seed ---"; echo "$r"
    else echo "$r" | kubectl apply -n "$NS" -f -; fi
  done
done
[ -n "$DRY" ] || cat <<MSG

submitted $(( ${#ARMS[@]} * ${#SEEDS[@]} )) $CAMPAIGN cells to $NS ($PREFIX, $WORKDIR)
  watch:  kubectl -n $NS get jobs -l app=$PREFIX-campaign -w
  logs:   kubectl -n $NS logs -f -l arm=mogp --tail=40
MSG
