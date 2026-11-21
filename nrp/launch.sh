#!/usr/bin/env bash
# Render and submit the rung-1 campaign cells.
#
#   ./nrp/launch.sh <namespace> <image> [--dry-run]
#
# 12 cells: 2 arms (mogp = shared ICM, indep = independent per-task GPs) x 6
# seeds. The independent arm is NOT optional: the rung-1 correlation sits above
# F19's floor, but "above the floor" is a prediction, and the baseline arm is
# what turns it into a measurement of how much sharing actually bought.
set -euo pipefail

NS="${1:?usage: launch.sh <namespace> <image> [--dry-run]}"
IMAGE="${2:?usage: launch.sh <namespace> <image> [--dry-run]}"
DRY="${3:-}"

ARMS=(mogp indep)
SEEDS=(0 1 2 3 4 5)

for arm in "${ARMS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    rendered=$(ARM="$arm" SEED="$seed" IMAGE="$IMAGE" \
               envsubst '${ARM} ${SEED} ${IMAGE}' < nrp/job-cell.yaml)
    if [[ "$DRY" == "--dry-run" ]]; then
      echo "--- $arm seed $seed ---"
      echo "$rendered"
    else
      echo "$rendered" | kubectl apply -n "$NS" -f -
    fi
  done
done

[[ "$DRY" == "--dry-run" ]] || {
  echo
  echo "submitted ${#ARMS[@]} x ${#SEEDS[@]} = $(( ${#ARMS[@]} * ${#SEEDS[@]} )) cells to namespace $NS"
  echo "watch:  kubectl get jobs -n $NS -l app=rung1-campaign -w"
  echo "logs:   kubectl logs -n $NS -l arm=mogp --tail=40 -f"
}
