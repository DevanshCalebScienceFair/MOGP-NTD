#!/usr/bin/env bash
# Launch the correlation + bounds gate for the campaigns that have not run.
#
# A GATE, not a campaign. Each campaign docks the same N-molecule sample of the
# SCREENED library against both of its targets, split across N_CHUNKS pods.
# Aggregate with nrp/pilot_report.py once they finish.
set -euo pipefail
NS=lahs-asi
N=${N:-300}
N_CHUNKS=${N_CHUNKS:-4}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# campaign name -> short name for the k8s object (RFC1123, <= 63 chars)
declare -a PAIRS=(
  "chikungunya:chikv"
  "chikungunya_protease:chikvpro"
  "dengue:denv"
)

for pair in "${PAIRS[@]}"; do
  CAMPAIGN="${pair%%:*}"
  CAMPAIGN_SHORT="${pair##*:}"
  for CHUNK in $(seq 0 $((N_CHUNKS - 1))); do
    export CAMPAIGN CAMPAIGN_SHORT CHUNK N_CHUNKS N
    envsubst '${CAMPAIGN} ${CAMPAIGN_SHORT} ${CHUNK} ${N_CHUNKS} ${N}' \
      < "$HERE/job-pilot.yaml" | kubectl apply -n "$NS" -f -
  done
done

echo
kubectl get jobs -n "$NS" -l app=correlation-pilot
