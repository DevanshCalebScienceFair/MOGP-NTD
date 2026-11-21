#!/usr/bin/env bash
# One-time setup on NRP. Safe to re-run.
#
#   ./nrp/bootstrap.sh
#
# 1. Uploads data/library/ (63 MB, gitignored generated data) to the volume.
# 2. Submits the job that builds the Linux environment and clones the repo onto
#    the same volume.
#
# No container registry, no token, no GitLab, no Docker. The cell jobs run from
# the stock public mambaorg/micromamba image and read everything off the volume.
set -euo pipefail

NS=lahs-asi
PVC=rung1-campaign
HERE="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> 1/2  uploading data/library to $PVC"
kubectl -n "$NS" delete pod lib-upload --ignore-not-found --wait=true
kubectl -n "$NS" apply -f - <<YAML
apiVersion: v1
kind: Pod
metadata: {name: lib-upload}
spec:
  restartPolicy: Never
  priorityClassName: opportunistic
  containers:
    - name: shell
      image: alpine:3.20
      # Exits on its own: a bare Pod is destroyed at 6 hours and NRP bans
      # accounts for a sleep that never terminates.
      command: ["sh", "-c", "mkdir -p /data/library && sleep 1800"]
      resources:
        requests: {cpu: "1", memory: "2Gi"}
        limits:   {cpu: "1", memory: "2Gi"}
      volumeMounts: [{name: data, mountPath: /data}]
  volumes:
    - name: data
      persistentVolumeClaim: {claimName: $PVC}
YAML
kubectl -n "$NS" wait --for=condition=Ready pod/lib-upload --timeout=300s
for f in smiles.csv admet_scores.csv fingerprints.npy quality_rejected.csv; do
  [ -f "$HERE/data/library/$f" ] || { echo "missing data/library/$f"; exit 1; }
  echo "    $f"
  kubectl -n "$NS" cp "$HERE/data/library/$f" "lib-upload:/data/library/$f"
done
kubectl -n "$NS" exec lib-upload -- ls -la /data/library
kubectl -n "$NS" delete pod lib-upload --wait=false

echo
echo "==> 2/2  building the environment on the volume"
kubectl -n "$NS" delete job rung1-setup-env --ignore-not-found --wait=true
kubectl -n "$NS" apply -f "$HERE/nrp/job-setup-env.yaml"
echo
echo "    watch:  kubectl -n $NS logs -f job/rung1-setup-env"
echo "    then:   ./nrp/launch.sh"
