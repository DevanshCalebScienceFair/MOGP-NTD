#!/usr/bin/env bash
# One-time setup on NRP. Run once; it is safe to re-run.
#
#   ./nrp/bootstrap.sh
#
# Does two things:
#   1. Uploads data/library/ (63 MB, gitignored generated data) to the campaign
#      volume, where all 12 cells read it. It is not in the image on purpose:
#      it is generated, not source, and baking it in would mean rebuilding the
#      image every time the library changes.
#   2. Builds the container image IN-CLUSTER with kaniko, straight from the
#      public GitHub repo, so no local Docker and no GitLab project are needed.
set -euo pipefail

NS=lahs-asi
PVC=rung1-campaign
REPO_URL="git://github.com/DevanshCalebScienceFair/MOGP-NTD.git#refs/heads/ExtraNovelPipeline"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> 1/2  uploading data/library to $PVC"
kubectl -n "$NS" delete pod lib-upload --ignore-not-found --wait=true
kubectl -n "$NS" apply -f - <<YAML
apiVersion: v1
kind: Pod
metadata:
  name: lib-upload
spec:
  restartPolicy: Never
  priorityClassName: opportunistic
  containers:
    - name: shell
      image: alpine:3.20
      # A bare Pod is destroyed at 6 hours and NRP bans accounts for a sleep
      # that never ends, so this one exits on its own well inside both rules.
      command: ["sh", "-c", "mkdir -p /data/library && sleep 900"]
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
echo "==> 2/2  building the image in-cluster with kaniko"
echo "    NOTE: set REGISTRY below to a registry you can push to before running."
echo "    NRP GitLab: log in once at https://gitlab.nrp-nautilus.io, create a"
echo "    project, then create a deploy token and store it as a secret:"
echo
echo "      kubectl -n $NS create secret docker-registry gitlab-registry \\"
echo "        --docker-server=gitlab-registry.nrp-nautilus.io \\"
echo "        --docker-username=<token-name> --docker-password=<token>"
echo
echo "    Then re-run with REGISTRY set, e.g."
echo "      REGISTRY=gitlab-registry.nrp-nautilus.io/<user>/mogp-ntd ./nrp/bootstrap.sh"

if [ -z "${REGISTRY:-}" ]; then
  echo
  echo "REGISTRY not set; stopping after the library upload."
  exit 0
fi

kubectl -n "$NS" delete job kaniko-build --ignore-not-found --wait=true
kubectl -n "$NS" apply -f - <<YAML
apiVersion: batch/v1
kind: Job
metadata:
  name: kaniko-build
spec:
  backoffLimit: 1
  template:
    spec:
      restartPolicy: Never
      priorityClassName: opportunistic
      containers:
        - name: kaniko
          image: gcr.io/kaniko-project/executor:latest
          args:
            - --context=$REPO_URL
            - --dockerfile=nrp/Dockerfile
            - --destination=$REGISTRY:rung1-latest
          resources:
            requests: {cpu: "4", memory: "8Gi", ephemeral-storage: "40Gi"}
            limits:   {cpu: "4", memory: "8Gi", ephemeral-storage: "40Gi"}
          volumeMounts:
            - name: creds
              mountPath: /kaniko/.docker
      volumes:
        - name: creds
          secret:
            secretName: gitlab-registry
            items: [{key: .dockerconfigjson, path: config.json}]
YAML
echo "    watch:  kubectl -n $NS logs -f job/kaniko-build"
