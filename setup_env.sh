#!/usr/bin/env bash
# Build the `mogp-drug` conda env from scratch, reproducibly.
#
#   bash setup_env.sh [env-name]        # default: mogp-drug
#
# Why this exists instead of a plain `pip install -r requirements.txt`:
# the working set of versions CANNOT be produced in a single resolver pass.
# PyTDC 1.1.15 declares numpy<2.0 and rdkit<2024.3.1, but the rest of the
# stack needs numpy 2.x — pip sees that as ResolutionImpossible and refuses.
# The env is therefore built in phases, letting TDC install under its own
# bounds and then lifting numpy/pyarrow above them. TDC is only used to pull
# ChEMBL and the ADMET training sets, and both still work.
#
# Four things here are load-bearing and look arbitrary if you skip the comments:
#   * python 3.11        - scikit-learn 1.9.0 requires >=3.11, and 1.9.0 is
#                          what models/pretrained_admet/ was serialized with.
#   * rdkit from conda   - the cp311 pip wheel is built against NumPy 1.x and
#                          raises "_ARRAY_API not found" under NumPy 2.
#   * ninja in the env   - torch needs it on PATH to compile botorch's fused
#                          qLogEHVI; loop.py aborts at startup without it.
#   * setuptools < 82    - PyTDC imports pkg_resources, removed in 82.0.

set -euo pipefail

ENV_NAME="${1:-mogp-drug}"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not on PATH." >&2
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "==> [1/5] creating '$ENV_NAME' (python 3.11 + docking binaries)"
# vina/openbabel are the docking engine; ninja is for the qLogEHVI kernel.
# rdkit comes from conda-forge here so it is compiled against this env's numpy.
conda create -n "$ENV_NAME" -c conda-forge -y \
  python=3.11 pip vina openbabel ninja rdkit

conda activate "$ENV_NAME"

echo "==> [2/5] installing PyTDC under its own (stale) bounds"
# Installed first and alone: it drags numpy down to 1.26 and pyarrow to 12.0.1.
# Phase 4 lifts both back. Doing it in this order keeps pip's resolver happy.
python -m pip install "PyTDC==1.1.15"

echo "==> [3/5] installing the modelling + docking stack"
python -m pip install \
  "scikit-learn==1.9.0" \
  "joblib==1.5.3" \
  "pandas==2.3.3" \
  "botorch==0.18.1" \
  "scipy==1.17.1" \
  "pytest==9.1.1" \
  "streamlit==1.61.1" \
  "matplotlib==3.11.1" \
  "meeko==0.7.1" \
  "biopython==1.88" \
  "gemmi==0.7.5"
#   gemmi: meeko 0.7.1 imports it at package-import time but never declares it,
#   so `import docking` dies with ModuleNotFoundError without this.

echo "==> [4/5] lifting the pins PyTDC held down"
# numpy 2.x per the repo's own requirement; pyarrow 12.0.1 is NumPy-1-compiled
# and blows up the moment pandas imports it. setuptools is capped below 82
# because PyTDC still imports pkg_resources.
python -m pip install "numpy==2.2.6" "pyarrow==24.0.0" "setuptools==81.0.0"

echo "==> [5/5] pinning the OpenMP workaround into the env"
# Three copies of libomp ship in this env (conda, torch, sklearn). Importing
# torch before sklearn aborts the process with "OMP: Error #15" unless this is
# set. Storing it on the env means every shell that activates it inherits it,
# rather than relying on each entry point to set it at module top.
conda env config vars set -n "$ENV_NAME" KMP_DUPLICATE_LIB_OK=TRUE >/dev/null

conda deactivate
conda activate "$ENV_NAME"

echo
echo "==> verifying"
python - <<'PY'
import importlib.metadata as md, shutil, sys, subprocess
print("  python      ", sys.version.split()[0])
for p in ("scikit-learn", "numpy", "pyarrow", "pandas", "botorch", "torch", "setuptools"):
    print(f"  {p:12s}", md.version(p))
import rdkit
print("  rdkit       ", rdkit.__version__)
for exe in ("vina", "ninja"):
    print(f"  {exe:12s}", shutil.which(exe) or "*** MISSING ***")
# The two failures this env is most likely to regress into, checked directly.
import torch, sklearn                      # OMP: Error #15 if libomp is unguarded
from torch.utils.cpp_extension import verify_ninja_availability
verify_ninja_availability()                 # loop.py aborts at startup if this throws
import joblib, glob, warnings
with warnings.catch_warnings():
    warnings.simplefilter("error")          # InconsistentVersionWarning -> hard error
    for f in glob.glob("models/pretrained_admet/*.joblib"):
        joblib.load(f)
print("  ADMET models load clean under this sklearn")
PY

echo
echo "Done. Use it with:  conda activate $ENV_NAME"
