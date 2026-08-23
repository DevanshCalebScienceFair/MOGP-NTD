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

echo "==> [1/6] creating '$ENV_NAME' (python 3.11 + docking binaries)"
# vina/openbabel are the docking engine; ninja is for the qLogEHVI kernel.
# rdkit comes from conda-forge here so it is compiled against this env's numpy.
conda create -n "$ENV_NAME" -c conda-forge -y \
  python=3.11 pip vina openbabel ninja rdkit

conda activate "$ENV_NAME"

echo "==> [2/6] installing PyTDC under its own (stale) bounds"
# Installed first and alone: it drags numpy down to 1.26 and pyarrow to 12.0.1.
# Phase 4 lifts both back. Doing it in this order keeps pip's resolver happy.
python -m pip install "PyTDC==1.1.15"

echo "==> [3/6] installing the modelling + docking stack"
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

echo "==> [4/6] lifting the pins PyTDC held down"
# numpy 2.x per the repo's own requirement; pyarrow 12.0.1 is NumPy-1-compiled
# and blows up the moment pandas imports it. setuptools is capped below 82
# because PyTDC still imports pkg_resources.
python -m pip install "numpy==2.2.6" "pyarrow==24.0.0" "setuptools==81.0.0"

echo "==> [5/6] restoring the conda RDKit that PyTDC clobbered"
# PyTDC 1.1.15 pins rdkit<2024.3.1, so phase 2 pip-DOWNGRADED the conda RDKit
# (built against this env's NumPy 2) to a wheel compiled against NumPy 1.x. Once
# phase 4 lifts numpy to 2.x, that wheel's C-extensions die with
# "AttributeError: _ARRAY_API not found" the moment rdkit.Chem / DataStructs
# load. A bare `import rdkit` never touches those, so the breakage is SILENT —
# the old verify block missed it. Restore the conda build (rdkit<2024.3.1 is a
# TDC constraint we deliberately override; TDC is only used to pull datasets).
python -m pip uninstall -y rdkit
conda install --force-reinstall -y -c conda-forge rdkit
# conda's --force-reinstall drags its dependency closure (numpy, pandas) back to
# conda versions; re-pin the two the rest of the stack was resolved against.
# numpy stays 2.x either way, so the just-restored conda RDKit still matches.
python -m pip install "pandas==2.3.3"
python -m pip install --force-reinstall --no-deps "numpy==2.2.6"

echo "==> [6/6] pinning the OpenMP workaround into the env"
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
# Exercise RDKit's COMPILED extensions, not just `import rdkit` (pure Python):
# this is the exact surface the PyTDC NumPy-1 wheel breaks, so the check has to
# touch Chem + DataStructs or the _ARRAY_API failure slips through silently.
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DataStructs import ConvertToNumpyArray
import numpy as _np
_fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles("Cc1ccccc1"), 2, nBits=2048)
_arr = _np.zeros(2048, dtype=_np.int8); ConvertToNumpyArray(_fp, _arr)
assert int(_arr.sum()) > 0, "RDKit fingerprint is empty — C-extension broken under NumPy 2"
print("  rdkit C-ext  functional (Chem+DataStructs under NumPy", _np.__version__ + ")")
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
