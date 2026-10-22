"""Shared definitions for the GP-MOBO head-to-head (Yong, arXiv:2508.14072).

Everything that more than one of ``run_one.py`` / ``run_sweep.py`` /
``report.py`` / ``test_bench_yong.py`` must agree on lives here, once: the
library and its seeded split, the three tasks and their per-objective
components, the arms, the stand-ins that let her published code import, and
the 2048-bit fingerprint both sides share.

This module imports NOTHING from the repo's own optimizer stack (mogp,
acquisition, mogp_core).  Her arms run in processes that never load our code,
and our arms pull it in explicitly in ``run_one.py``.

Modifications to her code, all of them, in one place (each is also written into
every run's ``run.json``):

M1  sys.modules stand-ins so her code imports at all:
    a. ``rdkit.six`` -- removed from RDKit long ago.  Her oracle module imports
       ``iteritems`` from it at module level and uses it only inside
       ``calculateScore`` (the SA score), which none of the three MPOs calls.
    b. ``utils.utils_final`` -- as shipped, line 23 calls
       ``Oracle("osimertinib_tpsa_score")``, which is not a registered oracle
       name and raises ValueError at import, so ``ehvi_mc`` (which imports
       ``evaluate_fex_objectives`` from it) cannot be imported.  The stand-in
       contains her ``evaluate_fex_objectives`` compiled from her own source
       (AST-extracted, original file name and line numbers) against her three
       registered Fexofenadine oracles, plus Amlodipine / Perindopril evaluators
       (see M4).
M2  Seeding.  Her ``random.shuffle`` and her Monte Carlo draws
    (``np.random.multivariate_normal``) use the unseeded global RNGs.  We call
    ``random.seed(seed)`` before the shuffle and ``np.random.seed(seed)`` before
    importing her driver, so a (task, arm, seed) cell is reproducible.
M3  Batched predictions (only if gate G3 passes): her
    ``independent_tanimoto_gp_predict`` is called once per iteration on the
    whole pool instead of once per candidate.  Her 1000-draw EHVI estimator is
    then called per candidate, in her loop's order, unchanged.
M4  Task binding.  Her ``bayesian_optimization_loop`` hard-codes
    ``evaluate_fex_objectives``.  For Amlodipine / Perindopril we rebind that
    module global to a two-component evaluator.  At the vendored commit those
    MPOs exist only as scalars; the split mirrors her own later ``guacamol-test``
    branch (``amlodipine_similarity_value`` / ``amlodipine_num_rings_value``,
    ``perindopril_similarity_value`` / ``perindopril_aromatic_rings``), and gate
    G2 proves gmean(components) == her scalar oracle to 1e-12.
ARM HER@FP2 only: ``kern_gp.gp_model.get_fingerprint`` is replaced by the
    2048-bit binary Morgan radius-2 fingerprint.  That is the arm's definition,
    not a repair; HER-PUB keeps her unfolded radius-3 count fingerprint.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import types

import numpy as np

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BENCH_DIR)
GP_DIR = os.path.join(REPO, "external", "GP-MOBO")
LIB_PATH = os.path.join(GP_DIR, "guacamol_dataset", "guacamol_v1_train.smiles")
UTILS_FINAL_PATH = os.path.join(GP_DIR, "utils", "utils_final.py")
FIXTURES = os.path.join(BENCH_DIR, "fixtures")

GP_MOBO_COMMIT = "ca623180f42fc24a33ed371b295841d2cce815ba"
REWRITE_BASE_COMMIT = "2a7c2b8dfe6fe75a829494e01a1af835231b5502"

N_LIBRARY = 10_000          # rows 0..9999: shuffled, start set + candidate pool
N_HOLDOUT = 1_000           # rows 10000..10999: her hyperparameter block
N_READ = N_LIBRARY + N_HOLDOUT
HER_MC_DRAWS = 1000         # her estimator's N, left alone

# Pre-registered grid (PREREGISTRATION.md).  run_sweep.py refuses any other.
PREREG_TIER1_SEEDS = list(range(10))
PREREG_TIER2_SEEDS = list(range(6))
PREREG_TIER3_SEEDS = list(range(6))

TASKS = {
    "fexofenadine": {
        "oracle": "fexofenadine_mpo",
        "evaluator": "evaluate_fex_objectives",
        "objectives": ["tpsa_maxgauss_90_10", "logp_mingauss_4_1",
                       "ap_similarity_clip_0p8"],
    },
    "amlodipine": {
        "oracle": "amlodipine_mpo",
        "evaluator": "evaluate_amlo_objectives",
        "objectives": ["ecfp4_count_similarity", "num_rings_gauss_3_0p5"],
    },
    "perindopril": {
        "oracle": "perindopril_mpo",
        "evaluator": "evaluate_perin_objectives",
        "objectives": ["ecfp4_count_similarity", "aromatic_rings_gauss_2_0p5"],
    },
}

ARMS = ["HER@FP2", "HER-PUB", "MOGP-ICM@FP2", "MOGP-IND@FP2", "RANDOM"]
ARM_ALIASES = {"MOGP-ICM": "MOGP-ICM@FP2", "MOGP-IND": "MOGP-IND@FP2"}
HER_ARMS = ("HER@FP2", "HER-PUB")
OUR_ARMS = ("MOGP-ICM@FP2", "MOGP-IND@FP2")

# The thesis prints these widths; her CODE uses others, and the code wins.
FEX_WIDTH_DISCREPANCY = {
    "thesis_table_2_1": {"TPSA": "MaxGaussian(90, 2)", "logP": "MinGaussian(4, 2)"},
    "code_ca623180": {"TPSA": "MaxGaussianModifier(mu=90, sigma=10)",
                      "logP": "MinGaussianModifier(mu=4, sigma=1)"},
    "used": "code",
    "note": "Using the thesis widths moves top-20 values by up to 0.1239 and "
            "reorders the table (measured before this benchmark).",
}

# Her amlodipine reference contains "\C"; a raw string keeps the backslash,
# exactly as her (non-raw) source literal does, since "\C" is not an escape.
AMLODIPINE_SMILES = r"Clc1ccccc1C2C(=C(/N/C(=C2/C(=O)OCC)COCCN)C)\C(=O)OC"
PERINDOPRIL_SMILES = "O=C(OCC)C(NC(C(=O)N1C(C(=O)O)CC2CCCCC12)C)CCC"


def canonical_arm(arm: str) -> str:
    arm = ARM_ALIASES.get(arm, arm)
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; choose from {ARMS}")
    return arm


def arm_slug(arm: str) -> str:
    return canonical_arm(arm).replace("@", "-at-")


def cell_name(task: str, arm: str, seed: int, n_init: int) -> str:
    return f"{task}__{arm_slug(arm)}__seed{int(seed)}__init{int(n_init)}"


# --------------------------------------------------------------------------
# Library and split
# --------------------------------------------------------------------------

def read_library_lines(n=N_READ):
    """Her loader, verbatim: ``pd.read_csv(header=None, names=["smiles"])``."""
    import pandas as pd
    df = pd.read_csv(LIB_PATH, header=None, names=["smiles"])
    return df["smiles"].tolist()[:n]


def split_for_seed(seed: int, n_init: int, lines=None):
    """Her main-block split, with the shuffle seeded (M2).

    ``ehvi_mc.py``: ``hyperparam_smiles = ALL[10000:]``, ``ALL = ALL[:10000]``,
    ``random.shuffle(ALL)``, ``training = ALL[:20]``, ``query = ALL[20:10000]``.
    ``random.seed(seed)`` immediately before the shuffle is the only change.

    Returns ``(start, pool, holdout, index_of)`` where ``index_of`` maps a SMILES
    to its 0-based line number in the library file.
    """
    if lines is None:
        lines = read_library_lines()
    holdout = list(lines[N_LIBRARY:N_READ])
    library = list(lines[:N_LIBRARY])
    index_of = {s: i for i, s in enumerate(library)}
    if len(index_of) != N_LIBRARY:
        raise ValueError("library SMILES are not unique (gate G4)")
    shuffled = list(library)
    random.seed(int(seed))
    random.shuffle(shuffled)
    start = shuffled[:int(n_init)]
    pool = shuffled[int(n_init):N_LIBRARY]
    return start, pool, holdout, index_of


# --------------------------------------------------------------------------
# Her code: stand-ins, loader, oracles
# --------------------------------------------------------------------------

def _install_rdkit_six():
    if "rdkit.six" not in sys.modules:
        six = types.ModuleType("rdkit.six")
        six.__doc__ = ("bench_yong stand-in (M1a): rdkit.six was removed from "
                       "RDKit; her oracle uses iteritems only in calculateScore.")
        six.iteritems = lambda d: iter(d.items())
        sys.modules["rdkit.six"] = six


def add_her_path():
    _install_rdkit_six()
    if GP_DIR not in sys.path:
        # Appended, never prepended: her package names must not shadow ours.
        sys.path.append(GP_DIR)


def her_oracle(name: str):
    add_her_path()
    from tdc_oracles_modified import Oracle
    return Oracle(name)


def _component_oracle(fn):
    """Her ``Oracle.__call__`` wrapping one of our component functions.

    Built from a registered oracle and re-pointed, so invalid-SMILES handling,
    call counting and list/str dispatch are exactly hers.
    """
    o = her_oracle("amlodipine_mpo")
    o.evaluator_func = fn
    return o


def _ecfp4_similarity(reference_smiles):
    from tdc_oracles_modified.chem_utils.oracle.oracle import (
        DataStructs, smiles_2_fingerprint_ECFP4)
    ref_fp = smiles_2_fingerprint_ECFP4(reference_smiles)

    def similarity_value(test_smiles):
        fp = smiles_2_fingerprint_ECFP4(test_smiles)
        return DataStructs.TanimotoSimilarity(fp, ref_fp)
    return similarity_value


def amlodipine_num_rings_value(test_smiles):
    from tdc_oracles_modified.chem_utils.oracle.oracle import (
        GaussianModifier, rdMolDescriptors, smiles_to_rdkit_mol)
    mol = smiles_to_rdkit_mol(test_smiles)
    return GaussianModifier(mu=3, sigma=0.5)(rdMolDescriptors.CalcNumRings(mol))


def perindopril_aromatic_rings(test_smiles):
    from tdc_oracles_modified.chem_utils.oracle.oracle import (
        GaussianModifier, rdMolDescriptors, smiles_to_rdkit_mol)
    mol = smiles_to_rdkit_mol(test_smiles)
    return GaussianModifier(mu=2, sigma=0.5)(
        rdMolDescriptors.CalcNumAromaticRings(mol))


def _two_component_evaluator(o1, o2):
    """Same shape and NaN rule as her ``evaluate_*_objectives``, no prints."""
    def evaluate(smiles_list):
        f1 = np.array(o1(smiles_list))
        f2 = np.array(o2(smiles_list))
        valid = ~np.isnan(f1)
        f1, f2 = f1[valid], f2[valid]
        if not (len(f1) == len(f2)):
            raise ValueError("All input arrays must have the same shape")
        return np.stack([f1, f2]).T
    return evaluate


def build_utils_final_stand_in():
    """The ``utils.utils_final`` stand-in (M1b, M4)."""
    add_her_path()
    from tdc_oracles_modified import Oracle

    mod = types.ModuleType("utils.utils_final")
    mod.__doc__ = ("bench_yong stand-in for GP-MOBO utils/utils_final.py "
                   "(M1b): her evaluate_fex_objectives compiled from her source; "
                   "Amlodipine/Perindopril evaluators added (M4).")
    mod.__file__ = UTILS_FINAL_PATH + " [bench_yong stand-in]"
    ns = mod.__dict__
    ns["np"] = np
    ns["Oracle"] = Oracle
    ns["TPSA_ORACLE"] = Oracle("tpsa_score_single")
    ns["LOGP_ORACLE"] = Oracle("logp_score_single")
    ns["FEXOFENADINE_SIM_ORACLE"] = Oracle("fex_similarity_value_single")

    with open(UTILS_FINAL_PATH) as fh:
        tree = ast.parse(fh.read(), filename=UTILS_FINAL_PATH)
    fns = [n for n in tree.body
           if isinstance(n, ast.FunctionDef) and n.name == "evaluate_fex_objectives"]
    if len(fns) != 1:
        raise RuntimeError("evaluate_fex_objectives not found exactly once in "
                           "her utils_final.py")
    code = compile(ast.Module(body=fns, type_ignores=[]), UTILS_FINAL_PATH, "exec")
    exec(code, ns)

    ns["evaluate_amlo_objectives"] = _two_component_evaluator(
        _component_oracle(_ecfp4_similarity(AMLODIPINE_SMILES)),
        _component_oracle(amlodipine_num_rings_value))
    ns["evaluate_perin_objectives"] = _two_component_evaluator(
        _component_oracle(_ecfp4_similarity(PERINDOPRIL_SMILES)),
        _component_oracle(perindopril_aromatic_rings))
    return mod


def install_stand_ins():
    add_her_path()
    if "utils.utils_final" not in sys.modules:
        sys.modules["utils.utils_final"] = build_utils_final_stand_in()
    return sys.modules["utils.utils_final"]


def task_evaluator(task: str):
    """The per-objective evaluator every arm uses for ``task``."""
    mod = install_stand_ins()
    return getattr(mod, TASKS[task]["evaluator"])


def scalar_oracle(task: str):
    return her_oracle(TASKS[task]["oracle"])


def score_scalar(task: str, smiles_list):
    return np.asarray(scalar_oracle(task)(list(smiles_list)), dtype=np.float64)


# --------------------------------------------------------------------------
# The shared fingerprint (FP2) as an RDKit bit vector, for HER@FP2
# --------------------------------------------------------------------------

_FP2_GENERATOR = None


def fp2_bitvect(smiles: str):
    """2048-bit binary Morgan radius 2 -- the generator ``utils/featurize.py`` uses."""
    global _FP2_GENERATOR
    from rdkit import Chem
    from rdkit.Chem import AllChem
    if _FP2_GENERATOR is None:
        _FP2_GENERATOR = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    mol = Chem.MolFromSmiles(smiles)
    assert mol is not None
    return _FP2_GENERATOR.GetFingerprint(mol)


# --------------------------------------------------------------------------
# Small shared utilities
# --------------------------------------------------------------------------

def nondominated_count(Y) -> int:
    """Size of the (maximization) non-dominated set, duplicates kept once."""
    Y = np.unique(np.asarray(Y, dtype=np.float64), axis=0)
    n = 0
    for i in range(Y.shape[0]):
        ge = np.all(Y >= Y[i], axis=1)
        gt = np.any(Y > Y[i], axis=1)
        if not np.any(ge & gt):
            n += 1
    return n


def fmt_float(x) -> str:
    return repr(float(x))


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def git_head(path: str = REPO):
    try:
        return subprocess.run(["git", "-C", path, "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True
                              ).stdout.strip()
    except Exception:
        return None


def git_dirty(path: str = REPO, rel: str = None) -> bool:
    cmd = ["git", "-C", path, "status", "--porcelain"]
    if rel:
        cmd += ["--", rel]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
        return bool(out.strip())
    except Exception:
        return True


def imported_versions():
    """Versions that actually IMPORT in this interpreter (gate G7)."""
    import importlib
    out = {"python": platform.python_version(), "executable": sys.executable,
           "platform": platform.platform()}
    for name in ("numpy", "scipy", "pandas", "rdkit", "sklearn", "networkx",
                 "matplotlib"):
        try:
            m = importlib.import_module(name)
            out[name] = getattr(m, "__version__", "?")
        except Exception as e:  # recorded, not raised
            out[name] = f"IMPORT FAILED: {type(e).__name__}: {e}"
    return out


def write_json_atomic(path: str, obj) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, sort_keys=True, default=str)
        fh.write("\n")
    os.replace(tmp, path)


def write_text_atomic(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(text)
    os.replace(tmp, path)
