"""
test_arm_objective_units.py — cross-arm objective-UNITS regression guard.

The GP-MOBO HV=0.0000 defect was baseline_gpmobo._evaluate storing ligand
efficiency (raw kcal / heavy atoms) in the optimized docking columns, while
loop.py:330 and every other arm store raw kcal/mol. This is the *fourth* time a
change has landed on one code path and not its sibling, so the point of this test
is to make CI catch a units divergence between arms rather than a post-hoc audit.

For EACH benchmark arm (discovered from run_benchmark_seeds so a future arm is
covered automatically) we run a tiny 2-molecule smoke evaluation with the docking
oracle stubbed to fixed raw kcal, and assert on the arm's OPTIMIZED objective
matrix:

  (a) each optimized docking column equals the raw-kcal it was fed, within 1e-9
      (the exact, load-bearing check);
  (b) every optimized docking value has |value| > 1.0 kcal/mol — ligand
      efficiency is ~0.3-0.5, so this catches LE-vs-raw specifically even if (a)
      is ever loosened.

We deliberately do NOT assert values lie inside DOCKING_KCAL_MIN/MAX = [-11, -5]:
that is the normalization frame, not a validity range — real raw scores fall
outside it in both directions and such an assertion would flake.
"""
import numpy as np
import pytest

import run_benchmark_seeds as R
from loop import DOCKING_TASKS, DOCKING_TARGETS   # (col_index, target) layout; shared by all arms

# Arm modules that expose `_evaluate` + `batch_dock_targets`; patched per-arm.
ARM_MODULES = ["loop", "baseline_gpmobo", "baseline_greedy",
               "baseline_random", "baseline_single_obj"]

# Stubbed raw docking scores for the 2 smoke molecules, per target. Magnitudes
# are well above 1 and each molecule has > 15 heavy atoms, so the LE a buggy arm
# would compute (raw / heavy) is < 1 — making check (b) discriminating.
FAKE_RAW = {t: np.array(v, float) for t, v in {
    "PfDHFR": [-7.3, -8.1],
    "hDHFR":  [-5.9, -6.7],
}.items()}

SMOKE_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",            # aspirin, 13 heavy
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",     # caffeine, 14 heavy
]


def _fake_dock(smiles, targets):
    n = len(smiles)
    return {t: FAKE_RAW[t][:n].copy() for t in targets}


@pytest.fixture
def stub_env(monkeypatch):
    """Tiny library + stubbed docking, applied to every arm module."""
    n = len(SMOKE_SMILES)
    rng = np.random.RandomState(0)
    lib = {
        "smiles": list(SMOKE_SMILES),
        "fingerprints": rng.randint(0, 2, size=(n, 2048)).astype(np.int8),
        "admet_scores": rng.rand(n, 3).astype(np.float32),
    }
    for mod in ARM_MODULES:
        monkeypatch.setattr(f"{mod}.load_library", lambda *a, **k: lib, raising=False)
        monkeypatch.setattr(f"{mod}.batch_dock_targets", _fake_dock, raising=False)
    return lib


# Every arm the benchmark knows about — a new arm added to run_benchmark_seeds is
# covered here automatically, no edit to this test required.
@pytest.mark.parametrize("arm_key", R.ALL_METHOD_KEYS)
def test_optimized_docking_columns_are_raw_kcal(arm_key, stub_env):
    params = {
        "n_init": 2, "batch_size": 1, "n_iterations": 1, "mogp_iters": 1,
        "n_total": 2, "densify": False, "acquisition_pool_size": None,
        "library_dir": "unused-stubbed",
    }
    runner = R._build_runner(arm_key, params, seed=0)

    # The optimized objective matrix is element [0] of every arm's _evaluate.
    Y = np.asarray(runner._evaluate([0, 1])[0], float)

    for col, target in DOCKING_TASKS:
        opt = Y[:, col]
        raw = FAKE_RAW[target]
        # (a) exact: optimized column IS the raw kcal it was fed.
        assert np.allclose(opt, raw, atol=1e-9), (
            f"[{arm_key}] objective column {col} ({DOCKING_TARGETS}) is not raw "
            f"kcal: got {opt.tolist()}, fed {raw.tolist()}. An arm is storing a "
            f"transformed value (e.g. ligand efficiency) where the others store "
            f"raw kcal/mol."
        )
        # (b) magnitude: catches LE (~0.3-0.5) even if (a) is loosened later.
        assert np.all(np.abs(opt) > 1.0), (
            f"[{arm_key}] objective column {col} has |value| <= 1.0 kcal/mol "
            f"({opt.tolist()}); ligand efficiency lives at ~0.3-0.5 — this arm is "
            f"almost certainly storing LE, not raw kcal."
        )
