"""
test_run_benchmark_seeds.py
===========================

Guards the ``--methods`` subset selection and per-method runner construction in
``run_benchmark_seeds.py`` — the wiring that lets one invocation request exactly
``mogp gpmobo greedy`` while every selected arm still shares the same seed,
library, budget and docking oracle.

These are fast unit tests: the library is monkeypatched to a tiny synthetic one
so constructing each runner does not load (or require) the 29k-molecule library
or launch any docking.
"""

import numpy as np
import pytest

import run_benchmark_seeds as R
from loop import BOLoop
from baseline_gpmobo import GPMOBOBaseline
from baseline_greedy import GreedyFilterThenDock


# A handful of real, RDKit-parseable drug-like SMILES: GP-MOBO recomputes its
# own fingerprints from SMILES at construction, so these must be valid.
TINY_SMILES = [
    "CCO",                                   # ethanol
    "c1ccccc1",                              # benzene
    "CC(=O)Oc1ccccc1C(=O)O",                 # aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",          # caffeine
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",            # ibuprofen
    "CN(C)C(=N)N=C(N)N",                      # metformin
]


@pytest.fixture
def tiny_library(monkeypatch):
    """Patch ``load_library`` in every runner's namespace to a tiny library.

    Row-aligned smiles / fingerprints / admet_scores, shaped exactly as the real
    ``load_library`` returns them, so each runner's ``__init__`` succeeds without
    touching disk.
    """
    n = len(TINY_SMILES)
    rng = np.random.RandomState(0)
    lib = {
        "smiles": list(TINY_SMILES),
        "fingerprints": rng.randint(0, 2, size=(n, 2048)).astype(np.int8),
        # admet columns [Caco2_logPapp, Half_Life_hours, hERG_Toxicity_Prob];
        # values only need to be finite for construction.
        "admet_scores": rng.rand(n, 3).astype(np.float32),
    }
    for module in (BOLoop.__module__, GPMOBOBaseline.__module__,
                   GreedyFilterThenDock.__module__):
        monkeypatch.setattr(f"{module}.load_library",
                            lambda library_dir=None, *a, **k: lib)
    return lib


def _params():
    """Minimal shared params, as run_benchmark_seeds.main() would build them."""
    n_init, batch, iters = 2, 2, 1
    return {
        "n_init": n_init, "batch_size": batch, "n_iterations": iters,
        "mogp_iters": 1, "n_total": n_init + iters * batch,
        "densify": False, "library_dir": "unused-monkeypatched",
    }


def test_methods_flag_selects_requested_subset():
    """--methods keys filter down to exactly those methods, in METHODS order."""
    selected = R.filter_methods(["mogp", "gpmobo", "greedy"])
    assert [key for _, key, _ in selected] == ["mogp", "gpmobo", "greedy"]

    # Order follows METHODS (legend/color stability), not the typed order.
    assert R.filter_methods(["greedy", "mogp"]) == R.filter_methods(["mogp", "greedy"])

    # Default is every method.
    assert R.filter_methods(R.ALL_METHOD_KEYS) == R.METHODS

    # A single key selects a single method.
    assert [k for _, k, _ in R.filter_methods(["gpmobo"])] == ["gpmobo"]

    # GP-MOBO is a first-class registered method with a distinct color.
    colors = {key: color for _, key, color in R.METHODS}
    assert "gpmobo" in colors
    assert len({colors[k] for k in ("mogp", "gpmobo", "greedy")}) == 3


def test_unknown_method_key_raises():
    """A typo'd key fails loudly instead of silently running a wrong subset."""
    with pytest.raises(ValueError):
        R.filter_methods(["mopg"])   # transposed 'mogp'


def test_selected_three_all_construct(tiny_library):
    """mogp, gpmobo and greedy each construct via _build_runner on one seed."""
    params = _params()
    expected = {"mogp": BOLoop, "gpmobo": GPMOBOBaseline, "greedy": GreedyFilterThenDock}
    for key, cls in expected.items():
        runner = R._build_runner(key, params, seed=0)
        assert isinstance(runner, cls)
        # Same library reached every arm (fair-comparison invariant).
        assert runner.library_size == len(TINY_SMILES)


def test_all_selected_receive_same_budget_and_seed(tiny_library):
    """Every selected arm is built with the same seed and evaluation budget."""
    params = _params()
    seed = 7
    runners = {key: R._build_runner(key, params, seed)
               for _, key, _ in R.filter_methods(["mogp", "gpmobo", "greedy"])}
    for runner in runners.values():
        assert runner.seed == seed
    # Batch arms carry n_init/batch_size/n_iterations; greedy carries the
    # equivalent total budget n_total = n_init + n_iterations * batch_size.
    assert runners["mogp"].n_init == runners["gpmobo"].n_init == params["n_init"]
    assert runners["mogp"].batch_size == runners["gpmobo"].batch_size == params["batch_size"]
    assert runners["greedy"].n_total == params["n_total"]


def test_acquisition_pool_size_propagates_identically(tiny_library):
    """--acquisition-pool-size reaches MOGP and GP-MOBO identically; not Greedy."""
    params = _params()
    params["acquisition_pool_size"] = 2000
    mogp = R._build_runner("mogp", params, seed=0)
    gpmobo = R._build_runner("gpmobo", params, seed=0)
    greedy = R._build_runner("greedy", params, seed=0)
    assert mogp.acquisition_pool_size == 2000
    assert gpmobo.acquisition_pool_size == 2000
    # Greedy has no acquisition, so it must not carry the knob.
    assert not hasattr(greedy, "acquisition_pool_size")


def test_default_pool_size_is_none_full_library(tiny_library):
    """Omitting the flag leaves both arms scoring the whole library (unchanged)."""
    mogp = R._build_runner("mogp", _params(), seed=0)
    gpmobo = R._build_runner("gpmobo", _params(), seed=0)
    assert mogp.acquisition_pool_size is None
    assert gpmobo.acquisition_pool_size is None


def test_arms_expose_timing_attributes(tiny_library):
    """Both BO arms expose a settable per-iteration timing log path + method tag."""
    mogp = R._build_runner("mogp", _params(), seed=0)
    gpmobo = R._build_runner("gpmobo", _params(), seed=0)
    assert mogp.timing_log_path is None and gpmobo.timing_log_path is None
    assert mogp.timing_method == "MOGP"
    assert gpmobo.timing_method == "GP-MOBO"


def test_gpmobo_shares_one_pool_per_round(tiny_library, monkeypatch, tmp_path):
    """GP-MOBO scores ONE shared pool per round, not a fresh one per q=1 pick.

    Regression guard for the candidate-exposure fix: with a pool cap of P and a
    batch of B q=1 picks, distinct candidates scored per round must be P (the
    single shared subsample), NOT ~B*P. Docking is monkeypatched so no Vina runs.
    """
    import csv
    from baseline_gpmobo import GPMOBOBaseline

    def fake_dock(smiles, targets):
        # Finite, distinct per-molecule scores; no Vina, no receptors.
        return {t: np.array([-7.0 - 0.1 * k for k in range(len(smiles))], float)
                for t in targets}
    monkeypatch.setattr("baseline_gpmobo.batch_dock_targets", fake_dock)

    pool, batch = 3, 3
    runner = GPMOBOBaseline(
        library_dir="unused", seed=0, n_init=2, batch_size=batch,
        n_iterations=1, acquisition_pool_size=pool,
    )
    runner.timing_log_path = str(tmp_path / "iteration_timings.csv")
    runner.run()

    with open(runner.timing_log_path) as fh:
        rows = list(csv.DictReader(fh))
    assert rows, "GP-MOBO wrote no timing rows"
    # The shared per-round pool => distinct exposure == pool size, not batch*pool.
    assert int(rows[0]["n_candidates_scored"]) == pool
    assert int(rows[0]["n_candidates_scored"]) != batch * pool


def test_timing_log_written_incrementally(tmp_path):
    """timing.append_timing_row is readable after each write, not just at end."""
    import csv
    import timing

    path = str(tmp_path / "sub" / "iteration_timings.csv")
    timing.init_timing_log(path)               # creates dirs + header
    for it in (1, 2, 3):
        timing.append_timing_row(path, {
            "timestamp": timing.now_iso(), "method": "MOGP", "seed": 0,
            "iteration": it, "gp_train_seconds": 1.0, "acquisition_seconds": 2.0,
            "docking_seconds": 0.5, "iteration_seconds": 3.5,
            "acquisition_pool_size": 2000, "n_candidates_scored": 2000,
            "pareto_size": 10 + it, "n_evaluated": 40 + it,
        })
        # Readable mid-run: re-open and count rows after each append.
        with open(path) as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) == it
        assert rows[-1]["method"] == "MOGP"
        assert int(rows[-1]["n_candidates_scored"]) == 2000
    # Header matches the documented schema.
    with open(path) as fh:
        header = next(csv.reader(fh))
    assert header == timing.TIMING_COLUMNS
