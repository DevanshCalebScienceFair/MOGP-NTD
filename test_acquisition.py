"""
test_acquisition.py
===================

Tests for the GREY-BOX / composite restructure:

  * The GP (``mogp.train_mogp`` / ``mogp.predict``) models ONLY the docking
    objectives; its trained-task count is 2 and ``predict`` returns NaN for the
    three ADMET columns.
  * The qNEHVI acquisition (``acquisition``) uses each candidate's KNOWN-EXACT
    ADMET values — from ``self.admet_scores`` — never a GP estimate. The composite
    objective's ADMET slots equal the known table and are invariant to the GP
    docking samples, and it reproduces ``evaluation.normalize`` exactly.
  * A tiny end-to-end BO loop (with docking mocked) still completes and writes
    history/evaluated/pareto CSVs, with hypervolume produced via ``evaluation.py``.

Runnable both as ``pytest test_acquisition.py`` and as ``python test_acquisition.py``.
"""

import os

import numpy as np
import torch
import pytest

from mogp import (
    TASK_NAMES,
    DOCKING_TASK_INDICES,
    OBJECTIVE_SOURCES,
    train_mogp,
    predict,
)
from acquisition import (
    compose_objective_points,
    _augment_with_admet,
    _resolve_admet_layout,
    DockingPosteriorModel,
    CompositeKnownADMETObjective,
)
import evaluation


LIBRARY_DIR = "data/library"
N_FP = 2048
_ADMET_TASKS = [j for j, n in enumerate(TASK_NAMES) if OBJECTIVE_SOURCES[n][0] != "dock"]


# ---------------------------------------------------------------------- #
# Helpers
# ---------------------------------------------------------------------- #
def _layout_or_skip():
    """Resolved ``(dock_idx, lib_idx, lib_admet_cols)`` or skip if unavailable."""
    try:
        return _resolve_admet_layout()
    except Exception as exc:                                  # pragma: no cover
        pytest.skip(f"objective layout unavailable: {exc}")


def _library_or_skip():
    """Cached library dict or skip if it has not been built."""
    try:
        from data import load_library

        return load_library(LIBRARY_DIR)
    except Exception as exc:                                  # pragma: no cover
        pytest.skip(f"cached library unavailable: {exc}")


def _fake_Y(rng, n):
    """A (n, 5) target matrix: docking columns filled, ADMET columns filled too
    (the GP must ignore the ADMET columns)."""
    Y = np.zeros((n, len(TASK_NAMES)), dtype=np.float32)
    for j, name in enumerate(TASK_NAMES):
        if OBJECTIVE_SOURCES[name][0] == "dock":
            Y[:, j] = rng.uniform(-11.0, -5.0, size=n)
        else:
            Y[:, j] = rng.uniform(0.0, 1.0, size=n)
    return Y


# ---------------------------------------------------------------------- #
# The GP models docking objectives ONLY
# ---------------------------------------------------------------------- #
def test_gp_trains_only_docking_and_predicts_nan_admet():
    """train_mogp fits exactly the 2 docking tasks; predict returns NaN ADMET."""
    rng = np.random.default_rng(0)
    n = 8
    X = (rng.random((n, N_FP)) < 0.05).astype(np.int8)
    Y = _fake_Y(rng, n)

    model, likelihood, y_mean, y_std = train_mogp(X, Y, n_iterations=10)

    # Only the docking columns carry finite normalization stats -> only they are
    # trained; the trained-task count is exactly len(DOCKING_TASK_INDICES) == 2.
    observed = np.where(np.isfinite(y_mean) & np.isfinite(y_std))[0].tolist()
    assert observed == list(DOCKING_TASK_INDICES)
    assert int(likelihood.num_tasks) == len(DOCKING_TASK_INDICES) == 2

    mean, variance = predict(model, likelihood, y_mean, y_std, X[:3])
    assert mean.shape == (3, len(TASK_NAMES))
    for j, name in enumerate(TASK_NAMES):
        if OBJECTIVE_SOURCES[name][0] == "dock":
            assert np.isfinite(mean[:, j]).all(), f"{name} should be predicted"
        else:
            assert np.isnan(mean[:, j]).all(), f"{name} must be NaN (not modelled)"


def test_docking_posterior_model_has_two_outputs():
    """The BoTorch wrapper exposes exactly the docking outputs (2), not 5."""
    rng = np.random.default_rng(1)
    n = 6
    X = (rng.random((n, N_FP)) < 0.05).astype(np.int8)
    Y = _fake_Y(rng, n)
    model, likelihood, y_mean, y_std = train_mogp(X, Y, n_iterations=8)

    wrap = DockingPosteriorModel(
        model, likelihood, y_mean, y_std, n_fp=N_FP,
        dock_task_indices=list(DOCKING_TASK_INDICES),
    )
    assert wrap.num_outputs == len(DOCKING_TASK_INDICES) == 2

    # A posterior over a few augmented points has a 2-dim (docking) event.
    tail = rng.random((4, len(_ADMET_TASKS)))
    X_aug = torch.as_tensor(np.concatenate([X[:4], tail], axis=1), dtype=torch.double)
    post = wrap.posterior(X_aug.unsqueeze(1))     # (4, 1, d) -> mean (4, 1, 2)
    assert post.mean.shape[-1] == 2


# ---------------------------------------------------------------------- #
# The acquisition uses KNOWN-EXACT ADMET, never the GP posterior
# ---------------------------------------------------------------------- #
def test_composite_admet_equals_known_scores_and_ignores_samples():
    """The ADMET the acquisition composes for a candidate equals
    ``self.admet_scores[candidate]`` (mapped to its task index) and does NOT
    depend on the GP-sampled docking values."""
    dock_idx, lib_idx, lib_admet_cols = _layout_or_skip()
    lib = _library_or_skip()

    cand = [0, 3, 7, 11]
    admet_rows = np.asarray(lib["admet_scores"][cand], dtype=float)   # admet_scores order
    tail = admet_rows[:, lib_admet_cols]                             # library-task order
    n_obj = len(TASK_NAMES)

    rng = np.random.default_rng(5)
    dock1 = rng.uniform(-12.0, -4.0, size=(len(cand), len(dock_idx)))
    dock2 = rng.uniform(-12.0, -4.0, size=(len(cand), len(dock_idx)))
    full1 = compose_objective_points(dock1, tail, dock_idx, lib_idx, n_obj)
    full2 = compose_objective_points(dock2, tail, dock_idx, lib_idx, n_obj)

    # Each ADMET objective column equals the exact known value from admet_scores.
    for k, task_j in enumerate(lib_idx):
        admet_col = lib_admet_cols[k]
        assert np.allclose(full1[:, task_j], admet_rows[:, admet_col]), (
            f"ADMET objective {TASK_NAMES[task_j]} must equal "
            f"self.admet_scores[:, {admet_col}]"
        )
    # ADMET columns are invariant to the GP docking samples (not read from the GP).
    assert np.allclose(full1[:, lib_idx], full2[:, lib_idx])
    # Docking columns DO come from the samples (and so change between draws).
    assert np.allclose(full1[:, dock_idx], dock1)
    assert not np.allclose(full1[:, dock_idx], full2[:, dock_idx])


def test_augment_appends_admet_in_task_order():
    """_augment_with_admet appends admet_scores rows reordered to library-task order."""
    dock_idx, lib_idx, lib_admet_cols = _layout_or_skip()
    lib = _library_or_skip()

    cand = [1, 4]
    X_fp = np.asarray(lib["fingerprints"][cand])
    admet_rows = np.asarray(lib["admet_scores"][cand], dtype=float)

    aug = _augment_with_admet(X_fp, admet_rows, lib_admet_cols)
    assert aug.shape == (len(cand), X_fp.shape[1] + len(lib_idx))
    # Fingerprint prefix preserved.
    assert np.allclose(aug[:, :X_fp.shape[1]], X_fp)
    # Tail equals admet_scores columns in library-task order.
    tail = aug[:, -len(lib_idx):]
    for k in range(len(lib_idx)):
        assert np.allclose(tail[:, k], admet_rows[:, lib_admet_cols[k]])


def test_composite_objective_matches_evaluation_normalize():
    """CompositeKnownADMETObjective reproduces evaluation.normalize exactly (so the
    acquisition scores in the same shared frame the hypervolume is reported in),
    and its ADMET output is invariant to the docking samples."""
    dock_idx, lib_idx, lib_admet_cols = _layout_or_skip()
    lib = _library_or_skip()

    cand = [2, 5, 9]
    admet_rows = np.asarray(lib["admet_scores"][cand], dtype=float)
    X_fp = np.asarray(lib["fingerprints"][cand])
    tail = admet_rows[:, lib_admet_cols]
    X_aug = torch.as_tensor(
        _augment_with_admet(X_fp, admet_rows, lib_admet_cols), dtype=torch.double
    )

    bounds = evaluation.compute_objective_bounds()
    signs = evaluation.OBJECTIVE_SIGNS
    obj = CompositeKnownADMETObjective(dock_idx, lib_idx, len(TASK_NAMES), bounds, signs)

    rng = np.random.default_rng(6)
    s1 = rng.uniform(-12.0, -4.0, size=(len(cand), len(dock_idx)))
    s2 = rng.uniform(-12.0, -4.0, size=(len(cand), len(dock_idx)))
    o1 = obj(torch.as_tensor(s1, dtype=torch.double), X=X_aug)
    o2 = obj(torch.as_tensor(s2, dtype=torch.double), X=X_aug)

    # Normalized ADMET columns are identical regardless of the docking samples.
    assert torch.allclose(o1[..., lib_idx], o2[..., lib_idx])

    # And the full normalized vector equals evaluation.normalize on the composed
    # original-unit points -> same frame as evaluation.compute_hypervolume.
    full_np = compose_objective_points(s1, tail, dock_idx, lib_idx, len(TASK_NAMES))
    expected = evaluation.normalize(
        full_np, objective_indices=list(range(len(TASK_NAMES))),
        bounds=bounds, signs=signs,
    )
    assert np.allclose(o1.cpu().numpy(), expected, atol=1e-6)


# ---------------------------------------------------------------------- #
# End-to-end loop (docking mocked)
# ---------------------------------------------------------------------- #
@pytest.mark.parametrize("model", ["coregionalized", "independent"])
def test_end_to_end_loop_writes_csvs_with_evaluation_hypervolume(tmp_path, monkeypatch, model):
    """A tiny BO loop completes with the grey-box qNEHVI acquisition and writes
    the three result CSVs, with hypervolume produced via evaluation.py — for both
    the primary coregionalized (ICM) model and the independent ablation model."""
    import loop as loopmod

    _library_or_skip()   # skip cleanly if the cached library is missing

    # Mock the expensive docking oracle with fast, finite scores per target.
    rng = np.random.default_rng(3)

    def fake_batch_dock_targets(smiles_list, targets):
        n = len(list(smiles_list))
        return {t: rng.uniform(-11.0, -5.0, size=n) for t in targets}

    monkeypatch.setattr(loopmod, "batch_dock_targets", fake_batch_dock_targets)

    bo = loopmod.BOLoop(
        library_dir=LIBRARY_DIR, seed=2,
        n_init=6, batch_size=4, n_iterations=1,
        mogp_train_iters=12, diversity_threshold=0.95,
        model=model,
    )
    assert bo.model_name == model
    # Truncate to a small library subset so the candidate scan is fast.
    K = 24
    bo.smiles = bo.smiles[:K]
    bo.fingerprints = bo.fingerprints[:K]
    bo.admet_scores = bo.admet_scores[:K]
    bo.library_size = K

    history = bo.run()
    assert len(history) == 1

    out_dir = str(tmp_path / "results")
    bo.save_results(output_dir=out_dir)

    import pandas as pd
    hist_df = pd.read_csv(os.path.join(out_dir, "history.csv"))
    eval_df = pd.read_csv(os.path.join(out_dir, "evaluated.csv"))
    pareto_df = pd.read_csv(os.path.join(out_dir, "pareto_front.csv"))

    assert "hypervolume" in hist_df.columns and len(hist_df) == 1
    assert len(eval_df) == bo.n_init + bo.batch_size
    for name in TASK_NAMES:
        assert name in eval_df.columns and name in pareto_df.columns
    assert evaluation.SELECTIVITY_COLUMN in eval_df.columns

    # Hypervolume is the evaluation.py single-source-of-truth value.
    hv = evaluation.compute_hypervolume(bo.Y_evaluated)
    assert abs(hv - float(history[-1]["hypervolume"])) < 1e-9
    assert 0.0 <= hv <= 1.0


if __name__ == "__main__":
    import sys

    tests = [obj for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    # The tmp_path/monkeypatch test needs pytest fixtures; run the rest directly.
    failed = 0
    for test in tests:
        if test.__code__.co_argcount:
            print(f"SKIP    {test.__name__} (needs pytest fixtures)")
            continue
        try:
            test()
            print(f"PASSED  {test.__name__}")
        except Exception as exc:                             # pragma: no cover
            failed += 1
            print(f"FAILED  {test.__name__}: {exc}")
    sys.exit(1 if failed else 0)
