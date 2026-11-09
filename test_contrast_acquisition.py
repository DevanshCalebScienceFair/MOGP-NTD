"""End-to-end: does `linear_objectives` reach qNEHVI and change what it picks?

This file exists because of silent-defeat bug #1 and #4 on this branch: a flag
that is accepted, recorded in the run config, and never forwarded. Asserting
that `compute_qnehvi` has a `linear_objectives` parameter proves nothing. These
tests assert that the SCORES differ when they should and are identical when they
should not.

Reproducibility note. `SobolQMCNormalSampler` with no `seed=` draws from torch's
GLOBAL RNG, so two `compute_qnehvi` calls in a row use different Sobol points.
Every comparison below calls `torch.manual_seed` immediately before each call,
without which the equivalence test would fail for a reason that has nothing to
do with objectives.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest
import torch

import acquisition as A
from mogp import TASK_NAMES, DOCKING_TASK_INDICES, OBJECTIVE_SOURCES

NUM_OBJ = len(TASK_NAMES)
SEED = 123

BOUNDS = np.array([[-11.0, -5.0], [-11.0, -5.0], [0.0, 1.0],
                   [-7.26, -3.83], [0.376, 64.85]])
SIGNS = list(A.DEFAULT_OBJECTIVE_SIGNS)

DOCK = list(DOCKING_TASK_INDICES)
LIB = [j for j, name in enumerate(TASK_NAMES) if OBJECTIVE_SOURCES[name][0] != "dock"]
LAYOUT = (DOCK, LIB, list(range(len(LIB))))


@pytest.fixture(scope="module")
def fitted():
    """A small 2-task ICM fit on synthetic correlated docking data.

    Correlated on purpose: with independent tasks the contrast and coordinate
    paths would agree and the test would pass for the wrong reason.
    """
    from mogp_coregionalized import train_mogp_coregionalized

    rng = np.random.default_rng(SEED)
    n_train, n_fp = 24, 2048
    X = (rng.random((n_train, n_fp)) < 0.02).astype(np.float32)

    w = rng.standard_normal(n_fp).astype(np.float32)
    lat = X @ w
    lat = (lat - lat.mean()) / (lat.std() + 1e-8)

    Y = np.full((n_train, NUM_OBJ), np.nan, dtype=np.float32)
    Y[:, DOCK[0]] = -8.0 + 2.0 * lat + 0.25 * rng.standard_normal(n_train)
    Y[:, DOCK[1]] = -8.0 + 1.9 * lat + 0.25 * rng.standard_normal(n_train)
    for j in LIB:
        Y[:, j] = rng.standard_normal(n_train)

    torch.manual_seed(SEED)
    model, lik, y_mean, y_std = train_mogp_coregionalized(X, Y, n_iterations=40, rank=1)

    n_cand, n_base = 12, 8
    Xc = (rng.random((n_cand, n_fp)) < 0.02).astype(np.float32)
    ac = rng.random((n_cand, len(LIB)))
    Xb = X[:n_base]
    ab = rng.random((n_base, len(LIB)))
    return dict(model=model, likelihood=lik, y_mean=y_mean, y_std=y_std,
                Xc=Xc, ac=ac, Xb=Xb, ab=ab)


def _score(fitted, **kw):
    torch.manual_seed(SEED)   # Sobol draws come from the GLOBAL rng
    return A.compute_qnehvi(
        fitted["model"], fitted["likelihood"], fitted["y_mean"], fitted["y_std"],
        fitted["Xc"], fitted["ac"], fitted["Xb"], fitted["ab"],
        objective_signs=SIGNS, bounds=BOUNDS, layout=LAYOUT,
        n_mc_samples=32, partitioning_alpha=1e-3, **kw,
    )


# ------------------------------------------------------- OFF path, unchanged


@pytest.mark.parametrize("emit", [None, [0, 1]])
def test_off_path_is_byte_identical(fitted, emit):
    hist = _score(fitted, posterior_mode="joint", emit_objectives=emit)
    keep = list(range(NUM_OBJ)) if emit is None else emit
    declared = _score(fitted, posterior_mode="joint",
                      linear_objectives=A.coordinate_objectives(SIGNS, BOUNDS, keep))
    assert np.array_equal(hist, declared), np.abs(hist - declared).max()


# ------------------------------------------------------- ON path, has an effect


def test_contrast_objective_changes_the_scores(fitted):
    """The flag must CHANGE the answer, not merely be accepted."""
    pf, h = DOCK
    two_task = _score(fitted, posterior_mode="joint", emit_objectives=[pf, h])

    si = A.selectivity_contrast_objective(NUM_OBJ)
    pf_axis = A.coordinate_objectives(SIGNS, BOUNDS, [pf])[0]
    contrast = _score(fitted, posterior_mode="joint", linear_objectives=[pf_axis, si])

    assert contrast.shape == two_task.shape
    assert not np.allclose(contrast, two_task), (
        "the contrast axis produced the same scores as two coordinate axes, so "
        "linear_objectives is not reaching the acquisition"
    )


def test_contrast_objective_changes_which_molecules_are_ranked_first(fitted):
    """A score difference that does not move the ranking would buy nothing."""
    pf, h = DOCK
    base = _score(fitted, posterior_mode="joint", emit_objectives=[pf, h])
    si = A.selectivity_contrast_objective(NUM_OBJ)
    pf_axis = A.coordinate_objectives(SIGNS, BOUNDS, [pf])[0]
    contrast = _score(fitted, posterior_mode="joint", linear_objectives=[pf_axis, si])
    assert int(np.argmax(base)) != int(np.argmax(contrast)) or \
        not np.array_equal(np.argsort(-base), np.argsort(-contrast)), (
            "the contrast objective left the candidate ranking identical"
        )


def test_contrast_scores_depend_on_the_posterior_mode(fitted):
    """The covariance must reach the contrast.

    `diag` is REJECTED for a contrast, so the comparison is made against a
    diagonalized joint posterior built by hand: same means, covariance stripped.
    If the two agreed, the contrast would not be reading off-diagonals.
    """
    pf = DOCK[0]
    si = A.selectivity_contrast_objective(NUM_OBJ)
    pf_axis = A.coordinate_objectives(SIGNS, BOUNDS, [pf])[0]

    joint = _score(fitted, posterior_mode="joint", linear_objectives=[pf_axis, si])

    # Strip the off-diagonals at the point they are PRODUCED. An earlier
    # version of this test rebuilt a plain MultivariateNormal from the
    # posterior, which cannot work: a MultitaskMultivariateNormal carries a
    # (batch, q, k) mean against a (batch, q*k, q*k) covariance, so torch tries
    # to expand a (k, k) shape over the (q*k, q*k) block and raises. Patching
    # `_joint_moments` keeps every other quantity bit-identical and changes
    # exactly one thing, which is what the comparison needs.
    real_joint = A.DockingPosteriorModel._joint_moments

    def stripped_joint(self, fp, batch, q, k):
        mean_d, covar = real_joint(self, fp, batch, q, k)
        d = torch.diagonal(covar, dim1=-2, dim2=-1)
        return mean_d, torch.diag_embed(d)

    A.DockingPosteriorModel._joint_moments = stripped_joint
    try:
        stripped = _score(fitted, posterior_mode="joint",
                          linear_objectives=[pf_axis, si])
    finally:
        A.DockingPosteriorModel._joint_moments = real_joint

    assert not np.allclose(joint, stripped), (
        "stripping the off-diagonals left the contrast scores unchanged, so the "
        "contrast is not using the cross-task covariance"
    )


# ------------------------------------------------------- refusals


def test_a_contrast_under_the_diagonal_posterior_is_refused(fitted):
    """It would run and report plausible numbers. That is the failure mode."""
    si = A.selectivity_contrast_objective(NUM_OBJ)
    with pytest.raises(ValueError, match="require posterior_mode='joint'"):
        _score(fitted, posterior_mode="diag", linear_objectives=[si])


def test_coordinate_objectives_are_allowed_under_diag(fitted):
    """Coordinates are exact under diag, so the refusal must not over-reach."""
    pf = DOCK[0]
    s = _score(fitted, posterior_mode="diag",
               linear_objectives=A.coordinate_objectives(SIGNS, BOUNDS, [pf, DOCK[1]]))
    assert s.shape == (fitted["Xc"].shape[0],)
    assert np.isfinite(s).all()


def test_emit_and_linear_together_are_refused(fitted):
    si = A.selectivity_contrast_objective(NUM_OBJ)
    with pytest.raises(ValueError, match="not both"):
        _score(fitted, posterior_mode="joint", linear_objectives=[si],
               emit_objectives=[0, 1])


def test_select_batch_forwards_linear_objectives(fitted):
    """select_batch is a second place the value could fail to arrive."""
    pf = DOCK[0]
    si = A.selectivity_contrast_objective(NUM_OBJ)
    pf_axis = A.coordinate_objectives(SIGNS, BOUNDS, [pf])[0]
    with pytest.raises(ValueError, match="require posterior_mode='joint'"):
        torch.manual_seed(SEED)
        A.select_batch(
            fitted["model"], fitted["likelihood"], fitted["y_mean"], fitted["y_std"],
            fitted["Xc"], fitted["ac"], fitted["Xb"], fitted["ab"],
            batch_size=3, objective_signs=SIGNS, bounds=BOUNDS, layout=LAYOUT,
            n_mc_samples=32, partitioning_alpha=1e-3, posterior_mode="diag",
            linear_objectives=[pf_axis, si],
        )
