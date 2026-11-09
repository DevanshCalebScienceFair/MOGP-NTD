"""Tests for the declared-linear-functional objective path.

Two things need pinning, and they are in tension, which is why they are tested
together rather than in separate files:

1. The OFF path must be BIT-IDENTICAL to the historical coordinate behaviour.
   Guardrail 2 of this branch: every change sits behind a flag defaulting to
   current behaviour, proven byte-identical before any long run.
2. The ON path must actually open a channel the old one did not have, namely a
   CONTRAST axis whose variance depends on the posterior cross-task covariance.
   A change that satisfies (1) and not (2) is a silent defeat: it would run,
   report plausible numbers, and measure nothing. Five bugs on this branch had
   exactly that shape.
"""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest
import torch

import acquisition as A
from mogp import TASK_NAMES, DOCKING_TASK_INDICES

NUM_OBJ = len(TASK_NAMES)
BOUNDS = np.array([[-11.0, -5.0], [-11.0, -5.0], [0.0, 1.0],
                   [-7.26, -3.83], [0.376, 64.85]])
SIGNS = list(A.DEFAULT_OBJECTIVE_SIGNS)


def _layout():
    return A._resolve_admet_layout()


def _objective(linear_objectives=None):
    dock, lib, _ = _layout()
    return A.CompositeKnownADMETObjective(
        dock, lib, NUM_OBJ, BOUNDS, SIGNS, linear_objectives=linear_objectives
    )


def _inputs(n=7, n_mc=3, q=4, seed=0, nan_admet=False):
    """Docking samples plus an X whose tail carries the known ADMET values."""
    rng = np.random.default_rng(seed)
    dock, lib, _ = _layout()
    n_admet = len(lib)
    samples = torch.as_tensor(
        rng.uniform(-13.0, -3.0, size=(n_mc, q, len(dock))), dtype=A._DTYPE
    )
    admet = rng.uniform(0.0, 1.0, size=(q, n_admet))
    if nan_admet:
        admet[0, 0] = np.nan
    X = torch.as_tensor(
        np.concatenate([rng.normal(size=(q, 6)), admet], axis=1), dtype=A._DTYPE
    )
    return samples, X


# ---------------------------------------------------------------- OFF path


@pytest.mark.parametrize("keep", [None, [0, 1], [0, 1, 2], [2, 3, 4], [1]])
def test_coordinate_objectives_are_bit_identical_to_emit_only(keep):
    samples, X = _inputs()
    keep_idx = list(range(NUM_OBJ)) if keep is None else keep

    historical = _objective()
    A.emit_only(historical, keep_idx)
    expected = historical(samples, X=X)

    declared = _objective(A.coordinate_objectives(SIGNS, BOUNDS, keep_idx))
    got = declared(samples, X=X)

    assert got.shape == expected.shape
    # Bit-identical, not merely close: a one-hot weight of exactly 1.0 makes the
    # new path `1.0 * x`, which IEEE 754 defines as exactly x.
    assert torch.equal(got, expected), (got - expected).abs().max()


def test_bit_identity_holds_with_a_nan_admet_value():
    """An out-of-domain ADMET value arrives as NaN and must behave identically.

    This is the case a dense `full @ W.T` would break: 0.0 * NaN is NaN, so a
    NaN in one objective would spread to every axis. The support-only sum cannot
    reach a column it has zero weight on.
    """
    samples, X = _inputs(nan_admet=True)
    keep_idx = list(range(NUM_OBJ))

    historical = _objective()
    A.emit_only(historical, keep_idx)
    expected = historical(samples, X=X)

    declared = _objective(A.coordinate_objectives(SIGNS, BOUNDS, keep_idx))
    got = declared(samples, X=X)

    assert torch.equal(torch.isnan(got), torch.isnan(expected))
    m = ~torch.isnan(expected)
    assert torch.equal(got[m], expected[m])


def test_a_nan_in_an_unread_objective_does_not_reach_a_docking_axis():
    """The NaN-isolation property, stated directly rather than via equality."""
    samples, X = _inputs(nan_admet=True)
    pf = DOCKING_TASK_INDICES[0]
    declared = _objective(A.coordinate_objectives(SIGNS, BOUNDS, [pf]))
    got = declared(samples, X=X)
    assert torch.isfinite(got).all(), "a NaN ADMET value leaked into a docking axis"


def test_contrast_axis_is_unaffected_by_a_nan_admet_value():
    samples, X = _inputs(nan_admet=True)
    declared = _objective([A.selectivity_contrast_objective(NUM_OBJ)])
    got = declared(samples, X=X)
    assert torch.isfinite(got).all()


# ---------------------------------------------------------------- ON path


def test_contrast_axis_equals_the_reported_selectivity_index():
    """The optimized axis must be the SAME quantity evaluation.py reports."""
    samples, X = _inputs()
    obj = A.selectivity_contrast_objective(NUM_OBJ)
    declared = _objective([obj])
    got = declared(samples, X=X).squeeze(-1)

    pf_slot, h_slot = 0, 1
    si = samples[..., h_slot] - samples[..., pf_slot]   # hDHFR - PfDHFR
    expected = ((si - obj.lo) / (obj.hi - obj.lo)).clamp(0.0, 1.0)
    assert torch.allclose(got, expected, atol=0.0, rtol=0.0)


def test_contrast_variance_tracks_the_posterior_covariance():
    """The whole point: a contrast axis READS the cross-task covariance.

    Draw correlated and uncorrelated docking samples with identical marginals.
    A coordinate axis cannot tell them apart; the contrast axis must, by the
    measured factor v1+v2 over v1+v2-2cov.
    """
    rng = np.random.default_rng(7)
    n_mc, q = 200_000, 1
    v1 = v2 = 1.0
    rho = 0.717            # the measured posterior cross-task correlation

    indep = rng.multivariate_normal([-9.0, -9.0], [[v1, 0.0], [0.0, v2]], size=(n_mc, q))
    joint = rng.multivariate_normal([-9.0, -9.0], [[v1, rho], [rho, v2]], size=(n_mc, q))

    dock, lib, _ = _layout()
    X = torch.as_tensor(
        np.concatenate([np.zeros((q, 6)), np.full((q, len(lib)), 0.5)], axis=1),
        dtype=A._DTYPE,
    )
    # Wide bounds so nothing clips: clamping would hide the variance difference.
    obj = A.selectivity_contrast_objective(NUM_OBJ, lo=-60.0, hi=60.0)
    declared = _objective([obj])

    def contrast_var(s):
        t = torch.as_tensor(s, dtype=A._DTYPE)
        return float(declared(t, X=X).squeeze(-1).squeeze(-1).var().item())

    var_diag = contrast_var(indep)
    var_joint = contrast_var(joint)
    ratio = var_diag / var_joint

    predicted = (v1 + v2) / (v1 + v2 - 2 * rho)
    assert ratio == pytest.approx(predicted, rel=0.03), (ratio, predicted)
    assert ratio > 2.0, "the contrast axis is not reading the covariance at all"


def test_a_coordinate_axis_cannot_distinguish_correlated_from_independent():
    """The converse, and the reason --posterior joint measured -0.0008.

    With matched marginals, a one-hot axis gives the same variance whether the
    tasks are correlated or not. No amount of covariance reaches it.
    """
    rng = np.random.default_rng(7)
    n_mc, q = 200_000, 1
    indep = rng.multivariate_normal([-9.0, -9.0], [[1.0, 0.0], [0.0, 1.0]], size=(n_mc, q))
    joint = rng.multivariate_normal([-9.0, -9.0], [[1.0, 0.717], [0.717, 1.0]], size=(n_mc, q))

    dock, lib, _ = _layout()
    X = torch.as_tensor(
        np.concatenate([np.zeros((q, 6)), np.full((q, len(lib)), 0.5)], axis=1),
        dtype=A._DTYPE,
    )
    pf = DOCKING_TASK_INDICES[0]
    declared = _objective(A.coordinate_objectives(SIGNS, [[-60.0, 60.0]] * NUM_OBJ, [pf]))

    def axis_var(s):
        t = torch.as_tensor(s, dtype=A._DTYPE)
        return float(declared(t, X=X).squeeze(-1).squeeze(-1).var().item())

    assert axis_var(indep) == pytest.approx(axis_var(joint), rel=0.02)


def test_mixed_axes_stack_in_declaration_order():
    samples, X = _inputs()
    pf, h = DOCKING_TASK_INDICES
    coords = A.coordinate_objectives(SIGNS, BOUNDS, [pf, h])
    si = A.selectivity_contrast_objective(NUM_OBJ)
    declared = _objective([coords[0], si, coords[1]])
    got = declared(samples, X=X)
    assert got.shape[-1] == 3

    only_coords = _objective(coords)
    ref = only_coords(samples, X=X)
    assert torch.equal(got[..., 0], ref[..., 0])
    assert torch.equal(got[..., 2], ref[..., 1])


# ---------------------------------------------------------------- validation


@pytest.mark.parametrize("kwargs,msg", [
    (dict(weights=(0.0, 0.0, 0.0, 0.0, 0.0)), "all zero"),
    (dict(weights=(np.nan, 1.0, 0.0, 0.0, 0.0)), "finite"),
    (dict(sign=0), "sign must be"),
    (dict(lo=1.0, hi=1.0), "hi > lo"),
    (dict(lo=5.0, hi=1.0), "hi > lo"),
    (dict(lo=np.inf), "finite"),
])
def test_linear_objective_rejects_bad_declarations(kwargs, msg):
    base = dict(name="x", weights=(1.0, 0.0, 0.0, 0.0, 0.0), sign=1, lo=0.0, hi=1.0)
    base.update(kwargs)
    with pytest.raises(ValueError, match=msg):
        A.LinearObjective(**base)


def test_weights_must_match_num_objectives():
    bad = A.LinearObjective(name="short", weights=(1.0, 0.0), sign=1, lo=0.0, hi=1.0)
    with pytest.raises(ValueError, match="expected"):
        _objective([bad])


def test_empty_objective_list_is_rejected():
    with pytest.raises(ValueError, match="at least one axis"):
        _objective([])


def test_none_keeps_the_historical_path_reachable():
    obj = _objective(None)
    assert obj.linear_objectives is None
    samples, X = _inputs()
    assert obj(samples, X=X).shape[-1] == NUM_OBJ


def test_is_coordinate_is_strict_about_the_weight_being_exactly_one():
    w = [0.0] * NUM_OBJ
    w[0] = 1.0
    assert A.LinearObjective("a", tuple(w), 1, 0.0, 1.0).is_coordinate
    w[0] = 2.0
    assert not A.LinearObjective("b", tuple(w), 1, 0.0, 1.0).is_coordinate
    w[0] = -1.0
    assert not A.LinearObjective("c", tuple(w), 1, 0.0, 1.0).is_coordinate


def test_selectivity_contrast_bounds_cover_the_observed_campaign_range():
    """Guard against re-censoring. Observed SI over 5,059 campaign molecules
    spans -3.261 to +12.582; the axis must contain that with margin."""
    obj = A.selectivity_contrast_objective(NUM_OBJ)
    assert obj.lo <= -3.261
    assert obj.hi >= 12.582
