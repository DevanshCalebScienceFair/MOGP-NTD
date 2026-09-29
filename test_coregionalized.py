"""
test_coregionalized.py
======================

Tests for the coregionalized (ICM) multi-output GP — the project's PRIMARY model.

  * It learns a NON-diagonal task covariance over the two docking objectives on a
    correlated fit (the cross-task structure the independent model forces to zero).
  * It obeys the grey-box contract: trained-task count is 2 (docking only) and it
    is a drop-in for mogp.predict (NaN ADMET columns), so the model-agnostic
    acquisition works unchanged.
  * loop.resolve_train_fn maps --model names to the right train_fn, with
    coregionalized the default/primary.

Runnable both as ``pytest test_coregionalized.py`` and ``python test_coregionalized.py``.

UPDATED FOR THE NUMPY CORE (2026-09-28) — three mechanical changes, no change of
claim:

* ``train_mogp_coregionalized`` now returns a ``mogp_core.icm.ICMState`` where it
  returned a ``mogp_coregionalized.MOGPCoregionalized``, so the "this is the ICM
  and not the independent model" checks are ``isinstance(model, ICMState)``.
  ``ICMState`` carries the same ``task_covariance_matrix()`` method with the same
  return contract, so every assertion ABOUT the task covariance is untouched.
  ``MOGPCoregionalized`` still exists (nothing in the production path constructs
  it); ``GPState`` is imported so the resolve-train-fn test can say explicitly
  that the coregionalized name does NOT hand back the independent model.
* Slot 1 of the 4-tuple is now ``None`` — there is no gpytorch likelihood object,
  the noise lives in ``model.theta`` — so the trained-task count is read off
  ``model.n_tasks`` instead of ``likelihood.num_tasks``. Same number, same claim.
* The ICM's ``W`` / ``v_raw`` initialization is drawn from numpy's **global** RNG
  where ``gpytorch.kernels.IndexKernel`` drew from torch's, so
  ``torch.manual_seed(0)`` no longer pins anything and is now ``np.random.seed(0)``.
  It is only there for determinism: the non-diagonal result is not seed-luck —
  measured ``corr = +0.9939 / +0.9984 / +0.9979`` under seeds 0 / 1 / 12345 and
  ``+0.9965`` unseeded.
"""

import numpy as np
import pytest

from mogp import TASK_NAMES, DOCKING_TASK_INDICES, OBJECTIVE_SOURCES, train_mogp
from mogp import predict as mogp_predict
from mogp_coregionalized import (
    train_mogp_coregionalized,
    predict_coregionalized,
)
from mogp_core.gp import GPState
from mogp_core.icm import ICMState
import loop as loopmod


N_FP = 2048


def _correlated_docking_Y(rng, X, noise=0.25):
    """A (N, 5) target matrix whose two docking columns are strongly correlated
    (both driven by a shared fingerprint-derived latent); ADMET columns are
    independent noise the grey-box GP ignores."""
    w = rng.standard_normal(X.shape[1])
    proj = X.astype(float) @ w
    latent = (proj - proj.mean()) / (proj.std() + 1e-8)
    Y = np.full((X.shape[0], len(TASK_NAMES)), np.nan, dtype=np.float32)
    pf, hd = DOCKING_TASK_INDICES[0], DOCKING_TASK_INDICES[1]
    Y[:, pf] = -8.0 + 2.0 * latent + noise * rng.standard_normal(X.shape[0])
    Y[:, hd] = -8.0 + 1.9 * latent + noise * rng.standard_normal(X.shape[0])
    for j, name in enumerate(TASK_NAMES):
        if OBJECTIVE_SOURCES[name][0] != "dock":
            Y[:, j] = rng.standard_normal(X.shape[0])
    return Y


def test_task_covariance_is_not_diagonal_on_correlated_fit():
    """The learned 2x2 task covariance has a real (positive) off-diagonal term."""
    # The ICM draws W / v_raw off numpy's GLOBAL stream (it was torch's global
    # stream before the numpy core), so this is the seed that pins the fit.
    np.random.seed(0)
    rng = np.random.default_rng(0)
    n = 30
    X = (rng.random((n, N_FP)) < 0.05).astype(np.int8)
    Y = _correlated_docking_Y(rng, X)

    model, likelihood, y_mean, y_std = train_mogp_coregionalized(
        X, Y, n_iterations=150, rank=1
    )
    assert isinstance(model, ICMState)
    assert likelihood is None
    assert int(model.n_tasks) == len(DOCKING_TASK_INDICES) == 2

    B = model.task_covariance_matrix()
    assert B.shape == (2, 2)
    # Off-diagonal is a real fraction of the diagonal -> NOT diagonal.
    d = np.sqrt(np.diag(B))
    corr = float(B[0, 1] / (d[0] * d[1]))
    assert abs(corr) > 0.1, f"task covariance is ~diagonal (corr={corr:+.4f})"
    assert corr > 0.0, f"homologous docking targets should correlate + (got {corr:+.4f})"
    # And the off-diagonal entry itself is materially non-zero vs the diagonal.
    assert abs(B[0, 1]) > 1e-2 * np.sqrt(B[0, 0] * B[1, 1])


def test_coregionalized_is_greybox_dropin():
    """Trains only the 2 docking tasks; mogp.predict returns NaN ADMET columns."""
    rng = np.random.default_rng(1)
    n = 8
    X = (rng.random((n, N_FP)) < 0.05).astype(np.int8)
    Y = np.zeros((n, len(TASK_NAMES)), dtype=np.float32)
    for j, name in enumerate(TASK_NAMES):
        Y[:, j] = (rng.uniform(-11, -5, n) if OBJECTIVE_SOURCES[name][0] == "dock"
                   else rng.uniform(0, 1, n))

    model, likelihood, y_mean, y_std = train_mogp_coregionalized(X, Y, n_iterations=10)
    assert likelihood is None
    assert int(model.n_tasks) == 2
    assert np.where(np.isfinite(y_mean))[0].tolist() == list(DOCKING_TASK_INDICES)

    # The model-agnostic acquisition decoder (mogp.predict) must work on the ICM.
    mean, _ = mogp_predict(model, likelihood, y_mean, y_std, X[:3])
    assert mean.shape == (3, len(TASK_NAMES))
    for j, name in enumerate(TASK_NAMES):
        if OBJECTIVE_SOURCES[name][0] == "dock":
            assert np.isfinite(mean[:, j]).all()
        else:
            assert np.isnan(mean[:, j]).all()

    # predict_coregionalized agrees with mogp.predict on the same trained model.
    mean2, _ = predict_coregionalized(model, likelihood, y_mean, y_std, X[:3])
    assert np.allclose(np.nan_to_num(mean), np.nan_to_num(mean2), atol=1e-5)


def test_resolve_train_fn_maps_model_names():
    """loop.resolve_train_fn maps names to train_fns; coregionalized is default."""
    assert loopmod.DEFAULT_MODEL == "coregionalized"
    assert loopmod.resolve_train_fn("independent") is train_mogp

    coreg_fn = loopmod.resolve_train_fn("coregionalized", rank=1)
    assert callable(coreg_fn)
    # It has train_mogp's calling convention and returns the ICM contract.
    rng = np.random.default_rng(2)
    n = 6
    X = (rng.random((n, N_FP)) < 0.05).astype(np.int8)
    Y = np.zeros((n, len(TASK_NAMES)), dtype=np.float32)
    for j, name in enumerate(TASK_NAMES):
        Y[:, j] = (rng.uniform(-11, -5, n) if OBJECTIVE_SOURCES[name][0] == "dock"
                   else rng.uniform(0, 1, n))
    model, likelihood, y_mean, y_std = coreg_fn(X, Y, n_iterations=5, lr=0.1)
    assert isinstance(model, ICMState)
    # ...and specifically NOT the independent model, which is what a wrong
    # mapping would silently hand back (train_mogp returns a GPState).
    assert not isinstance(model, GPState)
    assert likelihood is None
    assert int(model.n_tasks) == 2

    with pytest.raises(ValueError):
        loopmod.resolve_train_fn("nonsense")


if __name__ == "__main__":
    import sys

    tests = [obj for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failed = 0
    for test in tests:
        try:
            test()
            print(f"PASSED  {test.__name__}")
        except Exception as exc:                             # pragma: no cover
            failed += 1
            print(f"FAILED  {test.__name__}: {exc}")
    sys.exit(1 if failed else 0)
