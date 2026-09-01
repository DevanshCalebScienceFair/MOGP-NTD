"""The missing-data ICM must accept partial labels and stay faithful to the ICM."""
import numpy as np
import pytest
import torch

from mogp import TASK_NAMES, DOCKING_TASK_INDICES
from mogp_hadamard import (
    train_mogp_hadamard, predict_hadamard, predict_joint_hadamard,
)

RNG = np.random.default_rng(0)
PF, HD = DOCKING_TASK_INDICES[0], DOCKING_TASK_INDICES[1]


def _fp(n, bits=64, seed=0):
    g = np.random.default_rng(seed)
    return (g.random((n, bits)) < 0.15).astype(np.float32)


def _targets(X, corr=0.9, seed=0):
    """Two correlated docking columns driven by the same latent fingerprint signal."""
    g = np.random.default_rng(seed)
    latent = X @ g.normal(size=(X.shape[1],))
    Y = np.full((X.shape[0], len(TASK_NAMES)), np.nan, dtype=np.float32)
    Y[:, PF] = -8.0 + latent
    Y[:, HD] = -7.0 + corr * latent + np.sqrt(1 - corr**2) * g.normal(size=len(X))
    return Y


def test_trains_with_a_complete_matrix():
    X = _fp(30); Y = _targets(X)
    m, lik, mu, sd = train_mogp_hadamard(X, Y, n_iterations=15, verbose=False)
    assert np.isfinite(mu[PF]) and np.isfinite(mu[HD])
    assert np.isnan(mu[[j for j in range(len(TASK_NAMES)) if j not in (PF, HD)]]).all()


def test_trains_when_half_the_hdhfr_labels_are_missing():
    """The whole point: a molecule may carry one task and not the other."""
    X = _fp(40); Y = _targets(X)
    Y[::2, HD] = np.nan
    m, lik, mu, sd = train_mogp_hadamard(X, Y, n_iterations=15, verbose=False)
    mean, var = predict_hadamard(m, lik, mu, sd, _fp(7, seed=3))
    assert mean.shape == (7, len(TASK_NAMES))
    assert np.isfinite(mean[:, [PF, HD]]).all()
    assert (var[:, [PF, HD]] > 0).all()


def test_standardization_uses_only_observed_entries():
    """A task's mean must not be polluted by the rows where it is missing."""
    X = _fp(20); Y = _targets(X)
    Y[5:, HD] = np.nan
    _, _, mu, _ = train_mogp_hadamard(X, Y, n_iterations=2, verbose=False)
    assert mu[HD] == pytest.approx(float(np.mean(Y[:5, HD])), rel=1e-5)


def test_molecules_with_no_observations_are_dropped_not_fatal():
    X = _fp(25); Y = _targets(X)
    Y[3, PF] = np.nan; Y[3, HD] = np.nan          # a failed dock on both targets
    m, lik, mu, sd = train_mogp_hadamard(X, Y, n_iterations=5, verbose=False)
    assert np.isfinite(mu[PF]) and np.isfinite(mu[HD])


def test_task_with_no_observations_raises():
    X = _fp(15); Y = _targets(X)
    Y[:, HD] = np.nan
    with pytest.raises(ValueError, match="no observed values"):
        train_mogp_hadamard(X, Y, n_iterations=2, verbose=False)


def test_learns_a_NON_diagonal_task_covariance():
    """If the off-diagonal were zero this would be independent GPs, not an ICM."""
    X = _fp(60); Y = _targets(X, corr=0.95)
    m, _, _, _ = train_mogp_hadamard(X, Y, n_iterations=120, verbose=False)
    B = m.task_covariance_matrix()
    assert B.shape == (2, 2)
    rho = B[0, 1] / np.sqrt(B[0, 0] * B[1, 1])
    assert abs(rho) > 0.3, f"task correlation {rho:.3f} is too weak to be an ICM"


def test_joint_covariance_is_interleaved_and_matches_marginals():
    """predict_joint's diagonal must equal predict's variances, task-fastest."""
    X = _fp(35); Y = _targets(X)
    Y[::3, HD] = np.nan
    m, lik, mu, sd = train_mogp_hadamard(X, Y, n_iterations=15, verbose=False)
    Xn = _fp(5, seed=9)
    mean_m, var_m = predict_hadamard(m, lik, mu, sd, Xn)
    mean_j, cov = predict_joint_hadamard(m, lik, mu, sd, Xn)
    assert cov.shape == (5 * 2, 5 * 2)
    np.testing.assert_allclose(mean_m[:, [PF, HD]], mean_j[:, [PF, HD]], rtol=1e-5)
    d = np.diag(cov).reshape(5, 2)          # task varies fastest
    np.testing.assert_allclose(d[:, 0], var_m[:, PF], rtol=1e-4)
    np.testing.assert_allclose(d[:, 1], var_m[:, HD], rtol=1e-4)


def test_joint_covariance_is_symmetric_psd():
    X = _fp(30); Y = _targets(X)
    m, lik, mu, sd = train_mogp_hadamard(X, Y, n_iterations=15, verbose=False)
    _, cov = predict_joint_hadamard(m, lik, mu, sd, _fp(6, seed=11))
    np.testing.assert_allclose(cov, cov.T, atol=1e-6)
    w = np.linalg.eigvalsh(0.5 * (cov + cov.T))
    assert w.min() > -1e-6, f"most negative eigenvalue {w.min():.2e}"
