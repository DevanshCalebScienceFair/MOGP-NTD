"""Do the two ICM implementations agree when the data is COMPLETE?

`mogp_coregionalized.MOGPCoregionalized` (Kronecker) and
`mogp_hadamard.MOGPHadamardICM` (stacked index) are the same intrinsic
coregionalization model written two ways. On a complete (N, K) matrix they
should behave alike; the Hadamard form additionally accepts gaps.

They are NOT expected to be numerically identical:
  * Kronecker uses MultitaskGaussianLikelihood -> one noise PER TASK.
  * Hadamard flattens to a vector, so GaussianLikelihood gives ONE SHARED noise.
Targets are standardized per task first, which puts both on unit variance and
makes the shared noise defensible, but it is a real difference and these tests
pin down how large it is rather than assuming it away.
"""
import warnings

import numpy as np
import pytest
import torch

from mogp import TASK_NAMES, DOCKING_TASK_INDICES
from mogp_coregionalized import train_mogp_coregionalized, predict_coregionalized
from mogp_hadamard import train_mogp_hadamard, predict_hadamard

PF, HD = DOCKING_TASK_INDICES
ITERS = 120


def _data(n=70, corr=0.85, seed=0, bits=64):
    g = np.random.default_rng(seed)
    X = (g.random((n, bits)) < 0.15).astype(np.float32)
    latent = X @ g.normal(size=(bits,))
    Y = np.full((n, len(TASK_NAMES)), np.nan, dtype=np.float32)
    Y[:, PF] = -8.0 + latent
    Y[:, HD] = -7.0 + corr * latent + np.sqrt(1 - corr ** 2) * g.normal(size=n)
    return X, Y


@pytest.fixture(scope="module")
def fitted():
    warnings.simplefilter("ignore")
    torch.manual_seed(0)
    X, Y = _data()
    Xt, _ = _data(n=25, seed=99)
    k = train_mogp_coregionalized(X, Y, n_iterations=ITERS)
    h = train_mogp_hadamard(X, Y, n_iterations=ITERS, verbose=False)
    return (Xt,
            predict_coregionalized(*k, Xt),
            predict_hadamard(*h, Xt),
            k[0].task_covariance_matrix(),
            h[0].task_covariance_matrix())


def test_both_recover_the_same_task_correlation(fitted):
    """The one number coregionalization is FOR must agree between the two forms."""
    _, _, _, Bk, Bh = fitted
    rk = Bk[0, 1] / np.sqrt(Bk[0, 0] * Bk[1, 1])
    rh = Bh[0, 1] / np.sqrt(Bh[0, 0] * Bh[1, 1])
    assert abs(rk) > 0.3 and abs(rh) > 0.3, f"kron {rk:.3f}, hadamard {rh:.3f}"
    assert abs(abs(rk) - abs(rh)) < 0.35, (
        f"task correlations disagree: Kronecker {rk:.3f} vs Hadamard {rh:.3f}"
    )


def test_posterior_means_agree_closely(fitted):
    """Predictions should track each other; this pins down by how much."""
    _, (mk, _), (mh, _), _, _ = fitted
    for j, name in ((PF, "PfDHFR"), (HD, "hDHFR")):
        a, b = mk[:, j], mh[:, j]
        r = np.corrcoef(a, b)[0, 1]
        rel = np.abs(a - b).mean() / (np.abs(a).mean() + 1e-9)
        assert r > 0.95, f"{name}: correlation between the two forms is only {r:.3f}"
        assert rel < 0.10, f"{name}: mean relative difference {rel:.3%} is too large"


def test_hadamard_variances_are_positive_and_comparable(fitted):
    """The shared-noise simplification must not blow up or collapse uncertainty."""
    _, (_, vk), (_, vh), _, _ = fitted
    for j in (PF, HD):
        assert (vh[:, j] > 0).all()
        ratio = vh[:, j].mean() / vk[:, j].mean()
        assert 0.2 < ratio < 5.0, (
            f"task {j}: Hadamard variance is {ratio:.2f}x the Kronecker one"
        )


def test_only_hadamard_survives_a_gap():
    """The whole reason the rewrite exists."""
    warnings.simplefilter("ignore")
    X, Y = _data(n=40)
    Y[::2, HD] = np.nan
    m, lik, mu, sd = train_mogp_hadamard(X, Y, n_iterations=20, verbose=False)
    mean, _ = predict_hadamard(m, lik, mu, sd, X[:5])
    assert np.isfinite(mean[:, [PF, HD]]).all()


def test_kronecker_refuses_a_partial_column_instead_of_degrading_silently():
    """It used to accept the gap and quietly return a ONE-task model.

    Measured before the guard: 20/40 NaN in hDHFR gave a 1x1 task covariance
    ([[0.69]]) and an all-NaN hDHFR prediction column, with no error raised.
    That is the worst failure mode -- a run completes and reports nothing.
    """
    warnings.simplefilter("ignore")
    X, Y = _data(n=40)
    Y[::2, HD] = np.nan
    with pytest.raises(ValueError, match="PARTIALLY observed"):
        train_mogp_coregionalized(X, Y, n_iterations=5)


def test_kronecker_still_accepts_a_fully_inactive_task():
    """An all-NaN column is a task that has not been measured yet, not a gap."""
    warnings.simplefilter("ignore")
    X, Y = _data(n=30)
    Y[:, HD] = np.nan
    m, lik, mu, sd = train_mogp_coregionalized(X, Y, n_iterations=5)
    assert np.isfinite(mu[PF]) and np.isnan(mu[HD])
