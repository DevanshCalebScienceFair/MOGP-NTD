"""The joint posterior's covariance can be marginally indefinite; it must not crash.

Reproduces the failure that killed coregionalized seeds 5 and 6 and independent
seed 6 of the multi-seed sweep:

    torch._C._LinAlgError: linalg.cholesky: (Batch element 68): the leading
    minor of order 497 is not positive-definite

MultitaskMultivariateNormal factorizes eagerly, so a covariance block that is
indefinite only by float rounding raises instead of degrading.
"""
import warnings
import pytest
import torch

from acquisition import _psd_safe_multitask_mvn, _MAX_JITTER_REL

_DTYPE = torch.double


def _psd_block(b, n, seed):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn(b, n, n, generator=g, dtype=_DTYPE)
    return a @ a.transpose(-1, -2) / n


def test_clean_covariance_is_untouched():
    """A healthy block must not be jittered: successful runs stay bit-identical."""
    cov = _psd_block(3, 6, 0) + 1e-3 * torch.eye(6, dtype=_DTYPE)
    mean = torch.zeros(3, 3, 2, dtype=_DTYPE)
    ref = torch.distributions.MultivariateNormal(
        torch.zeros(3, 6, dtype=_DTYPE), covariance_matrix=cov
    ).covariance_matrix
    mvn = _psd_safe_multitask_mvn(mean, cov)
    assert torch.equal(mvn.covariance_matrix, ref)


def test_marginally_indefinite_is_rescued():
    """A block pushed just below PSD by rounding must be recovered, not raised."""
    n = 8
    cov = _psd_block(2, n, 1)
    # Drive the smallest eigenvalue slightly negative, as float error does.
    w, V = torch.linalg.eigh(cov)
    w[:, 0] = -1e-14 * w[:, -1]
    cov = V @ torch.diag_embed(w) @ V.transpose(-1, -2)
    with pytest.raises(torch.linalg.LinAlgError):
        torch.linalg.cholesky(cov)          # confirms the fixture is genuinely bad
    mean = torch.zeros(2, 4, 2, dtype=_DTYPE)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mvn = _psd_safe_multitask_mvn(mean, cov)
    assert mvn.covariance_matrix.shape == (2, n, n)
    assert any("not positive-definite" in str(w_.message) for w_ in caught)


def test_rescue_symmetrizes():
    """Asymmetry alone does NOT trip Cholesky -- it reads one triangle -- so a
    healthy asymmetric block is passed through untouched. But when a block is
    ALSO indefinite and the rescue runs, the result must come back symmetric."""
    n = 6
    healthy = _psd_block(1, n, 2) + 1e-2 * torch.eye(n, dtype=_DTYPE)
    healthy[0, 0, 1] += 1e-3
    mean = torch.zeros(1, 3, 2, dtype=_DTYPE)
    passed_through = _psd_safe_multitask_mvn(mean, healthy)
    assert torch.equal(passed_through.covariance_matrix, healthy), (
        "a factorizable block must not be modified"
    )

    bad = _psd_block(1, n, 4)
    w, V = torch.linalg.eigh(bad)
    w[:, 0] = -1e-14 * w[:, -1]
    bad = V @ torch.diag_embed(w) @ V.transpose(-1, -2)
    bad[0, 0, 1] += 1e-9
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rescued = _psd_safe_multitask_mvn(mean, bad).covariance_matrix
    assert torch.allclose(rescued, rescued.transpose(-1, -2))


def test_hopeless_block_still_raises():
    """A genuinely broken block must NOT be silently papered over."""
    n = 5
    cov = -torch.eye(n, dtype=_DTYPE).expand(1, n, n).clone()
    mean = torch.zeros(1, 1, 5, dtype=_DTYPE)
    with pytest.raises(torch.linalg.LinAlgError, match="not merely"):
        _psd_safe_multitask_mvn(mean, cov)


def test_jitter_is_small_enough_to_be_negligible():
    """The rescue must not materially change the distribution it rescues."""
    n = 8
    cov = _psd_block(1, n, 3)
    w, V = torch.linalg.eigh(cov)
    w[:, 0] = -1e-14 * w[:, -1]
    bad = V @ torch.diag_embed(w) @ V.transpose(-1, -2)
    mean = torch.zeros(1, 4, 2, dtype=_DTYPE)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mvn = _psd_safe_multitask_mvn(mean, bad)
    delta = (mvn.covariance_matrix - bad).abs().max()
    assert delta <= _MAX_JITTER_REL * torch.diagonal(bad, dim1=-2, dim2=-1).mean()
