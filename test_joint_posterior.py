"""
test_joint_posterior.py
=======================

Tests for the JOINT-covariance path added to the model->acquisition interface.

Background. ``acquisition.DockingPosteriorModel.posterior`` historically ended
with ``torch.diag_embed(...)``, handing qNEHVI an explicitly DIAGONAL covariance
over the ``q x k`` docking outputs. GPyTorch computes the full joint block; the
wrapper threw everything off the diagonal away. That deleted

  * the coregionalized (ICM) model's learned PfDHFR<->hDHFR task covariance —
    under this repo's co-located block design that off-diagonal block is the
    ONLY channel through which coregionalization can act (the predictive means
    largely cancel: Bonilla, Chai & Williams 2008 §2.3, "autokrigeability"), and
  * the cross-MOLECULE covariance between a candidate and the baseline points,
    which qNEHVI stacks into one t-batch.

``mogp.predict_joint`` returns that block and ``posterior_mode="joint"`` passes
it through. ``"diag"`` remains the DEFAULT and must stay bit-identical to the
benchmarked code, so the tests below pin both the equivalences (mean, marginal
variance) and the differences (off-diagonal mass).

Runnable as ``pytest test_joint_posterior.py`` or ``python test_joint_posterior.py``.
"""

import numpy as np
import pytest
import torch

from mogp import (
    TASK_NAMES,
    DOCKING_TASK_INDICES,
    OBJECTIVE_SOURCES,
    train_mogp,
    predict,
    predict_joint,
)
from mogp_coregionalized import train_mogp_coregionalized
import acquisition
from acquisition import (
    DockingPosteriorModel,
    POSTERIOR_MODES,
    DEFAULT_POSTERIOR_MODE,
    _unique_rows,
    _augment_with_admet,
    compute_qnehvi,
)

N_FP = 2048
DOCK = list(DOCKING_TASK_INDICES)
LIB = [j for j, n in enumerate(TASK_NAMES) if OBJECTIVE_SOURCES[n][0] != "dock"]
LAYOUT = (DOCK, LIB, list(range(len(LIB))))
K = len(DOCK)


def _fixture(n_baseline=14, n_candidates=25, seed=0):
    """A small correlated-docking training set plus candidates."""
    rng = np.random.default_rng(seed)
    Xb = (rng.random((n_baseline, N_FP)) < 0.05).astype(np.int8)
    Xc = (rng.random((n_candidates, N_FP)) < 0.05).astype(np.int8)
    ba = rng.uniform(0, 1, (n_baseline, len(LIB))).astype(np.float32)
    ca = rng.uniform(0, 1, (n_candidates, len(LIB))).astype(np.float32)
    # Both docking objectives driven by one shared chemical latent, so a
    # coregionalized model has real cross-task correlation to find.
    latent = Xb[:, :100].sum(axis=1).astype(float)
    Y = np.zeros((n_baseline, len(TASK_NAMES)), dtype=np.float32)
    Y[:, DOCK[0]] = -8.0 - 0.05 * latent + rng.normal(0, 0.3, n_baseline)
    Y[:, DOCK[1]] = -7.0 - 0.04 * latent + rng.normal(0, 0.3, n_baseline)
    for j in LIB:
        Y[:, j] = rng.uniform(0, 1, n_baseline)
    return Xb, Xc, ba, ca, Y


def _train(kind, Xb, Y):
    torch.manual_seed(0)
    if kind == "coregionalized":
        return train_mogp_coregionalized(Xb, Y, n_iterations=60, rank=1)
    return train_mogp(Xb, Y, n_iterations=60)


BOTH_MODELS = pytest.mark.parametrize("kind", ["independent", "coregionalized"])


# --------------------------------------------------------------------------- #
# _unique_rows — the dedup the joint path relies on for its memory bound
# --------------------------------------------------------------------------- #
def test_unique_rows_roundtrips_exactly():
    Xb, Xc, *_ = _fixture()
    # Mimic what qNEHVI produces: the same baseline block repeated per t-batch.
    rows = np.vstack([Xb, Xb, Xc[:4], Xb[:2]]).astype(np.float64)
    uniq, inverse = _unique_rows(rows)
    assert np.array_equal(uniq[inverse], rows)
    assert len(uniq) == len(Xb) + 4     # the repeats really collapsed


def test_unique_rows_exact_for_non_binary_input():
    """The bit-packing fast path is only valid for 0/1 rows; the fallback isn't."""
    rng = np.random.default_rng(1)
    rows = rng.random((12, 8))
    rows[5] = rows[2]
    uniq, inverse = _unique_rows(rows)
    assert np.array_equal(uniq[inverse], rows)
    assert len(uniq) == 11


# --------------------------------------------------------------------------- #
# predict_joint — the diagonal of the joint block IS predict's variance
# --------------------------------------------------------------------------- #
@BOTH_MODELS
def test_predict_joint_diagonal_matches_predict(kind):
    Xb, Xc, _, _, Y = _fixture()
    model, lik, ym, ys = _train(kind, Xb, Y)

    mean_marginal, var_marginal = predict(model, lik, ym, ys, Xc)
    mean_joint, cov = predict_joint(model, lik, ym, ys, Xc, task_indices=DOCK)

    assert cov.shape == (len(Xc) * K, len(Xc) * K)
    np.testing.assert_allclose(mean_joint, mean_marginal, equal_nan=True)
    # Interleaved layout: the flat slot of (molecule i, task a) is i*K + a.
    diagonal = np.diag(cov).reshape(len(Xc), K)
    np.testing.assert_allclose(diagonal, var_marginal[:, DOCK], rtol=1e-6)


@BOTH_MODELS
def test_predict_joint_is_a_valid_covariance(kind):
    Xb, Xc, _, _, Y = _fixture()
    model, lik, ym, ys = _train(kind, Xb, Y)
    _, cov = predict_joint(model, lik, ym, ys, Xc, task_indices=DOCK)
    np.testing.assert_array_equal(cov, cov.T)      # symmetrized exactly
    eigenvalues = np.linalg.eigvalsh(cov)
    assert eigenvalues.min() > -1e-8 * max(1.0, eigenvalues.max())


def test_independent_model_has_zero_cross_task_covariance():
    """The independent model's task blocks are zero BY CONSTRUCTION."""
    Xb, Xc, _, _, Y = _fixture()
    model, lik, ym, ys = _train("independent", Xb, Y)
    _, cov = predict_joint(model, lik, ym, ys, Xc, task_indices=DOCK)
    cross_task = [cov[i * K + 0, i * K + 1] for i in range(len(Xc))]
    assert np.all(np.asarray(cross_task) == 0.0)


def test_coregionalized_model_has_non_zero_cross_task_covariance():
    """The ICM's whole point: the off-diagonal block the diagonal path deleted."""
    Xb, Xc, _, _, Y = _fixture()
    model, lik, ym, ys = _train("coregionalized", Xb, Y)
    _, cov = predict_joint(model, lik, ym, ys, Xc, task_indices=DOCK)
    cross_task = np.asarray([cov[i * K + 0, i * K + 1] for i in range(len(Xc))])
    assert np.abs(cross_task).max() > 1e-4


@BOTH_MODELS
def test_cross_molecule_covariance_is_non_zero_for_both_models(kind):
    """Cross-MOLECULE covariance exists even without coregionalization, and the
    diagonal path threw it away in both arms of the original ablation."""
    Xb, Xc, _, _, Y = _fixture()
    model, lik, ym, ys = _train(kind, Xb, Y)
    _, cov = predict_joint(model, lik, ym, ys, Xc, task_indices=DOCK)
    same_task_other_molecule = np.asarray(
        [cov[i * K, (i + 1) * K] for i in range(len(Xc) - 1)]
    )
    assert np.abs(same_task_other_molecule).max() > 1e-9


def test_predict_joint_rejects_an_unmodelled_task():
    Xb, Xc, _, _, Y = _fixture()
    model, lik, ym, ys = _train("independent", Xb, Y)
    unmodelled = LIB[0]                     # an ADMET objective; the GP skips it
    with pytest.raises(ValueError, match="not.*modelled"):
        predict_joint(model, lik, ym, ys, Xc, task_indices=[unmodelled])


def test_predict_joint_task_subset_preserves_layout():
    """Asking for one task must return that task's own marginal block."""
    Xb, Xc, _, _, Y = _fixture()
    model, lik, ym, ys = _train("coregionalized", Xb, Y)
    _, cov_both = predict_joint(model, lik, ym, ys, Xc, task_indices=DOCK)
    _, cov_one = predict_joint(model, lik, ym, ys, Xc, task_indices=[DOCK[1]])
    assert cov_one.shape == (len(Xc), len(Xc))
    expected = cov_both[1::K, 1::K]
    np.testing.assert_allclose(cov_one, expected, rtol=1e-10)


# --------------------------------------------------------------------------- #
# DockingPosteriorModel — diag vs joint on qNEHVI's real tensor shape
# --------------------------------------------------------------------------- #
def _qnehvi_shaped_X(Xb, Xc, ba, ca, n_tbatch=6):
    """(t-batch, n_baseline + 1, d): what qNEHVI actually calls posterior with."""
    baseline = torch.as_tensor(_augment_with_admet(Xb, ba, LAYOUT[2]),
                               dtype=torch.double)
    candidates = torch.as_tensor(_augment_with_admet(Xc, ca, LAYOUT[2]),
                                 dtype=torch.double)
    return torch.cat(
        [baseline.expand(n_tbatch, len(Xb), -1),
         candidates[:n_tbatch].unsqueeze(1)], dim=-2
    )


@BOTH_MODELS
def test_posterior_modes_agree_on_mean_and_marginal_variance(kind):
    """joint must differ from diag ONLY in the off-diagonal entries."""
    Xb, Xc, ba, ca, Y = _fixture()
    model, lik, ym, ys = _train(kind, Xb, Y)
    X = _qnehvi_shaped_X(Xb, Xc, ba, ca)

    args = (model, lik, ym, ys, N_FP, DOCK)
    diag = DockingPosteriorModel(*args, posterior_mode="diag").posterior(X)
    joint = DockingPosteriorModel(*args, posterior_mode="joint").posterior(X)

    torch.testing.assert_close(diag.mean, joint.mean, rtol=0, atol=1e-9)
    torch.testing.assert_close(diag.variance, joint.variance, rtol=1e-6, atol=1e-12)


@BOTH_MODELS
def test_diag_mode_covariance_is_exactly_diagonal(kind):
    Xb, Xc, ba, ca, Y = _fixture()
    model, lik, ym, ys = _train(kind, Xb, Y)
    X = _qnehvi_shaped_X(Xb, Xc, ba, ca)
    post = DockingPosteriorModel(model, lik, ym, ys, N_FP, DOCK,
                                 posterior_mode="diag").posterior(X)
    covariance = post.distribution.covariance_matrix
    off = covariance - torch.diag_embed(
        torch.diagonal(covariance, dim1=-2, dim2=-1))
    assert off.abs().max().item() == 0.0


@BOTH_MODELS
def test_joint_mode_covariance_has_off_diagonal_mass(kind):
    Xb, Xc, ba, ca, Y = _fixture()
    model, lik, ym, ys = _train(kind, Xb, Y)
    X = _qnehvi_shaped_X(Xb, Xc, ba, ca)
    post = DockingPosteriorModel(model, lik, ym, ys, N_FP, DOCK,
                                 posterior_mode="joint").posterior(X)
    covariance = post.distribution.covariance_matrix
    off = covariance - torch.diag_embed(
        torch.diagonal(covariance, dim1=-2, dim2=-1))
    assert off.abs().max().item() > 1e-8


def test_joint_mode_recovers_the_icm_task_correlation():
    """The learned IndexKernel correlation must actually reach the posterior.

    This is the specific quantity ``diag_embed`` deleted, so it is worth pinning
    against the model's own ``task_covariance_matrix()`` rather than merely
    asserting "non-zero somewhere".
    """
    Xb, Xc, ba, ca, Y = _fixture()
    model, lik, ym, ys = _train("coregionalized", Xb, Y)
    B = model.task_covariance_matrix()
    learned = B[0, 1] / np.sqrt(B[0, 0] * B[1, 1])

    X = _qnehvi_shaped_X(Xb, Xc, ba, ca)
    post = DockingPosteriorModel(model, lik, ym, ys, N_FP, DOCK,
                                 posterior_mode="joint").posterior(X)
    covariance = post.distribution.covariance_matrix[0]
    # Last molecule in the t-batch is the candidate; its own 2x2 task block.
    last = covariance.shape[-1] - 2
    block = covariance[last:last + 2, last:last + 2].numpy()
    posterior_rho = block[0, 1] / np.sqrt(block[0, 0] * block[1, 1])
    assert np.sign(posterior_rho) == np.sign(learned)
    assert abs(posterior_rho) > 0.05


# --------------------------------------------------------------------------- #
# Flag hygiene — the default must remain the benchmarked path
# --------------------------------------------------------------------------- #
def test_default_posterior_mode_is_diag():
    """Every published number was produced on the diagonal path."""
    assert DEFAULT_POSTERIOR_MODE == "diag"
    assert set(POSTERIOR_MODES) == {"diag", "joint"}
    Xb, Xc, ba, ca, Y = _fixture()
    model, lik, ym, ys = _train("independent", Xb, Y)
    wrapper = DockingPosteriorModel(model, lik, ym, ys, N_FP, DOCK)
    assert wrapper._posterior_mode == "diag"


def test_unknown_posterior_mode_raises():
    Xb, _, _, _, Y = _fixture()
    model, lik, ym, ys = _train("independent", Xb, Y)
    with pytest.raises(ValueError, match="Unknown posterior mode"):
        DockingPosteriorModel(model, lik, ym, ys, N_FP, DOCK,
                              posterior_mode="full")


def test_compute_qnehvi_default_equals_explicit_diag():
    """Omitting the flag must be bit-identical to asking for 'diag'.

    ``SobolQMCNormalSampler`` draws its seed from torch's GLOBAL RNG when none is
    given, so two ``compute_qnehvi`` calls in one process use different Sobol
    draws and never match exactly. That is pre-existing behaviour (a whole BO run
    is reproducible because ``BOLoop.__init__`` seeds torch once, and the call
    sequence is then fixed), but it means an in-process A/B needs the global seed
    reset before each call.
    """
    Xb, Xc, ba, ca, Y = _fixture()
    model, lik, ym, ys = _train("coregionalized", Xb, Y)
    torch.manual_seed(1234)
    default = compute_qnehvi(model, lik, ym, ys, Xc, ca, Xb, ba, layout=LAYOUT)
    torch.manual_seed(1234)
    explicit = compute_qnehvi(model, lik, ym, ys, Xc, ca, Xb, ba, layout=LAYOUT,
                              posterior_mode="diag")
    np.testing.assert_array_equal(default, explicit)


def test_qnehvi_sampler_seed_comes_from_the_global_torch_rng():
    """Pin the reproducibility contract the equality check above relies on.

    If BoTorch ever started seeding the sampler independently of torch's global
    RNG, whole-run reproducibility would break silently, and so would every
    diag-vs-joint comparison that assumes identical Sobol draws.
    """
    Xb, Xc, ba, ca, Y = _fixture()
    model, lik, ym, ys = _train("independent", Xb, Y)
    kw = dict(layout=LAYOUT, posterior_mode="diag")
    torch.manual_seed(7)
    first = compute_qnehvi(model, lik, ym, ys, Xc, ca, Xb, ba, **kw)
    torch.manual_seed(7)
    again = compute_qnehvi(model, lik, ym, ys, Xc, ca, Xb, ba, **kw)
    np.testing.assert_array_equal(first, again)


@BOTH_MODELS
def test_compute_qnehvi_joint_is_finite_and_ranks_candidates(kind):
    Xb, Xc, ba, ca, Y = _fixture()
    model, lik, ym, ys = _train(kind, Xb, Y)
    scores = compute_qnehvi(model, lik, ym, ys, Xc, ca, Xb, ba, layout=LAYOUT,
                            posterior_mode="joint")
    assert scores.shape == (len(Xc),)
    assert np.isfinite(scores).all()
    assert scores.std() > 0


@BOTH_MODELS
def test_joint_scores_survive_chunking(kind):
    """Chunk size changes the Sobol draws, not the posterior; the ranking should
    stay close. Guards against the per-t-batch block gather being mis-indexed,
    which would scramble candidates against each other as the chunking changed."""
    Xb, Xc, ba, ca, Y = _fixture(n_candidates=40)
    model, lik, ym, ys = _train(kind, Xb, Y)
    kwargs = dict(layout=LAYOUT, posterior_mode="joint")
    torch.manual_seed(3)
    a = compute_qnehvi(model, lik, ym, ys, Xc, ca, Xb, ba,
                       candidate_chunk=40, **kwargs)
    torch.manual_seed(3)
    b = compute_qnehvi(model, lik, ym, ys, Xc, ca, Xb, ba,
                       candidate_chunk=7, **kwargs)
    assert np.corrcoef(a, b)[0, 1] > 0.99


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
