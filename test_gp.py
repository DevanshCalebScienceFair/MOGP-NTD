"""End-to-end test: build, train and predict from the project's Tanimoto GP.

Verifies that ``TanimotoKernel`` works inside the repo's exact-GP stack: the GP
model constructs over it, the training covariance factorizes (Cholesky), the
gradient the fit descends is the true gradient of the loss, the loss falls and
stays finite, predictions are finite with positive variance, and the kernel's
``diag`` path matches the diagonal of the full kernel.  This checks mechanical
correctness, not predictive quality (the target is synthetic and the set is
tiny).

WHY THIS FILE WAS REWRITTEN (2026-09-28)
----------------------------------------
The subject is unchanged; the stack under it moved.  The previous version of
this file made the same claims against
``gpytorch.models.ExactGP`` + ``gpytorch.kernels.ScaleKernel(TanimotoKernel())``
+ ``gpytorch.likelihoods.GaussianLikelihood`` + ``torch.optim.Adam``.  The BO
core is now pure numpy (``mogp_core``), ``kernel.TanimotoKernel`` is no longer a
``gpytorch.kernels.Kernel`` subclass (it declared no ``__init__`` of its own and
so inherited gpytorch's ``(ard_num_dims, batch_shape, active_dims,
lengthscale_prior, lengthscale_constraint, **kwargs)``, which has narrowed to
``()``, and with it every attribute ``ScaleKernel`` reads off a base kernel).  So
``ScaleKernel(TanimotoKernel())`` raised ``AttributeError: 'TanimotoKernel'
object has no attribute 'active_dims'``.

Two things changed beyond the import lines, and both are recorded here rather
than quietly absorbed:

* **These are now real test functions.**  The old file had ZERO ``def test_``
  functions — its whole body, asserts included, ran at module scope, i.e. at
  pytest COLLECTION time.  It therefore contributed 0 to the suite's test count,
  and when it started raising it aborted the entire pytest session (a collection
  error is not a test failure) rather than reporting one red test.  The same
  assertions are now seven collected tests.
* **"autograd flows through the kernel" is now "the analytic gradient is
  correct".**  The old file's evidence that gradients reach the kernel was that
  ``loss.backward()`` ran and the loss then fell.  ``mogp_core.gp`` has no
  autograd: ``nmll_and_grad`` returns a hand-derived raw-space gradient, and the
  fit descends exactly that.  The equivalent — strictly stronger — property is
  that this gradient agrees with a central finite difference of the loss, which
  is what :func:`test_analytic_gradient_matches_finite_differences` asserts.
  ``test_training_loss_is_finite_and_decreases`` keeps the old two asserts
  verbatim.

Deliberately NOT carried over: the old single-task gpytorch model had 3 raw
scalars (``raw_outputscale``, ``raw_constant``, ``raw_noise``) where
``mogp_core.gp`` has ``3T + 1 = 4`` (the noise is split into a per-task ``d_t``
and a shared ``sigma2``, which are not separately identifiable — see
``mogp_core/gp.py`` §6.1).  So the loss VALUES here are not comparable with the
old run's, and nothing in this file compares them: every assertion below is
about finiteness, positivity, monotone decrease or an exact identity, exactly as
before.

Runnable both as ``pytest test_gp.py`` and as ``python test_gp.py``.
"""

import functools

import numpy as np

import mogp
from kernel import TanimotoKernel
from mogp_core import gp as core_gp
from utils.featurize import smiles_to_morgan

# The old file's training schedule, kept verbatim: 50 Adam steps at lr=0.1.
N_STEPS = 50
LR = 0.1

# A small, structurally varied molecule set — the same eight, in the same order.
SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",       # aspirin
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
    "CC(=O)Nc1ccc(O)cc1",          # paracetamol
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",# caffeine
    "c1ccccc1",                    # benzene
    "CCO",                         # ethanol
    "C1CCCCC1",                    # cyclohexane
    "CC(=O)O",                     # acetic acid
]

# Hold out the last two molecules, as before.
N_TRAIN = 6


# ---------------------------------------------------------------------- #
# Fixtures (plain cached helpers, so `python test_gp.py` works too)
# ---------------------------------------------------------------------- #
@functools.lru_cache(maxsize=None)
def _dataset():
    """``(X, y, train_x, Y_norm, test_x)`` — the old file's data, unchanged.

    ``X`` is the raw int8 fingerprint matrix (what the kernel is asked about);
    ``train_x`` is its float32 view, as the old ``torch.from_numpy(fps).to(
    torch.float32)`` produced.  The synthetic target is the same deterministic
    function of the fingerprints (on-bit count, standardized) so there is real
    signal to learn; ``ddof=1`` reproduces ``torch.Tensor.std``'s unbiased
    default, which is what built the old target (numpy's default is ddof=0).
    """
    fps = np.vstack([smiles_to_morgan(s) for s in SMILES])
    X = fps.astype(np.float32)
    y = X.sum(axis=1)
    y = (y - y.mean()) / y.std(ddof=1)
    # (N, T) with T = 1: mogp_core.gp is multi-output and a single task is the
    # T = 1 case of the same code (mogp_core/gp.py §10.1 item 4).
    Y_norm = y[:N_TRAIN].reshape(-1, 1).astype(np.float64)
    return fps, y, X[:N_TRAIN], Y_norm, X[N_TRAIN:]


@functools.lru_cache(maxsize=None)
def _fit():
    """``(theta, losses, S)`` from 50 Adam steps at lr=0.1 on the training set."""
    _fps, _y, train_x, Y_norm, _test_x = _dataset()
    S = core_gp.gram(train_x, train_x, dtype=np.float64)
    theta, losses = core_gp.fit(train_x, Y_norm, n_steps=N_STEPS, lr=LR, S=S)
    return theta, losses, S


# ---------------------------------------------------------------------- #
# The GP model constructs over the Tanimoto kernel
# ---------------------------------------------------------------------- #
def test_gp_model_constructs_over_the_tanimoto_kernel():
    """``mogp.MOGPModel`` builds and its prior covariance IS the Tanimoto Gram.

    Replaces the old ``TanimotoGP(train_x, train_y, likelihood)`` construction
    plus ``gpytorch.kernels.ScaleKernel(TanimotoKernel())``: the assertion that
    the kernel is usable as a GP's covariance module is now the assertion that
    the model's prior covariance equals ``outputscale * Tanimoto(X, X)``.
    """
    _fps, _y, train_x, Y_norm, _test_x = _dataset()

    model = mogp.MOGPModel(train_x, Y_norm.astype(np.float32), None)
    assert model.num_tasks == 1
    # 3T + 1 raw scalars, all zero at construction (the gpytorch defaults).
    assert model.theta.shape == (core_gp.n_params(1),) == (4,)
    assert np.all(model.theta == 0.0)

    mean_x, covar_x = model.forward(train_x)
    assert mean_x.shape == (1, N_TRAIN)
    assert covar_x.shape == (1, N_TRAIN, N_TRAIN)
    assert np.isfinite(covar_x).all(), "prior covariance has non-finite values"

    s, mu, _d, _sigma2, _tau = model.natural()
    S = core_gp.gram(train_x, train_x, dtype=np.float64)
    assert np.array_equal(covar_x[0], s[0] * S), (
        "the model's prior covariance is not the scaled Tanimoto Gram")
    assert np.array_equal(mean_x[0], np.full(N_TRAIN, mu[0]))


def test_kernel_is_the_gp_stacks_covariance():
    """``kernel.TanimotoKernel`` and the GP's own Gram are the same numbers.

    The old file put the live kernel object INSIDE the model, so "the model uses
    this kernel" was structural.  The numpy stack calls ``mogp_core.gp.gram``
    instead, so the link is asserted numerically: bitwise equality at float32,
    the precision the kernel returns.
    """
    fps, _y, _train_x, _Y_norm, _test_x = _dataset()

    K_kernel = np.asarray(TanimotoKernel().forward(fps, fps))
    K_gp = np.asarray(core_gp.gram(fps, fps, dtype=np.float32, check_binary=False))
    assert K_kernel.dtype == K_gp.dtype == np.float32
    assert np.array_equal(K_kernel, K_gp), (
        "the GP's Gram is not the TanimotoKernel's Gram")
    assert np.allclose(K_kernel, K_kernel.T), "kernel matrix is not symmetric"


# ---------------------------------------------------------------------- #
# The training covariance factorizes (Cholesky)
# ---------------------------------------------------------------------- #
def test_training_covariance_factorizes():
    """``s*K + tau*I`` is positive definite and its Cholesky reconstructs it.

    The old file exercised this implicitly — ``mll(output, train_y)`` cannot be
    evaluated without gpytorch factorizing the covariance — so a failure showed
    up as an exception from ``ExactMarginalLogLikelihood``.  Here the factor is
    taken directly, at the FITTED parameters, and checked.
    """
    theta, _losses, S = _fit()
    s, _mu, _d, _sigma2, tau = core_gp.natural(theta, 1)

    K = s[0] * S + tau[0] * np.eye(N_TRAIN)
    L = core_gp.cholesky(K)                       # raises if not PD
    assert L.shape == (N_TRAIN, N_TRAIN)
    assert np.isfinite(L).all()
    assert np.allclose(np.triu(L, 1), 0.0), "Cholesky factor is not lower triangular"
    assert (np.diag(L) > 0).all(), "Cholesky factor has a non-positive diagonal"
    assert np.allclose(L @ L.T, K, atol=1e-12), "L L^T does not reconstruct K"


# ---------------------------------------------------------------------- #
# Training: the loss is finite and decreases
# ---------------------------------------------------------------------- #
def test_training_loss_is_finite_and_decreases():
    """The old file's two training asserts, unchanged."""
    _theta, losses, _S = _fit()

    assert len(losses) == N_STEPS
    print(f"Loss: start={losses[0]:.4f}  end={losses[-1]:.4f}")
    assert np.isfinite(losses).all(), "non-finite loss encountered during training"
    assert losses[-1] < losses[0], "training loss did not decrease"


def test_analytic_gradient_matches_finite_differences():
    """The gradient the fit descends is the true gradient of the loss.

    This is the numpy stack's version of the old ``loss.backward()``: there is no
    autograd, so instead of asserting that a backward pass ran, assert that the
    hand-derived raw-space gradient ``nmll_and_grad`` returns — the one
    ``mogp_core.gp.fit`` actually steps with — agrees with a central finite
    difference of the loss it claims to differentiate.  A sign error or a missing
    chain-rule factor anywhere in the kernel -> loss path fails here.
    """
    theta, _losses, S = _fit()
    _fps, _y, _train_x, Y_norm, _test_x = _dataset()

    def loss_at(th):
        loss, _ = core_gp.nmll_and_grad(np.asarray(th, dtype=np.float64), S,
                                        Y_norm, want_grad=False)
        return loss

    _loss, grad = core_gp.nmll_and_grad(theta, S, Y_norm)
    assert np.isfinite(grad).all()

    h = 1e-6
    fd = np.zeros_like(grad)
    for i in range(theta.size):
        up, down = theta.copy(), theta.copy()
        up[i] += h
        down[i] -= h
        fd[i] = (loss_at(up) - loss_at(down)) / (2.0 * h)

    print(f"analytic grad: {grad}")
    print(f"finite diff:   {fd}")
    assert np.allclose(grad, fd, rtol=1e-5, atol=1e-8), (
        f"analytic gradient {grad} != finite difference {fd} "
        f"(max |diff| = {np.abs(grad - fd).max():.3e})")


# ---------------------------------------------------------------------- #
# Prediction: finite, with positive variance
# ---------------------------------------------------------------------- #
def test_predictions_are_finite_with_positive_variance():
    """The old file's three prediction asserts, unchanged.

    ``posterior`` returns ``(mean, var_f, var_y)``; ``var_y = var_f + tau`` is
    the noise-inclusive marginal variance the old ``likelihood(model(test_x))``
    produced, so that is the one the old assertions are about.  ``var_f`` — the
    latent variance, which the old file could not see through the likelihood — is
    asserted positive too.
    """
    theta, _losses, _S = _fit()
    _fps, y, train_x, Y_norm, test_x = _dataset()

    mean, var_f, var_y = core_gp.posterior(theta, train_x, Y_norm, test_x)
    assert mean.shape == var_f.shape == var_y.shape == (len(SMILES) - N_TRAIN, 1)

    print(f"Test targets:    {y[N_TRAIN:]}")
    print(f"Pred mean:       {mean.ravel()}")
    print(f"Pred variance:   {var_y.ravel()}")

    assert np.isfinite(mean).all(), "prediction mean has non-finite values"
    assert np.isfinite(var_y).all(), "prediction variance has non-finite values"
    assert (var_y > 0).all(), "prediction variance must be positive"
    assert (var_f > 0).all(), "latent prediction variance must be positive"


# ---------------------------------------------------------------------- #
# The kernel's diag path
# ---------------------------------------------------------------------- #
def test_diag_path_matches_full_diagonal_and_self_similarity_is_one():
    """The old file's last two asserts, unchanged.

    ``TanimotoKernel.forward`` now returns numpy rather than torch, so
    ``torch.allclose`` / ``full.diagonal()`` become ``np.allclose`` /
    ``np.diagonal``; the tolerances and the claims are the originals.
    """
    fps, _y, _train_x, _Y_norm, _test_x = _dataset()

    k = TanimotoKernel()
    full = np.asarray(k.forward(fps, fps))
    diag = np.asarray(k.forward(fps, fps, diag=True))
    assert diag.shape == (len(fps),)
    assert np.allclose(np.diagonal(full), diag, atol=1e-6), "diag path mismatch"
    assert np.allclose(np.diagonal(full), np.ones(len(fps))), "self-sim != 1.0"


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
    print("\nALL END-TO-END CHECKS PASSED" if not failed
          else f"\n{failed} CHECK(S) FAILED")
    sys.exit(1 if failed else 0)
