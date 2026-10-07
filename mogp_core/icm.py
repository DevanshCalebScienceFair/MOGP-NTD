"""Coregionalized (ICM) multi-output exact GP with a Tanimoto kernel — pure numpy.

The numerical backend of ``mogp_coregionalized.py``
(``MOGPCoregionalized`` / ``train_mogp_coregionalized`` / ``predict_coregionalized``),
per ``spec/icm/SPEC.md``.

Provenance.  This module was first written as a behavioural mirror of the
gpytorch ICM the archived campaigns were produced with.  The model structure and
many conventions below (the plain-softplus link and low-rank form of ``B``, the
shared ``sigma2``, the ``1e-4`` noise floors, the ``N(0, 1)`` start for ``W`` and
``v_raw`` and the all-zero start for the rest, the 200-step Adam budget, the
``N*K`` loss divisor, the noise-inclusive predictive variance) were adopted from
that implementation.  Reproducing the archived runs is no longer a goal: the
archived results are superseded.  So where this file says "mirror", "shipped" or
"preserved", read it as a record of how a choice was arrived at, not of the
standard it is held to.  The origin and status of each such choice is tracked in
``PROVENANCE.md`` beside this file
(``ICM-07`` to ``ICM-13``, ``ICM-15``, ``ICM-16``, ``ICM-33``, ``ICM-34``,
``ICM-36``, ``ICM-46``).

This is the project's PRIMARY GP (``loop.DEFAULT_MODEL == "coregionalized"``),
so it is the model the benchmark runs by default.

The model (SPEC §1)
-------------------
``K`` modelled tasks (the two docking objectives; the three ADMET objectives are
known exactly and are not modelled).  One shared, hyperparameter-free Tanimoto
data kernel; one learned dense ``K x K`` task covariance::

    B        = W W^T + diag(softplus(v_raw))            W: (K, R),  v_raw: (K,)
    Sigma    = K_XX (x) B  +  I_N (x) diag(d)
    d_t      = tau_t + sigma2
    tau_t    = 1e-4 + softplus(tau_raw_t)               per-task noise
    sigma2   = 1e-4 + softplus(sigma_raw)               ONE SHARED global noise
    m[i, t]  = c_t                                      one constant per task

giving ``K*R + 3K + 1`` trainable raw scalars — **9** at ``K=2, R=1``, not the
charter's "3T+1" (that number describes the independent model in ``mogp.py``).

The raw vector layout (fixed once, used by every function here, and identical to
``spec/icm/fixtures/06_gradients.json``'s ``layout``) is::

    theta[0            : K*R        ] = W        (row-major, W.reshape(-1))
    theta[K*R          : K*R +   K  ] = v_raw    raw_var           softplus, NO offset
    theta[K*R +   K    : K*R + 2*K  ] = tau_raw  raw_task_noises   1e-4 + softplus
    theta[K*R + 2*K                 ] = sig_raw  raw_noise SHARED  1e-4 + softplus
    theta[K*R + 2*K + 1: K*R + 3*K+1] = c        mean_constant     identity

THE ORDERING — data-major, and the docstring is the thing that is wrong (SPEC §2)
--------------------------------------------------------------------------------
gpytorch's ``MultitaskKernel`` docstring says ``K_TT (x) K_XX``.  The matrix it
assembles is ``K_XX (x) B``.  Measured on the spec's asymmetric fixture: the
assembled ``6x6`` matches ``kron(K_data, B_task)`` to ``4.44e-16`` and differs
from ``kron(B_task, K_data)`` by ``3.0``.  So::

    flat index a of (molecule i, task t)  =  i*K + t                        (O1)
    Sigma[i*K+t, j*K+s] = K_XX[i,j]*B[t,s] + delta_ij delta_ts d_t          (O2)
    (P (x) Q) vec(V)    = vec(P V Q^T)      for V of shape (N, K)           (O3)
    var[i, t]           = diag(Sigma_post)[i*K + t]                         (O4)

which is plain C-order ``Y.reshape(-1)`` with row = molecule, column = task.
**A transpose here is invisible in every aggregate metric.**  ``kron(K_XX, B)``
and ``kron(B, K_XX)`` are permutation-similar, so the loss, the log-determinant,
the hypervolume — all of them — are bit-identically unchanged by getting it
backwards; only the per-task posterior moves, silently swapping PfDHFR's
prediction with hDHFR's.  That is why :func:`assemble_dense` exists and why
``test_core_icm.py`` checks it against a deliberately asymmetric pair.

Kronecker structure (SPEC §3)
-----------------------------
``Sigma`` is not itself a Kronecker product, so whiten by the noise first
(Rakitsch et al., NIPS 2013 §2) and then one eigendecomposition per factor
diagonalizes it exactly::

    K_XX    = U diag(lam) U^T                                 [N x N, ONCE]    (S1)
    Btil    = D^-1/2 B D^-1/2 = V diag(mu) V^T                [K x K, per step](S2)
    Vt      = D^-1/2 V                        (NOT orthogonal)                 (S3)
    S[p, q] = lam_p * mu_q + 1                (>= 1 always)                    (S4)
    Sigma   = (U (x) D^1/2 V) diag(vec S) (U (x) D^1/2 V)^T
    Sigma^-1= (U (x) Vt)     diag(vec S)^-1 (U (x) Vt)^T

``S >= 1 > 0`` and ``d_t >= 2e-4 > 0``, so **Sigma is positive definite at every
admissible parameter value**: no jitter is needed and none is added (adding it
would change the model).  ``K_XX`` carries no hyperparameters, so ``(lam, U)`` is
cached with the training set — that caching, not the Kronecker algebra, is the
entire measured speedup (SPEC §3.4: at ``K=2`` the factorization alone buys
between ``0.68x`` and ``1.79x``; the caching buys ``7x`` at ``N=40`` and ``184x``
at ``N=600``).

Behaviours inherited from the gpytorch implementation, kept as they were
-----------------------------------------------------------------------
* **The MLL divisor is ``N*K``, not ``N``** (SPEC §4.1, D7; ``ICM-16``).  Every
  loss value and every gradient carries that factor, as the gpytorch model's did.
  There are no priors on any parameter.
* **The ``1e-4`` noise floors are load-bearing, not a quirk** (SPEC D5).  The
  constrained MLE sits ON them: without the floors ``tau_PfDHFR`` and ``sigma2``
  run to zero, ``Sigma`` goes singular and :func:`prepare`'s ``1/sqrt(d)``
  divides by zero.  That a floor is needed is a property of the fit; the value
  ``1e-4`` is gpytorch's default, not derived from docking-score repeatability
  (``ICM-10``).  ``raw_var`` by contrast uses plain ``softplus`` with **no**
  offset (``Positive()``, not ``GreaterThan(1e-4)``).
* **``W`` and ``v_raw`` are redrawn at every construction** (SPEC §6), so every
  BO iteration's refit starts from a fresh ``randn``.  gpytorch drew them from
  torch's global RNG; ``BOLoop`` now passes an explicit ``Generator`` keyed by
  ``(run_seed, iteration, "icm_init")`` (``acquisition.keyed_generator``), so a
  refit's start is a pure function of that key and the loop never touches the
  global stream.  :func:`init_theta` with ``rng=None`` keeps numpy's **ambient**
  global RNG as a documented fallback for ad-hoc callers that pass no key.
* **200 Adam steps, unconditionally**, ``lr=0.1``, no schedule, no early stop.
  SPEC §7.3 shows this is *acting as* early-stopping regularization: the true
  MLE is a boundary solution with a numerically singular ``B``, perfect task
  correlation and both noises on the floor.  The 200-step fit and the MLE are
  different models and the step count is load-bearing, not incidental.
* **``predict_coregionalized`` duplicates ``mogp.predict`` bit for bit** (SPEC
  D6, measured difference exactly ``0.0``) and is called only from a test.
  There is ONE predict here, exposed under both spellings.
* **``var_observed`` adds the global ``sigma2`` as well as ``tau_t``**, because
  ``d = tau + sigma2``; ``observation_noise=True`` is the default inherited from
  the previous implementation (``ICM-33``).

Not fixed here, on purpose (SPEC §8)
------------------------------------
``acquisition.compute_qnehvi`` rebuilds the joint posterior from the *marginals*
with a strictly diagonal covariance (``mogp_core.qnehvi.draw_objective_samples``;
``ICM-34`` / ``QN-15`` in ``PROVENANCE.md``), discarding a measured within-point
cross-task correlation of ``0.994`` (latent) / ``0.919`` (observed) and
between-candidate blocks up to ``0.317``.  That is the one thing the ICM learns
which the independent model cannot, and it is deleted before the acquisition
sees it.  :func:`posterior_covariance` and :func:`posterior_block_covariance`
compute the full joint at the same asymptotic cost, so the information needed to
close the gap is available — but doing so is a deliberate scientific change, not
a port, and this module does not make it.  The default path returns marginals.

What is NOT identified, and must never be tested (SPEC §7)
----------------------------------------------------------
``B = W W^T + diag(v)`` is invariant under ``W -> W Q`` for any orthogonal ``Q``
(measured ``4.44e-16`` at ``K=3, R=2``), and at ``K=2, R=1`` there is in
addition a **continuous** family ``(w1, w2, v1, v2)`` giving one identical ``B``.
``W`` flips sign across seeds at every step count.  So ``W``, ``raw_var`` and
``raw_task_noises`` are not legitimate comparison targets; ``B`` and the task
correlation :func:`task_correlation` are.  ``R >= K`` is pure
over-parameterization (``rank=2`` at ``K=2`` adds 2 non-identifiable parameters
for an indistinguishable loss), kept only for ``K > 2``.

Batch shapes
------------
The acquisition gate showed that the *torch* GP's float32 output depends on the
batch shape it is called with, so the calling code had to reproduce botorch's
posterior call pattern exactly (baseline from a separate predict, candidates from
a stacked ``cat(X_baseline, X)`` layout) or lose six orders of magnitude.  That
hazard is a float32/batched-linear-algebra artefact.  This module is float64 and
evaluates every query row through the same row-independent formulas, so its
output is shape-invariant to within BLAS blocking; ``test_core_icm.py`` measures
that residual.  ``acquisition.compute_qnehvi`` still calls ``predict`` in that
pattern (baseline alone, candidates stacked behind it), because the *consumers*
downstream were verified against it; this module does not need it for
correctness, and ``QN-23`` in ``PROVENANCE.md`` records the pattern as a
bit-compatibility arrangement for the float32 torch path (audit action
``remove``).

numpy + stdlib only.  No torch / gpytorch / botorch / scipy / sklearn.
"""

from __future__ import annotations

import math

import numpy as np

from mogp_core import gp as _gp
from mogp_core import optim as _optim
from mogp_core import tanimoto as _tanimoto
from mogp_core import transforms as _transforms

__all__ = [
    "ADAM_BETAS",
    "ADAM_EPS",
    "ADAM_LR",
    "DEFAULT_RANK",
    "DOCKING_TASK_INDICES",
    "ICMState",
    "NOISE_LOWER_BOUND",
    "N_ADAM_STEPS",
    "PARAM_ORDER",
    "TASK_NAMES",
    "assemble_dense",
    "dense_covariance",
    "eig_data",
    "fit",
    "flat_index",
    "gram",
    "init_theta",
    "kron_matvec",
    "logdet",
    "n_params",
    "natural",
    "nmll_and_grad",
    "posterior",
    "posterior_block_covariance",
    "posterior_covariance",
    "predict",
    "predict_coregionalized",
    "prepare",
    "solve",
    "split_grad",
    "standardize",
    "task_correlation",
    "task_covariance",
    "theta_from_groups",
    "train_mogp_coregionalized",
    "unpack",
    "unravel_flat",
]


# --------------------------------------------------------------------------
# Constants.  Re-exported from ``mogp_core.gp`` so the two GP layers cannot
# drift apart on the objective layout.
# --------------------------------------------------------------------------

#: ``mogp.TASK_NAMES`` — the single source of truth for objective order.
TASK_NAMES = _gp.TASK_NAMES

#: ``mogp.DOCKING_TASK_INDICES`` — the only columns the grey-box ICM models.
DOCKING_TASK_INDICES = _gp.DOCKING_TASK_INDICES

#: ``gpytorch.constraints.GreaterThan(1e-4)`` on BOTH noise parameters.  Not on
#: ``raw_var``, which is ``Positive()`` = plain softplus (SPEC D5).
NOISE_LOWER_BOUND = _transforms.GREATER_THAN_LOWER

#: ``train_mogp_coregionalized`` defaults / ``torch.optim.Adam`` defaults.
ADAM_LR = _optim.DEFAULT_LR
ADAM_BETAS = _optim.DEFAULT_BETAS
ADAM_EPS = _optim.DEFAULT_EPS
N_ADAM_STEPS = _optim.DEFAULT_N_ITERATIONS

#: ``IndexKernel(rank=...)``: 1 is the project default and the only value the
#: loop ever passes.  ``R >= K`` is pure over-parameterization (SPEC D8).
DEFAULT_RANK = 1

#: The raw-vector / flat-gradient group order.  Identical to
#: ``spec/icm/fixtures/06_gradients.json``'s ``layout``, so a flat gradient from
#: here lines up with the stored analytic / autograd / finite-difference arrays
#: element for element.
PARAM_ORDER = (
    "W",
    "raw_var",
    "raw_task_noises",
    "raw_noise",
    "mean_constant",
)


def n_params(n_tasks: int, rank: int = DEFAULT_RANK) -> int:
    """``K*R + 3K + 1`` — the trainable scalar count.  **9** at ``K=2, R=1``.

    Not ``3T + 1``: that is the independent model's count (SPEC §1.5).  All of
    them are reached by ``torch.optim.Adam(model.parameters())`` because the
    likelihood is registered as a submodule of the ``ExactGP`` — there was no
    missed-parameter bug on the torch side to reproduce here.
    """
    K = int(n_tasks)
    R = int(rank)
    if K < 1:
        raise ValueError(f"n_tasks must be >= 1, got {n_tasks}")
    if R < 1:
        raise ValueError(f"rank must be >= 1, got {rank}")
    return K * R + 3 * K + 1


# --------------------------------------------------------------------------
# Index arithmetic (SPEC §2.1) — data-major / "interleaved"
# --------------------------------------------------------------------------

def flat_index(i, t, n_tasks: int):
    """``a = i*K + t`` (O1).  Row = molecule, column = task, always.

    This is the whole of the transpose trap.  The task-major alternative
    ``t*N + i`` gives a permutation-similar covariance whose loss, determinant
    and hypervolume are identical, so nothing aggregate can catch it.
    """
    K = int(n_tasks)
    if K < 1:
        raise ValueError(f"n_tasks must be >= 1, got {n_tasks}")
    return np.asarray(i) * K + np.asarray(t)


def unravel_flat(a, n_tasks: int):
    """Inverse of :func:`flat_index`: ``a -> (a // K, a % K)`` (O1)."""
    K = int(n_tasks)
    if K < 1:
        raise ValueError(f"n_tasks must be >= 1, got {n_tasks}")
    a = np.asarray(a)
    return a // K, a % K


def kron_matvec(P, Q, V):
    """``(P (x) Q) vec(V) = vec(P V Q^T)`` for ``V`` of shape ``(N, K)`` (O3).

    The reshape identity that replaces every Kronecker matrix-vector product.
    ``P (x) Q`` is never materialized anywhere in this module.

    Note this is the *row-major transpose* of the Matrix Cookbook's §10.2.3
    ``vec(A X B^T) = (B (x) A) vec X``, because that identity is stated for
    column-major ``vec`` and this repo's layout is data-major C-order.
    """
    P = np.asarray(P)
    Q = np.asarray(Q)
    V = np.asarray(V)
    if V.ndim != 2:
        raise ValueError(f"V must be (N, K), got shape {V.shape}")
    return P @ V @ Q.T


def assemble_dense(K_data, B_task):
    """The dense data-major covariance ``kron(K_data, B_task)`` (O2).

    Present **only** to be tested: it is the one function whose transpose is
    undetectable downstream, so ``test_core_icm.py`` compares it against
    gpytorch's assembled ``MultitaskKernel`` on a deliberately asymmetric pair
    where the two candidate orderings differ by ``3.0``.  Nothing on the fit or
    predict path calls it — those use (O3) and never form the product.
    """
    K_data = np.asarray(K_data, dtype=np.float64)
    B_task = np.asarray(B_task, dtype=np.float64)
    if K_data.ndim != 2 or K_data.shape[0] != K_data.shape[1]:
        raise ValueError(f"K_data must be square, got shape {K_data.shape}")
    if B_task.ndim != 2 or B_task.shape[0] != B_task.shape[1]:
        raise ValueError(f"B_task must be square, got shape {B_task.shape}")
    return np.kron(K_data, B_task)


# --------------------------------------------------------------------------
# Parameterization (SPEC §1.5)
# --------------------------------------------------------------------------

def unpack(theta, n_tasks: int, rank: int = DEFAULT_RANK):
    """``theta -> (W, v_raw, tau_raw, sigma_raw, c)``.

    ``W`` comes back as ``(K, R)``; ``sigma_raw`` as a Python float (it is ONE
    shared scalar, not one per task).  Views into ``theta``, not copies, except
    ``W`` which is a reshape (also a view when ``theta`` is contiguous).
    """
    K = int(n_tasks)
    R = int(rank)
    want = n_params(K, R)
    theta = np.asarray(theta)
    if theta.shape != (want,):
        raise ValueError(
            f"theta must have shape ({want},) for K={K}, R={R} "
            f"(K*R + 3K + 1 scalars, SPEC §1.5), got {theta.shape}")
    o = 0
    W = theta[o:o + K * R].reshape(K, R)
    o += K * R
    v_raw = theta[o:o + K]
    o += K
    tau_raw = theta[o:o + K]
    o += K
    sigma_raw = theta[o]
    o += 1
    c = theta[o:o + K]
    return W, v_raw, tau_raw, sigma_raw, c


def theta_from_groups(W, raw_var, raw_task_noises, raw_noise, mean_constant,
                      dtype=np.float64):
    """Assemble ``theta`` from the five ``named_parameters()`` groups.

    Group order is :data:`PARAM_ORDER`, which is the order
    ``spec/icm/fixtures/06_gradients/theta_raw.npy`` was written in.
    """
    W = np.atleast_2d(np.asarray(W, dtype=dtype))
    v = np.atleast_1d(np.asarray(raw_var, dtype=dtype))
    tn = np.atleast_1d(np.asarray(raw_task_noises, dtype=dtype))
    gn = np.atleast_1d(np.asarray(raw_noise, dtype=dtype))
    c = np.atleast_1d(np.asarray(mean_constant, dtype=dtype))
    K = W.shape[0]
    if not (v.size == tn.size == c.size == K):
        raise ValueError(
            f"raw_var/raw_task_noises/mean_constant must all have K={K} "
            f"entries; got {v.size}, {tn.size}, {c.size}")
    if gn.size != 1:
        raise ValueError(
            f"raw_noise is ONE shared scalar (SPEC §1.4), got {gn.size} values")
    return np.concatenate([W.reshape(-1), v, tn, gn, c]).astype(dtype)


def split_grad(grad, n_tasks: int, rank: int = DEFAULT_RANK):
    """Flat raw gradient -> the five groups of :data:`PARAM_ORDER`.

    Returned as a dict so a comparison against autograd can be made group by
    group: a sign error in one group cannot then hide behind another, which is
    exactly the failure mode the spec's per-group table is designed to expose.
    """
    K = int(n_tasks)
    R = int(rank)
    W, v, tn, gn, c = unpack(np.asarray(grad), K, R)
    return {
        "W": np.array(W, copy=True),
        "raw_var": np.array(v, copy=True),
        "raw_task_noises": np.array(tn, copy=True),
        "raw_noise": np.atleast_1d(np.array(gn, copy=True)),
        "mean_constant": np.array(c, copy=True),
    }


def init_theta(n_tasks: int, rank: int = DEFAULT_RANK, rng=None,
               dtype=np.float64):
    """The initialization adopted from gpytorch: ``W, v_raw ~ randn``;
    ``c = tau_raw = sigma_raw = 0`` (``ICM-11``, ``ICM-12``).

    ``IndexKernel`` draws ``covar_factor`` and ``raw_var`` from the **global**
    torch RNG at construction, so they are redrawn at every BO iteration's refit
    (measured: two back-to-back constructions differ; reseeding reproduces them,
    ``fixtures/01_parameterization.json``).  ``ConstantMean.raw_constant``,
    ``raw_task_noises`` and ``raw_noise`` all initialize to exactly ``0.0``.

    Args:
        rng: a ``Generator`` (what ``BOLoop`` passes: keyed by
            ``(run_seed, iteration, "icm_init")``, so the draw is a pure function
            of its key) or an ``int`` seed makes the fit reproducible on its own,
            independent of numpy's global RNG.  ``None`` is the documented
            AMBIENT fallback for ad-hoc callers that pass no key: it draws from
            numpy's **global** RNG, so reproducibility then needs
            ``np.random.seed`` and any individual refit depends on how many draws
            preceded it.  The loop never takes this path.

    ``W`` is drawn before ``v_raw``, matching ``IndexKernel``'s own construction
    order (``covar_factor`` then ``raw_var``), so a seeded numpy stream lays the
    two out in the same order the torch stream does.
    """
    K = int(n_tasks)
    R = int(rank)
    if K < 1:
        raise ValueError(f"n_tasks must be >= 1, got {n_tasks}")
    if R < 1:
        raise ValueError(f"rank must be >= 1, got {rank}")
    dt = np.dtype(dtype)
    if rng is None:
        W = np.random.standard_normal((K, R))          # AMBIENT fallback (no key given)
        v = np.random.standard_normal(K)
    else:
        g = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        W = g.standard_normal((K, R))
        v = g.standard_normal(K)
    return theta_from_groups(W, v, np.zeros(K), np.zeros(1), np.zeros(K),
                             dtype=dt)


def task_covariance(W, raw_var):
    """``B = W W^T + diag(softplus(raw_var))`` (B1).

    ``raw_var`` goes through plain ``softplus`` with **no** offset
    (``gpytorch.constraints.Positive``), so ``B`` is symmetric and strictly
    positive definite at every parameter value.

    ``mogp_coregionalized.MOGPCoregionalized.task_covariance_matrix``'s docstring
    used to say ``B B^T + diag(v)``, overloading ``B`` for both the factor and
    the result and omitting the ``softplus`` (SPEC D2); it has since been
    corrected to match.  The formula above is what the code computes; verified
    against ``IndexKernel._eval_covar_matrix`` to ``0.0`` at the fixture's raw
    values.

    There is no ``ScaleKernel`` around the Tanimoto kernel and no lengthscale, so
    ``B`` carries ALL of the output scale.  The usual ICM outputscale/``B``
    redundancy is therefore closed and ``B`` itself is identified — ``W`` is not
    (SPEC §7).
    """
    W = np.asarray(W)
    dt = W.dtype if W.dtype.kind == "f" else np.dtype(np.float64)
    W = np.atleast_2d(W.astype(dt, copy=False))
    v = np.asarray(raw_var, dtype=dt)
    return W @ W.T + np.diag(_transforms.positive(v))


def task_correlation(B):
    """``rho = D_B^-1/2 B D_B^-1/2`` — the identified, testable target (SPEC §7.1c).

    A smooth function of ``B``, hence identified for the same reason ``B`` is.
    Test this, or ``B``, and never ``W`` / ``raw_var`` / ``raw_task_noises``.
    """
    B = np.asarray(B, dtype=np.float64)
    s = np.sqrt(np.diag(B))
    return B / np.outer(s, s)


def natural(theta, n_tasks: int, rank: int = DEFAULT_RANK):
    """Raw -> natural: ``(B, tau, sigma2, d, c)``.

    ``tau_t = 1e-4 + softplus(tau_raw_t)`` and ``sigma2 = 1e-4 + softplus(sigma_raw)``
    — the offsets land AFTER the softplus (``GreaterThan(lb)`` is
    ``lb + softplus``, not ``softplus(z + lb)`` and not ``max(softplus, lb)``),
    and being additive constants they contribute nothing to any derivative.
    ``d = tau + sigma2 >= 2e-4`` is the per-task noise the Kronecker algebra
    actually uses, and it is what makes ``I_N (x) diag(d)`` a Kronecker product —
    the single fact the whole factorization rests on.

    ``c_t`` is UNCONSTRAINED (identity, not softplus): ``ConstantMean`` registers
    a constraint only when one is passed, and none is.
    """
    K = int(n_tasks)
    W, v_raw, tau_raw, sigma_raw, c = unpack(theta, K, rank)
    B = task_covariance(W, v_raw)
    tau = _transforms.greater_than(tau_raw, lower=NOISE_LOWER_BOUND)
    sigma2 = _transforms.greater_than(np.reshape(sigma_raw, (1,)),
                                      lower=NOISE_LOWER_BOUND)[0]
    d = (tau + sigma2).astype(tau.dtype)
    return B, tau, sigma2, d, _transforms.identity(c)


# --------------------------------------------------------------------------
# Kernel and the Kronecker factorization (SPEC §3)
# --------------------------------------------------------------------------

def gram(x1, x2, dtype=np.float64, check_binary: bool = True):
    """Tanimoto Gram, delegating to ``mogp_core.tanimoto.tanimoto_gram``.

    Parameter-free: ``K_XX`` is a *constant* of the training set and does not
    change during optimization.  ``eps`` clamps the denominator (it is not an
    additive ``+eps``), so ``k(x, x) == 1.0`` exactly for any fingerprint with an
    on-bit and an all-zero fingerprint gives a zero row rather than ``0/0``.

    ``kernel.TanimotoKernel.forward`` hard-casts both inputs to float32
    regardless of the model dtype (SPEC D3); the two Grams differ by at most
    ``2.29e-08``, which propagates to ``1.26e-07`` relative on the posterior
    mean.  This module works in float64 and says so rather than inheriting the
    cast silently.
    """
    return _tanimoto.tanimoto_gram(x1, x2, eps=_tanimoto.KERNEL_EPS,
                                   dtype=dtype, check_binary=check_binary)


def eig_data(K_xx):
    """``(lam, U)`` from ``eigh(K_XX)``, eigenvalues clamped at ``0`` (SPEC §10.1).

    **Cache this with the training set, not inside the objective.**  ``K_XX``
    carries no hyperparameters, so this is paid once per training set and every
    subsequent optimizer step is ``O(K^3 + N K^2 + N^2 K)``.  Recomputing it per
    step throws away the whole measured advantage (a 200-step fit at ``N=290``
    goes from ``0.027 s`` to ``2.51 s``).

    The clamp is not cosmetic: Tanimoto is PSD but *exactly* singular when two
    molecules share a fingerprint — which real libraries produce after
    densification — and LAPACK then returns small negative eigenvalues, which
    would drive ``S = lam*mu + 1`` below 1 and ``log(S)`` to ``nan``.  On the
    ``N=40`` real fixture the minimum eigenvalue is ``+0.2991`` and no clamping
    was needed; that is one fixture, not a guarantee.

    ``U``'s column signs are arbitrary and LAPACK-version dependent.  Every
    downstream quantity — logdet, solve, posterior, gradients — is invariant to
    them, so ``U`` is never compared or stored.
    """
    K_xx = np.asarray(K_xx)
    dt = K_xx.dtype if K_xx.dtype.kind == "f" else np.dtype(np.float64)
    lam, U = np.linalg.eigh(K_xx.astype(dt, copy=False))
    return np.maximum(lam, dt.type(0.0)), U


def prepare(lam, B, d):
    """Whiten by the noise, then diagonalize: ``(mu, Vt, S)`` (S2)-(S4).

    ``Btil = D^-1/2 B D^-1/2 = V diag(mu) V^T``, ``Vt = D^-1/2 V``,
    ``S[p, q] = lam_p mu_q + 1``.

    ``Vt`` scales the **rows** of ``V`` (``inv[:, None] * V``), because ``D``
    acts on the task index, which is ``V``'s row index.  ``V * inv[None, :]`` is
    a different and wrong matrix — it still produces a symmetric ``M_B`` and a
    plausible-looking loss (SPEC §10.7).  This is the second silent transpose in
    this module and, unlike the first, it has no library docstring to blame.

    ``mu`` is clamped at 0 for the same reason ``lam`` is; ``d`` is NEVER clamped
    (``d_t >= 2e-4`` by construction from the two ``1e-4`` floors — needing to
    clamp ``d`` means a floor was dropped).
    """
    lam = np.asarray(lam)
    dt = lam.dtype if lam.dtype.kind == "f" else np.dtype(np.float64)
    lam = lam.astype(dt, copy=False)
    d = np.asarray(d, dtype=dt)
    B = np.asarray(B, dtype=dt)
    inv = dt.type(1.0) / np.sqrt(d)
    Btil = B * inv[:, None] * inv[None, :]
    mu, V = np.linalg.eigh(Btil)
    mu = np.maximum(mu, dt.type(0.0))
    Vt = inv[:, None] * V                       # ROWS, not columns (SPEC §10.7)
    S = np.outer(lam, mu) + dt.type(1.0)
    return mu, Vt, S


def logdet(n_train: int, d, S):
    """``log det Sigma = N * sum_t log d_t + sum_{p,q} log S[p,q]`` (L1).

    ``O(NK)`` once ``lam`` and ``mu`` are in hand — no ``NK x NK`` factorization.
    Exact because ``|det U| = |det V| = 1``.
    """
    d = np.asarray(d, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    return float(int(n_train) * np.log(d).sum() + np.log(S).sum())


def solve(R, U, Vt, S):
    """``alpha = Sigma^-1 r`` in ``(N, K)`` matrix layout (S5).

    ``Z = U^T R Vt``; ``Ztil = Z / S``; ``A = U Ztil Vt^T``, and
    ``alpha = A.reshape(-1)`` because ``r`` is the data-major vectorization
    ``r[i*K + t] = R[i, t]`` (O1).  Two applications of (O3); the
    ``NK x NK`` matrix is never formed.  Cost ``O(N^2 K + N K^2)``.
    """
    R = np.asarray(R)
    dt = R.dtype if R.dtype.kind == "f" else np.dtype(np.float64)
    R = R.astype(dt, copy=False)
    U = np.asarray(U, dtype=dt)
    Vt = np.asarray(Vt, dtype=dt)
    S = np.asarray(S, dtype=dt)
    return U @ ((U.T @ R @ Vt) / S) @ Vt.T


def dense_covariance(K_xx, B, tau, sigma2):
    """``Sigma = kron(K_XX, B) + kron(I_N, diag(tau)) + sigma2 I_{NK}`` (C3).

    The explicit ``NK x NK`` matrix, for validation only: ``test_core_icm.py``
    checks :func:`logdet` and :func:`solve` against a dense Cholesky of exactly
    this, and checks that it is positive definite with **no jitter added**
    (SPEC §3.1 proves it always is — ``S >= 1`` and ``d_t >= 2e-4``).  Nothing on
    the fit or predict path calls it; it is ``O((NK)^2)`` memory.

    The three terms are kept separate rather than folded into ``d`` so the
    comparison is against the noise as (N1) *writes* it, not as the Kronecker
    path *uses* it — otherwise the two sides would share an algebraic step and
    the check would be weaker.
    """
    K_xx = np.asarray(K_xx, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    N = K_xx.shape[0]
    K = B.shape[0]
    return (np.kron(K_xx, B)
            + np.kron(np.eye(N), np.diag(tau))
            + float(sigma2) * np.eye(N * K))


# --------------------------------------------------------------------------
# Standardization and the grey-box NaN contract (SPEC §1.1, §10.9-§10.10)
# --------------------------------------------------------------------------

def standardize(Y, docking_task_indices=DOCKING_TASK_INDICES):
    """``train_mogp_coregionalized``'s standardization — the same one ``mogp`` uses.

    Delegates to ``mogp_core.gp.standardize``, which is already verified against
    the independent model's fixtures; the two code paths compute this identically
    (``mogp_coregionalized.train_mogp_coregionalized`` is a literal copy of
    ``mogp.train_mogp``'s block), so they must not be allowed to drift.

    Load-bearing, all four measured (SPEC §10.9, §10.10):

    1. ``ddof = 0`` (population std).
    2. Statistics come from the FULL ``len(TASK_NAMES)`` matrix and are masked to
       NaN off the docking columns **afterwards**, so the three ADMET columns are
       always NaN regardless of their values.
    3. ``y_std == 0 -> 1.0``.  Reachable in real runs: the first BO iteration can
       dock ``n_init`` molecules and get identical scores on one target.  NaN is
       not ``== 0.0``, so a NaN column survives the guard and drops the whole
       objective.
    4. ``acquisition.py`` and ``loop.py`` both index by ``DOCKING_TASK_INDICES``
       into a 5-wide array, so the full-width layout is a contract.
    """
    return _gp.standardize(Y, docking_task_indices=docking_task_indices)


def _normalize(Y, y_mean, y_std, observed):
    """``(Y[:, observed] - y_mean[observed]) / y_std[observed]``, float32."""
    return _gp._normalize(Y, y_mean, y_std, observed)


# --------------------------------------------------------------------------
# Marginal likelihood and its analytic raw-space gradient (SPEC §4)
# --------------------------------------------------------------------------

def nmll_and_grad(theta, K_xx, Y_norm, lam=None, U=None, rank=DEFAULT_RANK,
                  want_grad: bool = True):
    """``loss = -log p(Y | theta) / (N*K)`` and ``dloss/dtheta`` in RAW space.

    Args:
        theta: ``(K*R + 3K + 1,)`` raw parameters.  Its dtype sets the working
            dtype.
        K_xx: ``(N, N)`` Tanimoto Gram — constant across Adam steps.
        Y_norm: ``(N, K)`` standardized targets, row = molecule, column = task.
        lam, U: the cached ``eigh(K_xx)`` from :func:`eig_data`.  Pass them.
            Omitting them recomputes an ``O(N^3)`` eigendecomposition on every
            call, which is precisely the mistake SPEC §3.4 and §10.13 warn about.
        want_grad: ``False`` returns ``(loss, None, extras)``.

    Returns:
        ``(loss, grad, extras)``.  ``grad`` is flat in :data:`PARAM_ORDER`.
        ``extras`` carries ``B, tau, sigma2, d, alpha, logdet, quad, logp, Vt,
        mu, S, M_B, M_D`` for tests and diagnostics.

    The divisor is ``N*K``, not ``N`` (SPEC §4.1, D7) — gpytorch's
    ``ExactMarginalLogLikelihood`` divides by the event shape's element count,
    verified equal to ``log_prob / (N*K)`` to ``5.4e-13``.  There are no prior
    terms: the shipped model registers no priors on any parameter.

    The gradient is Rasmussen & Williams eq. (5.9),
    ``dlogp/dtheta = tr(G dSigma/dtheta)`` with ``G = (alpha alpha^T - Sigma^-1)/2``,
    contracted onto the only two directions ``Sigma`` depends on::

        g_q  = sum_p lam_p / S[p, q]        h_q = sum_p 1 / S[p, q]         (G3)
        M_B  = (A^T K_XX A  -  Vt diag(g) Vt^T) / 2                        (G4)
        M_D  = (colsum(A*A) -  (Vt*Vt) h)    / 2                           (G5)

    so that ``tr(G (K_XX (x) A)) = <A, M_B>_F`` and
    ``tr(G (I_N (x) A)) = <diag A, M_D>``.  Then, with
    ``dB/dW[:, r] = e_t w_r^T + w_r e_t^T`` and ``dB/dv_t = e_t e_t^T``::

        dlogp/dW           = 2 M_B W
        dlogp/dv_raw_t     = M_B[t, t] * sigmoid(v_raw_t)
        dlogp/dtau_raw_t   = M_D[t]    * sigmoid(tau_raw_t)
        dlogp/dsigma_raw   = sum_t M_D[t] * sigmoid(sigma_raw)     <-- COUPLES the tasks
        dlogp/dc_t         = sum_i A[i, t]
        dloss/dtheta       = -(1/(N*K)) dlogp/dtheta                       (G6)

    Cost ``O(N^2 K)``, dominated by ``K_XX A``.  No ``N x N`` inverse and no
    ``NK x NK`` anything — the dense analytic gradient needs an explicit
    ``Sigma^-1`` and that is what makes it 100x slower at ``N=290``.

    Note that ``dlogp/dsigma_raw`` sums ``M_D`` over ALL tasks: the shared global
    noise couples the tasks, so this is not ``K`` independent GPs even though the
    per-task noises are separate.
    """
    theta = np.asarray(theta)
    dt = theta.dtype if theta.dtype.kind == "f" else np.dtype(np.float64)
    theta = theta.astype(dt, copy=False)
    Y_norm = np.asarray(Y_norm, dtype=dt)
    if Y_norm.ndim != 2:
        raise ValueError(f"Y_norm must be (N, K), got shape {Y_norm.shape}")
    N, K = Y_norm.shape
    K_xx = np.asarray(K_xx, dtype=dt)
    if K_xx.shape != (N, N):
        raise ValueError(
            f"K_xx must be ({N}, {N}) to match Y_norm, got {K_xx.shape}")

    W, v_raw, tau_raw, sigma_raw, c = unpack(theta, K, rank)
    B, tau, sigma2, d, c = natural(theta, K, rank)

    if lam is None or U is None:
        lam, U = eig_data(K_xx)
    lam = np.asarray(lam, dtype=dt)
    U = np.asarray(U, dtype=dt)

    mu, Vt, S = prepare(lam, B, d)
    R = Y_norm - c[None, :]
    A = solve(R, U, Vt, S)
    ld = logdet(N, d, S)
    quad = float(R.reshape(-1) @ A.reshape(-1))
    logp = -0.5 * quad - 0.5 * ld - 0.5 * N * K * math.log(2.0 * math.pi)
    loss = -logp / (N * K)

    extras = {
        "B": B, "tau": tau, "sigma2": sigma2, "d": d, "alpha": A,
        "logdet": ld, "quad": quad, "logp": logp, "Vt": Vt, "mu": mu, "S": S,
        "lam": lam, "U": U,
    }
    if not want_grad:
        extras["M_B"] = None
        extras["M_D"] = None
        return loss, None, extras

    g = (lam[:, None] / S).sum(0)                     # (K,)   (G3)
    h = (dt.type(1.0) / S).sum(0)                     # (K,)   (G3)
    M_B = 0.5 * (A.T @ (K_xx @ A) - (Vt * g[None, :]) @ Vt.T)          # (G4)
    M_D = 0.5 * ((A ** 2).sum(0) - (Vt ** 2) @ h)                      # (G5)

    dlogp_W = 2.0 * (M_B @ W)
    dlogp_v = np.diag(M_B) * _transforms.softplus_grad(v_raw)
    dlogp_tn = M_D * _transforms.softplus_grad(tau_raw)
    dlogp_gn = np.atleast_1d(
        M_D.sum() * _transforms.softplus_grad(np.reshape(sigma_raw, (1,))))
    dlogp_c = A.sum(0)

    scale = dt.type(-1.0 / (N * K))
    grad = scale * np.concatenate([
        np.asarray(dlogp_W).reshape(-1),
        np.asarray(dlogp_v).reshape(-1),
        np.asarray(dlogp_tn).reshape(-1),
        np.asarray(dlogp_gn).reshape(-1),
        np.asarray(dlogp_c).reshape(-1),
    ]).astype(dt)

    extras["M_B"] = M_B
    extras["M_D"] = M_D
    return loss, grad, extras


# --------------------------------------------------------------------------
# Fitting (SPEC §6)
# --------------------------------------------------------------------------

def fit(X, Y_norm, n_steps: int = N_ADAM_STEPS, lr: float = ADAM_LR,
        rank: int = DEFAULT_RANK, dtype=np.float64, betas=ADAM_BETAS,
        eps: float = ADAM_EPS, theta0=None, rng=None, S=None,
        check_binary: bool = True, K_xx=None):
    """Fit the ``K*R + 3K + 1`` raw hyperparameters with Adam.

    Args:
        X: ``(N, D)`` binary fingerprints.
        Y_norm: ``(N, K)`` standardized targets.
        n_steps: 200, unconditionally.  No convergence check, no early stop, no
            LR schedule, no clamping beyond the softplus transforms.  SPEC §7.3
            shows this is acting as early-stopping regularization and is the only
            thing keeping the fit off a degenerate boundary MLE, so it is
            load-bearing.
        theta0: an explicit starting point.  ``None`` calls :func:`init_theta`,
            which draws ``W`` and ``v_raw`` from ``rng`` (a keyed ``Generator`` or
            int seed; ``None`` = numpy's ambient global stream, the fallback for
            callers that pass no key).
        S, K_xx: a precomputed Gram, to hoist the ``N x 2048 @ 2048 x N`` matmul
            out of a caller's loop.  Both names accepted; ``S`` matches
            ``mogp_core.gp.fit``, ``K_xx`` matches this module's own notation.

    Returns:
        ``(theta, losses)`` — ``losses[i]`` is evaluated at the iterate *before*
        step ``i+1``, so ``losses[0]`` is the loss at the initialization.

    ``eigh(K_xx)`` is computed ONCE here and reused by every step (SPEC §3.4,
    §10.13): that caching, not the Kronecker algebra, is the whole measured
    speedup.
    """
    dt = np.dtype(dtype)
    Y_norm = np.asarray(Y_norm, dtype=dt)
    if Y_norm.ndim != 2:
        raise ValueError(f"Y_norm must be (N, K), got shape {Y_norm.shape}")
    N, K = Y_norm.shape
    if K_xx is None:
        K_xx = S
    if K_xx is None:
        K_xx = gram(X, X, dtype=dt, check_binary=check_binary)
    K_xx = np.asarray(K_xx, dtype=dt)
    lam, U = eig_data(K_xx)                       # ONCE (SPEC §3.4)

    if theta0 is None:
        theta0 = init_theta(K, rank=rank, rng=rng, dtype=dt)
    theta0 = np.asarray(theta0, dtype=dt)
    if theta0.shape != (n_params(K, rank),):
        raise ValueError(
            f"theta0 must have shape ({n_params(K, rank)},) for K={K}, "
            f"R={rank}, got {theta0.shape}")

    opt = _optim.Adam(theta0, lr=lr, betas=betas, eps=eps, dtype=dt)
    losses = []

    def grad_only(th):
        loss, g, _ = nmll_and_grad(th, K_xx, Y_norm, lam=lam, U=U, rank=rank)
        losses.append(loss)
        return g

    for _ in range(int(n_steps)):
        opt.step(grad_only(opt.get()))
    return opt.get(), np.asarray(losses, dtype=np.float64)


# --------------------------------------------------------------------------
# Prediction (SPEC §5)
# --------------------------------------------------------------------------

def _posterior_pieces(theta, X_train, Y_norm, X_query, rank, dtype,
                      check_binary, K_xx, lam, U):
    """Shared setup for :func:`posterior` and the joint-covariance functions."""
    dt = np.dtype(dtype)
    Y_norm = np.asarray(Y_norm, dtype=dt)
    if Y_norm.ndim != 2:
        raise ValueError(f"Y_norm must be (N, K), got shape {Y_norm.shape}")
    N, K = Y_norm.shape
    if K_xx is None:
        K_xx = gram(X_train, X_train, dtype=dt, check_binary=check_binary)
    K_xx = np.asarray(K_xx, dtype=dt)
    if lam is None or U is None:
        lam, U = eig_data(K_xx)
    lam = np.asarray(lam, dtype=dt)
    U = np.asarray(U, dtype=dt)

    B, tau, sigma2, d, c = natural(np.asarray(theta, dtype=dt), K, rank)
    mu, Vt, S = prepare(lam, B, d)
    A = solve(Y_norm - c[None, :], U, Vt, S)

    K_sx = gram(X_query, X_train, dtype=dt, check_binary=check_binary)
    F = K_sx @ U                                  # (M, N)
    Gb = B @ Vt                                   # (K, K)
    return dict(K=K, B=B, tau=tau, sigma2=sigma2, d=d, c=c, S=S, Vt=Vt, U=U,
                A=A, K_sx=K_sx, F=F, Gb=Gb, dtype=dt)


def posterior(theta, X_train, Y_norm, X_query, rank: int = DEFAULT_RANK,
              dtype=np.float64, check_binary: bool = True, K_xx=None,
              lam=None, U=None):
    """Predictive mean, latent variance and noise-inclusive variance, ``(M, K)`` each.

    On the STANDARDIZED scale (SPEC §5)::

        mean       = 1_M c^T + K_*X A B                                    (R1)
        var_latent = diag(K_**) (x)_outer diag(B) - (F*F) (1/S) (Gb*Gb)^T   (R2)
        var_obs    = var_latent + 1_M d^T                                  (R3)

    with ``F = K_*X U`` and ``Gb = B Vt``.  (R2) is the diagonal of the full
    joint (R4) written without forming it: since
    ``(K_*X (x) B)(U (x) Vt) = (K_*X U) (x) (B Vt)``, the ``(i, t)`` entry of the
    correction is ``sum_{p,q} F[i,p]^2 Gb[t,q]^2 / S[p,q]``.  Cost
    ``O(M K + N K^2)`` on top of the Gram — the ``MK x MK`` matrix is never
    formed.

    ``var_obs`` adds the *global* ``sigma2`` as well as the per-task ``tau_t``,
    because ``d = tau + sigma2``.  That is what ``mogp.predict`` returns (the
    shipped code passes the latent posterior through the likelihood).

    ``gpytorch.settings.fast_pred_var()``, which the shipped
    ``predict_coregionalized`` wraps this in, changes the variance by at most
    ``5.96e-07`` absolute and the mean by exactly ``0.0`` (measured,
    ``fixtures/11_precision.json``).  It is not modelled here.

    Every query row goes through row-independent formulas, so the result does not
    depend on how the query rows are batched — unlike the float32 torch path,
    whose output changes with its batch shape by up to ``1.1e-06``.
    """
    p = _posterior_pieces(theta, X_train, Y_norm, X_query, rank, dtype,
                          check_binary, K_xx, lam, U)
    dt = p["dtype"]
    k_ss = _tanimoto.tanimoto_diag(np.asarray(X_query), np.asarray(X_query),
                                   eps=_tanimoto.KERNEL_EPS, dtype=dt,
                                   check_binary=check_binary)
    mean = p["c"][None, :] + p["K_sx"] @ p["A"] @ p["B"]
    corr = (p["F"] ** 2) @ (dt.type(1.0) / p["S"]) @ (p["Gb"] ** 2).T
    var_latent = np.outer(k_ss, np.diag(p["B"])) - corr
    var_observed = var_latent + p["d"][None, :]
    return mean, var_latent, var_observed


def posterior_block_covariance(theta, X_train, Y_norm, X_query,
                               rank: int = DEFAULT_RANK, dtype=np.float64,
                               check_binary: bool = True, K_xx=None,
                               lam=None, U=None, observation_noise: bool = False):
    """The within-point ``K x K`` cross-task blocks, ``(M, K, K)`` (R5).

    ``cov[i] = K_**[i,i] B - Gb diag(w_i) Gb^T`` with
    ``w_i[q] = sum_p F[i,p]^2 / S[p,q]``.  ``O(M K^2 + M N K)`` — the same
    asymptotic cost as the marginals, for the structure ``acquisition.py:323``
    throws away (measured latent cross-task correlation ``0.994``, observed
    ``0.919``; SPEC §8).

    Nothing on today's acquisition path calls this.  It exists so that closing
    that gap is a one-line change *and* so the gap can be measured rather than
    asserted.  ``observation_noise=True`` adds ``diag(d)`` to every block.
    """
    p = _posterior_pieces(theta, X_train, Y_norm, X_query, rank, dtype,
                          check_binary, K_xx, lam, U)
    dt = p["dtype"]
    k_ss = _tanimoto.tanimoto_diag(np.asarray(X_query), np.asarray(X_query),
                                   eps=_tanimoto.KERNEL_EPS, dtype=dt,
                                   check_binary=check_binary)
    Wq = (p["F"] ** 2) @ (dt.type(1.0) / p["S"])         # (M, K)
    Gb = p["Gb"]
    blocks = (k_ss[:, None, None] * p["B"][None, :, :]
              - np.einsum("tq,iq,sq->its", Gb, Wq, Gb))
    if observation_noise:
        blocks = blocks + np.diag(p["d"])[None, :, :]
    return blocks


def posterior_covariance(theta, X_train, Y_norm, X_query,
                         rank: int = DEFAULT_RANK, dtype=np.float64,
                         check_binary: bool = True, K_xx=None, lam=None,
                         U=None, observation_noise: bool = False):
    """The FULL ``MK x MK`` joint latent covariance, data-major (R4).

    ``cov = kron(K_**, B) - kron(F, Gb) diag(vec S)^-1 kron(F, Gb)^T``, evaluated
    as ``sum_q kron(F diag(1/S[:, q]) F^T, outer(Gb[:, q], Gb[:, q]))`` so the
    ``MK x NK`` intermediate is never allocated.  Algebraically identical; the
    spec's reference forms the literal Kronecker products, which makes the two
    an independent cross-check of each other rather than the same code twice.

    ``O(M^2 N K)`` time and ``O((MK)^2)`` memory: this is the expensive one.  Row
    ``i``, task ``t`` sits at flat index ``i*K + t`` (O1).
    """
    p = _posterior_pieces(theta, X_train, Y_norm, X_query, rank, dtype,
                          check_binary, K_xx, lam, U)
    dt = p["dtype"]
    K_ss = gram(X_query, X_query, dtype=dt, check_binary=check_binary)
    F = p["F"]
    Gb = p["Gb"]
    S = p["S"]
    cov = np.kron(K_ss, p["B"])
    for q in range(p["K"]):
        Fq = F * (dt.type(1.0) / S[:, q])[None, :]
        cov -= np.kron(Fq @ F.T, np.outer(Gb[:, q], Gb[:, q]))
    if observation_noise:
        M = F.shape[0]
        cov = cov + np.kron(np.eye(M, dtype=dt), np.diag(p["d"]))
    return cov


# --------------------------------------------------------------------------
# Fitted-state container and the drop-in train / predict pair
# --------------------------------------------------------------------------

class ICMState:
    """Everything needed to reproduce a fitted ICM, and nothing else.

    Serializes to plain JSON-able types via :meth:`to_dict`, so a run manifest
    can record ``theta``, the observed mask, ``N``, ``K`` and ``R`` and an
    iteration's GP becomes fully reconstructible — which the torch path never
    was, because ``W`` and ``raw_var`` came off an unrecorded global RNG draw.

    ``theta_raw`` is recorded for reproducibility, not for comparison: ``W`` and
    ``raw_var`` are NOT identified (SPEC §7) and two correct fits will disagree
    on them.  :meth:`task_covariance_matrix` and :meth:`task_correlation_matrix`
    are the legitimate targets.

    ``(lam, U)`` is cached here, not recomputed: it is a function of ``X_train``
    alone and it is the entire performance argument (SPEC §3.4, §10.13).
    """

    __slots__ = ("theta", "X_train", "Y_norm", "n_tasks", "rank", "observed",
                 "task_names", "losses", "dtype", "_K_xx", "_lam", "_U")

    def __init__(self, theta, X_train, Y_norm, observed, rank=DEFAULT_RANK,
                 losses=None, task_names=TASK_NAMES, dtype=np.float64,
                 K_xx=None, lam=None, U=None):
        self.dtype = np.dtype(dtype)
        self.theta = np.asarray(theta, dtype=self.dtype)
        self.X_train = np.asarray(X_train)
        self.Y_norm = np.asarray(Y_norm, dtype=np.float32)
        self.observed = np.asarray(observed, dtype=bool)
        self.n_tasks = int(self.Y_norm.shape[1])
        self.rank = int(rank)
        self.task_names = tuple(task_names)
        self.losses = None if losses is None else np.asarray(losses, dtype=np.float64)
        want = n_params(self.n_tasks, self.rank)
        if self.theta.shape != (want,):
            raise ValueError(
                f"theta has shape {self.theta.shape}, expected ({want},) for "
                f"K={self.n_tasks}, R={self.rank}")
        self._K_xx = None if K_xx is None else np.asarray(K_xx, dtype=self.dtype)
        self._lam = None if lam is None else np.asarray(lam, dtype=self.dtype)
        self._U = None if U is None else np.asarray(U, dtype=self.dtype)

    @property
    def n_train(self) -> int:
        return int(self.Y_norm.shape[0])

    def gram(self, check_binary: bool = True):
        """The cached training Gram, computed on first use."""
        if self._K_xx is None:
            self._K_xx = gram(self.X_train, self.X_train, dtype=self.dtype,
                              check_binary=check_binary)
        return self._K_xx

    def eig(self, check_binary: bool = True):
        """The cached ``(lam, U)``.  Never recompute this per optimizer step."""
        if self._lam is None or self._U is None:
            self._lam, self._U = eig_data(self.gram(check_binary=check_binary))
        return self._lam, self._U

    def natural(self):
        """``(B, tau, sigma2, d, c)`` at this state's ``theta``."""
        return natural(self.theta, self.n_tasks, self.rank)

    def task_covariance_matrix(self):
        """The learned dense ``K x K`` ``B`` — ``MOGPCoregionalized``'s method.

        Same name and same return contract (a numpy array), so a caller written
        against the torch class keeps working.  Non-diagonal means the ICM
        captured cross-objective correlation.
        """
        return self.natural()[0]

    def task_correlation_matrix(self):
        """``rho`` from :func:`task_correlation` — the identified target."""
        return task_correlation(self.task_covariance_matrix())

    def to_dict(self):
        """JSON-able summary.  ``X_train`` / ``Y_norm`` are left to the caller."""
        B, tau, sigma2, d, c = self.natural()
        rho = task_correlation(B)
        return {
            "theta_raw": [float(x) for x in self.theta],
            "n_train": self.n_train,
            "n_tasks": self.n_tasks,
            "rank": self.rank,
            "observed": [bool(b) for b in self.observed],
            "task_names": list(self.task_names),
            "natural": {
                "task_covariance_B": [[float(x) for x in row] for row in B],
                "task_correlation": [[float(x) for x in row] for row in rho],
                "task_noise_tau": [float(x) for x in tau],
                "shared_sigma2": float(sigma2),
                "d_effective": [float(x) for x in d],
                "constant_mean": [float(x) for x in c],
            },
            "final_loss": (None if self.losses is None
                           else float(self.losses[-1])),
        }

    def __repr__(self):
        return (f"ICMState(N={self.n_train}, K={self.n_tasks}, R={self.rank}, "
                f"theta={np.asarray(self.theta)!r})")


def train_mogp_coregionalized(train_x, train_y, n_iterations: int = N_ADAM_STEPS,
                              lr: float = ADAM_LR, rank: int = DEFAULT_RANK,
                              dtype=np.float64,
                              docking_task_indices=DOCKING_TASK_INDICES,
                              rng=None, verbose: bool = False,
                              check_binary: bool = True):
    """Drop-in replacement for ``mogp_coregionalized.train_mogp_coregionalized``.

    The first five parameters are that function's signature verbatim
    (``train_x, train_y, n_iterations=200, lr=0.1, rank=1``); everything after is
    appended and optional, so existing call sites are unchanged.

    Returns:
        ``(state, None, y_mean, y_std)`` — a 4-tuple so
        ``model, likelihood, y_mean, y_std = train_mogp_coregionalized(...)``
        call sites need no change; slot 1 is ``None`` because there is no
        ``likelihood`` object any more (the noise lives in ``state.theta``).
        ``state`` is an :class:`ICMState`.

    Raises:
        ValueError: when no docking column survives the NaN screen, with
            ``mogp_coregionalized``'s own message.

    Grey box: only the docking objectives are modelled; the three ADMET columns'
    ``y_mean`` / ``y_std`` are forced NaN and :func:`predict` returns NaN there.

    ``W`` and ``v_raw`` are redrawn at every call, as ``IndexKernel`` redrew them
    at every construction.  ``rng`` is where they come from: ``BOLoop`` passes a
    ``Generator`` keyed by ``(run_seed, iteration, "icm_init")`` so a refit is a
    pure function of its key; ``rng=None`` is the AMBIENT fallback for ad-hoc
    callers (numpy's global stream, so any individual refit then depends on how
    many draws preceded it).  ``verbose=True`` prints the
    every-20-steps loss line the shipped function prints unconditionally; the
    default is quiet because a 50-iteration campaign refits 50 times.
    """
    train_y = np.asarray(train_y, dtype=np.float32)
    y_mean, y_std, observed = standardize(train_y, docking_task_indices)
    if not observed.any():
        raise ValueError(
            "train_mogp_coregionalized: no observed docking target columns to "
            "train on (the grey-box ICM models only the docking objectives).")
    Y_norm = _normalize(train_y, y_mean, y_std, observed)
    dt = np.dtype(dtype)
    K_xx = gram(train_x, train_x, dtype=dt, check_binary=check_binary)
    lam, U = eig_data(K_xx)
    theta, losses = fit(train_x, Y_norm, n_steps=n_iterations, lr=lr, rank=rank,
                        dtype=dt, rng=rng, K_xx=K_xx,
                        check_binary=check_binary)
    if verbose:
        for i in range(len(losses)):
            if (i + 1) % 20 == 0:
                print(f"Iter {i + 1:>4}/{n_iterations} - loss: {losses[i]:.4f}")
    state = ICMState(theta, train_x, Y_norm, observed, rank=rank, losses=losses,
                     dtype=dt, K_xx=K_xx, lam=lam, U=U)
    return state, None, y_mean, y_std


def predict(state, y_mean, y_std, X_new, observation_noise: bool = True,
            dtype=None, check_binary: bool = True):
    """Drop-in replacement for ``mogp.predict`` / ``predict_coregionalized``.

    The two are bit-identical on the torch side (measured difference exactly
    ``0.0``, SPEC D6), so there is one implementation here, not two.

    Args:
        state: the :class:`ICMState` from :func:`train_mogp_coregionalized`.
        y_mean, y_std: the full-width standardization statistics, NaN in every
            un-modelled column.
        observation_noise: ``True`` (the shipped default) returns the
            noise-inclusive marginal variance ``var_latent + d``, which is what
            the shipped ``predict_coregionalized`` returns because it passes the
            latent posterior through the likelihood.  ``False`` returns the
            latent variance the EHVI mathematics actually wants.  The default is
            NOT flipped here — that is the acquisition cluster's call.

    Returns:
        ``(mean, variance)``, each ``(M, n_cols)`` float64 on the ORIGINAL
        de-normalized scale, with every un-modelled column NaN.

    De-standardization is ``mean * y_std + y_mean`` and ``var * y_std**2``.  The
    square is not optional: a bare ``y_std`` is a silent ~20% error at
    ``y_std ~ 1.2`` that no smoke test would catch.
    """
    y_mean = np.asarray(y_mean, dtype=np.float64)
    y_std = np.asarray(y_std, dtype=np.float64)
    observed = np.isfinite(y_mean) & np.isfinite(y_std)
    dt = state.dtype if dtype is None else np.dtype(dtype)
    if dt == state.dtype:
        # Reuse the state's cached (K_xx, lam, U); eig() populates the Gram too,
        # so read it back AFTERWARDS or the Gram is recomputed for nothing.
        lam, U = state.eig(check_binary=check_binary)
        K_xx = state.gram(check_binary=check_binary)
    else:
        lam, U, K_xx = None, None, None
    mean_n, var_lat_n, var_obs_n = posterior(
        state.theta, state.X_train, state.Y_norm, X_new, rank=state.rank,
        dtype=dt, check_binary=check_binary, K_xx=K_xx, lam=lam, U=U)
    var_n = var_obs_n if observation_noise else var_lat_n
    ys = y_std[observed]
    ym = y_mean[observed]
    n_rows = mean_n.shape[0]
    n_cols = y_mean.shape[0]
    mean = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    variance = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    mean[:, observed] = mean_n.astype(np.float64) * ys + ym
    variance[:, observed] = var_n.astype(np.float64) * ys ** 2
    return mean, variance


def predict_coregionalized(model, likelihood, y_mean, y_std, X_new,
                           observation_noise: bool = True, **kw):
    """``predict_coregionalized``'s exact 5-positional signature; ``likelihood`` ignored.

    Present so a call site written against the torch path
    (``predict_coregionalized(model, likelihood, y_mean, y_std, X)``) keeps
    working verbatim.  There is no likelihood object here — the noise lives in
    ``model.theta``.
    """
    del likelihood
    return predict(model, y_mean, y_std, X_new,
                   observation_noise=observation_noise, **kw)
