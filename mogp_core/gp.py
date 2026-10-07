"""Independent multi-output exact GP with a Tanimoto kernel — pure numpy.

The numerical backend of ``mogp.py`` (``MOGPModel`` / ``train_mogp`` /
``predict``), per ``spec/gp/SPEC.md``.

Provenance.  This module was first written as a behavioural mirror of the
gpytorch/botorch-based model the archived campaigns were produced with.  The
model structure and many conventions below (the per-task-plus-shared noise
parameterisation, the ``1e-4`` noise floors, the all-zero raw start, the
200-step Adam budget, the ``N*T`` loss divisor, NaN-propagating
standardization, the noise-inclusive predictive variance) were adopted from that
implementation.  Reproducing the archived runs is no longer a goal: the archived
results are superseded.  So where this file says "mirror" or "preserved", read
it as a record of how a choice was arrived at, not of the standard it is held
to.  The origin and status of each such choice is tracked in ``PROVENANCE.md``
beside this file (``GT-4``, ``GT-5``, ``GT-8``, ``GT-12``, ``GT-25``,
``GT-26``, ``GT-30``, ``GT-36``, ``GT-41``, ``GT-44``, ``GT-M2``, ``VER-9``).

The model (SPEC §3, §4)
----------------------
``T`` conditionally-independent scaled-Tanimoto GPs, one per *trained* task::

    f_t ~ GP( mu_t , s_t * Tanimoto(.,.) )
    y_t = f_t + eps_t ,   eps_t ~ N(0, tau_t I_N)
    tau_t = sigma2 + d_t          <-- sigma2 is ONE SHARED SCALAR

so there are exactly ``3T + 1`` trainable raw scalars, not ``3T``.  The raw
vector layout (fixed once, used by every function here) is::

    theta[0   : T ] = u_t   raw_outputscale     s_t    = softplus(u_t)
    theta[T   : 2T] = c_t   raw_constant        mu_t   = c_t          (identity)
    theta[2T  : 3T] = w_t   raw_task_noises     d_t    = 1e-4 + softplus(w_t)
    theta[3T]       = v     raw_noise  SHARED   sigma2 = 1e-4 + softplus(v)

Behaviours inherited from the gpytorch implementation, kept as they were
------------------------------------------------------------------------
* **The MLL divisor is ``N*T``, not ``N``** (SPEC §5.2; ``GT-12``).  gpytorch's
  ``ExactMarginalLogLikelihood`` divides by the event shape's element count.
  Every loss value and every gradient carries that factor.
* **The shared global noise couples the tasks** (SPEC §6; ``GT-4``).
  ``dloss/dv`` sums ``dell_t/dtau_t`` over *all* tasks; the likelihood
  decomposes additively but the optimization does not separate, so this is NOT
  ``T`` independent GPs.  ``mogp.MOGPModel``'s docstring used to say otherwise
  (SPEC §10.3) and has since been corrected; this module follows the
  measurement.
* **``sigma2`` and ``d_t`` are not separately identifiable** (SPEC §6.1): the map
  ``(sigma2, d) -> (sigma2 + e, d - e)`` leaves every ``tau_t`` and the loss
  EXACTLY unchanged.  Assert on ``tau_t``, never on the split.
* **``predict`` returns the noise-inclusive MARGINAL variance** ``var_f + tau``
  (SPEC §9.2, measured), on the de-standardized scale.  ``observation_noise``
  defaults to ``True`` to preserve that; passing ``False`` returns the latent
  ``var_f`` the EHVI mathematics actually wants.  The default is NOT flipped here
  — that is the acquisition cluster's call (SPEC §12 #5) and an open owner
  decision (``GT-41``).
* **NaN standardization is NaN-propagating and drops the whole column** (SPEC
  §10): one NaN row out of thirty discards 29 good observations of that
  objective.  The docking columns are all-NaN until docked, which is the case
  this convention serves; whether to keep it is an open owner decision
  (``GT-36``).
* **``ddof = 0``** in ``y_std`` (population std).  ``pandas.Series.std`` defaults
  to ``ddof=1`` and would shift every standardized target by ~1.7%.
* **No jitter** (SPEC §2.2, §13).  ``tau_t >= 2e-4`` *is* the jitter and it is a
  fitted parameter; adding a separate jitter term changes the model silently.
  :func:`cholesky` therefore performs the unjittered factorization only, and
  raises if the matrix does not factor.
* **200 Adam steps, unconditionally**, ``lr=0.1``, from all-zero raw parameters
  (``GT-25``, ``GT-8``).  No schedule, no convergence check, no early stop.  The
  fit is not run to convergence; whether it should be is an open owner decision.

Where the library stops being exact (SPEC §11) — CORRECTED 2026-09-27
--------------------------------------------------------------------
SPEC §11 and an earlier version of this docstring stated, as measured fact, that
gpytorch is exact at ``N*T <= 800`` and stochastic above it.  **That is only true
in a process where botorch has not been imported, and the pipeline as it stood at
``3c6573c`` (the last commit before the numpy port) always imported botorch**, so
that pipeline never ran in the regime the old claim described.  The corrected
statement:

* The route is chosen from **live, process-global, mutable** settings, not from a
  fixed size.  For the marginal log-likelihood, ``linear_operator`` 0.6.1
  (``operators/_linear_operator.py:1705``, ``inv_quad_logdet``) takes the exact
  Cholesky branch iff::

      settings.fast_computations.log_prob.off()
          or  N*T <= settings.max_cholesky_size.value()

  and for the predictive-covariance root (``_choose_root_method``, line 557) it
  returns ``"cholesky"`` iff ``N*T <= max_cholesky_size`` **or**
  ``fast_computations.covar_root_decomposition.off()``.  In both cases an *off*
  fast-computations flag makes the size irrelevant.
* ``import botorch`` **permanently rewrites those globals** at import time
  (botorch 0.18.1, ``botorch/__init__.py:50-55``): it sets
  ``_fast_covar_root_decomposition``, ``_fast_log_prob`` and ``_fast_solves``
  defaults to ``False``, ``max_cholesky_size`` to 4096, ``cholesky_max_tries``
  3 -> 6, and ``gpytorch.settings.max_eager_kernel_size`` 512 -> 4096.  Measured
  in this env::

      after gpytorch import   max_cholesky_size= 800   fast_computations(covar_root=True,  log_prob=True,  solves=True)
      after import botorch    max_cholesky_size=4096   fast_computations(covar_root=False, log_prob=False, solves=False)

  There is no un-import: the mutation is to module-level defaults and it outlives
  any context manager.
* So there are two regimes, both measured (8 identical
  ``ExactMarginalLogLikelihood`` calls on one model, spread = max - min):

  ===========================  ==========  ==========  ===========  ============
  process state                N*T = 800   N*T = 820   N*T = 840    N*T = 4098
  ===========================  ==========  ==========  ===========  ============
  vanilla gpytorch             0.0         3.87e-03    3.93e-03     1.86e-03
  after ``import botorch``     0.0         0.0         0.0          0.0
  ===========================  ==========  ==========  ===========  ============

  The 2x2 control that pins the predicate rather than the version: forcing
  ``max_cholesky_size`` x ``fast_computations`` by hand gives spread
  ``4.70e-03`` (800/on) and ``0.0`` (4096/on) at ``N*T = 820``, and ``2.39e-03``
  (800/on) / ``2.24e-03`` (4096/on) / ``0.0`` (either, off) at ``N*T = 4098``.
  The fast flags dominate; the size only matters while they are on.
* The same applied to the gpytorch-based ``mogp.predict``'s ``fast_pred_var()``.
  At ``N=410, T=2`` the randomized LOVE root is biased by a measured
  ``3.93e-02`` absolute / ``2.09e-02`` relative against the same call without
  ``fast_pred_var`` under vanilla gpytorch; after ``import botorch`` the same
  comparison is ``7.15e-07`` absolute / ``3.81e-07`` relative, i.e. the biased
  path is not taken at all.
* **Which regime applied, and which applies now.**  At ``3c6573c``,
  ``evaluation.py`` imported ``botorch.utils.multi_objective.hypervolume`` and
  ``acquisition.py`` imported botorch, both at module scope, and ``loop.py``
  imported ``acquisition``.  Every run of that pipeline therefore executed with
  the fast paths **off**, i.e. exact Cholesky at every size, no stochastic
  Lanczos log-determinant and no LOVE bias, and the SPEC §11.2 warning that
  "above ``N*T = 800`` the repo's training run is not reproducible at all" did
  not describe any process that pipeline started.  That is no longer the
  pipeline: at HEAD neither ``evaluation.py`` nor ``acquisition.py`` imports
  botorch at module scope (``acquisition.py`` imports it lazily, inside
  ``_build_botorch_adapters``, for callers that want the test oracle), and
  importing ``loop``, ``evaluation``, ``acquisition``, ``mogp``,
  ``mogp_coregionalized`` and ``kernel`` loads none of ``torch``, ``gpytorch``,
  ``botorch`` or ``linear_operator`` (checked 2026-10-07).  There is no library
  GP in the runtime path now, so this library-regime analysis matters only to
  code that imports those libraries on purpose: the differential oracle tests,
  the ``spec/*/capture_*.py`` scripts and the GP-MOBO comparator.  An earlier
  version of this docstring stated, in the present tense, that every production
  entry point imports botorch; that was false at HEAD (ledger ``GT-M2``) and is
  corrected here.
* This module is exact at every size in either regime, so nothing here changes.
  :data:`MAX_CHOLESKY_SIZE` and :func:`oracle_is_exact` are kept because the
  archived fixtures in ``spec/gp/fixtures/`` were captured by ``capture_gp.py``
  in a **vanilla-gpytorch** process, and 800 is the boundary that applied there.
  They are a record of the capture conditions, **not** a live reading — see
  :func:`oracle_is_exact` for the exact caveat.  They are library-regime
  records kept in production code (``GT-44``, audit action ``remove``, open).

Oracle coverage: where there is no library witness
--------------------------------------------------
The float64 Tanimoto Gram that this whole module rests on — every loss, every
gradient and every trained ``theta`` is computed from ``gram(X, X,
dtype=np.float64)`` — has **no library witness at all**, and cannot have one.
The gpytorch ``TanimotoKernel`` (``kernel.TanimotoKernel.forward`` at
``3c6573c``) began ``x1 = x1.to(torch.float32)``, so the library kernel silently
downcasts and returns float32 for a float64 input (measured: input
``torch.float64`` -> output ``torch.float32``).  gpytorch simply cannot be asked
for the float64 Gram.  What is actually witnessed:

* **float32**: bitwise identical to the gpytorch ``TanimotoKernel`` as of
  ``3c6573c`` (max abs diff exactly ``0.0``), in
  ``test_core_primitives.test_tanimoto_live_oracle_is_bit_exact``, which rebuilds
  that class (transcribed from the ``3c6573c`` source) and runs it.
  ``kernel.TanimotoKernel`` at HEAD is a numpy adapter over
  ``mogp_core.tanimoto``, so it is not an independent witness.
* **float64**: checked only against numpy references — ``capture_gp.tanimoto_gram``
  (a reimplementation of the same formula) and a sha256 of the float64 C-order
  bytes.  That is self-consistency against a second numpy expression of the same
  algebra, **not** an independent oracle.  If both numpy spellings share a
  misreading of the kernel, no test in this repo would catch it.

The downstream float64 losses and gradients *are* checked against live torch
autograd, which exercises the float32 kernel, so the float64 Gram is constrained
indirectly to float32 accuracy (~1e-7), not to float64.

numpy + stdlib only.  No torch / gpytorch / botorch / scipy / sklearn.
"""

from __future__ import annotations

import math

import numpy as np

from mogp_core import linalg as _linalg
from mogp_core import optim as _optim
from mogp_core import tanimoto as _tanimoto
from mogp_core import transforms as _transforms

__all__ = [
    "ADAM_BETAS",
    "ADAM_EPS",
    "ADAM_LR",
    "DOCKING_TASK_INDICES",
    "GPState",
    "MAX_CHOLESKY_SIZE",
    "N_ADAM_STEPS",
    "NOISE_LOWER_BOUND",
    "PARAM_ORDER",
    "TASK_NAMES",
    "cholesky",
    "fit",
    "gram",
    "n_params",
    "natural",
    "nmll_and_grad",
    "oracle_is_exact",
    "posterior",
    "predict",
    "predict_dropin",
    "split_grad",
    "standardize",
    "theta_from_groups",
    "train_mogp",
    "unpack",
]


# --------------------------------------------------------------------------
# Constants.  Copies of ``mogp.py``'s own, duplicated rather than imported
# because ``mogp.py`` imports this package (importing it back would be circular).
# --------------------------------------------------------------------------

#: ``mogp.TASK_NAMES`` — the single source of truth for objective order.
TASK_NAMES = (
    "PfDHFR_Docking",
    "hDHFR_Docking",
    "hERG_Toxicity_Prob",
    "Caco2_logPapp",
    "Half_Life_hours",
)

#: ``mogp.DOCKING_TASK_INDICES`` — the only columns the grey-box GP models.
DOCKING_TASK_INDICES = (0, 1)

#: ``gpytorch.constraints.GreaterThan(1e-4)`` on both noise parameters.
NOISE_LOWER_BOUND = _transforms.GREATER_THAN_LOWER

#: ``mogp.train_mogp`` defaults / ``torch.optim.Adam`` defaults.
ADAM_LR = _optim.DEFAULT_LR
ADAM_BETAS = _optim.DEFAULT_BETAS
ADAM_EPS = _optim.DEFAULT_EPS
N_ADAM_STEPS = _optim.DEFAULT_N_ITERATIONS

#: ``linear_operator.settings.max_cholesky_size``'s **vanilla-gpytorch** default,
#: which is what ``spec/gp/capture_gp.py`` ran under.  Compared against the JOINT
#: operator size ``N*T``, not ``N`` (SPEC §11.2, measured).
#:
#: NOT a live reading, and not the value a process that has imported botorch
#: sees: ``import botorch`` rewrites ``max_cholesky_size._global_value`` to 4096
#: and turns the fast-computations defaults off (measured; the pre-numpy
#: pipeline imported botorch, the numpy pipeline does not -- see the module
#: docstring).  Treat this as the capture-time constant it is; read
#: ``linear_operator.settings`` if you need the live value.
MAX_CHOLESKY_SIZE = 800

#: ``model.named_parameters()`` order, for splitting a flat gradient.
PARAM_ORDER = _transforms.PARAM_NAMES


def n_params(n_tasks: int) -> int:
    """``3T + 1`` — the trainable scalar count.  Not ``3T`` (SPEC §3.1)."""
    T = int(n_tasks)
    if T < 1:
        raise ValueError(f"n_tasks must be >= 1, got {n_tasks}")
    return 3 * T + 1


def oracle_is_exact(n_train: int, n_tasks: int) -> bool:
    """``N*T <= 800``: was gpytorch exact at this size **under the conditions the
    fixtures were captured in** (SPEC §11.2)?

    Read this as "is this size inside the validated envelope of
    ``spec/gp/fixtures/``", not as a live claim about the running process.  The
    function is a pure function of :data:`MAX_CHOLESKY_SIZE` and cannot be
    anything else: it takes no settings argument and does not import the library.

    The live route is decided by mutable process globals, and the real predicate
    for the marginal log-likelihood (``linear_operator`` 0.6.1,
    ``inv_quad_logdet``) is::

        exact  <=>  fast_computations.log_prob.off()
                    or  N*T <= max_cholesky_size.value()

    ``import botorch`` sets ``log_prob`` off and ``max_cholesky_size`` to 4096
    permanently, so in any process that has imported botorch — which the
    pre-numpy pipeline always did (see the module docstring) — the library is
    exact at **every** size
    and this function is pessimistic: it returns ``False`` at ``N*T > 800`` where
    the live library is in fact exact.  Measured: at ``N*T = 820`` the spread over
    8 identical MLL calls is ``3.87e-03`` without botorch and ``0.0`` with it.
    Both regimes are tabulated in the module docstring.

    It is deliberately NOT changed to read the live settings: the fixtures are
    fixed artifacts, so the envelope they validate is fixed too, and making this
    depend on import order would make the fixture tests order-dependent.  Callers
    that need the live answer must read ``linear_operator.settings`` themselves
    (``test_core_gp.live_mll_route_is_exact`` does exactly that).
    """
    return int(n_train) * int(n_tasks) <= MAX_CHOLESKY_SIZE


# --------------------------------------------------------------------------
# Parameterization (SPEC §3)
# --------------------------------------------------------------------------

def unpack(theta, n_tasks: int):
    """``theta -> (u, c, w, v)``; ``v`` is a 0-d view, the SHARED raw noise."""
    T = int(n_tasks)
    theta = np.asarray(theta)
    if theta.shape != (3 * T + 1,):
        raise ValueError(
            f"theta must have shape ({3 * T + 1},) for T={T} (3T+1 scalars, "
            f"SPEC §3.2), got {theta.shape}")
    return theta[:T], theta[T:2 * T], theta[2 * T:3 * T], theta[3 * T]


def theta_from_groups(raw_outputscale, raw_constant, raw_task_noises, raw_noise,
                      dtype=np.float64):
    """Assemble ``theta`` from the four ``named_parameters()`` groups."""
    u = np.atleast_1d(np.asarray(raw_outputscale, dtype=dtype))
    c = np.atleast_1d(np.asarray(raw_constant, dtype=dtype))
    w = np.atleast_1d(np.asarray(raw_task_noises, dtype=dtype))
    v = np.atleast_1d(np.asarray(raw_noise, dtype=dtype))
    if not (u.size == c.size == w.size):
        raise ValueError(
            f"outputscale/constant/task_noises must share T; got {u.size}, "
            f"{c.size}, {w.size}")
    if v.size != 1:
        raise ValueError(
            f"raw_noise is ONE shared scalar (SPEC §4.1), got {v.size} values")
    return np.concatenate([u, c, w, v]).astype(dtype)


def split_grad(grad, n_tasks: int):
    """Flat raw gradient -> the four ``named_parameters()`` groups.

    Returned as a dict keyed by :data:`PARAM_ORDER` so a per-parameter
    comparison against autograd can be made group by group — a sign error in one
    group cannot then hide behind another (SPEC §14 tier 2).
    """
    T = int(n_tasks)
    g = np.asarray(grad)
    u, c, w, v = unpack(g, T)
    return {
        "covar_module.raw_outputscale": np.array(u, copy=True),
        "mean_module.raw_constant": np.array(c, copy=True),
        "likelihood.raw_task_noises": np.array(w, copy=True),
        "likelihood.raw_noise": np.atleast_1d(np.array(v, copy=True)),
    }


def natural(theta, n_tasks: int):
    """Raw -> natural: ``(s, mu, d, sigma2, tau)``.

    ``s_t = softplus(u_t)``; ``mu_t = c_t`` (UNCONSTRAINED — identity, not
    softplus); ``d_t = 1e-4 + softplus(w_t)``; ``sigma2 = 1e-4 + softplus(v)``
    shared across every task; ``tau_t = sigma2 + d_t >= 2e-4``.

    The ``1e-4`` offsets land AFTER the softplus (``GreaterThan(lb)`` is
    ``lb + softplus``, not ``softplus(z + lb)`` and not ``max(softplus, lb)``),
    and they are additive constants so they contribute nothing to any derivative.
    """
    T = int(n_tasks)
    theta = np.asarray(theta)
    u, c, w, v = unpack(theta, T)
    s = _transforms.positive(u)
    mu = _transforms.identity(c)
    d = _transforms.greater_than(w, lower=NOISE_LOWER_BOUND)
    sigma2 = _transforms.greater_than(np.reshape(v, (1,)),
                                      lower=NOISE_LOWER_BOUND)[0]
    tau = (sigma2 + d).astype(d.dtype)
    return s, mu, d, sigma2, tau


# --------------------------------------------------------------------------
# Kernel and linear algebra
# --------------------------------------------------------------------------

def gram(x1, x2, dtype=np.float64, check_binary: bool = True):
    """Tanimoto Gram, delegating to ``mogp_core.tanimoto.tanimoto_gram``.

    Present only so the GP layer has one spelling of the call; it adds nothing.
    For 0/1 input every intermediate is an exact small integer, so ``S`` is
    bitwise reproducible in both precisions and ``S[i, i] == 1.0`` exactly for
    any fingerprint with at least one on-bit — no diagonal special-case.

    ORACLE COVERAGE.  The ``dtype=np.float32`` result is bitwise identical to the
    gpytorch ``TanimotoKernel`` as of ``3c6573c`` (max abs diff exactly ``0.0``).
    The ``dtype=np.float64`` result — the default, and the one every loss and
    gradient in this module is built from — has **no library witness**:
    that kernel's ``forward`` opened with ``x1 = x1.to(torch.float32)``, so a
    float64 input comes back float32 (measured) and gpytorch cannot be asked for
    the float64 Gram.  float64 is checked only against a second numpy expression
    of the same formula plus a byte hash, which is self-consistency, not an
    independent oracle.  See the module docstring.
    """
    return _tanimoto.tanimoto_gram(x1, x2, eps=_tanimoto.KERNEL_EPS,
                                   dtype=dtype, check_binary=check_binary)


def _self_similarity(x, dtype=np.float64, check_binary: bool = True):
    """``diag(Tanimoto(X, X))`` without building the ``M x M`` Gram.

    Bitwise identical to ``np.diag(gram(X, X))`` for binary input (both
    accumulate exact integers), and ``O(M*D)`` instead of ``O(M^2 D)`` — which is
    what makes a library-wide prediction possible at all: the reference's
    ``np.diag(tanimoto_gram(Xs, Xs))`` would allocate ~7 GB at M = 29678.
    """
    return _tanimoto.tanimoto_diag(x, x, eps=_tanimoto.KERNEL_EPS, dtype=dtype,
                                   check_binary=check_binary)


def cholesky(A):
    """Unjittered Cholesky of ``A``, lower factor.

    **No jitter is ever added** (SPEC §2.2, §13: ``tau_t`` is the jitter and it
    is a fitted parameter).  The factor is ``np.linalg.cholesky``'s, unchanged,
    for any positive-definite float32/float64 input.  What this adds is the
    screening numpy omits: ``potrf`` reads only the lower triangle, so a NaN
    below the diagonal factorizes *successfully* into a factor full of NaN and a
    NaN above it is ignored entirely.  The whole matrix is therefore checked with
    ``isfinite`` first, and a factor that is non-finite or has a non-positive
    diagonal is refused.

    Raises:
        ValueError: ``A`` is not a 2-D float32/float64 array.
        FloatingPointError: a non-finite entry anywhere in ``A``.
        numpy.linalg.LinAlgError: ``A`` is not square, or is not numerically
            positive definite (the message gives the size, the diagonal range
            and, up to n = 1000, the minimum eigenvalue).
    """
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError(
            f"cholesky expects a single (n, n) matrix, got shape {A.shape}; "
            f"factorize a stack one slice at a time.")
    if not np.isfinite(A).all():
        raise FloatingPointError(
            "cholesky: input has NaN/Inf. numpy's potrf reads only the lower "
            "triangle, propagates a lower-triangle NaN SILENTLY into the factor "
            "and ignores an upper-triangle NaN entirely.")
    if A.dtype not in (np.float32, np.float64):
        raise ValueError(f"cholesky expects float32 or float64, got {A.dtype}.")
    n = A.shape[0]
    if A.shape[1] != n:
        raise np.linalg.LinAlgError(
            f"cholesky: matrix must be square, got shape {A.shape}.")

    try:
        L = np.linalg.cholesky(A)
    except np.linalg.LinAlgError:
        L = None
    if L is not None and np.isfinite(L).all() and (np.diagonal(L) > 0).all():
        return L

    d = np.diagonal(A)
    detail = f"diagonal range [{float(d.min()):.3e}, {float(d.max()):.3e}]"
    if n <= 1000:                                  # O(n^3), failure path only
        try:
            detail += f", minimum eigenvalue {float(np.linalg.eigvalsh(A)[0]):.3e}"
        except np.linalg.LinAlgError:              # pragma: no cover
            pass
    raise np.linalg.LinAlgError(
        f"cholesky: matrix ({n}x{n}, {A.dtype}) is not numerically positive "
        f"definite; no jitter was applied (tau_t is the jitter, SPEC 2.2/13). "
        f"{detail}.")


def _solve_lower(L, B, use_primitive_solves: bool):
    """``L^-1 B`` for lower-triangular ``L``."""
    if use_primitive_solves:
        return _linalg.forward_substitution(L, B)
    return np.linalg.solve(L, B)


def _solve_upper(L, B, use_primitive_solves: bool):
    """``L^-T B`` for lower-triangular ``L``."""
    if use_primitive_solves:
        return _linalg.back_substitution_LT(L, B)
    return np.linalg.solve(L.T, B)


def _chol_solve(L, B, use_primitive_solves: bool):
    """``(L L^T)^-1 B``."""
    return _solve_upper(L, _solve_lower(L, B, use_primitive_solves),
                        use_primitive_solves)


# --------------------------------------------------------------------------
# Standardization and the NaN contract (SPEC §10)
# --------------------------------------------------------------------------

def standardize(Y, docking_task_indices=DOCKING_TASK_INDICES):
    """``mogp.train_mogp``'s standardization, NaN semantics included.

    Args:
        Y: ``(N, n_cols)`` targets in :data:`TASK_NAMES` order.  Cast to float32
            first — the statistics are computed in float32, as the repo does.
        docking_task_indices: the columns the grey-box GP is allowed to model.

    Returns:
        ``(y_mean, y_std, observed)``, each length ``n_cols``; ``observed`` is
        ``isfinite(y_mean) & isfinite(y_std)`` and ``T = observed.sum()``.

    Four things here are load-bearing (SPEC §10.1):

    1. ``ddof = 0`` — ``numpy.std``'s population default.  ``ddof=1`` shifts
       every standardized target by ~1.7% on a fixture column.
    2. **NaN propagates and that is the mechanism.**  A column with even one NaN
       row gets NaN statistics; the zero-variance guard tests ``y_std == 0.0``,
       which is False for NaN, so NaN survives to ``observed`` and **the entire
       objective is dropped** — not the NaN rows.  One NaN row out of thirty
       discards 29 good observations.  Preserved deliberately.
    3. The grey-box mask is applied **after** the statistics, so the three ADMET
       columns are always NaN regardless of their values.  ``predict`` therefore
       always returns NaN for them: contract, not failure.
    4. ``T`` is dynamic (0, 1 or 2) and changes both ``3T+1`` and the ``N*T``
       divisor.  ``T = 1`` must work.
    """
    Y = np.asarray(Y, dtype=np.float32)
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2-D (N, n_cols), got shape {Y.shape}")
    y_mean = Y.mean(axis=0)                       # NaN-PROPAGATING, by design
    y_std = Y.std(axis=0)                         # POPULATION std, ddof = 0
    y_std = np.where(y_std == 0.0, 1.0, y_std)    # NaN is not == 0.0, so it lives
    mask = np.zeros(Y.shape[1], dtype=bool)
    mask[[j for j in docking_task_indices if j < Y.shape[1]]] = True
    y_mean = np.where(mask, y_mean, np.nan)       # mask AFTER the statistics
    y_std = np.where(mask, y_std, np.nan)
    observed = np.isfinite(y_mean) & np.isfinite(y_std)
    return y_mean, y_std, observed


def _normalize(Y, y_mean, y_std, observed):
    """``(Y[:, observed] - y_mean[observed]) / y_std[observed]``, float32."""
    Y = np.asarray(Y, dtype=np.float32)
    return ((Y[:, observed] - y_mean[observed]) / y_std[observed]
            ).astype(np.float32)


# --------------------------------------------------------------------------
# The log marginal likelihood and its analytic raw-space gradient (SPEC §5, §7)
# --------------------------------------------------------------------------

def nmll_and_grad(theta, S, Y, want_grad: bool = True,
                  use_primitive_solves: bool = False):
    """``loss = -log p(Y | theta) / (N*T)`` and ``dloss/dtheta`` in RAW space.

    Args:
        theta: ``(3T+1,)`` raw parameters.  Its dtype sets the working dtype.
        S: ``(N, N)`` Tanimoto Gram, constant across Adam steps — hoist it.
        Y: ``(N, T)`` standardized targets.
        want_grad: skip the two traces (and the explicit inverse) when False.
        use_primitive_solves: route the triangular solves through
            ``mogp_core.linalg``'s hand-rolled substitutions instead of
            ``np.linalg.solve``.  Equally correct, rounds differently; see the
            module test for the measured gap.

    Returns:
        ``(loss, grad)`` with ``grad = None`` when ``want_grad`` is False.

    The maths (SPEC §5, §7), per task, R&W eq. 2.30 / Alg. 2.1 / eq. 5.9::

        K_t   = s_t S + tau_t I            r_t = y_t - mu_t
        ell_t = -1/2 r^T K^-1 r - 1/2 log|K| - N/2 log 2pi
        dell_t/ds_t   = 1/2 ( a^T S a - tr(K^-1 S) )
        dell_t/dtau_t = 1/2 ( a^T a     - tr(K^-1)   )
        dell_t/dmu_t  = sum(a)            <-- NO 1/2 tr(...) term

    ``dell_t/dmu_t`` carries no trace term because the mean does not enter the
    covariance, so eq. 5.9 does not apply to it.  Then to raw space::

        dloss/du_t = -( g^s_t   * sigmoid(u_t) ) / (N*T)
        dloss/dc_t = -( g^mu_t                ) / (N*T)
        dloss/dw_t = -( g^tau_t * sigmoid(w_t) ) / (N*T)
        dloss/dv   = -( SUM_t g^tau_t ) * sigmoid(v) / (N*T)   <-- SHARED

    The divisor is ``N*T``, the event shape's element count, **not** ``N``
    (SPEC §5.2, measured): at ``N=6, T=2`` gpytorch's mll is ``-1.4459356`` which
    is ``log_prob/12``, while ``log_prob/6`` is ``-2.8918711``.  For ``T=1`` the
    two coincide, which is exactly why this is easy to get wrong.

    The last line is the shared-noise coupling of SPEC §6: ``sigma2`` appears in
    *every* ``tau_t``, so ``dloss/dv`` is one number every task contributes to.
    Using task 0 alone trains a different model.

    Never builds the ``NT x NT`` covariance: ``T`` matrices of ``N x N``, and the
    log-determinant is therefore a sum.
    """
    theta = np.asarray(theta)
    Y = np.asarray(Y)
    if Y.ndim != 2:
        raise ValueError(f"Y must be (N, T), got shape {Y.shape}")
    N, T = Y.shape
    if T < 1:
        raise ValueError("Y has zero task columns; nothing to fit")
    dt = theta.dtype
    if dt.kind != "f":
        raise ValueError(f"theta must be a floating array, got dtype {dt}")
    u, c, w, v = unpack(theta, T)
    s, mu, d, sigma2, tau = natural(theta, T)
    S = np.asarray(S, dtype=dt)
    if S.shape != (N, N):
        raise ValueError(f"S must be ({N}, {N}) to match Y, got {S.shape}")
    I = np.eye(N, dtype=dt)

    log_p = dt.type(0.0)
    g_s = np.zeros(T, dtype=dt)
    g_tau = np.zeros(T, dtype=dt)
    g_mu = np.zeros(T, dtype=dt)
    half = dt.type(0.5)

    for t in range(T):
        K = s[t] * S + tau[t] * I
        L = cholesky(K)
        r = Y[:, t].astype(dt) - mu[t]
        beta = _solve_lower(L, r, use_primitive_solves)
        alpha = _solve_upper(L, beta, use_primitive_solves)      # K^-1 r
        log_det = dt.type(2.0) * np.log(np.diag(L)).sum()
        log_p = log_p + (-half * (r @ alpha) - half * log_det
                         - half * dt.type(N) * dt.type(math.log(2.0 * math.pi)))
        if want_grad:
            Kinv = _chol_solve(L, I, use_primitive_solves)
            g_s[t] = half * (alpha @ (S @ alpha) - np.sum(Kinv * S))
            g_tau[t] = half * (alpha @ alpha - np.trace(Kinv))
            g_mu[t] = alpha.sum()

    NT = dt.type(N * T)
    loss = -log_p / NT
    if not want_grad:
        return float(loss), None

    grad = np.empty_like(theta)
    grad[:T] = -(g_s * _transforms.softplus_grad(u)) / NT
    grad[T:2 * T] = -g_mu / NT
    grad[2 * T:3 * T] = -(g_tau * _transforms.softplus_grad(w)) / NT
    # THE SHARED-NOISE COUPLING: dtau_t/dsigma2 = 1 for EVERY t, so every task
    # accumulates into this single scalar (SPEC §6, §7.5).
    grad[3 * T] = -(g_tau.sum()
                    * _transforms.softplus_grad(np.reshape(v, (1,)))[0]) / NT
    return float(loss), grad


# --------------------------------------------------------------------------
# The optimizer (SPEC §8)
# --------------------------------------------------------------------------

def _adam_spec_arrangement(grad_fn, theta0, n_steps, lr, betas, eps):
    """Adam in ``torch.optim.Adam``'s own arrangement of the update.

    ``theta -= (lr/bc1) * m / (sqrt(v)/sqrt(bc2) + eps)`` — ``eps`` added to
    ``sqrt(v)/sqrt(bc2)``, not to ``v_hat``, and not scaled by either bias
    correction.  Algebraically identical to
    ``mogp_core.optim.Adam``'s ``lr * (m/bc1) / (sqrt(v/bc2) + eps)``; the two
    differ only in float rounding (measured in the module test).
    """
    b1, b2 = betas
    theta = np.array(theta0, copy=True)
    m = np.zeros_like(theta)
    v = np.zeros_like(theta)
    losses = []
    for step in range(1, n_steps + 1):
        loss, g = grad_fn(theta)
        losses.append(loss)
        m = b1 * m + (1.0 - b1) * g
        v = b2 * v + (1.0 - b2) * g * g
        bc1 = 1.0 - b1 ** step
        bc2 = 1.0 - b2 ** step
        theta = theta - (lr / bc1) * m / (np.sqrt(v) / math.sqrt(bc2) + eps)
    return theta, np.asarray(losses, dtype=np.float64)


def fit(X, Y_norm, n_steps: int = N_ADAM_STEPS, lr: float = ADAM_LR,
        dtype=np.float64, betas=ADAM_BETAS, eps: float = ADAM_EPS,
        adam: str = "primitives", use_primitive_solves: bool = False,
        S=None, check_binary: bool = True):
    """Fit the ``3T+1`` raw hyperparameters from the all-zero initialization.

    Args:
        X: ``(N, D)`` binary fingerprints.
        Y_norm: ``(N, T)`` standardized targets.
        n_steps: 200, unconditionally.  No convergence check, no early stop, no
            LR schedule, no clamping beyond the softplus transforms (SPEC §8).
        adam: ``"primitives"`` uses ``mogp_core.optim.Adam``; ``"spec_torch"``
            uses ``torch.optim.Adam``'s literal arrangement of the update.  The
            two are algebraically identical and differ only in rounding.
        S: a precomputed Gram, to hoist the ``N x 2048 @ 2048 x N`` matmul out of
            a caller's loop.  It does not depend on any hyperparameter.

    Returns:
        ``(theta, losses)`` — ``theta`` of shape ``(3T+1,)``, ``losses`` of shape
        ``(n_steps,)``, ``losses[i]`` evaluated at the iterate *before* step
        ``i+1`` (so ``losses[0]`` is the loss at the all-zero init).

    All ``3T+1`` raw parameters start at **exactly zero**, so ``s_t = log 2``,
    ``mu_t = 0``, ``d_t = sigma2 = 1e-4 + log 2``, ``tau_t = 1.386494``.  There is
    no random initialization anywhere, so the fit involves no RNG at all.
    """
    dt = np.dtype(dtype)
    Y_norm = np.asarray(Y_norm, dtype=dt)
    if Y_norm.ndim != 2:
        raise ValueError(f"Y_norm must be (N, T), got shape {Y_norm.shape}")
    N, T = Y_norm.shape
    if S is None:
        S = gram(X, X, dtype=dt, check_binary=check_binary)
    S = np.asarray(S, dtype=dt)
    theta0 = np.zeros(3 * T + 1, dtype=dt)        # EXACTLY zero (SPEC §3.4)

    def loss_and_grad(th):
        return nmll_and_grad(np.asarray(th, dtype=dt), S, Y_norm,
                             use_primitive_solves=use_primitive_solves)

    if adam == "spec_torch":
        return _adam_spec_arrangement(loss_and_grad, theta0, n_steps, dt.type(lr),
                                      (dt.type(betas[0]), dt.type(betas[1])),
                                      dt.type(eps))
    if adam != "primitives":
        raise ValueError(f"adam must be 'primitives' or 'spec_torch', got {adam!r}")

    opt = _optim.Adam(theta0, lr=lr, betas=betas, eps=eps, dtype=dt)
    losses = []

    def grad_only(th):
        loss, g = loss_and_grad(th)
        losses.append(loss)
        return g

    for _ in range(n_steps):
        opt.step(grad_only(opt.get()))
    return opt.get(), np.asarray(losses, dtype=np.float64)


# --------------------------------------------------------------------------
# Prediction (SPEC §9)
# --------------------------------------------------------------------------

def posterior(theta, X_train, Y_norm, X_query, dtype=np.float64,
              clamp_var_f: bool = True, S=None, check_binary: bool = True):
    """Predictive mean, latent-f variance and noise-inclusive variance.

    R&W eq. 2.25-2.26 per task, on the STANDARDIZED scale::

        K_*    = s_t * Tanimoto(X*, X)                     (M, N)
        mean_t = mu_t + K_* alpha_t     alpha_t = K_t^-1 (y_t - mu_t)
        var_f  = s_t * k_**  -  rowsum(V * V)   V = L_t^-1 K_*^T
        var_y  = var_f + tau_t

    ``(V*V).sum(axis=0)`` is ``diag(K_* K_t^-1 K_*^T)``; the ``M x M`` matrix is
    never formed.  ``k_**`` is ``diag(Tanimoto(X*, X*))``, which is the all-ones
    vector for any fingerprint with an on-bit.

    Returns:
        ``(mean, var_f, var_y)``, each ``(M, T)``.

    ``clamp_var_f`` floors ``var_f`` at 0 (SPEC §9.1): with ``s_t`` tiny and ``N``
    large it can go slightly negative through cancellation.  It does not fire on
    any fixture case — the module test measures how far from zero the minimum
    actually sits.
    """
    dt = np.dtype(dtype)
    theta = np.asarray(theta, dtype=dt)
    Y_norm = np.asarray(Y_norm, dtype=dt)
    N, T = Y_norm.shape
    s, mu, d, sigma2, tau = natural(theta, T)
    if S is None:
        S = gram(X_train, X_train, dtype=dt, check_binary=check_binary)
    S = np.asarray(S, dtype=dt)
    Ksx = gram(X_query, X_train, dtype=dt, check_binary=check_binary)
    k_self = _self_similarity(X_query, dtype=dt, check_binary=check_binary)
    M = Ksx.shape[0]
    I = np.eye(N, dtype=dt)
    mean = np.zeros((M, T), dtype=dt)
    var_f = np.zeros((M, T), dtype=dt)
    for t in range(T):
        K = s[t] * S + tau[t] * I
        L = cholesky(K)
        r = Y_norm[:, t] - mu[t]
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, r))
        Kst = s[t] * Ksx
        mean[:, t] = mu[t] + Kst @ alpha
        V = np.linalg.solve(L, Kst.T)
        var_f[:, t] = s[t] * k_self - (V * V).sum(axis=0)
    if clamp_var_f:
        var_f = np.maximum(var_f, dt.type(0.0))
    var_y = var_f + tau[None, :].astype(dt)
    return mean, var_f, var_y


# --------------------------------------------------------------------------
# The drop-in replacements for mogp.train_mogp / mogp.predict
# --------------------------------------------------------------------------

class GPState:
    """Everything needed to reproduce a fitted GP, and nothing else.

    Serializes to plain JSON-able types, so a run manifest can record ``theta``,
    the observed mask, ``N`` and ``T`` and an iteration's GP becomes fully
    reconstructible — which the torch path never was.
    """

    __slots__ = ("theta", "X_train", "Y_norm", "n_tasks", "observed",
                 "task_names", "losses", "dtype")

    def __init__(self, theta, X_train, Y_norm, observed, losses=None,
                 task_names=TASK_NAMES, dtype=np.float64):
        self.dtype = np.dtype(dtype)
        self.theta = np.asarray(theta, dtype=self.dtype)
        self.X_train = np.asarray(X_train)
        self.Y_norm = np.asarray(Y_norm, dtype=np.float32)
        self.observed = np.asarray(observed, dtype=bool)
        self.n_tasks = int(self.Y_norm.shape[1])
        self.task_names = tuple(task_names)
        self.losses = None if losses is None else np.asarray(losses, dtype=np.float64)
        if self.theta.shape != (3 * self.n_tasks + 1,):
            raise ValueError(
                f"theta has shape {self.theta.shape}, expected "
                f"({3 * self.n_tasks + 1},) for T={self.n_tasks}")

    @property
    def n_train(self) -> int:
        return int(self.Y_norm.shape[0])

    def natural(self):
        """``(s, mu, d, sigma2, tau)`` at this state's ``theta``."""
        return natural(self.theta, self.n_tasks)

    def oracle_was_exact(self) -> bool:
        """``N*T <= 800``: is this size inside the fixtures' validated envelope?

        Thin wrapper over :func:`oracle_is_exact` and carries all of its caveats:
        it is a capture-time statement, not a live reading of the library, and it
        is pessimistic in any process that has imported botorch.
        """
        return oracle_is_exact(self.n_train, self.n_tasks)

    def to_dict(self):
        """JSON-able summary.  ``X_train`` / ``Y_norm`` are left to the caller."""
        s, mu, d, sigma2, tau = self.natural()
        return {
            "theta_raw": [float(x) for x in self.theta],
            "n_train": self.n_train,
            "n_tasks": self.n_tasks,
            "observed": [bool(b) for b in self.observed],
            "task_names": list(self.task_names),
            "natural": {
                "outputscale": [float(x) for x in s],
                "constant_mean": [float(x) for x in mu],
                "task_noise_d": [float(x) for x in d],
                "shared_sigma2": float(sigma2),
                "tau_effective": [float(x) for x in tau],
            },
            "oracle_was_exact": self.oracle_was_exact(),
            "final_loss": (None if self.losses is None
                           else float(self.losses[-1])),
        }

    def __repr__(self):
        return (f"GPState(N={self.n_train}, T={self.n_tasks}, "
                f"theta={np.asarray(self.theta)!r})")


def train_mogp(train_x, train_y, n_iterations: int = N_ADAM_STEPS,
               lr: float = ADAM_LR, dtype=np.float64,
               docking_task_indices=DOCKING_TASK_INDICES,
               adam: str = "primitives", check_binary: bool = True):
    """Drop-in replacement for ``mogp.train_mogp``.

    Returns:
        ``(state, None, y_mean, y_std)`` — a 4-tuple so
        ``model, likelihood, y_mean, y_std = train_mogp(...)`` call sites need no
        change; slot 1 is ``None`` because there is no ``likelihood`` object any
        more.  ``state`` is a :class:`GPState`.

    Raises:
        ValueError: when no docking column survives the NaN screen, with
            ``mogp.train_mogp``'s own message.

    The GP models ONLY the docking objectives; the three ADMET objectives are
    known exactly for every candidate and are folded back in by the acquisition's
    composite objective, so their ``y_mean``/``y_std`` are forced NaN and
    ``predict`` returns NaN columns for them.
    """
    train_y = np.asarray(train_y, dtype=np.float32)
    y_mean, y_std, observed = standardize(train_y, docking_task_indices)
    if not observed.any():
        raise ValueError(
            "train_mogp: no observed docking target columns to train on "
            "(the grey-box GP models only the docking objectives).")
    Y_norm = _normalize(train_y, y_mean, y_std, observed)
    theta, losses = fit(train_x, Y_norm, n_steps=n_iterations, lr=lr,
                        dtype=dtype, adam=adam, check_binary=check_binary)
    state = GPState(theta, train_x, Y_norm, observed, losses=losses,
                    dtype=dtype)
    return state, None, y_mean, y_std


def predict(state, y_mean, y_std, X_new, observation_noise: bool = True,
            dtype=np.float64, check_binary: bool = True):
    """Drop-in replacement for ``mogp.predict``'s numerical contract.

    Args:
        state: the :class:`GPState` from :func:`train_mogp`.
        y_mean, y_std: the full-width standardization statistics, NaN in every
            un-modelled column.
        X_new: ``(M, D)`` binary fingerprints.
        observation_noise: ``True`` (the default) returns the noise-inclusive
            MARGINAL variance ``var_f + tau``, which is what ``mogp.predict``
            returns — measured, SPEC §9.2, and the gap is the whole of ``tau_t``
            (``2.07e-02`` to ``1.42e+00`` kcal^2/mol^2 depending on ``N``).
            ``False`` returns the latent ``var_f`` that the EHVI mathematics
            wants.  The default preserves today's behaviour; flipping it is the
            acquisition cluster's decision, not this module's (SPEC §12 #5).

    Returns:
        ``(mean, variance)``, each ``(M, n_cols)`` float64 on the ORIGINAL
        de-standardized scale, with every un-modelled column NaN.

    De-standardization: ``mean * y_std + y_mean`` and ``var * y_std**2``.  The
    square on the variance is not optional — a bare ``y_std`` is a silent ~20%
    error at ``y_std ~ 1.2`` that no smoke test would catch.
    """
    y_mean = np.asarray(y_mean, dtype=np.float64)
    y_std = np.asarray(y_std, dtype=np.float64)
    observed = np.isfinite(y_mean) & np.isfinite(y_std)
    mean_n, var_f_n, var_y_n = posterior(
        state.theta, state.X_train, state.Y_norm, X_new, dtype=dtype,
        check_binary=check_binary)
    var_n = var_y_n if observation_noise else var_f_n
    ys = y_std[observed]
    ym = y_mean[observed]
    n_rows = mean_n.shape[0]
    n_cols = y_mean.shape[0]
    mean = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    variance = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
    mean[:, observed] = mean_n.astype(np.float64) * ys + ym
    variance[:, observed] = var_n.astype(np.float64) * ys ** 2
    return mean, variance


def predict_dropin(state, likelihood, y_mean, y_std, X_new,
                   observation_noise: bool = True, **kw):
    """``mogp.predict``'s exact 5-positional signature; ``likelihood`` ignored.

    Present so a call site written against the torch path
    (``predict(model, likelihood, y_mean, y_std, X)``) keeps working verbatim.
    There is no likelihood object here — the noise lives in ``state.theta``.
    """
    del likelihood
    return predict(state, y_mean, y_std, X_new,
                   observation_noise=observation_noise, **kw)
