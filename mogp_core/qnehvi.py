"""qNEHVI in pure numpy: the acquisition, both estimators, and the splice.

A reimplementation of ``acquisition.py::compute_qnehvi`` and the two botorch
adapter classes it stands on (``DockingPosteriorModel``,
``CompositeKnownADMETObjective``).  Nothing here imports torch, gpytorch,
botorch, scipy or sklearn; only numpy, the standard library, and the two sibling
modules ``boxdecomp`` and ``sobol``.  ``torch``/``botorch`` appear in
``test_core_qnehvi.py`` only, as differential oracles.

Read ``spec/qnehvi/SPEC.md`` alongside this file; every section reference below
points into it.

TWO ESTIMATORS, AND WHY BOTH ARE HERE
-------------------------------------
``estimator="exact"``  -- SPEC section 1.7 / Appendix A.  The exact, non-log, q=1 MC
    qNEHVI: ``alpha_j = mean_t sum_k prod_m [min(Y_c[t,j,m], U_t[k,m]) - L_t[k,m]]_+``.
    This is the SPEC's normative choice (decision register #1, SPEC section 7.1) and
    the one the on-disk botorch fixture pins to 1e-12 (T-ID-1).  Its value is an
    interpretable hypervolume increment in the normalized cube, ``0 <= a <= 1``,
    and exact ``0.0`` is attainable -- hence the tiebreak in
    ``rank_for_selection``.

``estimator="log"``    -- a 1:1 mirror of what ``acquisition.py`` runs TODAY:
    botorch's ``qLogNoisyExpectedHypervolumeImprovement`` with its shipped
    defaults ``tau_relu=1e-6``, ``tau_max=1e-3``, ``fat=True``, including the
    unexplained ``clamp_max(1e10)`` on the cell upper bounds
    (``botorch/acquisition/multi_objective/logei.py:447``).  Returns a LOG value:
    typically negative, never exactly ``0.0``, no units.

    Both exist because the SPEC and the task brief disagree about which one this
    rewrite should ship (see ``ESTIMATOR_CHOICE_IS_UNSETTLED`` below).  The
    plumbing -- sampling, composition, normalization, the splice, chunking -- is
    shared, so the two differ only in the final per-cell kernel.  Nothing here
    picks a default silently: ``estimator`` is a required argument.

DELIBERATELY PRESERVED BEHAVIOURS (do not "fix" these)
------------------------------------------------------
* **Forced diagonal posterior.**  ``acquisition.py:316-324`` builds the joint
  posterior as ``torch.diag_embed(var)``, asserting exact independence across
  rows AND across the two tasks -- including between a candidate and the baseline
  molecules it is scored against, and between a candidate and an identical copy
  of itself.  Mirrored here: each row and each task is drawn independently from
  the marginal ``(mu, sigma^2)``.  SPEC section 8.2 flags the scientific objection
  (it removes exactly the correlation NEHVI exists to integrate over, so it
  over-credits near-analogs) and says to keep the diagonal as the benchmark
  configuration.  Kept.
* **Variance floor ``1e-9``**, exact constant, applied before ``sqrt``.
* **``clip(., 0, 1)``** inside the normalization, so ``ref = 0`` really is the
  worst corner.  A candidate with any known-ADMET value clipping to ``0.0``
  scores exactly ``0.0`` under the exact estimator, whatever its docking.
* **Draws are keyed by ROW POSITION, not molecule identity.**  A candidate that
  is a copy of a baseline molecule gets an independent draw and a positive score
  (SPEC section 2.3 T-SPLICE-3 measures ``2.159931e-04``).  Arguably wrong; preserved.
* **CRN candidate layout**: one shared ``(2,)`` normal per MC sample for every
  candidate, which is what botorch's collapsed t-batch
  (``batch_range=(0,-2)``, base samples ``(N, 1, B+1, 2)``) already produces.
* **No final clamp** on the exact estimator.  It is non-negative by
  construction; a ``clip(alpha, 0, None)`` at q=1 could only hide a sign error.

THE SPLICE (SPEC section 2) -- the one structural invariant
---------------------------------------------------
Within one MC sample ``t`` the SAME posterior draw supplies both the baseline
matrix the box decomposition is built from and the candidate values scored
against it, paired elementwise by index.  ``Y_baseline`` is built ONCE and
``Y_baseline[t]`` is handed to ``boxdecomp.nondominated_cells``; it is never
re-drawn inside the loop.  Under today's diagonal posterior a violation is
statistically invisible (SPEC measures the decoupled variant at 5.75 % median
relative, which is exactly the MC standard error), so it cannot be caught by any
tolerance test.  ``return_diagnostics=True`` exposes the buffer the
decomposition consumed so ``test_core_qnehvi.py`` can assert array equality AND
object identity.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from . import boxdecomp, sobol as _sobol

__all__ = [
    "N_OBJ",
    "TASK_NAMES",
    "ADMET_COLUMNS",
    "DOCK_IDX",
    "LIB_IDX",
    "DEFAULT_OBJECTIVE_SIGNS",
    "VAR_FLOOR",
    "TAU_RELU",
    "TAU_MAX",
    "FAT",
    "SPEC_N_MC",
    "LEGACY_N_MC",
    "CANDIDATE_CHUNK",
    "ESTIMATOR_CHOICE_IS_UNSETTLED",
    "acquisition_generator",
    "resolve_lib_admet_cols",
    "reorder_admet",
    "compose_objective_points",
    "normalize_frame",
    "compose_and_normalize",
    "draw_objective_samples",
    "hvi_q1_exact",
    "log_hvi_q1_smoothed",
    "log_fatplus",
    "fatmin_axis",
    "logsumexp_inf_safe",
    "logmeanexp",
    "qnehvi",
    "rank_for_selection",
    "manifest",
]


N_OBJ = 5

# mogp.TASK_NAMES order.  Hard-coded as DATA (not imported) so this module keeps
# its dependency floor; test_core_qnehvi.py asserts it against mogp.TASK_NAMES.
TASK_NAMES = (
    "PfDHFR_Docking",
    "hDHFR_Docking",
    "hERG_Toxicity_Prob",
    "Caco2_logPapp",
    "Half_Life_hours",
)
# data.ADMET_COLUMNS storage order -- NOT the same as the library-task order.
ADMET_COLUMNS = ("Caco2_logPapp", "Half_Life_hours", "hERG_Toxicity_Prob")

DOCK_IDX = (0, 1)  # mogp.DOCKING_TASK_INDICES
LIB_IDX = (2, 3, 4)
DEFAULT_OBJECTIVE_SIGNS = np.array([-1.0, +1.0, -1.0, +1.0, +1.0])

VAR_FLOOR = 1e-9  # PRESERVED, exact (acquisition.py:322)

# botorch's shipped qLogNEHVI defaults.  tau_max is 1e-3, NOT
# botorch.acquisition.logei.TAU_MAX (which is 1e-2) -- qLogNEHVI overrides it
# with a literal and leaves `# TAU_MAX` as a comment beside it.
TAU_RELU = 1e-6
TAU_MAX = 1e-3
FAT = True
_FAT_ALPHA = 2.0  # botorch.utils.safe_math.ALPHA
_SOFTPLUS_THRESHOLD = 20.0  # torch.nn.functional.softplus default
_FATPLUS_CAUCHY_ALPHA = 1e-1  # botorch.utils.safe_math.fatplus, hard-coded

SPEC_N_MC = 512  # SPEC decision register #10
LEGACY_N_MC = 128  # acquisition.N_MC_SAMPLES, what the archived runs used
CANDIDATE_CHUNK = 128  # acquisition.CANDIDATE_CHUNK; a pure memory knob here

ESTIMATOR_CHOICE_IS_UNSETTLED = """\
spec/qnehvi/SPEC.md and the task brief this module was written from disagree on
four points.  Neither default is chosen here silently; `estimator`, `n_mc` and
`sampler` are all explicit arguments.

  | question   | SPEC.md says                      | task brief says          |
  |------------|-----------------------------------|--------------------------|
  | estimator  | exact non-log (decision #1, s7.1) | log-smoothed qLogNEHVI   |
  | sampler    | iid Generator (decision #3, s7.2) | scrambled Sobol'         |
  | n_mc       | 512 (decision #10)                | 128, do not raise        |
  | oracle     | exact botorch qNEHVI (s4 C1-C3)   | botorch qLogNEHVI        |

SPEC.md gives measured reasons for each of its choices and section 7.2 explicitly
rejects "adopting a Sobol engine".  The task brief asserts its four choices are
"per the spec", which they are not.  Both are implemented; the caller decides.
"""


# --------------------------------------------------------------------------
# RNG policy (SPEC section 3.1)
# --------------------------------------------------------------------------


def acquisition_generator(
    run_seed: int, iteration: int, stream: int = 0
) -> np.random.Generator:
    """The one and only source of randomness in the acquisition.

    ``SeedSequence(entropy=(run_seed, iteration, stream))`` -> ``PCG64``.  A
    tuple entropy, never ``a * K + b`` (which collides on every key with the same
    sum) and never ``hash(str)`` (salted per process in CPython).  Streams:
    ``0`` = baseline block, ``1`` = candidate block, ``2..6`` reserved, ``7`` is
    the capture script's deliberately-decoupled splice probe -- do not reuse it.
    """
    ss = np.random.SeedSequence(
        entropy=(int(run_seed), int(iteration), int(stream))
    )
    return np.random.Generator(np.random.PCG64(ss))


# --------------------------------------------------------------------------
# The composite objective (SPEC section 1.1, section 8.3)
# --------------------------------------------------------------------------


def resolve_lib_admet_cols(
    task_names: Sequence[str] = TASK_NAMES,
    admet_columns: Sequence[str] = ADMET_COLUMNS,
    lib_idx: Sequence[int] = LIB_IDX,
) -> "list[int]":
    """Map ``data.ADMET_COLUMNS`` storage order into library-task order, BY NAME.

    This is ``_augment_with_admet``'s ``admet_rows[:, lib_admet_cols]`` reorder,
    the one piece of real logic inside the botorch adapter (SPEC section 8.3 item 2).
    Getting it wrong silently permutes hERG / Caco2 / Half-Life, which is a
    catastrophic and plausible-looking error -- so it is resolved by name and
    raises rather than guessing.
    """
    cols = []
    for j in lib_idx:
        name = task_names[j]
        if name not in admet_columns:
            raise ValueError(
                f"objective {j} ({name!r}) is supplied by neither the docking GP "
                f"nor the known-ADMET columns {list(admet_columns)}"
            )
        cols.append(list(admet_columns).index(name))
    return cols


def reorder_admet(admet_rows, lib_admet_cols: Sequence[int]) -> np.ndarray:
    """``admet_rows[:, lib_admet_cols]`` -- storage order -> library-task order."""
    admet_rows = np.asarray(admet_rows, dtype=np.float64)
    if admet_rows.ndim != 2:
        raise ValueError(f"admet_rows must be 2-D, got {admet_rows.shape}")
    return admet_rows[:, list(lib_admet_cols)]


def compose_objective_points(
    dock,
    admet,
    dock_idx: Sequence[int] = DOCK_IDX,
    lib_idx: Sequence[int] = LIB_IDX,
    num_objectives: int = N_OBJ,
) -> np.ndarray:
    """Scatter sampled docking + known-exact ADMET into TASK_NAMES order.

    ``dock`` is ``(..., len(dock_idx))``, ``admet`` is ``(..., len(lib_idx))``
    ALREADY in library-task order (use ``reorder_admet`` first if you hold raw
    ``data.ADMET_COLUMNS`` rows).  Raises if any objective column is supplied by
    neither source -- a silently missing column would produce a
    lower-dimensional hypervolume (SPEC section 8.3 item 1).
    """
    dock = np.asarray(dock, dtype=np.float64)
    admet = np.asarray(admet, dtype=np.float64)
    covered = set(int(j) for j in dock_idx) | set(int(j) for j in lib_idx)
    missing = sorted(set(range(num_objectives)) - covered)
    if missing:
        raise ValueError(
            f"objective columns {missing} are supplied by neither the docking GP "
            f"nor the known-ADMET library"
        )
    if dock.shape[-1] != len(dock_idx):
        raise ValueError(
            f"dock last axis {dock.shape[-1]} != len(dock_idx) {len(dock_idx)}"
        )
    if admet.shape[-1] != len(lib_idx):
        raise ValueError(
            f"admet last axis {admet.shape[-1]} != len(lib_idx) {len(lib_idx)}"
        )
    shape = np.broadcast_shapes(dock.shape[:-1], admet.shape[:-1])
    full = np.empty(tuple(shape) + (num_objectives,), dtype=np.float64)
    for k, j in enumerate(dock_idx):
        full[..., j] = np.broadcast_to(dock[..., k], shape)
    for k, j in enumerate(lib_idx):
        full[..., j] = np.broadcast_to(admet[..., k], shape)
    return full


def normalize_frame(full, bounds, signs=DEFAULT_OBJECTIVE_SIGNS) -> np.ndarray:
    """Map into ``evaluation.py``'s shared [0,1] maximization frame.

    ``where(s > 0, (y - lo)/span, (hi - y)/span)`` with ``span = 1`` where
    ``span == 0``, then ``clip(0, 1)``.  Numerically identical -- same
    expression, same order -- to ``evaluation.normalize``; SPEC section 8.3 requires
    EXACT equality there, not a tolerance, because the acquisition optimizing a
    different frame from the one the reported hypervolume uses is the failure
    this whole design exists to prevent.
    """
    full = np.asarray(full, dtype=np.float64)
    bounds = np.asarray(bounds, dtype=np.float64)
    signs = np.asarray(signs, dtype=np.float64)
    lo, hi = bounds[:, 0], bounds[:, 1]
    span = hi - lo
    span = np.where(span == 0.0, 1.0, span)
    norm = np.where(signs > 0, (full - lo) / span, (hi - full) / span)
    return np.clip(norm, 0.0, 1.0)


def compose_and_normalize(
    dock,
    admet,
    bounds,
    signs=DEFAULT_OBJECTIVE_SIGNS,
    dock_idx: Sequence[int] = DOCK_IDX,
    lib_idx: Sequence[int] = LIB_IDX,
    num_objectives: int = N_OBJ,
) -> np.ndarray:
    """``compose_objective_points`` then ``normalize_frame``."""
    return normalize_frame(
        compose_objective_points(dock, admet, dock_idx, lib_idx, num_objectives),
        bounds,
        signs,
    )


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------


def draw_objective_samples(
    mu_b,
    var_b,
    admet_b,
    mu_c,
    var_c,
    admet_c,
    bounds,
    *,
    signs=DEFAULT_OBJECTIVE_SIGNS,
    dock_idx: Sequence[int] = DOCK_IDX,
    lib_idx: Sequence[int] = LIB_IDX,
    num_objectives: int = N_OBJ,
    n_mc: int = SPEC_N_MC,
    run_seed: Optional[int] = None,
    iteration: int = 0,
    sampler: str = "iid",
    layout: str = "CRN",
    base_samples=None,
    baseline_stream: int = 0,
    candidate_stream: int = 1,
):
    """Draw the posterior samples and map them into the objective frame.

    Returns ``(Y_baseline, Y_candidates, z_baseline, z_candidates)`` with shapes
    ``(N, B, 5)``, ``(N, Mc, 5)``, ``(N, B, 2)``, ``(N, n_slots, 2)``.

    ``sampler``:
      ``"iid"``   -- SPEC section 3: two ``standard_normal`` draws from
                     ``acquisition_generator(run_seed, iteration, stream)``.
                     Draw SHAPE and ORDER are part of the spec because the stream
                     is a byte sequence: ``(n_mc, n_points, 2)``, C-order.
      ``"sobol"`` -- a scrambled Sobol' draw of dimension ``(B + 1) * 2``, laid
                     out exactly as botorch's sampler lays out
                     ``(N, 1, B + 1, 2)``: rows ``0..B-1`` baseline, row ``B``
                     the single shared candidate slot.  Only valid with
                     ``layout="CRN"``, because there is one candidate slot.

    ``base_samples`` overrides both draws with a caller-supplied
    ``(n_mc, B + 1, 2)`` array -- the substitution T-ID-1 needs to feed botorch's
    own normals in and remove sampling from the comparison.
    """
    mu_b = np.asarray(mu_b, dtype=np.float64)
    var_b = np.asarray(var_b, dtype=np.float64)
    mu_c = np.asarray(mu_c, dtype=np.float64)
    var_c = np.asarray(var_c, dtype=np.float64)
    admet_b = np.asarray(admet_b, dtype=np.float64)
    admet_c = np.asarray(admet_c, dtype=np.float64)

    B, k = mu_b.shape
    Mc = mu_c.shape[0]
    if B == 0:
        raise ValueError("qnehvi: empty baseline (B == 0)")
    if Mc == 0:
        raise ValueError("qnehvi: empty candidate set (Mc == 0)")
    for name, arr in (
        ("mu_b", mu_b),
        ("var_b", var_b),
        ("mu_c", mu_c),
        ("var_c", var_c),
    ):
        # After docking-column selection: predict() returns NaN in the three
        # ADMET columns by contract, so a NaN in a DOCKING column is a GP
        # failure, not a normal state (SPEC T-EDGE-4).
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"qnehvi: non-finite {name}")
    if np.any(var_b < 0.0) or np.any(var_c < 0.0):
        raise ValueError("qnehvi: negative variance")

    sd_b = np.sqrt(np.clip(var_b, VAR_FLOOR, None))
    sd_c = np.sqrt(np.clip(var_c, VAR_FLOOR, None))

    if base_samples is not None:
        base = np.asarray(base_samples, dtype=np.float64)
        if base.shape != (n_mc, B + 1, k):
            raise ValueError(
                f"base_samples must be {(n_mc, B + 1, k)}, got {base.shape}"
            )
        z_b = base[:, :B, :]
        z_c = base[:, B : B + 1, :]
    elif sampler == "iid":
        if run_seed is None:
            raise ValueError("sampler='iid' requires run_seed")
        n_slots = 1 if layout == "CRN" else Mc
        z_b = acquisition_generator(
            run_seed, iteration, baseline_stream
        ).standard_normal((n_mc, B, k))
        z_c = acquisition_generator(
            run_seed, iteration, candidate_stream
        ).standard_normal((n_mc, n_slots, k))
    elif sampler == "sobol":
        if run_seed is None:
            raise ValueError("sampler='sobol' requires run_seed")
        if layout != "CRN":
            raise ValueError(
                "sampler='sobol' mirrors botorch's collapsed t-batch, which has "
                "exactly one candidate slot, so it only supports layout='CRN'"
            )
        flat = _sobol.sobol_normal_samples(
            n_mc,
            (B + 1) * k,
            run_seed=run_seed,
            iteration=iteration,
            stream=baseline_stream,
            scramble=True,
        )
        base = flat.reshape(n_mc, B + 1, k)
        z_b = base[:, :B, :]
        z_c = base[:, B : B + 1, :]
    else:
        raise ValueError(f"unknown sampler {sampler!r}; use 'iid' or 'sobol'")

    if layout == "CRN":
        dock_c = mu_c[None, :, :] + sd_c[None, :, :] * z_c[:, 0, None, :]
    elif layout == "IND":
        dock_c = mu_c[None, :, :] + sd_c[None, :, :] * z_c
    else:
        raise ValueError(f"unknown layout {layout!r}; use 'CRN' or 'IND'")

    dock_b = mu_b[None, :, :] + sd_b[None, :, :] * z_b
    Y_b = compose_and_normalize(
        dock_b, admet_b[None], bounds, signs, dock_idx, lib_idx, num_objectives
    )
    Y_c = compose_and_normalize(
        dock_c, admet_c[None], bounds, signs, dock_idx, lib_idx, num_objectives
    )
    return Y_b, Y_c, z_b, z_c


# --------------------------------------------------------------------------
# Kernel A: the exact non-log q=1 cell formula (SPEC eq. 6)
# --------------------------------------------------------------------------


def hvi_q1_exact(y, lower, upper) -> np.ndarray:
    """``sum_k prod_m [min(y_m, U_km) - L_km]_+``.  ``y`` (M, m) -> (M,).

    ``min(y, U)`` is taken BEFORE subtracting ``L``, so a ``+inf`` upper bound
    never reaches an arithmetic operation that could produce ``inf - inf`` or
    ``inf * 0``.  ``[.]_+`` is exact arithmetic, not a smoothing: a negative
    overlap means the cell lies entirely above ``y`` on that axis and contributes
    nothing.  Delegated to ``boxdecomp.hypervolume_improvement`` so there is one
    implementation of formula (*), not two.
    """
    return boxdecomp.hypervolume_improvement(y, lower, upper)


# --------------------------------------------------------------------------
# Kernel B: botorch's log-space smoothed kernel, mirrored
# --------------------------------------------------------------------------
#
# All four helpers below are transliterations of botorch.utils.safe_math.  At
# q = 1 botorch's inclusion-exclusion scaffolding collapses to the identity --
# `fatmin` over the size-1 subset axis, `logsumexp` over the size-1 q_choose_i
# axis, `logplusexp(-inf, x)` and `logdiffexp(-inf, x)` are each exactly `x`,
# which is verified in test_core_qnehvi.py rather than assumed -- so only the
# per-cell kernel and the two outer reductions remain.


def _pareto(x, alpha: float = _FAT_ALPHA) -> np.ndarray:
    """botorch.utils.safe_math._pareto.  Requires ``x >= 0``."""
    a = alpha / 2.0
    beta_1 = 2.0 * a
    beta_0 = a * beta_1
    return np.power(beta_0 / (beta_0 + beta_1 * x + x * x), a)


def _softplus(t) -> np.ndarray:
    """``torch.nn.functional.softplus(t, beta=1, threshold=20)``."""
    t = np.asarray(t, dtype=np.float64)
    big = t > _SOFTPLUS_THRESHOLD
    safe = np.where(big, 0.0, t)
    return np.where(big, t, np.log1p(np.exp(safe)))


def log_fatplus(x, tau: float = TAU_RELU) -> np.ndarray:
    """``log(fatplus(x, tau))`` -- botorch's fat-tailed smooth ReLU.

    ``fatplus(x) = tau * (softplus(x/tau) + 0.1 * cauchy(x/tau))`` with
    ``cauchy(t) = 1/(1+t^2)``.  The ``0.1`` is hard-coded in botorch with the
    comment "guarantees monotonicity and convexity (TODO: ref + Lemma 4)"; it is
    mirrored, not derived.
    """
    x = np.asarray(x, dtype=np.float64)
    t = x / tau
    return np.log(tau * (_softplus(t) + _FATPLUS_CAUCHY_ALPHA / (1.0 + t * t)))


def _inf_max_helper(max_fun, x, axis):
    """botorch.utils.safe_math._inf_max_helper, keepdim=True, numpy.

    Generalizes the treatment of infinities for smooth approximations to the
    maximum.  A slice whose maximum is ``+-inf`` is reduced by a plain sum over
    the infinite entries instead of the smooth formula.
    """
    x = np.asarray(x, dtype=np.float64)
    M = np.max(x, axis=axis, keepdims=True)
    is_inf_max = np.isinf(M) & (x == M)
    has_inf_max = np.any(is_inf_max, axis=axis, keepdims=True)
    y_inf = np.where(is_inf_max, x, 0.0)
    M_no_inf = np.where(np.isinf(M), 0.0, M)
    y_no_inf = np.where(has_inf_max, 0.0, x) - M_no_inf
    return np.where(
        has_inf_max,
        np.sum(y_inf, axis=axis, keepdims=True),
        M_no_inf + max_fun(y_no_inf, axis),
    )


def logsumexp_inf_safe(x, axis: int, keepdims: bool = False) -> np.ndarray:
    """botorch.utils.safe_math.logsumexp."""

    def _lse(y, ax):
        return np.log(np.sum(np.exp(y), axis=ax, keepdims=True))

    res = _inf_max_helper(_lse, x, axis)
    return res if keepdims else np.squeeze(res, axis=axis)


def _fatmax(x, axis: int, tau: float, alpha: float = _FAT_ALPHA) -> np.ndarray:
    def max_fun(y, ax):
        return tau * np.log(
            np.sum(_pareto(-y / tau, alpha), axis=ax, keepdims=True)
        )

    return _inf_max_helper(max_fun, x, axis)


def fatmin_axis(x, axis: int, tau: float = TAU_MAX, keepdims: bool = False):
    """botorch.utils.safe_math.fatmin: ``-fatmax(-x, ...)``."""
    res = -_fatmax(-np.asarray(x, dtype=np.float64), axis, tau)
    return res if keepdims else np.squeeze(res, axis=axis)


def logmeanexp(x, axis: int) -> np.ndarray:
    """botorch.utils.safe_math.logmeanexp."""
    n = np.asarray(x).shape[axis]
    return logsumexp_inf_safe(x, axis=axis) - np.log(n)


def log_hvi_q1_smoothed(
    y,
    lower,
    upper,
    tau_relu: float = TAU_RELU,
    tau_max: float = TAU_MAX,
    fat: bool = FAT,
) -> np.ndarray:
    """botorch's ``_compute_log_qehvi`` inner kernel at q=1.  ``y`` (M, m) -> (M, K).

    Returns the per-cell LOG areas, before the reduction over cells and MC
    samples.  Steps numbered as in ``logei.py``:

      1. ``Z = y - L``;  ``log_Zi = log_fatplus(Z, tau_relu)``
      2. smooth-min over the subset axis -- identity at q=1
      3. ``log_lengths = fatmin(log_Zi, log(clamp_max(U, 1e10) - L), tau_max)``
      4. ``log_areas = log_lengths.sum(-1)``

    The ``clamp_max(1e10)`` is botorch's, unexplained, at ``logei.py:447``, and
    it is the reason ``boxdecomp`` emits literal ``+inf`` rather than clamping
    upstream.  ``fat=False`` would be ``log_softplus`` + ``smooth_amin``, which
    this repo never runs; only ``fat=True`` is implemented, and ``fat=False``
    raises rather than silently differing.
    """
    if not fat:
        raise NotImplementedError(
            "only fat=True is implemented: it is botorch's qLogNEHVI default and "
            "the only path acquisition.py ever takes"
        )
    y = np.asarray(y, dtype=np.float64)
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)

    Z = y[:, None, :] - lower[None, :, :]  # (M, K, m)
    log_Zi = log_fatplus(Z, tau=tau_relu)

    capped = boxdecomp.clamp_upper_bounds(upper, double=True)
    log_len = np.log(capped - lower)  # (K, m)

    stacked = np.stack(np.broadcast_arrays(log_Zi, log_len[None, :, :]), axis=-1)
    log_lengths = fatmin_axis(stacked, axis=-1, tau=tau_max)
    return log_lengths.sum(axis=-1)  # (M, K)


# --------------------------------------------------------------------------
# The estimator
# --------------------------------------------------------------------------


def qnehvi(
    mu_b,
    var_b,
    admet_b,
    mu_c,
    var_c,
    admet_c,
    bounds,
    *,
    estimator: str,
    n_mc: int,
    run_seed: Optional[int] = None,
    iteration: int = 0,
    signs=DEFAULT_OBJECTIVE_SIGNS,
    ref=None,
    dock_idx: Sequence[int] = DOCK_IDX,
    lib_idx: Sequence[int] = LIB_IDX,
    lib_admet_cols: Optional[Sequence[int]] = None,
    num_objectives: int = N_OBJ,
    sampler: str = "iid",
    layout: str = "CRN",
    base_samples=None,
    cand_chunk: Optional[int] = None,
    tau_relu: float = TAU_RELU,
    tau_max: float = TAU_MAX,
    fat: bool = FAT,
    return_diagnostics: bool = False,
):
    """qNEHVI per candidate, q=1, in the normalized [0,1]^5 maximization frame.

    ``estimator="exact"`` returns hypervolume increments in ``[0, 1]``;
    ``estimator="log"`` returns botorch's log-smoothed value (negative, never
    exactly zero).  These are NOT commensurable -- different scale, frequently
    different sign -- so ``loop.py``'s ``batch_ehvi_scores`` column means a
    different thing under each and archived values must not be plotted against
    new ones on a shared axis (SPEC section 6.4).

    ``lib_admet_cols``: if given, ``admet_b``/``admet_c`` are raw
    ``data.ADMET_COLUMNS`` rows and are reordered into library-task order.  If
    ``None`` they are assumed to be in library-task order already (which is how
    ``spec/qnehvi/fixtures/*_admet_*.npy`` store them).

    ``cand_chunk`` is a pure memory knob and is BIT-NEUTRAL: the standard normals
    do not depend on it and no reduction crosses a chunk boundary (SPEC section 5.4,
    T-CHUNK-1).  ``None`` means one chunk.

    With ``return_diagnostics=True`` returns ``(scores, diag)`` where ``diag``
    carries ``Y_baseline``, ``Y_candidates``, ``per_sample``, the cell counts, and
    ``decomp_inputs`` -- the exact array objects the decomposition consumed, for
    the splice test.
    """
    if estimator not in ("exact", "log"):
        raise ValueError(
            f"estimator must be 'exact' or 'log', got {estimator!r}.  "
            + ESTIMATOR_CHOICE_IS_UNSETTLED
        )
    n_mc = int(n_mc)
    if n_mc < 1:
        raise ValueError("n_mc must be >= 1")

    if lib_admet_cols is not None:
        admet_b = reorder_admet(admet_b, lib_admet_cols)
        admet_c = reorder_admet(admet_c, lib_admet_cols)

    if ref is None:
        ref = np.zeros(num_objectives, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64).reshape(-1)

    Y_b, Y_c, z_b, z_c = draw_objective_samples(
        mu_b,
        var_b,
        admet_b,
        mu_c,
        var_c,
        admet_c,
        bounds,
        signs=signs,
        dock_idx=dock_idx,
        lib_idx=lib_idx,
        num_objectives=num_objectives,
        n_mc=n_mc,
        run_seed=run_seed,
        iteration=iteration,
        sampler=sampler,
        layout=layout,
        base_samples=base_samples,
    )
    Mc = Y_c.shape[1]
    chunk = Mc if cand_chunk is None else max(1, int(cand_chunk))

    per = np.empty((n_mc, Mc), dtype=np.float64)
    decomp_inputs = []
    cell_counts = np.empty(n_mc, dtype=np.int64)

    for t in range(n_mc):
        # THE SPLICE (SPEC section 2): Y_b is built ONCE above and indexed here.  The
        # decomposition for sample t is built from the SAME draw that produced the
        # candidate values scored against it on the next line.  Never re-drawn.
        Yb_t = Y_b[t]
        decomp_inputs.append(Yb_t)
        lower, upper = boxdecomp.nondominated_cells(Yb_t, ref)
        cell_counts[t] = lower.shape[0]
        for s in range(0, Mc, chunk):
            block = Y_c[t, s : s + chunk]
            if estimator == "exact":
                per[t, s : s + chunk] = hvi_q1_exact(block, lower, upper)
            else:
                log_areas = log_hvi_q1_smoothed(
                    block, lower, upper, tau_relu, tau_max, fat
                )
                per[t, s : s + chunk] = logsumexp_inf_safe(log_areas, axis=-1)

    if estimator == "exact":
        # NORMATIVE reduction (SPEC Appendix A): build the full (n_mc, Mc) float64
        # matrix and take mean(axis=0).  No clamp -- already >= 0 by construction,
        # and a clamp at q=1 could only hide a sign error.
        scores = per.mean(axis=0)
    else:
        scores = logmeanexp(per, axis=0)

    if not return_diagnostics:
        return scores
    return scores, {
        "Y_baseline": Y_b,
        "Y_candidates": Y_c,
        "z_baseline": z_b,
        "z_candidates": z_c,
        "per_sample": per,
        "decomp_inputs": decomp_inputs,
        "cell_counts": cell_counts,
        "estimator": estimator,
        "n_mc": n_mc,
        "chunk": chunk,
    }


# --------------------------------------------------------------------------
# Selection (SPEC section 6.3)
# --------------------------------------------------------------------------


def rank_for_selection(scores) -> np.ndarray:
    """Descending score, ties broken by ASCENDING INDEX.  Stable, documented.

    ``acquisition.select_batch`` currently ranks with ``np.argsort(-scores)``,
    whose default ``kind="quicksort"`` is introsort and is NOT stable: the order
    among tied elements is a partitioning artifact that depends on the values and
    the array length, is undocumented, and can change when numpy changes its
    sort.  Under the exact estimator exact ``0.0`` is an attainable value, so
    ties are real and the choice among them must be a convention.  ``lexsort``
    sorts by its LAST key first, so this reads "descending score, then ascending
    index"; index order ties it to library order, which is itself reproducible.
    """
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 1:
        raise ValueError(f"scores must be 1-D, got {s.shape}")
    return np.lexsort((np.arange(s.size), -s))


# --------------------------------------------------------------------------
# Manifest (SPEC section 3.2)
# --------------------------------------------------------------------------


def manifest(
    run_seed: int,
    iteration: int,
    n_mc: int,
    estimator: str,
    sampler: str = "iid",
    layout: str = "CRN",
    baseline_stream: int = 0,
    candidate_stream: int = 1,
) -> dict:
    """The run record.  ``sample_layout`` and ``estimator`` are NOT optional.

    ``"CRN"`` vs ``"IND"`` changes every number and is invisible from the scores
    alone; ``"exact"`` vs ``"log"`` changes the scale and the sign of
    ``loop.py``'s ``batch_ehvi_scores`` column, so a future reader needs both
    recorded to know what the archived numbers mean.
    """
    entry = {
        "rng": "numpy.random.PCG64 via numpy.random.Generator",
        "seed_scheme": "SeedSequence(entropy=(run_seed, iteration, stream))",
        "run_seed": int(run_seed),
        "iteration": int(iteration),
        "streams": {
            "baseline": int(baseline_stream),
            "candidates": int(candidate_stream),
        },
        "entropy_baseline": [int(run_seed), int(iteration), int(baseline_stream)],
        "entropy_candidates": [
            int(run_seed),
            int(iteration),
            int(candidate_stream),
        ],
        "n_mc_samples": int(n_mc),
        "sample_layout": layout,
        "estimator": estimator,
        "sampler": sampler,
        "numpy_version": np.__version__,
    }
    if estimator == "log":
        entry["tau_relu"] = TAU_RELU
        entry["tau_max"] = TAU_MAX
        entry["fat"] = FAT
    if sampler == "sobol":
        entry["qmc"] = _sobol.sobol_manifest(
            run_seed, iteration, baseline_stream, n_mc, -1, scramble=True
        )
    return entry
