"""Differential tests for ``mogp_core.gp`` — the pure-numpy multi-output exact GP.

Two kinds of oracle, both used:

* the archived fixtures in ``spec/gp/fixtures/`` (captured 2026-09-27 from the
  live gpytorch path by ``spec/gp/capture_gp.py``), and
* **live** torch / gpytorch, imported *only here*.  ``mogp_core`` itself may
  import nothing but numpy, pandas and the standard library; this file is a test
  instrument, and importing the library it mirrors is how fidelity gets proven.

WHERE THE LIVE MODEL LIVES (changed 2026-09-28).  The live oracle used to be
``mogp.MOGPModel`` / ``mogp.train_mogp`` / ``mogp.predict``.  The numpy
integration rewrote those, so the gpytorch model is now transcribed verbatim from
git ``3c6573c`` into :func:`oracle` / :func:`live_train_mogp` /
:func:`live_predict` — see the comment above :func:`oracle` for why, and
:func:`test_the_live_oracle_reproduces_the_capture_time_library_bitwise` for the
measurement that proves the transcription is the model ``capture_gp.py`` ran
(agreement on ``loss_gpytorch`` and ``grad_gpytorch_autograd`` is exactly 0.0 at
all 12 case x setting pairs).  Nothing about what is compared, or at what
tolerance, changed.

Every tolerance in this file is roughly 10x a number actually measured by a run
of this file.  Where a quantity came out exact, the assertion says exact.  The
measured value is put in each assertion message, so a failure reports the real
disagreement instead of just "too big".

BLAS THREADS.  SPEC §13 flags reproducibility under multi-threaded BLAS as
UNVERIFIED, because that env segfaults on importing the repo's torch modules with
more than one OpenMP thread.  It is verified here, and it is not free:
:func:`test_blas_thread_count_changes_the_float64_loss_by_one_ulp` measures the
whole thing in subprocesses.  Under ``OMP_NUM_THREADS=1`` this module's float64
loss and gradients are **bitwise** identical to the archived numpy reference; at
2, 4, 8 or an unset thread count they differ by a measured ``4.44e-16`` (loss)
and ``1.39e-17`` (gradient), which is one ULP of the accumulated quantity.  The
tolerances below are set above that floor so the suite is honest at any thread
count, and the thread-sensitivity test pins the exact numbers.  Note that pinning
the thread environment from inside this module cannot work for numpy: a pytest
plugin imports numpy before this file is read, so OpenBLAS's pool is already
sized.  The pinning below still governs torch, which is imported lazily.

:data:`BITWISE_FLOOR` is that measured BLAS floor; :data:`REFERENCE_FLOOR` is the
slightly larger floor that also absorbs the one deliberate formula difference
from the spec's reference implementation (see
:func:`test_softplus_matches_live_torch_where_the_spec_reference_does_not`).

THE SCALE THRESHOLD (SPEC §11) — CORRECTED 2026-09-27.  There is **no fixed
threshold**.  ``linear_operator`` 0.6.1 takes the exact Cholesky route for the
log-determinant iff ``fast_computations.log_prob.off() or N*T <=
max_cholesky_size.value()`` (``operators/_linear_operator.py:1705``), and both of
those are mutable **process globals**.  ``import botorch`` rewrites them
permanently — measured here, in a subprocess, by
:func:`test_botorch_import_rewrites_the_gpytorch_globals_that_pick_the_route`::

    after gpytorch import   max_cholesky_size= 800   fast_computations(covar_root=True,  log_prob=True,  solves=True)
    after import botorch    max_cholesky_size=4096   fast_computations(covar_root=False, log_prob=False, solves=False)

so with botorch imported the library is exact at *every* size and the stochastic
route is unreachable.  That is not hypothetical: ``test_core_hv.py`` imports the
repo's ``evaluation.py`` as an oracle, ``evaluation.py`` imports botorch, and
running the two files in one pytest process is what used to make this module's
threshold test fail.  Every production entry point (``loop.py``, ``run.py``, the
baselines) imports botorch too, so the repo never runs in the ``N*T > 800`` is
randomized regime that SPEC §11 describes.

Under **vanilla gpytorch** the old statement does hold, and it is the regime
``spec/gp/capture_gp.py`` ran in: above ``N*T = 800`` the log-determinant is
stochastic Lanczos quadrature with probe vectors from the **unseeded global torch
RNG** and ``fast_pred_var()`` is a randomized, biased LOVE decomposition.  Every
fixture case validated here is at ``N*T <= 800``, which is why
``gp.MAX_CHOLESKY_SIZE`` is kept as a record of the capture conditions.
:func:`test_every_validated_case_is_below_the_stochastic_threshold` enumerates
the cases; :func:`test_live_oracle_is_deterministic_below_and_random_above_threshold`
reads the LIVE settings and asserts what follows from them, then forces each
regime in turn so the boundary is still demonstrated whatever the import order.

ORACLE COVERAGE GAP — stated rather than papered over.  The float64 Tanimoto Gram
that every loss, gradient and trained ``theta`` here is built from has **no
library witness and cannot have one**: the shipped ``TanimotoKernel.forward``
begins ``x1 = x1.to(torch.float32)``, so gpytorch returns float32 for a float64
input.  Re-measured 2026-09-28: still true of the library, and still true of the
numpy port (``kernel.TanimotoKernel`` returns ``float32`` for a float64 input, and
bitwise the same values as for the float32 input).
float64 is checked only against a second numpy expression of the same formula
(``capture_gp.tanimoto_gram``) and a byte hash, which is self-consistency, not an
independent oracle.  float32 *is* bitwise witnessed.
:func:`test_the_float64_gram_has_no_library_witness_because_the_kernel_downcasts`
measures both halves and goes red if the kernel ever stops downcasting.

WHAT HAPPENS WITHOUT TORCH — measured, not assumed.  Nothing in this file is
gated by ``pytest.importorskip``, so no test here can go green while proving
nothing.  MEASURED 2026-09-28 with torch, gpytorch, botorch, linear_operator,
sklearn, joblib and scipy all made unimportable by a meta-path blocker:
``101 passed, 37 failed, 0 skipped``.  The 37 are the ``LIVE`` tests, and they
fail LOUDLY on the missing import rather than skipping — a deliberate choice: an
honest red is a signal, a silent skip is not.  No evidence is lost when they go
red, because every claim they make is also made against the archived capture by a
test that needs no library at all:

===========================================================  =============================================================
live test (needs torch)                                      unconditional archived counterpart
===========================================================  =============================================================
``..._matches_LIVE_torch_autograd_per_parameter_group`` (12)  ``test_analytic_gradient_matches_fixture_autograd_per_parameter_group`` (12, same 1e-7, same four groups)
``..._reproduce_a_LIVE_gpytorch_fit`` (4)                     ``test_two_hundred_adam_steps_reproduce_the_archived_hyperparameters`` (7, same 1e-5 abs on theta / 1e-5 rel on tau)
``test_posterior_matches_a_LIVE_mogp_predict`` (4)            ``test_posterior_matches_the_archived_predict`` (6, same 1e-5 abs / 1e-5 rel)
``test_softplus_matches_live_torch_where_the_spec_...``       ``test_softplus_transforms_match_the_fixture_probe_grid``
``test_live_oracle_is_deterministic_below_and_random_...``    ``test_oracle_is_exact_boundary_is_N_times_T`` + ``test_every_validated_case_is_below_the_stochastic_threshold``
``test_the_float64_gram_has_no_library_witness_...``          its own half 1, which is unconditional by construction
``test_dloss_dv_sums_over_tasks_...`` / ``test_mean_gradient_carries_no_half_trace_term``  the positive halves are the fixture-autograd test above; the NEGATIVE controls (a task-0-only ``dloss/dv``, a spurious ``1/2 tr``) are live-only
``test_the_live_oracle_reproduces_the_capture_time_library_bitwise`` (12)  none, and none is possible: it exists to check the rebuilt oracle IS the shipped library, which is vacuous without the library
===========================================================  =============================================================

Run with ``-s`` to see the measured agreement table each test prints.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import os
import sys

# ---------------------------------------------------------------------------
# Thread pinning MUST precede any torch import: this env segfaults on import
# with more than one OpenMP thread (duplicate libomp).  SPEC §13.
# ---------------------------------------------------------------------------
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pytest

REPO = os.path.dirname(os.path.abspath(__file__))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from mogp_core import gp                                      # noqa: E402
from mogp_core import tanimoto as mtan                        # noqa: E402
from mogp_core import transforms as mtr                       # noqa: E402

FIXTURES = os.path.join(REPO, "spec", "gp", "fixtures")
CORE_DIR = os.path.join(REPO, "mogp_core")

#: ``linear_operator.settings.max_cholesky_size`` default (SPEC §11.2).
MAX_CHOLESKY_SIZE = 800

#: Measured multi-threaded-BLAS floor: the worst |loss - archived reference| over
#: all 12 case x setting pairs is 4.440892098500626e-16 at >= 2 threads and
#: exactly 0.0 at 1 thread.  Set ~11x above it.
BITWISE_FLOOR = 5e-15

#: The same, plus the one deliberate formula difference from the spec's reference
#: (``mogp_core.transforms.softplus`` is torch-faithful where the reference's
#: rearrangement is 1 ULP off).  Worst measured: 2.553e-15 on theta after 200
#: compounding Adam steps, 2.665e-15 on the posterior.  Set ~40x above it.
REFERENCE_FLOOR = 1e-13

#: The gradient/posterior fixture cases: ``(label, N, T)``.  All at ``N*T <= 800``.
CASES = (
    ("tiny_n12_t2", 12, 2),
    ("small_n30_t2", 30, 2),
    ("mid_n100_t2", 100, 2),
    ("large_n200_t2", 200, 2),
    ("edge_n400_t2", 400, 2),      # N*T = 800 exactly, the exact side
    ("single_task_n60_t1", 60, 1),
)

#: The two hyperparameter settings the gradient fixtures were captured at.
THETA_SETTINGS = {
    "init_zeros": lambda T: np.zeros(3 * T + 1, dtype=np.float64),
    "asymmetric": lambda T: np.concatenate([
        np.linspace(0.30, -0.40, T),
        np.linspace(0.10, -0.20, T),
        np.linspace(0.50, -0.70, T),
        np.array([-1.00]),
    ]),
}

PARAM_GROUPS = (
    "covar_module.raw_outputscale",
    "mean_module.raw_constant",
    "likelihood.raw_task_noises",
    "likelihood.raw_noise",
)


# ===========================================================================
# Fixture loading
# ===========================================================================

def _unjson(x):
    """Undo ``capture_gp.jnum``: the string sentinels for NaN / +-Infinity."""
    if isinstance(x, str):
        return {"NaN": np.nan, "Infinity": np.inf, "-Infinity": -np.inf}[x]
    return x


def farr(x):
    """A fixture list (possibly nested) as a float64 array, sentinels decoded."""
    o = np.asarray(x, dtype=object)
    flat = np.array([_unjson(v) for v in o.ravel()], dtype=np.float64)
    return flat.reshape(o.shape)


_FIXTURE_CACHE = {}


def fixture(name):
    if name not in _FIXTURE_CACHE:
        with open(os.path.join(FIXTURES, name)) as fh:
            _FIXTURE_CACHE[name] = json.load(fh)
    return _FIXTURE_CACHE[name]


@pytest.fixture(scope="session")
def fingerprints():
    """``data/library/fingerprints.npy``, verified against ``metadata.json``."""
    path = os.path.join(REPO, "data", "library", "fingerprints.npy")
    fp = np.load(path)
    meta = fixture("metadata.json")
    sha = hashlib.sha256(np.ascontiguousarray(fp).tobytes()).hexdigest()
    assert sha == meta["fingerprints_sha256"], (
        "fingerprints.npy has changed since the fixtures were captured: "
        f"{sha} != {meta['fingerprints_sha256']}. Every fixture below is void.")
    assert list(fp.shape) == meta["fingerprints_shape"]
    return fp


# ===========================================================================
# Case construction — byte-for-byte the construction ``capture_gp.py`` used
# ===========================================================================

def synthetic_docking_targets(X, seed=2026_09_27):
    """``capture_gp.synthetic_docking_targets``, reproduced exactly.

    A fixed random projection of the fingerprint, standardized, mapped onto
    ``PfDHFR ~ N(-8.0, 1.2)`` / ``hDHFR ~ N(-7.0, 1.0)`` kcal/mol from an
    explicitly threaded ``default_rng`` — no global RNG is touched.
    """
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((X.shape[1], 2)) / math.sqrt(X.shape[1])
    Z = X.astype(np.float64) @ W
    Z = (Z - Z.mean(axis=0)) / Z.std(axis=0)
    Y = np.full((X.shape[0], len(gp.TASK_NAMES)), np.nan, dtype=np.float32)
    Y[:, 0] = (-8.0 + 1.2 * Z[:, 0]).astype(np.float32)
    Y[:, 1] = (-7.0 + 1.0 * Z[:, 1]).astype(np.float32)
    Y[:, 2] = rng.random(X.shape[0]).astype(np.float32)
    Y[:, 3] = (-6.0 + 2.0 * rng.random(X.shape[0])).astype(np.float32)
    Y[:, 4] = (60.0 * rng.random(X.shape[0])).astype(np.float32)
    return Y


def real_docking_case(fp):
    """The 36 real archived Vina scores, matched back to the library by SMILES."""
    with open(os.path.join(REPO, "data", "library", "smiles.csv"), newline="") as fh:
        lib = [row["SMILES"] for row in csv.DictReader(fh)]
    index = {s: i for i, s in enumerate(lib)}
    ev = os.path.join(REPO, "benchmark_5way_densify_seed0_2026-08-19",
                      "mogp", "seed_0", "evaluated.csv")
    if not os.path.exists(ev):
        return None
    rows = []
    with open(ev, newline="") as fh:
        for r in csv.DictReader(fh):
            i = index.get(r["SMILES"])
            if i is None:
                continue
            try:
                pf, hd = float(r["PfDHFR_Docking"]), float(r["hDHFR_Docking"])
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(pf) and math.isfinite(hd)):
                continue
            rows.append((i, pf, hd))
    rows.sort()
    if len(rows) < 8:
        return None
    idx = np.array([r[0] for r in rows], dtype=np.int64)
    Y = np.full((len(rows), len(gp.TASK_NAMES)), np.nan, dtype=np.float32)
    Y[:, 0] = [r[1] for r in rows]
    Y[:, 1] = [r[2] for r in rows]
    return fp[idx].astype(np.float32), Y


def gradient_case(fp, N, T):
    """``(X, Y_norm, S64)`` as the gradient fixture built them.

    Note ``capture_gp.capture_mll_gradients`` reaches ``T=1`` by passing
    ``docking_task_indices=range(T)``, while the training/posterior captures
    reach it by NaN-ing one ``hDHFR`` row.  Both are faithfully reproduced, in
    :func:`gradient_case` and :func:`training_case` respectively.
    """
    X = fp[:N].astype(np.float32)
    Y5 = synthetic_docking_targets(X)
    y_mean, y_std, observed = gp.standardize(Y5, tuple(range(T)))
    Yn = ((Y5[:, observed] - y_mean[observed]) / y_std[observed]).astype(np.float32)
    S64 = gp.gram(X, X, dtype=np.float64)
    return X, Yn, S64


def training_case(fp, N, T):
    """``(X, Y5)`` as the training / posterior fixtures built them."""
    X = fp[:N].astype(np.float32)
    Y5 = synthetic_docking_targets(X)
    if T == 1:
        Y5 = Y5.copy()
        Y5[3, 1] = np.nan          # one NaN row drops the WHOLE hDHFR column
    return X, Y5


# ===========================================================================
# The live oracle — the ONLY place torch / gpytorch is imported
#
# THE ORACLE MOVED INTO THIS FILE (2026-09-28).  It used to be obtained by
# ``import mogp``, whose ``MOGPModel`` was a ``gpytorch.models.ExactGP`` over
# ``kernel.TanimotoKernel``.  The numpy integration rewrote both modules, so that
# import now yields the implementation under test instead of a library oracle:
# every ``LIVE`` test in this file failed with ``'MOGPModel' object has no
# attribute 'covar_module'``.
#
# The gpytorch model is therefore rebuilt HERE, in the test instrument, as a
# VERBATIM transcription of git ``3c6573c`` — ``kernel.py``'s ``TanimotoKernel``
# and ``mogp.py``'s ``MOGPModel`` / ``train_mogp`` / ``predict``, which is exactly
# the code ``spec/gp/capture_gp.py`` ran.  Building the library's classes in the
# test file (rather than importing the repo's) is what the dependency rule in
# TIER 0 demands, and it is strictly the same oracle as before, not a weaker one.
#
# The transcription is NOT taken on trust.
# :func:`test_the_live_oracle_reproduces_the_capture_time_library_bitwise`
# re-derives every archived ``loss_gpytorch`` and ``grad_gpytorch_autograd`` from
# it and requires agreement to be EXACTLY 0.0 at all 12 case x setting pairs, so
# a transcription error cannot pass unnoticed.
# ===========================================================================

_ORACLE = {}


def oracle():
    """Import torch/gpytorch lazily, build the library model, cache the handles."""
    if _ORACLE:
        return _ORACLE
    import torch
    torch.set_num_threads(1)
    import gpytorch
    import linear_operator.settings as lo_settings

    class _LiveTanimotoKernel(gpytorch.kernels.Kernel):
        """``3c6573c:kernel.py``'s ``TanimotoKernel``, character for character.

        Note the opening ``x1.to(torch.float32)``: this is the downcast that
        makes a float64 library witness impossible (see
        :func:`test_the_float64_gram_has_no_library_witness_because_the_kernel_downcasts`).
        """

        is_stationary = False

        def forward(self, x1, x2, diag=False, **params):
            x1 = x1.to(torch.float32)
            x2 = x2.to(torch.float32)

            eps = 1e-8

            if diag:
                intersection = (x1 * x2).sum(dim=-1)
                sum1 = x1.sum(dim=-1)
                sum2 = x2.sum(dim=-1)
                union = sum1 + sum2 - intersection
                sim = intersection / union.clamp(min=eps)
                return sim.clamp(0.0, 1.0)

            intersection = x1 @ x2.transpose(-1, -2)
            sum1 = x1.sum(dim=-1, keepdim=True)
            sum2 = x2.sum(dim=-1, keepdim=True)
            union = sum1 + sum2.transpose(-1, -2) - intersection
            sim = intersection / union.clamp(min=eps)
            return sim.clamp(0.0, 1.0)

    class _LiveMOGPModel(gpytorch.models.ExactGP):
        """``3c6573c:mogp.py``'s ``MOGPModel``, character for character."""

        def __init__(self, train_x, train_y, likelihood):
            super().__init__(train_x, train_y, likelihood)
            num_tasks = train_y.shape[1]
            self.mean_module = gpytorch.means.ConstantMean(
                batch_shape=torch.Size([num_tasks])
            )
            self.covar_module = gpytorch.kernels.ScaleKernel(
                _LiveTanimotoKernel(),
                batch_shape=torch.Size([num_tasks]),
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
                gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
            )

    _ORACLE.update(torch=torch, gpytorch=gpytorch, lo_settings=lo_settings,
                   TanimotoKernel=_LiveTanimotoKernel, MOGPModel=_LiveMOGPModel)
    return _ORACLE


def live_train_mogp(train_x, train_y, n_iterations=200, lr=0.1):
    """``3c6573c:mogp.py``'s ``train_mogp``, character for character.

    Only the two cosmetic lines differ: the per-20-step ``print`` is dropped
    (``oracle_train`` used to swallow it with ``redirect_stdout``), and the
    gpytorch classes come from :func:`oracle` instead of module globals.  Every
    numerical statement — the population ``std``, the zero-variance guard, the
    docking mask, the NaN "not modelled" marking, the Adam construction over
    ``model.parameters()`` and the 200-step loop — is unchanged.
    """
    o = oracle()
    torch, gpytorch = o["torch"], o["gpytorch"]

    train_x_t = torch.from_numpy(np.asarray(train_x)).to(torch.float32)
    train_y = np.asarray(train_y, dtype=np.float32)

    y_mean = train_y.mean(axis=0)
    y_std = train_y.std(axis=0)
    y_std = np.where(y_std == 0.0, 1.0, y_std)

    # ``mogp.DOCKING_TASK_INDICES`` was ``[0, 1]``; the value is taken from the
    # capture's own record rather than from the module under test, so the oracle
    # borrows nothing from ``mogp_core``.
    docking_task_indices = fixture("metadata.json")["docking_task_indices"]
    assert docking_task_indices == [0, 1], docking_task_indices
    docking_mask = np.zeros(train_y.shape[1], dtype=bool)
    docking_mask[[j for j in docking_task_indices
                  if j < train_y.shape[1]]] = True
    y_mean = np.where(docking_mask, y_mean, np.nan)
    y_std = np.where(docking_mask, y_std, np.nan)

    observed = np.isfinite(y_mean) & np.isfinite(y_std)
    if not observed.any():
        raise ValueError(
            "live_train_mogp: no observed docking target columns to train on.")

    train_y_norm = (train_y[:, observed] - y_mean[observed]) / y_std[observed]
    train_y_t = torch.from_numpy(train_y_norm).to(torch.float32)

    num_tasks = int(observed.sum())
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
        num_tasks=num_tasks)
    model = o["MOGPModel"](train_x_t, train_y_t, likelihood)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for _ in range(n_iterations):
        optimizer.zero_grad()
        output = model(train_x_t)
        loss = -mll(output, train_y_t)
        loss.backward()
        optimizer.step()

    return model, likelihood, y_mean, y_std


def live_predict(model, likelihood, y_mean, y_std, X_new):
    """``3c6573c:mogp.py``'s ``predict``, character for character.

    Including the ``gpytorch.settings.fast_pred_var()`` context, which is part of
    the behaviour under comparison and not an implementation detail.
    """
    o = oracle()
    torch, gpytorch = o["torch"], o["gpytorch"]
    X_new_t = torch.from_numpy(np.asarray(X_new)).to(torch.float32)

    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = likelihood(model(X_new_t))
        mean_obs = posterior.mean.cpu().numpy()
        variance_obs = posterior.variance.cpu().numpy()

    observed = np.isfinite(y_mean) & np.isfinite(y_std)
    n_rows = mean_obs.shape[0]
    n_tasks = y_mean.shape[0]
    mean = np.full((n_rows, n_tasks), np.nan, dtype=float)
    variance = np.full((n_rows, n_tasks), np.nan, dtype=float)
    mean[:, observed] = mean_obs * y_std[observed] + y_mean[observed]
    variance[:, observed] = variance_obs * (y_std[observed] ** 2)
    return mean, variance


def live_settings():
    """The LIVE ``linear_operator`` globals that pick exact vs randomized.

    Read, never assumed.  These are process-global mutable defaults and any
    import can rewrite them for good — ``import botorch`` does, which is the
    whole reason this module's threshold tests cannot hard-code 800.
    """
    ls = oracle()["lo_settings"]
    return {
        "max_cholesky_size": int(ls.max_cholesky_size.value()),
        "fast_log_prob": bool(ls.fast_computations.log_prob.on()),
        "fast_covar_root": bool(
            ls.fast_computations.covar_root_decomposition.on()),
        "fast_solves": bool(ls.fast_computations.solves.on()),
        "cholesky_max_tries": int(ls.cholesky_max_tries.value()),
    }


def live_mll_route_is_exact(N, T, settings=None):
    """Will ``inv_quad_logdet`` take the exact Cholesky branch at ``N*T``?

    Not a guess: this is ``linear_operator`` 0.6.1's own condition, from
    ``linear_operator/operators/_linear_operator.py:1705``::

        if settings.fast_computations.log_prob.off() or (
                self.size(-1) <= settings.max_cholesky_size.value()):
            ...  # exact Cholesky

    so an *off* ``log_prob`` flag makes the size irrelevant, and
    :data:`MAX_CHOLESKY_SIZE` alone cannot answer the question.
    """
    s = live_settings() if settings is None else settings
    return (not s["fast_log_prob"]) or (N * T <= s["max_cholesky_size"])


def capture_time_settings():
    """Context manager forcing the settings ``capture_gp.py`` ran under.

    Vanilla gpytorch: ``max_cholesky_size = 800`` and all three
    ``fast_computations`` flags on.  Used to demonstrate the ``N*T = 800``
    boundary itself no matter what an earlier import did to the globals.
    """
    import contextlib
    ls = oracle()["lo_settings"]
    stack = contextlib.ExitStack()
    stack.enter_context(ls.max_cholesky_size(MAX_CHOLESKY_SIZE))
    stack.enter_context(ls.fast_computations(covar_root_decomposition=True,
                                            log_prob=True, solves=True))
    return stack


def live_mll_spread(fingerprints, N, T, reps=8):
    """``(spread, bias_vs_mine)`` of ``reps`` IDENTICAL live MLL calls.

    ``spread = max - min`` over repeats of the very same call on the very same
    model: exactly 0 on the Cholesky route, nonzero when the log-determinant is
    stochastic Lanczos quadrature with unseeded probes.
    """
    o = oracle()
    gpytorch = o["gpytorch"]
    X = fingerprints[:N].astype(np.float32)
    Y5 = synthetic_docking_targets(X)
    y_mean, y_std, observed = gp.standardize(Y5, tuple(range(T)))
    Yn = ((Y5[:, observed] - y_mean[observed]) / y_std[observed]
          ).astype(np.float32)
    model, lik, Xt, Yt = build_oracle_model(X, Yn)
    model.train()
    lik.train()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
    losses = [float(-mll(model(Xt), Yt).item()) for _ in range(reps)]
    spread = max(losses) - min(losses)
    S = gp.gram(X, X, dtype=np.float64)
    mine, _ = gp.nmll_and_grad(np.zeros(3 * T + 1), S,
                               Yn.astype(np.float64), want_grad=False)
    return spread, float(np.mean(losses)) - mine


def build_oracle_model(X, Yn, theta=None):
    """``MOGPModel`` + ``MultitaskGaussianLikelihood``, optionally set at theta."""
    o = oracle()
    torch, gpytorch = o["torch"], o["gpytorch"]
    T = Yn.shape[1]
    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
    Yt = torch.from_numpy(np.ascontiguousarray(Yn, dtype=np.float32))
    lik = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=T)
    model = o["MOGPModel"](Xt, Yt, lik)
    if theta is not None:
        u, c, w, v = gp.unpack(np.asarray(theta, dtype=np.float64), T)
        with torch.no_grad():
            model.covar_module.raw_outputscale.copy_(
                torch.tensor(u, dtype=torch.float32))
            model.mean_module.raw_constant.copy_(
                torch.tensor(c, dtype=torch.float32))
            lik.raw_task_noises.copy_(torch.tensor(w, dtype=torch.float32))
            lik.raw_noise.copy_(
                torch.tensor(np.atleast_1d(v), dtype=torch.float32))
    return model, lik, Xt, Yt


def oracle_loss_and_grads(X, Yn, theta):
    """``-ExactMarginalLogLikelihood`` and autograd's per-parameter gradients."""
    o = oracle()
    gpytorch = o["gpytorch"]
    model, lik, Xt, Yt = build_oracle_model(X, Yn, theta)
    model.train()
    lik.train()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
    model.zero_grad()
    loss = -mll(model(Xt), Yt)
    loss.backward()
    grads = {n: p.grad.detach().numpy().astype(np.float64).copy()
             for n, p in model.named_parameters()}
    return float(loss.item()), grads


def oracle_theta(model, lik):
    """Read the oracle's raw parameters back out in the spec's theta layout."""
    return np.concatenate([
        model.covar_module.raw_outputscale.detach().numpy().astype(np.float64),
        model.mean_module.raw_constant.detach().numpy().astype(np.float64),
        lik.raw_task_noises.detach().numpy().astype(np.float64),
        lik.raw_noise.detach().numpy().astype(np.float64),
    ])


def oracle_train(X, Y5, n_steps=200, lr=0.1):
    """``3c6573c:mogp.train_mogp``, rebuilt here — see :func:`live_train_mogp`."""
    return live_train_mogp(X, Y5, n_iterations=n_steps, lr=lr)


# ===========================================================================
# Reporting helpers
# ===========================================================================

def maxabs(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def maxrel(a, b, floor=1e-12):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0:
        return 0.0
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), floor)))


def report(line):
    print("  MEASURED " + line)


# ===========================================================================
# TIER 0 — the dependency rule
# ===========================================================================

FORBIDDEN = ("torch", "gpytorch", "botorch", "scipy", "sklearn",
             "linear_operator")


def test_mogp_core_imports_only_numpy_pandas_and_stdlib():
    """No module under ``mogp_core/`` may import a forbidden dependency.

    Checked by parsing the AST rather than by importing, so a lazily-imported or
    conditionally-imported offender is still caught.
    """
    offenders = []
    for fn in sorted(os.listdir(CORE_DIR)):
        if not fn.endswith(".py"):
            continue
        path = os.path.join(CORE_DIR, fn)
        with open(path) as fh:
            tree = ast.parse(fh.read(), filename=path)
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for nm in names:
                root = nm.split(".")[0]
                if root in FORBIDDEN:
                    offenders.append(f"{fn}:{node.lineno} imports {nm}")
    assert not offenders, "forbidden imports inside mogp_core/: " + "; ".join(offenders)
    report(f"dependency scan: {len([f for f in os.listdir(CORE_DIR) if f.endswith('.py')])} "
           f"modules under mogp_core/, 0 forbidden imports")


def test_gp_module_itself_has_no_forbidden_import_at_runtime():
    """``mogp_core.gp`` must be importable with torch absent from the process."""
    assert "mogp_core.gp" in sys.modules
    for name in FORBIDDEN:
        assert not hasattr(gp, name), f"mogp_core.gp exposes {name}"


# ===========================================================================
# TIER 1 — closed form, no oracle needed (SPEC §14 tier 1)
# ===========================================================================

@pytest.mark.parametrize("T", [1, 2, 3, 5])
def test_parameter_count_is_three_T_plus_one(T):
    """``3T + 1``, not ``3T``: the shared global noise is a separate scalar."""
    assert gp.n_params(T) == 3 * T + 1
    theta = np.zeros(3 * T + 1)
    u, c, w, v = gp.unpack(theta, T)
    assert u.size == c.size == w.size == T
    assert np.ndim(v) == 0, "raw_noise is ONE shared scalar, not one per task"
    with pytest.raises(ValueError):
        gp.unpack(np.zeros(3 * T), T)          # 3T must be rejected


def test_gram_diagonal_is_exactly_one_and_matrix_exactly_symmetric(fingerprints):
    """No diagonal special-case: the general formula already gives exactly 1."""
    for N in (12, 30, 100, 200, 400):
        X = fingerprints[:N].astype(np.float32)
        assert (X.sum(axis=1) > 0).all(), "a zero fingerprint would give S[i,i]=0"
        for dtype in (np.float32, np.float64):
            S = gp.gram(X, X, dtype=dtype)
            assert np.all(np.diag(S) == 1.0), f"N={N} {dtype}: diag != 1 exactly"
            assert np.array_equal(S, S.T), f"N={N} {dtype}: not bitwise symmetric"
            assert S.min() >= 0.0 and S.max() <= 1.0
    report("gram: diag == 1.0 exactly and S == S.T bitwise at N in "
           "{12,30,100,200,400}, float32 and float64")


def test_gram_matches_fixture_bytes_exactly(fingerprints):
    """``S`` is bitwise portable, so the fixture sha256 must match exactly."""
    fx = fixture("tanimoto_gram.json")
    full12 = farr(fx["full_matrix_n12_float64"])
    S12 = gp.gram(fingerprints[:12].astype(np.float32),
                  fingerprints[:12].astype(np.float32), dtype=np.float64)
    assert np.array_equal(S12, full12), (
        f"n12 Gram differs from the fixture: max abs {maxabs(S12, full12):.3e}")
    for N in (12, 30, 100, 200, 400):
        X = fingerprints[:N].astype(np.float32)
        S = gp.gram(X, X, dtype=np.float64)
        sha = hashlib.sha256(np.ascontiguousarray(S).tobytes()).hexdigest()
        want = fx["by_size"][f"n{N}"]["sha256_float64_C_order"]
        assert sha == want, f"N={N}: sha256 {sha} != fixture {want}"
        assert float(np.sum(S)) == fx["by_size"][f"n{N}"]["sum"]
    report("gram: sha256 of the C-order float64 bytes matches the fixture "
           "EXACTLY at N in {12,30,100,200,400}; n12 full matrix bitwise")


def test_the_float64_gram_has_no_library_witness_because_the_kernel_downcasts(
        fingerprints):
    """Name the oracle gap instead of implying full coverage.

    Every loss, gradient and trained ``theta`` in ``mogp_core.gp`` is built from
    ``gram(X, X, dtype=np.float64)``, and **no library comparison for it is
    possible**: the shipped ``TanimotoKernel.forward`` opens with
    ``x1 = x1.to(torch.float32)``, so gpytorch returns float32 for a float64
    input and cannot be asked for the float64 Gram at all.  float64 is therefore
    checked only against a second numpy expression of the same formula
    (``capture_gp.tanimoto_gram``) plus a byte hash — self-consistency, not an
    independent oracle.  float32 *does* have a bitwise witness, which this test
    also re-measures here so the two statements sit side by side.

    REPAIRED 2026-09-28, same claim, same tolerances.  The downcast used to be
    read off ``kernel.TanimotoKernel``, which was the gpytorch class; ``kernel.py``
    is now a numpy adapter, so a ``torch.float32`` dtype assertion against it
    could not hold.  The claim has two halves and both are still asserted, each
    against the right object:

    * the LIBRARY downcasts — measured on the gpytorch kernel rebuilt by
      :func:`oracle`, which is why no float64 witness exists;
    * the PORT preserves the downcast — measured on the repo's own
      ``kernel.TanimotoKernel``, unconditionally, because that is the frozen
      behaviour every archived number was produced through.

    If either ever stops downcasting, this goes red and a real float64 witness
    becomes possible — at which point ``mogp_core.gp``'s docstring, which records
    this gap, must be re-measured.
    """
    X64 = np.ascontiguousarray(fingerprints[:24].astype(np.float64))
    X32 = np.ascontiguousarray(fingerprints[:24].astype(np.float32))

    # -- half 1, unconditional: the PORT still silently downcasts float64 ------
    from kernel import TanimotoKernel
    numpy_out64 = TanimotoKernel().forward(X64, X64)
    assert numpy_out64.dtype == np.float32, (
        f"kernel.TanimotoKernel no longer downcasts: a float64 input returned "
        f"{numpy_out64.dtype}. Every archived number was produced through that "
        f"downcast, so this is frozen behaviour, not a bug to fix.")
    numpy_out32 = TanimotoKernel().forward(X32, X32)
    assert np.array_equal(numpy_out64, numpy_out32), (
        "the float64-input result is not bitwise identical to the float32-input "
        "result, so the downcast is no longer the only thing happening")

    # -- half 2, live: the LIBRARY downcasts, hence no float64 witness ---------
    o = oracle()
    torch = o["torch"]
    K = o["TanimotoKernel"]()
    out64 = K.forward(torch.from_numpy(X64), torch.from_numpy(X64))
    assert out64.dtype == torch.float32, (
        f"the gpytorch TanimotoKernel no longer downcasts: a float64 input "
        f"returned {out64.dtype}. The float64 Gram now HAS a library witness and "
        f"should be compared against it; mogp_core.gp's docstring says it cannot "
        f"be.")

    # What the library CAN witness: float32, bitwise.
    got32 = K.forward(torch.from_numpy(X32), torch.from_numpy(X32)).numpy()
    mine32 = gp.gram(X32, X32, dtype=np.float32)
    assert np.array_equal(got32, mine32), (
        f"float32 gram is not bitwise vs the live kernel: "
        f"{maxabs(got32, mine32):.3e}")

    # The float64 Gram the module actually uses differs from the only thing the
    # library will hand over, so agreement at float32 does not certify float64.
    mine64 = gp.gram(X32, X32, dtype=np.float64)
    gap = maxabs(got32.astype(np.float64), mine64)
    assert gap > 0.0, (
        "float32 and float64 Grams are bitwise equal on this slice, so this "
        "test cannot show the gap; pick a slice where they differ")
    report("oracle coverage: float32 gram is BITWISE vs the live gpytorch "
           f"TanimotoKernel (0.0); float64 has NO witness — the library returns "
           f"{out64.dtype} for a float64 input and the numpy port returns "
           f"{numpy_out64.dtype} (bitwise the same values as its float32 input), "
           f"and |library_float32 - mine_float64| = {gap:.3e} is the uncertified "
           f"margin")


def test_zero_fingerprint_gives_zero_self_similarity():
    """``T(0, b) == 0`` for every ``b`` including itself — a landmine, not a crash."""
    X = np.zeros((3, 16), dtype=np.int8)
    X[1, :4] = 1
    X[2, 4:8] = 1
    S = gp.gram(X, X, dtype=np.float64)
    assert S[0, 0] == 0.0, "all-zero fingerprint must give S[0,0] = 0, not 1"
    assert np.all(S[0, :] == 0.0) and np.all(S[:, 0] == 0.0)
    assert gp._self_similarity(X, dtype=np.float64)[0] == 0.0


def test_self_similarity_is_bitwise_equal_to_the_gram_diagonal(fingerprints):
    """The ``O(M*D)`` shortcut must not change a single bit of ``k_**``."""
    for rows in (slice(0, 64), slice(20000, 20064)):
        Xs = fingerprints[rows].astype(np.float32)
        for dtype in (np.float32, np.float64):
            d1 = np.diag(gp.gram(Xs, Xs, dtype=dtype))
            d2 = gp._self_similarity(Xs, dtype=dtype)
            assert np.array_equal(d1, d2), (
                f"{dtype}: tanimoto_diag != diag(gram), max abs "
                f"{maxabs(d1, d2):.3e}")
    report("k_** shortcut: tanimoto_diag == diag(tanimoto_gram) BITWISE "
           "(max abs diff exactly 0.0), float32 and float64")


def test_softplus_transforms_match_the_fixture_probe_grid():
    """``Positive()`` IS softplus; ``GreaterThan(lb)`` IS ``lb + softplus``."""
    rows = fixture("model_parameters.json")["constraint_transforms"]["rows"]
    worst_pos = 0.0
    worst_off = 0.0
    for r in rows:
        raw = np.array([_unjson(r["raw"])], dtype=np.float32)
        mine = float(mtr.positive(raw)[0])
        worst_pos = max(worst_pos, abs(mine - _unjson(r["gpytorch_Positive"])))
        gt = float(mtr.greater_than(raw, lower=gp.NOISE_LOWER_BOUND)[0])
        worst_off = max(worst_off, abs((gt - mine)
                                       - _unjson(r["gpytorch_GreaterThan_1e-4_minus_Positive"])))
    assert worst_pos <= 1e-6, (
        f"positive() vs the fixture's gpytorch Positive: max abs {worst_pos:.3e}")
    assert worst_off <= 1e-6, (
        f"greater_than offset vs the fixture: max abs {worst_off:.3e}")
    # The linear branch is active strictly above 20 and returns z bit-for-bit.
    for z in (20.1, 30.0, 100.0, 1e30):
        a = np.array([z], dtype=np.float64)
        assert float(mtr.positive(a)[0]) == z, f"linear branch broken at z={z}"
        assert float(mtr.softplus_grad(a)[0]) == 1.0
    assert float(mtr.positive(np.array([20.0], dtype=np.float64))[0]) != 20.0, (
        "the cutover is STRICTLY > 20; z == 20 must take the smooth branch")
    report(f"transforms vs fixture probe grid (float32): max|Positive - "
           f"softplus| = {worst_pos:.3e}, max|GreaterThan offset - 1e-4| = "
           f"{worst_off:.3e}; no exp() overflow at z = 1e30")


def test_initialization_is_exactly_zero_with_the_expected_natural_values():
    """All ``3T+1`` raw parameters start at exactly 0 (SPEC §3.4) — no RNG."""
    for item in mtr.parameter_inventory(2):
        assert item["init"] == 0.0
    s, mu, d, sigma2, tau = gp.natural(np.zeros(7), 2)
    assert np.allclose(s, math.log(2.0), rtol=0, atol=1e-15)
    assert np.all(mu == 0.0)
    assert np.allclose(d, 1e-4 + math.log(2.0), rtol=0, atol=1e-15)
    assert abs(sigma2 - (1e-4 + math.log(2.0))) <= 1e-15
    assert np.allclose(tau, 2 * (1e-4 + math.log(2.0)), rtol=0, atol=1e-15)
    report(f"init: theta = 0 exactly -> s = {float(s[0]):.6f}, "
           f"d = sigma2 = {float(sigma2):.6f}, tau = {float(tau[0]):.6f}")


def test_mean_is_unconstrained_identity_not_softplus():
    """``mu_t = c_t``.  A softplus here would make negative means unreachable."""
    c = np.array([-3.0, 0.0, 2.5])
    theta = np.concatenate([np.zeros(3), c, np.zeros(3), [0.0]])
    _, mu, _, _, _ = gp.natural(theta, 3)
    assert np.array_equal(mu, c), "the constant mean must be the raw value itself"
    assert mu is not theta[3:6], "natural() must not alias raw storage"


def test_the_flat_direction_leaves_the_loss_exactly_unchanged(fingerprints):
    """``(sigma2 + e, d - e)`` holds every ``tau_t`` fixed, so ``L`` is EXACT-equal.

    Reproduces ``model_parameters.json -> flat_direction``, which measured a loss
    change of exactly ``0.0e+00``.  The consequence is that ``sigma2`` and ``d_t``
    are not separately identifiable and tests must be written on ``tau_t``.
    """
    fx = fixture("model_parameters.json")["flat_direction"]
    S = gp.gram(fingerprints[:60].astype(np.float32),
                fingerprints[:60].astype(np.float32), dtype=np.float64)
    Yf = np.random.default_rng(20260927).standard_normal((60, 2))

    def loss_natural(s, mu, sigma2, d):
        total = 0.0
        for t in range(2):
            K = s[t] * S + (sigma2 + d[t]) * np.eye(60)
            L = gp.cholesky(K)
            r = Yf[:, t] - mu[t]
            a = np.linalg.solve(L.T, np.linalg.solve(L, r))
            total += (-0.5 * r @ a - np.log(np.diag(L)).sum()
                      - 0.5 * 60 * math.log(2 * math.pi))
        return -total / 120.0

    s0 = np.array([0.8, 0.5])
    mu0 = np.array([0.1, -0.1])
    base = loss_natural(s0, mu0, 0.30, np.array([0.20, 0.40]))
    assert abs(base - _unjson(fx["base_loss"])) <= 1e-12, (
        f"base loss {base!r} != fixture {fx['base_loss']!r}")
    for shift in fx["shifts"]:
        e = _unjson(shift["eps"])
        delta = loss_natural(s0, mu0, 0.30 + e, np.array([0.20, 0.40]) - e) - base
        assert delta == 0.0, f"eps={e}: loss changed by {delta:.3e}, expected EXACTLY 0"
        assert _unjson(shift["loss_change"]) == 0.0
    report("flat direction: loss change under (sigma2+e, d-e) is EXACTLY 0.0 at "
           "e in {1e-3, 1e-2, 5e-2, 1e-1} — sigma2/d_t unidentifiable, "
           "Hessian singular by construction")


def test_standardize_uses_population_std_ddof_zero(fingerprints):
    """``ddof=0``.  ``ddof=1`` (pandas' default) is a silent ~1.7% shift."""
    X = fingerprints[:30].astype(np.float32)
    Y5 = synthetic_docking_targets(X)
    _, y_std, observed = gp.standardize(Y5)
    col = Y5[:, 0].astype(np.float32)
    ddof0 = np.std(col, ddof=0)
    ddof1 = np.std(col, ddof=1)
    assert y_std[0] == ddof0, f"y_std[0]={y_std[0]!r} != ddof=0 std {ddof0!r}"
    assert y_std[0] != ddof1, "ddof=1 must NOT pass"
    report(f"ddof: y_std[0] = {float(ddof0):.8f} (ddof=0); ddof=1 would give "
           f"{float(ddof1):.8f}, a {100 * abs(ddof1 / ddof0 - 1):.2f}% shift")


def test_standardization_nan_semantics_match_every_fixture_case(fingerprints):
    """All six cases: observed mask and trained-task count must match exactly."""
    fx = fixture("standardization_nan.json")
    assert fx["std_ddof"] == 0
    X = fingerprints[:30].astype(np.float32)
    base = synthetic_docking_targets(X)
    specs = {
        "all_docking_observed": {},
        "one_nan_row_in_PfDHFR": {0: [5]},
        "half_nan_rows_in_PfDHFR": {0: list(range(15))},
        "all_nan_PfDHFR": {0: list(range(30))},
        "all_nan_both_docking": {0: list(range(30)), 1: list(range(30))},
        "constant_PfDHFR_column": "constant",
    }
    seen = 0
    for entry in fx["cases"]:
        name = entry["case"]
        Y = base.copy()
        spec = specs[name]
        if spec == "constant":
            Y[:, 0] = -8.0
        else:
            for col, rows in spec.items():
                Y[rows, col] = np.nan
        y_mean, y_std, observed = gp.standardize(Y)
        assert [bool(b) for b in observed] == entry["observed_mask"], (
            f"{name}: observed {observed.tolist()} != {entry['observed_mask']}")
        assert int(observed.sum()) == entry["n_tasks_trained"], (
            f"{name}: T={int(observed.sum())} != {entry['n_tasks_trained']}")
        fm, fs = farr(entry["y_mean"]), farr(entry["y_std"])
        assert np.array_equal(np.isnan(y_mean), np.isnan(fm))
        assert np.array_equal(np.isnan(y_std), np.isnan(fs))
        assert np.allclose(y_mean[observed], fm[observed], rtol=0, atol=0), (
            "y_mean must match the fixture BITWISE, got max abs "
            f"{maxabs(y_mean[observed], fm[observed]):.3e}")
        assert np.allclose(y_std[observed], fs[observed], rtol=0, atol=0)
        if entry["train_mogp_raises"] is None:
            state, lik, ym, ys = gp.train_mogp(X, Y, n_iterations=2)
            assert lik is None, "the drop-in 4-tuple keeps None in slot 1"
            assert state.n_tasks == entry["n_tasks_trained"]
        else:
            with pytest.raises(ValueError) as ei:
                gp.train_mogp(X, Y, n_iterations=2)
            assert str(ei.value) == entry["train_mogp_raises"].split(": ", 1)[1], (
                f"raise message differs:\n got {str(ei.value)!r}\n"
                f" want {entry['train_mogp_raises']!r}")
        seen += 1
    assert seen == 6
    report("NaN contract: all 6 standardization cases match the fixture exactly "
           "(y_mean/y_std bitwise on observed columns); ONE NaN row out of 30 "
           "drops the whole PfDHFR column (T: 2 -> 1), 29 good rows discarded")


def test_the_three_admet_columns_are_always_dropped_even_when_finite(fingerprints):
    """The grey-box mask is applied AFTER the statistics, so values never matter."""
    X = fingerprints[:20].astype(np.float32)
    Y = np.zeros((20, 5), dtype=np.float32)
    Y[:, :] = np.arange(5, dtype=np.float32)[None, :]
    Y[:, 0] += np.linspace(0, 1, 20)
    Y[:, 1] += np.linspace(0, 1, 20)
    y_mean, y_std, observed = gp.standardize(Y)
    assert observed.tolist() == [True, True, False, False, False]
    assert np.all(np.isnan(y_mean[2:])) and np.all(np.isnan(y_std[2:]))


# ===========================================================================
# TIER 2 — the MLL and its analytic gradients
# ===========================================================================

def test_the_mll_divisor_is_N_times_T_not_N(fingerprints):
    """Measured, not inferred: gpytorch divides by the event shape's numel.

    At ``N=6, T=2`` the mll is ``log_prob/12``; ``log_prob/6`` is off by 1.446.
    For ``T=1`` the two coincide, which is exactly why testing single-task first
    hides this.
    """
    nc = fixture("mll_gradients.json")["normalization_check"]
    assert nc["event_shape"] == [6, 2] and nc["event_shape_numel"] == 12
    X = fingerprints[:6].astype(np.float32)
    Yn = np.asarray(np.random.default_rng(5).standard_normal((6, 2)),
                    dtype=np.float32)
    theta = THETA_SETTINGS["asymmetric"](2)
    S = gp.gram(X, X, dtype=np.float64)
    loss, _ = gp.nmll_and_grad(theta, S, Yn.astype(np.float64), want_grad=False)

    d_nt = abs(loss - _unjson(nc["numpy_reference_neg_logprob_over_NT"]))
    assert d_nt <= BITWISE_FLOOR, (
        f"loss vs the fixture reference: {d_nt:.3e} > {BITWISE_FLOOR:.1e} "
        f"(bitwise at 1 BLAS thread, 4.44e-16 above that)")
    d_mll = abs(loss + _unjson(nc["mll_value"]))
    assert d_mll <= 1e-6, f"loss vs live-captured gpytorch mll: {d_mll:.3e}"
    # And the /N variant must be nowhere near: this is what makes the test bite.
    d_n = abs(loss + _unjson(nc["raw_log_prob_over_N"]))
    assert d_n > 1.0, (
        f"dividing by N instead of N*T is only {d_n:.3e} away — the test cannot "
        f"distinguish them and is worthless")
    report(f"MLL divisor: loss = {loss:.10f} reproduces -mll to {d_mll:.3e}; "
           f"the /N variant is {d_n:.4f} away (T=1 would make them coincide)")


@pytest.mark.parametrize("label,N,T", CASES)
@pytest.mark.parametrize("setting", ["init_zeros", "asymmetric"])
def test_loss_matches_the_fixture(fingerprints, label, N, T, setting):
    """Loss value vs the archived gpytorch value and the archived reference."""
    fx = fixture("mll_gradients.json")["cases"][label]
    assert fx["below_or_at_max_cholesky_size"] is True
    assert fx["N_times_T"] == N * T <= MAX_CHOLESKY_SIZE
    entry = fx["settings"][setting]
    X, Yn, S64 = gradient_case(fingerprints, N, T)
    theta = THETA_SETTINGS[setting](T)
    assert np.allclose(theta, farr(entry["theta_raw"]), rtol=0, atol=0)
    loss, _ = gp.nmll_and_grad(theta, S64, Yn.astype(np.float64), want_grad=False)
    d_ref = abs(loss - _unjson(entry["loss_numpy_analytic_float64"]))
    d_gpy = abs(loss - _unjson(entry["loss_gpytorch"]))
    assert d_ref <= BITWISE_FLOOR, (
        f"vs the fixture's own reference: {d_ref:.3e} > {BITWISE_FLOOR:.1e}")
    assert d_gpy <= 1e-6, f"vs archived gpytorch: {d_gpy:.3e} (tol 1e-6)"
    report(f"loss {label}/{setting}: |mine - gpytorch| = {d_gpy:.3e}, "
           f"|mine - fixture reference| = {d_ref:.3e}")


@pytest.mark.parametrize("label,N,T", CASES)
@pytest.mark.parametrize("setting", ["init_zeros", "asymmetric"])
def test_analytic_gradient_matches_central_finite_differences(
        fingerprints, label, N, T, setting):
    """Torch-free confirmation of the whole §7 derivation, per parameter."""
    X, Yn, S64 = gradient_case(fingerprints, N, T)
    theta = THETA_SETTINGS[setting](T)
    Y64 = Yn.astype(np.float64)
    _, g = gp.nmll_and_grad(theta, S64, Y64)
    h = 1e-6
    fd = np.empty_like(theta)
    for i in range(theta.size):
        tp = theta.copy(); tp[i] += h
        tm = theta.copy(); tm[i] -= h
        fd[i] = (gp.nmll_and_grad(tp, S64, Y64, want_grad=False)[0]
                 - gp.nmll_and_grad(tm, S64, Y64, want_grad=False)[0]) / (2 * h)
    worst = maxabs(g, fd)
    entry = fixture("mll_gradients.json")["cases"][label]["settings"][setting]
    d_fd = maxabs(fd, farr(entry["grad_central_fd_h1e-6"]))
    # h = 1e-6, so a 4.4e-16 BLAS-order change in the loss is amplified to ~2e-10
    # here.  Measured worst over the whole case set: see the reported number.
    assert d_fd <= 1e-8, (
        f"my finite differences differ from the fixture's by {d_fd:.3e}")
    assert worst <= 3e-9, (
        f"analytic vs central FD (h=1e-6): max abs {worst:.3e} > 3e-9; "
        f"per-parameter {np.abs(g - fd)!r}")
    report(f"grad-vs-FD {label}/{setting}: max abs = {worst:.3e} (tol 3e-9)")


@pytest.mark.parametrize("label,N,T", CASES)
@pytest.mark.parametrize("setting", ["init_zeros", "asymmetric"])
def test_analytic_gradient_matches_fixture_autograd_per_parameter_group(
        fingerprints, label, N, T, setting):
    """Each of the four parameter groups checked SEPARATELY.

    A max over all ``3T+1`` scalars lets a sign error in one group hide behind a
    larger honest disagreement in another, so the four groups are asserted one at
    a time and each measured value is reported.
    """
    entry = fixture("mll_gradients.json")["cases"][label]["settings"][setting]
    X, Yn, S64 = gradient_case(fingerprints, N, T)
    theta = THETA_SETTINGS[setting](T)
    _, g = gp.nmll_and_grad(theta, S64, Yn.astype(np.float64))
    mine = gp.split_grad(g, T)
    measured = {}
    for key in PARAM_GROUPS:
        want_auto = farr(entry["grad_gpytorch_autograd"][key])
        want_ref = farr(entry["grad_numpy_analytic"][key])
        d_auto = maxabs(mine[key], want_auto)
        d_ref = maxabs(mine[key], want_ref)
        measured[key] = (d_auto, d_ref)
        assert d_ref <= BITWISE_FLOOR, (
            f"{key}: vs the fixture's own reference max abs {d_ref:.3e} > "
            f"{BITWISE_FLOOR:.1e} (bitwise at 1 BLAS thread)")
        assert d_auto <= 1e-7, (
            f"{key}: vs archived torch autograd max abs {d_auto:.3e} > 1e-7 "
            f"(mine {mine[key]!r} vs {want_auto!r})")
    report(f"grad-vs-archived-autograd {label}/{setting}: "
           + ", ".join(f"{k.split('.')[-1]}={v[0]:.3e}"
                       for k, v in measured.items()))


@pytest.mark.parametrize("label,N,T", CASES)
@pytest.mark.parametrize("setting", ["init_zeros", "asymmetric"])
def test_the_live_oracle_reproduces_the_capture_time_library_bitwise(
        fingerprints, label, N, T, setting):
    """The provenance check for the rebuilt oracle — see :func:`oracle`.

    The gpytorch model the ``LIVE`` tests compare against is now transcribed into
    this file rather than imported from ``mogp.py``, so the transcription itself
    has to be evidence-backed or every ``LIVE`` result below is worthless.  This
    runs it and requires it to reproduce the archived ``loss_gpytorch`` and
    ``grad_gpytorch_autograd`` — the numbers ``spec/gp/capture_gp.py`` recorded
    from the real shipped module on 2026-09-27 — EXACTLY, with zero tolerance.

    Bitwise is the right bar and not an optimistic one: both sides are the same
    library at the same version doing the same float32 arithmetic on the same
    inputs, single-threaded.  Anything nonzero here means the transcription is not
    the shipped model and the ``LIVE`` comparisons are measuring the wrong thing.

    It has to run under :func:`capture_time_settings`, for the reason the module
    docstring gives: ``import botorch`` permanently turns ``fast_computations``'
    ``covar_root_decomposition`` and ``solves`` off, and that changes the backward
    pass's internal route.  MEASURED with ``test_core_hv.py`` (which imports
    ``evaluation.py``, which imports botorch) in the same process and the settings
    NOT forced: ``loss_gpytorch`` still reproduces at exactly ``0.0`` in all 12
    pairs, while the autograd groups move by up to ``7.451e-09`` — one to two ULPs
    of a float32 gradient of that magnitude — in 9 of the 12.  Forcing the
    capture-time settings makes the comparison independent of import order, which
    is what a provenance check has to be.
    """
    entry = fixture("mll_gradients.json")["cases"][label]["settings"][setting]
    X, Yn, _ = gradient_case(fingerprints, N, T)
    theta = THETA_SETTINGS[setting](T)
    with capture_time_settings():
        live_loss, live_grads = oracle_loss_and_grads(X, Yn, theta)
    d_loss = abs(live_loss - entry["loss_gpytorch"])
    assert d_loss == 0.0, (
        f"the rebuilt gpytorch oracle gives loss {live_loss!r} where the capture "
        f"recorded {entry['loss_gpytorch']!r} (diff {d_loss:.3e}): the "
        f"transcription in oracle() is NOT the model capture_gp.py ran")
    worst = 0.0
    for key in PARAM_GROUPS:
        d = maxabs(live_grads[key], farr(entry["grad_gpytorch_autograd"][key]))
        assert d == 0.0, (
            f"{key}: the rebuilt oracle's autograd differs from the archived "
            f"autograd by {d:.3e}, expected exactly 0.0")
        worst = max(worst, d)
    report(f"oracle provenance {label}/{setting}: rebuilt gpytorch model vs the "
           f"archived capture — loss diff {d_loss:.3e}, autograd diff {worst:.3e} "
           f"(both must be exactly 0.0)")


@pytest.mark.parametrize("label,N,T", CASES)
@pytest.mark.parametrize("setting", ["init_zeros", "asymmetric"])
def test_analytic_gradient_matches_LIVE_torch_autograd_per_parameter_group(
        fingerprints, label, N, T, setting):
    """The same comparison against gpytorch running right now, not archived.

    The model is the one :func:`oracle` rebuilds; that it *is* the shipped model
    is established by
    :func:`test_the_live_oracle_reproduces_the_capture_time_library_bitwise`.
    """
    X, Yn, S64 = gradient_case(fingerprints, N, T)
    theta = THETA_SETTINGS[setting](T)
    live_loss, live_grads = oracle_loss_and_grads(X, Yn, theta)
    loss, g = gp.nmll_and_grad(theta, S64, Yn.astype(np.float64))
    mine = gp.split_grad(g, T)
    assert abs(loss - live_loss) <= 1e-6, (
        f"loss vs LIVE gpytorch: {abs(loss - live_loss):.3e}")
    measured = {}
    for key in PARAM_GROUPS:
        d = maxabs(mine[key], live_grads[key])
        measured[key] = d
        assert d <= 1e-7, (
            f"{key}: vs LIVE autograd max abs {d:.3e} > 1e-7 "
            f"(mine {mine[key]!r} vs live {live_grads[key]!r})")
    report(f"grad-vs-LIVE-autograd {label}/{setting}: loss "
           f"{abs(loss - live_loss):.3e}, "
           + ", ".join(f"{k.split('.')[-1]}={v:.3e}" for k, v in measured.items()))


def test_dloss_dv_sums_over_tasks_and_a_task_zero_only_version_fails(fingerprints):
    """The shared-noise coupling is load-bearing and must be detectable.

    A ``dloss/dv`` that uses only task 0 (i.e. treats the tasks as independent
    GPs, as ``mogp.MOGPModel``'s docstring wrongly claims) is computed here and
    shown to disagree with autograd by far more than the tolerance — so the
    gradient test above genuinely pins the coupling rather than passing by luck.
    """
    N, T = 30, 2
    X, Yn, S64 = gradient_case(fingerprints, N, T)
    theta = THETA_SETTINGS["asymmetric"](T)          # asymmetric w, as required
    Y64 = Yn.astype(np.float64)
    _, g = gp.nmll_and_grad(theta, S64, Y64)

    # Recover the per-task g^tau from the raw w-gradients (SPEC §7.5).
    u, c, w, v = gp.unpack(theta, T)
    NT = float(N * T)
    g_tau = -g[2 * T:3 * T] * NT / mtr.softplus_grad(w)
    sig_v = float(mtr.softplus_grad(np.reshape(v, (1,)))[0])
    correct = -(g_tau.sum() * sig_v) / NT
    task0_only = -(g_tau[0] * sig_v) / NT

    assert abs(g[3 * T] - correct) <= 1e-12, "internal inconsistency"
    _, live = oracle_loss_and_grads(X, Yn, theta)
    live_v = float(live["likelihood.raw_noise"][0])
    d_correct = abs(correct - live_v)
    d_task0 = abs(task0_only - live_v)
    assert d_correct <= 1e-7, f"the summed dloss/dv misses autograd by {d_correct:.3e}"
    assert d_task0 > 1e-3, (
        f"a task-0-only dloss/dv is only {d_task0:.3e} from autograd — the "
        f"coupling is not actually being tested here")
    report(f"shared-noise coupling: summed dloss/dv is {d_correct:.3e} from LIVE "
           f"autograd; the task-0-only version is {d_task0:.3e} away "
           f"({d_task0 / max(d_correct, 1e-30):.0f}x worse) -> the test bites")


def test_mean_gradient_carries_no_half_trace_term(fingerprints):
    """``dell/dmu = sum(alpha)``, with no ``1/2 tr(...)``.

    Applying R&W eq. 5.9 to the mean is the commonest hand-rolled-GP error, so
    the wrong version is constructed here and shown to fail loudly.
    """
    N, T = 30, 2
    X, Yn, S64 = gradient_case(fingerprints, N, T)
    theta = THETA_SETTINGS["asymmetric"](T)
    Y64 = Yn.astype(np.float64)
    _, g = gp.nmll_and_grad(theta, S64, Y64)
    s, mu, d, sigma2, tau = gp.natural(theta, T)
    NT = float(N * T)
    wrong = np.empty(T)
    for t in range(T):
        K = s[t] * S64 + tau[t] * np.eye(N)
        L = gp.cholesky(K)
        r = Y64[:, t] - mu[t]
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, r))
        Kinv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(N)))
        # dK/dmu is ZERO, so eq. 5.9 would contribute -0.5*tr(K^-1 * 0) = 0;
        # the classic error is to reuse the dtau expression instead.
        bogus = 0.5 * (alpha @ alpha - np.trace(Kinv))
        wrong[t] = -(alpha.sum() + bogus) / NT
    _, live = oracle_loss_and_grads(X, Yn, theta)
    live_c = live["mean_module.raw_constant"]
    d_right = maxabs(g[T:2 * T], live_c)
    d_wrong = maxabs(wrong, live_c)
    assert d_right <= 1e-7, f"dloss/dc misses autograd by {d_right:.3e}"
    assert d_wrong > 1e-2, (
        f"the spurious-trace version is only {d_wrong:.3e} from autograd")
    report(f"mean gradient: sum(alpha) form is {d_right:.3e} from LIVE autograd; "
           f"adding a 1/2 tr(...) term puts it {d_wrong:.3e} away")


def test_primitive_triangular_solves_give_the_same_gradient(fingerprints):
    """``mogp_core.linalg``'s substitutions vs ``np.linalg.solve``, measured.

    The primitives layer reports that its ``np.sum`` accumulation differs from a
    BLAS dot by ~1e-16 relative in float64.  This quantifies what that costs the
    GP layer, at a size small enough for the O(n) Python loop to be affordable.
    """
    N, T = 30, 2
    X, Yn, S64 = gradient_case(fingerprints, N, T)
    Y64 = Yn.astype(np.float64)
    worst_loss = 0.0
    worst_grad = 0.0
    for setting in ("init_zeros", "asymmetric"):
        theta = THETA_SETTINGS[setting](T)
        l1, g1 = gp.nmll_and_grad(theta, S64, Y64, use_primitive_solves=False)
        l2, g2 = gp.nmll_and_grad(theta, S64, Y64, use_primitive_solves=True)
        worst_loss = max(worst_loss, abs(l1 - l2))
        worst_grad = max(worst_grad, maxabs(g1, g2))
    assert worst_loss <= REFERENCE_FLOOR, f"loss differs by {worst_loss:.3e}"
    assert worst_grad <= REFERENCE_FLOOR, f"gradient differs by {worst_grad:.3e}"
    report(f"solver route (N=30, T=2, float64): np.linalg.solve vs "
           f"mogp_core.linalg substitution -> max |dloss| = {worst_loss:.3e}, "
           f"max |dgrad| = {worst_grad:.3e}")


def test_softplus_matches_live_torch_where_the_spec_reference_does_not():
    """The ONE deliberate formula difference from the spec's reference code.

    SPEC §3.3's reference writes softplus as ``log1p(exp(-|z|)) + max(z, 0)``;
    ``mogp_core.transforms.softplus`` writes torch's own
    ``log1p(exp(z))`` on the smooth branch.  They are algebraically identical.
    In float32 the primitives' form is **bit-exact against live torch** at
    ``z = 1`` while the spec's rearrangement is one ULP off, so this module is the
    more library-faithful of the two — which is why the difference is kept rather
    than "corrected" back to the spec's text.  Everything downstream absorbs it
    inside :data:`REFERENCE_FLOOR`.
    """
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(1)
    grid = np.array([-100, -30, -5, -1, 0, 1, 5, 15, 19.9, 20.0, 20.1, 30, 100])
    out = {}
    for np_dt, t_dt in ((np.float32, torch.float32), (np.float64, torch.float64)):
        g = grid.astype(np_dt)
        live = F.softplus(torch.tensor(g, dtype=t_dt)).numpy()
        mine = mtr.softplus(g)
        spec = (np.log1p(np.exp(-np.abs(np.where(g > 20.0, np_dt(0.0), g))))
                + np.maximum(np.where(g > 20.0, np_dt(0.0), g), np_dt(0.0)))
        spec = np.where(g > 20.0, g, spec).astype(np_dt)
        out[np.dtype(np_dt).name] = (maxabs(mine, live), maxabs(spec, live))
    m32, s32 = out["float32"]
    m64, s64 = out["float64"]
    assert m64 == 0.0, f"float64 softplus is not bitwise vs live torch: {m64:.3e}"
    assert s64 == 0.0
    assert m32 < s32, (
        f"the primitives' float32 softplus ({m32:.3e} from torch) should be at "
        f"least as faithful as the spec's rearrangement ({s32:.3e})")
    report(f"softplus vs LIVE torch: float64 bitwise for both forms (0.0); "
           f"float32 max abs mine = {m32:.3e} vs the SPEC reference's "
           f"{s32:.3e} -> mogp_core is the more faithful form; the ~1e-15 "
           f"REFERENCE_FLOOR downstream is this, not an error")


def test_blas_thread_count_changes_the_float64_loss_by_one_ulp():
    """Answers SPEC §13's explicitly UNVERIFIED reproducibility question.

    Runs the identical loss/gradient computation in two subprocesses, one with
    every BLAS thread variable pinned to 1 and one pinned to 8, and compares both
    against the archived numpy reference.  At one thread the agreement is
    bitwise; above one thread it is one ULP.  Nothing here is tolerated by a loose
    bound — both numbers are asserted.
    """
    import subprocess
    script = r"""
import sys, json, math
sys.path.insert(0, %r)
import numpy as np
from mogp_core import gp
fp = np.load(%r)
rng = np.random.default_rng(20260927)
X = fp[:200].astype(np.float32)
W = rng.standard_normal((X.shape[1], 2)) / math.sqrt(X.shape[1])
Z = X.astype(np.float64) @ W
Z = (Z - Z.mean(0)) / Z.std(0)
Y = np.full((200, 5), np.nan, dtype=np.float32)
Y[:, 0] = (-8.0 + 1.2 * Z[:, 0]).astype(np.float32)
Y[:, 1] = (-7.0 + 1.0 * Z[:, 1]).astype(np.float32)
ym, ys, obs = gp.standardize(Y, (0, 1))
Yn = ((Y[:, obs] - ym[obs]) / ys[obs]).astype(np.float32)
S = gp.gram(X, X, dtype=np.float64)
loss, g = gp.nmll_and_grad(np.zeros(7), S, Yn.astype(np.float64))
print(json.dumps({"loss": repr(loss), "grad": [repr(v) for v in g]}))
""" % (REPO, os.path.join(REPO, "data", "library", "fingerprints.npy"))
    results = {}
    for n in (1, 8):
        env = dict(os.environ)
        for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
            env[var] = str(n)
        proc = subprocess.run([sys.executable, "-c", script], env=env,
                              capture_output=True, text=True)
        assert proc.returncode == 0, f"{n} threads: {proc.stderr[-2000:]}"
        d = json.loads(proc.stdout.strip().splitlines()[-1])
        results[n] = (float(eval(d["loss"])),
                      np.array([float(eval(v)) for v in d["grad"]]))

    entry = fixture("mll_gradients.json")["cases"]["large_n200_t2"]["settings"]["init_zeros"]
    want_loss = _unjson(entry["loss_numpy_analytic_float64"])
    want_grad = np.concatenate([farr(entry["grad_numpy_analytic"][k])
                                for k in PARAM_GROUPS[:1]])
    d1_loss = abs(results[1][0] - want_loss)
    d8_loss = abs(results[8][0] - want_loss)
    d18_loss = abs(results[1][0] - results[8][0])
    d18_grad = maxabs(results[1][1], results[8][1])
    d1_grad = maxabs(results[1][1][:2], want_grad)

    assert d1_loss == 0.0, (
        f"at OMP_NUM_THREADS=1 the loss must be BITWISE vs the archived "
        f"reference, got {d1_loss:.3e}")
    assert d1_grad == 0.0, (
        f"at OMP_NUM_THREADS=1 the outputscale gradient must be BITWISE, got "
        f"{d1_grad:.3e}")
    assert d8_loss <= BITWISE_FLOOR, (
        f"at 8 threads the loss is {d8_loss:.3e} from the reference, beyond the "
        f"{BITWISE_FLOOR:.1e} floor this suite is built on")
    assert d18_grad <= BITWISE_FLOOR
    report(f"BLAS threads (N=200, T=2, float64): OMP=1 gives loss and gradient "
           f"BITWISE vs the archived reference (0.0 / 0.0); OMP=8 gives loss "
           f"{d8_loss:.3e} and 1-vs-8 differs by loss {d18_loss:.3e}, gradient "
           f"{d18_grad:.3e} -> reproducible to 1 ULP, bitwise only single-threaded")


def test_cholesky_never_adds_jitter_and_equals_numpy(fingerprints):
    """``tau_t`` IS the jitter (SPEC §2.2); no separate term is ever added."""
    X = fingerprints[:100].astype(np.float32)
    S = gp.gram(X, X, dtype=np.float64)
    K = 0.7 * S + 1.386 * np.eye(100)
    assert np.array_equal(gp.cholesky(K), np.linalg.cholesky(K))
    # A duplicate row makes S singular; tau alone must still carry it.
    Xd = np.vstack([fingerprints[:20], fingerprints[:1]]).astype(np.float32)
    Sd = gp.gram(Xd, Xd, dtype=np.float64)
    ev = float(np.linalg.eigvalsh(Sd).min())
    for tau in (2e-4, 1e-3, 1e-2):
        L = gp.cholesky(0.7 * Sd + tau * np.eye(21))
        assert np.isfinite(L).all() and (np.diag(L) > 0).all()
    with pytest.raises(np.linalg.LinAlgError):
        gp.cholesky(-np.eye(4))                 # no rescue by jitter
    with pytest.raises(FloatingPointError):
        gp.cholesky(np.array([[np.nan, 0.0], [0.0, 1.0]]))
    report(f"cholesky: bitwise == np.linalg.cholesky, zero jitter ever; with one "
           f"duplicate row min eig(S) = {ev:.3e} and tau in "
           f"{{2e-4, 1e-3, 1e-2}} all factor")


# ===========================================================================
# TIER 3 — the 200-step fit
# ===========================================================================

TRAIN_CASES = CASES + (("real_docking_n36_t2", 36, 2),)


def _train_inputs(fingerprints, label, N, T):
    if label == "real_docking_n36_t2":
        got = real_docking_case(fingerprints)
        assert got is not None, "the archived evaluated.csv is missing"
        X, Y5 = got
        assert X.shape[0] == N, f"real case has {X.shape[0]} rows, fixture says {N}"
        return X, Y5
    return training_case(fingerprints, N, T)


@pytest.mark.parametrize("label,N,T", TRAIN_CASES)
def test_two_hundred_adam_steps_reproduce_the_archived_hyperparameters(
        fingerprints, label, N, T):
    """The headline number: a spec-only numpy fit vs the archived gpytorch fit.

    Asserted on ``tau_t`` (identified) with a relative tolerance, and on
    ``sigma2`` / ``d_t`` separately only loosely, because those two live on the
    exactly-flat direction of SPEC §6.1 and where Adam lands along it is a
    property of the path, not of the likelihood.
    """
    fx = fixture("trained_hyperparameters.json")["cases"][label]
    assert fx["below_or_at_max_cholesky_size"] is True
    assert fx["N_times_T"] == N * T <= MAX_CHOLESKY_SIZE
    assert fixture("trained_hyperparameters.json")["adam"] == {
        "lr": 0.1, "betas": [0.9, 0.999], "eps": 1e-8, "n_steps": 200,
        "init_raw": "all zeros", "weight_decay": 0.0, "amsgrad": False}
    X, Y5 = _train_inputs(fingerprints, label, N, T)

    state, lik, y_mean, y_std = gp.train_mogp(X, Y5, dtype=np.float64)
    assert lik is None
    assert state.n_tasks == fx["T_trained"] == T
    assert [bool(b) for b in state.observed] == fx["observed_mask_full_5col"]
    theta = np.asarray(state.theta, dtype=np.float64)

    d_ref = maxabs(theta, farr(fx["theta_raw_numpy_float64"]))
    d_gpy = maxabs(theta, farr(fx["theta_raw_gpytorch"]))
    s, mu, d, sigma2, tau = state.natural()
    nat = fx["natural_gpytorch"]
    d_tau = maxabs(tau, farr(nat["tau_effective"]))
    r_tau = maxrel(tau, farr(nat["tau_effective"]))
    d_os = maxabs(s, farr(nat["outputscale"]))
    d_sig = abs(float(sigma2) - _unjson(nat["shared_sigma2"]))
    d_d = maxabs(d, farr(nat["task_noise_d"]))

    assert d_ref <= REFERENCE_FLOOR, (
        f"vs the fixture's own float64 reference: max abs {d_ref:.3e} > "
        f"{REFERENCE_FLOOR:.1e}")
    assert d_gpy <= 1e-5, f"vs archived gpytorch theta: max abs {d_gpy:.3e}"
    assert r_tau <= 1e-5, f"tau relative: {r_tau:.3e}"
    assert d_os <= 1e-6, f"outputscale abs: {d_os:.3e}"
    # Deliberately loose: the flat direction (SPEC §6.1). Do NOT tighten.
    assert d_sig <= 1e-4, f"shared sigma2 abs: {d_sig:.3e}"
    assert d_d <= 1e-4, f"task noise d abs: {d_d:.3e}"

    assert maxabs([d_gpy], [_unjson(fx["agreement"]["max_abs_theta_numpy64_vs_gpytorch"])]) <= 1e-9, (
        f"my theta-vs-gpytorch gap {d_gpy:.6e} differs from the gap the fixture "
        f"recorded for its own reference "
        f"{_unjson(fx['agreement']['max_abs_theta_numpy64_vs_gpytorch']):.6e}")
    report(f"fit {label} (N={N}, T={T}, N*T={N * T}): |theta - gpytorch| = "
           f"{d_gpy:.3e}, |theta - fixture ref| = {d_ref:.3e}, tau rel = "
           f"{r_tau:.3e}, outputscale abs = {d_os:.3e}, sigma2 abs = "
           f"{d_sig:.3e} (flat direction), d abs = {d_d:.3e} (flat direction)")


@pytest.mark.parametrize("label,N,T", [c for c in TRAIN_CASES if c[1] <= 200])
def test_float32_fit_also_reproduces_the_archived_hyperparameters(
        fingerprints, label, N, T):
    """The repo's declared GP dtype is float32; that path is checked too."""
    fx = fixture("trained_hyperparameters.json")["cases"][label]
    X, Y5 = _train_inputs(fingerprints, label, N, T)
    state, _, _, _ = gp.train_mogp(X, Y5, dtype=np.float32)
    theta = np.asarray(state.theta, dtype=np.float64)
    d_ref32 = maxabs(theta, farr(fx["theta_raw_numpy_float32"]))
    d_gpy = maxabs(theta, farr(fx["theta_raw_gpytorch"]))
    assert d_gpy <= 1e-5, f"float32 fit vs archived gpytorch: {d_gpy:.3e}"
    assert d_ref32 <= 1e-5, (
        f"float32 fit vs the fixture's float32 reference: {d_ref32:.3e}")
    report(f"fit-float32 {label}: |theta - gpytorch| = {d_gpy:.3e}, "
           f"|theta - fixture float32 ref| = {d_ref32:.3e}")


@pytest.mark.parametrize("label,N,T", [c for c in CASES if c[1] <= 100])
def test_two_hundred_adam_steps_reproduce_a_LIVE_gpytorch_fit(
        fingerprints, label, N, T):
    """Refit with the shipped ``train_mogp`` right now and compare.

    ``mogp.train_mogp`` is now numpy, so the gpytorch fit is the one
    :func:`live_train_mogp` rebuilds: the same 200 Adam steps at ``lr=0.1`` over
    ``model.parameters()``, the same population-std standardization and the same
    grey-box NaN masking, run live.  Same tolerances as before (1e-5 on theta,
    1e-5 relative on tau).
    """
    X, Y5 = training_case(fingerprints, N, T)
    model, lik, ym_o, ys_o = oracle_train(X, Y5)
    th_live = oracle_theta(model, lik)
    state, _, ym, ys = gp.train_mogp(X, Y5, dtype=np.float64)
    assert np.array_equal(np.isnan(ym), np.isnan(ym_o))
    obs = np.isfinite(ym)
    assert np.array_equal(ym[obs], ym_o[obs]), "standardization must be bitwise"
    assert np.array_equal(ys[obs], ys_o[obs])
    d = maxabs(state.theta, th_live)
    s, _, _, _, tau = state.natural()
    _, _, _, _, tau_live = gp.natural(th_live, T)
    r_tau = maxrel(tau, tau_live)
    assert d <= 1e-5, f"vs LIVE gpytorch theta: max abs {d:.3e}"
    assert r_tau <= 1e-5, f"vs LIVE gpytorch tau: max rel {r_tau:.3e}"
    report(f"fit-vs-LIVE {label}: max |theta| = {d:.3e}, tau max rel = {r_tau:.3e}")


def test_the_two_adam_arrangements_agree(fingerprints):
    """``mogp_core.optim.Adam`` vs ``torch.optim.Adam``'s literal arrangement.

    ``lr*(m/bc1)/(sqrt(v/bc2)+eps)`` against
    ``(lr/bc1)*m/(sqrt(v)/sqrt(bc2)+eps)``: algebraically identical, different
    rounding.  This measures what 200 compounding steps do with that difference,
    so nobody has to guess whether the choice matters.
    """
    X, Y5 = training_case(fingerprints, 100, 2)
    y_mean, y_std, observed = gp.standardize(Y5)
    Yn = ((Y5[:, observed] - y_mean[observed]) / y_std[observed]).astype(np.float32)
    S = gp.gram(X, X, dtype=np.float64)
    t_prim, l_prim = gp.fit(X, Yn, S=S, adam="primitives")
    t_spec, l_spec = gp.fit(X, Yn, S=S, adam="spec_torch")
    d_theta = maxabs(t_prim, t_spec)
    d_loss = abs(l_prim[-1] - l_spec[-1])
    fxref = farr(fixture("trained_hyperparameters.json")
                 ["cases"]["mid_n100_t2"]["theta_raw_numpy_float64"])
    assert d_theta <= REFERENCE_FLOOR, (
        f"the two arrangements diverged by {d_theta:.3e}")
    report(f"Adam arrangement (N=100, T=2, 200 steps, float64): max |theta| gap "
           f"= {d_theta:.3e}, final-loss gap = {d_loss:.3e}; vs fixture "
           f"reference primitives={maxabs(t_prim, fxref):.3e}, "
           f"spec_torch={maxabs(t_spec, fxref):.3e}")


def test_the_loss_trace_is_finite_and_still_descending_at_step_200(fingerprints):
    """200 steps, unconditionally; the fit is not converged and is not meant to be."""
    fx = fixture("trained_hyperparameters.json")["cases"]["mid_n100_t2"]
    X, Y5 = training_case(fingerprints, 100, 2)
    y_mean, y_std, observed = gp.standardize(Y5)
    Yn = ((Y5[:, observed] - y_mean[observed]) / y_std[observed]).astype(np.float32)
    theta, losses = gp.fit(X, Yn)
    assert losses.shape == (200,)
    assert np.isfinite(losses).all()
    assert losses[-1] < losses[0]
    assert losses[-1] < losses[-2], "still descending at step 200"
    first5 = farr(fx["numpy_loss_trace_first5"])
    last5 = farr(fx["numpy_loss_trace_last5"])
    assert maxabs(losses[:5], first5) <= REFERENCE_FLOOR, (
        f"loss trace head vs fixture: {maxabs(losses[:5], first5):.3e}")
    assert maxabs(losses[-5:], last5) <= REFERENCE_FLOOR, (
        f"loss trace tail vs fixture: {maxabs(losses[-5:], last5):.3e}")
    report(f"loss trace (mid_n100_t2): {losses[0]:.7f} -> {losses[-1]:.7f}, "
           f"last step still -{losses[-2] - losses[-1]:.2e}; head/tail match the "
           f"fixture to {max(maxabs(losses[:5], first5), maxabs(losses[-5:], last5)):.3e}")


def test_fit_touches_no_rng(fingerprints):
    """No random initialization anywhere: repeated fits are bitwise identical."""
    X, Y5 = training_case(fingerprints, 30, 2)
    y_mean, y_std, observed = gp.standardize(Y5)
    Yn = ((Y5[:, observed] - y_mean[observed]) / y_std[observed]).astype(np.float32)
    a, _ = gp.fit(X, Yn, n_steps=40)
    np.random.seed(1234)
    np.random.default_rng(99).standard_normal(10)
    b, _ = gp.fit(X, Yn, n_steps=40)
    assert np.array_equal(a, b), "the fit is not RNG-independent"


# ===========================================================================
# TIER 3 — the posterior
# ===========================================================================

@pytest.mark.parametrize("label,N,T", CASES)
def test_posterior_matches_the_archived_predict(fingerprints, label, N, T):
    """Mean and variance vs ``mogp.predict``'s archived output, de-standardized."""
    fx = fixture("posterior.json")["cases"][label]
    assert fx["N_times_T"] <= MAX_CHOLESKY_SIZE
    Xs = fingerprints[20000:20016].astype(np.float32)
    X, Y5 = training_case(fingerprints, N, T)
    y_mean, y_std, observed = gp.standardize(Y5)
    Yn = ((Y5[:, observed] - y_mean[observed]) / y_std[observed]).astype(np.float32)
    theta = farr(fx["theta_raw"])          # the theta the ORACLE reached
    state = gp.GPState(theta, X, Yn, observed)
    mean, var = gp.predict(state, y_mean, y_std, Xs, observation_noise=True)

    f_mean = farr(fx["predict_mean_5col"])
    f_var = farr(fx["predict_variance_5col"])
    d_mean = maxabs(mean[:, observed], f_mean[:, observed])
    d_var = maxabs(var[:, observed], f_var[:, observed])
    r_var = maxrel(var[:, observed], f_var[:, observed])
    r_mean = maxrel(mean[:, observed], f_mean[:, observed])
    d_ref = maxabs(mean[:, observed], farr(fx["numpy_mean_observed_cols"]))
    v_ref = maxabs(var[:, observed], farr(fx["numpy_marginal_variance_observed_cols"]))

    assert d_mean <= 1e-5, f"posterior mean abs: {d_mean:.3e} kcal/mol"
    assert r_var <= 1e-5, f"posterior variance rel: {r_var:.3e}"
    assert d_ref <= REFERENCE_FLOOR, (
        f"mean vs the fixture's own reference: {d_ref:.3e} > {REFERENCE_FLOOR:.1e}")
    assert v_ref <= REFERENCE_FLOOR, (
        f"variance vs the fixture's own reference: {v_ref:.3e} > "
        f"{REFERENCE_FLOOR:.1e}")
    # The std column is sqrt(variance): y_std**2 on variance, not y_std.
    f_std = farr(fx["predict_std_5col"])
    d_std = maxabs(np.sqrt(var[:, observed]), f_std[:, observed])
    assert d_std <= 1e-5, f"posterior std abs: {d_std:.3e}"
    report(f"posterior {label}: mean abs = {d_mean:.3e} (rel {r_mean:.3e}), var "
           f"abs = {d_var:.3e} (rel {r_var:.3e}), std abs = {d_std:.3e}; vs "
           f"fixture reference mean {d_ref:.3e} / var {v_ref:.3e}")


@pytest.mark.parametrize("label,N,T", CASES)
def test_predict_returns_the_noise_inclusive_marginal_not_the_latent_variance(
        fingerprints, label, N, T):
    """``predict`` returns ``var_f + tau`` on the original scale (SPEC §9.2).

    The gap is the WHOLE of ``tau_t``, which at these fits is comparable to or
    larger than the posterior variance itself, so a test that could not tell the
    two apart would be worthless.  The second assertion pins that.
    """
    fx = fixture("posterior.json")["cases"][label]
    Xs = fingerprints[20000:20016].astype(np.float32)
    X, Y5 = training_case(fingerprints, N, T)
    y_mean, y_std, observed = gp.standardize(Y5)
    Yn = ((Y5[:, observed] - y_mean[observed]) / y_std[observed]).astype(np.float32)
    theta = farr(fx["theta_raw"])
    state = gp.GPState(theta, X, Yn, observed)
    _, var_marg = gp.predict(state, y_mean, y_std, Xs, observation_noise=True)
    _, var_lat = gp.predict(state, y_mean, y_std, Xs, observation_noise=False)
    tau_orig = farr(fx["tau_effective_original_scale"])
    resid = maxabs(var_marg[:, observed] - var_lat[:, observed],
                   np.broadcast_to(tau_orig, var_lat[:, observed].shape))
    gap = maxabs(var_marg[:, observed], var_lat[:, observed])
    f_var = farr(fx["predict_variance_5col"])
    d_lat = maxabs(var_lat[:, observed], f_var[:, observed])
    assert resid <= 1e-6, f"var - var_f - tau*y_std^2 = {resid:.3e}, expected 0"
    assert gap > 1e-3, (
        f"var and var_f differ by only {gap:.3e}; this test cannot distinguish "
        f"them and proves nothing")
    assert d_lat > 1e-3, (
        f"returning var_f instead of var_y would only be {d_lat:.3e} from the "
        f"archived predict output — the fixture cannot tell them apart")
    # The gap IS max(tau * y_std^2), exactly.
    assert abs(gap - float(np.max(tau_orig))) <= 1e-12, (
        f"|var - var_f| = {gap:.6e} should equal max(tau*y_std^2) = "
        f"{float(np.max(tau_orig)):.6e}")
    # The fixture's own figure is oracle_variance - reference_latent, so it also
    # carries the oracle's float32 error; agreement there is at the 1e-8 level.
    d_fx = abs(gap - _unjson(fx["which_variance_predict_returns"]
                             ["max_abs_vs_latent_f"]))
    assert d_fx <= 1e-6, (
        f"my tau gap differs from the fixture's oracle-minus-latent figure by "
        f"{d_fx:.3e} (expected ~1e-8, the oracle's float32 noise)")
    report(f"marginal-vs-latent {label}: |var - var_f - tau*y_std^2| = "
           f"{resid:.3e}; |var - var_f| = {gap:.3e} kcal^2/mol^2 (the whole of "
           f"tau, and == max(tau*y_std^2) to {abs(gap - float(np.max(tau_orig))):.1e}); "
           f"latent-instead-of-marginal would miss the archived predict output "
           f"by {d_lat:.3e}")


@pytest.mark.parametrize("label,N,T", CASES)
def test_unmodelled_columns_are_all_nan(fingerprints, label, N, T):
    """The three ADMET columns are NaN by contract, not by failure."""
    fx = fixture("posterior.json")["cases"][label]
    Xs = fingerprints[20000:20016].astype(np.float32)
    X, Y5 = training_case(fingerprints, N, T)
    y_mean, y_std, observed = gp.standardize(Y5)
    Yn = ((Y5[:, observed] - y_mean[observed]) / y_std[observed]).astype(np.float32)
    state = gp.GPState(farr(fx["theta_raw"]), X, Yn, observed)
    mean, var = gp.predict(state, y_mean, y_std, Xs)
    unmodelled = np.flatnonzero(~observed)
    assert len(unmodelled) == len(fx["unmodelled_columns_all_nan"])
    for j, want in zip(unmodelled, fx["unmodelled_columns_all_nan"]):
        assert want is True
        assert np.all(np.isnan(mean[:, j])), f"column {j} is not all NaN"
        assert np.all(np.isnan(var[:, j]))
    assert np.isfinite(mean[:, observed]).all()
    assert np.isfinite(var[:, observed]).all()


def test_variance_carries_y_std_squared_not_y_std(fingerprints):
    """SPEC §9.3: a bare ``y_std`` on the variance is a silent ~20% error.

    At ``y_std ~ 1.2`` the wrong version is only ~20% off, which no smoke test
    would catch, so it is constructed here and shown to be far outside the
    posterior tolerance.
    """
    fx = fixture("posterior.json")["cases"]["mid_n100_t2"]
    Xs = fingerprints[20000:20016].astype(np.float32)
    X, Y5 = training_case(fingerprints, 100, 2)
    y_mean, y_std, observed = gp.standardize(Y5)
    Yn = ((Y5[:, observed] - y_mean[observed]) / y_std[observed]).astype(np.float32)
    state = gp.GPState(farr(fx["theta_raw"]), X, Yn, observed)
    _, var = gp.predict(state, y_mean, y_std, Xs)
    f_var = farr(fx["predict_variance_5col"])
    ys = y_std[observed].astype(np.float64)
    _, _, var_n = gp.posterior(state.theta, X, Yn, Xs)
    wrong = np.asarray(var_n, dtype=np.float64) * ys          # y_std, not squared
    r_right = maxrel(var[:, observed], f_var[:, observed])
    r_wrong = maxrel(wrong, f_var[:, observed])
    assert r_right <= 1e-5
    assert r_wrong > 0.1, (
        f"the bare-y_std version is only {r_wrong:.3e} relative from the archived "
        f"output; the test cannot catch that error")
    report(f"de-standardization: y_std^2 gives rel {r_right:.3e} vs the archived "
           f"variance; a bare y_std gives rel {r_wrong:.3f} "
           f"({100 * r_wrong:.0f}%), and y_std = {ys!r}")


@pytest.mark.parametrize("label,N,T", [c for c in CASES if c[1] <= 100])
def test_posterior_matches_a_LIVE_mogp_predict(fingerprints, label, N, T):
    """The shipped ``predict`` running right now, ``fast_pred_var()`` path included.

    ``mogp.predict`` is now numpy, so the gpytorch predict is the one
    :func:`live_predict` rebuilds — including the
    ``gpytorch.settings.fast_pred_var()`` context, which is part of the behaviour
    under comparison.  Same tolerances as before (1e-5 abs on the mean, 1e-5
    relative on the variance).
    """
    Xs = fingerprints[20000:20016].astype(np.float32)
    X, Y5 = training_case(fingerprints, N, T)
    model, lik, ym_o, ys_o = oracle_train(X, Y5)
    th = oracle_theta(model, lik)
    mean_o, var_o = live_predict(model, lik, ym_o, ys_o, Xs)
    y_mean, y_std, observed = gp.standardize(Y5)
    Yn = ((Y5[:, observed] - y_mean[observed]) / y_std[observed]).astype(np.float32)
    state = gp.GPState(th, X, Yn, observed)
    mean, var = gp.predict_dropin(state, None, y_mean, y_std, Xs)
    d_mean = maxabs(mean[:, observed], mean_o[:, observed])
    r_var = maxrel(var[:, observed], var_o[:, observed])
    assert d_mean <= 1e-5, f"mean vs LIVE predict: {d_mean:.3e}"
    assert r_var <= 1e-5, f"variance vs LIVE predict: rel {r_var:.3e}"
    report(f"posterior-vs-LIVE {label}: mean abs = {d_mean:.3e}, var rel = "
           f"{r_var:.3e}")


# ===========================================================================
# TIER 4 — property tests (no oracle)
# ===========================================================================

def test_marginal_variance_is_floored_at_tau(fingerprints):
    """``var_y >= tau_t`` everywhere, training points included."""
    X, Y5 = training_case(fingerprints, 100, 2)
    y_mean, y_std, observed = gp.standardize(Y5)
    Yn = ((Y5[:, observed] - y_mean[observed]) / y_std[observed]).astype(np.float32)
    theta, _ = gp.fit(X, Yn)
    _, _, _, _, tau = gp.natural(theta, 2)
    for Xq in (X, fingerprints[20000:20064].astype(np.float32)):
        _, var_f, var_y = gp.posterior(theta, X, Yn, Xq)
        assert np.all(var_y >= tau[None, :] - 1e-12)
        ratio = float((var_y / tau[None, :]).min())
    _, var_f_tr, var_y_tr = gp.posterior(theta, X, Yn, X)
    ratio_tr = float((var_y_tr / tau[None, :]).min())
    assert ratio_tr >= 1.0
    report(f"variance floor (mid_n100_t2, training set): min(var_y / tau) = "
           f"{ratio_tr:.4f}, tau = {np.asarray(tau)!r}")


def test_latent_variance_is_bounded_by_the_prior(fingerprints):
    """``0 <= var_f <= s_t * k_**`` — and the 0 clamp never actually fires."""
    Xs = fingerprints[20000:20064].astype(np.float32)
    worst_min = np.inf
    for label, N, T in CASES:
        fx = fixture("posterior.json")["cases"][label]
        X, Y5 = training_case(fingerprints, N, T)
        y_mean, y_std, observed = gp.standardize(Y5)
        Yn = ((Y5[:, observed] - y_mean[observed]) / y_std[observed]
              ).astype(np.float32)
        theta = farr(fx["theta_raw"])
        s, _, _, _, _ = gp.natural(theta, T)
        k_self = gp._self_similarity(Xs, dtype=np.float64)
        _, var_f, _ = gp.posterior(theta, X, Yn, Xs, clamp_var_f=False)
        assert np.all(var_f >= 0.0), f"{label}: var_f went negative"
        assert np.all(var_f <= s[None, :] * k_self[:, None] + 1e-12)
        worst_min = min(worst_min, float(var_f.min()))
        clamped = gp.posterior(theta, X, Yn, Xs, clamp_var_f=True)[1]
        assert np.array_equal(var_f, clamped), (
            f"{label}: the var_f >= 0 clamp CHANGED a value — it is supposed to "
            f"be a no-op on every fixture case")
    report(f"var_f bounds: 0 <= var_f <= s*k_** at every fixture case; the "
           f"smallest var_f anywhere is {worst_min:.6f}, so the >= 0 clamp is a "
           f"measured no-op (bitwise identical with and without it)")


def test_latent_variance_is_monotone_non_increasing_in_N(fingerprints):
    """More data cannot raise the latent posterior variance at a fixed query."""
    Xs = fingerprints[20000:20016].astype(np.float32)
    theta = THETA_SETTINGS["asymmetric"](2)
    X_all = fingerprints[:30].astype(np.float32)
    Y5 = synthetic_docking_targets(X_all)
    y_mean, y_std, observed = gp.standardize(Y5)
    Yn_all = ((Y5[:, observed] - y_mean[observed]) / y_std[observed]
              ).astype(np.float32)
    prev = None
    worst_increase = -np.inf
    for N in range(10, 31, 5):
        _, var_f, _ = gp.posterior(theta, X_all[:N], Yn_all[:N], Xs)
        if prev is not None:
            worst_increase = max(worst_increase, float((var_f - prev).max()))
        prev = var_f
    assert worst_increase <= 1e-8, (
        f"var_f increased by {worst_increase:.3e} when rows were added")
    report(f"var_f monotonicity: adding rows 10 -> 30 in steps of 5, the largest "
           f"INCREASE anywhere is {worst_increase:.3e} (i.e. always a decrease)")


def test_posterior_does_not_interpolate_the_training_targets(fingerprints):
    """Documenting a plausible-looking assertion that a CORRECT fit fails.

    SPEC §14 tier 4: the training-set posterior mean does not recover ``y``, and
    the amount by which it misses is a property of the fit.  Asserted here as a
    lower bound so nobody later "fixes" the model to make an interpolation test
    pass.
    """
    out = {}
    for label, N in (("small_n30_t2", 30), ("mid_n100_t2", 100)):
        X, Y5 = training_case(fingerprints, N, 2)
        y_mean, y_std, observed = gp.standardize(Y5)
        Yn = ((Y5[:, observed] - y_mean[observed]) / y_std[observed]
              ).astype(np.float32)
        theta, _ = gp.fit(X, Yn)
        mean, _, _ = gp.posterior(theta, X, Yn, X)
        out[label] = float(np.max(np.abs(mean - Yn)))
    assert out["mid_n100_t2"] > 0.1
    assert out["small_n30_t2"] > 1.0
    report("no-interpolation (normalized units): max|mean - y| = "
           + ", ".join(f"{k}={v:.3f}" for k, v in out.items())
           + " — expected, not a bug")


def test_a_duplicate_training_row_does_change_the_posterior(fingerprints):
    """Another intuitive test that would fail for a correct implementation."""
    Xs = fingerprints[20000:20016].astype(np.float32)
    theta = THETA_SETTINGS["asymmetric"](2)
    X, Y5 = training_case(fingerprints, 100, 2)
    y_mean, y_std, observed = gp.standardize(Y5)
    Yn = ((Y5[:, observed] - y_mean[observed]) / y_std[observed]).astype(np.float32)
    m0, vf0, _ = gp.posterior(theta, X, Yn, Xs)
    Xd = np.vstack([X, X[:1]])
    Yd = np.vstack([Yn, Yn[:1]])
    m1, vf1, _ = gp.posterior(theta, Xd, Yd, Xs)
    dm = float(np.max(np.abs(m1 - m0)))
    dv = float(np.max(np.abs(vf1 - vf0)))
    assert dm > 1e-4 and dv > 1e-6, (
        f"appending a duplicate row changed nothing (dmean {dm:.3e}, dvar "
        f"{dv:.3e}); a second noisy observation of the same input carries real "
        f"information and must move the posterior")
    report(f"duplicate row (N=100, hyperparameters fixed): max|dmean| = {dm:.3e}, "
           f"max|dvar_f| = {dv:.3e} at 16 OTHER query points")


def test_state_round_trips_and_records_the_threshold(fingerprints):
    """``GPState.to_dict`` must carry everything a run manifest needs."""
    X, Y5 = training_case(fingerprints, 30, 2)
    state, _, y_mean, y_std = gp.train_mogp(X, Y5, n_iterations=20)
    d = state.to_dict()
    json.dumps(d)
    assert d["n_train"] == 30 and d["n_tasks"] == 2
    assert d["oracle_was_exact"] is True
    assert len(d["theta_raw"]) == 7
    assert np.allclose(d["natural"]["tau_effective"],
                       np.asarray(state.natural()[4], dtype=np.float64))
    rebuilt = gp.GPState(d["theta_raw"], state.X_train, state.Y_norm,
                         state.observed)
    m1, _ = gp.predict(state, y_mean, y_std, X[:4])
    m2, _ = gp.predict(rebuilt, y_mean, y_std, X[:4])
    assert np.array_equal(m1, m2, equal_nan=True), (
        "state does not round-trip bitwise")


# ===========================================================================
# TIER 5 — the stochastic threshold
# ===========================================================================

def test_oracle_is_exact_boundary_is_N_times_T():
    """``N*T <= 800``, and it is the JOINT size that is compared, not ``N``.

    A pure function of ``gp.MAX_CHOLESKY_SIZE``, and deliberately so: it answers
    "is this size inside the envelope the fixtures were captured under", not
    "will the live library be exact right now".  The live question depends on
    process-global settings that ``import botorch`` rewrites — that is
    :func:`live_mll_route_is_exact`'s job, and
    :func:`test_live_oracle_is_deterministic_below_and_random_above_threshold`
    measures both regimes.
    """
    assert gp.MAX_CHOLESKY_SIZE == MAX_CHOLESKY_SIZE == 800
    assert gp.oracle_is_exact(400, 2) is True       # N*T = 800, inside
    assert gp.oracle_is_exact(401, 2) is False      # N*T = 802, outside
    assert gp.oracle_is_exact(420, 1) is True       # N*T = 420, inside at T=1
    assert gp.oracle_is_exact(800, 1) is True       # N*T = 800, inside at T=1
    assert gp.oracle_is_exact(420, 2) is False      # same N, T=2 -> outside


def test_every_validated_case_is_below_the_stochastic_threshold():
    """Enumerate the cases and state each one's ``N*T`` explicitly."""
    lines = []
    for label, N, T in TRAIN_CASES:
        NT = N * T
        assert NT <= MAX_CHOLESKY_SIZE, (
            f"{label} is at N*T={NT} > {MAX_CHOLESKY_SIZE}: above the boundary "
            f"that applied when capture_gp.py ran, the library was randomized "
            f"and the fixture is not a valid target")
        lines.append(f"{label} N*T={NT}")
    for name in ("mll_gradients.json", "trained_hyperparameters.json"):
        for label, case in fixture(name)["cases"].items():
            assert case["below_or_at_max_cholesky_size"] is True
            assert case["N_times_T"] <= MAX_CHOLESKY_SIZE
    for label, case in fixture("posterior.json")["cases"].items():
        assert case["N_times_T"] <= MAX_CHOLESKY_SIZE
    report("threshold: capture_gp.py ran under vanilla gpytorch, where the "
           "library is exact at N*T <= 800 and stochastic above it; every "
           "validated case is below/at it -> " + ", ".join(lines))


def test_botorch_import_rewrites_the_gpytorch_globals_that_pick_the_route():
    """The import-order hazard itself, pinned in a subprocess.

    ``import botorch`` mutates ``linear_operator``'s module-level defaults and
    there is no un-import, so this cannot be measured in-process once anything
    has pulled botorch in.  Two snapshots in a fresh interpreter instead.

    This test is the reason the threshold tests below read the live settings: if
    botorch ever stops doing this, the corrected claim in ``mogp_core.gp``'s
    docstring goes stale and this goes red, which is the point.
    """
    import subprocess
    src = (
        "import json\n"
        "import torch\n"
        "torch.set_num_threads(1)\n"
        "import gpytorch.settings as gs\n"
        "import linear_operator.settings as ls\n"
        "def snap():\n"
        "    return dict(max_cholesky_size=int(ls.max_cholesky_size.value()),\n"
        "                fast_log_prob=bool(ls.fast_computations.log_prob.on()),\n"
        "                fast_covar_root=bool("
        "ls.fast_computations.covar_root_decomposition.on()),\n"
        "                fast_solves=bool(ls.fast_computations.solves.on()),\n"
        "                cholesky_max_tries=int(ls.cholesky_max_tries.value()),\n"
        "                max_eager_kernel_size=int("
        "gs.max_eager_kernel_size.value()))\n"
        "before = snap()\n"
        "import botorch\n"
        "print(json.dumps({'before': before, 'after': snap(),\n"
        "                  'botorch': botorch.__version__}))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO + os.pathsep + env.get("PYTHONPATH", "")
    out = subprocess.run([sys.executable, "-c", src], capture_output=True,
                         text=True, env=env, cwd=REPO)
    assert out.returncode == 0, out.stderr[-2000:]
    got = json.loads(out.stdout.strip().splitlines()[-1])
    before, after = got["before"], got["after"]

    # The load-bearing direction: botorch pushes the library toward the EXACT
    # route -- fast log_prob off, and a larger Cholesky ceiling.
    assert before["fast_log_prob"] is True, (
        f"vanilla gpytorch no longer has fast log_prob on: {before}")
    assert after["fast_log_prob"] is False, (
        "importing botorch no longer turns fast log_prob off; "
        f"mogp_core.gp's docstring needs re-measuring: {after}")
    assert after["fast_covar_root"] is False, after
    assert after["max_cholesky_size"] >= before["max_cholesky_size"], (
        f"{before} -> {after}")
    assert before != after, "importing botorch changed nothing -- re-measure"
    # And the capture-time constant mogp_core.gp records is the vanilla default.
    assert before["max_cholesky_size"] == MAX_CHOLESKY_SIZE == 800, (
        f"gpytorch's default max_cholesky_size is now "
        f"{before['max_cholesky_size']}, not {MAX_CHOLESKY_SIZE}: every fixture "
        f"in spec/gp/fixtures/ was captured under the old boundary")
    report(f"botorch import (subprocess, botorch {got['botorch']}): "
           + "; ".join(f"{k}: {before[k]} -> {after[k]}" for k in sorted(before)))


def test_live_oracle_is_deterministic_below_and_random_above_threshold(
        fingerprints):
    """The boundary, demonstrated against the live library AND its live settings.

    CORRECTED 2026-09-27.  This test used to hard-code "exact at ``N*T <= 800``,
    stochastic above", and it failed whenever it ran in the same process as
    ``test_core_hv.py`` — which imports the repo's ``evaluation.py``, which
    imports botorch, which permanently rewrites ``linear_operator``'s defaults to
    ``max_cholesky_size=4096`` with the ``fast_computations`` flags off.  The
    failure was real and the assumption was the bug: there is no fixed threshold.

    ``linear_operator`` 0.6.1 takes the exact Cholesky branch iff
    ``fast_computations.log_prob.off() or N*T <= max_cholesky_size`` (see
    :func:`live_mll_route_is_exact` for the source line), so this test:

    A. reads the LIVE settings and asserts the determinism that FOLLOWS from
       them, at sizes on both sides of the live ceiling;
    B. forces the capture-time settings and demonstrates the ``N*T = 800``
       boundary itself, which is what the fixtures were captured under and is
       therefore what :func:`mogp_core.gp.oracle_is_exact` describes;
    C. forces the botorch-style settings and shows the *size* stops mattering —
       the flags dominate, so the repo's production processes (all of which
       import botorch) never take the stochastic path at any size.

    B and C do not depend on import order, so the boundary claim is still tested
    with teeth in either regime.
    """
    s_live = live_settings()

    # -- A. the LIVE process, whatever an earlier import did to it -------------
    live = []
    for N, T in ((400, 2), (410, 2), (420, 1), (420, 2)):
        expect_exact = live_mll_route_is_exact(N, T, s_live)
        spread, bias = live_mll_spread(fingerprints, N, T)
        live.append((N, T, N * T, spread, bias, expect_exact))
        if expect_exact:
            assert spread == 0.0, (
                f"LIVE settings {s_live} predict the exact Cholesky route at "
                f"N={N}, T={T} (N*T={N * T}), but 8 identical mll calls spread "
                f"by {spread:.3e}")
            assert abs(bias) <= 1e-6, (
                f"N={N}, T={T}: exact route but bias vs mogp_core {bias:.3e}")
        else:
            assert spread > 0.0, (
                f"LIVE settings {s_live} predict the STOCHASTIC route at N={N}, "
                f"T={T} (N*T={N * T}) — fast log_prob is on and "
                f"N*T > max_cholesky_size={s_live['max_cholesky_size']} — but "
                f"the spread is exactly 0. linear_operator's routing rule has "
                f"changed and SPEC §11 needs re-measuring")
    report("threshold, LIVE settings "
           f"(max_cholesky_size={s_live['max_cholesky_size']}, "
           f"fast_log_prob={s_live['fast_log_prob']}, "
           f"botorch_imported={'botorch' in sys.modules}), "
           "8 identical mll calls each: "
           + "; ".join(f"N={N} T={T} N*T={NT} spread={sp:.3e} "
                       f"bias_vs_mine={b:+.3e} predicted="
                       f"{'exact' if e else 'stochastic'}"
                       for N, T, NT, sp, b, e in live))

    # -- B. FORCED capture-time settings: the N*T = 800 boundary itself --------
    forced = []
    with capture_time_settings():
        s_cap = live_settings()
        assert s_cap["max_cholesky_size"] == MAX_CHOLESKY_SIZE
        assert s_cap["fast_log_prob"] is True and s_cap["fast_covar_root"] is True
        for N, T, want in ((400, 2, True), (410, 2, False), (420, 1, True),
                           (420, 2, False)):
            assert live_mll_route_is_exact(N, T, s_cap) is want
            spread, bias = live_mll_spread(fingerprints, N, T)
            forced.append((N, T, N * T, spread, bias, want))
            if want:
                assert spread == 0.0, (
                    f"capture-time settings, N={N}, T={T} (N*T={N * T} <= "
                    f"{MAX_CHOLESKY_SIZE}) must be exact Cholesky but 8 "
                    f"identical mll calls spread by {spread:.3e}")
                assert abs(bias) <= 1e-6, f"N={N}, T={T}: bias {bias:.3e}"
            else:
                assert spread > 0.0, (
                    f"capture-time settings (max_cholesky_size=800, fast "
                    f"computations ON), N={N}, T={T} (N*T={N * T} > 800) must "
                    f"be stochastic Lanczos quadrature with unseeded probes, "
                    f"but the spread over 8 identical mll calls is exactly 0 — "
                    f"the boundary the fixtures were captured under has moved "
                    f"and SPEC §11 needs re-measuring")
    report("threshold, FORCED capture-time settings "
           f"(max_cholesky_size={MAX_CHOLESKY_SIZE}, fast_computations ON): "
           + "; ".join(f"N={N} T={T} N*T={NT} spread={sp:.3e} "
                       f"bias_vs_mine={b:+.3e} expected="
                       f"{'exact' if e else 'stochastic'}"
                       for N, T, NT, sp, b, e in forced))

    # -- C. FORCED botorch-style settings: the flags dominate the size ---------
    ls = oracle()["lo_settings"]
    flags_off = []
    with ls.max_cholesky_size(MAX_CHOLESKY_SIZE), ls.fast_computations(
            covar_root_decomposition=False, log_prob=False, solves=False):
        s_off = live_settings()
        assert s_off["max_cholesky_size"] == MAX_CHOLESKY_SIZE
        assert s_off["fast_log_prob"] is False
        for N, T in ((410, 2), (420, 2), (2049, 2)):
            # The predicate says exact even though N*T > max_cholesky_size.
            assert live_mll_route_is_exact(N, T, s_off) is True
            assert N * T > MAX_CHOLESKY_SIZE
            spread, bias = live_mll_spread(fingerprints, N, T)
            flags_off.append((N, T, N * T, spread, bias))
            assert spread == 0.0, (
                f"fast log_prob is OFF, so N={N}, T={T} (N*T={N * T}) must take "
                f"the exact Cholesky route regardless of "
                f"max_cholesky_size={MAX_CHOLESKY_SIZE}, but the spread over 8 "
                f"identical mll calls is {spread:.3e}")
            assert abs(bias) <= 1e-6, (
                f"N={N}, T={T}: fast paths off, so the library is exact, yet it "
                f"disagrees with mogp_core by {bias:.3e}")
    report("threshold, FORCED botorch-style settings (max_cholesky_size="
           f"{MAX_CHOLESKY_SIZE} but fast_computations OFF -> size is "
           "irrelevant): "
           + "; ".join(f"N={N} T={T} N*T={NT} spread={sp:.3e} "
                       f"bias_vs_mine={b:+.3e}"
                       for N, T, NT, sp, b in flags_off))

    # mogp_core's constant is a record of the capture conditions, so it keeps
    # answering the capture-time question and must NOT track the live settings.
    assert gp.oracle_is_exact(400, 2) and not gp.oracle_is_exact(410, 2)
