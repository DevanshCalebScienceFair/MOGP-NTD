"""Differential tests for ``mogp_core.icm`` — the pure-numpy coregionalized (ICM) GP.

Three kinds of oracle, all used:

* the archived fixtures in ``spec/icm/fixtures/`` (captured 2026-09-27 from the
  live gpytorch path by ``spec/icm/capture_icm.py``),
* **live** torch / gpytorch, imported *only here* — ``mogp_core`` itself may
  import nothing but numpy and the standard library; this file is a test
  instrument, and importing the library it mirrors is how fidelity gets proven,
* **internal cross-validation** that needs no oracle at all: a dense Cholesky of
  the same ``Sigma`` the Kronecker path factorizes, central finite differences of
  this module's own loss, and the algebraic invariances of ``B``.

Every tolerance is roughly 10x a number actually measured by a run of this file,
and the measured value is put in each assertion message, so a failure reports the
real disagreement rather than just "too big".  Run with ``-s`` to see the
measured table.

WHICH TESTS NEED THE LIBRARY (measured 2026-09-28, with gpytorch, torch, botorch,
linear_operator, sklearn, joblib and scipy all made unimportable by a meta-path
blocker: 47 passed, 5 skipped, 0 failed).  Three tests that used to skip now
assert their claim against the ARCHIVED library array instead —
``test_ordering_against_live_gpytorch_on_asymmetric_kernels`` (both shapes) and
``test_task_covariance_matrix_method_matches_the_shipped_one`` — because
``02_ordering/assembled_dense.npy`` and
``01_parameterization.json → task_covariance.B_gpytorch`` ARE gpytorch's own
answers, captured 2026-09-27.  The five that still skip cannot be repointed, and
each names the unconditional test that keeps its evidence:

* ``test_analytic_gradient_matches_live_autograd_per_group`` (4 thetas) — only
  ``fixture_theta`` has a captured autograd gradient, and that comparison is
  already made unconditionally, per group and at the same ``1e-13``, by
  :func:`test_analytic_gradient_matches_the_archived_oracles`; all four thetas are
  cross-validated against central finite differences with no oracle at all by
  :func:`test_analytic_gradient_matches_central_finite_differences`.
* ``test_posterior_against_the_shipped_float32_predict`` — the shipped **float32**
  ``predict`` path is in no fixture (``05_posterior/`` stores the float64 latent
  reference, which :func:`test_posterior_mean_variance_and_full_covariance` uses
  unconditionally), so there is nothing to repoint it at.

THE TWO THINGS THIS SUITE EXISTS FOR
------------------------------------
**1. The transpose.**  ``kron(K_XX, B)`` and ``kron(B, K_XX)`` are
permutation-similar, so getting the ordering backwards leaves the loss, the
log-determinant, the hypervolume and every other aggregate metric *bit-identically
unchanged* while silently swapping PfDHFR's per-task posterior with hDHFR's.  No
end-to-end check can catch it.  It is therefore caught here directly, against
deliberately **asymmetric** data and task kernels on both sides — with live
gpytorch, at two different ``(N, K)`` shapes — where the two candidate orderings
differ by ``3.0`` and ``1.95``.

**2. The identifiability trap.**  ``B = W W^T + diag(softplus(v))`` is invariant
under ``W -> W Q`` for any orthogonal ``Q``, and at ``K=2, R=1`` there is in
addition a *continuous* family of ``(W, v)`` giving one identical ``B``.  ``W``
flips sign across seeds at every step count.  So this suite tests ``B`` and the
task correlation ``rho`` and **never** ``W``, ``raw_var``, ``raw_task_noises`` or
``raw_noise`` as fitted values.  ``test_no_test_here_pins_an_unidentified_fitted_parameter``
enforces that by parsing this file's own AST.

Comparing the *gradient* ``dloss/dW`` at a FIXED ``theta`` against autograd is a
different and legitimate thing: the derivative of the loss with respect to a
coordinate is well defined even where the coordinate itself is not identified.
The distinction is exactly "fitted value" vs "derivative at a given point", and
the AST guard above is written to allow the second and forbid the first.

WHAT IS DELIBERATELY NOT TESTED
-------------------------------
``acquisition.py:323``'s forced-diagonal posterior (SPEC §8) is a known
scientific gap that this port must PRESERVE.
:func:`test_the_discarded_cross_task_structure_is_still_there_to_discard`
measures what it throws away and asserts the structure exists; it does not
assert that anything uses it.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import sys

# ---------------------------------------------------------------------------
# Thread pinning MUST precede any torch import: this env segfaults on import
# with more than one OpenMP thread (duplicate libomp).  Same guard as
# test_core_gp.py.
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

from mogp_core import icm                                     # noqa: E402
from mogp_core import transforms as mtr                       # noqa: E402

FIXTURES = os.path.join(REPO, "spec", "icm", "fixtures")
CORE_DIR = os.path.join(REPO, "mogp_core")

K_TASKS = 2          # the two modelled docking objectives
RANK = 1             # the project default and the only rank the loop passes
NOISE_FLOOR = 1e-4   # GreaterThan(1e-4) on BOTH noise parameters

#: ``softplus`` here is gpytorch's (``threshold=20`` linear branch), while the
#: spec's ``_ref_softplus`` is the branch-free (P1) form.  The two agree to float
#: rounding below the branch and to ``1.4e-11`` at it; every fixture raw value is
#: well below 20, so the choice is invisible in these comparisons and this
#: module's form is the one torch actually computes.  Measured agreement with the
#: fixtures is 0.0 (see :func:`test_task_covariance_and_noise_links`).
SOFTPLUS_NOTE = "transforms.softplus = torch's (threshold=20); spec (P1) is branch-free"


# ===========================================================================
# Fixture loading and small helpers
# ===========================================================================

_JSON_CACHE = {}


def fixture(name):
    if name not in _JSON_CACHE:
        with open(os.path.join(FIXTURES, name)) as fh:
            _JSON_CACHE[name] = json.load(fh)
    return _JSON_CACHE[name]


def arr(rel):
    """A fixture ``.npy``, float64 unless it is the int8 fingerprints."""
    return np.load(os.path.join(FIXTURES, rel))


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
# Session fixtures — the real N=40 / M=40 docking case
# ===========================================================================

@pytest.fixture(scope="session")
def real_case():
    """The spec's real case: 40 train + 40 test library molecules, real Vina LE.

    ``(X_train, X_test, Y_norm, K_xx, lam, U, theta)`` where ``theta`` is the
    fixture's own ``06_gradients/theta_raw.npy`` — the raw parameter vector every
    stored gradient, logdet, solve and posterior array was produced at.
    """
    Xtr = arr("03_real_data/fingerprints_train.npy")
    Xte = arr("03_real_data/fingerprints_test.npy")
    Yn = arr("03_real_data/y_normalized_train.npy")
    theta = arr("06_gradients/theta_raw.npy")
    K_xx = icm.gram(Xtr, Xtr)
    lam, U = icm.eig_data(K_xx)
    assert Yn.shape == (40, K_TASKS)
    assert theta.shape == (icm.n_params(K_TASKS, RANK),)
    return Xtr, Xte, Yn, K_xx, lam, U, theta


@pytest.fixture(scope="session")
def library_fingerprints():
    """``data/library/fingerprints.npy``, memory-mapped."""
    path = os.path.join(REPO, "data", "library", "fingerprints.npy")
    if not os.path.exists(path):
        pytest.skip("data/library/fingerprints.npy absent")
    return np.load(path, mmap_mode="r")


def _oracle():
    """torch / gpytorch, imported lazily and only by the oracle tests."""
    try:
        import gpytorch
        import torch
    except ImportError as exc:                                # pragma: no cover
        pytest.skip(f"live oracle unavailable: {exc}")
    return torch, gpytorch


def _oracle_or_none():
    """``(torch, gpytorch)`` or ``(None, None)`` — the non-skipping variant.

    Added 2026-09-28.  A test that has an ARCHIVED library witness for its claim
    uses this and asserts that witness unconditionally, so a machine without
    gpytorch loses the live re-measurement rather than the assertion.  ``_oracle``
    is kept for the two claims the archive does not witness (the shipped float32
    ``predict`` path, and autograd at the three thetas no capture covers), which
    genuinely cannot be made without the library.
    """
    try:
        import gpytorch
        import torch
    except ImportError:                                       # pragma: no cover
        return None, None
    return torch, gpytorch


def live_assembled_multitask(K_data, B_task):
    """gpytorch's OWN assembled dense ``MultitaskKernel`` covariance.

    A fixed data kernel returning ``K_data`` verbatim, and an ``IndexKernel``
    forced to realize ``B_task`` (Cholesky factor into ``covar_factor``, a
    negligible ``1e-12`` into ``raw_var`` so ``softplus`` reproduces it).  What
    comes back is the library's answer to "which Kronecker order is it", with no
    interpretation by this file.  ``None`` when gpytorch is absent, so the callers
    can fall back to the archived matrix instead of skipping.
    """
    torch, gpytorch = _oracle_or_none()
    if torch is None:                                         # pragma: no cover
        return None
    K_data = np.asarray(K_data, dtype=np.float64)
    B_task = np.asarray(B_task, dtype=np.float64)
    N = K_data.shape[0]
    K = B_task.shape[0]

    class Fixed(gpytorch.kernels.Kernel):
        is_stationary = False

        def forward(self, x1, x2, diag=False, **kw):
            M = torch.as_tensor(K_data, dtype=torch.float64)
            return M.diagonal() if diag else M

    mk = gpytorch.kernels.MultitaskKernel(Fixed(), num_tasks=K, rank=K).double()
    v = np.full(K, 1e-12)
    L = np.linalg.cholesky(B_task - np.diag(v))
    with torch.no_grad():
        mk.task_covar_module.covar_factor.copy_(torch.tensor(L))
        mk.task_covar_module.raw_var.copy_(torch.tensor(np.log(np.expm1(v))))
    return mk(torch.zeros(N, 1, dtype=torch.float64)).to_dense().detach().numpy()


def live_autograd_groups(theta, K_xx, Y_norm, rank=RANK):
    """``(nll, per-group gradient)`` from torch autograd on a DENSE Cholesky NLL.

    Deliberately *not* the Kronecker path: ``Sigma`` is built as
    ``kron(K_xx, B) + kron(I, diag(tau)) + sigma2 I`` and factorized densely, so
    the oracle shares no algebraic step with :func:`icm.nmll_and_grad`.  A shared
    mistake in (L1)/(S5)/(G1)-(G6) therefore cannot cancel.
    """
    torch, _ = _oracle()
    Y_norm = np.asarray(Y_norm, dtype=np.float64)
    N, K = Y_norm.shape
    W, v, tn, gn, c = icm.unpack(np.asarray(theta, dtype=np.float64), K, rank)

    def leaf(z):
        return torch.tensor(np.array(np.atleast_1d(z), dtype=np.float64),
                            requires_grad=True)

    tW = torch.tensor(np.array(W, dtype=np.float64), requires_grad=True)
    tv, ttn, tgn, tc = leaf(v), leaf(tn), leaf(np.atleast_1d(gn)), leaf(c)
    spf = torch.nn.functional.softplus
    tB = tW @ tW.T + torch.diag(spf(tv))
    ttau = spf(ttn) + NOISE_FLOOR
    ts2 = spf(tgn)[0] + NOISE_FLOOR
    tK = torch.tensor(np.asarray(K_xx, dtype=np.float64))
    tSig = (torch.kron(tK, tB)
            + torch.kron(torch.eye(N, dtype=torch.float64), torch.diag(ttau))
            + ts2 * torch.eye(N * K, dtype=torch.float64))
    tr = (torch.tensor(Y_norm) - tc).reshape(-1)
    tL = torch.linalg.cholesky(tSig)
    ta = torch.cholesky_solve(tr.unsqueeze(-1), tL).squeeze(-1)
    tlogp = (-0.5 * (tr @ ta) - torch.log(torch.diag(tL)).sum()
             - 0.5 * N * K * math.log(2.0 * math.pi))
    tnll = -tlogp / (N * K)
    tnll.backward()
    return float(tnll.item()), {
        "W": tW.grad.numpy(),
        "raw_var": tv.grad.numpy(),
        "raw_task_noises": ttn.grad.numpy(),
        "raw_noise": tgn.grad.numpy(),
        "mean_constant": tc.grad.numpy(),
    }


# ===========================================================================
# The shipped gpytorch ICM model, rebuilt in the test instrument.
#
# ``mogp_coregionalized.MOGPCoregionalized`` was a ``gpytorch.models.ExactGP``
# over ``kernel.TanimotoKernel``; the numpy integration rewrote both modules, so
# importing them no longer supplies a library oracle (T13 failed with
# ``'MOGPCoregionalized' object has no attribute 'covar_module'``).  The classes
# below are VERBATIM transcriptions of git ``3c6573c`` —
# ``kernel.py``'s ``TanimotoKernel`` and ``mogp_coregionalized.py``'s
# ``MOGPCoregionalized`` / ``predict_coregionalized``, which is the code
# ``spec/icm/capture_icm.py`` ran.  Building them here rather than importing them
# is what the dependency rule demands, and the oracle is the same one as before.
# ===========================================================================

_LIVE_ICM = {}


def live_coregionalized_classes():
    """``(MOGPCoregionalized, predict_coregionalized)``, the shipped gpytorch pair."""
    if _LIVE_ICM:
        return _LIVE_ICM["model_cls"], _LIVE_ICM["predict"]
    torch, gpytorch = _oracle()

    class _LiveTanimotoKernel(gpytorch.kernels.Kernel):
        """``3c6573c:kernel.py``'s ``TanimotoKernel``, character for character."""

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

    class _LiveMOGPCoregionalized(gpytorch.models.ExactGP):
        """``3c6573c:mogp_coregionalized.py``'s class, character for character."""

        def __init__(self, train_x, train_y, likelihood, rank=1):
            super().__init__(train_x, train_y, likelihood)
            num_tasks = train_y.shape[1]
            self.mean_module = gpytorch.means.MultitaskMean(
                gpytorch.means.ConstantMean(),
                num_tasks=num_tasks,
            )
            self.covar_module = gpytorch.kernels.MultitaskKernel(
                _LiveTanimotoKernel(),
                num_tasks=num_tasks,
                rank=rank,
            )

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultitaskMultivariateNormal(mean_x,
                                                                     covar_x)

        def task_covariance_matrix(self):
            index_kernel = self.covar_module.task_covar_module
            return index_kernel._eval_covar_matrix().detach().cpu().numpy()

    def _live_predict_coregionalized(model, likelihood, y_mean, y_std, X_new):
        """``3c6573c``'s ``predict_coregionalized``, character for character.

        Including the ``fast_pred_var()`` context and the de-standardization with
        ``y_std ** 2`` on the variance — both are part of the behaviour compared.
        """
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

    _LIVE_ICM["model_cls"] = _LiveMOGPCoregionalized
    _LIVE_ICM["predict"] = _live_predict_coregionalized
    return _LIVE_ICM["model_cls"], _LIVE_ICM["predict"]


def make_theta(W, raw_var, raw_tn, raw_gn, c):
    return icm.theta_from_groups(W, raw_var, raw_tn, raw_gn, c)


#: Parameter settings the gradient checks run at.  The last one sits ABOVE
#: softplus's ``threshold=20`` linear branch on every constrained group, which is
#: where a naive ``log(1+exp(x))`` overflows and a naive derivative loses the
#: exact ``1.0``.
GRAD_CASES = {
    "fixture_theta": (None, RANK),
    "rank2_random": (make_theta(np.array([[0.62, -1.31], [0.90, 0.41]]),
                                np.array([-0.37, 0.55]),
                                np.array([0.21, -1.02]),
                                np.array([0.40]),
                                np.array([0.13, -0.28])), 2),
    "extreme_raws": (make_theta(np.array([[1.70], [-0.90]]),
                                np.array([-9.0, 13.0]),
                                np.array([-13.0, 2.5]),
                                np.array([-11.0]),
                                np.array([0.4, -0.6])), RANK),
    "above_softplus_linear_branch": (make_theta(np.array([[0.80], [-1.10]]),
                                                np.array([25.0, -30.0]),
                                                np.array([26.0, -25.0]),
                                                np.array([24.0]),
                                                np.array([0.2, -0.3])), RANK),
}


# ===========================================================================
# TIER 0 — the dependency rule
# ===========================================================================

FORBIDDEN = ("torch", "gpytorch", "botorch", "scipy", "sklearn",
             "linear_operator", "joblib")


def test_icm_module_imports_no_forbidden_dependency():
    """``mogp_core/icm.py`` may import nothing but numpy, stdlib and siblings.

    Checked by parsing the AST, not by importing, so a lazily- or
    conditionally-imported offender is still caught.
    """
    path = os.path.join(CORE_DIR, "icm.py")
    with open(path) as fh:
        tree = ast.parse(fh.read(), filename=path)
    offenders = []
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for nm in names:
            if nm.split(".")[0] in FORBIDDEN:
                offenders.append(f"icm.py:{node.lineno} imports {nm}")
    assert not offenders, "forbidden imports in mogp_core/icm.py: " + "; ".join(offenders)
    for name in FORBIDDEN:
        assert not hasattr(icm, name), f"mogp_core.icm exposes {name}"
    report("dependency scan: mogp_core/icm.py imports 0 forbidden modules")


def test_icm_imports_with_every_library_unimportable():
    """``import mogp_core.icm`` must succeed in a process where torch is absent.

    Run in a subprocess with a meta-path finder that refuses every forbidden
    root, which is stronger than checking the AST: it also catches an import
    hidden inside a sibling module that ``icm`` pulls in.
    """
    import subprocess
    prog = (
        "import sys\n"
        "BAD = " + repr(list(FORBIDDEN)) + "\n"
        "class Block:\n"
        "    def find_module(self, name, path=None):\n"
        "        return self.find_spec(name, path)\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name.split('.')[0] in BAD:\n"
        "            raise ImportError('blocked: ' + name)\n"
        "        return None\n"
        "sys.meta_path.insert(0, Block())\n"
        "sys.path.insert(0, " + repr(REPO) + ")\n"
        "from mogp_core import icm\n"
        "import numpy as np\n"
        "B = icm.task_covariance(np.array([[0.7],[-1.3]]), np.array([-0.5,2.0]))\n"
        "assert abs(B[0,0] - 0.9640769841801067) < 1e-12, B[0,0]\n"
        "print('OK', icm.n_params(2,1))\n"
    )
    out = subprocess.run([sys.executable, "-c", prog], capture_output=True,
                         text=True)
    assert out.returncode == 0, (
        "mogp_core.icm failed to import with the libraries blocked:\n"
        + out.stdout + out.stderr)
    assert out.stdout.strip() == "OK 9", out.stdout
    report("import purity: mogp_core.icm imports and computes B with "
           "torch/gpytorch/botorch/scipy/sklearn/joblib unimportable")


# ===========================================================================
# T1 / T2 — THE TRANSPOSE TRAP and the flat index arithmetic (SPEC §2)
# ===========================================================================

#: ``(label, K_data, B_task)``.  Both matrices are symmetric PSD but have
#: deliberately DIFFERENT off-diagonal values, which is what makes the two
#: candidate Kronecker orders numerically distinguishable at all.  The second
#: case has ``N != K`` and a ``3x3`` task matrix with a negative off-diagonal, so
#: no accidental squareness or sign symmetry can rescue a transpose.
ORDERING_CASES = [
    ("spec_N3_K2",
     np.array([[1.00, 0.50, 0.20],
               [0.50, 1.00, 0.30],
               [0.20, 0.30, 1.00]]),
     np.array([[1.00, 0.10],
               [0.10, 4.00]])),
    ("harder_N4_K3",
     np.array([[1.00, 0.31, 0.07, 0.55],
               [0.31, 1.00, 0.44, 0.12],
               [0.07, 0.44, 1.00, 0.23],
               [0.55, 0.12, 0.23, 1.00]]),
     np.array([[2.00, 0.30, -0.70],
               [0.30, 1.50, 0.11],
               [-0.70, 0.11, 3.25]])),
]


def test_the_archived_ordering_fixture_is_data_major():
    """T1, archived half: ``assembled == kron(K_data, B_task)``, NOT the reverse.

    The gpytorch ``MultitaskKernel`` docstring says ``K_TT (x) K_XX``.  Its code
    assembles ``K_XX (x) B``.  The docstring is wrong and is not a specification.
    """
    K_data = arr("02_ordering/K_data.npy")
    B_task = arr("02_ordering/B_task.npy")
    assembled = arr("02_ordering/assembled_dense.npy")
    mine = icm.assemble_dense(K_data, B_task)
    d_dm = maxabs(assembled, mine)
    d_tm = maxabs(assembled, np.kron(B_task, K_data))
    assert d_dm <= 1e-12, (
        f"data-major order disagrees with the archived gpytorch matrix: {d_dm:.3e}")
    assert d_tm >= 1.0, (
        "the task-major (docstring) order is NOT distinguishable on this "
        f"fixture ({d_tm:.3e} < 1.0) — the test cannot fail, so it proves nothing")
    assert maxabs(mine, arr("02_ordering/kron_data_major.npy")) == 0.0
    j = fixture("02_ordering.json")
    assert j["answer"] == "data-major: K_XX (x) B, flat index = i*num_tasks + t"
    report(f"T1 archived: |assembled - kron(K_data,B)| = {d_dm:.3e}; "
           f"|assembled - kron(B,K_data)| = {d_tm:.3e}")


@pytest.mark.parametrize("label,K_data,B_task", ORDERING_CASES,
                         ids=[c[0] for c in ORDERING_CASES])
def test_ordering_against_live_gpytorch_on_asymmetric_kernels(label, K_data, B_task):
    """T1, live half: assemble against gpytorch NOW, at two ``(N, K)`` shapes.

    A transpose here is invisible in the loss, the log-determinant and the
    hypervolume (they are permutation-invariant), and wrong per task.  So it is
    checked directly, with asymmetric VALUES on both factors, and the test
    asserts that the wrong order is actually far away — otherwise a passing
    result would be vacuous.

    2026-09-28: the ordering claim no longer disappears when gpytorch does.
    ``spec_N3_K2``'s inputs ARE ``02_ordering/K_data.npy`` / ``B_task.npy``, so its
    library witness (``02_ordering/assembled_dense.npy``, gpytorch's own assembled
    covariance) is asserted unconditionally at the same ``1e-12``; ``harder_N4_K3``
    has no captured counterpart, so without gpytorch it falls back to the
    hand-computed flat-index probe, which pins the ordering against arithmetic
    done in this file rather than against the library.
    """
    mine = icm.assemble_dense(K_data, B_task)

    # -- unconditional: the archived gpytorch matrix where one exists ----------
    if np.array_equal(K_data, arr("02_ordering/K_data.npy")) and \
            np.array_equal(B_task, arr("02_ordering/B_task.npy")):
        archived = arr("02_ordering/assembled_dense.npy")
        d_arch = maxabs(archived, mine)
        assert d_arch <= 1e-12, (
            f"{label}: the ARCHIVED gpytorch matrix disagrees with data-major: "
            f"{d_arch:.3e}")
        assert maxabs(archived, np.kron(B_task, K_data)) >= 1e-2
    # and the hand-computed probe, which needs no oracle at all
    assert abs(mine[0, 1] - K_data[0, 0] * B_task[0, 1]) <= 1e-12
    assert abs(mine[0, 1] - B_task[0, 0] * K_data[0, 1]) > 1e-2

    # -- live ------------------------------------------------------------------
    assembled = live_assembled_multitask(K_data, B_task)
    if assembled is None:                                     # pragma: no cover
        report(f"T1 {label}: gpytorch absent -> "
               + ("the archived gpytorch matrix is the oracle "
                  f"({maxabs(arr('02_ordering/assembled_dense.npy'), mine):.3e})"
                  if np.array_equal(K_data, arr("02_ordering/K_data.npy"))
                  else "no archived witness for this shape; the flat-index probe "
                       "is what still bites"))
        return
    d_dm = maxabs(assembled, mine)
    d_tm = maxabs(assembled, np.kron(B_task, K_data))
    assert d_dm <= 1e-12, (
        f"{label}: live gpytorch disagrees with data-major: {d_dm:.3e}")
    assert d_tm >= 1e-2, (
        f"{label}: the two orderings differ by only {d_tm:.3e}; this case "
        "cannot detect a transpose")
    N, K = K_data.shape[0], B_task.shape[0]
    probe_dm = K_data[0, 0] * B_task[0, 1]
    probe_tm = B_task[0, 0] * K_data[0, 1]
    assert abs(assembled[0, 1] - probe_dm) <= 1e-12
    assert abs(assembled[0, 1] - probe_tm) > 1e-2
    report(f"T1 live {label} (N={N},K={K}): |live - data-major| = {d_dm:.3e}; "
           f"|live - task-major| = {d_tm:.3e}; probe[0,1] {assembled[0,1]:.4f} "
           f"= K[0,0]*B[0,1] ({probe_dm:.4f}), not B[0,0]*K[0,1] ({probe_tm:.4f})")


@pytest.mark.parametrize("label,K_data,B_task", ORDERING_CASES,
                         ids=[c[0] for c in ORDERING_CASES])
def test_flat_index_arithmetic_indexes_the_right_entry(label, K_data, B_task):
    """T2: ``a = i*K + t`` and it addresses ``K_data[i,j]*B[t,s]`` (O1)/(O2).

    Not a tautology: the index functions are used to read the ASSEMBLED matrix,
    so an off-by-one or a swapped factor in :func:`icm.flat_index` /
    :func:`icm.unravel_flat` lands on the wrong entry and this fails.
    """
    N, K = K_data.shape[0], B_task.shape[0]
    Sigma = icm.assemble_dense(K_data, B_task)
    worst = 0.0
    for i in range(N):
        for t in range(K):
            a = int(icm.flat_index(i, t, K))
            assert (i, t) == tuple(int(x) for x in icm.unravel_flat(a, K))
            for j in range(N):
                for s in range(K):
                    b = int(icm.flat_index(j, s, K))
                    worst = max(worst, abs(Sigma[a, b] - K_data[i, j] * B_task[t, s]))
    assert worst <= 1e-15, f"{label}: (O2) violated by {worst:.3e}"
    # (O1) again, this time on the target vectorization: y = Y.reshape(-1).
    Y = np.arange(N * K, dtype=np.float64).reshape(N, K) * 1.5 - 3.0
    y = Y.reshape(-1)
    for i in range(N):
        for t in range(K):
            assert y[int(icm.flat_index(i, t, K))] == Y[i, t]
    report(f"T2 {label}: (O1)+(O2) exact over all {(N*K)**2} entries "
           f"(worst {worst:.3e}); Y.reshape(-1)[i*K+t] == Y[i,t] exactly")


def test_kron_matvec_reshape_identity(label="O3"):
    """(O3): ``(P (x) Q) vec(V) == vec(P V Q^T)`` — the identity that replaces
    every Kronecker matvec, checked against the explicit product."""
    rng = np.random.default_rng(20260927)
    N, K = 5, 3
    P = rng.standard_normal((N, N))
    Q = rng.standard_normal((K, K))
    V = rng.standard_normal((N, K))
    explicit = np.kron(P, Q) @ V.reshape(-1)
    d = maxabs(icm.kron_matvec(P, Q, V).reshape(-1), explicit)
    assert d <= 1e-12, f"(O3) violated by {d:.3e}"
    report(f"O3: |vec(P V Q^T) - (P(x)Q)vec(V)| = {d:.3e}")


def test_marginal_variance_reshape_rule_is_data_major():
    """(O4): ``var[i,t] = diag(Sigma_post)[i*K + t]``, as the library's own
    ``MultitaskMultivariateNormal`` reports it in the fixture."""
    j = fixture("02_ordering.json")
    want = np.asarray(j["MultitaskMultivariateNormal_variance_from_diag_100_600"],
                      dtype=np.float64)
    N, K = want.shape
    diagvals = np.arange(1.0, N * K + 1.0) * 100.0
    mine = diagvals.reshape(N, K)
    d = maxabs(mine, want)
    assert d <= 1e-12, f"(O4) reshape disagrees with the library: {d:.3e}"
    assert j["variance_reshape_rule"] == "variance[i, t] = diag(covar)[i*num_tasks + t]"
    assert j["MultitaskMultivariateNormal_interleaved"] is True
    report(f"O4: diag(Sigma).reshape(N,K) vs MultitaskMultivariateNormal.variance "
           f"= {d:.3e}")


# ===========================================================================
# T3 / T4 — parameterization (SPEC §1.5)
# ===========================================================================

def test_task_covariance_and_noise_links():
    """T3/T4: ``B``, ``tau`` and ``sigma2`` at the fixture's raw values.

    ``raw_var`` is ``Positive()`` = plain softplus with **no** offset, while both
    noises are ``GreaterThan(1e-4)`` = ``1e-4 + softplus``.  Getting that
    asymmetry backwards shifts ``B``'s diagonal by ``1e-4`` — invisible in a
    smoke test, wrong in the fit.
    """
    j = fixture("01_parameterization.json")
    tc = j["task_covariance"]
    B = icm.task_covariance(np.asarray(tc["raw_W"]), np.asarray(tc["raw_var"]))
    d_ref = maxabs(B, np.asarray(tc["B_reference"]))
    d_lib = maxabs(B, np.asarray(tc["B_gpytorch"]))
    assert d_ref <= 1e-12, f"B vs the spec reference: {d_ref:.3e}"
    assert d_lib <= 1e-12, f"B vs IndexKernel._eval_covar_matrix: {d_lib:.3e}"

    nz = j["noise"]
    tau = mtr.greater_than(np.asarray(nz["raw_task_noises"]), lower=NOISE_FLOOR)
    s2 = mtr.greater_than(np.asarray(nz["raw_noise"]), lower=NOISE_FLOOR)
    d_tau = maxabs(tau, np.asarray(nz["tau_reference"]))
    d_s2 = maxabs(s2, np.asarray(nz["sigma2_reference"]))
    d_tau_lib = maxabs(tau, np.asarray(nz["tau_gpytorch"]))
    d_s2_lib = maxabs(s2, np.asarray(nz["sigma2_gpytorch"]))
    assert d_tau <= 1e-12 and d_s2 <= 1e-12, (
        f"noise vs the spec reference: tau {d_tau:.3e}, sigma2 {d_s2:.3e}")
    assert d_tau_lib <= 1e-10 and d_s2_lib <= 1e-10, (
        f"noise vs gpytorch: tau {d_tau_lib:.3e}, sigma2 {d_s2_lib:.3e}")
    assert (tau >= NOISE_FLOOR).all() and (s2 >= NOISE_FLOOR).all()
    assert nz["floor"] == NOISE_FLOOR
    # raw_var has NO floor: softplus(-5) is far below 1e-4 and must stay there.
    tiny = icm.task_covariance(np.zeros((1, 1)), np.array([-20.0]))
    assert 0.0 < tiny[0, 0] < NOISE_FLOOR, (
        f"raw_var appears to carry a 1e-4 offset it must not have: {tiny[0,0]!r}")
    report(f"T3/T4: B vs gpytorch {d_lib:.3e}; tau vs gpytorch {d_tau_lib:.3e}; "
           f"sigma2 vs gpytorch {d_s2_lib:.3e}; raw_var offset-free "
           f"(softplus(-20) = {tiny[0,0]:.3e} < 1e-4)")


@pytest.mark.parametrize("K,R", [(1, 1), (2, 1), (2, 2), (3, 1), (3, 2), (5, 3)])
def test_parameter_count_and_roundtrip(K, R):
    """``K*R + 3K + 1`` — 9 at ``K=2, R=1``, not ``3T+1``; and pack/unpack is
    a bijection in :data:`icm.PARAM_ORDER`."""
    n = icm.n_params(K, R)
    assert n == K * R + 3 * K + 1
    if (K, R) == (2, 1):
        assert n == 9 == fixture("01_parameterization.json")["trainable_scalar_count"]["K2_R1"]
    theta = np.arange(n, dtype=np.float64) + 0.5
    W, v, tn, gn, c = icm.unpack(theta, K, R)
    assert W.shape == (K, R) and v.shape == (K,) and tn.shape == (K,)
    assert np.ndim(gn) == 0, "raw_noise is ONE shared scalar, not one per task"
    assert c.shape == (K,)
    back = icm.theta_from_groups(W, v, tn, np.atleast_1d(gn), c)
    assert maxabs(back, theta) == 0.0
    groups = icm.split_grad(theta, K, R)
    assert list(groups) == list(icm.PARAM_ORDER)
    flat = np.concatenate([np.asarray(groups[k]).reshape(-1) for k in icm.PARAM_ORDER])
    assert maxabs(flat, theta) == 0.0


def test_initialization_matches_the_shipped_scheme():
    """``W, v_raw ~ randn``; ``c = tau_raw = sigma_raw = 0`` exactly (SPEC §6).

    The three zeros are not stylistic: ``ConstantMean.raw_constant``,
    ``raw_task_noises`` and ``raw_noise`` all initialize to literal ``0.0`` in
    gpytorch, which puts ``tau = sigma2 = 1e-4 + log 2`` at step 0.
    """
    j = fixture("01_parameterization.json")["initialization"]
    assert j["constant_mean_raw_constant_init"] == 0.0
    assert j["raw_task_noises_init"] == [0.0, 0.0]
    assert j["raw_noise_init"] == [0.0]
    assert j["index_kernel_is_randn"] is True
    assert j["index_kernel_redrawn_each_construction"] is True
    theta = icm.init_theta(K_TASKS, RANK, rng=0)
    W, v, tn, gn, c = icm.unpack(theta, K_TASKS, RANK)
    assert maxabs(tn, np.zeros(K_TASKS)) == 0.0
    assert float(gn) == 0.0
    assert maxabs(c, np.zeros(K_TASKS)) == 0.0
    assert not np.allclose(W, 0.0) and not np.allclose(v, 0.0)
    # redrawn: two calls off the global stream differ, a seeded call repeats.
    a = icm.init_theta(K_TASKS, RANK)
    b = icm.init_theta(K_TASKS, RANK)
    assert maxabs(a, b) > 0.0, "init must be REDRAWN, as IndexKernel's is"
    assert maxabs(icm.init_theta(K_TASKS, RANK, rng=3),
                  icm.init_theta(K_TASKS, RANK, rng=3)) == 0.0
    B, tau, s2, d, _ = icm.natural(
        icm.theta_from_groups(np.zeros((K_TASKS, RANK)), np.zeros(K_TASKS),
                              np.zeros(K_TASKS), np.zeros(1), np.zeros(K_TASKS)),
        K_TASKS, RANK)
    assert abs(float(tau[0]) - (NOISE_FLOOR + math.log(2.0))) <= 1e-15
    assert abs(float(s2) - (NOISE_FLOOR + math.log(2.0))) <= 1e-15
    report(f"init: tau_raw = sigma_raw = c = 0 exactly; at zero raws "
           f"tau = sigma2 = {float(tau[0]):.9f} = 1e-4 + log 2; W/v redrawn")


# ===========================================================================
# T5 - T8 — the Kronecker factorization vs a dense Cholesky of the SAME Sigma
# ===========================================================================

def test_kronecker_logdet_solve_and_positive_definiteness(real_case):
    """T5/T6/T7/T8 in one place, because they share the same dense ``Sigma``.

    The Kronecker path (L1)/(S5) is compared against a dense Cholesky of the
    matrix assembled independently by :func:`icm.dense_covariance` — the two
    sides share no algebraic step.  And ``Sigma`` is checked positive definite
    with **no jitter added**: SPEC §3.1 proves it always is (``S >= 1`` and
    ``d_t >= 2e-4``), so any jitter would be a silent model change.
    """
    Xtr, _, Yn, K_xx, lam, U, theta = real_case
    N = Yn.shape[0]
    fx = fixture("04_kron.json")
    assert maxabs(K_xx, arr("04_kron/K_xx.npy")) == 0.0, "Gram drifted from the fixture"

    B, tau, s2, d, c = icm.natural(theta, K_TASKS, RANK)
    assert maxabs(B, arr("04_kron/B.npy")) <= 1e-12
    mu, Vt, S = icm.prepare(lam, B, d)
    assert (S >= 1.0).all(), "S = lam*mu + 1 dropped below 1; a clamp was lost"

    ld_kron = icm.logdet(N, d, S)
    R = Yn - c[None, :]
    A_kron = icm.solve(R, U, Vt, S)

    Sigma = icm.dense_covariance(K_xx, B, tau, s2)
    assert maxabs(Sigma, arr("04_kron/Sigma_dense.npy")) <= 1e-12
    L = np.linalg.cholesky(Sigma)                       # no jitter, T8
    ld_chol = float(2.0 * np.log(np.diag(L)).sum())
    A_chol = np.linalg.solve(Sigma, R.reshape(-1)).reshape(N, K_TASKS)

    d_ld = abs(ld_kron - ld_chol)
    d_solve = maxabs(A_kron, A_chol)
    res_kron = float(np.abs(Sigma @ A_kron.reshape(-1) - R.reshape(-1)).max())
    res_chol = float(np.abs(Sigma @ A_chol.reshape(-1) - R.reshape(-1)).max())
    min_eig = float(np.linalg.eigvalsh(Sigma).min())

    assert d_ld <= 1e-9, f"T5 logdet Kronecker vs dense Cholesky: {d_ld:.3e}"
    assert d_solve <= 1e-10, f"T6 solve Kronecker vs dense: {d_solve:.3e}"
    assert res_kron <= 1e-9, f"T7 |Sigma alpha - r|_inf, Kronecker: {res_kron:.3e}"
    assert min_eig > 0.0, f"T8 Sigma is not PD without jitter: min eig {min_eig:.3e}"

    # and against the archived arrays
    assert abs(ld_kron - fx["logdet_kron"]) <= 1e-9
    assert maxabs(A_kron, arr("04_kron/alpha_kron.npy")) <= 1e-10
    assert maxabs(A_kron, arr("04_kron/alpha_chol.npy")) <= 1e-10
    report(f"T5-T8 (N={N}, NK={N*K_TASKS}): logdet |kron - chol| {d_ld:.3e} "
           f"(rel {d_ld/abs(ld_chol):.3e}); solve {d_solve:.3e}; residual "
           f"kron {res_kron:.3e} vs chol {res_chol:.3e}; min eig(Sigma) "
           f"{min_eig:.4f} with zero jitter")


def test_vtilde_scales_rows_not_columns(real_case):
    """SPEC §10.7, the SECOND silent transpose: ``Vt = D^-1/2 V`` scales ROWS.

    ``V * inv[None, :]`` still yields a symmetric ``M_B`` and a plausible loss,
    so it is checked structurally: only the row-scaled ``Vt`` satisfies
    ``Vt diag(mu) Vt^T == B^-1``... in fact ``Vt`` diagonalizes ``Btil``, so the
    identity to check is ``Vt^T D Vt == I`` and ``Vt diag(mu) Vt^T == D^-1 B D^-1``.
    """
    _, _, Yn, K_xx, lam, U, theta = real_case
    B, tau, s2, d, _ = icm.natural(theta, K_TASKS, RANK)
    mu, Vt, S = icm.prepare(lam, B, d)
    D = np.diag(d)
    d1 = maxabs(Vt.T @ D @ Vt, np.eye(K_TASKS))
    d2 = maxabs(Vt @ np.diag(mu) @ Vt.T, np.linalg.solve(D, B) @ np.linalg.inv(D))
    assert d1 <= 1e-12, f"Vt^T D Vt != I by {d1:.3e} — Vt is scaled wrongly"
    assert d2 <= 1e-12, f"Vt diag(mu) Vt^T != D^-1 B D^-1 by {d2:.3e}"
    wrong = (1.0 / np.sqrt(d))[None, :] * np.linalg.eigh(
        B * (1.0 / np.sqrt(d))[:, None] * (1.0 / np.sqrt(d))[None, :])[1]
    assert maxabs(wrong.T @ D @ wrong, np.eye(K_TASKS)) > 1e-3, (
        "the column-scaled variant is NOT distinguishable here; the test is vacuous")
    report(f"Vt row-scaling: |Vt^T D Vt - I| {d1:.3e}, "
           f"|Vt diag(mu) Vt^T - D^-1 B D^-1| {d2:.3e}")


# ===========================================================================
# T9 - T11 — the loss and the analytic gradients
# ===========================================================================

def test_loss_normalization_is_N_times_K(real_case):
    """T11: ``loss == -log p /(N*K)``, and it equals the archived reference.

    gpytorch's ``ExactMarginalLogLikelihood`` divides by the event shape's
    element count, so the divisor is ``N*K`` and not ``N``.  Reproducing it is
    what keeps loss numbers comparable to the archive.
    """
    _, _, Yn, K_xx, lam, U, theta = real_case
    N, K = Yn.shape
    loss, grad, ex = icm.nmll_and_grad(theta, K_xx, Yn, lam=lam, U=U, rank=RANK)
    want = fixture("06_gradients.json")["nll_reference"]
    assert abs(loss - want) <= 1e-12, f"loss vs the archived reference: {abs(loss-want):.3e}"
    assert abs(loss - (-ex["logp"] / (N * K))) <= 1e-15
    wrong = -ex["logp"] / N
    assert abs(loss - wrong) > 1e-3, "the N vs N*K divisor is not distinguishable here"
    # the posterior fixture's gpytorch MLL is -loss
    p5 = fixture("05_posterior.json")
    d_mll = abs(p5["mll_normalized_gpytorch"] + loss)
    assert d_mll <= 1e-9, f"gpytorch MLL + loss should be ~0, got {d_mll:.3e}"
    report(f"T11: loss {loss!r} vs archive {want!r} (diff "
           f"{abs(loss-want):.3e}); gpytorch MLL + loss = {d_mll:.3e}; "
           f"the /N variant would be {wrong:.6f}")


@pytest.mark.parametrize("case", sorted(GRAD_CASES))
def test_analytic_gradient_matches_live_autograd_per_group(case, real_case):
    """T9, live and PER GROUP: every group checked separately against autograd.

    Per group matters: a sign error in one group is masked by the others in a
    flat-vector comparison.  The oracle is a dense-Cholesky NLL built
    independently of the Kronecker path, so (L1), (S5) and (G1)-(G6) are
    cross-validated rather than compared with themselves.
    """
    _, _, Yn, K_xx, lam, U, fixture_theta = real_case
    theta, rank = GRAD_CASES[case]
    if theta is None:
        theta = fixture_theta
    nll_t, auto = live_autograd_groups(theta, K_xx, Yn, rank=rank)
    loss, grad, _ = icm.nmll_and_grad(theta, K_xx, Yn, lam=lam, U=U, rank=rank)
    mine = icm.split_grad(grad, K_TASKS, rank)
    assert abs(loss - nll_t) <= 1e-12, (
        f"{case}: loss vs the dense autograd oracle: {abs(loss-nll_t):.3e}")
    worst = {}
    for name in icm.PARAM_ORDER:
        a = np.asarray(mine[name]).reshape(-1)
        b = np.asarray(auto[name]).reshape(-1)
        assert a.shape == b.shape, f"{case}/{name}: shape {a.shape} vs {b.shape}"
        da, dr = maxabs(a, b), maxrel(a, b)
        worst[name] = (da, dr)
        assert da <= 1e-13, (
            f"{case}: gradient group {name} vs autograd: max abs {da:.3e}, "
            f"max rel {dr:.3e}; mine {a.tolist()} vs autograd {b.tolist()}")
    report(f"T9 live {case} (R={rank}): loss {abs(loss-nll_t):.3e}; per group "
           + ", ".join(f"{k} {v[0]:.2e}" for k, v in worst.items()))


def test_analytic_gradient_matches_the_archived_oracles(real_case):
    """T9 archived: against the stored analytic / autograd / finite-difference
    arrays, in the fixture's own flat layout."""
    _, _, Yn, K_xx, lam, U, theta = real_case
    _, grad, _ = icm.nmll_and_grad(theta, K_xx, Yn, lam=lam, U=U, rank=RANK)
    j = fixture("06_gradients.json")
    assert j["layout"] == ["W (K*R)", "raw_var (K)", "raw_task_noises (K)",
                           "raw_noise (1)", "mean_constant (K)"]
    d_ana = maxabs(grad, arr("06_gradients/grad_analytic.npy"))
    d_auto = maxabs(grad, arr("06_gradients/grad_autograd.npy"))
    d_fd = maxabs(grad, arr("06_gradients/grad_finite_difference.npy"))
    assert d_ana <= 1e-13, f"vs the archived analytic gradient: {d_ana:.3e}"
    assert d_auto <= 1e-13, f"vs the archived autograd gradient: {d_auto:.3e}"
    assert d_fd <= 1e-8, f"vs the archived finite differences: {d_fd:.3e}"
    mine = icm.split_grad(grad, K_TASKS, RANK)
    for name, ref in j["per_group"].items():
        d = maxabs(np.asarray(mine[name]).reshape(-1), np.asarray(ref["autograd"]))
        assert d <= 1e-13, f"group {name} vs the archived autograd: {d:.3e}"
    report(f"T9 archived: vs analytic {d_ana:.3e}, vs autograd {d_auto:.3e}, "
           f"vs finite differences {d_fd:.3e}")


@pytest.mark.parametrize("case", sorted(GRAD_CASES))
def test_analytic_gradient_matches_central_finite_differences(case, real_case):
    """T10, NO oracle: central differences of this module's own loss, ``eps=1e-6``.

    Survives the removal of torch entirely.  The loss and the gradient come from
    the same function but by different routes — the gradient from the closed
    forms (G1)-(G6), the reference from re-evaluating the loss — so a wrong
    contraction cannot cancel.
    """
    _, _, Yn, K_xx, lam, U, fixture_theta = real_case
    theta, rank = GRAD_CASES[case]
    if theta is None:
        theta = fixture_theta
    theta = np.asarray(theta, dtype=np.float64)
    _, grad, _ = icm.nmll_and_grad(theta, K_xx, Yn, lam=lam, U=U, rank=rank)
    eps = 1e-6
    fd = np.empty_like(theta)
    for i in range(theta.size):
        hi = theta.copy(); hi[i] += eps
        lo = theta.copy(); lo[i] -= eps
        f_hi = icm.nmll_and_grad(hi, K_xx, Yn, lam=lam, U=U, rank=rank,
                                 want_grad=False)[0]
        f_lo = icm.nmll_and_grad(lo, K_xx, Yn, lam=lam, U=U, rank=rank,
                                 want_grad=False)[0]
        fd[i] = (f_hi - f_lo) / (2 * eps)
    d = maxabs(grad, fd)
    assert d <= 1e-8, (
        f"{case}: analytic gradient vs central differences at eps={eps}: "
        f"{d:.3e}\n  analytic {grad.tolist()}\n  finite   {fd.tolist()}")
    report(f"T10 {case} (R={rank}): |analytic - central FD(1e-6)| = {d:.3e}")


def test_the_global_noise_gradient_sums_over_all_tasks(real_case):
    """The shared ``sigma2`` couples the tasks: ``dlogp/dsigma_raw = sum_t M_D[t]``.

    If it were per task this model would be ``K`` independent GPs with extra
    steps.  Checked against the closed form and against autograd's own value.
    """
    _, _, Yn, K_xx, lam, U, theta = real_case
    N, K = Yn.shape
    loss, grad, ex = icm.nmll_and_grad(theta, K_xx, Yn, lam=lam, U=U, rank=RANK)
    g = icm.split_grad(grad, K, RANK)
    _, _, _, sigma_raw, _ = icm.unpack(theta, K, RANK)
    want = -(1.0 / (N * K)) * ex["M_D"].sum() * float(
        mtr.softplus_grad(np.reshape(sigma_raw, (1,)))[0])
    d = abs(float(g["raw_noise"][0]) - want)
    assert d <= 1e-15, f"dloss/dsigma_raw closed form: {d:.3e}"
    per_task_only = -(1.0 / (N * K)) * ex["M_D"][0] * float(
        mtr.softplus_grad(np.reshape(sigma_raw, (1,)))[0])
    assert abs(float(g["raw_noise"][0]) - per_task_only) > 1e-6, (
        "summing over tasks is not distinguishable here; the test is vacuous")
    report(f"global-noise coupling: dloss/dsigma_raw = {float(g['raw_noise'][0]):.9f} "
           f"= -(1/NK) sum_t M_D[t] sigmoid(sigma_raw) (diff {d:.3e}); the "
           f"single-task variant would be {per_task_only:.9f}")


# ===========================================================================
# T12 / T13 — the posterior
# ===========================================================================

def test_posterior_mean_variance_and_full_covariance(real_case):
    """T12: mean, latent variance, observed variance and the FULL ``MK x MK``
    joint against the archived gpytorch-validated arrays."""
    Xtr, Xte, Yn, K_xx, lam, U, theta = real_case
    mean, var_lat, var_obs = icm.posterior(theta, Xtr, Yn, Xte, rank=RANK,
                                           K_xx=K_xx, lam=lam, U=U)
    cov = icm.posterior_covariance(theta, Xtr, Yn, Xte, rank=RANK, K_xx=K_xx,
                                   lam=lam, U=U)
    d_mean = maxabs(mean, arr("05_posterior/mean_reference.npy"))
    d_vlat = maxabs(var_lat, arr("05_posterior/var_latent_reference.npy"))
    d_vobs = maxabs(var_obs, arr("05_posterior/var_observed_reference.npy"))
    cov_ref = arr("05_posterior/cov_latent_reference.npy")
    d_cov = maxabs(cov, cov_ref)
    for name, d in (("mean", d_mean), ("var_latent", d_vlat),
                    ("var_observed", d_vobs), ("cov_latent", d_cov)):
        assert d <= 1e-9, f"T12 {name} vs the archived array: {d:.3e}"
    # (O4): the marginal path must be the diagonal of the joint path.
    M = Xte.shape[0]
    d_o4 = maxabs(np.diag(cov).reshape(M, K_TASKS), var_lat)
    assert d_o4 <= 1e-12, f"diag(joint) != marginals by {d_o4:.3e} — (O4) broken"
    # (R3): observed = latent + d, with the GLOBAL sigma2 included.
    _, tau, s2, d_noise, _ = icm.natural(theta, K_TASKS, RANK)
    assert maxabs(var_obs - var_lat, np.tile(d_noise, (M, 1))) <= 1e-12
    assert maxabs(var_obs - var_lat, np.tile(tau, (M, 1))) > 1e-3, (
        "var_observed appears to add only tau, not tau + sigma2")
    # (R5): the within-point blocks must agree with the joint's diagonal blocks.
    blocks = icm.posterior_block_covariance(theta, Xtr, Yn, Xte, rank=RANK,
                                            K_xx=K_xx, lam=lam, U=U)
    d_blk = max(maxabs(blocks[i], cov_ref[i*K_TASKS:(i+1)*K_TASKS,
                                          i*K_TASKS:(i+1)*K_TASKS])
                for i in range(M))
    assert d_blk <= 1e-9, f"(R5) within-point blocks: {d_blk:.3e}"
    # the observation_noise option on both joint paths adds exactly I_M (x) diag(d)
    cov_n = icm.posterior_covariance(theta, Xtr, Yn, Xte, rank=RANK, K_xx=K_xx,
                                     lam=lam, U=U, observation_noise=True)
    blk_n = icm.posterior_block_covariance(theta, Xtr, Yn, Xte, rank=RANK,
                                          K_xx=K_xx, lam=lam, U=U,
                                          observation_noise=True)
    d_cn = maxabs(cov_n - cov, np.kron(np.eye(M), np.diag(d_noise)))
    d_bn = max(maxabs(blk_n[i] - blocks[i], np.diag(d_noise)) for i in range(M))
    assert d_cn <= 1e-12 and d_bn <= 1e-12, (
        f"observation_noise on the joint paths: full {d_cn:.3e}, blocks {d_bn:.3e}")
    report(f"T12 (N=40, M=40): mean {d_mean:.3e}, var_latent {d_vlat:.3e}, "
           f"var_observed {d_vobs:.3e}, full {M*K_TASKS}x{M*K_TASKS} covariance "
           f"{d_cov:.3e}; diag(joint) vs marginals {d_o4:.3e}; (R5) blocks {d_blk:.3e}")


def test_posterior_is_invariant_to_how_query_rows_are_batched(real_case):
    """The float32 torch path's output depends on its batch shape (measured
    ``1.1e-06`` in the acquisition gate).  This float64 path must not.

    Every query row goes through row-independent formulas, so predicting 40 rows
    at once, in two halves, or one at a time must agree to BLAS rounding.  A
    caller still has to keep botorch's stacked call pattern, because the
    consumers downstream were verified against it — but the GP itself no longer
    punishes getting it wrong.
    """
    Xtr, Xte, Yn, K_xx, lam, U, theta = real_case
    kw = dict(rank=RANK, K_xx=K_xx, lam=lam, U=U)
    full = icm.posterior(theta, Xtr, Yn, Xte, **kw)
    half = [np.concatenate(z, axis=0) for z in zip(
        icm.posterior(theta, Xtr, Yn, Xte[:17], **kw),
        icm.posterior(theta, Xtr, Yn, Xte[17:], **kw))]
    rows = [np.concatenate(z, axis=0) for z in zip(
        *[icm.posterior(theta, Xtr, Yn, Xte[i:i+1], **kw) for i in range(len(Xte))])]
    d_half = max(maxabs(a, b) for a, b in zip(full, half))
    d_rows = max(maxabs(a, b) for a, b in zip(full, rows))
    assert d_half <= 1e-12 and d_rows <= 1e-12, (
        f"posterior depends on the query batching: split {d_half:.3e}, "
        f"per row {d_rows:.3e}")
    report(f"batch-shape invariance: |40-at-once - 17+23| = {d_half:.3e}; "
           f"|40-at-once - one row at a time| = {d_rows:.3e} "
           f"(the float32 torch path moves by 1.1e-06)")


def test_posterior_against_the_shipped_float32_predict(real_case):
    """T13: the numpy posterior vs the shipped ``predict_coregionalized``.

    SAME fitted parameters on both sides, so the only difference is float32 vs
    float64 linear algebra (plus ``fast_pred_var``, worth at most ``5.96e-07``
    on the variance and exactly ``0.0`` on the mean).  This is the only test here
    that needs the live library and it can be deleted once measured.

    REPAIRED 2026-09-28, same claim, same tolerances (1e-5 relative on both mean
    and variance).  ``mogp_coregionalized`` is numpy now, so the gpytorch model is
    the one :func:`live_coregionalized_classes` rebuilds from git ``3c6573c``.
    This test stays torch-conditional and cannot be made unconditional: what it
    compares is the shipped **float32** ``predict`` path, ``fast_pred_var`` and
    de-standardization included, and ``spec/icm/fixtures/05_posterior/`` holds only
    the float64 *latent* reference arrays (used by
    :func:`test_posterior_mean_variance_and_full_covariance`).  No fixture stores
    this quantity, so there is nothing to repoint it at.
    """
    torch, gpytorch = _oracle()
    model_cls, predict_coregionalized = live_coregionalized_classes()
    Xtr, Xte, Yn, K_xx, lam, U, theta = real_case
    le = arr("03_real_data/le_train.npy")
    Y5 = np.full((Xtr.shape[0], len(icm.TASK_NAMES)), np.nan, dtype=np.float32)
    Y5[:, 0], Y5[:, 1] = le[:, 0], le[:, 1]
    y_mean, y_std, observed = icm.standardize(Y5)
    assert maxabs(y_mean[:2], fixture("03_real_data.json")["y_mean"]) == 0.0
    assert maxabs(y_std[:2], fixture("03_real_data.json")["y_std"]) == 0.0

    state = icm.ICMState(theta, Xtr, Yn.astype(np.float32), observed, rank=RANK,
                         K_xx=K_xx, lam=lam, U=U)
    mean_np, var_np = icm.predict(state, y_mean, y_std, Xte)

    Xtr32 = np.ascontiguousarray(Xtr, dtype=np.float32)
    Yn32 = np.ascontiguousarray(Yn, dtype=np.float32)
    lik = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=K_TASKS)
    mo = model_cls(torch.from_numpy(Xtr32),
                   torch.from_numpy(Yn32), lik, rank=RANK)
    W, v, tn, gn, c = icm.unpack(theta, K_TASKS, RANK)
    with torch.no_grad():
        mo.covar_module.task_covar_module.covar_factor.copy_(
            torch.tensor(np.asarray(W, dtype=np.float32)))
        mo.covar_module.task_covar_module.raw_var.copy_(
            torch.tensor(np.asarray(v, dtype=np.float32)))
        lik.raw_task_noises.copy_(torch.tensor(np.asarray(tn, dtype=np.float32)))
        lik.raw_noise.copy_(torch.tensor(np.asarray([gn], dtype=np.float32)))
        for t in range(K_TASKS):
            mo.mean_module.base_means[t].raw_constant.copy_(torch.tensor(float(c[t])))
    mean_t, var_t = predict_coregionalized(mo, lik, y_mean, y_std,
                                           np.ascontiguousarray(Xte,
                                                                dtype=np.float32))
    dock = list(icm.DOCKING_TASK_INDICES)
    r_mean = maxrel(mean_np[:, dock], mean_t[:, dock])
    r_var = maxrel(var_np[:, dock], var_t[:, dock])
    a_mean = maxabs(mean_np[:, dock], mean_t[:, dock])
    a_var = maxabs(var_np[:, dock], var_t[:, dock])
    assert r_mean <= 1e-5, f"T13 posterior mean vs shipped float32: rel {r_mean:.3e}"
    assert r_var <= 1e-5, f"T13 posterior variance vs shipped float32: rel {r_var:.3e}"
    # the grey-box NaN columns must agree too
    admet = [j for j in range(len(icm.TASK_NAMES)) if j not in dock]
    assert np.all(np.isnan(mean_np[:, admet])) and np.all(np.isnan(mean_t[:, admet]))
    d_B = maxabs(state.task_covariance_matrix(), mo.task_covariance_matrix())
    report(f"T13 vs shipped float32 predict: mean abs {a_mean:.3e} rel "
           f"{r_mean:.3e}; variance abs {a_var:.3e} rel {r_var:.3e}; "
           f"B (float64 vs float32) {d_B:.3e}")


def test_task_covariance_matrix_method_matches_the_shipped_one(real_case):
    """``ICMState.task_covariance_matrix`` reproduces ``MOGPCoregionalized``'s.

    The shipped method reads ``IndexKernel._eval_covar_matrix()``; the docstring
    around it says ``B B^T + diag(v)``, omitting the ``softplus`` (SPEC D2).  The
    code, not the docstring, is what is mirrored.

    UNCONDITIONAL since 2026-09-28, same claim at the same ``1e-12``.
    ``01_parameterization.json → task_covariance.B_gpytorch`` IS
    ``IndexKernel._eval_covar_matrix()`` at ``raw_W`` / ``raw_var``, captured
    2026-09-27, so both halves of the D2 claim — that the code's ``softplus`` form
    matches the library and that the docstring's offset-free form does not — are
    asserted from the archive with no gpytorch present, and the
    ``ICMState.task_covariance_matrix`` method (the only place it is exercised in
    this file) runs either way.  The live half then rebuilds the ``IndexKernel``
    and re-measures.
    """
    Xtr, _, Yn, K_xx, lam, U, theta = real_case

    # -- unconditional: the archived IndexKernel matrix as the oracle ----------
    tc = fixture("01_parameterization.json")["task_covariance"]
    W_fx = np.asarray(tc["raw_W"])
    v_fx = np.asarray(tc["raw_var"])
    B_lib = np.asarray(tc["B_gpytorch"])
    d_arch = maxabs(icm.task_covariance(W_fx, v_fx), B_lib)
    assert d_arch <= 1e-12, (
        f"B vs the ARCHIVED IndexKernel._eval_covar_matrix: {d_arch:.3e}")
    assert maxabs(W_fx @ W_fx.T + np.diag(v_fx), B_lib) > 1e-2, (
        "the docstring's softplus-free formula is not distinguishable on the "
        "archived parameters either, so D2 has no negative control left")
    # the METHOD, not just the free function, at the real case's own theta
    W, v, _, _, _ = icm.unpack(theta, K_TASKS, RANK)
    state_u = icm.ICMState(theta, Xtr, Yn, np.ones(K_TASKS, dtype=bool),
                           rank=RANK, K_xx=K_xx, lam=lam, U=U)
    assert maxabs(state_u.task_covariance_matrix(),
                  icm.task_covariance(W, v)) == 0.0, (
        "ICMState.task_covariance_matrix is not icm.task_covariance(W, v)")

    # -- live -----------------------------------------------------------------
    torch, gpytorch = _oracle_or_none()
    if torch is None:                                         # pragma: no cover
        report(f"D2: gpytorch absent -> archived IndexKernel matrix is the "
               f"oracle ({d_arch:.3e}); the docstring's diag(raw_var) form is "
               f"off by {maxabs(W_fx @ W_fx.T + np.diag(v_fx), B_lib):.3e}")
        return
    # first re-derive the archived matrix, so the stored oracle stays honest
    ik_fx = gpytorch.kernels.IndexKernel(num_tasks=K_TASKS,
                                         rank=W_fx.shape[1]).double()
    with torch.no_grad():
        ik_fx.covar_factor.copy_(torch.tensor(np.ascontiguousarray(W_fx)))
        ik_fx.raw_var.copy_(torch.tensor(np.ascontiguousarray(v_fx)))
    d_fx = maxabs(ik_fx._eval_covar_matrix().detach().numpy(), B_lib)
    assert d_fx <= 1e-12, (
        f"live IndexKernel no longer reproduces the archived B_gpytorch: "
        f"{d_fx:.3e}")

    ik = gpytorch.kernels.IndexKernel(num_tasks=K_TASKS, rank=RANK).double()
    with torch.no_grad():
        ik.covar_factor.copy_(torch.tensor(np.asarray(W, dtype=np.float64)))
        ik.raw_var.copy_(torch.tensor(np.asarray(v, dtype=np.float64)))
    d = maxabs(icm.task_covariance(W, v), ik._eval_covar_matrix().detach().numpy())
    assert d <= 1e-12, f"B vs IndexKernel._eval_covar_matrix: {d:.3e}"
    no_softplus = np.asarray(W) @ np.asarray(W).T + np.diag(v)
    assert maxabs(no_softplus, ik._eval_covar_matrix().detach().numpy()) > 1e-2, (
        "the docstring's softplus-free formula is not distinguishable here")
    report(f"D2: B = W W^T + diag(softplus(raw_var)) vs IndexKernel = {d:.3e}; "
           f"the docstring's diag(raw_var) form is off by "
           f"{maxabs(no_softplus, ik._eval_covar_matrix().detach().numpy()):.3e}")


# ===========================================================================
# T14 / T15 / T23 — identifiability.  NO oracle needed.
# ===========================================================================

def test_B_is_invariant_under_W_times_any_orthogonal_Q():
    """T14: ``W -> W Q`` leaves ``B`` — and the whole likelihood — unchanged.

    The negative control for every ``W`` comparison: a correct reimplementation
    will disagree with the library on ``W`` BY CONSTRUCTION, so ``W`` is not a
    legitimate target at any rank.
    """
    j = fixture("07_identifiability.json")["invariance_proof"]["orthogonal_check"]
    rng = np.random.default_rng(20260927)
    worst = 0.0
    for K, R in ((3, 2), (2, 1), (2, 2), (5, 3)):
        W = rng.standard_normal((K, R))
        v = rng.standard_normal(K)
        A = rng.standard_normal((R, R))
        Q, _ = np.linalg.qr(A)
        assert abs(abs(np.linalg.det(Q)) - 1.0) <= 1e-12
        worst = max(worst, maxabs(icm.task_covariance(W, v),
                                  icm.task_covariance(W @ Q, v)))
    assert worst <= 1e-12, f"T14 B is not W-Q invariant: {worst:.3e}"
    assert j["max_abs_diff_B_vs_B_WQ"] <= 1e-12
    # at R=1 the invariance degenerates to a sign flip
    W1 = np.array([[0.83], [-1.42]])
    v1 = np.array([-0.2, 0.6])
    assert maxabs(icm.task_covariance(W1, v1), icm.task_covariance(-W1, v1)) == 0.0
    report(f"T14: max|B(W) - B(WQ)| = {worst:.3e} over K,R in "
           f"(3,2),(2,1),(2,2),(5,3); at R=1, B(W) == B(-W) exactly")


def test_a_continuous_family_of_W_v_gives_one_identical_B():
    """T15: at ``K=2, R=1`` the map ``(W, v) -> B`` has a one-parameter fibre.

    4 parameters, 3 free entries in ``B``.  So ``W`` is not identified even up to
    sign, and neither is ``raw_var``.
    """
    fam = fixture("07_identifiability.json")["invariance_proof"]["continuous_family_K2_R1"]
    target = np.asarray(fam["target_B"], dtype=np.float64)
    b11, b12, b22 = target[0, 0], target[0, 1], target[1, 1]
    worst = 0.0
    n = 0
    for entry in fam["family"]:
        t = float(entry["t"])
        W = np.array([[t], [b12 / t]])
        v = np.array([b11 - t ** 2, b22 - (b12 / t) ** 2])
        assert (v > 0).all(), f"t={t}: v must stay positive, got {v.tolist()}"
        raw = mtr.inv_softplus(v)
        worst = max(worst, maxabs(icm.task_covariance(W, raw), target))
        n += 1
    assert n >= 5, f"expected the fixture's 5 members, got {n}"
    assert worst <= 1e-12, f"T15 the family does not rebuild one B: {worst:.3e}"
    report(f"T15: {n} distinct (W, v) with v > 0 rebuild the same B to "
           f"{worst:.3e}; W ranges {fam['family'][0]['W']} .. {fam['family'][-1]['W']}")


FORBIDDEN_FITTED = ("W", "raw_var", "raw_task_noises", "raw_noise",
                    "covar_factor")


def test_no_test_here_pins_an_unidentified_fitted_parameter():
    """T23, anti-regression, enforced by parsing THIS FILE's AST.

    ``W``, ``raw_var``, ``raw_task_noises`` and ``raw_noise`` are not identified
    (T14, T15), so no test may assert on their FITTED values.  A test is
    "fit-touching" when it calls :func:`icm.fit` or
    :func:`icm.train_mogp_coregionalized`, or when it consumes a fixture that
    did; such a test must not mention any of those names inside an ``assert``.

    Two things are deliberately NOT caught, because they are different claims:

    * comparing ``dloss/dW`` at a FIXED theta against autograd — the derivative
      of the loss along a coordinate is well defined even where the coordinate is
      not identified, which is why the guard keys on "did this test fit?" rather
      than on the identifier alone;
    * :func:`icm.init_theta`, which draws rather than estimates.  Asserting that
      a ``randn`` draw is nonzero pins nothing.
    """
    with open(os.path.abspath(__file__)) as fh:
        tree = ast.parse(fh.read(), filename=__file__)
    FITTERS = {"fit", "train_mogp_coregionalized"}
    FIT_FIXTURES = {"schedule_matched_fits"}
    offenders = []
    checked = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.FunctionDef) and node.name.startswith("test_")):
            continue
        calls = {n.func.attr for n in ast.walk(node)
                 if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
        params = {a.arg for a in node.args.args}
        if not ((calls & FITTERS) or (params & FIT_FIXTURES)):
            continue
        checked.append(node.name)
        for st in ast.walk(node):
            if not isinstance(st, ast.Assert):
                continue
            for sub in ast.walk(st.test):
                bad = None
                if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                    bad = sub.value if sub.value in FORBIDDEN_FITTED else None
                elif isinstance(sub, ast.Attribute):
                    bad = sub.attr if sub.attr in FORBIDDEN_FITTED else None
                elif isinstance(sub, ast.Name):
                    bad = sub.id if sub.id in FORBIDDEN_FITTED else None
                if bad:
                    offenders.append(f"{node.name}:{st.lineno} asserts on {bad!r}")
    assert not offenders, (
        "a fitting test asserts on an UNIDENTIFIED raw parameter (SPEC §7.4): "
        + "; ".join(offenders))
    assert len(checked) >= 3, (
        f"the guard only examined {checked}; it must cover the fitting tests")
    report(f"T23: {len(checked)} fitting tests scanned, 0 assertions on "
           f"{list(FORBIDDEN_FITTED)}")


# ===========================================================================
# T16 - T18 — the fit, on REAL docking data
# ===========================================================================

#: 8 numpy seeds at the shipped schedule, and the decayed-Adam schedule used to
#: drive the loss to its stationary point.  The decay is a TEST instrument for
#: reaching the MLE (SPEC T16 requires ``|grad|_inf < 1e-4``, which a fixed
#: ``lr=0.05`` does not reach on this flat boundary); the shipped model is, and
#: stays, 200 steps at ``lr=0.1``.
MLE_SCHEDULE = ((40000, 0.05), (20000, 5e-3), (20000, 5e-4), (20000, 5e-5))


@pytest.fixture(scope="session")
def schedule_matched_fits(real_case):
    """8 fits at the shipped schedule (200 Adam steps, ``lr=0.1``), real data."""
    Xtr, _, Yn, K_xx, lam, U, _ = real_case
    out = []
    for seed in range(8):
        theta, losses = icm.fit(Xtr, Yn, n_steps=200, lr=0.1, rank=RANK,
                                rng=seed, K_xx=K_xx)
        loss, grad, _ = icm.nmll_and_grad(theta, K_xx, Yn, lam=lam, U=U, rank=RANK)
        B, tau, s2, d, c = icm.natural(theta, K_TASKS, RANK)
        out.append(dict(seed=seed, loss=float(loss), B=B,
                        rho=float(icm.task_correlation(B)[0, 1]),
                        sigma2=float(s2), losses=losses,
                        first_component=float(np.asarray(
                            icm.unpack(theta, K_TASKS, RANK)[0]).reshape(-1)[0])))
    return out


def test_schedule_matched_task_correlation_on_real_data(schedule_matched_fits):
    """T17: 200 Adam steps, ``lr=0.1``, 8 seeds — ``rho`` and its spread.

    The target is the task CORRELATION, never a raw parameter, and the optimizer
    and stopping rule are stated with it because the ICM's ``B`` is only
    well-defined once they are (SPEC §7.4).
    """
    j = fixture("08_task_correlation.json")["shipped_schedule_200_adam_steps_lr0.1"]
    rho = np.array([f["rho"] for f in schedule_matched_fits])
    loss = np.array([f["loss"] for f in schedule_matched_fits])
    B01 = np.array([f["B"][0, 1] for f in schedule_matched_fits])
    B00 = np.array([f["B"][0, 0] for f in schedule_matched_fits])
    spread_rho = float(rho.max() - rho.min())
    spread_loss = float(loss.max() - loss.min())
    d_mean = abs(float(rho.mean()) - j["correlation_01_mean"])
    assert d_mean <= 1e-2, (
        f"T17 mean rho {rho.mean():.7f} vs the torch reference "
        f"{j['correlation_01_mean']:.7f}: {d_mean:.3e} > 1e-2 "
        f"(measured seed spread on the torch side {j['correlation_01_spread']:.3e})")
    assert spread_rho <= 3e-2, f"T17 rho spread {spread_rho:.3e} > 3e-2"
    # SPEC §10.4: B's DIAGONAL scatters several times more than B01, because
    # k(x,x) = 1 makes B[t,t], tau_t and sigma2 only jointly identified.
    assert float(B01.max() - B01.min()) < float(B00.max() - B00.min()), (
        "B01 should scatter LESS than B00; if not, the joint identifiability "
        "structure of SPEC §10.4 is not being reproduced")
    report(f"T17 real data, 200 steps lr=0.1, 8 numpy seeds: rho mean "
           f"{rho.mean():.7f} (torch reference {j['correlation_01_mean']:.7f}, "
           f"diff {d_mean:.3e}), range {rho.min():.7f}..{rho.max():.7f}, spread "
           f"{spread_rho:.3e} (torch {j['correlation_01_spread']:.3e}); loss "
           f"{loss.min():.7f}..{loss.max():.7f} spread {spread_loss:.3e} (torch "
           f"{fixture('07_identifiability.json')['shipped_model_multi_seed_real_data']['200']['loss_spread']:.3e}); "
           f"B00 spread {B00.max()-B00.min():.3e}, B01 spread {B01.max()-B01.min():.3e}")


def test_W_flips_sign_across_seeds_so_it_cannot_be_a_target(schedule_matched_fits):
    """The empirical half of T14/T23: the fitted factor's first component
    straddles zero across seeds while ``B`` barely moves.

    Reported, not asserted as a value — the point is precisely that there is no
    value to assert.
    """
    comp = np.array([f["first_component"] for f in schedule_matched_fits])
    B01 = np.array([f["B"][0, 1] for f in schedule_matched_fits])
    assert comp.min() < 0.0 < comp.max(), (
        "the factor's first component did not change sign across 8 seeds; the "
        "R=1 sign invariance is not being exercised, so this evidence is weak")
    assert float(B01.max() - B01.min()) < 5e-2, (
        f"B01 moved by {B01.max()-B01.min():.3e} across seeds, which is too "
        "much for the 'B is identified, the factor is not' claim")
    report(f"identifiability, empirical: the factor's first component ranges "
           f"{comp.min():+.4f}..{comp.max():+.4f} (sign flips) while B01 stays "
           f"in {B01.min():.6f}..{B01.max():.6f} (spread "
           f"{B01.max()-B01.min():.3e})")


def test_B_is_non_diagonal_and_positively_correlated(schedule_matched_fits):
    """T18: ``rho > 0.5`` on real PfDHFR/hDHFR data — the homology claim.

    This is the one thing the ICM learns that the independent model cannot.  It
    is then deleted by ``acquisition.py:323`` before the acquisition sees it
    (SPEC §8); see
    :func:`test_the_discarded_cross_task_structure_is_still_there_to_discard`.
    """
    rho = np.array([f["rho"] for f in schedule_matched_fits])
    assert (rho > 0.5).all(), f"T18 rho = {rho.tolist()}"
    assert (rho > 0.0).all(), "PfDHFR/hDHFR correlation must be POSITIVE"
    for f in schedule_matched_fits:
        off = abs(f["B"][0, 1])
        assert off > 1e-2, f"seed {f['seed']}: B is (near-)diagonal, B01={off:.3e}"
    emp = fixture("08_task_correlation.json")["empirical_pearson_r_of_the_training_targets"]
    report(f"T18: rho in {rho.min():.6f}..{rho.max():.6f}, all > 0.5; the "
           f"empirical Pearson r of the targets is {emp:.6f}")


def test_mle_matched_fit_reaches_the_boundary_stationary_point(real_case):
    """T16: drive ``|grad|_inf < 1e-4`` and land on the spec's boundary MLE.

    The sharper of the two honest test forms, because it does not depend on
    reproducing an optimizer trajectory.  What it finds, independently of torch,
    is SPEC §7.3's uncomfortable result: the constrained MLE has ``rho = 1``, a
    numerically singular ``B``, and both ``tau_PfDHFR`` and ``sigma2`` sitting ON
    the ``1e-4`` floor.  200 Adam steps is acting as early-stopping
    regularization and is the only thing keeping the shipped fit off that
    boundary.  This test verifies; it does not fix.
    """
    Xtr, _, Yn, K_xx, lam, U, _ = real_case
    ref = fixture("08_task_correlation.json")["reference_mle"]
    rows = []
    for seed in range(2):
        theta = icm.init_theta(K_TASKS, RANK, rng=seed)
        for n_steps, lr in MLE_SCHEDULE:
            theta, _ = icm.fit(Xtr, Yn, n_steps=n_steps, lr=lr, rank=RANK,
                               theta0=theta, K_xx=K_xx)
        loss, grad, _ = icm.nmll_and_grad(theta, K_xx, Yn, lam=lam, U=U, rank=RANK)
        B, tau, s2, d, c = icm.natural(theta, K_TASKS, RANK)
        rows.append(dict(seed=seed, nll=float(loss), B=B,
                         rho=float(icm.task_correlation(B)[0, 1]),
                         gradinf=float(np.abs(grad).max()),
                         detB=float(np.linalg.det(B)),
                         tau=tau, sigma2=float(s2)))
    for r in rows:
        assert r["gradinf"] < 1e-4, (
            f"seed {r['seed']}: |grad|_inf {r['gradinf']:.3e} — not at a "
            "stationary point, so the MLE claim is not being tested")
        assert abs(r["rho"] - 1.0) <= 1e-4, (
            f"seed {r['seed']}: |rho - 1| = {abs(r['rho']-1.0):.3e}")
        assert abs(r["nll"] - 0.8718243) <= 1e-4, (
            f"seed {r['seed']}: |nll - 0.8718243| = {abs(r['nll']-0.8718243):.3e}")
        # the boundary itself: SPEC §7.3
        assert r["detB"] < 1e-8, (
            f"seed {r['seed']}: det B = {r['detB']:.3e}; the MLE should be "
            "numerically singular")
        assert r["tau"][0] - NOISE_FLOOR < 1e-6 and r["sigma2"] - NOISE_FLOOR < 1e-6, (
            f"seed {r['seed']}: tau_0 and sigma2 should be ON the 1e-4 floor, "
            f"got {r['tau'][0]:.6e} and {r['sigma2']:.6e}")
    rho_spread = max(r["rho"] for r in rows) - min(r["rho"] for r in rows)
    assert abs(float(np.mean([r["rho"] for r in rows]))
               - ref["correlation_01_mean"]) <= 1e-6
    report("T16 MLE-matched (100k decayed Adam steps, float64, 2 inits): "
           + "; ".join(
               f"seed {r['seed']} nll {r['nll']:.9f} |grad|inf {r['gradinf']:.2e} "
               f"rho {r['rho']:.12f} detB {r['detB']:.3e} tau0-1e-4 "
               f"{r['tau'][0]-NOISE_FLOOR:.2e} sigma2-1e-4 {r['sigma2']-NOISE_FLOOR:.2e}"
               for r in rows)
           + f"; rho spread {rho_spread:.3e} (torch reference mean "
             f"{ref['correlation_01_mean']:.12f}, det B "
             f"{ref['det_B_per_init'][0]:.3e}..{ref['det_B_per_init'][2]:.3e})")


def test_rank_two_is_redundant_at_two_tasks(real_case):
    """D8: ``R=2`` at ``K=2`` adds 2 non-identifiable parameters and no fit.

    ``W W^T + diag(v)`` at ``R=1`` already spans every strictly-PD ``2x2``
    matrix (T15), so ``rank`` above 1 is pure over-parameterization.  Kept as an
    argument for ``K > 2``, not as a knob to turn here.
    """
    Xtr, _, Yn, K_xx, lam, U, _ = real_case
    out = {}
    for R in (1, 2):
        losses = []
        for seed in range(4):
            theta, _ = icm.fit(Xtr, Yn, n_steps=200, lr=0.1, rank=R, rng=seed,
                               K_xx=K_xx)
            loss, _, _ = icm.nmll_and_grad(theta, K_xx, Yn, lam=lam, U=U,
                                           rank=R, want_grad=False)
            losses.append(float(loss))
        out[R] = (min(losses), max(losses), float(np.mean(losses)))
    assert icm.n_params(2, 2) - icm.n_params(2, 1) == 2
    overlap = min(out[1][1], out[2][1]) - max(out[1][0], out[2][0])
    assert overlap > 0.0, (
        f"the rank-1 and rank-2 loss ranges do not overlap: {out}; rank 2 is "
        "not behaving as redundant")
    assert abs(out[1][2] - out[2][2]) <= 5e-3, (
        f"mean loss differs by {abs(out[1][2]-out[2][2]):.3e} between ranks")
    j = fixture("07_identifiability.json")["rank_redundancy_K2"]
    report(f"D8: rank 1 loss {out[1][0]:.7f}..{out[1][1]:.7f} (mean "
           f"{out[1][2]:.7f}) vs rank 2 {out[2][0]:.7f}..{out[2][1]:.7f} (mean "
           f"{out[2][2]:.7f}) over 4 seeds; ranges overlap by {overlap:.3e}; "
           f"torch reference means {j['1']['loss_mean']:.7f} / {j['2']['loss_mean']:.7f}")


def test_caching_the_eigendecomposition_does_not_change_the_answer(real_case):
    """SPEC §3.4 / §10.13: ``(lam, U)`` is cached, and caching is exact.

    The entire performance argument is that ``K_XX`` has no hyperparameters.  If
    passing the cache changed any number, the cache would be a bug rather than an
    optimization.
    """
    Xtr, _, Yn, K_xx, lam, U, theta = real_case
    cached = icm.nmll_and_grad(theta, K_xx, Yn, lam=lam, U=U, rank=RANK)
    fresh = icm.nmll_and_grad(theta, K_xx, Yn, rank=RANK)
    assert cached[0] == fresh[0], (
        f"loss differs with/without the cache: {cached[0]!r} vs {fresh[0]!r}")
    assert maxabs(cached[1], fresh[1]) == 0.0
    a, b = icm.fit(Xtr, Yn, n_steps=25, rank=RANK, rng=11, K_xx=K_xx)[0], \
        icm.fit(Xtr, Yn, n_steps=25, rank=RANK, rng=11)[0]
    assert maxabs(a, b) == 0.0, "the fit depends on whether the Gram was passed"
    report("eigendecomposition cache: loss, gradient and 25-step fit are all "
           "BITWISE identical with and without the cached (lam, U)")


# ===========================================================================
# T19 - T22 — edge cases.  NO oracle needed.
# ===========================================================================

def test_duplicate_fingerprints_stay_finite(library_fingerprints):
    """T19: a repeated row makes ``K_XX`` exactly singular; nothing may go NaN.

    Real libraries produce this after densification.  LAPACK can then return
    small negative eigenvalues, which is why :func:`icm.eig_data` clamps at 0 —
    without the clamp, ``S = lam*mu + 1`` drops below 1 and ``log(S)`` is NaN.
    """
    X = np.asarray(library_fingerprints[:12], dtype=np.int8).copy()
    X[5] = X[3]
    X[9] = X[3]
    K_xx = icm.gram(X, X)
    eig_raw = np.linalg.eigvalsh(K_xx)
    lam, U = icm.eig_data(K_xx)
    assert (lam >= 0.0).all(), "eig_data must clamp at 0"
    rng = np.random.default_rng(5)
    Yn = rng.standard_normal((X.shape[0], K_TASKS))
    theta = icm.init_theta(K_TASKS, RANK, rng=1)
    loss, grad, ex = icm.nmll_and_grad(theta, K_xx, Yn, lam=lam, U=U, rank=RANK)
    mean, vlat, vobs = icm.posterior(theta, X, Yn, X, rank=RANK, K_xx=K_xx,
                                     lam=lam, U=U)
    for name, a in (("loss", np.array([loss])), ("grad", grad), ("mean", mean),
                    ("var_latent", vlat), ("var_observed", vobs),
                    ("S", ex["S"]), ("alpha", ex["alpha"])):
        assert np.all(np.isfinite(a)), f"T19 {name} is not finite with duplicates"
    assert (ex["S"] >= 1.0).all()
    assert (vobs > 0.0).all(), "observed variance must stay positive"
    fitted, losses = icm.fit(X, Yn, n_steps=30, rank=RANK, rng=2, K_xx=K_xx)
    assert np.all(np.isfinite(fitted)) and np.all(np.isfinite(losses))
    report(f"T19 duplicates (rows 5, 9 := row 3): raw eig(K_XX) min "
           f"{eig_raw.min():.3e}, clamped min {lam.min():.3e}; loss "
           f"{loss:.6f}, gradient, posterior and a 30-step fit all finite")


def test_all_zero_fingerprint_gives_pure_noise_variance(library_fingerprints):
    """T20: a zero row has ``k(x, .) = 0``, so its observed variance is exactly ``d_t``.

    ``0 * B[t,t] - 0 + d_t = d_t``.  Not special-cased, and the ``eps = 1e-8``
    denominator clamp cannot blow up because the numerator is 0 too.
    """
    X = np.asarray(library_fingerprints[:10], dtype=np.int8).copy()
    Xq = X.copy()
    Xq[4] = 0
    K_xx = icm.gram(X, X)
    K_sx = icm.gram(Xq, X)
    assert maxabs(K_sx[4], np.zeros(X.shape[0])) == 0.0, "zero row must give a zero row"
    lam, U = icm.eig_data(K_xx)
    rng = np.random.default_rng(7)
    Yn = rng.standard_normal((X.shape[0], K_TASKS))
    theta = icm.init_theta(K_TASKS, RANK, rng=4)
    _, tau, s2, d, c = icm.natural(theta, K_TASKS, RANK)
    mean, vlat, vobs = icm.posterior(theta, X, Yn, Xq, rank=RANK, K_xx=K_xx,
                                     lam=lam, U=U)
    assert np.all(np.isfinite(vobs)) and np.all(np.isfinite(mean))
    d_lat = maxabs(vlat[4], np.zeros(K_TASKS))
    d_obs = maxabs(vobs[4], d)
    d_mean = maxabs(mean[4], c)
    assert d_lat <= 1e-12, f"T20 latent variance at a zero row: {d_lat:.3e}"
    assert d_obs <= 1e-12, f"T20 observed variance != d: {d_obs:.3e}"
    assert d_mean <= 1e-12, f"T20 mean at a zero row should be the prior c: {d_mean:.3e}"
    report(f"T20 zero fingerprint: latent var {d_lat:.3e} from 0, observed var "
           f"{d_obs:.3e} from d = tau + sigma2 = {d.tolist()}, mean {d_mean:.3e} "
           f"from the prior constant")


def test_zero_variance_target_column_is_guarded():
    """T21: ``y_std == 0 -> 1.0``; the column normalizes to 0 and reverses to its mean.

    Reachable in real runs: the first BO iteration can dock ``n_init`` molecules
    and get identical scores on one target.
    """
    N = 6
    Y = np.full((N, len(icm.TASK_NAMES)), np.nan, dtype=np.float32)
    Y[:, 0] = -0.42                                # CONSTANT column
    Y[:, 1] = np.linspace(-0.5, -0.3, N)
    y_mean, y_std, observed = icm.standardize(Y)
    assert y_std[0] == 1.0, f"the zero-variance guard did not fire: {y_std[0]!r}"
    assert abs(float(y_mean[0]) - (-0.42)) <= 1e-7
    Yn = icm._normalize(Y, y_mean, y_std, observed)
    assert np.all(np.isfinite(Yn))
    assert maxabs(Yn[:, 0], np.zeros(N)) <= 1e-7, "constant column must normalize to 0"
    back = Yn[:, 0] * y_std[0] + y_mean[0]
    assert maxabs(back, np.full(N, -0.42)) <= 1e-7
    report(f"T21: constant column -> y_std {y_std[0]!r}, normalizes to 0 "
           f"(max {np.abs(Yn[:,0]).max():.3e}), reverses to its mean "
           f"{float(y_mean[0]):.7f}")


def test_grey_box_layout_contract(library_fingerprints):
    """T22: ``y_mean``/``y_std`` are NaN off ``DOCKING_TASK_INDICES``, and so are
    the predictions.  ``acquisition.py`` and ``loop.py`` index a 5-wide array."""
    X = np.asarray(library_fingerprints[:14], dtype=np.int8)
    rng = np.random.default_rng(13)
    Y = np.empty((X.shape[0], len(icm.TASK_NAMES)), dtype=np.float32)
    Y[:, 0] = -0.4 + 0.05 * rng.standard_normal(X.shape[0])
    Y[:, 1] = -0.4 + 0.05 * rng.standard_normal(X.shape[0])
    for j in range(2, len(icm.TASK_NAMES)):        # FINITE ADMET columns
        Y[:, j] = rng.standard_normal(X.shape[0])
    state, lik, y_mean, y_std = icm.train_mogp_coregionalized(X, Y, n_iterations=20,
                                                              rng=0)
    assert lik is None, "slot 1 must be None; there is no likelihood object"
    dock = list(icm.DOCKING_TASK_INDICES)
    admet = [j for j in range(len(icm.TASK_NAMES)) if j not in dock]
    assert np.all(np.isfinite(y_mean[dock])) and np.all(np.isfinite(y_std[dock]))
    assert np.all(np.isnan(y_mean[admet])), (
        "ADMET stats must be NaN even when the columns are FINITE")
    assert np.all(np.isnan(y_std[admet]))
    assert list(state.observed) == [j in dock for j in range(len(icm.TASK_NAMES))]
    assert state.n_tasks == len(dock) == K_TASKS
    mean, var = icm.predict(state, y_mean, y_std, X)
    assert mean.shape == var.shape == (X.shape[0], len(icm.TASK_NAMES))
    assert np.all(np.isnan(mean[:, admet])) and np.all(np.isnan(var[:, admet]))
    assert np.all(np.isfinite(mean[:, dock])) and np.all(np.isfinite(var[:, dock]))
    assert (var[:, dock] > 0).all()
    # the drop-in spelling with the ignored likelihood slot
    m2, v2 = icm.predict_coregionalized(state, None, y_mean, y_std, X)
    assert maxabs(np.nan_to_num(m2), np.nan_to_num(mean)) == 0.0
    assert maxabs(np.nan_to_num(v2), np.nan_to_num(var)) == 0.0
    with pytest.raises(ValueError, match="no observed docking target"):
        icm.train_mogp_coregionalized(X, Y, n_iterations=2,
                                      docking_task_indices=())
    report(f"T22: 5-wide layout preserved; ADMET columns {admet} NaN in "
           f"y_mean/y_std and in both predictions; n_tasks = {state.n_tasks}")


def test_variance_denormalization_squares_the_scale(library_fingerprints):
    """``var_orig = var * y_std**2``.  A bare ``y_std`` is a silent ~20% error at
    ``y_std ~ 1.2`` that no smoke test would catch."""
    X = np.asarray(library_fingerprints[:12], dtype=np.int8)
    rng = np.random.default_rng(21)
    Y = np.full((X.shape[0], len(icm.TASK_NAMES)), np.nan, dtype=np.float32)
    Y[:, 0] = -8.0 + 1.2 * rng.standard_normal(X.shape[0])
    Y[:, 1] = -7.0 + 1.2 * rng.standard_normal(X.shape[0])
    state, _, y_mean, y_std = icm.train_mogp_coregionalized(X, Y, n_iterations=15,
                                                            rng=0)
    dock = list(icm.DOCKING_TASK_INDICES)
    mean, var = icm.predict(state, y_mean, y_std, X)
    mean_n, _, vobs_n = icm.posterior(state.theta, state.X_train, state.Y_norm, X,
                                      rank=state.rank, K_xx=state.gram())
    # y_std comes back float32 (standardize computes in float32, as the repo
    # does); predict casts to float64 BEFORE squaring, so the comparison has to
    # as well or it measures a 1.4e-08 float32 rounding difference and nothing else.
    ys = np.asarray(y_std, dtype=np.float64)[dock]
    ym = np.asarray(y_mean, dtype=np.float64)[dock]
    d_sq = maxabs(var[:, dock], vobs_n * ys ** 2)
    d_lin = maxabs(var[:, dock], vobs_n * ys)
    assert d_sq <= 1e-9, f"variance de-normalization is not squared: {d_sq:.3e}"
    assert d_lin > 1e-3, "the linear variant is not distinguishable here"
    d_mean = maxabs(mean[:, dock], mean_n * ys + ym)
    assert d_mean <= 1e-9, f"mean de-normalization: {d_mean:.3e}"
    report(f"de-normalization: |var - var_n*y_std^2| = {d_sq:.3e}; the bare "
           f"y_std variant is off by {d_lin:.3e} at y_std = "
           f"{ys.round(4).tolist()}")


def test_observation_noise_flag_adds_exactly_tau_plus_sigma2(library_fingerprints):
    """``observation_noise=True`` (the shipped default) adds ``d = tau + sigma2``.

    Flipping the default is the acquisition cluster's decision, not this
    module's, so the default is checked to still be ``True``.
    """
    X = np.asarray(library_fingerprints[:12], dtype=np.int8)
    rng = np.random.default_rng(31)
    Y = np.full((X.shape[0], len(icm.TASK_NAMES)), np.nan, dtype=np.float32)
    Y[:, 0] = -0.4 + 0.05 * rng.standard_normal(X.shape[0])
    Y[:, 1] = -0.4 + 0.05 * rng.standard_normal(X.shape[0])
    state, _, y_mean, y_std = icm.train_mogp_coregionalized(X, Y, n_iterations=25,
                                                             rng=0)
    dock = list(icm.DOCKING_TASK_INDICES)
    _, var_on = icm.predict(state, y_mean, y_std, X)
    _, var_off = icm.predict(state, y_mean, y_std, X, observation_noise=False)
    _, tau, s2, d, _ = state.natural()
    ys = np.asarray(y_std, dtype=np.float64)[dock]     # float64 BEFORE squaring
    gap = (var_on[:, dock] - var_off[:, dock]) / ys ** 2
    d_full = maxabs(gap, np.tile(d, (X.shape[0], 1)))
    d_tau = maxabs(gap, np.tile(tau, (X.shape[0], 1)))
    assert d_full <= 1e-9, f"the noise gap is not tau + sigma2: {d_full:.3e}"
    assert d_tau > 1e-6, "tau alone is not distinguishable from tau + sigma2 here"
    report(f"observation_noise: gap = d = tau + sigma2 to {d_full:.3e}; tau "
           f"alone would be off by {d_tau:.3e} (sigma2 = {s2:.6e})")


# ===========================================================================
# SPEC §8 — the pre-existing gap.  VERIFY, do not fix.
# ===========================================================================

def test_the_discarded_cross_task_structure_is_still_there_to_discard(real_case):
    """SPEC §8: measure what ``acquisition.py:323`` deletes.  Change nothing.

    The acquisition rebuilds the joint posterior from the marginals with
    ``torch.diag_embed``, so every cross-task and between-candidate covariance it
    sees is ``0.0``.  This test asserts the structure EXISTS in this module's
    posterior — it does not assert that anything downstream uses it.  Closing the
    gap is a deliberate scientific change the charter did not authorize.
    """
    Xtr, Xte, Yn, K_xx, lam, U, theta = real_case
    M = Xte.shape[0]
    cov = icm.posterior_covariance(theta, Xtr, Yn, Xte, rank=RANK, K_xx=K_xx,
                                   lam=lam, U=U)
    rho = np.array([cov[i*K_TASKS, i*K_TASKS+1]
                    / math.sqrt(cov[i*K_TASKS, i*K_TASKS]
                                * cov[i*K_TASKS+1, i*K_TASKS+1])
                    for i in range(M)])
    ref = fixture("05_posterior.json")["within_point_cross_task_correlation"]
    for name, mine, want in (("min", rho.min(), ref["min"]),
                             ("median", float(np.median(rho)), ref["median"]),
                             ("max", rho.max(), ref["max"])):
        assert abs(mine - want) <= 1e-9, (
            f"within-point cross-task correlation {name}: {mine!r} vs the "
            f"archived {want!r}")
    off = np.abs(cov - np.diag(np.diag(cov)))
    assert off.max() > 1e-3, "there is no off-diagonal structure to discard"

    def between_blocks(C):
        return np.array([np.abs(C[i*K_TASKS:(i+1)*K_TASKS,
                                 j*K_TASKS:(j+1)*K_TASKS]).max()
                         for i in range(M) for j in range(M) if i != j])

    # The archived §8 table was measured on a FITTED model (200 steps, lr=0.1,
    # rank=1), not at this fixture's random raw theta, so the between-candidate
    # magnitudes are only comparable when the model is fitted.  Do both, and
    # assert only against the archived within-point correlations, which the
    # fixture theta does pin.
    theta_fit, _ = icm.fit(Xtr, Yn, n_steps=200, lr=0.1, rank=RANK, rng=0,
                           K_xx=K_xx)
    cov_fit = icm.posterior_covariance(theta_fit, Xtr, Yn, Xte, rank=RANK,
                                       K_xx=K_xx, lam=lam, U=U)
    rho_fit = np.array([cov_fit[i*K_TASKS, i*K_TASKS+1]
                        / math.sqrt(cov_fit[i*K_TASKS, i*K_TASKS]
                                    * cov_fit[i*K_TASKS+1, i*K_TASKS+1])
                        for i in range(M)])
    b_fix, b_fit = between_blocks(cov), between_blocks(cov_fit)
    assert (rho_fit > 0.9).all(), (
        f"the fitted latent cross-task correlation should be ~0.994 "
        f"(SPEC §8), got {rho_fit.min():.5f}..{rho_fit.max():.5f}")
    report(f"SPEC §8 (verify only). At the fixture theta: within-point "
           f"cross-task correlation {rho.min():.5f}..{rho.max():.5f} (median "
           f"{np.median(rho):.5f}, matches the archive), between-candidate "
           f"blocks max {b_fix.max():.5f} median {np.median(b_fix):.5f}. At a "
           f"200-step fit (lr=0.1, rank=1): correlation {rho_fit.min():.5f}.."
           f"{rho_fit.max():.5f} median {np.median(rho_fit):.5f} (archive "
           f"0.99424..0.99519), between-candidate blocks max {b_fit.max():.5f} "
           f"median {np.median(b_fit):.5f} (archive 0.31714 / 0.03798). "
           f"acquisition.py:323 replaces every one of these with 0.0 and this "
           f"port PRESERVES that.")


# ===========================================================================
# Provenance — the fixtures really are the real library
# ===========================================================================

def test_the_real_fixtures_come_from_data_library(library_fingerprints):
    """Every fixture fingerprint row is bitwise a row of ``data/library/``.

    Matched back through ``smiles.csv``, so "real fingerprints from
    data/library" is a checked claim, not an assertion in a docstring.  If the
    library file changes, every real-data number above is void and this fails
    first.
    """
    import csv
    path = os.path.join(REPO, "data", "library", "smiles.csv")
    if not os.path.exists(path):
        pytest.skip("data/library/smiles.csv absent")
    with open(path, newline="") as fh:
        lib = [r["SMILES"] for r in csv.DictReader(fh)]
    pos = {}
    for i, s in enumerate(lib):
        pos.setdefault(s, i)
    with open(os.path.join(FIXTURES, "03_real_data", "smiles.txt")) as fh:
        sel = [ln.strip() for ln in fh if ln.strip()]
    j = fixture("03_real_data.json")
    assert hashlib.sha256("\n".join(sel).encode()).hexdigest() == j["smiles_sha256"]
    assert len(sel) == j["n_train"] + j["n_test"]
    missing = [s for s in sel if s not in pos]
    assert not missing, f"{len(missing)} fixture SMILES absent from the library"
    rows = np.asarray([np.asarray(library_fingerprints[pos[s]]) for s in sel],
                      dtype=np.int8)
    Xtr = arr("03_real_data/fingerprints_train.npy")
    Xte = arr("03_real_data/fingerprints_test.npy")
    assert np.array_equal(rows[:j["n_train"]], Xtr), (
        "the training fingerprints are NOT the library rows any more")
    assert np.array_equal(rows[j["n_train"]:], Xte)
    sums = Xtr.sum(1)
    assert [int(sums.min()), int(sums.max())] == j["train_fp_row_sums_min_max"]
    le = arr("03_real_data/le_train.npy")
    r = float(np.corrcoef(le.T)[0, 1])
    assert abs(r - j["pearson_r_le_train"]) <= 1e-12
    report(f"provenance: all {len(sel)} fixture fingerprints are bitwise rows of "
           f"data/library/fingerprints.npy (shape {tuple(library_fingerprints.shape)}); "
           f"real Vina ligand-efficiency Pearson r = {r:.7f}")


def test_gram_diagonal_is_exactly_one_on_real_fingerprints(real_case):
    """``k(x, x) == 1`` EXACTLY for binary fingerprints, so
    ``diag(Sigma) = B[t,t] + d_t``.

    Which is SPEC §10.4: ``B[t,t]``, ``tau_t`` and ``sigma2`` are only jointly
    identified, and that is why ``B``'s diagonal scatters 3-4x more across seeds
    than ``B01`` does.  The ``eps`` clamp must not turn this into ``0.99999999``.
    """
    Xtr, _, Yn, K_xx, lam, U, theta = real_case
    assert maxabs(np.diag(K_xx), np.ones(K_xx.shape[0])) == 0.0, (
        "k(x,x) != 1 exactly; the eps became additive instead of a clamp")
    assert maxabs(K_xx, K_xx.T) == 0.0, "the Gram is not exactly symmetric"
    B, tau, s2, d, _ = icm.natural(theta, K_TASKS, RANK)
    Sigma = icm.dense_covariance(K_xx, B, tau, s2)
    want = np.tile(np.diag(B) + d, K_xx.shape[0])
    assert maxabs(np.diag(Sigma), want) <= 1e-12
    report("Gram: diag(K_XX) == 1 bitwise and diag(Sigma) == B[t,t] + d_t "
           "exactly, so B's diagonal is only jointly identified with the noises")
