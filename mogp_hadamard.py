"""Coregionalized (ICM) GP that tolerates MISSING task labels.

Why this file exists
--------------------
``mogp_coregionalized.MOGPCoregionalized`` builds its covariance with
``MultitaskKernel``, whose Kronecker structure requires a *complete* ``(N, K)``
target matrix: every molecule must carry a value for every task. Under that
design the ICM provably cannot beat independent GPs in the posterior mean --
tasks observed at identical inputs make inter-task transfer cancel
(autokrigeability; Bonilla, Chai & Williams, *Multi-task Gaussian Process
Prediction*, NeurIPS 20, 2008, sec 2.3). We measured exactly that: across 10
paired seeds the ICM led 197/400 matched checkpoints, a coin flip, with every
endpoint null (``MULTISEED_ICM_VERDICT.md``).

The fix the theory names is to stop observing every task on every molecule. This
module implements the **Hadamard** (a.k.a. stacked, or "index") form of the same
ICM, in which each *observation* is a ``(molecule, task)`` pair rather than a
row of a complete matrix:

    inputs   x_1 ... x_n   with task indices  i_1 ... i_n
    kernel   k((x, i), (x', i')) = k_Tanimoto(x, x') * B[i, i']

``B`` is the same learned dense ``K x K`` task covariance (``IndexKernel``) the
Kronecker model uses, so this is not a different model -- it is the same ICM
written so that an arbitrary observation pattern is expressible. A molecule
docked against PfDHFR but not hDHFR contributes one row instead of two, and the
GP borrows across tasks to fill the gap. That is the regime where
coregionalization has something to do.

Per-task noise
--------------
The Kronecker model uses ``MultitaskGaussianLikelihood`` (one noise per task).
In Hadamard form the observations are a flat vector, so a plain
``GaussianLikelihood`` gives a single shared noise. Targets are standardized
per task before fitting, which puts both tasks on unit variance and makes a
shared noise defensible; it is nonetheless a real difference from the Kronecker
model and is stated in any comparison.
"""

import gpytorch
import numpy as np
import torch

from kernel import TanimotoKernel
from mogp import TASK_NAMES, DOCKING_TASK_INDICES

_FDTYPE = torch.float32


class MOGPHadamardICM(gpytorch.models.ExactGP):
    """ICM over ``(molecule, task)`` pairs, so labels may be missing per task.

    Args:
        train_x: Fingerprints of the OBSERVATIONS, shape ``(n_obs, 2048)``. A
            molecule observed on two tasks appears twice.
        train_i: Task index per observation, shape ``(n_obs, 1)``, long.
        train_y: Observed values, shape ``(n_obs,)``, already standardized.
        likelihood: A ``GaussianLikelihood``.
        num_tasks: Number of tasks ``K`` the IndexKernel spans.
        rank: Rank ``R`` of the ``K x R`` task-covariance factor.
    """

    def __init__(self, train_x, train_i, train_y, likelihood, num_tasks, rank=1):
        super().__init__((train_x, train_i), train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(TanimotoKernel())
        self.task_covar_module = gpytorch.kernels.IndexKernel(
            num_tasks=num_tasks, rank=rank
        )

    def forward(self, x, i):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)          # data covariance
        covar_i = self.task_covar_module(i)     # task covariance B[i, i']
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x.mul(covar_i))

    def task_covariance_matrix(self):
        """The learned dense ``K x K`` task covariance ``B B^T + diag(v)``."""
        return self.task_covar_module._eval_covar_matrix().detach().cpu().numpy()


def _stack_observations(train_y_norm, task_positions):
    """Flatten a ``(N, K)`` matrix WITH NaNs into observation triples.

    Returns ``(row_index, task_index, value)`` arrays covering only the finite
    entries, which is what lets an arbitrary missing pattern be expressed.
    """
    rows, cols = np.nonzero(np.isfinite(train_y_norm))
    return rows, np.asarray(task_positions)[cols], train_y_norm[rows, cols]


def train_mogp_hadamard(train_x, train_y, n_iterations=200, lr=0.1, rank=1,
                        verbose=True):
    """Train the missing-data ICM over the docking objectives.

    Mirrors ``mogp_coregionalized.train_mogp_coregionalized``'s signature and
    return contract, with one deliberate difference: **NaN entries in the
    docking columns of ``train_y`` are allowed** and are simply not observed.

    Args:
        train_x: Fingerprint matrix ``(N, 2048)``.
        train_y: Targets ``(N, len(TASK_NAMES))``. Docking columns may contain
            NaN; non-docking columns are ignored (grey-box).
        n_iterations, lr, rank: As for the Kronecker ICM.
        verbose: Print the loss every 20 steps.

    Returns:
        ``(model, likelihood, y_mean, y_std)``; ``y_mean`` / ``y_std`` are full
        ``len(TASK_NAMES)`` vectors with NaN outside the docking columns.

    Raises:
        ValueError: If a docking task has no observations at all, or if any
            molecule has no observation on any task.
    """
    train_y = np.asarray(train_y, dtype=np.float32)
    docking = [j for j in DOCKING_TASK_INDICES if j < train_y.shape[1]]

    # Standardize per task using only the OBSERVED entries of that task.
    y_mean = np.full(train_y.shape[1], np.nan, dtype=np.float64)
    y_std = np.full(train_y.shape[1], np.nan, dtype=np.float64)
    for j in docking:
        col = train_y[:, j]
        obs = np.isfinite(col)
        if not obs.any():
            raise ValueError(
                f"train_mogp_hadamard: task {TASK_NAMES[j]!r} has no observed "
                "values; the ICM needs at least one observation per task."
            )
        y_mean[j] = col[obs].mean()
        s = col[obs].std()
        y_std[j] = 1.0 if s == 0.0 else s

    norm = np.full((train_y.shape[0], len(docking)), np.nan, dtype=np.float64)
    for p, j in enumerate(docking):
        norm[:, p] = (train_y[:, j] - y_mean[j]) / y_std[j]

    keep = np.isfinite(norm).any(axis=1)
    if not keep.all():
        norm = norm[keep]
    if norm.shape[0] == 0:
        raise ValueError("train_mogp_hadamard: no molecule has any observed task.")

    x_keep = np.asarray(train_x)[keep]
    rows, task_pos, values = _stack_observations(norm, range(len(docking)))

    tx = torch.from_numpy(x_keep[rows]).to(_FDTYPE)
    ti = torch.from_numpy(task_pos.astype(np.int64)).reshape(-1, 1)
    ty = torch.from_numpy(values.astype(np.float32))

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = MOGPHadamardICM(tx, ti, ty, likelihood, num_tasks=len(docking), rank=rank)

    model.train(); likelihood.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for step in range(n_iterations):
        opt.zero_grad()
        loss = -mll(model(tx, ti), ty)
        loss.backward()
        opt.step()
        if verbose and (step + 1) % 20 == 0:
            print(f"Iter {step + 1:>4}/{n_iterations} - loss: {loss.item():.4f}")

    return model, likelihood, y_mean, y_std


def predict_hadamard(model, likelihood, y_mean, y_std, X_new):
    """Per-task posterior mean and variance for new molecules, de-normalized.

    Returns ``(mean, variance)`` of shape ``(n, len(TASK_NAMES))`` with NaN in
    every non-docking column, matching ``mogp.predict``'s contract. Variance is
    returned on the ORIGINAL scale (multiplied by ``y_std ** 2``).
    """
    docking = [j for j in DOCKING_TASK_INDICES if j < len(y_mean)]
    X = torch.from_numpy(np.asarray(X_new)).to(_FDTYPE)
    n = X.shape[0]

    model.eval(); likelihood.eval()
    mean = np.full((n, len(y_mean)), np.nan, dtype=float)
    var = np.full((n, len(y_mean)), np.nan, dtype=float)
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for p, j in enumerate(docking):
            idx = torch.full((n, 1), p, dtype=torch.long)
            post = likelihood(model(X, idx))
            mean[:, j] = post.mean.numpy() * y_std[j] + y_mean[j]
            var[:, j] = post.variance.numpy() * (y_std[j] ** 2)
    return mean, var


def predict_joint_hadamard(model, likelihood, y_mean, y_std, X_new,
                           task_indices=None):
    """Joint posterior over every ``(molecule, task)`` pair, INTERLEAVED.

    Returns ``(mean, cov)`` where ``mean`` has shape ``(n, len(TASK_NAMES))``
    (NaN outside docking) and ``cov`` is ``(n*t, n*t)`` over the ``t`` requested
    tasks, laid out with the TASK index varying fastest -- flat slot
    ``i * t + a`` -- matching ``mogp.predict_joint`` and what
    ``MultitaskMultivariateNormal(interleaved=True)`` expects.
    """
    docking = [j for j in DOCKING_TASK_INDICES if j < len(y_mean)]
    if task_indices is None:
        task_indices = docking
    task_indices = list(task_indices)
    pos = {j: p for p, j in enumerate(docking)}

    X = torch.from_numpy(np.asarray(X_new)).to(_FDTYPE)
    n, t = X.shape[0], len(task_indices)

    # Interleaved stacking: molecule i, task a -> row i*t + a.
    xr = X.repeat_interleave(t, dim=0)
    ir = torch.tensor([pos[j] for j in task_indices], dtype=torch.long).repeat(n)
    ir = ir.reshape(-1, 1)

    model.eval(); likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        post = likelihood(model(xr, ir))
        m = post.mean.numpy()
        c = post.covariance_matrix.numpy()

    scale = np.array([y_std[j] for j in task_indices], dtype=float)
    shift = np.array([y_mean[j] for j in task_indices], dtype=float)
    s_full = np.tile(scale, n)
    cov = c * s_full[:, None] * s_full[None, :]

    mean = np.full((n, len(y_mean)), np.nan, dtype=float)
    m2 = m.reshape(n, t) * scale[None, :] + shift[None, :]
    for a, j in enumerate(task_indices):
        mean[:, j] = m2[:, a]
    return mean, cov
