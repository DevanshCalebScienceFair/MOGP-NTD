"""Coregionalized (ICM) multi-output Gaussian Process for molecular objectives.

This is an *alternative* to the batch-independent ``mogp.MOGPModel``. Both are
kept side by side so they can be compared in an ablation:

  * ``mogp.MOGPModel`` places one **independent** scaled-Tanimoto GP on each
    objective. Cross-task structure is block-diagonal: learning about one
    objective tells the model nothing about the others.
  * ``MOGPCoregionalized`` (this file) is an **Intrinsic Coregionalization Model
    (ICM)**. It shares a single Tanimoto data kernel across objectives and adds a
    learned, *dense* ``K x K`` task-covariance matrix (via ``IndexKernel``), so
    correlated objectives borrow statistical strength from one another.

Model construction mirrors GPyTorch's standard multitask ICM recipe:

  * data covariance:  ``TanimotoKernel`` over fingerprints
  * task covariance:  ``IndexKernel(num_tasks=K, rank=R)`` -> dense ``K x K``
  * combined:         ``MultitaskKernel(TanimotoKernel(), num_tasks=K, rank=R)``
  * mean:             ``MultitaskMean(ConstantMean(), num_tasks=K)``
  * likelihood:       ``MultitaskGaussianLikelihood(num_tasks=K)``

``train_mogp_coregionalized`` and ``predict_coregionalized`` deliberately expose
the **same signatures and return shapes** as ``mogp.train_mogp`` /
``mogp.predict`` so ``acquisition.py`` and ``loop.py`` can swap models with a
one-line change.

Unlike ``mogp.train_mogp``, this model is trained only on **fully-observed**
molecules (every objective present). That is always the case for docked
molecules inside the BO loop, so no NaN masking is done here; all ``K`` task
columns are trained and every returned column is finite.

Objective order follows ``mogp.TASK_NAMES`` (the single source of truth).

Run ``python mogp_coregionalized.py`` for a self-test that trains on ~30
molecules and confirms the learned task-covariance matrix is non-diagonal.
"""

import gpytorch
import numpy as np
import torch

from kernel import TanimotoKernel
from mogp import TASK_NAMES  # re-exported so callers importing from either module agree


class MOGPCoregionalized(gpytorch.models.ExactGP):
    """Intrinsic Coregionalization Model (ICM) exact GP with a Tanimoto kernel.

    A single Tanimoto kernel over fingerprints is shared across all objectives;
    a learned ``IndexKernel`` supplies a dense ``K x K`` task covariance. The
    Kronecker structure of ``MultitaskKernel`` combines the two, and the model
    emits a single ``MultitaskMultivariateNormal`` over all objectives jointly.

    Args:
        train_x: Fingerprint tensor of shape ``(N, 2048)``, float32.
        train_y: Target tensor of shape ``(N, K)``, float32 (fully observed).
            ``K`` (number of tasks) is inferred from ``train_y.shape[1]``.
        likelihood: A ``MultitaskGaussianLikelihood`` with matching ``num_tasks``.
        rank: Rank ``R`` of the ``IndexKernel`` low-rank task-covariance factor.
    """

    def __init__(self, train_x, train_y, likelihood, rank=2):
        super().__init__(train_x, train_y, likelihood)
        num_tasks = train_y.shape[1]

        # Shared constant mean per task.
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(),
            num_tasks=num_tasks,
        )
        # Kronecker of a shared data kernel (Tanimoto over fingerprints) and a
        # dense KxK task covariance (IndexKernel). This is what distinguishes the
        # ICM from the batch-independent block-diagonal MOGPModel: the off-block
        # (cross-task) terms are learned rather than forced to zero.
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            TanimotoKernel(),
            num_tasks=num_tasks,
            rank=rank,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)        # (N, K)
        covar_x = self.covar_module(x)      # (N*K, N*K) lazy Kronecker
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

    def task_covariance_matrix(self):
        """Return the learned dense ``K x K`` task-covariance matrix.

        Read straight off the ``IndexKernel`` inside the ``MultitaskKernel``:
        ``B B^T + diag(v)`` where ``B`` is the ``K x R`` factor. A non-diagonal
        result means the model has captured cross-objective correlation.
        """
        index_kernel = self.covar_module.task_covar_module
        return index_kernel._eval_covar_matrix().detach().cpu().numpy()


def train_mogp_coregionalized(train_x, train_y, n_iterations=200, lr=0.1, rank=2):
    """Train the coregionalized MOGP on fingerprints and normalized targets.

    Same signature/return contract as ``mogp.train_mogp`` (plus a ``rank`` knob),
    so callers can swap models with a one-line change.

    Args:
        train_x: Fingerprint matrix of shape ``(N, 2048)``.
        train_y: Target matrix of shape ``(N, K)``, float32, columns in
            ``TASK_NAMES`` order. Must be fully observed (no NaNs): this model
            is only used on docked molecules, where every objective is present.
        n_iterations: Number of Adam steps.
        lr: Adam learning rate.
        rank: Rank of the ``IndexKernel`` task-covariance factor (default 2).

    Returns:
        A tuple ``(model, likelihood, y_mean, y_std)`` where ``y_mean`` and
        ``y_std`` are numpy arrays of shape ``(K,)`` used to reverse the
        per-column target normalization at prediction time.
    """
    train_x_t = torch.from_numpy(np.asarray(train_x)).to(torch.float32)
    train_y = np.asarray(train_y, dtype=np.float32)

    if not np.isfinite(train_y).all():
        raise ValueError(
            "train_mogp_coregionalized requires fully-observed targets "
            "(no NaNs); this model trains only on molecules with every "
            "objective present."
        )

    # Per-column standardization. Every column is observed here, so all stats are
    # finite (contrast with mogp.train_mogp, which skips all-NaN columns).
    y_mean = train_y.mean(axis=0)
    y_std = train_y.std(axis=0)
    y_std = np.where(y_std == 0.0, 1.0, y_std)  # guard constant columns

    train_y_norm = (train_y - y_mean) / y_std
    train_y_t = torch.from_numpy(train_y_norm).to(torch.float32)

    num_tasks = train_y.shape[1]
    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=num_tasks)
    model = MOGPCoregionalized(train_x_t, train_y_t, likelihood, rank=rank)

    model.train()
    likelihood.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(n_iterations):
        optimizer.zero_grad()
        output = model(train_x_t)
        loss = -mll(output, train_y_t)
        loss.backward()
        optimizer.step()
        if (i + 1) % 20 == 0:
            print(f"Iter {i + 1:>4}/{n_iterations} - loss: {loss.item():.4f}")

    return model, likelihood, y_mean, y_std


def predict_coregionalized(model, likelihood, y_mean, y_std, X_new):
    """Predict all objectives and per-task uncertainty for new molecules.

    Same signature/return contract as ``mogp.predict``.

    Args:
        model: A trained ``MOGPCoregionalized``.
        likelihood: The matching ``MultitaskGaussianLikelihood``.
        y_mean: Normalization means, shape ``(K,)``.
        y_std: Normalization stds, shape ``(K,)``.
        X_new: Fingerprint matrix of shape ``(M, 2048)``.

    Returns:
        A tuple ``(mean, variance)`` of numpy arrays, each shape ``(M, K)``, with
        columns in ``TASK_NAMES`` order on the original (de-normalized) scale.
        ``variance`` is the per-objective (marginal) predictive variance.
    """
    X_new_t = torch.from_numpy(np.asarray(X_new)).to(torch.float32)

    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = likelihood(model(X_new_t))
        mean_norm = posterior.mean.cpu().numpy()        # (M, K)
        variance_norm = posterior.variance.cpu().numpy()  # (M, K)

    # Reverse the per-column standardization. Every column is trained here.
    mean = mean_norm * y_std + y_mean
    variance = variance_norm * (y_std ** 2)
    return mean, variance


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Self-test: train on ~30 fully-observed molecules and confirm the learned
    # task-covariance matrix is NON-diagonal, i.e. the ICM actually captures
    # cross-task correlation (the whole point vs. the independent MOGPModel).
    # ------------------------------------------------------------------
    N_MOL = 30
    rng = np.random.default_rng(0)

    try:
        from data import load_library

        lib = load_library()
        n = min(N_MOL, len(lib["smiles"]))
        train_x = lib["fingerprints"][:n].astype(np.float32)
        admet = lib["admet_scores"][:n].astype(np.float32)  # (n, 3), real ADMET
        print(f"Loaded {n} molecules from data/library for the self-test.")

        # The docking objective isn't computed up front, so synthesize a 4th
        # column that is genuinely correlated with the ADMET objectives. This
        # gives the ICM real cross-task structure to recover (K = 4, matching
        # the loop's objective count).
        z = (admet - admet.mean(0)) / (admet.std(0) + 1e-8)
        docking = -8.0 + 1.5 * z[:, 0] - 1.0 * z[:, 2] + 0.3 * rng.standard_normal(n)
        Y = np.column_stack([admet, docking.astype(np.float32)])
    except (FileNotFoundError, ImportError) as exc:
        # Fallback so the self-test still runs without a built library: random
        # binary fingerprints and correlated targets.
        print(f"Library unavailable ({exc}); using synthetic data for self-test.")
        n = N_MOL
        train_x = (rng.random((n, 2048)) < 0.02).astype(np.float32)
        base = rng.standard_normal((n, 2))
        Y = np.column_stack([
            base[:, 0],
            0.8 * base[:, 0] + 0.2 * rng.standard_normal(n),   # correlated w/ col 0
            base[:, 1],
            -0.9 * base[:, 0] + 0.1 * rng.standard_normal(n),  # anti-correlated w/ col 0
        ]).astype(np.float32)

    K = Y.shape[1]
    print(f"\nTraining coregionalized MOGP (ICM) on {n} molecules, K={K} tasks...")
    model, likelihood, y_mean, y_std = train_mogp_coregionalized(
        train_x, Y, n_iterations=200, rank=2
    )

    B = model.task_covariance_matrix()  # (K, K)
    task_labels = TASK_NAMES[:K]

    print("\nLearned task-covariance matrix (from IndexKernel, K x K):")
    print("             " + "".join(f"{l[:10]:>12}" for l in task_labels))
    for i, li in enumerate(task_labels):
        row = "".join(f"{B[i, j]:12.4f}" for j in range(K))
        print(f"{li[:12]:>12} {row}")

    # Convert to a correlation matrix so "non-diagonal" is scale-free.
    d = np.sqrt(np.diag(B))
    corr = B / np.outer(d, d)
    off_diag = corr - np.diag(np.diag(corr))
    max_off = np.abs(off_diag).max()

    print(f"\nMax |off-diagonal task correlation| = {max_off:.4f}")
    assert max_off > 1e-3, (
        "Task covariance is (near-)diagonal; the ICM did not capture any "
        "cross-task correlation."
    )
    print(
        "PASS: task covariance is non-diagonal -> the ICM captures cross-task "
        "correlation (unlike the independent MOGPModel)."
    )
