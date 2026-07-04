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
molecules, prints the learned ``K x K`` task-covariance matrix, and confirms the
strongest off-diagonal term is the PfDHFR/hDHFR pair — the biological
correlation the ICM is meant to exploit: the two dihydrofolate reductases are
homologous, so a molecule that binds one tends to bind the other, which is
exactly why selectivity is hard and why sharing statistical strength across the
two docking tasks helps.
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
    # Self-test: train on ~30 fully-observed molecules over the real objective
    # set TASK_NAMES = [PfDHFR_Docking, hDHFR_Docking, hERG_Toxicity_Prob,
    # Caco2_logPapp, Half_Life_hours], print the learned K x K task covariance,
    # and confirm the STRONGEST off-diagonal term is the PfDHFR/hDHFR pair (the
    # ADMET tasks are independent axes). The two dihydrofolate reductases are
    # homologous enzymes, so raw docking scores against them co-vary strongly
    # (good binders bind both) — that positive coupling is the biological
    # correlation the ICM borrows strength from, and is exactly why selectivity
    # (a *weak* human hit alongside a strong parasite hit) is hard.
    # ------------------------------------------------------------------
    N_MOL = 30
    rng = np.random.default_rng(0)

    # The two docking-objective columns, in TASK_NAMES order. These are the pair
    # whose learned task covariance the self-test checks; every other objective
    # is filled as an independent axis inside _build_targets.
    PF, HD = (TASK_NAMES.index("PfDHFR_Docking"),
              TASK_NAMES.index("hDHFR_Docking"))

    def _build_targets(latent, admet, admet_columns, n):
        """PfDHFR & hDHFR share a dominant chemical latent (homologous targets)
        -> strongly correlated raw docking scores. Every OTHER objective is an
        independent axis (its real ADMET value when the library is available,
        else independent noise), so the PfDHFR/hDHFR pair stays the strongest
        off-diagonal task correlation the ICM should recover across all K tasks."""
        Y = np.empty((n, len(TASK_NAMES)), dtype=np.float32)
        noise = rng.standard_normal((n, 2))
        Y[:, PF] = -8.0 + 2.0 * latent + 0.25 * noise[:, 0]   # parasite docking
        Y[:, HD] = -8.0 + 1.9 * latent + 0.25 * noise[:, 1]   # human docking
        # Fill the remaining (non-docking) objective columns as independent axes.
        for j, name in enumerate(TASK_NAMES):
            if j in (PF, HD):
                continue
            if admet is not None and name in admet_columns:
                Y[:, j] = admet[:, admet_columns.index(name)]
            else:
                Y[:, j] = rng.standard_normal(n)
        return Y

    try:
        from data import load_library, ADMET_COLUMNS

        lib = load_library()
        n = min(N_MOL, len(lib["smiles"]))
        train_x = lib["fingerprints"][:n].astype(np.float32)
        admet = lib["admet_scores"][:n].astype(np.float32)   # (n, len(ADMET_COLUMNS))
        print(f"Loaded {n} molecules from data/library for the self-test.")

        # Dominant chemical latent grounded in fingerprint structure but
        # INDEPENDENT of the ADMET objective columns: a fixed random projection
        # of the Morgan fingerprints. It drives BOTH docking tasks so the
        # Tanimoto data kernel can model them and the shared structure lands in
        # the task covariance, while the three real ADMET objectives stay
        # separate axes uncorrelated with that latent.
        w = rng.standard_normal(train_x.shape[1]).astype(np.float32)
        proj = train_x @ w
        latent = (proj - proj.mean()) / (proj.std() + 1e-8)
        Y = _build_targets(latent, admet, ADMET_COLUMNS, n)
    except (FileNotFoundError, ImportError) as exc:
        # Fallback so the self-test still runs without a built library.
        print(f"Library unavailable ({exc}); using synthetic data for self-test.")
        n = N_MOL
        train_x = (rng.random((n, 2048)) < 0.02).astype(np.float32)
        w = rng.standard_normal(train_x.shape[1]).astype(np.float32)
        proj = train_x @ w
        latent = (proj - proj.mean()) / (proj.std() + 1e-8)
        Y = _build_targets(latent, None, None, n)

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

    # Scale-free correlation matrix so cross-task strength is comparable.
    d = np.sqrt(np.diag(B))
    corr = B / np.outer(d, d)

    print("\nTask CORRELATION matrix:")
    print("             " + "".join(f"{l[:10]:>12}" for l in task_labels))
    for i, li in enumerate(task_labels):
        row = "".join(f"{corr[i, j]:12.4f}" for j in range(K))
        print(f"{li[:12]:>12} {row}")

    # Rank every unordered off-diagonal task pair by |correlation|.
    pairs = [((a, b), abs(corr[a, b]))
             for a in range(K) for b in range(a + 1, K)]
    pairs.sort(key=lambda kv: kv[1], reverse=True)

    print("\nOff-diagonal task correlations, strongest first:")
    for (a, b), mag in pairs:
        print(f"  {task_labels[a]:>18} <-> {task_labels[b]:<18} "
              f"corr={corr[a, b]:+.4f}")

    max_off = pairs[0][1]
    assert max_off > 1e-3, (
        "Task covariance is (near-)diagonal; the ICM did not capture any "
        "cross-task correlation."
    )

    # The headline biological check: PfDHFR/hDHFR must be the STRONGEST pair.
    strongest_pair = set(pairs[0][0])
    pf_hd_pair = {PF, HD}
    assert strongest_pair == pf_hd_pair, (
        "Expected PfDHFR/hDHFR to be the strongest off-diagonal task "
        f"correlation, but the strongest pair was "
        f"{task_labels[pairs[0][0][0]]} <-> {task_labels[pairs[0][0][1]]}."
    )
    # Homologous targets -> raw docking scores co-vary POSITIVELY.
    assert corr[PF, HD] > 0.0, (
        "PfDHFR/hDHFR correlation should be positive (homologous targets: good "
        f"binders bind both), got {corr[PF, HD]:+.4f}."
    )
    print(
        f"\nPASS: the PfDHFR/hDHFR pair is the strongest off-diagonal term "
        f"(corr={corr[PF, HD]:+.4f}) -> the ICM captures the homologous-target "
        "correlation the independent MOGPModel forces to zero."
    )
