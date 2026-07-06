"""Coregionalized (ICM) multi-output Gaussian Process over the docking objectives.

This is the project's **primary** GP model — a *correlated* multi-output GP —
and the intended default for the headline benchmark. It sits on top of the
grey-box setup (``mogp.py`` / ``acquisition.py``): only the **docking**
objectives (``mogp.DOCKING_TASK_INDICES``) are modelled; the three ADMET
objectives are known exactly and folded into the acquisition's composite
objective, never predicted. So this is a **2-task ICM** over
``PfDHFR_Docking`` / ``hDHFR_Docking``.

It is kept side by side with the batch-independent ``mogp.MOGPModel`` for an
ablation:

  * ``mogp.MOGPModel`` places one **independent** scaled-Tanimoto GP on each
    modelled task. Cross-task structure is block-diagonal: learning about one
    docking target tells the model nothing about the other.
  * ``MOGPCoregionalized`` (this file) is an **Intrinsic Coregionalization Model
    (ICM)**. It shares a single Tanimoto data kernel across the docking tasks and
    adds a learned, *dense* task-covariance matrix (via ``IndexKernel``), so the
    two correlated docking objectives borrow statistical strength from each other.

The two dihydrofolate reductases (PfDHFR / hDHFR) are homologous enzymes, so raw
docking scores against them co-vary strongly (good binders bind both). That
positive coupling is exactly the biological correlation the ICM exploits, and is
why the *selective* (strong-PfDHFR / weak-hDHFR) corner of the Pareto front is
hard to reach with an independent model.

Model construction mirrors GPyTorch's standard multitask ICM recipe:

  * data covariance:  ``TanimotoKernel`` over fingerprints
  * task covariance:  ``IndexKernel(num_tasks=K, rank=R)`` -> dense ``K x K``
  * combined:         ``MultitaskKernel(TanimotoKernel(), num_tasks=K, rank=R)``
  * mean:             ``MultitaskMean(ConstantMean(), num_tasks=K)``
  * likelihood:       ``MultitaskGaussianLikelihood(num_tasks=K)``  (per-task noise)

``train_mogp_coregionalized`` and ``predict_coregionalized`` deliberately expose
the **same signatures and return contract** as ``mogp.train_mogp`` /
``mogp.predict`` — including the grey-box behaviour that ``y_mean`` / ``y_std``
are full ``len(TASK_NAMES)`` vectors whose non-docking entries are NaN, and that
predictions come back in the full ``TASK_NAMES`` layout with NaN ADMET columns —
so ``acquisition.py`` and ``loop.py`` treat it as a drop-in for the independent
model (``loop.BOLoop(train_fn=...)`` / ``--model coregionalized``).

Run ``python mogp_coregionalized.py`` for a self-test that fits the 2-task ICM on
a small set where the docking objectives are correlated, prints the learned
``2 x 2`` task-covariance matrix, and confirms it is NOT diagonal.
"""

import gpytorch
import numpy as np
import torch

from kernel import TanimotoKernel
# TASK_NAMES + the grey-box docking-task indices are the single source of truth;
# re-exported so callers importing from either module agree on the layout.
from mogp import TASK_NAMES, DOCKING_TASK_INDICES


class MOGPCoregionalized(gpytorch.models.ExactGP):
    """Intrinsic Coregionalization Model (ICM) exact GP with a Tanimoto kernel.

    A single Tanimoto kernel over fingerprints is shared across the modelled
    (docking) tasks; a learned ``IndexKernel`` supplies a dense ``K x K`` task
    covariance. The Kronecker structure of ``MultitaskKernel`` combines the two,
    and the model emits a single ``MultitaskMultivariateNormal`` over all
    modelled tasks jointly.

    Args:
        train_x: Fingerprint tensor of shape ``(N, 2048)``, float32.
        train_y: Target tensor of shape ``(N, K)``, float32 (fully observed over
            the ``K`` MODELLED tasks). ``K`` is inferred from ``train_y.shape[1]``.
        likelihood: A ``MultitaskGaussianLikelihood`` with matching ``num_tasks``.
        rank: Rank ``R`` of the ``IndexKernel`` low-rank task-covariance factor.
    """

    def __init__(self, train_x, train_y, likelihood, rank=1):
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
        result means the model has captured cross-objective correlation (for the
        grey-box docking pair, the PfDHFR/hDHFR coupling).
        """
        index_kernel = self.covar_module.task_covar_module
        return index_kernel._eval_covar_matrix().detach().cpu().numpy()


def train_mogp_coregionalized(train_x, train_y, n_iterations=200, lr=0.1, rank=1):
    """Train the coregionalized (ICM) GP over the docking objectives.

    Same signature and return contract as ``mogp.train_mogp`` (plus a ``rank``
    knob), so callers swap models with a one-line change / a ``--model`` flag.
    Grey-box: only the docking tasks (``DOCKING_TASK_INDICES``) are modelled; the
    three ADMET columns are masked out (their ``y_mean`` / ``y_std`` stay NaN),
    exactly as ``mogp.train_mogp`` does, so the ICM is a 2-task model.

    Args:
        train_x: Fingerprint matrix of shape ``(N, 2048)``.
        train_y: Target matrix of shape ``(N, len(TASK_NAMES))``, float32, columns
            in ``TASK_NAMES`` order. Only the docking columns need be finite; the
            ADMET columns are ignored.
        n_iterations: Number of Adam steps.
        lr: Adam learning rate.
        rank: Rank of the ``IndexKernel`` task-covariance factor (default 1).

    Returns:
        A tuple ``(model, likelihood, y_mean, y_std)`` where ``y_mean`` / ``y_std``
        are numpy arrays of shape ``(len(TASK_NAMES),)`` used to reverse the
        per-column target normalization at prediction time. Only the docking
        columns are trained (so the model's trained-task count is
        ``len(DOCKING_TASK_INDICES)``, i.e. 2); every other column's stats are NaN.
    """
    train_x_t = torch.from_numpy(np.asarray(train_x)).to(torch.float32)
    train_y = np.asarray(train_y, dtype=np.float32)

    # Per-column standardization stats over the full task set. Guard zero-variance
    # columns (constant -> normalizes to 0, reverses to its mean).
    y_mean = train_y.mean(axis=0)
    y_std = train_y.std(axis=0)
    y_std = np.where(y_std == 0.0, 1.0, y_std)

    # Grey-box: model ONLY the docking objectives. Mask every non-docking column's
    # normalization stats to NaN so they are excluded from the GP (and predict()
    # returns NaN there) — mirrors mogp.train_mogp exactly, but with the ICM.
    docking_mask = np.zeros(train_y.shape[1], dtype=bool)
    docking_mask[[j for j in DOCKING_TASK_INDICES if j < train_y.shape[1]]] = True
    y_mean = np.where(docking_mask, y_mean, np.nan)
    y_std = np.where(docking_mask, y_std, np.nan)

    observed = np.isfinite(y_mean) & np.isfinite(y_std)
    if not observed.any():
        raise ValueError(
            "train_mogp_coregionalized: no observed docking target columns to "
            "train on (the grey-box ICM models only the docking objectives)."
        )

    train_y_norm = (train_y[:, observed] - y_mean[observed]) / y_std[observed]
    train_y_t = torch.from_numpy(train_y_norm).to(torch.float32)

    num_tasks = int(observed.sum())
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
    """Predict the docking objectives and per-task uncertainty for new molecules.

    Same signature and return contract as ``mogp.predict``: predictions come back
    in the full ``TASK_NAMES`` layout on the original (de-normalized) scale, with
    the un-modelled ADMET columns (NaN normalization stats) returned as NaN. Only
    the docking columns carry real predictions.

    Args:
        model: A trained ``MOGPCoregionalized``.
        likelihood: The matching ``MultitaskGaussianLikelihood``.
        y_mean: Normalization means, shape ``(len(TASK_NAMES),)`` (NaN off-docking).
        y_std: Normalization stds, shape ``(len(TASK_NAMES),)`` (NaN off-docking).
        X_new: Fingerprint matrix of shape ``(M, 2048)``.

    Returns:
        A tuple ``(mean, variance)`` of numpy arrays, each shape
        ``(M, len(TASK_NAMES))``, columns in ``TASK_NAMES`` order.
    """
    X_new_t = torch.from_numpy(np.asarray(X_new)).to(torch.float32)

    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = likelihood(model(X_new_t))
        mean_obs = posterior.mean.cpu().numpy()          # (M, K_modelled)
        variance_obs = posterior.variance.cpu().numpy()  # (M, K_modelled)

    # Scatter the trained-task predictions back into the full TASK_NAMES layout,
    # reversing the per-column standardization. Un-modelled columns stay NaN.
    observed = np.isfinite(y_mean) & np.isfinite(y_std)
    n_rows = mean_obs.shape[0]
    n_tasks = y_mean.shape[0]
    mean = np.full((n_rows, n_tasks), np.nan, dtype=float)
    variance = np.full((n_rows, n_tasks), np.nan, dtype=float)
    mean[:, observed] = mean_obs * y_std[observed] + y_mean[observed]
    variance[:, observed] = variance_obs * (y_std[observed] ** 2)
    return mean, variance


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Self-test: fit the 2-task ICM on a small set where the two DOCKING
    # objectives are strongly correlated (both driven by a shared chemical
    # latent), print the learned 2x2 task covariance, and confirm it is NOT
    # diagonal -> the ICM captured the PfDHFR/hDHFR coupling the independent
    # MOGPModel forces to zero. The two dihydrofolate reductases are homologous,
    # so their raw docking scores co-vary POSITIVELY.
    # ------------------------------------------------------------------
    N_MOL = 30
    rng = np.random.default_rng(0)

    PF, HD = DOCKING_TASK_INDICES[0], DOCKING_TASK_INDICES[1]

    try:
        from data import load_library

        lib = load_library()
        n = min(N_MOL, len(lib["smiles"]))
        train_x = lib["fingerprints"][:n].astype(np.float32)
        print(f"Loaded {n} molecules from data/library for the self-test.")
    except (FileNotFoundError, ImportError) as exc:
        print(f"Library unavailable ({exc}); using synthetic fingerprints.")
        n = N_MOL
        train_x = (rng.random((n, 2048)) < 0.02).astype(np.float32)

    # A dominant chemical latent grounded in fingerprint structure (a fixed random
    # projection of the Morgan fingerprints) drives BOTH docking tasks, so the
    # shared component lands in the task covariance the ICM learns.
    w = rng.standard_normal(train_x.shape[1]).astype(np.float32)
    proj = train_x @ w
    latent = (proj - proj.mean()) / (proj.std() + 1e-8)

    Y = np.full((n, len(TASK_NAMES)), np.nan, dtype=np.float32)
    noise = rng.standard_normal((n, 2))
    Y[:, PF] = -8.0 + 2.0 * latent + 0.25 * noise[:, 0]   # parasite docking
    Y[:, HD] = -8.0 + 1.9 * latent + 0.25 * noise[:, 1]   # human docking
    # The ADMET columns are present but ignored by the grey-box GP; fill with
    # independent noise so nothing accidentally depends on them.
    for j, name in enumerate(TASK_NAMES):
        if j not in (PF, HD):
            Y[:, j] = rng.standard_normal(n)

    print(f"\nTraining coregionalized (ICM) GP on {n} molecules over the "
          f"{len(DOCKING_TASK_INDICES)} docking tasks (rank=1)...")
    model, likelihood, y_mean, y_std = train_mogp_coregionalized(
        train_x, Y, n_iterations=200, rank=1
    )
    print(f"Trained-task count: {int(likelihood.num_tasks)} "
          f"(expect {len(DOCKING_TASK_INDICES)} = docking only)")

    B = model.task_covariance_matrix()   # (2, 2)
    labels = [TASK_NAMES[PF], TASK_NAMES[HD]]

    print("\nLearned task-covariance matrix (IndexKernel, 2 x 2):")
    print("            " + "".join(f"{l[:12]:>14}" for l in labels))
    for i, li in enumerate(labels):
        print(f"{li[:12]:>12}" + "".join(f"{B[i, j]:14.4f}" for j in range(2)))

    d = np.sqrt(np.diag(B))
    corr = B / np.outer(d, d)
    off = float(corr[0, 1])
    print(f"\nOff-diagonal task correlation (PfDHFR <-> hDHFR): {off:+.4f}")

    assert B.shape == (len(DOCKING_TASK_INDICES),) * 2
    assert abs(off) > 1e-2, (
        "Task covariance is (near-)diagonal; the ICM captured no cross-task "
        "correlation between the docking objectives."
    )
    assert off > 0.0, (
        "PfDHFR/hDHFR correlation should be positive (homologous targets: good "
        f"binders bind both), got {off:+.4f}."
    )
    print(
        "\nPASS: the learned 2x2 task covariance is non-diagonal "
        f"(corr={off:+.4f}) -> the ICM captures the docking cross-task "
        "correlation the independent MOGPModel forces to zero."
    )
