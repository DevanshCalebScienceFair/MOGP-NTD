"""
baseline_single_obj.py
======================

Single-objective Bayesian optimization baseline for the molecular optimization
pipeline. This is a control that optimizes **only the docking score** (PfDHFR
binding affinity) and ignores ADMET entirely during acquisition.

The structure mirrors ``loop.py`` and ``baseline_random.py``, with one
difference: instead of a multi-output GP + EHVI over four objectives, this uses
a **single-output GP** (fingerprints -> docking score) and **Expected
Improvement (EI)** as the acquisition function. ADMET is never seen by the GP or
the acquisition; it is recorded only so the Pareto front and hypervolume can
still be computed across all four objectives for a fair comparison.

The hypothesis this baseline tests: optimizing docking alone finds strong
binders, but those binders tend to have poor ADMET profiles, so the four-
objective Pareto-front hypervolume should end up *worse* than the true multi-
objective (MOGP + EHVI) loop, even though docking scores themselves look great.

Objective layout (matches ``mogp.TASK_NAMES`` / ``loop.py``):
    Y columns = [Caco2_Permeability, Half_Life, hERG_Toxicity_Prob, PfDHFR_Docking]
    The first three (ADMET) come from the cached library; the fourth (docking)
    is evaluated on the fly for each selected batch and is the *only* objective
    the GP optimizes.

The single-output GP reuses the same setup as ``mogp.py`` (a scaled Tanimoto
kernel, constant mean, Adam on the exact marginal log-likelihood, per-column
target standardization) but with one task instead of a multitask layout.

Run ``python baseline_single_obj.py --help`` for the command-line options.
"""

import os
import time
import argparse

import numpy as np
import torch
import pandas as pd
import gpytorch
from scipy.stats import norm

from data import load_library
from mogp import TASK_NAMES
from kernel import TanimotoKernel
from acquisition import (
    compute_pareto_front,
    get_active_objectives,
    DEFAULT_OBJECTIVE_SIGNS,
)
import evaluation
from docking import batch_dock


# Objective layout (matches TASK_NAMES, identical to loop.py). The library
# supplies the first three (ADMET) columns; docking fills the fourth at
# evaluation time and is the sole objective the GP optimizes.
N_OBJECTIVES = len(TASK_NAMES)
ADMET_COLUMNS = slice(0, 3)      # Caco2_Permeability, Half_Life, hERG_Toxicity_Prob
DOCKING_COLUMN = 3               # PfDHFR_Docking


class SingleTaskTanimotoGP(gpytorch.models.ExactGP):
    """Single-output exact GP with a scaled Tanimoto kernel over fingerprints.

    The single-objective counterpart to ``mogp.MOGPModel``: same constant mean
    and ``ScaleKernel(TanimotoKernel())`` covariance, but with no task batch
    dimension and an ordinary ``MultivariateNormal`` output, so it models one
    scalar target (the docking score).

    Args:
        train_x: Fingerprint tensor of shape ``(N, 2048)``, float32.
        train_y: Target tensor of shape ``(N,)``, float32 (normalized docking).
        likelihood: A ``GaussianLikelihood`` (single-output, not multitask).
    """

    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(TanimotoKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_docking_gp(train_x, train_y, n_iterations=200, lr=0.1):
    """Train the single-output GP on fingerprints and (normalized) docking scores.

    Mirrors ``mogp.train_mogp`` but for one output: standardizes the target,
    fits a ``SingleTaskTanimotoGP`` with a ``GaussianLikelihood`` by Adam on the
    exact marginal log-likelihood, and returns the de-normalization stats.

    Args:
        train_x: Fingerprint matrix of shape ``(N, 2048)``, int8/float.
        train_y: Docking scores of shape ``(N,)``, float (must be all finite).
        n_iterations: Number of Adam steps.
        lr: Adam learning rate.

    Returns:
        A tuple ``(model, likelihood, y_mean, y_std)`` where ``y_mean`` and
        ``y_std`` reverse the target standardization at prediction time.
    """
    train_x_t = torch.from_numpy(np.asarray(train_x)).to(torch.float32)
    train_y = np.asarray(train_y, dtype=np.float32)

    # Standardize the target (constant -> std 0 guarded to 1 so it maps to 0
    # and reverses to its mean).
    y_mean = float(train_y.mean())
    y_std = float(train_y.std())
    if y_std == 0.0:
        y_std = 1.0
    train_y_norm = (train_y - y_mean) / y_std
    train_y_t = torch.from_numpy(train_y_norm).to(torch.float32)

    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = SingleTaskTanimotoGP(train_x_t, train_y_t, likelihood)

    model.train()
    likelihood.train()

    # model.parameters() includes the likelihood's parameters because ExactGP
    # registers the likelihood as a submodule.
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


def predict_docking(model, likelihood, y_mean, y_std, X_new):
    """Predict docking mean and variance for new molecules (original units).

    Args:
        model: A trained ``SingleTaskTanimotoGP``.
        likelihood: The matching ``GaussianLikelihood``.
        y_mean, y_std: Normalization stats from ``train_docking_gp``.
        X_new: Fingerprint matrix of shape ``(M, 2048)``.

    Returns:
        A tuple ``(mean, variance)`` of numpy arrays, each shape ``(M,)``, on the
        original (de-normalized) docking scale.
    """
    X_new_t = torch.from_numpy(np.asarray(X_new)).to(torch.float32)

    model.eval()
    likelihood.eval()

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        posterior = likelihood(model(X_new_t))
        mean = posterior.mean.cpu().numpy()
        variance = posterior.variance.cpu().numpy()

    mean = mean * y_std + y_mean
    variance = variance * (y_std ** 2)
    return mean, variance


def expected_improvement(mean, variance, best_so_far):
    """Expected Improvement for a minimization objective (more negative = better).

    Docking scores are free energies of binding, so lower is better and the
    incumbent is the minimum observed score. For predicted mean ``mu`` and std
    ``sigma`` at a candidate::

        z   = (best_so_far - mu) / sigma
        EI  = (best_so_far - mu) * Phi(z) + sigma * phi(z)

    where ``Phi`` is the standard normal CDF and ``phi`` its PDF. Candidates with
    zero predictive std fall back to the deterministic improvement
    ``max(best_so_far - mu, 0)``.

    Args:
        mean: Predicted docking means, shape ``(M,)``.
        variance: Predicted docking variances, shape ``(M,)``.
        best_so_far: Best (minimum) docking score observed so far.

    Returns:
        EI scores, shape ``(M,)``; higher means more valuable to evaluate next.
    """
    mean = np.asarray(mean, dtype=float)
    std = np.sqrt(np.clip(np.asarray(variance, dtype=float), 0.0, None))
    improvement = best_so_far - mean

    ei = np.zeros_like(mean)
    positive_std = std > 0
    z = improvement[positive_std] / std[positive_std]
    ei[positive_std] = (
        improvement[positive_std] * norm.cdf(z)
        + std[positive_std] * norm.pdf(z)
    )
    # Degenerate (zero-variance) candidates: only the certain improvement counts.
    ei[~positive_std] = np.maximum(improvement[~positive_std], 0.0)
    return ei


def select_batch_ei(X_candidates, ei, batch_size, diversity_threshold):
    """Greedily select a diverse, high-EI batch (same diversity rule as loop.py).

    Candidates are ranked by EI (highest first), then walked in descending
    order; a candidate is added only if its maximum Tanimoto similarity to the
    already-selected molecules is below ``diversity_threshold``. Selection stops
    at ``batch_size`` or when candidates run out.

    Args:
        X_candidates: Candidate fingerprints, shape ``(M, 2048)``.
        ei: EI score per candidate, shape ``(M,)``.
        batch_size: Number of molecules to select.
        diversity_threshold: Max allowed Tanimoto similarity to any already-
            selected molecule.

    Returns:
        A tuple ``(selected_indices, selected_ei)`` of int and float arrays
        (indices into ``X_candidates`` and their EI scores).
    """
    X_candidates = np.asarray(X_candidates)
    ranked = np.argsort(-ei)

    kernel = TanimotoKernel()
    X_t = torch.from_numpy(X_candidates).to(torch.float32)

    selected = []
    for idx in ranked:
        if len(selected) >= batch_size:
            break
        if not selected:
            selected.append(int(idx))
            continue
        # Tanimoto similarity of this candidate to every already-selected one.
        sims = kernel.forward(X_t[idx:idx + 1], X_t[selected]).squeeze(0)
        if float(sims.max()) < diversity_threshold:
            selected.append(int(idx))

    selected_indices = np.asarray(selected, dtype=int)
    selected_ei = ei[selected_indices]
    return selected_indices, selected_ei


class SingleObjectiveBOLoop:
    """Single-objective BO over a fixed molecule library, optimizing docking only.

    Mirrors ``loop.BOLoop`` but replaces the multi-output GP + EHVI stage with a
    single-output Tanimoto GP on the docking score and an Expected Improvement
    acquisition. ADMET objectives are recorded (so the four-objective Pareto /
    hypervolume math is shared with the BO loop and directly comparable) but are
    never used to choose molecules.
    """

    def __init__(self, library_dir="data/library", seed=77,
                 n_init=10, batch_size=10, n_iterations=10,
                 gp_train_iters=200, gp_lr=0.1,
                 diversity_threshold=0.7):
        # --- Reproducibility ---
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        # --- Library (cheap precomputed features) ---
        library = load_library(library_dir)
        self.library_dir = library_dir
        self.smiles = library["smiles"]                          # list, length N
        self.fingerprints = np.asarray(library["fingerprints"])  # (N, 2048) int8
        self.admet_scores = np.asarray(library["admet_scores"])  # (N, 3) float32
        self.library_size = len(self.smiles)

        # --- Hyperparameters ---
        self.n_init = n_init
        self.batch_size = batch_size
        self.n_iterations = n_iterations
        self.gp_train_iters = gp_train_iters
        self.gp_lr = gp_lr
        self.diversity_threshold = diversity_threshold

        # --- Tracking state ---
        self.evaluated_indices = []                              # library indices
        self.Y_evaluated = np.empty((0, N_OBJECTIVES), dtype=np.float64)
        self.history = []

    # ------------------------------------------------------------------ #
    # Evaluation helper (identical to loop.BOLoop._evaluate)
    # ------------------------------------------------------------------ #
    def _evaluate(self, library_indices):
        """Build the (k, 4) objective matrix for the given library indices.

        ADMET objectives come from the cached library; the docking objective is
        evaluated on the fly with ``batch_dock``. Failed docks stay NaN.

        Returns:
            A tuple ``(Y, docking)`` where ``Y`` has shape ``(k, 4)`` and
            ``docking`` is the ``(k,)`` docking score vector.
        """
        library_indices = list(library_indices)
        smiles = [self.smiles[i] for i in library_indices]

        docking = batch_dock(smiles)                             # (k,), NaN on fail

        Y = np.full((len(library_indices), N_OBJECTIVES), np.nan, dtype=np.float64)
        Y[:, ADMET_COLUMNS] = self.admet_scores[library_indices]
        Y[:, DOCKING_COLUMN] = docking
        return Y, docking

    # ------------------------------------------------------------------ #
    # Pareto / hypervolume helpers (shared math with loop.BOLoop)
    # ------------------------------------------------------------------ #
    def _active_signs(self, active):
        """Objective signs (+1/-1) restricted to the active objective columns."""
        return np.asarray(DEFAULT_OBJECTIVE_SIGNS, dtype=float)[active]

    def _pareto_mask(self):
        """Boolean mask over evaluated rows: True for Pareto-optimal molecules.

        Uses only objectives that currently carry data, and only rows fully
        observed across those objectives (a failed dock cannot sit on the front).
        """
        Y = self.Y_evaluated
        full_mask = np.zeros(len(Y), dtype=bool)
        if len(Y) == 0:
            return full_mask

        active = get_active_objectives(Y)
        signs = self._active_signs(active)
        Y_active = Y[:, active]
        finite = np.isfinite(Y_active).all(axis=1)
        if finite.any():
            sub_mask, _ = compute_pareto_front(Y_active[finite], signs)
            full_mask[np.where(finite)[0]] = sub_mask
        return full_mask

    def _hypervolume(self):
        """Hypervolume in the shared, fixed, normalized frame (evaluation.py).

        Delegates to ``evaluation.compute_hypervolume`` — the single source of
        truth — so this baseline reports hypervolume identically to the MOGP
        loop and every other baseline for the same evaluated set.
        """
        return evaluation.compute_hypervolume(self.Y_evaluated)

    # ------------------------------------------------------------------ #
    # Main loop stages
    # ------------------------------------------------------------------ #
    def initialize(self):
        """Seed the loop with ``n_init`` random, freshly-docked molecules."""
        init_indices = np.random.choice(
            self.library_size, size=self.n_init, replace=False
        )
        init_indices = [int(i) for i in init_indices]

        print(f"Initializing with {self.n_init} random molecules...")
        Y, docking = self._evaluate(init_indices)

        self.evaluated_indices = list(init_indices)
        self.Y_evaluated = Y

        n_docked = int(np.isfinite(docking).sum())
        print(f"Initialized {self.n_init} molecules; "
              f"{n_docked}/{self.n_init} docked successfully.")

    def step(self):
        """Run one BO iteration: train docking GP, score by EI, dock, record."""
        iteration = len(self.history) + 1

        # --- Training data: evaluated molecules with a finite docking score ---
        # The GP target is docking, so failed docks (NaN) cannot be training
        # rows. ADMET is deliberately not used here.
        evaluated_arr = np.asarray(self.evaluated_indices, dtype=int)
        docking_obs = self.Y_evaluated[:, DOCKING_COLUMN]
        finite = np.isfinite(docking_obs)
        if finite.sum() < 2:
            print(f"[Iteration {iteration}] not enough successful docks "
                  f"({int(finite.sum())}) to train a GP; stopping early.")
            return False

        train_x = self.fingerprints[evaluated_arr[finite]]
        train_y = docking_obs[finite]
        best_so_far = float(train_y.min())   # most negative = strongest binder

        print(f"\n[Iteration {iteration}] Training single-output docking GP on "
              f"{int(finite.sum())} molecules (best docking so far "
              f"{best_so_far:.4f})...")
        model, likelihood, y_mean, y_std = train_docking_gp(
            train_x, train_y,
            n_iterations=self.gp_train_iters, lr=self.gp_lr,
        )

        # --- Candidate pool: library molecules not yet evaluated ---
        evaluated_set = set(self.evaluated_indices)
        candidate_library_indices = np.array(
            [i for i in range(self.library_size) if i not in evaluated_set],
            dtype=int,
        )
        if len(candidate_library_indices) == 0:
            print(f"[Iteration {iteration}] no candidates left; stopping early.")
            return False
        X_candidates = self.fingerprints[candidate_library_indices]

        # --- EI scoring + diverse batch selection ---
        # select_batch_ei returns indices into X_candidates (the candidate
        # array), NOT into the full library. Map them back below.
        mean, variance = predict_docking(model, likelihood, y_mean, y_std,
                                          X_candidates)
        ei = expected_improvement(mean, variance, best_so_far)
        selected_local, selected_ei = select_batch_ei(
            X_candidates, ei,
            batch_size=self.batch_size,
            diversity_threshold=self.diversity_threshold,
        )
        selected_library_indices = candidate_library_indices[selected_local]

        # --- Validate the candidate -> library index mapping (as in loop.py) ---
        assert np.array_equal(
            self.fingerprints[selected_library_indices],
            X_candidates[selected_local],
        ), "candidate->library index mapping is broken (fingerprint mismatch)"
        assert not (set(int(i) for i in selected_library_indices) & evaluated_set), \
            "select_batch_ei returned an already-evaluated molecule"

        # --- Dock the selected batch ---
        Y_new, docking_new = self._evaluate(list(selected_library_indices))
        n_docked = int(np.isfinite(docking_new).sum())

        self.evaluated_indices.extend(int(i) for i in selected_library_indices)
        self.Y_evaluated = np.vstack([self.Y_evaluated, Y_new])

        # --- Track Pareto front + hypervolume (across ALL 4 objectives) ---
        pareto_size = int(self._pareto_mask().sum())
        hypervolume = self._hypervolume()

        self.history.append({
            "iteration": iteration,
            "n_evaluated": len(self.evaluated_indices),
            "pareto_size": pareto_size,
            "hypervolume": hypervolume,
            "batch_indices": [int(i) for i in selected_library_indices],
            "batch_ei_scores": [float(s) for s in selected_ei],
        })

        print(f"[Iteration {iteration}] "
              f"evaluated={len(self.evaluated_indices)}, "
              f"batch={len(selected_library_indices)}, "
              f"docked_this_batch={n_docked}/{len(selected_library_indices)}, "
              f"pareto_size={pareto_size}, hypervolume={hypervolume:.4f}")
        return True

    def run(self):
        """Run the complete loop: initialize, then ``n_iterations`` steps."""
        self.initialize()
        for _ in range(self.n_iterations):
            if not self.step():
                break

        final = self.history[-1] if self.history else {}
        print("\n=== Single-objective BO complete ===")
        print(f"  Total molecules evaluated: {len(self.evaluated_indices)}")
        print(f"  Final Pareto front size:   {final.get('pareto_size', 0)}")
        print(f"  Final hypervolume:         {final.get('hypervolume', 0.0):.4f}")
        return self.history

    # ------------------------------------------------------------------ #
    # Outputs
    # ------------------------------------------------------------------ #
    def get_pareto_front(self):
        """Return the current Pareto front as a dict of aligned fields."""
        mask = self._pareto_mask()
        rows = np.where(mask)[0]
        indices = [self.evaluated_indices[r] for r in rows]
        smiles = [self.smiles[i] for i in indices]
        objectives = self.Y_evaluated[rows]
        return {
            "indices": indices,
            "smiles": smiles,
            "objectives": objectives,
            "task_names": TASK_NAMES,
        }

    def save_results(self, output_dir="baseline_single_obj_results"):
        """Write history, all evaluations, and the Pareto front to ``output_dir``."""
        os.makedirs(output_dir, exist_ok=True)

        # history.csv
        history_df = pd.DataFrame([
            {
                "iteration": h["iteration"],
                "n_evaluated": h["n_evaluated"],
                "pareto_size": h["pareto_size"],
                "hypervolume": h["hypervolume"],
            }
            for h in self.history
        ])
        history_path = os.path.join(output_dir, "history.csv")
        history_df.to_csv(history_path, index=False)

        # evaluated.csv — every evaluated molecule with all 4 objectives.
        evaluated_df = pd.DataFrame(
            {"SMILES": [self.smiles[i] for i in self.evaluated_indices]}
        )
        for j, name in enumerate(TASK_NAMES):
            evaluated_df[name] = self.Y_evaluated[:, j]
        evaluated_path = os.path.join(output_dir, "evaluated.csv")
        evaluated_df.to_csv(evaluated_path, index=False)

        # pareto_front.csv — only the Pareto-optimal molecules.
        pareto = self.get_pareto_front()
        pareto_df = pd.DataFrame({"SMILES": pareto["smiles"]})
        for j, name in enumerate(TASK_NAMES):
            pareto_df[name] = pareto["objectives"][:, j]
        pareto_path = os.path.join(output_dir, "pareto_front.csv")
        pareto_df.to_csv(pareto_path, index=False)

        print(f"Saved results to {output_dir}/:")
        print(f"  {history_path}")
        print(f"  {evaluated_path}")
        print(f"  {pareto_path}")
        return output_dir


# ---------------------------------------------------------------------- #
# Comparison against the MOGP run (results/)
# ---------------------------------------------------------------------- #
def _load_history(path):
    """Load a history.csv if it exists, else return None."""
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def print_comparison(baseline_history_df, mogp_results_dir="results"):
    """Print a side-by-side summary of the single-objective baseline vs MOGP.

    Reads ``<mogp_results_dir>/history.csv`` (the MOGP loop's output). If it is
    not present, only the baseline's final numbers are reported.
    """
    mogp_path = os.path.join(mogp_results_dir, "history.csv")
    mogp_df = _load_history(mogp_path)

    base_final = baseline_history_df.iloc[-1] if len(baseline_history_df) else None

    print("\n=== MOGP vs Single-Objective BO ===")
    if base_final is None:
        print("  Single-objective baseline produced no history.")
        return

    if mogp_df is None or not len(mogp_df):
        print(f"  (no MOGP results found at {mogp_path}; showing baseline only)")
        print(f"  {'metric':<22}{'Single-Obj':>14}")
        print(f"  {'molecules evaluated':<22}{int(base_final['n_evaluated']):>14}")
        print(f"  {'pareto size':<22}{int(base_final['pareto_size']):>14}")
        print(f"  {'final hypervolume':<22}{base_final['hypervolume']:>14.4f}")
        return

    mogp_final = mogp_df.iloc[-1]
    print(f"  {'metric':<22}{'MOGP':>14}{'Single-Obj':>14}")
    print(f"  {'molecules evaluated':<22}"
          f"{int(mogp_final['n_evaluated']):>14}{int(base_final['n_evaluated']):>14}")
    print(f"  {'pareto size':<22}"
          f"{int(mogp_final['pareto_size']):>14}{int(base_final['pareto_size']):>14}")
    print(f"  {'final hypervolume':<22}"
          f"{mogp_final['hypervolume']:>14.4f}{base_final['hypervolume']:>14.4f}")

    hv_gain = mogp_final["hypervolume"] - base_final["hypervolume"]
    print(f"\n  MOGP hypervolume advantage: {hv_gain:+.4f}")
    if hv_gain > 0:
        print("  -> Multi-objective beats docking-only, as hypothesized.")
    else:
        print("  -> Docking-only matched/beat MOGP here (unexpected; inspect run).")


def save_comparison_plot(baseline_history_df,
                         output_dir="baseline_single_obj_results",
                         mogp_results_dir="results"):
    """Save a hypervolume-vs-molecules-evaluated plot comparing MOGP and Single-Obj.

    Blue line = MOGP (from ``<mogp_results_dir>/history.csv``), orange line =
    Single-Objective BO (this baseline). If the MOGP history is missing, only the
    single-objective curve is drawn and a note is printed.
    """
    import matplotlib
    matplotlib.use("Agg")          # headless: write a file, never open a window
    import matplotlib.pyplot as plt

    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "comparison.png")

    fig, ax = plt.subplots(figsize=(8, 6))

    mogp_df = _load_history(os.path.join(mogp_results_dir, "history.csv"))
    if mogp_df is not None and len(mogp_df):
        ax.plot(mogp_df["n_evaluated"], mogp_df["hypervolume"],
                color="blue", marker="o", label="MOGP")
    else:
        print(f"  (no MOGP history at {mogp_results_dir}/history.csv; "
              "plotting Single-Objective only)")

    if len(baseline_history_df):
        ax.plot(baseline_history_df["n_evaluated"],
                baseline_history_df["hypervolume"],
                color="orange", marker="s", label="Single-Objective BO")

    ax.set_title("MOGP vs Single-Objective BO")
    ax.set_xlabel("Number of molecules evaluated")
    ax.set_ylabel("Hypervolume")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"Saved comparison plot to {plot_path}")
    return plot_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a single-objective (docking-only) BO baseline for the "
                    "PfDHFR MOGP pipeline."
    )
    parser.add_argument("--library-dir", default="data/library")
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--n-iterations", type=int, default=10)
    parser.add_argument("--mogp-iters", type=int, default=200,
                        help="GP training iterations per round.")
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--output-dir", default="baseline_single_obj_results")
    parser.add_argument(
        "--mogp-results-dir", default="results",
        help="Directory holding the MOGP run's history.csv for comparison.",
    )
    args = parser.parse_args()

    start = time.time()

    baseline = SingleObjectiveBOLoop(
        library_dir=args.library_dir,
        seed=args.seed,
        n_init=args.n_init,
        batch_size=args.batch_size,
        n_iterations=args.n_iterations,
        gp_train_iters=args.mogp_iters,
    )
    baseline.run()

    pareto = baseline.get_pareto_front()
    print(f"\nPareto-optimal molecules: {len(pareto['smiles'])}")
    print(f"{'SMILES':<50}" + "".join(f"{n:>22}" for n in pareto["task_names"]))
    for smiles, row in zip(pareto["smiles"], pareto["objectives"]):
        print(f"{smiles:<50}" + "".join(f"{v:22.4f}" for v in row))

    baseline.save_results(output_dir=args.output_dir)

    history_df = pd.DataFrame([
        {
            "iteration": h["iteration"],
            "n_evaluated": h["n_evaluated"],
            "pareto_size": h["pareto_size"],
            "hypervolume": h["hypervolume"],
        }
        for h in baseline.history
    ])
    print_comparison(history_df, mogp_results_dir=args.mogp_results_dir)
    save_comparison_plot(history_df, output_dir=args.output_dir,
                         mogp_results_dir=args.mogp_results_dir)

    elapsed = time.time() - start
    print(f"\nTotal wall-clock time: {elapsed:.1f}s")
