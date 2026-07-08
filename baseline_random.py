"""
baseline_random.py
==================

Random-search baseline for the multi-objective molecular optimization pipeline,
provided as a control to measure how much the MOGP + EHVI loop (``loop.py``)
actually buys over naive sampling.

The structure mirrors ``loop.py`` exactly, with one difference: there is **no
GP and no EHVI**. Each round simply draws ``batch_size`` molecules uniformly at
random from the pool of not-yet-evaluated library molecules, docks them, and
folds the results back in. Everything else — the cheap precomputed ADMET
objectives pulled from the cached library, the on-the-fly docking objective, and
the Pareto-front / hypervolume bookkeeping — is identical to the BO loop, so the
two runs are directly comparable on the same axes.

Objective layout (matches ``mogp.TASK_NAMES`` / ``loop.py``):
    Y columns = [PfDHFR_Docking, hDHFR_Docking, hERG_Toxicity_Prob,
                 Caco2_logPapp, Half_Life_hours]
    The three ADMET objectives (hERG, Caco2, Half_Life) come from the cached
    library; the two docking objectives are evaluated on the fly for each
    selected batch, against both targets.

Run ``python baseline_random.py --help`` for the command-line options.
"""

import os
import time
import argparse

import numpy as np
import pandas as pd

from data import load_library, ADMET_COLUMNS as LIBRARY_ADMET_COLUMNS
from mogp import TASK_NAMES, resolve_objective_layout
from acquisition import (
    compute_pareto_front,
    get_active_objectives,
    DEFAULT_OBJECTIVE_SIGNS,
)
import evaluation
from docking import batch_dock_targets, docked_summary, raw_to_ligand_efficiency


# Objective -> data-source layout (identical to loop.py): which columns come
# from the cached library (cheap ADMET, e.g. hERG) and which are docked, against
# which targets. Resolved from TASK_NAMES, not hard-coded to column positions.
N_OBJECTIVES = len(TASK_NAMES)
LIBRARY_TASKS, DOCKING_TASKS, DOCKING_TARGETS = resolve_objective_layout(
    LIBRARY_ADMET_COLUMNS
)


class RandomSearchBaseline:
    """Random-selection baseline over a fixed molecule library.

    Mirrors ``loop.BOLoop`` but replaces the train -> EHVI -> select stage with a
    uniform random draw from the unevaluated pool. The Pareto / hypervolume math
    is shared with the BO loop so the two are comparable.
    """

    def __init__(self, library_dir="data/library", seed=99,
                 n_init=10, batch_size=10, n_iterations=10):
        # --- Reproducibility (numpy only; there is no torch model here) ---
        self.seed = seed
        np.random.seed(seed)

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

        # --- Tracking state ---
        self.evaluated_indices = []                              # library indices
        self.Y_evaluated = np.empty((0, N_OBJECTIVES), dtype=np.float64)
        # Raw docking kcal/mol (docking columns only; NaN elsewhere), row-aligned
        # to Y_evaluated; the optimized docking columns are ligand efficiency.
        self.raw_docking = np.empty((0, N_OBJECTIVES), dtype=np.float64)
        self.history = []

    # ------------------------------------------------------------------ #
    # Evaluation helper (identical to loop.BOLoop._evaluate)
    # ------------------------------------------------------------------ #
    def _evaluate(self, library_indices):
        """Build the ``(k, N_OBJECTIVES)`` objective matrix for the given indices.

        Library objectives (e.g. hERG) come from the cached library; the docking
        objectives are evaluated on the fly against every target
        (``DOCKING_TARGETS``). Failed docks stay NaN. Identical to
        ``loop.BOLoop._evaluate``: the docking oracle/cache return RAW kcal/mol,
        but the OPTIMIZED docking objective is size-corrected LIGAND EFFICIENCY
        (raw / heavy-atom count, ``docking.raw_to_ligand_efficiency``), applied
        here downstream of the cache. Raw kcal is retained in ``Y_raw``.

        Returns:
            A tuple ``(Y, Y_raw, docking_by_target)`` where ``Y`` has LIGAND
            EFFICIENCY in the docking columns, ``Y_raw`` has RAW kcal/mol there
            (NaN elsewhere), and ``docking_by_target`` maps each target name to
            its ``(k,)`` RAW docking-score vector.
        """
        library_indices = list(library_indices)
        smiles = [self.smiles[i] for i in library_indices]
        admet_rows = self.admet_scores[library_indices]

        docking_by_target = batch_dock_targets(smiles, DOCKING_TARGETS)

        Y = np.full((len(library_indices), N_OBJECTIVES), np.nan, dtype=np.float64)
        Y_raw = np.full((len(library_indices), N_OBJECTIVES), np.nan, dtype=np.float64)
        for j, col in LIBRARY_TASKS:
            Y[:, j] = admet_rows[:, col]
        for j, target in DOCKING_TASKS:
            raw = docking_by_target[target]
            Y_raw[:, j] = raw
            Y[:, j] = [raw_to_ligand_efficiency(r, s) for r, s in zip(raw, smiles)]
        return Y, Y_raw, docking_by_target

    # ------------------------------------------------------------------ #
    # Pareto / hypervolume helpers (shared math with loop.BOLoop)
    # ------------------------------------------------------------------ #
    def _active_signs(self, active):
        """Objective signs (+1/-1) restricted to the active objective columns."""
        return np.asarray(DEFAULT_OBJECTIVE_SIGNS, dtype=float)[active]

    def _pareto_mask(self):
        """Boolean mask over evaluated rows: True for Pareto-optimal molecules.

        Uses only objectives that currently carry data, and only rows that are
        fully observed across those objectives (rows missing an active value —
        e.g. a failed dock — cannot sit on the front).
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
    def _random_indices(self, k):
        """Pick ``k`` random library indices not yet evaluated (without replacement).

        Returns fewer than ``k`` only if the unevaluated pool is smaller than
        ``k`` (the pool is then exhausted).
        """
        evaluated_set = set(self.evaluated_indices)
        candidate_library_indices = np.array(
            [i for i in range(self.library_size) if i not in evaluated_set],
            dtype=int,
        )
        k = min(k, len(candidate_library_indices))
        chosen = np.random.choice(candidate_library_indices, size=k, replace=False)
        return [int(i) for i in chosen]

    def initialize(self):
        """Seed the baseline with ``n_init`` random, freshly-docked molecules."""
        init_indices = self._random_indices(self.n_init)

        print(f"Initializing with {len(init_indices)} random molecules...")
        Y, Y_raw, docking = self._evaluate(init_indices)

        self.evaluated_indices = list(init_indices)
        self.Y_evaluated = Y
        self.raw_docking = Y_raw

        print(f"Initialized {len(init_indices)} molecules; "
              f"docked {docked_summary(docking, len(init_indices))}.")

    def step(self):
        """Run one random-search round: pick a random batch, dock, record."""
        iteration = len(self.history) + 1

        # --- Random batch from the unevaluated pool (no GP, no EHVI) ---
        selected_library_indices = self._random_indices(self.batch_size)
        if not selected_library_indices:
            print(f"[Iteration {iteration}] no candidates left; stopping early.")
            return False

        # Sanity check: nothing already evaluated should slip through.
        assert not (set(selected_library_indices) & set(self.evaluated_indices)), \
            "random selection returned an already-evaluated molecule"

        # --- Dock the selected batch ---
        print(f"\n[Iteration {iteration}] Randomly selected "
              f"{len(selected_library_indices)} molecules; docking...")
        Y_new, Y_raw_new, docking_new = self._evaluate(selected_library_indices)
        batch_docked = docked_summary(docking_new, len(selected_library_indices))

        self.evaluated_indices.extend(selected_library_indices)
        self.Y_evaluated = np.vstack([self.Y_evaluated, Y_new])
        self.raw_docking = np.vstack([self.raw_docking, Y_raw_new])

        # --- Track Pareto front + hypervolume ---
        pareto_size = int(self._pareto_mask().sum())
        hypervolume = self._hypervolume()

        self.history.append({
            "iteration": iteration,
            "n_evaluated": len(self.evaluated_indices),
            "pareto_size": pareto_size,
            "hypervolume": hypervolume,
            "batch_indices": [int(i) for i in selected_library_indices],
        })

        print(f"[Iteration {iteration}] "
              f"evaluated={len(self.evaluated_indices)}, "
              f"batch={len(selected_library_indices)}, "
              f"docked_this_batch=[{batch_docked}], "
              f"pareto_size={pareto_size}, hypervolume={hypervolume:.4f}")
        return True

    def run(self):
        """Run the complete baseline: initialize, then ``n_iterations`` rounds."""
        self.initialize()
        for _ in range(self.n_iterations):
            if not self.step():
                break

        final = self.history[-1] if self.history else {}
        print("\n=== Random-search baseline complete ===")
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
            "raw_docking": self.raw_docking[rows],   # raw kcal/mol (docking cols)
            "task_names": TASK_NAMES,
        }

    def save_results(self, output_dir="baseline_random_results"):
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

        # evaluated.csv — every evaluated molecule with all objectives (docking
        # columns are ligand efficiency), the RAW docking kcal as ``*_kcal``
        # columns, plus the reported-only Selectivity Index.
        evaluated_df = pd.DataFrame(
            {"SMILES": [self.smiles[i] for i in self.evaluated_indices]}
        )
        for j, name in enumerate(TASK_NAMES):
            evaluated_df[name] = self.Y_evaluated[:, j]
        for j, _target in DOCKING_TASKS:
            evaluated_df[f"{TASK_NAMES[j]}_kcal"] = self.raw_docking[:, j]
        evaluation.add_selectivity_index(evaluated_df)
        evaluated_path = os.path.join(output_dir, "evaluated.csv")
        evaluated_df.to_csv(evaluated_path, index=False)

        # pareto_front.csv — only the Pareto-optimal molecules, with the raw
        # docking kcal (``*_kcal``) and the Selectivity Index (hDHFR - PfDHFR).
        pareto = self.get_pareto_front()
        pareto_df = pd.DataFrame({"SMILES": pareto["smiles"]})
        for j, name in enumerate(TASK_NAMES):
            pareto_df[name] = pareto["objectives"][:, j]
        for j, _target in DOCKING_TASKS:
            pareto_df[f"{TASK_NAMES[j]}_kcal"] = pareto["raw_docking"][:, j]
        evaluation.add_selectivity_index(pareto_df)
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
    """Print a side-by-side summary of the random baseline vs the MOGP run.

    Reads ``<mogp_results_dir>/history.csv`` (the MOGP loop's output). If it is
    not present, only the baseline's final numbers are reported.
    """
    mogp_path = os.path.join(mogp_results_dir, "history.csv")
    mogp_df = _load_history(mogp_path)

    rand_final = baseline_history_df.iloc[-1] if len(baseline_history_df) else None

    print("\n=== MOGP vs Random Search ===")
    if rand_final is None:
        print("  Random baseline produced no history.")
        return

    if mogp_df is None or not len(mogp_df):
        print(f"  (no MOGP results found at {mogp_path}; showing baseline only)")
        print(f"  {'metric':<22}{'Random':>14}")
        print(f"  {'molecules evaluated':<22}{int(rand_final['n_evaluated']):>14}")
        print(f"  {'pareto size':<22}{int(rand_final['pareto_size']):>14}")
        print(f"  {'final hypervolume':<22}{rand_final['hypervolume']:>14.4f}")
        return

    mogp_final = mogp_df.iloc[-1]
    print(f"  {'metric':<22}{'MOGP':>14}{'Random':>14}")
    print(f"  {'molecules evaluated':<22}"
          f"{int(mogp_final['n_evaluated']):>14}{int(rand_final['n_evaluated']):>14}")
    print(f"  {'pareto size':<22}"
          f"{int(mogp_final['pareto_size']):>14}{int(rand_final['pareto_size']):>14}")
    print(f"  {'final hypervolume':<22}"
          f"{mogp_final['hypervolume']:>14.4f}{rand_final['hypervolume']:>14.4f}")

    hv_gain = mogp_final["hypervolume"] - rand_final["hypervolume"]
    print(f"\n  MOGP hypervolume advantage: {hv_gain:+.4f}")


def save_comparison_plot(baseline_history_df, output_dir="baseline_random_results",
                         mogp_results_dir="results"):
    """Save a hypervolume-vs-molecules-evaluated plot comparing MOGP and Random.

    Blue line = MOGP (from ``<mogp_results_dir>/history.csv``), red line =
    Random (this baseline). If the MOGP history is missing, only the random
    curve is drawn and a note is printed.
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
              "plotting Random only)")

    if len(baseline_history_df):
        ax.plot(baseline_history_df["n_evaluated"],
                baseline_history_df["hypervolume"],
                color="red", marker="s", label="Random")

    ax.set_title("MOGP vs Random Search")
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
        description="Run a random-search baseline for the PfDHFR MOGP pipeline."
    )
    parser.add_argument("--library-dir", default="data/library")
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--n-iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--output-dir", default="baseline_random_results")
    parser.add_argument(
        "--mogp-results-dir", default="results",
        help="Directory holding the MOGP run's history.csv for comparison.",
    )
    args = parser.parse_args()

    start = time.time()

    baseline = RandomSearchBaseline(
        library_dir=args.library_dir,
        seed=args.seed,
        n_init=args.n_init,
        batch_size=args.batch_size,
        n_iterations=args.n_iterations,
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
