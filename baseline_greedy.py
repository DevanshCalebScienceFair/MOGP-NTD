"""
baseline_greedy.py
==================

Industry-standard *filter-then-dock* baseline for the multi-objective molecular
optimization pipeline — the way most pharmaceutical computational chemistry
campaigns actually triage a library.

The recipe here is deliberately rigid and has **no GP, no Bayesian optimization,
and no uncertainty**:

    1. Apply hard ADMET cutoffs to the whole library at once (a molecule is in
       or out — there is no notion of "close to the boundary").
    2. Take whatever survives, dock it (the expensive step), and rank by binding
       score.

This is the standard the MOGP + EHVI loop (``loop.py``) is meant to beat. The
weakness this baseline exposes is structural: a hard cutoff throws away every
molecule that just misses a threshold, even if it would have docked superbly and
sat on the true Pareto front. Greedy filtering optimizes each property in
isolation and never sees the *tradeoff* between them; the MOGP approach models
all five objectives jointly and recovers Pareto-optimal molecules that the
filter discards outright.

Objective layout (matches ``mogp.TASK_NAMES`` / ``loop.py``):
    Y columns = [PfDHFR_Docking, hDHFR_Docking, hERG_Toxicity_Prob,
                 Caco2_logPapp, Half_Life_hours]
    The three ADMET objectives (hERG, Caco2, Half_Life) come from the cached
    library; the two docking objectives are evaluated on the fly for each
    docked batch.

Run ``python baseline_greedy.py --help`` for the command-line options.
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
# from the cached library and which are docked, against which targets.
N_OBJECTIVES = len(TASK_NAMES)
LIBRARY_TASKS, DOCKING_TASKS, DOCKING_TARGETS = resolve_objective_layout(
    LIBRARY_ADMET_COLUMNS
)

# Column indices within the (N, 3) library ADMET matrix, resolved by name from
# load_library() order [Caco2_logPapp, Half_Life_hours, hERG_Toxicity_Prob].
# The hard filters use ALL three cheap ADMET properties (the "filter-then-dock"
# premise), independent of which of them are GP objectives.
CACO2_COL = LIBRARY_ADMET_COLUMNS.index("Caco2_logPapp")
HALFLIFE_COL = LIBRARY_ADMET_COLUMNS.index("Half_Life_hours")
HERG_COL = LIBRARY_ADMET_COLUMNS.index("hERG_Toxicity_Prob")


class GreedyFilterThenDock:
    """Filter-then-dock baseline over a fixed molecule library.

    Applies hard ADMET cutoffs to the whole library, randomly draws up to
    ``n_total`` survivors within the docking budget, and docks them in batches —
    recording the same Pareto / hypervolume metrics as ``loop.BOLoop`` so the
    runs are directly comparable.
    """

    def __init__(self, library_dir="data/library", seed=55,
                 batch_size=10, n_total=110,
                 herg_threshold=0.5, halflife_min=2.0, caco2_min=-5.5):
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

        # --- Hyperparameters / filter thresholds ---
        self.batch_size = batch_size
        self.n_total = n_total
        self.herg_threshold = herg_threshold
        self.halflife_min = halflife_min
        self.caco2_min = caco2_min

        # --- Tracking state ---
        self.passed_indices = []                                 # library indices
        self.selected_indices = []                               # the docking set
        self.evaluated_indices = []                              # docked so far
        self.Y_evaluated = np.empty((0, N_OBJECTIVES), dtype=np.float64)
        # Raw docking kcal/mol (docking columns only; NaN elsewhere), row-aligned
        # to Y_evaluated; the optimized docking columns are ligand efficiency.
        self.raw_docking = np.empty((0, N_OBJECTIVES), dtype=np.float64)
        self.history = []

    # ------------------------------------------------------------------ #
    # Hard ADMET filter (the whole point of this baseline)
    # ------------------------------------------------------------------ #
    def apply_filters(self):
        """Apply the three hard ADMET cutoffs to the entire library.

        A molecule must pass ALL three gates to survive:
            hERG_Toxicity_Prob < herg_threshold   (cardiac safety)
            Half_Life_hours    > halflife_min      (lasts long enough)
            Caco2_logPapp      > caco2_min         (permeable enough)

        Populates ``self.passed_indices`` and prints the pass/fail counts.
        """
        caco2 = self.admet_scores[:, CACO2_COL]
        halflife = self.admet_scores[:, HALFLIFE_COL]
        herg = self.admet_scores[:, HERG_COL]

        herg_ok = herg < self.herg_threshold
        halflife_ok = halflife > self.halflife_min
        caco2_ok = caco2 > self.caco2_min
        passed = herg_ok & halflife_ok & caco2_ok

        self.passed_indices = [int(i) for i in np.where(passed)[0]]
        n_passed = len(self.passed_indices)
        n_filtered = self.library_size - n_passed

        print("=== Hard ADMET filter ===")
        print(f"  Library size:               {self.library_size}")
        print(f"  hERG  < {self.herg_threshold:<6} passed:   {int(herg_ok.sum())}")
        print(f"  Half-life > {self.halflife_min:<4} passed:   {int(halflife_ok.sum())}")
        print(f"  Caco2 > {self.caco2_min:<6} passed:   {int(caco2_ok.sum())}")
        print(f"  Passed ALL three filters:   {n_passed}")
        print(f"  Filtered out:               {n_filtered}")
        return self.passed_indices

    def select_docking_set(self):
        """Randomly pick up to ``n_total`` molecules from the filter survivors.

        If fewer molecules passed the filter than ``n_total``, the whole
        surviving set is docked. Populates ``self.selected_indices``.
        """
        passed = np.asarray(self.passed_indices, dtype=int)
        k = min(self.n_total, len(passed))
        chosen = np.random.choice(passed, size=k, replace=False)
        self.selected_indices = [int(i) for i in chosen]

        if k < self.n_total:
            print(f"\nOnly {len(passed)} molecules passed the filter "
                  f"(< n_total={self.n_total}); docking all {k}.")
        else:
            print(f"\nRandomly selected {k} of {len(passed)} survivors to dock "
                  f"(budget n_total={self.n_total}).")
        return self.selected_indices

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
    # Main loop: dock the selected set in batches
    # ------------------------------------------------------------------ #
    def run(self):
        """Filter, select, then dock the survivors batch by batch."""
        self.apply_filters()
        self.select_docking_set()

        if not self.selected_indices:
            print("\nNo molecules passed the filter; nothing to dock.")
            return self.history

        n_selected = len(self.selected_indices)
        n_batches = int(np.ceil(n_selected / self.batch_size))
        print(f"\nDocking {n_selected} molecules in {n_batches} batches "
              f"of up to {self.batch_size}...")

        for b in range(n_batches):
            start = b * self.batch_size
            batch = self.selected_indices[start:start + self.batch_size]
            iteration = b + 1

            print(f"\n[Iteration {iteration}] Docking batch of {len(batch)} "
                  f"molecules...")
            Y_new, Y_raw_new, docking_new = self._evaluate(batch)
            batch_docked = docked_summary(docking_new, len(batch))

            self.evaluated_indices.extend(batch)
            self.Y_evaluated = np.vstack([self.Y_evaluated, Y_new])
            self.raw_docking = np.vstack([self.raw_docking, Y_raw_new])

            pareto_size = int(self._pareto_mask().sum())
            hypervolume = self._hypervolume()

            self.history.append({
                "iteration": iteration,
                "n_evaluated": len(self.evaluated_indices),
                "pareto_size": pareto_size,
                "hypervolume": hypervolume,
                "batch_indices": [int(i) for i in batch],
            })

            print(f"[Iteration {iteration}] "
                  f"evaluated={len(self.evaluated_indices)}, "
                  f"batch={len(batch)}, "
                  f"docked_this_batch=[{batch_docked}], "
                  f"pareto_size={pareto_size}, hypervolume={hypervolume:.4f}")

        final = self.history[-1] if self.history else {}
        print("\n=== Greedy filter-then-dock baseline complete ===")
        print(f"  Total molecules docked:    {len(self.evaluated_indices)}")
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

    def save_results(self, output_dir="baseline_greedy_results"):
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

        # evaluated.csv — every docked molecule with all objectives (docking
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
# Comparison against the MOGP run (results/) and the other baselines
# ---------------------------------------------------------------------- #
def _load_history(path):
    """Load a history.csv if it exists, else return None."""
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def print_comparison(greedy_history_df, mogp_results_dir="results"):
    """Print a side-by-side summary of the greedy baseline vs the MOGP run.

    Reads ``<mogp_results_dir>/history.csv``. If it is not present, only the
    greedy baseline's final numbers are reported.
    """
    mogp_df = _load_history(os.path.join(mogp_results_dir, "history.csv"))
    greedy_final = greedy_history_df.iloc[-1] if len(greedy_history_df) else None

    print("\n=== MOGP vs Greedy Filter-Then-Dock ===")
    if greedy_final is None:
        print("  Greedy baseline produced no history.")
        return

    if mogp_df is None or not len(mogp_df):
        print(f"  (no MOGP results found at {mogp_results_dir}/history.csv; "
              "showing greedy only)")
        print(f"  {'metric':<22}{'Greedy':>14}")
        print(f"  {'molecules evaluated':<22}{int(greedy_final['n_evaluated']):>14}")
        print(f"  {'pareto size':<22}{int(greedy_final['pareto_size']):>14}")
        print(f"  {'final hypervolume':<22}{greedy_final['hypervolume']:>14.4f}")
        return

    mogp_final = mogp_df.iloc[-1]
    print(f"  {'metric':<22}{'MOGP':>14}{'Greedy':>14}")
    print(f"  {'molecules evaluated':<22}"
          f"{int(mogp_final['n_evaluated']):>14}{int(greedy_final['n_evaluated']):>14}")
    print(f"  {'pareto size':<22}"
          f"{int(mogp_final['pareto_size']):>14}{int(greedy_final['pareto_size']):>14}")
    print(f"  {'final hypervolume':<22}"
          f"{mogp_final['hypervolume']:>14.4f}{greedy_final['hypervolume']:>14.4f}")

    hv_gain = mogp_final["hypervolume"] - greedy_final["hypervolume"]
    print(f"\n  MOGP hypervolume advantage: {hv_gain:+.4f}")


def save_comparison_plot(greedy_history_df, output_dir="baseline_greedy_results",
                         mogp_results_dir="results"):
    """Save a hypervolume-vs-evaluations plot comparing MOGP and Greedy.

    Blue line = MOGP (from ``<mogp_results_dir>/history.csv``), green line =
    Greedy (this baseline). If the MOGP history is missing, only the greedy
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
              "plotting Greedy only)")

    if len(greedy_history_df):
        ax.plot(greedy_history_df["n_evaluated"],
                greedy_history_df["hypervolume"],
                color="green", marker="^", label="Greedy Filter-Then-Dock")

    ax.set_title("MOGP vs Greedy Filter-Then-Dock")
    ax.set_xlabel("Number of molecules evaluated")
    ax.set_ylabel("Hypervolume")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(f"Saved comparison plot to {plot_path}")
    return plot_path


# The four runs that make up the paper figure: (label, directory, color).
ALL_BASELINES = [
    ("MOGP", "results", "blue"),
    ("Random", "baseline_random_results", "red"),
    ("Single-Obj BO", "baseline_single_obj_results", "orange"),
    ("Greedy Filter-Then-Dock", "baseline_greedy_results", "green"),
]


def save_all_baselines_plot(output_path="all_baselines_comparison.png"):
    """Save the paper figure: every available run's hypervolume curve on one axes.

    Loads ``history.csv`` from each of the four run directories in
    ``ALL_BASELINES`` that exists and plots its hypervolume against molecules
    evaluated. Missing directories are skipped with a note (the figure still
    renders from whatever is present).
    """
    import matplotlib
    matplotlib.use("Agg")          # headless: write a file, never open a window
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))

    n_plotted = 0
    markers = {"MOGP": "o", "Random": "s",
               "Single-Obj BO": "D", "Greedy Filter-Then-Dock": "^"}
    for label, directory, color in ALL_BASELINES:
        df = _load_history(os.path.join(directory, "history.csv"))
        if df is None or not len(df):
            print(f"  (skipping {label}: no history at {directory}/history.csv)")
            continue
        ax.plot(df["n_evaluated"], df["hypervolume"],
                color=color, marker=markers.get(label, "o"), label=label)
        n_plotted += 1

    ax.set_title("Multi-Objective GP-BO vs Baselines")
    ax.set_xlabel("Number of molecules evaluated")
    ax.set_ylabel("Hypervolume")
    ax.grid(True)
    if n_plotted:
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    print(f"Saved all-baselines comparison plot to {output_path} "
          f"({n_plotted} run(s) plotted)")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the industry-standard filter-then-dock baseline "
                    "for the PfDHFR MOGP pipeline."
    )
    parser.add_argument("--library-dir", default="data/library")
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--n-total", type=int, default=110,
                        help="Total molecules to dock (same budget as the MOGP run).")
    parser.add_argument("--herg-threshold", type=float, default=0.5)
    parser.add_argument("--halflife-min", type=float, default=2.0)
    parser.add_argument("--caco2-min", type=float, default=-5.5)
    parser.add_argument("--seed", type=int, default=55)
    parser.add_argument("--output-dir", default="baseline_greedy_results")
    parser.add_argument(
        "--mogp-results-dir", default="results",
        help="Directory holding the MOGP run's history.csv for comparison.",
    )
    args = parser.parse_args()

    start = time.time()

    baseline = GreedyFilterThenDock(
        library_dir=args.library_dir,
        seed=args.seed,
        batch_size=args.batch_size,
        n_total=args.n_total,
        herg_threshold=args.herg_threshold,
        halflife_min=args.halflife_min,
        caco2_min=args.caco2_min,
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
    save_all_baselines_plot()

    elapsed = time.time() - start
    print(f"\nTotal wall-clock time: {elapsed:.1f}s")
