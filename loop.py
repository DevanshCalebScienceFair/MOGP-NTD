"""
loop.py
=======

Orchestrates the full multi-objective Bayesian optimization (BO) loop for
antimalarial drug discovery against *Plasmodium falciparum* dihydrofolate
reductase (PfDHFR).

Each iteration:
    1. Train the multi-output GP (``mogp.py``) on every molecule evaluated so
       far, across the four objectives in ``TASK_NAMES``:
       [Caco2_Permeability, Half_Life, hERG_Toxicity_Prob, PfDHFR_Docking].
    2. Score the un-evaluated library with EHVI and pick a diverse batch
       (``acquisition.select_batch``).
    3. Run the expensive structure-based docking oracle (``docking.batch_dock``)
       on only that batch — docking is the costly objective the cheap ADMET /
       fingerprint features (precomputed in ``data.py``) are there to avoid.
    4. Append the new evaluations, recompute the Pareto front and hypervolume,
       and record history.

The cheap per-molecule quantities (Morgan fingerprints + the first three ADMET
objectives) are pulled straight from the cached library built by ``data.py``;
the loop never recomputes them. Only the 4th objective (docking) is evaluated
on the fly, for the selected batch.

Run ``python loop.py --help`` for the command-line options.
"""

import os
import time
import argparse

import numpy as np
import torch
import pandas as pd

from data import load_library
from mogp import train_mogp, predict, TASK_NAMES
from acquisition import (
    select_batch,
    compute_pareto_front,
    get_active_objectives,
    DEFAULT_OBJECTIVE_SIGNS,
)
import evaluation
from docking import batch_dock


# Objective layout (matches TASK_NAMES). The library supplies the first three
# (ADMET) columns; docking fills the fourth at evaluation time.
N_OBJECTIVES = len(TASK_NAMES)
ADMET_COLUMNS = slice(0, 3)      # Caco2_Permeability, Half_Life, hERG_Toxicity_Prob
DOCKING_COLUMN = 3               # PfDHFR_Docking


class BOLoop:
    """Multi-objective Bayesian optimization loop over a fixed molecule library.

    The library (fingerprints + precomputed ADMET objectives) is loaded once;
    each BO iteration trains the MOGP, selects a batch via EHVI, docks it, and
    folds the results back in.
    """

    def __init__(self, library_dir="data/library", seed=42,
                 n_init=10, batch_size=20, n_iterations=10,
                 mogp_train_iters=200, mogp_lr=0.1,
                 diversity_threshold=0.7):
        # --- Reproducibility ---
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        # --- Library (cheap precomputed features) ---
        library = load_library(library_dir)
        self.library_dir = library_dir
        self.smiles = library["smiles"]                       # list, length N
        self.fingerprints = np.asarray(library["fingerprints"])  # (N, 2048) int8
        self.admet_scores = np.asarray(library["admet_scores"])  # (N, 3) float32
        self.library_size = len(self.smiles)

        # --- Hyperparameters ---
        self.n_init = n_init
        self.batch_size = batch_size
        self.n_iterations = n_iterations
        self.mogp_train_iters = mogp_train_iters
        self.mogp_lr = mogp_lr
        self.diversity_threshold = diversity_threshold

        # --- Tracking state ---
        self.evaluated_indices = []                           # library indices
        self.Y_evaluated = np.empty((0, N_OBJECTIVES), dtype=np.float64)
        self.history = []

    # ------------------------------------------------------------------ #
    # Evaluation helper
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

        docking = batch_dock(smiles)                          # (k,), NaN on fail

        Y = np.full((len(library_indices), N_OBJECTIVES), np.nan, dtype=np.float64)
        Y[:, ADMET_COLUMNS] = self.admet_scores[library_indices]
        Y[:, DOCKING_COLUMN] = docking
        return Y, docking

    # ------------------------------------------------------------------ #
    # Pareto / hypervolume helpers (on the currently-active objectives)
    # ------------------------------------------------------------------ #
    def _active_signs(self, active):
        """Objective signs (+1/-1) restricted to the active objective columns."""
        return np.asarray(DEFAULT_OBJECTIVE_SIGNS, dtype=float)[active]

    def _pareto_mask(self):
        """Boolean mask over evaluated rows: True for Pareto-optimal molecules.

        Uses only objectives that currently carry data, and only rows that are
        fully observed across those objectives (rows missing an active value —
        e.g. a failed dock once docking is active — cannot sit on the front).
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
        truth — so this metric is identical across the MOGP loop and every
        baseline for the same evaluated set, and no longer depends on a
        per-method reference point derived from this run's own data.
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
        """Run one BO iteration: train, select, dock, record."""
        iteration = len(self.history) + 1

        # --- Train the MOGP on everything evaluated so far ---
        train_x = self.fingerprints[self.evaluated_indices]
        train_y = self.Y_evaluated.astype(np.float32)
        print(f"\n[Iteration {iteration}] Training MOGP on "
              f"{len(self.evaluated_indices)} molecules...")
        model, likelihood, y_mean, y_std = train_mogp(
            train_x, train_y,
            n_iterations=self.mogp_train_iters, lr=self.mogp_lr,
        )

        # --- Candidate pool: library molecules not yet evaluated ---
        evaluated_set = set(self.evaluated_indices)
        candidate_library_indices = np.array(
            [i for i in range(self.library_size) if i not in evaluated_set],
            dtype=int,
        )
        X_candidates = self.fingerprints[candidate_library_indices]

        # --- EHVI batch selection ---
        # select_batch returns indices into X_candidates (the candidate array),
        # NOT into the full library. Map them back via candidate_library_indices.
        selected_local, selected_ehvi = select_batch(
            model, likelihood, y_mean, y_std,
            X_candidates, self.Y_evaluated,
            batch_size=self.batch_size,
            diversity_threshold=self.diversity_threshold,
        )
        selected_library_indices = candidate_library_indices[selected_local]

        # --- Validate the candidate -> library index mapping ---
        # Most BO bugs hide here: a wrong remap silently re-docks or mislabels
        # molecules. Assert the remapped fingerprints match the ones select_batch
        # actually scored, and that nothing already evaluated slipped through.
        assert np.array_equal(
            self.fingerprints[selected_library_indices],
            X_candidates[selected_local],
        ), "candidate->library index mapping is broken (fingerprint mismatch)"
        assert not (set(int(i) for i in selected_library_indices) & evaluated_set), \
            "select_batch returned an already-evaluated molecule"

        # --- Dock the selected batch ---
        Y_new, docking_new = self._evaluate(list(selected_library_indices))
        n_docked = int(np.isfinite(docking_new).sum())

        self.evaluated_indices.extend(int(i) for i in selected_library_indices)
        self.Y_evaluated = np.vstack([self.Y_evaluated, Y_new])

        # --- Track Pareto front + hypervolume ---
        pareto_size = int(self._pareto_mask().sum())
        hypervolume = self._hypervolume()

        self.history.append({
            "iteration": iteration,
            "n_evaluated": len(self.evaluated_indices),
            "pareto_size": pareto_size,
            "hypervolume": hypervolume,
            "batch_indices": [int(i) for i in selected_library_indices],
            "batch_ehvi_scores": [float(s) for s in selected_ehvi],
        })

        print(f"[Iteration {iteration}] "
              f"evaluated={len(self.evaluated_indices)}, "
              f"batch={len(selected_library_indices)}, "
              f"docked_this_batch={n_docked}/{len(selected_library_indices)}, "
              f"pareto_size={pareto_size}, hypervolume={hypervolume:.4f}")

    def run(self):
        """Run the complete loop: initialize, then ``n_iterations`` steps."""
        self.initialize()
        for _ in range(self.n_iterations):
            self.step()

        final = self.history[-1] if self.history else {}
        print("\n=== BO run complete ===")
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

    def save_results(self, output_dir="results"):
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the multi-objective BO loop for PfDHFR drug discovery."
    )
    parser.add_argument("--library-dir", default="data/library")
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--n-iterations", type=int, default=10)
    parser.add_argument("--mogp-iters", type=int, default=200)
    parser.add_argument("--output-dir", default="results")
    args = parser.parse_args()

    start = time.time()

    loop = BOLoop(
        library_dir=args.library_dir,
        n_init=args.n_init,
        batch_size=args.batch_size,
        n_iterations=args.n_iterations,
        mogp_train_iters=args.mogp_iters,
    )
    loop.run()

    pareto = loop.get_pareto_front()
    print(f"\nPareto-optimal molecules: {len(pareto['smiles'])}")
    print(f"{'SMILES':<50}" + "".join(f"{n:>22}" for n in pareto["task_names"]))
    for smiles, row in zip(pareto["smiles"], pareto["objectives"]):
        print(f"{smiles:<50}" + "".join(f"{v:22.4f}" for v in row))

    loop.save_results(output_dir=args.output_dir)

    elapsed = time.time() - start
    print(f"\nTotal wall-clock time: {elapsed:.1f}s")
