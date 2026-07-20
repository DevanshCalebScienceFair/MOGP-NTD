"""
loop.py
=======

Orchestrates the full multi-objective Bayesian optimization (BO) loop for
antimalarial drug discovery against *Plasmodium falciparum* dihydrofolate
reductase (PfDHFR).

Each iteration:
    1. Train the multi-output GP (``mogp.py``) on every molecule evaluated so
       far, across the objectives in ``TASK_NAMES`` (a potency / selectivity /
       safety / ADMET set: [PfDHFR_Docking, hDHFR_Docking, hERG_Toxicity_Prob,
       Caco2_logPapp, Half_Life_hours]).
    2. Score the un-evaluated library with qNEHVI — a grey-box composite
       acquisition that models only the docking objectives and folds in each
       candidate's KNOWN-EXACT ADMET values — and pick a diverse batch
       (``acquisition.select_batch``).
    3. Run the expensive structure-based docking oracle on only that batch,
       against EVERY target (``docking.batch_dock_targets`` — PfDHFR and hDHFR).
       Docking is the costly objective the cheap ADMET / fingerprint features
       (precomputed in ``data.py``) are there to avoid; two targets ~double it.
    4. Append the new evaluations, recompute the Pareto front and hypervolume,
       and record history.

The cheap per-molecule quantities (Morgan fingerprints + the library ADMET
objectives, e.g. hERG) are pulled straight from the cached library built by
``data.py``; the loop never recomputes them. Only the docking objectives are
evaluated on the fly, for the selected batch.

Run ``python loop.py --help`` for the command-line options.
"""

import os

# ---------------------------------------------------------------------------
# Threading / OpenMP guard — MUST run before numpy/torch/botorch are imported.
# On Apple Silicon a duplicate libomp runtime can crash on import (OMP Error #15)
# or deadlock inside torch autograd (_C.backward); pinning every math backend to
# a single thread and tolerating the duplicate runtime avoids both. launch.py
# sets the same variables (before it imports this module); setdefault means
# whichever runs first wins and an explicit outer override is always respected,
# so `python loop.py` is safe standalone without the old `env KMP_...` prefix.
# ---------------------------------------------------------------------------
for _thread_var in ("KMP_DUPLICATE_LIB_OK", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(
        _thread_var, "TRUE" if _thread_var == "KMP_DUPLICATE_LIB_OK" else "1"
    )

import time
import argparse
import warnings

import numpy as np
import torch
import pandas as pd

from data import (
    load_library,
    process_smiles,
    ADMET_COLUMNS as LIBRARY_ADMET_COLUMNS,
    heavy_atom_stats,
    pareto_heavy_summary,
    FRAGMENT_MEDIAN_WARN,
)
from admet_oracle import ADMETOracle
from densify import generate_analogs, canonical_smiles
import quality_filter
from mogp import train_mogp, predict, TASK_NAMES, resolve_objective_layout
from mogp_coregionalized import train_mogp_coregionalized
from acquisition import (
    select_batch,
    compute_pareto_front,
    get_active_objectives,
    DEFAULT_OBJECTIVE_SIGNS,
)
import evaluation
from docking import batch_dock_targets, docked_summary, raw_to_ligand_efficiency


# Objective -> data-source layout, resolved once from TASK_NAMES. Some objectives
# come from the cached library (cheap ADMET), the rest are docked (expensive),
# possibly against several targets. This replaces the old "ADMET cols 0-2,
# docking col 3" assumption, which no longer holds now that two of the three
# objectives are docking scores against different receptors.
N_OBJECTIVES = len(TASK_NAMES)
LIBRARY_TASKS, DOCKING_TASKS, DOCKING_TARGETS = resolve_objective_layout(
    LIBRARY_ADMET_COLUMNS
)
# The generalization of the old single DOCKING_COLUMN: the set of objective
# columns filled by docking.
DOCKING_COLUMNS = [j for j, _ in DOCKING_TASKS]

# Margin (kcal/mol per heavy atom) below evaluation.DOCKING_LE_MIN at which we
# warn that observed docking ligand efficiencies are saturating the fixed
# normalization floor (see BOLoop._warn_if_docking_saturates). On the LE scale
# (range ~0.55 wide) this is far smaller than the old raw-kcal margin.
DOCKING_SATURATION_MARGIN = 0.02


# The two GP models this loop can run over the docking objectives. The
# coregionalized (ICM) model is the PRIMARY / headline model: it learns the
# PfDHFR/hDHFR cross-task correlation the independent model forces to zero. The
# independent model is retained for the ablation (run_ablation.py).
MODEL_CHOICES = ("coregionalized", "independent")
DEFAULT_MODEL = "coregionalized"


# Locked-in run profiles selected by the --smoke flag (shared with launch.py).
# --smoke is a fast (~5 min) end-to-end verification; without it the loop runs
# the full "Grand Campaign" configuration. Explicit --n-init / --batch-size /
# --n-iterations on the CLI still override whichever profile is active.
SMOKE_PARAMS = {"n_init": 5, "batch_size": 2, "n_iterations": 2}
GRAND_CAMPAIGN_PARAMS = {"n_init": 40, "batch_size": 5, "n_iterations": 50}


def resolve_train_fn(model, rank=1):
    """Map a ``--model`` name to a ``train_fn`` with ``mogp.train_mogp``'s signature.

    Both models share the grey-box contract (train the docking tasks only, return
    ``(model, likelihood, y_mean, y_std)``), so the rest of the loop — qNEHVI
    selection via ``acquisition.select_batch``, which decodes the posterior
    through the model-agnostic ``mogp.predict`` — is identical either way.

    Args:
        model: ``"coregionalized"`` (ICM) or ``"independent"``.
        rank: ``IndexKernel`` rank for the coregionalized model (ignored otherwise).

    Returns:
        A callable ``train_fn(train_x, train_y, n_iterations=..., lr=...)``.
    """
    if model == "independent":
        return train_mogp
    if model == "coregionalized":
        def _train_coregionalized(train_x, train_y, n_iterations=200, lr=0.1):
            return train_mogp_coregionalized(
                train_x, train_y, n_iterations=n_iterations, lr=lr, rank=rank
            )
        return _train_coregionalized
    raise ValueError(
        f"Unknown model {model!r}; choose one of {MODEL_CHOICES}."
    )


class BOLoop:
    """Multi-objective Bayesian optimization loop over a fixed molecule library.

    The library (fingerprints + precomputed ADMET objectives) is loaded once;
    each BO iteration trains the docking GP (coregionalized ICM by default, or
    the independent model), selects a batch via grey-box qNEHVI, docks it, and
    folds the results back in.
    """

    def __init__(self, library_dir="data/library", seed=42,
                 n_init=10, batch_size=20, n_iterations=10,
                 mogp_train_iters=200, mogp_lr=0.1,
                 diversity_threshold=0.7,
                 model=DEFAULT_MODEL, coregionalization_rank=1, train_fn=None,
                 densify=False, densify_every=1, densify_per_parent=20,
                 densify_max_pool=None):
        # --- Reproducibility ---
        self.seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        # --- GP training function ---
        # The PRIMARY / headline model is the coregionalized (ICM) GP, which
        # learns the PfDHFR/hDHFR cross-task correlation. `model` selects it by
        # name ("coregionalized" or "independent") and `coregionalization_rank`
        # sets the ICM rank; run_ablation.py compares the two arms. An explicit
        # `train_fn` overrides the name (it must match mogp.train_mogp's signature
        # and return contract) — the ablation harness uses either path. The rest
        # of the loop — qNEHVI selection via acquisition.select_batch, which
        # decodes the posterior through the model-agnostic mogp.predict — is
        # identical for either model, so the model is the only thing that varies.
        self.model_name = model
        self.coregionalization_rank = coregionalization_rank
        self.train_fn = (train_fn if train_fn is not None
                         else resolve_train_fn(model, rank=coregionalization_rank))

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

        # --- Densification (grow candidates around the Pareto front) ---
        # OFF by default so the base benchmark is unchanged. When on, after each
        # iteration we enumerate analogs of the current front, score/filter them
        # exactly like the base library, and inject the survivors as new
        # candidates. See _densify.
        self.densify = densify
        self.densify_every = densify_every
        self.densify_per_parent = densify_per_parent
        self.densify_max_pool = densify_max_pool
        # Fail loudly if densification is requested but can never add a molecule.
        # densify_max_pool caps the TOTAL library size, so a cap already met by
        # the base library leaves _densify with room == 0 every iteration: it
        # returns before adding anything and densify becomes a silent no-op (the
        # only signal is a buried per-iteration print). That makes a --densify run
        # byte-identical to a plain run — an easy, expensive mistake to miss. Raise
        # instead, so the invocation is fixed before a whole run is wasted.
        if self.densify and self.densify_max_pool is not None \
                and self.densify_max_pool <= self.library_size:
            raise ValueError(
                f"--densify-max-pool ({self.densify_max_pool}) is <= the current "
                f"library size ({self.library_size}), so densification can never "
                "add a molecule and would be a silent no-op. densify_max_pool caps "
                "the TOTAL library size after densification, not the number of "
                "analogs added. Set it ABOVE the library size (e.g. "
                f"{self.library_size} + N to allow ~N analogs) or omit it for no cap."
            )
        # A reused ADMET oracle and a canonical-SMILES set of everything already
        # in the library (for novelty dedup), both built lazily only when
        # densification is enabled so the base loop pays nothing for them.
        self._oracle = None
        self._library_canonical = None
        if self.densify:
            self._oracle = ADMETOracle()
            self._library_canonical = set()
            for s in self.smiles:
                canon = canonical_smiles(s)
                if canon is not None:
                    self._library_canonical.add(canon)

        # --- Tracking state ---
        self.evaluated_indices = []                           # library indices
        self.Y_evaluated = np.empty((0, N_OBJECTIVES), dtype=np.float64)
        # Raw docking kcal/mol (docking columns only; NaN elsewhere), row-aligned
        # to Y_evaluated. The OPTIMIZED docking columns in Y_evaluated are ligand
        # efficiency; this keeps the raw kcal for reporting so it is never lost.
        self.raw_docking = np.empty((0, N_OBJECTIVES), dtype=np.float64)
        self.history = []

    # ------------------------------------------------------------------ #
    # Evaluation helper
    # ------------------------------------------------------------------ #
    def _evaluate(self, library_indices):
        """Build the ``(k, N_OBJECTIVES)`` objective matrix for the given indices.

        Library objectives (cheap ADMET, e.g. hERG) come from the cached library;
        the docking objectives are evaluated on the fly, docking EACH selected
        molecule against EVERY required target (``DOCKING_TARGETS`` — PfDHFR and
        hDHFR), which roughly doubles docking cost. Failed docks stay NaN.

        The docking oracle/cache still return RAW kcal/mol; the OPTIMIZED docking
        objective is size-corrected LIGAND EFFICIENCY (raw / heavy-atom count, via
        ``docking.raw_to_ligand_efficiency``), applied here — downstream of the
        cache — so the GP and hypervolume see LE while the cache stays raw and
        valid. The raw kcal is retained separately (``Y_raw``) for reporting.

        Returns:
            A tuple ``(Y, Y_raw, docking_by_target)`` where ``Y`` has shape
            ``(k, N_OBJECTIVES)`` with LIGAND EFFICIENCY in the docking columns,
            ``Y_raw`` has the same shape with RAW kcal/mol in the docking columns
            (NaN elsewhere), and ``docking_by_target`` maps each target name to
            its ``(k,)`` RAW docking-score vector.
        """
        library_indices = list(library_indices)
        smiles = [self.smiles[i] for i in library_indices]
        admet_rows = self.admet_scores[library_indices]

        # Dock the batch against every required target (PfDHFR + hDHFR).
        docking_by_target = batch_dock_targets(smiles, DOCKING_TARGETS)

        Y = np.full((len(library_indices), N_OBJECTIVES), np.nan, dtype=np.float64)
        Y_raw = np.full((len(library_indices), N_OBJECTIVES), np.nan, dtype=np.float64)
        for j, col in LIBRARY_TASKS:
            Y[:, j] = admet_rows[:, col]
        for j, target in DOCKING_TASKS:
            raw = docking_by_target[target]
            Y_raw[:, j] = raw
            # Convert raw kcal -> ligand efficiency per molecule (size-corrected);
            # NaN raw / unparseable SMILES propagate to NaN, like a failed dock.
            Y[:, j] = [raw_to_ligand_efficiency(r, s) for r, s in zip(raw, smiles)]
        return Y, Y_raw, docking_by_target

    def _warn_if_docking_saturates(self, Y_new):
        """Warn if any observed docking LE sits at/below the normalization floor.

        The docking columns are now size-corrected ligand efficiency, so the
        relevant floor is ``evaluation.DOCKING_LE_MIN`` (kcal/mol per heavy atom),
        the fixed lower bound of the shared docking LE normalization. An LE at or
        below it maps to the best normalized value (1.0) and clips, so an even
        more efficient binder earns NO additional hypervolume — the metric
        saturates. This is a real risk once densification starts proposing more
        efficient binders. We only WARN (via ``warnings.warn``, so result files
        are untouched); we deliberately do NOT move the bound, because changing it
        mid-study would break cross-method hypervolume comparability. Widen
        ``evaluation.DOCKING_LE_MIN`` yourself, once, for a fresh comparison if
        this fires.
        """
        if not DOCKING_COLUMNS:
            return
        dock_vals = np.asarray(Y_new)[:, DOCKING_COLUMNS]
        finite = dock_vals[np.isfinite(dock_vals)]
        if finite.size == 0:
            return
        strongest = float(finite.min())
        if strongest < evaluation.DOCKING_LE_MIN + DOCKING_SATURATION_MARGIN:
            warnings.warn(
                f"Observed docking ligand efficiency {strongest:.3f} kcal/mol/atom "
                f"is within {DOCKING_SATURATION_MARGIN} of (or below) the "
                f"normalization floor evaluation.DOCKING_LE_MIN="
                f"{evaluation.DOCKING_LE_MIN}; such values saturate to hypervolume "
                "1.0 and stop earning credit. Consider widening DOCKING_LE_MIN for "
                "a fresh comparison (do NOT change it mid-study — it breaks "
                "comparability)."
            )

    def _densify(self, iteration):
        """Grow the candidate pool with analogs of the current Pareto front.

        Enumerates novel analogs of the front molecules (``densify.generate_analogs``),
        pushes them through the SAME per-molecule pipeline as the base library
        (``data.process_smiles``: drug-likeness filter -> Morgan featurize ->
        ADMET score -> domain/NaN drop), and appends the survivors to the live
        library arrays (``smiles`` / ``fingerprints`` / ``admet_scores``) so they
        become eligible candidates on the NEXT iteration. Docking is NOT run here
        — densification only grows the candidate pool.

        Injected rows are byte-compatible with base-library rows (int8
        fingerprints, float32 ADMET in ``ADMET_COLUMNS`` order), so the
        candidate->library remap and its assertions in ``step`` still hold.
        """
        parents = self.get_pareto_front()["smiles"]
        if not parents:
            print(f"[Densify iter {iteration}] no Pareto front yet; nothing to do.")
            return

        analogs = generate_analogs(
            parents,
            n_per_parent=self.densify_per_parent,
            seed=self.seed + iteration,
            exclude=self._library_canonical,
        )
        n_proposed = len(analogs)

        # Apply the SAME PAINS+synthesizability screen the base library passes
        # through (data._apply_quality_filter -> quality_filter.passes_quality),
        # BEFORE the expensive ADMET pipeline. This is the load-bearing bit: the
        # densify ablation showed generated analogs of an anthraquinone-sulfonate
        # PAINS parent were what boosted the front. Without this gate, analogs
        # could bypass a filter every library molecule respects.
        quality_kept = []
        n_pains_rej = 0
        n_synth_rej = 0
        n_parse_rej = 0
        for smi in analogs:
            ok, reason = quality_filter.passes_quality(smi)
            if ok:
                quality_kept.append(smi)
                continue
            if reason == "unparseable":
                n_parse_rej += 1
            elif reason and reason.startswith("PAINS"):
                n_pains_rej += 1
            else:
                n_synth_rej += 1
        n_quality_rej = n_pains_rej + n_synth_rej + n_parse_rej
        analogs = quality_kept

        # Respect the pool cap up front so we never ADMET-score analogs we cannot
        # keep. densify_max_pool bounds the TOTAL library size.
        room = None
        if self.densify_max_pool is not None:
            room = max(0, self.densify_max_pool - self.library_size)
            if room == 0:
                print(f"[Densify iter {iteration}] parents={len(parents)}, "
                      f"proposed={n_proposed}, quality_rejected={n_quality_rej} "
                      f"({n_pains_rej} PAINS + {n_synth_rej} synth + "
                      f"{n_parse_rej} unparseable), passed_filter=0, added=0 "
                      f"(pool cap {self.densify_max_pool} reached), "
                      f"library_size={self.library_size}")
                return

        n_passed = 0
        add_smiles, add_fp, add_admet = [], [], []
        if analogs:
            processed = process_smiles(analogs, oracle=self._oracle)
            n_passed = processed.n_final
            surv_admet = processed.admet_df[LIBRARY_ADMET_COLUMNS].to_numpy(
                dtype=np.float32
            )
            # Defensive novelty dedup (process_smiles preserves SMILES identity,
            # but two analogs can canonicalize together): keep first occurrence
            # of anything not already in the library.
            for k, smiles in enumerate(processed.smiles):
                canon = canonical_smiles(smiles)
                if canon is None or canon in self._library_canonical:
                    continue
                if room is not None and len(add_smiles) >= room:
                    break
                self._library_canonical.add(canon)
                add_smiles.append(smiles)
                add_fp.append(processed.fingerprints[k])
                add_admet.append(surv_admet[k])

        if add_smiles:
            self.smiles = list(self.smiles) + add_smiles
            self.fingerprints = np.vstack([self.fingerprints, np.asarray(add_fp)])
            self.admet_scores = np.vstack([self.admet_scores, np.asarray(add_admet)])
            self.library_size = len(self.smiles)

        print(f"[Densify iter {iteration}] parents={len(parents)}, "
              f"proposed={n_proposed}, quality_rejected={n_quality_rej} "
              f"({n_pains_rej} PAINS + {n_synth_rej} synth + "
              f"{n_parse_rej} unparseable), quality_accepted={len(analogs)}, "
              f"passed_filter={n_passed}, added={len(add_smiles)}, "
              f"library_size={self.library_size}")

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
        Y, Y_raw, docking = self._evaluate(init_indices)

        self.evaluated_indices = list(init_indices)
        self.Y_evaluated = Y
        self.raw_docking = Y_raw

        print(f"Initialized {self.n_init} molecules; "
              f"docked {docked_summary(docking, self.n_init)}.")

    def step(self):
        """Run one BO iteration: train, select, dock, record."""
        iteration = len(self.history) + 1

        # --- Train the GP on everything fully evaluated so far ---
        # Restrict to rows that are finite across the currently-active objectives
        # (a failed dock leaves a NaN that would poison per-column standardization
        # and, for the coregionalized model, is disallowed outright). This makes
        # both the independent and coregionalized train_fn robust to dock failures.
        active = get_active_objectives(self.Y_evaluated)
        eval_idx = np.asarray(self.evaluated_indices)
        finite_rows = np.isfinite(self.Y_evaluated[:, active]).all(axis=1)
        # These fully-evaluated molecules are BOTH the GP's training set and the
        # baseline front the qNEHVI acquisition scores against; keep their library
        # indices so the acquisition can look up their known-exact ADMET rows.
        baseline_library_indices = eval_idx[finite_rows]
        train_x = self.fingerprints[baseline_library_indices]
        train_y = self.Y_evaluated[finite_rows].astype(np.float32)
        print(f"\n[Iteration {iteration}] Training GP on "
              f"{int(finite_rows.sum())}/{len(self.evaluated_indices)} "
              f"fully-evaluated molecules...")
        model, likelihood, y_mean, y_std = self.train_fn(
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

        # --- qNEHVI batch selection ---
        # Grey-box acquisition: the GP scores the docking objectives, while the
        # candidates' and baseline's KNOWN-EXACT ADMET rows (straight from the
        # cached library, in data.ADMET_COLUMNS order) are folded in exactly by
        # the composite objective — no ADMET value is ever read from the GP.
        # select_batch returns indices into X_candidates (the candidate array),
        # NOT into the full library. Map them back via candidate_library_indices.
        candidate_admet = self.admet_scores[candidate_library_indices]
        baseline_admet = self.admet_scores[baseline_library_indices]
        selected_local, selected_ehvi = select_batch(
            model, likelihood, y_mean, y_std,
            X_candidates, candidate_admet,
            train_x, baseline_admet,
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

        # --- Dock the selected batch (against every target) ---
        Y_new, Y_raw_new, docking_new = self._evaluate(list(selected_library_indices))
        batch_docked = docked_summary(docking_new, len(selected_library_indices))
        self._warn_if_docking_saturates(Y_new)

        self.evaluated_indices.extend(int(i) for i in selected_library_indices)
        self.Y_evaluated = np.vstack([self.Y_evaluated, Y_new])
        self.raw_docking = np.vstack([self.raw_docking, Y_raw_new])

        # --- Track Pareto front + hypervolume + size-drift monitor ---
        pareto_mask = self._pareto_mask()
        pareto_size = int(pareto_mask.sum())
        hypervolume = self._hypervolume()
        # Drift monitor: median/min heavy-atom count of the CURRENT front. The LE
        # objective rewards small molecules, so a front sliding toward the fragment
        # floor is the early-warning signal, recorded every iteration.
        pareto_rows = np.where(pareto_mask)[0]
        pareto_smiles = [self.smiles[self.evaluated_indices[r]] for r in pareto_rows]
        pareto_median_heavy, pareto_min_heavy = heavy_atom_stats(pareto_smiles)

        self.history.append({
            "iteration": iteration,
            "n_evaluated": len(self.evaluated_indices),
            "pareto_size": pareto_size,
            "hypervolume": hypervolume,
            "pareto_median_heavy": pareto_median_heavy,
            "pareto_min_heavy": pareto_min_heavy,
            "batch_indices": [int(i) for i in selected_library_indices],
            "batch_ehvi_scores": [float(s) for s in selected_ehvi],
        })

        print(f"[Iteration {iteration}] "
              f"evaluated={len(self.evaluated_indices)}, "
              f"batch={len(selected_library_indices)}, "
              f"docked_this_batch=[{batch_docked}], "
              f"pareto_size={pareto_size}, hypervolume={hypervolume:.4f}, "
              f"pareto_median_heavy={pareto_median_heavy:.0f}")

    def run(self):
        """Run the complete loop: initialize, then ``n_iterations`` steps.

        When densification is enabled, each iteration (except the last, which has
        no successor to select the new molecules) is followed by ``_densify``,
        which grows the candidate pool with analogs of the current front.
        """
        self.initialize()
        for iteration in range(1, self.n_iterations + 1):
            self.step()
            if (self.densify and iteration < self.n_iterations
                    and iteration % self.densify_every == 0):
                self._densify(iteration)

        final = self.history[-1] if self.history else {}
        print("\n=== BO run complete ===")
        print(f"  Total molecules evaluated: {len(self.evaluated_indices)}")
        print(f"  Final Pareto front size:   {final.get('pareto_size', 0)}")
        print(f"  Final hypervolume:         {final.get('hypervolume', 0.0):.4f}")
        # Size-drift summary: the final front's heavy-atom distribution + a flag if
        # its median has slid toward the fragment floor (LE over-correction).
        line, med, flagged = pareto_heavy_summary(self.get_pareto_front()["smiles"])
        print(f"  {line}")
        if flagged:
            print(f"  WARNING: Pareto median heavy-atom count {med:.0f} < "
                  f"{FRAGMENT_MEDIAN_WARN} — front drifting toward FRAGMENTS; the LE "
                  "objective may be over-corrected. Investigate before trusting it.")
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
                "pareto_median_heavy": h.get("pareto_median_heavy", float("nan")),
                "pareto_min_heavy": h.get("pareto_min_heavy", float("nan")),
            }
            for h in self.history
        ])
        history_path = os.path.join(output_dir, "history.csv")
        history_df.to_csv(history_path, index=False)

        # evaluated.csv — every evaluated molecule with all objectives (docking
        # columns are ligand efficiency), the RAW docking kcal retained as
        # ``*_kcal`` columns, plus the reported-only Selectivity Index.
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
        # docking kcal (``*_kcal``) and the Selectivity Index (hDHFR - PfDHFR;
        # higher = more parasite-selective).
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the multi-objective BO loop for PfDHFR drug discovery."
    )
    parser.add_argument("--library-dir", default="data/library")
    parser.add_argument(
        "--smoke", action="store_true",
        help="Fast end-to-end smoke test (n_init=5, batch_size=2, "
             "n_iterations=2). Without it the loop runs the Grand Campaign "
             "(n_init=40, batch_size=5, n_iterations=50). Explicit --n-init / "
             "--batch-size / --n-iterations override whichever profile applies.")
    # Default None so we can tell "user gave a value" from "use the profile":
    # --smoke picks SMOKE_PARAMS, otherwise GRAND_CAMPAIGN_PARAMS; an explicit
    # flag always wins over both (resolved after parse_args below).
    parser.add_argument("--n-init", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-iterations", type=int, default=None)
    parser.add_argument("--mogp-iters", type=int, default=200)
    parser.add_argument("--model", choices=MODEL_CHOICES, default=DEFAULT_MODEL,
                        help="GP model over the docking objectives: "
                             "coregionalized (ICM, primary) or independent.")
    parser.add_argument("--rank", type=int, default=1,
                        help="IndexKernel rank for the coregionalized model.")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument(
        "--densify", action="store_true",
        help="Grow candidates by enumerating analogs of the Pareto front each "
             "iteration (off by default; the base benchmark is unchanged).",
    )
    parser.add_argument("--densify-every", type=int, default=1,
                        help="Densify every N iterations (default 1).")
    parser.add_argument("--densify-per-parent", type=int, default=20,
                        help="Target analogs generated per front molecule.")
    parser.add_argument("--densify-max-pool", type=int, default=None,
                        help="Cap the TOTAL library size after densification (not "
                             "the number of analogs added). Must be ABOVE the "
                             "current library size or densification is a no-op; "
                             "omit for no cap.")
    args = parser.parse_args()

    # Resolve the run profile: --smoke -> SMOKE_PARAMS, else the Grand Campaign;
    # an explicit --n-init / --batch-size / --n-iterations overrides the profile.
    profile = SMOKE_PARAMS if args.smoke else GRAND_CAMPAIGN_PARAMS
    n_init = args.n_init if args.n_init is not None else profile["n_init"]
    batch_size = args.batch_size if args.batch_size is not None else profile["batch_size"]
    n_iterations = (args.n_iterations if args.n_iterations is not None
                    else profile["n_iterations"])

    start = time.time()

    profile_name = "SMOKE" if args.smoke else "GRAND CAMPAIGN"
    print(f"Running BO loop [{profile_name}] with the {args.model!r} GP model"
          + (f" (rank {args.rank})" if args.model == "coregionalized" else "") + ".")
    print(f"  n_init={n_init}, batch_size={batch_size}, n_iterations={n_iterations}")
    loop = BOLoop(
        library_dir=args.library_dir,
        n_init=n_init,
        batch_size=batch_size,
        n_iterations=n_iterations,
        mogp_train_iters=args.mogp_iters,
        model=args.model,
        coregionalization_rank=args.rank,
        densify=args.densify,
        densify_every=args.densify_every,
        densify_per_parent=args.densify_per_parent,
        densify_max_pool=args.densify_max_pool,
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
