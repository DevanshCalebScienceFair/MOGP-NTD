"""
run_ablation.py
===============

Multi-seed ablation: **independent** vs **coregionalized (ICM)** GP, head to
head, on the SAME seeds / library / parameters. Only the GP model differs; the
grey-box qNEHVI acquisition, the docking oracle and the evaluation frame are
identical, so any difference in the final hypervolume is attributable to the
cross-task (PfDHFR/hDHFR) structure the coregionalized model captures and the
independent model forces to zero.

Because a single BO run is noisy (random init, MC acquisition, stochastic
docking), one run tells you little. This harness repeats each arm across several
seeds and reports the final hypervolume as **mean ± std**, so the comparison is
statistically meaningful rather than a single lucky/unlucky run. Hypervolume is
read from each run's history, i.e. ``evaluation.compute_hypervolume`` — the
single source of truth shared with every baseline — so the numbers are directly
comparable.

For each seed, both arms start from the identical random initial molecules (same
seed -> same ``np.random`` draw) and get the same docking budget
(``n_init + n_iterations * batch_size``), so the comparison is fair.

Run with::

    python run_ablation.py --seeds 0,1,2,3,4
    python run_ablation.py --seeds 0,1,2 --n-init 10 --batch-size 10 \
        --n-iterations 8 --rank 1 --save

Docking dominates the wall clock, and this runs BOTH arms for EVERY seed, so
scale the budget accordingly. ``--save`` writes each arm's last-seed run to its
own results directory for validate_known_actives.py.
"""

# KMP_DUPLICATE_LIB_OK must be set BEFORE numpy/torch import their native
# libraries, otherwise macOS aborts with the "libomp already initialized" error.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import time

import numpy as np

from loop import BOLoop, MODEL_CHOICES
import evaluation


# Where each arm's representative (last-seed) run is written when --save is set.
# Coregionalized matches validate_known_actives.py's default; independent uses a
# distinct dir so it does not clobber the headline run in "results".
RESULTS_DIRS = {
    "coregionalized": "results_coregionalized",
    "independent": "results_independent",
}


def run_arm(model, seed, cfg, save_dir=None):
    """Run one BO arm (one model, one seed) and return its final metrics.

    Args:
        model: ``"coregionalized"`` or ``"independent"``.
        seed: Random seed shared by both arms for this repeat.
        cfg: Dict of loop parameters (n_init, batch_size, n_iterations,
            mogp_iters, rank, library_dir, and optional max_library).
        save_dir: If given, also write the run's three result CSVs there.

    Returns:
        Dict with ``hypervolume``, ``pareto_size``, ``n_evaluated``, ``seed``.
    """
    loop = BOLoop(
        library_dir=cfg["library_dir"],
        seed=seed,
        n_init=cfg["n_init"],
        batch_size=cfg["batch_size"],
        n_iterations=cfg["n_iterations"],
        mogp_train_iters=cfg["mogp_iters"],
        model=model,
        coregionalization_rank=cfg["rank"],
    )

    # Optional library truncation for quick runs (keeps the candidate scan small).
    max_library = cfg.get("max_library")
    if max_library is not None and max_library < loop.library_size:
        loop.smiles = loop.smiles[:max_library]
        loop.fingerprints = loop.fingerprints[:max_library]
        loop.admet_scores = loop.admet_scores[:max_library]
        loop.library_size = max_library

    history = loop.run()
    if save_dir is not None:
        loop.save_results(output_dir=save_dir)

    # Final hypervolume is evaluation.compute_hypervolume (the single source of
    # truth) as recorded in the loop history; recompute defensively if empty.
    if history:
        final = history[-1]
        hv = float(final["hypervolume"])
        pareto = int(final["pareto_size"])
        n_eval = int(final["n_evaluated"])
    else:
        hv = evaluation.compute_hypervolume(loop.Y_evaluated)
        pareto = int(loop._pareto_mask().sum())
        n_eval = len(loop.evaluated_indices)
    return {"hypervolume": hv, "pareto_size": pareto,
            "n_evaluated": n_eval, "seed": seed}


def run_ablation(seeds, cfg, models=MODEL_CHOICES, save=False):
    """Run every model across every seed; return per-model metric lists.

    Args:
        seeds: Iterable of integer seeds. Both arms use the SAME seed per repeat.
        cfg: Loop parameters (see ``run_arm``).
        models: Model names to compare (default both arms).
        save: If True, write each model's LAST-seed run to ``RESULTS_DIRS[model]``.

    Returns:
        Dict ``{model: {"hypervolume": [...], "pareto_size": [...],
        "seeds": [...]}}`` aligned by seed order.
    """
    seeds = list(seeds)
    models = list(models)
    results = {m: {"hypervolume": [], "pareto_size": [], "seeds": []}
               for m in models}

    for seed in seeds:
        for model in models:
            save_dir = (RESULTS_DIRS.get(model)
                        if (save and seed == seeds[-1]) else None)
            print(f"\n--- model={model!r}  seed={seed} ---")
            metrics = run_arm(model, seed, cfg, save_dir=save_dir)
            results[model]["hypervolume"].append(metrics["hypervolume"])
            results[model]["pareto_size"].append(metrics["pareto_size"])
            results[model]["seeds"].append(seed)
            print(f"    -> hypervolume={metrics['hypervolume']:.4f}  "
                  f"pareto={metrics['pareto_size']}  "
                  f"evaluated={metrics['n_evaluated']}")
    return results


def summarize(results):
    """Print a mean ± std hypervolume comparison across seeds for each model."""
    bar = "=" * 68
    print("\n" + bar)
    print("ABLATION SUMMARY — final hypervolume (evaluation.compute_hypervolume)")
    print(bar)
    print(f"{'Model':<16}{'n_seeds':>9}{'HV mean':>11}{'HV std':>10}"
          f"{'HV min':>10}{'HV max':>10}")

    stats = {}
    for model, data in results.items():
        hv = np.asarray(data["hypervolume"], dtype=float)
        if hv.size == 0:
            print(f"{model:<16}{'—':>9}{'(no runs)':>11}")
            continue
        stats[model] = hv
        print(f"{model:<16}{hv.size:>9}{hv.mean():>11.4f}{hv.std():>10.4f}"
              f"{hv.min():>10.4f}{hv.max():>10.4f}")
    print(bar)

    # Head-to-head delta when both arms ran on matching seeds.
    if {"coregionalized", "independent"} <= set(stats):
        cor, ind = stats["coregionalized"], stats["independent"]
        if cor.size == ind.size:
            delta = cor - ind
            print(f"\nPaired delta (coregionalized - independent) per seed: "
                  f"mean {delta.mean():+.4f} ± {delta.std():.4f}")
            wins = int((delta > 0).sum())
            print(f"Coregionalized >= independent on "
                  f"{int((delta >= 0).sum())}/{delta.size} seeds "
                  f"(strictly better on {wins}).")
            if delta.mean() > 0:
                print("=> Coregionalized (ICM) yields the higher mean hypervolume.")
            elif delta.mean() < 0:
                print("=> Independent yields the higher mean hypervolume here "
                      "(inspect scale/seeds; ICM helps most with real docking signal).")
            else:
                print("=> The two arms tie on average.")
    return stats


def _parse_seeds(raw):
    """Parse ``--seeds``: a comma list ('0,1,2') or a count ('5' -> 0..4)."""
    raw = raw.strip()
    if "," in raw:
        return [int(s) for s in raw.split(",") if s.strip() != ""]
    n = int(raw)
    return list(range(n))


def main():
    parser = argparse.ArgumentParser(
        description="Ablation: independent vs coregionalized GP across seeds."
    )
    parser.add_argument("--seeds", default="0,1,2",
                        help="Comma list of seeds ('0,1,2') or a count ('5' -> 0..4).")
    parser.add_argument("--library-dir", default="data/library")
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--n-iterations", type=int, default=8)
    parser.add_argument("--mogp-iters", type=int, default=200)
    parser.add_argument("--rank", type=int, default=1,
                        help="IndexKernel rank for the coregionalized arm.")
    parser.add_argument("--max-library", type=int, default=None,
                        help="Optional cap on library size (for quick runs).")
    parser.add_argument("--models", default=",".join(MODEL_CHOICES),
                        help="Comma list of models to compare.")
    parser.add_argument("--save", action="store_true",
                        help="Save each arm's last-seed run for validate_known_actives.py.")
    args = parser.parse_args()

    seeds = _parse_seeds(args.seeds)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for m in models:
        if m not in MODEL_CHOICES:
            parser.error(f"unknown model {m!r}; choose from {MODEL_CHOICES}")

    cfg = {
        "library_dir": args.library_dir,
        "n_init": args.n_init,
        "batch_size": args.batch_size,
        "n_iterations": args.n_iterations,
        "mogp_iters": args.mogp_iters,
        "rank": args.rank,
        "max_library": args.max_library,
    }

    print("=" * 68)
    print("RUN ABLATION — independent vs coregionalized (ICM)")
    print("=" * 68)
    print(f"seeds={seeds}  models={models}  rank={args.rank}")
    print(f"n_init={args.n_init}  batch_size={args.batch_size}  "
          f"n_iterations={args.n_iterations}  mogp_iters={args.mogp_iters}")

    start = time.time()
    results = run_ablation(seeds, cfg, models=models, save=args.save)
    summarize(results)
    print(f"\nTotal wall-clock time: {time.time() - start:.1f}s")

    if args.save:
        saved = ", ".join(f"{m} -> {RESULTS_DIRS[m]}/" for m in models
                          if m in RESULTS_DIRS)
        print(f"\nSaved last-seed runs: {saved}")
        print("Compare recovery of known actives with:")
        print("  python validate_known_actives.py "
              f"--independent-dir {RESULTS_DIRS.get('independent', 'results_independent')} "
              f"--coregionalized-dir {RESULTS_DIRS.get('coregionalized', 'results_coregionalized')}")


if __name__ == "__main__":
    main()
