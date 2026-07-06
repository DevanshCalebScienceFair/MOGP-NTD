"""
run_benchmark_seeds.py
======================

Multi-seed benchmark harness. Runs all four optimization methods across a list
of random seeds and aggregates their learning curves with **mean ± std** bands,
so single-run noise can no longer masquerade as a real difference between
methods.

The four methods (the same runner classes ``run_all.py`` drives):

    1. MOGP           — multi-output GP + EHVI                     (loop.BOLoop)
    2. Random Search  — uniform random batches      (RandomSearchBaseline)
    3. Single-Obj BO  — single-output GP + EI on docking (SingleObjectiveBOLoop)
    4. Greedy Filter  — hard ADMET cutoffs then dock     (GreedyFilterThenDock)

For each seed we run every method **with that same seed**, so the initial random
molecule set matches across the stochastic methods and the comparison is fair.
Per-seed results are written to ``<output_dir>/<method>/seed_<seed>/`` (the usual
three CSVs). We then aggregate, across seeds, two curves per method:

    * hypervolume vs molecules evaluated
    * Pareto-front size vs molecules evaluated

reporting the mean and ±1 std across seeds, and save a single figure (two curve
panels with shaded std bands + a final-hypervolume table) plus a CSV of the
aggregated numbers.

Hypervolume is NEVER recomputed here: each run already records it through
``evaluation.compute_hypervolume`` (the single source of truth) into its
``history.csv``, and this harness only reads those columns.

Run with, e.g.::

    python run_benchmark_seeds.py --seeds 0 1 2 --lib-size 1000 \
        --n-init 10 --batch-size 10 --n-iterations 10 --mogp-iters 200
"""

# KMP_DUPLICATE_LIB_OK must be set BEFORE numpy/torch/rdkit import native libs.
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import argparse

import numpy as np
import pandas as pd

from loop import BOLoop
from baseline_random import RandomSearchBaseline
from baseline_single_obj import SingleObjectiveBOLoop
from baseline_greedy import GreedyFilterThenDock
from run_all import ensure_library, LIBRARY_DIR, fmt_time
import docking


# One entry per method: (display label, subdirectory key, plot color). Colors
# match the single-run comparison plots in the baselines.
METHODS = [
    ("MOGP", "mogp", "tab:blue"),
    ("Random Search", "random", "tab:red"),
    ("Single-Obj BO", "single_obj", "tab:orange"),
    ("Greedy Filter", "greedy", "tab:green"),
]


# ---------------------------------------------------------------------- #
# Per-method runner construction. Each builder returns a configured runner with
# a run()+save_results() contract; all take the SAME seed within a seed-run.
# ---------------------------------------------------------------------- #
def _build_runner(method_key, params, seed):
    """Construct the runner for ``method_key`` at ``seed`` with shared params."""
    if method_key == "mogp":
        # Only the MOGP loop supports densification (growing candidates around
        # the Pareto front); the baselines have no acquisition to feed. With
        # --densify off, this is exactly the base MOGP run.
        return BOLoop(
            library_dir=LIBRARY_DIR, seed=seed,
            n_init=params["n_init"], batch_size=params["batch_size"],
            n_iterations=params["n_iterations"],
            mogp_train_iters=params["mogp_iters"],
            densify=params.get("densify", False),
            densify_every=params.get("densify_every", 1),
            densify_per_parent=params.get("densify_per_parent", 20),
            densify_max_pool=params.get("densify_max_pool"),
        )
    if method_key == "random":
        return RandomSearchBaseline(
            library_dir=LIBRARY_DIR, seed=seed,
            n_init=params["n_init"], batch_size=params["batch_size"],
            n_iterations=params["n_iterations"],
        )
    if method_key == "single_obj":
        return SingleObjectiveBOLoop(
            library_dir=LIBRARY_DIR, seed=seed,
            n_init=params["n_init"], batch_size=params["batch_size"],
            n_iterations=params["n_iterations"],
            gp_train_iters=params["mogp_iters"],
        )
    if method_key == "greedy":
        # Greedy has no iteration loop; it docks a budget equal to the total the
        # other methods evaluate (n_init + n_iterations * batch_size).
        return GreedyFilterThenDock(
            library_dir=LIBRARY_DIR, seed=seed,
            batch_size=params["batch_size"], n_total=params["n_total"],
        )
    raise ValueError(f"Unknown method key {method_key!r}")


def seed_run_dir(output_dir, method_key, seed):
    """Directory for one (method, seed) run's CSVs."""
    return os.path.join(output_dir, method_key, f"seed_{seed}")


def run_all_seeds(params, seeds, output_dir):
    """Run every method at every seed, writing per-seed CSVs. Returns timings."""
    elapsed_by_method = {label: 0.0 for label, _, _ in METHODS}

    for seed in seeds:
        print("\n" + "#" * 64)
        print(f"# SEED {seed}")
        print("#" * 64)
        for label, key, _ in METHODS:
            out_dir = seed_run_dir(output_dir, key, seed)
            print("\n" + "=" * 64)
            print(f"[seed {seed}] {label}")
            print("=" * 64)
            start = time.time()
            try:
                runner = _build_runner(key, params, seed)
                runner.run()
                runner.save_results(output_dir=out_dir)
            except Exception as exc:
                print(f"  ERROR: {label} (seed {seed}) failed: {exc}")
            elapsed_by_method[label] += time.time() - start

    return elapsed_by_method


# ---------------------------------------------------------------------- #
# Aggregation across seeds
# ---------------------------------------------------------------------- #
def _load_history(output_dir, method_key, seed):
    """Load one run's history.csv, or None if missing/empty."""
    path = os.path.join(seed_run_dir(output_dir, method_key, seed), "history.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df if len(df) else None


def aggregate_method(output_dir, method_key, seeds, ycol):
    """Mean/std of ``ycol`` vs molecules evaluated across seeds for one method.

    Runs are aligned by iteration index (position in history) and truncated to
    the shortest seed's length, so a method that stops early on some seed still
    aggregates cleanly. The x value at each index is the across-seed mean
    ``n_evaluated`` (identical across seeds for the fixed-budget methods).

    Returns:
        ``(x, mean, std, n_seeds)`` numpy arrays (empty if no seed produced a
        history), where ``n_seeds`` is how many seeds contributed.
    """
    histories = [
        h for h in (_load_history(output_dir, method_key, s) for s in seeds)
        if h is not None
    ]
    if not histories:
        return np.array([]), np.array([]), np.array([]), 0

    min_len = min(len(h) for h in histories)
    x_stack = np.stack([h["n_evaluated"].to_numpy()[:min_len] for h in histories])
    y_stack = np.stack([h[ycol].to_numpy()[:min_len] for h in histories])

    x = x_stack.mean(axis=0)
    mean = y_stack.mean(axis=0)
    # ddof=0 so a single seed yields std 0 (rather than NaN).
    std = y_stack.std(axis=0, ddof=0)
    return x, mean, std, len(histories)


def final_hypervolume_stats(output_dir, method_key, seeds):
    """Return ``(mean, std, n_seeds, per_seed_list)`` of final hypervolume."""
    finals = []
    for s in seeds:
        h = _load_history(output_dir, method_key, s)
        if h is not None:
            finals.append(float(h["hypervolume"].iloc[-1]))
    if not finals:
        return float("nan"), float("nan"), 0, []
    arr = np.asarray(finals, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0)), len(arr), finals


# ---------------------------------------------------------------------- #
# Figure + table
# ---------------------------------------------------------------------- #
def save_figure(output_dir, seeds, fig_path=None):
    """Save the aggregated figure: hv + Pareto curves with ±1 std bands + table.

    Returns the figure path, or None if no method produced any history.
    """
    import matplotlib
    matplotlib.use("Agg")          # headless: write a file, never open a window
    import matplotlib.pyplot as plt

    if fig_path is None:
        fig_path = os.path.join(output_dir, "benchmark_seeds.png")

    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.2], hspace=0.35, wspace=0.25)
    ax_hv = fig.add_subplot(gs[0, 0])
    ax_pareto = fig.add_subplot(gs[0, 1])
    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis("off")

    n_seeds = len(seeds)
    plotted = 0
    for label, key, color in METHODS:
        for ax, ycol, ylabel in (
            (ax_hv, "hypervolume", "Hypervolume"),
            (ax_pareto, "pareto_size", "Pareto-front size"),
        ):
            x, mean, std, k = aggregate_method(output_dir, key, seeds, ycol)
            if len(x) == 0:
                continue
            ax.plot(x, mean, color=color, marker="o", label=label)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.18)
            ax.set_ylabel(ylabel)
        # count only once (via hv curve presence)
        x, _, _, _ = aggregate_method(output_dir, key, seeds, "hypervolume")
        if len(x):
            plotted += 1

    if plotted == 0:
        plt.close(fig)
        print("  (no histories found; skipping figure)")
        return None

    for ax, title in ((ax_hv, "Hypervolume vs evaluated"),
                      (ax_pareto, "Pareto size vs evaluated")):
        ax.set_xlabel("Number of molecules evaluated")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()

    # --- Final-hypervolume table (mean ± std across seeds) ---
    rows = []
    for label, key, _ in METHODS:
        mean, std, k, _ = final_hypervolume_stats(output_dir, key, seeds)
        if k == 0:
            rows.append([label, "—", "0"])
        else:
            rows.append([label, f"{mean:.3f} ± {std:.3f}", str(k)])
    table = ax_table.table(
        cellText=rows,
        colLabels=["Method", "Final hypervolume (mean ± std)", "Seeds"],
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.6)
    ax_table.set_title(f"Final hypervolume across {n_seeds} seed(s): "
                       f"{list(seeds)}", pad=12)

    fig.suptitle("MOGP vs baselines — multi-seed benchmark "
                 "(mean lines, ±1 std bands)", fontsize=14)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved aggregated figure to {fig_path}")
    return fig_path


def save_aggregate_csv(output_dir, seeds, csv_path=None):
    """Write the aggregated per-point mean/std curves to a tidy CSV."""
    if csv_path is None:
        csv_path = os.path.join(output_dir, "benchmark_seeds_aggregate.csv")

    records = []
    for label, key, _ in METHODS:
        for ycol in ("hypervolume", "pareto_size"):
            x, mean, std, k = aggregate_method(output_dir, key, seeds, ycol)
            for xi, mi, si in zip(x, mean, std):
                records.append({
                    "method": label, "metric": ycol,
                    "n_evaluated": xi, "mean": mi, "std": si, "n_seeds": k,
                })
    df = pd.DataFrame.from_records(records)
    df.to_csv(csv_path, index=False)
    print(f"Saved aggregated curves to {csv_path}")
    return csv_path


def print_final_table(output_dir, seeds):
    """Print the final-hypervolume (mean ± std) summary table to stdout."""
    bar = "=" * 60
    print("\n" + bar)
    print(f"FINAL HYPERVOLUME (mean ± std over seeds {list(seeds)})")
    print(bar)
    print(f"{'Method':<18}{'mean ± std':>26}{'seeds':>8}")
    for label, key, _ in METHODS:
        mean, std, k, _ = final_hypervolume_stats(output_dir, key, seeds)
        cell = "—" if k == 0 else f"{mean:.4f} ± {std:.4f}"
        print(f"{label:<18}{cell:>26}{k:>8}")
    print(bar)


# ---------------------------------------------------------------------- #
# CLI
# ---------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed benchmark of MOGP vs baselines with mean±std "
                    "aggregation."
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                        help="Random seeds; every method is run once per seed.")
    parser.add_argument("--lib-size", type=int, default=1000,
                        help="Library pull size (built once, shared by all runs).")
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--n-iterations", type=int, default=10)
    parser.add_argument("--mogp-iters", type=int, default=200)
    parser.add_argument("--output-dir", default="benchmark_seeds_results")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable the persistent docking cache for this run.")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Wipe the docking cache before running (retry failures).")
    parser.add_argument(
        "--densify", action="store_true",
        help="Enable Pareto-front analog densification in the MOGP loop (the "
             "baselines are unaffected). Lets you quantify the hypervolume gain "
             "from densification with error bars across seeds.",
    )
    parser.add_argument("--densify-per-parent", type=int, default=20,
                        help="Target analogs generated per front molecule.")
    parser.add_argument("--densify-max-pool", type=int, default=None,
                        help="Cap the total MOGP library size after densification.")
    args = parser.parse_args()

    if args.clear_cache:
        docking.clear_cache()
        print("Cleared the docking cache.")
    if args.no_cache:
        docking.set_cache_enabled(False)
        print("Docking cache disabled (--no-cache).")

    params = {
        "n_init": args.n_init,
        "batch_size": args.batch_size,
        "n_iterations": args.n_iterations,
        "mogp_iters": args.mogp_iters,
        "n_total": args.n_init + args.n_iterations * args.batch_size,
        "densify": args.densify,
        "densify_per_parent": args.densify_per_parent,
        "densify_max_pool": args.densify_max_pool,
    }

    print("=" * 64)
    print("MULTI-SEED BENCHMARK — MOGP vs baselines")
    print("=" * 64)
    print(f"Seeds:        {args.seeds}")
    print(f"Per method:   {params['n_total']} molecules "
          f"(= {args.n_init} init + {args.n_iterations} x {args.batch_size})")
    print(f"Densify:      {'ON' if args.densify else 'off'}"
          + (f" (per_parent={args.densify_per_parent}, "
             f"max_pool={args.densify_max_pool})" if args.densify else ""))
    print(f"Output dir:   {args.output_dir}/")

    ensure_library(args.lib_size)
    os.makedirs(args.output_dir, exist_ok=True)

    overall_start = time.time()
    elapsed_by_method = run_all_seeds(params, args.seeds, args.output_dir)

    # --- Aggregate + report ---
    print_final_table(args.output_dir, args.seeds)
    save_aggregate_csv(args.output_dir, args.seeds)
    save_figure(args.output_dir, args.seeds)

    print("\nPer-method total time across all seeds:")
    for label, _, _ in METHODS:
        print(f"  {label:<18}{fmt_time(elapsed_by_method[label])}")
    print(f"\nTotal wall-clock time: {fmt_time(time.time() - overall_start)}")


if __name__ == "__main__":
    main()
