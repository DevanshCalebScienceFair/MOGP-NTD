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
panels showing each method's mean line, its individual per-seed traces, and a
shaded band + a final-hypervolume table) plus a CSV of the aggregated numbers.
The ``--band`` flag selects the band type (``std`` / ``sem`` / ``ci95``).

Because every method runs on the SAME seeds, the final hypervolumes are PAIRED
across methods, so we also run a paired significance test (Wilcoxon signed-rank,
primary; paired t-test, secondary) of MOGP vs each baseline, printing a
significance table and saving it to ``benchmark_seeds_significance.csv``.

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
from scipy import stats

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


def method_curves(output_dir, method_key, seeds, ycol):
    """Per-seed curves of ``ycol`` vs molecules evaluated for one method.

    Runs are aligned by iteration index (position in history) and truncated to
    the shortest seed's length, so a method that stops early on some seed still
    aggregates cleanly. The x value at each index is the across-seed mean
    ``n_evaluated`` (identical across seeds for the fixed-budget methods).

    Returns:
        ``(x, y_stack, n_seeds)`` where ``x`` is a length-``min_len`` array and
        ``y_stack`` is a ``(n_seeds, min_len)`` array of per-seed traces (both
        empty if no seed produced a history).
    """
    histories = [
        h for h in (_load_history(output_dir, method_key, s) for s in seeds)
        if h is not None
    ]
    if not histories:
        return np.array([]), np.empty((0, 0)), 0

    min_len = min(len(h) for h in histories)
    x_stack = np.stack([h["n_evaluated"].to_numpy()[:min_len] for h in histories])
    y_stack = np.stack([h[ycol].to_numpy()[:min_len] for h in histories])
    return x_stack.mean(axis=0), y_stack, len(histories)


def band_halfwidth(y_stack, band):
    """Half-width of the shaded band per x point, for the chosen ``band`` type.

    ``std``  — ±1 sample std (ddof=0; a single seed yields 0, not NaN).
    ``sem``  — ±1 standard error of the mean = std(ddof=1)/sqrt(n).
    ``ci95`` — 95% CI ≈ mean ± 1.96·sem (normal approximation).

    ``sem``/``ci95`` need ≥2 seeds; with <2 they collapse to 0 (no band).
    """
    n = y_stack.shape[0]
    if band == "std":
        return y_stack.std(axis=0, ddof=0)
    if n < 2:
        return np.zeros(y_stack.shape[1])
    sem = y_stack.std(axis=0, ddof=1) / np.sqrt(n)
    return sem if band == "sem" else 1.96 * sem


def aggregate_method(output_dir, method_key, seeds, ycol, band="std"):
    """Mean and band half-width of ``ycol`` vs evaluated across seeds.

    Returns ``(x, mean, half, n_seeds)`` numpy arrays (empty if no seed produced
    a history); ``half`` is the ``band``-type half-width from
    :func:`band_halfwidth`. Back-compat: ``band`` defaults to ``"std"``.
    """
    x, y_stack, k = method_curves(output_dir, method_key, seeds, ycol)
    if k == 0:
        return np.array([]), np.array([]), np.array([]), 0
    return x, y_stack.mean(axis=0), band_halfwidth(y_stack, band), k


def final_hv_by_seed(output_dir, method_key, seeds):
    """Ordered ``(seed, final_hypervolume)`` pairs for seeds that produced one.

    Preserves ``seeds`` order so callers can pair two methods by seed. Seeds
    whose run left no history are omitted (not zero-filled).
    """
    pairs = []
    for s in seeds:
        h = _load_history(output_dir, method_key, s)
        if h is not None:
            pairs.append((s, float(h["hypervolume"].iloc[-1])))
    return pairs


def final_hypervolume_stats(output_dir, method_key, seeds):
    """Return ``(mean, std, n_seeds, per_seed_list)`` of final hypervolume."""
    finals = [v for _, v in final_hv_by_seed(output_dir, method_key, seeds)]
    if not finals:
        return float("nan"), float("nan"), 0, []
    arr = np.asarray(finals, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0)), len(arr), finals


# ---------------------------------------------------------------------- #
# Paired significance tests on final hypervolume
# ---------------------------------------------------------------------- #
# Wilcoxon signed-rank has essentially no power below a handful of paired
# samples (its smallest attainable two-sided p is 2 / 2**n). Warn below this.
MIN_SEEDS_FOR_POWER = 6

MOGP_KEY = "mogp"


def paired_significance(output_dir, seeds):
    """Paired MOGP-vs-baseline tests on per-seed final hypervolume.

    For each baseline, the MOGP and baseline final-hypervolume vectors are
    paired by seed (only seeds where BOTH methods produced a history are used,
    matched in ``seeds`` order). Runs a Wilcoxon signed-rank test (primary) and
    a paired t-test (secondary) on the paired differences ``mogp - baseline``.

    Returns a list of dicts, one per baseline, with the mean paired difference,
    both test statistics/p-values, the number of paired seeds, and whether the
    difference is significant (Wilcoxon p < 0.05). Degenerate cases (too few
    pairs, all-zero differences) are reported with NaN p-values and a note
    rather than raising.
    """
    mogp_map = dict(final_hv_by_seed(output_dir, MOGP_KEY, seeds))
    results = []
    for label, key, _ in METHODS:
        if key == MOGP_KEY:
            continue
        base_map = dict(final_hv_by_seed(output_dir, key, seeds))
        common = [s for s in seeds if s in mogp_map and s in base_map]
        mogp_vals = np.asarray([mogp_map[s] for s in common], dtype=float)
        base_vals = np.asarray([base_map[s] for s in common], dtype=float)
        diff = mogp_vals - base_vals
        n = len(common)

        rec = {
            "comparison": f"MOGP vs {label}",
            "baseline": label,
            "n_pairs": n,
            "mean_diff": float(diff.mean()) if n else float("nan"),
            "wilcoxon_stat": float("nan"),
            "wilcoxon_p": float("nan"),
            "ttest_stat": float("nan"),
            "ttest_p": float("nan"),
            "significant": False,
            "note": "",
        }

        if n < 2:
            rec["note"] = "too few paired seeds (<2) to test"
            results.append(rec)
            continue
        if np.allclose(diff, 0.0):
            rec["note"] = "all paired differences are zero; no test applicable"
            results.append(rec)
            continue

        # Wilcoxon signed-rank (primary). Zero differences are dropped by the
        # default 'wilcox' zero-method, so an effective n below can be < n.
        try:
            w_stat, w_p = stats.wilcoxon(mogp_vals, base_vals)
            rec["wilcoxon_stat"] = float(w_stat)
            rec["wilcoxon_p"] = float(w_p)
        except ValueError as exc:
            rec["note"] = f"Wilcoxon skipped: {exc}"

        # Paired t-test (secondary).
        t_stat, t_p = stats.ttest_rel(mogp_vals, base_vals)
        rec["ttest_stat"] = float(t_stat)
        rec["ttest_p"] = float(t_p)

        rec["significant"] = bool(rec["wilcoxon_p"] < 0.05)
        if n < MIN_SEEDS_FOR_POWER:
            rec["note"] = (f"only {n} paired seeds (< {MIN_SEEDS_FOR_POWER}); "
                           "low power — add more seeds before trusting p")
        results.append(rec)
    return results


def print_significance_table(output_dir, seeds):
    """Print the paired MOGP-vs-baseline significance table to stdout.

    Returns the ``paired_significance`` records so callers can also persist them.
    """
    recs = paired_significance(output_dir, seeds)
    bar = "=" * 78
    print("\n" + bar)
    print("PAIRED SIGNIFICANCE — final hypervolume, MOGP vs each baseline")
    print("(paired by seed; Wilcoxon signed-rank primary, paired t-test secondary)")
    print(bar)

    n_pairs_max = max((r["n_pairs"] for r in recs), default=0)
    if 0 < n_pairs_max < MIN_SEEDS_FOR_POWER:
        print(f"WARNING: only {n_pairs_max} paired seed(s) — below the ~"
              f"{MIN_SEEDS_FOR_POWER} needed for meaningful Wilcoxon power.")
        print("         Treat p-values as indicative only; add more seeds.\n")

    header = (f"{'Comparison':<22}{'n':>3}{'mean Δhv':>12}"
              f"{'Wilcoxon p':>13}{'t-test p':>12}")
    print(header)
    print("-" * len(header))
    for r in recs:
        if r["n_pairs"] < 2 or (np.isnan(r["wilcoxon_p"]) and np.isnan(r["ttest_p"])):
            print(f"{r['comparison']:<22}{r['n_pairs']:>3}"
                  f"{r['mean_diff']:>12.4f}{'n/a':>13}{'n/a':>12}")
        else:
            wp = "n/a" if np.isnan(r["wilcoxon_p"]) else f"{r['wilcoxon_p']:.4f}"
            print(f"{r['comparison']:<22}{r['n_pairs']:>3}"
                  f"{r['mean_diff']:>12.4f}{wp:>13}{r['ttest_p']:>12.4f}")
    print("-" * len(header))

    # Plain-language verdicts.
    for r in recs:
        if r["n_pairs"] < 2 or np.isnan(r["wilcoxon_p"]):
            print(f"  {r['comparison']}: inconclusive — {r['note']}")
            continue
        direction = "higher" if r["mean_diff"] > 0 else "lower"
        verb = "IS" if r["significant"] else "is NOT"
        print(f"  {r['comparison']}: MOGP's final hypervolume {verb} significantly "
              f"{direction} (mean Δ = {r['mean_diff']:+.4f}, "
              f"Wilcoxon p = {r['wilcoxon_p']:.4f}).")
        if r["note"]:
            print(f"      ⚠ {r['note']}")
    print(bar)
    return recs


def save_significance_csv(recs, output_dir, csv_path=None):
    """Persist the paired significance records to a sibling CSV."""
    if csv_path is None:
        csv_path = os.path.join(output_dir, "benchmark_seeds_significance.csv")
    cols = ["comparison", "baseline", "n_pairs", "mean_diff",
            "wilcoxon_stat", "wilcoxon_p", "ttest_stat", "ttest_p",
            "significant", "note"]
    df = pd.DataFrame.from_records(recs, columns=cols)
    df.to_csv(csv_path, index=False)
    print(f"Saved significance table to {csv_path}")
    return csv_path


# ---------------------------------------------------------------------- #
# Figure + table
# ---------------------------------------------------------------------- #
BAND_LABELS = {
    "std": "±1 std",
    "sem": "±1 s.e.m.",
    "ci95": "95% CI",
}


def save_figure(output_dir, seeds, fig_path=None, band="std"):
    """Save the aggregated figure: hv + Pareto curves + final-hypervolume table.

    Each method draws its across-seed mean as a bold marker line, its individual
    per-seed traces as thin low-alpha lines behind it (so spread is visible, not
    just summarized), and a shaded ``band``-type band (``std``/``sem``/``ci95``).

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
            x, y_stack, k = method_curves(output_dir, key, seeds, ycol)
            if k == 0:
                continue
            # Individual per-seed traces (thin, low-alpha) behind the mean.
            for row in y_stack:
                ax.plot(x, row, color=color, alpha=0.22, linewidth=0.8, zorder=1)
            mean = y_stack.mean(axis=0)
            half = band_halfwidth(y_stack, band)
            ax.plot(x, mean, color=color, marker="o", label=label, zorder=3)
            ax.fill_between(x, mean - half, mean + half, color=color,
                            alpha=0.18, zorder=2)
            ax.set_ylabel(ylabel)
        # count only once (via hv curve presence)
        _, _, k_hv = method_curves(output_dir, key, seeds, "hypervolume")
        if k_hv:
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
                 f"(mean lines, per-seed traces, {BAND_LABELS[band]} bands)",
                 fontsize=14)
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
    parser.add_argument(
        "--band", choices=["std", "sem", "ci95"], default="std",
        help="Shaded band on the curves: 'std' = ±1 std (default, back-compat), "
             "'sem' = ±1 standard error, 'ci95' = 95%% CI (mean ± 1.96·sem). "
             "sem/ci95 are more honest for few seeds (they shrink as seeds are "
             "added, whereas std does not).",
    )
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
    sig_recs = print_significance_table(args.output_dir, args.seeds)
    save_aggregate_csv(args.output_dir, args.seeds)
    save_significance_csv(sig_recs, args.output_dir)
    save_figure(args.output_dir, args.seeds, band=args.band)

    print("\nPer-method total time across all seeds:")
    for label, _, _ in METHODS:
        print(f"  {label:<18}{fmt_time(elapsed_by_method[label])}")
    print(f"\nTotal wall-clock time: {fmt_time(time.time() - overall_start)}")


if __name__ == "__main__":
    main()
