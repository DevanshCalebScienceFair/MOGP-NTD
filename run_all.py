"""
run_all.py
==========

One-shot benchmark driver: run all four optimization methods back to back on the
*same* molecule library with *matching* parameters, print a results summary, and
launch the side-by-side comparison dashboard.

The four methods (each writes the same three CSVs to its own results directory):

    1. MOGP            — multi-output GP + EHVI over all five objectives (loop.py)
    2. Random Search   — uniform random batches            (baseline_random.py)
    3. Single-Obj BO   — single-output GP + EI on docking   (baseline_single_obj.py)
    4. Greedy Filter   — hard ADMET cutoffs then dock        (baseline_greedy.py)

Each method exposes a programmatic class with ``run()`` + ``save_results()``, so
this driver imports and calls them directly (no subprocess). All methods share
``n_init``, ``batch_size``, ``n_iterations`` and ``seed``; the greedy baseline
gets a matching docking budget ``n_total = n_init + n_iterations * batch_size``.

Run with::

    python run_all.py
"""

# KMP_DUPLICATE_LIB_OK must be set BEFORE numpy/torch/rdkit import their native
# libraries, otherwise macOS aborts with the "libomp already initialized" error.
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import shutil
import subprocess
import webbrowser

import pandas as pd

from data import load_library, build_library
from loop import BOLoop, DEFAULT_MODEL
from baseline_random import RandomSearchBaseline
from baseline_single_obj import SingleObjectiveBOLoop
from baseline_greedy import GreedyFilterThenDock
from docking import clear_cache


# The four runs: (display name, results directory, terminal-friendly key).
RESULTS_DIRS = {
    "MOGP": "results",
    "Random Search": "baseline_random_results",
    "Single-Obj BO": "baseline_single_obj_results",
    "Greedy Filter": "baseline_greedy_results",
}

LIBRARY_DIR = "data/library"

# Rough per-molecule docking cost (seconds), used only for the up-front estimate.
# Docking (AutoDock Vina, exhaustiveness 8) dominates wall-clock; GP training is
# negligible by comparison.
DOCK_SECONDS_ESTIMATE = 40

# Marker file recording the library pull-size run_all last built, so we can skip
# the (slow, network-bound) rebuild when the cached library already matches.
BUILD_MARKER = os.path.join(LIBRARY_DIR, ".run_all_build_size")


# ---------------------------------------------------------------------- #
# Small input / formatting helpers
# ---------------------------------------------------------------------- #
def ask(label, default, cast=str):
    """Prompt ``"<label> [<default>]: "`` and return the parsed value or default.

    An empty line (the user just hits enter) yields ``default``. A value that
    fails to parse falls back to ``default`` with a note.
    """
    raw = input(f"{label} [{default}]: ").strip()
    if raw == "":
        return cast(default)
    try:
        return cast(raw)
    except ValueError:
        print(f"  invalid value {raw!r}; using default {default}")
        return cast(default)


def ask_yes_no(label, default=True):
    """Prompt a yes/no question; empty input returns ``default``."""
    suffix = "[Y/n]" if default else "[y/N]"
    raw = input(f"{label} {suffix}: ").strip().lower()
    if raw == "":
        return default
    return raw in ("y", "yes")


def fmt_time(seconds):
    """Compact human-readable duration: ``45s`` / ``12.3m`` / ``1.4h``."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


# ---------------------------------------------------------------------- #
# Library preparation
# ---------------------------------------------------------------------- #
def ensure_library(lib_size):
    """Make sure ``data/library`` holds a library built from ``lib_size`` molecules.

    Rebuilds via ``data.build_library`` only when needed — i.e. the library is
    missing, or the marker shows it was last built at a different pull size.
    Building pulls from ChEMBL and runs the ADMET oracle, so it is skipped
    whenever the cached library already matches. If a (re)build fails but a
    library already exists on disk, we warn and reuse it rather than abort.
    """
    smiles_path = os.path.join(LIBRARY_DIR, "smiles.csv")
    have_library = os.path.exists(smiles_path)

    marker_matches = False
    if os.path.exists(BUILD_MARKER):
        try:
            with open(BUILD_MARKER) as fh:
                marker_matches = int(fh.read().strip()) == lib_size
        except (ValueError, OSError):
            marker_matches = False

    if have_library and marker_matches:
        n = len(pd.read_csv(smiles_path))
        print(f"Using cached library at {LIBRARY_DIR}/ "
              f"({n} molecules, built from {lib_size} pulled).")
        return

    print(f"Building library from {lib_size} molecules "
          f"(pull from ChEMBL + ADMET scoring)...")
    try:
        build_library(n_molecules=lib_size, output_dir=LIBRARY_DIR)
        with open(BUILD_MARKER, "w") as fh:
            fh.write(str(lib_size))
    except Exception as exc:
        if have_library:
            print(f"  WARNING: library build failed ({exc}); "
                  f"reusing the existing cached library.")
        else:
            raise

    n = len(pd.read_csv(smiles_path))
    print(f"Library ready: {n} molecules at {LIBRARY_DIR}/.")


# ---------------------------------------------------------------------- #
# Results loading for the summary table
# ---------------------------------------------------------------------- #
def load_final_metrics(results_dir):
    """Return ``(hypervolume, pareto_size, n_evaluated)`` from a run's history.

    Reads ``<results_dir>/history.csv`` and returns its last row's metrics, or
    ``None`` if the file is missing or empty.
    """
    path = os.path.join(results_dir, "history.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if not len(df):
        return None
    last = df.iloc[-1]
    return (
        float(last["hypervolume"]),
        int(last["pareto_size"]),
        int(last["n_evaluated"]),
    )


def print_summary(elapsed_by_method):
    """Print the RESULTS SUMMARY table, highlighting the best hypervolume."""
    rows = []
    for name, results_dir in RESULTS_DIRS.items():
        metrics = load_final_metrics(results_dir)
        elapsed = elapsed_by_method.get(name)
        rows.append((name, metrics, elapsed))

    # Best hypervolume among methods that actually produced results.
    finished = [(n, m) for n, m, _ in rows if m is not None]
    best_hv = max((m[0] for _, m in finished), default=None)

    bar = "=" * 64
    print("\n" + bar)
    print("RESULTS SUMMARY")
    print(bar)
    print(f"{'Method':<18}{'Hypervolume':>13}{'Pareto':>9}"
          f"{'Evaluated':>11}{'Time':>8}")
    for name, metrics, elapsed in rows:
        time_str = fmt_time(elapsed) if elapsed is not None else "—"
        if metrics is None:
            print(f"{name:<18}{'(no results)':>13}{'—':>9}{'—':>11}{time_str:>8}")
            continue
        hv, pareto, evaluated = metrics
        marker = "  ← best" if best_hv is not None and hv == best_hv else ""
        print(f"{name:<18}{hv:>13.2f}{pareto:>9}{evaluated:>11}"
              f"{time_str:>8}{marker}")
    print(bar)


# ---------------------------------------------------------------------- #
# The four method runs
# ---------------------------------------------------------------------- #
def run_mogp(params):
    """Method 1/4 — multi-output GP + qNEHVI (our method).

    Uses the PRIMARY coregionalized (ICM) model by default; ``params["model"]``
    (``"coregionalized"`` / ``"independent"``) and ``params["rank"]`` select the
    model and its ICM rank. run_ablation.py compares the two arms head to head.
    """
    loop = BOLoop(
        library_dir=LIBRARY_DIR,
        seed=params["seed"],
        n_init=params["n_init"],
        batch_size=params["batch_size"],
        n_iterations=params["n_iterations"],
        mogp_train_iters=params["mogp_iters"],
        model=params.get("model", DEFAULT_MODEL),
        coregionalization_rank=params.get("rank", 1),
    )
    loop.run()
    loop.save_results(output_dir=RESULTS_DIRS["MOGP"])


def run_random(params):
    """Method 2/4 — uniform random search."""
    baseline = RandomSearchBaseline(
        library_dir=LIBRARY_DIR,
        seed=params["seed"],
        n_init=params["n_init"],
        batch_size=params["batch_size"],
        n_iterations=params["n_iterations"],
    )
    baseline.run()
    baseline.save_results(output_dir=RESULTS_DIRS["Random Search"])


def run_single_obj(params):
    """Method 3/4 — single-objective (docking-only) BO with EI."""
    baseline = SingleObjectiveBOLoop(
        library_dir=LIBRARY_DIR,
        seed=params["seed"],
        n_init=params["n_init"],
        batch_size=params["batch_size"],
        n_iterations=params["n_iterations"],
        gp_train_iters=params["mogp_iters"],
    )
    baseline.run()
    baseline.save_results(output_dir=RESULTS_DIRS["Single-Obj BO"])


def run_greedy(params):
    """Method 4/4 — greedy filter-then-dock with a matching docking budget."""
    baseline = GreedyFilterThenDock(
        library_dir=LIBRARY_DIR,
        seed=params["seed"],
        batch_size=params["batch_size"],
        n_total=params["n_total"],
    )
    baseline.run()
    baseline.save_results(output_dir=RESULTS_DIRS["Greedy Filter"])


# Ordered (label, runner) pairs so the driver can loop methods 1..4 uniformly.
METHODS = [
    ("MOGP", run_mogp),
    ("Random Search", run_random),
    ("Single-Obj BO", run_single_obj),
    ("Greedy Filter", run_greedy),
]


def launch_dashboard():
    """Launch the Streamlit comparison dashboard and block until interrupted."""
    print("\nLaunching comparison dashboard...")
    proc = subprocess.Popen(["streamlit", "run", "dashboard_compare.py"])

    # Give Streamlit a moment to bind the port before opening the browser.
    time.sleep(3)
    webbrowser.open("http://localhost:8501")
    print("Comparison dashboard at http://localhost:8501")
    print("Press Ctrl+C to stop")

    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\nStopping dashboard...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("Dashboard stopped.")


def main():
    print("=" * 64)
    print("RUN ALL — MOGP vs Baselines benchmark")
    print("=" * 64)

    # --- 1. Collect parameters (enter = default) ---
    lib_size = ask("Number of molecules in library", 1000, int)
    n_init = ask("Initial molecules", 5, int)
    batch_size = ask("Batch size", 5, int)
    n_iterations = ask("Number of iterations", 2, int)
    mogp_iters = ask("MOGP training iterations", 50, int)
    model = ask("MOGP model (coregionalized/independent)", DEFAULT_MODEL, str)
    rank = ask("Coregionalization rank", 1, int)
    seed = ask("Seed", 42, int)

    # Fair wall-clock: the docking cache is keyed by (SMILES, target) and shared,
    # so whichever method runs first pays the real docking cost while later methods
    # get cache hits — making the time column reflect run order, not the method.
    # Clearing the cache before EACH method makes all four pay full cold docking,
    # so their wall-clocks are directly comparable (results are identical either
    # way — a cached dock returns the same deterministic score). Off by default so
    # normal runs stay fast and keep the accumulated cache.
    fair_timing = ask_yes_no(
        "Clear docking cache before each method (fair, comparable wall-clock; "
        "much slower)?", default=False,
    )

    per_method = n_init + n_iterations * batch_size
    params = {
        "n_init": n_init,
        "batch_size": batch_size,
        "n_iterations": n_iterations,
        "mogp_iters": mogp_iters,
        "model": model,
        "rank": rank,
        "seed": seed,
        "n_total": per_method,   # greedy budget = same total as the others
    }

    # --- 2. Report scale + rough time estimate ---
    # 3 sequential methods dock `per_method` molecules each; greedy docks up to
    # `per_method` of the filter survivors. Docking dominates the total.
    total_docks = 3 * per_method + per_method
    est_seconds = total_docks * DOCK_SECONDS_ESTIMATE
    print(f"\nTotal molecules per method: {per_method} "
          f"(= {n_init} init + {n_iterations} iters x {batch_size} batch)")
    print(f"Approx docking calls across all 4 methods: {total_docks}")
    print(f"Estimated total time (rough, docking-bound): "
          f"~{fmt_time(est_seconds)}")

    # --- 3. Confirm ---
    if not ask_yes_no("\nStart all 4 runs?", default=True):
        print("Aborted.")
        return

    # --- Make sure the shared library exists at the requested size ---
    ensure_library(lib_size)

    # --- 4. Clear old results, then run all four in order ---
    for d in RESULTS_DIRS.values():
        if os.path.exists(d):
            shutil.rmtree(d)

    overall_start = time.time()
    elapsed_by_method = {}

    for i, (name, runner) in enumerate(METHODS, start=1):
        print("\n" + "=" * 64)
        print(f"Method {i}/4: {name}")
        print("=" * 64)
        # Give every method the SAME (cold) cache state so the timing is fair.
        if fair_timing:
            clear_cache()
            print("Cleared docking cache (fair timing: this method docks cold).")
        method_start = time.time()
        try:
            runner(params)
        except Exception as exc:
            print(f"  ERROR: {name} failed: {exc}")
        elapsed = time.time() - method_start
        elapsed_by_method[name] = elapsed
        print(f"\n{name} elapsed: {fmt_time(elapsed)}")

    # --- 5 + 6. Summary table and total wall-clock ---
    print_summary(elapsed_by_method)
    total_elapsed = time.time() - overall_start
    print(f"\nTotal wall-clock time for all methods: {fmt_time(total_elapsed)}")

    # --- 7 + 8. Launch dashboard (Ctrl+C handled inside) ---
    launch_dashboard()


if __name__ == "__main__":
    main()
