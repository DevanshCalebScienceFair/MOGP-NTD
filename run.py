"""
run.py
======

Interactive terminal front-end for the full MOGP antimalarial drug-discovery
pipeline. It auto-detects what has already been computed, collects run
parameters interactively (with sensible defaults), then drives the four
stages end to end:

    1. Train the ADMET oracle        (train_admet_oracle.main, --refit-on-full)
    2. Build the molecule library    (data.build_library)
    3. Run the BO loop               (loop.BOLoop)
    4. Launch the Streamlit dashboard (streamlit run dashboard.py)

Run it with::

    python run.py
"""

import os

# Stopgap for the libomp duplicate-runtime crash (OMP Error #15) that can fire
# when torch/botorch and scikit-learn pull in different OpenMP runtimes. Must be
# set before any of those libraries are imported, so it lives at module top.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import time
import subprocess
import webbrowser


# ---------------------------------------------------------------------------
# Locations (kept in sync with the modules this script drives)
# ---------------------------------------------------------------------------
MODEL_DIR = os.path.join("models", "pretrained_admet")
MODEL_FILES = ["caco2.joblib", "half_life.joblib", "herg.joblib"]

LIBRARY_DIR = os.path.join("data", "library")
LIBRARY_FILES = ["smiles.csv", "fingerprints.npy", "admet_scores.csv"]

DASHBOARD_URL = "http://localhost:8501"


# ---------------------------------------------------------------------------
# Status detection
# ---------------------------------------------------------------------------
def models_ready():
    """True if all three pretrained ADMET model files are present."""
    return all(
        os.path.exists(os.path.join(MODEL_DIR, f)) for f in MODEL_FILES
    )


def library_ready():
    """True if every cached library file is present."""
    return all(
        os.path.exists(os.path.join(LIBRARY_DIR, f)) for f in LIBRARY_FILES
    )


def library_size():
    """Number of molecules in the cached library (0 if it cannot be read)."""
    smiles_path = os.path.join(LIBRARY_DIR, "smiles.csv")
    if not os.path.exists(smiles_path):
        return 0
    try:
        with open(smiles_path) as fh:
            # One header row ("SMILES") + one row per molecule.
            rows = sum(1 for line in fh if line.strip())
        return max(rows - 1, 0)
    except OSError:
        return 0


def results_present(output_dir):
    """True if any of the loop's result CSVs already exist in output_dir."""
    return any(
        os.path.exists(os.path.join(output_dir, f))
        for f in ("history.csv", "evaluated.csv", "pareto_front.csv")
    )


# ---------------------------------------------------------------------------
# Interactive input helpers
# ---------------------------------------------------------------------------
def ask_int(prompt, default):
    """Prompt for a positive integer, falling back to default on empty input."""
    while True:
        raw = input(prompt).strip()
        if raw == "":
            return default
        try:
            value = int(raw)
        except ValueError:
            print(f"  '{raw}' is not a whole number. Please try again.")
            continue
        if value <= 0:
            print("  Please enter a positive whole number.")
            continue
        return value


def ask_str(prompt, default):
    """Prompt for a string, falling back to default on empty input."""
    raw = input(prompt).strip()
    return raw if raw else default


def ask_yes_no(prompt, default=True):
    """Prompt for yes/no. Empty input returns default."""
    while True:
        raw = input(prompt).strip().lower()
        if raw == "":
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  Please answer 'y' or 'n'.")


# ---------------------------------------------------------------------------
# Banner / status / summary
# ---------------------------------------------------------------------------
def print_banner():
    print("====================================")
    print("MOGP Antimalarial Drug Discovery")
    print("PfDHFR Target | Multi-Objective BO")
    print("====================================")


def print_status(output_dir):
    print("\nStatus:")
    if models_ready():
        print("  [✓] ADMET models trained")
    else:
        print("  [✗] ADMET models not found")

    if library_ready():
        print(f"  [✓] Molecule library built ({library_size()} molecules)")
    else:
        print("  [✗] Molecule library not found")

    if results_present(output_dir):
        print(f"  [✓] Previous results in {output_dir}/")
    else:
        print("  [✗] No previous results found")


def format_duration(total_minutes):
    """Render a minute count as 'Xh Ym'."""
    hours = int(total_minutes) // 60
    minutes = int(total_minutes) % 60
    return f"{hours}h {minutes}m"


def estimate_minutes(n_init, batch_size, n_iterations):
    """Rough wall-clock estimate, in minutes.

    Docking dominates: ~3 min per docked molecule, across the initial set plus
    every selected batch. EHVI acquisition adds ~25 min per BO iteration.
    """
    n_docked = n_init + n_iterations * batch_size
    docking_minutes = n_docked * 3
    ehvi_minutes = n_iterations * 25
    return docking_minutes + ehvi_minutes


def print_summary(params):
    n_molecules = params["n_molecules"]
    n_init = params["n_init"]
    batch_size = params["batch_size"]
    n_iterations = params["n_iterations"]

    lib_state = "already exists" if library_ready() else "will build"
    total_docking = n_init + n_iterations * batch_size
    eta = format_duration(
        estimate_minutes(n_init, batch_size, n_iterations)
    )

    print("\n=== Run Summary ===")
    print(f"Library:        {n_molecules} molecules ({lib_state})")
    print(f"Initial set:    {n_init} molecules")
    print(f"Batch size:     {batch_size} per iteration")
    print(f"Iterations:     {n_iterations}")
    print(f"Total docking:  ~{total_docking} molecules")
    print(f"Estimated time: ~{eta}")


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------
def step_train_oracle():
    print("\nStep 1/4: Training ADMET Oracle...")
    if models_ready():
        print("  ADMET models already present — skipping training.")
        return

    import train_admet_oracle

    # main() reads --refit-on-full from sys.argv; inject it temporarily so the
    # production models are refit on 100% of the data.
    saved_argv = sys.argv
    sys.argv = ["train_admet_oracle.py", "--refit-on-full"]
    try:
        train_admet_oracle.main()
    finally:
        sys.argv = saved_argv


def step_build_library(n_molecules):
    print("\nStep 2/4: Building Molecule Library...")
    if library_ready():
        print(f"  Library already present ({library_size()} molecules) "
              "— skipping build.")
        return

    import data

    data.build_library(n_molecules=n_molecules)


def step_run_loop(params):
    print("\nStep 3/4: Running Bayesian Optimization Loop...")

    from loop import BOLoop

    loop = BOLoop(
        library_dir=LIBRARY_DIR,
        n_init=params["n_init"],
        batch_size=params["batch_size"],
        n_iterations=params["n_iterations"],
        mogp_train_iters=params["mogp_train_iters"],
    )
    loop.run()
    loop.save_results(output_dir=params["output_dir"])

    pareto = loop.get_pareto_front()
    print(f"\nFinal Pareto front: {len(pareto['smiles'])} molecules")
    header = f"{'SMILES':<50}" + "".join(
        f"{n:>22}" for n in pareto["task_names"]
    )
    print(header)
    for smiles, row in zip(pareto["smiles"], pareto["objectives"]):
        print(f"{smiles:<50}" + "".join(f"{v:22.4f}" for v in row))


def step_launch_dashboard():
    print("\nStep 4/4: Launching Dashboard...")

    proc = subprocess.Popen(
        ["streamlit", "run", "dashboard.py"]
    )

    # Give Streamlit a moment to bind the port before opening the browser.
    time.sleep(3)
    webbrowser.open(DASHBOARD_URL)

    print(f"Dashboard running at {DASHBOARD_URL}")
    print("Press Ctrl+C to stop")

    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\nStopping dashboard...")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    print_banner()

    # First pass of status uses the default output dir; re-evaluated against the
    # chosen dir inside the summary.
    print_status("results")

    print("\nConfigure the run (press Enter to accept the [default]):")
    params = {
        "n_molecules": ask_int(
            "Number of molecules to pull from ChEMBL [10000]: ", 10000),
        "n_init": ask_int(
            "Number of initial random molecules [10]: ", 10),
        "batch_size": ask_int(
            "Batch size per iteration [10]: ", 10),
        "n_iterations": ask_int(
            "Number of BO iterations [10]: ", 10),
        "mogp_train_iters": ask_int(
            "MOGP training iterations [200]: ", 200),
        "output_dir": ask_str(
            "Output directory [results]: ", "results"),
    }

    print_summary(params)

    if not ask_yes_no("\nStart the run? [Y/n]: ", default=True):
        print("Aborted. No changes made.")
        return

    step_train_oracle()
    step_build_library(params["n_molecules"])
    step_run_loop(params)
    step_launch_dashboard()


if __name__ == "__main__":
    main()
