#!/usr/bin/env python
"""
run_matrix.py
=============

Exhaustive feature-combination test harness for MOGP-NTD.

This script adds **no new pipeline behaviour**. Every case it runs is an
existing entry point driven through its existing CLI flags; the only thing
here is the enumeration, isolation, timing and reporting around them. Nothing
in the pipeline modules is imported at module scope, so ``--list`` /
``--dry-run`` work from any interpreter.

Tiers (cumulative — each tier also runs the ones before it):

  fast   No docking, no network. Byte-compile everything, ``--help`` every
         entry point, run the unit tests, run the dry-run/skip-docking paths.
         Minutes.
  smoke  + the real combinatorial BO matrix at tiny scale (n_init=5,
         batch=2, iters=2) plus every baseline. Docking-bound on the first
         pass; the shared docking cache makes repeats cheap.
  full   + multi-seed benchmark, ablation sweep, docking validation, and the
         verification/ harness. Hours.

Usage::

    python run_matrix.py --list                 # enumerate cases, run nothing
    python run_matrix.py --tier fast            # quick correctness sweep
    python run_matrix.py --tier smoke           # the full combination matrix
    python run_matrix.py --tier smoke --dry-run # print the exact commands
    python run_matrix.py --tier smoke --resume  # skip cases that already passed

Each case runs as its own subprocess with its own ``--output-dir`` under
``matrix_results/``, its own log under ``matrix_results/logs/``, and a
timeout. Results land in ``matrix_results/results.csv`` and ``report.md``.
"""

import argparse
import csv
import datetime
import fnmatch
import json
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_ROOT = os.path.join(ROOT, "matrix_results")
LOG_DIR = os.path.join(OUT_ROOT, "logs")
RESULTS_CSV = os.path.join(OUT_ROOT, "results.csv")
REPORT_MD = os.path.join(OUT_ROOT, "report.md")

TIERS = ("fast", "smoke", "full")
TIER_RANK = {t: i for i, t in enumerate(TIERS)}

# Child-process environment. Mirrors the guard launch.py installs before torch
# is imported: duplicate-libomp tolerance plus single-threaded BLAS, which is
# what keeps the GP fits stable on Apple Silicon. Setting it here means every
# case gets it, including the ones that call loop.py directly.
CHILD_ENV = {
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "PYTHONUNBUFFERED": "1",
}

# Every entry point that exposes an argparse CLI. `--help` on each is the
# cheapest possible check that its imports and parser still construct. Entries
# absent from the checkout are dropped, so the same harness covers branches
# that do and do not carry the GP-MOBO / cached-arena work.
CLI_SCRIPTS = [
    "loop.py",
    "launch.py",
    "data.py",
    "rebuild_library.py",
    "run_ablation.py",
    "run_benchmark_seeds.py",
    "baseline_random.py",
    "baseline_greedy.py",
    "baseline_single_obj.py",
    "baseline_gpmobo.py",
    "build_cached_arena.py",
    "validate_docking.py",
    "validate_known_actives.py",
    "train_admet_oracle.py",
]

UNIT_TESTS = [
    "test_gp.py",
    "test_acquisition.py",
    "test_admet_oracle.py",
    "test_coregionalized.py",
    "test_densify.py",
    "test_evaluation.py",
    "test_gpmobo.py",
]

# Modules that must import cleanly but have no CLI of their own.
IMPORT_ONLY = [
    "acquisition", "admet_oracle", "densify", "docking", "docking_cache",
    "evaluation", "kernel", "mogp", "mogp_coregionalized", "quality_filter",
    "gpmobo_ref", "utils.featurize",
]

# Streamlit apps: `streamlit run` never exits, so they get a compile check only.
STREAMLIT_APPS = ["app.py", "dashboard.py", "dashboard_compare.py"]

# Scale knobs per tier for the BO cases.
SCALE = {
    "smoke": {"n_init": 5, "batch_size": 2, "n_iterations": 2, "mogp_iters": 30,
              "lib_pull": 400},
    "full": {"n_init": 10, "batch_size": 5, "n_iterations": 4, "mogp_iters": 100,
             "lib_pull": 2000},
}

# Per-case timeouts in seconds. Docking dominates, at roughly 40s per
# (molecule, target); a smoke BO run docks ~9 molecules against 2 targets.
TIMEOUTS = {
    "compile": 300,
    "cli": 300,
    "unit": 1800,
    "dryrun": 1800,
    "bo": 3600,
    "baseline": 3600,
    "harness": 14400,
    "validate": 7200,
    "prereq": 7200,
}


class Case:
    """One subprocess invocation, plus the metadata needed to report on it."""

    def __init__(self, cid, group, tier, cmd, timeout=900, needs_library=False):
        self.id = cid
        self.group = group
        self.tier = tier
        self.cmd = cmd
        self.timeout = timeout
        self.needs_library = needs_library

    @property
    def log_path(self):
        return os.path.join(LOG_DIR, self.id + ".log")

    def pretty(self):
        return " ".join(shlex.quote(c) for c in self.cmd)


def have(relpath):
    """True if the checkout carries this file.

    Feature detection, not version detection: the GP-MOBO baseline, the cached
    arena builder and their tests live on a branch that may or may not be
    merged, and cases that depend on them are simply omitted when absent.
    """
    return os.path.exists(os.path.join(ROOT, relpath))


def script_supports(script, flag):
    """True if ``script`` mentions ``flag``, used to gate flag-level cases.

    Cheaper and more robust than parsing ``--help`` output for every candidate
    flag: the argparse call sites spell the flag literally.
    """
    path = os.path.join(ROOT, script)
    if not os.path.exists(path):
        return False
    with open(path) as fh:
        return '"{}"'.format(flag) in fh.read()


def library_size(library_dir):
    """Rows in the cached library, or None if it has not been built yet."""
    path = os.path.join(library_dir, "smiles.csv")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        return max(0, sum(1 for _ in fh) - 1)  # minus the header


def test_command(python, path):
    """pytest for files defining ``def test_``, plain execution otherwise.

    The suite is mixed: some files are pytest modules, others assert at module
    scope with a ``__main__`` block. Running a module-scope file under pytest
    exits 5 ("no tests collected") even when it passed, so dispatch on content.
    """
    try:
        with open(os.path.join(ROOT, path)) as fh:
            src = fh.read()
    except OSError:
        src = ""
    if "def test_" in src:
        return [python, "-m", "pytest", path, "-q"]
    return [python, path]


def build_cases(args):
    """Enumerate every case at or below the requested tier."""
    py = args.python
    cases = []
    lib = args.library_dir
    # Mutable copy: the run-scale-dependent budgets are filled in below, once
    # n_init/batch/iters are known. Fast-tier cases are added before that and
    # only use the fixed keys, so they are unaffected.
    timeouts = dict(TIMEOUTS)

    def add(cid, group, tier, cmd, timeout_key, needs_library=False):
        if TIER_RANK[tier] > TIER_RANK[args.tier]:
            return
        cases.append(Case(cid, group, tier, cmd, timeouts[timeout_key],
                          needs_library))

    def outdir(cid):
        return os.path.join(OUT_ROOT, "runs", cid)

    # ---------------------------------------------------------------- fast --
    # Byte-compile the whole repo, Streamlit apps included. This is the only
    # coverage the dashboards get: `streamlit run` does not terminate.
    all_py = sorted(f for f in os.listdir(ROOT) if f.endswith(".py"))
    add("compile-all", "compile", "fast",
        [py, "-m", "py_compile"] + all_py, "compile")
    add("compile-streamlit", "compile", "fast",
        [py, "-m", "py_compile"] + STREAMLIT_APPS, "compile")

    for mod in IMPORT_ONLY:
        if not have(mod.replace(".", os.sep) + ".py"):
            continue
        add("import-" + mod.replace(".", "-"), "import", "fast",
            [py, "-c", "import " + mod], "cli")

    for script in CLI_SCRIPTS:
        if not have(script):
            continue
        add("cli-" + script[:-3].replace("_", "-"), "cli", "fast",
            [py, script, "--help"], "cli")

    for test in UNIT_TESTS:
        if not have(test):
            continue
        add("unit-" + test[5:-3].replace("_", "-"), "unit", "fast",
            test_command(py, test), "unit")

    # Paths that exercise real logic without paying for docking.
    add("dryrun-known-actives-nodock", "dryrun", "fast",
        [py, "validate_known_actives.py", "--skip-docking"], "dryrun")

    # --------------------------------------------------------------- smoke --
    scale = SCALE["smoke" if args.tier == "smoke" else "full"]
    n_init = args.n_init if args.n_init is not None else scale["n_init"]
    batch = args.batch_size if args.batch_size is not None else scale["batch_size"]
    iters = (args.n_iterations if args.n_iterations is not None
             else scale["n_iterations"])
    mogp_iters = (args.mogp_iters if args.mogp_iters is not None
                  else scale["mogp_iters"])
    n_total = n_init + iters * batch

    # Timeouts MUST track the run scale. A fixed budget silently kills cases
    # that would have passed: measured pace at 290 molecules / 20 iterations
    # was ~4 min per BO iteration (the GP fit grows with the evaluated set), so
    # a full case ran ~80 min against a flat 60-min budget and was recorded as
    # a timeout. Budget per evaluated molecule instead, generously — an
    # over-long timeout costs nothing when a case finishes early, whereas an
    # over-short one destroys the case's results.
    per_run = max(1800, int(n_total * 90))
    timeouts["bo"] = per_run
    timeouts["baseline"] = per_run
    # The multi-seed harness runs 4 methods inside a single case, and the
    # ablation sweep runs seeds x models; give both room for all of them.
    timeouts["harness"] = per_run * 5
    timeouts["validate"] = max(timeouts["validate"], per_run)
    # The ChEMBL PULL size (~60% survives Lipinski + the ADMET domain check).
    # This single value feeds BOTH the prereq build and run_benchmark_seeds'
    # --lib-size, which is load-bearing: run_all.ensure_library REBUILDS
    # data/library whenever the build marker disagrees with the pull size it is
    # handed, so a seeds case carrying a different number would silently swap
    # the library out from under every case that runs after it.
    lib_pull = args.lib_pull if args.lib_pull is not None else scale["lib_pull"]

    # The library is a prerequisite for everything below, and it is not
    # committed. Build it first if it is missing (this one hits the network).
    if library_size(args.library_dir) is None:
        add("prereq-build-library", "prereq", "smoke",
            [py, "data.py", "--n-molecules", str(lib_pull)], "prereq")

    # --arena redirects every library-consuming case at the cached arena: the
    # sub-library of molecules that already have an OK docking score for every
    # target, so the whole matrix runs with zero Vina calls. It is only
    # meaningful once the docking cache is warm — the arena is built FROM the
    # cache, so on a cold cache it comes out empty. Run the matrix normally
    # once, then re-run it with --arena for fast iteration.
    if args.arena:
        if not have("build_cached_arena.py"):
            raise SystemExit(
                "ERROR: --arena needs build_cached_arena.py, which this "
                "checkout does not have (it is on feat/gpmobo-baseline).")
        lib = args.arena_dir
        if library_size(lib) is None:
            cmd = [py, "build_cached_arena.py",
                   "--library-dir", args.library_dir,
                   "--output-dir", args.arena_dir]
            if args.arena_limit is not None:
                cmd += ["--limit", str(args.arena_limit)]
            add("prereq-build-arena", "prereq", "smoke", cmd, "prereq")

    size = library_size(lib)

    # --densify-max-pool caps the TOTAL library size and is a no-op unless it
    # sits above the current size, so derive it from the library when we have
    # one and fall back to a value comfortably above the expected build size.
    max_pool = (size + 200) if size else int(lib_pull * 0.6) + 200

    scale_flags = ["--n-init", str(n_init), "--batch-size", str(batch),
                   "--n-iterations", str(iters), "--mogp-iters", str(mogp_iters)]

    # The combinatorial core: GP model x coregionalization rank x
    # densification. Rank only applies to the coregionalized model, and the
    # densify sub-flags only apply when --densify is on, so the cross is over
    # meaningful configurations rather than a raw product with dead cells.
    model_arms = [
        ("independent", ["--model", "independent"]),
        ("coreg-r1", ["--model", "coregionalized", "--rank", "1"]),
        ("coreg-r2", ["--model", "coregionalized", "--rank", "2"]),
    ]
    densify_arms = [
        ("nodensify", []),
        ("densify-every1", ["--densify", "--densify-every", "1",
                            "--densify-per-parent", "5"]),
        ("densify-every2-cap", ["--densify", "--densify-every", "2",
                                "--densify-per-parent", "5",
                                "--densify-max-pool", str(max_pool)]),
    ]
    for m_name, m_flags in model_arms:
        for d_name, d_flags in densify_arms:
            cid = "bo-{}-{}".format(m_name, d_name)
            add(cid, "bo", "smoke",
                [py, "loop.py", "--library-dir", lib, "--output-dir", outdir(cid)]
                + scale_flags + m_flags + d_flags, "bo", needs_library=True)

    # launch.py is the threading-guard wrapper around the same loop; it has no
    # densify flags, so it crosses over the model arms only.
    for m_name, m_flags in model_arms:
        cid = "launch-" + m_name
        add(cid, "bo", "smoke",
            [py, "launch.py", "--library-dir", lib, "--output-dir", outdir(cid)]
            + scale_flags + m_flags, "bo", needs_library=True)

    # The --smoke profile flag itself, on both runners.
    for runner in ("loop.py", "launch.py"):
        cid = "profile-smoke-" + runner[:-3]
        add(cid, "bo", "smoke",
            [py, runner, "--library-dir", lib, "--output-dir", outdir(cid),
             "--smoke", "--mogp-iters", str(mogp_iters)], "bo",
            needs_library=True)

    # Every baseline, at the same scale so the hypervolumes stay comparable.
    add("baseline-random", "baseline", "smoke",
        [py, "baseline_random.py", "--library-dir", lib,
         "--output-dir", outdir("baseline-random"),
         "--n-init", str(n_init), "--batch-size", str(batch),
         "--n-iterations", str(iters), "--seed", "99"],
        "baseline", needs_library=True)

    add("baseline-single-obj", "baseline", "smoke",
        [py, "baseline_single_obj.py", "--library-dir", lib,
         "--output-dir", outdir("baseline-single-obj"),
         "--n-init", str(n_init), "--batch-size", str(batch),
         "--n-iterations", str(iters), "--mogp-iters", str(mogp_iters),
         "--seed", "77"],
        "baseline", needs_library=True)

    # Greedy is the filter-then-dock control; its thresholds are the feature
    # surface, so cover the default gate and a permissive one.
    add("baseline-greedy-default", "baseline", "smoke",
        [py, "baseline_greedy.py", "--library-dir", lib,
         "--output-dir", outdir("baseline-greedy-default"),
         "--batch-size", str(batch), "--n-total", str(n_total), "--seed", "55"],
        "baseline", needs_library=True)

    add("baseline-greedy-permissive", "baseline", "smoke",
        [py, "baseline_greedy.py", "--library-dir", lib,
         "--output-dir", outdir("baseline-greedy-permissive"),
         "--batch-size", str(batch), "--n-total", str(n_total), "--seed", "55",
         "--herg-threshold", "0.8", "--halflife-min", "0.5",
         "--caco2-min", "-6.5"],
        "baseline", needs_library=True)

    # GP-MOBO is the external published comparator. It needs the pinned clone
    # at external/GP-MOBO (gpmobo_ref.py prints the two setup commands), so
    # gate on that rather than emitting cases that can only fail. Its three
    # feature axes cross here: EHVI estimator x objective frame x hyper-
    # parameter mode. The sampled estimators are correctness oracles and are
    # far too slow for a real run, so they stay at full tier on tiny scale.
    if have("baseline_gpmobo.py"):
        gp_ok = os.path.isdir(os.path.join(ROOT, "external", "GP-MOBO"))
        gp_flags = ["--library-dir", lib, "--n-init", str(n_init),
                    "--batch-size", str(batch), "--n-iterations", str(iters),
                    "--seed", "99"]
        if not gp_ok:
            print("NOTE: skipping GP-MOBO cases; external/GP-MOBO is not "
                  "cloned.\n      git clone https://github.com/anabelyong/"
                  "GP-MOBO.git external/GP-MOBO")
        else:
            for frame in ("raw", "normalized"):
                for hp in ("budget", "holdout"):
                    cid = "gpmobo-analytic-{}-{}".format(frame, hp)
                    add(cid, "baseline", "smoke",
                        [py, "baseline_gpmobo.py"] + gp_flags
                        + ["--ehvi-impl", "analytic",
                           "--objective-frame", frame,
                           "--hparam-mode", hp,
                           "--output-dir", outdir(cid)],
                        "baseline", needs_library=True)

            # The sampled estimators exist to cross-check 'analytic'; run them
            # at the smallest scale that still exercises the code path.
            for impl in ("fast", "reference"):
                cid = "gpmobo-" + impl
                add(cid, "baseline", "full",
                    [py, "baseline_gpmobo.py", "--library-dir", lib,
                     "--n-init", str(n_init), "--batch-size", str(batch),
                     "--n-iterations", "1", "--seed", "99",
                     "--ehvi-impl", impl, "--mc-samples", "16",
                     "--output-dir", outdir(cid)],
                    "baseline", needs_library=True)

    # ---------------------------------------------------------------- full --
    # The ablation sweep runs both GP models across seeds internally.
    add("harness-ablation", "harness", "full",
        [py, "run_ablation.py", "--library-dir", lib, "--seeds", "0,1",
         "--n-init", str(n_init), "--batch-size", str(batch),
         "--n-iterations", str(iters), "--mogp-iters", str(mogp_iters),
         "--models", "coregionalized,independent", "--save"],
        "harness", needs_library=True)

    # The multi-seed benchmark owns several feature axes of its own: the error
    # band, densification, the docking cache toggle, and — where the branch
    # provides them — an explicit library dir and the GP-MOBO arm's settings.
    seeds_base = [py, "run_benchmark_seeds.py", "--seeds", "0",
                  "--lib-size", str(lib_pull),
                  "--n-init", str(n_init), "--batch-size", str(batch),
                  "--n-iterations", str(iters), "--mogp-iters", str(mogp_iters)]
    # --library-dir lets the whole benchmark run off an existing library
    # (including the zero-Vina arena) instead of building its own.
    if script_supports("run_benchmark_seeds.py", "--library-dir"):
        seeds_base += ["--library-dir", lib]

    for band in ("std", "sem", "ci95"):
        cid = "harness-seeds-band-" + band
        add(cid, "harness", "full",
            seeds_base + ["--band", band, "--output-dir", outdir(cid)],
            "harness", needs_library=True)

    add("harness-seeds-densify", "harness", "full",
        seeds_base + ["--densify", "--densify-per-parent", "5",
                      "--densify-max-pool", str(max_pool),
                      "--output-dir", outdir("harness-seeds-densify")],
        "harness", needs_library=True)

    # --no-cache is the only CLI surface for the docking cache toggle, so this
    # is the case that proves the uncached docking path still works.
    add("harness-seeds-nocache", "harness", "full",
        seeds_base + ["--no-cache",
                      "--output-dir", outdir("harness-seeds-nocache")],
        "harness", needs_library=True)

    # --aggregate-only re-reads the per-seed CSVs a previous case wrote rather
    # than running anything, so point it at the band-std output dir. Ordering
    # holds because cases run in the order built.
    if script_supports("run_benchmark_seeds.py", "--aggregate-only"):
        add("harness-seeds-aggregate-only", "harness", "full",
            seeds_base + ["--aggregate-only",
                          "--output-dir", outdir("harness-seeds-band-std")],
            "harness", needs_library=True)

    # The GP-MOBO arm's knobs, exercised through the seeds harness.
    if script_supports("run_benchmark_seeds.py", "--gpmobo-frame"):
        add("harness-seeds-gpmobo-normalized", "harness", "full",
            seeds_base + ["--gpmobo-frame", "normalized",
                          "--gpmobo-hparam-mode", "holdout",
                          "--output-dir",
                          outdir("harness-seeds-gpmobo-normalized")],
            "harness", needs_library=True)

    # --dry-run still reads the existing library (rebuild_library.py:308 opens
    # smiles.csv unguarded) and still pulls from ChEMBL to size the rebuild, so
    # this is a library-and-network case, not a fast-tier one.
    add("dryrun-rebuild-library", "dryrun", "full",
        [py, "rebuild_library.py", "--dry-run", "--target-size", "200"],
        "dryrun", needs_library=True)

    add("validate-docking", "validate", "full",
        [py, "validate_docking.py", "--library-dir", lib, "--n-sample", "20",
         "--top-k", "5", "--seed", "42",
         "--output", os.path.join(OUT_ROOT, "validate_docking_scatter.png")],
        "validate", needs_library=True)

    # The docking arm of the known-actives check (the fast tier already covers
    # --skip-docking), pointed at two BO runs the matrix produced above.
    add("validate-known-actives", "validate", "full",
        [py, "validate_known_actives.py",
         "--independent-dir", outdir("bo-independent-nodensify"),
         "--coregionalized-dir", outdir("bo-coreg-r1-nodensify")],
        "validate", needs_library=True)

    # The BioMOBO harness targets a different design and is expected to skip
    # rather than pass; it is here to catch collection errors.
    add("verification-harness", "unit", "full",
        [py, "-m", "pytest", "verification", "-q"], "unit")

    return cases


def load_previous():
    """Case ids that passed on an earlier invocation, for --resume."""
    if not os.path.exists(RESULTS_CSV):
        return set()
    passed = set()
    with open(RESULTS_CSV) as fh:
        for row in csv.DictReader(fh):
            if row.get("status") == "pass":
                passed.add(row["id"])
    return passed


def run_case(case, env):
    """Execute one case, tee its output to a log, and return a result row."""
    os.makedirs(os.path.dirname(case.log_path), exist_ok=True)
    start = time.time()
    started_at = datetime.datetime.now()
    with open(case.log_path, "w") as log:
        log.write("$ {}\n\n".format(case.pretty()))
        log.flush()
        try:
            proc = subprocess.run(case.cmd, cwd=ROOT, env=env, stdout=log,
                                  stderr=subprocess.STDOUT,
                                  timeout=case.timeout)
            code = proc.returncode
            status = "pass" if code == 0 else "fail"
        except subprocess.TimeoutExpired:
            code = -1
            status = "timeout"
            log.write("\n\n*** TIMEOUT after {}s ***\n".format(case.timeout))
        except OSError as exc:
            code = -2
            status = "error"
            log.write("\n\n*** COULD NOT LAUNCH: {} ***\n".format(exc))
    ended_at = datetime.datetime.now()
    return {
        "id": case.id,
        "group": case.group,
        "tier": case.tier,
        "status": status,
        "exit_code": code,
        "seconds": round(time.time() - start, 1),
        # Wall-clock start/end per case. A long run's timing outliers are
        # otherwise undiagnosable after the fact: duration alone cannot
        # distinguish "this case did more work" from "something else was
        # competing for the CPU during that window".
        "started_at": started_at.isoformat(timespec="seconds"),
        "ended_at": ended_at.isoformat(timespec="seconds"),
        "log": os.path.relpath(case.log_path, ROOT),
        "command": case.pretty(),
    }


def _capture(cmd, cwd=ROOT):
    """Run a command and return its stripped stdout, or '' if it fails."""
    try:
        out = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                             timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return out.stdout.strip()


def _objective_description(args):
    """What the docking objectives optimize, for the manifest.

    Two sweeps can otherwise be indistinguishable from their files: the Aug-8
    run optimized ligand efficiency and this one optimizes raw kcal/mol, and
    their hypervolumes are NOT commensurable. Recording it means a later audit
    can tell them apart without reading the git log.
    """
    try:
        out = _capture([args.python, "-c",
                        "import json, evaluation;"
                        "print(json.dumps({"
                        "'docking_units': 'kcal/mol'"
                        " if evaluation.DOCKING_KCAL_MIN < -1 else"
                        " 'ligand_efficiency',"
                        "'docking_bounds': [evaluation.DOCKING_KCAL_MIN,"
                        " evaluation.DOCKING_KCAL_MAX]}))"])
        return json.loads(out) if out else {"docking_units": "unknown"}
    except (ValueError, OSError):
        return {"docking_units": "unknown"}


def write_manifest(args, cases, effective_lib):
    """Record what was run, against what code, with which dependencies.

    A hypervolume is only meaningful alongside the objective bounds and the
    library it was computed over, and a pass/fail table is only actionable if
    you can tell which commit produced it. This captures all of that up front,
    so the sweep's outputs stay interpretable later. Best-effort throughout —
    a missing git or vina degrades a field rather than failing the run.
    """
    versions = _capture([
        args.python, "-c",
        "import json,sys;"
        "d={'python':sys.version.split()[0]};"
        "\nfor m in ['numpy','torch','gpytorch','botorch','sklearn','rdkit',"
        "'pandas','matplotlib']:\n"
        "    try:\n"
        "        d[m]=__import__(m).__version__\n"
        "    except Exception as e:\n"
        "        d[m]='unavailable: %s' % type(e).__name__\n"
        "print(json.dumps(d))"
    ])
    try:
        packages = json.loads(versions) if versions else {}
    except ValueError:
        packages = {"raw": versions}

    manifest = {
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
        "host": platform.node(),
        "platform": platform.platform(),
        "invocation": " ".join(shlex.quote(a) for a in sys.argv),
        "python": args.python,
        "packages": packages,
        "git": {
            "commit": _capture(["git", "rev-parse", "HEAD"]),
            "branch": _capture(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            # A dirty tree means the commit alone does not identify the code
            # that ran, so record exactly which files were modified.
            "dirty": _capture(["git", "status", "--porcelain"]).splitlines(),
        },
        "vina": _capture(["vina", "--version"]).splitlines()[:1],
        "tier": args.tier,
        "scale": {
            "lib_pull": args.lib_pull,
            "n_init": args.n_init,
            "batch_size": args.batch_size,
            "n_iterations": args.n_iterations,
            "mogp_iters": args.mogp_iters,
        },
        "objective": _objective_description(args),
        "library_dir": effective_lib,
        "library_molecules": library_size(effective_lib),
        "arena": bool(args.arena),
        "n_cases": len(cases),
        "cases": [c.id for c in cases],
    }
    with open(os.path.join(OUT_ROOT, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    # Hypervolume is defined relative to these bounds; snapshot them so the
    # numbers stay interpretable even if the bounds are regenerated later.
    bounds_src = os.path.join(ROOT, "evaluation_bounds.json")
    if os.path.exists(bounds_src):
        shutil.copy2(bounds_src,
                     os.path.join(OUT_ROOT, "evaluation_bounds.snapshot.json"))
    return manifest


def write_results(rows):
    """Write results.csv, MERGING with any rows already recorded there.

    A partial invocation (--only / --resume, e.g. re-running one failed case)
    must not erase the record of a full sweep. Rows from this run replace
    same-id rows from a previous one; everything else is preserved.
    """
    fields = ["id", "group", "tier", "status", "exit_code", "seconds",
              "started_at", "ended_at", "log", "command"]
    merged = {}
    if os.path.exists(RESULTS_CSV):
        try:
            with open(RESULTS_CSV, newline="") as fh:
                for prev in csv.DictReader(fh):
                    if prev.get("id"):
                        merged[prev["id"]] = prev
        except (OSError, csv.Error):
            pass
    for row in rows:
        merged[row["id"]] = row
    rows = list(merged.values())
    with open(RESULTS_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_report(rows, args):
    total = len(rows)
    counts = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    lines = [
        "# MOGP-NTD feature matrix report",
        "",
        "- tier: `{}`".format(args.tier),
        "- cases: {}".format(total),
        "- wall-clock: {:.1f}s".format(sum(r["seconds"] for r in rows)),
        "- " + ", ".join("{}: {}".format(k, v) for k, v in sorted(counts.items())),
        "",
        "| case | group | status | secs | log |",
        "|---|---|---|---|---|",
    ]
    for row in sorted(rows, key=lambda r: (r["status"] != "pass", r["group"],
                                           r["id"])):
        lines.append("| `{id}` | {group} | {status} | {seconds} | `{log}` |"
                     .format(**row))
    failures = [r for r in rows if r["status"] != "pass"]
    if failures:
        lines += ["", "## Failures", ""]
        for row in failures:
            lines += ["### `{}` ({})".format(row["id"], row["status"]),
                      "", "```", row["command"], "```", ""]
    with open(REPORT_MD, "w") as fh:
        fh.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Run every MOGP-NTD feature in every meaningful "
                    "combination and report pass/fail per case.")
    parser.add_argument("--tier", choices=TIERS, default="fast",
                        help="How much to run; tiers are cumulative "
                             "(default: fast).")
    parser.add_argument("--list", action="store_true",
                        help="List the cases for the tier and exit.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the exact command for each case and exit.")
    parser.add_argument("--only", action="append", default=[],
                        help="Glob on case id; repeatable. Only matches run.")
    parser.add_argument("--skip", action="append", default=[],
                        help="Glob on case id; repeatable. Matches are dropped.")
    parser.add_argument("--group", action="append", default=[],
                        help="Restrict to these groups (compile, import, cli, "
                             "unit, dryrun, prereq, bo, baseline, harness, "
                             "validate).")
    parser.add_argument("--resume", action="store_true",
                        help="Skip cases recorded as passing in the existing "
                             "results.csv.")
    parser.add_argument("--fail-fast", action="store_true",
                        help="Stop at the first non-passing case.")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Cases in parallel. Leave at 1: concurrent "
                             "processes contend on the SQLite docking cache.")
    parser.add_argument("--timeout-scale", type=float, default=1.0,
                        help="Multiply every per-case timeout.")
    parser.add_argument("--library-dir", default="data/library")
    parser.add_argument("--arena", action="store_true",
                        help="Point every case at the cached arena instead of "
                             "the full library, so the matrix runs with ZERO "
                             "Vina calls. Needs a warm docking cache: the "
                             "arena is built from it. Run the matrix once "
                             "normally, then re-run with --arena.")
    parser.add_argument("--arena-dir", default="data/library_cached_arena",
                        help="Where the cached arena lives.")
    parser.add_argument("--arena-limit", type=int, default=None,
                        help="Cap the arena size when building it.")
    parser.add_argument("--python", default=sys.executable,
                        help="Interpreter for the child processes "
                             "(default: the one running this script).")
    # Scale overrides; omitted values come from the tier's profile.
    parser.add_argument("--n-init", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-iterations", type=int, default=None)
    parser.add_argument("--mogp-iters", type=int, default=None)
    parser.add_argument("--output-root", default=None,
                        help="Directory for this sweep's artifacts (default "
                             "matrix_results). Use a fresh name to leave a "
                             "previous sweep untouched — required when the two "
                             "are not comparable, e.g. across an objective "
                             "change.")
    parser.add_argument("--no-report", action="store_true",
                        help="Skip the aggregate table + comparison figure "
                             "that normally run after the last case.")
    parser.add_argument("--lib-pull", type=int, default=None,
                        help="ChEMBL PULL size for the library (~60%% survives "
                             "filtering). Drives both the prereq build and "
                             "run_benchmark_seeds' --lib-size, so every case "
                             "searches one library.")
    args = parser.parse_args()

    # Rebind the artifact paths before anything writes. Kept as module globals
    # because Case.log_path and the writers below read them directly.
    if args.output_root:
        global OUT_ROOT, LOG_DIR, RESULTS_CSV, REPORT_MD
        OUT_ROOT = os.path.abspath(args.output_root)
        LOG_DIR = os.path.join(OUT_ROOT, "logs")
        RESULTS_CSV = os.path.join(OUT_ROOT, "results.csv")
        REPORT_MD = os.path.join(OUT_ROOT, "report.md")

    cases = build_cases(args)

    if args.group:
        cases = [c for c in cases if c.group in args.group]
    if args.only:
        cases = [c for c in cases
                 if any(fnmatch.fnmatch(c.id, p) for p in args.only)]
    if args.skip:
        cases = [c for c in cases
                 if not any(fnmatch.fnmatch(c.id, p) for p in args.skip)]
    if args.resume:
        done = load_previous()
        before = len(cases)
        cases = [c for c in cases if c.id not in done]
        print("--resume: skipping {} previously passing case(s)."
              .format(before - len(cases)))

    for case in cases:
        case.timeout = int(case.timeout * args.timeout_scale)

    if args.list or args.dry_run:
        print("{} case(s) at tier '{}':\n".format(len(cases), args.tier))
        for case in cases:
            if args.dry_run:
                print("# [{}] {}".format(case.group, case.id))
                print(case.pretty())
                print()
            else:
                print("  {:<32} {:<10} {}".format(case.id, case.group,
                                                  case.tier))
        return 0

    if not cases:
        print("No cases selected.")
        return 0

    # Cases needing the library are pointless without it; the prereq case
    # builds it, so only bail when it is absent and the prereq is not running.
    effective_lib = args.arena_dir if args.arena else args.library_dir
    missing_lib = library_size(effective_lib) is None
    building = any(c.group == "prereq" for c in cases)
    if missing_lib and not building and any(c.needs_library for c in cases):
        hint = ("python build_cached_arena.py" if args.arena
                else "python data.py --n-molecules 400")
        print("ERROR: {} is not built, and no prereq case is in this "
              "selection.\n       Build it first:  {}"
              .format(effective_lib, hint))
        return 2

    os.makedirs(LOG_DIR, exist_ok=True)
    env = dict(os.environ)
    env.update(CHILD_ENV)

    manifest = write_manifest(args, cases, effective_lib)

    print("=" * 72)
    print("MOGP-NTD FEATURE MATRIX  |  tier={}  cases={}  jobs={}"
          .format(args.tier, len(cases), args.jobs))
    print("python: {}".format(args.python))
    print("commit: {} ({}){}".format(
        manifest["git"]["commit"][:9] or "unknown",
        manifest["git"]["branch"] or "unknown",
        " DIRTY" if manifest["git"]["dirty"] else ""))
    print("library: {} ({} molecules)".format(
        effective_lib, manifest["library_molecules"]))
    print("started: {}".format(manifest["started"]))
    print("=" * 72)

    rows = []
    start = time.time()

    def report(row, index):
        mark = {"pass": "PASS", "fail": "FAIL", "timeout": "TIME",
                "error": "ERR "}[row["status"]]
        print("[{:>3}/{}] {} {:<34} {:>7.1f}s  {}..{}"
              .format(index, len(cases), mark, row["id"], row["seconds"],
                      row["started_at"][11:], row["ended_at"][11:]))
        if row["status"] != "pass":
            print("          log: {}".format(row["log"]))

    if args.jobs > 1:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = [pool.submit(run_case, c, env) for c in cases]
            for i, future in enumerate(futures, start=1):
                row = future.result()
                rows.append(row)
                report(row, i)
                # Rewrite after every case: a multi-hour sweep that is killed
                # or crashes must still leave a complete record of everything
                # that finished, not just of a run that reached the end.
                write_results(rows)
    else:
        for i, case in enumerate(cases, start=1):
            row = run_case(case, env)
            rows.append(row)
            report(row, i)
            write_results(rows)
            if args.fail_fast and row["status"] != "pass":
                print("\n--fail-fast: stopping at '{}'.".format(case.id))
                break

    write_results(rows)
    write_report(rows, args)

    # Aggregate every case into one comparable table + the overlay figure.
    # Run as a subprocess so this module keeps its "no pandas/matplotlib at
    # import time" property, which is what lets --list/--dry-run work anywhere.
    if not args.no_report and any(r["status"] == "pass" for r in rows):
        print("\nBuilding matrix report...")
        try:
            subprocess.run([args.python, "matrix_report.py",
                            "--sweep-dir", OUT_ROOT],
                           cwd=ROOT, env=env, timeout=1800)
        except (subprocess.TimeoutExpired, OSError) as exc:
            print("  matrix_report failed: {}".format(exc))

    passed = sum(1 for r in rows if r["status"] == "pass")
    print("=" * 72)
    print("{}/{} passed in {:.1f}s".format(passed, len(rows),
                                           time.time() - start))
    print("report:  {}".format(os.path.relpath(REPORT_MD, ROOT)))
    print("results: {}".format(os.path.relpath(RESULTS_CSV, ROOT)))
    print("=" * 72)
    for row in rows:
        if row["status"] != "pass":
            print("  {:<34} {:<8} {}".format(row["id"], row["status"],
                                             row["log"]))
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
