"""
probe_acquisition_cost.py
=========================

A READ-ONLY cost probe for the grey-box qNEHVI acquisition.

It replays the TERMINAL state of a finished ablation arm
(``ablation_icm_vs_independent/armA_coregionalized_seed0``) and runs exactly ONE
GP-train + acquisition-scoring cycle per configuration, measuring wall clock and
peak RSS. It never runs a BO loop, never docks anything, and never modifies any
existing module: alpha / prune_baseline are injected by monkeypatching the
qLogNEHVI constructor *inside this process only*.

Design notes that matter for correctness
----------------------------------------

* **One process per cell.** ``resource.getrusage(RUSAGE_SELF).ru_maxrss`` is a
  high-water mark that never falls, so two cells in one process would report the
  max of the two. The parent re-execs this file with ``--cell N`` per cell and
  reads a JSON file back. Cells run STRICTLY SEQUENTIALLY (48 GB machine).

* **ru_maxrss is BYTES on macOS** (verified: a 1.5 GB allocation moved it by
  1,500,020,736). It is KiB on Linux. The unit actually used is recorded in the
  result JSON as ``rss_unit``.

* **torch.manual_seed(0) immediately before the acquisition call.**
  ``SobolQMCNormalSampler`` with no explicit ``seed=`` draws one from torch's
  GLOBAL RNG, so two qNEHVI calls in one process are NOT reproducible against
  each other. Pinned by ``test_joint_posterior.py``.

* **The environment block below mirrors ``loop.py``'s own module-level
  ``setdefault`` block verbatim** (and ``run_ablation.py``'s
  ``KMP_DUPLICATE_LIB_OK``). It is not an ad-hoc workaround: it reproduces the
  process environment the ablation arms actually ran under, including the
  single-threaded BLAS/OMP settings that materially affect wall clock. Without
  ``KMP_DUPLICATE_LIB_OK`` this environment aborts on ``import botorch``
  (torch's libiomp vs scipy's libomp), so the campaign could not have run
  without it.

Usage
-----
    python probe_acquisition_cost.py                 # prep (if needed) + all 6 cells
    python probe_acquisition_cost.py --smoke         # fast plumbing check
    python probe_acquisition_cost.py --prep-only
    python probe_acquisition_cost.py --cell 1 --out /tmp/cell1.json   # child
"""

import os

# --- Process environment: mirrors loop.py's module-level block exactly. -------
# Must run BEFORE numpy/torch/botorch import their native libraries.
for _thread_var in ("KMP_DUPLICATE_LIB_OK", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(
        _thread_var, "TRUE" if _thread_var == "KMP_DUPLICATE_LIB_OK" else "1"
    )

import sys

# --- PATH: mirrors go.sh lines 43-49. Calling the env's python by absolute path
# --- leaves the env's bin/ OFF PATH, so torch cannot find `ninja` and botorch
# --- silently falls back to the pure-Python qLogEHVI ("Failed to compile fused
# --- qLogEHVI C++ extension") -- roughly 3x slower on exactly the hot path this
# --- probe measures. The campaign ran under go.sh, i.e. WITH the fused kernel,
# --- so the probe must too or its absolute numbers mean nothing.
_ENV_BIN = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), "bin")
if os.path.isdir(_ENV_BIN) and _ENV_BIN not in os.environ.get("PATH", "").split(os.pathsep):
    os.environ["PATH"] = _ENV_BIN + os.pathsep + os.environ.get("PATH", "")

import json
import time
import shutil
import resource
import argparse
import platform
import subprocess
from time import perf_counter

import numpy as np


REPO = os.path.dirname(os.path.abspath(__file__))

ARM_DIR = os.path.join(
    REPO, "ablation_icm_vs_independent", "armA_coregionalized_seed0"
)
EVALUATED_CSV = os.path.join(ARM_DIR, "evaluated.csv")
RUN_CONFIG = os.path.join(ARM_DIR, "run_config.json")

DEFAULT_STATE_DIR = os.path.join(
    "/private/tmp/claude-502/-Users-devansh/"
    "415d2b3e-c25f-4388-9ab1-5611fc6ca7ec/scratchpad",
    "probe_state",
)

RESULTS_JSON = os.path.join(REPO, "probe_results.json")
RESULTS_MD = os.path.join(REPO, "PROBE_RESULTS.md")

# The iteration whose acquisition pool we replay. The campaign's iteration 50 is
# the last one; subsample_candidates is reseeded from (seed, iteration).
REPLAY_ITERATION = 50

# ru_maxrss unit by platform (verified empirically on this machine for Darwin).
RSS_UNIT = "bytes" if platform.system() == "Darwin" else "kib"
RSS_DIVISOR = 1e9 if RSS_UNIT == "bytes" else 1e6      # -> GB (decimal)

# Hard peak-RSS budget for one probe process. This is a 48 GB machine that also
# runs an IDE, a browser and a language server; a 20 GB transient pages it out
# and costs the user their applications. It is also a validity precondition: a
# wall-clock number measured while the system is swapping is noise. Enforced by
# _start_rss_watchdog, not merely reported.
RSS_BUDGET_GB = 12.0

# qNEHVI's dominant allocation is its (n_mc_samples, chunk, num_boxes, m)
# tensor -- acquisition.py's CANDIDATE_CHUNK comment names it explicitly.
# Measured peak / that tensor: 3.05 (coreg diag), 2.73 (coreg joint), 4.71
# (independent diag). 3.0 is used for the PRE-FLIGHT projection so an
# infeasible configuration is refused before it allocates, while still
# reporting num_boxes -- which is the quantity that actually drives the cost.
RSS_PEAK_OVER_TENSOR = 3.0


class BudgetExceeded(Exception):
    """Raised when the projected peak RSS exceeds the budget, pre-allocation."""

    def __init__(self, n_boxes, tensor_gb, projected_gb, cap_gb):
        self.n_boxes = n_boxes
        self.tensor_gb = tensor_gb
        self.projected_gb = projected_gb
        self.cap_gb = cap_gb
        super().__init__(
            f"projected peak {projected_gb:.2f} GB > cap {cap_gb:.2f} GB "
            f"(num_boxes={n_boxes:,}, dominant tensor {tensor_gb:.2f} GB)")


# --------------------------------------------------------------------------- #
# The six configurations.
# --------------------------------------------------------------------------- #
CELLS = [
    {"cell": 1, "model": "coregionalized", "posterior": "diag",
     "alpha": 0.0,   "prune_baseline": False, "extras": True},
    {"cell": 2, "model": "coregionalized", "posterior": "joint",
     "alpha": 0.0,   "prune_baseline": False, "extras": True},
    {"cell": 3, "model": "independent",    "posterior": "diag",
     "alpha": 0.0,   "prune_baseline": False, "extras": False},
    {"cell": 4, "model": "independent",    "posterior": "joint",
     "alpha": 0.0,   "prune_baseline": False, "extras": False},
    {"cell": 5, "model": "coregionalized", "posterior": "diag",
     "alpha": 1e-3,  "prune_baseline": False, "extras": False},
    {"cell": 6, "model": "coregionalized", "posterior": "diag",
     "alpha": 1e-3,  "prune_baseline": True,  "extras": False},
]


def _arm_terminal_history(arm_dir):
    """Return ``(pareto_size, n_evaluated)`` from the arm's LAST history row.

    Read with the stdlib csv module so the per-cell child never imports pandas.
    """
    import csv
    with open(os.path.join(arm_dir, "history.csv")) as fh:
        rows = list(csv.DictReader(fh))
    last = rows[-1]
    return int(last["pareto_size"]), int(last["n_evaluated"])


def _start_rss_watchdog(cap_gb, tag=""):
    """Hard-abort this process if peak RSS crosses ``cap_gb``.

    A 48 GB machine that is also running an IDE and a browser cannot absorb a
    20 GB transient: it pages, the user loses applications, and -- just as
    important for this probe -- a wall-clock number measured while the system is
    swapping is noise, not a measurement. So the cap is enforced rather than
    merely reported. ``ru_maxrss`` is a high-water mark, so this fires just
    after the offending allocation rather than before it; the ladder is walked
    from small B upward so the overshoot stays bounded.
    """
    if not cap_gb:
        return
    import threading

    def _watch():
        while True:
            g = _peak_rss_gb()
            if g > cap_gb:
                sys.stdout.write(
                    f"\nRSS_CAP_EXCEEDED {g:.3f} GB > cap {cap_gb} GB "
                    f"-- aborting {tag}\n")
                sys.stdout.flush()
                os._exit(17)
            time.sleep(0.2)

    threading.Thread(target=_watch, daemon=True).start()


def _peak_rss():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _peak_rss_gb():
    return _peak_rss() / RSS_DIVISOR


# ========================================================================== #
# STAGE A -- prep. Imports rdkit (via data) AND torch (via acquisition); not
# timed, so the heavyweight import mix is irrelevant here.
# ========================================================================== #
def prep(state_dir, arm_dir=ARM_DIR, library_dir="data/library"):
    """Materialize the replayable terminal state into ``state_dir``.

    Writes the FILTERED library arrays (fingerprints / admet_scores) plus a
    meta.json holding the evaluated molecules' library indices, their objective
    matrix, and the exact objective layout ``acquisition._resolve_admet_layout``
    resolves. The per-cell children then need neither rdkit nor pandas.
    """
    import pandas as pd

    os.chdir(REPO)
    from data import load_library, ADMET_COLUMNS
    import acquisition as acq
    from mogp import TASK_NAMES, train_mogp
    import loop as loop_mod
    from mogp_coregionalized import train_mogp_coregionalized

    # Verify the 3-line train_fn dispatch this probe replicates (the child must
    # not import loop.py: that pulls rdkit + the docking oracle into the RSS
    # measurement).
    assert loop_mod.resolve_train_fn("independent") is train_mogp
    _tf = loop_mod.resolve_train_fn("coregionalized", rank=1)
    assert _tf.__name__ == "_train_coregionalized"
    assert loop_mod.BOLoop.__init__.__defaults__ is not None

    lib = load_library(library_dir)
    smiles = list(lib["smiles"])
    fingerprints = np.asarray(lib["fingerprints"])
    admet_scores = np.asarray(lib["admet_scores"])
    library_size = len(smiles)

    # SMILES -> library index. Verified unambiguous: the filtered library has
    # 26,660 rows and 26,660 distinct SMILES.
    if len(set(smiles)) != library_size:
        raise RuntimeError("library SMILES are not unique; index mapping is ambiguous")
    index_of = {s: i for i, s in enumerate(smiles)}

    ev = pd.read_csv(os.path.join(arm_dir, "evaluated.csv"))
    missing = [s for s in ev["SMILES"] if s not in index_of]
    if missing:
        raise RuntimeError(f"{len(missing)} evaluated SMILES absent from the library")
    evaluated_indices = [int(index_of[s]) for s in ev["SMILES"]]
    if len(set(evaluated_indices)) != len(evaluated_indices):
        raise RuntimeError("evaluated SMILES map to duplicate library indices")

    # evaluated.csv writes Y_evaluated column-for-column in TASK_NAMES order
    # (loop.BOLoop.save_results), so this reconstruction is exact.
    Y_evaluated = np.column_stack(
        [ev[name].to_numpy(dtype=np.float64) for name in TASK_NAMES]
    )

    # Cross-check: the library's own ADMET rows for these molecules must equal
    # the ADMET columns evaluated.csv recorded (loop._evaluate copies them
    # straight from admet_scores). This proves the index mapping.
    from mogp import resolve_objective_layout
    library_tasks, _dock_tasks, _ = resolve_objective_layout(ADMET_COLUMNS)
    admet_ok = True
    for j, col in library_tasks:
        a = admet_scores[evaluated_indices, col].astype(np.float64)
        b = Y_evaluated[:, j]
        if not np.allclose(a, b, rtol=0, atol=1e-5, equal_nan=True):
            admet_ok = False
    if not admet_ok:
        raise RuntimeError(
            "ADMET cross-check failed: library rows for the mapped indices do "
            "not match evaluated.csv's ADMET columns."
        )

    layout = acq._resolve_admet_layout()

    os.makedirs(state_dir, exist_ok=True)
    np.save(os.path.join(state_dir, "fingerprints.npy"), fingerprints)
    np.save(os.path.join(state_dir, "admet_scores.npy"), admet_scores)
    np.save(os.path.join(state_dir, "Y_evaluated.npy"), Y_evaluated)

    with open(os.path.join(arm_dir, "run_config.json")) as fh:
        run_config = json.load(fh)

    meta = {
        "arm_dir": arm_dir,
        "library_dir": library_dir,
        "library_size": library_size,
        "n_evaluated": len(evaluated_indices),
        "evaluated_indices": evaluated_indices,
        "task_names": list(TASK_NAMES),
        "admet_columns": list(ADMET_COLUMNS),
        "layout": [list(layout[0]), list(layout[1]), list(layout[2])],
        "run_config": run_config,
        "replay_iteration": REPLAY_ITERATION,
        "admet_cross_check": "passed",
    }
    with open(os.path.join(state_dir, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    print(f"[prep] library={library_size}  evaluated={len(evaluated_indices)}  "
          f"layout={layout}")
    print(f"[prep] wrote state to {state_dir}")
    return meta


# ========================================================================== #
# STAGE B -- one cell, in its own process. torch only, no rdkit, no loop.py.
# ========================================================================== #
def run_cell(spec, state_dir, out_path, smoke=False,
             baseline_cap=None, rss_cap=None, no_instr=False):
    import torch
    import acquisition as acq
    from mogp import train_mogp
    from mogp_coregionalized import train_mogp_coregionalized
    from kernel import TanimotoKernel
    from botorch.acquisition.multi_objective import logei as _logei

    os.chdir(REPO)

    # Load (compiling on first use) botorch's fused C++ qLogEHVI kernel BEFORE
    # the timed section, so a one-off compile never lands inside a cell's wall
    # clock. Requires `ninja` on PATH, which the module header arranges.
    _logei._try_load_fused_kernel()
    fused_kernel = _logei._C is not None
    print(f"[cell {spec['cell']}] botorch fused qLogEHVI C++ kernel: "
          f"{'LOADED' if fused_kernel else 'NOT loaded (pure-Python fallback)'}",
          flush=True)

    _start_rss_watchdog(rss_cap, tag=f"cell {spec['cell']}")

    # BOLoop.__init__ seeds both RNGs once from the run seed (0 here).
    np.random.seed(0)
    torch.manual_seed(0)

    with open(os.path.join(state_dir, "meta.json")) as fh:
        meta = json.load(fh)

    layout = tuple(tuple(x) for x in meta["layout"])
    # Keep loop.py's exact call signature (it never passes `layout`), while
    # avoiding the rdkit import `_resolve_admet_layout` would trigger. The tuple
    # was produced by the REAL `_resolve_admet_layout()` during prep.
    acq._resolve_admet_layout = lambda: layout

    fp_all = np.load(os.path.join(state_dir, "fingerprints.npy"), mmap_mode="r")
    admet_all = np.load(os.path.join(state_dir, "admet_scores.npy"), mmap_mode="r")
    Y_evaluated = np.load(os.path.join(state_dir, "Y_evaluated.npy"))
    evaluated_indices = np.asarray(meta["evaluated_indices"], dtype=int)
    library_size = int(meta["library_size"])
    rc = meta["run_config"]

    pool_size = int(rc["acquisition_pool_size"])
    batch_size = int(rc["batch_size"])
    diversity_threshold = float(rc["diversity_threshold"])
    mogp_iters = int(rc["mogp_iters"])
    seed = int(rc["seed"])
    # Smoke mode is a PLUMBING check only. The dominant real cost is the box
    # decomposition over the B-point baseline, which is independent of the pool
    # size, so the baseline is capped too or the "fast" check is not fast.
    if smoke:
        pool_size, mogp_iters = 64, 5
        if baseline_cap is None:
            baseline_cap = 40

    # ---------------------------------------------------------------- #
    # Instrumentation. All of it is installed by monkeypatch in THIS
    # process; acquisition.py / mogp.py on disk are untouched.
    # ---------------------------------------------------------------- #
    S = {
        "gp_predict_s": 0.0,        # inside mogp.predict / mogp.predict_joint
        "posterior_s": 0.0,         # whole DockingPosteriorModel.posterior call
        "acqf_init_s": 0.0,         # qLogNEHVI constructor (initial box decomp)
        "instr_s": 0.0,             # measurement overhead, subtracted out
        "n_posterior_calls": 0,
        "n_predict_calls": 0,
        "rows": [],                 # per predict-call row accounting
        "n_boxes": None,
        "acqf_baseline_rows": None,
        "acqf_kwargs": None,
    }
    ctx = {"q": None, "batch": None, "phase": "init"}
    measure_redundancy = bool(spec.get("extras")) and not no_instr

    # --- alpha / prune_baseline injection (probe-local; acquisition.py never
    # --- passes alpha, and hard-codes prune_baseline=False).
    _RealAcqf = acq.qLogNoisyExpectedHypervolumeImprovement

    def _acqf_factory(*args, **kwargs):
        kwargs["alpha"] = float(spec["alpha"])
        kwargs["prune_baseline"] = bool(spec["prune_baseline"])
        S["acqf_kwargs"] = {"alpha": kwargs["alpha"],
                            "prune_baseline": kwargs["prune_baseline"],
                            "cache_root": kwargs.get("cache_root")}
        t = perf_counter()
        obj = _RealAcqf(*args, **kwargs)
        S["acqf_init_s"] += perf_counter() - t
        try:
            S["n_boxes"] = int(obj.cell_lower_bounds.shape[-2])
        except Exception:
            S["n_boxes"] = None
        try:
            # After construction this is the (possibly PRUNED) baseline, so it
            # shows whether prune_baseline=True actually removed anything.
            S["acqf_baseline_rows"] = int(obj.X_baseline.shape[0])
        except Exception:
            S["acqf_baseline_rows"] = None

        # PRE-FLIGHT budget check. num_boxes is known here, before the first
        # forward pass allocates anything large, so an infeasible configuration
        # is refused while still yielding the number that explains why.
        if rss_cap and S["n_boxes"]:
            tensor_gb = (acq.N_MC_SAMPLES * acq.CANDIDATE_CHUNK
                         * S["n_boxes"] * len(acq.TASK_NAMES) * 8) / 1e9
            projected = _peak_rss_gb() + RSS_PEAK_OVER_TENSOR * tensor_gb
            if projected > rss_cap:
                raise BudgetExceeded(S["n_boxes"], tensor_gb, projected,
                                     rss_cap)
        return obj

    acq.qLogNoisyExpectedHypervolumeImprovement = _acqf_factory

    # --- GP prediction timing + row accounting (diag path) ---
    _real_predict = acq.predict

    def _predict(model, likelihood, y_mean, y_std, X_new, *a, **k):
        n_rows = int(np.asarray(X_new).shape[0])
        t = perf_counter()
        out = _real_predict(model, likelihood, y_mean, y_std, X_new, *a, **k)
        S["gp_predict_s"] += perf_counter() - t
        S["n_predict_calls"] += 1
        ti = perf_counter()
        n_unique = None
        if measure_redundancy:
            arr = np.asarray(X_new)
            key = np.packbits(arr.astype(np.uint8), axis=1)
            n_unique = int(np.unique(key, axis=0).shape[0])
        S["instr_s"] += perf_counter() - ti
        S["rows"].append({"phase": ctx["phase"], "q": ctx["q"],
                          "batch": ctx["batch"], "rows_to_gp": n_rows,
                          "unique_rows": n_unique})
        return out

    acq.predict = _predict

    # --- GP prediction timing (joint path) ---
    _real_predict_joint = acq.predict_joint

    def _predict_joint(model, likelihood, y_mean, y_std, X_new, *a, **k):
        n_rows = int(np.asarray(X_new).shape[0])
        t = perf_counter()
        out = _real_predict_joint(model, likelihood, y_mean, y_std, X_new, *a, **k)
        S["gp_predict_s"] += perf_counter() - t
        S["n_predict_calls"] += 1
        S["rows"].append({"phase": ctx["phase"], "q": ctx["q"],
                          "batch": ctx["batch"], "rows_to_gp": n_rows,
                          "unique_rows": n_rows,
                          "rows_before_dedup": UNIQ["last_in"]})
        return out

    acq.predict_joint = _predict_joint

    # --- exact dedup accounting for the joint path (free: the joint path
    # --- already computes it) ---
    UNIQ = {"last_in": None}
    _real_unique_rows = acq._unique_rows

    def _unique_rows(rows):
        UNIQ["last_in"] = int(np.asarray(rows).shape[0])
        return _real_unique_rows(rows)

    acq._unique_rows = _unique_rows

    # --- whole-posterior timing + progress ---
    _real_posterior = acq.DockingPosteriorModel.posterior
    progress = {"n": 0, "t0": None}

    def _posterior(self, X, *a, **k):
        *batch, q, _ = X.shape
        ctx["q"] = int(q)
        ctx["batch"] = [int(b) for b in batch]
        t = perf_counter()
        out = _real_posterior(self, X, *a, **k)
        S["posterior_s"] += perf_counter() - t
        S["n_posterior_calls"] += 1
        if ctx["phase"] == "score":
            progress["n"] += 1
            el = perf_counter() - progress["t0"]
            print(f"    [chunk {progress['n']:>3}] batch={ctx['batch']} q={q} "
                  f"elapsed={el:7.1f}s peak_rss={_peak_rss_gb():5.2f} GB",
                  flush=True)
        return out

    acq.DockingPosteriorModel.posterior = _posterior

    # --- compute_qnehvi total ---
    _real_compute_qnehvi = acq.compute_qnehvi
    QN = {"t": 0.0}

    def _compute_qnehvi(*a, **k):
        ctx["phase"] = "init"
        progress["t0"] = perf_counter()
        t = perf_counter()
        out = _real_compute_qnehvi(*a, **k)
        QN["t"] = perf_counter() - t
        return out

    acq.compute_qnehvi = _compute_qnehvi

    # After the acqf is constructed the remaining posterior calls are the
    # per-chunk candidate scans; flip the phase there.
    def _flip_phase_after_init(obj):
        ctx["phase"] = "score"
        return obj

    _factory_inner = _acqf_factory

    def _acqf_factory2(*args, **kwargs):
        obj = _factory_inner(*args, **kwargs)
        return _flip_phase_after_init(obj)

    acq.qLogNoisyExpectedHypervolumeImprovement = _acqf_factory2

    # ---------------------------------------------------------------- #
    # Replay loop.BOLoop.step's inputs verbatim.
    # ---------------------------------------------------------------- #
    active = acq.get_active_objectives(Y_evaluated)
    finite_rows = np.isfinite(Y_evaluated[:, active]).all(axis=1)

    # --- Reconstruction assertion. Recompute the arm's OWN terminal Pareto
    # --- front from the replayed state (exactly as loop.BOLoop._pareto_mask
    # --- does) and require it to match the front the arm actually recorded.
    # --- A broken SMILES->library-index mapping, a wrong NaN filter or a
    # --- mis-ordered objective matrix all move this number, so it is the check
    # --- that fails loudly instead of silently measuring the wrong state.
    signs_active = np.asarray(acq.DEFAULT_OBJECTIVE_SIGNS, dtype=float)[active]
    front_mask, _ = acq.compute_pareto_front(
        Y_evaluated[:, active][finite_rows], signs_active
    )
    recon_front = int(front_mask.sum())
    n_baseline_uncapped = int(finite_rows.sum())
    exp_front, exp_n_eval = _arm_terminal_history(meta["arm_dir"])
    if recon_front != exp_front or len(evaluated_indices) != exp_n_eval:
        raise AssertionError(
            "Replayed state does not reproduce the arm's terminal record: "
            f"reconstructed Pareto front {recon_front} vs recorded {exp_front}; "
            f"evaluated {len(evaluated_indices)} vs recorded {exp_n_eval}. "
            f"(B={n_baseline_uncapped} finite of {len(evaluated_indices)}.)"
        )
    print(f"[cell {spec['cell']}] reconstruction OK: Pareto front "
          f"{recon_front} == recorded {exp_front}; "
          f"B={n_baseline_uncapped} finite of {len(evaluated_indices)} "
          f"({len(evaluated_indices) - n_baseline_uncapped} dropped for a NaN "
          f"in an active objective)", flush=True)

    baseline_library_indices = evaluated_indices[finite_rows]
    train_y_full = Y_evaluated[finite_rows]
    if baseline_cap is not None:
        # Replay a PREFIX of the arm's own trajectory. evaluated.csv is in
        # evaluation order, so the first B successfully-docked rows are exactly
        # the campaign's state after (B - n_init) / batch_size iterations -- not
        # an arbitrary subsample.
        baseline_library_indices = baseline_library_indices[:baseline_cap]
        train_y_full = train_y_full[:baseline_cap]
    train_x = np.ascontiguousarray(fp_all[baseline_library_indices])
    train_y = train_y_full.astype(np.float32)

    train_fn = (train_mogp if spec["model"] == "independent"
                else (lambda tx, ty, n_iterations=200, lr=0.1:
                      train_mogp_coregionalized(tx, ty, n_iterations=n_iterations,
                                                lr=lr, rank=1)))

    print(f"[cell {spec['cell']}] model={spec['model']} "
          f"posterior={spec['posterior']} alpha={spec['alpha']} "
          f"prune_baseline={spec['prune_baseline']}", flush=True)
    print(f"[cell {spec['cell']}] GP training set: {train_x.shape[0]}/"
          f"{len(evaluated_indices)} finite rows, active objectives={active}",
          flush=True)

    t_cell = perf_counter()
    t0 = perf_counter()
    model, likelihood, y_mean, y_std = train_fn(
        train_x, train_y, n_iterations=mogp_iters, lr=0.1
    )
    gp_train_s = perf_counter() - t0
    rss_after_train = _peak_rss_gb()

    # --- pool construction (loop.py counts this as acquisition time) ---
    t_acq = perf_counter()
    evaluated_set = set(int(i) for i in evaluated_indices)
    candidate_library_indices = np.array(
        [i for i in range(library_size) if i not in evaluated_set], dtype=int
    )
    n_unevaluated = int(len(candidate_library_indices))
    candidate_library_indices = acq.subsample_candidates(
        candidate_library_indices, pool_size, seed, REPLAY_ITERATION,
    )
    n_candidates_scored = int(len(candidate_library_indices))
    X_candidates = np.ascontiguousarray(fp_all[candidate_library_indices])
    candidate_admet = np.ascontiguousarray(admet_all[candidate_library_indices])
    baseline_admet = np.ascontiguousarray(admet_all[baseline_library_indices])
    pool_prep_s = perf_counter() - t_acq

    print(f"[cell {spec['cell']}] pool: {n_candidates_scored} of "
          f"{n_unevaluated} unevaluated (seed={seed}, iteration="
          f"{REPLAY_ITERATION}); baseline B={train_x.shape[0]}", flush=True)

    # --- THE acquisition call. Seed immediately before it: the Sobol sampler
    # --- draws its seed from torch's global RNG.
    torch.manual_seed(0)
    t_sel = perf_counter()
    try:
        selected_local, selected_scores = acq.select_batch(
            model, likelihood, y_mean, y_std,
            X_candidates, candidate_admet,
            train_x, baseline_admet,
            batch_size=batch_size,
            diversity_threshold=diversity_threshold,
            posterior_mode=spec["posterior"],
        )
    except BudgetExceeded as exc:
        result = {
            "cell": spec["cell"], "model": spec["model"],
            "posterior": spec["posterior"], "alpha": spec["alpha"],
            "prune_baseline": spec["prune_baseline"],
            "status": "projected_over_budget",
            "baseline_cap": baseline_cap,
            "n_baseline": int(train_x.shape[0]),
            "n_baseline_uncapped": n_baseline_uncapped,
            "n_candidates_scored": n_candidates_scored,
            "n_boxes": exc.n_boxes,
            "dominant_tensor_gb": exc.tensor_gb,
            "projected_peak_rss_gb": exc.projected_gb,
            "rss_cap_gb": exc.cap_gb,
            "peak_rss_gb": _peak_rss() / RSS_DIVISOR,
            "rss_unit": RSS_UNIT,
            "gp_train_s": gp_train_s,
            "acqf_init_s": S["acqf_init_s"],
            "reason": str(exc),
        }
        with open(out_path, "w") as fh:
            json.dump(result, fh, indent=2)
        print(f"BUDGET_REFUSED {exc}", flush=True)
        return result
    select_batch_s = perf_counter() - t_sel
    acquisition_s_raw = perf_counter() - t_acq
    wall_clock_s_raw = perf_counter() - t_cell

    # Subtract the measurement overhead we introduced (unique-row counting).
    instr_s = S["instr_s"]
    acquisition_s = acquisition_s_raw - instr_s
    wall_clock_s = wall_clock_s_raw - instr_s
    qnehvi_s = QN["t"] - instr_s
    diversity_s = select_batch_s - QN["t"]

    gp_predict_s = S["gp_predict_s"]
    posterior_assembly_s = S["posterior_s"] - gp_predict_s
    acqf_forward_s = qnehvi_s - S["posterior_s"] - S["acqf_init_s"]
    acqf_s = acquisition_s - gp_predict_s

    # --- batch diversity ---
    selected_library_indices = [int(i) for i in
                                candidate_library_indices[selected_local]]
    sel_fp = torch.from_numpy(
        np.asarray(X_candidates[selected_local])
    ).to(torch.float32)
    sims = TanimotoKernel().forward(sel_fp, sel_fp).detach().cpu().numpy()
    n_sel = sims.shape[0]
    iu = np.triu_indices(n_sel, 1)
    pair = sims[iu].astype(float)
    diversity = {
        "n_selected": int(n_sel),
        "selected_local_indices": [int(i) for i in selected_local],
        "selected_library_indices": selected_library_indices,
        "selected_qnehvi_scores": [float(s) for s in selected_scores],
        "mean_pairwise_tanimoto": float(pair.mean()) if pair.size else None,
        "max_pairwise_tanimoto": float(pair.max()) if pair.size else None,
        "min_pairwise_tanimoto": float(pair.min()) if pair.size else None,
        "pairwise_matrix": [[float(v) for v in row] for row in sims],
    }

    # --- redundancy accounting ---
    score_rows = [r for r in S["rows"] if r["phase"] == "score"]
    total_rows = int(sum(r["rows_to_gp"] for r in score_rows))
    if spec["posterior"] == "joint":
        # predict_joint receives the DEDUPLICATED matrix; the pre-dedup count is
        # what posterior() was handed.
        rows_in = [r.get("rows_before_dedup") for r in score_rows]
        uniq = [r["unique_rows"] for r in score_rows]
        total_presented = int(sum(x for x in rows_in if x is not None))
    else:
        rows_in = [r["rows_to_gp"] for r in score_rows]
        uniq = [r["unique_rows"] for r in score_rows]
        total_presented = total_rows
    total_unique = (int(sum(u for u in uniq if u is not None))
                    if all(u is not None for u in uniq) and uniq else None)
    redundancy = {
        "measured": measure_redundancy or spec["posterior"] == "joint",
        "n_scoring_chunks": len(score_rows),
        "rows_presented_to_posterior_total": total_presented,
        "rows_passed_to_gp_predict_total": total_rows,
        "unique_rows_total": total_unique,
        "ratio_presented_over_unique": (
            float(total_presented) / total_unique if total_unique else None),
        "per_chunk": [
            {"chunk": i + 1, "q": r["q"], "batch": r["batch"],
             "rows_presented": (r.get("rows_before_dedup")
                                if spec["posterior"] == "joint"
                                else r["rows_to_gp"]),
             "rows_to_gp": r["rows_to_gp"],
             "unique_rows": r["unique_rows"]}
            for i, r in enumerate(score_rows)
        ],
    }

    peak = _peak_rss()
    result = {
        "cell": spec["cell"],
        "model": spec["model"],
        "posterior": spec["posterior"],
        "alpha": spec["alpha"],
        "prune_baseline": spec["prune_baseline"],
        "status": "ok",
        "smoke": bool(smoke),
        "fused_qlogehvi_kernel": bool(fused_kernel),
        "wall_clock_s": wall_clock_s,
        "wall_clock_s_raw": wall_clock_s_raw,
        "peak_rss_bytes_raw": peak,
        "rss_unit": RSS_UNIT,
        "peak_rss_gb": peak / RSS_DIVISOR,
        "peak_rss_gb_after_gp_train": rss_after_train,
        "gp_train_s": gp_train_s,
        "acquisition_s": acquisition_s,
        "pool_prep_s": pool_prep_s,
        "qnehvi_s": qnehvi_s,
        "diversity_s": diversity_s,
        "gp_predict_s": gp_predict_s,
        "acqf_s": acqf_s,
        "posterior_total_s": S["posterior_s"],
        "posterior_assembly_s": posterior_assembly_s,
        "acqf_init_s": S["acqf_init_s"],
        "acqf_forward_s": acqf_forward_s,
        "instrumentation_s": instr_s,
        "n_posterior_calls": S["n_posterior_calls"],
        "n_predict_calls": S["n_predict_calls"],
        "n_boxes": S["n_boxes"],
        "acqf_baseline_rows": S["acqf_baseline_rows"],
        "acqf_kwargs": S["acqf_kwargs"],
        "n_baseline": int(train_x.shape[0]),
        "n_baseline_uncapped": n_baseline_uncapped,
        "baseline_cap": baseline_cap,
        "rss_cap_gb": rss_cap,
        "instrumented_unique_rows": bool(measure_redundancy),
        "reconstructed_pareto_front": recon_front,
        "recorded_pareto_front": exp_front,
        "n_evaluated": int(len(evaluated_indices)),
        "n_unevaluated": n_unevaluated,
        "n_candidates_scored": n_candidates_scored,
        "candidate_chunk": int(acq.CANDIDATE_CHUNK),
        "n_mc_samples": int(acq.N_MC_SAMPLES),
        "mogp_iters": mogp_iters,
        "batch_size": batch_size,
        "diversity_threshold": diversity_threshold,
        "redundancy": redundancy,
        "batch_diversity": diversity,
        "exceeded_rss_budget": bool(peak / RSS_DIVISOR > RSS_BUDGET_GB),
    }

    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print("PROBE_JSON " + json.dumps(
        {k: v for k, v in result.items()
         if k not in ("redundancy", "batch_diversity")}), flush=True)
    return result


# ========================================================================== #
# Parent orchestrator
# ========================================================================== #
def _spawn_cell(n, state_dir, out_path, log_path, smoke, baseline_cap,
                rss_cap, no_instr, label):
    """Run one cell in its OWN process and return (returncode, result-or-None)."""
    if os.path.exists(out_path):
        os.remove(out_path)
    cmd = [sys.executable, os.path.abspath(__file__), "--cell", str(n),
           "--state-dir", state_dir, "--out", out_path]
    if smoke:
        cmd.append("--smoke")
    if baseline_cap is not None:
        cmd += ["--baseline-cap", str(baseline_cap)]
    if rss_cap:
        cmd += ["--rss-cap", str(rss_cap)]
    if no_instr:
        cmd.append("--no-instr")
    print(f"\n=== {label} ===", flush=True)
    t0 = time.time()
    with open(log_path, "w") as lf:
        proc = subprocess.run(cmd, cwd=REPO, stdout=lf, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    res = None
    if proc.returncode == 0 and os.path.exists(out_path):
        with open(out_path) as fh:
            res = json.load(fh)
        res["parent_elapsed_s"] = elapsed
        res["log"] = log_path
    return proc.returncode, res, elapsed


def _cap_exceeded_rss(log_path):
    """Peak RSS reported by the watchdog just before it aborted a child."""
    try:
        with open(log_path) as fh:
            for line in fh:
                if line.startswith("RSS_CAP_EXCEEDED"):
                    return float(line.split()[1])
    except OSError:
        pass
    return None


def run_ladder(state_dir, b_values, rss_cap, log_dir=None, no_instr=True,
               spec=None):
    """Peak RSS as a function of baseline size B, at the REAL pool size.

    One process per rung, ascending, stopping at the first rung that exceeds the
    budget (or trips the watchdog). Establishes both the budget-feasible B and
    the mechanism: ``n_boxes`` is recorded alongside, because qNEHVI's dominant
    allocation is the (n_mc, chunk, n_boxes, m) tensor.
    """
    log_dir = log_dir or os.path.join(state_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    spec = spec or CELLS[0]
    n = spec["cell"]
    rungs = []
    for b in b_values:
        out_path = os.path.join(log_dir, f"ladder_B{b}.json")
        log_path = os.path.join(log_dir, f"ladder_B{b}.log")
        rc, res, elapsed = _spawn_cell(
            n, state_dir, out_path, log_path, False, b, rss_cap, no_instr,
            f"ladder B={b}: {spec['model']} / {spec['posterior']} / "
            f"alpha={spec['alpha']}")
        if rc == 0 and res:
            rung = {"B": b, "status": res.get("status", "ok"),
                    "peak_rss_gb": res.get("peak_rss_gb"),
                    "n_boxes": res.get("n_boxes"),
                    "dominant_tensor_gb": res.get("dominant_tensor_gb"),
                    "projected_peak_rss_gb": res.get("projected_peak_rss_gb"),
                    "wall_clock_s": res.get("wall_clock_s"),
                    "acquisition_s": res.get("acquisition_s"),
                    "gp_predict_s": res.get("gp_predict_s"),
                    "acqf_s": res.get("acqf_s"),
                    "acqf_init_s": res.get("acqf_init_s"),
                    "acqf_forward_s": res.get("acqf_forward_s"),
                    "n_candidates_scored": res.get("n_candidates_scored"),
                    "log": log_path}
            if rung["status"] == "ok":
                print(f"    ok  peak={rung['peak_rss_gb']:.2f} GB  "
                      f"boxes={rung['n_boxes']:,}  "
                      f"wall={rung['wall_clock_s']:.1f}s", flush=True)
            else:
                print(f"    REFUSED pre-flight: boxes={rung['n_boxes']:,}  "
                      f"tensor={rung['dominant_tensor_gb']:.2f} GB  "
                      f"projected={rung['projected_peak_rss_gb']:.2f} GB",
                      flush=True)
        elif rc == 17:
            g = _cap_exceeded_rss(log_path)
            rung = {"B": b, "status": "rss_cap_exceeded",
                    "peak_rss_gb": g, "n_boxes": None,
                    "wall_clock_s": None, "log": log_path}
            print(f"    ABORTED by watchdog at {g} GB (cap {rss_cap} GB)",
                  flush=True)
        else:
            rung = {"B": b, "status": "failed", "returncode": rc,
                    "peak_rss_gb": None, "n_boxes": None, "log": log_path}
            print(f"    FAILED rc={rc}", flush=True)
        rungs.append(rung)
        with open(os.path.join(log_dir, "ladder.json"), "w") as fh:
            json.dump(rungs, fh, indent=2)
        if rung["status"] != "ok" or (rung["peak_rss_gb"] or 0) > RSS_BUDGET_GB:
            print(f"    stopping ladder at B={b}", flush=True)
            break
    return rungs


def run_all(state_dir, cells, smoke=False, log_dir=None, baseline_cap=None,
            rss_cap=None, no_instr=False):
    log_dir = log_dir or os.path.join(state_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    results = []
    for spec in cells:
        n = spec["cell"]
        out_path = os.path.join(log_dir, f"cell{n}.json")
        log_path = os.path.join(log_dir, f"cell{n}.log")
        if os.path.exists(out_path):
            os.remove(out_path)
        cmd = [sys.executable, os.path.abspath(__file__), "--cell", str(n),
               "--state-dir", state_dir, "--out", out_path]
        if smoke:
            cmd.append("--smoke")
        if baseline_cap is not None:
            cmd += ["--baseline-cap", str(baseline_cap)]
        if rss_cap:
            cmd += ["--rss-cap", str(rss_cap)]
        if no_instr:
            cmd.append("--no-instr")
        print(f"\n=== cell {n}: {spec['model']} / {spec['posterior']} / "
              f"alpha={spec['alpha']} / prune={spec['prune_baseline']} ===",
              flush=True)
        t0 = time.time()
        with open(log_path, "w") as lf:
            proc = subprocess.run(cmd, cwd=REPO, stdout=lf,
                                  stderr=subprocess.STDOUT)
        elapsed = time.time() - t0
        if proc.returncode == 0 and os.path.exists(out_path):
            with open(out_path) as fh:
                res = json.load(fh)
            res["parent_elapsed_s"] = elapsed
            res["log"] = log_path
            if res.get("status") == "ok":
                print(f"    ok  wall={res['wall_clock_s']:.1f}s  "
                      f"peak={res['peak_rss_gb']:.2f} GB", flush=True)
            else:
                print(f"    {res.get('status')}: {res.get('reason', '')}",
                      flush=True)
        else:
            tail = ""
            if os.path.exists(log_path):
                with open(log_path) as fh:
                    tail = "".join(fh.readlines()[-40:])
            res = dict(spec)
            res.update({"status": "failed", "returncode": proc.returncode,
                        "parent_elapsed_s": elapsed, "log": log_path,
                        "error_tail": tail})
            print(f"    FAILED rc={proc.returncode} after {elapsed:.1f}s",
                  flush=True)
        results.append(res)
        with open(os.path.join(log_dir, "partial_results.json"), "w") as fh:
            json.dump(results, fh, indent=2)
    return results


def _fmt(v, nd=1):
    if v is None:
        return "n/a"
    return f"{v:.{nd}f}"


def _load_json(path, default=None):
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def write_reports(results, meta, smoke=False, log_dir=None):
    log_dir = log_dir or os.path.join(DEFAULT_STATE_DIR, "logs")
    ladder = _load_json(os.path.join(log_dir, "ladder_full.json"),
                        _load_json(os.path.join(log_dir, "ladder.json"), []))
    for extra in ("ladder_B100.json",):
        d = _load_json(os.path.join(log_dir, extra))
        if d:
            ladder = ladder + [{"B": d.get("baseline_cap"), "status": "ok",
                                "peak_rss_gb": d.get("peak_rss_gb"),
                                "n_boxes": d.get("n_boxes"),
                                "wall_clock_s": d.get("wall_clock_s")}]
    ladder = sorted([r for r in ladder if r.get("B")], key=lambda r: r["B"])
    uncapped = []
    up = os.path.join(os.path.dirname(log_dir), "logs_uncapped_B284")
    if os.path.isdir(up):
        for fn in sorted(os.listdir(up)):
            d = _load_json(os.path.join(up, fn))
            if d:
                uncapped.append(d)
    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "python": sys.executable,
        "platform": platform.platform(),
        "rss_unit": RSS_UNIT,
        "rss_unit_note": (
            "resource.getrusage(RUSAGE_SELF).ru_maxrss is BYTES on macOS "
            "(verified: a 1.5 GB allocation moved it by 1,500,020,736)."),
        "smoke": bool(smoke),
        "replayed_state": {
            "arm_dir": meta["arm_dir"],
            "library_size": meta["library_size"],
            "n_evaluated": meta["n_evaluated"],
            "replay_iteration": meta["replay_iteration"],
            "run_config": meta["run_config"],
            "admet_cross_check": meta["admet_cross_check"],
        },
        "rss_budget_gb": RSS_BUDGET_GB,
        "rss_vs_baseline_ladder": ladder,
        "uncapped_B284_reference": [
            {k: v for k, v in d.items()
             if k not in ("redundancy", "batch_diversity")} for d in uncapped],
        "cells": results,
    }
    with open(RESULTS_JSON, "w") as fh:
        json.dump(payload, fh, indent=2)

    ok = [r for r in results if r.get("status") == "ok"]
    by_cell = {r["cell"]: r for r in ok}

    L = []
    A = L.append
    A("# Acquisition cost probe -- terminal-state replay\n")
    A(f"Generated {payload['generated']} on {platform.platform()}.")
    if smoke:
        A("\n**SMOKE RUN -- reduced pool / GP iterations. Not a real measurement.**")
    A("")
    A("Replayed state: `ablation_icm_vs_independent/armA_coregionalized_seed0` "
      "(terminal, 290 evaluated, final Pareto front 162).")
    A("")
    b_used = ok[0].get("baseline_cap") if ok else None
    if b_used:
        A(f"> ## Measured at B = {b_used}, not B = 284")
        A(f">")
        A(f"> Peak RSS is bounded by a hard **{RSS_BUDGET_GB:g} GB** budget on "
          f"this 48 GB machine (which also runs an IDE, a browser and a "
          f"language server). The RSS-vs-B ladder below shows the full "
          f"terminal state, B = 284, costs **24.3 GB** for a single "
          f"acquisition call -- twice the budget. B = {b_used} is the largest "
          f"rung that fits.")
        A(f">")
        A(f"> B = {b_used} is not an arbitrary subsample: `evaluated.csv` is in "
          f"evaluation order, so the first {b_used} successfully-docked rows "
          f"are exactly the campaign's own state after "
          f"{(b_used - 40) // 5} BO iterations. Every cell below is a faithful "
          f"replay of that earlier point on the same trajectory.")
        A(f">")
        A(f"> A wall-clock number measured while the machine is swapping is "
          f"noise, so the budget is a validity precondition, not only a "
          f"stability one. Cells are aborted by a watchdog rather than allowed "
          f"to page.")
    if ok:
        r0 = ok[0]
        nb_full = r0.get("n_baseline_uncapped", r0["n_baseline"])
        A(f"The terminal state has {nb_full} finite rows of "
          f"{r0['n_evaluated']} evaluated "
          f"({r0['n_evaluated'] - nb_full} dropped for a NaN in an active "
          f"objective). **The GP training set / qNEHVI baseline actually used "
          f"below is B = {r0['n_baseline']}**, for the budget reason above.")
        if r0.get("reconstructed_pareto_front") is not None:
            A(f"Reconstruction assertion: the replayed state's Pareto front is "
              f"**{r0['reconstructed_pareto_front']}**, matching the "
              f"**{r0['recorded_pareto_front']}** the arm recorded in its own "
              f"`history.csv`. Every cell asserts this before measuring.")
        if smoke and r0.get("smoke_baseline_cap"):
            A(f"**SMOKE: the baseline was capped to "
              f"{r0['smoke_baseline_cap']} rows**, so B used for the "
              f"measurement was {r0['n_baseline']}, not {nb_full}. The box "
              f"decomposition dominates cost and scales with B, so no smoke "
              f"number is comparable to a real one.")
        A(f"Candidate pool: **{r0['n_candidates_scored']}** drawn by "
          f"`acquisition.subsample_candidates` from {r0['n_unevaluated']} "
          f"unevaluated library molecules (seed 0, iteration "
          f"{meta['replay_iteration']}).")
        A(f"`CANDIDATE_CHUNK = {r0['candidate_chunk']}`, "
          f"`N_MC_SAMPLES = {r0['n_mc_samples']}`, "
          f"GP Adam steps = {r0['mogp_iters']}, batch_size = {r0['batch_size']}, "
          f"diversity_threshold = {r0['diversity_threshold']}.")
    A("")
    A("Peak RSS is `resource.getrusage(RUSAGE_SELF).ru_maxrss` read inside each "
      "child process. **On macOS that value is BYTES**, verified empirically "
      "(a 1.5 GB allocation moved it by 1,500,020,736); it is divided by 1e9 "
      "for the GB column.")
    A("")
    if ladder:
        A("## Peak RSS vs baseline size B (coregionalized / diag / alpha=0, "
          "pool = 2000)\n")
        A("| B | BO iteration replayed | peak RSS (GB) | num_boxes | "
          "wall clock (s) | status |")
        A("|---:|---:|---:|---:|---:|---|")
        for r in ladder:
            it = (r["B"] - 40) // 5 if r["B"] >= 40 else 0
            boxes = f"{r['n_boxes']:,}" if r.get("n_boxes") else "n/a"
            A(f"| {r['B']} | {it} | {_fmt(r.get('peak_rss_gb'), 2)} | {boxes} | "
              f"{_fmt(r.get('wall_clock_s'), 1)} | {r.get('status')} |")
        for d in uncapped:
            if d.get("posterior") == "diag" and d.get("model") == "coregionalized":
                A(f"| {d['n_baseline']} | {(d['n_baseline'] - 40) // 5} | "
                  f"{d['peak_rss_gb']:.2f} | {d['n_boxes']:,} | "
                  f"{d['wall_clock_s']:.1f} | measured before the budget was "
                  f"imposed |")
        A("")
        A("Rungs marked `rss_cap_exceeded` were aborted by the watchdog at the "
          "RSS shown; their true peak is higher, and the figure is the value "
          "the 0.2 s poll happened to catch, so those two rows are lower "
          "bounds and are NOT comparable to each other (B=100 reading above "
          "B=120 is a polling artifact, not a non-monotonicity). The "
          "mechanism is explicit: "
          "qNEHVI materializes an `(n_mc_samples, chunk, num_boxes, m)` tensor "
          "-- the allocation `acquisition.py`'s own `CANDIDATE_CHUNK` comment "
          "names -- so peak RSS tracks `num_boxes`, and `num_boxes` grows "
          "steeply with the size of the baseline Pareto front.")
        A("")
        A("| config | B | num_boxes | dominant tensor (GB) | measured peak (GB) "
          "| peak / tensor |")
        A("|---|---:|---:|---:|---:|---:|")
        for d in uncapped:
            t = (d["n_mc_samples"] * d["candidate_chunk"] * d["n_boxes"] * 5
                 * 8) / 1e9
            A(f"| {d['model']} / {d['posterior']} | {d['n_baseline']} | "
              f"{d['n_boxes']:,} | {t:.2f} | {d['peak_rss_gb']:.2f} | "
              f"{d['peak_rss_gb'] / t:.2f} |")
        A("")
    A("## Cost table\n")
    A("| cell | model | posterior | alpha | prune_baseline | wall_clock_s | "
      "peak_rss_gb | gp_predict_s | acqf_s | gp_train_s | acquisition_s |")
    A("|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for r in results:
        if r.get("status") != "ok":
            A(f"| {r['cell']} | {r['model']} | {r['posterior']} | "
              f"{r['alpha']:g} | {r['prune_baseline']} | FAILED | FAILED | "
              f"FAILED | FAILED | FAILED | FAILED |")
            continue
        flag = (" **OVER BUDGET**"
                if r.get("exceeded_rss_budget", r.get("exceeded_30gb", False))
                else "")
        A(f"| {r['cell']} | {r['model']} | {r['posterior']} | {r['alpha']:g} | "
          f"{r['prune_baseline']} | {_fmt(r['wall_clock_s'])} | "
          f"{_fmt(r['peak_rss_gb'], 2)}{flag} | {_fmt(r['gp_predict_s'])} | "
          f"{_fmt(r['acqf_s'])} | {_fmt(r['gp_train_s'], 2)} | "
          f"{_fmt(r['acquisition_s'])} |")
    A("")
    A("`wall_clock_s` = GP train + acquisition (pool construction + "
      "`select_batch`), i.e. one full `loop.BOLoop.step` minus docking. "
      "`acqf_s` = `acquisition_s - gp_predict_s`. Measurement overhead "
      "(unique-row counting, cells 1-2) is timed separately and subtracted; "
      "raw values are in `probe_results.json`.")
    A("")

    failed = [r for r in results if r.get("status") != "ok"]
    over = [r for r in ok
            if r.get("exceeded_rss_budget", r.get("exceeded_30gb", False))]
    if failed or over:
        A("## Flags\n")
        for r in failed:
            A(f"- **Cell {r['cell']} FAILED** (rc={r.get('returncode')}). "
              f"Log: `{r.get('log')}`")
        for r in over:
            A(f"- **Cell {r['cell']} exceeded the {RSS_BUDGET_GB:g} GB peak-RSS "
              f"budget**: {r['peak_rss_gb']:.2f} GB.")
        A("")

    A("## Redundancy: rows re-predicted per chunk\n")
    A("`DockingPosteriorModel.posterior()` receives `X` of shape "
      "`(chunk, B+1, d)` and `acquisition.py:411` flattens it with "
      "`.reshape(-1, n_fp)`, so the B-molecule baseline is presented once per "
      "chunk element.\n")
    for n in (1, 2):
        r = by_cell.get(n)
        if r is None:
            A(f"- Cell {n}: not available.")
            continue
        red = r["redundancy"]
        A(f"**Cell {n} ({r['model']} / {r['posterior']})** -- "
          f"{red['n_scoring_chunks']} scoring chunks.\n")
        A("| chunk | t-batch | q | rows presented | rows reaching the GP | "
          "unique molecules | presented/unique |")
        A("|---:|---:|---:|---:|---:|---:|---:|")
        for c in red["per_chunk"]:
            b = c["batch"][0] if c["batch"] else 1
            ratio = (c["rows_presented"] / c["unique_rows"]
                     if c["unique_rows"] else None)
            A(f"| {c['chunk']} | {b} | {c['q']} | {c['rows_presented']} | "
              f"{c['rows_to_gp']} | {c['unique_rows']} | {_fmt(ratio, 2)} |")
        A("")
        A(f"Iteration total: **{red['rows_presented_to_posterior_total']:,} rows "
          f"presented**, **{red['rows_passed_to_gp_predict_total']:,} rows "
          f"actually passed to the GP predict call**, "
          f"**{red['unique_rows_total']:,} unique molecule-rows**, "
          f"ratio **{_fmt(red['ratio_presented_over_unique'], 2)}x**.\n")

    A("## Timing split: GP prediction vs qNEHVI\n")
    A("| cell | acquisition_s | gp_predict_s | posterior_assembly_s | "
      "acqf_init_s (box decomp) | acqf_forward_s | pool_prep_s | diversity_s | "
      "gp_predict % of acq |")
    A("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in ok:
        pct = 100.0 * r["gp_predict_s"] / r["acquisition_s"] if r["acquisition_s"] else 0
        A(f"| {r['cell']} | {_fmt(r['acquisition_s'])} | "
          f"{_fmt(r['gp_predict_s'])} | {_fmt(r['posterior_assembly_s'])} | "
          f"{_fmt(r['acqf_init_s'])} | {_fmt(r['acqf_forward_s'])} | "
          f"{_fmt(r['pool_prep_s'], 2)} | {_fmt(r['diversity_s'], 2)} | "
          f"{pct:.1f}% |")
    A("")
    A("`posterior_assembly_s` is the time inside "
      "`DockingPosteriorModel.posterior` outside the GP call (`diag_embed` for "
      "`diag`; dedup + block gather for `joint`). `acqf_init_s` is the "
      "`qLogNEHVI` constructor, which is where the initial box decomposition of "
      "the baseline front is built. `acqf_forward_s` is the remaining forward "
      "cost (MC sampling, composite objective, incremental hypervolume).")
    if ok and ok[0].get("n_boxes"):
        A("")
        A("Box counts (`acqf.cell_lower_bounds.shape[-2]`): "
          + ", ".join(f"cell {r['cell']}: {r['n_boxes']:,}" for r in ok
                      if r.get("n_boxes")))
    A("")

    A("## Batch diversity (batch_size=5, diversity_threshold=0.7)\n")
    A("| cell | model | posterior | n selected | mean pairwise Tanimoto | "
      "max pairwise Tanimoto | selected library indices |")
    A("|---:|---|---|---:|---:|---:|---|")
    for r in ok:
        d = r["batch_diversity"]
        A(f"| {r['cell']} | {r['model']} | {r['posterior']} | "
          f"{d['n_selected']} | {_fmt(d['mean_pairwise_tanimoto'], 4)} | "
          f"{_fmt(d['max_pairwise_tanimoto'], 4)} | "
          f"{', '.join(str(i) for i in d['selected_library_indices'])} |")
    A("")
    c5, c6 = by_cell.get(5), by_cell.get(6)
    if c5 and c6:
        A(f"`prune_baseline=True` (cell 6) does prune -- the qNEHVI baseline "
          f"goes from {c5['acqf_baseline_rows']} to "
          f"{c6['acqf_baseline_rows']} points and num_boxes from "
          f"{c5['n_boxes']:,} to {c6['n_boxes']:,}, and it changes which "
          f"molecules are picked -- but it does not move cost: "
          f"{c5['wall_clock_s']:.1f} s / {c5['peak_rss_gb']:.2f} GB vs "
          f"{c6['wall_clock_s']:.1f} s / {c6['peak_rss_gb']:.2f} GB.")
        A("")
    c1, c2 = by_cell.get(1), by_cell.get(2)
    if c1 and c2:
        d1, d2 = c1["batch_diversity"], c2["batch_diversity"]
        overlap = set(d1["selected_library_indices"]) & set(
            d2["selected_library_indices"])
        A(f"Cell 1 (diag) vs cell 2 (joint): mean pairwise Tanimoto "
          f"{d1['mean_pairwise_tanimoto']:.4f} -> "
          f"{d2['mean_pairwise_tanimoto']:.4f}; max "
          f"{d1['max_pairwise_tanimoto']:.4f} -> "
          f"{d2['max_pairwise_tanimoto']:.4f}. "
          f"{len(overlap)}/5 selected molecules in common.")
    A("")
    A("## Provenance\n")
    A("- Nothing under `campaign_results/`, `evaluation_bounds.json` or any "
      "existing module was modified. `alpha` / `prune_baseline` are injected by "
      "monkeypatching `acquisition.qLogNoisyExpectedHypervolumeImprovement` "
      "inside the probe process only.")
    A("- One OS process per cell; cells ran strictly sequentially.")
    A("- `torch.manual_seed(0)` is called immediately before every "
      "`select_batch` call (the Sobol sampler takes its seed from torch's "
      "global RNG).")
    A(f"- Full numbers, per-chunk rows and the pairwise similarity matrices are "
      f"in `probe_results.json`.")

    with open(RESULTS_MD, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"\nWrote {RESULTS_JSON}\nWrote {RESULTS_MD}")


def collect_from_disk(log_dir):
    """Rebuild the whole-sweep result set from the per-cell artifacts on disk.

    The sweep-file rule (CLAUDE.md): a file that describes the WHOLE sweep must
    be derived from what is on disk, never from the set of cells the current
    invocation happened to run -- otherwise a `--cells 3,4` resume silently
    republishes a six-cell report from a sample of two. Mirrors the pattern
    `matrix_report.discover` gets right by construction.

    A cell with no JSON on disk is reported as failed, not omitted, so a partial
    sweep cannot masquerade as a complete one.
    """
    results = []
    for spec in CELLS:
        p = os.path.join(log_dir, f"cell{spec['cell']}.json")
        log = os.path.join(log_dir, f"cell{spec['cell']}.log")
        if os.path.exists(p):
            with open(p) as fh:
                r = json.load(fh)
            r["log"] = log
        else:
            r = dict(spec)
            r.update({"status": "failed", "returncode": None, "log": log,
                      "error_tail": "no result JSON on disk for this cell"})
        results.append(r)
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state-dir", default=DEFAULT_STATE_DIR)
    ap.add_argument("--cell", type=int, default=None,
                    help="Child mode: run this one cell and exit.")
    ap.add_argument("--out", default=None, help="Child mode: JSON output path.")
    ap.add_argument("--cells", default=None,
                    help="Comma-separated subset of cells for the parent.")
    ap.add_argument("--prep-only", action="store_true")
    ap.add_argument("--force-prep", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--baseline-cap", type=int, default=None,
                    help="Replay only the first B finite evaluated rows (a "
                         "PREFIX of the arm's own trajectory) as GP training "
                         "set / qNEHVI baseline.")
    ap.add_argument("--rss-cap", type=float, default=None,
                    help="Hard-abort a child whose peak RSS exceeds this many "
                         "GB (exit code 17).")
    ap.add_argument("--no-instr", action="store_true",
                    help="Disable unique-row counting so the timing/RSS is the "
                         "bare acquisition call.")
    ap.add_argument("--ladder", default=None,
                    help="Comma-separated B values: measure peak RSS vs "
                         "baseline size and stop at the first over-budget rung.")
    args = ap.parse_args()

    state_dir = args.state_dir
    meta_path = os.path.join(state_dir, "meta.json")

    if args.cell is not None:
        spec = next(c for c in CELLS if c["cell"] == args.cell)
        out = args.out or os.path.join(state_dir, f"cell{args.cell}.json")
        run_cell(spec, state_dir, out, smoke=args.smoke,
                 baseline_cap=args.baseline_cap, rss_cap=args.rss_cap,
                 no_instr=args.no_instr)
        return

    if args.force_prep and os.path.exists(state_dir):
        shutil.rmtree(state_dir)
    if not os.path.exists(meta_path):
        prep(state_dir)
    with open(meta_path) as fh:
        meta = json.load(fh)
    if args.prep_only:
        print(json.dumps({k: v for k, v in meta.items()
                          if k != "evaluated_indices"}, indent=2))
        return

    log_dir = os.path.join(state_dir, "logs")
    if args.report_only:
        write_reports(collect_from_disk(log_dir), meta, smoke=args.smoke,
                      log_dir=log_dir)
        return

    if args.ladder:
        b_values = [int(x) for x in args.ladder.split(",")]
        rungs = run_ladder(state_dir, b_values,
                           args.rss_cap or RSS_BUDGET_GB - 0.5,
                           log_dir=log_dir, no_instr=True)
        print("\nLADDER:")
        for r in rungs:
            print(f"  B={r['B']:>4}  status={r['status']:<18} "
                  f"peak={r['peak_rss_gb']}  boxes={r['n_boxes']}  "
                  f"wall={r['wall_clock_s']}")
        return

    cells = CELLS
    if args.cells:
        want = {int(x) for x in args.cells.split(",")}
        cells = [c for c in CELLS if c["cell"] in want]
    run_all(state_dir, cells, smoke=args.smoke, log_dir=log_dir,
            baseline_cap=args.baseline_cap, rss_cap=args.rss_cap,
            no_instr=args.no_instr)
    # Sweep-file rule: rebuild the sweep-level report from every cell artifact
    # on disk, not from the cells this invocation ran.
    write_reports(collect_from_disk(log_dir), meta, smoke=args.smoke,
                  log_dir=log_dir)


if __name__ == "__main__":
    main()
