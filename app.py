"""
app.py — Streamlit CONTROL PANEL for the MOGP-NTD pipeline.
===========================================================

A THIN WRAPPER. This file contains NO science. It does not compute a
hypervolume, seed an RNG, build a library, or run a BO loop. It builds a command
line, hands it to the EXISTING scripts as a detached subprocess, tails their log,
and renders the CSVs they produce (reusing ``dashboard_compare``'s plotting).

If this panel and the command line ever disagree, that is a bug in this file —
the command line is the source of truth. Every screen shows the exact shell
command it will run, so any run here is reproducible by hand.

Design notes:

* **Detached launches.** Runs take hours. Each launch is a new process SESSION
  (``start_new_session=True``) writing to a timestamped log under ``runs/``, so
  it survives a page refresh, a browser close, and a Streamlit restart. The page
  tails the log file; it never holds the process.
* **Nothing is swallowed.** stdout and stderr are merged into the log verbatim.
  A nonzero exit shows the FULL log (tracebacks included), not a friendly
  message.
* **run_all.py is interactive**, not argparse — it asks 10 questions via
  ``input()``. The panel answers them by piping stdin, and shows the exact
  ``printf ... | python run_all.py`` equivalent.

Run:
    streamlit run app.py
"""

import glob
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime

import pandas as pd
import streamlit as st

# Reused rendering from the existing dashboard — NOT reimplemented.
import dashboard_compare as dc
from data import HEAVY_ATOM_FLOOR

REPO = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(REPO, "runs")
BOUNDS_PATH = os.path.join(REPO, "evaluation_bounds.json")
BUILD_MARKER = os.path.join(REPO, "data", "library", ".run_all_build_size")
SMILES_CSV = os.path.join(REPO, "data", "library", "smiles.csv")

PYTHON = sys.executable


# --------------------------------------------------------------------------- #
# Run registry — a directory per launch, so state survives a page refresh.
# --------------------------------------------------------------------------- #
def _run_dir(run_id):
    return os.path.join(RUNS_DIR, run_id)


def launch(script, argv, stdin_lines=None, output_dirs=()):
    """Launch ``script`` detached, logging to ``runs/<id>/run.log``.

    The process is started in its own session so it outlives this Streamlit
    session. A wrapper shell records the exit code to ``exit_code`` on the way
    out, which is how a later page load can tell "finished nonzero" from
    "still running" without holding a handle to the process.
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{stamp}_{script.replace('.py', '')}"
    rd = _run_dir(run_id)
    os.makedirs(rd, exist_ok=True)

    log_path = os.path.join(rd, "run.log")
    exit_path = os.path.join(rd, "exit_code")

    shell_cmd = build_shell_command(script, argv, stdin_lines)
    # -u / PYTHONUNBUFFERED so the log streams instead of buffering for hours.
    wrapped = (f"{shell_cmd} > {shlex.quote(log_path)} 2>&1; "
               f"echo $? > {shlex.quote(exit_path)}")

    env = {**os.environ, "PYTHONUNBUFFERED": "1", "KMP_DUPLICATE_LIB_OK": "TRUE"}
    proc = subprocess.Popen(
        ["bash", "-c", wrapped],
        cwd=REPO, env=env, start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    meta = {
        "run_id": run_id,
        "script": script,
        "command": shell_cmd,
        "pid": proc.pid,
        "started": time.time(),
        "output_dirs": list(output_dirs),
        # Snapshot of the bounds these results will be computed against, so the
        # results viewer can later tell whether they are still comparable.
        "bounds_sha256": bounds_hash(),
    }
    with open(os.path.join(rd, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return run_id


def build_shell_command(script, argv, stdin_lines=None):
    """The exact shell command for a launch — also what the UI displays."""
    cmd = " ".join(shlex.quote(p) for p in [PYTHON, "-u", script, *argv])
    if stdin_lines:
        answers = "".join(f"{line}\\n" for line in stdin_lines)
        cmd = f"printf {shlex.quote(answers)} | {cmd}"
    return cmd


def load_runs():
    """All launches, newest first."""
    runs = []
    for meta_path in glob.glob(os.path.join(RUNS_DIR, "*", "meta.json")):
        try:
            with open(meta_path) as fh:
                runs.append(json.load(fh))
        except (OSError, ValueError):
            continue
    return sorted(runs, key=lambda m: m["started"], reverse=True)


def run_status(meta):
    """``(state, exit_code)`` where state is running / finished / vanished."""
    exit_path = os.path.join(_run_dir(meta["run_id"]), "exit_code")
    if os.path.exists(exit_path):
        try:
            with open(exit_path) as fh:
                return "finished", int(fh.read().strip())
        except (OSError, ValueError):
            return "finished", None
    if pid_alive(meta["pid"]):
        return "running", None
    # No exit code and no process: killed hard (OOM, reboot, SIGKILL).
    return "vanished", None


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except (OSError, ProcessLookupError):
        return False
    return True


def stop_run(meta):
    """SIGTERM the whole process group (the wrapper shell AND the python child)."""
    try:
        os.killpg(os.getpgid(meta["pid"]), signal.SIGTERM)
        return True, None
    except (OSError, ProcessLookupError) as exc:
        return False, str(exc)


def read_log(run_id, max_bytes=200_000):
    path = os.path.join(_run_dir(run_id), "run.log")
    if not os.path.exists(path):
        return ""
    size = os.path.getsize(path)
    with open(path, errors="replace") as fh:
        if size > max_bytes:
            fh.seek(size - max_bytes)
            return f"... [truncated {size - max_bytes} bytes; full log: {path}]\n" + fh.read()
        return fh.read()


def fmt_elapsed(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.2f}h"


# --------------------------------------------------------------------------- #
# Guardrails — library, floor, bounds
# --------------------------------------------------------------------------- #
def bounds_hash():
    if not os.path.exists(BOUNDS_PATH):
        return None
    with open(BOUNDS_PATH, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


@st.cache_data(show_spinner=False)
def library_facts(_bust):
    """Library size on disk and after the floor. Uses data.load_library — the
    SAME load path the pipeline uses, so the numbers cannot drift from a run."""
    facts = {"floor": HEAVY_ATOM_FLOOR, "marker": None,
             "n_disk": None, "n_usable": None, "error": None}
    try:
        with open(BUILD_MARKER) as fh:
            facts["marker"] = int(fh.read().strip())
    except (OSError, ValueError):
        pass
    try:
        facts["n_disk"] = len(pd.read_csv(SMILES_CSV))
        from data import load_library
        facts["n_usable"] = len(load_library()["smiles"])
    except Exception as exc:
        facts["error"] = str(exc)
    return facts


def current_bounds():
    if not os.path.exists(BOUNDS_PATH):
        return None
    with open(BOUNDS_PATH) as fh:
        return json.load(fh)


def bounds_warning_for(results_dir):
    """Are ``results_dir``'s hypervolumes computed against the CURRENT bounds?

    Two signals, best first:
      1. If this panel launched the run, meta.json holds the bounds hash that was
         live at launch — an exact answer.
      2. Otherwise fall back to mtimes: bounds newer than history.csv means the
         bounds were regenerated AFTER the results, so they no longer match.
         This is a heuristic and is labelled as one.
    """
    history = os.path.join(REPO, results_dir, dc.HISTORY_FILE)
    if not os.path.exists(history):
        return None
    now = bounds_hash()

    for meta in load_runs():                      # newest first
        if results_dir in meta.get("output_dirs", []):
            if meta.get("bounds_sha256") and meta["bounds_sha256"] != now:
                return ("stale", "This run was launched under a DIFFERENT "
                        "evaluation_bounds.json than the one on disk now. Its "
                        "hypervolumes are on a different scale — not comparable.")
            return None

    if os.path.exists(BOUNDS_PATH) and os.path.getmtime(BOUNDS_PATH) > os.path.getmtime(history):
        return ("maybe-stale",
                "evaluation_bounds.json was modified AFTER these results were "
                "written (mtime heuristic — this run predates the panel). The "
                "bounds define the hypervolume scale, so these numbers are "
                "probably NOT comparable to anything computed now. Re-run.")
    return None


# --------------------------------------------------------------------------- #
# Forms — one per script. Every widget maps to a REAL flag (from each argparse).
# --------------------------------------------------------------------------- #
def form_run_all(facts):
    st.info(
        "**run_all.py has no CLI flags** — it asks 10 questions interactively via "
        "`input()`. The panel answers them by piping stdin, which is why the "
        "command below is a `printf ... | python run_all.py`. It is exactly "
        "equivalent to typing the answers by hand.",
        icon="ℹ️",
    )
    st.warning(
        "run_all.py **deletes** `results/` and the three `baseline_*_results/` "
        "dirs before it runs.", icon="⚠️",
    )
    c1, c2, c3 = st.columns(3)
    # Default the pull size to the build marker, which is what run_all itself
    # defaults to (run_all.default_lib_size) — anything else rebuilds the library.
    lib_size = c1.number_input("ChEMBL pull size", value=facts["marker"] or 1000,
                               step=1, min_value=1)
    n_init = c1.number_input("Initial molecules", value=5, step=1, min_value=1)
    batch_size = c2.number_input("Batch size", value=5, step=1, min_value=1)
    n_iterations = c2.number_input("Iterations", value=2, step=1, min_value=1)
    mogp_iters = c3.number_input("MOGP training iters", value=50, step=10, min_value=1)
    model = c3.selectbox("MOGP model", ["coregionalized", "independent"], index=0)
    rank = c1.number_input("Coregionalization rank", value=1, step=1, min_value=1)
    seed = c2.number_input("Seed", value=42, step=1)
    fair_timing = c3.checkbox("Clear docking cache before each method "
                              "(fair timing, much slower)", value=False)

    library_rebuild_warning(lib_size, facts)

    # Order matters: it is the order run_all.py asks its questions.
    stdin_lines = [str(lib_size), str(n_init), str(batch_size), str(n_iterations),
                   str(mogp_iters), model, str(rank), str(seed),
                   "y" if fair_timing else "n", "y"]
    outputs = ["results", "baseline_random_results",
               "baseline_single_obj_results", "baseline_greedy_results"]
    return "run_all.py", [], stdin_lines, outputs


def form_benchmark_seeds(facts):
    c1, c2, c3 = st.columns(3)
    seeds = c1.text_input("--seeds (space-separated)", "0 1 2")
    lib_size = c1.number_input("--lib-size", value=facts["marker"] or 1000,
                               step=1, min_value=1)
    n_init = c2.number_input("--n-init", value=10, step=1, min_value=1)
    batch_size = c2.number_input("--batch-size", value=10, step=1, min_value=1)
    n_iterations = c3.number_input("--n-iterations", value=10, step=1, min_value=1)
    mogp_iters = c3.number_input("--mogp-iters", value=200, step=10, min_value=1)
    output_dir = c1.text_input("--output-dir", "benchmark_seeds_results")
    band = c2.selectbox("--band", ["std", "sem", "ci95"], index=0,
                        help="sem/ci95 are more honest for few seeds — they "
                             "shrink as seeds are added; std does not.")

    st.markdown("**Densification** (MOGP loop only; baselines unaffected)")
    d1, d2, d3 = st.columns(3)
    densify = d1.checkbox("--densify", value=False)
    per_parent = d2.number_input("--densify-per-parent", value=20, step=1,
                                 min_value=1, disabled=not densify)
    max_pool = d3.text_input("--densify-max-pool (blank = no cap)", "",
                             disabled=not densify)

    st.markdown("**Docking cache**")
    k1, k2 = st.columns(2)
    no_cache = k1.checkbox("--no-cache (disable cache for this run)", value=False)
    clear_cache = k2.checkbox("--clear-cache (WIPE the cache first)", value=False)
    if clear_cache:
        st.error(
            "`--clear-cache` **deletes every cached docking score**. Those are the "
            "expensive results (hours of Vina). Only use this to retry failures.",
            icon="🚨",
        )

    library_rebuild_warning(lib_size, facts)

    argv = ["--seeds", *seeds.split(), "--lib-size", str(lib_size),
            "--n-init", str(n_init), "--batch-size", str(batch_size),
            "--n-iterations", str(n_iterations), "--mogp-iters", str(mogp_iters),
            "--output-dir", output_dir, "--band", band]
    if densify:
        argv += ["--densify", "--densify-per-parent", str(per_parent)]
        if max_pool.strip():
            argv += ["--densify-max-pool", max_pool.strip()]
    if no_cache:
        argv.append("--no-cache")
    if clear_cache:
        argv.append("--clear-cache")
    return "run_benchmark_seeds.py", argv, None, [output_dir]


def form_ablation(facts):
    c1, c2, c3 = st.columns(3)
    seeds = c1.text_input("--seeds (comma-separated)", "0,1,2")
    library_dir = c1.text_input("--library-dir", "data/library")
    n_init = c2.number_input("--n-init ", value=10, step=1, min_value=1)
    batch_size = c2.number_input("--batch-size ", value=10, step=1, min_value=1)
    n_iterations = c3.number_input("--n-iterations ", value=8, step=1, min_value=1)
    mogp_iters = c3.number_input("--mogp-iters ", value=200, step=10, min_value=1)
    rank = c1.number_input("--rank", value=1, step=1, min_value=1)
    max_library = c2.text_input("--max-library (blank = whole library)", "")
    models = c3.multiselect("--models", ["coregionalized", "independent"],
                            default=["coregionalized", "independent"])
    save = st.checkbox("--save (write per-run CSVs)", value=False)

    argv = ["--seeds", seeds, "--library-dir", library_dir,
            "--n-init", str(n_init), "--batch-size", str(batch_size),
            "--n-iterations", str(n_iterations), "--mogp-iters", str(mogp_iters),
            "--rank", str(rank)]
    if max_library.strip():
        argv += ["--max-library", max_library.strip()]
    if models:
        argv += ["--models", ",".join(models)]
    if save:
        argv.append("--save")
    return "run_ablation.py", argv, None, []


def form_validate_docking(facts):
    c1, c2, c3 = st.columns(3)
    library_dir = c1.text_input("--library-dir ", "data/library")
    n_sample = c2.number_input("--n-sample", value=150, step=10, min_value=1)
    top_k = c3.number_input("--top-k", value=15, step=1, min_value=1)
    seed = c1.number_input("--seed", value=42, step=1)
    output = c2.text_input("--output", "validate_docking_scatter.png")
    argv = ["--library-dir", library_dir, "--n-sample", str(n_sample),
            "--top-k", str(top_k), "--seed", str(seed), "--output", output]
    return "validate_docking.py", argv, None, []


def form_validate_known_actives(facts):
    c1, c2, c3 = st.columns(3)
    threshold = c1.number_input("--threshold", value=0.4, step=0.05,
                                min_value=0.0, max_value=1.0)
    independent_dir = c2.text_input("--independent-dir", "results")
    coregionalized_dir = c3.text_input("--coregionalized-dir",
                                       "results_coregionalized")
    skip_docking = st.checkbox("--skip-docking", value=False)
    argv = ["--threshold", str(threshold),
            "--independent-dir", independent_dir,
            "--coregionalized-dir", coregionalized_dir]
    if skip_docking:
        argv.append("--skip-docking")
    return "validate_known_actives.py", argv, None, []


def library_rebuild_warning(lib_size, facts):
    """The footgun: a lib-size != the build marker triggers a full REBUILD.

    ``run_benchmark_seeds`` and ``run_all`` both call ``ensure_library(lib_size)``,
    which rebuilds from ChEMBL whenever the marker disagrees. The argparse default
    (1000) would silently replace the cached library with a much smaller one.
    """
    marker = facts.get("marker")
    if marker is None or int(lib_size) == int(marker):
        return
    st.error(
        f"**This will REBUILD the molecule library.** The cached library was built "
        f"from a pull of **{marker}** ({facts.get('n_disk')} molecules on disk), but "
        f"you asked for **{lib_size}**. `ensure_library` rebuilds whenever these "
        f"disagree — the current library would be replaced (a pull of {lib_size} "
        f"yields roughly {int(lib_size * 0.6)} molecules) and the ADMET bounds "
        f"would shift again. Set it to {marker} to reuse what is on disk.",
        icon="🚨",
    )


FORMS = {
    "run_all.py — single seed, all 4 methods": form_run_all,
    "run_benchmark_seeds.py — multi-seed + significance": form_benchmark_seeds,
    "run_ablation.py — coregionalized vs independent": form_ablation,
    "validate_docking.py — docking objective validity": form_validate_docking,
    "validate_known_actives.py — known-drug rediscovery": form_validate_known_actives,
}


# --------------------------------------------------------------------------- #
# Pages
# --------------------------------------------------------------------------- #
def page_launch(facts):
    st.header("Configure & launch")
    choice = st.selectbox("Script", list(FORMS))
    script, argv, stdin_lines, outputs = FORMS[choice](facts)

    st.subheader("Command")
    st.caption("The exact command this panel will run. Copy it to reproduce this "
               "run from a terminal — the panel adds nothing.")
    st.code(build_shell_command(script, argv, stdin_lines), language="bash")

    if st.button("Launch", type="primary"):
        run_id = launch(script, argv, stdin_lines, outputs)
        st.session_state["watching"] = run_id
        st.success(f"Launched {run_id} (detached — safe to refresh or close the tab).")
        st.rerun()


def page_monitor():
    st.header("Runs")
    runs = load_runs()
    if not runs:
        st.info("No runs launched yet.")
        return

    labels = []
    for meta in runs:
        state, code = run_status(meta)
        badge = {"running": "🟢", "finished": "✅" if code == 0 else "❌",
                 "vanished": "⚠️"}[state]
        labels.append(f"{badge} {meta['run_id']}")

    default = 0
    if st.session_state.get("watching") in [m["run_id"] for m in runs]:
        default = [m["run_id"] for m in runs].index(st.session_state["watching"])
    idx = st.selectbox("Run", range(len(runs)), format_func=lambda i: labels[i],
                       index=default)
    meta = runs[idx]
    render_run(meta)


@st.fragment(run_every=2)
def render_run(meta):
    """Tail the log. A fragment so it refreshes WITHOUT rerunning the whole page."""
    state, code = run_status(meta)
    started = meta["started"]
    log_path = os.path.join(_run_dir(meta["run_id"]), "run.log")

    if state == "running":
        elapsed = time.time() - started
    else:
        end = os.path.getmtime(log_path) if os.path.exists(log_path) else started
        elapsed = max(end - started, 0)

    c1, c2, c3 = st.columns(3)
    c1.metric("Status", {"running": "RUNNING", "finished": f"EXIT {code}",
                         "vanished": "VANISHED"}[state])
    c2.metric("Elapsed", fmt_elapsed(elapsed))
    c3.metric("PID", meta["pid"])

    st.code(meta["command"], language="bash")
    st.caption(f"Log: `{log_path}`")

    if state == "running":
        if st.button("Stop this run", type="secondary", key=f"stop_{meta['run_id']}"):
            ok, err = stop_run(meta)
            st.warning("Sent SIGTERM." if ok else f"Could not stop it: {err}")

    if state == "finished" and code not in (0, None):
        st.error(f"**The run exited {code}.** Full log below — the traceback is in "
                 f"it verbatim, nothing is filtered.", icon="🚨")
    elif state == "vanished":
        st.warning("The process is gone but wrote no exit code — it was killed "
                   "hard (SIGKILL, OOM, or a reboot). The log is whatever it "
                   "managed to write.", icon="⚠️")

    log = read_log(meta["run_id"])
    st.text_area("Output (live)", log or "(no output yet)", height=460,
                 key=f"log_{meta['run_id']}_{len(log)}")


def page_results(facts):
    st.header("Results")

    st.subheader("Single-seed comparison (results/ + baseline_*_results/)")
    methods = dc.available_methods()          # dashboard_compare, not reimplemented
    if not methods:
        st.info("No single-seed results on disk yet.")
    else:
        for _label, results_dir, _color, _h in methods:
            warn = bounds_warning_for(results_dir)
            if warn:
                st.warning(f"**{results_dir}** — {warn[1]}", icon="⚠️")

        rows = []
        for label, results_dir, _color, history in methods:
            row = {"Method": label, "Final hypervolume": dc.final_hv(history),
                   "Molecules evaluated": int(history["n_evaluated"].iloc[-1]),
                   "Pareto size": int(history["pareto_size"].iloc[-1])}
            # Pareto size-drift monitor — only present in newer runs.
            for col in ("pareto_median_heavy", "pareto_min_heavy"):
                if col in history.columns:
                    row[col] = history[col].iloc[-1]
            rows.append(row)
        table = pd.DataFrame(rows)
        st.dataframe(table.style.apply(dc._highlight_best,
                                       subset=["Final hypervolume"]),
                     width="stretch")

        if "pareto_median_heavy" not in table.columns:
            st.caption("No Pareto drift-monitor columns in these histories — they "
                       "predate the heavy-atom monitor. Re-run to get them.")
        else:
            st.caption(f"Drift monitor: with the LE objective rewarding small "
                       f"molecules, a Pareto median approaching the floor "
                       f"({HEAVY_ATOM_FLOOR}) means the front is drifting to "
                       f"fragments.")

        st.pyplot(dc._history_line_plot(methods, "hypervolume",
                                        "Hypervolume vs molecules evaluated",
                                        "Hypervolume"))

    st.divider()
    st.subheader("Multi-seed benchmark")
    bench_dir = st.text_input("Benchmark output dir", "benchmark_seeds_results")
    bdir = os.path.join(REPO, bench_dir)

    png = os.path.join(bdir, "benchmark_seeds.png")
    if os.path.exists(png):
        st.image(png, caption=png)

    sig = os.path.join(bdir, "benchmark_seeds_significance.csv")
    if os.path.exists(sig):
        st.markdown("**Paired significance (Wilcoxon signed-rank), MOGP vs baseline**")
        st.dataframe(pd.read_csv(sig), width="stretch")
    agg = os.path.join(bdir, "benchmark_seeds_aggregate.csv")
    if os.path.exists(agg):
        st.markdown("**Aggregate across seeds**")
        st.dataframe(pd.read_csv(agg), width="stretch")
    if not any(os.path.exists(p) for p in (png, sig, agg)):
        st.info(f"No benchmark artifacts in `{bench_dir}/` yet.")

    st.divider()
    st.subheader("Raw CSVs")
    all_dirs = [d for _l, d, _c in dc.METHODS if os.path.isdir(os.path.join(REPO, d))]
    if all_dirs:
        rdir = st.selectbox("Results dir", all_dirs)
        fname = st.selectbox("File", [dc.HISTORY_FILE, "evaluated.csv", dc.PARETO_FILE])
        df = dc.load_csv(rdir, fname)
        if df is None:
            st.info(f"`{fname}` not in `{rdir}/`.")
        else:
            st.dataframe(df, width="stretch")
            st.download_button(f"Download {fname}", df.to_csv(index=False),
                               file_name=f"{rdir}_{fname}", mime="text/csv")


def sidebar_guardrails(facts):
    st.sidebar.header("Guardrails")
    if facts["error"]:
        st.sidebar.error(f"Could not load the library: {facts['error']}")
    else:
        st.sidebar.metric("Library on disk", facts["n_disk"])
        st.sidebar.metric(f"Usable (heavy-atom floor ≥ {facts['floor']})",
                          facts["n_usable"])
        dropped = (facts["n_disk"] or 0) - (facts["n_usable"] or 0)
        st.sidebar.caption(f"{dropped} molecule(s) below the floor are dropped at "
                           f"load. Floor = `data.HEAVY_ATOM_FLOOR` = {facts['floor']}.")
    st.sidebar.caption(f"Build marker (ChEMBL pull size): **{facts['marker']}**")

    bounds = current_bounds()
    if bounds:
        with st.sidebar.expander("evaluation_bounds.json"):
            st.caption(f"sha256 `{(bounds_hash() or '')[:12]}…`")
            st.dataframe(pd.DataFrame(
                [{"objective": k, "min": v[0], "max": v[1]}
                 for k, v in bounds["bounds"].items()]),
                hide_index=True)
            st.caption("Hypervolumes are normalized against these bounds. Results "
                       "produced under different bounds are NOT comparable.")


def main():
    st.set_page_config(page_title="MOGP-NTD Control Panel", layout="wide")
    st.title("MOGP-NTD — control panel")
    st.caption("A thin wrapper: it runs the existing scripts and shows their "
               "output. Every screen shows the equivalent command line.")

    os.makedirs(RUNS_DIR, exist_ok=True)
    facts = library_facts(bounds_hash())
    sidebar_guardrails(facts)

    launch_tab, monitor_tab, results_tab = st.tabs(
        ["Launch", "Monitor", "Results"])
    with launch_tab:
        page_launch(facts)
    with monitor_tab:
        page_monitor()
    with results_tab:
        page_results(facts)


if __name__ == "__main__":
    main()
