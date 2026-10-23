#!/usr/bin/env python
"""Run the GP-MOBO head-to-head grid, one subprocess per cell.

Refuses to launch unless all of these hold:

* ``bench_yong/PREREGISTRATION.md`` is committed and unmodified, and nothing
  under ``bench_yong/`` is uncommitted (so every result ties to a commit);
* the GP-MOBO checkout is at the pinned commit;
* the seed lists are EXACTLY the pre-registered ones (passed as explicit
  lists -- a bare count is the trap that once produced a sweep of identical
  runs), and the batch size is 1;
* ``pytest -q bench_yong/test_bench_yong.py`` exits 0 (aborting gates G1, G2,
  G4, G6, G7-STOP, G8).  Gate G3's record picks her prediction path.

Memory: macOS will not enforce an rlimit (``ulimit -v`` is rejected and
``setrlimit`` cannot lower the hard limit), so a watchdog samples every child's
resident set with ``ps`` every ``--poll-s`` seconds, records the peak, and on a
breach of ``--mem-cap-gb`` (GB = 1e9 bytes) sends SIGTERM, then SIGKILL after
30 s.  That fails the one cell, not the machine.  ``--per-run-timeout-h`` is
enforced the same way.  Any ``vina`` process stops the whole sweep.

Result code (provenance.py: run_one.py, yong_common.py and every repo module a
cell imports) is fingerprinted into the manifest.  A ``--resume`` against
different result code is refused, and the fingerprint is re-checked before
every launch, so a cell can never start on code the sweep did not start with.
One sweep per output directory: ``sweep.lock`` holds the live sweep's pid, and a
cell whose previous child is still alive is never moved or relaunched.

Tier 4 (HER-PUBLISHED, Amendment 2) runs as its own self-contained sweep in its
own output directory and is refused in the same sweep as Tiers 1-3.  Its cells
are dispatched to ``run_one_published.py``, which drives her ``guacamol-test``
drivers; the pinned ``external/GP-MOBO`` tree is only ever read.

Sweep-level files follow the repo's sweep-file rule.  ``manifest.json`` holds
the PLAN, written once; a later ``--resume`` invocation must match it exactly
and is appended to ``invocations``.  ``sweep_status.json`` is rebuilt from the
cell directories ON DISK at every poll, never from what this invocation ran.
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BENCH_DIR)
sys.path.insert(0, BENCH_DIR)
import her_published as hp  # noqa: E402
import provenance  # noqa: E402
import yong_common as yc  # noqa: E402

RUN_ONE = os.path.join(BENCH_DIR, "run_one.py")
RUN_ONE_PUBLISHED = os.path.join(BENCH_DIR, "run_one_published.py")
TEST_FILE = os.path.join(BENCH_DIR, "test_bench_yong.py")
PREREG = os.path.join(BENCH_DIR, "PREREGISTRATION.md")
KILL_GRACE_S = 30
GB = 1e9
# --test-mode accepts pool-limited cells as complete; a real sweep never does.
ACCEPT_POOL_LIMIT = False

ARM_ORDER = {"HER-PUBLISHED": 0, "HER@FP2": 1, "HER-PUB": 2,
             "MOGP-ICM@FP2": 3, "MOGP-IND@FP2": 4, "RANDOM": 5}
TASK_ORDER = {"fexofenadine": 0, "amlodipine": 1, "perindopril": 2}


def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------

def build_plan(args):
    tiers = [
        (1, args.tier1_tasks, args.tier1_arms, args.tier1_seeds, args.n_init),
        (2, args.tier2_tasks, args.tier2_arms, args.tier2_seeds, args.n_init),
        (3, args.tier3_tasks, args.tier3_arms, args.tier3_seeds, args.tier3_n_init),
        (4, args.tier4_tasks, args.tier4_arms, args.tier4_seeds, args.tier4_n_init),
    ]
    plan, seen = [], set()
    for tier, tasks, arms, seeds, n_init in tiers:
        if not tasks:
            continue
        cells = []
        for task in tasks:
            if task not in yc.TASKS:
                raise SystemExit(f"unknown task {task!r}")
            for arm in arms:
                arm = yc.canonical_arm(arm)
                for seed in seeds:
                    name = yc.cell_name(task, arm, seed, n_init)
                    if name in seen:
                        raise SystemExit(f"cell planned twice: {name}")
                    seen.add(name)
                    cells.append({"name": name, "tier": tier, "task": task,
                                  "arm": arm, "seed": int(seed),
                                  "n_init": int(n_init)})
        # Tiers run in order; inside a tier the long cells (her arms, then the
        # three-objective task) go first so they are not the tail.
        cells.sort(key=lambda c: (ARM_ORDER[c["arm"]], TASK_ORDER[c["task"]],
                                  c["seed"]))
        plan.extend(cells)
    return plan


def check_preregistered_tier4(args):
    """Amendment 2 section 5: 3 tasks x {HER-PUBLISHED, MOGP-ICM@FP2, RANDOM} x
    seeds 0-9 at n_init 10.  Amendment 2 section 3 forbids any other grid."""
    problems = []
    if list(args.tier4_seeds) != yc.PREREG_TIER4_SEEDS:
        problems.append(f"tier4 seeds {args.tier4_seeds} != pre-registered "
                        f"{yc.PREREG_TIER4_SEEDS}")
    if sorted(args.tier4_tasks) != sorted(yc.TASKS):
        problems.append(f"tier4 tasks {args.tier4_tasks} != {sorted(yc.TASKS)}")
    if sorted(yc.canonical_arm(a) for a in args.tier4_arms) != sorted(
            yc.PREREG_TIER4_ARMS):
        problems.append(f"tier4 arms {args.tier4_arms} are not the "
                        f"pre-registered {yc.PREREG_TIER4_ARMS}")
    if args.tier4_n_init != hp.RECIPE["n_init"]:
        problems.append(f"tier4 n_init {args.tier4_n_init} != "
                        f"{hp.RECIPE['n_init']} (her published protocol)")
    if args.n_iterations != 20:
        problems.append("tier4 keeps 20 BO iterations")
    ok, _prov, bad = hp.verify_provenance()
    if not ok:
        problems += [f"tier4 provenance: {b}" for b in bad]
    return problems


def check_preregistered(args):
    """Tier 1-3 (PREREGISTRATION.md section 4) or Tier 4 alone (Amendment 2)."""
    tier123 = bool(args.tier1_tasks or args.tier2_tasks or args.tier3_tasks)
    tier4 = bool(args.tier4_tasks or args.tier4_arms)
    if tier4 and tier123:
        return ["Tier 4 must run as its own sweep, in its own output directory "
                "(Amendment 2 section 5): it is self-contained, so no Tier 4 "
                "statistic is ever assembled from another sweep's files"]
    if tier4:
        return check_preregistered_tier4(args)
    problems = []
    for label, got, want in (("tier1", args.tier1_seeds, yc.PREREG_TIER1_SEEDS),
                             ("tier2", args.tier2_seeds, yc.PREREG_TIER2_SEEDS),
                             ("tier3", args.tier3_seeds, yc.PREREG_TIER3_SEEDS)):
        if list(got) != want:
            problems.append(f"{label} seeds {got} != pre-registered {want}")
    if sorted(args.tier1_tasks) != sorted(yc.TASKS):
        problems.append(f"tier1 tasks {args.tier1_tasks} != {sorted(yc.TASKS)}")
    t1 = sorted(yc.canonical_arm(a) for a in args.tier1_arms)
    if t1 != sorted(["HER@FP2", "MOGP-ICM@FP2", "MOGP-IND@FP2", "RANDOM"]):
        problems.append(f"tier1 arms {args.tier1_arms} are not the pre-registered four")
    if sorted(args.tier2_tasks) != sorted(yc.TASKS) or \
            [yc.canonical_arm(a) for a in args.tier2_arms] != ["HER-PUB"]:
        problems.append("tier2 must be all three tasks x HER-PUB (it is not droppable)")
    if args.tier3_tasks or args.tier3_arms:       # tier 3 may be dropped entirely
        if args.tier3_tasks != ["fexofenadine"] or \
                sorted(yc.canonical_arm(a) for a in args.tier3_arms) != \
                sorted(["HER@FP2", "MOGP-ICM@FP2"]):
            problems.append("tier3 must be fexofenadine x {HER@FP2, MOGP-ICM@FP2}")
    if args.n_init != 20 or args.tier3_n_init != 10 or args.n_iterations != 20:
        problems.append("budget differs from the pre-registered 20 + 20 (tier 3: 10)")
    return problems


def git_preflight():
    problems = []
    tracked = subprocess.run(["git", "-C", REPO, "ls-files", "--error-unmatch",
                              PREREG], capture_output=True)
    if tracked.returncode != 0:
        problems.append("PREREGISTRATION.md is not committed")
    if yc.git_dirty(rel="bench_yong"):
        problems.append("bench_yong/ has uncommitted changes")
    prereg_commit = subprocess.run(
        ["git", "-C", REPO, "log", "--diff-filter=A", "--format=%H", "--",
         "bench_yong/PREREGISTRATION.md"], capture_output=True, text=True
    ).stdout.strip().splitlines()
    if yc.git_head(yc.GP_DIR) != yc.GP_MOBO_COMMIT:
        problems.append("GP-MOBO is not at the pinned commit")
    if yc.git_dirty(yc.GP_DIR):
        problems.append("GP-MOBO working tree is dirty")
    return problems, (prereg_commit[-1] if prereg_commit else None)


def run_gates(out_dir):
    gate_dir = os.path.join(out_dir, "gates")
    os.makedirs(gate_dir, exist_ok=True)
    env = dict(os.environ, BENCH_YONG_GATE_DIR=gate_dir, PYTHONUNBUFFERED="1")
    log_path = os.path.join(out_dir, "gates_pytest.log")
    log(f"running gates: pytest -q {os.path.relpath(TEST_FILE, REPO)} "
        f"(log {log_path})")
    with open(log_path, "w") as fh:
        rc = subprocess.run([sys.executable, "-m", "pytest", "-q", TEST_FILE,
                             "-p", "no:cacheprovider", "-W", "ignore"],
                            cwd=REPO, env=env, stdout=fh,
                            stderr=subprocess.STDOUT).returncode
    records = {}
    for fname in sorted(os.listdir(gate_dir)):
        if fname.endswith(".json"):
            with open(os.path.join(gate_dir, fname)) as fh:
                r = json.load(fh)
            records[fname[:-5]] = r.get("pass")
    return rc, records, gate_dir


# --------------------------------------------------------------------------
# Disk state
# --------------------------------------------------------------------------

def cell_dir(out_dir, name):
    return os.path.join(out_dir, "cells", name)


def is_complete(out_dir, name):
    path = os.path.join(cell_dir(out_dir, name), "run.json")
    try:
        with open(path) as fh:
            r = json.load(fh)
    except (OSError, ValueError):
        return False
    return (r.get("status") == "complete" and
            (r.get("pool_limit") is None or ACCEPT_POOL_LIMIT) and
            os.path.exists(os.path.join(cell_dir(out_dir, name), "evaluated.csv")))


def read_cell_status(out_dir, name):
    try:
        with open(os.path.join(cell_dir(out_dir, name), "cell_status.json")) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def write_sweep_status(out_dir, plan):
    """Rebuilt from the cell directories on disk, for the WHOLE plan."""
    cells, counts = {}, {}
    for c in plan:
        st = read_cell_status(out_dir, c["name"])
        if is_complete(out_dir, c["name"]):
            state = "complete"
        elif st is None:
            state = "pending"
        elif st.get("ended") is None:
            state = "running" if pid_alive(st.get("pid")) else "stale_running"
        else:
            state = f"failed:{st.get('reason')}"
        st = st or {}
        elapsed = st.get("elapsed_s")
        if elapsed is None and st.get("started") and state == "running":
            elapsed = time.time() - st["started"]
        cells[c["name"]] = {"state": state, "tier": c["tier"],
                            "peak_rss_gb_watchdog": st.get("peak_rss_kb", 0) * 1024 / GB,
                            "elapsed_h": (elapsed or 0) / 3600}
        key = f"tier{c['tier']}|{c['arm']}"
        counts.setdefault(key, {}).setdefault(state.split(":")[0], 0)
        counts[key][state.split(":")[0]] += 1
    n_done = sum(1 for v in cells.values() if v["state"] == "complete")
    yc.write_json_atomic(os.path.join(out_dir, "sweep_status.json"), {
        "built_from": "cell directories on disk",
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_planned": len(plan), "n_complete": n_done,
        "sweep_status": "COMPLETE" if n_done == len(plan) else "PARTIAL",
        "counts": counts, "cells": cells})
    return n_done


# --------------------------------------------------------------------------
# Children
# --------------------------------------------------------------------------

def launch(out_dir, cell, args, her_predict):
    d = cell_dir(out_dir, cell["name"])
    attempt = 1
    if os.path.isdir(d):
        prev = read_cell_status(out_dir, cell["name"]) or {}
        if prev.get("ended") is None and pid_alive(prev.get("pid")):
            log(f"NOT relaunching {cell['name']}: its previous child (pid "
                f"{prev.get('pid')}) is still alive")
            return None
        if is_complete(out_dir, cell["name"]):
            log(f"NOT relaunching {cell['name']}: it completed meanwhile")
            return None
        attempt = int(prev.get("attempt", 0)) + 1
        keep = os.path.join(out_dir, "attempts", cell["name"], f"attempt_{attempt - 1}")
        os.makedirs(os.path.dirname(keep), exist_ok=True)
        shutil.move(d, keep)
    os.makedirs(d)
    # HER-PUBLISHED needs her guacamol-test tree, which must never share a
    # process with the pinned one, so it has its own runner (Amendment 2).
    published = cell["arm"] == yc.HER_PUBLISHED_ARM
    runner = RUN_ONE_PUBLISHED if published else RUN_ONE
    predict = her_predict[1] if published else her_predict[0]
    cmd = [sys.executable, "-u", runner, "--task", cell["task"], "--arm",
           cell["arm"], "--seed", str(cell["seed"]), "--n-init", str(cell["n_init"]),
           "--n-iterations", str(args.n_iterations), "--batch-size",
           str(args.batch_size), "--her-predict", predict, "--output-dir", d]
    if args.pool_limit is not None:
        cmd += ["--pool-limit", str(args.pool_limit)]
    env = dict(os.environ, PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
    fh = open(os.path.join(d, "stdout.log"), "w")
    proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env,
                            cwd=REPO, start_new_session=True)
    status = {"name": cell["name"], "tier": cell["tier"], "attempt": attempt,
              "pid": proc.pid, "cmd": cmd, "started": time.time(), "ended": None,
              "peak_rss_kb": 0, "samples": 0, "reason": None, "exit_code": None}
    yc.write_json_atomic(os.path.join(d, "cell_status.json"), status)
    log(f"launch {cell['name']} (tier {cell['tier']}, attempt {attempt}, "
        f"pid {proc.pid})")
    return {"proc": proc, "fh": fh, "status": status, "cell": cell, "dir": d,
            "kill_sent": None}


def sample_rss_kb(pids):
    if not pids:
        return {}
    p = subprocess.run(["ps", "-o", "pid=,rss=", "-p", ",".join(map(str, pids))],
                       capture_output=True, text=True)
    out = {}
    for line in p.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[int(parts[0])] = int(parts[1])
    return out


def vina_running():
    p = subprocess.run(["ps", "-axo", "pid=,comm="], capture_output=True, text=True)
    return [l.strip() for l in p.stdout.splitlines()
            if "vina" in os.path.basename(l.split(None, 1)[-1]).lower()]


def signal_group(job, sig):
    try:
        os.killpg(os.getpgid(job["proc"].pid), sig)
    except (ProcessLookupError, PermissionError):
        pass


def finalize(out_dir, job, reason=None):
    st = job["status"]
    st["ended"] = time.time()
    st["elapsed_s"] = st["ended"] - st["started"]
    st["exit_code"] = job["proc"].returncode
    complete = is_complete(out_dir, job["cell"]["name"])
    st["reason"] = reason or ("ok" if complete else f"exit_{st['exit_code']}")
    st["complete"] = complete
    job["fh"].close()
    yc.write_json_atomic(os.path.join(job["dir"], "cell_status.json"), st)
    yc.write_text_atomic(os.path.join(job["dir"], "peak_rss_watchdog.txt"),
                         f"{st['peak_rss_kb'] * 1024}\n# bytes, peak of ps rss "
                         f"sampled every poll ({st['samples']} samples)\n")
    log(f"end {job['cell']['name']}: {st['reason']} after "
        f"{st['elapsed_s'] / 3600:.2f} h, peak {st['peak_rss_kb'] * 1024 / GB:.3f} GB")


# --------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for t in (1, 2, 3, 4):
        p.add_argument(f"--tier{t}-tasks", nargs="+", default=[])
        p.add_argument(f"--tier{t}-arms", nargs="+", default=[])
        p.add_argument(f"--tier{t}-seeds", nargs="+", type=int, default=[])
    p.add_argument("--tier3-n-init", type=int, default=10)
    p.add_argument("--tier4-n-init", type=int, default=hp.RECIPE["n_init"])
    p.add_argument("--n-init", type=int, default=20)
    p.add_argument("--n-iterations", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--jobs", type=int, required=True)
    p.add_argument("--mem-cap-gb", type=float, required=True)
    p.add_argument("--per-run-timeout-h", type=float, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--poll-s", type=float, default=10.0)
    p.add_argument("--test-mode", action="store_true",
                   help="harness testing only: allows a non-pre-registered grid, "
                        "skips the git checks and the gates; report.py refuses it")
    p.add_argument("--pool-limit", type=int, default=None,
                   help="harness testing only (requires --test-mode)")
    return p.parse_args(argv)


def main(argv=None):
    global ACCEPT_POOL_LIMIT
    args = parse_args(argv)
    ACCEPT_POOL_LIMIT = bool(args.test_mode)
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest.json")

    problems = []
    if args.batch_size != 1:
        problems.append("batch size must be 1 (her loop is q=1)")
    if args.pool_limit is not None and not args.test_mode:
        problems.append("--pool-limit requires --test-mode")
    if args.per_run_timeout_h < 12 and not args.test_mode:
        problems.append("--per-run-timeout-h must be >= 12")
    prereg_commit = None
    if not args.test_mode:
        problems += check_preregistered(args)
        git_problems, prereg_commit = git_preflight()
        problems += git_problems
    if os.path.exists(manifest_path) and not args.resume:
        problems.append(f"{manifest_path} exists; pass --resume to continue it")
    vina = vina_running()
    if vina:
        problems.append(f"vina processes are running: {vina}")
    lock_path = os.path.join(out_dir, "sweep.lock")
    try:
        with open(lock_path) as fh:
            other = int(fh.read().split()[0])
        if other != os.getpid() and pid_alive(other):
            problems.append(f"another sweep (pid {other}) holds {lock_path}")
    except (OSError, ValueError, IndexError):
        pass
    if problems:
        for pr in problems:
            log(f"REFUSING TO LAUNCH: {pr}")
        return 2

    plan = build_plan(args)
    if not plan:
        log("REFUSING TO LAUNCH: empty plan")
        return 2

    if args.test_mode:
        gate_rc, gate_records, gate_dir = 0, {}, None
        her_predict = ("batched", "batched")
        log("TEST MODE: gates skipped")
    else:
        gate_rc, gate_records, gate_dir = run_gates(out_dir)
        if gate_rc != 0:
            failed = [k for k, v in gate_records.items() if v is False]
            log(f"REFUSING TO LAUNCH: gates failed (pytest exit {gate_rc}); "
                f"failing records: {failed}")
            return 2
        with open(os.path.join(gate_dir, "G3.json")) as fh:
            main_predict = json.load(fh)["her_predict"]
        published_predict = "unbatched"
        g3t4 = os.path.join(gate_dir, "G3_TIER4.json")
        if os.path.exists(g3t4):
            with open(g3t4) as fh:
                published_predict = json.load(fh)["her_predict"]
        elif any(c["arm"] == yc.HER_PUBLISHED_ARM for c in plan):
            log("REFUSING TO LAUNCH: the plan contains HER-PUBLISHED cells but "
                "the Tier 4 batching gate wrote no G3_TIER4.json")
            return 2
        her_predict = (main_predict, published_predict)
        log(f"gates passed; G3 -> her predictions {main_predict}, Tier 4 "
            f"{published_predict}; G5 determinism record: {gate_records.get('G5')}")

    if any(c["arm"] == yc.HER_PUBLISHED_ARM for c in plan):
        try:
            log(f"materialising her guacamol-test tree: {hp.materialize()}")
        except Exception as e:
            log(f"REFUSING TO LAUNCH: cannot materialise her tree: {e}")
            return 2
    code_digest, code_pairs = provenance.fingerprint_working_tree()
    top = subprocess.run(["ps", "-Ao", "pid,pcpu,rss,comm", "-r"],
                         capture_output=True, text=True).stdout.splitlines()[:12]
    invocation = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "argv": sys.argv,
                  "jobs": args.jobs, "mem_cap_gb": args.mem_cap_gb,
                  "per_run_timeout_h": args.per_run_timeout_h,
                  "gate_exit": gate_rc, "gate_records": gate_records,
                  "her_predict": her_predict[0],
                  "her_predict_published": her_predict[1],
                  "repo_head": yc.git_head(),
                  "result_code_sha256": code_digest,
                  "top_processes_at_launch": top}
    config = {"n_init": args.n_init, "tier3_n_init": args.tier3_n_init,
              "n_iterations": args.n_iterations, "batch_size": args.batch_size,
              "test_mode": args.test_mode, "pool_limit": args.pool_limit}
    if os.path.exists(manifest_path):
        with open(manifest_path) as fh:
            manifest = json.load(fh)
        if [c["name"] for c in manifest["plan"]] != [c["name"] for c in plan] \
                or manifest["config"] != config:
            log("REFUSING TO LAUNCH: --resume plan/config differs from the "
                "manifest's; a recovery run may not redefine the sweep")
            return 2
        if (manifest.get("her_predict"),
                manifest.get("her_predict_published")) != her_predict:
            log(f"REFUSING TO LAUNCH: her_predict {her_predict} != the sweep's "
                f"{(manifest.get('her_predict'), manifest.get('her_predict_published'))}")
            return 2
        pinned = manifest.get("result_code_files")
        if pinned is None:   # manifest written before fingerprints were pinned
            pinned = provenance.fingerprint_at(manifest["invocations"][0]["repo_head"])[1]
            manifest["result_code_files"] = pinned
        changed = provenance.diff(pinned, code_pairs)
        if changed:
            log(f"REFUSING TO LAUNCH: result code differs from the sweep's: {changed}")
            return 2
        manifest["invocations"].append(invocation)
    else:
        manifest = {"plan": plan, "config": config,
                    "her_predict": her_predict[0],
                    "her_predict_published": her_predict[1],
                    "result_code_sha256": code_digest, "result_code_files": code_pairs,
                    "preregistration_commit": prereg_commit,
                    "preregistration_sha256": (yc.sha256_file(PREREG)
                                               if os.path.exists(PREREG) else None),
                    "created": invocation["time"], "invocations": [invocation]}
    yc.write_json_atomic(manifest_path, manifest)

    yc.write_text_atomic(lock_path, f"{os.getpid()}\n")
    queue = [c for c in plan if not is_complete(out_dir, c["name"])]
    log(f"plan: {len(plan)} cells; {len(plan) - len(queue)} already complete; "
        f"{len(queue)} to run with {args.jobs} jobs, cap {args.mem_cap_gb} GB, "
        f"timeout {args.per_run_timeout_h} h")
    running = {}
    stop = {"reason": None}

    def on_signal(signum, _frame):
        stop["reason"] = f"signal_{signum}"
    for s in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s, on_signal)

    cap_kb = args.mem_cap_gb * GB / 1024
    timeout_s = args.per_run_timeout_h * 3600
    try:
        poll_loop(out_dir, plan, queue, running, stop, args, her_predict, cap_kb,
                  timeout_s, code_pairs)
    except BaseException as e:            # anything unexpected still stops children
        stop["reason"] = stop["reason"] or f"exception_{type(e).__name__}: {e}"
        log(f"sweep loop raised: {e!r}")
    finally:
        shutdown(out_dir, running, stop)
        try:
            os.remove(lock_path)
        except OSError:
            pass

    n_done = write_sweep_status(out_dir, plan)
    log(f"sweep finished: {n_done}/{len(plan)} cells complete (from disk)")
    if not args.test_mode:
        rc = subprocess.run([sys.executable, os.path.join(BENCH_DIR, "report.py"),
                             "--output-dir", out_dir], cwd=REPO).returncode
        log(f"report.py exit {rc}")
    return 0 if n_done == len(plan) else 1


def poll_loop(out_dir, plan, queue, running, stop, args, her_predict, cap_kb,
              timeout_s, code_pairs):
    while (queue or running) and stop["reason"] is None:
        while queue and len(running) < args.jobs and stop["reason"] is None:
            changed = provenance.diff(code_pairs, provenance.fingerprint_working_tree()[1])
            if changed:
                log(f"STOPPING: result code changed on disk since launch: {changed}")
                stop["reason"] = "result_code_changed"
                break
            cell = queue.pop(0)
            job = launch(out_dir, cell, args, her_predict)
            if job is not None:
                running[cell["name"]] = job
        time.sleep(args.poll_s)

        rss = sample_rss_kb([j["proc"].pid for j in running.values()])
        now = time.time()
        for name, job in list(running.items()):
            st = job["status"]
            kb = rss.get(job["proc"].pid)
            if kb is not None:
                st["samples"] += 1
                if kb > st["peak_rss_kb"]:
                    st["peak_rss_kb"] = kb
                    yc.write_json_atomic(os.path.join(job["dir"], "cell_status.json"), st)
            breach = None
            if kb is not None and kb > cap_kb:
                breach = "mem_cap"
            elif now - st["started"] > timeout_s:
                breach = "timeout"
            if breach and job["kill_sent"] is None:
                log(f"{breach}: {name} (rss {(kb or 0) * 1024 / GB:.3f} GB, "
                    f"{(now - st['started']) / 3600:.2f} h) -> SIGTERM")
                signal_group(job, signal.SIGTERM)
                job["kill_sent"] = (now, breach)
            elif job["kill_sent"] and now - job["kill_sent"][0] > KILL_GRACE_S \
                    and job["proc"].poll() is None:
                log(f"{name}: still alive {KILL_GRACE_S}s after SIGTERM -> SIGKILL")
                signal_group(job, signal.SIGKILL)
            if job["proc"].poll() is not None:
                finalize(out_dir, job, job["kill_sent"][1] if job["kill_sent"] else None)
                del running[name]

        vina = vina_running()
        if vina:
            log(f"STOPPING: a vina process appeared: {vina}")
            stop["reason"] = "vina_detected"
        write_sweep_status(out_dir, plan)


def shutdown(out_dir, running, stop):
    if stop["reason"] and running:
        log(f"stopping ({stop['reason']}): terminating {len(running)} children")
        for job in running.values():
            signal_group(job, signal.SIGTERM)
        deadline = time.time() + KILL_GRACE_S
        while time.time() < deadline and any(j["proc"].poll() is None
                                              for j in running.values()):
            time.sleep(1)
        for job in running.values():
            if job["proc"].poll() is None:
                signal_group(job, signal.SIGKILL)
                job["proc"].wait()
            finalize(out_dir, job, f"stopped_{stop['reason']}")


if __name__ == "__main__":
    sys.exit(main())
