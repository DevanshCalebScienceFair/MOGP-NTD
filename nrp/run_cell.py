#!/usr/bin/env python
"""In-container entrypoint for ONE rung-1 Chagas campaign cell (one arm, one seed).

WHAT A CELL IS. `--arm mogp` is the shared Hadamard ICM; `--arm indep` is
independent per-task GPs on the same molecules, budget and seed. Everything else
is held fixed, so the only difference between the two arms is whether the model
shares information between TcDHFR and hDHFR. That is the claim this whole
project exists to test, and the independent arm is what makes it falsifiable.

HOW RESUME WORKS, AND WHY IT NEEDS NO CHANGES TO loop.py.
This namespace can only run `priorityClassName: opportunistic`, so every pod is
preemptible and `loop.py` has no warm start. Resume is instead a property of two
facts that are already true:

  1. A run is deterministic given its seed. `BOLoop.__init__` seeds torch once
     and the call sequence is fixed, so a restarted cell replays the IDENTICAL
     trajectory: same initial design, same batches, same molecules.
  2. Docking is cached by `oracle_fingerprint`, and the cache lives on the
     persistent volume.

So a preempted cell restarts from iteration 0 and pays nothing for docking,
because every call it repeats is a cache hit. It re-pays GP training and
acquisition, which are the cheap half. The result is bit-identical to an
uninterrupted run, which a warm-started one would NOT be.

ONE CACHE PER CELL, deliberately. A single shared SQLite file on CephFS with 12
concurrent writers is a locking hazard over a network filesystem. Cells overlap
only partially anyway, since their seeds differ. Correctness over a modest
saving.

The library is read from the volume, not the image: `data/library/` is 63 MB of
generated data, it is gitignored, and all 12 cells share one copy.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent

# sys.path[0] is the SCRIPT's directory, nrp/, not the repo root, and
# os.chdir() does not change that. Without this line `import campaigns` fails
# with ModuleNotFoundError after the container has already extracted a 2.9 GB
# environment and verified Vina.
sys.path.insert(0, str(REPO))

# Matched to the asym campaign's budget so the two are comparable: 40 initial +
# 50 iterations x 5 per batch = 290 molecules, 580 docking calls at two targets.
LIBRARY_DIR = "/data/library"
N_INIT, BATCH_SIZE, N_ITERATIONS = 40, 5, 50

# Held fixed across both arms. alpha=1e-3 is BoTorch's own default for five
# objectives and is the 17.0x cost fix; pool 2000 bounds acquisition cost and
# MUST be identical across arms or an advantage is a harness artifact.
COMMON = ["--posterior", "joint", "--acquisition-alpha", "1e-3",
          "--acquisition-pool-size", "2000", "--rank", "1"]

# THE THREE ARMS, and why these three.
#
# `full` vs `indep` at complete labelling is NOT the experiment. Every molecule
# carries both docking labels there, the design is block-complete,
# autokrigeability applies (Bonilla, Chai & Williams 2008 s2.3) and the ICM
# posterior mean collapses to independent per-task GPs. F12 measured exactly
# that over 10 seeds: 197 of 400 checkpoints, every endpoint null. Repeating it
# on a new target would reproduce a known negative, not test anything.
#
# `asym` is the headline. F19's correlation floor is a statement about the ICM's
# advantage at a SMALL co-located-label count; at 100% co-location there is no
# advantage to threshold. So measuring rho = +0.719 on TcDHFR/hDHFR licenses an
# ASYMMETRIC-label campaign, and that cell has never been run inside F17's band.
# F14 ran it at 25% co-location, 2.5x above the k=8 peak, and it lost 6/6.
# CLAUDE.md lists this under "NOT shown, and do not claim it".
#
# `indep` stays as the control that says whether sharing matters here at all.
ARMS = {
    "full":  ["--model", "hadamard"],
    "asym":  ["--model", "hadamard",
              "--hdhfr-policy", "uniform",
              "--hdhfr-fraction", "0.05",
              "--hdhfr-floor", "8"],
    # Identical to `asym` except for the round count, which is set by
    # ASYM102_N_ITERATIONS below. Separate key so the two never share a
    # workdir and the 85-round result stays on disk untouched.
    "asym102": ["--model", "hadamard",
                "--hdhfr-policy", "uniform",
                "--hdhfr-fraction", "0.05",
                "--hdhfr-floor", "8"],
    "indep": ["--model", "independent"],
}

# EXTERNAL BASELINES. These do NOT run loop.py; they are separate programs with
# their own CLIs, so run_cell builds a different command for them.
#
# Chagas had no external competitor at all. full / indep / asym are three
# variants of OUR method, so rung 1 could show our variants tying each other
# and nothing about whether any of them beats the field. On malaria the field
# was GP-MOBO and greedy filter-then-dock; this adds both to Chagas at the same
# docking budget.
BASELINE_ARMS = {"gpmobo", "greedy"}

# The asym arm spends fewer docking calls per molecule (only ~5% of molecules
# get the second target), so at the SAME iteration count it would use a smaller
# budget and the comparison would be about budget, not about labelling. F14
# matched the budget by giving the asymmetric arm more iterations; the same is
# done here. 40 + 5*50 = 290 molecules x 2 targets = 580 calls for `full`;
# `asym` gets the iteration count that spends 580 calls at its own label rate.
ASYM_N_ITERATIONS = 85

# THE BUDGET-MATCHED, IN-BAND ASYMMETRIC ARM.
#
# The 85-round arm was in F17's co-location band but UNDER budget: 465
# molecules + 24 second-target labels = 489 docking calls against the full
# arm's 290 x 2 = 580, i.e. 16% cheaper. It lost, and the obvious objection is
# that it was never given the same money.
#
# 102 rounds closes that. 40 + 102*5 = 550 molecules; the uniform policy's
# quota is ceil(0.05 * 550) = 28 second-target labels, so 550 + 28 = 578 calls
# against 580. Matched to 0.3%, and 28/550 = 5.1% co-located, still inside the
# 3.6%-9.8% band.
#
# This is the cell that could not exist at 465 molecules, where the band caps
# labels at ~46 and the total at ~511. Growing the molecule count is what buys
# both constraints at once.
ASYM102_N_ITERATIONS = 102

# Greedy has no acquisition and no iteration count; it is a total molecule
# budget. 290 x 2 targets = 580 calls, matching every other arm.
GREEDY_N_TOTAL = 290


def link(path: pathlib.Path, target: pathlib.Path):
    """Point `path` at `target`, replacing whatever is there.

    `docking_cache.DEFAULT_CACHE_PATH` and `data.load_library` both resolve
    relative to the working directory, so redirecting them onto the volume is a
    symlink rather than a code change.
    """
    target.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.exists():
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            import shutil
            shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(target, target_is_directory=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True,
                    choices=sorted(set(ARMS) | BASELINE_ARMS))
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--workdir", default="/data/rung1")
    # OPTIONAL and defaulting to "chagas" on purpose. Pods that are already
    # running were launched with a command line that has no --campaign, and a
    # preempted pod is restarted with that SAME command line, so making this
    # required would kill every in-flight retry. The default reproduces the
    # previous behaviour exactly.
    ap.add_argument("--campaign", default="chagas",
                    help="campaigns.py key this cell must run; asserted against "
                         "MOGP_CAMPAIGN so a manifest cannot run one objective "
                         "set and file it under another campaign's directory")
    a = ap.parse_args()

    # The campaign is selected by environment variable and read at IMPORT time,
    # so it must be set before anything imports mogp. Asserting it here rather
    # than defaulting means a manifest that forgets it fails loudly instead of
    # running malaria and filing the results under Chagas.
    campaign = os.environ.get("MOGP_CAMPAIGN")
    if campaign != a.campaign:
        sys.exit(f"MOGP_CAMPAIGN must be {a.campaign!r} for this cell, got "
                 f"{campaign!r}. Without it this would run a DIFFERENT objective "
                 f"set and write it into {a.workdir}, where every reader would "
                 f"take it for a {a.campaign} result.")

    work = pathlib.Path(a.workdir)
    cell = work / f"{a.arm}_seed{a.seed}"
    cell.mkdir(parents=True, exist_ok=True)

    # Idempotent on retry: Kubernetes restarts a preempted pod with the same
    # input, and a finished cell must not re-run and overwrite itself.
    done = cell / "pareto_front.csv"
    if done.exists():
        print(f"[done] {cell} already has pareto_front.csv; nothing to do.")
        return 0

    os.chdir(REPO)

    # The library is uploaded once to the ROOT of the PVC, not into the campaign
    # workdir: /data/library, not /data/rung1/library. Pointing the symlink at
    # `work / "library"` silently created an empty directory, and the loop then
    # failed 0.1 min in with "Library file not found: data/library/smiles.csv"
    # after every pod had already spent minutes extracting a 2.9 GB env.
    lib = pathlib.Path(LIBRARY_DIR)
    if not (lib / "smiles.csv").exists():
        sys.exit(f"library not found at {lib}/smiles.csv (contents: "
                 f"{sorted(x.name for x in lib.iterdir()) if lib.is_dir() else 'no such dir'}). "
                 f"Upload it with nrp/bootstrap.sh before launching cells.")
    link(REPO / "data" / "library", lib)
    link(REPO / "data" / "docking_cache", cell / "docking_cache")

    # Every receptor this campaign docks against has to be ON DISK before the
    # loop starts. On a laptop the raw PDB is already sitting in the repo from
    # an earlier run, so prepare_protein() finds it; in a fresh clone it does
    # not exist and the run dies with FileNotFoundError: '3INV.pdb' two minutes
    # in, after the 2.9 GB env extract. Downloading here also means the
    # receptor is prepared once per pod rather than lazily inside the first
    # docking call, so a prep failure surfaces before any compute is spent.
    import campaigns as _c, docking as _d
    for _task, (_kind, _recv) in _c.active_campaign()["sources"].items():
        if _kind != "dock":
            continue
        _d.download_protein(_recv)
        _d.prepare_protein(_recv)
        print(f"[recv] {_recv} ready ({_d.TARGETS[_recv]['pdb_id']})", flush=True)

    n_iter = {"asym": ASYM_N_ITERATIONS,
              "asym102": ASYM102_N_ITERATIONS}.get(a.arm, N_ITERATIONS)

    if a.arm == "gpmobo":
        # external/ is GITIGNORED, so the shallow clone of this repo does NOT
        # carry the GP-MOBO reference implementation and gpmobo_ref.load()
        # raises FileNotFoundError the moment GPMOBOBaseline.__init__ runs.
        # All six seed-0..5 pods died on exactly this. Clone it here, pinned to
        # the same commit gpmobo_ref.PINNED_COMMIT names, so "GP-MOBO" in the
        # result table means one fixed upstream revision on every backend.
        #
        # Only kern_gp/{gp_model,kern_gp_matrices}.py are imported, and they
        # need nothing beyond numpy, scipy and rdkit, all already in the env.
        import gpmobo_ref as _g
        ext = REPO / "external" / "GP-MOBO"
        if not (ext / "kern_gp" / "kern_gp_matrices.py").exists():
            ext.mkdir(parents=True, exist_ok=True)
            # A --depth 1 clone cannot land on an arbitrary SHA, so fetch the
            # one commit by name. GitHub serves uploadpack.allowReachableSHA1;
            # a full clone is the fallback because an 18 MB history is cheap
            # next to a dead cell.
            def _git(*args):
                return subprocess.run(["git", "-C", str(ext), *args]).returncode
            ok = (_git("init", "-q") == 0
                  and _git("remote", "add", "origin", _g.REPO_URL) == 0
                  and _git("fetch", "-q", "--depth", "1",
                           "origin", _g.PINNED_COMMIT) == 0
                  and _git("checkout", "-q", "FETCH_HEAD") == 0)
            if not ok:
                shutil.rmtree(ext, ignore_errors=True)
                subprocess.run(["git", "clone", "-q", _g.REPO_URL, str(ext)],
                               check=True)
                subprocess.run(["git", "-C", str(ext), "checkout", "-q",
                                _g.PINNED_COMMIT], check=True)
        # Fail here, loudly, rather than inside the baseline's constructor.
        _g.ensure_available(str(ext))
        print(f"[gpmobo] upstream {_g.head_commit(str(ext))} at {ext}",
              flush=True)

        # --acquisition-pool-size MUST be the same 2000 every other arm uses.
        # Without it this arm scores the whole ~26,600-molecule library per
        # pick, which is both a harness advantage and ~29 min/iteration.
        cmd = [sys.executable, "baseline_gpmobo.py",
               "--library-dir", "data/library",
               "--seed", str(a.seed),
               "--n-init", str(N_INIT),
               "--batch-size", str(BATCH_SIZE),
               "--n-iterations", str(N_ITERATIONS),
               "--acquisition-pool-size", "2000",
               "--output-dir", str(cell)]
    elif a.arm == "greedy":
        cmd = [sys.executable, "baseline_greedy.py",
               "--library-dir", "data/library",
               "--seed", str(a.seed),
               "--batch-size", str(BATCH_SIZE),
               "--n-total", str(GREEDY_N_TOTAL),
               "--output-dir", str(cell)]
    else:
        cmd = [sys.executable, "loop.py",
               "--seed", str(a.seed),
               "--n-init", str(N_INIT),
               "--batch-size", str(BATCH_SIZE),
               "--n-iterations", str(n_iter),
               "--output-dir", str(cell),
               *COMMON, *ARMS[a.arm]]

    # Provenance beside the results, so the artifact states what produced it
    # rather than only the environment that is gone once the pod exits.
    import campaigns
    (cell / "run_config.json").write_text(json.dumps({
        "arm": a.arm, "seed": a.seed, "cmd": cmd,
        "n_init": N_INIT, "batch_size": BATCH_SIZE,
        "n_iterations": n_iter,
        "n_molecules": N_INIT + BATCH_SIZE * n_iter,
        # For `asym` this is an UPPER bound: only ~5% of molecules get the
        # second target, so the realized call count is read from evaluated.csv,
        # never assumed from this number.
        "budget_dock_calls_upper": (N_INIT + BATCH_SIZE * n_iter) * 2,
        "backend": "NRP/Nautilus", "node": os.environ.get("NODE_NAME", "?"),
        # The conda env has no ninja and the cell image has no C++ compiler, so
        # botorch cannot JIT its fused qLogEHVI kernel. That path is ~3x faster
        # but NUMERICALLY IDENTICAL, so the override is a wall-clock decision,
        # not a scientific one. Recorded here because a timing claim read off
        # these cells must say which acquisition path produced it.
        "slow_ehvi_path": os.environ.get("MOGP_ALLOW_SLOW_EHVI") == "1",
        **campaigns.record()}, indent=2))
    print(campaigns.describe(), flush=True)
    print(f"[cell] {a.arm} seed {a.seed} -> {cell}", flush=True)
    print(f"[cmd ] {' '.join(cmd)}", flush=True)

    t0 = time.time()
    rc = subprocess.call(cmd)
    print(f"[exit] rc={rc} after {(time.time() - t0) / 60:.1f} min", flush=True)

    # iteration_timings.csv is what decides the GPU question: it splits each
    # iteration into GP training, acquisition and docking seconds. Printed so
    # the split is in the pod log even if the volume is later cleaned.
    timings = cell / "iteration_timings.csv"
    if timings.exists():
        print("\n[timings] " + timings.name)
        print(timings.read_text()[-2000:], flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
