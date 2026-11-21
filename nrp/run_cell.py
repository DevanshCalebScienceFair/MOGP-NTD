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
    "indep": ["--model", "independent"],
}

# The asym arm spends fewer docking calls per molecule (only ~5% of molecules
# get the second target), so at the SAME iteration count it would use a smaller
# budget and the comparison would be about budget, not about labelling. F14
# matched the budget by giving the asymmetric arm more iterations; the same is
# done here. 40 + 5*50 = 290 molecules x 2 targets = 580 calls for `full`;
# `asym` gets the iteration count that spends 580 calls at its own label rate.
ASYM_N_ITERATIONS = 85


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
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--workdir", default="/data/rung1")
    a = ap.parse_args()

    # The campaign is selected by environment variable and read at IMPORT time,
    # so it must be set before anything imports mogp. Asserting it here rather
    # than defaulting means a manifest that forgets it fails loudly instead of
    # running malaria and filing the results under Chagas.
    campaign = os.environ.get("MOGP_CAMPAIGN")
    if campaign != "chagas":
        sys.exit(f"MOGP_CAMPAIGN must be 'chagas' for this cell, got {campaign!r}. "
                 f"Without it this would run the malaria objective set and write "
                 f"it into the Chagas output directory.")

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

    n_iter = ASYM_N_ITERATIONS if a.arm == "asym" else N_ITERATIONS
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
