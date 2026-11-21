#!/usr/bin/env python
"""Correlation + bounds gate for a NEW campaign, run as an NRP Job.

Two questions must be answered before any campaign is worth launching, and
both are answered by docking the SAME small sample against both of that
campaign's targets:

  1. TASK CORRELATION rho.  F19 puts the floor at 0.45 at 3.6% co-located
     labels and 0.259 at 9.8%.  Below the floor the ICM has no demonstrated
     benefit over independent per-task GPs, so the campaign's headline arm
     would be measuring nothing.

  2. BOUNDS SATURATION.  Every new campaign's bounds file was seeded from the
     DHFR docking range [-11, -5] and is marked PROVISIONAL.  If the observed
     range runs past either end, molecules clip to 0 or 1 on that axis and
     hypervolume stops discriminating -- silently.  This is the only chance to
     catch that before 18 jobs burn six hours each.

WHY THE GATE IS STRICTER THAN THE FLOOR.  Measured on rung 1: the pilot read
rho = +0.719 on random library molecules, but the campaign's OWN BO-selected
molecules read 0.532, and 0.330 among binders.  A pilot samples the library;
a campaign samples what the optimizer chose, and the optimizer is actively
hunting molecules on which the two targets DISAGREE.  So a pilot systematically
reads HIGH relative to the correlation the model actually operates at.  Treat
the pilot as an upper bound and require real headroom over the floor, not a
bound that merely clears it.

Chunked and idempotent: a chunk that already wrote its CSV returns it instead
of re-docking, so an opportunistic preemption costs only the chunk in flight.
"""
from __future__ import annotations
import argparse, json, os, pathlib, sys

REPO = pathlib.Path(__file__).resolve().parent.parent
LIBRARY_DIR = "/data/library"
SAMPLE_SEED = 20261121          # fixed so every chunk agrees on the sample


def link(dst: pathlib.Path, src: pathlib.Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        return
    dst.symlink_to(src, target_is_directory=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--campaign", required=True)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--chunk", type=int, required=True)
    p.add_argument("--n-chunks", type=int, required=True)
    p.add_argument("--workdir", default="/data/pilots")
    a = p.parse_args()

    if os.environ.get("MOGP_CAMPAIGN") != a.campaign:
        sys.exit(f"MOGP_CAMPAIGN={os.environ.get('MOGP_CAMPAIGN')!r} but "
                 f"--campaign={a.campaign!r}; the campaign is read at IMPORT "
                 f"time, so these must agree or the wrong targets get docked.")

    lib = pathlib.Path(LIBRARY_DIR)
    if not (lib / "smiles.csv").exists():
        sys.exit(f"library not found at {lib}/smiles.csv")
    link(REPO / "data" / "library", lib)

    out = pathlib.Path(a.workdir) / a.campaign
    out.mkdir(parents=True, exist_ok=True)
    chunk_csv = out / f"chunk{a.chunk:02d}.csv"
    if chunk_csv.exists():
        print(f"[skip] {chunk_csv} already written", flush=True)
        return 0

    import numpy as np, pandas as pd
    import campaigns, data, docking

    camp = campaigns.active_campaign()
    targets = [recv for kind, recv in camp["sources"].values() if kind == "dock"]
    if len(targets) != 2:
        sys.exit(f"expected exactly 2 docking targets, got {targets}")
    print(f"[pilot] campaign={a.campaign} targets={targets}", flush=True)

    # Receptors first. A fresh pod has no PDB on disk; rung 1 lost 18 cells to
    # exactly this.
    for t in targets:
        docking.download_protein(t)
        docking.prepare_protein(t)
        print(f"[recv] {t} ready ({docking.TARGETS[t]['pdb_id']})", flush=True)

    # Sample from the SCREENED library -- the 26,660 the optimizer actually
    # searches, not the 29,678 curated. Fixed seed, so every chunk slices the
    # same ordered sample.
    smiles = data.load_library(library_dir=str(lib))["smiles"]
    rng = np.random.default_rng(SAMPLE_SEED)
    pick = rng.choice(len(smiles), size=min(a.n, len(smiles)), replace=False)
    pick.sort()
    mine = np.array_split(pick, a.n_chunks)[a.chunk]
    mols = [smiles[i] for i in mine]
    print(f"[pilot] chunk {a.chunk}/{a.n_chunks}: {len(mols)} molecules "
          f"of {len(pick)} sampled from {len(smiles)} screened", flush=True)

    scores = docking.batch_dock_targets(mols, targets)
    df = pd.DataFrame({"SMILES": mols, "library_index": mine})
    for t in targets:
        df[t] = scores[t]
    df.to_csv(chunk_csv, index=False)
    ok = int(df[targets].notna().all(axis=1).sum())
    print(f"[done] wrote {chunk_csv}  ({ok}/{len(df)} scored on both)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
