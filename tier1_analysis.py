"""Tier 1 coverage diagnostics for the 10-seed campaign — post-hoc, read-only.

Three analyses over already-cached campaign output. Nothing here imports or
touches the benchmarked code path (``loop.py``'s acquisition, the surrogate, the
docking oracle), so the 117-hour campaign stays valid:

1. ``umap``    — project the library's Morgan fingerprints to 2D and overlay the
                 oracle front, split into what MOGP finds and what it misses,
                 coloured by the iteration at which each was acquired. Answers
                 whether the missed molecules sit in SPARSE PERIPHERAL regions
                 (=> fix the pool sampler) or INSIDE regions MOGP already sampled
                 densely (=> fix the acquisition rule). The verdict is decided by
                 the numbers in ``umap_verdict.json``, not by eye; the figure is
                 the illustration.
2. ``circles`` — #Circles diversity (Yong et al., arXiv:2507.13704) at Tanimoto
                 DISTANCE thresholds 0.60 and 0.75.
3. ``igdplus`` — IGD+ (Ishibuchi, Imada, Masuyama & Nojima, EMO 2019,
                 DOI 10.1007/978-3-030-12598-1_27) against the oracle front.
                 IGD+, never plain IGD, which is Pareto non-compliant.

All objective-space work happens in the project's own shared normalized frame
(``evaluation.normalize``: fixed min-max bounds, flipped to pure maximization,
clipped to [0,1]) so every number here is commensurable with the campaign's
hypervolumes. Reusing that frame is also why IGD+ agrees with hypervolume by
construction — see the note ``igdplus`` writes into its own output.

Usage
-----
    python tier1_analysis.py all \
        --campaign-root campaign_results \
        --out campaign_results/aggregate_10seed_cleanGPMOBO/tier1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

import evaluation
from mogp import TASK_NAMES

OBJ = list(TASK_NAMES)

# Directories that are campaign RECORD and must never be written to. Guarded in
# resolve_out_dir rather than trusted, because a single stray --out would edit
# the artifacts the whole campaign is reported from.
PROTECTED = ("aggregate_10seed", "seed_0", "seed_1", "seed_2", "seed_3", "seed_4",
             "seed_5", "seed_6", "seed_7", "seed_8", "seed_9")

# Method label -> the subdirectory names it may appear under, most specific
# first. The GP-MOBO arm was re-run after the ligand-efficiency units defect, so
# a clean re-run directory wins over the original if both are present.
# Method label -> candidate paths, as format strings over {root} and {s}, most
# specific first. The GP-MOBO arm was re-run after the ligand-efficiency units
# defect and its re-run lives under a SEPARATE root, not a sibling directory of
# the original, so a plain subdirectory search finds the superseded LE arm and
# silently reports it as "GP-MOBO". The corrected root is therefore tried first.
METHOD_DIRS = {
    "MOGP": (
        "{root}/seed_{s}/mogp/seed_{s}",
        "{root}/seed_{s}/mogp",
    ),
    "GPMOBO": (
        "{root}/aggregate_10seed_cleanGPMOBO/seed_{s}/gpmobo/seed_{s}",
        "{root}/seed_{s}/gpmobo_clean/seed_{s}",
        "{root}/seed_{s}/gpmobo/seed_{s}",
        "{root}/seed_{s}/gpmobo",
    ),
    "GPMOBO_BROKEN": (
        "{root}/seed_{s}/gpmobo/seed_{s}",
    ),
    "Greedy": (
        "{root}/seed_{s}/greedy/seed_{s}",
        "{root}/seed_{s}/greedy",
    ),
}

# NaN policy. About 2.9% of evaluated rows carry NaN in BOTH docking columns and
# never in the three ADMET columns -- i.e. a failed dock, not a partial one. Two
# sets are therefore kept distinct throughout, and conflating them is the error
# this constant exists to prevent:
#
#   EVALUATED -- every row. Budget was spent and the loop visited that chemistry,
#                so this is the right set for "did MOGP sample near here?".
#   COMPLETE  -- rows finite on all five objectives. A molecule with no docking
#                score cannot sit on a Pareto front or carry hypervolume, so this
#                is the right set for anything in objective space.
#
# COMPLETE is also what evaluation.compute_hypervolume already does internally
# (it drops non-finite rows before normalizing), so using it keeps every number
# here commensurable with the campaign's reported hypervolumes.
NAN_POLICY = "complete-case in objective space; all evaluated rows for coverage"


# --------------------------------------------------------------------------- #
# Loading. Read-only, and loud about anything it cannot find.
# --------------------------------------------------------------------------- #
def find_run_dir(root, method, seed):
    """Locate one (method, seed) run directory, trying candidates in order."""
    for pattern in METHOD_DIRS[method]:
        candidate = pattern.format(root=root, s=seed)
        if os.path.isfile(os.path.join(candidate, "evaluated.csv")):
            return candidate
    return None


def load_run(root, method, seed):
    """One run's evaluated set, front and history, with acquisition iterations.

    ``evaluated.csv`` carries no iteration column, but its rows are written in
    acquisition order (``self.evaluated_indices`` is appended to), and
    ``history.csv`` records the cumulative ``n_evaluated`` after each iteration.
    That pair reconstructs the acquisition iteration exactly: everything before
    the first history row is the initial design (iteration 0).
    """
    run_dir = find_run_dir(root, method, seed)
    if run_dir is None:
        raise FileNotFoundError(
            f"No run for method={method} seed={seed} under {root}. Looked for "
            f"{METHOD_DIRS[method]} in seed_{seed}/.")

    ev = pd.read_csv(os.path.join(run_dir, "evaluated.csv"))
    pf_path = os.path.join(run_dir, "pareto_front.csv")
    pf = pd.read_csv(pf_path) if os.path.isfile(pf_path) else None
    hist_path = os.path.join(run_dir, "history.csv")
    hist = pd.read_csv(hist_path) if os.path.isfile(hist_path) else None

    # The arms diverge on their REPORTED-only columns: MOGP and Greedy carry
    # *_LE and Selectivity_Index_LE, both GP-MOBO arms carry *_kcal and no
    # Selectivity_Index_LE. Nothing here reads those, so the divergence is
    # harmless -- but only the five optimized objectives may be assumed present,
    # so require exactly those and nothing more.
    missing = [c for c in OBJ if c not in ev.columns]
    if missing:
        raise ValueError(f"{run_dir}/evaluated.csv is missing objectives: {missing}")

    ev["acq_iteration"] = acquisition_iteration(len(ev), hist)
    ev["complete"] = np.isfinite(ev[OBJ].to_numpy(float)).all(axis=1)
    ev["method"], ev["seed"], ev["run_dir"] = method, seed, run_dir
    return {"dir": run_dir, "evaluated": ev, "pareto": pf, "history": hist,
            "schema_extra": sorted(c for c in ev.columns
                                   if c not in OBJ + ["SMILES", "acq_iteration",
                                                      "complete", "method", "seed",
                                                      "run_dir"])}


def acquisition_iteration(n_rows, history):
    """Per-row acquisition iteration; 0 for the initial design.

    ``history.csv``'s ``n_evaluated`` is CUMULATIVE and its first row already
    includes the initial design, so the size of that design is not recorded
    directly. It is recovered from the first inter-iteration difference, which is
    the batch size. Returns all-NaN when history is unavailable, so a missing
    file degrades the UMAP colouring rather than inventing an ordering.
    """
    it = np.full(n_rows, np.nan)
    if history is None or "n_evaluated" not in history.columns or history.empty:
        return it

    n_eval = history["n_evaluated"].to_numpy(int)
    iters = (history["iteration"].to_numpy(int) if "iteration" in history.columns
             else np.arange(1, len(n_eval) + 1))

    batch = int(n_eval[1] - n_eval[0]) if len(n_eval) >= 2 else 0
    n_init = max(int(n_eval[0]) - batch, 0)

    lo = min(n_init, n_rows)
    it[:lo] = 0.0                                  # the initial design
    for cum, iteration in zip(n_eval, iters):
        hi = min(int(cum), n_rows)
        if hi > lo:
            it[lo:hi] = float(iteration)
            lo = hi
    if lo < n_rows:                                # rows past the last boundary
        it[lo:] = float(iters[-1])
    return it


def load_campaign(root, seeds, methods):
    runs, problems = {}, []
    for method in methods:
        for seed in seeds:
            try:
                runs[(method, seed)] = load_run(root, method, seed)
            except (FileNotFoundError, ValueError) as exc:
                problems.append(str(exc))
    if problems:
        for p in problems:
            print(f"  ! {p}", file=sys.stderr)
    if not runs:
        raise SystemExit(
            "No campaign runs loaded. Point --campaign-root at the directory "
            "holding seed_0 .. seed_9.")
    return runs


# --------------------------------------------------------------------------- #
# The oracle front: the union of everything any run docked.
# --------------------------------------------------------------------------- #
def build_oracle(runs):
    """Pooled docked set and its Pareto front, in the shared normalized frame.

    One molecule can appear in several runs; docking is cached and
    deterministic, so the duplicates agree. Rather than assume it, the objective
    values are taken from the first occurrence and disagreements are counted and
    reported.
    """
    frames = [r["evaluated"][["SMILES"] + OBJ] for r in runs.values()]
    pooled_all = pd.concat(frames, ignore_index=True)
    complete = np.isfinite(pooled_all[OBJ].to_numpy(float)).all(axis=1)
    pooled = pooled_all[complete]

    first = pooled.drop_duplicates(subset="SMILES", keep="first").reset_index(drop=True)
    merged = pooled.merge(first, on="SMILES", suffixes=("", "_ref"))
    diffs = np.abs(merged[OBJ].to_numpy(float)
                   - merged[[f"{c}_ref" for c in OBJ]].to_numpy(float))
    inconsistent = int((diffs > 1e-6).any(axis=1).sum())

    Y = first[OBJ].to_numpy(float)
    Y_norm = evaluation.normalize(Y, objective_indices=list(range(len(OBJ))))
    ones = np.ones(len(OBJ))
    # compute_pareto_front returns a boolean MASK, not indices. Coercing that
    # mask with astype(int) yields a 0/1 index array that silently selects rows
    # 0 and 1 over and over, so convert it explicitly.
    mask, front_norm = evaluation.compute_pareto_front(Y_norm, ones)
    idx = np.flatnonzero(np.asarray(mask, bool))

    # The SAME pool has two different Pareto fronts depending on whether
    # dominance is tested in raw units or in the normalized frame, because
    # normalize() CLIPS to [0,1]: values outside the fixed bounds saturate, and
    # saturation creates ties that make some raw-front members weakly dominated.
    # The campaign's published 411-molecule front is the RAW one; 
    # everything else in the project (hypervolume, and therefore IGD+) lives in
    # the normalized frame. Both are computed so capture can be reported against
    # either, and so the difference is visible rather than a silent discrepancy.
    raw_mask, _ = evaluation.compute_pareto_front(
        Y, np.asarray(evaluation.OBJECTIVE_SIGNS, dtype=float))
    raw_idx = np.flatnonzero(np.asarray(raw_mask, bool))
    saturating = int(((Y_norm <= 0.0) | (Y_norm >= 1.0)).any(axis=1).sum())

    return {
        "docked": first,                       # every unique docked molecule
        "docked_norm": Y_norm,
        "front_idx": idx,
        "front_smiles": first["SMILES"].to_numpy()[idx],
        "front_norm": np.asarray(front_norm, float),
        "n_docked": int(len(first)),
        "n_front": int(len(idx)),
        "n_rows_all": int(len(pooled_all)),
        "n_rows_incomplete": int((~complete).sum()),
        "n_unique_smiles_all_rows": int(pooled_all["SMILES"].nunique()),
        "n_unique_smiles_complete": int(len(first)),
        "raw_front_idx": raw_idx,
        "raw_front_smiles": first["SMILES"].to_numpy()[raw_idx],
        "n_front_raw_units": int(len(raw_idx)),
        "n_rows_saturating_a_bound": saturating,
        "inconsistent_duplicate_rows": inconsistent,
        "hv": float(evaluation.compute_hypervolume(Y[idx])),
    }


# --------------------------------------------------------------------------- #
# Fingerprints and Tanimoto.
# --------------------------------------------------------------------------- #
def load_library_fingerprints(library_dir):
    """Raw cached library: every row on disk, before the shared heavy-atom floor.

    The floor is what the loop SEARCHES; the full cached set is what the UMAP is
    asked to depict, so both counts are returned and reported.
    """
    smiles = pd.read_csv(os.path.join(library_dir, "smiles.csv"))["SMILES"].to_numpy()
    fps = np.load(os.path.join(library_dir, "fingerprints.npy"))
    if len(smiles) != len(fps):
        raise ValueError(f"library rows disagree: {len(smiles)} smiles, {len(fps)} fps")
    return smiles, np.ascontiguousarray(fps, dtype=np.float32)


def tanimoto_matrix(A, B, block=2048):
    """Pairwise Tanimoto SIMILARITY between two binary fingerprint blocks.

    c / (a + b - c) with c the bit intersection. Blocked over rows of A so a
    large query set cannot allocate an (|A|, |B|) intermediate all at once.
    """
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)
    b_pop = B.sum(axis=1)
    out = np.empty((A.shape[0], B.shape[0]), dtype=np.float32)
    for lo in range(0, A.shape[0], block):
        hi = min(lo + block, A.shape[0])
        c = A[lo:hi] @ B.T
        denom = A[lo:hi].sum(axis=1)[:, None] + b_pop[None, :] - c
        np.divide(c, np.where(denom == 0, 1.0, denom), out=out[lo:hi])
    return out


# --------------------------------------------------------------------------- #
# 2. #Circles  (Yong et al., arXiv:2507.13704)
# --------------------------------------------------------------------------- #
def n_circles(fps, threshold):
    """#Circles: size of a set whose members are pairwise farther apart than
    ``threshold`` in Tanimoto DISTANCE (= 1 - similarity).

    The exact quantity is a maximum independent set and is NP-hard; the standard
    greedy sphere-exclusion approximation is used, which is what the reference
    implementation does. It is order-dependent, so the input order (acquisition
    order, as written by the run) is used unchanged and identically for every
    method — the comparison across methods is like-for-like even though the
    absolute value is a lower bound on the true maximum.
    """
    n = len(fps)
    if n == 0:
        return 0
    sim = tanimoto_matrix(fps, fps)
    keep_sim_below = 1.0 - float(threshold)     # distance > t  <=>  sim < 1 - t
    kept = [0]
    for i in range(1, n):
        if np.all(sim[i, kept] < keep_sim_below):
            kept.append(i)
    return len(kept)


def run_circles(runs, oracle, lib_smiles, lib_fps, out_dir, thresholds=(0.60, 0.75)):
    fp_of = {s: i for i, s in enumerate(lib_smiles)}

    def fps_for(smiles_iter):
        rows = [fp_of[s] for s in smiles_iter if s in fp_of]
        return lib_fps[rows], len(rows)

    per_seed = []
    for (method, seed), run in sorted(runs.items()):
        pf = run["pareto"]
        if pf is None:
            continue
        fps, n_matched = fps_for(pf["SMILES"].tolist())
        row = {"method": method, "seed": seed, "set": "final_pareto_front",
               "n_molecules": len(pf), "n_matched_to_library": n_matched}
        for t in thresholds:
            row[f"n_circles_t{t:.2f}"] = n_circles(fps, t)
        per_seed.append(row)

    # The pooled docked set, per method: every unique molecule that method
    # docked across all its seeds.
    for method in sorted({m for m, _ in runs}):
        smiles = pd.unique(pd.concat(
            [r["evaluated"]["SMILES"] for (m, _), r in runs.items() if m == method],
            ignore_index=True))
        fps, n_matched = fps_for(smiles)
        row = {"method": method, "seed": "pooled", "set": "docked_union_all_seeds",
               "n_molecules": len(smiles), "n_matched_to_library": n_matched}
        for t in thresholds:
            row[f"n_circles_t{t:.2f}"] = n_circles(fps, t)
        per_seed.append(row)

    # And the oracle front + the whole docked pool, as reference ceilings.
    for label, smiles in (("oracle_front", oracle["front_smiles"]),
                          ("docked_pool_all_methods", oracle["docked"]["SMILES"].to_numpy())):
        fps, n_matched = fps_for(smiles)
        row = {"method": "REFERENCE", "seed": "pooled", "set": label,
               "n_molecules": len(smiles), "n_matched_to_library": n_matched}
        for t in thresholds:
            row[f"n_circles_t{t:.2f}"] = n_circles(fps, t)
        per_seed.append(row)

    df = pd.DataFrame(per_seed)
    df.to_csv(os.path.join(out_dir, "circles_per_seed.csv"), index=False)

    cols = [f"n_circles_t{t:.2f}" for t in thresholds]
    front = df[df["set"] == "final_pareto_front"]
    summary = (front.groupby("method")[cols + ["n_molecules"]]
                    .agg(["mean", "std", "count"]).round(3))
    summary.to_csv(os.path.join(out_dir, "circles_summary.csv"))
    print("\n#Circles over each seed's final Pareto front (mean +/- sd, n seeds):")
    for method, g in front.groupby("method"):
        parts = [f"{c.replace('n_circles_', '')}: "
                 f"{g[c].mean():.1f} +/- {g[c].std(ddof=1):.1f}" for c in cols]
        print(f"  {method:<7} n={len(g):<3} front={g['n_molecules'].mean():.1f} "
              f"| " + " | ".join(parts))
    print("\n#Circles over each method's docked union, and the references:")
    for _, r in df[df["seed"] == "pooled"].iterrows():
        parts = [f"{c.replace('n_circles_', '')}: {int(r[c])}" for c in cols]
        print(f"  {r['method']:<9} {r['set']:<24} n={int(r['n_molecules']):<6} "
              + " | ".join(parts))
    return df


# --------------------------------------------------------------------------- #
# 3. IGD+  (Ishibuchi, Imada, Masuyama & Nojima, EMO 2019)
# --------------------------------------------------------------------------- #
def igd_plus(A_norm, Z_norm):
    """IGD+(A) with reference set Z, in a MAXIMIZATION frame.

    IGD+(A) = (1/|Z|) sum_z min_a sqrt( sum_k max{z_k - a_k, 0}^2 )

    The one-sided ``max{z_k - a_k, 0}`` counts only the objectives on which ``a``
    is genuinely worse than the reference point, which is what makes IGD+ weakly
    Pareto compliant where plain IGD is not. Lower is better; 0.0 means every
    reference point is weakly dominated by something in A.
    """
    A = np.asarray(A_norm, float)
    Z = np.asarray(Z_norm, float)
    if A.size == 0:
        return float("nan")
    total = 0.0
    for lo in range(0, len(Z), 512):
        z = Z[lo:lo + 512][:, None, :]                  # (b, 1, k)
        shortfall = np.maximum(z - A[None, :, :], 0.0)  # (b, |A|, k)
        d = np.sqrt((shortfall ** 2).sum(axis=2))       # (b, |A|)
        total += d.min(axis=1).sum()
    return float(total / len(Z))


def run_igdplus(runs, oracle, out_dir):
    Z = oracle["front_norm"]
    rows = []
    for (method, seed), run in sorted(runs.items()):
        ev = run["evaluated"]
        Y = ev[OBJ].to_numpy(float)
        Y = Y[np.isfinite(Y).all(axis=1)]
        A_all = evaluation.normalize(Y, objective_indices=list(range(len(OBJ))))
        _, A_front = evaluation.compute_pareto_front(A_all, np.ones(len(OBJ)))
        rows.append({
            "method": method, "seed": seed,
            "n_evaluated": int(len(Y)), "n_front": int(len(A_front)),
            "igd_plus_front": igd_plus(A_front, Z),
            "igd_plus_all_evaluated": igd_plus(A_all, Z),
            "hypervolume": float(evaluation.compute_hypervolume(Y)),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "igdplus_per_seed.csv"), index=False)

    summary = (df.groupby("method")[["igd_plus_front", "igd_plus_all_evaluated",
                                     "hypervolume", "n_front"]]
                 .agg(["mean", "std", "count"]))
    summary.to_csv(os.path.join(out_dir, "igdplus_summary.csv"))

    note = (
        "IGD+ is computed in the project's own shared normalized frame -- the\n"
        "same fixed min-max bounds, sign flips and [0,1] clipping that\n"
        "evaluation.compute_hypervolume uses -- with the reference set Z being\n"
        "the {n} molecules of the oracle Pareto front (the front of all {d}\n"
        "unique molecules docked across every run). Lower is better.\n\n"
        "READ THIS BEFORE CITING IT: IGD+ agrees with hypervolume here BY\n"
        "CONSTRUCTION. Both are computed over the same normalized objective\n"
        "vectors, and a set that dominates more of the frame both gains volume\n"
        "and shortens its distance to every reference point. IGD+ is therefore\n"
        "the CONVERGENCE column of the Li & Yao (2019) four-aspect framing -- a\n"
        "second, standard, Pareto-compliant confirmation that MOGP converges\n"
        "closer to the oracle front. It is NOT independent evidence about the\n"
        "coverage gap, and must not be presented as such. The coverage question\n"
        "is answered by the cardinality/spread columns: #Circles, and the\n"
        "scope table -- NOT by the count-vs-volume mismatch as stated in Table\n"
        "7 of the campaign write-up, which compared a union-over-seeds count to\n"
        "a single-run hypervolume. At equal scope that mismatch disappears (see\n"
        "scope_table.csv).\n"
    ).format(n=oracle["n_front"], d=oracle["n_docked"])
    with open(os.path.join(out_dir, "igdplus_NOTE.txt"), "w") as fh:
        fh.write(note)

    print(f"\nIGD+ against the {oracle['n_front']}-molecule oracle front "
          f"(lower is better):")
    for method, g in df.groupby("method"):
        print(f"  {method:<7} front {g['igd_plus_front'].mean():.4f} "
              f"+/- {g['igd_plus_front'].std(ddof=1):.4f}   "
              f"all-evaluated {g['igd_plus_all_evaluated'].mean():.4f} "
              f"+/- {g['igd_plus_all_evaluated'].std(ddof=1):.4f}   (n={len(g)})")
    print("\n" + note)
    return df


# --------------------------------------------------------------------------- #
# 0. Oracle-capture scope table.
# --------------------------------------------------------------------------- #
def run_scope_table(runs, oracle, out_dir):
    """Oracle capture at BOTH scopes and on BOTH axes, for every method.

    Table 7 of the campaign write-up put a single-run quantity ("Oracle HV
    captured, single run", with a s.d.) beside a union-over-seeds quantity
    ("Oracle-front molecules found", without one) and read the difference between
    the columns as a finding. Two of those cells are not measured at the same
    scope, so the comparison does not license the conclusion. This table computes
    all four cells at both scopes so the count-vs-volume claim can be made -- or
    withdrawn -- within a single scope.

    Everything is complete-case: a molecule with no docking score cannot sit on
    a front or carry hypervolume.
    """
    oracle_hv = oracle["hv"]
    fronts = {"normalized_frame": (set(oracle["front_smiles"]), oracle["n_front"]),
              "raw_units_published": (set(oracle["raw_front_smiles"]),
                                      oracle["n_front_raw_units"])}
    rows = []
    for front_label, (Zsm, n_front) in fronts.items():
      for method in sorted({m for m, _ in runs}):
          marm = {s: r for (m, s), r in runs.items() if m == method}

          per_seed_count, per_seed_hv = [], []
          for seed, r in sorted(marm.items()):
              ev = r["evaluated"]
              ev_c = ev[ev["complete"]]
              per_seed_count.append(len(Zsm & set(ev_c["SMILES"])))
              per_seed_hv.append(evaluation.compute_hypervolume(ev_c[OBJ].to_numpy(float)))

          union = pd.concat([r["evaluated"] for r in marm.values()], ignore_index=True)
          union_c = union[union["complete"]].drop_duplicates("SMILES")
          union_count = len(Zsm & set(union_c["SMILES"]))
          union_hv = evaluation.compute_hypervolume(union_c[OBJ].to_numpy(float))

          rows.append({
              "oracle_front": front_label,
              "method": method,
              "n_seeds": len(marm),
              "union_front_molecules_found": union_count,
              "union_share_of_front_by_count": union_count / n_front,
              "union_hypervolume": union_hv,
              "union_share_of_oracle_hv": union_hv / oracle_hv,
              "single_run_front_molecules_found_mean": float(np.mean(per_seed_count)),
              "single_run_front_molecules_found_sd": float(np.std(per_seed_count, ddof=1)),
              "single_run_share_of_front_by_count":
                  float(np.mean(per_seed_count)) / n_front,
              "single_run_hypervolume_mean": float(np.mean(per_seed_hv)),
              "single_run_hypervolume_sd": float(np.std(per_seed_hv, ddof=1)),
              "single_run_share_of_oracle_hv": float(np.mean(per_seed_hv)) / oracle_hv,
              "per_seed_counts": ";".join(str(c) for c in per_seed_count),
          })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "scope_table.csv"), index=False)

    print(f"\nOracle capture by scope and axis (oracle HV {oracle_hv:.4f}, "
          f"complete-case). Fronts: normalized frame "
          f"{oracle['n_front']}, raw units {oracle['n_front_raw_units']}:")
    print(f"  {'front':<20} {'method':<8} {'UNION count':>18} {'UNION hv':>18} "
          f"{'SINGLE-RUN count':>20} {'SINGLE-RUN hv':>20}")
    for _, r in df.iterrows():
        print(f"  {r['oracle_front']:<20} {r['method']:<8} "
              f"{r['union_front_molecules_found']:>6.0f} "
              f"({r['union_share_of_front_by_count']:>5.1%}) "
              f"{r['union_hypervolume']:>8.4f} ({r['union_share_of_oracle_hv']:>5.1%}) "
              f"{r['single_run_front_molecules_found_mean']:>8.1f}"
              f"+/-{r['single_run_front_molecules_found_sd']:<4.1f}"
              f"({r['single_run_share_of_front_by_count']:>5.1%}) "
              f"{r['single_run_hypervolume_mean']:>7.4f}"
              f"+/-{r['single_run_hypervolume_sd']:<6.4f}"
              f"({r['single_run_share_of_oracle_hv']:>5.1%})")
    return df


# --------------------------------------------------------------------------- #
# 1. UMAP diagnostic — the one that decides which Tier-2 fix to run first.
# --------------------------------------------------------------------------- #
def run_umap(runs, oracle, lib_smiles, lib_fps, out_dir, seed_for_panel=None,
             umap_seed=0, neighbors=25, min_dist=0.1):
    """Project the library, overlay the oracle front, and decide the verdict.

    The verdict is NOT read off the picture. The picture shows where things are;
    the decision is made in the ORIGINAL 2048-bit fingerprint space, where
    distances are the ones the Tanimoto kernel and the diversity filter actually
    see. UMAP is a non-metric embedding tuned for local structure, and "looks
    peripheral in 2D" is exactly the kind of claim it is known to distort.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import umap

    fp_index = {s: i for i, s in enumerate(lib_smiles)}
    front_smiles = list(oracle["front_smiles"])

    # Which oracle-front molecules does MOGP actually evaluate? Both readings
    # are computed: a single seed, and the union across all MOGP seeds. The
    # campaign's "247" is reproduced by whichever matches, and we say which.
    mogp_runs = {s: r for (m, s), r in runs.items() if m == "MOGP"}
    if not mogp_runs:
        raise SystemExit("UMAP diagnostic needs the MOGP arm; none loaded.")

    per_seed_evaluated = {s: set(r["evaluated"]["SMILES"]) for s, r in mogp_runs.items()}
    union_evaluated = set().union(*per_seed_evaluated.values())
    per_seed_found = {s: len(set(front_smiles) & ev) for s, ev in per_seed_evaluated.items()}
    union_found = len(set(front_smiles) & union_evaluated)

    if seed_for_panel is None:
        seed_for_panel = min(mogp_runs)
    panel_evaluated = per_seed_evaluated[seed_for_panel]

    found = [s for s in front_smiles if s in union_evaluated]
    missed = [s for s in front_smiles if s not in union_evaluated]

    # Acquisition iteration for the coloured overlay, from the panel seed.
    panel_ev = mogp_runs[seed_for_panel]["evaluated"]
    iter_of = dict(zip(panel_ev["SMILES"], panel_ev["acq_iteration"]))

    # ---------------- the decision, in fingerprint space ---------------- #
    def rows(smiles):
        return np.array([fp_index[s] for s in smiles if s in fp_index], dtype=int)

    r_found, r_missed = rows(found), rows(missed)
    r_sampled = rows(sorted(union_evaluated))

    # (a) How far is each oracle-front molecule from the nearest molecule MOGP
    #     actually evaluated? Peripheral => far. Inside a sampled region => near.
    sim_missed = tanimoto_matrix(lib_fps[r_missed], lib_fps[r_sampled])
    sim_found = tanimoto_matrix(lib_fps[r_found], lib_fps[r_sampled])
    # A found molecule is its own nearest neighbour at similarity 1.0; drop that
    # self-match, or the two groups are not measuring the same thing.
    d_missed = 1.0 - sim_missed.max(axis=1)
    sim_found_nonself = np.where(sim_found > 0.999999, -1.0, sim_found)
    d_found = 1.0 - sim_found_nonself.max(axis=1)

    # (b) How densely did MOGP sample each molecule's neighbourhood?
    RADIUS = 0.4                      # Tanimoto distance; sim >= 0.6
    dens_missed = (sim_missed >= 1.0 - RADIUS).sum(axis=1)
    dens_found = (sim_found_nonself >= 1.0 - RADIUS).sum(axis=1)

    # (c) Is the missed molecule in a sparse part of the LIBRARY, independent of
    #     what MOGP did? This separates "peripheral chemistry" from "MOGP just
    #     did not go there". Distance to the 10th nearest library neighbour.
    def library_sparsity(r, k=10):
        sim = tanimoto_matrix(lib_fps[r], lib_fps)
        sim[np.arange(len(r)), r] = -1.0            # drop self
        kth = np.partition(sim, -k, axis=1)[:, -k]
        return 1.0 - kth

    sparse_missed = library_sparsity(r_missed)
    sparse_found = library_sparsity(r_found)

    def describe(x):
        """Summary that survives an empty group (MOGP missing nothing is a
        legitimate outcome, not a crash)."""
        x = np.asarray(x, float)
        if x.size == 0:
            return {"n": 0, "mean": None, "median": None, "p90": None}
        return {"n": int(x.size), "mean": float(x.mean()),
                "median": float(np.median(x)), "p90": float(np.percentile(x, 90))}

    from scipy.stats import mannwhitneyu
    def compare(a, b):
        try:
            u = mannwhitneyu(a, b, alternative="two-sided")
            return float(u.pvalue)
        except ValueError:
            return float("nan")

    # A missed molecule is "inside a densely sampled region" when MOGP evaluated
    # a close analogue of it -- it saw that chemistry and passed.
    # The 0.4 cut is a judgement call, so the verdict is swept across a range of
    # defensible thresholds and reported at every one. A conclusion that survives
    # 0.3 through 0.5 is a conclusion about the data; one that flips inside that
    # range is a conclusion about the threshold, and must be reported as such.
    SWEEP = (0.3, 0.4, 0.5)
    INSIDE_D = 0.4                         # the headline cut
    sweep = {}
    for t in SWEEP:
        fi = float((d_missed <= t).mean()) if d_missed.size else float("nan")
        ff = float((d_found <= t).mean()) if d_found.size else float("nan")
        sweep[f"{t:.1f}"] = {
            "fraction_missed_inside": fi,
            "fraction_found_inside": ff,
            "n_missed_inside": int((d_missed <= t).sum()),
            "verdict": ("no_gap" if not d_missed.size
                        else "acquisition_rule" if fi >= 0.5 else "pool_sampling"),
        }
    frac_inside = sweep[f"{INSIDE_D:.1f}"]["fraction_missed_inside"]
    frac_found_inside = sweep[f"{INSIDE_D:.1f}"]["fraction_found_inside"]
    verdict = sweep[f"{INSIDE_D:.1f}"]["verdict"]
    verdicts = {t: s["verdict"] for t, s in sweep.items()}
    verdict_is_stable = len(set(verdicts.values())) == 1

    stats = {
        "oracle_front_size": oracle["n_front"],
        "oracle_docked_pool": oracle["n_docked"],
        "oracle_hypervolume": oracle["hv"],
        "mogp_found_union_all_seeds": union_found,
        "mogp_found_per_seed": {int(k): int(v) for k, v in sorted(per_seed_found.items())},
        "mogp_found_per_seed_mean": float(np.mean(list(per_seed_found.values()))),
        "mogp_missed_union_all_seeds": oracle["n_front"] - union_found,
        "panel_seed": int(seed_for_panel),
        "library_rows_on_disk": int(len(lib_smiles)),
        "distance_to_nearest_mogp_evaluated": {
            "missed": describe(d_missed), "found": describe(d_found),
            "mannwhitney_p": compare(d_missed, d_found)},
        f"n_mogp_evaluated_within_tanimoto_distance_{RADIUS}": {
            "missed": describe(dens_missed), "found": describe(dens_found),
            "mannwhitney_p": compare(dens_missed, dens_found)},
        "library_sparsity_distance_to_10th_neighbour": {
            "missed": describe(sparse_missed), "found": describe(sparse_found),
            "mannwhitney_p": compare(sparse_missed, sparse_found)},
        "fraction_missed_within_0.4_of_something_mogp_evaluated": frac_inside,
        "fraction_found_within_0.4_of_another_evaluated": frac_found_inside,
        "threshold_sweep": sweep,
        "verdict_stable_across_0.3_to_0.5": verdict_is_stable,
        "VERDICT": verdict,
        "verdict_rule": (
            "A missed oracle-front molecule counts as INSIDE a sampled region "
            "when MOGP evaluated some molecule within Tanimoto distance 0.4 of "
            "it (similarity >= 0.6) -- i.e. the loop saw that chemistry and did "
            "not pick this member. >= 50% inside => the acquisition rule is "
            "what to fix; < 50% => the molecules are chemistry MOGP's pool "
            "never showed it, and the fix is cluster-stratified pool sampling."),
    }

    # ------------------------------- figure ------------------------------- #
    print(f"  fitting UMAP on {len(lib_fps)} x {lib_fps.shape[1]} bits "
          f"(jaccard metric) ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reducer = umap.UMAP(n_neighbors=neighbors, min_dist=min_dist,
                            metric="jaccard", random_state=umap_seed, verbose=False)
        emb = reducer.fit_transform(lib_fps)
    np.save(os.path.join(out_dir, "umap_embedding.npy"), emb)

    fig, axes = plt.subplots(1, 3, figsize=(19, 6.2))
    bg = dict(s=1.2, c="#d9d9d9", linewidths=0, rasterized=True)

    ax = axes[0]
    ax.scatter(emb[:, 0], emb[:, 1], **bg)
    ax.scatter(emb[rows(front_smiles), 0], emb[rows(front_smiles), 1],
               s=13, c="#1a1a1a", linewidths=0, label=f"oracle front (n={len(front_smiles)})")
    ax.set_title(f"Library ({len(lib_smiles):,}) and the oracle front")

    ax = axes[1]
    ax.scatter(emb[:, 0], emb[:, 1], **bg)
    r_panel = rows(sorted(panel_evaluated))
    ax.scatter(emb[r_panel, 0], emb[r_panel, 1], s=6, c="#8fbfe0",
               linewidths=0, label=f"MOGP seed {seed_for_panel} evaluated (n={len(r_panel)})")
    ax.scatter(emb[r_found, 0], emb[r_found, 1], s=16, c="#1f6f3f", linewidths=0,
               label=f"oracle front FOUND (n={len(r_found)})")
    ax.scatter(emb[r_missed, 0], emb[r_missed, 1], s=26, c="#c0392b", marker="X",
               linewidths=0, label=f"oracle front MISSED (n={len(r_missed)})")
    ax.set_title("What MOGP finds vs what it misses")

    ax = axes[2]
    ax.scatter(emb[:, 0], emb[:, 1], **bg)
    it_vals = np.array([iter_of.get(s, np.nan) for s in sorted(panel_evaluated)], float)
    ok = np.isfinite(it_vals)
    sc = ax.scatter(emb[r_panel[ok], 0], emb[r_panel[ok], 1], s=11, c=it_vals[ok],
                    cmap="viridis", linewidths=0)
    fig.colorbar(sc, ax=ax, label="acquisition iteration")
    ax.scatter(emb[r_missed, 0], emb[r_missed, 1], s=26, c="#c0392b", marker="X",
               linewidths=0, label="never evaluated")
    ax.set_title(f"MOGP seed {seed_for_panel}, by acquisition iteration")

    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        ax.legend(loc="best", fontsize=8, framealpha=0.9, markerscale=1.6)
    headline = {"acquisition_rule": "INSIDE densely sampled regions",
                "pool_sampling": "SPARSE peripheral regions",
                "no_gap": "no missed oracle-front molecules"}[verdict]
    detail = ("" if verdict == "no_gap" else
              f"  ({frac_inside:.0%} of missed molecules have an evaluated "
              f"analogue within Tanimoto distance 0.4)")
    fig.suptitle(f"Oracle-front coverage in chemical space  |  verdict: "
                 f"{headline}{detail}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(os.path.join(out_dir, "umap_oracle_coverage.png"), dpi=180)
    plt.close(fig)

    with open(os.path.join(out_dir, "umap_verdict.json"), "w") as fh:
        json.dump(stats, fh, indent=2)

    pd.DataFrame({
        "SMILES": missed,
        "distance_to_nearest_mogp_evaluated": d_missed,
        f"n_mogp_evaluated_within_{RADIUS}": dens_missed,
        "library_sparsity_10th_nn_distance": sparse_missed,
        "inside_sampled_region": d_missed <= INSIDE_D,
    }).sort_values("distance_to_nearest_mogp_evaluated").to_csv(
        os.path.join(out_dir, "umap_missed_molecules.csv"), index=False)

    print(f"\nUMAP diagnostic: oracle front {oracle['n_front']}, MOGP found "
          f"{union_found} (union over {len(mogp_runs)} seeds), missed "
          f"{oracle['n_front'] - union_found}")
    def med(x):
        return f"{np.median(x):.3f}" if np.asarray(x).size else "n/a"
    print(f"  distance to nearest MOGP-evaluated molecule (Tanimoto):")
    print(f"    missed  median {med(d_missed)}   found  median {med(d_found)}")
    print(f"  library sparsity (distance to 10th library neighbour):")
    print(f"    missed  median {med(sparse_missed)}   found  median {med(sparse_found)}")
    print("  threshold sweep (fraction of missed molecules with an "
          "MOGP-evaluated analogue within t):")
    for t, s in sweep.items():
        print(f"    t={t}   missed inside {s['fraction_missed_inside']:6.1%} "
              f"({s['n_missed_inside']}/{len(d_missed)})   "
              f"found inside {s['fraction_found_inside']:6.1%}   "
              f"-> {s['verdict']}")
    print(f"  VERDICT: {verdict}  "
          f"({'stable' if verdict_is_stable else 'NOT STABLE'} across 0.3-0.5)")
    return stats


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def resolve_out_dir(path):
    real = os.path.realpath(path)
    parts = set(real.split(os.sep))
    clash = parts & set(PROTECTED)
    if clash:
        raise SystemExit(
            f"Refusing to write inside the campaign record ({sorted(clash)}). "
            "Tier 1 is read-only over the campaign; choose an --out outside "
            "aggregate_10seed/ and the seed directories.")
    os.makedirs(real, exist_ok=True)
    return real


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("analysis",
                   choices=["umap", "circles", "igdplus", "scope", "all"])
    p.add_argument("--campaign-root", default="campaign_results")
    p.add_argument("--out",
                   default="campaign_results/aggregate_10seed_cleanGPMOBO/tier1")
    p.add_argument("--library-dir", default="data/library")
    p.add_argument("--seeds", default="0-9",
                   help="Seed list or range, e.g. '0-9' or '0,1,2'.")
    p.add_argument("--methods", default="MOGP,GPMOBO,Greedy")
    p.add_argument("--panel-seed", type=int, default=None,
                   help="Which MOGP seed the per-iteration UMAP panel shows.")
    p.add_argument("--umap-seed", type=int, default=0)
    p.add_argument("--circles-thresholds", default="0.60,0.75")
    args = p.parse_args(argv)

    if "-" in args.seeds:
        lo, hi = args.seeds.split("-")
        seeds = list(range(int(lo), int(hi) + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    thresholds = tuple(float(t) for t in args.circles_thresholds.split(","))

    out_dir = resolve_out_dir(args.out)
    print(f"campaign root : {os.path.realpath(args.campaign_root)}")
    print(f"output        : {out_dir}")

    runs = load_campaign(args.campaign_root, seeds, methods)
    print(f"loaded {len(runs)} runs: " + ", ".join(
        f"{m}x{sum(1 for mm, _ in runs if mm == m)}" for m in methods))

    oracle = build_oracle(runs)

    # State the NaN handling with the numbers it produced, in the output and on
    # disk. ~2.9% of rows in every arm are failed docks (NaN in BOTH docking
    # columns, never in ADMET). Whether the oracle was built over all rows or
    # complete rows changes its size, and every capture percentage divides by it.
    provenance = {
        "nan_policy": NAN_POLICY,
        "objective_space_sets": "complete-case (all 5 objectives finite)",
        "coverage_sets": "all evaluated rows (budget spent, chemistry visited)",
        "pooled_rows_all": oracle["n_rows_all"],
        "pooled_rows_dropped_incomplete": oracle["n_rows_incomplete"],
        "unique_smiles_over_all_rows": oracle["n_unique_smiles_all_rows"],
        "unique_smiles_over_complete_rows": oracle["n_unique_smiles_complete"],
        "oracle_front_size_normalized_frame": oracle["n_front"],
        "oracle_front_size_raw_units_published": oracle["n_front_raw_units"],
        "rows_saturating_a_normalization_bound": oracle["n_rows_saturating_a_bound"],
        "oracle_hypervolume": oracle["hv"],
        "front_frame_note": (
            "normalize() clips to [0,1], so values outside the fixed bounds "
            "saturate and the ties that creates make some raw-front members "
            "weakly dominated. The campaign's published 411 is the RAW front; "
            "hypervolume and IGD+ both live in the normalized frame, so IGD+ "
            "uses the normalized front as its reference set Z. Capture is "
            "reported against both in scope_table.csv."),
        "per_arm": {f"{m}_seed{s}": {
            "rows": int(len(r["evaluated"])),
            "incomplete": int((~r["evaluated"]["complete"]).sum()),
            "reported_only_columns": r["schema_extra"],
        } for (m, s), r in sorted(runs.items())},
    }
    with open(os.path.join(out_dir, "nan_policy_and_oracle_provenance.json"), "w") as fh:
        json.dump(provenance, fh, indent=2)

    print(f"NaN policy: {NAN_POLICY}")
    print(f"  pooled rows {oracle['n_rows_all']}, dropped as incomplete "
          f"{oracle['n_rows_incomplete']} "
          f"({oracle['n_rows_incomplete'] / max(oracle['n_rows_all'], 1):.2%})")
    print(f"  unique SMILES: {oracle['n_unique_smiles_all_rows']} over all rows, "
          f"{oracle['n_unique_smiles_complete']} over complete rows")
    print(f"oracle: {oracle['n_docked']} unique docked molecules -> "
          f"{oracle['n_front']}-molecule front, HV {oracle['hv']:.4f}")
    if oracle["inconsistent_duplicate_rows"]:
        print(f"  ! {oracle['inconsistent_duplicate_rows']} duplicate rows "
              f"disagree on objective values across runs", file=sys.stderr)
    pd.DataFrame({"SMILES": oracle["front_smiles"]}).assign(
        **{c: oracle["docked"][c].to_numpy()[oracle["front_idx"]] for c in OBJ}
    ).to_csv(os.path.join(out_dir, "oracle_front.csv"), index=False)

    lib_smiles, lib_fps = load_library_fingerprints(args.library_dir)
    print(f"library: {len(lib_smiles)} rows on disk")

    if args.analysis in ("scope", "all"):
        run_scope_table(runs, oracle, out_dir)
    if args.analysis in ("umap", "all"):
        run_umap(runs, oracle, lib_smiles, lib_fps, out_dir,
                 seed_for_panel=args.panel_seed, umap_seed=args.umap_seed)
    if args.analysis in ("circles", "all"):
        run_circles(runs, oracle, lib_smiles, lib_fps, out_dir, thresholds)
    if args.analysis in ("igdplus", "all"):
        run_igdplus(runs, oracle, out_dir)

    print(f"\nWrote Tier 1 outputs to {out_dir}")


if __name__ == "__main__":
    main()
