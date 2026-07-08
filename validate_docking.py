"""
validate_docking.py
===================

Standalone diagnostic: is the PfDHFR docking objective measuring real binding, or
just molecular size / lipophilicity?

AutoDock Vina's scoring function sums per-atom interaction terms, so its scores
drift more negative ("better") for bigger, greasier molecules almost regardless
of complementarity. If our docking objective is dominated by that artifact, then
the whole multi-objective BO pipeline is optimizing molecular weight and logP in
disguise. This script answers the question with five checks and one figure, so a
reader can decide in ~30 seconds whether the docking signal is trustworthy:

  1. Per-molecule MW, logP (Crippen) and heavy-atom count from SMILES (RDKit).
  2. A PfDHFR docking-score sample, preferring molecules already docked (existing
     run outputs, then the persistent docking cache) and only docking fresh to
     top up to --n-sample. The provenance breakdown is printed.
  3. Pearson AND Spearman correlation of docking score vs MW and vs logP, with a
     plain-language verdict (|r| > ~0.5 -> the objective is partly a size/lipo
     artifact, flagged explicitly).
  4. Ligand efficiency (docking score / heavy-atom count) — the standard
     size-debiased view — and how much re-ranking by LE changes the top molecules
     vs the raw score (a remedy, not just a diagnosis).
  5. The four KNOWN_ACTIVES (clinical antifolates, reused from
     validate_known_actives) docked against PfDHFR, each reported as a percentile
     within the library docking distribution: do real drugs land among the good
     scores, or do library "winners" dock dramatically better than real drugs?

It imports the pipeline's own docking / library code but MODIFIES NOTHING; it is
a read-only observer that uses the docking cache like every other entry point.

Run ``python validate_docking.py --help`` for options. Docking is expensive, so
the first run on a cold cache will take a while for the fresh top-up docks; every
subsequent run is nearly instant (cache hits).
"""

import os
import glob
import argparse

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen

import docking
from docking_cache import canonicalize_smiles
from data import load_library
from validate_known_actives import KNOWN_ACTIVES


# The docking objective column as written by the pipeline (mogp.TASK_NAMES).
PFDHFR_COLUMN = "PfDHFR_Docking"
DOCK_TARGET = "PfDHFR"

# Default directories that may hold a method's evaluated.csv (loop + baselines +
# ablation arms). Any of these that exist are scanned for already-docked scores;
# so is any other ``*results*`` directory found next to this script.
DEFAULT_RESULTS_DIRS = [
    "results",
    "results_coregionalized",
    "baseline_random_results",
    "baseline_single_obj_results",
    "baseline_greedy_results",
]

# Correlation strength thresholds for the plain-language verdict.
CORR_STRONG = 0.5
CORR_MODERATE = 0.3


# ---------------------------------------------------------------------- #
# RDKit descriptors
# ---------------------------------------------------------------------- #
def compute_descriptors(smiles):
    """Return ``(MW, logP, heavy_atom_count)`` for a SMILES, or None if unparseable.

    MW is ``Descriptors.MolWt``, logP is Crippen ``MolLogP`` — the same
    definitions a medicinal chemist would quote — and the heavy-atom count is the
    denominator for ligand efficiency.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return (
        float(Descriptors.MolWt(mol)),
        float(Crippen.MolLogP(mol)),
        int(mol.GetNumHeavyAtoms()),
    )


# ---------------------------------------------------------------------- #
# Assemble a PfDHFR docking-score sample (outputs -> cache -> fresh)
# ---------------------------------------------------------------------- #
def _discover_results_dirs():
    """Existing result dirs to scan: the known defaults plus any ``*results*``."""
    dirs = list(DEFAULT_RESULTS_DIRS)
    dirs += [d for d in glob.glob("*results*") if os.path.isdir(d)]
    # De-dupe, preserve order, keep only those that exist.
    seen, out = set(), []
    for d in dirs:
        if d in seen or not os.path.isdir(d):
            continue
        seen.add(d)
        out.append(d)
    return out


def scores_from_outputs(canon_to_index):
    """Collect PfDHFR scores from existing ``evaluated.csv`` run outputs.

    Only rows whose (canonicalized) SMILES is in the current library are kept —
    matched by canonical SMILES to a library index — so every collected score
    aligns with a molecule we can compute descriptors for.

    Returns ``(index_to_score, source_dirs)``: a dict library-index -> score, and
    the list of directories that actually contributed.
    """
    index_to_score = {}
    source_dirs = []
    for directory in _discover_results_dirs():
        path = os.path.join(directory, "evaluated.csv")
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception:                                        # pragma: no cover
            continue
        if "SMILES" not in df.columns or PFDHFR_COLUMN not in df.columns:
            continue
        contributed = False
        for smiles, score in zip(df["SMILES"].astype(str), df[PFDHFR_COLUMN]):
            if not np.isfinite(score):
                continue
            idx = canon_to_index.get(canonicalize_smiles(smiles))
            if idx is None or idx in index_to_score:
                continue
            index_to_score[idx] = float(score)
            contributed = True
        if contributed:
            source_dirs.append(directory)
    return index_to_score, source_dirs


def scores_from_cache(library_smiles, canon_list, skip_indices):
    """Look up already-cached PfDHFR docks for library molecules not already scored.

    Consults the persistent docking cache directly (read-only) so molecules docked
    by any previous run are reused without re-docking. Returns a dict
    library-index -> score for cache HITS (finite affinity) only.
    """
    cache = docking.get_cache()
    index_to_score = {}
    for idx, canon in enumerate(canon_list):
        if idx in skip_indices:
            continue
        cached = cache.get(canon, DOCK_TARGET)
        if cached is None:
            continue
        status, affinity = cached
        if status == docking.STATUS_OK and affinity is not None and np.isfinite(affinity):
            index_to_score[idx] = float(affinity)
    return index_to_score


def dock_fresh(library_smiles, indices):
    """Freshly dock the given library indices against PfDHFR (cache ON).

    ``dock_target`` writes each result to the shared cache, so a later run reuses
    them. Returns a dict library-index -> score for the docks that succeeded.
    """
    index_to_score = {}
    n = len(indices)
    for i, idx in enumerate(indices, start=1):
        print(f"  fresh dock {i}/{n} (library #{idx}) against {DOCK_TARGET}...",
              flush=True)
        score = docking.dock_target(library_smiles[idx], target=DOCK_TARGET,
                                    use_cache=True)
        if score is not None and np.isfinite(score):
            index_to_score[idx] = float(score)
    return index_to_score


def build_sample(library, n_sample, seed):
    """Build the docking-score sample, preferring cheap sources over fresh docks.

    Order of preference (all avoid re-docking except the last):
        existing run outputs  ->  docking cache  ->  fresh docks to top up.

    Returns ``(sample_df, provenance)`` where ``sample_df`` has columns
    ``[library_index, SMILES, docking, MW, logP, heavy_atoms]`` (rows with an
    unparseable SMILES dropped), and ``provenance`` is a dict of counts.
    """
    smiles = library["smiles"]
    n_lib = len(smiles)

    # Canonical SMILES for the whole library, once, so outputs/cache match by
    # canonical identity (the same key the docking cache uses).
    canon_list = [canonicalize_smiles(s) for s in smiles]
    canon_to_index = {}
    for idx, canon in enumerate(canon_list):
        canon_to_index.setdefault(canon, idx)

    # 1) Existing run outputs.
    out_scores, source_dirs = scores_from_outputs(canon_to_index)
    # 2) Docking cache (for molecules not already covered by outputs).
    cache_scores = scores_from_cache(smiles, canon_list, skip_indices=set(out_scores))

    scores = dict(out_scores)
    scores.update(cache_scores)
    n_outputs = len(out_scores)
    n_cache = len(cache_scores)

    # 3) Fresh docks only if the free scores fall short of the requested sample.
    n_fresh = 0
    if len(scores) < n_sample:
        rng = np.random.default_rng(seed)
        remaining = [i for i in range(n_lib) if i not in scores]
        rng.shuffle(remaining)
        need = n_sample - len(scores)
        to_dock = remaining[:need]
        if to_dock:
            print(f"\nFree scores ({len(scores)}) < --n-sample ({n_sample}); "
                  f"docking {len(to_dock)} fresh molecule(s) against {DOCK_TARGET} "
                  "(cache ON)...")
            fresh = dock_fresh(smiles, to_dock)
            scores.update(fresh)
            n_fresh = len(fresh)

    # Assemble the sample table with descriptors.
    rows = []
    for idx, score in scores.items():
        desc = compute_descriptors(smiles[idx])
        if desc is None:
            continue
        mw, logp, hac = desc
        rows.append((idx, smiles[idx], score, mw, logp, hac))
    sample_df = pd.DataFrame(
        rows, columns=["library_index", "SMILES", "docking", "MW", "logP", "heavy_atoms"]
    )

    provenance = {
        "n_outputs": n_outputs,
        "n_cache": n_cache,
        "n_fresh": n_fresh,
        "n_total": len(sample_df),
        "source_dirs": source_dirs,
    }
    return sample_df, provenance


# ---------------------------------------------------------------------- #
# Correlations + verdict
# ---------------------------------------------------------------------- #
def _strength_word(r):
    """Plain-language strength for a correlation coefficient's magnitude."""
    a = abs(r)
    if a > CORR_STRONG:
        return "STRONG"
    if a > CORR_MODERATE:
        return "moderate"
    return "weak"


def report_correlations(sample_df):
    """Print Pearson + Spearman of docking vs MW and vs logP, with a verdict.

    Returns ``{"MW": (pearson, spearman), "logP": (pearson, spearman)}`` and a
    boolean ``flagged`` (any |r| > CORR_STRONG), so the caller can fold it into
    the final summary.
    """
    print("\n" + "=" * 78)
    print("3. DOCKING vs SIZE / LIPOPHILICITY  (Pearson + Spearman)")
    print("=" * 78)

    y = sample_df["docking"].to_numpy()
    n = len(y)
    if n < 3:
        print(f"  Only {n} scored molecule(s); need >=3 for a correlation. "
              "Increase --n-sample or dock more.")
        return {}, False

    print(f"  Docking score = PfDHFR affinity (kcal/mol, more negative = "
          f"stronger binding).  n = {n}.")
    print(f"  A negative r means BIGGER / GREASIER molecules get 'better' "
          "(more negative) scores.\n")

    out = {}
    flagged = False
    for col in ("MW", "logP"):
        x = sample_df[col].to_numpy()
        pear = float(pearsonr(x, y)[0])
        spear = float(spearmanr(x, y)[0])
        out[col] = (pear, spear)
        strong = abs(pear) > CORR_STRONG or abs(spear) > CORR_STRONG
        flagged = flagged or strong
        label = "docking vs " + col
        print(f"  {label:<16} Pearson r = {pear:+.3f} ({_strength_word(pear)}), "
              f"Spearman rho = {spear:+.3f} ({_strength_word(spear)})")
        if strong:
            print(f"      -> FLAG: the docking objective is PARTLY measuring "
                  f"{col} (|r| > {CORR_STRONG}), not binding alone.")
        else:
            print(f"      -> ok: no strong {col} dependence (|r| <= {CORR_STRONG}).")
    return out, flagged


# ---------------------------------------------------------------------- #
# Ligand efficiency (size-debiased view)
# ---------------------------------------------------------------------- #
def report_ligand_efficiency(sample_df, top_k):
    """Compare the top molecules ranked by raw score vs by ligand efficiency.

    LE = docking score / heavy-atom count (both negative here, so more negative =
    more binding per atom). If LE re-ranks the leaders heavily, the raw score was
    rewarding size; LE is the standard size-debiased remedy.

    Returns the overlap fraction of the two top-``top_k`` sets.
    """
    print("\n" + "=" * 78)
    print(f"4. LIGAND EFFICIENCY (size-debiased):  LE = docking / heavy_atoms")
    print("=" * 78)

    df = sample_df.copy()
    df["LE"] = df["docking"] / df["heavy_atoms"]

    k = min(top_k, len(df))
    # More negative = better for both raw score and LE.
    by_score = df.sort_values("docking").head(k)
    by_le = df.sort_values("LE").head(k)

    score_set = set(by_score["library_index"])
    le_set = set(by_le["library_index"])
    overlap = score_set & le_set
    overlap_frac = len(overlap) / k if k else 0.0
    dropped = score_set - le_set

    print(f"  Top {k} by RAW docking score vs top {k} by LIGAND EFFICIENCY:")
    print(f"    shared molecules:            {len(overlap)}/{k} "
          f"({overlap_frac * 100:.0f}%)")
    print(f"    raw-score leaders LE demotes: {len(dropped)}/{k}")

    corr_le_mw = (float(spearmanr(df["MW"], df["LE"])[0])
                  if len(df) >= 3 else float("nan"))
    print(f"    Spearman(LE, MW) = {corr_le_mw:+.3f} "
          "(closer to 0 = LE has removed more of the size trend)")

    # Show the raw-score leaders that LE pushes out of the top-k, with their MW —
    # these are the "big molecule wins the raw score" cases LE corrects.
    if dropped:
        show = by_score[by_score["library_index"].isin(dropped)] \
            .sort_values("docking")
        print(f"\n  Raw-score leaders demoted by LE (likely size-driven):")
        print(f"    {'lib#':>6}{'docking':>10}{'MW':>9}{'heavy':>7}{'LE':>9}")
        for _, r in show.head(8).iterrows():
            print(f"    {int(r['library_index']):>6}{r['docking']:>10.2f}"
                  f"{r['MW']:>9.1f}{int(r['heavy_atoms']):>7}{r['LE']:>9.3f}")

    if overlap_frac < 0.5:
        print(f"\n  -> LE substantially re-ranks the leaders "
              f"({overlap_frac * 100:.0f}% overlap): raw docking was rewarding "
              "SIZE. Prefer LE (or add a size penalty) downstream.")
    else:
        print(f"\n  -> LE largely agrees with the raw ranking "
              f"({overlap_frac * 100:.0f}% overlap): the leaders are not purely "
              "size-driven.")
    return overlap_frac


# ---------------------------------------------------------------------- #
# Known actives vs the library distribution
# ---------------------------------------------------------------------- #
def _percentile_better_than(score, distribution):
    """Percent of ``distribution`` this score binds MORE STRONGLY than.

    Docking is more-negative-is-better, so this is the fraction of the library
    with a WEAKER (higher) score — i.e. "this molecule out-docks X% of the
    library".
    """
    dist = np.asarray(distribution, dtype=float)
    if dist.size == 0:
        return float("nan")
    return float(np.mean(score < dist) * 100.0)


def report_known_actives(sample_df):
    """Dock the KNOWN_ACTIVES against PfDHFR; report each one's library percentile.

    Returns ``(active_rows, best_library_score)`` for the final summary.
    """
    print("\n" + "=" * 78)
    print("5. KNOWN CLINICAL ANTIFOLATES vs THE LIBRARY DOCKING DISTRIBUTION")
    print("=" * 78)

    dist = sample_df["docking"].to_numpy()
    best_lib = float(np.min(dist)) if dist.size else float("nan")
    median_lib = float(np.median(dist)) if dist.size else float("nan")
    print(f"  Library docking sample (n={dist.size}): "
          f"best {best_lib:.2f}, median {median_lib:.2f} kcal/mol.")
    print("  Percentile = % of the library this drug out-docks "
          "(higher = stronger binder).\n")

    header = (f"  {'compound':<15}{'PfDHFR':>9}{'MW':>8}{'logP':>7}"
              f"{'heavy':>7}{'LE':>8}{'pctile':>9}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    active_rows = []
    for a in KNOWN_ACTIVES:
        desc = compute_descriptors(a["smiles"])
        score = docking.dock_target(a["smiles"], target=DOCK_TARGET, use_cache=True)
        if desc is None or score is None or not np.isfinite(score):
            print(f"  {a['name']:<15}{'  dock/parse failed':>40}")
            active_rows.append({"name": a["name"], "score": float("nan"),
                                "MW": float("nan"), "logP": float("nan"),
                                "heavy": float("nan"), "LE": float("nan"),
                                "percentile": float("nan")})
            continue
        mw, logp, hac = desc
        le = score / hac
        pct = _percentile_better_than(score, dist)
        active_rows.append({"name": a["name"], "score": float(score), "MW": mw,
                            "logP": logp, "heavy": hac, "LE": le,
                            "percentile": pct})
        print(f"  {a['name']:<15}{score:>9.2f}{mw:>8.1f}{logp:>7.2f}"
              f"{hac:>7}{le:>8.3f}{pct:>8.0f}%")

    finite = [r for r in active_rows if np.isfinite(r["percentile"])]
    if finite:
        med_pct = float(np.median([r["percentile"] for r in finite]))
        print(f"\n  Median known-active percentile: {med_pct:.0f}%.")
        if np.isfinite(best_lib):
            best_active = min(r["score"] for r in finite)
            gap = best_active - best_lib   # positive => library out-docks best drug
            print(f"  Best library score {best_lib:.2f} vs best known drug "
                  f"{best_active:.2f}  (library is {gap:+.2f} kcal/mol "
                  f"{'stronger' if gap < 0 else 'weaker'}).")
    return active_rows, best_lib


# ---------------------------------------------------------------------- #
# Figure
# ---------------------------------------------------------------------- #
def save_figure(sample_df, active_rows, output_path):
    """Save docking-vs-MW and docking-vs-logP scatter with known actives overlaid."""
    import matplotlib
    matplotlib.use("Agg")            # headless: write a file, never open a window
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    y = sample_df["docking"].to_numpy()

    active_ok = [r for r in active_rows if np.isfinite(r["score"])]

    for ax, col, xlabel in ((axes[0], "MW", "Molecular weight (Da)"),
                            (axes[1], "logP", "Crippen logP")):
        ax.scatter(sample_df[col], y, s=18, c="lightsteelblue",
                   edgecolors="none", alpha=0.7, label="Library sample")
        # Known actives overlaid as distinct labeled markers.
        for r in active_ok:
            ax.scatter(r[col], r["score"], marker="*", s=260, c="crimson",
                       edgecolors="black", linewidths=0.6, zorder=5)
            ax.annotate(r["name"], (r[col], r["score"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
        if active_ok:
            # A single proxy legend entry for the star markers.
            ax.scatter([], [], marker="*", s=160, c="crimson",
                       edgecolors="black", label="Known actives")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("PfDHFR docking (kcal/mol)  ↓ = stronger binding")
        ax.invert_yaxis()            # stronger (more negative) binders at the top
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle("Is the PfDHFR docking objective tracking binding, or size/lipophilicity?",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved scatter figure to {output_path}")


# ---------------------------------------------------------------------- #
# Entry point
# ---------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Diagnose whether the PfDHFR docking objective tracks real "
                    "binding or a molecular-size / lipophilicity artifact."
    )
    parser.add_argument("--library-dir", default="data/library",
                        help="Cached library directory (default data/library).")
    parser.add_argument("--n-sample", type=int, default=150,
                        help="Target docking-score sample size (default 150). "
                             "Existing outputs + cache are reused first; only the "
                             "shortfall is docked fresh.")
    parser.add_argument("--top-k", type=int, default=15,
                        help="Top-K set size for the LE re-ranking comparison.")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for the random fresh-dock top-up sample.")
    parser.add_argument("--output", default="validate_docking_scatter.png",
                        help="Path for the saved scatter figure.")
    args = parser.parse_args()

    print("=" * 78)
    print("DOCKING OBJECTIVE VALIDATION — size / lipophilicity artifact check")
    print("=" * 78)

    library = load_library(args.library_dir)
    print(f"Loaded library: {len(library['smiles'])} molecules from "
          f"{args.library_dir}.")

    # --- Build the docking-score sample (outputs -> cache -> fresh) ---
    print("\n" + "=" * 78)
    print("2. ASSEMBLING PfDHFR DOCKING SAMPLE")
    print("=" * 78)
    sample_df, prov = build_sample(library, args.n_sample, args.seed)
    print(f"\n  Docking-score sample: {prov['n_total']} molecule(s) with "
          "descriptors —")
    print(f"    {prov['n_outputs']:>5} reused from existing run outputs"
          + (f" ({', '.join(prov['source_dirs'])})" if prov["source_dirs"] else "")
          )
    print(f"    {prov['n_cache']:>5} reused from the docking cache")
    print(f"    {prov['n_fresh']:>5} freshly docked this run (written to the cache)")

    if prov["n_total"] < 3:
        print("\n  Too few docking scores to analyze. Re-run with a larger "
              "--n-sample once docking is available (vina on PATH).")
        return

    # --- Correlations, LE, known actives ---
    corrs, flagged = report_correlations(sample_df)
    overlap_frac = report_ligand_efficiency(sample_df, args.top_k)
    active_rows, best_lib = report_known_actives(sample_df)

    # --- Figure ---
    save_figure(sample_df, active_rows, args.output)

    # --- 30-second verdict ---
    print("\n" + "=" * 78)
    print("SUMMARY VERDICT (read this first)")
    print("=" * 78)
    if corrs:
        mw_r = corrs["MW"][0]
        logp_r = corrs["logP"][0]
        print(f"  - docking vs MW:   Pearson {mw_r:+.2f} ({_strength_word(mw_r)})")
        print(f"  - docking vs logP: Pearson {logp_r:+.2f} ({_strength_word(logp_r)})")
    print(f"  - LE re-ranking:   top-{args.top_k} overlap "
          f"{overlap_frac * 100:.0f}% (low overlap => raw score was size-driven)")
    finite_pct = [r["percentile"] for r in active_rows if np.isfinite(r["percentile"])]
    if finite_pct:
        print(f"  - known drugs:     median percentile {np.median(finite_pct):.0f}% "
              "of the library (low => library 'winners' out-dock real drugs)")

    trustworthy = (not flagged) and overlap_frac >= 0.5
    if flagged:
        print("\n  VERDICT: CAUTION — docking correlates strongly with size/lipophilicity.")
        print("  Treat raw PfDHFR docking as partly an artifact; use ligand")
        print("  efficiency (or a MW/logP penalty) before trusting a 'better' score.")
    elif not trustworthy:
        print("\n  VERDICT: MIXED — no single strong correlation, but LE re-ranks the")
        print("  leaders, so the top raw-score molecules are somewhat size-inflated.")
    else:
        print("\n  VERDICT: OK — docking is not dominated by size/lipophilicity here;")
        print("  the objective appears to track binding rather than an artifact.")
    print("  (Corroborate with the known-active percentiles above and the figure.)")


if __name__ == "__main__":
    main()
