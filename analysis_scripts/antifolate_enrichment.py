"""
antifolate_enrichment.py  —  F6b
================================

Does the pipeline's Pareto front contain MORE known-antifolate chemistry than
the same number of molecules drawn at random would?

WHY THE EXISTING CHECK IS NOT ENOUGH. ``validate_known_actives.py`` reports, for
each reference drug, the highest Tanimoto similarity any Pareto molecule reaches
to it, and calls the drug "recovered" above 0.4. That number has no null. The
maximum of a set grows with the size of the set, so a 172-molecule front beats a
16-molecule front on it for arithmetic reasons alone, and neither tells you
whether the SELECTION did anything. This script supplies the null, and it tests
three further things the similarity number cannot: whether the hits were found by
the optimizer or handed to it in the random initial design, whether they survive
a chemically defined hit set instead of a similarity threshold, and whether they
survive a molecular-size-matched null.

QUANTITIES, DEFINED BEFORE ANY VALUE IS QUOTED.

  Tanimoto similarity. Dimensionless, 0 to 1. Computed on 2048-bit Morgan
  fingerprints at radius 2, the same featurization the GP kernel uses, so this is
  the similarity the model itself sees. 1.0 is identical bit vectors. On this
  representation, values above roughly 0.4 usually mean a shared scaffold rather
  than scattered shared fragments.

  Enrichment factor (EF). Dimensionless, a ratio of two rates. Numerator: the
  fraction of a selected set that is a hit. Denominator: the fraction of the whole
  searchable library that is. EF = 1 means the selection is indistinguishable
  from a random draw of the same size. EF = 10 means hits are ten times denser on
  the front than in the library. Because EF is a rate ratio it does not inflate
  with set size, which is the defect of maximum similarity.

  Heavy-atom count. The number of atoms excluding hydrogen. Used only to build
  the size-matched null.

THREE HIT SETS, reported side by side because the choice is contestable.

  1. PHARMACOPHORE CORE. A library molecule matching 2,4-diaminopyrimidine or
     2,4-diamino-1,6-dihydro-1,3,5-triazine by SMARTS. This is the group that
     mimics the pterin of dihydrofolate and anchors the DHFR active site; all four
     reference drugs match it. This set is chemically defined, not tuned by a
     threshold the author picked, so it is the PRIMARY definition.
  2. CLOSE ANALOG. Tanimoto >= 0.4 to at least one reference drug. Narrow and
     specific: 18 of its 20 members also carry the core.
  3. CORE AND Tanimoto >= 0.3. The conjunction, as a middle case.

  Looser similarity thresholds are deliberately NOT reported as headline numbers.
  At Tanimoto >= 0.25 only 52 of 214 "near-neighbors" carry a diamino core while
  59 carry nothing but a chlorophenyl, a fragment in 13.5% of the library. A loose
  threshold measures chlorophenyl, not antifolate pharmacology.

THE TEST. Ten seeds are the replicates, and the per-seed front is the unit. Under
the null that selection ignores antifolate-likeness, each seed's front is a
uniform draw of its own size from the library; the null distribution of the total
hit count over all ten seeds is obtained by Monte Carlo. The per-seed unit matters:
pooling the ten fronts into their union penalizes a REPRODUCIBLE method, because a
method returning nearly the same front every seed has a small union and therefore
few distinct hits. MOGP's union test gives p = 0.065 for that reason alone while
its per-seed test gives p < 0.0001. The union is reported for completeness and is
not the headline.

FOUR AUDITS, each able to kill the result.
  A. Initial design. The campaign draws 40 random molecules before the optimizer
     acts (``loop.GRAND_CAMPAIGN_PARAMS``), then 50 iterations of 5. A hit in rows
     1-40 of evaluated.csv was not found, it was handed over. Reported separately.
  B. Search versus selection. Enrichment can enter when the acquisition chooses
     what to dock, or when the Pareto filter chooses what to keep. Tested apart:
     evaluated-versus-library isolates the acquisition, front-versus-its-own-
     evaluated-set isolates the filter.
  C. Size. Docking affinity rises with molecular size and the reference drugs are
     small, so a size preference could manufacture or mask the effect. The
     size-matched null replaces each front member with a random library molecule
     within 2 heavy atoms of it.
  D. Distinct versus repeated. Twenty hits across ten seeds can be two molecules
     found ten times or twenty found once. Both counts are reported.

NO DOCKING. Everything reads from disk: library fingerprints, the campaign's
committed CSVs, and the persistent docking cache opened read-only. No Vina call,
no cache write, no cache re-keying.

Run::

    python analysis_scripts/antifolate_enrichment.py
    python analysis_scripts/antifolate_enrichment.py --draws 50000
"""

import os
import sys
import sqlite3
import argparse

import numpy as np
import pandas as pd
from scipy.stats import hypergeom, wilcoxon

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rdkit import Chem, RDLogger                               # noqa: E402
from utils.featurize import smiles_to_morgan                   # noqa: E402

RDLogger.DisableLog("rdApp.*")

# --------------------------------------------------------------------------- #
# Ground truth. Same four compounds and SMILES as validate_known_actives.py,
# duplicated rather than imported because that module's import pulls in the
# ADMET oracle and the docking stack.
# --------------------------------------------------------------------------- #
KNOWN_ACTIVES = [
    ("Pyrimethamine", "PubChem CID 4993",
     "CCC1=C(C(=NC(=N1)N)N)C2=CC=C(C=C2)Cl"),
    ("Cycloguanil", "PubChem CID 9049",
     "CC1(N=C(N=C(N1C2=CC=C(C=C2)Cl)N)N)C"),
    ("WR99210", "PubChem CID 121749 (free base)",
     "CC1(N=C(N=C(N1OCCCOC2=CC(=C(C=C2Cl)Cl)Cl)N)N)C"),
    ("P218", "ChEMBL CHEMBL3040038",
     "CCc1nc(N)nc(N)c1OCCCOc1ccccc1CCC(=O)O"),
]

# The pterin-mimicking cores that anchor DHFR. All four reference drugs match.
SMARTS_CORES = {
    "2,4-diaminopyrimidine":
        "[nX2]1[cX3]([NX3;H2,H1])[nX2][cX3]([NX3;H2,H1])[cX3][cX3]1",
    "2,4-diamino-dihydrotriazine":
        "[NX3]1[CX3]([NX3;H2,H1])=[NX2][CX3]([NX3;H2,H1])=[NX2][CX4]1",
}
SMARTS_CONTRAST = {"chlorophenyl": "Clc1ccccc1"}

LIB_DIR = os.path.join(_ROOT, "data", "library")
CAMPAIGN = os.path.join(_ROOT, "campaign_results")
AGG = os.path.join(CAMPAIGN, "aggregate_10seed_cleanGPMOBO", "tier1")
CACHE = os.path.join(_ROOT, "data", "docking_cache", "docking_cache.sqlite")

METHODS = [("gpmobo", "GPMOBO"), ("mogp", "MOGP (ours)"), ("greedy", "Greedy")]
SEEDS = list(range(10))
N_INIT = 40            # loop.GRAND_CAMPAIGN_PARAMS: 40 random, then 50 x 5
SIZE_WINDOW = 2        # heavy atoms, for the size-matched null


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_library():
    """Fingerprints, row-aligned SMILES, RDKit mols and heavy-atom counts."""
    fp = np.load(os.path.join(LIB_DIR, "fingerprints.npy")).astype(np.uint8)
    smiles = pd.read_csv(os.path.join(LIB_DIR, "smiles.csv"))["SMILES"].tolist()
    if len(smiles) != fp.shape[0]:
        raise RuntimeError(f"library not row-aligned: {fp.shape[0]} vs "
                           f"{len(smiles)}")
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    heavy = np.array([m.GetNumHeavyAtoms() if m else 0 for m in mols], float)
    return fp, smiles, mols, heavy


def similarity_matrix(lib_fp):
    """(4, N) Tanimoto of each reference drug against the whole library."""
    pop = lib_fp.sum(axis=1).astype(np.float64)
    rows = []
    for _, _, smi in KNOWN_ACTIVES:
        a = smiles_to_morgan(smi).astype(np.uint8)
        inter = (lib_fp @ a).astype(np.float64)
        union = pop + float(a.sum()) - inter
        rows.append(np.divide(inter, union, out=np.zeros_like(inter),
                              where=union > 0))
    return np.vstack(rows)


def substructure_masks(mols):
    """Boolean library masks for each SMARTS core and the contrast fragment."""
    out = {}
    for label, sma in {**SMARTS_CORES, **SMARTS_CONTRAST}.items():
        patt = Chem.MolFromSmarts(sma)
        if patt is None:
            raise RuntimeError(f"unparseable SMARTS for {label}: {sma}")
        out[label] = np.array(
            [bool(m and m.HasSubstructMatch(patt)) for m in mols])
    return out


def load_campaign(row_of):
    """Per (method, seed): library row indices of evaluated and front, in order.

    ``evaluated`` keeps file order, which is evaluation order, so the first
    ``N_INIT`` entries are the random initial design.
    """
    out = {}
    for m, _ in METHODS:
        for s in SEEDS:
            d = os.path.join(CAMPAIGN, f"seed_{s}", m, f"seed_{s}")
            pe = os.path.join(d, "evaluated.csv")
            pp = os.path.join(d, "pareto_front.csv")
            if not (os.path.exists(pe) and os.path.exists(pp)):
                continue
            ev = pd.read_csv(pe)["SMILES"].astype(str).tolist()
            fr = pd.read_csv(pp)["SMILES"].astype(str).tolist()
            out[(m, s)] = {
                "eval_smiles": ev,
                "eval_idx": np.array([row_of[x] for x in ev if x in row_of]),
                "front_idx": np.array([row_of[x] for x in fr if x in row_of]),
                "front_smiles": set(fr),
                "pos": {x: i for i, x in enumerate(ev, start=1)},
            }
    return out


def load_cache():
    """``{smiles: {target: affinity}}`` for OK rows. Read-only, never written."""
    if not os.path.exists(CACHE):
        return {}, {}
    con = sqlite3.connect(f"file:{CACHE}?mode=ro", uri=True)
    rows = con.execute("SELECT smiles, target, oracle_fingerprint, affinity "
                       "FROM docking_scores WHERE status = 'ok'").fetchall()
    con.close()
    scores, fps = {}, {}
    for smi, tgt, ofp, aff in rows:
        scores.setdefault(smi, {})[tgt] = aff
        fps[tgt] = ofp
    return scores, fps


# Column semantics differ BY ARM, and the names do not say so. In the GPMOBO
# arm ``PfDHFR_Docking`` holds LIGAND EFFICIENCY (kcal/mol per heavy atom) and
# ``PfDHFR_Docking_kcal`` holds the raw affinity; in the MOGP and Greedy arms it
# is the reverse, ``PfDHFR_Docking`` is raw kcal/mol and ``PfDHFR_Docking_LE`` is
# the efficiency. Each arm's column named ``*_Docking`` is the objective that arm
# actually optimized, so the two are not interchangeable: reading ``*_Docking``
# across arms puts kcal/mol and kcal/mol-per-atom on one axis. This map pins the
# RAW kcal/mol column per arm so nothing here can make that mistake.
KCAL_COLUMN = {
    "gpmobo": {"PfDHFR": "PfDHFR_Docking_kcal", "hDHFR": "hDHFR_Docking_kcal"},
    "mogp":   {"PfDHFR": "PfDHFR_Docking",      "hDHFR": "hDHFR_Docking"},
    "greedy": {"PfDHFR": "PfDHFR_Docking",      "hDHFR": "hDHFR_Docking"},
}


def campaign_scores():
    """``{smiles: {"PfDHFR": kcal, "hDHFR": kcal}}`` from the campaign CSVs.

    The committed campaign CSVs, not the cache, are the authoritative record of
    what the published campaign scored: 53.5% of the molecules those runs
    evaluated are no longer present in the live docking cache, because the cache
    was re-keyed after the campaign (a new ``oracle_fingerprint``) and the old
    rows did not carry over. Nothing published is affected, the CSVs are intact
    and read-only, but the cache cannot be used as a record of the campaign.
    Scores are read through KCAL_COLUMN so the units are raw kcal/mol in all
    arms.
    """
    out = {}
    for m, _ in METHODS:
        cols = KCAL_COLUMN[m]
        for s in SEEDS:
            f = os.path.join(CAMPAIGN, f"seed_{s}", m, f"seed_{s}",
                             "evaluated.csv")
            if not os.path.exists(f):
                continue
            df = pd.read_csv(f)
            if not all(c in df.columns for c in cols.values()):
                continue
            for _, r in df.iterrows():
                d = out.setdefault(str(r["SMILES"]), {})
                for tgt, col in cols.items():
                    v = r[col]
                    if pd.notna(v):
                        d[tgt] = float(v)
    return out


def perseed_mc_test(hit, fronts, n_lib, draws, rng):
    """Total hits over seeds, with a Monte-Carlo null of same-size draws.

    ``fronts`` is a list of index arrays, one per seed. Under the null each
    seed's front is a uniform sample of its own size from the library, so the
    null total is the sum of those per-seed hit counts.
    """
    sizes = np.array([len(f) for f in fronts])
    observed = int(sum(int(hit[f].sum()) for f in fronts))
    null = np.empty(draws)
    for i in range(draws):
        null[i] = sum(int(hit[rng.choice(n_lib, size=n, replace=False)].sum())
                      for n in sizes)
    h = int(hit.sum())
    expected = float(sizes.sum() * h / n_lib)
    rate = observed / sizes.sum() if sizes.sum() else float("nan")
    return dict(observed=observed, expected=expected,
                ef=(rate / (h / n_lib)) if h else float("nan"),
                p=float((null >= observed).mean()),
                null_mean=float(null.mean()),
                null_p95=float(np.percentile(null, 95)),
                seeds_with_hit=int(sum(1 for f in fronts if hit[f].sum() > 0)),
                n_per_seed=float(sizes.mean()))


def size_matched_test(hit, fronts, heavy, draws, rng):
    """Null that preserves the front's heavy-atom profile molecule by molecule."""
    idx = np.concatenate([f for f in fronts if len(f)]) if any(
        len(f) for f in fronts) else np.array([], int)
    if len(idx) == 0:
        return dict(observed=0, null_mean=float("nan"), p=float("nan"))
    cache = {}
    pools = []
    for i in idx:
        key = int(heavy[i])
        if key not in cache:
            cache[key] = np.flatnonzero(np.abs(heavy - key) <= SIZE_WINDOW)
        pools.append(cache[key])
    observed = int(hit[idx].sum())
    null = np.array([sum(int(hit[rng.choice(p)]) for p in pools)
                     for _ in range(draws)], float)
    return dict(observed=observed, null_mean=float(null.mean()),
                null_p95=float(np.percentile(null, 95)),
                p=float((null >= observed).mean()))


def main():
    ap = argparse.ArgumentParser(description="F6b antifolate enrichment")
    ap.add_argument("--draws", type=int, default=20_000,
                    help="Monte-Carlo draws for the per-seed null.")
    ap.add_argument("--size-draws", type=int, default=3_000)
    ap.add_argument("--out", default=os.path.join(
        _ROOT, "analysis_scripts", "antifolate_enrichment.csv"))
    args = ap.parse_args()

    lib_fp, lib_smiles, mols, heavy = load_library()
    n_lib = len(lib_smiles)
    row_of = {s: i for i, s in enumerate(lib_smiles)}
    sims = similarity_matrix(lib_fp)
    best = sims.max(axis=0)
    masks = substructure_masks(mols)
    core = masks["2,4-diaminopyrimidine"] | masks["2,4-diamino-dihydrotriazine"]
    arms = load_campaign(row_of)
    cached, oracle_fps = load_cache()
    scored = campaign_scores()

    HIT_SETS = [
        ("pharmacophore core", core),
        ("close analog T>=0.4", best >= 0.40),
        ("core AND T>=0.3", core & (best >= 0.30)),
    ]

    print("=" * 100)
    print("F6b  IS THE PARETO FRONT ENRICHED FOR KNOWN ANTIFOLATE CHEMISTRY?")
    print("=" * 100)
    print(f"library        : {n_lib} molecules, 2048-bit Morgan radius 2")
    print(f"campaign arms  : {len(arms)} (method x seed), "
          f"{N_INIT} random initial + 50 iterations x 5")
    print(f"docking cache  : {len(cached)} molecules with an OK score, "
          "opened read-only")
    for t, f in sorted(oracle_fps.items()):
        print(f"                 {t:<8} oracle fingerprint {f[:16]}...")
    pooled = {x for k in arms for x in arms[k]["eval_smiles"]}
    miss = sum(1 for x in pooled if x not in cached)
    print(f"docking scores : read from the campaign CSVs, which are the "
          f"authoritative record.")
    print(f"                 {len(pooled)} molecules were evaluated by the "
          f"campaign; {miss} of them\n                 ({100*miss/len(pooled):.1f}%) "
          f"are NO LONGER in the live cache, which was re-keyed\n"
          f"                 after the campaign ran. Published numbers are "
          f"unaffected; the cache\n                 is not a usable record of "
          f"the campaign.")

    # ------------------------------------------------------------------ #
    print("\n" + "-" * 100)
    print("1. HIT-SET DEFINITIONS, AND WHY A LOOSE SIMILARITY CUT IS NOT ONE")
    print("-" * 100)
    for label, sma in SMARTS_CORES.items():
        m = masks[label]
        print(f"  {label:<30}{int(m.sum()):>6} molecules "
              f"({100*m.mean():.3f}% of library)")
    for label in SMARTS_CONTRAST:
        m = masks[label]
        print(f"  {label + ' (contrast)':<30}{int(m.sum()):>6} molecules "
              f"({100*m.mean():.3f}% of library)")
    print(f"  {'EITHER diamino core':<30}{int(core.sum()):>6} molecules "
          f"({100*core.mean():.3f}%)   <- PRIMARY hit set")
    ok = all(Chem.MolFromSmiles(s).HasSubstructMatch(
        Chem.MolFromSmarts(SMARTS_CORES["2,4-diaminopyrimidine"]))
        or Chem.MolFromSmiles(s).HasSubstructMatch(
        Chem.MolFromSmarts(SMARTS_CORES["2,4-diamino-dihydrotriazine"]))
        for _, _, s in KNOWN_ACTIVES)
    print(f"  all four reference drugs match the core: {ok}")

    print("\n  Closest library molecule to each reference drug (Tanimoto):")
    for j, (name, src, _) in enumerate(KNOWN_ACTIVES):
        i = int(np.argmax(sims[j]))
        print(f"    {name:<15} T={sims[j][i]:.3f}  core="
              f"{'yes' if core[i] else 'no':<3}  {lib_smiles[i]}")
        print(f"    {'':<15} ({src})")

    for t in (0.25, 0.30, 0.40):
        s = best >= t
        print(f"\n  at T >= {t:g}: {int(s.sum()):>4} molecules, of which "
              f"{int((s & core).sum()):>3} carry a diamino core and "
              f"{int((s & masks['chlorophenyl'] & ~core).sum()):>3} carry only "
              "a chlorophenyl")
    print("\n  -> the loose cut is substantially a chlorophenyl detector, so "
          "the\n     pharmacophore core is used as the primary hit set.")

    # ------------------------------------------------------------------ #
    print("\n" + "-" * 100)
    print("2. PRIMARY TEST. PER-SEED FRONTS, 10 SEEDS AS REPLICATES")
    print("-" * 100)
    print(f"Null: each seed's front is a uniform draw of its own size from the "
          f"library.\n{args.draws} Monte-Carlo draws. EF = 1 means "
          "indistinguishable from random. One-sided.\n")
    records = []
    rng = np.random.default_rng(20261116)
    for label, hit in HIT_SETS:
        h = int(hit.sum())
        print(f"--- hit set: {label}  ({h} of {n_lib} library molecules, "
              f"base rate {100*h/n_lib:.4f}%) ---")
        hdr = (f"  {'method':<14}{'front n/seed':>13}{'hits':>6}{'exp':>7}"
               f"{'EF':>7}{'p':>9}{'seeds>=1':>10}{'from init':>11}"
               f"{'distinct':>10}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for m, lab in METHODS:
            fronts = [arms[(m, s)]["front_idx"] for s in SEEDS if (m, s) in arms]
            if not fronts:
                continue
            r = perseed_mc_test(hit, fronts, n_lib, args.draws, rng)
            from_init = sum(
                1 for s in SEEDS if (m, s) in arms
                for i in arms[(m, s)]["front_idx"]
                if hit[i] and arms[(m, s)]["pos"].get(lib_smiles[i], 10**9)
                <= N_INIT)
            distinct = len({int(i) for f in fronts for i in f if hit[i]})
            print(f"  {lab:<14}{r['n_per_seed']:>13.1f}{r['observed']:>6}"
                  f"{r['expected']:>7.2f}{r['ef']:>7.1f}{r['p']:>9.4f}"
                  f"{str(r['seeds_with_hit']) + '/10':>10}{from_init:>11}"
                  f"{distinct:>10}")
            records.append(dict(test="per_seed_mc", hit_set=label,
                                n_hits_library=h, method=lab,
                                from_initial_design=from_init,
                                distinct_molecules=distinct, **r))
        print()

    # ------------------------------------------------------------------ #
    print("-" * 100)
    print("3. AUDIT: SIZE-MATCHED NULL")
    print("-" * 100)
    print(f"Each front member is replaced by a random library molecule within "
          f"{SIZE_WINDOW} heavy atoms of it,\nso the null front carries the same "
          "molecular-size profile. Docking favours larger molecules,\nand the "
          "reference drugs are small (17 to 26 heavy atoms), so this is the "
          "confound that matters.\n")
    print(f"  {'hit set':<22}{'method':<14}{'hits':>6}{'null mean':>11}"
          f"{'null p95':>10}{'p':>9}")
    print("  " + "-" * 70)
    for label, hit in HIT_SETS[:2]:
        for m, lab in METHODS:
            fronts = [arms[(m, s)]["front_idx"] for s in SEEDS if (m, s) in arms]
            if not fronts:
                continue
            r = size_matched_test(hit, fronts, heavy, args.size_draws, rng)
            print(f"  {label:<22}{lab:<14}{r['observed']:>6}"
                  f"{r['null_mean']:>11.2f}{r['null_p95']:>10.1f}"
                  f"{r['p']:>9.4f}")
            records.append(dict(test="size_matched", hit_set=label,
                                method=lab, **r))
    fi = np.concatenate([arms[k]["front_idx"] for k in arms])
    print(f"\n  median heavy atoms: library {np.median(heavy):.0f}, "
          f"core set {np.median(heavy[core]):.0f}, "
          f"all fronts {np.median(heavy[fi]):.0f}; reference drugs "
          + ", ".join(str(Chem.MolFromSmiles(s).GetNumHeavyAtoms())
                      for _, _, s in KNOWN_ACTIVES))

    # ------------------------------------------------------------------ #
    print("\n" + "-" * 100)
    print("4. AUDIT: WAS IT THE ACQUISITION OR THE PARETO FILTER?")
    print("-" * 100)
    print("evaluated vs library  isolates the acquisition, which chooses what to "
          "dock.")
    print("front vs its own evaluated set  isolates the Pareto filter, which "
          "chooses what to keep.\n")
    print(f"  {'hit set':<22}{'method':<14}{'EF search':>11}{'p':>9}"
          f"{'EF filter':>11}{'p':>9}")
    print("  " + "-" * 76)
    for label, hit in HIT_SETS[:2]:
        h = int(hit.sum())
        for m, lab in METHODS:
            ev = np.concatenate([arms[(m, s)]["eval_idx"] for s in SEEDS
                                 if (m, s) in arms])
            fr = np.concatenate([arms[(m, s)]["front_idx"] for s in SEEDS
                                 if (m, s) in arms])
            ke, kf = int(hit[ev].sum()), int(hit[fr].sum())
            ef_s = (ke / len(ev)) / (h / n_lib)
            p_s = float(hypergeom.sf(ke - 1, n_lib, h, len(ev))) if ke else 1.0
            ef_f = ((kf / len(fr)) / (ke / len(ev))) if ke else float("nan")
            p_f = float(hypergeom.sf(kf - 1, len(ev), ke, len(fr))) if kf else 1.0
            print(f"  {label:<22}{lab:<14}{ef_s:>11.2f}{p_s:>9.4f}"
                  f"{ef_f:>11.2f}{p_f:>9.4f}")
            records.append(dict(test="search_vs_filter", hit_set=label,
                                method=lab, ef_search=ef_s, p_search=p_s,
                                ef_filter=ef_f, p_filter=p_f,
                                eval_hits=ke, front_hits=kf))

    # ------------------------------------------------------------------ #
    print("\n" + "-" * 100)
    print("5. HEAD TO HEAD, PAIRED ON THE 10 SHARED SEEDS")
    print("-" * 100)
    print("Hits per seed on the pharmacophore core. Paired Wilcoxon signed-rank; "
          "with n=10\nthe smallest attainable p is 0.0020.\n")
    per = {m: np.array([int(core[arms[(m, s)]["front_idx"]].sum())
                        for s in SEEDS if (m, s) in arms])
           for m, _ in METHODS}
    for b, blab in (("gpmobo", "GPMOBO"), ("greedy", "Greedy")):
        d = per["mogp"] - per[b]
        try:
            p = float(wilcoxon(per["mogp"], per[b]).pvalue)
        except Exception:                                        # noqa: BLE001
            p = float("nan")
        print(f"  MOGP - {blab:<8} mean {d.mean():+.2f} hits/seed, "
              f"MOGP ahead in {int((d > 0).sum())}/10 seeds, "
              f"tied {int((d == 0).sum())}, p={p:.4f}")
        records.append(dict(test="paired_vs_baseline", hit_set="pharmacophore "
                            "core", method=f"MOGP vs {blab}",
                            mean_diff=float(d.mean()),
                            wins=int((d > 0).sum()), p=p))

    # ------------------------------------------------------------------ #
    print("\n" + "-" * 100)
    print("6. THE MOLECULES RECOVERED (close analog set, T >= 0.4)")
    print("-" * 100)
    names = [n for n, _, _ in KNOWN_ACTIVES]
    print(f"  {'method':<14}{'T':>6}  {'closest drug':<15}{'core':>6}"
          f"{'heavy':>7}{'seeds':>7}  PfDHFR  hDHFR  SelIdx  smiles")
    print("  Docking affinity in kcal/mol, more negative = tighter. Selectivity")
    print("  Index = hDHFR - PfDHFR in kcal/mol; POSITIVE = parasite-selective.")
    rows = []
    for m, lab in METHODS:
        fronts = [arms[(m, s)]["front_idx"] for s in SEEDS if (m, s) in arms]
        hits = sorted({int(i) for f in fronts for i in f if best[i] >= 0.40},
                      key=lambda i: -best[i])
        if not hits:
            print(f"  {lab:<14}(none)")
            continue
        for i in hits:
            smi = lib_smiles[i]
            sc = scored.get(smi, {})
            pf, hd = sc.get("PfDHFR"), sc.get("hDHFR")
            si = (hd - pf) if (pf is not None and hd is not None) else None
            f = lambda x: f"{x:7.2f}" if x is not None else f"{'-':>7}"
            ns = sum(1 for fr in fronts if i in set(fr.tolist()))
            print(f"  {lab:<14}{best[i]:>6.3f}  "
                  f"{names[int(np.argmax(sims[:, i]))]:<15}"
                  f"{'yes' if core[i] else 'no':>6}{int(heavy[i]):>7}"
                  f"{str(ns) + '/10':>7} {f(pf)}{f(hd)}{f(si)}  {smi}")
            rows.append(dict(method=lab, smiles=smi, max_tanimoto=best[i],
                             closest_drug=names[int(np.argmax(sims[:, i]))],
                             has_core=bool(core[i]), heavy_atoms=int(heavy[i]),
                             seeds_recovered=ns, pfdhfr=pf, hdhfr=hd,
                             selectivity_index=si))

    out_n = args.out.replace(".csv", "_molecules.csv")
    pd.DataFrame(records).to_csv(args.out, index=False)
    pd.DataFrame(rows).to_csv(out_n, index=False)
    print(f"\n  wrote {args.out}")
    print(f"  wrote {out_n}")


if __name__ == "__main__":
    main()
