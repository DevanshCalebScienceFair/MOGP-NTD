"""F6c: are the molecules we propose any harder to make than ordinary chemistry?

The quantity is the SA score (synthetic accessibility, Ertl & Schuffenhauer
2009): 1 is trivially easy to synthesise, 10 is very hard. Lower is better.

THE POINT OF THIS SCRIPT IS THE BASELINE, NOT THE SCORE.

Comparing our Pareto front against the screening library is circular. The
library was built by filtering ChEMBL through `quality_filter`, which includes
a synthesizability gate, so hard-to-make molecules were already removed before
the optimizer ever saw them. Beating that baseline would prove nothing: the
baseline cannot contain the molecules it would need to lose to.

The honest baseline is the UNFILTERED SOURCE, the raw ChEMBL_V29 set the
library was drawn from, before Lipinski, before the heavy-atom floor, and
before the quality gate. Four populations are reported so the filtering
pipeline's own effect is visible:

  1. ChEMBL source   unfiltered, the real baseline
  2. curated library after Lipinski
  3. searched set    after the heavy-atom floor and quality gate (circular)
  4. Pareto fronts   what the optimizer actually proposed

The claim F6c can support is only as strong as (4) vs (1).

No docking. Read-only.
"""
import os
import sys
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
import sa_score  # noqa: E402

# The project's own data/chembl_v29.csv is a 0-byte placeholder; the real
# 2,084,723-molecule source lives in the earlier project layout. Resolved by
# SIZE, not by path alone, so an empty placeholder cannot be used silently.
SOURCE_CANDIDATES = [
    os.path.join(_ROOT, "data", "chembl_v29.csv"),
    "/Users/devansh/Downloads/ScienceFair/data/chembl_v29.csv",
    "/Users/devansh/Downloads/ExperimentalScienceFair/data/chembl_v29.csv",
]
N_SOURCE = 30000           # matched to the curated library's scale
from data import HEAVY_ATOM_FLOOR   # 14; imported, never hardcoded


def resolve_source():
    for p in SOURCE_CANDIDATES:
        if os.path.exists(p) and os.path.getsize(p) > 1_000_000:
            return p
    raise SystemExit("no non-empty chembl_v29.csv found; F6c cannot run on a "
                     "circular baseline, so it refuses rather than substituting "
                     "the library")


def sa_of(smiles_list, label):
    vals = []
    bad = 0
    for s in smiles_list:
        m = Chem.MolFromSmiles(s)
        if m is None:
            bad += 1
            continue
        try:
            vals.append(sa_score.calculate_score(m))
        except Exception:
            bad += 1
    print(f"  {label:<34} n={len(vals):6d}  unparseable/failed={bad}")
    return np.asarray(vals)


def main():
    src = resolve_source()
    print(f"unfiltered source: {src} ({os.path.getsize(src)/1e6:.0f} MB)")
    if not sa_score.SA_AVAILABLE:
        raise SystemExit("SA scorer unavailable; refusing to report a partial "
                         "answer (sa_score raises rather than degrading)")
    print(f"SA backend: {sa_score.SA_BACKEND}\n")

    rng = np.random.default_rng(17)
    src_df = pd.read_csv(src, usecols=["smiles"])
    print(f"source rows: {len(src_df)}")
    idx = rng.choice(len(src_df), size=min(N_SOURCE, len(src_df)), replace=False)
    source_smiles = src_df["smiles"].iloc[idx].astype(str).tolist()
    del src_df

    lib = pd.read_csv(os.path.join(_ROOT, "data", "library",
                                   "smiles.csv"))["SMILES"].astype(str).tolist()
    rejected = pd.read_csv(os.path.join(_ROOT, "data", "library",
                                        "quality_rejected.csv"))["SMILES"].astype(str).tolist()
    # The searched set is the curated library minus what the quality gate and
    # the heavy-atom floor removed. Reconstructed, not assumed: quality_rejected
    # holds the gate's removals.
    rej = set(rejected)
    searched = [s for s in lib
                if s not in rej
                and (Chem.MolFromSmiles(s) is not None)
                and Chem.MolFromSmiles(s).GetNumHeavyAtoms() >= HEAVY_ATOM_FLOOR]

    fronts = {}
    for tag, pat in [("malaria", "campaign_results/seed_{s}/mogp/seed_{s}/pareto_front.csv"),
                     ("chagas", "rung1_chagas_results/full_seed{s}/pareto_front.csv")]:
        smi = []
        for s in range(10):
            p = os.path.join(_ROOT, pat.format(s=s))
            if os.path.exists(p):
                smi += pd.read_csv(p)["SMILES"].astype(str).tolist()
        fronts[tag] = sorted(set(smi))

    print("\nscoring:")
    pops = {
        "1. ChEMBL source (UNFILTERED)": sa_of(source_smiles, "ChEMBL source"),
        "2. curated library": sa_of(lib, "curated library"),
        "3. searched set (circular)": sa_of(searched, "searched set"),
        "4. Pareto front, malaria": sa_of(fronts["malaria"], "front malaria"),
        "4. Pareto front, Chagas": sa_of(fronts["chagas"], "front Chagas"),
    }

    base = pops["1. ChEMBL source (UNFILTERED)"]
    print(f"\n{'population':<32}{'n':>7}{'median':>9}{'mean':>8}{'p90':>8}"
          f"{'% > 6 (hard)':>14}{'vs source':>11}")
    print("-" * 92)
    rows = []
    for name, v in pops.items():
        delta = "" if v is base else f"{np.median(v)-np.median(base):+.2f}"
        print(f"{name:<32}{len(v):>7}{np.median(v):>9.2f}{v.mean():>8.2f}"
              f"{np.percentile(v,90):>8.2f}{100*(v>6).mean():>13.1f}%{delta:>11}")
        rows.append(dict(population=name, n=len(v), median=np.median(v),
                         mean=v.mean(), p90=np.percentile(v, 90),
                         pct_hard=100 * (v > 6).mean()))

    from scipy.stats import mannwhitneyu
    print("\nfront vs UNFILTERED source (Mann-Whitney, two-sided):")
    for tag in ("malaria", "Chagas"):
        v = pops[f"4. Pareto front, {tag}"]
        u = mannwhitneyu(v, base, alternative="two-sided")
        easier = "EASIER" if np.median(v) < np.median(base) else "harder"
        print(f"  {tag:<9} median {np.median(v):.2f} vs {np.median(base):.2f} "
              f"({easier})  p = {u.pvalue:.3g}")

    out = os.path.join(_ROOT, "analysis_scripts", "f6c_synthesizability.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    np.save(os.path.join(_ROOT, "analysis_scripts", "f6c_sa_distributions.npy"),
            {k: v for k, v in pops.items()}, allow_pickle=True)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
