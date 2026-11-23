"""F6b for Chagas: does the search find real antifolate chemistry on TcDHFR?

The malaria version of this test asks whether the Pareto front is enriched in
the pterin-mimicking cores that anchor DHFR, and in close analogs of known
antimalarial antifolates. The first question transfers to Chagas unchanged,
because T. cruzi DHFR-TS is a dihydrofolate reductase and binds the same
folate-mimicking cores. The second does not: pyrimethamine, cycloguanil,
WR99210 and P218 are antimalarials, and "close analog of an antimalarial" is
not evidence about a Chagas target.

So the primary test here is the SMARTS core, at the enzyme-family level. The
analog test is run against METHOTREXATE only, which is a classical DHFR
inhibitor with documented activity against trypanosomatid DHFR-TS, and which
this repo already carries as the 1U72 crystal ligand. It is reported as a
secondary line, not as a headline.

Rules carried over from the malaria version, each of which was learned the
hard way:

  1. THE PER-SEED FRONT IS THE UNIT, NEVER THE UNION OF FRONTS. The union
     penalizes a reproducible method: near-identical fronts collapse into one
     small set, which gave p = 0.065 on malaria where the per-seed test gave
     p < 0.0001.
  2. A LOOSE TANIMOTO CUT IS A CHLOROPHENYL DETECTOR. Report T >= 0.4 or the
     SMARTS core. Never quote a loose-threshold number.
  3. SCORES COME FROM THE CAMPAIGN CSVs, NEVER THE DOCKING CACHE. The cache was
     re-keyed and no longer records what the campaign scored.
  4. The random initial design must contribute zero hits, or the enrichment is
     an artifact of the starting set rather than of the search.

Nothing here docks anything. Read-only.
"""
import argparse
import os
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Descriptors import HeavyAtomCount

RDLogger.DisableLog("rdApp.*")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.path.join(_ROOT, "data", "library", "smiles.csv")
CAMPAIGN = os.path.join(_ROOT, "rung1_chagas_results")

# All four arms write RAW kcal/mol into *_Docking on this campaign (the units
# fix landed before it ran, verified: gpmobo TcDHFR spans -10.51 to -5.32, not
# the -0.6 to -0.15 of a ligand efficiency). gpmobo also writes a duplicate
# *_kcal column; prefer it where present so the choice is explicit rather than
# relying on the fix having landed.
ARMS = [("full", "MOGP shared"), ("indep", "MOGP independent"),
        ("gpmobo", "GP-MOBO"), ("greedy", "Filter-then-dock")]
SEEDS = list(range(10))
N_INIT = 40
SIZE_WINDOW = 2

SMARTS_CORES = {
    "2,4-diaminopyrimidine":
        "[nX2]1[cX3]([NX3;H2,H1])[nX2][cX3]([NX3;H2,H1])[cX3][cX3]1",
    "2,4-diamino-dihydrotriazine":
        "[NX3]1[CX3]([NX3;H2,H1])=[NX2][CX3]([NX3;H2,H1])=[NX2][CX4]1",
}
SMARTS_CONTRAST = {"chlorophenyl": "Clc1ccccc1"}

# Classical DHFR inhibitor, documented against trypanosomatid DHFR-TS, and
# already in this repo as the hDHFR (1U72) crystal ligand.
METHOTREXATE = ("methotrexate",
                "CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(C(=O)NC(CCC(=O)O)C(=O)O)cc1")


def fp(mol):
    return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)


def main():
    ap = argparse.ArgumentParser(description="F6b antifolate enrichment, Chagas")
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--out", default=os.path.join(_ROOT, "analysis_scripts",
                                                  "f6b_chagas_enrichment.csv"))
    a = ap.parse_args()
    rng = np.random.default_rng(11)

    lib = pd.read_csv(LIB)["SMILES"].astype(str).tolist()
    row_of = {s: i for i, s in enumerate(lib)}
    n_lib = len(lib)
    mols = [Chem.MolFromSmiles(s) for s in lib]
    ok = np.array([m is not None for m in mols])
    print(f"library: {n_lib} molecules, {int(ok.sum())} parse")

    # ---- masks
    masks = {}
    for label, sma in {**SMARTS_CORES, **SMARTS_CONTRAST}.items():
        patt = Chem.MolFromSmarts(sma)
        if patt is None:
            raise RuntimeError(f"unparseable SMARTS for {label}")
        masks[label] = np.array([bool(m and m.HasSubstructMatch(patt))
                                 for m in mols])
    core = masks["2,4-diaminopyrimidine"] | masks["2,4-diamino-dihydrotriazine"]

    mtx = Chem.MolFromSmiles(METHOTREXATE[1])
    mtx_fp = fp(mtx)
    sims = np.array([DataStructs.TanimotoSimilarity(mtx_fp, fp(m)) if m else 0.0
                     for m in mols])
    analog = sims >= 0.40

    heavy = np.array([HeavyAtomCount(m) if m else 0 for m in mols])

    print(f"\nlibrary base rates")
    print(f"  DHFR pharmacophore core        {core.sum():5d}  "
          f"{100*core.mean():.3f}%")
    print(f"  methotrexate analog T>=0.40    {analog.sum():5d}  "
          f"{100*analog.mean():.3f}%")
    print(f"  chlorophenyl (contrast only)   {masks['chlorophenyl'].sum():5d}  "
          f"{100*masks['chlorophenyl'].mean():.3f}%")

    # ---- load fronts
    arms = {}
    for key, _ in ARMS:
        for s in SEEDS:
            d = os.path.join(CAMPAIGN, f"{key}_seed{s}")
            pe, pp = (os.path.join(d, "evaluated.csv"),
                      os.path.join(d, "pareto_front.csv"))
            if not (os.path.exists(pe) and os.path.exists(pp)):
                continue
            ev = pd.read_csv(pe)["SMILES"].astype(str).tolist()
            fr = pd.read_csv(pp)["SMILES"].astype(str).tolist()
            arms[(key, s)] = dict(
                eval_idx=np.array([row_of[x] for x in ev if x in row_of]),
                init_idx=np.array([row_of[x] for x in ev[:N_INIT]
                                   if x in row_of]),
                front_idx=np.array([row_of[x] for x in fr if x in row_of]))

    rows = []
    for hit_name, hit in [("DHFR core", core), ("MTX analog", analog),
                          ("chlorophenyl (contrast)", masks["chlorophenyl"])]:
        base = hit.mean()
        print(f"\n{'='*78}\n{hit_name.upper()}  (library rate {100*base:.3f}%)")
        print(f"{'arm':<20}{'fronts':>7}{'hits':>6}{'front rate':>12}"
              f"{'EF':>7}{'p (per-seed MC)':>17}")
        print("-" * 78)
        for key, label in ARMS:
            fronts = [arms[(key, s)]["front_idx"] for s in SEEDS
                      if (key, s) in arms]
            if not fronts:
                continue
            obs = sum(int(hit[f].sum()) for f in fronts)
            tot = sum(len(f) for f in fronts)
            # Per-seed null: for each seed draw a random set of the SAME size as
            # that seed's front, from the whole library. Summing across seeds
            # keeps each seed as its own replicate.
            null = np.zeros(a.draws, dtype=int)
            for f in fronts:
                k = len(f)
                null += rng.binomial(k, base, size=a.draws)
            p = (1 + int((null >= obs).sum())) / (a.draws + 1)
            rate = obs / tot if tot else 0.0
            ef = rate / base if base > 0 else float("nan")
            print(f"{label:<20}{len(fronts):>7}{obs:>6}{100*rate:>11.2f}%"
                  f"{ef:>7.1f}{p:>17.4f}")
            rows.append(dict(test=hit_name, arm=label, n_fronts=len(fronts),
                             hits=obs, front_total=tot, front_rate=rate,
                             library_rate=base, EF=ef, p_perseed=p))

    # ---- audit 1: the random initial design must contribute nothing
    print(f"\n{'='*78}\nAUDIT: hits inside the 40-molecule RANDOM initial design")
    for key, label in ARMS:
        inits = [arms[(key, s)]["init_idx"] for s in SEEDS if (key, s) in arms]
        if not inits:
            continue
        print(f"  {label:<20} core {sum(int(core[i].sum()) for i in inits):>3}"
              f"   MTX analog {sum(int(analog[i].sum()) for i in inits):>3}")

    # ---- audit 2: size-matched null for the core test
    print(f"\n{'='*78}\nAUDIT: heavy-atom-matched null (is this just molecule size?)")
    hit_sizes = heavy[core]
    for key, label in ARMS:
        fronts = [arms[(key, s)]["front_idx"] for s in SEEDS if (key, s) in arms]
        if not fronts:
            continue
        obs = sum(int(core[f].sum()) for f in fronts)
        null = np.zeros(2000, dtype=int)
        for f in fronts:
            sizes = heavy[f]
            # draw replacements with matched heavy-atom count
            pool = [np.where(np.abs(heavy - z) <= SIZE_WINDOW)[0] for z in sizes]
            for d in range(2000):
                pick = np.array([p[rng.integers(len(p))] for p in pool])
                null[d] += int(core[pick].sum())
        p = (1 + int((null >= obs).sum())) / 2001
        print(f"  {label:<20} observed {obs:>3}   size-matched null mean "
              f"{null.mean():6.1f}   p = {p:.4f}")

    pd.DataFrame(rows).to_csv(a.out, index=False)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
