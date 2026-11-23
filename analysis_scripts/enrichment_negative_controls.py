"""Is the DHFR-core enrichment specific, or is the search just enriching
lipophilic chemistry generally?

This exists because the original contrast fragment was not a control.
`antifolate_enrichment.py` used chlorophenyl as its foil, and chlorophenyl
enriches on the Pareto front at 2.8x on malaria and 3.1x on Chagas, the same
factor as the DHFR pharmacophore core. Read naively that kills the claim.

It does not, because chlorophenyl is PART OF THE ANTIFOLATE PHARMACOPHORE in
this series: pyrimethamine, cycloguanil, WR99210 and C50 (the 3INV crystal
ligand) all carry a chlorinated aryl ring. A fragment present in four of the
six reference inhibitors cannot serve as a negative control for finding those
inhibitors' chemistry. Its co-enrichment is the expected result.

A real control is a fragment that is common in the library, absent from every
reference inhibitor, and mechanistically unrelated to dihydrofolate reductase.
Five are tested here. All five are flat or depleted on both campaigns while the
core sits at 2.8x, which is what specificity looks like.
"""
import os
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REFERENCE_INHIBITORS = {
    "Pyrimethamine": "CCc1nc(N)nc(N)c1-c1ccc(Cl)cc1",
    "Cycloguanil": "CC1(C)N=C(N)N=C(N)N1c1ccc(Cl)cc1",
    "WR99210": "CC1(N=C(N=C(N1OCCCOC2=CC(=C(C=C2Cl)Cl)Cl)N)N)C",
    "P218": "CCc1nc(N)nc(N)c1OCCCOc1ccccc1CCC(=O)O",
    "C50 (3INV crystal ligand)": "CC1(C)N=C(N)N=C(N)N1OCCCOc2cccc(Cl)c2Cl",
    "Methotrexate": "CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(C(=O)NC(CCC(=O)O)C(=O)O)cc1",
}
CORES = ["[nX2]1[cX3]([NX3;H2,H1])[nX2][cX3]([NX3;H2,H1])[cX3][cX3]1",
         "[NX3]1[CX3]([NX3;H2,H1])=[NX2][CX3]([NX3;H2,H1])=[NX2][CX4]1"]
CONTROLS = {
    "sulfonamide": "S(=O)(=O)[NX3]",
    "morpholine": "C1COCCN1",
    "ester": "[CX3](=O)O[CX4]",
    "piperazine": "C1CNCCN1",
    "thiophene": "c1ccsc1",
}
NOT_A_CONTROL = {"chlorophenyl": "Clc1ccccc1"}
CAMPAIGNS = {
    "MALARIA": "campaign_results/seed_{s}/mogp/seed_{s}/pareto_front.csv",
    "CHAGAS": "rung1_chagas_results/full_seed{s}/pareto_front.csv",
}


def main(draws=20000):
    lib = pd.read_csv(os.path.join(_ROOT, "data", "library",
                                   "smiles.csv"))["SMILES"].astype(str).tolist()
    row = {s: i for i, s in enumerate(lib)}
    mols = [Chem.MolFromSmiles(s) for s in lib]
    refs = {n: Chem.MolFromSmiles(s) for n, s in REFERENCE_INHIBITORS.items()}

    def mask(sma):
        p = Chem.MolFromSmarts(sma)
        if p is None:
            raise RuntimeError(f"unparseable SMARTS: {sma}")
        return np.array([bool(m and m.HasSubstructMatch(p)) for m in mols]), p

    core = np.zeros(len(lib), bool)
    for s in CORES:
        core |= mask(s)[0]

    fronts = {}
    for tag, pat in CAMPAIGNS.items():
        f = []
        for s in range(10):
            p = os.path.join(_ROOT, pat.format(s=s))
            if os.path.exists(p):
                smi = pd.read_csv(p)["SMILES"].astype(str)
                f.append(np.array([row[x] for x in smi if x in row]))
        fronts[tag] = f

    rng = np.random.default_rng(3)

    def test(hit):
        base = hit.mean()
        out = {}
        for tag, f in fronts.items():
            obs = sum(int(hit[x].sum()) for x in f)
            tot = sum(len(x) for x in f)
            null = np.zeros(draws, int)
            for x in f:
                null += rng.binomial(len(x), base, draws)
            out[tag] = ((obs / tot) / base, (1 + int((null >= obs).sum())) / (draws + 1))
        return base, out

    print("Which reference inhibitors contain each fragment?")
    for name, sma in {**NOT_A_CONTROL, **CONTROLS}.items():
        p = Chem.MolFromSmarts(sma)
        inref = [n for n, m in refs.items() if m.HasSubstructMatch(p)]
        print(f"  {name:<14} {len(inref)}/6  {', '.join(inref) if inref else '(none)'}")

    print(f"\n{'fragment':<22}{'valid control?':>15}{'lib %':>8}"
          f"{'MAL EF':>8}{'p':>8}{'CHA EF':>8}{'p':>8}")
    print("-" * 78)
    rows = []
    for name, sma, valid in ([("DHFR core (the claim)", None, "n/a")]
                             + [(k, v, "NO, in 4/6 refs") for k, v in NOT_A_CONTROL.items()]
                             + [(k, v, "yes") for k, v in CONTROLS.items()]):
        hit = core if sma is None else mask(sma)[0]
        if hit.mean() < 1e-4:
            print(f"{name:<22}{valid:>15}  too rare to test")
            continue
        base, out = test(hit)
        print(f"{name:<22}{valid:>15}{100*base:>7.2f}%"
              f"{out['MALARIA'][0]:>8.1f}{out['MALARIA'][1]:>8.4f}"
              f"{out['CHAGAS'][0]:>8.1f}{out['CHAGAS'][1]:>8.4f}")
        rows.append(dict(fragment=name, valid_control=valid, library_rate=base,
                         malaria_EF=out["MALARIA"][0], malaria_p=out["MALARIA"][1],
                         chagas_EF=out["CHAGAS"][0], chagas_p=out["CHAGAS"][1]))
    out = os.path.join(_ROOT, "analysis_scripts", "enrichment_negative_controls.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
