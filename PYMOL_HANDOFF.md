# PfDHFR lead candidates — handoff for PyMOL follow-up

Everything needed to load, inspect, and re-dock the top candidates from the
MOGP-NTD Bayesian-optimization sweep. All paths are relative to the repo root
`/Users/devansh/mogp-main-vscode/MOGP-NTD`.

## 1. What this is

A multi-objective BO pipeline searched a 3610-molecule ChEMBL library for
antimalarial leads against *Plasmodium falciparum* DHFR, scoring 5 objectives at
once. 38 optimizer configurations were run; 533 distinct molecules reached a
Pareto front. The molecules below are those fronts, pooled and re-ranked.

**The goal is SELECTIVITY, not raw potency.** A useful compound binds PfDHFR
strongly and human DHFR weakly. That is the whole point — an antifolate that
also inhibits human DHFR is toxic.

    Selectivity Index (kcal) = hDHFR_affinity - PfDHFR_affinity
    More positive = more parasite-selective. Negative = binds human harder = bad.

## 2. Targets and docking setup

Both receptors are already prepared and cached in the repo root.

| | PfDHFR (parasite, **minimize**) | hDHFR (human, **maximize**) |
|---|---|---|
| PDB ID | 1J3I | 1U72 |
| Raw structure | `1J3I.pdb` | `1U72.pdb` |
| Cleaned protein | `1J3I_clean.pdb` | `1U72_clean.pdb` |
| Vina receptor | `1J3I_clean.pdbqt` | `1U72_clean.pdbqt` |
| Box center (x, y, z) | `30.5, 5.2, 57.3` | `28.4, 13.0, -2.7` |
| Box size | `20.0, 20.0, 20.0` | `20.0, 20.0, 20.0` |
| Box anchored on | diaminotriazine head of WR99210 (HETATM `WRA`, chain A) | 2,4-diaminopteridine head of methotrexate (chain A) |

AutoDock Vina settings used throughout: `--exhaustiveness 8`, `--num_modes 9`,
`--seed 42` (deterministic — re-running reproduces scores exactly).

Receptor prep recipe (already applied): strip waters/heteroatoms/altlocs, keep
chain A, add hydrogens + Gasteiger charges via Open Babel, rigid receptor.

PyMOL box visualization:

    load 1J3I_clean.pdb
    pseudoatom box_center, pos=[30.5, 5.2, 57.3]
    show spheres, box_center

## 3. Priority list

### Tier A — selective AND potent (test these first)
Filter: selectivity > +0.25 kcal AND PfDHFR < -8.5 kcal/mol. 23 molecules qualify; top 12:

```
 1. O=C1Nc2ccccc2[C@]12N1CSC[C@H]1[C@H](c1cccs1)[C@]21Cc2ccccc2C1=O
     PfDHFR   -9.52 | hDHFR   -5.88 | selectivity  +3.64 | hERG 0.092 | t1/2 30h | score 0.718
 2. Cc1csc([C@H]2[C@@H]3CSCN3[C@@]3(C(=O)Nc4ccccc43)[C@@]23Cc2ccccc2C3=O)c1
     PfDHFR   -9.94 | hDHFR   -6.63 | selectivity  +3.31 | hERG 0.080 | t1/2 28h | score 0.691
 3. CN[C@@H](C)C(=O)N[C@H](C(=O)N1C[C@H]2C[C@H]3C[C@H]3N2C[C@H]1C(=O)N[C@@H]1CCOc2ccccc21)C1CCCCC1
     PfDHFR  -10.04 | hDHFR   -8.70 | selectivity  +1.34 | hERG 0.285 | t1/2 8h | score 0.499
 4. O=C1Nc2ccccc2[C@@]12C(c1ccc(Cl)cc1)C([N+](=O)[O-])C1CSCN12
     PfDHFR   -9.68 | hDHFR   -8.55 | selectivity  +1.13 | hERG 0.124 | t1/2 30h | score 0.663
 5. CS(=O)(=O)N1CC2(CCN(C(=O)C(COCc3ccccc3)NCc3cccc([N+](=O)[O-])c3)CC2)c2ccccc21
     PfDHFR  -10.96 | hDHFR   -9.98 | selectivity  +0.98 | hERG 0.237 | t1/2 7h | score 0.482
 6. CCOc1cc(S(=O)(=O)N2CCCC3(CCCCC3)C2)c(OCC)cc1-n1cnnn1
     PfDHFR   -9.70 | hDHFR   -8.92 | selectivity  +0.77 | hERG 0.134 | t1/2 3h | score 0.524
 7. O=C(CC1CCCC1)NNC(=O)c1ccccc1O
     PfDHFR   -8.67 | hDHFR   -7.95 | selectivity  +0.73 | hERG 0.622 | t1/2 3h | score 0.376
 8. O=C1N(Cl)C(=O)C(c2ccc(F)cc2)(c2ccc(Cl)cc2)N1Cl
     PfDHFR   -9.51 | hDHFR   -8.80 | selectivity  +0.71 | hERG 0.089 | t1/2 27h | score 0.701
 9. Nc1cc(CN2C(=O)N(Cc3ccc4[nH]ncc4c3)[C@H](Cc3ccccc3)[C@H](O)[C@@H](O)[C@H]2Cc2ccccc2)ccc1F
     PfDHFR   -8.79 | hDHFR   -8.19 | selectivity  +0.60 | hERG 0.934 | t1/2 5h | score 0.338
10. Cc1ccc2c(c1)[C@]1(C(=O)N2)C(c2ccc(Cl)cc2)C([N+](=O)[O-])C2CSCN21
     PfDHFR   -9.22 | hDHFR   -8.62 | selectivity  +0.59 | hERG 0.122 | t1/2 39h | score 0.686
11. CC(C)N1CCC[C@@]1(C)c1nc2c(C(N)=O)c(F)ccc2[nH]1
     PfDHFR   -9.20 | hDHFR   -8.65 | selectivity  +0.54 | hERG 0.984 | t1/2 14h | score 0.419
12. COc1ccc(CNC(=O)COc2ccc(C)nc2[N+](=O)[O-])cc1
     PfDHFR   -8.69 | hDHFR   -8.20 | selectivity  +0.49 | hERG 0.205 | t1/2 2h | score 0.500
```

### Tier B — selective, best all-round profile
Positive selectivity, ranked by the 5-objective composite score:

```
 1. Nc1cccc2c1C(=O)N/C2=C\c1ccccc1Cl
     PfDHFR   -9.04 | hDHFR   -8.73 | selectivity  +0.30 | hERG 0.041 | t1/2 46h | score 0.762
 2. Nc1nc(N)c(-c2ccc(Cl)cc2)c(C#Cc2cc(Cl)c(N)cc2F)n1
     PfDHFR   -9.73 | hDHFR   -9.68 | selectivity  +0.05 | hERG 0.117 | t1/2 45h | score 0.759
 3. Nc1cc(F)cc(/C=C2\NC(=O)c3c(Cl)cccc32)c1
     PfDHFR   -9.21 | hDHFR   -8.90 | selectivity  +0.31 | hERG 0.107 | t1/2 49h | score 0.751
 4. O=C1N/C(=C\c2cc(F)cc(Cl)c2)c2cccc(Cl)c21
     PfDHFR   -9.37 | hDHFR   -9.24 | selectivity  +0.13 | hERG 0.256 | t1/2 46h | score 0.730
 5. O=C1Nc2ccccc2[C@]12N1CSC[C@H]1[C@H](c1cccs1)[C@]21Cc2ccccc2C1=O
     PfDHFR   -9.52 | hDHFR   -5.88 | selectivity  +3.64 | hERG 0.092 | t1/2 30h | score 0.718
 6. CC(=O)Nc1ccc(NC(=O)c2cc(C)cc3c2-c2ccccc2C3=O)c(N)c1Br
     PfDHFR  -10.03 | hDHFR   -9.69 | selectivity  +0.34 | hERG 0.057 | t1/2 50h | score 0.717
 7. Nc1ccc(N)c(Sc2ccc(Cl)cc2)n1
     PfDHFR   -7.25 | hDHFR   -7.08 | selectivity  +0.17 | hERG 0.035 | t1/2 36h | score 0.717
 8. Nc1c(F)ccnc1Sc1ccc(Cl)cc1
     PfDHFR   -7.12 | hDHFR   -7.07 | selectivity  +0.05 | hERG 0.038 | t1/2 34h | score 0.716
 9. CC(=O)Nc1ccc(NC(=O)c2cc(O)cc3c2-c2ccccc2C3=O)c(N)c1Br
     PfDHFR   -9.88 | hDHFR   -9.83 | selectivity  +0.04 | hERG 0.028 | t1/2 51h | score 0.714
10. Nc1c(Cl)ccnc1Sc1ccc(Cl)cc1
     PfDHFR   -7.11 | hDHFR   -6.95 | selectivity  +0.16 | hERG 0.095 | t1/2 39h | score 0.712
```

### Tier C — highest composite score overall (CAUTION: mostly NOT selective)
These topped the optimizer's own ranking. Most bind human DHFR *harder* than the
parasite enzyme — included so you can see what the pipeline optimized toward:

```
 1. Nc1nc(N)c(-c2ccc(Cl)cc2)c(C#Cc2ccc(N)c(F)c2F)n1
     PfDHFR   -9.60 | hDHFR  -10.02 | selectivity  -0.42 | hERG 0.054 | t1/2 50h | score 0.770
 2. Nc1nc(N)c(-c2ccc(Cl)cc2)c(C#Cc2ccc(F)cc2)n1
     PfDHFR   -9.56 | hDHFR  -10.21 | selectivity  -0.65 | hERG 0.049 | t1/2 50h | score 0.770
 3. Nc1nc(N)c(-c2ccc(Cl)cc2)c(C#Cc2cccc(N)c2F)n1
     PfDHFR   -9.22 | hDHFR   -9.65 | selectivity  -0.44 | hERG 0.077 | t1/2 43h | score 0.769
 4. Nc1ccc(F)c(C#Cc2nc(Cl)nc(N)c2-c2ccc(Cl)cc2)c1F
     PfDHFR   -9.53 | hDHFR   -9.97 | selectivity  -0.45 | hERG 0.056 | t1/2 50h | score 0.765
 5. Nc1nc(N)c(-c2ccc(Cl)cc2)c(C#Cc2c(Cl)ccc(Br)c2F)n1
     PfDHFR   -9.87 | hDHFR  -10.11 | selectivity  -0.24 | hERG 0.100 | t1/2 52h | score 0.763
 6. Nc1nc(N)c(-c2ccc(Cl)cc2)c(C#Cc2c(F)ccc(F)c2F)n1
     PfDHFR   -9.77 | hDHFR  -10.41 | selectivity  -0.64 | hERG 0.066 | t1/2 47h | score 0.763
 7. Nc1cccc2c1C(=O)N/C2=C\c1ccccc1Cl
     PfDHFR   -9.04 | hDHFR   -8.73 | selectivity  +0.30 | hERG 0.041 | t1/2 46h | score 0.762
 8. Nc1nc(Br)nc(C#Cc2ccc(F)cc2)c1-c1ccc(Cl)cc1
     PfDHFR   -9.48 | hDHFR  -10.19 | selectivity  -0.71 | hERG 0.048 | t1/2 50h | score 0.761
 9. Nc1nc(N)c(-c2ccc(Cl)cc2)c(C#Cc2ccc(N)c(N)c2F)n1
     PfDHFR   -9.20 | hDHFR   -9.38 | selectivity  -0.19 | hERG 0.050 | t1/2 42h | score 0.761
10. Nc1nc(N)c(-c2ccc(Cl)cc2)c(C#Cc2cc(F)cc(N)c2N)n1
     PfDHFR   -8.88 | hDHFR   -9.23 | selectivity  -0.36 | hERG 0.058 | t1/2 42h | score 0.759
```

## 4. Reference compounds (positive controls)

Dock these alongside the candidates to calibrate. Known selective antifolate
antimalarials; pyrimethamine should land near -7 kcal/mol against PfDHFR.

    Pyrimethamine   CCC1=C(C(=NC(=N1)N)N)C2=CC=C(C=C2)Cl
    Cycloguanil     CC1(N=C(N=C(N1C2=CC=C(C=C2)Cl)N)N)C
    WR99210         CC1(N=C(N=C(N1OCCCOC2=CC(=C(C=C2Cl)Cl)Cl)N)N)C

## 5. Suggested checks

1. **Pose sanity.** Does the ligand actually sit in the folate pocket, or has it
   drifted to a surface groove? A good score with a nonsense pose is noise.
2. **Key contacts.** PfDHFR: Asp54, Ile14, Tyr170. The 2,4-diaminopyrimidine
   head should hydrogen-bond to Asp54 the way pyrimethamine's does.
3. **Selectivity mechanism.** For Tier A compounds, compare the pose in 1J3I vs
   1U72 — what does the parasite pocket accommodate that the human one does not?
   Pf has a larger pocket near residue 108; that is the usual selectivity handle.
4. **Resistance.** Check the S108N and C59R mutants — pyrimethamine resistance
   comes from these. Does the candidate still fit?
5. **Strain.** The spiro-oxindoles in Tier A are conformationally complex; check
   the docked conformer is not badly strained.

## 6. Caveats — please read

- Scores are **AutoDock Vina docking scores**, not measured affinities. Useful
  for ranking, unreliable as absolute numbers. Do not report them as Kd/IC50.
- The pipeline's headline "winner"
  (`Nc1nc(N)c(-c2ccc(Cl)cc2)c(C#Cc2ccc(N)c(F)c2F)n1`) has selectivity
  **-0.42 kcal** — it binds human DHFR slightly harder. It won on hypervolume,
  which rewards spread across all 5 objectives, not on selectivity. That is why
  Tier A is ordered by selectivity instead.
- Only **103 of 533** pooled Pareto molecules (19%) are selective at all;
  median selectivity is -0.31 kcal.
- The two docking objectives inside the optimizer are **ligand efficiency**
  (kcal per heavy atom). All numbers in this document are **raw kcal/mol**.
- Receptors are rigid, no explicit waters, no induced fit.
- ADMET values (hERG, half-life, Caco2) are **model predictions** from
  gradient-boosted models, not measurements.

## 7. Data files

| File | Contents |
|---|---|
| `matrix_results/summary.csv` | one row per configuration: hypervolume, Pareto size, runtime |
| `matrix_results/best_molecules.csv` | full Pareto row of each configuration's champion |
| `matrix_results/best_per_objective.csv` | winning molecule per objective, per configuration |
| `matrix_results/runs/<case>/pareto_front.csv` | full Pareto front for each run |
| `matrix_results/manifest.json` | commit, package versions, Vina version, provenance |

Column meanings in those CSVs: `PfDHFR_Docking` / `hDHFR_Docking` are ligand
efficiency; `*_kcal` are raw docking energies; `Selectivity_Index_kcal` is the
raw-kcal selectivity used here; `Half_Life_hours`, `hERG_Toxicity_Prob`,
`Caco2_logPapp` are ADMET predictions.
