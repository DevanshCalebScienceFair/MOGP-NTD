# F6b. Is the Pareto front enriched for known antifolate chemistry?

**Yes.** The front carries known-antifolate chemistry at 2.8 times the library
rate on a broad pharmacophore class and 17.7 times on close analogs of the four
reference drugs, p < 0.0001 either way. Every hit was chosen by the optimizer;
none came from the random initial design. The greedy baseline never recovers a
close analog in any of ten seeds.

Script: `analysis_scripts/antifolate_enrichment.py`.
Figure: `F6b_antifolate_enrichment.png`, from `analysis_scripts/fig_f6b.py`.
Data: `analysis_scripts/antifolate_enrichment.csv`,
`analysis_scripts/antifolate_enrichment_molecules.csv`.
No docking was run. `campaign_results/` was opened read-only and the docking
cache was never written or re-keyed.

---

## What is measured, and in what units

**Tanimoto similarity.** Dimensionless, 0 to 1. Computed on 2048-bit Morgan
fingerprints at radius 2, the same featurization the GP kernel uses, so this is
the similarity the model itself sees. 1.0 is identical bit vectors. Above about
0.4 on this representation usually means a shared scaffold rather than scattered
shared fragments.

**Enrichment factor (EF).** Dimensionless, a ratio of two rates. Numerator: the
fraction of a selected set that is a hit. Denominator: the fraction of the whole
29,678-molecule library that is. **EF = 1 means the selection is
indistinguishable from a random draw of the same size.** Because EF is a rate
ratio it does not inflate with set size, which is the defect of the maximum
similarity that `validate_known_actives.py` reports.

**Docking affinity**, kcal/mol, more negative is tighter.
**Selectivity Index**, kcal/mol, defined as hDHFR minus PfDHFR; positive means
the molecule binds the parasite enzyme more tightly, which is the direction a
useful antimalarial needs.

## The hit sets, and why the obvious one is wrong

| hit set | definition | n | base rate |
|---|---|---:|---:|
| **pharmacophore core** (primary) | matches 2,4-diaminopyrimidine or 2,4-diamino-1,6-dihydro-1,3,5-triazine by SMARTS | 231 | 0.778% |
| **close analog** | Tanimoto >= 0.4 to pyrimethamine, cycloguanil, WR99210 or P218 | 20 | 0.067% |
| core and Tanimoto >= 0.3 | the conjunction | 30 | 0.101% |

The pharmacophore core is the group that mimics the pterin of dihydrofolate and
anchors the DHFR active site. All four reference drugs match it. It is the
primary definition because it is chemically defined rather than tuned by a
threshold the author picked.

**A loose similarity cut is not a hit set.** At Tanimoto >= 0.25 there are 214
"near-neighbors", of which only 52 carry a diamino core while 59 carry nothing
but a chlorophenyl, a fragment present in 13.5% of the library. At that
threshold the statistic is a chlorophenyl detector. At Tanimoto >= 0.4 it is
clean: 18 of 20 members carry the core. Loose thresholds are therefore not
reported as headline numbers.

## The test

Ten seeds are the replicates and the **per-seed front is the unit**. Under the
null that selection ignores antifolate-likeness, each seed's front is a uniform
draw of its own size from the library; the null distribution of the total hit
count over the ten seeds comes from 20,000 Monte-Carlo draws.

The per-seed unit matters and this is not a detail. Pooling the ten fronts into
their union **penalizes a reproducible method**: a method that returns nearly the
same front every seed has a small union and therefore few distinct hits. MOGP's
union test gives p = 0.065 for that reason alone, while its per-seed test gives
p < 0.0001. The union is the wrong unit and is not used.

## Result

Hits are totals over the ten seeds. "exp" is the count expected by chance.

### Pharmacophore core (231 molecules, 0.778%)

| method | front n/seed | hits | exp | EF | p | seeds with >= 1 | from initial design | distinct molecules |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GPMOBO | 143.8 | 29 | 11.19 | 2.6 | **<0.0001** | 4/10 | 0 | 12 |
| **MOGP (ours)** | 168.0 | **37** | 13.08 | **2.8** | **<0.0001** | **10/10** | 0 | 5 |
| Greedy | 111.3 | 11 | 8.66 | 1.3 | 0.260 | 7/10 | 1 | 10 |

### Close analog, Tanimoto >= 0.4 (20 molecules, 0.067%)

| method | front n/seed | hits | exp | EF | p | seeds with >= 1 | from initial design | distinct molecules |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GPMOBO | 143.8 | 13 | 0.97 | 13.4 | **<0.0001** | 5/10 | 0 | 5 |
| **MOGP (ours)** | 168.0 | **20** | 1.13 | **17.7** | **<0.0001** | **10/10** | 0 | 2 |
| Greedy | 111.3 | 0 | 0.75 | 0.0 | 1.000 | 0/10 | 0 | 0 |

### Core and Tanimoto >= 0.3 (30 molecules, 0.101%)

| method | hits | exp | EF | p | seeds with >= 1 |
|---|---:|---:|---:|---:|---:|
| GPMOBO | 16 | 1.45 | 11.0 | **<0.0001** | 4/10 |
| **MOGP (ours)** | 29 | 1.70 | **17.1** | **<0.0001** | **10/10** |
| Greedy | 3 | 1.13 | 2.7 | 0.104 | 3/10 |

## Four audits, each able to kill the result

**A. Initial design.** The campaign draws 40 random molecules before the
optimizer acts, then runs 50 iterations of 5. A hit in rows 1 to 40 of
`evaluated.csv` was not found, it was handed over. **Zero of the 66 front hits
across all methods and thresholds came from the initial design**, with one
exception: one Greedy pharmacophore hit. Everything else is an optimizer choice.

**B. Size.** Docking affinity rises with molecular size and the reference drugs
are small (17, 17, 24 and 26 heavy atoms), so a size preference could manufacture
or mask the effect. In the size-matched null each front member is replaced by a
random library molecule within 2 heavy atoms of it, preserving the front's size
profile exactly.

| hit set | method | hits | null mean | p |
|---|---|---:|---:|---:|
| pharmacophore core | GPMOBO | 29 | 9.88 | **<0.0001** |
| pharmacophore core | MOGP | 37 | 12.38 | **<0.0001** |
| pharmacophore core | Greedy | 11 | 8.10 | 0.193 |
| close analog | GPMOBO | 13 | 1.26 | **<0.0001** |
| close analog | MOGP | 20 | 1.30 | **<0.0001** |
| close analog | Greedy | 0 | 0.85 | 1.000 |

Size is not the explanation.

**C. Where does the enrichment enter?** The acquisition chooses what to dock; the
Pareto filter chooses what to keep. Tested apart:

| hit set | method | EF from acquisition | p | EF from Pareto filter | p |
|---|---|---:|---:|---:|---:|
| pharmacophore core | GPMOBO | 2.44 | <0.0001 | 1.06 | 0.369 |
| pharmacophore core | MOGP | 2.22 | <0.0001 | 1.28 | 0.013 |
| pharmacophore core | Greedy | 1.06 | 0.407 | 1.19 | 0.290 |
| close analog | GPMOBO | 8.19 | <0.0001 | 1.64 | 0.010 |
| close analog | MOGP | 10.23 | <0.0001 | 1.73 | <0.0001 |
| close analog | Greedy | 0.00 | 1.000 | - | 1.000 |

**The acquisition does almost all of the work** (EF 8 to 10 on close analogs);
the Pareto filter adds about 1.7 on top. The claim belongs to the search, not to
the non-dominated filter.

**D. Distinct versus repeated.** Twenty hits across ten seeds could be two
molecules found ten times or twenty found once. MOGP: 20 hits, **2 distinct**
molecules, both in 10/10 seeds. GPMOBO: 13 hits, 5 distinct, none in more than
4/10 seeds. The two methods behave differently: MOGP is reproducible and narrow,
GPMOBO is broader and erratic.

## Head to head, paired on the ten shared seeds

Hits per seed on the pharmacophore core. Paired Wilcoxon signed-rank; with n = 10
the smallest attainable p is 0.0020.

| comparison | mean difference | MOGP ahead | p |
|---|---:|---:|---:|
| MOGP minus GPMOBO | +0.80 hits/seed | 6/10 | 0.522 |
| MOGP minus Greedy | +2.60 hits/seed | **10/10** | **0.0020** |

**This diagnostic separates Bayesian optimization from greedy selection. It does
not separate our model from the other Bayesian optimizer.** MOGP is ahead of
GPMOBO on every point estimate and is far more consistent (10/10 seeds against
4-5/10), but the paired test over ten seeds cannot distinguish them. Stating
otherwise would overclaim.

## The molecules recovered

| method | Tanimoto | closest drug | core | heavy atoms | seeds | PfDHFR | hDHFR | Sel. Index |
|---|---:|---|---|---:|---:|---:|---:|---:|
| MOGP | 0.732 | Pyrimethamine | yes | 25 | **10/10** | -10.20 | -10.30 | **-0.10** |
| MOGP | 0.610 | Pyrimethamine | yes | 21 | **10/10** | -7.61 | -8.48 | **-0.87** |
| GPMOBO | 0.732 | Pyrimethamine | yes | 25 | 2/10 | -10.20 | -10.30 | -0.10 |
| GPMOBO | 0.610 | Pyrimethamine | yes | 21 | 4/10 | -7.61 | -8.48 | -0.87 |
| GPMOBO | 0.575 | Pyrimethamine | yes | 17 | 1/10 | -9.66 | -9.36 | +0.30 |
| GPMOBO | 0.561 | Cycloguanil | yes | 17 | 1/10 | -9.80 | -9.50 | +0.30 |
| GPMOBO | 0.404 | Pyrimethamine | no | 19 | 5/10 | -8.81 | -7.67 | +1.15 |

The two MOGP recovers in every seed are genuine pyrimethamine analogs:
`CCc1nc(N)nc(N)c1-c1ccc(CNc2ccc(Cl)cc2)cc1`, a 2,4-diamino-6-ethyl-5-arylpyrimidine
with a 4-chloroanilinomethyl extension, and
`Nc1nc(N)c(-c2ccc(Cl)cc2)c(CCCCCO)n1`, a 2,4-diamino-5-(4-chlorophenyl)pyrimidine
with a pentanol tail. Both carry the core and both reproduce the
2,4-diaminopyrimidine / para-chlorophenyl arrangement of pyrimethamine itself.

## The caveat that has to go in the paper

**Both molecules MOGP recovers in every seed have a NEGATIVE Selectivity Index**,
-0.10 and -0.87 kcal/mol. They bind human DHFR as tightly or more tightly than
the parasite enzyme. They are potent, and they are not selective. The pipeline
rediscovers antifolate **potency** chemistry; it does not, in these two
molecules, rediscover antifolate **selectivity**. The three parasite-selective
analogs in the table (+0.30, +0.30, +1.15) are all GPMOBO's and each appears in
at most 5 of 10 seeds.

So the supportable claim for F6 is "the front is enriched for the chemistry of
known antifolates, by a factor of 2.8 to 17.7 depending on how a hit is defined,
and the enrichment is produced by the acquisition rather than the Pareto
filter". The claim "and the recovered molecules are selective" is **not**
supported and must not be made.

## Two defects found while doing this

**1. The same column name holds different quantities in different arms.** In the
GPMOBO arm `PfDHFR_Docking` is **ligand efficiency** (kcal/mol per heavy atom,
around -0.4) and `PfDHFR_Docking_kcal` is the raw affinity (around -10). In the
MOGP and Greedy arms it is the reverse: `PfDHFR_Docking` is raw kcal/mol and
`PfDHFR_Docking_LE` is the efficiency. Each arm's `*_Docking` column is the
objective that arm actually optimized, so the naming is deliberate, but reading
`*_Docking` across arms silently puts kcal/mol and kcal/mol-per-atom on one axis.
`matrix_report.py:53` does exactly that, and carries a comment asserting the
column IS raw kcal/mol. **The bug is latent, not triggered**: no existing matrix
run contains a `gpmobo` case, so no published figure is affected. It would fire
the first time one is added. `antifolate_enrichment.py` pins the raw column per
arm in `KCAL_COLUMN` so it cannot make the mistake.

**2. The docking cache is no longer a record of the campaign.** Of the 3,754
unique molecules the published campaign evaluated, **2,008 (53.5%) are absent
from the live cache**, which was re-keyed after the campaign ran under a new
oracle fingerprint. Scores that do appear in both differ slightly (-10.11 in the
cache against -10.20 in the campaign CSV for the same molecule), consistent with
a different receptor preparation. Nothing published is affected: the campaign
CSVs are intact and read-only, and they are the authoritative record. But any
future diagnostic must read scores from them, not from the cache, and arena
membership defined by cache presence no longer corresponds to what the campaign
searched.

## What is NOT shown

- **Not a prospective result.** These are library molecules rediscovered by a
  docking oracle, not new compounds and not experimental activity.
- **Not evidence that coregionalization pays.** The enrichment is in the
  acquisition, which both Bayesian optimizers share, and MOGP is not
  distinguishable from GPMOBO on this metric over ten seeds.
- **Not selectivity.** See above.
- **Enrichment against the library, not against a docking-score-matched null.**
  Antifolate analogs dock well against PfDHFR because they are DHFR ligands, and
  the optimizer selects molecules that dock well. That causal route is the
  mechanism, not a confound, but it means the result says the docking oracle
  recovers real DHFR pharmacology and the optimizer exploits it, which is a
  weaker and more honest statement than "the method rediscovers known drugs".
