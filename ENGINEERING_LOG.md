# Engineering log: a silent-failure audit of a multi-objective BO pipeline

**What this document is.** Every defect this project found in its own code, with the
number it reported before detection and the number it reported after. It exists
because the defects share a shape that is worth reporting in its own right: **the
code ran, raised nothing, and produced plausible numbers that measured something
other than what was claimed.** A crash is cheap. These were not crashes.

**Why it is a result and not an appendix.** Eighteen defects are recorded below.
**Eleven of them produced output that passed inspection.** Three changed a
published claim and one changed a GO/NO-GO decision. The distribution of how they
were caught is the finding: four were caught by a deliberate verification, five by
an unrelated crash or an accident, and the rest only because a number was compared
against something it should have equalled. Nothing in the test suite as originally
written would have caught any of the five worst.

**Definitions used throughout.**

| quantity | what it is | units |
|---|---|---|
| hypervolume (HV) | volume of objective space dominated by the observed Pareto front, against a fixed reference point | dimensionless, normalized |
| selectivity index (SI) | hDHFR affinity minus PfDHFR affinity; positive means parasite-selective | kcal/mol |
| Jaccard index | size of intersection over size of union of two nominated molecule sets | dimensionless, 0 to 1 |
| task correlation rho | Pearson correlation between two targets' affinities over the same molecules | dimensionless |
| enrichment factor (EF) | hit rate in the selected set over hit rate in the library | dimensionless, 1.0 means no enrichment |
| Vina affinity | predicted binding free energy; more negative is tighter. **A positive value is a failed pose search, not a weak binder.** | kcal/mol |

---

## Part 1. The ledger

Ordered by date found. "Published?" asks whether a figure or claim that was
circulated outside this repository was affected.

### 1. NADPH was being deleted from both receptors (2026-08-10)

`prepare_protein` stripped all HETATM records, which removed the NADPH cofactor
(resname `NDP`). NADPH is structural, not incidental: it lies inside both Vina
boxes and packs against the co-crystallized inhibitor at about 3.3 Angstrom.
Deleting it merged the folate and cofactor sites into one oversized cavity, so Vina
rewarded volume-filling over active-site recognition.

| | before | after |
|---|---|---|
| docked poses overlapping the NADPH position | **every pose, by 0.14 to 1.72 Angstrom** | none within 2.0 Angstrom |
| redock of WR99210 into 1J3I | not run | 0.98 Angstrom RMSD |
| redock of methotrexate into 1U72 | not run | 0.94 Angstrom RMSD |

**Every docking score computed before this is wrong, and not by a constant offset.**
The shift reorders molecules, so no rescaling recovers the old numbers.
**Published: yes**, everything in `matrix_results/` is superseded.
Guard: `test_docking_controls.py` asserts no pose comes within 2.0 Angstrom of
NADPH, and it reads NADPH from the **raw** PDB, not the prepared receptor, or it
would be a tautology.

### 2. Ligand efficiency was correcting a bug it did not fix (2026-08-11)

Ligand efficiency (affinity per heavy atom) was adopted as the objective because the
apo oracle's raw score was size-confounded. Fixing defect 1 removed most of that
confound, which left LE over-correcting in the opposite direction.

| | apo receptor | holo receptor |
|---|---|---|
| Spearman(raw score, heavy atoms) | -0.726 | **-0.257** |
| Spearman(LE, heavy atoms) | +0.891 | +0.880 |

LE was correcting about 3.4 times harder than the remaining distortion, with the
opposite sign. Reverted to raw kcal/mol; LE is still computed and reported as
`*_LE` columns. **Published: no.**

### 3. PAINS alone passes reactive electrophiles (2026-08-11)

PAINS targets assay interference. It passes acyl halides, epoxides, aziridines and
Michael acceptors, which are chemically disqualifying for a different reason.

| | before | after |
|---|---|---|
| best-scoring molecule in the library | **a 1,3-dichlorohydantoin**, a chloramine oxidant that cannot be a reversible binder | passes the reactive-group screen |

A reactive-group screen now runs alongside PAINS and rejects are logged to
`data/library/quality_rejected.csv` with their rejection class. **Published: no.**

### 4. The docking cache was keyed on `(smiles, target)` (2026-08-12)

Changing the receptor, box, exhaustiveness or seed returned stale scores silently.
**This is the mechanism that let defect 1 survive across runs.** A cache key that
omits the thing you changed turns a one-run bug into a permanent one.

Now keyed on `oracle_fingerprint`, which hashes the receptor **PDBQT** (downstream
of Open Babel, so it catches charge changes too) plus box centre and size,
exhaustiveness and seed. Prepared receptors carry sidecar `.stamp` files and rebuild
on mismatch. **Published: yes, indirectly**, via defect 1.

### 5. qNEHVI's `alpha` was never set, and the cost claim inverted (2026-09-01)

`acquisition.py` never passed `alpha` to qNEHVI, so it ran exact hypervolume
partitioning where BoTorch's own `get_default_partitioning_alpha(5)` returns `1e-3`.
The unset default was read as a property of the method.

| | before | after |
|---|---|---|
| wall clock, end to end | 8.14 h | **0.48 h (17.0x faster)** |
| projected campaign cost | 11.8x penalty vs baselines | 0.69 h per seed |
| rank per CPU-hour | **worst (0.035)** | **best (0.592)** |

**The ranking inverts.** The retracted claim was "MOGP is worst per CPU-hour by
11.8x". This also retired the accuracy-versus-compute trade-off in the discussion
and the 2,000-candidate pool cap, which existed only to work around the cost.
Honest cost of the fix, measured: `alpha=1e-3` needs 124 boxes and 0.084 s against
120,829 boxes and 119 s at 24 front points (1,412x), but it **overestimates
hypervolume by 18 to 90% and does not preserve the candidate ranking**
(Spearman 0.505, top-10 overlap 4 of 10).
**Published: yes.** `F4_compute_cost.png` is superseded and must not be used;
`F4_compute_cost_REVISED.png` is current.

### 6. `loop.py` had no `--seed` flag, so six seeds were one seed (2026-09-01)

The script exposed `--model`, `--posterior`, `--n-init` and eight other knobs. A
flag that does not exist is not an error: argparse rejects it only if it is passed
to argparse, and the sweep driver was building its own command strings.

| | before | after |
|---|---|---|
| distinct runs from a 6-seed campaign | **1** (all at default seed 42) | 6 |
| two "different" seeds, final HV | 0.3930 / 0.2728, byte-identical to each other's pair | distinct |
| same, top-20 selectivity | 2.3948 / 1.1373, likewise | distinct |

Caught only because two supposedly different seeds gave byte-identical output.
**The verification gate I had written checked the model name and would never have
caught this.** Rule adopted: whenever a script grows a sweep, diff two runs' outputs
and confirm they differ. **Published: no.**

### 7. `loop.py` had no `--acquisition-pool-size`, so direct runs were incomparable (2026-09-01)

Any direct `loop.py` run scored the whole roughly 26,600-molecule library every
iteration, while every `run_ablation.py` result used `--acquisition-pool-size 2000`.

| profiled share of wall clock | value |
|---|---|
| **acquisition** | **98.2% (178 of 181 min)** |
| GP training | 0.3% |
| docking | 0.2% |

About 6x slower and not comparable to any ablation number. **Published: no.**

### 8. `str.replace` no-ops on a missing anchor (two instances)

A scripted edit whose anchor string is absent succeeds and changes nothing.

| flag | symptom |
|---|---|
| `--acquisition-alpha` | accepted, recorded in the run config, **never forwarded to `select_batch`** |
| `--hdhfr-fraction` | accepted, **never reached `BOLoop.__init__`** |

Both ran, both logged the value the user asked for, neither applied it. Rule: any
scripted edit must assert its anchor exists **and** is unambiguous. **Published: no.**

### 9. One row filter serving two jobs would have inverted the experiment

`loop.py step()` used `isfinite(...).all(axis=1)` for both the GP training set and
the qNEHVI baseline. Under `--hdhfr-fraction` that discards every partly-labelled
molecule before the GP sees it, so **the asymmetric arm would have been a smaller
symmetric arm**, which is the exact opposite of what it exists to test.

Caught only because a 5-molecule smoke run drove hDHFR to zero observations and
tripped an unrelated guard. **At campaign scale there would have been no crash and
no symptom.** The correct split: the qNEHVI baseline must stay fully observed,
because hypervolume is undefined for a molecule missing an objective; the training
set need not. **Published: no.** This is the closest the project came to publishing
a null result caused by its own plumbing.

### 10. A hash that looked random and was not

Combining library index and seed with a bare XOR left adjacent seeds' subsets
**disjoint**.

| | before | after |
|---|---|---|
| measured overlap between adjacent seeds' subsets | **0%** | **25.4%** |
| expected | 25% | 25% |

Verified only because the overlap was measured instead of assumed. Now a SplitMix64
finalizer. **Published: no.**

### 11. A metric biased toward the arm it was comparing

`score_asym_campaign.py` ranked each arm's own measured molecules. The arms measure
different numbers of them.

| arm | candidates its top-20 was drawn from |
|---|---|
| full labelling | 246 |
| asymmetric | 99 |

**Both endpoints favoured the full arm before any science happened.** Fixed by
`nominate_and_score.py`, where both arms rank the same unmeasured library.
Rule adopted: before comparing two arms, ask what each metric is structurally biased
toward. **Published: no.**

### 12. A Modal local entrypoint has no torch, and the failure was caught (2026-11-15)

The Modal **local entrypoint** runs in the CLI's interpreter, not the container.
Importing `correlation_floor` there to reach `report()` raised, the exception was
caught by a broad handler, **the CSV was written and the summary table was silently
skipped**. Valid output, no error, nothing measured.

Fixed by splitting the reporting half into `analysis_scripts/correlation_floor_report.py`,
which imports only numpy, pandas and scipy. **Lesson for any Modal script: the local
entrypoint and the container do not share an environment.** **Published: no.**

### 13. `*_Docking` means different quantities in different arms (2026-11-16)

| arm | `*_Docking` holds | raw kcal/mol lives in |
|---|---|---|
| GPMOBO | ligand efficiency (kcal/mol per heavy atom) | `*_Docking_kcal` |
| MOGP, Greedy | raw kcal/mol | `*_Docking` |

Each arm's `*_Docking` is the objective that arm optimized. Reading it across arms
puts kcal/mol and kcal/mol-per-atom on one axis. `matrix_report.py:53` did this and
asserted in a comment that the column **is** raw kcal/mol.

**Four consumers were mixing units, not the two the column constants named.**
`normalized_scores` and `best_by_objective` also read `*_Docking` by name, and would
have normalized a ligand efficiency of -0.42 against a kcal/mol range of [-11, -5],
scoring a GPMOBO case's **strongest** binder as the worst possible value.

| measured on a synthetic mixed sweep | before | after |
|---|---|---|
| `best_score`, GPMOBO case | **0.602** | 0.709 |
| `best_score`, MOGP case holding the identical molecules | 0.709 | 0.709 |

**Latent, not triggered: no existing matrix run has a `gpmobo` case, so no published
figure is affected.** It fires the first time one is added. Fixed with
`resolve_kcal_column(df, target)` per case, resolved from the columns present on disk
rather than from the case label, applied once in `summarize`. The rewrite is
lossless, it is loud (a `UserWarning` plus a `DOCKING UNITS` block when both columns
are present and differ by more than 2.0x; the real gap is about 25x, the heavy-atom
count), `summary.csv` now states its own units in new provenance columns, and 11
tests in `test_matrix_report_units.py` pin the behaviour. Verified a no-op against a
copy of the 38 published cases. **Published: no.**

### 14. The docking cache is no longer a record of the campaign (2026-11-16)

| | count | share |
|---|---|---|
| molecules the published campaign evaluated | 3,754 | |
| **absent from the live cache** | **2,008** | **53.5%** |

The cache was re-keyed under a new oracle fingerprint after the campaign ran, which
is the correct behaviour from defect 4 working as designed. Scores present in both
differ (-10.11 cache against -10.20 campaign CSV for one molecule), consistent with a
different receptor prep. **Published numbers are unaffected**, but any analysis that
defines arena membership by cache presence no longer matches what the campaign
searched. **Read campaign scores from `campaign_results/*/evaluated.csv`, never from
the cache.**

### 15. The nomination endpoint was not reproducible (2026-11-16)

GPyTorch initialises `IndexKernel.covar_factor` from `torch.randn`. `MOGPHadamardICM`
uses it, `train_mogp_hadamard` constructs it, and `nominate_and_score.py` never
called `torch.manual_seed`. Measured over 12 cells x 5 repeats x 2 regimes, 120 GP
fits, zero docking calls.

| | unseeded (as published) | seeded (the fix) |
|---|---|---|
| cells returning a different top-20 | **6 of 12** | **0 of 12** |
| mean pairwise Jaccard | 0.973 | **1.000** |
| worst cell | 0.818 (17 of 20 stable) | 1.000 |
| predicted-SI spread within a cell | 0.024 kcal/mol | **0.000** |

0.024 kcal/mol is 5.6% of the 0.423 kcal/mol mean gap between arms, so **`mean_SI`
survives this and `best_SI` does not**: a maximum over 20 molecules is not robust to
a swap. Asymmetric seed 3's committed `best_SI` is 4.24 kcal/mol and the molecule it
drops on some repeats is worth +6.19, a 46% swing from one RNG draw.

Fixed with `torch.manual_seed(seed)` at `nominate_and_score.py:93`. The fix was
**measured** sufficient, not assumed: every seeded repeat is bit-identical, so there
is no second randomness source on that path. **Published: yes**, for `best_SI` only.

### 16. Selenomethionine mistaken for a bound ligand (2026-11-16)

Found in a tool written the same day to audit receptors, which is the point: new
verification code is code. A modified amino acid (MSE selenomethionine, SEP, TPO)
carries `het_flag "H"` in an mmCIF file but is chemically part of the polypeptide.

| | before | after |
|---|---|---|
| 3TRK's reported box ligand | **MSE, 8 heavy atoms** | none |
| consequence if trusted | a docking box centred on the protein backbone | rung 2 correctly blocked |

Fixed with `_is_real_ligand()`, which asks `gemmi.find_tabulated_residue(name)`
whether the residue is an amino acid or nucleotide rather than consulting a
hand-maintained exclusion list. **The corrected answer, that both CHIKV nsP2
candidates are apo, is the finding that blocked rung 2.** **Published: no.**

### 17. One non-physical row flipped a GO/NO-GO (2026-11-16)

One molecule in the 120-molecule rung-1 pilot scored **+4.89 kcal/mol** on human
DHFR, in a column whose other values span -11.30 to -5.23. A positive Vina affinity
is a failed pose search formatted like a measurement. Pearson correlation averages
products of z-scores, so a point that far out on one axis and ordinary on the other
is a lever.

| | unfiltered | artifact-filtered |
|---|---|---|
| n | 113 | 112 |
| rho | +0.544 | **+0.654** |
| 95% CI lower bound | +0.418 | **+0.458** |
| against the 0.45 correlation floor | **fails** | **clears** |

The project already excludes positive affinities everywhere (`PF_MAX = -7.0`,
`HD_MAX = 0.0`). The new pilot harness had reimplemented the analysis without
carrying the filter over. **An existing safeguard not inherited by new code is the
single most common shape in this ledger.** The filter now runs inside the harness,
before the correlation, and prints how many rows it dropped. **Published: no**, the
pilot was a gate, not a result.

### 18. 0.612 kcal/mol was used as a ceiling when it is about a p90 (2026-11-16)

This one is mine rather than the code's. From a 2-molecule smoke test I reported a
cross-machine docking discrepancy of 0.094 kcal/mol and suggested the project's
cross-machine ban might be over-strict. At n=24 the distribution is heavy-tailed.

| statistic, absolute delta | kcal/mol |
|---|---|
| median | 0.031 |
| p75 | 0.198 |
| **p90** | **0.840** |
| p95 | 1.350 |
| **max** | **1.604** |
| exceeding the recorded 0.612 | **3 of 24** |

Most molecules reproduce to 0.03 kcal/mol and about one in eight is off by more than
0.612. **The project's 0.612 figure has been cited as a bound and is approximately a
90th percentile.** The failure is occasional and large, so it is not correctable by
subtracting an offset, and version-matching the image (Vina 1.2.7, Python 3.11,
RDKit 2024.03.6) did not fix it. The cross-machine ban stands and was understated.
Extended to n=48 by the n=300 run. **Published: yes, as an understatement.**

---

## Part 2. The taxonomy

Each defect assigned to the mechanism that made it silent rather than loud.
**The classes are not mutually exclusive and the count column sums to 19, not 18:**
defect 2 is both a mis-aimed correction and a biased metric, defect 16 is both an
uninherited safeguard and a cheminformatics gap, and defect 17 is both an
uninherited safeguard and a failure code read as data. Those overlaps are the
interesting cases, so collapsing them would lose information.

| class | count | defects | what makes it silent |
|---|---|---|---|
| **Accepted but unwired** | 3 | 6, 8 (x2) | the value is logged where the user looks and dropped before it is used |
| **An existing safeguard not inherited by new code** | 4 | 13, 15, 16, 17 | the repository knows the rule; the new file does not |
| **A key or filter that omits the thing that changed** | 3 | 4, 9, 14 | correct code, wrong scope |
| **A library default read as a method property** | 1 | 5 | nothing is missing, so nothing warns |
| **A metric biased toward its own conclusion** | 2 | 11, 2 | the number is computed correctly and answers a different question |
| **A correction applied where the bug was elsewhere** | 1 | 2 | the symptom improves, which is taken as confirmation |
| **A quantity mistaken for a measurement** | 2 | 17, 18 | a failure code and a tail percentile both look like data |
| **Chemistry a cheminformatics default does not cover** | 2 | 3, 16 | the standard filter is standard for a different purpose |
| **Environment split across a process boundary** | 1 | 12 | the broad exception handler is the bug, not the import |

### How each was caught

| route | count | defects |
|---|---|---|
| a deliberate verification designed to catch it | 4 | 10, 13, 15, 16 |
| an unrelated crash or a lucky smoke-test configuration | 3 | 9, 12, 3 |
| a number compared against one it should have equalled | 5 | 1, 5, 6, 14, 17 |
| profiling for an unrelated reason | 2 | 7, 4 |
| re-measuring my own earlier claim at larger n | 2 | 18, 2 |
| reading code for a different task | 2 | 8, 11 |

**Four of eighteen were caught by a test designed to catch them.** That ratio is the
honest headline of this document, and it did not improve by writing more tests; it
improved by adopting the four rules in part 4.

---

## Part 3. The six retractions

Claims this project made and then withdrew on its own measurements. Listed because a
retraction with the number attached is more useful than a claim that was never wrong.

| # | the claim | status | the number |
|---|---|---|---|
| 1 | "MOGP is worst per CPU-hour by 11.8x" | **false** | 8.14 h to 0.48 h = 17.0x; best per CPU-hour (0.592), not worst (0.035). Defect 5. |
| 2 | "The joint posterior fix caused the improvement" | **false** | pure alpha effect +0.0060, pure posterior effect **-0.0008**. The joint posterior is still required, because without it the ICM cannot affect selection at all, and it is 1.8x faster, but that speed is mostly a molecule dedup bundled into the same code path. |
| 3 | "Coregionalization buys sample efficiency" | **false at 10 seeds** | ICM leads 197 of 400 matched checkpoints = 49%. Per seed 38, 26, 30, 21, 22, 11, **0**, **49**. Seed 0 was noise. Every endpoint null: final HV p=0.375, AUC p=0.846, molecules-to-target p=0.22 to 0.64. |
| 4 | "Representative compounds are the sensible default for the expensive second assay" | **false** | it is the one selection rule reliably **worse than random**, on both accuracy and ranking. F18. |
| 5 | "0.612 kcal/mol bounds the cross-machine docking discrepancy" | **understated** | p90 0.840, max 1.604, 3 of 24 above 0.612. Defect 18. |
| 6 | "The nominated top-20 is a deterministic function of the campaign" | **false** | 6 of 12 cells differ across repeats; `best_SI` swings 46%. Defect 15. |

Two further self-corrections are recorded in the F19 write-up: I predicted a bias
direction for the synthetic task twice and neither held, and the raw p-value at low
rho reads as "sharing is harmful" but does not survive Holm correction.

---

## Part 4. The four rules that came out of this

1. **After wiring any flag, verify the value arrives where it is consumed and
   changes what it should.** Acceptance is not arrival. Defects 6 and 8.
2. **Whenever a script grows a sweep, diff two runs' outputs and confirm they
   differ.** Defect 6 produced six identical runs and a verification gate that
   checked the model name instead.
3. **Before comparing two arms, ask what each metric is structurally biased
   toward.** Defect 11 had both endpoints favouring one arm before any science
   happened.
4. **New analysis code does not inherit the repository's filters. Carry them over
   explicitly, or import them.** Four defects, including the one that flipped a
   decision.

A fifth, narrower rule for this project: **a cache key must contain everything that
changes the value.** Defect 4 is why defect 1 survived, and defect 14 is defect 4's
fix working as intended.

---

## Part 5. What is now guarded, and what is not

| guard | covers | file |
|---|---|---|
| no pose within 2.0 Angstrom of NADPH, read from the raw PDB | defect 1 | `test_docking_controls.py` |
| crystal-pose redock, 0.98 and 0.94 Angstrom | defect 1 | `validate_redock.py` |
| oracle fingerprint over receptor PDBQT, box, exhaustiveness, seed, with `.stamp` sidecars | defect 4 | `docking.py` |
| per-case raw-kcal column resolution plus a loud unit-disagreement warning, 11 tests | defect 13 | `test_matrix_report_units.py` |
| `torch.manual_seed(seed)` before the fit | defect 15 | `nominate_and_score.py:93` |
| residue-table lookup instead of a het-flag heuristic | defect 16 | `modal_receptor_audit.py` |
| artifact filter inside the harness, before the statistic, with a dropped-row count | defect 17 | `modal_rung1_correlation_pilot.py` |
| self-checking pool reproduction against the expected 26,660 survivors | future library drift | `modal_rung1_correlation_pilot.py` |
| import closure computed, not guessed, so a no-docking container cannot dock | future Modal scripts | `modal_endpoint_reproducibility.py` |
| per-chunk checkpoint to a Volume plus resume | client disconnect and container preemption | `modal_rung1_correlation_pilot.py` |

**Not guarded.** Defects 2, 5, 7, 9, 10, 11, 12, 14 and 18 are fixed but have no
regression test. Of those, **9 and 11 are the two that could recur and would again be
silent**, because both are questions about whether a metric or a filter is scoped to
the right set, and that is not a property a unit test can assert without being told
the right answer first.

---

## Part 6. For the paper

The claim this log supports is narrow and defensible: **in a multi-objective Bayesian
optimization pipeline with a physics-based oracle, the dominant failure mode is not
incorrect mathematics but correctly-executing code measuring the wrong quantity, and
the standard testing discipline does not catch it.** Four of eighteen defects were
caught by a test written to catch them. Three changed a published claim; one would
have inverted an experiment's result; one changed a GO decision.

The useful numbers for F5 panel d are the before-and-after pairs in part 1, and the
two count tables in part 2. The honest framing for the discussion is that this
project's six retractions are a feature of having measured its own claims twice, not
evidence of unusual carelessness, and that the four rules in part 4 are cheap enough
that a reviewer can adopt them.

**Source of record.** `CLAUDE.md` holds the operational detail per defect and is
authoritative where this document and it disagree; `FIGURES.md` holds figure status
including which files are superseded.
