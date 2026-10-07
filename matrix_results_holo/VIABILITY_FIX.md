# Docking-viability fix: what changed in the reported artifacts

**Date applied:** 2026-10-06 · **Branch:** `feat/numpy-bo-rewrite` · **Code baseline:** `9f5500b`

This note records a correction to how a *failed* docking calculation enters the
optimization objective, and exactly which published files were regenerated as a
result. It is written to be quoted directly in a methods or limitations
section.

Everything below is a measured number. The harness used to produce it is
validated in [§7](#7-how-this-was-verified): it reproduces all 53 archived
`pareto_front.csv` files **byte-for-byte** from their `evaluated.csv` before
applying any change, so the "before" column is literally the published record
rather than a re-derivation of it.

---

## 1. The defect

The pipeline optimizes five objectives. The second, `hDHFR_Docking` (binding to
*human* dihydrofolate reductase), is **maximized**, because weak human binding
is what makes a compound selective for the parasite enzyme over the host one.

AutoDock Vina always returns a number. When no favourable binding mode exists,
it returns the best pose it could find anyway — and that score can be near zero
or **positive**. A positive affinity is not weak binding; it is a net
*unfavourable* interaction, i.e. the absence of a binding mode.

Because `hDHFR_Docking` is maximized and normalized against a fixed
`[-11, -5] kcal/mol` cube, a positive score clipped to the top of that cube and
scored **1.0 — the best possible value on that axis**. A molecule that failed to
dock against human DHFR was therefore scored as *maximally selective*, and the
derived, human-facing Selectivity Index
(`SI = hDHFR_kcal - PfDHFR_kcal`) inherited the same inflation.

Measured incidence, pooled over all 53 archived benchmark runs (4,165 evaluated
molecules):

| | count |
|---|---|
| rows with `hDHFR_Docking` ≥ 0 | **15** (9 distinct molecules) |
| rows with `PfDHFR_Docking` ≥ 0 | **1** (the same row as one of the above) |
| rows with an outright failed dock (`NaN`) | 104 per target |
| of the 15, how many sat on their run's published Pareto front | **15 of 15** |

## 2. The rule, and the threshold

> **A dock is non-viable when its best parsed Vina affinity is ≥ 0.0 kcal/mol
> (or absent). A non-viable dock receives that objective's *worst* value instead
> of its best. The raw kcal/mol measurement is preserved unchanged; only what
> the *objective* says is altered.**

The rule is applied symmetrically, and the worst value is **derived** from each
objective's frozen bounds and sign rather than hard-coded:

| objective | direction | worst raw value | normalized |
|---|---|---|---|
| `hDHFR_Docking` | maximize (+1) | `-11.0` (bottom of the cube) | 0.0 |
| `PfDHFR_Docking` | minimize (−1) | `-5.0` (top of the cube) | 0.0 |

Opposite raw ends, identical normalized 0.0 — the reference corner — so a
substituted value earns no hypervolume and is dominated on that axis by every
real observation.

An outright failed dock (`NaN`) stays `NaN`: it is still dropped from GP
training, from the reported front and from the hypervolume, exactly as before.
It is not resurrected at the worst corner.

### Why ≥ 0.0, and why it is safe

**It is physically the right line.** A negative Vina score is favourable
binding. A score ≥ 0 means the best pose found is net unfavourable: no binding
mode was located. That is absence of evidence, not evidence of weak binding.

**It excludes no known active.** The weakest reference antifolate anywhere in
this repo's validation set is P218 at −8.548 kcal/mol (PfDHFR); pyrimethamine
−9.087 / −8.992, cycloguanil −9.110, WR99210 −8.818. The margin from the
weakest to the threshold is 8.55 kcal/mol — wider than the entire 6.0-kcal/mol
normalization cube.

**It is not knife-edge.** Measured on the 4,165 pooled archived rows, the
threshold sits in the middle of a hole in the data, not on an edge:

| target | greatest value below 0 | least value at/above 0 | empty band |
|---|---|---|---|
| `hDHFR_Docking` | −1.0050 | +0.6735 | **1.6785 kcal/mol wide** |
| `PfDHFR_Docking` | −0.9254 | +4.2130 | **5.1384 kcal/mol wide** |

Twenty-five thresholds spanning the joint empty band (−0.9254, +0.6735) select
an **identical** set of rows on this data (25/25). The choice of 0.0 within that
band is therefore immaterial to every number in this document.

### What was deliberately *not* changed

The frozen `[-11, -5]` normalization bounds
(`spec/bounds/objective_bounds.frozen.json`), the objective names, and the
objective signs are all unchanged, and are asserted unchanged by the test
suite. This is a separate, independent physical threshold, not an adjustment of
the normalization frame. Three known, out-of-scope issues — the width of the
`[-11, -5]` cube, the loop's docking-saturation margin, and the acquisition
function's forced-diagonal posterior — are likewise untouched.

## 3. Artifacts regenerated, and where the pre-fix copies live

**Every file listed here was copied to `<stem>.prefix-viability.csv` beside
itself before being touched**, and each copy was verified byte-identical to its
source at the moment of copying. The pre-fix version of every published number
therefore survives on disk.

| artifact | files | regenerated? | pre-fix copy |
|---|---|---|---|
| `benchmark_*/<method>/seed_*/pareto_front.csv` | 53 | yes — **14 changed content**, 39 came out byte-identical | `pareto_front.prefix-viability.csv` (all 53) |
| `matrix_results_holo/leads_annotated.csv` | 1 | yes — 2 of 50 Selectivity Indices gated, 2 columns added | `leads_annotated.prefix-viability.csv` |
| `matrix_results_holo/best_per_objective.csv` | 1 | **yes for 18 of 38 cases** from the sweep's own logs; the other 20 corrected but not re-selected (see §6) | `best_per_objective.prefix-viability.csv`, plus `best_per_objective.annotated-uncorrected.csv` |
| `matrix_results_holo/summary.csv` | 1 | **yes for 18 of 38 cases** from the sweep's own logs; the other 20 corrected but not re-selected (see §6) | `summary.prefix-viability.csv`, plus `summary.annotated-uncorrected.csv` |
| `matrix_results_holo/best_per_objective.viability_ledger.csv` | 1 | new — per-row audit trail for the above (see §6.3) | n/a (post-fix only) |

**Not regenerated, by design:**

- `evaluated.csv` and `history.csv` (all 53 runs) are the **record of what was
  measured** and are deliberately untouched. Note the consequence, because it is
  the key to reading this archive: an archived `evaluated.csv`'s `*_Docking`
  columns **predate** the substitution, so they are the *raw measurement*, and
  replaying the hypervolume from one reproduces the **pre-fix** number in
  `history.csv`. That is correct and self-consistent, not a discrepancy.
- `benchmark_*/benchmark_seeds_aggregate.csv`,
  `benchmark_*/benchmark_seeds_significance.csv` and
  `benchmark_*/benchmark_seeds.png` are exact functions of those frozen
  `history.csv` files. Overwriting them would leave each sweep directory holding
  a summary that could no longer be derived from its own inputs. The post-fix
  versions are therefore written **beside** them, as
  `*.postfix-viability.{csv,png}`, and nothing is overwritten. Each sweep also
  gains `hypervolume_before_after.postfix-viability.csv`, the per-run table
  behind [§5](#5-hypervolume-before-and-after).

### Telling pre- from post-fix output

1. **The presence of a `.prefix-viability.csv` sibling** marks a file as
   regenerated; that sibling is the pre-fix version.
2. **A `.postfix-viability.*` file** is a corrected summary published alongside
   an original that was left in place.
3. A regenerated `pareto_front.csv` keeps the published 10-column header. This
   is intentional: `spec/hv/fixtures/output_column_contract.json` is a measured
   census of the archive's headers, and `test_core_hv.py` asserts every archived
   file matches it — adding `*_raw_kcal` / `*_viable` columns here turned that
   test red. It is also unnecessary: on a front from which every non-viable row
   has been removed, the two readings of a `*_Docking` column **coincide**,
   because for a viable dock the objective *is* the raw measurement. The
   10-column file is correct under both the pre-fix and the post-fix semantics.
   Freshly generated runs do carry the four extra columns; archived ones keep
   the published contract.
4. **One deliberate inconsistency, so it is not mistaken for an error.** For the
   14 affected runs, `pareto_front.csv` now has **fewer rows** than the final
   `pareto_size` in the same directory's `history.csv` (e.g.
   `thicc/greedy/seed_1`: `pareto_size` 41, front file 40 rows). That is
   expected and intended: `history.csv` is the frozen pre-fix record and the
   front is regenerated post-fix. The two agree in the other 39 runs. The
   post-fix `pareto_size` for every run and iteration is in each sweep's
   `benchmark_seeds_aggregate.postfix-viability.csv`.

## 4. Pareto fronts and the lead table

### 4.1 Published Pareto fronts

| | before | after |
|---|---|---|
| pooled front rows (53 runs) | 2,422 | **2,407** |
| rows removed | | **−15** |
| rows that joined a front | | **0** |
| runs whose front grew | | **0** |
| runs whose front shrank | | **14 of 53** |

Every one of the 15 removed rows is a non-viable dock, and no viable row was
displaced from any front — both asserted per run. The 14 affected files:

| run | front before → after |
|---|---|
| `benchmark_thicc_2026-08-14/greedy/seed_1` | 41 → 40 |
| `benchmark_thicc_2026-08-14/mogp/seed_1` | 37 → 36 |
| `benchmark_thicccc_2026-08-16/greedy/seed_1` | 52 → 51 |
| `benchmark_thicccc_2026-08-16/greedy/seed_7` | 49 → **47** (two rows) |
| `benchmark_thicccc_2026-08-16/greedy/seed_8` | 47 → 46 |
| `benchmark_thicccc_2026-08-16/mogp/seed_0` | 62 → 61 |
| `benchmark_thicccc_2026-08-16/mogp/seed_1` | 53 → 52 |
| `benchmark_thicccc_2026-08-16/mogp/seed_3` | 52 → 51 |
| `benchmark_thicccc_2026-08-16/mogp/seed_7` | 56 → 55 |
| `benchmark_thicccc_2026-08-16/mogp/seed_9` | 48 → 47 |
| `benchmark_thicccc_2026-08-16/random/seed_0` | 48 → 47 |
| `benchmark_thicccc_2026-08-16/random/seed_1` | 53 → 52 |
| `benchmark_thicccc_2026-08-16/random/seed_7` | 52 → 51 |
| `benchmark_thicccc_2026-08-16/single_obj/seed_8` | 45 → 44 |

The nine distinct offending molecules, with the raw affinities that put them on
a front:

| PfDHFR kcal | hDHFR kcal | pre-fix SI | front rows |
|---|---|---|---|
| −1.850 | **+6.191** | 8.041 | 6 |
| −8.484 | **+1.199** | 9.683 | 2 |
| −0.925 | **+10.730** | 11.655 | 1 |
| −6.103 | **+5.413** | 11.516 | 1 |
| −8.226 | **+4.258** | 12.484 | 1 |
| **+4.213** | **+6.244** | 2.031 | 1 |
| −3.914 | **+5.054** | 8.968 | 1 |
| −6.884 | **+0.998** | 7.882 | 1 |
| −7.795 | **+0.674** | 8.469 | 1 |

After the fix, every published front row has a viable dock on **both** targets:
of 2,407 rows, zero have a non-negative affinity, zero have a `NaN` affinity,
and zero have a `NaN` Selectivity Index.

### 4.2 `leads_annotated.csv` — the ranking defect

This is the table the defect visibly corrupted. It holds 50 pooled Pareto
molecules, each re-docked against both targets, ranked by Selectivity Index.
**The published table was wrong at rank 1 and rank 2**: both were molecules
whose own hDHFR re-dock found no binding mode.

Under the fix, the Selectivity Index of such a row is `NaN`, not a number. A
difference of binding free energies is **undefined** when one of the two binding
modes was never found — it is neither large nor small. Reporting it from the raw
positive score is the defect itself; reporting it from the *substituted* value
would be misleading in the opposite direction, posting a strongly
anti-selective SI that no instrument measured. `NaN` is already this repo's
representation of a missing docking measurement, and pandas sorts `NaN` last by
default, so a failed dock can never again top a selectivity-ranked table.

**Published (pre-fix) top 5 by Selectivity Index:**

| rank | SI | PfDHFR kcal | hDHFR kcal | class | viable? |
|---|---|---|---|---|---|
| 1 | 8.2000 | −2.506 | **+5.694** | WEAK | **no** |
| 2 | 6.8412 | −6.448 | **+0.3932** | SIZE-EXCLUSION | **no** |
| 3 | 5.4140 | −6.461 | −1.047 | WEAK | yes |
| 4 | 5.0400 | −6.303 | −1.263 | WEAK | yes |
| 5 | 4.5030 | −8.253 | −3.750 | SIZE-EXCLUSION | yes |

**Corrected (post-fix) top 5 by Selectivity Index:**

| rank | SI | PfDHFR kcal | hDHFR kcal | class | was |
|---|---|---|---|---|---|
| 1 | 5.4140 | −6.461 | −1.047 | WEAK | rank 3 |
| 2 | 5.0400 | −6.303 | −1.263 | WEAK | rank 4 |
| 3 | 4.5030 | −8.253 | −3.750 | SIZE-EXCLUSION | rank 5 |
| 4 | 4.2440 | −10.810 | −6.566 | SIZE-EXCLUSION | rank 6 |
| 5 | **4.1130** | **−10.210** | **−6.097** | **RECOGNITION** | **rank 7** |

The two non-viable rows leave the ranking entirely (SI → `NaN`, sorted last).
Everything else shifts up two places, and the best genuine lead — the only
RECOGNITION-class molecule in the top ten, and rank 1 of `FINAL_LEADS.csv` —
rises from **rank 7 to rank 5**.

Verified on the regenerated file: 48 of 50 Selectivity Index values are
**bitwise identical** to the published column; exactly 2 became `NaN`, and the
`NaN` set is exactly the non-viable set. Every other column — including both raw
affinity columns, all geometry columns and the mechanism class — is bitwise
unchanged, and the row order is unchanged. Two derived columns,
`PfDHFR_viable` and `hDHFR_viable`, were added so the file states its own
generation. Re-applying `annotate_leads.classify` changes **0 of 50**
mechanism labels, and the RECOGNITION set is identical (8 molecules, all
viable) — which is what makes §4.3 hold.

### 4.3 `FINAL_LEADS.csv` is unchanged — proven

`FINAL_LEADS.csv` is the headline deliverable. It is built from the
RECOGNITION-class rows of `leads_annotated.csv`, sorted by PfDHFR potency.

**Proof by reconstruction.** `build_final_leads.py` was re-run against the
*corrected* `leads_annotated.csv`. Its output is **byte-identical** to the
published `FINAL_LEADS.csv`:

```
md5 (published, before any change) : 011f504ad41e8ffdbe50752d889b78e6
md5 (rebuilt from corrected table) : 011f504ad41e8ffdbe50752d889b78e6
```

**Proof by content.** All 8 rows:

- are RECOGNITION class, and are exactly the RECOGNITION rows of the corrected
  `leads_annotated.csv`, in the same order;
- have `dock_is_viable` true on **both** targets;
- lie in PfDHFR `[-10.210, -8.143]` and hDHFR `[-8.505, -6.097]` kcal/mol;
- include **zero** non-negative affinities and **zero** values clipped at either
  wall of the `[-11, -5]` cube;
- satisfy `Selectivity_Index_kcal == round(hDHFR - PfDHFR, 3)` exactly.

The closest any of the 16 affinities comes to the viability threshold is
**6.097 kcal/mol**. The headline lead table was never affected by this defect
and did not move.

## 5. Hypervolume, before and after

Mean final hypervolume per sweep and per method. Full per-run numbers are in
each sweep's `hypervolume_before_after.postfix-viability.csv`.

| sweep | method | seeds | mean HV before | mean HV after | Δ | % | rank before | rank after |
|---|---|---|---|---|---|---|---|---|
| 5way_densify | MOGP | 1 | 0.349380911 | 0.349380911 | 0 (exact) | 0.0000% | 1 | 1 |
| 5way_densify | GP-MOBO | 1 | 0.190962891 | 0.190962891 | 0 (exact) | 0.0000% | 2 | 2 |
| 5way_densify | Greedy Filter | 1 | 0.106124140 | 0.106124140 | 0 (exact) | 0.0000% | 3 | 3 |
| 5way_densify | Single-Obj BO | 1 | 0.067942838 | 0.067942838 | 0 (exact) | 0.0000% | 4 | 4 |
| 5way_densify | Random Search | 1 | 0.065201377 | 0.065201377 | 0 (exact) | 0.0000% | 5 | 5 |
| thicc | MOGP | 5 | 0.403096689 | 0.403096689 | 0 (exact) | 0.0000% | 1 | 1 |
| thicc | Greedy Filter | 5 | 0.130552320 | 0.129203108 | −1.349e−03 | −1.0335% | 2 | 2 |
| thicc | Single-Obj BO | 5 | 0.107660047 | 0.107660047 | 0 (exact) | 0.0000% | 3 | 3 |
| thicc | Random Search | 5 | 0.084726722 | 0.084726722 | 0 (exact) | 0.0000% | 4 | 4 |
| thicccc | MOGP | 7 | 0.408895387 | 0.408895387 | 0 (exact) | 0.0000% | 1 | 1 |
| thicccc | Greedy Filter | 7 | 0.139590793 | 0.138645375 | −9.454e−04 | −0.6773% | 2 | 2 |
| thicccc | Single-Obj BO | 7 | 0.094447170 | 0.094447170 | 0 (exact) | 0.0000% | 3 | 3 |
| thicccc | Random Search | 7 | 0.089927709 | 0.089905257 | −2.245e−05 | −0.0250% | 4 | 4 |

**Method ranking is preserved in all three sweeps.**

**5 of 53 runs** move at all:

| run | HV before | HV after | % |
|---|---|---|---|
| `thicc/greedy/seed_1` | 0.169320243 | 0.162574182 | **−3.9842%** |
| `thicccc/greedy/seed_1` | 0.178735713 | 0.174842715 | −2.1781% |
| `thicccc/greedy/seed_8` | 0.139313426 | 0.136612809 | −1.9385% |
| `thicccc/random/seed_7` | 0.072528241 | 0.072371082 | −0.2167% |
| `thicccc/greedy/seed_7` | 0.137366319 | 0.137342006 | −0.0177% |

**MOGP's hypervolume is exactly unchanged: 0 of 13 runs move, maximum |Δ| =
0.000e+00** — exactly zero, not merely small. This is not because the rule
misses MOGP: **6 of 13 MOGP runs contain a non-viable dock, and all 6 lose a
row from their published front.** Those rows simply earned no hypervolume,
because a substituted value lands on the reference corner. The MOGP result does
not depend on the defect.

**Direction of the effect.** Every change is a *decrease*, and every decrease
falls on a baseline. The published comparison therefore **understated** MOGP's
margin. In the paired significance tests the margin widens slightly and no
verdict changes:

| sweep | comparison | mean paired diff before → after | Wilcoxon *p* | verdict |
|---|---|---|---|---|
| thicc | MOGP − Greedy Filter | 0.272544369 → 0.273893581 | 0.0625 (unchanged) | unchanged |
| thicccc | MOGP − Greedy Filter | 0.269304594 → 0.270250012 | 0.015625 (unchanged) | unchanged |
| thicccc | MOGP − Random Search | 0.318967679 → 0.318990130 | 0.015625 (unchanged) | unchanged |

**Pareto-front cardinality does move for MOGP**, even though hypervolume does
not: in the `thicccc` sweep MOGP's mean front size falls from 53.286 to 52.571
at n=90 (−1.34%), and in `thicc` from 36.8 to 36.6 at n=65. Front *size* counts
rows; hypervolume measures dominated volume, and these rows dominated nothing.
Both are in the `*.postfix-viability.csv` aggregates.

## 6. `summary.csv` and `best_per_objective.csv` — regenerated where the front survives, corrected where it does not

This section supersedes an earlier version of itself, which stated that these
two files "cannot be recomputed" because their per-case run directories no
longer exist. That is true of the directories, but it was **too pessimistic
about the archive**: this sweep's own logs print each case's *final Pareto
front in full*, and for 18 of the 38 cases that is enough to regenerate the
file properly, with no run directory and no AutoDock Vina.

The sweep therefore splits into two populations, and every row of both files
now says in-band which one it belongs to, via a `viability_status` column.

| population | cases | what was possible | `viability_status` |
|---|---|---|---|
| front recoverable from `logs/<case>.log` | 18 | **regenerated** by the repo's own generator | `REGENERATED` |
| front never copied to this machine | 20 | **corrected, not re-selected** | `CORRECTED_NOT_RE-SELECTED_FRONT_ABSENT` |

### 6.1 Why the run directories are absent, and what replaced them

`matrix_results_holo/runs/` does not exist and never did on this machine. The
logs name their output paths as
`/Users/devansh/mogp-main-vscode/MOGP-NTD/matrix_results_holo/runs/…` — the
sweep ran on a different host on 2026-08-11 and only its *sweep-level* outputs
and `logs/` were brought over. `matrix_report.py` is therefore not merely
unhelpful here, it is inoperable:

```
$ python matrix_report.py --sweep-dir matrix_results_holo
No runs/ directory under matrix_results_holo — nothing to report.
$ echo $?
1
```

What *did* survive is the console log of every case. 18 of them end with

```
Pareto-optimal molecules: 47
SMILES                        PfDHFR_Docking   hDHFR_Docking   hERG_Toxicity_Prob   Caco2_logPapp   Half_Life_hours
...47 rows...
```

That table **is** the per-case `pareto_front.csv`, minus its derived columns.
Honouring the repo's rule that "any process writing a sweep-level file must
derive it from the artifacts on disk, never from the set of cases the current
invocation ran" (`CLAUDE.md`), these 18 fronts were reconstructed from those
logs and handed to the repo's real generator. The remaining 20 cases — the
`harness-seeds-*/<method>/seed_0` runs — print per-iteration progress but
**never print a front table**, so for them nothing on disk reconstructs the
candidate set.

### 6.2 The reconstruction reproduces the pre-fix archive before changing it

The reconstruction was validated against the published pre-fix files *with the
viability rule off*, exactly as §7 validates the benchmark replay:

- **Row counts: 18 of 18 exact.** Each reconstructed front has the row count
  its log declares *and* the `pareto_size` the published `summary.csv` records.
- **Non-dominance: 18 of 18.** Re-running the repo's own Pareto filter
  (`acquisition.compute_pareto_front`, as `loop._pareto_mask` calls it) over
  each pre-fix reconstruction keeps **every** row, confirming each recovered
  table really is a Pareto front and not a truncated print.
- **Docking-derived values: bitwise exact.** Every pre-fix
  `PfDHFR_Docking`, `hDHFR_Docking` and `Selectivity_Index` winner in
  `best_per_objective.prefix-viability.csv` is reproduced **bitwise** (69 of
  126 rows bitwise exact overall; the other 57 are the three ADMET objectives
  and `Selectivity_Index_LE`, see the precision note below). Zero mismatches.
- **`best_SMILES`: 18 of 18**, and `final_hypervolume` to worst
  |residual| **1.369e−05**, `best_score` to worst |residual| **1.020e−05**.

Because the quantities the fix moves are reproduced *bitwise*, the corrected
values computed from the same reconstruction are trustworthy to the same
precision.

**Two precision caveats, both bounded and both the reason some columns are
carried rather than recomputed.**

1. **The logs print ADMET at 4 decimal places.** `hERG_Toxicity_Prob` is
   published as `0.0166221335530281` but logged as `0.0166`. Docking is logged
   at full precision (`-10.01`, `5.694`), which is why the docking-derived
   numbers come back bitwise and the ADMET ones come back to ~5e−05. The fix
   does not touch the ADMET objectives, so their rows are carried **verbatim**
   from the pre-fix file rather than re-derived, and no precision is lost in the
   published artifact.
2. **This sweep's normalization frame is the one the spec rejects.**
   `spec/bounds/objective_bounds.frozen.json` names
   `matrix_results_holo/evaluation_bounds.snapshot.json` under `rejected`: this
   sweep ran against a 3,610-molecule library, so its three ADMET bounds are
   that smaller library's min/max, and it "reproduces NO archived
   `history.csv`". Measured here: recomputing in the **canonical** frame moves
   `final_hypervolume` by up to **0.1805**, while recomputing in the sweep's
   **own archived** frame reproduces it to **1.369e−05**. The regeneration is
   therefore done in the sweep's own archived frame, so that the only thing
   that moves in these files is the viability fix and not a bounds change
   nobody asked for. The two **docking** bounds are `[-11, -5]` in both frames,
   so the substituted worst value is identical either way and this choice
   cannot affect the gate.

### 6.3 `best_per_objective.csv` — old and new value of every docking-derived row

The file keeps one row per case per objective (38 × 7 = 266 rows, unchanged).
Every row's old value, new value, raw kcal/mol and resolution is in the
companion ledger **`best_per_objective.viability_ledger.csv`**. Summary:

| | rows | moved | unchanged | repr-only | unknowable |
|---|---|---|---|---|---|
| docking-derived (`PfDHFR_Docking`, `hDHFR_Docking`, `Selectivity_Index`, `Selectivity_Index_LE`) | 152 | 36 | 68 | 3 | 45 |
| ADMET (`hERG_Toxicity_Prob`, `Caco2_logPapp`, `Half_Life_hours`) | 114 | 0 | 114 | — | — |

**The headline correction.** In **22 of 38 cases** the published "best
`hDHFR_Docking`, higher is better" was a **non-viable dock** — `+5.694`,
`+6.191` or `+0.3932` kcal/mol. The file advertised a molecule that found no
binding mode as the best human-DHFR result achieved. 12 of those 22 are now
regenerated; the other 10 are blanked because their front is gone.

**The 18 regenerated cases — `hDHFR_Docking` (raw kcal/mol):**

| case | old | new |
|---|---|---|
| `baseline-greedy-default` | −5.432 | −5.432 |
| `baseline-greedy-permissive` | −5.432 | −5.432 |
| `baseline-random` | −4.085 | −4.085 |
| `baseline-single-obj` | −3.75 | −3.75 |
| `bo-coreg-r1-densify-every1` | **+5.694** | **−1.047** |
| `bo-coreg-r1-densify-every2-cap` | **+5.694** | **−3.5** |
| `bo-coreg-r1-nodensify` | **+5.694** | **−3.5** |
| `bo-coreg-r2-densify-every1` | **+5.694** | **−3.5** |
| `bo-coreg-r2-densify-every2-cap` | **+5.694** | **−3.5** |
| `bo-coreg-r2-nodensify` | **+5.694** | **−3.5** |
| `bo-independent-densify-every1` | **+5.694** | **−3.184** |
| `bo-independent-densify-every2-cap` | **+5.694** | **−3.5** |
| `bo-independent-nodensify` | **+5.694** | **−3.5** |
| `launch-coreg-r1` | **+5.694** | **−3.5** |
| `launch-coreg-r2` | **+5.694** | **−3.5** |
| `launch-independent` | **+5.694** | **−3.5** |
| `profile-smoke-launch` | −7.321 | −7.321 |
| `profile-smoke-loop` | −7.321 | −7.321 |

**The 18 regenerated cases — `Selectivity_Index` (kcal/mol):**

| case | old | new |
|---|---|---|
| `baseline-greedy-default` | 3.0610 | 3.0610 |
| `baseline-greedy-permissive` | 3.0610 | 3.0610 |
| `baseline-random` | 3.2220 | 3.2220 |
| `baseline-single-obj` | 4.5030 | 4.5030 |
| `bo-coreg-r1-densify-every1` | **8.2000** | **5.4140** |
| `bo-coreg-r1-densify-every2-cap` | **8.2000** | **2.2940** |
| `bo-coreg-r1-nodensify` | **8.2000** | **3.7850** |
| `bo-coreg-r2-densify-every1` | **8.2000** | **4.2440** |
| `bo-coreg-r2-densify-every2-cap` | **8.2000** | **2.2940** |
| `bo-coreg-r2-nodensify` | **8.2000** | **3.7850** |
| `bo-independent-densify-every1` | **8.2000** | **2.4760** |
| `bo-independent-densify-every2-cap` | **8.2000** | **2.2940** |
| `bo-independent-nodensify` | **8.2000** | **3.7850** |
| `launch-coreg-r1` | **8.2000** | **3.7850** |
| `launch-coreg-r2` | **8.2000** | **3.7850** |
| `launch-independent` | **8.2000** | **3.7850** |
| `profile-smoke-launch` | 2.2940 | 2.2940 |
| `profile-smoke-loop` | 2.2940 | 2.2940 |

`PfDHFR_Docking` does **not** move in any case: a non-viable dock is positive
and can never win a minimum. `Selectivity_Index_LE` tracks
`Selectivity_Index` (e.g. 0.264516 → 0.126167 for the six `*-nodensify` /
`launch-*` cases). Three `Selectivity_Index_LE` rows are tagged `REPR_ONLY`:
`baseline-greedy-default`, `baseline-greedy-permissive`
(`0.0987419354838709` → `0.09874193548387097`) and `baseline-random`
(`0.111103448275862` → `0.11110344827586205`) differ by ≤ 7e−17 — the pre-fix
CSV's 16-significant-figure serialization of the same double, same winning
molecule, not a correction. They are recorded as `REPR_ONLY`, not as moves.

### 6.4 The 20 cases whose front is gone — corrected, never re-selected

For these, no replacement winner can be computed. But the published winner's
*validity* can still be settled, because **gating only removes candidates**:
`matrix_report.gated_best` excludes non-viable rows rather than substituting
them, so a published winner that is itself viable is still the extremum of the
smaller gated set. That gives three rigorous outcomes:

| outcome | rows | meaning |
|---|---|---|
| `CONFIRMED_UNCHANGED` | 35 | the published winner is viable, so it survives the gate and is **still** the correct answer. Published value retained. |
| `INVALIDATED_NOT_RE-SELECTABLE` | 30 | the published winner is non-viable, so it is excluded — the number is **wrong** and its replacement is unknowable. Value **blanked**. |
| `UNDETERMINED_WINNER_VIABILITY_UNKNOWN` | 15 | the winner's viability cannot be established from any surviving artifact. Value **blanked** as a precaution. |

How each verdict was reached, in order of strength:

1. For the two docking objectives the published `best_value` **is** the raw
   Vina affinity (these files predate substitution), so
   `evaluation.dock_is_viable` applies to it directly. All 40 of those rows are
   decided this way.
2. For a `Selectivity_Index` whose winning molecule is *the same molecule* as
   that case's published best-`hDHFR` winner, the hDHFR affinity is on record;
   if it is non-viable the SI is invalid whatever the PfDHFR value. This decides
   the 20 `mogp` and `random` rows (SI 8.2000, 8.0410 and 6.8412 — all
   invalidated).
3. For the five `single_obj` SI rows, the winning molecule is present in the
   live docking cache (`PfDHFR −9.773`, `hDHFR −7.407`, both viable) →
   `CONFIRMED_UNCHANGED` at SI 2.3660.
4. The remaining 15 rows (the five `greedy` SI rows, five `greedy` SI_LE rows
   and five `single_obj` SI_LE rows) have winners absent from the current
   docking cache — it is a later generation than this 2026-08-11 sweep and
   resolves only 11 of 80 of these winners — and no front on disk.
   `UNDETERMINED`.

The separation remains as clean as the earlier version of this section
reported: every invalidated SI sits at **≥ 6.8412** and every confirmed one at
**≤ 4.5030**.

### 6.5 `summary.csv` — every column that moves

38 rows, unchanged. Columns that move:

| column | rows moved | before → after |
|---|---|---|
| `best_Selectivity_Index` | 27 | 12 regenerated (8.2000 → 2.2940…5.4140); 15 blanked |
| `best_Selectivity_Index_LE` | 35 | 12 regenerated (0.264516 → 0.081929…0.193357); 20 blanked; **3 `REPR_ONLY`** |
| `best_Selectivity_Index_viable` | all 38 | new column (see §6.6) |
| `sweep_status` | all 38 | `complete` → **`PARTIAL`** (see §6.6) |

Columns that move **nowhere**, and why — each is a claim, not an omission:

- **`best_PfDHFR_Docking`** — a non-viable dock is positive, so it can never
  win a minimum. Gating it is belt-and-braces.
- **`best_score` and `best_SMILES`** — *provably* unmoved for all 38 cases.
  `matrix_report.normalized_scores` now routes each docking column through
  `evaluation.apply_dock_viability`, which only ever **lowers** a non-viable
  row's score; it cannot raise a viable one. `best_molecules.csv` records each
  case's champion in full, and **0 of its 38 rows** has a non-negative docking
  value, so no champion was ever a clash. A viable argmax whose competitors can
  only fall remains the argmax. Recomputing from the reconstruction agrees:
  champion SMILES identical in 18 of 18, `best_score` within the 1.020e−05
  reconstruction floor. The published full-precision values are therefore
  retained rather than replaced by slightly noisier recomputations.
- **`final_hypervolume` and `n_evaluated`** — `matrix_report.summarize` reads
  both from `history.csv`, the **record of measurement**, which this fix
  deliberately does not rewrite anywhere in the repo (§3). They are carried
  from the archive for exactly the reason the 53 benchmark runs keep their
  pre-fix `history.csv`. **Consequence, stated plainly: `final_hypervolume` in
  this file is a PRE-FIX number and is not comparable to a hypervolume
  recomputed under the corrected objective.** It is also computed in the
  rejected bounds generation (§6.2). Do not quote it as a post-fix result.
- **`pareto_size`** — see §6.6.

### 6.6 Three deliberate inconsistencies, so none reads as corruption

1. **`sweep_status` is now `PARTIAL`, not `complete`.** This is
   `matrix_report.assess_completeness`'s own verdict, not a judgement added
   here. Given the 18 cases whose artifacts survive on disk, it reports:
   *"5 case(s) passed but left no artifacts: harness-seeds-band-ci95,
   harness-seeds-band-sem, harness-seeds-band-std, harness-seeds-densify,
   harness-seeds-nocache."* The earlier `complete` stamp was produced when
   `runs/` still existed on the original host. It is now false on this machine
   and was not preserved. Per the generator's own instruction: do not quote
   these numbers as the sweep's result without reading this section.

2. **`pareto_size` is the ARCHIVED front size; the corrected size sits beside
   it.** Under the corrected objective a clash that was non-dominated *only*
   because its hDHFR was the front's maximum becomes dominated and leaves the
   front. **6 of the 18 reconstructed fronts shrink by exactly one row:**

   | case | archived | corrected |
   |---|---|---|
   | `bo-coreg-r1-densify-every1` | 37 | 36 |
   | `bo-coreg-r1-densify-every2-cap` | 38 | 37 |
   | `bo-coreg-r2-densify-every1` | 43 | 42 |
   | `bo-coreg-r2-densify-every2-cap` | 38 | 37 |
   | `bo-independent-densify-every1` | 36 | 35 |
   | `bo-independent-densify-every2-cap` | 40 | 39 |

   `pareto_size` keeps the archived value (it is what the measurement record
   says) and the new column **`pareto_size_viability_corrected`** carries the
   post-fix size, blank for the 20 cases where it cannot be computed. This is
   the same deliberate split as the benchmark sweeps, where **14 of 53**
   regenerated `pareto_front.csv` files now have fewer rows than their own
   `history.csv` final `pareto_size` — e.g.
   `benchmark_thicc_2026-08-14/greedy/seed_1`: front file **40** rows,
   `history.csv` `pareto_size` **41** — because `history.csv` is the
   measurement record and was correctly not rewritten. Verified here: all 53
   `pareto_front.prefix-viability.csv` backups still match their `history.csv`
   `pareto_size` exactly, which is what proves the 14 differences are the fix
   and not drift.

3. **A blank cell means "unknowable", not "zero" and not "missing".** 45 of
   266 `best_value` cells and the corresponding `summary.csv` cells are empty.
   Every one is explained by its row's `viability_status` and by the ledger's
   `resolution` column. Nothing was imputed.

### 6.7 Schema

Both files keep the current generator's columns and add `viability_status`,
following the precedent `matrix_report.py` sets for `sweep_status` — stamped on
every row so "anyone loading this CSV sees the sweep's status without having to
know to look for a separate file."

- `summary.csv`: `sweep_status, viability_status, case, family,
  final_hypervolume, n_evaluated, pareto_size,
  pareto_size_viability_corrected, best_score, best_SMILES,
  best_Selectivity_Index, best_Selectivity_Index_viable,
  best_Selectivity_Index_LE, best_PfDHFR_Docking, seconds, minutes, status`
- `best_per_objective.csv`: `case, objective, direction, best_value,
  best_value_viable, SMILES, viability_status`. No separate raw-kcal column is
  needed or emitted: for the two docking objectives `best_value` **is** the raw
  Vina affinity, because `best_by_objective` ranks the raw measurement among
  viable rows rather than the substituted objective. For the two Selectivity
  Indices it is the measured difference of two viable affinities.
- `best_per_objective.viability_ledger.csv` (new): the full audit trail, one
  row per published row — `case, objective, population, prefix_best_value,
  prefix_SMILES, prefix_raw_kcal, prefix_viable, best_value, SMILES, raw_kcal,
  moved, resolution`, where `moved` is `True` / `False` / `REPR_ONLY` /
  `UNKNOWN` and `resolution` names the verdict and the evidence for it.

### 6.8 Pre-fix copies

| file | pre-fix copy | contents |
|---|---|---|
| `summary.csv` | `summary.prefix-viability.csv` | the original 2026-08-14 publication |
| `best_per_objective.csv` | `best_per_objective.prefix-viability.csv` | the original 2026-08-14 publication |
| `summary.csv` | `summary.annotated-uncorrected.csv` | the intermediate state described by the previous version of this section: values **still the defect's**, with a viability flag added |
| `best_per_objective.csv` | `best_per_objective.annotated-uncorrected.csv` | same, for the per-objective file |
| `VIABILITY_FIX.md` | `VIABILITY_FIX.prefix-viability.md` | this document before this revision |

The `.annotated-uncorrected.csv` pair is kept deliberately: it is the evidence
for what §6 used to claim, and it is the only place the "annotate but do not
correct" intermediate survives.

### 6.9 Every file in this directory carrying a docking-derived figure

| file | state |
|---|---|
| `summary.csv` | **regenerated** (18 cases) / **corrected** (20), this section |
| `best_per_objective.csv` | **regenerated** (18) / **corrected** (20), this section |
| `best_per_objective.viability_ledger.csv` | **new**, post-fix audit trail |
| `best_molecules.csv` | **clean, verified, not rewritten.** 38 champion rows, **0** with a non-negative docking value. The champion cannot move (§6.5), so this file is correct under both semantics. |
| `leads_annotated.csv` | **corrected, not re-selected** (§4.2, §8.1) |
| `FINAL_LEADS.csv` | **unchanged, proven** (§4.3) |
| `matrix_comparison.png` | **STALE — still plots the defect.** Its selectivity panel scatters every pooled Pareto molecule's raw `hDHFR_Docking`, so the `+5.694` and `+0.3932` clashes are still drawn in the "desirable" corner, and its hypervolume/champion panels are the pre-fix numbers. It **cannot be regenerated**: `matrix_report.plot` needs every case's full `history.csv` trajectory and `pareto_front.csv`, and 20 of 38 cases have neither on this machine. Treat the figure as pre-fix. |
| `validate_docking_scatter.png` | pre-fix diagnostic from `validate_docking.py`; not regenerated (needs the validation run's inputs, absent). Out of scope here. |
| `results.csv`, `manifest.json`, `report.md`, `console.log`, `logs/*.log` | run records, no derived docking figure; untouched by design. `logs/` is the *source* this regeneration derives from. |
| `evaluation_bounds.snapshot.json` | the rejected bounds generation (§6.2); untouched. |

**Outside this directory, still stale** (reported, not changed — modifying them
was out of scope for this pass):

- `benchmark_*/benchmark_seeds_aggregate.csv` and
  `benchmark_*/benchmark_seeds_significance.csv` — the canonical filenames
  still hold the **pre-fix** numbers for `benchmark_thicc_2026-08-14` and
  `benchmark_thicccc_2026-08-16` (both differ from their
  `.postfix-viability.csv` siblings; `benchmark_5way_densify_seed0_2026-08-19`
  is identical in both, i.e. unaffected). §3 explains why they were published
  beside rather than over the originals — but a reader who opens the canonical
  name gets the defect's numbers.
- `benchmark_*/benchmark_seeds.png` — same, pre-fix; corrected version is
  `benchmark_seeds.postfix-viability.png`.

## 7. How this was verified

**The harness reproduces the archive before changing it.** Recomputing each
run's front and hypervolume from its `evaluated.csv`, with the viability rule
*off*, reproduces:

- all **53 of 53** archived `pareto_front.csv` files **byte-for-byte**
  (including 17-digit float literals such as `0.30134737491607666`);
- the final hypervolume in `history.csv` to worst |residual| **5.551e−17**;
- the entire archived hypervolume trajectory — 448 (run, iteration) points — to
  worst |residual| **1.110e−16**, with `pareto_size` matching at **all 448**
  points;
- all three `benchmark_seeds_aggregate.csv` files byte-for-byte.

So the "before" column of every table above *is* the published record.

**The regenerated files were checked three independent ways**: every surviving
line is a byte-verbatim archived line in archived order and no line was
invented; every file agrees cell-for-cell with the live
`evaluation.apply_dock_viability` / `evaluation.add_selectivity_index` code
path; and the content invariants of §4.1 hold on all 2,407 rows.

**Edits are text surgery, not re-serialization.** pandas' CSV reader parses some
archived literals one unit-in-the-last-place off (`0.30134737491607666` →
`0.3013473749160766`), so a read/write round trip would silently perturb the
record. Working on the archived text leaves every surviving byte as published.
This same parser behaviour is the source of the `1.665e-16` residual noted
elsewhere in the repo.

**The rule itself is pinned by tests.** `test_dock_viability.py` (33 tests)
covers the threshold and predicate, the symmetry of the substitution and its
derivation from the signs, that no known active is excluded, that the raw
measurement survives, the Selectivity Index gate, and that the frozen frame is
untouched:

```
conda run -n mogp-drug python -m pytest test_dock_viability.py -q
```

### Reproducing the tables above

Every number here is derived from files still in the repository — the 53
`evaluated.csv` records and the `.prefix-viability.csv` copies — using the live
`evaluation` module. The recipe:

1. For each run, read `evaluated.csv`. Its `*_Docking` columns predate the fix,
   so they are the raw measurement. Call that matrix `Y_raw`.
2. `Y_before = Y_raw`. For `Y_after`, replace each docking column with
   `evaluation.apply_dock_viability(Y_raw[:, j], j)[0]`.
3. Fronts: `acquisition.compute_pareto_front` over the objectives carrying data,
   restricted to fully observed rows — i.e. `loop.BOLoop._pareto_mask`.
   Hypervolume: `evaluation.compute_hypervolume`.
4. Intermediate iterations are recoverable because `evaluated.csv` rows are in
   evaluation order: the first *n* rows are the evaluated set at the iteration
   where `history.csv` records `n_evaluated == n`. This is what reproduces the
   448-point trajectory.
5. The `*.postfix-viability.*` summaries come from a post-fix `history.csv`
   mirror anchored on the published record,
   `hv_after_published = hv_archived + (hv_after - hv_before)`, fed to this
   repo's own `run_benchmark_seeds.save_aggregate_csv`,
   `paired_significance` and `save_figure`. The correction term is exactly
   `0.0` for the 48 runs that do not move, so their published numbers are
   preserved bitwise, and no aggregation or statistics logic is reimplemented.

**Reproducing §6** (`matrix_results_holo`) needs no `runs/` directory:

1. For each `logs/<case>.log`, take the last `Pareto-optimal molecules: N`
   line, then the `SMILES  <TASK_NAMES…>` header, then the next N
   whitespace-separated rows. That is the case's archived Pareto front; its
   `*_Docking` columns are the raw measurement.
2. Add `*_Docking_LE` (`raw / Chem.MolFromSmiles(s).GetNumHeavyAtoms()`, i.e.
   `docking.raw_to_ligand_efficiency`) and call
   `evaluation.add_selectivity_index`, which gates both Selectivity Indices to
   `NaN` on any row where either dock is non-viable.
3. Pin the frame to this sweep's own snapshot —
   `evaluation.compute_objective_bounds` → the array in
   `evaluation_bounds.snapshot.json` — then call the generator unmodified:
   `matrix_report.best_by_objective`, `matrix_report.gated_best`,
   `matrix_report.docking_viability`, `matrix_report.both_docks_viable`,
   `matrix_report.champion` and `matrix_report.assess_completeness`.
4. Corrected front sizes: substitute each docking column with
   `evaluation.apply_dock_viability(raw, j)[0]`, then
   `acquisition.compute_pareto_front` over the objectives carrying data,
   restricted to fully observed rows — `loop.BOLoop._pareto_mask` again.
5. Nothing in this recipe re-implements the threshold; every viability decision
   reduces to `evaluation.dock_is_viable` / `evaluation.apply_dock_viability`
   and `evaluation.DOCK_VIABILITY_KCAL_MAX`.

**Provenance of the §6 regeneration.** `matrix_report.py` was itself being
gated while these files were produced, so the regeneration was run against two
successive revisions of it — `sha256[:16]` `713a93e842b02d83` and
`4f9104967da75eaf` — and both produced **byte-identical** `summary.csv`,
`best_per_objective.csv` and `best_per_objective.viability_ledger.csv`. Two
back-to-back runs against the same revision are also byte-identical, so the
output is reproducible and does not depend on which of those generator
revisions is in the tree.

## 8. Limitations

1. **`leads_annotated.csv` was corrected, not re-selected.** Its 50 molecules
   are the top 50 by Selectivity Index from the pooled Pareto fronts of
   `matrix_results_holo/runs/**`, which no longer exist on disk — only **6 of
   the 50** appear anywhere in the 53 surviving benchmark fronts — and each row
   carries a *fresh* pair of Vina docks that cannot be repeated without the
   binary. What was corrected is the ranking defect *inside* the table. Under a
   full re-selection the two non-viable molecules would drop out and two
   molecules from former ranks 51–52 would enter; those two cannot be docked
   here, so they are absent. This does not affect `FINAL_LEADS.csv`, whose 8
   rows are all viable and all retained (§4.3).
2. **`annotate_leads.py` is not itself guarded.** It computes its Selectivity
   Index inline as `hd["score"] - pf["score"]` rather than through
   `evaluation.add_selectivity_index`, so re-running it would again write a raw
   difference. The fix removes the defect at the *source* — a non-viable row can
   no longer reach any Pareto front, so it can no longer be selected — but that
   one inline computation remains unguarded. Same note for its `classify()`
   thresholds.
3. **`summary.csv` and `best_per_objective.csv` are only partly
   re-selectable.** 18 of the 38 cases were genuinely regenerated from the
   Pareto tables printed in this sweep's own logs; the other 20
   (`harness-seeds-*/<method>/seed_0`) never printed a front, so their
   docking-derived "best" cells could only be **confirmed or invalidated**, not
   replaced — 30 cells are blanked as wrong-but-unknowable and 15 as
   undetermined. Two further limits apply to all 38: `final_hypervolume` is a
   **pre-fix** number carried from the measurement record, and it is computed in
   the bounds generation that `spec/bounds/objective_bounds.frozen.json`
   rejects, so it should not be quoted as a post-fix result. `sweep_status` is
   now `PARTIAL`. See §6, which supersedes the earlier claim that these two
   files could not be recomputed at all.
4. **The timeout justification is indirect.** The docking subprocess timeout
   added alongside this fix (600 s, ≈209× the measured 2.87 s mean per dock) is
   justified from the archived `--no-cache` sweep log, not from a direct timing:
   Vina is not installed on this machine.
5. **Reference affinities are partly documentary.** Of the eight known-active
   affinities quoted in §2, five were verified against the live docking cache
   (pyrimethamine PfDHFR −9.087 / hDHFR −8.992, cycloguanil −9.110, WR99210
   −8.818, P218 −8.548); the remaining three have no cached row and rest on the
   repo's documentation. All five measured values are viable with margins above
   8.5 kcal/mol. Separately, `evaluation.py` cites pyrimethamine's PfDHFR
   affinity as −9.18 while the cache holds −9.087 — a pre-existing
   documentation discrepancy, left unchanged, and immaterial here since both
   are viable by more than 9 kcal/mol.
6. **The post-fix significance tables carry one pre-existing numeric
   difference.** Re-deriving the *published* significance CSVs with the current
   code reproduces `mean_diff`, `wilcoxon_stat`, `wilcoxon_p`, `ttest_stat`,
   the significance verdict and the note bitwise, but `ttest_p` differs in its
   last two or three digits (relative ≈1e−15) because the archived values came
   from SciPy and the current code uses this repo's numpy port. That difference
   predates this fix and is unrelated to it.
