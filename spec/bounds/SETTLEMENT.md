# Objective-bounds settlement

**Determination:** `evaluation_bounds.json` (repo root) is **canonical**.
`matrix_results_holo/evaluation_bounds.snapshot.json` is **rejected** — it is a
faithful archive of a superseded generation, not a competing candidate.

Settled by measurement on 2026-09-27. Every number below was produced by the two
scripts in this directory, run in the `mogp-drug` env against the repo's own
current `evaluation.py`. Nothing here is asserted without a measurement behind
it.

Frozen artifact: `objective_bounds.frozen.json`
sha256 `abaa99a4d5c21b62dd84b0a8227076d510f79ed8cc4dd2ecfbe4625d2244dfa6`
(sidecar `objective_bounds.frozen.json.sha256`, verified with `shasum -a 256 -c`).

---

## 1. What is on disk

There are **four** bounds files in the repo, not two. Three are byte-identical.

| label | path | sha256 |
|---|---|---|
| **root (canonical)** | `evaluation_bounds.json` | `9c9c791c66a3eb8d0796bed42b29eec531f825c1c7c1444b0267b4c13549e27e` |
| confirming | `matrix_thick_2026-08-14/evaluation_bounds.snapshot.json` | `9c9c791c…27e` (identical) |
| confirming | `matrix_thick_2026-08-15/evaluation_bounds.snapshot.json` | `9c9c791c…27e` (identical) |
| **rejected** | `matrix_results_holo/evaluation_bounds.snapshot.json` | `479f1f076018e1f236deb51159a0bcf527c8d519f08ce01961e833cfa962786f` |

So only **two distinct bound sets** exist. The task brief described the holo
snapshot as "the" snapshot; in fact two later snapshots exist and both agree
with the root file. That asymmetry (2-vs-1) is itself evidence.

### Field-by-field diff

`task_names` and `signs` are identical in all four files:
`[PfDHFR_Docking, hDHFR_Docking, hERG_Toxicity_Prob, Caco2_logPapp, Half_Life_hours]`
with signs `[-1, +1, -1, +1, +1]`.

| objective | canonical (root) | rejected (holo) | differs |
|---|---|---|---|
| `PfDHFR_Docking` | `-11.0`, `-5.0` | `-11.0`, `-5.0` | no |
| `hDHFR_Docking` | `-11.0`, `-5.0` | `-11.0`, `-5.0` | no |
| `hERG_Toxicity_Prob` | `0.001999363536015153`, `0.9998417496681213` | `0.0038742402102798223`, `0.9996427297592163` | **both ends** |
| `Caco2_logPapp` | `-7.260476112365723`, `-3.825927495956421` | `-6.839174747467041`, `-3.958852767944336` | **both ends** |
| `Half_Life_hours` | `0.37630030512809753`, `64.84938049316406` | `0.37630030512809753`, `42.01264953613281` | **max only** |

Three of five objectives differ, exactly as reported. The two docking bounds are
identical because they are *fixed constants*, not data-derived.

---

## 2. The decisive test and its result

`reproduce_archived_hypervolume.py`. For every archived directory holding both
`evaluated.csv` and `history.csv`, replay the trajectory using the repo's own
code and compare to the archived `hypervolume` column:

```
hv_hat(iteration i) = evaluation.compute_hypervolume(Y[: n_evaluated_i], bounds=B)
```

This is valid because `loop._save_results` writes `evaluated.csv` rows in
evaluation order (`self.evaluated_indices`), and `history.csv` records
`n_evaluated` per iteration.

**53 archived runs qualified** (not one — 53), across four campaigns:
`benchmark_thicc_2026-08-14` (20 runs), `benchmark_thicccc_2026-08-16` (28),
`benchmark_5way_densify_seed0_2026-08-19` (5, including the `gpmobo` arm).
`matrix_results_holo` itself does **not** qualify: its `results.csv` is a
harness status table (`id,group,tier,status,exit_code,…`), and it ships no
`evaluated.csv`, so it cannot be replayed.

| bound set | runs | worst max&#124;residual&#124; | median max&#124;residual&#124; |
|---|---|---|---|
| **root / thick14 / thick15** | 53 | **1.665335e-16** | 8.326673e-17 |
| holo snapshot | 53 | 1.441488e-01 | 6.564532e-02 |

The canonical set reproduces the archived hypervolume column to
**floating-point round-off (worst case 1.665e-16 — one to three ulp at these
magnitudes, where float64 ulp is 5.6e-17 near 0.25 and 1.1e-16 near 0.5)** on
all 53 runs and all **448** iteration rows. The holo set misses by up to
**0.144 absolute hypervolume**.

**Per-run winner tally: root wins 53 of 53. There is no disagreement between
archived runs.** Point 4 of the brief — "if different archived runs used
different bounds, say so loudly" — does not fire: every replayable archived run
was produced under the same, canonical bounds.

Headline residual, `benchmark_thicccc_2026-08-16/mogp/seed_0` (7 iterations):
max |residual| = **1.1102230246251565e-16**; archived final hypervolume
`0.356205027964296`, replayed `0.356205027964296` (all 15 printed digits).

### The 0.356-vs-0.478 swing: confirmed exactly

The prior reader's figure is reproduced to six decimals, and it is this run:

| run | archived | canonical | holo | holo − canonical |
|---|---|---|---|---|
| `benchmark_thicccc_2026-08-16/mogp/seed_0` | 0.356205 | **0.356205** | **0.478292** | +0.122087 |

Across all 53 runs the holo frame **inflates** final hypervolume by
min `+0.032386`, max `+0.144149`, mean `+0.070838`. The inflation is systematic
and always positive: the holo cube is narrower on Half_Life (max 42.0 vs 64.8)
and on Caco2, so the same molecules occupy a larger fraction of it.

---

## 3. Why the holo snapshot differs (provenance, from the manifests)

Not a bug, and not a rival — a different generation.

| | `matrix_results_holo` | `matrix_thick_2026-08-14` and later |
|---|---|---|
| ran | 2026-08-11 14:52 → 22:34 | 2026-08-14 23:17 → |
| `library_molecules` | **3,610** | **29,678** |
| git branch | `Devansh_Local` @ `67069df` | `main` @ `9ae0e49` |
| `objective` block | `docking_units: kcal/mol`, `docking_bounds: [-11.0, -5.0]` | identical |

The three ADMET bounds are `min`/`max` over the whole cached library, so a
library that grew 3,610 → 29,678 necessarily widened them. The holo snapshot is
the correct record of the bounds *for that campaign*; it is simply superseded.

`matrix_results_holo/manifest.json` says so itself, under `comparability`:

> `NOT_comparable: "HYPERVOLUME. matrix_results/ optimized size-corrected
> LIGAND EFFICIENCY; this sweep optimizes RAW kcal/mol. … the two numbers
> measure different volumes and must never share an axis."`

Corroboration that the holo `Half_Life` max of 42.0126 is a real library
molecule and not a stale constant: `matrix_results_holo/logs/launch-independent.log`
line 276 lists `Clc1ccc(C(NCC2CCCCC2)c2cccnc2)cc1` with `Half_Life = 42.0126`.

### Independent confirmation of the canonical numbers

`evaluation.compute_objective_bounds(force=True)` run against the **current**
`data/library`, with `bounds_path` redirected to a scratch file so nothing in the
repo was touched, reproduces the canonical bounds **bit-exactly** (elementwise
difference all `0.0`). The library today is 29,678 raw → 26,660 after the
heavy-atom floor and quality screen, and its per-column min/max are exactly the
three ADMET bounds in `evaluation_bounds.json`.

So the canonical set has three independent supports: it reproduces 53 archived
run histories, it is byte-identical to two archived snapshots, and it is
recomputable from today's library.

---

## 4. Stale comments in `evaluation.py` — verified, and worse than described

**The code is right; the prose is wrong.** `DOCKING_KCAL_MIN = -11.0` /
`DOCKING_KCAL_MAX = -5.0` are **raw whole-molecule Vina kcal/mol**. The brief's
suspicion is confirmed, and confirmed by falsification rather than by reading.

Falsification test on `benchmark_thicccc_2026-08-16/mogp/seed_0`:

| objective columns fed to `compute_hypervolume` | result |
|---|---|
| `PfDHFR_Docking`, `hDHFR_Docking` (raw kcal, range −10.35 … −1.85) | `0.356205027964296` = archived, exactly |
| `PfDHFR_Docking_LE`, `hDHFR_Docking_LE` (LE, range −0.573 … −0.060) | **`0.000000000000000`** |

Under bounds `[-11, -5]` every LE value clips: PfDHFR LE → exactly 0.0 on all
rows, collapsing the hypervolume to zero. So the optimized columns cannot be LE,
and the bounds cannot be an LE range. Both archived manifests independently
record `objective: {docking_units: "kcal/mol", docking_bounds: [-11.0, -5.0]}`.

Stale passages (line numbers as of the reading; **not edited**):

- **L32–38** — "Historical note — they were SIZE-CORRECTED LIGAND EFFICIENCY …
  **Their fixed normalization range is therefore an LE range (per atom), not a
  kcal range.**" The final sentence is false for the current constants. Note
  L29–32 *already* says "the two docking objectives are RAW kcal/mol", so the
  docstring contradicts itself within ten lines.
- **L41–43** — "Both docking objectives share the same fixed LE range".
- **L79–100** (the constant's own comment block) — "Fixed LIGAND-EFFICIENCY
  range (kcal/mol PER HEAVY ATOM)", and "the span is set from the LE
  distribution `validate_docking.py` observed … best/most-negative LE ~ −0.53,
  worst ~ −0.19, mean ~ −0.36". A span of −11 … −5 is arithmetically impossible
  to derive from an observed −0.53 … −0.19 distribution. This is the most
  misleading block in the file.
- **L99–100** — the `DOCKING_LE_MIN` / `DOCKING_LE_MAX` aliases perpetuate the
  wrong unit in the *name*, and `loop.py` uses them in a user-facing warning
  that prints "kcal/mol/atom".
- **L126–133** — `compute_objective_bounds` docstring: "fixed configurable
  LIGAND-EFFICIENCY range (kcal/mol per heavy atom, not raw kcal)".
- **L208–213** — `normalize` docstring: "The docking columns are ligand
  efficiency (per atom), so their bounds are the fixed LE range."
- **L340–349** — `add_selectivity_index` docstring: "the two docking columns are
  now size-corrected LIGAND EFFICIENCY … this is a PER-ATOM selectivity". The
  code at L361 computes `Selectivity_Index` from the **raw kcal** columns; the
  per-atom version is the separate `Selectivity_Index_LE` at L363. L107–108
  ("the primary column is now whole-molecule kcal") is correct and contradicts
  L340–349.
- **L379–392** (`__main__` self-check) — labels `-11.0` as `strong_le` and
  prints "strong LE −11.000", which as a ligand efficiency is physically absurd.
  The assertions still pass because they test the normalization identity, not
  the units, so this check cannot catch the error.

`loop.py` L324–353 (`_warn_if_docking_saturates`) carries the same stale prose
("ligand efficiency", "kcal/mol/atom" in the warning text), but the
**comparison at L346 is dimensionally correct** — it compares a raw kcal score
against `DOCKING_KCAL_MIN`, which is raw kcal. `DOCKING_SATURATION_MARGIN = 0.25`
(L93–98) is likewise labelled "Margin (kcal/mol)" in its first clause, which is
right; only the trailing claim "On the LE scale (range ~0.55 wide) this is far
smaller than the old raw-kcal margin" is stale. The trigger itself is live and
correct in kcal: the archived data reaches −11.3 (PfDHFR) and −12.38 (hDHFR)
against a threshold of −10.75, so this warning did fire during the archived
campaigns. I did not check the logs for it.

Per the charter this is a docstring fix, not a model or constant change. Nothing
here was edited.

---

## 5. Clipping under the canonical bounds (measured)

`measure_clipping.py`, pooled over all 53 runs = 4,165 rows (2,437 distinct
objective tuples). "at 0.0" / "at 1.0" mean the value is **exactly** 0.0 or 1.0
after `np.clip` in `evaluation.normalize`.

| objective | finite | at 0.0 | at 1.0 | observed range | bounds |
|---|---|---|---|---|---|
| `PfDHFR_Docking` | 4061 | 18 | 18 | −11.3 … **+4.213** | −11, −5 |
| `hDHFR_Docking` | 4061 | 48 | **79** | −12.38 … **+10.73** | −11, −5 |
| `hERG_Toxicity_Prob` | 4165 | **0** | **0** | 0.00499953 … 0.999371 | 0.002 … 0.99984 |
| `Caco2_logPapp` | 4165 | **0** | **0** | −7.06545 … −3.8692 | −7.26 … −3.826 |
| `Half_Life_hours` | 4165 | 1 | 6 | 0.3763 … **81.633** | 0.3763 … 64.849 |

Deduplicated to distinct objective tuples: PfDHFR 11@0 / 10@1, hDHFR 33@0 /
43@1, hERG 0/0, Caco2 0/0, Half_Life 1@0 / 5@1.

### The prior reader's 2-and-5 figure: confirmed exactly

It is `benchmark_thicccc_2026-08-16/mogp/seed_0`, and it matches to the row:

> `PfDHFR_Docking: 2@0 (of 89 finite), hDHFR_Docking: 5@1 (of 89 finite)`

90 rows, 89 finite on the docking columns (one failed dock). **2 rows clip to
exactly 0.0 on PfDHFR** and **5 saturate at exactly 1.0 on hDHFR**. Verified.

### Three hazards this measurement exposes

These are properties of the *frame*, not of the choice between the two files.
Both bound sets have them, since the docking bounds are identical.

1. **A failed dock can score as a perfect result.** 15 pooled rows have
   *positive* hDHFR Vina scores (up to **+10.73 kcal/mol**) — physically
   non-binding or clashing poses. Because hDHFR is a *maximization* objective
   ("we want weak human binding"), all 15 normalize to **exactly 1.0**. A pose
   that failed is credited as ideal selectivity. This is the most serious
   finding in this report and it is not a bounds question — it needs a validity
   gate on the docking oracle. Flagging it, not fixing it; out of my scope.
2. **ADMET saturation is unwarned.** 6 rows exceed the Half_Life library max
   (up to 81.633 h vs bound 64.849) and saturate at 1.0. These are densified
   analogs whose oracle prediction escapes a cube derived from the *pre*-
   densification library. `loop._warn_if_docking_saturates` covers the docking
   axes only; there is no equivalent ADMET check.
3. **A float round-trip, not science, causes the one Half_Life zero.** The
   library-minimum molecule is stored in `evaluated.csv` as
   `0.3763003051280975` while the bound is `0.37630030512809753` — i.e.
   `-5.551115123125783e-17` *below* the bound — so it clips to exactly 0.0. The
   bounds derive from float32 ADMET artifacts, are persisted at full float64
   repr, and are compared against CSV-serialized float64. The numpy
   reimplementation must expect values to straddle a bound this way. It is
   numerically harmless here (the value would normalize to ~1e-17 anyway) but it
   is exactly the class of silent error worth naming.

---

## 6. Caveats and what remains unresolved

- **`matrix_results_holo` was never replayed.** It ships no `evaluated.csv`, so
  its snapshot could not be positively confirmed against its own history by the
  same method. Its rejection rests on (a) reproducing no archived history, (b)
  its manifest recording a 3,610-molecule library, and (c) its own
  `comparability.NOT_comparable` note. I could not verify that its ADMET bounds
  are the exact min/max of that 3,610-molecule library, because that library
  snapshot no longer exists on disk.
- **The reference-point convention was not independently re-derived.** I took
  `fixed_reference_point` = all-zeros as given, because that is what reproduces
  the archive. Whether the all-zeros corner is the *right* reference (as opposed
  to, say, a small negative offset, which is what botorch's own tutorials
  usually use) is a separate question and is untouched here. With the reference
  at the exact cube corner, any row clipping to 0.0 on *any* axis contributes
  literally nothing, which is why the clipping counts above matter.
- **The residuals prove which bounds produced the archive; they do not prove the
  bounds are scientifically best.** 18 clipped and 18 saturated rows on PfDHFR,
  and 48/79 on hDHFR, say the docking cube `[-11, -5]` is too narrow for the
  data actually collected. Per the charter I did not widen it — the archived
  campaigns are superseded and will be re-run, so if the frame is to be widened,
  that is a deliberate decision for the re-run, taken once, and it belongs to
  whoever owns the campaign design. It is not settled by this document.
- **Residual magnitude is not a tolerance.** `1.665e-16` is the worst observed
  value over 53 runs on this machine (numpy 2.4.6, botorch 0.18.1, torch 2.13.0,
  float64 hypervolume). It is not a claimed bound for other platforms.
- **`gpmobo` arm replays too.** `benchmark_5way_densify_seed0_2026-08-19/gpmobo/seed_0`
  reproduces at `9.714e-17` under the canonical bounds, confirming the external
  GP-MOBO baseline arm was scored through this same repo-side frame. Nothing
  under `external/GP-MOBO` was read, run, or modified.

---

## 7. Files in this directory

| file | what it is |
|---|---|
| `objective_bounds.frozen.json` | **the frozen canonical bounds** + hashes + provenance + measured residuals + clipping |
| `objective_bounds.frozen.json.sha256` | `shasum -a 256 -c` sidecar |
| `reproduce_archived_hypervolume.py` | the decisive test (§2) |
| `reproduction_residuals.json` | its full per-iteration residuals, all 53 runs × 4 bound sets |
| `measure_clipping.py` | clipping and swing measurement (§3, §5) |
| `clipping_measurements.json` | its output |
| `freeze_bounds.py` | generates the frozen artifact; every number computed, none typed |
| `SETTLEMENT.md` | this document |

Reproduce from the repo root with the `mogp-drug` env active:

```
python spec/bounds/reproduce_archived_hypervolume.py
python spec/bounds/measure_clipping.py
python spec/bounds/freeze_bounds.py
```

All three are read-only with respect to pre-existing repo files. `freeze_bounds.py`
writes only into `spec/bounds/`.

### For the numpy reimplementation

Read `objective_bounds.frozen.json` and check `bounds_payload_sha256`
(`03bd72e40053b6a9343e44d35b9f55e2101f1f0579f1d37aa0a9742e1b1e707f` — a
formatting-independent hash of the numbers). Do **not** call
`compute_objective_bounds`, whose "load if the file exists" behaviour means a
deleted or regenerated file silently redefines the frame from whatever the
library happens to contain that day. Freezing the numbers is the point.
