# Independent re-verification of the bounds settlement

A second pass over the same question, with **independently written** measurement
code (not a re-run of `reproduce_archived_hypervolume.py` / `measure_clipping.py`),
to check that `objective_bounds.frozen.json` and `SETTLEMENT.md` are not resting
on a single implementation's arithmetic.

Env: `mogp-drug`, numpy 2.4.6, torch 2.13.0, botorch 0.18.1.
Nothing pre-existing in the repo was modified. Scratch scripts lived outside the
repo; only this file was added.

## Verdict

**The settlement holds.** `/Users/calebsuh/MOGP-NTD/evaluation_bounds.json`
is canonical. Every material number in the frozen artifact reproduced.

## What reproduced, independently

| claim in the frozen artifact | my independent measurement | agrees |
|---|---|---|
| canonical source sha256 `9c9c791c…49e27e` | same | yes |
| `matrix_thick_2026-08-14` + `-15` snapshots byte-identical to root | same sha256 on all three | yes |
| rejected holo snapshot sha256 `479f1f07…62786f` | same | yes |
| differs on exactly 3 of 5 objectives (hERG, Caco2, Half_Life) | same; `task_names` and `signs` identical | yes |
| validation-run residuals, 7 iterations, max `1.1102230246251565e-16` | identical residual list, element for element | yes |
| 53 qualifying runs, worst residual `1.6653345369377348e-16` | max abs residual `1.665e-16`, mean `4.661e-17` | yes |
| holo snapshot worst residual `0.14414879083395205` | max abs `1.441e-01`, mean `7.084e-02` | yes |
| holo inflation min/max/mean `0.032386 / 0.144149 / 0.070838` | `0.0324 / 0.1441 / 0.0708` (mean **+47.9 %** relative) | yes |
| the 0.356-vs-0.478 pair | archived `0.3562050279642957`, canonical `0.35620502796429576`, holo `0.4782917308944767` | yes |
| pooled clipping: PfDHFR 18@0 / 18@1, hDHFR 48@0 / 79@1, hERG 0/0, Caco2 0/0, Half_Life 1@0 / 6@1 | identical counts; `n_finite` 4061 docking / 4165 ADMET | yes |
| 2 of 89 clip on PfDHFR, 5 saturate on hDHFR (`benchmark_thicccc_2026-08-16/mogp/seed_0`) | exactly 2 and exactly 5; 89 finite of 90 rows | yes |
| raw-kcal units, LE falsification | raw → `0.35620502796429576`; LE-substituted → **exactly 0.0** | yes |
| root bounds bit-exact vs current library | all three ADMET bounds bit-exact against the 26,660-molecule screened library | yes |
| Half_Life float round-trip clip | 1 row at `0.3763003051280975`, exactly `5.551115123125783e-17` below the bound | yes |

Two further checks of my own:

- **Recomputed `pareto_size` matches the archive.** On
  `benchmark_thicccc_2026-08-16/mogp/seed_0` the final front recomputes to
  **62** members under the canonical bounds, equal to the archived
  `history.csv` `pareto_size` of 62. The bounds reproduce not just the scalar
  hypervolume but the front's cardinality.
- **The "zero contribution despite being on the front" claim is exact, not
  approximate.** Both PfDHFR-clipped rows (56 and 80) *are* on the recomputed
  Pareto front. Deleting both and recomputing gives
  `0.35620502796429576` — identical, delta **exactly 0.0**. So two reported
  front members contribute literally nothing.

## One number that differed, and why (resolved, not an error)

The artifact says the pooled rows contain **2437** distinct objective tuples.
My first count said 2473. This is a dedup convention difference, not a defect:

```
np.unique(Y, axis=0)                  -> 2473   (NaN compares unequal)
pandas drop_duplicates (NaN == NaN)   -> 2437   <- the artifact's convention
  of those, docking-finite            -> 2369   <- the artifact's deduplicated n_finite
```

The artifact's 2437 / 2369 pair is internally consistent and correct. Mine was
counting NaN-bearing rows as distinct. No number in the artifact changes.

## One phrasing imprecision worth knowing

The frozen artifact's `rejected.why_rejected` says the holo sweep's *"own
manifest declares its hypervolume NOT comparable to the preceding sweep."*
That is true but easy to misread. The manifest's `comparability.NOT_comparable`
field compares holo to **`matrix_results/`** (the earlier ligand-efficiency
sweep), on the grounds that LE and raw kcal measure different volumes. It says
nothing about the later `benchmark_*` sweeps. The holo snapshot's rejection
rests on the two load-bearing facts, which both hold independently: it
reproduces no archived `history.csv`, and its manifest records a
**3,610-molecule** library (`lib_pull` 6000) against the 29,678-molecule pull
behind the canonical numbers.

## Confirming the additional stale-prose finding in `loop.py`

`SETTLEMENT.md` §4 flags this; I confirm it and note the scale consequence.

```python
# loop.py L94-99
# Margin (kcal/mol) below evaluation.DOCKING_KCAL_MIN at which we
# warn that observed docking ligand efficiencies are saturating ...
# On the LE scale (range ~0.55 wide) this is far smaller than the old raw-kcal margin.
DOCKING_SATURATION_MARGIN = 0.25
```

The comment reasons about an **LE span of ~0.55**, where 0.25 is a large
fraction. The live frame is raw kcal with a span of **6.0**, where 0.25 is
**4.2 %**. So this is not only stale prose: the warning threshold itself was
sized for a scale the code no longer uses. `_warn_if_docking_saturates`
(L324-353) carries the same stale "ligand efficiency" wording while comparing
against `DOCKING_KCAL_MIN`. Per the charter I changed nothing — flagging only.
This sits in `loop.py`, outside this directory.

## What I did not verify

- **`matrix_results_holo` still cannot be replayed.** It ships no
  `evaluated.csv` (its per-case run directories are absent; only sweep-level
  artifacts and `logs/` survive), so its snapshot cannot be positively confirmed
  against its own history by the decisive test. Its rejection is inferential —
  strongly supported, but inferential. I reached the same dead end the first
  pass did, by the same route.
- I did not re-derive the all-zeros reference point from first principles, and
  I did not widen the docking cube. Both remain open design questions for the
  re-run, exactly as §6 states.

## The critical finding, restated

Archived runs **did** use different bounds. All 53 replayable
`benchmark_*` runs used the canonical set. `matrix_results_holo` — whose
`FINAL_LEADS.csv` CLAUDE.md calls *"the paper's lead table"* — snapshotted a
different set at `2026-08-14 14:00`, nine hours before the root file was
regenerated at `22:55`. Its hypervolumes are in a different frame and are
**not** comparable to any benchmark hypervolume. The lead *molecules* are
unaffected; the *hypervolume numbers* are not transferable.
