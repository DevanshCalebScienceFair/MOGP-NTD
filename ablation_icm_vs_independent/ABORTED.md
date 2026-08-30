# ICM vs independent ablation — ABORTED at Gate 0

**Nothing was launched.** The ablation was not run, the benchmarked path was not
modified, and `campaign_results/`, `evaluation_bounds.json` and the analysis
branch's work are untouched. This directory holds only the gate evidence.

## Gate 0: FAIL

Ten molecules from `campaign_results/seed_0/mogp/seed_0/evaluated.csv` with
non-NaN stored scores, re-docked here against both receptors at the same box,
exhaustiveness and seed, cache disabled. Tolerance 0.05 kcal/mol.

| # | Pf stored | Pf here | Δ | hD stored | hD here | Δ |
|---|---|---|---|---|---|---|
| 1 | −6.955 | −6.935 | 0.020 | −6.944 | −6.834 | **0.110** |
| 2 | −9.249 | −9.210 | 0.039 | −9.091 | −9.089 | 0.002 |
| 3 | −7.903 | −7.903 | 0.000 | −8.342 | −8.352 | 0.010 |
| 4 | −9.497 | −9.421 | **0.076** | −10.130 | −10.110 | 0.020 |
| 5 | −7.736 | −7.553 | **0.183** | −7.878 | −8.029 | **0.151** |
| 6 | −7.038 | −7.650 | **0.612** | −8.021 | −8.462 | **0.441** |
| 7 | −6.568 | −6.526 | 0.042 | −7.232 | −7.194 | 0.038 |
| 8 | −8.956 | −8.990 | 0.034 | −7.889 | −7.902 | 0.013 |
| 9 | −8.382 | −8.759 | **0.377** | −7.172 | −7.163 | 0.009 |
| 10 | −9.510 | −9.482 | 0.028 | −9.014 | −8.961 | 0.053 |

**Worst 0.612 kcal/mol. 8 of 20 docks exceed the tolerance. Mean |Δ| 0.113.**

Per the gate rule, the ablation was aborted. A machine effect of this size would
be confounded with the surrogate effect, and molecule 6 alone (0.612 on PfDHFR)
is comparable to real differences in binding quality.

## Why it failed: a machine effect, not oracle noise

The two causes look identical in the table above and lead to opposite responses,
so they were separated by re-docking the same ten molecules a SECOND time on this
machine and comparing it to itself:

| comparison | max Δ | mean Δ | over 0.05 |
|---|---|---|---|
| local vs local | **0.000** | 0.000 | 0/20 |
| local vs Studio | 0.612 | 0.113 | 8/20 |

**This machine reproduces itself bit-for-bit on all 20 docks.** Vina here is
fully deterministic — the ETKDGv3 conformer is seeded (`randomSeed = 0xF00D`) and
the Vina seed is fixed. So the disagreement is not nondeterminism that the Studio
would also show; the two machines are computing genuinely different numbers.

Signed local − Studio: mean −0.051, sd 0.198. Scatter around a near-zero mean,
not a constant offset, which is what a changed receptor produces rather than a
changed box or scoring constant.

## Where the difference most likely is

The two `.stamp` files match the Studio's committed values, but **that does not
clear the receptor.** `_prep_stamp` hashes only `prep_version | cofactors |
pdb_id` — deliberately "short of the file itself". It exists to force a rebuild
when prep *logic* changes; it is blind to the prepared file's contents.

The quantity that actually determines comparability is `receptor_fingerprint()`,
the sha256 of the prepared PDBQT — downstream of Open Babel, so it carries the
added hydrogens and Gasteiger charges. `docking.py` documents exactly this risk:

> Hashing the PDB would catch a prep-logic change but miss an Open Babel version
> bump silently altering charges — which changes every score without changing any
> file we control.

That is the leading hypothesis: **a different Open Babel build produced different
partial charges, changing every score while every tracked file stayed identical.**
This machine has Open Babel 3.1.0 (Nov 30 2023), RDKit 2024.03.6, Vina 1.2.7.

It cannot be confirmed from here, because the Studio's PDBQT hash was never
recorded on this machine — precisely the check listed as still outstanding.

## The decisive comparison (10 seconds on the Studio)

```
shasum -a 256 1J3I_clean.pdbqt 1U72_clean.pdbqt
obabel -V
python -c "import docking; print(docking.oracle_fingerprint('PfDHFR')); print(docking.oracle_fingerprint('hDHFR'))"
```

This machine's values:

| | |
|---|---|
| `1J3I_clean.pdbqt` sha256 | `d9c3ee0da6f09bfbcf7ca0ba3385bec67eb64c3f0457f22d0dfc05b743eb6b26` |
| `1U72_clean.pdbqt` sha256 | `0bfe59fbcf6d597e6d72e30ab65ca35dc58eb0123e9e399b6cba548489ff83a2` |
| `oracle_fingerprint("PfDHFR")` | `35ec75e9000381147734d9dd9de0e66a082d2faa7800728a2d0e6b64ec725a43` |
| `oracle_fingerprint("hDHFR")` | `577ef708b5ecea338bb6cfe4941b7e4baa100d5a6f04609263eefda2c5106483` |
| Open Babel | 3.1.0 (Nov 30 2023) |

If the PDBQT hashes differ, the receptors differ and copying the Studio's
prepared PDBQTs (or its docking cache) fixes it. If they match, the difference is
in the ligand path instead and needs a different look.

## Gate 1 was not reached, and would not have needed new code

Investigated before building anything: **the ablation already exists in the
codebase.** The `campaign` branch's `loop.py` carries
`--model {coregionalized,independent}` with `DEFAULT_MODEL = "coregionalized"`,
and `run_benchmark_seeds.py` constructs `BOLoop(...)` without passing `model=`,
so the campaign's MOGP arm is the ICM. `resolve_train_fn("independent")` returns
`mogp.train_mogp`, whose `MOGPModel` is a batch-independent multi-output GP —
one scaled-Tanimoto GP per task over the same 2048-bit Morgan fingerprints, with
no cross-task terms. `MOGPCoregionalized` differs only by the `IndexKernel` task
covariance inside a `MultitaskKernel`. That is exactly the single-axis swap the
ablation calls for, so no `ModelListGP` needed writing and there was no flag to
leak.

**One correction to the brief.** It specifies five independent GPs, one per
objective. The architecture models **two**: the loop is grey-box, the GP covers
only the two docking objectives, and the three ADMET objectives are known exactly
for every candidate and enter through `CompositeKnownADMETObjective` rather than
being predicted. Fitting GPs to all five would model values that are already
known exactly, confounding "no coregionalization" with "predict ADMET instead of
using it" — the opposite of isolating the ICM. The correct ablation is two
independent Tanimoto GPs against the ICM over the same two objectives, which is
what `--model independent` already does.

## What is needed to run this

1. Resolve the oracle difference (the comparison above), **or**
2. Re-run the ICM seed-0 arm on this machine too, so both arms share one oracle.
   That is clean and needs no Studio access, but it doubles the compute and
   changes the experiment as specified, so it was not started unattended.

Once the oracle question is settled the run itself is one command on the
`campaign` branch, no code changes:

```
python loop.py --model independent --seed 0 --n-init 40 --batch-size 5 \
    --n-iterations 50 --mogp-iters 200 --acquisition-pool-size 2000 \
    --output-dir ablation_icm_vs_independent/seed_0_independent
```

Still to label as **n=1, a pilot, not a benchmark result.**
