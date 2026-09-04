# Does the missing-data rewrite cost anything when nothing is missing? No.

Six paired seeds, complete data on both arms. Figure: `F18_old_vs_new_model.png`.

## What differs

Both arms are the same intrinsic coregionalization model, same seeds, same
initial molecules, same acquisition, same 2,000-candidate pool.

| | OLD | NEW |
|---|---|---|
| file | `mogp_coregionalized.py` | `mogp_hadamard.py` |
| structure | Kronecker `MultitaskKernel` | one entry per `(molecule, task)` |
| missing labels | **cannot** — silently dropped the task and returned NaN | **yes** |
| noise | one per task | one shared |

## Result: six endpoints, six ties

| endpoint | new | old | delta | 95% CI | new wins | p |
|---|---|---|---|---|---|---|
| final hypervolume | 0.3948 | 0.4010 | -0.0063 | [-0.0169, +0.0004] | 2/6 | 0.313 |
| AUC of the HV curve | 0.3345 | 0.3408 | -0.0063 | [-0.0143, +0.0005] | 3/6 | 0.438 |
| Pareto front size | 169.3 | 170.7 | -1.33 | [-7.67, +4.67] | 2/6 | 0.688 |
| physical molecules | 251.3 | 251.7 | -0.33 | [-2.50, +1.67] | 3/6 | 0.719 |
| top-20 selectivity | 2.550 | 2.644 | -0.094 | [-0.278, +0.082] | 3/6 | 0.563 |
| best PfDHFR (kcal/mol) | **-11.202** | -11.103 | **+0.098** | [+0.030, +0.180] | 4/6 | 0.125 |

Nothing separates them. The rewrite's one real concession — a single shared
noise instead of one per task — appears harmless.

**One seed carries the gap.** Per-seed hypervolume deltas are -0.0057, **-0.0316**,
+0.0024, -0.0016, -0.0014, +0.0004. Drop seed 1 and the mean is **-0.0013**. That
is the same shape that looked like a finding in the ICM sweep and turned out to
be noise.

## A determinism check, for free

The new arm re-ran the identical configuration from the earlier campaign,
independently invoked hours later. All six seeds reproduced to four decimals:

| seed | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| earlier | 0.3963 | 0.3703 | 0.4012 | 0.4000 | 0.4017 | 0.3991 |
| this run | 0.3963 | 0.3703 | 0.4012 | 0.4000 | 0.4017 | 0.3991 |

Not what the run was for, but it is the first explicit determinism check on the
pipeline and it passed 6/6. Worth stating in the paper: same machine, same
config, same answer.

## The complete picture: both regimes

`F23_model_comparison_full.png` puts the two regimes side by side, because the
tie above is only half the comparison.

| hDHFR labels | OLD (Kronecker) | NEW (Hadamard) |
|---|---|---|
| 100% | trains | trains |
| 75% | **refuses** | trains, predicts both |
| 50% | **refuses** | trains, predicts both |
| 25% | **refuses** | trains, predicts both |

Measured directly, not asserted. And "refuses" is the *fixed* behaviour — before
the guard added on this branch it did not raise at all: it silently dropped the
task and returned NaN predictions for half the objective, so a run would finish
and report plausible numbers for the other half.

Under missing labels the new model also beats a non-sharing model given the same
labels, by a margin that grows as labels thin out (F13: p ≤ 0.004 below 50%).

## Verdict

**The new model strictly dominates for practical use.** It matches the old one
on every endpoint measured, and it accepts missing labels — which the old one
could not, and worse, did not refuse: it silently dropped the task and returned
NaN predictions for it (see `test_icm_equivalence.py`).

## Honest limits

- **n = 6** caps the two-sided Wilcoxon at p = 0.0312, so this is *absence of
  evidence*, not proof of equivalence.
- Point estimates lean slightly to the OLD model on **5 of 6** endpoints, and the
  hypervolume interval `[-0.0169, +0.0004]` only barely includes zero. **A small
  penalty cannot be ruled out at this sample size.** If it matters, extend:
  `SEEDS="0 1 2 3 4 5 6 7 8 9" ./run_model_comparison.sh`
- The one endpoint favouring the new model, best PfDHFR, has a CI excluding zero
  (`[+0.030, +0.180]`) but a Wilcoxon p of 0.125 — the sign test and the bootstrap
  disagree because several seeds tie exactly. Do not quote it as significant.

## Reproducing

```bash
./run_model_comparison.sh
python analysis_scripts/model_comparison_analysis.py
```
