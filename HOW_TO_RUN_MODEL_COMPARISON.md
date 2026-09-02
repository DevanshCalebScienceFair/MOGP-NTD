# Old vs new model: how to run it

## What it compares

| | old | new |
|---|---|---|
| name | `coregionalized` | `hadamard` |
| file | `mogp_coregionalized.py` | `mogp_hadamard.py` |
| structure | Kronecker `MultitaskKernel` | one entry per `(molecule, task)` |
| missing labels | **cannot** — used to drop the task silently | **yes** |
| noise | one per task | one shared |

Both are the same intrinsic coregionalization model. This run gives every
molecule both docking scores, so it asks exactly one thing:

> **Does the rewrite cost anything when the data is complete?**

A tie means the new model strictly dominates: same quality, and it also handles
gaps. Anything worse is a real price for the flexibility and belongs in the paper.

## Run it

```bash
cd /Users/devansh/mogp-main-vscode/MOGP-NTD
./run_model_comparison.sh
```

- 12 runs (6 seeds x 2 models), **~25 min each, so roughly 5 hours**.
- Sequential, 20 GB memory watchdog, **resumable** — re-running skips anything
  already finished, so it is safe to stop with Ctrl-C and restart.
- Fewer seeds: `SEEDS="0 1 2" ./run_model_comparison.sh`

Watch it:

```bash
tail -f /tmp/model_comparison.log
```

(or `tail -f model_comparison/logs/hadamard_seed0.log` for one run's detail)

## Then analyse it

```bash
python analysis_scripts/model_comparison_analysis.py
```

Prints a paired comparison on six endpoints — final hypervolume, AUC, front
size, physical molecules found, top-20 selectivity, best PfDHFR — each with a
bootstrap CI, a per-seed win count, and a Wilcoxon p-value.

## Two things to know before reading the result

**The gate.** Each run echoes its own resolved settings and the script aborts if
they do not match what was asked. This exists because `loop.py` had no `--seed`
flag until 2026-09-01 and a whole 6-seed sweep silently reused seed 42. If you
see `ABORT`, nothing was wasted — read the two lines it prints.

**n = 6.** The minimum two-sided Wilcoxon p at six pairs is 0.0312, so a "tie" is
*absence of evidence at small sample size*, not proof of equivalence. If the
point estimates lean one way and you care about the answer, run more seeds:
`SEEDS="0 1 2 3 4 5 6 7 8 9"`.

## Expected runtime and cost

Docking is cached, so seeds already explored by earlier campaigns run faster.
Acquisition is ~98% of wall clock; the `--acquisition-pool-size 2000` in the
script is what keeps a run at 25 minutes instead of 3 hours, and it matches every
other experiment on this branch. Do not remove it.
