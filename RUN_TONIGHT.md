# Tonight's runs

Nothing is running now. Both experiments are staged, gated and resumable.

```bash
cd /Users/devansh/mogp-main-vscode/MOGP-NTD
./run_tonight.sh
```

That runs both, **strictly one at a time** (~7.5 h). To run just one:
`./run_tonight.sh compare` or `./run_tonight.sh hdhfr`.

Watch it: `tail -f /tmp/tonight_*.log`

---

## 1 · Old vs new model (~5 h, 12 runs)

`coregionalized` (Kronecker ICM) vs `hadamard` (stacked-index ICM), 6 paired
seeds, **complete data on both arms**. One question:

> Does the rewrite cost anything when nothing is missing?

A tie means the new model strictly dominates — same quality, and it also handles
gaps. Anything worse is a real price for that flexibility and belongs in the paper.

The two differ deliberately in one place: Kronecker has one noise **per task**,
Hadamard has one **shared** noise. State that with any result.

## 2 · hDHFR ceiling (~2.5 h, 6 runs)

The shared −5.0 upper bound truncates exactly the direction hDHFR is optimized
toward. Measured on this repo's data: **19 of the 50 most selective molecules
clip**, collapsing **13.14 kcal/mol** onto the single value 1.0. This arm reruns
the same configuration with the ceiling at 0.0.

**The two arms are scored in different normalization frames, so their
hypervolumes are NOT comparable.** The analyser refuses to compare them and uses
raw-kcal endpoints instead. Do not quote a hypervolume across frames.

---

## Then

```bash
python analysis_scripts/model_comparison_analysis.py
python analysis_scripts/hdhfr_bound_analysis.py
```

## If something goes wrong

**`ABORT: header does not confirm ...`** — a gate caught a mismatch before
wasting time. Nothing is lost; read the two lines it prints. These gates exist
because `loop.py` had no `--seed` flag until yesterday and a whole 6-seed sweep
silently reused seed 42.

**`WATCHDOG: ... hit NNGB`** — a run exceeded 20 GB and was killed. The others
continue; re-run to retry that one.

**Nothing appears to happen for ~2 min at the start of each run** — that is the
library load. Normal.

## Reading the results honestly

`n = 6` gives a minimum two-sided Wilcoxon p of **0.0312**. A "tie" is *absence
of evidence at small sample size*, not proof of equivalence. If the point
estimates lean one way and it matters, extend:
`SEEDS="0 1 2 3 4 5 6 7 8 9" ./run_model_comparison.sh`
