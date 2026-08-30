# Tier 1 coverage diagnostics — how to run, and what the outputs mean

Implements the three Tier 1 items from `LITERATURE_REVIEW.md`. All three are
post-hoc over already-cached campaign output. `tier1_analysis.py` imports
`evaluation` and `mogp` for the shared normalization frame and objective order
and nothing else from the loop — it never constructs a GP, an acquisition
function or a docking oracle, so **the 117-hour 10-seed campaign stays valid**.
It also refuses to write inside `aggregate_10seed/` or any `seed_N/` directory.

## Running it

```bash
python tier1_analysis.py all \
    --campaign-root campaign_results \
    --out campaign_results/aggregate_10seed_cleanGPMOBO/tier1 \
    --library-dir data/library \
    --seeds 0-9
```

Subcommands `umap`, `circles`, `igdplus` run one analysis each. Runtime is
dominated by the UMAP fit: **~56 s** for the 29,678 × 2048 library on an M1
(measured), everything else seconds. Budget two to three minutes for `all`.

`--methods` maps labels to directories. `GPMOBO` resolves to `gpmobo_clean`,
`gpmobo_cleaned`, `gpmobo_rawkcal` or `gpmobo`, first match wins, so the
corrected raw-kcal re-run is picked up ahead of the superseded LE arm. Check the
`loaded N runs` line and the `oracle:` line before trusting any output: the
oracle front should reproduce **3,671 unique docked molecules → 411-molecule
front, HV 0.4215**. If it does not, the wrong arms were loaded.

## Outputs

| File | What it is |
|---|---|
| `umap_oracle_coverage.png` | 3 panels: library + oracle front; found vs missed; MOGP coloured by acquisition iteration |
| `umap_verdict.json` | The decision and every number behind it |
| `umap_missed_molecules.csv` | Each missed oracle-front molecule with its distance to the nearest MOGP-evaluated molecule |
| `circles_per_seed.csv` / `circles_summary.csv` | #Circles at t = 0.60 and 0.75 |
| `igdplus_per_seed.csv` / `igdplus_summary.csv` / `igdplus_NOTE.txt` | IGD+ per method, with the caveat that must travel with it |
| `oracle_front.csv` | The reference set, so downstream work does not rebuild it |
| `umap_embedding.npy` | The 2D embedding, for re-plotting without a refit |

## How the UMAP verdict is decided

**Not by looking at the figure.** UMAP is a non-metric embedding tuned for local
structure, and "sits on the periphery in 2D" is exactly the kind of claim it
distorts. The verdict is computed in the original 2048-bit fingerprint space,
where the distances are the ones the Tanimoto kernel and the `diversity_threshold`
filter actually see:

> A missed oracle-front molecule counts as **INSIDE** a sampled region when MOGP
> evaluated some molecule within Tanimoto distance 0.4 of it (similarity ≥ 0.6) —
> the loop saw that chemistry and did not pick this member.
>
> - **≥ 50% inside → `acquisition_rule`.** The molecules were reachable and were
>   passed over, so the fix is the acquisition reference point (Tier 2 §4.1).
> - **< 50% inside → `pool_sampling`.** They are chemistry the 2,000-candidate
>   pool never showed the acquisition function, so the fix is cluster-stratified
>   pool sampling (Tier 2 §4.2).

Three supporting distributions are reported alongside, missed vs found, each
with a Mann-Whitney p: distance to the nearest MOGP-evaluated molecule, the count
of MOGP-evaluated molecules within distance 0.4, and library sparsity (distance
to the 10th nearest library neighbour). The last one matters because it separates
"peripheral chemistry" from "MOGP simply did not go there" — if the missed
molecules are in *library*-sparse regions, that is a property of the library, not
of the acquisition rule.

The panel colouring uses one seed. Found/missed use the **union across all MOGP
seeds**, and `umap_verdict.json` reports both the union count and the per-seed
mean so it is unambiguous which one reproduces the campaign's `247`.

## Reading IGD+

`igdplus_NOTE.txt` is written next to the numbers and should be quoted with them.
The short version: IGD+ and hypervolume are computed over the same normalized
vectors, so they **agree by construction**. IGD+ is the *convergence* column of
the Li & Yao (2019) four-aspect framing — a standard, Pareto-compliant second
confirmation that MOGP converges closer to the oracle front. It is **not**
independent evidence about the coverage gap.

`igd_plus_front` and `igd_plus_all_evaluated` are expected to be **exactly
equal**: d+ is monotone under domination and the metric minimizes over A, so
dominated points can never supply the minimum. A discrepancy means the front
extraction disagrees with the dominance test, and is a bug, not a finding.

## Reading #Circles

The threshold is a Tanimoto **distance**, so the keep test is
`similarity < 1 - t`. Exact #Circles is a maximum independent set and NP-hard;
this is the standard greedy sphere-exclusion approximation, which is
order-dependent. Input order is each run's acquisition order, applied identically
to every method, so cross-method comparison is like-for-like while each absolute
value is a lower bound on the true maximum.

Reported over each seed's final Pareto front (mean ± sd across seeds), over each
method's docked union, and — as ceilings — over the oracle front and the whole
docked pool.
