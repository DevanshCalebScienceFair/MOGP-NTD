# MatrixRunResults: everything from the GP-MOBO head-to-head run

Branch created 2026-10-27 from `feat/numpy-bo-rewrite` at `4232619`, to carry the
complete record of the 2026-10-22 benchmark off this machine. It holds the code,
the pre-registration chain, all 240 run cells and both generated reports.

**Start here: [`bench_yong/FINDINGS.md`](bench_yong/FINDINGS.md).** It is the
single deliverable and replaces every earlier summary.

## What ran

| sweep | output directory | cells | result |
|---|---|---|---|
| Tiers 1 to 3 | `bench_yong_results_20261022/` | 150 of 150 complete | primary INCONCLUSIVE; L1-REVISED floor cleared |
| Tier 4 (her published protocol) | `bench_yong_results_tier4_20261022/` | 90 of 90 complete | primary INCONCLUSIVE; L1-REVISED floor FIRED |

## Commit chain, in order

| commit | what it is |
|---|---|
| `f8e5254` | pre-registration (`bench_yong/PREREGISTRATION.md`) and the harness, committed before any sweep ran |
| `0cdc0d0` | report fix: say UNDETERMINED / NO DATA instead of implying a result |
| `e040d65` | fixes from an independent adversarial review of the harness |
| `1cfcff2` | `bench_yong/HER_CONFIG.md`: which configuration produced her published GuacaMol results |
| `d2303e0` | **Amendment 1**, committed before Tiers 1 to 3 finished and before any result was read |
| `e17685c` | report: Amendment 1 applied, and the oracle-budget paragraph corrected |
| `53d994e` | **Amendment 2** (Tier 4), which declares in its own section 1 that it was written after the Tier 1 to 3 results were read |
| `4124a17` | Tier 4 harness |
| `0e9af20` | Tier 4 gates |
| `4232619` | `bench_yong/FINDINGS.md` |
| this commit | the run data, normally gitignored, force-added for analysis elsewhere |

Read Amendment 1 before Amendment 2. Amendment 1 has pre-unblinding standing;
Amendment 2 does not, and says so.

## Data layout

Both result directories have the same shape.

```
bench_yong_results_<name>/
  manifest.json              the sweep PLAN, written once, plus every invocation
  sweep_status.json          rebuilt from the cell directories on disk at each poll
  gates/*.json               one record per gate, with its measured tolerances
  gates_pytest.log           the gate suite's own output
  report/report.md           the generated report
  report/report.json         every statistic the report computed, machine readable
  report/per_cell.csv        one row per cell: best-of-acquired, library rank, cost
  cells/<task>__<arm>__seed<N>__init<M>/
    evaluated.csv            THE RESULT: every scored molecule, is_initial per row
    run.json                 full config, provenance, versions, modifications
    timing.csv               per-iteration wall clock, candidates scored, front size
    peak_rss.txt             peak resident memory, getrusage
    peak_rss_watchdog.txt    peak resident memory, ps sampling by the sweep driver
    cell_status.json         launch, end, attempt, exit code, watchdog peak
    her_stdout.log           her driver's own prints (her arms only)
  cells_stdout_logs.tar.gz   see below
```

`evaluated.csv` is the primary artifact. `report.py` recomputes every number in
the reports from these files alone, re-scoring each molecule from its SMILES, so
the reports are reproducible from the cells without re-running anything:

```bash
python bench_yong/report.py --output-dir bench_yong_results_20261022
python bench_yong/report.py --output-dir bench_yong_results_tier4_20261022
```

Those two commands rewrite `report/` in place and should reproduce the committed
files byte for byte, except for one heading line that names the design.

## What is compressed, and why

`cells/*/stdout.log` (240 files, 549 MB raw, 2.0 MB gzipped) is archived as
`cells_stdout_logs.tar.gz` in each result directory. It is the subprocess stdout
captured by the sweep driver, which is her driver's prints duplicated from
`her_stdout.log` plus one `f1 shape: (1,)` line per oracle call, about 200,000
lines per HER cell. Nothing in it is needed to reproduce a number. Expand with:

```bash
tar -xzf bench_yong_results_20261022/cells_stdout_logs.tar.gz
```

`her_stdout.log` is kept raw in every cell, because it is small and it is her
code's own record of what it chose and why.

## What is deliberately NOT in this branch

| excluded | why | how to restore |
|---|---|---|
| `external/GP-MOBO/` (78 MB), her pinned comparator | the upstream repository publishes no LICENSE, so its code is not ours to redistribute (`gpmobo_ref.py` lines 19 to 24). It is fetched, never vendored | `git clone https://github.com/anabelyong/GP-MOBO.git external/GP-MOBO` then `git -C external/GP-MOBO checkout ca623180f42fc24a33ed371b295841d2cce815ba` |
| `external/GP-MOBO-guacamol-test/` (50 MB), her published drivers | same reason | `bench_yong/her_published.py::materialize` recreates it from the clone above with `git archive`, and verifies tree `825f7562` and the three driver blob hashes before use |
| the GuacaMol library file | ships inside her clone; its SHA-256 is recorded in every `run.json` | restored with the clone |

Every hash needed to reconstruct her side exactly is recorded in
`bench_yong/her_published.py`, `bench_yong/PREREGISTRATION_AMENDMENT_2.md`
section 5, and each cell's `run.json`.

## What has expired and cannot be recovered

These lived in temporary directories that the operating system has since cleared.
Their conclusions are committed; the raw files are gone.

| lost | where its conclusions live |
|---|---|
| the power-analysis scripts and their JSON output (`power.py`, `power_refine.py`, `power_amend.py`) | `bench_yong/PREREGISTRATION_AMENDMENT_1.md` section 3 carries the full operating-characteristic tables they produced, including the method, the sample counts and the exact-Wilcoxon validation |
| the independent code review's JSON output (7 agents, 6 confirmed findings) | the `e040d65` commit message lists every confirmed finding and its fix; `bench_yong/FINDINGS.md` section 3 carries the one remaining observation |
| the arXiv HTML of her thesis, and the intermediate table parse | `bench_yong/fixtures/thesis_tables.json` holds all 60 rows of her Tables 8.4 to 8.6 resolved to library line numbers, which is what gate G1 checks against |
| the nine pre-launch smoke cells (fexofenadine seed 0, all five arms) | gate G5 makes them identical to the sweep's own fexofenadine seed-0 cells, which are present here; the measurements taken from them are in `bench_yong/PREREGISTRATION_AMENDMENT_1.md` section 1 and `FINDINGS.md` |

The power tables can be regenerated from scratch: the method is fully described
in Amendment 1 section 3, and it needs only the library scores, which the oracle
reproduces in about 20 seconds.

## Reproducing the gates on another machine

```bash
pytest -q bench_yong/test_bench_yong.py
```

Needs her clone (above) and the `mogp-drug` environment. 44 tests, 33 gate
records. Gate G1 reproduces her thesis Tables 8.4 to 8.6 from her library file
and her oracle code, 60 of 60 rows; that is what licenses every number
downstream.

## One honest note on the timings

A third-party process (a game) held about one CPU core throughout the Tier 4
sweep, and is recorded in `bench_yong_results_tier4_20261022/manifest.json` under
`top_processes_at_launch`. It cannot affect any score, because every cell is
seeded and pinned to a single BLAS thread and all three Tier 4 arms reproduce
byte for byte, but it inflates the wall-clock and core-hour figures by an
unmeasured amount.
