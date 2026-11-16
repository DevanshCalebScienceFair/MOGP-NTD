# Head-to-head against Yong (arXiv:2508.14072): the plan

Written 2026-10-11. Read section 1 before anything else, because it changes the
claim you asked for.

---

## 1. The claim you wanted is not available, and I verified that myself

You asked to show "her number 1 would be our number 4". Against the tables on
thesis pages 110 to 115, that is arithmetically impossible.

**Those tables are not her results. They are her library sorted.** Her appendix
headings say so, "Dataset BEST SMILES (Top 20 SMILES)", and her Section 5.1
describes them as "the 20 best SMILES evaluated in the dataset".

**Verified independently on this laptop**, not taken on trust. Scoring the first
10,000 molecules of her library with her own vendored oracle code and sorting
reproduces her printed tables exactly:

| task | her printed top 5 | recomputed top 5 | agreement |
|---|---|---|---|
| Amlodipine MPO | 0.61237244, 0.58747999, 0.58387421, 0.57983351, 0.55856960 | identical | within 4.3e-09 |
| Perindopril MPO | 0.48795004, 0.48760869, 0.47756693, 0.47050403, 0.46770717 | identical | within 3.8e-09 |
| Fexofenadine MPO | 0.72892475, 0.72478167, 0.72041918, 0.71171583, 0.70899797 | identical | within 4.7e-09 |

An exhaustive sort reproduces her tables, in exact rank order, which means her
tables **are** the exhaustive sort. No optimizer can beat an enumeration of the
thing it is searching. Writing "we beat her 0.7289" would be refuted by anyone who
opens page 113, in about one minute.

### What is available instead, and it is better

**Her method does not reach her own tables.** Her repository ships its own run
logs. Computing the geometric mean of the three objectives she records, for each
of her four runs:

| her own run file | best geometric mean she achieved | where that lands in her own Table 8.3 |
|---|---|---|
| `bo_results_experiment_1.csv` | 1.4116668811 | **row 10** |
| `bo_results_experiment_2.csv` | 1.4116668811 | **row 10** |
| `bo_results_1.csv` | 1.3989038673 | **row 13** |
| `bo_results.csv` | 1.2955965840 | below her top 20 |

**Her method's number one is her library's number ten.** That is the real target.
If your optimizer, on her library, with her oracle, at her budget, finds three
molecules better than her method's best, then her best genuinely is your number
four. Same sentence, earned against her optimizer rather than against a sort.

---

## 2. What gets run: YONG-MATCH-R

Your optimizer and her actual published code, side by side, on her tasks.

**The primary quantity.** Best-of-acquired task score, the highest oracle score
among the 20 molecules an optimizer **chose**, excluding the 20 starting molecules
handed to both arms. Dimensionless, 0 to 1, higher is better. The starting set is
excluded because it is an achievement of neither method, and including it would
score the harness.

**The oracle.** Her vendored code at
`external/GP-MOBO/tdc_oracles_modified/`, not the public `guacamol` package, whose
benchmark score averages over a population and sits on a different scale.

> One discrepancy found and resolved in her favour: her thesis Tables 2.1 and 4.2
> print Gaussian widths of 2 and 2 for the polar-surface-area and logP terms. Her
> **code** uses 10 square angstroms and 1 log unit. Using the printed values moves
> top-20 scores by up to 0.1239 and reorders the table. The code wins, and the
> methods section says so.

**The library.** The first 11,000 lines of `guacamol_v1_train.smiles` in file
order, which is how her loader reads it. Rows 10,000 to 10,999 are her held-out
block for setting model hyperparameters. Rows 0 to 9,999 are shuffled by seed, the
first 20 become the shared start, the remaining 9,980 are the candidate pool.

### The grid

| arm | what it is |
|---|---|
| **HER@FP2** | her optimizer on the same 2048-bit binary fingerprint as yours. **The primary comparator.** |
| HER-PUB | her optimizer on her own unfolded count fingerprint. Reference arm. |
| **MOGP-ICM** | your coregionalized model plus your qNEHVI. **The method under test.** |
| MOGP-IND | same with independent models. Isolates coregionalization. |
| RANDOM | uniform sampling. The floor. |

Three tasks (Fexofenadine, Amlodipine, Perindopril MPO). The DockSTRING Toy MPO is
deliberately out of this batch. Budget 20 starting molecules plus 20 iterations of
one molecule each, for every arm.

- **Tier 1**, primary: 3 tasks x 10 seeds x 4 arms = **120 runs**
- **Tier 2**, reference: 3 tasks x 6 seeds x HER-PUB = **18 runs**
- **Tier 3**, sensitivity at 10 starting molecules: **12 runs**
- **Total 150 runs**

**Why ten seeds and not her three.** A paired signed-rank test asks whether
differences lean consistently one way. With 3 pairs its smallest possible
two-sided p is far above 0.05, so no outcome can ever be significant, which is why
her thesis contains no statistical test anywhere. With 10 pairs the floor is about
0.002 and the design survives two adverse seeds.

---

## 3. Pre-registration, committed before the sweep launches

**Primary endpoint.** Per task, the paired difference in best-of-acquired score
between MOGP-ICM@FP2 and HER@FP2 across seeds 0 to 9. Two-sided Wilcoxon
signed-rank, Holm-corrected across the three tasks.

- **WIN**: corrected p below 0.05 with median difference above 0, on 2 of 3 tasks.
- **LOSS**: median difference at or below 0 on 2 or more tasks.
- **INCONCLUSIVE**: positive medians but nothing clears correction. This is a
  genuinely likely outcome and is a verdict in its own right, not a near-win.

**Four additional loss conditions, any of which gets published as a negative.**

1. **The floor.** Your median must exceed the random arm's 95th percentile on
   every task. A method that cannot beat uniform sampling at 40 calls has nothing
   to say. This is reported first, before anything else. It is a live risk: her own
   figures put every method below the 20th-best library molecule on every task.
2. **The interleave.** If the median rank of her best molecule in the pooled table
   is 1, 2 or 3, the sentence you want is false and the report prints the actual
   integer. There is no code path that hides an unflattering rank.
3. **The expected null.** Coregionalized versus independent is **expected to show
   nothing**, for two reasons already on record: the acquisition forces a diagonal
   posterior so the learned coupling never reaches it, and your own earlier work
   found no benefit under co-located labels, which these tasks are. If it wins
   here, that is a surprise to explain, not a victory to bank.
4. **Sensitivity.** If the verdict flips between 20 and 10 starting molecules, the
   honest report is that the comparison is sensitive to a number her own sources
   disagree about. Her prose says 10, her code says 20.

**Seeds are fixed at 0 through 9 in advance.** Adding seeds after seeing a p value
is the practice your five published negatives exist to prevent.

---

## 4. Running it on the Mac Studio

Two sessions. The harness does not exist yet, so a same-night launch would be
fire-and-hope.

### Day one, attended, four to six hours

**Step 1.** Record the machine, get a clean worktree.
```
cd ~/mogp-main-vscode/MOGP-NTD
system_profiler SPHardwareDataType | grep -E "Chip|Cores|Memory"
sysctl -n hw.physicalcpu
git fetch origin feat/numpy-bo-rewrite
git worktree add ../MOGP-rewrite feat/numpy-bo-rewrite
cd ../MOGP-rewrite && git rev-parse HEAD
```

**Step 2.** Her code at the pinned commit. Copy from the laptop so the bytes match.
```
rsync -a ~/mogp-main-vscode/MOGP-NTD/external/GP-MOBO/ external/GP-MOBO/
git -C external/GP-MOBO rev-parse HEAD   # ca623180f42fc24a33ed371b295841d2cce815ba
```

**Step 3.** Settle the environment. Check whether two RDKit dist-info directories
appear; one stale metadata record is fine and gets recorded, two genuinely
importable RDKits is a stop. **Do not install the `guacamol` package**, it is not
needed and pulls a third RDKit.

**Step 4.** Build the harness (prompt at the end of this file).

**Step 5.** Gates. Nothing launches until this exits 0.
```
conda run -n mogp-drug python -m pytest -q bench_yong/test_bench_yong.py
```

**Step 6.** Smoke test one run to replace every estimate with a measurement, and
to set the timeout.
```
/usr/bin/time -l conda run -n mogp-drug python -u bench_yong/run_one.py \
  --task fexofenadine --arm HER@FP2 --seed 0 --n-init 20 --n-iterations 20 \
  --output-dir /tmp/yong_smoke
```
Read wall clock, peak memory, and final Pareto front size. Front size drives her
acquisition cost almost entirely: her Monte Carlo expected-improvement costs
0.0317 seconds per candidate at a 5-row front and 0.0851 at 15 rows.

**Step 7.** Job count from measurement:
`JOBS = min(physical_cores - 2, floor(0.25 x RAM_GB / 2.0))`. Default 8.

### Night one, unattended

```
cd ~/mogp-main-vscode/MOGP-rewrite
OUT=bench_yong_results_$(date +%Y%m%d)
PYTHONUNBUFFERED=1 caffeinate -dimsu nohup conda run -n mogp-drug python -u \
  bench_yong/run_sweep.py \
  --tier1-tasks fexofenadine amlodipine perindopril \
  --tier1-arms HER@FP2 MOGP-ICM@FP2 MOGP-IND@FP2 RANDOM \
  --tier1-seeds 0 1 2 3 4 5 6 7 8 9 \
  --tier2-tasks fexofenadine amlodipine perindopril \
  --tier2-arms HER-PUB --tier2-seeds 0 1 2 3 4 5 \
  --tier3-tasks fexofenadine --tier3-arms HER@FP2 MOGP-ICM@FP2 \
  --tier3-seeds 0 1 2 3 4 5 --tier3-n-init 10 \
  --n-init 20 --n-iterations 20 --batch-size 1 \
  --jobs 8 --mem-cap-gb 2.0 --per-run-timeout-h 12 --resume \
  --output-dir $OUT > $OUT.log 2>&1 &
```

**Seeds are listed one by one on purpose.** Passing a count instead of a list is
the trap that already produced a sweep of identical runs on this project.

### Cost and memory, measured rather than modelled

Her acquisition is the entire critical path: **105.7 of the 110.8 core-hours**.
Your arm costs about 4 minutes per run against her 1.1 to 2.8 hours, which is
roughly 185 times cheaper per candidate. At 8 jobs expect **14 to 17 hours**; at
16 jobs, 7 to 9.

Memory per child peaked at **0.550 GB** measured. Budget 0.8 GB, hard cap 2.0 GB.
At 8 jobs that is 6.4 GB expected and 16 GB worst case, below the 20 GB line that
took the machine down before.

> **A correction worth knowing.** The obvious memory guard, `ulimit -v` or
> `setrlimit`, **does not work on macOS**. The shell rejects it and `setrlimit`
> raises an error because the kernel refuses to lower the hard limit; a 4 GB
> allocation succeeded under an attempted 3 GB cap. A plan relying on it would have
> had no cap at all and would have looked correct. The harness samples resident
> memory every 10 seconds and kills the offending child instead.

---

## 5. What the deliverable looks like

Per task, ten rows, one per seed:

```
seed | her_best_score | her_best_rank | n_mogp_above | mogp_best | Delta
```

plus the median rank and **the full distribution of all ten ranks**, always
printed together. A single-seed interleave is close to meaningless: her own logs
put her best at library rank 10 twice, rank 13 once, and outside the top 20 once.

Every molecule is **re-scored from its SMILES inside the report script**, never
copied from run files, so a logging bug in one arm cannot move the ranking.

Her published numbers appear **exactly once** in the whole deliverable, in a
visually separated block captioned as the library ceiling that neither method
reached, and never as her method's performance.

---

## 6. What cannot be claimed

- **Not** that you beat 1.9478, 0.7289, 0.6124 or 0.4880. Those are library maxima.
- **Not** a speedup over her. Her thesis reports no hardware and no runtime
  anywhere, so there is no number of hers to beat. Report absolute cost on one
  named machine, beside her code's cost on the same machine.
- **Not** a hypervolume comparison against her traces. Her reference point is
  re-inferred every iteration from parameters she never states, which is why her
  own logged starting hypervolumes range from 0.299 to 0.583 across her own runs.
- **Not** that you beat her best method. Her own Section 5.2 says her UCB-PT
  baseline beats her GP-MOBO on all three of these tasks. Phrase every claim as
  "better than GP-MOBO, the arm her own thesis reports losing with."
- **Not** that a win is due to coregionalization. Your arm differs from hers on at
  least four axes at once.

**One asymmetry that runs in her favour and stays.** Her recipe scores a separate
1,000-molecule block to set her hyperparameters, so her arms consume 1,040 oracle
calls against your 40, a factor of 26. Removing it would stop being her method. It
stays, and the budget table states it, so a win reads as conservative.

---

## 7. Laptop track, starting the same night

Shares nothing with the Studio: no files, no cache, no library directory.

**Three hard isolation rules.** Never write to `data/docking_cache` from the
laptop. Never modify `external/GP-MOBO`. **Never start AutoDock Vina on the laptop
during the sweep**, because a stray dock writes cache rows under the laptop's
different receptor fingerprint and destroys the Studio's zero-docking guarantee.

1. **Data preparation**, zero risk. The full `guacamol_v1_train.smiles` holds
   1,273,104 molecules. Canonicalize, deduplicate, hold out a validation split, and
   flag the first 10,000 as the benchmark subset so a generated molecule can later
   be labelled inside or outside the search library. That flag is what separates a
   de novo claim from a retrieval claim.
2. **Representation.** Build tokenizers for both SELFIES and SMILES on day one,
   train SELFIES first, keep SMILES as the ablation.
3. **Baseline model**, small enough to train in an hour. Report validity,
   uniqueness and novelty, defined before they are measured, and round-trip
   similarity, which is the check that catches a latent space that looks trained
   and means nothing.

---

## Paste-ready, for the Mac Studio

```
Build bench_yong/ in ~/mogp-main-vscode/MOGP-rewrite implementing YONG_HEADTOHEAD_PLAN.md:
run_one.py (one task x arm x seed, writes evaluated.csv with an is_initial flag, run.json,
peak_rss.txt), run_sweep.py (subprocess per cell, --jobs, --resume, 10s RSS sampling with
SIGTERM/SIGKILL on breach, never setrlimit), report.py (reads only finished files from disk,
re-scores every SMILES from the oracle, paired Wilcoxon + Holm, per-seed interleave table with
the full rank distribution), and test_bench_yong.py with gates G1-G9. Commit
bench_yong/PREREGISTRATION.md before run_sweep.py can launch.
```
