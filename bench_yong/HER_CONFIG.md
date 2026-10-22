# Which configuration produced Yong's GuacaMol results

Settled 2026-10-22 from her repository and her thesis, while the head-to-head
sweep was running. Read-only investigation: no run, no file in `external/GP-MOBO`
changed. The question decides whether we may say she had a 26-fold oracle budget
advantage (1,040 oracle evaluations per run against our 40).

## Answer

| | her published GuacaMol runs | the HER arms in this benchmark |
|---|---|---|
| starting molecules | **10** | 20 (tier 3: 10) |
| GP hyperparameters | **fixed**: mean 0, amplitude 1, noise 1e-4 (thesis Table 4.1) | mean / var / 0.1 x var of a 1,000-molecule hold-out block |
| hold-out block scored | **no** | yes, 1,000 molecules |
| oracle evaluations per run | **30** (10 + 20) | 1,040 (1,000 + 20 + 20) |
| driver | `ehvi_{fexofenadine,amlodipine,perindopril}.py`, branch `guacamol-test` | `ehvi_mc.py`, branch `main` (vendored commit `ca623180`) |

* **We may not claim she had a 26-fold oracle budget advantage.** Her published
  GuacaMol runs used 30 oracle evaluations per run, fewer than our arm's 40.
* The 1,040-versus-40 asymmetry is real **only inside this benchmark**. It exists
  because the HER arms were specified to run her main-branch `ehvi_mc.py` recipe.
  That recipe is a configuration she ran for Fexofenadine, then filed under
  results "not used for thesis" (evidence 4). It is our configuration choice,
  not her protocol.
* **Erratum to PREREGISTRATION.md section 7**, which says removing the hold-out
  block "would stop being her method". That is wrong as a description of her
  published method, which never used it. The `report.py` paragraph "Standing
  asymmetry and modifications" repeats the framing. The pre-registration file is
  not edited, because it is a pre-registration; this file is the erratum. Section
  8.2 of the pre-registration already flagged the discrepancy, and the evidence
  below settles it.
* **Consequence for interpretation**: no arm in the running sweep reproduces her
  published GuacaMol protocol (10 starts, fixed hyperparameters, her count
  fingerprint, 30 evaluations). Tier 3 changes `n_init` to 10 but keeps the
  hold-out hyperparameters. So a result against HER@FP2 is a result against her
  code under her main-branch recipe, not against her published method. Whether
  the hold-out recipe makes her arm stronger or weaker than her published fixed
  hyperparameters is not known. For example, Fexofenadine's hold-out recipe gives
  amplitudes [0.209, 0.090, 0.0052] with noise at 10% of each; the published
  setting is amplitude 1 and noise 1e-4. So "a win reads as conservative" holds
  for oracle information only, not necessarily for optimizer strength.

## Evidence

### 1. The thesis (arXiv:2508.14072)

* Section 4.1 (GuacaMol dataset): all benchmarking models and GP-MOBO were
  trained from 10 initial known SMILES, then ran 20 BO iterations.
* Section 4.4: the GuacaMol GP hyperparameters are "as standardized from the
  Toy MPO setup", i.e. Table 4.1: mean 0.0, noise 1e-4, amplitude 1.0 for every
  objective. MC-EHVI keeps N = 1000.
* Section 4.5: 10 randomly selected initial SMILES for both the Toy MPO and the
  GuacaMol tasks.
* The captions of Figures 5.4, 5.6 and 5.8 (Fexofenadine, Amlodipine,
  Perindopril) give three independent runs, each with 10 random initial SMILES.
* Section 6 lists tuning the GP hyperparameters by negative log marginal
  likelihood as future work, so the reported runs did not tune them.
* Table 8.1 ("Initial Training Set") lists 10 molecules, and they are Toy MPO
  (DockSTRING) molecules. Table 8.2, for the single-objective setup, also has 10
  rows. Both are consistent with the 10-start protocol.
* Inconsistency noted: section 4.1 says about 9,980 query SMILES, but 10,000
  minus 10 starts leaves 9,990. The figure 9,980 matches the 20-start
  main-branch driver.

### 2. Her GuacaMol drivers (branch `guacamol-test`)

`ehvi_fexofenadine.py`, `ehvi_amlodipine.py` and `ehvi_perindopril.py` each have
exactly one committed version, `8e4f5b3` (2024-08-29 09:54). Every one uses
`training_smiles = ALL_SMILES[:10]` and fixed `gp_means = 0`,
`gp_amplitudes = 1`, `gp_noises = 1e-4`, and none scores a hold-out block. The
hold-out script, `N-D Guacamol MPO/ehvi_gp_hyperparameters_3_runs.py`
(`ALL_SMILES[:20]` plus the 1,000-molecule hold-out), arrived three minutes later
in commit `89b6141`, titled "Add potential 3D and 4D scripts from Guacamol MPO to
run & results".

### 3. Her logged GP-MOBO runs on GuacaMol

The table below covers every logged run, across all nine branches and both
locally available pull-request heads. For each, `Dataset size=(N, K)` at BO
iteration 0 gives the starting molecules. Her evaluators print
`f1 shape: (n,)` on every oracle call, so the logged evaluation sizes are an
oracle-call ledger.

| log (branch `guacamol-test` unless noted) | script | runs | starts | hyperparameters | evaluations logged | per-run oracle evaluations | thesis? |
|---|---|---|---|---|---|---|---|
| `guacamol-MPO-results/terminal-output/ehvi_amlodipine_terminal.txt` | `ehvi_amlodipine.py` | 3 | **10** | fixed 0 / 1 / 1e-4 | 10 x3, 1 x120 | **30** | yes (evidence 5) |
| `guacamol-MPO-results/terminal-output/ehvi_perindopril_terminal_final.txt` | `ehvi_perindopril.py` | 3 | **10** | fixed 0 / 1 / 1e-4 | 10 x3, 1 x120 | **30** | yes (evidence 5) |
| `N-D Guacamol MPO/ehvi_fexofenadine.txt` | `ehvi_gp_hyperparameters_3_runs.py` | 3 | 20 | hold-out: means [0.4854 0.8390 0.1874] | **1000 x3**, 20 x3, 1 x60 | 1,040 | **no**: its results CSV was dumped as "not used for thesis" (evidence 4) |
| `compare-results/guacamol_result_trail1.txt` (`results-dump`) | `ehvi_guacamol.py` (early trial, 2024-08-08) | 1 | 20 | hand-set: means [0 0 1], amplitudes [1 0.5 1], noises [1 1e-4 0.1] | 100 (her utils' import-time self-test), 20, 1 x20 | 40 (plus the 100-molecule self-test) | no (dumped) |
| `compare-results/ehvi_guacamol_results_lastrun.txt`, `ehvi_mpo_400samples.txt`, `results/first_ehvi_guacamol_result.txt` | (log tails only) | 1 each | 20 | not in the tail | 1 x20 | unknown | no (dumped / trial) |

In the 10-start logs the count of 120 single-molecule calls is 3 experiments x
(20 BO picks + 20 picks of her random-sampling baseline). The random-sampling
loop is a separate baseline that reuses the start set; it is not part of a BO
run's budget.

There is no terminal log for the thesis Fexofenadine run. Its results file is
evidence 4.

Not GuacaMol: `results/results.txt` on `main` (20 starts, K = 3) is a DockSTRING
trial. None of its 20 chosen SMILES is in the GuacaMol library, and it was
committed in "results folder for trial runs". The single-objective EI logs
(`N-D Guacamol MPO/ei_*.txt`) belong to a different method.

### 4. Her own labelling, in commit messages

* `97387a3` / `8bb6424` / `f0daba3` (2024-08-29 09:50-09:52) added
  `guacamol-MPO-results/`, the GuacaMol EHVI results folder, holding
  `ehvi_bo_{fexofenadine,amlodipine,perindopril}_results.csv`. Each name is
  exactly the file the 10-start driver writes.
* `21944ce` (2024-08-29 10:37), "dumping results not used for thesis", added
  `compare-results/`. That includes `ehvi_bo_fexofenadine_results_2.csv`, whose
  60 rows are exactly the 60 chosen-molecule values of the 20-start hold-out log
  (`N-D Guacamol MPO/ehvi_fexofenadine.txt`, 60/60 found). So the only logged
  hold-out GuacaMol EHVI run was filed by her as not used for the thesis.
* The hold-out script writes `bo_experiments_3_runs_results.csv`, which is not in
  the results folder.

### 5. Cross-checks

* `guacamol-MPO-results/ehvi_bo_amlodipine_results.csv` and
  `..._perindopril_results.csv` are identical, 60 of 60 rows in order, to the
  chosen-molecule values in the 10-start terminal logs.
* `guacamol-MPO-results/ehvi_bo_fexofenadine_results.csv` (blob `765480bb`) shares
  0 of 60 ordered rows with the 20-start hold-out log. Only 5 of its 60 values
  appear anywhere in that log, which is the overlap expected from good molecules
  both runs found. It is a different run, written under the 10-start driver's
  file name.
* Repository state: the local refs of `external/GP-MOBO` equal GitHub's for all
  nine branches (`git ls-remote https://github.com/anabelyong/GP-MOBO.git`,
  2026-10-22). Pull-request heads `4d393df` and `58de8ec` are present locally and
  contain nothing beyond the above.

## How the checks were run (read-only)

```bash
git -C external/GP-MOBO for-each-ref refs/remotes
git ls-remote https://github.com/anabelyong/GP-MOBO.git
git -C external/GP-MOBO grep -c -E "Dataset size=|GP Means|f1 shape" <ref> -- "*.txt"
git -C external/GP-MOBO log --all --format='%h %ad %s' -- <file>
git -C external/GP-MOBO show <commit>:<file>
```

The per-log parsing (iteration-0 dataset sizes, `GP Means/Amplitudes/Noises`,
`f1 shape` counts, row matching between CSVs and logs) was a short Python
script over `git show` output. The thesis text is the arXiv HTML rendering,
<https://arxiv.org/html/2508.14072>.
