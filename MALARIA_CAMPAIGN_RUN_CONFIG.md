# Malaria campaign: run configuration, reconstructed from evidence

**Why this file exists.** `campaign_results/` carries no `run_config.json`; the
per-run config file was only added later, with the NRP runner that produced
`rung1_chagas_results/`. Everything below is RECONSTRUCTED from artifacts on
disk and from git, and every line names the evidence that fixes it. Nothing here
is recalled from memory or inferred from what the code does today.

**Bottom line for the methods section.** The malaria campaign's MOGP arm used a
**rank-1 intrinsic coregionalization model (ICM)** over the two docking
objectives, behind a **diagonal posterior**, with qLogNEHVI at **`alpha = 0.0`
(exact box partitioning)**. The joint-posterior path and the `alpha` flag did not
exist when it ran.

---

## 1. When it ran, and on what code

| fact | value | evidence |
|---|---|---|
| run dates | 2026-08-23, 10:35 to 23:40 PDT (seed 0) | `timestamp` column, `campaign_results/seed_0/*/seed_0/iteration_timings.csv` |
| code commit | `c7d31bb` "Add GP-MOBO to benchmark runner, per-iteration timing, and acquisition-pool cap", dated 2026-08-23 | it is the only commit that both introduces `iteration_timings.csv` (which the campaign wrote) and predates the run |
| runner | `run_benchmark_seeds.py` | writes the `{seed}/{method}/{seed}/` directory layout the campaign has, and the `method` column in the timings file (`MOGP`, `GP-MOBO`) |

**Two flags the campaign could not have used, because they did not exist yet:**

| flag | landed | commit | days after the campaign |
|---|---|---|---|
| `--posterior {diag,joint}` | 2026-08-30 | `3e4c496` | 7 |
| `--acquisition-alpha` | 2026-08-31 | `27d96e6` | 8 |

This is the decisive point. It is not an assumption about defaults; the code
paths were not in the tree.

---

## 2. Surrogate

The MOGP arm's surrogate is `mogp_coregionalized.MOGPCoregionalized`, selected
by `loop.DEFAULT_MODEL = "coregionalized"` (`loop.py:108` at `c7d31bb`).
`run_benchmark_seeds._build_runner` constructs `BOLoop(...)` without passing
`model=` or `coregionalization_rank=`, so both defaults apply.

| component | class | setting |
|---|---|---|
| data kernel | `kernel.TanimotoKernel` | scaled |
| task covariance | `gpytorch.kernels.IndexKernel` | `num_tasks=K`, **`rank=1`** |
| combined | `gpytorch.kernels.MultitaskKernel` | Kronecker form |
| mean | `MultitaskMean(ConstantMean(), num_tasks=K)` | |
| likelihood | `MultitaskGaussianLikelihood(num_tasks=K)` | per-task noise |
| training | Adam, exact marginal log likelihood | **200 iterations, lr 0.1** |

**K = 2, not 5.** This is a grey-box model: only the two docking objectives are
modelled. `train_mogp_coregionalized` trains on the observed docking target
columns alone (`mogp_coregionalized.py:155-166`) and raises if there are none.
The three ADMET objectives are known exactly for the whole library and enter the
acquisition through `CompositeKnownADMETObjective`, never through the GP.

**This is the Kronecker ICM, NOT the Hadamard one.** `mogp_hadamard.py` is not
in the tree at `c7d31bb`; it was added for the asymmetric-label work. The
Kronecker form cannot represent a molecule with only one docking label, which is
consistent with the campaign's complete block design (every molecule was docked
against both targets or neither).

---

## 3. Posterior path: DIAGONAL

`acquisition.DockingPosteriorModel.posterior()` at `c7d31bb` ends at line 365
with

```python
covar = torch.diag_embed(var_d.reshape(*batch, q * k))
```

and there is no other branch. **The ICM's learned PfDHFR-hDHFR task covariance,
and all cross-molecule covariance, were discarded before qLogNEHVI's Monte Carlo
sampling.** The coregionalization therefore affected the campaign only through
predictive means and marginal variances, never through the sampled joint.

State this plainly in the paper. It is the finding that the later
`--posterior joint` work exists to act on, and the measured effect of fixing it
was small and negative on hypervolume (pure posterior effect **-0.0008**), so
disclosing it costs nothing and removes a reviewer's question.

---

## 4. Acquisition

| setting | value | evidence |
|---|---|---|
| acquisition function | `qLogNoisyExpectedHypervolumeImprovement` | `acquisition.py:522` at `c7d31bb` |
| **box-decomposition `alpha`** | **0.0 (exact partitioning)** | `acquisition.py` at `c7d31bb` contains no occurrence of `alpha`, so BoTorch's default of 0.0 applies |
| MC sampler | `SobolQMCNormalSampler`, **128 samples** | `N_MC_SAMPLES = 128`, `acquisition.py:84` |
| `prune_baseline` | `False` | `acquisition.py:530` |
| `cache_root` | `False` | `acquisition.py:531` |
| reference point | `fixed_reference_point(num_objectives)` | `acquisition.py:519`; fixed, not adaptive |
| objective signs | `[-1, +1, -1, +1, +1]` | `DEFAULT_OBJECTIVE_SIGNS`, `acquisition.py:81` |
| dtype | `torch.double` | `_DTYPE`, `acquisition.py:96` |
| batch diversity | greedy max-Tanimoto filter, **threshold 0.7** | `select_batch` default, `acquisition.py:556`; the runner does not override |

**`alpha = 0.0` is independently corroborated by the timings.** For seed 0's
MOGP arm, acquisition seconds grow from 95.6 s at iteration 1 to 1,933.8 s at
iteration 50 while the Pareto front grows from 26 to 172 members. That
superlinear growth in front size is the signature of exact box partitioning. At
`alpha = 1e-3` the cost is bounded and does not track the front this way
(measured 8.4x at B=80, `PROBE_RESULTS.md`).

---

## 5. Budget and library, all three arms

| setting | value | evidence |
|---|---|---|
| seeds | 0 through 9 | ten `campaign_results/seed_*/` directories |
| `n_init` | 40 | `history.csv` iteration 1 has `n_evaluated = 45 = 40 + 5` |
| `batch_size` | 5 | consecutive `n_evaluated` differ by 5 |
| `n_iterations` | 50 | `history.csv` is 51 rows including the header |
| molecules evaluated | 290 per arm per seed | `history.csv` final row |
| `acquisition_pool_size` | **2000** | `iteration_timings.csv` columns `acquisition_pool_size = 2000` and `n_candidates_scored = 2000`, every iteration, in BOTH BO arms |
| densification | OFF | `densify` default `False`; the runner does not enable it |
| library | 29,678 curated / **26,660 searched** | heavy-atom floor removes 68, quality screen removes 2,950 |

**Candidate exposure is equal across the two BO arms.** `n_candidates_scored` is
2,000 for MOGP and 2,000 for GP-MOBO at every iteration, which is the auditable
form of the parity described in `CLAUDE.md`. Greedy has no acquisition and
writes no timings file.

**Greedy is not an iterated BO loop.** `GreedyFilterThenDock` with
`n_total = 290` and `batch_size = 5`; its `history.csv` has 58 rows because it
starts at `n_evaluated = 5` with no initial design.

---

## 6. What is NOT recoverable from disk

- The **exact command line**. No log, shell script or manifest for this campaign
  is committed. The settings above are read off the artifacts and the code, not
  off a recorded invocation.
- The **wall-clock total across all 30 cells**. Only per-iteration timings for
  the two BO arms survive; greedy wrote none.
- Whether `--lib-size` was passed. It is moot: `--library-dir data/library`
  loads the cached 29,678-molecule library whole, and the searched count of
  26,660 matches it.

If the paper needs a stated command, write it as a reconstruction, for example:

```
python run_benchmark_seeds.py \
  --seeds 0 1 2 3 4 5 6 7 8 9 \
  --methods mogp gpmobo greedy \
  --library-dir data/library \
  --n-init 40 --batch-size 5 --n-iterations 50 \
  --mogp-iters 200 --acquisition-pool-size 2000
```

and say in the caption that it is reconstructed from the run artifacts, not a
transcribed invocation. Do not present it as a recorded command.

---

## 7. The one-paragraph methods text

> Each campaign arm evaluated 290 molecules from a 26,660-molecule searched
> library under a shared 5-objective problem, from a 40-molecule random initial
> design followed by 50 iterations of batch size 5. The MOGP arm's surrogate was
> a grey-box rank-1 intrinsic coregionalization model over the two docking
> objectives only: a scaled Tanimoto data kernel combined with a learned dense
> 2x2 `IndexKernel` task covariance in Kronecker (`MultitaskKernel`) form, with a
> multitask Gaussian likelihood carrying per-task noise, trained for 200 Adam
> iterations at learning rate 0.1. The three ADMET objectives were known exactly
> for the whole library and entered the acquisition directly rather than through
> the GP. Candidates were scored by qLogNEHVI with 128 Sobol quasi-Monte-Carlo
> samples against a fixed reference point, over a per-iteration uniform
> subsample of 2,000 un-evaluated candidates, and a batch was assembled greedily
> subject to a maximum pairwise Tanimoto similarity of 0.7. **Two
> implementation details of this campaign should be noted: the posterior passed
> to the acquisition function was diagonal, so the coregionalization influenced
> selection only through predictive means and marginal variances; and the
> hypervolume box decomposition inside the acquisition used exact partitioning
> (`alpha = 0`). Both were changed in later work, where the joint posterior was
> measured to change final hypervolume by -0.0008 and `alpha = 1e-3` gave a
> 17.0x wall-clock speedup at equal measured solution quality.**
