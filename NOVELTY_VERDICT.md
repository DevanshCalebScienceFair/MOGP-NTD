# Novelty verdict — prior-art sweep, 2026-08-30

Produced by 11 agents: 5 parallel prior-art sweeps, each adversarially fact-checked, then a
synthesis. Verbatim quotes were verified against primary sources by two independent checkers.

**Bottom line: fixing `diag_embed` is a bug fix, not a contribution.** Passing a correlated
joint posterior to qNEHVI is documented, intended BoTorch usage. Do not claim novelty for it.
The science underneath got better anyway — see sections 1 and 4.

---

# NOVELTY VERDICT — MOGP-NTD

## 1. THE CENTRAL QUESTION

**(a). Plainly and without qualification: passing a correlated joint posterior to qNEHVI is the documented, intended usage of BoTorch. The team broke it with `diag_embed`. Fixing it is a one-line bug fix, not a contribution.**

What was verified, verbatim, by two independent checkers working from different sources (arXiv/ar5iv and the NeurIPS proceedings PDF):

- **qNEHVI paper, Sec. 5.1** (arXiv:2105.08195): *"Note that this 'full-MC' variant of NEHVI does not require objectives to be modeled independently, and supports multi-task covariance functions across correlated objectives."*
- **Same paper, appendix**, costing the exact operation `diag_embed` skips: *"Sampling is more costly when using a multi-task GP model, as it requires a root decomposition of the Mn x Mn posterior covariance across data points and tasks."*
- **qEHVI paper** (arXiv:2006.05078): *"These results hold for any covariance function satisfying the regularity conditions, including such ones that model correlation between outcomes. In particular, our results do not require the outputs to be modeled by independent GPs."*
- **BoTorch CHANGELOG v0.6.1 (2022-02-28):** "Modify NEHVI to support MTGPs (#1037)". **v0.5.0 (2021-06-29):** "KroneckerMultiTaskGP model for efficient multi-task modeling for **block-design settings (all tasks observed at all inputs)**". MOGP-NTD is exactly a block design.
- **BoTorch source** (`botorch/posteriors/gpytorch.py`): `rsample` routes through `_reshape_base_samples_non_interleaved` conditioned on the posterior being a `MultitaskMultivariateNormal`. The full covariance root is what supplies the correlation. A diagonal MMVN makes cross-task and cross-molecule correlation mathematically absent before a single base sample is drawn.
- **BoTorch `ModelListGPyTorchModel` docs**, describing the team's bug as a known library caveat: *"If any model returns a MultitaskMultivariateNormal posterior, then that will be split into individual MVNs per task, with inter-task covariance ignored."* The `diag_embed` line is a hand-rolled reimplementation of a documented failure mode.
- **Official BoTorch tutorial `composite_mtbo`**: `KroneckerMultiTaskGP` (ICM kernel, block design) into MC sampling of the joint posterior into a `GenericMCObjective`. End to end, this is the pipeline the team thinks they need to invent.

The correct write-up sentence is: *"our custom BoTorch `Model` wrapper silently discarded the ICM cross-task covariance; returning the model's own `MultitaskMultivariateNormal` restores documented library behaviour."* Anything stronger will be corrected by the first GP-literate reader.

**One thing you should add that neither survey emphasized.** `diag_embed(var_d.reshape(*batch, q*k))` is diagonal over `q*k`, so it destroys **cross-molecule covariance as well as cross-task**. For batch qNEHVI with q=5, the cross-candidate block is the mechanism that suppresses redundant batch members: under the correct joint posterior a near-duplicate's MC samples are highly correlated with an already-selected point's, so its incremental hypervolume contribution collapses; under a diagonal posterior the samples are independent, so on some draws the duplicate looks better than the point it duplicates and gets selected anyway. This effect is independent of the ICM and would have degraded the independent-GP arm too, which means **the n=1 ablation compared two arms that were both broken in the same way**. This is my inference from the code line plus the qNEHVI formulation, not something the survey verified. Test it cheaply: mean intra-batch Tanimoto similarity, diagonal vs corrected. Expect the corrected posterior to select more diverse batches. This is probably the largest measurable consequence of the bug.

**The single sliver of oxygen, stated honestly:** the qNEHVI authors never ran MTGP + qNEHVI themselves. Verified verbatim from their Sec. 6: *"We model each outcome with an independent GP with a Matérn 5/2 ARD kernel..."*. That leaves an **empirical** hole, not a methodological one, and even the empirical hole is partly filled by Shah & Ghahramani (2016) and Liu et al. (2022). See §2.

---

## 2. WHAT IS ACTUALLY NOVEL, RANKED BY HOSTILE-REVIEWER SURVIVABILITY

### Tier A — survives a hostile reviewer

**A1. The empirical ablation itself: correctly-wired ICM vs independent GPs, under *batch* qNEHVI, on a *fixed discrete molecular library*, for *homologous-target* selectivity, multi-seed.**

- Closest prior: **Fromer, Graff & Coley (2024)** — same application, same discrete-library MOBO, same homolog-selectivity framing (DRD3 over DRD2, JAK2 over LCK), but explicitly independent single-task surrogates. Their Table 1 caption states the choice verbatim: *"N(mu, sigma) implies that the covariance matrix is treated as diagonal with entries sigma^2_i, i.e., uncertainty is uncorrelated across objective functions."* They also name multi-task surrogates as the road not taken.
- Also close: **Shah & Ghahramani (2016)** ran exactly this ablation (CEIPV vs IEIPV over an ICM prior) but analytically, sequentially (q=1), on 2 objectives, in continuous space. Their own Future Work says batch was unsolved: *"Next, we could work on how to select a batch of points where we can evaluate next in parallel, in a multi-objective setting."*
- Also close: **LaMBO (Stanton et al., ICML 2022)** already runs ICM into NEHVI on molecules including a docking objective, but in a generative latent space with learned features, with objectives docking + synthetic accessibility (not two homologs), and reports no isolated correlated-vs-independent ablation.
- **What remains new:** the ablation in the batch / noisy / discrete-library / biologically-grounded-correlation regime that neither 2016 nor 2021 nor 2022 ran. This is a **benchmarking** contribution. A reviewer can call it incremental. A reviewer cannot call it done.

**A2. The defect itself, reported as a quantified methods-hygiene finding.**

- Closest prior: BoTorch's own `ModelListGP` caveat and issue #1036 (opened 2021-12-14, `AttributeError: 'TransformedPosterior' object has no attribute 'mvn'`, closed by PR #1037).
- **What remains new:** nothing conceptual, but nobody has published *what it costs you* to silently factorize an ICM posterior in a real MOBO campaign. "Here is a plausible custom wrapper, here is the hypervolume and batch-diversity penalty it imposes, here is the one-line fix" is a legitimate cautionary result at science-fair / workshop level. It is not a venue paper.

**A3. Mixed exact-and-uncertain objectives inside a coregionalized surrogate.**

- Neither survey found any literature on this case. The team's ICM is nominally 5x5 but is really 2x2 over PfDHFR/hDHFR padded with three deterministic, zero-variance columns. Fitting an ICM over exactly-known outputs plausibly **corrupts the learned coregionalization matrix**, which connects directly to Hvarfner et al. (2026)'s correlation-attenuation mechanism.
- Closest prior: **Loka et al. (2023)** says don't model cheap objectives with a GP at all; **Hvarfner et al. (2026)** supplies the attenuation mechanism.
- **What remains new:** the specific measurement — does deterministic padding change the recovered PfDHFR/hDHFR task correlation? Cheap to run, genuinely unaddressed, small. Report it as a diagnostic, not a method.

### Tier B — legitimate framing, not a contribution

**B1. Selectivity-contrast variance mis-estimation.** For w = [1, -1], Var = Var_h + Var_p − 2Cov is exactly what a diagonal posterior destroys. This is a corollary of **Astudillo & Frazier (2019)** and one line of linear algebra, and BoTorch ships `LinearMCObjective` for it. Use it as motivation. Do not claim it.

**B2. The antifolate application.** No BO/AL paper exists for PfDHFR/hDHFR selectivity. But the *framing* is taken twice over: Fromer et al. (2024) for homolog selectivity as discrete-library MOBO, and **Yoshizawa et al. (2022)**, whose title is literally "Selective Inhibitor Design for Kinase Homologs Using Multiobjective Monte Carlo Tree Search." New proteins, old framing. Fine for a science fair. Not a novelty claim.

---

## 3. WHAT IS DEFINITELY NOT NOVEL

Do not spend a single day on any of these.

| Idea | Killed by |
|---|---|
| "We propose a correlation-aware hypervolume acquisition over an ICM multi-task GP, motivated by correlated drug-discovery objectives." | **Shah & Ghahramani, ICML 2016.** Verbatim: *"The key contribution of this work is to include the modelling of correlations amongst objective functions using multi-output Gaussian process priors."* CEIPV vs IEIPV, ICM parameterized as K = AA^T, drug-discovery motivation, correlation sweep at rho = −0.5, 0, +0.25, +0.75. Ten years old. |
| "Existing acquisitions assume independent objectives; ours consumes the multi-task posterior covariance." | **cPoI (Yang, Chen, Affenzeller, Werth, GECCO '23)** and **Liu, Qu, Liu & Lyu (Aerospace Sci. Tech. 2022, CMOGP + cPFES)**. Both published, both report gains. |
| "MC sampling from a joint posterior is how you handle correlated objectives in hypervolume acquisitions." | **qEHVI (2020) and qNEHVI (2021)** state this as design intent. **Rahat et al. (2022)**: *"there are no closed-form expressions for a problem where the predictive densities are dependent... Monte Carlo approximation is used instead in such cases."* |
| "Model only the expensive objectives; inject the exactly-known ADMET values." | **Loka et al. (2023)**, verbatim: *"Instead of modeling the cheap function with a GP, we directly integrate it in the hypervolume-based acquisition functions."* Named acquisitions CHVEI/CHVPOI. Generalized in **Loka et al. (2024)**. And BoTorch shipped it: `ModelList(SingleTaskGP, GenericDeterministicModel)` plus CHANGELOG "Modify qNEHVI to support deterministic models (#1026)", **BoTorch 0.6.0, 8 Dec 2021**. The umbrella name is **grey-box BO** (Astudillo & Frazier, WSC 2021). |
| "Selectivity = hDHFR − PfDHFR as a composite objective over a correlated joint posterior." | **Astudillo & Frazier, ICML 2019** (f = g(h), h a multi-output GP, g cheap and known). Plus BoTorch's official `composite_mtbo` tutorial and `LinearMCObjective`. One documented API call. |
| "ICM + Tanimoto kernel on Morgan fingerprints is a novel molecular surrogate." | **GAUCHE (NeurIPS 2023)** ships the exact recipe as a tutorial: `TanimotoKernel() * IndexKernel(num_tasks, rank=1)`. Copy-pasteable. |
| "ICM into a hypervolume acquisition on molecules with a docking objective." | **LaMBO (ICML 2022)**, verbatim: *"...an intrinsic model of coregionalization (ICM) kernel over f_i... The resulting GP outputs a posterior predictive distribution p(f\|D), which is passed as input to the acquisition function. We use the noisy expected hypervolume improvement (NEHVI) acquisition..."* One of its own baselines is named "GA + MTGP + NEHVI". |
| "Selectivity between homologous targets as multi-objective BO over a docking library." | **Fromer, Graff & Coley (2024)**, at 260k and 4M scale, with the Pareto-vs-scalarization question already settled. |
| "qNEHVI is too expensive at M=5, so we built something cheaper." | The qNEHVI paper's own complexity analysis (box count K = O(\|P\|^(⌊M/2⌋+1)), i.e. cubic in front size at M=5) already predicts your 96 s → 1934 s curve. And BoTorch's `alpha` caps the box count at 2/alpha regardless of front size; `get_default_partitioning_alpha` returns 1e-3 at M=5, but the `qNEHVI` constructor itself defaults to `alpha=0.0` (exact partitioning). Your blowup is almost certainly a configuration choice. **ESPI/SPMO (2026)** reports that at M=3 or 5 all methods including EHVI/NEHVI stay under 98 seconds. |
| "We invented a front-size-independent multi-objective acquisition." | **Golovin & Zhang (ICML 2020)** and **Zhang (NeurIPS 2024)**, with matching lower bounds. Also qPMHI (2026), qPOTS (AISTATS 2025), HypE (2011), ER2I (2019), DeepHV (ICLR 2023). |

---

## 4. KNOWN NEGATIVE RESULTS

**Short answer: "coregionalization does not improve MOBO over independent GPs" is NOT an established universal finding, and the surveys leaned too hard on the nulls. But something sharper and more damaging to your working hypothesis *is* established, and it applies to your design specifically.**

**The theorem that predicts your null.** Bonilla, Chai & Williams (NIPS 2008), §2.3, "Noiseless observations and the cancellation of inter-task transfer," verified verbatim: for a separable/ICM multi-task GP with observations at the same locations for all tasks, *"the predictions for task l depend only on the targets y_·l. In other words, there is a cancellation of transfer."* Geostatistics calls it **autokrigeability**. Their caveat, also verbatim: *"if the observations are noisy, or if there is not a block design, then this result on cancellation of transfer will not hold."*

Your setup is the pathological case: every molecule is docked against both targets (exact co-located block design), and Vina is near-deterministic. So the ICM's predictive **means** are approximately identical to independent GPs by construction. Write "approximately identical / the transfer channel is largely cancelled", not "provably identical" — your GP fits nonzero likelihood noise and Vina is a stochastic search, deterministic only under a fixed seed. The three exact ADMET columns are genuinely noise-free, so the cancellation is *stronger* there.

**This cuts both ways, and this is your whole story.** Under the same block design the cross-task predictive covariance retains K^f exactly, rescaled by the single-task predictive variance. Means and marginals match independent GPs; the off-diagonal block does not vanish. **That off-diagonal block is precisely and only what `diag_embed` destroyed.** So: the ICM could never have helped you through the mean channel, and you deleted the only channel through which it could have helped. That is a clean, correct, mechanistically explained story, and it also tells you the fix will not move predictive accuracy at all. Whatever it moves will be acquisition behaviour.

**The empirical record is genuinely mixed, not negative.** Report it that way or a reviewer will accuse you of cherry-picking:

- **Positive:** Shah & Ghahramani (2016) — CEIPV beats IEIPV, benefit grows with |rho|. Liu et al. (2022) — correlation-aware CMOGP+cPFES beats independence-assuming baselines. cPoI (2023) — reports gains.
- **Negative:** Hvarfner, Daulton, Balandat & Bakshy (2026) — *"the textbook MTGP loses to a single-task Gaussian process (GP) on most base functions, the textbook signature of negative transfer"*, and it *"attenuates the recovered correlation and, once more than one source is present, can even flip its sign."*
- **Lukewarm:** Alvi et al. (2025) — correlated deep-GP surrogate wins on a non-convex synthetic function but only modestly on the real materials case, attributed verbatim to *"the relatively smooth, deterministic nature of our in-silico optimization landscape."* Vina on a fixed library is exactly that kind of landscape.
- **Shah & Ghahramani's own caveat**, which matches your ablation: on one task the independent version won, and they attribute it to *"a correlated multi-task GP may be prone to consider more complicated explanations than necessary."*

**Do not overclaim Hvarfner et al. against yourself.** It is source/target **transfer** learning in **single-objective** BO with qLogNEI. A checker grepped the full text and found no occurrence of EHVI, hypervolume, Pareto, or multi-objective. Two of its three failure conditions do not apply to you: the sign-flip result is conditioned on more than one source task, and the non-overlapping-design dilution does not apply because you co-locate every molecule. Co-location is that paper's own Remedy 3, which you already satisfy. It is a diagnostic to run, not an equally likely explanation of your null.

**Do NOT cite** the sentence *"In extensive unpublished experiments, making a true multi-task acquisition function did not lead to any benefit."* Both checkers independently confirmed it is absent from arXiv:2607.09073 and untraceable. It appears only on an aggregator page.

**Verdict on your n=1:** consistent with theory, uninformative as evidence (n=1, and both arms had a broken batch-diversity mechanism). Report it as a pre-registered expectation, not a discovery. A well-instrumented null with a named mechanism is a stronger deliverable than a weak positive you cannot defend.

---

## 5. THE HONEST RECOMMENDATION

### The reframe that saves your benchmark

**Your completed 10-seed benchmark is not invalidated. It is relabelled, and it becomes more interesting.** Describe it precisely as: *"ICM surrogate with a factorized (diagonal) acquisition posterior."* That is the exact modelling choice Fromer, Graff & Coley (2024) made **deliberately** and stated in print. Your finished run is therefore a faithful instance of the published state of the art for discrete-library selectivity MOBO. Keep it, untouched, as the control arm. Do not re-run it. Do not touch its config.

### Do this, in priority order

1. **Delete the `diag_embed` line.** Return the model's own `MultitaskMultivariateNormal`. Verify with `posterior.mvn.covariance_matrix` that the off-diagonal blocks are nonzero. Hours, not days.
2. **Report the learned ICM task-correlation matrix.** Highest value per unit effort in this entire document. If rho(PfDHFR, hDHFR) is near zero or negative, your biological premise is falsified before you spend a single CPU-hour. This is Hvarfner et al.'s core recommendation and it is nearly free.
3. **Audit standardization order.** Is per-task standardization applied before ICM fitting? That is the named suspect for correlation attenuation. Consider constraining task correlation non-negative, which is defensible for homologous DHFRs.
4. **Measure intra-batch Tanimoto similarity, diagonal vs corrected.** Free, and it tests the mechanism most likely to show a real effect (§1).
5. **Run the timing table with `alpha=1e-3, prune_baseline=True`** before writing one word about acquisition cost. If it flattens, delete the cost narrative entirely.
6. **Take the three exact ADMET objectives out of the GP.** `ModelList(MultiTaskGP, GenericDeterministicModel)`, supported since BoTorch 0.6.0. Cite Loka et al. for the pattern, do not claim it. This simultaneously tests A3.
7. **Add the delta baseline.** Model selectivity = hDHFR − PfDHFR directly as an objective. Burggraaff et al. (2020) showed on a homologous receptor pair that this beats subtracting two independently-modelled predictions. A reviewer will ask why an ICM plus a rewritten posterior is needed when a delta model is one line. If correlated qNEHVI cannot beat it, say so in the abstract.
8. **Run exactly two new 10-seed arms**, not four: (i) corrected joint posterior + ICM, (ii) the delta baseline. At ~290 evals x 15 s that is roughly 12 CPU-hours per arm serial, tractable in parallel on an M4 Pro. Everything else in this list is a diagnostic that runs in minutes.

### Frame the paper as

> "A standard custom BoTorch `Model` wrapper silently factorized our multi-task posterior, discarding both the ICM's learned cross-task covariance and all cross-candidate covariance before qNEHVI's Monte Carlo sampling. We quantify what that costs on a 26,660-molecule antifolate library with two homologous docking targets. Because our design is co-located and near-noiseless, autokrigeability (Bonilla et al., 2008) predicts the ICM cannot help through the predictive mean, so the covariance channel is the only one available; we measure it and report the result."

That is honest, mechanistically grounded, defensible against every citation in §6, and it is a real contribution at the level you are actually operating at.

### Do NOT attempt

- **Do not build a new acquisition function.** Three published precedents (Shah 2016, Liu 2022, Yang 2023). You will lose.
- **Do not switch to HVKG or decoupled MOBO.** It invalidates your benchmark, requires abandoning qNEHVI, and is large machinery for a deadline project.
- **Do not switch to scalarization for speed.** Yong et al. (2025) find EHVI consistently beats fixed-weight scalarized EI on molecular tasks, and Fromer et al. find Pareto beats scalarization across three virtual-screening case studies. Fix the cost with `alpha` instead.
- **Do not claim any of §3.**
- **Do not claim "multi-task structure over 5 objectives."** It is a 2x2 problem padded with three deterministic columns, and a reviewer will notice.
- **Do not chase qAEHVI** (OpenReview verification wall, entirely unverified).
- **Do not quote these without re-reading the source yourself:** Bakshy's closing comment on BoTorch issue #1036 (recovered via a paraphrasing proxy); the qEHVI appendix sentence about constraints (did not come back verbatim); the two ER2I complexity claims (paywalled).
- **Do not cite** "Bayesian Optimization for Molecules Should Be Pareto-Aware." That is the withdrawn v1 title of arXiv:2507.13704, now "A study of EHVI vs fixed scalarization for molecule design." Citing the dead title signals you never opened the paper.

### Paste-ready, for the M4 Pro running MOGP-NTD

```
In the custom BoTorch Model wrapper, replace the
`covar = torch.diag_embed(var_d.reshape(*batch, q*k))` posterior construction
with the model's own MultitaskMultivariateNormal so the ICM cross-task and
cross-molecule covariance reach qNEHVI. Then, without touching the existing
10-seed benchmark config:
1. print the learned ICM task-correlation matrix (PfDHFR vs hDHFR) and assert
   posterior.mvn.covariance_matrix has non-zero off-diagonal blocks;
2. check whether per-task standardization runs before or after ICM fitting;
3. log mean intra-batch Tanimoto similarity for q=5 batches under both the old
   diagonal posterior and the corrected one;
4. re-time one iteration with alpha=1e-3 and prune_baseline=True vs the current
   alpha=0.0;
5. move the 3 exactly-known ADMET objectives out of the GP into
   ModelList(MultiTaskGP, GenericDeterministicModel).
Report all five as a table. Do not re-run or modify the completed benchmark.
```

---

## 6. CITATION TABLE (everything that survived verification)

### Kills a methods novelty claim

| Title | Authors | Year | Venue | Identifier |
|---|---|---|---|---|
| Pareto Frontier Learning with Expensive Correlated Objectives (CEIPV) | Shah, Ghahramani | 2016 | ICML 33 | PMLR 48:1919–1927 |
| Parallel Bayesian Optimization of Multiple Noisy Objectives with EHVI (qNEHVI) | Daulton, Balandat, Bakshy | 2021 | NeurIPS 34 | arXiv:2105.08195 |
| Differentiable EHVI for Parallel Multi-Objective BO (qEHVI) | Daulton, Balandat, Bakshy | 2020 | NeurIPS 33 | arXiv:2006.05078 |
| Bayesian Optimization with High-Dimensional Outputs | Maddox, Balandat, Wilson, Bakshy | 2021 | NeurIPS 34 | arXiv:2106.12997 |
| Accelerating BO for Biological Sequence Design with Denoising Autoencoders (LaMBO) | Stanton, Maddox, Gruver, Maffettone, Delaney, Greenside, Wilson | 2022 | ICML 39 | arXiv:2203.12742; PMLR v162 |
| A New Acquisition Function for MOBO: Correlated Probability of Improvement (cPoI) | Yang, Chen, Affenzeller, Werth | 2023 | GECCO '23 Companion | DOI 10.1145/3583133.3596374 |
| Correlation-concerned Bayesian optimization for multi-objective airfoil design | Liu, Qu, Liu, Lyu | 2022 | Aerospace Sci. & Tech. 129 | DOI 10.1016/j.ast.2022.107867 |
| Bayesian Optimization of Composite Functions | Astudillo, Frazier | 2019 | ICML 36 | arXiv:1906.01537; PMLR 97:354–363 |
| Bi-objective BO of engineering problems with cheap and expensive cost functions (CHVEI/CHVPOI) | Loka, Couckuyt, Garbuglia, Spina, Van Nieuwenhuyse, Dhaene | 2023 | Engineering with Computers 39:1923–1933 | DOI 10.1007/s00366-021-01573-7 |
| Cheap-expensive MOBO for permanent magnet synchronous motor design | Loka, Ibrahim, Couckuyt, Van Nieuwenhuyse, Dhaene | 2024 | Engineering with Computers 40(4):2143–2159 | DOI 10.1007/s00366-023-01900-0 |

### Kills an application novelty claim

| Title | Authors | Year | Venue | Identifier |
|---|---|---|---|---|
| Pareto Optimization to Accelerate Multi-Objective Virtual Screening | Fromer, Graff, Coley | 2023/2024 | arXiv; Digital Discovery 3:467–481 | arXiv:2310.10598 |
| Selective Inhibitor Design for Kinase Homologs Using Multiobjective MCTS | Yoshizawa, Ishida, Sato, Ohta, Honma, Terayama | 2022 | JCIM 62(22):5351–5360 | DOI 10.1021/acs.jcim.2c00787 |
| BO in the Latent Space of a VAE for Selective FLT3 Inhibitors | Chandra, Horne, Vendruscolo | 2023 | J. Chem. Theory Comput. | DOI 10.1021/acs.jctc.3c01224 |
| Quantitative prediction of selectivity between the A1 and A2A adenosine receptors | Burggraaff, van Vlijmen, IJzerman, van Westen | 2020 | J. Cheminform. 12:33 | DOI 10.1186/s13321-020-00438-3 |
| GAUCHE: A Library for Gaussian Processes in Chemistry | Griffiths, Klarner, Moss, et al. | 2023 | NeurIPS D&B | arXiv:2212.04450 |
| MOBO with Independent Tanimoto Kernel GPs (GP-MOBO) *(MSc thesis, weight accordingly)* | Yong | 2025 | arXiv | arXiv:2508.14072 |
| A study of EHVI vs fixed scalarization for molecule design *(note: title changed from v1)* | Yong, Tripp, Hosseini-Gerami, Paige | 2025 | NeurIPS AI4Science Workshop | arXiv:2507.13704 |
| DOCKSTRING | García-Ortegón, Simm, Tripp, Hernández-Lobato, Bender, Bacallado | 2021/2022 | arXiv; JCIM | arXiv:2110.15486 |

### Explains or contextualizes the null result

| Title | Authors | Year | Venue | Identifier |
|---|---|---|---|---|
| Multi-task Gaussian Process Prediction (autokrigeability, §2.3) | Bonilla, Chai, Williams | 2008 | NIPS 20 | S2 10d10df314c1b58f5c83629e73a35185876cd4e2 |
| Pitfalls and Remedies for Multi-Task Bayesian Optimization | Hvarfner, Daulton, Balandat, Bakshy | 2026 | arXiv preprint | arXiv:2607.09073 |
| Deep GP-based Cost-Aware Batch BO for Materials Design | Alvi, Vela, Attari, Janssen, Perez, Allaire, Arroyave | 2025 | arXiv; npj Comput. Mater. 2026 | arXiv:2509.14408 |
| Deep Gaussian process for multi-objective Bayesian optimization | Hebbal, Balesdent, Brevault, Melab, Talbi | 2023 | Optimization and Engineering 24:1809–1848 | DOI 10.1007/s11081-022-09753-0 |
| SMOG: Scalable Meta-Learning for Multi-Objective BO | Papenmeier, Tighineanu | 2026 | arXiv preprint | arXiv:2601.22131 |

### Cost, scaling, and cheap-acquisition alternatives

| Title | Authors | Year | Venue | Identifier |
|---|---|---|---|---|
| Thinking inside the box: A tutorial on grey-box Bayesian optimization | Astudillo, Frazier | 2021/2022 | Winter Simulation Conf. 2021 | arXiv:2201.00272 |
| Approximation of Box Decomposition Algorithm for Fast HV-Based MOO | Watanabe | 2025 | arXiv preprint | arXiv:2512.05825 |
| Efficient Approximation of EHVI using Gauss-Hermite Quadrature | Rahat, Chugh, Fieldsend, Allmendinger, Miettinen | 2022 | PPSN XVII | arXiv:2206.07834 |
| Random Hypervolume Scalarizations for Provable MO Black Box Optimization | Golovin, Zhang | 2020 | ICML 2020 | arXiv:2006.04655 |
| Optimal Scalarizations for Sublinear Hypervolume Regret | Zhang | 2024 | NeurIPS 2024 | arXiv:2307.03288 |
| Hypervolume Knowledge Gradient | Daulton, Balandat, Bakshy | 2023 | ICML 2023 | PMLR 202:7167–7204 |
| Generative MOBO with Scalable Batch Evaluations (qPMHI) | Muthyala, Sorourifar, Tan, Peng, Paulson | 2025/2026 | arXiv; Ind. Eng. Chem. Res. 65(1):628–642 | arXiv:2512.17659 |
| qPOTS: Efficient Batch MOBO via Pareto Optimal Thompson Sampling | Renganathan, Carlson | 2025 | AISTATS 2025 | arXiv:2310.15788 |
| Do We Really Need to Approach the Entire Pareto Front in Many-Objective BO? (SPMO/ESPI) | Jiang, Huang, Li | 2026 | arXiv preprint | arXiv:2604.09417 |
| HypE: An Algorithm for Fast Hypervolume-Based Many-Objective Optimization | Bader, Zitzler | 2011 | Evolutionary Computation 19(1):45–76 | DOI 10.1162/EVCO_a_00009 |
| The Expected R2-Indicator Improvement for MOBO | Deutz, Emmerich, Yang | 2019 | EMO 2019, LNCS 11411 | DOI 10.1007/978-3-030-12598-1_29 |
| Multi-objective optimization via equivariant deep hypervolume approximation (DeepHV) | Boelrijk, Ensing, Forré | 2023 | ICLR 2023 | arXiv:2210.02177 |

### Software artifacts (engineering context, not literature — do not put these in a related-work section)

| Item | Detail | Identifier |
|---|---|---|
| BoTorch CHANGELOG v0.5.0 (2021-06-29) | "KroneckerMultiTaskGP model for efficient multi-task modeling for block-design settings (all tasks observed at all inputs) (#637)" | github.com/meta-pytorch/botorch/blob/main/CHANGELOG.md |
| BoTorch CHANGELOG v0.6.0 (2021-12-08) | "Modify qNEHVI to support deterministic models (#1026)" | ibid. |
| BoTorch CHANGELOG v0.6.1 (2022-02-28) | "Modify NEHVI to support MTGPs (#1037)"; "More efficient sampling from KroneckerMultiTaskGP (#460)" | ibid. |
| BoTorch CHANGELOG v0.10.0 (2024-02-26) | "Add support for multitask models to ModelListGP (#2154)" | ibid. |
| BoTorch issue #1036 | "[question] Can qNEHVI be used with KroneckerMultiTaskGP?", alinaselega, 2021-12-14, CLOSED, fixed by PR #1037 | github.com/meta-pytorch/botorch/issues/1036 |
| BoTorch tutorial `composite_mtbo` | KroneckerMultiTaskGP (ICM) + IIDNormalSampler + GenericMCObjective + qLogEI | archive.botorch.org/tutorials/composite_mtbo |
| BoTorch `ModelList` / `GenericDeterministicModel` docs | Includes the "inter-task covariance ignored" caveat | botorch.readthedocs.io/en/latest/models.html |
| `get_default_partitioning_alpha` | Returns 0.0 for M≤4, 1e-3 for M=5, 1e-2 for M≥6; qNEHVI constructor itself defaults to `alpha=0.0` | archive.botorch.org/v/latest/api/_modules/botorch/acquisition/multi_objective/utils.html |

### Known-unverified — do not cite from this document

- **qAEHVI** ("Approximate Expected Hypervolume Improvement for Parallel Expensive Multi-objective Optimization", OpenReview `qIBR0GdwSY`): authors, venue, year, complexity all unverified behind a browser-verification wall.
- **ER2I** specific claims (linear growth in weight combinations; utopian vs nadir reference point): Springer chapter behind an IdP redirect, never read firsthand.
- **STAGE-BO** (arXiv:2604.15959): "ICML 2026" venue unverified.
- **"In extensive unpublished experiments, making a true multi-task acquisition function did not lead to any benefit"**: untraceable, absent from arXiv:2607.09073. Do not cite under any circumstances.