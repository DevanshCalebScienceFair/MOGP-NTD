# `spec/` MANIFEST — index, checksums, determinism status, agreement criteria

**216 files, 17,572,329 bytes total** under the repository's `spec/` directory, of which
**188 are fixtures / frozen artifacts totalling 16,396,254 bytes**; the rest are the nine
`SPEC.md` documents, the capture scripts and two reference modules.

Produced by an adversarial verification pass: every capture script was re-run in a fresh
interpreter and byte-compared, every cross-cluster convention was checked against every other
cluster, and five quantitative claims were re-derived independently. Nothing in this file was
copied from a cluster's self-report without being either measured or explicitly labelled as
transcribed prose.

Verify with:

```bash
conda run -n mogp-drug python spec/verify_fixtures.py     # from the repository root
```

`verify_fixtures.py` parses the `CHECKSUMS` block at the end of this document, so the manifest
is the single source of truth: changing a fixture without regenerating the manifest fails, and
changing the manifest without regenerating the fixture fails too.

---

## 0. Repo integrity

```
$ git -C /Users/calebsuh/MOGP-NTD status --porcelain
?? .DS_Store
?? benchmark_5way_densify_seed0_2026-08-19/
?? benchmark_thicc_2026-08-14/
?? benchmark_thicccc_2026-08-16/
?? matrix_thick_2026-08-14/
?? matrix_thick_2026-08-15/
?? mogp_core/
?? spec/
?? test_core_admet.py  test_core_boxdecomp.py  test_core_gp.py  test_core_hv.py
?? test_core_primitives.py  test_core_qnehvi.py  test_core_stats.py

$ git -C /Users/calebsuh/MOGP-NTD stash list      # (empty)
$ git -C /Users/calebsuh/MOGP-NTD diff HEAD --name-status   # (empty)
$ git -C /Users/calebsuh/MOGP-NTD ls-files -m     # (empty)
$ git -C /Users/calebsuh/MOGP-NTD ls-files -d     # (empty)
```

**No tracked file was modified, moved or deleted.** All 15 working-tree entries are `??`
(untracked); `git diff HEAD`, `git ls-files -m` and `git ls-files -d` are all empty, and the
stash is empty. Branch `feat/numpy-bo-rewrite` at `3c6573c`.

One caveat, reported because it is true and not because git shows it:
`data/library/quality_rejected.csv` has an mtime of **today 20:29** and was therefore
rewritten during this work. It is a generated cache excluded by `.gitignore:31`
(`data/library/`), so it is untracked and invisible to `git status`. The `qnehvi` agent
disclosed causing this by calling `data.load_library()` once; its capture script now reads the
three cached files directly and never calls that function. The previous bytes cannot be
recovered, so I cannot verify the content is unchanged — only that no *tracked* file was
touched and that the screen is deterministic. `.pytest_cache/` was also rewritten (also
git-ignored).

---

## 1. Reproducibility: every capture script re-run in a fresh interpreter

Method: `spec/` was copied aside, a scratch mirror of the repo was built from symlinks (so the
captures read the real `data/`, `models/`, `external/` and archived campaign directories but
wrote into a scratch `spec/<cluster>/fixtures/`), the three scripts that then hard-coded
`REPO = "/Users/calebsuh/MOGP-NTD"` had that one literal repointed at the mirror (that literal
has since been replaced by a path computed from `__file__`; see the note after the verdict
below), and every output was byte-compared against the on-disk fixture. **No file in the repo's
`spec/` tree was written to at any point.**

| capture script | ran | byte-identical | detail |
|---|---|---|---|
| `primitives/capture_primitives.py` | yes | **yes — 9/9** | All four `.npz` + `.json` pairs and `INDEX.json` byte-identical. Cluster claims no exclusions; none needed. |
| `gp/capture_gp.py` | yes | **yes — 8/9** | The 7 files in `fixtures/MANIFEST.json` plus the manifest itself byte-identical. Only `stochastic_threshold.json` differed, which the cluster excludes by design; a key-by-key diff shows the difference confined to exactly the 4 `loss_scan` rows above `N·T = 800` (`NT = 802, 820, 840, 1000`) and nothing else — the `is_exact` pattern, the controls at `NT = 420` and `NT = 800`, `predictive_scan` and `probe_rng_source` all reproduced identically. **Not a defect.** |
| `icm/capture_icm.py` | yes | **yes — 35/37** | `10_scaling.json` (wall clock, explicitly declared the one fixture that differs between fresh interpreters) and `MANIFEST.json` (which indexes it). The MANIFEST diff is exactly 2 keys: `fixtures[34].bytes` and `fixtures[34].sha256`, both the `10_scaling` entry. The five fixtures the cluster conservatively flags stochastic (`00`, `07`, `08`, `09`, `11`) were **byte-identical** here. **Not a defect.** |
| `hv/capture_hv.py` | yes | **yes — 14/16** | `manifest.json` itself byte-identical, so `manifest_digest` reproduces. The 2 differing files are `timings.json` and `environment.json` (elapsed seconds), both explicitly excluded from the digest. **Not a defect.** |
| `boxdecomp/capture_boxdecomp.py` | yes | **yes — 3/4** | `arrays.bin` (2,709,800 bytes of raw little-endian float64), `manifest.json` and `measurements.json` all byte-identical. Only `timings.json` differed, which carries its own `warning` field saying not to byte-compare it. **Not a defect.** |
| `boxdecomp/mutation_study.py` | yes | **yes — 1/1** | `mutations.json` byte-identical, sha256 `f99c3060a57a3315a8155e16a03430f4a0a4c0000e929deaf3c4ac545beb2fe7` matching the claim exactly. The full firing pattern reproduced, including `thm21_strict` 14/16 15/16 15/16 16/16 0/16, `step4_nonstrict diverged=3`, `pass2_ideal_minus_inf` 0/16 on all five tests, and `no_empty_filter` firing T4 **only**. |
| `qnehvi/capture_qnehvi.py` | yes | **yes — 42/45 across the cluster** | `qnehvi_capture.json` byte-identical with sha256 `27362fb089e2b94c72060547c9a4983871d6951ed00d0a992627414166e0f5bd` as claimed (1770 s runtime), and all 40 `.npy` fixtures byte-identical. The only differences in the whole cluster are `capture_runtime.txt`, `decomposition_timings.txt` and `probe_runtime.txt` — all wall clock. **Not a defect.** |
| `qnehvi/capture_probes.py` | yes | **yes — 1/1 JSON** | `qnehvi_probes.json` byte-identical, sha256 `d7093ad39853c7508d7da1d7333200275610d79b7b74aa89f282422df4b69c47` as claimed. `probe_runtime.txt` differs (wall clock). |
| `qnehvi/check_decomposition_contract.py` | yes | **yes — 1/1** | `decomposition_contract.json` byte-identical, sha256 `8fc0ab80f3b725790a74b94aad312e1bdfa0bfa4a9f9e52ea221824fef28cc01` as claimed. **And** re-run against `spec/boxdecomp`'s numpy Lacour decomposition instead of botorch's it produced the *same bytes and the same sha256* — see section 5. |
| `stats/capture_stats.py` | yes | **yes — 9/9** | All seven byte-comparable fixtures **plus** `MANIFEST.json` **plus** `environment.json`, which the cluster excludes by design. Better than claimed. |
| `admet/capture_admet.py` | yes | **yes — 53/54** | All 53 byte-stable fixtures (export `.npz`, `_meta.json`, `export_manifest.json`, the 15 `ref/` arrays, the 13 `ad/` arrays, the 18 `synthetic/` arrays) byte-identical. Only `METADATA.json` differed (40,864 → 40,832 bytes), which the cluster declares non-byte-comparable because it records wall-clock timings and its own size. **Not a defect.** |

**Verdict: no script that claims determinism is defective.** Every file that differed between
runs is one its own cluster had already declared non-byte-comparable, and in three cases
(`gp/fixtures/stochastic_threshold.json`, `icm/fixtures/MANIFEST.json`,
`boxdecomp/fixtures/timings.json`) I confirmed the difference is confined to exactly the fields
the declaration predicts. Two clusters did *better* than they claimed: `stats` reproduced all
nine files including the `environment.json` it excludes, and `icm`'s five "stochastic"
torch-fit fixtures (`07`, `08`, `09`, `11`, `00`) were byte-identical here.

**Edited after this pass (2026-10-07); the scripts below have not been re-run since.** The
byte-identical results above describe the scripts as they were then. No pinned fixture was
regenerated or hand-edited; the digests of the changed scripts and documents in `CHECKSUMS` and in
section 6 were re-pinned mechanically.

* `gp/capture_gp.py`, `icm/capture_icm.py` and `admet/capture_admet.py` no longer hard-code
  `REPO`; each computes it from `__file__`. The absolute paths in run commands, a directory
  label and a file reference (twelve scripts, four `SPEC.md` files, this file's own `Verify with`
  command and `bounds/INDEPENDENT_REVERIFICATION.md`) are now relative to the repository root;
  the `git -C /Users/...` transcripts in section 0 are verbatim records and keep their paths.
  Apart from the hardening below, an AST comparison with `HEAD` shows only docstring text and
  those three assignments changed in these scripts.
* `hv/capture_hv.py` S9 now counts, from the raw CSV cells, the rows its three `np.isfinite`
  filters must keep (a blank failed dock is the only allowed shortfall) and raises if they keep
  fewer. Run in memory against the archive it reproduced `output_column_contract.json` exactly
  except for three added keys under `verification`: `rows_total` 90, `rows_blank_failed_dock` 1,
  `rows_expected_finite` 89. On its next real run that fixture, the digest recorded in
  `hv/fixtures/manifest.json` and `hv/capture_log.txt` will therefore differ from the pinned
  copies.
* `stats/capture_stats.py` computes the `leads_annotated.csv` Spearman over the pairs where both
  values are finite, records the count used and the count dropped, and raises rather than write a
  nan; rows with a non-finite value are also dropped from the `leads_annotated` reference dataset.
  The pinned `stats/fixtures/spearman.json` and `stats/fixtures/measurements.json` still record
  the all-rows figure, rho 0.3965 on n = 50 (scipy returns nan when any value is NaN, so every
  `Selectivity_Index` was defined when that was captured). The file now has 2 undefined values and
  the same computation gives rho 0.3589 (p 1.223e-02) on the 48 defined rows. Those two fixtures,
  `stats/fixtures/MANIFEST.json` and the passages of `stats/SPEC.md` that quote the 50-row figure
  are stale until the owner regenerates them.

---

## 2. Cross-cluster convention register

The five conventions most likely to disagree silently, checked in every file that mentions them.

| convention | settled value | files checked | verdict |
|---|---|---|---|
| **Tanimoto epsilon** | `eps = 1e-8` **clamping the denominator**, `inter / maximum(union, 1e-8)`, then `clip(·,0,1)` | `gp/SPEC.md:66-70`, `icm/SPEC.md:77,747-748`, `primitives/SPEC.md:79,99,127,330-338`, `qnehvi/SPEC.md:1328` | **agree**. No file places eps in the numerator or inside a sqrt. `primitives` deviation rows 10 and 11 explicitly overrule two earlier mis-statements ("eps inside the sqrt denominator", "eps in the numerator destroys the diagonal") and record that the second is false in float32 (residual exactly `0.0`) and true only in float64 (`1.5e-10`). |
| **Reference point & filter** | `r = 0` in the normalized maximization cube, never changing; rows admitted by the **strict** filter `(Y > ref).all(-1)` | `hv/SPEC.md:375,623-710,1023`, `boxdecomp/SPEC.md:465,1057-1104`, `qnehvi/SPEC.md:111,242` | **agree**, and both `hv` and `boxdecomp` independently derive that strict and non-strict are measure-equivalent, then ship strict for a stated reason (`hv`: work saved and front reporting; `boxdecomp`: botorch parity plus the unclamped-volume trap, `sum = +2.0` for `y=(-1,-1,2)` against a true HV of `0.0`). |
| **Dominance convention** | weak dominance (`≥` on all, `>` on at least one); **duplicates kept for the reported front, deduplicated internally** | `hv/SPEC.md:446-448,491,515,608,1083-1089`, `boxdecomp/SPEC.md:208,466-480`, `qnehvi/SPEC.md:227` | **agree on the convention, with a deliberate difference in *precondition* that must not be read as a contradiction.** `hv`'s HSO indicator requires a deduplicated non-dominated input and says so; `boxdecomp`'s decomposition tolerates and ignores dominated rows (verified byte-identical cells with and without a dominated row), and `qnehvi` *relies* on that tolerance and forbids pre-filtering in the estimator. Two different preconditions for two different algorithms. |
| **Front frame** | HV is computed in the **normalized** frame; the **reported** Pareto front is computed in **original units** with signs | `hv/SPEC.md:436-622,727-991`, `qnehvi/SPEC.md:82-130` | **agree**. `hv` measures the consequence honestly: 19 of 448 archived iterations have a different front size in the two frames, because clipping creates ties the original-unit frame does not have. `qnehvi` works only in the normalized frame, which is consistent. |
| **Kronecker ordering** | **data-major**: `Σ = kron(K_data, B_task)`, flat index `a = i·K + t` | `icm/SPEC.md:108,194-222,777-778,832`, `gp/SPEC.md:208-222`, `primitives/SPEC.md:514,822-834,1094` | **agree.** `icm` calls it data-major, `gp` calls the same thing interleaved (`row = n*T + t`); both are `i·K + t`. `icm` settles the transpose trap with a measurement (`kron(B,K)` is off by `3.0`) and records that the docstring's ordering is simply wrong. `gp` and `primitives` add the separate, non-contradictory fact that for the *independent* model the joint covariance is exactly block diagonal per task (off-block max abs `0.0`), so per-task factorization is mathematically identical and `T×` cheaper. |
| **float32/float64 boundary** | GP layer float32; **everything downstream float64** | `gp/SPEC.md:42-43`, `icm/SPEC.md:674-681`, `hv/SPEC.md:351-352`, `boxdecomp/SPEC.md:1151`, `qnehvi/SPEC.md:1227-1228`, `stats/SPEC.md:46-47`, `primitives/SPEC.md:123,164,451,822-828` | **agree**, with two dtype-conditional tolerances flagged in section 4. `hv` quantifies why the boundary must stop at the GP: a float32 round-trip of the HV inputs moves the indicator by `3.18e-09`, seven orders of magnitude worse than the algorithm-portability error. |
| **Shared constants** | Tanimoto eps `1e-8`; noise floors `1e-4` (both `raw_task_noises` and `raw_noise`, so `τ ≥ 2e-4`); acquisition variance floor `1e-9`; softplus linear branch above `20`; Adam `lr 0.1`, `betas (0.9, 0.999)`, `eps 1e-8`, `200` steps; `max_cholesky_size 800` compared against `N·T` | all nine SPEC.md | **agree**. `gp/SPEC.md:836` explicitly defers ownership of the `1e-9` variance floor to the acquisition cluster and `qnehvi/SPEC.md:139,290` takes it, so there is exactly one owner. |

**No contradiction between any two SPEC.md files was found on any of these axes.** The four
things that look like disagreements on a fast read are all reconciled above, and each is
reconciled *by the specs themselves*, not by me.

---

## 3. Independent spot-checks

Five claims re-derived from the SPEC text alone, choosing the ones whose failure would be most
damaging. Botorch/gpytorch/scipy were **not** used in any of the reimplementations; where a
decomposition was needed I used `spec/boxdecomp/reference_lacour.py`, which is paper-derived.

| # | claim re-derived | their number | **my number** | verdict |
|---|---|---|---|---|
| 1 | `gp` analytic MLL gradients vs the recorded gpytorch autograd, 6 cases × 2 hyperparameter settings | worst `1.168e-08` | **`1.1675833355706544e-08`** | **confirmed**, and the per-case table matched row for row. Loss: their `1.024e-07`, mine `1.0235877190289955e-07`. My analytic gradient vs theirs: `≤ 1.39e-17`. |
| 2 | `admet` — the `.npz` export is lossless: a predictor built from `SPEC.md` §5–6 reading **only** the export reproduces sklearn bitwise on all 29,678 library molecules | `max abs diff 0.0`, all 3 models | **`0.0`, all 3 models**; leaf indices exact; heads (`10**raw`, sigmoid) bitwise; AD gate exact with 0 out-of-domain; `np.sum(axis=1)` failure counts **25895 / 25776 / 24772** with `1.066e-14 / 2.442e-15 / 7.105e-15`; knife edge **766**; on-threshold **222 / 220 / 373** | **confirmed, every number exactly.** No sklearn, no joblib, no `.joblib` file touched. |
| 3 | `qnehvi` T-ID-1 — the estimator fed botorch's own recorded Sobol normals reproduces botorch's exact qNEHVI scores | `1.388e-17` abs, `7.494e-16` rel, Spearman `1.0` | **`1.387779e-17` abs, `7.493658e-16` rel, Spearman `1.0`, identical `argsort`** | **confirmed** — and obtained with a **fully numpy pipeline** (my estimator + `spec/boxdecomp`'s Lacour cells, no botorch anywhere), which is what the cluster listed as unresolved. See section 5. |
| 4 | `hv` archive replay — 448 iterations of 53 archived runs against `history.csv` | max residual `1.110e-16`, mean `8.983e-18` | **max `3.053113e-16`, mean `8.13e-17`** by a *third* algorithm (Lacour cells + the volume-complement identity) | **confirmed.** My residual is ~3× theirs because it is a different exact algorithm; both sit inside the stated 4-ulp gate (`≈8.9e-16`), which is exactly why the spec forbids tightening to bitwise across algorithms. Front sizes `3/62`, zero-coordinate front members `256`, non-monotone steps `0`, `n_active == 5` on 448/448 — all reproduced exactly. |
| 5 | `stats` — the paired t-test tail against the mpmath 50-dps arbiter, the exact Wilcoxon null distribution, and the Spearman shortcut regression | t-tail `9.201e-12` (df≤5000) / `4.260e-14` (df≤50); scipy's own error `3.120e-09` | **`9.2005e-12` / `4.2631e-14` / `3.1202e-09`**; Wilcoxon `exact_table` reproduced for all 15 rows (`sum == 2**n`, palindromic) with `p(n=5) == 0.0625` and `p(n=7) == 0.015625` exactly; Spearman `2.220e-16`; midrank `7/7`; shortcut wrong on exactly **15** pairs | **confirmed, every number.** |

### What the spot-checks exposed

Three of the five only reproduced after I followed a step I had first skipped, and each miss is
a measured warning about how little headroom the stated gate has:

1. **`gp`'s `1e-7` per-group gradient gate has essentially no headroom against the
   standardization.** `gp/SPEC.md` §10.1 specifies `y_mean = Y.mean(axis=0)` over the **full
   `(N,5)` float32 matrix** with the grey-box mask applied *afterwards*. Computing the same
   statistics in float64 instead moved my gradient by **`1.413e-07`**; using
   `nanmean`/`nanstd` over the `(N,T)` observed slice moved it by **`1.005e-07`**. Both blow the
   `1e-7` gate, and both are reasonable-looking readings of "plain float32 mean/std with
   `ddof=0`". The SPEC is correct; the *fixture* is unforgiving, and §10.1's dtype must be read
   as normative, not incidental.
2. **`stats`' t-tail gates are only meetable with §4.4's accurate-logarithm entry point.**
   Passing `math.log` of the rounded beta argument instead of
   `log_x = -log1p(t²/ν)`, `log_y = -log1p(ν/t²)` gave me **`7.98e-09`** — a 1000× miss of the
   `1e-11` gate, and the same order as the `6.4e-09` the SPEC itself measures for that mistake.
3. **`stats`' BITWISE Pearson gate is formulation-specific.** It holds for §5.3's
   normalise-then-dot with max-scaling. The textbook `(a·b)/√((a·a)(b·b))` gave me only
   **10/37 bitwise, worst `5.551e-16`**. §5.3 says to use its arrangement; this quantifies the
   cost of not doing so.

---

## 4. Defects and gaps found

### 4.1 `hv/fixtures/normalized_frames.json` cannot be matched bitwise from `SPEC.md` §4.2 — CONFIRMED DEFECT

`hv/SPEC.md` §4.2 states the map as a per-column affine transform plus a clip, says
*"a reimplementation must reproduce `Y_normalized` **bitwise** ... Do not allow a tolerance
here"*, and adds *"algebraically-equivalent rearrangements ... are **not** bitwise equivalent
and will fail the fixture"* (I verified the latter: `(y-lo)*(1/span)` differs on **498 of 1570**
cells, worst `1.110e-16`).

But `capture_hv.py:1024-1025` does

```python
U = np.full(Y.shape, np.nan)
U[finite] = normalize_frame(Y[finite], lo, hi, signs)     # finite = isfinite(Y).all(axis=1)
```

i.e. it pre-masks **every row that is not fully observed to NaN across all five columns**.
§4.2's elementwise formula leaves the finite ADMET entries of such rows finite. Applying §4.2
exactly therefore fails on **24 of 1570 cells** in the shipped fixture — e.g.
`benchmark_thicc_2026-08-14/random/seed_0` row 8 column 2, raw `0.9583320617675781`, §4.2 gives
`0.04159944343659868`, the fixture holds `nan`. The rule appears only in the fixture's own
`_doc` string as *"NaN rows are NaN in both"*, which does not say that a *partially* NaN row's
finite entries are discarded.

Impact on reported science: **none.** `scored_hypervolume` drops non-fully-observed rows before
normalizing, so those cells never reach a front or a hypervolume. Impact on the rewrite: an
implementer following §4.2 fails a gate the spec says admits no tolerance. **Fix is one
sentence in §4.2**, not a code change.

### 4.2 `gp`'s numeric fixtures do not carry their own inputs — GAP

`gp` is the only cluster whose numeric fixtures are not self-describing. `mll_gradients.json`,
`trained_hyperparameters.json` and `posterior.json` store `theta_raw` and the results but **not
`X` or `Y`**, and `SPEC.md` §16 describes the synthetic construction only in prose (*"a fixed
random projection of the fingerprint, standardized, mapped to `PfDHFR ~ N(-8.0, 1.2)` /
`hDHFR ~ N(-7.0, 1.0)` from `default_rng(20260927)`"*). It does **not** state:

* which library rows each case uses (`capture_gp.py:708`: `X = fp[:N]`, the first `N`);
* the projection's draw (`rng.standard_normal((D, 2)) / sqrt(D)`);
* that columns 2–4 are drawn from the **same** generator afterwards, so the stream order is
  part of the construction;
* the standardization dtype (see section 3, item 1).

Every other cluster embeds its inputs: `boxdecomp` in `arrays.bin`, `hv` as base64 float64 with
per-array sha256, `qnehvi`/`icm` as `.npy`, `stats` as explicit grids, `admet` as a referenced
library sha plus a saved synthetic `X`, `primitives` inside its `.npz`. Consequence: the `gp`
fixtures are only usable via `capture_gp.py`. That is legitimate — `capture_gp.py` is a spec
artifact, not library source — but it should be stated, or §16 should be completed. I
reconstructed the inputs from the capture script and then reproduced their numbers exactly
(section 3, item 1), so the fixtures are *correct*; they are just not *self-contained*.

### 4.3 Two dtype-conditional agreement criteria are stated without their dtype — GAP

* `gp/fixtures/tanimoto_gram.json`: `full_matrix_n12_float64` and the
  `sha256_float64_C_order` digests are float64 values, and the cluster's own criterion says
  *"EXACT, not a tolerance ... assert bitwise equality"*. A float32 Gram — which the settled
  dtype decision mandates for the GP layer — differs from them by up to `2.454e-08`, a number
  the same fixture records in its `max_abs_float32_minus_float64` field. The test must compute
  in float64; the criterion should say so.
* `icm/fixtures/05_posterior/*`: the criterion is `≤ 1e-9` absolute, but
  `icm/fixtures/11_precision.json` measures the shipped float32 path at `5.20e-08` absolute on
  the mean — **~50× over the gate** — and `icm/SPEC.md` §11 licenses `1e-5` relative for a
  float32 port. The `1e-9` gate is for a float64 reference implementation only. `04_kron` and
  `06_gradients` have the same property.

Neither is a wrong number. Both are criteria that will be misapplied by a reader who does not
also read §11 / the precision fixture, which is exactly the failure mode a manifest exists to
prevent, so both are recorded per-file in section 6.

### 4.4 Criteria that live nowhere machine-readable — GAP (addressed by this file)

`primitives`, `icm`, `stats` and `hv` embed per-array or per-file agreement criteria inside
their fixtures. `gp`, `boxdecomp`, `qnehvi` and `admet` do **not** — their criteria exist only
in `SPEC.md` prose and in the agents' self-reports, which are not files in the repo. Section 6
of this manifest is now the durable record.

### 4.5 Three copies of the objective bounds, one checksum — WATCH ITEM

The normalized frame is defined by ten numbers, and they exist in **three** places:

| location | checksummed? | asserted at load? |
|---|---|---|
| `spec/bounds/objective_bounds.frozen.json` | yes — `objective_bounds.frozen.json.sha256`, and a canonical payload hash `03bd72e4...707f` | `hv/SPEC.md` §4.1 mandates asserting the payload hash with no tolerance |
| `evaluation_bounds.json` (repo root, the runtime copy) | no | no |
| `spec/qnehvi/capture_qnehvi.py:73` `BOUNDS`, a hardcoded literal | no | no |

I verified all three are **bitwise identical** today (max abs diff `0.0` on all ten numbers,
signs and task names equal), and that the frozen artifact's own sidecar and canonical payload
hash both reproduce. So nothing is wrong now. But two of the three copies have no checksum and
no load-time assertion, so a future re-freeze would silently desynchronize them, and the
`spec/qnehvi` fixtures — which depend on the frame through every normalized value — would then
be measuring a different frame from the one `spec/hv` asserts. **Recommend: `capture_qnehvi.py`
assert the payload hash the way `hv` does, or read the frozen file.** One line, and it converts
a silent failure into a loud one.

### 4.6 What the specs are honest about, and it is worth saying

The clusters volunteered 55 unresolved items between them, and the ones that matter are real
engineering risks rather than hedges. Three deserve to be read before any schedule is set:

* **`boxdecomp` and `qnehvi` both flag acquisition cost as harder than the charter assumes.**
  qNEHVI at `m > 2` needs one box decomposition per MC sample. At the settled `N = 512` and the
  archived median front size of 44 that projects to roughly 2.5–3 minutes of CPU per
  acquisition call, ~6 minutes at the archived maximum front of 60, and `qnehvi` extrapolates
  ~149 s per BO iteration at `B = 110`. Raising `N` from 128 to 512 multiplies it by four.
  Three mitigations are identified (reuse the 512 decompositions across all candidates in an
  iteration; Lacour's incremental insertion to update pass 1 across iterations; failing that,
  reduce `N` and say so) and **none is prototyped**. The 'docking dominates wall clock' premise
  was argued about `N` and may fail for `B` late in a run.
* **`icm` needs a decision, not more measurement.** The shipped 200-step fit and the true MLE
  are different models (task correlation `0.994` vs `1.000000000`, `σ² 0.0155` vs `0.0001`,
  `det B 5e-4` vs `1.2e-10`). Reproducing the archive's *behaviour* means pinning an Adam
  trajectory, which contradicts 'derive from the published mathematics'. Reproducing the
  *mathematics* means accepting a degenerate boundary MLE with a singular `B` and a task
  correlation of exactly `1`, which makes the ICM's headline claim read as *'the model learns
  the two tasks are the same task'*. That is a scientific choice the charter did not make.
* **The `hv` cluster found two frame-level hazards it cannot fix inside its own scope:**
  `HV = 0.0` is an overloaded return value (emitted for empty input, no active objective, no
  fully-observed row, everything at or below the reference, *and* a genuine zero-measure front),
  and because `hDHFR_Docking` is maximized, 79 of 4061 pooled archived rows saturate at exactly
  `1.0` — 15 of them with *positive* Vina scores up to `+10.73` kcal/mol, i.e. physically
  non-binding poses scoring as a perfect selectivity result. **Any correct reimplementation of
  this spec reproduces that faithfully.** Both need an owner decision before the campaigns are
  re-run, because both change reported numbers.

---

## 5. An unresolved item closed by measurement

`spec/qnehvi`'s unresolved item 4 reads: *"The `spec/boxdecomp` contract is now executable
(`check_decomposition_contract.py`) but has only been run against botorch's decomposition.
Every number in this SPEC uses botorch cells. They do not transfer to a fully-numpy pipeline
until that script passes against the numpy decomposition."*

I ran it. Pointing `check_decomposition_contract.py` at
`spec/boxdecomp/reference_lacour.nondominated_cells` in place of botorch's
`FastNondominatedPartitioning` produced a **byte-identical** `decomposition_contract.json`,
sha256 `8fc0ab80f3b725790a74b94aad312e1bdfa0bfa4a9f9e52ea221824fef28cc01` — the same file, with
the same `K_min = 1`, `K_max = 328`, `n_infinite_upper_bound_entries = 4378`,
`n_degenerate_cells_L_ge_U = 0` and the same `all_zero_front` block. And the estimator built on
those numpy cells reproduced botorch's exact qNEHVI scores to `1.387779e-17` absolute —
bit-for-bit the agreement figure the cluster reports with botorch cells (section 3, item 3).

**That item is closed.** The `spec/qnehvi` numbers do transfer to a fully-numpy pipeline.

---

## 6. Per-cluster index

Columns: path relative to `spec/`; bytes; first 16 hex of sha256 (full digests in `CHECKSUMS`);
determinism class; the agreement criterion an implementation must meet. Criteria are transcribed
from each cluster's `SPEC.md` and fixture sidecars; anything marked **VERIFIED this session** was
measured here.

### `spec/bounds/` — Frozen objective bounds + the archived-hypervolume settlement (phase-1 artifact, no `fixtures/` subtree)

Capture: `freeze_bounds.py, measure_clipping.py, reproduce_archived_hypervolume.py` · 9 files · 238,188 bytes (fixtures/artifacts 187,434 bytes)

| path | bytes | sha256[:16] | determinism | agreement criterion |
|---|---:|---|---|---|
| `bounds/INDEPENDENT_REVERIFICATION.md` | 6,559 | `cad46df38574ddc0` | `n/a` | Prose settlement record. |
| `bounds/SETTLEMENT.md` | 17,648 | `5f3e34815a83a822` | `n/a` | Prose settlement record. |
| `bounds/clipping_measurements.json` | 88,946 | `7845ebbabb2160d5` | `bitwise` | Measurement record; informational. Re-run measure_clipping.py to reproduce. |
| `bounds/freeze_bounds.py` | 12,166 | `81670d22befeaa92` | `n/a` | Capture/verification script. |
| `bounds/measure_clipping.py` | 7,489 | `5fb9789e3665e0a9` | `n/a` | Capture/verification script. |
| `bounds/objective_bounds.frozen.json` | 9,452 | `abaa99a4d5c21b62` | `bitwise` | Frozen artifact. Assert its own sha256 sidecar AND the canonical payload hash sha256(json.dumps({task_names,signs,bounds},sort_keys=True,separators=(',',':'))) == 03bd72e4...707f. No tolerance. VERIFIED this session: both hashes reproduce. |
| `bounds/objective_bounds.frozen.json.sha256` | 95 | `e5da30adfcd84c5b` | `bitwise` | Sidecar for the above; must equal shasum -a 256 of the frozen json. |
| `bounds/reproduce_archived_hypervolume.py` | 6,892 | `f89662657aba24b9` | `n/a` | Capture/verification script. |
| `bounds/reproduction_residuals.json` | 88,941 | `85be77500dc3abe7` | `bitwise` | Measurement record; informational. |

### `spec/primitives/` — Tanimoto kernel, hyperparameter transforms, Cholesky + jitter ladder + triangular substitution, Adam

Capture: `capture_primitives.py` · 11 files · 1,875,316 bytes (fixtures/artifacts 1,732,797 bytes)

| path | bytes | sha256[:16] | determinism | agreement criterion |
|---|---:|---|---|---|
| `primitives/SPEC.md` | 66,411 | `6cd589d07aad4307` | `n/a` | The specification. Not a fixture; hashed here so drift in a spec is detectable alongside drift in its fixtures. |
| `primitives/capture_primitives.py` | 76,108 | `46636477922948aa` | `n/a` | The capture script that generates this cluster's fixtures. Re-run it in a fresh interpreter and byte-compare to detect drift. |
| `primitives/fixtures/INDEX.json` | 1,658 | `f000552aed1ef419` | `bitwise` | Per-array sidecar: dtype, shape, `reproducibility` tag and agreement_criterion for every array in the matching .npz. Read it before comparing anything. |
| `primitives/fixtures/adam.json` | 10,529 | `a3136e09c9f04faa` | `bitwise` | Per-array sidecar: dtype, shape, `reproducibility` tag and agreement_criterion for every array in the matching .npz. Read it before comparing anything. |
| `primitives/fixtures/adam.npz` | 55,674 | `fcdf2b9324f11c17` | `bitwise` | All bitwise. quad_traj_x_f64 atol 1e-12/rtol 1e-11; m,v rtol 1e-11; rosen_traj_x_f64 rtol 1e-9; quad_traj_x_f32 rtol 1e-4 (float32 floor is the measured 9.666e-07 dtype gap). HARD: oracle_torch_adam_max_abs_f64 < 1e-14; _f32 < 1e-6; oracle_foreach_vs_single_max_abs == 0.0 exactly; every entry of quad_first_step_f64 within 1e-6 of lr=0.1 in magnitude. NEGATIVE tests -- the implementation must be FAR from: variant_inside_sqrt (~1.9e-3), variant_no_bias_correction (~3.3); variant_torch_stepsize must be TINY (~4.4e-16, the acceptable rearrangement). |
| `primitives/fixtures/cholesky_and_solves.json` | 41,018 | `03510363f46bbf25` | `bitwise` | Per-array sidecar: dtype, shape, `reproducibility` tag and agreement_criterion for every array in the matching .npz. Read it before comparing anything. |
| `primitives/fixtures/cholesky_and_solves.npz` | 1,234,741 | `c7504a6df857ec1d` | `bitwise` | MIXED per-array; read the tag in cholesky_and_solves.json. `bitwise`: all input matrices, ladder_rungs_f32 == [9.999999974752427e-07, 9.999999747378752e-06, 9.999999747378752e-05], ladder_rungs_f64, ladder_raises_on_unrescuable==1, both sum-vs-BLAS gap arrays. `lapack-dependent` (every factor/solve/eigenvalue/condition number): compare L via L@L.T ~= A + jitter*I at rtol 1e-5 (f32) / 1e-13 (f64); solves rtol 1e-13 (f64) / 1e-5 (f32); tri_residual_f64 < 1e-11, tri_residual_f32 < 1e-4. DO NOT ASSERT numpy_cholesky_succeeds_unjittered, torch_cholesky_ex_info, ladder_chosen_jitter_f32, ladder_n_attempts -- INFORMATIONAL; numpy and torch disagree on 3 of 4 cases in both directions. Assert the ladder RULE instead: rungs are base*10^i (not cumulative 1.1e-5), max_tries=3 gives 4 attempts, exhaustion raises. |
| `primitives/fixtures/hyperparameter_transforms.json` | 14,682 | `7a4280a7d55a8d06` | `bitwise` | Per-array sidecar: dtype, shape, `reproducibility` tag and agreement_criterion for every array in the matching .npz. Read it before comparing anything. |
| `primitives/fixtures/hyperparameter_transforms.npz` | 9,508 | `fda45befb7c7907f` | `bitwise` | All arrays bitwise. Grids/masks/flags/inventory/constants exact. Transcendentals rtol 1e-15 (f64) / 1e-6 (f32). HARD exact: linear_branch_is_identity all True; softplus_grad==1.0 wherever z>20; oracle_greaterthan_minus_positive min==max==9.999999747378752e-05; oracle_positive_f64-softplus_f64==0; param_total_T{1,2,3}=={4,7,10}==3T+1; param_constraint_code_T*==[2,2,0,1]; constant_mean_is_raw==1; constant_mean_negative_roundtrip==[-3.5,7.25]. inv_softplus_roundtrip_err < 1e-12*max(1,y). |
| `primitives/fixtures/tanimoto_kernel.json` | 26,536 | `ac9a58a86d52b3a9` | `bitwise` | Per-array sidecar: dtype, shape, `reproducibility` tag and agreement_criterion for every array in the matching .npz. Read it before comparing anything. |
| `primitives/fixtures/tanimoto_kernel.npz` | 338,451 | `3f8e8b9842516c1e` | `bitwise` | MIXED per-array; read the `reproducibility` tag in tanimoto_kernel.json. `bitwise` arrays: np.array_equal. HARD: diag(gram_f32_n*)==1.0 exactly; gram_f32_rect_13x7 == gram_f32_rect_7x13.T to 0 ULP; all 3 slices of gram_f32_batched_3x9x9 equal; oracle_gram_max_abs_diff == 0.0 exactly. `lapack-dependent` (eigvalsh_*, *_min_eig): min eig >= -1e-12 and sorted eigenvalues at rtol 1e-10, never byte-compare. `stochastic` minhash_collision_estimate: \|est-exact\| < 4*sqrt(J(1-J)/200000). `rdkit-version-dependent` rdkit_*: only within RDKit 2026.03.6, else smoke test (diag==1.0, symmetry, range [0,1]). |

### `spec/gp/` — Independent multi-output exact GP (`mogp.py`): 3T+1 scalars, MLL, analytic gradients, posterior

Capture: `capture_gp.py` · 11 files · 295,197 bytes (fixtures/artifacts 184,414 bytes)

| path | bytes | sha256[:16] | determinism | agreement criterion |
|---|---:|---|---|---|
| `gp/SPEC.md` | 53,956 | `405d3ed1768353ff` | `n/a` | The specification. Not a fixture; hashed here so drift in a spec is detectable alongside drift in its fixtures. |
| `gp/capture_gp.py` | 56,827 | `d4926498f54724fe` | `n/a` | The capture script that generates this cluster's fixtures. Re-run it in a fresh interpreter and byte-compare to detect drift. |
| `gp/fixtures/MANIFEST.json` | 1,464 | `8a9986c6add3d7a5` | `bitwise` | sha256 of the 7 byte-comparable fixtures; stochastic_threshold.json excluded by design. `capture_gp.py --verify` checks it. |
| `gp/fixtures/metadata.json` | 1,435 | `78a492911cc05a3c` | `bitwise` | Provenance only. An implementation must read the same fingerprints.npy (sha256 match) and the same constants (noise floor 1e-4, Tanimoto eps 1e-8, softplus threshold 20, Adam lr 0.1 / betas (0.9,0.999) / eps 1e-8 / 200 steps, max_cholesky_size 800). |
| `gp/fixtures/mll_gradients.json` | 38,661 | `90272dc26c56d33b` | `bitwise` | Loss abs <= 1e-6. Gradients vs grad_gpytorch_autograd: abs <= 1e-7 PER PARAMETER GROUP (check the four groups separately). Your own analytic-vs-central-FD self-check: abs <= 3e-9. Normalization discriminator: assert abs(loss-(-log_prob/(N*T))) <= 1e-6 AND abs(loss-(-log_prob/N)) > 1. WARNING measured this session: the fixture carries no X/Y, so you must regenerate the inputs (X = fingerprints.npy[:N]; targets from the section 16 recipe) AND standardize EXACTLY as section 10.1 says -- float32 mean/std over the FULL (N,5) matrix. Doing the reduction in float64 instead moves the gradient by 1.413e-07, and using nanmean/nanstd over the (N,T) observed slice moves it by 1.005e-07: both BLOW the 1e-7 gate. The gate has no headroom against a standardization deviation. |
| `gp/fixtures/model_parameters.json` | 8,173 | `b8e3227e3ca78d47` | `bitwise` | Structural assertions exact: len(theta)==3T+1, the four groups with the right shapes, all inits exactly 0.0, no priors, no lengthscale. Softplus max abs diff <= 1e-6 in float32. Flat direction: loss change under (sigma2+e, d_t-e) must be exactly 0.0. |
| `gp/fixtures/posterior.json` | 68,461 | `9257dccd4e7f4d0a` | `bitwise` | Mean abs <= 1e-5 kcal/mol. Variance rel <= 1e-5. Which variance: assert abs(var-var_f-tau*y_std^2) <= 1e-6 AND abs(var-var_f) > 1e-3 so a latent-variance implementation cannot pass. Assert the 3 ADMET columns are entirely NaN. Evaluated at the ORACLE's theta -- do not use as a target for an implementation that fits its own hyperparameters first. |
| `gp/fixtures/standardization_nan.json` | 4,905 | `cef9fd459b368e1d` | `bitwise` | EXACT for the discrete outcomes: observed mask and trained task count for all 6 cases; the all-NaN-both case must raise. y_mean/y_std bitwise on observed columns, NaN in exactly the same positions. Assert ddof=0 explicitly: ddof=1 changes a fixture column from 0.80918384 to 0.82301706. |
| `gp/fixtures/stochastic_threshold.json` | 7,961 | `8bc687641cd268c3` | `stochastic` | NOT a comparison target; excluded from gp/fixtures/MANIFEST.json by design. Only the qualitative assertions are reproducible: is_exact True for every N*T <= 800 (spread exactly 0.0) and False for every N*T > 800; the N=420/T=1 and N=800/T=1 controls exact (pins the threshold to N*T, not N); seeded and deterministic_probes spreads 0.0 while the unseeded spread is nonzero. VERIFIED this session: differs between runs in exactly the 4 rows above N*T=800 and nowhere else. |
| `gp/fixtures/tanimoto_gram.json` | 27,340 | `65063bdd55dd6943` | `bitwise` | EXACT -- but ONLY IN THE PRECISION THE FIXTURE RECORDS. full_matrix_n12_float64 and by_size.n{N}.sha256_float64_C_order are FLOAT64 values: compute S in float64 for this test. A float32 Gram differs from them by up to 2.454e-08 (the fixture's own max_abs_float32_minus_float64 field), so a float32 implementation CANNOT byte-match and must not be asserted against them. Separately assert S[i,i]==1.0 exactly and S==S.T bitwise in BOTH precisions. |
| `gp/fixtures/trained_hyperparameters.json` | 26,014 | `e01f642427648f00` | `bitwise` | Tiered by identifiability. tau_effective rel <= 1e-5. outputscale abs <= 1e-6. theta_raw abs <= 1e-5. shared_sigma2 and task_noise_d SEPARATELY abs <= 1e-4 only -- DO NOT TIGHTEN: they lie on the exactly-flat direction. Weight mid_n100_t2 and edge_n400_t2 most; small_n30_t2 and real_docking_n36_t2 are numerically insensitive (outputscale collapses). |

### `spec/icm/` — Coregionalized ICM GP (`mogp_coregionalized.py`): Kronecker structure, gradients, identifiability

Capture: `capture_icm.py` · 39 files · 449,634 bytes (fixtures/artifacts 342,073 bytes)

| path | bytes | sha256[:16] | determinism | agreement criterion |
|---|---:|---|---|---|
| `icm/SPEC.md` | 46,702 | `8c15e00a89ac198c` | `n/a` | The specification. Not a fixture; hashed here so drift in a spec is detectable alongside drift in its fixtures. |
| `icm/capture_icm.py` | 60,859 | `b0d8b3527656421d` | `n/a` | The capture script that generates this cluster's fixtures. Re-run it in a fresh interpreter and byte-compare to detect drift. |
| `icm/fixtures/00_environment.json` | 273 | `3f969bc1640b5395` | `stochastic` | Not compared. Records which oracle versions produced the rest. |
| `icm/fixtures/01_parameterization.json` | 2,133 | `9747eb88c90c690a` | `bitwise` | Exact equality of every structural field (booleans, counts, formula strings); every max_abs_diff <= 1e-12. |
| `icm/fixtures/02_ordering.json` | 759 | `f614c2b278667708` | `bitwise` | THE TRANSPOSE TRAP. max\|Sigma_impl - kron(K_data,B_task)\| <= 1e-12 AND max\|Sigma_impl - kron(B_task,K_data)\| >= 1.0. The `answer` string must match exactly. Passing only the first test means the docstring ordering was silently accepted. |
| `icm/fixtures/02_ordering/B_task.npy` | 160 | `768d8ccb4618713c` | `bitwise` | THE TRANSPOSE TRAP. max\|Sigma_impl - kron(K_data,B_task)\| <= 1e-12 AND max\|Sigma_impl - kron(B_task,K_data)\| >= 1.0. The `answer` string must match exactly. Passing only the first test means the docstring ordering was silently accepted. |
| `icm/fixtures/02_ordering/K_data.npy` | 200 | `9f3bd5a41bd74caa` | `bitwise` | THE TRANSPOSE TRAP. max\|Sigma_impl - kron(K_data,B_task)\| <= 1e-12 AND max\|Sigma_impl - kron(B_task,K_data)\| >= 1.0. The `answer` string must match exactly. Passing only the first test means the docstring ordering was silently accepted. |
| `icm/fixtures/02_ordering/assembled_dense.npy` | 416 | `747160ebdc7f6faf` | `bitwise` | THE TRANSPOSE TRAP. max\|Sigma_impl - kron(K_data,B_task)\| <= 1e-12 AND max\|Sigma_impl - kron(B_task,K_data)\| >= 1.0. The `answer` string must match exactly. Passing only the first test means the docstring ordering was silently accepted. |
| `icm/fixtures/02_ordering/kron_data_major.npy` | 416 | `1fcd0cb7dbcf7a15` | `bitwise` | THE TRANSPOSE TRAP. max\|Sigma_impl - kron(K_data,B_task)\| <= 1e-12 AND max\|Sigma_impl - kron(B_task,K_data)\| >= 1.0. The `answer` string must match exactly. Passing only the first test means the docstring ordering was silently accepted. |
| `icm/fixtures/02_ordering/kron_task_major.npy` | 416 | `2e13e39b4353440a` | `bitwise` | THE TRANSPOSE TRAP. max\|Sigma_impl - kron(K_data,B_task)\| <= 1e-12 AND max\|Sigma_impl - kron(B_task,K_data)\| >= 1.0. The `answer` string must match exactly. Passing only the first test means the docstring ordering was silently accepted. |
| `icm/fixtures/03_real_data.json` | 674 | `22cf519e05b74fea` | `bitwise` | Exact byte equality of the .npy/.txt files and every field of 03_real_data.json including smiles_sha256. y_normalized_train.npy to <= 1e-6 relative. |
| `icm/fixtures/03_real_data/fingerprints_test.npy` | 82,048 | `8f461a9e9d354ac7` | `bitwise` | Exact byte equality of the .npy/.txt files and every field of 03_real_data.json including smiles_sha256. y_normalized_train.npy to <= 1e-6 relative. |
| `icm/fixtures/03_real_data/fingerprints_train.npy` | 82,048 | `b29c9f5304925430` | `bitwise` | Exact byte equality of the .npy/.txt files and every field of 03_real_data.json including smiles_sha256. y_normalized_train.npy to <= 1e-6 relative. |
| `icm/fixtures/03_real_data/le_train.npy` | 448 | `aba82779f3006a67` | `bitwise` | Exact byte equality of the .npy/.txt files and every field of 03_real_data.json including smiles_sha256. y_normalized_train.npy to <= 1e-6 relative. |
| `icm/fixtures/03_real_data/smiles.txt` | 3,585 | `ff27ee73a4712914` | `bitwise` | Exact byte equality of the .npy/.txt files and every field of 03_real_data.json including smiles_sha256. y_normalized_train.npy to <= 1e-6 relative. |
| `icm/fixtures/03_real_data/y_normalized_train.npy` | 768 | `a88c4185f4c3af05` | `bitwise` | Exact byte equality of the .npy/.txt files and every field of 03_real_data.json including smiles_sha256. y_normalized_train.npy to <= 1e-6 relative. |
| `icm/fixtures/04_kron.json` | 586 | `615de79e67d03048` | `bitwise` | FLOAT64 reference. logdet abs diff <= 1e-9; solve max abs diff <= 1e-10; \|\|Sigma alpha - r\|\|_inf <= 1e-9; eigvalsh(Sigma_dense).min() > 0 and cholesky succeeds WITHOUT jitter. |
| `icm/fixtures/04_kron/B.npy` | 160 | `e6bf2ca1cb8e3968` | `bitwise` | FLOAT64 reference. logdet abs diff <= 1e-9; solve max abs diff <= 1e-10; \|\|Sigma alpha - r\|\|_inf <= 1e-9; eigvalsh(Sigma_dense).min() > 0 and cholesky succeeds WITHOUT jitter. |
| `icm/fixtures/04_kron/K_xx.npy` | 12,928 | `720aac9ad5d6887a` | `bitwise` | FLOAT64 reference. logdet abs diff <= 1e-9; solve max abs diff <= 1e-10; \|\|Sigma alpha - r\|\|_inf <= 1e-9; eigvalsh(Sigma_dense).min() > 0 and cholesky succeeds WITHOUT jitter. |
| `icm/fixtures/04_kron/Sigma_dense.npy` | 51,328 | `0f750e49b70fa722` | `bitwise` | FLOAT64 reference. logdet abs diff <= 1e-9; solve max abs diff <= 1e-10; \|\|Sigma alpha - r\|\|_inf <= 1e-9; eigvalsh(Sigma_dense).min() > 0 and cholesky succeeds WITHOUT jitter. |
| `icm/fixtures/04_kron/alpha_chol.npy` | 768 | `a0cfcfc93634a590` | `bitwise` | FLOAT64 reference. logdet abs diff <= 1e-9; solve max abs diff <= 1e-10; \|\|Sigma alpha - r\|\|_inf <= 1e-9; eigvalsh(Sigma_dense).min() > 0 and cholesky succeeds WITHOUT jitter. |
| `icm/fixtures/04_kron/alpha_kron.npy` | 768 | `5128da294fe30e92` | `bitwise` | FLOAT64 reference. logdet abs diff <= 1e-9; solve max abs diff <= 1e-10; \|\|Sigma alpha - r\|\|_inf <= 1e-9; eigvalsh(Sigma_dense).min() > 0 and cholesky succeeds WITHOUT jitter. |
| `icm/fixtures/04_kron/raw_params.npy` | 200 | `2f01024b977bfe62` | `bitwise` | FLOAT64 reference. logdet abs diff <= 1e-9; solve max abs diff <= 1e-10; \|\|Sigma alpha - r\|\|_inf <= 1e-9; eigvalsh(Sigma_dense).min() > 0 and cholesky succeeds WITHOUT jitter. |
| `icm/fixtures/05_posterior.json` | 778 | `25bfddbc5954aaec` | `bitwise` | FLOAT64 reference: reproduce each array to <= 1e-9 absolute. NOTE this gate is only meetable by a float64 port; 11_precision.json measures the shipped float32 path at 5.20e-08 abs on the mean, ~50x this gate, and licenses 1e-5 relative for a float32 implementation. State which dtype you are testing. |
| `icm/fixtures/05_posterior/cov_latent_reference.npy` | 51,328 | `64d5c99244c9a61e` | `bitwise` | FLOAT64 reference: reproduce each array to <= 1e-9 absolute. NOTE this gate is only meetable by a float64 port; 11_precision.json measures the shipped float32 path at 5.20e-08 abs on the mean, ~50x this gate, and licenses 1e-5 relative for a float32 implementation. State which dtype you are testing. |
| `icm/fixtures/05_posterior/mean_reference.npy` | 768 | `9ee8087cee1b685a` | `bitwise` | FLOAT64 reference: reproduce each array to <= 1e-9 absolute. NOTE this gate is only meetable by a float64 port; 11_precision.json measures the shipped float32 path at 5.20e-08 abs on the mean, ~50x this gate, and licenses 1e-5 relative for a float32 implementation. State which dtype you are testing. |
| `icm/fixtures/05_posterior/var_latent_reference.npy` | 768 | `30d212de3259c315` | `bitwise` | FLOAT64 reference: reproduce each array to <= 1e-9 absolute. NOTE this gate is only meetable by a float64 port; 11_precision.json measures the shipped float32 path at 5.20e-08 abs on the mean, ~50x this gate, and licenses 1e-5 relative for a float32 implementation. State which dtype you are testing. |
| `icm/fixtures/05_posterior/var_observed_reference.npy` | 768 | `d9150025bfe26f69` | `bitwise` | FLOAT64 reference: reproduce each array to <= 1e-9 absolute. NOTE this gate is only meetable by a float64 port; 11_precision.json measures the shipped float32 path at 5.20e-08 abs on the mean, ~50x this gate, and licenses 1e-5 relative for a float32 implementation. State which dtype you are testing. |
| `icm/fixtures/06_gradients.json` | 1,485 | `049bcaed284a747c` | `bitwise` | Analytic vs autograd max abs <= 1e-13; analytic vs central FD (eps=1e-6) <= 1e-8; nll <= 1e-12. The FD test needs NO oracle and survives torch removal -- keep it. |
| `icm/fixtures/06_gradients/grad_analytic.npy` | 200 | `8629e8927bb7ba1c` | `bitwise` | Analytic vs autograd max abs <= 1e-13; analytic vs central FD (eps=1e-6) <= 1e-8; nll <= 1e-12. The FD test needs NO oracle and survives torch removal -- keep it. |
| `icm/fixtures/06_gradients/grad_autograd.npy` | 200 | `1819224a5d588b64` | `bitwise` | Analytic vs autograd max abs <= 1e-13; analytic vs central FD (eps=1e-6) <= 1e-8; nll <= 1e-12. The FD test needs NO oracle and survives torch removal -- keep it. |
| `icm/fixtures/06_gradients/grad_finite_difference.npy` | 200 | `261cf3c8dc0fe06b` | `bitwise` | Analytic vs autograd max abs <= 1e-13; analytic vs central FD (eps=1e-6) <= 1e-8; nll <= 1e-12. The FD test needs NO oracle and survives torch removal -- keep it. |
| `icm/fixtures/06_gradients/theta_raw.npy` | 200 | `2f01024b977bfe62` | `bitwise` | Analytic vs autograd max abs <= 1e-13; analytic vs central FD (eps=1e-6) <= 1e-8; nll <= 1e-12. The FD test needs NO oracle and survives torch removal -- keep it. |
| `icm/fixtures/07_identifiability.json` | 24,641 | `fef60b8e607990e4` | `stochastic` | The invariance_proof block IS bitwise (max\|B-B_WQ\| <= 1e-12; all five continuous-family t values rebuild B to <= 1e-12). The fitted blocks are optimizer traces: compare SPREADS to within a factor of 3, NEVER individual values. The reference MLE is the sharp target: nll 0.871824281 +/- 1e-4, task correlation 1.0 +/- 1e-4. |
| `icm/fixtures/08_task_correlation.json` | 1,154 | `22501277a5331b88` | `stochastic` | Schedule-matched: \|mean rho - 0.99378\| <= 1e-2 and spread <= 3e-2 when both sides run 200 Adam steps at lr=0.1 from a randn init. MLE-matched: \|rho - 1.0\| <= 1e-4 when driven to \|\|grad\|\|_inf < 1e-4. NEVER compare W, raw_var or raw_task_noises. |
| `icm/fixtures/09_acquisition_gap.json` | 2,847 | `e4fa794278089682` | `stochastic` | diag_embed_lines must be exactly [323] and the acquisition.py source window 306-330 must match byte for byte (this fixture is EXPECTED to change if acquisition.py changes). The measured_on_fitted_icm correlations reproduce to +/- 0.02. |
| `icm/fixtures/10_scaling.json` | 3,287 | `553b32d8c83b1c9c` | `stochastic` | WALL CLOCK -- do NOT byte-compare. VERIFIED this session: it is the one fixture that genuinely differs between fresh interpreters. Reproduce only the ORDER OF MAGNITUDE of per_step_speedup: ~7-8x at N=40, 60-75x at N=100, 117-138x at N=290, 184-214x at N=600. |
| `icm/fixtures/11_precision.json` | 1,009 | `784d5bc3413a2392` | `stochastic` | Reproduce each *_rel_diff to within a factor of 10; the verdict strings must not change. Licenses 1e-5 relative on the posterior for a float32 port (two orders of headroom over the measured 1.4e-07). |
| `icm/fixtures/MANIFEST.json` | 11,358 | `1686c5009583c884` | `stochastic` | Index of all 36 fixtures with per-file determinism and criterion. Itself changes whenever a stochastic fixture does -- VERIFIED this session: it differed only in the 10_scaling.json entry. |

### `spec/hv/` — Hypervolume indicator, Pareto dominance, and the normalized objective frame

Capture: `capture_hv.py` · 19 files · 777,585 bytes (fixtures/artifacts 614,603 bytes)

| path | bytes | sha256[:16] | determinism | agreement criterion |
|---|---:|---|---|---|
| `hv/SPEC.md` | 66,845 | `57567b54e96cd3aa` | `n/a` | The specification. Not a fixture; hashed here so drift in a spec is detectable alongside drift in its fixtures. |
| `hv/capture_hv.py` | 79,757 | `4a273c6368a027cf` | `n/a` | The capture script that generates this cluster's fixtures. Re-run it in a fresh interpreter and byte-compare to detect drift. |
| `hv/capture_log.txt` | 16,380 | `aee69648d188c90e` | `n/a` | Transcript of one capture run; not a comparison target. |
| `hv/fixtures/archived_hypervolume.json` | 203,368 | `07f5114a850a6a6a` | `bitwise` | Scalar HV <= 4 ulp; front SIZES exact integers. Archived residual max 1.110e-16, mean 8.983e-18; pareto_size matched 448/448; 0 non-monotone steps; 256 pooled front members with a zero coordinate. VERIFIED this session by an INDEPENDENT route (spec/boxdecomp Lacour cells + the volume-complement identity): max residual 3.053e-16, mean 8.13e-17, front sizes 3/62, zero-coordinate members 256, non-monotone 0. My 3.05e-16 is a third algorithm and sits inside the 4-ulp gate -- evidence the gate is correctly calibrated, not that the fixture is wrong. |
| `hv/fixtures/bound_roundtrip_edge.json` | 679 | `879503172b76b0b8` | `bitwise` | Exact hex equality on every field. Proves the clip is load-bearing and must come BEFORE the strict reference filter. Do NOT 'fix' it by widening the bound or adding an epsilon. |
| `hv/fixtures/determinism.json` | 829 | `949e07ff8b7a3330` | `bitwise` | Every bitwise flag must be true: repeat call, row reversal, 25/25 permutations, 252/252 random-bank permutations, worst \|d\| 0.0. The float32 field is a WARNING, not a target: a float32 round-trip of the inputs moved HV by 3.184e-09, seven orders worse than the algorithm-portability error. The float32 GP decision must stop at the GP. |
| `hv/fixtures/environment.json` | 273 | `111e1f16b7452a50` | `stochastic` | NEVER compared; excluded from the manifest digest (it records elapsed seconds). VERIFIED this session: differs between runs. |
| `hv/fixtures/frame.json` | 1,266 | `bcbe77fc102068b8` | `bitwise` | Exact string/hex equality on every field. bounds_payload_sha256 must be asserted at load with no tolerance. VERIFIED this session: the payload hash reproduces from the frozen artifact. |
| `hv/fixtures/hv_160_replication.json` | 46,317 | `03ad316c02aed937` | `bitwise` | botorch == GP-MOBO must be bitwise 160/160 (a transliteration-fidelity check on the untouched external baseline). The candidate is held to <= 4 ulp: 71/160 bitwise, worst 2.220e-16. Treating 160/160 as evidence of portability is the error this fixture exists to prevent. |
| `hv/fixtures/hv_canonical_cases.json` | 13,410 | `5f0431b10e18d7a7` | `bitwise` | \|HV_new - hv_candidate\| <= 4 ulp = 4*2^-52*max(1,\|HV\|). Candidate == botorch bitwise 18/18; candidate vs inclusion-exclusion worst 7.105e-15 (cancellation on the IE side). DO NOT tighten to bitwise across algorithms. |
| `hv/fixtures/hv_random_fronts.json` | 264,550 | `ad250cbfa25629ec` | `bitwise` | \|HV_new - hv_candidate\| <= 4 ulp. vs botorch 171/252 bitwise, worst 4.441e-16. The coarse_grid family is bitwise 36/36 at every m -- use it as a hard bitwise sub-check that isolates algorithmic error from rounding. |
| `hv/fixtures/independent_checks.json` | 20,728 | `07fb083cc4bf8ef6` | `stochastic` | MIXED. (a) Monte Carlo is an ESTIMATOR: compare with \|z\| <= 4, never bitwise (measured z in [-1.80,-0.33]). (b) axis permutation, (c) dominated-point invariance, (d) monotonicity are exact and deterministic: 4/10 and 58/81 bitwise with worst <= 4 ulp; 160/160 strict increases, smallest 0.017743700248506022; adding a dominated point bitwise neutral 160/160 (exact). |
| `hv/fixtures/manifest.json` | 2,168 | `9cfd6df43296482c` | `bitwise` | sha256 + byte size of the 13 deterministic fixtures plus manifest_digest 49f3d877...0ffd. timings.json and environment.json are excluded by design. If the digest moves, PRUNE_FROM_DIM or a sort key changed and every downstream tolerance must be re-derived. |
| `hv/fixtures/normalized_frames.json` | 37,079 | `a536c18d54ccca12` | `bitwise` | Y_normalized must match BITWISE (no tolerance: a tolerance would hide a swapped sign or transposed bound). Algebraically equivalent rearrangements such as (y-lo)*(1/span) are NOT bitwise equivalent -- VERIFIED this session: that form differs on 498 of 1570 cells, worst 1.110e-16. CAVEAT FOUND THIS SESSION: the fixture additionally pre-masks every row that is not fully observed to NaN across ALL FIVE columns (capture_hv.py:1024-1025), which SPEC section 4.2's elementwise formula does not do. Applying section 4.2 alone leaves finite ADMET values in those rows and fails on 24 of 1570 cells. Apply the whole-row not-fully-observed NaN mask before comparing. |
| `hv/fixtures/output_column_contract.json` | 2,499 | `c1ff9a4e0c1e336c` | `bitwise` | Header lists: exact ordered string equality. Selectivity_Index == hDHFR_Docking - PfDHFR_Docking bitwise 89/89; the implied heavy-atom denominator deviates from an integer by at most 7.105e-15. |
| `hv/fixtures/pareto_convention_cases.json` | 14,548 | `05af74b167ad4391` | `bitwise` | EXACT boolean equality, no tolerance. Match mask_candidate_keep_duplicates for REPORTED front membership and mask_candidate_dedup internally. Disagreements with the candidate weak convention out of 13: botorch dedup=True 4, dedup=False 1, GP-MOBO 2/2, repo 2 (both NaN cases). |
| `hv/fixtures/ref_filter_strict_vs_weak.json` | 4,515 | `354097098674c62d` | `bitwise` | Row counts exact integers; HV values <= 4 ulp. Weak (>=) admitted 10 extra rows pooled and changed the value in 0 cases. The decisive number is sum_of_UNCLAMPED_box_volumes = 2.0 for y=(-1,-1,2), r=0 against a true HV of 0.0. |
| `hv/fixtures/repo_oracle_agreement.json` | 548 | `cf2b7e1937930d0d` | `bitwise` | normalize: 20305/20305 cells BITWISE required exactly. compute_hypervolume: 30/53 runs bitwise, worst 5.551e-17 -- held to 4 ulp only (different exact algorithm). Pareto masks 53/53 identical, required exactly. |
| `hv/fixtures/timings.json` | 1,826 | `f9bf8d1c5d9410a2` | `stochastic` | NEVER compared; excluded from the manifest digest. Order-of-magnitude budget only. VERIFIED this session: differs between runs. |

### `spec/boxdecomp/` — Non-dominated box decomposition (Lacour 2017 local upper bounds, applied twice)

Capture: `capture_boxdecomp.py, mutation_study.py` · 9 files · 3,470,392 bytes (fixtures/artifacts 3,335,409 bytes)

| path | bytes | sha256[:16] | determinism | agreement criterion |
|---|---:|---|---|---|
| `boxdecomp/SPEC.md` | 69,988 | `346b8d5fe0b94b02` | `n/a` | The specification. Not a fixture; hashed here so drift in a spec is detectable alongside drift in its fixtures. |
| `boxdecomp/capture_boxdecomp.py` | 38,742 | `9ae0c3a02abe2734` | `n/a` | The capture script that generates this cluster's fixtures. Re-run it in a fresh interpreter and byte-compare to detect drift. |
| `boxdecomp/fixtures/arrays.bin` | 2,709,800 | `993f4351bb02a4a9` | `bitwise` | GATE T1 (the only test that maps onto what the acquisition consumes): compute cells from front_in_oracle_order and ref, evaluate the exact non-log MC EHVI per sample, and require max abs diff <= 1e-12 AND max rel diff <= 1e-9 against ehvi_per_sample. (Reference agrees with botorch to 1.11e-16 abs / 4.33e-16 rel over all 161 cases; the permutation study bounds the partition-choice contribution at 2.2e-16.) If your front arrives float32, loosen to 1e-7 relative. SECONDARY T4: byte-compare canonically-sorted cell rows against cells_canonical -- valid ONLY under the exact insertion and append order of SPEC 5.1; reproduces botorch row for row on all 119 m>=3 cases, and the 40 non-empty m=2 cases are EXPECTED to differ (deviation D2). A T4 failure with T1/T2 green means a different valid partition, not a bug. Raw little-endian float64, C order, header-free -- index it with manifest.json. |
| `boxdecomp/fixtures/manifest.json` | 593,384 | `ce4439941f54bd2a` | `bitwise` | GATE T2 (pointwise membership -- catches individually-wrong cells and cannot false-alarm on a valid-but-different partition): 20000 uniform query points in [ref,1]^m, half-open membership lower <= q < upper, require n_overlapping == 0, n_nondominated_uncovered == 0, n_dominated_covered == 0. Totals over 161 cases x 20000 probes x 2 decompositions = 6.44M probes are 0/0/0. T3 (total volume, cap upper at 1.0): \|sum_cells vol - (prod_j(1-r_j) - HV)\| <= 1e-12 absolute (worst stored 3.331e-16). T5 (bound provenance, intrinsic, no tolerance): every cell bound bit-identical to a front coordinate, the reference coordinate, or +inf -- 0 violations. Structural: upper[:,0]==+inf for every cell; at m=2 the cell count is exactly max(\|P\|+1,1). |
| `boxdecomp/fixtures/measurements.json` | 24,879 | `1c217776ddc679af` | `bitwise` | Evidence base for SPEC's claims and for T1/T3's tolerances, not a numeric target. The one number to carry into the test suite is the permutation study's max_abs_ehvi_deviation <= 2.2e-16 -- the floor below which no tolerance can meaningfully be set. |
| `boxdecomp/fixtures/mutations.json` | 5,079 | `f99c3060a57a3315` | `bitwise` | Validates the TEST SUITE, not the implementation. Run mutation_study.py with your own cells() as the control: the control row must be 0/16 on all five tests and every real bug must be caught by T2. sha256 must be f99c3060a57a3315a8155e16a03430f4a0a4c0000e929deaf3c4ac545beb2fe7. VERIFIED this session: byte-identical, sha matches, and the full firing pattern reproduced (including step4_nonstrict diverged=3 and no_empty_filter firing T4 only). |
| `boxdecomp/fixtures/timings.json` | 2,267 | `48ba7573e37a3b45` | `stochastic` | Do NOT byte-compare or assert the seconds / traced megabytes. The CELL COUNTS in this file ARE deterministic (43/181/577/1649/2861/3936) and may be asserted. VERIFIED this session: the file differs between runs. |
| `boxdecomp/mutation_study.py` | 13,709 | `b7f70f046d4a0422` | `n/a` | Bug-injection study that validates the boxdecomp test suite. |
| `boxdecomp/reference_lacour.py` | 12,544 | `b9e8c16cee599e99` | `n/a` | Paper-derived numpy decomposition. Used this session as the independent decomposition in the qNEHVI and hypervolume spot-checks. |

### `spec/qnehvi/` — q=1 exact non-log qNEHVI estimator + the determinism / RNG policy

Capture: `capture_qnehvi.py, capture_probes.py, check_decomposition_contract.py` · 49 files · 3,397,039 bytes (fixtures/artifacts 3,198,350 bytes)

| path | bytes | sha256[:16] | determinism | agreement criterion |
|---|---:|---|---|---|
| `qnehvi/SPEC.md` | 112,063 | `f62cc44b0856e01e` | `n/a` | The specification. Not a fixture; hashed here so drift in a spec is detectable alongside drift in its fixtures. |
| `qnehvi/capture_probes.py` | 12,299 | `0e5e95fd39e3f861` | `n/a` | The capture script that generates this cluster's fixtures. Re-run it in a fresh interpreter and byte-compare to detect drift. |
| `qnehvi/capture_qnehvi.py` | 70,185 | `aa4b6578d6798a18` | `n/a` | The capture script that generates this cluster's fixtures. Re-run it in a fresh interpreter and byte-compare to detect drift. |
| `qnehvi/check_decomposition_contract.py` | 4,142 | `9b898f2e4f8116cf` | `n/a` | Executable interface contract between spec/qnehvi and spec/boxdecomp. |
| `qnehvi/fixtures/capture_runtime.txt` | 23 | `bfd9fad461e73b84` | `stochastic` | Wall clock; never compared. |
| `qnehvi/fixtures/decomposition_contract.json` | 858 | `8fc0ab80f3b72579` | `bitwise` | Re-run check_decomposition_contract.py: byte-identical, sha256 8fc0ab80f3b725790a74b94aad312e1bdfa0bfa4a9f9e52ea221824fef28cc01. The same script MUST also pass against spec/boxdecomp's numpy decomposition. VERIFIED this session: pointed at spec/boxdecomp/reference_lacour.nondominated_cells it produced a BYTE-IDENTICAL file with the same sha256, including K_min=1, K_max=328 and n_infinite_upper_bound_entries=4378. The cluster's unresolved item 4 is therefore CLOSED by measurement. |
| `qnehvi/fixtures/decomposition_timings.txt` | 161 | `1f6deb6b16f224a6` | `stochastic` | Wall clock; never compared. |
| `qnehvi/fixtures/probe_runtime.txt` | 126 | `c835b97f13c14420` | `stochastic` | Wall clock; never compared. VERIFIED this session: differs between runs. |
| `qnehvi/fixtures/qnehvi_capture.json` | 66,447 | `27362fb089e2b94c` | `bitwise` | Re-run capture_qnehvi.py in a fresh interpreter: byte-identical, sha256 27362fb089e2b94c72060547c9a4983871d6951ed00d0a992627414166e0f5bd. For a NEW implementation the per-key criteria differ: reference_vs_botorch_same_normals and cells_vs_inclusion_exclusion require max\|d\| <= 1e-12 absolute; rng_policy must match EXACTLY; numpy_chunk_invariance bitwise True at every chunk size; real_library_pool.lib_admet_cols_resolved_by_name exactly [2,0,1]; legacy_main_fixture_ties integer tie/zero counts exact with 1e-6 relative on scores; mc_standard_error and iid_vs_sobol reproduce bitwise IFF the specified generator, draw shape and CRN layout are used. DO NOT reproduce chunk_sweep.real_gp or chunk_sweep.fixed_posterior_no_gp -- they record the shipped code's float artifacts; reproduce the PATTERN (25==64==128; 13/7/1 differ at 1e-9..1e-7 with the GP and ~3.5e-18 without). current_impl_log_qnehvi records superseded behaviour and is not a target. |
| `qnehvi/fixtures/qnehvi_probes.json` | 6,031 | `d7093ad39853c750` | `bitwise` | Re-run capture_probes.py: byte-identical, sha256 d7093ad39853c7508d7da1d7333200275610d79b7b74aa89f282422df4b69c47. VERIFIED this session: byte-identical, sha matches. numpy_chunk_invariance bitwise True everywhere; reduction_order bitwise for Mc>=2 and NOT equal at Mc==1; gp_call_shape tanimoto_gram diffs 0.0 while predict's are float32-eps scale. |
| `qnehvi/fixtures/rank200_Xb.npy` | 24,704 | `713104d9471a0b8b` | `bitwise` | The 12-baseline / 200-candidate problem behind every MC standard-error measurement. Byte-identical across reruns. An implementation using the specified generator and CRN layout must reproduce mc_standard_error and the probes' crn_vs_ind_se / diverse_top20_stability bitwise. |
| `qnehvi/fixtures/rank200_Xc.npy` | 409,728 | `d0354b06857fee0c` | `bitwise` | The 12-baseline / 200-candidate problem behind every MC standard-error measurement. Byte-identical across reruns. An implementation using the specified generator and CRN layout must reproduce mc_standard_error and the probes' crn_vs_ind_se / diverse_top20_stability bitwise. |
| `qnehvi/fixtures/rank200_Yd.npy` | 320 | `7867fd63a1e37c16` | `bitwise` | The 12-baseline / 200-candidate problem behind every MC standard-error measurement. Byte-identical across reruns. An implementation using the specified generator and CRN layout must reproduce mc_standard_error and the probes' crn_vs_ind_se / diverse_top20_stability bitwise. |
| `qnehvi/fixtures/rank200_admet_b.npy` | 416 | `3cc7c7c2f144edc4` | `bitwise` | The 12-baseline / 200-candidate problem behind every MC standard-error measurement. Byte-identical across reruns. An implementation using the specified generator and CRN layout must reproduce mc_standard_error and the probes' crn_vs_ind_se / diverse_top20_stability bitwise. |
| `qnehvi/fixtures/rank200_admet_c.npy` | 4,928 | `bf8258727df2b34f` | `bitwise` | The 12-baseline / 200-candidate problem behind every MC standard-error measurement. Byte-identical across reruns. An implementation using the specified generator and CRN layout must reproduce mc_standard_error and the probes' crn_vs_ind_se / diverse_top20_stability bitwise. |
| `qnehvi/fixtures/rank200_mean_b.npy` | 320 | `5336590e2a323cca` | `bitwise` | The 12-baseline / 200-candidate problem behind every MC standard-error measurement. Byte-identical across reruns. An implementation using the specified generator and CRN layout must reproduce mc_standard_error and the probes' crn_vs_ind_se / diverse_top20_stability bitwise. |
| `qnehvi/fixtures/rank200_mean_c.npy` | 3,328 | `980769f567b70d22` | `bitwise` | The 12-baseline / 200-candidate problem behind every MC standard-error measurement. Byte-identical across reruns. An implementation using the specified generator and CRN layout must reproduce mc_standard_error and the probes' crn_vs_ind_se / diverse_top20_stability bitwise. |
| `qnehvi/fixtures/rank200_var_b.npy` | 320 | `e2f80cf6e2a541e1` | `bitwise` | The 12-baseline / 200-candidate problem behind every MC standard-error measurement. Byte-identical across reruns. An implementation using the specified generator and CRN layout must reproduce mc_standard_error and the probes' crn_vs_ind_se / diverse_top20_stability bitwise. |
| `qnehvi/fixtures/rank200_var_c.npy` | 3,328 | `f8c733ac0c99198d` | `bitwise` | The 12-baseline / 200-candidate problem behind every MC standard-error measurement. Byte-identical across reruns. An implementation using the specified generator and CRN layout must reproduce mc_standard_error and the probes' crn_vs_ind_se / diverse_top20_stability bitwise. |
| `qnehvi/fixtures/realpool_Xb.npy` | 32,896 | `e0c9de8746355160` | `bitwise` | The real-library pool of SPEC section 11. Byte-identical across reruns PROVIDED data/library/ is the same cached library (library_n_molecules_raw = 29678, lib_admet_cols = [2,0,1]); if either differs the library was rebuilt and every number in section 11 must be RE-MEASURED, not compared. Read the three cached files DIRECTLY -- do not call data.load_library(), which rewrites data/library/quality_rejected.csv as a side effect. |
| `qnehvi/fixtures/realpool_Yd.npy` | 384 | `01bc03d0e705d47c` | `bitwise` | The real-library pool of SPEC section 11. Byte-identical across reruns PROVIDED data/library/ is the same cached library (library_n_molecules_raw = 29678, lib_admet_cols = [2,0,1]); if either differs the library was rebuilt and every number in section 11 must be RE-MEASURED, not compared. Read the three cached files DIRECTLY -- do not call data.load_library(), which rewrites data/library/quality_rejected.csv as a side effect. |
| `qnehvi/fixtures/realpool_admet_b.npy` | 512 | `fa25c566c60ee662` | `bitwise` | The real-library pool of SPEC section 11. Byte-identical across reruns PROVIDED data/library/ is the same cached library (library_n_molecules_raw = 29678, lib_admet_cols = [2,0,1]); if either differs the library was rebuilt and every number in section 11 must be RE-MEASURED, not compared. Read the three cached files DIRECTLY -- do not call data.load_library(), which rewrites data/library/quality_rejected.csv as a side effect. |
| `qnehvi/fixtures/realpool_analog_Xc.npy` | 1,228,928 | `49cf9fb4214d6d3e` | `bitwise` | The real-library pool of SPEC section 11. Byte-identical across reruns PROVIDED data/library/ is the same cached library (library_n_molecules_raw = 29678, lib_admet_cols = [2,0,1]); if either differs the library was rebuilt and every number in section 11 must be RE-MEASURED, not compared. Read the three cached files DIRECTLY -- do not call data.load_library(), which rewrites data/library/quality_rejected.csv as a side effect. |
| `qnehvi/fixtures/realpool_analog_admet_c.npy` | 14,528 | `fb2e5174fa93f6cb` | `bitwise` | The real-library pool of SPEC section 11. Byte-identical across reruns PROVIDED data/library/ is the same cached library (library_n_molecules_raw = 29678, lib_admet_cols = [2,0,1]); if either differs the library was rebuilt and every number in section 11 must be RE-MEASURED, not compared. Read the three cached files DIRECTLY -- do not call data.load_library(), which rewrites data/library/quality_rejected.csv as a side effect. |
| `qnehvi/fixtures/realpool_analog_lib_row_idx.npy` | 4,928 | `a220d02ffadfd956` | `bitwise` | The real-library pool of SPEC section 11. Byte-identical across reruns PROVIDED data/library/ is the same cached library (library_n_molecules_raw = 29678, lib_admet_cols = [2,0,1]); if either differs the library was rebuilt and every number in section 11 must be RE-MEASURED, not compared. Read the three cached files DIRECTLY -- do not call data.load_library(), which rewrites data/library/quality_rejected.csv as a side effect. |
| `qnehvi/fixtures/realpool_analog_mean_c.npy` | 9,728 | `23031ddb7ffa5e5c` | `bitwise` | The real-library pool of SPEC section 11. Byte-identical across reruns PROVIDED data/library/ is the same cached library (library_n_molecules_raw = 29678, lib_admet_cols = [2,0,1]); if either differs the library was rebuilt and every number in section 11 must be RE-MEASURED, not compared. Read the three cached files DIRECTLY -- do not call data.load_library(), which rewrites data/library/quality_rejected.csv as a side effect. |
| `qnehvi/fixtures/realpool_analog_var_c.npy` | 9,728 | `d659349a7c077504` | `bitwise` | The real-library pool of SPEC section 11. Byte-identical across reruns PROVIDED data/library/ is the same cached library (library_n_molecules_raw = 29678, lib_admet_cols = [2,0,1]); if either differs the library was rebuilt and every number in section 11 must be RE-MEASURED, not compared. Read the three cached files DIRECTLY -- do not call data.load_library(), which rewrites data/library/quality_rejected.csv as a side effect. |
| `qnehvi/fixtures/realpool_lib_row_idx_baseline.npy` | 256 | `7923257d3cd0a6a7` | `bitwise` | The real-library pool of SPEC section 11. Byte-identical across reruns PROVIDED data/library/ is the same cached library (library_n_molecules_raw = 29678, lib_admet_cols = [2,0,1]); if either differs the library was rebuilt and every number in section 11 must be RE-MEASURED, not compared. Read the three cached files DIRECTLY -- do not call data.load_library(), which rewrites data/library/quality_rejected.csv as a side effect. |
| `qnehvi/fixtures/realpool_mean_b.npy` | 384 | `c354313e01d18111` | `bitwise` | The real-library pool of SPEC section 11. Byte-identical across reruns PROVIDED data/library/ is the same cached library (library_n_molecules_raw = 29678, lib_admet_cols = [2,0,1]); if either differs the library was rebuilt and every number in section 11 must be RE-MEASURED, not compared. Read the three cached files DIRECTLY -- do not call data.load_library(), which rewrites data/library/quality_rejected.csv as a side effect. |
| `qnehvi/fixtures/realpool_random_Xc.npy` | 1,228,928 | `0f9fb4d1c800f059` | `bitwise` | The real-library pool of SPEC section 11. Byte-identical across reruns PROVIDED data/library/ is the same cached library (library_n_molecules_raw = 29678, lib_admet_cols = [2,0,1]); if either differs the library was rebuilt and every number in section 11 must be RE-MEASURED, not compared. Read the three cached files DIRECTLY -- do not call data.load_library(), which rewrites data/library/quality_rejected.csv as a side effect. |
| `qnehvi/fixtures/realpool_random_admet_c.npy` | 14,528 | `e7c4b5c471104710` | `bitwise` | The real-library pool of SPEC section 11. Byte-identical across reruns PROVIDED data/library/ is the same cached library (library_n_molecules_raw = 29678, lib_admet_cols = [2,0,1]); if either differs the library was rebuilt and every number in section 11 must be RE-MEASURED, not compared. Read the three cached files DIRECTLY -- do not call data.load_library(), which rewrites data/library/quality_rejected.csv as a side effect. |
| `qnehvi/fixtures/realpool_random_lib_row_idx.npy` | 4,928 | `8756ea7aea100d14` | `bitwise` | The real-library pool of SPEC section 11. Byte-identical across reruns PROVIDED data/library/ is the same cached library (library_n_molecules_raw = 29678, lib_admet_cols = [2,0,1]); if either differs the library was rebuilt and every number in section 11 must be RE-MEASURED, not compared. Read the three cached files DIRECTLY -- do not call data.load_library(), which rewrites data/library/quality_rejected.csv as a side effect. |
| `qnehvi/fixtures/realpool_random_mean_c.npy` | 9,728 | `d40c9c4c2163a375` | `bitwise` | The real-library pool of SPEC section 11. Byte-identical across reruns PROVIDED data/library/ is the same cached library (library_n_molecules_raw = 29678, lib_admet_cols = [2,0,1]); if either differs the library was rebuilt and every number in section 11 must be RE-MEASURED, not compared. Read the three cached files DIRECTLY -- do not call data.load_library(), which rewrites data/library/quality_rejected.csv as a side effect. |
| `qnehvi/fixtures/realpool_random_var_c.npy` | 9,728 | `b6ea08aba00de50c` | `bitwise` | The real-library pool of SPEC section 11. Byte-identical across reruns PROVIDED data/library/ is the same cached library (library_n_molecules_raw = 29678, lib_admet_cols = [2,0,1]); if either differs the library was rebuilt and every number in section 11 must be RE-MEASURED, not compared. Read the three cached files DIRECTLY -- do not call data.load_library(), which rewrites data/library/quality_rejected.csv as a side effect. |
| `qnehvi/fixtures/realpool_var_b.npy` | 384 | `3664bc52c12dffa2` | `bitwise` | The real-library pool of SPEC section 11. Byte-identical across reruns PROVIDED data/library/ is the same cached library (library_n_molecules_raw = 29678, lib_admet_cols = [2,0,1]); if either differs the library was rebuilt and every number in section 11 must be RE-MEASURED, not compared. Read the three cached files DIRECTLY -- do not call data.load_library(), which rewrites data/library/quality_rejected.csv as a side effect. |
| `qnehvi/fixtures/small25_Xb.npy` | 24,704 | `e312ed2c37c4223a` | `bitwise` | The 12-baseline / 25-candidate problem: int8 fingerprints, known-exact ADMET, docking targets, bounds, signs, ref, and the fixed-hyperparameter GP's (mean,var). Byte-identical across reruns. |
| `qnehvi/fixtures/small25_Xc.npy` | 51,328 | `e04131467288a370` | `bitwise` | The 12-baseline / 25-candidate problem: int8 fingerprints, known-exact ADMET, docking targets, bounds, signs, ref, and the fixed-hyperparameter GP's (mean,var). Byte-identical across reruns. |
| `qnehvi/fixtures/small25_Yd.npy` | 320 | `2eb02b822d086301` | `bitwise` | The 12-baseline / 25-candidate problem: int8 fingerprints, known-exact ADMET, docking targets, bounds, signs, ref, and the fixed-hyperparameter GP's (mean,var). Byte-identical across reruns. |
| `qnehvi/fixtures/small25_admet_b.npy` | 416 | `e6148d81c3a525d0` | `bitwise` | The 12-baseline / 25-candidate problem: int8 fingerprints, known-exact ADMET, docking targets, bounds, signs, ref, and the fixed-hyperparameter GP's (mean,var). Byte-identical across reruns. |
| `qnehvi/fixtures/small25_admet_c.npy` | 728 | `475d3564477e405e` | `bitwise` | The 12-baseline / 25-candidate problem: int8 fingerprints, known-exact ADMET, docking targets, bounds, signs, ref, and the fixed-hyperparameter GP's (mean,var). Byte-identical across reruns. |
| `qnehvi/fixtures/small25_botorch_exact_scores_N128.npy` | 328 | `714f9f4f3b9929be` | `bitwise` | T-ID-1 target: feed small25_sobol_base_samples_N128.npy into the numpy estimator in place of its own draws and reproduce these scores to <= 1e-12 absolute (or <= 1e-10 relative on non-zero scores). VERIFIED this session with an estimator written from SPEC section 1 alone and spec/boxdecomp's NUMPY Lacour cells (no botorch in the pipeline): max abs 1.387779e-17, max rel 7.493658e-16, Spearman 1.0, identical argsort -- the identical figures the cluster reports with botorch cells. |
| `qnehvi/fixtures/small25_bounds.npy` | 208 | `fc83efb335a5d7e5` | `bitwise` | The 12-baseline / 25-candidate problem: int8 fingerprints, known-exact ADMET, docking targets, bounds, signs, ref, and the fixed-hyperparameter GP's (mean,var). Byte-identical across reruns. |
| `qnehvi/fixtures/small25_mean_b.npy` | 320 | `fb9f055f285e6ba2` | `bitwise` | The 12-baseline / 25-candidate problem: int8 fingerprints, known-exact ADMET, docking targets, bounds, signs, ref, and the fixed-hyperparameter GP's (mean,var). Byte-identical across reruns. |
| `qnehvi/fixtures/small25_mean_c.npy` | 528 | `b625170bc42f1db9` | `bitwise` | The 12-baseline / 25-candidate problem: int8 fingerprints, known-exact ADMET, docking targets, bounds, signs, ref, and the fixed-hyperparameter GP's (mean,var). Byte-identical across reruns. |
| `qnehvi/fixtures/small25_ref.npy` | 168 | `f7caabe6b9fbc594` | `bitwise` | The 12-baseline / 25-candidate problem: int8 fingerprints, known-exact ADMET, docking targets, bounds, signs, ref, and the fixed-hyperparameter GP's (mean,var). Byte-identical across reruns. |
| `qnehvi/fixtures/small25_signs.npy` | 168 | `4b0114a9aeb29040` | `bitwise` | The 12-baseline / 25-candidate problem: int8 fingerprints, known-exact ADMET, docking targets, bounds, signs, ref, and the fixed-hyperparameter GP's (mean,var). Byte-identical across reruns. |
| `qnehvi/fixtures/small25_sobol_base_samples_N128.npy` | 26,752 | `c8eb4731d61051b5` | `bitwise` | The (128,13,2) standard normals botorch's Sobol sampler actually used (rows 0..11 baseline, row 12 the single shared candidate slot). Input to T-ID-1; needs no botorch to use. |
| `qnehvi/fixtures/small25_var_b.npy` | 320 | `3126091c42cf4bc6` | `bitwise` | The 12-baseline / 25-candidate problem: int8 fingerprints, known-exact ADMET, docking targets, bounds, signs, ref, and the fixed-hyperparameter GP's (mean,var). Byte-identical across reruns. |
| `qnehvi/fixtures/small25_var_c.npy` | 528 | `859896404619ea6b` | `bitwise` | The 12-baseline / 25-candidate problem: int8 fingerprints, known-exact ADMET, docking targets, bounds, signs, ref, and the fixed-hyperparameter GP's (mean,var). Byte-identical across reruns. |

### `spec/stats/` — Standard normal PDF/CDF, closed-form EI, Wilcoxon signed-rank, paired t-test, Spearman/Pearson

Capture: `capture_stats.py` · 13 files · 1,323,290 bytes (fixtures/artifacts 1,144,506 bytes)

| path | bytes | sha256[:16] | determinism | agreement criterion |
|---|---:|---|---|---|
| `stats/SPEC.md` | 60,376 | `8009b668c7f8e8d2` | `n/a` | The specification. Not a fixture; hashed here so drift in a spec is detectable alongside drift in its fixtures. |
| `stats/capture_stats.py` | 80,777 | `b70f9b598e31c90d` | `n/a` | The capture script that generates this cluster's fixtures. Re-run it in a fresh interpreter and byte-compare to detect drift. |
| `stats/fixtures/MANIFEST.json` | 1,250 | `011458a344959033` | `bitwise` | sha256 of the 7 byte-comparable fixtures. stats/verify_fixtures.py checks this FIRST so a stale or edited fixture fails loudly. VERIFIED this session: all 9 files in the directory, including environment.json, were byte-identical on a fresh re-run. |
| `stats/fixtures/ei.json` | 227,998 | `5e81971e75125b94` | `bitwise` | max abs dev vs ei_ref <= 1e-15; restricted to the 1103 candidates with EI >= 1e-6*max(EI), max rel <= 1e-12. Do NOT impose a whole-block relative bound (worst is 8.121e-11 at a candidate whose EI is 4.024e-130). argsort(-ei)[:10] and [:50] must match top10/top50 ELEMENT FOR ELEMENT despite 12 exactly-tied EI values. All 15 edge rows exact, NaN via isnan. Assert all(ei >= 0). |
| `stats/fixtures/environment.json` | 317 | `ba16cf3c085033af` | `provenance` | EXCLUDED from byte-compare by design (changes with the environment). Its job is to record which scipy produced the reference values -- the Wilcoxon branch selector in particular is scipy-version-dependent: the n<=13 permutation cap did not exist in older scipy. If the oracle version changes, wilcoxon.json must be recaptured and library-deviation rows 4 and 5 re-adjudicated. NOTE: it nonetheless reproduced byte-identically this session. |
| `stats/fixtures/measurements.json` | 14,523 | `59360b35b710905d` | `bitwise` | Informational provenance: every quantitative claim in stats/SPEC.md traces to a line here. If a claim is not in this file it was not measured. |
| `stats/fixtures/normal.json` | 62,914 | `f440a716cbe3ce62` | `bitwise` | Phi(x) = 0.5*math.erfc(-x/sqrt(2)) vs ndtr: max abs <= 2e-16 AND max rel <= 2e-14 on `x`. On `specials_x` the NaN pattern must match, max abs <= 2e-16, max REL <= 5e-13 -- bitwise FAILS a correct implementation in the deep left tail (x=-37.6, measured 1.747e-13 relative). phi(x)=exp(-x*x/2)/sqrt(2*pi) must be EXACTLY equal on both grids. Assert Phi(-inf)==0.0, Phi(+inf)==1.0, Phi(0.0)==0.5 exactly. |
| `stats/fixtures/properties.json` | 1,303 | `26f1cf7a0442413a` | `bitwise` | Each identity must hold at least as tightly as the deviation recorded here. These catch classes of error a value-for-value comparison against scipy cannot: an off-by-one in the SF, a non-stable rank sort, a sign error in the one-sided tails. |
| `stats/fixtures/spearman.json` | 186,451 | `171789d3c0167c7a` | `bitwise` | Spearman (Pearson on midranks) vs scipy_spearman: abs <= 1e-14. Pearson vs scipy_pearson BITWISE. midrank must equal scipy_rankdata_* exactly on all 7 edge cases. Regression test: on the 15 pairs where shortcut_spearman differs from the correct value by > 1e-6 the implementation must NOT be within 1e-12 of the shortcut. Constant input -> nan, not 0. VERIFIED this session: Spearman 2.220e-16 (claim 3.331e-16), midrank 7/7, shortcut-wrong count exactly 15, regression test passes. NOTE the BITWISE Pearson gate is FORMULATION-SPECIFIC: it holds for SPEC section 5.3's normalise-then-dot with max-scaling, and the textbook (a.b)/sqrt((a.a)(b.b)) gave me only 10/37 bitwise, worst 5.551e-16. Implement 5.3 as written. |
| `stats/fixtures/ttest.json` | 598,409 | `6e5bc1a974ef980e` | `bitwise` | betainc_reg vs scipy_betainc: rel <= 1e-11 on the t slice, <= 1e-10 on the wide grid. The two-sided p is judged against exact_two_sided_p_50dps (mpmath is the ARBITER, not scipy): rel <= 1e-11 for df <= 5000 and <= 1e-13 for df <= 50. Do NOT require better than 4e-09 against scipy_two_sided_p -- scipy's own worst error vs exact on this grid is 3.120e-09. ttest_rel: t rel <= 1e-13, p rel <= 1e-12. Degenerate rows exact, nan via isnan, inf via isinf AND sign. VERIFIED this session from SPEC section 4.4: 9.2005e-12 vs mpmath (claim 9.201e-12), 4.2631e-14 at df<=50 (claim 4.260e-14), 3.1202e-09 vs scipy (matching the stated scipy error). CRITICAL: those gates are only meetable WITH section 4.4's accurate-logarithm entry point (log_x = -log1p(t^2/nu), log_y = -log1p(nu/t^2)); passing math.log of the rounded argument instead gave me 7.98e-09, a 1000x miss. |
| `stats/fixtures/wilcoxon.json` | 51,341 | `aebd7ff5e054d1cb` | `bitwise` | The branch chosen must equal ref_branch on all 95 selector rows and the p-value must equal auto_p. Exact and permutation branches are rationals with denominator 2^n: BITWISE on exact_table, real, selector_* and boundary. Asymptotic branch: 1e-12 absolute. signed_rank_null_counts(n) must equal `counts` as exact ints with sum == 2**n and palindromic. Assert n=5 -> p == 0.0625 and n=7 -> p == 0.015625 exactly. statistic: two-sided == min(r_plus,r_minus), one-sided == r_plus, to 1e-12. d=[0.0] must return (0.0,1.0) and NOT raise. sweep_disagreements must stay empty. VERIFIED this session: exact_table reproduced for all 15 rows (sum==2**n, palindromic) and both hard-coded p-values matched exactly from an independently written generating-function recursion. |
| `stats/refstats.py` | 21,184 | `bf8f0d88a826daeb` | `n/a` | The paper/stdlib-derived reference implementations the fixtures were generated with. Section 5.3's Pearson arrangement and section 4.4's betainc live here. |
| `stats/verify_fixtures.py` | 16,447 | `6fc6232733cb0966` | `n/a` | The stats cluster's own fixture checker (checks stats/fixtures/MANIFEST.json). spec/verify_fixtures.py is the repo-wide superset. |

### `spec/admet/` — ADMET HistGradientBoosting export to numpy + Tanimoto applicability-domain gate

Capture: `capture_admet.py` · 56 files · 5,745,688 bytes (fixtures/artifacts 5,656,668 bytes)

| path | bytes | sha256[:16] | determinism | agreement criterion |
|---|---:|---|---|---|
| `admet/SPEC.md` | 43,522 | `c1f7afba4f113933` | `n/a` | The specification. Not a fixture; hashed here so drift in a spec is detectable alongside drift in its fixtures. |
| `admet/capture_admet.py` | 45,498 | `2772518c5c66c2d9` | `n/a` | The capture script that generates this cluster's fixtures. Re-run it in a fresh interpreter and byte-compare to detect drift. |
| `admet/fixtures/METADATA.json` | 40,864 | `9ced5e2f086e7ec4` | `stochastic` | NOT byte-comparable: it records wall-clock timings, so its content and its own length vary by a few bytes per run. A re-runner must EXCLUDE this file and byte-compare the other 53. VERIFIED this session: it was the ONLY one of the 54 that differed (40864 -> 40832 bytes). |
| `admet/fixtures/ad/caco2_max_tanimoto_float32.npy` | 118,840 | `1dca396f401e2946` | `bitwise` | Bitwise IF you reproduce the float32 path; not required if you use the exact-integer gate, which needs only the boolean fixtures. The float32 intersection is bitwise equal to the int32 matmul on the full product, so this file is BLAS-order-independent. |
| `admet/fixtures/ad/caco2_max_tanimoto_float64.npy` | 237,552 | `be9adbf0f855b458` | `bitwise` | Exact int32 numerator/denominator with a float64 division. |
| `admet/fixtures/ad/caco2_on_threshold_bool.npy` | 29,806 | `1e2aa3d7e1d27b29` | `bitwise` | Exact boolean equality and exact counts 222 (caco2) / 220 (half_life) / 373 (herg). Cause is exact arithmetic, not rounding: 5*intersection == union at the argmax. Additionally assert that switching the operator to <= flips exactly those counts out of domain, with the numbers hard-coded. VERIFIED this session: 222/220/373. |
| `admet/fixtures/ad/caco2_ood_bool.npy` | 29,806 | `39e8f75ca7432f3b` | `bitwise` | Exact boolean equality, ZERO tolerance. All False -- no library molecule is out of domain for any model, because data.py already dropped the out-of-domain ones at build time. Reproducible by the float32 path OR the exact-integer gate 5*inter < union (0 disagreements between them on all 29678 x 3). VERIFIED this session: exact, 0 out of domain. |
| `admet/fixtures/ad/half_life_max_tanimoto_float32.npy` | 118,840 | `c5f731193a3aed5e` | `bitwise` | Bitwise IF you reproduce the float32 path; not required if you use the exact-integer gate, which needs only the boolean fixtures. The float32 intersection is bitwise equal to the int32 matmul on the full product, so this file is BLAS-order-independent. |
| `admet/fixtures/ad/half_life_max_tanimoto_float64.npy` | 237,552 | `b4cb771bf030529d` | `bitwise` | Exact int32 numerator/denominator with a float64 division. |
| `admet/fixtures/ad/half_life_on_threshold_bool.npy` | 29,806 | `150433606fda032b` | `bitwise` | Exact boolean equality and exact counts 222 (caco2) / 220 (half_life) / 373 (herg). Cause is exact arithmetic, not rounding: 5*intersection == union at the argmax. Additionally assert that switching the operator to <= flips exactly those counts out of domain, with the numbers hard-coded. VERIFIED this session: 222/220/373. |
| `admet/fixtures/ad/half_life_ood_bool.npy` | 29,806 | `39e8f75ca7432f3b` | `bitwise` | Exact boolean equality, ZERO tolerance. All False -- no library molecule is out of domain for any model, because data.py already dropped the out-of-domain ones at build time. Reproducible by the float32 path OR the exact-integer gate 5*inter < union (0 disagreements between them on all 29678 x 3). VERIFIED this session: exact, 0 out of domain. |
| `admet/fixtures/ad/herg_max_tanimoto_float32.npy` | 118,840 | `90fa2ab810bc73a4` | `bitwise` | Bitwise IF you reproduce the float32 path; not required if you use the exact-integer gate, which needs only the boolean fixtures. The float32 intersection is bitwise equal to the int32 matmul on the full product, so this file is BLAS-order-independent. |
| `admet/fixtures/ad/herg_max_tanimoto_float64.npy` | 237,552 | `8e3ff3fe5d1c09f7` | `bitwise` | Exact int32 numerator/denominator with a float64 division. |
| `admet/fixtures/ad/herg_on_threshold_bool.npy` | 29,806 | `d17b4f1c59451de3` | `bitwise` | Exact boolean equality and exact counts 222 (caco2) / 220 (half_life) / 373 (herg). Cause is exact arithmetic, not rounding: 5*intersection == union at the argmax. Additionally assert that switching the operator to <= flips exactly those counts out of domain, with the numbers hard-coded. VERIFIED this session: 222/220/373. |
| `admet/fixtures/ad/herg_ood_bool.npy` | 29,806 | `39e8f75ca7432f3b` | `bitwise` | Exact boolean equality, ZERO tolerance. All False -- no library molecule is out of domain for any model, because data.py already dropped the out-of-domain ones at build time. Reproducible by the float32 path OR the exact-integer gate 5*inter < union (0 disagreements between them on all 29678 x 3). VERIFIED this session: exact, 0 out of domain. |
| `admet/fixtures/ad/knife_edge_union_bool.npy` | 29,806 | `29f8ec08da5c7e3c` | `bitwise` | Exact boolean equality; sum must be 766 -- the number of library molecules a rebuilt data/library/ would lose if the gate operator changed from < to <=. VERIFIED this session: 766. |
| `admet/fixtures/export/caco2_meta.json` | 1,918 | `9023a2342dbed71e` | `bitwise` | The hex scalars must round-trip bit-exactly: assert float.fromhex(baseline_prediction_hex) == the npz baseline_prediction. |
| `admet/fixtures/export/caco2_trees.npz` | 149,264 | `493bd82d6f02f592` | `bitwise` | Rebuild the flat arrays from models/pretrained_admet/<key>.joblib and assert byte equality against the sha256 in export_manifest.json. ZERO tolerance. |
| `admet/fixtures/export/export_manifest.json` | 1,988 | `8f06649c26be7eed` | `bitwise` | Read it to locate and verify every other export artifact. Not itself a numeric target. Carries the AD threshold 0.20 with rational pair [1,5] and the operator (strict <). |
| `admet/fixtures/export/half_life_meta.json` | 1,932 | `e51b169f378b9668` | `bitwise` | The hex scalars must round-trip bit-exactly: assert float.fromhex(baseline_prediction_hex) == the npz baseline_prediction. |
| `admet/fixtures/export/half_life_trees.npz` | 108,378 | `638c2b66eba42281` | `bitwise` | Rebuild the flat arrays from models/pretrained_admet/<key>.joblib and assert byte equality against the sha256 in export_manifest.json. ZERO tolerance. |
| `admet/fixtures/export/herg_meta.json` | 1,934 | `220e23ffff61bd98` | `bitwise` | The hex scalars must round-trip bit-exactly: assert float.fromhex(baseline_prediction_hex) == the npz baseline_prediction. |
| `admet/fixtures/export/herg_trees.npz` | 105,996 | `743d074ce4a3ba82` | `bitwise` | Rebuild the flat arrays from models/pretrained_admet/<key>.joblib and assert byte equality against the sha256 in export_manifest.json. ZERO tolerance. |
| `admet/fixtures/ref/caco2_leafidx_first64_int32.npy` | 25,728 | `4e267dd8901e8f02` | `bitwise` | Exact integer equality. Run this BEFORE the raw-score test when it fails: matching leaf indices with a wrong raw score means the reduction order is wrong; differing leaf indices mean the split/child indexing is wrong. VERIFIED this session: exact for all three models. |
| `admet/fixtures/ref/caco2_leafval_first64_float64.npy` | 51,328 | `6332e1aad81fd191` | `bitwise` | The leaf values at those indices; exact equality. |
| `admet/fixtures/ref/caco2_oracle_float64.npy` | 237,552 | `e0fc155d18a2e542` | `bitwise` | As admet_oracle emits the column. Identity link for caco2 -> byte-identical to raw. |
| `admet/fixtures/ref/caco2_postlink_float64.npy` | 237,552 | `e0fc155d18a2e542` | `bitwise` | Identity link for caco2 and half_life, so byte-identical to the corresponding raw file. np.array_equal. |
| `admet/fixtures/ref/caco2_raw_float64.npy` | 237,552 | `e0fc155d18a2e542` | `bitwise` | np.array_equal, ZERO tolerance. Achievable ONLY with baseline-first SEQUENTIAL accumulation over trees 0..99; np.sum(axis=1) fails on 25895 / 25776 / 24772 of 29678 rows. VERIFIED this session: a predictor written from SPEC sections 5-6 reading ONLY the .npz (no sklearn, no joblib) reproduced all three models BITWISE (max abs diff 0.0) on all 29678 molecules, and the np.sum(axis=1) failure counts and magnitudes (1.066e-14 / 2.442e-15 / 7.105e-15) reproduced exactly. |
| `admet/fixtures/ref/half_life_leafidx_first64_int32.npy` | 25,728 | `af4ddd22f95c8eba` | `bitwise` | Exact integer equality. Run this BEFORE the raw-score test when it fails: matching leaf indices with a wrong raw score means the reduction order is wrong; differing leaf indices mean the split/child indexing is wrong. VERIFIED this session: exact for all three models. |
| `admet/fixtures/ref/half_life_leafval_first64_float64.npy` | 51,328 | `2c1d42db1fd17a88` | `bitwise` | The leaf values at those indices; exact equality. |
| `admet/fixtures/ref/half_life_oracle_float64.npy` | 237,552 | `15c5805b7b9f2d8b` | `libm-dependent` | rtol=1e-15, NOT bitwise. np.power(10.0,v) is bitwise identical to 10.0**v (same ufunc) but calls libm pow, which is not guaranteed correctly rounded: np.exp(v*np.log(10)) differs by up to 7.87e-16 relative and np.exp2(v*np.log2(10)) by 6.27e-16. Bitwise on this machine; UNVERIFIED on a second platform. VERIFIED this session: bitwise here. |
| `admet/fixtures/ref/half_life_postlink_float64.npy` | 237,552 | `6dbf61417524eaee` | `bitwise` | Identity link for caco2 and half_life, so byte-identical to the corresponding raw file. np.array_equal. |
| `admet/fixtures/ref/half_life_raw_float64.npy` | 237,552 | `6dbf61417524eaee` | `bitwise` | np.array_equal, ZERO tolerance. Achievable ONLY with baseline-first SEQUENTIAL accumulation over trees 0..99; np.sum(axis=1) fails on 25895 / 25776 / 24772 of 29678 rows. VERIFIED this session: a predictor written from SPEC sections 5-6 reading ONLY the .npz (no sklearn, no joblib) reproduced all three models BITWISE (max abs diff 0.0) on all 29678 molecules, and the np.sum(axis=1) failure counts and magnitudes (1.066e-14 / 2.442e-15 / 7.105e-15) reproduced exactly. |
| `admet/fixtures/ref/herg_leafidx_first64_int32.npy` | 25,728 | `98f90a734edbc12c` | `bitwise` | Exact integer equality. Run this BEFORE the raw-score test when it fails: matching leaf indices with a wrong raw score means the reduction order is wrong; differing leaf indices mean the split/child indexing is wrong. VERIFIED this session: exact for all three models. |
| `admet/fixtures/ref/herg_leafval_first64_float64.npy` | 51,328 | `b6cb1f2c91875acf` | `bitwise` | The leaf values at those indices; exact equality. |
| `admet/fixtures/ref/herg_oracle_float64.npy` | 237,552 | `0f040e37a47b392a` | `bitwise` | As admet_oracle emits the column. Identity link for caco2 -> byte-identical to raw. |
| `admet/fixtures/ref/herg_postlink_float64.npy` | 237,552 | `0f040e37a47b392a` | `libm-dependent` | Bitwise equal to 1.0/(1.0+np.exp(-raw)) under np.errstate(over='ignore') -- 0 of 29678 differing on this machine. A test should ALSO assert that the branch-stable sigmoid DIFFERS (it differs on 4974 of 29678), to stop a future 'fix'. Cross-platform this depends on libm exp in the last ulp; fall back to rtol=1e-15 elsewhere. VERIFIED this session: bitwise here. |
| `admet/fixtures/ref/herg_raw_float64.npy` | 237,552 | `3784d7f14312c7e7` | `bitwise` | np.array_equal, ZERO tolerance. Achievable ONLY with baseline-first SEQUENTIAL accumulation over trees 0..99; np.sum(axis=1) fails on 25895 / 25776 / 24772 of 29678 rows. VERIFIED this session: a predictor written from SPEC sections 5-6 reading ONLY the .npz (no sklearn, no joblib) reproduced all three models BITWISE (max abs diff 0.0) on all 29678 molecules, and the np.sum(axis=1) failure counts and magnitudes (1.066e-14 / 2.442e-15 / 7.105e-15) reproduced exactly. |
| `admet/fixtures/synthetic/caco2_synth_X_int8.npy` | 524,416 | `de453f07795e7216` | `bitwise` | (256,2048) int8 edge-case input from default_rng(20260927) at ~3% bit density, rows 0-5 overridden (all-zero, all-ones, single bit at 0, single bit at 2047, train_features[0], fingerprints.npy[0]). Row 4 differs per model. |
| `admet/fixtures/synthetic/caco2_synth_max_tanimoto_float32.npy` | 1,152 | `1de42db4f1ede16d` | `bitwise` | Raw predictions bitwise. Measured invariants to assert: all-zero fingerprint -> max_sim exactly 0.0 and out-of-domain True; a train_features row fed back -> max_sim exactly 1.0 and in-domain; all-ones fingerprint -> 0.0522/0.0747/0.0459; 254 of 256 out of domain for every model; exact-integer and float32 gates agree on all 256. |
| `admet/fixtures/synthetic/caco2_synth_ood_bool.npy` | 384 | `f95e04e4aa8650fd` | `bitwise` | Raw predictions bitwise. Measured invariants to assert: all-zero fingerprint -> max_sim exactly 0.0 and out-of-domain True; a train_features row fed back -> max_sim exactly 1.0 and in-domain; all-ones fingerprint -> 0.0522/0.0747/0.0459; 254 of 256 out of domain for every model; exact-integer and float32 gates agree on all 256. |
| `admet/fixtures/synthetic/caco2_synth_oracle_float64.npy` | 2,176 | `344cfe335f7aa5ba` | `bitwise` | Raw predictions bitwise. Measured invariants to assert: all-zero fingerprint -> max_sim exactly 0.0 and out-of-domain True; a train_features row fed back -> max_sim exactly 1.0 and in-domain; all-ones fingerprint -> 0.0522/0.0747/0.0459; 254 of 256 out of domain for every model; exact-integer and float32 gates agree on all 256. |
| `admet/fixtures/synthetic/caco2_synth_postlink_float64.npy` | 2,176 | `344cfe335f7aa5ba` | `bitwise` | Raw predictions bitwise. Measured invariants to assert: all-zero fingerprint -> max_sim exactly 0.0 and out-of-domain True; a train_features row fed back -> max_sim exactly 1.0 and in-domain; all-ones fingerprint -> 0.0522/0.0747/0.0459; 254 of 256 out of domain for every model; exact-integer and float32 gates agree on all 256. |
| `admet/fixtures/synthetic/caco2_synth_raw_float64.npy` | 2,176 | `344cfe335f7aa5ba` | `bitwise` | Raw predictions bitwise. Measured invariants to assert: all-zero fingerprint -> max_sim exactly 0.0 and out-of-domain True; a train_features row fed back -> max_sim exactly 1.0 and in-domain; all-ones fingerprint -> 0.0522/0.0747/0.0459; 254 of 256 out of domain for every model; exact-integer and float32 gates agree on all 256. |
| `admet/fixtures/synthetic/half_life_synth_X_int8.npy` | 524,416 | `38a44b8930a00c80` | `bitwise` | (256,2048) int8 edge-case input from default_rng(20260927) at ~3% bit density, rows 0-5 overridden (all-zero, all-ones, single bit at 0, single bit at 2047, train_features[0], fingerprints.npy[0]). Row 4 differs per model. |
| `admet/fixtures/synthetic/half_life_synth_max_tanimoto_float32.npy` | 1,152 | `02802656e2a17e7c` | `bitwise` | Raw predictions bitwise. Measured invariants to assert: all-zero fingerprint -> max_sim exactly 0.0 and out-of-domain True; a train_features row fed back -> max_sim exactly 1.0 and in-domain; all-ones fingerprint -> 0.0522/0.0747/0.0459; 254 of 256 out of domain for every model; exact-integer and float32 gates agree on all 256. |
| `admet/fixtures/synthetic/half_life_synth_ood_bool.npy` | 384 | `f95e04e4aa8650fd` | `bitwise` | Raw predictions bitwise. Measured invariants to assert: all-zero fingerprint -> max_sim exactly 0.0 and out-of-domain True; a train_features row fed back -> max_sim exactly 1.0 and in-domain; all-ones fingerprint -> 0.0522/0.0747/0.0459; 254 of 256 out of domain for every model; exact-integer and float32 gates agree on all 256. |
| `admet/fixtures/synthetic/half_life_synth_oracle_float64.npy` | 2,176 | `d70df488f1b9b44c` | `bitwise` | Raw predictions bitwise. Measured invariants to assert: all-zero fingerprint -> max_sim exactly 0.0 and out-of-domain True; a train_features row fed back -> max_sim exactly 1.0 and in-domain; all-ones fingerprint -> 0.0522/0.0747/0.0459; 254 of 256 out of domain for every model; exact-integer and float32 gates agree on all 256. |
| `admet/fixtures/synthetic/half_life_synth_postlink_float64.npy` | 2,176 | `fba714d9de07b2e9` | `bitwise` | Raw predictions bitwise. Measured invariants to assert: all-zero fingerprint -> max_sim exactly 0.0 and out-of-domain True; a train_features row fed back -> max_sim exactly 1.0 and in-domain; all-ones fingerprint -> 0.0522/0.0747/0.0459; 254 of 256 out of domain for every model; exact-integer and float32 gates agree on all 256. |
| `admet/fixtures/synthetic/half_life_synth_raw_float64.npy` | 2,176 | `fba714d9de07b2e9` | `bitwise` | Raw predictions bitwise. Measured invariants to assert: all-zero fingerprint -> max_sim exactly 0.0 and out-of-domain True; a train_features row fed back -> max_sim exactly 1.0 and in-domain; all-ones fingerprint -> 0.0522/0.0747/0.0459; 254 of 256 out of domain for every model; exact-integer and float32 gates agree on all 256. |
| `admet/fixtures/synthetic/herg_synth_X_int8.npy` | 524,416 | `eb4ed0dd2eebecd1` | `bitwise` | (256,2048) int8 edge-case input from default_rng(20260927) at ~3% bit density, rows 0-5 overridden (all-zero, all-ones, single bit at 0, single bit at 2047, train_features[0], fingerprints.npy[0]). Row 4 differs per model. |
| `admet/fixtures/synthetic/herg_synth_max_tanimoto_float32.npy` | 1,152 | `5d1653e8fafce4b1` | `bitwise` | Raw predictions bitwise. Measured invariants to assert: all-zero fingerprint -> max_sim exactly 0.0 and out-of-domain True; a train_features row fed back -> max_sim exactly 1.0 and in-domain; all-ones fingerprint -> 0.0522/0.0747/0.0459; 254 of 256 out of domain for every model; exact-integer and float32 gates agree on all 256. |
| `admet/fixtures/synthetic/herg_synth_ood_bool.npy` | 384 | `f95e04e4aa8650fd` | `bitwise` | Raw predictions bitwise. Measured invariants to assert: all-zero fingerprint -> max_sim exactly 0.0 and out-of-domain True; a train_features row fed back -> max_sim exactly 1.0 and in-domain; all-ones fingerprint -> 0.0522/0.0747/0.0459; 254 of 256 out of domain for every model; exact-integer and float32 gates agree on all 256. |
| `admet/fixtures/synthetic/herg_synth_oracle_float64.npy` | 2,176 | `3f7322472b6233a6` | `bitwise` | Raw predictions bitwise. Measured invariants to assert: all-zero fingerprint -> max_sim exactly 0.0 and out-of-domain True; a train_features row fed back -> max_sim exactly 1.0 and in-domain; all-ones fingerprint -> 0.0522/0.0747/0.0459; 254 of 256 out of domain for every model; exact-integer and float32 gates agree on all 256. |
| `admet/fixtures/synthetic/herg_synth_postlink_float64.npy` | 2,176 | `3f7322472b6233a6` | `bitwise` | Raw predictions bitwise. Measured invariants to assert: all-zero fingerprint -> max_sim exactly 0.0 and out-of-domain True; a train_features row fed back -> max_sim exactly 1.0 and in-domain; all-ones fingerprint -> 0.0522/0.0747/0.0459; 254 of 256 out of domain for every model; exact-integer and float32 gates agree on all 256. |
| `admet/fixtures/synthetic/herg_synth_raw_float64.npy` | 2,176 | `f1894212a3d19186` | `bitwise` | Raw predictions bitwise. Measured invariants to assert: all-zero fingerprint -> max_sim exactly 0.0 and out-of-domain True; a train_features row fed back -> max_sim exactly 1.0 and in-domain; all-ones fingerprint -> 0.0522/0.0747/0.0459; 254 of 256 out of domain for every model; exact-integer and float32 gates agree on all 256. |

---

## 7. Files that are NOT byte-comparable

Byte-comparing any of these is a false alarm. Each row states what *is* assertable instead.

| path | why | what to assert instead |
|---|---|---|
| `admet/fixtures/METADATA.json` | records wall-clock timings and its own byte size | exclude it and byte-compare the other 53 admet fixtures; diff it key-by-key and expect only timing fields to move |
| `boxdecomp/fixtures/timings.json` | wall clock and traced memory | the CELL COUNTS in it are deterministic: 43/181/577/1649/2861/3936 |
| `gp/fixtures/stochastic_threshold.json` | measures gpytorch's stochastic-Lanczos log-det above N·T=800 | is_exact True for every N·T ≤ 800 (spread exactly 0.0) and False for every N·T > 800; the N=420/T=1 and N=800/T=1 controls exact; seeded and deterministic_probes spreads 0.0, unseeded nonzero |
| `hv/fixtures/timings.json` | wall clock | order-of-magnitude budget only |
| `hv/fixtures/environment.json` | records elapsed seconds | never compared; provenance |
| `hv/fixtures/independent_checks.json` | contains a seeded Monte-Carlo ESTIMATOR | the MC block by |z| ≤ 4; sub-blocks (b), (c), (d) are exact and deterministic |
| `icm/fixtures/10_scaling.json` | wall clock — the one fixture that genuinely differed between fresh interpreters | order of magnitude of per_step_speedup only |
| `icm/fixtures/MANIFEST.json` | indexes 10_scaling.json, so it moves when that does | every other entry; it differed in exactly the 10_scaling row here |
| `icm/fixtures/00_environment.json` | records oracle versions | not compared |
| `icm/fixtures/07_identifiability.json` | float32/float64 optimizer traces | the invariance_proof block IS bitwise; compare fitted SPREADS to a factor of 3, never individual values; the reference MLE is sharp (nll 0.871824281 ± 1e-4) |
| `icm/fixtures/08_task_correlation.json` | 200-step Adam traces from a randn init | schedule-matched |mean ρ − 0.99378| ≤ 1e-2 and spread ≤ 3e-2; MLE-matched |ρ − 1.0| ≤ 1e-4. NEVER compare W, raw_var or raw_task_noises |
| `icm/fixtures/09_acquisition_gap.json` | quotes acquisition.py by line number | diag_embed_lines == [323] and the source window byte-exact; correlations to ±0.02. EXPECTED to change if acquisition.py changes |
| `icm/fixtures/11_precision.json` | float32-vs-float64 measurement | each *_rel_diff to within a factor of 10; the verdict strings must not change |
| `qnehvi/fixtures/capture_runtime.txt` | wall clock | nothing |
| `qnehvi/fixtures/probe_runtime.txt` | wall clock | nothing |
| `qnehvi/fixtures/decomposition_timings.txt` | wall clock | nothing |
| `stats/fixtures/environment.json` | records the environment, and the Wilcoxon branch selector is scipy-version-dependent | excluded by design (it reproduced byte-identically here anyway). If scipy is upgraded, recapture wilcoxon.json and re-adjudicate library-deviation rows 4 and 5 |
| `admet/fixtures/ref/half_life_oracle_float64.npy` | libm `pow` | rtol=1e-15 off this machine; bitwise here |
| `admet/fixtures/ref/herg_postlink_float64.npy` | libm `exp` | rtol=1e-15 off this machine; bitwise here. ALSO assert the branch-stable sigmoid DIFFERS (4974 of 29678) so a future 'fix' is caught |
| `hv/capture_log.txt` | transcript of one run | nothing |

**Quantities proven NOT reproducible today, and therefore permanently disqualified as bitwise targets:**

1. **Any gpytorch loss, gradient or posterior above `N·T = 800`** — stochastic Lanczos
   quadrature on unseeded probes. Measured `is_exact = False` with nonzero spread at
   `N·T = 802, 820, 840, 1000`, and exactly those four rows of `gp/fixtures/stochastic_threshold.json`
   changed between my two runs while every row at `N·T ≤ 800` was byte-identical. `mogp.train_mogp`
   seeds nothing, so **above `N·T = 800` the shipped training run is not reproducible at all**
   (unseeded spread `3.245e-03`).
2. **All wall-clock and traced-memory measurements** — `icm/10_scaling.json`,
   `boxdecomp/timings.json`, `hv/timings.json`, `hv/environment.json` (elapsed seconds),
   the three `*runtime*.txt`/`*timings*.txt` files, and the timing fields inside
   `admet/METADATA.json`.
3. **`hv/fixtures/independent_checks.json`'s Monte-Carlo hypervolume estimator** — seeded, but
   the *quantity* is an estimator; compare with `|z| ≤ 4`.
4. **`primitives`' `minhash_collision_estimate`** — a 200,000-draw MC estimate of the Jaccard
   index; the gate is `|estimate − exact| < 4·sqrt(J(1−J)/200000)` per entry.
5. **Which jitter rung a near-singular float32 Gram lands on** — numpy and torch disagree on 3
   of 4 cases *in both directions*, so `numpy_cholesky_succeeds_unjittered`,
   `torch_cholesky_ex_info`, `ladder_chosen_jitter_f32` and `ladder_n_attempts` are
   informational and must never be asserted. Assert the ladder's **rule** instead.
6. **Where Adam lands along the `gp` model's exactly-flat `(σ², d_t)` direction** — a property
   of the optimizer path, not of the likelihood. Measured spread across inits: `7.447e-03` for
   `σ²` vs `4.170e-03` for `τ`. `shared_sigma2` and `task_noise_d` are held to `1e-4` only and
   must not be tightened; `τ_effective` is the identifiable combination.
7. **`icm`'s fitted `W`, `raw_var` and `raw_task_noises`** — non-identifiable up to an
   orthogonal `Q` (and a sign at `K=2`). Only `B = W Wᵀ + diag(softplus(v))` and the task
   correlation are comparable.
8. **Anything below the float32 noise floor.** The two legitimate accumulation orders for
   triangular substitution differ by `1.2497964974045317e-07` relative in float32, and torch's
   own float32 Adam agreement floor is `8.344650268554688e-07`, so **no float32 criterion
   tighter than about `1e-6` is meaningful anywhere in this spec set.**

**Not verified, and therefore not claimed:** cross-platform and multi-threaded-BLAS bitwise
reproducibility. Every "bitwise" label in this manifest was measured on one machine
(macOS 26.5.1 arm64, Python 3.11.15, numpy 2.4.6, torch 2.13.0, gpytorch 1.15.2, botorch
0.18.1, scikit-learn 1.9.0, scipy 1.17.1) at `OMP_NUM_THREADS=1`. The `gp` cluster could not
test multi-threaded BLAS at all because importing the repo with more than one OpenMP thread
segfaults in `mogp-drug`. The Tanimoto Gram *is* provably thread- and BLAS-independent (all
integer intermediates below `2^24`; binary float32 matmul is exact, cross-checked against the
int64 matmul at n = 4…1024); LAPACK's Cholesky and the float32 reductions are not.

---

## CHECKSUMS

`sha256  path` (paths relative to `spec/`), the format `shasum -a 256` emits.
`spec/verify_fixtures.py` parses this block. Regenerate it whenever a fixture legitimately changes.

```
c1f7afba4f1139330085cbc0f333ba16b938733d590845c44caed00406cbb5f5  admet/SPEC.md
2772518c5c66c2d9944573d1ed74fb906b2abe01a3d5dc603954fffd0140b72b  admet/capture_admet.py
9ced5e2f086e7ec462e38cb412fea1f74f8f530e0a10a17d4872ff99420faab9  admet/fixtures/METADATA.json
1dca396f401e29467809ae8be71a97be66f90313dcee6b774b96817653db8f26  admet/fixtures/ad/caco2_max_tanimoto_float32.npy
be9adbf0f855b45802fbcb9dbce8d556075769626f1b23f95ccbdef130f25d4b  admet/fixtures/ad/caco2_max_tanimoto_float64.npy
1e2aa3d7e1d27b29525426ac41442c2367d7f4fd0b42c4ee123bff90304a5de3  admet/fixtures/ad/caco2_on_threshold_bool.npy
39e8f75ca7432f3b7658091fa98cecf515f81d1248e6513a4501d7cc5520f06a  admet/fixtures/ad/caco2_ood_bool.npy
c5f731193a3aed5e7508b3eb7d2651c4909368cd7efb26e77b589295ff866501  admet/fixtures/ad/half_life_max_tanimoto_float32.npy
b4cb771bf030529d91c21994952623e1168dc8ff3025823b66c41c95bbe4b62d  admet/fixtures/ad/half_life_max_tanimoto_float64.npy
150433606fda032b87d160d707195732088c28dcab7ced2bd385c5171af7f84f  admet/fixtures/ad/half_life_on_threshold_bool.npy
39e8f75ca7432f3b7658091fa98cecf515f81d1248e6513a4501d7cc5520f06a  admet/fixtures/ad/half_life_ood_bool.npy
90fa2ab810bc73a43a5b9d8e07452f39cee1a070a2ef871c12a675c245139241  admet/fixtures/ad/herg_max_tanimoto_float32.npy
8e3ff3fe5d1c09f76b35c2c8b322b97f3d379b4f494e0a24cb41494ca4fcab5b  admet/fixtures/ad/herg_max_tanimoto_float64.npy
d17b4f1c59451de370f3e65b359e26d1bf6225f4e4cba9d0bb28955a6f26450d  admet/fixtures/ad/herg_on_threshold_bool.npy
39e8f75ca7432f3b7658091fa98cecf515f81d1248e6513a4501d7cc5520f06a  admet/fixtures/ad/herg_ood_bool.npy
29f8ec08da5c7e3c9e841782113490505fac256fd426a7723474aab650d28ab8  admet/fixtures/ad/knife_edge_union_bool.npy
9023a2342dbed71e755e75d49368e6c577d88a6fa129516545c2791364d69d12  admet/fixtures/export/caco2_meta.json
493bd82d6f02f592c5d189807aada3efca72e58f270621e1840f2c1c992475c3  admet/fixtures/export/caco2_trees.npz
8f06649c26be7eed1b56eff6105b326b982b818b976cad93dfaa744a20d51641  admet/fixtures/export/export_manifest.json
e51b169f378b9668fcab26b03c7fb970fbcac4b180e19ee2cac1f4a94e739488  admet/fixtures/export/half_life_meta.json
638c2b66eba422814db041ffa8494aede8f6165fcb11a1915849dc8bdffe6726  admet/fixtures/export/half_life_trees.npz
220e23ffff61bd982214de104b9da38a785058770b88998b722afc5907034a28  admet/fixtures/export/herg_meta.json
743d074ce4a3ba82190b169857b2046d734e2ef9ee9b25f173a1babe6d25208e  admet/fixtures/export/herg_trees.npz
4e267dd8901e8f0255ea6c3f834651c08fa5adbf880b99ba9d520570ad76b133  admet/fixtures/ref/caco2_leafidx_first64_int32.npy
6332e1aad81fd1913bd460dff3cdd43d4f800270ac61adb67ef515b58446a496  admet/fixtures/ref/caco2_leafval_first64_float64.npy
e0fc155d18a2e542d4440b564e1d88a369ac6f85556189b08c97267b9a993db5  admet/fixtures/ref/caco2_oracle_float64.npy
e0fc155d18a2e542d4440b564e1d88a369ac6f85556189b08c97267b9a993db5  admet/fixtures/ref/caco2_postlink_float64.npy
e0fc155d18a2e542d4440b564e1d88a369ac6f85556189b08c97267b9a993db5  admet/fixtures/ref/caco2_raw_float64.npy
af4ddd22f95c8ebaf213fd49c64592ae1a345be8781fa8836dc16047bce35cd1  admet/fixtures/ref/half_life_leafidx_first64_int32.npy
2c1d42db1fd17a88ab415dc3dcbf7861bcaa3b933b45a580e7817f0f6072944b  admet/fixtures/ref/half_life_leafval_first64_float64.npy
15c5805b7b9f2d8b32467ae6dd3befc63d92de5168809be76af334794da0b131  admet/fixtures/ref/half_life_oracle_float64.npy
6dbf61417524eaee7ffa37de90bb893a0a632b9d6ae8688190ad070a073c3629  admet/fixtures/ref/half_life_postlink_float64.npy
6dbf61417524eaee7ffa37de90bb893a0a632b9d6ae8688190ad070a073c3629  admet/fixtures/ref/half_life_raw_float64.npy
98f90a734edbc12c7cb1729a03c442a7071477175fcc38b66b9772fa61000dd0  admet/fixtures/ref/herg_leafidx_first64_int32.npy
b6cb1f2c91875acf6685a742e1c95abc0825aef4f8f0ccebeec9c7976304efe0  admet/fixtures/ref/herg_leafval_first64_float64.npy
0f040e37a47b392add1871c50de1e36a1c9b51bcf1bc73e81589519401f5843f  admet/fixtures/ref/herg_oracle_float64.npy
0f040e37a47b392add1871c50de1e36a1c9b51bcf1bc73e81589519401f5843f  admet/fixtures/ref/herg_postlink_float64.npy
3784d7f14312c7e708ad3887280c38336230656f5da31d9eae7c50d8d7786d82  admet/fixtures/ref/herg_raw_float64.npy
de453f07795e7216e717ccc8d749c3650ed6951d2e90c99f564a3ae7c9b04cf3  admet/fixtures/synthetic/caco2_synth_X_int8.npy
1de42db4f1ede16d0f21771a53ca69128cba69c4e332c73a1e4efd35b72dde21  admet/fixtures/synthetic/caco2_synth_max_tanimoto_float32.npy
f95e04e4aa8650fdf6b7447fc21b3b6d3a2c0ee73efb8a2d166fba795d2d01a3  admet/fixtures/synthetic/caco2_synth_ood_bool.npy
344cfe335f7aa5ba1dd4396c1b9bbb53770ef7c233928037f7865fc1cdd4de13  admet/fixtures/synthetic/caco2_synth_oracle_float64.npy
344cfe335f7aa5ba1dd4396c1b9bbb53770ef7c233928037f7865fc1cdd4de13  admet/fixtures/synthetic/caco2_synth_postlink_float64.npy
344cfe335f7aa5ba1dd4396c1b9bbb53770ef7c233928037f7865fc1cdd4de13  admet/fixtures/synthetic/caco2_synth_raw_float64.npy
38a44b8930a00c80f4b95c36cd3e82aeefd58939a37f60518cff5e30d47195cc  admet/fixtures/synthetic/half_life_synth_X_int8.npy
02802656e2a17e7cd6ad92ebc238a9739a100b4262c1a954a3f0db8e1ba870f8  admet/fixtures/synthetic/half_life_synth_max_tanimoto_float32.npy
f95e04e4aa8650fdf6b7447fc21b3b6d3a2c0ee73efb8a2d166fba795d2d01a3  admet/fixtures/synthetic/half_life_synth_ood_bool.npy
d70df488f1b9b44c52721738ac187f40f339f850c09a9dac5002f4bc7147e9a1  admet/fixtures/synthetic/half_life_synth_oracle_float64.npy
fba714d9de07b2e9fa27a0988d94d8de770fb2b239f421fee30b3b316a50a200  admet/fixtures/synthetic/half_life_synth_postlink_float64.npy
fba714d9de07b2e9fa27a0988d94d8de770fb2b239f421fee30b3b316a50a200  admet/fixtures/synthetic/half_life_synth_raw_float64.npy
eb4ed0dd2eebecd15c3742205d9f0c7d8b9583a6b3ffaeb7c6d6e1f9896bbc36  admet/fixtures/synthetic/herg_synth_X_int8.npy
5d1653e8fafce4b1847e859250cacf9619f75a6beb89837c7e23d49f00c346e5  admet/fixtures/synthetic/herg_synth_max_tanimoto_float32.npy
f95e04e4aa8650fdf6b7447fc21b3b6d3a2c0ee73efb8a2d166fba795d2d01a3  admet/fixtures/synthetic/herg_synth_ood_bool.npy
3f7322472b6233a6824d43d858328c225308224a675f396c873e8a2f38e058f0  admet/fixtures/synthetic/herg_synth_oracle_float64.npy
3f7322472b6233a6824d43d858328c225308224a675f396c873e8a2f38e058f0  admet/fixtures/synthetic/herg_synth_postlink_float64.npy
f1894212a3d191863b6415f43f655858b054b6836445aece1b9b0ae8b61296e7  admet/fixtures/synthetic/herg_synth_raw_float64.npy
cad46df38574ddc0504a90a67ea620a6a38856bddbfe0b434d0063b929ac7d26  bounds/INDEPENDENT_REVERIFICATION.md
5f3e34815a83a82296e579dd89e88358fd62c9cb416e6da4799bb1eb20f7aa22  bounds/SETTLEMENT.md
7845ebbabb2160d5567cc912ad6aad35d721ea1a8f902b043e1a23cc37ee8ec2  bounds/clipping_measurements.json
81670d22befeaa92da8980e5f9664d7d39b080b89c947771b1d2203bd61da2be  bounds/freeze_bounds.py
5fb9789e3665e0a9a8db730ebdfd003e9dda1dcff5cd80d61a8e9eb010ad03ca  bounds/measure_clipping.py
abaa99a4d5c21b62dd84b0a8227076d510f79ed8cc4dd2ecfbe4625d2244dfa6  bounds/objective_bounds.frozen.json
e5da30adfcd84c5bcba4e1dae6a7e1568a8a4aa73933f45a252f720ff00221bc  bounds/objective_bounds.frozen.json.sha256
f89662657aba24b9881a121939d6cb5aa56ec079de39f9b569fd93ada7a06c4d  bounds/reproduce_archived_hypervolume.py
85be77500dc3abe7fd966de4601dff686c4bd5e25af9f7c4c405a3423e94474f  bounds/reproduction_residuals.json
346b8d5fe0b94b021010fa9e06dc9d2da58f79605020c76917da86889db0e801  boxdecomp/SPEC.md
9ae0c3a02abe2734ca95e7aefec2b2255fe37f6ce57b0ba2482d0f21655e55f9  boxdecomp/capture_boxdecomp.py
993f4351bb02a4a97c1539b2b59d0d1380d64ebb90be520209c469ddf53a6f7b  boxdecomp/fixtures/arrays.bin
ce4439941f54bd2afa0cb4cd2d615bef2454a98ef9b1d8ce366edc3869cbae63  boxdecomp/fixtures/manifest.json
1c217776ddc679af07f10f5bb3839644c77af3a2838bd4abe60dc688a7bbd77f  boxdecomp/fixtures/measurements.json
f99c3060a57a3315a8155e16a03430f4a0a4c0000e929deaf3c4ac545beb2fe7  boxdecomp/fixtures/mutations.json
48ba7573e37a3b450f8ecebb98ff01e7a03fa00f044e91d5698eecfa6569899e  boxdecomp/fixtures/timings.json
b7f70f046d4a0422edaea303e7183f024ce9cde78d9166f6e7465182ccbb3244  boxdecomp/mutation_study.py
b9e8c16cee599e99808fb4b070766f9c7b6f0709f85778cd72e7ac9ae4026bb6  boxdecomp/reference_lacour.py
405d3ed1768353ffc33d952b124a37d25bfe1dec181d3f686de7c4b337391891  gp/SPEC.md
d4926498f54724fefebe30b4fe6f617aaff37c62d4cac7167da55acdf2be829a  gp/capture_gp.py
8a9986c6add3d7a585e3ceb6bfcad32ecf66a70737895eaf61653f476a3abc92  gp/fixtures/MANIFEST.json
78a492911cc05a3c7bb46812fb6ef16ab61b37ca78195b63f9cc8ddca64653d8  gp/fixtures/metadata.json
90272dc26c56d33bcce187d60622c1967d2990c3c1cc89ef197d47a04132301e  gp/fixtures/mll_gradients.json
b8e3227e3ca78d473aa23318e1ad0bf8dcacddd963c5c6a415c504de891e8454  gp/fixtures/model_parameters.json
9257dccd4e7f4d0a61c309c7c8769af0d7c76924fc5b88b2888eb2367bd70f61  gp/fixtures/posterior.json
cef9fd459b368e1d85e453e7cade61479fc1c0df614949fcba01145a9c533c07  gp/fixtures/standardization_nan.json
8bc687641cd268c3ad519f1fbb0a8694849ff9c4e17b6c698e4c9da18c75d129  gp/fixtures/stochastic_threshold.json
65063bdd55dd6943604d879d3cf4a10078b8cff1fd3b726d91e9b5997a390614  gp/fixtures/tanimoto_gram.json
e01f642427648f00ad5bd2de8f02c946bce0b673d5ed2cde4be67721f729ddfd  gp/fixtures/trained_hyperparameters.json
57567b54e96cd3aa3f6d0dad2cdcfd0d27cde94ea6e742b27c00e1997c7db1d1  hv/SPEC.md
4a273c6368a027cf3f2ede7bdb470e70f992442be9e1b3a2d5799d3866472c57  hv/capture_hv.py
aee69648d188c90e652c8eac414a559981b67c8067fab8f5d4fe7dbdf893edc3  hv/capture_log.txt
07f5114a850a6a6a69bb2488eec8d68f9efcb8daeb48adaa45152d89b9392771  hv/fixtures/archived_hypervolume.json
879503172b76b0b850221456517c850fda34cf5cc6bb9ae812ff62e0a5d60811  hv/fixtures/bound_roundtrip_edge.json
949e07ff8b7a3330bc402ecaabed16180b2967ec7aa8649cf55cfe2518e5edfa  hv/fixtures/determinism.json
111e1f16b7452a50121138fb5810e699e1922c4abff11dc006409ee3b21b4ff1  hv/fixtures/environment.json
bcbe77fc102068b8cca8e0956fccbaf7099de8acb5a4d35f29643bba18af175c  hv/fixtures/frame.json
03ad316c02aed9375666d581989f96ee44cc37fc8535cdd30b8c7cd6a644226a  hv/fixtures/hv_160_replication.json
5f0431b10e18d7a7716ae6ea2cd313d7e7a8a2d24d473251ca981cf498ab2ae8  hv/fixtures/hv_canonical_cases.json
ad250cbfa25629ec110ffe2d3e823a22b1ed0ab295662c8c620d9e53a3b7e5d5  hv/fixtures/hv_random_fronts.json
07fb083cc4bf8ef6d91aadfec093dc9b1c9232a334442ba5e9376666166eda65  hv/fixtures/independent_checks.json
9cfd6df43296482c8806ece64ed5382cda118690b79ebe5e5a7619f0faa737b3  hv/fixtures/manifest.json
a536c18d54ccca12bc5f80852b4551828186c281827d6ca105ea360590fd14c9  hv/fixtures/normalized_frames.json
c1ff9a4e0c1e336c7ec98efb778b753a02533017a53723f383c6bfcbabea901e  hv/fixtures/output_column_contract.json
05af74b167ad4391cb0d27769db69c7d61f35374fdc1778cb673fd87d4971657  hv/fixtures/pareto_convention_cases.json
354097098674c62d601af707d9ae77f9eba2ba0b1a3bb5f54ba55f69220b97ab  hv/fixtures/ref_filter_strict_vs_weak.json
cf2b7e1937930d0d4e1ac50a6031f331fbd9e05d9ee97400186c9d1493e46527  hv/fixtures/repo_oracle_agreement.json
f9bf8d1c5d9410a21f1a2a582c74b92f31ad8c27bef0d4aded29f2ab7babbd85  hv/fixtures/timings.json
8c15e00a89ac198ceaef8d9096afce90d02540ce1e14adbbb25c2dc0cc1548d5  icm/SPEC.md
b0d8b3527656421d5ad427b068d1b0b8c9f7595dcf90c7ce5b0993fb7aa1cee5  icm/capture_icm.py
3f969bc1640b53958b546303653546f6b87b187a61056131ad400a171f2fb471  icm/fixtures/00_environment.json
9747eb88c90c690a146157975720ff983b46473922468ecbbf6dd04f0ffef82e  icm/fixtures/01_parameterization.json
f614c2b278667708b595bbc74836e7e349da2017758ff52f49d534705948fb05  icm/fixtures/02_ordering.json
768d8ccb4618713cde42ad511633a9905a4a50e5df4f0556f3d86788c9d207f7  icm/fixtures/02_ordering/B_task.npy
9f3bd5a41bd74caab9a6b8de5e7a341e563cc5ac7ee1b1baf5290de92b0c4e7b  icm/fixtures/02_ordering/K_data.npy
747160ebdc7f6faf5f33746bc9d0dd7263699501edfff6584b52a096a6be98cd  icm/fixtures/02_ordering/assembled_dense.npy
1fcd0cb7dbcf7a1587ddfede0bba8d22c7cd5bc360814ca4af630a342b794f6f  icm/fixtures/02_ordering/kron_data_major.npy
2e13e39b4353440afd683f732e2c1257d738f3a9f6278827cf4ae21c1284bd1d  icm/fixtures/02_ordering/kron_task_major.npy
22cf519e05b74fea1cd7ac3219d158a69bcfa3b4a8efd1401748594bf7e38fd7  icm/fixtures/03_real_data.json
8f461a9e9d354ac78996cce7c98d0ce04e93c9bf7ef9d47a0724b0f191a055e3  icm/fixtures/03_real_data/fingerprints_test.npy
b29c9f530492543091ff531de378a65a8c12f12a59cd35acab9a4b4b5d1b394c  icm/fixtures/03_real_data/fingerprints_train.npy
aba82779f3006a679751bb424273996b04b1017b3e0885f365b465ddd452dae9  icm/fixtures/03_real_data/le_train.npy
ff27ee73a4712914836f0c02132dc489dc19c4d74da3eb312fdba164b83c74b5  icm/fixtures/03_real_data/smiles.txt
a88c4185f4c3af057b26a05427ab284156b7d4fc2e50298def57bacd84c29662  icm/fixtures/03_real_data/y_normalized_train.npy
615de79e67d030483c0ac570048930eddcf110fcbef6b932d8f19d61340198f9  icm/fixtures/04_kron.json
e6bf2ca1cb8e396830bc5a91ab77140e97403377e9d637f03a6aad9dd6cfc5eb  icm/fixtures/04_kron/B.npy
720aac9ad5d6887ac105bb85e1e0dea0dec4ccfb4c66e5ae1591c1072ca268f3  icm/fixtures/04_kron/K_xx.npy
0f750e49b70fa7227d2c16a525d51a24c2688615a670459da9487babbedf293e  icm/fixtures/04_kron/Sigma_dense.npy
a0cfcfc93634a590e3aa92d056885058c2ac7eb19ba5ab0cfd73fb81296ecb22  icm/fixtures/04_kron/alpha_chol.npy
5128da294fe30e921c6c1ddcfaa06a52996abfd96c29670a18203ddc66d369ac  icm/fixtures/04_kron/alpha_kron.npy
2f01024b977bfe628ad8c283a0be759a50e48cd0556ca3d01962a22d2677d16d  icm/fixtures/04_kron/raw_params.npy
25bfddbc5954aaec9bbf0696fe1f2e18d95ff035e91c7c3ecd4c612a26cee150  icm/fixtures/05_posterior.json
64d5c99244c9a61eb16542a4140aba269da9ffe83a9fbe5f97cf43972dd08977  icm/fixtures/05_posterior/cov_latent_reference.npy
9ee8087cee1b685a22fd8b4b70ef3b9a09e1665626d708649d74682ea552b7b0  icm/fixtures/05_posterior/mean_reference.npy
30d212de3259c3152eb9538da911a84005f3339164855a593f32c7232e7aa366  icm/fixtures/05_posterior/var_latent_reference.npy
d9150025bfe26f69cc8b7b7284e3e14e3b62da31b7e6c0c6f1c4e7003baf915c  icm/fixtures/05_posterior/var_observed_reference.npy
049bcaed284a747cd0335de62fa599e9e0a0362b6c0f8cbc26457ffb60ecfdfb  icm/fixtures/06_gradients.json
8629e8927bb7ba1cc0e4e364a8a504c4e1545a5afa20e8a643139d88cf315334  icm/fixtures/06_gradients/grad_analytic.npy
1819224a5d588b64f56a57175a9dd53d04035d2e9f763fdff2b53176a981978e  icm/fixtures/06_gradients/grad_autograd.npy
261cf3c8dc0fe06b80eda3437ec60137b452b0ff86a8d9b96363949f0944c918  icm/fixtures/06_gradients/grad_finite_difference.npy
2f01024b977bfe628ad8c283a0be759a50e48cd0556ca3d01962a22d2677d16d  icm/fixtures/06_gradients/theta_raw.npy
fef60b8e607990e48d2f05ff7387a4fc2a1e851706b4e689821b31f8e842e74d  icm/fixtures/07_identifiability.json
22501277a5331b8805066b169a84ad069e61da9313613561fa85faa7a963fe24  icm/fixtures/08_task_correlation.json
e4fa79427808968280f2741b99794fc976fb3fd2e3625888bea54492638a08e3  icm/fixtures/09_acquisition_gap.json
553b32d8c83b1c9cf474403ae2c51c58000ea067fce30825e534e5737df291d9  icm/fixtures/10_scaling.json
784d5bc3413a2392d3e304b22437af0d732b5cda2e68f06aa5e6d9f18c78af6c  icm/fixtures/11_precision.json
1686c5009583c884083f62a61e9a98e51430d08790636f5315f79ac67fe0355f  icm/fixtures/MANIFEST.json
6cd589d07aad4307bc66e4b54248f850e8430132da6b67e73e51c650dad27a53  primitives/SPEC.md
46636477922948aa1cbab81d396919af891460c4c3d0e1ecadd0ddd4cc8a8210  primitives/capture_primitives.py
f000552aed1ef419675f5459952791b3ea8edff1ba82de0a5e8ec18f43933eee  primitives/fixtures/INDEX.json
a3136e09c9f04faa1346bef8f1540813359ef4c44ec92faae1900e7bd0e9009b  primitives/fixtures/adam.json
fcdf2b9324f11c17db112baf87fccdd1bd063c70c18035cdcb011daa190963b1  primitives/fixtures/adam.npz
03510363f46bbf25ba63c8f192b5c94cf3079c03c60e1c96c82d6d5264a1324d  primitives/fixtures/cholesky_and_solves.json
c7504a6df857ec1d25b56cb08e7f22262bde87124c4b504cdc1e1f5ab5d1b3c3  primitives/fixtures/cholesky_and_solves.npz
7a4280a7d55a8d064632f1d1d4ddbf7b72aefcf7d02e0a69e59429b08f71f895  primitives/fixtures/hyperparameter_transforms.json
fda45befb7c7907fd68be69d2140bf4ab66e01844782e41eccfb73a68e1c81d4  primitives/fixtures/hyperparameter_transforms.npz
ac9a58a86d52b3a97d6e4fc7c1f496e52c358dc408c3f52dde42739a72cefb19  primitives/fixtures/tanimoto_kernel.json
3f8e8b9842516c1e1ffb45a6ba72df8d2f9517fd6f6d587eef7ae26364cd23ca  primitives/fixtures/tanimoto_kernel.npz
f62cc44b0856e01ecf52d777ec34e1a507598cad040dd2cadd69b89a74ffd81f  qnehvi/SPEC.md
0e5e95fd39e3f861ce995002ed4b389bcd7d9a132dc4cec88498f4f15827873f  qnehvi/capture_probes.py
aa4b6578d6798a18476d5adbf8aac80faf6d62d53a321055bd8ecb78da95442a  qnehvi/capture_qnehvi.py
9b898f2e4f8116cfb4afdd2238eb112cd279b10622a23a98ce7cd374cfdd4396  qnehvi/check_decomposition_contract.py
bfd9fad461e73b841901d29e916aa8e328e671bc48e0f6adc37333cab114d9a3  qnehvi/fixtures/capture_runtime.txt
8fc0ab80f3b725790a74b94aad312e1bdfa0bfa4a9f9e52ea221824fef28cc01  qnehvi/fixtures/decomposition_contract.json
1f6deb6b16f224a6883ce4d62bc43960cf9dded874d29e9b4f84f477c2c18e69  qnehvi/fixtures/decomposition_timings.txt
c835b97f13c1442072ab7ed0085cac318f2030564f0d8f7cb4ed1b30bba0ace0  qnehvi/fixtures/probe_runtime.txt
27362fb089e2b94c72060547c9a4983871d6951ed00d0a992627414166e0f5bd  qnehvi/fixtures/qnehvi_capture.json
d7093ad39853c7508d7da1d7333200275610d79b7b74aa89f282422df4b69c47  qnehvi/fixtures/qnehvi_probes.json
713104d9471a0b8b0f528e35d6f8efe539ebd48eecf8448d9f0c6c1d08297ccf  qnehvi/fixtures/rank200_Xb.npy
d0354b06857fee0c1459cf06df609e54baf8683ca833a4dfa40af82cb60df3ef  qnehvi/fixtures/rank200_Xc.npy
7867fd63a1e37c162676d1b7d003a3df7d026c3e20cb83fc2983fafacfb4be01  qnehvi/fixtures/rank200_Yd.npy
3cc7c7c2f144edc4816043acab399ccda2f2af3ae5b4899c2e69e1e1014c3c46  qnehvi/fixtures/rank200_admet_b.npy
bf8258727df2b34ff4ecab8d1716404ed8ba250f16e0ee0dfc029d9a21611f3e  qnehvi/fixtures/rank200_admet_c.npy
5336590e2a323cca6cf5ee32852e0ab43887e1ba28bd4a9a636efe0b81a75e2d  qnehvi/fixtures/rank200_mean_b.npy
980769f567b70d22d57ba8d4958962e07bba8d5c8c047ec41d6ba694edb07c4b  qnehvi/fixtures/rank200_mean_c.npy
e2f80cf6e2a541e17b323db4a2acdcb68fab154b496a3ac1e84ab41b1ef7f9b5  qnehvi/fixtures/rank200_var_b.npy
f8c733ac0c99198d871875724d88ea5471525cb713b139beca656615791b4761  qnehvi/fixtures/rank200_var_c.npy
e0c9de874635516086d8898a885568e48385784502d24ec01524afb2f513d6d3  qnehvi/fixtures/realpool_Xb.npy
01bc03d0e705d47cc510802434f7107846a14093d577995b87ef8834dac7ed6f  qnehvi/fixtures/realpool_Yd.npy
fa25c566c60ee662f98370c2da176f19b8bcff4eced09eddfc5861458b3907d7  qnehvi/fixtures/realpool_admet_b.npy
49cf9fb4214d6d3e4dd49f9c7279367d3cd52f902c9f14cb456b0fac40cf3ee0  qnehvi/fixtures/realpool_analog_Xc.npy
fb2e5174fa93f6cbc490f96e55442dbebd08d900bc0ce2c29cdee9245875004a  qnehvi/fixtures/realpool_analog_admet_c.npy
a220d02ffadfd956ea68fba51978b5c81dda6974ef0cef4f60a07fdaff54fc07  qnehvi/fixtures/realpool_analog_lib_row_idx.npy
23031ddb7ffa5e5c7f41414febd2c9165ed303b20f76bd032afa9d9b4b2132c4  qnehvi/fixtures/realpool_analog_mean_c.npy
d659349a7c0775042d81b4c0207d6427a55eb6df782a795219eb51284f1ed2c6  qnehvi/fixtures/realpool_analog_var_c.npy
7923257d3cd0a6a72aa5a189f9aebaad6f4d1b82fa500f66e882df4ede724446  qnehvi/fixtures/realpool_lib_row_idx_baseline.npy
c354313e01d181117624599a7031244b24ec381874384c2235b13c89b8e859b3  qnehvi/fixtures/realpool_mean_b.npy
0f9fb4d1c800f059d41c63b3480d38039892ec8ed065a8b529aed6092b778a34  qnehvi/fixtures/realpool_random_Xc.npy
e7c4b5c471104710dd2136c3d03b2f204a4a7dd01cb278e5fafb75cff3e32e62  qnehvi/fixtures/realpool_random_admet_c.npy
8756ea7aea100d1422a889412776c71976a69c3da7a9ba66b0c69ac0fb8664f7  qnehvi/fixtures/realpool_random_lib_row_idx.npy
d40c9c4c2163a37588c60942972af7d8a6c4df653aef8999f0c5d218a5f51907  qnehvi/fixtures/realpool_random_mean_c.npy
b6ea08aba00de50c4fca9947067c9e10e40970fdbcd9086791712279d9913aa1  qnehvi/fixtures/realpool_random_var_c.npy
3664bc52c12dffa20b7946a0ba29d4251599818a1062ea5bf2c5cdce2f28ce68  qnehvi/fixtures/realpool_var_b.npy
e312ed2c37c4223a6f4a1ce61debd1b665d4bd07f41ad33915cd35e58e5c70c0  qnehvi/fixtures/small25_Xb.npy
e04131467288a37090dd449f5e7112e70fd723c695a824e1a90b8115abeaaad5  qnehvi/fixtures/small25_Xc.npy
2eb02b822d08630126369eaad50c6bab78e8a0549d1f7ddb338e7d7886142ea4  qnehvi/fixtures/small25_Yd.npy
e6148d81c3a525d072ebcda55cd877e9ed80338abb52c2e83d220ad2b09717f1  qnehvi/fixtures/small25_admet_b.npy
475d3564477e405e0164e80f4586fa514945a17826eb11bd06907f51ee3fc41a  qnehvi/fixtures/small25_admet_c.npy
714f9f4f3b9929bea6994e0d53c1799c1fed41e8db366f945b1907b9464e1d31  qnehvi/fixtures/small25_botorch_exact_scores_N128.npy
fc83efb335a5d7e576386f79131d360dc9f33c052098b26b0c0e19ac08c1e6c2  qnehvi/fixtures/small25_bounds.npy
fb9f055f285e6ba25f21bbcf4d6fe5ee4d831416ba646b95c3f812b83ae3201e  qnehvi/fixtures/small25_mean_b.npy
b625170bc42f1db9db61838b7f2199dd97261159afe181830d28a81e509196e0  qnehvi/fixtures/small25_mean_c.npy
f7caabe6b9fbc594e0dc431279d21129c38a85327d4c7469c594fb84c2b87fb3  qnehvi/fixtures/small25_ref.npy
4b0114a9aeb290409bc0b4fbaaef43303499d94b6238a58130545f8e22710c6b  qnehvi/fixtures/small25_signs.npy
c8eb4731d61051b566d8ad0a709a0bad455126264b96a066cb76283b0f655ccf  qnehvi/fixtures/small25_sobol_base_samples_N128.npy
3126091c42cf4bc6587bcc33c76989fb7dfb3434f9c615b46a05e68f5f947feb  qnehvi/fixtures/small25_var_b.npy
859896404619ea6b9a0a8b5a7da0a7afd43550434aba659785766e8841a2fc6d  qnehvi/fixtures/small25_var_c.npy
8009b668c7f8e8d2823d3a7e4df120387719eaa558dad919c0d6c59854307056  stats/SPEC.md
b70f9b598e31c90dbbeaed327258b52f274750d62463e4a00fddca5524628b41  stats/capture_stats.py
011458a344959033de11483902f302d49051589d35083d7ec46c0a0fa8239536  stats/fixtures/MANIFEST.json
5e81971e75125b94e7e335b0a8831950b5c7785450b3c122b17f7d9c45ab13ad  stats/fixtures/ei.json
ba16cf3c085033af7a1bc29f67769a6606862045dee8577d248563eb17f0b719  stats/fixtures/environment.json
59360b35b710905d5e69375b08f1acde0d1192c72c0331a1abaaceb2d0934b21  stats/fixtures/measurements.json
f440a716cbe3ce623114118c7ee43a8ae93587aadc2f8dd5d50d1b803e87894e  stats/fixtures/normal.json
26f1cf7a0442413aa9940f0f94ef79d40051d3c6946ac7e6e894e25de0667f5d  stats/fixtures/properties.json
171789d3c0167c7ae871d2eab138cf861947c89c83a1206960b3d305c422e9ec  stats/fixtures/spearman.json
6e5bc1a974ef980ee93d5bfff79cf14e3ddc3caf1d01b22bee9cd21047ae7723  stats/fixtures/ttest.json
aebd7ff5e054d1cb41c616491fe6bc6b35318eb44651f9e9dafcd079848422ca  stats/fixtures/wilcoxon.json
bf8f0d88a826daeb14aa42e9fdb0346ad3e8f71b61e955aafc550f98baa2dc94  stats/refstats.py
6fc6232733cb0966070c90d9fd0090c6bd4fb95ae62140eb7e3e3d43e9beaa7c  stats/verify_fixtures.py
```

## EXCLUDED-FROM-BYTE-COMPARE

`verify_fixtures.py` still checksums these (drift is worth seeing) but reports them as
`STOCHASTIC` rather than `FAIL` when they move, and `--strict` is required to fail on them.

```
stochastic  admet/fixtures/METADATA.json
libm-dependent  admet/fixtures/ref/half_life_oracle_float64.npy
libm-dependent  admet/fixtures/ref/herg_postlink_float64.npy
stochastic  boxdecomp/fixtures/timings.json
stochastic  gp/fixtures/stochastic_threshold.json
stochastic  hv/fixtures/environment.json
stochastic  hv/fixtures/independent_checks.json
stochastic  hv/fixtures/timings.json
stochastic  icm/fixtures/00_environment.json
stochastic  icm/fixtures/07_identifiability.json
stochastic  icm/fixtures/08_task_correlation.json
stochastic  icm/fixtures/09_acquisition_gap.json
stochastic  icm/fixtures/10_scaling.json
stochastic  icm/fixtures/11_precision.json
stochastic  icm/fixtures/MANIFEST.json
stochastic  qnehvi/fixtures/capture_runtime.txt
stochastic  qnehvi/fixtures/decomposition_timings.txt
stochastic  qnehvi/fixtures/probe_runtime.txt
provenance  stats/fixtures/environment.json
```

---

Manifest covers 216 files, 17,572,329 bytes. Generated by the adversarial
verification pass on branch `feat/numpy-bo-rewrite` at commit `3c6573c`.
