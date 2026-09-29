#!/usr/bin/env python
"""capture_probes.py -- the secondary measurements cited by SPEC.md.

Run:  conda run -n mogp-drug python /Users/calebsuh/MOGP-NTD/spec/qnehvi/capture_probes.py

These are the measurements that answer "is that claim actually true?" rather than
"what does the current code produce". They are separated from
``capture_qnehvi.py`` because each one is slow and none of them is a reference
value an implementation has to reproduce -- they justify decisions.

Everything here is seeded, so the JSON output
(``fixtures/qnehvi_probes.json``) is byte-reproducible. Wall-clock timings go to
``fixtures/probe_runtime.txt``, never into the JSON.

Reuses ``capture_qnehvi`` as a library (same estimator, same generator, same
fixed-hyperparameter GP), so the two capture scripts cannot drift apart.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import capture_qnehvi as C  # noqa: E402

FIX = C.FIX


def probe_reduction_order() -> dict:
    """mean(axis=0) vs a streaming accumulator, at the magnitudes we actually see.

    Cited by SPEC.md Appendix A. The Mc == 1 case is the trap.
    """
    rng = np.random.default_rng(5)
    out = {}
    for (N, M) in ((512, 25), (512, 200), (512, 1), (128, 3), (128, 1),
                   (8192, 200), (512, 5000)):
        worst = 0.0
        trials = 20 if M <= 200 else 3
        for _ in range(trials):
            per = rng.lognormal(mean=-8.0, sigma=2.0, size=(N, M))
            per[rng.random((N, M)) < 0.4] = 0.0
            a = per.mean(axis=0)
            acc = np.zeros(M)
            for t in range(N):
                acc += per[t]
            b = acc / float(N)
            nz = a > 0
            if nz.any():
                worst = max(worst, float(np.max(np.abs(a - b)[nz] / a[nz])))
        out[f"N{N}_M{M}"] = {
            "n_trials": trials,
            "max_rel_diff": worst,
            "bitwise_identical": bool(worst == 0.0),
        }
    return {
        "claim": (
            "numpy reduces a 2-D array along axis 0 row by row, so mean(axis=0) "
            "equals a streaming accumulator BITWISE -- except at Mc == 1, where "
            "it switches to pairwise summation over a contiguous vector."
        ),
        "per_shape": out,
    }


def probe_numpy_chunk_invariance() -> dict:
    """The numpy estimator must be BITWISE invariant to candidate chunking."""
    out = {}
    for tag, (nb, nc, seed) in (("small25", (12, 25, 20260927)),
                                ("rank200", (12, 200, 777))):
        P = C.make_problem(nb, nc, seed)
        m, l, ym, ys = C.fixed_gp(P["Xb"], P["Yd"])
        mb, vb = C.gp_posterior(m, l, ym, ys, P["Xb"])
        mc, vc = C.gp_posterior(m, l, ym, ys, P["Xc"])
        key = (f"probe-chunk-{tag}",)

        def run(chunk):
            sd_b = np.sqrt(np.clip(vb, 1e-9, None))
            sd_c = np.sqrt(np.clip(vc, 1e-9, None))
            zb = C.draw_normals(128, nb, C.RUN_SEED, C.ITERATION, 0)
            zc = C.draw_normals(128, 1, C.RUN_SEED, C.ITERATION, 1)
            Yb = C.compose_and_normalize(mb[None] + sd_b[None] * zb,
                                         P["admet_b"][None])
            Yc = C.compose_and_normalize(
                mc[None] + sd_c[None] * zc[:, 0, None, :], P["admet_c"][None])
            cells = C.cached_cells(key, Yb, 128)
            per = np.empty((128, nc))
            for t in range(128):
                L, U = cells[t]
                for s in range(0, nc, chunk):
                    per[t, s:s + chunk] = C.hvi_from_cells(Yc[t, s:s + chunk], L, U)
            return per.mean(axis=0)

        ref = run(nc)
        d = {}
        for ch in (nc, 64, 25, 13, 7, 3, 2, 1):
            if ch > nc:
                continue
            s = run(ch)
            d[str(ch)] = {
                "bitwise_identical": bool(np.array_equal(s, ref)),
                "max_abs_diff": float(np.max(np.abs(s - ref))),
            }
        out[tag] = d
    return {"n_mc": 128, "per_problem": out,
            "claim": "bitwise identical at every candidate chunk size"}


def probe_crn_vs_ind_se(R: int = 128, N: int = 128) -> dict:
    """Do CRN and IND really differ in per-candidate value SE, or is it the
    estimator of the SE that differs?  Measures the cross-candidate correlation
    of the replicate deviations, which is the mechanism."""
    PR = C.make_problem(12, 200, 777)
    m, l, ym, ys = C.fixed_gp(PR["Xb"], PR["Yd"])
    mb, vb = C.gp_posterior(m, l, ym, ys, PR["Xb"])
    mc, vc = C.gp_posterior(m, l, ym, ys, PR["Xc"])
    out = {"n_replicates": R, "n_mc": N}
    for lay in ("CRN", "IND"):
        reps = np.stack([
            C.reference_qnehvi(mb, vb, PR["admet_b"], mc, vc, PR["admet_c"],
                               n_mc=N, run_seed=C.RUN_SEED, iteration=r,
                               layout=lay, cache_key=("probe-crnind", r))
            for r in range(R)
        ])
        mu = reps.mean(0)
        sd = reps.std(0, ddof=1)
        nz = mu > 0
        rse = 100.0 * sd[nz] / mu[nz]
        z = (reps[:, nz] - mu[nz]) / sd[nz]
        corr = np.corrcoef(z.T)
        off = corr[np.triu_indices_from(corr, 1)]
        out[lay] = {
            "median_relative_se_pct": float(np.median(rse)),
            "mean_relative_se_pct": float(rse.mean()),
            "p90_relative_se_pct": float(np.percentile(rse, 90)),
            "n_nonzero_candidates": int(nz.sum()),
            "median_cross_candidate_corr_of_replicate_deviations": float(
                np.median(off)),
        }
    out["claim"] = (
        "Theory: identical per-candidate variance. The measured medians differ "
        "because CRN's cross-candidate correlation (~0.91) stops the median over "
        "candidates from averaging down its own estimation error."
    )
    return out


def probe_diversity_filter() -> dict:
    """Is the greedy Tanimoto filter actually biting (so C3 is not vacuous),
    and how stable is the DIVERSE top-20 across replicates?"""
    out = {}
    bite = []
    for trial in range(3):
        Pt = C.make_problem(12, 200, 4000 + trial)
        sim = C.tanimoto(Pt["Xc"])
        np.fill_diagonal(sim, 0.0)
        m, l, ym, ys = C.fixed_gp(Pt["Xb"], Pt["Yd"])
        mb, vb = C.gp_posterior(m, l, ym, ys, Pt["Xb"])
        mc, vc = C.gp_posterior(m, l, ym, ys, Pt["Xc"])
        s = C.reference_qnehvi(mb, vb, Pt["admet_b"], mc, vc, Pt["admet_c"],
                               n_mc=64, cache_key=("probe-div", trial))
        order = C.rank_stable(s)
        sel = C.greedy_diverse(order, sim, 20, 0.7)
        depth = max(int(np.where(order == i)[0][0]) for i in sel) + 1
        bite.append({
            "trial": trial,
            "n_pairs_above_threshold_0.7": int((sim > 0.7).sum() // 2),
            "largest_mutually_dissimilar_subset": int(
                len(C.greedy_diverse(np.arange(200), sim, 200, 0.7))),
            "n_selected": int(sel.size),
            "greedy_walk_depth": int(depth),
            "overlap_selected_with_raw_top20": int(
                len(set(sel.tolist()) & set(order[:20].tolist()))),
        })
    out["filter_bites"] = bite

    PR = C.make_problem(12, 200, 777)
    m, l, ym, ys = C.fixed_gp(PR["Xb"], PR["Yd"])
    mb, vb = C.gp_posterior(m, l, ym, ys, PR["Xb"])
    mc, vc = C.gp_posterior(m, l, ym, ys, PR["Xc"])
    sim = C.tanimoto(PR["Xc"])
    np.fill_diagonal(sim, 0.0)
    R = 16
    stab = {}
    for N in (128, 512):
        reps = np.stack([
            C.reference_qnehvi(mb, vb, PR["admet_b"], mc, vc, PR["admet_c"],
                               n_mc=N, run_seed=C.RUN_SEED, iteration=r,
                               layout="CRN", cache_key=("probe-stab", r))
            for r in range(R)
        ])
        sels, depths = [], []
        for r in range(R):
            o = C.rank_stable(reps[r])
            sl = C.greedy_diverse(o, sim, 20, 0.7)
            sels.append(set(sl.tolist()))
            depths.append(max(int(np.where(o == i)[0][0]) for i in sl) + 1)
        raws, divs = [], []
        for i in range(R):
            for j in range(i + 1, R):
                raws.append(C.topk_overlap(reps[i], reps[j], 20))
                divs.append(len(sels[i] & sels[j]) / 20.0)
        stab[f"CRN_N{N}"] = {
            "n_replicates": R,
            "pairwise_raw_top20_overlap_mean": float(np.mean(raws)),
            "pairwise_raw_top20_overlap_min": float(np.min(raws)),
            "pairwise_diverse_top20_overlap_mean": float(np.mean(divs)),
            "pairwise_diverse_top20_overlap_min": float(np.min(divs)),
            "greedy_walk_depth_median": int(np.median(depths)),
            "greedy_walk_depth_max": int(max(depths)),
        }
    out["diverse_top20_stability"] = stab
    out["claim"] = (
        "The diversity filter is not saturating (so C3 is a real test), and it "
        "is the dominant source of batch-to-batch variability: the raw top-20 is "
        "perfectly stable across replicates while the DIVERSE top-20 is not."
    )
    return out


def probe_gp_call_shape() -> dict:
    """Is the chunk dependence of the shipped code in the Tanimoto Gram or
    downstream of it?  Reproduces the exact (chunk*(B+1), n_fp) matrices."""
    import torch
    from kernel import TanimotoKernel
    P = C.make_problem(12, 25, 20260927)
    model, lik, ym, ys = C.fixed_gp(P["Xb"], P["Yd"])
    Xb = P["Xb"].astype(np.float64)
    Xc = P["Xc"].astype(np.float64)
    tk = TanimotoKernel()
    means, grams = {}, {}
    for c in (25, 13, 7, 1):
        ms, vs, gs = [], [], []
        for s in range(0, 25, c):
            blk = Xc[s:s + c]
            full = np.concatenate(
                [np.broadcast_to(Xb, (blk.shape[0], 12, 2048)), blk[:, None, :]],
                axis=1).reshape(-1, 2048)
            m1, v1 = C.gp_posterior(model, lik, ym, ys, full)
            ms.append(m1)
            vs.append(v1)
            gs.append(tk.forward(torch.from_numpy(full).to(torch.float32),
                                 torch.from_numpy(Xb).to(torch.float32)).numpy())
        means[str(c)] = (np.concatenate(ms, 0), np.concatenate(vs, 0))
        grams[str(c)] = np.concatenate(gs, 0)
    mref, vref = means["25"]
    gref = grams["25"]
    return {
        "claim": (
            "The Tanimoto Gram is bitwise chunk-invariant; mogp.predict's float32 "
            "posterior solve downstream of it is not. So the chunk dependence of "
            "the shipped acquisition lives in the GP, not the sampler."
        ),
        "n_rows_per_predict_call": {str(c): int(min(c, 25) * 13)
                                    for c in (25, 13, 7, 1)},
        "mean_max_abs_diff_vs_chunk25": {
            k: float(np.max(np.abs(v[0] - mref))) for k, v in means.items()},
        "var_max_abs_diff_vs_chunk25": {
            k: float(np.max(np.abs(v[1] - vref))) for k, v in means.items()},
        "tanimoto_gram_max_abs_diff_vs_chunk25": {
            k: float(np.max(np.abs(v - gref))) for k, v in grams.items()},
        "float32_eps": float(np.finfo(np.float32).eps),
    }


def main():
    os.makedirs(FIX, exist_ok=True)
    t0 = time.time()
    res = {"env": {"numpy": np.__version__}}
    stage_times = []
    for name, fn in (
        ("reduction_order", probe_reduction_order),
        ("numpy_chunk_invariance", probe_numpy_chunk_invariance),
        ("gp_call_shape", probe_gp_call_shape),
        ("diversity_filter", probe_diversity_filter),
        ("crn_vs_ind_se", probe_crn_vs_ind_se),
    ):
        t1 = time.time()
        print(f"[probe] {name} ...", flush=True)
        res[name] = fn()
        stage_times.append((name, time.time() - t1))
        print(f"[probe] {name} done", flush=True)
        C._CELLS_CACHE.clear()
    path = os.path.join(FIX, "qnehvi_probes.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2, sort_keys=True, default=C._json_default)
        f.write("\n")
    with open(os.path.join(FIX, "probe_runtime.txt"), "w") as f:
        for n, dt in stage_times:
            f.write(f"{n}\t{dt:.1f}s\n")
        f.write(f"total\t{time.time() - t0:.1f}s\n")
    print(f"\nwrote {path} ({os.path.getsize(path)} bytes)")
    print("sha256", C.sha256_file(path))


if __name__ == "__main__":
    main()
