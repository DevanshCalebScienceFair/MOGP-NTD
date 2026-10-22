"""Gates for the GP-MOBO head-to-head.

``run_sweep.py`` refuses to launch unless ``pytest -q bench_yong/test_bench_yong.py``
exits 0.

Aborting (a failure fails the suite):
  G1 oracle + library reproduce thesis Tables 8.4-8.6, 20/20, SMILES and value
  G2 gmean(per-objective components) == scalar oracle to 1e-12, 500 molecules
  G4 library hygiene: all 10,000 candidates parse and are unique
  G6 her driver imports (with the stand-in) and does not without it
  G7 two genuinely importable rdkits, or a moved GP-MOBO checkout, is a STOP
  G8 our arm: committed reference to 1e-10, dense-algebra posterior, brute-force
     hypervolume improvement, sign/bounds errors raise
Non-aborting, recorded to ``$BENCH_YONG_GATE_DIR`` for run_sweep.py to act on:
  G3 batched vs unbatched predictions of her GP agree to 1e-12 (else unbatched)
  G5 determinism: a cell run twice gives a byte-identical evaluated.csv
  G7 the import-level environment record
G9 (memory) is measured on the smoke run, not here.
"""

import glob
import json
import os
import subprocess
import sys

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BENCH_DIR)
for _p in (REPO, BENCH_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import thread_pin  # noqa: E402,F401

import numpy as np  # noqa: E402
import pytest  # noqa: E402

import yong_common as yc  # noqa: E402

GATE_DIR = os.environ.get("BENCH_YONG_GATE_DIR", os.path.join(BENCH_DIR, "_gates"))
PY = sys.executable
THESIS = os.path.join(yc.FIXTURES, "thesis_tables.json")
G8_REF = os.path.join(yc.FIXTURES, "g8_reference.json")
EXPECTED_TOP = {"fexofenadine": "0.72892475", "amlodipine": "0.61237244",
                "perindopril": "0.48795004"}


def record(name, payload):
    os.makedirs(GATE_DIR, exist_ok=True)
    payload = dict(payload)
    payload.update({"gate": name, "repo_head": yc.git_head(),
                    "bench_yong_dirty": yc.git_dirty(rel="bench_yong")})
    yc.write_json_atomic(os.path.join(GATE_DIR, f"{name}.json"), payload)


@pytest.fixture(scope="module")
def lines():
    return yc.read_library_lines()


# --------------------------------------------------------------------------
# G4 library hygiene
# --------------------------------------------------------------------------

def test_G4_library_hygiene(lines):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    with open(yc.LIB_PATH) as fh:
        raw = [next(fh).rstrip("\n") for _ in range(yc.N_READ)]
    assert lines == raw, "her pandas loader does not return the file's lines in order"
    lib = lines[:yc.N_LIBRARY]
    bad = [i for i, s in enumerate(lib) if Chem.MolFromSmiles(s) is None]
    dup = len(lib) - len(set(lib))
    overlap = len(set(lib) & set(lines[yc.N_LIBRARY:]))
    record("G4", {"pass": not bad and dup == 0, "unparseable": bad,
                  "duplicates": dup, "library_holdout_overlap": overlap,
                  "library_sha256": yc.sha256_file(yc.LIB_PATH)})
    assert not bad, f"{len(bad)} candidate SMILES do not parse: {bad[:5]}"
    assert dup == 0, f"{dup} duplicate candidate SMILES"


# --------------------------------------------------------------------------
# G1 oracle and library
# --------------------------------------------------------------------------

@pytest.mark.parametrize("task", ["fexofenadine", "amlodipine", "perindopril"])
def test_G1_oracle_reproduces_thesis_top20(task, lines):
    with open(THESIS) as fh:
        table = json.load(fh)["tables"][task]
    rows = table["rows"]
    assert len(rows) == 20 and rows[0]["printed_value"] == EXPECTED_TOP[task]
    lib = lines[:yc.N_LIBRARY]
    scores = yc.score_scalar(task, lib)
    order = np.argsort(-scores, kind="stable")
    mism, worst = [], 0.0
    for rank, row in enumerate(rows, start=1):
        i = int(order[rank - 1])
        ok_smiles = lib[i] == row["library_smiles"] and i == row["library_index"]
        # The thesis drops trailing zeros (0.4612656), so compare as numbers:
        # our score rounded to 8 places must equal the printed value.
        ok_value = abs(round(float(scores[i]), 8) - float(row["printed_value"])) < 1e-12
        worst = max(worst, abs(scores[i] - float(row["printed_value"])))
        if not (ok_smiles and ok_value):
            mism.append({"rank": rank, "ours": [i, f"{scores[i]:.8f}"],
                         "thesis": [row["library_index"], row["printed_value"]]})
    # A tie at the rank-20/21 boundary is recorded, not failed: the printed
    # table is matched 20/20 by a stable (file-order) sort, which is then the
    # tie rule her table follows.
    boundary_gap = float(scores[order[19]] - scores[order[20]])
    record(f"G1_{task}", {"pass": not mism, "table": table["table"],
                          "matched": 20 - len(mism), "max_abs_value_diff": worst,
                          "mismatches": mism, "rank20_rank21_gap": boundary_gap,
                          "tie_rule": "np.argsort(-score, kind='stable'), file order"})
    assert not mism, f"Table {table['table']} not reproduced: {mism}"


# --------------------------------------------------------------------------
# G2 objective split
# --------------------------------------------------------------------------

@pytest.mark.parametrize("task", ["fexofenadine", "amlodipine", "perindopril"])
def test_G2_objective_split(task, lines):
    yc.add_her_path()
    from scipy.stats.mstats import gmean          # the function her MPOs call
    mols = lines[:yc.N_LIBRARY][::20][:500]
    comp = np.asarray(yc.task_evaluator(task)(mols), dtype=np.float64)
    assert comp.shape == (500, len(yc.TASKS[task]["objectives"]))
    g = np.array([float(gmean(list(row))) for row in comp])
    s = yc.score_scalar(task, mols)
    err = float(np.max(np.abs(g - s)))
    record(f"G2_{task}", {"pass": err <= 1e-12, "max_abs_diff": err,
                          "n": len(mols), "components": yc.TASKS[task]["objectives"],
                          "component_min": comp.min(0).tolist(),
                          "component_max": comp.max(0).tolist()})
    assert err <= 1e-12, f"gmean(components) != {yc.TASKS[task]['oracle']}: {err}"
    assert comp.min() >= 0.0 and comp.max() <= 1.0


# --------------------------------------------------------------------------
# G6 her driver imports
# --------------------------------------------------------------------------

_G6_SCRIPT = r"""
import sys, json, inspect
sys.path.insert(0, {bench!r})
import yong_common as yc
out = {{}}
yc.add_her_path()                   # rdkit.six stand-in + her path only
try:
    import ehvi_mc
    out["as_shipped"] = "imported"
except Exception as e:
    out["as_shipped"] = f"{{type(e).__name__}}: {{e}}"
for m in [k for k in sys.modules if k in ("ehvi_mc", "utils.utils_final")]:
    del sys.modules[m]
yc.install_stand_ins()
import ehvi_mc
sig = inspect.signature(ehvi_mc.expected_hypervolume_improvement)
out.update(
    loop_callable=callable(ehvi_mc.bayesian_optimization_loop),
    loop_params=list(inspect.signature(ehvi_mc.bayesian_optimization_loop).parameters),
    ehvi_N_default=sig.parameters["N"].default,
    evaluator_source=ehvi_mc.evaluate_fex_objectives.__code__.co_filename,
    ehvi_mc_file=ehvi_mc.__file__)
print(json.dumps(out))
"""


def test_G6_her_driver_imports():
    script = _G6_SCRIPT.format(bench=BENCH_DIR)
    p = subprocess.run([PY, "-W", "ignore", "-c", script], capture_output=True,
                       text=True, cwd=yc.GP_DIR, timeout=300)
    assert p.returncode == 0, p.stderr[-2000:]
    out = json.loads(p.stdout.strip().splitlines()[-1])
    record("G6", dict(out, **{"pass": True}))
    assert "osimertinib_tpsa_score" in out["as_shipped"], (
        "her driver imported without the stand-in: the vendored code changed")
    assert out["loop_callable"]
    assert out["ehvi_N_default"] == yc.HER_MC_DRAWS
    assert out["evaluator_source"] == yc.UTILS_FINAL_PATH
    assert os.path.dirname(out["ehvi_mc_file"]) == yc.GP_DIR


# --------------------------------------------------------------------------
# G7 environment
# --------------------------------------------------------------------------

def test_G7_environment():
    versions = yc.imported_versions()
    importable = sorted({os.path.realpath(os.path.join(p, "rdkit"))
                         for p in sys.path
                         if p and os.path.isfile(os.path.join(p, "rdkit", "__init__.py"))})
    site = [p for p in sys.path if p.endswith("site-packages")]
    dist_info = sorted({d for p in site
                        for d in glob.glob(os.path.join(p, "rdkit*.dist-info")) +
                        glob.glob(os.path.join(p, "rdkit*.egg-info"))})
    req_numpy = None
    with open(os.path.join(REPO, "requirements.txt")) as fh:
        for line in fh:
            if line.strip().startswith("numpy"):
                req_numpy = line.strip()
    gp_head = yc.git_head(yc.GP_DIR)
    gp_dirty = yc.git_dirty(yc.GP_DIR)
    rec = {"versions": versions, "importable_rdkit_dirs": importable,
           "rdkit_dist_info": dist_info, "requirements_numpy_pin": req_numpy,
           "gp_mobo_head": gp_head, "gp_mobo_dirty": gp_dirty,
           "stale_rdkit_metadata": len(dist_info) > 1 and len(importable) == 1}
    rec["pass"] = len(importable) == 1 and gp_head == yc.GP_MOBO_COMMIT and not gp_dirty
    record("G7", rec)
    assert len(importable) == 1, f"STOP: {len(importable)} importable rdkits: {importable}"
    assert gp_head == yc.GP_MOBO_COMMIT, f"GP-MOBO checkout moved: {gp_head}"
    assert not gp_dirty, "GP-MOBO working tree has local changes"


# --------------------------------------------------------------------------
# G3 batched predictions (non-aborting)
# --------------------------------------------------------------------------

def test_G3_batched_predictions_record():
    yc.install_stand_ins()
    from kern_gp import gp_model
    original = gp_model.get_fingerprint
    results = {}
    try:
        for task in ("fexofenadine", "amlodipine"):
            start, pool, holdout, _ = yc.split_for_seed(0, 20)
            ev = yc.task_evaluator(task)
            hy, Y = ev(holdout), ev(start)
            kw = dict(known_smiles=start, known_Y=Y, gp_means=hy.mean(0),
                      gp_amplitudes=hy.var(0), gp_noises=hy.var(0) * 0.1)
            cands = pool[:500]
            for fp_name, fp in (("PUB", original), ("FP2", yc.fp2_bitvect)):
                gp_model.get_fingerprint = fp
                mb, vb = gp_model.independent_tanimoto_gp_predict(query_smiles=cands, **kw)
                one = [gp_model.independent_tanimoto_gp_predict(query_smiles=[s], **kw)
                       for s in cands]
                mu = np.vstack([m for m, _ in one])
                va = np.vstack([v for _, v in one])
                results[f"{task}|{fp_name}"] = {
                    "max_abs_mean": float(np.abs(mb - mu).max()),
                    "max_abs_var": float(np.abs(vb - va).max()),
                    "bitwise": bool(np.array_equal(mb, mu) and np.array_equal(vb, va))}
    finally:
        gp_model.get_fingerprint = original
    ok = all(r["max_abs_mean"] <= 1e-12 and r["max_abs_var"] <= 1e-12
             for r in results.values())
    record("G3", {"pass": ok, "her_predict": "batched" if ok else "unbatched",
                  "tolerance": 1e-12, "n_candidates": 500, "results": results})


# --------------------------------------------------------------------------
# G5 determinism (non-aborting)
# --------------------------------------------------------------------------

def test_G5_determinism_record(tmp_path):
    cells = [("fexofenadine", a) for a in yc.ARMS] + [("amlodipine", "HER@FP2"),
                                                       ("amlodipine", "MOGP-ICM@FP2")]
    env = dict(os.environ, PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
    procs = []
    for task, arm in cells:
        for copy in ("a", "b"):
            out = tmp_path / f"{task}_{yc.arm_slug(arm)}_{copy}"
            cmd = [PY, "-W", "ignore", "-u", os.path.join(BENCH_DIR, "run_one.py"),
                   "--task", task, "--arm", arm, "--seed", "1", "--n-init", "20",
                   "--n-iterations", "3", "--pool-limit", "120",
                   "--her-predict", "batched", "--output-dir", str(out)]
            procs.append(((task, arm, copy, out), subprocess.Popen(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env)))
    rc = {}
    for key, pr in procs:
        _, err = pr.communicate(timeout=1800)
        rc[key[:3]] = (pr.returncode, err.decode()[-500:] if pr.returncode else "")
    results = {}
    for task, arm in cells:
        a = tmp_path / f"{task}_{yc.arm_slug(arm)}_a" / "evaluated.csv"
        b = tmp_path / f"{task}_{yc.arm_slug(arm)}_b" / "evaluated.csv"
        same = a.exists() and b.exists() and a.read_bytes() == b.read_bytes()
        results[f"{task}|{arm}"] = {"identical": same,
                                    "returncodes": [rc[(task, arm, "a")][0],
                                                    rc[(task, arm, "b")][0]],
                                    "stderr": rc[(task, arm, "a")][1]}
    record("G5", {"pass": all(r["identical"] for r in results.values()),
                  "scope": "reduced: pool 120, 3 iterations, seed 1; the full-size "
                           "check is the duplicated smoke cell",
                  "results": results})


# --------------------------------------------------------------------------
# G8 the method under test
# --------------------------------------------------------------------------

import make_g8_reference as g8  # noqa: E402

G8_CASES = [f"{t}|{a}" for t, a in g8.CASES]


@pytest.fixture(scope="module")
def g8_ref():
    with open(G8_REF) as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def g8_runs():
    return {f"{t}|{a}": g8.compute_case(t, a) for t, a in g8.CASES}


def _tanimoto(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    inter = a @ b.T
    return inter / (a.sum(1)[:, None] + b.sum(1)[None, :] - inter)


def _dense_posterior(arm, model, y_mean, y_std, Xk, Xc):
    """The posterior by direct dense linear algebra, from the fitted theta alone."""
    from mogp_core import gp as core_gp
    from mogp_core import icm as core_icm
    Kxx, Ksx = _tanimoto(Xk, Xk), _tanimoto(Xc, Xk)
    Yn = np.asarray(model.Y_norm, dtype=np.float64)
    N, K = Yn.shape
    M = Xc.shape[0]
    if arm == "MOGP-ICM@FP2":
        B, tau, sigma2, d, c = core_icm.natural(model.theta, K, model.rank)
        Sigma = np.kron(Kxx, B) + np.kron(np.eye(N), np.diag(tau)) + sigma2 * np.eye(N * K)
        Kc = np.kron(Ksx, B)
        mean = c[None, :] + (Kc @ np.linalg.solve(Sigma, (Yn - c[None, :]).reshape(-1))
                             ).reshape(M, K)
        corr = np.einsum("ij,ji->i", Kc, np.linalg.solve(Sigma, Kc.T)).reshape(M, K)
        var = np.diag(B)[None, :] - corr + d[None, :]
    else:
        s, mu, dd, sigma2, tau = core_gp.natural(model.theta, K)
        mean = np.empty((M, K))
        var = np.empty((M, K))
        for t in range(K):
            Kt = s[t] * Kxx + tau[t] * np.eye(N)
            Ks = s[t] * Ksx
            mean[:, t] = mu[t] + Ks @ np.linalg.solve(Kt, Yn[:, t] - mu[t])
            vf = s[t] - np.einsum("ij,ji->i", Ks, np.linalg.solve(Kt, Ks.T))
            var[:, t] = np.maximum(vf, 0.0) + tau[t]
    ym = np.asarray(y_mean, dtype=np.float64)
    ys = np.asarray(y_std, dtype=np.float64)
    return mean * ys + ym, var * ys ** 2


@pytest.mark.parametrize("case", G8_CASES)
def test_G8_reproduces_committed_reference(case, g8_ref, g8_runs):
    ref, r = g8_ref["cases"][case], g8_runs[case]
    worst = {k: float(np.max(np.abs(np.asarray(r[k]) - np.asarray(ref[k]))))
             for k in ("mu_b", "var_b", "mu_c", "var_c", "scores")}
    ok = max(worst.values()) <= 1e-10 and r["best"] == ref["best"]
    record(f"G8_reference_{case.replace('|', '_')}",
           {"pass": ok, "max_abs_diff": worst, "best": r["best"],
            "ref_best": ref["best"], "best_score": r["best_score"]})
    assert ok, f"{case}: {worst}, best {r['best']} vs {ref['best']}"


@pytest.mark.parametrize("case", G8_CASES)
def test_G8_posterior_known_answer_dense(case, g8_runs):
    task, arm = case.split("|")
    _, Xk, Yk, Xc, _ = g8.g8_problem(task)
    r = g8_runs[case]
    mean, var = _dense_posterior(arm, r["model"], r["y_mean"], r["y_std"], Xk, Xc)
    dm = float(np.max(np.abs(mean - r["mu_c"])))
    dv = float(np.max(np.abs(np.clip(var, 1e-9, None) - r["var_c"])))
    record(f"G8_dense_{case.replace('|', '_')}",
           {"pass": dm <= 1e-10 and dv <= 1e-10, "max_abs_mean": dm, "max_abs_var": dv})
    assert dm <= 1e-10 and dv <= 1e-10, (dm, dv)


@pytest.mark.parametrize("case", ["fexofenadine|MOGP-ICM@FP2", "amlodipine|MOGP-ICM@FP2"])
def test_G8_acquisition_known_answer_bruteforce(case, g8_runs):
    """Exact estimator on fixed normals == hypervolume difference by HER
    Hypervolume class, an implementation independent of mogp_core.boxdecomp."""
    from mogp_core import qnehvi as core_qn
    yc.add_her_path()
    from acquisition_funcs.hypervolume import Hypervolume
    r = g8_runs[case]
    K = r["mu_b"].shape[1]
    idx = np.arange(0, 470, 40)
    mu_b, var_b = r["mu_b"], r["var_b"]
    mu_c, var_c = r["mu_c"][idx], r["var_c"][idx]
    B, n_mc = mu_b.shape[0], 24
    z = np.random.default_rng(12345).standard_normal((n_mc, B + 1, K))
    bounds = np.tile([0.0, 1.0], (K, 1))
    got = core_qn.qnehvi(mu_b, var_b, np.zeros((B, 0)), mu_c, var_c,
                         np.zeros((len(idx), 0)), bounds, estimator="exact",
                         n_mc=n_mc, signs=np.ones(K), ref=np.zeros(K),
                         dock_idx=tuple(range(K)), lib_idx=(), num_objectives=K,
                         base_samples=z)
    sd_b = np.sqrt(np.clip(var_b, 1e-9, None))
    sd_c = np.sqrt(np.clip(var_c, 1e-9, None))
    hv = Hypervolume(np.zeros(K))
    want = np.zeros(len(idx))
    for t in range(n_mc):
        Yb = np.clip(mu_b + sd_b * z[t, :B], 0.0, 1.0)
        base = hv.compute(Yb)
        for j in range(len(idx)):
            y = np.clip(mu_c[j] + sd_c[j] * z[t, B], 0.0, 1.0)
            want[j] += max(0.0, hv.compute(np.vstack([Yb, y])) - base)
    want /= n_mc
    err = float(np.max(np.abs(got - want)))
    record(f"G8_bruteforce_{case.replace('|', '_')}",
           {"pass": err <= 1e-10, "max_abs_diff": err, "m": K,
            "nonzero_candidates": int(np.sum(want > 0))})
    assert np.sum(want > 0) >= 3, "degenerate check: no candidate improves"
    assert err <= 1e-10, err


@pytest.mark.parametrize("K", [2, 3])
def test_G8_dominance_known_answer(K):
    """Shipped estimator config: a dominating candidate must score highest and a
    dominated one lowest.  A flipped sign or a wrong frame inverts this."""
    sys.path.insert(0, REPO)
    import acquisition
    from mogp_core import qnehvi as core_qn
    rng = np.random.default_rng(7)
    mu_b = rng.uniform(0.3, 0.6, size=(25, K))
    var_b = np.full_like(mu_b, 1e-6)
    mu_c = np.array([[0.95] * K, [0.05] * K, [0.5] * K])
    var_c = np.full_like(mu_c, 1e-6)
    s = core_qn.qnehvi(mu_b, var_b, np.zeros((25, 0)), mu_c, var_c,
                       np.zeros((3, 0)), np.tile([0.0, 1.0], (K, 1)),
                       estimator=acquisition._ESTIMATOR,
                       n_mc=acquisition.N_MC_SAMPLES, run_seed=0, iteration=1,
                       signs=np.ones(K), ref=np.zeros(K), dock_idx=tuple(range(K)),
                       lib_idx=(), num_objectives=K, sampler=acquisition._SAMPLER,
                       layout=acquisition._SAMPLE_LAYOUT)
    assert np.argmax(s) == 0 and np.argmin(s) == 1, s


def test_G8_sign_and_bounds_errors_raise(g8_runs):
    import run_one
    _, Xk, Yk, Xc, _ = g8.g8_problem("fexofenadine")
    with pytest.raises(ValueError):
        run_one.ours_step("MOGP-ICM@FP2", Xk, -Yk, Xc, 0, 1)     # sign error
    with pytest.raises(ValueError):
        run_one.ours_step("MOGP-ICM@FP2", Xk, Yk * 100.0, Xc, 0, 1)  # bounds error
    with pytest.raises(ValueError):
        run_one._check_frame(Yk[:, :2], 3, "wrong width")


def test_G8_fp2_is_the_shared_fingerprint(lines):
    """HER@FP2 and our arms must see the same bits."""
    sys.path.insert(0, REPO)
    from utils.featurize import smiles_to_morgan
    mols = lines[:yc.N_LIBRARY][::20][:500]
    bad = 0
    for s in mols:
        bv = yc.fp2_bitvect(s)
        arr = np.zeros(2048, dtype=np.int8)
        arr[list(bv.GetOnBits())] = 1
        bad += int(not np.array_equal(arr, smiles_to_morgan(s)))
    record("G8_fp2_identity", {"pass": bad == 0, "n": len(mols), "mismatches": bad})
    assert bad == 0


# --------------------------------------------------------------------------
# Report statistics (not gates: checks of report.py's pre-registered rules)
# --------------------------------------------------------------------------

import report  # noqa: E402


def test_report_holm():
    assert np.allclose(report.holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    assert np.allclose(report.holm([0.5, 0.5, 0.5]), [1.0, 1.0, 1.0])


def test_report_tie_rule_and_exact_p():
    r = report.signed_rank([0.1] * 10)
    assert r["p"] == pytest.approx(2 / 1024) and not r["inconclusive_by_construction"]
    r = report.signed_rank([0.1, 0.2, 0.3, 0.4, 0.5, 0, 0, 0, 0, 0])
    assert r["n_nontied"] == 5 and r["inconclusive_by_construction"]
    assert r["p"] == pytest.approx(0.0625)          # the smallest p at n = 5


def test_report_verdict_rules():
    def t(med, hp, inc=False):
        return {"median_diff": med, "holm_p": hp, "inconclusive_by_construction": inc}
    assert report.verdict({"a": t(.1, .01), "b": t(.1, .02), "c": t(-.1, .9)})[0] == "WIN"
    assert report.verdict({"a": t(0, 1), "b": t(-.1, .02), "c": t(.1, .01)})[0] == "LOSS"
    assert report.verdict({"a": t(.1, .2), "b": t(.1, .3), "c": t(.1, .01)})[0] == "INCONCLUSIVE"
    assert report.verdict({"a": t(.1, .01, True), "b": t(.1, .02), "c": t(.1, .5)})[0] \
        == "INCONCLUSIVE"


def test_report_library_rank():
    lib = np.array([0.9, 0.8, 0.8, 0.5])
    assert report.library_rank(0.9, lib) == 1
    assert report.library_rank(0.8, lib) == 2
    assert report.library_rank(0.5, lib) == 4


def test_report_verdict_no_data():
    t = {"median_diff": float("nan"), "holm_p": 1.0,
         "inconclusive_by_construction": True, "n_pairs": 0}
    ok = {"median_diff": .1, "holm_p": .01, "inconclusive_by_construction": False,
          "n_pairs": 10}
    assert report.verdict({"a": t, "b": ok, "c": ok})[0].startswith("UNDETERMINED")
