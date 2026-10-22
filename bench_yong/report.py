#!/usr/bin/env python
"""Report for the GP-MOBO head-to-head, built ONLY from finished files on disk.

* Discovers cells by walking ``<output-dir>/cells``; never consults what any
  invocation ran.  Completeness is checked against the plan in
  ``manifest.json``; a shortfall stamps the whole report PARTIAL and lists
  every missing cell and every pair it removes from a signed-rank test.
* Re-scores every molecule from its SMILES with her scalar oracle.  The
  ``mpo_logged`` column the runs wrote is only compared against, never used,
  so a logging bug in one arm cannot move the ranking.
* Order of output is fixed by PREREGISTRATION.md as amended by
  PREREGISTRATION_AMENDMENT_1.md: L1-REVISED (the random floor) first, with the
  superseded L1-ORIGINAL printed beside it; then the primary endpoint and its
  verdict (WIN, LOSS-REVISED, INCONCLUSIVE), with the verdict the superseded
  LOSS rule would have given; then L2 (every rank, never only the median), L3,
  L4, the reference arm, and the context tables.

Writes ``<output-dir>/report/{report.md,report.json,per_cell.csv}``.
"""

import argparse
import csv
import datetime
import io
import json
import os
import subprocess
import sys

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BENCH_DIR)
sys.path.insert(0, REPO)
sys.path.insert(0, BENCH_DIR)
import thread_pin  # noqa: E402,F401

import numpy as np  # noqa: E402

import yong_common as yc  # noqa: E402
from mogp_core import stats as core_stats  # noqa: E402

ALPHA = 0.05
MIN_NONTIED = 6            # with <= 5 pairs the smallest two-sided p is 0.0625
TASK_LIST = ["fexofenadine", "amlodipine", "perindopril"]
PRIMARY = ("MOGP-ICM@FP2", "HER@FP2")
AMENDMENT_1 = "bench_yong/PREREGISTRATION_AMENDMENT_1.md"


# --------------------------------------------------------------------------
# Statistics (pre-registered definitions)
# --------------------------------------------------------------------------

def signed_rank(d):
    """Two-sided Wilcoxon signed-rank, zeros dropped (zero_method='wilcox').

    ``mogp_core.stats.wilcoxon`` with method='auto': exact null when there are
    no ties or zeros, exhaustive sign-flip enumeration otherwise (n <= 13).
    """
    d = np.asarray(d, dtype=np.float64)
    n_nontied = int(np.count_nonzero(d != 0.0))
    out = {"n_pairs": int(d.size), "n_nontied": n_nontied,
           "median_diff": float(np.median(d)) if d.size else float("nan"),
           "mean_diff": float(np.mean(d)) if d.size else float("nan"),
           "n_positive": int(np.sum(d > 0)), "n_negative": int(np.sum(d < 0)),
           "n_zero": int(np.sum(d == 0))}
    if n_nontied == 0:
        out.update(statistic=float("nan"), p=1.0, branch="all-tied")
    else:
        r = core_stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided",
                                method="auto")
        out.update(statistic=float(r.statistic), p=float(r.pvalue), branch=r.branch)
    out["inconclusive_by_construction"] = n_nontied < MIN_NONTIED
    return out


def holm(pvals):
    """Holm step-down adjusted p-values (family = the list given)."""
    p = np.asarray(pvals, dtype=np.float64)
    m = p.size
    order = np.argsort(p, kind="stable")
    adj = np.empty(m)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (m - k) * p[i]))
        adj[i] = running
    return adj.tolist()


def task_category(res, p_key):
    """WIN-like / LOSS-like / INCONCLUSIVE for one task (used by L4)."""
    if res["n_pairs"] == 0:
        return "no data"
    if res["median_diff"] <= 0:
        return "non-positive median"
    if res[p_key] < ALPHA and not res["inconclusive_by_construction"]:
        return "positive, significant"
    return "positive, not significant"


def _significant(r, sign):
    """Holm p < 0.05, >= 6 non-tied seeds, median strictly on the ``sign`` side."""
    return (r["holm_p"] < ALPHA and not r["inconclusive_by_construction"]
            and sign * r["median_diff"] > 0)


def verdict(per_task):
    """PREREGISTRATION section 5 as amended by Amendment 1.

    WIN: significant with a positive median on >= 2 of 3 tasks (unchanged).
    LOSS (LOSS-REVISED): significant with a NEGATIVE median on >= 2 of 3 tasks.
    Everything else is INCONCLUSIVE, a negative median without significance
    included.  A task with no pairs has no median and counts towards neither;
    the report names it and stamps the verdict PROVISIONAL while PARTIAL.
    Returns ``(verdict, winning_tasks, losing_tasks)``.
    """
    wins = [t for t, r in per_task.items() if _significant(r, +1)]
    losses = [t for t, r in per_task.items() if _significant(r, -1)]
    if len(wins) >= 2:
        return "WIN", wins, losses
    if len(losses) >= 2:
        return "LOSS", wins, losses
    return "INCONCLUSIVE", wins, losses


def verdict_original(per_task):
    """SUPERSEDED by Amendment 1.  LOSS was a median <= 0 on >= 2 of 3 tasks
    with no test, which fires ~50% of the time when the methods are identical.
    Kept only so the report can show what the original rule would have said."""
    wins = [t for t, r in per_task.items() if _significant(r, +1)]
    nonpos = [t for t, r in per_task.items() if r["median_diff"] <= 0]
    if len(wins) >= 2:
        return "WIN", wins, nonpos
    if len(nonpos) >= 2:
        return "LOSS", wins, nonpos
    return "INCONCLUSIVE", wins, nonpos


def floor_revised(per_task):
    """L1-REVISED (Amendment 1): cleared if MOGP-ICM beats RANDOM significantly
    (Holm p < 0.05, median > 0, >= 6 non-tied) on >= 2 of 3 tasks; otherwise it
    FIRES.  Undetermined while a task has no pairs and the floor is not yet
    cleared."""
    clearing = [t for t, r in per_task.items() if _significant(r, +1)]
    no_data = [t for t, r in per_task.items() if r["n_pairs"] == 0]
    cleared = len(clearing) >= 2
    return {"cleared": cleared, "clearing_tasks": clearing, "no_data_tasks": no_data,
            "determined": cleared or not no_data,
            "fires": (not cleared) and not no_data}


def amendment_commit():
    """``(hash, committer ISO time)`` of the commit that added Amendment 1."""
    try:
        out = subprocess.run(["git", "-C", REPO, "log", "--diff-filter=A",
                              "--format=%H %cI", "--", AMENDMENT_1],
                             capture_output=True, text=True, check=True).stdout
        line = out.strip().splitlines()[-1].split()
        return line[0], line[1]
    except Exception:
        return None, None


def last_cell_end(out_dir):
    """Latest ``ended`` time over the cell_status.json files (operational data)."""
    latest = None
    cells_dir = os.path.join(out_dir, "cells")
    for name in (os.listdir(cells_dir) if os.path.isdir(cells_dir) else []):
        try:
            with open(os.path.join(cells_dir, name, "cell_status.json")) as fh:
                ended = json.load(fh).get("ended")
        except (OSError, ValueError):
            continue
        if ended and (latest is None or ended > latest):
            latest = ended
    if latest is None:
        return None
    return datetime.datetime.fromtimestamp(latest).astimezone().isoformat(timespec="seconds")


def l4_analysis(icm10, her10, icm20, her20, seeds):
    """L4 on the seeds present in all four series (seed -> best-of-acquired).

    The n_init 10 and n_init 20 differences must be computed on the SAME seeds,
    or a single missing cell changes the attainable p and can flip the category
    with identical per-seed differences.  No common seed -> undetermined.
    """
    common = sorted(set(seeds) & set(icm10) & set(her10) & set(icm20) & set(her20))
    d10 = signed_rank([icm10[s] - her10[s] for s in common])
    d20 = signed_rank([icm20[s] - her20[s] for s in common])
    for d, (a, b) in ((d10, (icm10, her10)), (d20, (icm20, her20))):
        d.update(seeds=common, seeds_missing=sorted(set(seeds) - set(common)),
                 diffs=[float(a[s] - b[s]) for s in common])
    if not common:
        return {"n_init_10": d10, "n_init_20_same_seeds": d20, "determined": False,
                "category_10": "no data", "category_20": "no data", "flips": None}
    c10, c20 = task_category(d10, "p"), task_category(d20, "p")
    return {"n_init_10": d10, "n_init_20_same_seeds": d20, "determined": True,
            "category_10": c10, "category_20": c20, "flips": c10 != c20}


def code_integrity(per_cell, manifest):
    """Same result code, GP-MOBO commit and imported versions for every cell."""
    import provenance
    heads = sorted({r["repo_head"] for r in per_cell if r["repo_head"]})
    ref_head = None
    if manifest and manifest.get("invocations"):
        ref_head = manifest["invocations"][0].get("repo_head")
    ref_head = ref_head or (heads[0] if heads else None)
    out = {"reference_head": ref_head, "cell_heads": heads, "differing_files": {},
           "gp_mobo_heads": sorted({r["gp_mobo_head"] for r in per_cell}),
           "cells_bench_yong_dirty": sum(bool(r["bench_yong_dirty"]) for r in per_cell),
           "versions_seen": sorted({r["versions_key"] for r in per_cell})}
    if ref_head:
        _, ref_pairs = provenance.fingerprint_at(ref_head)
        for h in heads:
            changed = provenance.diff(ref_pairs, provenance.fingerprint_at(h)[1])
            if changed:
                out["differing_files"][h] = changed
    out["ok"] = (not out["differing_files"] and len(out["versions_seen"]) <= 1 and
                 out["gp_mobo_heads"] in ([], [yc.GP_MOBO_COMMIT]))
    return out


def library_rank(value, library_scores):
    """Competition rank in the 10,000-row library table: 1 + #strictly better."""
    return int(1 + np.sum(library_scores > value))


# --------------------------------------------------------------------------
# Disk
# --------------------------------------------------------------------------

def discover(out_dir):
    cells_dir = os.path.join(out_dir, "cells")
    found, rejected = {}, {}
    if not os.path.isdir(cells_dir):
        return found, rejected
    for name in sorted(os.listdir(cells_dir)):
        d = os.path.join(cells_dir, name)
        try:
            with open(os.path.join(d, "run.json")) as fh:
                run = json.load(fh)
        except (OSError, ValueError):
            rejected[name] = "no readable run.json"
            continue
        if run.get("status") != "complete":
            rejected[name] = f"status {run.get('status')}"
            continue
        if run.get("pool_limit") is not None:
            rejected[name] = "pool-limited (gate/test run)"
            continue
        ev = os.path.join(d, "evaluated.csv")
        if not os.path.exists(ev):
            rejected[name] = "no evaluated.csv"
            continue
        with open(ev) as fh:
            rows = list(csv.DictReader(fh))
        if sum(1 for r in rows if r["is_initial"] == "0") != run["n_iterations"]:
            rejected[name] = "acquired-row count != n_iterations"
            continue
        if name != yc.cell_name(run["task"], run["arm"], run["seed"], run["n_init"]):
            rejected[name] = "directory name disagrees with run.json"
            continue
        found[name] = {"run": run, "rows": rows}
    return found, rejected


def rescore(found):
    by_task = {}
    for c in found.values():
        by_task.setdefault(c["run"]["task"], set()).update(r["smiles"] for r in c["rows"])
    scores = {}
    for task, smiles in by_task.items():
        smiles = sorted(smiles)
        vals = yc.score_scalar(task, smiles)
        scores[task] = dict(zip(smiles, vals.tolist()))
    return scores


# --------------------------------------------------------------------------

def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "nan"
    return f"{x:.{nd}f}"


def build(out_dir):
    manifest = None
    mpath = os.path.join(out_dir, "manifest.json")
    if os.path.exists(mpath):
        with open(mpath) as fh:
            manifest = json.load(fh)
    if manifest and manifest["config"].get("test_mode"):
        raise SystemExit("refusing to report a --test-mode sweep")

    found, rejected = discover(out_dir)
    planned = {c["name"]: c for c in (manifest["plan"] if manifest else [])}
    missing = sorted(set(planned) - set(found))
    unplanned = sorted(set(found) - set(planned))
    status = "COMPLETE" if manifest and not missing else "PARTIAL"

    scores = rescore(found)
    lines = yc.read_library_lines()[:yc.N_LIBRARY]
    lib_scores = {t: yc.score_scalar(t, lines) for t in TASK_LIST}

    per_cell = []
    worst_log = 0.0
    for name, c in sorted(found.items()):
        run, rows = c["run"], c["rows"]
        sc = scores[run["task"]]
        acq = [r for r in rows if r["is_initial"] == "0"]
        ini = [r for r in rows if r["is_initial"] == "1"]
        acq_vals = [sc[r["smiles"]] for r in acq]
        for r in rows:
            worst_log = max(worst_log, abs(float(r["mpo_logged"]) - sc[r["smiles"]]))
        best = float(max(acq_vals))
        best_smiles = acq[int(np.argmax(acq_vals))]["smiles"]
        per_cell.append({
            "cell": name, "tier": planned.get(name, {}).get("tier"),
            "task": run["task"], "arm": run["arm"], "seed": run["seed"],
            "n_init": run["n_init"], "best_acquired": best,
            "best_acquired_smiles": best_smiles,
            "library_rank": library_rank(best, lib_scores[run["task"]]),
            "best_start_excluded": float(max(sc[r["smiles"]] for r in ini)),
            "start_set_sha256": run["start_set_sha256"],
            "oracle_calls": run.get("oracle_calls"),
            "repo_head": run.get("repo_head"),
            "bench_yong_dirty": run.get("repo_dirty_bench_yong"),
            "gp_mobo_head": run.get("gp_mobo_head"),
            "versions_key": json.dumps({k: run["versions"].get(k) for k in
                                        ("python", "numpy", "scipy", "pandas", "rdkit")},
                                       sort_keys=True),
            "wall_clock_h": run["wall_clock_s"] / 3600,
            "peak_rss_gb": run["peak_rss_bytes"] / 1e9,
            "final_front_size": run["final_front_size"]})

    def series(task, arm, n_init, seeds=None):
        d = {r["seed"]: r for r in per_cell
             if r["task"] == task and r["arm"] == arm and r["n_init"] == n_init}
        if seeds is not None:
            d = {s: v for s, v in d.items() if s in seeds}
        return d

    def paired(task, a, b, n_init, seeds):
        A, B = series(task, a, n_init, seeds), series(task, b, n_init, seeds)
        common = sorted(set(A) & set(B))
        lost = sorted(set(seeds) - set(common))
        d = [A[s]["best_acquired"] - B[s]["best_acquired"] for s in common]
        res = signed_rank(d)
        res.update(seeds=common, seeds_missing=lost,
                   diffs=[float(x) for x in d])
        return res

    T1, T2 = yc.PREREG_TIER1_SEEDS, yc.PREREG_TIER2_SEEDS
    a_hash, a_time = amendment_commit()
    rep = {"sweep_status": status, "missing_cells": missing,
           "rejected_cells": rejected, "unplanned_cells": unplanned,
           "n_cells_found": len(found), "n_cells_planned": len(planned),
           "preregistration_commit": (manifest or {}).get("preregistration_commit"),
           "amendment_1_commit": a_hash, "amendment_1_committed": a_time,
           "sweep_last_cell_ended": last_cell_end(out_dir),
           "max_abs_logged_vs_rescored": worst_log}

    # ---- L1-REVISED, the floor (Amendment 1) ---------------------------------
    l1r = {t: paired(t, "MOGP-ICM@FP2", "RANDOM", 20, T1) for t in TASK_LIST}
    for t, a in zip(TASK_LIST, holm([l1r[t]["p"] for t in TASK_LIST])):
        l1r[t]["holm_p"] = a
    rep["L1_REVISED"] = dict(floor_revised(l1r), per_task=l1r,
                             provisional=status != "COMPLETE")

    # ---- L1-ORIGINAL (over-strict; superseded by Amendment 1) ---------------
    l1 = {}
    for t in TASK_LIST:
        icm = [v["best_acquired"] for v in series(t, "MOGP-ICM@FP2", 20, T1).values()]
        rnd = [v["best_acquired"] for v in series(t, "RANDOM", 20, T1).values()]
        if icm and rnd:
            med, p95 = float(np.median(icm)), float(np.percentile(rnd, 95))
            l1[t] = {"icm_median": med, "random_p95": p95, "n_icm": len(icm),
                     "n_random": len(rnd), "clears_floor": med > p95}
        else:
            l1[t] = {"icm_median": None, "random_p95": None, "n_icm": len(icm),
                     "n_random": len(rnd), "clears_floor": None}
    l1_fires = any(v["clears_floor"] is False for v in l1.values())
    rep["L1_ORIGINAL"] = {"per_task": l1, "fires": l1_fires, "superseded": True,
                          "undetermined": any(v["clears_floor"] is None
                                              for v in l1.values())}

    # ---- primary -----------------------------------------------------------
    prim = {t: paired(t, *PRIMARY, 20, T1) for t in TASK_LIST}
    adj = holm([prim[t]["p"] for t in TASK_LIST])
    for t, a in zip(TASK_LIST, adj):
        prim[t]["holm_p"] = a
    v, wins, losses = verdict(prim)
    rep["primary"] = {"per_task": prim, "verdict": v, "winning_tasks": wins,
                      "losing_tasks": losses,
                      "verdict_original_rule_superseded": verdict_original(prim)[0],
                      "no_data_tasks": [t for t in TASK_LIST if prim[t]["n_pairs"] == 0],
                      "provisional": status != "COMPLETE"}

    # ---- L2 interleave -------------------------------------------------------
    l2 = {}
    for t in TASK_LIST:
        l2[t] = {}
        for arm, n_init, seeds in (("HER@FP2", 20, T1), ("HER-PUB", 20, T2),
                                   ("MOGP-ICM@FP2", 20, T1), ("MOGP-IND@FP2", 20, T1),
                                   ("RANDOM", 20, T1)):
            s = series(t, arm, n_init, seeds)
            ranks = [s[k]["library_rank"] for k in sorted(s)]
            l2[t][arm] = {"seeds": sorted(s), "ranks": ranks,
                          "median_rank": float(np.median(ranks)) if ranks else None}
    her_meds = {t: l2[t]["HER@FP2"]["median_rank"] for t in TASK_LIST}
    rep["L2"] = {"per_task": l2, "her_median_rank": her_meds,
                 "fires_on": [t for t, m in her_meds.items()
                              if m is not None and m <= 3]}

    # ---- L3 coregionalization ---------------------------------------------
    l3 = {t: paired(t, "MOGP-ICM@FP2", "MOGP-IND@FP2", 20, T1) for t in TASK_LIST}
    for t, a in zip(TASK_LIST, holm([l3[t]["p"] for t in TASK_LIST])):
        l3[t]["holm_p"] = a
    l3v, l3w, _ = verdict(l3)
    rep["L3"] = {"per_task": l3, "icm_beats_ind": l3v == "WIN", "verdict_shape": l3v,
                 "no_data_tasks": [t for t in TASK_LIST if l3[t]["n_pairs"] == 0]}

    # ---- L4 sensitivity ------------------------------------------------------
    s6 = yc.PREREG_TIER3_SEEDS

    def best_by_seed(arm, n_init):
        return {s: v["best_acquired"]
                for s, v in series("fexofenadine", arm, n_init, s6).items()}
    rep["L4"] = l4_analysis(best_by_seed("MOGP-ICM@FP2", 10), best_by_seed("HER@FP2", 10),
                            best_by_seed("MOGP-ICM@FP2", 20), best_by_seed("HER@FP2", 20),
                            s6)

    # ---- reference arm -----------------------------------------------------
    rep["reference_HER-PUB_minus_HER@FP2"] = {
        t: paired(t, "HER-PUB", "HER@FP2", 20, T2) for t in TASK_LIST}
    rep["primary_vs_HER-PUB_context"] = {
        t: paired(t, "MOGP-ICM@FP2", "HER-PUB", 20, T2) for t in TASK_LIST}

    # ---- integrity ---------------------------------------------------------
    start_ok = True
    for t in TASK_LIST:
        for n_init in (20, 10):
            for seed in range(10):
                shas = {r["start_set_sha256"] for r in per_cell
                        if r["task"] == t and r["seed"] == seed and r["n_init"] == n_init}
                start_ok &= len(shas) <= 1
    rep["integrity"] = {"start_set_shared_across_arms": start_ok,
                        "max_abs_logged_vs_rescored": worst_log,
                        "code": code_integrity(per_cell, manifest)}
    rep["per_cell"] = per_cell
    return rep


def render(rep):
    o = io.StringIO()
    w = o.write
    w("# GP-MOBO head-to-head: report\n\n")
    w(f"**Sweep status: {rep['sweep_status']}** ({rep['n_cells_found']} complete "
      f"of {rep['n_cells_planned']} planned).  Pre-registration commit: "
      f"`{rep['preregistration_commit']}`.  Built from files on disk; every "
      f"molecule re-scored with her oracle (max |logged - rescored| = "
      f"{rep['max_abs_logged_vs_rescored']:.2e}).\n\n")
    a_hash, a_time, last = (rep["amendment_1_commit"], rep["amendment_1_committed"],
                            rep["sweep_last_cell_ended"])
    before = None
    if a_time and last:
        before = (datetime.datetime.fromisoformat(a_time) <
                  datetime.datetime.fromisoformat(last))
    w(f"Rules: PREREGISTRATION.md as amended by Amendment 1, commit `{a_hash}` "
      f"({a_time}); the sweep's last cell ended {last}"
      + ("" if before is None else
         f" (amendment committed {'BEFORE' if before else 'AFTER'} the sweep "
         f"finished)") + ".  Amendment 1 replaced L1 and LOSS; both originals "
      "are still printed, labelled as superseded.\n\n")
    if not a_hash:
        w("**WARNING: Amendment 1 not found in git; the amended rules are applied "
          "anyway, but their provenance is not verified.**\n\n")
    ci = rep["integrity"]["code"]
    if not ci["ok"] or not rep["integrity"]["start_set_shared_across_arms"]:
        w("**INTEGRITY FAILURE: the cells were not all produced by the same result "
          f"code / environment / starting sets.** {json.dumps(ci)}\n\n")
    if rep["missing_cells"]:
        w(f"Missing cells ({len(rep['missing_cells'])}): "
          + ", ".join(f"`{m}`" for m in rep["missing_cells"]) + "\n\n")
    if rep["rejected_cells"]:
        w("Rejected cell directories: " + ", ".join(
            f"`{k}` ({v})" for k, v in rep["rejected_cells"].items()) + "\n\n")

    L1R = rep["L1_REVISED"]
    w("## L1-REVISED, THE FLOOR (Amendment 1; reported first)\n\n"
      "Per task, paired two-sided Wilcoxon of MOGP-ICM@FP2 minus RANDOM over seeds "
      "0-9, Holm across the three tasks.  The floor is cleared if ICM is "
      "significantly better (Holm p < 0.05, median > 0, >= 6 non-tied seeds) on at "
      "least 2 of 3 tasks.\n\n"
      "| task | pairs | non-tied | median diff | p | Holm p | +/-/0 | missing seeds | note |\n"
      "|---|---|---|---|---|---|---|---|---|\n")
    for t, r in L1R["per_task"].items():
        note = ("NO DATA" if r["n_pairs"] == 0 else
                "INCONCLUSIVE BY CONSTRUCTION" if r["inconclusive_by_construction"] else "")
        w(f"| {t} | {r['n_pairs']} | {r['n_nontied']} | {r['median_diff']:+.5f} | "
          f"{r['p']:.4f} | {r['holm_p']:.4f} | {r['n_positive']}/{r['n_negative']}/"
          f"{r['n_zero']} | {r['seeds_missing']} | {note} |\n")
    if L1R["cleared"]:
        l1_text = ("**L1-REVISED: floor cleared** (significantly better than RANDOM "
                   "on " + ", ".join(L1R["clearing_tasks"]) + ")")
    elif L1R["fires"]:
        l1_text = "**L1-REVISED FIRES: published as a negative**"
    else:
        l1_text = ("L1-REVISED not yet determined: no pairs on "
                   + ", ".join(L1R["no_data_tasks"]))
    w("\n" + l1_text + (" (PROVISIONAL: sweep incomplete)" if L1R["provisional"] else "")
      + "\n\n")

    L1O = rep["L1_ORIGINAL"]
    w("### L1-ORIGINAL (over-strict; superseded by Amendment 1)\n\n"
      "The original rule: MOGP-ICM's median best-of-acquired must exceed RANDOM's "
      "95th percentile on every task.  Superseded because it is not a floor test: "
      "in Amendment 1's power analysis it passes only 31% of the time even when "
      "ICM is 10x as efficient as random search.  Printed so nothing is hidden.\n\n"
      "| task | ICM median | RANDOM p95 | clears |\n|---|---|---|---|\n")
    for t, v in L1O["per_task"].items():
        w(f"| {t} | {fmt(v['icm_median'])} | {fmt(v['random_p95'])} | "
          f"{v['clears_floor']} (n={v['n_icm']}/{v['n_random']}) |\n")
    w(f"\nL1-ORIGINAL (superseded) "
      f"{'would fire' if L1O['fires'] else 'would not fire'}"
      + (" (undetermined on some task: missing cells)" if L1O["undetermined"] else "")
      + ".\n\n")

    P = rep["primary"]
    w("## Primary endpoint: MOGP-ICM@FP2 minus HER@FP2, best-of-acquired, seeds 0-9\n\n")
    w("| task | pairs | non-tied | median diff | W | p | Holm p | missing seeds | note |\n"
      "|---|---|---|---|---|---|---|---|---|\n")
    for t, r in P["per_task"].items():
        note = ("NO DATA" if r["n_pairs"] == 0 else
                "INCONCLUSIVE BY CONSTRUCTION" if r["inconclusive_by_construction"] else "")
        w(f"| {t} | {r['n_pairs']} | {r['n_nontied']} | {r['median_diff']:+.5f} | "
          f"{fmt(r['statistic'], 1)} | {r['p']:.4f} | {r['holm_p']:.4f} | "
          f"{r['seeds_missing']} | {note} |\n")
    w(f"\n**Verdict: {P['verdict']}**" + (" (PROVISIONAL: sweep incomplete)"
                                          if P["provisional"] else "") + "\n\n")
    if P["no_data_tasks"]:
        w(f"No complete pairs yet on: {', '.join(P['no_data_tasks'])} (they count "
          f"towards neither WIN nor LOSS).\n\n")
    w("Rules (Amendment 1): WIN = Holm p < 0.05 with a positive median on >= 2 of 3 "
      "tasks; LOSS = Holm p < 0.05 with a negative median on >= 2 of 3 tasks; "
      "anything else, a negative median without significance included, is "
      "INCONCLUSIVE.  At 10 seeds the design has 80% power only for about a 6.5x "
      "difference in sample efficiency (Amendment 1, section 3), so INCONCLUSIVE "
      "must not be read as equivalence.  Under the superseded original LOSS rule "
      f"(median <= 0 on >= 2 of 3 tasks, no test) the verdict would read: "
      f"{P['verdict_original_rule_superseded']}.\n\n")

    w("## L2 THE INTERLEAVE: library rank of each arm's best acquired molecule\n\n"
      "Rank in the 10,000-row library table (the table Tables 8.4-8.6 are the "
      "top of), 1 = the library maximum.  Every seed's rank is printed.\n\n")
    for t, arms in rep["L2"]["per_task"].items():
        w(f"**{t}**\n\n| arm | median | ranks by seed |\n|---|---|---|\n")
        for arm, v in arms.items():
            w(f"| {arm} | {v['median_rank']} | {v['ranks']} (seeds {v['seeds']}) |\n")
        w("\n")
    for t, m in rep["L2"]["her_median_rank"].items():
        if m is not None and m <= 3:
            w(f"**L2 FIRES on {t}: HER@FP2 median rank = {m:g}**\n\n")
    if not rep["L2"]["fires_on"]:
        w(f"L2 does not fire: HER@FP2 median ranks {rep['L2']['her_median_rank']}\n\n")

    w("## L3 THE EXPECTED NULL: MOGP-ICM minus MOGP-IND\n\n"
      "| task | median diff | p | Holm p | +/-/0 |\n|---|---|---|---|---|\n")
    for t, r in rep["L3"]["per_task"].items():
        w(f"| {t} | {r['median_diff']:+.5f} | {r['p']:.4f} | {r['holm_p']:.4f} | "
          f"{r['n_positive']}/{r['n_negative']}/{r['n_zero']} |\n")
    if rep["L3"]["icm_beats_ind"]:
        l3_text = "**ICM beats IND: a surprise to explain, not a victory to bank.**"
    elif rep["L3"]["no_data_tasks"]:
        l3_text = ("L3 not yet determined: no ICM-IND pairs on "
                   + ", ".join(rep["L3"]["no_data_tasks"]) + ".")
    else:
        l3_text = "No coregionalization benefit, as expected."
    w("\n" + l3_text + "\n\n")

    L4 = rep["L4"]
    w("## L4 SENSITIVITY: n_init 10 vs 20 (fexofenadine, seeds 0-5)\n\n")
    if not L4["determined"]:
        w("L4 undetermined: no seed has all four cells (Tier 3 and Tier 1, both "
          "arms).\n\n")
    else:
        w(f"Seeds used (common to all four series): {L4['n_init_10']['seeds']}; "
          f"missing: {L4['n_init_10']['seeds_missing']}\n\n"
          f"- n_init 10: median {L4['n_init_10']['median_diff']:+.5f}, p "
          f"{L4['n_init_10']['p']:.4f}, {L4['category_10']}\n"
          f"- n_init 20: median {L4['n_init_20_same_seeds']['median_diff']:+.5f}, p "
          f"{L4['n_init_20_same_seeds']['p']:.4f}, {L4['category_20']}\n\n"
          + ("**L4 FIRES: the comparison is sensitive to a number her own sources "
             "disagree about (her main-branch driver uses 20, her GuacaMol "
             "drivers and prose use 10).**" if L4["flips"] else
             "L4 does not fire.") + "\n\n")

    w("## Reference arm: HER-PUB minus HER@FP2 (seeds 0-5)\n\n"
      "| task | median diff | p |\n|---|---|---|\n")
    for t, r in rep["reference_HER-PUB_minus_HER@FP2"].items():
        w(f"| {t} | {r['median_diff']:+.5f} | {r['p']:.4f} |\n")

    w("\n## Per-arm best-of-acquired (starting set excluded)\n\n"
      "| task | arm | n_init | n | median | min | max | start-set best (excluded) |\n"
      "|---|---|---|---|---|---|---|---|\n")
    groups = {}
    for r in rep["per_cell"]:
        groups.setdefault((r["task"], r["arm"], r["n_init"]), []).append(r)
    for (t, a, ni), rs in sorted(groups.items()):
        b = [r["best_acquired"] for r in rs]
        s = [r["best_start_excluded"] for r in rs]
        w(f"| {t} | {a} | {ni} | {len(b)} | {np.median(b):.4f} | {min(b):.4f} | "
          f"{max(b):.4f} | {np.median(s):.4f} |\n")

    w("\n## Oracle budget, her protocol, and modifications\n\n"
      "The 1,040-evaluation recipe exists only inside this benchmark, because the "
      "HER arms run her main-branch code (`ehvi_mc.py`), which scores a "
      "1,000-molecule hold-out block to set its GP hyperparameters; that was our "
      "choice.  Her published protocol used 10 starting molecules, fixed "
      "hyperparameters (mean 0, amplitude 1, noise 1e-4) and 30 oracle evaluations "
      "per run, fewer than our 40, and **NO ARM IN THIS SWEEP REPRODUCES IT**.  "
      "Results against HER@FP2 and HER-PUB are results against her code under her "
      "main-branch recipe, not against her published method (bench_yong/"
      "HER_CONFIG.md; erratum in PREREGISTRATION_AMENDMENT_1.md).  "
      "Modifications to her code: M1 import stand-ins "
      "(rdkit.six, utils.utils_final), M2 seeding, M3 batched predictions (if G3 "
      "passed), M4 evaluator binding for amlodipine/perindopril; HER@FP2 swaps "
      "her fingerprint by definition.  The thesis prints Fexofenadine widths 2 "
      "and 2; her code uses 10 and 1, and the code is used.\n\n")
    w(f"Integrity: starting set shared across arms = "
      f"{rep['integrity']['start_set_shared_across_arms']}; result code identical "
      f"across cells = {not ci['differing_files']} (reference head "
      f"`{ci['reference_head']}`, cell heads {ci['cell_heads']}); GP-MOBO "
      f"{ci['gp_mobo_heads']}; environments {len(ci['versions_seen'])}; cells "
      f"launched while bench_yong/ had uncommitted edits: "
      f"{ci['cells_bench_yong_dirty']}.\n")
    return o.getvalue()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--output-dir", required=True)
    args = p.parse_args(argv)
    out_dir = os.path.abspath(args.output_dir)
    rep = build(out_dir)
    rdir = os.path.join(out_dir, "report")
    os.makedirs(rdir, exist_ok=True)
    md = render(rep)
    yc.write_text_atomic(os.path.join(rdir, "report.md"), md)
    yc.write_json_atomic(os.path.join(rdir, "report.json"), rep)
    with open(os.path.join(rdir, "per_cell.csv"), "w") as fh:
        if rep["per_cell"]:
            wr = csv.DictWriter(fh, fieldnames=list(rep["per_cell"][0]),
                                lineterminator="\n")
            wr.writeheader()
            wr.writerows(rep["per_cell"])
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
