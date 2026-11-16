"""F6b: is the Pareto front enriched for known antifolate chemistry? Four panels.

(a) THE RESULT. Enrichment factor by method on three hit-set definitions. EF is
    a rate ratio, so EF=1 is a random draw of the same size and the dashed line
    is the null. The broad pharmacophore class and the narrow close-analog set
    are shown together because the choice of hit set is contestable and the two
    give different magnitudes: EF 2.8 and EF 17.7.
(b) THE CONSISTENCY. Hits per seed, ten seeds as replicates. This is what the
    union-of-fronts test hides: MOGP recovers analogs in 10/10 seeds, GPMOBO in
    5/10, Greedy never.
(c) WHERE THE ENRICHMENT ENTERS. The acquisition, which chooses what to dock,
    against the Pareto filter, which chooses what to keep. The acquisition does
    almost all of the work.
(d) THE CAVEAT. The recovered analogs on the potency/selectivity plane. They are
    potent, and they are not selective: both of the molecules MOGP recovers in
    every seed sit on the wrong side of the selectivity line.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis_scripts"))
from antifolate_enrichment import (                              # noqa: E402
    load_library, similarity_matrix, substructure_masks, load_campaign,
    campaign_scores, perseed_mc_test, METHODS, SEEDS, SMARTS_CORES)

COL = {"GPMOBO": "#4C72B0", "MOGP (ours)": "#C44E52", "Greedy": "#8C8C8C"}
SHORT = {"GPMOBO": "GPMOBO", "MOGP (ours)": "MOGP\n(ours)", "Greedy": "Greedy"}

lib_fp, lib_smiles, mols, heavy = load_library()
n_lib = len(lib_smiles)
row_of = {s: i for i, s in enumerate(lib_smiles)}
sims = similarity_matrix(lib_fp)
best = sims.max(axis=0)
masks = substructure_masks(mols)
core = masks["2,4-diaminopyrimidine"] | masks["2,4-diamino-dihydrotriazine"]
arms = load_campaign(row_of)
scored = campaign_scores()

HIT_SETS = [("pharmacophore\ncore", core),
            ("core and\nT>=0.3", core & (best >= 0.30)),
            ("close analog\nT>=0.4", best >= 0.40)]
rng = np.random.default_rng(20261116)
res = {}
for label, hit in HIT_SETS:
    for m, lab in METHODS:
        fronts = [arms[(m, s)]["front_idx"] for s in SEEDS if (m, s) in arms]
        res[(label, lab)] = perseed_mc_test(hit, fronts, n_lib, 20000, rng)

fig, ax = plt.subplots(2, 2, figsize=(13.2, 9.4))
fig.suptitle("F6b  The Pareto front is enriched for known antifolate chemistry, "
             "and the enrichment is in the acquisition",
             fontsize=13.5, fontweight="bold", y=0.975)

# ---------------------------------------------------------------- (a)
a = ax[0, 0]
w, xs = 0.26, np.arange(len(HIT_SETS))
for j, (m, lab) in enumerate(METHODS):
    ef = [max(res[(l, lab)]["ef"], 0.04) for l, _ in HIT_SETS]
    p = [res[(l, lab)]["p"] for l, _ in HIT_SETS]
    b = a.bar(xs + (j - 1) * w, ef, w, color=COL[lab], label=lab,
              edgecolor="white", linewidth=0.8)
    for r, v, pv in zip(b, ef, p):
        if v <= 0.05:            # EF = 0: no bar to annotate
            a.text(r.get_x() + r.get_width() / 2, 0.055, "0 hits",
                   ha="center", fontsize=8, color="#8C8C8C", rotation=90,
                   va="bottom")
            continue
        txt = "***" if pv < 0.001 else "**" if pv < 0.01 else \
              "*" if pv < 0.05 else "n.s."
        a.text(r.get_x() + r.get_width() / 2, v * 1.12, txt, ha="center",
               fontsize=9, color="#333" if pv < 0.05 else "#999")
a.axhline(1, color="k", ls="--", lw=1.2, zorder=0)
a.text(2.42, 0.80, "EF = 1: a random draw\nof the same size", fontsize=8.5,
       color="#555", ha="right", va="top")
a.set_yscale("log")
a.set_ylim(0.04, 60)
a.set_xticks(xs)
a.set_xticklabels([l for l, _ in HIT_SETS], fontsize=9)
a.set_ylabel("enrichment factor (dimensionless rate ratio)", fontsize=10)
a.set_title("(a) Enrichment of the per-seed Pareto front, 10 seeds",
            fontsize=11, loc="left")
a.legend(fontsize=9, loc="upper left", framealpha=0.95)

# ---------------------------------------------------------------- (b)
b = ax[0, 1]
hit = best >= 0.40
exp = None
for j, (m, lab) in enumerate(METHODS):
    k = np.array([int(hit[arms[(m, s)]["front_idx"]].sum()) for s in SEEDS])
    ns = np.array([len(arms[(m, s)]["front_idx"]) for s in SEEDS])
    exp = ns.mean() * hit.sum() / n_lib
    b.scatter(np.full(10, j) + rng.uniform(-.13, .13, 10), k, s=62,
              color=COL[lab], alpha=0.85, edgecolor="white", linewidth=1,
              zorder=3)
    b.plot([j - .27, j + .27], [k.mean()] * 2, color=COL[lab], lw=2.6, zorder=4)
    b.text(j, -0.42, f"{int((k > 0).sum())}/10 seeds", ha="center", fontsize=9,
           color=COL[lab], fontweight="bold")
b.axhline(exp, color="k", ls="--", lw=1.2)
b.text(2.42, exp + 0.07, f"expected by chance ({exp:.2f})", fontsize=8.5,
       color="#555", ha="right")
b.set_xticks(range(3))
b.set_xticklabels([SHORT[l] for _, l in METHODS], fontsize=9.5)
b.set_ylim(-0.75, 3.4)
b.set_ylabel("close analogs on the front (count)", fontsize=10)
b.set_title("(b) Per seed. Bars are means; 10 seeds, close-analog set",
            fontsize=11, loc="left")

# ---------------------------------------------------------------- (c)
c = ax[1, 0]
h = int(hit.sum())
xs2 = np.arange(3)
for j, (m, lab) in enumerate(METHODS):
    ev = np.concatenate([arms[(m, s)]["eval_idx"] for s in SEEDS])
    fr = np.concatenate([arms[(m, s)]["front_idx"] for s in SEEDS])
    ke, kf = int(hit[ev].sum()), int(hit[fr].sum())
    ef_s = (ke / len(ev)) / (h / n_lib)
    ef_f = ((kf / len(fr)) / (ke / len(ev))) if ke else 0.0
    c.bar(j - 0.17, max(ef_s, 0.04), 0.32, color=COL[lab],
          edgecolor="white", linewidth=0.8)
    c.bar(j + 0.17, max(ef_f, 0.04), 0.32, color=COL[lab], alpha=0.42,
          edgecolor="white", linewidth=0.8, hatch="///")
    c.text(j - 0.17, max(ef_s, 0.05) * 1.1, f"{ef_s:.1f}", ha="center",
           fontsize=8.5)
    if ke:
        c.text(j + 0.17, max(ef_f, 0.05) * 1.1, f"{ef_f:.2f}", ha="center",
               fontsize=8.5)
c.axhline(1, color="k", ls="--", lw=1.2, zorder=0)
c.set_yscale("log")
c.set_ylim(0.04, 40)
c.set_xticks(xs2)
c.set_xticklabels([SHORT[l] for _, l in METHODS], fontsize=9.5)
c.set_ylabel("enrichment factor", fontsize=10)
c.set_title("(c) Solid: acquisition (what to dock).  Hatched: Pareto filter "
            "(what to keep)", fontsize=11, loc="left")
c.text(0.98, 0.95, "the acquisition does\nalmost all of the work",
       transform=c.transAxes, ha="right", va="top", fontsize=9, color="#444",
       bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#ccc"))

# ---------------------------------------------------------------- (d)
d = ax[1, 1]
allfr = sorted({int(i) for k in arms for i in arms[k]["front_idx"]})
pf, si, isa = [], [], []
for i in allfr:
    sc = scored.get(lib_smiles[i], {})
    if "PfDHFR" in sc and "hDHFR" in sc:
        pf.append(sc["PfDHFR"])
        si.append(sc["hDHFR"] - sc["PfDHFR"])
        isa.append(bool(hit[i]))
pf, si, isa = np.array(pf), np.array(si), np.array(isa)
d.scatter(pf[~isa], si[~isa], s=11, color="#C8C8C8", alpha=0.55,
          edgecolor="none", label=f"other front molecules (n={int((~isa).sum())})")
d.axhline(0, color="k", lw=1.1)
d.fill_between([-13, -4], 0, 4.2, color="#4C72B0", alpha=0.06, zorder=0)
d.text(-12.75, 3.55, "parasite-selective\n(binds PfDHFR more tightly)",
       fontsize=8.5, color="#3A5F8F", va="top")
d.text(-12.75, -3.55, "human-selective: the wrong side", fontsize=8.5,
       color="#A03A3A", va="bottom")
for i in sorted({int(i) for k in arms for i in arms[k]["front_idx"]
                 if hit[i]}, key=lambda i: -best[i]):
    sc = scored.get(lib_smiles[i], {})
    if "PfDHFR" not in sc or "hDHFR" not in sc:
        continue
    x, y = sc["PfDHFR"], sc["hDHFR"] - sc["PfDHFR"]
    inm = sum(1 for s in SEEDS if i in set(arms[("mogp", s)]["front_idx"].tolist()))
    d.scatter([x], [y], s=165 if inm == 10 else 95,
              marker="*" if inm == 10 else "o",
              color="#C44E52" if inm == 10 else "#4C72B0",
              edgecolor="k", linewidth=0.9, zorder=5)
    off = {0.732: (-46, 6), 0.610: (9, -14), 0.575: (8, 10),
           0.561: (-52, -16), 0.404: (9, 8)}.get(round(best[i], 3), (8, 8))
    d.annotate(f"T={best[i]:.2f}", (x, y), textcoords="offset points",
               xytext=off, fontsize=8.5,
               bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none",
                         alpha=0.8))
d.scatter([], [], marker="*", s=165, color="#C44E52", edgecolor="k",
          label="recovered by MOGP in 10/10 seeds")
d.scatter([], [], marker="o", s=95, color="#4C72B0", edgecolor="k",
          label="recovered by GPMOBO only")
d.set_xlim(-13, -4)
d.set_ylim(-4.2, 4.2)
d.set_xlabel("PfDHFR docking affinity (kcal/mol, more negative = tighter)",
             fontsize=10)
d.set_ylabel("Selectivity Index = hDHFR - PfDHFR (kcal/mol)", fontsize=10)
d.set_title("(d) The recovered analogs are potent but NOT selective",
            fontsize=11, loc="left")
d.legend(fontsize=8.5, loc="lower right", framealpha=0.95)

for a_ in ax.ravel():
    a_.spines[["top", "right"]].set_visible(False)
    a_.grid(axis="y", alpha=0.25, lw=0.6)
    a_.set_axisbelow(True)

fig.text(0.5, 0.040,
         "Hit sets: 'pharmacophore core' = matches 2,4-diaminopyrimidine or "
         "2,4-diamino-dihydrotriazine by SMARTS (231 of 29,678 library "
         "molecules, 0.78%).\n'close analog' = Tanimoto >= 0.4 to "
         "pyrimethamine, cycloguanil, WR99210 or P218 (20 molecules, 0.067%). "
         "Tanimoto on 2048-bit Morgan fingerprints, radius 2.\nNull: each "
         "seed's front is a uniform draw of its own size from the library, "
         "20,000 Monte-Carlo draws.  *** p<0.001, ** p<0.01, * p<0.05.\n"
         "All hits were chosen by the optimizer; none came from the "
         "40-molecule random initial design. The result survives a "
         "heavy-atom-matched null.",
         ha="center", va="bottom", fontsize=8.1, color="#444", linespacing=1.5)
fig.tight_layout(rect=[0, 0.098, 1, 0.955])
out = os.path.join(ROOT, "F6b_antifolate_enrichment.png")
fig.savefig(out, dpi=200)
print("wrote", out)
