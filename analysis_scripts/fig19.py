"""F19: the correlation floor. Three panels, one per pre-registered clause group.

(a) THE RESULT. ICM advantage over independent per-task GPs against the task
    correlation rho, at two co-located-label budgets. The zero crossing is the
    floor. Holm-corrected significance is marked, because the point estimate
    crosses zero well before the effect is demonstrable.
(b) THE MECHANISM, clause 3. Across-repeat sd of the LEARNED task correlation.
    It rises monotonically as rho falls, which is B losing identifiability --
    the same mechanism F17 found along the co-located-count axis.
(c) THE TWO CONTROLS, clauses 2 and 4. Sign symmetry, and the synthetic task
    against the real hDHFR column at the same correlation.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = pd.read_csv(os.path.join(ROOT, "analysis_scripts/correlation_floor.csv"))
RHOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.711, 0.8, 0.9, 0.95]
KS = [8, 22]
COL = {8: "#C44E52", 22: "#4C72B0"}
PCT = {8: "3.6%", 22: "9.8%"}


def paired(label, k):
    s = d[(d.label == label) & (d.k == k)]
    a = s[s.model == "ICM"].sort_values("rep")
    b = s[s.model == "independent"].sort_values("rep")
    return b.rmse.values - a.rmse.values, a


def holm(ps):
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    out, run = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        run = min(1.0, max(run, (m - rank) * ps[i]))
        out[i] = run
    return out


fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.0))

# ---------------- (a) the floor ----------------
floors = {}
for k in KS:
    adv, se, praw, sdb = [], [], [], []
    for r in RHOS:
        dd, a = paired(f"rho{r:+.3f}", k)
        adv.append(dd.mean()); se.append(dd.std(ddof=1) / np.sqrt(len(dd)))
        praw.append(wilcoxon(dd).pvalue); sdb.append(a.rho_B.std())
    ph = holm(praw)
    adv = np.array(adv); se = np.array(se)
    ax[0].errorbar(RHOS, adv, yerr=se, color=COL[k], marker="o", ms=5, lw=1.8,
                   capsize=3, label=f"k={k} co-located ({PCT[k]} of training set)")
    sig = [i for i in range(len(RHOS)) if ph[i] < 0.05]
    ax[0].scatter([RHOS[i] for i in sig], [adv[i] for i in sig], s=130,
                  facecolors="none", edgecolors=COL[k], lw=2.0, zorder=5)
    for (r1, a1), (r2, a2) in zip(list(zip(RHOS, adv)), list(zip(RHOS, adv))[1:]):
        if a1 < 0 <= a2:
            floors[k] = r1 + (r2 - r1) * (-a1) / (a2 - a1)
    dr, _ = paired("real-hDHFR", k)
    ax[0].scatter([0.711], [dr.mean()], marker="*", s=320, color=COL[k],
                  edgecolors="k", lw=0.7, zorder=6)

ax[0].axhline(0, color="k", lw=1.0, ls="--")
for k in KS:
    ax[0].axvline(floors[k], color=COL[k], lw=1.2, ls=":", alpha=0.8)
    ax[0].text(floors[k] - 0.016, 0.045, f"floor {floors[k]:.2f}", color=COL[k],
               fontsize=9, fontweight="bold", rotation=90, va="center", ha="center")
ax[0].axvspan(0.05, 0.5, color="0.85", alpha=0.45, zorder=0)
ax[0].text(0.235, 0.145, "no demonstrated benefit\n(Holm p > 0.05 either way)",
           ha="center", fontsize=8.5, color="0.30", style="italic")
ax[0].set_xlabel("task correlation rho  (dimensionless, Pearson across molecules)")
ax[0].set_ylabel("ICM advantage, kcal/mol RMSE\n(positive = ICM predicts task 2 better)")
ax[0].set_title("(a) The floor. Monotone in rho, Spearman = +1.000 at both budgets.",
                fontsize=10.5, loc="left")
ax[0].legend(loc="upper left", fontsize=8.5, framealpha=0.95)
ax[0].text(0.985, 0.03, "open ring = Holm p < 0.05\nstar = REAL hDHFR column",
           transform=ax[0].transAxes, ha="right", va="bottom", fontsize=8,
           color="0.25", bbox=dict(fc="white", ec="0.8", alpha=0.9))
ax[0].set_xlim(0.04, 1.0); ax[0].set_ylim(-0.11, 0.26)

# ---------------- (b) identifiability ----------------
for k in KS:
    sdb = [paired(f"rho{r:+.3f}", k)[1].rho_B.std() for r in RHOS]
    ax[1].plot(RHOS, sdb, color=COL[k], marker="o", ms=5, lw=1.8,
               label=f"k={k} ({PCT[k]})")
    dm, am = paired("rho-0.711", k)
    ax[1].scatter([0.711], [am.rho_B.std()], marker="v", s=90, color=COL[k],
                  edgecolors="k", lw=0.7, zorder=5)
ax[1].set_yscale("log")
ax[1].set_xlabel("task correlation rho")
ax[1].set_ylabel("sd of the LEARNED task correlation\nacross 20 repeats (dimensionless)")
ax[1].set_title("(b) Mechanism. Weak coupling is a flat direction in the likelihood.",
                fontsize=10.5, loc="left")
ax[1].legend(loc="lower left", fontsize=8.5, framealpha=0.95)
ax[1].set_ylim(0.025, 0.62)
ax[1].text(0.40, 0.040,
           "Spearman(rho, sd) = -1.000 at both budgets\n"
           "triangle = rho = -0.711: a NEGATIVE coupling is\n"
           "2.3x less identifiable at k=8, yet pays the same",
           fontsize=8.2, color="0.25")
ax[1].set_xlim(0.04, 1.0)

# ---------------- (c) the controls ----------------
labels, vals, errs, cols, hatch = [], [], [], [], []
for k in KS:
    for lab, cell in (("rho = +0.711", f"rho+0.711"), ("rho = -0.711", "rho-0.711"),
                      ("REAL hDHFR", "real-hDHFR")):
        dd, _ = paired(cell, k)
        labels.append(lab.replace("rho = ", "").replace("REAL hDHFR", "0.711\nREAL"))
        vals.append(dd.mean()); errs.append(dd.std(ddof=1) / np.sqrt(len(dd)))
        cols.append(COL[k]); hatch.append("" if "REAL" not in lab else "///")
x = np.arange(len(vals))
bars = ax[2].bar(x, vals, yerr=errs, capsize=4, color=cols, edgecolor="k", lw=0.7)
for b, h in zip(bars, hatch):
    b.set_hatch(h)
ax[2].set_xticks(x); ax[2].set_xticklabels(labels, fontsize=8.5)
for i, k in enumerate(KS):
    ax[2].text(3 * i + 1, -0.030, f"k = {k}  ({PCT[k]} co-located)", ha="center",
               fontsize=9, color=COL[k], fontweight="bold")
    if i:
        ax[2].axvline(3 * i - 0.5, color="0.75", lw=0.8, ls=":")
ax[2].axhline(0, color="k", lw=1.0)
ax[2].set_ylabel("ICM advantage, kcal/mol RMSE")
ax[2].set_title("(c) Both controls PASS. Sign-symmetric; synthetic matches real.",
                fontsize=10.5, loc="left")
ax[2].text(0.5, 0.97,
           "clause 2  sign gap +0.020 at BOTH budgets, paired p = 0.73 / 0.50\n"
           "clause 4  real minus synthetic +0.025 (k=8) and -0.017 (k=22),\n"
           "          p = 0.33 / 0.19 -- opposite signs, neither significant,\n"
           "          so NEITHER of my two predicted bias directions holds",
           transform=ax[2].transAxes, ha="center", va="top", fontsize=8.0,
           color="0.2", bbox=dict(fc="white", ec="0.7", alpha=0.95))
ax[2].set_ylim(-0.045, 0.215)
ax[2].spines["bottom"].set_position(("data", 0.0))

for a in ax:
    a.spines[["top", "right"]].set_visible(False)
    a.grid(axis="y", alpha=0.25, lw=0.6)

fig.suptitle(
    "F19  The correlation floor: how similar must two docking targets be before "
    "sharing information between them helps?",
    fontsize=12.5, y=1.005)
fig.tight_layout()
out = os.path.join(ROOT, "F19_correlation_floor.png")
fig.savefig(out, dpi=200, bbox_inches="tight")
print(f"wrote {out}")
print("floors:", {k: round(v, 3) for k, v in floors.items()})
