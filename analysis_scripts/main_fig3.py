"""MAIN FIGURE 3 -- when sharing information between targets is worth it.

One statement covers the whole arc: coregionalization is a MITIGATION for
missing labels, not a reason to create them. Four panels, four experiments,
each recomputed from the committed CSVs.

  a  F12. Nothing missing -> useless. 10 paired seeds, every endpoint null.
  b  F13. Labels already missing -> genuinely helps, monotonically.
  c  F17. How many co-located labels. A HUMP, not a monotone curve.
  d  F19. How correlated the two targets must be. The floor, and it MOVES
         with the label budget.

Panel (d) also carries the calibration the rung-1 campaign forced: a pilot
samples the library and reads HIGH relative to the correlation a campaign
actually operates at, because the optimizer hunts molecules on which the two
targets disagree.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd, os
from scipy.stats import wilcoxon

B = "/Users/devansh/mogp-main-vscode/MOGP-NTD"
S = B + "/analysis_scripts"
OUT = "/Users/devansh/Downloads/aggregate_10seed/figures"
ICM, IND = "#1b5e9c", "#b2182b"
plt.rcParams.update({
    "font.size": 9, "axes.grid": True, "grid.alpha": .22, "grid.linewidth": .6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": .9,
    "legend.frameon": False, "figure.dpi": 110})

fig = plt.figure(figsize=(12.6, 7.9))
gs = fig.add_gridspec(2, 2, hspace=.56, wspace=.26,
                      left=.075, right=.985, top=.89, bottom=.088)

# (a) F12 -- block-complete design, the ICM has nothing to add ------------
ax = fig.add_subplot(gs[0, 0])
m = pd.read_csv(f"{S}/multiseed_metrics.csv")
for _, r in m.iterrows():
    ax.plot([0, 1], [r.icm_final, r.ind_final], "-o", color="#666", ms=4, lw=1.0, alpha=.7)
ax.set_xticks([0, 1]); ax.set_xticklabels(["shared model\n(ICM)", "separate models"])
ax.set_ylabel("final hypervolume")
p = wilcoxon(m.icm_final, m.ind_final).pvalue
ax.set_title(f"a   Nothing missing: no benefit\n10 paired seeds, p = {p:.2f}",
             loc="left", fontsize=10, weight="bold")
ax.set_xlim(-.45, 1.45)
ax.text(.5, -.155, "Every molecule carries both labels, so the design is block-complete\n"
                  "and the shared posterior collapses onto independent per-task GPs.",
        transform=ax.transAxes, fontsize=7.4, color="#444", ha="center", va="top")

# (b) F13 -- break the co-location and it helps ---------------------------
ax = fig.add_subplot(gs[0, 1])
a = pd.read_csv(f"{S}/asym_results_20.csv")
piv = a.pivot_table(index=["rep", "frac"], columns="model", values="rmse").reset_index()
g = piv.groupby("frac").apply(lambda d: pd.Series({
    "adv": (d.independent - d.ICM).mean(),
    "se": (d.independent - d.ICM).std(ddof=1) / np.sqrt(len(d))}), include_groups=False).reset_index()
g = g.sort_values("frac")
ax.errorbar(g.frac * 100, g.adv, yerr=g.se, fmt="-o", color=ICM, lw=1.9, ms=6, capsize=3)
ax.axhline(0, color="#999", lw=.9, ls="--")
ax.set_xlabel("% of second-target labels kept"); ax.set_ylabel("ICM advantage (RMSE, kcal/mol)")
ax.invert_xaxis()
ax.set_title("b   Labels already missing: it helps\nmonotone, Spearman = -1.000",
             loc="left", fontsize=10, weight="bold")
for _, r in g.iterrows():
    ax.annotate(f"{r.adv:+.3f}", (r.frac * 100, r.adv), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=7.4)

# (c) F17 -- how many co-located labels, a hump ---------------------------
ax = fig.add_subplot(gs[1, 0])
c = pd.read_csv(f"{S}/colocation_sweep.csv")
cp = c.pivot_table(index=["rep", "k"], columns="model", values="rmse").reset_index()
cg = cp.groupby("k").apply(lambda d: pd.Series({
    "adv": (d.independent - d.ICM).mean(),
    "se": (d.independent - d.ICM).std(ddof=1) / np.sqrt(len(d))}), include_groups=False).reset_index()
cg = cg.sort_values("k")
ax.errorbar(cg.k, cg.adv, yerr=cg.se, fmt="-o", color=ICM, lw=1.9, ms=5.5, capsize=3)
ax.axhline(0, color="#999", lw=.9, ls="--")
kbest = cg.loc[cg.adv.idxmax()]
ax.axvspan(8, 22, color="#1b5e9c", alpha=.09, lw=0)
ax.annotate(f"peak k={kbest.k:.0f}, {100*kbest.k/225:.1f}% of 225",
            (kbest.k, kbest.adv), textcoords="offset points", xytext=(6, 26),
            fontsize=7.8, color=ICM, weight="bold",
            arrowprops=dict(arrowstyle="-", color=ICM, lw=.8))
ax.set_xscale("log"); ax.set_xlabel("co-located molecules k (log scale)")
ax.set_ylabel("ICM advantage (RMSE, kcal/mol)")
ax.set_title("c   How many: a hump, not a ramp\noperating band 3.6% to 9.8% (shaded)",
             loc="left", fontsize=10, weight="bold")

# (d) F19 -- the correlation floor, and the pilot calibration -------------
ax = fig.add_subplot(gs[1, 1])
f = pd.read_csv(f"{S}/correlation_floor.csv")
f = f[~f.is_real]
fp = f.pivot_table(index=["rep", "k", "rho_target"], columns="model", values="rmse").reset_index()
for k, col, mk in [(8, "#1b5e9c", "o"), (22, "#5aae61", "s")]:
    d = fp[fp.k == k].groupby("rho_target").apply(
        lambda x: (x.independent - x.ICM).mean(), include_groups=False)
    d = d.sort_index()
    ax.plot(d.index, d.values, mk + "-", color=col, lw=1.8, ms=5,
            label=f"k={k} ({100*k/225:.1f}% co-located)")
ax.axhline(0, color="#999", lw=.9, ls="--")
ax.axvline(0.532, color="#e08214", lw=1.8, ls="-")
ax.axvline(0.719, color="#e08214", lw=1.4, ls=":")
ax.set_xlim(-.82, 1.16)
_lo = ax.get_ylim()[0]
ax.annotate("campaign\n0.53", (0.532, _lo), xytext=(-3, 6), textcoords="offset points",
            fontsize=7.4, color="#b06000", ha="right", va="bottom", weight="bold")
ax.annotate("pilot\n0.72", (0.719, _lo), xytext=(3, 6), textcoords="offset points",
            fontsize=7.4, color="#b06000", ha="left", va="bottom")
ax.set_xlabel(r"task correlation $\rho$ between the two targets")
ax.set_ylabel("ICM advantage (RMSE, kcal/mol)")
ax.legend(fontsize=7.8, loc="upper left")
ax.set_title("d   How similar: the floor moves with the budget\n"
             "and a pilot reads HIGH against a real campaign",
             loc="left", fontsize=10, weight="bold")

fig.suptitle("Figure 3   Sharing information between two docking targets is a mitigation "
             "for missing labels, not a reason to create them",
             fontsize=11.2, weight="bold", y=.968, x=.075, ha="left")
os.makedirs(OUT, exist_ok=True)
for d in (OUT, B):
    fig.savefig(f"{d}/MAIN_FIG3_when_sharing_helps.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{d}/MAIN_FIG3_when_sharing_helps.pdf", bbox_inches="tight")
print("wrote MAIN_FIG3_when_sharing_helps.{png,pdf}")
print(f"  (a) F12 p={p:.3f}   (b) F13 advantage at 10% = {g.adv.iloc[0]:+.3f}")
print(f"  (c) F17 peak k={kbest.k:.0f} adv={kbest.adv:+.3f}")
