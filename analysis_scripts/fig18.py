"""F18: selection policy for the second label. Three panels.

(a) held-out hDHFR RMSE by policy and k, with the pre-registered prediction
    marked so the failure is visible rather than described;
(b) the dissociation: subset spread helps RANKING and hurts RMSE;
(c) the mechanism that did hold: farthest-point selection stabilizes the learned
    task correlation at every k, and that stability does not buy accuracy.
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

ROOT = "/Users/devansh/mogp-main-vscode/MOGP-NTD"
d = pd.read_csv(f"{ROOT}/analysis_scripts/colocation_policy.csv")
KS = sorted(d.k.unique())
POL = ["central", "random", "diverse"]
COL = {"central": "#b2182b", "random": "#4d4d4d", "diverse": "#2166ac"}
LBL = {"central": "central (clustered)", "random": "random (what the code did)",
       "diverse": "diverse (farthest-point)"}
NTRAIN = 225

fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.0))

# ---- (a) RMSE by policy ----
w = 0.26
x = np.arange(len(KS))
for i, p in enumerate(POL):
    m = [d[(d.k == k) & (d.policy == p)].rmse.mean() for k in KS]
    se = [d[(d.k == k) & (d.policy == p)].rmse.std() / np.sqrt(20) for k in KS]
    ax[0].bar(x + (i - 1) * w, m, w, yerr=se, capsize=3, color=COL[p],
              label=LBL[p], edgecolor="white", linewidth=0.6)
ax[0].set_xticks(x)
ax[0].set_xticklabels([f"{k}\n({k / NTRAIN:.1%})" for k in KS])
ax[0].set_xlabel("co-located molecules k  (% of 225 training molecules)")
ax[0].set_ylabel("held-out hDHFR RMSE (kcal/mol), lower is better")
ax[0].set_ylim(1.30, 1.84)
ax[0].legend(fontsize=8.5, loc="lower right", framealpha=0.95)
ax[0].set_title("(a) The pre-registered ordering FAILED", loc="left",
                fontweight="bold", fontsize=11)
ax[0].annotate("predicted: central worst, diverse best\n"
               "measured:  central worst, diverse SECOND worst\n"
               "diverse vs random  -0.059 kcal/mol, 38/80, p = 0.11",
               xy=(0.02, 0.97), xycoords="axes fraction", va="top", fontsize=8.3,
               bbox=dict(boxstyle="round,pad=0.45", fc="#fff3e0", ec="#e65100",
                         lw=0.9))
ax[0].grid(axis="y", alpha=0.25, zorder=0)
ax[0].set_axisbelow(True)

# ---- (b) the dissociation ----
g = d.groupby(["k", "policy"]).agg(spread=("spread", "mean"),
                                   rmse=("rmse", "mean"),
                                   sp=("spearman", "mean")).reset_index()
a2 = ax[1]
for p in POL:
    s = g[g.policy == p]
    a2.scatter(s.spread, s.sp, s=90, color=COL[p], label=LBL[p], zorder=3,
               edgecolor="white", linewidth=1.2)
a2.set_xlabel("subset spread (mean pairwise Tanimoto distance)")
a2.set_ylabel("rank correlation with true hDHFR", color="#2166ac")
a2.tick_params(axis="y", labelcolor="#2166ac")
r1 = spearmanr(g.spread, g.sp).statistic
a3 = a2.twinx()
for p in POL:
    s = g[g.policy == p]
    a3.scatter(s.spread, s.rmse, s=90, color=COL[p], marker="s", zorder=3,
               alpha=0.45, edgecolor="white", linewidth=1.2)
a3.set_ylabel("RMSE (kcal/mol)", color="#b2182b")
a3.tick_params(axis="y", labelcolor="#b2182b")
r2 = spearmanr(g.spread, g.rmse).statistic
a2.set_title("(b) Spread helps RANKING and hurts RMSE", loc="left",
             fontweight="bold", fontsize=11)
a2.annotate(f"circles, left axis: rank rho    Spearman {r1:+.3f}\n"
            f"squares, right axis: RMSE      Spearman {r2:+.3f}\n"
            "12 policy x k cells, 20 repeats each",
            xy=(0.03, 0.97), xycoords="axes fraction", va="top", fontsize=8.3,
            bbox=dict(boxstyle="round,pad=0.45", fc="#f5f5f5", ec="#9e9e9e", lw=0.9))
a2.legend(fontsize=8, loc="lower right", framealpha=0.95)
a2.grid(alpha=0.25, zorder=0)
a2.set_axisbelow(True)

# ---- (c) the mechanism that held ----
for p in POL:
    sd = [d[(d.k == k) & (d.policy == p)].rho_B.std() for k in KS]
    ax[2].plot(KS, sd, "o-", color=COL[p], label=LBL[p], lw=2, ms=7,
               markeredgecolor="white", markeredgewidth=1.2)
ax[2].set_xscale("log")
ax[2].set_xticks(KS); ax[2].set_xticklabels([str(k) for k in KS])
ax[2].set_xticks([], minor=True)
ax[2].set_xlabel("co-located molecules k (log scale)")
ax[2].set_ylabel("sd of the learned task correlation across 20 repeats")
ax[2].set_title("(c) Diverse DOES stabilize the task covariance", loc="left",
                fontweight="bold", fontsize=11)
ax[2].annotate("lower = the model is more certain\nabout the transfer strength.\n"
               "Diverse is lowest at every k,\nand still does not predict better.\n"
               "Stability is not the binding constraint.",
               xy=(0.97, 0.95), xycoords="axes fraction", va="top", ha="right",
               fontsize=8.3,
               bbox=dict(boxstyle="round,pad=0.45", fc="#e3f2fd", ec="#1565c0", lw=0.9))
ax[2].legend(fontsize=8, loc="lower left", framealpha=0.95)
ax[2].grid(alpha=0.25)

fig.suptitle("F18  Which molecules carry the expensive second label? "
             "Only one rule survives: do not cluster them.",
             fontsize=12.5, fontweight="bold", y=0.99)
fig.tight_layout(rect=[0, 0.02, 1, 0.95])
OUT = f"{ROOT}/F18_colocation_policy.png"
fig.savefig(OUT, dpi=160, facecolor="white")
print("wrote", OUT, os.path.getsize(OUT), "bytes")
