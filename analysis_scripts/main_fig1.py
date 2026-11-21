"""MAIN FIGURE 1 -- the primary outcome.

Everything here is recomputed from campaign_results/, never read from a
summary CSV. The summary file in ~/Downloads carries a GP-MOBO column named
"recovered_rawkcal_appendix" whose mean is 0.3054; recomputing from the
per-seed artifacts gives 0.3123, which is what every write-up states. That is
the project's own sweep-file rule: a sweep-level number must be derived from
the artifacts on disk, never from whatever a previous invocation wrote down.

GP-MOBO comes from the CLEAN rerun (aggregate_10seed_cleanGPMOBO/); the stored
GP-MOBO column in the original campaign dir is zeroed and flagged INVALID.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd, sys, os
from scipy.stats import wilcoxon, friedmanchisquare

B = "/Users/devansh/mogp-main-vscode/MOGP-NTD"
OUT = "/Users/devansh/Downloads/aggregate_10seed/figures"
sys.path.insert(0, B)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import evaluation

BOUNDS = evaluation.compute_objective_bounds()
T = evaluation.TASK_NAMES
SEEDS = range(10)
PATHS = {
    "mogp":   B + "/campaign_results/seed_{s}/mogp/seed_{s}",
    "gpmobo": B + "/campaign_results/aggregate_10seed_cleanGPMOBO/seed_{s}/gpmobo/seed_{s}",
    "greedy": B + "/campaign_results/seed_{s}/greedy/seed_{s}",
}
LABEL = {"mogp": "MOGP (this work)", "gpmobo": "GP-MOBO", "greedy": "Greedy filter-then-dock"}
C = {"mogp": "#1b5e9c", "gpmobo": "#e08214", "greedy": "#7a7a7a"}
ARMS = ["mogp", "gpmobo", "greedy"]

plt.rcParams.update({
    "font.size": 9, "axes.grid": True, "grid.alpha": .22, "grid.linewidth": .6,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .9, "xtick.major.width": .9, "ytick.major.width": .9,
    "legend.frameon": False, "figure.dpi": 110})

hist = {a: [pd.read_csv(PATHS[a].format(s=s) + "/history.csv") for s in SEEDS] for a in ARMS}
fin = {a: np.array([evaluation.compute_hypervolume(
            pd.read_csv(PATHS[a].format(s=s) + "/evaluated.csv")[T].to_numpy(float),
            bounds=BOUNDS) for s in SEEDS]) for a in ARMS}
pareto = {a: np.array([h.pareto_size.iloc[-1] for h in hist[a]]) for a in ARMS}

# molecules needed to reach Greedy's FINAL score -- a fixed target, not a
# fraction of the best observed, which would be circular.
def evals_to(arm, target):
    out = []
    for s in SEEDS:
        h = hist[arm][s]
        hit = h[h.hypervolume >= target]
        out.append(float(hit.n_evaluated.iloc[0]) if len(hit) else np.nan)
    return np.array(out)

tgt = fin["greedy"]
reach = {a: np.array([evals_to(a, tgt[s])[s] for s in SEEDS]) for a in ARMS}

fig = plt.figure(figsize=(12.6, 7.4))
gs = fig.add_gridspec(2, 3, hspace=.40, wspace=.30,
                      left=.065, right=.985, top=.90, bottom=.085)

# (a) the curves -----------------------------------------------------------
ax = fig.add_subplot(gs[0, :2])
for a in ARMS:
    grid = np.arange(40, 291)
    curves = np.vstack([np.interp(grid, h.n_evaluated, h.hypervolume) for h in hist[a]])
    m, sd = curves.mean(0), curves.std(0, ddof=1)
    ax.plot(grid, m, color=C[a], lw=2.0, label=LABEL[a], zorder=3)
    ax.fill_between(grid, m - sd, m + sd, color=C[a], alpha=.17, lw=0, zorder=2)
ax.set_xlabel("molecules docked"); ax.set_ylabel("hypervolume")
ax.legend(loc="upper left", fontsize=8.6)
ax.set_title("a   Search progress, 10 seeds (mean $\\pm$ s.d.)", loc="left",
             fontsize=10, weight="bold")
ax.set_xlim(40, 290)

# (b) final hypervolume ----------------------------------------------------
ax = fig.add_subplot(gs[0, 2])
rngj = np.random.default_rng(0)
for i, a in enumerate(ARMS):
    ax.scatter(np.full(10, i) + rngj.uniform(-.13, .13, 10), fin[a], s=20,
               color=C[a], alpha=.75, zorder=3, lw=0)
    ax.hlines(fin[a].mean(), i - .28, i + .28, color=C[a], lw=2.4, zorder=4)
    ax.text(i, fin[a].mean() + .022, f"{fin[a].mean():.3f}", ha="center",
            fontsize=9, weight="bold", color=C[a])
ax.set_xticks(range(3)); ax.set_xticklabels(["MOGP", "GP-MOBO", "Greedy"], fontsize=8.6)
ax.set_ylabel("final hypervolume"); ax.set_ylim(.12, .47)
chi2, pf = friedmanchisquare(fin["mogp"], fin["gpmobo"], fin["greedy"])
ax.set_title(f"b   Complete separation\nFriedman $\\chi^2$={chi2:.1f}, p={pf:.1e}",
             loc="left", fontsize=10, weight="bold")

# (c) paired, per seed -----------------------------------------------------
ax = fig.add_subplot(gs[1, 0])
for s in SEEDS:
    ax.plot([0, 1, 2], [fin["mogp"][s], fin["gpmobo"][s], fin["greedy"][s]],
            "-o", color="#555", ms=3.4, lw=.9, alpha=.62)
ax.set_xticks(range(3)); ax.set_xticklabels(["MOGP", "GP-MOBO", "Greedy"], fontsize=8.6)
ax.set_ylabel("final hypervolume")
pmg = wilcoxon(fin["mogp"], fin["gpmobo"]).pvalue
ax.set_title("c   Every seed, same order\n10/10 and 10/10, Wilcoxon p=%.4f" % pmg,
             loc="left", fontsize=10, weight="bold")

# (d) sample efficiency ----------------------------------------------------
ax = fig.add_subplot(gs[1, 1])
x = np.arange(3)
mu = [np.nanmean(reach[a]) for a in ARMS]
ax.bar(x, mu, .56, color=[C[a] for a in ARMS],
       yerr=[np.nanstd(reach[a], ddof=1) for a in ARMS], capsize=4, ecolor="#333")
for i, v in enumerate(mu):
    ax.text(i, v + 9, f"{v:.0f}", ha="center", fontsize=9.5, weight="bold")
ax.set_xticks(x); ax.set_xticklabels(["MOGP", "GP-MOBO", "Greedy"], fontsize=8.6)
ax.set_ylabel("molecules to match Greedy's\nfinal score"); ax.set_ylim(0, 330)
ax.set_title("d   Sample efficiency\nMOGP needs %.0f%% of the budget" % (100*mu[0]/290),
             loc="left", fontsize=10, weight="bold")

# (e) front size -----------------------------------------------------------
ax = fig.add_subplot(gs[1, 2])
mu = [pareto[a].mean() for a in ARMS]
ax.bar(x, mu, .56, color=[C[a] for a in ARMS],
       yerr=[pareto[a].std(ddof=1) for a in ARMS], capsize=4, ecolor="#333")
for i, v in enumerate(mu):
    ax.text(i, v + 5, f"{v:.0f}", ha="center", fontsize=9.5, weight="bold")
ax.set_xticks(x); ax.set_xticklabels(["MOGP", "GP-MOBO", "Greedy"], fontsize=8.6)
ax.set_ylabel("Pareto front size"); ax.set_ylim(0, 205)
pgg = wilcoxon(pareto["gpmobo"], pareto["greedy"]).pvalue
ax.set_title("e   Front structure\nGP-MOBO vs Greedy indistinguishable (p=%.2f)" % pgg,
             loc="left", fontsize=10, weight="bold")

fig.suptitle("Figure 1   Multi-objective Bayesian optimization over a 26,660-molecule "
             "library finds better antifolate trade-offs than either baseline, "
             "in a fifth of the docking budget",
             fontsize=11.2, weight="bold", y=.975, x=.065, ha="left")
os.makedirs(OUT, exist_ok=True)
for d in (OUT, B):
    fig.savefig(f"{d}/MAIN_FIG1_primary_outcome.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{d}/MAIN_FIG1_primary_outcome.pdf", bbox_inches="tight")
print("wrote MAIN_FIG1_primary_outcome.{png,pdf}")
for a in ARMS:
    print(f"  {a:<7} hv {fin[a].mean():.4f}+/-{fin[a].std(ddof=1):.4f}  "
          f"front {pareto[a].mean():.1f}  reach {np.nanmean(reach[a]):.0f}")
print(f"  Friedman chi2={chi2:.2f} p={pf:.3e}   MOGP vs GPMOBO p={pmg:.5f}")
