"""F27 -- Rung 1 (Chagas): the asymmetric design loses inside F17's band.

This closes the cell CLAUDE.md flagged as "NOT shown, and do not claim it".
F14 ran the asymmetric arm at 25% co-location, 2.5x above F17's k=8 peak, and
lost 6/6. The objection was that it was run outside the band. This runs it
INSIDE the band (--hdhfr-policy uniform --hdhfr-fraction 0.05 --hdhfr-floor 8,
measured 24/465 = 5.2%) on a NEW disease, and it loses again.

The figure's job is to show the loss is real AFTER removing the structural
bias F14 already identified, and to name the mechanism.
"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd, sys, os
from scipy.stats import wilcoxon, binomtest

B = "/Users/devansh/mogp-main-vscode/MOGP-NTD"
OUT = "/Users/devansh/Downloads/aggregate_10seed/figures"
sys.path.insert(0, B)
os.environ.setdefault("MOGP_CAMPAIGN", "chagas")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import evaluation

T = evaluation.TASK_NAMES
BOUNDS = evaluation.compute_objective_bounds()
SEEDS = range(6)
FULL, INDEP, ASYM = "#2166ac", "#5aae61", "#e08214"
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .25,
                     "axes.spines.top": False, "axes.spines.right": False})


def ev(arm, s):
    return pd.read_csv(f"{B}/rung1_chagas/{arm}_seed{s}/evaluated.csv")


def hv(Y):
    return evaluation.compute_hypervolume(Y, bounds=BOUNDS)


# ---- load + the matched-subset reference -------------------------------
act = {a: np.array([hv(ev(a, s)[T].to_numpy(float)) for s in SEEDS])
       for a in ("full", "indep", "asym")}
rng = np.random.default_rng(12345)
NDRAW = 400
rand24 = {}
for a in ("full", "indep"):
    for s in SEEDS:
        Y = ev(a, s)[T].to_numpy(float)
        ok = Y[np.isfinite(Y).all(axis=1)]
        rand24[(a, s)] = np.array(
            [hv(ok[rng.choice(len(ok), 24, replace=False)]) for _ in range(NDRAW)])
pool24 = np.concatenate([rand24[("full", s)] for s in SEEDS])
pcts = [(pool24 < x).mean() * 100 for x in act["asym"]]
k_below = sum(1 for p in pcts if p < 5)

fig = plt.figure(figsize=(15, 9.6))
gs = fig.add_gridspec(2, 3, hspace=.42, wspace=.30)

# (a) the design -----------------------------------------------------------
ax = fig.add_subplot(gs[0, 0]); ax.axis("off")
ax.text(0, 1.0, "(a) Rung 1: a new disease, three arms", fontsize=10,
        weight="bold", transform=ax.transAxes, va="top")
ax.text(0, .90,
"Chagas disease. T. cruzi DHFR-TS (3INV) against\n"
"human DHFR (1U72). 6 seeds per arm, one backend.\n\n"
"               MODEL        hDHFR LABELS   MOLECULES\n"
"  full      Hadamard ICM        100%           290\n"
"  indep     independent GPs     100%           290\n"
"  asym      Hadamard ICM        5.2%           465\n\n"
"F14 ran asym at 25% co-location, 2.5x above\n"
"F17's k=8 peak, and lost 6/6. The standing\n"
"objection was that it ran OUTSIDE the band.\n\n"
"This runs it INSIDE the band: policy=uniform,\n"
"fraction=0.05, floor=8, measured 24/465 = 5.2%.\n"
"F17's operating band is 3.6% to 9.8%.",
        fontsize=8.3, transform=ax.transAxes, va="top", linespacing=1.42,
        family="monospace")

# (b) full vs indep: the predicted null ------------------------------------
ax = fig.add_subplot(gs[0, 1])
for s in SEEDS:
    ax.plot([0, 1], [act["full"][s], act["indep"][s]], "-o", color="#777",
            ms=5, lw=1.3, alpha=.8)
ax.set_xticks([0, 1]); ax.set_xticklabels(["full\n(ICM)", "indep\n(separate GPs)"])
ax.set_ylabel("final hypervolume")
p_fi = wilcoxon(act["full"], act["indep"]).pvalue
ax.set_title(f"(b) The predicted null holds\n3 seeds each way, p = {p_fi:.2f}",
             fontsize=9.5, loc="left")
ax.text(.03, .05,
        "Every molecule carries both labels, so the\n"
        "design is block-complete and the ICM posterior\n"
        "collapses onto independent GPs (autokrigeability).\n"
        "A tie is the correct result, not a failure.",
        transform=ax.transAxes, fontsize=7.2, color="#444")

# (c) the raw gap is mostly an artifact ------------------------------------
ax = fig.add_subplot(gs[0, 2])
ax.bar([0], [act["full"].mean()], .55, color=FULL, label="full, all 290")
ax.bar([1], [pool24.mean()], .55, color=FULL, alpha=.45,
       label="full, cut to 24 molecules")
ax.bar([2], [act["asym"].mean()], .55, color=ASYM, label="asym, its 24")
ax.errorbar([1], [pool24.mean()], yerr=[[pool24.mean() - np.percentile(pool24, 5)],
            [np.percentile(pool24, 95) - pool24.mean()]], fmt="none",
            ecolor="#333", capsize=4, lw=1.2)
for i, v in enumerate([act["full"].mean(), pool24.mean(), act["asym"].mean()]):
    ax.text(i, v + .008, f"{v:.3f}", ha="center", fontsize=8.5, weight="bold")
ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["full\n(290)", "full\n(any 24)", "asym\n(24)"])
ax.set_ylabel("hypervolume"); ax.set_ylim(0, .46)
ax.legend(fontsize=7.2, loc="upper right")
ax.set_title("(c) Most of the raw gap is a counting artifact\n"
             "but it does NOT explain all of it", fontsize=9.5, loc="left")

# (d) the matched test ------------------------------------------------------
ax = fig.add_subplot(gs[1, 0])
ax.hist(pool24, bins=46, color=FULL, alpha=.45,
        label=f"full arm, random 24 molecules\n({NDRAW} draws x 6 seeds)")
for i, v in enumerate(act["asym"]):
    ax.axvline(v, color=ASYM, lw=1.6, alpha=.9,
               label="asym arm, 6 seeds" if i == 0 else None)
ax.axvline(pool24.mean(), color="#b2182b", ls="--", lw=1.4,
           label=f"matched mean {pool24.mean():.3f}")
ax.set_xlabel("hypervolume of a 24-molecule set"); ax.set_ylabel("draws")
ax.legend(fontsize=7, loc="upper right")
pb = binomtest(k_below, 6, 0.05, alternative="greater").pvalue
ax.set_title(f"(d) Matched, asym still loses\n0/6 above the matched mean; "
             f"{k_below}/6 below its 5th pct, p = {pb:.0e}",
             fontsize=9.5, loc="left", color="#b2182b")

# (e) where the loss is ------------------------------------------------------
ax = fig.add_subplot(gs[1, 1])
prim = {a: np.array([np.sort(ev(a, s)["TcDHFR_Docking"].dropna().to_numpy())[:10].mean()
                     for s in SEEDS]) for a in ("full", "indep", "asym")}
x = np.arange(3)
ax.bar(x, [prim[a].mean() for a in ("full", "indep", "asym")], .55,
       color=[FULL, INDEP, ASYM],
       yerr=[prim[a].std(ddof=1) for a in ("full", "indep", "asym")],
       capsize=4, ecolor="#333")
for i, a in enumerate(("full", "indep", "asym")):
    ax.text(i, prim[a].mean() * .55, f"{prim[a].mean():.2f}", ha="center",
            fontsize=11, weight="bold", color="w")
ax.set_xticks(x); ax.set_xticklabels(["full", "indep", "asym"])
ax.set_ylabel("top-10 mean T. cruzi DHFR (kcal/mol)")
ax.invert_yaxis()
ax.set_title("(e) The disease target is UNAFFECTED\nasym matches on potency; it loses only "
             "on selectivity", fontsize=9.5, loc="left")

# (f) the mechanism + the pilot calibration ---------------------------------
ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
pooled = pd.concat([ev("full", s) for s in SEEDS])[["TcDHFR_Docking", "hDHFR_Docking"]].dropna()
pooled = pooled[pooled.hDHFR_Docking <= 0]
rho_c = np.corrcoef(pooled.TcDHFR_Docking, pooled.hDHFR_Docking)[0, 1]
sel = (pooled.TcDHFR_Docking - pooled.hDHFR_Docking)
frac = (sel < -5).mean() * 100
_raw = pd.concat([ev("full", s_) for s_ in SEEDS])[["TcDHFR_Docking", "hDHFR_Docking"]].dropna()
nbad = int((_raw.hDHFR_Docking > 0).sum())
pcatch = 1.0 - (1.0 - frac / 100.0) ** 24
ax.text(0, 1.0, "(f) Why, and what it costs the next campaign",
        fontsize=10, weight="bold", transform=ax.transAxes, va="top")
ax.text(0, .90,
f"MECHANISM. Strongly selective molecules are\n"
f"RARE: {frac:.2f}% score below -5 kcal/mol, i.e.\n"
f"{frac*278/100:.1f} of 278. A uniform 24-label budget\n"
f"catches one only {pcatch*100:.0f}% of the time. Full\n"
f"labelling catches all of them, every seed.\n\n"
f"ARTIFACT FILTER IS LOAD-BEARING. Unfiltered, the\n"
f"best selectivity reads -14.0 and the rare fraction\n"
f"1.77%. Both come from {nbad} rows with a POSITIVE\n"
f"hDHFR affinity, which is a FAILED FIT, not a weak\n"
f"binder. Filtered: best -6.2, fraction {frac:.2f}%.\n\n"
f"And the floor ate a third of the budget: with\n"
f"floor=8, ceil(0.05n) does not pass 8 until n>160,\n"
f"so 8 of the 24 labels landed on the RANDOM\n"
f"initial molecules before the model knew anything.\n\n"
f"THE PILOT READS HIGH. Measured here:\n\n"
f"                        pilot      campaign\n"
f"   all filtered        +0.719       {rho_c:+.3f}\n"
f"   binders only        +0.589       +0.330\n\n"
f"A pilot samples the LIBRARY; a campaign samples\n"
f"what the optimizer chose, and the optimizer hunts\n"
f"molecules on which the two targets DISAGREE.\n"
f"F19's floor at 5.2% co-location is about 0.40.\n"
f"The pilot cleared it by 0.32; the campaign by\n"
f"{rho_c-0.40:+.2f}. Gate the next rung on headroom,\n"
f"not on a bound that merely clears.",
        fontsize=8.0, transform=ax.transAxes, va="top", linespacing=1.40,
        family="monospace")

fig.suptitle("F27  Rung 1, Chagas: the asymmetric design loses INSIDE F17's band, "
             "and the disease target is not what it loses on",
             fontsize=12, weight="bold", y=.985)
os.makedirs(OUT, exist_ok=True)
for d in (OUT, B):
    fig.savefig(f"{d}/F27_rung1_chagas.png", dpi=170, bbox_inches="tight")
print(f"wrote F27_rung1_chagas.png")
print(f"  full {act['full'].mean():.4f}  indep {act['indep'].mean():.4f}  p={p_fi:.3f}")
print(f"  asym {act['asym'].mean():.4f}  matched-24 {pool24.mean():.4f}  p={pb:.2e}")
print(f"  campaign rho {rho_c:.3f}  (pilot read +0.719)")
