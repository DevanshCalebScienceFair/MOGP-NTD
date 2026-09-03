import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from scipy.stats import wilcoxon
B="/Users/devansh/mogp-main-vscode/MOGP-NTD"; OUT="/Users/devansh/Downloads/aggregate_10seed/figures"
df=pd.read_csv(f"{B}/ref_point_arm/scored.csv")
Z,N,RED,GRN,GRY="#2166ac","#762a83","#b2182b","#1a9850","#999999"
plt.rcParams.update({"font.size":9,"axes.grid":True,"grid.alpha":.22,
                     "axes.spines.top":False,"axes.spines.right":False})
fig=plt.figure(figsize=(15,5.4)); gs=fig.add_gridspec(1,4,wspace=.34)

# (a) what changed
ax=fig.add_subplot(gs[0,0]); ax.axis("off")
ax.text(0,1.0,"(a) What changed",fontsize=10.5,weight="bold",transform=ax.transAxes,va="top")
ax.text(0,.88,
"qNEHVI measures improvement against a\n"
"REFERENCE POINT. Every published run used\n"
"the all-zeros corner of the normalized cube.\n\n"
"That is safe and method-independent, but the\n"
"dominated region then includes a large block\n"
"far below any real molecule, so improvement\n"
"where it matters is a small share of the total.\n\n"
"'nadir' puts the reference just under the\n"
"worst OBSERVED value on each objective.\n\n"
"The METRIC still uses the fixed all-zeros\n"
"reference, so these arms are directly\n"
"comparable -- unlike a bounds change.",
        fontsize=8.1,transform=ax.transAxes,va="top",linespacing=1.42)

# (b) hypervolume: null
ax=fig.add_subplot(gs[0,1])
for _,r in df.iterrows():
    win=r.nadir_hv>r.zeros_hv
    ax.plot([0,1],[r.zeros_hv,r.nadir_hv],"-o",ms=5.5,lw=1.4,alpha=.85,
            color=GRN if win else RED)
ax.set_xticks([0,1]); ax.set_xticklabels(["zeros\n(published)","nadir"])
ax.set_ylabel("final hypervolume")
d=df.nadir_hv-df.zeros_hv
ax.set_title(f"(b) Hypervolume: null\n{int((d>0).sum())}/6 · {d.mean():+.4f} · p=0.44",
             fontsize=9.4,loc="left")

# (c) the one signal
ax=fig.add_subplot(gs[0,2])
for _,r in df.iterrows():
    win=r.nadir_physical>r.zeros_physical
    ax.plot([0,1],[r.zeros_physical,r.nadir_physical],"-o",ms=5.5,lw=1.4,alpha=.85,
            color=GRN if win else RED)
ax.set_xticks([0,1]); ax.set_xticklabels(["zeros","nadir"])
ax.set_ylabel("physical molecules found (of ~290)")
dp=df.nadir_physical-df.zeros_physical
ax.set_title(f"(c) The one signal: +{dp.mean():.1f} physical\n"
             f"{int((dp>0).sum())}/6 seeds · p=0.0625 · CI [+1.5, +5.3]",
             fontsize=9.4,loc="left",color=GRN)
ax.text(.04,.06,"borderline at n=6, which caps\nWilcoxon at 0.0312.",
        transform=ax.transAxes,fontsize=7.4,color="#555")

# (d) verdict
ax=fig.add_subplot(gs[0,3]); ax.axis("off")
ax.text(0,1.0,"(d) Verdict",fontsize=10.5,weight="bold",transform=ax.transAxes,va="top")
ax.text(0,.88,
"Mostly null. Hypervolume, AUC, front size and\n"
"selectivity are all ties.\n\n"
"ONE suggestive effect: the tighter reference\n"
"found +3.5 more PHYSICAL molecules per\n"
"campaign, in 5 of 6 seeds, CI excluding zero.\n"
"About +1.4% of a 290-molecule budget.\n\n"
"Consistent with the mechanism -- concentrating\n"
"improvement where the front actually is should\n"
"waste fewer docks on the far corner -- but\n"
"top-20 selectivity did NOT improve (−0.066,\n"
"1/6), so it found more real molecules without\n"
"finding better ones.\n\n"
"NOT worth changing the default on. p=0.0625 is\n"
"the n=6 boundary and one endpoint out of six.\n"
"Worth 10 seeds if anyone wants to settle it.",
        fontsize=8.1,transform=ax.transAxes,va="top",linespacing=1.40,color="#333")

fig.suptitle("F22 — A tighter acquisition reference point: mostly null, one suggestive effect",
             fontsize=12.5,weight="bold",y=1.005)
fig.text(.5,-.04,"6 paired seeds, identical configuration except --acquisition-ref-point. The reported metric always uses the fixed all-zeros reference, so both arms are directly comparable. "
                 "n=6 gives a minimum two-sided Wilcoxon p of 0.0312.",
         ha="center",fontsize=7.3,color="#666")
fig.savefig(f"{OUT}/F22_reference_point.png",dpi=170,bbox_inches="tight",facecolor="w")
print("wrote F22_reference_point.png")
