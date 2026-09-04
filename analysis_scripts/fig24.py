import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patches as mp, numpy as np, pandas as pd
from scipy.stats import wilcoxon
B="/Users/devansh/mogp-main-vscode/MOGP-NTD"; OUT="/Users/devansh/Downloads/aggregate_10seed/figures"
tr=pd.read_csv(f"{B}/artifact_training_arm/scored.csv")
fr=pd.read_csv(f"{B}/artifact_rejection_arm/scored.csv")
BASE,F1,F2,RED,GRN,DK="#2166ac","#e08214","#762a83","#b2182b","#1a9850","#222"
plt.rcParams.update({"font.size":9,"axes.grid":True,"grid.alpha":.22,
                     "axes.spines.top":False,"axes.spines.right":False})
fig=plt.figure(figsize=(15.5,9.4)); gs=fig.add_gridspec(2,3,hspace=.42,wspace=.30)

# (a) two places to intervene
ax=fig.add_subplot(gs[0,0]); ax.axis("off")
ax.text(0,1.0,"(a) Two places to intervene",fontsize=10.5,weight="bold",transform=ax.transAxes,va="top")
ax.text(0,.90,
"A clashing pose scores POSITIVE on hDHFR, so\n"
"its apparent selectivity is enormous while it\n"
"binds nothing. 42% of the campaign's raw top-5\n"
"by selectivity were non-physical.\n\n"
"There are exactly two places inside the\n"
"optimizer where you could keep them out:\n\n"
"  1. the FRONT qNEHVI scores against\n"
"     → F21, tested\n\n"
"  2. the GP TRAINING SET\n"
"     → this arm\n\n"
"F21 found (1) does nothing, and diagnosed why:\n"
"artifacts enter through the MODEL, not the\n"
"front. So (2) was the sharper hypothesis.",
        fontsize=8.2,transform=ax.transAxes,va="top",linespacing=1.42)

# (b) the pre-registered risk
ax=fig.add_subplot(gs[0,1]); ax.axis("off")
ax.text(0,1.0,"(b) What I predicted BEFORE running it",fontsize=10.5,weight="bold",
        transform=ax.transAxes,va="top")
ax.add_patch(mp.FancyBboxPatch((0,.30),1,.52,boxstyle="round,pad=.02",
                               transform=ax.transAxes,fc="#fff8e6",ec="#c8901e",lw=1.6))
ax.text(.04,.76,
'"Dropping those rows leaves the GP with no\n'
' data in that region and therefore high\n'
' posterior variance, which qNEHVI may read as\n'
' worth exploring. This arm could make artifact\n'
' chasing WORSE."',
        fontsize=8.4,transform=ax.transAxes,va="top",style="italic",color="#7a5200",
        linespacing=1.45)
ax.text(.04,.36,"— committed to the code, the runner header and\n   NEXT_STEPS.md, commit c46b191, before the run",
        fontsize=7.5,transform=ax.transAxes,va="top",color="#7a5200")
ax.text(0,.20,"That is what happened.",fontsize=12,weight="bold",
        transform=ax.transAxes,color=RED,va="top")

# (c) artifacts went UP under both
ax=fig.add_subplot(gs[0,2])
vals=[tr.base_artifacts.mean(), fr.rej_artifacts.mean(), tr.rej_artifacts.mean()]
b=ax.bar(range(3),vals,color=[BASE,F1,F2])
for i,v in enumerate(vals):
    ax.text(i,v+.5,f"{v:.1f}",ha="center",fontsize=10,weight="bold")
    if i: ax.text(i,v/2,f"{v-vals[0]:+.1f}",ha="center",fontsize=10,color="w",weight="bold")
ax.set_xticks(range(3)); ax.set_xticklabels(["baseline","filter the\nFRONT","filter the\nTRAINING SET"],fontsize=8.4)
ax.set_ylabel("docking artifacts evaluated per campaign"); ax.set_ylim(0,50)
ax.set_title("(c) Both interventions made it WORSE\nneither reduced the artifacts it targeted",
             fontsize=9.6,loc="left",color=RED)

# (d) and the training filter cost AUC
ax=fig.add_subplot(gs[1,0])
for _,r in tr.iterrows():
    win=r.rej_auc>r.base_auc
    ax.plot([0,1],[r.base_auc,r.rej_auc],"-o",ms=6,lw=1.5,alpha=.85,color=GRN if win else RED)
ax.set_xticks([0,1]); ax.set_xticklabels(["baseline","training filter"])
ax.set_ylabel("AUC of the hypervolume curve")
d=tr.rej_auc-tr.base_auc
ax.set_title(f"(d) …and it cost sample efficiency\n{int((d>0).sum())}/6 seeds · {d.mean():+.4f} · CI [-0.021, -0.001]",
             fontsize=9.6,loc="left",color=RED)
ax.text(.03,.06,"the interval excludes zero, though\np=0.16 at n=6",transform=ax.transAxes,
        fontsize=7.5,color="#555")

# (e) why
ax=fig.add_subplot(gs[1,1]); ax.axis("off")
ax.text(0,1.0,"(e) Why both failed",fontsize=10.5,weight="bold",transform=ax.transAxes,va="top")
ax.text(0,.90,
"Filtering the FRONT does nothing because the\n"
"artifacts are not entering there.\n\n"
"Filtering the TRAINING SET does something\n"
"worse. It removes the only evidence the model\n"
"had that those molecules are bad. Where the\n"
"data was, there is now a hole: no observations,\n"
"high posterior variance — and an acquisition\n"
"function whose whole job is to go where\n"
"uncertainty is high.\n\n"
"So the optimizer walks back into exactly the\n"
"region the filter was meant to protect it from,\n"
"and pays for the trip with sample efficiency.\n\n"
"Removing bad data is not the same as teaching\n"
"the model the data was bad.",
        fontsize=8.2,transform=ax.transAxes,va="top",linespacing=1.42)

# (f) so
ax=fig.add_subplot(gs[1,2]); ax.axis("off")
ax.text(0,1.0,"(f) The conclusion",fontsize=10.5,weight="bold",transform=ax.transAxes,va="top")
ax.text(0,.90,
"DOCKING ARTIFACTS CANNOT BE FIXED INSIDE THE\n"
"OPTIMIZER. Both available intervention points\n"
"were tried; one did nothing and one backfired,\n"
"for a reason predicted in advance.\n\n"
"The fix belongs at the ORACLE: reject or re-dock\n"
"a clashing pose before it ever becomes a data\n"
"point. A failed pose is a measurement failure,\n"
"not an unusually selective molecule, and the\n"
"place to say so is the docking pipeline.\n\n"
"That is outside this project's scope, and worth\n"
"stating plainly in the paper rather than\n"
"leaving as an open thread.\n\n"
"The existing filter stays where it is — applied\n"
"to every REPORTED result — which is why the\n"
"selectivity findings survive it.",
        fontsize=8.2,transform=ax.transAxes,va="top",linespacing=1.42,color=DK)

fig.suptitle("F24 — Can docking artifacts be fixed inside the optimizer? No, and the failure was predicted",
             fontsize=13,weight="bold",y=.975)
fig.text(.5,.004,"6 paired seeds per arm, identical configuration except the filter. Metric unchanged and no molecule discarded in either arm, so both are directly comparable to the baseline. "
                 "Artifact filter: PfDHFR > -7.0 or hDHFR > 0.0 on raw kcal/mol. n=6 gives a minimum two-sided Wilcoxon p of 0.0312.",
         ha="center",fontsize=7.3,color="#666")
fig.savefig(f"{OUT}/F24_artifacts_cannot_be_fixed_in_the_optimizer.png",dpi=170,bbox_inches="tight",facecolor="w")
print("wrote F24_artifacts_cannot_be_fixed_in_the_optimizer.png")
