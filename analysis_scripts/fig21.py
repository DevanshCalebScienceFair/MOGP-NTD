import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patches as mp, numpy as np, pandas as pd
from scipy.stats import wilcoxon
B="/Users/devansh/mogp-main-vscode/MOGP-NTD"; OUT="/Users/devansh/Downloads/aggregate_10seed/figures"
df=pd.read_csv(f"{B}/artifact_rejection_arm/scored.csv")
BASE,REJ,RED,GRN,GRY="#2166ac","#1a9850","#b2182b","#1a9850","#999999"
plt.rcParams.update({"font.size":9,"axes.grid":True,"grid.alpha":.22,
                     "axes.spines.top":False,"axes.spines.right":False})
fig=plt.figure(figsize=(15,9.2)); gs=fig.add_gridspec(2,3,hspace=.44,wspace=.30)

# (a) the idea
ax=fig.add_subplot(gs[0,0]); ax.axis("off")
ax.text(0,1.0,"(a) The idea",fontsize=10.5,weight="bold",transform=ax.transAxes,va="top")
ax.text(0,.88,
"A clashing pose scores POSITIVE on hDHFR.\n"
"Its apparent selectivity is therefore huge --\n"
"it looks like the perfect drug while binding\n"
"nothing at all.\n\n"
"42% of the campaign's raw top-5 by selectivity\n"
"were non-physical.\n\n"
"F19 showed that widening the hDHFR axis to\n"
"un-censor selectivity mostly BOUGHT more of\n"
"them. So attack it from the other side:\n\n"
"  keep clashing poses off the front that\n"
"  qNEHVI optimizes against, so the optimizer\n"
"  is never told that corner is already won.\n\n"
"Metric unchanged, molecules not discarded --\n"
"so this arm is directly comparable, no\n"
"re-scoring needed.",
        fontsize=8.2,transform=ax.transAxes,va="top",linespacing=1.42)

# (b) it engaged
ax=fig.add_subplot(gs[0,1])
ax.bar([0],[282],.5,color=GRY,label="fully evaluated")
ax.bar([1],[255],.5,color=BASE)
ax.bar([2],[27],.5,color=RED)
for i,v in enumerate([282,255,27]): ax.text(i,v+5,str(v),ha="center",fontsize=10,weight="bold")
ax.set_xticks([0,1,2])
ax.set_xticklabels(["evaluated","physical\n(on the front)","REJECTED\nfrom the front"],fontsize=8.5)
ax.set_ylabel("molecules (seed 0)"); ax.set_ylim(0,320)
ax.set_title("(b) The mechanism did engage\n9.6% of the front removed, from iteration 1 onward",
             fontsize=9.6,loc="left")
ax.text(.97,.92,"verified in the logs:\n\"3 artifacts rejected;\n qNEHVI baseline 37\"",
        transform=ax.transAxes,fontsize=7.6,color=GRN,family="monospace",ha="right",va="top")

# (c) and changed nothing
ax=fig.add_subplot(gs[0,2])
specs=[("final HV","hv",True),("AUC","auc",True),("top-20 SI","top20_SI",True),
       ("physical found","physical",True),("artifacts","artifacts",False),
       ("best PfDHFR","best_pf",False)]
rng=np.random.default_rng(0)
for i,(lab,col,hi) in enumerate(specs):
    a=df[f"rej_{col}"].values.astype(float); b=df[f"base_{col}"].values.astype(float)
    dd=(a-b) if hi else (b-a); sd=np.std(dd,ddof=1)
    sc=sd if sd>0 else 1.0
    boot=np.array([rng.choice(dd,len(dd),replace=True).mean() for _ in range(10000)])
    lo,up=np.percentile(boot,[2.5,97.5])
    ax.plot([lo/sc,up/sc],[i,i],color="#444",lw=2)
    ax.plot(dd.mean()/sc,i,"o",ms=8,color=GRY)
ax.axvline(0,color="#111",lw=1.6)
ax.set_yticks(range(6)); ax.set_yticklabels([s[0] for s in specs],fontsize=8.4); ax.invert_yaxis()
ax.set_xlabel("standardized paired effect (rejecting better →)")
ax.set_title("(c) …and changed nothing\nsix endpoints, every interval crosses zero",fontsize=9.6,loc="left")

# (d) the direct test went the WRONG way
ax=fig.add_subplot(gs[1,0])
x=np.arange(2); w=.36
ax.bar(x-w/2,[df.base_artifacts.mean(),df.base_top20_SI.mean()*15],w,color=BASE,label="baseline")
ax.bar(x+w/2,[df.rej_artifacts.mean(),df.rej_top20_SI.mean()*15],w,color=REJ,label="rejecting")
for i,(a,b) in enumerate([(df.base_artifacts.mean(),df.rej_artifacts.mean()),
                          (df.base_top20_SI.mean()*15,df.rej_top20_SI.mean()*15)]):
    ax.text(i-w/2,a+1,f"{a:.1f}" if i==0 else f"{a/15:.2f}",ha="center",fontsize=8.5)
    ax.text(i+w/2,b+1,f"{b:.1f}" if i==0 else f"{b/15:.2f}",ha="center",fontsize=8.5,weight="bold")
ax.set_xticks(x)
ax.set_xticklabels(["ARTIFACTS EVALUATED\n(the direct test)","top-20 selectivity\n(×15 for scale)"],fontsize=8)
ax.set_ylabel("count / scaled"); ax.set_ylim(0,54); ax.legend(fontsize=8,loc="upper left")
ax.set_title("(d) The direct test went the WRONG way\nrejecting evaluated MORE artifacts, not fewer",
             fontsize=9.6,loc="left",color=RED)
ax.text(.97,.97,"40.7 vs 38.7 — rejecting them from\nthe front did not stop the search\nfinding them.",
        transform=ax.transAxes,fontsize=7.8,color=RED,va="top",ha="right")

# (e) why: the leak is upstream
ax=fig.add_subplot(gs[1,1]); ax.axis("off")
ax.text(0,1.0,"(e) Why — the leak is upstream",fontsize=10.5,weight="bold",
        transform=ax.transAxes,va="top")
ax.text(0,.87,
"Artifacts do not enter through the FRONT.\n"
"They enter through the MODEL.\n\n"
"The GP is trained on every fully-docked\n"
"molecule, artifacts included. Seed 0:\n\n"
"   31 of 290 evaluated are artifacts (10.7%)\n"
"   their mean apparent selectivity  +0.68\n"
"   physical molecules               +0.13\n\n"
"So the model is fitted on labels saying\n"
"'extremely selective', learns the fragments\n"
"that produce them, and keeps proposing more.\n\n"
"Removing them from the baseline never touched\n"
"that. It changed which region looked unclaimed;\n"
"it did not change what the model believed.",
        fontsize=8.2,transform=ax.transAxes,va="top",linespacing=1.42)

# (f) next
ax=fig.add_subplot(gs[1,2]); ax.axis("off")
ax.text(0,1.0,"(f) What to try next",fontsize=10.5,weight="bold",transform=ax.transAxes,va="top")
ax.text(0,.88,
"Filter the GP TRAINING SET, not the front.\n\n"
"One line, in the same place `train_rows` is\n"
"already chosen. If the model never sees a\n"
"clashing pose labelled 'selective', it cannot\n"
"learn to chase them.\n\n"
"That is a sharper hypothesis than this arm\n"
"tested, and this arm is what produced it.\n\n"
"CAVEAT worth keeping: seed-level swings were\n"
"large in BOTH directions (+0.032, −0.024) and\n"
"net to nothing. Whatever the next arm shows,\n"
"six seeds will not settle it — this endpoint\n"
"is noisier than the effect being chased.",
        fontsize=8.2,transform=ax.transAxes,va="top",linespacing=1.42,color="#333")

fig.suptitle("F21 — Keeping clashing poses off the front: it engaged, and it changed nothing",
             fontsize=13,weight="bold",y=.975)
fig.text(.5,.004,"6 paired seeds, identical configuration except --reject-artifacts. Metric unchanged and no molecule discarded, so both arms are directly comparable with no re-scoring. "
                 "Artifact filter: PfDHFR > -7.0 or hDHFR > 0.0 on raw kcal/mol. n=6 gives a minimum two-sided Wilcoxon p of 0.0312.",
         ha="center",fontsize=7.3,color="#666")
fig.savefig(f"{OUT}/F21_artifact_rejection.png",dpi=170,bbox_inches="tight",facecolor="w")
print("wrote F21_artifact_rejection.png")
