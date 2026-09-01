import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd, os
from scipy.stats import wilcoxon
S="/private/tmp/claude-502/-Users-devansh/db50257a-6170-49a1-bcb6-f56dd539b550/scratchpad"
BASE="/Users/devansh/mogp-main-vscode/MOGP-NTD"; OUT="/Users/devansh/Downloads/aggregate_10seed/figures"
df=pd.read_csv(f"{S}/multiseed_metrics.csv")
def path(m,s): return (f"{BASE}/ablation_joint_alpha/{m}_seed0/history.csv" if s==0
                       else f"{BASE}/ablation_multiseed/{m}_seed{s}/history.csv")
seeds=list(df.seed.astype(int))
plt.rcParams.update({"font.size":9,"axes.grid":True,"grid.alpha":.25,
                     "axes.spines.top":False,"axes.spines.right":False})
fig=plt.figure(figsize=(15,9.4)); gs=fig.add_gridspec(2,3,hspace=.40,wspace=.28)

# (a) paired slope chart, final HV
ax=fig.add_subplot(gs[0,0])
for _,r in df.iterrows():
    win = r["icm_final"]>r["ind_final"]
    ax.plot([0,1],[r["ind_final"],r["icm_final"]],"-o",ms=5,lw=1.4,
            color="#1a9850" if win else "#b2182b",alpha=.85)
ax.set_xticks([0,1]); ax.set_xticklabels(["independent","ICM"])
ax.set_ylabel("final hypervolume")
ax.set_title(f"(a) Paired, {len(df)} seeds\nICM better in {int((df.icm_final>df.ind_final).sum())}/{len(df)} — a coin flip",
             fontsize=9.5,loc="left")
ax.text(.02,.03,"green = ICM wins that seed\nred = independent wins",fontsize=7.4,
        transform=ax.transAxes,color="#555")

# (b) checkpoint lead per seed
ax=fig.add_subplot(gs[0,1])
lead=[]
for s in seeds:
    A=pd.read_csv(path("coregionalized",s)); B=pd.read_csv(path("independent",s))
    m=A.merge(B,on="n_evaluated",suffixes=("_a","_b")); d=m.hypervolume_a-m.hypervolume_b
    lead.append(int((d>0).sum()))
cols=["#1a9850" if l>25 else "#b2182b" for l in lead]
ax.bar(range(len(seeds)),lead,color=cols)
ax.axhline(25,ls="--",color="#111",lw=1.4)
ax.text(len(seeds)-.4,26.5,"coin flip",fontsize=7.5,ha="right")
for i,l in enumerate(lead): ax.text(i,l+1,str(l),ha="center",fontsize=7.5)
ax.set_xticks(range(len(seeds))); ax.set_xticklabels([f"s{s}" for s in seeds],fontsize=8)
ax.set_ylabel("checkpoints (of 50) where ICM leads"); ax.set_ylim(0,55)
ax.set_title(f"(b) The seed-0 result did not replicate\ntotal {sum(lead)}/{50*len(seeds)} = {100*sum(lead)/(50*len(seeds)):.0f}%",
             fontsize=9.5,loc="left")
ax.annotate("seed 0: 38/50\n(what we reported)",xy=(0,38),xytext=(1.1,46),fontsize=7.4,
            arrowprops=dict(arrowstyle="->",color="#555"))

# (c) forest plot
ax=fig.add_subplot(gs[0,2])
rng=np.random.default_rng(0)
specs=[("final","final HV",False),("auc","AUC of HV curve",False),
       ("n@0.3","molecules to HV 0.30",True),("n@0.34","molecules to HV 0.34",True),
       ("n@0.36","molecules to HV 0.36",True),("n@0.38","molecules to HV 0.38",True)]
ys=[]; labs=[]
for i,(k,lab,low) in enumerate(specs):
    a=df[f"icm_{k}"].values; b=df[f"ind_{k}"].values
    ok=np.isfinite(a)&np.isfinite(b); a,b=a[ok],b[ok]
    d=(b-a) if low else (a-b)
    d=d/np.std(d,ddof=1) if np.std(d,ddof=1)>0 else d      # standardized
    boot=np.array([rng.choice(d,len(d),replace=True).mean() for _ in range(10000)])
    lo,hi=np.percentile(boot,[2.5,97.5])
    ax.plot([lo,hi],[i,i],color="#444",lw=2)
    ax.plot(d.mean(),i,"o",ms=8,color="#1a9850" if lo>0 else "#777")
    ys.append(i); labs.append(lab)
ax.axvline(0,color="#b2182b",lw=1.6)
ax.set_yticks(ys); ax.set_yticklabels(labs,fontsize=8); ax.invert_yaxis()
ax.set_xlabel("standardized paired effect (ICM better →)")
ax.set_title("(c) Every interval crosses zero\nno endpoint separates the two models",
             fontsize=9.5,loc="left",color="#b2182b")

# (d) two seeds, opposite conclusions
ax=fig.add_subplot(gs[1,0])
for s,st in [(0,"-"),(6,"--")]:
    A=pd.read_csv(path("coregionalized",s)); B=pd.read_csv(path("independent",s))
    ax.plot(A.n_evaluated,A.hypervolume,st,color="#1a9850",lw=1.9,label=f"ICM seed {s}")
    ax.plot(B.n_evaluated,B.hypervolume,st,color="#b2182b",lw=1.9,label=f"independent seed {s}")
ax.set_xlabel("molecules docked"); ax.set_ylabel("hypervolume")
ax.legend(fontsize=7,loc="lower right")
ax.set_title("(d) Why one seed proves nothing\nseed 0 ICM leads 38/50; seed 6 leads 0/50",
             fontsize=9.5,loc="left")

# (e) the mechanism
ax=fig.add_subplot(gs[1,1]); ax.axis("off")
ax.text(0,1.0,"(e) There is a reason it cannot help",fontsize=10,weight="bold",
        transform=ax.transAxes,va="top")
ax.text(0,.90,
"A multi-output GP earns its keep by borrowing:\n"
"knowing PfDHFR should sharpen its guess about\n"
"hDHFR. That only pays when some molecules are\n"
"missing one of the two.\n\n"
"In our design NOTHING is missing. Every molecule\n"
"is docked against both targets, or neither:\n\n"
"      1,683 / 1,740 have BOTH\n"
"          0 / 1,740 have exactly ONE   (0.00%)\n\n"
"That is a complete block design — the exact\n"
"autokrigeability condition. Bonilla, Chai &\n"
"Williams (2008) §2.3: with tasks observed at the\n"
"same inputs, the ICM posterior MEAN collapses to\n"
"independent GPs per task. Transfer cancels.\n\n"
"The tasks are strongly correlated (ρ = 0.788).\n"
"It does not help. There is nothing to transfer.",
        fontsize=8.0,transform=ax.transAxes,va="top",linespacing=1.42)

# (f) so what
ax=fig.add_subplot(gs[1,2]); ax.axis("off")
ax.text(0,1.0,"(f) What this means, plainly",fontsize=10,weight="bold",
        transform=ax.transAxes,va="top")
ax.text(0,.90,
"The coregionalized GP — the piece we treated as\n"
"the novel core — does not beat a plain independent\n"
"GP on this problem, on any endpoint we measured.\n\n"
"This is a real finding, not a failure. It is\n"
"mechanistically explained, it was invisible at one\n"
"seed, and it tells us exactly where to go next:\n\n"
"To make coregionalization pay, BREAK the\n"
"co-location. Dock cheaply or partially on one\n"
"target so some molecules carry only one label.\n"
"Then the model has something to borrow, and\n"
"autokrigeability no longer applies.\n\n"
"That is a sharper contribution than 'we used a\n"
"multi-output GP', and it is testable.",
        fontsize=8.0,transform=ax.transAxes,va="top",linespacing=1.42)

fig.suptitle("F12 — Does coregionalization actually help? Across seeds: no.",
             fontsize=13,weight="bold",y=.975)
fig.text(.5,.004,f"{len(df)} paired seeds, identical initial molecules and configuration per seed; only the GP model differs. "
                 "posterior=joint, alpha=1e-3, pool=2000, 290 molecules. Seeds 8-9 still running. "
                 "Autokrigeability is exact for equal per-task noise and approximate otherwise.",
         ha="center",fontsize=7.4,color="#666")
fig.savefig(f"{OUT}/F12_icm_verdict.png",dpi=165,bbox_inches="tight",facecolor="w")
print("wrote F12_icm_verdict.png")
