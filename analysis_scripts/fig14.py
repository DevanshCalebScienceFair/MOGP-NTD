import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from scipy.stats import wilcoxon
B="/Users/devansh/mogp-main-vscode/MOGP-NTD"; OUT="/Users/devansh/Downloads/aggregate_10seed/figures"
sc=pd.read_csv(f"{B}/asym_campaign/scored.csv"); nom=pd.read_csv(f"{B}/asym_campaign/nominated_scored.csv")
F,A="#2166ac","#e08214"
plt.rcParams.update({"font.size":9,"axes.grid":True,"grid.alpha":.25,
                     "axes.spines.top":False,"axes.spines.right":False})
fig=plt.figure(figsize=(15,9.4)); gs=fig.add_gridspec(2,3,hspace=.40,wspace=.29)

# (a) the trade
ax=fig.add_subplot(gs[0,0]); ax.axis("off")
ax.text(0,1.0,"(a) The trade, at a matched docking budget",fontsize=10,weight="bold",
        transform=ax.transAxes,va="top")
ax.text(0,.88,
"Both arms spend ~558 dock calls. Nothing else\n"
"is equalised.\n\n"
"                       FULL        ASYM\n"
"  hDHFR docked on     100%         25%\n"
"  molecules reached    290    ->   465   (1.6x)\n"
"  hDHFR labels         280    <-   105   (2.7x)\n\n"
"So the asymmetric arm buys BREADTH and pays\n"
"for it in DIRECT MEASUREMENT of the second\n"
"target -- which is half the selectivity score.\n\n"
"Both arms use the SAME model (Hadamard ICM)\n"
"and the same seed, initial molecules,\n"
"acquisition and pool cap. The only difference\n"
"is --hdhfr-fraction.",
        fontsize=8.3,transform=ax.transAxes,va="top",linespacing=1.42,family="monospace")

# (b) hypervolume, biased
ax=fig.add_subplot(gs[0,1])
for _,r in sc.iterrows():
    ax.plot([0,1],[r.full_final_hv,r.asym_final_hv],"-o",color="#b2182b",ms=5,lw=1.3,alpha=.8)
ax.set_xticks([0,1]); ax.set_xticklabels(["full\n(100% hDHFR)","asym\n(25% hDHFR)"])
ax.set_ylabel("final hypervolume")
ax.set_title("(b) Hypervolume: full wins 6/6, p=0.031\nBUT this endpoint is biased -- see (c)",
             fontsize=9.5,loc="left")
ax.text(.03,.06,"asym's front is built from 105 of its\n465 molecules; a molecule missing\nhDHFR cannot sit on a Pareto front.",
        transform=ax.transAxes,fontsize=7.3,color="#b2182b")

# (c) why both simple endpoints are biased
ax=fig.add_subplot(gs[0,2])
x=np.arange(2); w=.36
meas=[sc.full_physical.mean(),sc.asym_physical.mean()]
ax.bar(x-w/2,[290,465],w,label="molecules evaluated",color="#bbb")
ax.bar(x+w/2,meas,w,label="usable for scoring\n(both labels, physical)",color=[F,A])
for i,(a,b) in enumerate(zip([290,465],meas)):
    ax.text(i-w/2,a+8,f"{a:.0f}",ha="center",fontsize=8)
    ax.text(i+w/2,b+8,f"{b:.0f}",ha="center",fontsize=8,weight="bold")
ax.set_xticks(x); ax.set_xticklabels(["full","asym"]); ax.set_ylabel("molecules")
ax.legend(fontsize=7.5,loc="upper left"); ax.set_ylim(0,560)
ax.set_title("(c) Why (b) cannot be trusted alone\nfull scores from 251 candidates, asym from 94",
             fontsize=9.5,loc="left",color="#b2182b")

# (d) THE UNBIASED TEST
ax=fig.add_subplot(gs[1,0])
for _,r in nom.iterrows():
    ax.plot([0,1],[r.full_mean_SI,r.asym_mean_SI],"-o",color="#1a9850",ms=5,lw=1.3,alpha=.85)
ax.set_xticks([0,1]); ax.set_xticklabels(["full","asym"])
ax.set_ylabel("mean TRUE selectivity of the 20 nominated")
d=nom.full_mean_SI-nom.asym_mean_SI
ax.set_title(f"(d) UNBIASED: rank the SAME library,\ndock 20 each. full wins 6/6, p=0.031",
             fontsize=9.5,loc="left")
ax.text(.03,.93,"Both arms rank the same ~26,300\nunmeasured molecules and pay the\nsame 40 verification docks.\nNeither is helped by how many it\nhappened to measure.",
        transform=ax.transAxes,fontsize=7.3,va="top",color="#444")

# (e) but three of four unbiased endpoints TIE
ax=fig.add_subplot(gs[1,1])
labs=["mean SI\nof nominees","best SI\nfound","physical\n/20","best PfDHFR"]
res=[]
for col,hi in [("mean_SI",1),("best_SI",1),("physical",1),("best_PfDHFR",0)]:
    a,b=nom[f"asym_{col}"].values,nom[f"full_{col}"].values
    dd=(a-b) if hi else (b-a)
    res.append((dd.mean()/ (np.std(dd,ddof=1) or 1), wilcoxon(dd).pvalue, int((dd>0).sum())))
xs=np.arange(4)
cols=["#b2182b" if p<0.05 else "#999" for _,p,_ in res]
ax.bar(xs,[r[0] for r in res],color=cols)
ax.axhline(0,color="#111",lw=1.2)
for i,(e,p,w_) in enumerate(res):
    ax.text(i,e+(0.06 if e>=0 else -0.16),f"p={p:.3f}\nasym {w_}/6",ha="center",fontsize=7.2)
ax.set_xticks(xs); ax.set_xticklabels(labs,fontsize=8)
ax.set_ylabel("standardized paired effect (asym better →)")
ax.set_ylim(-2.6,1.2)
ax.set_title("(e) Only the AVERAGE separates them\nvalidity and best-find are ties",fontsize=9.5,loc="left")

# (f) reconciliation
ax=fig.add_subplot(gs[1,2]); ax.axis("off")
ax.text(0,1.0,"(f) This is what F13 predicted",fontsize=10,weight="bold",transform=ax.transAxes,va="top")
ax.text(0,.90,
"F13 measured hDHFR prediction error:\n\n"
"   ICM, 100% of hDHFR labels   RMSE 1.360\n"
"   ICM,  25% + borrowing       RMSE 1.430\n\n"
"Borrowing recovers PART of what missing\n"
"labels cost. It does not make missing labels\n"
"BETTER than having them.\n\n"
"This campaign spent its budget on 1.6x more\n"
"molecules and 2.7x fewer hDHFR labels. F13\n"
"already said that trade loses on hDHFR --\n"
"and hDHFR is half the selectivity objective.\n\n"
"THE STATEMENT THAT SURVIVES:\n"
"Coregionalization is a mitigation for missing\n"
"labels, not a reason to create them.\n\n"
"  - useless when nothing is missing   (F12)\n"
"  - genuinely helps when it is        (F13)\n"
"  - not worth engineering gaps to get (here)",
        fontsize=8.0,transform=ax.transAxes,va="top",linespacing=1.40)

fig.suptitle("F14 — Closed loop: does buying breadth beat buying complete measurement? No.",
             fontsize=13,weight="bold",y=.975)
fig.text(.5,.004,"6 paired seeds. Matched docking budget (~558 calls per arm, ratio 0.996). Both arms: Hadamard ICM, joint posterior, alpha=1e-3, pool=2000, identical initial molecules per seed. "
                 "Panels (b)-(c) are structurally biased toward the full arm and are shown WITH that bias; panels (d)-(e) are the unbiased nomination test. n=6 gives a minimum Wilcoxon p of 0.0312.",
         ha="center",fontsize=7.2,color="#666")
fig.savefig(f"{OUT}/F14_closed_loop_verdict.png",dpi=165,bbox_inches="tight",facecolor="w")
print("wrote F14_closed_loop_verdict.png")
