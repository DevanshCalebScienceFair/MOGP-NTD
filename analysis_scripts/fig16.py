import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from scipy.stats import wilcoxon
B="/Users/devansh/mogp-main-vscode/MOGP-NTD"; OUT="/Users/devansh/Downloads/aggregate_10seed/figures"
hv=pd.read_csv(f"{B}/analysis_scripts/f_sweep_metrics.csv")
nm=pd.read_csv(f"{B}/analysis_scripts/f_sweep_nominated.csv")
Fs=[1.00,0.75,0.50,0.25]
BLU,ORA,RED,GRN,GRY="#2166ac","#e08214","#b2182b","#1a9850","#999999"
plt.rcParams.update({"font.size":9,"axes.grid":True,"grid.alpha":.22,
                     "axes.spines.top":False,"axes.spines.right":False})
fig=plt.figure(figsize=(15.5,9.8)); gs=fig.add_gridspec(2,3,hspace=.40,wspace=.30)

def pvals(df,col,hi=True):
    ref=df[df.F==1.00].set_index("seed")[col]; out={}
    for F in (0.75,0.50,0.25):
        r=df[df.F==F].set_index("seed")[col]; c=ref.index.intersection(r.index)
        d=(r[c]-ref[c]).values if hi else (ref[c]-r[c]).values
        out[F]=(wilcoxon(d).pvalue if len(set(d))>1 else np.nan, int((d>0).sum()), len(d))
    return out

# (a) design
ax=fig.add_subplot(gs[0,0]); ax.axis("off")
ax.text(0,1.0,"(a) Four designs, one budget",fontsize=10.2,weight="bold",transform=ax.transAxes,va="top")
g=hv.groupby("F")
lines=["  F      molecules   hDHFR labels   dock calls",""]
for F in Fs:
    r=g.get_group(F)
    lines.append(f" {F:.2f}        {r.molecules.mean():4.0f}          {r.hdhfr_labels.mean():4.0f}          {r.calls.mean():4.0f}")
ax.text(0,.86,"\n".join(lines),fontsize=9,family="monospace",transform=ax.transAxes,va="top",linespacing=1.5)
ax.text(0,.44,
"F = the fraction of each batch that ALSO gets\n"
"docked against hDHFR, the second target.\n\n"
"Budgets match within 0.5%. Everything else is\n"
"held fixed: same Hadamard ICM, same seeds, same\n"
"initial molecules, same acquisition, same pool.\n\n"
"Lower F buys MORE MOLECULES and pays in DIRECT\n"
"MEASUREMENT of a target that is half the score.",
        fontsize=8.2,transform=ax.transAxes,va="top",linespacing=1.42,color="#444")

# (b) hypervolume
ax=fig.add_subplot(gs[0,1])
m=[hv[hv.F==F].hv.mean() for F in Fs]; s=[hv[hv.F==F].hv.std(ddof=1)/np.sqrt(6) for F in Fs]
ax.errorbar(Fs,m,yerr=s,fmt="o-",color=RED,lw=2.4,ms=9,capsize=4)
for F in hv.F.unique():
    ax.scatter([F]*6,hv[hv.F==F].hv,s=14,color=RED,alpha=.28)
P=pvals(hv,"hv")
for F in (0.75,0.50,0.25):
    ax.text(F,hv[hv.F==F].hv.max()+.012,f"p={P[F][0]:.3f}",ha="center",fontsize=7.6,color=RED,weight="bold")
ax.set_xlabel("F — fraction docked against hDHFR"); ax.set_ylabel("final hypervolume")
ax.invert_xaxis()
ax.set_title("(b) HYPERVOLUME: monotone, every step significant\nSpearman = +1.000 · 0/18 comparisons favour less",
             fontsize=9.4,loc="left")
ax.text(.04,.10,"BUT this endpoint is biased toward high F:\na molecule missing hDHFR cannot sit on a front.",
        transform=ax.transAxes,fontsize=7.4,color=RED)

# (c) unbiased nomination
ax=fig.add_subplot(gs[0,2])
m2=[nm[nm.F==F].mean_SI.mean() for F in Fs]; s2=[nm[nm.F==F].mean_SI.std(ddof=1)/np.sqrt(6) for F in Fs]
ax.errorbar(Fs,m2,yerr=s2,fmt="o-",color=GRN,lw=2.4,ms=9,capsize=4)
for F in Fs: ax.scatter([F]*6,nm[nm.F==F].mean_SI,s=14,color=GRN,alpha=.28)
Q=pvals(nm,"mean_SI")
for F in (0.75,0.50,0.25):
    p,w,n=Q[F]
    ax.text(F,nm[nm.F==F].mean_SI.max()+.07,
            ("p=%.3f\n%s"%(p,"SIG" if p<.05 else "tie")),ha="center",fontsize=7.6,
            color=RED if p<.05 else "#666",weight="bold")
ax.axvspan(0.60,1.05,color=GRN,alpha=.10)
ax.text(0.80,0.35,"indistinguishable\nfrom F=1.00",ha="center",fontsize=8,color=GRN,weight="bold")
ax.set_xlabel("F — fraction docked against hDHFR"); ax.set_ylabel("mean TRUE selectivity of 20 nominees")
ax.invert_xaxis()
ax.set_title("(c) UNBIASED: only F=0.25 is actually worse\nsame library ranked by every arm, 20 nominees docked for real",
             fontsize=9.4,loc="left",color=GRN)

# (d) best find per seed
ax=fig.add_subplot(gs[1,0])
piv=nm.pivot(index="seed",columns="F",values="best_SI")[Fs]
for s_ in piv.index: ax.plot(Fs,piv.loc[s_],"-o",ms=5,lw=1.2,alpha=.65,color=GRY)
ax.plot(Fs,[piv[F].mean() for F in Fs],"-o",color=ORA,lw=2.8,ms=10,label="mean",zorder=5)
ax.invert_xaxis(); ax.set_xlabel("F"); ax.set_ylabel("BEST true selectivity found")
ax.legend(fontsize=8)
ax.set_title("(d) The best single find peaks in the MIDDLE\n5.47 at F=0.75 vs 4.78 at F=1.00 (p=0.25, not significant)",
             fontsize=9.4,loc="left")
ax.text(.04,.40,"F=1.00 returns the identical best molecule\n(SI 4.240) in 5 of 6 seeds — it converges.\nBroader arms explore and sometimes find\nbetter, but too noisily to call at n=6.",
        transform=ax.transAxes,fontsize=7.5,va="top",color="#444")

# (e) the disagreement
ax=fig.add_subplot(gs[1,1])
x=np.arange(3); w=.36
def eff(df,col):
    ref=df[df.F==1.00].set_index("seed")[col]; o=[]
    for F in (0.75,0.50,0.25):
        r=df[df.F==F].set_index("seed")[col]; c=ref.index.intersection(r.index)
        d=(r[c]-ref[c]).values; o.append(d.mean()/(d.std(ddof=1) or 1))
    return o
ax.bar(x-w/2,eff(hv,"hv"),w,color=RED,label="hypervolume (biased)")
ax.bar(x+w/2,eff(nm,"mean_SI"),w,color=GRN,label="nomination (unbiased)")
ax.axhline(0,color="#111",lw=1.2)
for i,F in enumerate((0.75,0.50,0.25)):
    ax.text(i-w/2,.10,f"p={pvals(hv,'hv')[F][0]:.3f}",ha="center",fontsize=7,color=RED,weight="bold")
    ax.text(i+w/2,.36,f"p={pvals(nm,'mean_SI')[F][0]:.3f}",ha="center",fontsize=7,color=GRN,weight="bold")
ax.set_ylim(-5.0,.72)
ax.set_xticks(x); ax.set_xticklabels(["F=0.75","F=0.50","F=0.25"])
ax.set_ylabel("standardized paired effect vs F=1.00")
ax.legend(fontsize=8,loc="lower right")
ax.set_title("(e) The two endpoints DISAGREE at F=0.75 and 0.50\nand agree at F=0.25",fontsize=9.4,loc="left")

# (f) recommendation
ax=fig.add_subplot(gs[1,2]); ax.axis("off")
ax.text(0,1.0,"(f) What to actually do",fontsize=10.2,weight="bold",transform=ax.transAxes,va="top")
ax.text(0,.90,
"DO NOT go below ~50%. At F=0.25 both endpoints\n"
"agree the design is worse (p=0.031 each).\n\n"
"BETWEEN 50% and 100% the endpoints disagree:\n"
"  · hypervolume prefers 100% (p=0.031)\n"
"  · shortlist quality cannot tell them apart\n"
"    (p=0.69; F=0.50 beats F=1.00 in 3/6 seeds)\n\n"
"Hypervolume is biased toward high F and the\n"
"nomination test is not, so for the question\n"
"'which design hands me better candidates?'\n"
"the honest answer is that halving the second\n"
"assay costs nothing we can measure — and buys\n"
"33% more molecules explored.\n\n"
"CAVEAT, and it is real: n=6. 'No evidence of a\n"
"difference' is not 'evidence of no difference'.\n"
"The point estimates still favour F=1.00.",
        fontsize=8.0,transform=ax.transAxes,va="top",linespacing=1.38,color="#333")

fig.suptitle("F16 — How much of the expensive second assay should you buy?",
             fontsize=13.5,weight="bold",y=.975)
fig.text(.5,.004,"6 paired seeds per F, matched docking budget (557-559 calls, within 0.5%). Identical Hadamard ICM, seeds, initial molecules, acquisition and 2,000-candidate pool throughout; only --hdhfr-fraction differs. "
                 "Wilcoxon signed-rank against F=1.00; n=6 gives a minimum two-sided p of 0.0312. Error bars are standard errors; faint points are individual seeds.",
         ha="center",fontsize=7.2,color="#666")
fig.savefig(f"{OUT}/F16_f_sweep.png",dpi=170,bbox_inches="tight",facecolor="w")
print("wrote F16_f_sweep.png")
