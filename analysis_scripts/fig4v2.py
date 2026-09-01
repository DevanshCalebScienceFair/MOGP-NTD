import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
OUT="/Users/devansh/Downloads/aggregate_10seed/figures"
plt.rcParams.update({"font.size":9,"axes.grid":True,"grid.alpha":.25,"axes.spines.top":False,"axes.spines.right":False})

# campaign (measured, 10 seeds)
camp={"MOGP":(11.705,0.741,0.40790),"GP-MOBO":(0.760,0.160,0.31229),"Greedy":(0.547,0.056,0.19505)}
SPEEDUP=8.14/0.4794           # measured on the ICM ablation config
mogp_proj=camp["MOGP"][0]/SPEEDUP

fig=plt.figure(figsize=(14,8.6)); gs=fig.add_gridspec(2,3,hspace=.42,wspace=.30)

# (a) the claim as published
ax=fig.add_subplot(gs[0,0])
m=list(camp); hv=[camp[k][2] for k in m]; hr=[camp[k][0] for k in m]
hph=[h/t for h,t in zip(hv,hr)]
ax.bar(range(3),hph,color=["#b2182b","#2166ac","#999"])
for i,v in enumerate(hph): ax.text(i,v+.012,f"{v:.3f}",ha="center",fontsize=9,weight="bold")
ax.set_xticks(range(3)); ax.set_xticklabels(m,fontsize=8.5)
ax.set_ylabel("hypervolume per CPU-hour"); ax.set_ylim(0,.72)
ax.set_title("(a) WHAT F4 CLAIMED\nMOGP worst per CPU-hour by 11.5×",fontsize=9.5,loc="left",color="#b2182b")
ax.text(.5,.80,"the headline the paper\ncurrently carries",fontsize=7.5,color="#b2182b",ha="center",transform=ax.transAxes)

# (b) corrected
ax=fig.add_subplot(gs[0,1])
hph2=[camp["MOGP"][2]/mogp_proj]+hph[1:]
bars=ax.bar(range(3),hph2,color=["#1a9850","#2166ac","#999"])
bars[0].set_hatch("//"); bars[0].set_edgecolor("w")
for i,v in enumerate(hph2): ax.text(i,v+.012,f"{v:.3f}",ha="center",fontsize=9,weight="bold")
ax.set_xticks(range(3)); ax.set_xticklabels(["MOGP\n(corrected)","GP-MOBO","Greedy"],fontsize=8.5)
ax.set_ylabel("hypervolume per CPU-hour"); ax.set_ylim(0,.72)
ax.set_title("(b) AFTER SETTING alpha\nMOGP becomes the BEST, by 1.5×",fontsize=9.5,loc="left",color="#1a9850")
ax.text(.70,.72,"hatched = projected:\ncampaign cost\n÷ measured 17.0×",fontsize=7.5,color="#1a9850",ha="center",transform=ax.transAxes)

# (c) cost vs quality
ax=fig.add_subplot(gs[0,2])
for k,c,mk in [("MOGP","#b2182b","o"),("GP-MOBO","#2166ac","s"),("Greedy","#999","^")]:
    ax.errorbar(camp[k][0],camp[k][2],xerr=camp[k][1],fmt=mk,color=c,ms=9,capsize=3,label=k)
ax.plot(mogp_proj,camp["MOGP"][2],"*",color="#1a9850",ms=20,label="MOGP corrected")
ax.annotate("",xy=(mogp_proj,camp["MOGP"][2]),xytext=(camp["MOGP"][0],camp["MOGP"][2]),
            arrowprops=dict(arrowstyle="->",color="#1a9850",lw=2))
ax.text(2.6,.428,"17.0×",color="#1a9850",fontsize=11,weight="bold")
ax.set_xscale("log"); ax.set_xlabel("CPU-hours per campaign (log)"); ax.set_ylabel("final hypervolume")
ax.set_title("(c) Up and to the LEFT is better\nthe fix moves MOGP left, not down",fontsize=9.5,loc="left")
ax.legend(fontsize=7,loc="lower left"); ax.set_ylim(.15,.46)

# (d) where the 17x comes from
ax=fig.add_subplot(gs[1,0])
steps=["original\ndiag, a=0","+ alpha=1e-3","+ joint\n(bundles dedup)"]
hrs=[8.14,0.858,0.479]
ax.plot(range(3),hrs,"o-",color="#b2182b",lw=2,ms=9)
for i,h in enumerate(hrs): ax.text(i,h*1.28,f"{h:.2f} h",ha="center",fontsize=9,weight="bold")
ax.set_yscale("log"); ax.set_xticks(range(3)); ax.set_xticklabels(steps,fontsize=8)
ax.set_ylabel("wall clock, 290 molecules (h, log)"); ax.set_ylim(.3,14)
ax.text(.5,.62,"9.5×",color="#c33",fontsize=10,weight="bold",transform=ax.transAxes)
ax.text(1.5,.9,"1.8×",color="#c33",fontsize=10,weight="bold")
ax.set_title("(d) Two keyword arguments\nno new algorithm, no approximation to the model",fontsize=9.5,loc="left")

# (e) measured microbenchmark at baseline size 80
ax=fig.add_subplot(gs[1,1])
metrics=["wall clock\n(s)","peak RSS\n(GB)","boxes\n(hundreds)"]
before=[194.9,10.22,23.03]; after=[23.1,4.00,1.83]
x=np.arange(3); w=.36
ax.bar(x-w/2,before,w,label="alpha = 0.0",color="#b2182b")
ax.bar(x+w/2,after,w,label="alpha = 1e-3",color="#1a9850")
for i,(b,a) in enumerate(zip(before,after)):
    ax.text(i-w/2,b*1.06,f"{b:g}",ha="center",fontsize=7.5)
    ax.text(i+w/2,a*1.06,f"{a:g}",ha="center",fontsize=7.5)
ax.set_yscale("log"); ax.set_ylim(1,420)
ax.set_xticks(x); ax.set_xticklabels(metrics,fontsize=8)
ax.set_ylabel("log scale"); ax.legend(fontsize=8,loc="upper right")
ax.set_title("(e) One acquisition call, baseline size 80\nmeasured directly",fontsize=9.5,loc="left")


# (f) plain-language
ax=fig.add_subplot(gs[1,2]); ax.axis("off")
ax.text(0,1.0,"(f) What this means, plainly",fontsize=10,weight="bold",transform=ax.transAxes,va="top")
ax.text(0,.93,
"The paper currently says our method is the most\n"
"accurate but by far the most expensive to run.\n\n"
"The expense was not the method. It was one\n"
"BoTorch setting nobody set — the code used an\n"
"exact geometric calculation where the library's\n"
"own default is a fast approximation.\n\n"
"Turning it on: 17× less compute. Not the same\n"
"answers — it picks a different set of molecules —\n"
"but molecules of equal measured quality.\n\n"
"The trade-off in the paper's discussion — accuracy\n"
"bought with compute — was never real. So was the\n"
"reason we capped the candidate pool at 2,000.\n\n"
"Practical effect: a campaign that took a full\n"
"working day of compute now takes half an hour,\n"
"which is what made replicating across 10 random\n"
"restarts affordable at all.\n\n"
"Mechanically: qNEHVI carves the objective space\n"
"into boxes to measure improvement. alpha lets it\n"
"use slightly coarser ones — 183 boxes instead of\n"
"2,303 — which is where the time and the memory go.",
fontsize=7.7,transform=ax.transAxes,va="top",linespacing=1.32)

fig.suptitle("F4 (revised) — The cost result was an unset default, not a property of the method",
             fontsize=12.5,weight="bold",y=.985)
fig.text(.5,.005,"Campaign bars: 10 seeds, measured. MOGP-corrected: campaign cost divided by the 17.0× measured on the matched ICM ablation config (8.14 h -> 0.48 h, 290 molecules, pool=2000). "
                 "The projection assumes the factor transfers; it is dominated by box-decomposition cost, which scales with Pareto-front size, and front sizes were comparable (162 vs 179).",
         ha="center",fontsize=7,color="#666")
fig.savefig(f"{OUT}/F4_compute_cost_REVISED.png",dpi=170,bbox_inches="tight",facecolor="w")
print("wrote F4_compute_cost_REVISED.png")
print(f"  speedup used: {SPEEDUP:.2f}x ; projected MOGP campaign = {mogp_proj:.3f} h")
print(f"  hv/hour: MOGP {camp['MOGP'][2]/mogp_proj:.3f} vs GP-MOBO {camp['GP-MOBO'][2]/camp['GP-MOBO'][0]:.3f} "
      f"-> MOGP better by {(camp['MOGP'][2]/mogp_proj)/(camp['GP-MOBO'][2]/camp['GP-MOBO'][0]):.2f}x")
print(f"  old claim: MOGP worse by {(camp['GP-MOBO'][2]/camp['GP-MOBO'][0])/(camp['MOGP'][2]/camp['MOGP'][0]):.2f}x")
