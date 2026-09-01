import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.image as mpimg, os, textwrap
D="/Users/devansh/Downloads/aggregate_10seed/figures"
SHEET=[
 ("F0_flowchart_coded.png","F0  Pipeline, colour-coded by provenance","Which blocks are library code, which are adapted, which are ours."),
 ("F0_provenance.png","F0b  Provenance summary","How much of the stack we actually wrote."),
 ("F1_primary_outcome.png","F1  Primary outcome","MOGP 0.408 vs GP-MOBO 0.312 vs Greedy 0.195, 10 seeds. Holds 10/10."),
 ("F2_paired_effect.png","F2  Paired effect sizes","Per-seed differences with CIs and Holm correction."),
 ("F3_sample_efficiency.png","F3  Sample efficiency","Molecules needed to match the Greedy baseline's final score."),
 ("F4_compute_cost_REVISED.png","F4 REVISED  Compute cost","The 11.8x cost penalty was an unset default. Ranking INVERTS: MOGP becomes best per CPU-hour."),
 ("F5_pareto_structure.png","F5  Pareto-front structure","Front sizes and spread; GP-MOBO and Greedy are indistinguishable."),
 ("F6_validity_reliability.png","F6  Harness validity","Docking-oracle reliability, machine dependence, cache behaviour."),
 ("F7_oracle_capture.png","F7  Oracle-front capture","Corrected: the count-vs-volume claim inverts at equal scope."),
 ("F8_ideals_selectivity.png","F8  Selectivity, our actual objective","Artifact-filtered top-5 SI 4.37 / 3.57 / 3.12. MOGP 10/10 vs GP-MOBO."),
 ("F9_chemist_view.png","F9  The chemist's view","What the winning molecules look like, and the docking artifacts."),
 ("F10_alpha_vs_posterior.png","F10  Closing the 2x2","The 17x speedup was alpha, NOT the joint posterior (-0.0008)."),
 ("F11_what_alpha_is.png","F11  What alpha is","1,412x fewer boxes. It overestimates 18-90%, and the ranking is NOT preserved."),
 ("F12_icm_verdict.png","F12  Does coregionalization help?","No. 197/400 checkpoints across 10 seeds. Autokrigeability explains why."),
 ("F13_asymmetric_labels.png","F13  ...and when it DOES help","Break the co-location and it wins: p<=0.004 below 50% labels, monotone, rho=-1.000."),
]
have=[s for s in SHEET if os.path.exists(os.path.join(D,s[0]))]
ncol=3; nrow=(len(have)+ncol-1)//ncol
fig,axes=plt.subplots(nrow,ncol,figsize=(19,4.6*nrow))
axes=axes.ravel()
for ax,(f,title,sub) in zip(axes,have):
    ax.imshow(mpimg.imread(os.path.join(D,f))); ax.axis("off")
    ax.set_title(title,fontsize=11,weight="bold",loc="left",pad=3)
    ax.text(0,-0.045,"\n".join(textwrap.wrap(sub,72)),transform=ax.transAxes,
            fontsize=8.2,va="top",color="#444")
for ax in axes[len(have):]: ax.axis("off")
fig.suptitle("MOGP-NTD — every figure, in order",fontsize=19,weight="bold",y=.997)
fig.tight_layout(rect=[0,0,1,.985])
fig.savefig(f"{D}/F00_CONTACT_SHEET.png",dpi=105,bbox_inches="tight",facecolor="w")
print(f"wrote F00_CONTACT_SHEET.png with {len(have)} panels")
