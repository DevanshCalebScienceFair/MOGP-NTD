"""The 5-to-2 pivot, scored as an ATTRIBUTION CHAIN rather than one comparison.

NO RE-SCORING IS NEEDED, and it is worth saying why. Only the ACQUISITION sees
two objectives; `BOLoop._hypervolume` still calls `compute_hypervolume` over all
five, so every arm's reported hypervolume is in the same published frame and they
compare directly. (Verified on a smoke run: reported 0.0341, recomputed
5-objective 0.0341.) This is the opposite of the hDHFR bound arm, where the
metric itself moved and re-scoring was mandatory.

WHY THREE ARMS. The headline change bundles two independent edits, and the pitch
credits them separately, so one comparison cannot support it:

    A  model_comparison/hadamard_seed*   2,000 draw, 5 objectives
         |  + --admet-constraints                  <- THE PIVOT
    B  pivot_ablation/ablate_seed*       2,000 draw, 2 objectives
         |  - --acquisition-pool-size              <- THE UNCAP
    D  pivot_arm/pivot_seed*             full library, 2 objectives

A->B is the pivot alone, B->D is the uncap alone, A->D is what shipped. Each
neighbouring pair differs in exactly one flag; model, posterior, alpha, n_init,
batch size, iteration count and seed are identical throughout.

The pivot's real deliverable is not hypervolume, it is that every molecule it
hands a chemist clears the safety bar by construction. That is `admet_pass_pct`,
and it is the number to lead with.
"""
import os, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
B = "/Users/devansh/mogp-main-vscode/MOGP-NTD"; sys.path.insert(0, B)
import evaluation as E
from data import ADMET_COLUMNS   # NOT TASK_NAMES order; passes_admet resolves by name
import torch
from botorch.utils.multi_objective.pareto import is_non_dominated

DOCK = [0, 1]
ARMS = [
    ("A_base",  "model_comparison/hadamard_seed{s}", "2,000 draw, 5 obj"),
    ("B_pivot", "pivot_ablation/ablate_seed{s}",     "2,000 draw, 2 obj"),
    ("D_full",  "pivot_arm/pivot_seed{s}",           "full library, 2 obj"),
]
# Each step in the chain, and the single flag that produces it.
STEPS = [("A_base", "B_pivot", "THE PIVOT  (+--admet-constraints)"),
         ("B_pivot", "D_full", "THE UNCAP  (-pool cap)"),
         ("A_base", "D_full",  "COMBINED   (what shipped)")]


def summarize(d):
    e = f"{d}/evaluated.csv"
    if not os.path.exists(e):
        return None
    ev = pd.read_csv(e).dropna(subset=list(E.TASK_NAMES))
    if not len(ev):
        return None
    Y = ev[list(E.TASK_NAMES)].to_numpy(float)
    phys = E.is_physical(Y)
    sub = ev[phys]
    Yn = torch.as_tensor(E.normalize(Y), dtype=torch.double)
    front5, front2 = int(is_non_dominated(Yn).sum()), int(is_non_dominated(Yn[:, DOCK]).sum())
    # The safety bar, applied post hoc to whatever each arm chose to evaluate.
    # Resolved BY NAME inside passes_admet -- ADMET_COLUMNS is not TASK_NAMES order.
    admet = ev[list(ADMET_COLUMNS)].to_numpy(float)
    npass = int(E.passes_admet(admet).sum())
    return dict(
        hv=float(E.compute_hypervolume(Y)), n=len(ev), physical=int(phys.sum()),
        admet_pass=npass, admet_pass_pct=100.0 * npass / len(ev),
        front5_pct=100.0 * front5 / len(ev), front2_pct=100.0 * front2 / len(ev),
        top20_SI=float(sub.nlargest(20, "Selectivity_Index").Selectivity_Index.mean())
        if len(sub) else np.nan,
        best_SI=float(sub.Selectivity_Index.max()) if len(sub) else np.nan,
        best_pf=float(sub.PfDHFR_Docking.min()) if len(sub) else np.nan,
    )


rows = []
for s in range(10):
    got = {k: summarize(f"{B}/{tpl.format(s=s)}") for k, tpl, _ in ARMS}
    if got["A_base"] is None:
        continue
    r = {"seed": s}
    for k, v in got.items():
        if v is not None:
            r.update({f"{k}__{m}": x for m, x in v.items()})
    rows.append(r)
if not rows:
    print("No seeds found. Run ./run_pivot_arm.sh and ./run_pivot_ablation.sh"); sys.exit(0)
df = pd.DataFrame(rows)
rng = np.random.default_rng(0)

print("=" * 100)
print("ARM INVENTORY")
print("=" * 100)
for k, tpl, desc in ARMS:
    col = f"{k}__hv"
    n = int(df[col].notna().sum()) if col in df else 0
    print(f"  {k:9s} {desc:22s} seeds complete: {n}")
    if n == 0:
        print(f"            (missing: {tpl.format(s='*')})")

METRICS = [("hypervolume (5-obj, PUBLISHED)", "hv", True),
           ("ADMET pass rate %", "admet_pass_pct", True),
           ("top-20 selectivity", "top20_SI", True),
           ("best selectivity", "best_SI", True),
           ("physical molecules", "physical", True),
           ("best PfDHFR (kcal/mol)", "best_pf", False)]

for lo_arm, hi_arm, title in STEPS:
    la, ha = f"{lo_arm}__", f"{hi_arm}__"
    if f"{la}hv" not in df or f"{ha}hv" not in df:
        print(f"\n--- {title}: arm missing, skipped ---"); continue
    print("\n" + "=" * 100)
    print(f"{title}   [{lo_arm} -> {hi_arm}]")
    print("=" * 100)
    for label, col, hi in METRICS:
        a = df.get(f"{ha}{col}", pd.Series(dtype=float)).values.astype(float)
        b = df.get(f"{la}{col}", pd.Series(dtype=float)).values.astype(float)
        if not len(a) or not len(b):
            print(f"  {label:32s} (missing)"); continue
        ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]
        if len(a) < 2:
            print(f"  {label:32s} n={len(a)}, no test"); continue
        d = (a - b) if hi else (b - a)
        p = wilcoxon(d).pvalue if len(set(d)) > 1 else float("nan")
        boot = [rng.choice(d, len(d), replace=True).mean() for _ in range(10000)]
        lo_ci, up_ci = np.percentile(boot, [2.5, 97.5])
        print(f"  {label:32s} {b.mean():9.4f} -> {a.mean():9.4f} | "
              f"delta {d.mean():+8.4f} [{lo_ci:+.4f},{up_ci:+.4f}] | "
              f"wins {int((d>0).sum())}/{len(d)} | p={p:.4f}")

# Do the two single-flag effects add up to the shipped effect? If they do not,
# the two changes interact and neither can be quoted on its own.
if all(f"{k}__hv" in df for k, _, _ in ARMS):
    for col, lab in [("hv", "hypervolume"), ("admet_pass_pct", "ADMET pass %")]:
        s1 = (df[f"B_pivot__{col}"] - df[f"A_base__{col}"]).mean()
        s2 = (df[f"D_full__{col}"] - df[f"B_pivot__{col}"]).mean()
        tot = (df[f"D_full__{col}"] - df[f"A_base__{col}"]).mean()
        print(f"\nADDITIVITY, {lab}: pivot {s1:+.4f} + uncap {s2:+.4f} = {s1+s2:+.4f} "
              f"vs shipped {tot:+.4f}  (residual {tot-(s1+s2):+.4f})")

print("\n" + "=" * 100)
print("WHAT THE PIVOT WAS FOR — a shortlist that means something")
print("=" * 100)
hdr = f"  {'':24s}" + "".join(f"{k:>16}" for k, _, _ in ARMS)
print(hdr)
for lab, col in [("front, 5 objectives %", "front5_pct"),
                 ("front, 2 objectives %", "front2_pct"),
                 ("ADMET pass %", "admet_pass_pct")]:
    line = f"  {lab:24s}"
    for k, _, _ in ARMS:
        c = f"{k}__{col}"
        line += f"{df[c].mean():15.1f}%" if c in df else f"{'--':>16}"
    print(line)
print("\n  ADMET pass % is the deliverable: the fraction of evaluated molecules a")
print("  chemist could actually take forward. The pivot arms enforce it by")
print("  construction; the baseline pays no attention to it.")
df.to_csv(f"{B}/pivot_arm/scored.csv", index=False)
print(f"\nWrote pivot_arm/scored.csv   (n=6 -> minimum two-sided Wilcoxon p is 0.0312)")
