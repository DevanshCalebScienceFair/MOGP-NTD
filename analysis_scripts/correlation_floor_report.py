"""Reporting half of F19 (`correlation_floor.py`), kept torch-free.

Why this is a separate file. The Modal local entrypoint runs in whatever python
invoked `modal run`, which is the CLI's interpreter and has no torch. Importing
`correlation_floor` there to reach `report()` fails at its module-scope
`import torch`, which is exactly what happened on the first smoke run: the CSV
was written and the summary table was silently skipped. Nothing here imports
torch, gpytorch or rdkit, so both the container and the laptop can print the
same table.
"""
import numpy as np
from scipy.stats import spearmanr, wilcoxon


def report(df):
    print("\n" + "=" * 104)
    print("F19 CORRELATION FLOOR. Held-out task-2 prediction, RMSE in kcal/mol.")
    print("advantage > 0 means the ICM predicts task 2 better than an independent GP.")
    print("=" * 104)
    hdr = (f"{'cell':>12} {'k':>4} {'rho_real':>9} {'ICM':>8} {'indep':>8} "
           f"{'advantage':>10} {'sd':>7} {'wins':>8} {'p':>8} "
           f"{'mean rho_B':>11} {'sd(rho_B)':>10}")
    print(hdr)
    print("-" * len(hdr))
    out = []
    for (label, k), s in df.groupby(["label", "k"], sort=False):
        a = s[s.model == "ICM"].sort_values("rep")
        b = s[s.model == "independent"].sort_values("rep")
        dd = b.rmse.values - a.rmse.values
        dd = dd[np.isfinite(dd)]
        if len(dd) == 0:
            continue
        p = (wilcoxon(dd).pvalue
             if len(dd) > 1 and len(set(np.round(dd, 12))) > 1 else float("nan"))
        print(f"{label:>12} {k:>4} {a.rho_realized.iloc[0]:>+9.3f} "
              f"{a.rmse.mean():>8.3f} {b.rmse.mean():>8.3f} {dd.mean():>+10.4f} "
              f"{dd.std():>7.3f} {int((dd > 0).sum()):>5}/{len(dd)} {p:>8.4f} "
              f"{a.rho_B.mean():>+11.3f} {a.rho_B.std():>10.3f}")
        out.append(dict(label=label, k=int(k), rho=float(a.rho_realized.iloc[0]),
                        advantage=float(dd.mean()), wins=int((dd > 0).sum()),
                        n=len(dd), p=float(p),
                        is_real=bool(a.is_real.iloc[0])))
    print("\nAgainst the pre-registered prediction:")
    syn = [r for r in out if not r["is_real"]]
    real = [r for r in out if r["is_real"]]
    for k in sorted({r["k"] for r in syn}):
        cells = sorted([r for r in syn if r["k"] == k and r["rho"] > 0],
                       key=lambda r: r["rho"])
        if len(cells) < 2:
            continue
        pos = [c for c in cells if c["advantage"] > 0]
        floor = (f"between rho={cells[cells.index(pos[0]) - 1]['rho']:.3f} "
                 f"and rho={pos[0]['rho']:.3f}" if pos and cells.index(pos[0]) > 0
                 else ("below the smallest rho tested" if pos
                       else "ABOVE every rho tested"))
        rs = [c["rho"] for c in cells]
        advs = [c["advantage"] for c in cells]
        sp = spearmanr(rs, advs).statistic if len(rs) > 2 else float("nan")
        print(f"  k={k}: Spearman(rho, advantage) = {sp:+.3f}   "
              f"zero crossing {floor}")
    for k in sorted({r["k"] for r in syn}):
        plus = next((r for r in syn if r["k"] == k
                     and abs(r["rho"] - 0.711) < 0.02 and r["rho"] > 0), None)
        minus = next((r for r in syn if r["k"] == k and r["rho"] < -0.5), None)
        if plus and minus:
            print(f"  k={k}: SIGN CONTROL  +{plus['rho']:.3f} -> "
                  f"{plus['advantage']:+.4f}   {minus['rho']:.3f} -> "
                  f"{minus['advantage']:+.4f}")
    for r in real:
        print(f"  k={r['k']}: REAL ANCHOR advantage {r['advantage']:+.4f} "
              f"({r['wins']}/{r['n']}, p={r['p']:.4f})   "
              f"F17 measured +0.122 at k=8, 19/20, p=0.0002")
    return out
