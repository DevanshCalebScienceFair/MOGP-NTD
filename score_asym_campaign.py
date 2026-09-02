"""Score the matched-budget asymmetric campaign fairly.

Two endpoints, reported together because they answer different questions and
one of them is structurally biased.

1. HYPERVOLUME over fully-docked molecules. Weak, and it FAVOURS the full arm by
   construction: a molecule missing hDHFR cannot sit on a Pareto front, so the
   asymmetric arm's front is built from a quarter of its molecules. Reported
   because a tie here would already be informative, not because it is fair.

2. SHORTLIST QUALITY among each arm's OWN measured molecules. **ALSO BIASED
   TOWARD THE FULL ARM, and not the headline test.** It ranks each arm's
   fully-measured molecules, and those pools are not the same size: at seed 0 the
   full arm chose from 246 and the asymmetric arm from 99. I originally described
   this as "the decision-relevant comparison"; that was wrong, and it is not what
   CLOSED_LOOP_DESIGN.md section A specified.

   The unbiased test lives in **nominate_and_score.py**: each arm ranks the SAME
   ~26,300 unmeasured library molecules with its own retrained model, nominates
   the top-K by PREDICTED selectivity, and those K are docked for real at
   identical added cost. Use that for any claim about which design finds better
   molecules. Both endpoints in THIS file favour the full arm by construction.

Usage:  python score_asym_campaign.py [asym_campaign]
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = sys.argv[1] if len(sys.argv) > 1 else "asym_campaign"
PF_MAX, HD_MAX = -7.0, 0.0          # docking-artifact filter, as used throughout
TOP_K = 20


def load(arm, seed):
    d = os.path.join(ROOT, f"{arm}_seed{seed}")
    ev = os.path.join(d, "evaluated.csv")
    hi = os.path.join(d, "history.csv")
    if not (os.path.exists(ev) and os.path.exists(hi)):
        return None
    return pd.read_csv(ev), pd.read_csv(hi), json.load(open(os.path.join(d, "run_config.json")))


def summarize(ev, hist, cfg):
    both = np.isfinite(ev.PfDHFR_Docking) & np.isfinite(ev.hDHFR_Docking)
    phys = ev[both & (ev.PfDHFR_Docking <= PF_MAX) & (ev.hDHFR_Docking <= HD_MAX)]
    top = phys.nlargest(TOP_K, "Selectivity_Index")
    n_calls = int(np.isfinite(ev.PfDHFR_Docking).sum() + np.isfinite(ev.hDHFR_Docking).sum())
    return dict(
        molecules=len(ev),
        dock_calls=n_calls,
        both_labels=int(both.sum()),
        physical=len(phys),
        final_hv=float(hist.hypervolume.iloc[-1]),
        top_k_mean_SI=float(top.Selectivity_Index.mean()) if len(top) else np.nan,
        best_SI=float(phys.Selectivity_Index.max()) if len(phys) else np.nan,
        best_PfDHFR=float(phys.PfDHFR_Docking.min()) if len(phys) else np.nan,
    )


rows = []
for seed in range(10):
    a, b = load("full", seed), load("asym", seed)
    if a is None or b is None:
        continue
    rows.append(dict(seed=seed,
                     **{f"full_{k}": v for k, v in summarize(*a).items()},
                     **{f"asym_{k}": v for k, v in summarize(*b).items()}))

if not rows:
    print(f"No completed seed pairs under {ROOT}/ yet."); sys.exit(0)
df = pd.DataFrame(rows)
print(f"Completed seed pairs: {list(df.seed)}  (n={len(df)})\n")

print("=" * 96)
print("BUDGET CHECK -- the arms must have spent the same number of dock calls")
print("=" * 96)
print(f"  full: {df.full_dock_calls.mean():.0f} calls for {df.full_molecules.mean():.0f} molecules")
print(f"  asym: {df.asym_dock_calls.mean():.0f} calls for {df.asym_molecules.mean():.0f} molecules "
      f"({df.asym_molecules.mean() / df.full_molecules.mean():.2f}x more molecules)")
ratio = df.asym_dock_calls.mean() / df.full_dock_calls.mean()
print(f"  budget ratio asym/full = {ratio:.3f}" +
      ("  OK" if 0.95 < ratio < 1.05 else "  *** BUDGETS DO NOT MATCH -- comparison invalid ***"))

def paired(name, col, better_high=True):
    a, b = df[f"asym_{col}"].values, df[f"full_{col}"].values
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 2:
        print(f"  {name:26s} n={len(a)} -- too few pairs"); return
    d = (a - b) if better_high else (b - a)
    p = wilcoxon(d).pvalue if len(set(d)) > 1 else float("nan")
    rng = np.random.default_rng(0)
    boot = [rng.choice(d, len(d), replace=True).mean() for _ in range(10000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"  {name:26s} asym {a.mean():8.4f} | full {b.mean():8.4f} | "
          f"delta {d.mean():+8.4f} [{lo:+.4f}, {hi:+.4f}] | asym wins {int((d>0).sum())}/{len(d)} | p={p:.4f}")

print("\n" + "=" * 96)
print("ENDPOINT 1 -- hypervolume (BIASED TOWARD full; a tie here already favours asym)")
print("=" * 96)
paired("final hypervolume", "final_hv")

print("\n" + "=" * 96)
print(f"ENDPOINT 2 -- shortlist quality (the decision-relevant one), top-{TOP_K}, artifact-filtered")
print("=" * 96)
paired(f"top-{TOP_K} mean selectivity", "top_k_mean_SI")
paired("best selectivity found", "best_SI")
paired("best PfDHFR (kcal/mol)", "best_PfDHFR", better_high=False)
paired("physical molecules found", "physical")

print("\n" + "=" * 96)
print("PER-SEED")
print("=" * 96)
show = df[["seed", "full_molecules", "asym_molecules", "full_final_hv", "asym_final_hv",
           "full_top_k_mean_SI", "asym_top_k_mean_SI"]]
show.columns = ["seed", "full n", "asym n", "full HV", "asym HV", "full top20 SI", "asym top20 SI"]
print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
df.to_csv(os.path.join(ROOT, "scored.csv"), index=False)
print(f"\nWrote {ROOT}/scored.csv")
