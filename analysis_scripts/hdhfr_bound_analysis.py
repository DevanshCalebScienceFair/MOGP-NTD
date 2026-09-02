"""Does un-truncating the hDHFR axis help the SEARCH, not just the metric?

THE TRAP THIS SCRIPT EXISTS TO AVOID: the two arms are scored in DIFFERENT
normalization frames, so their hypervolumes are not comparable. Changing a bound
moves every number for reasons unrelated to the method. This script therefore
refuses to compare hypervolume across frames, and judges the arms on
frame-INDEPENDENT quantities computed from raw kcal/mol:

  * artifact-filtered selectivity of the molecules each arm actually found
  * how many physical binders it turned up
  * how much of the selective tail it reached past the old -5.0 ceiling
"""
import glob, json, os, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon

B = "/Users/devansh/mogp-main-vscode/MOGP-NTD"
sys.path.insert(0, B)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import evaluation as E

PF_MAX, HD_MAX = -7.0, 0.0
OLD_CEIL = E.DOCKING_KCAL_MAX          # -5.0


def summarize(path):
    ev = pd.read_csv(path).dropna(subset=["PfDHFR_Docking", "hDHFR_Docking"])
    phys = ev[(ev.PfDHFR_Docking <= PF_MAX) & (ev.hDHFR_Docking <= HD_MAX)]
    top = phys.nlargest(20, "Selectivity_Index")
    return dict(
        evaluated=len(ev), physical=len(phys),
        top20_SI=float(top.Selectivity_Index.mean()) if len(top) else np.nan,
        best_SI=float(phys.Selectivity_Index.max()) if len(phys) else np.nan,
        best_pf=float(phys.PfDHFR_Docking.min()) if len(phys) else np.nan,
        # the quantity the old ceiling was blind to: molecules whose hDHFR sits
        # in the band the published frame collapsed onto a single value.
        in_censored_band=int(((ev.hDHFR_Docking > OLD_CEIL)
                              & (ev.hDHFR_Docking <= HD_MAX)).sum()),
        artifacts=int((ev.hDHFR_Docking > HD_MAX).sum()),
    )


rows = []
for seed in range(10):
    a = f"{B}/asym_campaign/full_seed{seed}/evaluated.csv"       # published frame
    b = f"{B}/hdhfr_bound_arm/hdhfr0_seed{seed}/evaluated.csv"   # alternative frame
    if not (os.path.exists(a) and os.path.exists(b)):
        continue
    rows.append(dict(seed=seed,
                     **{f"old_{k}": v for k, v in summarize(a).items()},
                     **{f"new_{k}": v for k, v in summarize(b).items()}))

if not rows:
    print("No paired seeds yet. Run:  python make_alt_bounds.py && ./run_hdhfr_bound_arm.sh")
    sys.exit(0)
df = pd.DataFrame(rows)

print("=" * 92)
print("FRAME CHECK — hypervolume is NOT compared here, and this is why")
print("=" * 92)
pub = E.compute_objective_bounds()
print(f"  published    hDHFR bounds {pub[1].tolist()}   fingerprint {E.bounds_fingerprint(pub)}")
if os.path.exists(E.ALT_BOUNDS_PATH_HDHFR):
    alt = E.compute_objective_bounds(bounds_path=E.ALT_BOUNDS_PATH_HDHFR)
    print(f"  alternative  hDHFR bounds {alt[1].tolist()}   fingerprint {E.bounds_fingerprint(alt)}")
    assert E.bounds_fingerprint(pub) != E.bounds_fingerprint(alt)
print("  -> different frames. Every hypervolume differs for reasons unrelated to")
print("     the method, so the endpoints below are computed from RAW kcal/mol only.")

print("\n" + "=" * 92)
print(f"PAIRED, alternative frame vs published  (n={len(df)}), frame-independent endpoints")
print("=" * 92)
rng = np.random.default_rng(0)
for label, col, hi in [("top-20 selectivity", "top20_SI", True),
                       ("best selectivity found", "best_SI", True),
                       ("physical molecules", "physical", True),
                       ("best PfDHFR (kcal/mol)", "best_pf", False),
                       ("molecules in the censored band", "in_censored_band", True),
                       ("docking artifacts (hDHFR > 0)", "artifacts", False)]:
    a = df[f"new_{col}"].values.astype(float)
    b = df[f"old_{col}"].values.astype(float)
    ok = np.isfinite(a) & np.isfinite(b); a, b = a[ok], b[ok]
    if len(a) < 2:
        print(f"  {label:32s} n={len(a)} — too few pairs"); continue
    d = (a - b) if hi else (b - a)
    p = wilcoxon(d).pvalue if len(set(d)) > 1 else float("nan")
    boot = [rng.choice(d, len(d), replace=True).mean() for _ in range(10000)]
    lo, up = np.percentile(boot, [2.5, 97.5])
    print(f"  {label:32s} new {a.mean():8.3f} | old {b.mean():8.3f} | "
          f"delta {d.mean():+8.3f} [{lo:+.3f},{up:+.3f}] | new wins {int((d>0).sum())}/{len(d)} | p={p:.4f}")

print("\n  'censored band' = molecules whose hDHFR falls between the old ceiling")
print(f"  ({OLD_CEIL}) and 0.0 — the range the published frame collapsed to a single")
print("  value. If the new frame is doing what it was designed to do, it should")
print("  reach MORE of them without buying more artifacts (hDHFR > 0).")
df.to_csv(f"{B}/hdhfr_bound_arm/scored.csv", index=False)
print(f"\nWrote hdhfr_bound_arm/scored.csv")
