"""Gate report for the rung-2 / rung-3 correlation pilots.

Two questions per campaign, both answered from the same docked sample:

  rho  -- task correlation, against F19's floor. At ~5% co-location the floor
          interpolates to about 0.40 (0.450 at 3.6%, 0.259 at 9.8%).

  bounds -- the PROVISIONAL [-11, -5] window was copied from DHFR. A score
          outside it clips to 0 or 1 on that axis and hypervolume stops
          discriminating, silently.

THE GATE IS STRICTER THAN THE FLOOR, and rung 1 is why. Its pilot read
rho = +0.719 on random library molecules; the campaign's own BO-selected
molecules read +0.532. A pilot samples the library, a campaign samples what
the optimizer chose, and the optimizer hunts molecules on which the two
targets DISAGREE. Shrinkage observed there: -0.187. n=1, so it is an
observation and not a calibrated constant, but the direction is mechanistic
and should be expected to repeat.
"""
import numpy as np, pandas as pd, pathlib, sys

SP = pathlib.Path("/private/tmp/claude-502/-Users-devansh/"
                  "72ba70a1-b23d-4a04-b4e5-5c299f117f4d/scratchpad/pilots")
BLO, BHI = -11.0, -5.0           # the provisional normalization window
FLOOR_5PCT = 0.40                # F19, interpolated to ~5% co-location
SHRINK = 0.187                   # rung 1: pilot 0.719 -> campaign 0.532


def boot_ci(x, y, n=20000, seed=7):
    r = np.random.default_rng(seed)
    idx = r.integers(0, len(x), size=(n, len(x)))
    rs = np.array([np.corrcoef(x[i], y[i])[0, 1] for i in idx])
    return np.percentile(rs, 2.5), np.percentile(rs, 97.5)


print(f"{'campaign':<22} {'n':>4} {'rho':>7} {'95% CI':>18} {'Spearman':>9}  verdict")
print("-" * 92)
notes = []
for d in sorted(p for p in SP.iterdir() if p.is_dir()):
    df = pd.concat([pd.read_csv(c) for c in sorted(d.glob("chunk*.csv"))])
    tcols = [c for c in df.columns if c not in ("SMILES", "library_index")]
    a, b = tcols
    raw = len(df)
    df = df.dropna(subset=tcols)
    prepped = len(df)
    # A POSITIVE Vina affinity is a FAILED FIT, not a weak binder. One such row
    # flipped the rung-1 pilot's verdict (0.544 -> 0.654).
    bad = int(((df[a] > 0) | (df[b] > 0)).sum())
    df = df[(df[a] <= 0) & (df[b] <= 0)]
    x, y = df[a].to_numpy(float), df[b].to_numpy(float)
    rho = np.corrcoef(x, y)[0, 1]
    lo, hi = boot_ci(x, y)
    sp = pd.Series(x).corr(pd.Series(y), method="spearman")
    adj = rho - SHRINK
    if lo - SHRINK >= FLOOR_5PCT:
        v = "GO"
    elif adj >= FLOOR_5PCT:
        v = "GO, thin"
    elif rho >= FLOOR_5PCT:
        v = "HOLD, clears only unadjusted"
    else:
        v = "NO"
    print(f"{d.name:<22} {len(df):>4} {rho:>+7.3f}  [{lo:>+6.3f}, {hi:>+6.3f}]  "
          f"{sp:>+9.3f}  {v}")
    notes.append((d.name, a, b, df, raw, prepped, bad, rho, lo, hi, adj))

print()
print(f"floor at ~5% co-location = {FLOOR_5PCT:.2f};  shrinkage applied = {SHRINK:.3f}")
print(f"(rung 1 for reference: pilot +0.719, campaign +0.532, floor cleared by +0.13)")
print()
print("=== BOUNDS CHECK: the provisional window is [%.1f, %.1f] kcal/mol ===" % (BLO, BHI))
print(f"{'campaign':<22} {'target':<14} {'min':>8} {'max':>8} {'<lo':>5} {'>hi':>5}  status")
print("-" * 80)
for name, a, b, df, raw, prepped, bad, *_ in notes:
    for t in (a, b):
        v = df[t].to_numpy(float)
        under, over = int((v < BLO).sum()), int((v > BHI).sum())
        pct = 100 * (under + over) / len(v)
        st = "OK" if under + over == 0 else (
             f"SATURATES: {pct:.1f}% clip" if pct > 2 else f"marginal: {pct:.1f}% clip")
        print(f"{name:<22} {t:<14} {v.min():>8.2f} {v.max():>8.2f} "
              f"{under:>5} {over:>5}  {st}")

print()
print("=== prep yield ===")
for name, a, b, df, raw, prepped, bad, *_ in notes:
    print(f"  {name:<22} {raw} sampled -> {prepped} docked both "
          f"({100*(raw-prepped)/raw:.1f}% ligand-prep failure) -> {len(df)} after "
          f"dropping {bad} non-physical row(s)")
