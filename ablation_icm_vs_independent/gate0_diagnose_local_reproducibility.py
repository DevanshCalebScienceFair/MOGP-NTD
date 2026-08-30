"""Gate 0 failed. Is the cause the MACHINE, or is the oracle itself nondeterministic?

Gate 0 compared this machine's docking to scores computed on the Studio and found
disagreements up to 0.612 kcal/mol. Two very different causes produce that:

  (a) the receptors or their preparation differ between machines -- a real
      machine effect, and the ablation would be confounded; or
  (b) Vina is simply not reproducible at this tolerance, in which case the
      Studio would not reproduce its OWN scores either, and the 0.05 gate was
      never achievable by any machine.

The test that separates them: re-dock the SAME molecules AGAIN, here, and compare
this machine to ITSELF. Same receptor, same box, same seed, same binary, same
core count. Any spread that survives is (b).
"""
import json, os, random, sys
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import docking

OUT = os.path.dirname(os.path.abspath(__file__))
TARGETS = ["PfDHFR", "hDHFR"]

prev = pd.read_csv(os.path.join(OUT, "gate0_redock_comparison.csv"))
rows = []
for i, r in prev.iterrows():
    row = {"SMILES": r["SMILES"]}
    for t in TARGETS:
        again = docking.dock_target(r["SMILES"], t, use_cache=False)
        again = float("nan") if again is None else float(again)
        row[f"{t}_studio"] = float(r[f"{t}_stored"])
        row[f"{t}_local_run1"] = float(r[f"{t}_redocked"])
        row[f"{t}_local_run2"] = again
        row[f"{t}_local_vs_local"] = abs(again - float(r[f"{t}_redocked"]))
        row[f"{t}_local_vs_studio"] = float(r[f"{t}_abs_diff"])
    rows.append(row)
    print(f"  {i+1:2d}/10  Pf local1 {row['PfDHFR_local_run1']:7.3f} vs local2 "
          f"{row['PfDHFR_local_run2']:7.3f} (d={row['PfDHFR_local_vs_local']:.3f})   "
          f"h local1 {row['hDHFR_local_run1']:7.3f} vs local2 "
          f"{row['hDHFR_local_run2']:7.3f} (d={row['hDHFR_local_vs_local']:.3f})", flush=True)

df = pd.DataFrame(rows)
df.to_csv(os.path.join(OUT, "gate0_local_reproducibility.csv"), index=False)

ll = np.concatenate([df[f"{t}_local_vs_local"].to_numpy(float) for t in TARGETS])
ls = np.concatenate([df[f"{t}_local_vs_studio"].to_numpy(float) for t in TARGETS])
# Signed local-vs-studio: a systematic receptor difference shifts the mean off
# zero; pure nondeterminism scatters around it.
signed = np.concatenate([(df[f"{t}_local_run1"] - df[f"{t}_studio"]).to_numpy(float)
                         for t in TARGETS])

summary = {
    "local_vs_local": {"max": float(ll.max()), "mean": float(ll.mean()),
                       "median": float(np.median(ll)),
                       "n_over_0.05": int((ll > 0.05).sum()), "n": int(ll.size)},
    "local_vs_studio": {"max": float(ls.max()), "mean": float(ls.mean()),
                        "median": float(np.median(ls)),
                        "n_over_0.05": int((ls > 0.05).sum()), "n": int(ls.size)},
    "signed_local_minus_studio": {
        "mean": float(signed.mean()), "sd": float(signed.std(ddof=1))},
    "interpretation": None,
}
# If this machine cannot reproduce itself about as badly as it fails to
# reproduce the Studio, the failure is oracle nondeterminism, not a machine
# effect. Compared on the max, because the gate is a max-based rule.
if ll.max() >= 0.5 * ls.max():
    summary["interpretation"] = (
        "ORACLE NONDETERMINISM. This machine does not reproduce its own scores "
        "at 0.05 kcal/mol either, so the gate could not have been passed by any "
        "machine and the disagreement with the Studio is not evidence of a "
        "receptor or preparation difference.")
else:
    summary["interpretation"] = (
        "MACHINE EFFECT. Local docking is self-consistent but disagrees with the "
        "Studio, which points at the receptors or their preparation.")
with open(os.path.join(OUT, "gate0_diagnosis.json"), "w") as fh:
    json.dump(summary, fh, indent=2)

print(f"\nlocal vs local  : max {ll.max():.3f}  mean {ll.mean():.3f}  "
      f"{summary['local_vs_local']['n_over_0.05']}/{ll.size} exceed 0.05")
print(f"local vs studio : max {ls.max():.3f}  mean {ls.mean():.3f}  "
      f"{summary['local_vs_studio']['n_over_0.05']}/{ls.size} exceed 0.05")
print(f"signed local-studio: mean {signed.mean():+.4f} sd {signed.std(ddof=1):.4f}")
print(f"\n{summary['interpretation']}")
