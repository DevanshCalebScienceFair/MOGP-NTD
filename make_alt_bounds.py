"""Build the alternative normalization frame (hDHFR ceiling at 0.0 kcal/mol).

Writes `evaluation_bounds_hdhfr0.json`. The published frame
(`evaluation_bounds.json`) is never touched.

WHY: the shared -5.0 ceiling truncates exactly the direction hDHFR is optimized
toward. Measured over 750 distinct fully-docked molecules from the six full-arm
campaigns, 19 of the 50 MOST SELECTIVE molecules (38%) clip above it, collapsing
13.14 kcal/mol of real difference onto the single normalized value 1.0. The
optimizer gets no gradient on the axis that carries the whole clinical argument.

WHY 0.0 AND NOT WIDER: a positive Vina score means a clashing or failed pose,
not measured non-binding. All 5 positive-hDHFR molecules in that set sit in the
top 50 by selectivity -- exactly the artifacts a wider ceiling would reward.

Runs using this frame are comparable ONLY to each other.

    python make_alt_bounds.py
    ./run_hdhfr_bound_arm.sh
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import evaluation as E

published = E.compute_objective_bounds()
alt = E.compute_objective_bounds(bounds_path=E.ALT_BOUNDS_PATH_HDHFR,
                                 docking_bounds=E.HDHFR_ALT_BOUNDS, force=True)
print(f"published frame  {E.BOUNDS_PATH}")
print(f"  PfDHFR {published[0].tolist()}   hDHFR {published[1].tolist()}")
print(f"  fingerprint {E.bounds_fingerprint(published)}")
print(f"\nalternative frame  {E.ALT_BOUNDS_PATH_HDHFR}")
print(f"  PfDHFR {alt[0].tolist()}   hDHFR {alt[1].tolist()}")
print(f"  fingerprint {E.bounds_fingerprint(alt)}")
assert E.bounds_fingerprint(published) != E.bounds_fingerprint(alt)
assert (published[2:] == alt[2:]).all(), "ADMET bounds must be untouched"
print("\nADMET bounds identical; only the hDHFR ceiling moved.")
print("Hypervolumes under the two frames are NOT comparable.")
