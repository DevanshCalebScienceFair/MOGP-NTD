"""
validate_known_actives.py
=========================

Ground-truth validation of the multi-objective BO loop against KNOWN selective
antifolate antimalarials — the compounds medicinal chemists already accept as
good answers on the potency/selectivity plane. A method that works should
rediscover molecules like these; the ICM (coregionalized) model should do so
with fewer expensive docking calls than the batch-independent baseline, because
it borrows strength across the correlated PfDHFR/hDHFR docking tasks.

The known actives are classic *P. falciparum* DHFR inhibitors. Canonical SMILES
are hardcoded below with their PubChem/ChEMBL source IDs:

    Pyrimethamine — first-line antifolate; wild-type PfDHFR inhibitor.
    Cycloguanil   — active metabolite of proguanil (dihydrotriazine).
    WR99210       — potent inhibitor of pyrimethamine-RESISTANT PfDHFR.
    P218          — rationally designed, PfDHFR-selective clinical candidate.

Three things are reported:

  1. Reference profile (potency/selectivity ground truth): for each known
     compound, its hERG probability from our ADMET oracle and its docking score
     against BOTH PfDHFR and human DHFR, plus the Selectivity Index
     (hDHFR - PfDHFR docking; positive = binds the parasite enzyme more tightly
     than the human one -> selective).

  2. Recovery: for each method's FINAL Pareto front, the maximum Tanimoto
     similarity of any Pareto molecule to each known active, and how many known
     actives are "recovered" (max similarity >= threshold, default 0.4).

  3. Headline: whether the loop rediscovers the known selective antifolates at
     all, and whether the coregionalized model recovers them with fewer docking
     calls (molecules evaluated, in evaluation order) than the independent
     baseline.

Run ``python validate_known_actives.py --help`` for options. Parts 2-3 read a
method's ``pareto_front.csv`` / ``evaluated.csv`` from its results directory, so
they are meaningful only AFTER an ablation run has produced those files; missing
directories are reported and skipped.
"""

import os
import argparse

import numpy as np
import pandas as pd

import evaluation
from utils.featurize import smiles_to_morgan


# ---------------------------------------------------------------------- #
# Known selective antifolate antimalarials (ground truth)
# ---------------------------------------------------------------------- #
# Canonical SMILES looked up from PubChem / ChEMBL. Salts are stripped to the
# free base/acid actually docked. Each entry cites its source ID.
KNOWN_ACTIVES = [
    {
        "name": "Pyrimethamine",
        "source": "PubChem CID 4993",
        # 2,4-diamino-5-(4-chlorophenyl)-6-ethylpyrimidine (C12H13ClN4).
        "smiles": "CCC1=C(C(=NC(=N1)N)N)C2=CC=C(C=C2)Cl",
    },
    {
        "name": "Cycloguanil",
        "source": "PubChem CID 9049",
        # Dihydrotriazine active metabolite of proguanil (C11H14ClN5).
        "smiles": "CC1(N=C(N=C(N1C2=CC=C(C=C2)Cl)N)N)C",
    },
    {
        "name": "WR99210",
        # PubChem lists the hydrochloride; we dock the free base (C14H18Cl3N5O2).
        "source": "PubChem CID 121749 (free base of the listed HCl salt)",
        "smiles": "CC1(N=C(N=C(N1OCCCOC2=CC(=C(C=C2Cl)Cl)Cl)N)N)C",
    },
    {
        "name": "P218",
        "source": "ChEMBL CHEMBL3040038",
        # Rationally designed flexible diaminopyrimidine antifolate (C18H24N4O4).
        "smiles": "CCc1nc(N)nc(N)c1OCCCOc1ccccc1CCC(=O)O",
    },
]


# Method -> results directory. The ablation writes the same three CSVs
# (history/evaluated/pareto_front) to each method's own directory. The headline
# compares the two ablation arms; baselines are shown in the recovery table when
# present. Directories that do not exist are skipped.
INDEPENDENT_KEY = "Independent MOGP"
COREGIONALIZED_KEY = "Coregionalized MOGP"
DEFAULT_METHOD_DIRS = [
    (INDEPENDENT_KEY, "results"),
    (COREGIONALIZED_KEY, "results_coregionalized"),
    ("Random Search", "baseline_random_results"),
    ("Single-Obj BO", "baseline_single_obj_results"),
    ("Greedy Filter", "baseline_greedy_results"),
]

DEFAULT_SIM_THRESHOLD = 0.4


# ---------------------------------------------------------------------- #
# Similarity helpers
# ---------------------------------------------------------------------- #
def tanimoto(fp_a, fp_b):
    """Tanimoto similarity between two 0/1 fingerprint arrays."""
    a = np.asarray(fp_a, dtype=bool)
    b = np.asarray(fp_b, dtype=bool)
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(a, b).sum()) / float(union)


def _fingerprint(smiles):
    """Morgan fingerprint for a SMILES, or None if RDKit cannot parse it."""
    try:
        return smiles_to_morgan(smiles)
    except ValueError:
        return None


def max_similarity_to_set(active_fp, candidate_fps):
    """Max Tanimoto of ``active_fp`` against a list of candidate fingerprints."""
    if not candidate_fps:
        return 0.0
    return max(tanimoto(active_fp, c) for c in candidate_fps)


def first_recovery_index(active_fp, ordered_fps, threshold):
    """1-based position of the first candidate with similarity >= threshold.

    ``ordered_fps`` must be in evaluation order (as evaluated.csv is written), so
    the returned index is the number of docking calls made before this known
    active was first matched. Returns None if it is never matched.
    """
    for i, fp in enumerate(ordered_fps, start=1):
        if fp is not None and tanimoto(active_fp, fp) >= threshold:
            return i
    return None


# ---------------------------------------------------------------------- #
# Part 1 — reference potency/selectivity profile
# ---------------------------------------------------------------------- #
def gated_selectivity_index(pf_kcal, hd_kcal):
    """Selectivity Index for raw PfDHFR / hDHFR affinities, VIABILITY-GATED.

    Both the arithmetic and the gate come from
    ``evaluation.add_selectivity_index``: the raw affinities are handed over
    under the ``*_Docking_raw_kcal`` names it reads, and the predicate is
    ``evaluation.dock_is_viable``. The viability threshold is NEVER written down
    in this file.

    A non-viable dock (``>= 0 kcal/mol``, or an outright failure) means no
    binding mode was found, so the DIFFERENCE of the two binding free energies
    is undefined and the index is ``NaN`` — the same choice
    ``evaluation.add_selectivity_index`` already makes for every optimizer arm.
    The raw affinities are still printed, beside their viability flags; what
    changes is that a clash cannot produce a selectivity number.

    Args:
        pf_kcal: raw PfDHFR Vina affinity/affinities in kcal/mol.
        hd_kcal: raw hDHFR Vina affinity/affinities in kcal/mol, same shape.

    Returns:
        ``(si, pf_viable, hd_viable)`` as 1-D arrays.
    """
    pf = np.asarray(pf_kcal, dtype=float).ravel()
    hd = np.asarray(hd_kcal, dtype=float).ravel()
    probe = pd.DataFrame({
        "PfDHFR_Docking" + evaluation.RAW_KCAL_SUFFIX: pf,
        "hDHFR_Docking" + evaluation.RAW_KCAL_SUFFIX: hd,
    })
    evaluation.add_selectivity_index(probe)
    si = probe[evaluation.SELECTIVITY_COLUMN].to_numpy(dtype=float)
    return si, evaluation.dock_is_viable(pf), evaluation.dock_is_viable(hd)



def reference_profiles(actives, skip_docking=False):
    """Print hERG + PfDHFR/hDHFR docking + Selectivity Index for each active.

    Returns the actives list annotated with a precomputed ``fp`` (fingerprint)
    for reuse in the recovery analysis.
    """
    from admet_oracle import ADMETOracle

    print("=" * 78)
    print("1. KNOWN-ACTIVE REFERENCE PROFILE (potency / selectivity ground truth)")
    print("=" * 78)

    oracle = ADMETOracle()

    if not skip_docking:
        from docking import dock_target
    else:
        print("  (--skip-docking: docking omitted; hERG still reported)")

    # Source IDs printed as a legend so the metrics table below stays aligned.
    print("  Known actives (canonical SMILES hardcoded from):")
    for a in actives:
        print(f"    {a['name']:<15} {a['source']}")
    print()

    # Two passes, so the Selectivity Index can be derived by ONE call into
    # evaluation rather than restated per row. Pass 1 docks and predicts; pass 2
    # gates and prints. "ok" carries the two viability flags beside the raw
    # affinities they describe, so the raw measurement is never hidden.
    for a in actives:
        a["fp"] = _fingerprint(a["smiles"])

        herg = float("nan")
        try:
            pred = oracle.predict([a["smiles"]])
            herg = float(pred["hERG_Toxicity_Prob"].iloc[0])
        except Exception as exc:                                  # noqa: BLE001
            print(f"  (hERG prediction failed for {a['name']}: {exc})")

        pf = hd = float("nan")
        if not skip_docking:
            try:
                pf = float(dock_target(a["smiles"], target="PfDHFR"))
                hd = float(dock_target(a["smiles"], target="hDHFR"))
            except Exception as exc:                              # noqa: BLE001
                print(f"  (docking failed for {a['name']}: {exc})")

        a["herg"], a["pfdhfr"], a["hdhfr"] = herg, pf, hd

    # The whole Selectivity Index column, gated, in one delegated call.
    # Computing it inline as ``hd - pf`` is the defect this gate removes.
    si_all, pf_viable, hd_viable = gated_selectivity_index(
        [a["pfdhfr"] for a in actives], [a["hdhfr"] for a in actives])

    header = (f"{'compound':<15}{'hERG':>8}{'PfDHFR':>10}{'hDHFR':>10}"
              f"{'ok':>5}{'SelIdx':>10}")
    print(header)
    print("-" * len(header))

    for a, si, pv, hv in zip(actives, si_all, pf_viable, hd_viable):
        a["selectivity_index"] = float(si)
        a["pfdhfr_viable"] = bool(pv)
        a["hdhfr_viable"] = bool(hv)
        ok = ("Y" if pv else "n") + ("Y" if hv else "n")
        si_s = f"{si:>10.2f}" if np.isfinite(si) else f"{'n/a':>10}"
        print(f"{a['name']:<15}{a['herg']:>8.3f}{a['pfdhfr']:>10.2f}"
              f"{a['hdhfr']:>10.2f}{ok:>5}{si_s}")

    print("\n  Selectivity Index = hDHFR - PfDHFR docking (kcal/mol). Positive "
          "means\n  the compound binds the PARASITE enzyme more tightly than the "
          "human one\n  -> selective. These are the reference points the loop "
          "should rediscover.")
    # The threshold in this note is READ from evaluation, so the explanation
    # cannot drift from the rule it explains.
    n_bad = int((~(pf_viable & hd_viable)).sum())
    note = [
        "",
        f"  DOCKING VIABILITY (evaluation.dock_is_viable): "
        f"{len(actives) - n_bad}/{len(actives)} actives viable on both targets.",
    ]
    if skip_docking:
        note.append("  (--skip-docking: nothing was docked, so nothing is "
                    "viable.)")
    note += [
        "  'ok' is (PfDHFR, hDHFR). 'n' means the dock found NO binding mode",
        f"  (affinity >= {evaluation.DOCK_VIABILITY_KCAL_MAX:g} kcal/mol, or an "
        f"outright failure), so the DIFFERENCE",
        "  of the two binding free energies is undefined and SelIdx reads n/a",
        "  rather than a number. The raw affinity is still shown. A non-viable",
        "  known active would be evidence the DOCKING is broken, not that the",
        "  compound is selective.",
    ]
    print("\n".join(note))
    return actives


# ---------------------------------------------------------------------- #
# Parts 2-3 — recovery from each method's Pareto front / evaluated set
# ---------------------------------------------------------------------- #
def _load_smiles_column(path):
    """Return the SMILES column of a CSV in file order, or None if absent."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    if "SMILES" not in df.columns:
        return None
    return df["SMILES"].astype(str).tolist()


def recovery_table(actives, method_dirs, threshold):
    """Per method, report max similarity to each active and #recovered.

    Returns ``{method_name: {"n_recovered": int, "similarities": {active: sim}}}``
    for every method whose ``pareto_front.csv`` exists.
    """
    print("\n" + "=" * 78)
    print(f"2. RECOVERY FROM FINAL PARETO FRONTS (Tanimoto >= {threshold})")
    print("=" * 78)

    active_names = [a["name"] for a in actives]
    header = f"{'method':<22}" + "".join(f"{n[:11]:>12}" for n in active_names) \
        + f"{'#recov':>9}"
    print(header)
    print("-" * len(header))

    results = {}
    for name, directory in method_dirs:
        pareto_smiles = _load_smiles_column(
            os.path.join(directory, "pareto_front.csv"))
        if pareto_smiles is None:
            print(f"{name:<22}(no pareto_front.csv at {directory}/)")
            continue

        pareto_fps = [fp for fp in (_fingerprint(s) for s in pareto_smiles)
                      if fp is not None]

        sims, n_recovered = {}, 0
        cells = ""
        for a in actives:
            if a.get("fp") is None:
                sims[a["name"]] = float("nan")
                cells += f"{'n/a':>12}"
                continue
            sim = max_similarity_to_set(a["fp"], pareto_fps)
            sims[a["name"]] = sim
            recovered = sim >= threshold
            n_recovered += int(recovered)
            mark = "*" if recovered else " "
            cells += f"{sim:>11.3f}{mark}"

        results[name] = {"n_recovered": n_recovered, "similarities": sims,
                         "n_pareto": len(pareto_fps)}
        print(f"{name:<22}{cells}{n_recovered:>9}")

    print(f"\n  '*' marks a recovered active (>= {threshold}). "
          f"{len(active_names)} known actives total.")
    return results


def docking_calls_to_recover(actives, directory, threshold):
    """For one method, #docking calls until each active is first matched.

    Walks ``evaluated.csv`` in evaluation order. Returns
    ``{active_name: calls_or_None}``, or None if the file is absent.
    """
    evaluated = _load_smiles_column(os.path.join(directory, "evaluated.csv"))
    if evaluated is None:
        return None
    ordered_fps = [_fingerprint(s) for s in evaluated]

    calls = {}
    for a in actives:
        if a.get("fp") is None:
            calls[a["name"]] = None
            continue
        calls[a["name"]] = first_recovery_index(a["fp"], ordered_fps, threshold)
    return calls


def headline(actives, results, independent_dir, coregionalized_dir, threshold):
    """State the validation verdict: rediscovery + docking-call efficiency."""
    print("\n" + "=" * 78)
    print("3. HEADLINE VALIDATION RESULT")
    print("=" * 78)

    n_total = len(actives)

    # --- Rediscovery across whichever ablation arms produced a Pareto front ---
    arms = [k for k in (INDEPENDENT_KEY, COREGIONALIZED_KEY) if k in results]
    if not arms:
        print("  No ablation Pareto fronts found (looked for "
              f"{INDEPENDENT_KEY!r} at '{independent_dir}/' and "
              f"{COREGIONALIZED_KEY!r} at '{coregionalized_dir}/').")
        print("  Run the ablation first, then re-run this script.")
        return

    best_recovered = max(results[k]["n_recovered"] for k in arms)
    if best_recovered > 0:
        print(f"  REDISCOVERY: yes — the loop recovers {best_recovered}/{n_total} "
              "known selective antifolates on its Pareto front "
              f"(Tanimoto >= {threshold}).")
    else:
        print(f"  REDISCOVERY: no — no known active was recovered at threshold "
              f"{threshold}. Loosen --threshold or evaluate more molecules.")

    # --- Docking-call efficiency: coregionalized vs independent ---
    calls_ind = docking_calls_to_recover(actives, independent_dir, threshold)
    calls_cor = docking_calls_to_recover(actives, coregionalized_dir, threshold)

    if calls_ind is None or calls_cor is None:
        missing = independent_dir if calls_ind is None else coregionalized_dir
        print(f"\n  EFFICIENCY: cannot compare — missing evaluated.csv under "
              f"'{missing}/'. Provide both ablation arms to judge docking-call "
              "efficiency.")
        return

    print(f"\n  DOCKING CALLS TO FIRST RECOVER (evaluation order; lower is "
          "better):")
    print(f"    {'active':<15}{'Independent':>13}{'Coregionalized':>16}"
          f"{'winner':>14}")
    coreg_wins = ind_wins = ties = 0
    both_recovered = 0
    for a in actives:
        ci = calls_ind.get(a["name"])
        cc = calls_cor.get(a["name"])
        ci_s = str(ci) if ci is not None else "—"
        cc_s = str(cc) if cc is not None else "—"
        winner = "—"
        if ci is not None and cc is not None:
            both_recovered += 1
            if cc < ci:
                winner = "coregionalized"
                coreg_wins += 1
            elif ci < cc:
                winner = "independent"
                ind_wins += 1
            else:
                winner = "tie"
                ties += 1
        elif cc is not None and ci is None:
            winner = "coregionalized"      # only the ICM ever found it
            coreg_wins += 1
        elif ci is not None and cc is None:
            winner = "independent"
            ind_wins += 1
        print(f"    {a['name']:<15}{ci_s:>13}{cc_s:>16}{winner:>14}")

    print()
    if coreg_wins > ind_wins:
        print("  EFFICIENCY: yes — the coregionalized (ICM) model recovers the "
              "known\n  actives with FEWER docking calls than the independent "
              f"baseline\n  (coregionalized wins {coreg_wins}, independent "
              f"{ind_wins}, ties {ties}). The shared PfDHFR/hDHFR task covariance "
              "pays off.")
    elif ind_wins > coreg_wins:
        print("  EFFICIENCY: no — the independent baseline recovered them in "
              f"fewer\n  docking calls here (independent {ind_wins}, "
              f"coregionalized {coreg_wins}, ties {ties}). Inspect the run / seed "
              "/ iteration budget.")
    else:
        print(f"  EFFICIENCY: inconclusive — coregionalized {coreg_wins}, "
              f"independent {ind_wins}, ties {ties}.")


# ---------------------------------------------------------------------- #
# Entry point
# ---------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Validate the BO loop against known selective antifolate "
                    "antimalarials.")
    parser.add_argument("--threshold", type=float, default=DEFAULT_SIM_THRESHOLD,
                        help="Tanimoto recovery threshold (default 0.4).")
    parser.add_argument("--independent-dir", default="results",
                        help="Results dir for the independent-MOGP ablation arm.")
    parser.add_argument("--coregionalized-dir", default="results_coregionalized",
                        help="Results dir for the coregionalized (ICM) arm.")
    parser.add_argument("--skip-docking", action="store_true",
                        help="Skip Part-1 docking of the known actives (still "
                             "reports hERG). Parts 2-3 are unaffected.")
    args = parser.parse_args()

    # Method directories for the recovery table: the two ablation arms (from the
    # CLI) followed by any baseline dirs from the defaults.
    method_dirs = [(INDEPENDENT_KEY, args.independent_dir),
                   (COREGIONALIZED_KEY, args.coregionalized_dir)]
    method_dirs += [(name, d) for name, d in DEFAULT_METHOD_DIRS
                    if name not in (INDEPENDENT_KEY, COREGIONALIZED_KEY)]

    actives = reference_profiles(KNOWN_ACTIVES, skip_docking=args.skip_docking)
    results = recovery_table(actives, method_dirs, args.threshold)
    headline(actives, results, args.independent_dir, args.coregionalized_dir,
             args.threshold)


if __name__ == "__main__":
    main()
