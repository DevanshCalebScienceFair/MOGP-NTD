#!/usr/bin/env python
"""TASK 6 — size/lipophilicity confound, apo vs holo, on IDENTICAL molecules.

Controls for the population change: the same fixed sample is docked against
BOTH receptors, so any difference in correlation is attributable to the
receptor and not to PAINS/reactive screening having removed large greasy
classes. The sample is drawn from the RAW cached library (3610, unfiltered) so
the screened-out classes are still represented.

Deliberately does NOT reuse validate_docking.build_sample: that harvests scores
from existing run outputs, and every such output under matrix_results/ is
apo-derived, which would silently blend apo scores into the holo arm.

MEASUREMENT vs OBJECTIVE
------------------------
Every affinity here comes straight from ``docking.dock_target`` / ``dock_apo``,
i.e. from the oracle, so this script never has an OBJECTIVE column to confuse
with a measurement — there is no ``*_Docking`` column anywhere in it. What it
DID need is the other half of the rule: a dock whose best affinity is
``>= 0 kcal/mol`` found no binding mode, and such a row must not sit in a
distribution of binding affinities. The archived PfDHFR cache holds one at
**+4.213 kcal/mol** — roughly 9 kcal/mol outside the [-11, -5] band these
correlations are computed over, where a single row has real leverage on a
Pearson coefficient. ``evaluation.dock_is_viable`` is therefore applied to BOTH
arms, the excluded rows are reported, and the written ``task6_sample.csv``
records every dock that was attempted (with its viability flags) so the
exclusion is auditable rather than invisible.

The viability threshold is NOT written down in this file; it is
``evaluation.dock_is_viable``.

REQUIRED INPUT THAT IS CURRENTLY ABSENT
---------------------------------------
The apo arm docks against ``APO_RECEPTOR``
(``apo_backup_2026-08-10/1J3I_clean.pdbqt``), which is **not in the checkout**
and never was committed (``.gitignore`` ignores ``apo_backup_*/``). Without it
``dock_apo`` returned ``None`` for every molecule, every row was skipped, and
the empty DataFrame reached ``df["holo"] / df["heavy_atoms"]`` and died with a
bare ``KeyError: 'holo'`` — a message that names neither the missing receptor
nor the arm that needed it. ``require_apo_receptor`` now refuses to start and
says exactly which file is missing; ``require_paired_sample`` catches the same
situation arising for any other reason. Neither fabricates the missing data: the
apo half of this comparison cannot be recomputed until that receptor is restored
or re-derived.
"""
import os
import subprocess
import sys
import tempfile

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors

# Own-code statistics (mogp_core.stats) in place of scipy.stats. ``pearson_r`` is
# scipy's normalise-then-dot formulation and ``spearman_rho`` is Pearson on
# TIE-CORRECTED average ranks — NOT the 1-6*sum(d^2)/(n(n^2-1)) shortcut, which
# is only an algebraic simplification for tie-FREE samples. On integer
# heavy_atoms data at this script's own N_SAMPLE = 150 the shortcut is wrong by a
# measured 4.323e-03 (and by 4.571e-01 at n = 5). Agreement with scipy over 400
# x 150-molecule real subsamples x both score columns: max |diff| 5.551e-16.
# See mogp_core/stats.py and spec/stats/SPEC.md.
from mogp_core.stats import pearson_r, spearman_rho

import docking
import evaluation
from validate_known_actives import KNOWN_ACTIVES

RDLogger.DisableLog("rdApp.*")

N_SAMPLE = 150
SEED = 42
TARGET = "PfDHFR"
APO_RECEPTOR = "apo_backup_2026-08-10/1J3I_clean.pdbqt"
OUT = os.path.dirname(os.path.abspath(__file__))

# The two score columns every analysis below reads. Both hold RAW
# whole-molecule Vina affinities in kcal/mol exactly as the oracle returned
# them; there is no objective column in this script.
SCORE_COLUMNS = ("holo", "apo")


# ---------------------------------------------------------------------- #
# Preflight: refuse to run without the inputs, rather than crash later
# ---------------------------------------------------------------------- #
def require_apo_receptor(path=APO_RECEPTOR):
    """Return the apo receptor's absolute path, or FAIL LOUDLY naming the file.

    The apo arm is half of this script's whole point: it is what makes the
    comparison PAIRED. Without the receptor every ``dock_apo`` call returns
    ``None``, every molecule is skipped, and the failure used to surface as
    ``KeyError: 'holo'`` on an empty DataFrame — after a long docking loop, and
    without naming the cause. Checking up front costs nothing and the message
    says what to restore.

    Args:
        path: receptor path to require (default ``APO_RECEPTOR``).

    Returns:
        The absolute path, when it exists.

    Raises:
        SystemExit: with a message naming the missing input.
    """
    absolute = os.path.abspath(path)
    if os.path.exists(absolute):
        return absolute
    raise SystemExit(
        "validate_size_confound.py CANNOT RUN — a required input is missing.\n"
        f"  missing: {absolute}\n"
        f"  needed by: dock_apo() / the APO arm (APO_RECEPTOR = {path!r})\n"
        "\n"
        "  This script's result is a PAIRED apo-vs-holo comparison on identical\n"
        "  molecules. Without the apo receptor the apo arm skips every molecule,\n"
        "  so there is no comparison to report and nothing here can be salvaged\n"
        "  into a one-armed number: the holo-only correlations are already\n"
        "  reported by validate_docking.py.\n"
        "\n"
        "  The file was never committed (.gitignore ignores apo_backup_*/), so it\n"
        "  cannot be recovered from git. To run this again, restore the\n"
        "  pre-NADPH-fix apo receptor from a backup, or re-derive it by preparing\n"
        "  1J3I with the cofactor STRIPPED (docking.COFACTOR_RESNAMES emptied) and\n"
        "  write it to that path. Do NOT substitute the current holo receptor:\n"
        "  that would make both arms the same arm and report a difference of zero\n"
        "  as a scientific finding.\n"
        "\n"
        "  The apo numbers CLAUDE.md quotes (Spearman -0.726 -> -0.257,\n"
        "  LE +0.891 -> +0.880, logP -0.293 -> -0.245) came from this script when\n"
        "  that receptor existed. Until it is restored, their apo half is not\n"
        "  reproducible from this checkout."
    )


def require_paired_sample(df, n_attempted):
    """Fail loudly unless enough molecules docked against BOTH receptors.

    The backstop for every other way the paired sample can come up empty or
    near-empty (vina absent from PATH, a cold cache, ligand prep failing on the
    whole draw). Three is the minimum for a correlation, and ``corr_block``
    would otherwise return a silent ``nan`` or raise deep inside the statistics
    code.

    Args:
        df: the assembled paired frame, before any column is derived from it.
        n_attempted: how many molecules were drawn and attempted.

    Raises:
        SystemExit: with a message naming the likely cause.
    """
    missing = [c for c in SCORE_COLUMNS if c not in df.columns]
    if len(df) >= 3 and not missing:
        return
    raise SystemExit(
        "validate_size_confound.py CANNOT REPORT — the paired sample is too "
        "small.\n"
        f"  molecules attempted: {n_attempted}\n"
        f"  docked against BOTH receptors: {len(df)} (need >= 3 for a "
        "correlation)\n"
        + (f"  missing score column(s): {', '.join(missing)}\n" if missing
           else "")
        + "\n"
        "  A molecule is kept only when BOTH the holo and the apo dock return a\n"
        "  number, so an empty or tiny paired sample means one of the arms is\n"
        "  failing for (almost) every molecule. Most likely causes, in order:\n"
        f"    1. the apo receptor {APO_RECEPTOR!r} is unreadable or not a valid\n"
        "       PDBQT (require_apo_receptor only checks that it EXISTS);\n"
        "    2. `vina` is not on PATH, so every fresh dock fails;\n"
        "    3. ligand preparation (meeko/openbabel) is failing.\n"
        "\n"
        "  Fix the arm and re-run. Reporting a correlation from fewer than 3\n"
        "  paired molecules, or from one arm alone, would not be the measurement\n"
        "  this script exists to make."
    )


def dock_apo(smiles):
    """Dock against the pre-fix apo receptor, same box/effort/seed as the oracle."""
    spec = docking.TARGETS[TARGET]
    cx, cy, cz = spec["center"]
    sx, sy, sz = spec["size"]
    lig = out = None
    try:
        lig = docking.prepare_ligand(smiles)
        out = tempfile.NamedTemporaryFile(suffix=".pdbqt", delete=False).name
        res = subprocess.run(
            ["vina", "--receptor", APO_RECEPTOR, "--ligand", lig,
             "--center_x", str(cx), "--center_y", str(cy), "--center_z", str(cz),
             "--size_x", str(sx), "--size_y", str(sy), "--size_z", str(sz),
             "--exhaustiveness", str(docking.DEFAULT_EXHAUSTIVENESS),
             "--num_modes", "9", "--seed", str(docking.DEFAULT_VINA_SEED),
             "--out", out],
            capture_output=True, text=True, check=True)
        return docking._parse_best_affinity(res.stdout)
    except Exception:                                                  # noqa: BLE001
        return None
    finally:
        for p in (lig, out):
            if p and os.path.exists(p):
                os.unlink(p)


def descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {"MW": Descriptors.MolWt(mol), "logP": Crippen.MolLogP(mol),
            "heavy_atoms": mol.GetNumHeavyAtoms()}


def viable_pairs(df):
    """Split a paired frame into the analysable rows and the excluded ones.

    A row is kept only when BOTH docks found a binding mode
    (``evaluation.dock_is_viable`` on each raw affinity). Dropping a row from
    one arm only would leave the two correlations computed over DIFFERENT
    populations, which is precisely the confound this script exists to control
    for — so viability, like every other inclusion criterion here, is applied
    pairwise. The same choice ``evaluation.add_selectivity_index`` and
    ``matrix_report.both_docks_viable`` already make for quantities that need
    both docks.

    Viability is recomputed from the affinities, never read from a stored flag.

    Args:
        df: paired frame carrying the ``SCORE_COLUMNS``.

    Returns:
        ``(kept, dropped)`` — two frames partitioning ``df``, each with a fresh
        ``holo_viable`` / ``apo_viable`` bool column.
    """
    out = df.copy()
    both = np.ones(len(out), dtype=bool)
    for col in SCORE_COLUMNS:
        viable = evaluation.dock_is_viable(out[col].to_numpy(float))
        out[col + "_viable"] = viable
        both &= viable
    return out[both].reset_index(drop=True), out[~both].reset_index(drop=True)


def corr_block(df, score_col, label):
    """Pearson + Spearman of MW/logP against one arm's RAW affinities.

    ``df`` must already be viability-filtered (``viable_pairs``); the assertion
    below is the backstop, because a non-viable row at +4.213 kcal/mol sits
    about 9 kcal/mol outside the [-11, -5] band and moves a Pearson coefficient
    measurably in one direction, which reads as noise rather than as an error.
    """
    raw = df[score_col].to_numpy(float)
    if not bool(np.all(evaluation.dock_is_viable(raw))):
        bad = raw[~evaluation.dock_is_viable(raw)]
        raise AssertionError(
            f"corr_block({label}): {bad.size} non-viable dock(s) in the "
            f"correlation sample (raw kcal/mol "
            f"{', '.join(f'{v:+.3f}' for v in bad[:8])}). A dock with no "
            "binding mode is not an affinity; viable_pairs should have removed "
            "it."
        )
    out = {}
    for desc in ("MW", "logP"):
        x = df[desc].to_numpy(float)
        out[desc] = (float(pearson_r(x, raw)), float(spearman_rho(x, raw)))
    return out


def main():
    # Preflight BEFORE the docking loop: a missing apo receptor makes every
    # result below impossible, and discovering that after 150 docks is worse
    # than discovering it now.
    apo_path = require_apo_receptor()
    print(f"apo receptor: {apo_path}")

    lib = pd.read_csv("data/library/smiles.csv")
    col = "smiles" if "smiles" in lib.columns else lib.columns[0]
    all_smiles = lib[col].tolist()
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(all_smiles), size=N_SAMPLE, replace=False)
    sample = [all_smiles[i] for i in sorted(idx)]
    print(f"sample: {len(sample)} molecules drawn seed={SEED} from the RAW "
          f"{len(all_smiles)}-molecule library (screened classes included)\n")

    rows = []
    for n, smi in enumerate(sample, 1):
        d = descriptors(smi)
        if d is None:
            continue
        holo = docking.dock_target(smi, target=TARGET)     # fingerprinted cache
        apo = dock_apo(smi)
        if holo is None or apo is None:
            continue
        rows.append({"SMILES": smi, "holo": holo, "apo": apo, **d})
        if n % 25 == 0:
            print(f"  docked {n}/{len(sample)}")
    attempted = pd.DataFrame(rows)
    require_paired_sample(attempted, len(sample))

    # Viability, pairwise, once. ``kept`` is what every analysis below reads;
    # ``dropped`` is reported and still written to the CSV, so the record shows
    # what was measured AND what was excluded.
    df, dropped = viable_pairs(attempted)
    for frame in (df, dropped):
        frame["holo_LE"] = frame["holo"] / frame["heavy_atoms"]
        frame["apo_LE"] = frame["apo"] / frame["heavy_atoms"]
    # LE of a binding mode that was not found is not a quantity — the same rule
    # evaluation.build_objective_matrices applies to the *_LE report columns.
    for frame in (df, dropped):
        for col in SCORE_COLUMNS:
            frame.loc[~frame[col + "_viable"], col + "_LE"] = np.nan
    record = pd.concat([df, dropped], ignore_index=True) if len(dropped) else df
    record.to_csv(os.path.join(OUT, "task6_sample.csv"), index=False)

    print(f"\npaired sample: {len(df)} molecules docked against BOTH receptors "
          f"with a binding mode found on both")
    print(f"  {len(dropped)} row(s) EXCLUDED as non-viable "
          "(evaluation.dock_is_viable: no binding mode found on at least one "
          "receptor)")
    if len(dropped):
        print("  excluded rows (raw kcal/mol as measured, 'n' = non-viable):")
        for _, r in dropped.iterrows():
            print(f"    holo {r['holo']:+8.3f} "
                  f"[{'Y' if r['holo_viable'] else 'n'}]   "
                  f"apo {r['apo']:+8.3f} "
                  f"[{'Y' if r['apo_viable'] else 'n'}]   "
                  f"heavy {int(r['heavy_atoms'])}")
        print("  A score at or above 0 kcal/mol is absence of a binding mode, "
              "not weak binding;")
        print("  it is excluded PAIRWISE so both arms keep the same population.")
    print(f"  task6_sample.csv records all {len(record)} attempted row(s) with "
          "their viability flags.\n")

    print("=" * 74)
    print("1. SIZE / LIPOPHILICITY CONFOUND — identical molecules, both receptors")
    print("=" * 74)
    print(f"{'':<12}{'APO (pre-fix)':>26}{'HOLO (corrected)':>26}")
    print(f"{'descriptor':<12}{'Pearson':>13}{'Spearman':>13}"
          f"{'Pearson':>13}{'Spearman':>13}")
    apo_c = corr_block(df, "apo", "apo")
    holo_c = corr_block(df, "holo", "holo")
    for desc in ("MW", "logP"):
        ap, asp = apo_c[desc]
        hp, hsp = holo_c[desc]
        print(f"{desc:<12}{ap:>13.3f}{asp:>13.3f}{hp:>13.3f}{hsp:>13.3f}")
    print(f"\n  recorded apo baseline for reference: MW -0.44, logP -0.51")

    print("\n" + "=" * 74)
    print("2. NON-MONOTONICITY CHECK — mean score by heavy-atom bin")
    print("=" * 74)
    bins = [0, 18, 22, 26, 30, 34, 100]
    df["bin"] = pd.cut(df["heavy_atoms"], bins)
    g = df.groupby("bin", observed=True).agg(
        n=("holo", "size"), apo=("apo", "mean"), holo=("holo", "mean"))
    print(f"{'heavy atoms':<14}{'n':>5}{'apo mean':>12}{'holo mean':>12}")
    for b, r in g.iterrows():
        print(f"{str(b):<14}{int(r['n']):>5}{r['apo']:>12.2f}{r['holo']:>12.2f}")

    print("\n" + "=" * 74)
    print("3. KNOWN CLINICAL ANTIFOLATES — rank by RAW kcal vs by LE (holo)")
    print("=" * 74)
    # Reference distributions: the viability-filtered paired sample only. A
    # non-viable row left in here would inflate every percentile below it.
    raw_dist = df["holo"].to_numpy(float)
    le_dist = df["holo_LE"].to_numpy(float)
    apo_raw_dist = df["apo"].to_numpy(float)
    apo_le_dist = df["apo_LE"].to_numpy(float)
    print(f"{'drug':<16}{'raw kcal':>10}{'raw pct':>9}{'LE':>9}{'LE pct':>8}"
          f"{'| apo raw pct':>15}{'apo LE pct':>12}")
    pcts = {"raw": [], "le": [], "apo_raw": [], "apo_le": []}
    n_nonviable_actives = 0
    for a in KNOWN_ACTIVES:
        smi = a["smiles"]
        holo = docking.dock_target(smi, target=TARGET)
        apo = dock_apo(smi)
        mol = Chem.MolFromSmiles(smi)
        ha = mol.GetNumHeavyAtoms()
        if holo is None or apo is None:
            print(f"{a['name']:<16} docking failed")
            n_nonviable_actives += 1
            continue
        # The drug's OWN dock is gated too: a percentile for a binding mode
        # that was not found is not a quantity. The measurement is still
        # printed — withholding the rank is not the same as hiding the number.
        hv = bool(evaluation.dock_is_viable(holo))
        av = bool(evaluation.dock_is_viable(apo))
        if not (hv and av):
            n_nonviable_actives += 1
            print(f"{a['name']:<16}{holo:>10.2f}{'n/a':>9}{'n/a':>9}{'n/a':>8}"
                  f"{'n/a':>15}{'n/a':>12}   <- NON-VIABLE "
                  f"(holo {'Y' if hv else 'n'}, apo {'Y' if av else 'n'}; "
                  f"apo raw {apo:+.2f})")
            continue
        hle, ale = holo / ha, apo / ha
        p_raw = 100.0 * (raw_dist > holo).mean()
        p_le = 100.0 * (le_dist > hle).mean()
        p_araw = 100.0 * (apo_raw_dist > apo).mean()
        p_ale = 100.0 * (apo_le_dist > ale).mean()
        for k, v in (("raw", p_raw), ("le", p_le), ("apo_raw", p_araw),
                     ("apo_le", p_ale)):
            pcts[k].append(v)
        print(f"{a['name']:<16}{holo:>10.2f}{p_raw:>8.0f}%{hle:>9.3f}"
              f"{p_le:>7.0f}%{p_araw:>14.0f}%{p_ale:>11.0f}%")
    if pcts["raw"]:
        print(f"\n  mean percentile  RAW {np.mean(pcts['raw']):.0f}%   "
              f"LE {np.mean(pcts['le']):.0f}%   "
              f"(apo: RAW {np.mean(pcts['apo_raw']):.0f}%, "
              f"LE {np.mean(pcts['apo_le']):.0f}%)")
        print(f"  ({len(pcts['raw'])}/{len(KNOWN_ACTIVES)} known actives "
              "contributed; the mean is over those only)")
    else:
        print("\n  NO known active produced a viable dock on both receptors, so "
              "there is no\n  mean percentile to report. That is evidence the "
              "DOCKING is broken, not a\n  result about the compounds.")
    if n_nonviable_actives:
        print(f"  {n_nonviable_actives}/{len(KNOWN_ACTIVES)} known active(s) "
              "excluded as non-viable or failed. A clinical antifolate that "
              "does not\n  dock is a docking problem; investigate before "
              "reading anything above.")
    print("  percentile = % of the sample this drug BEATS (higher is better)")


if __name__ == "__main__":
    main()
