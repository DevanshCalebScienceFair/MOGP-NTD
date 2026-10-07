"""
validate_docking.py
===================

Standalone diagnostic: is the PfDHFR docking objective measuring real binding, or
just molecular size / lipophilicity?

AutoDock Vina's scoring function sums per-atom interaction terms, so its scores
drift more negative ("better") for bigger, greasier molecules almost regardless
of complementarity. If our docking objective is dominated by that artifact, then
the whole multi-objective BO pipeline is optimizing molecular weight and logP in
disguise. This script answers the question with five checks and one figure, so a
reader can decide in ~30 seconds whether the docking signal is trustworthy:

  1. Per-molecule MW, logP (Crippen) and heavy-atom count from SMILES (RDKit).
  2. A PfDHFR docking-score sample, preferring molecules already docked (existing
     run outputs, then the persistent docking cache) and only docking fresh to
     top up to --n-sample. The provenance breakdown is printed.
  3. Pearson AND Spearman of MW/logP against BOTH raw docking AND ligand
     efficiency (LE = docking / heavy_atoms), printed side by side so the drop is
     visible at a glance. LE targets SIZE directly and lipophilicity only partly,
     so a residual LE-vs-logP correlation (|r| > ~0.3) is flagged as needing a
     lipophilicity-aware escalation (LLE) rather than declaring victory.
  4. Ligand efficiency (docking score / heavy-atom count) — the standard
     size-debiased view — and how much re-ranking by LE changes the top molecules
     vs the raw score (a remedy, not just a diagnosis).
  5. The four KNOWN_ACTIVES (clinical antifolates, reused from
     validate_known_actives) docked against PfDHFR, each reported as a percentile
     within the library distribution under BOTH raw docking and LE: do real drugs
     rise once size is corrected, or does LE over-correct into tiny fragments?

MEASUREMENT vs OBJECTIVE (read before changing anything here)
-------------------------------------------------------------
Every number this script prints is a MEASUREMENT — "what Vina returned for this
molecule" — and every distribution it measures a percentile against is a
reference distribution of real binding modes. Two consequences, both enforced in
code below rather than by convention:

  1. The affinity is read from the ``*_Docking_raw_kcal`` MEASUREMENT column,
     never from the ``*_Docking`` OBJECTIVE column. Since the docking-viability
     fix, a dock that found no binding mode has its OBJECTIVE column replaced by
     that objective's worst value (``evaluation.apply_dock_viability``), so
     reading the objective column here would print a substituted cube corner as
     if an instrument had produced it. Old pre-fix CSVs have no raw column and
     their ``*_Docking`` column IS the raw measurement; ``measured_affinity_column``
     decides which generation a file is, by column presence.
  2. Non-viable docks (``evaluation.dock_is_viable`` False) are EXCLUDED from the
     library reference distribution. A score at or above 0 kcal/mol is absence of
     a binding mode, not weak binding; leaving it in moves the correlations and
     the known-active percentiles it is compared against. The exclusion count is
     printed, so the sample is never silently smaller than it looks.

The exclusion happens in exactly ONE place — ``split_viable``, called once per
score source inside ``build_sample`` — and every function that consumes a
reference distribution re-checks the invariant through
``assert_viable_reference``. So removing the filter does not quietly change a
correlation by a few hundredths; it raises.

The viability threshold itself appears NOWHERE in this file — everything calls
``evaluation.dock_is_viable``.

It imports the pipeline's own docking / library code but MODIFIES NOTHING; it is
a read-only observer that uses the docking cache like every other entry point.

Run ``python validate_docking.py --help`` for options. Docking is expensive, so
the first run on a cold cache will take a while for the fresh top-up docks; every
subsequent run is nearly instant (cache hits).
"""

import os
import glob
import argparse

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen

import docking
import evaluation
from docking_cache import canonicalize_smiles
from data import load_library
from validate_known_actives import KNOWN_ACTIVES

# Own-code statistics (mogp_core.stats) in place of scipy.stats. ``pearson_r``
# reproduces scipy's normalise-then-dot formulation; ``spearman_rho`` is Pearson
# on TIE-CORRECTED average ranks, NOT the 1-6*sum(d^2)/(n(n^2-1)) shortcut, which
# is only an algebraic simplification for tie-FREE samples. heavy_atoms here is
# integer and heavily tied (measured on this script's own 2108-molecule sample:
# 30 distinct values, largest tie group 148), and the shortcut is wrong by a
# measured 1.836e-03 on that sample, 4.323e-03 at n=150 and 4.571e-01 at n=5.
# Agreement with scipy on the same sample: pearson max|diff| 0.0, spearman
# max|diff| 3.331e-16.
from mogp_core.stats import pearson_r, spearman_rho


# The two PfDHFR docking columns a results CSV may carry.
#
#   PFDHFR_COLUMN      the OBJECTIVE column (mogp.TASK_NAMES). Since the
#                      viability fix a non-viable dock's entry here is the
#                      SUBSTITUTED worst value for this objective's direction
#                      (-5.0 kcal/mol, PfDHFR being minimized) — a cube corner
#                      no instrument produced.
#   PFDHFR_RAW_COLUMN  the MEASUREMENT column, written beside it by
#                      ``evaluation.add_docking_measurement_columns``. Exactly
#                      what Vina returned, never substituted.
#
# This script reports MEASUREMENTS, so it reads the second whenever a file has
# it. ``measured_affinity_column`` is the only place that choice is made.
PFDHFR_COLUMN = "PfDHFR_Docking"
PFDHFR_RAW_COLUMN = PFDHFR_COLUMN + evaluation.RAW_KCAL_SUFFIX
DOCK_TARGET = "PfDHFR"

# The sample table's affinity column, named for what it holds: the raw
# whole-molecule Vina measurement in kcal/mol. It is deliberately NOT called
# "docking", which would read as the objective and invite the confusion this
# whole module docstring is about.
RAW_KCAL_COLUMN = "raw_kcal"

# Default directories that may hold a method's evaluated.csv (loop + baselines +
# ablation arms). Any of these that exist are scanned for already-docked scores;
# so is any other ``*results*`` directory found next to this script.
DEFAULT_RESULTS_DIRS = [
    "results",
    "results_coregionalized",
    "baseline_random_results",
    "baseline_single_obj_results",
    "baseline_greedy_results",
]

# Correlation strength thresholds for the plain-language verdict.
CORR_STRONG = 0.5
CORR_MODERATE = 0.3

# Below this heavy-atom count an "efficient" LE winner is really a fragment. LE
# (= docking / heavy_atoms) mechanically rewards tiny molecules, so if the LE-top
# set is dominated by sub-fragment-sized molecules, LE is OVER-correcting size
# rather than debiasing it — the opposite failure mode to a raw size confound.
FRAGMENT_HEAVY_ATOMS = 15


# ---------------------------------------------------------------------- #
# Measurement vs objective: which column, and which rows
# ---------------------------------------------------------------------- #
def measured_affinity_column(df):
    """Which column of a results frame holds the MEASURED PfDHFR affinity.

    HOW A PRE-FIX FILE IS DISTINGUISHED FROM A POST-FIX ONE: by COLUMN
    PRESENCE, not by a version stamp, a mtime or a value heuristic — there is no
    version field in these CSVs, and no value test can tell a measured -5.0 from
    a substituted one.

      * ``PfDHFR_Docking_raw_kcal`` present -> POST-FIX file. The frame was
        written by ``evaluation.add_docking_measurement_columns``, so the
        measurement lives in that column and the ``PfDHFR_Docking`` column
        beside it may already hold a substituted worst value. Return the raw
        column.
      * ``PfDHFR_Docking_raw_kcal`` absent, ``PfDHFR_Docking`` present ->
        PRE-FIX file. Nothing substituted anything, so its ``PfDHFR_Docking``
        column IS the unmodified measurement and reading it is exactly right
        (this is why the fallback is correct, not merely tolerated). Return the
        objective column.
      * neither present -> ``None``; the caller skips the file.

    The same two-generation resolution ``evaluation._docking_measurement_from_frame``
    and ``matrix_report.docking_viability`` already use, restricted to the one
    target this script docks. The suffix itself is read from
    ``evaluation.RAW_KCAL_SUFFIX``, so the name cannot drift from the writer.

    Args:
        df: a ``pandas.DataFrame`` read from an ``evaluated.csv``.

    Returns:
        The column name to read the measurement from, or ``None``.
    """
    if PFDHFR_RAW_COLUMN in df.columns:
        return PFDHFR_RAW_COLUMN
    if PFDHFR_COLUMN in df.columns:
        return PFDHFR_COLUMN
    return None


def split_viable(index_to_raw):
    """Partition ``{library_index: raw kcal/mol}`` into viable and non-viable.

    THE ONLY viability filter in this file. Every score source funnels through
    it, so there is one decision point to audit and one to fault-inject, and the
    predicate is ``evaluation.dock_is_viable`` — the threshold is never written
    down here.

    A non-viable dock is one whose best Vina affinity is ``>= 0 kcal/mol`` (no
    favourable pose was found) or non-finite (the dock failed). Neither is a
    binding affinity, so neither belongs in a distribution of binding
    affinities. The non-viable half is RETURNED rather than discarded, so the
    caller can report how many rows were excluded and what they measured.

    Args:
        index_to_raw: ``{library_index: raw whole-molecule kcal/mol}``.

    Returns:
        ``(viable, non_viable)``, two dicts of the same shape as the input,
        together partitioning it.
    """
    viable, non_viable = {}, {}
    for idx, raw in index_to_raw.items():
        target = viable if bool(evaluation.dock_is_viable(raw)) else non_viable
        target[idx] = float(raw)
    return viable, non_viable


def assert_viable_reference(sample_df, where):
    """Fail loudly if a non-viable dock reached a reference distribution.

    ``build_sample`` filters, but a filter that is silently removed moves every
    correlation and every known-active percentile by a small amount that looks
    like noise. This re-checks the invariant at each point of USE, so the
    failure is an exception naming the offending values rather than a plausible
    wrong number in a write-up.

    Args:
        sample_df: the sample table, which must carry ``RAW_KCAL_COLUMN``.
        where: name of the calling report, for the message.

    Raises:
        AssertionError: if any row's raw affinity is not
            ``evaluation.dock_is_viable``.
    """
    raw = np.asarray(sample_df[RAW_KCAL_COLUMN], dtype=float)
    viable = evaluation.dock_is_viable(raw)
    if bool(np.all(viable)):
        return
    bad = raw[~viable]
    raise AssertionError(
        f"{where}: {bad.size} non-viable dock(s) reached the reference "
        f"distribution (raw kcal/mol = "
        f"{', '.join(f'{v:+.3f}' for v in bad[:8])}"
        f"{', ...' if bad.size > 8 else ''}). A dock with no binding mode is "
        "absence of evidence, not weak binding, and must not be a reference "
        "point. build_sample/split_viable should have excluded it."
    )


# ---------------------------------------------------------------------- #
# RDKit descriptors
# ---------------------------------------------------------------------- #
def compute_descriptors(smiles):
    """Return ``(MW, logP, heavy_atom_count)`` for a SMILES, or None if unparseable.

    MW is ``Descriptors.MolWt``, logP is Crippen ``MolLogP`` — the same
    definitions a medicinal chemist would quote — and the heavy-atom count is the
    denominator for ligand efficiency.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return (
        float(Descriptors.MolWt(mol)),
        float(Crippen.MolLogP(mol)),
        int(mol.GetNumHeavyAtoms()),
    )


# ---------------------------------------------------------------------- #
# Assemble a PfDHFR docking-score sample (outputs -> cache -> fresh)
# ---------------------------------------------------------------------- #
def _discover_results_dirs():
    """Existing result dirs to scan: the known defaults plus any ``*results*``."""
    dirs = list(DEFAULT_RESULTS_DIRS)
    dirs += [d for d in glob.glob("*results*") if os.path.isdir(d)]
    # De-dupe, preserve order, keep only those that exist.
    seen, out = set(), []
    for d in dirs:
        if d in seen or not os.path.isdir(d):
            continue
        seen.add(d)
        out.append(d)
    return out


def scores_from_outputs(canon_to_index):
    """Collect PfDHFR MEASUREMENTS from existing ``evaluated.csv`` run outputs.

    The affinity is read from whichever column actually holds the measurement —
    ``measured_affinity_column`` decides, per file, by column presence — so a
    post-fix file's substituted ``PfDHFR_Docking`` objective value is never
    mistaken for something Vina returned. A file with neither column is skipped.

    Only rows whose (canonicalized) SMILES is in the current library are kept —
    matched by canonical SMILES to a library index — so every collected score
    aligns with a molecule we can compute descriptors for.

    Viability is NOT applied here: this returns the measurements, and
    ``build_sample`` makes the single viability decision through
    ``split_viable``. Non-finite entries are dropped, because a NaN is the
    absence of a measurement rather than one.

    Returns ``(index_to_raw, source_dirs, column_by_dir)``: a dict
    library-index -> raw kcal/mol, the list of directories that actually
    contributed, and ``{directory: column_name_read}`` so the provenance report
    can state which generation each file was read as.
    """
    index_to_raw = {}
    source_dirs = []
    column_by_dir = {}
    for directory in _discover_results_dirs():
        path = os.path.join(directory, "evaluated.csv")
        if not os.path.exists(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception:                                        # pragma: no cover
            continue
        if "SMILES" not in df.columns:
            continue
        column = measured_affinity_column(df)
        if column is None:
            continue
        contributed = False
        for smiles, score in zip(df["SMILES"].astype(str), df[column]):
            if not np.isfinite(score):
                continue
            idx = canon_to_index.get(canonicalize_smiles(smiles))
            if idx is None or idx in index_to_raw:
                continue
            index_to_raw[idx] = float(score)
            contributed = True
        if contributed:
            source_dirs.append(directory)
            column_by_dir[directory] = column
    return index_to_raw, source_dirs, column_by_dir


def scores_from_cache(library_smiles, canon_list, skip_indices):
    """Look up already-cached PfDHFR docks for library molecules not already scored.

    Consults the persistent docking cache directly (read-only) so molecules docked
    by any previous run are reused without re-docking. Returns a dict
    library-index -> raw kcal/mol for cache HITS (finite affinity) only.

    The cache stores what the oracle MEASURED — it predates and is untouched by
    the objective substitution, so there is no column question here. Viability
    is still not decided here: ``build_sample`` applies ``split_viable`` to this
    dict exactly as it does to the run-output dict, so a cached ``+4.213``
    kcal/mol clash cannot enter the reference distribution by the back door.
    """
    cache = docking.get_cache()
    # The cache is keyed on the oracle fingerprint (receptor + box + effort +
    # seed), so a read must name the configuration it wants. Asking for the
    # CURRENT one means scores computed against a different receptor are misses
    # rather than silently mixed into this sample.
    fingerprint = docking.oracle_fingerprint(DOCK_TARGET)
    index_to_score = {}
    for idx, canon in enumerate(canon_list):
        if idx in skip_indices:
            continue
        cached = cache.get(canon, DOCK_TARGET, fingerprint)
        if cached is None:
            continue
        status, affinity = cached
        if status == docking.STATUS_OK and affinity is not None and np.isfinite(affinity):
            index_to_score[idx] = float(affinity)
    return index_to_score


def dock_fresh(library_smiles, indices):
    """Freshly dock the given library indices against PfDHFR (cache ON).

    ``dock_target`` writes each result to the shared cache, so a later run reuses
    them. Returns a dict library-index -> raw kcal/mol for the docks that
    succeeded, i.e. the MEASUREMENT straight from the oracle; viability is
    decided once, by ``build_sample``.
    """
    index_to_score = {}
    n = len(indices)
    for i, idx in enumerate(indices, start=1):
        print(f"  fresh dock {i}/{n} (library #{idx}) against {DOCK_TARGET}...",
              flush=True)
        score = docking.dock_target(library_smiles[idx], target=DOCK_TARGET,
                                    use_cache=True)
        if score is not None and np.isfinite(score):
            index_to_score[idx] = float(score)
    return index_to_score


def build_sample(library, n_sample, seed):
    """Build the docking-MEASUREMENT sample, preferring cheap sources over fresh docks.

    Order of preference (all avoid re-docking except the last):
        existing run outputs  ->  docking cache  ->  fresh docks to top up.

    Every source yields raw kcal/mol (see each helper), and each one is passed
    through ``split_viable`` — the single viability filter — before it joins the
    sample. Non-viable molecules are also removed from the fresh-dock candidate
    pool, so the top-up cannot re-dock a molecule that was just excluded and the
    ``--n-sample`` target counts VIABLE molecules.

    Returns ``(sample_df, provenance)`` where ``sample_df`` has columns
    ``[library_index, SMILES, raw_kcal, MW, logP, heavy_atoms]`` — rows with an
    unparseable SMILES dropped, and NO non-viable row — and ``provenance`` is a
    dict of counts plus the exclusion ledger.
    """
    smiles = library["smiles"]
    n_lib = len(smiles)

    # Canonical SMILES for the whole library, once, so outputs/cache match by
    # canonical identity (the same key the docking cache uses).
    canon_list = [canonicalize_smiles(s) for s in smiles]
    canon_to_index = {}
    for idx, canon in enumerate(canon_list):
        canon_to_index.setdefault(canon, idx)

    # 1) Existing run outputs (measurement column resolved per file).
    out_raw, source_dirs, column_by_dir = scores_from_outputs(canon_to_index)
    # 2) Docking cache (for molecules not already covered by outputs).
    cache_raw = scores_from_cache(smiles, canon_list, skip_indices=set(out_raw))

    # THE viability decision, once per source. ``excluded`` keeps the measured
    # values so the report can name what was dropped instead of just counting.
    out_scores, out_bad = split_viable(out_raw)
    cache_scores, cache_bad = split_viable(cache_raw)
    scores = dict(out_scores)
    scores.update(cache_scores)
    excluded = dict(out_bad)
    excluded.update(cache_bad)
    n_outputs = len(out_scores)
    n_cache = len(cache_scores)
    n_excluded_outputs = len(out_bad)
    n_excluded_cache = len(cache_bad)

    # 3) Fresh docks only if the free VIABLE scores fall short of the request.
    n_fresh = 0
    n_excluded_fresh = 0
    if len(scores) < n_sample:
        rng = np.random.default_rng(seed)
        remaining = [i for i in range(n_lib)
                     if i not in scores and i not in excluded]
        rng.shuffle(remaining)
        need = n_sample - len(scores)
        to_dock = remaining[:need]
        if to_dock:
            print(f"\nFree viable scores ({len(scores)}) < --n-sample "
                  f"({n_sample}); docking {len(to_dock)} fresh molecule(s) "
                  f"against {DOCK_TARGET} (cache ON)...")
            fresh_raw = dock_fresh(smiles, to_dock)
            fresh, fresh_bad = split_viable(fresh_raw)
            scores.update(fresh)
            excluded.update(fresh_bad)
            n_fresh = len(fresh)
            n_excluded_fresh = len(fresh_bad)

    # Assemble the sample table with descriptors.
    rows = []
    for idx, score in scores.items():
        desc = compute_descriptors(smiles[idx])
        if desc is None:
            continue
        mw, logp, hac = desc
        rows.append((idx, smiles[idx], score, mw, logp, hac))
    sample_df = pd.DataFrame(
        rows,
        columns=["library_index", "SMILES", RAW_KCAL_COLUMN, "MW", "logP",
                 "heavy_atoms"],
    )

    provenance = {
        "n_outputs": n_outputs,
        "n_cache": n_cache,
        "n_fresh": n_fresh,
        "n_total": len(sample_df),
        "source_dirs": source_dirs,
        "measured_columns": column_by_dir,
        "n_excluded_outputs": n_excluded_outputs,
        "n_excluded_cache": n_excluded_cache,
        "n_excluded_fresh": n_excluded_fresh,
        "n_excluded": len(excluded),
        "excluded_raw_kcal": sorted(excluded.values(), reverse=True),
    }
    return sample_df, provenance


# ---------------------------------------------------------------------- #
# Correlations + verdict
# ---------------------------------------------------------------------- #
def _strength_word(r):
    """Plain-language strength for a correlation coefficient's magnitude."""
    a = abs(r)
    if a > CORR_STRONG:
        return "STRONG"
    if a > CORR_MODERATE:
        return "moderate"
    return "weak"


def report_correlations(sample_df):
    """Print Pearson + Spearman of MW/logP against RAW docking AND ligand efficiency.

    The RAW docking block is the original diagnostic (unchanged); the LIGAND
    EFFICIENCY (LE = docking / heavy_atoms) block follows immediately so the
    size/lipophilicity confound before vs after the correction sits side by side.

    The affinities are MEASUREMENTS and the sample is viability-filtered
    (``assert_viable_reference``), so a dock that found no binding mode cannot
    pull a correlation. That matters most here: a non-viable score sits several
    kcal/mol outside the [-11, -5] distribution, where a single row has leverage.

    Returns ``(corrs, raw_flagged)`` where ``corrs`` is
    ``{"MW": {"raw": (p, s), "le": (p, s)}, "logP": {...}, "heavy_atoms": {...}}``
    and ``raw_flagged`` is True if any RAW MW/logP |r| > CORR_STRONG.
    """
    print("\n" + "=" * 78)
    print("3. SIZE / LIPOPHILICITY CONFOUND — RAW docking vs LIGAND EFFICIENCY")
    print("=" * 78)

    assert_viable_reference(sample_df, "report_correlations")
    y = sample_df[RAW_KCAL_COLUMN].to_numpy()
    n = len(y)
    if n < 3:
        print(f"  Only {n} scored molecule(s); need >=3 for a correlation. "
              "Increase --n-sample or dock more.")
        return {}, False

    print(f"  Docking score = PfDHFR affinity AS MEASURED by Vina (kcal/mol, "
          f"more negative = stronger binding).  n = {n}.")
    print(f"  A negative r means BIGGER / GREASIER molecules get 'better' "
          "(more negative) scores.\n")

    # --- RAW docking (the original objective) — output preserved verbatim ---
    out = {}
    flagged = False
    for col in ("MW", "logP"):
        x = sample_df[col].to_numpy()
        pear = float(pearson_r(x, y))
        spear = float(spearman_rho(x, y))
        out[col] = {"raw": (pear, spear)}
        strong = abs(pear) > CORR_STRONG or abs(spear) > CORR_STRONG
        flagged = flagged or strong
        label = "docking vs " + col
        print(f"  {label:<16} Pearson r = {pear:+.3f} ({_strength_word(pear)}), "
              f"Spearman rho = {spear:+.3f} ({_strength_word(spear)})")
        if strong:
            print(f"      -> FLAG: the docking objective is PARTLY measuring "
                  f"{col} (|r| > {CORR_STRONG}), not binding alone.")
        else:
            print(f"      -> ok: no strong {col} dependence (|r| <= {CORR_STRONG}).")

    # --- LIGAND EFFICIENCY (the size-corrected objective we switched to) ---
    le = (sample_df[RAW_KCAL_COLUMN] / sample_df["heavy_atoms"]).to_numpy()
    print("\n  Now the SAME descriptors vs LIGAND EFFICIENCY "
          "(LE = docking / heavy_atoms):")
    print("  The fix works if LE's |r| vs MW drops toward 0 — size no longer "
          "buys score.")
    for col in ("MW", "logP"):
        x = sample_df[col].to_numpy()
        lp = float(pearson_r(x, le))
        ls = float(spearman_rho(x, le))
        out[col]["le"] = (lp, ls)
        raw_abs = abs(out[col]["raw"][0])
        label = "LE vs " + col
        print(f"  {label:<16} Pearson r = {lp:+.3f} ({_strength_word(lp)}), "
              f"Spearman rho = {ls:+.3f} ({_strength_word(ls)})"
              f"   [raw |r|={raw_abs:.2f} -> LE |r|={abs(lp):.2f}]")
        if col == "MW":
            if abs(lp) > CORR_STRONG:
                print("      -> SIZE OVER-SWING: LE now tracks MW STRONGLY in the "
                      "opposite sense (it favours smaller molecules); watch the "
                      "known-drug LE percentile and LE-top size for fragment bias.")
            elif abs(lp) < raw_abs:
                print(f"      -> good: size dependence shrank "
                      f"({raw_abs:.2f} -> {abs(lp):.2f}).")
            else:
                print("      -> LE did NOT reduce the MW dependence.")
        else:  # logP — LE corrects size directly, lipophilicity only partly.
            if abs(lp) > CORR_MODERATE:
                print(f"      -> RESIDUAL LIPOPHILICITY confound (|r| > "
                      f"{CORR_MODERATE}): LE debiases size, not logP. Escalate to "
                      "a lipophilicity-aware correction (LLE).")
            else:
                print(f"      -> logP confound now weak (|r| <= {CORR_MODERATE}).")

    # --- heavy_atoms, the descriptor CLAUDE.md's own figures are quoted in ---
    # CLAUDE.md quotes "paired Spearman(score, heavy_atoms) went -0.726 ->
    # -0.257, while LE's counter-bias barely moved (+0.891 -> +0.880)". Neither
    # number was reproducible from a checkout, because no script correlated
    # against heavy_atoms at all (it was only ever the LE denominator). These
    # two lines ARE that pair's holo-receptor side, measured here, so the
    # quoted figures have a live source again. The apo side cannot be measured:
    # it needs the deleted apo_backup_2026-08-10 receptor (see
    # validate_size_confound.py, which fails loudly about exactly that).
    ha = sample_df["heavy_atoms"].to_numpy(dtype=float)
    ha_raw_p, ha_raw_s = float(pearson_r(ha, y)), float(spearman_rho(ha, y))
    ha_le_p, ha_le_s = float(pearson_r(ha, le)), float(spearman_rho(ha, le))
    out["heavy_atoms"] = {"raw": (ha_raw_p, ha_raw_s), "le": (ha_le_p, ha_le_s)}
    print("\n  heavy_atoms (the descriptor CLAUDE.md's quoted size figures use, "
          "holo side only):")
    print(f"  {'docking vs heavy':<16} Pearson r = {ha_raw_p:+.3f}, "
          f"Spearman rho = {ha_raw_s:+.3f}")
    print(f"  {'LE vs heavy':<16} Pearson r = {ha_le_p:+.3f}, "
          f"Spearman rho = {ha_le_s:+.3f}")
    print("      -> the apo counterparts are NOT measurable here (the apo "
          "receptor is gone); do not pair these with the apo numbers in "
          "CLAUDE.md as if both came from this run.")
    return out, flagged


# ---------------------------------------------------------------------- #
# Ligand efficiency (size-debiased view)
# ---------------------------------------------------------------------- #
def report_ligand_efficiency(sample_df, top_k):
    """Compare the top molecules ranked by raw score vs by ligand efficiency.

    LE = docking score / heavy-atom count (both negative here, so more negative =
    more binding per atom). If LE re-ranks the leaders heavily, the raw score was
    rewarding size; LE is the standard size-debiased remedy.

    Returns the overlap fraction of the two top-``top_k`` sets.
    """
    print("\n" + "=" * 78)
    print(f"4. LIGAND EFFICIENCY (size-debiased):  LE = docking / heavy_atoms")
    print("=" * 78)

    assert_viable_reference(sample_df, "report_ligand_efficiency")
    df = sample_df.copy()
    df["LE"] = df[RAW_KCAL_COLUMN] / df["heavy_atoms"]

    k = min(top_k, len(df))
    # More negative = better for both raw score and LE.
    by_score = df.sort_values(RAW_KCAL_COLUMN).head(k)
    by_le = df.sort_values("LE").head(k)

    score_set = set(by_score["library_index"])
    le_set = set(by_le["library_index"])
    overlap = score_set & le_set
    overlap_frac = len(overlap) / k if k else 0.0
    dropped = score_set - le_set

    print(f"  Top {k} by RAW docking score vs top {k} by LIGAND EFFICIENCY:")
    print(f"    shared molecules:            {len(overlap)}/{k} "
          f"({overlap_frac * 100:.0f}%)")
    print(f"    raw-score leaders LE demotes: {len(dropped)}/{k}")

    corr_le_mw = (float(spearman_rho(df["MW"], df["LE"]))
                  if len(df) >= 3 else float("nan"))
    print(f"    Spearman(LE, MW) = {corr_le_mw:+.3f} "
          "(closer to 0 = LE has removed more of the size trend)")

    # Show the raw-score leaders that LE pushes out of the top-k, with their MW —
    # these are the "big molecule wins the raw score" cases LE corrects.
    if dropped:
        show = by_score[by_score["library_index"].isin(dropped)] \
            .sort_values(RAW_KCAL_COLUMN)
        print(f"\n  Raw-score leaders demoted by LE (likely size-driven):")
        print(f"    {'lib#':>6}{'raw kcal':>10}{'MW':>9}{'heavy':>7}{'LE':>9}")
        for _, r in show.head(8).iterrows():
            print(f"    {int(r['library_index']):>6}{r[RAW_KCAL_COLUMN]:>10.2f}"
                  f"{r['MW']:>9.1f}{int(r['heavy_atoms']):>7}{r['LE']:>9.3f}")

    if overlap_frac < 0.5:
        print(f"\n  -> LE substantially re-ranks the leaders "
              f"({overlap_frac * 100:.0f}% overlap): raw docking was rewarding "
              "SIZE. Prefer LE (or add a size penalty) downstream.")
    else:
        print(f"\n  -> LE largely agrees with the raw ranking "
              f"({overlap_frac * 100:.0f}% overlap): the leaders are not purely "
              "size-driven.")
    return overlap_frac


# ---------------------------------------------------------------------- #
# Known actives vs the library distribution
# ---------------------------------------------------------------------- #
def _percentile_better_than(score, distribution):
    """Percent of ``distribution`` this score binds MORE STRONGLY than.

    Docking is more-negative-is-better, so this is the fraction of the library
    with a WEAKER (higher) score — i.e. "this molecule out-docks X% of the
    library".

    ``distribution`` MUST already be viability-filtered. Its callers take it
    from a sample that ``assert_viable_reference`` has checked, which is where
    that is enforced; it cannot be re-checked here, because this function is
    also handed LE (per-atom) values, and the kcal/mol viability threshold is
    not a statement about per-atom numbers. A non-viable row left in would
    inflate every percentile below it and deflate none — a one-sided error, not
    noise.
    """
    dist = np.asarray(distribution, dtype=float)
    if dist.size == 0:
        return float("nan")
    return float(np.mean(score < dist) * 100.0)


def report_known_actives(sample_df):
    """Dock the KNOWN_ACTIVES against PfDHFR; report library percentile RAW and LE.

    Each drug's percentile is reported both under the RAW docking distribution
    (does it out-dock the library?) and under the LIGAND EFFICIENCY distribution
    (does it out-efficiency the library?). Real drugs that were mid-pack on raw
    score should RISE under LE once size is corrected — unless LE over-corrects,
    in which case they fall (library winners become tiny fragments).

    DOCKING VIABILITY applies on BOTH sides. The reference distributions are the
    viability-filtered sample (checked by ``assert_viable_reference``), and an
    ACTIVE whose own dock found no binding mode is reported with its raw
    measurement and ``n/a`` percentiles rather than being ranked — a non-viable
    known active is evidence the docking is broken, not that the drug is weak.

    Returns ``(active_rows, best_library_score)``; each row carries both
    ``percentile`` (raw) and ``le_percentile``, plus a ``viable`` flag.
    """
    print("\n" + "=" * 78)
    print("5. KNOWN CLINICAL ANTIFOLATES vs THE LIBRARY — RAW and LE percentiles")
    print("=" * 78)

    assert_viable_reference(sample_df, "report_known_actives")
    dist = sample_df[RAW_KCAL_COLUMN].to_numpy()
    dist_le = (sample_df[RAW_KCAL_COLUMN] / sample_df["heavy_atoms"]).to_numpy()
    best_lib = float(np.min(dist)) if dist.size else float("nan")
    median_lib = float(np.median(dist)) if dist.size else float("nan")
    best_le = float(np.min(dist_le)) if dist_le.size else float("nan")
    print(f"  Library sample (n={dist.size}, viable docks only): raw docking "
          f"best {best_lib:.2f}, median {median_lib:.2f} kcal/mol;  "
          f"best LE {best_le:.3f}.")
    print("  pRaw = % of library this drug OUT-DOCKS; pLE = % it OUT-EFFICIENCIES "
          "(both higher = better).\n")

    header = (f"  {'compound':<15}{'PfDHFR':>9}{'MW':>8}{'logP':>7}"
              f"{'heavy':>7}{'LE':>8}{'pRaw':>7}{'pLE':>7}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    active_rows = []
    for a in KNOWN_ACTIVES:
        desc = compute_descriptors(a["smiles"])
        score = docking.dock_target(a["smiles"], target=DOCK_TARGET, use_cache=True)
        if desc is None or score is None or not np.isfinite(score):
            print(f"  {a['name']:<15}{'  dock/parse failed':>40}")
            active_rows.append({"name": a["name"], "score": float("nan"),
                                "MW": float("nan"), "logP": float("nan"),
                                "heavy": float("nan"), "LE": float("nan"),
                                "percentile": float("nan"),
                                "le_percentile": float("nan"),
                                "viable": False})
            continue
        mw, logp, hac = desc
        if not bool(evaluation.dock_is_viable(score)):
            # The measurement is printed; the RANKING is withheld. Percentiles
            # of a non-existent binding mode are not quantities.
            print(f"  {a['name']:<15}{score:>9.2f}{mw:>8.1f}{logp:>7.2f}"
                  f"{hac:>7}{'n/a':>8}{'n/a':>7}{'n/a':>7}"
                  f"   <- NON-VIABLE dock (no binding mode found)")
            active_rows.append({"name": a["name"], "score": float(score),
                                "MW": mw, "logP": logp, "heavy": hac,
                                "LE": float("nan"),
                                "percentile": float("nan"),
                                "le_percentile": float("nan"),
                                "viable": False})
            continue
        le = score / hac
        pct = _percentile_better_than(score, dist)
        pct_le = _percentile_better_than(le, dist_le)
        active_rows.append({"name": a["name"], "score": float(score), "MW": mw,
                            "logP": logp, "heavy": hac, "LE": le,
                            "percentile": pct, "le_percentile": pct_le,
                            "viable": True})
        print(f"  {a['name']:<15}{score:>9.2f}{mw:>8.1f}{logp:>7.2f}"
              f"{hac:>7}{le:>8.3f}{pct:>6.0f}%{pct_le:>6.0f}%")

    finite = [r for r in active_rows if np.isfinite(r["percentile"])]
    if finite:
        med_pct = float(np.median([r["percentile"] for r in finite]))
        med_pct_le = float(np.median([r["le_percentile"] for r in finite]))
        move = med_pct_le - med_pct
        direction = ("ROSE" if move > 2 else "FELL" if move < -2 else "held")
        print(f"\n  Median known-active percentile: raw {med_pct:.0f}% -> "
              f"LE {med_pct_le:.0f}%  (drugs {direction} under LE).")
        if move > 2:
            print("  Real drugs climb once size is corrected -> the raw 'winners' "
                  "were size-inflated, and LE ranks the drugs more fairly.")
        elif move < -2:
            print("  Real drugs FALL under LE -> LE is likely over-correcting "
                  "(library 'winners' are tiny fragments, not the real drugs).")
    n_bad_actives = sum(1 for r in active_rows if not r["viable"])
    if n_bad_actives:
        print(f"\n  {n_bad_actives}/{len(active_rows)} known active(s) are "
              "NON-VIABLE or failed to dock and are excluded from the medians "
              "above. A clinical antifolate that does not dock is a DOCKING "
              "problem; investigate before reading anything else here.")
    return active_rows, best_lib


# ---------------------------------------------------------------------- #
# Figure
# ---------------------------------------------------------------------- #
def save_figure(sample_df, active_rows, output_path):
    """Save a 2x2 scatter grid: RAW docking (top) and LE (bottom) vs MW and logP.

    Top row is the original raw-docking diagnostic; bottom row is the same
    molecules under ligand efficiency, so the size/lipophilicity confound before
    vs after the correction is visible in one figure. Known actives are overlaid
    as labeled stars on every panel.

    Only VIABLE actives are plotted: a star at a positive kcal/mol would be a
    point on an affinity axis that is not an affinity.
    """
    import matplotlib
    matplotlib.use("Agg")            # headless: write a file, never open a window
    import matplotlib.pyplot as plt

    assert_viable_reference(sample_df, "save_figure")
    fig, axes = plt.subplots(2, 2, figsize=(13, 10.5))
    sample_raw = sample_df[RAW_KCAL_COLUMN].to_numpy()
    sample_le = (sample_df[RAW_KCAL_COLUMN] / sample_df["heavy_atoms"]).to_numpy()
    active_ok = [r for r in active_rows
                 if r["viable"] and np.isfinite(r["score"])]

    # (axis, x-column, x-label, sample-y, active-y-key, y-label, title)
    panels = [
        (axes[0, 0], "MW", "Molecular weight (Da)",
         sample_raw, "score",
         "PfDHFR docking (kcal/mol)  ↓ stronger", "RAW docking vs MW"),
        (axes[0, 1], "logP", "Crippen logP",
         sample_raw, "score",
         "PfDHFR docking (kcal/mol)  ↓ stronger", "RAW docking vs logP"),
        (axes[1, 0], "MW", "Molecular weight (Da)",
         sample_le, "LE",
         "Ligand efficiency (kcal/mol/atom)  ↓ better", "LE vs MW"),
        (axes[1, 1], "logP", "Crippen logP",
         sample_le, "LE",
         "Ligand efficiency (kcal/mol/atom)  ↓ better", "LE vs logP"),
    ]

    for ax, col, xlabel, ysample, akey, ylabel, title in panels:
        ax.scatter(sample_df[col], ysample, s=18, c="lightsteelblue",
                   edgecolors="none", alpha=0.7, label="Library sample")
        for r in active_ok:
            ax.scatter(r[col], r[akey], marker="*", s=260, c="crimson",
                       edgecolors="black", linewidths=0.6, zorder=5)
            ax.annotate(r["name"], (r[col], r[akey]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)
        if active_ok:
            ax.scatter([], [], marker="*", s=160, c="crimson",
                       edgecolors="black", label="Known actives")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.invert_yaxis()            # stronger / more efficient at the top
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    fig.suptitle("PfDHFR objective: RAW docking (top) vs LIGAND EFFICIENCY (bottom) "
                 "— does LE remove the size/lipophilicity confound?", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved scatter figure to {output_path}")


# ---------------------------------------------------------------------- #
# Entry point
# ---------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Diagnose whether the PfDHFR docking objective tracks real "
                    "binding or a molecular-size / lipophilicity artifact."
    )
    parser.add_argument("--library-dir", default="data/library",
                        help="Cached library directory (default data/library).")
    parser.add_argument("--n-sample", type=int, default=150,
                        help="Target docking-score sample size (default 150). "
                             "Existing outputs + cache are reused first; only the "
                             "shortfall is docked fresh.")
    parser.add_argument("--top-k", type=int, default=15,
                        help="Top-K set size for the LE re-ranking comparison.")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for the random fresh-dock top-up sample.")
    parser.add_argument("--output", default="validate_docking_scatter.png",
                        help="Path for the saved scatter figure.")
    args = parser.parse_args()

    print("=" * 78)
    print("DOCKING OBJECTIVE VALIDATION — size / lipophilicity artifact check")
    print("=" * 78)

    library = load_library(args.library_dir)
    print(f"Loaded library: {len(library['smiles'])} molecules from "
          f"{args.library_dir}.")

    # --- Build the docking-score sample (outputs -> cache -> fresh) ---
    print("\n" + "=" * 78)
    print("2. ASSEMBLING PfDHFR DOCKING SAMPLE")
    print("=" * 78)
    sample_df, prov = build_sample(library, args.n_sample, args.seed)
    print(f"\n  Docking-MEASUREMENT sample: {prov['n_total']} viable "
          "molecule(s) with descriptors —")
    print(f"    {prov['n_outputs']:>5} reused from existing run outputs"
          + (f" ({', '.join(prov['source_dirs'])})" if prov["source_dirs"] else "")
          )
    print(f"    {prov['n_cache']:>5} reused from the docking cache")
    print(f"    {prov['n_fresh']:>5} freshly docked this run (written to the cache)")

    # Which column each run output was read as — the pre-fix / post-fix
    # decision, stated rather than assumed.
    for directory, column in sorted(prov["measured_columns"].items()):
        generation = ("post-fix: measurement column, beside a possibly "
                      "SUBSTITUTED objective column"
                      if column == PFDHFR_RAW_COLUMN else
                      "pre-fix: no measurement column exists, so the objective "
                      "column IS the measurement")
        print(f"      {directory}/evaluated.csv -> read '{column}' "
              f"({generation})")

    # Viability exclusions, always printed — including the zero case, so a
    # reader can tell "no clashes in this sample" from "nobody checked".
    print(f"    {prov['n_excluded']:>5} EXCLUDED as non-viable "
          f"(evaluation.dock_is_viable: no binding mode found) — "
          f"{prov['n_excluded_outputs']} from outputs, "
          f"{prov['n_excluded_cache']} from the cache, "
          f"{prov['n_excluded_fresh']} fresh")
    if prov["excluded_raw_kcal"]:
        shown = ", ".join(f"{v:+.3f}" for v in prov["excluded_raw_kcal"][:10])
        more = ", ..." if len(prov["excluded_raw_kcal"]) > 10 else ""
        print(f"          excluded raw kcal/mol: {shown}{more}")
        print("          These are NOT weak binders: a score at or above "
              "0 kcal/mol means Vina found no")
        print("          favourable pose at all, so the value is not an "
              "affinity and cannot be a reference point.")

    if prov["n_total"] < 3:
        print("\n  Too few docking scores to analyze. Re-run with a larger "
              "--n-sample once docking is available (vina on PATH).")
        return

    # --- Correlations, LE, known actives ---
    corrs, flagged = report_correlations(sample_df)
    overlap_frac = report_ligand_efficiency(sample_df, args.top_k)
    active_rows, best_lib = report_known_actives(sample_df)

    # --- Figure ---
    save_figure(sample_df, active_rows, args.output)

    # --- 30-second verdict: RAW -> LE for every diagnostic ---
    print("\n" + "=" * 78)
    print("SUMMARY VERDICT (read this first):  RAW docking  ->  LIGAND EFFICIENCY")
    print("=" * 78)

    mw_raw = mw_le = logp_raw = logp_le = float("nan")
    if corrs:
        mw_raw, mw_le = corrs["MW"]["raw"][0], corrs["MW"]["le"][0]
        logp_raw, logp_le = corrs["logP"]["raw"][0], corrs["logP"]["le"][0]
        print(f"  - MW confound:    |r| {abs(mw_raw):.2f} ({_strength_word(mw_raw)})"
              f"  ->  {abs(mw_le):.2f} ({_strength_word(mw_le)})")
        print(f"  - logP confound:  |r| {abs(logp_raw):.2f} ({_strength_word(logp_raw)})"
              f"  ->  {abs(logp_le):.2f} ({_strength_word(logp_le)})")
    print(f"  - LE re-ranking:  top-{args.top_k} raw-vs-LE overlap "
          f"{overlap_frac * 100:.0f}% (low => raw score was size-driven)")

    raw_pcts = [r["percentile"] for r in active_rows if np.isfinite(r["percentile"])]
    le_pcts = [r["le_percentile"] for r in active_rows
               if np.isfinite(r["le_percentile"])]
    med_raw = float(np.median(raw_pcts)) if raw_pcts else float("nan")
    med_le = float(np.median(le_pcts)) if le_pcts else float("nan")
    if raw_pcts:
        print(f"  - known drugs:    median library percentile {med_raw:.0f}%  ->  "
              f"{med_le:.0f}%  (should RISE if size was the confound)")

    # Over-correction probe: are the LE leaders implausibly tiny fragments?
    le_series = sample_df[RAW_KCAL_COLUMN] / sample_df["heavy_atoms"]
    k = min(args.top_k, len(sample_df))
    le_top = sample_df.assign(_LE=le_series).sort_values("_LE").head(k)
    le_top_heavy = float(le_top["heavy_atoms"].median()) if k else float("nan")
    print(f"  - LE top-{k} size:  median {le_top_heavy:.0f} heavy atoms "
          f"(< {FRAGMENT_HEAVY_ATOMS} => LE favouring fragments)")

    # --- Decide the verdict, honestly ---
    # Two distinct LE failure modes, in priority order:
    #   (a) PATHOLOGICAL over-correction — LE-top are true fragments, or the known
    #       drugs actually FALL under LE (the task's explicit "watch the known-drug
    #       LE percentile" signal);
    #   (b) SIZE OVER-SWING — LE did not merely remove the size trend, it inverted
    #       it into a strong preference for smaller molecules (|LE-MW r| large).
    #       Here the drugs may still benefit and LE-top may still be drug-sized, so
    #       it is a caution (cap minimum size), not yet a failure.
    #   plus the residual-lipophilicity case that needs LLE.
    residual_logp = bool(corrs) and abs(logp_le) > CORR_MODERATE
    drugs_fell = bool(raw_pcts and le_pcts) and (med_le + 2 < med_raw)
    le_fragments = np.isfinite(le_top_heavy) and le_top_heavy < FRAGMENT_HEAVY_ATOMS
    size_over_swing = bool(corrs) and abs(mw_le) > CORR_STRONG

    if le_fragments or drugs_fell:
        print("\n  VERDICT: OVER-CORRECTION — LE rewards implausibly small molecules")
        print(f"  (LE-top median {le_top_heavy:.0f} heavy atoms; known drugs "
              f"{med_raw:.0f}% -> {med_le:.0f}% under LE). LE divides by size, so it")
        print("  has swung from a size confound to a fragment bias. Add a "
              "minimum-size floor")
        print("  (or use fit-quality / group efficiency) and re-check.")
        if residual_logp:
            print(f"  It ALSO leaves a lipophilicity confound (LE-logP |r|="
                  f"{abs(logp_le):.2f} > {CORR_MODERATE}); pair the size floor with "
                  "an LLE-style logP term.")
    elif residual_logp:
        print("\n  VERDICT: PARTIAL FIX — LE removed the SIZE confound "
              f"(MW |r| {abs(mw_raw):.2f} -> {abs(mw_le):.2f}) but a LIPOPHILICITY")
        print(f"  confound REMAINS (LE-logP |r|={abs(logp_le):.2f} > {CORR_MODERATE}). "
              "LE corrects size directly and")
        print("  logP only partly, so do NOT declare victory: escalate to a "
              "lipophilicity-aware")
        print("  correction (LLE = docking - c*logP, or LE with a logP penalty).")
    elif size_over_swing:
        print("\n  VERDICT: SIZE OVER-SWING — LE did NOT just remove the size trend, "
              "it INVERTED it")
        print(f"  (MW |r| {abs(mw_raw):.2f} -> {abs(mw_le):.2f}, now favouring "
              "smaller molecules). Here it HELPS:")
        print(f"  the known drugs rise {med_raw:.0f}% -> {med_le:.0f}% and the logP "
              f"confound is gone ({abs(logp_raw):.2f} -> {abs(logp_le):.2f}), and "
              f"LE-top are still")
        print(f"  drug-sized (median {le_top_heavy:.0f} heavy atoms). But LE is now a "
              "size-ANTIcorrelated objective:")
        print("  floor the minimum size and monitor LE-top so optimization does not "
              "drift into fragments.")
    else:
        print("\n  VERDICT: FIX VALIDATED — no residual size/lipophilicity confound "
              "strong enough to worry about")
        print(f"  (MW |r| {abs(mw_raw):.2f} -> {abs(mw_le):.2f}, logP |r| "
              f"{abs(logp_raw):.2f} -> {abs(logp_le):.2f}); known drugs "
              f"{med_raw:.0f}% -> {med_le:.0f}% and")
        print(f"  LE leaders are drug-sized (median {le_top_heavy:.0f} heavy atoms). "
              "Optimizing LE is sound.")
    print("\n  (Corroborate with the RAW-vs-LE panels in the figure and the "
          "percentiles above.)")


if __name__ == "__main__":
    main()
