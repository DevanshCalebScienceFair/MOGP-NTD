"""
data.py
=======

Build and cache the molecule search library for downstream Bayesian
optimization.

This module precomputes the CHEAP per-molecule quantities for the entire
library upfront: Morgan fingerprints and low-fidelity ADMET scores. The
EXPENSIVE quantity (docking) is deliberately NOT computed here — docking only
happens inside loop.py when the EHVI acquisition function selects specific
molecules to evaluate.

Pipeline:
    pull_molecules()    pull drug-like SMILES from ChEMBL via TDC
    filter_druglike()   keep molecules passing Lipinski's Rule of Five
    build_library()     featurize + ADMET-score + drop out-of-domain, then cache
    load_library()      reload the cached library aligned across all three files

The three cached files (smiles.csv, fingerprints.npy, admet_scores.csv) are
row-aligned: row i refers to the same molecule in every file.

Run as a script to (re)build and sanity-check the library:
    python data.py --n-molecules 10000
"""

import os
import argparse
from collections import namedtuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen

from tdc.generation import MolGen

from utils.featurize import batch_smiles_to_morgan
from admet_oracle import ADMETOracle
import quality_filter


# Order of the ADMET columns in the cached score matrix. Kept as a module-level
# constant so build_library and load_library cannot drift out of sync.
ADMET_COLUMNS = ["Caco2_logPapp", "Half_Life_hours", "hERG_Toxicity_Prob"]

# Fragment ceiling for the SHARED candidate library. The optimized docking
# objective is ligand efficiency (raw docking / heavy-atom count), which
# mechanically rewards SMALL molecules, so without a floor the Pareto front can
# drift into fragments over a full run. `load_library` drops every molecule below
# this many heavy atoms ONCE, in the shared load path, so the MOGP loop and all
# baselines see the SAME floored library (never filtered per-method).
#
# This is a principled drug-likeness floor, NOT a tunable knob: it is set safely
# below the smallest known clinical antifolate (Pyrimethamine/Cycloguanil, 17
# heavy atoms), and `load_library` ASSERTS that no KNOWN_ACTIVE is ever excluded.
HEAVY_ATOM_FLOOR = 14

# Pareto size-drift warning threshold (one above the floor): if a run's final
# Pareto median heavy-atom count slips below this, the front is approaching the
# fragment floor and the LE objective may be over-corrected toward tiny molecules.
FRAGMENT_MEDIAN_WARN = HEAVY_ATOM_FLOOR + 1

# The flag columns the ADMET oracle emits alongside its predictions; a molecule
# is dropped from the library if ANY of these is True (see process_smiles).
ADMET_FLAG_COLUMNS = [
    "Featurization_Failed",
    "Caco2_OutOfDomain",
    "Half_Life_OutOfDomain",
    "hERG_OutOfDomain",
]

# Row-aligned survivors of the per-molecule library pipeline (process_smiles).
# The three arrays/frames are aligned: row i is the same molecule in each.
#   smiles        list[str]
#   fingerprints  (M, 2048) int8
#   admet_df      DataFrame, columns ["SMILES"] + ADMET_COLUMNS (== admet_scores.csv)
#   n_input/n_druglike/n_featurized/n_final  per-stage counts (for logging)
ProcessedMolecules = namedtuple(
    "ProcessedMolecules",
    ["smiles", "fingerprints", "admet_df",
     "n_input", "n_druglike", "n_featurized", "n_final"],
)

# Print an ADMET-scoring progress line every this many molecules.
ADMET_PROGRESS_EVERY = 1000


def pull_molecules(n_molecules=10000):
    """Pull a fixed, shuffled sample of drug-like molecules from ChEMBL.

    Uses TDC's MolGen ChEMBL_V29 generation dataset. The shuffle uses a fixed
    random seed (42) so the same call always returns the same molecules.

    Args:
        n_molecules: Number of SMILES to return.

    Returns:
        A list of SMILES strings (length up to ``n_molecules``).
    """
    data = MolGen(name="ChEMBL_V29")
    df = data.get_data()

    shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    smiles_list = shuffled["smiles"].head(n_molecules).tolist()
    print(f"Pulled {len(smiles_list)} molecules from ChEMBL_V29.")
    return smiles_list


def filter_druglike(smiles_list):
    """Filter SMILES to those passing Lipinski's Rule of Five.

    Criteria (all must hold):
        200 <= molecular weight <= 600
        -2 <= Crippen logP <= 5
        hydrogen bond donors <= 5
        hydrogen bond acceptors <= 10

    SMILES that RDKit cannot parse are skipped.

    Args:
        smiles_list: An iterable of SMILES strings.

    Returns:
        The filtered list of SMILES strings (in input order).
    """
    passed = []
    n_unparseable = 0

    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            n_unparseable += 1
            continue

        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        h_donors = Descriptors.NumHDonors(mol)
        h_acceptors = Descriptors.NumHAcceptors(mol)

        if (200 <= mw <= 600
                and -2 <= logp <= 5
                and h_donors <= 5
                and h_acceptors <= 10):
            passed.append(smiles)

    n_total = len(smiles_list)
    n_passed = len(passed)
    n_filtered = n_total - n_passed
    print(
        f"Drug-likeness filter: {n_passed} passed, {n_filtered} filtered out "
        f"(of {n_total}; {n_unparseable} unparseable)."
    )
    return passed


def process_smiles(smiles_list, oracle=None):
    """Run the per-molecule library pipeline and return row-aligned survivors.

    This is the SINGLE per-molecule processing path shared by ``build_library``
    (the base ChEMBL library) and the on-line densification path in ``loop.py``,
    so a molecule injected mid-run is processed identically to one from the base
    library — same drug-likeness filter, same Morgan featurization, same ADMET
    scoring, same applicability-domain / NaN drop. Reusing one function is what
    guarantees the two can never drift.

    Steps (each drops molecules that fail it):
        1. Lipinski drug-likeness filter (``filter_druglike``).
        2. Morgan featurization (``batch_smiles_to_morgan``).
        3. ADMET scoring via ``ADMETOracle`` (chunked, with progress).
        4. Applicability-domain / NaN drop: keep only rows that featurized, are
           in-domain for EVERY ADMET model, and have no missing ADMET value.

    Args:
        smiles_list: Candidate SMILES strings.
        oracle: Optional pre-loaded ``ADMETOracle`` (reused across densification
            iterations so the three models are not re-read from disk each call).
            A fresh oracle is constructed when ``None``.

    Returns:
        A ``ProcessedMolecules`` namedtuple whose ``smiles`` (list),
        ``fingerprints`` ((M, 2048) int8) and ``admet_df`` (columns
        ``["SMILES"] + ADMET_COLUMNS``) are row-aligned survivors, plus the
        per-stage counts.
    """
    smiles_list = list(smiles_list)
    n_input = len(smiles_list)

    # --- Drug-likeness filter (cheap) ----------------------------------------
    filtered = filter_druglike(smiles_list)
    n_druglike = len(filtered)

    # --- Fingerprints (cheap) ------------------------------------------------
    # batch_smiles_to_morgan drops anything it cannot featurize and returns the
    # surviving SMILES in input order, so fingerprints and valid_smiles align.
    fingerprints, valid_smiles = batch_smiles_to_morgan(filtered)
    n_featurized = len(valid_smiles)
    print(f"Featurization: {n_featurized} molecules produced fingerprints.")

    # --- ADMET scoring (the slow part; ~minutes for 10k) ---------------------
    if oracle is None:
        oracle = ADMETOracle()

    if n_featurized:
        print(f"Scoring {n_featurized} molecules through the ADMET oracle...")
        admet_frames = []
        for start in range(0, n_featurized, ADMET_PROGRESS_EVERY):
            chunk = valid_smiles[start:start + ADMET_PROGRESS_EVERY]
            admet_frames.append(oracle.predict(chunk))
            done = min(start + ADMET_PROGRESS_EVERY, n_featurized)
            print(f"  ADMET scored {done}/{n_featurized}")
        admet_df = pd.concat(admet_frames, ignore_index=True)
    else:
        # An empty predict() yields a correctly-columned empty frame, so the
        # domain/NaN masks below are well-defined without a hard-coded schema.
        admet_df = oracle.predict([])

    # --- Domain / NaN drop ---------------------------------------------------
    # The mask indexes both admet_df and fingerprints, which are still
    # row-aligned with valid_smiles at this point.
    flagged = admet_df[ADMET_FLAG_COLUMNS].to_numpy(dtype=bool).any(axis=1)
    in_domain = ~flagged
    no_nan = admet_df[ADMET_COLUMNS].notna().all(axis=1).to_numpy(dtype=bool)
    keep_mask = in_domain & no_nan

    final_smiles = [s for s, k in zip(valid_smiles, keep_mask) if k]
    final_fingerprints = fingerprints[keep_mask].astype(np.int8)
    final_admet = admet_df.loc[keep_mask, ["SMILES"] + ADMET_COLUMNS]
    final_admet = final_admet.reset_index(drop=True)
    n_final = len(final_smiles)

    return ProcessedMolecules(
        smiles=final_smiles,
        fingerprints=final_fingerprints,
        admet_df=final_admet,
        n_input=n_input,
        n_druglike=n_druglike,
        n_featurized=n_featurized,
        n_final=n_final,
    )


def build_library(n_molecules=10000, output_dir="data/library"):
    """Build the molecule library and cache it to ``output_dir``.

    Pulls molecules, then runs the shared ``process_smiles`` pipeline (drug-
    likeness filter, Morgan fingerprints, ADMET scoring, out-of-domain / NaN
    drop) and writes three row-aligned files:

        smiles.csv         one column "SMILES", one row per molecule
        fingerprints.npy   np.ndarray shape (N, 2048) int8
        admet_scores.csv   columns: SMILES, Caco2_logPapp, Half_Life_hours,
                           hERG_Toxicity_Prob

    Args:
        n_molecules: Number of molecules to pull before filtering.
        output_dir: Directory to write the cached library into (created if
            missing).

    Returns:
        The ``output_dir`` path.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Pull (cheap) --------------------------------------------------------
    pulled = pull_molecules(n_molecules)
    n_pulled = len(pulled)

    # --- Shared per-molecule pipeline (filter -> featurize -> ADMET -> drop) --
    processed = process_smiles(pulled)

    # --- Persist (all three files row-aligned) -------------------------------
    pd.DataFrame({"SMILES": processed.smiles}).to_csv(
        os.path.join(output_dir, "smiles.csv"), index=False
    )
    np.save(
        os.path.join(output_dir, "fingerprints.npy"),
        processed.fingerprints.astype(np.int8),
    )
    processed.admet_df.to_csv(
        os.path.join(output_dir, "admet_scores.csv"), index=False
    )

    # --- Summary -------------------------------------------------------------
    print("\n=== Library build summary ===")
    print(f"  Total pulled:              {n_pulled}")
    print(f"  Passed drug-likeness:      {processed.n_druglike}")
    print(f"  Passed featurization:      {processed.n_featurized}")
    print(f"  Passed ADMET domain check: {processed.n_final}")
    print(f"  Final library size:        {processed.n_final}")
    print(f"  Saved to:                  {output_dir}")
    return output_dir


def heavy_atom_count(smiles):
    """Heavy-atom count for a SMILES, or None if RDKit cannot parse it.

    ``Chem.MolFromSmiles(smiles).GetNumHeavyAtoms()`` — the SAME definition
    ``docking.raw_to_ligand_efficiency`` uses for the LE denominator and
    ``validate_docking.py`` uses for its diagnostics, so heavy-atom counts are
    consistent across the floor, the objective, and the size-drift monitor.
    """
    mol = Chem.MolFromSmiles(smiles)
    return None if mol is None else int(mol.GetNumHeavyAtoms())


def heavy_atom_stats(smiles_list):
    """Return ``(median, min)`` heavy-atom count over a list of SMILES.

    Unparseable SMILES are skipped; ``(nan, nan)`` if none parse (e.g. an empty
    Pareto front). Used as the per-iteration size-drift monitor in the loop and
    baselines.
    """
    counts = [c for c in (heavy_atom_count(s) for s in smiles_list) if c is not None]
    if not counts:
        return float("nan"), float("nan")
    return float(np.median(counts)), float(np.min(counts))


def pareto_heavy_summary(smiles_list, warn_below=FRAGMENT_MEDIAN_WARN):
    """Summarize a Pareto front's heavy-atom distribution and flag fragment drift.

    Returns ``(summary_line, median, flagged)``. With the LE objective rewarding
    small molecules, a front whose MEDIAN heavy-atom count slips below
    ``warn_below`` is approaching the fragment floor; ``flagged`` is True then.
    """
    counts = [c for c in (heavy_atom_count(s) for s in smiles_list) if c is not None]
    if not counts:
        return "Final Pareto heavy atoms: (empty front)", float("nan"), False
    arr = np.asarray(counts)
    med = float(np.median(arr))
    line = (f"Final Pareto heavy atoms: n={arr.size}, min {int(arr.min())}, "
            f"median {med:.0f}, max {int(arr.max())}; "
            f"{int((arr < warn_below).sum())} below {warn_below}")
    return line, med, med < warn_below


def _assert_known_actives_survive_floor(floor):
    """Assert none of the four KNOWN_ACTIVES would be excluded by ``floor``.

    The heavy-atom floor is a principled drug-likeness ceiling, not a knob to tune
    results, so it must NEVER cut a real clinical antifolate (smallest = 17 heavy
    atoms). Fails loudly if the floor is ever raised past one. Imported lazily so
    ``data`` carries no import-time dependency on the validation script.
    """
    from validate_known_actives import KNOWN_ACTIVES

    counts = {a["name"]: heavy_atom_count(a["smiles"]) for a in KNOWN_ACTIVES}
    cut = {name: hc for name, hc in counts.items() if hc is None or hc < floor}
    smallest = min(hc for hc in counts.values() if hc is not None)
    assert not cut, (
        f"Heavy-atom floor {floor} would EXCLUDE known clinical antifolate(s) "
        f"{cut} (heavy-atom counts {counts}). The floor is a principled "
        f"drug-likeness ceiling below the smallest known active ({smallest} heavy "
        "atoms), NOT a tunable knob — lower it; do not exclude a real drug."
    )
    return counts, smallest


def _apply_heavy_atom_floor(smiles, fingerprints, admet_scores, floor):
    """Filter the cached library to molecules with >= ``floor`` heavy atoms.

    Applied ONCE in ``load_library`` so every method sees the same floored library.
    Operates on the EXISTING cached arrays (no ChEMBL rebuild) and never touches
    the docking cache — the cache is keyed by SMILES, so removed molecules are
    simply never queried and every cached score stays valid. The three arrays are
    filtered by one shared mask, so they stay row-aligned.
    """
    counts, smallest = _assert_known_actives_survive_floor(floor)

    keep = np.zeros(len(smiles), dtype=bool)
    kept_heavy, kept_mw = [], []
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue                       # unparseable -> drop (defensive)
        h = mol.GetNumHeavyAtoms()
        if h >= floor:
            keep[i] = True
            kept_heavy.append(h)
            kept_mw.append(float(Descriptors.MolWt(mol)))

    n_before, n_after = len(smiles), int(keep.sum())
    kept_smiles = [s for s, k in zip(smiles, keep) if k]
    if kept_heavy:
        kh, km = np.asarray(kept_heavy), np.asarray(kept_mw)
        ranges = (f"surviving heavy atoms [{int(kh.min())}, {int(kh.max())}], "
                  f"MW [{km.min():.1f}, {km.max():.1f}]")
    else:
        ranges = "no survivors"
    print(f"load_library: heavy-atom floor >= {floor}: {n_after}/{n_before} "
          f"molecules survive ({n_before - n_after} fragment(s) removed); {ranges}. "
          f"Known actives retained (smallest {smallest} heavy atoms).")

    return kept_smiles, fingerprints[keep], admet_scores[keep]


def _apply_quality_filter(smiles, fingerprints, admet_scores):
    """Drop molecules matching PAINS or failing the synthesizability screen.

    Applied ONCE in ``load_library``, immediately after the heavy-atom floor and
    on the same three row-aligned arrays, so the MOGP loop and every baseline
    see the SAME filtered candidate pool (never per-method). The same
    ``quality_filter.passes_quality`` gate is also applied to densify-generated
    analogs (``loop.BOLoop._densify``) so injected molecules cannot bypass a
    filter the base library passes through. Never touches the docking cache
    (keyed by SMILES) -- dropped molecules are simply never queried.
    """
    # Fail loudly here too if the thresholds ever shift enough to exclude a
    # known active -- the base library must never quietly discard a real drug.
    quality_filter.assert_known_actives_survive()

    keep = []
    pains_drops = 0
    synth_drops = 0
    unparseable = 0
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            unparseable += 1
            keep.append(False)
            continue
        ok, reason = quality_filter.passes_quality(mol)
        if ok:
            keep.append(True)
            continue
        keep.append(False)
        if reason and reason.startswith("PAINS"):
            pains_drops += 1
        else:
            synth_drops += 1

    keep = np.asarray(keep, dtype=bool)
    n_before, n_after = len(smiles), int(keep.sum())
    kept_smiles = [s for s, k in zip(smiles, keep) if k]
    print(
        f"load_library: quality screen ({quality_filter.ACTIVE_SYNTH_METRIC}): "
        f"{n_after}/{n_before} molecules survive "
        f"({pains_drops} PAINS + {synth_drops} synth + {unparseable} unparseable "
        f"removed). Known actives asserted to survive."
    )
    return kept_smiles, fingerprints[keep], admet_scores[keep]


def load_library(library_dir="data/library", heavy_atom_floor=HEAVY_ATOM_FLOOR):
    """Load the cached molecule library from disk, applying the shared size floor.

    A drug-likeness heavy-atom floor (``heavy_atom_floor``, default
    ``HEAVY_ATOM_FLOOR`` = 14) is applied ONCE here, in the shared load path, so
    the MOGP loop and every baseline see the SAME floored candidate library (never
    filtered per-method). Filtering runs on the EXISTING cached files — it does
    NOT rebuild from ChEMBL and does NOT touch the docking cache (keyed by SMILES;
    removed molecules are simply never docked). Pass ``heavy_atom_floor=None`` to
    load the raw cached library unfiltered.

    Args:
        library_dir: Directory containing smiles.csv, fingerprints.npy, and
            admet_scores.csv.
        heavy_atom_floor: Minimum heavy-atom count to keep; ``None``/0 disables
            the floor. Asserts no KNOWN_ACTIVE is excluded (see
            ``_assert_known_actives_survive_floor``).

    Returns:
        A dict with keys:
            "smiles":       list of SMILES strings
            "fingerprints": np.ndarray shape (N, 2048) int8
            "admet_scores": np.ndarray shape (N, 3) float32, columns in order
                            [Caco2_logPapp, Half_Life_hours, hERG_Toxicity_Prob]
        with the three arrays row-aligned after the floor.

    Raises:
        FileNotFoundError: If any of the three library files is missing.
    """
    smiles_path = os.path.join(library_dir, "smiles.csv")
    fingerprints_path = os.path.join(library_dir, "fingerprints.npy")
    admet_path = os.path.join(library_dir, "admet_scores.csv")

    for path in (smiles_path, fingerprints_path, admet_path):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Library file not found: {path}. "
                "Build the library first by running: python data.py"
            )

    smiles = pd.read_csv(smiles_path)["SMILES"].tolist()
    fingerprints = np.load(fingerprints_path).astype(np.int8)
    admet_df = pd.read_csv(admet_path)
    admet_scores = admet_df[ADMET_COLUMNS].to_numpy(dtype=np.float32)

    # Shared, once-only fragment floor (fair across every method).
    if heavy_atom_floor:
        smiles, fingerprints, admet_scores = _apply_heavy_atom_floor(
            smiles, fingerprints, admet_scores, heavy_atom_floor
        )

    # Shared, once-only quality gate: PAINS + synthesizability. Applied here so
    # every method (MOGP + all baselines) searches the same filtered pool. The
    # same gate is applied to densify-generated analogs in loop.BOLoop._densify.
    smiles, fingerprints, admet_scores = _apply_quality_filter(
        smiles, fingerprints, admet_scores
    )

    return {
        "smiles": smiles,
        "fingerprints": fingerprints,
        "admet_scores": admet_scores,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build and cache the molecule search library."
    )
    parser.add_argument(
        "--n-molecules",
        type=int,
        default=10000,
        help="Number of molecules to pull from ChEMBL before filtering.",
    )
    args = parser.parse_args()

    build_library(n_molecules=args.n_molecules)

    library = load_library()
    smiles = library["smiles"]
    fingerprints = library["fingerprints"]
    admet_scores = library["admet_scores"]

    print("\n=== Loaded library sanity check ===")
    print(f"  Library size:           {len(smiles)}")
    print(f"  Fingerprint matrix:     {fingerprints.shape}")
    print(f"  ADMET score matrix:     {admet_scores.shape}")
    print("  First 5 SMILES:")
    for s in smiles[:5]:
        print(f"    {s}")

    if admet_scores.shape[0] > 0:
        means = admet_scores.mean(axis=0)
        stds = admet_scores.std(axis=0)
        print("  ADMET column stats (mean +/- std):")
        for name, mean, std in zip(ADMET_COLUMNS, means, stds):
            print(f"    {name}: {mean:.4f} +/- {std:.4f}")
