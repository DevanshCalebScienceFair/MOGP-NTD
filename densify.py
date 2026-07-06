"""
densify.py
==========

RDKit-only analog generator for **library densification**.

The BO loop searches a *fixed* cached library, so no acquisition function can
ever propose a molecule the initial ChEMBL pull did not contain — the Pareto
front is capped by what that pull happened to include. This module grows the
candidate pool on the fly by enumerating novel analogs of the current Pareto-
front molecules, densifying the neighborhood of the best trade-offs where front
expansion tends to hide.

Two RDKit-only operators (no heavy new dependency), both deterministic:

  * **BRICS recombination.** Decompose the front molecules into BRICS fragments
    and recombine them (``BRICS.BRICSBuild`` with ``scrambleReagents=False`` for
    reproducibility). This explores larger scaffold hops around the front.
  * **Reaction-SMARTS mutations.** A small fixed set of valence-safe local edits
    (halogen swaps, aromatic C-H -> substituent, acid <-> amide/ester, etc.)
    applied per parent. These stay close to the parent, so they reliably pass the
    downstream drug-likeness filter.

``generate_analogs`` canonicalizes every candidate (RDKit), drops anything that
duplicates a parent or an ``exclude`` set (the existing library + already-
evaluated molecules), deduplicates, and returns a deterministic, seed-shuffled
list. It does NOT filter for drug-likeness / ADMET / applicability domain — that
is done by ``data.process_smiles`` on the way into the library, exactly as for
the base library, so injected analogs are indistinguishable downstream.
"""

import random

from rdkit import Chem
from rdkit.Chem import AllChem, BRICS
from rdkit import RDLogger


# Silence RDKit's per-molecule sanitize/parse chatter; we handle failures by
# skipping the offending candidate.
RDLogger.DisableLog("rdApp.*")


# Valence-safe local edits, as atom-mapped reaction SMARTS. Each is applied to
# every matching site of a parent; RDKit adjusts implicit Hs. Anything that
# fails to sanitize is skipped, so an over-eager rule simply yields fewer
# analogs rather than a crash.
_MUTATION_SMARTS = [
    # Halogen walk.
    "[F:1]>>[Cl:1]",
    "[Cl:1]>>[F:1]",
    "[Cl:1]>>[Br:1]",
    "[Br:1]>>[Cl:1]",
    "[F:1]>>[Br:1]",
    # Decorate an aromatic C-H with a small substituent.
    "[cH:1]>>[c:1]C",
    "[cH:1]>>[c:1]F",
    "[cH:1]>>[c:1]Cl",
    "[cH:1]>>[c:1]O",
    "[cH:1]>>[c:1]N",
    # Grow a methyl to an ethyl.
    "[CH3:1]>>[CH2:1]C",
    # Carboxylic acid bioisosteres / derivatives.
    "[CX3:1](=[O:2])[OX2H1]>>[C:1](=[O:2])[NH2]",
    "[CX3:1](=[O:2])[OX2H1]>>[C:1](=[O:2])OC",
    # Hydroxyl <-> primary amine / methyl ether.
    "[OX2H1:1]>>[NH2:1]",
    "[OX2H1:1]>>[O:1]C",
]

_MUTATIONS = [AllChem.ReactionFromSmarts(s) for s in _MUTATION_SMARTS]

# Bound the combinatorics so a large front cannot blow up BRICS: cap the fragment
# pool and the recombination depth. Both keep generation fast and deterministic.
_MAX_BRICS_FRAGMENTS = 40
_BRICS_MAX_DEPTH = 2


def canonical_smiles(smiles):
    """Return the RDKit canonical SMILES, or ``None`` if RDKit cannot parse it."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def _sanitized_smiles(mol):
    """Canonical SMILES for a freshly built/mutated mol, or None if invalid."""
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    smi = Chem.MolToSmiles(mol)
    return smi or None


def _mutate(mol):
    """All valence-safe single-edit products of ``mol`` (canonical SMILES)."""
    outs = []
    for rxn in _MUTATIONS:
        try:
            product_sets = rxn.RunReactants((mol,))
        except Exception:
            continue
        for products in product_sets:
            for product in products:
                smi = _sanitized_smiles(product)
                if smi is not None:
                    outs.append(smi)
    return outs


def _brics_recombine(mols, max_build):
    """Recombine BRICS fragments of ``mols`` into up to ``max_build`` molecules.

    Deterministic: fragments are sorted and ``scrambleReagents=False`` so the
    build order is fixed for a given input.
    """
    fragments = set()
    for mol in mols:
        try:
            fragments |= BRICS.BRICSDecompose(mol)
        except Exception:
            continue
    if not fragments:
        return []

    frag_mols = [Chem.MolFromSmiles(f) for f in sorted(fragments)]
    frag_mols = [f for f in frag_mols if f is not None][:_MAX_BRICS_FRAGMENTS]
    if not frag_mols:
        return []

    outs = []
    try:
        builder = BRICS.BRICSBuild(
            frag_mols, onlyCompleteMols=True,
            scrambleReagents=False, maxDepth=_BRICS_MAX_DEPTH,
        )
        for i, mol in enumerate(builder):
            if i >= max_build:
                break
            smi = _sanitized_smiles(mol)
            if smi is not None:
                outs.append(smi)
    except Exception:
        pass
    return outs


def generate_analogs(parent_smiles, n_per_parent=20, seed=0, exclude=frozenset()):
    """Generate novel canonical analog SMILES around a set of parent molecules.

    Args:
        parent_smiles: SMILES of the current Pareto-front molecules.
        n_per_parent: Target analogs per parent; the returned list is capped at
            ``n_per_parent * (#parseable parents)``.
        seed: Seed for the final deterministic shuffle (so repeated calls with
            the same parents+seed return the same list, and different seeds pick
            different subsets of the candidate pool).
        exclude: Canonical SMILES to treat as "already known" — typically the
            existing library plus already-evaluated molecules. Parents are always
            excluded too, so no analog duplicates its own parent.

    Returns:
        A deterministic list of novel canonical SMILES (never containing a parent
        or an ``exclude`` member, deduplicated). Empty if no parents parse or no
        novel analog could be built. Drug-likeness / ADMET / domain filtering is
        intentionally NOT done here — ``data.process_smiles`` handles it on
        injection, identically to the base library.
    """
    parent_mols = []
    parent_canonical = set()
    for smiles in parent_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        parent_mols.append(mol)
        parent_canonical.add(Chem.MolToSmiles(mol))

    if not parent_mols:
        return []

    target_total = max(1, n_per_parent) * len(parent_mols)
    exclude_set = set(exclude) | parent_canonical

    # Deterministic candidate order: local mutations per parent, then a single
    # global BRICS recombination over the whole front's fragment pool.
    candidates = []
    for mol in parent_mols:
        candidates.extend(_mutate(mol))
    candidates.extend(_brics_recombine(parent_mols, max_build=target_total * 3))

    # Canonicalize, drop invalid / known / duplicate, preserving first-seen order.
    seen = set()
    novel = []
    for smiles in candidates:
        canon = canonical_smiles(smiles)
        if canon is None or canon in exclude_set or canon in seen:
            continue
        seen.add(canon)
        novel.append(canon)

    # Sort for an order independent of generation, then a seeded shuffle so the
    # cap doesn't always keep the alphabetically-first analogs. Deterministic
    # for a given (parents, seed).
    novel.sort()
    random.Random(seed).shuffle(novel)
    return novel[:target_total]


if __name__ == "__main__":
    # Self-check: generate analogs for a couple of drug-like parents, and confirm
    # novelty, exclusion, and determinism.
    parents = [
        "CCOc1ccc(cc1)C(=O)O",            # an aryl ether acid
        "c1ccc(cc1)CC(=O)Nc1ccccc1",      # a phenylacetamide
    ]
    a = generate_analogs(parents, n_per_parent=15, seed=0)
    b = generate_analogs(parents, n_per_parent=15, seed=0)
    c = generate_analogs(parents, n_per_parent=15, seed=1)

    parent_canon = {canonical_smiles(s) for s in parents}
    print(f"parents: {len(parents)}   analogs (seed 0): {len(a)}")
    print(f"deterministic (seed 0 == seed 0): {a == b}")
    print(f"seed 1 differs from seed 0:        {a != c}")
    print(f"no parent leaked into analogs:     {parent_canon.isdisjoint(a)}")
    print(f"all canonical & unique:            "
          f"{len(a) == len(set(a)) and all(canonical_smiles(s) == s for s in a)}")

    excl = set(a[:5])
    d = generate_analogs(parents, n_per_parent=15, seed=0, exclude=excl)
    print(f"exclude respected:                 {excl.isdisjoint(d)}")
    print("Sample analogs:")
    for s in a[:5]:
        print(f"  {s}")
