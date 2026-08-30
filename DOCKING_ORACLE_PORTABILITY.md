# The docking oracle is machine-dependent

A limitation, with a correction to the obvious diagnosis and a concrete fix.
Discovered 2026-08-30 by Gate 0 of the ICM ablation; evidence in
`ablation_icm_vs_independent/ABORTED.md`.

## The observation

Ten molecules from the campaign's seed-0 MOGP run, re-docked on a second machine
against the "same" receptors at the same box, exhaustiveness and seed:

| | max \|Δ\| | mean \|Δ\| | over 0.05 kcal/mol |
|---|---|---|---|
| machine 2 vs machine 2 (repeat) | **0.000** | 0.000 | 0 / 20 |
| machine 2 vs Studio | **0.612** | 0.113 | 8 / 20 |

Within a machine the oracle is **bit-identical** — the ETKDGv3 conformer is
seeded (`randomSeed = 0xF00D`) and Vina's seed is fixed. Across machines the same
inputs give different numbers, up to 0.612 kcal/mol, which is comparable to a
real difference in binding quality. Signed Δ is −0.051 ± 0.198: scatter about
zero, not a constant offset.

Every tracked file agreed. Both `.stamp` files matched the Studio's committed
values.

## Correction: `oracle_fingerprint` is NOT the defect

The natural reading is that the fingerprint hashes metadata and should hash file
contents instead. **It already hashes contents.** `oracle_fingerprint` folds in
`receptor_fingerprint(target)`, which is `_sha256_file()` of the prepared PDBQT —
deliberately downstream of Open Babel so it catches added hydrogens and Gasteiger
charges. That design is correct and needs no change.

The blind component is `_prep_stamp`, which hashes only
`prep_version | cofactors | pdb_id`. But that is not the cache key — it is the
**rebuild trigger**, and being logic-only is what it is for: it fires when *our*
prep code changes.

The consequence is subtle and worth stating exactly:

> A toolchain change does not move the stamp, so an existing prepared receptor is
> **reused rather than rebuilt**. Its hash is therefore unchanged, and the cache
> stays valid — correctly, because the receptor on that machine really did not
> change. A machine that prepares the receptor *fresh* under a different Open
> Babel gets a different PDBQT, a different fingerprint, and a correct cache
> miss.

So both machines are internally consistent, and each is right about itself.
Nothing is stale. The two simply hold **different receptors**, and no mechanism
in the project ever compares one machine's oracle to another's.

## The actual gap: run outputs do not record their oracle

`history.csv`, `evaluated.csv` and `pareto_front.csv` carry no oracle identity.
Results and the oracle that produced them are stored apart, so nothing detects
two result sets being combined across incompatible oracles — which is exactly
what the ablation would have done had the gate not caught it. Gate 0 cost 20
docks to answer a question a stored 64-character string would have answered in
one comparison.

Leading hypothesis for the underlying difference, unconfirmed: a different Open
Babel build emitting different Gasteiger partial charges. `docking.py`'s own
docstring names it — *"an Open Babel version bump silently altering charges,
which changes every score without changing any file we control."* Machine 2 has
Open Babel 3.1.0 (Nov 30 2023), RDKit 2024.03.6, Vina 1.2.7.

## Fix

1. **Stamp results with their oracle.** Write `oracle_fingerprint` for every
   target into each run directory, and refuse to pool or compare two runs whose
   fingerprints differ. Cheap, and it converts this class of error from a
   20-dock investigation into an assertion.
2. **Publish the fingerprint in the environment record**, next to the library
   checksum and `evaluation_bounds.json`, so a second machine can verify
   comparability before spending compute rather than after.
3. **Do not** make `_prep_stamp` content-sensitive. It cannot be: it is computed
   to decide whether to build the file whose contents it would need to hash.
   Toolchain identity (Open Babel version) could reasonably join its payload,
   which would force a rebuild on upgrade — a real improvement, and a behaviour
   change that should be made deliberately rather than as a bug fix.

## Test coverage

`test_fingerprint_covers_every_score_determining_input` swept the box, seed,
exhaustiveness and target but **never the receptor's contents** — the one axis
that broke, and the same axis the NADPH bug moved. Two tests now close it
(`test_cache_invalidation.py`):

- `test_receptor_contents_participate_in_the_oracle_fingerprint` — stubs
  `receptor_fingerprint` to two values and asserts the oracle fingerprint moves.
  Drop the receptor term from the payload and every other test in the file still
  passes; this one fails.
- `test_prep_stamp_is_blind_to_receptor_contents` — pins the limitation itself,
  so the stamp is never mistaken for a content check.

## For the write-up

Report as a reproducibility limitation, phrased for what it is:

> Vina scores are reproducible to the bit within a machine but not across
> machines: identical inputs differed by up to 0.612 kcal/mol between two
> installations, while agreeing exactly on repeat within each. Docking results
> are therefore comparable only within a single prepared-receptor fingerprint,
> and all results reported here were produced under one.

This strengthens rather than weakens the campaign: all 30 campaign runs came from
one machine and one oracle, so every comparison in it is internally valid. It
does mean cross-machine pooling requires a fingerprint check first, and that
absolute kcal/mol values are less portable than the rankings built from them —
consistent with the existing caveat that Vina scores are rankings, not affinities.
