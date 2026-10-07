"""Own-code numerical core for MOGP-NTD.

Pure numpy (plus the Python standard library) reimplementations of the
numerical machinery this project previously took from torch / gpytorch /
botorch / scipy / scikit-learn.

Two different properties are tracked here, and only one of them is finished.

Dependency purity -- achieved.  Nothing in this package imports torch,
gpytorch, botorch, scipy or sklearn, and none of them is in its import closure.
Test modules may import them as differential oracles; the library code may not.

Provenance -- not finished.  The modules were built as behavioural mirrors of
what they replace, library quirks included, so that replaying an archived run
would reproduce the archived numbers.  The project owner has since waived that
archive-compatibility goal: the archived results are superseded and agreeing
with the libraries bit for bit is no longer a requirement.  A module that is
pure numpy can therefore still contain a choice (a constant, an initialization,
a draw order, a formula arranged the way a library arranges it) that came from a
library rather than from published mathematics.  The origin of each such choice
is tracked in PROVENANCE.md in this directory, and the ones that came from a
library are being re-derived from the published math.  Where a module header or
docstring still calls the code a "mirror", read that as a record of how it was
built, not of the standard it is held to now.

The *_GATE.md and BOXDECOMP_VERDICT.md files beside this one are historical
records of the parity-with-the-libraries acceptance criterion; see the banner at
the top of each.  Derivations and measured agreement figures are in
../spec/<layer>/SPEC.md.
"""
