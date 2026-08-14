"""
sa_score.py
===========

Crash-safe access to the RDKit contrib **synthetic accessibility (SA)** score.

Why this module exists
----------------------
``rdkit/Contrib/SA_Score/sascorer.py`` scores fragments with the *deprecated*
``rdMolDescriptors.GetMorganFingerprint``. On some builds (reproduced here:
RDKit 2024.03.6, osx-arm64, Python 3.11) that call does not merely warn — it
kills the interpreter with **SIGBUS** (exit 138) immediately after emitting
``DEPRECATION WARNING: please use MorganGenerator``.

That failure mode is uniquely nasty for a filter:

* ``import sascorer`` *succeeds*, so an import-time ``try/except`` sees nothing
  wrong and reports SA as the active metric.
* The crash happens on the first ``calculateScore`` call, and **SIGBUS cannot be
  caught** — no ``except`` clause runs, no fallback fires. The process dies.
* On a machine where the deprecated call still works, everything behaves. So the
  same commit silently screens differently on two machines.

The fix is to stop calling the deprecated API at all. ``_GeneratorShim`` below
substitutes ``rdFingerprintGenerator.GetMorganGenerator(...)`` for that one
function and proxies every other attribute straight through to the real
``rdMolDescriptors``. Upstream's scoring arithmetic, fragment-score table and
thresholds are untouched — only the fingerprint call site changes.

The substitution is **exact, not approximate**: the generator's sparse count
fingerprint produces the same fragment hashes as the deprecated function, which
matters because ``sascorer``'s ``_fscores`` table is keyed by those hashes. If
they disagreed, every fragment would miss the table and silently take the -4
default penalty. ``test_sa_score.py`` pins this against the 100 published
reference scores shipped in the contrib's ``data/zim.100.txt``.

Failing loudly
--------------
If the patched scorer still cannot run, this module raises rather than quietly
degrading to a weaker metric. A silent fallback would mean the synthesizability
screen stops doing anything on one machine while continuing to work on another
— the same class of invisible divergence as the NADPH bug. Set
``MOGP_ALLOW_QED_FALLBACK=1`` to opt into the degraded path deliberately; it
announces itself on every import.

Because SIGBUS cannot be caught in-process, availability is established with a
**subprocess probe** at import: a child interpreter scores one molecule, and we
read its exit code. That costs one short subprocess per process start.
"""

import os
import subprocess
import sys

SA_THRESHOLD_DEFAULT = 6.0

# Set in the probe child so it does not probe itself (which would recurse).
_PROBE_ENV = "_MOGP_SA_PROBE"

# Opt-in escape hatch for environments where the SA score genuinely cannot run.
_FALLBACK_ENV = "MOGP_ALLOW_QED_FALLBACK"


class _GeneratorShim:
    """``rdMolDescriptors`` with the one crashing function replaced.

    ``sascorer`` calls ``rdMolDescriptors.GetMorganFingerprint`` for fragment
    hashes and ``CalcNumSpiroAtoms`` / ``CalcNumBridgeheadAtoms`` for the
    feature term. Only the first is replaced; ``__getattr__`` forwards the rest
    to the genuine module, so this stays a targeted substitution rather than a
    fork of upstream behaviour.
    """

    def __init__(self, real_module):
        self._real = real_module
        from rdkit.Chem import rdFingerprintGenerator
        self._rfg = rdFingerprintGenerator
        self._cache = {}

    def GetMorganFingerprint(self, mol, radius, **kwargs):   # noqa: N802
        """Sparse count Morgan fingerprint via the modern generator.

        Same fragment hashes as the deprecated call, so ``sascorer._fscores``
        lookups still hit. Generators are cached per radius because building one
        per molecule dominates the runtime of a library-wide screen.
        """
        gen = self._cache.get(radius)
        if gen is None:
            gen = self._rfg.GetMorganGenerator(radius=radius)
            self._cache[radius] = gen
        return gen.GetSparseCountFingerprint(mol)

    def __getattr__(self, name):
        return getattr(self._real, name)


def _load_patched_sascorer():
    """Import the contrib ``sascorer`` and patch its fingerprint call site."""
    from rdkit.Chem import RDConfig
    sa_dir = os.path.join(RDConfig.RDContribDir, "SA_Score")
    if not os.path.isdir(sa_dir):
        raise ImportError(f"SA_Score contrib directory not found at {sa_dir}")
    if sa_dir not in sys.path:
        sys.path.append(sa_dir)
    import sascorer  # noqa: WPS433 — contrib lives outside the package tree

    # Swap the module-level name sascorer.calculateScore resolves against.
    if not isinstance(sascorer.rdMolDescriptors, _GeneratorShim):
        sascorer.rdMolDescriptors = _GeneratorShim(sascorer.rdMolDescriptors)
    return sascorer


def _probe_in_subprocess():
    """True if a child interpreter can score a molecule without dying.

    SIGBUS terminates the process, so the only way to test survivability
    without risking this one is to spend a child. The child imports this module
    with ``_PROBE_ENV`` set, which suppresses its own probe.
    """
    code = (
        "import sa_score, sys;"
        "from rdkit import Chem;"
        "m = Chem.MolFromSmiles('CCO');"
        "s = sa_score._SASCORER.calculateScore(m);"
        "sys.exit(0 if s == s else 3)"
    )
    env = dict(os.environ)
    env[_PROBE_ENV] = "1"
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True, env=env, timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False, "probe could not run"
    if proc.returncode == 0:
        return True, "ok"
    signal_note = ""
    if proc.returncode < 0 or proc.returncode > 128:
        sig = -proc.returncode if proc.returncode < 0 else proc.returncode - 128
        signal_note = f" (killed by signal {sig})"
    return False, f"probe exited {proc.returncode}{signal_note}"


# --------------------------------------------------------------------------- #
# Import-time wiring
# --------------------------------------------------------------------------- #
_SASCORER = None
SA_AVAILABLE = False
SA_BACKEND = "unavailable"
_PROBE_DETAIL = "not run"

try:
    _SASCORER = _load_patched_sascorer()
except Exception as _exc:                                          # noqa: BLE001
    _PROBE_DETAIL = f"import failed: {_exc}"
else:
    if os.environ.get(_PROBE_ENV):
        # We *are* the probe child: assume usable and let the crash speak.
        SA_AVAILABLE = True
        SA_BACKEND = "SA_Score (MorganGenerator shim, probe child)"
    else:
        _ok, _PROBE_DETAIL = _probe_in_subprocess()
        SA_AVAILABLE = _ok
        if _ok:
            SA_BACKEND = "SA_Score (MorganGenerator shim)"

if not SA_AVAILABLE:
    _allow = os.environ.get(_FALLBACK_ENV) == "1"
    _msg = (
        "sa_score: the RDKit SA_Score contrib cannot run in this environment "
        f"({_PROBE_DETAIL}).\n"
        "  The synthesizability screen would be INACTIVE, so a library built "
        "here would not match one built elsewhere.\n"
        f"  Set {_FALLBACK_ENV}=1 to accept a weaker QED-based screen instead."
    )
    if not _allow:
        raise RuntimeError(_msg)
    SA_BACKEND = "QED fallback (SA unavailable, explicitly allowed)"
    print(f"sa_score: WARNING — {_msg}", file=sys.stderr)


def calculate_score(mol):
    """SA score in [1, 10] (1 = easy to make, 10 = hard).

    Raises RuntimeError when the SA backend is unavailable; callers that want a
    weaker screen must opt in via the environment, so the degradation is never
    silent.
    """
    if not SA_AVAILABLE:
        raise RuntimeError(
            "sa_score.calculate_score called with no working SA backend "
            f"({_PROBE_DETAIL})")
    return float(_SASCORER.calculateScore(mol))
