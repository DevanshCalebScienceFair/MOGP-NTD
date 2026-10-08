#!/usr/bin/env python3
"""verify_fixtures.py -- check every file under spec/ against spec/MANIFEST.md.

Run (from the repository root):
    conda run -n mogp-drug python spec/verify_fixtures.py

Exit status 0 = every checksum matched (stochastic files may differ; see below).
Exit status 1 = at least one deterministic file drifted, is missing, or is new.

WHY THIS EXISTS
---------------
The nine spec clusters captured ~16 MB of differential-reference fixtures that a
pure-numpy reimplementation will be validated against.  A fixture that changes
silently -- because someone regenerated it with a different library version, or
edited a tolerance, or the cached library was rebuilt -- destroys the value of
every number derived from it, and does so invisibly.  This script makes that
drift loud.

It reads the CHECKSUMS block at the end of spec/MANIFEST.md, so MANIFEST.md is
the single source of truth.  Editing a fixture without regenerating the manifest
fails here; editing the manifest without regenerating the fixture also fails.

STOCHASTIC FILES
----------------
Some files are not byte-reproducible by construction: wall-clock timings, a
seeded Monte-Carlo estimator, and gpytorch's stochastic-Lanczos log-determinant
above N*T = 800.  They are listed in MANIFEST.md's EXCLUDED-FROM-BYTE-COMPARE
block.  This script still checksums them -- drift is worth seeing -- but reports
them as STOCHASTIC rather than FAIL and does not fail the run over them.  Pass
--strict to fail on them too (useful only when you believe nothing should have
moved at all).

MODES
-----
    (no flags)   verify; print a per-cluster summary and any drift
    -v           also list every OK file
    --strict     treat stochastic drift as failure
    --cluster X  restrict to spec/X/ (repeatable)
    --list-only  print the parsed manifest and exit without hashing
    --regen      rewrite MANIFEST.md's CHECKSUMS and EXCLUDED blocks in place
                 from the current tree, leaving all prose untouched.  Use this
                 ONLY after deliberately regenerating a fixture, and say in the
                 commit message which capture script produced the new bytes.

WHAT THIS SCRIPT DOES NOT DO
----------------------------
It checks that the fixtures on disk are the ones the manifest describes.  It does
NOT re-run the capture scripts and it does NOT check that an implementation meets
the agreement criteria.  For the first, re-run the capture script named in the
cluster's section of MANIFEST.md in a fresh interpreter and byte-compare.  For
the second, the criterion for every file is in MANIFEST.md section 6.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MANIFEST = os.path.join(HERE, "MANIFEST.md")

# Files under spec/ that are not indexed and must not be reported as new.
SELF = {"MANIFEST.md", "verify_fixtures.py"}

CHECKSUM_HEADING = "## CHECKSUMS"
EXCLUDED_HEADING = "## EXCLUDED-FROM-BYTE-COMPARE"

# Terminal colours, only when stdout is a tty.
_TTY = sys.stdout.isatty()


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _TTY else s


RED = lambda s: _c("31", s)
GREEN = lambda s: _c("32", s)
YELLOW = lambda s: _c("33", s)
BOLD = lambda s: _c("1", s)


# --------------------------------------------------------------------------- #
# manifest parsing
# --------------------------------------------------------------------------- #

def _fenced_block_after(text: str, heading: str) -> str:
    """Return the contents of the first ``` fenced block after `heading`."""
    i = text.find(heading)
    if i < 0:
        raise SystemExit(f"{MANIFEST}: heading {heading!r} not found")
    m = re.search(r"^```[^\n]*\n(.*?)^```", text[i:], re.S | re.M)
    if not m:
        raise SystemExit(f"{MANIFEST}: no fenced block after {heading!r}")
    return m.group(1)


def parse_manifest(path: str = MANIFEST):
    """-> (expected {relpath: sha256}, stochastic {relpath: class})."""
    if not os.path.exists(path):
        raise SystemExit(f"missing manifest: {path}")
    text = open(path, encoding="utf-8").read()

    expected: dict[str, str] = {}
    for line in _fenced_block_after(text, CHECKSUM_HEADING).splitlines():
        line = line.strip()
        if not line:
            continue
        # "sha256  relative/path" -- the format shasum -a 256 emits.
        m = re.match(r"^([0-9a-f]{64})\s\s?(.+)$", line)
        if not m:
            raise SystemExit(f"{path}: unparseable CHECKSUMS line: {line!r}")
        rel = m.group(2).strip()
        if rel in expected:
            raise SystemExit(f"{path}: duplicate CHECKSUMS entry for {rel!r}")
        expected[rel] = m.group(1)

    stochastic: dict[str, str] = {}
    for line in _fenced_block_after(text, EXCLUDED_HEADING).splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise SystemExit(f"{path}: unparseable EXCLUDED line: {line!r}")
        stochastic[parts[1].strip()] = parts[0]

    unknown = sorted(set(stochastic) - set(expected))
    if unknown:
        raise SystemExit(
            f"{path}: EXCLUDED lists paths absent from CHECKSUMS: {unknown}")
    if not expected:
        raise SystemExit(f"{path}: CHECKSUMS block is empty")
    return expected, stochastic


# --------------------------------------------------------------------------- #
# hashing
# --------------------------------------------------------------------------- #

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def walk_spec(root: str = HERE) -> list[str]:
    """Every file under spec/, relative, excluding __pycache__ and this pair."""
    out = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames
                       if d != "__pycache__" and not d.startswith(".")]
        for fn in filenames:
            if fn.startswith("."):
                continue
            rel = os.path.relpath(os.path.join(dirpath, fn), root)
            if rel in SELF:
                continue
            out.append(rel)
    return sorted(out)


# --------------------------------------------------------------------------- #
# regeneration
# --------------------------------------------------------------------------- #

def regen(path: str = MANIFEST) -> int:
    """Rewrite the CHECKSUMS and EXCLUDED blocks from the current tree.

    Prose is untouched.  The EXCLUDED block is rebuilt from the classes the
    existing manifest already records, so a file's determinism class is never
    silently changed by a regeneration -- only its digest is.
    """
    old_expected, stochastic = parse_manifest(path)
    text = open(path, encoding="utf-8").read()
    rels = walk_spec()

    added = sorted(set(rels) - set(old_expected))
    removed = sorted(set(old_expected) - set(rels))

    lines = [f"{sha256_file(os.path.join(HERE, r))}  {r}" for r in rels]
    new_checks = "\n".join(lines) + "\n"
    new_excl = "\n".join(f"{stochastic[r]}  {r}"
                         for r in rels if r in stochastic) + "\n"

    def replace_block(t: str, heading: str, body: str) -> str:
        i = t.find(heading)
        m = re.search(r"^```[^\n]*\n(.*?)^```", t[i:], re.S | re.M)
        s, e = i + m.start(1), i + m.end(1)
        return t[:s] + body + t[e:]

    text = replace_block(text, CHECKSUM_HEADING, new_checks)
    text = replace_block(text, EXCLUDED_HEADING, new_excl)
    open(path, "w", encoding="utf-8").write(text)

    print(f"regenerated {path}: {len(rels)} files")
    for r in added:
        print(f"  {GREEN('added')}   {r}")
    for r in removed:
        print(f"  {YELLOW('removed')} {r}"
              f"  <- if this was a fixture, say why in the commit message")
    changed = [r for r in rels
               if r in old_expected
               and old_expected[r] != sha256_file(os.path.join(HERE, r))]
    for r in changed:
        tag = f"({stochastic[r]})" if r in stochastic else RED("(DETERMINISTIC)")
        print(f"  {YELLOW('digest changed')} {r} {tag}")
    if changed or added or removed:
        print("\nRecord in the commit message WHICH capture script produced the "
              "new bytes, and re-run it in a fresh interpreter to confirm the "
              "new digest is itself reproducible.")
    return 0


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Check every file under spec/ against spec/MANIFEST.md.")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="also list files that matched")
    ap.add_argument("--strict", action="store_true",
                    help="treat drift in a stochastic file as a failure")
    ap.add_argument("--cluster", action="append", default=None, metavar="NAME",
                    help="restrict to spec/NAME/ (repeatable)")
    ap.add_argument("--list-only", action="store_true",
                    help="print the parsed manifest and exit")
    ap.add_argument("--regen", action="store_true",
                    help="rewrite MANIFEST.md's CHECKSUMS/EXCLUDED blocks")
    args = ap.parse_args(argv)

    if args.regen:
        return regen()

    expected, stochastic = parse_manifest()

    if args.list_only:
        for rel in sorted(expected):
            # This is the verifier's binary view: byte-comparable, or one of the
            # classes MANIFEST.md's EXCLUDED block records.  The fine-grained
            # determinism class and the agreement criterion are in section 6.
            cls = stochastic.get(rel, "byte-comparable")
            print(f"{expected[rel]}  {cls:16s}  {rel}")
        print(f"\n{len(expected)} indexed, {len(stochastic)} not byte-comparable")
        return 0

    on_disk = set(walk_spec())
    keep = None
    if args.cluster:
        keep = tuple(c.strip("/") + os.sep for c in args.cluster)

    def sel(rel: str) -> bool:
        return keep is None or rel.startswith(keep)

    ok, drift, stoch_drift, missing, new = [], [], [], [], []
    for rel in sorted(expected):
        if not sel(rel):
            continue
        p = os.path.join(HERE, rel)
        if not os.path.exists(p):
            missing.append(rel)
            continue
        got = sha256_file(p)
        if got == expected[rel]:
            ok.append(rel)
        elif rel in stochastic:
            stoch_drift.append((rel, stochastic[rel], expected[rel], got))
        else:
            drift.append((rel, expected[rel], got))
    for rel in sorted(on_disk - set(expected)):
        if sel(rel):
            new.append(rel)

    # ---- per-cluster summary ------------------------------------------- #
    clusters: dict[str, list[int]] = {}
    for rel in sorted(expected):
        if not sel(rel):
            continue
        cl = rel.split(os.sep)[0]
        c = clusters.setdefault(cl, [0, 0, 0, 0])   # ok, drift, stoch, missing
        if rel in ok:
            c[0] += 1
        elif any(rel == d[0] for d in drift):
            c[1] += 1
        elif any(rel == d[0] for d in stoch_drift):
            c[2] += 1
        elif rel in missing:
            c[3] += 1

    print(BOLD("spec/ fixture verification") + f"   manifest: {MANIFEST}")
    print()
    print(f"  {'cluster':<12} {'ok':>5} {'DRIFT':>6} {'stoch':>6} {'MISSING':>8}")
    for cl in sorted(clusters):
        o, d, s, m = clusters[cl]
        dd = RED(f"{d:>6}") if d else f"{d:>6}"
        mm = RED(f"{m:>8}") if m else f"{m:>8}"
        ss = YELLOW(f"{s:>6}") if s else f"{s:>6}"
        print(f"  {cl:<12} {o:>5} {dd} {ss} {mm}")
    print()

    if args.verbose:
        for rel in ok:
            print(f"  {GREEN('ok')}   {rel}")
        print()

    if stoch_drift:
        print(YELLOW("NOT BYTE-COMPARABLE, moved as expected "
                     "(see MANIFEST.md section 7 for what IS assertable):"))
        for rel, cls, exp, got in stoch_drift:
            print(f"  {cls:<16} {rel}")
            print(f"      manifest {exp}")
            print(f"      on disk  {got}")
        print()

    if drift:
        print(RED(BOLD("DRIFT -- a deterministic fixture changed:")))
        for rel, exp, got in drift:
            print(f"  {rel}")
            print(f"      manifest {exp}")
            print(f"      on disk  {got}")
        print()
        print("  Every number derived from that fixture is now unverified.")
        print("  Re-run the capture script named in MANIFEST.md section 6 for")
        print("  this cluster in a FRESH interpreter.  If the new bytes are")
        print("  themselves reproducible, the fixture was legitimately")
        print("  regenerated: run --regen and say which script did it.  If they")
        print("  are NOT reproducible, the capture script has become")
        print("  non-deterministic and that is a defect.")
        print()

    if missing:
        print(RED(BOLD("MISSING -- indexed but not on disk:")))
        for rel in missing:
            print(f"  {rel}")
        print()

    if new:
        print(YELLOW(BOLD("NEW -- on disk but not indexed:")))
        for rel in new:
            print(f"  {rel}")
        print("  Add it with --regen, or delete it.  An unindexed fixture is an")
        print("  unverified fixture.")
        print()

    total = len(ok) + len(drift) + len(stoch_drift) + len(missing)
    failed = bool(drift or missing or new) or (args.strict and bool(stoch_drift))
    verdict = RED("FAIL") if failed else GREEN("PASS")
    print(f"{verdict}  {len(ok)}/{total} byte-identical"
          + (f", {len(stoch_drift)} stochastic" if stoch_drift else "")
          + (f", {len(drift)} DRIFT" if drift else "")
          + (f", {len(missing)} MISSING" if missing else "")
          + (f", {len(new)} NEW" if new else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
