"""Which code produced a cell's numbers, and is it the same code for every cell.

Every cell is a fresh ``run_one.py`` subprocess that imports the RESULT CODE
from disk when it starts, so an edit made while a sweep runs (or before a
``--resume``) would silently mix code versions inside one Wilcoxon test.  The
result code is every repo file a cell imports (measured with a live cell):

    bench_yong/run_one.py, bench_yong/yong_common.py, acquisition.py, mogp.py,
    thread_pin.py, utils/__init__.py, utils/featurize.py, mogp_core/*.py

plus external/GP-MOBO, which is pinned by commit separately.

The fingerprint is the sorted list of ``path git-blob-sha`` lines, hashed.  Git
blob ids are used on both sides, so a working tree (``git hash-object``) and a
commit (``git ls-tree``) are compared on the same scheme.

This module is deliberately NOT imported by run_one.py: it is harness code and
may change while a sweep runs; the result code may not.
"""

import hashlib
import os
import subprocess

BENCH_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(BENCH_DIR)

RESULT_CODE_FILES = (
    "bench_yong/run_one.py", "bench_yong/yong_common.py", "acquisition.py",
    "mogp.py", "thread_pin.py", "utils/__init__.py", "utils/featurize.py",
    # Tier 4 (Amendment 2).  Absent from the Tier 1-3 commits, so they are
    # absent on both sides of that sweep's comparison and cannot affect it.
    "bench_yong/run_one_published.py", "bench_yong/her_published.py",
)
RESULT_CODE_DIRS = ("mogp_core",)


def _git(*args):
    return subprocess.run(["git", "-C", REPO, *args], capture_output=True,
                          text=True, check=True).stdout


def _digest(pairs):
    lines = "\n".join(f"{p} {b}" for p, b in sorted(pairs))
    return hashlib.sha256(lines.encode()).hexdigest()


def fingerprint_at(ref):
    """Result-code fingerprint of commit ``ref``: ``(digest, {path: blob})``."""
    out = _git("ls-tree", "-r", ref, "--", *RESULT_CODE_FILES, *RESULT_CODE_DIRS)
    pairs = {}
    for line in out.splitlines():
        meta, path = line.split("\t", 1)
        if path.endswith(".py"):
            pairs[path] = meta.split()[2]
    return _digest(pairs.items()), pairs


def fingerprint_working_tree():
    """Result-code fingerprint of the files on disk now."""
    paths = list(RESULT_CODE_FILES)
    for d in RESULT_CODE_DIRS:
        for name in sorted(os.listdir(os.path.join(REPO, d))):
            if name.endswith(".py"):
                paths.append(f"{d}/{name}")
    blobs = _git("hash-object", "--", *paths).split()
    pairs = dict(zip(paths, blobs))
    return _digest(pairs.items()), pairs


def diff(a_pairs, b_pairs):
    keys = sorted(set(a_pairs) | set(b_pairs))
    return [k for k in keys if a_pairs.get(k) != b_pairs.get(k)]
