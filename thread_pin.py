"""
thread_pin.py
=============

The single-threaded BLAS / OpenMP pin. Import it BEFORE numpy, pandas, torch or
any repo module that loads one, in every entry-point script::

    import thread_pin  # noqa: F401  -- must precede numpy / pandas / torch

Two reasons, both from the block that used to live at the top of ``loop.py``:

* On Apple Silicon a duplicate ``libomp`` runtime can crash on import (OMP
  Error #15). ``KMP_DUPLICATE_LIB_OK=TRUE`` tolerates it.
* Single-threaded BLAS keeps the numerics independent of the machine: a threaded
  BLAS splits a reduction across however many cores it finds, and the summation
  order, hence the last bits, goes with it. ``test_core_gp.py`` measures it on
  the float64 GP loss (the BLAS-thread-count test): bitwise against the archived
  reference at one thread, only within the suite's bitwise floor at eight.

It only works if it runs first. A BLAS reads these variables once, when numpy
loads it, so importing this after numpy has been imported changes the process
environment (and so any child process) but not the BLAS already running here.
This module does not detect that case.

Semantics are ``loop.py``'s, unchanged: ``setdefault``, so a variable that is
already in the environment, including one set to something other than ``1``, is
left alone and an explicit outer override always wins. ``launch.py`` is the one
caller that forces the same values instead (see there), and ``run_matrix.py``
sets them, plus ``OPENBLAS_NUM_THREADS``, on every case's subprocess.
"""

import os

#: The variables and the values they get when unset. ``KMP_DUPLICATE_LIB_OK`` is
#: the duplicate-runtime tolerance; the rest pin a math backend to one thread.
THREAD_ENV_DEFAULTS = {
    "KMP_DUPLICATE_LIB_OK": "TRUE",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


def pin(environ=None):
    """Give each variable in :data:`THREAD_ENV_DEFAULTS` its value unless it is set.

    ``environ`` is a mapping to write into; the default is ``os.environ``. Returns
    it.
    """
    if environ is None:
        environ = os.environ
    for name, value in THREAD_ENV_DEFAULTS.items():
        environ.setdefault(name, value)
    return environ


pin()
