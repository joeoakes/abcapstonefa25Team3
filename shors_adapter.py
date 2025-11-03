# shors_adapter.py  (compatible with ClassicalShorsOld.py / QiskitShorsMain.py)
# Simple, student-style adapter that gives:
#   classical_factor(n) -> (p, q)
#   quantum_factor(n)   -> (p, q)
#
# It tries to import common module names; if not found, it tries to run the
# scripts; if that still fails, it uses a tiny naive factor fallback (fine for
# small n used in demos/class projects).

import importlib
import json
import re
import subprocess
import sys
import shutil
import os
from pathlib import Path
from math import isqrt

_INT_RE = re.compile(r"\b(\d+)\b")
HERE = Path(__file__).parent

def _to_tuple_factors(x):
    if isinstance(x, (tuple, list)) and len(x) == 2:
        return int(x[0]), int(x[1])
    if isinstance(x, dict) and "p" in x and "q" in x:
        return int(x["p"]), int(x["q"])
    if isinstance(x, str):
        nums = [int(z) for z in _INT_RE.findall(x)]
        if len(nums) >= 2:
            return nums[0], nums[1]
    raise ValueError(f"could not parse factors from: {x!r}")

def _try_func_names(mod, n, names):
    for name in names:
        fn = getattr(mod, name, None)
        if callable(fn):
            try:
                try:
                    out = fn(n)
                except TypeError:
                    out = fn(N=n)
                return _to_tuple_factors(out)
            except Exception:
                pass
    return None

def _run_as_script(pyfile, n):
    if not (HERE / pyfile).exists():
        raise FileNotFoundError(pyfile)
    python_exe = shutil.which(sys.executable) or "python"
    proc = subprocess.run(
        [python_exe, pyfile, "--N", str(n)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(HERE)
    )
    out = proc.stdout or ""
    # try JSON first
    try:
        j = json.loads(out)
        if isinstance(j, dict):
            if "p" in j and "q" in j:
                return int(j["p"]), int(j["q"])
            if "factors" in j and len(j["factors"]) >= 2:
                return int(j["factors"][0]), int(j["factors"][1])
    except Exception:
        pass
    # then first two ints
    nums = [int(z) for z in _INT_RE.findall(out)]
    if len(nums) >= 2:
        return nums[0], nums[1]
    raise RuntimeError(f"could not find factors in output of {pyfile}:\n{out}")

def _naive_factor(n):
    # last-resort fallback for small classroom n
    for i in range(2, isqrt(n) + 1):
        if n % i == 0:
            return i, n // i
    raise RuntimeError("naive factor failed (n may be prime or too large)")

def classical_factor(n):
    # Try to import one of these module names:
    module_names = [
        "ClassicalShors",      # preferred name if it exists
        "classicalshors",      # lowercase variant
        "ClassicalShorsOld",   # your repo currently has this
    ]
    func_names = ["factor_n", "factorN", "factor", "shor_classical",
                  "classical_shor", "get_factors", "run", "main"]

    for modname in module_names:
        try:
            mod = importlib.import_module(modname)
            out = _try_func_names(mod, n, func_names)
            if out:
                return out
        except Exception:
            pass

    # Try running likely script filenames if present
    for pyfile in ["ClassicalShors.py", "ClassicalShorsOld.py"]:
        try:
            return _run_as_script(pyfile, n)
        except Exception:
            pass

    # Fallback: naive factor (fine for small n like 1333)
    return _naive_factor(n)

def quantum_factor(n):
    module_names = [
        "QiskitShorsMain",
        "quantumshors",
        "QiskitShorsOld",
    ]
    func_names = ["factor_n", "factorN", "factor", "shor_quantum",
                  "quantum_shor", "get_factors", "run", "main"]

    for modname in module_names:
        try:
            mod = importlib.import_module(modname)
            out = _try_func_names(mod, n, func_names)
            if out:
                return out
        except Exception:
            pass

    for pyfile in ["QiskitShorsMain.py", "QiskitShorsOld.py"]:
        try:
            return _run_as_script(pyfile, n)
        except Exception:
            pass

    raise RuntimeError("quantum factoring not available (no module/script found)")
