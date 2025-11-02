# shors_adapter.py
# goal: give me two simple functions I can call from anywhere:
#   classical_factor(n) -> (p, q)
#   quantum_factor(n)   -> (p, q)
# it tries to call functions inside the teammate files first.
# if that fails, it runs the file as a script and grabs two numbers from output.

import importlib
import json
import re
import subprocess
import sys
import shutil
from pathlib import Path

# gets all integers from a string 
_INT_RE = re.compile(r"\b(\d+)\b")

def _to_tuple_factors(x):
    # accept (p,q), [p,q], {"p":..,"q":..}, or a string with two ints
    if isinstance(x, (tuple, list)) and len(x) == 2:
        return int(x[0]), int(x[1])
    if isinstance(x, dict) and "p" in x and "q" in x:
        return int(x["p"]), int(x["q"])
    if isinstance(x, str):
        nums = [int(z) for z in _INT_RE.findall(x)]
        if len(nums) >= 2:
            return nums[0], nums[1]
    raise ValueError(f"could not parse factors from: {x}")

def _try_func_names(mod, n, names):
    
    for name in names:
        fn = getattr(mod, name, None)
        if callable(fn):
            try:
                try:
                    out = fn(n)      # most functions take (n)
                except TypeError:
                    out = fn(N=n)    # sometimes it's N=n
                return _to_tuple_factors(out)
            except Exception:
                # if one name fails, just try the next
                pass
    return None

def _run_as_script(pyfile, n):
    # last resort: run the file and parse two integers from its stdout
    python_exe = shutil.which(sys.executable) or "python"
    here = str(Path(__file__).parent)
    proc = subprocess.run(
        [python_exe, pyfile, "--N", str(n)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=here
    )
    out = proc.stdout or ""

    # first try JSON
    try:
        j = json.loads(out)
        if isinstance(j, dict):
            if "p" in j and "q" in j:
                return int(j["p"]), int(j["q"])
            if "factors" in j and len(j["factors"]) >= 2:
                return int(j["factors"][0]), int(j["factors"][1])
    except Exception:
        pass

    # then just grab the first two ints we see
    nums = [int(z) for z in _INT_RE.findall(out)]
    if len(nums) >= 2:
        return nums[0], nums[1]

    raise RuntimeError(f"could not find factors in output of {pyfile}:\n{out}")

def classical_factor(n):
    mod = importlib.import_module("ClassicalShors")
    names = ["factor_n", "factorN", "factor", "shor_classical", "classical_shor", "get_factors", "run", "main"]
    out = _try_func_names(mod, n, names)
    if out:
        return out
    return _run_as_script("ClassicalShors.py", n)

def quantum_factor(n):
    mod = importlib.import_module("QiskitShorsMain")
    names = ["factor_n", "factorN", "factor", "shor_quantum", "quantum_shor", "get_factors", "run", "main"]
    out = _try_func_names(mod, n, names)
    if out:
        return out
    return _run_as_script("QiskitShorsMain.py", n)
