import unittest
import importlib.util
from pathlib import Path

def _load_classical_module():
    root = Path(__file__).resolve().parents[1]

    candidate_paths = [
        root / "GUI" / "ClassicalShors.py",
        root / "src" / "ClassicalShors.py",
        root / "src" / "ShorNonQuantum.py",
        root / "src" / "classical_shor.py",
    ]

    for p in candidate_paths:
        if p.exists():
            spec = importlib.util.spec_from_file_location(p.stem, p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod

    raise AssertionError("Could not find classical Shor file.")

_mod = _load_classical_module()

def _get_factor_function():
    names = [
        "classical_shor",
        "ClassicalShor",
        "factor_semiprime",
        "factor_N",
        "factor",
    ]
    for name in names:
        if hasattr(_mod, name):
            return getattr(_mod, name)
    raise AssertionError("No factoring function found.")

FACT = _get_factor_function()

def _norm_result(n, out):
    if out is None:
        return None
    if isinstance(out, (tuple, list, set)):
        a, b = list(out)[:2]
        a, b = int(a), int(b)
        return tuple(sorted((a, b)))
    f = int(out)
    return tuple(sorted((f, n // f)))

class TestClassicalShor(unittest.TestCase):
    def test_semiprimes(self):
        cases = [
            (15, (3, 5)),
            (21, (3, 7)),
            (33, (3, 11)),
            (35, (5, 7)),
        ]
        for n, expected in cases:
            with self.subTest(N=n):
                out = FACT(n)
                self.assertEqual(_norm_result(n, out), expected)

    def test_primes_return_handle_gracefully(self):
        primes = [2, 3, 5, 7, 11, 13]
        for p in primes:
            with self.subTest(N=p):
                try:
                    FACT(p)
                except Exception as e:
                    self.fail(f"Prime {p} caused an unexpected error: {e}")

    def test_even_composites_allowed(self):
        nums = [8, 12, 16, 18]
        for n in nums:
            with self.subTest(N=n):
                try:
                    out = FACT(n)
                    if out is not None:
                        _norm_result(n, out)
                except:
                    continue

if __name__ == "__main__":
    unittest.main()

