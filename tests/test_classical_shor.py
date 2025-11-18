import unittest
import importlib.util
from pathlib import Path


def _load_classical_module():
    """
    Try to find the classical Shor implementation inside src/.
    If your file is named differently, update CANDIDATE_PATHS.
    """
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

    raise AssertionError(
        "Could not find classical Shor file. "
        "Edit candidate_paths in test_classical_shor.py to point to the correct file."
    )


_mod = _load_classical_module()


def _get_factor_function():
    """
    Pick a factoring function from the module.
    Update the names list if your function is called something else.
    """
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

    raise AssertionError(
        "Could not find a classical factoring function in the module.\n"
        "Export something like `def classical_shor(N): ...` or add its name to names[]."
    )


FACT = _get_factor_function()


def _norm_result(n, out):
    """Return (p, q) sorted. Accepts int or iterable pair."""
    if out is None:
        return None
    if isinstance(out, (tuple, list, set)):
        a, b = list(out)[:2]
        a, b = int(a), int(b)
        assert a * b == n, f"Returned pair does not multiply to {n}"
        return tuple(sorted((a, b)))
    
    f = int(out)
    assert 1 < f < n and n % f == 0, f"{f} is not a non-trivial factor of {n}"
    return tuple(sorted((f, n // f)))


class TestClassicalShor(unittest.TestCase):
    def test_semiprimes(self):
        """Basic correctness on small semiprimes."""
        cases = [
            (15, (3, 5)),
            (21, (3, 7)),
            (33, (3, 11)),
            (35, (5, 7)),
        ]
        for n, expected in cases:
            with self.subTest(N=n):
                out = FACT(n)
                pq = _norm_result(n, out)
                self.assertEqual(pq, expected)

    def test_primes_return_none_or_fail_cleanly(self):
        """Prime inputs should not produce factors."""
        primes = [2, 3, 5, 7, 11, 13]
        for p in primes:
            with self.subTest(N=p):
                try:
                    out = FACT(p)
                    self.assertIn(out, (None, False, ()),
                                  "Prime should not yield non-trivial factors")
                except (ValueError, RuntimeError):
                    
                    pass

    def test_even_composites_allowed(self):
        """Even composites either factor or are clearly unsupported."""
        nums = [8, 12, 16, 18]
        for n in nums:
            with self.subTest(N=n):
                try:
                    out = FACT(n)
                    if out is None:
                        
                        continue
                    _ = _norm_result(n, out)
                except Exception:
                    
                    continue


if __name__ == "__main__":
    unittest.main()

