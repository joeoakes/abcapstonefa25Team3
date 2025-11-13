import pytest
import importlib.util
import os

# Load ClassicalShors.py from root 
MODULE_PATH = os.path.join(os.path.dirname(__file__), "..", "ClassicalShors.py")
spec = importlib.util.spec_from_file_location("ClassicalShors", MODULE_PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)  # type: ignore

# Locate the classical factoring function 
def get_factor_function():
    for name in ["classical_shor", "ClassicalShor", "factor_semiprime", "factor"]:
        if hasattr(mod, name):
            return getattr(mod, name)
    raise ImportError("Could not find a classical shor factoring function in ClassicalShors.py")

factor_func = get_factor_function()

@pytest.mark.parametrize("n,expected", [
    (15, (3, 5)),
    (21, (3, 7)),
    (33, (3, 11)),
])
def test_classical_shor_factoring(n, expected):
    """Basic correctness test for Classical Shor factoring."""
    result = factor_func(n)
    if isinstance(result, (tuple, list)):
        result = tuple(sorted(map(int, result)))
        assert result == expected
    elif isinstance(result, int):
        assert n % result == 0, "Returned integer is not a nontrivial factor"
    else:
        pytest.fail(f"Unexpected return type: {type(result)}")

@pytest.mark.parametrize("n", [2, 3, 5, 7, 11])
def test_classical_shor_primes_return_none(n):
    """Prime numbers should return None or raise."""
    try:
        result = factor_func(n)
        assert result in (None, False, ()), "Prime should not yield factors"
    except (ValueError, RuntimeError):
        pytest.xfail("Prime input raises acceptable exception")
