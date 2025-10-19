import math
import random
from math import gcd

# --- Step 1: Classical Order-Finding ---
def find_period_classical(a, N):
    """
    Finds the smallest r > 0 such that a^r ≡ 1 (mod N)
    This is a brute-force classical substitute for the quantum step.
    """
    if gcd(a, N) != 1:
        return None  # invalid if a shares factors with N

    r = 1
    value = pow(a, r, N)
    while value != 1:
        r += 1
        value = pow(a, r, N)
        if r > N:  # failsafe for large N
            return None
    return r


# --- Step 2: Attempt to Factor Using Shor’s Idea ---
def classical_shor(N):
    """
    Classical version of Shor's algorithm.
    Attempts to find non-trivial factors of N by simulating the quantum period finding.
    """
    # Check for trivial cases first
    if N % 2 == 0:
        return 2, N // 2

    # Try several random 'a' values
    for attempt in range(5):
        a = random.randint(2, N - 2)
        print(f"Trying a = {a}")

        g = gcd(a, N)
        if g > 1:
            # Lucky guess — already found a factor
            return g, N // g

        # Find period r using classical method
        r = find_period_classical(a, N)
        print(f"  Found r = {r}")

        if r is None or r % 2 != 0:
            continue

        # Compute possible factors
        x = pow(a, r // 2, N)
        if x == 1 or x == N - 1:
            continue

        p = gcd(x - 1, N)
        q = gcd(x + 1, N)
        if p * q == N:
            return p, q

    return None


# --- Step 3: Run Test ---
if __name__ == "__main__":
    N = 15  # You can change this to test other numbers like 21, 33, etc.
    print(f"\nAttempting to factor N = {N} using classical Shor's algorithm...\n")

    factors = classical_shor(N)
    if factors:
        print(f"\n Factors of {N}: {factors[0]} × {factors[1]} = {N}")
    else:
        print("\n Failed to find factors.")
