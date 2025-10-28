import math
import random
from math import gcd, isqrt
import os

#text based database system for logging factor results

def check_database(N, filename="shor_database.txt"):
    """
    Check if the given N value already exists in the text database.
    Returns True if found, False otherwise.
    """
    try:
        with open(filename, "r") as f:
            for line in f:
                if f"N={N}" in line:  # Correct indentation
                    print(f"[Database] Found existing entry for N={N}: {line.strip()}")
                    return True
    except FileNotFoundError:
        pass
    return False

def log_to_database(N, p, q, a, r, filename="shor_database.txt"):
    """
    Appends a new result to the text file only if it's not already present.
    """
    # Create header if file doesn’t exist yet
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            f.write("N | p | q | a | r\n")
            f.write("-----------------------------------\n")

    # Avoid duplicates
    if check_database(N, filename):
        print(f"[Database] N={N} already recorded. Skipping log entry.")
        return

    with open(filename, "a") as f:
        f.write(f"N={N}, p={p}, q={q}, a={a}, r={r}\n")
    print(f"[Database] Logged new entry for N={N}: p={p}, q={q}, a={a}, r={r}")


# Prime number helper functions
def is_prime(n):
    """Check if n is prime."""
    if n < 2:
        return False
    if n in (2, 3):
        return True
    if n % 2 == 0:
        return False
    for i in range(3, isqrt(n) + 1, 2):
        if n % i == 0:
            return False
    return True

def generate_prime(min_val=5, max_val=50):
    """Return a random prime number between min_val and max_val."""
    primes = [n for n in range(min_val, max_val + 1) if is_prime(n)]
    return random.choice(primes)


# --- Step 1: Classical Order-Finding ---
def find_period_classical(a, N):
    """Find smallest r > 0 such that a^r ≡ 1 (mod N)."""
    if gcd(a, N) != 1:
        return None

    r = 1
    value = pow(a, r, N)
    while value != 1:
        r += 1
        value = pow(a, r, N)
        if r > N:
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
        return 2, N // 2, N, None, None

    # Try several random 'a' values
    for attempt in range(5):
        a = random.randint(2, N - 2)
        print(f"Trying a = {a}")

        g = gcd(a, N)
        if g > 1:
            # Lucky guess — already found a factor
            return g, N // g, N, a, None

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
            return p, q, N, a, r

    return None, None, N, a, r


# Database writer

def log_to_database(N, p, q, a, r, filename="shor_database.txt"):
    """Write results to a text file, skipping duplicates."""
    try:
        # Check for duplicates
        with open(filename, "r") as f:
            lines = f.readlines()
            for line in lines:
                if f"N={N}" in line:
                    return  # Already recorded
    except FileNotFoundError:
        # Create file if it doesn't exist
        with open(filename, "w") as f:
            f.write("N | p | q | a | r\n")
            f.write("-" * 35 + "\n")

    with open(filename, "a") as f:
        f.write(f"N={N}, p={p}, q={q}, a={a}, r={r}\n")


# Main Execution
def run_multiple(limit=20, min_p=5, max_p=50, max_attempts=50):
    """Run classical Shor’s factoring on random semiprimes and store results."""
    print(f"\nRunning classical Shor’s algorithm for up to {limit} values...\n")
    count = 0
    attempts = 0

    while count < limit and attempts < max_attempts:
        attempts += 1
        p = generate_prime(min_p, max_p)
        q = generate_prime(min_p, max_p)
        if p == q:
            continue
        N = p * q

        # Skip if already logged
        if check_database(N):
            print(f"[Skip] N={N} already recorded — moving to next.")
            continue

        p_val, q_val, N_val, a, r = classical_shor(N)
        if p_val and q_val:
            log_to_database(N_val, p_val, q_val, a, r)
            print(f"[{count+1}] Success → N={N_val}, p={p_val}, q={q_val}, a={a}, r={r}")
            count += 1

    print(f"\n✅ Done. Found {count} new entries after {attempts} total attempts.")
    if attempts >= max_attempts:
        print("⚠️  Max attempts reached — no new N values found.")
    


# Run program
if __name__ == "__main__":
    run_multiple(limit=30, min_p=50, max_p=70)
