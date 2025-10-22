# Uncomment or comment this out
!pip install qiskit qiskit-aer-gpu-cu11 --upgrade

from math import gcd, log2, pi
from fractions import Fraction
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
try:
    from qiskit.circuit.library import QFTGate as QFT  # not used now, but kept for compat
except Exception:
    from qiskit.circuit.library import QFT
from qiskit.circuit.library import UnitaryGate
try:
    from qiskit_aer import AerSimulator
except ModuleNotFoundError:
    from qiskit.providers.aer import AerSimulator


def n_qubits_for(N: int):
    # Return the number of work qubits needed to represent integers from 0 to N-1.
    return max(1, int(np.ceil(np.log2(N))))

def continued_fraction_phase(p, d):
    return Fraction(p).limit_denominator(d)

def try_factor_from_order(a, r, N):
    # Factor Ordering helper for Shor's Algorithm testing. (Validating if the R value is even nontrivial)
    if r % 2:
        return None
    x = pow(a, r // 2, N)
    if x in (1, 0, N - 1):
        return None
    p, q = gcd(x - 1, N), gcd(x + 1, N)
    if 1 < p < N and 1 < q < N:
        return (p, q)
    return None

def primes_upto(n):
    # Prime Number helper
    if n < 3:
        return []
    sieve = [True] * n
    sieve[0:2] = [False, False]
    for p in range(2, int(n ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p:n:p] = [False] * len(range(p * p, n, p))
    return [i for i, v in enumerate(sieve) if v]

_CU_CACHE = {}

def c_mult_mod_N(a, N, n_work):
    # Build a one-controlled unitary that multiplies the work register by 'a modulo N'.
    key = (a, N, n_work)
    if key in _CU_CACHE:
        return _CU_CACHE[key]

    dim = 1 << n_work
    U = np.zeros((dim, dim), dtype=complex)
    for y in range(dim):
        tgt = (a * y) % N if y < N else y
        U[tgt, y] = 1.0

    Ugate = UnitaryGate(U, label=f"{a}_mod_{N}")
    CU = Ugate.control(1) # turn the unitary into a controlled unitary with one control qubit
    _CU_CACHE[key] = CU
    return CU


def inverse_qft_no_swaps(qc: QuantumCircuit, qubits):
    # Apply the inverse Quantum Fourier Transform to the given list of qubits.
    # This version does not include final swap gates. It is written for the convention where the first counting qubit represents the least significant bit. 
    n = len(qubits)
    # Walk from the most significant position down to the least significant position
    for j in range(n - 1, -1, -1):
        # Apply controlled phase rotations with decreasing angle
        for m in range(j - 1, -1, -1):
            angle = -pi / (1 << (j - m))
            qc.cp(angle, qubits[m], qubits[j])
        # Apply a Hadamard to convert phase into amplitude on this qubit
        qc.h(qubits[j])

def order_finding_qpe(a, N, n_count, work_prep="one"):
    # Quantum Phase Estimation function!!!!!!!!
    n_work = n_qubits_for(N)
    count = QuantumRegister(n_count, "count")
    work  = QuantumRegister(n_work, "work")
    cl    = ClassicalRegister(n_count, "c")
    qc = QuantumCircuit(count, work, cl)

    # Prepare the work register initial state
    # If "one", set the integer value one by flipping the least significant work qubit.
    # Otherwise, create a uniform superposition across all work states.
    
    # It's more efficient to start from one because that's ultimately the condition we are looking to satisfy for our 'R' value finder.
    if work_prep == "one":
        qc.x(work[0])  # prepare the integer value one
    else:
        for q in work:
            qc.h(q)

    # Put the counting register into superposition
    qc.h(count)

    # Apply controlled modular multiplications by powers of a
    # The k-th counting qubit controls multiplication by a^(2^k) modulo N
    for k in range(n_count):
        a_k = pow(a, 1 << k, N)
        if a_k == 1:
            continue
        qc.append(c_mult_mod_N(a_k, N, n_work), [count[k]] + list(work))

    # Decode the phase with the inverse Quantum Fourier Transform
    inverse_qft_no_swaps(qc, list(count))

    # Measure the counting register to obtain a binary approximation of the phase
    qc.measure(count, cl)
    return qc

def flatten_circuit(qc: QuantumCircuit) -> QuantumCircuit:
    # Recursively flatten the circuit. There are much better ways to do it, but I had to deal with headache errors from stuff like composite gates. (PLEASE FIX THIS LATER SAM)
    while any(getattr(inst.operation, "definition", None) is not None for inst in qc.data):
        qc = qc.decompose()
    return qc.decompose()

def shor_factor_anyN(N: int,
                     n_count=None,
                     shots=4096,
                     max_trials=3,
                     work_prep="one",
                     verbose=True):
    if N < 4:
        print("[Info] N too small.")
        return None
    # Auto select counting qubits if none are selected.
    if n_count is None:
        n_count = max(4, int(np.ceil(2 * log2(N))))

    # Build the simulator. Try to use the graphics processing unit. Fall back to the central processor. Graphics unit works in Colab and the QuantumX.
    backend = AerSimulator(method="statevector")
    try:
        backend.set_options(device="GPU", precision="single", fusion_enable=True)
        if verbose:
            print("[Backend] GPU statevector + fusion (single precision).")
    except Exception:
        print("[Backend] CPU fallback.")
    # Choose some base values that are prime, odd, and coprime with N
    primes = [p for p in primes_upto(N) if p % 2 and gcd(p, N) == 1 and p > 2][:max_trials]
    # If none are available (almost never happens, just for redundancy sake), choose odd coprime values
    if not primes:
        primes = [a for a in range(3, N, 2) if gcd(a, N) == 1][:max_trials]

    if verbose:
        print(f"[Setup] N={N}, n_count={n_count}, work_qubits={n_qubits_for(N)}")
        print(f"[Setup] Trying a values (prime, odd, coprime, <=3): {primes}")

    for i, a in enumerate(primes, 1):
        if verbose:
            print(f"\n[Trial {i}/{len(primes)}] a={a}")

        # Build and flatten the circuit
        qc = order_finding_qpe(a, N, n_count, work_prep)
        tqc = flatten_circuit(qc)

        # Run the circuit and collect a histogram of measurement results
        result = backend.run(tqc, shots=shots).result()
        counts = result.get_counts()

        # Identify the most frequent outcomes to reduce the effect of sampling noise
        top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:3]
        top_raw = top[0][0]

        # Qiskit prints bitstrings with the most significant bit on the left. Kind of annoying. We have to reverse the bitstring before converting to an integer.
        top_raw_little = top_raw[::-1]

        # Convert the average index into a phase in the range [0, 1)
        avg = sum(int(k[::-1], 2) * v for k, v in top) / sum(v for _, v in top)
        phase = avg / (1 << n_count)
        # Use a continued fraction with a denominator limited by twice N to estimate s over r
        frac = continued_fraction_phase(round(phase, 8), d=2 * N)
        r = frac.denominator

        if verbose:
            print(f"  result(msb to lsb)={top_raw}  result(lsb to msb)={top_raw_little}")
            print(f"  phase = {phase:.6f}  = {frac}  to r={r}")

        # Try to turn the candidate order into nontrivial factors of N
        fac = try_factor_from_order(a, r, N)
        if fac:
            p, q = fac
            print(f"[SUCCESS] {N} = {p} × {q}  (a={a}, r={r})")
            return (p, q)
        if verbose:
            print("  No nontrivial factors for this a.")

    print("[FAIL] No factors found with current settings.")
    return None


if __name__ == "__main__":
    # You can change N, the number of counting qubits, the number of shots, and the initial work register preparation here.
    # For larger N, consider reducing the number of counting qubits to keep runtime reasonable. Right now, it's automatic, but i'll add a seperate setting later.
    shor_factor_anyN(N=35, n_count=10, shots=8192, work_prep="one", verbose=True)
