# =========================================================
# shor_from_scratch_qiskit21_hamming_TIMED_ELAPSED_ONLY.py
# Shor's Algorithm from scratch with detailed performance timing
# Compatible with Qiskit >= 2.1
# Logs elapsed time per major function and per trial
# =========================================================

# Optional installation command for a clean environment (uncomment if needed)
# !pip install qiskit qiskit-aer pylatexenc --upgrade

import time                       # ⏱ High-resolution timing for performance measurement
from math import gcd, log2        # gcd for factorization checks; log2 for qubit estimation
from fractions import Fraction    # Used to approximate phase measurement as a rational number
import numpy as np                # Used to build modular multiplication matrices
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile

# Import the Quantum Fourier Transform gate (QFT)
# (QFT is used to extract periodicity from the phase estimation)
try:
    from qiskit.circuit.library import QFTGate as QFT
except Exception:
    from qiskit.circuit.library import QFT

# UnitaryGate lets us define custom controlled modular multiplication gates
from qiskit.circuit.library import UnitaryGate

# AerSimulator simulates the quantum circuit on classical hardware
try:
    from qiskit_aer import AerSimulator
except ModuleNotFoundError:
    from qiskit.providers.aer import AerSimulator


# ---------------------------------------------------------
# Helper: Performance logging function
# ---------------------------------------------------------
def log_elapsed(label, start, end):
    """
    Logs elapsed time for a labeled section of the code.

    This function is used throughout the algorithm to record how long
    each step takes. It helps us break down the total runtime into:
    - circuit construction
    - modular multiplication setup
    - quantum execution
    - classical post-processing.

    Parameters:
        label (str): name of the function or code section
        start (float): time.perf_counter() at the beginning
        end (float): time.perf_counter() at the end
    """
    print(f"⏳ {label} elapsed: {end - start:.6f} s")


# ---------------------------------------------------------
# Determine number of work qubits needed for N
# ---------------------------------------------------------
def n_qubits_for(N: int) -> int:
    # ⏱ Start timing for this helper computation
    t0 = time.perf_counter()

    # Minimum number of qubits needed to represent integers 0 to N-1
    result = max(1, int(np.ceil(np.log2(N))))

    # ⏱ End timing
    t1 = time.perf_counter()
    log_elapsed("n_qubits_for", t0, t1)
    return result


# ---------------------------------------------------------
# Continued fraction expansion of measured phase
# ---------------------------------------------------------
def continued_fraction_phase(phase: float, max_denominator: int):
    """
    Approximates a floating-point phase (from QPE measurement)
    as a fraction p/q with denominator limited to max_denominator.

    This is a key classical step in converting the observed quantum
    phase into an estimate of the order r.
    """
    # ⏱ Start timing classical post-processing
    t0 = time.perf_counter()

    # Convert phase to a rational approximation
    result = Fraction(phase).limit_denominator(max_denominator)

    # ⏱ End timing
    t1 = time.perf_counter()
    log_elapsed("continued_fraction_phase", t0, t1)
    return result


# ---------------------------------------------------------
# Classical factorization attempt using the order r
# ---------------------------------------------------------
def try_factor_from_order(a: int, r: int, N: int):
    """
    Given a coprime 'a' and an order 'r' (period), try to find
    non-trivial factors of N using the formula:
        p = gcd(a^(r/2) - 1, N)
        q = gcd(a^(r/2) + 1, N)

    This is a purely classical step and is typically very fast.
    """
    # ⏱ Start timing factor check
    t0 = time.perf_counter()

    result = None
    if r % 2 == 0:
        x = pow(a, r // 2, N)        # compute a^(r/2) mod N
        # If x is 1 or N-1, no useful factors are found
        if x not in (1, 0, N - 1):
            p = gcd(x - 1, N)
            q = gcd(x + 1, N)
            if 1 < p < N and 1 < q < N:
                result = (p, q)

    # ⏱ End timing
    t1 = time.perf_counter()
    log_elapsed("try_factor_from_order", t0, t1)
    return result


# ---------------------------------------------------------
# Classical Hamming(7,4)-style single-bit error correction
# ---------------------------------------------------------
def hamming_correct(meas: str) -> str:
    """
    Applies simple single-bit error correction on the measurement
    bitstring using a (7,4) Hamming code. This improves robustness
    of the measured phase in the presence of single-bit noise.

    meas (str): measured bitstring (MSB first)
    returns: corrected bitstring (MSB first)
    """
    # ⏱ Start timing for error correction step
    t0 = time.perf_counter()

    # Reverse to LSB-first for easier indexing
    bits = [int(b) for b in meas[::-1]]
    n = len(bits)
    corrected = bits[:]

    # Process bitstring in 7-bit blocks
    for start in range(0, n, 7):
        seg = bits[start:start + 7]
        if len(seg) < 7:
            break
        # Calculate parity bits
        p1 = (seg[0] ^ seg[2] ^ seg[4] ^ seg[6])
        p2 = (seg[1] ^ seg[2] ^ seg[5] ^ seg[6])
        p3 = (seg[3] ^ seg[4] ^ seg[5] ^ seg[6])
        # Syndrome tells us which bit is wrong
        syndrome = (p3 << 2) | (p2 << 1) | p1
        if syndrome != 0 and syndrome <= len(seg):
            seg[syndrome - 1] ^= 1
        corrected[start:start + 7] = seg

    # Reverse back to MSB-first
    result = ''.join(str(b) for b in corrected[::-1])

    # ⏱ End timing
    t1 = time.perf_counter()
    log_elapsed("hamming_correct", t0, t1)
    return result


# ---------------------------------------------------------
# Controlled modular multiplication gate
# ---------------------------------------------------------
def c_mult_mod_N(a: int, N: int, n_work: int):
    """
    Builds a unitary operator that performs modular multiplication:
        |y⟩ -> |(a * y) mod N⟩
    and makes it controlled by a single counting qubit.

    This is one of the most expensive *classical pre-processing*
    steps since it involves constructing a large unitary matrix.
    """
    # ⏱ Start timing
    t0 = time.perf_counter()

    # Build matrix representing modular multiplication
    dim = 1 << n_work
    U = np.zeros((dim, dim), dtype=complex)
    for y in range(dim):
        tgt = (a * y) % N if y < N else y
        U[tgt, y] = 1.0

    # Convert matrix into a UnitaryGate and control it
    Ugate = UnitaryGate(U, label=f"{a}_mod_{N}")
    CU = Ugate.control(1)

    # ⏱ End timing
    t1 = time.perf_counter()
    log_elapsed("c_mult_mod_N", t0, t1)
    return CU


# ---------------------------------------------------------
# Quantum Phase Estimation (QPE) circuit builder
# ---------------------------------------------------------
def order_finding_qpe(a: int, N: int, n_count: int, work_prep: str = "one") -> QuantumCircuit:
    """
    Build the QPE circuit for order finding:
      - Prepare work register in |1⟩ or uniform superposition
      - Prepare counting register in superposition
      - Apply controlled modular multiplications
      - Apply inverse QFT
      - Measure counting register to get phase
    """
    # ⏱ Start timing total circuit build
    t0 = time.perf_counter()

    # Allocate qubits
    n_work = n_qubits_for(N)
    count = QuantumRegister(n_count, "count")
    work = QuantumRegister(n_work, "work")
    cl = ClassicalRegister(n_count, "c")
    qc = QuantumCircuit(count, work, cl)

    # Step 1: Prepare work register
    if work_prep == "one":
        qc.x(work[0])     # |1⟩
    else:
        for q in work:    # Uniform superposition
            qc.h(q)

    # Step 2: Put counting register in superposition
    qc.h(count)

    # Step 3: Controlled-U^(2^k) for each counting qubit
    for k in range(n_count):
        a_k = pow(a, 1 << k, N)
        if a_k != 1:
            qc.append(c_mult_mod_N(a_k, N, n_work), [count[k]] + list(work))

    # Step 4: Apply inverse Quantum Fourier Transform
    try:
        qft = QFT(n_count)
        qft_inv = qft.inverse()
    except TypeError:
        qft_inv = QFT(num_qubits=n_count, inverse=True)
    qc.append(qft_inv, count)

    # Step 5: Measure counting register
    qc.measure(count, cl)

    # ⏱ End timing
    t1 = time.perf_counter()
    log_elapsed("order_finding_qpe", t0, t1)
    return qc


# ---------------------------------------------------------
# Full Shor's algorithm driver
# ---------------------------------------------------------
def shor_factor_anyN(
    N: int,
    n_count: int | None = None,
    shots: int = 8192,
    max_trials: int = 10,
    work_prep: str = "one",
    verbose: bool = True,
):
    """
    Full pipeline for factoring N using Shor's Algorithm:
      1. Choose random coprime 'a'
      2. Build and run order-finding circuit using QPE
      3. Estimate order r from measured phase
      4. Attempt classical factorization using r
      5. Repeat for multiple trials if needed
    """
    # Create Aer simulator backend (CPU or GPU)
    backend = AerSimulator(method="statevector")

    # Trivial check: N must be >= 4
    if N < 4:
        print("[Info] N too small.")
        return None

    # Choose number of counting qubits if not specified
    if n_count is None:
        n_count = max(4, int(np.ceil(3 * log2(N))))

    # Find coprime candidates for 'a' (used in modular multiplication)
    coprimes = [a for a in range(2, N) if gcd(a, N) == 1]

    if verbose:
        print(f"[Setup] N={N}, n_count={n_count}, work_qubits={n_qubits_for(N)}")
        print(f"[Setup] Coprime a values: {coprimes[:max_trials]}")

    # -----------------------------------------------------
    # Trial loop: repeat with different coprime 'a' values
    # -----------------------------------------------------
    for idx, a in enumerate(coprimes[:max_trials], start=1):
        if verbose:
            print(f"\n[Trial {idx}] a={a}")

        # ⏱ Start timing the entire trial
        t0 = time.perf_counter()

        # 1. Build QPE order finding circuit
        qc = order_finding_qpe(a, N, n_count, work_prep)
        tqc = transpile(qc, backend, optimization_level=0, basis_gates=None)

        # 2. Run simulation (timed separately for clarity)
        run_start = time.perf_counter()
        result = backend.run(tqc, shots=shots).result()
        run_end = time.perf_counter()
        log_elapsed("backend.run", run_start, run_end)

        # 3. Classical post-processing
        counts = result.get_counts()
        # Top measurement outcomes (highest frequency)
        top_k = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:3]
        top_raw = top_k[0][0]
        # Optional: correct single-bit measurement errors
        top_corr = hamming_correct(top_raw)

        # Estimate phase as weighted average of top outcomes
        avg_int = sum(int(k, 2) * v for k, v in top_k) / sum(v for _, v in top_k)
        phase = avg_int / (1 << n_count)
        frac = continued_fraction_phase(round(phase, 8), max_denominator=2 * N)
        r = frac.denominator

        # Attempt classical factorization from r
        factors = try_factor_from_order(a, r, N)

        # ⏱ End timing for this trial
        log_elapsed(f"shor_factor_anyN (trial {idx})", t0, time.perf_counter())

        # 4. If successful, report and return factors
        if factors:
            p, q = factors
            print(f"[SUCCESS] {N} = {p} × {q} (a={a}, r={r})")
            return (p, q)

    # -----------------------------------------------------
    # If no factors found after all trials
    # -----------------------------------------------------
    print("[FAIL] No factors found.")
    return None


# ---------------------------------------------------------
# Main execution entrypoint
# ---------------------------------------------------------
if __name__ == "__main__":
    # Run Shor's algorithm with performance timing enabled.
    # This will:
    # - Build and run the QPE circuit
    # - Log timing for each major step
    # - Try to factor the integer N
    shor_factor_anyN(N=15, n_count=10, shots=8192, work_prep="one")
