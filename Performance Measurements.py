# =========================================================
# shor_from_scratch_qiskit21_hamming_TIMED_ELAPSED_ONLY.py
# Shor's Algorithm from scratch with detailed performance timing
# Compatible with Qiskit >= 2.1
# Logs elapsed time per major function and per trial
# =========================================================

# !pip install qiskit qiskit-aer-gpu-cu11 --upgrade  # Uncomment if needed

import time                       # Used for high-resolution performance measurement
from math import gcd, log2
from fractions import Fraction
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile

# Quantum Fourier Transform
try:
    from qiskit.circuit.library import QFTGate as QFT
except Exception:
    from qiskit.circuit.library import QFT

# UnitaryGate: used to build controlled modular multiplication
from qiskit.circuit.library import UnitaryGate

# AerSimulator: used for backend execution (CPU or GPU)
try:
    from qiskit_aer import AerSimulator
except ModuleNotFoundError:
    from qiskit.providers.aer import AerSimulator


# ---------------------------------------------------------
# Performance measurement helper
# ---------------------------------------------------------
def log_elapsed(label, start, end):
    """
    Logs elapsed time for a labeled section of the code.
    This allows fine-grained visibility into which parts
    of the algorithm are the most time-consuming.

    Parameters:
        label (str): descriptive name of the function or code section
        start (float): timestamp taken with time.perf_counter() at the beginning
        end (float): timestamp taken with time.perf_counter() at the end
    """
    print(f"⏳ {label} elapsed: {end - start:.6f} s")


def n_qubits_for(N: int) -> int:
    # ⏱ Start timing before the computation
    t0 = time.perf_counter()
    result = max(1, int(np.ceil(np.log2(N))))
    # ⏱ End timing after the computation
    t1 = time.perf_counter()
    log_elapsed("n_qubits_for", t0, t1)
    return result


def continued_fraction_phase(phase: float, max_denominator: int):
    # ⏱ Timing this classical phase post-processing step
    # This step is typically fast but gives a baseline vs quantum parts
    t0 = time.perf_counter()
    result = Fraction(phase).limit_denominator(max_denominator)
    t1 = time.perf_counter()
    log_elapsed("continued_fraction_phase", t0, t1)
    return result


def try_factor_from_order(a: int, r: int, N: int):
    # ⏱ Measure how long the classical factor check takes.
    # This helps highlight that classical post-processing is negligible compared to QPE.
    t0 = time.perf_counter()
    result = None
    if r % 2 == 0:
        x = pow(a, r // 2, N)
        if x not in (1, 0, N - 1):
            p = gcd(x - 1, N)
            q = gcd(x + 1, N)
            if 1 < p < N and 1 < q < N:
                result = (p, q)
    t1 = time.perf_counter()
    log_elapsed("try_factor_from_order", t0, t1)
    return result


def hamming_correct(meas: str) -> str:
    # ⏱ Timing this error-correction step allows us to see
    # how much (or little) classical correction contributes to total runtime.
    t0 = time.perf_counter()
    bits = [int(b) for b in meas[::-1]]
    n = len(bits)
    corrected = bits[:]
    for start in range(0, n, 7):
        seg = bits[start:start + 7]
        if len(seg) < 7:
            break
        p1 = (seg[0] ^ seg[2] ^ seg[4] ^ seg[6])
        p2 = (seg[1] ^ seg[2] ^ seg[5] ^ seg[6])
        p3 = (seg[3] ^ seg[4] ^ seg[5] ^ seg[6])
        syndrome = (p3 << 2) | (p2 << 1) | p1
        if syndrome != 0 and syndrome <= len(seg):
            seg[syndrome - 1] ^= 1
        corrected[start:start + 7] = seg
    result = ''.join(str(b) for b in corrected[::-1])
    t1 = time.perf_counter()
    log_elapsed("hamming_correct", t0, t1)
    return result


def c_mult_mod_N(a: int, N: int, n_work: int):
    # ⏱ This is one of the more computationally expensive classical pre-processing steps.
    # It builds a controlled modular multiplication gate from scratch.
    t0 = time.perf_counter()
    dim = 1 << n_work
    U = np.zeros((dim, dim), dtype=complex)
    for y in range(dim):
        tgt = (a * y) % N if y < N else y
        U[tgt, y] = 1.0
    Ugate = UnitaryGate(U, label=f"{a}_mod_{N}")
    CU = Ugate.control(1)
    t1 = time.perf_counter()
    log_elapsed("c_mult_mod_N", t0, t1)
    return CU


def order_finding_qpe(a: int, N: int, n_count: int, work_prep: str = "one") -> QuantumCircuit:
    # ⏱ Timing the full circuit construction for QPE.
    # This includes controlled modular multiplication and inverse QFT.
    t0 = time.perf_counter()

    n_work = n_qubits_for(N)
    count = QuantumRegister(n_count, "count")
    work = QuantumRegister(n_work, "work")
    cl = ClassicalRegister(n_count, "c")
    qc = QuantumCircuit(count, work, cl)

    # Initial state prep
    if work_prep == "one":
        qc.x(work[0])
    else:
        for q in work:
            qc.h(q)

    # Superposition on counting register
    qc.h(count)

    # Controlled-U^(2^k) for each counting qubit
    for k in range(n_count):
        a_k = pow(a, 1 << k, N)
        if a_k != 1:
            qc.append(c_mult_mod_N(a_k, N, n_work), [count[k]] + list(work))

    # Apply inverse QFT
    try:
        qft = QFT(n_count)
        qft_inv = qft.inverse()
    except TypeError:
        qft_inv = QFT(num_qubits=n_count, inverse=True)
    qc.append(qft_inv, count)

    # Final measurement
    qc.measure(count, cl)

    t1 = time.perf_counter()
    log_elapsed("order_finding_qpe", t0, t1)
    return qc


def shor_factor_anyN(
    N: int,
    n_count: int | None = None,
    shots: int = 8192,
    max_trials: int = 10,
    work_prep: str = "one",
    verbose: bool = True,
):
    backend = AerSimulator(method="statevector")
    if N < 4:
        print("[Info] N too small.")
        return None
    if n_count is None:
        n_count = max(4, int(np.ceil(3 * log2(N))))

    coprimes = [a for a in range(2, N) if gcd(a, N) == 1]

    if verbose:
        print(f"[Setup] N={N}, n_count={n_count}, work_qubits={n_qubits_for(N)}")
        print(f"[Setup] Coprime a values: {coprimes[:max_trials]}")

    for idx, a in enumerate(coprimes[:max_trials], start=1):
        if verbose:
            print(f"\n[Trial {idx}] a={a}")

        # ⏱ Start timing the full trial: includes circuit build + run + post-processing
        t0 = time.perf_counter()

        # Circuit build (QPE)
        qc = order_finding_qpe(a, N, n_count, work_prep)
        tqc = transpile(qc, backend, optimization_level=0, basis_gates=None)

        # ⏱ Backend simulation timing is measured separately
        # This isolates quantum simulation cost from classical prep.
        run_start = time.perf_counter()
        result = backend.run(tqc, shots=shots).result()
        run_end = time.perf_counter()
        log_elapsed("backend.run", run_start, run_end)

        # Classical post-processing
        counts = result.get_counts()
        top_k = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:3]
        top_raw = top_k[0][0]
        top_corr = hamming_correct(top_raw)

        avg_int = sum(int(k, 2) * v for k, v in top_k) / sum(v for _, v in top_k)
        phase = avg_int / (1 << n_count)
        frac = continued_fraction_phase(round(phase, 8), max_denominator=2 * N)
        r = frac.denominator
        factors = try_factor_from_order(a, r, N)

        # ⏱ End timing for this trial
        log_elapsed(f"shor_factor_anyN (trial {idx})", t0, time.perf_counter())

        if factors:
            p, q = factors
            print(f"[SUCCESS] {N} = {p} × {q} (a={a}, r={r})")
            return (p, q)

    print("[FAIL] No factors found.")
    return None


if __name__ == "__main__":
    # Run Shor's algorithm with performance timing enabled
    # This prints the time taken for each major step and each trial
    shor_factor_anyN(N=15, n_count=10, shots=8192, work_prep="one")
