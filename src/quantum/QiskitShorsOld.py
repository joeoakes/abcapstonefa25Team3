# shor_from_scratch_qiskit21_hamming.py
# Compatible with Qiskit >= 2.1 (Colab-safe)
# Shor's Algorithm from scratch + Hamming-style error correction on measured output

# Uncomment or comment this out
!pip install qiskit qiskit-aer-gpu-cu11 --upgrade

from math import gcd, log2
from fractions import Fraction
import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
try:
    from qiskit.circuit.library import QFTGate as QFT
except Exception:
    from qiskit.circuit.library import QFT
from qiskit.circuit.library import UnitaryGate

try:
    from qiskit_aer import AerSimulator
except ModuleNotFoundError:
    from qiskit.providers.aer import AerSimulator


def n_qubits_for(N: int) -> int:
    return max(1, int(np.ceil(np.log2(N))))


def continued_fraction_phase(phase: float, max_denominator: int):
    return Fraction(phase).limit_denominator(max_denominator)


def try_factor_from_order(a: int, r: int, N: int):
    if r % 2 != 0:
        return None
    x = pow(a, r // 2, N)
    if x in (1, 0, N - 1):
        return None
    p = gcd(x - 1, N)
    q = gcd(x + 1, N)
    if 1 < p < N and 1 < q < N:
        return (p, q)
    return None


def hamming_correct(meas: str) -> str:
    """
    Classical Hamming(7,4)-style single-bit correction applied in 7-bit chunks.
    For non-multiples of 7, leaves remaining bits unchanged.
    """
    bits = [int(b) for b in meas[::-1]]  # reverse to LSB-first
    n = len(bits)
    corrected = bits[:]

    for start in range(0, n, 7):
        seg = bits[start:start + 7]
        if len(seg) < 7:
            break

        # Compute parity bits
        p1 = (seg[0] ^ seg[2] ^ seg[4] ^ seg[6])
        p2 = (seg[1] ^ seg[2] ^ seg[5] ^ seg[6])
        p3 = (seg[3] ^ seg[4] ^ seg[5] ^ seg[6])
        syndrome = (p3 << 2) | (p2 << 1) | p1

        if syndrome != 0 and syndrome <= len(seg):
            seg[syndrome - 1] ^= 1  # flip erroneous bit

        corrected[start:start + 7] = seg

    return ''.join(str(b) for b in corrected[::-1])  # restore MSB-first


def c_mult_mod_N(a: int, N: int, n_work: int):
    dim = 1 << n_work
    U = np.zeros((dim, dim), dtype=complex)

    for y in range(dim):
        tgt = (a * y) % N if y < N else y
        U[tgt, y] = 1.0

    Ugate = UnitaryGate(U, label=f"{a}_mod_{N}")
    return Ugate.control(1)


def order_finding_qpe(a: int, N: int, n_count: int, work_prep: str = "one") -> QuantumCircuit:
    if gcd(a, N) != 1:
        raise ValueError(f"'a'={a} not coprime with N={N}")

    n_work = n_qubits_for(N)

    count = QuantumRegister(n_count, "count")
    work = QuantumRegister(n_work, "work")
    cl = ClassicalRegister(n_count, "c")
    qc = QuantumCircuit(count, work, cl)

    # prepare work register
    if work_prep == "one":
        qc.x(work[0])
    else:
        for q in work:
            qc.h(q)

    # counting register
    qc.h(count)

    for k in range(n_count):
        a_k = pow(a, 1 << k, N)
        if a_k == 1:
            continue
        cU = c_mult_mod_N(a_k, N, n_work)
        qc.append(cU, [count[k]] + list(work))

    # Inverse QFT (Qiskit 2.x)
    try:
        qft = QFT(n_count)
        qft_inv = qft.inverse()
    except TypeError:
        qft_inv = QFT(num_qubits=n_count, inverse=True)

    qc.append(qft_inv, count)
    qc.measure(count, cl)
    return qc


def shor_factor_anyN(
    N: int,
    n_count: int | None = None,
    shots: int = 8192,
    max_trials: int = 10,
    work_prep: str = "one",
    verbose: bool = True,
):
    if N < 4:
        print("[Info] N too small.")
        return None

    if n_count is None:
        n_count = max(4, int(np.ceil(3 * log2(N))))

    backend = AerSimulator(method="statevector")
    coprimes = [a for a in range(2, N) if gcd(a, N) == 1]

    if verbose:
        print(f"[Setup] N={N}, n_count={n_count}, work_qubits={n_qubits_for(N)}")
        print(f"[Setup] Coprime a values: {coprimes[:max_trials]}")

    for idx, a in enumerate(coprimes[:max_trials], start=1):
        if verbose:
            print(f"\n[Trial {idx}] a={a}")

        qc = order_finding_qpe(a, N, n_count, work_prep)
        tqc = transpile(qc, backend, optimization_level=0, basis_gates=None)
        result = backend.run(tqc, shots=shots).result()
        counts = result.get_counts()

        # Top 3 most frequent results
        top_k = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:3]
        top_raw = top_k[0][0]
        top_corr = hamming_correct(top_raw)

        if verbose and top_corr != top_raw:
            print(f" [Hamming correction] {top_raw} → {top_corr}")

        # Compute phase from weighted average of top 3
        avg_int = sum(int(k, 2) * v for k, v in top_k) / sum(v for _, v in top_k)
        phase = avg_int / (1 << n_count)
        frac = continued_fraction_phase(round(phase, 8), max_denominator=2 * N)
        r = frac.denominator

        if verbose:
            print(f" result={top_corr} phase={phase:.6f} = {frac} so r={r}")

        factors = try_factor_from_order(a, r, N)
        if factors:
            p, q = factors
            print(f"[SUCCESS] {N} = {p} × {q} (a={a}, r={r})")
            return (p, q)

    print(" No nontrivial factors.")
    print("[FAIL] No factors found. Try work_prep='h' or increase n_count/shots.")
    return None


if __name__ == "__main__":
    shor_factor_anyN(N=15, n_count=10, shots=8192, work_prep="one")
