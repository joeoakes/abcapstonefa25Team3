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
    return max(1, int(np.ceil(np.log2(N))))

def continued_fraction_phase(p, d):
    return Fraction(p).limit_denominator(d)

def try_factor_from_order(a, r, N):
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
    """Return 1-controlled multiply-by-a (mod N) gate."""
    key = (a, N, n_work)
    if key in _CU_CACHE:
        return _CU_CACHE[key]

    dim = 1 << n_work
    U = np.zeros((dim, dim), dtype=complex)
    for y in range(dim):
        tgt = (a * y) % N if y < N else y
        U[tgt, y] = 1.0

    Ugate = UnitaryGate(U, label=f"{a}_mod_{N}")
    CU = Ugate.control(1)       # <-- real controlled version
    _CU_CACHE[key] = CU
    return CU


def inverse_qft_no_swaps(qc: QuantumCircuit, qubits):
    """Inverse QFT on 'qubits' with NO swaps, matching count[0] as the LSB."""
    # Apply in-place, little-endian: qubits[0] is LSB (k=0 control)
    n = len(qubits)
    for j in range(n - 1, -1, -1):  # from MSB down to LSB
        # controlled phase rotations
        for m in range(j - 1, -1, -1):
            angle = -pi / (1 << (j - m))
            qc.cp(angle, qubits[m], qubits[j])
        qc.h(qubits[j])

def order_finding_qpe(a, N, n_count, work_prep="one"):
    n_work = n_qubits_for(N)
    count = QuantumRegister(n_count, "count")
    work  = QuantumRegister(n_work, "work")
    cl    = ClassicalRegister(n_count, "c")
    qc = QuantumCircuit(count, work, cl)

    # Init work
    if work_prep == "one":
        qc.x(work[0])  # |1> assuming work[0] is LSB
    else:
        for q in work:
            qc.h(q)

    # Put count in superposition
    qc.h(count)

    # Controlled-U^(2^k) with count[k] as the control (k=0 is LSB)
    for k in range(n_count):
        a_k = pow(a, 1 << k, N)
        if a_k == 1:
            continue
        qc.append(c_mult_mod_N(a_k, N, n_work), [count[k]] + list(work))

    # Inverse QFT in little-endian (no swaps), consistent with loop above
    inverse_qft_no_swaps(qc, list(count))

    # Measure count
    qc.measure(count, cl)
    return qc

def flatten_circuit(qc: QuantumCircuit) -> QuantumCircuit:
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
    if n_count is None:
        n_count = max(4, int(np.ceil(2 * log2(N))))

    backend = AerSimulator(method="statevector")
    try:
        backend.set_options(device="GPU", precision="single", fusion_enable=True)
        if verbose:
            print("[Backend] GPU statevector + fusion (single precision).")
    except Exception:
        print("[Backend] CPU fallback.")

    primes = [p for p in primes_upto(N) if p % 2 and gcd(p, N) == 1 and p > 2][:max_trials]
    if not primes:
        primes = [a for a in range(3, N, 2) if gcd(a, N) == 1][:max_trials]

    if verbose:
        print(f"[Setup] N={N}, n_count={n_count}, work_qubits={n_qubits_for(N)}")
        print(f"[Setup] Trying a values (prime, odd, coprime, <=3): {primes}")

    for i, a in enumerate(primes, 1):
        if verbose:
            print(f"\n[Trial {i}/{len(primes)}] a={a}")

        qc = order_finding_qpe(a, N, n_count, work_prep)
        tqc = flatten_circuit(qc)

        result = backend.run(tqc, shots=shots).result()
        counts = result.get_counts()

        # Top-3 outcomes
        top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:3]
        top_raw = top[0][0]

        # IMPORTANT: reverse bitstring (Qiskit prints MSB -> LSB)
        top_raw_little = top_raw[::-1]

        # Weighted average (also reverse for each key)
        avg = sum(int(k[::-1], 2) * v for k, v in top) / sum(v for _, v in top)
        phase = avg / (1 << n_count)
        frac = continued_fraction_phase(round(phase, 8), d=2 * N)
        r = frac.denominator

        if verbose:
            print(f"  result(msb to lsb)={top_raw}  result(lsb to msb)={top_raw_little}")
            print(f"  phase = {phase:.6f}  = {frac}  to r={r}")

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
    shor_factor_anyN(N=35, n_count=10, shots=8192, work_prep="one", verbose=True)
