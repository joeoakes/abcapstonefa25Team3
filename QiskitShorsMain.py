# Uncomment or comment this out
#!pip install qiskit qiskit-aer-gpu-cu11 pylatexenc --upgrade
from colorama import Fore, Style, init
import datetime
init(autoreset=True)

def Log(message, color=Fore.WHITE):
    """Prints a colored message to console and writes it to log.txt."""
    timestamp = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    formatted_message = f"{timestamp} {message}"

    # Print to console
    print(color + formatted_message + Style.RESET_ALL)

    # Write to log file
    with open("log.txt", "a", encoding="utf-8") as log_file:
        log_file.write(formatted_message + "\n")


import time
from math import gcd, log2, pi
from fractions import Fraction
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, transpile
from qiskit.quantum_info import Operator
from IPython.display import display, Markdown
from qiskit.visualization import plot_circuit_layout, plot_histogram
try:
    from qiskit.circuit.library import QFTGate as QFT  # not used now, but kept for compat
except Exception:
    from qiskit.circuit.library import QFT
from qiskit.circuit.library import UnitaryGate, PermutationGate
try:
    from qiskit_aer import AerSimulator
except ModuleNotFoundError:
    from qiskit.providers.aer import AerSimulator
import matplotlib.pyplot as plt


# colors
RED = "\033[91m"
BLUE = "\033[34m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RESET = "\033[0m"

# if True: do a shallow, single-level decompose for safety; if False: send as-is to Aer (Faster as False)
USE_FLATTEN = False

def visualize_qpe_circuit(qc, a, N, backend_mode="mpl", show_layout=False):
    """
    Visualizes the QPE circuit for Shor's algorithm in a compact, educational way.

    backend_mode = "mpl" -> Matplotlib visualization (recommended)
    backend_mode = "text" -> ASCII fallback
    show_layout = True to also display physical layout (transpiled map)
    """
    log(f"\n[Circuit Visualization] Showing QPE circuit for a={a}, N={N}", Fore.CYAN)

    try:
        # Transpile for a small fake backend layout to make it clean
        backend = AerSimulator(method="statevector")
        tqc = transpile(qc, backend=backend, optimization_level=1, seed_transpiler=7)

        # Render diagram
        if backend_mode == "mpl":
            display(Markdown(f"**Quantum Circuit for a={a}, N={N}**"))
            display(tqc.draw(output="mpl", fold=160))
        elif backend_mode == "text":
            print(tqc.draw(output="text", fold=160))
        else:
            raise ValueError("backend_mode must be 'mpl' or 'text'")

        # Optional layout map visualization
        if show_layout and backend_mode == "mpl":
            display(Markdown("**Qubit Layout Map (transpiled)**"))
            display(plot_circuit_layout(tqc, backend))
    except Exception as e:
        print(f"[Visualization Error] {e}")

def visualize_counts(counts, N, n_count, TOP_K=5):
    # Sort outcomes by integer value (LSB-first convention)
    items = sorted(counts.items(), key=lambda kv: int(kv[0][::-1], 2))
    xs = [int(k[::-1], 2) for k, _ in items]
    ys = [v for _, v in items]
    total = max(1, sum(ys))
    probs = [v / total for v in ys]

    # Normalize x-axis to fractional phase in [0, 1)
    phases = [x / (1 << n_count) for x in xs]

    plt.figure(figsize=(10, 4))
    plt.bar(phases, probs, width=1 / (1 << n_count), color="skyblue", edgecolor="black")
    plt.title(f"Phase histogram for N={N} (top {TOP_K} peaks highlighted)")
    plt.xlabel("Measured phase (fraction of 2pi)")
    plt.ylabel("Probability")

    # Highlight the top-K peaks in red
    top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:TOP_K]
    for k, v in top:
        idx = int(k[::-1], 2)
        phase = idx / (1 << n_count)
        plt.axvline(phase, color="red", linestyle="--", alpha=0.7)

    plt.show()


def n_qubits_for(N: int):
    # Return the number of work qubits needed to represent integers from 0 to N-1.
    return max(1, int(np.ceil(np.log2(N))))

def continued_fraction_phase(p, d):
    return Fraction(p).limit_denominator(d)

def try_factor_from_order(a, r, N):
    # Factor Ordering helper for Shor's Algorithm testing. (Validating if the R value is even nontrivial)
    if r <= 0:
        print(f"[Order Check] r={r} is invalid. Skipping.")
        return None
    if r % 2 != 0:
        print(f"[Order Check] r={r} is odd. Skipping.")
        return None

    # Compute a^(r/2) mod N
    x = pow(a, r // 2, N)

    # If x is congruent to 1 or N-1, it gives trivial factors
    if x in (1, 0, N - 1):
        print(f"[Trivial Result] a^({r//2}) mod {N} = {x}. No useful factors.")
        return None

    # Compute the candidate factors
    p = gcd(x - 1, N)
    q = gcd(x + 1, N)

    # Display intermediate gcd results for debugging
    log(f"[GCD Test] a={a}, r={r}, x={x}, gcd(x-1,N)={p}, gcd(x+1,N)={q}", Fore.CYAN)

    # Accept if either side is a nontrivial factor
    if 1 < p < N:
        log("P was nontrivial!", Fore.CYAN)
        return (p, N // p)
    if 1 < q < N:
        log("Q was nontrivial!", Fore.CYAN)
        return (q, N // q)

    log(f"[No Factors] gcd results not useful (p={p}, q={q}) for a={a}, r={r}", Fore.CYAN)
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


def c_mult_mod_N(a, N, n_work):
    # Build a one-controlled unitary that multiplies the work register by 'a modulo N'.
    # Fully Aer-compatible: build a block-diagonal controlled matrix manually.
    t0 = time.perf_counter()

    dim = 1 << n_work
    base = np.zeros((dim, dim), dtype=complex)

    # Build modular multiplication as a permutation matrix
    for y in range(N):
        base[(a * y) % N, y] = 1.0
    for y in range(N, dim):
        base[y, y] = 1.0  # keep extra states as identity

    # Construct controlled version manually as a block matrix:
    ctrl_dim = 2 * dim
    CU = np.eye(ctrl_dim, dtype=complex)
    CU[dim:, dim:] = base  # lower-right block = base operation

    gate = UnitaryGate(CU, label=f"CU_{a}_mod_{N}")

    t1 = time.perf_counter()
    log(f"{GREEN}[Built Controlled-U]{RESET} a={a}, N={N}, time={t1 - t0:.3f}s", Fore.CYAN)
    return gate


def inverse_qft_no_swaps(qc: QuantumCircuit, qubits):
    # Apply the inverse Quantum Fourier Transform to the given list of qubits.
    # This version does not include final swap gates. It is written for the convention where the first counting qubit represents the least significant bit.
    t0 = time.perf_counter()
    n = len(qubits)
    # Walk from the most significant position down to the least significant position
    for j in range(n - 1, -1, -1):
        # Apply controlled phase rotations with decreasing angle
        for m in range(j - 1, -1, -1):
            angle = -pi / (1 << (j - m))
            qc.cp(angle, qubits[m], qubits[j])
        # Apply a Hadamard to convert phase into amplitude on this qubit
        qc.h(qubits[j])
    t1 = time.perf_counter()
    log(f"{YELLOW}[Inverse QFT Complete]{RESET} on {n} qubits, time={t1 - t0:.3f}s", Fore.CYAN)


def order_finding_qpe(a, N, n_count, work_prep="one"):
    # Quantum Phase Estimation function!!!!!!!!
    build_start = time.perf_counter()
    n_work = n_qubits_for(N)
    count = QuantumRegister(n_count, "count")
    work  = QuantumRegister(n_work, "work")
    cl    = ClassicalRegister(n_count, "c")
    qc = QuantumCircuit(count, work, cl)

    # Prepare the work register initial state
    # If "one", set the integer value one by flipping the least significant work qubit.
    # Otherwise, create a uniform superposition across all work states.

    # It is more efficient to start from one because that is ultimately the condition we are looking to satisfy for our "R" value finder.
    t0 = time.perf_counter()
    if work_prep == "one":
        qc.x(work[0])  # prepare the integer value one
    else:
        for q in work:
            qc.h(q)
    t1 = time.perf_counter()
    log(f"{GREEN}[Work Register Prepared]{RESET} time={t1 - t0:.3f}s", Fore.CYAN)

    # Put the counting register into superposition
    t0 = time.perf_counter()
    qc.h(count)
    t1 = time.perf_counter()
    log(f"{GREEN}[Count Register Superposition]{RESET} time={t1 - t0:.3f}s",Fore.CYAN)

    # Apply controlled modular multiplications by powers of a
    # The k-th counting qubit controls multiplication by a^(2^k) modulo N
    t0 = time.perf_counter()
    seen = set()            # avoid building duplicate a^(2^k) mod N values inside this circuit
    local_gates = {}        # reuse within the same circuit build, no cross-trial caching
    for k in range(n_count):
        a_k = pow(a, 1 << k, N)
        if a_k == 1:
            continue
        if a_k not in local_gates:
            # build once for this circuit
            local_gates[a_k] = c_mult_mod_N(a_k, N, n_work)
        # append the gate for this control qubit
        qc.append(local_gates[a_k], [count[k]] + list(work))
    t1 = time.perf_counter()
    log(f"{GREEN}[Controlled Multiplications Done]{RESET} time={t1 - t0:.3f}s", Fore.CYAN)

    # Decode the phase with the inverse Quantum Fourier Transform
    inverse_qft_no_swaps(qc, list(count))

    # Measure the counting register to obtain a binary approximation of the phase
    t0 = time.perf_counter()
    qc.measure(count, cl)
    t1 = time.perf_counter()
    log(f"{GREEN}[Measurement Added]{RESET} time={t1 - t0:.3f}s", Fore.CYAN)

    build_end = time.perf_counter()
    Fore.CYAN(f"{CYAN}[QPE Circuit Built]{RESET} total build time={build_end - build_start:.3f}s\n", Fore.CYAN)
    return qc


def flatten_circuit(qc: QuantumCircuit) -> QuantumCircuit:
    # Recursively flatten the circuit. There are much better ways to do it, but I had to deal with headache errors from stuff like composite gates. (PLEASE FIX THIS LATER SAM) ((Fixed it later. Now it's optimized))
    # Optimized: either skip or only do a shallow decompose. Sending composite gates to Aer is fine and much faster.
    t0 = time.perf_counter()
    if USE_FLATTEN:
        out = qc.decompose(reps=1)  # shallow only
    else:
        out = qc                    # skip flattening entirely
    t1 = time.perf_counter()
    log(f"{YELLOW}[Circuit Flattened]{RESET} time={t1 - t0:.3f}s", Fore.CYAN)
    return out


def shor_factor_anyN(N: int,
                     n_count=None,
                     shots=4096,
                     work_prep="one",
                     a_trials: int = 3,
                     verbose=True,
                     visualize=False):
    total_start = time.perf_counter()

    if N < 4:
        log("[Info] N too small.", Fore.CYAN)
        return None
    # Auto select counting qubits if none are selected.
    if n_count is None:
        n_count = max(4, int(np.ceil(2 * log2(N))))

    # Build the simulator. Try to use the graphics processing unit. Fall back to the central processor.
    backend = AerSimulator(method="statevector")
    try:
        backend.set_options(
            device="GPU",
            precision="single",
            fusion_enable=True,
            max_parallel_threads=4,
            max_parallel_experiments=2
        )
        if verbose:
            log("[Backend] GPU statevector + fusion (single precision).", Fore.CYAN)
    except Exception:
        log("[Backend] CPU fallback.",Fore.CYAN)

    # Choose candidate a values
    primes = [p for p in primes_upto(N) if p % 2 and gcd(p, N) == 1 and p > 2][:a_trials]
    if not primes:
        primes = [a for a in range(3, N, 2) if gcd(a, N) == 1][:a_trials]

    if verbose:
        log(f"[Setup] N={N}, n_count={n_count}, work_qubits={n_qubits_for(N)}",Fore.CYAN)
        log(f"[Setup] Trying {len(primes)} a values (prime, odd, coprime): {primes}",Fore.CYAN)

    # Try each candidate a value
    for i, a in enumerate(primes, 1):
        if verbose:
            log(f"\n\n\n{CYAN}[Trial {i}/{len(primes)}] a={a}{RESET}", Fore.CYAN)
        trial_start = time.perf_counter()

        # Build and (optionally) flatten the circuit
        qc = order_finding_qpe(a, N, n_count, work_prep)
        if(visualize):
          visualize_qpe_circuit(qc, a, N, backend_mode="mpl", show_layout=False)
        tqc = flatten_circuit(qc)

        # Run simulation
        sim_start = time.perf_counter()
        result = backend.run(tqc, shots=shots).result()
        counts = result.get_counts()
        if(visualize):
          visualize_counts(counts, N, n_count, TOP_K=5)
        sim_end = time.perf_counter()
        log(f"{YELLOW}[Simulation Complete]{RESET} time={sim_end - sim_start:.3f}s", Fore.CYAN)

        # Analyze result
        # Identify the most frequent outcomes to reduce the effect of sampling noise
        analyze_start = time.perf_counter()
        TOP_K = 6  # number of most frequent bitstrings to test (you can change this)
        top = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:TOP_K]

        # Qiskit prints bitstrings with the most significant bit on the left.
        # Kind of annoying. We have to reverse the bitstring before converting to an integer.
        found = None
        picked_peak = None
        picked_frac = None
        picked_phase = None

        # Initialize safe defaults in case no valid order is found
        phase = 0.0
        frac = Fraction(0, 1)
        r = 0

        tested_rs = []  # keep a log of all r candidates for debugging

        # Sam Optimization: Instead of averaging, check each of the top peaks individually.
        # Each bitstring corresponds to a measured phase that may reveal the correct order (r).
        for j, (bitstr, weight) in enumerate(top, start=1):
            idx = int(bitstr[::-1], 2)
            phase = idx / float(1 << n_count)
            # Use a continued fraction with a denominator limited by twice N to estimate s over r
            frac = Fraction(round(phase, 8)).limit_denominator(2 * N)
            r_candidate = frac.denominator
            r = r_candidate  # update for reporting even if not useful
            tested_rs.append(r_candidate)

            # Always print the candidate phase and r, even if r == 1 or r is odd
            log(f"{BLUE}[Peak {j}/{TOP_K}]{RESET} bitstr={bitstr}  weight={weight, }  "
                  f"phase={phase:.6f}  = {frac}  -> r={r_candidate}", Fore.CYAN)

            # Try to turn the candidate order into nontrivial factors of N
            test = try_factor_from_order(a, r_candidate, N)
            if test:
                found = test
                picked_peak = bitstr
                picked_frac = frac
                picked_phase = phase
                break

        analyze_end = time.perf_counter()

        # If we found a valid factor
        if found:
            p, q = found
            top_raw = picked_peak
            top_raw_little = top_raw[::-1]
            log(f"{BLUE}[Result]{RESET} result(msb->lsb)={top_raw}  result(lsb->msb)={top_raw_little}", Fore.CYAN)
            log(f"{BLUE}[Phase]{RESET} phase={picked_phase:.6f}  = {picked_frac}  -> r={picked_frac.denominator}", Fore.CYAN)
            log(f"{YELLOW}[Analysis Time]{RESET} {analyze_end - analyze_start:.3f}s",Fore.CYAN)
            log(f"{GREEN}[Nontrivial Factors Found]{RESET} p={p}, q={q} from a={a}, r={picked_frac.denominator}",Fore.CYAN)
            trial_end = time.perf_counter()
            log(f"{CYAN}[Trial Time]{RESET} {trial_end - trial_start:.3f}s", Fore.CYAN)
            log(f"{GREEN}[SUCCESS]{RESET} {N} = {p} x {q}  (a={a}, r={picked_frac.denominator})", Fore.CYAN)
            total_end = time.perf_counter()
            log(f"{YELLOW}[TOTAL TIME]{RESET} {total_end - total_start:.3f}s\n", Fore.CYAN)
            return (p, q)

        # Otherwise, if no peaks factored N
        else:
            # Just print info from the most frequent bitstring for logging
            top_raw = top[0][0] if top else "?"
            top_raw_little = top_raw[::-1] if top else "?"
            log(f"{BLUE}[Result]{RESET} result(msb->lsb)={top_raw}  result(lsb->msb)={top_raw_little}", Fore.CYAN)
            log(f"{BLUE}[Phase]{RESET} phase={phase:.6f} = {frac} -> r={r}", Fore.CYAN)
            log(f"{YELLOW}[Analysis Time]{RESET} {analyze_end - analyze_start:.3f}s", Fore.CYAN)
            log(f"{YELLOW}[Tried r values]{RESET} {tested_rs}", Fore.CYAN)

        trial_end = time.perf_counter()
        log(f"{CYAN}[Trial Time]{RESET} {trial_end - trial_start:.3f}s", Fore.CYAN)
        if verbose:
            log(f"{BLUE}[Result]{RESET} No nontrivial factors for this a.", Fore.CYAN)

    total_end = time.perf_counter()
    log(f"{RED}[FAIL]{RESET} No factors found with current settings. Total run time={total_end - total_start:.3f}s", Fore.CYAN)
    return None


if __name__ == "__main__":
    # You can change N, the number of counting qubits, the number of shots, and the initial work register preparation here.
    # For larger N, consider reducing the number of counting qubits to keep runtime reasonable. Right now, it is automatic, but I will add a separate setting later.
    # N = 143: 10 qubits
    # N = 437: 
    shor_factor_anyN(N=143, n_count=10, shots=8024, work_prep="one", a_trials=10,visualize=False)
