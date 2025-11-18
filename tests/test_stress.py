import time
import GUI.RSAKeyGen
import random

# Imported external files to stress test
from GUI.ClassicalShors import classical_shor, generate_prime
from GUI.QiskitShorsMain import factor_N   # import Sam’s quantum shor

def rsa_stress_test(num_tests=500, bits=8):
#  RSA Stress Test
    print(f"\nRSA Stress Test: ")
    print(f"Running {num_tests} random key generations ({bits}-bit each)...\n")

    start_time = time.time()
    success_count = 0
    fail_count = 0
    times = []

    for i in range(num_tests):
        test_start = time.time()

        try:
            e, d, n = GUI.RSAKeyGen.generate_rsa_keys(bits=bits)
            success_count += 1
        except Exception as e:
            print(f"[Error] Test {i+1} failed: {e}")
            fail_count += 1
            continue

        test_end = time.time()
        elapsed = test_end - test_start
        times.append(elapsed)

        if (i + 1) % 50 == 0:
            print(f"Completed {i+1}/{num_tests} tests...")

    total_time = time.time() - start_time
    avg_time = sum(times) / len(times) if times else 0

    print("\nRSA Stress Test Results: ")
    print(f"Total tests: {num_tests}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Total runtime: {total_time:.3f} seconds")
    print(f"Average time per key: {avg_time:.5f} s")
    print(f"Fastest: {min(times):.5f} s")
    print(f"Slowest: {max(times):.5f} s")

#  Classical Shor's Stress test
def classical_shor_stress(num_tests=500, min_p=10, max_p=500):

    print(f"\nClassical Shor Stress Test ")
    print(f"Running {num_tests} random semiprimes (p,q range: {min_p}-{max_p})...\n")

    start_time = time.time()
    success_count = 0
    fail_count = 0
    times = []

    for i in range(num_tests):

        p = generate_prime(min_val=min_p, max_val=max_p)
        q = generate_prime(min_val=min_p, max_val=max_p)
        while p == q:
            q = generate_prime(min_val=min_p, max_val=max_p)

        N = p * q

        test_start = time.time()

        try:
            p_val, q_val, N_val, a, r = classical_shor(N)
        except Exception as e:
            print(f"[Error] Test {i+1} crashed on N={N}: {e}")
            fail_count += 1
            continue

        test_end = time.time()
        elapsed = test_end - test_start
        times.append(elapsed)

        if p_val is None or q_val is None or p_val * q_val != N:
            print(f"[Fail] Incorrect factors for N={N}")
            fail_count += 1
        else:
            success_count += 1

        if (i + 1) % 50 == 0:
            print(f"Completed {i+1}/{num_tests} tests...")

    total_time = time.time() - start_time
    avg_time = sum(times) / len(times) if times else 0

    print("\nClassical Shor Stress Test Results: ")
    print(f"Total tests: {num_tests}")
    print(f"Successful factorizations: {success_count}")
    print(f"Failed factorizations: {fail_count}")
    print(f"Total runtime: {total_time:.3f} seconds")
    print(f"Average time per factorization: {avg_time:.5f} s")
    print(f"Fastest: {min(times):.5f} s")
    print(f"Slowest: {max(times):.5f} s")

#  Quantum Shor's Algorithm Test
def qiskit_shor_stress(num_tests=20, min_p=5, max_p=20):
    """
    Stress tests Qiskit's Shor implementation by factoring random
    small semiprimes using the quantum simulator. N must stay small
    (< 437) due to Qiskit simulation limits.
    """

    print(f"\nQiskit Shor Stress Test: ")
    print(f"Running {num_tests} random semiprimes (range: {min_p}-{max_p})...\n")

    start_time = time.time()
    success_count = 0
    fail_count = 0
    runtimes = []

    for i in range(num_tests):

        # generate random primes
        p = generate_prime(min_val=min_p, max_val=max_p)
        q = generate_prime(min_val=min_p, max_val=max_p)
        while p == q:
            q = generate_prime(min_val=min_p, max_val=max_p)

        N = p * q
        if N > 430:
            continue   # skip values Qiskit's implementation cannot handle

        test_start = time.time()

        try:
            result = factor_N(
                N=N,
                n_count=None,        # auto
                shots=2048,          # decent balance between speed & accuracy
                work_prep="one",
                a_trials=5,
                visualize=False      # keep disabled for speed
            )
        except Exception as e:
            print(f"[Error] Test {i+1} crashed for N={N}: {e}")
            fail_count += 1
            continue

        elapsed = time.time() - test_start
        runtimes.append(elapsed)

        if result is None:
            print(f"[Fail] No result for N={N}")
            fail_count += 1
        else:
            found_p, found_q = result
            if found_p * found_q == N:
                success_count += 1
            else:
                print(f"[Fail] Incorrect factors for N={N}: got {result}")
                fail_count += 1

        if (i + 1) % 5 == 0:
            print(f"Completed {i+1}/{num_tests} tests...")

    total_time = time.time() - start_time
    avg = sum(runtimes) / len(runtimes) if runtimes else 0

    print("\nQiskit Shor Stress Test Results: ")
    print(f"Total tests: {num_tests}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Total runtime: {total_time:.3f} sec")
    print(f"Average time per test: {avg:.3f} sec")
    if runtimes:
        print(f"Fastest: {min(runtimes):.3f} sec")
        print(f"Slowest: {max(runtimes):.3f} sec")

#  Menu
def display_menu():
    print("\n============================")
    print("      STRESS TEST MENU      ")
    print("============================")
    print("1. RSA Stress Test")
    print("2. Classical Shor Stress Test")
    print("3. Qiskit Quantum Shor Stress Test")
    print("4. Run ALL Tests")
    print("5. Exit")
    print("============================")


def run_menu():
    while True:
        display_menu()
        choice = input("Enter your choice (1-5): ").strip()

        if choice == "1":
            rsa_stress_test()
        elif choice == "2":
            classical_shor_stress()
        elif choice == "3":
            qiskit_shor_stress()
        elif choice == "4":
            rsa_stress_test()
            classical_shor_stress()
            qiskit_shor_stress()
        elif choice == "5":
            print("Exiting stress test menu.")
            break
        else:
            print("Invalid selection. Please try again.\n")

if __name__ == "__main__":
    run_menu()
