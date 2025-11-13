import time
import GUI.RSAKeyGen
import random

def rsa_stress_test(num_tests=500, bits=8):

    print(f"RSA Stress Test")
    print(f"Running {num_tests} random key generations ({bits}-bit each)...\n")

    #Begins timer and count of the tests
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

        # Print progress every 50 tests
        if (i + 1) % 50 == 0:
            print(f"Completed {i+1}/{num_tests} tests...")

    total_time = time.time() - start_time
    avg_time = sum(times) / len(times) if times else 0
    max_time = max(times) if times else 0
    min_time = min(times) if times else 0

    #Stress Results
    print("\nStress Test Results")
    print(f"Total tests: {num_tests}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    print(f"Total runtime: {total_time:.3f} seconds")
    print(f"Average time per key: {avg_time:.5f} s")
    print(f"Fastest: {min_time:.5f} s")
    print(f"Slowest: {max_time:.5f} s")



if __name__ == "__main__":
    # You can change these values for longer tests or bigger keys
    rsa_stress_test(num_tests=500, bits=8)
