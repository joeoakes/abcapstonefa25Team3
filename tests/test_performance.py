import unittest
import time
import sys
import os

# Allow imports from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------- Project Imports ----------------
from src.crypto.encrypt import encrypt_bytes, read_public_key
from src.crypto.decrypt import decrypt_bytes
from GUI.RSAKeyGen import generate_rsa_keys
from GUI.ClassicalShors import classical_shor   # <-- your factoring function

# ============================================================
# PERFORMANCE TEST SUITE
# ============================================================

class TestPerformance(unittest.TestCase):

    # ----------------------------------------------------------
    # 1. FACTORING TIME (using your classical_shor)
    # ----------------------------------------------------------
    def test_factoring_time(self):
        """
        Performance: Ensure classical_shor factors small semiprimes fast.
        """
        test_values = [55, 77, 143, 221]  # small RSA-like N
        max_seconds = 1.0                 # threshold for regression

        for n in test_values:
            start = time.perf_counter()
            p, q, N, a, r = classical_shor(n)
            end = time.perf_counter()

            elapsed = end - start

            # correctness
            self.assertIsNotNone(p, f"classical_shor ({n}) failed: p=None")
            self.assertIsNotNone(q, f"classical_shor ({n}) failed: q=None")
            self.assertEqual(p * q, n, f"Incorrect factors for {n}")

            # performance
            self.assertLess(
                elapsed, 
                max_seconds,
                f"classical_shor({n}) took too long: {elapsed:.4f}s"
            )

            print(f"[Factoring] N={n} → p={p}, q={q}, time={elapsed:.4f}s")

    # ----------------------------------------------------------
    # 2. RSA KEY GENERATION TIME (uses your generate_rsa_keys)
    # ----------------------------------------------------------
    def test_rsa_keygen_time(self):
        """
        Performance: Test RSA key generation time for small bit sizes.
        """
        key_sizes = [8, 10, 12]           # your implementation uses toy sizes
        max_seconds = 1.0

        for bits in key_sizes:
            start = time.perf_counter()
            e, d, n = generate_rsa_keys(bits)
            end = time.perf_counter()

            elapsed = end - start

            self.assertIsInstance(e, int)
            self.assertIsInstance(d, int)
            self.assertIsInstance(n, int)

            self.assertLess(
                elapsed,
                max_seconds,
                f"generate_rsa_keys({bits}) too slow: {elapsed:.4f}s"
            )

            print(f"[KeyGen] {bits}-bit → {elapsed:.4f}s")

    # ----------------------------------------------------------
    # 3. RSA ENCRYPT/DECRYPT THROUGHPUT
    # ----------------------------------------------------------
    def test_encrypt_decrypt_throughput(self):
        """
        Performance: Ensure RSA encrypt/decrypt throughput is acceptable.
        """
        # --- Load public key ---
        try:
            pub_path = os.path.join("public_key.txt")
            n_pub, e = read_public_key(pub_path)
        except Exception:
            self.skipTest("public_key.txt missing — run keygen first.")

        # --- Load private key ---
        try:
            txt = open("private_key.txt", "r").read().strip()
            d = int(txt.split(",")[0].strip("()"))
        except Exception:
            self.skipTest("private_key.txt missing — run keygen first.")

        # --- Test messages ---
        messages = [b"perf-test-" + bytes([i]) for i in range(200)]

        # Encryption speed
        start_enc = time.perf_counter()
        ciphertexts = [encrypt_bytes(m, e, n_pub) for m in messages]
        end_enc = time.perf_counter()

        enc_rate = len(messages) / (end_enc - start_enc)

        # Decryption speed
        start_dec = time.perf_counter()
        plains = [decrypt_bytes(c, d, n_pub) for c in ciphertexts]
        end_dec = time.perf_counter()

        dec_rate = len(messages) / (end_dec - start_dec)

        # thresholds (tune if needed)
        self.assertGreater(enc_rate, 50, f"Encryption too slow: {enc_rate:.2f} msg/s")
        self.assertGreater(dec_rate, 50, f"Decryption too slow: {dec_rate:.2f} msg/s")

        print(f"[Throughput] Encrypt={enc_rate:.2f} msg/s | Decrypt={dec_rate:.2f} msg/s")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    unittest.main()
