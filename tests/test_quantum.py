import unittest
import sys
import os
import numpy as np

# Allow importing the Shor module from the project
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GUI.QiskitShorsMain import (
    n_qubits_for,
    primes_upto,
    pick_diverse_a_values,
    try_factor_from_order,
    c_mult_mod_N
)

from src.crypto.encrypt import encrypt_bytes, read_public_key
from src.crypto.decrypt import decrypt_bytes, get_pk_from_pq


class TestShorNonQuantum(unittest.TestCase):

    # Smoke testing and sanity testing
    def test_smoke_imports(self):
        print("Running Smoke Test: import verification")
        self.assertTrue(True)

    def test_sanity_functions_exist(self):
        print("Running Sanity Test: ensure all major functions exist")
        self.assertTrue(callable(n_qubits_for))
        self.assertTrue(callable(primes_upto))
        self.assertTrue(callable(pick_diverse_a_values))
        self.assertTrue(callable(try_factor_from_order))

    # Accessibility testing
    def test_accessibility_input_validation(self):
        print("Running Accessibility Test: invalid input handling")
        with self.assertRaises(TypeError):
            n_qubits_for("abc")

        with self.assertRaises(ValueError):
            pick_diverse_a_values(1)

        res = try_factor_from_order(2, -3, 21)
        self.assertIsNone(res)

    # Regression testing
    def test_regression_primes_list(self):
        print("Running Regression Test: prime list stability")
        expected = [2, 3, 5, 7, 11, 13, 17, 19]
        self.assertEqual(primes_upto(20), expected)

    def test_regression_qubit_count(self):
        print("Running Regression Test: qubit count stability")
        self.assertEqual(n_qubits_for(15), 4)
        self.assertEqual(n_qubits_for(8), 3)

    # Installation testing
    def test_installation_numpy_basic(self):
        print("Running Installation Test: numpy sanity check")
        m = np.eye(3)
        self.assertEqual(m.shape, (3, 3))

    def test_installation_unitary_matrix(self):
        print("Running Installation Test: unitary matrix validation")
        N = 7
        a = 3
        n_work = 3
        gate = c_mult_mod_N(a, N, n_work)
        mat = gate.to_matrix()
        identity = mat @ mat.conj().T
        self.assertTrue(np.allclose(identity, np.eye(2 ** (n_work + 1))))

    # Stress testing and load testing
    def test_stress_pick_a_values(self):
        print("Running Stress Test: pick_diverse_a_values with large N")
        arr = pick_diverse_a_values(9973, a_trials=25, seed=10)
        self.assertTrue(len(arr) <= 25)

    def test_stress_prime_generation(self):
        print("Running Stress Test: primes_upto for high range")
        primes = primes_upto(5000)
        self.assertGreater(len(primes), 600)

    # Alpha and Beta testing
    def test_alpha_factor_order(self):
        print("Running Alpha Test: simple known factor recovery path")
        result = try_factor_from_order(2, 4, 15)
        self.assertIn(result, [(3, 5), (5, 3)])

    def test_beta_diverse_values(self):
        print("Running Beta Test: diverse a values remain coprime")
        arr = pick_diverse_a_values(55, a_trials=6, seed=2)
        for a in arr:
            self.assertEqual(np.gcd(a, 55), 1)

    # User Acceptance Testing
    def test_user_acceptance_normal_flow(self):
        print("Running User Acceptance Test: simulate front end usage flow")
        N = 91
        q = n_qubits_for(N)
        self.assertGreaterEqual(q, 7)

        arr = pick_diverse_a_values(N, a_trials=5)
        self.assertLessEqual(len(arr), 5)

        trial = try_factor_from_order(10, 6, N)
        self.assertTrue(trial is None or isinstance(trial, tuple))


class PipelineTest(unittest.TestCase):

    def test_get_private_key_flow_expect_result(self):
        # Test that the decrypt function works with the quantum shors output, which is either (p,q) or None
        # without running the quantum factoring engine
        print("Running Crypto Test: deriving private key via factor_N")

        testoutput = (43,59)
        p = testoutput[0]
        q = testoutput[1]
        n = p * q
        e = 17

        # Patch factor_n so this test does not call quantum code
        def fake_factor(nval, engine="auto"):
            return (p, q)

        original_factor_n = globals().get("factor_n")
        globals()["factor_n"] = fake_factor

        d = get_pk_from_pq(p,q,e)
        print(d)

        self.assertTrue(isinstance(d, int))

        # Restore original factoring function
        globals()["factor_n"] = original_factor_n

    def test_get_private_key_flow_expect_failure(self):
        # Test that the decrypt function works with the quantum shors output, which is either (p,q) or None
        # without running the quantum factoring engine
        print("Running Crypto Test: deriving private key via factor_N")

        testoutput = None
        p = None
        q = None
        n = None
        e = 17

        # Patch factor_n so this test does not call quantum code
        def fake_factor(nval, engine="auto"):
            return (p, q)

        original_factor_n = globals().get("factor_n")
        globals()["factor_n"] = fake_factor

        d = get_pk_from_pq(p,q,e)
        print(d)

        self.assertFalse(isinstance(d, int))

        # Restore original factoring function
        globals()["factor_n"] = original_factor_n



if __name__ == "__main__":
    print("Starting Non Quantum Shor Tests")
    unittest.main()