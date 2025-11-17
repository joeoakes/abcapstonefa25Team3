import unittest
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(PROJECT_ROOT)

from GUI.ClassicalShors import classical_shor, log_to_database


class TestIntegrationShorMultiple(unittest.TestCase):

    def run_case(self, N, expected_p, expected_q):
        """Helper that runs Shor for a single N."""
        p, q, N_val, a, r = classical_shor(N)

        # Basic correctness checks
        self.assertEqual(N_val, N)
        self.assertEqual({p, q}, {expected_p, expected_q})
        self.assertIsNotNone(r)

        # Log entry should append
        log_to_database(N, p, q, a, r)

        # Verify database file contains the correct entry
        db_path = os.path.join(PROJECT_ROOT, "shor_database.txt")
        with open(db_path, "r") as f:
            contents = f.read()
            self.assertIn(f"N={N}", contents)

    def test_21(self):
        # 21 = 3 × 7
        self.run_case(21, 3, 7)

    def test_33(self):
        # 33 = 3 × 11
        self.run_case(33, 3, 11)

    def test_55(self):
        # 55 = 5 × 11
        self.run_case(55, 5, 11)


if __name__ == "__main__":
    unittest.main()
