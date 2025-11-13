import unittest
import sys
import os

# Ensure project root is in path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

# Correct import path (project uses src/crypto/encrypt.py)
from src.crypto.encrypt import encrypt_bytes, read_public_key


class TestRSA(unittest.TestCase):

    def setUp(self):
        # Test RSA parameters
        self.test_n = 1333  # 31 * 43
        self.test_e = 143

    def test_public_key_reading(self):
        """Test reading public key from file"""
        key_path = os.path.join("data", "keys", "public_key.txt")

        if not os.path.exists(key_path):
            self.skipTest("public_key.txt not found in data/keys")

        try:
            n, e = read_public_key(key_path)
        except Exception as exc:
            self.fail(f"Failed to read public key file: {exc}")

        # Validate types and values
        self.assertIsInstance(n, int)
        self.assertIsInstance(e, int)
        self.assertGreater(n, 0)
        self.assertGreater(e, 0)


if __name__ == '__main__':
    unittest.main()
