import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from encrypt import encrypt_bytes, read_public_key

class TestRSA(unittest.TestCase):
    def setUp(self):
        # Use small prime numbers for testing
        self.test_n = 1333  # 31 * 43
        self.test_e = 143
        
    def test_public_key_reading(self):
        """Test reading public key from file"""
        try:
            n, e = read_public_key("public_key.txt")
            self.assertIsInstance(n, int)
            self.assertIsInstance(e, int)
            self.assertTrue(n > 0)
            self.assertTrue(e > 0)
        except FileNotFoundError:
            self.skipTest("public_key.txt not found")

if __name__ == '__main__':
    unittest.main()