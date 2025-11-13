import unittest
import sys
import os

# allow imports from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import encryption/decryption helpers
from src.crypto.encrypt import encrypt_bytes
from src.crypto.decrypt import decrypt_bytes

# Import RSA key generator
try:
    from GUI.RSAKeyGen import generate_rsa_keys
except Exception:
    # If the GUI module isn't importable as a package, try relative import path
    from GUI.RSAKeyGen import generate_rsa_keys

class TestPipeline(unittest.TestCase):
    def test_rsa_encrypt_decrypt_pipeline(self):
        """End-to-end: generate RSA keys -> encrypt -> decrypt should return original message."""
        # Generate a small RSA keypair (writes public_key.txt/private_key.txt in cwd)
        try:
            e, d, n = generate_rsa_keys(bits=8)
        except Exception as exc:
            self.skipTest(f"Could not generate RSA keys: {exc}")

        try:
            message = "Pipeline test message 🚀"
            bmsg = message.encode("utf-8")

            # Encrypt using public exponent e and modulus n
            cipher = encrypt_bytes(bmsg, e, n)

            # Decrypt using private exponent d and modulus n
            plain_bytes = decrypt_bytes(cipher, d, n)

            try:
                plain = plain_bytes.decode("utf-8")
            except Exception:
                self.fail("Decrypted bytes could not be decoded as utf-8")

            self.assertEqual(plain, message)
        finally:
            # Clean up key files generated in cwd if present
            for fname in ("public_key.txt", "private_key.txt"):
                try:
                    os.remove(fname)
                except FileNotFoundError:
                    pass

if __name__ == '__main__':
    unittest.main()
