import unittest
import sys
import os
import re

# allow imports from project root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# project code is packaged under src/, import from there
from src.crypto.encrypt import encrypt_bytes, read_public_key
from src.crypto.decrypt import decrypt_bytes


def read_private_key(path="private_key.txt"):
    """Parse a private key file containing a tuple like (d, n) or two-line format.
    Returns (d, n).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    txt = open(path, "r", encoding="utf-8").read()
    # Try tuple format (d, n)
    m = re.search(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", txt)
    if m:
        return int(m.group(1)), int(m.group(2))
    # Try two-line d= / n=
    d = n = None
    for line in txt.splitlines():
        s = line.strip()
        if s.startswith("d="):
            d = int(s.split("=", 1)[1].strip())
        elif s.startswith("n="):
            n = int(s.split("=", 1)[1].strip())
    if d is not None and n is not None:
        return d, n
    raise ValueError("Could not parse private key file")


class TestRegression(unittest.TestCase):
    def test_encrypt_decrypt_roundtrip_with_stored_keys(self):
        """Regression: encrypt with public key and decrypt with private key should return original message."""
        # Read public key
        try:
            key_path = os.path.join("data", "keys", "public_key.txt")
            n_pub, e = read_public_key(key_path)
        except FileNotFoundError:
            self.skipTest("public_key.txt not found")
        except Exception as exc:
            self.fail(f"Failed to read public_key.txt: {exc}")

        # Read private key
        try:
            priv_path = os.path.join("data", "keys", "private_key.txt")
            d, n_priv = read_private_key(priv_path)
        except FileNotFoundError:
            self.skipTest("private_key.txt not found")
        except Exception as exc:
            self.fail(f"Failed to read private_key.txt: {exc}")

        # Keys must refer to same modulus
        self.assertEqual(n_pub, n_priv, "Public and private key modulus differ")

        message = "Regression test — RSA roundtrip"
        message_bytes = message.encode("utf-8")

        cipher = encrypt_bytes(message_bytes, e, n_pub)
        plaintext_bytes = decrypt_bytes(cipher, d, n_pub)

        try:
            plaintext = plaintext_bytes.decode("utf-8")
        except Exception:
            self.fail("Decrypted bytes could not be decoded to utf-8")

        self.assertEqual(plaintext, message)


if __name__ == "__main__":
    unittest.main()
