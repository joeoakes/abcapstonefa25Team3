import unittest
import os
from src.database.shor_db import append_shor_record


class TestShorDatabase(unittest.TestCase):
    def setUp(self):
        self.test_file = "test_shor_database.txt"
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

    def test_append_record(self):
        out = append_shor_record(21, 3, 7, 5, 10)

        self.assertIn("N=21", out)
        self.assertIn("p=3", out)
        self.assertIn("q=7", out)
        self.assertIn("a=5", out)
        self.assertIn("r=10", out)

        with open("shor_database.txt", "r", encoding="utf-8") as f:
            txt = f.read()

        self.assertIn("N=21", txt)
        self.assertIn("p=3", txt)
        self.assertIn("q=7", txt)

if __name__ == "__main__":
    unittest.main()
