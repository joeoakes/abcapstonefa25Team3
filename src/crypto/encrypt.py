# encrypt.py
# simple RSA encrypt for our project (byte-by-byte demo)
# usage:
#   python encrypt.py --text "hello"         (reads n,e from public_key.txt)
#   python encrypt.py --n 3233 --e 17 --text "hi"

import argparse, json, sys, os

def read_public_key(path="public_key.txt"):
    """Accepts either format:
       A) Two-line:
          n=1333
          e=143
       B) Tuple:
          (143, 1333)   # (e, n)
    """
    import os, re
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. pass --n and --e instead.")

    txt = open(path, "r", encoding="utf-8").read().strip()

    # Format A: two lines n=/e=
    if "n=" in txt and "e=" in txt:
        n = e = None
        for line in txt.splitlines():
            s = line.strip()
            if s.startswith("n="):
                n = int(s.split("=", 1)[1].strip())
            elif s.startswith("e="):
                e = int(s.split("=", 1)[1].strip())
        if n is None or e is None:
            raise ValueError("public_key.txt must have lines like n=... and e=...")
        return n, e

    # Format B: tuple (e, n)
    m = re.search(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", txt)
    if m:
        e = int(m.group(1))
        n = int(m.group(2))
        return n, e

    raise ValueError("public_key.txt must be either 'n=...\\ne=...' or '(e, n)'.")


def encrypt_bytes(bdata, e, n):
    # WARNING: this is just a class demo (no padding etc.)
    return [pow(b, e, n) for b in bdata]

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="RSA encrypt (very simple demo)")
    p.add_argument("--n", type=int)
    p.add_argument("--e", type=int)
    p.add_argument("--text", type=str, help="plaintext string (utf-8). if omitted, reads stdin")
    p.add_argument("--pubkey", type=str, default="public_key.txt")
    args = p.parse_args()

    if args.n is None or args.e is None:
        n, e = read_public_key(args.pubkey)
    else:
        n, e = args.n, args.e

    plaintext = (args.text if args.text is not None else sys.stdin.read())
    cipher = encrypt_bytes(plaintext.encode("utf-8"), e, n)

    print(json.dumps({"n": n, "e": e, "cipher": cipher}, ensure_ascii=False))
