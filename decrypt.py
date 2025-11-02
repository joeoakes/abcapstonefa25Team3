# decrypt.py
# decrypts by factoring n with shor (quantum or classical)
# it tries quantum first, then classical (you can force with --engine)
#
# usage:
#   python encrypt.py --text "hello" > cipher.json
#   python decrypt.py --engine auto --cipher-json cipher.json
#
# or pass n/e on the command line and pipe json from encrypt:
#   python encrypt.py --n 3233 --e 17 --text "hi" | python decrypt.py

import argparse, json, os, sys
from math import gcd
from shors_adapter import classical_factor, quantum_factor

def egcd(a, b):
    if b == 0:
        return (a, 1, 0)
    g, x1, y1 = egcd(b, a % b)
    return (g, y1, x1 - (a // b) * y1)

def modinv(a, m):
    g, x, _ = egcd(a, m)
    if g != 1:
        raise ValueError("no modular inverse (e and phi(n) not coprime)")
    return x % m

def decrypt_bytes(cipher_list, d, n):
    return bytes(pow(c, d, n) for c in cipher_list)

def factor_n(n, engine="auto"):
    # try quantum first, then classical
    if engine in ("auto", "quantum"):
        try:
            print("[info] trying quantum shor...")
            return quantum_factor(n)
        except Exception as e:
            print(f"[warn] quantum failed: {e}")
            if engine == "quantum":
                raise
    if engine in ("auto", "classical"):
        try:
            print("[info] trying classical shor...")
            return classical_factor(n)
        except Exception as e:
            print(f"[warn] classical failed: {e}")
    raise RuntimeError("could not factor n with available engines")

def get_private_key(n, e, engine="auto"):
    p, q = factor_n(n, engine)
    if p * q != n or p in (0, 1) or q in (0, 1):
        raise ValueError("bad factors returned by factoring engine")
    if p == q:
        raise ValueError("n should be product of two DIFFERENT primes")
    phi = (p - 1) * (q - 1)
    if gcd(e, phi) != 1:
        raise ValueError("e must be coprime with phi(n)")
    d = modinv(e, phi)
    return p, q, d

def load_json(arg):
    # if arg is a path: open it; if not: treat it as json string; else read stdin
    if arg:
        if os.path.exists(arg):
            with open(arg, "r", encoding="utf-8") as f:
                return json.load(f)
        return json.loads(arg)
    return json.load(sys.stdin)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="RSA decrypt by factoring n (Shor)")
    p.add_argument("--n", type=int, help="optional if present in JSON")
    p.add_argument("--e", type=int, help="optional if present in JSON")
    p.add_argument("--engine", choices=["auto", "classical", "quantum"], default="auto")
    p.add_argument("--cipher-json", type=str, help="path or raw json. if missing, reads stdin")
    args = p.parse_args()

    data = load_json(args.cipher_json)
    n = args.n if args.n is not None else data.get("n")
    e = args.e if args.e is not None else data.get("e")
    if n is None or e is None:
        sys.exit("need n and e either in JSON or as flags")

    cipher = data.get("cipher")
    if not isinstance(cipher, list) or not all(isinstance(x, int) for x in cipher):
        sys.exit("JSON must have 'cipher': [ints]")

    print(f"[info] factoring n={n} (engine={args.engine})")
    p_val, q_val, d = get_private_key(n, e, engine=args.engine)
    print(f"[ok] factors found p={p_val}, q={q_val}")

    pt = decrypt_bytes(cipher, d, n)

    # print a friendly summary
    result = {
        "n": n,
        "e": e,
        "p": p_val,
        "q": q_val,
        "d": d,
        "plaintext_utf8": pt.decode("utf-8", errors="replace")
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
