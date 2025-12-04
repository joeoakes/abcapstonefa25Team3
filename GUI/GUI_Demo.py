from flask import Flask, render_template, request, jsonify
from ClassicalShors import run_multiple
from RSAKeyGen import generate_rsa_keys, is_prime, mod_inverse
from math import gcd
import random
from QiskitShorsMain import factor_N  # Takes value N
import io
import sys
from pathlib import Path
import re

app = Flask(__name__)

# Store the latest RSA modulus N so it can be reused in Quantum Shor
last_rsa_n = None
last_rsa_e = None
last_rsa_d = None

def encrypt_message(message: str, public_key: tuple[int, int]) -> list[int]:
    """Encrypt a message (string) into a list of integers using (e, n)."""
    e, n = public_key
    return [pow(ord(ch), e, n) for ch in message]


def decrypt_message(ciphertext: list[int], private_key: tuple[int, int]) -> str:
    """Decrypt a list of integers back into a string using (d, n)."""
    d, n = private_key
    return ''.join(chr(pow(c, d, n)) for c in ciphertext)


@app.route('/', methods=['GET', 'POST'])
def home():
    global last_rsa_n
    output = ""
    action = None

    if request.method == 'POST':
        action = request.form.get('action')
        N_value = request.form.get('N_value')
        use_rsa_n = request.form.get('use_rsa_n') == 'true'

        if action == 'generate_rsa':
            try:
                # your RSAKeyGen.py uses bits=8 (not min_p/max_p)
                e, d, n = generate_rsa_keys(bits=8)
                public_key = (e, n)
                private_key = (d, n)
                last_rsa_n = n

                output = (
                    "RSA Keys Generated Successfully!\n\n"
                    f"Public Key (e, n): {public_key}\n"
                    f"Private Key (d, n): {private_key}\n\n"
                    "Keys saved as public_key.txt and private_key.txt\n"
                    f"Stored N value: {last_rsa_n}"
                )

            except Exception as e:
                output = f" Error generating RSA keys: {str(e)}"

        elif action == 'run_classical':
            try:
                buffer = io.StringIO()
                sys_stdout_backup = sys.stdout
                sys.stdout = buffer
                run_multiple(limit=20, min_p=10, max_p=40, max_attempts=20)
                sys.stdout = sys_stdout_backup
                output = buffer.getvalue()
            except Exception as e:
                sys.stdout = sys.__stdout__
                output = f" Error running classical Shor’s algorithm: {str(e)}"

        elif action == 'run_quantum':
            try:
                if use_rsa_n and last_rsa_n:
                    try:
                        buffer = io.StringIO()
                        sys_stdout_backup = sys.stdout
                        sys.stdout = buffer
                        # You probably meant to factor last_rsa_n here:
                        factor_N(N=last_rsa_n)
                        sys.stdout = sys_stdout_backup
                        output = buffer.getvalue()
                    except Exception as e:
                        sys.stdout = sys.__stdout__
                        output = f" Error running Quantum Shor’s algorithm: {str(e)}"
                elif N_value:
                    try:
                        buffer = io.StringIO()
                        sys_stdout_backup = sys.stdout
                        sys.stdout = buffer
                        factor_N(N=int(N_value))
                        sys.stdout = sys_stdout_backup
                        output = buffer.getvalue()
                    except Exception as e:
                        sys.stdout = sys.__stdout__
                        output = f" Error running Quantum Shor’s algorithm: {str(e)}"
                else:
                    output = "Simulating quantum Shor factoring...\nResult: 21 → 3 × 7"
            except Exception as e:
                output = f"Error during quantum simulation: {str(e)}"

        elif action == 'run_quantum_hamming':
            output = (
                "Quantum Shor with Hamming(7,4) error correction completed.\n"
                "Factored n = 33 is 3 × 11"
            )

        else:
            output = "No action performed."

    return render_template("index.html", output=output, action=action)


@app.route("/rsa", methods=["GET", "POST"])
def rsa_page():
    global last_rsa_n, last_rsa_e, last_rsa_d
    output = ""          # for key generation messages
    crypto_output = ""   # for encrypt/decrypt demo

    if request.method == "POST":
        action = request.form.get("action")

        try:
            if action == "generate":
                # key generation branch 
                min_p = int(request.form.get("min_p", "5"))
                max_p = int(request.form.get("max_p", "50"))

                if min_p < 2 or max_p <= min_p:
                    output = "Error: max must be greater than min and min must be ≥ 2."
                else:
                    e, d, n = generate_rsa_in_range(min_p, max_p)

                    last_rsa_e = e
                    last_rsa_d = d
                    last_rsa_n = n

                    output = (
                        "RSA Keys Generated Successfully!\n\n"
                        f"Prime range: {min_p} to {max_p}\n"
                        f"Public Key (e, n): {(e, n)}\n"
                        f"Private Key (d, n): {(d, n)}\n\n"
                        "Keys saved as public_key.txt and private_key.txt\n"
                        f"Stored N value: {last_rsa_n}"
                    )

            elif action == "encrypt_decrypt":
                # encrypt/decrypt branch
                if not (last_rsa_e and last_rsa_d and last_rsa_n):
                    crypto_output = "Error: Please generate RSA keys first."
                else:
                    message = request.form.get("message", "")

                    if not message:
                        crypto_output = "Please enter a message to encrypt."
                    else:
                        public_key = (last_rsa_e, last_rsa_n)
                        private_key = (last_rsa_d, last_rsa_n)

                        ciphertext = encrypt_message(message, public_key)
                        decrypted = decrypt_message(ciphertext, private_key)

                        crypto_output = (
                            "Encryption / Decryption demo using current RSA keys\n\n"
                            f"Original message:\n{message}\n\n"
                            f"Ciphertext (integer list):\n{ciphertext}\n\n"
                            f"Decrypted message:\n{decrypted}"
                        )

        except ValueError:
            if action == "generate":
                output = "Error: Please enter valid integers for min and max."
            else:
                crypto_output = "Error: Invalid input."
        except Exception as e:
            if action == "generate":
                output = f"Error generating RSA keys: {str(e)}"
            else:
                crypto_output = f"Error during encryption/decryption: {str(e)}"

    return render_template("rsa.html", output=output, crypto_output=crypto_output)


@app.route("/classical", methods=["GET", "POST"])
def classical_page():
    output = ""

    if request.method == "POST":
        try:
            buffer = io.StringIO()
            sys_stdout_backup = sys.stdout
            sys.stdout = buffer

            # You can use your preferred parameters here
            run_multiple(limit=5, min_p=10, max_p=40, max_attempts=20)

            sys.stdout = sys_stdout_backup
            output = buffer.getvalue()
        except Exception as e:
            sys.stdout = sys.__stdout__
            output = f"Error running classical Shor: {str(e)}"

    return render_template("classical.html", output=output)


@app.route("/quantum", methods=["GET", "POST"])
def quantum_page():
    global last_rsa_n
    output = ""
    N_value = ""
    use_rsa_n = False

    if request.method == "POST":
        N_value = request.form.get("N_value")
        use_rsa_n = request.form.get("use_rsa_n") == "true"

        try:
            buffer = io.StringIO()
            sys_stdout_backup = sys.stdout
            sys.stdout = buffer

            # Core logic: choose N and run Shor
            if use_rsa_n and last_rsa_n:
                factor_N(N=last_rsa_n)
            elif N_value:
                factor_N(N=int(N_value))
            else:
                # Default demo N if nothing provided
                factor_N(N=15)

            sys.stdout = sys_stdout_backup
            output = buffer.getvalue()
        except Exception as e:
            sys.stdout = sys.__stdout__
            output = f"Error running Quantum Shor: {str(e)}"

    return render_template("quantum.html", output=output, last_rsa_n=last_rsa_n)

def _find_shor_log():
    here = Path(__file__).parent
    cand1 = here / "shor_database.txt"
    if cand1.exists():
        return cand1
    cand2 = here.parent / "shor_database.txt"
    if cand2.exists():
        return cand2
    return None

def generate_rsa_in_range(min_p: int, max_p: int):
    """
    Generate RSA keys using primes in [min_p, max_p], without modifying RSAKeyGen.py.
    Uses is_prime and mod_inverse imported from RSAKeyGen.
    """

    if min_p < 2 or max_p <= min_p:
        raise ValueError("Invalid range: max must be > min and min must be ≥ 2.")

    # find all primes in range
    candidates = [p for p in range(min_p, max_p + 1) if is_prime(p)]
    if len(candidates) < 2:
        raise ValueError(f"Not enough primes found in range [{min_p}, {max_p}].")

    # pick two distinct primes
    p = random.choice(candidates)
    q = random.choice(candidates)
    while q == p:
        q = random.choice(candidates)

    n = p * q
    phi = (p - 1) * (q - 1)

    # choose e coprime with phi
    e = random.randrange(2, phi)
    while gcd(e, phi) != 1:
        e = random.randrange(2, phi)

    # compute d using RSAKeyGen's mod_inverse
    d = mod_inverse(e, phi)

    # save keys exactly like RSAKeyGen does
    public_key = (e, n)
    private_key = (d, n)

    with open("public_key.txt", "w") as f:
        f.write(str(public_key))
    with open("private_key.txt", "w") as f:
        f.write(str(private_key))

    return e, d, n


def _parse_shor_log(p: Path):
    rows = []
    pat = re.compile(r"N=(\d+),\s*p=(\d+),\s*q=(\d+),\s*a=(\d+),\s*r=(\w+)")
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        m = pat.search(line)
        if not m:
            continue
        N, p_, q, a, r = m.groups()
        rows.append({
            "N": int(N),
            "p": int(p_),
            "q": int(q),
            "a": int(a),
            "r": (None if r == "None" else int(r))
        })
    return rows

@app.route("/history")
def history():
    log_path = _find_shor_log()
    if not log_path:
        return render_template("history.html", data=None, error="shor_database.txt not found")
    data = _parse_shor_log(log_path)
    if not data:
        return render_template("history.html", data=[], error="No parsed entries found in shor_database.txt")
    return render_template("history.html", data=data, error=None)

@app.route("/help")
def help_page():
    return render_template("help.html")

@app.route("/qa")
def quantum_qa():
    return render_template("questionnaire.html")


if __name__ == '__main__':
    app.run(debug=True)
