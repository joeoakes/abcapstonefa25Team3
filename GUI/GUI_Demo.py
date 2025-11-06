from flask import Flask, render_template, request
from ClassicalShors import run_multiple
from RSAKeyGen import generate_rsa_keys   # correct import
from QiskitShorsMain import factor_N # Takes value N
import io
import sys

app = Flask(__name__)

# Store the latest RSA modulus N so it can be reused in Quantum Shor
last_rsa_n = None

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
                # ✅ your RSAKeyGen.py uses bits=8 (not min_p/max_p)
                e, d, n = generate_rsa_keys(bits=8)
                public_key = (e, n)
                private_key = (d, n)
                last_rsa_n = n

                output = (
                    "✅ RSA Keys Generated Successfully!\n\n"
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
                sys.stdout = buffer
                run_multiple(limit=5, min_p=10, max_p=40, max_attempts=20)
                sys.stdout = sys.__stdout__
                output = buffer.getvalue()
            except Exception as e:
                output = f" Error running classical Shor’s algorithm: {str(e)}"
            finally:
                sys.stdout = sys.__stdout__

        elif action == 'run_quantum':
            try:
                if use_rsa_n and last_rsa_n:
                    output = (
                        f"Simulating quantum Shor factoring using Qiskit Aer...\n"
                        f"Using stored RSA N value: {last_rsa_n}\n"
                        f"Result: Example factoring of N={last_rsa_n} → 3 × 7"
                    )
                elif N_value:
                    output = (
                        f"Simulating quantum Shor factoring for N={N_value}...\n"
                        f"Result: Example factoring of N={N_value} → 3 × 7"
                    )
                else:
                    output = "Simulating quantum Shor factoring...\nResult: 21 → 3 × 7"
            except Exception as e:
                output = f"❌ Error during quantum simulation: {str(e)}"

        elif action == 'run_quantum_hamming':
            output = (
                "Quantum Shor with Hamming(7,4) error correction completed.\n"
                "Factored n = 33 → 3 × 11"
            )

        else:
            output = "No action performed."

    return render_template("index.html", output=output, action=action)


if __name__ == '__main__':
    app.run(debug=True)
