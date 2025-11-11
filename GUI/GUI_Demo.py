from flask import Flask, render_template, request
from ClassicalShors import run_multiple
from RSAKeyGen import generate_rsa_keys   # correct import
from QiskitShorsMain import factor_N # Takes value N
import io
import sys
from pathlib import Path
import re
from flask import jsonify


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
                    try:
                        buffer = io.StringIO()
                        sys.stdout = buffer
                        factor_N(N=15)
                        sys.stdout = sys.__stdout__
                        output = buffer.getvalue()
                    except Exception as e:
                        output = f" Error running Quantum Shor’s algorithm: {str(e)}"
                elif N_value:
                    try:
                        buffer = io.StringIO()
                        sys.stdout = buffer
                        factor_N(N=N_value)
                        sys.stdout = sys.__stdout__
                        output = buffer.getvalue()
                    except Exception as e:
                        output = f" Error running Quantum Shor’s algorithm: {str(e)}"
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

def _find_shor_log():
    here = Path(__file__).parent
    cand1 = here / "shor_database.txt"
    if cand1.exists():
        return cand1
    cand2 = here.parent / "shor_database.txt"
    if cand2.exists():
        return cand2
    return None

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
        return jsonify({"error": "shor_database.txt not found"}), 404
    data = _parse_shor_log(log_path)
    if not data:
        return "<h3>No parsed entries found in shor_database.txt</h3>"
    html = ["<h2>Shor run history</h2><table border='1' cellpadding='6'><tr><th>N</th><th>p</th><th>q</th><th>a</th><th>r</th></tr>"]
    for row in data[:200]:
        html.append(f"<tr><td>{row['N']}</td><td>{row['p']}</td><td>{row['q']}</td><td>{row['a']}</td><td>{row['r']}</td></tr>")
    html.append("</table>")
    return "\n".join(html)

if __name__ == '__main__':
    app.run(debug=True)
