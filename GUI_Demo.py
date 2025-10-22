from flask import Flask, render_template_string, request
import subprocess

app = Flask(__name__)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Team 3 Capstone — RSA & Shor’s Algorithm GUI</title>
    <style>
        body { font-family: Arial, sans-serif; background: #eef2f7; text-align: center; margin: 0; padding: 0; }
        h1 { margin-top: 40px; color: #2c3e50; }
        h3 { color: #555; }
        .container {
            display: inline-block;
            background: #fff;
            padding: 30px 50px;
            border-radius: 15px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.1);
            margin-top: 40px;
            text-align: center;
        }
        button {
            padding: 12px 25px;
            margin: 10px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            transition: 0.2s ease-in-out;
        }
        button:hover { transform: scale(1.05); opacity: 0.9; }
        .rsa { background-color: #4CAF50; color: white; }
        .classical { background-color: #2196F3; color: white; }
        .quantum { background-color: #9C27B0; color: white; }
        .quantum_hamming { background-color: #FF5722; color: white; }
        pre {
            text-align: left;
            background: #1e1e1e;
            color: #00ff99;
            padding: 15px;
            border-radius: 10px;
            width: 80%;
            margin: 20px auto;
            overflow-x: auto;
            max-height: 400px;
        }
        footer {
            margin-top: 50px;
            color: #888;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <h1>Team 3 Capstone: RSA & Shor’s Algorithm</h1>
    <h3>Penn State AB — Fall 2025 | Professor Joe Oakes</h3>

    <div class="container">
        <form method="POST">
            <button name="action" value="generate_rsa" class="rsa"> Generate RSA Keys</button>
            <button name="action" value="run_classical" class="classical"> Run Classical Shor</button>
            <button name="action" value="run_quantum" class="quantum"> Run Quantum Shor</button>
            <button name="action" value="run_quantum_hamming" class="quantum_hamming"> Quantum + Hamming</button>
        </form>
    </div>

    {% if action %}
        <h2>Output: {{ action.replace('_', ' ').title() }}</h2>
        <pre>{{ output }}</pre>
    {% endif %}

    <footer>
        <p>Developed by Team Xtreme — Sam Axler, Matthew Danese, Chris Joo, John Teetz, William Lawther, Rahul Reji, Martin Shestani</p>
    </footer>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    output = ""
    action = None

    if request.method == 'POST':
        action = request.form.get('action')

        # Dummy test commands for demo purposes
        if action == 'generate_rsa':
            output = "RSA keys generated successfully! (p, q, n, e, d)"
        elif action == 'run_classical':
            output = "Running classical Shor's algorithm...\nFactored n = 15 → 3 × 5"
        elif action == 'run_quantum':
            output = "Simulating quantum Shor factoring using Qiskit Aer...\nResult: 21 → 3 × 7"
        elif action == 'run_quantum_hamming':
            output = "Quantum Shor with Hamming(7,4) error correction completed.\nFactored n = 33 → 3 × 11"
        else:
            output = "No action performed."

    return render_template_string(HTML_TEMPLATE, output=output, action=action)

if __name__ == '__main__':
    app.run(debug=True)
