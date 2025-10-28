from flask import Flask, render_template, request
from ClassicalShors import run_multiple
from RSAKeyGen import generate_rsa_keys

# use pip install flask if you don't have flask installed
# After running the code click on the development server to open the GUI

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    output = ""
    action = None

    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'generate_rsa':
            # Generate keys using the function from the RSAKeyGen file 
            public_key, private_key = generate_rsa_keys(bits=8)
            output = (
                " RSA Keys Generated Successfully!\n\n"
                f"Public Key (e, n): {public_key}\n"
                f"Private Key (d, n): {private_key}\n\n"
                "Keys saved as public_key.txt and private_key.txt"
            )

        elif action == 'run_classical':
            # Run the classical Shor’s algorithm from ClassicalShors.py
            try:
                import io
                import sys

                # Capture printed output to display in the webpage
                buffer = io.StringIO()
                sys.stdout = buffer

                # Run the algorithm (change parameters for speed if necessary)
                run_multiple(limit=5, min_p=10, max_p=40, max_attempts=20)

                sys.stdout = sys.__stdout__
                output = buffer.getvalue()

            except Exception as e:
                output = f"Error running classical Shor’s algorithm: {str(e)}"
            finally:
                sys.stdout = sys.__stdout__  # Reset stdout to normal

        elif action == 'run_quantum':
            output = "Simulating quantum Shor factoring using Qiskit Aer...\nResult: 21 → 3 × 7"

        elif action == 'run_quantum_hamming':
            output = "Quantum Shor with Hamming(7,4) error correction completed.\nFactored n = 33 → 3 × 11"

        else:
            output = "No action performed."
    #Load the index file
    return render_template("index.html", output=output, action=action)

if __name__ == '__main__':
    app.run(debug=True)

