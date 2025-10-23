# Team 3 Capstone Project (Fall 2025)

This repository was developed by **Team 3** as part of the Penn State AB Capstone Project.  
It explores how **RSA encryption** and **Shor’s quantum factoring algorithm** connect — demonstrating classical and quantum approaches to integer factorization using Qiskit.

---

##  Team Members
- Chris Joo  
- Sam Axler  
- Matthew Danese  
- John Teetz  
- William Lawther  
- Rahul Reji  
- Martin Shestani  

**Professor:** Joe Oakes  

---

##  Repository Structure
```
├─ ClassicalShors.py            # Classical factoring baseline (brute-force order finding)
├─ QuiskitShorsMain.py          # Final Quantum Shor algorithm (QPE + continued fraction decoding)
├─ QiskitShorsOld.py            # Early experimental version of quantum factoring
├─ GUI_Demo.py                  # Interactive front-end for RSA & Shor demonstrations
├─ Performance Measurements.py  # Performance logging and benchmark script
├─ RSAKeyGen.py                 # RSA key generation (p, q, n, φ, e, d)
├─ EncryptDecrypt.ipynb         # RSA encryption/decryption walkthrough notebook
└─ shor_database.txt            # Log of previous factoring runs (N, a, r, success/fail)
```

---

##  Setup Instructions
```bash
# 1) Create & activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\Activate.ps1

# macOS/Linux
source venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt
```

---

##  Quick Start

### Classical Baseline
```bash
python ClassicalShors.py --N 15
```

### Quantum Shor (Main)
```bash
python QuiskitShorsMain.py --N 15 --n-count 10 --shots 4096
```
> Start with N = 15 or 21. Increase `--n-count` or `--shots` for higher accuracy.

### GUI Demo
```bash
python GUI_Demo.py
```

### Performance Testing
```bash
python "Performance Measurements.py" --Ns 15 21 33 --shots 4096 --repeat 5
```
---

# Team 3 — RSA & Shor’s Algorithm (Qiskit + Classical)

This repo contains four scripts that tie RSA key generation to quantum factoring demonstrations using Qiskit, plus a classical baseline:

- **`rsa_keygen.py`** — Generates RSA keys (p, q, n, φ, e, d) and saves `public_key.txt` and `private_key.txt`.
- **`shor_factor_anyN.py`** — Shor-style factoring via **Quantum Phase Estimation (QPE)** and inverse QFT on Aer simulator.
- **`shor_from_scratch_qiskit21_hamming.py`** — Qiskit ≥ 2.1 compatible Shor implementation with **Hamming(7,4)** single-bit correction on measured phase bits.
- **`classical_shor.py`** — Purely classical order-finding (brute-force) to mirror Shor’s logic on small N.

---

##  Educational Objective
This project demonstrates how **RSA’s modulus `n = p × q`**—the cornerstone of classical cryptography—becomes vulnerable to **quantum factoring** via Shor’s algorithm, highlighting the contrast between quantum and classical computational power.

---
