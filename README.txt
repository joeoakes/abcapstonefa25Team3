##abcapstonefa25Team3/
├─ .vscode/
│  ├─ launch.json
│  └─ settings.json
├─ .gitignore
├─ requirements.txt
├─ README.md
├─ rsa_keygen.py
├─ shor_factor_anyN.py
├─ shor_from_scratch_qiskit21_hamming.py
└─ classical_shor.py


# Team 3 — RSA & Shor’s Algorithm (Qiskit + Classical)

This repo contains four scripts that tie RSA key generation to quantum factoring demonstrations using Qiskit, plus a classical baseline:

- **`rsa_keygen.py`** — Generates RSA keys (p, q, n, φ, e, d) and saves `public_key.txt` and `private_key.txt`.
- **`shor_factor_anyN.py`** — Shor-style factoring via **Quantum Phase Estimation (QPE)** and inverse QFT on Aer simulator.
- **`shor_from_scratch_qiskit21_hamming.py`** — Qiskit ≥ 2.1 compatible Shor implementation with **Hamming(7,4)** single-bit correction on measured phase bits.
- **`classical_shor.py`** — Purely classical order-finding (brute-force) to mirror Shor’s logic on small N.

> Educational purpose: show how RSA’s modulus `n = p*q` becomes vulnerable if factoring is efficient (quantum), contrasted with classical difficulty.

---

## Setup

```bash
# 1) Create & activate a venv (recommended)
python -m venv venv
# Windows (PowerShell):
venv\Scripts\Activate.ps1
# macOS/Linux:
source venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt
