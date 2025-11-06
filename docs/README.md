# Team 3 — Capstone Project
# CMPSC 488 - Pennsylvania State University Abington - Fall 2025

This repository was developed by **Team 3** as part of the Penn State AB Capstone Project.

A compact repository demonstrating the connection between classical RSA encryption and Shor’s quantum factoring algorithm using Qiskit. The project contains classical baselines, quantum Shor implementations, utilities for RSA key generation, a demo GUI, and measurement scripts.

## Technologies used

- Python 3.9+ (recommended)
- Qiskit (Terra & Aer — use a version compatible with Python 3.9+)
- Jupyter / JupyterLab (for `EncryptDecrypt.ipynb`)
- Standard Python tooling: pip, venv

## Overview

This repo explores how RSA’s security (modulus n = p × q) is affected by quantum factoring techniques, specifically Shor’s algorithm. It includes both classical order-finding baselines and Qiskit-based Shor implementations that run on local simulators (Aer) for educational and experimental purposes.

## Team

- Chris Joo
- Sam Axler
- Matthew Danese
- John Teetz
- William Lawther
- Rahul Reji
- Martin Shestani

Professor: Joe Oakes

## Repository structure

Files in the project root (brief purpose):

```
├─ ClassicalShors.py            # Classical factoring baseline (brute-force order finding)
├─ QiskitShorsMain.py           # Quantum Shor algorithm (QPE + continued fraction decoding)
├─ QiskitShorsOld.py            # Early/experimental quantum factoring versions
├─ GUI_Demo.py                  # Interactive front-end for RSA & Shor demonstrations
├─ "Performance Measurements.py"  # Performance logging and benchmark script (note space in filename)
├─ RSAKeyGen.py                 # RSA key generation (p, q, n, φ, e, d)
├─ EncryptDecrypt.ipynb         # RSA encryption/decryption walkthrough notebook
├─ shor_database.txt            # Log of previous factoring runs (N, a, r, success/fail)
└─ README.txt                   # Original plaintext README (kept for reference)
```

## Requirements

- Python 3.9+ recommended
- Qiskit (test with Qiskit Terra/Aer compatible with your Python version)
- pip and a virtual environment tool

If you plan to run the quantum circuits on local simulators, install Qiskit Aer. For larger experiments, consider appropriate simulators/backends and increased memory/time.

## Setup

1. Create and activate a virtual environment:

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

2. Install dependencies (if `requirements.txt` exists):

```powershell
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, install Qiskit manually:

```powershell
pip install qiskit
```

## Quick start

Run the classical baseline for a small example:

```powershell
python ClassicalShors.py --N 15
```

Run the main quantum Shor script (start with small N like 15 or 21):

```powershell
python QiskitShorsMain.py --N 15 --n-count 10 --shots 4096
```

Run the GUI demo:

```powershell
python GUI_Demo.py
```

Run performance measurements (example):

```powershell
python "Performance Measurements.py" --Ns 15 21 33 --shots 4096 --repeat 5
```

---

##  Educational Objective
This project demonstrates how **RSA’s modulus `n = p × q`**—the cornerstone of classical cryptography—becomes vulnerable to **quantum factoring** via Shor’s algorithm, highlighting the contrast between quantum and classical computational power.

---