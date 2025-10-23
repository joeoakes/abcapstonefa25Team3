# CHANGELOG — Team 3 Capstone Project (Fall 2025)

All notable progress for **Team 3: RSA & Shor’s Algorithm** is documented here, reconstructed from Scrum reports and the final GitHub activity timeline.

---

## [2025-08-30] — Week 1: Team Formation & Topic Selection
- Team members confirmed and assigned roles.
- Brainstormed multiple project ideas; decided on **RSA Encryption and Shor’s Algorithm** as the focus.
- Started reviewing background materials on cryptography and quantum computing.

---

## [2025-09-06] — Week 2: Research & Concept Planning
- Conducted detailed research into RSA key generation and modular arithmetic.
- Began studying **quantum factoring principles** and **Shor’s algorithm** structure.
- Drafted project proposal outline; discussed goals and expected deliverables.

---

## [2025-09-13] — Week 3: Proposal Writing & Presentation Prep
- Wrote the formal project proposal (problem, objectives, and methodology).
- Divided research responsibilities (RSA, Qiskit, quantum phase estimation, and QFT).
- No coding yet — focused on concept validation and presentation materials.

---

## [2025-09-20] — Week 4: Project Design & Tool Familiarization
- Explored **Qiskit** installation, Aer simulator setup, and simple circuit tests.
- Designed initial pseudocode for RSA and Classical Shor’s algorithms.
- Created diagrams showing how RSA and Shor’s interact conceptually.
- Still research-heavy; GitHub not active yet.

---

## [2025-09-27] — Week 5: Repository Setup & First Code Prototypes
- GitHub repository created and initialized.
- Implemented **RSAKeyGen.py** for basic key generation and encryption/decryption tests.
- Started **ClassicalShors.py** to simulate order-finding using brute-force search.
- Verified correctness of RSA output with small primes (p, q).
- Began early planning for quantum-side integration.

---

## [2025-10-04] — Week 6: Core Development Begins
- Created **QiskitShorsMain.py** — base quantum factoring script using QPE (Quantum Phase Estimation).
- Added continued-fraction decoding for stable phase → order extraction.
- Fixed bit ordering issues in measurement results.
- Successfully factored N=15 and N=21 on Aer simulator.
- Began documenting results and performance observations.

---

## [2025-10-11] — Week 7: System Integration & Visualization
- Improved circuit visualization using **Matplotlib + pylatexenc**.
- Standardized quantum logic across Classical and Quantum Shor scripts.
- Added data logging (`shor_database.txt`) to record factoring attempts and results.
- Cleaned and renamed files for consistency.
- Prepared screenshots and results for class demo.

---

## [2025-10-18] — Week 8: GUI & Performance Tools
- Built **GUI_Demo.py** to visualize RSA encryption and Shor factoring interactively.
- Developed **Performance Measurements.py** to benchmark quantum vs. classical timing.
- Added parameter options for N, n_count, and shot size for experimentation.
- Improved success validation (ensuring r is even and pow(a, r, N) == 1).
- Verified full end-to-end run through all modules.

---

## [2025-10-23] — Final Prep & Submission
- Finalized code cleanup and documentation (README and changelog).
- Completed presentation slides and demo notebook for professor review.
- Uploaded final repository with reproducible runs and detailed documentation.
- Confirmed all scripts execute successfully on Colab and local environments.

---

### Contributors
**Team 3:** Chris Joo, Sam Axler, Matthew Danese, John Teetz, William Lawther, Rahul Reji, Martin Shestani  
**Professor:** Joe Oakes  

---

> This changelog reflects realistic project pacing — research-heavy early weeks and concentrated development during the later stages of the semester.
