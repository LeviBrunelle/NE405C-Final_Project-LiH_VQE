# LiH VQE Project

This repo implements a Qiskit-based variational quantum eigensolver workflow for LiH. Created for a final project in ECE405C - Intro Quantum Algorithms.

## Structure

- `src/config.py` contains all tunable project parameters.
- `src/build_hamiltonian.py` builds the LiH electronic-structure problem and maps it to qubits.
- `src/run_exact.py` computes the exact minimum eigenvalue of the qubit Hamiltonian.
- `src/run_vqe.py` builds the ansatz and runs VQE.
- `src/scan_bond_lengths.py` evaluates exact and VQE energies across multiple bond lengths.
- `src/plot_results.py` generates convergence and energy plots.
- `main.py` runs the full pipeline and writes outputs to `results/`.

## Setup

```bash
pip install -r requirements.txt