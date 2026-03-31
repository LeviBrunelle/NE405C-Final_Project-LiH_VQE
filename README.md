# LiH VQE Project

A Qiskit-based variational quantum eigensolver (VQE) workflow for small molecular systems, built for the ECE405C final project.

This project studies how different design choices affect VQE performance, including:

- ansatz type
- circuit depth
- optimizer choice
- bond length
- basis set
- noisy vs noiseless simulation settings

The main reference problem is LiH, with optional extensions to other small molecules such as BeH2.

---

## Project Goal

The goal of this repo is to compare exact diagonalization against VQE estimates for molecular Hamiltonians mapped to qubits.

The project focuses on a few core questions:

- How well does a hardware-efficient ansatz approximate the ground-state energy?
- How does it compare to a chemistry-motivated ansatz such as UCCSD with a Hartree–Fock reference?
- How do optimizer choice and circuit depth affect convergence?
- How does performance change across bond lengths?
- What happens when shot noise and simple hardware-inspired noise are introduced?

---

## Method Overview

The workflow is:

1. Build an electronic structure problem with `PySCFDriver`
2. Convert the fermionic Hamiltonian to a qubit Hamiltonian with the Jordan–Wigner mapping
3. Compute the exact minimum eigenvalue as a classical baseline
4. Build a parameterized ansatz circuit
5. Run VQE with a selected optimizer and backend
6. Save numerical results and plots to `results/`

