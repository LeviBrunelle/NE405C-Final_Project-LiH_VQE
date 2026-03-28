"""
run_vqe.py
----------
Run the Variational Quantum Eigensolver (VQE) on a qubit Hamiltonian and return
the estimated ground-state energy (including nuclear repulsion).

The hardware-efficient EfficientSU2 ansatz is used together with the SLSQP
classical optimizer.
"""

import numpy as np

from qiskit.circuit.library import EfficientSU2
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP


def run_vqe(qubit_op, problem, mapper, reps: int = 2, max_iter: int = 500):
    """Estimate the LiH ground-state energy using VQE.

    Parameters
    ----------
    qubit_op : SparsePauliOp
        The qubit Hamiltonian to minimize.
    problem : ElectronicStructureProblem
        The electronic-structure problem (used to retrieve the nuclear
        repulsion energy).
    mapper : ParityMapper
        The mapper used when building *qubit_op* (kept for API consistency).
    reps : int, optional
        Number of repetition layers in the EfficientSU2 ansatz (default 2).
    max_iter : int, optional
        Maximum number of optimizer iterations (default 500).

    Returns
    -------
    total_energy : float
        Electronic + nuclear-repulsion ground-state energy in Hartree.
    result : MinimumEigensolverResult
        Raw VQE result object for inspection.
    """
    ansatz = EfficientSU2(qubit_op.num_qubits, reps=reps, entanglement="linear")

    # Initialise parameters near zero to help the optimizer
    rng = np.random.default_rng(seed=42)
    initial_point = rng.uniform(-np.pi / 4, np.pi / 4, ansatz.num_parameters)

    optimizer = SLSQP(maxiter=max_iter)
    estimator = StatevectorEstimator()

    vqe = VQE(estimator, ansatz, optimizer, initial_point=initial_point)
    result = vqe.compute_minimum_eigenvalue(qubit_op)

    nuclear_repulsion = problem.nuclear_repulsion_energy
    total_energy = float(result.eigenvalue.real) + nuclear_repulsion

    return total_energy, result
