"""
run_exact.py
------------
Compute the exact ground-state energy of a qubit Hamiltonian by full
diagonalisation using the NumPy minimum eigensolver (classical reference).
"""

from qiskit_algorithms import NumPyMinimumEigensolver


def run_exact(qubit_op, problem):
    """Compute the exact ground-state energy via full diagonalisation.

    Parameters
    ----------
    qubit_op : SparsePauliOp
        The qubit Hamiltonian.
    problem : ElectronicStructureProblem
        The electronic-structure problem (used to retrieve the nuclear
        repulsion energy).

    Returns
    -------
    total_energy : float
        Electronic + nuclear-repulsion ground-state energy in Hartree.
    """
    solver = NumPyMinimumEigensolver()
    result = solver.compute_minimum_eigenvalue(qubit_op)

    nuclear_repulsion = problem.nuclear_repulsion_energy
    total_energy = float(result.eigenvalue.real) + nuclear_repulsion

    return total_energy
