from qiskit_algorithms import NumPyMinimumEigensolver


def run_exact(qubit_hamiltonian) -> dict[str, float]:
    result = NumPyMinimumEigensolver().compute_minimum_eigenvalue(qubit_hamiltonian)
    return {"energy": float(result.eigenvalue.real)}