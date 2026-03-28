from __future__ import annotations

from typing import Any

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA

from .config import (
    COBYLA_MAXITER,
    DEFAULT_REPS,
    ENTANGLEMENT_PATTERN,
    OPTIMIZER_NAME,
)


def build_ansatz(num_qubits: int, reps: int = DEFAULT_REPS) -> QuantumCircuit:
    """
    Build a simple hardware-efficient ansatz manually.

    Structure per repetition:
    - Ry layer on all qubits
    - Rz layer on all qubits
    - entangling CX layer

    This avoids Qiskit circuit-library API/version issues.
    """
    if ENTANGLEMENT_PATTERN not in {"linear", "circular"}:
        raise ValueError(
            f"Unsupported entanglement pattern: {ENTANGLEMENT_PATTERN}. "
            "Use 'linear' or 'circular'."
        )

    qc = QuantumCircuit(num_qubits)

    num_rotation_params = reps * num_qubits * 2
    theta = ParameterVector("theta", num_rotation_params)

    param_index = 0

    for _ in range(reps):
        for q in range(num_qubits):
            qc.ry(theta[param_index], q)
            param_index += 1

        for q in range(num_qubits):
            qc.rz(theta[param_index], q)
            param_index += 1

        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)

        if ENTANGLEMENT_PATTERN == "circular" and num_qubits > 2:
            qc.cx(num_qubits - 1, 0)

    return qc


def build_optimizer():
    if OPTIMIZER_NAME != "COBYLA":
        raise ValueError(f"Unsupported optimizer: {OPTIMIZER_NAME}")
    return COBYLA(maxiter=COBYLA_MAXITER)


def run_vqe(qubit_hamiltonian: Any, reps: int = DEFAULT_REPS) -> dict[str, Any]:
    ansatz = build_ansatz(num_qubits=qubit_hamiltonian.num_qubits, reps=reps)
    estimator = StatevectorEstimator()
    optimizer = build_optimizer()

    counts: list[int] = []
    energies: list[float] = []

    def callback(eval_count, parameters, mean, metadata):
        counts.append(int(eval_count))
        energies.append(float(mean))

    solver = VQE(
        estimator=estimator,
        ansatz=ansatz,
        optimizer=optimizer,
        callback=callback,
    )

    result = solver.compute_minimum_eigenvalue(qubit_hamiltonian)

    return {
        "energy": float(result.eigenvalue.real),
        "optimal_point": [float(x) for x in result.optimal_point],
        "counts": counts,
        "energies": energies,
        "ansatz_num_qubits": int(ansatz.num_qubits),
        "ansatz_num_parameters": int(ansatz.num_parameters),
        "ansatz_depth": int(ansatz.depth()),
        "ansatz": ansatz,
    }


def run_reps_experiment(qubit_hamiltonian: Any, reps_values: list[int]) -> dict[int, dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    for reps in reps_values:
        results[reps] = run_vqe(qubit_hamiltonian=qubit_hamiltonian, reps=reps)
    return results