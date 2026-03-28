from __future__ import annotations

from typing import Any

from tqdm.auto import tqdm

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA, SPSA

from .config import (
    COBYLA_MAXITER,
    DEFAULT_REPS,
    ENTANGLEMENT_PATTERN,
    OPTIMIZER_NAME,
    SPSA_MAXITER,
)


def build_ansatz(num_qubits: int, reps: int = DEFAULT_REPS) -> QuantumCircuit:
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


def build_optimizer(optimizer_name: str | None = None):
    name = optimizer_name or OPTIMIZER_NAME

    if name == "COBYLA":
        return COBYLA(maxiter=COBYLA_MAXITER)
    if name == "SPSA":
        return SPSA(maxiter=SPSA_MAXITER)

    raise ValueError(f"Unsupported optimizer: {name}")


def get_optimizer_maxiter(optimizer_name: str | None = None) -> int:
    name = optimizer_name or OPTIMIZER_NAME
    if name == "COBYLA":
        return COBYLA_MAXITER
    if name == "SPSA":
        return SPSA_MAXITER
    raise ValueError(f"Unsupported optimizer: {name}")


def run_vqe(
    qubit_hamiltonian: Any,
    reps: int = DEFAULT_REPS,
    optimizer_name: str | None = None,
    show_progress: bool = False,
) -> dict[str, Any]:
    chosen_optimizer = optimizer_name or OPTIMIZER_NAME

    ansatz = build_ansatz(num_qubits=qubit_hamiltonian.num_qubits, reps=reps)
    estimator = StatevectorEstimator()
    optimizer = build_optimizer(chosen_optimizer)

    counts: list[int] = []
    energies: list[float] = []

    pbar = None
    last_eval_count = 0

    if show_progress:
        pbar = tqdm(
            total=get_optimizer_maxiter(chosen_optimizer),
            desc=f"VQE {chosen_optimizer} reps={reps}",
            leave=False,
        )

    def callback(eval_count, parameters, mean, metadata):
        nonlocal last_eval_count

        counts.append(int(eval_count))
        energies.append(float(mean))

        if pbar is not None:
            increment = max(0, int(eval_count) - last_eval_count)
            if increment > 0:
                pbar.update(increment)
            pbar.set_postfix({"energy": f"{float(mean):.6f}"})
            last_eval_count = int(eval_count)

    solver = VQE(
        estimator=estimator,
        ansatz=ansatz,
        optimizer=optimizer,
        callback=callback,
    )

    result = solver.compute_minimum_eigenvalue(qubit_hamiltonian)

    if pbar is not None:
        pbar.close()

    return {
        "optimizer_name": chosen_optimizer,
        "energy": float(result.eigenvalue.real),
        "optimal_point": [float(x) for x in result.optimal_point],
        "counts": counts,
        "energies": energies,
        "ansatz_num_qubits": int(ansatz.num_qubits),
        "ansatz_num_parameters": int(ansatz.num_parameters),
        "ansatz_depth": int(ansatz.depth()),
        "ansatz": ansatz,
    }


def run_reps_experiment(
    qubit_hamiltonian: Any,
    reps_values: list[int],
    optimizer_name: str | None = None,
    show_progress: bool = False,
) -> dict[int, dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    for reps in reps_values:
        results[reps] = run_vqe(
            qubit_hamiltonian=qubit_hamiltonian,
            reps=reps,
            optimizer_name=optimizer_name,
            show_progress=show_progress,
        )
    return results


def run_optimizer_experiment(
    qubit_hamiltonian: Any,
    optimizer_names: list[str],
    reps: int,
    show_progress: bool = False,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for optimizer_name in optimizer_names:
        results[optimizer_name] = run_vqe(
            qubit_hamiltonian=qubit_hamiltonian,
            reps=reps,
            optimizer_name=optimizer_name,
            show_progress=show_progress,
        )
    return results