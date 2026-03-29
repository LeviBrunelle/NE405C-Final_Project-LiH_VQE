from __future__ import annotations

from typing import Any

import numpy as np
from tqdm.auto import tqdm

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorEstimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA, SPSA
from qiskit_algorithms.utils import algorithm_globals

from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD

from .config import (
    COBYLA_MAXITER,
    DEFAULT_ANSATZ_MODE,
    DEFAULT_REPS,
    ENTANGLEMENT_PATTERN,
    OPTIMIZER_NAME,
    RANDOM_SEED,
    SPSA_MAXITER,
)


def build_hardware_efficient_ansatz(num_qubits: int, reps: int = DEFAULT_REPS) -> QuantumCircuit:
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


def build_uccsd_hf_ansatz(problem: Any, mapper: Any):
    hf_state = HartreeFock(
        num_spatial_orbitals=problem.num_spatial_orbitals,
        num_particles=problem.num_particles,
        qubit_mapper=mapper,
    )

    ansatz = UCCSD(
        num_spatial_orbitals=problem.num_spatial_orbitals,
        num_particles=problem.num_particles,
        qubit_mapper=mapper,
        initial_state=hf_state,
    )

    # Pragmatic replacement for HFInitialPoint helper import:
    # zero excitation amplitudes + HartreeFock initial state
    initial_point = np.zeros(ansatz.num_parameters, dtype=float)

    return ansatz, initial_point


def build_ansatz(
    num_qubits: int,
    reps: int = DEFAULT_REPS,
    ansatz_mode: str = DEFAULT_ANSATZ_MODE,
    problem: Any | None = None,
    mapper: Any | None = None,
):
    if ansatz_mode == "hardware_efficient":
        ansatz = build_hardware_efficient_ansatz(num_qubits=num_qubits, reps=reps)
        initial_point = np.zeros(ansatz.num_parameters, dtype=float)
        return ansatz, initial_point

    if ansatz_mode == "uccsd_hf":
        if problem is None or mapper is None:
            raise ValueError("UCCSD + HartreeFock ansatz requires both problem and mapper.")
        return build_uccsd_hf_ansatz(problem=problem, mapper=mapper)

    raise ValueError(f"Unsupported ansatz mode: {ansatz_mode}")


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

def run_repeated_trials(
    qubit_hamiltonian: Any,
    num_trials: int,
    reps: int = DEFAULT_REPS,
    optimizer_name: str | None = None,
    ansatz_mode: str = DEFAULT_ANSATZ_MODE,
    problem: Any | None = None,
    mapper: Any | None = None,
    show_progress: bool = False,
    base_seed: int = RANDOM_SEED,
) -> list[dict[str, Any]]:
    results = []

    for trial_idx in range(num_trials):
        trial_seed = base_seed + trial_idx
        result = run_vqe(
            qubit_hamiltonian=qubit_hamiltonian,
            reps=reps,
            optimizer_name=optimizer_name,
            ansatz_mode=ansatz_mode,
            problem=problem,
            mapper=mapper,
            show_progress=show_progress,
            progress_label=f"{optimizer_name} trial={trial_idx+1}",
            random_seed=trial_seed,
        )
        result["trial_index"] = trial_idx
        result["random_seed"] = trial_seed
        results.append(result)

    return results

def run_vqe(
    qubit_hamiltonian: Any,
    reps: int = DEFAULT_REPS,
    optimizer_name: str | None = None,
    ansatz_mode: str = DEFAULT_ANSATZ_MODE,
    problem: Any | None = None,
    mapper: Any | None = None,
    show_progress: bool = False,
    progress_label: str | None = None,
    random_seed: int | None = None,
) -> dict[str, Any]:
    chosen_optimizer = optimizer_name or OPTIMIZER_NAME

    seed = RANDOM_SEED if random_seed is None else random_seed
    algorithm_globals.random_seed = seed
    np.random.seed(seed)

    ansatz, initial_point = build_ansatz(
        num_qubits=qubit_hamiltonian.num_qubits,
        reps=reps,
        ansatz_mode=ansatz_mode,
        problem=problem,
        mapper=mapper,
    )

    estimator = StatevectorEstimator()
    optimizer = build_optimizer(chosen_optimizer)

    counts: list[int] = []
    energies: list[float] = []

    pbar = None
    last_eval_count = 0

    if show_progress:
        desc = progress_label or f"VQE {ansatz_mode} {chosen_optimizer}"
        pbar = tqdm(
            total=get_optimizer_maxiter(chosen_optimizer),
            desc=desc,
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
        initial_point=initial_point,
        callback=callback,
    )

    result = solver.compute_minimum_eigenvalue(qubit_hamiltonian)

    if pbar is not None:
        pbar.close()

    return {
        "optimizer_name": chosen_optimizer,
        "ansatz_mode": ansatz_mode,
        "energy": float(result.eigenvalue.real),
        "optimal_point": [float(x) for x in result.optimal_point],
        "counts": counts,
        "energies": energies,
        "ansatz_num_qubits": int(ansatz.num_qubits),
        "ansatz_num_parameters": int(ansatz.num_parameters),
        "ansatz_depth": int(ansatz.decompose().depth() if hasattr(ansatz, "decompose") else ansatz.depth()),
        "ansatz": ansatz,
    }


def run_reps_experiment(
    qubit_hamiltonian: Any,
    reps_values: list[int],
    optimizer_name: str | None = None,
    ansatz_mode: str = DEFAULT_ANSATZ_MODE,
    problem: Any | None = None,
    mapper: Any | None = None,
    show_progress: bool = False,
) -> dict[int, dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    for reps in reps_values:
        results[reps] = run_vqe(
            qubit_hamiltonian=qubit_hamiltonian,
            reps=reps,
            optimizer_name=optimizer_name,
            ansatz_mode=ansatz_mode,
            problem=problem,
            mapper=mapper,
            show_progress=show_progress,
        )
    return results


def run_optimizer_experiment(
    qubit_hamiltonian: Any,
    optimizer_names: list[str],
    reps: int,
    ansatz_mode: str = DEFAULT_ANSATZ_MODE,
    problem: Any | None = None,
    mapper: Any | None = None,
    show_progress: bool = False,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for optimizer_name in optimizer_names:
        results[optimizer_name] = run_vqe(
            qubit_hamiltonian=qubit_hamiltonian,
            reps=reps,
            optimizer_name=optimizer_name,
            ansatz_mode=ansatz_mode,
            problem=problem,
            mapper=mapper,
            show_progress=show_progress,
        )
    return results


def run_ansatz_mode_experiment(
    qubit_hamiltonian: Any,
    ansatz_modes: list[str],
    reps: int,
    optimizer_name: str,
    problem: Any,
    mapper: Any,
    show_progress: bool = False,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for ansatz_mode in ansatz_modes:
        results[ansatz_mode] = run_vqe(
            qubit_hamiltonian=qubit_hamiltonian,
            reps=reps,
            optimizer_name=optimizer_name,
            ansatz_mode=ansatz_mode,
            problem=problem,
            mapper=mapper,
            show_progress=show_progress,
        )
    return results