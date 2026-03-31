from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import StatevectorEstimator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
from qiskit_aer.primitives import EstimatorV2 as AerEstimator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA, GradientDescent, SPSA
from qiskit_algorithms.utils import algorithm_globals
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD

from .config import (
    AER_DEFAULT_PRECISION,
    ANSATZ_KIND,
    BACKEND_MODE,
    COBYLA_MAXITER,
    ENTANGLEMENT_PATTERN,
    GD_LEARNING_RATE,
    GD_MAXITER,
    NOISE_1Q,
    NOISE_2Q,
    OPTIMIZER,
    RANDOM_SEED,
    READOUT_ERROR,
    REPS,
    SPSA_MAXITER,
    USE_TQDM,
)


def build_hardware_efficient_ansatz(num_qubits: int, reps: int) -> QuantumCircuit:
    qc = QuantumCircuit(num_qubits)
    theta = ParameterVector("theta", reps * num_qubits * 2)
    k = 0
    for _ in range(reps):
        for q in range(num_qubits):
            qc.ry(theta[k], q)
            k += 1
        for q in range(num_qubits):
            qc.rz(theta[k], q)
            k += 1
        for q in range(num_qubits - 1):
            qc.cx(q, q + 1)
        if ENTANGLEMENT_PATTERN == "circular" and num_qubits > 2:
            qc.cx(num_qubits - 1, 0)
    return qc


def build_ansatz(
    num_qubits: int,
    ansatz_kind: str = ANSATZ_KIND,
    reps: int = REPS,
    problem: Any | None = None,
    mapper: Any | None = None,
):
    if ansatz_kind == "hardware_efficient":
        ansatz = build_hardware_efficient_ansatz(num_qubits, reps)
        return ansatz, np.zeros(ansatz.num_parameters, dtype=float)
    if ansatz_kind == "uccsd_hf":
        if problem is None or mapper is None:
            raise ValueError("UCCSD+HF requires problem and mapper")
        hf = HartreeFock(
            num_spatial_orbitals=problem.num_spatial_orbitals,
            num_particles=problem.num_particles,
            qubit_mapper=mapper,
        )
        ansatz = UCCSD(
            num_spatial_orbitals=problem.num_spatial_orbitals,
            num_particles=problem.num_particles,
            qubit_mapper=mapper,
            initial_state=hf,
        )
        return ansatz, np.zeros(ansatz.num_parameters, dtype=float)
    raise ValueError(f"Unsupported ansatz_kind: {ansatz_kind}")


def build_optimizer(name: str = OPTIMIZER):
    if name == "COBYLA":
        return COBYLA(maxiter=COBYLA_MAXITER)
    if name == "SPSA":
        return SPSA(maxiter=SPSA_MAXITER)
    if name == "GRADIENT_DESCENT":
        return GradientDescent(maxiter=GD_MAXITER, learning_rate=GD_LEARNING_RATE)
    raise ValueError(f"Unsupported optimizer: {name}")


def optimizer_eval_budget(name: str = OPTIMIZER) -> int:
    if name == "COBYLA":
        return COBYLA_MAXITER
    if name == "SPSA":
        return SPSA_MAXITER * 2
    if name == "GRADIENT_DESCENT":
        return GD_MAXITER * 50
    raise ValueError(f"Unsupported optimizer: {name}")


def build_noise_model() -> NoiseModel:
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(NOISE_1Q, 1), ["ry", "rz", "x"])
    noise_model.add_all_qubit_quantum_error(depolarizing_error(NOISE_2Q, 2), ["cx"])
    readout = ReadoutError([[1 - READOUT_ERROR, READOUT_ERROR], [READOUT_ERROR, 1 - READOUT_ERROR]])
    noise_model.add_all_qubit_readout_error(readout)
    return noise_model


def build_estimator(mode: str = BACKEND_MODE, seed: int | None = RANDOM_SEED):
    if mode == "statevector":
        return StatevectorEstimator()
    options = {
        "default_precision": AER_DEFAULT_PRECISION,
        "backend_options": {"method": "automatic", "seed_simulator": seed},
    }
    if mode == "aer_noise":
        options["backend_options"]["noise_model"] = build_noise_model()
    if mode not in {"aer_shots", "aer_noise"}:
        raise ValueError(f"Unsupported backend mode: {mode}")
    return AerEstimator(options=options)


def run_vqe(
    qubit_hamiltonian: Any,
    *,
    ansatz_kind: str = ANSATZ_KIND,
    optimizer_name: str = OPTIMIZER,
    reps: int = REPS,
    backend_mode: str = BACKEND_MODE,
    problem: Any | None = None,
    mapper: Any | None = None,
    seed: int | None = RANDOM_SEED,
    show_progress: bool = USE_TQDM,
    label: str | None = None,
) -> dict[str, Any]:
    if seed is not None:
        algorithm_globals.random_seed = seed
        np.random.seed(seed)

    ansatz, initial_point = build_ansatz(
        qubit_hamiltonian.num_qubits,
        ansatz_kind=ansatz_kind,
        reps=reps,
        problem=problem,
        mapper=mapper,
    )
    estimator = build_estimator(backend_mode, seed)
    optimizer = build_optimizer(optimizer_name)

    counts, energies = [], []
    bar = None
    last_eval = 0
    if show_progress:
        bar = tqdm(total=optimizer_eval_budget(optimizer_name), desc=label or f"{backend_mode}:{optimizer_name}", leave=False)

    def callback(eval_count, parameters, mean, metadata):
        nonlocal last_eval
        counts.append(int(eval_count))
        energies.append(float(mean))
        if bar is not None:
            inc = max(0, int(eval_count) - last_eval)
            if inc:
                bar.update(inc)
                last_eval = int(eval_count)
            bar.set_postfix(best=f"{min(energies):.6f}")

    result = VQE(
        estimator=estimator,
        ansatz=ansatz,
        optimizer=optimizer,
        initial_point=initial_point,
        callback=callback,
    ).compute_minimum_eigenvalue(qubit_hamiltonian)

    if bar is not None:
        bar.close()

    return {
        "energy": float(result.eigenvalue.real),
        "counts": counts,
        "energies": energies,
        "optimal_point": [float(x) for x in result.optimal_point],
        "ansatz_num_parameters": int(ansatz.num_parameters),
        "ansatz_depth": int(ansatz.depth()) if ansatz_kind == "hardware_efficient" else None,
        "ansatz_kind": ansatz_kind,
        "optimizer_name": optimizer_name,
        "backend_mode": backend_mode,
        "seed": seed,
    }


def sweep(values: list[Any], run_fn: Callable[[Any], dict[str, Any]]) -> dict[Any, dict[str, Any]]:
    return {value: run_fn(value) for value in values}


def repeat(num_trials: int, run_fn: Callable[[int | None], dict[str, Any]], base_seed: int | None = RANDOM_SEED) -> list[dict[str, Any]]:
    results = []
    for i in range(num_trials):
        trial_seed = None if base_seed is None else base_seed + i
        result = run_fn(trial_seed)
        result["trial_index"] = i
        result["seed"] = trial_seed
        results.append(result)
    return results
