from __future__ import annotations

from typing import Any

from tqdm.auto import tqdm

from .build_hamiltonian import get_problem_bundle, summarize_problem
from .config import BASIS, VQE_BACKEND_MODE
from .run_exact import run_exact
from .run_vqe import run_vqe


def scan_bond_lengths(
    bond_lengths: list[float],
    reps: int,
    optimizer_name: str,
    ansatz_mode: str = "hardware_efficient",
    basis: str = BASIS,
    simplify_hamiltonian: bool = True,
    show_progress: bool = False,
    backend_mode: str = VQE_BACKEND_MODE,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    iterator = tqdm(bond_lengths, desc="Bond scan", leave=False) if show_progress else bond_lengths

    for bond_length in iterator:
        bundle = get_problem_bundle(
            bond_length=bond_length,
            basis=basis,
            simplify=simplify_hamiltonian,
        )
        problem = bundle["problem"]
        mapper = bundle["mapper"]
        qubit_hamiltonian = bundle["qubit_hamiltonian"]

        exact = run_exact(qubit_hamiltonian)
        vqe = run_vqe(
            qubit_hamiltonian,
            reps=reps,
            optimizer_name=optimizer_name,
            ansatz_mode=ansatz_mode,
            problem=problem,
            mapper=mapper,
            show_progress=False,
            backend_mode=backend_mode,
        )

        summary = summarize_problem(
            problem=problem,
            qubit_hamiltonian=qubit_hamiltonian,
            bond_length=bond_length,
            basis=basis,
        )

        rows.append(
            {
                **summary,
                "optimizer_name": optimizer_name,
                "ansatz_mode": ansatz_mode,
                "backend_mode": backend_mode,
                "exact_energy": exact["energy"],
                "vqe_energy": vqe["energy"],
                "absolute_error": abs(vqe["energy"] - exact["energy"]),
                "ansatz_num_parameters": vqe["ansatz_num_parameters"],
                "ansatz_depth": vqe["ansatz_depth"],
            }
        )

    return rows
