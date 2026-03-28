from __future__ import annotations

from typing import Any

from tqdm.auto import tqdm

from .build_hamiltonian import get_problem_bundle, summarize_problem
from .run_exact import run_exact
from .run_vqe import run_vqe


def scan_bond_lengths(
    bond_lengths: list[float],
    reps: int,
    optimizer_name: str,
    simplify_hamiltonian: bool = True,
    show_progress: bool = False,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    iterator = tqdm(bond_lengths, desc="Bond scan", leave=False) if show_progress else bond_lengths

    for bond_length in iterator:
        bundle = get_problem_bundle(bond_length=bond_length, simplify=simplify_hamiltonian)
        problem = bundle["problem"]
        qubit_hamiltonian = bundle["qubit_hamiltonian"]

        exact = run_exact(qubit_hamiltonian)
        vqe = run_vqe(
            qubit_hamiltonian,
            reps=reps,
            optimizer_name=optimizer_name,
            show_progress=False,
        )

        summary = summarize_problem(problem, qubit_hamiltonian, bond_length)

        rows.append(
            {
                **summary,
                "optimizer_name": optimizer_name,
                "exact_energy": exact["energy"],
                "vqe_energy": vqe["energy"],
                "absolute_error": abs(vqe["energy"] - exact["energy"]),
                "ansatz_num_parameters": vqe["ansatz_num_parameters"],
                "ansatz_depth": vqe["ansatz_depth"],
            }
        )

    return rows