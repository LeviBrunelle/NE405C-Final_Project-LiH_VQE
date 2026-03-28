from __future__ import annotations

from typing import Any

from .build_hamiltonian import get_problem_bundle, summarize_problem
from .run_exact import run_exact
from .run_vqe import run_vqe


def scan_bond_lengths(
    bond_lengths: list[float],
    reps: int,
    simplify_hamiltonian: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for bond_length in bond_lengths:
        bundle = get_problem_bundle(bond_length=bond_length, simplify=simplify_hamiltonian)
        problem = bundle["problem"]
        qubit_hamiltonian = bundle["qubit_hamiltonian"]

        exact = run_exact(qubit_hamiltonian)
        vqe = run_vqe(qubit_hamiltonian, reps=reps)

        summary = summarize_problem(problem, qubit_hamiltonian, bond_length)

        rows.append(
            {
                **summary,
                "exact_energy": exact["energy"],
                "vqe_energy": vqe["energy"],
                "absolute_error": abs(vqe["energy"] - exact["energy"]),
                "ansatz_num_parameters": vqe["ansatz_num_parameters"],
                "ansatz_depth": vqe["ansatz_depth"],
            }
        )

    return rows