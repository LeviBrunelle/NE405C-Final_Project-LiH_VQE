from __future__ import annotations

from typing import Any

from tqdm.auto import tqdm

from .build_hamiltonian import get_problem_bundle, summarize_problem
from .run_exact import run_exact
from .run_vqe import run_vqe


def scan_bond_lengths(
    bond_lengths: list[float],
    *,
    ansatz_kind: str,
    optimizer_name: str,
    reps: int,
    backend_mode: str,
    basis: str,
    show_progress: bool,
) -> list[dict[str, Any]]:
    rows = []
    iterator = tqdm(bond_lengths, desc="Bond scan", leave=False) if show_progress else bond_lengths
    for bond_length in iterator:
        bundle = get_problem_bundle(bond_length=bond_length, basis=basis, simplify=True)
        problem, mapper, qubit_h = bundle["problem"], bundle["mapper"], bundle["qubit_hamiltonian"]
        exact = run_exact(qubit_h)
        vqe = run_vqe(
            qubit_h,
            ansatz_kind=ansatz_kind,
            optimizer_name=optimizer_name,
            reps=reps,
            backend_mode=backend_mode,
            problem=problem,
            mapper=mapper,
            show_progress=False,
        )
        rows.append({
            **summarize_problem(problem, qubit_h, bond_length, basis),
            "ansatz_kind": ansatz_kind,
            "optimizer_name": optimizer_name,
            "backend_mode": backend_mode,
            "exact_energy": exact["energy"],
            "vqe_energy": vqe["energy"],
            "absolute_error": abs(vqe["energy"] - exact["energy"]),
            "ansatz_num_parameters": vqe["ansatz_num_parameters"],
            "ansatz_depth": vqe["ansatz_depth"],
        })
    return rows
