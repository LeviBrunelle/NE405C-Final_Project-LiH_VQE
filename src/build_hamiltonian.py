from __future__ import annotations

from typing import Any

from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper

from .config import BASIS, CHARGE, PREVIEW_TERM_COUNT, SPIN


def build_driver(bond_length: float) -> PySCFDriver:
    return PySCFDriver(
        atom=f"Li 0 0 0; H 0 0 {bond_length}",
        basis=BASIS,
        charge=CHARGE,
        spin=SPIN,
        unit=DistanceUnit.ANGSTROM,
    )


def build_problem(bond_length: float) -> Any:
    driver = build_driver(bond_length)
    return driver.run()


def get_fermionic_hamiltonian(problem: Any) -> Any:
    return problem.hamiltonian.second_q_op()


def get_mapper() -> JordanWignerMapper:
    return JordanWignerMapper()


def get_qubit_hamiltonian(problem: Any, simplify: bool = True) -> Any:
    fermionic_hamiltonian = get_fermionic_hamiltonian(problem)
    mapper = get_mapper()
    qubit_hamiltonian = mapper.map(fermionic_hamiltonian)
    if simplify:
        qubit_hamiltonian = qubit_hamiltonian.simplify()
    return qubit_hamiltonian


def get_problem_bundle(bond_length: float, simplify: bool = True) -> dict[str, Any]:
    problem = build_problem(bond_length)
    fermionic_hamiltonian = get_fermionic_hamiltonian(problem)
    qubit_hamiltonian = get_qubit_hamiltonian(problem, simplify=simplify)

    return {
        "bond_length": bond_length,
        "problem": problem,
        "fermionic_hamiltonian": fermionic_hamiltonian,
        "qubit_hamiltonian": qubit_hamiltonian,
    }


def preview_fermionic_terms(fermionic_hamiltonian: Any, limit: int = PREVIEW_TERM_COUNT) -> list[dict[str, float | str]]:
    terms: list[dict[str, float | str]] = []
    for i, (label, coeff) in enumerate(fermionic_hamiltonian.items()):
        terms.append(
            {
                "index": i,
                "label": label,
                "coefficient_real": float(coeff.real),
            }
        )
        if i + 1 >= limit:
            break
    return terms


def preview_qubit_terms(qubit_hamiltonian: Any, limit: int = PREVIEW_TERM_COUNT) -> list[dict[str, float | str]]:
    terms: list[dict[str, float | str]] = []
    for i, (pauli, coeff) in enumerate(qubit_hamiltonian.to_list()):
        terms.append(
            {
                "index": i,
                "pauli": pauli,
                "coefficient_real": float(coeff.real),
            }
        )
        if i + 1 >= limit:
            break
    return terms


def summarize_problem(problem: Any, qubit_hamiltonian: Any, bond_length: float) -> dict[str, Any]:
    return {
        "bond_length_angstrom": bond_length,
        "num_spin_orbitals": int(problem.num_spin_orbitals),
        "num_particles_alpha": int(problem.num_particles[0]),
        "num_particles_beta": int(problem.num_particles[1]),
        "num_qubits": int(qubit_hamiltonian.num_qubits),
        "num_pauli_terms": int(len(qubit_hamiltonian)),
    }