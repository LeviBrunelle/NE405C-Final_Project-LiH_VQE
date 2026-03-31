from __future__ import annotations

from typing import Any

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.units import DistanceUnit

from .config import BASIS, CHARGE, MOLECULE_NAME, PREVIEW_TERM_COUNT, SPIN


def build_geometry(bond_length: float) -> str:
    if MOLECULE_NAME == "LiH":
        return f"Li 0 0 0; H 0 0 {bond_length}"
    if MOLECULE_NAME == "BeH2":
        return f"H 0 0 {-bond_length}; Be 0 0 0; H 0 0 {bond_length}"
    raise ValueError(f"Unsupported molecule: {MOLECULE_NAME}")


def build_problem(bond_length: float, basis: str | None = None) -> Any:
    driver = PySCFDriver(
        atom=build_geometry(bond_length),
        basis=basis or BASIS,
        charge=CHARGE,
        spin=SPIN,
        unit=DistanceUnit.ANGSTROM,
    )
    return driver.run()


def get_mapper() -> JordanWignerMapper:
    return JordanWignerMapper()


def get_problem_bundle(bond_length: float, basis: str | None = None, simplify: bool = True) -> dict[str, Any]:
    problem = build_problem(bond_length, basis=basis)
    fermionic_hamiltonian = problem.hamiltonian.second_q_op()
    mapper = get_mapper()
    qubit_hamiltonian = mapper.map(fermionic_hamiltonian)
    if simplify:
        qubit_hamiltonian = qubit_hamiltonian.simplify()
    return {
        "problem": problem,
        "mapper": mapper,
        "fermionic_hamiltonian": fermionic_hamiltonian,
        "qubit_hamiltonian": qubit_hamiltonian,
        "basis": basis or BASIS,
        "bond_length": bond_length,
    }


def preview_fermionic_terms(fermionic_hamiltonian: Any, limit: int = PREVIEW_TERM_COUNT) -> list[dict[str, Any]]:
    out = []
    for i, (label, coeff) in enumerate(fermionic_hamiltonian.items()):
        out.append({"index": i, "label": label, "coefficient_real": float(coeff.real)})
        if i + 1 >= limit:
            break
    return out


def preview_qubit_terms(qubit_hamiltonian: Any, limit: int = PREVIEW_TERM_COUNT) -> list[dict[str, Any]]:
    out = []
    for i, (pauli, coeff) in enumerate(qubit_hamiltonian.to_list()):
        out.append({"index": i, "pauli": pauli, "coefficient_real": float(coeff.real)})
        if i + 1 >= limit:
            break
    return out


def summarize_problem(problem: Any, qubit_hamiltonian: Any, bond_length: float, basis: str) -> dict[str, Any]:
    return {
        "molecule_name": MOLECULE_NAME,
        "bond_length_angstrom": bond_length,
        "basis": basis,
        "num_spatial_orbitals": int(problem.num_spatial_orbitals),
        "num_spin_orbitals": int(problem.num_spin_orbitals),
        "num_particles_alpha": int(problem.num_particles[0]),
        "num_particles_beta": int(problem.num_particles[1]),
        "num_qubits": int(qubit_hamiltonian.num_qubits),
        "num_pauli_terms": int(len(qubit_hamiltonian)),
    }
