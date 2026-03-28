from __future__ import annotations

import csv
import json
from pathlib import Path

from src.build_hamiltonian import (
    get_problem_bundle,
    preview_fermionic_terms,
    preview_qubit_terms,
    summarize_problem,
)
from src.config import (
    ANSATZ_FOLD,
    BOND_LENGTHS,
    DEFAULT_BOND_LENGTH,
    DEFAULT_REPS,
    FIG_DPI,
    PREVIEW_TERM_COUNT,
    REPS_EXPERIMENT,
    SAVE_ANSATZ_FIGURE,
    SAVE_BOND_SCAN_PLOT,
    SAVE_CONVERGENCE_PLOT,
    SAVE_HAMILTONIAN_PREVIEW,
    SAVE_REPS_OVERLAY_PLOT,
    ensure_results_dir,
)
from src.plot_results import plot_bond_scan, plot_convergence, plot_reps_overlay
from src.run_exact import run_exact
from src.run_vqe import build_ansatz, run_reps_experiment, run_vqe
from src.scan_bond_lengths import scan_bond_lengths


def write_json(path: Path, data) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    results_dir = ensure_results_dir()

    bundle = get_problem_bundle(DEFAULT_BOND_LENGTH, simplify=True)
    problem = bundle["problem"]
    fermionic_hamiltonian = bundle["fermionic_hamiltonian"]
    qubit_hamiltonian = bundle["qubit_hamiltonian"]

    summary = summarize_problem(problem, qubit_hamiltonian, DEFAULT_BOND_LENGTH)
    exact = run_exact(qubit_hamiltonian)
    vqe = run_vqe(qubit_hamiltonian, reps=DEFAULT_REPS)

    single_run_summary = {
        **summary,
        "exact_energy": exact["energy"],
        "vqe_energy": vqe["energy"],
        "absolute_error": abs(vqe["energy"] - exact["energy"]),
        "ansatz_num_parameters": vqe["ansatz_num_parameters"],
        "ansatz_depth": vqe["ansatz_depth"],
        "optimal_point": vqe["optimal_point"],
    }

    write_json(results_dir / "single_run_summary.json", single_run_summary)

    if SAVE_HAMILTONIAN_PREVIEW:
        write_json(
            results_dir / "fermionic_terms_preview.json",
            preview_fermionic_terms(fermionic_hamiltonian, limit=PREVIEW_TERM_COUNT),
        )
        write_json(
            results_dir / "qubit_terms_preview.json",
            preview_qubit_terms(qubit_hamiltonian, limit=PREVIEW_TERM_COUNT),
        )

    if SAVE_CONVERGENCE_PLOT:
        plot_convergence(
            counts=vqe["counts"],
            energies=vqe["energies"],
            exact_energy=exact["energy"],
            output_path=results_dir / "lih_vqe_convergence.png",
        )

    if SAVE_ANSATZ_FIGURE:
        ansatz = build_ansatz(qubit_hamiltonian.num_qubits, reps=DEFAULT_REPS)
        fig = ansatz.decompose().draw(output="mpl", fold=ANSATZ_FOLD)
        fig.savefig(results_dir / "lih_ansatz_circuit.png", dpi=FIG_DPI, bbox_inches="tight")

    reps_results = run_reps_experiment(qubit_hamiltonian, reps_values=REPS_EXPERIMENT)

    reps_summary = {
        reps: {
            "energy": data["energy"],
            "absolute_error": abs(data["energy"] - exact["energy"]),
            "ansatz_num_parameters": data["ansatz_num_parameters"],
            "ansatz_depth": data["ansatz_depth"],
        }
        for reps, data in reps_results.items()
    }
    write_json(results_dir / "reps_experiment_summary.json", reps_summary)

    if SAVE_REPS_OVERLAY_PLOT:
        plot_reps_overlay(
            histories=reps_results,
            exact_energy=exact["energy"],
            output_path=results_dir / "lih_reps_overlay.png",
        )

    bond_scan_rows = scan_bond_lengths(BOND_LENGTHS, reps=DEFAULT_REPS, simplify_hamiltonian=True)
    write_csv(results_dir / "bond_scan.csv", bond_scan_rows)

    if SAVE_BOND_SCAN_PLOT:
        plot_bond_scan(
            rows=bond_scan_rows,
            output_path=results_dir / "lih_bond_scan.png",
        )

    print("Single-run summary:")
    for key, value in single_run_summary.items():
        print(f"{key}: {value}")

    print("\nReps experiment summary:")
    for reps, row in reps_summary.items():
        print(f"reps={reps}: {row}")

    print("\nSaved outputs to:", results_dir)


if __name__ == "__main__":
    main()