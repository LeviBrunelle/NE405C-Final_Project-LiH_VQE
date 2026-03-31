from __future__ import annotations

import csv
import json
import statistics
from datetime import datetime
from pathlib import Path

from tqdm.auto import tqdm

from src import config as project_config
from src.build_hamiltonian import (
    get_problem_bundle,
    preview_fermionic_terms,
    preview_qubit_terms,
    summarize_problem,
)
from src.config import (
    ANSATZ_FOLD,
    ANSATZ_MODES_TO_COMPARE,
    BASIS,
    BASIS_SETS_TO_COMPARE,
    BOND_LENGTHS,
    DEFAULT_ANSATZ_MODE,
    DEFAULT_BOND_LENGTH,
    DEFAULT_REPS,
    FIG_DPI,
    MAX_EXACT_QUBITS,
    NUM_TRIALS,
    OPTIMIZERS_TO_COMPARE,
    OPTIMIZER_NAME,
    PREVIEW_TERM_COUNT,
    RANDOM_SEED,
    REPS_EXPERIMENT,
    RUN_ANSATZ_MODE_COMPARISON,
    RUN_BASIS_COMPARISON,
    RUN_BOND_SCAN,
    RUN_LABEL,
    RUN_OPTIMIZER_EXPERIMENT,
    RUN_OPTIMIZER_REPS_GRID,
    RUN_NOISE_EXPERIMENT,
    RUN_REPEATED_ANSATZ_MODE_EXPERIMENT,
    RUN_REPEATED_OPTIMIZER_EXPERIMENT,
    RUN_REPEATED_SINGLE_POINT,
    RUN_REPS_EXPERIMENT,
    RUN_SINGLE_POINT,
    SAVE_ANSATZ_FIGURE,
    SAVE_ANSATZ_MODE_ERROR_PLOT,
    SAVE_BASIS_ERROR_PLOT,
    SAVE_BOND_ERROR_PLOT,
    SAVE_BOND_SCAN_PLOT,
    SAVE_CONVERGENCE_PLOT,
    SAVE_HAMILTONIAN_PREVIEW,
    SAVE_OPTIMIZER_OVERLAY_PLOT,
    SAVE_REPEATED_TRIALS_PLOT,
    SAVE_REPEATED_TRIALS_SUMMARY,
    SAVE_REPS_ERROR_PLOT,
    SAVE_REPS_OVERLAY_PLOT,
    USE_TQDM,
    VQE_BACKEND_MODE,
    BACKEND_MODES_TO_COMPARE,
    SAVE_NOISE_EXPERIMENT_PLOT,
    ensure_results_dir,
)
from src.plot_results import (
    plot_ansatz_mode_error,
    plot_ansatz_mode_final_energy,
    plot_ansatz_mode_overlay,
    plot_basis_error,
    plot_bond_error,
    plot_bond_scan,
    plot_convergence,
    plot_optimizer_overlay,
    plot_optimizer_reps_overlay,
    plot_repeated_trials_comparison,
    plot_repeated_trials_mean_std,
    plot_reps_error,
    plot_reps_overlay,
    plot_backend_mode_optimizer_grid,
)
from src.run_exact import run_exact
from src.run_vqe import (
    build_ansatz,
    run_ansatz_mode_experiment,
    run_backend_mode_experiment,
    run_optimizer_experiment,
    run_optimizer_reps_grid,
    run_repeated_trials,
    run_reps_experiment,
    run_vqe,
)
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


def make_run_results_dir(base_results_dir: Path, run_label: str) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = base_results_dir / f"{timestamp}_{run_label}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def dump_config_snapshot(path: Path) -> None:
    snapshot = {}
    for name in dir(project_config):
        if name.isupper():
            value = getattr(project_config, name)
            if isinstance(value, (str, int, float, bool, list, tuple, type(None))):
                snapshot[name] = value
    write_json(path / "config_snapshot.json", snapshot)


def summarize_repeated_trials(trial_results: list[dict], exact_energy: float) -> dict:
    final_energies = [result["energy"] for result in trial_results]
    abs_errors = [abs(result["energy"] - exact_energy) for result in trial_results]

    return {
        "num_trials": len(trial_results),
        "mean_final_energy": statistics.mean(final_energies),
        "std_final_energy": statistics.pstdev(final_energies) if len(final_energies) > 1 else 0.0,
        "mean_absolute_error": statistics.mean(abs_errors),
        "std_absolute_error": statistics.pstdev(abs_errors) if len(abs_errors) > 1 else 0.0,
        "trial_final_energies": final_energies,
        "trial_absolute_errors": abs_errors,
        "trial_seeds": [result["random_seed"] for result in trial_results],
    }


def main() -> None:
    base_results_dir = ensure_results_dir()
    results_dir = make_run_results_dir(base_results_dir, RUN_LABEL)
    dump_config_snapshot(results_dir)

    print(f"Writing results to: {results_dir}")
    print("[1/9] Building default LiH problem bundle...")

    bundle = get_problem_bundle(DEFAULT_BOND_LENGTH, basis=BASIS, simplify=True)
    problem = bundle["problem"]
    mapper = bundle["mapper"]
    fermionic_hamiltonian = bundle["fermionic_hamiltonian"]
    qubit_hamiltonian = bundle["qubit_hamiltonian"]

    summary = summarize_problem(problem, qubit_hamiltonian, DEFAULT_BOND_LENGTH, BASIS)
    summary["backend_mode"] = VQE_BACKEND_MODE
    molecule_name = summary["molecule_name"]
    exact = run_exact(qubit_hamiltonian)

    if SAVE_HAMILTONIAN_PREVIEW:
        write_json(
            results_dir / "fermionic_terms_preview.json",
            preview_fermionic_terms(fermionic_hamiltonian, limit=PREVIEW_TERM_COUNT),
        )
        write_json(
            results_dir / "qubit_terms_preview.json",
            preview_qubit_terms(qubit_hamiltonian, limit=PREVIEW_TERM_COUNT),
        )

    if RUN_SINGLE_POINT:
        print("[2/9] Running single-point VQE...")
        vqe = run_vqe(
            qubit_hamiltonian,
            reps=DEFAULT_REPS,
            optimizer_name=OPTIMIZER_NAME,
            ansatz_mode=DEFAULT_ANSATZ_MODE,
            problem=problem,
            mapper=mapper,
            show_progress=USE_TQDM,
            progress_label=f"single {DEFAULT_ANSATZ_MODE} {OPTIMIZER_NAME}",
            random_seed=RANDOM_SEED,
            backend_mode=VQE_BACKEND_MODE,
        )

        single_run_summary = {
            **summary,
            "optimizer_name": OPTIMIZER_NAME,
            "ansatz_mode": DEFAULT_ANSATZ_MODE,
            "backend_mode": VQE_BACKEND_MODE,
            "exact_energy": exact["energy"],
            "vqe_energy": vqe["energy"],
            "absolute_error": abs(vqe["energy"] - exact["energy"]),
            "ansatz_num_parameters": vqe["ansatz_num_parameters"],
            "ansatz_depth": vqe["ansatz_depth"],
            "optimal_point": vqe["optimal_point"],
            "random_seed": RANDOM_SEED,
        }

        write_json(results_dir / "single_run_summary.json", single_run_summary)

        if SAVE_CONVERGENCE_PLOT:
            plot_convergence(
                counts=vqe["counts"],
                energies=vqe["energies"],
                exact_energy=exact["energy"],
                output_path=results_dir / f"{molecule_name.lower()}_vqe_convergence.png",
                title=f"VQE convergence for {molecule_name}",
            )

        if SAVE_ANSATZ_FIGURE:
            ansatz, _ = build_ansatz(
                num_qubits=qubit_hamiltonian.num_qubits,
                reps=DEFAULT_REPS,
                ansatz_mode=DEFAULT_ANSATZ_MODE,
                problem=problem,
                mapper=mapper,
            )

            try:
                if DEFAULT_ANSATZ_MODE == "uccsd_hf":
                    circuit_to_draw = ansatz
                    fig = circuit_to_draw.draw(
                        output="mpl",
                        fold=20,
                        idle_wires=False,
                    )
                else:
                    circuit_to_draw = ansatz.decompose()
                    fig = circuit_to_draw.draw(
                        output="mpl",
                        fold=ANSATZ_FOLD,
                        idle_wires=False,
                    )

                fig.savefig(
                    results_dir / f"{molecule_name}_{DEFAULT_ANSATZ_MODE}_ansatz_circuit.png",
                    dpi=FIG_DPI,
                    bbox_inches="tight",
                )
            except Exception as exc:
                print(f"Warning: could not save ansatz figure: {exc}")

            try:
                text_diagram = circuit_to_draw.draw(output="text", fold=120)
                with open(results_dir / f"{molecule_name}_{DEFAULT_ANSATZ_MODE}_ansatz_circuit.txt", "w", encoding="utf-8") as f:
                    f.write(str(text_diagram))
            except Exception as exc:
                print(f"Warning: could not save text ansatz diagram: {exc}")

        print("Single-run summary:")
        for key, value in single_run_summary.items():
            print(f"{key}: {value}")

    if RUN_REPEATED_SINGLE_POINT:
        print("[3/9] Running repeated single-point trials...")
        repeated_trials = run_repeated_trials(
            qubit_hamiltonian=qubit_hamiltonian,
            num_trials=NUM_TRIALS,
            reps=DEFAULT_REPS,
            optimizer_name=OPTIMIZER_NAME,
            ansatz_mode=DEFAULT_ANSATZ_MODE,
            problem=problem,
            mapper=mapper,
            show_progress=USE_TQDM,
            base_seed=RANDOM_SEED,
            backend_mode=VQE_BACKEND_MODE,
        )

        repeated_summary = {
            **summary,
            "optimizer_name": OPTIMIZER_NAME,
            "ansatz_mode": DEFAULT_ANSATZ_MODE,
            "backend_mode": VQE_BACKEND_MODE,
            **summarize_repeated_trials(repeated_trials, exact["energy"]),
        }

        if SAVE_REPEATED_TRIALS_SUMMARY:
            write_json(results_dir / "repeated_single_point_summary.json", repeated_summary)

        if SAVE_REPEATED_TRIALS_PLOT:
            plot_repeated_trials_mean_std(
                trial_results=repeated_trials,
                exact_energy=exact["energy"],
                output_path=results_dir / "repeated_single_point_mean_std.png",
                title=f"Repeated trials: {DEFAULT_ANSATZ_MODE} + {OPTIMIZER_NAME}",
            )

        print("Repeated single-point summary:")
        for key, value in repeated_summary.items():
            print(f"{key}: {value}")

    if RUN_REPS_EXPERIMENT:
        print("[4/9] Running reps experiment...")
        reps_results = run_reps_experiment(
            qubit_hamiltonian,
            reps_values=REPS_EXPERIMENT,
            optimizer_name=OPTIMIZER_NAME,
            ansatz_mode=DEFAULT_ANSATZ_MODE,
            problem=problem,
            mapper=mapper,
            show_progress=USE_TQDM,
            backend_mode=VQE_BACKEND_MODE,
        )

        reps_summary = {
            reps: {
                "optimizer_name": OPTIMIZER_NAME,
                "ansatz_mode": DEFAULT_ANSATZ_MODE,
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
                title="VQE convergence with different ansatz depths",
            )

        if SAVE_REPS_ERROR_PLOT:
            plot_reps_error(
                reps_summary=reps_summary,
                output_path=results_dir / "lih_reps_error.png",
                title="Absolute error vs ansatz depth",
            )

        print("\nReps experiment summary:")
        for reps, row in reps_summary.items():
            print(f"reps={reps}: {row}")

    if RUN_OPTIMIZER_EXPERIMENT:
        print("[5/9] Running optimizer experiment...")
        optimizer_results = run_optimizer_experiment(
            qubit_hamiltonian,
            optimizer_names=OPTIMIZERS_TO_COMPARE,
            reps=DEFAULT_REPS,
            ansatz_mode=DEFAULT_ANSATZ_MODE,
            problem=problem,
            mapper=mapper,
            show_progress=USE_TQDM,
            backend_mode=VQE_BACKEND_MODE,
        )

        optimizer_summary = {
            optimizer_name: {
                "energy": data["energy"],
                "absolute_error": abs(data["energy"] - exact["energy"]),
                "ansatz_num_parameters": data["ansatz_num_parameters"],
                "ansatz_depth": data["ansatz_depth"],
            }
            for optimizer_name, data in optimizer_results.items()
        }
        write_json(results_dir / "optimizer_experiment_summary.json", optimizer_summary)

        if SAVE_OPTIMIZER_OVERLAY_PLOT:
            plot_optimizer_overlay(
                histories=optimizer_results,
                exact_energy=exact["energy"],
                output_path=results_dir / "lih_optimizer_overlay.png",
                title="VQE convergence with different optimizers",
            )

        print("\nOptimizer experiment summary:")
        for optimizer_name, row in optimizer_summary.items():
            print(f"{optimizer_name}: {row}")

    if RUN_REPEATED_OPTIMIZER_EXPERIMENT:
        print("[6/9] Running repeated optimizer comparison...")
        repeated_optimizer_results = {}

        for optimizer_name in OPTIMIZERS_TO_COMPARE:
            print(f"  optimizer={optimizer_name}")
            repeated_optimizer_results[optimizer_name] = run_repeated_trials(
                qubit_hamiltonian=qubit_hamiltonian,
                num_trials=NUM_TRIALS,
                reps=DEFAULT_REPS,
                optimizer_name=optimizer_name,
                ansatz_mode=DEFAULT_ANSATZ_MODE,
                problem=problem,
                mapper=mapper,
                show_progress=USE_TQDM,
                base_seed=RANDOM_SEED,
                backend_mode=VQE_BACKEND_MODE,
            )

        repeated_optimizer_summary = {
            optimizer_name: summarize_repeated_trials(trials, exact["energy"])
            for optimizer_name, trials in repeated_optimizer_results.items()
        }

        if SAVE_REPEATED_TRIALS_SUMMARY:
            write_json(results_dir / "repeated_optimizer_summary.json", repeated_optimizer_summary)

        if SAVE_REPEATED_TRIALS_PLOT:
            plot_repeated_trials_comparison(
                grouped_trial_results=repeated_optimizer_results,
                exact_energy=exact["energy"],
                output_path=results_dir / "repeated_optimizer_comparison.png",
                title=f"Repeated optimizer comparison ({DEFAULT_ANSATZ_MODE})",
            )

        print("\nRepeated optimizer summary:")
        for optimizer_name, row in repeated_optimizer_summary.items():
            print(f"{optimizer_name}: {row}")

    if RUN_BOND_SCAN:
        print("[7/9] Running bond-length scan...")
        bond_scan_rows = scan_bond_lengths(
            BOND_LENGTHS,
            reps=DEFAULT_REPS,
            optimizer_name=OPTIMIZER_NAME,
            ansatz_mode=DEFAULT_ANSATZ_MODE,
            basis=BASIS,
            simplify_hamiltonian=True,
            show_progress=USE_TQDM,
            backend_mode=VQE_BACKEND_MODE,
        )
        write_csv(results_dir / "bond_scan.csv", bond_scan_rows)

        if SAVE_BOND_SCAN_PLOT:
            plot_bond_scan(
                rows=bond_scan_rows,
                output_path=results_dir / f"{molecule_name.lower()}_bond_scan.png",
                title=f"{molecule_name} energy vs bond length",
            )

        if SAVE_BOND_ERROR_PLOT:
            plot_bond_error(
                rows=bond_scan_rows,
                output_path=results_dir / f"{molecule_name.lower()}_bond_error.png",
                title=f"VQE absolute error vs bond length",
            )

    if RUN_BASIS_COMPARISON:
        print("[8/9] Running basis comparison...")
        basis_rows = []

        basis_iterator = tqdm(BASIS_SETS_TO_COMPARE, desc="Basis comparison", leave=False) if USE_TQDM else BASIS_SETS_TO_COMPARE

        for basis in basis_iterator:
            if USE_TQDM:
                basis_iterator.set_postfix({"basis": basis})
            else:
                print(f"  basis={basis}")

            print(f"\n--- basis={basis}: building problem bundle ---")
            basis_bundle = get_problem_bundle(DEFAULT_BOND_LENGTH, basis=basis, simplify=True)

            print(f"--- basis={basis}: bundle built ---")
            basis_problem = basis_bundle["problem"]
            basis_mapper = basis_bundle["mapper"]
            basis_qubit_hamiltonian = basis_bundle["qubit_hamiltonian"]

            print(
                f"--- basis={basis}: "
                f"spin_orbitals={basis_problem.num_spin_orbitals} | "
                f"qubits={basis_qubit_hamiltonian.num_qubits} | "
                f"pauli_terms={len(basis_qubit_hamiltonian)} ---"
            )

            basis_summary = summarize_problem(
                basis_problem,
                basis_qubit_hamiltonian,
                DEFAULT_BOND_LENGTH,
                basis,
            )

            exact_energy = None
            vqe_energy = None
            absolute_error = None
            ran_exact = False
            ran_vqe = False
            ansatz_num_parameters = None
            ansatz_depth = None

            if basis_qubit_hamiltonian.num_qubits <= MAX_EXACT_QUBITS:
                print(f"--- basis={basis}: running exact solver ---")
                basis_exact = run_exact(basis_qubit_hamiltonian)
                exact_energy = basis_exact["energy"]
                ran_exact = True

                print(f"--- basis={basis}: exact done, running VQE ---")
                basis_vqe = run_vqe(
                    basis_qubit_hamiltonian,
                    reps=DEFAULT_REPS,
                    optimizer_name=OPTIMIZER_NAME,
                    ansatz_mode=DEFAULT_ANSATZ_MODE,
                    problem=basis_problem,
                    mapper=basis_mapper,
                    show_progress=USE_TQDM,
                    progress_label=f"VQE basis={basis}",
                    random_seed=RANDOM_SEED,
                    backend_mode=VQE_BACKEND_MODE,
                )
                vqe_energy = basis_vqe["energy"]
                absolute_error = abs(vqe_energy - exact_energy)
                ran_vqe = True
                ansatz_num_parameters = basis_vqe["ansatz_num_parameters"]
                ansatz_depth = basis_vqe["ansatz_depth"]

                print(f"--- basis={basis}: VQE done ---")
            else:
                print(
                    f"--- basis={basis}: skipping exact/VQE because "
                    f"num_qubits={basis_qubit_hamiltonian.num_qubits} exceeds MAX_EXACT_QUBITS={MAX_EXACT_QUBITS} ---"
                )

            basis_rows.append(
                {
                    **basis_summary,
                    "ansatz_mode": DEFAULT_ANSATZ_MODE,
                    "optimizer_name": OPTIMIZER_NAME,
                    "ran_exact": ran_exact,
                    "ran_vqe": ran_vqe,
                    "exact_energy": exact_energy,
                    "vqe_energy": vqe_energy,
                    "absolute_error": absolute_error,
                    "ansatz_num_parameters": ansatz_num_parameters,
                    "ansatz_depth": ansatz_depth,
                }
            )

        write_csv(results_dir / "basis_comparison.csv", basis_rows)

        if SAVE_BASIS_ERROR_PLOT:
            plot_basis_error(
                rows=basis_rows,
                output_path=results_dir / "lih_basis_error.png",
                title="Absolute error vs basis set",
            )

        print("\nBasis comparison:")
        for row in basis_rows:
            print(row)

    if RUN_ANSATZ_MODE_COMPARISON:
        print("[9/9] Running ansatz-mode comparison...")
        ansatz_mode_results = run_ansatz_mode_experiment(
            qubit_hamiltonian=qubit_hamiltonian,
            ansatz_modes=ANSATZ_MODES_TO_COMPARE,
            reps=DEFAULT_REPS,
            optimizer_name=OPTIMIZER_NAME,
            problem=problem,
            mapper=mapper,
            show_progress=USE_TQDM,
            backend_mode=VQE_BACKEND_MODE,
        )

        ansatz_mode_rows = []
        for ansatz_mode, data in ansatz_mode_results.items():
            ansatz_mode_rows.append(
                {
                    **summary,
                    "ansatz_mode": ansatz_mode,
                    "optimizer_name": OPTIMIZER_NAME,
                    "backend_mode": VQE_BACKEND_MODE,
                    "exact_energy": exact["energy"],
                    "vqe_energy": data["energy"],
                    "absolute_error": abs(data["energy"] - exact["energy"]),
                    "ansatz_num_parameters": data["ansatz_num_parameters"],
                    "ansatz_depth": data["ansatz_depth"],
                }
            )

        write_csv(results_dir / "ansatz_mode_comparison.csv", ansatz_mode_rows)

        if SAVE_ANSATZ_MODE_ERROR_PLOT:
            plot_ansatz_mode_error(
                rows=ansatz_mode_rows,
                output_path=results_dir / "lih_ansatz_mode_error.png",
                title="Absolute error vs ansatz mode",
            )

        plot_ansatz_mode_final_energy(
            rows=ansatz_mode_rows,
            output_path=results_dir / "lih_ansatz_mode_final_energy.png",
            title="Final energy vs ansatz mode",
        )

        plot_ansatz_mode_overlay(
            histories=ansatz_mode_results,
            exact_energy=exact["energy"],
            output_path=results_dir / "lih_ansatz_mode_overlay.png",
            title="VQE convergence with different ansatz modes",
        )

        print("\nAnsatz mode comparison:")
        for row in ansatz_mode_rows:
            print(row)
   
    if RUN_REPEATED_ANSATZ_MODE_EXPERIMENT:
        print("[extra] Running repeated ansatz-mode comparison...")
        repeated_ansatz_results = {}

        for ansatz_mode in ANSATZ_MODES_TO_COMPARE:
            print(f"  ansatz_mode={ansatz_mode}")
            repeated_ansatz_results[ansatz_mode] = run_repeated_trials(
                qubit_hamiltonian=qubit_hamiltonian,
                num_trials=NUM_TRIALS,
                reps=DEFAULT_REPS,
                optimizer_name=OPTIMIZER_NAME,
                ansatz_mode=ansatz_mode,
                problem=problem,
                mapper=mapper,
                show_progress=USE_TQDM,
                base_seed=RANDOM_SEED,
                backend_mode=VQE_BACKEND_MODE,
            )

        repeated_ansatz_summary = {
            ansatz_mode: summarize_repeated_trials(trials, exact["energy"])
            for ansatz_mode, trials in repeated_ansatz_results.items()
        }

        if SAVE_REPEATED_TRIALS_SUMMARY:
            write_json(results_dir / "repeated_ansatz_mode_summary.json", repeated_ansatz_summary)

        if SAVE_REPEATED_TRIALS_PLOT:
            plot_repeated_trials_comparison(
                grouped_trial_results=repeated_ansatz_results,
                exact_energy=exact["energy"],
                output_path=results_dir / "repeated_ansatz_mode_comparison.png",
                title=f"Repeated ansatz-mode comparison ({OPTIMIZER_NAME})",
            )

        print("\nRepeated ansatz-mode summary:")
        for ansatz_mode, row in repeated_ansatz_summary.items():
            print(f"{ansatz_mode}: {row}")


    if RUN_NOISE_EXPERIMENT:
        print("[extra] Running backend-mode / noise comparison...")

        backend_mode_results = run_backend_mode_experiment(
            qubit_hamiltonian=qubit_hamiltonian,
            backend_modes=BACKEND_MODES_TO_COMPARE,
            optimizer_names=OPTIMIZERS_TO_COMPARE,
            reps=DEFAULT_REPS,
            ansatz_mode=DEFAULT_ANSATZ_MODE,
            problem=problem,
            mapper=mapper,
            show_progress=USE_TQDM,
        )

        backend_mode_summary = {}
        for backend_mode, optimizer_histories in backend_mode_results.items():
            backend_mode_summary[backend_mode] = {}
            for optimizer_name, data in optimizer_histories.items():
                backend_mode_summary[backend_mode][optimizer_name] = {
                    "energy": data["energy"],
                    "absolute_error": abs(data["energy"] - exact["energy"]),
                    "ansatz_num_parameters": data["ansatz_num_parameters"],
                    "ansatz_depth": data["ansatz_depth"],
                }

        write_json(results_dir / "backend_mode_experiment_summary.json", backend_mode_summary)

        if SAVE_NOISE_EXPERIMENT_PLOT:
            plot_backend_mode_optimizer_grid(
                histories_by_backend=backend_mode_results,
                exact_energy=exact["energy"],
                output_path=results_dir / f"{molecule_name.lower()}_backend_mode_optimizer_grid.png",
                title=f"{molecule_name}: exact vs Aer shots vs Aer noise",
            )

        print("\nBackend-mode comparison:")
        for backend_mode, optimizer_rows in backend_mode_summary.items():
            print(backend_mode)
            for optimizer_name, row in optimizer_rows.items():
                print(f"  {optimizer_name}: {row}")

    if RUN_OPTIMIZER_REPS_GRID:
        print("[extra] Running optimizer-reps grid...")

        optimizer_reps_results = run_optimizer_reps_grid(
            qubit_hamiltonian=qubit_hamiltonian,
            optimizer_names=OPTIMIZERS_TO_COMPARE,
            reps_values=REPS_EXPERIMENT,
            ansatz_mode=DEFAULT_ANSATZ_MODE,
            problem=problem,
            mapper=mapper,
            show_progress=USE_TQDM,
            backend_mode=VQE_BACKEND_MODE,
        )

        optimizer_reps_summary = {}
        for optimizer_name, reps_dict in optimizer_reps_results.items():
            optimizer_reps_summary[optimizer_name] = {}
            for reps, data in reps_dict.items():
                optimizer_reps_summary[optimizer_name][reps] = {
                    "energy": data["energy"],
                    "absolute_error": abs(data["energy"] - exact["energy"]),
                    "ansatz_num_parameters": data["ansatz_num_parameters"],
                    "ansatz_depth": data["ansatz_depth"],
                }

        write_json(results_dir / "optimizer_reps_grid_summary.json", optimizer_reps_summary)

        plot_optimizer_reps_overlay(
            histories=optimizer_reps_results,
            exact_energy=exact["energy"],
            output_path=results_dir / "lih_optimizer_reps_overlay.png",
            title=f"Optimizer vs ansatz depth ({DEFAULT_ANSATZ_MODE})",
            use_running_best=True,
        )

        print("\nOptimizer × reps summary:")
        for optimizer_name, reps_dict in optimizer_reps_summary.items():
            print(optimizer_name)
            for reps, row in reps_dict.items():
                print(f"  reps={reps}: {row}")

    print("\nSaved outputs to:", results_dir)


if __name__ == "__main__":
    main()