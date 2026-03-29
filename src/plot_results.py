from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_convergence(
    counts: list[int],
    energies: list[float],
    exact_energy: float,
    output_path: str | Path | None = None,
    title: str = "VQE convergence for LiH",
) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(counts, energies, marker="o", markersize=3, label="VQE")
    plt.axhline(exact_energy, linestyle="--", label="Exact energy")
    plt.xlabel("Evaluation count")
    plt.ylabel("Energy (Hartree)")
    plt.grid()
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close()


def plot_reps_overlay(
    histories: dict[int, dict],
    exact_energy: float,
    output_path: str | Path | None = None,
    title: str = "VQE convergence with different ansatz depths",
) -> None:
    plt.figure(figsize=(8, 5))

    for reps, data in histories.items():
        plt.plot(
            data["counts"],
            data["energies"],
            marker="o",
            markersize=3,
            label=f"reps={reps}",
        )

    plt.axhline(exact_energy, linestyle="--", label="Exact energy")
    plt.xlabel("Evaluation count")
    plt.ylabel("Energy (Hartree)")
    plt.grid()
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close()


def plot_optimizer_overlay(
    histories: dict[str, dict],
    exact_energy: float,
    output_path: str | Path | None = None,
    title: str = "VQE convergence with different optimizers",
) -> None:
    plt.figure(figsize=(8, 5))

    for optimizer_name, data in histories.items():
        plt.plot(
            data["counts"],
            data["energies"],
            marker="o",
            markersize=3,
            label=optimizer_name,
        )

    plt.axhline(exact_energy, linestyle="--", label="Exact energy")
    plt.xlabel("Evaluation count")
    plt.ylabel("Energy (Hartree)")
    plt.grid()
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close()


def plot_bond_scan(
    rows: list[dict],
    output_path: str | Path | None = None,
    title: str = "LiH energy vs bond length",
) -> None:
    bond_lengths = [row["bond_length_angstrom"] for row in rows]
    exact_energies = [row["exact_energy"] for row in rows]
    vqe_energies = [row["vqe_energy"] for row in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(bond_lengths, exact_energies, marker="o", label="Exact")
    plt.plot(bond_lengths, vqe_energies, marker="s", label="VQE")
    plt.xlabel("Bond length (Angstrom)")
    plt.ylabel("Energy (Hartree)")
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close()


def plot_bond_error(
    rows: list[dict],
    output_path: str | Path | None = None,
    title: str = "VQE absolute error vs bond length",
) -> None:
    bond_lengths = [row["bond_length_angstrom"] for row in rows]
    errors = [row["absolute_error"] for row in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(bond_lengths, errors, marker="o")
    plt.xlabel("Bond length (Angstrom)")
    plt.ylabel("Absolute error")
    plt.title(title)
    plt.tight_layout()
    plt.grid()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close()


def plot_reps_error(
    reps_summary: dict[int, dict],
    output_path: str | Path | None = None,
    title: str = "Absolute error vs ansatz depth",
) -> None:
    reps_values = sorted(reps_summary.keys())
    errors = [reps_summary[reps]["absolute_error"] for reps in reps_values]

    plt.figure(figsize=(8, 5))
    plt.plot(reps_values, errors, marker="o")
    plt.xlabel("reps")
    plt.ylabel("Absolute error")
    plt.title(title)
    plt.tight_layout()
    plt.grid()
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close()

def plot_basis_error(
    rows: list[dict],
    output_path: str | Path | None = None,
    title: str = "Absolute error vs basis set",
) -> None:
    basis_labels = [row["basis"] for row in rows]
    errors = [row["absolute_error"] for row in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(basis_labels, errors, marker="o")
    plt.xlabel("Basis set")
    plt.ylabel("Absolute error")
    plt.title(title)
    plt.tight_layout()
    plt.grid()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close()


def plot_ansatz_mode_error(
    rows: list[dict],
    output_path: str | Path | None = None,
    title: str = "Absolute error vs ansatz mode",
) -> None:
    labels = [row["ansatz_mode"] for row in rows]
    errors = [row["absolute_error"] for row in rows]

    plt.figure(figsize=(8, 5))
    plt.plot(labels, errors, marker="o")
    plt.xlabel("Ansatz mode")
    plt.ylabel("Absolute error")
    plt.title(title)
    plt.tight_layout()
    plt.grid()
    
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close()


def plot_repeated_trials_mean_std(
    trial_results: list[dict],
    exact_energy: float,
    output_path: str | Path | None = None,
    title: str = "Repeated-trial VQE convergence",
) -> None:
    if not trial_results:
        return

    min_len = min(len(result["energies"]) for result in trial_results)
    if min_len == 0:
        return

    energy_matrix = np.array([result["energies"][:min_len] for result in trial_results], dtype=float)
    x = np.arange(1, min_len + 1)

    mean_energy = energy_matrix.mean(axis=0)
    std_energy = energy_matrix.std(axis=0)

    final_energies = np.array([result["energy"] for result in trial_results], dtype=float)
    final_std = final_energies.std()

    plt.figure(figsize=(8, 5))
    plt.plot(x, mean_energy, label=f"Mean energy (final std={final_std:.4f} Ha)")
    plt.fill_between(x, mean_energy - std_energy, mean_energy + std_energy, alpha=0.25, label="±1 std band")
    plt.axhline(exact_energy, linestyle="--", label="Exact energy")
    plt.xlabel("Evaluation index")
    plt.ylabel("Energy (Hartree)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close()


def plot_repeated_trials_comparison(
    grouped_trial_results: dict[str, list[dict]],
    exact_energy: float,
    output_path: str | Path | None = None,
    title: str = "Repeated-trial VQE comparison",
) -> None:
    plt.figure(figsize=(8, 5))

    for label, trial_results in grouped_trial_results.items():
        if not trial_results:
            continue

        min_len = min(len(result["energies"]) for result in trial_results)
        if min_len == 0:
            continue

        energy_matrix = np.array([result["energies"][:min_len] for result in trial_results], dtype=float)
        x = np.arange(1, min_len + 1)

        mean_energy = energy_matrix.mean(axis=0)
        std_energy = energy_matrix.std(axis=0)

        final_energies = np.array([result["energy"] for result in trial_results], dtype=float)
        final_std = final_energies.std()

        plt.plot(x, mean_energy, label=f"{label} (final std={final_std:.4f} Ha)")
        plt.fill_between(x, mean_energy - std_energy, mean_energy + std_energy, alpha=0.15)

    plt.axhline(exact_energy, linestyle="--", label="Exact energy")
    plt.xlabel("Evaluation index")
    plt.ylabel("Energy (Hartree)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close()


def plot_ansatz_mode_overlay(
    histories: dict[str, dict],
    exact_energy: float,
    output_path: str | Path | None = None,
    title: str = "VQE convergence with different ansatz modes",
) -> None:
    plt.figure(figsize=(8, 5))

    for ansatz_mode, data in histories.items():
        plt.plot(
            data["counts"],
            data["energies"],
            marker="o",
            markersize=3,
            label=ansatz_mode,
        )

    plt.axhline(exact_energy, linestyle="--", label="Exact energy")
    plt.xlabel("Evaluation count")
    plt.ylabel("Energy (Hartree)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close()


def plot_ansatz_mode_final_energy(
    rows: list[dict],
    output_path: str | Path | None = None,
    title: str = "Final energy vs ansatz mode",
) -> None:
    labels = [row["ansatz_mode"] for row in rows]
    energies = [row["vqe_energy"] for row in rows]
    exact_energy = rows[0]["exact_energy"] if rows else None

    plt.figure(figsize=(8, 5))
    plt.bar(labels, energies)

    if exact_energy is not None:
        plt.axhline(exact_energy, linestyle="--", label="Exact energy")
        plt.legend()

    plt.xlabel("Ansatz mode")
    plt.ylabel("Energy (Hartree)")
    plt.title(title)
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close()


def plot_ansatz_mode_error(
    rows: list[dict],
    output_path: str | Path | None = None,
    title: str = "Absolute error vs ansatz mode",
) -> None:
    labels = [row["ansatz_mode"] for row in rows]
    errors = [row["absolute_error"] for row in rows]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, errors)
    plt.xlabel("Ansatz mode")
    plt.ylabel("Absolute error (Hartree)")
    plt.title(title)
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close()


def plot_optimizer_reps_overlay(
    histories: dict[str, dict[int, dict]],
    exact_energy: float,
    output_path: str | Path | None = None,
    title: str = "VQE convergence by optimizer and ansatz depth",
    use_running_best: bool = True,
) -> None:
    """
    histories format:
    {
        "COBYLA": {
            1: {"counts": [...], "energies": [...]},
            2: {...},
            3: {...},
        },
        "SPSA": {
            1: {...},
            2: {...},
            3: {...},
        },
    }
    """

    # blue gradient for COBYLA, red gradient for SPSA
    color_map = {
        "COBYLA": ["#9ecae1", "#3182bd", "#08519c"],
        "SPSA": ["#fcae91", "#fb6a4a", "#cb181d"],
    }

    plt.figure(figsize=(9, 6))

    for optimizer_name, reps_dict in histories.items():
        reps_sorted = sorted(reps_dict.keys())
        palette = color_map.get(optimizer_name, None)

        for idx, reps in enumerate(reps_sorted):
            data = reps_dict[reps]
            counts = np.array(data["counts"], dtype=float)
            energies = np.array(data["energies"], dtype=float)

            if use_running_best:
                y = np.minimum.accumulate(energies)
            else:
                y = energies

            if palette is not None and idx < len(palette):
                color = palette[idx]
            else:
                color = None

            plt.plot(
                counts,
                y,
                marker="o",
                markersize=2.5,
                linewidth=1.8,
                color=color,
                label=f"{optimizer_name} reps={reps}",
            )

    plt.axhline(exact_energy, linestyle="--", linewidth=2, label="Exact energy")

    plt.xlabel("Evaluation count")
    plt.ylabel("Energy (Hartree)")
    plt.title(title)
    plt.legend(ncol=2)
    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close()
