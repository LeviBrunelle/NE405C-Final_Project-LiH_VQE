from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


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
    plt.ylabel("Energy")
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
    plt.ylabel("Energy")
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
    plt.ylabel("Energy")
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
    plt.ylabel("Energy")
    plt.title(title)
    plt.legend()
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

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close()