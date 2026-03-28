"""
plot_results.py
---------------
Load the energies CSV produced by scan_bond_lengths and generate a publication-
quality plot comparing VQE and exact (FCI) ground-state energies as a function
of bond length.
"""

import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_results(
    input_file: str = "results/energies.csv",
    output_file: str = "results/energy_curve.png",
) -> None:
    """Plot VQE vs exact energy curves and save the figure.

    Parameters
    ----------
    input_file : str, optional
        Path to the CSV file containing ``bond_length``, ``vqe_energy``, and
        ``exact_energy`` columns (default ``results/energies.csv``).
    output_file : str, optional
        Path where the PNG figure is saved (default
        ``results/energy_curve.png``).
    """
    df = pd.read_csv(input_file)

    # Drop rows where the VQE energy was not computed
    df_exact = df.dropna(subset=["exact_energy"])
    df_vqe = df.dropna(subset=["vqe_energy"])

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(
        df_exact["bond_length"],
        df_exact["exact_energy"],
        "b-o",
        label="Exact (FCI)",
        linewidth=2,
        markersize=5,
    )

    if not df_vqe.empty:
        ax.plot(
            df_vqe["bond_length"],
            df_vqe["vqe_energy"],
            "r--s",
            label="VQE",
            linewidth=2,
            markersize=5,
        )

    ax.set_xlabel("Bond Length (Å)", fontsize=14)
    ax.set_ylabel("Energy (Hartree)", fontsize=14)
    ax.set_title("LiH Ground-State Energy vs Bond Length", fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {output_file}", flush=True)
