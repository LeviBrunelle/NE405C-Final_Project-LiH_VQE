"""
scan_bond_lengths.py
--------------------
Sweep over a range of Li–H bond lengths, compute the ground-state energy with
both VQE and exact diagonalisation at each geometry, and write results to CSV.
"""

import os

import numpy as np
import pandas as pd

from src.build_hamiltonian import build_hamiltonian
from src.run_exact import run_exact
from src.run_vqe import run_vqe


def scan_bond_lengths(
    bond_lengths=None,
    output_file: str = "results/energies.csv",
    run_vqe_flag: bool = True,
) -> pd.DataFrame:
    """Scan bond lengths and collect VQE and exact ground-state energies.

    Parameters
    ----------
    bond_lengths : array-like, optional
        Sequence of Li–H bond lengths in Angstroms to evaluate.
        Defaults to a grid from 1.0 Å to 3.8 Å in steps of 0.2 Å.
    output_file : str, optional
        Path to the CSV file where results are saved (default
        ``results/energies.csv``).
    run_vqe_flag : bool, optional
        When ``False``, skip the VQE calculation and record NaN for the VQE
        energy.  Useful for quickly generating the exact reference curve.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns ``bond_length``, ``vqe_energy``, and
        ``exact_energy``.
    """
    if bond_lengths is None:
        bond_lengths = np.arange(1.0, 4.0, 0.2)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    records = []
    for r in bond_lengths:
        print(f"Bond length {r:.2f} Å …", flush=True)
        qubit_op, problem, mapper = build_hamiltonian(r)

        exact_energy = run_exact(qubit_op, problem)
        print(f"  Exact  : {exact_energy:.6f} Ha", flush=True)

        if run_vqe_flag:
            vqe_energy, _ = run_vqe(qubit_op, problem, mapper)
            print(f"  VQE    : {vqe_energy:.6f} Ha", flush=True)
        else:
            vqe_energy = float("nan")

        records.append(
            {
                "bond_length": round(float(r), 4),
                "vqe_energy": vqe_energy,
                "exact_energy": exact_energy,
            }
        )

    df = pd.DataFrame(records, columns=["bond_length", "vqe_energy", "exact_energy"])
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}", flush=True)

    return df
