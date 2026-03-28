"""
main.py
-------
Entry point for the LiH VQE ground-state energy scan.

Usage
-----
    python main.py

The script:
  1. Scans Li–H bond lengths from 1.0 Å to 3.8 Å in 0.2 Å increments.
  2. Computes the ground-state energy at each geometry using both VQE and
     exact diagonalisation.
  3. Writes the results to ``results/energies.csv``.
  4. Produces the energy-vs-bond-length plot at ``results/energy_curve.png``.
"""

import os

from src.scan_bond_lengths import scan_bond_lengths
from src.plot_results import plot_results


def main() -> None:
    os.makedirs("results", exist_ok=True)

    # Run the bond-length scan (VQE + exact at each geometry)
    scan_bond_lengths()

    # Generate the energy-curve figure
    plot_results()


if __name__ == "__main__":
    main()
