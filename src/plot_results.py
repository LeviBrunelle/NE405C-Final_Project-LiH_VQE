from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _save(output_path: str | Path | None):
    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def running_best(values):
    return np.minimum.accumulate(np.array(values, dtype=float))


def plot_overlay(histories: dict, exact_energy: float, output_path: str | Path | None = None, *, title: str, running_min: bool = True, xlabel: str = "Evaluation count", ylabel: str = "Energy (Hartree)"):
    plt.figure(figsize=(8, 5))
    for label, data in histories.items():
        x = np.array(data["counts"], dtype=float)
        y = np.array(data["energies"], dtype=float)
        if running_min:
            y = running_best(y)
        plt.plot(x, y, marker="o", markersize=3, label=str(label))
    plt.axhline(exact_energy, linestyle="--", label="Exact energy")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.tight_layout()
    _save(output_path)


def plot_series(x, y, output_path: str | Path | None = None, *, title: str, xlabel: str, ylabel: str):
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.tight_layout()
    _save(output_path)


def plot_bars(labels, values, output_path: str | Path | None = None, *, title: str, ylabel: str, exact_energy: float | None = None):
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    if exact_energy is not None:
        plt.axhline(exact_energy, linestyle="--", label="Exact energy")
        plt.legend()
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    _save(output_path)


def plot_mean_std(groups: dict[str, list[dict]], exact_energy: float, output_path: str | Path | None = None, *, title: str):
    plt.figure(figsize=(8, 5))
    for label, trials in groups.items():
        if not trials:
            continue
        n = min(len(t["energies"]) for t in trials)
        if n == 0:
            continue
        matrix = np.array([t["energies"][:n] for t in trials], dtype=float)
        x = np.arange(1, n + 1)
        mean = matrix.mean(axis=0)
        std = matrix.std(axis=0)
        final_std = np.std([t["energy"] for t in trials])
        plt.plot(x, mean, label=f"{label} (final std={final_std:.4f} Ha)")
        plt.fill_between(x, mean - std, mean + std, alpha=0.15)
    plt.axhline(exact_energy, linestyle="--", label="Exact energy")
    plt.xlabel("Evaluation index")
    plt.ylabel("Energy (Hartree)")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    _save(output_path)


def plot_noise_panels(results_by_mode: dict[str, list[dict]], exact_energy: float, output_path: str | Path | None = None, *, title: str):
    fig, axes = plt.subplots(1, len(results_by_mode), figsize=(6 * len(results_by_mode), 5), sharey=True)
    if len(results_by_mode) == 1:
        axes = [axes]
    for ax, (mode, results) in zip(axes, results_by_mode.items()):
        for result in results:
            x = np.array(result["counts"], dtype=float)
            y = running_best(result["energies"])
            err = abs(result["energy"] - exact_energy)
            ax.plot(x, y, linewidth=2, label=f"{result['optimizer_name']} (err={err:.3f} Ha)")
        ax.axhline(exact_energy, linestyle="--", linewidth=2, label="Exact")
        ax.set_title(mode)
        ax.set_xlabel("Evaluation count")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Energy (Hartree)")
    axes[-1].legend(fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
