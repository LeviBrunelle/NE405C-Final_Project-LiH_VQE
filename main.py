from __future__ import annotations

import csv
import json
import statistics
from datetime import datetime
from pathlib import Path

from src import config as C
from src.build_hamiltonian import get_problem_bundle, preview_fermionic_terms, preview_qubit_terms, summarize_problem
from src.plot_results import plot_bars, plot_mean_std, plot_noise_panels, plot_overlay, plot_series
from src.run_exact import run_exact
from src.run_vqe import build_ansatz, repeat, run_vqe, sweep
from src.scan_bond_lengths import scan_bond_lengths


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_run_dir() -> Path:
    root = C.ensure_results_dir()
    run_dir = root / f"{datetime.now().strftime('%Y-%m-%d_%H%M%S')}_{C.RUN_NAME}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def dump_config(run_dir: Path) -> None:
    snapshot = {name: getattr(C, name) for name in dir(C) if name.isupper() and isinstance(getattr(C, name), (str, int, float, bool, list, tuple, type(None)))}
    write_json(run_dir / "config_snapshot.json", snapshot)


def summarize_trials(trials: list[dict], exact_energy: float) -> dict:
    finals = [t["energy"] for t in trials]
    errors = [abs(t["energy"] - exact_energy) for t in trials]
    return {
        "num_trials": len(trials),
        "mean_final_energy": statistics.mean(finals),
        "std_final_energy": statistics.pstdev(finals) if len(finals) > 1 else 0.0,
        "mean_absolute_error": statistics.mean(errors),
        "std_absolute_error": statistics.pstdev(errors) if len(errors) > 1 else 0.0,
        "trial_final_energies": finals,
        "trial_absolute_errors": errors,
        "trial_seeds": [t.get("seed") for t in trials],
    }


def common_kwargs(problem, mapper):
    return {
        "ansatz_kind": C.ANSATZ_KIND,
        "optimizer_name": C.OPTIMIZER,
        "reps": C.REPS,
        "backend_mode": C.BACKEND_MODE,
        "problem": problem,
        "mapper": mapper,
        "show_progress": C.USE_TQDM,
    }


def save_ansatz_figure(run_dir: Path, qubit_hamiltonian, problem, mapper, molecule_name: str):
    if not C.SAVE_ANSATZ_FIGURE:
        return
    ansatz, _ = build_ansatz(qubit_hamiltonian.num_qubits, ansatz_kind=C.ANSATZ_KIND, reps=C.REPS, problem=problem, mapper=mapper)
    try:
        circuit = ansatz if C.ANSATZ_KIND == "uccsd_hf" else ansatz.decompose()
        fig = circuit.draw(output="mpl", fold=C.ANSATZ_FOLD, idle_wires=False)
        fig.savefig(run_dir / f"{molecule_name}_{C.ANSATZ_KIND}_ansatz.png", dpi=C.FIG_DPI, bbox_inches="tight")
    except Exception as exc:
        print(f"Warning: could not save ansatz figure: {exc}")


def main() -> None:
    run_dir = make_run_dir()
    dump_config(run_dir)
    print(f"Writing results to: {run_dir}")

    bundle = get_problem_bundle(C.DEFAULT_BOND_LENGTH, basis=C.BASIS, simplify=True)
    problem = bundle["problem"]
    mapper = bundle["mapper"]
    fermionic_h = bundle["fermionic_hamiltonian"]
    qubit_h = bundle["qubit_hamiltonian"]
    summary = summarize_problem(problem, qubit_h, C.DEFAULT_BOND_LENGTH, C.BASIS)
    molecule_name = summary["molecule_name"]
    exact = run_exact(qubit_h)

    if C.SAVE_SUMMARIES:
        write_json(run_dir / "fermionic_terms_preview.json", preview_fermionic_terms(fermionic_h, C.PREVIEW_TERM_COUNT))
        write_json(run_dir / "qubit_terms_preview.json", preview_qubit_terms(qubit_h, C.PREVIEW_TERM_COUNT))

    kwargs = common_kwargs(problem, mapper)

    if C.EXPERIMENT == "single":
        result = run_vqe(qubit_h, seed=C.RANDOM_SEED, label=f"{C.BACKEND_MODE}:{C.OPTIMIZER}", **kwargs)
        payload = {**summary, **result, "exact_energy": exact["energy"], "absolute_error": abs(result["energy"] - exact["energy"])}
        write_json(run_dir / "single_summary.json", payload)
        plot_overlay({f"{C.ANSATZ_KIND}/{C.OPTIMIZER}": result}, exact["energy"], run_dir / "convergence.png", title=f"{molecule_name} convergence")
        save_ansatz_figure(run_dir, qubit_h, problem, mapper, molecule_name)
        return

    if C.EXPERIMENT == "reps":
        results = sweep(C.REPS_VALUES, lambda reps: run_vqe(qubit_h, reps=reps, seed=C.RANDOM_SEED, problem=problem, mapper=mapper, ansatz_kind=C.ANSATZ_KIND, optimizer_name=C.OPTIMIZER, backend_mode=C.BACKEND_MODE, show_progress=C.USE_TQDM, label=f"reps={reps}"))
        summary_rows = {reps: {"energy": r["energy"], "absolute_error": abs(r["energy"] - exact["energy"]), "ansatz_num_parameters": r["ansatz_num_parameters"], "ansatz_depth": r["ansatz_depth"]} for reps, r in results.items()}
        write_json(run_dir / "reps_summary.json", summary_rows)
        plot_overlay({f"reps={k}": v for k, v in results.items()}, exact["energy"], run_dir / "reps_overlay.png", title=f"{molecule_name} ansatz depth")
        plot_series(sorted(summary_rows), [summary_rows[k]["absolute_error"] for k in sorted(summary_rows)], run_dir / "reps_error.png", title="Absolute error vs reps", xlabel="reps", ylabel="Absolute error (Hartree)")
        return

    if C.EXPERIMENT == "optimizer":
        results = sweep(
            C.OPTIMIZERS,
            lambda name: run_vqe(
                qubit_h,
                optimizer_name=name,
                seed=C.RANDOM_SEED,
                problem=problem,
                mapper=mapper,
                ansatz_kind=C.ANSATZ_KIND,
                reps=C.REPS,
                backend_mode=C.BACKEND_MODE,
                show_progress=C.USE_TQDM,
                label=name,
            ),
        )
        rows = {name: {"energy": r["energy"], "absolute_error": abs(r["energy"] - exact["energy"]), "ansatz_num_parameters": r["ansatz_num_parameters"], "ansatz_depth": r["ansatz_depth"]} for name, r in results.items()}
        write_json(run_dir / "optimizer_summary.json", rows)
        plot_overlay(results, exact["energy"], run_dir / "optimizer_overlay.png", title=f"{molecule_name} optimizer comparison")
        return

    if C.EXPERIMENT == "ansatz":
        results = sweep(C.ANSATZ_KINDS, lambda kind: run_vqe(qubit_h, ansatz_kind=kind, seed=C.RANDOM_SEED, problem=problem, mapper=mapper, optimizer_name=C.OPTIMIZER, reps=C.REPS, backend_mode=C.BACKEND_MODE, show_progress=C.USE_TQDM, label=kind))
        rows = []
        for kind, r in results.items():
            rows.append({**summary, "ansatz_kind": kind, "exact_energy": exact["energy"], "vqe_energy": r["energy"], "absolute_error": abs(r["energy"] - exact["energy"]), "ansatz_num_parameters": r["ansatz_num_parameters"], "ansatz_depth": r["ansatz_depth"]})
        write_csv(run_dir / "ansatz_summary.csv", rows)
        plot_overlay(results, exact["energy"], run_dir / "ansatz_overlay.png", title=f"{molecule_name} ansatz comparison")
        plot_bars([r["ansatz_kind"] for r in rows], [r["absolute_error"] for r in rows], run_dir / "ansatz_error.png", title="Absolute error vs ansatz", ylabel="Absolute error (Hartree)")
        plot_bars([r["ansatz_kind"] for r in rows], [r["vqe_energy"] for r in rows], run_dir / "ansatz_energy.png", title="Final energy vs ansatz", ylabel="Energy (Hartree)", exact_energy=exact["energy"])
        return

    if C.EXPERIMENT == "repeat_single":
        trials = repeat(C.NUM_TRIALS, lambda seed: run_vqe(qubit_h, seed=seed, **kwargs, label=f"trial"), base_seed=C.RANDOM_SEED)
        write_json(run_dir / "repeat_single_summary.json", {**summary, **summarize_trials(trials, exact["energy"])})
        plot_mean_std({f"{C.ANSATZ_KIND}/{C.OPTIMIZER}": trials}, exact["energy"], run_dir / "repeat_single.png", title="Repeated single-point trials")
        return

    if C.EXPERIMENT == "repeat_optimizer":
        groups = {
            name: repeat(
                C.NUM_TRIALS,
                lambda seed, name=name: run_vqe(
                    qubit_h,
                    optimizer_name=name,
                    seed=seed,
                    problem=problem,
                    mapper=mapper,
                    ansatz_kind=C.ANSATZ_KIND,
                    reps=C.REPS,
                    backend_mode=C.BACKEND_MODE,
                    show_progress=C.USE_TQDM,
                    label=name,
                ),
                base_seed=C.RANDOM_SEED,
            )
            for name in C.OPTIMIZERS
        }
        write_json(run_dir / "repeat_optimizer_summary.json", {name: summarize_trials(trials, exact["energy"]) for name, trials in groups.items()})
        plot_mean_std(groups, exact["energy"], run_dir / "repeat_optimizer.png", title=f"Repeated optimizer comparison ({C.ANSATZ_KIND})")
        return

    if C.EXPERIMENT == "repeat_ansatz":
        groups = {kind: repeat(C.NUM_TRIALS, lambda seed, kind=kind: run_vqe(qubit_h, ansatz_kind=kind, seed=seed, problem=problem, mapper=mapper, optimizer_name=C.OPTIMIZER, reps=C.REPS, backend_mode=C.BACKEND_MODE, show_progress=C.USE_TQDM, label=kind), base_seed=C.RANDOM_SEED) for kind in C.ANSATZ_KINDS}
        write_json(run_dir / "repeat_ansatz_summary.json", {kind: summarize_trials(trials, exact["energy"]) for kind, trials in groups.items()})
        plot_mean_std(groups, exact["energy"], run_dir / "repeat_ansatz.png", title=f"Repeated ansatz comparison ({C.OPTIMIZER})")
        return

    if C.EXPERIMENT == "bond_scan":
        rows = scan_bond_lengths(C.BOND_LENGTHS, ansatz_kind=C.ANSATZ_KIND, optimizer_name=C.OPTIMIZER, reps=C.REPS, backend_mode=C.BACKEND_MODE, basis=C.BASIS, show_progress=C.USE_TQDM)
        write_csv(run_dir / "bond_scan.csv", rows)
        plot_series([r["bond_length_angstrom"] for r in rows], [r["exact_energy"] for r in rows], None, title=f"{molecule_name} exact energy vs bond length", xlabel="Bond length (Angstrom)", ylabel="Energy (Hartree)")
        # combined exact/vqe plot
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        x = [r["bond_length_angstrom"] for r in rows]
        plt.plot(x, [r["exact_energy"] for r in rows], marker="o", label="Exact")
        plt.plot(x, [r["vqe_energy"] for r in rows], marker="s", label="VQE")
        plt.xlabel("Bond length (Angstrom)")
        plt.ylabel("Energy (Hartree)")
        plt.title(f"{molecule_name} energy vs bond length")
        plt.grid(); plt.legend(); plt.tight_layout(); plt.savefig(run_dir / "bond_scan.png", dpi=300, bbox_inches="tight"); plt.close()
        plot_series(x, [r["absolute_error"] for r in rows], run_dir / "bond_error.png", title=f"{molecule_name} absolute error vs bond length", xlabel="Bond length (Angstrom)", ylabel="Absolute error (Hartree)")
        return

    if C.EXPERIMENT == "basis":
        rows = []
        for basis in C.BASIS_SETS:
            bundle = get_problem_bundle(C.DEFAULT_BOND_LENGTH, basis=basis, simplify=True)
            p, m, qh = bundle["problem"], bundle["mapper"], bundle["qubit_hamiltonian"]
            exact_energy = None
            vqe_energy = None
            abs_error = None
            params = None
            depth = None
            if qh.num_qubits <= C.MAX_EXACT_QUBITS:
                exact_energy = run_exact(qh)["energy"]
                vqe = run_vqe(qh, ansatz_kind=C.ANSATZ_KIND, optimizer_name=C.OPTIMIZER, reps=C.REPS, backend_mode=C.BACKEND_MODE, problem=p, mapper=m, seed=C.RANDOM_SEED, show_progress=C.USE_TQDM, label=basis)
                vqe_energy = vqe["energy"]
                abs_error = abs(vqe_energy - exact_energy)
                params, depth = vqe["ansatz_num_parameters"], vqe["ansatz_depth"]
            rows.append({**summarize_problem(p, qh, C.DEFAULT_BOND_LENGTH, basis), "exact_energy": exact_energy, "vqe_energy": vqe_energy, "absolute_error": abs_error, "ansatz_num_parameters": params, "ansatz_depth": depth})
        write_csv(run_dir / "basis_summary.csv", rows)
        valid = [r for r in rows if r["absolute_error"] is not None]
        if valid:
            plot_bars([r["basis"] for r in valid], [r["absolute_error"] for r in valid], run_dir / "basis_error.png", title="Absolute error vs basis", ylabel="Absolute error (Hartree)")
        return

    if C.EXPERIMENT == "optimizer_reps":
        grid = {opt: sweep(C.REPS_VALUES, lambda reps, opt=opt: run_vqe(qubit_h, optimizer_name=opt, reps=reps, seed=C.RANDOM_SEED, **{k:v for k,v in kwargs.items() if k not in {'optimizer_name','reps'}}, label=f"{opt} reps={reps}")) for opt in C.OPTIMIZERS}
        write_json(run_dir / "optimizer_reps_summary.json", {opt: {reps: {"energy": r["energy"], "absolute_error": abs(r["energy"] - exact["energy"])} for reps, r in d.items()} for opt, d in grid.items()})
        flat = {f"{opt} reps={reps}": r for opt, d in grid.items() for reps, r in d.items()}
        plot_overlay(flat, exact["energy"], run_dir / "optimizer_reps.png", title=f"{molecule_name} optimizer × reps")
        return

    if C.EXPERIMENT == "noise":
        results_by_mode = {mode: [run_vqe(qubit_h, backend_mode=mode, optimizer_name=opt, ansatz_kind=C.ANSATZ_KIND, reps=C.REPS, problem=problem, mapper=mapper, seed=C.RANDOM_SEED, show_progress=C.USE_TQDM, label=f"{mode}:{opt}") for opt in C.OPTIMIZERS] for mode in C.BACKEND_MODES}
        write_json(run_dir / "noise_summary.json", {mode: {r["optimizer_name"]: {"energy": r["energy"], "absolute_error": abs(r["energy"] - exact["energy"])} for r in results} for mode, results in results_by_mode.items()})
        plot_noise_panels(results_by_mode, exact["energy"], run_dir / "noise_compare.png", title=f"{molecule_name}: exact vs shots vs noise")
        return

    raise ValueError(f"Unsupported EXPERIMENT: {C.EXPERIMENT}")


if __name__ == "__main__":
    main()