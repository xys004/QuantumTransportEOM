"""Gate 65: mixed-grid and finite-lead-size convergence controls."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantum_transport import (  # noqa: E402
    hubbard_second_born_self_energy_mixed,
    kbe_initial_correlation_kernel,
    partition_free_finite_lead_two_time,
    self_consistent_hubbard_second_born_contour_two_time,
)


def _check(name: str, condition: bool, checks: dict[str, bool]) -> None:
    checks[name] = bool(condition)
    print(f"CHECK {name}: {'PASS' if condition else 'FAIL'}")


def _chain(size: int) -> np.ndarray:
    result = np.zeros((2 * size, 2 * size), dtype=complex)
    for site in range(size):
        onsite = 0.06 * np.diag([1.0, -1.0]).astype(complex)
        onsite += 0.035 * np.array([[0.0, 1.0 - 0.2j], [1.0 + 0.2j, 0.0]], dtype=complex)
        result[2 * site : 2 * site + 2, 2 * site : 2 * site + 2] = onsite
        if site + 1 < size:
            hopping = 0.22 * np.eye(2, dtype=complex)
            result[2 * site : 2 * site + 2, 2 * site + 2 : 2 * site + 4] = hopping
            result[2 * site + 2 : 2 * site + 4, 2 * site : 2 * site + 2] = hopping
    return result


def _mixed_grid_run(imaginary_size: int) -> dict[str, object]:
    time = np.linspace(0.0, 0.3, 7)
    imaginary = np.linspace(0.0, 1.0 / 0.3, imaginary_size)
    initial_device = np.diag([-0.25, -0.18]).astype(complex)
    final_device = np.diag([0.08, -0.02]).astype(complex)
    leads = [np.diag([-0.8, -0.65]).astype(complex), np.diag([0.5, 0.62]).astype(complex)]
    couplings = [np.diag([0.25, 0.25]).astype(complex), np.diag([0.2, 0.2]).astype(complex)]
    reference = partition_free_finite_lead_two_time(
        time,
        imaginary,
        initial_device_hamiltonian=initial_device,
        final_device_hamiltonian=final_device,
        lead_hamiltonians=leads,
        coupling_matrices=couplings,
        lead_shifts=[0.15, -0.12],
        temperature=0.3,
    )
    result = self_consistent_hubbard_second_born_contour_two_time(
        time,
        imaginary,
        bare_retarded=reference.retarded,
        bare_lesser=reference.lesser,
        bare_mixed=reference.green_mixed.swapaxes(0, 1).conj().swapaxes(-1, -2),
        green_matsubara=reference.green_matsubara,
        hamiltonian=final_device,
        interaction_u=0.3,
        embedding_self_energy_retarded=reference.self_energy_retarded,
        embedding_self_energy_mixed=reference.self_energy_mixed,
        spin_pairs=((0, 1), (1, 0)),
        max_iterations=60,
        dyson_iterations=60,
        mixing=0.3,
        tolerance=1e-7,
        include_hartree=True,
        include_full_contour_lesser=True,
        self_energy_matsubara=np.zeros_like(reference.green_matsubara),
    )
    correction = result.lesser_contour_correction
    return {
        "imaginary_size": imaginary_size,
        "result": result,
        "correction": correction.correction,
        "correction_max": float(np.max(np.abs(correction.correction))),
        "antihermiticity_error": correction.antihermiticity_error,
    }


def _lead_source(size: int) -> dict[str, object]:
    time = np.linspace(0.0, 0.3, 7)
    imaginary = np.linspace(0.0, 1.0 / 0.3, 21)
    initial_device = np.array([[0.14, 0.04 - 0.02j], [0.04 + 0.02j, -0.11]], dtype=complex)
    final_device = np.array([[0.06, 0.035 + 0.03j], [0.035 - 0.03j, -0.05]], dtype=complex)
    coupling = np.zeros((2, 2 * size), dtype=complex)
    coupling[:, :2] = np.array([[0.18, 0.025j], [0.012, 0.16]], dtype=complex)
    finite = partition_free_finite_lead_two_time(
        time,
        imaginary,
        initial_device_hamiltonian=initial_device,
        final_device_hamiltonian=final_device,
        lead_hamiltonians=[_chain(size), _chain(size)],
        coupling_matrices=[coupling, coupling.conj()],
        lead_shifts=[0.18, -0.16],
        temperature=1.0 / 3.0,
    )
    sigma = hubbard_second_born_self_energy_mixed(
        time,
        imaginary,
        green_rceil=finite.green_mixed.swapaxes(0, 1),
        green_lceil=finite.green_mixed,
        interaction_u=0.5,
        spin_pairs=((0, 1), (1, 0)),
    )
    source = kbe_initial_correlation_kernel(
        time,
        imaginary,
        self_energy_mixed=sigma,
        green_mixed=finite.green_mixed,
    )
    return {
        "size": size,
        "source": source.density_source,
        "source_max": float(np.max(np.abs(source.density_source))),
        "hermiticity_error": source.hermiticity_error,
        "spectral_identity_error": finite.spectral_identity_error,
    }


def run_gate() -> dict[str, object]:
    grid_runs = [_mixed_grid_run(size) for size in (11, 21, 41)]
    grid_differences = [
        float(np.max(np.abs(grid_runs[index]["correction"] - grid_runs[index + 1]["correction"])))
        for index in range(2)
    ]
    lead_runs = [_lead_source(size) for size in (2, 3, 4, 6)]
    reference = lead_runs[-1]["source"]
    lead_differences = {
        str(item["size"]): float(np.max(np.abs(item["source"] - reference)))
        for item in lead_runs[:-1]
    }
    checks: dict[str, bool] = {}
    _check("all_full_contour_grid_runs_converge", all(item["result"].converged for item in grid_runs), checks)
    _check("mixed_grid_refinement_reduces_difference", grid_differences[1] < grid_differences[0], checks)
    _check("mixed_correction_is_resolved", max(item["correction_max"] for item in grid_runs) > 1e-3, checks)
    _check("all_finite_lead_sources_are_hermitian", all(item["hermiticity_error"] < 1e-14 for item in lead_runs), checks)
    _check("all_finite_lead_oracles_are_spectral", all(item["spectral_identity_error"] < 1e-12 for item in lead_runs), checks)
    _check("size4_source_converges_to_size6", lead_differences["4"] < 1e-8, checks)
    _check("finite_lead_source_is_resolved", max(item["source_max"] for item in lead_runs) > 1e-4, checks)
    report = {
        "gate": "GATE_65_MIXED_GRID_LEAD_SIZE_CONVERGENCE",
        "checks": checks,
        "passed": all(checks.values()),
        "grid_runs": [
            {
                "imaginary_size": item["imaginary_size"],
                "correction_max": item["correction_max"],
                "antihermiticity_error": item["antihermiticity_error"],
                "iterations": item["result"].iterations,
            }
            for item in grid_runs
        ],
        "grid_differences": grid_differences,
        "lead_runs": [
            {
                "size": item["size"],
                "source_max": item["source_max"],
                "hermiticity_error": item["hermiticity_error"],
                "spectral_identity_error": item["spectral_identity_error"],
            }
            for item in lead_runs
        ],
        "lead_differences_vs_size6": lead_differences,
        "assessment": "PASS_MIXED_GRID_AND_FINITE_LEAD_SIZE_CONTROLS",
        "claim_boundary": (
            "The explicit full-contour correction refines on the imaginary grid and the seeded mixed source converges with finite lead size. "
            "These are numerical controls before a continuum extrapolation; they do not establish arbitrary-lead conservation or a topological invariant."
        ),
    }
    return report


def main() -> None:
    report = run_gate()
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"VERDICT: {'PASS' if report['passed'] else 'FAIL'}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
