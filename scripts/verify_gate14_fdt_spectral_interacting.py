"""Gate 14: interacting FDT, adjunction, and finite-window spectral controls."""

from __future__ import annotations

import json

import numpy as np

from quantum_transport import (
    equilibrium_one_body_density,
    self_consistent_born_electron_boson,
    self_consistent_born_two_time,
    two_time_greens,
)


def run_gate() -> dict:
    hamiltonian = np.array([[0.15]], dtype=complex)
    gammas = np.array([[[0.4]], [[0.35]]], dtype=complex)
    chemical_potentials = [0.0, 0.0]
    spectral_windows = []
    fdt_errors = []
    sum_errors = []
    for cutoff, points in ((6.0, 401), (10.0, 601), (20.0, 801)):
        energy = np.linspace(-cutoff, cutoff, points)
        stationary = self_consistent_born_electron_boson(
            hamiltonian,
            energy,
            gammas,
            chemical_potentials,
            coupling=np.array([[0.08]], dtype=complex),
            boson_frequency=0.7,
            temperature=0.12,
            max_iterations=50,
            mixing=0.4,
            tolerance=1e-9,
        )
        spectral_windows.append(stationary.spectral_sum_rule_error)
        fdt_errors.append(stationary.fdt_error())

    time = np.linspace(0.0, 1.2, 17)
    density = equilibrium_one_body_density(hamiltonian, mu=0.0, temperature=0.1)
    bare = two_time_greens(time, lambda _: hamiltonian, density)
    transient = self_consistent_born_two_time(
        time,
        bare_retarded=bare.retarded,
        bare_lesser=bare.lesser,
        coupling=np.array([[0.02]], dtype=complex),
        boson_frequency=0.4,
        boson_temperature=0.1,
        max_iterations=40,
        dyson_iterations=80,
        mixing=0.35,
        tolerance=1e-8,
    )
    checks = [
        {
            "name": "equilibrium_interacting_fdt",
            "passed": max(fdt_errors) < 3e-6,
            "details": {"fdt_error_by_cutoff": fdt_errors},
        },
        {
            "name": "finite_window_spectral_sum_refinement",
            "passed": spectral_windows[1] < spectral_windows[0] and spectral_windows[2] < spectral_windows[1],
            "details": {"spectral_sum_error_by_cutoff": spectral_windows},
        },
        {
            "name": "transient_adjoint_and_spectral_sum",
            "passed": transient.converged
            and transient.advanced_adjoint_error < 2e-14
            and transient.spectral_identity_error < 2e-14
            and transient.equal_time_spectral_sum_error < 2e-6,
            "details": {
                "advanced_adjoint_error": transient.advanced_adjoint_error,
                "spectral_identity_error": transient.spectral_identity_error,
                "equal_time_spectral_sum_error": transient.equal_time_spectral_sum_error,
            },
        },
        {
            "name": "transient_density_hermiticity_and_bounds",
            "passed": transient.density_hermiticity_error < 2e-9 and transient.occupation_bounds_violation < 2e-9,
            "details": {
                "density_hermiticity_error": transient.density_hermiticity_error,
                "occupation_bounds_violation": transient.occupation_bounds_violation,
            },
        },
    ]
    return {
        "gate": "GATE_14_FDT_SPECTRAL_INTERACTING",
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
        "claim_scope": "equilibrium interacting FDT controls, finite-window spectral sum convergence, and transient Keldysh/positivity diagnostics",
        "not_yet_claimed": [
            "exact continuum spectral normalization at finite cutoff",
            "interacting reservoir spin injection or Kane–Mele observables",
        ],
    }


def main() -> None:
    report = run_gate()
    print(json.dumps(report, indent=2, sort_keys=True))
    for check in report["checks"]:
        print(f"CHECK {check['name']}: {'PASS' if check['passed'] else 'FAIL'}")
    print(f"VERDICT: {'PASS' if report['passed'] else 'FAIL'}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

