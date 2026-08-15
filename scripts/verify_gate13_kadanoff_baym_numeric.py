"""Gate 13: numerical two-time Dyson/KBE and self-consistent Born control."""

from __future__ import annotations

import json

import numpy as np

from quantum_transport import (
    equilibrium_one_body_density,
    kadanoff_baym_dyson_two_time,
    self_consistent_born_two_time,
    two_time_convolution,
    two_time_greens,
)


def _trapz(values: np.ndarray, grid: np.ndarray) -> float:
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(trapezoid(values, grid))


def _single_level(time: np.ndarray):
    hamiltonian = np.array([[0.2]], dtype=complex)
    density = equilibrium_one_body_density(hamiltonian, mu=0.0, temperature=0.1)
    return two_time_greens(time, lambda _: hamiltonian, density)


def run_gate() -> dict:
    time = np.array([0.0, 0.2, 0.7, 1.4])
    left = np.empty((time.size, time.size, 1, 1), dtype=complex)
    right = np.empty_like(left)
    for i, t in enumerate(time):
        for k, tau in enumerate(time):
            for j, tp in enumerate(time):
                left[i, k, 0, 0] = t + tau
                right[k, j, 0, 0] = tau + tp
    convolution = two_time_convolution(left, right, time)
    expected = np.empty_like(convolution)
    for i, t in enumerate(time):
        for j, tp in enumerate(time):
            expected[i, j, 0, 0] = _trapz((t + time) * (time + tp), time)
    convolution_error = float(np.max(np.abs(convolution - expected)))

    base_time = np.linspace(0.0, 0.8, 7)
    bare = _single_level(base_time)
    zeros = np.zeros_like(bare.retarded)
    zero_sigma = kadanoff_baym_dyson_two_time(
        base_time,
        bare_retarded=bare.retarded,
        bare_lesser=bare.lesser,
        self_energy_retarded=zeros,
        self_energy_lesser=zeros,
    )
    zero_sigma_error = float(
        max(
            np.max(np.abs(zero_sigma.retarded - bare.retarded)),
            np.max(np.abs(zero_sigma.lesser - bare.lesser)),
        )
    )

    drifts = []
    identities = []
    iterations = []
    for points in (9, 17, 33):
        grid = np.linspace(0.0, 1.2, points)
        bare = _single_level(grid)
        result = self_consistent_born_two_time(
            grid,
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
        drifts.append(result.particle_number_drift())
        identities.append(
            max(
                result.spectral_identity_error,
                result.advanced_adjoint_error,
                result.lesser_adjoint_error,
                result.retarded_causality_error,
            )
        )
        iterations.append(result.iterations)

    checks = [
        {
            "name": "nonuniform_two_time_convolution",
            "passed": convolution_error < 2e-14,
            "details": {"maximum_error": convolution_error},
        },
        {
            "name": "zero_self_energy_dyson_control",
            "passed": zero_sigma.converged and zero_sigma_error < 2e-14,
            "details": {"maximum_error": zero_sigma_error},
        },
        {
            "name": "scba_keldysh_identity_and_adjoint",
            "passed": max(identities) < 2e-9,
            "details": {"maximum_residuals": identities, "iterations": iterations},
        },
        {
            "name": "scba_particle_number_refinement",
            "passed": drifts[1] < 0.3 * drifts[0] and drifts[2] < 0.3 * drifts[1],
            "details": {"drift_by_grid": drifts},
        },
    ]
    return {
        "gate": "GATE_13_KADANOFF_BAYM_NUMERIC_SCBA",
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
        "claim_scope": "finite-grid numerical two-time Dyson/KBE and self-consistent electron–boson Fock/SCBA control",
        "not_yet_claimed": [
            "continuum-limit convergence for arbitrary reservoir memory",
            "interacting Kane–Mele/Corbino transport currents",
            "a production-scale contour solver beyond the finite-grid fixed point",
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
