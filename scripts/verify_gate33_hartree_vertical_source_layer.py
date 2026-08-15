"""Gate 33: explicit Hartree collocation and interacting-source boundary.

The gate validates the instantaneous Hubbard Hartree layer against a static
one-body quench and then enables it in the contacted second-Born KBE run.  It
does not promote the raw continuity residual to a conservation result: the
interacting vertical-branch source and a production delta/Volterra treatment
remain separate publication gates.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantum_transport import (  # noqa: E402
    equilibrium_one_body_density,
    finite_interacting_partition_free_two_time,
    hubbard_hartree_self_energy_two_time,
    kadanoff_baym_dyson_two_time,
    partition_free_finite_lead_two_time,
    self_consistent_hubbard_second_born_two_time,
    two_time_greens,
    two_time_kbe_continuity_balance,
)


def _check(name: str, condition: bool, checks: dict[str, bool]) -> None:
    checks[name] = bool(condition)
    print(f"CHECK {name}: {'PASS' if condition else 'FAIL'}")


def _static_refinement() -> list[float]:
    hamiltonian = np.diag([0.2, -0.15]).astype(complex)
    density = equilibrium_one_body_density(hamiltonian, mu=0.0, temperature=0.2)
    errors: list[float] = []
    for points in (9, 17, 33):
        time = np.linspace(0.0, 1.2, points)
        bare = two_time_greens(time, lambda _: hamiltonian, density)
        density_stack = np.broadcast_to(density, (points, 2, 2)).copy()
        sigma_r, sigma_a, sigma_l, _ = hubbard_hartree_self_energy_two_time(
            time,
            density=density_stack,
            interaction_u=0.1,
            spin_pairs=((0, 1), (1, 0)),
        )
        result = kadanoff_baym_dyson_two_time(
            time,
            bare_retarded=bare.retarded,
            bare_lesser=bare.lesser,
            self_energy_retarded=sigma_r,
            self_energy_lesser=sigma_l,
            self_energy_advanced=sigma_a,
            max_iterations=160,
            mixing=0.5,
            tolerance=1e-10,
        )
        reference = two_time_greens(
            time,
            lambda _: hamiltonian + np.diag([0.1 * density[1, 1].real, 0.1 * density[0, 0].real]),
            density,
        )
        errors.append(float(np.max(np.abs(result.retarded - reference.retarded))))
    return errors


def run_gate() -> dict[str, object]:
    static_errors = _static_refinement()
    time = np.linspace(0.0, 0.5, 9)
    imaginary = np.linspace(0.0, 1.0 / 0.3, 41)
    initial_device = np.diag([-0.25, -0.18]).astype(complex)
    final_device = np.diag([0.08, -0.02]).astype(complex)
    leads = [np.diag([-0.8, -0.65]).astype(complex), np.diag([0.5, 0.62]).astype(complex)]
    couplings = [np.diag([0.25, 0.25]).astype(complex), np.diag([0.2, 0.2]).astype(complex)]
    bare = partition_free_finite_lead_two_time(
        time,
        imaginary,
        initial_device_hamiltonian=initial_device,
        final_device_hamiltonian=final_device,
        lead_hamiltonians=leads,
        coupling_matrices=couplings,
        lead_shifts=[0.15, -0.12],
        temperature=0.3,
    )
    result = self_consistent_hubbard_second_born_two_time(
        time,
        bare_retarded=bare.retarded,
        bare_lesser=bare.lesser,
        interaction_u=0.5,
        spin_pairs=((0, 1), (1, 0)),
        include_hartree=True,
        max_iterations=100,
        dyson_iterations=80,
        mixing=0.2,
        tolerance=1e-7,
    )
    exact = finite_interacting_partition_free_two_time(
        time,
        initial_one_body_hamiltonian=bare.initial_hamiltonian,
        final_one_body_hamiltonian=bare.final_hamiltonian,
        interactions=[(0, 1, 0.5)],
        temperature=0.3,
        device_indices=[0, 1],
        lead_indices=[[2, 3], [4, 5]],
        spin_z=[1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
    )
    balance = two_time_kbe_continuity_balance(
        time,
        green_retarded=result.retarded,
        green_lesser=result.lesser,
        hamiltonian=final_device,
        self_energy_retarded=result.self_energy_retarded,
        self_energy_lesser=result.self_energy_lesser,
        self_energy_advanced=result.self_energy_advanced,
    )
    density = result.density_matrices()
    density_eigenvalues = np.linalg.eigvalsh(density)
    charge_residual = balance.observable_balance(np.eye(2, dtype=complex))["residual"]
    spin_residual = balance.observable_balance(np.diag([1.0, -1.0]).astype(complex))["residual"]
    hartree_max = float(np.max(np.abs(result.hartree_retarded)))
    density_gap = float(np.max(np.abs(density - exact.density_matrices)))
    checks: dict[str, bool] = {}
    _check("static_hartree_refinement", static_errors[1] < 0.6 * static_errors[0] and static_errors[2] < 0.6 * static_errors[1], checks)
    _check("hartree_second_born_converged", result.converged, checks)
    _check("green_spectral_identity", result.spectral_identity_error < 2e-13, checks)
    _check("self_energy_spectral_identity", result.self_energy_spectral_identity_error < 2e-8, checks)
    _check("hartree_layer_is_nonzero", hartree_max > 1e-3, checks)
    _check("density_is_physical", np.min(density_eigenvalues) > -2e-8 and np.max(density_eigenvalues) < 1.0 + 2e-8, checks)
    _check("exact_oracle_spectral_identity", exact.spectral_identity_error < 1e-12, checks)
    report = {
        "gate": "GATE_33_HARTREE_COLLOCATION_VERTICAL_SOURCE_BOUNDARY",
        "checks": checks,
        "passed": all(checks.values()),
        "model": {"one_body_modes": 6, "device_spin_orbitals": 2, "interaction_u": 0.5, "temperature": 0.3, "time_points": 9, "imaginary_points": 41},
        "metrics": {
            "static_hartree_refinement_errors": static_errors,
            "iterations": int(result.iterations),
            "maximum_update": float(result.maximum_update),
            "green_spectral_identity_error": float(result.spectral_identity_error),
            "self_energy_spectral_identity_error": float(result.self_energy_spectral_identity_error),
            "hartree_retarded_max": hartree_max,
            "density_eigenvalue_min": float(np.min(density_eigenvalues)),
            "density_eigenvalue_max": float(np.max(density_eigenvalues)),
            "hartree_second_born_exact_density_max_gap": density_gap,
            "raw_kbe_continuity_residual_charge_max": float(np.max(np.abs(charge_residual))),
            "raw_kbe_continuity_residual_spin_max": float(np.max(np.abs(spin_residual))),
            "raw_kbe_continuity_residual_matrix_max": balance.maximum_residual,
        },
        "assessment": "PASS_HARTREE_LAYER_REFINED_SOURCE_CLOSURE_OPEN",
        "claim_boundary": (
            "The instantaneous Hartree contribution is explicit, symbolically "
            "defined, and converges under time refinement in a static control. "
            "It does not close the interacting contacted continuity equation: "
            "the vertical-branch source, endpoint delta treatment, and a "
            "production conserving contour solver remain open. The exact-U "
            "density gap is reported and is not a novelty claim."
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
