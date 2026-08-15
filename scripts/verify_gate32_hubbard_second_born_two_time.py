"""Gate 32: self-consistent Hubbard second-Born KBE closure.

This gate exercises the new symbolic/numeric two-time correlation self-energy
on a contacted two-spin-orbital device.  The finite-lead partition-free
kernel supplies the exact noninteracting initial branch; a finite-U many-body
oracle is used only as a reference to quantify the approximation error.  The
continuity residual is deliberately reported as an open diagnostic because
the present layer omits the instantaneous Hartree and full vertical-branch
interacting source.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantum_transport import (  # noqa: E402
    finite_interacting_partition_free_two_time,
    partition_free_finite_lead_two_time,
    self_consistent_hubbard_second_born_two_time,
    two_time_adjoint,
    two_time_kbe_continuity_balance,
)


def _model() -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    device_initial = np.diag([-0.25, -0.18]).astype(complex)
    device_final = np.diag([0.08, -0.02]).astype(complex)
    lead_a = np.diag([-0.8, -0.65]).astype(complex)
    lead_b = np.diag([0.5, 0.62]).astype(complex)
    coupling_a = np.diag([0.25, 0.25]).astype(complex)
    coupling_b = np.diag([0.2, 0.2]).astype(complex)
    return device_initial, device_final, [lead_a, lead_b], [coupling_a, coupling_b]


def _check(name: str, condition: bool, checks: dict[str, bool]) -> None:
    checks[name] = bool(condition)
    print(f"CHECK {name}: {'PASS' if condition else 'FAIL'}")


def run_gate() -> dict[str, object]:
    time = np.linspace(0.0, 0.5, 9)
    imaginary = np.linspace(0.0, 1.0 / 0.3, 41)
    initial_device, final_device, leads, couplings = _model()
    bare = partition_free_finite_lead_two_time(
        time,
        imaginary,
        initial_device_hamiltonian=initial_device,
        final_device_hamiltonian=final_device,
        lead_hamiltonians=leads,
        coupling_matrices=couplings,
        lead_shifts=[0.15, -0.12],
        chemical_potential=0.0,
        temperature=0.3,
    )
    scba = self_consistent_hubbard_second_born_two_time(
        time,
        bare_retarded=bare.retarded,
        bare_lesser=bare.lesser,
        interaction_u=0.5,
        spin_pairs=((0, 1), (1, 0)),
        max_iterations=50,
        dyson_iterations=70,
        mixing=0.2,
        tolerance=1e-7,
    )
    exact = finite_interacting_partition_free_two_time(
        time,
        initial_one_body_hamiltonian=bare.initial_hamiltonian,
        final_one_body_hamiltonian=bare.final_hamiltonian,
        interactions=[(0, 1, 0.5)],
        chemical_potential=0.0,
        temperature=0.3,
        device_indices=[0, 1],
        lead_indices=[[2, 3], [4, 5]],
        spin_z=[1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
    )

    checks: dict[str, bool] = {}
    spectral = scba.spectral_identity_error
    sigma_spectral = scba.self_energy_spectral_identity_error
    advanced_error = float(np.max(np.abs(scba.advanced - two_time_adjoint(scba.retarded))))
    causal_mask = np.triu(np.ones(scba.retarded.shape[:2], dtype=bool), k=1)
    causality_error = float(np.max(np.abs(scba.retarded[causal_mask])))
    density = scba.density_matrices()
    density_hermiticity = float(np.max(np.abs(density - density.swapaxes(-1, -2).conj())))
    density_eigenvalues = np.linalg.eigvalsh(density)
    density_gap = float(np.max(np.abs(density - exact.density_matrices)))
    sigma_strength = float(max(np.max(np.abs(scba.self_energy_lesser)), np.max(np.abs(scba.self_energy_greater))))

    balance = two_time_kbe_continuity_balance(
        time,
        green_retarded=scba.retarded,
        green_lesser=scba.lesser,
        hamiltonian=final_device,
        self_energy_retarded=scba.self_energy_retarded,
        self_energy_lesser=scba.self_energy_lesser,
        self_energy_advanced=scba.self_energy_advanced,
    )
    charge_residual = balance.observable_balance(np.eye(2, dtype=complex))["residual"]
    spin_residual = balance.observable_balance(np.diag([1.0, -1.0]).astype(complex))["residual"]

    _check("scba_converged", scba.converged, checks)
    _check("green_spectral_identity", spectral < 2e-14, checks)
    _check("self_energy_spectral_identity", sigma_spectral < 2e-10, checks)
    _check("retarded_advanced_adjoint", advanced_error < 2e-14, checks)
    _check("retarded_causality", causality_error < 2e-14, checks)
    _check("density_is_hermitian", density_hermiticity < 2e-10, checks)
    _check("density_is_physical_on_grid", np.min(density_eigenvalues) > -2e-8 and np.max(density_eigenvalues) < 1.0 + 2e-8, checks)
    _check("second_born_self_energy_is_nonzero", sigma_strength > 1e-6, checks)
    _check("exact_oracle_spectral_identity", exact.spectral_identity_error < 1e-12, checks)

    report = {
        "gate": "GATE_32_HUBBARD_SECOND_BORN_TWO_TIME_KBE",
        "checks": checks,
        "passed": all(checks.values()),
        "model": {
            "one_body_modes": 6,
            "device_spin_orbitals": 2,
            "lead_spin_orbitals": [2, 2],
            "interaction_u": 0.5,
            "temperature": 0.3,
            "time_points": int(time.size),
            "imaginary_points": int(imaginary.size),
            "spin_pairs": [[0, 1], [1, 0]],
        },
        "metrics": {
            "scba_iterations": int(scba.iterations),
            "scba_maximum_update": float(scba.maximum_update),
            "green_spectral_identity_error": float(spectral),
            "self_energy_spectral_identity_error": float(sigma_spectral),
            "advanced_adjoint_error": advanced_error,
            "retarded_causality_error": causality_error,
            "density_hermiticity_error": density_hermiticity,
            "density_eigenvalue_min": float(np.min(density_eigenvalues)),
            "density_eigenvalue_max": float(np.max(density_eigenvalues)),
            "second_born_self_energy_max": sigma_strength,
            "exact_oracle_spectral_identity_error": float(exact.spectral_identity_error),
            "scba_exact_density_max_gap": density_gap,
            "raw_kbe_continuity_residual_charge_max": float(np.max(np.abs(charge_residual))),
            "raw_kbe_continuity_residual_spin_max": float(np.max(np.abs(spin_residual))),
            "raw_kbe_continuity_residual_matrix_max": balance.maximum_residual,
        },
        "assessment": "PASS_SECOND_BORN_CONVERGES_EXACT_GAP_AND_OPEN_SOURCE_REPORTED",
        "claim_boundary": (
            "The correlation-only Hubbard second-Born self-energy is now a "
            "reproducible symbolic/numeric two-time KBE layer and converges on "
            "a contacted finite-lead benchmark. The exact finite-U density gap "
            "is reported as an approximation error. The raw continuity residual "
            "is not claimed to be a conservation failure: instantaneous Hartree "
            "and the full interacting vertical-branch initial source remain open."
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
