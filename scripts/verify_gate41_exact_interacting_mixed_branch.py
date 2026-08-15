"""Gate 41: exact interacting mixed Keldysh branch and source seed."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantum_transport import (  # noqa: E402
    finite_interacting_partition_free_two_time,
    hubbard_second_born_self_energy_mixed,
    kbe_initial_correlation_kernel,
    partition_free_finite_lead_two_time,
)


def _check(name: str, condition: bool, checks: dict[str, bool]) -> None:
    checks[name] = bool(condition)
    print(f"CHECK {name}: {'PASS' if condition else 'FAIL'}")


def run_gate() -> dict[str, object]:
    time = np.linspace(0.0, 0.5, 9)
    imaginary = np.linspace(0.0, 1.0 / 0.3, 41)
    initial_device = np.diag([-0.25, -0.18]).astype(complex)
    final_device = np.diag([0.08, -0.02]).astype(complex)
    leads = [np.diag([-0.8, -0.65]).astype(complex), np.diag([0.5, 0.62]).astype(complex)]
    couplings = [np.diag([0.25, 0.25]).astype(complex), np.diag([0.2, 0.2]).astype(complex)]
    finite = partition_free_finite_lead_two_time(
        time,
        imaginary,
        initial_device_hamiltonian=initial_device,
        final_device_hamiltonian=final_device,
        lead_hamiltonians=leads,
        coupling_matrices=couplings,
        lead_shifts=[0.15, -0.12],
        temperature=0.3,
    )
    common = {
        "time": time,
        "initial_one_body_hamiltonian": finite.initial_hamiltonian,
        "final_one_body_hamiltonian": finite.final_hamiltonian,
        "temperature": 0.3,
        "device_indices": [0, 1],
        "lead_indices": [[2, 3], [4, 5]],
        "spin_z": [1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
        "imaginary_time": imaginary,
    }
    exact_zero = finite_interacting_partition_free_two_time(interactions=[], **common)
    exact_u = finite_interacting_partition_free_two_time(interactions=[(0, 1, 0.5)], **common)
    sigma_zero = hubbard_second_born_self_energy_mixed(
        time,
        imaginary,
        green_rceil=exact_u.green_rceil,
        green_lceil=exact_u.green_mixed,
        interaction_u=0.0,
        spin_pairs=((0, 1), (1, 0)),
    )
    sigma_u = hubbard_second_born_self_energy_mixed(
        time,
        imaginary,
        green_rceil=exact_u.green_rceil,
        green_lceil=exact_u.green_mixed,
        interaction_u=0.5,
        spin_pairs=((0, 1), (1, 0)),
    )
    source_u = kbe_initial_correlation_kernel(
        time,
        imaginary,
        self_energy_mixed=sigma_u,
        green_mixed=exact_u.green_mixed,
    )
    sigma_reference = hubbard_second_born_self_energy_mixed(
        time,
        imaginary,
        green_rceil=exact_zero.green_rceil,
        green_lceil=exact_zero.green_mixed,
        interaction_u=0.5,
        spin_pairs=((0, 1), (1, 0)),
    )
    source_reference = kbe_initial_correlation_kernel(
        time,
        imaginary,
        self_energy_mixed=sigma_reference,
        green_mixed=exact_zero.green_mixed,
    )
    checks: dict[str, bool] = {}
    mixed_zero_error = float(np.max(np.abs(exact_zero.green_mixed - finite.green_mixed)))
    mixed_u_change = float(np.max(np.abs(exact_u.green_mixed - exact_zero.green_mixed)))
    source_u_max = float(np.max(np.abs(source_u.density_source)))
    source_change = float(np.max(np.abs(source_u.density_source - source_reference.density_source)))
    _check("exact_mixed_branch_is_exposed", exact_u.green_mixed is not None and exact_u.green_rceil is not None, checks)
    _check("u0_mixed_branch_matches_quadratic_reference", mixed_zero_error < 2e-12, checks)
    _check("finite_u_changes_mixed_branch", mixed_u_change > 1e-2, checks)
    _check("exact_mixed_branch_has_finite_u_source", source_u_max > 1e-4, checks)
    _check("mixed_source_is_hermitian", source_u.hermiticity_error < 1e-12, checks)
    _check("zero_u_second_born_mixed_kernel_is_zero", np.max(np.abs(sigma_zero)) < 1e-15, checks)
    _check("interacting_source_differs_from_quadratic_seed", source_change > 1e-4, checks)
    _check("exact_real_time_oracle_remains_spectral", exact_u.spectral_identity_error < 1e-12, checks)
    report = {
        "gate": "GATE_41_EXACT_INTERACTING_MIXED_KELDYSH_BRANCH",
        "checks": checks,
        "passed": all(checks.values()),
        "model": {"one_body_modes": 6, "device_spin_orbitals": 2, "interaction_u": 0.5, "temperature": 0.3, "time_points": 9, "imaginary_points": 41},
        "metrics": {
            "u0_mixed_branch_max_error": mixed_zero_error,
            "finite_u_mixed_branch_change_max": mixed_u_change,
            "exact_finite_u_second_born_source_max": source_u_max,
            "exact_finite_u_source_hermiticity_error": source_u.hermiticity_error,
            "exact_vs_quadratic_seed_source_change_max": source_change,
            "exact_real_time_spectral_identity_error": exact_u.spectral_identity_error,
        },
        "assessment": "PASS_EXACT_INTERACTING_MIXED_ORACLE_SOURCE_READY_CONTOUR_CLOSURE_OPEN",
        "claim_boundary": (
            "The exact finite Fock-space oracle now supplies interacting "
            "G^lceil/G^rceil, and the mixed second-Born source can be evaluated "
            "from it. The U=0 branch matches the finite-lead quadratic oracle. "
            "This still does not solve the interacting contour Dyson equation "
            "self-consistently or establish a continuum Corbino result."
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
