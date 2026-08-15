"""Gate 43: exact-interacting source error budget for charge and spin."""

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
    two_time_kbe_continuity_balance,
)


def _check(name: str, condition: bool, checks: dict[str, bool]) -> None:
    checks[name] = bool(condition)
    print(f"CHECK {name}: {'PASS' if condition else 'FAIL'}")


def run_gate() -> dict[str, object]:
    time = np.linspace(0.0, 0.5, 9)
    imaginary = np.linspace(0.0, 1.0 / 0.3, 41)
    final_device = np.diag([0.08, -0.02]).astype(complex)
    finite = partition_free_finite_lead_two_time(
        time,
        imaginary,
        initial_device_hamiltonian=np.diag([-0.25, -0.18]).astype(complex),
        final_device_hamiltonian=final_device,
        lead_hamiltonians=[np.diag([-0.8, -0.65]).astype(complex), np.diag([0.5, 0.62]).astype(complex)],
        coupling_matrices=[np.diag([0.25, 0.25]).astype(complex), np.diag([0.2, 0.2]).astype(complex)],
        lead_shifts=[0.15, -0.12],
        temperature=0.3,
    )
    exact = finite_interacting_partition_free_two_time(
        time,
        initial_one_body_hamiltonian=finite.initial_hamiltonian,
        final_one_body_hamiltonian=finite.final_hamiltonian,
        interactions=[(0, 1, 0.5)],
        temperature=0.3,
        device_indices=[0, 1],
        lead_indices=[[2, 3], [4, 5]],
        spin_z=[1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
        imaginary_time=imaginary,
    )
    balance = two_time_kbe_continuity_balance(
        time,
        green_retarded=exact.retarded,
        green_lesser=exact.lesser,
        hamiltonian=final_device,
        self_energy_retarded=finite.self_energy_retarded,
        self_energy_lesser=finite.self_energy_lesser,
        self_energy_advanced=finite.self_energy_advanced,
    )
    sigma_mixed = hubbard_second_born_self_energy_mixed(
        time,
        imaginary,
        green_rceil=exact.green_rceil,
        green_lceil=exact.green_mixed,
        interaction_u=0.5,
        spin_pairs=((0, 1), (1, 0)),
    )
    source = kbe_initial_correlation_kernel(time, imaginary, self_energy_mixed=sigma_mixed, green_mixed=exact.green_mixed)
    charge = np.eye(2, dtype=complex)
    spin = np.diag([1.0, -1.0]).astype(complex)
    project = lambda op, value: np.real(np.trace(op @ value, axis1=-2, axis2=-1))
    required_charge = project(charge, balance.residual)
    required_spin = project(spin, balance.residual)
    secondborn_charge = project(charge, source.density_source)
    secondborn_spin = project(spin, source.density_source)
    corrected = balance.residual - source.density_source
    charge_current_error = exact.device_rate() - exact.lead_current(0) - exact.lead_current(1)
    spin_operator = np.diag([1.0, -1.0, 0.0, 0.0, 0.0, 0.0]).astype(complex)
    spin_current_error = exact.device_rate(observable=spin_operator) - exact.lead_current(0, spin=True) - exact.lead_current(1, spin=True)
    checks: dict[str, bool] = {}
    _check("exact_charge_continuity_oracle", np.max(np.abs(charge_current_error)) < 2e-11, checks)
    _check("exact_spin_continuity_oracle", np.max(np.abs(spin_current_error)) < 2e-11, checks)
    _check("required_source_is_hermitian", np.max(np.abs(balance.residual - balance.residual.swapaxes(-1, -2).conj())) < 2e-10, checks)
    _check("second_born_source_is_hermitian", source.hermiticity_error < 1e-12, checks)
    _check("charge_source_error_is_resolved", np.max(np.abs(required_charge - secondborn_charge)) > 1e-2, checks)
    _check("spin_source_error_is_resolved", np.max(np.abs(required_spin - secondborn_spin)) > 1e-3, checks)
    _check("second_born_does_not_close_exact_matrix_balance", np.max(np.abs(corrected)) > 1.2 * balance.maximum_residual, checks)
    report = {
        "gate": "GATE_43_EXACT_INTERACTING_SOURCE_ERROR_BUDGET",
        "checks": checks,
        "passed": all(checks.values()),
        "metrics": {
            "exact_charge_continuity_error": float(np.max(np.abs(charge_current_error))),
            "exact_spin_continuity_error": float(np.max(np.abs(spin_current_error))),
            "required_source_charge_max": float(np.max(np.abs(required_charge))),
            "second_born_source_charge_max": float(np.max(np.abs(secondborn_charge))),
            "charge_source_error_max": float(np.max(np.abs(required_charge - secondborn_charge))),
            "required_source_spin_max": float(np.max(np.abs(required_spin))),
            "second_born_source_spin_max": float(np.max(np.abs(secondborn_spin))),
            "spin_source_error_max": float(np.max(np.abs(required_spin - secondborn_spin))),
            "required_source_matrix_max": float(np.max(np.abs(balance.residual))),
            "second_born_source_matrix_max": float(np.max(np.abs(source.density_source))),
            "second_born_corrected_matrix_residual_max": float(np.max(np.abs(corrected))),
        },
        "assessment": "PASS_EXACT_SOURCE_ERROR_BUDGET_SECOND_BORN_CLOSURE_INCOMPLETE",
        "claim_boundary": (
            "The exact finite-U oracle conserves charge and spin while its "
            "required source differs measurably from the second-Born mixed "
            "source in both projections. This quantifies a missing interacting "
            "closure contribution; it is not a topological or novelty claim."
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
