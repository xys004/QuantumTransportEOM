"""Gate 34: microscopic vertical-source attachment to continuity diagnostics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantum_transport import (  # noqa: E402
    partition_free_finite_lead_two_time,
    two_time_kbe_continuity_balance,
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
    raw = two_time_kbe_continuity_balance(
        time,
        green_retarded=finite.retarded,
        green_lesser=finite.lesser,
        hamiltonian=final_device,
        self_energy_retarded=finite.self_energy_retarded,
        self_energy_lesser=finite.self_energy_lesser,
        self_energy_advanced=finite.self_energy_advanced,
    )
    attached = two_time_kbe_continuity_balance(
        time,
        green_retarded=finite.retarded,
        green_lesser=finite.lesser,
        hamiltonian=final_device,
        self_energy_retarded=finite.self_energy_retarded,
        self_energy_lesser=finite.self_energy_lesser,
        self_energy_advanced=finite.self_energy_advanced,
        initial_correlation_source=finite.continuity_initial_source,
    )
    project = lambda operator, values: np.real(np.trace(operator @ values, axis1=-2, axis2=-1))
    charge_raw = project(np.eye(2, dtype=complex), raw.residual)
    charge_corrected = project(np.eye(2, dtype=complex), attached.source_corrected_residual)
    spin = np.diag([1.0, -1.0]).astype(complex)
    spin_raw = project(spin, raw.residual)
    spin_corrected = project(spin, attached.source_corrected_residual)
    source = finite.continuity_initial_source
    checks: dict[str, bool] = {}
    _check("microscopic_source_is_hermitian", finite.initial_correlation.hermiticity_error < 1e-14, checks)
    _check("raw_residual_is_preserved", np.max(np.abs(attached.residual - raw.residual)) < 1e-14, checks)
    _check("vertical_source_reduces_matrix_residual", attached.maximum_source_corrected_residual < 0.2 * raw.maximum_residual, checks)
    _check("vertical_source_reduces_charge_residual", np.max(np.abs(charge_corrected)) < 0.2 * np.max(np.abs(charge_raw)), checks)
    _check("vertical_source_reduces_spin_residual", np.max(np.abs(spin_corrected)) < 0.2 * np.max(np.abs(spin_raw)), checks)
    _check("corrected_residual_is_finite", np.all(np.isfinite(attached.source_corrected_residual)), checks)
    report = {
        "gate": "GATE_34_VERTICAL_SOURCE_CONTINUITY_ATTACHMENT",
        "checks": checks,
        "passed": all(checks.values()),
        "model": {"device_spin_orbitals": 2, "lead_spin_orbitals": [2, 2], "temperature": 0.3, "time_points": 9, "imaginary_points": 41},
        "metrics": {
            "source_hermiticity_error": finite.initial_correlation.hermiticity_error,
            "raw_matrix_residual_max": raw.maximum_residual,
            "source_corrected_matrix_residual_max": attached.maximum_source_corrected_residual,
            "raw_charge_residual_max": float(np.max(np.abs(charge_raw))),
            "source_corrected_charge_residual_max": float(np.max(np.abs(charge_corrected))),
            "raw_spin_residual_max": float(np.max(np.abs(spin_raw))),
            "source_corrected_spin_residual_max": float(np.max(np.abs(spin_corrected))),
            "microscopic_source_max": float(np.max(np.abs(source))),
        },
        "assessment": "PASS_MICROSCOPIC_VERTICAL_SOURCE_ATTACHMENT_QUADRATIC_REFERENCE",
        "claim_boundary": (
            "The continuity API now accepts the microscopic finite-lead "
            "vertical source and keeps raw and source-corrected residuals "
            "separate. This closes the quadratic reference at finite-grid "
            "accuracy only. An interacting second-Born mixed kernel and its "
            "continuum convergence remain open and are not inferred from this "
            "source."
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
