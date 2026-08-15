"""Gate 26: microscopic initial-correlation closure with finite leads."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

from quantum_transport import (  # noqa: E402
    partition_free_finite_lead_two_time,
    two_time_kbe_continuity_balance,
)


def _check(name: str, condition: bool, checks: dict[str, bool]) -> None:
    checks[name] = bool(condition)
    print(f"CHECK {name}: {'PASS' if condition else 'FAIL'}")


def run_gate() -> dict:
    h_initial = np.array([[0.2, 0.12 - 0.03j], [0.12 + 0.03j, -0.15]], dtype=complex)
    h_final = np.array([[0.1, 0.08 + 0.04j], [0.08 - 0.04j, -0.05]], dtype=complex)
    leads = [
        np.diag([-1.4, -0.3]).astype(complex),
        np.diag([0.2, 0.8]).astype(complex),
    ]
    couplings = [
        np.array([[0.1, 0.02j], [0.03, 0.08j]], dtype=complex),
        np.array([[0.06, -0.04j], [0.07, 0.02]], dtype=complex),
    ]
    result = partition_free_finite_lead_two_time(
        np.linspace(0.0, 1.0, 201),
        np.linspace(0.0, 3.0, 321),
        initial_device_hamiltonian=h_initial,
        final_device_hamiltonian=h_final,
        lead_hamiltonians=leads,
        coupling_matrices=couplings,
        lead_shifts=(0.23, -0.19),
        chemical_potential=0.0,
        temperature=1.0 / 3.0,
    )
    balance = two_time_kbe_continuity_balance(
        result.time,
        green_retarded=result.retarded,
        green_lesser=result.lesser,
        hamiltonian=result.final_device_hamiltonian,
        self_energy_retarded=result.self_energy_retarded,
        self_energy_lesser=result.self_energy_lesser,
        self_energy_advanced=result.self_energy_advanced,
    )
    raw_residual = balance.residual
    corrected = raw_residual + result.initial_correlation.density_source
    sigma_z = np.diag([1.0, -1.0]).astype(complex)
    charge_error = float(np.max(np.abs(corrected[2:-2])))
    spin_raw = np.real(np.trace(sigma_z @ raw_residual, axis1=-2, axis2=-1))
    spin_source = np.real(
        np.trace(sigma_z @ result.initial_correlation.density_source, axis1=-2, axis2=-1)
    )
    spin_error = float(np.max(np.abs((spin_raw + spin_source)[2:-2])))
    source_norm = float(np.max(np.abs(result.initial_correlation.density_source)))
    initial_cross = float(np.max(np.abs(result.initial_density[:2, 2:])))
    checks: dict[str, bool] = {}
    _check("contacted_initial_state_has_device_lead_coherence", initial_cross > 1e-4, checks)
    _check("device_spectral_identity", result.spectral_identity_error < 1e-12, checks)
    _check("microscopic_source_is_hermitian", result.initial_correlation.hermiticity_error < 1e-12, checks)
    _check("charge_continuity_closes_after_microscopic_source", charge_error < 2e-5, checks)
    _check("spin_continuity_closes_after_microscopic_source", spin_error < 2e-5, checks)
    _check("initial_source_is_nonzero_and_not_markovian_zero", source_norm > 1e-3, checks)
    report = {
        "gate": "GATE_26_FINITE_LEAD_INITIAL_CORRELATION_CLOSURE",
        "checks": checks,
        "passed": all(checks.values()),
        "geometry": {"device_orbitals": 2, "lead_orbitals": 6, "time_points": result.time.size, "imaginary_points": result.imaginary_time.size},
        "metrics": {
            "initial_device_lead_coherence": initial_cross,
            "spectral_identity_error": result.spectral_identity_error,
            "initial_source_hermiticity_error": result.initial_correlation.hermiticity_error,
            "raw_residual_max": float(np.max(np.abs(raw_residual))),
            "microscopic_source_max": source_norm,
            "corrected_charge_interior_max": charge_error,
            "corrected_spin_interior_max": spin_error,
        },
        "assessment": "PASS_MICROSCOPIC_FINITE_LEAD_CLOSURE_CONTINUUM_LIMIT_OPEN",
        "claim_boundary": (
            "The vertical-branch initial-contact source closes charge and spin "
            "continuity for an exact finite quadratic device-plus-leads benchmark. "
            "This does not yet establish the continuum/WBL interacting Corbino limit; "
            "lead-size, recurrence, and interaction convergence remain required."
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
