"""Gate 48: charge/spin projection of the microscopic vertical source."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantum_transport import (  # noqa: E402
    initial_correlation_charge_spin_source,
    partition_free_finite_lead_two_time,
    project_initial_correlation_source,
)


def _check(name: str, condition: bool, checks: dict[str, bool]) -> None:
    checks[name] = bool(condition)
    print(f"CHECK {name}: {'PASS' if condition else 'FAIL'}")


def run_gate() -> dict[str, object]:
    time = np.linspace(0.0, 0.5, 5)
    temperature = 0.3
    result = partition_free_finite_lead_two_time(
        time,
        np.linspace(0.0, 1.0 / temperature, 21),
        initial_device_hamiltonian=np.diag([-0.25, -0.18]).astype(complex),
        final_device_hamiltonian=np.diag([0.08, -0.02]).astype(complex),
        lead_hamiltonians=[np.diag([-0.8, -0.65]).astype(complex), np.diag([0.5, 0.62]).astype(complex)],
        coupling_matrices=[np.diag([0.25, 0.25]).astype(complex), np.diag([0.2, 0.2]).astype(complex)],
        lead_shifts=[0.15, -0.12],
        temperature=temperature,
    )
    spin_z = np.diag([1.0, -1.0]).astype(complex)
    projections = initial_correlation_charge_spin_source(result.initial_correlation.density_source, spin_z)
    charge_manual = project_initial_correlation_source(result.initial_correlation.density_source, np.eye(2))
    spin_manual = project_initial_correlation_source(result.initial_correlation.density_source, spin_z)
    checks: dict[str, bool] = {}
    _check("charge_projection_is_finite", np.all(np.isfinite(projections["charge"])), checks)
    _check("spin_projection_is_finite", np.all(np.isfinite(projections["spin"])), checks)
    _check("charge_api_matches_manual_trace", np.max(np.abs(projections["charge"] - charge_manual)) < 1e-14, checks)
    _check("spin_api_matches_manual_trace", np.max(np.abs(projections["spin"] - spin_manual)) < 1e-14, checks)
    _check("spin_channel_is_resolved", np.max(np.abs(projections["spin"])) > 1e-5, checks)
    _check("source_is_hermitian", result.initial_correlation.hermiticity_error < 1e-12, checks)
    report = {
        "gate": "GATE_48_CHARGE_SPIN_SOURCE_PROJECTION",
        "checks": checks,
        "passed": all(checks.values()),
        "charge_max": float(np.max(np.abs(projections["charge"]))),
        "spin_max": float(np.max(np.abs(projections["spin"]))),
        "source_hermiticity_error": result.initial_correlation.hermiticity_error,
        "assessment": "PASS_EXPLICIT_CHARGE_SPIN_VERTICAL_SOURCE_PROJECTION",
        "claim_boundary": (
            "Charge and spin projections of the microscopic vertical source are "
            "now public and reproducible. A resolved spin source is an observable "
            "diagnostic, not evidence for topological protection."
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
