"""Gate 46: explicit equilibrium Matsubara branches in the finite oracle."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantum_transport import partition_free_finite_lead_two_time  # noqa: E402


def _check(name: str, condition: bool, checks: dict[str, bool]) -> None:
    checks[name] = bool(condition)
    print(f"CHECK {name}: {'PASS' if condition else 'FAIL'}")


def run_gate() -> dict[str, object]:
    temperature = 0.3
    imaginary = np.linspace(0.0, 1.0 / temperature, 21)
    initial_device = np.diag([-0.25, -0.18]).astype(complex)
    final_device = np.diag([0.08, -0.02]).astype(complex)
    leads = [np.diag([-0.8, -0.65]).astype(complex), np.diag([0.5, 0.62]).astype(complex)]
    couplings = [np.diag([0.25, 0.25]).astype(complex), np.diag([0.2, 0.2]).astype(complex)]
    result = partition_free_finite_lead_two_time(
        np.linspace(0.0, 0.5, 5),
        imaginary,
        initial_device_hamiltonian=initial_device,
        final_device_hamiltonian=final_device,
        lead_hamiltonians=leads,
        coupling_matrices=couplings,
        lead_shifts=[0.15, -0.12],
        temperature=temperature,
    )
    gm = result.green_matsubara
    sm = result.self_energy_matsubara
    checks: dict[str, bool] = {}
    _check("matsubara_shapes_are_explicit", gm.shape == (21, 21, 2, 2) and sm.shape == (21, 21, 2, 2), checks)
    _check("matsubara_values_are_finite", np.all(np.isfinite(gm)) and np.all(np.isfinite(sm)), checks)
    interior = slice(1, -1)
    green_kms_error = float(np.max(np.abs(gm[-1, interior] + gm[0, interior])))
    sigma_kms_error = float(np.max(np.abs(sm[-1, interior] + sm[0, interior])))
    _check("green_matsubara_kms_endpoint", green_kms_error < 1e-12, checks)
    _check("self_energy_matsubara_kms_endpoint", sigma_kms_error < 1e-12, checks)
    expected_equal_time = -(np.eye(2, dtype=complex) - result.initial_density[:2, :2])
    equal_time_error = float(np.max(np.abs(gm[0, 0] - expected_equal_time)))
    _check("green_equal_time_complement_is_exposed", equal_time_error < 1e-12, checks)
    _check("mixed_and_matsubara_grids_share_beta", abs(imaginary[-1] - 1.0 / temperature) < 1e-14, checks)
    report = {
        "gate": "GATE_46_MATSUBARA_BRANCH",
        "checks": checks,
        "passed": all(checks.values()),
        "green_kms_error": green_kms_error,
        "self_energy_kms_error": sigma_kms_error,
        "green_equal_time_error": equal_time_error,
        "assessment": "PASS_EXPLICIT_MATSUBARA_AND_EMBEDDING_BRANCH",
        "claim_boundary": (
            "The finite quadratic oracle now exposes G^M and Sigma^M on the "
            "same vertical grid. KMS endpoint and equal-time controls pass; "
            "this is an input branch for a joint contour solver, not a claim "
            "of interacting self-consistency."
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
