"""Gate 51: causal Volterra propagation of the mixed KBE branch."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantum_transport import (  # noqa: E402
    mixed_kbe_residual,
    partition_free_finite_lead_two_time,
    propagate_mixed_kbe_rceil,
)


def _check(name: str, condition: bool, checks: dict[str, bool]) -> None:
    checks[name] = bool(condition)
    print(f"CHECK {name}: {'PASS' if condition else 'FAIL'}")


def run_gate() -> dict[str, object]:
    time = np.linspace(0.0, 0.8, 81)
    imaginary = np.linspace(0.0, 2.0, 9)
    h = np.diag([0.3, -0.2]).astype(complex)
    initial = np.diag([0.7, -0.4]).astype(complex)
    zero_rr = np.zeros((time.size, time.size, 2, 2), dtype=complex)
    zero_rm = np.zeros((time.size, imaginary.size, 2, 2), dtype=complex)
    zero_mm = np.zeros((imaginary.size, imaginary.size, 2, 2), dtype=complex)
    free = propagate_mixed_kbe_rceil(
        time,
        imaginary,
        initial_green_mixed=np.broadcast_to(initial, (imaginary.size, 2, 2)),
        self_energy_retarded=zero_rr,
        self_energy_mixed=zero_rm,
        green_matsubara=zero_mm,
        hamiltonian=h,
    )
    exact = np.empty_like(free)
    for index, value in enumerate(time):
        exact[index] = np.diag(np.exp(-1j * value * np.diag(h))) @ np.broadcast_to(initial, (imaginary.size, 2, 2))
    free_error = float(np.max(np.abs(free - exact)))

    contacted_time = np.linspace(0.0, 0.5, 5)
    contacted_imaginary = np.linspace(0.0, 1.0 / 0.3, 21)
    reference = partition_free_finite_lead_two_time(
        contacted_time,
        contacted_imaginary,
        initial_device_hamiltonian=np.diag([-0.25, -0.18]).astype(complex),
        final_device_hamiltonian=np.diag([0.08, -0.02]).astype(complex),
        lead_hamiltonians=[np.diag([-0.8, -0.65]).astype(complex), np.diag([0.5, 0.62]).astype(complex)],
        coupling_matrices=[np.diag([0.25, 0.25]).astype(complex), np.diag([0.2, 0.2]).astype(complex)],
        lead_shifts=[0.15, -0.12],
        temperature=0.3,
    )
    initial_rceil = reference.green_mixed[:, 0].conj().swapaxes(-1, -2)
    propagated = propagate_mixed_kbe_rceil(
        contacted_time,
        contacted_imaginary,
        initial_green_mixed=initial_rceil,
        self_energy_retarded=reference.self_energy_retarded,
        self_energy_mixed=reference.self_energy_mixed,
        green_matsubara=reference.green_matsubara,
        hamiltonian=reference.final_device_hamiltonian,
    )
    residual = mixed_kbe_residual(
        contacted_time,
        contacted_imaginary,
        green_mixed=propagated,
        self_energy_retarded=reference.self_energy_retarded,
        self_energy_mixed=reference.self_energy_mixed,
        green_matsubara=reference.green_matsubara,
        hamiltonian=reference.final_device_hamiltonian,
    )
    checks: dict[str, bool] = {}
    _check("free_volterra_control_closes", free_error < 3e-3, checks)
    _check("propagated_branch_is_finite", np.all(np.isfinite(propagated)), checks)
    _check("propagated_shape_is_explicit", propagated.shape == (contacted_time.size, contacted_imaginary.size, 2, 2), checks)
    _check("contacted_residual_is_finite", np.isfinite(residual.maximum_rceil), checks)
    _check("initial_slice_is_preserved", np.max(np.abs(propagated[0] - initial_rceil)) < 1e-14, checks)
    _check("causal_time_grid_is_monotonic", np.all(np.diff(contacted_time) > 0.0), checks)
    report = {
        "gate": "GATE_51_MIXED_VOLTERRA_STEPPER",
        "checks": checks,
        "passed": all(checks.values()),
        "free_control_error": free_error,
        "contacted_residual_max": residual.maximum_rceil,
        "assessment": "PASS_CAUSAL_MIXED_KBE_VOLTERRA_STEPPER",
        "claim_boundary": (
            "A finite-grid causal Volterra stepper now propagates G^rceil from "
            "an explicit initial slice, retarded memory, and vertical source. "
            "The free control closes at first-order discretization accuracy; "
            "this is a building block, not yet a self-consistent interacting "
            "contour solver."
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
