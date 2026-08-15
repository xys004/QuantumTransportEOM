"""Gate 47: numerical residual evaluator for mixed KBE branches."""

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
)


def _check(name: str, condition: bool, checks: dict[str, bool]) -> None:
    checks[name] = bool(condition)
    print(f"CHECK {name}: {'PASS' if condition else 'FAIL'}")


def _free_control() -> tuple[float, float]:
    time = np.linspace(0.0, 0.8, 81)
    imaginary = np.linspace(0.0, 2.0, 9)
    h = np.diag([0.3, -0.2]).astype(complex)
    diagonal = np.diag([0.7, -0.4]).astype(complex)
    evolution = np.exp(-1j * time[:, None] * np.diag(h)[None, :])
    rceil = np.zeros((time.size, imaginary.size, 2, 2), dtype=complex)
    lceil = np.zeros((imaginary.size, time.size, 2, 2), dtype=complex)
    for index in range(time.size):
        rceil[index] = np.diag(evolution[index]) @ np.broadcast_to(diagonal, (imaginary.size, 2, 2))
        lceil[:, index] = np.broadcast_to(diagonal, (imaginary.size, 2, 2)) @ np.diag(evolution[index].conj())
    zero_rr = np.zeros((time.size, time.size, 2, 2), dtype=complex)
    zero_rm = np.zeros_like(rceil)
    zero_mm = np.zeros((imaginary.size, imaginary.size, 2, 2), dtype=complex)
    residual = mixed_kbe_residual(
        time,
        imaginary,
        green_mixed=rceil,
        self_energy_retarded=zero_rr,
        self_energy_mixed=zero_rm,
        green_matsubara=zero_mm,
        hamiltonian=h,
        green_lmixed=lceil,
    )
    return residual.maximum_rceil, float(residual.maximum_lceil or 0.0)


def run_gate() -> dict[str, object]:
    free_rceil_error, free_lceil_error = _free_control()
    temperature = 0.3
    time = np.linspace(0.0, 0.5, 5)
    imaginary = np.linspace(0.0, 1.0 / temperature, 21)
    result = partition_free_finite_lead_two_time(
        time,
        imaginary,
        initial_device_hamiltonian=np.diag([-0.25, -0.18]).astype(complex),
        final_device_hamiltonian=np.diag([0.08, -0.02]).astype(complex),
        lead_hamiltonians=[np.diag([-0.8, -0.65]).astype(complex), np.diag([0.5, 0.62]).astype(complex)],
        coupling_matrices=[np.diag([0.25, 0.25]).astype(complex), np.diag([0.2, 0.2]).astype(complex)],
        lead_shifts=[0.15, -0.12],
        temperature=temperature,
    )
    green_rceil = result.green_mixed.swapaxes(0, 1).copy()
    residual = mixed_kbe_residual(
        time,
        imaginary,
        green_mixed=green_rceil,
        self_energy_retarded=result.self_energy_retarded,
        self_energy_mixed=result.self_energy_mixed,
        green_matsubara=result.green_matsubara,
        hamiltonian=result.final_device_hamiltonian,
        green_lmixed=result.green_mixed,
        self_energy_advanced=result.self_energy_advanced,
    )
    checks: dict[str, bool] = {}
    _check("free_rceil_residual_closes", free_rceil_error < 2e-3, checks)
    _check("free_lceil_residual_closes", free_lceil_error < 2e-3, checks)
    _check("exact_rceil_residual_is_finite", np.isfinite(residual.maximum_rceil), checks)
    _check("exact_lceil_residual_is_finite", residual.maximum_lceil is not None and np.isfinite(residual.maximum_lceil), checks)
    _check("residual_grid_is_preserved", residual.rceil.shape == green_rceil.shape, checks)
    _check("residual_has_both_branches", residual.lceil is not None and residual.lceil.shape == result.green_mixed.shape, checks)
    report = {
        "gate": "GATE_47_MIXED_KBE_RESIDUAL",
        "checks": checks,
        "passed": all(checks.values()),
        "free_rceil_residual_max": free_rceil_error,
        "free_lceil_residual_max": free_lceil_error,
        "exact_rceil_residual_max": residual.maximum_rceil,
        "exact_lceil_residual_max": residual.maximum_lceil,
        "assessment": "PASS_NUMERICAL_MIXED_KBE_RESIDUAL_DIAGNOSTIC",
        "claim_boundary": (
            "The package now evaluates both mixed KBE residual branches with "
            "causal and vertical quadrature. The free control closes at finite "
            "difference accuracy; the exact finite benchmark is reported as a "
            "diagnostic, not relabeled as an interacting self-consistent solver."
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
