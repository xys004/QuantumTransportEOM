"""Gate 54: propagate the mixed Keldysh source into the real lesser branch."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantum_transport import (  # noqa: E402
    initial_correlation_charge_spin_source,
    kadanoff_baym_lesser_initial_correlation_symbolic,
    partition_free_finite_lead_two_time,
    required_initial_source_from_residual,
    self_consistent_hubbard_second_born_contour_two_time,
    two_time_kbe_continuity_balance,
)


def _check(name: str, condition: bool, checks: dict[str, bool]) -> None:
    checks[name] = bool(condition)
    print(f"CHECK {name}: {'PASS' if condition else 'FAIL'}")


def _run_result(reference, time, imaginary, *, include_vertical_lesser: bool):
    return self_consistent_hubbard_second_born_contour_two_time(
        time,
        imaginary,
        bare_retarded=reference.retarded,
        bare_lesser=reference.lesser,
        bare_mixed=reference.green_mixed.swapaxes(0, 1).conj().swapaxes(-1, -2),
        green_matsubara=reference.green_matsubara,
        hamiltonian=reference.final_device_hamiltonian,
        interaction_u=0.5,
        embedding_self_energy_retarded=reference.self_energy_retarded,
        embedding_self_energy_mixed=reference.self_energy_mixed,
        spin_pairs=((0, 1), (1, 0)),
        max_iterations=100,
        dyson_iterations=100,
        mixing=0.2,
        tolerance=1e-7,
        include_hartree=True,
        include_vertical_lesser=include_vertical_lesser,
    )


def run_gate() -> dict[str, object]:
    time = np.linspace(0.0, 0.5, 9)
    imaginary = np.linspace(0.0, 1.0 / 0.3, 21)
    reference = partition_free_finite_lead_two_time(
        time,
        imaginary,
        initial_device_hamiltonian=np.diag([-0.25, -0.18]).astype(complex),
        final_device_hamiltonian=np.diag([0.08, -0.02]).astype(complex),
        lead_hamiltonians=[
            np.diag([-0.8, -0.65]).astype(complex),
            np.diag([0.5, 0.62]).astype(complex),
        ],
        coupling_matrices=[
            np.diag([0.25, 0.25]).astype(complex),
            np.diag([0.2, 0.2]).astype(complex),
        ],
        lead_shifts=[0.15, -0.12],
        temperature=0.3,
    )
    bare = _run_result(reference, time, imaginary, include_vertical_lesser=False)
    closed = _run_result(reference, time, imaginary, include_vertical_lesser=True)
    bare_balance = two_time_kbe_continuity_balance(
        time,
        green_retarded=bare.retarded,
        green_lesser=bare.lesser,
        hamiltonian=reference.final_device_hamiltonian,
        self_energy_retarded=reference.self_energy_retarded,
        self_energy_lesser=reference.self_energy_lesser,
        self_energy_advanced=reference.self_energy_advanced,
    )
    closed_balance = two_time_kbe_continuity_balance(
        time,
        green_retarded=closed.retarded,
        green_lesser=closed.lesser,
        hamiltonian=reference.final_device_hamiltonian,
        self_energy_retarded=reference.self_energy_retarded,
        self_energy_lesser=reference.self_energy_lesser,
        self_energy_advanced=reference.self_energy_advanced,
    )
    vertical = closed.lesser_initial_correlation
    diagonal_source = vertical.source_kernel[np.arange(time.size), np.arange(time.size)]
    diagonal_source = diagonal_source + diagonal_source.swapaxes(-1, -2).conj()
    required = required_initial_source_from_residual(closed_balance.residual)
    projections = initial_correlation_charge_spin_source(
        diagonal_source, np.diag([1.0, -1.0])
    )
    symbolic = kadanoff_baym_lesser_initial_correlation_symbolic(
        time=sp.Symbol("t"), time_prime=sp.Symbol("t_prime")
    )
    checks: dict[str, bool] = {}
    _check("symbolic_vertical_propagation_is_explicit", "G_r(t, t_bar)" in str(symbolic["propagated_source"]), checks)
    _check("lesser_vertical_result_is_exposed", vertical is not None, checks)
    _check("lesser_correction_is_antihermitian", vertical is not None and vertical.antihermiticity_error < 1e-12, checks)
    _check("charge_spin_channels_are_finite", np.all(np.isfinite(projections["charge"])) and np.all(np.isfinite(projections["spin"])), checks)
    _check("closed_run_converges", closed.converged, checks)
    _check("closure_changes_lesser_branch", np.max(np.abs(closed.lesser - bare.lesser)) > 1e-5, checks)
    report = {
        "gate": "GATE_54_LESSER_VERTICAL_INITIAL_CORRELATION",
        "checks": checks,
        "passed": all(checks.values()),
        "bare_residual_max": bare_balance.maximum_residual,
        "vertical_residual_max": closed_balance.maximum_residual,
        "residual_ratio": float(closed_balance.maximum_residual / bare_balance.maximum_residual),
        "lesser_correction_max": float(np.max(np.abs(vertical.correction))),
        "source_kernel_max": float(np.max(np.abs(vertical.source_kernel))),
        "source_charge_max": float(np.max(np.abs(projections["charge"]))),
        "source_spin_max": float(np.max(np.abs(projections["spin"]))),
        "required_source_error_max": float(np.max(np.abs(required - diagonal_source))),
        "assessment": "PASS_EXPLICIT_LESSER_VERTICAL_TERM_WITH_RESIDUAL_AUDIT",
        "claim_boundary": (
            "The real lesser update now accepts the propagated mixed-branch term "
            "G^R*(-i Sigma^rceil*G^lceil) plus its adjoint. The gate reports the "
            "continuity residual before and after the term; any remaining "
            "residual is an open closure error, not evidence of conservation or "
            "topological protection."
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
