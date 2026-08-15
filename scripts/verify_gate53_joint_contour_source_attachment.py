"""Gate 53: attach the joint-contour mixed source to charge/spin balance."""

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
    required_initial_source_from_residual,
    self_consistent_hubbard_second_born_contour_two_time,
    two_time_kbe_continuity_balance,
)


def _check(name: str, condition: bool, checks: dict[str, bool]) -> None:
    checks[name] = bool(condition)
    print(f"CHECK {name}: {'PASS' if condition else 'FAIL'}")


def run_gate() -> dict[str, object]:
    time = np.linspace(0.0, 0.5, 9)
    imaginary = np.linspace(0.0, 1.0 / 0.3, 21)
    reference = partition_free_finite_lead_two_time(
        time,
        imaginary,
        initial_device_hamiltonian=np.diag([-0.25, -0.18]).astype(complex),
        final_device_hamiltonian=np.diag([0.08, -0.02]).astype(complex),
        lead_hamiltonians=[np.diag([-0.8, -0.65]).astype(complex), np.diag([0.5, 0.62]).astype(complex)],
        coupling_matrices=[np.diag([0.25, 0.25]).astype(complex), np.diag([0.2, 0.2]).astype(complex)],
        lead_shifts=[0.15, -0.12],
        temperature=0.3,
    )
    bare_rceil = reference.green_mixed.swapaxes(0, 1).conj().swapaxes(-1, -2)
    result = self_consistent_hubbard_second_born_contour_two_time(
        time,
        imaginary,
        bare_retarded=reference.retarded,
        bare_lesser=reference.lesser,
        bare_mixed=bare_rceil,
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
    )
    balance = two_time_kbe_continuity_balance(
        time,
        green_retarded=result.retarded,
        green_lesser=result.lesser,
        hamiltonian=reference.final_device_hamiltonian,
        self_energy_retarded=reference.self_energy_retarded,
        self_energy_lesser=reference.self_energy_lesser,
        self_energy_advanced=reference.self_energy_advanced,
    )
    source = result.initial_correlation.density_source
    required = required_initial_source_from_residual(balance.residual)
    projections = initial_correlation_charge_spin_source(source, np.diag([1.0, -1.0]))
    corrected = balance.residual + source
    source_error = float(np.max(np.abs(required - source)))
    checks: dict[str, bool] = {}
    _check("joint_result_exposes_microscopic_source", result.initial_correlation is not None, checks)
    _check("mixed_self_energy_is_resolved", result.self_energy_mixed is not None and np.max(np.abs(result.self_energy_mixed)) > 1e-3, checks)
    _check("source_is_hermitian", result.initial_correlation.hermiticity_error < 1e-12, checks)
    _check("charge_spin_projections_are_finite", np.all(np.isfinite(projections["charge"])) and np.all(np.isfinite(projections["spin"])), checks)
    _check("required_source_is_separately_computable", np.all(np.isfinite(required)), checks)
    _check("joint_source_error_is_resolved", source_error > 1e-3, checks)
    report = {
        "gate": "GATE_53_JOINT_CONTOUR_SOURCE_ATTACHMENT",
        "checks": checks,
        "passed": all(checks.values()),
        "mixed_self_energy_max": float(np.max(np.abs(result.self_energy_mixed))),
        "source_density_max": float(np.max(np.abs(source))),
        "source_charge_max": float(np.max(np.abs(projections["charge"]))),
        "source_spin_max": float(np.max(np.abs(projections["spin"]))),
        "raw_continuity_residual_max": balance.maximum_residual,
        "source_corrected_residual_max": float(np.max(np.abs(corrected))),
        "required_source_error_max": source_error,
        "assessment": "PASS_JOINT_SOURCE_ATTACHED_NEGATIVE_CLOSURE_DIAGNOSTIC",
        "claim_boundary": (
            "The joint contour result now carries its microscopic mixed source "
            "and charge/spin projections. The required source still differs "
            "from that attached source, so the real lesser initial-correlation "
            "equation is not closed; this is a negative reproducible diagnostic, "
            "not an interacting conservation or topological claim."
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
