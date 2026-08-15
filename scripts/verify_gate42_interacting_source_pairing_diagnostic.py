"""Gate 42: negative diagnostic for mismatched real/mixed interacting closures."""

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
    self_consistent_hubbard_second_born_two_time,
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
    scba = self_consistent_hubbard_second_born_two_time(
        time,
        bare_retarded=finite.retarded,
        bare_lesser=finite.lesser,
        interaction_u=0.5,
        spin_pairs=((0, 1), (1, 0)),
        include_hartree=True,
        max_iterations=100,
        dyson_iterations=80,
        mixing=0.2,
        tolerance=1e-7,
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
    sigma_mixed = hubbard_second_born_self_energy_mixed(
        time,
        imaginary,
        green_rceil=exact.green_rceil,
        green_lceil=exact.green_mixed,
        interaction_u=0.5,
        spin_pairs=((0, 1), (1, 0)),
    )
    exact_source = kbe_initial_correlation_kernel(
        time,
        imaginary,
        self_energy_mixed=sigma_mixed,
        green_mixed=exact.green_mixed,
    )
    raw = two_time_kbe_continuity_balance(
        time,
        green_retarded=scba.retarded,
        green_lesser=scba.lesser,
        hamiltonian=final_device,
        self_energy_retarded=scba.self_energy_retarded,
        self_energy_lesser=scba.self_energy_lesser,
        self_energy_advanced=scba.self_energy_advanced,
    )
    mismatched = two_time_kbe_continuity_balance(
        time,
        green_retarded=scba.retarded,
        green_lesser=scba.lesser,
        hamiltonian=final_device,
        self_energy_retarded=scba.self_energy_retarded,
        self_energy_lesser=scba.self_energy_lesser,
        self_energy_advanced=scba.self_energy_advanced,
        initial_correlation_source=exact_source.density_source,
    )
    quadratic = two_time_kbe_continuity_balance(
        time,
        green_retarded=finite.retarded,
        green_lesser=finite.lesser,
        hamiltonian=final_device,
        self_energy_retarded=finite.self_energy_retarded,
        self_energy_lesser=finite.self_energy_lesser,
        self_energy_advanced=finite.self_energy_advanced,
        initial_correlation_source=finite.continuity_initial_source,
    )
    raw_max = raw.maximum_residual
    mismatched_max = mismatched.maximum_source_corrected_residual
    quadratic_max = quadratic.maximum_source_corrected_residual
    checks: dict[str, bool] = {}
    _check("real_time_scba_converged", scba.converged, checks)
    _check("exact_interacting_source_is_hermitian", exact_source.hermiticity_error < 1e-12, checks)
    _check("quadratic_source_pairing_closes_reference", quadratic_max < 2e-4, checks)
    _check("mismatched_interacting_source_is_resolved", exact_source.density_source.size > 0 and np.max(np.abs(exact_source.density_source)) > 1e-4, checks)
    _check("mismatched_pairing_does_not_fake_closure", mismatched_max > 1.5 * raw_max, checks)
    _check("negative_diagnostic_is_finite", np.isfinite(mismatched_max), checks)
    report = {
        "gate": "GATE_42_INTERACTING_SOURCE_PAIRING_DIAGNOSTIC",
        "checks": checks,
        "passed": all(checks.values()),
        "metrics": {
            "scba_iterations": int(scba.iterations),
            "raw_scba_continuity_residual_max": raw_max,
            "mismatched_exact_source_corrected_residual_max": mismatched_max,
            "quadratic_source_corrected_residual_max": quadratic_max,
            "exact_interacting_source_max": float(np.max(np.abs(exact_source.density_source))),
            "mismatch_amplification": float(mismatched_max / raw_max),
        },
        "assessment": "PASS_NEGATIVE_MISMATCHED_BRANCH_DIAGNOSTIC_JOINT_CLOSURE_REQUIRED",
        "claim_boundary": (
            "The quadratic source/self-energy pairing closes its reference, but "
            "attaching an exact-interacting mixed source to the real-time SCBA "
            "solution worsens the residual. This is a negative reproducible "
            "criterion: real and vertical interacting branches must be solved "
            "jointly. No interacting conservation or topological claim is made."
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
