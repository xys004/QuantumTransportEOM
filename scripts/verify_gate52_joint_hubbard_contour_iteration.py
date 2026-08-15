"""Gate 52: joint real/mixed Hubbard second-Born contour iteration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantum_transport import (  # noqa: E402
    partition_free_finite_lead_two_time,
    self_consistent_hubbard_second_born_contour_two_time,
)


def _check(name: str, condition: bool, checks: dict[str, bool]) -> None:
    checks[name] = bool(condition)
    print(f"CHECK {name}: {'PASS' if condition else 'FAIL'}")


def _run(interaction_u: float):
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
        interaction_u=interaction_u,
        embedding_self_energy_retarded=reference.self_energy_retarded,
        embedding_self_energy_mixed=reference.self_energy_mixed,
        spin_pairs=((0, 1), (1, 0)),
        max_iterations=100,
        dyson_iterations=100,
        mixing=0.2,
        tolerance=1e-7,
        include_hartree=True,
    )
    return result, bare_rceil


def run_gate() -> dict[str, object]:
    zero, bare = _run(0.0)
    finite, _ = _run(0.5)
    zero_mixed = zero.green_rceil
    finite_mixed = finite.green_rceil
    checks: dict[str, bool] = {}
    _check("u0_joint_iteration_converges", zero.converged, checks)
    _check("finite_u_joint_iteration_converges", finite.converged, checks)
    _check("real_spectral_identity_is_preserved", zero.spectral_identity_error < 2e-14 and finite.spectral_identity_error < 2e-14, checks)
    _check("u0_mixed_source_is_zero", zero.self_energy_mixed is not None and np.max(np.abs(zero.self_energy_mixed)) < 1e-14, checks)
    _check("finite_u_mixed_source_is_resolved", finite.self_energy_mixed is not None and np.max(np.abs(finite.self_energy_mixed)) > 1e-3, checks)
    _check("mixed_branch_changes_at_finite_u", zero_mixed is not None and finite_mixed is not None and np.max(np.abs(finite_mixed - zero_mixed)) > 1e-3, checks)
    report = {
        "gate": "GATE_52_JOINT_HUBBARD_CONTOUR_ITERATION",
        "checks": checks,
        "passed": all(checks.values()),
        "u0_iterations": zero.iterations,
        "finite_u_iterations": finite.iterations,
        "u0_maximum_update": zero.maximum_update,
        "finite_u_maximum_update": finite.maximum_update,
        "u0_mixed_source_max": float(np.max(np.abs(zero.self_energy_mixed))),
        "finite_u_mixed_source_max": float(np.max(np.abs(finite.self_energy_mixed))),
        "mixed_branch_change_max": float(np.max(np.abs(finite_mixed - zero_mixed))),
        "bare_to_u0_mixed_change_max": float(np.max(np.abs(zero_mixed - bare))),
        "assessment": "PASS_JOINT_REAL_MIXED_HUBBARD_FIXED_POINT_BOUNDED",
        "claim_boundary": (
            "The real and mixed Hubbard second-Born self-energies now iterate "
            "on the same Green branches, with embedding memory supplied to the "
            "mixed Volterra equation. The real lesser equation still retains "
            "its supplied bare initial term; full conserving contour closure "
            "remains open and no topological claim is made."
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
