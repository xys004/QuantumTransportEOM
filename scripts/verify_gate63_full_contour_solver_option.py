"""Gate 63: audit the optional three-term contour lesser solver branch."""

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
    common = dict(
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
    )
    legacy = self_consistent_hubbard_second_born_contour_two_time(
        time, imaginary, include_vertical_lesser=True, **common
    )
    full = self_consistent_hubbard_second_born_contour_two_time(
        time,
        imaginary,
        include_full_contour_lesser=True,
        self_energy_matsubara=np.zeros_like(reference.green_matsubara),
        **common,
    )
    correction = full.lesser_contour_correction
    checks: dict[str, bool] = {}
    _check("full_branch_converges", full.converged, checks)
    _check("legacy_branch_remains_available", legacy.lesser_initial_correlation is not None, checks)
    _check("full_correction_is_attached", correction is not None, checks)
    _check(
        "all_three_terms_are_finite",
        correction is not None
        and all(
            np.all(np.isfinite(value))
            for value in (correction.mixed_advanced, correction.propagated_mixed, correction.matsubara)
        ),
        checks,
    )
    _check(
        "full_branch_resolves_vertical_terms",
        correction is not None
        and np.max(np.abs(correction.mixed_advanced)) > 1e-8
        and np.max(np.abs(correction.propagated_mixed)) > 1e-8,
        checks,
    )
    _check(
        "full_and_legacy_lesser_differ",
        np.max(np.abs(full.lesser - legacy.lesser)) > 1e-8,
        checks,
    )
    report = {
        "gate": "GATE_63_FULL_CONTOUR_SOLVER_OPTION",
        "checks": checks,
        "passed": all(checks.values()),
        "full_iterations": full.iterations,
        "legacy_iterations": legacy.iterations,
        "full_maximum_update": full.maximum_update,
        "full_correction_max": float(np.max(np.abs(correction.correction))) if correction is not None else None,
        "mixed_advanced_max": float(np.max(np.abs(correction.mixed_advanced))) if correction is not None else None,
        "propagated_mixed_max": float(np.max(np.abs(correction.propagated_mixed))) if correction is not None else None,
        "matsubara_max": float(np.max(np.abs(correction.matsubara))) if correction is not None else None,
        "legacy_full_lesser_difference_max": float(np.max(np.abs(full.lesser - legacy.lesser))),
        "antihermiticity_error": correction.antihermiticity_error if correction is not None else None,
        "assessment": "PASS_OPTIONAL_FULL_CONTOUR_SOLVER_INTERFACE",
        "claim_boundary": (
            "The solver can select the explicit three-term lesser contour branch and retain the legacy branch. "
            "A zero supplied Matsubara interaction remains an approximation; no conserving closure or protection claim follows."
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
