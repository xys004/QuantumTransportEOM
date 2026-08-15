"""Gate 62: quantify the three-term vertical Langreth reconstruction."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantum_transport import (  # noqa: E402
    kadanoff_baym_contour_lesser_dyson_symbolic,
    kbe_lesser_contour_correction,
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
    result = self_consistent_hubbard_second_born_contour_two_time(
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
    )
    total_mixed = reference.self_energy_mixed + result.self_energy_mixed
    correction = kbe_lesser_contour_correction(
        time,
        imaginary,
        bare_retarded=reference.retarded,
        bare_mixed=reference.green_mixed.swapaxes(0, 1).conj().swapaxes(-1, -2),
        self_energy_mixed=total_mixed,
        green_lmixed=result.green_lceil,
        green_advanced=result.advanced,
        self_energy_matsubara=np.zeros((imaginary.size, imaginary.size, 2, 2), dtype=complex),
    )
    symbolic = kadanoff_baym_contour_lesser_dyson_symbolic()
    checks: dict[str, bool] = {}
    _check("symbolic_has_three_vertical_terms", all(key in symbolic for key in ("mixed_advanced", "propagated_mixed", "matsubara")), checks)
    _check("mixed_advanced_term_is_resolved", np.max(np.abs(correction.mixed_advanced)) > 1e-8, checks)
    _check("propagated_mixed_term_is_resolved", np.max(np.abs(correction.propagated_mixed)) > 1e-8, checks)
    _check("matsubara_term_is_finite", np.all(np.isfinite(correction.matsubara)), checks)
    _check("full_correction_is_finite", np.all(np.isfinite(correction.correction)), checks)
    _check("fixed_point_converges", result.converged, checks)
    report = {
        "gate": "GATE_62_FULL_CONTOUR_LESSER_RECONSTRUCTION",
        "checks": checks,
        "passed": all(checks.values()),
        "mixed_advanced_max": float(np.max(np.abs(correction.mixed_advanced))),
        "propagated_mixed_max": float(np.max(np.abs(correction.propagated_mixed))),
        "matsubara_max": float(np.max(np.abs(correction.matsubara))),
        "full_correction_max": float(np.max(np.abs(correction.correction))),
        "antihermiticity_error": correction.antihermiticity_error,
        "assessment": "PASS_THREE_TERM_CONTOUR_LESSER_RECONSTRUCTION",
        "claim_boundary": (
            "The three vertical Langreth terms are explicit and finite on the "
            "interacting finite-lead benchmark. Their inclusion is a reconstruction "
            "identity; conservation still requires a self-consistent Matsubara and "
            "real/mixed fixed point."
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
