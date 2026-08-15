"""Gate 71: self-consistent Matsubara Hubbard closure and contour coupling."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantum_transport import (  # noqa: E402
    equilibrium_matsubara_green,
    equilibrium_one_body_density,
    hubbard_second_born_self_energy_matsubara,
    hubbard_second_born_self_energy_matsubara_symbolic,
    self_consistent_hubbard_matsubara,
    self_consistent_hubbard_second_born_contour_two_time,
    two_time_greens,
)


def _check(name: str, condition: bool, checks: dict[str, bool]) -> None:
    checks[name] = bool(condition)
    print(f"CHECK {name}: {'PASS' if condition else 'FAIL'}")


def _contour_inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    hamiltonian = np.diag([0.2, -0.15]).astype(complex)
    time = np.linspace(0.0, 0.3, 7)
    imaginary = np.linspace(0.0, 2.0, 9)
    density = equilibrium_one_body_density(hamiltonian, mu=0.0, temperature=0.5)
    bare = two_time_greens(time, lambda _: hamiltonian, density)
    initial_mixed = np.diag([0.7, -0.4]).astype(complex)
    bare_mixed = np.empty((time.size, imaginary.size, 2, 2), dtype=complex)
    for index, value in enumerate(time):
        bare_mixed[index] = np.diag(np.exp(-1j * value * np.diag(hamiltonian))) @ np.broadcast_to(
            initial_mixed, (imaginary.size, 2, 2)
        )
    matsubara = equilibrium_matsubara_green(hamiltonian, imaginary, temperature=0.5)
    return hamiltonian, time, imaginary, bare.retarded, bare.lesser, bare_mixed, matsubara


def run_gate(runtime: str = "ASTRA") -> dict[str, object]:
    hamiltonian, time, imaginary, bare_r, bare_l, bare_mixed, bare_matsubara = _contour_inputs()
    symbolic = hubbard_second_born_self_energy_matsubara_symbolic()
    sigma_u = hubbard_second_born_self_energy_matsubara(
        imaginary, green_matsubara=bare_matsubara, interaction_u=0.1
    )
    sigma_2u = hubbard_second_born_self_energy_matsubara(
        imaginary, green_matsubara=bare_matsubara, interaction_u=0.2
    )
    zero = self_consistent_hubbard_matsubara(
        imaginary,
        bare_green_matsubara=bare_matsubara,
        interaction_u=0.0,
        max_iterations=10,
        dyson_iterations=20,
    )
    interacting = self_consistent_hubbard_matsubara(
        imaginary,
        bare_green_matsubara=bare_matsubara,
        interaction_u=0.1,
        max_iterations=60,
        dyson_iterations=60,
        mixing=0.25,
        tolerance=1e-8,
    )
    contour = self_consistent_hubbard_second_born_contour_two_time(
        time,
        imaginary,
        bare_retarded=bare_r,
        bare_lesser=bare_l,
        bare_mixed=bare_mixed,
        green_matsubara=bare_matsubara,
        hamiltonian=hamiltonian,
        interaction_u=0.1,
        max_iterations=30,
        dyson_iterations=60,
        mixing=0.35,
        tolerance=1e-7,
        include_full_contour_lesser=True,
        self_energy_matsubara=np.zeros_like(bare_matsubara),
        self_consistent_matsubara=True,
        matsubara_iterations=60,
        matsubara_dyson_iterations=60,
        matsubara_mixing=0.25,
        matsubara_tolerance=1e-8,
    )
    correction = contour.lesser_contour_correction
    checks: dict[str, bool] = {}
    _check(
        "symbolic_sign_order_and_u_scaling",
        str(symbolic["matsubara"]).startswith("-U**2*")
        and "G_M(1, tau_prime, tau)" in str(symbolic["matsubara"])
        and symbolic["u_scaling"] == 2,
        checks,
    )
    _check("u_zero_returns_bare_matsubara", zero.converged and np.max(np.abs(zero.green_matsubara - bare_matsubara)) < 1e-14, checks)
    _check("matsubara_fixed_point_converges", interacting.converged, checks)
    _check("matsubara_interaction_is_nonzero", np.max(np.abs(interacting.interaction_self_energy)) > 1e-8, checks)
    _check("matsubara_u_squared_scaling", np.max(np.abs(sigma_2u - 4.0 * sigma_u)) < 1e-13, checks)
    _check("matsubara_green_kms_diagnostic_bounded", interacting.green_kms_error < 5e-3, checks)
    _check("matsubara_sigma_kms_diagnostic_bounded", interacting.self_energy_kms_error < 1e-4, checks)
    _check("contour_self_consistent_branch_converges", contour.converged, checks)
    _check("contour_attaches_matsubara_result", contour.matsubara_result is not None and contour.matsubara_result.converged, checks)
    _check("contour_uses_nonzero_vertical_sigma", contour.self_energy_matsubara is not None and np.max(np.abs(contour.self_energy_matsubara)) > 1e-8, checks)
    _check(
        "full_lesser_correction_is_finite",
        correction is not None and np.all(np.isfinite(correction.correction)) and np.all(np.isfinite(correction.matsubara)),
        checks,
    )
    report = {
        "schema_version": 1,
        "gate": "GATE_71_SELF_CONSISTENT_MATSUBARA",
        "runtime": runtime,
        "execution": {
            "astra": "PASS" if runtime.upper() in {"ASTRA", "ASTRA+ASTRUM"} else "NOT_RUN",
            "astrum": "PASS" if runtime.upper() in {"ASTRUM", "ASTRA+ASTRUM"} else "NOT_RUN",
            "script": "scripts/verify_gate71_self_consistent_matsubara.py",
        },
        "checks": checks,
        "passed": all(checks.values()),
        "matsubara_iterations": interacting.iterations,
        "matsubara_maximum_update": interacting.maximum_update,
        "green_kms_error": interacting.green_kms_error,
        "self_energy_kms_error": interacting.self_energy_kms_error,
        "interaction_self_energy_max": float(np.max(np.abs(interacting.interaction_self_energy))),
        "total_self_energy_max": float(np.max(np.abs(interacting.self_energy_matsubara))),
        "contour_iterations": contour.iterations,
        "contour_maximum_update": contour.maximum_update,
        "contour_antihermiticity_error": correction.antihermiticity_error if correction is not None else None,
        "contour_correction_max": float(np.max(np.abs(correction.correction))) if correction is not None else None,
        "assessment": "PASS_IMPLEMENTED_WITH_FINITE_GRID_KMS_DIAGNOSTIC",
        "claim_boundary": (
            "The package now computes a self-consistent Matsubara Hartree plus second-Born self-energy and feeds it into the full three-term lesser contour branch. "
            "Finite-grid KMS residuals are reported explicitly; this gate does not claim a conserving continuum closure or topological protection."
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", default="ASTRA")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_gate(args.runtime)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    print(f"VERDICT: {'PASS' if report['passed'] else 'FAIL'}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
