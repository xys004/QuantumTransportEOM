"""Gate 17: controlled EOM/Hubbard-I versus SCBA benchmark."""

from __future__ import annotations

import json

import numpy as np
import sympy as sp

from quantum_transport import (
    anderson_hubbard_i_green_function,
    self_consistent_born_electron_boson,
)


def _hubbard_i_stack(energy: np.ndarray, interaction: float) -> np.ndarray:
    omega = sp.Symbol("omega", real=True)
    expression = anderson_hubbard_i_green_function(
        "up",
        omega,
        sp.Float(0.2),
        sp.Float(0.15),
        sp.Float(0.15),
        sp.Float(interaction),
        occupations={"down": sp.Float(0.35)},
    )
    return np.asarray(sp.lambdify(omega, expression, "numpy")(energy), dtype=np.complex128)


def _spectral_sum(stack: np.ndarray, energy: np.ndarray) -> float:
    spectral = -2.0 * np.imag(stack)
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return float(np.real(trapezoid(spectral, energy) / (2.0 * np.pi)))


def run_gate() -> dict:
    energy = np.linspace(-12.0, 12.0, 1201)
    hamiltonian = np.array([[0.15]], dtype=complex)
    gammas = np.array([[[0.4]], [[0.0]]], dtype=complex)
    common = dict(
        hamiltonian=hamiltonian,
        energy=energy,
        lead_broadenings=gammas,
        lead_chemical_potentials=[0.0, 0.0],
        boson_frequency=0.7,
        temperature=0.2,
        max_iterations=60,
        mixing=0.4,
        tolerance=1e-9,
    )
    hubbard_noninteracting = _hubbard_i_stack(energy, 0.0)
    scba_noninteracting = self_consistent_born_electron_boson(
        **common, coupling=np.array([[0.0]], dtype=complex)
    )
    noninteracting_error = float(np.max(np.abs(hubbard_noninteracting - scba_noninteracting.retarded[:, 0, 0])))

    hubbard_interacting = _hubbard_i_stack(energy, 0.8)
    scba_interacting = self_consistent_born_electron_boson(
        **common, coupling=np.array([[0.06]], dtype=complex)
    )
    interacting_difference = float(np.max(np.abs(hubbard_interacting - scba_interacting.retarded[:, 0, 0])))
    hubbard_sum_error = abs(_spectral_sum(hubbard_interacting, energy) - 1.0)
    scba_sum_error = scba_interacting.spectral_sum_rule_error
    hubbard_spectral = -2.0 * np.imag(hubbard_interacting)
    scba_spectral = np.real(1j * (scba_interacting.retarded - scba_interacting.advanced)[:, 0, 0])

    checks = [
        {
            "name": "noninteracting_eom_scba_control",
            "passed": noninteracting_error < 2e-12 and scba_noninteracting.converged,
            "details": {"maximum_retarded_error": noninteracting_error},
        },
        {
            "name": "interacting_approximations_are_distinct",
            "passed": interacting_difference > 0.1,
            "details": {"maximum_retarded_difference": interacting_difference},
        },
        {
            "name": "spectral_positivity_and_window_controls",
            "passed": np.min(hubbard_spectral) > -1e-12
            and np.min(scba_spectral) > -1e-12
            and hubbard_sum_error < 0.03
            and scba_sum_error < 0.03,
            "details": {
                "hubbard_i_sum_error": hubbard_sum_error,
                "scba_sum_error": scba_sum_error,
                "scba_min_spectral": float(np.min(scba_spectral)),
            },
        },
    ]
    return {
        "gate": "GATE_17_EOM_HUBBARD_I_VS_SCBA",
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
        "claim_scope": "noninteracting control and controlled distinction between EOM/Hubbard-I and electron–boson SCBA approximations",
        "not_yet_claimed": [
            "Hubbard electron–electron SCBA for the same Anderson interaction",
            "benchmark against an exact many-body solver at finite U",
            "quantitative equivalence of the two interacting approximations",
        ],
    }


def main() -> None:
    report = run_gate()
    print(json.dumps(report, indent=2, sort_keys=True))
    for check in report["checks"]:
        print(f"CHECK {check['name']}: {'PASS' if check['passed'] else 'FAIL'}")
    print(f"VERDICT: {'PASS' if report['passed'] else 'FAIL'}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
