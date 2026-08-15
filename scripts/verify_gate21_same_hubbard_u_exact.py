"""Gate 21: same-Hubbard-U exact atomic benchmark for EOM/Hubbard-I."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantum_transport import (  # noqa: E402
    anderson_hubbard_i_green_function,
    anderson_impurity_model,
    atomic_hubbard_u_probabilities,
    atomic_hubbard_u_retarded_frequency,
    atomic_hubbard_u_two_time,
)


def _check(name: str, condition: bool, checks: dict[str, bool]) -> None:
    checks[name] = bool(condition)
    print(f"CHECK {name}: {'PASS' if condition else 'FAIL'}")


def run_gate() -> dict:
    epsilon_up, epsilon_down, interaction_u = -0.35, 0.18, 1.05
    chemical_potential, temperature, eta = 0.07, 0.23, 0.02
    probabilities = atomic_hubbard_u_probabilities(
        epsilon_up, epsilon_down, interaction_u,
        chemical_potential=chemical_potential, temperature=temperature,
    )
    opposite = float(probabilities[2] + probabilities[3])
    energy = np.linspace(-3.0, 3.0, 801)
    exact_frequency = atomic_hubbard_u_retarded_frequency(
        energy, epsilon_up, epsilon_down, interaction_u, spin="up", eta=eta,
        chemical_potential=chemical_potential, temperature=temperature,
    )
    eom_hubbard_i = np.asarray([
        complex(anderson_hubbard_i_green_function(
            "up", float(omega), eta, epsilon_up, epsilon_down, interaction_u,
            occupations={"down": opposite},
        ))
        for omega in energy
    ])
    time_result = atomic_hubbard_u_two_time(
        np.linspace(0.0, 4.0, 81), epsilon_up, epsilon_down, interaction_u,
        spin="up", chemical_potential=chemical_potential, temperature=temperature,
    )
    eps_up, eps_down, u = sp.symbols("eps_up eps_down U", real=True)
    symbolic_eom = anderson_impurity_model(eps_up, eps_down, u).eom(auto_expand_steps=1)
    checks: dict[str, bool] = {}
    _check("probability_normalization", abs(float(np.sum(probabilities)) - 1.0) < 2e-15, checks)
    _check("same_hubbard_u_eom_basis_closed", symbolic_eom.is_closed and symbolic_eom.eom_matrix.shape == (4, 4), checks)
    _check("same_hubbard_u_hubbard_i_matches_exact_frequency", float(np.max(np.abs(eom_hubbard_i - exact_frequency))) < 3e-12, checks)
    _check("two_time_spectral_identity", time_result.spectral_identity_error < 3e-14, checks)
    _check("two_time_advanced_adjoint", time_result.advanced_adjoint_error < 3e-14, checks)
    _check("two_time_lesser_antihermitian", time_result.lesser_antihermiticity_error < 3e-14, checks)
    _check("two_time_equal_time_occupation", time_result.equal_time_lesser_error < 3e-14, checks)
    return {
        "gate": "GATE_21_SAME_HUBBARD_U_EXACT",
        "scope": "exact atomic Hubbard-U diagonal ensemble versus EOM/Hubbard-I with identical epsilon and U",
        "parameters": {
            "epsilon_up": epsilon_up, "epsilon_down": epsilon_down,
            "interaction_u": interaction_u, "chemical_potential": chemical_potential,
            "temperature": temperature, "eta": eta,
        },
        "probabilities": probabilities.tolist(),
        "opposite_spin_occupation": opposite,
        "checks": checks,
        "metrics": {
            "frequency_max_residual": float(np.max(np.abs(eom_hubbard_i - exact_frequency))),
            "two_time_spectral_identity_error": time_result.spectral_identity_error,
            "two_time_advanced_adjoint_error": time_result.advanced_adjoint_error,
            "two_time_lesser_antihermiticity_error": time_result.lesser_antihermiticity_error,
            "two_time_equal_time_occupation_error": time_result.equal_time_lesser_error,
        },
        "claim_boundary": "The equality is exact only for the atomic diagonal Hubbard-U Hamiltonian; it does not certify Hubbard-I for hopping, leads, or arbitrary interacting memory.",
        "passed": all(checks.values()),
    }


def main() -> None:
    report = run_gate()
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"VERDICT: {'PASS' if report['passed'] else 'FAIL'}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
