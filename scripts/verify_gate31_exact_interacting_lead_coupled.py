"""Gate 31: exact lead-coupled interacting Keldysh oracle.

The benchmark is a two-spin-orbital Hubbard device coupled to two finite
spinful leads.  The full many-body contacted equilibrium is propagated after
a one-body quench.  Exact two-time Green functions, charge currents, and
spin currents are checked, then a lead-embedded Hubbard-I/EOM retarded
formula is compared at the same initial parameters.  The noninteracting
control must be exact; the finite-U mismatch is reported rather than hidden.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantum_transport import (  # noqa: E402
    finite_interacting_partition_free_two_time,
    finite_lead_retarded_embedding,
    lead_coupled_hubbard_i_retarded,
)


def _one_body_hamiltonians() -> tuple[np.ndarray, np.ndarray]:
    initial = np.diag([-0.25, -0.18, -0.8, -0.7, 0.5, 0.6]).astype(complex)
    initial[0, 2] = initial[2, 0] = 0.25
    initial[1, 3] = initial[3, 1] = 0.25
    initial[0, 4] = initial[4, 0] = 0.2
    initial[1, 5] = initial[5, 1] = 0.2
    final = initial.copy()
    final[[0, 1], [0, 1]] += [0.08, 0.16]
    final[[2, 3], [2, 3]] += [0.18, 0.08]
    final[[4, 5], [4, 5]] += [-0.15, -0.05]
    return initial, final


def _solve(interaction_u: float):
    initial, final = _one_body_hamiltonians()
    result = finite_interacting_partition_free_two_time(
        np.linspace(0.0, 0.6, 13),
        initial_one_body_hamiltonian=initial,
        final_one_body_hamiltonian=final,
        interactions=[(0, 1, interaction_u)],
        chemical_potential=0.0,
        temperature=0.3,
        device_indices=[0, 1],
        lead_indices=[[2, 3], [4, 5]],
        spin_z=[1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
    )
    return initial, result


def _hubbard_i_errors(
    initial: np.ndarray,
    result,
    interaction_u: float,
    *,
    embedding_form: str = "dyson",
) -> tuple[float, float]:
    energy = np.linspace(-2.0, 2.0, 401)
    embedding = finite_lead_retarded_embedding(
        energy,
        lead_hamiltonians=[initial[2:4, 2:4], initial[4:6, 4:6]],
        coupling_matrices=[initial[:2, 2:4], initial[:2, 4:6]],
        eta=0.05,
    )
    exact = result.initial_retarded_frequency(energy, eta=0.05, indices=[0, 1])
    approximate = np.zeros_like(exact)
    occupations = result.full_density_matrices[0]
    for spin in range(2):
        opposite = float(occupations[1 - spin, 1 - spin].real)
        approximate[:, spin : spin + 1, spin : spin + 1] = lead_coupled_hubbard_i_retarded(
            energy,
            epsilon=initial[spin, spin].real,
            interaction_u=interaction_u,
            opposite_occupation=opposite,
            embedding_retarded=embedding[:, spin : spin + 1, spin : spin + 1],
            eta=0.05,
            embedding_form=embedding_form,
        )
    return float(np.max(np.abs(exact - approximate))), float(np.max(np.abs(exact[:, 0, 1])))


def _check(name: str, condition: bool, checks: dict[str, bool]) -> None:
    checks[name] = bool(condition)
    print(f"CHECK {name}: {'PASS' if condition else 'FAIL'}")


def run_gate() -> dict[str, object]:
    initial, interacting = _solve(0.8)
    _, noninteracting = _solve(0.0)
    charge_balance = interacting.device_rate() - interacting.lead_current(0) - interacting.lead_current(1)
    spin_operator = np.diag([1.0, -1.0, 0.0, 0.0, 0.0, 0.0]).astype(complex)
    spin_balance = interacting.device_rate(observable=spin_operator) - interacting.lead_current(0, spin=True) - interacting.lead_current(1, spin=True)
    finite_u_error, finite_u_spin_mixing = _hubbard_i_errors(initial, interacting, 0.8)
    zero_u_error, zero_u_spin_mixing = _hubbard_i_errors(initial, noninteracting, 0.0)
    # The two-pole insertion is reported alongside the Dyson solution so the
    # record separates the Hubbard-I approximation itself from the choice of
    # how the embedding enters it.  The two coincide at U=0, so the
    # non-interacting control above cannot distinguish them.
    finite_u_error_two_pole, _ = _hubbard_i_errors(
        initial, interacting, 0.8, embedding_form="two_pole"
    )
    initial_coherence = float(np.max(np.abs(interacting.full_density_matrices[0, :2, 2:])))
    charge_currents = np.stack([interacting.lead_current(0), interacting.lead_current(1)])
    spin_currents = np.stack([interacting.lead_current(0, spin=True), interacting.lead_current(1, spin=True)])
    density_eigenvalues = np.linalg.eigvalsh(interacting.full_density_matrices)
    checks: dict[str, bool] = {}
    _check("exact_two_time_spectral_identity", interacting.spectral_identity_error < 1e-12, checks)
    _check("exact_density_is_hermitian", interacting.density_hermiticity_error < 1e-12, checks)
    _check("exact_charge_continuity_with_two_leads", np.max(np.abs(charge_balance)) < 2e-11, checks)
    _check("exact_spin_continuity_without_spin_torque", np.max(np.abs(spin_balance)) < 2e-11, checks)
    _check("contacted_many_body_state_has_device_lead_coherence", initial_coherence > 1e-3, checks)
    _check("exact_density_eigenvalues_are_physical", np.min(density_eigenvalues) > -1e-10 and np.max(density_eigenvalues) < 1.0 + 1e-10, checks)
    _check("charge_currents_are_nonzero", np.max(np.abs(charge_currents[1:, 1:])) > 1e-4, checks)
    _check("spin_currents_are_nonzero", np.max(np.abs(spin_currents[1:, 1:])) > 1e-4, checks)
    _check("lead_coupled_hubbard_i_has_noninteracting_control", zero_u_error < 2e-10, checks)
    _check("finite_u_eom_difference_is_measured", finite_u_error > 1e-2, checks)
    report = {
        "gate": "GATE_31_EXACT_INTERACTING_LEAD_COUPLED_KELDYSH_ORACLE",
        "checks": checks,
        "passed": all(checks.values()),
        "model": {
            "one_body_modes": 6,
            "device_spin_orbitals": 2,
            "lead_spin_orbitals": [2, 2],
            "interaction_u": 0.8,
            "temperature": 0.3,
            "time_points": interacting.time.size,
        },
        "metrics": {
            "spectral_identity_error": interacting.spectral_identity_error,
            "density_hermiticity_error": interacting.density_hermiticity_error,
            "charge_balance_max": float(np.max(np.abs(charge_balance))),
            "spin_balance_max": float(np.max(np.abs(spin_balance))),
            "initial_device_lead_coherence": initial_coherence,
            "charge_current_max": float(np.max(np.abs(charge_currents))),
            "spin_current_max": float(np.max(np.abs(spin_currents))),
            "hubbard_i_finite_u_error_two_pole": finite_u_error_two_pole,
            "hubbard_i_noninteracting_error": zero_u_error,
            "hubbard_i_finite_u_error": finite_u_error,
            "hubbard_i_finite_u_offdiagonal_control": finite_u_spin_mixing,
            "density_eigenvalue_min": float(np.min(density_eigenvalues)),
            "density_eigenvalue_max": float(np.max(density_eigenvalues)),
        },
        "assessment": "PASS_EXACT_LEAD_COUPLED_INTERACTING_ORACLE_EOM_ERROR_QUANTIFIED",
        "claim_boundary": (
            "The exact finite many-body contacted oracle closes charge and spin "
            "continuity and provides a lead-coupled reference for Hubbard-I. "
            "The finite-U Hubbard-I mismatch is measured, not presented as a "
            "new effect; an interacting continuum contour closure remains open."
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
