"""Gate 64: exact finite-contact charge/spin oracle and approximate comparison."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantum_transport import (  # noqa: E402
    finite_interacting_partition_free_two_time,
    partition_free_finite_lead_two_time,
    self_consistent_hubbard_second_born_contour_two_time,
    two_time_meir_wingreen_charge_spin_currents,
)


def _check(name: str, condition: bool, checks: dict[str, bool]) -> None:
    checks[name] = bool(condition)
    print(f"CHECK {name}: {'PASS' if condition else 'FAIL'}")


def run_gate() -> dict[str, object]:
    time = np.linspace(0.0, 0.5, 9)
    imaginary = np.linspace(0.0, 1.0 / 0.3, 21)
    initial_device = np.diag([-0.25, -0.18]).astype(complex)
    final_device = np.diag([0.08, -0.02]).astype(complex)
    lead_hamiltonians = [np.diag([-0.8, -0.65]).astype(complex), np.diag([0.5, 0.62]).astype(complex)]
    couplings = [np.diag([0.25, 0.25]).astype(complex), np.diag([0.2, 0.2]).astype(complex)]
    shifts = [0.15, -0.12]
    reference = partition_free_finite_lead_two_time(
        time,
        imaginary,
        initial_device_hamiltonian=initial_device,
        final_device_hamiltonian=final_device,
        lead_hamiltonians=lead_hamiltonians,
        coupling_matrices=couplings,
        lead_shifts=shifts,
        temperature=0.3,
    )
    exact = finite_interacting_partition_free_two_time(
        time,
        initial_one_body_hamiltonian=reference.initial_hamiltonian,
        final_one_body_hamiltonian=reference.final_hamiltonian,
        interactions=[(0, 1, 0.5)],
        temperature=0.3,
        device_indices=[0, 1],
        lead_indices=[[2, 3], [4, 5]],
        spin_z=[1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
        imaginary_time=imaginary,
    )
    approximate = self_consistent_hubbard_second_born_contour_two_time(
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
        include_full_contour_lesser=True,
        self_energy_matsubara=np.zeros_like(reference.green_matsubara),
    )
    spin_operator = np.diag([1.0, -1.0]).astype(complex)
    exact_charge_error = exact.device_rate() - exact.lead_current(0) - exact.lead_current(1)
    exact_spin_error = (
        exact.device_rate(observable=np.diag([1.0, -1.0, 0.0, 0.0, 0.0, 0.0]).astype(complex))
        - exact.lead_current(0, spin=True)
        - exact.lead_current(1, spin=True)
    )
    approximate_channels: dict[str, list[np.ndarray]] = {"charge": [], "sz": []}
    for lead_hamiltonian, coupling, shift in zip(lead_hamiltonians, couplings, shifts):
        lead_reference = partition_free_finite_lead_two_time(
            time,
            imaginary,
            initial_device_hamiltonian=initial_device,
            final_device_hamiltonian=final_device,
            lead_hamiltonians=[lead_hamiltonian],
            coupling_matrices=[coupling],
            lead_shifts=[shift],
            temperature=0.3,
        )
        channels = two_time_meir_wingreen_charge_spin_currents(
            time,
            green_retarded=approximate.retarded,
            green_lesser=approximate.lesser,
            lead_self_energy_lesser=lead_reference.self_energy_lesser,
            lead_self_energy_advanced=lead_reference.self_energy_advanced,
            spin_operators={"sz": spin_operator},
        )
        for name in approximate_channels:
            approximate_channels[name].append(channels[name])
    approximate_charge = sum(approximate_channels["charge"])
    approximate_spin = sum(approximate_channels["sz"])
    exact_charge = exact.lead_current(0) + exact.lead_current(1)
    exact_spin = exact.lead_current(0, spin=True) + exact.lead_current(1, spin=True)
    checks: dict[str, bool] = {}
    _check("exact_charge_continuity_oracle", np.max(np.abs(exact_charge_error)) < 2e-11, checks)
    _check("exact_spin_continuity_oracle", np.max(np.abs(exact_spin_error)) < 2e-11, checks)
    _check("exact_spin_current_is_resolved", np.max(np.abs(exact_spin)) > 1e-5, checks)
    _check("approximate_full_branch_converges", approximate.converged, checks)
    _check("approximate_charge_spin_channels_are_finite", np.all(np.isfinite(approximate_charge)) and np.all(np.isfinite(approximate_spin)), checks)
    _check("approximate_spin_channel_is_resolved", np.max(np.abs(approximate_spin)) > 1e-8, checks)
    report = {
        "gate": "GATE_64_EXACT_CONTACT_CHARGE_SPIN",
        "checks": checks,
        "passed": all(checks.values()),
        "metrics": {
            "exact_charge_continuity_error": float(np.max(np.abs(exact_charge_error))),
            "exact_spin_continuity_error": float(np.max(np.abs(exact_spin_error))),
            "exact_net_charge_max": float(np.max(np.abs(exact_charge))),
            "exact_net_spin_max": float(np.max(np.abs(exact_spin))),
            "approximate_net_charge_max": float(np.max(np.abs(approximate_charge))),
            "approximate_net_spin_max": float(np.max(np.abs(approximate_spin))),
            "approximate_charge_error_vs_exact_max": float(np.max(np.abs(approximate_charge - exact_charge))),
            "approximate_spin_error_vs_exact_max": float(np.max(np.abs(approximate_spin - exact_spin))),
            "approximate_lesser_contour_antihermiticity_error": approximate.lesser_contour_correction.antihermiticity_error if approximate.lesser_contour_correction is not None else None,
        },
        "assessment": "PASS_EXACT_CHARGE_SPIN_ORACLE_APPROXIMATE_COMPARISON",
        "claim_boundary": (
            "The exact finite contacted many-body oracle conserves device charge and the selected spin component. "
            "The interacting EOM/Keldysh branch exposes finite charge and spin reservoir channels and its discrepancy is recorded; "
            "the comparison is not a conserving-closure or topological-protection claim."
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
