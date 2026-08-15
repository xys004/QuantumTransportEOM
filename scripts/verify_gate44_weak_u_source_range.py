"""Gate 44: weak-to-intermediate-U source error range."""

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
    two_time_kbe_continuity_balance,
)


def _check(name: str, condition: bool, checks: dict[str, bool]) -> None:
    checks[name] = bool(condition)
    print(f"CHECK {name}: {'PASS' if condition else 'FAIL'}")


def run_gate() -> dict[str, object]:
    time = np.linspace(0.0, 0.5, 5)
    imaginary = np.linspace(0.0, 1.0 / 0.3, 21)
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
    interactions = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8]
    rows = []
    for interaction_u in interactions:
        exact = finite_interacting_partition_free_two_time(
            time,
            initial_one_body_hamiltonian=finite.initial_hamiltonian,
            final_one_body_hamiltonian=finite.final_hamiltonian,
            interactions=[] if interaction_u == 0.0 else [(0, 1, interaction_u)],
            temperature=0.3,
            device_indices=[0, 1],
            lead_indices=[[2, 3], [4, 5]],
            spin_z=[1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
            imaginary_time=imaginary,
        )
        balance = two_time_kbe_continuity_balance(
            time,
            green_retarded=exact.retarded,
            green_lesser=exact.lesser,
            hamiltonian=final_device,
            self_energy_retarded=finite.self_energy_retarded,
            self_energy_lesser=finite.self_energy_lesser,
            self_energy_advanced=finite.self_energy_advanced,
        )
        sigma = hubbard_second_born_self_energy_mixed(
            time,
            imaginary,
            green_rceil=exact.green_rceil,
            green_lceil=exact.green_mixed,
            interaction_u=interaction_u,
            spin_pairs=((0, 1), (1, 0)),
        )
        source = kbe_initial_correlation_kernel(time, imaginary, self_energy_mixed=sigma, green_mixed=exact.green_mixed)
        charge = np.eye(2, dtype=complex)
        spin = np.diag([1.0, -1.0]).astype(complex)
        project = lambda op, value: np.real(np.trace(op @ value, axis1=-2, axis2=-1))
        rows.append({
            "u": interaction_u,
            "required_charge_max": float(np.max(np.abs(project(charge, balance.residual)))),
            "required_spin_max": float(np.max(np.abs(project(spin, balance.residual)))),
            "source_charge_max": float(np.max(np.abs(project(charge, source.density_source)))),
            "source_spin_max": float(np.max(np.abs(project(spin, source.density_source)))),
            "source_hermiticity_error": source.hermiticity_error,
            "spectral_identity_error": exact.spectral_identity_error,
        })
    baseline = rows[0]
    for row in rows[1:]:
        row["charge_error_after_baseline"] = abs((row["required_charge_max"] - baseline["required_charge_max"]) - row["source_charge_max"])
        row["spin_error_after_baseline"] = abs((row["required_spin_max"] - baseline["required_spin_max"]) - row["source_spin_max"])
    checks: dict[str, bool] = {}
    _check("all_exact_oracles_are_spectral", all(row["spectral_identity_error"] < 1e-12 for row in rows), checks)
    _check("all_mixed_sources_are_hermitian", all(row["source_hermiticity_error"] < 1e-12 for row in rows), checks)
    _check("u0_source_is_zero", rows[0]["source_charge_max"] < 1e-15 and rows[0]["source_spin_max"] < 1e-15, checks)
    _check("finite_u_source_is_resolved", all(row["source_charge_max"] > 1e-4 for row in rows[1:]), checks)
    _check("charge_error_grows_into_intermediate_u", rows[-1]["charge_error_after_baseline"] > rows[1]["charge_error_after_baseline"] > 1e-3, checks)
    _check("spin_error_is_resolved_at_weak_u", rows[1]["spin_error_after_baseline"] > 1e-4, checks)
    report = {
        "gate": "GATE_44_WEAK_U_SOURCE_ERROR_RANGE",
        "checks": checks,
        "passed": all(checks.values()),
        "rows": rows,
        "assessment": "PASS_WEAK_U_RANGE_AUDIT_SECOND_BORN_ERROR_NOT_CLOSED",
        "claim_boundary": (
            "The exact finite-U source audit resolves a nonzero second-Born "
            "source at weak coupling, but its charge/spin error remains "
            "measurable and grows into intermediate U. No controlled accuracy "
            "window is claimed beyond this finite benchmark."
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
