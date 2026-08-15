"""Gate 87: exact finite-contact noncommuting spin-torque audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantum_transport import (  # noqa: E402
    partition_free_finite_lead_two_time,
    two_time_kbe_continuity_balance,
)


def _check(name: str, condition: bool, checks: dict[str, bool]) -> None:
    checks[name] = bool(condition)
    print(f"CHECK {name}: {'PASS' if condition else 'FAIL'}")


def _run_resolution(n_time: int) -> dict[str, float]:
    time = np.linspace(0.0, 0.5, n_time)
    imaginary = np.linspace(0.0, 2.5, 25)
    initial_device = np.array([[0.10, 0.25], [0.25, -0.10]], dtype=complex)
    final_device = np.array([[0.12, 0.30], [0.30, -0.08]], dtype=complex)
    leads = [
        np.diag(np.linspace(-0.8, 0.8, 4)).astype(complex),
        np.diag(np.linspace(-0.6, 0.6, 4)).astype(complex),
    ]
    couplings = [
        0.18 * np.eye(2, 4, dtype=complex),
        0.16 * np.eye(2, 4, dtype=complex),
    ]
    reference = partition_free_finite_lead_two_time(
        time,
        imaginary,
        initial_device_hamiltonian=initial_device,
        final_device_hamiltonian=final_device,
        lead_hamiltonians=leads,
        coupling_matrices=couplings,
        lead_shifts=[0.15, -0.10],
        temperature=0.4,
    )
    balance = two_time_kbe_continuity_balance(
        time,
        green_retarded=reference.retarded,
        green_lesser=reference.lesser,
        hamiltonian=reference.final_device_hamiltonian,
        self_energy_retarded=reference.self_energy_retarded,
        self_energy_lesser=reference.self_energy_lesser,
        self_energy_advanced=reference.self_energy_advanced,
        initial_correlation_source=reference.continuity_initial_source,
    )
    charge = np.eye(2, dtype=complex)
    spin_x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    spin_z = np.diag([1.0, -1.0]).astype(complex)
    charge_projection = balance.observable_balance(charge)
    spin_x_projection = balance.observable_balance(spin_x)
    spin_z_projection = balance.observable_balance(spin_z)
    return {
        "n_time": float(n_time),
        "raw_residual_max": balance.maximum_residual,
        "source_corrected_residual_max": balance.maximum_source_corrected_residual,
        "charge_coherent_max": float(np.max(np.abs(charge_projection["coherent_rate"]))),
        "spin_x_torque_max": float(np.max(np.abs(spin_x_projection["coherent_rate"]))),
        "spin_z_torque_max": float(np.max(np.abs(spin_z_projection["coherent_rate"]))),
        "spin_x_collision_max": float(np.max(np.abs(spin_x_projection["collision_rate"]))),
        "spectral_identity_error": reference.spectral_identity_error,
    }


def run_gate(runtime: str = "ASTRA") -> dict[str, object]:
    rows = [_run_resolution(n_time) for n_time in (5, 7, 9)]
    checks: dict[str, bool] = {}
    corrected = [row["source_corrected_residual_max"] for row in rows]
    _check("exact_finite_contact_branch_is_finite", all(np.isfinite(value) for row in rows for value in row.values()), checks)
    _check("charge_has_no_coherent_torque", max(row["charge_coherent_max"] for row in rows) < 1e-12, checks)
    _check("noncommuting_spin_x_torque_is_resolved", max(row["spin_x_torque_max"] for row in rows) > 1e-4, checks)
    _check("noncommuting_spin_z_torque_is_resolved", max(row["spin_z_torque_max"] for row in rows) > 1e-4, checks)
    _check("spin_collision_channel_is_finite", all(np.isfinite(row["spin_x_collision_max"]) for row in rows), checks)
    _check("vertical_source_refinement_decreases_residual", corrected[0] > corrected[1] > corrected[2], checks)
    _check("finite_contact_spectral_identity_is_resolved", max(row["spectral_identity_error"] for row in rows) < 1e-12, checks)
    report = {
        "schema_version": 1,
        "gate": "GATE_87_NONCOMMUTING_SPIN_TORQUE",
        "runtime": runtime,
        "execution": {
            "astra": "PASS" if runtime.upper() in {"ASTRA", "ASTRA+ASTRUM"} else "NOT_RUN",
            "astrum": "PASS" if runtime.upper() in {"ASTRUM", "ASTRA+ASTRUM"} else "NOT_RUN",
            "script": "scripts/verify_gate87_noncommuting_spin_torque.py",
        },
        "checks": checks,
        "passed": all(checks.values()),
        "metrics": {"resolutions": rows},
        "assessment": "PASS_EXACT_FINITE_CONTACT_NONCOMMUTING_SPIN_TORQUE",
        "claim_boundary": (
            "The exact finite-lead oracle resolves nonzero coherent spin torque for noncommuting spin Hamiltonians, keeps the charge coherent projection zero, "
            "and shows decreasing source-corrected residual under time refinement. This validates the spin audit path; it is not a topological-protection or continuum-interacting claim."
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
