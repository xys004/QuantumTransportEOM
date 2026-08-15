"""Gate 22: KBE collision, charge continuity, and spin projection diagnostics."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantum_transport import (  # noqa: E402
    equilibrium_one_body_density,
    kadanoff_baym_dyson_two_time,
    two_time_greens,
    two_time_kbe_collision_integral,
    two_time_kbe_continuity_balance,
    two_time_greens,
)


def _check(name: str, condition: bool, checks: dict[str, bool]) -> None:
    checks[name] = bool(condition)
    print(f"CHECK {name}: {'PASS' if condition else 'FAIL'}")


def run_gate() -> dict:
    time = np.linspace(0.0, 1.0, 33)
    hamiltonian = np.array([[0.2, 0.07j], [-0.07j, -0.1]], dtype=np.complex128)
    density = equilibrium_one_body_density(hamiltonian, mu=0.0, temperature=0.2)
    green = two_time_greens(time, lambda _t: hamiltonian, density)
    zero = np.zeros_like(green.retarded)
    collision = two_time_kbe_collision_integral(
        time,
        green_retarded=green.retarded,
        green_lesser=green.lesser,
        self_energy_retarded=zero,
        self_energy_lesser=zero,
    )
    balance = two_time_kbe_continuity_balance(
        time,
        green_retarded=green.retarded,
        green_lesser=green.lesser,
        hamiltonian=hamiltonian,
        self_energy_retarded=zero,
        self_energy_lesser=zero,
    )
    checks: dict[str, bool] = {}
    _check("zero_self_energy_collision", float(np.max(np.abs(collision))) < 2e-14, checks)
    _check("closed_charge_continuity", balance.maximum_residual < 2e-14, checks)
    sigma_z = np.diag([1.0, -1.0]).astype(np.complex128)
    spin = balance.observable_balance(sigma_z)
    _check("closed_spin_projection", float(np.max(np.abs(spin["residual"]))) < 2e-14, checks)
    _check("coherent_spin_torque_resolved", float(np.max(np.abs(spin["coherent_rate"]))) < 2e-14, checks)
    interacting_time = np.linspace(0.0, 2.0, 161)
    interacting_hamiltonian = np.array([[0.2]], dtype=np.complex128)
    interacting_density = equilibrium_one_body_density(interacting_hamiltonian, mu=0.0, temperature=0.2)
    bare = two_time_greens(interacting_time, lambda _t: interacting_hamiltonian, interacting_density)
    theta = np.tril(np.ones((interacting_time.size, interacting_time.size)), k=-1) + 0.5 * np.eye(interacting_time.size)
    sigma_r = -1j * 0.4 * theta[:, :, None, None]
    sigma_l = np.zeros_like(sigma_r)
    interacting = kadanoff_baym_dyson_two_time(
        interacting_time,
        bare_retarded=bare.retarded,
        bare_lesser=bare.lesser,
        self_energy_retarded=sigma_r,
        self_energy_lesser=sigma_l,
        max_iterations=100,
        mixing=0.7,
        tolerance=1e-11,
    )
    interacting_balance = two_time_kbe_continuity_balance(
        interacting_time,
        green_retarded=interacting.retarded,
        green_lesser=interacting.lesser,
        hamiltonian=interacting_hamiltonian,
        self_energy_retarded=sigma_r,
        self_energy_lesser=sigma_l,
    )
    interior_residual = float(np.max(np.abs(interacting_balance.residual[2:-2])))
    collision_signal = float(np.max(np.abs(interacting_balance.collision_rate[2:-2])))
    _check("nonzero_collision_is_resolved", interacting.converged and collision_signal > 1e-2 and interior_residual < 2e-4, checks)
    return {
        "gate": "GATE_22_CONTINUITY_DIAGNOSTICS",
        "scope": "closed finite quadratic KBE identity with explicit collision and Hermitian charge/spin projections",
        "checks": checks,
        "metrics": {
            "collision_max": float(np.max(np.abs(collision))),
            "charge_residual_max": balance.maximum_residual,
            "spin_residual_max": float(np.max(np.abs(spin["residual"]))),
            "spin_coherent_rate_max": float(np.max(np.abs(spin["coherent_rate"]))),
            "nonzero_collision_signal": collision_signal,
            "nonzero_collision_interior_residual": interior_residual,
        },
        "claim_boundary": "This gate validates the bookkeeping identity and closed limit; lead-coupled interacting continuity with reservoir injection and Rashba torque remains an application-level publication gate.",
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
