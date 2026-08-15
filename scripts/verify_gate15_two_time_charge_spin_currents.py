"""Gate 15: charge and spin Meir–Wingreen currents from two-time kernels."""

from __future__ import annotations

import json

import numpy as np

from quantum_transport import (
    equilibrium_one_body_density,
    electron_boson_scba_self_energy_two_time,
    self_consistent_born_two_time,
    two_time_meir_wingreen_current,
    two_time_spin_meir_wingreen_current,
    two_time_greens,
)


def _spin_selective_kernels():
    time = np.array([0.0, 0.2, 0.7, 1.0])
    n = time.size
    retarded = np.zeros((n, n, 2, 2), dtype=complex)
    lesser = np.zeros_like(retarded)
    sigma_lesser = np.zeros_like(retarded)
    sigma_advanced = np.zeros_like(retarded)
    for i in range(n):
        for j in range(n):
            retarded[i, j] = np.diag([0.2 + 0.1j, 0.0])
            lesser[i, j] = np.diag([0.15j, 0.0])
            sigma_lesser[i, j] = np.diag([0.25j, 0.0])
            sigma_advanced[i, j] = np.diag([0.1j, 0.0])
    return time, retarded, lesser, sigma_lesser, sigma_advanced


def run_gate() -> dict:
    time, retarded, lesser, sigma_lesser, sigma_advanced = _spin_selective_kernels()
    zero = np.zeros_like(retarded)
    zero_current = two_time_meir_wingreen_current(
        time,
        green_retarded=retarded,
        green_lesser=lesser,
        lead_self_energy_lesser=zero,
        lead_self_energy_advanced=zero,
    )
    charge = two_time_meir_wingreen_current(
        time,
        green_retarded=retarded,
        green_lesser=lesser,
        lead_self_energy_lesser=sigma_lesser,
        lead_self_energy_advanced=sigma_advanced,
    )
    spin = two_time_spin_meir_wingreen_current(
        time,
        green_retarded=retarded,
        green_lesser=lesser,
        lead_self_energy_lesser=sigma_lesser,
        lead_self_energy_advanced=sigma_advanced,
        spin_operator=np.diag([0.5, -0.5]),
    )

    scalar_h = np.array([[0.2]], dtype=complex)
    scalar_density = equilibrium_one_body_density(scalar_h, mu=0.0, temperature=0.1)
    scalar_time = np.linspace(0.0, 1.0, 13)
    bare = two_time_greens(scalar_time, lambda _: scalar_h, scalar_density)
    interacting = self_consistent_born_two_time(
        scalar_time,
        bare_retarded=bare.retarded,
        bare_lesser=bare.lesser,
        coupling=np.array([[0.02]], dtype=complex),
        boson_frequency=0.4,
        boson_temperature=0.1,
        max_iterations=30,
        dyson_iterations=60,
        mixing=0.35,
        tolerance=1e-8,
    )
    interaction_r, interaction_a, interaction_l, _ = electron_boson_scba_self_energy_two_time(
        scalar_time,
        interacting.lesser,
        interacting.greater,
        coupling=np.array([[0.02]], dtype=complex),
        boson_frequency=0.4,
        boson_temperature=0.1,
    )
    interaction_current = two_time_meir_wingreen_current(
        scalar_time,
        green_retarded=interacting.retarded,
        green_lesser=interacting.lesser,
        lead_self_energy_lesser=interaction_l,
        lead_self_energy_advanced=interaction_a,
    )
    checks = [
        {
            "name": "zero_injection_current",
            "passed": bool(np.max(np.abs(zero_current)) < 1e-14),
            "details": {"maximum_current": float(np.max(np.abs(zero_current)))},
        },
        {
            "name": "spin_charge_projection",
            "passed": bool(np.max(np.abs(spin - 0.5 * charge)) < 1e-14),
            "details": {
                "charge_current": charge.tolist(),
                "spin_current": spin.tolist(),
            },
        },
        {
            "name": "interacting_two_time_current_pipeline",
            "passed": bool(interacting.converged and np.all(np.isfinite(interaction_current))),
            "details": {
                "scba_iterations": interacting.iterations,
                "current_min": float(np.min(interaction_current)),
                "current_max": float(np.max(interaction_current)),
            },
        },
    ]
    return {
        "gate": "GATE_15_TWO_TIME_CHARGE_SPIN_CURRENTS",
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
        "claim_scope": "finite-memory Meir–Wingreen charge and Hermitian-observable spin currents from two-time Keldysh kernels",
        "not_yet_claimed": [
            "spin-polarized reservoir self-energy fitting to a material lead",
            "interacting Kane–Mele/Corbino edge-current protection",
            "continuity closure including all lead and interaction torque channels",
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
