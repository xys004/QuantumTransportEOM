"""Gate 11: stationary interacting SCBA inventory and two-time export."""

from __future__ import annotations

import json

import numpy as np

from quantum_transport import (
    ScbaResult,
    scba_two_time_greens,
    self_consistent_born_electron_boson,
)


def _solve(mu_left: float = 0.0, mu_right: float = 0.0):
    return self_consistent_born_electron_boson(
        np.array([[0.1]], dtype=np.complex128),
        np.linspace(-3.0, 3.0, 121),
        np.array([[[0.6]], [[0.4]]], dtype=np.complex128),
        [mu_left, mu_right],
        coupling=np.array([[0.08]], dtype=np.complex128),
        boson_frequency=0.5,
        temperature=0.2,
        boson_temperature=0.2,
        max_iterations=60,
        mixing=0.5,
        tolerance=1e-9,
    )


def run_gate() -> dict:
    equilibrium = _solve()
    biased = _solve(0.3, -0.3)
    two_time = scba_two_time_greens(equilibrium, np.linspace(0.0, 1.0, 5))
    checks = [
        {
            "name": "public_scba_result_contract",
            "passed": isinstance(equilibrium, ScbaResult)
            and equilibrium.converged
            and equilibrium.iterations > 1,
            "details": {
                "iterations": equilibrium.iterations,
                "maximum_update": equilibrium.maximum_update,
            },
        },
        {
            "name": "equilibrium_fdt_and_spectral_identity",
            "passed": equilibrium.fdt_error() < 2e-12
            and equilibrium.spectral_identity_error < 2e-12,
            "details": {
                "fdt_error": equilibrium.fdt_error(),
                "spectral_identity_error": equilibrium.spectral_identity_error,
            },
        },
        {
            "name": "nonequilibrium_terminal_conservation",
            "passed": biased.lead_currents[0] > 0.0
            and biased.lead_currents[1] < 0.0
            and biased.current_conservation_error < 2e-10,
            "details": {
                "lead_currents": biased.lead_currents.tolist(),
                "conservation_error": biased.current_conservation_error,
            },
        },
        {
            "name": "stationary_scba_two_time_export",
            "passed": two_time.consistency_report().maximum < 2e-12,
            "details": two_time.consistency_report().as_dict(),
        },
    ]
    return {
        "gate": "GATE_11_SCBA_INTERACTING_INVENTORY",
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
        "claim_scope": "stationary matrix electron-boson SCBA plus explicit two-time Fourier export",
        "not_yet_claimed": [
            "time-dependent Kadanoff-Baym propagation with self-consistent memory",
            "interacting Kane–Mele/Corbino transient protection",
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
