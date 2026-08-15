"""Gate 16: analytic Lorentzian reservoir memory and smooth gauge dressing."""

from __future__ import annotations

import json

import numpy as np

from quantum_transport import lorentzian_reservoir_two_time


def run_gate() -> dict:
    time = np.array([0.0, 0.2, 0.4, 0.8, 1.4])
    gamma = np.array([[0.6]], dtype=complex)
    energy = np.linspace(-80.0, 80.0, 16001)
    base = lorentzian_reservoir_two_time(
        time,
        gamma,
        bandwidth=1.2,
        center=0.3,
        chemical_potential=0.0,
        temperature=0.1,
        energy_grid=energy,
    )
    lag = time[:, None] - time[None, :]
    causal = np.tril(np.ones_like(lag), k=-1) + 0.5 * np.eye(time.size)
    expected = -0.5j * 0.6 * 1.2 * causal * np.exp(-(1.2 + 0.3j) * np.maximum(lag, 0.0))
    analytic_error = float(np.max(np.abs(base.retarded[:, :, 0, 0] - expected)))

    phase = 0.15 * time**2
    dressed = lorentzian_reservoir_two_time(
        time,
        gamma,
        bandwidth=1.2,
        center=0.3,
        chemical_potential=0.0,
        temperature=0.1,
        energy_grid=energy,
        phase=phase,
    )
    factor = np.exp(-1j * (phase[:, None] - phase[None, :]))[:, :, None, None]
    phase_error = float(np.max(np.abs(dressed.retarded - base.retarded * factor)))
    lesser_phase_error = float(np.max(np.abs(dressed.lesser - base.lesser * factor)))
    advanced_error = float(np.max(np.abs(dressed.advanced - dressed.retarded.swapaxes(0, 1).swapaxes(-1, -2).conj())))
    checks = [
        {
            "name": "lorentzian_analytic_retarded_exponential",
            "passed": analytic_error < 2e-13 and base.retarded_causality_error == 0.0,
            "details": {
                "maximum_error": analytic_error,
                "memory_time": base.memory_time,
                "causality_error": base.retarded_causality_error,
            },
        },
        {
            "name": "smooth_gauge_phase_covariance",
            "passed": phase_error < 2e-13 and lesser_phase_error < 2e-11 and advanced_error < 2e-13,
            "details": {
                "retarded_phase_error": phase_error,
                "lesser_phase_error": lesser_phase_error,
                "advanced_adjoint_error": advanced_error,
            },
        },
    ]
    return {
        "gate": "GATE_16_ANALYTIC_RESERVOIR_MEMORY",
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
        "claim_scope": "analytic finite-band Lorentzian retarded memory and exact scalar gauge dressing for smooth bias/flux protocols",
        "not_yet_claimed": [
            "finite-temperature lesser kernel without a numerical energy quadrature",
            "full interacting transient solution on the driven Kane–Mele ring",
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

