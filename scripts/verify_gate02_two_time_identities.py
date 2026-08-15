"""Gate 2: causal, equal-time, spectral, gauge, and continuity identities."""

from __future__ import annotations

import json

import numpy as np

from quantum_transport import (
    equilibrium_one_body_density,
    one_body_bond_current,
    propagate_density_matrix,
    region_interface_current,
    two_time_greens,
)


def _scalar_identities() -> tuple[bool, dict]:
    time = np.linspace(0.0, 1.2, 7)
    epsilon = 0.37
    occupation = 0.31
    result = two_time_greens(
        time,
        lambda _time: np.array([[epsilon]], dtype=np.complex128),
        np.array([[occupation]], dtype=np.complex128),
    )
    lag = time[:, None] - time[None, :]
    phase = np.exp(-1j * epsilon * lag)
    lesser_expected = 1j * occupation * phase
    greater_expected = -1j * (1.0 - occupation) * phase
    theta = np.tril(np.ones_like(lag), k=-1) + 0.5 * np.eye(time.size)
    retarded_expected = -1j * theta * phase
    advanced_expected = retarded_expected.conj().T
    future_retarded = np.triu(result.retarded[:, :, 0, 0], k=1)
    future_advanced = np.tril(result.advanced[:, :, 0, 0], k=-1)
    equal_time_density = -1j * np.diagonal(result.lesser[:, :, 0, 0])
    errors = {
        "lesser": float(np.max(np.abs(result.lesser[:, :, 0, 0] - lesser_expected))),
        "greater": float(np.max(np.abs(result.greater[:, :, 0, 0] - greater_expected))),
        "retarded": float(np.max(np.abs(result.retarded[:, :, 0, 0] - retarded_expected))),
        "advanced": float(np.max(np.abs(result.advanced[:, :, 0, 0] - advanced_expected))),
        "retarded_causality": float(np.max(np.abs(future_retarded))),
        "advanced_causality": float(np.max(np.abs(future_advanced))),
        "equal_time_density": float(np.max(np.abs(equal_time_density - occupation))),
        "spectral_identity": float(result.spectral_identity_error()),
    }
    return max(errors.values()) < 1e-13, errors


def _noncommuting_equal_time() -> tuple[bool, dict]:
    time = np.linspace(0.0, 1.5, 31)
    h0 = np.array([[0.2, -0.7], [-0.7, -0.1]], dtype=np.complex128)
    density = equilibrium_one_body_density(h0, mu=0.0, temperature=0.15)

    def hamiltonian(value: float) -> np.ndarray:
        phase = 0.4 * np.sin(value)
        return np.array(
            [
                [0.2, -0.7 * np.exp(1j * phase)],
                [-0.7 * np.exp(-1j * phase), -0.1],
            ],
            dtype=np.complex128,
        )

    result = two_time_greens(
        time,
        hamiltonian,
        density,
        components=("lesser", "greater"),
    )
    propagated = propagate_density_matrix(density, time, hamiltonian)
    density_error = float(np.max(np.abs(result.density_matrices() - propagated)))
    hermiticity_error = float(
        max(np.linalg.norm(value - value.conj().T) for value in propagated)
    )
    trace_drift = float(
        np.ptp(np.trace(propagated, axis1=1, axis2=2).real)
    )
    errors = {
        "equal_time_density": density_error,
        "density_hermiticity": hermiticity_error,
        "trace_drift": trace_drift,
    }
    return max(errors.values()) < 1e-12, errors


def _temporal_gauge_covariance() -> tuple[bool, dict]:
    hopping = 0.9
    charge_profile = np.diag([0.0, 0.37])
    time = np.linspace(0.0, 1.2, 241)

    def phi(value: float) -> float:
        return 0.6 * np.sin(0.8 * value)

    def phi_rate(value: float) -> float:
        return 0.48 * np.cos(0.8 * value)

    reference = np.array([[0.2, -hopping], [-hopping, -0.1]], dtype=np.complex128)
    density = equilibrium_one_body_density(reference, mu=0.0, temperature=0.2)

    def unitary(value: float) -> np.ndarray:
        return np.diag(np.exp(1j * np.diag(charge_profile) * phi(value)))

    def transformed(value: float) -> np.ndarray:
        current = unitary(value)
        return (
            current @ reference @ current.conj().T
            - phi_rate(value) * charge_profile
        )

    reference_density = propagate_density_matrix(
        density, time, lambda _time: reference
    )
    transformed_density = propagate_density_matrix(
        unitary(time[0]) @ density @ unitary(time[0]).conj().T,
        time,
        transformed,
    )
    expected = np.array(
        [
            unitary(value) @ reference_density[index] @ unitary(value).conj().T
            for index, value in enumerate(time)
        ]
    )
    error = float(np.max(np.linalg.norm(transformed_density - expected, axis=(1, 2))))
    return error < 3e-6, {"maximum_gauge_covariance_error": error}


def _finite_embedding_continuity() -> tuple[bool, dict]:
    time = np.linspace(0.0, 1.0, 41)

    def hamiltonian(value: float) -> np.ndarray:
        center = 0.4 * np.tanh(2.0 * (value - 0.5))
        return np.array(
            [[-0.3, -0.55, 0.0], [-0.55, center, -0.42], [0.0, -0.42, 0.25]],
            dtype=np.complex128,
        )

    density = equilibrium_one_body_density(
        hamiltonian(time[0]), mu=0.0, temperature=0.18
    )
    propagated = propagate_density_matrix(density, time, hamiltonian)
    residuals = []
    for value, rho in zip(time, propagated):
        matrix = hamiltonian(value)
        derivative = -1j * (matrix @ rho - rho @ matrix)
        central_rate = float(np.real(derivative[1, 1]))
        outgoing = region_interface_current(matrix, rho, [1])
        direct = one_body_bond_current(matrix, rho, 1, 0) + one_body_bond_current(
            matrix, rho, 1, 2
        )
        residuals.extend([abs(outgoing - direct), abs(central_rate + outgoing)])
    error = float(max(residuals))
    return error < 1e-12, {"maximum_continuity_error": error}


def run_gate() -> dict:
    checks = []
    for name, function in (
        ("two_time_scalar_identities", _scalar_identities),
        ("noncommuting_equal_time_density", _noncommuting_equal_time),
        ("temporal_gauge_covariance", _temporal_gauge_covariance),
        ("finite_embedding_continuity", _finite_embedding_continuity),
    ):
        passed, details = function()
        checks.append({"name": name, "passed": passed, "details": details})
    return {
        "gate": "GATE_02_TWO_TIME_IDENTITIES",
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
        "claim_scope": "finite quadratic systems and the tested temporal-gauge protocol",
        "not_yet_claimed": [
            "arbitrary-time interacting self-energies",
            "smooth reservoir memory beyond the implemented protocols",
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
