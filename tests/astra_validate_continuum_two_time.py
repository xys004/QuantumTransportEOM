"""ASTRA gate for stationary continuum two-time NEGF."""

from __future__ import annotations

import json
import sys

import numpy as np


# ASTRA executes a copy from its own workspace.  The explicit canonical source
# path keeps that audit tied to the implementation under review; PYTHONPATH is
# used naturally on ASTRUM and in ordinary repository runs.
if sys.platform == "win32":
    sys.path.insert(0, r"C:\Users\Nelson\Dev\quantum\QuantumTransportEOM\src")

from quantum_transport import (  # noqa: E402
    LeadSelfEnergy,
    MatrixTransportView,
    stationary_greens_two_time,
    stationary_self_energy_two_time,
)


integrate = getattr(np, "trapezoid", None)
if integrate is None:
    integrate = np.trapz
epsilon = 0.23
gamma_l = np.array([[0.35, 0.04j], [-0.04j, 0.28]], dtype=np.complex128)
gamma_r = np.array([[0.25, -0.02], [-0.02, 0.32]], dtype=np.complex128)
hamiltonian = np.array(
    [[epsilon, 0.08 - 0.03j], [0.08 + 0.03j, -0.17]],
    dtype=np.complex128,
)
left = LeadSelfEnergy.wide_band(
    gamma_l, mu=0.27, temperature=0.11, name="left"
)
right = LeadSelfEnergy.wide_band(
    gamma_r, mu=-0.19, temperature=0.11, name="right"
)
transport = MatrixTransportView(hamiltonian, ["a", "b"], left, right)
omega = np.linspace(-50.0, 50.0, 20001)
time = np.array([0.0, 0.13, 0.37, 0.82, 1.41])

green = stationary_greens_two_time(transport, time, omega)
sigma = stationary_self_energy_two_time(left, time, omega)
green_report = green.consistency_report()
sigma_report = sigma.consistency_report()
rho = green.density_matrices()
rho_frequency = (
    -1j
    * integrate(transport.lesser_values(omega), omega, axis=0)
    / (2.0 * np.pi)
)
rho_match = float(np.max(np.abs(rho - rho_frequency[None, :, :])))
rho_hermiticity = float(
    np.max(np.abs(rho - rho.swapaxes(-1, -2).conj()))
)
rho_eigenvalues = np.linalg.eigvalsh(rho[0]).real
stationary_drift = green.equal_time_drift()

# Independent scalar WBL oracle away from the tau=0 retarded jump.
scalar_gamma = 0.6
scalar = MatrixTransportView(
    np.array([[epsilon]], dtype=np.complex128),
    ["level"],
    LeadSelfEnergy.wide_band(np.array([[0.35]]), temperature=0.11),
    LeadSelfEnergy.wide_band(np.array([[0.25]]), temperature=0.11),
)
omega_scalar = np.linspace(-120.0, 120.0, 24001)
time_scalar = np.array([0.0, 0.2, 0.5, 1.0, 1.8])
scalar_green = stationary_greens_two_time(
    scalar, time_scalar, omega_scalar
)
lag = time_scalar[:, None] - time_scalar[None, :]
positive = lag >= 0.2
negative = lag <= -0.2
oracle = -1j * np.exp(
    (-1j * epsilon - 0.5 * scalar_gamma) * lag
)
retarded_oracle_error = float(
    np.max(
        np.abs(
            scalar_green.retarded[:, :, 0, 0][positive]
            - oracle[positive]
        )
    )
)
acausal_tail = float(
    np.max(np.abs(scalar_green.retarded[:, :, 0, 0][negative]))
)

metrics = {
    "green_consistency": green_report.as_dict(),
    "self_energy_consistency": sigma_report.as_dict(),
    "equal_time_density_frequency_error": rho_match,
    "equal_time_density_hermiticity_error": rho_hermiticity,
    "equal_time_density_drift": stationary_drift,
    "density_eigenvalue_min": float(rho_eigenvalues.min()),
    "density_eigenvalue_max": float(rho_eigenvalues.max()),
    "scalar_retarded_oracle_error": retarded_oracle_error,
    "scalar_acausal_tail": acausal_tail,
}
checks = {
    "green_keldysh_identities": bool(green_report.maximum < 5e-13),
    "self_energy_keldysh_identities": bool(sigma_report.maximum < 5e-13),
    "equal_time_matches_frequency": bool(rho_match < 5e-13),
    "density_is_hermitian": bool(rho_hermiticity < 5e-13),
    "stationary_density_has_no_drift": bool(stationary_drift < 5e-13),
    "density_is_fermion_physical": bool(
        rho_eigenvalues.min() >= -1e-10
        and rho_eigenvalues.max() <= 1.0 + 1e-10
    ),
    "scalar_retarded_matches_exact_wbl": bool(
        retarded_oracle_error < 6e-3
    ),
    "negative_time_retarded_tail_is_cutoff_small": bool(
        acausal_tail < 6e-3
    ),
}

print(
    json.dumps(
        {
            "claim": (
                "stationary continuum two-time matrix NEGF is internally "
                "and analytically validated"
            ),
            "metrics": metrics,
            "checks": checks,
        },
        indent=2,
    )
)
for name, passed in checks.items():
    print(f"CHECK {name}: {'PASS' if passed else 'FAIL'}")
print("VERDICT: PASS" if all(checks.values()) else "VERDICT: FAIL")
