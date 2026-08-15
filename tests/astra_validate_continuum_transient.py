"""ASTRA gate for the partition-free matrix wide-band transient."""

from __future__ import annotations

import json
import sys

import numpy as np


if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz
if sys.platform == "win32":
    sys.path.insert(0, r"C:\Users\Nelson\Dev\quantum\QuantumTransportEOM\src")

from quantum_transport import (  # noqa: E402
    LeadSelfEnergy,
    MatrixTransportView,
    partition_free_resonant_level_bias_quench,
    partition_free_wide_band_matrix_quench,
    partition_free_wide_band_two_time_greens,
    stationary_greens_two_time,
)


h_initial = np.array(
    [[0.20, 0.12 - 0.03j], [0.12 + 0.03j, -0.15]],
    dtype=np.complex128,
)
h_final = np.array(
    [[0.25, 0.08 + 0.04j], [0.08 - 0.04j, -0.10]],
    dtype=np.complex128,
)
gamma_left = np.array(
    [[0.35, 0.04j], [-0.04j, 0.18]], dtype=np.complex128
)
gamma_right = np.array(
    [[0.22, -0.03], [-0.03, 0.31]], dtype=np.complex128
)
gammas = np.stack([gamma_left, gamma_right])
shifts = np.array([0.4, -0.3])
temperature = 0.1

# Independent existing scalar oracle.
scalar_time = np.linspace(0.0, 2.0, 21)
scalar_energy = np.linspace(-30.0, 30.0, 8001)
scalar_gamma = np.array([0.3, 0.2])
scalar_shift = np.array([0.5, -0.5])
scalar_oracle = partition_free_resonant_level_bias_quench(
    scalar_time,
    scalar_energy,
    level_energy=0.15,
    broadening=scalar_gamma,
    bias_shift=scalar_shift,
    temperature=0.08,
)
scalar_matrix = partition_free_wide_band_matrix_quench(
    scalar_time,
    scalar_energy,
    initial_hamiltonian=np.array([[0.15]]),
    lead_broadenings=scalar_gamma[:, None, None],
    bias_shift=scalar_shift,
    temperature=0.08,
    max_memory_bytes=64 * 1024**2,
)
scalar_density_error = float(
    np.max(
        np.abs(
            scalar_matrix.density_matrix[:, 0, 0].real
            - scalar_oracle.occupation
        )
    )
)
scalar_current_error = float(
    np.max(
        np.abs(
            scalar_matrix.current_into_device
            - scalar_oracle.current_into_level
        )
    )
)

# Matrix continuity under simultaneous Hamiltonian and bias steps.
continuity_errors = []
initial_current = 0.0
density_eigenvalue_min = 1.0
density_eigenvalue_max = 0.0
energy = np.linspace(-30.0, 30.0, 8001)
for step in (0.04, 0.02):
    time = np.arange(0.0, 1.2 + 0.5 * step, step)
    result = partition_free_wide_band_matrix_quench(
        time,
        energy,
        initial_hamiltonian=h_initial,
        final_hamiltonian=h_final,
        lead_broadenings=gammas,
        bias_shift=shifts,
        temperature=temperature,
        max_memory_bytes=64 * 1024**2,
    )
    derivative = np.gradient(result.particle_number, time, edge_order=2)
    interior = (time > 0.16) & (time < 1.1)
    continuity_errors.append(
        float(
            np.max(
                np.abs(
                    derivative[interior]
                    - result.net_current_into_device[interior]
                )
            )
        )
    )
    initial_current = max(
        initial_current, float(np.max(np.abs(result.current_into_device[0])))
    )
    eigenvalues = np.linalg.eigvalsh(result.density_matrix).real
    density_eigenvalue_min = min(density_eigenvalue_min, float(eigenvalues.min()))
    density_eigenvalue_max = max(density_eigenvalue_max, float(eigenvalues.max()))
continuity_ratio = continuity_errors[0] / continuity_errors[1]

# Final stationary endpoint.
long_energy = np.linspace(-50.0, 50.0, 16001)
long_time = partition_free_wide_band_matrix_quench(
    np.array([0.0, 25.0]),
    long_energy,
    initial_hamiltonian=h_initial,
    final_hamiltonian=h_final,
    lead_broadenings=gammas,
    bias_shift=shifts,
    temperature=temperature,
    max_memory_bytes=64 * 1024**2,
)
stationary_view = MatrixTransportView(
    h_final,
    ["a", "b"],
    LeadSelfEnergy.wide_band(
        gamma_left, mu=shifts[0], temperature=temperature
    ),
    LeadSelfEnergy.wide_band(
        gamma_right, mu=shifts[1], temperature=temperature
    ),
)
stationary_density = (
    -1j
    * np.trapezoid(
        stationary_view.lesser_values(long_energy), long_energy, axis=0
    )
    / (2.0 * np.pi)
)
stationary_currents = np.array(
    [
        stationary_view.current_from_keldysh(long_energy, lead="left"),
        stationary_view.current_from_keldysh(long_energy, lead="right"),
    ]
)
stationary_density_error = float(
    np.max(np.abs(long_time.density_matrix[-1] - stationary_density))
)
stationary_current_error = float(
    np.max(np.abs(long_time.current_into_device[-1] - stationary_currents))
)

# Full two-time equilibrium cross-check against the independent stationary path.
two_time_grid = np.array([0.0, 0.2, 0.5, 1.0])
two_time_energy = np.linspace(-80.0, 80.0, 16001)
single_view = MatrixTransportView(
    np.array([[0.15]], dtype=np.complex128),
    ["level"],
    LeadSelfEnergy.wide_band(np.array([[0.3]]), temperature=0.08),
    LeadSelfEnergy.wide_band(np.array([[0.2]]), temperature=0.08),
)
stationary_two_time = stationary_greens_two_time(
    single_view,
    two_time_grid,
    two_time_energy,
    max_memory_bytes=64 * 1024**2,
)
transient_two_time = partition_free_wide_band_two_time_greens(
    two_time_grid,
    two_time_energy,
    initial_hamiltonian=np.array([[0.15]]),
    lead_broadenings=np.array([0.3, 0.2])[:, None, None],
    bias_shift=np.zeros(2),
    temperature=0.08,
    max_memory_bytes=64 * 1024**2,
)
two_time_lesser_error = float(
    np.max(np.abs(transient_two_time.lesser - stationary_two_time.lesser))
)
two_time_greater_error = float(
    np.max(np.abs(transient_two_time.greater - stationary_two_time.greater))
)
two_time_spectral_cutoff_error = transient_two_time.consistency_report().maximum

metrics = {
    "scalar_density_error": scalar_density_error,
    "scalar_current_error": scalar_current_error,
    "continuity_errors_dt_004_002": continuity_errors,
    "continuity_convergence_ratio": continuity_ratio,
    "initial_current_maximum": initial_current,
    "density_eigenvalue_minimum": density_eigenvalue_min,
    "density_eigenvalue_maximum": density_eigenvalue_max,
    "stationary_density_error_t25": stationary_density_error,
    "stationary_current_error_t25": stationary_current_error,
    "two_time_lesser_stationary_error": two_time_lesser_error,
    "two_time_greater_stationary_error": two_time_greater_error,
    "two_time_spectral_cutoff_error": two_time_spectral_cutoff_error,
}
checks = {
    "scalar_oracle_density": bool(scalar_density_error < 3e-14),
    "scalar_oracle_currents": bool(scalar_current_error < 3e-14),
    "matrix_continuity_second_order": bool(
        continuity_errors[1] < 3.2e-6 and continuity_ratio > 3.7
    ),
    "partition_free_initial_current_zero": bool(initial_current < 3e-14),
    "density_matrix_is_fermion_physical": bool(
        density_eigenvalue_min >= -1e-10
        and density_eigenvalue_max <= 1.0 + 1e-10
    ),
    "long_time_stationary_density": bool(stationary_density_error < 7e-6),
    "long_time_stationary_currents": bool(stationary_current_error < 7e-6),
    "two_time_equilibrium_matches_stationary": bool(
        two_time_lesser_error < 8e-15
        and two_time_greater_error < 8e-15
        and two_time_spectral_cutoff_error < 2.1e-3
    ),
}

print(
    json.dumps(
        {
            "claim": (
                "the partition-free matrix WBL quench preserves contacted "
                "initial equilibrium, continuity, the stationary endpoint, "
                "and the scalar/two-time exact limits"
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
