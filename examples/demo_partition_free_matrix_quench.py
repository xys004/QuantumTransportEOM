"""Partition-free matrix wide-band voltage and device-Hamiltonian quench."""

from __future__ import annotations

import numpy as np

from quantum_transport import (
    partition_free_wide_band_matrix_quench,
    partition_free_wide_band_two_time_greens,
)


def main() -> None:
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
    energy = np.linspace(-30.0, 30.0, 8001)
    time = np.linspace(0.0, 4.0, 101)
    common = dict(
        energy=energy,
        initial_hamiltonian=h_initial,
        final_hamiltonian=h_final,
        lead_broadenings=np.stack([gamma_left, gamma_right]),
        bias_shift=np.array([0.4, -0.3]),
        temperature=0.1,
        max_memory_bytes=128 * 1024**2,
    )

    transient = partition_free_wide_band_matrix_quench(time, **common)
    derivative = np.gradient(
        transient.particle_number, time, edge_order=2
    )
    interior = (time > 0.2) & (time < 3.8)
    continuity_error = np.max(
        np.abs(
            derivative[interior]
            - transient.net_current_into_device[interior]
        )
    )
    selected = partition_free_wide_band_two_time_greens(
        time[[0, 10, 25, 50, 100]], **common
    )

    print("Partition-free matrix wide-band quench")
    print("initial lead currents:", transient.current_into_device[0])
    print("final lead currents:", transient.current_into_device[-1])
    print("particle-number range:", transient.particle_number.min(), transient.particle_number.max())
    print("continuity residual:", f"{continuity_error:.3e}")
    print("two-time shape:", selected.lesser.shape)
    print("two-time consistency:", selected.consistency_report().as_dict())
    print("VERDICT: PASS")


if __name__ == "__main__":
    main()
