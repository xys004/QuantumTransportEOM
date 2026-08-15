"""Exact finite-system two-time Keldysh demonstration."""

from __future__ import annotations

import numpy as np

from quantum_transport import (
    equilibrium_one_body_density,
    region_interface_current,
    two_time_greens,
)


def main() -> None:
    time = np.linspace(0.0, 4.0, 101)

    def hamiltonian(value: float) -> np.ndarray:
        central_energy = 0.45 * np.tanh(2.0 * (value - 0.6))
        phase = 0.35 * np.sin(1.2 * value)
        return np.array(
            [
                [-0.35, -0.55, 0.0],
                [
                    -0.55,
                    central_energy,
                    -0.42 * np.exp(1j * phase),
                ],
                [
                    0.0,
                    -0.42 * np.exp(-1j * phase),
                    0.25,
                ],
            ],
            dtype=np.complex128,
        )

    initial_density = equilibrium_one_body_density(
        hamiltonian(time[0]), mu=0.0, temperature=0.18
    )
    result = two_time_greens(
        time,
        hamiltonian,
        initial_density,
    )
    density = result.density_matrices()
    central_population = density[:, 1, 1].real
    outgoing = np.array(
        [
            region_interface_current(
                hamiltonian(value), rho, [1]
            )
            for value, rho in zip(time, density)
        ]
    )

    print("Exact finite-system two-time Keldysh")
    print("shape G<(t,t'):", result.lesser.shape)
    print(
        "spectral identity error:",
        f"{result.spectral_identity_error():.3e}",
    )
    print(
        "central population range:",
        f"[{central_population.min():.6f}, "
        f"{central_population.max():.6f}]",
    )
    print(
        "maximum interface current:",
        f"{np.max(np.abs(outgoing)):.6f}",
    )
    print("VERDICT: PASS")


if __name__ == "__main__":
    main()
