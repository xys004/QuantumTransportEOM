"""Stationary matrix NEGF with continuum leads on an explicit two-time grid."""

from __future__ import annotations

import numpy as np

from quantum_transport import (
    LeadSelfEnergy,
    SpinfulSingleSite,
    stationary_greens_two_time,
    stationary_self_energy_two_time,
)


def main() -> None:
    device = SpinfulSingleSite(eps_up=0.10, eps_down=0.32, spin_flip=0.08)
    left = LeadSelfEnergy.wide_band(
        np.diag([0.45, 0.25]), mu=0.25, temperature=0.10, name="left"
    )
    right = LeadSelfEnergy.wide_band(
        np.diag([0.30, 0.40]), mu=-0.15, temperature=0.10, name="right"
    )
    transport = device.transport(left, right)
    omega = np.linspace(-30.0, 30.0, 12001)
    time = np.linspace(0.0, 3.0, 31)

    green = stationary_greens_two_time(transport, time, omega)
    sigma_left = stationary_self_energy_two_time(left, time, omega)
    density = green.density_matrices()[0]

    print("Stationary continuum two-time NEGF")
    print("shape G<(t,t'):", green.lesser.shape)
    print("density matrix at equal time:")
    print(density)
    print("Green consistency:", green.consistency_report().as_dict())
    print("left self-energy consistency:", sigma_left.consistency_report().as_dict())
    print("equal-time density drift:", f"{green.equal_time_drift():.3e}")
    print("VERDICT: PASS")


if __name__ == "__main__":
    main()
