from __future__ import annotations

import numpy as np

from quantum_transport import LeadSelfEnergy, RashbaRingDevice, SpinfulDimer, SpinfulSingleSite


def main() -> None:
    print("Spinful single site with explicit leads:")
    site = SpinfulSingleSite(eps_up=0.1, eps_down=-0.1, spin_flip=0.25)
    left = LeadSelfEnergy.wide_band(np.diag([0.5, 0.5]), mu=0.2, name="L")
    right = LeadSelfEnergy.wide_band(np.diag([0.4, 0.4]), mu=-0.2, name="R")
    transport = site.transport(left, right)
    print("T(omega=0):", transport.transmission(0.0))
    print("T_up->up(omega=0):", transport.spin_transmission(0.0, "up", "up"))
    print("T_up->down(omega=0):", transport.spin_transmission(0.0, "up", "down"))
    print("P_spin_z(omega=0):", transport.spin_polarization(0.0, axis="z"))
    print("P_spin_x(omega=0):", transport.spin_polarization(0.0, axis="x"))

    print("\nEnergy-dependent sampled lead example:")
    omega_grid = np.array([-1.0, 0.0, 1.0])
    sigma_values = np.array([[[-0.1j]], [[-0.2j]], [[-0.3j]]], dtype=np.complex128)
    sampled = LeadSelfEnergy.sampled(omega_grid, sigma_values, mu=0.0, name="sampled")
    print("Sigma^r_sampled(0.5):", sampled.sigma_retarded(0.5))

    print("\nSemi-infinite chain lead example:")
    chain_lead = LeadSelfEnergy.semi_infinite_chain(np.array([[0.4]], dtype=np.complex128), onsite=0.0, hopping=1.0, mu=0.1)
    print("Sigma^r_chain(0.0):", chain_lead.sigma_retarded(0.0))

    print("\nFerromagnetic leads with explicit rotations:")
    basis = ["left_up", "left_down", "right_up", "right_down"]
    left_fm = LeadSelfEnergy.ferromagnetic_wide_band(basis, gamma_majority=0.7, gamma_minority=0.2, theta=np.pi / 2.0, phi=0.0, mu=0.15, name="L_fm")
    right_mix = LeadSelfEnergy.rotated_spin_mixing_wide_band(
        basis,
        np.array([[0.6, 0.12j], [-0.12j, 0.25]], dtype=np.complex128),
        theta=np.pi / 3.0,
        phi=np.pi / 4.0,
        mu=-0.15,
        name="R_mix",
    )
    dimer_pol = SpinfulDimer(hopping=1.0, spin_orbit=0.2, onsite_spin_flip_left=0.1)
    pol_transport = dimer_pol.transport(left_fm, right_mix)
    dense_grid = np.linspace(-6.0, 6.0, 4001)
    print("Spin conductance vector:", pol_transport.spin_conductance_vector(mu=0.0))
    print("Spin Landauer current vector:", pol_transport.spin_landauer_current_vector(dense_grid, mu_left=0.15, mu_right=-0.15))
    print("Spin Keldysh current vector:", pol_transport.spin_current_vector_from_keldysh(dense_grid, lead="left"))
    print("Current spin polarization vector:", pol_transport.current_spin_polarization_vector(dense_grid, mu_left=0.15, mu_right=-0.15))

    print("\nSpinful dimer current, conductance, and spin transport:")
    dimer = SpinfulDimer(hopping=1.0, spin_orbit=0.2, onsite_spin_flip_left=0.1)
    left_dimer = LeadSelfEnergy.wide_band(np.diag([0.7, 0.2, 0.0, 0.0]), mu=0.15, name="L_dimer")
    right_dimer = LeadSelfEnergy.wide_band(np.diag([0.0, 0.0, 0.6, 0.1]), mu=-0.15, name="R_dimer")
    dimer_transport = dimer.transport(left_dimer, right_dimer)
    print("G(mu=0):", dimer_transport.conductance(mu=0.0))
    print("G_spin_z(mu=0):", dimer_transport.spin_conductance(mu=0.0, axis="z"))
    print("G_spin_x(mu=0):", dimer_transport.spin_conductance(mu=0.0, axis="x"))
    print("I_Landauer(mu_L=0.15, mu_R=-0.15):", dimer_transport.landauer_current(dense_grid, mu_left=0.15, mu_right=-0.15))
    print("I_spin_z_Landauer:", dimer_transport.spin_landauer_current(dense_grid, mu_left=0.15, mu_right=-0.15, axis="z"))
    print("I_spin_x_Landauer:", dimer_transport.spin_landauer_current(dense_grid, mu_left=0.15, mu_right=-0.15, axis="x"))
    print("I_Keldysh(left lead):", dimer_transport.current_from_keldysh(dense_grid, lead="left"))
    print("I_spin_z_Keldysh(left lead):", dimer_transport.spin_current_from_keldysh(dense_grid, lead="left", axis="z"))
    print("I_spin_x_Keldysh(left lead):", dimer_transport.spin_current_from_keldysh(dense_grid, lead="left", axis="x"))
    print("Current spin polarization z:", dimer_transport.current_spin_polarization(dense_grid, mu_left=0.15, mu_right=-0.15, axis="z"))
    print("Current spin polarization x:", dimer_transport.current_spin_polarization(dense_grid, mu_left=0.15, mu_right=-0.15, axis="x"))

    print("\nRashba ring open device:")
    ring = RashbaRingDevice(n_sites=4, gamma=1.0, lambda_r=0.3, phi_over_phi0=0.15)
    gamma_left_ring = np.diag([0.5, 0.5] + [0.0] * 6)
    gamma_right_ring = np.diag([0.0] * 6 + [0.5, 0.5])
    left_ring = LeadSelfEnergy.wide_band(gamma_left_ring, mu=0.05, name="L_ring")
    right_ring = LeadSelfEnergy.wide_band(gamma_right_ring, mu=-0.05, name="R_ring")
    ring_transport = ring.transport(left_ring, right_ring)
    print("Ring transmission at omega=0:", ring_transport.transmission(0.0))
    print("Ring spin polarization z at omega=0:", ring_transport.spin_polarization(0.0, axis="z"))
    print("Ring spin polarization x at omega=0:", ring_transport.spin_polarization(0.0, axis="x"))


if __name__ == "__main__":
    main()
