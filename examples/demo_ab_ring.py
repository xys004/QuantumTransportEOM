from __future__ import annotations

import numpy as np

from quantum_transport import AharonovBohmRing


def main() -> None:
    ring = AharonovBohmRing(n_sites=4, gamma=1.0, lambda_r=0.25, u_hubbard=0.4)
    omega_grid = np.linspace(-6.0, 6.0, 1601)
    flux_values = np.linspace(-0.2, 0.2, 5)

    print("Aharonov-Bohm ring persistent current demo:")
    print("Spectrum at phi=0.10:", ring.spectrum(phi_over_phi0=0.10))

    currents = ring.persistent_current_vs_flux(flux_values, omega_grid, mu=0.0, eta=1e-2)
    spin_currents_z = ring.persistent_spin_current_vs_flux(flux_values, omega_grid, axis="z", mu=0.0, eta=1e-2)
    spin_currents_x = ring.persistent_spin_current_vs_flux(flux_values, omega_grid, axis="x", mu=0.0, eta=1e-2)
    for phi, current, spin_z, spin_x in zip(flux_values, currents, spin_currents_z, spin_currents_x):
        print(f"I(phi/phi0={phi:+.3f}) = {current:+.6f}; Iz_spin={spin_z:+.6f}; Ix_spin={spin_x:+.6f}")

    plus_z = ring.persistent_spin_resolved_current(omega_grid, phi_over_phi0=0.10, axis="z", component="+", mu=0.0, eta=1e-2)
    minus_z = ring.persistent_spin_resolved_current(omega_grid, phi_over_phi0=0.10, axis="z", component="-", mu=0.0, eta=1e-2)
    print("I_z,+ at phi=0.10:", plus_z)
    print("I_z,- at phi=0.10:", minus_z)

    drude = ring.drude_weight(omega_grid, phi_over_phi0=0.0, delta_phi=1e-3, mu=0.0, eta=1e-2)
    print("Drude weight at phi=0:", drude)

    h_k, k_values = ring.k_space(phi_over_phi0=0.10)
    print("k-values:", k_values)
    print("First diagonal k subblock (Rashba couples k sectors in this basis):\n", ring.k_blocks(phi_over_phi0=0.10, require_block_diagonal=False)[0])
    print("H_k shape:", h_k.shape)

    print("\nCollinear-Hartree Aharonov-Bohm ring:")
    hf_result = ring.hartree_fock(n_electrons=4, phi_over_phi0=0.10, tol=1e-7, max_iter=120)
    print("Collinear Hartree converged:", hf_result.converged)
    print("Collinear Hartree iterations:", hf_result.iterations)
    print("Collinear Hartree densities up:", hf_result.n_up)
    print("Collinear Hartree densities down:", hf_result.n_down)

    currents_hf = ring.persistent_current_vs_flux_hf(flux_values, omega_grid, n_electrons=4, eta=1e-2, tol=1e-7, max_iter=120)
    spin_currents_hf_z = ring.persistent_spin_current_vs_flux_hf(flux_values, omega_grid, n_electrons=4, axis="z", eta=1e-2, tol=1e-7, max_iter=120)
    for phi, current, spin_z in zip(flux_values, currents_hf, spin_currents_hf_z):
        print(f"I_CH(phi/phi0={phi:+.3f}) = {current:+.6f}; Iz_spin_CH={spin_z:+.6f}")

    plus_hf = ring.persistent_spin_resolved_current_hf(omega_grid, n_electrons=4, phi_over_phi0=0.10, axis="z", component="+", eta=1e-2, tol=1e-7, max_iter=120)
    minus_hf = ring.persistent_spin_resolved_current_hf(omega_grid, n_electrons=4, phi_over_phi0=0.10, axis="z", component="-", eta=1e-2, tol=1e-7, max_iter=120)
    print("I_z,+_CH at phi=0.10:", plus_hf)
    print("I_z,-_CH at phi=0.10:", minus_hf)

    drude_hf = ring.drude_weight_hf(omega_grid, n_electrons=4, phi_over_phi0=0.0, delta_phi=1e-3, eta=1e-2, tol=1e-7, max_iter=120)
    print("Collinear Hartree Drude weight at phi=0:", drude_hf)


if __name__ == "__main__":
    main()
