"""Demo for the quantum_transport package."""

from __future__ import annotations

import numpy as np
import sympy as sp

from quantum_transport import (
    advanced_green,
    anticommutator,
    check_eom_closure,
    commutator,
    current_density_omega,
    drude_weight,
    greater_green_equilibrium,
    collinear_hartree_self_consistent,
    lesser_green_equilibrium,
    persistent_current,
    real_to_k_space,
    retarded_green,
    split_k_blocks,
)


def main() -> None:
    n_sites = 8
    n_electrons = 8
    gamma = 1.0
    lambda_r = 0.4
    u_hubbard = 0.4
    phi = 0.2

    hf = collinear_hartree_self_consistent(
        n_sites=n_sites,
        n_electrons=n_electrons,
        gamma=gamma,
        lambda_r=lambda_r,
        phi_over_phi0=phi,
        u_hubbard=u_hubbard,
        mixing=0.5,
        tol=1e-10,
    )
    print(f"Collinear Hartree converged: {hf.converged} in {hf.iterations} iterations")

    h_k, k_values = real_to_k_space(hf.hamiltonian, n_sites=n_sites, spin_dim=2)
    blocks = split_k_blocks(h_k, n_sites=n_sites, spin_dim=2, require_block_diagonal=False)
    print("First 3 k points:", np.round(k_values[:3], 5))
    print("First diagonal k subblock (Rashba couples k sectors in this basis):\n", np.round(blocks[0], 5))

    mu = 0.5 * (hf.eigenvalues[n_electrons - 1] + hf.eigenvalues[n_electrons])
    omega = 0.0
    g_r = retarded_green(hf.hamiltonian, omega=omega, eta=1e-3)
    g_a = advanced_green(hf.hamiltonian, omega=omega, eta=1e-3)
    g_lesser = lesser_green_equilibrium(g_r, g_a, omega=omega, mu=mu, temperature=0.0)
    g_greater = greater_green_equilibrium(g_r, g_a, omega=omega, mu=mu, temperature=0.0)
    print("||G^<||_F =", float(np.linalg.norm(g_lesser)))
    print("||G^>||_F =", float(np.linalg.norm(g_greater)))

    j_omega = current_density_omega(
        g_ret=g_r,
        g_adv=g_a,
        omega=omega,
        n_sites=n_sites,
        gamma=gamma,
        lambda_r=lambda_r,
        phi_over_phi0=phi,
        mu=mu,
        temperature=0.0,
    )
    print("J_c(omega=0) =", j_omega)

    omega_grid = np.linspace(-6.0, 6.0, 2001)
    i_phi = persistent_current(
        hamiltonian=hf.hamiltonian,
        omega_grid=omega_grid,
        n_sites=n_sites,
        gamma=gamma,
        lambda_r=lambda_r,
        phi_over_phi0=phi,
        mu=mu,
        temperature=0.0,
        eta=1e-3,
    )

    dphi = 1e-3
    hf_plus = collinear_hartree_self_consistent(
        n_sites=n_sites,
        n_electrons=n_electrons,
        gamma=gamma,
        lambda_r=lambda_r,
        phi_over_phi0=phi + dphi,
        u_hubbard=u_hubbard,
    )
    hf_minus = collinear_hartree_self_consistent(
        n_sites=n_sites,
        n_electrons=n_electrons,
        gamma=gamma,
        lambda_r=lambda_r,
        phi_over_phi0=phi - dphi,
        u_hubbard=u_hubbard,
    )
    i_plus = persistent_current(
        hamiltonian=hf_plus.hamiltonian,
        omega_grid=omega_grid,
        n_sites=n_sites,
        gamma=gamma,
        lambda_r=lambda_r,
        phi_over_phi0=phi + dphi,
        mu=mu,
        temperature=0.0,
        eta=1e-3,
    )
    i_minus = persistent_current(
        hamiltonian=hf_minus.hamiltonian,
        omega_grid=omega_grid,
        n_sites=n_sites,
        gamma=gamma,
        lambda_r=lambda_r,
        phi_over_phi0=phi - dphi,
        mu=mu,
        temperature=0.0,
        eta=1e-3,
    )
    d_est = drude_weight(i_plus, i_minus, delta_phi=dphi)
    print("Persistent current I(phi) =", i_phi)
    print("Drude estimate dI/dphi =", d_est)

    c0, c1 = sp.symbols("c0 c1", commutative=False)
    h_sym = c0 * c1 + c1 * c0
    print("[c0, H] =", commutator(c0, h_sym))
    print("{c0, c1} =", anticommutator(c0, c1))
    closure = check_eom_closure([c0, c1], h_sym)
    print("EOM closed:", closure.is_closed)


if __name__ == "__main__":
    main()
