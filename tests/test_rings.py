import numpy as np
import pytest

from quantum_transport import AharonovBohmRing, build_rashba_hubbard_ring_real_space, persistent_current, persistent_spin_current, persistent_spin_resolved_current


def test_ab_ring_spectrum_matches_direct_builder():
    ring = AharonovBohmRing(n_sites=4, gamma=1.0, lambda_r=0.2)
    direct = build_rashba_hubbard_ring_real_space(n_sites=4, gamma=1.0, lambda_r=0.2, phi_over_phi0=0.15)

    assert np.allclose(ring.hamiltonian(phi_over_phi0=0.15), direct)
    assert np.allclose(np.sort(ring.spectrum(phi_over_phi0=0.15)), np.sort(np.linalg.eigvalsh(direct)))


def test_ab_ring_k_space_and_blocks_have_expected_shapes():
    ring = AharonovBohmRing(n_sites=6, gamma=1.0, lambda_r=0.0)
    h_k, k_values = ring.k_space(phi_over_phi0=0.2)
    blocks = ring.k_blocks(phi_over_phi0=0.2)

    assert h_k.shape == (12, 12)
    assert k_values.shape == (6,)
    assert len(blocks) == 6
    assert all(block.shape == (2, 2) for block in blocks)


def test_ab_ring_k_blocks_reject_rashba_coupled_k_sectors():
    ring = AharonovBohmRing(n_sites=6, gamma=1.0, lambda_r=0.1)
    with pytest.raises(ValueError, match="not block diagonal"):
        ring.k_blocks(phi_over_phi0=0.2)

    inspected = ring.k_blocks(phi_over_phi0=0.2, require_block_diagonal=False)
    assert len(inspected) == 6


def test_ab_ring_persistent_current_matches_functional_helper():
    ring = AharonovBohmRing(n_sites=4, gamma=1.0, lambda_r=0.2)
    omega_grid = np.linspace(-6.0, 6.0, 2001)
    phi = 0.12
    h = ring.hamiltonian(phi_over_phi0=phi)

    current_class = ring.persistent_current(omega_grid, phi_over_phi0=phi, mu=0.0, eta=1e-2)
    current_function = persistent_current(h, omega_grid=omega_grid, n_sites=4, gamma=1.0, lambda_r=0.2, phi_over_phi0=phi, mu=0.0, eta=1e-2)

    assert abs(current_class - current_function) < 1e-10


def test_ab_ring_spin_resolved_persistent_currents_sum_to_total():
    ring = AharonovBohmRing(n_sites=4, gamma=1.0, lambda_r=0.2)
    omega_grid = np.linspace(-6.0, 6.0, 2001)
    phi = 0.12
    h = ring.hamiltonian(phi_over_phi0=phi)

    total = ring.persistent_current(omega_grid, phi_over_phi0=phi, mu=0.0, eta=1e-2)
    plus = ring.persistent_spin_resolved_current(omega_grid, phi_over_phi0=phi, axis="z", component="+", mu=0.0, eta=1e-2)
    minus = ring.persistent_spin_resolved_current(omega_grid, phi_over_phi0=phi, axis="z", component="-", mu=0.0, eta=1e-2)
    total_function = persistent_spin_resolved_current(h, omega_grid=omega_grid, n_sites=4, gamma=1.0, lambda_r=0.2, phi_over_phi0=phi, axis="z", component="+", mu=0.0, eta=1e-2)

    assert abs(total - (plus + minus)) < 1e-8
    assert abs(plus - total_function) < 1e-10



def test_ab_ring_spin_current_is_finite_and_matches_helper():
    ring = AharonovBohmRing(n_sites=4, gamma=1.0, lambda_r=0.2)
    omega_grid = np.linspace(-6.0, 6.0, 2001)
    phi = 0.12
    h = ring.hamiltonian(phi_over_phi0=phi)

    current_class = ring.persistent_spin_current(omega_grid, phi_over_phi0=phi, axis="x", mu=0.0, eta=1e-2)
    current_function = persistent_spin_current(h, omega_grid=omega_grid, n_sites=4, gamma=1.0, lambda_r=0.2, phi_over_phi0=phi, axis="x", mu=0.0, eta=1e-2)

    assert np.isfinite(current_class)
    assert abs(current_class - current_function) < 1e-10


def test_ab_ring_persistent_current_vs_flux_and_drude_are_finite():
    ring = AharonovBohmRing(n_sites=4, gamma=1.0, lambda_r=0.1)
    omega_grid = np.linspace(-6.0, 6.0, 2001)
    fluxes = np.linspace(-0.2, 0.2, 5)

    currents = ring.persistent_current_vs_flux(fluxes, omega_grid, mu=0.0, eta=1e-2)
    drude = ring.drude_weight(omega_grid, phi_over_phi0=0.0, delta_phi=1e-3, mu=0.0, eta=1e-2)

    assert currents.shape == fluxes.shape
    assert np.all(np.isfinite(currents))
    assert np.isfinite(drude)


def test_ab_ring_hartree_fock_returns_converged_result_and_hf_hamiltonian():
    ring = AharonovBohmRing(n_sites=4, gamma=1.0, lambda_r=0.1, u_hubbard=0.4)
    result = ring.hartree_fock(n_electrons=4, phi_over_phi0=0.1, tol=1e-7, max_iter=200)
    h_hf = ring.hf_hamiltonian(n_electrons=4, phi_over_phi0=0.1, tol=1e-7, max_iter=200)

    assert result.hamiltonian.shape == (8, 8)
    assert h_hf.shape == (8, 8)
    assert np.allclose(result.hamiltonian, h_hf)
    assert np.all(result.n_up >= -1e-10)
    assert np.all(result.n_down >= -1e-10)


def test_ab_ring_hf_persistent_current_and_drude_are_finite():
    ring = AharonovBohmRing(n_sites=4, gamma=1.0, lambda_r=0.1, u_hubbard=0.4)
    omega_grid = np.linspace(-6.0, 6.0, 2001)
    fluxes = np.linspace(-0.2, 0.2, 5)

    current_hf = ring.persistent_current_hf(omega_grid, n_electrons=4, phi_over_phi0=0.1, eta=1e-2, tol=1e-7, max_iter=200)
    currents_hf = ring.persistent_current_vs_flux_hf(fluxes, omega_grid, n_electrons=4, eta=1e-2, tol=1e-7, max_iter=200)
    drude_hf = ring.drude_weight_hf(omega_grid, n_electrons=4, phi_over_phi0=0.0, delta_phi=1e-3, eta=1e-2, tol=1e-7, max_iter=200)

    assert np.isfinite(current_hf)
    assert currents_hf.shape == fluxes.shape
    assert np.all(np.isfinite(currents_hf))
    assert np.isfinite(drude_hf)


def test_ab_ring_hf_spin_resolved_currents_sum_to_total():
    ring = AharonovBohmRing(n_sites=4, gamma=1.0, lambda_r=0.1, u_hubbard=0.4)
    omega_grid = np.linspace(-6.0, 6.0, 2001)
    phi = 0.1

    total = ring.persistent_current_hf(omega_grid, n_electrons=4, phi_over_phi0=phi, eta=1e-2, tol=1e-7, max_iter=200)
    plus = ring.persistent_spin_resolved_current_hf(omega_grid, n_electrons=4, phi_over_phi0=phi, axis="z", component="+", eta=1e-2, tol=1e-7, max_iter=200)
    minus = ring.persistent_spin_resolved_current_hf(omega_grid, n_electrons=4, phi_over_phi0=phi, axis="z", component="-", eta=1e-2, tol=1e-7, max_iter=200)
    spin = ring.persistent_spin_current_hf(omega_grid, n_electrons=4, phi_over_phi0=phi, axis="z", eta=1e-2, tol=1e-7, max_iter=200)

    assert abs(total - (plus + minus)) < 1e-6
    assert abs(spin - (plus - minus)) < 1e-6
