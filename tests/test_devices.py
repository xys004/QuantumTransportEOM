import numpy as np

from quantum_transport import (
    LeadSelfEnergy,
    RashbaRingDevice,
    SpinfulDimer,
    SpinfulSingleSite,
    rotated_local_spin_matrix_numeric,
    spin_axis_operator_numeric,
    spin_axis_projector_numeric,
    spin_rotation_matrix_numeric,
)


def test_wide_band_lead_produces_expected_retarded_and_lesser_self_energy():
    gamma = np.diag([0.4, 0.2])
    lead = LeadSelfEnergy.wide_band(gamma, mu=0.0, temperature=0.0)

    sigma_r = lead.sigma_retarded(-1.0)
    sigma_less = lead.sigma_lesser(-1.0)

    assert np.allclose(sigma_r, -0.5j * gamma)
    assert np.allclose(sigma_less, 1.0j * gamma)


def test_polarized_wide_band_lead_supports_noncollinear_spin_axis():
    basis = ["left_up", "left_down", "right_up", "right_down"]
    lead = LeadSelfEnergy.polarized_wide_band(basis, np.eye(4), polarization=0.6, axis="x", mu=0.0)
    gamma = lead.gamma(0.0)
    p_plus_x = spin_axis_projector_numeric(basis, axis="x", component="+")
    p_minus_x = spin_axis_projector_numeric(basis, axis="x", component="-")

    gamma_plus = np.real(np.trace(p_plus_x @ gamma))
    gamma_minus = np.real(np.trace(p_minus_x @ gamma))

    assert gamma_plus > gamma_minus


def test_ferromagnetic_wide_band_with_explicit_rotation_creates_expected_x_polarization():
    basis = ["left_up", "left_down", "right_up", "right_down"]
    lead = LeadSelfEnergy.ferromagnetic_wide_band(basis, gamma_majority=1.0, gamma_minority=0.2, theta=np.pi / 2.0, phi=0.0, mu=0.0)
    gamma = lead.gamma(0.0)
    p_plus_x = spin_axis_projector_numeric(basis, axis="x", component="+")
    p_minus_x = spin_axis_projector_numeric(basis, axis="x", component="-")

    assert np.real(np.trace(p_plus_x @ gamma)) > np.real(np.trace(p_minus_x @ gamma))



def test_rotated_spin_mixing_wide_band_supports_general_local_spin_matrix():
    basis = ["left_up", "left_down", "right_up", "right_down"]
    local = np.array([[0.8, 0.1j], [-0.1j, 0.3]], dtype=np.complex128)
    lead = LeadSelfEnergy.rotated_spin_mixing_wide_band(basis, local, theta=np.pi / 3.0, phi=np.pi / 4.0, mu=0.0)
    gamma = lead.gamma(0.0)

    assert np.allclose(gamma, gamma.conj().T)
    assert np.all(np.linalg.eigvalsh(gamma) >= -1e-12)



def test_spin_rotation_and_rotated_local_spin_matrix_helpers_are_consistent():
    basis = ["left_up", "left_down"]
    rotation = spin_rotation_matrix_numeric(theta=np.pi / 3.0, phi=np.pi / 5.0)
    local = np.diag([1.0, 0.2]).astype(np.complex128)
    embedded = rotated_local_spin_matrix_numeric(basis, local, theta=np.pi / 3.0, phi=np.pi / 5.0)
    expected = rotation @ local @ rotation.conj().T

    assert np.allclose(embedded, expected)


def test_sampled_lead_interpolates_energy_dependent_self_energy():
    omega_grid = np.array([-1.0, 0.0, 1.0])
    sigma_values = np.array([
        [[-0.1j]],
        [[-0.2j]],
        [[-0.3j]],
    ], dtype=np.complex128)
    lead = LeadSelfEnergy.sampled(omega_grid, sigma_values, mu=0.0)
    sigma_mid = lead.sigma_retarded(0.5)
    assert np.allclose(sigma_mid, np.array([[-0.25j]]))


def test_semi_infinite_chain_lead_has_negative_imaginary_retarded_part_inside_band():
    coupling = np.array([[0.4]], dtype=np.complex128)
    lead = LeadSelfEnergy.semi_infinite_chain(coupling, onsite=0.0, hopping=1.0)
    sigma_r = lead.sigma_retarded(0.0)
    assert np.imag(sigma_r[0, 0]) < 0.0


def test_spin_axis_projectors_form_a_resolution_of_identity():
    basis = ["left_up", "left_down", "right_up", "right_down"]
    p_plus = spin_axis_projector_numeric(basis, axis="x", component="+")
    p_minus = spin_axis_projector_numeric(basis, axis="x", component="-")
    s_z = spin_axis_operator_numeric(basis, axis="z")

    assert np.allclose(p_plus + p_minus, np.eye(4))
    assert np.allclose(s_z, np.diag([0.5, -0.5, 0.5, -0.5]))


def test_spinful_single_site_transport_supports_spin_resolved_transmission():
    device = SpinfulSingleSite(eps_up=0.0, eps_down=0.0, spin_flip=0.3)
    gamma_l = np.diag([0.5, 0.5])
    gamma_r = np.diag([0.4, 0.4])
    left = LeadSelfEnergy.wide_band(gamma_l, mu=0.2)
    right = LeadSelfEnergy.wide_band(gamma_r, mu=-0.2)
    transport = device.transport(left, right)

    total = transport.transmission(0.0)
    channels = sum(
        transport.spin_transmission(0.0, left_spin, right_spin)
        for left_spin in ("up", "down")
        for right_spin in ("up", "down")
    )

    assert total >= 0.0
    assert abs(total - channels) < 1e-10


def test_spin_axis_transmissions_sum_to_total_for_x_basis():
    device = SpinfulSingleSite(eps_up=0.0, eps_down=0.0, spin_flip=0.25)
    left = LeadSelfEnergy.wide_band(np.diag([0.5, 0.5]), mu=0.1)
    right = LeadSelfEnergy.wide_band(np.diag([0.4, 0.4]), mu=-0.1)
    transport = device.transport(left, right)

    total = transport.transmission(0.0)
    total_x = transport.spin_resolved_transmission(0.0, "+", axis="x") + transport.spin_resolved_transmission(0.0, "-", axis="x")

    assert abs(total - total_x) < 1e-10


def test_spin_resolved_transmission_and_polarization_are_zero_for_spin_symmetric_case():
    device = SpinfulSingleSite(eps_up=0.0, eps_down=0.0, spin_flip=0.0)
    left = LeadSelfEnergy.wide_band(np.diag([0.5, 0.5]), mu=0.1)
    right = LeadSelfEnergy.wide_band(np.diag([0.4, 0.4]), mu=-0.1)
    transport = device.transport(left, right)

    t_up = transport.spin_resolved_transmission(0.0, "up")
    t_down = transport.spin_resolved_transmission(0.0, "down")

    assert abs(t_up - t_down) < 1e-12
    assert abs(transport.spin_polarization(0.0)) < 1e-12


def test_spinful_dimer_current_vanishes_at_zero_bias():
    device = SpinfulDimer(hopping=1.0, spin_orbit=0.2)
    gamma_l = np.diag([0.6, 0.6, 0.0, 0.0])
    gamma_r = np.diag([0.0, 0.0, 0.7, 0.7])
    left = LeadSelfEnergy.wide_band(gamma_l, mu=0.0)
    right = LeadSelfEnergy.wide_band(gamma_r, mu=0.0)
    transport = device.transport(left, right)
    omega_grid = np.linspace(-6.0, 6.0, 2001)

    current = transport.landauer_current(omega_grid, mu_left=0.0, mu_right=0.0)
    assert abs(current) < 1e-12


def test_keldysh_current_matches_landauer_for_noninteracting_two_terminal_case():
    device = SpinfulSingleSite(eps_up=0.1, eps_down=0.1, spin_flip=0.0)
    gamma_l = np.diag([0.5, 0.5])
    gamma_r = np.diag([0.4, 0.4])
    left = LeadSelfEnergy.wide_band(gamma_l, mu=0.2)
    right = LeadSelfEnergy.wide_band(gamma_r, mu=-0.2)
    transport = device.transport(left, right)
    omega_grid = np.linspace(-6.0, 6.0, 4001)

    current_landauer = transport.landauer_current(omega_grid, mu_left=0.2, mu_right=-0.2)
    current_keldysh = transport.current_from_keldysh(omega_grid, lead="left")

    assert abs(current_landauer - current_keldysh) < 1e-3


def test_spin_resolved_currents_sum_to_total_and_spin_landauer_matches_keldysh():
    device = SpinfulSingleSite(eps_up=0.05, eps_down=0.35, spin_flip=0.0)
    left = LeadSelfEnergy.wide_band(np.diag([0.7, 0.2]), mu=0.3)
    right = LeadSelfEnergy.wide_band(np.diag([0.5, 0.1]), mu=-0.1)
    transport = device.transport(left, right)
    omega_grid = np.linspace(-6.0, 6.0, 4001)

    current_total = transport.landauer_current(omega_grid, mu_left=0.3, mu_right=-0.1)
    current_up = transport.spin_resolved_landauer_current(omega_grid, "up", mu_left=0.3, mu_right=-0.1)
    current_down = transport.spin_resolved_landauer_current(omega_grid, "down", mu_left=0.3, mu_right=-0.1)
    spin_landauer = transport.spin_landauer_current(omega_grid, mu_left=0.3, mu_right=-0.1)
    spin_keldysh = transport.spin_current_from_keldysh(omega_grid, lead="left")

    assert abs(current_total - (current_up + current_down)) < 1e-8
    assert abs(spin_landauer - (current_up - current_down)) < 1e-8
    assert abs(spin_landauer - spin_keldysh) < 1e-3
    assert abs(transport.current_spin_polarization(omega_grid, mu_left=0.3, mu_right=-0.1)) <= 1.0 + 1e-12


def test_spin_x_current_is_supported_and_bounded():
    device = SpinfulDimer(hopping=1.0, spin_orbit=0.2, onsite_spin_flip_left=0.1)
    left = LeadSelfEnergy.wide_band(np.diag([0.7, 0.2, 0.0, 0.0]), mu=0.2)
    right = LeadSelfEnergy.wide_band(np.diag([0.0, 0.0, 0.6, 0.3]), mu=-0.1)
    transport = device.transport(left, right)
    omega_grid = np.linspace(-6.0, 6.0, 4001)

    current_x = transport.spin_landauer_current(omega_grid, mu_left=0.2, mu_right=-0.1, axis="x")
    polarization_x = transport.current_spin_polarization(omega_grid, mu_left=0.2, mu_right=-0.1, axis="x")

    assert np.isfinite(current_x)
    assert abs(polarization_x) <= 1.0 + 1e-12


def test_spin_vector_helpers_return_xyz_components():
    device = SpinfulDimer(hopping=1.0, spin_orbit=0.2, onsite_spin_flip_left=0.1)
    left = LeadSelfEnergy.polarized_wide_band(["left_up", "left_down", "right_up", "right_down"], np.diag([0.7, 0.7, 0.0, 0.0]), polarization=0.4, axis="x", mu=0.2)
    right = LeadSelfEnergy.polarized_wide_band(["left_up", "left_down", "right_up", "right_down"], np.diag([0.0, 0.0, 0.6, 0.6]), polarization=0.3, axis="z", mu=-0.1)
    transport = device.transport(left, right)
    omega_grid = np.linspace(-6.0, 6.0, 2001)

    current_vec = transport.spin_landauer_current_vector(omega_grid, mu_left=0.2, mu_right=-0.1)
    conductance_vec = transport.spin_conductance_vector(mu=0.0)
    keldysh_vec = transport.spin_current_vector_from_keldysh(omega_grid, lead="left")

    assert set(current_vec) == {"x", "y", "z"}
    assert set(conductance_vec) == {"x", "y", "z"}
    assert set(keldysh_vec) == {"x", "y", "z"}
    assert all(np.isfinite(value) for value in current_vec.values())
    assert all(np.isfinite(value) for value in conductance_vec.values())
    assert all(np.isfinite(value) for value in keldysh_vec.values())


def test_rashba_ring_device_builds_spinful_basis_labels_and_matrix_shape():
    device = RashbaRingDevice(n_sites=4, gamma=1.0, lambda_r=0.2, phi_over_phi0=0.1)
    assert device.hamiltonian.shape == (8, 8)
    assert device.basis_labels[0] == "site0_up"
    assert device.basis_labels[1] == "site0_down"
