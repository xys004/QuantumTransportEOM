import numpy as np
import sympy as sp

from quantum_transport import (
    KeldyshSelfEnergy,
    KeldyshTransportView,
    LeadSelfEnergy,
    SpinfulSingleSite,
    greater_from_retarded_sigma,
    greater_from_retarded_sigma_symbolic,
    green_keldysh_from_lesser_greater,
    green_keldysh_symbolic,
    keldysh_from_retarded_sigma,
    keldysh_from_retarded_sigma_symbolic,
    lesser_from_retarded_sigma,
    lesser_from_retarded_sigma_symbolic,
    meir_wingreen_current,
    meir_wingreen_current_density,
    sigma_keldysh,
    sigma_keldysh_symbolic,
)


def test_keldysh_component_sum_relations_hold_for_numeric_arrays():
    sigma_less = np.array([[0.0 + 0.2j]], dtype=np.complex128)
    sigma_greater = np.array([[0.0 - 0.7j]], dtype=np.complex128)
    g_less = np.array([[0.0 + 0.4j]], dtype=np.complex128)
    g_greater = np.array([[0.0 - 0.9j]], dtype=np.complex128)

    assert np.allclose(sigma_keldysh(sigma_less, sigma_greater), sigma_less + sigma_greater)
    assert np.allclose(green_keldysh_from_lesser_greater(g_less, g_greater), g_less + g_greater)


def test_keldysh_component_sum_relations_hold_symbolically():
    gl, gg, sl, sg = sp.symbols('gl gg sl sg')

    assert sp.simplify(green_keldysh_symbolic(gl, gg) - (gl + gg)) == 0
    assert sp.simplify(sigma_keldysh_symbolic(sl, sg) - (sl + sg)) == 0


def test_lesser_greater_keldysh_from_retarded_sigma_build_expected_products():
    g_r = np.array([[1.0 + 1.0j]], dtype=np.complex128)
    g_a = np.array([[1.0 - 1.0j]], dtype=np.complex128)
    sigma_less = np.array([[0.0 + 0.5j]], dtype=np.complex128)
    sigma_greater = np.array([[0.0 - 0.8j]], dtype=np.complex128)

    g_less = lesser_from_retarded_sigma(g_r, sigma_less, g_a)
    g_greater = greater_from_retarded_sigma(g_r, sigma_greater, g_a)
    g_k = keldysh_from_retarded_sigma(g_r, sigma_keldysh(sigma_less, sigma_greater), g_a)

    assert np.allclose(g_less, g_r @ sigma_less @ g_a)
    assert np.allclose(g_greater, g_r @ sigma_greater @ g_a)
    assert np.allclose(g_k, g_less + g_greater)


def test_symbolic_retarded_sigma_builders_match_expected_products():
    gr, ga, sl, sg, sk = sp.symbols('gr ga sl sg sk')

    assert sp.simplify(lesser_from_retarded_sigma_symbolic(gr, sl, ga) - gr * sl * ga) == 0
    assert sp.simplify(greater_from_retarded_sigma_symbolic(gr, sg, ga) - gr * sg * ga) == 0
    assert sp.simplify(keldysh_from_retarded_sigma_symbolic(gr, sk, ga) - gr * sk * ga) == 0


def test_keldysh_self_energy_from_lead_matches_lead_components():
    gamma = np.diag([0.5, 0.2])
    lead = LeadSelfEnergy.wide_band(gamma, mu=0.1, temperature=0.0, name='L')
    sigma = KeldyshSelfEnergy.from_lead(lead)

    assert np.allclose(sigma.sigma_retarded(0.0), lead.sigma_retarded(0.0))
    assert np.allclose(sigma.sigma_lesser(0.0), lead.sigma_lesser(0.0))
    assert np.allclose(sigma.sigma_greater(0.0), lead.sigma_greater(0.0))
    assert np.allclose(sigma.sigma_keldysh(0.0), sigma.sigma_lesser(0.0) + sigma.sigma_greater(0.0))


def test_keldysh_transport_view_matches_matrix_transport_components_and_currents():
    device = SpinfulSingleSite(eps_up=0.1, eps_down=0.1, spin_flip=0.0)
    left = LeadSelfEnergy.wide_band(np.diag([0.5, 0.5]), mu=0.2)
    right = LeadSelfEnergy.wide_band(np.diag([0.4, 0.4]), mu=-0.2)
    transport = device.transport(left, right)
    kview = transport.keldysh_view()
    omega_grid = np.linspace(-6.0, 6.0, 4001)

    omega0 = 0.0
    assert np.allclose(kview.sigma_lesser(omega0, lead='total'), transport.sigma_lesser_total(omega0))
    assert np.allclose(kview.sigma_greater(omega0, lead='total'), transport.sigma_greater_total(omega0))
    assert np.allclose(kview.lesser(omega0), transport.lesser(omega0))
    assert np.allclose(kview.greater(omega0), transport.greater(omega0))
    assert np.allclose(kview.keldysh(omega0), kview.lesser(omega0) + kview.greater(omega0))

    current_landauer = transport.landauer_current(omega_grid, mu_left=0.2, mu_right=-0.2)
    current_mw = kview.meir_wingreen_current(omega_grid, lead='left')
    current_existing = transport.current_from_keldysh(omega_grid, lead='left')

    assert abs(current_mw - current_existing) < 1e-10
    assert abs(current_mw - current_landauer) < 1e-3


def test_meir_wingreen_current_numeric_helper_matches_density_integration():
    device = SpinfulSingleSite(eps_up=0.0, eps_down=0.3, spin_flip=0.0)
    left = LeadSelfEnergy.wide_band(np.diag([0.6, 0.3]), mu=0.25)
    right = LeadSelfEnergy.wide_band(np.diag([0.4, 0.2]), mu=-0.1)
    kview = KeldyshTransportView(device.transport(left, right))
    omega_grid = np.linspace(-6.0, 6.0, 2001)

    sigma_less = np.array([kview.sigma_lesser(float(omega), lead='left') for omega in omega_grid], dtype=np.complex128)
    sigma_greater = np.array([kview.sigma_greater(float(omega), lead='left') for omega in omega_grid], dtype=np.complex128)
    g_less = np.array([kview.lesser(float(omega)) for omega in omega_grid], dtype=np.complex128)
    g_greater = np.array([kview.greater(float(omega)) for omega in omega_grid], dtype=np.complex128)

    helper = meir_wingreen_current(omega_grid, sigma_less, sigma_greater, g_less, g_greater)
    direct = kview.meir_wingreen_current(omega_grid, lead='left')
    density0 = meir_wingreen_current_density(sigma_less[1000], sigma_greater[1000], g_less[1000], g_greater[1000])

    assert np.isfinite(density0)
    assert abs(helper - direct) < 1e-10


def test_meir_wingreen_spin_currents_match_existing_spin_keldysh_layer():
    device = SpinfulSingleSite(eps_up=0.05, eps_down=0.35, spin_flip=0.0)
    left = LeadSelfEnergy.wide_band(np.diag([0.7, 0.2]), mu=0.3)
    right = LeadSelfEnergy.wide_band(np.diag([0.5, 0.1]), mu=-0.1)
    transport = device.transport(left, right)
    kview = transport.keldysh_view()
    omega_grid = np.linspace(-6.0, 6.0, 3001)

    spin_mw = kview.meir_wingreen_spin_current(omega_grid, lead='left', axis='z')
    spin_existing = transport.spin_current_from_keldysh(omega_grid, lead='left', axis='z')
    vector = kview.meir_wingreen_spin_current_vector(omega_grid, lead='left')

    assert abs(spin_mw - spin_existing) < 1e-10
    assert set(vector) == {'x', 'y', 'z'}
    assert all(np.isfinite(value) for value in vector.values())



def test_equilibrium_self_energy_reconstructs_lesser_and_greater_from_retarded():
    gamma = 0.4

    def sigma_r(omega: float) -> np.ndarray:
        return np.array([[-0.5j * gamma]], dtype=np.complex128)

    sigma = KeldyshSelfEnergy.equilibrium_from_retarded(
        dim=1,
        sigma_retarded_fn=sigma_r,
        mu=0.0,
        temperature=0.0,
        name='eq',
    )

    assert np.allclose(sigma.gamma(-1.0), np.array([[gamma]], dtype=np.complex128))
    assert np.allclose(sigma.sigma_lesser(-1.0), np.array([[1j * gamma]], dtype=np.complex128))
    assert np.allclose(sigma.sigma_greater(-1.0), np.zeros((1, 1), dtype=np.complex128))
    assert np.allclose(sigma.sigma_lesser(1.0), np.zeros((1, 1), dtype=np.complex128))
    assert np.allclose(sigma.sigma_greater(1.0), np.array([[-1j * gamma]], dtype=np.complex128))


def test_sampled_self_energy_interpolates_retarded_lesser_and_greater_components():
    omega_grid = np.array([-1.0, 1.0], dtype=float)
    sigma_r_values = np.array([
        [[-0.2j]],
        [[-0.6j]],
    ], dtype=np.complex128)
    sigma_l_values = np.array([
        [[0.1j]],
        [[0.3j]],
    ], dtype=np.complex128)
    sigma_g_values = np.array([
        [[-0.4j]],
        [[-0.8j]],
    ], dtype=np.complex128)

    sigma = KeldyshSelfEnergy.sampled(
        omega_grid,
        sigma_r_values,
        sigma_lesser_values=sigma_l_values,
        sigma_greater_values=sigma_g_values,
        name='sampled',
    )

    assert np.allclose(sigma.sigma_retarded(0.0), np.array([[-0.4j]], dtype=np.complex128))
    assert np.allclose(sigma.sigma_lesser(0.0), np.array([[0.2j]], dtype=np.complex128))
    assert np.allclose(sigma.sigma_greater(0.0), np.array([[-0.6j]], dtype=np.complex128))


def test_combined_and_extra_self_energies_modify_total_keldysh_problem():
    device = SpinfulSingleSite(eps_up=0.1, eps_down=0.1, spin_flip=0.0)
    left = LeadSelfEnergy.wide_band(np.diag([0.5, 0.5]), mu=0.2)
    right = LeadSelfEnergy.wide_band(np.diag([0.4, 0.4]), mu=-0.2)
    base = device.transport(left, right).keldysh_view()

    omega_grid = np.array([-1.0, 1.0], dtype=float)
    sigma_r_values = np.array([
        [[-0.05j, 0.0], [0.0, 0.0]],
        [[-0.15j, 0.0], [0.0, 0.0]],
    ], dtype=np.complex128)
    sampled = KeldyshSelfEnergy.sampled(
        omega_grid,
        sigma_r_values,
        mu=0.0,
        temperature=0.0,
        name='sampled_int',
    )
    extra_eq = KeldyshSelfEnergy.equilibrium_from_retarded(
        dim=2,
        sigma_retarded_fn=lambda omega: np.array([[0.0, 0.0], [0.0, -0.05j]], dtype=np.complex128),
        mu=0.0,
        temperature=0.0,
        name='eq_int',
    )
    combined = KeldyshSelfEnergy.combine(sampled, extra_eq, name='combined_int')
    dressed = base.with_self_energy(sampled, extra_eq)

    omega0 = 0.0
    expected_total = base.sigma_retarded(omega0, lead='total') + combined.sigma_retarded(omega0)
    assert dressed.sigma_interactions() is not None
    assert np.allclose(dressed.sigma_interactions().sigma_retarded(omega0), combined.sigma_retarded(omega0))
    assert np.allclose(dressed.sigma_retarded(omega0, lead='interaction'), combined.sigma_retarded(omega0))
    assert np.allclose(dressed.sigma_retarded(omega0, lead='total'), expected_total)
    assert not np.allclose(dressed.retarded(omega0), base.retarded(omega0))
    assert np.isfinite(dressed.meir_wingreen_current(np.linspace(-6.0, 6.0, 801), lead='left'))
