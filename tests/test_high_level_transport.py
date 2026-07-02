import numpy as np
import sympy as sp

from quantum_transport import AndersonImpurity, FermionicSingleLevel, KeldyshSelfEnergy, SpinfulSingleSite


def test_high_level_transport_transmission_connects_model_and_observables():
    eps, omega, eta, gamma_l, gamma_r = sp.symbols("eps omega eta gamma_l gamma_r")
    model = FermionicSingleLevel(eps=eps)

    t_expr = model.transport(gamma_l, gamma_r).transmission(omega=omega, eta=eta)
    expected = gamma_l * gamma_r / ((omega - eps) ** 2 + eta**2)

    assert sp.simplify(t_expr - expected) == 0


def test_high_level_transport_conductance_is_available_directly_from_model():
    eps, omega, eta, gamma_l, gamma_r, mu = sp.symbols("eps omega eta gamma_l gamma_r mu")
    model = FermionicSingleLevel(eps=eps)

    g_expr = model.transport(gamma_l, gamma_r).conductance(omega=omega, eta=eta, mu=mu)
    expected = gamma_l * gamma_r / (2 * sp.pi * ((mu - eps) ** 2 + eta**2))

    assert sp.simplify(g_expr - expected) == 0


def test_high_level_transport_channel_dictionary_feels_natural_for_anderson_spin_channels():
    eps, U, omega, eta = sp.symbols("eps U omega eta")
    gamma_l_up, gamma_l_down, gamma_r_up, gamma_r_down = sp.symbols("gamma_l_up gamma_l_down gamma_r_up gamma_r_down")
    n_down_avg = sp.symbols("n_down_avg")
    model = AndersonImpurity(eps=eps, U=U)

    t_up = model.transport(
        {"up": gamma_l_up, "down": gamma_l_down},
        {"up": gamma_r_up, "down": gamma_r_down},
    ).transmission(
        omega=omega,
        eta=eta,
        channel="up",
        method="hubbard_i",
        occupations={"down": n_down_avg},
    )

    g_up = model.gf("up").retarded(omega=omega, eta=eta, method="hubbard_i", occupations={"down": n_down_avg})
    g_up_a = model.gf("up").advanced(omega=omega, eta=eta, method="hubbard_i", occupations={"down": n_down_avg})
    expected = gamma_l_up * gamma_r_up * g_up * g_up_a

    assert sp.simplify(t_up - expected) == 0


def test_open_anderson_u_zero_matches_noninteracting_open_device_results():
    model = AndersonImpurity(eps=0.1, U=0.0)
    open_view = model.open({"up": 0.5, "down": 0.5}, {"up": 0.4, "down": 0.4}, mu_left=0.2, mu_right=-0.2)
    device = SpinfulSingleSite(eps_up=0.1, eps_down=0.1, spin_flip=0.0).transport(
        open_view.left_lead,
        open_view.right_lead,
    )
    omega_grid = np.linspace(-6.0, 6.0, 2001)

    assert np.allclose(open_view.retarded(0.0, method="hartree_fock"), device.retarded(0.0))
    assert abs(open_view.transmission(0.0, method="hartree_fock") - device.transmission(0.0)) < 1e-10
    assert abs(open_view.meir_wingreen_current(omega_grid, method="hartree_fock") - device.meir_wingreen_current(omega_grid)) < 1e-10


def test_open_anderson_hubbard_i_retarded_matches_expected_lead_dressed_two_pole_result():
    model = AndersonImpurity(eps=0.2, U=1.5)
    open_view = model.open({"up": 0.6, "down": 0.3}, {"up": 0.2, "down": 0.1})
    occupations = {"up": 0.4, "down": 0.3}
    omega = 0.15

    sigma_up = open_view.sigma_retarded(omega, lead="total")[0, 0]
    z = omega - 0.2 - sigma_up
    expected = (1.0 - occupations["down"]) / z + occupations["down"] / (z - 1.5)
    actual = open_view.gf("up").retarded(omega, method="hubbard_i", occupations=occupations)

    assert abs(actual - expected) < 1e-10


def test_open_anderson_self_consistent_occupations_converge_in_equilibrium():
    model = AndersonImpurity(eps=-1.0, U=2.0)
    open_view = model.open({"up": 0.35, "down": 0.35}, {"up": 0.35, "down": 0.35}, mu_left=0.0, mu_right=0.0)
    omega_grid = np.linspace(-10.0, 10.0, 2001)

    result = open_view.self_consistent_occupations(
        omega_grid,
        eta=0.05,
        method="hartree_fock",
        initial={"up": 0.5, "down": 0.5},
        mixing=0.7,
        tol=1e-4,
        max_iter=200,
    )

    assert result.converged is True
    assert abs(result.occupations["up"] - result.occupations["down"]) < 5e-3
    assert 0.25 < result.occupations["up"] < 0.75


def test_open_anderson_accepts_extra_keldysh_self_energy():
    model = AndersonImpurity(eps=0.1, U=0.0)
    base = model.open({"up": 0.5, "down": 0.5}, {"up": 0.4, "down": 0.4}, mu_left=0.2, mu_right=-0.2)
    extra = KeldyshSelfEnergy.equilibrium_from_retarded(
        dim=2,
        sigma_retarded_fn=lambda omega: np.array([[-0.1j, 0.0], [0.0, -0.02j]], dtype=np.complex128),
        mu=0.0,
        temperature=0.0,
        name="extra",
    )
    dressed = base.with_self_energy(extra)

    assert np.allclose(dressed.sigma_retarded(0.0, lead="extra"), extra.sigma_retarded(0.0))
    assert not np.allclose(dressed.retarded(0.0, method="noninteracting"), base.retarded(0.0, method="noninteracting"))
    assert np.isfinite(dressed.meir_wingreen_current(np.linspace(-6.0, 6.0, 801), method="noninteracting"))


def test_open_anderson_spin_resolved_meir_wingreen_currents_add_to_total_for_z_diagonal_case():
    model = AndersonImpurity(eps=0.0, U=0.0, zeeman=0.3)
    open_view = model.open({"up": 0.6, "down": 0.2}, {"up": 0.4, "down": 0.1}, mu_left=0.25, mu_right=-0.15)
    omega_grid = np.linspace(-6.0, 6.0, 1201)

    total = open_view.meir_wingreen_current(omega_grid, method="noninteracting")
    plus = open_view.spin_resolved_meir_wingreen_current(omega_grid, "+", axis="z", method="noninteracting")
    minus = open_view.spin_resolved_meir_wingreen_current(omega_grid, "-", axis="z", method="noninteracting")
    spin = open_view.spin_meir_wingreen_current(omega_grid, axis="z", method="noninteracting")

    assert abs((plus + minus) - total) < 1e-10
    assert abs((plus - minus) - spin) < 1e-10


def test_open_anderson_zeeman_and_spin_flip_enter_local_hamiltonian():
    model = AndersonImpurity(eps=0.2, U=0.0, zeeman=0.4, spin_flip=0.15j)
    open_view = model.open({"up": 0.5, "down": 0.5}, {"up": 0.4, "down": 0.4})

    expected_h = np.array([[0.4, 0.15j], [-0.15j, 0.0]], dtype=np.complex128)
    assert np.allclose(open_view.local_hamiltonian(), expected_h)

    g_ret = open_view.retarded(0.0, method="noninteracting")
    assert abs(g_ret[0, 1]) > 1e-12


def test_anderson_spin_flip_enters_symbolic_hamiltonian_and_noninteracting_gf():
    omega, eta, t = sp.symbols("omega eta t", real=True)
    model = AndersonImpurity(eps=0, U=0, spin_flip=t)

    assert model.model.metadata["spin_flip"] == t
    assert str(t) in str(model.hamiltonian)

    g_up = model.gf("up").retarded(omega=omega, eta=eta)
    z = omega + sp.I * eta
    expected = z / (z**2 - t**2)
    assert sp.simplify(g_up - expected) == 0

    numeric = AndersonImpurity(eps=0.0, U=0.0, spin_flip=0.25)
    assert np.allclose(
        numeric.open(0.1, 0.1).local_hamiltonian(),
        np.array([[0.0, 0.25], [0.25, 0.0]], dtype=np.complex128),
    )
