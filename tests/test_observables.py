import numpy as np
import sympy as sp

from quantum_transport import (
    ObservableExpr,
    conductance,
    conductance_numeric,
    contour_green_observable,
    imag_part,
    keldysh_dyson_lesser,
    keldysh_meir_wingreen_current,
    keldysh_population,
    keldysh_two_terminal_wide_band_current,
    kgf,
    kt,
    landauer_current,
    landauer_current_numeric,
    landauer_integrand,
    langreth_double_observable,
    langreth_observable,
    lorentzian_density,
    obs,
    real_part,
    spin_projector,
    spin_transmission,
    trace_expr,
    transmission,
)


def test_observable_builder_trace_and_integral_chain_is_composable():
    omega = sp.Symbol("omega", real=True)
    M = sp.Matrix([[omega, 0], [0, 2 * omega]])

    expr = obs(M).trace().integrate(omega, limits=(0, 1), prefactor=2)

    assert isinstance(expr, ObservableExpr)
    assert sp.simplify(expr.doit() - 2 * sp.Integral(3 * omega, (omega, 0, 1))) == 0


def test_transmission_for_scalar_channel_matches_expected_form():
    omega, eta, eps, gamma_l, gamma_r = sp.symbols("omega eta eps gamma_l gamma_r")
    g_r = 1 / (omega + sp.I * eta - eps)
    g_a = 1 / (omega - sp.I * eta - eps)

    t_expr = transmission(g_r, g_a, gamma_l, gamma_r)
    expected = gamma_l * gamma_r / ((omega - eps) ** 2 + eta**2)

    assert sp.simplify(t_expr.doit() - expected) == 0


def test_spin_projector_selects_requested_spin_sector():
    projector = spin_projector("up", ["site0_up", "site0_down", "site1_up", "site1_down"])
    expected = sp.diag(1, 0, 1, 0)
    assert projector == expected


def test_spin_transmission_extracts_spin_flip_channel_from_matrix_green_function():
    g12, g21, gamma_l, gamma_r = sp.symbols("g12 g21 gamma_l gamma_r")
    g_r = sp.Matrix([[0, g12], [g21, 0]])
    g_a = sp.Matrix([[0, g21], [g12, 0]])
    gamma_left = sp.diag(gamma_l, gamma_l)
    gamma_right = sp.diag(gamma_r, gamma_r)

    t_ud = spin_transmission(g_r, g_a, gamma_left, gamma_right, "up", "down", ["up", "down"])
    assert sp.simplify(t_ud.doit() - gamma_l * gamma_r * g12**2) == 0


def test_landauer_integrand_vanishes_at_zero_bias():
    omega, mu, eps = sp.symbols("omega mu eps")
    g_r = 1 / (omega - eps + sp.I)
    g_a = 1 / (omega - eps - sp.I)
    gamma_l = sp.Integer(1)
    gamma_r = sp.Integer(1)

    integrand = landauer_integrand(g_r, g_a, gamma_l, gamma_r, omega, mu, mu)
    assert sp.simplify(integrand.doit()) == 0


def test_landauer_current_builds_symbolic_integral():
    omega, mu_l, mu_r = sp.symbols("omega mu_l mu_r")
    g_r = 1 / (omega + sp.I)
    g_a = 1 / (omega - sp.I)
    current = landauer_current(g_r, g_a, 1, 1, omega, mu_l, mu_r)
    assert isinstance(current, ObservableExpr)
    assert isinstance(current.doit(), (sp.Integral, sp.Mul))


def test_conductance_at_zero_temperature_samples_transmission_at_mu():
    omega, mu = sp.symbols("omega mu")
    expr = omega**2 + 1
    g_expr = conductance(expr, omega=omega, mu=mu, charge=2)
    expected = (4 / (2 * sp.pi)) * (mu**2 + 1)
    assert sp.simplify(g_expr.doit() - expected) == 0


def test_real_and_imag_parts_are_available_for_manual_formulas():
    omega = sp.Symbol("omega", real=True)
    expr = 1 + sp.I * omega
    assert sp.simplify(real_part(expr).doit() - 1) == 0
    assert sp.simplify(imag_part(expr).doit() - omega) == 0


def test_numeric_landauer_current_is_zero_at_zero_bias_for_real_transmission():
    omega_grid = np.linspace(-5.0, 5.0, 2001)
    transmission_values = np.exp(-omega_grid**2)
    current = landauer_current_numeric(transmission_values, omega_grid, 0.0, 0.0)
    assert abs(current) < 1e-12


def test_numeric_conductance_matches_zero_temperature_sampling_rule():
    omega_grid = np.linspace(-5.0, 5.0, 2001)
    transmission_values = 1.0 + omega_grid**2
    value = conductance_numeric(transmission_values, omega_grid, mu=0.5, charge=2.0)
    idx = int(np.argmin(np.abs(omega_grid - 0.5)))
    expected = (4.0 / (2.0 * np.pi)) * transmission_values[idx]
    assert abs(value - expected) < 1e-12


def test_keldysh_contour_green_observable_wraps_symbolic_result():
    t, tp = sp.symbols("t tp", real=True)
    G = kgf("G")

    result = contour_green_observable(G.greater(t, tp), G.lesser(t, tp), kt(t, "-"), kt(tp, "+"))

    assert isinstance(result, ObservableExpr)
    assert result.doit() == G.lesser(t, tp)
    assert "G^{<}" in result.latex()


def test_keldysh_langreth_observable_returns_composable_components():
    ar, aa, al, ag, br, ba, bl, bg = sp.symbols("ar aa al ag br ba bl bg")

    result = langreth_observable(
        {"r": obs(ar), "a": aa, "<": al, ">": ag},
        {"r": br, "a": ba, "<": bl, ">": bg},
    )

    assert isinstance(result["<"], ObservableExpr)
    assert result["r"].doit() == ar * br
    assert result["<"].doit() == ar * bl + al * ba
    assert sp.simplify((result["<"] + result[">"]).doit() - (ar * bl + al * ba + ar * bg + ag * ba)) == 0


def test_keldysh_double_langreth_observable_matches_notes():
    wr, wa, wl, xr, xa, xl, yr, ya, yl = sp.symbols("wr wa wl xr xa xl yr ya yl")

    result = langreth_double_observable(
        {"r": wr, "a": wa, "<": wl, ">": 0},
        {"r": xr, "a": xa, "<": xl, ">": 0},
        {"r": yr, "a": ya, "<": yl, ">": 0},
    )

    expected = wr * xr * yl + wr * xl * ya + wl * xa * ya
    assert isinstance(result["<"], ObservableExpr)
    assert sp.simplify(result["<"].doit() - expected) == 0


def test_keldysh_dyson_population_and_meir_wingreen_are_observable_exprs():
    omega, gr, ga, sigma_l, sigma_r, e, hbar = sp.symbols("omega gr ga Sigma_l Sigma_r e hbar")

    g_less = keldysh_dyson_lesser(gr, sigma_l, ga)
    population = keldysh_population(g_less, omega)
    current = keldysh_meir_wingreen_current(sigma_r, g_less, sigma_l, ga, omega, charge=e, hbar=hbar)

    assert isinstance(g_less, ObservableExpr)
    assert sp.simplify(g_less.doit() - gr * sigma_l * ga) == 0
    assert isinstance(population.doit(), sp.Integral)
    assert population.doit().function == gr * sigma_l * ga / (2 * sp.pi * sp.I)
    assert isinstance(current.doit(), sp.Integral)
    assert current.doit().limits[0][0] == omega


def test_keldysh_wide_band_current_observable_preserves_latex_workflow():
    omega, xi, gamma, gamma_l, gamma_r, e, h = sp.symbols("omega xi Gamma Gamma_L Gamma_R e h", positive=True)
    rho = lorentzian_density(omega, xi, gamma)
    f_l = sp.Function("f_L")(omega)
    f_r = sp.Function("f_R")(omega)

    current = keldysh_two_terminal_wide_band_current(omega, rho, gamma_l, gamma_r, f_l, f_r, charge=e, h=h)

    assert isinstance(current, ObservableExpr)
    assert isinstance(current.doit(), sp.Integral)
    assert "Gamma" in current.latex()
