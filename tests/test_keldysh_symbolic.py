import sympy as sp

from quantum_transport import (
    KeldyshExpression,
    advanced_from_lesser_greater,
    contour_delta,
    contour_green_from_lesser_greater,
    contour_heaviside,
    dyson_lesser_stationary,
    dyson_retarded,
    dyson_retarded_from_level,
    keldysh_component_from_lesser_greater,
    keldysh_leq,
    keldysh_system,
    kgf,
    kt,
    langreth_convolution,
    langreth_double_convolution,
    lorentzian_density,
    meir_wingreen_current_symbolic,
    retarded_from_lesser_greater,
    stationary_population,
    two_terminal_wide_band_current_symbolic,
)


def test_contour_ordering_heaviside_and_delta_follow_notes():
    t, tp = sp.symbols("t tp", real=True)
    t_minus = kt(t, "-")
    tp_minus = kt(tp, "-")
    t_plus = kt(t, "+")
    tp_plus = kt(tp, "+")

    assert keldysh_leq(t_minus, tp_plus) is sp.S.true
    assert keldysh_leq(t_plus, tp_minus) is sp.S.false
    assert contour_heaviside(t_minus, tp_plus) == 0
    assert contour_heaviside(t_plus, tp_minus) == 1
    assert contour_heaviside(t_minus, tp_minus) == sp.Heaviside(t - tp)
    assert contour_heaviside(t_plus, tp_plus) == sp.Heaviside(tp - t)
    assert contour_delta(t_minus, tp_minus) == sp.DiracDelta(t - tp)
    assert contour_delta(t_plus, tp_plus) == -sp.DiracDelta(t - tp)


def test_contour_green_and_retarded_advanced_relations():
    t, tp, gp, gm = sp.symbols("t tp Ggreater Glesser")
    expr = contour_green_from_lesser_greater(gp, gm, kt(t, "-"), kt(tp, "-"))

    assert sp.simplify(expr - (sp.Heaviside(t - tp) * gp + sp.Heaviside(tp - t) * gm)) == 0
    assert sp.simplify(retarded_from_lesser_greater(gp, gm, t, tp) - sp.Heaviside(t - tp) * (gp - gm)) == 0
    assert sp.simplify(advanced_from_lesser_greater(gp, gm, t, tp) + sp.Heaviside(tp - t) * (gp - gm)) == 0
    assert keldysh_component_from_lesser_greater(gp, gm) == gp + gm


def test_keldysh_function_components_are_readable_sympy_functions():
    t, tp = sp.symbols("t tp")
    g = kgf("G")

    assert str(g.retarded(t, tp).func) == "G^r"
    assert str(g.advanced(t, tp).func) == "G^a"
    assert str(g.lesser(t, tp).func) == "G^<"
    assert str(g.greater(t, tp).func) == "G^>"
    assert str(g.keldysh(t, tp).func) == "G^K"


def test_langreth_rules_for_single_and_double_convolutions():
    ar, aa, al, ag, br, ba, bl, bg = sp.symbols("ar aa al ag br ba bl bg")
    cr = langreth_convolution({"r": ar, "a": aa, "<": al, ">": ag}, {"r": br, "a": ba, "<": bl, ">": bg})

    assert cr["r"] == ar * br
    assert cr["a"] == aa * ba
    assert cr["<"] == ar * bl + al * ba
    assert cr[">"] == ar * bg + ag * ba

    wr, wa, wl, wg, xr, xa, xl, xg, yr, ya, yl, yg = sp.symbols("wr wa wl wg xr xa xl xg yr ya yl yg")
    z = langreth_double_convolution(
        {"r": wr, "a": wa, "<": wl, ">": wg},
        {"r": xr, "a": xa, "<": xl, ">": xg},
        {"r": yr, "a": ya, "<": yl, ">": yg},
    )

    assert z["r"] == wr * xr * yr
    assert z["a"] == wa * xa * ya
    assert z["<"] == wr * xr * yl + wr * xl * ya + wl * xa * ya
    assert z[">"] == wr * xr * yg + wr * xg * ya + wg * xa * ya


def test_stationary_dyson_and_level_green_function_relations():
    omega, xi, sigma_r, g_r, sigma_l, g_a = sp.symbols("omega xi Sigma_r g_r Sigma_l G_a")

    assert sp.simplify(dyson_retarded(g_r, sigma_r) - g_r / (1 - g_r * sigma_r)) == 0
    assert sp.simplify(dyson_retarded_from_level(omega, xi, sigma_r) - 1 / (omega - xi - sigma_r)) == 0
    assert sp.simplify(dyson_lesser_stationary(g_r, sigma_l, g_a) - g_r * sigma_l * g_a) == 0


def test_stationary_population_and_current_are_symbolic_integrals():
    omega, g_less, sigma_r, sigma_l, g_a, e, hbar = sp.symbols("omega G_less Sigma_r Sigma_less G_a e hbar")

    population = stationary_population(g_less, omega)
    current = meir_wingreen_current_symbolic(sigma_r, g_less, sigma_l, g_a, omega, charge=e, hbar=hbar)

    assert isinstance(population, sp.Integral)
    assert population.function == g_less / (2 * sp.pi * sp.I)
    assert isinstance(current, sp.Integral)
    assert current.limits[0][0] == omega


def test_wide_band_density_and_two_terminal_current_match_note_structure():
    omega, xi, gamma, gl, gr, fl, fr, e, h = sp.symbols("omega xi Gamma Gamma_L Gamma_R f_L f_R e h", positive=True)
    rho = lorentzian_density(omega, xi, gamma)
    current = two_terminal_wide_band_current_symbolic(omega, rho, gl, gr, fl, fr, charge=e, h=h)

    assert sp.simplify(rho - (gamma / sp.pi) / ((omega - xi) ** 2 + gamma**2)) == 0
    assert isinstance(current, sp.Integral)
    assert current.limits[0][0] == omega


def test_keldysh_system_provides_intuitive_object_api():
    omega, xi = sp.symbols("omega xi")
    k = keldysh_system(omega)
    G = k.green("G")
    Sigma = k.self_energy("Sigma")

    g_less = G.lesser(omega)
    g_ret = G.retarded(omega)
    sigma_less = Sigma.lesser(omega)
    product = k.langreth(
        {"r": g_ret, "a": G.advanced(omega), "<": g_less, ">": G.greater(omega)},
        {"r": Sigma.retarded(omega), "a": Sigma.advanced(omega), "<": sigma_less, ">": Sigma.greater(omega)},
    )
    dyson = k.dyson_retarded(xi, Sigma.retarded(omega))
    lesser = k.dyson_lesser(dyson, sigma_less, G.advanced(omega))

    assert isinstance(g_less, KeldyshExpression)
    assert str(g_less.doit().func) == "G^<"
    assert product["<"].doit() == G.retarded(omega).doit() * Sigma.lesser(omega).doit() + G.lesser(omega).doit() * Sigma.advanced(omega).doit()
    assert sp.simplify(dyson.doit() - 1 / (omega - xi - Sigma.retarded(omega).doit())) == 0
    assert sp.simplify(lesser.doit() - dyson.doit() * sigma_less.doit() * G.advanced(omega).doit()) == 0


def test_keldysh_system_population_and_current_return_observables():
    omega, gr, ga, sigma_l, sigma_r, e, hbar = sp.symbols("omega gr ga Sigma_l Sigma_r e hbar")
    k = keldysh_system(omega)
    g_less = k.dyson_lesser(gr, sigma_l, ga)

    population = k.population(g_less)
    current = k.meir_wingreen_current(sigma_r, g_less, sigma_l, ga, charge=e, hbar=hbar)

    assert isinstance(population.doit(), sp.Integral)
    assert isinstance(current.doit(), sp.Integral)
    assert population.latex()
    assert current.latex()
