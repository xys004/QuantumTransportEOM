import numpy as np
import sympy as sp

from quantum_transport import AndersonImpurity, BosonicHarmonicMode, FermionicSingleLevel, b, bd, f, fd, n


def test_short_operator_aliases_are_intuitive():
    assert str(f(0).doit()) == str(FermionicSingleLevel(eps=sp.Symbol("eps")).operators["d"].doit())
    assert str(fd(0).doit()) == str(f(0).dag().doit())
    assert str(b(0).doit()) == str(BosonicHarmonicMode(omega0=sp.Symbol("omega0")).operators["b"].doit())
    assert str(bd(0).doit()) == str(b(0).dag().doit())
    assert str(n(0).doit()) == str(fd(0).doit() * f(0).doit())


def test_high_level_single_level_green_function_is_clean():
    eps, omega, eta = sp.symbols("eps omega eta")
    model = FermionicSingleLevel(eps=eps)

    g_ret = model.gf().retarded(omega=omega, eta=eta)

    assert sp.simplify(g_ret - 1 / (omega + sp.I * eta - eps)) == 0


def test_high_level_single_level_spectral_function_has_lorentzian_form():
    eps, omega, eta = sp.symbols("eps omega eta", real=True)
    model = FermionicSingleLevel(eps=eps)

    spectral = model.gf().spectral_function(omega=omega, eta=eta)
    density = model.gf().spectral_density(omega=omega, eta=eta)
    expected_spectral = 2 * eta / ((omega - eps) ** 2 + eta**2)
    expected_density = eta / (sp.pi * ((omega - eps) ** 2 + eta**2))

    assert sp.simplify(spectral - expected_spectral) == 0
    assert sp.simplify(density - expected_density) == 0


def test_high_level_single_level_lesser_and_greater_satisfy_equilibrium_identity():
    eps, omega, eta, mu, T = sp.symbols("eps omega eta mu T", real=True, positive=True)
    model = FermionicSingleLevel(eps=eps)

    g_less = model.gf().lesser(omega=omega, eta=eta, mu=mu, temperature=T)
    g_greater = model.gf().greater(omega=omega, eta=eta, mu=mu, temperature=T)
    g_ret = model.gf().retarded(omega=omega, eta=eta)
    g_adv = model.gf().advanced(omega=omega, eta=eta)

    assert sp.simplify((g_greater - g_less) - (g_ret - g_adv)) == 0


def test_high_level_bosonic_lesser_uses_bose_distribution():
    omega0, omega, eta, mu, T = sp.symbols("omega0 omega eta mu T", real=True, positive=True)
    model = BosonicHarmonicMode(omega0=omega0)

    g_less = model.gf().lesser(omega=omega, eta=eta, mu=mu, temperature=T)
    n_be = 1 / (sp.exp((omega - mu) / T) - 1)
    expected = n_be * (model.gf().advanced(omega=omega, eta=eta) - model.gf().retarded(omega=omega, eta=eta))

    assert sp.simplify(g_less - expected) == 0


def test_high_level_single_level_numeric_occupation_matches_lesser_integral():
    omega = sp.Symbol("omega", real=True)
    grid = np.linspace(-8.0, 8.0, 4001)
    model = FermionicSingleLevel(eps=-2.0)

    occ = model.gf().occupation(
        omega_symbol=omega,
        omega_grid=grid,
        eta=0.05,
        mu=0.0,
        temperature=0.0,
    )
    g_less = model.gf().lesser_values(
        omega_symbol=omega,
        omega_grid=grid,
        eta=0.05,
        mu=0.0,
        temperature=0.0,
    )
    occ_from_lesser = float(np.trapezoid(g_less / (2j * np.pi), grid).real)

    assert abs(occ - occ_from_lesser) < 5e-3
    assert 0.9 < occ < 1.05


def test_high_level_anderson_hubbard_i_api_matches_two_pole_form():
    eps, U, omega, eta = sp.symbols("eps U omega eta")
    n_down_avg = sp.symbols("n_down_avg")
    model = AndersonImpurity(eps=eps, U=U)

    g_ret = model.gf("up").retarded(
        omega=omega,
        eta=eta,
        method="hubbard_i",
        occupations={"down": n_down_avg},
    )

    expected = (1 - n_down_avg) / (omega + sp.I * eta - eps) + n_down_avg / (omega + sp.I * eta - eps - U)
    assert sp.simplify(g_ret - expected) == 0


def test_high_level_anderson_basis_expansion_feels_natural():
    eps, U = sp.symbols("eps U")
    model = AndersonImpurity(eps=eps, U=U)

    expanded = model.eom_basis().expand(levels=1)
    analysis = model.eom_basis().analyze(levels=1)

    assert len(expanded) == 4
    assert analysis.success is True
    assert analysis.is_closed is True


def test_high_level_anderson_self_consistent_occupations_converge_near_half_filling():
    omega = sp.Symbol("omega", real=True)
    grid = np.linspace(-10.0, 10.0, 5001)
    model = AndersonImpurity(eps=-1.0, U=2.0)

    result = model.self_consistent_occupations(
        omega_symbol=omega,
        omega_grid=grid,
        eta=0.08,
        method="hubbard_i",
        mu=0.0,
        temperature=0.0,
        initial={"up": 0.4, "down": 0.6},
        mixing=0.8,
        tol=1e-4,
        max_iter=200,
    )

    assert result.converged is True
    assert abs(result.occupations["up"] - 0.5) < 0.05
    assert abs(result.occupations["down"] - 0.5) < 0.05


def test_advanced_green_does_not_flip_coefficients_equal_to_eta():
    """``G^r.subs(eta, -eta)`` rewrites every occurrence of that value.

    A level energy numerically equal to the broadening had its sign flipped
    too, moving the pole from ``+eps`` to ``-eps``.  The advanced branch must
    come from the resolvent evaluated at ``-eta``, not from a substitution.
    """

    import sympy as sp

    from quantum_transport import CustomModel, n

    omega = sp.Symbol("omega", real=True)
    for level, eta in ((0.01, 0.01), (0.02, 0.01)):
        view = CustomModel(level * n("d")).gf("c_d")
        advanced = sp.simplify(view.advanced(omega=omega, eta=eta))
        expected = sp.simplify(1 / (omega - level - sp.I * eta))
        assert sp.simplify(advanced - expected) == 0, f"level={level}, eta={eta}"

        spectral = sp.simplify(view.spectral_function(omega=omega, eta=eta))
        expected_spectral = sp.simplify(
            sp.I * (1 / (omega - level + sp.I * eta) - 1 / (omega - level - sp.I * eta))
        )
        assert sp.simplify(spectral - expected_spectral) == 0
