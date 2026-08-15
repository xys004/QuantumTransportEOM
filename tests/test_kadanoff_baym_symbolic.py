from __future__ import annotations

import sympy as sp

from quantum_transport import (
    electron_boson_scba_symbolic,
    kadanoff_baym_collision_integral_symbolic,
    kadanoff_baym_continuity_symbolic,
    kadanoff_baym_dyson_equations,
    kadanoff_baym_mixed_equations,
    langreth_two_time_convolution_symbolic,
    hubbard_second_born_self_energy_matsubara_symbolic,
    time_convolution_symbolic,
)


def test_symbolic_two_time_convolution_and_langreth_rules() -> None:
    t, tp = sp.symbols("t t_prime", real=True)
    ar, aa, al, ag = (sp.Function(name) for name in ("A_r", "A_a", "A_l", "A_g"))
    br, ba, bl, bg = (sp.Function(name) for name in ("B_r", "B_a", "B_l", "B_g"))
    convolution = time_convolution_symbolic(ar, br, t, tp)
    assert "A_r(t, tau)*B_r(tau, t_prime)" in str(convolution)
    result = langreth_two_time_convolution_symbolic(
        {"r": ar, "a": aa, "<": al, ">": ag},
        {"r": br, "a": ba, "<": bl, ">": bg},
        t,
        tp,
    )
    assert "A_r(t, tau)*B_l(tau, t_prime)" in str(result["<"])
    assert "A_l(t, tau)*B_a(tau, t_prime)" in str(result["<"])


def test_symbolic_kadanoff_baym_dyson_equations_keep_initial_lesser_term() -> None:
    equations = kadanoff_baym_dyson_equations(
        bare_retarded=sp.Function("g_r"),
        bare_lesser=sp.Function("g_l"),
        self_energy_retarded=sp.Function("S_r"),
        self_energy_lesser=sp.Function("S_l"),
        self_energy_advanced=sp.Function("S_a"),
    )
    lesser = str(equations["lesser"])
    assert "g_l(t, t_prime)" in lesser
    assert "S_l(tau, nu)" in lesser
    assert "G_lesser(nu, t_prime)" in lesser


def test_symbolic_electron_boson_scba_shift_rules() -> None:
    omega, omega0, n, vertex = sp.symbols("omega omega0 n vertex")
    formulas = electron_boson_scba_symbolic(
        energy=omega,
        boson_frequency=omega0,
        boson_occupation=n,
        coupling=vertex,
    )
    expected = vertex**2 * (
        n * sp.Function("G_lesser")(omega - omega0)
        + (n + 1) * sp.Function("G_lesser")(omega + omega0)
    )
    assert sp.simplify(formulas["lesser"] - expected) == 0


def test_symbolic_mixed_kadanoff_baym_equations_keep_vertical_measure_and_order() -> None:
    equations = kadanoff_baym_mixed_equations(
        self_energy_retarded=sp.Function("S_r", commutative=False),
        self_energy_mixed=sp.Function("S_rceil", commutative=False),
        self_energy_matsubara=sp.Function("S_M", commutative=False),
        self_energy_advanced=sp.Function("S_a", commutative=False),
        self_energy_lmixed=sp.Function("S_lceil", commutative=False),
    )
    rceil = str(equations["rceil"])
    lceil = str(equations["lceil"])
    assert "Integral(S_r(t, t_prime)*G_rceil(t_prime, tau)" in rceil
    assert "- I*Integral(S_rceil(t, tau_prime)*G_M(tau_prime, tau)" in rceil
    assert "Integral(G_lceil(tau, t_prime)*S_a(t_prime, t)" in lceil
    assert "-I*Integral(G_M(tau, tau_prime)*S_lceil(tau_prime, t)" in lceil


def test_symbolic_hubbard_matsubara_closure_keeps_sign_order_and_u_scaling() -> None:
    formula = hubbard_second_born_self_energy_matsubara_symbolic()
    expression = str(formula["matsubara"])
    assert expression.startswith("-U**2*")
    assert "G_M(0, tau, tau_prime)" in expression
    assert "G_M(1, tau_prime, tau)" in expression
    assert formula["u_scaling"] == 2


def test_symbolic_collision_integral_keeps_all_four_ordered_terms() -> None:
    formulas = kadanoff_baym_collision_integral_symbolic()
    collision = str(formulas["collision"])
    assert "G_r(t, tau)*Sigma_l(tau, t_prime)" in collision
    assert "G_lesser(t, tau)*Sigma_a(tau, t_prime)" in collision
    assert "Sigma_r(t, tau)*G_lesser(tau, t_prime)" in collision
    # The fourth term is Sigma^< * G^a.  This assertion previously demanded
    # G_r, pinning the implementation against its own docstring and against
    # two_time_kbe_collision_integral.
    assert "Sigma_l(t, tau)*G_a(tau, t_prime)" in collision
    assert "equal_time_collision" not in collision


def test_symbolic_continuity_projects_charge_or_spin_without_claiming_conservation() -> None:
    spin_z = sp.Symbol("S_z", commutative=False)
    formulas = kadanoff_baym_continuity_symbolic(
        observable=spin_z,
        initial_correlation_source=sp.Function("I_ic")(sp.Symbol("t", real=True)),
    )
    assert "Derivative(-I*G_lesser(t, t), t)" in str(formulas["continuity"])
    assert "Tr(S_z*" in str(formulas["observable_continuity"])
    assert "I_ic(t)" in str(formulas["initial_correlation_source"])


def test_collision_integral_fourth_term_uses_the_advanced_green_function():
    """``C = G^r S^< + G^< S^a - S^r G^< - S^< G^a``.

    The fourth ordered convolution contracted the *retarded* Green function
    instead of the advanced one, contradicting both the docstring and the
    numerical kernel in ``two_time_kbe_collision_integral``.  There was no
    ``green_advanced`` argument, so the advanced symbol was simply unavailable.
    """

    from quantum_transport import (
        kadanoff_baym_collision_integral_symbolic,
        kadanoff_baym_continuity_symbolic,
    )

    time, time_prime, integration = sp.symbols("t t_prime t_bar", real=True)
    green_retarded = sp.Function("GR", commutative=False)
    green_advanced = sp.Function("GA", commutative=False)
    green_lesser = sp.Function("GL", commutative=False)
    sigma_retarded = sp.Function("SR", commutative=False)
    sigma_lesser = sp.Function("SL", commutative=False)
    sigma_advanced = sp.Function("SA", commutative=False)

    terms = kadanoff_baym_collision_integral_symbolic(
        green_retarded=green_retarded,
        green_advanced=green_advanced,
        green_lesser=green_lesser,
        self_energy_retarded=sigma_retarded,
        self_energy_lesser=sigma_lesser,
        self_energy_advanced=sigma_advanced,
        time=time,
        time_prime=time_prime,
        integration_time=integration,
    )

    expected_pairs = {
        "green_retarded_lesser": (green_retarded, sigma_lesser),
        "green_lesser_advanced": (green_lesser, sigma_advanced),
        "self_energy_retarded_lesser": (sigma_retarded, green_lesser),
        "self_energy_lesser_advanced": (sigma_lesser, green_advanced),
    }
    for key, (left, right) in expected_pairs.items():
        integrand = terms[key].function
        assert integrand == left(time, integration) * right(integration, time_prime), key

    # The signed combination must match the package convention exactly.
    expected_collision = (
        terms["green_retarded_lesser"]
        + terms["green_lesser_advanced"]
        - terms["self_energy_retarded_lesser"]
        - terms["self_energy_lesser_advanced"]
    )
    assert sp.expand(terms["collision"] - expected_collision) == 0

    # The continuity generator must inherit the corrected kernel.
    continuity = kadanoff_baym_continuity_symbolic(
        green_retarded=green_retarded,
        green_advanced=green_advanced,
        green_lesser=green_lesser,
        self_energy_retarded=sigma_retarded,
        self_energy_lesser=sigma_lesser,
        self_energy_advanced=sigma_advanced,
        time=time,
        time_prime=time_prime,
        integration_time=integration,
    )
    assert "GA(" in str(continuity["collision_rate"])
    assert "GR(t_bar, t)" not in str(continuity["collision_rate"])
