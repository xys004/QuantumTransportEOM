import sympy as sp
import numpy as np
import pytest

from quantum_transport import (
    BosonicSCBAConfig,
    ElectronBosonSCBAConfig,
    EOMContourResult,
    SelfConsistentClosure,
    annihilate_boson,
    build_eom_hierarchy,
    create_boson,
    f,
    fd,
)


def test_generic_hierarchy_supports_closed_bosonic_mode():
    omega0, omega, eta = sp.symbols("omega0 omega eta")
    hamiltonian = omega0 * create_boson(0) * annihilate_boson(0)

    hierarchy = build_eom_hierarchy({"free_boson": hamiltonian}, max_depth=0)

    assert hierarchy.statistics == "boson"
    assert hierarchy.is_closed
    green = hierarchy.retarded_green(omega, eta)
    assert sp.simplify(green[0, 0] - 1 / (omega + sp.I * eta - omega0)) == 0


def test_generic_hierarchy_supports_mixed_quadratic_hamiltonian():
    epsilon, omega0 = sp.symbols("epsilon omega0")
    hamiltonian = (
        epsilon * fd("d") * f("d")
        + omega0 * create_boson(0) * annihilate_boson(0)
    )

    hierarchy = build_eom_hierarchy({"quadratic": hamiltonian}, max_depth=0)

    assert hierarchy.statistics == "mixed"
    assert hierarchy.is_closed
    assert hierarchy.eom_matrix.shape == (2, 2)


def test_contour_projection_contains_real_and_vertical_langreth_components():
    epsilon, omega0 = sp.symbols("epsilon omega0")
    hamiltonian = (
        epsilon * fd("d") * f("d")
        + omega0 * create_boson(0) * annihilate_boson(0)
    )
    hierarchy = build_eom_hierarchy({"quadratic": hamiltonian}, max_depth=0)

    contour = hierarchy.contour_equations()

    assert len(contour.equations) == 4
    assert len(contour.component("r")) == 4
    assert len(contour.component("lesser")) == 4
    assert len(contour.component("rceil")) == 4
    assert len(contour.component("M")) == 4
    assert any(equation.source != 0 for equation in contour.equations)


def test_langreth_convolution_keeps_vertical_branch_terms():
    components = {
        "r": sp.Function("A_r"),
        "a": sp.Function("A_a"),
        "<": sp.Function("A_lesser"),
        ">": sp.Function("A_greater"),
        "rceil": sp.Function("A_rceil"),
        "lceil": sp.Function("A_lceil"),
        "M": sp.Function("A_M"),
    }
    other = {
        key: sp.Function(f"B_{label}")
        for key, label in {
            "r": "r",
            "a": "a",
            "<": "lesser",
            ">": "greater",
            "rceil": "rceil",
            "lceil": "lceil",
            "M": "M",
        }.items()
    }

    result = EOMContourResult.langreth_convolution(components, other)

    assert set(result) == {"r", "a", "<", ">", "rceil", "lceil", "M"}
    assert any(term.has(sp.Integral) for term in sp.Add.make_args(result["rceil"]))
    assert result["M"].has(sp.Integral)


def test_self_consistent_closure_records_fixed_point_iterations():
    epsilon_up, epsilon_down, interaction = sp.symbols(
        "epsilon_up epsilon_down U", real=True
    )
    alpha = sp.Symbol("alpha", real=True)
    hierarchy = build_eom_hierarchy(
        {
            "one_body": epsilon_up * fd("up") * f("up") + epsilon_down * fd("down") * f("down"),
            "interaction": interaction * fd("up") * f("up") * fd("down") * f("down"),
        },
        max_depth=0,
    )
    rules = {operator: alpha * hierarchy.basis[0] for operator in hierarchy.unresolved_operators}
    closure = SelfConsistentClosure(
        rules=rules,
        initial_values={alpha: 0},
        update=lambda values, green: {alpha: sp.Rational(1, 5)},
        max_iterations=4,
    )

    result = closure.solve(hierarchy, sp.Symbol("omega"), sp.Symbol("eta"))

    assert result.converged
    assert result.values[alpha] == sp.Rational(1, 5)
    assert len(result.iterations) == 2
    assert result.green.shape == (2, 2)


def test_contour_result_feeds_directly_into_two_time_eom_propagation():
    epsilon = sp.Symbol("epsilon", real=True)
    hierarchy = build_eom_hierarchy(
        {"one_body": epsilon * fd("d") * f("d")},
        max_depth=0,
    )
    contour = hierarchy.contour_equations()
    time = [0.0, 0.2, 0.4, 0.6]

    result = contour.propagate_two_time(
        time,
        [[0.25]],
        parameters={epsilon: 0.7},
    )

    assert result.solver == "finite_eom"
    assert not result.used_self_energy
    assert result.retarded.shape == (4, 4, 1, 1)
    assert result.lesser.shape == (4, 4, 1, 1)
    assert result.converged


def test_contour_result_automatically_attaches_kadanoff_baym_dyson():
    epsilon = sp.Symbol("epsilon", real=True)
    hierarchy = build_eom_hierarchy(
        {"one_body": epsilon * fd("d") * f("d")},
        max_depth=0,
    )
    contour = hierarchy.contour_equations()
    time = [0.0, 0.2, 0.4, 0.6]
    sigma_r = np.zeros((4, 4, 1, 1), dtype=complex)
    sigma_l = np.zeros_like(sigma_r)

    result = contour.propagate_two_time(
        time,
        [[0.25]],
        parameters={epsilon: 0.7},
        self_energy_retarded=sigma_r,
        self_energy_lesser=sigma_l,
    )

    assert result.solver == "kadanoff_baym_dyson"
    assert result.used_self_energy
    assert result.converged
    assert np.max(np.abs(result.retarded - result.green.retarded)) == 0.0


def test_contour_result_requires_numeric_coefficients_and_supports_bosonic_sources():
    epsilon = sp.Symbol("epsilon", real=True)
    fermion = build_eom_hierarchy(
        {"one_body": epsilon * fd("d") * f("d")},
        max_depth=0,
    )
    with pytest.raises(ValueError, match="unresolved symbolic coefficients"):
        fermion.contour_equations().propagate_two_time([0.0, 0.1], [[0.2]])

    omega0 = sp.Symbol("omega0", real=True)
    boson = build_eom_hierarchy(
        {"one_body": omega0 * create_boson(0) * annihilate_boson(0)},
        max_depth=0,
    )
    bosonic_result = boson.contour_equations().propagate_two_time(
        [0.0, 0.1], [[2.0]], parameters={omega0: 0.7}
    )
    assert bosonic_result.solver == "finite_eom"
    assert np.allclose(bosonic_result.lesser[0, 0, 0, 0], 2j)

    mixed = build_eom_hierarchy(
        {
            "fermion": epsilon * fd("d") * f("d"),
            "boson": omega0 * create_boson(0) * annihilate_boson(0),
        },
        max_depth=0,
    )
    mixed_result = mixed.contour_equations().propagate_two_time(
        [0.0, 0.1],
        [[0.0, 0.0], [0.0, 0.0]],
        parameters={epsilon: 0.2, omega0: 0.7},
        initial_lesser=np.diag([0.2j, 2.0j]),
    )
    assert mixed_result.solver == "finite_eom"
    assert mixed_result.lesser.shape == (2, 2, 2, 2)


def test_contour_result_runs_automatic_electron_boson_scba():
    epsilon = sp.Symbol("epsilon", real=True)
    hierarchy = build_eom_hierarchy(
        {"one_body": epsilon * fd("d") * f("d")},
        max_depth=0,
    )
    config = ElectronBosonSCBAConfig(
        coupling=np.array([[0.05]], dtype=complex),
        boson_frequency=0.8,
        max_iterations=2,
        dyson_iterations=3,
        tolerance=1e-4,
    )

    result = hierarchy.contour_equations().propagate_two_time(
        np.linspace(0.0, 0.4, 5),
        [[0.2]],
        parameters={epsilon: 0.3},
        electron_boson_scba=config,
    )

    assert result.solver == "self_consistent_born_two_time"
    assert result.green.self_energy_retarded.shape == (5, 5, 1, 1)
    assert result.green.self_energy_lesser.shape == (5, 5, 1, 1)
    assert result.retarded.shape == (5, 5, 1, 1)


def test_pure_bosonic_scba_closes_real_mixed_and_matsubara_branches():
    omega0 = sp.Symbol("omega0", real=True)
    hierarchy = build_eom_hierarchy(
        {"one_body": omega0 * create_boson(0) * annihilate_boson(0)},
        max_depth=0,
    )
    config = BosonicSCBAConfig(
        coupling=np.array([[0.05]], dtype=complex),
        boson_frequency=0.8,
        boson_temperature=0.5,
        cubic_vertex=np.array([[[0.01]]], dtype=complex),
        quartic_vertex=np.array([[[[0.005]]]], dtype=complex),
        max_iterations=1,
        dyson_iterations=1,
        matsubara_iterations=3,
        matsubara_dyson_iterations=3,
        tolerance=1e-3,
        matsubara_tolerance=1e-3,
    )

    result = hierarchy.contour_equations().propagate_two_time(
        np.linspace(0.0, 0.2, 3),
        [[2.0]],
        imaginary_time=np.linspace(0.0, 2.0, 4),
        parameters={omega0: 0.7},
        bosonic_scba=config,
    )

    assert result.solver == "self_consistent_bosonic_scba_contour_two_time"
    assert result.green_rceil.shape == (3, 4, 1, 1)
    assert result.green_lceil.shape == (4, 3, 1, 1)
    assert result.green_matsubara.shape == (4, 4, 1, 1)
    assert result.self_energy_mixed.shape == (3, 4, 1, 1)
    assert result.self_energy_lmixed.shape == (4, 3, 1, 1)
    assert result.self_energy_matsubara.shape == (4, 4, 1, 1)
    assert np.all(np.isfinite(result.self_energy_matsubara))
    assert np.max(np.abs(result.self_energy_mixed)) > 0.0
    assert np.max(np.abs(result.self_energy_matsubara)) > 0.0
    assert result.mixed_adjoint_error < 1e-12


def test_advanced_and_lceil_equations_are_the_adjoint_off_diagonal():
    """Right multiplication by the EOM matrix must use the right-operator index.

    Conjugating ``i d_t G^r_{b,a} = sum_c M[b,c] G^r_{c,a}`` and using
    ``G^a_{ab}(t,t') = conj(G^r_{ba}(t',t))`` with Hermitian ``M`` gives
    ``-i d_t G^a_{row,ri} = sum_c M[c,ri] G^a_{row,c}``.  Indexing ``M`` by
    ``row`` instead coincides only on the diagonal, so a one-operator basis
    or a diagonal-only check cannot see the difference.
    """

    from quantum_transport.eom_contour import contour_eom_from_hierarchy

    e0, e1, hop = sp.symbols("e0 e1 hop", real=True)
    hamiltonian = (
        e0 * fd(0) * f(0)
        + e1 * fd(1) * f(1)
        + hop * (fd(0) * f(1) + fd(1) * f(0))
    )
    hierarchy = build_eom_hierarchy({"chain": hamiltonian}, max_depth=0)
    contour = contour_eom_from_hierarchy(hierarchy)
    matrix = hierarchy.eom_matrix
    dimension = len(hierarchy.basis)
    assert dimension == 2
    # The bug is invisible unless the EOM matrix has distinct off-diagonal and
    # diagonal entries.
    assert matrix[0, 1] != matrix[0, 0]

    time, time_prime = contour.time, contour.time_prime
    imaginary = contour.imaginary_time
    for index, equation in enumerate(contour.equations):
        row, right_index = divmod(index, dimension)

        expected_advanced = -sp.I * sp.Derivative(
            sp.Function(f"G_a_{row}_{right_index}", commutative=False)(time_prime, time),
            time,
        )
        for column in range(dimension):
            expected_advanced -= (
                sp.Function(f"G_a_{row}_{column}", commutative=False)(time_prime, time)
                * matrix[column, right_index]
            )
        assert sp.simplify(equation.components["a"].lhs - expected_advanced) == 0

        expected_lceil = -sp.I * sp.Derivative(
            sp.Function(f"G_lceil_{row}_{right_index}", commutative=False)(imaginary, time),
            time,
        )
        for column in range(dimension):
            expected_lceil -= (
                sp.Function(f"G_lceil_{row}_{column}", commutative=False)(imaginary, time)
                * matrix[column, right_index]
            )
        assert sp.simplify(equation.components["lceil"].lhs - expected_lceil) == 0


class _Green:
    """Module-level stand-in so the facade can be pickled in the test below."""

    time = np.array([0.0, 1.0])
    retarded = advanced = lesser = greater = None
    converged = True
    iterations = 7


def test_propagation_result_facade_survives_copy_and_pickle():
    """``__getattr__`` must not delegate the copy/pickle dunders to ``green``.

    Forwarding them re-enters ``__getattr__`` through ``self.green`` on a
    partially built instance and recurses without bound, which breaks any
    path that pickles a result (multiprocessing ``workers=``) or snapshots
    one with ``deepcopy``.
    """

    import copy
    import pickle

    from quantum_transport.eom_contour import EOMTwoTimePropagationResult

    result = EOMTwoTimePropagationResult(
        contour=None, green=_Green(), solver="finite_eom", used_self_energy=False
    )
    assert result.iterations == 7  # forwarding still works

    duplicate = copy.deepcopy(result)
    assert duplicate.solver == "finite_eom"
    assert duplicate.iterations == 7

    restored = pickle.loads(pickle.dumps(result))
    assert restored.solver == "finite_eom"
    assert restored.used_self_energy is False
