import sympy as sp

from quantum_transport import (
    BQObj,
    annihilate_boson,
    bosonic_commutator,
    bosonic_eom_rhs,
    build_bosonic_eom_system,
    create_boson,
    destroy_b,
    num_b,
    number_operator_boson,
    retarded_green_from_bosonic_eom,
)


def test_bosonic_commutator_returns_identity_for_same_mode():
    expr = bosonic_commutator(annihilate_boson(0), create_boson(0))
    assert sp.simplify(expr - 1) == 0


def test_single_boson_mode_eom_closes_on_annihilation_operator():
    omega0 = sp.Symbol("omega0")
    b = annihilate_boson(0)
    hamiltonian = omega0 * number_operator_boson(0)

    rhs = bosonic_eom_rhs(b, hamiltonian)
    eom = build_bosonic_eom_system([b], hamiltonian)

    assert eom.is_closed
    assert eom.eom_matrix.shape == (1, 1)
    assert eom.residuals == [0]
    assert sp.simplify(rhs - omega0 * b) == 0
    assert sp.simplify(eom.eom_matrix[0, 0] - omega0) == 0


def test_retarded_green_from_bosonic_eom_solves_eom_equation():
    omega0, omega, eta = sp.symbols("omega0 omega eta")
    b = annihilate_boson(0)
    bd = create_boson(0)
    hamiltonian = omega0 * number_operator_boson(0)

    eom = build_bosonic_eom_system([b], hamiltonian)
    g_ret = retarded_green_from_bosonic_eom(eom, [b], [bd], omega=omega, eta=eta)
    lhs = (omega + sp.I * eta - eom.eom_matrix[0, 0]) * g_ret[0, 0]
    assert sp.simplify(lhs - 1) == 0


def test_qutip_like_bosonic_api_is_usable():
    omega0, omega, eta = sp.symbols("omega0 omega eta")
    b = destroy_b(0)
    bd = b.dag()
    hamiltonian = omega0 * num_b(0)

    assert isinstance(b, BQObj)
    assert isinstance(bd, BQObj)
    assert sp.simplify(b.comm(bd).doit() - 1) == 0

    rhs = b.eom_rhs(hamiltonian)
    eom = b.eom(hamiltonian)
    g_ret = b.retarded(bd, hamiltonian, omega=omega, eta=eta)

    assert isinstance(rhs, BQObj)
    assert eom.is_closed
    assert sp.simplify(rhs.doit() - omega0 * b.doit()) == 0
    assert sp.simplify((omega + sp.I * eta - eom.eom_matrix[0, 0]) * g_ret - 1) == 0
