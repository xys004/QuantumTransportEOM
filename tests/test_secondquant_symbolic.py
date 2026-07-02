import sympy as sp

from quantum_transport import (
    SQObj,
    annihilate,
    build_fermionic_eom_system,
    create,
    destroy,
    fermionic_anticommutator,
    fermionic_eom_rhs,
    latex_expr,
    num,
    number_operator,
    retarded_green_from_fermionic_eom,
    sqobj,
)


def test_fermionic_anticommutator_returns_identity_for_same_mode():
    expr = fermionic_anticommutator(annihilate(0), create(0))
    assert sp.simplify(expr - 1) == 0


def test_single_level_eom_closes_on_annihilation_operator():
    epsilon = sp.Symbol("epsilon")
    d = annihilate(0)
    hamiltonian = epsilon * number_operator(0, simplify=False)

    rhs = fermionic_eom_rhs(d, hamiltonian)
    eom = build_fermionic_eom_system([d], hamiltonian)

    assert eom.is_closed
    assert eom.eom_matrix.shape == (1, 1)
    assert eom.residuals == [0]
    assert sp.simplify(rhs - eom.eom_matrix[0, 0] * d) == 0


def test_retarded_green_from_single_level_eom_solves_eom_equation():
    epsilon, omega, eta = sp.symbols("epsilon omega eta")
    d = annihilate(0)
    dd = create(0)
    hamiltonian = epsilon * number_operator(0, simplify=False)

    eom = build_fermionic_eom_system([d], hamiltonian)
    g_ret = retarded_green_from_fermionic_eom(eom, [d], [dd], omega=omega, eta=eta)
    lhs = (omega + sp.I * eta - eom.eom_matrix[0, 0]) * g_ret[0, 0]
    assert sp.simplify(lhs - 1) == 0


def test_latex_export_for_single_level_green_function():
    epsilon, omega, eta = sp.symbols("epsilon omega eta")
    d = annihilate(0)
    dd = create(0)
    hamiltonian = epsilon * number_operator(0, simplify=False)

    eom = build_fermionic_eom_system([d], hamiltonian)
    g_ret = retarded_green_from_fermionic_eom(eom, [d], [dd], omega=omega, eta=eta)
    latex_string = latex_expr(g_ret[0, 0])
    assert "epsilon" in latex_string
    assert "omega" in latex_string


def test_qutip_like_sqobj_api_is_usable():
    epsilon, omega, eta = sp.symbols("epsilon omega eta")
    d = destroy(0)
    dd = d.dag()
    hamiltonian = epsilon * num(0)

    assert isinstance(d, SQObj)
    assert isinstance(dd, SQObj)
    assert sp.simplify(d.anticomm(dd).doit() - 1) == 0

    rhs = d.eom_rhs(hamiltonian)
    eom = d.eom(hamiltonian)
    g_ret = d.retarded(dd, hamiltonian, omega=omega, eta=eta)

    assert isinstance(rhs, SQObj)
    assert eom.is_closed
    assert sp.simplify(rhs.doit() - eom.eom_matrix[0, 0] * d.doit()) == 0
    assert sp.simplify((omega + sp.I * eta - eom.eom_matrix[0, 0]) * g_ret - 1) == 0


def test_sqobj_wrap_and_latex_helpers():
    op = sqobj(annihilate(0))
    assert isinstance(op, SQObj)
    latex_string = (op.dag() * op).latex()
    assert "a_" in latex_string
    assert "dagger" in latex_string
