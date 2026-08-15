from __future__ import annotations

import sympy as sp

from quantum_transport import build_eom_hierarchy, f, fd, n


def _anderson_terms():
    eps_up, eps_down, interaction = sp.symbols(
        "eps_up eps_down U", real=True
    )
    one_body = (
        eps_up * fd("up") * f("up")
        + eps_down * fd("down") * f("down")
    )
    quartic = interaction * n("up") * n("down")
    return {
        "one_body": one_body,
        "interaction": quartic,
    }


def test_hierarchy_preserves_hamiltonian_contribution_provenance():
    result = build_eom_hierarchy(_anderson_terms(), max_depth=1)

    assert result.is_closed
    assert result.reached_depth == 1
    assert len(result.basis) == 4

    equation = result.equation(f("up"))
    assert {item.hamiltonian_label for item in equation.contributions} == {
        "one_body",
        "interaction",
    }
    one_body = next(item for item in equation.contributions if item.hamiltonian_label == "one_body")
    interaction = next(item for item in equation.contributions if item.hamiltonian_label == "interaction")
    assert one_body.residual == 0
    assert interaction.residual == 0
    assert sp.simplify(interaction.rhs) != 0


def test_depth_zero_reports_unresolved_operators_and_requires_explicit_approximation():
    omega, eta = sp.symbols("omega eta", real=True)
    result = build_eom_hierarchy(_anderson_terms(), max_depth=0)

    assert not result.is_closed
    assert result.reached_depth == 0
    assert result.unresolved_operators

    try:
        result.retarded_green(omega, eta)
    except ValueError as error:
        assert "not closed" in str(error)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("An open hierarchy must not be solved as exact.")

    approximate = result.retarded_green(omega, eta, approximate=True)
    assert approximate.shape == (2, 2)

    explicit_closure = {operator: 0 for operator in result.unresolved_operators}
    closed_by_rule = result.retarded_green(
        omega,
        eta,
        residual_closure=explicit_closure,
    )
    assert closed_by_rule.shape == (2, 2)


def test_hierarchy_depth_is_independent_from_operator_order_and_can_be_capped():
    result = build_eom_hierarchy(_anderson_terms(), max_depth=3, max_operators=3)

    assert result.max_operators_reached
    assert len(result.basis) == 3
    assert not result.is_closed


def test_stationary_lesser_keeps_the_advanced_branch_independent_of_eta_collisions():
    """``G^< = G^r Sigma^< G^a`` must survive a hopping numerically equal to eta."""

    import numpy as np
    import sympy as sp

    from quantum_transport import build_eom_hierarchy, f, fd

    omega = sp.Symbol("omega", real=True)
    identity = sp.eye(2)
    eta = 0.01

    def residual(hopping):
        hamiltonian = (
            0.3 * fd(0) * f(0)
            - 0.2 * fd(1) * f(1)
            + hopping * (fd(0) * f(1) + fd(1) * f(0))
        )
        hierarchy = build_eom_hierarchy({"chain": hamiltonian}, max_depth=0)
        evaluate = lambda matrix: np.array(
            sp.Matrix(matrix).subs({omega: 0.15}).evalf(), dtype=complex
        )
        lesser = evaluate(hierarchy.stationary_lesser_green(omega, eta, identity))
        retarded = evaluate(hierarchy.retarded_green(omega, eta))
        return float(np.max(np.abs(lesser - retarded @ retarded.conj().T)))

    assert residual(eta) < 1e-12       # the colliding case was 2.5 before the fix
    assert residual(0.011) < 1e-12
    assert residual(0.25) < 1e-12
