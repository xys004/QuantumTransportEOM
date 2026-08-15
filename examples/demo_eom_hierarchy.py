"""Generic labelled EOM hierarchy and stationary Keldysh closure demo."""

from __future__ import annotations

import sympy as sp

from quantum_transport import SelfConsistentClosure, build_eom_hierarchy, f, fd, n


def main() -> None:
    eps_up, eps_down, u = sp.symbols("epsilon_up epsilon_down U", real=True)
    omega, eta = sp.symbols("omega eta", real=True, positive=True)
    sigma_lesser = sp.diag(*sp.symbols("sigma_0:4"))

    hierarchy = build_eom_hierarchy(
        {
            "one_body": eps_up * fd("up") * f("up") + eps_down * fd("down") * f("down"),
            "interaction": u * n("up") * n("down"),
        },
        max_depth=1,
    )

    print("Basis size:", len(hierarchy.basis))
    print("Closed:", hierarchy.is_closed)
    print("Depths:", sorted(set(hierarchy.depth_by_operator.values())))
    print("Unresolved:", hierarchy.unresolved_operators)
    for equation in hierarchy.equations[:2]:
        print("\nEOM:", equation.operator)
        for contribution in equation.contributions:
            print(f"  {contribution.hamiltonian_label}: {contribution.rhs}")

    g_retarded = hierarchy.retarded_green(omega, eta)
    g_lesser = hierarchy.stationary_lesser_green(omega, eta, sigma_lesser)
    print("\nG^r shape:", g_retarded.shape)
    print("G^< shape:", g_lesser.shape)
    contour = hierarchy.contour_equations()
    print("Contour equations:", len(contour.equations))
    print("Langreth components:", ["r", "a", "<", ">", "rceil", "lceil", "M"])

    shallow = build_eom_hierarchy(
        {
            "one_body": eps_up * fd("up") * f("up") + eps_down * fd("down") * f("down"),
            "interaction": u * n("up") * n("down"),
        },
        max_depth=0,
    )
    approximate = shallow.retarded_green(omega, eta, approximate=True)
    print("Approximate shallow G^r shape:", approximate.shape)

    alpha = sp.Symbol("alpha", real=True)
    closure = SelfConsistentClosure(
        rules={operator: alpha * shallow.basis[0] for operator in shallow.unresolved_operators},
        initial_values={alpha: 0},
        update=lambda values, green: {alpha: sp.Rational(1, 5)},
    )
    closed = shallow.solve_self_consistent(omega, eta, closure)
    print("Self-consistent closure:", closed.converged, "iterations:", len(closed.iterations))


if __name__ == "__main__":
    main()
