from __future__ import annotations

import sympy as sp


def kron3(a: sp.Matrix, b: sp.Matrix, c: sp.Matrix) -> sp.Matrix:
    return sp.kronecker_product(a, b, c)


def commutator(a: sp.Matrix, b: sp.Matrix) -> sp.Matrix:
    return sp.simplify(a * b - b * a)


def matrix_equal(a: sp.Matrix, b: sp.Matrix) -> bool:
    diff = a - b
    return all(sp.simplify(entry) == 0 for entry in diff)


def print_identity(name: str, lhs: sp.Matrix, rhs: sp.Matrix) -> None:
    print(f"{name}: {matrix_equal(lhs, rhs)}")


def operator_identities() -> None:
    i = sp.I
    dx, dy, dz = sp.symbols("d_x d_y d_z")

    sx = sp.Matrix([[0, 1], [1, 0]])
    sy = sp.Matrix([[0, -i], [i, 0]])
    sz = sp.Matrix([[1, 0], [0, -1]])
    s0 = sp.eye(2)

    rx, ry, rz, r0 = sx, sy, sz, s0
    tx, ty, tz, t0 = sx, sy, sz, s0

    identity = kron3(r0, t0, s0)
    rho_x = kron3(rx, t0, s0)
    tau_x = kron3(r0, tx, s0)
    tau_y = kron3(r0, ty, s0)
    tau_z = kron3(r0, tz, s0)
    sigma_x = kron3(r0, t0, sx)
    sigma_y = kron3(r0, t0, sy)
    sigma_z = kron3(r0, t0, sz)

    spin_axis = dx * sigma_x + dy * sigma_y + dz * sigma_z
    soc = tau_x * spin_axis
    tau_plus_probe = sp.Rational(1, 2) * (identity + tau_z)
    gamma_hybrid = rho_x * tau_x

    print("Operator identities in rho x tau x sigma space")
    print_identity("[tau_z, tau_x d.sigma] = 2 i tau_y d.sigma", commutator(tau_z, soc), 2 * i * tau_y * spin_axis)
    print_identity(
        "[sigma_z, tau_x d.sigma] = 2 i tau_x (d_x sigma_y - d_y sigma_x)",
        commutator(sigma_z, soc),
        2 * i * tau_x * (dx * sigma_y - dy * sigma_x),
    )
    print_identity("[P_tau+, tau_x d.sigma] = i tau_y d.sigma", commutator(tau_plus_probe, soc), i * tau_y * spin_axis)
    print_identity("[P_tau+, rho_x tau_x] = i rho_x tau_y", commutator(tau_plus_probe, gamma_hybrid), i * rho_x * tau_y)
    print()


def linear_response_voltage_probe_formula() -> None:
    m = sp.symbols("m")
    glr0, glp0, gpl0, gpr0 = sp.symbols("G_LR0 G_LP0 G_PL0 G_PR0", positive=True)
    dglr, dglp, dgpl, dgpr = sp.symbols("dG_LR dG_LP dG_PL dG_PR")

    glr = glr0 + m * dglr
    glp = glp0 + m * dglp
    gpl = gpl0 + m * dgpl
    gpr = gpr0 + m * dgpr

    g_eff = glr + glp * gpr / (gpl + gpr)
    odd_coefficient = sp.factor(sp.diff(g_eff, m).subs(m, 0))

    print("Linear-response single-voltage-probe conductance")
    print("G_eff = G_LR + G_LP G_PR / (G_PL + G_PR)")
    print("Odd coefficient under M -> -M, for G_ab = G_ab0 + m dG_ab:")
    sp.pprint(odd_coefficient)
    print()

    no_probe_odd = sp.simplify(odd_coefficient.subs({dglp: 0, dgpl: 0, dgpr: 0}))
    no_direct_odd = sp.simplify(odd_coefficient.subs({dglr: 0}))
    print("If only two-terminal conductance is allowed to be odd:")
    sp.pprint(no_probe_odd)
    print("If direct L-R odd part cancels but probe transmissions are odd:")
    sp.pprint(no_direct_odd)
    print()


def nonlinear_voltage_feedback_formula() -> None:
    m, mu = sp.symbols("m mu_p")
    i_left = sp.Function("I_L")(m, mu)
    i_probe = sp.Function("I_P")(m, mu)

    dmu_dm = -sp.diff(i_probe, m) / sp.diff(i_probe, mu)
    total_derivative = sp.diff(i_left, m) + sp.diff(i_left, mu) * dmu_dm

    print("Nonlinear finite-bias voltage-probe feedback")
    print("Condition: I_P(m, mu_p(m)) = 0")
    print("d mu_p / d m =")
    sp.pprint(dmu_dm)
    print("d I_L(m, mu_p(m)) / d m =")
    sp.pprint(sp.factor(total_derivative))
    print()


def main() -> None:
    operator_identities()
    linear_response_voltage_probe_formula()
    nonlinear_voltage_feedback_formula()


if __name__ == "__main__":
    main()
