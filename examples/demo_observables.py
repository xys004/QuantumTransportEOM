from __future__ import annotations

import numpy as np
import sympy as sp

from quantum_transport import (
    AndersonImpurity,
    ObservableExpr,
    landauer_current,
    landauer_current_numeric,
    obs,
    trace_expr,
    transmission,
)


def main() -> None:
    omega, eta = sp.symbols("omega eta", real=True, positive=True)
    eps, U = sp.symbols("eps U", real=True)
    mu_l, mu_r = sp.symbols("mu_l mu_r", real=True)
    n_down_avg = sp.symbols("n_down_avg", real=True)

    anderson = AndersonImpurity(eps=eps, U=U)
    g_r = anderson.gf("up").retarded(
        omega=omega,
        eta=eta,
        method="hubbard_i",
        occupations={"down": n_down_avg},
    )
    g_a = anderson.gf("up").advanced(
        omega=omega,
        eta=eta,
        method="hubbard_i",
        occupations={"down": n_down_avg},
    )

    gamma_l, gamma_r = sp.symbols("Gamma_L Gamma_R", positive=True, real=True)

    print("Standard transmission T(omega):")
    sp.pprint(transmission(g_r, g_a, gamma_l, gamma_r).doit())

    print("\nManual observable builder example:")
    manual = obs(sp.Matrix([[g_r * gamma_l * g_a * gamma_r]])).trace()
    sp.pprint(manual.doit())

    print("\nSymbolic Landauer current:")
    current = landauer_current(g_r, g_a, gamma_l, gamma_r, omega, mu_l, mu_r)
    sp.pprint(current.doit())

    print("\nNumeric Landauer current example:")
    grid = np.linspace(-6.0, 6.0, 3001)
    transmission_vals = np.exp(-grid**2)
    numeric_current = landauer_current_numeric(transmission_vals, grid, 0.2, -0.2)
    print(numeric_current)


if __name__ == "__main__":
    main()
