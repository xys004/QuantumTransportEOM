from __future__ import annotations

import sympy as sp

from quantum_transport import keldysh_system, lorentzian_density


def main() -> None:
    omega = sp.Symbol("omega", real=True)
    xi, gamma, gamma_l, gamma_r = sp.symbols("xi Gamma Gamma_L Gamma_R", positive=True)
    e, h = sp.symbols("e h")

    k = keldysh_system(omega)
    f_l = sp.Function("f_L")(omega)
    f_r = sp.Function("f_R")(omega)
    rho = lorentzian_density(omega, xi, gamma)
    current = k.wide_band_current(rho, gamma_l, gamma_r, f_l, f_r, charge=e, h=h)

    print("Two-terminal wide-band current")
    print("rho(omega) =")
    sp.pprint(rho)

    print("\nI =")
    sp.pprint(current.doit())

    print("\nZero-bias check with f_L = f_R:")
    zero_bias = current.subs(f_l, f_r).simplify()
    sp.pprint(zero_bias.doit())

    print("\nLaTeX:")
    print(current.latex())


if __name__ == "__main__":
    main()
