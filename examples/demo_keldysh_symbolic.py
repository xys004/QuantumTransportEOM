from __future__ import annotations

import sympy as sp

from quantum_transport import (
    contour_green_from_lesser_greater,
    dyson_lesser_stationary,
    dyson_retarded_from_level,
    kgf,
    kt,
    langreth_convolution,
    latex_keldysh,
    lorentzian_density,
    meir_wingreen_current_symbolic,
    stationary_population,
    two_terminal_wide_band_current_symbolic,
)


def main() -> None:
    t, tp, omega = sp.symbols("t tp omega", real=True)
    xi, gamma, gamma_l, gamma_r = sp.symbols("xi Gamma Gamma_L Gamma_R", positive=True)
    sigma_r, sigma_l = sp.symbols("Sigma_r Sigma_less")
    e, h, hbar = sp.symbols("e h hbar")

    G = kgf("G")
    Sigma = kgf("Sigma")

    contour_g = contour_green_from_lesser_greater(
        G.greater(t, tp),
        G.lesser(t, tp),
        kt(t, "-"),
        kt(tp, "+"),
    )
    product = langreth_convolution(
        {"r": G.retarded(omega), "a": G.advanced(omega), "<": G.lesser(omega), ">": G.greater(omega)},
        {"r": Sigma.retarded(omega), "a": Sigma.advanced(omega), "<": Sigma.lesser(omega), ">": Sigma.greater(omega)},
    )

    g_ret = dyson_retarded_from_level(omega, xi, sigma_r)
    g_adv = sp.conjugate(g_ret)
    g_less = dyson_lesser_stationary(g_ret, sigma_l, g_adv)
    rho = lorentzian_density(omega, xi, gamma)
    occupation = stationary_population(g_less, omega)
    mw_current = meir_wingreen_current_symbolic(sigma_r, g_less, sigma_l, g_adv, omega, charge=e, hbar=hbar)
    wb_current = two_terminal_wide_band_current_symbolic(
        omega,
        rho,
        gamma_l,
        gamma_r,
        sp.Function("f_L")(omega),
        sp.Function("f_R")(omega),
        charge=e,
        h=h,
    )

    print("Contour G(t-, t'+):")
    sp.pprint(contour_g)
    print("\nLangreth product (lesser component):")
    sp.pprint(product["<"])
    print("\nStationary G^r:")
    sp.pprint(g_ret)
    print("\nStationary G^<:")
    sp.pprint(g_less)
    print("\nOccupation:")
    sp.pprint(occupation)
    print("\nMeir-Wingreen current:")
    sp.pprint(mw_current)
    print("\nTwo-terminal wide-band current:")
    sp.pprint(wb_current)
    print("\nLaTeX G^r:")
    print(latex_keldysh(g_ret))


if __name__ == "__main__":
    main()
