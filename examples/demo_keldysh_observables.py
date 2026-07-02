from __future__ import annotations

import sympy as sp

from quantum_transport import (
    contour_green_observable,
    keldysh_system,
    kt,
    lorentzian_density,
)


def main() -> None:
    t, tp, omega = sp.symbols("t tp omega", real=True)
    xi, gamma, gamma_l, gamma_r = sp.symbols("xi Gamma Gamma_L Gamma_R", positive=True)
    sigma_r, sigma_l, g_r, g_a = sp.symbols("Sigma_r Sigma_less G_r G_a")
    e, h, hbar = sp.symbols("e h hbar")

    k = keldysh_system(omega)
    G = k.green("G")
    Sigma = k.self_energy("Sigma")

    contour = contour_green_observable(
        G.greater(t, tp),
        G.lesser(t, tp),
        kt(t, "-"),
        kt(tp, "+"),
    )
    langreth = k.langreth(
        {"r": G.retarded(omega), "a": G.advanced(omega), "<": G.lesser(omega), ">": G.greater(omega)},
        {"r": Sigma.retarded(omega), "a": Sigma.advanced(omega), "<": Sigma.lesser(omega), ">": Sigma.greater(omega)},
    )
    g_less = k.dyson_lesser(g_r, sigma_l, g_a)
    population = k.population(g_less)
    current = k.meir_wingreen_current(sigma_r, g_less, sigma_l, g_a, charge=e, hbar=hbar)
    rho = lorentzian_density(omega, xi, gamma)
    wide_band = k.wide_band_current(
        rho,
        gamma_l,
        gamma_r,
        sp.Function("f_L")(omega),
        sp.Function("f_R")(omega),
        charge=e,
        h=h,
    )

    print("Contour observable:")
    sp.pprint(contour.doit())
    print("\nLangreth lesser observable:")
    sp.pprint(langreth["<"].doit())
    print("\nDyson lesser observable:")
    sp.pprint(g_less.doit())
    print("\nPopulation observable LaTeX:")
    print(population.latex())
    print("\nMeir-Wingreen current observable:")
    sp.pprint(current.doit())
    print("\nWide-band current observable:")
    sp.pprint(wide_band.doit())


if __name__ == "__main__":
    main()
