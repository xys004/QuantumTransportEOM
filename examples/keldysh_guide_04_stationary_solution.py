from __future__ import annotations

import sympy as sp

from quantum_transport import keldysh_system, lorentzian_density


def main() -> None:
    omega = sp.Symbol("omega", real=True)
    xi, gamma = sp.symbols("xi Gamma", positive=True)
    sigma_r, sigma_less = sp.symbols("Sigma_r Sigma_less")

    k = keldysh_system(omega)
    G = k.green("G")

    g_ret = k.dyson_retarded(xi, sigma_r)
    g_adv = G.advanced(omega)
    g_less = k.dyson_lesser(g_ret, sigma_less, g_adv)
    rho = lorentzian_density(omega, xi, gamma)
    population = k.population(g_less)

    print("Remote-time stationary solution")
    print("G^r(omega) =")
    sp.pprint(g_ret.doit())

    print("\nG^<(omega) = G^r Sigma^< G^a =")
    sp.pprint(g_less.doit())

    print("\nrho(omega) in wide-band form =")
    sp.pprint(rho)

    print("\n<n> =")
    sp.pprint(population.doit())

    print("\nLaTeX G^<:")
    print(g_less.latex())


if __name__ == "__main__":
    main()
