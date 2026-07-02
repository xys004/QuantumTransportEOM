from __future__ import annotations

import sympy as sp

from quantum_transport import keldysh_system


def main() -> None:
    omega = sp.Symbol("omega", real=True)
    eta, kvec = sp.symbols("eta k")
    V_eta_k = sp.Symbol("V_eta_k")
    e, hbar = sp.symbols("e hbar")

    k = keldysh_system(omega)
    G = k.green("G")
    F = k.green("F_eta_k")
    Sigma_eta = k.self_energy("Sigma_eta")

    print("Quantum dot coupled to reservoirs")
    print("F_eta_k(t,t') = -i <T_K a_eta_k(t) b^dagger(t')>")
    print("G(t,t')       = -i <T_K b(t) b^dagger(t')>")

    print("\nEOM structure in frequency space")
    xi_eta_k, xi = sp.symbols("xi_eta_k xi")
    eom_f = sp.Eq((omega - xi_eta_k) * F.retarded(omega).doit(), sp.conjugate(V_eta_k) * G.retarded(omega).doit())
    eom_g = sp.Eq((omega - xi) * G.retarded(omega).doit(), 1 + sp.Symbol("sum_eta_k") * V_eta_k * F.retarded(omega).doit())
    sp.pprint(eom_f)
    sp.pprint(eom_g)

    print("\nSelf-energy definition")
    sigma_definition = sp.Eq(Sigma_eta.retarded(omega).doit(), sp.Symbol("sum_k") * sp.Abs(V_eta_k) ** 2 * sp.Function("g_eta_k^r")(omega))
    sp.pprint(sigma_definition)

    print("\nCurrent from reservoir eta")
    current = k.meir_wingreen_current(
        Sigma_eta.retarded(omega),
        G.lesser(omega),
        Sigma_eta.lesser(omega),
        G.advanced(omega),
        charge=e,
        hbar=hbar,
    )
    sp.pprint(current.doit())

    print("\nPopulation in the dot")
    population = k.population(G.lesser(omega))
    sp.pprint(population.doit())

    print("\nLaTeX current:")
    print(current.latex())


if __name__ == "__main__":
    main()
