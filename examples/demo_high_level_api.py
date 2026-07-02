from __future__ import annotations

import numpy as np
import sympy as sp

from quantum_transport import AndersonImpurity, BosonicHarmonicMode, FermionicSingleLevel, b, bd, f, fd, n


def main() -> None:
    epsilon, omega0, omega, eta, U, mu, T = sp.symbols("epsilon omega0 omega eta U mu T", real=True, positive=True)
    n_down_avg = sp.symbols("n_down_avg", real=True)

    print("Short operators:")
    print("f(0) ->", f(0).doit())
    print("fd(0) ->", fd(0).doit())
    print("b(0) ->", b(0).doit())
    print("bd(0) ->", bd(0).doit())
    print("n('up') ->", n("up").doit())

    fermion = FermionicSingleLevel(eps=epsilon)
    boson = BosonicHarmonicMode(omega0=omega0)
    anderson = AndersonImpurity(eps=epsilon, U=U)

    print("\nSingle-level fermionic G^r:")
    sp.pprint(fermion.gf().retarded(omega=omega, eta=eta))

    print("\nSingle-level fermionic G^< in equilibrium:")
    sp.pprint(fermion.gf().lesser(omega=omega, eta=eta, mu=mu, temperature=T))

    print("\nSingle-level fermionic G^> in equilibrium:")
    sp.pprint(fermion.gf().greater(omega=omega, eta=eta, mu=mu, temperature=T))

    print("\nSingle-level spectral function A(omega):")
    sp.pprint(fermion.gf().spectral_function(omega=omega, eta=eta))

    print("\nHarmonic bosonic G^< in equilibrium:")
    sp.pprint(boson.gf().lesser(omega=omega, eta=eta, mu=mu, temperature=T))

    print("\nAnderson basis expanded one level:")
    for operator in anderson.eom_basis().expand(levels=1):
        sp.pprint(operator)

    print("\nAnderson Hartree-Fock G_up^r:")
    sp.pprint(
        anderson.gf("up").retarded(
            omega=omega,
            eta=eta,
            method="hartree_fock",
            occupations={"down": n_down_avg},
        )
    )

    print("\nAnderson Hubbard-I G_up^r:")
    sp.pprint(
        anderson.gf("up").retarded(
            omega=omega,
            eta=eta,
            method="hubbard_i",
            occupations={"down": n_down_avg},
        )
    )

    numeric_omega = sp.Symbol("omega", real=True)
    grid = np.linspace(-10.0, 10.0, 5001)
    occ_result = AndersonImpurity(eps=-1.0, U=2.0).self_consistent_occupations(
        omega_symbol=numeric_omega,
        omega_grid=grid,
        eta=0.08,
        method="hubbard_i",
        mu=0.0,
        temperature=0.0,
        initial={"up": 0.4, "down": 0.6},
        mixing=0.8,
        tol=1e-4,
        max_iter=200,
    )
    print("\nSelf-consistent Anderson occupations (Hubbard-I):")
    print(occ_result)


if __name__ == "__main__":
    main()
