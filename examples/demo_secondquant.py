"""Small symbolic demo for second-quantized EOM workflows."""

from __future__ import annotations

import sympy as sp

from quantum_transport import (
    annihilate,
    build_fermionic_eom_system,
    create,
    fermionic_anticommutator,
    fermionic_eom_rhs,
    latex_expr,
    retarded_green_from_fermionic_eom,
)


def main() -> None:
    epsilon, omega, eta = sp.symbols("epsilon omega eta", real=True)

    d = annihilate(0)
    dd = create(0)
    hamiltonian = epsilon * dd * d

    rhs = fermionic_eom_rhs(d, hamiltonian)
    print("[d_0, H] =", rhs)

    basis = [d]
    eom = build_fermionic_eom_system(basis, hamiltonian)
    print("Closed basis:", eom.is_closed)
    print("EOM matrix:")
    sp.pprint(eom.eom_matrix)
    print("Residuals:", eom.residuals)
    print("Note: SymPy secondquant may keep the above/below-Fermi partition deltas explicit.")

    source = fermionic_anticommutator(d, dd)
    print("{d_0, d_0^dagger} =", source)

    g_ret = retarded_green_from_fermionic_eom(eom, [d], [dd], omega=omega, eta=eta)
    print("G^r(omega) =")
    sp.pprint(g_ret)
    print("LaTeX:", latex_expr(g_ret[0, 0]))


if __name__ == "__main__":
    main()
