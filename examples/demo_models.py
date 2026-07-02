from __future__ import annotations

import sympy as sp

from quantum_transport import physical_simplify_fermionic
from quantum_transport.models import (
    anderson_impurity_model,
    bosonic_harmonic_mode_model,
    fermionic_single_level_model,
    holstein_single_site_model,
    jaynes_cummings_like_model,
)


def describe_model(model, *, auto_expand_steps: int = 0, truncation: str | None = None, truncation_params=None) -> None:
    print(f"\n=== {model.name} ===")
    print("statistics:", model.statistics)
    print("H LaTeX:", model.latex_hamiltonian())
    if auto_expand_steps > 0:
        expanded_basis = model.expand_basis(max_steps=auto_expand_steps)
        print("expanded basis size:", len(expanded_basis))
        print("expanded basis:")
        for operator in expanded_basis:
            sp.pprint(operator)
    analysis = model.analyze_eom(
        auto_expand_steps=auto_expand_steps,
        truncation=truncation,
        truncation_params=truncation_params,
    )
    print("EOM success:", analysis.success)
    if analysis.success:
        print("EOM closed:", analysis.is_closed)
        print("EOM matrix:")
        sp.pprint(analysis.result.eom_matrix)
        print("Residuals:")
        for residual in analysis.result.residuals:
            sp.pprint(residual)
    else:
        print("EOM error:", type(analysis.error).__name__, analysis.error)


def main() -> None:
    epsilon, omega0, eta, omega, g, u = sp.symbols("epsilon omega0 eta omega g u", real=True)
    n_up_avg, n_down_avg = sp.symbols("n_up_avg n_down_avg", real=True)

    fermion = fermionic_single_level_model(epsilon)
    boson = bosonic_harmonic_mode_model(omega0)
    holstein = holstein_single_site_model(epsilon, omega0, g)
    jc = jaynes_cummings_like_model(epsilon, omega0, g)
    anderson = anderson_impurity_model(epsilon, epsilon, u)

    for model in [fermion, boson, holstein, jc]:
        describe_model(model)

    describe_model(anderson)
    describe_model(anderson, auto_expand_steps=1)
    describe_model(
        anderson,
        truncation="hartree_fock",
        truncation_params={"occupations": {"up": n_up_avg, "down": n_down_avg}},
    )
    describe_model(
        anderson,
        truncation="hubbard_i",
        truncation_params={"occupations": {"up": n_up_avg, "down": n_down_avg}},
    )

    d = fermion.operators["d"]
    rhs = d.eom_rhs(fermion.hamiltonian).doit()
    print("\nPhysical simplification of fermionic RHS:")
    sp.pprint(physical_simplify_fermionic(rhs))

    g_ret = fermion.retarded(fermion.operators["d"], fermion.operators["d_dag"], omega=omega, eta=eta)
    print("\nFermionic single-level G^r(omega):")
    sp.pprint(g_ret)

    g_anderson_hf = anderson.retarded(
        anderson.operators["d_up"],
        anderson.operators["d_up_dag"],
        omega=omega,
        eta=eta,
        truncation="hartree_fock",
        truncation_params={"occupations": {"up": n_up_avg, "down": n_down_avg}},
    )
    print("\nAnderson Hartree-Fock G_up^r(omega):")
    sp.pprint(g_anderson_hf)

    g_anderson_hubbard_i = anderson.retarded(
        anderson.operators["d_up"],
        anderson.operators["d_up_dag"],
        omega=omega,
        eta=eta,
        truncation="hubbard_i",
        truncation_params={"occupations": {"up": n_up_avg, "down": n_down_avg}},
    )
    print("\nAnderson Hubbard-I G_up^r(omega):")
    sp.pprint(g_anderson_hubbard_i)


if __name__ == "__main__":
    main()
