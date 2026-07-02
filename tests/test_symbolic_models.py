import sympy as sp

from quantum_transport import physical_simplify_fermionic
from quantum_transport import BQObj, SQObj
from quantum_transport.models import (
    anderson_impurity_model,
    bosonic_harmonic_mode_model,
    coupled_bosonic_dimer_model,
    fermionic_dimer_model,
    fermionic_single_level_model,
    holstein_single_site_model,
    jaynes_cummings_like_model,
)


def test_fermionic_single_level_model_is_closed_and_has_expected_green_function():
    epsilon, omega, eta = sp.symbols("epsilon omega eta")
    model = fermionic_single_level_model(epsilon, index=0)

    assert model.statistics == "fermion"
    assert isinstance(model.operators["d"], SQObj)

    eom = model.eom()
    g_ret = model.retarded(model.operators["d"], model.operators["d_dag"], omega=omega, eta=eta)

    assert eom.is_closed
    assert eom.residuals == [0]
    assert sp.simplify(g_ret - 1 / (omega + sp.I * eta - epsilon)) == 0


def test_physical_simplify_fermionic_makes_single_level_rhs_more_presentable():
    epsilon = sp.Symbol("epsilon")
    model = fermionic_single_level_model(epsilon, index=0)
    rhs = model.operators["d"].eom_rhs(model.hamiltonian).doit()
    simplified = physical_simplify_fermionic(rhs)
    assert "KroneckerDelta" not in str(simplified)


def test_fermionic_dimer_model_basis_closes():
    eps0, eps1, t = sp.symbols("eps0 eps1 t")
    model = fermionic_dimer_model(eps0, eps1, t)

    eom = model.eom()
    assert model.statistics == "fermion"
    assert len(model.basis) == 2
    assert eom.is_closed
    assert eom.eom_matrix.shape == (2, 2)


def test_anderson_impurity_model_minimal_basis_runs_but_is_not_closed():
    eps_up, eps_down, u = sp.symbols("eps_up eps_down u")
    model = anderson_impurity_model(eps_up, eps_down, u)

    analysis = model.analyze_eom()
    assert model.metadata["interacting"] is True
    assert len(model.basis) == 2
    assert analysis.success is True
    assert analysis.is_closed is False


def test_anderson_impurity_auto_expansion_closes_atomic_hierarchy():
    eps_up, eps_down, u = sp.symbols("eps_up eps_down u")
    model = anderson_impurity_model(eps_up, eps_down, u)

    expanded_basis = model.expand_basis(max_steps=1)
    analysis = model.analyze_eom(auto_expand_steps=1)

    assert len(expanded_basis) == 4
    assert analysis.success is True
    assert analysis.is_closed is True
    assert analysis.result.eom_matrix.shape == (4, 4)


def test_anderson_impurity_hartree_fock_truncation_gives_shifted_green_function():
    eps_up, eps_down, u, omega, eta = sp.symbols("eps_up eps_down u omega eta")
    n_up_avg, n_down_avg = sp.symbols("n_up_avg n_down_avg")
    model = anderson_impurity_model(eps_up, eps_down, u)

    g_ret = model.retarded(
        model.operators["d_up"],
        model.operators["d_up_dag"],
        omega=omega,
        eta=eta,
        truncation="hartree_fock",
        truncation_params={"occupations": {"up": n_up_avg, "down": n_down_avg}},
    )

    assert sp.simplify(g_ret - 1 / (omega + sp.I * eta - eps_up - u * n_down_avg)) == 0


def test_anderson_impurity_hubbard_i_analysis_is_closed_on_expanded_atomic_basis():
    eps_up, eps_down, u = sp.symbols("eps_up eps_down u")
    model = anderson_impurity_model(eps_up, eps_down, u)

    analysis = model.analyze_eom(truncation="hubbard_i")

    assert analysis.success is True
    assert analysis.is_closed is True
    assert analysis.result.eom_matrix.shape == (4, 4)


def test_anderson_impurity_hubbard_i_green_function_has_two_poles():
    eps_up, eps_down, u, omega, eta = sp.symbols("eps_up eps_down u omega eta")
    n_up_avg, n_down_avg = sp.symbols("n_up_avg n_down_avg")
    model = anderson_impurity_model(eps_up, eps_down, u)

    g_ret = model.retarded(
        model.operators["d_up"],
        model.operators["d_up_dag"],
        omega=omega,
        eta=eta,
        truncation="hubbard_i",
        truncation_params={"occupations": {"up": n_up_avg, "down": n_down_avg}},
    )

    expected = (1 - n_down_avg) / (omega + sp.I * eta - eps_up) + n_down_avg / (omega + sp.I * eta - eps_up - u)
    assert sp.simplify(g_ret - expected) == 0


def test_bosonic_harmonic_mode_model_is_closed_and_has_expected_green_function():
    omega0, omega, eta = sp.symbols("omega0 omega eta")
    model = bosonic_harmonic_mode_model(omega0, index=0)

    assert model.statistics == "boson"
    assert isinstance(model.operators["b"], BQObj)

    eom = model.eom()
    g_ret = model.retarded(model.operators["b"], model.operators["b_dag"], omega=omega, eta=eta)

    assert eom.is_closed
    assert eom.residuals == [0]
    assert sp.simplify(g_ret - 1 / (omega + sp.I * eta - omega0)) == 0


def test_coupled_bosonic_dimer_model_basis_closes():
    omega0, omega1, g = sp.symbols("omega0 omega1 g")
    model = coupled_bosonic_dimer_model(omega0, omega1, g)

    eom = model.eom()
    assert model.statistics == "boson"
    assert len(model.basis) == 2
    assert eom.is_closed
    assert eom.eom_matrix.shape == (2, 2)


def test_holstein_model_mixed_eom_analysis_runs_and_expands_basis():
    epsilon, omega0, g = sp.symbols("epsilon omega0 g")
    model = holstein_single_site_model(epsilon, omega0, g)
    analysis = model.analyze_eom()
    expanded_basis = model.expand_basis(max_steps=1)

    assert model.statistics == "mixed"
    assert model.metadata["mixed"] is True
    assert analysis.success is True
    assert analysis.is_closed is False
    assert len(expanded_basis) > len(model.basis)


def test_jaynes_cummings_like_model_mixed_eom_analysis_runs_and_expands_basis():
    epsilon, omega0, g = sp.symbols("epsilon omega0 g")
    model = jaynes_cummings_like_model(epsilon, omega0, g)
    analysis = model.analyze_eom()
    expanded_basis = model.expand_basis(max_steps=1)

    assert model.statistics == "mixed"
    assert analysis.success is True
    assert analysis.is_closed is False
    assert len(expanded_basis) > len(model.basis)


def test_model_latex_hamiltonian_returns_string():
    epsilon = sp.Symbol("epsilon")
    model = fermionic_single_level_model(epsilon)
    latex_string = model.latex_hamiltonian()
    assert "epsilon" in latex_string
