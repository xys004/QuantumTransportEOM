"""Tests for custom_model / CustomModel: detection, exact EOM engine, Hartree, open transport."""

import numpy as np
import pytest
import sympy as sp

from quantum_transport import (
    CustomModel,
    LeadSelfEnergy,
    SpinfulDimer,
    b,
    bd,
    build_fermionic_hartree_eom_system,
    custom_model,
    f,
    fd,
    n,
    single_particle_hamiltonian_matrix,
)


EPS, U = sp.symbols("epsilon U", real=True)
OMEGA = sp.Symbol("omega", real=True)
ETA = sp.Symbol("eta", positive=True)


def _anderson_by_hand() -> CustomModel:
    hamiltonian = EPS * (n("up") + n("down")) + U * n("up") * n("down")
    return CustomModel(hamiltonian, name="anderson_by_hand")


class TestCustomModelDetection:
    def test_detects_fermionic_statistics(self):
        model = custom_model(EPS * n(0))
        assert model.statistics == "fermion"
        assert set(model.operators) == {"c_0", "c_0_dag"}

    def test_detects_bosonic_statistics(self):
        omega0 = sp.Symbol("omega_0", positive=True)
        model = custom_model(omega0 * bd("x") * b("x"))
        assert model.statistics == "boson"
        assert set(model.operators) == {"b_x", "b_x_dag"}

    def test_detects_mixed_statistics(self):
        g = sp.Symbol("g", real=True)
        model = custom_model(EPS * n(0) + g * (bd(0) * f(0) + fd(0) * b(0)))
        assert model.statistics == "mixed"
        assert "c_0" in model.operators and "b_0" in model.operators

    def test_rejects_operator_free_expression(self):
        with pytest.raises(ValueError):
            custom_model(EPS * sp.Symbol("x"))

    def test_metadata_records_modes(self):
        model = custom_model(EPS * (n("up") + n("down")))
        assert tuple(map(str, model.metadata["fermion_indices"])) == ("down", "up")


class TestExactEOMEngine:
    def test_quadratic_hopping_model_closes(self):
        t = sp.Symbol("t", real=True)
        hamiltonian = EPS * (n("L") + n("R")) + t * (fd("L") * f("R") + fd("R") * f("L"))
        result = custom_model(hamiltonian).eom()
        assert result.is_closed
        matrix = sp.simplify(result.eom_matrix - sp.Matrix([[EPS, t], [t, EPS]]))
        assert matrix == sp.zeros(2, 2)

    def test_interacting_model_residual_is_cubic(self):
        model = _anderson_by_hand().model
        result = model.eom()
        assert not result.is_closed
        assert sp.simplify(result.eom_matrix - EPS * sp.eye(2)) == sp.zeros(2, 2)

    def test_anderson_atomic_hierarchy_closes_on_expanded_basis(self):
        model = _anderson_by_hand().model
        result = model.eom(auto_expand_steps=1)
        assert result.is_closed
        assert result.eom_matrix.shape == (4, 4)
        eigenvalues = set(result.eom_matrix.eigenvals())
        assert {EPS, EPS + U} == {sp.simplify(value) for value in eigenvalues}

    def test_bosonic_custom_model_closes(self):
        omega0 = sp.Symbol("omega_0", positive=True)
        result = custom_model(omega0 * bd("x") * b("x")).eom()
        assert result.is_closed
        assert sp.simplify(result.eom_matrix[0, 0] - omega0) == 0


class TestGenericHartree:
    def test_hartree_matrix_matches_mean_field(self):
        model = _anderson_by_hand().model
        result = model.eom(truncation="hartree")
        assert result.is_closed
        n_up = sp.Symbol("n_up_avg", real=True)
        n_down = sp.Symbol("n_down_avg", real=True)
        expected = sp.diag(EPS + U * n_up, EPS + U * n_down)
        assert sp.simplify(result.eom_matrix - expected) == sp.zeros(2, 2)

    def test_hartree_accepts_explicit_occupations(self):
        model = _anderson_by_hand().model
        result = model.eom(
            truncation="hartree",
            truncation_params={"occupations": {"up": sp.Rational(1, 2), "down": sp.Rational(1, 4)}},
        )
        expected = sp.diag(EPS + U * sp.Rational(1, 2), EPS + U * sp.Rational(1, 4))
        assert sp.simplify(result.eom_matrix - expected) == sp.zeros(2, 2)

    def test_hartree_retarded_green_function(self):
        api = _anderson_by_hand()
        green = api.gf("c_up").retarded(omega=OMEGA, eta=ETA, method="hartree")
        n_down = sp.Symbol("n_down_avg", real=True)
        expected = 1 / (OMEGA + sp.I * ETA - EPS - U * n_down)
        assert sp.simplify(green - expected) == 0

    def test_direct_builder_works_for_integer_indices(self):
        hamiltonian = EPS * (n(0) + n(1)) + U * n(0) * n(1)
        result = build_fermionic_hartree_eom_system([f(0), f(1)], hamiltonian)
        assert result.is_closed


class TestSingleParticleMatrix:
    def test_extracts_quadratic_matrix(self):
        t = sp.Symbol("t", real=True)
        hamiltonian = EPS * n("A") + 2 * EPS * n("B") + t * fd("A") * f("B") + t * fd("B") * f("A")
        matrix, modes = single_particle_hamiltonian_matrix(hamiltonian)
        assert [str(mode) for mode in modes] == ["A", "B"]
        expected = sp.Matrix([[EPS, t], [t, 2 * EPS]])
        assert sp.simplify(matrix - expected) == sp.zeros(2, 2)

    def test_rejects_interacting_hamiltonian(self):
        with pytest.raises(ValueError, match="not quadratic"):
            single_particle_hamiltonian_matrix(U * n("up") * n("down"))

    def test_rejects_bosonic_hamiltonian(self):
        with pytest.raises(ValueError, match="fermionic"):
            single_particle_hamiltonian_matrix(sp.Symbol("w") * bd(0) * b(0))


class TestCustomModelOpenTransport:
    def test_matches_spinful_dimer_reference(self):
        hopping, spin_orbit = 0.7, 0.15
        hamiltonian = (
            sp.Float(0.2) * (n("L_up") + n("L_down"))
            - sp.Float(0.1) * (n("R_up") + n("R_down"))
            + hopping * (fd("R_up") * f("L_up") + fd("L_up") * f("R_up"))
            + hopping * (fd("R_down") * f("L_down") + fd("L_down") * f("R_down"))
            + spin_orbit * (fd("R_up") * f("L_down") + fd("L_down") * f("R_up"))
            - spin_orbit * (fd("R_down") * f("L_up") + fd("L_up") * f("R_down"))
        )
        view = CustomModel(hamiltonian).open(0.5, 0.5)
        reference_device = SpinfulDimer(
            eps_left_up=0.2,
            eps_left_down=0.2,
            eps_right_up=-0.1,
            eps_right_down=-0.1,
            hopping=hopping,
            spin_orbit=spin_orbit,
        )
        reference = reference_device.transport(
            LeadSelfEnergy.wide_band(np.eye(4) * 0.5),
            LeadSelfEnergy.wide_band(np.eye(4) * 0.5),
        )
        grid = np.linspace(-3.0, 3.0, 101)
        np.testing.assert_allclose(
            view.transmission_values(grid, eta=1e-6),
            reference.transmission_values(grid, eta=1e-6),
            atol=1e-12,
        )

    def test_symbolic_parameters_require_substitution(self):
        model = CustomModel(EPS * n(0))
        with pytest.raises(ValueError, match="free symbols"):
            model.open(0.5, 0.5)
        view = model.open(0.5, 0.5, parameters={"epsilon": -0.3})
        transmission = view.transmission(-0.3, eta=1e-9)
        assert transmission == pytest.approx(1.0, abs=1e-6)

    def test_lead_dimension_validation(self):
        model = CustomModel(sp.Float(0.0) * n(0) + sp.Float(0.5) * n(1))
        with pytest.raises(ValueError, match="shape"):
            model.open(np.eye(3), 0.5)


class TestLatexRepr:
    def test_model_repr_latex(self):
        api = _anderson_by_hand()
        rendered = api._repr_latex_()
        assert rendered.startswith("$H = ") and rendered.endswith("$")

    def test_eom_result_repr_latex(self):
        result = _anderson_by_hand().model.eom(truncation="hartree")
        rendered = result._repr_latex_()
        assert rendered.startswith("$") and "epsilon" in rendered
