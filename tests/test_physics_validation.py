"""Physics validation against exact analytic results.

These tests pin the *physics*, not the implementation: resonant-level
Lorentzian transmission, spectral sum rule, the fluctuation-dissipation
relation at equilibrium, current conservation, and unitarity bounds.
"""

import warnings

import numpy as np
import pytest
import sympy as sp

from quantum_transport import (
    CustomModel,
    LeadSelfEnergy,
    MatrixDevice,
    custom_model,
    f,
    fd,
    n,
)


GRID = np.linspace(-12.0, 12.0, 4001)


def _resonant_level(eps: float, gamma_left: float, gamma_right: float):
    device = MatrixDevice(hamiltonian=np.array([[eps]], dtype=complex), basis_labels=["dot"])
    left = LeadSelfEnergy.wide_band(np.array([[gamma_left]]), mu=0.0)
    right = LeadSelfEnergy.wide_band(np.array([[gamma_right]]), mu=0.0)
    return device.transport(left, right)


class TestResonantLevelAnalytic:
    """Single level + wide-band leads has closed-form answers."""

    def test_lorentzian_transmission(self):
        eps, gl, gr = 0.35, 0.4, 0.7
        view = _resonant_level(eps, gl, gr)
        transmission = view.transmission_values(GRID)
        gamma_total = gl + gr
        analytic = gl * gr / ((GRID - eps) ** 2 + (gamma_total / 2.0) ** 2)
        np.testing.assert_allclose(transmission, analytic, atol=1e-12)

    def test_unitarity_at_resonance_for_symmetric_coupling(self):
        eps, gamma = -0.2, 0.5
        view = _resonant_level(eps, gamma, gamma)
        assert view.transmission(eps) == pytest.approx(1.0, abs=1e-12)
        transmission = view.transmission_values(GRID)
        assert np.all(transmission <= 1.0 + 1e-12)

    def test_spectral_sum_rule(self):
        """The spectral density of the dot must integrate to 1 (one orbital)."""
        view = _resonant_level(0.1, 0.3, 0.5)
        wide = np.linspace(-150.0, 150.0, 30001)
        weight = np.trapezoid(view.retarded_values(wide)[:, 0, 0].imag, wide) / (-np.pi)
        # remaining Lorentzian tail weight outside the window ~ Gamma/(pi*W)
        assert weight == pytest.approx(1.0, abs=3e-3)

    def test_fluctuation_dissipation_at_equilibrium(self):
        """G^< = -f(omega) (G^r - G^a) when both leads share mu and T."""
        eps, gamma, mu, temperature = 0.2, 0.4, 0.1, 0.3
        device = MatrixDevice(hamiltonian=np.array([[eps]], dtype=complex), basis_labels=["dot"])
        left = LeadSelfEnergy.wide_band(np.array([[gamma]]), mu=mu, temperature=temperature)
        right = LeadSelfEnergy.wide_band(np.array([[gamma]]), mu=mu, temperature=temperature)
        view = device.transport(left, right)
        subgrid = np.linspace(-4.0, 4.0, 41)
        lesser = view.lesser_values(subgrid)[:, 0, 0]
        retarded = view.retarded_values(subgrid)[:, 0, 0]
        occupation = 1.0 / (np.exp((subgrid - mu) / temperature) + 1.0)
        expected = -occupation * (retarded - np.conj(retarded))
        np.testing.assert_allclose(lesser, expected, atol=1e-12)

    def test_equilibrium_current_vanishes(self):
        view = _resonant_level(0.0, 0.5, 0.5)  # both leads at mu = 0
        current = view.meir_wingreen_current(GRID, lead="left")
        assert current == pytest.approx(0.0, abs=1e-12)


class TestCurrentConservation:
    def test_left_and_right_currents_balance(self):
        """Steady state: current entering from the left equals current leaving right."""
        device = MatrixDevice(
            hamiltonian=np.array([[0.2, 1.0], [1.0, -0.3]], dtype=complex),
            basis_labels=["a", "b"],
        )
        left = LeadSelfEnergy.wide_band(np.diag([0.6, 0.0]).astype(complex), mu=0.8, temperature=0.05)
        right = LeadSelfEnergy.wide_band(np.diag([0.0, 0.4]).astype(complex), mu=-0.8, temperature=0.05)
        view = device.transport(left, right)
        current_left = view.meir_wingreen_current(GRID, lead="left")
        current_right = view.meir_wingreen_current(GRID, lead="right")
        assert current_left == pytest.approx(-current_right, rel=1e-8)
        assert abs(current_left) > 1e-4  # a real, finite bias current

    def test_landauer_matches_meir_wingreen_noninteracting(self):
        """For noninteracting devices Meir-Wingreen reduces to Landauer."""
        device = MatrixDevice(
            hamiltonian=np.array([[0.0, 0.8], [0.8, 0.0]], dtype=complex),
            basis_labels=["a", "b"],
        )
        left = LeadSelfEnergy.wide_band(np.diag([0.5, 0.0]).astype(complex), mu=0.6)
        right = LeadSelfEnergy.wide_band(np.diag([0.0, 0.5]).astype(complex), mu=-0.6)
        view = device.transport(left, right)
        meir_wingreen = view.meir_wingreen_current(GRID, lead="left")
        landauer = view.landauer_current(GRID, mu_left=0.6, mu_right=-0.6)
        assert meir_wingreen == pytest.approx(landauer, rel=1e-8)


class TestCustomModelPhysics:
    def test_chain_resonances_are_single_channel(self):
        """End-contacted chain: transmission bounded by 1, unit resonances at weak coupling."""
        t = 1.0
        hamiltonian = sum(t * (fd(i) * f(i + 1) + fd(i + 1) * f(i)) for i in range(2))
        model = CustomModel(hamiltonian + sp.Float(0.0) * n(0))
        view = model.open({"0": 0.4}, {"2": 0.4})
        transmission = view.transmission_values(GRID, eta=1e-9)
        assert np.all(transmission <= 1.0 + 1e-9)
        # Perfect resonant transmission is exact only in the weak-coupling
        # limit, where neighboring levels no longer interfere.
        weak = model.open({"0": 0.02}, {"2": 0.02})
        for resonance in (-np.sqrt(2.0), 0.0, np.sqrt(2.0)):
            assert weak.transmission(resonance, eta=1e-12) == pytest.approx(1.0, abs=1e-3)

    def test_hermiticity_warning_fires(self):
        t = sp.Symbol("t", complex=True)
        with pytest.warns(UserWarning, match="not Hermitian"):
            custom_model(t * fd("L") * f("R"))  # missing conjugate hopping

    def test_hermitian_hamiltonian_stays_silent(self):
        eps = sp.Symbol("epsilon", real=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            custom_model(eps * (n("up") + n("down")))

    def test_check_can_be_disabled(self):
        t = sp.Symbol("t", complex=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            custom_model(t * fd("L") * f("R"), check_hermitian=False)

    def test_hartree_green_function_reduces_to_free_at_zero_u(self):
        eps = sp.Symbol("epsilon", real=True)
        omega, eta = sp.symbols("omega eta", positive=True)
        model = CustomModel(eps * (n("up") + n("down")) + sp.Integer(0) * n("up") * n("down"))
        green = model.gf("c_up").retarded(omega=omega, eta=eta, method="hartree")
        assert sp.simplify(green - 1 / (omega + sp.I * eta - eps)) == 0


class TestSiteResolvedLeads:
    def test_unknown_site_raises(self):
        model = CustomModel(sp.Float(0.5) * (n(0) + n(1)))
        with pytest.raises(ValueError, match="unknown site"):
            model.open({"7": 0.5}, {"1": 0.5})

    def test_dict_lead_equals_matrix_lead(self):
        t = 1.0
        hamiltonian = t * (fd(0) * f(1) + fd(1) * f(0)) + sp.Float(0.2) * (n(0) + n(1))
        model = CustomModel(hamiltonian)
        by_dict = model.open({"0": 0.5}, {"1": 0.5})
        by_matrix = model.open(np.diag([0.5, 0.0]), np.diag([0.0, 0.5]))
        subgrid = np.linspace(-3, 3, 61)
        np.testing.assert_allclose(
            by_dict.transmission_values(subgrid),
            by_matrix.transmission_values(subgrid),
            atol=1e-14,
        )
