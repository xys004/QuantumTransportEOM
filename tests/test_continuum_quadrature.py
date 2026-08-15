import numpy as np

from quantum_transport import flat_band_star_quadrature


def test_flat_band_star_quadrature_preserves_broadening_factor_shape():
    gamma = np.array([[0.22, 0.04 - 0.01j], [0.04 + 0.01j, 0.16]], dtype=complex)
    result = flat_band_star_quadrature(gamma, half_bandwidth=12.0, n_points=16)
    assert result.lead_hamiltonian.shape == (32, 32)
    assert result.coupling_matrix.shape == (2, 32)
    assert result.energies.size == 16
    assert np.allclose(result.lead_hamiltonian, result.lead_hamiltonian.conj().T)
    assert np.all(np.isfinite(result.coupling_matrix))


def test_flat_band_star_quadrature_rejects_non_positive_broadening():
    with np.testing.assert_raises(ValueError):
        flat_band_star_quadrature(np.diag([0.2, -0.1]), half_bandwidth=4.0, n_points=8)
