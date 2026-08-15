import numpy as np

from quantum_transport import lorentzian_reservoir_two_time


def test_lorentzian_retarded_kernel_matches_analytic_exponential():
    time = np.array([0.0, 0.2, 0.4, 0.8, 1.4])
    gamma = np.array([[0.6]], dtype=complex)
    result = lorentzian_reservoir_two_time(
        time,
        gamma,
        bandwidth=1.2,
        center=0.3,
        chemical_potential=0.0,
        temperature=0.1,
        energy_grid=np.linspace(-80.0, 80.0, 16001),
    )
    lag = time[:, None] - time[None, :]
    expected = -0.5j * 0.6 * 1.2 * (
        np.tril(np.ones_like(lag), k=-1) + 0.5 * np.eye(time.size)
    ) * np.exp(-(1.2 + 0.3j) * np.maximum(lag, 0.0))
    np.testing.assert_allclose(result.retarded[:, :, 0, 0], expected, atol=2e-14)
    assert result.memory_time == 1.0 / 1.2
    assert result.retarded_causality_error == 0.0


def test_smooth_gauge_phase_dresses_all_memory_components_and_preserves_adjoint():
    time = np.linspace(0.0, 1.0, 7)
    gamma = np.array([[0.6]], dtype=complex)
    energy = np.linspace(-60.0, 60.0, 12001)
    base = lorentzian_reservoir_two_time(
        time,
        gamma,
        bandwidth=1.2,
        center=0.3,
        chemical_potential=0.0,
        temperature=0.1,
        energy_grid=energy,
    )
    phase = 0.15 * time**2
    dressed = lorentzian_reservoir_two_time(
        time,
        gamma,
        bandwidth=1.2,
        center=0.3,
        chemical_potential=0.0,
        temperature=0.1,
        energy_grid=energy,
        phase=phase,
    )
    factor = np.exp(-1j * (phase[:, None] - phase[None, :]))[:, :, None, None]
    np.testing.assert_allclose(dressed.retarded, base.retarded * factor, atol=2e-14)
    np.testing.assert_allclose(dressed.lesser, base.lesser * factor, atol=2e-13)
    np.testing.assert_allclose(dressed.advanced, dressed.retarded.swapaxes(0, 1).swapaxes(-1, -2).conj(), atol=2e-14)

