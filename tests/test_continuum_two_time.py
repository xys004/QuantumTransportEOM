import numpy as np
import pytest

from quantum_transport import (
    LeadSelfEnergy,
    MatrixTransportView,
    inverse_fourier_two_time,
    partition_free_wide_band_self_energy_two_time,
    stationary_greens_two_time,
    stationary_self_energy_two_time,
)


def _single_level_transport(
    *,
    level_energy: float = 0.23,
    gamma_left: float = 0.35,
    gamma_right: float = 0.25,
    mu_left: float = 0.0,
    mu_right: float = 0.0,
    temperature: float = 0.12,
) -> MatrixTransportView:
    return MatrixTransportView(
        hamiltonian=np.array([[level_energy]], dtype=np.complex128),
        basis_labels=["level"],
        left_lead=LeadSelfEnergy.wide_band(
            np.array([[gamma_left]]),
            mu=mu_left,
            temperature=temperature,
            name="left",
        ),
        right_lead=LeadSelfEnergy.wide_band(
            np.array([[gamma_right]]),
            mu=mu_right,
            temperature=temperature,
            name="right",
        ),
    )


def test_inverse_fourier_two_time_supports_nonuniform_frequency_grid():
    omega = np.array([-2.0, -0.5, 0.25, 1.5])
    time = np.array([0.0, 0.3, 0.9])
    values = np.ones((omega.size, 1, 1), dtype=np.complex128)

    transformed = inverse_fourier_two_time(omega, values, time)
    direct_equal_time = (omega[-1] - omega[0]) / (2.0 * np.pi)

    np.testing.assert_allclose(
        transformed[np.arange(time.size), np.arange(time.size), 0, 0],
        direct_equal_time,
        atol=2e-15,
    )
    np.testing.assert_allclose(
        transformed[:, :, 0, 0],
        transformed[:, :, 0, 0].conj().T,
        atol=2e-15,
    )


def test_wide_band_self_energy_becomes_band_limited_delta_kernel():
    gamma = 0.7
    cutoff = 60.0
    omega = np.linspace(-cutoff, cutoff, 12001)
    time = np.array([0.0, 0.17, 0.41])
    lead = LeadSelfEnergy.wide_band(
        np.array([[gamma]]),
        mu=0.1,
        temperature=0.15,
        name="probe",
    )

    result = stationary_self_energy_two_time(lead, time, omega)
    lag = time[:, None] - time[None, :]
    delta_cutoff = (cutoff / np.pi) * np.sinc(cutoff * lag / np.pi)
    expected_retarded = -0.5j * gamma * delta_cutoff

    np.testing.assert_allclose(result.retarded[:, :, 0, 0], expected_retarded, atol=3e-7)
    assert result.lead_name == "probe"
    assert result.consistency_report().maximum < 2e-13


def test_partition_free_wide_band_self_energy_applies_step_bias_phase():
    lead = LeadSelfEnergy.wide_band(
        np.array([[0.7]]), mu=0.3, temperature=0.15, name="quench"
    )
    time = np.array([0.0, 0.17, 0.41])
    omega = np.linspace(-40.0, 40.0, 8001)
    shifted = partition_free_wide_band_self_energy_two_time(
        lead,
        time,
        omega,
        bias_shift=0.4,
        initial_chemical_potential=0.0,
        initial_temperature=0.15,
    )
    reference = partition_free_wide_band_self_energy_two_time(
        lead,
        time,
        omega,
        bias_shift=0.0,
        initial_chemical_potential=0.0,
        initial_temperature=0.15,
    )
    phase = np.exp(-1j * 0.4 * (time[:, None] - time[None, :]))
    np.testing.assert_allclose(shifted.lesser[:, :, 0, 0], phase * reference.lesser[:, :, 0, 0], atol=2e-14)
    np.testing.assert_allclose(shifted.greater[:, :, 0, 0], phase * reference.greater[:, :, 0, 0], atol=2e-14)
    assert shifted.consistency_report().maximum < 2e-13


def test_stationary_two_time_greens_obey_keldysh_identities_and_equal_time_density():
    transport = _single_level_transport()
    omega = np.linspace(-45.0, 45.0, 18001)
    time = np.linspace(0.0, 2.0, 9)

    result = stationary_greens_two_time(transport, time, omega)
    report = result.consistency_report()
    density = result.density_matrices()
    density_frequency = -1j * np.trapezoid(
        transport.lesser_values(omega),
        omega,
        axis=0,
    ) / (2.0 * np.pi)

    assert report.advanced_adjoint == 0.0
    assert report.lesser_antihermiticity < 2e-14
    assert report.greater_antihermiticity < 2e-14
    assert report.keldysh_spectral < 2e-14
    assert result.equal_time_drift() < 2e-15
    np.testing.assert_allclose(
        density,
        np.broadcast_to(density_frequency, density.shape),
        atol=2e-14,
    )
    np.testing.assert_allclose(density, density.swapaxes(-1, -2).conj(), atol=2e-14)


def test_matrix_two_time_greens_support_noncommuting_nonequilibrium_contacts():
    hamiltonian = np.array(
        [[0.21, 0.09 - 0.04j], [0.09 + 0.04j, -0.16]],
        dtype=np.complex128,
    )
    gamma_left = np.array(
        [[0.42, 0.05j], [-0.05j, 0.24]], dtype=np.complex128
    )
    gamma_right = np.array(
        [[0.23, -0.03], [-0.03, 0.38]], dtype=np.complex128
    )
    transport = MatrixTransportView(
        hamiltonian,
        ["a", "b"],
        LeadSelfEnergy.wide_band(
            gamma_left, mu=0.3, temperature=0.1
        ),
        LeadSelfEnergy.wide_band(
            gamma_right, mu=-0.2, temperature=0.1
        ),
    )
    omega = np.linspace(-35.0, 35.0, 14001)
    result = stationary_greens_two_time(
        transport, np.array([0.0, 0.16, 0.49, 1.1]), omega
    )
    density = result.density_matrices()[0]
    eigenvalues = np.linalg.eigvalsh(density).real

    assert np.linalg.norm(hamiltonian @ gamma_left - gamma_left @ hamiltonian) > 1e-3
    assert result.consistency_report().maximum < 3e-13
    np.testing.assert_allclose(density, density.conj().T, atol=3e-14)
    assert eigenvalues.min() >= -1e-10
    assert eigenvalues.max() <= 1.0 + 1e-10


def test_single_level_retarded_kernel_matches_wide_band_oracle_away_from_jump():
    level_energy = 0.23
    total_gamma = 0.6
    transport = _single_level_transport(level_energy=level_energy)
    omega = np.linspace(-120.0, 120.0, 24001)
    time = np.array([0.0, 0.2, 0.5, 1.0, 1.8])

    result = stationary_greens_two_time(transport, time, omega)
    lag = time[:, None] - time[None, :]
    positive = lag >= 0.2
    expected = -1j * np.exp((-1j * level_energy - 0.5 * total_gamma) * lag)

    np.testing.assert_allclose(
        result.retarded[:, :, 0, 0][positive],
        expected[positive],
        atol=6e-3,
        rtol=2e-3,
    )
    assert np.max(np.abs(result.retarded[:, :, 0, 0][lag <= -0.2])) < 6e-3


def test_two_time_allocation_guard_fails_before_large_result_is_created():
    omega = np.linspace(-1.0, 1.0, 11)
    time = np.linspace(0.0, 1.0, 20)
    values = np.zeros((omega.size, 3, 3), dtype=np.complex128)

    with pytest.raises(MemoryError, match="two-time result allocation"):
        inverse_fourier_two_time(
            omega,
            values,
            time,
            max_memory_bytes=1024,
        )


@pytest.mark.parametrize(
    "omega,time,error",
    [
        (np.array([0.0, 0.0]), np.array([0.0]), "strictly increasing"),
        (np.array([-1.0, 1.0]), np.array([0.0, 0.0]), "strictly increasing"),
    ],
)
def test_two_time_grids_must_be_strictly_increasing(omega, time, error):
    values = np.zeros((omega.size, 1, 1), dtype=np.complex128)
    with pytest.raises(ValueError, match=error):
        inverse_fourier_two_time(omega, values, time)
