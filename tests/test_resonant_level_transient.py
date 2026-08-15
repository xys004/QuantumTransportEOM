import numpy as np
import pytest

from quantum_transport import (
    resonant_level_spectral_amplitude,
    wide_band_resonant_level_quench,
    zero_temperature_resonant_level_steady_state,
)


def test_spectral_amplitude_obeys_exact_norm_identity():
    energy = np.linspace(-3.0, 3.0, 301)
    time = np.linspace(0.0, 2.0, 17)
    epsilon = 0.27
    gamma = 0.63
    amplitude = resonant_level_spectral_amplitude(
        energy,
        time,
        level_energy=epsilon,
        total_broadening=gamma,
    )
    detuning = energy - epsilon
    exponential = np.exp(
        (1j * detuning[None, :] - 0.5 * gamma)
        * time[:, None]
    )
    derivative = -1j * exponential
    norm_rate = 2.0 * np.real(
        amplitude.conj() * derivative
    )

    np.testing.assert_allclose(
        norm_rate,
        -gamma * np.abs(amplitude) ** 2
        - 2.0 * amplitude.imag,
        atol=2e-14,
    )


def test_quench_has_correct_initial_condition_and_current_sign():
    result = wide_band_resonant_level_quench(
        np.array([0.0, 0.2]),
        np.linspace(-10.0, 10.0, 4001),
        level_energy=0.1,
        broadening=np.array([0.3, 0.2]),
        chemical_potential=np.array([0.4, -0.4]),
        temperature=0.1,
        initial_occupation=0.7,
    )

    assert result.occupation[0] == pytest.approx(0.7)
    np.testing.assert_allclose(
        result.current_into_level[0],
        np.array([-0.21, -0.14]),
        atol=1e-14,
    )
    np.testing.assert_allclose(
        result.net_current_into_level,
        result.occupation_rate,
        atol=0.0,
    )


def test_zero_temperature_steady_state_matches_landauer_current():
    gamma = np.array([0.25, 0.25])
    mu = np.array([0.5, -0.5])
    state = zero_temperature_resonant_level_steady_state(
        level_energy=0.0,
        broadening=gamma,
        chemical_potential=mu,
    )
    total_gamma = float(np.sum(gamma))
    landauer = (
        gamma[0]
        * gamma[1]
        / (np.pi * total_gamma)
        * (
            np.arctan(2.0 * mu[0] / total_gamma)
            - np.arctan(2.0 * mu[1] / total_gamma)
        )
    )

    assert state.occupation == pytest.approx(0.5)
    assert state.current_into_level[0] == pytest.approx(
        landauer, abs=1e-15
    )
    assert state.current_into_level[1] == pytest.approx(
        -landauer, abs=1e-15
    )
    assert np.sum(state.current_into_level) == pytest.approx(
        0.0, abs=1e-15
    )


def test_transient_converges_to_closed_zero_temperature_state():
    energy = np.linspace(-160.0, 160.0, 160001)
    gamma = np.array([0.3, 0.2])
    mu = np.array([0.6, -0.4])
    transient = wide_band_resonant_level_quench(
        np.array([0.0, 30.0]),
        energy,
        level_energy=0.15,
        broadening=gamma,
        chemical_potential=mu,
        temperature=0.0,
    )
    steady = zero_temperature_resonant_level_steady_state(
        level_energy=0.15,
        broadening=gamma,
        chemical_potential=mu,
    )

    assert transient.occupation[-1] == pytest.approx(
        steady.occupation, abs=3e-4
    )
    np.testing.assert_allclose(
        transient.current_into_level[-1],
        steady.current_into_level,
        atol=2e-5,
    )


def test_finite_difference_continuity_converges_at_second_order():
    energy = np.linspace(-30.0, 30.0, 8001)
    errors = []
    for step in (0.01, 0.005):
        time = np.arange(0.0, 1.5 + 0.5 * step, step)
        result = wide_band_resonant_level_quench(
            time,
            energy,
            level_energy=0.15,
            broadening=np.array([0.3, 0.2]),
            chemical_potential=np.array([0.6, -0.4]),
            temperature=0.08,
            max_memory_bytes=16 * 1024**2,
        )
        numerical_rate = np.gradient(
            result.occupation, time, edge_order=2
        )
        interior = (time > 0.3) & (time < 1.45)
        errors.append(
            np.max(
                np.abs(
                    numerical_rate[interior]
                    - result.net_current_into_level[interior]
                )
            )
        )

    assert errors[1] < 7e-5
    assert errors[1] < errors[0] / 3.8


def test_resonant_level_input_validation():
    with pytest.raises(ValueError, match="positive"):
        wide_band_resonant_level_quench(
            np.array([0.0]),
            np.array([-1.0, 1.0]),
            level_energy=0.0,
            broadening=np.array([0.0]),
            chemical_potential=np.array([0.0]),
        )
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        wide_band_resonant_level_quench(
            np.array([0.0]),
            np.array([-1.0, 1.0]),
            level_energy=0.0,
            broadening=np.array([0.2]),
            chemical_potential=np.array([0.0]),
            initial_occupation=1.2,
        )
    with pytest.raises(MemoryError, match="workspace estimate"):
        wide_band_resonant_level_quench(
            np.array([0.0]),
            np.linspace(-1.0, 1.0, 100),
            level_energy=0.0,
            broadening=np.array([0.2]),
            chemical_potential=np.array([0.0]),
            max_memory_bytes=1,
        )
