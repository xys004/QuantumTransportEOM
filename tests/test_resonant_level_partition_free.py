import numpy as np

from quantum_transport import (
    partition_free_resonant_level_amplitude,
    partition_free_resonant_level_bias_quench,
    wide_band_resonant_level_quench,
    zero_temperature_resonant_level_steady_state,
)


def test_partition_free_amplitude_starts_from_connected_equilibrium():
    energy = np.linspace(-4.0, 4.0, 401)
    time = np.linspace(0.0, 2.0, 21)
    epsilon = 0.23
    gamma = 0.7
    shifts = np.array([0.6, -0.4])
    amplitude = partition_free_resonant_level_amplitude(
        energy,
        time,
        level_energy=epsilon,
        total_broadening=gamma,
        bias_shift=shifts,
    )
    equilibrium = 1.0 / (
        energy - epsilon + 0.5j * gamma
    )

    np.testing.assert_allclose(
        amplitude[0],
        np.broadcast_to(equilibrium, amplitude[0].shape),
        atol=2e-14,
    )
    no_bias = partition_free_resonant_level_amplitude(
        energy,
        time,
        level_energy=epsilon,
        total_broadening=gamma,
        bias_shift=np.zeros(2),
    )
    np.testing.assert_allclose(
        no_bias,
        np.broadcast_to(equilibrium, no_bias.shape),
        atol=2e-14,
    )


def test_partition_free_quench_has_zero_initial_current():
    result = partition_free_resonant_level_bias_quench(
        np.array([0.0, 0.2]),
        np.linspace(-50.0, 50.0, 20001),
        level_energy=0.15,
        broadening=np.array([0.3, 0.2]),
        bias_shift=np.array([0.5, -0.5]),
        initial_chemical_potential=0.0,
        temperature=0.08,
    )

    np.testing.assert_allclose(
        result.current_into_level[0], 0.0, atol=2e-14
    )
    assert 0.0 < result.occupation[0] < 1.0
    np.testing.assert_allclose(
        result.final_chemical_potential,
        np.array([0.5, -0.5]),
    )


def test_partition_free_long_time_recovers_final_steady_state():
    energy = np.linspace(-160.0, 160.0, 160001)
    gamma = np.array([0.3, 0.2])
    shifts = np.array([0.5, -0.5])
    transient = partition_free_resonant_level_bias_quench(
        np.array([0.0, 30.0]),
        energy,
        level_energy=0.15,
        broadening=gamma,
        bias_shift=shifts,
        temperature=0.0,
    )
    steady = zero_temperature_resonant_level_steady_state(
        level_energy=0.15,
        broadening=gamma,
        chemical_potential=shifts,
    )

    assert abs(
        transient.occupation[-1] - steady.occupation
    ) < 5e-4
    np.testing.assert_allclose(
        transient.current_into_level[-1],
        steady.current_into_level,
        atol=1e-5,
    )


def test_partition_free_continuity_converges_at_second_order():
    energy = np.linspace(-40.0, 40.0, 12001)
    errors = []
    for step in (0.02, 0.01):
        time = np.arange(0.0, 2.0 + 0.5 * step, step)
        result = partition_free_resonant_level_bias_quench(
            time,
            energy,
            level_energy=0.15,
            broadening=np.array([0.3, 0.2]),
            bias_shift=np.array([0.5, -0.5]),
            temperature=0.08,
            max_memory_bytes=16 * 1024**2,
        )
        derivative = np.gradient(
            result.occupation, time, edge_order=2
        )
        interior = (time > 0.2) & (time < 1.9)
        errors.append(
            np.max(
                np.abs(
                    derivative[interior]
                    - result.net_current_into_level[interior]
                )
            )
        )

    assert errors[1] < 3e-6
    assert errors[1] < errors[0] / 3.8


def test_symmetric_bias_reversal_swaps_lead_currents():
    time = np.linspace(0.0, 5.0, 101)
    energy = np.linspace(-30.0, 30.0, 12001)
    common = dict(
        level_energy=0.0,
        broadening=np.array([0.25, 0.25]),
        initial_chemical_potential=0.0,
        temperature=0.1,
        max_memory_bytes=16 * 1024**2,
    )
    forward = partition_free_resonant_level_bias_quench(
        time,
        energy,
        bias_shift=np.array([0.5, -0.5]),
        **common,
    )
    reverse = partition_free_resonant_level_bias_quench(
        time,
        energy,
        bias_shift=np.array([-0.5, 0.5]),
        **common,
    )

    np.testing.assert_allclose(
        forward.occupation, reverse.occupation, atol=2e-14
    )
    np.testing.assert_allclose(
        forward.current_into_level[:, ::-1],
        reverse.current_into_level,
        atol=2e-14,
    )


def test_partitioned_and_partition_free_share_long_time_limit():
    time = np.array([0.0, 20.0])
    energy = np.linspace(-80.0, 80.0, 40001)
    gamma = np.array([0.3, 0.2])
    final_mu = np.array([0.5, -0.5])
    partitioned = wide_band_resonant_level_quench(
        time,
        energy,
        level_energy=0.15,
        broadening=gamma,
        chemical_potential=final_mu,
        temperature=0.0,
    )
    partition_free = partition_free_resonant_level_bias_quench(
        time,
        energy,
        level_energy=0.15,
        broadening=gamma,
        bias_shift=final_mu,
        temperature=0.0,
    )

    assert abs(
        partitioned.occupation[-1]
        - partition_free.occupation[-1]
    ) < 2e-4
    np.testing.assert_allclose(
        partitioned.current_into_level[-1],
        partition_free.current_into_level[-1],
        atol=7e-5,
    )
