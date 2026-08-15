import numpy as np
import pytest

from quantum_transport import (
    equilibrium_one_body_density,
    iterate_density_matrices,
    one_body_bond_current,
    propagate_density_matrix,
    propagate_unitaries,
    region_interface_current,
    two_time_greens,
)


def test_constant_single_level_two_time_components_are_analytic():
    epsilon = 0.37
    occupation = 0.31
    time = np.linspace(0.0, 2.0, 21)
    density = np.array([[occupation]], dtype=np.complex128)
    result = two_time_greens(
        time,
        lambda _time: np.array(
            [[epsilon]], dtype=np.complex128
        ),
        density,
    )

    difference = time[:, None] - time[None, :]
    phase = np.exp(-1j * epsilon * difference)
    expected_lesser = 1j * occupation * phase
    expected_greater = -1j * (1.0 - occupation) * phase
    theta = np.tril(np.ones((time.size, time.size)), k=-1)
    theta += 0.5 * np.eye(time.size)
    expected_retarded = -1j * theta * phase

    assert np.allclose(
        result.lesser[:, :, 0, 0], expected_lesser
    )
    assert np.allclose(
        result.greater[:, :, 0, 0], expected_greater
    )
    assert np.allclose(
        result.retarded[:, :, 0, 0], expected_retarded
    )
    assert np.allclose(
        result.advanced,
        result.retarded.transpose(1, 0, 3, 2).conj(),
    )
    assert result.spectral_identity_error() < 1e-13


def test_equal_time_lesser_recovers_propagated_density_matrix():
    time = np.linspace(0.0, 3.0, 61)
    initial_hamiltonian = np.array(
        [[0.2, -0.7], [-0.7, -0.1]], dtype=np.complex128
    )
    density = equilibrium_one_body_density(
        initial_hamiltonian, mu=0.0, temperature=0.15
    )

    def hamiltonian(value):
        phase = 0.4 * np.sin(value)
        return np.array(
            [
                [0.2, -0.7 * np.exp(1j * phase)],
                [-0.7 * np.exp(-1j * phase), -0.1],
            ],
            dtype=np.complex128,
        )

    result = two_time_greens(
        time,
        hamiltonian,
        density,
        components=("lesser", "greater"),
    )
    propagated = propagate_density_matrix(
        density, time, hamiltonian
    )

    assert np.allclose(result.density_matrices(), propagated)
    assert np.ptp(np.trace(propagated, axis1=1, axis2=2).real) < 1e-12
    assert max(
        np.linalg.norm(value - value.conj().T)
        for value in propagated
    ) < 1e-13


def test_streaming_density_matches_stored_midpoint_trajectory():
    time = np.linspace(0.0, 3.0, 121)
    initial_hamiltonian = np.array(
        [[0.2, -0.7], [-0.7, -0.1]], dtype=np.complex128
    )
    density = equilibrium_one_body_density(
        initial_hamiltonian, mu=0.0, temperature=0.15
    )

    def hamiltonian(value):
        phase = 0.4 * min(value, 0.5)
        return np.array(
            [
                [0.2, -0.7 * np.exp(1j * phase)],
                [-0.7 * np.exp(-1j * phase), -0.1],
            ],
            dtype=np.complex128,
        )

    stored = propagate_density_matrix(density, time, hamiltonian)
    streamed = np.array(
        list(iterate_density_matrices(density, time, hamiltonian))
    )

    np.testing.assert_allclose(streamed, stored, atol=3e-14)
    assert np.ptp(np.trace(streamed, axis1=1, axis2=2).real) < 3e-13


def test_stationary_hamiltonian_matches_exact_matrix_exponential():
    hamiltonian = np.array(
        [[0.3, -0.8], [-0.8, -0.4]], dtype=np.complex128
    )
    time = np.linspace(0.0, 2.0, 17)
    evolution = propagate_unitaries(
        time, lambda _time: hamiltonian
    )
    energies, states = np.linalg.eigh(hamiltonian)

    for index, value in enumerate(time):
        exact = (
            states * np.exp(-1j * energies * value)
        ) @ states.conj().T
        assert np.allclose(evolution[index], exact, atol=2e-14)


def test_time_dependent_gauge_requires_scalar_connection():
    hopping = 0.9
    charge_profile = np.diag([0.0, 0.37])
    time = np.linspace(0.0, 3.0, 601)

    def phi(value):
        return 0.6 * np.sin(0.8 * value)

    def phi_rate(value):
        return 0.48 * np.cos(0.8 * value)

    reference = np.array(
        [[0.2, -hopping], [-hopping, -0.1]],
        dtype=np.complex128,
    )
    density = equilibrium_one_body_density(
        reference, mu=0.0, temperature=0.2
    )

    def gauge_unitary(value):
        return np.diag(
            np.exp(1j * np.diag(charge_profile) * phi(value))
        )

    def transformed(value):
        unitary = gauge_unitary(value)
        return (
            unitary @ reference @ unitary.conj().T
            - phi_rate(value) * charge_profile
        )

    reference_density = propagate_density_matrix(
        density, time, lambda _time: reference
    )
    initial_unitary = gauge_unitary(time[0])
    transformed_density = propagate_density_matrix(
        initial_unitary @ density @ initial_unitary.conj().T,
        time,
        transformed,
    )
    expected = np.array(
        [
            gauge_unitary(value)
            @ reference_density[index]
            @ gauge_unitary(value).conj().T
            for index, value in enumerate(time)
        ]
    )

    assert np.max(
        np.linalg.norm(
            transformed_density - expected, axis=(1, 2)
        )
    ) < 2e-6


def test_finite_embedding_interface_current_obeys_continuity():
    time = np.linspace(0.0, 4.0, 161)

    def hamiltonian(value):
        central_energy = 0.4 * np.tanh(2.0 * (value - 0.5))
        return np.array(
            [
                [-0.3, -0.55, 0.0],
                [-0.55, central_energy, -0.42],
                [0.0, -0.42, 0.25],
            ],
            dtype=np.complex128,
        )

    initial_hamiltonian = hamiltonian(time[0])
    density = equilibrium_one_body_density(
        initial_hamiltonian, mu=0.0, temperature=0.18
    )
    propagated = propagate_density_matrix(
        density, time, hamiltonian
    )

    for value, rho in zip(time, propagated):
        matrix = hamiltonian(value)
        exact_derivative = -1j * (
            matrix @ rho - rho @ matrix
        )
        central_rate = float(np.real(exact_derivative[1, 1]))
        outgoing = region_interface_current(
            matrix, rho, [1]
        )
        direct = (
            one_body_bond_current(matrix, rho, 1, 0)
            + one_body_bond_current(matrix, rho, 1, 2)
        )
        assert abs(outgoing - direct) < 1e-14
        assert abs(central_rate + outgoing) < 1e-12


def test_two_time_memory_guard_and_component_selection():
    time = np.linspace(0.0, 1.0, 20)
    density = np.array([[0.5]], dtype=np.complex128)
    hamiltonian = lambda _time: np.array(
        [[0.0]], dtype=np.complex128
    )

    selected = two_time_greens(
        time,
        hamiltonian,
        density,
        components=("lesser",),
    )
    assert selected.lesser is not None
    assert selected.retarded is None
    with pytest.raises(ValueError, match="was not requested"):
        selected.component("retarded")
    with pytest.raises(MemoryError, match="allocation estimate"):
        two_time_greens(
            time,
            hamiltonian,
            density,
            max_memory_bytes=1,
        )
