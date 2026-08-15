import numpy as np
import pytest

from quantum_transport import (
    LeadSelfEnergy,
    MatrixTransportView,
    partition_free_resonant_level_bias_quench,
    partition_free_wide_band_matrix_quench,
    partition_free_wide_band_two_time_greens,
    stationary_greens_two_time,
)


def _matrix_quench_parameters():
    initial_hamiltonian = np.array(
        [[0.20, 0.12 - 0.03j], [0.12 + 0.03j, -0.15]],
        dtype=np.complex128,
    )
    final_hamiltonian = np.array(
        [[0.25, 0.08 + 0.04j], [0.08 - 0.04j, -0.10]],
        dtype=np.complex128,
    )
    gamma_left = np.array(
        [[0.35, 0.04j], [-0.04j, 0.18]], dtype=np.complex128
    )
    gamma_right = np.array(
        [[0.22, -0.03], [-0.03, 0.31]], dtype=np.complex128
    )
    return initial_hamiltonian, final_hamiltonian, gamma_left, gamma_right


def test_matrix_partition_free_quench_reproduces_scalar_exact_oracle():
    time = np.linspace(0.0, 2.0, 21)
    energy = np.linspace(-30.0, 30.0, 8001)
    broadenings = np.array([0.3, 0.2])
    shifts = np.array([0.5, -0.5])
    scalar = partition_free_resonant_level_bias_quench(
        time,
        energy,
        level_energy=0.15,
        broadening=broadenings,
        bias_shift=shifts,
        temperature=0.08,
    )
    matrix = partition_free_wide_band_matrix_quench(
        time,
        energy,
        initial_hamiltonian=np.array([[0.15]]),
        lead_broadenings=broadenings[:, None, None],
        bias_shift=shifts,
        temperature=0.08,
        max_memory_bytes=64 * 1024**2,
    )

    np.testing.assert_allclose(
        matrix.density_matrix[:, 0, 0].real,
        scalar.occupation,
        atol=3e-15,
    )
    np.testing.assert_allclose(
        matrix.current_into_device,
        scalar.current_into_level,
        atol=3e-15,
    )
    np.testing.assert_allclose(
        matrix.lead_orbital_current.sum(axis=2),
        matrix.current_into_device,
        atol=3e-15,
    )
    np.testing.assert_allclose(
        matrix.net_current_into_device,
        scalar.occupation_rate,
        atol=3e-15,
    )


def test_matrix_partition_free_quench_obeys_continuity_at_second_order():
    h_initial, h_final, gamma_left, gamma_right = _matrix_quench_parameters()
    energy = np.linspace(-30.0, 30.0, 8001)
    errors = []
    local_errors = []
    for step in (0.04, 0.02):
        time = np.arange(0.0, 1.2 + 0.5 * step, step)
        result = partition_free_wide_band_matrix_quench(
            time,
            energy,
            initial_hamiltonian=h_initial,
            final_hamiltonian=h_final,
            lead_broadenings=np.stack([gamma_left, gamma_right]),
            bias_shift=np.array([0.4, -0.3]),
            temperature=0.1,
            max_memory_bytes=64 * 1024**2,
        )
        derivative = np.gradient(result.particle_number, time, edge_order=2)
        interior = (time > 0.16) & (time < 1.1)
        errors.append(
            np.max(
                np.abs(
                    derivative[interior]
                    - result.net_current_into_device[interior]
                )
            )
        )
        orbital_density = np.real(
            np.diagonal(result.density_matrix, axis1=-2, axis2=-1)
        )
        orbital_derivative = np.gradient(
            orbital_density, time, axis=0, edge_order=2
        )
        internal_rate = np.real(
            np.diagonal(
                -1j
                * (
                    h_final[None] @ result.density_matrix
                    - result.density_matrix @ h_final[None]
                ),
                axis1=-2,
                axis2=-1,
            )
        )
        local_errors.append(
            np.max(
                np.abs(
                    orbital_derivative[interior]
                    - internal_rate[interior]
                    - result.lead_orbital_current.sum(axis=1)[interior]
                )
            )
        )
        np.testing.assert_allclose(
            result.current_into_device[0], np.zeros(2), atol=3e-14
        )
        np.testing.assert_allclose(
            result.lead_orbital_current.sum(axis=2),
            result.current_into_device,
            atol=3e-14,
        )

    assert errors[1] < 3.2e-6
    assert errors[1] < errors[0] / 3.7
    assert local_errors[1] < 7e-6
    assert local_errors[1] < local_errors[0] / 3.5


def test_matrix_partition_free_quench_is_covariant_under_fixed_unitary_basis():
    h_initial, h_final, gamma_left, gamma_right = _matrix_quench_parameters()
    time = np.array([0.0, 0.3, 0.8])
    energy = np.linspace(-20.0, 20.0, 4001)
    shifts = np.array([0.4, -0.3])
    reference = partition_free_wide_band_matrix_quench(
        time,
        energy,
        initial_hamiltonian=h_initial,
        final_hamiltonian=h_final,
        lead_broadenings=np.stack([gamma_left, gamma_right]),
        bias_shift=shifts,
        temperature=0.1,
        max_memory_bytes=64 * 1024**2,
    )
    generator = np.random.default_rng(41)
    unitary, _ = np.linalg.qr(
        generator.normal(size=(2, 2))
        + 1j * generator.normal(size=(2, 2))
    )

    def transformed(matrix):
        return unitary @ matrix @ unitary.conj().T

    changed = partition_free_wide_band_matrix_quench(
        time,
        energy,
        initial_hamiltonian=transformed(h_initial),
        final_hamiltonian=transformed(h_final),
        lead_broadenings=np.stack(
            [transformed(gamma_left), transformed(gamma_right)]
        ),
        bias_shift=shifts,
        temperature=0.1,
        max_memory_bytes=64 * 1024**2,
    )
    np.testing.assert_allclose(
        changed.density_matrix,
        unitary[None] @ reference.density_matrix @ unitary.conj().T[None],
        atol=2e-14,
    )
    np.testing.assert_allclose(
        changed.current_into_device, reference.current_into_device, atol=2e-14
    )
    derivative = np.array([[0.2, 0.04j], [-0.04j, -0.1]])
    reference_force = -np.einsum(
        "tij,ji->t", reference.density_matrix, derivative
    ).real
    changed_force = -np.einsum(
        "tij,ji->t", changed.density_matrix, transformed(derivative)
    ).real
    np.testing.assert_allclose(changed_force, reference_force, atol=2e-14)


def test_matrix_partition_free_long_time_matches_final_stationary_negf():
    h_initial, h_final, gamma_left, gamma_right = _matrix_quench_parameters()
    energy = np.linspace(-50.0, 50.0, 16001)
    shifts = np.array([0.4, -0.3])
    temperature = 0.1
    transient = partition_free_wide_band_matrix_quench(
        np.array([0.0, 25.0]),
        energy,
        initial_hamiltonian=h_initial,
        final_hamiltonian=h_final,
        lead_broadenings=np.stack([gamma_left, gamma_right]),
        bias_shift=shifts,
        temperature=temperature,
        max_memory_bytes=64 * 1024**2,
    )
    stationary = MatrixTransportView(
        h_final,
        ["a", "b"],
        LeadSelfEnergy.wide_band(
            gamma_left, mu=shifts[0], temperature=temperature
        ),
        LeadSelfEnergy.wide_band(
            gamma_right, mu=shifts[1], temperature=temperature
        ),
    )
    stationary_density = (
        -1j
        * np.trapezoid(stationary.lesser_values(energy), energy, axis=0)
        / (2.0 * np.pi)
    )

    np.testing.assert_allclose(
        transient.density_matrix[-1], stationary_density, atol=7e-6
    )
    np.testing.assert_allclose(
        transient.current_into_device[-1],
        np.array(
            [
                stationary.current_from_keldysh(energy, lead="left"),
                stationary.current_from_keldysh(energy, lead="right"),
            ]
        ),
        atol=7e-6,
    )


def test_partition_free_two_time_equilibrium_matches_stationary_continuum():
    time = np.array([0.0, 0.2, 0.5, 1.0])
    energy = np.linspace(-80.0, 80.0, 16001)
    broadenings = np.array([0.3, 0.2])
    left = LeadSelfEnergy.wide_band(
        np.array([[broadenings[0]]]), temperature=0.08
    )
    right = LeadSelfEnergy.wide_band(
        np.array([[broadenings[1]]]), temperature=0.08
    )
    stationary_view = MatrixTransportView(
        np.array([[0.15]], dtype=np.complex128),
        ["level"],
        left,
        right,
    )
    stationary = stationary_greens_two_time(
        stationary_view, time, energy, max_memory_bytes=64 * 1024**2
    )
    transient = partition_free_wide_band_two_time_greens(
        time,
        energy,
        initial_hamiltonian=np.array([[0.15]]),
        lead_broadenings=broadenings[:, None, None],
        bias_shift=np.zeros(2),
        temperature=0.08,
        max_memory_bytes=64 * 1024**2,
    )

    np.testing.assert_allclose(transient.lesser, stationary.lesser, atol=8e-15)
    np.testing.assert_allclose(transient.greater, stationary.greater, atol=8e-15)
    assert transient.consistency_report().maximum < 2.1e-3


def test_partition_free_matrix_quench_validates_physical_inputs_and_memory():
    time = np.array([0.0, 0.1])
    energy = np.linspace(-2.0, 2.0, 21)
    with pytest.raises(ValueError, match="positive semidefinite"):
        partition_free_wide_band_matrix_quench(
            time,
            energy,
            initial_hamiltonian=np.zeros((2, 2)),
            lead_broadenings=np.array([np.diag([0.2, -0.1])]),
            bias_shift=np.zeros(1),
        )
    with pytest.raises(MemoryError, match="output exceeds"):
        partition_free_wide_band_matrix_quench(
            np.linspace(0.0, 1.0, 20),
            energy,
            initial_hamiltonian=np.zeros((3, 3)),
            lead_broadenings=np.array([np.eye(3)]),
            bias_shift=np.zeros(1),
            max_memory_bytes=1024,
        )
