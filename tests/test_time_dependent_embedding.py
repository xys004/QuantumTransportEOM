from __future__ import annotations

import numpy as np
import pytest

from quantum_transport import (
    equilibrium_one_body_density,
    solve_time_dependent_matrix_embedding,
    two_time_greens,
)


def _problem() -> tuple[np.ndarray, object, np.ndarray, np.ndarray, np.ndarray]:
    hamiltonian = np.array([[0.2, 0.04 - 0.02j], [0.04 + 0.02j, -0.1]], dtype=complex)
    density = equilibrium_one_body_density(hamiltonian, temperature=0.3)
    time = np.linspace(0.0, 0.8, 9)
    bare = two_time_greens(
        time,
        lambda value: hamiltonian + 0.08 * np.sin(1.4 * value) * np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
        density,
    )
    weights = np.empty(time.size)
    weights[0] = 0.5 * (time[1] - time[0])
    weights[-1] = 0.5 * (time[-1] - time[-2])
    weights[1:-1] = 0.5 * (time[2:] - time[:-2])
    gamma = np.diag([0.18, 0.1]).astype(complex)
    retarded = np.zeros_like(bare.retarded)
    lesser = np.zeros_like(bare.retarded)
    amplitudes = 0.7 + 0.3 * np.sin(time)
    for index, amplitude in enumerate(amplitudes):
        retarded[index, index] = -0.5j * amplitude * gamma / weights[index]
        for other, other_amplitude in enumerate(amplitudes):
            lesser[index, other] = 1j * 0.08 * amplitude * other_amplitude * np.exp(
                -1j * 0.4 * (time[index] - time[other])
            ) * gamma
    return time, bare, retarded, lesser, amplitudes


def test_time_dependent_matrix_embedding_preserves_causal_keldysh_identities():
    time, bare, retarded, lesser, amplitudes = _problem()
    result = solve_time_dependent_matrix_embedding(
        time,
        bare_retarded=bare.retarded,
        bare_lesser=bare.lesser,
        embedding_self_energy_retarded=retarded,
        embedding_self_energy_lesser=lesser,
        max_iterations=80,
        mixing=0.4,
        tolerance=1e-9,
    )
    assert result.converged
    assert result.iterations < 80
    assert result.retarded_causality_error < 1e-14
    assert result.advanced_adjoint_error < 1e-14
    assert result.lesser_antihermiticity_error < 1e-14
    assert result.keldysh_spectral_error < 1e-14
    assert result.green_spectral_error < 1e-12
    assert np.max(np.abs(result.green.retarded[1, 1] - result.green.retarded[1, 2])) > 1e-8
    assert np.max(np.abs(amplitudes - amplitudes[0])) > 1e-3


def test_time_dependent_matrix_embedding_rejects_shape_mismatch():
    time, bare, retarded, lesser, _ = _problem()
    with pytest.raises(ValueError, match="embedding_self_energy_lesser"):
        solve_time_dependent_matrix_embedding(
            time,
            bare_retarded=bare.retarded,
            bare_lesser=bare.lesser,
            embedding_self_energy_retarded=retarded,
            embedding_self_energy_lesser=lesser[:, :, :1, :1],
        )
