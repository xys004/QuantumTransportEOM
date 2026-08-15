import numpy as np

from quantum_transport import (
    anderson_hubbard_i_green_function,
    atomic_hubbard_u_probabilities,
    atomic_hubbard_u_retarded_frequency,
    atomic_hubbard_u_two_time,
)


def test_atomic_probabilities_are_normalized_and_stable_at_zero_temperature():
    probabilities = atomic_hubbard_u_probabilities(-0.3, 0.2, 1.1, chemical_potential=0.0, temperature=0.0)
    assert np.all(probabilities >= 0.0)
    assert np.isclose(np.sum(probabilities), 1.0)
    assert probabilities[1] == 1.0


def test_same_hubbard_u_retarded_oracle_matches_hubbard_i_with_exact_occupation():
    epsilon_up, epsilon_down, interaction_u = -0.35, 0.18, 1.05
    energy = np.linspace(-3.0, 3.0, 401)
    probabilities = atomic_hubbard_u_probabilities(
        epsilon_up, epsilon_down, interaction_u, chemical_potential=0.07, temperature=0.23
    )
    exact = atomic_hubbard_u_retarded_frequency(
        energy, epsilon_up, epsilon_down, interaction_u, spin="up", eta=0.02,
        chemical_potential=0.07, temperature=0.23,
    )
    hubbard_i = np.asarray([
        complex(anderson_hubbard_i_green_function(
            "up", float(value), 0.02, epsilon_up, epsilon_down, interaction_u,
            occupations={"down": float(probabilities[2] + probabilities[3])},
        ))
        for value in energy
    ])
    np.testing.assert_allclose(hubbard_i, exact, atol=2e-13, rtol=2e-13)


def test_atomic_two_time_components_obey_keldysh_and_equal_time_identities():
    result = atomic_hubbard_u_two_time(
        np.linspace(0.0, 4.0, 81), -0.35, 0.18, 1.05, spin="down",
        chemical_potential=0.07, temperature=0.23,
    )
    assert result.spectral_identity_error < 2e-14
    assert result.advanced_adjoint_error < 2e-14
    assert result.lesser_antihermiticity_error < 2e-14
    assert result.equal_time_lesser_error < 2e-14
