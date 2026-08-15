import numpy as np
import pytest

from quantum_transport import (
    finite_interacting_partition_free_two_time,
    finite_lead_retarded_embedding,
    lead_coupled_hubbard_i_retarded,
    partition_free_finite_lead_two_time,
)


def _model(interaction: float = 0.6):
    h = np.diag([-0.25, -0.18, -0.8, -0.7, 0.5, 0.6]).astype(complex)
    h[0, 2] = h[2, 0] = 0.25
    h[1, 3] = h[3, 1] = 0.25
    h[0, 4] = h[4, 0] = 0.2
    h[1, 5] = h[5, 1] = 0.2
    final = h.copy()
    final[[0, 1], [0, 1]] += [0.08, 0.16]
    final[[2, 3], [2, 3]] += [0.18, 0.08]
    final[[4, 5], [4, 5]] += [-0.15, -0.05]
    result = finite_interacting_partition_free_two_time(
        np.linspace(0.0, 0.4, 9),
        initial_one_body_hamiltonian=h,
        final_one_body_hamiltonian=final,
        interactions=[(0, 1, interaction)],
        chemical_potential=0.0,
        temperature=0.3,
        device_indices=[0, 1],
        lead_indices=[[2, 3], [4, 5]],
        spin_z=[1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
    )
    return h, result


def test_exact_interacting_two_time_spectral_and_charge_continuity():
    _, result = _model()
    assert result.spectral_identity_error < 1e-12
    assert result.density_hermiticity_error < 1e-12
    np.testing.assert_allclose(
        result.device_rate(), result.lead_current(0) + result.lead_current(1), atol=2e-12
    )


def test_exact_interacting_spin_continuity_without_spin_torque():
    _, result = _model()
    spin_operator = np.diag([1.0, -1.0, 0.0, 0.0, 0.0, 0.0]).astype(complex)
    np.testing.assert_allclose(
        result.device_rate(observable=spin_operator),
        result.lead_current(0, spin=True) + result.lead_current(1, spin=True),
        atol=2e-12,
    )


def test_lead_coupled_hubbard_i_has_exact_noninteracting_control():
    h, result = _model(interaction=0.0)
    energy = np.linspace(-2.0, 2.0, 301)
    embedding = finite_lead_retarded_embedding(
        energy,
        lead_hamiltonians=[h[2:4, 2:4], h[4:6, 4:6]],
        coupling_matrices=[h[:2, 2:4], h[:2, 4:6]],
        eta=0.04,
    )
    approximate = lead_coupled_hubbard_i_retarded(
        energy,
        epsilon=h[0, 0].real,
        interaction_u=0.0,
        opposite_occupation=0.4,
        embedding_retarded=embedding[:, :1, :1],
        eta=0.04,
    )[:, 0, 0]
    exact = result.initial_retarded_frequency(energy, eta=0.04, indices=[0])[:, 0, 0]
    np.testing.assert_allclose(approximate, exact, atol=2e-10)


def test_exact_interacting_mixed_branch_matches_quadratic_reference_and_changes_at_finite_u():
    h, reference = _model(interaction=0.0)
    time = np.linspace(0.0, 0.3, 5)
    imaginary = np.linspace(0.0, 1.0 / 0.3, 9)
    finite = partition_free_finite_lead_two_time(
        time,
        imaginary,
        initial_device_hamiltonian=h[:2, :2],
        final_device_hamiltonian=reference.final_one_body_hamiltonian[:2, :2],
        lead_hamiltonians=[h[2:4, 2:4], h[4:6, 4:6]],
        coupling_matrices=[h[:2, 2:4], h[:2, 4:6]],
        lead_shifts=[0.18, -0.15],
        temperature=0.3,
    )
    kwargs = {
        "time": time,
        "initial_one_body_hamiltonian": finite.initial_hamiltonian,
        "final_one_body_hamiltonian": finite.final_hamiltonian,
        "temperature": 0.3,
        "device_indices": [0, 1],
        "lead_indices": [[2, 3], [4, 5]],
        "spin_z": [1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
        "imaginary_time": imaginary,
    }
    noninteracting = finite_interacting_partition_free_two_time(interactions=[], **kwargs)
    interacting = finite_interacting_partition_free_two_time(interactions=[(0, 1, 0.6)], **kwargs)
    assert noninteracting.green_mixed is not None
    assert interacting.green_rceil is not None
    assert noninteracting.green_mixed.shape == (imaginary.size, time.size, 2, 2)
    np.testing.assert_allclose(noninteracting.green_mixed, finite.green_mixed, atol=2e-12)
    assert np.max(np.abs(interacting.green_mixed - noninteracting.green_mixed)) > 1e-3


def test_mixed_branch_respects_the_kms_bound_at_low_temperature():
    """``exp(tau (E_m - E_n))`` must never be formed apart from the weights.

    ``G^lceil(tau,t)`` is bounded by one for ``0 <= tau <= beta``, but the
    imaginary-time Heisenberg factor alone reaches ``exp(beta * width)``.
    Forming it first destroys the cancellation against the thermal weights
    well before it overflows: with a spectral width near seven the branch was
    wrong by fifteen orders of magnitude at ``T = 0.05`` and by two hundred at
    ``T = 0.01``, with no warning and no NaN.
    """

    hamiltonian = np.diag([1.5, -1.2, 2.0, -1.8]).astype(complex)
    hamiltonian[0, 2] = hamiltonian[2, 0] = 0.3
    hamiltonian[1, 3] = hamiltonian[3, 1] = 0.3
    time = np.linspace(0.0, 0.4, 3)

    for temperature in (0.5, 0.1, 0.05, 0.01):
        imaginary = np.linspace(0.0, 1.0 / temperature, 4)
        result = finite_interacting_partition_free_two_time(
            time,
            initial_one_body_hamiltonian=hamiltonian,
            interactions=((0, 1, 0.5),),
            temperature=temperature,
            chemical_potential=0.0,
            device_indices=(0, 1),
            lead_indices=((2,), (3,)),
            imaginary_time=imaginary,
        )
        mixed = result.green_mixed
        assert np.all(np.isfinite(mixed))
        assert np.max(np.abs(mixed)) <= 1.0 + 1e-9, (
            f"KMS bound violated at T={temperature}: {np.max(np.abs(mixed))}"
        )


def test_mixed_branch_matches_the_analytic_decoupled_limit():
    """Two decoupled modes have a closed-form mixed branch at any temperature."""

    energy, other = -1.8, 2.0
    hamiltonian = np.diag([energy, other]).astype(complex)
    time = np.linspace(0.0, 0.4, 3)

    for temperature in (0.5, 0.1, 0.02):
        beta = 1.0 / temperature
        imaginary = np.linspace(0.0, beta, 4)
        result = finite_interacting_partition_free_two_time(
            time,
            initial_one_body_hamiltonian=hamiltonian,
            interactions=(),
            temperature=temperature,
            chemical_potential=0.0,
            device_indices=(0,),
            lead_indices=((1,),),
            imaginary_time=imaginary,
        )
        # G^lceil_00(tau,t) = -i exp(-xi tau) exp(i xi t) (1 - f(xi)), with the
        # complement evaluated logarithmically so the reference is safe too.
        log_complement = -np.logaddexp(0.0, -energy * beta)
        expected = (
            -1j
            * np.exp(log_complement - energy * imaginary[:, None])
            * np.exp(1j * energy * time[None, :])
        )
        np.testing.assert_allclose(result.green_mixed[:, :, 0, 0], expected, atol=1e-12)


def test_lead_coupled_hubbard_i_solves_the_dyson_equation():
    """``embedding_form='dyson'`` must satisfy ``G^-1 = g_at^-1 - Sigma``.

    The two-pole alternative inserts the embedding into each atomic
    denominator separately.  Both agree at ``n_o = 0`` and ``n_o = 1`` and at
    ``U = 0``, so the existing non-interacting control cannot tell them apart;
    they differ substantially at intermediate occupation.
    """

    energy = np.linspace(-4.0, 4.0, 1601)
    level, interaction, eta = 0.1, 1.0, 0.02
    lead = np.diag([-0.5, 0.5]).astype(complex)
    coupling = np.array([[0.3, 0.3]], dtype=complex)
    embedding = finite_lead_retarded_embedding(
        energy, lead_hamiltonians=[lead], coupling_matrices=[coupling], eta=eta
    )
    sigma = embedding[:, 0, 0]
    argument = energy + 1j * eta

    def green(occupation, form):
        return lead_coupled_hubbard_i_retarded(
            energy,
            epsilon=level,
            interaction_u=interaction,
            opposite_occupation=occupation,
            embedding_retarded=embedding,
            eta=eta,
            embedding_form=form,
        )[:, 0, 0]

    for occupation in (0.0, 0.25, 0.5, 0.75, 1.0):
        atomic = (1.0 - occupation) / (argument - level) + occupation / (
            argument - level - interaction
        )
        dyson = green(occupation, "dyson")
        np.testing.assert_allclose(1.0 / dyson, 1.0 / atomic - sigma, atol=1e-12)

        two_pole = green(occupation, "two_pole")
        if occupation in (0.0, 1.0):
            np.testing.assert_allclose(two_pole, dyson, atol=1e-12)
        else:
            assert np.max(np.abs(two_pole - dyson)) > 1.0

    # Switching the embedding off must return the atomic propagator exactly.
    zero_embedding = np.zeros_like(embedding)
    atomic = 0.7 / (argument - level) + 0.3 / (argument - level - interaction)
    recovered = lead_coupled_hubbard_i_retarded(
        energy,
        epsilon=level,
        interaction_u=interaction,
        opposite_occupation=0.3,
        embedding_retarded=zero_embedding,
        eta=eta,
    )[:, 0, 0]
    np.testing.assert_allclose(recovered, atomic, atol=0.0)

    with pytest.raises(ValueError):
        lead_coupled_hubbard_i_retarded(
            energy,
            epsilon=level,
            interaction_u=interaction,
            opposite_occupation=0.3,
            embedding_retarded=embedding,
            embedding_form="bogus",
        )
