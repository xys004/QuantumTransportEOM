import numpy as np

from quantum_transport import (
    finite_lead_retarded_self_energy,
    finite_lead_spectral_density,
    match_wide_band_broadening_from_finite_lead,
    partition_free_finite_lead_two_time,
    partitioned_finite_lead_two_time,
    two_time_kbe_continuity_balance,
)


def _benchmark():
    h_initial = np.array([[0.2, 0.12 - 0.03j], [0.12 + 0.03j, -0.15]], dtype=complex)
    h_final = np.array([[0.1, 0.08 + 0.04j], [0.08 - 0.04j, -0.05]], dtype=complex)
    leads = [np.diag([-1.4, -0.3]), np.diag([0.2, 0.8])]
    couplings = [
        np.array([[0.1, 0.02j], [0.03, 0.08j]], dtype=complex),
        np.array([[0.06, -0.04j], [0.07, 0.02]], dtype=complex),
    ]
    result = partition_free_finite_lead_two_time(
        np.linspace(0.0, 1.0, 21),
        np.linspace(0.0, 1.0 / 0.35, 41),
        initial_device_hamiltonian=h_initial,
        final_device_hamiltonian=h_final,
        lead_hamiltonians=leads,
        coupling_matrices=couplings,
        lead_shifts=(0.2, -0.15),
        temperature=0.35,
    )
    balance = two_time_kbe_continuity_balance(
        result.time,
        green_retarded=result.retarded,
        green_lesser=result.lesser,
        hamiltonian=result.final_device_hamiltonian,
        self_energy_retarded=result.self_energy_retarded,
        self_energy_lesser=result.self_energy_lesser,
        self_energy_advanced=result.self_energy_advanced,
    )
    return result, balance


def test_finite_lead_partition_free_greens_obey_spectral_identity():
    result, _ = _benchmark()
    assert result.spectral_identity_error < 1e-12
    assert result.green_mixed.shape == (result.imaginary_time.size, result.time.size, 2, 2)
    assert result.self_energy_mixed.shape == (result.time.size, result.imaginary_time.size, 2, 2)
    assert result.green_matsubara.shape == (result.imaginary_time.size, result.imaginary_time.size, 2, 2)
    assert result.self_energy_matsubara.shape == (result.imaginary_time.size, result.imaginary_time.size, 2, 2)


def test_finite_lead_exposes_lead_resolved_embedding_branches():
    result, _ = _benchmark()
    assert len(result.lead_self_energy_retarded) == 2
    np.testing.assert_allclose(
        sum(result.lead_self_energy_retarded), result.self_energy_retarded, atol=2e-14
    )
    np.testing.assert_allclose(
        sum(result.lead_self_energy_lesser), result.self_energy_lesser, atol=2e-14
    )
    for retarded, advanced, lesser, greater in zip(
        result.lead_self_energy_retarded,
        result.lead_self_energy_advanced,
        result.lead_self_energy_lesser,
        result.lead_self_energy_greater,
    ):
        np.testing.assert_allclose(advanced, retarded.swapaxes(0, 1).swapaxes(-1, -2).conj(), atol=2e-14)
        assert np.all(np.isfinite(lesser)) and np.all(np.isfinite(greater))


def test_finite_lead_matsubara_branch_has_kms_endpoint_relation_away_from_jump():
    result, _ = _benchmark()
    tau_index = result.imaginary_time.size // 2
    np.testing.assert_allclose(
        result.green_matsubara[-1, tau_index],
        -result.green_matsubara[0, tau_index],
        atol=2e-12,
        rtol=2e-12,
    )


def test_finite_lead_microscopic_initial_source_closes_continuity_interior():
    result, balance = _benchmark()
    corrected = balance.residual + result.initial_correlation.density_source
    assert np.max(np.abs(corrected[2:-2])) < 3e-4


def test_finite_lead_spin_projection_has_same_initial_source_closure():
    result, balance = _benchmark()
    sigma_z = np.diag([1.0, -1.0]).astype(complex)
    projected = balance.observable_balance(sigma_z)
    source = np.real(np.trace(sigma_z @ result.initial_correlation.density_source, axis1=-2, axis2=-1))
    assert np.max(np.abs(projected["residual"][2:-2] + source[2:-2])) < 3e-4


def test_finite_lead_spectral_density_is_hermitian_and_positive():
    lead = np.diag([-0.8, 0.35]).astype(complex)
    coupling = np.array([[0.12, 0.03j], [0.04, 0.09]], dtype=complex)
    omega = np.linspace(-2.0, 2.0, 401)
    gamma = finite_lead_spectral_density(
        omega,
        lead_hamiltonian=lead,
        coupling_matrix=coupling,
        lead_shift=0.15,
        eta=0.04,
    )
    assert gamma.shape == (omega.size, 2, 2)
    np.testing.assert_allclose(gamma, gamma.swapaxes(-1, -2).conj(), atol=2e-14)
    assert min(np.linalg.eigvalsh(value).min() for value in gamma) > -2e-13


def test_finite_lead_spectral_density_matches_single_level_lorentzian():
    omega = np.linspace(-1.0, 1.0, 301)
    eta = 0.04
    coupling = np.array([[0.3 + 0.1j]], dtype=complex)
    gamma = finite_lead_spectral_density(
        omega,
        lead_hamiltonian=np.array([[0.2]], dtype=complex),
        coupling_matrix=coupling,
        lead_shift=0.15,
        eta=eta,
    )[:, 0, 0].real
    expected = 2.0 * abs(coupling[0, 0]) ** 2 * eta / (
        (omega - 0.35) ** 2 + eta**2
    )
    np.testing.assert_allclose(gamma, expected, atol=2e-14, rtol=2e-13)
    sigma = finite_lead_retarded_self_energy(
        omega,
        lead_hamiltonian=np.array([[0.2]], dtype=complex),
        coupling_matrix=coupling,
        lead_shift=0.15,
        eta=eta,
    )[:, 0, 0]
    np.testing.assert_allclose(sigma.imag, -0.5 * expected, atol=2e-14, rtol=2e-13)


def test_finite_lead_spectral_window_match_returns_positive_constant_gamma():
    lead = np.diag([-0.8, 0.35]).astype(complex)
    coupling = np.array([[0.12, 0.03j], [0.04, 0.09]], dtype=complex)
    gamma = match_wide_band_broadening_from_finite_lead(
        np.linspace(-3.0, 3.0, 601),
        lead_hamiltonian=lead,
        coupling_matrix=coupling,
        lead_shift=0.15,
        chemical_potential=-0.1,
        temperature=0.35,
        eta=0.04,
    )
    np.testing.assert_allclose(gamma, gamma.conj().T, atol=2e-14)
    assert np.linalg.eigvalsh(gamma).min() > -2e-13


def test_partitioned_finite_lead_quench_has_exact_keldysh_branches():
    result = partitioned_finite_lead_two_time(
        np.linspace(0.0, 0.8, 9),
        initial_device_hamiltonian=np.diag([0.1, -0.1]).astype(complex),
        final_device_hamiltonian=np.diag([0.2, -0.05]).astype(complex),
        lead_hamiltonians=(np.diag([-0.6, 0.2]).astype(complex), np.diag([0.4, 0.9]).astype(complex)),
        coupling_matrices=(
            np.array([[0.12, 0.0], [0.0, 0.08]], dtype=complex),
            np.array([[0.07, 0.0], [0.0, 0.09]], dtype=complex),
        ),
        lead_shifts=(0.15, -0.15),
        chemical_potential=-0.2,
        temperature=0.35,
    )
    assert result.spectral_identity_error < 2e-14
    assert len(result.lead_self_energy_retarded) == 2
    assert np.ptp(np.trace(result.density_matrices, axis1=1, axis2=2).real) > 1e-8


def test_current_correlations_accept_a_partitioned_result():
    """The partitioned branch was advertised but crashed on a missing field.

    ``finite_lead_current_current_correlations`` accepts both result types,
    yet only the partition-free one stores the contacted ``final_hamiltonian``.
    The partitioned Hamiltonian has to be rebuilt from its blocks, and the
    rebuild must carry the coupling blocks: without them the lead-current
    vertex ``i[P_lead, H]`` vanishes and every correlation is identically zero.
    """

    from quantum_transport import finite_lead_current_current_correlations

    time = np.linspace(0.0, 0.6, 5)
    device = np.array([[0.1]], dtype=complex)
    lead = np.diag([-0.3, 0.4]).astype(complex)

    def correlations(coupling_value):
        result = partitioned_finite_lead_two_time(
            time,
            initial_device_hamiltonian=device,
            lead_hamiltonians=(lead,),
            coupling_matrices=(np.array([[coupling_value, coupling_value]], dtype=complex),),
            lead_shifts=(0.15,),
            chemical_potential=0.0,
            temperature=0.2,
        )
        return finite_lead_current_current_correlations(result, 0)["charge"]["charge"]

    coupled = correlations(0.25)
    assert coupled.shape == (time.size, time.size)
    assert np.all(np.isfinite(coupled))
    # Equal-time charge autocorrelation of a Hermitian vertex is real.
    assert np.max(np.abs(coupled.diagonal().imag)) < 1e-12
    # The rebuilt Hamiltonian must retain the coupling blocks.
    assert np.max(np.abs(coupled)) > 1e-6
    decoupled = correlations(0.0)
    np.testing.assert_allclose(decoupled, 0.0, atol=1e-14)
