import numpy as np
import sympy as sp

from quantum_transport import (
    equilibrium_matsubara_green,
    equilibrium_one_body_density,
    hubbard_hartree_self_energy_symbolic,
    hubbard_hartree_self_energy_two_time,
    hubbard_second_born_self_energy_matsubara,
    hubbard_second_born_self_energy_mixed,
    hubbard_second_born_self_energy_mixed_symbolic,
    hubbard_second_born_self_energy_symbolic,
    one_body_correlation_symbolic,
    kadanoff_baym_dyson_two_time,
    self_consistent_hubbard_second_born_two_time,
    self_consistent_hubbard_second_born_contour_two_time,
    self_consistent_hubbard_matsubara,
    self_consistent_born_two_time,
    finite_lead_current_current_correlations,
    partition_free_finite_lead_two_time,
    two_time_one_body_correlations,
    ladder_vertex_corrected_one_body_correlations,
    bethe_salpeter_ladder_symbolic,
    two_time_meir_wingreen_charge_spin_currents,
    two_time_convolution,
    two_time_greens,
)


def test_two_time_convolution_uses_nonuniform_trapezoid_weights():
    time = np.array([0.0, 0.2, 0.7, 1.4])
    left = np.empty((time.size, time.size, 1, 1), dtype=complex)
    right = np.empty_like(left)
    for i, t in enumerate(time):
        for k, tau in enumerate(time):
            for j, tp in enumerate(time):
                left[i, k, 0, 0] = t + tau
                right[k, j, 0, 0] = tau + tp
    result = two_time_convolution(left, right, time)
    expected = np.empty_like(result)
    for i, t in enumerate(time):
        for j, tp in enumerate(time):
            expected[i, j, 0, 0] = np.trapezoid((t + time) * (time + tp), time)
    np.testing.assert_allclose(result, expected, atol=2e-15)


def test_two_time_meir_wingreen_charge_spin_channels_share_one_kernel():
    time = np.linspace(0.0, 0.6, 7)
    hamiltonian = np.array([[0.2, 0.08 - 0.03j], [0.08 + 0.03j, -0.15]], dtype=complex)
    density = equilibrium_one_body_density(hamiltonian, mu=0.0, temperature=0.4)
    greens = two_time_greens(time, lambda _: hamiltonian, density)
    sigma_l = np.zeros_like(greens.retarded)
    sigma_l[:, :, 0, 0] = 0.03j
    sigma_l[:, :, 1, 1] = 0.02j
    sigma_l[:, :, 0, 1] = 0.01j
    sigma_l[:, :, 1, 0] = 0.01j
    sigma_a = np.zeros_like(greens.retarded)
    for left in range(time.size):
        sigma_a[left, left] = 0.1j * np.eye(2)
    channels = two_time_meir_wingreen_charge_spin_currents(
        time,
        green_retarded=greens.retarded,
        green_lesser=greens.lesser,
        lead_self_energy_lesser=sigma_l,
        lead_self_energy_advanced=sigma_a,
        spin_operators={
            "sx": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex) / 2.0,
            "sy": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex) / 2.0,
            "sz": np.diag([1.0, -1.0]).astype(complex) / 2.0,
        },
    )
    assert set(channels) == {"charge", "sx", "sy", "sz"}
    assert all(values.shape == time.shape and np.all(np.isfinite(values)) for values in channels.values())
    assert not np.allclose(channels["charge"], channels["sz"])


def test_one_body_charge_spin_correlations_have_wick_variance_and_symbolic_form():
    time = np.linspace(0.0, 0.6, 5)
    hamiltonian = np.array([[0.2]], dtype=complex)
    density = equilibrium_one_body_density(hamiltonian, mu=0.0, temperature=0.4)
    greens = two_time_greens(time, lambda _: hamiltonian, density)
    correlations = two_time_one_body_correlations(
        time,
        green_lesser=greens.lesser,
        green_greater=greens.greater,
        observables={"charge": np.eye(1, dtype=complex)},
    )
    occupation = float(density[0, 0].real)
    np.testing.assert_allclose(correlations["charge"]["charge"][0, 0].real, occupation * (1.0 - occupation), atol=1e-13)
    assert abs(correlations["charge"]["charge"][0, 0].imag) < 1e-13
    symbolic = one_body_correlation_symbolic("A", "B")
    assert "G_lesser" in str(symbolic["connected"])
    assert "G_greater" in str(symbolic["connected"])


def test_particle_hole_ladder_reduces_to_wick_bubble_at_zero_kernel():
    time = np.linspace(0.0, 0.6, 5)
    hamiltonian = np.diag([0.2, -0.1]).astype(complex)
    density = equilibrium_one_body_density(hamiltonian, mu=0.0, temperature=0.4)
    greens = two_time_greens(time, lambda _: hamiltonian, density)
    observables = {
        "charge": np.eye(2, dtype=complex),
        "spin": np.diag([1.0, -1.0]).astype(complex),
    }
    bubble = two_time_one_body_correlations(
        time,
        green_lesser=greens.lesser,
        green_greater=greens.greater,
        observables=observables,
    )
    result = ladder_vertex_corrected_one_body_correlations(
        time,
        observables=observables,
        interaction_kernel=np.zeros((2, 2)),
        channel_names=("charge", "spin"),
        bubble=bubble,
    )
    np.testing.assert_allclose(
        result["correlations"]["charge"]["charge"],
        bubble["charge"]["charge"],
        atol=1e-14,
    )
    assert result["diagnostics"]["maximum_vertex_correction"] < 1e-14


def test_particle_hole_ladder_scalar_matches_symbolic_resummation():
    chi = sp.Symbol("chi")
    interaction = sp.Symbol("U")
    symbolic = bethe_salpeter_ladder_symbolic(chi, interaction)
    expected = chi / (1 - interaction * chi)
    assert sp.simplify(symbolic["corrected"] - expected) == 0


def test_finite_lead_current_current_correlations_are_real_at_equal_time():
    time = np.linspace(0.0, 0.4, 4)
    imaginary = np.linspace(0.0, 5.0, 5)
    finite = partition_free_finite_lead_two_time(
        time,
        imaginary,
        initial_device_hamiltonian=np.array([[0.1]], dtype=complex),
        final_device_hamiltonian=np.array([[0.1]], dtype=complex),
        lead_hamiltonians=(np.array([[0.3]], dtype=complex),),
        coupling_matrices=(np.array([[0.12]], dtype=complex),),
        chemical_potential=0.0,
        temperature=0.2,
    )
    correlations = finite_lead_current_current_correlations(finite, 0)
    charge_noise = correlations["charge"]["charge"]
    assert charge_noise.shape == (time.size, time.size)
    assert np.all(np.isfinite(charge_noise))
    assert np.max(np.abs(charge_noise.diagonal().imag)) < 1e-12
    assert np.min(charge_noise.diagonal().real) > -1e-12


def test_two_time_dyson_zero_self_energy_returns_the_bare_kernels():
    hamiltonian = np.array([[0.2]], dtype=complex)
    density = equilibrium_one_body_density(hamiltonian, mu=0.0, temperature=0.1)
    time = np.linspace(0.0, 0.8, 7)
    bare = two_time_greens(time, lambda _: hamiltonian, density)
    zeros = np.zeros_like(bare.retarded)
    result = kadanoff_baym_dyson_two_time(
        time,
        bare_retarded=bare.retarded,
        bare_lesser=bare.lesser,
        self_energy_retarded=zeros,
        self_energy_lesser=zeros,
    )
    assert result.converged
    np.testing.assert_allclose(result.retarded, bare.retarded, atol=1e-14)
    np.testing.assert_allclose(result.lesser, bare.lesser, atol=1e-14)
    np.testing.assert_allclose(result.greater, bare.greater, atol=1e-14)


def test_two_time_scba_is_keldysh_consistent_and_refines_particle_number_drift():
    hamiltonian = np.array([[0.2]], dtype=complex)
    density = equilibrium_one_body_density(hamiltonian, mu=0.0, temperature=0.1)
    drifts = []
    for points in (9, 17, 33):
        time = np.linspace(0.0, 1.2, points)
        bare = two_time_greens(time, lambda _: hamiltonian, density)
        result = self_consistent_born_two_time(
            time,
            bare_retarded=bare.retarded,
            bare_lesser=bare.lesser,
            coupling=np.array([[0.02]], dtype=complex),
            boson_frequency=0.4,
            boson_temperature=0.1,
            max_iterations=40,
            dyson_iterations=80,
            mixing=0.35,
            tolerance=1e-8,
        )
        assert result.converged
        assert result.spectral_identity_error < 2e-14
        assert result.advanced_adjoint_error < 2e-14
        assert result.lesser_adjoint_error < 2e-9
        assert result.retarded_causality_error < 2e-14
        assert result.equal_time_spectral_sum_error < 3e-6
        assert result.density_hermiticity_error < 2e-9
        assert result.occupation_bounds_violation < 2e-9
        drifts.append(result.particle_number_drift())
    assert drifts[1] < 0.3 * drifts[0]
    assert drifts[2] < 0.3 * drifts[1]


def test_hubbard_second_born_symbolic_and_numeric_closures_are_explicit():
    formulas = hubbard_second_born_self_energy_symbolic()
    assert "U**2" in str(formulas["lesser"])
    assert "G_greater(1, t_prime, t)" in str(formulas["lesser"])

    hamiltonian = np.diag([0.2, -0.15]).astype(complex)
    density = equilibrium_one_body_density(hamiltonian, mu=0.0, temperature=0.2)
    time = np.linspace(0.0, 0.8, 9)
    bare = two_time_greens(time, lambda _: hamiltonian, density)
    result = self_consistent_hubbard_second_born_two_time(
        time,
        bare_retarded=bare.retarded,
        bare_lesser=bare.lesser,
        interaction_u=0.2,
        max_iterations=80,
        dyson_iterations=80,
        mixing=0.2,
        tolerance=1e-7,
    )
    assert result.converged
    assert result.spectral_identity_error < 2e-14
    assert result.self_energy_spectral_identity_error < 2e-12


def test_hubbard_hartree_symbolic_and_delta_collocation_refine_against_static_quench():
    formulas = hubbard_hartree_self_energy_symbolic()
    assert "DiracDelta" in str(formulas["retarded"])
    assert str(formulas["lesser"]) == "0"

    hamiltonian = np.diag([0.2, -0.15]).astype(complex)
    density = equilibrium_one_body_density(hamiltonian, mu=0.0, temperature=0.2)
    errors = []
    for points in (9, 17, 33):
        time = np.linspace(0.0, 1.2, points)
        bare = two_time_greens(time, lambda _: hamiltonian, density)
        hartree = np.broadcast_to(density, (points, 2, 2)).copy()
        sigma_r, sigma_a, sigma_l, sigma_g = hubbard_hartree_self_energy_two_time(
            time,
            density=hartree,
            interaction_u=0.1,
            spin_pairs=((0, 1), (1, 0)),
        )
        result = kadanoff_baym_dyson_two_time(
            time,
            bare_retarded=bare.retarded,
            bare_lesser=bare.lesser,
            self_energy_retarded=sigma_r,
            self_energy_lesser=sigma_l,
            self_energy_advanced=sigma_a,
            max_iterations=160,
            mixing=0.5,
            tolerance=1e-10,
        )
        reference = two_time_greens(
            time,
            lambda _: hamiltonian + np.diag([0.1 * density[1, 1].real, 0.1 * density[0, 0].real]),
            density,
        )
        errors.append(float(np.max(np.abs(result.retarded - reference.retarded))))
        assert np.max(np.abs(sigma_l)) == 0.0
        assert np.max(np.abs(sigma_g)) == 0.0
    assert errors[1] < 0.6 * errors[0]
    assert errors[2] < 0.6 * errors[1]


def test_hubbard_second_born_mixed_kernel_is_explicit_and_scales_as_u_squared():
    formulas = hubbard_second_born_self_energy_mixed_symbolic()
    assert "G_rceil(0, t, tau)" in str(formulas["mixed"])
    assert "G_lceil(1, tau, t)" in str(formulas["mixed"])
    time = np.linspace(0.0, 0.5, 4)
    imaginary = np.linspace(0.0, 2.0, 5)
    rceil = np.zeros((time.size, imaginary.size, 2, 2), dtype=complex)
    lceil = np.zeros((imaginary.size, time.size, 2, 2), dtype=complex)
    rceil[:, :, 0, 0] = 0.3 + 0.1j
    rceil[:, :, 1, 1] = -0.2j
    lceil[:, :, 1, 1] = 0.4 - 0.05j
    sigma_u = hubbard_second_born_self_energy_mixed(
        time,
        imaginary,
        green_rceil=rceil,
        green_lceil=lceil,
        interaction_u=0.2,
        spin_pairs=((0, 1), (1, 0)),
    )
    sigma_2u = hubbard_second_born_self_energy_mixed(
        time,
        imaginary,
        green_rceil=rceil,
        green_lceil=lceil,
        interaction_u=0.4,
        spin_pairs=((0, 1), (1, 0)),
    )
    assert sigma_u.shape == (time.size, imaginary.size, 2, 2)
    assert np.max(np.abs(sigma_u)) > 0.0
    np.testing.assert_allclose(sigma_2u, 4.0 * sigma_u, atol=1e-14)


def test_hubbard_matsubara_self_consistency_exposes_u_scaling_and_kms_diagnostics():
    hamiltonian = np.diag([0.2, -0.15]).astype(complex)
    imaginary = np.linspace(0.0, 2.0, 11)
    bare = equilibrium_matsubara_green(hamiltonian, imaginary, temperature=0.5)
    zero = self_consistent_hubbard_matsubara(
        imaginary,
        bare_green_matsubara=bare,
        interaction_u=0.0,
        max_iterations=10,
        dyson_iterations=20,
    )
    assert zero.converged
    np.testing.assert_allclose(zero.green_matsubara, bare, atol=1e-14)
    np.testing.assert_allclose(zero.self_energy_matsubara, 0.0, atol=1e-14)

    interacting = self_consistent_hubbard_matsubara(
        imaginary,
        bare_green_matsubara=bare,
        interaction_u=0.1,
        max_iterations=60,
        dyson_iterations=60,
        mixing=0.25,
        tolerance=1e-8,
    )
    assert interacting.converged
    assert interacting.iterations <= 60
    assert np.max(np.abs(interacting.interaction_self_energy)) > 1e-8
    assert np.all(np.isfinite(interacting.self_energy_matsubara))
    assert interacting.green_kms_error < 5e-3
    assert interacting.self_energy_kms_error < 1e-4
    sigma_u = hubbard_second_born_self_energy_matsubara(
        imaginary,
        green_matsubara=bare,
        interaction_u=0.1,
    )
    sigma_2u = hubbard_second_born_self_energy_matsubara(
        imaginary,
        green_matsubara=bare,
        interaction_u=0.2,
    )
    np.testing.assert_allclose(sigma_2u, 4.0 * sigma_u, atol=1e-14)


def test_joint_contour_iteration_can_use_self_consistent_matsubara_branch():
    hamiltonian = np.diag([0.2, -0.15]).astype(complex)
    density = equilibrium_one_body_density(hamiltonian, mu=0.0, temperature=0.5)
    time = np.linspace(0.0, 0.3, 7)
    imaginary = np.linspace(0.0, 2.0, 9)
    bare = two_time_greens(time, lambda _: hamiltonian, density)
    initial_mixed = np.diag([0.7, -0.4]).astype(complex)
    bare_mixed = np.empty((time.size, imaginary.size, 2, 2), dtype=complex)
    for index, value in enumerate(time):
        bare_mixed[index] = np.diag(np.exp(-1j * value * np.diag(hamiltonian))) @ np.broadcast_to(
            initial_mixed, (imaginary.size, 2, 2)
        )
    matsubara = equilibrium_matsubara_green(hamiltonian, imaginary, temperature=0.5)
    result = self_consistent_hubbard_second_born_contour_two_time(
        time,
        imaginary,
        bare_retarded=bare.retarded,
        bare_lesser=bare.lesser,
        bare_mixed=bare_mixed,
        green_matsubara=matsubara,
        hamiltonian=hamiltonian,
        interaction_u=0.1,
        max_iterations=30,
        dyson_iterations=60,
        mixing=0.35,
        tolerance=1e-7,
        include_full_contour_lesser=True,
        self_energy_matsubara=np.zeros_like(matsubara),
        self_consistent_matsubara=True,
        matsubara_iterations=60,
        matsubara_dyson_iterations=60,
        matsubara_mixing=0.25,
        matsubara_tolerance=1e-8,
    )
    assert result.converged
    assert result.matsubara_result is not None
    assert result.matsubara_result.converged
    assert result.self_energy_matsubara is not None
    assert np.max(np.abs(result.self_energy_matsubara)) > 1e-8
    assert result.lesser_contour_correction is not None
    assert np.all(np.isfinite(result.lesser_contour_correction.correction))


def test_joint_hubbard_contour_iteration_exposes_real_and_mixed_branches():
    hamiltonian = np.diag([0.2, -0.15]).astype(complex)
    density = equilibrium_one_body_density(hamiltonian, mu=0.0, temperature=0.5)
    time = np.linspace(0.0, 0.4, 9)
    imaginary = np.linspace(0.0, 2.0, 9)
    bare = two_time_greens(time, lambda _: hamiltonian, density)
    initial_mixed = np.diag([0.7, -0.4]).astype(complex)
    bare_mixed = np.empty((time.size, imaginary.size, 2, 2), dtype=complex)
    for index, value in enumerate(time):
        bare_mixed[index] = np.diag(np.exp(-1j * value * np.diag(hamiltonian))) @ np.broadcast_to(
            initial_mixed, (imaginary.size, 2, 2)
        )
    matsubara = equilibrium_matsubara_green(hamiltonian, imaginary, temperature=0.5)
    result = self_consistent_hubbard_second_born_contour_two_time(
        time,
        imaginary,
        bare_retarded=bare.retarded,
        bare_lesser=bare.lesser,
        bare_mixed=bare_mixed,
        green_matsubara=matsubara,
        hamiltonian=hamiltonian,
        interaction_u=0.0,
        max_iterations=30,
        dyson_iterations=40,
        mixing=0.5,
        tolerance=1e-8,
        include_vertical_lesser=True,
    )
    assert result.converged
    assert result.green_rceil is not None
    assert result.green_lceil is not None
    assert result.self_energy_mixed is not None
    assert result.initial_correlation is not None
    assert result.lesser_initial_correlation is not None
    assert result.lesser_initial_correlation.antihermiticity_error < 1e-14
    assert np.max(np.abs(result.lesser_initial_correlation.correction)) < 1e-14
    assert result.green_rceil.shape == bare_mixed.shape
    assert np.max(np.abs(result.self_energy_mixed)) < 1e-14
    assert result.initial_correlation.hermiticity_error < 1e-14
    assert result.spectral_identity_error < 2e-14


def test_joint_contour_iteration_can_select_full_three_term_lesser_reconstruction():
    hamiltonian = np.diag([0.15, -0.12]).astype(complex)
    density = equilibrium_one_body_density(hamiltonian, mu=0.0, temperature=0.45)
    time = np.linspace(0.0, 0.3, 7)
    imaginary = np.linspace(0.0, 1.0 / 0.45, 9)
    bare = two_time_greens(time, lambda _: hamiltonian, density)
    initial_mixed = np.diag([0.65, -0.35]).astype(complex)
    bare_mixed = np.empty((time.size, imaginary.size, 2, 2), dtype=complex)
    for index, value in enumerate(time):
        bare_mixed[index] = np.diag(np.exp(-1j * value * np.diag(hamiltonian))) @ np.broadcast_to(
            initial_mixed, (imaginary.size, 2, 2)
        )
    matsubara = equilibrium_matsubara_green(hamiltonian, imaginary, temperature=0.45)
    result = self_consistent_hubbard_second_born_contour_two_time(
        time,
        imaginary,
        bare_retarded=bare.retarded,
        bare_lesser=bare.lesser,
        bare_mixed=bare_mixed,
        green_matsubara=matsubara,
        hamiltonian=hamiltonian,
        interaction_u=0.2,
        max_iterations=40,
        dyson_iterations=50,
        mixing=0.35,
        tolerance=1e-7,
        include_full_contour_lesser=True,
        self_energy_matsubara=np.zeros((imaginary.size, imaginary.size, 2, 2), dtype=complex),
    )
    assert result.converged
    assert result.lesser_contour_correction is not None
    correction = result.lesser_contour_correction
    assert np.all(np.isfinite(correction.correction))
    assert np.max(np.abs(correction.propagated_mixed)) > 0.0
    assert np.max(np.abs(correction.mixed_advanced)) > 0.0
    assert result.lesser_initial_correlation is None


def test_joint_contour_iteration_accepts_explicit_real_time_embedding_branches():
    hamiltonian = np.diag([0.2, -0.15]).astype(complex)
    density = equilibrium_one_body_density(hamiltonian, mu=0.0, temperature=0.5)
    time = np.linspace(0.0, 0.25, 7)
    imaginary = np.linspace(0.0, 2.0, 9)
    bare = two_time_greens(time, lambda _: hamiltonian, density)
    initial_mixed = np.diag([0.7, -0.4]).astype(complex)
    bare_mixed = np.empty((time.size, imaginary.size, 2, 2), dtype=complex)
    for index, value in enumerate(time):
        bare_mixed[index] = np.diag(np.exp(-1j * value * np.diag(hamiltonian))) @ np.broadcast_to(
            initial_mixed, (imaginary.size, 2, 2)
        )
    matsubara = equilibrium_matsubara_green(hamiltonian, imaginary, temperature=0.5)
    embedding_r = np.zeros_like(bare.retarded)
    embedding_l = np.zeros_like(bare.lesser)
    for left in range(time.size):
        for right in range(left):
            embedding_r[left, right] = -0.02j * np.eye(2)
            embedding_l[left, right] = 0.01j * np.eye(2)
            embedding_l[right, left] = embedding_l[left, right].conj().T
    result = self_consistent_hubbard_second_born_contour_two_time(
        time,
        imaginary,
        bare_retarded=bare.retarded,
        bare_lesser=bare.lesser,
        bare_mixed=bare_mixed,
        green_matsubara=matsubara,
        hamiltonian=hamiltonian,
        interaction_u=0.0,
        embedding_self_energy_retarded=embedding_r,
        embedding_self_energy_lesser=embedding_l,
        embedding_self_energy_mixed=np.zeros_like(bare_mixed),
        include_full_contour_lesser=True,
        max_iterations=20,
        dyson_iterations=30,
        mixing=0.5,
        tolerance=1e-7,
    )
    assert result.converged
    assert result.spectral_identity_error < 2e-14
    assert np.all(np.isfinite(result.lesser))


def test_bosonic_contour_vertical_branches_survive_large_beta_omega():
    """``N`` and ``exp(omega tau)`` overflow separately on the vertical branch.

    Their products are bounded for ``0 <= tau <= beta``, so factoring them
    returns ``0 * inf = nan`` in the mixed and Matsubara branches once
    ``omega/T`` passes about 709 — including the ``T=0`` case, where the
    absorption weight is exactly zero.
    """

    from quantum_transport import bosonic_scba_self_energy_contour

    n_time, n_imaginary = 3, 4
    real = np.linspace(0.0, 0.5, n_time)
    frequency, beta = 1.0, 800.0
    imaginary = np.linspace(0.0, beta, n_imaginary)
    filled = lambda *shape: np.full(shape + (1, 1), 0.1 + 0.05j, dtype=complex)
    vertex = np.array([[0.3]], dtype=complex)

    for temperature in (0.0, 1.0 / beta):
        branches = bosonic_scba_self_energy_contour(
            real,
            imaginary,
            green_lesser=filled(n_time, n_time),
            green_greater=filled(n_time, n_time),
            green_rceil=filled(n_time, n_imaginary),
            green_lceil=filled(n_imaginary, n_time),
            green_matsubara=filled(n_imaginary, n_imaginary),
            coupling=vertex,
            boson_frequency=frequency,
            boson_temperature=temperature,
        )
        for name, values in branches.items():
            assert np.all(np.isfinite(values)), f"branch {name!r} is not finite"

    # A benign scale must still reproduce the direct harmonic factor exactly.
    frequency, beta, temperature = 1.0, 2.0, 0.5
    imaginary = np.linspace(0.0, beta, n_imaginary)
    branches = bosonic_scba_self_energy_contour(
        real,
        imaginary,
        green_lesser=filled(n_time, n_time),
        green_greater=filled(n_time, n_time),
        green_rceil=filled(n_time, n_imaginary),
        green_lceil=filled(n_imaginary, n_time),
        green_matsubara=filled(n_imaginary, n_imaginary),
        coupling=vertex,
        boson_frequency=frequency,
        boson_temperature=temperature,
    )
    delta = imaginary[:, None] - imaginary[None, :]
    occupation = 1.0 / np.expm1(frequency / temperature)
    expected = np.where(
        delta >= 0.0,
        -(occupation + 1.0) * np.exp(-frequency * delta),
        -occupation * np.exp(-frequency * delta),
    )
    scale = vertex[0, 0] * (0.1 + 0.05j) * vertex[0, 0]
    np.testing.assert_allclose(branches["M"][:, :, 0, 0] / scale, expected, atol=1e-14)
