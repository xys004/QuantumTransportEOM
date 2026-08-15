import numpy as np
import sympy as sp

from quantum_transport import (
    continuity_residual_after_initial_correlation,
    equilibrium_bosonic_matsubara_green,
    initial_correlation_charge_spin_source,
    kadanoff_baym_initial_correlation_symbolic,
    kadanoff_baym_contour_lesser_dyson_symbolic,
    kadanoff_baym_lesser_initial_correlation_symbolic,
    kbe_initial_correlation_kernel,
    kbe_lesser_contour_correction,
    kbe_lesser_initial_correlation,
    mixed_kbe_residual,
    propagate_mixed_kbe_rceil,
    project_initial_correlation_source,
    required_initial_source_from_residual,
)


def test_mixed_keldysh_branch_returns_hermitian_equal_time_source():
    time = np.linspace(0.0, 1.0, 5)
    imaginary = np.linspace(0.0, 2.0, 9)
    sigma = np.zeros((time.size, imaginary.size, 2, 2), dtype=complex)
    green = np.zeros((imaginary.size, time.size, 2, 2), dtype=complex)
    sigma[..., 0, 0] = 0.4 + 0.2j
    sigma[..., 1, 1] = -0.3j
    green[..., 0, 0] = 0.7j
    green[..., 1, 1] = 0.2 - 0.1j
    result = kbe_initial_correlation_kernel(
        time,
        imaginary,
        self_energy_mixed=sigma,
        green_mixed=green,
    )
    assert result.kernel.shape == (time.size, time.size, 2, 2)
    assert result.hermiticity_error < 1e-14
    np.testing.assert_allclose(
        result.density_source,
        result.density_source.swapaxes(-1, -2).conj(),
        atol=1e-14,
    )


def test_initial_source_removes_a_declared_continuity_residual():
    time = np.linspace(0.0, 1.0, 4)
    source = np.zeros((time.size, 1, 1), dtype=complex)
    source[:, 0, 0] = np.linspace(0.1, 0.4, time.size)
    residual = source.copy()
    corrected = continuity_residual_after_initial_correlation(residual, source)
    np.testing.assert_allclose(corrected, 0.0, atol=1e-15)


def test_required_source_diagnostic_is_hermitian_but_explicitly_residual_based():
    residual = np.array([[[0.3 + 0.2j, 0.1 - 0.4j], [0.5 + 0.7j, -0.2j]]], dtype=complex)
    source = required_initial_source_from_residual(residual)
    np.testing.assert_allclose(source, source.swapaxes(-1, -2).conj(), atol=1e-15)
    np.testing.assert_allclose(source[0, 0, 0], 0.3)


def test_symbolic_initial_correlation_keeps_vertical_branch_and_adjoint():
    t, tp, tau, beta = sp.symbols("t t_prime tau beta", real=True, positive=True)
    formulas = kadanoff_baym_initial_correlation_symbolic(
        time=t,
        time_prime=tp,
        imaginary_time=tau,
        beta=beta,
    )
    assert "Sigma_rceil(t, tau)" in str(formulas["mixed_kernel"])
    assert "G_lceil(tau, t_prime)" in str(formulas["mixed_kernel"])
    assert "dagger" in str(formulas["density_source"])


def test_mixed_kbe_residual_closes_free_homogeneous_branches():
    time = np.linspace(0.0, 0.8, 81)
    imaginary = np.linspace(0.0, 2.0, 9)
    h = np.diag([0.3, -0.2]).astype(complex)
    diagonal = np.diag([0.7, -0.4]).astype(complex)
    evolution = np.exp(-1j * time[:, None] * np.diag(h)[None, :])
    green_rceil = np.zeros((time.size, imaginary.size, 2, 2), dtype=complex)
    green_lceil = np.zeros((imaginary.size, time.size, 2, 2), dtype=complex)
    for left in range(time.size):
        green_rceil[left] = np.diag(evolution[left]) @ np.broadcast_to(diagonal, (imaginary.size, 2, 2))
    for right in range(time.size):
        green_lceil[:, right] = np.broadcast_to(diagonal, (imaginary.size, 2, 2)) @ np.diag(evolution[right].conj())
    zero_rr = np.zeros((time.size, time.size, 2, 2), dtype=complex)
    zero_rm = np.zeros_like(green_rceil)
    zero_mm = np.zeros((imaginary.size, imaginary.size, 2, 2), dtype=complex)
    residual = mixed_kbe_residual(
        time,
        imaginary,
        green_mixed=green_rceil,
        self_energy_retarded=zero_rr,
        self_energy_mixed=zero_rm,
        green_matsubara=zero_mm,
        hamiltonian=h,
        green_lmixed=green_lceil,
    )
    assert residual.maximum_rceil < 2e-3
    assert residual.maximum_lceil is not None and residual.maximum_lceil < 2e-3


def test_initial_correlation_source_projects_to_charge_and_spin():
    source = np.array(
        [
            [[0.2, 0.1j], [-0.1j, -0.05]],
            [[0.3, 0.0], [0.0, 0.1]],
        ],
        dtype=complex,
    )
    spin_z = np.diag([1.0, -1.0]).astype(complex)
    projections = initial_correlation_charge_spin_source(source, spin_z)
    np.testing.assert_allclose(projections["charge"], [0.15, 0.4])
    np.testing.assert_allclose(projections["spin"], [0.25, 0.2])
    np.testing.assert_allclose(project_initial_correlation_source(source, spin_z), projections["spin"])


def test_mixed_kbe_volterra_stepper_recovers_free_unitary_branch():
    time = np.linspace(0.0, 0.8, 81)
    imaginary = np.linspace(0.0, 2.0, 9)
    h = np.diag([0.3, -0.2]).astype(complex)
    initial = np.diag([0.7, -0.4]).astype(complex)
    zero_rr = np.zeros((time.size, time.size, 2, 2), dtype=complex)
    zero_rm = np.zeros((time.size, imaginary.size, 2, 2), dtype=complex)
    zero_mm = np.zeros((imaginary.size, imaginary.size, 2, 2), dtype=complex)
    propagated = propagate_mixed_kbe_rceil(
        time,
        imaginary,
        initial_green_mixed=np.broadcast_to(initial, (imaginary.size, 2, 2)),
        self_energy_retarded=zero_rr,
        self_energy_mixed=zero_rm,
        green_matsubara=zero_mm,
        hamiltonian=h,
    )
    exact = np.empty_like(propagated)
    for index, value in enumerate(time):
        exact[index] = np.diag(np.exp(-1j * value * np.diag(h))) @ np.broadcast_to(initial, (imaginary.size, 2, 2))
    assert np.max(np.abs(propagated - exact)) < 3e-3


def test_lesser_vertical_initial_correlation_is_causal_and_antihermitian():
    time = np.linspace(0.0, 0.8, 41)
    imaginary = np.linspace(0.0, 2.0, 9)
    h = np.diag([0.3, -0.2]).astype(complex)
    retarded = np.zeros((time.size, time.size, 2, 2), dtype=complex)
    for left, left_time in enumerate(time):
        for right, right_time in enumerate(time):
            if left >= right:
                retarded[left, right] = -1j * np.diag(
                    np.exp(-1j * (left_time - right_time) * np.diag(h))
                )
    sigma = np.zeros((time.size, imaginary.size, 2, 2), dtype=complex)
    sigma[..., 0, 0] = 0.2 + 0.1j
    sigma[..., 1, 1] = -0.15j
    green_lmixed = np.zeros((imaginary.size, time.size, 2, 2), dtype=complex)
    green_lmixed[..., 0, 0] = 0.4 - 0.05j
    green_lmixed[..., 1, 1] = -0.2j
    result = kbe_lesser_initial_correlation(
        time,
        imaginary,
        green_retarded=retarded,
        self_energy_mixed=sigma,
        green_lmixed=green_lmixed,
    )
    assert result.source_kernel.shape == (time.size, time.size, 2, 2)
    assert result.propagated_source.shape == result.source_kernel.shape
    assert result.antihermiticity_error < 1e-14
    zero = kbe_lesser_initial_correlation(
        time,
        imaginary,
        green_retarded=retarded,
        self_energy_mixed=np.zeros_like(sigma),
        green_lmixed=green_lmixed,
    )
    np.testing.assert_allclose(zero.correction, 0.0, atol=1e-14)


def _one_level_retarded(time: np.ndarray, energy: float) -> np.ndarray:
    """``G^r(t,t')`` for a single level in the package ``theta(0)=1/2`` convention."""

    lag = time[:, None] - time[None, :]
    theta = np.tril(np.ones((time.size, time.size)), k=-1) + 0.5 * np.eye(time.size)
    return (-1j * theta * np.exp(-1j * energy * lag))[:, :, None, None]


def test_lesser_vertical_correlation_converges_at_second_order():
    """The causal quadrature must compensate the stored ``theta(0)=1/2``.

    With a constant mixed self-energy and a constant left mixed Green
    function the vertical integral is exact, so the only discretisation error
    left is the causal real-time convolution.  Applying the plain trapezoid
    endpoint weight on top of the stored one half halves the equal-time
    contribution twice and silently degrades this convolution to first order.
    """

    energy, beta, sigma_value, green_value = 1.3, 2.0, 0.35 + 0.2j, 0.4 - 0.05j
    imaginary = np.linspace(0.0, beta, 5)
    prefactor = -1j * beta * sigma_value * green_value

    errors = []
    for n_time in (41, 81, 161):
        time = np.linspace(0.0, 4.0, n_time)
        retarded = _one_level_retarded(time, energy)
        sigma = np.full((n_time, imaginary.size, 1, 1), sigma_value, dtype=complex)
        green_lmixed = np.full((imaginary.size, n_time, 1, 1), green_value, dtype=complex)
        result = kbe_lesser_initial_correlation(
            time,
            imaginary,
            green_retarded=retarded,
            self_energy_mixed=sigma,
            green_lmixed=green_lmixed,
        )
        # C(t,t') = int_0^t dtbar G^r(t,tbar) I, with I constant in both times.
        exact = prefactor * (-(1.0 - np.exp(-1j * energy * time)) / energy)
        errors.append(float(np.max(np.abs(result.propagated_source[:, 0, 0, 0] - exact))))

    assert errors[-1] < 1e-4
    for coarse, fine in zip(errors, errors[1:]):
        assert coarse / fine > 3.5, f"expected second-order refinement, got {errors}"


def test_bosonic_matsubara_kernel_stays_finite_at_large_beta_epsilon():
    """``N`` and ``exp(-eps*tau)`` overflow separately while their product is bounded."""

    beta = 100.0
    imaginary = np.linspace(0.0, beta, 5)
    hamiltonian = np.diag([8.0, 1.0]).astype(complex)  # beta*eps up to 800
    kernel = equilibrium_bosonic_matsubara_green(hamiltonian, imaginary)
    assert np.all(np.isfinite(kernel))
    # At tau-tau' = -beta the exact kernel is -N exp(beta*eps) -> -1.
    np.testing.assert_allclose(np.diag(kernel[0, -1]).real, [-1.0, -1.0], atol=1e-12)

    # Benign scale still reproduces the direct formula exactly.
    beta = 2.0
    imaginary = np.linspace(0.0, beta, 7)
    energies = np.array([0.8, 1.7])
    kernel = equilibrium_bosonic_matsubara_green(np.diag(energies).astype(complex), imaginary)
    occupation = 1.0 / np.expm1(energies * beta)
    for left, tau in enumerate(imaginary):
        for right, tau_prime in enumerate(imaginary):
            delta = tau - tau_prime
            weight = (1.0 + occupation) if delta >= 0.0 else occupation
            expected = -weight * np.exp(-energies * delta)
            np.testing.assert_allclose(np.diag(kernel[left, right]).real, expected, atol=1e-12)


def test_symbolic_lesser_vertical_term_keeps_retarded_propagation_explicit():
    t, tp, tau, tbar, beta = sp.symbols("t t_prime tau t_bar beta", real=True, positive=True)
    formulas = kadanoff_baym_lesser_initial_correlation_symbolic(
        time=t,
        time_prime=tp,
        imaginary_time=tau,
        real_integration_time=tbar,
        beta=beta,
    )
    assert "Sigma_rceil(t, tau)" in str(formulas["source_kernel"])
    assert "G_lceil(tau, t_prime)" in str(formulas["source_kernel"])
    assert "G_r(t, t_bar)" in str(formulas["propagated_source"])
    assert "dagger" in str(formulas["lesser_correction"])


def test_full_contour_lesser_correction_exposes_three_vertical_terms():
    time = np.linspace(0.0, 0.6, 7)
    imaginary = np.linspace(0.0, 2.0, 9)
    dim = 2
    bare_retarded = np.zeros((time.size, time.size, dim, dim), dtype=complex)
    bare_mixed = np.zeros((time.size, imaginary.size, dim, dim), dtype=complex)
    sigma_mixed = np.zeros_like(bare_mixed)
    green_lmixed = np.zeros((imaginary.size, time.size, dim, dim), dtype=complex)
    green_advanced = np.zeros_like(bare_retarded)
    sigma_matsubara = np.zeros((imaginary.size, imaginary.size, dim, dim), dtype=complex)
    bare_retarded[..., 0, 0] = -0.2j
    bare_mixed[..., 0, 0] = 0.3
    green_lmixed[..., 0, 0] = 0.4j
    sigma_mixed[..., 0, 0] = 0.1j
    green_advanced[..., 0, 0] = 0.2j
    sigma_matsubara[..., 0, 0] = 0.05
    result = kbe_lesser_contour_correction(
        time,
        imaginary,
        bare_retarded=bare_retarded,
        bare_mixed=bare_mixed,
        self_energy_mixed=sigma_mixed,
        green_lmixed=green_lmixed,
        green_advanced=green_advanced,
        self_energy_matsubara=sigma_matsubara,
    )
    assert result.mixed_advanced.shape == (time.size, time.size, dim, dim)
    assert result.propagated_mixed.shape == result.mixed_advanced.shape
    assert result.matsubara.shape == result.mixed_advanced.shape
    assert np.all(np.isfinite(result.correction))
    zero = kbe_lesser_contour_correction(
        time,
        imaginary,
        bare_retarded=bare_retarded,
        bare_mixed=bare_mixed,
        self_energy_mixed=np.zeros_like(sigma_mixed),
        green_lmixed=green_lmixed,
        green_advanced=green_advanced,
        self_energy_matsubara=np.zeros_like(sigma_matsubara),
    )
    np.testing.assert_allclose(zero.correction, 0.0, atol=1e-14)
    symbolic = kadanoff_baym_contour_lesser_dyson_symbolic()
    assert "Sigma_lceil" in str(symbolic["mixed_advanced"])
    assert "Sigma_rceil" in str(symbolic["propagated_mixed"])
    assert "Sigma_M" in str(symbolic["matsubara"])
