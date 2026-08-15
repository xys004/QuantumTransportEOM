import numpy as np

from quantum_transport import (
    kadanoff_baym_dyson_two_time,
    equilibrium_one_body_density,
    two_time_greens,
    two_time_kbe_collision_integral,
    two_time_kbe_continuity_balance,
    two_time_kbe_continuity_components,
)


def test_zero_self_energy_collision_kernel_is_zero():
    time = np.linspace(0.0, 1.0, 9)
    hamiltonian = np.diag([0.2, -0.1]).astype(complex)
    density = equilibrium_one_body_density(hamiltonian, mu=0.0, temperature=0.2)
    green = two_time_greens(time, lambda _t: hamiltonian, density)
    zero = np.zeros_like(green.retarded)
    collision = two_time_kbe_collision_integral(
        time,
        green_retarded=green.retarded,
        green_lesser=green.lesser,
        self_energy_retarded=zero,
        self_energy_lesser=zero,
    )
    np.testing.assert_allclose(collision, 0.0, atol=1e-14)


def test_static_equilibrium_continuity_balance_has_zero_charge_and_spin_residual():
    time = np.linspace(0.0, 1.0, 17)
    hamiltonian = np.diag([0.2, -0.1]).astype(complex)
    density = equilibrium_one_body_density(hamiltonian, mu=0.0, temperature=0.2)
    green = two_time_greens(time, lambda _t: hamiltonian, density)
    zero = np.zeros_like(green.retarded)
    balance = two_time_kbe_continuity_balance(
        time,
        green_retarded=green.retarded,
        green_lesser=green.lesser,
        hamiltonian=hamiltonian,
        self_energy_retarded=zero,
        self_energy_lesser=zero,
    )
    assert balance.maximum_residual < 2e-14
    sigma_z = np.diag([1.0, -1.0]).astype(complex)
    projected = balance.observable_balance(sigma_z)
    np.testing.assert_allclose(projected["density_rate"], 0.0, atol=2e-14)
    np.testing.assert_allclose(projected["collision_rate"], 0.0, atol=2e-14)
    np.testing.assert_allclose(projected["residual"], 0.0, atol=2e-14)


def test_nonzero_self_energy_collision_is_resolved_by_the_kbe_balance():
    time = np.linspace(0.0, 2.0, 161)
    hamiltonian = np.array([[0.2]], dtype=complex)
    density = equilibrium_one_body_density(hamiltonian, mu=0.0, temperature=0.2)
    bare = two_time_greens(time, lambda _t: hamiltonian, density)
    theta = np.tril(np.ones((time.size, time.size)), k=-1) + 0.5 * np.eye(time.size)
    sigma_r = -1j * 0.4 * theta[:, :, None, None]
    sigma_l = np.zeros_like(sigma_r)
    interacting = kadanoff_baym_dyson_two_time(
        time,
        bare_retarded=bare.retarded,
        bare_lesser=bare.lesser,
        self_energy_retarded=sigma_r,
        self_energy_lesser=sigma_l,
        max_iterations=100,
        mixing=0.7,
        tolerance=1e-11,
    )
    balance = two_time_kbe_continuity_balance(
        time,
        green_retarded=interacting.retarded,
        green_lesser=interacting.lesser,
        hamiltonian=hamiltonian,
        self_energy_retarded=sigma_r,
        self_energy_lesser=sigma_l,
    )
    assert interacting.converged
    assert np.max(np.abs(balance.collision_rate[2:-2])) > 1e-2
    assert np.max(np.abs(balance.residual[2:-2])) < 2e-4


def test_vertical_source_attachment_keeps_raw_and_corrected_residuals_separate():
    time = np.linspace(0.0, 1.0, 9)
    hamiltonian = np.array([[0.2]], dtype=complex)
    density = equilibrium_one_body_density(hamiltonian, mu=0.0, temperature=0.2)
    green = two_time_greens(time, lambda _t: hamiltonian, density)
    zero = np.zeros_like(green.retarded)
    raw = two_time_kbe_continuity_balance(
        time,
        green_retarded=green.retarded,
        green_lesser=green.lesser,
        hamiltonian=hamiltonian,
        self_energy_retarded=zero,
        self_energy_lesser=zero,
    )
    source = np.zeros_like(raw.residual)
    source[:, 0, 0] = np.linspace(0.1, 0.3, time.size)
    attached = two_time_kbe_continuity_balance(
        time,
        green_retarded=green.retarded,
        green_lesser=green.lesser,
        hamiltonian=hamiltonian,
        self_energy_retarded=zero,
        self_energy_lesser=zero,
        initial_correlation_source=source,
    )
    np.testing.assert_allclose(attached.residual, raw.residual, atol=1e-14)
    np.testing.assert_allclose(attached.source_corrected_residual, raw.residual - source, atol=1e-14)
    np.testing.assert_allclose(attached.initial_correlation_source, source, atol=1e-14)


def test_continuity_components_are_additive_and_projectable():
    time = np.linspace(0.0, 1.0, 17)
    hamiltonian = np.diag([0.2, -0.1]).astype(complex)
    density = equilibrium_one_body_density(hamiltonian, mu=0.0, temperature=0.2)
    green = two_time_greens(time, lambda _t: hamiltonian, density)
    zero = np.zeros_like(green.retarded)
    theta = np.tril(np.ones((time.size, time.size)), k=-1) + 0.5 * np.eye(time.size)
    embedding_r = -1j * 0.1 * theta[:, :, None, None] * np.eye(2)[None, None]
    interaction_r = -1j * 0.07 * theta[:, :, None, None] * np.eye(2)[None, None]
    components = two_time_kbe_continuity_components(
        time,
        green_retarded=green.retarded,
        green_lesser=green.lesser,
        hamiltonian=hamiltonian,
        embedding_self_energy_retarded=embedding_r,
        embedding_self_energy_lesser=zero,
        interaction_self_energy_retarded=interaction_r,
        interaction_self_energy_lesser=zero,
    )
    assert components.collision_additivity_error < 2e-14
    sigma_z = np.diag([1.0, -1.0]).astype(complex)
    projected = components.observable_balance(sigma_z)
    assert set(projected) == {"total", "embedding", "interaction"}
    np.testing.assert_allclose(
        projected["total"]["collision_rate"],
        projected["embedding"]["collision_rate"] + projected["interaction"]["collision_rate"],
        atol=2e-14,
    )
