r"""Partition-free matrix transients for noninteracting wide-band devices.

The device and all leads are contacted and in a common equilibrium state for
``t < 0``.  At ``t = 0`` the device Hamiltonian may jump from ``h_initial`` to
``h_final`` and each lead energy acquires a constant shift ``Delta_alpha``.
For time-independent wide-band broadenings the exact post-quench spectral
amplitude is

.. math::

   A_\alpha(E,t) = U_\alpha(E,t)G_0^r(E)
      + [1-U_\alpha(E,t)]G_f^r(E+\Delta_\alpha),

where

.. math::

   U_\alpha(E,t)=
   \exp\{-i[h_f-i\Gamma/2-(E+\Delta_\alpha)]t\}.

This yields the equal-time density, terminal currents, and full two-time
lesser/greater functions without replacing the Fermi sea by a Markovian source.
It is exact for quadratic devices in the wide-band limit, up to the explicit
energy cutoff and quadrature supplied by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .continuum_two_time import (
    ContinuumTwoTimeGreenResult,
    _guard_allocation,
    _trapezoid_weights,
    _two_time_adjoint,
    _validated_grid,
)
from .greens import fermi_dirac


_COMPLEX_BYTES = np.dtype(np.complex128).itemsize


def _square_matrix(values: Any, *, name: str, dim: int | None = None) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix.")
    if dim is not None and matrix.shape != (dim, dim):
        raise ValueError(f"{name} must have shape ({dim}, {dim}).")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    return matrix


def _hermitian_matrix(values: Any, *, name: str, dim: int | None = None) -> np.ndarray:
    matrix = _square_matrix(values, name=name, dim=dim)
    if not np.allclose(matrix, matrix.conj().T, atol=1e-12, rtol=1e-12):
        raise ValueError(f"{name} must be Hermitian.")
    return matrix


def _broadening_stack(values: Sequence[Any] | np.ndarray, *, dim: int) -> np.ndarray:
    try:
        stack = np.asarray(values, dtype=np.complex128)
    except ValueError as exc:
        raise ValueError("lead_broadenings must be a stack of equally sized matrices.") from exc
    if stack.ndim != 3 or stack.shape[1:] != (dim, dim) or stack.shape[0] == 0:
        raise ValueError("lead_broadenings must have shape (n_leads, dim, dim).")
    if not np.all(np.isfinite(stack)):
        raise ValueError("lead_broadenings must contain only finite values.")
    for index, gamma in enumerate(stack):
        if not np.allclose(gamma, gamma.conj().T, atol=1e-12, rtol=1e-12):
            raise ValueError(f"lead_broadenings[{index}] must be Hermitian.")
        eigenvalues = np.linalg.eigvalsh(gamma)
        if eigenvalues.min() < -1e-10:
            raise ValueError(f"lead_broadenings[{index}] must be positive semidefinite.")
        if np.max(np.abs(gamma)) <= 0.0:
            raise ValueError(f"lead_broadenings[{index}] must not vanish identically.")
    return stack


def _pade13_exponential(matrix: np.ndarray) -> np.ndarray:
    """Scaling-and-squaring Padé(13) fallback for a dense complex matrix."""

    coefficients = np.array(
        [
            64764752532480000.0,
            32382376266240000.0,
            7771770303897600.0,
            1187353796428800.0,
            129060195264000.0,
            10559470521600.0,
            670442572800.0,
            33522128640.0,
            1323241920.0,
            40840800.0,
            960960.0,
            16380.0,
            182.0,
            1.0,
        ]
    )
    theta_13 = 5.371920351148152
    norm_1 = float(np.linalg.norm(matrix, 1))
    squarings = 0 if norm_1 <= theta_13 else int(np.ceil(np.log2(norm_1 / theta_13)))
    scaled = matrix / (2**squarings)
    identity = np.eye(matrix.shape[0], dtype=np.complex128)
    a2 = scaled @ scaled
    a4 = a2 @ a2
    a6 = a4 @ a2
    u = scaled @ (
        a6 @ (coefficients[13] * a6 + coefficients[11] * a4 + coefficients[9] * a2)
        + coefficients[7] * a6
        + coefficients[5] * a4
        + coefficients[3] * a2
        + coefficients[1] * identity
    )
    v = (
        a6 @ (coefficients[12] * a6 + coefficients[10] * a4 + coefficients[8] * a2)
        + coefficients[6] * a6
        + coefficients[4] * a4
        + coefficients[2] * a2
        + coefficients[0] * identity
    )
    result = np.linalg.solve(v - u, v + u)
    for _ in range(squarings):
        result = result @ result
    return result


def _matrix_exponential_stack(generator: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Evaluate ``exp(generator * t)`` once per time, with a stable fallback."""

    eigenvalues, eigenvectors = np.linalg.eig(generator)
    condition = float(np.linalg.cond(eigenvectors))
    if np.isfinite(condition) and condition < 1.0e10:
        inverse = np.linalg.inv(eigenvectors)
        exponentials = np.exp(time[:, None] * eigenvalues[None, :])
        return np.einsum(
            "ij,tj,jk->tik",
            eigenvectors,
            exponentials,
            inverse,
            optimize=True,
        )
    return np.stack([_pade13_exponential(generator * value) for value in time])


def _gamma_factors(gammas: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return compact factors ``R_alpha`` with ``Gamma_alpha = R R^dagger``."""

    roots: list[np.ndarray] = []
    for gamma in gammas:
        values, vectors = np.linalg.eigh(gamma)
        clipped = np.clip(values.real, 0.0, None)
        cutoff = max(1.0, float(clipped.max())) * 1.0e-13
        retained = clipped > cutoff
        roots.append(vectors[:, retained] * np.sqrt(clipped[retained])[None, :])
    return tuple(roots)


@dataclass(frozen=True)
class PartitionFreeWideBandTransient:
    """Equal-time observables after a partition-free matrix quench."""

    time: np.ndarray
    density_matrix: np.ndarray
    current_into_device: np.ndarray
    lead_orbital_current: np.ndarray
    particle_number_rate: np.ndarray
    lead_broadenings: np.ndarray
    bias_shift: np.ndarray
    initial_hamiltonian: np.ndarray
    final_hamiltonian: np.ndarray
    initial_chemical_potential: float
    temperature: float

    @property
    def particle_number(self) -> np.ndarray:
        return np.trace(self.density_matrix, axis1=-2, axis2=-1).real

    @property
    def net_current_into_device(self) -> np.ndarray:
        return self.current_into_device.sum(axis=1)


@dataclass(frozen=True)
class _PartitionFreeSetup:
    time: np.ndarray
    energy: np.ndarray
    initial_hamiltonian: np.ndarray
    final_hamiltonian: np.ndarray
    gammas: np.ndarray
    shifts: np.ndarray
    filling: np.ndarray
    weights: np.ndarray
    initial_effective: np.ndarray
    final_effective: np.ndarray
    final_propagator: np.ndarray
    initial_chemical_potential: float
    temperature: float


def _partition_free_setup(
    time: Any,
    energy: Any,
    *,
    initial_hamiltonian: Any,
    final_hamiltonian: Any | None,
    lead_broadenings: Sequence[Any] | np.ndarray,
    bias_shift: Any,
    initial_chemical_potential: float,
    temperature: float,
) -> _PartitionFreeSetup:
    times = _validated_grid(time, name="time")
    if times[0] < 0.0:
        raise ValueError("time is measured from the quench and must be >= 0.")
    energies = _validated_grid(energy, name="energy", minimum_size=2)
    h_initial = _hermitian_matrix(initial_hamiltonian, name="initial_hamiltonian")
    dim = h_initial.shape[0]
    h_final = (
        h_initial.copy()
        if final_hamiltonian is None
        else _hermitian_matrix(final_hamiltonian, name="final_hamiltonian", dim=dim)
    )
    gammas = _broadening_stack(lead_broadenings, dim=dim)
    shifts = np.asarray(bias_shift, dtype=float)
    if shifts.shape != (gammas.shape[0],) or not np.all(np.isfinite(shifts)):
        raise ValueError("bias_shift must be a finite vector with one entry per lead.")
    mu0 = float(initial_chemical_potential)
    thermal = float(temperature)
    if not np.isfinite(mu0):
        raise ValueError("initial_chemical_potential must be finite.")
    if not np.isfinite(thermal) or thermal < 0.0:
        raise ValueError("temperature cannot be negative.")

    total_gamma = np.sum(gammas, axis=0)
    h_eff_initial = h_initial - 0.5j * total_gamma
    h_eff_final = h_final - 0.5j * total_gamma
    filling = fermi_dirac(energies, mu=mu0, temperature=thermal)
    weights = _trapezoid_weights(energies) / (2.0 * np.pi)
    propagator = _matrix_exponential_stack(-1j * h_eff_final, times)
    return _PartitionFreeSetup(
        time=times,
        energy=energies,
        initial_hamiltonian=h_initial,
        final_hamiltonian=h_final,
        gammas=gammas,
        shifts=shifts,
        filling=filling,
        weights=weights,
        initial_effective=h_eff_initial,
        final_effective=h_eff_final,
        final_propagator=propagator,
        initial_chemical_potential=mu0,
        temperature=thermal,
    )


def _green_values(energy: np.ndarray, effective_hamiltonian: np.ndarray) -> np.ndarray:
    identity = np.eye(effective_hamiltonian.shape[0], dtype=np.complex128)
    return np.linalg.inv(energy[:, None, None] * identity - effective_hamiltonian)


def _amplitude_block(
    setup: _PartitionFreeSetup,
    energy: np.ndarray,
    initial_green: np.ndarray,
    lead_index: int,
) -> np.ndarray:
    shift = setup.shifts[lead_index]
    final_green = _green_values(energy + shift, setup.final_effective)
    difference = initial_green - final_green
    propagated = np.matmul(
        setup.final_propagator[:, None, :, :],
        difference[None, :, :, :],
    )
    phase = np.exp(
        1j * setup.time[:, None] * (energy[None, :] + shift)
    )
    return final_green[None, :, :, :] + phase[:, :, None, None] * propagated


def _energy_block_size(
    *,
    n_energy: int,
    n_time: int,
    dim: int,
    max_memory_bytes: int,
    two_time: bool,
) -> int:
    if max_memory_bytes <= 0:
        raise ValueError("max_memory_bytes must be positive.")
    # The spectral amplitude carries a full time axis even for the equal-time
    # observable path.  Count both the propagated difference and the returned
    # amplitude; the two-time path additionally holds dressed/weighted copies.
    multiplier = 8 * n_time if two_time else 4 * n_time + 8
    bytes_per_energy = max(1, multiplier * dim * dim * _COMPLEX_BYTES)
    if bytes_per_energy > max_memory_bytes:
        raise MemoryError(
            "partition-free energy-block workspace for one energy requires "
            f"about {bytes_per_energy / 1024**2:.3f} MiB, above the "
            f"{max_memory_bytes / 1024**2:.3f} MiB limit."
        )
    return max(1, min(n_energy, max_memory_bytes // bytes_per_energy))


def partition_free_wide_band_matrix_quench(
    time: Any,
    energy: Any,
    *,
    initial_hamiltonian: Any,
    lead_broadenings: Sequence[Any] | np.ndarray,
    bias_shift: Any,
    final_hamiltonian: Any | None = None,
    initial_chemical_potential: float = 0.0,
    temperature: float = 0.0,
    max_memory_bytes: int = 512 * 1024**2,
) -> PartitionFreeWideBandTransient:
    r"""Solve an exact quadratic partition-free step quench in the WBL.

    Currents are positive from each lead into the device and obey
    ``d Tr(rho)/dt = sum_alpha I_alpha``.  The energy grid controls both the
    initially contacted Fermi sea and the wide-band ultraviolet cutoff.
    """

    setup = _partition_free_setup(
        time,
        energy,
        initial_hamiltonian=initial_hamiltonian,
        final_hamiltonian=final_hamiltonian,
        lead_broadenings=lead_broadenings,
        bias_shift=bias_shift,
        initial_chemical_potential=initial_chemical_potential,
        temperature=temperature,
    )
    n_time = setup.time.size
    n_leads, dim, _ = setup.gammas.shape
    output_bytes = (
        n_time * dim * dim * _COMPLEX_BYTES
        + n_time * n_leads * (dim + 1) * np.dtype(float).itemsize
    )
    if output_bytes > max_memory_bytes:
        raise MemoryError("partition-free equal-time output exceeds max_memory_bytes.")
    block_size = _energy_block_size(
        n_energy=setup.energy.size,
        n_time=n_time,
        dim=dim,
        max_memory_bytes=max_memory_bytes - output_bytes,
        two_time=False,
    )

    density = np.zeros((n_time, dim, dim), dtype=np.complex128)
    injection = np.zeros((n_time, n_leads), dtype=float)
    orbital_injection = np.zeros((n_time, n_leads, dim), dtype=float)
    roots = _gamma_factors(setup.gammas)
    for start in range(0, setup.energy.size, block_size):
        stop = min(start + block_size, setup.energy.size)
        local_energy = setup.energy[start:stop]
        occupied_weight = setup.weights[start:stop] * setup.filling[start:stop]
        initial_green = _green_values(local_energy, setup.initial_effective)
        for lead_index, gamma in enumerate(setup.gammas):
            root = roots[lead_index]
            amplitude = _amplitude_block(
                setup, local_energy, initial_green, lead_index
            )
            for time_index in range(n_time):
                local_amplitude = amplitude[time_index]
                dressed = np.matmul(local_amplitude, root[None, :, :])
                density_integrand = np.matmul(
                    dressed, dressed.swapaxes(-1, -2).conj()
                )
                density[time_index] += np.tensordot(
                    occupied_weight,
                    density_integrand,
                    axes=(0, 0),
                )
                trace_gamma_amplitude = np.einsum(
                    "ir,eir->e", root.conj(), dressed, optimize=True
                )
                injection[time_index, lead_index] += float(
                    np.sum(occupied_weight * trace_gamma_amplitude.imag)
                )
                orbital_injection[time_index, lead_index] += np.tensordot(
                    occupied_weight,
                    np.sum(dressed * root.conj()[None, :, :], axis=-1).imag,
                    axes=(0, 0),
                )

    loss = np.einsum("aij,tji->ta", setup.gammas, density, optimize=True).real
    currents = -loss - 2.0 * injection
    lead_orbital_current = np.empty((n_time, n_leads, dim), dtype=float)
    for lead_index, gamma in enumerate(setup.gammas):
        gamma_density = np.matmul(gamma[None, :, :], density)
        density_gamma = np.matmul(density, gamma[None, :, :])
        orbital_loss = -0.5 * np.real(
            np.diagonal(
                gamma_density + density_gamma, axis1=-2, axis2=-1
            )
        )
        lead_orbital_current[:, lead_index] = (
            orbital_loss - 2.0 * orbital_injection[:, lead_index]
        )
    rate = np.sum(currents, axis=1)
    return PartitionFreeWideBandTransient(
        time=setup.time.copy(),
        density_matrix=density,
        current_into_device=currents,
        lead_orbital_current=lead_orbital_current,
        particle_number_rate=rate,
        lead_broadenings=setup.gammas.copy(),
        bias_shift=setup.shifts.copy(),
        initial_hamiltonian=setup.initial_hamiltonian.copy(),
        final_hamiltonian=setup.final_hamiltonian.copy(),
        initial_chemical_potential=setup.initial_chemical_potential,
        temperature=setup.temperature,
    )


def _retarded_after_quench(setup: _PartitionFreeSetup) -> np.ndarray:
    n_time = setup.time.size
    dim = setup.initial_hamiltonian.shape[0]
    result = np.zeros((n_time, n_time, dim, dim), dtype=np.complex128)
    identity = np.eye(dim, dtype=np.complex128)
    lag_matrix = setup.time[:, None] - setup.time[None, :]
    equal_time = np.abs(lag_matrix) <= 1e-14
    result[equal_time] = -0.5j * identity
    positive_lags = np.unique(lag_matrix[lag_matrix > 1e-14])
    if positive_lags.size:
        propagators = _matrix_exponential_stack(
            -1j * setup.final_effective, positive_lags
        )
        for lag, propagator in zip(positive_lags, propagators):
            result[lag_matrix == lag] = -1j * propagator
    return result


def partition_free_wide_band_two_time_greens(
    time: Any,
    energy: Any,
    *,
    initial_hamiltonian: Any,
    lead_broadenings: Sequence[Any] | np.ndarray,
    bias_shift: Any,
    final_hamiltonian: Any | None = None,
    initial_chemical_potential: float = 0.0,
    temperature: float = 0.0,
    max_memory_bytes: int = 512 * 1024**2,
) -> ContinuumTwoTimeGreenResult:
    r"""Return post-quench ``G^{r,a,<,>}(t,t')`` for a WBL device.

    The lesser and greater components retain the full contacted Fermi sea.  A
    finite energy window produces the expected cutoff error in the equal-time
    anticommutator/spectral identity; enlarge the window to converge it.
    """

    setup = _partition_free_setup(
        time,
        energy,
        initial_hamiltonian=initial_hamiltonian,
        final_hamiltonian=final_hamiltonian,
        lead_broadenings=lead_broadenings,
        bias_shift=bias_shift,
        initial_chemical_potential=initial_chemical_potential,
        temperature=temperature,
    )
    n_time = setup.time.size
    n_leads, dim, _ = setup.gammas.shape
    _guard_allocation(n_time, dim, 4, max_memory_bytes)
    result_bytes = 4 * n_time * n_time * dim * dim * _COMPLEX_BYTES
    block_size = _energy_block_size(
        n_energy=setup.energy.size,
        n_time=n_time,
        dim=dim,
        max_memory_bytes=max_memory_bytes - result_bytes,
        two_time=True,
    )
    roots = _gamma_factors(setup.gammas)
    lesser = np.zeros((n_time, n_time, dim, dim), dtype=np.complex128)
    greater = np.zeros_like(lesser)

    for start in range(0, setup.energy.size, block_size):
        stop = min(start + block_size, setup.energy.size)
        local_energy = setup.energy[start:stop]
        local_weights = setup.weights[start:stop]
        occupied_weight = local_weights * setup.filling[start:stop]
        empty_weight = local_weights * (1.0 - setup.filling[start:stop])
        initial_green = _green_values(local_energy, setup.initial_effective)
        for lead_index in range(n_leads):
            amplitude = _amplitude_block(
                setup, local_energy, initial_green, lead_index
            )
            physical_phase = np.exp(
                -1j
                * setup.time[:, None]
                * (local_energy[None, :] + setup.shifts[lead_index])
            )
            dressed = (
                physical_phase[:, :, None, None]
                * (amplitude @ roots[lead_index])
            )
            for weights, target, prefactor in (
                (occupied_weight, lesser, 1j),
                (empty_weight, greater, -1j),
            ):
                weighted = dressed * np.sqrt(weights)[None, :, None, None]
                flattened = weighted.transpose(0, 2, 1, 3).reshape(
                    n_time * dim, -1
                )
                gram = (flattened @ flattened.conj().T).reshape(
                    n_time, dim, n_time, dim
                ).transpose(0, 2, 1, 3)
                target += prefactor * gram

    retarded = _retarded_after_quench(setup)
    advanced = _two_time_adjoint(retarded)
    return ContinuumTwoTimeGreenResult(
        time=setup.time.copy(),
        omega_grid=setup.energy.copy(),
        retarded=retarded,
        advanced=advanced,
        lesser=lesser,
        greater=greater,
    )
