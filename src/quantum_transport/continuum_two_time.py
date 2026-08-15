r"""Stationary continuum-reservoir kernels in the two-time domain.

This module is the bridge between the frequency-domain open-system NEGF
implemented in :mod:`quantum_transport.devices` and explicit ``(t, t')``
arrays.  It performs the inverse Fourier transform

.. math::

   X(t,t') = \int \frac{d\omega}{2\pi}
              e^{-i\omega(t-t')} X(\omega)

on a caller-supplied energy grid.  Consequently the returned kernels are
band-limited by that grid and quadrature-converged, rather than finite-lead
recurrences or a Markov replacement for Fermi reservoirs.

The implementation is deliberately stationary.  It does not yet solve a
general voltage/flux quench with ``Sigma(t,t')`` inside a Kadanoff--Baym
propagator.  Keeping that boundary explicit prevents a stationary Fourier
transform from being mistaken for a partition-free transient solver.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .devices import LeadSelfEnergy, MatrixTransportView
from .greens import fermi_dirac
from .numerics import sigma_stack


_COMPLEX_BYTES = np.dtype(np.complex128).itemsize


def _validated_grid(values: Any, *, name: str, minimum_size: int = 1) -> np.ndarray:
    grid = np.asarray(values, dtype=float)
    if grid.ndim != 1 or grid.size < minimum_size:
        raise ValueError(f"{name} must be a one-dimensional array with at least {minimum_size} entries.")
    if not np.all(np.isfinite(grid)):
        raise ValueError(f"{name} must contain only finite values.")
    if grid.size > 1 and np.any(np.diff(grid) <= 0.0):
        raise ValueError(f"{name} must be strictly increasing.")
    return grid


def _trapezoid_weights(grid: np.ndarray) -> np.ndarray:
    if grid.size < 2:
        raise ValueError("omega_grid must contain at least two entries for quadrature.")
    weights = np.empty_like(grid)
    weights[0] = 0.5 * (grid[1] - grid[0])
    weights[-1] = 0.5 * (grid[-1] - grid[-2])
    if grid.size > 2:
        weights[1:-1] = 0.5 * (grid[2:] - grid[:-2])
    return weights


def _validate_frequency_stack(omega_grid: np.ndarray, values: Any) -> np.ndarray:
    stack = np.asarray(values, dtype=np.complex128)
    if stack.ndim != 3 or stack.shape[0] != omega_grid.size:
        raise ValueError("frequency_values must have shape (n_omega, dim, dim).")
    if stack.shape[1] != stack.shape[2]:
        raise ValueError("frequency_values must contain square matrices.")
    if not np.all(np.isfinite(stack)):
        raise ValueError("frequency_values must contain only finite values.")
    return stack


def _allocation_bytes(n_time: int, dim: int, components: int) -> int:
    return int(components * n_time * n_time * dim * dim * _COMPLEX_BYTES)


def _guard_allocation(n_time: int, dim: int, components: int, max_memory_bytes: int) -> None:
    if max_memory_bytes <= 0:
        raise ValueError("max_memory_bytes must be positive.")
    required = _allocation_bytes(n_time, dim, components)
    if required > max_memory_bytes:
        gib = required / 1024**3
        allowed = max_memory_bytes / 1024**3
        raise MemoryError(
            f"two-time result allocation requires about {gib:.3f} GiB, "
            f"above max_memory_bytes={allowed:.3f} GiB. Reduce the time grid "
            "or request fewer/smaller matrices."
        )


def inverse_fourier_two_time(
    omega_grid: Any,
    frequency_values: Any,
    time: Any,
    *,
    max_memory_bytes: int = 512 * 1024**2,
) -> np.ndarray:
    r"""Transform a matrix frequency stack to ``X(t,t')``.

    ``frequency_values`` has shape ``(n_omega, dim, dim)`` and the returned
    array has shape ``(n_time, n_time, dim, dim)``.  Nonuniform energy grids
    are supported through explicit trapezoid weights.  Pair-time phases are
    evaluated in memory-capped blocks, so the temporary phase matrix never
    scales as ``n_time**2 * n_omega`` all at once.
    """

    omega = _validated_grid(omega_grid, name="omega_grid", minimum_size=2)
    times = _validated_grid(time, name="time")
    stack = _validate_frequency_stack(omega, frequency_values)
    dim = stack.shape[1]
    _guard_allocation(times.size, dim, 1, max_memory_bytes)

    weights = _trapezoid_weights(omega) / (2.0 * np.pi)
    weighted = stack * weights[:, None, None]
    lags = (times[:, None] - times[None, :]).reshape(-1)
    flat = np.empty((lags.size, dim, dim), dtype=np.complex128)

    # One phase row costs n_omega complex numbers.  Reserve at most one half
    # of the budget for phases, leaving room for the returned kernel and BLAS
    # temporaries.  A minimum of one row keeps tiny explicit budgets useful.
    result_bytes = flat.nbytes
    workspace_bytes = max(_COMPLEX_BYTES * omega.size, (max_memory_bytes - result_bytes) // 2)
    pairs_per_block = max(1, int(workspace_bytes // (_COMPLEX_BYTES * omega.size)))
    pairs_per_block = min(pairs_per_block, lags.size)

    for start in range(0, lags.size, pairs_per_block):
        stop = min(start + pairs_per_block, lags.size)
        phase = np.exp(-1j * lags[start:stop, None] * omega[None, :])
        flat[start:stop] = np.tensordot(phase, weighted, axes=(1, 0))

    return flat.reshape(times.size, times.size, dim, dim)


def _two_time_adjoint(values: np.ndarray) -> np.ndarray:
    """Return ``X(t',t)^dagger`` for a two-time matrix stack."""

    return values.swapaxes(0, 1).swapaxes(-1, -2).conj()


@dataclass(frozen=True)
class TwoTimeConsistencyReport:
    """Maximum absolute residuals of the fermionic two-time identities."""

    advanced_adjoint: float
    lesser_antihermiticity: float
    greater_antihermiticity: float
    keldysh_spectral: float

    @property
    def maximum(self) -> float:
        return max(
            self.advanced_adjoint,
            self.lesser_antihermiticity,
            self.greater_antihermiticity,
            self.keldysh_spectral,
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "advanced_adjoint": self.advanced_adjoint,
            "lesser_antihermiticity": self.lesser_antihermiticity,
            "greater_antihermiticity": self.greater_antihermiticity,
            "keldysh_spectral": self.keldysh_spectral,
            "maximum": self.maximum,
        }


def _consistency_report(
    retarded: np.ndarray,
    advanced: np.ndarray,
    lesser: np.ndarray,
    greater: np.ndarray,
) -> TwoTimeConsistencyReport:
    return TwoTimeConsistencyReport(
        advanced_adjoint=float(np.max(np.abs(advanced - _two_time_adjoint(retarded)))),
        lesser_antihermiticity=float(np.max(np.abs(lesser + _two_time_adjoint(lesser)))),
        greater_antihermiticity=float(np.max(np.abs(greater + _two_time_adjoint(greater)))),
        keldysh_spectral=float(np.max(np.abs((retarded - advanced) - (greater - lesser)))),
    )


@dataclass(frozen=True)
class ContinuumTwoTimeSelfEnergy:
    """Band-limited stationary lead self-energy on an explicit time grid."""

    time: np.ndarray
    omega_grid: np.ndarray
    retarded: np.ndarray
    advanced: np.ndarray
    lesser: np.ndarray
    greater: np.ndarray
    lead_name: str

    def consistency_report(self) -> TwoTimeConsistencyReport:
        return _consistency_report(self.retarded, self.advanced, self.lesser, self.greater)


@dataclass(frozen=True)
class ContinuumTwoTimeGreenResult:
    """Open-device Green functions in the explicit two-time domain."""

    time: np.ndarray
    omega_grid: np.ndarray
    retarded: np.ndarray
    advanced: np.ndarray
    lesser: np.ndarray
    greater: np.ndarray

    def consistency_report(self) -> TwoTimeConsistencyReport:
        return _consistency_report(self.retarded, self.advanced, self.lesser, self.greater)

    def density_matrices(self) -> np.ndarray:
        r"""Return ``rho(t) = -i G^<(t,t)`` for every requested time."""

        diagonal = self.lesser[np.arange(self.time.size), np.arange(self.time.size)]
        return -1j * diagonal

    def equal_time_drift(self) -> float:
        """Maximum equal-time density change relative to the first time."""

        density = self.density_matrices()
        return float(np.max(np.abs(density - density[0])))


def _lead_frequency_values(
    lead: LeadSelfEnergy,
    omega: np.ndarray,
    component: str,
    *,
    workers: int | None,
) -> np.ndarray:
    if component == "retarded" and lead.omega_independent:
        return np.broadcast_to(lead.sigma_retarded(0.0), (omega.size, lead.dim, lead.dim))
    if component in {"lesser", "greater"} and lead.omega_independent:
        occupation = fermi_dirac(omega, mu=lead.mu, temperature=lead.temperature)
        factor = 1j * occupation if component == "lesser" else 1j * (occupation - 1.0)
        return factor[:, None, None] * lead.gamma(0.0)
    return sigma_stack(getattr(lead, f"sigma_{component}"), omega, workers=workers)


def stationary_self_energy_two_time(
    lead: LeadSelfEnergy,
    time: Any,
    omega_grid: Any,
    *,
    workers: int | None = None,
    max_memory_bytes: int = 512 * 1024**2,
) -> ContinuumTwoTimeSelfEnergy:
    r"""Return stationary ``Sigma^{r,a,<,>}(t,t')`` for one continuum lead.

    For a nominal wide-band lead the finite ``omega_grid`` is an explicit
    ultraviolet cutoff.  Increasing the cutoff resolves the retarded delta
    kernel more sharply; the Fermi edge controls the long-time lesser/greater
    memory.
    """

    omega = _validated_grid(omega_grid, name="omega_grid", minimum_size=2)
    times = _validated_grid(time, name="time")
    _guard_allocation(times.size, lead.dim, 4, max_memory_bytes)

    transform_budget = max(max_memory_bytes // 2, _allocation_bytes(times.size, lead.dim, 1) + 1)
    sigma_r_w = _lead_frequency_values(lead, omega, "retarded", workers=workers)
    sigma_l_w = _lead_frequency_values(lead, omega, "lesser", workers=workers)
    sigma_g_w = _lead_frequency_values(lead, omega, "greater", workers=workers)
    sigma_r = inverse_fourier_two_time(omega, sigma_r_w, times, max_memory_bytes=transform_budget)
    sigma_l = inverse_fourier_two_time(omega, sigma_l_w, times, max_memory_bytes=transform_budget)
    sigma_g = inverse_fourier_two_time(omega, sigma_g_w, times, max_memory_bytes=transform_budget)
    sigma_a = _two_time_adjoint(sigma_r)
    return ContinuumTwoTimeSelfEnergy(
        time=times.copy(),
        omega_grid=omega.copy(),
        retarded=sigma_r,
        advanced=sigma_a,
        lesser=sigma_l,
        greater=sigma_g,
        lead_name=lead.name,
    )


def partition_free_wide_band_self_energy_two_time(
    lead: LeadSelfEnergy,
    time: Any,
    omega_grid: Any,
    *,
    bias_shift: float = 0.0,
    initial_chemical_potential: float | None = None,
    initial_temperature: float | None = None,
    max_memory_bytes: int = 512 * 1024**2,
) -> ContinuumTwoTimeSelfEnergy:
    r"""Return a step-quench partition-free WBL lead kernel.

    The contacted Fermi sea is filled with the pre-quench chemical potential
    while a constant post-quench lead shift contributes the gauge phase

    ``exp[-i bias_shift (t-t')]``

    to every finite-window branch.  In the strict WBL limit the retarded
    phase collapses to the delta kernel and is therefore invisible; retaining
    it on a finite energy window removes a spurious cutoff inconsistency.  This
    is the kernel used by the partition-free
    continuum Green-function constructor and avoids silently replacing a
    voltage quench by a stationary NESS kernel in a KBE continuity audit.
    """

    shift = float(bias_shift)
    if not np.isfinite(shift):
        raise ValueError("bias_shift must be finite.")
    mu = lead.mu if initial_chemical_potential is None else float(initial_chemical_potential)
    temperature = lead.temperature if initial_temperature is None else float(initial_temperature)
    if not np.isfinite(mu) or not np.isfinite(temperature) or temperature < 0.0:
        raise ValueError("initial chemical potential and temperature must be finite, with T >= 0.")
    initial_lead = replace(lead, mu=mu, temperature=temperature)
    stationary = stationary_self_energy_two_time(
        initial_lead,
        time,
        omega_grid,
        max_memory_bytes=max_memory_bytes,
    )
    grid = _validated_grid(time, name="time")
    phase = np.exp(-1j * shift * (grid[:, None] - grid[None, :]))[:, :, None, None]
    return ContinuumTwoTimeSelfEnergy(
        time=stationary.time.copy(),
        omega_grid=stationary.omega_grid.copy(),
        retarded=phase * stationary.retarded,
        advanced=phase * stationary.advanced,
        lesser=phase * stationary.lesser,
        greater=phase * stationary.greater,
        lead_name=lead.name,
    )


def stationary_greens_two_time(
    transport: MatrixTransportView,
    time: Any,
    omega_grid: Any,
    *,
    eta: float = 0.0,
    backend: Any = None,
    workers: int | None = None,
    max_memory_bytes: int = 512 * 1024**2,
) -> ContinuumTwoTimeGreenResult:
    r"""Return stationary open-device ``G^{r,a,<,>}(t,t')``.

    The frequency-domain Dyson and Keldysh equations are evaluated by
    :class:`~quantum_transport.devices.MatrixTransportView`, then transformed
    without replacing the Fermi functions by time-local injection.  ``eta``
    should normally remain zero: a nonzero numerical broadening contributes to
    ``G^r-G^a`` but has no associated lesser/greater bath and therefore changes
    the Keldysh spectral residual reported by :meth:`consistency_report`.
    """

    omega = _validated_grid(omega_grid, name="omega_grid", minimum_size=2)
    times = _validated_grid(time, name="time")
    _guard_allocation(times.size, transport.dim, 4, max_memory_bytes)
    transform_budget = max(max_memory_bytes // 2, _allocation_bytes(times.size, transport.dim, 1) + 1)

    g_r_w = transport.retarded_values(omega, eta=eta, backend=backend, workers=workers)
    g_l_w = transport.lesser_values(omega, eta=eta, backend=backend, workers=workers)
    g_g_w = transport.greater_values(omega, eta=eta, backend=backend, workers=workers)
    g_r = inverse_fourier_two_time(omega, g_r_w, times, max_memory_bytes=transform_budget)
    g_l = inverse_fourier_two_time(omega, g_l_w, times, max_memory_bytes=transform_budget)
    g_g = inverse_fourier_two_time(omega, g_g_w, times, max_memory_bytes=transform_budget)
    g_a = _two_time_adjoint(g_r)
    return ContinuumTwoTimeGreenResult(
        time=times.copy(),
        omega_grid=omega.copy(),
        retarded=g_r,
        advanced=g_a,
        lesser=g_l,
        greater=g_g,
    )
