"""Kadanoff--Baym continuity and collision-integral diagnostics.

For a two-time solution the equal-time density obeys

``d rho/dt = -i [h, rho] + C``

with

``C = G^r * Sigma^< + G^< * Sigma^a - Sigma^r * G^< - Sigma^< * G^a``.

The explicit finite-grid contraction is useful for separating coherent bond
currents, reservoir/interacting collision terms, and spin torque.  It is a
diagnostic layer: the accuracy is controlled by the time grid and by the
supplied self-energy closure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .kadanoff_baym import _stack, _time_grid, two_time_adjoint, two_time_convolution


def _hamiltonian_stack(hamiltonian: Any, time: np.ndarray, dim: int) -> np.ndarray:
    values = np.asarray(hamiltonian, dtype=np.complex128)
    if values.ndim == 2:
        if values.shape != (dim, dim):
            raise ValueError("hamiltonian must match the Green-function dimension.")
        values = np.broadcast_to(values, (time.size, dim, dim)).copy()
    elif values.ndim == 3 and values.shape == (time.size, dim, dim):
        values = values.copy()
    else:
        raise ValueError("hamiltonian must have shape (dim, dim) or (n_time, dim, dim).")
    if not np.all(np.isfinite(values)):
        raise ValueError("hamiltonian must contain finite values.")
    if not np.allclose(values, values.swapaxes(-1, -2).conj(), atol=1e-12, rtol=1e-12):
        raise ValueError("hamiltonian must be Hermitian at every time.")
    return values


def two_time_kbe_collision_integral(
    time: Any,
    *,
    green_retarded: Any,
    green_lesser: Any,
    self_energy_retarded: Any,
    self_energy_lesser: Any,
    self_energy_advanced: Any | None = None,
) -> np.ndarray:
    """Return the finite-grid KBE collision kernel ``C(t,t')``."""

    grid = _time_grid(time)
    retarded = _stack(green_retarded, grid, name="green_retarded")
    dim = retarded.shape[2]
    lesser = _stack(green_lesser, grid, name="green_lesser", dim=dim)
    sigma_r = _stack(self_energy_retarded, grid, name="self_energy_retarded", dim=dim)
    sigma_l = _stack(self_energy_lesser, grid, name="self_energy_lesser", dim=dim)
    sigma_a = two_time_adjoint(sigma_r) if self_energy_advanced is None else _stack(
        self_energy_advanced, grid, name="self_energy_advanced", dim=dim
    )
    advanced = two_time_adjoint(retarded)
    left = two_time_convolution(sigma_r, lesser, grid) + two_time_convolution(sigma_l, advanced, grid)
    right = two_time_convolution(retarded, sigma_l, grid) + two_time_convolution(lesser, sigma_a, grid)
    return right - left


@dataclass(frozen=True)
class TwoTimeContinuityBalance:
    """Equal-time density balance and its coherent/collision decomposition."""

    time: np.ndarray
    density: np.ndarray
    density_rate: np.ndarray
    coherent_rate: np.ndarray
    collision_rate: np.ndarray
    residual: np.ndarray
    initial_correlation_source: np.ndarray | None = None

    @property
    def maximum_residual(self) -> float:
        return float(np.max(np.abs(self.residual)))

    @property
    def maximum_collision_trace(self) -> float:
        trace = np.trace(self.collision_rate, axis1=-2, axis2=-1)
        return float(np.max(np.abs(trace)))

    @property
    def source_corrected_residual(self) -> np.ndarray:
        """Return the residual after subtracting a supplied vertical source.

        ``residual`` always retains the raw real-time KBE convention.  When a
        microscopic ``Sigma^rceil * G^lceil`` source was supplied to the
        balance routine, this property applies the package sign convention
        used by :func:`continuity_residual_after_initial_correlation`.
        """

        if self.initial_correlation_source is None:
            return self.residual.copy()
        return self.residual - self.initial_correlation_source

    @property
    def maximum_source_corrected_residual(self) -> float:
        return float(np.max(np.abs(self.source_corrected_residual)))

    def observable_balance(self, observable: Any) -> dict[str, np.ndarray]:
        """Project the matrix balance onto a Hermitian observable ``O``."""

        operator = np.asarray(observable, dtype=np.complex128)
        dim = self.density.shape[-1]
        if operator.shape != (dim, dim):
            raise ValueError("observable must match the density dimension.")
        if not np.allclose(operator, operator.conj().T, atol=1e-12, rtol=1e-12):
            raise ValueError("observable must be Hermitian.")
        project = lambda values: np.real(np.trace(operator @ values, axis1=-2, axis2=-1))
        return {
            "density_rate": project(self.density_rate),
            "coherent_rate": project(self.coherent_rate),
            "collision_rate": project(self.collision_rate),
            "residual": project(self.residual),
        }


@dataclass(frozen=True)
class TwoTimeContinuityComponents:
    """Continuity balance decomposed into embedding and interaction parts.

    The total balance is evaluated with the exact same sum of self-energy
    kernels that is supplied by the caller.  The component balances are kept
    separate so reservoir injection, interaction collision terms, and their
    charge/spin projections cannot be conflated after the fact.
    """

    total: TwoTimeContinuityBalance
    embedding: TwoTimeContinuityBalance
    interaction: TwoTimeContinuityBalance

    @property
    def collision_additivity_error(self) -> float:
        value = (
            self.total.collision_rate
            - self.embedding.collision_rate
            - self.interaction.collision_rate
        )
        return float(np.max(np.abs(value)))

    @property
    def maximum_total_residual(self) -> float:
        return self.total.maximum_residual

    def observable_balance(self, observable: Any) -> dict[str, dict[str, np.ndarray]]:
        """Project total, embedding, and interaction balances onto ``observable``."""

        return {
            "total": self.total.observable_balance(observable),
            "embedding": self.embedding.observable_balance(observable),
            "interaction": self.interaction.observable_balance(observable),
        }


def two_time_kbe_continuity_components(
    time: Any,
    *,
    green_retarded: Any,
    green_lesser: Any,
    hamiltonian: Any,
    embedding_self_energy_retarded: Any,
    embedding_self_energy_lesser: Any,
    interaction_self_energy_retarded: Any,
    interaction_self_energy_lesser: Any,
    embedding_self_energy_advanced: Any | None = None,
    interaction_self_energy_advanced: Any | None = None,
    initial_correlation_source: Any | None = None,
) -> TwoTimeContinuityComponents:
    """Return one continuity ledger for total, embedding, and interaction Σ.

    This routine is intentionally algebraic: it does not repair a
    non-conserving approximation.  It verifies that the total collision term
    is built from the same embedding-plus-interaction kernels used by the
    component diagnostics and exposes any residual/additivity error.
    """

    embedding_r = np.asarray(embedding_self_energy_retarded, dtype=np.complex128)
    embedding_l = np.asarray(embedding_self_energy_lesser, dtype=np.complex128)
    interaction_r = np.asarray(interaction_self_energy_retarded, dtype=np.complex128)
    interaction_l = np.asarray(interaction_self_energy_lesser, dtype=np.complex128)
    if embedding_r.shape != interaction_r.shape or embedding_l.shape != interaction_l.shape:
        raise ValueError("embedding and interaction self-energies must have matching shapes.")
    total_r = embedding_r + interaction_r
    total_l = embedding_l + interaction_l
    total_a = None
    if embedding_self_energy_advanced is not None or interaction_self_energy_advanced is not None:
        if embedding_self_energy_advanced is None or interaction_self_energy_advanced is None:
            raise ValueError("both component advanced self-energies are required when either is supplied.")
        total_a = np.asarray(embedding_self_energy_advanced, dtype=np.complex128) + np.asarray(
            interaction_self_energy_advanced, dtype=np.complex128
        )
    total = two_time_kbe_continuity_balance(
        time,
        green_retarded=green_retarded,
        green_lesser=green_lesser,
        hamiltonian=hamiltonian,
        self_energy_retarded=total_r,
        self_energy_lesser=total_l,
        self_energy_advanced=total_a,
        initial_correlation_source=initial_correlation_source,
    )
    embedding = two_time_kbe_continuity_balance(
        time,
        green_retarded=green_retarded,
        green_lesser=green_lesser,
        hamiltonian=hamiltonian,
        self_energy_retarded=embedding_r,
        self_energy_lesser=embedding_l,
        self_energy_advanced=embedding_self_energy_advanced,
    )
    interaction = two_time_kbe_continuity_balance(
        time,
        green_retarded=green_retarded,
        green_lesser=green_lesser,
        hamiltonian=hamiltonian,
        self_energy_retarded=interaction_r,
        self_energy_lesser=interaction_l,
        self_energy_advanced=interaction_self_energy_advanced,
    )
    return TwoTimeContinuityComponents(total=total, embedding=embedding, interaction=interaction)


def two_time_kbe_continuity_balance(
    time: Any,
    *,
    green_retarded: Any,
    green_lesser: Any,
    hamiltonian: Any,
    self_energy_retarded: Any,
    self_energy_lesser: Any,
    self_energy_advanced: Any | None = None,
    initial_correlation_source: Any | None = None,
) -> TwoTimeContinuityBalance:
    """Evaluate the equal-time KBE balance and optionally attach a vertical source.

    The returned ``residual`` remains the raw
    ``d rho/dt - coherent_rate - collision_rate``.  If
    ``initial_correlation_source`` is supplied, it is validated and exposed
    through ``source_corrected_residual`` rather than silently changing the
    historical residual convention.
    """

    grid = _time_grid(time)
    retarded = _stack(green_retarded, grid, name="green_retarded")
    dim = retarded.shape[2]
    lesser = _stack(green_lesser, grid, name="green_lesser", dim=dim)
    hamiltonian_stack = _hamiltonian_stack(hamiltonian, grid, dim)
    density = -1j * lesser[np.arange(grid.size), np.arange(grid.size)]
    density = 0.5 * (density + density.swapaxes(-1, -2).conj())
    density_rate = np.gradient(density, grid, axis=0, edge_order=min(2, grid.size - 1))
    coherent_rate = -1j * (hamiltonian_stack @ density - density @ hamiltonian_stack)
    collision_kernel = two_time_kbe_collision_integral(
        grid,
        green_retarded=retarded,
        green_lesser=lesser,
        self_energy_retarded=self_energy_retarded,
        self_energy_lesser=self_energy_lesser,
        self_energy_advanced=self_energy_advanced,
    )
    collision_rate = collision_kernel[np.arange(grid.size), np.arange(grid.size)]
    residual = density_rate - coherent_rate - collision_rate
    source = None
    if initial_correlation_source is not None:
        source = np.asarray(initial_correlation_source, dtype=np.complex128)
        if source.shape != residual.shape or not np.all(np.isfinite(source)):
            raise ValueError("initial_correlation_source must have shape (n_time, dim, dim) and be finite.")
        if not np.allclose(source, source.swapaxes(-1, -2).conj(), atol=1e-8, rtol=1e-8):
            raise ValueError("initial_correlation_source must be Hermitian on every time slice.")
    return TwoTimeContinuityBalance(
        time=grid.copy(),
        density=density,
        density_rate=density_rate,
        coherent_rate=coherent_rate,
        collision_rate=collision_rate,
        residual=residual,
        initial_correlation_source=None if source is None else source.copy(),
    )


__all__ = [
    "TwoTimeContinuityBalance",
    "TwoTimeContinuityComponents",
    "two_time_kbe_collision_integral",
    "two_time_kbe_continuity_balance",
    "two_time_kbe_continuity_components",
]
