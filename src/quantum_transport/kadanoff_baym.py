"""Numerical two-time Dyson/Kadanoff--Baym building blocks.

The routines in this module intentionally operate on explicit finite time
grids.  They are a transparent Volterra discretisation of the Langreth Dyson
equations, useful for small EOM/Keldysh benchmarks and for testing memory
kernels before moving a scan to ASTRUM.  They do not replace a production
contour solver: the time step and memory window are caller-controlled and the
fixed-point iteration is guarded by convergence metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
from typing import Any

import numpy as np

from .initial_correlations import (
    InitialCorrelationResult,
    LesserContourCorrectionResult,
    LesserInitialCorrelationResult,
    kbe_initial_correlation_kernel,
    kbe_lesser_contour_correction,
    kbe_lesser_initial_correlation,
    propagate_mixed_kbe_rceil,
)


ComplexStack = np.ndarray


def _time_grid(time: Any) -> np.ndarray:
    grid = np.asarray(time, dtype=float)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("time must be a one-dimensional grid with at least two entries.")
    if not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0.0):
        raise ValueError("time must be finite and strictly increasing.")
    return grid


def _weights(time: np.ndarray) -> np.ndarray:
    weights = np.empty_like(time)
    weights[0] = 0.5 * (time[1] - time[0])
    weights[-1] = 0.5 * (time[-1] - time[-2])
    if time.size > 2:
        weights[1:-1] = 0.5 * (time[2:] - time[:-2])
    return weights


def _stack(value: Any, time: np.ndarray, *, name: str, dim: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.complex128)
    if result.ndim != 4 or result.shape[0] != time.size or result.shape[1] != time.size:
        raise ValueError(f"{name} must have shape (n_time, n_time, dim, dim).")
    if result.shape[2] != result.shape[3]:
        raise ValueError(f"{name} must contain square matrices.")
    if dim is not None and result.shape[2:] != (dim, dim):
        raise ValueError(f"{name} must have matrix shape {(dim, dim)}.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def two_time_adjoint(value: Any) -> np.ndarray:
    """Return ``X(t',t)^dagger`` for an explicit two-time matrix stack."""

    result = np.asarray(value, dtype=np.complex128)
    if result.ndim != 4 or result.shape[0] != result.shape[1] or result.shape[2] != result.shape[3]:
        raise ValueError("value must have shape (n_time, n_time, dim, dim).")
    return result.swapaxes(0, 1).swapaxes(-1, -2).conj()


def two_time_convolution(left: Any, right: Any, time: Any) -> np.ndarray:
    r"""Evaluate ``(A*B)(t,t') = integral dτ A(t,τ)B(τ,t')``.

    The trapezoidal weights are applied only to the integration index.  This
    makes the operation valid for nonuniform grids and keeps the matrix order
    explicit, which is useful when spin-orbit blocks do not commute.
    """

    grid = _time_grid(time)
    a = _stack(left, grid, name="left")
    b = _stack(right, grid, name="right", dim=a.shape[2])
    return np.einsum("ikab,kjbc,k->ijac", a, b, _weights(grid), optimize=True)


def two_time_greens_statistics(
    time: Any,
    hamiltonian_at_time: Any,
    *,
    initial_lesser: Any,
    source_matrix: Any,
    max_memory_bytes: int = 512 * 1024**2,
) -> Any:
    r"""Build free two-time Green functions for a graded operator basis.

    ``source_matrix`` is the equal-time graded source

    ``C_ij = <[O_i, O_j^dagger]_+>`` for odd (fermionic) operators and
    ``C_ij = <[O_i, O_j^dagger]_- >`` for even (bosonic) operators.  The
    initial lesser kernel is supplied directly, so bosonic occupations are not
    forced into the fermionic ``0 <= rho <= 1`` interval.

    The spectral identity is reconstructed as ``G^> = G^< + G^r - G^a``;
    this is the common Keldysh discontinuity and does not assume fermionic
    statistics.  The finite-grid evolution still uses the package's unitary
    midpoint propagator, so the numerical generator must be Hermitian.
    """

    from .transient import TwoTimeGreenResult, propagate_unitaries

    grid = _time_grid(time)
    source = np.asarray(source_matrix, dtype=np.complex128)
    lesser_initial = np.asarray(initial_lesser, dtype=np.complex128)
    if source.ndim != 2 or source.shape[0] != source.shape[1]:
        raise ValueError("source_matrix must be a square matrix.")
    dimension = source.shape[0]
    if lesser_initial.shape != (dimension, dimension):
        raise ValueError("initial_lesser must match source_matrix dimensions.")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(lesser_initial)):
        raise ValueError("source_matrix and initial_lesser must be finite.")
    if max_memory_bytes <= 0:
        raise ValueError("max_memory_bytes must be positive.")
    estimated = 4 * grid.size * grid.size * dimension * dimension * np.dtype(np.complex128).itemsize
    if estimated > max_memory_bytes:
        raise MemoryError(
            "statistics-aware two-time allocation estimate "
            f"{estimated / 1024**2:.1f} MiB exceeds limit "
            f"{max_memory_bytes / 1024**2:.1f} MiB."
        )

    evolution = propagate_unitaries(grid, hamiltonian_at_time)
    theta = np.tril(np.ones((grid.size, grid.size), dtype=float), k=-1)
    theta += 0.5 * np.eye(grid.size)
    retarded = -1j * theta[:, :, None, None] * np.einsum(
        "tia,ab,sjb->tsij",
        evolution,
        source,
        evolution.conj(),
        optimize=True,
    )
    advanced = two_time_adjoint(retarded)
    lesser = np.einsum(
        "tia,ab,sjb->tsij",
        evolution,
        lesser_initial,
        evolution.conj(),
        optimize=True,
    )
    greater = greater_from_keldysh_discontinuity(retarded, advanced, lesser)
    return TwoTimeGreenResult(
        time=grid,
        evolution=evolution,
        retarded=retarded,
        advanced=advanced,
        lesser=lesser,
        greater=greater,
    )


def two_time_meir_wingreen_current(
    time: Any,
    *,
    green_retarded: Any,
    green_lesser: Any,
    lead_self_energy_lesser: Any,
    lead_self_energy_advanced: Any,
    observable: Any | None = None,
    prefactor: float = 2.0,
) -> np.ndarray:
    r"""Return a transient lead current for a charge or spin observable.

    For each time ``t`` this evaluates the finite-memory Meir--Wingreen
    contraction

    ``I_O(t) = prefactor Re integral Tr[O (Gr(t,tau) Sigma<(tau,t)
    + G<(t,tau) Sigma^a(tau,t))] d tau``.

    ``O=I`` gives charge current in the package convention.  Passing a
    Hermitian spin matrix (e.g. ``sigma_z/2`` in a local spin block) returns
    the corresponding spin current.  The function intentionally accepts the
    lead kernel explicitly so multiple reservoirs and spin-polarized leads can
    be compared without hiding a Markov approximation.
    """

    grid = _time_grid(time)
    retarded = _stack(green_retarded, grid, name="green_retarded")
    dim = retarded.shape[2]
    lesser = _stack(green_lesser, grid, name="green_lesser", dim=dim)
    sigma_l = _stack(lead_self_energy_lesser, grid, name="lead_self_energy_lesser", dim=dim)
    sigma_a = _stack(lead_self_energy_advanced, grid, name="lead_self_energy_advanced", dim=dim)
    if observable is None:
        operator = np.eye(dim, dtype=np.complex128)
    else:
        operator = np.asarray(observable, dtype=np.complex128)
        if operator.shape != (dim, dim):
            raise ValueError("observable must match the Green-function matrix dimension.")
        if not np.allclose(operator, operator.conj().T, atol=1e-12, rtol=1e-12):
            raise ValueError("observable must be Hermitian.")
    if not np.isfinite(prefactor):
        raise ValueError("prefactor must be finite.")
    weights = _weights(grid)
    current = np.zeros(grid.size, dtype=float)
    for index in range(grid.size):
        contraction = 0.0j
        for integration_index in range(grid.size):
            contraction += weights[integration_index] * np.trace(
                operator
                @ (
                    retarded[index, integration_index] @ sigma_l[integration_index, index]
                    + lesser[index, integration_index] @ sigma_a[integration_index, index]
                )
            )
        current[index] = float(prefactor * np.real(contraction))
    return current


def two_time_spin_meir_wingreen_current(
    time: Any,
    *,
    green_retarded: Any,
    green_lesser: Any,
    lead_self_energy_lesser: Any,
    lead_self_energy_advanced: Any,
    spin_operator: Any,
) -> np.ndarray:
    """Convenience wrapper for a Hermitian spin observable current."""

    return two_time_meir_wingreen_current(
        time,
        green_retarded=green_retarded,
        green_lesser=green_lesser,
        lead_self_energy_lesser=lead_self_energy_lesser,
        lead_self_energy_advanced=lead_self_energy_advanced,
        observable=spin_operator,
    )


def two_time_meir_wingreen_charge_spin_currents(
    time: Any,
    *,
    green_retarded: Any,
    green_lesser: Any,
    lead_self_energy_lesser: Any,
    lead_self_energy_advanced: Any,
    spin_operators: Mapping[str, Any],
    prefactor: float = 2.0,
) -> dict[str, np.ndarray]:
    r"""Return charge and named spin Meir--Wingreen current channels.

    All channels contract the same two-time kernels and lead self-energy.  The
    charge channel uses the identity; ``spin_operators`` supplies Hermitian
    matrices such as ``sigma_x/2``, ``sigma_y/2``, and ``sigma_z/2`` embedded
    in the device orbital basis.  No commutation or spin-conservation
    assumption is made, so Rashba torque remains a separate balance term.
    """

    if not isinstance(spin_operators, Mapping) or not spin_operators:
        raise ValueError("spin_operators must be a non-empty mapping of names to matrices.")
    result = {
        "charge": two_time_meir_wingreen_current(
            time,
            green_retarded=green_retarded,
            green_lesser=green_lesser,
            lead_self_energy_lesser=lead_self_energy_lesser,
            lead_self_energy_advanced=lead_self_energy_advanced,
            observable=None,
            prefactor=prefactor,
        )
    }
    for name, operator in spin_operators.items():
        if not isinstance(name, str) or not name:
            raise ValueError("spin_operators keys must be non-empty strings.")
        result[name] = two_time_spin_meir_wingreen_current(
            time,
            green_retarded=green_retarded,
            green_lesser=green_lesser,
            lead_self_energy_lesser=lead_self_energy_lesser,
            lead_self_energy_advanced=lead_self_energy_advanced,
            spin_operator=operator,
        )
    return result


def two_time_one_body_correlations(
    time: Any,
    *,
    green_lesser: Any,
    green_greater: Any,
    observables: Mapping[str, Any],
    symmetrized: bool = True,
) -> dict[str, dict[str, np.ndarray]]:
    r"""Return connected Wick bubbles for one-body charge/spin observables.

    For ``O_A=d^dagger A d`` and ``O_B=d^dagger B d`` the connected
    quadratic contraction is

    ``C_AB(t,t') = Tr[A G<(t,t') B G>(t',t)]``.

    ``symmetrized=True`` returns the experimentally relevant anticommutator
    correlator ``(C_AB(t,t') + C_BA(t',t))/2``.  This is an exact Wick
    result for a quadratic state and the bubble part of an interacting KBE
    calculation; vertex corrections are intentionally not silently claimed.
    """

    grid = _time_grid(time)
    lesser = _stack(green_lesser, grid, name="green_lesser")
    greater = _stack(green_greater, grid, name="green_greater", dim=lesser.shape[2])
    if not isinstance(observables, Mapping) or not observables:
        raise ValueError("observables must be a non-empty mapping of Hermitian matrices.")
    parsed: dict[str, np.ndarray] = {}
    for name, value in observables.items():
        if not isinstance(name, str) or not name:
            raise ValueError("observable names must be non-empty strings.")
        operator = np.asarray(value, dtype=np.complex128)
        if operator.shape != (lesser.shape[2], lesser.shape[2]):
            raise ValueError("observable must match the Green-function matrix dimension.")
        if not np.all(np.isfinite(operator)) or not np.allclose(operator, operator.conj().T, atol=1e-12, rtol=1e-12):
            raise ValueError("observables must be finite Hermitian matrices.")
        parsed[name] = operator
    result: dict[str, dict[str, np.ndarray]] = {}
    for left_name, left_operator in parsed.items():
        result[left_name] = {}
        for right_name, right_operator in parsed.items():
            raw = np.empty((grid.size, grid.size), dtype=np.complex128)
            for left in range(grid.size):
                for right in range(grid.size):
                    raw[left, right] = np.trace(
                        left_operator
                        @ lesser[left, right]
                        @ right_operator
                        @ greater[right, left]
                    )
            if symmetrized:
                reverse = np.empty_like(raw)
                for left in range(grid.size):
                    for right in range(grid.size):
                        reverse[left, right] = np.trace(
                            right_operator
                            @ lesser[right, left]
                            @ left_operator
                            @ greater[left, right]
                        )
                raw = 0.5 * (raw + reverse)
            result[left_name][right_name] = raw
    return result


def ladder_vertex_corrected_one_body_correlations(
    time: Any,
    *,
    observables: Mapping[str, Any],
    interaction_kernel: Any,
    channel_names: tuple[str, ...] | list[str],
    target_names: tuple[str, ...] | list[str] | None = None,
    green_lesser: Any | None = None,
    green_greater: Any | None = None,
    bubble: Mapping[str, Mapping[str, Any]] | None = None,
    symmetrized: bool = True,
) -> dict[str, Any]:
    r"""Apply a finite-grid particle--hole ladder (Bethe--Salpeter) vertex.

    The connected Wick bubble is dressed in a declared channel basis,

    ``chi = chi0 + chi0 K chi``.

    Here ``chi0`` is the two-time one-body bubble, ``K`` is a user-supplied
    irreducible channel kernel, and the time convolution is discretised with
    trapezoidal weights.  The routine is deliberately explicit about its
    scope: it is a local ladder/vertex estimate, not an exact interacting
    current-noise theorem.  It is useful for charge/spin fluctuation channels,
    for checking the size of omitted vertices, and for benchmarking a future
    contour current-current Bethe--Salpeter implementation.

    ``channel_names`` selects the rows/columns dressed by the ladder.  The
    returned ``correlations`` contain the corrected pairs in ``target_names``
    (all observables by default), while ``bubble`` retains the undressed Wick
    reference.  A singular or ill-conditioned discrete ladder raises a clear
    error rather than silently returning a no-vertex result.
    """

    grid = _time_grid(time)
    if bubble is None:
        if green_lesser is None or green_greater is None:
            raise ValueError("green_lesser and green_greater are required when bubble is omitted.")
        bubble_value: Mapping[str, Mapping[str, Any]] = two_time_one_body_correlations(
            grid,
            green_lesser=green_lesser,
            green_greater=green_greater,
            observables=observables,
            symmetrized=symmetrized,
        )
    else:
        bubble_value = bubble
    names = tuple(observables)
    if not names or set(names) != set(bubble_value):
        raise ValueError("observables and bubble must contain the same non-empty observable names.")
    channels = tuple(channel_names)
    if not channels or len(set(channels)) != len(channels) or any(name not in names for name in channels):
        raise ValueError("channel_names must be distinct names present in observables.")
    targets = tuple(names if target_names is None else target_names)
    if not targets or any(name not in names for name in targets):
        raise ValueError("target_names must contain observable names.")
    kernel = np.asarray(interaction_kernel, dtype=np.complex128)
    n_channels = len(channels)
    if kernel.ndim == 0:
        kernel = np.eye(n_channels, dtype=np.complex128) * kernel
    if kernel.shape != (n_channels, n_channels) or not np.all(np.isfinite(kernel)):
        raise ValueError("interaction_kernel must be a finite scalar or square channel matrix.")
    weights = np.empty(grid.size, dtype=float)
    weights[0] = 0.5 * (grid[1] - grid[0])
    weights[-1] = 0.5 * (grid[-1] - grid[-2])
    if grid.size > 2:
        weights[1:-1] = 0.5 * (grid[2:] - grid[:-2])

    def _array(left: str, right: str) -> np.ndarray:
        value = np.asarray(bubble_value[left][right], dtype=np.complex128)
        if value.shape != (grid.size, grid.size) or not np.all(np.isfinite(value)):
            raise ValueError(f"bubble[{left!r}][{right!r}] must be a finite (time,time) array.")
        return value

    # Build A = I - chi0 K in the composite (time, channel) basis.  The
    # unknown is Gamma = K chi, so each target column shares this factorisation.
    composite = grid.size * n_channels
    ladder_matrix = np.eye(composite, dtype=np.complex128)
    for left_time in range(grid.size):
        for right_time in range(grid.size):
            block = np.empty((n_channels, n_channels), dtype=np.complex128)
            for left_channel, left_name in enumerate(channels):
                for right_channel, right_name in enumerate(channels):
                    block[left_channel, right_channel] = _array(left_name, right_name)[left_time, right_time]
            row = slice(left_time * n_channels, (left_time + 1) * n_channels)
            column = slice(right_time * n_channels, (right_time + 1) * n_channels)
            ladder_matrix[row, column] -= weights[right_time] * block @ kernel
    condition_number = float(np.linalg.cond(ladder_matrix))
    if not np.isfinite(condition_number) or condition_number > 1.0e12:
        raise np.linalg.LinAlgError(
            f"particle-hole ladder is singular or ill-conditioned (cond={condition_number:.3e})."
        )

    corrected: dict[str, dict[str, np.ndarray]] = {left: {} for left in targets}
    maximum_correction = 0.0
    for right_name in targets:
        source = np.empty((composite, grid.size), dtype=np.complex128)
        for time_index in range(grid.size):
            for channel_index, channel_name in enumerate(channels):
                source[time_index * n_channels + channel_index] = _array(channel_name, right_name)[time_index]
        gamma = np.linalg.solve(ladder_matrix, source).reshape(grid.size, n_channels, grid.size)
        for left_name in targets:
            cross = np.stack([_array(left_name, channel_name) for channel_name in channels], axis=-1)
            value = _array(left_name, right_name) + np.einsum(
                "tuc,cd,udv,u->tv", cross, kernel, gamma, weights, optimize=True
            )
            if symmetrized:
                value = 0.5 * (value + value.T.conj())
            corrected[left_name][right_name] = value
            maximum_correction = max(
                maximum_correction,
                float(np.max(np.abs(value - _array(left_name, right_name)))),
            )
    return {
        "correlations": corrected,
        "bubble": {
            left: {right: _array(left, right).copy() for right in names}
            for left in names
        },
        "diagnostics": {
            "channel_names": list(channels),
            "target_names": list(targets),
            "condition_number": condition_number,
            "maximum_vertex_correction": maximum_correction,
            "kernel_norm": float(np.linalg.norm(kernel)),
            "time_quadrature": "trapezoidal",
            "claim_boundary": "finite-grid local particle-hole ladder; not an exact interacting reservoir current-noise result",
        },
    }


def _double_convolution(left: np.ndarray, middle: np.ndarray, right: np.ndarray, time: np.ndarray) -> np.ndarray:
    return two_time_convolution(two_time_convolution(left, middle, time), right, time)


@dataclass(frozen=True)
class TwoTimeDysonResult:
    """Numerical two-time Keldysh components and convergence diagnostics."""

    time: np.ndarray
    retarded: np.ndarray
    advanced: np.ndarray
    lesser: np.ndarray
    greater: np.ndarray
    iterations: int
    converged: bool
    maximum_update: float
    # Optional converged self-energy branches.  Quadratic Dyson calls leave
    # these as ``None``; self-consistent KBE closures attach the final iterate
    # so continuity/Ward diagnostics can use exactly the kernel that produced
    # the returned Green functions.
    self_energy_retarded: np.ndarray | None = None
    self_energy_advanced: np.ndarray | None = None
    self_energy_lesser: np.ndarray | None = None
    self_energy_greater: np.ndarray | None = None

    @property
    def spectral_identity_error(self) -> float:
        return float(np.max(np.abs(self.greater - self.lesser - self.retarded + self.advanced)))

    @property
    def advanced_adjoint_error(self) -> float:
        return float(np.max(np.abs(self.advanced - two_time_adjoint(self.retarded))))

    @property
    def lesser_adjoint_error(self) -> float:
        return float(np.max(np.abs(self.lesser + two_time_adjoint(self.lesser))))

    @property
    def retarded_causality_error(self) -> float:
        mask = np.triu(np.ones(self.retarded.shape[:2], dtype=bool), k=1)
        return float(np.max(np.abs(self.retarded[mask]))) if np.any(mask) else 0.0

    @property
    def equal_time_spectral_sum_error(self) -> float:
        """Residual of the fermionic equal-time retarded sum rule."""

        diagonal_r = self.retarded[np.arange(self.time.size), np.arange(self.time.size)]
        diagonal_a = self.advanced[np.arange(self.time.size), np.arange(self.time.size)]
        identity = np.eye(self.retarded.shape[2], dtype=np.complex128)
        return float(np.max(np.abs(1j * (diagonal_r - diagonal_a) - identity)))

    @property
    def density_hermiticity_error(self) -> float:
        density = self.density_matrices()
        return float(np.max(np.abs(density - density.swapaxes(-1, -2).conj())))

    @property
    def occupation_bounds_violation(self) -> float:
        density = self.density_matrices()
        eigenvalues = np.linalg.eigvalsh(0.5 * (density + density.swapaxes(-1, -2).conj()))
        return float(max(0.0, -np.min(eigenvalues), np.max(eigenvalues) - 1.0))

    def density_matrices(self) -> np.ndarray:
        """Return the equal-time one-body density ``rho=-i G<``."""

        diagonal = self.lesser[np.arange(self.time.size), np.arange(self.time.size)]
        return -1j * diagonal

    def particle_number_drift(self) -> float:
        numbers = np.trace(self.density_matrices(), axis1=1, axis2=2).real
        return float(np.max(np.abs(numbers - numbers[0])))


def kadanoff_baym_dyson_two_time(
    time: Any,
    *,
    bare_retarded: Any,
    bare_lesser: Any,
    self_energy_retarded: Any,
    self_energy_lesser: Any,
    self_energy_advanced: Any | None = None,
    initial_correlation_lesser: Any | None = None,
    max_iterations: int = 100,
    mixing: float = 0.5,
    tolerance: float = 1e-10,
) -> TwoTimeDysonResult:
    r"""Solve the retarded and lesser two-time Dyson equations by iteration.

    The lesser update is the explicit initial-correlation form

    ``G< = g< + gr*Sr*G< + gr*S<*Ga + g<*Sa*Ga + I_IC``.

    The greater component is reconstructed from the Keldysh discontinuity,
    ``G> = G< + Gr - Ga``.  This enforces the fermionic spectral identity at
    every iteration while retaining the user-supplied initial ``g<`` term. If
    supplied, ``I_IC`` is the explicit two-time vertical initial-correlation
    correction returned by :func:`kbe_lesser_initial_correlation`.
    """

    grid = _time_grid(time)
    bare_r = _stack(bare_retarded, grid, name="bare_retarded")
    dim = bare_r.shape[2]
    bare_l = _stack(bare_lesser, grid, name="bare_lesser", dim=dim)
    sigma_r = _stack(self_energy_retarded, grid, name="self_energy_retarded", dim=dim)
    sigma_l = _stack(self_energy_lesser, grid, name="self_energy_lesser", dim=dim)
    sigma_a = two_time_adjoint(sigma_r) if self_energy_advanced is None else _stack(
        self_energy_advanced, grid, name="self_energy_advanced", dim=dim
    )
    vertical_lesser = np.zeros_like(bare_l)
    if initial_correlation_lesser is not None:
        vertical_lesser = _stack(
            initial_correlation_lesser,
            grid,
            name="initial_correlation_lesser",
            dim=dim,
        )
    if max_iterations < 1 or not (0.0 < mixing <= 1.0) or tolerance <= 0.0:
        raise ValueError("invalid Dyson iteration controls.")

    retarded = bare_r.copy()
    lesser = bare_l.copy()
    maximum_update = float("inf")
    converged = False
    for iteration in range(1, max_iterations + 1):
        new_retarded = bare_r + _double_convolution(bare_r, sigma_r, retarded, grid)
        advanced = two_time_adjoint(new_retarded)
        new_lesser = (
            bare_l
            + _double_convolution(bare_r, sigma_r, lesser, grid)
            + _double_convolution(bare_r, sigma_l, advanced, grid)
            + _double_convolution(bare_l, sigma_a, advanced, grid)
            + vertical_lesser
        )
        mixed_r = mixing * new_retarded + (1.0 - mixing) * retarded
        mixed_l = mixing * new_lesser + (1.0 - mixing) * lesser
        maximum_update = float(max(np.max(np.abs(mixed_r - retarded)), np.max(np.abs(mixed_l - lesser))))
        retarded, lesser = mixed_r, mixed_l
        if maximum_update < tolerance:
            converged = True
            break

    advanced = two_time_adjoint(retarded)
    greater = greater_from_keldysh_discontinuity(retarded, advanced, lesser)
    return TwoTimeDysonResult(
        time=grid.copy(),
        retarded=retarded,
        advanced=advanced,
        lesser=lesser,
        greater=greater,
        iterations=iteration,
        converged=converged,
        maximum_update=maximum_update,
        self_energy_retarded=sigma_r.copy(),
        self_energy_advanced=sigma_a.copy(),
        self_energy_lesser=sigma_l.copy(),
    )


def greater_from_keldysh_discontinuity(retarded: Any, advanced: Any, lesser: Any) -> np.ndarray:
    """Construct ``G>`` from ``G^r-G^a+G<`` with shape validation."""

    r = np.asarray(retarded, dtype=np.complex128)
    a = np.asarray(advanced, dtype=np.complex128)
    l = np.asarray(lesser, dtype=np.complex128)
    if r.shape != a.shape or r.shape != l.shape:
        raise ValueError("retarded, advanced, and lesser shapes must agree.")
    return r - a + l


def electron_boson_scba_self_energy_two_time(
    time: Any,
    lesser_green: Any,
    greater_green: Any,
    *,
    coupling: Any,
    boson_frequency: float,
    boson_temperature: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Build local Einstein-mode Fock self-energies in the time domain.

    The phase factors are the inverse-Fourier images of the energy shifts in
    :func:`electron_boson_scba_symbolic`.  The retarded kernel is assembled from
    ``theta(t-t') (Sigma> - Sigma<)`` and the advanced kernel by adjunction.
    """

    grid = _time_grid(time)
    lesser = _stack(lesser_green, grid, name="lesser_green")
    greater = _stack(greater_green, grid, name="greater_green", dim=lesser.shape[2])
    vertex = np.asarray(coupling, dtype=np.complex128)
    dim = lesser.shape[2]
    if vertex.shape != (dim, dim) or not np.allclose(vertex, vertex.conj().T, atol=1e-12):
        raise ValueError("coupling must be a Hermitian matrix matching the Green-function dimension.")
    frequency = float(boson_frequency)
    temperature = float(boson_temperature)
    if frequency <= 0.0 or not np.isfinite(frequency) or temperature < 0.0 or not np.isfinite(temperature):
        raise ValueError("boson_frequency must be positive and boson_temperature nonnegative.")
    occupation = 0.0 if temperature == 0.0 or frequency / temperature > 500.0 else 1.0 / np.expm1(frequency / temperature)
    lag = grid[:, None] - grid[None, :]
    lesser_factor = occupation * np.exp(-1j * frequency * lag) + (occupation + 1.0) * np.exp(1j * frequency * lag)
    greater_factor = occupation * np.exp(1j * frequency * lag) + (occupation + 1.0) * np.exp(-1j * frequency * lag)
    sigma_l = np.einsum("ab,ijbc,cd,ij->ijad", vertex, lesser, vertex, lesser_factor, optimize=True)
    sigma_g = np.einsum("ab,ijbc,cd,ij->ijad", vertex, greater, vertex, greater_factor, optimize=True)
    causal = np.tril(np.ones((grid.size, grid.size), dtype=float), k=-1) + 0.5 * np.eye(grid.size)
    sigma_r = causal[:, :, None, None] * (sigma_g - sigma_l)
    sigma_a = two_time_adjoint(sigma_r)
    return sigma_r, sigma_a, sigma_l, sigma_g


def _boson_occupation(frequency: float, temperature: float) -> float:
    frequency = float(frequency)
    temperature = float(temperature)
    if frequency <= 0.0 or not np.isfinite(frequency) or temperature < 0.0 or not np.isfinite(temperature):
        raise ValueError("boson_frequency must be positive and boson_temperature nonnegative.")
    if temperature == 0.0 or frequency / temperature > 500.0:
        return 0.0
    return float(1.0 / np.expm1(frequency / temperature))


def _boson_log_occupations(frequency: float, temperature: float) -> tuple[float, float]:
    r"""Return ``(log N, log(N+1))`` for an Einstein mode.

    The vertical branch carries factors ``N exp(omega tau)`` and
    ``(N+1) exp(-omega tau)`` which stay bounded for ``0 <= tau <= beta``,
    but forming ``N`` and the exponential separately overflows once
    ``omega/T`` passes about 709 and returns ``0 * inf = nan``.  Working with
    the logarithms keeps the physically bounded products representable.
    ``T=0`` gives ``log N = -inf``, so the absorption term evaluates to an
    exact zero rather than a NaN.
    """

    frequency = float(frequency)
    temperature = float(temperature)
    if frequency <= 0.0 or not np.isfinite(frequency) or temperature < 0.0 or not np.isfinite(temperature):
        raise ValueError("boson_frequency must be positive and boson_temperature nonnegative.")
    if temperature == 0.0:
        return float("-inf"), 0.0
    scaled = frequency / temperature
    if scaled < 1.0:
        log_expm1 = float(np.log(np.expm1(scaled)))
    else:
        log_expm1 = float(scaled + np.log1p(-np.exp(-scaled)))
    return -log_expm1, scaled - log_expm1


def _validate_bosonic_vertices(
    dimension: int,
    coupling: Any | None,
    cubic_vertex: Any | None,
    quartic_vertex: Any | None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Validate the optional quadratic, cubic, and quartic boson vertices."""

    if coupling is None:
        quadratic = np.zeros((dimension, dimension), dtype=np.complex128)
    else:
        quadratic = np.asarray(coupling, dtype=np.complex128)
        if quadratic.shape != (dimension, dimension) or not np.allclose(quadratic, quadratic.conj().T, atol=1e-12):
            raise ValueError("coupling must be a Hermitian matrix matching the Green-function dimension.")
    cubic = None if cubic_vertex is None else np.asarray(cubic_vertex, dtype=np.complex128)
    if cubic is not None and (cubic.shape != (dimension, dimension, dimension) or not np.all(np.isfinite(cubic))):
        raise ValueError("cubic_vertex must have shape (dim, dim, dim) and contain finite values.")
    quartic = None if quartic_vertex is None else np.asarray(quartic_vertex, dtype=np.complex128)
    if quartic is not None and (quartic.shape != (dimension, dimension, dimension, dimension) or not np.all(np.isfinite(quartic))):
        raise ValueError("quartic_vertex must have shape (dim, dim, dim, dim) and contain finite values.")
    if not np.all(np.isfinite(quadratic)):
        raise ValueError("coupling must contain finite values.")
    if np.allclose(quadratic, 0.0) and cubic is None and quartic is None:
        raise ValueError("at least one bosonic interaction vertex must be supplied.")
    return quadratic, cubic, quartic


def _bosonic_vertex_memory(
    lesser: np.ndarray,
    greater: np.ndarray,
    *,
    cubic: np.ndarray | None,
    quartic: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cubic/quartic lesser and greater contour memory kernels."""

    sigma_l = np.zeros_like(lesser)
    sigma_g = np.zeros_like(greater)
    if cubic is not None:
        sigma_l += -0.5j * np.einsum(
            "u a b,v c d,tkac,tkbd->tkuv".replace(" ", ""),
            cubic,
            cubic.conj(),
            lesser,
            lesser,
            optimize=True,
        )
        sigma_g += -0.5j * np.einsum(
            "u a b,v c d,tkac,tkbd->tkuv".replace(" ", ""),
            cubic,
            cubic.conj(),
            greater,
            greater,
            optimize=True,
        )
    if quartic is not None:
        sigma_l += -1j / 6.0 * np.einsum(
            "u a b c,v d e f,tkad,tkbe,tkcf->tkuv".replace(" ", ""),
            quartic,
            quartic.conj(),
            lesser,
            lesser,
            lesser,
            optimize=True,
        )
        sigma_g += -1j / 6.0 * np.einsum(
            "u a b c,v d e f,tkad,tkbe,tkcf->tkuv".replace(" ", ""),
            quartic,
            quartic.conj(),
            greater,
            greater,
            greater,
            optimize=True,
        )
    return sigma_l, sigma_g


def _bosonic_quartic_hartree(
    lesser: np.ndarray,
    *,
    quartic: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Collocate the instantaneous quartic Hartree potential on the grid."""

    density = -1j * lesser[np.arange(lesser.shape[0]), np.arange(lesser.shape[0])]
    potential = 0.5 * np.einsum("uvab,tab->tuv", quartic, density, optimize=True)
    result = np.zeros_like(lesser)
    result[np.arange(lesser.shape[0]), np.arange(lesser.shape[0])] = potential / weights[:, None, None]
    return result


def bosonic_scba_self_energy_two_time(
    time: Any,
    lesser_green: Any,
    greater_green: Any,
    *,
    coupling: Any | None = None,
    boson_frequency: float = 1.0,
    boson_temperature: float = 0.0,
    cubic_vertex: Any | None = None,
    quartic_vertex: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Build the pure-boson Einstein-mode SCBA on the real-time branches.

    This is the bosonic analogue of the local Fock/SCBA closure already used
    for an electron dressed by an Einstein mode.  A bosonic Green function is
    dressed by a second harmonic mode with vertex ``coupling``; the package's
    convention ``G^> = G^< + G^R - G^A`` is used for both statistics.  The
    function therefore also works for a bosonic block embedded in a larger
    numerical matrix, provided the supplied block has a consistent source.
    """

    grid = _time_grid(time)
    lesser = _stack(lesser_green, grid, name="lesser_green")
    greater = _stack(greater_green, grid, name="greater_green", dim=lesser.shape[2])
    dim = lesser.shape[2]
    vertex, cubic, quartic = _validate_bosonic_vertices(dim, coupling, cubic_vertex, quartic_vertex)
    occupation = _boson_occupation(boson_frequency, boson_temperature)
    lag = grid[:, None] - grid[None, :]
    lesser_factor = occupation * np.exp(-1j * boson_frequency * lag) + (occupation + 1.0) * np.exp(1j * boson_frequency * lag)
    greater_factor = occupation * np.exp(1j * boson_frequency * lag) + (occupation + 1.0) * np.exp(-1j * boson_frequency * lag)
    sigma_l = np.einsum("ab,ijbc,cd,ij->ijad", vertex, lesser, vertex, lesser_factor, optimize=True)
    sigma_g = np.einsum("ab,ijbc,cd,ij->ijad", vertex, greater, vertex, greater_factor, optimize=True)
    vertex_l, vertex_g = _bosonic_vertex_memory(lesser, greater, cubic=cubic, quartic=quartic)
    sigma_l += vertex_l
    sigma_g += vertex_g
    if quartic is not None:
        hartree = _bosonic_quartic_hartree(lesser, quartic=quartic, weights=_weights(grid))
    else:
        hartree = np.zeros_like(lesser)
    causal = np.tril(np.ones((grid.size, grid.size), dtype=float), k=-1) + 0.5 * np.eye(grid.size)
    sigma_r = causal[:, :, None, None] * (sigma_g - sigma_l) + hartree
    return sigma_r, two_time_adjoint(sigma_r), sigma_l, sigma_g


def bosonic_scba_self_energy_contour(
    time: Any,
    imaginary_time: Any,
    *,
    green_lesser: Any,
    green_greater: Any,
    green_rceil: Any,
    green_lceil: Any,
    green_matsubara: Any,
    coupling: Any | None = None,
    boson_frequency: float = 1.0,
    boson_temperature: float = 0.0,
    cubic_vertex: Any | None = None,
    quartic_vertex: Any | None = None,
) -> dict[str, np.ndarray]:
    r"""Build all seven bosonic SCBA self-energy branches.

    Besides ``r/a/< />``, the returned dictionary contains ``rceil``,
    ``lceil`` and ``M``.  The imaginary factor is the periodic free harmonic
    propagator analytically continued to the mixed branch.  This makes the
    closure a genuine contour fixed point rather than a real-time SCBA with
    a post-processed Matsubara kernel.
    """

    real = _time_grid(time)
    imaginary = _time_grid(imaginary_time)
    lesser = _stack(green_lesser, real, name="green_lesser")
    greater = _stack(green_greater, real, name="green_greater", dim=lesser.shape[2])
    dim = lesser.shape[2]
    rceil = np.asarray(green_rceil, dtype=np.complex128)
    lceil = np.asarray(green_lceil, dtype=np.complex128)
    matsubara = np.asarray(green_matsubara, dtype=np.complex128)
    if rceil.shape != (real.size, imaginary.size, dim, dim):
        raise ValueError("green_rceil must have shape (n_time, n_imaginary, dim, dim).")
    if lceil.shape != (imaginary.size, real.size, dim, dim):
        raise ValueError("green_lceil must have shape (n_imaginary, n_time, dim, dim).")
    if matsubara.shape != (imaginary.size, imaginary.size, dim, dim):
        raise ValueError("green_matsubara must have shape (n_imaginary, n_imaginary, dim, dim).")
    if not all(np.all(np.isfinite(value)) for value in (rceil, lceil, matsubara)):
        raise ValueError("mixed and Matsubara Green functions must contain finite values.")
    vertex, cubic, quartic = _validate_bosonic_vertices(dim, coupling, cubic_vertex, quartic_vertex)
    occupation = _boson_occupation(boson_frequency, boson_temperature)
    frequency = float(boson_frequency)
    origin = float(imaginary[0])
    tau = imaginary - origin
    real_lag = real[:, None] - real[None, :]
    lesser_factor = occupation * np.exp(-1j * frequency * real_lag) + (occupation + 1.0) * np.exp(1j * frequency * real_lag)
    greater_factor = occupation * np.exp(1j * frequency * real_lag) + (occupation + 1.0) * np.exp(-1j * frequency * real_lag)
    sigma_l = np.einsum("ab,ijbc,cd,ij->ijad", vertex, lesser, vertex, lesser_factor, optimize=True)
    sigma_g = np.einsum("ab,ijbc,cd,ij->ijad", vertex, greater, vertex, greater_factor, optimize=True)
    vertex_l, vertex_g = _bosonic_vertex_memory(lesser, greater, cubic=cubic, quartic=quartic)
    sigma_l += vertex_l
    sigma_g += vertex_g
    if quartic is not None:
        sigma_r_hartree = _bosonic_quartic_hartree(lesser, quartic=quartic, weights=_weights(real))
    else:
        sigma_r_hartree = np.zeros_like(lesser)
    causal = np.tril(np.ones((real.size, real.size), dtype=float), k=-1) + 0.5 * np.eye(real.size)
    sigma_r = causal[:, :, None, None] * (sigma_g - sigma_l) + sigma_r_hartree
    sigma_a = two_time_adjoint(sigma_r)

    # Evaluate the vertical emission/absorption weights logarithmically; see
    # :func:`_boson_log_occupations` for why the factored form is unsafe.
    log_occupation, log_occupation_plus = _boson_log_occupations(
        boson_frequency, boson_temperature
    )
    real_phase = 1j * frequency * (real[:, None] - real[0])
    mixed_factor = np.exp(
        log_occupation_plus - real_phase - frequency * tau[None, :]
    ) + np.exp(log_occupation + real_phase + frequency * tau[None, :])
    mixed_l_factor = mixed_factor.swapaxes(0, 1).conj()
    sigma_mixed = np.einsum("ab,ikbc,cd,ik->ikad", vertex, rceil, vertex, mixed_factor, optimize=True)
    sigma_lmixed = np.einsum("ab,kjbc,cd,kj->kjad", vertex, lceil, vertex, mixed_l_factor, optimize=True)
    if cubic is not None:
        sigma_mixed += -0.5j * np.einsum(
            "u a b,v c d,ikac,ikbd->ikuv".replace(" ", ""),
            cubic,
            cubic.conj(),
            rceil,
            rceil,
            optimize=True,
        )
        sigma_lmixed += -0.5j * np.einsum(
            "u a b,v c d,kjac,kjbd->kjuv".replace(" ", ""),
            cubic,
            cubic.conj(),
            lceil,
            lceil,
            optimize=True,
        )
    if quartic is not None:
        sigma_mixed += -1j / 6.0 * np.einsum(
            "u a b c,v d e f,ikad,ikbe,ikcf->ikuv".replace(" ", ""),
            quartic,
            quartic.conj(),
            rceil,
            rceil,
            rceil,
            optimize=True,
        )
        sigma_lmixed += -1j / 6.0 * np.einsum(
            "u a b c,v d e f,kjad,kjbe,kjcf->kjuv".replace(" ", ""),
            quartic,
            quartic.conj(),
            lceil,
            lceil,
            lceil,
            optimize=True,
        )

    delta = imaginary[:, None] - imaginary[None, :]
    log_weight = np.where(delta >= 0.0, log_occupation_plus, log_occupation)
    matsubara_factor = -np.exp(log_weight - frequency * delta)
    sigma_matsubara = np.einsum(
        "ab,ijbc,cd,ij->ijad", vertex, matsubara, vertex, matsubara_factor, optimize=True
    )
    if cubic is not None:
        sigma_matsubara += -0.5j * np.einsum(
            "u a b,v c d,ijac,ijbd->ijuv".replace(" ", ""),
            cubic,
            cubic.conj(),
            matsubara,
            matsubara,
            optimize=True,
        )
    if quartic is not None:
        sigma_matsubara += -1j / 6.0 * np.einsum(
            "u a b c,v d e f,ijad,ijbe,ijcf->ijuv".replace(" ", ""),
            quartic,
            quartic.conj(),
            matsubara,
            matsubara,
            matsubara,
            optimize=True,
        )
        density_M = -matsubara[0, 0]
        hartree_M = 0.5 * np.einsum("uvab,ab->uv", quartic, density_M, optimize=True)
        sigma_matsubara[np.arange(imaginary.size), np.arange(imaginary.size)] += hartree_M / _weights(imaginary)[:, None, None]
    return {
        "r": sigma_r,
        "a": sigma_a,
        "<": sigma_l,
        ">": sigma_g,
        "rceil": sigma_mixed,
        "lceil": sigma_lmixed,
        "M": sigma_matsubara,
    }


def hubbard_second_born_self_energy_two_time(
    time: Any,
    *,
    lesser_green: Any,
    greater_green: Any,
    interaction_u: Any,
    spin_pairs: Any = ((0, 1),),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Build local density-density Hubbard second-Born kernels.

    For each pair ``(s, o)`` this uses the correlation-only closure

    ``Sigma_s^<(t,t') = U_s² G_s^<(t,t')G_o^<(t,t')G_o^>(t',t)``

    and the corresponding greater expression.  The retarded component is the
    causal Keldysh discontinuity.  Hartree terms are instantaneous and must be
    added by a separate mean-field layer when desired.
    """

    grid = _time_grid(time)
    lesser = _stack(lesser_green, grid, name="lesser_green")
    greater = _stack(greater_green, grid, name="greater_green", dim=lesser.shape[2])
    dim = lesser.shape[2]
    couplings = np.asarray(interaction_u, dtype=float)
    if couplings.ndim == 0:
        couplings = np.full(dim, float(couplings))
    if couplings.shape != (dim,) or not np.all(np.isfinite(couplings)):
        raise ValueError("interaction_u must be a finite scalar or one value per mode.")
    pairs = tuple((int(pair[0]), int(pair[1])) for pair in spin_pairs)
    if not pairs:
        raise ValueError("spin_pairs must contain at least one pair.")
    sigma_lesser = np.zeros_like(lesser)
    sigma_greater = np.zeros_like(lesser)
    for spin, opposite in pairs:
        if spin < 0 or opposite < 0 or spin >= dim or opposite >= dim or spin == opposite:
            raise ValueError("spin_pairs must contain distinct valid mode indices.")
        opposite_lesser_reverse = lesser[:, :, opposite, opposite].swapaxes(0, 1)
        opposite_greater_reverse = greater[:, :, opposite, opposite].swapaxes(0, 1)
        sigma_lesser[:, :, spin, spin] += (
            couplings[spin] ** 2
            * lesser[:, :, spin, spin]
            * lesser[:, :, opposite, opposite]
            * opposite_greater_reverse
        )
        sigma_greater[:, :, spin, spin] += (
            couplings[spin] ** 2
            * greater[:, :, spin, spin]
            * greater[:, :, opposite, opposite]
            * opposite_lesser_reverse
        )
    causal = np.tril(np.ones((grid.size, grid.size), dtype=float), k=-1) + 0.5 * np.eye(grid.size)
    sigma_retarded = causal[:, :, None, None] * (sigma_greater - sigma_lesser)
    sigma_advanced = two_time_adjoint(sigma_retarded)
    return sigma_retarded, sigma_advanced, sigma_lesser, sigma_greater


def hubbard_hartree_self_energy_two_time(
    time: Any,
    *,
    density: Any,
    interaction_u: Any,
    spin_pairs: Any = ((0, 1),),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r"""Build a finite-grid collocation of the instantaneous Hubbard Hartree term.

    For a time grid with trapezoidal weights ``w_i`` this represents

    ``Sigma_H^r(t_i,t_j) = U n_opposite(t_i) delta_ij / w_i``.

    The diagonal normalization is the discrete delta convention used by
    :func:`two_time_convolution`; it is first-order at the endpoints and
    converges under time refinement.  Hartree has no lesser/greater memory
    kernel, so those components are returned as zero arrays.  Opposite-spin
    pairs should be supplied in both directions for a spinful Hubbard site.
    """

    grid = _time_grid(time)
    values = np.asarray(density, dtype=np.complex128)
    if values.ndim != 3 or values.shape[0] != grid.size or values.shape[1] != values.shape[2]:
        raise ValueError("density must have shape (n_time, dim, dim).")
    if not np.all(np.isfinite(values)):
        raise ValueError("density must contain finite values.")
    if not np.allclose(values, values.swapaxes(-1, -2).conj(), atol=1e-9, rtol=1e-9):
        raise ValueError("density must be Hermitian on every time slice.")
    dim = values.shape[1]
    couplings = np.asarray(interaction_u, dtype=float)
    if couplings.ndim == 0:
        couplings = np.full(dim, float(couplings))
    if couplings.shape != (dim,) or not np.all(np.isfinite(couplings)):
        raise ValueError("interaction_u must be a finite scalar or one value per mode.")
    pairs = tuple((int(pair[0]), int(pair[1])) for pair in spin_pairs)
    if not pairs:
        raise ValueError("spin_pairs must contain at least one pair.")
    weights = _weights(grid)
    sigma_retarded = np.zeros((grid.size, grid.size, dim, dim), dtype=np.complex128)
    for spin, opposite in pairs:
        if spin < 0 or opposite < 0 or spin >= dim or opposite >= dim or spin == opposite:
            raise ValueError("spin_pairs must contain distinct valid mode indices.")
        potential = couplings[spin] * values[:, opposite, opposite].real
        sigma_retarded[np.arange(grid.size), np.arange(grid.size), spin, spin] += potential / weights
    sigma_advanced = two_time_adjoint(sigma_retarded)
    sigma_lesser = np.zeros_like(sigma_retarded)
    sigma_greater = np.zeros_like(sigma_retarded)
    return sigma_retarded, sigma_advanced, sigma_lesser, sigma_greater


def hubbard_second_born_self_energy_mixed(
    time: Any,
    imaginary_time: Any,
    *,
    green_rceil: Any,
    green_lceil: Any,
    interaction_u: Any,
    spin_pairs: Any = ((0, 1),),
) -> np.ndarray:
    r"""Build the local Hubbard second-Born ``Sigma^rceil`` kernel.

    ``green_rceil`` has shape ``(n_time, n_imaginary, dim, dim)`` and
    ``green_lceil`` has shape ``(n_imaginary, n_time, dim, dim)``.  The
    returned diagonal-in-spin kernel is the explicit mixed-branch product
    ``U^2 G^rceil_s G^rceil_o G^lceil_o``.  It is a source kernel for a
    contour solver; this function does not claim self-consistency by itself.
    """

    real_grid = _time_grid(time)
    imaginary_grid = _time_grid(imaginary_time)
    rceil = np.asarray(green_rceil, dtype=np.complex128)
    lceil = np.asarray(green_lceil, dtype=np.complex128)
    if rceil.ndim != 4 or rceil.shape[0:2] != (real_grid.size, imaginary_grid.size) or rceil.shape[2] != rceil.shape[3]:
        raise ValueError("green_rceil must have shape (n_time, n_imaginary, dim, dim).")
    dim = rceil.shape[-1]
    if lceil.shape != (imaginary_grid.size, real_grid.size, dim, dim):
        raise ValueError("green_lceil must have shape (n_imaginary, n_time, dim, dim).")
    if not np.all(np.isfinite(rceil)) or not np.all(np.isfinite(lceil)):
        raise ValueError("mixed Green functions must contain finite values.")
    couplings = np.asarray(interaction_u, dtype=float)
    if couplings.ndim == 0:
        couplings = np.full(dim, float(couplings))
    if couplings.shape != (dim,) or not np.all(np.isfinite(couplings)):
        raise ValueError("interaction_u must be a finite scalar or one value per mode.")
    pairs = tuple((int(pair[0]), int(pair[1])) for pair in spin_pairs)
    if not pairs:
        raise ValueError("spin_pairs must contain at least one pair.")
    sigma = np.zeros((real_grid.size, imaginary_grid.size, dim, dim), dtype=np.complex128)
    for spin, opposite in pairs:
        if spin < 0 or opposite < 0 or spin >= dim or opposite >= dim or spin == opposite:
            raise ValueError("spin_pairs must contain distinct valid mode indices.")
        sigma[:, :, spin, spin] += (
            couplings[spin] ** 2
            * rceil[:, :, spin, spin]
            * rceil[:, :, opposite, opposite]
            * lceil[:, :, opposite, opposite].swapaxes(0, 1)
        )
    return sigma


def _imaginary_convolution(left: np.ndarray, right: np.ndarray, imaginary: np.ndarray) -> np.ndarray:
    """Contract two imaginary-time kernels with the finite-grid measure."""

    return np.einsum("ikab,kjbc,k->ijac", left, right, _weights(imaginary), optimize=True)


def hubbard_second_born_self_energy_matsubara(
    imaginary_time: Any,
    *,
    green_matsubara: Any,
    interaction_u: Any,
    spin_pairs: Any = ((0, 1),),
) -> np.ndarray:
    r"""Build the local Hubbard second-Born Matsubara self-energy.

    With ``G^M(tau,tau')`` carrying the package's fermionic convention, the
    correlation closure is

    ``Sigma^M_s(tau,tau') = -U_s^2 G^M_s(tau,tau')
    G^M_o(tau,tau') G^M_o(tau',tau)``.

    The interaction self-energy is returned separately from the instantaneous
    Hartree layer so a contour solver can expose both contributions and their
    KMS diagnostics.
    """

    imaginary = _time_grid(imaginary_time)
    green = np.asarray(green_matsubara, dtype=np.complex128)
    if green.ndim != 4 or green.shape[0:2] != (imaginary.size, imaginary.size):
        raise ValueError("green_matsubara must have shape (n_imaginary, n_imaginary, dim, dim).")
    if green.shape[2] != green.shape[3] or not np.all(np.isfinite(green)):
        raise ValueError("green_matsubara must contain finite square matrices.")
    dim = green.shape[-1]
    couplings = np.asarray(interaction_u, dtype=float)
    if couplings.ndim == 0:
        couplings = np.full(dim, float(couplings))
    if couplings.shape != (dim,) or not np.all(np.isfinite(couplings)):
        raise ValueError("interaction_u must be a finite scalar or one value per mode.")
    pairs = tuple((int(pair[0]), int(pair[1])) for pair in spin_pairs)
    if not pairs:
        raise ValueError("spin_pairs must contain at least one pair.")
    sigma = np.zeros_like(green)
    reverse = green.swapaxes(0, 1)
    for spin, opposite in pairs:
        if spin < 0 or opposite < 0 or spin >= dim or opposite >= dim or spin == opposite:
            raise ValueError("spin_pairs must contain distinct valid mode indices.")
        sigma[:, :, spin, spin] -= (
            couplings[spin] ** 2
            * green[:, :, spin, spin]
            * green[:, :, opposite, opposite]
            * reverse[:, :, opposite, opposite]
        )
    return sigma


def hubbard_hartree_self_energy_matsubara(
    imaginary_time: Any,
    *,
    green_matsubara: Any,
    interaction_u: Any,
    spin_pairs: Any = ((0, 1),),
) -> np.ndarray:
    r"""Return the instantaneous Hubbard Hartree kernel on the Matsubara grid."""

    imaginary = _time_grid(imaginary_time)
    green = np.asarray(green_matsubara, dtype=np.complex128)
    if green.ndim != 4 or green.shape[0:2] != (imaginary.size, imaginary.size):
        raise ValueError("green_matsubara must have shape (n_imaginary, n_imaginary, dim, dim).")
    if green.shape[2] != green.shape[3] or not np.all(np.isfinite(green)):
        raise ValueError("green_matsubara must contain finite square matrices.")
    dim = green.shape[-1]
    couplings = np.asarray(interaction_u, dtype=float)
    if couplings.ndim == 0:
        couplings = np.full(dim, float(couplings))
    if couplings.shape != (dim,) or not np.all(np.isfinite(couplings)):
        raise ValueError("interaction_u must be a finite scalar or one value per mode.")
    density = np.eye(dim, dtype=np.complex128) + green[0, 0]
    density = 0.5 * (density + density.conj().T)
    weights = _weights(imaginary)
    sigma = np.zeros_like(green)
    pairs = tuple((int(pair[0]), int(pair[1])) for pair in spin_pairs)
    if not pairs:
        raise ValueError("spin_pairs must contain at least one pair.")
    for spin, opposite in pairs:
        if spin < 0 or opposite < 0 or spin >= dim or opposite >= dim or spin == opposite:
            raise ValueError("spin_pairs must contain distinct valid mode indices.")
        sigma[np.arange(imaginary.size), np.arange(imaginary.size), spin, spin] += (
            couplings[spin] * float(density[opposite, opposite].real) / weights
        )
    return sigma


@dataclass(frozen=True)
class MatsubaraHubbardResult:
    """Self-consistent finite-grid Matsubara Hubbard closure and diagnostics."""

    imaginary_time: np.ndarray
    green_matsubara: np.ndarray
    self_energy_matsubara: np.ndarray
    interaction_self_energy: np.ndarray
    hartree_self_energy: np.ndarray
    embedding_self_energy: np.ndarray
    iterations: int
    converged: bool
    maximum_update: float
    green_kms_error: float
    self_energy_kms_error: float


def self_consistent_hubbard_matsubara(
    imaginary_time: Any,
    *,
    bare_green_matsubara: Any,
    interaction_u: Any,
    embedding_self_energy_matsubara: Any | None = None,
    spin_pairs: Any = ((0, 1),),
    include_hartree: bool = True,
    max_iterations: int = 60,
    dyson_iterations: int = 100,
    mixing: float = 0.25,
    tolerance: float = 1e-8,
) -> MatsubaraHubbardResult:
    r"""Solve a self-consistent Matsubara Hubbard Dyson equation.

    ``bare_green_matsubara`` is the supplied reference ``g^M``.  It may
    already contain a quadratic embedding, in which case
    ``embedding_self_energy_matsubara`` should be omitted.  When supplied, the
    latter is added as a fixed contour self-energy.  The nonlinear iteration
    updates the second-Born correlation and Hartree terms, then solves
    ``G^M = g^M + g^M * Sigma^M * G^M`` with the same trapezoidal grid measure
    used by the real-time solver.
    """

    imaginary = _time_grid(imaginary_time)
    bare = np.asarray(bare_green_matsubara, dtype=np.complex128)
    if bare.ndim != 4 or bare.shape[0:2] != (imaginary.size, imaginary.size):
        raise ValueError("bare_green_matsubara must have shape (n_imaginary, n_imaginary, dim, dim).")
    if bare.shape[2] != bare.shape[3] or not np.all(np.isfinite(bare)):
        raise ValueError("bare_green_matsubara must contain finite square matrices.")
    dim = bare.shape[-1]
    if max_iterations < 1 or dyson_iterations < 1 or not (0.0 < mixing <= 1.0) or tolerance <= 0.0:
        raise ValueError("invalid Matsubara iteration controls.")
    if embedding_self_energy_matsubara is None:
        embedding = np.zeros_like(bare)
    else:
        embedding = np.asarray(embedding_self_energy_matsubara, dtype=np.complex128)
        if embedding.shape != bare.shape or not np.all(np.isfinite(embedding)):
            raise ValueError("embedding_self_energy_matsubara must match bare_green_matsubara and be finite.")
    green = bare.copy()
    sigma_corr = np.zeros_like(bare)
    sigma_hartree = np.zeros_like(bare)
    sigma_total = embedding.copy()
    maximum_update = float("inf")
    converged = False
    for iteration in range(1, max_iterations + 1):
        new_corr = hubbard_second_born_self_energy_matsubara(
            imaginary,
            green_matsubara=green,
            interaction_u=interaction_u,
            spin_pairs=spin_pairs,
        )
        new_hartree = (
            hubbard_hartree_self_energy_matsubara(
                imaginary,
                green_matsubara=green,
                interaction_u=interaction_u,
                spin_pairs=spin_pairs,
            )
            if include_hartree
            else np.zeros_like(bare)
        )
        new_total = embedding + new_corr + new_hartree
        sigma_total = mixing * new_total + (1.0 - mixing) * sigma_total
        sigma_corr = mixing * new_corr + (1.0 - mixing) * sigma_corr
        sigma_hartree = mixing * new_hartree + (1.0 - mixing) * sigma_hartree
        dressed = green.copy()
        for _ in range(dyson_iterations):
            updated = bare + _imaginary_convolution(
                _imaginary_convolution(bare, sigma_total, imaginary), dressed, imaginary
            )
            mixed = mixing * updated + (1.0 - mixing) * dressed
            inner_update = float(np.max(np.abs(mixed - dressed)))
            dressed = mixed
            if inner_update < tolerance * 0.1:
                break
        maximum_update = float(max(np.max(np.abs(dressed - green)), np.max(np.abs(sigma_total - (embedding + sigma_corr + sigma_hartree)))))
        green = dressed
        if maximum_update < tolerance:
            converged = True
            break
    interior = slice(1, -1) if imaginary.size > 2 else slice(None)
    green_kms = float(np.max(np.abs(green[-1, interior] + green[0, interior])))
    sigma_kms = float(np.max(np.abs(sigma_total[-1, interior] + sigma_total[0, interior])))
    return MatsubaraHubbardResult(
        imaginary_time=imaginary.copy(),
        green_matsubara=green.copy(),
        self_energy_matsubara=sigma_total.copy(),
        interaction_self_energy=sigma_corr.copy(),
        hartree_self_energy=sigma_hartree.copy(),
        embedding_self_energy=embedding.copy(),
        iterations=iteration,
        converged=converged,
        maximum_update=maximum_update,
        green_kms_error=green_kms,
        self_energy_kms_error=sigma_kms,
    )


@dataclass(frozen=True)
class HubbardSecondBornResult:
    """Self-consistent two-time Hubbard second-Born Green functions."""

    time: np.ndarray
    retarded: np.ndarray
    advanced: np.ndarray
    lesser: np.ndarray
    greater: np.ndarray
    self_energy_retarded: np.ndarray
    self_energy_advanced: np.ndarray
    self_energy_lesser: np.ndarray
    self_energy_greater: np.ndarray
    iterations: int
    converged: bool
    maximum_update: float
    hartree_retarded: np.ndarray | None = None
    hartree_advanced: np.ndarray | None = None
    imaginary_time: np.ndarray | None = None
    green_rceil: np.ndarray | None = None
    green_lceil: np.ndarray | None = None
    green_matsubara: np.ndarray | None = None
    self_energy_mixed: np.ndarray | None = None
    initial_correlation: InitialCorrelationResult | None = None
    lesser_initial_correlation: LesserInitialCorrelationResult | None = None
    lesser_contour_correction: LesserContourCorrectionResult | None = None
    self_energy_matsubara: np.ndarray | None = None
    matsubara_result: MatsubaraHubbardResult | None = None

    @property
    def spectral_identity_error(self) -> float:
        return float(np.max(np.abs(self.greater - self.lesser - self.retarded + self.advanced)))

    @property
    def self_energy_spectral_identity_error(self) -> float:
        return float(np.max(np.abs(self.self_energy_greater - self.self_energy_lesser - self.self_energy_retarded + self.self_energy_advanced)))

    def density_matrices(self) -> np.ndarray:
        diagonal = self.lesser[np.arange(self.time.size), np.arange(self.time.size)]
        density = -1j * diagonal
        return 0.5 * (density + density.swapaxes(-1, -2).conj())


@dataclass(frozen=True)
class BosonicSCBAResult:
    """Full-contour pure-boson SCBA result and branch diagnostics."""

    time: np.ndarray
    imaginary_time: np.ndarray
    retarded: np.ndarray
    advanced: np.ndarray
    lesser: np.ndarray
    greater: np.ndarray
    green_rceil: np.ndarray
    green_lceil: np.ndarray
    green_matsubara: np.ndarray
    self_energy_retarded: np.ndarray
    self_energy_advanced: np.ndarray
    self_energy_lesser: np.ndarray
    self_energy_greater: np.ndarray
    self_energy_mixed: np.ndarray
    self_energy_lmixed: np.ndarray
    self_energy_matsubara: np.ndarray
    iterations: int
    converged: bool
    maximum_update: float
    lesser_contour_correction: LesserContourCorrectionResult | None = None

    @property
    def spectral_identity_error(self) -> float:
        return float(np.max(np.abs(self.greater - self.lesser - self.retarded + self.advanced)))

    @property
    def self_energy_spectral_identity_error(self) -> float:
        return float(np.max(np.abs(self.self_energy_greater - self.self_energy_lesser - self.self_energy_retarded + self.self_energy_advanced)))

    @property
    def green_matsubara_kms_error(self) -> float:
        interior = slice(1, -1) if self.imaginary_time.size > 2 else slice(None)
        return float(np.max(np.abs(self.green_matsubara[-1, interior] - self.green_matsubara[0, interior])))

    @property
    def self_energy_matsubara_kms_error(self) -> float:
        interior = slice(1, -1) if self.imaginary_time.size > 2 else slice(None)
        return float(np.max(np.abs(self.self_energy_matsubara[-1, interior] - self.self_energy_matsubara[0, interior])))

    @property
    def mixed_adjoint_error(self) -> float:
        expected = self.green_rceil.swapaxes(0, 1).conj().swapaxes(-1, -2)
        return float(np.max(np.abs(self.green_lceil - expected)))


def self_consistent_bosonic_scba_contour_two_time(
    time: Any,
    imaginary_time: Any,
    *,
    bare_retarded: Any,
    bare_lesser: Any,
    bare_mixed: Any,
    green_matsubara: Any,
    hamiltonian: Any,
    coupling: Any | None = None,
    boson_frequency: float = 1.0,
    boson_temperature: float = 0.0,
    cubic_vertex: Any | None = None,
    quartic_vertex: Any | None = None,
    bare_lmixed: Any | None = None,
    embedding_self_energy_retarded: Any | None = None,
    embedding_self_energy_lesser: Any | None = None,
    embedding_self_energy_advanced: Any | None = None,
    embedding_self_energy_mixed: Any | None = None,
    embedding_self_energy_lmixed: Any | None = None,
    embedding_self_energy_matsubara: Any | None = None,
    max_iterations: int = 30,
    dyson_iterations: int = 80,
    mixing: float = 0.5,
    tolerance: float = 1e-9,
    matsubara_iterations: int = 60,
    matsubara_dyson_iterations: int = 100,
    matsubara_mixing: float = 0.25,
    matsubara_tolerance: float = 1e-8,
) -> BosonicSCBAResult:
    r"""Solve pure-boson SCBA self-consistently on the complete contour.

    The nonlinear loop updates the real ``r/a/< />`` branches, propagates
    ``G^rceil`` and its adjoint ``G^lceil``, and solves the periodic
    Matsubara Dyson equation at every outer iteration.  Consequently the
    returned ``Sigma^rceil``, ``Sigma^lceil`` and ``Sigma^M`` are the same
    iterate that generated the returned real-time Green functions.  All
    optional embedding branches are treated as fixed contour kernels.
    """

    grid = _time_grid(time)
    imaginary = _time_grid(imaginary_time)
    bare_r = _stack(bare_retarded, grid, name="bare_retarded")
    dim = bare_r.shape[2]
    bare_l = _stack(bare_lesser, grid, name="bare_lesser", dim=dim)
    bare_mixed_array = np.asarray(bare_mixed, dtype=np.complex128)
    green_M = np.asarray(green_matsubara, dtype=np.complex128)
    expected_rm = (grid.size, imaginary.size, dim, dim)
    expected_mr = (imaginary.size, grid.size, dim, dim)
    expected_mm = (imaginary.size, imaginary.size, dim, dim)
    if bare_mixed_array.shape != expected_rm or not np.all(np.isfinite(bare_mixed_array)):
        raise ValueError("bare_mixed must have shape (n_time, n_imaginary, dim, dim) and be finite.")
    if green_M.shape != expected_mm or not np.all(np.isfinite(green_M)):
        raise ValueError("green_matsubara must have shape (n_imaginary, n_imaginary, dim, dim) and be finite.")
    lmixed = (
        bare_mixed_array.swapaxes(0, 1).conj().swapaxes(-1, -2)
        if bare_lmixed is None
        else np.asarray(bare_lmixed, dtype=np.complex128)
    )
    if lmixed.shape != expected_mr or not np.all(np.isfinite(lmixed)):
        raise ValueError("bare_lmixed must have shape (n_imaginary, n_time, dim, dim) and be finite.")
    vertex, cubic, quartic = _validate_bosonic_vertices(dim, coupling, cubic_vertex, quartic_vertex)
    if max_iterations < 1 or dyson_iterations < 1 or matsubara_iterations < 1 or matsubara_dyson_iterations < 1:
        raise ValueError("iteration limits must be positive.")
    if not (0.0 < mixing <= 1.0) or not (0.0 < matsubara_mixing <= 1.0) or tolerance <= 0.0 or matsubara_tolerance <= 0.0:
        raise ValueError("invalid contour iteration controls.")
    h = np.asarray(hamiltonian, dtype=np.complex128)
    if h.ndim == 2 and h.shape == (dim, dim):
        h_stepper = h
    elif h.ndim == 3 and h.shape == (grid.size, dim, dim):
        h_stepper = h
    else:
        raise ValueError("hamiltonian must have shape (dim, dim) or (n_time, dim, dim).")
    if not np.all(np.isfinite(h_stepper)):
        raise ValueError("hamiltonian must contain only finite values.")

    def branch(value: Any | None, shape: tuple[int, ...], name: str) -> np.ndarray:
        if value is None:
            return np.zeros(shape, dtype=np.complex128)
        result = np.asarray(value, dtype=np.complex128)
        if result.shape != shape or not np.all(np.isfinite(result)):
            raise ValueError(f"{name} must match shape {shape} and contain finite values.")
        return result

    embed_r = branch(embedding_self_energy_retarded, bare_r.shape, "embedding_self_energy_retarded")
    embed_l = branch(embedding_self_energy_lesser, bare_l.shape, "embedding_self_energy_lesser")
    embed_a = two_time_adjoint(embed_r) if embedding_self_energy_advanced is None else branch(embedding_self_energy_advanced, bare_r.shape, "embedding_self_energy_advanced")
    embed_mixed = branch(embedding_self_energy_mixed, expected_rm, "embedding_self_energy_mixed")
    embed_lmixed = branch(embedding_self_energy_lmixed, expected_mr, "embedding_self_energy_lmixed")
    embed_M = branch(embedding_self_energy_matsubara, expected_mm, "embedding_self_energy_matsubara")

    bare_a = two_time_adjoint(bare_r)
    bare_g = greater_from_keldysh_discontinuity(bare_r, bare_a, bare_l)
    green = TwoTimeDysonResult(grid.copy(), bare_r, bare_a, bare_l, bare_g, 0, True, 0.0)
    mixed = bare_mixed_array.copy()
    sigma_r = np.zeros_like(bare_r)
    sigma_a = np.zeros_like(bare_r)
    sigma_l = np.zeros_like(bare_l)
    sigma_g = np.zeros_like(bare_l)
    sigma_mixed = np.zeros(expected_rm, dtype=np.complex128)
    sigma_lmixed = np.zeros(expected_mr, dtype=np.complex128)
    sigma_M = np.zeros(expected_mm, dtype=np.complex128)
    maximum_update = float("inf")
    converged = False
    correction = None

    for iteration in range(1, max_iterations + 1):
        new = bosonic_scba_self_energy_contour(
            grid,
            imaginary,
            green_lesser=green.lesser,
            green_greater=green.greater,
            green_rceil=mixed,
            green_lceil=lmixed,
            green_matsubara=green_M,
            coupling=vertex,
            boson_frequency=boson_frequency,
            boson_temperature=boson_temperature,
            cubic_vertex=cubic,
            quartic_vertex=quartic,
        )
        sigma_r = mixing * new["r"] + (1.0 - mixing) * sigma_r
        sigma_a = mixing * new["a"] + (1.0 - mixing) * sigma_a
        sigma_l = mixing * new["<"] + (1.0 - mixing) * sigma_l
        sigma_g = mixing * new[">"] + (1.0 - mixing) * sigma_g
        sigma_mixed = mixing * new["rceil"] + (1.0 - mixing) * sigma_mixed
        sigma_lmixed = mixing * new["lceil"] + (1.0 - mixing) * sigma_lmixed
        sigma_M = mixing * new["M"] + (1.0 - mixing) * sigma_M

        # Close the periodic branch with its own Dyson fixed point before it
        # enters the mixed KBE source.  This is the vertical analogue of the
        # real-time Dyson iteration and is intentionally kept explicit.
        dressed_M = green_M.copy()
        vertical_interaction = sigma_M.copy()
        for _ in range(matsubara_iterations):
            vertical_update = bosonic_scba_self_energy_contour(
                grid,
                imaginary,
                green_lesser=green.lesser,
                green_greater=green.greater,
                green_rceil=mixed,
                green_lceil=lmixed,
                green_matsubara=dressed_M,
                coupling=vertex,
                boson_frequency=boson_frequency,
                boson_temperature=boson_temperature,
                cubic_vertex=cubic,
                quartic_vertex=quartic,
            )["M"]
            vertical_interaction = matsubara_mixing * vertical_update + (1.0 - matsubara_mixing) * vertical_interaction
            vertical_sigma = embed_M + vertical_interaction
            dyson_state = dressed_M.copy()
            for _ in range(matsubara_dyson_iterations):
                updated_M = np.asarray(green_matsubara, dtype=np.complex128) + _imaginary_convolution(
                    _imaginary_convolution(np.asarray(green_matsubara, dtype=np.complex128), vertical_sigma, imaginary),
                    dyson_state,
                    imaginary,
                )
                mixed_M = matsubara_mixing * updated_M + (1.0 - matsubara_mixing) * dyson_state
                if float(np.max(np.abs(mixed_M - dyson_state))) < matsubara_tolerance:
                    dyson_state = mixed_M
                    break
                dyson_state = mixed_M
            branch_update = float(max(np.max(np.abs(dyson_state - dressed_M)), np.max(np.abs(vertical_interaction - sigma_M))))
            dressed_M = dyson_state
            if branch_update < matsubara_tolerance:
                break
        sigma_M = vertical_interaction
        vertical_sigma = embed_M + sigma_M
        vertical_update = float(np.max(np.abs(dressed_M - green_M)))
        green_M = dressed_M

        correction = kbe_lesser_contour_correction(
            grid,
            imaginary,
            bare_retarded=bare_r,
            bare_mixed=bare_mixed_array,
            self_energy_mixed=embed_mixed + sigma_mixed,
            green_lmixed=lmixed,
            green_advanced=green.advanced,
            self_energy_matsubara=vertical_sigma,
            self_energy_lmixed=embed_lmixed + sigma_lmixed,
        )
        updated = kadanoff_baym_dyson_two_time(
            grid,
            bare_retarded=bare_r,
            bare_lesser=bare_l,
            self_energy_retarded=embed_r + sigma_r,
            self_energy_lesser=embed_l + sigma_l,
            self_energy_advanced=embed_a + sigma_a,
            initial_correlation_lesser=correction.correction,
            max_iterations=dyson_iterations,
            mixing=mixing,
            tolerance=tolerance * 0.1,
        )
        new_mixed = propagate_mixed_kbe_rceil(
            grid,
            imaginary,
            initial_green_mixed=bare_mixed_array[0],
            self_energy_retarded=embed_r + sigma_r,
            self_energy_mixed=embed_mixed + sigma_mixed,
            green_matsubara=green_M,
            hamiltonian=h_stepper,
        )
        mixed_update = mixing * new_mixed + (1.0 - mixing) * mixed
        maximum_update = float(
            max(
                np.max(np.abs(updated.retarded - green.retarded)),
                np.max(np.abs(updated.lesser - green.lesser)),
                np.max(np.abs(mixed_update - mixed)),
                vertical_update,
            )
        )
        green = updated
        mixed = mixed_update
        lmixed = mixed.swapaxes(0, 1).conj().swapaxes(-1, -2)
        if maximum_update < tolerance:
            converged = True
            break

    return BosonicSCBAResult(
        time=grid.copy(),
        imaginary_time=imaginary.copy(),
        retarded=green.retarded.copy(),
        advanced=green.advanced.copy(),
        lesser=green.lesser.copy(),
        greater=green.greater.copy(),
        green_rceil=mixed.copy(),
        green_lceil=lmixed.copy(),
        green_matsubara=green_M.copy(),
        self_energy_retarded=sigma_r.copy(),
        self_energy_advanced=sigma_a.copy(),
        self_energy_lesser=sigma_l.copy(),
        self_energy_greater=sigma_g.copy(),
        self_energy_mixed=sigma_mixed.copy(),
        self_energy_lmixed=sigma_lmixed.copy(),
        self_energy_matsubara=(embed_M + sigma_M).copy(),
        iterations=iteration,
        converged=converged,
        maximum_update=maximum_update,
        lesser_contour_correction=correction,
    )


def self_consistent_hubbard_second_born_two_time(
    time: Any,
    *,
    bare_retarded: Any,
    bare_lesser: Any,
    interaction_u: Any,
    spin_pairs: Any = ((0, 1),),
    max_iterations: int = 30,
    dyson_iterations: int = 80,
    mixing: float = 0.5,
    tolerance: float = 1e-9,
    include_hartree: bool = False,
) -> HubbardSecondBornResult:
    """Solve a finite-grid self-consistent Hubbard second-Born KBE closure.

    ``include_hartree=True`` adds the instantaneous density-density Hartree
    potential with the finite-grid delta collocation implemented by
    :func:`hubbard_hartree_self_energy_two_time`.  The default remains the
    correlation-only Gate32 layer for backwards-compatible comparisons.
    """

    grid = _time_grid(time)
    bare_r = _stack(bare_retarded, grid, name="bare_retarded")
    bare_l = _stack(bare_lesser, grid, name="bare_lesser", dim=bare_r.shape[2])
    if max_iterations < 1 or dyson_iterations < 1 or not (0.0 < mixing <= 1.0) or tolerance <= 0.0:
        raise ValueError("invalid Hubbard second-Born iteration controls.")
    bare_a = two_time_adjoint(bare_r)
    bare_g = greater_from_keldysh_discontinuity(bare_r, bare_a, bare_l)
    green = TwoTimeDysonResult(grid.copy(), bare_r, bare_a, bare_l, bare_g, 0, True, 0.0)
    sigma_r = np.zeros_like(bare_r)
    sigma_a = np.zeros_like(bare_r)
    sigma_l = np.zeros_like(bare_r)
    sigma_g = np.zeros_like(bare_r)
    hartree_r = np.zeros_like(bare_r)
    hartree_a = np.zeros_like(bare_r)
    maximum_update = float("inf")
    converged = False
    for iteration in range(1, max_iterations + 1):
        new_sigma_r, new_sigma_a, new_sigma_l, new_sigma_g = hubbard_second_born_self_energy_two_time(
            grid,
            lesser_green=green.lesser,
            greater_green=green.greater,
            interaction_u=interaction_u,
            spin_pairs=spin_pairs,
        )
        if include_hartree:
            density_for_hartree = green.density_matrices()
            density_for_hartree = 0.5 * (
                density_for_hartree + density_for_hartree.swapaxes(-1, -2).conj()
            )
            hartree_r, hartree_a, _, _ = hubbard_hartree_self_energy_two_time(
                grid,
                density=density_for_hartree,
                interaction_u=interaction_u,
                spin_pairs=spin_pairs,
            )
            new_sigma_r = new_sigma_r + hartree_r
            new_sigma_a = new_sigma_a + hartree_a
        sigma_r = mixing * new_sigma_r + (1.0 - mixing) * sigma_r
        sigma_a = mixing * new_sigma_a + (1.0 - mixing) * sigma_a
        sigma_l = mixing * new_sigma_l + (1.0 - mixing) * sigma_l
        sigma_g = mixing * new_sigma_g + (1.0 - mixing) * sigma_g
        updated = kadanoff_baym_dyson_two_time(
            grid,
            bare_retarded=bare_r,
            bare_lesser=bare_l,
            self_energy_retarded=sigma_r,
            self_energy_lesser=sigma_l,
            self_energy_advanced=sigma_a,
            max_iterations=dyson_iterations,
            mixing=mixing,
            tolerance=tolerance * 0.1,
        )
        maximum_update = float(
            max(
                np.max(np.abs(updated.retarded - green.retarded)),
                np.max(np.abs(updated.lesser - green.lesser)),
            )
        )
        green = updated
        if maximum_update < tolerance:
            converged = True
            break
    return HubbardSecondBornResult(
        time=grid.copy(),
        retarded=green.retarded.copy(),
        advanced=green.advanced.copy(),
        lesser=green.lesser.copy(),
        greater=green.greater.copy(),
        self_energy_retarded=sigma_r.copy(),
        self_energy_advanced=sigma_a.copy(),
        self_energy_lesser=sigma_l.copy(),
        self_energy_greater=sigma_g.copy(),
        iterations=iteration,
        converged=converged,
        maximum_update=maximum_update,
        hartree_retarded=hartree_r.copy() if include_hartree else None,
        hartree_advanced=hartree_a.copy() if include_hartree else None,
    )


def self_consistent_hubbard_second_born_contour_two_time(
    time: Any,
    imaginary_time: Any,
    *,
    bare_retarded: Any,
    bare_lesser: Any,
    bare_mixed: Any,
    green_matsubara: Any,
    hamiltonian: Any,
    interaction_u: Any,
    embedding_self_energy_retarded: Any | None = None,
    embedding_self_energy_lesser: Any | None = None,
    embedding_self_energy_advanced: Any | None = None,
    embedding_self_energy_mixed: Any | None = None,
    bare_lmixed: Any | None = None,
    spin_pairs: Any = ((0, 1),),
    max_iterations: int = 30,
    dyson_iterations: int = 80,
    mixing: float = 0.5,
    tolerance: float = 1e-9,
    include_hartree: bool = False,
    include_vertical_lesser: bool = False,
    include_full_contour_lesser: bool = False,
    self_energy_matsubara: Any | None = None,
    self_consistent_matsubara: bool = False,
    matsubara_iterations: int = 60,
    matsubara_dyson_iterations: int = 100,
    matsubara_mixing: float = 0.25,
    matsubara_tolerance: float = 1e-8,
) -> HubbardSecondBornResult:
    r"""Iterate real and mixed Hubbard second-Born branches together.

    The real-time Dyson update and the mixed Volterra update share the same
    interaction self-energy iterate.  ``embedding_self_energy_*`` carries a
    contacted reference's lead memory into the real and mixed equations.  The
    real-time embedding is optional for backwards compatibility: when omitted,
    ``bare_retarded``/``bare_lesser`` are interpreted as already dressed by the
    leads; when supplied, they are the isolated-device Green functions and the
    embedding is added to the KBE update.  The returned object exposes both branches and
    convergence metadata.  The real lesser equation retains its supplied
    ``bare_lesser`` initial term by default.  With ``include_vertical_lesser``
    enabled, the finite-grid term ``G^R * (-i Sigma^rceil * G^lceil)`` and its
    right-acting adjoint are added to the real lesser update.  This explicit
    research closure is returned in ``lesser_initial_correlation``; it remains
    distinct from a production contour solver with a fully converged initial
    state.  ``include_full_contour_lesser`` selects the complete three-term
    Langreth reconstruction of ``(g Sigma G)^<``.  In that mode the optional
    ``self_energy_matsubara`` is the total vertical Matsubara self-energy;
    omitting it uses a zero Matsubara interaction as an explicit research
    approximation and is recorded by the returned correction object.  If
    ``self_consistent_matsubara=True``, ``green_matsubara`` is interpreted as
    the supplied bare/reference ``g^M`` and the optional
    ``self_energy_matsubara`` is a fixed embedding contribution.  The local
    Hubbard Hartree plus second-Born Matsubara self-energy is then solved
    self-consistently before the real/mixed iteration.  The resulting
    ``MatsubaraHubbardResult`` and its KMS diagnostics are attached to the
    return value; the default branch is unchanged.
    """
    grid = _time_grid(time)
    imaginary = _time_grid(imaginary_time)
    bare_r = _stack(bare_retarded, grid, name="bare_retarded")
    dim = bare_r.shape[2]
    bare_l = _stack(bare_lesser, grid, name="bare_lesser", dim=dim)
    bare_m = np.asarray(bare_mixed, dtype=np.complex128)
    green_M = np.asarray(green_matsubara, dtype=np.complex128)
    if bare_m.shape != (grid.size, imaginary.size, dim, dim):
        raise ValueError("bare_mixed must have shape (n_time, n_imaginary, dim, dim).")
    if green_M.shape != (imaginary.size, imaginary.size, dim, dim):
        raise ValueError("green_matsubara must have shape (n_imaginary, n_imaginary, dim, dim).")
    embed_r = np.zeros_like(bare_r) if embedding_self_energy_retarded is None else _stack(
        embedding_self_energy_retarded, grid, name="embedding_self_energy_retarded", dim=dim
    )
    embed_l = np.zeros_like(bare_l) if embedding_self_energy_lesser is None else _stack(
        embedding_self_energy_lesser, grid, name="embedding_self_energy_lesser", dim=dim
    )
    embed_a = two_time_adjoint(embed_r) if embedding_self_energy_advanced is None else _stack(
        embedding_self_energy_advanced, grid, name="embedding_self_energy_advanced", dim=dim
    )
    embed_m = np.zeros_like(bare_m) if embedding_self_energy_mixed is None else np.asarray(
        embedding_self_energy_mixed, dtype=np.complex128
    )
    if embed_m.shape != bare_m.shape:
        raise ValueError("embedding_self_energy_mixed must match bare_mixed shape.")
    if not all(np.all(np.isfinite(value)) for value in (embed_r, embed_a, embed_l, embed_m)):
        raise ValueError("embedding self-energy branches must contain only finite values.")
    lmixed = (
        bare_m.swapaxes(0, 1).conj().swapaxes(-1, -2)
        if bare_lmixed is None
        else np.asarray(bare_lmixed, dtype=np.complex128)
    )
    if lmixed.shape != (imaginary.size, grid.size, dim, dim):
        raise ValueError("bare_lmixed must have shape (n_imaginary, n_time, dim, dim).")
    h = np.asarray(hamiltonian, dtype=np.complex128)
    if h.ndim == 2 and h.shape == (dim, dim):
        h_for_stepper = h
    elif h.ndim == 3 and h.shape == (grid.size, dim, dim):
        h_for_stepper = h
    else:
        raise ValueError("hamiltonian must have shape (dim, dim) or (n_time, dim, dim).")
    if max_iterations < 1 or dyson_iterations < 1 or not (0.0 < mixing <= 1.0) or tolerance <= 0.0:
        raise ValueError("invalid contour iteration controls.")
    if include_vertical_lesser and include_full_contour_lesser:
        raise ValueError("choose either include_vertical_lesser or include_full_contour_lesser, not both.")
    supplied_sigma_M = None if self_energy_matsubara is None else np.asarray(self_energy_matsubara, dtype=np.complex128)
    if supplied_sigma_M is not None:
        vertical_sigma_M = supplied_sigma_M
        if vertical_sigma_M.shape != (imaginary.size, imaginary.size, dim, dim):
            raise ValueError("self_energy_matsubara must have shape (n_imaginary, n_imaginary, dim, dim).")
        if not np.all(np.isfinite(vertical_sigma_M)):
            raise ValueError("self_energy_matsubara must contain only finite values.")
    else:
        vertical_sigma_M = np.zeros((imaginary.size, imaginary.size, dim, dim), dtype=np.complex128)
    matsubara_result = None
    if self_consistent_matsubara:
        matsubara_result = self_consistent_hubbard_matsubara(
            imaginary,
            bare_green_matsubara=green_M,
            interaction_u=interaction_u,
            embedding_self_energy_matsubara=supplied_sigma_M,
            spin_pairs=spin_pairs,
            include_hartree=include_hartree,
            max_iterations=matsubara_iterations,
            dyson_iterations=matsubara_dyson_iterations,
            mixing=matsubara_mixing,
            tolerance=matsubara_tolerance,
        )
        green_M = matsubara_result.green_matsubara.copy()
        vertical_sigma_M = matsubara_result.self_energy_matsubara.copy()
    bare_a = two_time_adjoint(bare_r)
    bare_g = greater_from_keldysh_discontinuity(bare_r, bare_a, bare_l)
    green = TwoTimeDysonResult(grid.copy(), bare_r, bare_a, bare_l, bare_g, 0, True, 0.0)
    sigma_r = np.zeros_like(bare_r)
    sigma_a = np.zeros_like(bare_r)
    sigma_l = np.zeros_like(bare_r)
    sigma_g = np.zeros_like(bare_r)
    sigma_m = np.zeros_like(bare_m)
    mixed = bare_m.copy()
    hartree_r = np.zeros_like(bare_r)
    hartree_a = np.zeros_like(bare_r)
    maximum_update = float("inf")
    converged = False
    lesser_initial_correlation = None
    lesser_contour_correction = None
    for iteration in range(1, max_iterations + 1):
        new_sigma_r, new_sigma_a, new_sigma_l, new_sigma_g = hubbard_second_born_self_energy_two_time(
            grid,
            lesser_green=green.lesser,
            greater_green=green.greater,
            interaction_u=interaction_u,
            spin_pairs=spin_pairs,
        )
        if include_hartree:
            density_for_hartree = green.density_matrices()
            density_for_hartree = 0.5 * (density_for_hartree + density_for_hartree.swapaxes(-1, -2).conj())
            hartree_r, hartree_a, _, _ = hubbard_hartree_self_energy_two_time(
                grid,
                density=density_for_hartree,
                interaction_u=interaction_u,
                spin_pairs=spin_pairs,
            )
            new_sigma_r = new_sigma_r + hartree_r
            new_sigma_a = new_sigma_a + hartree_a
        new_sigma_m = hubbard_second_born_self_energy_mixed(
            grid,
            imaginary,
            green_rceil=mixed,
            green_lceil=lmixed,
            interaction_u=interaction_u,
            spin_pairs=spin_pairs,
        )
        sigma_r = mixing * new_sigma_r + (1.0 - mixing) * sigma_r
        sigma_a = mixing * new_sigma_a + (1.0 - mixing) * sigma_a
        sigma_l = mixing * new_sigma_l + (1.0 - mixing) * sigma_l
        sigma_g = mixing * new_sigma_g + (1.0 - mixing) * sigma_g
        sigma_m = mixing * new_sigma_m + (1.0 - mixing) * sigma_m
        updated = kadanoff_baym_dyson_two_time(
            grid,
            bare_retarded=bare_r,
            bare_lesser=bare_l,
            self_energy_retarded=embed_r + sigma_r,
            self_energy_lesser=embed_l + sigma_l,
            self_energy_advanced=embed_a + sigma_a,
            max_iterations=dyson_iterations,
            mixing=mixing,
            tolerance=tolerance * 0.1,
        )
        if include_full_contour_lesser:
            lesser_contour_correction = kbe_lesser_contour_correction(
                grid,
                imaginary,
                bare_retarded=bare_r,
                bare_mixed=bare_m,
                self_energy_mixed=embed_m + sigma_m,
                green_lmixed=lmixed,
                green_advanced=updated.advanced,
                self_energy_matsubara=vertical_sigma_M,
            )
            updated = kadanoff_baym_dyson_two_time(
                grid,
                bare_retarded=bare_r,
                bare_lesser=bare_l,
                self_energy_retarded=embed_r + sigma_r,
                self_energy_lesser=embed_l + sigma_l,
                self_energy_advanced=embed_a + sigma_a,
                initial_correlation_lesser=lesser_contour_correction.correction,
                max_iterations=dyson_iterations,
                mixing=mixing,
                tolerance=tolerance * 0.1,
            )
        elif include_vertical_lesser:
            lesser_initial_correlation = kbe_lesser_initial_correlation(
                grid,
                imaginary,
                green_retarded=updated.retarded,
                self_energy_mixed=embed_m + sigma_m,
                green_lmixed=lmixed,
            )
            updated = kadanoff_baym_dyson_two_time(
                grid,
                bare_retarded=bare_r,
                bare_lesser=bare_l,
                self_energy_retarded=embed_r + sigma_r,
                self_energy_lesser=embed_l + sigma_l,
                self_energy_advanced=embed_a + sigma_a,
                initial_correlation_lesser=lesser_initial_correlation.correction,
                max_iterations=dyson_iterations,
                mixing=mixing,
                tolerance=tolerance * 0.1,
            )
        new_mixed = propagate_mixed_kbe_rceil(
            grid,
            imaginary,
            initial_green_mixed=bare_m[0],
            self_energy_retarded=embed_r + sigma_r,
            self_energy_mixed=embed_m + sigma_m,
            green_matsubara=green_M,
            hamiltonian=h_for_stepper,
        )
        mixed_update = mixing * new_mixed + (1.0 - mixing) * mixed
        maximum_update = float(
            max(
                np.max(np.abs(updated.retarded - green.retarded)),
                np.max(np.abs(updated.lesser - green.lesser)),
                np.max(np.abs(mixed_update - mixed)),
            )
        )
        green = updated
        mixed = mixed_update
        lmixed = mixed.swapaxes(0, 1).conj().swapaxes(-1, -2)
        if maximum_update < tolerance:
            converged = True
            break
    return HubbardSecondBornResult(
        time=grid.copy(),
        retarded=green.retarded.copy(),
        advanced=green.advanced.copy(),
        lesser=green.lesser.copy(),
        greater=green.greater.copy(),
        self_energy_retarded=sigma_r.copy(),
        self_energy_advanced=sigma_a.copy(),
        self_energy_lesser=sigma_l.copy(),
        self_energy_greater=sigma_g.copy(),
        iterations=iteration,
        converged=converged,
        maximum_update=maximum_update,
        hartree_retarded=hartree_r.copy() if include_hartree else None,
        hartree_advanced=hartree_a.copy() if include_hartree else None,
        imaginary_time=imaginary.copy(),
        green_rceil=mixed.copy(),
        green_lceil=lmixed.copy(),
        green_matsubara=green_M.copy(),
        self_energy_mixed=sigma_m.copy(),
        initial_correlation=kbe_initial_correlation_kernel(
            grid,
            imaginary,
            self_energy_mixed=sigma_m,
            green_mixed=lmixed,
        ),
        lesser_initial_correlation=lesser_initial_correlation,
        lesser_contour_correction=lesser_contour_correction,
        self_energy_matsubara=vertical_sigma_M.copy(),
        matsubara_result=matsubara_result,
    )


def self_consistent_born_two_time(
    time: Any,
    *,
    bare_retarded: Any,
    bare_lesser: Any,
    coupling: Any,
    boson_frequency: float,
    boson_temperature: float = 0.0,
    max_iterations: int = 30,
    dyson_iterations: int = 80,
    mixing: float = 0.5,
    tolerance: float = 1e-9,
) -> TwoTimeDysonResult:
    """Solve a finite-grid self-consistent electron--boson KBE problem.

    The noninteracting two-time kernels carry the initial state and any lead
    approximation.  The interaction is then iterated through the conserving
    Fock/SCBA closure.  The return value contains the final Green functions;
    iteration metadata is sufficient for a gate to reject unconverged runs.
    """

    grid = _time_grid(time)
    bare_r = _stack(bare_retarded, grid, name="bare_retarded")
    bare_l = _stack(bare_lesser, grid, name="bare_lesser", dim=bare_r.shape[2])
    bare_a = two_time_adjoint(bare_r)
    bare_g = greater_from_keldysh_discontinuity(bare_r, bare_a, bare_l)
    sigma_r = np.zeros_like(bare_r)
    sigma_a = np.zeros_like(bare_r)
    sigma_l = np.zeros_like(bare_r)
    sigma_g = np.zeros_like(bare_r)
    green = TwoTimeDysonResult(grid.copy(), bare_r, bare_a, bare_l, bare_g, 0, True, 0.0)
    maximum_update = float("inf")
    converged = False
    for iteration in range(1, max_iterations + 1):
        new_sigma_r, new_sigma_a, new_sigma_l, new_sigma_g = electron_boson_scba_self_energy_two_time(
            grid,
            green.lesser,
            green.greater,
            coupling=coupling,
            boson_frequency=boson_frequency,
            boson_temperature=boson_temperature,
        )
        sigma_r = mixing * new_sigma_r + (1.0 - mixing) * sigma_r
        sigma_a = mixing * new_sigma_a + (1.0 - mixing) * sigma_a
        sigma_l = mixing * new_sigma_l + (1.0 - mixing) * sigma_l
        sigma_g = mixing * new_sigma_g + (1.0 - mixing) * sigma_g
        updated = kadanoff_baym_dyson_two_time(
            grid,
            bare_retarded=bare_r,
            bare_lesser=bare_l,
            self_energy_retarded=sigma_r,
            self_energy_lesser=sigma_l,
            self_energy_advanced=sigma_a,
            max_iterations=dyson_iterations,
            mixing=mixing,
            tolerance=tolerance * 0.1,
        )
        maximum_update = float(
            max(
                np.max(np.abs(updated.retarded - green.retarded)),
                np.max(np.abs(updated.lesser - green.lesser)),
            )
        )
        green = updated
        if maximum_update < tolerance:
            converged = True
            break
    return TwoTimeDysonResult(
        time=grid.copy(),
        retarded=green.retarded.copy(),
        advanced=green.advanced.copy(),
        lesser=green.lesser.copy(),
        greater=green.greater.copy(),
        iterations=iteration,
        converged=converged,
        maximum_update=maximum_update,
        self_energy_retarded=sigma_r.copy(),
        self_energy_advanced=sigma_a.copy(),
        self_energy_lesser=sigma_l.copy(),
        self_energy_greater=sigma_g.copy(),
    )
