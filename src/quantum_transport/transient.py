"""Exact real-time and two-time Green functions for finite quadratic systems.

This module is the noninteracting finite-system oracle for transient Keldysh
calculations.  It propagates a one-body Hamiltonian ``h(t)`` with a unitary
midpoint rule and constructs

``G^<(t,t') = i U(t,t0) rho0 U(t',t0)^dagger``

and the corresponding greater, retarded, and advanced components.

It does not approximate continuum reservoirs.  Open devices can be represented
exactly by including finite lead orbitals in ``h(t)``; continuum self-energies,
partition-free wide-band injection, and interaction memory kernels belong to a
later solver layer.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .greens import fermi_dirac


ComplexMatrix = NDArray[np.complex128]
ComplexStack = NDArray[np.complex128]
RealArray = NDArray[np.float64]
HamiltonianFn = Callable[[float], ComplexMatrix]

_VALID_COMPONENTS = frozenset(
    {"retarded", "advanced", "lesser", "greater"}
)


def _time_grid(time: RealArray) -> RealArray:
    grid = np.asarray(time, dtype=float)
    if grid.ndim != 1 or grid.size < 2:
        raise ValueError("time must be a one-dimensional grid.")
    if not np.all(np.isfinite(grid)):
        raise ValueError("time grid must contain finite values.")
    if np.any(np.diff(grid) <= 0):
        raise ValueError("time grid must be strictly increasing.")
    return grid


def _hermitian_matrix(
    value: ComplexMatrix,
    *,
    name: str,
    dimension: int | None = None,
) -> ComplexMatrix:
    matrix = np.asarray(value, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix.")
    if dimension is not None and matrix.shape != (dimension, dimension):
        raise ValueError(
            f"{name} must have shape {(dimension, dimension)}, "
            f"got {matrix.shape}."
        )
    if not np.allclose(matrix, matrix.conj().T, atol=1e-12):
        raise ValueError(f"{name} must be Hermitian.")
    return matrix


def equilibrium_one_body_density(
    hamiltonian: ComplexMatrix,
    *,
    mu: float = 0.0,
    temperature: float = 0.0,
) -> ComplexMatrix:
    """Return the grand-canonical one-body density matrix ``f[h-mu]``."""
    matrix = _hermitian_matrix(
        hamiltonian, name="hamiltonian"
    )
    if temperature < 0:
        raise ValueError("temperature cannot be negative.")
    energies, states = np.linalg.eigh(matrix)
    occupations = fermi_dirac(
        energies, mu=float(mu), temperature=float(temperature)
    )
    return (states * occupations) @ states.conj().T


def unitary_from_hermitian(
    hamiltonian: ComplexMatrix,
    duration: float,
) -> ComplexMatrix:
    """Return ``exp(-i h duration)`` from a Hermitian eigendecomposition."""
    matrix = _hermitian_matrix(
        hamiltonian, name="hamiltonian"
    )
    energies, states = np.linalg.eigh(matrix)
    phases = np.exp(-1j * energies * float(duration))
    return (states * phases) @ states.conj().T


def propagate_unitaries(
    time: RealArray,
    hamiltonian_at_time: HamiltonianFn,
) -> ComplexStack:
    """Return ``U(t_i,t_0)`` using exact midpoint exponentials per step."""
    grid = _time_grid(time)
    initial_hamiltonian = _hermitian_matrix(
        hamiltonian_at_time(float(grid[0])),
        name="time-dependent Hamiltonian",
    )
    dimension = initial_hamiltonian.shape[0]
    evolution = np.empty(
        (grid.size, dimension, dimension), dtype=np.complex128
    )
    evolution[0] = np.eye(dimension, dtype=np.complex128)
    for index, (left, right) in enumerate(
        zip(grid[:-1], grid[1:]), start=1
    ):
        midpoint = 0.5 * (left + right)
        hamiltonian = _hermitian_matrix(
            hamiltonian_at_time(float(midpoint)),
            name="time-dependent Hamiltonian",
            dimension=dimension,
        )
        step = unitary_from_hermitian(
            hamiltonian, float(right - left)
        )
        evolution[index] = step @ evolution[index - 1]
    return evolution


def density_matrices_from_unitaries(
    evolution: ComplexStack,
    initial_density_matrix: ComplexMatrix,
) -> ComplexStack:
    """Return ``rho(t)=U(t,t0) rho0 U(t,t0)^dagger``."""
    unitary_values = np.asarray(
        evolution, dtype=np.complex128
    )
    if (
        unitary_values.ndim != 3
        or unitary_values.shape[1] != unitary_values.shape[2]
    ):
        raise ValueError(
            "evolution must have shape (n_time, dim, dim)."
        )
    dimension = unitary_values.shape[1]
    density = _hermitian_matrix(
        initial_density_matrix,
        name="initial density matrix",
        dimension=dimension,
    )
    occupations = np.linalg.eigvalsh(density)
    if np.min(occupations) < -1e-10 or np.max(occupations) > 1.0 + 1e-10:
        raise ValueError(
            "one-body density eigenvalues must lie in [0, 1]."
        )
    values = np.einsum(
        "tia,ab,tjb->tij",
        unitary_values,
        density,
        unitary_values.conj(),
        optimize=True,
    )
    return 0.5 * (
        values + values.transpose(0, 2, 1).conj()
    )


def propagate_density_matrix(
    initial_density_matrix: ComplexMatrix,
    time: RealArray,
    hamiltonian_at_time: HamiltonianFn,
) -> ComplexStack:
    """Propagate a physical one-body density matrix on a time grid."""
    evolution = propagate_unitaries(time, hamiltonian_at_time)
    return density_matrices_from_unitaries(
        evolution, initial_density_matrix
    )


def iterate_density_matrices(
    initial_density_matrix: ComplexMatrix,
    time: RealArray,
    hamiltonian_at_time: HamiltonianFn,
) -> Iterator[ComplexMatrix]:
    """Yield ``rho(t)`` sequentially without storing its full history.

    The midpoint propagator is identical to :func:`propagate_unitaries`.
    Consecutive equal Hamiltonians on an equal-step grid reuse the same matrix
    exponential, which is especially useful after a finite drive ends.  The
    yielded matrices are independent arrays and may be retained by callers.
    """
    grid = _time_grid(time)
    initial_hamiltonian = _hermitian_matrix(
        hamiltonian_at_time(float(grid[0])),
        name="time-dependent Hamiltonian",
    )
    dimension = initial_hamiltonian.shape[0]
    density = _hermitian_matrix(
        initial_density_matrix,
        name="initial density matrix",
        dimension=dimension,
    ).copy()
    occupations = np.linalg.eigvalsh(density)
    if np.min(occupations) < -1e-10 or np.max(occupations) > 1.0 + 1e-10:
        raise ValueError(
            "one-body density eigenvalues must lie in [0, 1]."
        )
    yield density.copy()

    cached_hamiltonian: ComplexMatrix | None = None
    cached_duration: float | None = None
    cached_step: ComplexMatrix | None = None
    for left, right in zip(grid[:-1], grid[1:]):
        midpoint = 0.5 * (left + right)
        hamiltonian = _hermitian_matrix(
            hamiltonian_at_time(float(midpoint)),
            name="time-dependent Hamiltonian",
            dimension=dimension,
        )
        duration = float(right - left)
        if (
            cached_hamiltonian is not None
            and cached_duration == duration
            and np.array_equal(hamiltonian, cached_hamiltonian)
        ):
            step = cached_step
        else:
            step = unitary_from_hermitian(hamiltonian, duration)
            cached_hamiltonian = hamiltonian.copy()
            cached_duration = duration
            cached_step = step
        density = step @ density @ step.conj().T
        density = 0.5 * (density + density.conj().T)
        yield density.copy()


@dataclass(frozen=True)
class TwoTimeGreenResult:
    """Selected exact two-time Green-function components."""

    time: RealArray
    evolution: ComplexStack
    retarded: ComplexStack | None = None
    advanced: ComplexStack | None = None
    lesser: ComplexStack | None = None
    greater: ComplexStack | None = None

    @property
    def dimension(self) -> int:
        return int(self.evolution.shape[1])

    def component(self, name: str) -> ComplexStack:
        """Return a requested component or raise if it was not built."""
        if name not in _VALID_COMPONENTS:
            raise ValueError(
                f"component must be one of {sorted(_VALID_COMPONENTS)}."
            )
        value = getattr(self, name)
        if value is None:
            raise ValueError(
                f"component {name!r} was not requested."
            )
        return value

    def density_matrices(self) -> ComplexStack:
        """Return ``rho(t)=-i G^<(t,t)``."""
        lesser = self.component("lesser")
        diagonal = lesser[
            np.arange(self.time.size),
            np.arange(self.time.size),
        ]
        density = -1j * diagonal
        return 0.5 * (
            density + density.transpose(0, 2, 1).conj()
        )

    def keldysh(self) -> ComplexStack:
        """Return ``G^K=G^>+G^<``."""
        return self.component("greater") + self.component("lesser")

    def spectral_identity_error(self) -> float:
        """Return max norm of ``G^>-G^<-(G^r-G^a)``."""
        residual = (
            self.component("greater")
            - self.component("lesser")
            - self.component("retarded")
            + self.component("advanced")
        )
        return float(
            np.max(np.linalg.norm(residual, axis=(-2, -1)))
        )


def _normalize_components(
    components: Iterable[str],
) -> frozenset[str]:
    selected = frozenset(str(value) for value in components)
    unknown = selected - _VALID_COMPONENTS
    if unknown:
        raise ValueError(
            f"unknown two-time components: {sorted(unknown)}."
        )
    if not selected:
        raise ValueError("request at least one two-time component.")
    return selected


def _estimated_two_time_bytes(
    n_time: int,
    dimension: int,
    n_components: int,
) -> int:
    complex_bytes = np.dtype(np.complex128).itemsize
    evolution = n_time * dimension * dimension * complex_bytes
    components = (
        n_components
        * n_time
        * n_time
        * dimension
        * dimension
        * complex_bytes
    )
    pair_workspace = (
        n_time
        * n_time
        * dimension
        * dimension
        * complex_bytes
    )
    return int(evolution + components + pair_workspace)


def two_time_greens(
    time: RealArray,
    hamiltonian_at_time: HamiltonianFn,
    initial_density_matrix: ComplexMatrix,
    *,
    components: Sequence[str] = (
        "retarded",
        "advanced",
        "lesser",
        "greater",
    ),
    max_memory_bytes: int = 512 * 1024**2,
) -> TwoTimeGreenResult:
    r"""Construct exact finite-system ``G^{r,a,<,>}(t,t')``.

    The equal-time convention is ``theta(0)=1/2``.  It makes

    ``G^>(t,t') - G^<(t,t') = G^r(t,t') - G^a(t,t')``

    hold also on the discrete time diagonal.

    Memory scales as ``O(n_time**2 * dimension**2)`` per component.  Use
    :func:`propagate_density_matrix` when only equal-time observables are
    required.
    """
    grid = _time_grid(time)
    selected = _normalize_components(components)
    evolution = propagate_unitaries(grid, hamiltonian_at_time)
    dimension = evolution.shape[1]
    density = _hermitian_matrix(
        initial_density_matrix,
        name="initial density matrix",
        dimension=dimension,
    )
    occupations = np.linalg.eigvalsh(density)
    if np.min(occupations) < -1e-10 or np.max(occupations) > 1.0 + 1e-10:
        raise ValueError(
            "one-body density eigenvalues must lie in [0, 1]."
        )
    estimate = _estimated_two_time_bytes(
        grid.size, dimension, len(selected)
    )
    if max_memory_bytes <= 0:
        raise ValueError("max_memory_bytes must be positive.")
    if estimate > max_memory_bytes:
        raise MemoryError(
            "two-time allocation estimate "
            f"{estimate / 1024**2:.1f} MiB exceeds limit "
            f"{max_memory_bytes / 1024**2:.1f} MiB; request fewer "
            "components or use propagate_density_matrix."
        )

    pair_propagator = np.einsum(
        "tia,sja->tsij",
        evolution,
        evolution.conj(),
        optimize=True,
    )
    values: dict[str, ComplexStack | None] = {
        "retarded": None,
        "advanced": None,
        "lesser": None,
        "greater": None,
    }
    if "retarded" in selected or "advanced" in selected:
        theta = np.tril(
            np.ones((grid.size, grid.size), dtype=float), k=-1
        )
        theta += 0.5 * np.eye(grid.size)
        retarded = (
            -1j
            * theta[:, :, None, None]
            * pair_propagator
        )
        if "retarded" in selected:
            values["retarded"] = retarded
        if "advanced" in selected:
            values["advanced"] = (
                retarded.transpose(1, 0, 3, 2).conj()
            )
    if "lesser" in selected:
        values["lesser"] = 1j * np.einsum(
            "tia,ab,sjb->tsij",
            evolution,
            density,
            evolution.conj(),
            optimize=True,
        )
    if "greater" in selected:
        complement = np.eye(dimension) - density
        values["greater"] = -1j * np.einsum(
            "tia,ab,sjb->tsij",
            evolution,
            complement,
            evolution.conj(),
            optimize=True,
        )
    return TwoTimeGreenResult(
        time=grid,
        evolution=evolution,
        retarded=values["retarded"],
        advanced=values["advanced"],
        lesser=values["lesser"],
        greater=values["greater"],
    )


def one_body_bond_current(
    hamiltonian: ComplexMatrix,
    density_matrix: ComplexMatrix,
    source: int,
    target: int,
    *,
    charge: float = 1.0,
) -> float:
    """Return current oriented from ``source`` to ``target``."""
    matrix = np.asarray(hamiltonian, dtype=np.complex128)
    density = np.asarray(density_matrix, dtype=np.complex128)
    if matrix.shape != density.shape:
        raise ValueError(
            "hamiltonian and density matrix dimensions differ."
        )
    dimension = matrix.shape[0]
    if not (
        0 <= int(source) < dimension
        and 0 <= int(target) < dimension
    ):
        raise ValueError("bond index lies outside the matrix.")
    return float(
        -2.0
        * charge
        * np.imag(matrix[int(source), int(target)]
                  * density[int(target), int(source)])
    )


def _normalize_block_indices(
    indices: int | Sequence[int],
    dimension: int,
    *,
    name: str,
) -> np.ndarray:
    if np.isscalar(indices):
        values = np.array([int(indices)], dtype=int)
    else:
        values = np.asarray(list(indices), dtype=int)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"{name} must contain at least one orbital index.")
    if np.any(values < 0) or np.any(values >= dimension):
        raise ValueError(f"{name} contains an index outside the matrix.")
    if np.unique(values).size != values.size:
        raise ValueError(f"{name} must not contain repeated orbital indices.")
    return values


def one_body_spin_bond_current(
    hamiltonian: ComplexMatrix,
    density_matrix: ComplexMatrix,
    source: int | Sequence[int],
    target: int | Sequence[int],
    spin_operator: ComplexMatrix,
    *,
    charge: float = 1.0,
) -> float:
    """Return a symmetrized spin current from one orbital block to another.

    Source and target are equally sized site/orbital blocks ordered in the
    same local basis. For a spinful block, passing sigma_z / 2 gives the
    conventional symmetrized z-spin current. The symmetrization is essential
    when the hopping does not commute with the spin operator, as in Rashba
    systems. Local torque is a separate term in spin continuity.
    """

    matrix = np.asarray(hamiltonian, dtype=np.complex128)
    density = np.asarray(density_matrix, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("hamiltonian must be a square matrix.")
    if density.shape != matrix.shape:
        raise ValueError("hamiltonian and density matrix dimensions differ.")
    dimension = matrix.shape[0]
    source_indices = _normalize_block_indices(
        source, dimension, name="source block"
    )
    target_indices = _normalize_block_indices(
        target, dimension, name="target block"
    )
    if source_indices.size != target_indices.size:
        raise ValueError("source and target blocks must have equal sizes.")
    if np.intersect1d(source_indices, target_indices).size:
        raise ValueError("source and target blocks must be disjoint.")
    spin = np.asarray(spin_operator, dtype=np.complex128)
    block_size = source_indices.size
    if spin.shape != (block_size, block_size):
        raise ValueError(
            "spin_operator must have the same shape as each orbital block."
        )
    if not np.all(np.isfinite(spin)):
        raise ValueError("spin_operator must contain finite values.")
    if not np.allclose(spin, spin.conj().T, atol=1e-12):
        raise ValueError("spin_operator must be Hermitian.")
    hopping = matrix[np.ix_(source_indices, target_indices)]
    coherence = density[np.ix_(target_indices, source_indices)]
    symmetrized = 0.5 * (spin @ hopping + hopping @ spin)
    return float(-2.0 * charge * np.imag(np.trace(symmetrized @ coherence)))


def region_interface_current(
    hamiltonian: ComplexMatrix,
    density_matrix: ComplexMatrix,
    region: Sequence[int],
    *,
    charge: float = 1.0,
) -> float:
    """Return total current leaving a declared orbital region."""
    matrix = np.asarray(hamiltonian, dtype=np.complex128)
    density = np.asarray(density_matrix, dtype=np.complex128)
    if matrix.shape != density.shape:
        raise ValueError(
            "hamiltonian and density matrix dimensions differ."
        )
    dimension = matrix.shape[0]
    inside = {int(index) for index in region}
    if not inside:
        raise ValueError("region cannot be empty.")
    if min(inside) < 0 or max(inside) >= dimension:
        raise ValueError("region index lies outside the matrix.")
    outside = set(range(dimension)) - inside
    return float(
        sum(
            one_body_bond_current(
                matrix,
                density,
                source,
                target,
                charge=charge,
            )
            for source in inside
            for target in outside
        )
    )
