"""Exact finite many-body Keldysh oracle for small interacting contacts.

This module is intentionally a reference layer, not a production solver.  It
builds a number-conserving density-density Hubbard Hamiltonian in Fock space,
equilibrates the complete contacted system, propagates a post-quench
Hamiltonian exactly, and returns device-selected two-time Green functions.
It is the finite lead-coupled benchmark needed to audit EOM/Hubbard-I before
using an interacting contour approximation on a Corbino device.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


def _hermitian(value: Any, *, name: str, dimension: int | None = None) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be square.")
    if dimension is not None and matrix.shape != (dimension, dimension):
        raise ValueError(f"{name} must have shape {(dimension, dimension)}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be finite.")
    if not np.allclose(matrix, matrix.conj().T, atol=1e-12, rtol=1e-12):
        raise ValueError(f"{name} must be Hermitian.")
    return matrix


def _time_grid(value: Any) -> np.ndarray:
    grid = np.asarray(value, dtype=float)
    if grid.ndim != 1 or grid.size < 2 or not np.all(np.isfinite(grid)):
        raise ValueError("time must be a finite one-dimensional grid with at least two points.")
    if np.any(np.diff(grid) <= 0.0) or grid[0] < 0.0:
        raise ValueError("time must be increasing and start at zero or later.")
    return grid


def _annihilation_operators(mode_count: int) -> tuple[np.ndarray, ...]:
    if mode_count < 1 or mode_count > 10:
        raise ValueError("exact_interacting supports between one and ten fermionic modes.")
    dimension = 1 << mode_count
    operators: list[np.ndarray] = []
    for mode in range(mode_count):
        matrix = np.zeros((dimension, dimension), dtype=np.complex128)
        mask = (1 << mode) - 1
        for state in range(dimension):
            if state & (1 << mode):
                target = state ^ (1 << mode)
                sign = -1.0 if (state & mask).bit_count() % 2 else 1.0
                matrix[target, state] = sign
        operators.append(matrix)
    return tuple(operators)


def _validate_interactions(interactions: Any, mode_count: int) -> tuple[tuple[int, int, float], ...]:
    result: list[tuple[int, int, float]] = []
    for term in interactions or ():
        if len(term) != 3:
            raise ValueError("each interaction must be (mode_i, mode_j, U).")
        left, right, strength = int(term[0]), int(term[1]), float(term[2])
        if left == right or not (0 <= left < mode_count and 0 <= right < mode_count):
            raise ValueError("interaction mode indices must be distinct and in range.")
        if not np.isfinite(strength):
            raise ValueError("interaction strengths must be finite.")
        if left > right:
            left, right = right, left
        result.append((left, right, strength))
    return tuple(result)


def _many_body_hamiltonian(
    one_body: np.ndarray,
    interactions: tuple[tuple[int, int, float], ...],
    operators: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray]:
    mode_count = one_body.shape[0]
    dimension = operators[0].shape[0]
    result = np.zeros((dimension, dimension), dtype=np.complex128)
    occupations = []
    for mode in range(mode_count):
        occupations.append(operators[mode].conj().T @ operators[mode])
    for left in range(mode_count):
        for right in range(mode_count):
            result += one_body[left, right] * operators[left].conj().T @ operators[right]
    for left, right, strength in interactions:
        result += strength * occupations[left] @ occupations[right]
    number = sum(occupations, start=np.zeros_like(result))
    result = 0.5 * (result + result.conj().T)
    return result, number


def _thermal_density(hamiltonian: np.ndarray, number: np.ndarray, *, chemical_potential: float, temperature: float) -> np.ndarray:
    if temperature <= 0.0 or not np.isfinite(temperature):
        raise ValueError("temperature must be finite and positive for the exact grand-canonical oracle.")
    grand = hamiltonian - float(chemical_potential) * number
    energies, states = np.linalg.eigh(grand)
    shifted = energies - float(np.min(energies))
    weights = np.exp(-shifted / float(temperature))
    weights /= np.sum(weights)
    return (states * weights[None, :]) @ states.conj().T


@dataclass(frozen=True)
class ExactInteractingTwoTimeResult:
    """Exact finite interacting Keldysh components and contact observables."""

    time: np.ndarray
    initial_one_body_hamiltonian: np.ndarray
    final_one_body_hamiltonian: np.ndarray
    initial_many_body_hamiltonian: np.ndarray
    final_many_body_hamiltonian: np.ndarray
    initial_density: np.ndarray
    device_indices: np.ndarray
    lead_indices: tuple[np.ndarray, ...]
    spin_z: np.ndarray
    retarded: np.ndarray
    advanced: np.ndarray
    lesser: np.ndarray
    greater: np.ndarray
    full_lesser: np.ndarray
    imaginary_time: np.ndarray | None = None
    green_mixed: np.ndarray | None = None

    @property
    def density_matrices(self) -> np.ndarray:
        diagonal = self.lesser[np.arange(self.time.size), np.arange(self.time.size)]
        density = -1j * diagonal
        return 0.5 * (density + density.swapaxes(-1, -2).conj())

    @property
    def full_density_matrices(self) -> np.ndarray:
        diagonal = self.full_lesser[np.arange(self.time.size), np.arange(self.time.size)]
        density = -1j * diagonal
        return 0.5 * (density + density.swapaxes(-1, -2).conj())

    @property
    def spectral_identity_error(self) -> float:
        return float(np.max(np.abs(self.greater - self.lesser - self.retarded + self.advanced)))

    @property
    def density_hermiticity_error(self) -> float:
        density = self.full_density_matrices
        return float(np.max(np.abs(density - density.swapaxes(-1, -2).conj())))

    @property
    def green_rceil(self) -> np.ndarray | None:
        """Return ``G^rceil(t,tau)`` as the transpose of ``G^lceil(tau,t)``."""

        if self.green_mixed is None:
            return None
        return self.green_mixed.swapaxes(0, 1).copy()

    def device_rate(self, *, observable: Any | None = None) -> np.ndarray:
        operator = np.eye(self.initial_one_body_hamiltonian.shape[0], dtype=complex) if observable is None else np.asarray(observable, dtype=complex)
        if operator.shape != self.initial_one_body_hamiltonian.shape or not np.allclose(operator, operator.conj().T, atol=1e-12):
            raise ValueError("observable must be a Hermitian full one-body matrix.")
        density = self.full_density_matrices
        commutator = self.final_one_body_hamiltonian @ density - density @ self.final_one_body_hamiltonian
        projector = np.zeros_like(operator)
        projector[np.ix_(self.device_indices, self.device_indices)] = operator[np.ix_(self.device_indices, self.device_indices)]
        return np.real(-1j * np.einsum("ij,tji->t", projector, commutator, optimize=True))

    def lead_current(self, lead: int, *, spin: bool = False) -> np.ndarray:
        indices = self.lead_indices[int(lead)]
        observable = np.zeros_like(self.final_one_body_hamiltonian)
        values = self.spin_z[indices] if spin else np.ones(indices.size)
        observable[np.ix_(indices, indices)] = np.diag(values)
        density = self.full_density_matrices
        commutator = self.final_one_body_hamiltonian @ density - density @ self.final_one_body_hamiltonian
        # Positive sign means flow from the selected lead into the device.
        return np.real(1j * np.einsum("ij,tji->t", observable, commutator, optimize=True))

    def initial_retarded_frequency(self, omega: Any, *, eta: float = 0.02, indices: Sequence[int] | None = None) -> np.ndarray:
        grid = np.asarray(omega, dtype=float)
        if grid.ndim != 1 or not np.all(np.isfinite(grid)) or eta < 0.0 or not np.isfinite(eta):
            raise ValueError("omega must be finite and eta nonnegative.")
        selected = self.device_indices if indices is None else np.asarray(indices, dtype=int)
        if selected.ndim != 1 or np.any(selected < 0) or np.any(selected >= self.initial_one_body_hamiltonian.shape[0]):
            raise ValueError("indices must be valid one-body mode indices.")
        energies, states = np.linalg.eigh(self.initial_many_body_hamiltonian)
        rho_eigen = states.conj().T @ self.initial_density @ states
        probabilities = np.real(np.diag(rho_eigen))
        operators = _annihilation_operators(self.initial_one_body_hamiltonian.shape[0])
        transformed = [states.conj().T @ operators[int(index)] @ states for index in selected]
        denominator = grid[:, None, None] + energies[None, :, None] - energies[None, None, :] + 1j * float(eta)
        result = np.zeros((grid.size, selected.size, selected.size), dtype=np.complex128)
        for left, operator_left in enumerate(transformed):
            for right, operator_right in enumerate(transformed):
                numerator = (probabilities[:, None] + probabilities[None, :]) * operator_left * operator_right.conj()
                result[:, left, right] = np.einsum("mn,wmn->w", numerator, 1.0 / denominator, optimize=True)
        return result


def finite_interacting_partition_free_two_time(
    time: Any,
    *,
    initial_one_body_hamiltonian: Any,
    interactions: Sequence[Sequence[Any]] = (),
    final_one_body_hamiltonian: Any | None = None,
    chemical_potential: float = 0.0,
    temperature: float,
    device_indices: Sequence[int],
    lead_indices: Sequence[Sequence[int]],
    spin_z: Sequence[float] | None = None,
    imaginary_time: Any | None = None,
) -> ExactInteractingTwoTimeResult:
    """Build the exact finite interacting partition-free two-time oracle.

    If ``imaginary_time`` is supplied, the result also contains the exact
    mixed branch ``G^lceil(tau,t)`` and its real/vertical transpose
    ``green_rceil``.  The branch is evaluated from the interacting initial
    grand-canonical density, so it can seed an explicit second-Born mixed
    source without inferring it from a real-time residual.
    """

    times = _time_grid(time)
    h_initial = _hermitian(initial_one_body_hamiltonian, name="initial_one_body_hamiltonian")
    mode_count = h_initial.shape[0]
    h_final = h_initial.copy() if final_one_body_hamiltonian is None else _hermitian(final_one_body_hamiltonian, name="final_one_body_hamiltonian", dimension=mode_count)
    device = np.asarray(device_indices, dtype=int)
    leads = tuple(np.asarray(group, dtype=int) for group in lead_indices)
    if device.ndim != 1 or device.size == 0 or np.any(device < 0) or np.any(device >= mode_count):
        raise ValueError("device_indices must be nonempty valid indices.")
    if any(group.ndim != 1 or group.size == 0 or np.any(group < 0) or np.any(group >= mode_count) for group in leads):
        raise ValueError("lead_indices must contain nonempty valid groups.")
    if np.unique(np.concatenate((device,) + leads)).size != mode_count:
        raise ValueError("device and lead groups must partition all one-body modes.")
    spins = np.ones(mode_count, dtype=float) if spin_z is None else np.asarray(spin_z, dtype=float)
    if spins.shape != (mode_count,) or not np.all(np.isfinite(spins)):
        raise ValueError("spin_z must have one finite value per mode.")
    interaction_terms = _validate_interactions(interactions, mode_count)
    operators = _annihilation_operators(mode_count)
    initial_mb, number = _many_body_hamiltonian(h_initial, interaction_terms, operators)
    final_mb, _ = _many_body_hamiltonian(h_final, interaction_terms, operators)
    initial_density = _thermal_density(initial_mb, number, chemical_potential=float(chemical_potential), temperature=float(temperature))
    energies, states = np.linalg.eigh(final_mb)
    evolution = np.einsum("ik,tk,kj->tij", states, np.exp(-1j * times[:, None] * energies[None, :]), states.conj().T, optimize=True)
    transformed = [np.einsum("tab,bc,tcd->tad", evolution.conj().transpose(0, 2, 1), operator, evolution, optimize=True) for operator in operators]
    full_dimension = 1 << mode_count
    lesser_full = np.zeros((times.size, times.size, mode_count, mode_count), dtype=np.complex128)
    greater_full = np.zeros_like(lesser_full)
    for left in range(mode_count):
        for right in range(mode_count):
            for ti in range(times.size):
                for tj in range(times.size):
                    annihilation_left = transformed[left][ti]
                    creation_right = transformed[right][tj].conj().T
                    lesser_full[ti, tj, left, right] = 1j * np.trace(initial_density @ creation_right @ annihilation_left)
                    greater_full[ti, tj, left, right] = -1j * np.trace(initial_density @ annihilation_left @ creation_right)
    theta = np.tril(np.ones((times.size, times.size), dtype=float), k=-1) + 0.5 * np.eye(times.size)
    retarded_full = theta[:, :, None, None] * (greater_full - lesser_full)
    advanced_full = retarded_full.swapaxes(0, 1).swapaxes(-1, -2).conj()
    selected = np.ix_(device, device)
    imaginary = None
    mixed = None
    if imaginary_time is not None:
        imaginary = np.asarray(imaginary_time, dtype=float)
        if imaginary.ndim != 1 or imaginary.size < 2 or not np.all(np.isfinite(imaginary)) or np.any(np.diff(imaginary) <= 0.0) or imaginary[0] < 0.0:
            raise ValueError("imaginary_time must be a finite increasing nonnegative grid.")
        grand_initial = initial_mb - float(chemical_potential) * number
        grand_energies, grand_states = np.linalg.eigh(grand_initial)
        transformed_initial = [grand_states.conj().T @ operator @ grand_states for operator in operators]
        # Boltzmann weights in the same eigenbasis, kept logarithmic.  The
        # imaginary-time Heisenberg factor ``exp(tau (E_m - E_n))`` must never
        # be formed on its own: it reaches ``exp(beta * spectral_width)`` while
        # the trace it enters is bounded by one for ``0 <= tau <= beta``.
        # Building it first destroys the cancellation against the thermal
        # weights long before it overflows -- with a spectral width of about
        # seven the mixed branch is already wrong by fifteen orders of
        # magnitude at ``T = 0.05``.  Combining the exponents keeps every
        # entry at or below one, which is the KMS bound.
        log_weights = -(grand_energies - float(np.min(grand_energies))) / float(temperature)
        log_weights = log_weights - float(np.log(np.sum(np.exp(log_weights))))
        energy_difference = grand_energies[:, None] - grand_energies[None, :]
        mixed_full = np.zeros((imaginary.size, times.size, mode_count, mode_count), dtype=np.complex128)
        for imaginary_index, tau in enumerate(imaginary):
            kernel = np.exp(log_weights[:, None] + float(tau) * energy_difference)
            for right in range(times.size):
                creation_eigen = [
                    grand_states.conj().T @ transformed[mode][right].conj().T @ grand_states
                    for mode in range(mode_count)
                ]
                for left in range(mode_count):
                    weighted_left = kernel * transformed_initial[left]
                    for mode_right in range(mode_count):
                        # Tr[rho A(tau) B] = sum_mn p_m A(tau)_mn B_nm.
                        mixed_full[imaginary_index, right, left, mode_right] = -1j * np.sum(
                            weighted_left * creation_eigen[mode_right].T
                        )
        mixed = mixed_full[:, :, device[:, None], device[None, :]]
    return ExactInteractingTwoTimeResult(
        time=times.copy(),
        initial_one_body_hamiltonian=h_initial,
        final_one_body_hamiltonian=h_final,
        initial_many_body_hamiltonian=initial_mb,
        final_many_body_hamiltonian=final_mb,
        initial_density=initial_density,
        device_indices=device.copy(),
        lead_indices=tuple(group.copy() for group in leads),
        spin_z=spins,
        retarded=retarded_full[:, :, selected[0], selected[1]],
        advanced=advanced_full[:, :, selected[0], selected[1]],
        lesser=lesser_full[:, :, selected[0], selected[1]],
        greater=greater_full[:, :, selected[0], selected[1]],
        full_lesser=lesser_full,
        imaginary_time=None if imaginary is None else imaginary.copy(),
        green_mixed=mixed,
    )


__all__ = ["ExactInteractingTwoTimeResult", "finite_interacting_partition_free_two_time"]
