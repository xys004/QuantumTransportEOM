"""Stationary self-consistent Born (SCBA) electron--boson transport.

The solver is deliberately matrix-valued and keeps all four Keldysh
components.  A local Einstein mode is represented by a Hermitian coupling
matrix ``V`` and the Fock self-energy is iterated on an explicit energy grid.
The retarded component is reconstructed from its Keldysh discontinuity using
the discrete Kramers--Kronig transform, so the approximation is auditable
rather than a phenomenological broadening.

This is a stationary conserving benchmark.  It is not yet a general
time-dependent Kadanoff--Baym propagator; :func:`scba_two_time_greens` only
Fourier-transforms the converged stationary solution to explicit ``(t,t')``
kernels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .continuum_two_time import (
    ContinuumTwoTimeGreenResult,
    _guard_allocation,
    inverse_fourier_two_time,
)


ComplexMatrix = np.ndarray
ComplexStack = np.ndarray


def _trapz(values: np.ndarray, grid: np.ndarray, axis: int = 0) -> np.ndarray:
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return trapezoid(values, grid, axis=axis)


def _grid(values: Any, *, name: str, minimum: int = 3) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or result.size < minimum:
        raise ValueError(f"{name} must be a one-dimensional grid with at least {minimum} points.")
    if not np.all(np.isfinite(result)) or np.any(np.diff(result) <= 0.0):
        raise ValueError(f"{name} must be finite and strictly increasing.")
    return result


def _hermitian(value: Any, *, name: str, dim: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.complex128)
    if result.ndim != 2 or result.shape[0] != result.shape[1]:
        raise ValueError(f"{name} must be square.")
    if dim is not None and result.shape != (dim, dim):
        raise ValueError(f"{name} must have shape {(dim, dim)}.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite.")
    if not np.allclose(result, result.conj().T, atol=1e-12, rtol=1e-12):
        raise ValueError(f"{name} must be Hermitian.")
    return result


def _gammas(values: Sequence[Any] | np.ndarray, dim: int) -> np.ndarray:
    result = np.asarray(values, dtype=np.complex128)
    if result.ndim != 3 or result.shape[1:] != (dim, dim) or result.shape[0] == 0:
        raise ValueError("lead_broadenings must have shape (n_leads, dim, dim).")
    for index, gamma in enumerate(result):
        _hermitian(gamma, name=f"lead_broadenings[{index}]", dim=dim)
        if np.min(np.linalg.eigvalsh(gamma)) < -1e-10:
            raise ValueError(f"lead_broadenings[{index}] must be positive semidefinite.")
    return result


def _sample_stack(values: np.ndarray, grid: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Linearly sample a matrix stack with zero padding outside the grid."""
    result = np.empty((query.size, values.shape[1], values.shape[2]), dtype=np.complex128)
    for left in range(values.shape[1]):
        for right in range(values.shape[2]):
            result[:, left, right] = np.interp(
                query,
                grid,
                values[:, left, right].real,
                left=0.0,
                right=0.0,
            ) + 1j * np.interp(
                query,
                grid,
                values[:, left, right].imag,
                left=0.0,
                right=0.0,
            )
    return result


def _retarded_from_discontinuity(
    sigma_lesser: np.ndarray,
    sigma_greater: np.ndarray,
    energy: np.ndarray,
) -> np.ndarray:
    """Reconstruct ``Sigma^r`` from ``Sigma^>-Sigma^<`` by Kramers--Kronig."""
    discontinuity = sigma_greater - sigma_lesser
    gamma = 1j * discontinuity
    gamma = 0.5 * (gamma + gamma.swapaxes(-1, -2).conj())
    weights = np.empty_like(energy)
    weights[0] = 0.5 * (energy[1] - energy[0])
    weights[-1] = 0.5 * (energy[-1] - energy[-2])
    if energy.size > 2:
        weights[1:-1] = 0.5 * (energy[2:] - energy[:-2])
    result = np.empty_like(gamma)
    for index, value in enumerate(energy):
        denominator = value - energy
        denominator[index] = np.inf
        principal = np.sum(
            weights[:, None, None] * gamma / denominator[:, None, None], axis=0
        ) / (2.0 * np.pi)
        result[index] = 0.5 * (principal + principal.conj().T) - 0.5j * gamma[index]
    return result


@dataclass(frozen=True)
class ScbaResult:
    """Converged stationary SCBA Green functions and self-energies."""

    energy: np.ndarray
    hamiltonian: np.ndarray
    lead_broadenings: np.ndarray
    lead_chemical_potentials: np.ndarray
    temperature: float
    boson_frequency: float
    boson_temperature: float
    coupling: np.ndarray
    retarded: np.ndarray
    advanced: np.ndarray
    lesser: np.ndarray
    greater: np.ndarray
    interaction_retarded: np.ndarray
    interaction_lesser: np.ndarray
    interaction_greater: np.ndarray
    converged: bool
    iterations: int
    maximum_update: float
    lead_currents: np.ndarray

    @property
    def current_conservation_error(self) -> float:
        return float(abs(np.sum(self.lead_currents)))

    @property
    def spectral_identity_error(self) -> float:
        return float(
            np.max(np.abs(self.greater - self.lesser - self.retarded + self.advanced))
        )

    @property
    def spectral_sum_rule_error(self) -> float:
        """Finite energy-window error in ``integral A(E)/(2 pi) = I``."""

        spectral = 1j * (self.retarded - self.advanced)
        integrated = _trapz(spectral, self.energy, axis=0) / (2.0 * np.pi)
        identity = np.eye(self.hamiltonian.shape[0], dtype=np.complex128)
        return float(np.max(np.abs(integrated - identity)))

    @property
    def interaction_keldysh_error(self) -> float:
        return float(
            np.max(
                np.abs(
                    self.interaction_greater
                    - self.interaction_lesser
                    - self.interaction_retarded
                    + self.interaction_retarded.swapaxes(-1, -2).conj()
                )
            )
        )

    def fdt_error(self, chemical_potential: float | None = None) -> float:
        """Return equilibrium FDT residual, or ``inf`` for unequal lead mus."""
        if chemical_potential is None:
            if not np.allclose(self.lead_chemical_potentials, self.lead_chemical_potentials[0]):
                return float("inf")
            chemical_potential = float(self.lead_chemical_potentials[0])
        from .greens import fermi_dirac

        occupation = fermi_dirac(
            self.energy, mu=float(chemical_potential), temperature=self.temperature
        )
        expected = -occupation[:, None, None] * (self.retarded - self.advanced)
        return float(np.max(np.abs(self.lesser - expected)))


def self_consistent_born_electron_boson(
    hamiltonian: Any,
    energy: Any,
    lead_broadenings: Sequence[Any] | np.ndarray,
    lead_chemical_potentials: Sequence[float],
    *,
    coupling: Any,
    boson_frequency: float,
    temperature: float = 0.0,
    boson_temperature: float | None = None,
    max_iterations: int = 100,
    mixing: float = 0.5,
    tolerance: float = 1e-10,
) -> ScbaResult:
    r"""Solve a stationary electron--boson SCBA problem on an energy grid.

    The lead components are ``Sigma_alpha^< = i f_alpha Gamma_alpha`` and
    ``Sigma_alpha^> = -i(1-f_alpha) Gamma_alpha``.  For an Einstein mode,

    ``Sigma_ph^<(E)=V[N G^<(E-w)+(N+1)G^<(E+w)]V``

    and the greater component has the corresponding emission/absorption
    interchange.  The retarded part is reconstructed by a discrete Hilbert
    transform, preserving the Keldysh discontinuity on the supplied grid.
    """
    from .greens import fermi_dirac

    matrix = _hermitian(hamiltonian, name="hamiltonian")
    dim = matrix.shape[0]
    energies = _grid(energy, name="energy")
    gammas = _gammas(lead_broadenings, dim)
    mus = np.asarray(lead_chemical_potentials, dtype=float)
    if mus.shape != (gammas.shape[0],) or not np.all(np.isfinite(mus)):
        raise ValueError("lead_chemical_potentials must match the number of leads.")
    if temperature < 0.0 or not np.isfinite(temperature):
        raise ValueError("temperature must be finite and nonnegative.")
    mode_temperature = temperature if boson_temperature is None else float(boson_temperature)
    if mode_temperature < 0.0 or not np.isfinite(mode_temperature):
        raise ValueError("boson_temperature must be finite and nonnegative.")
    frequency = float(boson_frequency)
    if frequency <= 0.0 or not np.isfinite(frequency):
        raise ValueError("boson_frequency must be finite and positive.")
    vertex = _hermitian(coupling, name="coupling", dim=dim)
    if max_iterations < 1 or mixing <= 0.0 or mixing > 1.0 or tolerance <= 0.0:
        raise ValueError("invalid SCBA iteration controls.")

    n_bose = 0.0 if mode_temperature == 0.0 or frequency / mode_temperature > 500.0 else 1.0 / np.expm1(frequency / mode_temperature)
    identity = np.eye(dim, dtype=np.complex128)
    total_gamma = np.sum(gammas, axis=0)
    sigma_lead_r = -0.5j * total_gamma
    fermi = np.stack(
        [fermi_dirac(energies, mu=float(mu), temperature=temperature) for mu in mus]
    )
    sigma_lead_l = np.einsum("ak,aij->kij", 1j * fermi, gammas, optimize=True)
    sigma_lead_g = np.einsum("ak,aij->kij", 1j * (fermi - 1.0), gammas, optimize=True)
    sigma_ph_l = np.zeros((energies.size, dim, dim), dtype=np.complex128)
    sigma_ph_g = np.zeros_like(sigma_ph_l)
    green_l = np.zeros_like(sigma_ph_l)
    green_g = np.zeros_like(sigma_ph_l)
    sigma_ph_r = np.zeros_like(sigma_ph_l)
    maximum_update = float("inf")
    converged = False
    for iteration in range(1, max_iterations + 1):
        old_l = sigma_ph_l.copy()
        old_g = sigma_ph_g.copy()
        for index, value in enumerate(energies):
            retarded = np.linalg.inv(
                (value * identity - matrix - sigma_lead_r - sigma_ph_r[index])
            )
            advanced = retarded.conj().T
            green_l[index] = retarded @ (sigma_lead_l[index] + sigma_ph_l[index]) @ advanced
            green_g[index] = retarded @ (sigma_lead_g[index] + sigma_ph_g[index]) @ advanced
        less_minus = _sample_stack(green_l, energies, energies - frequency)
        less_plus = _sample_stack(green_l, energies, energies + frequency)
        great_minus = _sample_stack(green_g, energies, energies - frequency)
        great_plus = _sample_stack(green_g, energies, energies + frequency)
        new_l = np.einsum(
            "ab,kbc,cd->kad", vertex, n_bose * less_minus + (n_bose + 1.0) * less_plus, vertex, optimize=True
        )
        new_g = np.einsum(
            "ab,kbc,cd->kad", vertex, n_bose * great_plus + (n_bose + 1.0) * great_minus, vertex, optimize=True
        )
        sigma_ph_l = mixing * new_l + (1.0 - mixing) * sigma_ph_l
        sigma_ph_g = mixing * new_g + (1.0 - mixing) * sigma_ph_g
        sigma_ph_r = _retarded_from_discontinuity(sigma_ph_l, sigma_ph_g, energies)
        maximum_update = float(
            max(np.max(np.abs(sigma_ph_l - old_l)), np.max(np.abs(sigma_ph_g - old_g)))
        )
        if maximum_update < tolerance:
            converged = True
            break

    retarded_stack = np.empty_like(green_l)
    advanced_stack = np.empty_like(green_l)
    for index, value in enumerate(energies):
        retarded_stack[index] = np.linalg.inv(
            value * identity - matrix - sigma_lead_r - sigma_ph_r[index]
        )
        advanced_stack[index] = retarded_stack[index].conj().T
        green_l[index] = retarded_stack[index] @ (sigma_lead_l[index] + sigma_ph_l[index]) @ advanced_stack[index]
        green_g[index] = retarded_stack[index] @ (sigma_lead_g[index] + sigma_ph_g[index]) @ advanced_stack[index]
    currents = np.empty(gammas.shape[0], dtype=float)
    for lead in range(gammas.shape[0]):
        # Use explicit matrices because sigma_lead_l/g are already summed over
        # leads; reconstruct the selected lead's Keldysh components here.
        selected_l = 1j * fermi[lead, :, None, None] * gammas[lead]
        selected_g = 1j * (fermi[lead, :, None, None] - 1.0) * gammas[lead]
        integrand = np.real(
            np.trace(selected_l @ green_g - selected_g @ green_l, axis1=1, axis2=2)
        ) / (2.0 * np.pi)
        currents[lead] = float(_trapz(integrand, energies))
    return ScbaResult(
        energy=energies.copy(),
        hamiltonian=matrix.copy(),
        lead_broadenings=gammas.copy(),
        lead_chemical_potentials=mus.copy(),
        temperature=float(temperature),
        boson_frequency=frequency,
        boson_temperature=mode_temperature,
        coupling=vertex.copy(),
        retarded=retarded_stack,
        advanced=advanced_stack,
        lesser=green_l.copy(),
        greater=green_g.copy(),
        interaction_retarded=sigma_ph_r.copy(),
        interaction_lesser=sigma_ph_l.copy(),
        interaction_greater=sigma_ph_g.copy(),
        converged=converged,
        iterations=iteration,
        maximum_update=maximum_update,
        lead_currents=currents,
    )


def scba_two_time_greens(
    result: ScbaResult,
    time: Any,
    *,
    max_memory_bytes: int = 512 * 1024**2,
) -> ContinuumTwoTimeGreenResult:
    """Fourier-transform a converged stationary SCBA result to ``(t,t')``."""
    times = np.asarray(time, dtype=float)
    if times.ndim != 1 or times.size < 2 or not np.all(np.isfinite(times)):
        raise ValueError("time must be a finite one-dimensional grid with at least two points.")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("time must be strictly increasing.")
    _guard_allocation(times.size, result.hamiltonian.shape[0], 4, max_memory_bytes)
    retarded = inverse_fourier_two_time(
        result.energy, result.retarded, times, max_memory_bytes=max_memory_bytes // 2
    )
    lesser = inverse_fourier_two_time(
        result.energy, result.lesser, times, max_memory_bytes=max_memory_bytes // 2
    )
    greater = inverse_fourier_two_time(
        result.energy, result.greater, times, max_memory_bytes=max_memory_bytes // 2
    )
    return ContinuumTwoTimeGreenResult(
        time=times.copy(),
        omega_grid=result.energy.copy(),
        retarded=retarded,
        advanced=retarded.swapaxes(0, 1).swapaxes(-1, -2).conj(),
        lesser=lesser,
        greater=greater,
    )
