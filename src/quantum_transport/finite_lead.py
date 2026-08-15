"""Exact partition-free finite-lead benchmark with mixed Keldysh kernels.

This module is a microscopic validation oracle for the initial-correlation
term.  A finite device is coupled to noninteracting finite leads, the complete
device-plus-leads Hamiltonian is equilibrated at ``t < 0``, and the device or
lead Hamiltonians are quenched at ``t = 0``.  The full quadratic propagation
then supplies device Green functions, embedding self-energies, and the mixed
real--imaginary kernels entering the Konstantinov--Perel' contour.

It is intentionally separate from the continuum/WBL solver.  Passing this
benchmark proves the contour bookkeeping and the sign convention of the
initial source; it does not prove that a finite lead reproduces a continuum
reservoir without a lead-size and recurrence convergence study.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .continuum_two_time import _two_time_adjoint
from .greens import fermi_dirac
from .initial_correlations import (
    InitialCorrelationResult,
    equilibrium_matsubara_green,
    kbe_initial_correlation_kernel,
)
from .transient import equilibrium_one_body_density


def finite_lead_retarded_self_energy(
    omega: Any,
    *,
    lead_hamiltonian: Any,
    coupling_matrix: Any,
    lead_shift: float = 0.0,
    eta: float = 0.05,
) -> np.ndarray:
    r"""Return the broadened retarded embedding ``Sigma^r(omega)``.

    For a finite lead the retarded embedding is

    ``Sigma^r(w) = V [ (w + i eta) I - (h_lead + shift I) ]^-1 V^dagger``

    The positive ``eta`` is a controlled resolution/broadening parameter for
    the discrete lead levels; it is not a wide-band assumption.  The result
    has shape ``(omega.size, device_dimension, device_dimension)``.
    """

    frequencies = _grid(omega, name="omega")
    if not np.isfinite(lead_shift):
        raise ValueError("lead_shift must be finite.")
    if not np.isfinite(eta) or eta <= 0.0:
        raise ValueError("eta must be finite and positive.")
    lead = _hermitian(lead_hamiltonian, name="lead_hamiltonian")
    coupling = np.asarray(coupling_matrix, dtype=np.complex128)
    if coupling.ndim != 2 or coupling.shape[1] != lead.shape[0]:
        raise ValueError(
            "coupling_matrix must have shape (device_dimension, lead_dimension)."
        )
    if not np.all(np.isfinite(coupling)):
        raise ValueError("coupling_matrix must be finite.")

    energies, states = np.linalg.eigh(lead + float(lead_shift) * np.eye(lead.shape[0]))
    resolvent = np.einsum(
        "ik,wk,kj->wij",
        states,
        1.0 / (frequencies[:, None] + 1j * float(eta) - energies[None, :]),
        states.conj().T,
        optimize=True,
    )
    sigma = np.einsum(
        "ai,wij,bj->wab",
        coupling,
        resolvent,
        coupling.conj(),
        optimize=True,
    )
    return sigma


def finite_lead_spectral_density(
    omega: Any,
    *,
    lead_hamiltonian: Any,
    coupling_matrix: Any,
    lead_shift: float = 0.0,
    eta: float = 0.05,
) -> np.ndarray:
    r"""Return the broadened embedding spectrum ``Gamma(omega)``.

    ``Gamma(w) = i (Sigma^r(w) - Sigma^a(w))`` is obtained from
    :func:`finite_lead_retarded_self_energy`.  The result is Hermitian and
    positive semidefinite up to roundoff at every sampled frequency.
    """

    sigma = finite_lead_retarded_self_energy(
        omega,
        lead_hamiltonian=lead_hamiltonian,
        coupling_matrix=coupling_matrix,
        lead_shift=lead_shift,
        eta=eta,
    )
    gamma = 1j * (sigma - sigma.swapaxes(-1, -2).conj())
    return 0.5 * (gamma + gamma.swapaxes(-1, -2).conj())


def match_wide_band_broadening_from_finite_lead(
    omega: Any,
    *,
    lead_hamiltonian: Any,
    coupling_matrix: Any,
    lead_shift: float = 0.0,
    chemical_potential: float = 0.0,
    temperature: float = 0.35,
    eta: float = 0.05,
) -> np.ndarray:
    r"""Match a constant WBL ``Gamma`` to a finite lead near the Fermi window.

    The matrix is the ``f(1-f)``-weighted average of the microscopic
    ``Gamma(omega)`` returned by :func:`finite_lead_spectral_density`.  This
    is an explicit calibration of a *constant* WBL model; it does not erase
    the finite-lead energy dependence or its recurrences.
    """

    frequencies = _grid(omega, name="omega")
    if not np.isfinite(chemical_potential):
        raise ValueError("chemical_potential must be finite.")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive.")
    window = fermi_dirac(
        frequencies,
        mu=float(chemical_potential),
        temperature=float(temperature),
    )
    window = window * (1.0 - window)
    normalization = float(np.trapezoid(window, frequencies))
    if not np.isfinite(normalization) or normalization <= 0.0:
        raise ValueError("omega must cover a nonzero thermal Fermi window.")
    gamma = finite_lead_spectral_density(
        frequencies,
        lead_hamiltonian=lead_hamiltonian,
        coupling_matrix=coupling_matrix,
        lead_shift=lead_shift,
        eta=eta,
    )
    effective = np.trapezoid(gamma * window[:, None, None], frequencies, axis=0) / normalization
    effective = 0.5 * (effective + effective.conj().T)
    # Roundoff can produce tiny negative eigenvalues although every broadened
    # Gamma(w) is positive semidefinite.  Project only that numerical noise.
    eigenvalues, eigenvectors = np.linalg.eigh(effective)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    return (eigenvectors * eigenvalues[None, :]) @ eigenvectors.conj().T


def _grid(value: Any, *, name: str, minimum: int = 2) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or result.size < minimum:
        raise ValueError(f"{name} must be a one-dimensional grid with at least {minimum} points.")
    if not np.all(np.isfinite(result)) or np.any(np.diff(result) <= 0.0):
        raise ValueError(f"{name} must be finite and strictly increasing.")
    return result


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


def _unitary_stack(hamiltonian: np.ndarray, time: np.ndarray) -> np.ndarray:
    energies, states = np.linalg.eigh(hamiltonian)
    return np.einsum(
        "ik,tk,kj->tij",
        states,
        np.exp(-1j * time[:, None] * energies[None, :]),
        states.conj().T,
        optimize=True,
    )


def _stable_thermal_mixed_stack(
    hamiltonian: np.ndarray,
    time: np.ndarray,
    *,
    chemical_potential: float,
    temperature: float,
    occupied: bool,
) -> np.ndarray:
    """Evaluate the bounded KMS products on the vertical branch directly.

    Computing ``exp(-h*tau) @ (1-f[h])`` or ``f[h] @ exp(h*tau)`` as two
    separate matrices is ill-conditioned when a continuum quadrature spans
    many thermal energies.  Both factors commute, so their product is formed
    mode by mode in the eigenbasis.  The logarithmic Fermi factors keep the
    cancellation bounded at the endpoint ``tau=beta``.
    """

    energies, states = np.linalg.eigh(hamiltonian)
    scaled = (energies - float(chemical_potential)) / float(temperature)
    if occupied:
        # log f = -log(1 + exp((e-mu)/T))
        log_fermi = -np.logaddexp(0.0, scaled)
        exponents = log_fermi[None, :] + time[:, None] * energies[None, :]
    else:
        # log(1-f) = -log(1 + exp(-(e-mu)/T))
        log_complement = -np.logaddexp(0.0, -scaled)
        exponents = log_complement[None, :] - time[:, None] * energies[None, :]
    factors = np.exp(exponents)
    return np.einsum(
        "ik,tk,kj->tij",
        states,
        factors,
        states.conj().T,
        optimize=True,
    )


def _block_hamiltonian(
    device: np.ndarray,
    leads: tuple[np.ndarray, ...],
    couplings: tuple[np.ndarray, ...],
    shifts: np.ndarray,
) -> np.ndarray:
    dimension = device.shape[0]
    total = dimension + sum(lead.shape[0] for lead in leads)
    result = np.zeros((total, total), dtype=np.complex128)
    result[:dimension, :dimension] = device
    offset = dimension
    for lead, coupling, shift in zip(leads, couplings, shifts):
        size = lead.shape[0]
        result[offset : offset + size, offset : offset + size] = lead + shift * np.eye(size)
        result[:dimension, offset : offset + size] = coupling
        result[offset : offset + size, :dimension] = coupling.conj().T
        offset += size
    return result


@dataclass(frozen=True)
class FiniteLeadPartitionFreeResult:
    """Exact finite-lead device kernels and the microscopic IC source."""

    time: np.ndarray
    imaginary_time: np.ndarray
    device_dimension: int
    initial_hamiltonian: np.ndarray
    final_hamiltonian: np.ndarray
    initial_device_hamiltonian: np.ndarray
    final_device_hamiltonian: np.ndarray
    lead_hamiltonians: tuple[np.ndarray, ...]
    coupling_matrices: tuple[np.ndarray, ...]
    lead_shifts: np.ndarray
    initial_density: np.ndarray
    retarded: np.ndarray
    advanced: np.ndarray
    lesser: np.ndarray
    greater: np.ndarray
    self_energy_retarded: np.ndarray
    self_energy_advanced: np.ndarray
    self_energy_lesser: np.ndarray
    self_energy_greater: np.ndarray
    lead_self_energy_retarded: tuple[np.ndarray, ...]
    lead_self_energy_advanced: tuple[np.ndarray, ...]
    lead_self_energy_lesser: tuple[np.ndarray, ...]
    lead_self_energy_greater: tuple[np.ndarray, ...]
    green_mixed: np.ndarray
    self_energy_mixed: np.ndarray
    green_matsubara: np.ndarray
    self_energy_matsubara: np.ndarray
    initial_correlation: InitialCorrelationResult

    @property
    def density_matrices(self) -> np.ndarray:
        diagonal = self.lesser[np.arange(self.time.size), np.arange(self.time.size)]
        density = -1j * diagonal
        return 0.5 * (density + density.swapaxes(-1, -2).conj())

    @property
    def continuity_initial_source(self) -> np.ndarray:
        """Source to subtract from the package residual convention.

        ``two_time_kbe_continuity_balance`` reports
        ``d rho/dt - coherent_rate - collision_rate``.  With the mixed-kernel
        convention in :func:`kbe_initial_correlation_kernel`, the exact source
        in this residual is ``-(I^IC + I^{IC,dagger})``.
        """

        return -self.initial_correlation.density_source

    @property
    def spectral_identity_error(self) -> float:
        residual = self.greater - self.lesser - self.retarded + self.advanced
        return float(np.max(np.abs(residual)))


@dataclass(frozen=True)
class FiniteLeadPartitionedResult:
    """Exact quadratic transient after switching on initially disconnected leads."""

    time: np.ndarray
    device_dimension: int
    initial_device_hamiltonian: np.ndarray
    final_device_hamiltonian: np.ndarray
    lead_hamiltonians: tuple[np.ndarray, ...]
    coupling_matrices: tuple[np.ndarray, ...]
    lead_shifts: np.ndarray
    initial_density: np.ndarray
    retarded: np.ndarray
    advanced: np.ndarray
    lesser: np.ndarray
    greater: np.ndarray
    self_energy_retarded: np.ndarray
    self_energy_advanced: np.ndarray
    self_energy_lesser: np.ndarray
    self_energy_greater: np.ndarray
    lead_self_energy_retarded: tuple[np.ndarray, ...]
    lead_self_energy_advanced: tuple[np.ndarray, ...]
    lead_self_energy_lesser: tuple[np.ndarray, ...]
    lead_self_energy_greater: tuple[np.ndarray, ...]

    @property
    def density_matrices(self) -> np.ndarray:
        diagonal = self.lesser[np.arange(self.time.size), np.arange(self.time.size)]
        density = -1j * diagonal
        return 0.5 * (density + density.swapaxes(-1, -2).conj())

    @property
    def spectral_identity_error(self) -> float:
        residual = self.greater - self.lesser - self.retarded + self.advanced
        return float(np.max(np.abs(residual)))


def partitioned_finite_lead_two_time(
    time: Any,
    *,
    initial_device_hamiltonian: Any,
    final_device_hamiltonian: Any | None = None,
    lead_hamiltonians: Sequence[Any],
    coupling_matrices: Sequence[Any],
    lead_shifts: Sequence[float] | np.ndarray | None = None,
    chemical_potential: float = 0.0,
    temperature: float,
) -> FiniteLeadPartitionedResult:
    r"""Solve a quadratic partitioned contact quench exactly.

    Before ``t=0`` the device and every lead are disconnected and separately
    equilibrated.  At ``t=0`` the couplings, a device quench, and constant lead
    shifts are switched on.  The returned lead-resolved kernels are the exact
    finite-lead memory functions for this non-equilibrium transient; unlike
    :func:`partition_free_finite_lead_two_time`, no contacted initial
    equilibrium or vertical-branch source is included.
    """

    times = _grid(time, name="time")
    if times[0] < 0.0:
        raise ValueError("time must start at or after the quench.")
    if temperature <= 0.0 or not np.isfinite(temperature):
        raise ValueError("temperature must be positive.")
    mu = float(chemical_potential)
    if not np.isfinite(mu):
        raise ValueError("chemical_potential must be finite.")
    h_initial_device = _hermitian(initial_device_hamiltonian, name="initial_device_hamiltonian")
    dimension = h_initial_device.shape[0]
    h_final_device = (
        h_initial_device.copy()
        if final_device_hamiltonian is None
        else _hermitian(final_device_hamiltonian, name="final_device_hamiltonian", dimension=dimension)
    )
    leads = tuple(_hermitian(value, name=f"lead_hamiltonians[{index}]") for index, value in enumerate(lead_hamiltonians))
    if not leads:
        raise ValueError("at least one lead is required.")
    couplings_list: list[np.ndarray] = []
    for index, (lead, value) in enumerate(zip(leads, coupling_matrices)):
        coupling = np.asarray(value, dtype=np.complex128)
        if coupling.shape != (dimension, lead.shape[0]):
            raise ValueError(f"coupling_matrices[{index}] must have shape {(dimension, lead.shape[0])}.")
        if not np.all(np.isfinite(coupling)):
            raise ValueError("coupling_matrices must be finite.")
        couplings_list.append(coupling)
    if len(couplings_list) != len(leads) or len(coupling_matrices) != len(leads):
        raise ValueError("one coupling matrix is required per lead.")
    couplings = tuple(couplings_list)
    shifts = np.zeros(len(leads), dtype=float) if lead_shifts is None else np.asarray(lead_shifts, dtype=float)
    if shifts.shape != (len(leads),) or not np.all(np.isfinite(shifts)):
        raise ValueError("lead_shifts must contain one finite value per lead.")

    full_dimension = dimension + sum(lead.shape[0] for lead in leads)
    initial_density = np.zeros((full_dimension, full_dimension), dtype=np.complex128)
    initial_density[:dimension, :dimension] = equilibrium_one_body_density(
        h_initial_device, mu=mu, temperature=float(temperature)
    )
    offset = dimension
    for lead in leads:
        size = lead.shape[0]
        initial_density[offset : offset + size, offset : offset + size] = equilibrium_one_body_density(
            lead, mu=mu, temperature=float(temperature)
        )
        offset += size

    final_full = _block_hamiltonian(h_final_device, leads, couplings, shifts)
    final_evolution = _unitary_stack(final_full, times)
    projector = np.zeros((dimension, full_dimension), dtype=np.complex128)
    projector[:, :dimension] = np.eye(dimension)
    identity = np.eye(dimension, dtype=np.complex128)
    retarded = np.zeros((times.size, times.size, dimension, dimension), dtype=np.complex128)
    lesser = np.zeros_like(retarded)
    greater = np.zeros_like(retarded)
    complement = np.eye(full_dimension, dtype=np.complex128) - initial_density
    for left, left_time in enumerate(times):
        for right, right_time in enumerate(times):
            lesser[left, right] = 1j * projector @ final_evolution[left] @ initial_density @ final_evolution[right].conj().T @ projector.conj().T
            greater[left, right] = -1j * projector @ final_evolution[left] @ complement @ final_evolution[right].conj().T @ projector.conj().T
            lag = left_time - right_time
            if lag > 1e-14:
                retarded[left, right] = -1j * projector @ _unitary_stack(final_full, np.array([lag]))[0] @ projector.conj().T
            elif abs(lag) <= 1e-14:
                retarded[left, right] = -0.5j * identity
    advanced = _two_time_adjoint(retarded)

    total_r = np.zeros_like(retarded)
    total_l = np.zeros_like(retarded)
    total_g = np.zeros_like(retarded)
    lead_r = tuple(np.zeros_like(retarded) for _ in leads)
    lead_l = tuple(np.zeros_like(retarded) for _ in leads)
    lead_g = tuple(np.zeros_like(retarded) for _ in leads)
    for lead_index, (lead, coupling, shift) in enumerate(zip(leads, couplings, shifts)):
        size = lead.shape[0]
        lead_final = lead + shift * np.eye(size)
        lead_evolution = _unitary_stack(lead_final, times)
        lead_density = equilibrium_one_body_density(lead, mu=mu, temperature=float(temperature))
        lead_complement = np.eye(size, dtype=np.complex128) - lead_density
        sigma_r = lead_r[lead_index]
        sigma_l = lead_l[lead_index]
        sigma_g = lead_g[lead_index]
        for left, left_time in enumerate(times):
            for right, right_time in enumerate(times):
                sigma_l[left, right] = 1j * coupling @ lead_evolution[left] @ lead_density @ lead_evolution[right].conj().T @ coupling.conj().T
                sigma_g[left, right] = -1j * coupling @ lead_evolution[left] @ lead_complement @ lead_evolution[right].conj().T @ coupling.conj().T
                lag = left_time - right_time
                if lag > 1e-14:
                    sigma_r[left, right] = -1j * coupling @ _unitary_stack(lead_final, np.array([lag]))[0] @ coupling.conj().T
                elif abs(lag) <= 1e-14:
                    sigma_r[left, right] = -0.5j * coupling @ coupling.conj().T
        total_r += sigma_r
        total_l += sigma_l
        total_g += sigma_g
    return FiniteLeadPartitionedResult(
        time=times.copy(),
        device_dimension=dimension,
        initial_device_hamiltonian=h_initial_device,
        final_device_hamiltonian=h_final_device,
        lead_hamiltonians=leads,
        coupling_matrices=couplings,
        lead_shifts=shifts.copy(),
        initial_density=initial_density,
        retarded=retarded,
        advanced=advanced,
        lesser=lesser,
        greater=greater,
        self_energy_retarded=total_r,
        self_energy_advanced=_two_time_adjoint(total_r),
        self_energy_lesser=total_l,
        self_energy_greater=total_g,
        lead_self_energy_retarded=tuple(value.copy() for value in lead_r),
        lead_self_energy_advanced=tuple(_two_time_adjoint(value) for value in lead_r),
        lead_self_energy_lesser=tuple(value.copy() for value in lead_l),
        lead_self_energy_greater=tuple(value.copy() for value in lead_g),
    )


def partition_free_finite_lead_two_time(
    time: Any,
    imaginary_time: Any,
    *,
    initial_device_hamiltonian: Any,
    final_device_hamiltonian: Any | None = None,
    lead_hamiltonians: Sequence[Any],
    coupling_matrices: Sequence[Any],
    lead_shifts: Sequence[float] | np.ndarray | None = None,
    chemical_potential: float = 0.0,
    temperature: float,
) -> FiniteLeadPartitionFreeResult:
    r"""Build an exact finite-lead partition-free two-time benchmark.

    The initial full Hamiltonian contains the device and all leads with zero
    lead shifts.  The final Hamiltonian applies ``final_device_hamiltonian``
    and the constant ``lead_shifts``.  The initial state is the grand-canonical
    equilibrium of the *coupled* full Hamiltonian.  The mixed kernels are
    evaluated as

    ``Sigma^rceil = i V U_L^f(t) f_L exp(h_L tau) V^dagger`` and
    ``G^lceil = -i P exp(-H_0 tau) (1-F_0) U_f(t)^dagger P^dagger``.

    The finite lead is an exact microscopic oracle; recurrences are physical
    and should be controlled by increasing lead size before interpreting a
    continuum limit.
    """

    times = _grid(time, name="time")
    imaginary = _grid(imaginary_time, name="imaginary_time")
    if times[0] < 0.0:
        raise ValueError("time must start at or after the quench.")
    if imaginary[0] < 0.0:
        raise ValueError("imaginary_time must be nonnegative.")
    if temperature <= 0.0 or not np.isfinite(temperature):
        raise ValueError("temperature must be positive for a finite beta branch.")
    mu = float(chemical_potential)
    if not np.isfinite(mu):
        raise ValueError("chemical_potential must be finite.")

    h_initial_device = _hermitian(initial_device_hamiltonian, name="initial_device_hamiltonian")
    dimension = h_initial_device.shape[0]
    h_final_device = (
        h_initial_device.copy()
        if final_device_hamiltonian is None
        else _hermitian(final_device_hamiltonian, name="final_device_hamiltonian", dimension=dimension)
    )
    leads = tuple(_hermitian(value, name=f"lead_hamiltonians[{index}]") for index, value in enumerate(lead_hamiltonians))
    if not leads:
        raise ValueError("at least one lead is required.")
    couplings_list: list[np.ndarray] = []
    for index, (lead, value) in enumerate(zip(leads, coupling_matrices)):
        coupling = np.asarray(value, dtype=np.complex128)
        if coupling.shape != (dimension, lead.shape[0]):
            raise ValueError(
                f"coupling_matrices[{index}] must have shape {(dimension, lead.shape[0])}."
            )
        if not np.all(np.isfinite(coupling)):
            raise ValueError("coupling_matrices must be finite.")
        couplings_list.append(coupling)
    if len(couplings_list) != len(leads) or len(coupling_matrices) != len(leads):
        raise ValueError("one coupling matrix is required per lead.")
    couplings = tuple(couplings_list)
    shifts = np.zeros(len(leads), dtype=float) if lead_shifts is None else np.asarray(lead_shifts, dtype=float)
    if shifts.shape != (len(leads),) or not np.all(np.isfinite(shifts)):
        raise ValueError("lead_shifts must contain one finite value per lead.")

    initial_full = _block_hamiltonian(h_initial_device, leads, couplings, np.zeros(len(leads)))
    final_full = _block_hamiltonian(h_final_device, leads, couplings, shifts)
    full_dimension = initial_full.shape[0]
    initial_density = equilibrium_one_body_density(
        initial_full,
        mu=mu,
        temperature=float(temperature),
    )
    final_evolution = _unitary_stack(final_full, times)
    # The imaginary branch is evaluated in a thermal-combined form below;
    # keeping this separate stack is unnecessary and can overflow for broad
    # continuum quadratures.
    projector = np.zeros((dimension, full_dimension), dtype=np.complex128)
    projector[:, :dimension] = np.eye(dimension)

    retarded = np.zeros((times.size, times.size, dimension, dimension), dtype=np.complex128)
    lesser = np.zeros_like(retarded)
    greater = np.zeros_like(retarded)
    identity = np.eye(dimension, dtype=np.complex128)
    complement = np.eye(full_dimension, dtype=np.complex128) - initial_density
    for left, left_time in enumerate(times):
        for right, right_time in enumerate(times):
            lesser[left, right] = 1j * projector @ final_evolution[left] @ initial_density @ final_evolution[right].conj().T @ projector.conj().T
            greater[left, right] = -1j * projector @ final_evolution[left] @ complement @ final_evolution[right].conj().T @ projector.conj().T
            lag = left_time - right_time
            if lag > 1e-14:
                retarded[left, right] = -1j * projector @ _unitary_stack(final_full, np.array([lag]))[0] @ projector.conj().T
            elif abs(lag) <= 1e-14:
                retarded[left, right] = -0.5j * identity
    advanced = _two_time_adjoint(retarded)

    self_energy_retarded = np.zeros_like(retarded)
    self_energy_lesser = np.zeros_like(retarded)
    self_energy_greater = np.zeros_like(retarded)
    lead_sigma_retarded = tuple(np.zeros_like(retarded) for _ in leads)
    lead_sigma_lesser = tuple(np.zeros_like(retarded) for _ in leads)
    lead_sigma_greater = tuple(np.zeros_like(retarded) for _ in leads)
    sigma_mixed = np.zeros((times.size, imaginary.size, dimension, dimension), dtype=np.complex128)
    green_mixed = np.zeros((imaginary.size, times.size, dimension, dimension), dtype=np.complex128)
    green_matsubara_full = equilibrium_matsubara_green(
        initial_full,
        imaginary,
        chemical_potential=mu,
        temperature=float(temperature),
    )
    green_matsubara = np.einsum(
        "ai,xyij,bj->xyab",
        projector,
        green_matsubara_full,
        projector.conj(),
        optimize=True,
    )
    self_energy_matsubara = np.zeros_like(green_matsubara)
    for lead, coupling in zip(leads, couplings):
        lead_matsubara = equilibrium_matsubara_green(
            lead,
            imaginary,
            chemical_potential=mu,
            temperature=float(temperature),
        )
        self_energy_matsubara += np.einsum(
            "ai,xyij,bj->xyab",
            coupling,
            lead_matsubara,
            coupling.conj(),
            optimize=True,
        )
    for lead_index, (lead, coupling, shift) in enumerate(zip(leads, couplings, shifts)):
        lead_size = lead.shape[0]
        lead_final = lead + shift * np.eye(lead_size)
        lead_evolution = _unitary_stack(lead_final, times)
        lead_density = equilibrium_one_body_density(lead, mu=mu, temperature=float(temperature))
        lead_complement = np.eye(lead_size, dtype=np.complex128) - lead_density
        sigma_r = lead_sigma_retarded[lead_index]
        sigma_l = lead_sigma_lesser[lead_index]
        sigma_g = lead_sigma_greater[lead_index]
        for left, left_time in enumerate(times):
            for right, right_time in enumerate(times):
                sigma_l[left, right] = 1j * coupling @ lead_evolution[left] @ lead_density @ lead_evolution[right].conj().T @ coupling.conj().T
                sigma_g[left, right] = -1j * coupling @ lead_evolution[left] @ lead_complement @ lead_evolution[right].conj().T @ coupling.conj().T
                lag = left_time - right_time
                if lag > 1e-14:
                    sigma_r[left, right] = -1j * coupling @ _unitary_stack(lead_final, np.array([lag]))[0] @ coupling.conj().T
                elif abs(lag) <= 1e-14:
                    sigma_r[left, right] = -0.5j * coupling @ coupling.conj().T
        self_energy_retarded += sigma_r
        self_energy_lesser += sigma_l
        self_energy_greater += sigma_g

    green_imaginary_complement = _stable_thermal_mixed_stack(
        initial_full,
        imaginary,
        chemical_potential=mu,
        temperature=float(temperature),
        occupied=False,
    )
    for imaginary_index in range(imaginary.size):
        for right in range(times.size):
            green_mixed[imaginary_index, right] = (
                -1j
                * projector
                @ green_imaginary_complement[imaginary_index]
                @ final_evolution[right].conj().T
                @ projector.conj().T
            )

    # Sigma^rceil needs exp(+h_lead tau) on the vertical branch.  The helper
    # with ``-lead`` supplies this positive exponential directly.
    for lead, coupling, shift in zip(leads, couplings, shifts):
        lead_size = lead.shape[0]
        lead_final = lead + shift * np.eye(lead_size)
        lead_evolution = _unitary_stack(lead_final, times)
        lead_imaginary_occupied = _stable_thermal_mixed_stack(
            lead,
            imaginary,
            chemical_potential=mu,
            temperature=float(temperature),
            occupied=True,
        )
        for left in range(times.size):
            for imaginary_index in range(imaginary.size):
                sigma_mixed[left, imaginary_index] += (
                    1j
                    * coupling
                    @ lead_evolution[left]
                    @ lead_imaginary_occupied[imaginary_index]
                    @ coupling.conj().T
                )

    initial_correlation = kbe_initial_correlation_kernel(
        times,
        imaginary,
        self_energy_mixed=sigma_mixed,
        green_mixed=green_mixed,
    )
    return FiniteLeadPartitionFreeResult(
        time=times.copy(),
        imaginary_time=imaginary.copy(),
        device_dimension=dimension,
        initial_hamiltonian=initial_full,
        final_hamiltonian=final_full,
        initial_device_hamiltonian=h_initial_device,
        final_device_hamiltonian=h_final_device,
        lead_hamiltonians=leads,
        coupling_matrices=couplings,
        lead_shifts=shifts.copy(),
        initial_density=initial_density,
        retarded=retarded,
        advanced=advanced,
        lesser=lesser,
        greater=greater,
        self_energy_retarded=self_energy_retarded,
        self_energy_advanced=_two_time_adjoint(self_energy_retarded),
        self_energy_lesser=self_energy_lesser,
        self_energy_greater=self_energy_greater,
        lead_self_energy_retarded=tuple(value.copy() for value in lead_sigma_retarded),
        lead_self_energy_advanced=tuple(_two_time_adjoint(value) for value in lead_sigma_retarded),
        lead_self_energy_lesser=tuple(value.copy() for value in lead_sigma_lesser),
        lead_self_energy_greater=tuple(value.copy() for value in lead_sigma_greater),
        green_mixed=green_mixed,
        self_energy_mixed=sigma_mixed,
        green_matsubara=green_matsubara,
        self_energy_matsubara=self_energy_matsubara,
        initial_correlation=initial_correlation,
    )


def finite_lead_current_current_correlations(
    result: FiniteLeadPartitionFreeResult | FiniteLeadPartitionedResult,
    lead_index: int,
    *,
    spin_observables: Mapping[str, Any] | None = None,
    symmetrized: bool = True,
) -> dict[str, dict[str, np.ndarray]]:
    r"""Return exact quadratic lead-current charge/spin correlations.

    The finite contacted system is known explicitly, so the lead current
    one-body vertex is constructed from

    ``J_O = i [P_lead O_lead P_lead, H_final]``.

    The result is the connected Wick bubble built from the full finite-system
    ``G^<`` and ``G^>``.  It is an exact noninteracting benchmark, useful for
    validating a KBE current-noise implementation.  Applying the same bubble
    to an interacting device is only the no-vertex approximation; this helper
    therefore keeps the microscopic finite-lead oracle explicit.
    """

    if not isinstance(result, (FiniteLeadPartitionFreeResult, FiniteLeadPartitionedResult)):
        raise TypeError("result must be a finite-lead partition-free or partitioned result.")
    if not isinstance(lead_index, (int, np.integer)):
        raise TypeError("lead_index must be an integer.")
    if lead_index < 0 or lead_index >= len(result.lead_hamiltonians):
        raise IndexError("lead_index is outside the finite-lead list.")
    # Only the partition-free result stores the contacted Hamiltonian; the
    # partitioned one keeps its blocks, so rebuild the post-quench matrix from
    # them rather than crashing on a missing attribute.
    if isinstance(result, FiniteLeadPartitionFreeResult):
        full_hamiltonian = np.asarray(result.final_hamiltonian, dtype=np.complex128)
    else:
        full_hamiltonian = _block_hamiltonian(
            np.asarray(result.final_device_hamiltonian, dtype=np.complex128),
            tuple(result.lead_hamiltonians),
            tuple(result.coupling_matrices),
            np.asarray(result.lead_shifts, dtype=float),
        )
    full_dimension = full_hamiltonian.shape[0]
    device_dimension = result.initial_device_hamiltonian.shape[0]
    offset = device_dimension + sum(lead.shape[0] for lead in result.lead_hamiltonians[:lead_index])
    lead_dimension = result.lead_hamiltonians[lead_index].shape[0]
    projector = np.zeros((full_dimension, full_dimension), dtype=np.complex128)
    projector[offset : offset + lead_dimension, offset : offset + lead_dimension] = np.eye(lead_dimension)
    base_vertex = 1j * (projector @ full_hamiltonian - full_hamiltonian @ projector)
    observables: dict[str, np.ndarray] = {"charge": base_vertex}
    if spin_observables is not None:
        if not isinstance(spin_observables, Mapping):
            raise TypeError("spin_observables must be a mapping of local Hermitian matrices.")
        for name, local in spin_observables.items():
            if not isinstance(name, str) or not name:
                raise ValueError("spin observable names must be non-empty strings.")
            local_matrix = np.asarray(local, dtype=np.complex128)
            if local_matrix.shape != (lead_dimension, lead_dimension):
                raise ValueError("each spin observable must match the lead dimension.")
            if not np.allclose(local_matrix, local_matrix.conj().T, atol=1e-12, rtol=1e-12):
                raise ValueError("spin observables must be Hermitian.")
            weighted_projector = np.zeros_like(projector)
            weighted_projector[offset : offset + lead_dimension, offset : offset + lead_dimension] = local_matrix
            observables[name] = 1j * (weighted_projector @ full_hamiltonian - full_hamiltonian @ weighted_projector)

    evolution = _unitary_stack(full_hamiltonian, np.asarray(result.time, dtype=float))
    initial_density = np.asarray(result.initial_density, dtype=np.complex128)
    complement = np.eye(full_dimension, dtype=np.complex128) - initial_density
    green_lesser = np.einsum(
        "tac,cd,sbd->tsab",
        evolution,
        initial_density,
        evolution.conj(),
        optimize=True,
    ) * 1j
    green_greater = np.einsum(
        "tac,cd,sbd->tsab",
        evolution,
        complement,
        evolution.conj(),
        optimize=True,
    ) * (-1j)
    from .kadanoff_baym import two_time_one_body_correlations

    return two_time_one_body_correlations(
        result.time,
        green_lesser=green_lesser,
        green_greater=green_greater,
        observables=observables,
        symmetrized=symmetrized,
    )


__all__ = [
    "FiniteLeadPartitionFreeResult",
    "FiniteLeadPartitionedResult",
    "finite_lead_retarded_self_energy",
    "finite_lead_spectral_density",
    "finite_lead_current_current_correlations",
    "match_wide_band_broadening_from_finite_lead",
    "partition_free_finite_lead_two_time",
    "partitioned_finite_lead_two_time",
]
