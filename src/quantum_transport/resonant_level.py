"""Analytic transient oracles for a wide-band resonant level.

The model contains one spinless level of energy ``epsilon_d`` coupled at
``t=0`` to noninteracting reservoirs with constant broadenings ``Gamma_a``.
The reservoirs remain in Fermi states ``f_a(energy)``.  In the wide-band
limit the exact spectral amplitude is

``A(E,t) = [1-exp((i(E-epsilon_d)-Gamma/2)t)] / (E-epsilon_d+i Gamma/2)``.

The module also implements a partition-free sudden bias quench: the level is
already contacted and in global equilibrium before constant lead-energy
shifts are applied.  These closed single-level cases are analytic oracles,
not a general matrix-valued time-dependent continuum self-energy solver.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .greens import fermi_dirac


ComplexArray = NDArray[np.complex128]
RealArray = NDArray[np.float64]


def _strict_grid(
    value: RealArray,
    *,
    name: str,
    minimum_size: int,
) -> RealArray:
    grid = np.asarray(value, dtype=float)
    if grid.ndim != 1 or grid.size < minimum_size:
        raise ValueError(
            f"{name} must be a one-dimensional grid with at least "
            f"{minimum_size} point(s)."
        )
    if not np.all(np.isfinite(grid)):
        raise ValueError(f"{name} must contain finite values.")
    if grid.size > 1 and np.any(np.diff(grid) <= 0):
        raise ValueError(f"{name} must be strictly increasing.")
    return grid


def _lead_parameters(
    broadening: RealArray,
    chemical_potential: RealArray,
    temperature: float | RealArray,
) -> tuple[RealArray, RealArray, RealArray]:
    gamma = np.asarray(broadening, dtype=float)
    mu = np.asarray(chemical_potential, dtype=float)
    if gamma.ndim != 1 or gamma.size == 0:
        raise ValueError("broadening must be a nonempty vector.")
    if mu.shape != gamma.shape:
        raise ValueError(
            "chemical_potential must match broadening."
        )
    if not np.all(np.isfinite(gamma)) or np.any(gamma <= 0):
        raise ValueError("every lead broadening must be positive.")
    if not np.all(np.isfinite(mu)):
        raise ValueError("chemical potentials must be finite.")
    thermal = np.asarray(temperature, dtype=float)
    if thermal.ndim == 0:
        thermal = np.full(gamma.shape, float(thermal))
    if thermal.shape != gamma.shape:
        raise ValueError("temperature must be scalar or one per lead.")
    if not np.all(np.isfinite(thermal)) or np.any(thermal < 0):
        raise ValueError("temperatures cannot be negative.")
    return gamma, mu, thermal


def resonant_level_spectral_amplitude(
    energy: RealArray,
    time: RealArray,
    *,
    level_energy: float,
    total_broadening: float,
) -> ComplexArray:
    r"""Return the exact wide-band contact-quench amplitude ``A(E,t)``.

    .. math::

       A(E,t)=\frac{1-e^{[i(E-\epsilon_d)-\Gamma/2]t}}
       {E-\epsilon_d+i\Gamma/2}.
    """
    energies = _strict_grid(
        energy, name="energy", minimum_size=2
    )
    times = _strict_grid(time, name="time", minimum_size=1)
    if times[0] < 0:
        raise ValueError("time is measured from the quench and must be >= 0.")
    gamma = float(total_broadening)
    epsilon = float(level_energy)
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("total_broadening must be positive.")
    if not np.isfinite(epsilon):
        raise ValueError("level_energy must be finite.")
    detuning = energies - epsilon
    exponent = (
        1j * detuning[None, :] - 0.5 * gamma
    ) * times[:, None]
    denominator = detuning + 0.5j * gamma
    return (1.0 - np.exp(exponent)) / denominator[None, :]


@dataclass(frozen=True)
class ResonantLevelTransient:
    """Occupation and lead currents after a partitioned contact quench.

    ``current_into_level[t, alpha]`` is positive when lead ``alpha`` injects
    particles into the resonant level.  Its lead sum equals ``dn/dt``.
    """

    time: RealArray
    occupation: RealArray
    current_into_level: RealArray
    occupation_rate: RealArray
    lead_broadening: RealArray
    chemical_potential: RealArray
    temperature: RealArray
    level_energy: float
    initial_occupation: float

    @property
    def net_current_into_level(self) -> RealArray:
        """Return the sum of all lead currents."""
        return np.sum(self.current_into_level, axis=1)


@dataclass(frozen=True)
class ResonantLevelSteadyState:
    """Zero-temperature wide-band steady state of one resonant level."""

    occupation: float
    current_into_level: RealArray


@dataclass(frozen=True)
class PartitionFreeResonantLevelTransient:
    """Bias-quench transient from an initially contacted equilibrium state."""

    time: RealArray
    occupation: RealArray
    current_into_level: RealArray
    occupation_rate: RealArray
    lead_broadening: RealArray
    bias_shift: RealArray
    initial_chemical_potential: float
    temperature: float
    level_energy: float

    @property
    def net_current_into_level(self) -> RealArray:
        """Return the displacement current ``sum_alpha I_alpha``."""
        return np.sum(self.current_into_level, axis=1)

    @property
    def final_chemical_potential(self) -> RealArray:
        """Return ``mu_alpha=mu_0+Delta_alpha`` after the step."""
        return self.initial_chemical_potential + self.bias_shift


def wide_band_resonant_level_quench(
    time: RealArray,
    energy: RealArray,
    *,
    level_energy: float,
    broadening: RealArray,
    chemical_potential: RealArray,
    temperature: float | RealArray = 0.0,
    initial_occupation: float = 0.0,
    max_memory_bytes: int = 256 * 1024**2,
) -> ResonantLevelTransient:
    r"""Solve an exact partitioned wide-band resonant-level contact quench.

    With ``Gamma=sum(Gamma_alpha)`` and initial dot occupation ``n_0``,

    .. math::

       n(t)=n_0e^{-\Gamma t}
       +\sum_\alpha\Gamma_\alpha\int\frac{dE}{2\pi}
       f_\alpha(E)|A(E,t)|^2,

    .. math::

       I_\alpha(t)=-\Gamma_\alpha n(t)
       -2\Gamma_\alpha\int\frac{dE}{2\pi}
       f_\alpha(E)\,\mathrm{Im}\,A(E,t).

    Currents are particle currents from each lead into the level.  Energy
    integrals use ``numpy.trapezoid`` on the caller-supplied grid.
    """
    times = _strict_grid(time, name="time", minimum_size=1)
    energies = _strict_grid(
        energy, name="energy", minimum_size=2
    )
    if times[0] < 0:
        raise ValueError("time is measured from the quench and must be >= 0.")
    gamma, mu, thermal = _lead_parameters(
        broadening, chemical_potential, temperature
    )
    initial = float(initial_occupation)
    if not np.isfinite(initial) or not 0.0 <= initial <= 1.0:
        raise ValueError("initial_occupation must lie in [0, 1].")
    if max_memory_bytes <= 0:
        raise ValueError("max_memory_bytes must be positive.")
    total_gamma = float(np.sum(gamma))
    fillings = np.array(
        [
            fermi_dirac(
                energies,
                mu=float(mu_value),
                temperature=float(temp_value),
            )
            for mu_value, temp_value in zip(mu, thermal)
        ],
        dtype=float,
    )
    weighted_filling = np.sum(gamma[:, None] * fillings, axis=0)
    occupation = np.empty(times.size, dtype=float)
    currents = np.empty((times.size, gamma.size), dtype=float)
    bytes_per_time = max(1, energies.size * 64)
    if max_memory_bytes < bytes_per_time:
        raise MemoryError(
            "resonant-level time-chunk workspace estimate "
            f"{bytes_per_time / 1024**2:.1f} MiB exceeds limit "
            f"{max_memory_bytes / 1024**2:.1f} MiB."
        )
    chunk_size = max(
        1,
        min(times.size, int(max_memory_bytes // bytes_per_time)),
    )
    for start in range(0, times.size, chunk_size):
        stop = min(start + chunk_size, times.size)
        chunk_time = times[start:stop]
        amplitude = resonant_level_spectral_amplitude(
            energies,
            chunk_time,
            level_energy=level_energy,
            total_broadening=total_gamma,
        )
        injection = np.trapezoid(
            weighted_filling[None, :] * np.abs(amplitude) ** 2,
            energies,
            axis=1,
        ) / (2.0 * np.pi)
        chunk_occupation = (
            initial * np.exp(-total_gamma * chunk_time)
            + injection
        )
        lead_integrals = np.array(
            [
                np.trapezoid(
                    filling[None, :] * amplitude.imag,
                    energies,
                    axis=1,
                )
                / (2.0 * np.pi)
                for filling in fillings
            ]
        ).T
        occupation[start:stop] = chunk_occupation
        currents[start:stop] = (
            -chunk_occupation[:, None] * gamma[None, :]
            - 2.0 * lead_integrals * gamma[None, :]
        )
    occupation_rate = np.sum(currents, axis=1)
    return ResonantLevelTransient(
        time=times,
        occupation=np.asarray(occupation, dtype=float),
        current_into_level=np.asarray(currents, dtype=float),
        occupation_rate=np.asarray(occupation_rate, dtype=float),
        lead_broadening=gamma,
        chemical_potential=mu,
        temperature=thermal,
        level_energy=float(level_energy),
        initial_occupation=initial,
    )


def zero_temperature_resonant_level_steady_state(
    *,
    level_energy: float,
    broadening: RealArray,
    chemical_potential: RealArray,
) -> ResonantLevelSteadyState:
    r"""Return the closed zero-temperature wide-band steady state.

    Defining

    ``F_alpha = 1/2 + atan(2(mu_alpha-epsilon_d)/Gamma)/pi``,

    the occupation is ``sum(Gamma_alpha F_alpha)/Gamma`` and the particle
    current from lead ``alpha`` is ``Gamma_alpha(F_alpha-n)``.
    """
    gamma, mu, _ = _lead_parameters(
        broadening, chemical_potential, 0.0
    )
    epsilon = float(level_energy)
    if not np.isfinite(epsilon):
        raise ValueError("level_energy must be finite.")
    total_gamma = float(np.sum(gamma))
    reservoir_weights = 0.5 + np.arctan(
        2.0 * (mu - epsilon) / total_gamma
    ) / np.pi
    occupation = float(
        np.dot(gamma, reservoir_weights) / total_gamma
    )
    currents = gamma * (reservoir_weights - occupation)
    return ResonantLevelSteadyState(
        occupation=occupation,
        current_into_level=np.asarray(currents, dtype=float),
    )


def partition_free_resonant_level_amplitude(
    energy: RealArray,
    time: RealArray,
    *,
    level_energy: float,
    total_broadening: float,
    bias_shift: RealArray,
) -> ComplexArray:
    r"""Return the contacted-equilibrium amplitude after a step bias.

    For ``x=E-epsilon_d``, ``gamma=Gamma/2`` and a lead shift
    ``Delta_alpha``,

    .. math::

       A_\alpha(E,t)=
       \frac{e^{[i(x+\Delta_\alpha)-\gamma]t}}{x+i\gamma}
       +\frac{1-e^{[i(x+\Delta_\alpha)-\gamma]t}}
       {x+\Delta_\alpha+i\gamma}.

    Hence ``A_alpha(E,0)=1/(E-epsilon_d+i Gamma/2)`` for every lead.
    """
    energies = _strict_grid(
        energy, name="energy", minimum_size=2
    )
    times = _strict_grid(time, name="time", minimum_size=1)
    if times[0] < 0:
        raise ValueError("time is measured from the quench and must be >= 0.")
    gamma = float(total_broadening)
    epsilon = float(level_energy)
    shifts = np.asarray(bias_shift, dtype=float)
    if not np.isfinite(gamma) or gamma <= 0:
        raise ValueError("total_broadening must be positive.")
    if not np.isfinite(epsilon):
        raise ValueError("level_energy must be finite.")
    if shifts.ndim != 1 or shifts.size == 0:
        raise ValueError("bias_shift must be a nonempty vector.")
    if not np.all(np.isfinite(shifts)):
        raise ValueError("bias shifts must be finite.")
    detuning = energies - epsilon
    half_gamma = 0.5 * gamma
    exponent = (
        1j
        * (
            detuning[None, None, :]
            + shifts[None, :, None]
        )
        - half_gamma
    ) * times[:, None, None]
    transient = np.exp(exponent)
    initial_denominator = detuning + 1j * half_gamma
    final_denominator = (
        detuning[None, :]
        + shifts[:, None]
        + 1j * half_gamma
    )
    return (
        transient / initial_denominator[None, None, :]
        + (1.0 - transient) / final_denominator[None, :, :]
    )


def partition_free_resonant_level_bias_quench(
    time: RealArray,
    energy: RealArray,
    *,
    level_energy: float,
    broadening: RealArray,
    bias_shift: RealArray,
    initial_chemical_potential: float = 0.0,
    temperature: float = 0.0,
    max_memory_bytes: int = 256 * 1024**2,
) -> PartitionFreeResonantLevelTransient:
    r"""Solve a sudden partition-free lead-bias quench in the WBL.

    The dot and leads are contacted and in common equilibrium for ``t<0``.
    At ``t=0`` lead energies acquire constant shifts ``Delta_alpha``.  The
    common initial Fermi function remains ``f(E; mu_0, T)`` in the integration
    variable, while the final electrochemical potentials are
    ``mu_alpha=mu_0+Delta_alpha``.

    .. math::

       n(t)=\sum_\alpha\Gamma_\alpha\int\frac{dE}{2\pi}
       f(E)|A_\alpha(E,t)|^2,

    .. math::

       I_\alpha(t)=-\Gamma_\alpha n(t)
       -2\Gamma_\alpha\int\frac{dE}{2\pi}
       f(E)\,\mathrm{Im}\,A_\alpha(E,t).

    ``I_alpha`` is positive from lead ``alpha`` into the level and obeys
    ``dn/dt=sum_alpha I_alpha``.
    """
    times = _strict_grid(time, name="time", minimum_size=1)
    energies = _strict_grid(
        energy, name="energy", minimum_size=2
    )
    if times[0] < 0:
        raise ValueError("time is measured from the quench and must be >= 0.")
    gamma = np.asarray(broadening, dtype=float)
    shifts = np.asarray(bias_shift, dtype=float)
    if gamma.ndim != 1 or gamma.size == 0:
        raise ValueError("broadening must be a nonempty vector.")
    if shifts.shape != gamma.shape:
        raise ValueError("bias_shift must match broadening.")
    if not np.all(np.isfinite(gamma)) or np.any(gamma <= 0):
        raise ValueError("every lead broadening must be positive.")
    if not np.all(np.isfinite(shifts)):
        raise ValueError("bias shifts must be finite.")
    mu0 = float(initial_chemical_potential)
    thermal = float(temperature)
    if not np.isfinite(mu0):
        raise ValueError("initial_chemical_potential must be finite.")
    if not np.isfinite(thermal) or thermal < 0:
        raise ValueError("temperature cannot be negative.")
    if max_memory_bytes <= 0:
        raise ValueError("max_memory_bytes must be positive.")

    total_gamma = float(np.sum(gamma))
    filling = fermi_dirac(
        energies, mu=mu0, temperature=thermal
    )
    occupation = np.empty(times.size, dtype=float)
    currents = np.empty((times.size, gamma.size), dtype=float)
    bytes_per_time = max(
        1, energies.size * gamma.size * 64
    )
    if max_memory_bytes < bytes_per_time:
        raise MemoryError(
            "partition-free time-chunk workspace estimate "
            f"{bytes_per_time / 1024**2:.1f} MiB exceeds limit "
            f"{max_memory_bytes / 1024**2:.1f} MiB."
        )
    chunk_size = max(
        1,
        min(times.size, int(max_memory_bytes // bytes_per_time)),
    )
    for start in range(0, times.size, chunk_size):
        stop = min(start + chunk_size, times.size)
        amplitudes = partition_free_resonant_level_amplitude(
            energies,
            times[start:stop],
            level_energy=level_energy,
            total_broadening=total_gamma,
            bias_shift=shifts,
        )
        lead_occupations = np.trapezoid(
            filling[None, None, :] * np.abs(amplitudes) ** 2,
            energies,
            axis=2,
        ) / (2.0 * np.pi)
        chunk_occupation = np.sum(
            lead_occupations * gamma[None, :], axis=1
        )
        injection_integrals = np.trapezoid(
            filling[None, None, :] * amplitudes.imag,
            energies,
            axis=2,
        ) / (2.0 * np.pi)
        occupation[start:stop] = chunk_occupation
        currents[start:stop] = (
            -chunk_occupation[:, None] * gamma[None, :]
            - 2.0 * injection_integrals * gamma[None, :]
        )
    occupation_rate = np.sum(currents, axis=1)
    return PartitionFreeResonantLevelTransient(
        time=times,
        occupation=occupation,
        current_into_level=currents,
        occupation_rate=occupation_rate,
        lead_broadening=gamma,
        bias_shift=shifts,
        initial_chemical_potential=mu0,
        temperature=thermal,
        level_energy=float(level_energy),
    )
