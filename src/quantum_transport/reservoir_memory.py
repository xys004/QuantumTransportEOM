"""Analytic finite-band reservoir memory kernels.

For a Lorentzian hybridisation the retarded kernel is an exponential in the
time difference.  A smooth scalar drive enters through the exact gauge phase
``exp[-i(phi(t)-phi(t')]``; this is the useful control for flux/bias ramps that
should not be represented as an instantaneous jump.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .continuum_two_time import ContinuumTwoTimeSelfEnergy, _two_time_adjoint, inverse_fourier_two_time
from .greens import fermi_dirac


def _grid(time: Any) -> np.ndarray:
    values = np.asarray(time, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)) or np.any(np.diff(values) <= 0.0):
        raise ValueError("time must be finite and strictly increasing with at least two entries.")
    return values


def _hermitian(value: Any, dim: int | None = None) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or (dim is not None and matrix.shape != (dim, dim)):
        raise ValueError("broadening must be a square matrix.")
    if not np.allclose(matrix, matrix.conj().T, atol=1e-12, rtol=1e-12):
        raise ValueError("broadening must be Hermitian.")
    if np.min(np.linalg.eigvalsh(matrix)) < -1e-10:
        raise ValueError("broadening must be positive semidefinite.")
    return matrix


def _phase_factor(time: np.ndarray, phase: Any | None) -> np.ndarray:
    if phase is None:
        return np.ones((time.size, time.size), dtype=np.complex128)
    values = np.asarray(phase, dtype=float)
    if values.shape != time.shape or not np.all(np.isfinite(values)):
        raise ValueError("phase must have one finite value per time point.")
    return np.exp(-1j * (values[:, None] - values[None, :]))


@dataclass(frozen=True)
class LorentzianReservoirMemory:
    """Two-time Lorentzian reservoir components and their analytic controls."""

    time: np.ndarray
    broadening: np.ndarray
    bandwidth: float
    center: float
    chemical_potential: float
    temperature: float
    phase: np.ndarray | None
    retarded: np.ndarray
    advanced: np.ndarray
    lesser: np.ndarray
    greater: np.ndarray

    @property
    def memory_time(self) -> float:
        return 1.0 / self.bandwidth

    @property
    def retarded_causality_error(self) -> float:
        mask = np.triu(np.ones(self.retarded.shape[:2], dtype=bool), k=1)
        return float(np.max(np.abs(self.retarded[mask]))) if np.any(mask) else 0.0


def lorentzian_reservoir_two_time(
    time: Any,
    broadening: Any,
    *,
    bandwidth: float,
    center: float = 0.0,
    chemical_potential: float = 0.0,
    temperature: float = 0.0,
    energy_grid: Any | None = None,
    phase: Any | None = None,
) -> LorentzianReservoirMemory:
    r"""Build a finite-band Lorentzian ``Sigma^{r,a,<,>}(t,t')`` kernel.

    The retarded component is analytic,

    ``Sigma^r(t,t') = -i theta(t-t') Gamma W/2 exp[-(W+i eps_c)(t-t')]``.

    Lesser and greater components are obtained from the same Lorentzian
    spectral density on a caller-supplied energy grid.  If ``phase`` is given,
    every component is dressed by the exact scalar gauge factor for a smooth
    bias/flux protocol.
    """

    times = _grid(time)
    gamma = _hermitian(broadening)
    width = float(bandwidth)
    eps_c = float(center)
    mu = float(chemical_potential)
    thermal = float(temperature)
    if width <= 0.0 or not np.isfinite(width) or not np.isfinite(eps_c) or not np.isfinite(mu):
        raise ValueError("bandwidth, center, and chemical_potential must be finite with bandwidth > 0.")
    if thermal < 0.0 or not np.isfinite(thermal):
        raise ValueError("temperature must be finite and nonnegative.")
    lag = times[:, None] - times[None, :]
    causal = np.tril(np.ones((times.size, times.size), dtype=float), k=-1) + 0.5 * np.eye(times.size)
    decay = np.exp(-(width + 1j * eps_c) * np.maximum(lag, 0.0))
    retarded = -0.5j * width * causal[:, :, None, None] * decay[:, :, None, None] * gamma[None, None, :, :]
    if energy_grid is None:
        span = max(20.0 * width, 20.0)
        energy = np.linspace(eps_c - span, eps_c + span, 4001)
    else:
        energy = np.asarray(energy_grid, dtype=float)
        if energy.ndim != 1 or energy.size < 3 or not np.all(np.isfinite(energy)) or np.any(np.diff(energy) <= 0.0):
            raise ValueError("energy_grid must be finite and strictly increasing with at least three points.")
    lorentzian = width**2 / ((energy - eps_c) ** 2 + width**2)
    occupation = fermi_dirac(energy, mu=mu, temperature=thermal)
    lesser_w = 1j * occupation[:, None, None] * lorentzian[:, None, None] * gamma[None, :, :]
    greater_w = 1j * (occupation - 1.0)[:, None, None] * lorentzian[:, None, None] * gamma[None, :, :]
    lesser = inverse_fourier_two_time(energy, lesser_w, times)
    greater = inverse_fourier_two_time(energy, greater_w, times)
    phase_factor = _phase_factor(times, phase)
    retarded = retarded * phase_factor[:, :, None, None]
    lesser = lesser * phase_factor[:, :, None, None]
    greater = greater * phase_factor[:, :, None, None]
    advanced = _two_time_adjoint(retarded)
    return LorentzianReservoirMemory(
        time=times.copy(),
        broadening=gamma.copy(),
        bandwidth=width,
        center=eps_c,
        chemical_potential=mu,
        temperature=thermal,
        phase=None if phase is None else np.asarray(phase, dtype=float).copy(),
        retarded=retarded,
        advanced=advanced,
        lesser=lesser,
        greater=greater,
    )

