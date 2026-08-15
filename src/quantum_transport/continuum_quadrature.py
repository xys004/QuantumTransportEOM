"""Finite star discretisations of flat-band continuum reservoirs.

The object returned here is a controlled quadrature oracle: a positive
semidefinite broadening matrix is represented by equally spaced lead modes
with weights ``Gamma * dE / (2*pi)``.  It is useful for comparing a
partition-free finite-lead calculation with the analytic wide-band solver on
times shorter than the star's recurrence time.  It is not a replacement for
the continuum self-energy and carries an explicit half-bandwidth cutoff.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _broadening(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("broadening must be a square matrix.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("broadening must be finite.")
    if not np.allclose(matrix, matrix.conj().T, atol=1e-12, rtol=1e-12):
        raise ValueError("broadening must be Hermitian.")
    if np.min(np.linalg.eigvalsh(matrix)) < -1e-10:
        raise ValueError("broadening must be positive semidefinite.")
    if np.max(np.abs(matrix)) <= 0.0:
        raise ValueError("broadening must not vanish identically.")
    return matrix


@dataclass(frozen=True)
class FlatBandStarQuadrature:
    """Mode Hamiltonian and coupling for a finite flat-band quadrature."""

    energies: np.ndarray
    lead_hamiltonian: np.ndarray
    coupling_matrix: np.ndarray
    broadening: np.ndarray
    half_bandwidth: float

    @property
    def spacing(self) -> float:
        return float(2.0 * self.half_bandwidth / self.energies.size)

    @property
    def mode_count(self) -> int:
        return int(self.lead_hamiltonian.shape[0])


def flat_band_star_quadrature(
    broadening: Any,
    *,
    half_bandwidth: float,
    n_points: int,
    center: float = 0.0,
) -> FlatBandStarQuadrature:
    r"""Return a midpoint star quadrature for a constant ``Gamma``.

    For eigenfactor ``R`` with ``Gamma = R R^dagger``, each energy node uses
    coupling ``R sqrt(dE/(2*pi))``.  Hence the discrete hybridisation is the
    midpoint quadrature approximation to
    ``Gamma * integral[dE/(2*pi)/(omega-epsilon+i0)]``.  The returned lead
    modes are ordered as ``(energy_0, channel_0..channel_r, energy_1, ...)``.
    """

    gamma = _broadening(broadening)
    width = float(half_bandwidth)
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("half_bandwidth must be finite and positive.")
    count = int(n_points)
    if count < 2:
        raise ValueError("n_points must be at least two.")
    midpoint_spacing = 2.0 * width / count
    energies = float(center) - width + (np.arange(count, dtype=float) + 0.5) * midpoint_spacing
    values, vectors = np.linalg.eigh(gamma)
    retained = values > max(1.0, float(values.max())) * 1e-13
    root = vectors[:, retained] * np.sqrt(np.clip(values[retained], 0.0, None))[None, :]
    channels = root.shape[1]
    lead_hamiltonian = np.kron(np.diag(energies), np.eye(channels, dtype=complex))
    coupling_block = root * np.sqrt(midpoint_spacing / (2.0 * np.pi))
    coupling_matrix = np.tile(coupling_block, (1, count))
    return FlatBandStarQuadrature(
        energies=energies,
        lead_hamiltonian=lead_hamiltonian,
        coupling_matrix=coupling_matrix,
        broadening=gamma.copy(),
        half_bandwidth=width,
    )


__all__ = ["FlatBandStarQuadrature", "flat_band_star_quadrature"]
