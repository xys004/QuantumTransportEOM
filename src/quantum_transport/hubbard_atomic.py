"""Exact two-time Green functions for the atomic spinful Hubbard level.

The atomic Hamiltonian ``H = eps_up n_up + eps_down n_down + U n_up n_down``
has four many-body sectors.  Keeping those sectors explicitly provides a
same-Hamiltonian reference for the EOM/Hubbard-I closure used in Gate 21.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _validate_spin(spin: str) -> str:
    key = str(spin).lower()
    if key not in {"up", "down"}:
        raise ValueError("spin must be 'up' or 'down'.")
    return key


def _validate_probabilities(probabilities: Any) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    if values.shape != (4,) or not np.all(np.isfinite(values)):
        raise ValueError("state_probabilities must contain four finite entries.")
    if np.min(values) < -1e-12 or float(np.sum(values)) <= 0.0:
        raise ValueError("state_probabilities must be nonnegative with positive sum.")
    return values / np.sum(values)


def atomic_hubbard_u_probabilities(
    epsilon_up: float,
    epsilon_down: float,
    interaction_u: float,
    *,
    chemical_potential: float = 0.0,
    temperature: float = 0.0,
) -> np.ndarray:
    """Return exact grand-canonical probabilities ``(empty, up, down, double)``."""

    values = np.asarray(
        [epsilon_up, epsilon_down, interaction_u, chemical_potential, temperature],
        dtype=float,
    )
    if not np.all(np.isfinite(values)) or temperature < 0.0:
        raise ValueError("energies and temperature must be finite; temperature >= 0.")
    grand = np.asarray(
        [
            0.0,
            epsilon_up - chemical_potential,
            epsilon_down - chemical_potential,
            epsilon_up + epsilon_down + interaction_u - 2.0 * chemical_potential,
        ],
        dtype=float,
    )
    minimum = float(np.min(grand))
    if temperature == 0.0:
        ground = np.isclose(grand, minimum, rtol=0.0, atol=1e-13).astype(float)
        return ground / np.sum(ground)
    weights = np.exp(-(grand - minimum) / temperature)
    return weights / np.sum(weights)


def _resolve_probabilities(
    epsilon_up: float,
    epsilon_down: float,
    interaction_u: float,
    *,
    chemical_potential: float,
    temperature: float,
    state_probabilities: Any | None,
) -> np.ndarray:
    if state_probabilities is not None:
        return _validate_probabilities(state_probabilities)
    return atomic_hubbard_u_probabilities(
        epsilon_up,
        epsilon_down,
        interaction_u,
        chemical_potential=chemical_potential,
        temperature=temperature,
    )


def atomic_hubbard_u_retarded_frequency(
    omega: Any,
    epsilon_up: float,
    epsilon_down: float,
    interaction_u: float,
    *,
    spin: str = "up",
    eta: float = 0.0,
    chemical_potential: float = 0.0,
    temperature: float = 0.0,
    state_probabilities: Any | None = None,
) -> np.ndarray:
    r"""Exact atomic retarded Green function on a frequency grid."""

    key = _validate_spin(spin)
    if eta < 0.0 or not np.isfinite(eta):
        raise ValueError("eta must be finite and nonnegative.")
    probabilities = _resolve_probabilities(
        epsilon_up,
        epsilon_down,
        interaction_u,
        chemical_potential=chemical_potential,
        temperature=temperature,
        state_probabilities=state_probabilities,
    )
    opposite = float(probabilities[2] + probabilities[3] if key == "up" else probabilities[1] + probabilities[3])
    grid = np.asarray(omega, dtype=float)
    if not np.all(np.isfinite(grid)):
        raise ValueError("omega must contain finite values.")
    epsilon = float(epsilon_up if key == "up" else epsilon_down)
    z = grid.astype(np.complex128) + 1j * float(eta)
    return (1.0 - opposite) / (z - epsilon) + opposite / (z - epsilon - float(interaction_u))


@dataclass(frozen=True)
class AtomicHubbardTwoTimeResult:
    """Exact scalar Green-function components for one spin channel."""

    time: np.ndarray
    spin: str
    probabilities: np.ndarray
    retarded: np.ndarray
    advanced: np.ndarray
    lesser: np.ndarray
    greater: np.ndarray

    @property
    def occupation(self) -> float:
        return float(self.probabilities[1] + self.probabilities[3] if self.spin == "up" else self.probabilities[2] + self.probabilities[3])

    @property
    def opposite_occupation(self) -> float:
        return float(self.probabilities[2] + self.probabilities[3] if self.spin == "up" else self.probabilities[1] + self.probabilities[3])

    @property
    def spectral_identity_error(self) -> float:
        return float(np.max(np.abs(self.greater - self.lesser - self.retarded + self.advanced)))

    @property
    def advanced_adjoint_error(self) -> float:
        return float(np.max(np.abs(self.advanced - self.retarded.T.conj())))

    @property
    def lesser_antihermiticity_error(self) -> float:
        return float(np.max(np.abs(self.lesser + self.lesser.T.conj())))

    @property
    def equal_time_lesser_error(self) -> float:
        diagonal_occupation = -1j * np.diag(self.lesser)
        return float(np.max(np.abs(diagonal_occupation - self.occupation)))


def atomic_hubbard_u_two_time(
    time: Any,
    epsilon_up: float,
    epsilon_down: float,
    interaction_u: float,
    *,
    spin: str = "up",
    chemical_potential: float = 0.0,
    temperature: float = 0.0,
    state_probabilities: Any | None = None,
) -> AtomicHubbardTwoTimeResult:
    r"""Construct exact ``G^{r,a,<,>}(t,t')`` for the atomic Hubbard level.

    ``state_probabilities`` may be supplied for a stationary diagonal
    nonequilibrium ensemble.  Otherwise the exact grand-canonical ensemble is
    used.  The finite-grid convention is ``theta(0)=1/2``.
    """

    key = _validate_spin(spin)
    grid = np.asarray(time, dtype=float)
    if grid.ndim != 1 or grid.size < 2 or not np.all(np.isfinite(grid)) or np.any(np.diff(grid) <= 0.0):
        raise ValueError("time must be finite, strictly increasing, and contain at least two entries.")
    probabilities = _resolve_probabilities(
        epsilon_up,
        epsilon_down,
        interaction_u,
        chemical_potential=chemical_potential,
        temperature=temperature,
        state_probabilities=state_probabilities,
    )
    if key == "up":
        opposite_occupation = float(probabilities[2] + probabilities[3])
        lesser_weights = probabilities[1], probabilities[3]
        greater_weights = probabilities[0], probabilities[2]
        epsilon = float(epsilon_up)
    else:
        opposite_occupation = float(probabilities[1] + probabilities[3])
        lesser_weights = probabilities[2], probabilities[3]
        greater_weights = probabilities[0], probabilities[1]
        epsilon = float(epsilon_down)
    lag = grid[:, None] - grid[None, :]
    phase_empty = np.exp(-1j * epsilon * lag)
    phase_double = np.exp(-1j * (epsilon + float(interaction_u)) * lag)
    theta = np.tril(np.ones((grid.size, grid.size), dtype=float), k=-1) + 0.5 * np.eye(grid.size)
    retarded = -1j * theta * ((1.0 - opposite_occupation) * phase_empty + opposite_occupation * phase_double)
    lesser = 1j * (float(lesser_weights[0]) * phase_empty + float(lesser_weights[1]) * phase_double)
    greater = -1j * (float(greater_weights[0]) * phase_empty + float(greater_weights[1]) * phase_double)
    advanced = retarded.T.conj()
    return AtomicHubbardTwoTimeResult(
        time=grid.copy(),
        spin=key,
        probabilities=probabilities.copy(),
        retarded=retarded,
        advanced=advanced,
        lesser=lesser,
        greater=greater,
    )


__all__ = [
    "AtomicHubbardTwoTimeResult",
    "atomic_hubbard_u_probabilities",
    "atomic_hubbard_u_retarded_frequency",
    "atomic_hubbard_u_two_time",
]
