"""Generic matrix-valued time-dependent embedding on a two-time grid.

This module deliberately accepts the embedding kernels themselves.  A lead,
flux ramp, gauge phase, or numerically tabulated self-energy may therefore be
nonstationary and noncommuting with the device Hamiltonian; no Fourier or
wide-band assumption is made here.  The finite-grid Dyson layer remains
explicit about its convergence and Keldysh residuals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .kadanoff_baym import (
    TwoTimeDysonResult,
    _time_grid,
    greater_from_keldysh_discontinuity,
    kadanoff_baym_dyson_two_time,
    two_time_adjoint,
)


def _stack(value: Any, grid: np.ndarray, *, name: str, dim: int | None = None) -> np.ndarray:
    result = np.asarray(value, dtype=np.complex128)
    if result.ndim != 4 or result.shape[:2] != (grid.size, grid.size):
        raise ValueError(f"{name} must have shape (n_time, n_time, dim, dim).")
    if result.shape[2] != result.shape[3] or (dim is not None and result.shape[2:] != (dim, dim)):
        raise ValueError(f"{name} must contain square matrices of dimension {dim or result.shape[2]}.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


@dataclass(frozen=True)
class TimeDependentEmbeddingResult:
    """Dyson result plus the supplied nonstationary embedding and diagnostics."""

    time: np.ndarray
    green: TwoTimeDysonResult
    embedding_retarded: np.ndarray
    embedding_advanced: np.ndarray
    embedding_lesser: np.ndarray
    embedding_greater: np.ndarray

    @property
    def converged(self) -> bool:
        return bool(self.green.converged)

    @property
    def iterations(self) -> int:
        return int(self.green.iterations)

    @property
    def maximum_update(self) -> float:
        return float(self.green.maximum_update)

    @property
    def retarded_causality_error(self) -> float:
        upper = np.triu(np.ones((self.time.size, self.time.size), dtype=bool), k=1)
        return float(np.max(np.abs(self.embedding_retarded[upper])))

    @property
    def advanced_adjoint_error(self) -> float:
        return float(np.max(np.abs(self.embedding_advanced - two_time_adjoint(self.embedding_retarded))))

    @property
    def lesser_antihermiticity_error(self) -> float:
        return float(np.max(np.abs(self.embedding_lesser + two_time_adjoint(self.embedding_lesser))))

    @property
    def greater_antihermiticity_error(self) -> float:
        return float(np.max(np.abs(self.embedding_greater + two_time_adjoint(self.embedding_greater))))

    @property
    def keldysh_spectral_error(self) -> float:
        return float(
            np.max(
                np.abs(
                    (self.embedding_retarded - self.embedding_advanced)
                    - (self.embedding_greater - self.embedding_lesser)
                )
            )
        )

    @property
    def green_spectral_error(self) -> float:
        return float(np.max(np.abs(self.green.greater - self.green.lesser - self.green.retarded + self.green.advanced)))


def solve_time_dependent_matrix_embedding(
    time: Any,
    *,
    bare_retarded: Any,
    bare_lesser: Any,
    embedding_self_energy_retarded: Any,
    embedding_self_energy_lesser: Any,
    embedding_self_energy_advanced: Any | None = None,
    embedding_self_energy_greater: Any | None = None,
    initial_correlation_lesser: Any | None = None,
    max_iterations: int = 100,
    mixing: float = 0.5,
    tolerance: float = 1e-10,
) -> TimeDependentEmbeddingResult:
    r"""Solve Dyson/KBE for arbitrary matrix-valued ``Sigma(t,t')`` kernels.

    The supplied retarded kernel is required to be causal on the sampled grid;
    its advanced component defaults to the exact two-time adjoint.  The lesser
    kernel may carry arbitrary time-dependent bias phases and matrix rotations.
    ``embedding_self_energy_greater`` is optional and is reconstructed from the
    Keldysh discontinuity when omitted, so the returned embedding diagnostics
    cannot silently hide a spectral mismatch.
    """

    grid = _time_grid(time)
    bare_r = _stack(bare_retarded, grid, name="bare_retarded")
    dim = bare_r.shape[-1]
    bare_l = _stack(bare_lesser, grid, name="bare_lesser", dim=dim)
    sigma_r = _stack(embedding_self_energy_retarded, grid, name="embedding_self_energy_retarded", dim=dim)
    sigma_l = _stack(embedding_self_energy_lesser, grid, name="embedding_self_energy_lesser", dim=dim)
    sigma_a = (
        two_time_adjoint(sigma_r)
        if embedding_self_energy_advanced is None
        else _stack(embedding_self_energy_advanced, grid, name="embedding_self_energy_advanced", dim=dim)
    )
    sigma_g = (
        greater_from_keldysh_discontinuity(sigma_r, sigma_a, sigma_l)
        if embedding_self_energy_greater is None
        else _stack(embedding_self_energy_greater, grid, name="embedding_self_energy_greater", dim=dim)
    )
    if initial_correlation_lesser is not None:
        initial = _stack(initial_correlation_lesser, grid, name="initial_correlation_lesser", dim=dim)
    else:
        initial = None
    result = kadanoff_baym_dyson_two_time(
        grid,
        bare_retarded=bare_r,
        bare_lesser=bare_l,
        self_energy_retarded=sigma_r,
        self_energy_lesser=sigma_l,
        self_energy_advanced=sigma_a,
        initial_correlation_lesser=initial,
        max_iterations=max_iterations,
        mixing=mixing,
        tolerance=tolerance,
    )
    return TimeDependentEmbeddingResult(
        time=grid.copy(),
        green=result,
        embedding_retarded=sigma_r.copy(),
        embedding_advanced=sigma_a.copy(),
        embedding_lesser=sigma_l.copy(),
        embedding_greater=sigma_g.copy(),
    )


__all__ = ["TimeDependentEmbeddingResult", "solve_time_dependent_matrix_embedding"]
