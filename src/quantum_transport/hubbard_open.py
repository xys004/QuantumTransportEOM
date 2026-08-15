"""Lead-coupled Hubbard-I formulas for controlled exact benchmarks."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def finite_lead_retarded_embedding(
    omega: Any,
    *,
    lead_hamiltonians: Sequence[Any],
    coupling_matrices: Sequence[Any],
    eta: float = 0.02,
) -> np.ndarray:
    """Return ``Σ^r(ω)=Σ V(ω-h_lead+iη)⁻¹V†`` for finite leads."""

    grid = np.asarray(omega, dtype=float)
    if grid.ndim != 1 or not np.all(np.isfinite(grid)) or eta < 0.0 or not np.isfinite(eta):
        raise ValueError("omega must be finite and eta nonnegative.")
    if len(lead_hamiltonians) != len(coupling_matrices) or not lead_hamiltonians:
        raise ValueError("one coupling matrix is required per lead.")
    first = np.asarray(coupling_matrices[0], dtype=np.complex128)
    if first.ndim != 2:
        raise ValueError("coupling matrices must be two-dimensional.")
    dimension = first.shape[0]
    result = np.zeros((grid.size, dimension, dimension), dtype=np.complex128)
    identity_cache: list[np.ndarray] = []
    for index, (lead_value, coupling_value) in enumerate(zip(lead_hamiltonians, coupling_matrices)):
        lead = np.asarray(lead_value, dtype=np.complex128)
        coupling = np.asarray(coupling_value, dtype=np.complex128)
        if lead.ndim != 2 or lead.shape[0] != lead.shape[1] or coupling.shape != (dimension, lead.shape[0]):
            raise ValueError(f"lead {index} has incompatible dimensions.")
        if not np.allclose(lead, lead.conj().T, atol=1e-12, rtol=1e-12):
            raise ValueError("lead Hamiltonians must be Hermitian.")
        identity_cache.append(np.eye(lead.shape[0], dtype=np.complex128))
        for point, value in enumerate(grid):
            result[point] += coupling @ np.linalg.inv((value + 1j * eta) * identity_cache[-1] - lead) @ coupling.conj().T
    return result


def lead_coupled_hubbard_i_retarded(
    omega: Any,
    *,
    epsilon: float,
    interaction_u: float,
    opposite_occupation: float,
    embedding_retarded: Any,
    eta: float = 0.02,
    embedding_form: str = "dyson",
) -> np.ndarray:
    r"""Hubbard-I impurity Green function with an arbitrary lead embedding.

    The atomic Hubbard-I propagator is

    ``g_at(z) = (1-n_o)/(z-eps) + n_o/(z-eps-U)``,

    and ``embedding_form`` selects how the lead embedding enters it.

    ``"dyson"`` (default) solves the Dyson equation
    ``G = [g_at^{-1} - Sigma]^{-1}``, evaluated as ``g_at/(1 - Sigma g_at)``
    so that neither the zero nor the poles of ``g_at`` have to be formed
    explicitly.  This is what "lead-coupled Hubbard-I" conventionally means.

    ``"two_pole"`` inserts the embedding into each atomic denominator
    separately, ``(1-n_o)/(z-eps-Sigma) + n_o/(z-eps-U-Sigma)``.  That is a
    different, cruder ansatz.  It is retained because the Gate 31 record
    before 2026-08-15 used it, and because the difference between the two is
    itself a useful diagnostic; note that the two coincide exactly at
    ``n_o = 0`` and ``n_o = 1``, so a ``U = 0`` control cannot distinguish
    them.

    The controlled reference uses the exact opposite-spin occupation when
    requested by a benchmark; replacing it by a self-consistent estimate is a
    separate approximation and should be reported as such.
    """

    grid = np.asarray(omega, dtype=float)
    embedding = np.asarray(embedding_retarded, dtype=np.complex128)
    if grid.ndim != 1 or embedding.shape != (grid.size, 1, 1):
        raise ValueError("embedding_retarded must have shape (n_omega, 1, 1).")
    values = float(opposite_occupation)
    if not np.isfinite(values) or values < -1e-12 or values > 1.0 + 1e-12:
        raise ValueError("opposite_occupation must lie in [0, 1].")
    if not np.isfinite(epsilon) or not np.isfinite(interaction_u) or eta < 0.0 or not np.isfinite(eta):
        raise ValueError("epsilon, interaction_u, and eta must be finite with eta >= 0.")
    form = str(embedding_form).lower()
    if form not in {"dyson", "two_pole"}:
        raise ValueError("embedding_form must be 'dyson' or 'two_pole'.")
    z = grid + 1j * float(eta)
    sigma = embedding[:, 0, 0]
    level = float(epsilon)
    interaction = float(interaction_u)
    if form == "two_pole":
        result = (1.0 - values) / (z - level - sigma) + values / (z - level - interaction - sigma)
        return result[:, None, None]
    atomic = (1.0 - values) / (z - level) + values / (z - level - interaction)
    return (atomic / (1.0 - sigma * atomic))[:, None, None]


__all__ = ["finite_lead_retarded_embedding", "lead_coupled_hubbard_i_retarded"]
