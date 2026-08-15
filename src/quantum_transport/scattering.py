"""Wide-band scattering matrices derived from a retarded device Green function."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


ComplexMatrix = NDArray[np.complex128]


def _validated_hermitian_matrix(
    value: ComplexMatrix, *, name: str, dimension: int | None = None
) -> ComplexMatrix:
    matrix = np.asarray(value, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"{name} must be a square matrix.")
    if dimension is not None and matrix.shape != (dimension, dimension):
        raise ValueError(f"{name} has incompatible dimensions.")
    if not np.allclose(matrix, matrix.conj().T, atol=1e-12):
        raise ValueError(f"{name} must be Hermitian.")
    return matrix


def _positive_semidefinite_sqrt(
    broadening: ComplexMatrix, *, tolerance: float = 1e-11
) -> ComplexMatrix:
    eigenvalues, eigenvectors = np.linalg.eigh(broadening)
    if np.min(eigenvalues) < -tolerance:
        raise ValueError("broadening must be positive semidefinite.")
    clipped = np.maximum(eigenvalues, 0.0)
    return (eigenvectors * np.sqrt(clipped)) @ eigenvectors.conj().T


def wide_band_scattering_matrix(
    energy: float,
    hamiltonian: ComplexMatrix,
    broadening: ComplexMatrix,
) -> ComplexMatrix:
    r"""Return the Fisher--Lee WBL scattering matrix.

    For ``G^r(E)=[E-H+i Gamma/2]^{-1}`` and ``W W^dagger=Gamma``, this
    returns ``S(E)=I-i W^dagger G^r(E) W`` in the eigenchannel basis of
    ``Gamma``.  Zero-eigenvalue directions are retained as identity channels,
    which makes unitarity and time-reversal tests dimensionally explicit.
    """
    matrix = _validated_hermitian_matrix(hamiltonian, name="hamiltonian")
    dimension = matrix.shape[0]
    gamma = _validated_hermitian_matrix(
        broadening, name="broadening", dimension=dimension
    )
    if not np.isfinite(energy):
        raise ValueError("energy must be finite.")
    coupling = _positive_semidefinite_sqrt(gamma)
    retarded = np.linalg.inv(
        float(energy) * np.eye(dimension, dtype=np.complex128)
        - matrix
        + 0.5j * gamma
    )
    return np.eye(dimension, dtype=np.complex128) - 1j * coupling @ retarded @ coupling


def scattering_unitarity_error(scattering: ComplexMatrix) -> float:
    """Return ``||S^dagger S-I||`` for a scattering matrix candidate."""
    matrix = np.asarray(scattering, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("scattering must be a square matrix.")
    return float(
        np.linalg.norm(matrix.conj().T @ matrix - np.eye(matrix.shape[0]))
    )
