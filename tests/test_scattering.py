from __future__ import annotations

import numpy as np
import pytest

from quantum_transport import scattering_unitarity_error, wide_band_scattering_matrix


def test_wide_band_scattering_is_unitary() -> None:
    hamiltonian = np.array([[0.2, -0.4], [-0.4, -0.1]], dtype=np.complex128)
    broadening = np.diag([0.7, 0.3]).astype(np.complex128)
    scattering = wide_band_scattering_matrix(0.17, hamiltonian, broadening)
    assert scattering_unitarity_error(scattering) < 1e-12


def test_wide_band_scattering_rejects_nonpositive_broadening() -> None:
    with pytest.raises(ValueError, match="positive semidefinite"):
        wide_band_scattering_matrix(
            0.0,
            np.eye(2, dtype=np.complex128),
            np.diag([0.3, -0.1]).astype(np.complex128),
        )
