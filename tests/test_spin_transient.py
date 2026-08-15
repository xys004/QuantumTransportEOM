from __future__ import annotations

import numpy as np
import pytest

from quantum_transport import one_body_bond_current, one_body_spin_bond_current


def test_spin_bond_current_reduces_to_weighted_spin_channel_currents() -> None:
    hamiltonian = np.zeros((4, 4), dtype=np.complex128)
    hamiltonian[0, 2] = hamiltonian[2, 0] = -0.7
    hamiltonian[1, 3] = hamiltonian[3, 1] = -0.3
    density = np.zeros((4, 4), dtype=np.complex128)
    density[2, 0] = 0.11j
    density[0, 2] = density[2, 0].conjugate()
    density[3, 1] = -0.07j
    density[1, 3] = density[3, 1].conjugate()
    spin_z = np.diag([0.5, -0.5]).astype(np.complex128)

    up = one_body_bond_current(hamiltonian, density, 0, 2)
    down = one_body_bond_current(hamiltonian, density, 1, 3)
    spin = one_body_spin_bond_current(
        hamiltonian,
        density,
        [0, 1],
        [2, 3],
        spin_z,
    )

    np.testing.assert_allclose(spin, 0.5 * (up - down), atol=1e-14)


def test_spin_bond_current_rejects_nonhermitian_or_mismatched_blocks() -> None:
    hamiltonian = np.eye(4, dtype=np.complex128)
    density = np.eye(4, dtype=np.complex128)
    with pytest.raises(ValueError, match="Hermitian"):
        one_body_spin_bond_current(
            hamiltonian,
            density,
            [0, 1],
            [2, 3],
            np.array([[0.0, 1.0], [0.0, 0.0]]),
        )
    with pytest.raises(ValueError, match="equal sizes"):
        one_body_spin_bond_current(
            hamiltonian,
            density,
            [0, 1],
            [2],
            np.eye(2),
        )
