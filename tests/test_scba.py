from __future__ import annotations

import numpy as np

from quantum_transport import scba_two_time_greens, self_consistent_born_electron_boson


def _solve(mu_left: float = 0.0, mu_right: float = 0.0):
    return self_consistent_born_electron_boson(
        np.array([[0.1]], dtype=np.complex128),
        np.linspace(-3.0, 3.0, 121),
        np.array([[[0.6]], [[0.4]]], dtype=np.complex128),
        [mu_left, mu_right],
        coupling=np.array([[0.08]], dtype=np.complex128),
        boson_frequency=0.5,
        temperature=0.2,
        boson_temperature=0.2,
        max_iterations=60,
        mixing=0.5,
        tolerance=1e-9,
    )


def test_equilibrium_scba_converges_and_obeys_fdt() -> None:
    result = _solve()
    assert result.converged
    assert result.fdt_error() < 2e-12
    assert result.spectral_identity_error < 2e-12
    assert result.current_conservation_error < 2e-12


def test_nonequilibrium_scba_conserves_terminal_current() -> None:
    result = _solve(0.3, -0.3)
    assert result.converged
    assert result.lead_currents[0] > 0.0
    assert result.lead_currents[1] < 0.0
    assert result.current_conservation_error < 2e-10


def test_scba_stationary_solution_has_two_time_keldysh_identities() -> None:
    result = _solve()
    two_time = scba_two_time_greens(result, np.linspace(0.0, 1.0, 5))
    assert two_time.consistency_report().maximum < 2e-12
