import numpy as np

from quantum_transport import two_time_meir_wingreen_current, two_time_spin_meir_wingreen_current


def _spin_selective_kernels():
    time = np.array([0.0, 0.2, 0.7, 1.0])
    n = time.size
    retarded = np.zeros((n, n, 2, 2), dtype=complex)
    lesser = np.zeros_like(retarded)
    sigma_lesser = np.zeros_like(retarded)
    sigma_advanced = np.zeros_like(retarded)
    for i in range(n):
        for j in range(n):
            retarded[i, j] = np.diag([0.2 + 0.1j, 0.0])
            lesser[i, j] = np.diag([0.15j, 0.0])
            sigma_lesser[i, j] = np.diag([0.25j, 0.0])
            sigma_advanced[i, j] = np.diag([0.1j, 0.0])
    return time, retarded, lesser, sigma_lesser, sigma_advanced


def test_two_time_meir_wingreen_zeroes_when_lead_injection_is_zero():
    time, retarded, lesser, _, _ = _spin_selective_kernels()
    zero = np.zeros_like(retarded)
    current = two_time_meir_wingreen_current(
        time,
        green_retarded=retarded,
        green_lesser=lesser,
        lead_self_energy_lesser=zero,
        lead_self_energy_advanced=zero,
    )
    np.testing.assert_allclose(current, 0.0, atol=1e-14)


def test_spin_current_tracks_spin_selective_charge_current():
    time, retarded, lesser, sigma_lesser, sigma_advanced = _spin_selective_kernels()
    charge = two_time_meir_wingreen_current(
        time,
        green_retarded=retarded,
        green_lesser=lesser,
        lead_self_energy_lesser=sigma_lesser,
        lead_self_energy_advanced=sigma_advanced,
    )
    spin = two_time_spin_meir_wingreen_current(
        time,
        green_retarded=retarded,
        green_lesser=lesser,
        lead_self_energy_lesser=sigma_lesser,
        lead_self_energy_advanced=sigma_advanced,
        spin_operator=np.diag([0.5, -0.5]),
    )
    np.testing.assert_allclose(spin, 0.5 * charge, atol=1e-14)
