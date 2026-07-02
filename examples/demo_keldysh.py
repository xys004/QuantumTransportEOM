from __future__ import annotations

import numpy as np

from quantum_transport import KeldyshSelfEnergy, LeadSelfEnergy, SpinfulSingleSite


def main() -> None:
    device = SpinfulSingleSite(eps_up=0.1, eps_down=0.25, spin_flip=0.0)
    left = LeadSelfEnergy.ferromagnetic_wide_band(
        ["up", "down"],
        gamma_majority=0.6,
        gamma_minority=0.2,
        theta=np.pi / 3.0,
        phi=0.0,
        mu=0.2,
        name="L_fm",
    )
    right = LeadSelfEnergy.wide_band(np.diag([0.4, 0.4]), mu=-0.2, name="R")

    transport = device.transport(left, right)
    keldysh = transport.keldysh_view()
    omega_grid = np.linspace(-6.0, 6.0, 3001)
    omega0 = 0.0

    sampled_sigma = KeldyshSelfEnergy.sampled(
        omega_grid=np.array([-2.0, 0.0, 2.0], dtype=float),
        sigma_retarded_values=np.array(
            [
                [[-0.05j, 0.0], [0.0, 0.0]],
                [[-0.10j, 0.0], [0.0, -0.03j]],
                [[-0.15j, 0.0], [0.0, -0.06j]],
            ],
            dtype=np.complex128,
        ),
        mu=0.0,
        temperature=0.0,
        name="sampled_int",
    )
    dressed = keldysh.with_self_energy(sampled_sigma)

    print("Stationary Keldysh demo:")
    print("Sigma^r_total(0) bare:\n", keldysh.sigma_retarded(omega0, lead="total"))
    print("Sigma^r_int(0):\n", dressed.sigma_retarded(omega0, lead="interaction"))
    print("Sigma^r_total(0) dressed:\n", dressed.sigma_retarded(omega0, lead="total"))
    print("Sigma^<(0)_left:\n", dressed.sigma_lesser(omega0, lead="left"))
    print("Sigma^K(0)_int:\n", dressed.sigma_keldysh(omega0, lead="interaction"))
    print("G^r(0) bare:\n", keldysh.retarded(omega0))
    print("G^r(0) dressed:\n", dressed.retarded(omega0))
    print("G^<(0) dressed:\n", dressed.lesser(omega0))
    print("G^>(0) dressed:\n", dressed.greater(omega0))
    print("G^K(0) dressed:\n", dressed.keldysh(omega0))

    print("\nCurrents:")
    print("Landauer current (bare):", transport.landauer_current(omega_grid, mu_left=0.2, mu_right=-0.2))
    print("Meir-Wingreen current (bare):", keldysh.meir_wingreen_current(omega_grid, lead="left"))
    print("Meir-Wingreen current (dressed):", dressed.meir_wingreen_current(omega_grid, lead="left"))
    print("Spin Meir-Wingreen current z (dressed):", dressed.meir_wingreen_spin_current(omega_grid, lead="left", axis="z"))
    print("Spin Meir-Wingreen current vector (dressed):", dressed.meir_wingreen_spin_current_vector(omega_grid, lead="left"))


if __name__ == "__main__":
    main()
