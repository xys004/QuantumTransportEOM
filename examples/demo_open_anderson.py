from __future__ import annotations

import numpy as np

from quantum_transport import AndersonImpurity, KeldyshSelfEnergy


def main() -> None:
    omega_grid = np.linspace(-8.0, 8.0, 401)

    model = AndersonImpurity(eps=-1.0, U=2.0, zeeman=0.15, spin_flip=0.03j)
    open_view = model.open(
        {"up": 0.35, "down": 0.20},
        {"up": 0.25, "down": 0.15},
        mu_left=0.15,
        mu_right=-0.15,
    )
    extra_dephasing = KeldyshSelfEnergy.equilibrium_from_retarded(
        dim=2,
        sigma_retarded_fn=lambda omega: np.array([[-0.02j, 0.0], [0.0, -0.01j]], dtype=np.complex128),
        mu=0.0,
        temperature=0.0,
        name="extra_dephasing",
    )
    dressed_view = open_view.with_self_energy(extra_dephasing)

    hf_occ = open_view.self_consistent_occupations(
        omega_grid,
        eta=0.05,
        method="hartree_fock",
        initial={"up": 0.5, "down": 0.5},
        mixing=0.7,
        tol=5e-4,
        max_iter=80,
    )
    hi_occupations = {"up": 0.47, "down": 0.22}

    omega0 = 0.0
    print("Open Anderson impurity demo:")
    print("HF occupations:", hf_occ)
    print("Hubbard-I occupations:", hi_occupations)
    print("Local Hamiltonian:\n", open_view.local_hamiltonian())
    print("G_up^r(0) HF:", open_view.gf("up").retarded(omega0, eta=0.05, method="hartree_fock", occupations=hf_occ.occupations))
    print("G_up^r(0) Hubbard-I:", open_view.gf("up").retarded(omega0, eta=0.05, method="hubbard_i", occupations=hi_occupations))
    print("Transmission(0) HF:", open_view.transmission(omega0, eta=0.05, method="hartree_fock", occupations=hf_occ.occupations))
    print("Transmission(0) Hubbard-I:", open_view.transmission(omega0, eta=0.05, method="hubbard_i", occupations=hi_occupations))
    print("Meir-Wingreen current HF:", open_view.meir_wingreen_current(omega_grid, eta=0.05, method="hartree_fock", occupations=hf_occ.occupations))
    print("Meir-Wingreen current Hubbard-I:", open_view.meir_wingreen_current(omega_grid, eta=0.05, method="hubbard_i", occupations=hi_occupations))
    print("Meir-Wingreen current Hubbard-I dressed:", dressed_view.meir_wingreen_current(omega_grid, eta=0.05, method="hubbard_i", occupations=hi_occupations))
    print("Spin-resolved current z+ Hubbard-I:", open_view.spin_resolved_meir_wingreen_current(omega_grid, "+", eta=0.05, method="hubbard_i", occupations=hi_occupations, axis="z"))
    print("Spin-resolved current z- Hubbard-I:", open_view.spin_resolved_meir_wingreen_current(omega_grid, "-", eta=0.05, method="hubbard_i", occupations=hi_occupations, axis="z"))
    print("Spin Meir-Wingreen current z Hubbard-I:", open_view.spin_meir_wingreen_current(omega_grid, eta=0.05, method="hubbard_i", occupations=hi_occupations, axis="z"))


if __name__ == "__main__":
    main()
