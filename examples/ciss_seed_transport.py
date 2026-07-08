from __future__ import annotations

import numpy as np

from quantum_transport import KeldyshSelfEnergy, LeadSelfEnergy, MatrixDevice


SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
IDENTITY_2 = np.eye(2, dtype=np.complex128)


def orbital_name(site: int) -> str:
    return f"site{site}"


def basis_labels(n_sites: int) -> list[str]:
    return [f"{orbital_name(site)}_{spin}" for site in range(n_sites) for spin in ("up", "down")]


def contact_gamma(n_sites: int, site: int, gamma: float) -> np.ndarray:
    dim = 2 * n_sites
    out = np.zeros((dim, dim), dtype=np.complex128)
    block = 2 * site
    out[block : block + 2, block : block + 2] = gamma * IDENTITY_2
    return out


def orbital_contact_dict(n_sites: int, site: int, value: float) -> dict[str, float]:
    return {orbital_name(index): (value if index == site else 0.0) for index in range(n_sites)}


def chiral_spinful_chain(
    *,
    n_sites: int = 8,
    hopping: float = 1.0,
    lambda_soc: float = 0.12,
    chirality: int = 1,
    pitch_tilt: float = 0.25,
) -> MatrixDevice:
    """Toy helical spinful chain with a rotating spin-orbit hopping axis.

    This is a research seed, not a molecularly calibrated Hamiltonian.  It is
    meant to test which observables are robust under chirality reversal,
    magnetization reversal, bias, and phenomenological broadening.
    """

    labels = basis_labels(n_sites)
    hamiltonian = np.zeros((2 * n_sites, 2 * n_sites), dtype=np.complex128)

    for site in range(n_sites - 1):
        phi = chirality * 2.0 * np.pi * site / 3.0
        axis = (
            np.cos(phi) * SIGMA_X
            + np.sin(phi) * SIGMA_Y
            + pitch_tilt * SIGMA_Z
        )
        hop_block = hopping * IDENTITY_2 + 1.0j * lambda_soc * axis
        left = 2 * site
        right = 2 * (site + 1)
        hamiltonian[right : right + 2, left : left + 2] = hop_block
        hamiltonian[left : left + 2, right : right + 2] = hop_block.conj().T

    return MatrixDevice(hamiltonian, labels, name=f"toy_chiral_chain_chi_{chirality:+d}")


def dephasing_self_energy(dim: int, eta_phi: float, *, mu: float = 0.0) -> KeldyshSelfEnergy:
    return KeldyshSelfEnergy.equilibrium_from_retarded(
        dim=dim,
        sigma_retarded_fn=lambda _omega: -0.5j * eta_phi * np.eye(dim, dtype=np.complex128),
        mu=mu,
        temperature=0.0,
        name=f"equilibrium_broadening_eta_{eta_phi:g}",
    )


def normal_normal_scan() -> None:
    n_sites = 8
    omega_grid = np.linspace(-4.0, 4.0, 1001)
    mu_left = 0.25
    mu_right = -0.15

    print("\nNormal leads: charge and spin currents")
    print("chi  eta_phi    I_charge        I_spin_x        I_spin_y        I_spin_z")
    for chirality in (+1, -1):
        device = chiral_spinful_chain(n_sites=n_sites, chirality=chirality)
        left = LeadSelfEnergy.wide_band(contact_gamma(n_sites, 0, 0.5), mu=mu_left, name="L")
        right = LeadSelfEnergy.wide_band(contact_gamma(n_sites, n_sites - 1, 0.5), mu=mu_right, name="R")
        base = device.transport(left, right).keldysh_view()

        for eta_phi in (0.0, 0.02, 0.08):
            view = base if eta_phi == 0.0 else base.with_self_energy(dephasing_self_energy(device.dim, eta_phi))
            current = view.meir_wingreen_current(omega_grid, lead="left")
            spin = view.meir_wingreen_spin_current_vector(omega_grid, lead="left")
            print(
                f"{chirality:+d}   {eta_phi:6.3f}   {current:+.8e}   "
                f"{spin['x']:+.8e}   {spin['y']:+.8e}   {spin['z']:+.8e}"
            )


def ferromagnetic_analyzer_scan() -> None:
    n_sites = 8
    omega_grid = np.linspace(-4.0, 4.0, 1001)
    mu_left = 0.25
    mu_right = -0.15
    gamma_majority = orbital_contact_dict(n_sites, 0, 0.7)
    gamma_minority = orbital_contact_dict(n_sites, 0, 0.2)

    print("\nFM analyzer at left, normal right: magnetization-reversal asymmetry")
    print("chi  eta_phi    I_M_plus       I_M_minus      asymmetry")
    for chirality in (+1, -1):
        device = chiral_spinful_chain(n_sites=n_sites, chirality=chirality)
        right = LeadSelfEnergy.wide_band(contact_gamma(n_sites, n_sites - 1, 0.5), mu=mu_right, name="R")

        for eta_phi in (0.0, 0.02, 0.08):
            currents = []
            for theta in (0.0, np.pi):
                left = LeadSelfEnergy.ferromagnetic_wide_band(
                    device.basis_labels,
                    gamma_majority=gamma_majority,
                    gamma_minority=gamma_minority,
                    theta=theta,
                    phi=0.0,
                    mu=mu_left,
                    name="L_fm",
                )
                base = device.transport(left, right).keldysh_view()
                view = base if eta_phi == 0.0 else base.with_self_energy(dephasing_self_energy(device.dim, eta_phi))
                currents.append(view.meir_wingreen_current(omega_grid, lead="left"))

            denom = abs(currents[0]) + abs(currents[1])
            asymmetry = 0.0 if denom < 1e-15 else (currents[0] - currents[1]) / denom
            print(f"{chirality:+d}   {eta_phi:6.3f}   {currents[0]:+.8e}   {currents[1]:+.8e}   {asymmetry:+.8e}")


def main() -> None:
    print("CISS seed transport experiment using QuantumTransportEOM")
    print("Toy model: rotating SOC hopping axis, finite bias, Keldysh currents")
    normal_normal_scan()
    ferromagnetic_analyzer_scan()


if __name__ == "__main__":
    main()
