from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantum_transport import KeldyshSelfEnergy, LeadSelfEnergy, MatrixDevice


@dataclass(frozen=True)
class LadderParameters:
    n_sites: int = 6
    gamma_chain_1: float = 1.0
    gamma_chain_2: float = 1.0
    gamma_parallel: float = 0.0
    gamma_hybrid: float = 1.0
    lambda_soc_1: float = 0.10
    lambda_soc_2: float = 0.10
    dresselhaus: float = 0.0
    beta: float = np.pi
    phase_site_count: int = 10
    chirality: int = 1
    onsite_chain_1: float = 0.0
    onsite_chain_2: float = 0.0

    def phase_step(self) -> float:
        return self.chirality * 2.0 * np.pi / (self.phase_site_count - 1)


BLOCKS: tuple[tuple[str, str], ...] = (
    ("c1_xip", "up"),
    ("c1_xip", "down"),
    ("c1_xim", "up"),
    ("c1_xim", "down"),
    ("c2_xip", "up"),
    ("c2_xip", "down"),
    ("c2_xim", "up"),
    ("c2_xim", "down"),
)


def idx(block: int, site: int, n_sites: int) -> int:
    return block * n_sites + site


def ladder_labels(n_sites: int) -> list[str]:
    return [f"{orbital}_site{site}_{spin}" for orbital, spin in BLOCKS for site in range(n_sites)]


def edge_gamma(n_sites: int, site: int, gamma: float) -> np.ndarray:
    dim = 8 * n_sites
    out = np.zeros((dim, dim), dtype=np.complex128)
    for block in range(8):
        out[idx(block, site, n_sites), idx(block, site, n_sites)] = gamma
    return out


def orbital_from_label(label: str) -> str:
    if label.endswith("_up"):
        return label[:-3]
    if label.endswith("_down"):
        return label[:-5]
    raise ValueError(f"Expected spinful label ending in _up or _down, got {label!r}.")


def edge_orbital_couplings(labels: list[str], site: int, value: float) -> dict[str, float]:
    out: dict[str, float] = {}
    for label in labels:
        orbital = orbital_from_label(label)
        out[orbital] = value if orbital.endswith(f"_site{site}") else 0.0
    return out


def dephasing_self_energy(dim: int, eta_phi: float, *, mu: float = 0.0) -> KeldyshSelfEnergy:
    return KeldyshSelfEnergy.equilibrium_from_retarded(
        dim=dim,
        sigma_retarded_fn=lambda _omega: -0.5j * eta_phi * np.eye(dim, dtype=np.complex128),
        mu=mu,
        temperature=0.0,
        name=f"equilibrium_broadening_eta_{eta_phi:g}",
    )


def build_ladder_device(params: LadderParameters) -> MatrixDevice:
    n_sites = params.n_sites
    h = np.zeros((8 * n_sites, 8 * n_sites), dtype=np.complex128)
    phase_step = params.phase_step()

    for n in range(1, n_sites + 1):
        site = n - 1
        for block in range(4):
            h[idx(block, site, n_sites), idx(block, site, n_sites)] = params.onsite_chain_1
            h[idx(block + 4, site, n_sites), idx(block + 4, site, n_sites)] = params.onsite_chain_2

        # Inter-chain hybridization. gamma_parallel preserves the xi sector;
        # gamma_hybrid mixes the complementary xi/spin channels used in the manuscript.
        for left, right in ((0, 4), (1, 5), (2, 6), (3, 7)):
            h[idx(left, site, n_sites), idx(right, site, n_sites)] = params.gamma_parallel
            h[idx(right, site, n_sites), idx(left, site, n_sites)] = params.gamma_parallel

        for left, right in ((0, 6), (1, 7), (2, 4), (3, 5)):
            h[idx(left, site, n_sites), idx(right, site, n_sites)] = params.gamma_hybrid
            h[idx(right, site, n_sites), idx(left, site, n_sites)] = params.gamma_hybrid

        if n < n_sites:
            forward_site = n
            phase = (n - 1) * phase_step
            for block in range(4):
                h[idx(block, site, n_sites), idx(block, forward_site, n_sites)] += params.gamma_chain_1
                h[idx(block + 4, site, n_sites), idx(block + 4, forward_site, n_sites)] += params.gamma_chain_2

            h[idx(0, site, n_sites), idx(2, forward_site, n_sites)] = (
                -1j * params.lambda_soc_1 * np.exp(-1j * phase)
                + params.dresselhaus * np.exp(1j * phase)
            )
            h[idx(1, site, n_sites), idx(3, forward_site, n_sites)] = (
                -1j * params.lambda_soc_1 * np.exp(1j * phase)
                - params.dresselhaus * np.exp(-1j * phase)
            )
            h[idx(2, site, n_sites), idx(0, forward_site, n_sites)] = (
                -1j * params.lambda_soc_1 * np.exp(1j * phase)
                - params.dresselhaus * np.exp(-1j * phase)
            )
            h[idx(3, site, n_sites), idx(1, forward_site, n_sites)] = (
                -1j * params.lambda_soc_1 * np.exp(-1j * phase)
                + params.dresselhaus * np.exp(1j * phase)
            )

            h[idx(4, site, n_sites), idx(6, forward_site, n_sites)] = (
                -1j * params.lambda_soc_2 * np.exp(-1j * phase) * np.exp(-1j * params.beta)
                + params.dresselhaus * np.exp(1j * phase)
            )
            h[idx(5, site, n_sites), idx(7, forward_site, n_sites)] = (
                -1j * params.lambda_soc_2 * np.exp(1j * phase) * np.exp(1j * params.beta)
                - params.dresselhaus * np.exp(-1j * phase)
            )
            h[idx(6, site, n_sites), idx(4, forward_site, n_sites)] = (
                -1j * params.lambda_soc_2 * np.exp(1j * phase) * np.exp(1j * params.beta)
                - params.dresselhaus * np.exp(-1j * phase)
            )
            h[idx(7, site, n_sites), idx(5, forward_site, n_sites)] = (
                -1j * params.lambda_soc_2 * np.exp(-1j * phase) * np.exp(-1j * params.beta)
                + params.dresselhaus * np.exp(1j * phase)
            )

        if n > 1:
            backward_site = n - 2
            phase = (n - 2) * phase_step
            for block in range(4):
                h[idx(block, site, n_sites), idx(block, backward_site, n_sites)] += params.gamma_chain_1
                h[idx(block + 4, site, n_sites), idx(block + 4, backward_site, n_sites)] += params.gamma_chain_2

            h[idx(0, site, n_sites), idx(2, backward_site, n_sites)] += (
                1j * params.lambda_soc_1 * np.exp(-1j * phase)
                - params.dresselhaus * np.exp(1j * phase)
            )
            h[idx(1, site, n_sites), idx(3, backward_site, n_sites)] += (
                1j * params.lambda_soc_1 * np.exp(1j * phase)
                + params.dresselhaus * np.exp(-1j * phase)
            )
            h[idx(2, site, n_sites), idx(0, backward_site, n_sites)] += (
                1j * params.lambda_soc_1 * np.exp(1j * phase)
                + params.dresselhaus * np.exp(-1j * phase)
            )
            h[idx(3, site, n_sites), idx(1, backward_site, n_sites)] += (
                1j * params.lambda_soc_1 * np.exp(-1j * phase)
                - params.dresselhaus * np.exp(1j * phase)
            )

            h[idx(4, site, n_sites), idx(6, backward_site, n_sites)] += (
                1j * params.lambda_soc_2 * np.exp(-1j * phase) * np.exp(-1j * params.beta)
                - params.dresselhaus * np.exp(1j * phase)
            )
            h[idx(5, site, n_sites), idx(7, backward_site, n_sites)] += (
                1j * params.lambda_soc_2 * np.exp(1j * phase) * np.exp(1j * params.beta)
                + params.dresselhaus * np.exp(-1j * phase)
            )
            h[idx(6, site, n_sites), idx(4, backward_site, n_sites)] += (
                1j * params.lambda_soc_2 * np.exp(1j * phase) * np.exp(1j * params.beta)
                + params.dresselhaus * np.exp(-1j * phase)
            )
            h[idx(7, site, n_sites), idx(5, backward_site, n_sites)] += (
                1j * params.lambda_soc_2 * np.exp(-1j * phase) * np.exp(-1j * params.beta)
                - params.dresselhaus * np.exp(1j * phase)
            )

    hermiticity_error = float(np.max(np.abs(h - h.conj().T)))
    if hermiticity_error > 1e-10:
        raise ValueError(f"Ladder Hamiltonian is not Hermitian: max error {hermiticity_error:.3e}.")

    return MatrixDevice(h, ladder_labels(n_sites), name=f"ciss_ladder_chi_{params.chirality:+d}")


def asymmetry(current_plus: float, current_minus: float) -> float:
    denom = abs(current_plus) + abs(current_minus)
    if denom < 1e-15:
        return 0.0
    return float((current_plus - current_minus) / denom)


def run_scan() -> None:
    omega_grid = np.linspace(-4.0, 4.0, 301)
    mu_left = 0.25
    mu_right = -0.15
    gamma_contact = 2.0

    print("CISS two-channel ladder in the QuantumTransportEOM Keldysh layer")
    print("Controls: gamma_hybrid=0 and lambda_soc=0 should suppress spin-z current")
    print()
    print("chi  gamma_hyb  lambda_soc  eta_phi    I_charge        I_spin_z        Pz_current")

    for chirality in (+1, -1):
        for gamma_hybrid, lambda_soc in ((0.0, 0.10), (1.0, 0.0), (0.25, 0.10), (1.0, 0.10)):
            for eta_phi in (0.0, 0.05):
                params = LadderParameters(
                    chirality=chirality,
                    gamma_hybrid=gamma_hybrid,
                    lambda_soc_1=lambda_soc,
                    lambda_soc_2=lambda_soc,
                )
                device = build_ladder_device(params)
                left = LeadSelfEnergy.wide_band(edge_gamma(params.n_sites, 0, gamma_contact), mu=mu_left, name="L")
                right = LeadSelfEnergy.wide_band(edge_gamma(params.n_sites, params.n_sites - 1, gamma_contact), mu=mu_right, name="R")
                view = device.transport(left, right).keldysh_view()
                if eta_phi > 0.0:
                    view = view.with_self_energy(dephasing_self_energy(device.dim, eta_phi, mu=0.5 * (mu_left + mu_right)))

                charge = view.meir_wingreen_current(omega_grid, lead="left")
                spin_z = view.meir_wingreen_spin_current(omega_grid, lead="left", axis="z")
                polarization = 0.0 if abs(charge) < 1e-15 else spin_z / charge
                print(
                    f"{chirality:+d}    {gamma_hybrid:7.3f}    {lambda_soc:7.3f}   {eta_phi:6.3f}   "
                    f"{charge:+.8e}   {spin_z:+.8e}   {polarization:+.8e}"
                )

    print()
    print("FM-left analyzer: charge-current asymmetry under magnetization reversal")
    print("chi  axis    I_M_plus       I_M_minus      A_current       Iz_M_plus      Iz_M_minus")
    for chirality in (+1, -1):
        params = LadderParameters(chirality=chirality, gamma_hybrid=1.0, lambda_soc_1=0.10, lambda_soc_2=0.10)
        device = build_ladder_device(params)
        labels = device.basis_labels
        right = LeadSelfEnergy.wide_band(
            edge_gamma(params.n_sites, params.n_sites - 1, gamma_contact),
            mu=mu_right,
            name="R",
        )
        gamma_majority = edge_orbital_couplings(labels, 0, 2.6)
        gamma_minority = edge_orbital_couplings(labels, 0, 0.8)
        for axis, orientations in {
            "z": ((0.0, 0.0), (np.pi, 0.0)),
            "x": ((0.5 * np.pi, 0.0), (0.5 * np.pi, np.pi)),
        }.items():
            currents = []
            spin_currents = []
            for theta, phi in orientations:
                left = LeadSelfEnergy.ferromagnetic_wide_band(
                    labels,
                    gamma_majority=gamma_majority,
                    gamma_minority=gamma_minority,
                    theta=theta,
                    phi=phi,
                    mu=mu_left,
                    name="L_fm",
                )
                view = device.transport(left, right).keldysh_view()
                currents.append(view.meir_wingreen_current(omega_grid, lead="left"))
                spin_currents.append(view.meir_wingreen_spin_current(omega_grid, lead="left", axis="z"))
            print(
                f"{chirality:+d}    {axis}   {currents[0]:+.8e}   {currents[1]:+.8e}   "
                f"{asymmetry(currents[0], currents[1]):+.8e}   {spin_currents[0]:+.8e}   {spin_currents[1]:+.8e}"
            )


if __name__ == "__main__":
    run_scan()
