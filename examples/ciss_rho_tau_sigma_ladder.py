from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantum_transport import LeadSelfEnergy, MatrixDevice


PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
PAULI_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
PAULI_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
IDENTITY_2 = np.eye(2, dtype=np.complex128)


@dataclass(frozen=True)
class RhoTauSigmaParameters:
    n_sites: int = 6
    onsite: float = 0.0
    hopping: float = 1.0
    lambda_soc: float = 0.12
    gamma_parallel: float = 0.0
    gamma_hybrid: float = 0.8
    chain_detuning: float = 0.0
    channel_detuning: float = 0.0
    soc_tilt_z: float = 0.25
    helix_period: float = 3.0
    chirality: int = 1


def kron3(rho: np.ndarray, tau: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    return np.kron(np.kron(rho, tau), sigma)


RHO_X = kron3(PAULI_X, IDENTITY_2, IDENTITY_2)
RHO_Z = kron3(PAULI_Z, IDENTITY_2, IDENTITY_2)
TAU_X = kron3(IDENTITY_2, PAULI_X, IDENTITY_2)
TAU_Z = kron3(IDENTITY_2, PAULI_Z, IDENTITY_2)
SIGMA_X = kron3(IDENTITY_2, IDENTITY_2, PAULI_X)
SIGMA_Y = kron3(IDENTITY_2, IDENTITY_2, PAULI_Y)
SIGMA_Z = kron3(IDENTITY_2, IDENTITY_2, PAULI_Z)
LOCAL_IDENTITY = kron3(IDENTITY_2, IDENTITY_2, IDENTITY_2)


def site_slice(site: int) -> slice:
    start = 8 * site
    return slice(start, start + 8)


def basis_labels(n_sites: int) -> list[str]:
    labels: list[str] = []
    for site in range(n_sites):
        for chain in ("rho_plus", "rho_minus"):
            for channel in ("tau_plus", "tau_minus"):
                for spin in ("up", "down"):
                    labels.append(f"site{site}_{chain}_{channel}_{spin}")
    return labels


def helical_axis(site: int, params: RhoTauSigmaParameters) -> np.ndarray:
    phase = params.chirality * 2.0 * np.pi * site / params.helix_period
    return (
        np.cos(phase) * SIGMA_X
        + np.sin(phase) * SIGMA_Y
        + params.soc_tilt_z * SIGMA_Z
    )


def build_rho_tau_sigma_ladder(params: RhoTauSigmaParameters) -> MatrixDevice:
    dim = 8 * params.n_sites
    hamiltonian = np.zeros((dim, dim), dtype=np.complex128)

    onsite_block = (
        params.onsite * LOCAL_IDENTITY
        + params.chain_detuning * RHO_Z
        + params.channel_detuning * TAU_Z
        + params.gamma_parallel * RHO_X
        + params.gamma_hybrid * (RHO_X @ TAU_X)
    )

    for site in range(params.n_sites):
        local = site_slice(site)
        hamiltonian[local, local] = onsite_block

    for site in range(params.n_sites - 1):
        left = site_slice(site)
        right = site_slice(site + 1)
        hop_block = params.hopping * LOCAL_IDENTITY + 1.0j * params.lambda_soc * (TAU_X @ helical_axis(site, params))
        hamiltonian[right, left] = hop_block
        hamiltonian[left, right] = hop_block.conj().T

    hermiticity_error = float(np.max(np.abs(hamiltonian - hamiltonian.conj().T)))
    if hermiticity_error > 1e-10:
        raise ValueError(f"Hamiltonian is not Hermitian: max error {hermiticity_error:.3e}.")

    return MatrixDevice(
        hamiltonian,
        basis_labels(params.n_sites),
        name=f"rho_tau_sigma_ladder_chi_{params.chirality:+d}",
    )


def orbital_weight_matrix(
    *,
    chain_weights: tuple[float, float] = (1.0, 1.0),
    channel_weights: tuple[float, float] = (1.0, 1.0),
) -> np.ndarray:
    weights = np.array(
        [
            chain_weights[0] * channel_weights[0],
            chain_weights[0] * channel_weights[1],
            chain_weights[1] * channel_weights[0],
            chain_weights[1] * channel_weights[1],
        ],
        dtype=float,
    )
    return np.diag(weights).astype(np.complex128)


def embed_site_block(n_sites: int, site: int, local_block: np.ndarray) -> np.ndarray:
    out = np.zeros((8 * n_sites, 8 * n_sites), dtype=np.complex128)
    local = site_slice(site)
    out[local, local] = local_block
    return out


def normal_edge_gamma(
    n_sites: int,
    site: int,
    gamma: float,
    *,
    chain_weights: tuple[float, float] = (1.0, 1.0),
    channel_weights: tuple[float, float] = (1.0, 1.0),
) -> np.ndarray:
    orbital_weights = orbital_weight_matrix(chain_weights=chain_weights, channel_weights=channel_weights)
    local_gamma = gamma * np.kron(orbital_weights, IDENTITY_2)
    return embed_site_block(n_sites, site, local_gamma)


def magnetization_matrix(theta: float, phi: float) -> np.ndarray:
    mx = np.sin(theta) * np.cos(phi)
    my = np.sin(theta) * np.sin(phi)
    mz = np.cos(theta)
    return mx * PAULI_X + my * PAULI_Y + mz * PAULI_Z


def ferromagnetic_edge_gamma(
    n_sites: int,
    site: int,
    gamma: float,
    polarization: float,
    *,
    theta: float,
    phi: float,
    chain_weights: tuple[float, float] = (1.0, 1.0),
    channel_weights: tuple[float, float] = (1.0, 1.0),
) -> np.ndarray:
    if abs(polarization) > 1.0:
        raise ValueError("polarization must satisfy |polarization| <= 1.")
    orbital_weights = orbital_weight_matrix(chain_weights=chain_weights, channel_weights=channel_weights)
    spin_gamma = gamma * (IDENTITY_2 + polarization * magnetization_matrix(theta, phi))
    local_gamma = np.kron(orbital_weights, spin_gamma)
    return embed_site_block(n_sites, site, local_gamma)


def local_density_matrix(view, omega_grid: np.ndarray, site: int) -> np.ndarray:
    local = site_slice(site)
    values = np.array([view.lesser(float(energy))[local, local] for energy in omega_grid], dtype=np.complex128)
    rho = -1.0j * np.trapezoid(values, omega_grid, axis=0) / (2.0 * np.pi)
    return 0.5 * (rho + rho.conj().T)


def local_spin_z(view, omega_grid: np.ndarray, site: int) -> float:
    rho = local_density_matrix(view, omega_grid, site)
    return float(np.real(np.trace(0.5 * SIGMA_Z @ rho)))


def asymmetry(current_plus: float, current_minus: float) -> float:
    denom = abs(current_plus) + abs(current_minus)
    if denom < 1e-15:
        return 0.0
    return float((current_plus - current_minus) / denom)


def evaluate_fm_pair(
    params: RhoTauSigmaParameters,
    *,
    gamma_left: float = 2.0,
    gamma_right: float = 2.0,
    polarization: float = 0.65,
    right_chain_weights: tuple[float, float] = (1.0, 1.0),
    right_channel_weights: tuple[float, float] = (1.0, 1.0),
    omega_grid: np.ndarray,
    mu_left: float,
    mu_right: float,
) -> dict[str, float]:
    device = build_rho_tau_sigma_ladder(params)
    right = LeadSelfEnergy.wide_band(
        normal_edge_gamma(
            params.n_sites,
            params.n_sites - 1,
            gamma_right,
            chain_weights=right_chain_weights,
            channel_weights=right_channel_weights,
        ),
        mu=mu_right,
        name="R",
    )

    outputs: dict[str, float] = {}
    for label, theta, phi in (("plus", 0.0, 0.0), ("minus", np.pi, 0.0)):
        left = LeadSelfEnergy.wide_band(
            ferromagnetic_edge_gamma(
                params.n_sites,
                0,
                gamma_left,
                polarization,
                theta=theta,
                phi=phi,
            ),
            mu=mu_left,
            name=f"L_fm_{label}",
        )
        view = device.transport(left, right).keldysh_view()
        outputs[f"I_{label}"] = view.meir_wingreen_current(omega_grid, lead="left")
        outputs[f"Iz_{label}"] = view.meir_wingreen_spin_current(omega_grid, lead="left", axis="z")
        outputs[f"Sz_left_{label}"] = local_spin_z(view, omega_grid, 0)
    outputs["A_current"] = asymmetry(outputs["I_plus"], outputs["I_minus"])
    return outputs


def run_controls() -> None:
    omega_grid = np.linspace(-4.0, 4.0, 301)
    mu_left = 0.25
    mu_right = -0.15

    print("Rho/tau/sigma CISS ladder with explicit physical spin")
    print("Basis per site: chain rho x orbital channel tau x physical spin sigma")
    print()
    print("Normal leads controls")
    print("chi  gamma_hyb  lambda_soc  I_charge        I_spin_z        Sz_left")
    for chirality in (+1, -1):
        for gamma_hybrid, lambda_soc in ((0.0, 0.12), (0.8, 0.0), (0.8, 0.12)):
            params = RhoTauSigmaParameters(
                chirality=chirality,
                gamma_hybrid=gamma_hybrid,
                lambda_soc=lambda_soc,
            )
            device = build_rho_tau_sigma_ladder(params)
            left = LeadSelfEnergy.wide_band(normal_edge_gamma(params.n_sites, 0, 2.0), mu=mu_left, name="L")
            right = LeadSelfEnergy.wide_band(normal_edge_gamma(params.n_sites, params.n_sites - 1, 2.0), mu=mu_right, name="R")
            view = device.transport(left, right).keldysh_view()
            charge = view.meir_wingreen_current(omega_grid, lead="left")
            spin_z = view.meir_wingreen_spin_current(omega_grid, lead="left", axis="z")
            spin_left = local_spin_z(view, omega_grid, 0)
            print(f"{chirality:+d}    {gamma_hybrid:7.3f}    {lambda_soc:7.3f}  {charge:+.8e}  {spin_z:+.8e}  {spin_left:+.8e}")

    print()
    print("FM-left magnetization reversal scan")
    print("case                         chi  I_plus         I_minus        A_current      Iz_plus        Iz_minus       SzL_plus")
    cases = {
        "symmetric": {},
        "chain_detuned": {"chain_detuning": 0.35},
        "channel_detuned": {"channel_detuning": 0.35},
        "right_chain_filter": {"right_chain_weights": (1.0, 0.25)},
        "right_tau_filter": {"right_channel_weights": (1.0, 0.25)},
        "combined_filter_detuned": {
            "chain_detuning": 0.35,
            "channel_detuning": 0.20,
            "right_chain_weights": (1.0, 0.25),
            "right_channel_weights": (1.0, 0.40),
        },
    }
    for case_name, options in cases.items():
        for chirality in (+1, -1):
            case_options = dict(options)
            right_chain_weights = case_options.pop("right_chain_weights", (1.0, 1.0))
            right_channel_weights = case_options.pop("right_channel_weights", (1.0, 1.0))
            params = RhoTauSigmaParameters(chirality=chirality, **case_options)
            result = evaluate_fm_pair(
                params,
                right_chain_weights=right_chain_weights,
                right_channel_weights=right_channel_weights,
                omega_grid=omega_grid,
                mu_left=mu_left,
                mu_right=mu_right,
            )
            print(
                f"{case_name:<28s} {chirality:+d}  "
                f"{result['I_plus']:+.8e}  {result['I_minus']:+.8e}  {result['A_current']:+.8e}  "
                f"{result['Iz_plus']:+.8e}  {result['Iz_minus']:+.8e}  {result['Sz_left_plus']:+.8e}"
            )


if __name__ == "__main__":
    run_controls()
