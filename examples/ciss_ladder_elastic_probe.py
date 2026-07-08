from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quantum_transport import (
    LeadSelfEnergy,
    MatrixDevice,
    meir_wingreen_current_density,
    spin_axis_projector_numeric,
)

from ciss_ladder_keldysh import (
    LadderParameters,
    build_ladder_device,
    edge_gamma,
    edge_orbital_couplings,
    idx,
)


@dataclass(frozen=True)
class ProbeResult:
    chirality: int
    gamma_probe: float
    magnetization: str
    current: float
    spin_current_z: float
    max_probe_residual: float


def fermi_zero_temperature(energy: float, mu: float) -> float:
    if energy < mu:
        return 1.0
    if energy > mu:
        return 0.0
    return 0.5


def site_probe_gammas(n_sites: int, gamma_probe: float) -> list[np.ndarray]:
    if gamma_probe <= 0.0:
        return []

    dim = 8 * n_sites
    probes: list[np.ndarray] = []
    for site in range(1, n_sites - 1):
        gamma = np.zeros((dim, dim), dtype=np.complex128)
        for block in range(8):
            gamma[idx(block, site, n_sites), idx(block, site, n_sites)] = gamma_probe
        probes.append(gamma)
    return probes


def retarded_green(energy: float, device: MatrixDevice, gammas: list[np.ndarray]) -> np.ndarray:
    gamma_total = sum(gammas, np.zeros((device.dim, device.dim), dtype=np.complex128))
    sigma_retarded = -0.5j * gamma_total
    identity = np.eye(device.dim, dtype=np.complex128)
    return np.linalg.inv(energy * identity - device.hamiltonian - sigma_retarded)


def transmission_matrix(g_retarded: np.ndarray, gammas: list[np.ndarray]) -> np.ndarray:
    g_advanced = g_retarded.conj().T
    n_terminals = len(gammas)
    transmissions = np.zeros((n_terminals, n_terminals), dtype=float)
    for alpha, gamma_alpha in enumerate(gammas):
        left_part = gamma_alpha @ g_retarded
        for beta, gamma_beta in enumerate(gammas):
            transmissions[alpha, beta] = float(np.real(np.trace(left_part @ gamma_beta @ g_advanced)))
    transmissions[np.abs(transmissions) < 1e-14] = 0.0
    return transmissions


def solve_elastic_probe_occupations(
    transmissions: np.ndarray,
    *,
    f_left: float,
    f_right: float,
) -> np.ndarray:
    n_terminals = transmissions.shape[0]
    n_probes = n_terminals - 2
    occupations = np.zeros(n_terminals, dtype=float)
    occupations[0] = f_left
    occupations[1] = f_right

    if n_probes <= 0:
        return occupations

    matrix = np.zeros((n_probes, n_probes), dtype=float)
    source = np.zeros(n_probes, dtype=float)
    for row, terminal in enumerate(range(2, n_terminals)):
        total_out = float(np.sum(transmissions[terminal, :]) - transmissions[terminal, terminal])
        matrix[row, row] = total_out
        source[row] = transmissions[terminal, 0] * f_left + transmissions[terminal, 1] * f_right
        for col, other_terminal in enumerate(range(2, n_terminals)):
            if other_terminal != terminal:
                matrix[row, col] = -transmissions[terminal, other_terminal]

        if total_out < 1e-14:
            matrix[row, :] = 0.0
            matrix[row, row] = 1.0
            source[row] = 0.5

    occupations[2:] = np.clip(np.linalg.solve(matrix, source), 0.0, 1.0)
    return occupations


def terminal_current_density(transmissions: np.ndarray, occupations: np.ndarray, terminal: int) -> float:
    flow = np.sum(transmissions[terminal, :] * (occupations[terminal] - occupations))
    return float(flow / (2.0 * np.pi))


def keldysh_components(g_retarded: np.ndarray, gammas: list[np.ndarray], occupations: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sigma_lesser = np.zeros_like(g_retarded)
    sigma_greater = np.zeros_like(g_retarded)
    for gamma, occupation in zip(gammas, occupations):
        sigma_lesser += 1.0j * occupation * gamma
        sigma_greater += 1.0j * (occupation - 1.0) * gamma
    g_advanced = g_retarded.conj().T
    return g_retarded @ sigma_lesser @ g_advanced, g_retarded @ sigma_greater @ g_advanced


def projected_spin_current_density(
    labels: list[str],
    gamma_left: np.ndarray,
    f_left: float,
    g_lesser: np.ndarray,
    g_greater: np.ndarray,
    *,
    axis: str = "z",
) -> float:
    spin_current = 0.0
    for component, sign in (("+", 1.0), ("-", -1.0)):
        projector = spin_axis_projector_numeric(labels, axis=axis, component=component)
        sigma_lesser = projector @ (1.0j * f_left * gamma_left) @ projector
        sigma_greater = projector @ (1.0j * (f_left - 1.0) * gamma_left) @ projector
        spin_current += sign * meir_wingreen_current_density(sigma_lesser, sigma_greater, g_lesser, g_greater)
    return float(spin_current)


def integrate_with_elastic_probes(
    device: MatrixDevice,
    *,
    gamma_left: np.ndarray,
    gamma_right: np.ndarray,
    probe_gammas: list[np.ndarray],
    omega_grid: np.ndarray,
    mu_left: float,
    mu_right: float,
) -> tuple[float, float, float]:
    charge_density = np.zeros_like(omega_grid, dtype=float)
    spin_density_z = np.zeros_like(omega_grid, dtype=float)
    max_probe_residual = 0.0
    gammas = [gamma_left, gamma_right, *probe_gammas]

    for index, energy in enumerate(omega_grid):
        f_left = fermi_zero_temperature(float(energy), mu_left)
        f_right = fermi_zero_temperature(float(energy), mu_right)
        g_retarded = retarded_green(float(energy), device, gammas)
        transmissions = transmission_matrix(g_retarded, gammas)
        occupations = solve_elastic_probe_occupations(transmissions, f_left=f_left, f_right=f_right)

        charge_density[index] = terminal_current_density(transmissions, occupations, terminal=0)
        if len(gammas) > 2:
            residuals = [abs(terminal_current_density(transmissions, occupations, terminal=p)) for p in range(2, len(gammas))]
            max_probe_residual = max(max_probe_residual, max(residuals, default=0.0))

        g_lesser, g_greater = keldysh_components(g_retarded, gammas, occupations)
        spin_density_z[index] = projected_spin_current_density(
            device.basis_labels,
            gamma_left,
            f_left,
            g_lesser,
            g_greater,
            axis="z",
        )

    current = float(np.trapezoid(charge_density, omega_grid))
    spin_current_z = float(np.trapezoid(spin_density_z, omega_grid))
    return current, spin_current_z, max_probe_residual


def asymmetry(current_plus: float, current_minus: float) -> float:
    denom = abs(current_plus) + abs(current_minus)
    if denom < 1e-15:
        return 0.0
    return float((current_plus - current_minus) / denom)


def run_scan() -> None:
    n_sites = 6
    omega_grid = np.linspace(-4.0, 4.0, 301)
    mu_left = 0.25
    mu_right = -0.15
    gamma_contact = 2.0

    print("CISS ladder with elastic Buttiker dephasing probes")
    print("Probe condition: I_p(E)=0 at every energy; probes sit on internal ladder rungs")
    print()
    print("chi  gamma_probe  M    I_charge        I_spin_z        max|Ip(E)|")

    rows: dict[tuple[int, float, str], ProbeResult] = {}
    for chirality in (+1, -1):
        params = LadderParameters(
            n_sites=n_sites,
            chirality=chirality,
            gamma_hybrid=1.0,
            lambda_soc_1=0.10,
            lambda_soc_2=0.10,
        )
        device = build_ladder_device(params)
        right_gamma = edge_gamma(n_sites, n_sites - 1, gamma_contact)
        majority = edge_orbital_couplings(device.basis_labels, 0, 2.6)
        minority = edge_orbital_couplings(device.basis_labels, 0, 0.8)

        for gamma_probe in (0.0, 0.05, 0.20, 0.60):
            probes = site_probe_gammas(n_sites, gamma_probe)
            for label, theta, phi in (("+z", 0.0, 0.0), ("-z", np.pi, 0.0)):
                left = LeadSelfEnergy.ferromagnetic_wide_band(
                    device.basis_labels,
                    gamma_majority=majority,
                    gamma_minority=minority,
                    theta=theta,
                    phi=phi,
                    mu=mu_left,
                    name="L_fm",
                )
                current, spin_current_z, residual = integrate_with_elastic_probes(
                    device,
                    gamma_left=left.gamma(0.0),
                    gamma_right=right_gamma,
                    probe_gammas=probes,
                    omega_grid=omega_grid,
                    mu_left=mu_left,
                    mu_right=mu_right,
                )
                rows[(chirality, gamma_probe, label)] = ProbeResult(
                    chirality=chirality,
                    gamma_probe=gamma_probe,
                    magnetization=label,
                    current=current,
                    spin_current_z=spin_current_z,
                    max_probe_residual=residual,
                )
                print(
                    f"{chirality:+d}    {gamma_probe:7.3f}   {label:>2s}   "
                    f"{current:+.8e}   {spin_current_z:+.8e}   {residual:.3e}"
                )

    print()
    print("Magnetization-reversal current asymmetry A = (I_M - I_-M) / (|I_M| + |I_-M|)")
    print("chi  gamma_probe    A_current")
    for chirality in (+1, -1):
        for gamma_probe in (0.0, 0.05, 0.20, 0.60):
            plus = rows[(chirality, gamma_probe, "+z")].current
            minus = rows[(chirality, gamma_probe, "-z")].current
            print(f"{chirality:+d}    {gamma_probe:7.3f}    {asymmetry(plus, minus):+.8e}")


if __name__ == "__main__":
    run_scan()
