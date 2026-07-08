from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq

from ciss_ladder_elastic_probe import (
    keldysh_components,
    projected_spin_current_density,
    retarded_green,
    terminal_current_density,
    transmission_matrix,
)
from ciss_rho_tau_sigma_ladder import (
    RhoTauSigmaParameters,
    asymmetry,
    build_rho_tau_sigma_ladder,
    ferromagnetic_edge_gamma,
    normal_edge_gamma,
)


@dataclass(frozen=True)
class VoltageProbeResult:
    chirality: int
    gamma_probe: float
    probe_kind: str
    magnetization: str
    current: float
    spin_current_z: float
    mu_probe: float
    residual: float


def fermi(energy: np.ndarray | float, *, mu: float, temperature: float) -> np.ndarray | float:
    if temperature <= 0.0:
        values = np.asarray(energy, dtype=float)
        out = np.where(values < mu, 1.0, np.where(values > mu, 0.0, 0.5))
        return float(out) if np.isscalar(energy) else out
    x = np.clip((np.asarray(energy, dtype=float) - mu) / temperature, -700.0, 700.0)
    out = 1.0 / (np.exp(x) + 1.0)
    return float(out) if np.isscalar(energy) else out


def internal_probe_gamma(
    n_sites: int,
    gamma_probe: float,
    *,
    kind: str,
) -> np.ndarray:
    if gamma_probe <= 0.0:
        return np.zeros((8 * n_sites, 8 * n_sites), dtype=np.complex128)

    if kind == "all":
        chain_weights = (1.0, 1.0)
        channel_weights = (1.0, 1.0)
    elif kind == "tau_plus":
        chain_weights = (1.0, 1.0)
        channel_weights = (1.0, 0.0)
    elif kind == "tau_minus":
        chain_weights = (1.0, 1.0)
        channel_weights = (0.0, 1.0)
    elif kind == "rho_plus":
        chain_weights = (1.0, 0.0)
        channel_weights = (1.0, 1.0)
    elif kind == "rho_minus":
        chain_weights = (0.0, 1.0)
        channel_weights = (1.0, 1.0)
    else:
        raise ValueError("kind must be all, tau_plus, tau_minus, rho_plus, or rho_minus.")

    gamma = np.zeros((8 * n_sites, 8 * n_sites), dtype=np.complex128)
    for site in range(1, n_sites - 1):
        gamma += normal_edge_gamma(
            n_sites,
            site,
            gamma_probe,
            chain_weights=chain_weights,
            channel_weights=channel_weights,
        )
    return gamma


def precompute_transport(device, gammas: list[np.ndarray], omega_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    transmissions = []
    greens = []
    for energy in omega_grid:
        g_retarded = retarded_green(float(energy), device, gammas)
        greens.append(g_retarded)
        transmissions.append(transmission_matrix(g_retarded, gammas))
    return np.array(transmissions, dtype=float), np.array(greens, dtype=np.complex128)


def integrated_terminal_current(
    transmissions: np.ndarray,
    omega_grid: np.ndarray,
    *,
    terminal: int,
    mu_left: float,
    mu_right: float,
    mu_probe: float,
    temperature: float,
) -> float:
    values = np.zeros_like(omega_grid, dtype=float)
    f_left = fermi(omega_grid, mu=mu_left, temperature=temperature)
    f_right = fermi(omega_grid, mu=mu_right, temperature=temperature)
    f_probe = fermi(omega_grid, mu=mu_probe, temperature=temperature)
    for index in range(omega_grid.size):
        occupations = np.array([f_left[index], f_right[index], f_probe[index]], dtype=float)
        values[index] = terminal_current_density(transmissions[index], occupations, terminal)
    return float(np.trapezoid(values, omega_grid))


def solve_probe_mu(
    transmissions: np.ndarray,
    omega_grid: np.ndarray,
    *,
    mu_left: float,
    mu_right: float,
    temperature: float,
) -> tuple[float, float]:
    def probe_current(mu_probe: float) -> float:
        return integrated_terminal_current(
            transmissions,
            omega_grid,
            terminal=2,
            mu_left=mu_left,
            mu_right=mu_right,
            mu_probe=mu_probe,
            temperature=temperature,
        )

    lower = float(min(omega_grid[0], mu_left, mu_right) - 1.0)
    upper = float(max(omega_grid[-1], mu_left, mu_right) + 1.0)
    f_lower = probe_current(lower)
    f_upper = probe_current(upper)
    if f_lower * f_upper > 0.0:
        candidates = np.linspace(mu_right - 1.0, mu_left + 1.0, 101)
        values = np.array([abs(probe_current(float(candidate))) for candidate in candidates])
        best = float(candidates[int(np.argmin(values))])
        return best, probe_current(best)

    mu_probe = float(brentq(probe_current, lower, upper, xtol=1e-8, rtol=1e-8, maxiter=100))
    return mu_probe, probe_current(mu_probe)


def integrate_voltage_probe_case(
    params: RhoTauSigmaParameters,
    *,
    gamma_probe: float,
    probe_kind: str,
    polarization: float = 0.65,
    right_chain_weights: tuple[float, float] = (1.0, 0.25),
    right_channel_weights: tuple[float, float] = (1.0, 0.40),
    magnetization_label: str,
    theta: float,
    phi: float,
    omega_grid: np.ndarray,
    mu_left: float,
    mu_right: float,
    temperature: float,
) -> VoltageProbeResult:
    device = build_rho_tau_sigma_ladder(params)
    gamma_left = ferromagnetic_edge_gamma(params.n_sites, 0, 2.0, polarization, theta=theta, phi=phi)
    gamma_right = normal_edge_gamma(
        params.n_sites,
        params.n_sites - 1,
        2.0,
        chain_weights=right_chain_weights,
        channel_weights=right_channel_weights,
    )
    gamma_probe_matrix = internal_probe_gamma(params.n_sites, gamma_probe, kind=probe_kind)
    gammas = [gamma_left, gamma_right, gamma_probe_matrix]
    transmissions, greens = precompute_transport(device, gammas, omega_grid)
    mu_probe, residual = solve_probe_mu(
        transmissions,
        omega_grid,
        mu_left=mu_left,
        mu_right=mu_right,
        temperature=temperature,
    )

    f_left = fermi(omega_grid, mu=mu_left, temperature=temperature)
    f_right = fermi(omega_grid, mu=mu_right, temperature=temperature)
    f_probe = fermi(omega_grid, mu=mu_probe, temperature=temperature)
    charge_density = np.zeros_like(omega_grid, dtype=float)
    spin_density = np.zeros_like(omega_grid, dtype=float)
    for index in range(omega_grid.size):
        occupations = np.array([f_left[index], f_right[index], f_probe[index]], dtype=float)
        charge_density[index] = terminal_current_density(transmissions[index], occupations, terminal=0)
        g_lesser, g_greater = keldysh_components(greens[index], gammas, occupations)
        spin_density[index] = projected_spin_current_density(
            device.basis_labels,
            gamma_left,
            float(f_left[index]),
            g_lesser,
            g_greater,
            axis="z",
        )

    return VoltageProbeResult(
        chirality=params.chirality,
        gamma_probe=gamma_probe,
        probe_kind=probe_kind,
        magnetization=magnetization_label,
        current=float(np.trapezoid(charge_density, omega_grid)),
        spin_current_z=float(np.trapezoid(spin_density, omega_grid)),
        mu_probe=mu_probe,
        residual=abs(residual),
    )


def run_scan() -> None:
    omega_grid = np.linspace(-4.0, 4.0, 201)
    mu_left = 0.25
    mu_right = -0.15
    temperature = 0.03

    print("Rho/tau/sigma ladder with one inelastic voltage probe")
    print("Probe condition: integrated I_p = 0; finite T smooths the voltage solve")
    print("Candidate detunings: chain_detuning=0.5, channel_detuning=1.2")
    print()
    print("chi  kind       gamma_probe  M    I_charge        I_spin_z        mu_probe   |Ip|")

    rows: dict[tuple[int, str, float, str], VoltageProbeResult] = {}
    for chirality in (+1, -1):
        params = RhoTauSigmaParameters(
            chirality=chirality,
            chain_detuning=0.50,
            channel_detuning=1.20,
        )
        for probe_kind in ("all", "tau_plus", "rho_plus"):
            for gamma_probe in (0.05, 0.25, 0.80):
                for label, theta, phi in (("+z", 0.0, 0.0), ("-z", np.pi, 0.0)):
                    result = integrate_voltage_probe_case(
                        params,
                        gamma_probe=gamma_probe,
                        probe_kind=probe_kind,
                        magnetization_label=label,
                        theta=theta,
                        phi=phi,
                        omega_grid=omega_grid,
                        mu_left=mu_left,
                        mu_right=mu_right,
                        temperature=temperature,
                    )
                    rows[(chirality, probe_kind, gamma_probe, label)] = result
                    print(
                        f"{chirality:+d}  {probe_kind:<8s}  {gamma_probe:7.3f}   {label:>2s}  "
                        f"{result.current:+.8e}  {result.spin_current_z:+.8e}  "
                        f"{result.mu_probe:+.5f}  {result.residual:.2e}"
                    )

    print()
    print("Magnetization-reversal current asymmetry")
    print("chi  kind       gamma_probe   A_current")
    for chirality in (+1, -1):
        for probe_kind in ("all", "tau_plus", "rho_plus"):
            for gamma_probe in (0.05, 0.25, 0.80):
                plus = rows[(chirality, probe_kind, gamma_probe, "+z")].current
                minus = rows[(chirality, probe_kind, gamma_probe, "-z")].current
                print(f"{chirality:+d}  {probe_kind:<8s}  {gamma_probe:7.3f}   {asymmetry(plus, minus):+.8e}")

    print()
    print("Focused controls for the strongest candidate: kind=tau_plus, gamma_probe=0.8")
    print("case        chi  lambda_soc  gamma_hyb  p_FM    A_current       I_plus         I_minus")
    controls = (
        ("candidate", RhoTauSigmaParameters(chirality=+1, chain_detuning=0.50, channel_detuning=1.20), 0.65),
        ("chi_flip", RhoTauSigmaParameters(chirality=-1, chain_detuning=0.50, channel_detuning=1.20), 0.65),
        ("lambda0", RhoTauSigmaParameters(chirality=+1, chain_detuning=0.50, channel_detuning=1.20, lambda_soc=0.0), 0.65),
        ("pFM0", RhoTauSigmaParameters(chirality=+1, chain_detuning=0.50, channel_detuning=1.20), 0.0),
    )
    fine_grid = np.linspace(-4.0, 4.0, 501)
    for case_name, params, polarization in controls:
        currents = []
        for label, theta, phi in (("+z", 0.0, 0.0), ("-z", np.pi, 0.0)):
            result = integrate_voltage_probe_case(
                params,
                gamma_probe=0.80,
                probe_kind="tau_plus",
                polarization=polarization,
                magnetization_label=label,
                theta=theta,
                phi=phi,
                omega_grid=fine_grid,
                mu_left=mu_left,
                mu_right=mu_right,
                temperature=temperature,
            )
            currents.append(result.current)
        print(
            f"{case_name:<10s}  {params.chirality:+d}    {params.lambda_soc:7.3f}    "
            f"{params.gamma_hybrid:7.3f}  {polarization:5.2f}  "
            f"{asymmetry(currents[0], currents[1]):+.8e}  {currents[0]:+.8e}  {currents[1]:+.8e}"
        )


if __name__ == "__main__":
    run_scan()
