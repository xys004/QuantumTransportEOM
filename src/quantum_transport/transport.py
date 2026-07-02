"""Transport observables from Green functions for the ring model."""

from __future__ import annotations

import numpy as np

from .greens import advanced_green, fermi_dirac, retarded_green


SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
SIGMA_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
IDENTITY_2 = np.eye(2, dtype=np.complex128)


def _idx(site: int, spin: int) -> int:
    return 2 * site + spin


def _normalize_axis(axis: str) -> str:
    key = str(axis).lower()
    aliases = {"sx": "x", "sy": "y", "sz": "z"}
    key = aliases.get(key, key)
    if key not in {"x", "y", "z"}:
        raise ValueError("axis must be one of 'x', 'y', 'z', 'sx', 'sy', 'sz'.")
    return key


def _normalize_component(component: str | int | float) -> int:
    if isinstance(component, (int, float, np.integer, np.floating)):
        return 1 if float(component) >= 0 else -1
    key = str(component).lower()
    if key in {"+", "plus", "up", "positive", "p"}:
        return 1
    if key in {"-", "minus", "down", "negative", "m"}:
        return -1
    raise ValueError("component must be '+', '-', 'up', 'down', or a signed number.")


def _axis_matrix(axis: str) -> np.ndarray:
    key = _normalize_axis(axis)
    if key == "x":
        return SIGMA_X
    if key == "y":
        return SIGMA_Y
    return SIGMA_Z


def ring_spin_projector(n_sites: int, axis: str = "z", component: str | int | float = "+") -> np.ndarray:
    sign = _normalize_component(component)
    local = 0.5 * (IDENTITY_2 + sign * _axis_matrix(axis))
    return np.kron(np.eye(n_sites, dtype=np.complex128), local)


def ring_current_operator(
    n_sites: int,
    gamma: float,
    lambda_r: float,
    phi_over_phi0: float,
) -> np.ndarray:
    theta = 2.0 * np.pi * phi_over_phi0 / float(n_sites)
    p_plus = np.exp(1.0j * theta)
    p_minus = np.exp(-1.0j * theta)
    operator = np.zeros((2 * n_sites, 2 * n_sites), dtype=np.complex128)

    for n in range(n_sites):
        m = (n + 1) % n_sites
        phi_n = 2.0 * np.pi * n / float(n_sites)
        phi_m = 2.0 * np.pi * m / float(n_sites)
        phi_bar = 0.5 * (phi_n + phi_m)

        n_up = _idx(n, 0)
        n_dn = _idx(n, 1)
        m_up = _idx(m, 0)
        m_dn = _idx(m, 1)

        operator[m_up, n_up] += gamma * p_plus
        operator[m_dn, n_dn] += gamma * p_plus
        operator[n_up, m_up] += -gamma * p_minus
        operator[n_dn, m_dn] += -gamma * p_minus

        operator[m_up, n_dn] += -1.0j * lambda_r * p_plus * np.exp(-1.0j * phi_bar)
        operator[m_dn, n_up] += -1.0j * lambda_r * p_plus * np.exp(1.0j * phi_bar)
        operator[n_up, m_dn] += -1.0j * lambda_r * p_minus * np.exp(-1.0j * phi_bar)
        operator[n_dn, m_up] += -1.0j * lambda_r * p_minus * np.exp(1.0j * phi_bar)

    return operator


def current_density_from_operator(
    g_ret: np.ndarray,
    g_adv: np.ndarray,
    omega: float,
    operator: np.ndarray,
    n_sites: int,
    mu: float = 0.0,
    temperature: float = 0.0,
    charge: float = 1.0,
) -> float:
    delta_g = g_ret - g_adv
    f = float(fermi_dirac(np.array([omega]), mu=mu, temperature=temperature)[0])
    value = (charge / float(n_sites)) * f * np.trace(operator @ delta_g)
    return float(np.real(value))


def current_density_omega(
    g_ret: np.ndarray,
    g_adv: np.ndarray,
    omega: float,
    n_sites: int,
    gamma: float,
    lambda_r: float,
    phi_over_phi0: float,
    mu: float = 0.0,
    temperature: float = 0.0,
    charge: float = 1.0,
) -> float:
    """
    Current density J_c(omega) in the structure of the manuscript equation.
    """
    operator = ring_current_operator(n_sites=n_sites, gamma=gamma, lambda_r=lambda_r, phi_over_phi0=phi_over_phi0)
    return current_density_from_operator(
        g_ret=g_ret,
        g_adv=g_adv,
        omega=omega,
        operator=operator,
        n_sites=n_sites,
        mu=mu,
        temperature=temperature,
        charge=charge,
    )


def spin_resolved_current_density_omega(
    g_ret: np.ndarray,
    g_adv: np.ndarray,
    omega: float,
    n_sites: int,
    gamma: float,
    lambda_r: float,
    phi_over_phi0: float,
    *,
    axis: str = "z",
    component: str | int | float = "+",
    mu: float = 0.0,
    temperature: float = 0.0,
    charge: float = 1.0,
) -> float:
    operator = ring_current_operator(n_sites=n_sites, gamma=gamma, lambda_r=lambda_r, phi_over_phi0=phi_over_phi0)
    projector = ring_spin_projector(n_sites=n_sites, axis=axis, component=component)
    projected_operator = 0.5 * (projector @ operator + operator @ projector)
    return current_density_from_operator(
        g_ret=g_ret,
        g_adv=g_adv,
        omega=omega,
        operator=projected_operator,
        n_sites=n_sites,
        mu=mu,
        temperature=temperature,
        charge=charge,
    )


def spin_current_density_omega(
    g_ret: np.ndarray,
    g_adv: np.ndarray,
    omega: float,
    n_sites: int,
    gamma: float,
    lambda_r: float,
    phi_over_phi0: float,
    *,
    axis: str = "z",
    mu: float = 0.0,
    temperature: float = 0.0,
    charge: float = 1.0,
) -> float:
    plus = spin_resolved_current_density_omega(
        g_ret=g_ret,
        g_adv=g_adv,
        omega=omega,
        n_sites=n_sites,
        gamma=gamma,
        lambda_r=lambda_r,
        phi_over_phi0=phi_over_phi0,
        axis=axis,
        component="+",
        mu=mu,
        temperature=temperature,
        charge=charge,
    )
    minus = spin_resolved_current_density_omega(
        g_ret=g_ret,
        g_adv=g_adv,
        omega=omega,
        n_sites=n_sites,
        gamma=gamma,
        lambda_r=lambda_r,
        phi_over_phi0=phi_over_phi0,
        axis=axis,
        component="-",
        mu=mu,
        temperature=temperature,
        charge=charge,
    )
    return float(plus - minus)


def persistent_current(
    hamiltonian: np.ndarray,
    omega_grid: np.ndarray,
    n_sites: int,
    gamma: float,
    lambda_r: float,
    phi_over_phi0: float,
    mu: float = 0.0,
    temperature: float = 0.0,
    eta: float = 1e-3,
    charge: float = 1.0,
) -> float:
    """Compute persistent current by integrating J_c(omega) over omega."""
    omega_grid = np.asarray(omega_grid, dtype=float)
    g_ret = retarded_green(hamiltonian, omega_grid, eta=eta)
    g_adv = advanced_green(hamiltonian, omega_grid, eta=eta)
    j_vals = np.array(
        [
            current_density_omega(
                g_ret=g_ret[i],
                g_adv=g_adv[i],
                omega=w,
                n_sites=n_sites,
                gamma=gamma,
                lambda_r=lambda_r,
                phi_over_phi0=phi_over_phi0,
                mu=mu,
                temperature=temperature,
                charge=charge,
            )
            for i, w in enumerate(omega_grid)
        ]
    )
    return float(np.trapezoid(j_vals, omega_grid))


def persistent_spin_resolved_current(
    hamiltonian: np.ndarray,
    omega_grid: np.ndarray,
    n_sites: int,
    gamma: float,
    lambda_r: float,
    phi_over_phi0: float,
    *,
    axis: str = "z",
    component: str | int | float = "+",
    mu: float = 0.0,
    temperature: float = 0.0,
    eta: float = 1e-3,
    charge: float = 1.0,
) -> float:
    omega_grid = np.asarray(omega_grid, dtype=float)
    g_ret = retarded_green(hamiltonian, omega_grid, eta=eta)
    g_adv = advanced_green(hamiltonian, omega_grid, eta=eta)
    j_vals = np.array(
        [
            spin_resolved_current_density_omega(
                g_ret=g_ret[i],
                g_adv=g_adv[i],
                omega=w,
                n_sites=n_sites,
                gamma=gamma,
                lambda_r=lambda_r,
                phi_over_phi0=phi_over_phi0,
                axis=axis,
                component=component,
                mu=mu,
                temperature=temperature,
                charge=charge,
            )
            for i, w in enumerate(omega_grid)
        ]
    )
    return float(np.trapezoid(j_vals, omega_grid))


def persistent_spin_current(
    hamiltonian: np.ndarray,
    omega_grid: np.ndarray,
    n_sites: int,
    gamma: float,
    lambda_r: float,
    phi_over_phi0: float,
    *,
    axis: str = "z",
    mu: float = 0.0,
    temperature: float = 0.0,
    eta: float = 1e-3,
    charge: float = 1.0,
) -> float:
    plus = persistent_spin_resolved_current(
        hamiltonian,
        omega_grid,
        n_sites=n_sites,
        gamma=gamma,
        lambda_r=lambda_r,
        phi_over_phi0=phi_over_phi0,
        axis=axis,
        component="+",
        mu=mu,
        temperature=temperature,
        eta=eta,
        charge=charge,
    )
    minus = persistent_spin_resolved_current(
        hamiltonian,
        omega_grid,
        n_sites=n_sites,
        gamma=gamma,
        lambda_r=lambda_r,
        phi_over_phi0=phi_over_phi0,
        axis=axis,
        component="-",
        mu=mu,
        temperature=temperature,
        eta=eta,
        charge=charge,
    )
    return float(plus - minus)


def drude_weight(current_plus: float, current_minus: float, delta_phi: float) -> float:
    """
    Finite-difference Drude proxy from flux-sensitive current response.
    D ~ dI/dphi.
    """
    if delta_phi == 0.0:
        raise ValueError("delta_phi must be non-zero.")
    return float((current_plus - current_minus) / (2.0 * delta_phi))
