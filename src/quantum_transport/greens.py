"""Green-function utilities for quadratic Hamiltonians."""

from __future__ import annotations

import numpy as np


def _omega_array(omega: float | np.ndarray) -> np.ndarray:
    arr = np.asarray(omega, dtype=float)
    return arr.reshape(1) if arr.ndim == 0 else arr


def fermi_dirac(omega: float | np.ndarray, mu: float = 0.0, temperature: float = 0.0) -> np.ndarray:
    """Fermi-Dirac occupation with k_B = 1."""
    w = np.asarray(omega, dtype=float)
    if temperature <= 0.0:
        out = np.zeros_like(w, dtype=float)
        out[w < mu] = 1.0
        out[w == mu] = 0.5
        return out
    x = (w - mu) / temperature
    x = np.clip(x, -700.0, 700.0)
    return 1.0 / (np.exp(x) + 1.0)


def bose_einstein(omega: float | np.ndarray, mu: float = 0.0, temperature: float = 0.0) -> np.ndarray:
    """Bose-Einstein occupation with k_B = 1."""
    w = np.asarray(omega, dtype=float)
    if temperature <= 0.0:
        return np.zeros_like(w, dtype=float)
    x = (w - mu) / temperature
    x = np.clip(x, -700.0, 700.0)
    denom = np.expm1(x)
    out = np.empty_like(w, dtype=float)
    mask = np.abs(denom) > 1e-14
    out[mask] = 1.0 / denom[mask]
    out[~mask] = np.where(x[~mask] < 0.0, -np.inf, np.inf)
    return out


def retarded_green(h: np.ndarray, omega: float | np.ndarray, eta: float = 1e-6) -> np.ndarray:
    """Retarded Green function: G^r = [ (w + i*eta)I - H ]^{-1}."""
    h = np.asarray(h, dtype=np.complex128)
    dim = h.shape[0]
    identity = np.eye(dim, dtype=np.complex128)
    w_arr = _omega_array(omega)
    out = np.empty((w_arr.size, dim, dim), dtype=np.complex128)
    for i, w in enumerate(w_arr):
        out[i] = np.linalg.inv((w + 1.0j * eta) * identity - h)
    return out[0] if np.asarray(omega).ndim == 0 else out


def advanced_green(h: np.ndarray, omega: float | np.ndarray, eta: float = 1e-6) -> np.ndarray:
    """Advanced Green function: G^a = [ (w - i*eta)I - H ]^{-1}."""
    h = np.asarray(h, dtype=np.complex128)
    dim = h.shape[0]
    identity = np.eye(dim, dtype=np.complex128)
    w_arr = _omega_array(omega)
    out = np.empty((w_arr.size, dim, dim), dtype=np.complex128)
    for i, w in enumerate(w_arr):
        out[i] = np.linalg.inv((w - 1.0j * eta) * identity - h)
    return out[0] if np.asarray(omega).ndim == 0 else out


def spectral_function(g_ret: np.ndarray, g_adv: np.ndarray) -> np.ndarray:
    """Spectral function matrix A = i(G^r - G^a)."""
    return 1.0j * (g_ret - g_adv)


def spectral_density(g_ret: np.ndarray, g_adv: np.ndarray) -> np.ndarray:
    """Density matrix rho(omega) = A(omega)/(2*pi)."""
    return spectral_function(g_ret, g_adv) / (2.0 * np.pi)


def lesser_green_equilibrium(
    g_ret: np.ndarray,
    g_adv: np.ndarray,
    omega: float | np.ndarray,
    mu: float = 0.0,
    temperature: float = 0.0,
) -> np.ndarray:
    """Lesser fermionic Green function in equilibrium: G^< = f(w) (G^a - G^r)."""
    f = fermi_dirac(omega, mu=mu, temperature=temperature)
    if np.asarray(omega).ndim == 0:
        return f * (g_adv - g_ret)
    return f[:, None, None] * (g_adv - g_ret)


def greater_green_equilibrium(
    g_ret: np.ndarray,
    g_adv: np.ndarray,
    omega: float | np.ndarray,
    mu: float = 0.0,
    temperature: float = 0.0,
) -> np.ndarray:
    """Greater fermionic Green function in equilibrium: G^> = (f(w)-1) (G^a - G^r)."""
    f = fermi_dirac(omega, mu=mu, temperature=temperature)
    if np.asarray(omega).ndim == 0:
        return (f - 1.0) * (g_adv - g_ret)
    return (f - 1.0)[:, None, None] * (g_adv - g_ret)
