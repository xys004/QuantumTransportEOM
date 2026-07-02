"""Self-consistent collinear-Hartree routines for the Rashba-Hubbard ring."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hamiltonians import build_rashba_hubbard_ring_real_space


@dataclass
class CollinearHartreeResult:
    hamiltonian: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    n_up: np.ndarray
    n_down: np.ndarray
    occupations: np.ndarray
    iterations: int
    converged: bool


def _occupations_zero_temperature(evals: np.ndarray, n_electrons: float) -> np.ndarray:
    occ = np.zeros_like(evals, dtype=float)
    full = int(np.floor(n_electrons))
    frac = float(n_electrons - full)
    if full > 0:
        occ[:full] = 1.0
    if full < evals.size and frac > 0.0:
        occ[full] = frac
    return occ


def collinear_hartree_self_consistent(
    n_sites: int,
    n_electrons: float,
    gamma: float = 1.0,
    lambda_r: float = 0.0,
    phi_over_phi0: float = 0.0,
    u_hubbard: float = 0.0,
    onsite_up: np.ndarray | None = None,
    onsite_down: np.ndarray | None = None,
    mixing: float = 0.5,
    tol: float = 1e-8,
    max_iter: int = 500,
) -> CollinearHartreeResult:
    """
    Self-consistent collinear-Hartree on-site decoupling:
    E'_{n,up} = E_{n,up} + U<n_{n,down}>
    E'_{n,down} = E_{n,down} + U<n_{n,up}>
    """
    if not (0.0 < mixing <= 1.0):
        raise ValueError("mixing must satisfy 0 < mixing <= 1.")

    density_guess = np.clip(n_electrons / (2.0 * n_sites), 0.0, 1.0)
    n_up = np.full(n_sites, density_guess, dtype=float)
    n_down = np.full(n_sites, density_guess, dtype=float)

    converged = False
    h = None
    evals = None
    evecs = None
    occ = None

    for it in range(1, max_iter + 1):
        h = build_rashba_hubbard_ring_real_space(
            n_sites=n_sites,
            gamma=gamma,
            lambda_r=lambda_r,
            phi_over_phi0=phi_over_phi0,
            onsite_up=onsite_up,
            onsite_down=onsite_down,
            u_hubbard=u_hubbard,
            mean_n_up=n_up,
            mean_n_down=n_down,
        )
        evals, evecs = np.linalg.eigh(h)
        occ = _occupations_zero_temperature(evals, n_electrons=n_electrons)

        rho = (evecs * occ[None, :]) @ evecs.conj().T
        n_up_new = np.real(np.diag(rho)[0::2])
        n_down_new = np.real(np.diag(rho)[1::2])

        delta = max(np.max(np.abs(n_up_new - n_up)), np.max(np.abs(n_down_new - n_down)))
        n_up = mixing * n_up_new + (1.0 - mixing) * n_up
        n_down = mixing * n_down_new + (1.0 - mixing) * n_down

        if delta < tol:
            converged = True
            break

    return CollinearHartreeResult(
        hamiltonian=h,
        eigenvalues=evals,
        eigenvectors=evecs,
        n_up=n_up,
        n_down=n_down,
        occupations=occ,
        iterations=it,
        converged=converged,
    )


HartreeFockResult = CollinearHartreeResult
hartree_fock_self_consistent = collinear_hartree_self_consistent
