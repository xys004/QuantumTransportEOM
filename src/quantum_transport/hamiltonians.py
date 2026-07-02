"""Hamiltonian builders and basis transforms."""

from __future__ import annotations

import numpy as np


SIGMA_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
SIGMA_Y = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=np.complex128)
IDENTITY_2 = np.eye(2, dtype=np.complex128)


def _site_spin_index(site: int, spin: int) -> int:
    return 2 * site + spin


def build_rashba_hubbard_ring_real_space(
    n_sites: int,
    gamma: float = 1.0,
    lambda_r: float = 0.0,
    phi_over_phi0: float = 0.0,
    onsite_up: np.ndarray | None = None,
    onsite_down: np.ndarray | None = None,
    u_hubbard: float = 0.0,
    mean_n_up: np.ndarray | None = None,
    mean_n_down: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build the spinful collinear-Hartree Rashba-Hubbard ring Hamiltonian in real space.

    Basis ordering: (0,up), (0,down), (1,up), (1,down), ...
    """
    if n_sites < 2:
        raise ValueError("n_sites must be >= 2.")

    onsite_up = np.zeros(n_sites) if onsite_up is None else np.asarray(onsite_up, dtype=float)
    onsite_down = np.zeros(n_sites) if onsite_down is None else np.asarray(onsite_down, dtype=float)
    mean_n_up = np.zeros(n_sites) if mean_n_up is None else np.asarray(mean_n_up, dtype=float)
    mean_n_down = np.zeros(n_sites) if mean_n_down is None else np.asarray(mean_n_down, dtype=float)
    if onsite_up.shape != (n_sites,) or onsite_down.shape != (n_sites,):
        raise ValueError("onsite_up/down must have shape (n_sites,).")
    if mean_n_up.shape != (n_sites,) or mean_n_down.shape != (n_sites,):
        raise ValueError("mean_n_up/down must have shape (n_sites,).")

    dim = 2 * n_sites
    h = np.zeros((dim, dim), dtype=np.complex128)

    e_up = onsite_up + u_hubbard * mean_n_down
    e_down = onsite_down + u_hubbard * mean_n_up
    for n in range(n_sites):
        h[_site_spin_index(n, 0), _site_spin_index(n, 0)] = e_up[n]
        h[_site_spin_index(n, 1), _site_spin_index(n, 1)] = e_down[n]

    theta = 2.0 * np.pi * phi_over_phi0 / float(n_sites)
    peierls = np.exp(1.0j * theta)

    for n in range(n_sites):
        m = (n + 1) % n_sites
        phi_n = 2.0 * np.pi * n / float(n_sites)
        phi_m = 2.0 * np.pi * m / float(n_sites)
        phi_bar = 0.5 * (phi_n + phi_m)

        rashba_matrix = -1.0j * lambda_r * (np.cos(phi_bar) * SIGMA_X + np.sin(phi_bar) * SIGMA_Y)
        forward_hop = peierls * (gamma * IDENTITY_2 + rashba_matrix)

        n_slice = slice(2 * n, 2 * n + 2)
        m_slice = slice(2 * m, 2 * m + 2)
        h[m_slice, n_slice] += forward_hop
        h[n_slice, m_slice] += forward_hop.conj().T

    return h


def real_to_k_space(h_real: np.ndarray, n_sites: int, spin_dim: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """
    Transform ``h_real`` from site basis to reciprocal (k) basis via DFT.
    """
    h_real = np.asarray(h_real, dtype=np.complex128)
    dim = n_sites * spin_dim
    if h_real.shape != (dim, dim):
        raise ValueError(f"h_real must have shape ({dim}, {dim}).")

    indices = np.arange(n_sites)
    k_values = 2.0 * np.pi * np.arange(n_sites) / float(n_sites)
    f = np.exp(-1.0j * np.outer(indices, indices) * (2.0 * np.pi / n_sites)) / np.sqrt(n_sites)
    u = np.kron(f, np.eye(spin_dim, dtype=np.complex128))
    h_k = u.conj().T @ h_real @ u
    return h_k, k_values


def split_k_blocks(
    h_k: np.ndarray,
    n_sites: int,
    spin_dim: int = 2,
    *,
    require_block_diagonal: bool = True,
    atol: float = 1e-10,
) -> list[np.ndarray]:
    """
    Split a k-space Hamiltonian into contiguous spin blocks of size ``spin_dim``.

    The simple DFT used by :func:`real_to_k_space` only produces independent
    k-blocks when the transformed Hamiltonian is block diagonal. Rashba terms
    with site-dependent spin texture generally couple different k sectors in
    this basis, so by default this function refuses to return misleading
    independent blocks.
    """
    h_k = np.asarray(h_k, dtype=np.complex128)
    dim = n_sites * spin_dim
    if h_k.shape != (dim, dim):
        raise ValueError(f"h_k must have shape ({dim}, {dim}).")
    if require_block_diagonal:
        off_block = h_k.copy()
        for i in range(n_sites):
            sl = slice(i * spin_dim, (i + 1) * spin_dim)
            off_block[sl, sl] = 0.0
        scale = max(1.0, float(np.linalg.norm(h_k)))
        off_norm = float(np.linalg.norm(off_block))
        if off_norm > atol * scale:
            raise ValueError(
                "h_k is not block diagonal in contiguous k-spin sectors "
                f"(off-block norm {off_norm:.3e}, tolerance {atol * scale:.3e}). "
                "Use require_block_diagonal=False only to inspect diagonal subblocks."
            )
    blocks = []
    for i in range(n_sites):
        sl = slice(i * spin_dim, (i + 1) * spin_dim)
        blocks.append(h_k[sl, sl].copy())
    return blocks
