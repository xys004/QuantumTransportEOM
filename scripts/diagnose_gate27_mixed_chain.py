"""Gate 27 diagnosis: exact derivative versus the finite-lead IC branch.

This is intentionally a diagnostic, not yet a pass gate.  It compares the
finite-grid KBE collision integral against the exact derivative of the full
contacted quadratic system and records the mixed-branch source used by the
engine.  The same code is run on ASTRA and ASTRUM before deciding whether a
lead-size convergence gate is meaningful.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantum_transport import (  # noqa: E402
    partition_free_finite_lead_two_time,
    two_time_kbe_collision_integral,
)


def _weights(grid: np.ndarray) -> np.ndarray:
    result = np.empty_like(grid)
    result[0] = 0.5 * (grid[1] - grid[0])
    result[-1] = 0.5 * (grid[-1] - grid[-2])
    result[1:-1] = 0.5 * (grid[2:] - grid[:-2])
    return result


def _chain(size: int, *, spin_mixing: bool = True) -> np.ndarray:
    h = np.zeros((2 * size, 2 * size), dtype=complex)
    for site in range(size):
        block = 0.06 * np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
        if spin_mixing:
            block += 0.035 * np.array([[0.0, 1.0 - 0.2j], [1.0 + 0.2j, 0.0]], dtype=complex)
        h[2 * site : 2 * site + 2, 2 * site : 2 * site + 2] = block
        if site + 1 < size:
            hop = 0.22 * np.eye(2, dtype=complex)
            h[2 * site : 2 * site + 2, 2 * (site + 1) : 2 * (site + 1) + 2] = hop
            h[2 * (site + 1) : 2 * (site + 1) + 2, 2 * site : 2 * site + 2] = hop.conj().T
    return h


def _source_variants(result, *, complement: bool, positive_exponential: bool, sigma_sign: int) -> np.ndarray:
    t = result.time
    tau = result.imaginary_time
    beta = tau[-1] - tau[0]
    h0 = result.initial_hamiltonian
    hf = result.final_hamiltonian
    e0, v0 = np.linalg.eigh(h0)
    ef, vf = np.linalg.eigh(hf)
    f0 = result.initial_density
    proj = np.zeros((result.device_dimension, h0.shape[0]), dtype=complex)
    proj[:, : result.device_dimension] = np.eye(result.device_dimension)
    u_f = np.einsum("ik,tk,kj->tij", vf, np.exp(-1j * t[:, None] * ef[None, :]), vf.conj().T)
    exp0 = np.einsum("ik,tk,kj->tij", v0, np.exp(-tau[:, None] * e0[None, :]), v0.conj().T)
    lead_sources = []
    for h, v, shift in zip(result.lead_hamiltonians, result.coupling_matrices, result.lead_shifts):
        el, vl = np.linalg.eigh(h)
        uf = np.einsum(
            "ik,tk,kj->tij",
            vl,
            np.exp(-1j * t[:, None] * (el[None, :] + shift)),
            vl.conj().T,
            optimize=True,
        )
        sign = 1.0 if positive_exponential else -1.0
        ex = np.einsum(
            "ik,tk,kj->tij",
            vl,
            np.exp(sign * tau[:, None] * el[None, :]),
            vl.conj().T,
            optimize=True,
        )
        fl = vl @ np.diag(1.0 / (np.exp((el - 0.0) / (1.0 / beta)) + 1.0)) @ vl.conj().T
        source = np.eye(h.shape[0], dtype=complex) - fl if complement else fl
        lead_sources.append((uf, ex, source, v))
    sigma = np.zeros((t.size, tau.size, result.device_dimension, result.device_dimension), dtype=complex)
    green = np.zeros((tau.size, t.size, result.device_dimension, result.device_dimension), dtype=complex)
    occ = np.eye(h0.shape[0], dtype=complex) - f0 if complement else f0
    for a, (uf, ex, fl, v) in enumerate(lead_sources):
        for i in range(t.size):
            for k in range(tau.size):
                sigma[i, k] += sigma_sign * 1j * v @ uf[i] @ fl @ ex[k] @ v.conj().T
    for k in range(tau.size):
        for j in range(t.size):
            green[k, j] = -1j * proj @ exp0[k] @ occ @ u_f[j].conj().T @ proj.conj().T
    weights = _weights(tau)
    kernel = -1j * np.einsum("ikab,kjbc,k->ijac", sigma, green, weights, optimize=True)
    diag = kernel[np.arange(t.size), np.arange(t.size)]
    return diag + diag.swapaxes(-1, -2).conj()


def _run(size: int, *, spin_mixing: bool) -> dict[str, float]:
    t = np.linspace(0.0, 0.5, 81)
    tau = np.linspace(0.0, 3.0, 241)
    h_device_i = np.array([[0.14, 0.04 - 0.02j], [0.04 + 0.02j, -0.11]], dtype=complex)
    h_device_f = np.array([[0.06, 0.035 + 0.03j], [0.035 - 0.03j, -0.05]], dtype=complex)
    lead = _chain(size, spin_mixing=spin_mixing)
    # Couple both device spin orbitals to the first lead site.
    coupling = np.zeros((2, 2 * size), dtype=complex)
    coupling[:, :2] = np.array([[0.18, 0.025j], [0.012, 0.16]], dtype=complex)
    result = partition_free_finite_lead_two_time(
        t,
        tau,
        initial_device_hamiltonian=h_device_i,
        final_device_hamiltonian=h_device_f,
        lead_hamiltonians=(lead, lead.copy()),
        coupling_matrices=(coupling, coupling.conj()),
        lead_shifts=(0.18, -0.16),
        temperature=1.0 / 3.0,
    )
    collision = two_time_kbe_collision_integral(
        t,
        green_retarded=result.retarded,
        green_lesser=result.lesser,
        self_energy_retarded=result.self_energy_retarded,
        self_energy_lesser=result.self_energy_lesser,
        self_energy_advanced=result.self_energy_advanced,
    )
    cdiag = collision[np.arange(t.size), np.arange(t.size)]
    density = result.density_matrices
    full_rho = np.einsum("tik,kl,tjl->tij", np.array([]), np.array([]), np.array([])) if False else None
    e, v = np.linalg.eigh(result.final_hamiltonian)
    uf = np.einsum("ik,tk,kj->tij", v, np.exp(-1j * t[:, None] * e[None, :]), v.conj().T)
    full_rho = np.einsum("tik,kl,tjl->tij", uf, result.initial_density, uf.conj(), optimize=True)
    dr_full = np.empty_like(full_rho[:, :2, :2])
    dr_full[:] = -1j * (
        result.final_hamiltonian[:2, :] @ full_rho[:, :, :2]
        - full_rho[:, :2, :] @ result.final_hamiltonian[:, :2]
    )
    dr = 0.5 * (dr_full + dr_full.swapaxes(-1, -2).conj())
    rho = result.density_matrices
    coherent = -1j * (result.final_device_hamiltonian @ rho - rho @ result.final_device_hamiltonian)
    raw = dr - coherent - cdiag
    grad = np.gradient(rho, t, axis=0, edge_order=2)
    grad_error = grad - dr
    source = result.initial_correlation.density_source
    source_alt = _source_variants(result, complement=True, positive_exponential=True, sigma_sign=1)
    source_alt2 = _source_variants(result, complement=False, positive_exponential=True, sigma_sign=1)
    source_alt3 = _source_variants(result, complement=True, positive_exponential=False, sigma_sign=1)
    metrics = {
        "exact_raw_interior": float(np.max(np.abs(raw[3:-3]))),
        "finite_difference_rate_error_interior": float(np.max(np.abs(grad_error[3:-3]))),
        "gradient_raw_interior": float(np.max(np.abs((grad - coherent - cdiag)[3:-3]))),
        "exact_corrected_engine_interior": float(np.max(np.abs((raw + source)[3:-3]))),
        "exact_corrected_alt_complement_interior": float(np.max(np.abs((raw + source_alt)[3:-3]))),
        "exact_corrected_alt_occupation_interior": float(np.max(np.abs((raw + source_alt2)[3:-3]))),
        "exact_corrected_alt_negative_exponential_interior": float(np.max(np.abs((raw + source_alt3)[3:-3]))),
        "source_engine_max": float(np.max(np.abs(source))),
        "initial_cross": float(np.max(np.abs(result.initial_density[:2, 2:]))),
    }
    return metrics


def main() -> None:
    report = {
        "chain_spin_mixing": _run(2, spin_mixing=True),
        "chain_spin_conserving": _run(2, spin_mixing=False),
        "larger_spin_mixing": _run(3, spin_mixing=True),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
