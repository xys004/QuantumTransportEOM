"""Self-consistent Born (SCBA) electron-vibron NEGF for the rho/tau/sigma CISS ladder.

Microscopic check of the voltage-probe mechanism: a local Einstein mode of
frequency omega0, coupled to interior rungs through a (possibly channel-selective)
Hermitian operator V = g * Pi, generates an energy-redistributing self-energy

    Sigma^<_ph(E) = V [ N G^<(E - w0) + (N+1) G^<(E + w0) ] V
    Sigma^>_ph(E) = V [ N G^>(E + w0) + (N+1) G^>(E - w0) ] V
    Sigma^r_ph(E) = (Sigma^> - Sigma^<)(E) / 2      (real part neglected)

iterated to self-consistency.  Charge conservation (|I_L + I_R| after energy
integration) is monitored and reported; it vanishes in the converged limit up to
the neglected real part of Sigma^r.

The magnetocurrent asymmetry A_M = (I(+M) - I(-M)) / (|I(+M)| + |I(-M)|) is the
observable of interest.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ciss_rho_tau_sigma_ladder import (
    RhoTauSigmaParameters,
    asymmetry,
    build_rho_tau_sigma_ladder,
    ferromagnetic_edge_gamma,
    normal_edge_gamma,
)
from ciss_rho_tau_sigma_voltage_probe import fermi, internal_probe_gamma


@dataclass(frozen=True)
class ScbaResult:
    current_left: float
    current_right: float
    conservation: float          # |I_L + I_R|
    iterations: int
    converged: bool
    max_update: float


def scba_currents(
    params: RhoTauSigmaParameters,
    *,
    g_vibron: float,
    omega0: float,
    coupling_kind: str = "tau_plus",
    polarization: float = 0.65,
    theta: float = 0.0,
    mu_left: float = 0.25,
    mu_right: float = -0.15,
    temperature: float = 0.03,
    grid: np.ndarray,
    max_iter: int = 200,
    mixing: float = 0.6,
    tol: float = 1e-11,
) -> ScbaResult:
    """Run the SCBA loop for one magnetization and return terminal currents."""
    n_grid = grid.size
    de = float(grid[1] - grid[0])
    shift = int(round(omega0 / de))
    if abs(shift * de - omega0) > 1e-12:
        raise ValueError("omega0 must be an integer multiple of the grid spacing.")

    device = build_rho_tau_sigma_ladder(params)
    dim = device.hamiltonian.shape[0]

    gamma_left = ferromagnetic_edge_gamma(params.n_sites, 0, 2.0, polarization, theta=theta, phi=0.0)
    gamma_right = normal_edge_gamma(
        params.n_sites, params.n_sites - 1, 2.0,
        chain_weights=(1.0, 0.25), channel_weights=(1.0, 0.40),
    )
    # coupling operator: same spatial/channel structure as the voltage probe
    coupling = internal_probe_gamma(params.n_sites, 1.0, kind=coupling_kind)
    coupling = g_vibron * coupling  # Hermitian, positive

    f_left = np.asarray(fermi(grid, mu=mu_left, temperature=temperature))
    f_right = np.asarray(fermi(grid, mu=mu_right, temperature=temperature))
    n_bose = 1.0 / np.expm1(omega0 / temperature) if omega0 / temperature < 500 else 0.0

    sig_less_leads = np.empty((n_grid, dim, dim), dtype=np.complex128)
    sig_grtr_leads = np.empty((n_grid, dim, dim), dtype=np.complex128)
    for k in range(n_grid):
        sig_less_leads[k] = 1.0j * (f_left[k] * gamma_left + f_right[k] * gamma_right)
        sig_grtr_leads[k] = -1.0j * ((1.0 - f_left[k]) * gamma_left + (1.0 - f_right[k]) * gamma_right)

    sigma_r_leads = -0.5j * (gamma_left + gamma_right)
    identity = np.eye(dim, dtype=np.complex128)

    sig_less_ph = np.zeros((n_grid, dim, dim), dtype=np.complex128)
    sig_grtr_ph = np.zeros((n_grid, dim, dim), dtype=np.complex128)

    def shifted(array: np.ndarray, offset: int) -> np.ndarray:
        """array[k + offset] with zero padding outside the grid."""
        out = np.zeros_like(array)
        if offset >= 0:
            out[: n_grid - offset] = array[offset:]
        else:
            out[-offset:] = array[: n_grid + offset]
        return out

    g_less = np.zeros((n_grid, dim, dim), dtype=np.complex128)
    g_grtr = np.zeros((n_grid, dim, dim), dtype=np.complex128)

    converged = False
    max_update = np.inf
    for iteration in range(1, max_iter + 1):
        g_less_old = g_less.copy()
        # solve Dyson + Keldysh with current phonon self-energy
        for k in range(n_grid):
            sigma_r_ph = 0.5 * (sig_grtr_ph[k] - sig_less_ph[k])
            g_r = np.linalg.inv(grid[k] * identity - device.hamiltonian - sigma_r_leads - sigma_r_ph)
            g_a = g_r.conj().T
            g_less[k] = g_r @ (sig_less_leads[k] + sig_less_ph[k]) @ g_a
            g_grtr[k] = g_r @ (sig_grtr_leads[k] + sig_grtr_ph[k]) @ g_a

        # update phonon self-energies (Einstein mode -> grid shifts)
        g_less_m = shifted(g_less, -shift)   # G<(E - w0)
        g_less_p = shifted(g_less, +shift)   # G<(E + w0)
        g_grtr_m = shifted(g_grtr, -shift)
        g_grtr_p = shifted(g_grtr, +shift)

        new_less = coupling @ (n_bose * g_less_m + (n_bose + 1.0) * g_less_p) @ coupling
        new_grtr = coupling @ (n_bose * g_grtr_p + (n_bose + 1.0) * g_grtr_m) @ coupling

        sig_less_ph = mixing * new_less + (1.0 - mixing) * sig_less_ph
        sig_grtr_ph = mixing * new_grtr + (1.0 - mixing) * sig_grtr_ph

        max_update = float(np.max(np.abs(g_less - g_less_old)))
        if max_update < tol:
            converged = True
            break

    # Meir-Wingreen terminal currents: I_a = (1/2pi) Int Tr[Sig<_a G> - Sig>_a G<]
    def terminal_current(gamma: np.ndarray, f_lead: np.ndarray) -> float:
        density = np.empty(n_grid, dtype=float)
        for k in range(n_grid):
            sl = 1.0j * f_lead[k] * gamma
            sg = -1.0j * (1.0 - f_lead[k]) * gamma
            density[k] = float(np.real(np.trace(sl @ g_grtr[k] - sg @ g_less[k]))) / (2.0 * np.pi)
        return float(np.trapezoid(density, grid))

    current_left = terminal_current(gamma_left, f_left)
    current_right = terminal_current(gamma_right, f_right)

    return ScbaResult(
        current_left=current_left,
        current_right=current_right,
        conservation=abs(current_left + current_right),
        iterations=iteration,
        converged=converged,
        max_update=max_update,
    )


def magnetization_pair(params: RhoTauSigmaParameters, **kwargs) -> tuple[ScbaResult, ScbaResult, float]:
    plus = scba_currents(params, theta=0.0, **kwargs)
    minus = scba_currents(params, theta=np.pi, **kwargs)
    return plus, minus, asymmetry(plus.current_left, minus.current_left)


def run_demo() -> None:
    grid = np.linspace(-5.0, 5.0, 1001)
    omega0 = 0.20
    base = dict(chirality=+1, chain_detuning=0.50, channel_detuning=1.20)

    print("SCBA electron-vibron check (omega0=0.2, T=0.03, tau_plus coupling)")
    print("case            g     A_M            I_L(+M)        |I_L+I_R|   iters")
    cases = [
        ("candidate", RhoTauSigmaParameters(**base), 0.30, "tau_plus", 0.65),
        ("chi_flip", RhoTauSigmaParameters(chirality=-1, chain_detuning=0.50, channel_detuning=1.20), 0.30, "tau_plus", 0.65),
        ("lambda0", RhoTauSigmaParameters(**base, lambda_soc=0.0), 0.30, "tau_plus", 0.65),
        ("pFM0", RhoTauSigmaParameters(**base), 0.30, "tau_plus", 0.0),
        ("g0", RhoTauSigmaParameters(**base), 0.0, "tau_plus", 0.65),
        ("uniform", RhoTauSigmaParameters(**base), 0.30, "all", 0.65),
    ]
    for name, params, g, kind, pol in cases:
        plus, minus, a_m = magnetization_pair(
            params, g_vibron=g, omega0=omega0, coupling_kind=kind,
            polarization=pol, grid=grid,
        )
        print(
            f"{name:<12s}  {g:4.2f}  {a_m:+.6e}  {plus.current_left:+.6e}  "
            f"{plus.conservation:.2e}  {plus.iterations}/{minus.iterations}"
        )


if __name__ == "__main__":
    run_demo()
