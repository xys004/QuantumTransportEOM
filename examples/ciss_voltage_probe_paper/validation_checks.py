"""Independent validation checks for the CISS voltage-probe paper.

Checks performed:
 1. Reproduce the headline number A_M(tau_plus, Gamma_p=0.8, chi=+1).
 2. Energy-grid convergence of the headline number (201 -> 2001 points).
 3. Exact symmetry of the coherent two-terminal transmission:
    T_LR(E, +M) == T_LR(E, -M) pointwise (Onsager + unitarity).
 4. Elastic (per-energy) probe: effective transmission even in M pointwise.
 5. Linear-response limit of the voltage probe: A_M -> 0 as bias -> 0.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).absolute().parents[2]
EXAMPLES = ROOT / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from ciss_ladder_elastic_probe import (
    retarded_green,
    solve_elastic_probe_occupations,
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
from ciss_rho_tau_sigma_voltage_probe import (
    integrate_voltage_probe_case,
    internal_probe_gamma,
)

MU_LEFT = 0.25
MU_RIGHT = -0.15
TEMPERATURE = 0.03

PARAMS = RhoTauSigmaParameters(chirality=+1, chain_detuning=0.50, channel_detuning=1.20)


def gammas_for(theta: float, gamma_probe: float, probe_kind: str):
    gamma_left = ferromagnetic_edge_gamma(PARAMS.n_sites, 0, 2.0, 0.65, theta=theta, phi=0.0)
    gamma_right = normal_edge_gamma(
        PARAMS.n_sites, PARAMS.n_sites - 1, 2.0,
        chain_weights=(1.0, 0.25), channel_weights=(1.0, 0.40),
    )
    gamma_p = internal_probe_gamma(PARAMS.n_sites, gamma_probe, kind=probe_kind)
    return [gamma_left, gamma_right, gamma_p]


def check_headline(n_grid: int) -> float:
    grid = np.linspace(-4.0, 4.0, n_grid)
    out = []
    for label, theta in (("+z", 0.0), ("-z", np.pi)):
        res = integrate_voltage_probe_case(
            PARAMS, gamma_probe=0.80, probe_kind="tau_plus",
            magnetization_label=label, theta=theta, phi=0.0,
            omega_grid=grid, mu_left=MU_LEFT, mu_right=MU_RIGHT,
            temperature=TEMPERATURE,
        )
        out.append(res)
    a = asymmetry(out[0].current, out[1].current)
    print(f"  n_grid={n_grid:5d}  I+={out[0].current:+.8e}  I-={out[1].current:+.8e}  "
          f"A_M={a:+.6e}  mu_p(+)={out[0].mu_probe:+.6f}  mu_p(-)={out[1].mu_probe:+.6f}  "
          f"res={max(out[0].residual, out[1].residual):.1e}")
    return a


def check_coherent_transmission_symmetry() -> float:
    """T_LR(E,+M) vs T_LR(E,-M) for the pure two-terminal device, pointwise in E."""
    device = build_rho_tau_sigma_ladder(PARAMS)
    energies = np.linspace(-4.0, 4.0, 161)
    worst = 0.0
    for energy in energies:
        t_vals = []
        for theta in (0.0, np.pi):
            gams = gammas_for(theta, 0.0, "all")[:2]
            g_r = retarded_green(float(energy), device, gams)
            t = transmission_matrix(g_r, gams)
            t_vals.append(t[0, 1])
        worst = max(worst, abs(t_vals[0] - t_vals[1]))
    print(f"  max_E |T_LR(E,+M) - T_LR(E,-M)| (coherent 2-terminal) = {worst:.3e}")
    return worst


def check_elastic_effective_transmission() -> float:
    """Per-energy dephasing probes: effective L->R current density even in M pointwise."""
    device = build_rho_tau_sigma_ladder(PARAMS)
    probes = [normal_edge_gamma(PARAMS.n_sites, site, 0.60) for site in range(1, PARAMS.n_sites - 1)]
    energies = np.linspace(-4.0, 4.0, 161)
    worst = 0.0
    for energy in energies:
        vals = []
        for theta in (0.0, np.pi):
            gamma_left = ferromagnetic_edge_gamma(PARAMS.n_sites, 0, 2.0, 0.65, theta=theta, phi=0.0)
            gamma_right = normal_edge_gamma(
                PARAMS.n_sites, PARAMS.n_sites - 1, 2.0,
                chain_weights=(1.0, 0.25), channel_weights=(1.0, 0.40),
            )
            gams = [gamma_left, gamma_right, *probes]
            g_r = retarded_green(float(energy), device, gams)
            t = transmission_matrix(g_r, gams)
            occ = solve_elastic_probe_occupations(t, f_left=1.0, f_right=0.0)
            vals.append(terminal_current_density(t, occ, 0))
        worst = max(worst, abs(vals[0] - vals[1]))
    print(f"  max_E |dI/dE(+M) - dI/dE(-M)| (elastic probes)        = {worst:.3e}")
    return worst


def check_bias_scan() -> None:
    """A_M as the bias window shrinks around the mean chemical potential."""
    grid = np.linspace(-4.0, 4.0, 801)
    mu_mean = 0.5 * (MU_LEFT + MU_RIGHT)
    full_bias = MU_LEFT - MU_RIGHT
    print("  bias/bias0   A_M            I+")
    for scale in (1.0, 0.5, 0.25, 0.1, 0.05, 0.02):
        mu_l = mu_mean + 0.5 * scale * full_bias
        mu_r = mu_mean - 0.5 * scale * full_bias
        out = []
        for theta in (0.0, np.pi):
            res = integrate_voltage_probe_case(
                PARAMS, gamma_probe=0.80, probe_kind="tau_plus",
                magnetization_label="x", theta=theta, phi=0.0,
                omega_grid=grid, mu_left=mu_l, mu_right=mu_r,
                temperature=TEMPERATURE,
            )
            out.append(res.current)
        print(f"  {scale:8.2f}   {asymmetry(out[0], out[1]):+.6e}  {out[0]:+.6e}")


def main() -> None:
    print("[1] Headline reproduction + grid convergence")
    for n in (201, 401, 801, 1601):
        check_headline(n)
    print("[2] Coherent two-terminal symmetry (pointwise in E)")
    check_coherent_transmission_symmetry()
    print("[3] Elastic probe symmetry (pointwise in E)")
    check_elastic_effective_transmission()
    print("[4] Bias scan (linear-response limit)")
    check_bias_scan()


if __name__ == "__main__":
    main()
