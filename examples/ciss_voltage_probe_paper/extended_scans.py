"""Extended parameter scans for the journal version of the CISS voltage-probe paper.

Outputs (data/):
  bias_scan.csv          A_M and Delta I vs bias (both signs) -> linear-response onset
  lambda_scan.csv        A_M vs chiral SOC strength lambda
  polarization_scan.csv  A_M vs FM polarization p_FM
  length_scan.csv        A_M vs number of sites N
  temperature_scan.csv   A_M vs probe temperature T
  detuning_map_fine.csv  9x9 detuning map (tau_plus probe, Gamma_p=0.8)
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).absolute().parents[2]
EXAMPLES = ROOT / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from ciss_rho_tau_sigma_ladder import RhoTauSigmaParameters, asymmetry
from ciss_rho_tau_sigma_voltage_probe import integrate_voltage_probe_case

OUT = Path(__file__).absolute().parent
DATA = OUT / "data"
DATA.mkdir(exist_ok=True)

GRID = np.linspace(-4.0, 4.0, 401)
MU_MEAN = 0.05
BIAS0 = 0.40  # mu_L - mu_R of the reference point (0.25 / -0.15)
TEMPERATURE = 0.03

BASE = dict(chirality=+1, chain_detuning=0.50, channel_detuning=1.20)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path.name} ({len(rows)} rows)", flush=True)


def pair(params, *, mu_left, mu_right, temperature=TEMPERATURE, polarization=0.65,
         gamma_probe=0.80, probe_kind="tau_plus", grid=GRID):
    out = []
    for label, theta in (("+z", 0.0), ("-z", np.pi)):
        res = integrate_voltage_probe_case(
            params, gamma_probe=gamma_probe, probe_kind=probe_kind,
            polarization=polarization, magnetization_label=label,
            theta=theta, phi=0.0, omega_grid=grid,
            mu_left=mu_left, mu_right=mu_right, temperature=temperature,
        )
        out.append(res)
    plus, minus = out
    return {
        "I_plus": plus.current,
        "I_minus": minus.current,
        "Delta_I": plus.current - minus.current,
        "A_current": asymmetry(plus.current, minus.current),
        "Iz_plus": plus.spin_current_z,
        "Iz_minus": minus.spin_current_z,
        "mu_probe_plus": plus.mu_probe,
        "mu_probe_minus": minus.mu_probe,
        "residual": max(plus.residual, minus.residual),
    }


def bias_scan() -> None:
    rows = []
    params = RhoTauSigmaParameters(**BASE)
    for scale in (-1.0, -0.75, -0.5, -0.25, -0.1, -0.05, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0):
        bias = scale * BIAS0
        result = pair(params, mu_left=MU_MEAN + 0.5 * bias, mu_right=MU_MEAN - 0.5 * bias)
        rows.append({"bias": bias, **result})
    write_csv(DATA / "bias_scan.csv", rows)


def lambda_scan() -> None:
    rows = []
    for lam in (0.0, 0.02, 0.04, 0.06, 0.09, 0.12, 0.16, 0.20, 0.24):
        params = RhoTauSigmaParameters(**BASE, lambda_soc=lam)
        result = pair(params, mu_left=MU_MEAN + 0.5 * BIAS0, mu_right=MU_MEAN - 0.5 * BIAS0)
        rows.append({"lambda_soc": lam, **result})
    write_csv(DATA / "lambda_scan.csv", rows)


def polarization_scan() -> None:
    rows = []
    params = RhoTauSigmaParameters(**BASE)
    for pol in (0.0, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95):
        result = pair(params, mu_left=MU_MEAN + 0.5 * BIAS0, mu_right=MU_MEAN - 0.5 * BIAS0,
                      polarization=pol)
        rows.append({"polarization": pol, **result})
    write_csv(DATA / "polarization_scan.csv", rows)


def length_scan() -> None:
    rows = []
    for n_sites in (4, 5, 6, 7, 8, 10, 12):
        params = RhoTauSigmaParameters(**BASE, n_sites=n_sites)
        result = pair(params, mu_left=MU_MEAN + 0.5 * BIAS0, mu_right=MU_MEAN - 0.5 * BIAS0)
        rows.append({"n_sites": n_sites, **result})
    write_csv(DATA / "length_scan.csv", rows)


def temperature_scan() -> None:
    rows = []
    params = RhoTauSigmaParameters(**BASE)
    for temp in (0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20):
        result = pair(params, mu_left=MU_MEAN + 0.5 * BIAS0, mu_right=MU_MEAN - 0.5 * BIAS0,
                      temperature=temp)
        rows.append({"temperature": temp, **result})
    write_csv(DATA / "temperature_scan.csv", rows)


def detuning_map_fine() -> None:
    rows = []
    values = np.linspace(0.0, 1.4, 8)
    for chain in values:
        for channel in values:
            params = RhoTauSigmaParameters(
                chirality=+1, chain_detuning=float(chain), channel_detuning=float(channel)
            )
            result = pair(params, mu_left=MU_MEAN + 0.5 * BIAS0, mu_right=MU_MEAN - 0.5 * BIAS0,
                          grid=np.linspace(-4.0, 4.0, 201))
            rows.append({"chain_detuning": float(chain), "channel_detuning": float(channel), **result})
    write_csv(DATA / "detuning_map_fine.csv", rows)


if __name__ == "__main__":
    bias_scan()
    lambda_scan()
    polarization_scan()
    length_scan()
    temperature_scan()
    detuning_map_fine()
    print("done")
