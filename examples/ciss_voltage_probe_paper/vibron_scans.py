"""SCBA vibron scans for the paper: A_M vs coupling g and vs mode frequency omega0.

Outputs: data/vibron_g_scan.csv, data/vibron_omega_scan.csv, data/vibron_controls.csv
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

from ciss_rho_tau_sigma_ladder import RhoTauSigmaParameters
from ciss_vibron_scba import magnetization_pair

OUT = Path(__file__).absolute().parent
DATA = OUT / "data"
DATA.mkdir(exist_ok=True)

GRID = np.linspace(-5.0, 5.0, 1001)
BASE = dict(chirality=+1, chain_detuning=0.50, channel_detuning=1.20)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path.name}", flush=True)


def g_scan() -> None:
    rows = []
    params = RhoTauSigmaParameters(**BASE)
    for g in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
        plus, minus, a_m = magnetization_pair(
            params, g_vibron=g, omega0=0.20, coupling_kind="tau_plus", grid=GRID,
        )
        rows.append({
            "g": g, "A_current": a_m,
            "I_plus": plus.current_left, "I_minus": minus.current_left,
            "conservation": max(plus.conservation, minus.conservation),
            "iters": max(plus.iterations, minus.iterations),
        })
        print(f"g={g:.2f}  A={a_m:+.4e}  cons={rows[-1]['conservation']:.1e}", flush=True)
    write_csv(DATA / "vibron_g_scan.csv", rows)


def omega_scan() -> None:
    rows = []
    params = RhoTauSigmaParameters(**BASE)
    for omega0 in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60):
        plus, minus, a_m = magnetization_pair(
            params, g_vibron=0.30, omega0=omega0, coupling_kind="tau_plus", grid=GRID,
        )
        rows.append({
            "omega0": omega0, "A_current": a_m,
            "I_plus": plus.current_left, "I_minus": minus.current_left,
            "conservation": max(plus.conservation, minus.conservation),
            "iters": max(plus.iterations, minus.iterations),
        })
        print(f"w0={omega0:.2f}  A={a_m:+.4e}  cons={rows[-1]['conservation']:.1e}", flush=True)
    write_csv(DATA / "vibron_omega_scan.csv", rows)


def controls() -> None:
    rows = []
    cases = [
        ("vibron_tau_plus", RhoTauSigmaParameters(**BASE), 0.30, "tau_plus", 0.65),
        ("vibron_chi_flip", RhoTauSigmaParameters(chirality=-1, chain_detuning=0.50, channel_detuning=1.20), 0.30, "tau_plus", 0.65),
        ("vibron_lambda0", RhoTauSigmaParameters(**BASE, lambda_soc=0.0), 0.30, "tau_plus", 0.65),
        ("vibron_pFM0", RhoTauSigmaParameters(**BASE), 0.30, "tau_plus", 0.0),
        ("vibron_g0", RhoTauSigmaParameters(**BASE), 0.0, "tau_plus", 0.65),
        ("vibron_uniform", RhoTauSigmaParameters(**BASE), 0.30, "all", 0.65),
    ]
    for name, params, g, kind, pol in cases:
        plus, minus, a_m = magnetization_pair(
            params, g_vibron=g, omega0=0.20, coupling_kind=kind,
            polarization=pol, grid=GRID,
        )
        rows.append({
            "case": name, "chirality": params.chirality, "g": g,
            "coupling": kind, "polarization": pol, "A_current": a_m,
            "I_plus": plus.current_left, "I_minus": minus.current_left,
            "conservation": max(plus.conservation, minus.conservation),
        })
        print(f"{name:<18s} A={a_m:+.4e}", flush=True)
    write_csv(DATA / "vibron_controls.csv", rows)


if __name__ == "__main__":
    g_scan()
    omega_scan()
    controls()
