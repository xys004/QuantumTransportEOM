"""Odd-in-V magnetocurrent from electrostatic (capacitive) asymmetry.

The molecular levels ride the electrostatic potential U(V) = eta_L mu_L + eta_R mu_R
(eta_L + eta_R = 1).  For symmetric coupling (eta_L = 1/2) the probe-generated
magnetocurrent difference Delta I(V) is even in V; capacitive asymmetry
(eta_L > 1/2, molecule pinned to the source) generates an odd-in-V component,
the parity observed experimentally.

Output: data/electrostatic_bias_scan.csv
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
TEMPERATURE = 0.03


def delta_current(eta_left: float, bias: float) -> dict[str, float]:
    mu_left = MU_MEAN + 0.5 * bias
    mu_right = MU_MEAN - 0.5 * bias
    onsite = eta_left * mu_left + (1.0 - eta_left) * mu_right
    params = RhoTauSigmaParameters(
        chirality=+1, chain_detuning=0.50, channel_detuning=1.20, onsite=onsite
    )
    currents = []
    for theta in (0.0, np.pi):
        res = integrate_voltage_probe_case(
            params, gamma_probe=0.80, probe_kind="tau_plus",
            polarization=0.65, magnetization_label="pm",
            theta=theta, phi=0.0, omega_grid=GRID,
            mu_left=mu_left, mu_right=mu_right, temperature=TEMPERATURE,
        )
        currents.append(res.current)
    return {
        "I_plus": currents[0],
        "I_minus": currents[1],
        "Delta_I": currents[0] - currents[1],
        "A_current": asymmetry(currents[0], currents[1]),
    }


def main() -> None:
    rows = []
    biases = [0.05, 0.10, 0.20, 0.30, 0.40]
    for eta_left in (0.50, 0.65, 0.80):
        for bias in biases:
            fwd = delta_current(eta_left, +bias)
            bwd = delta_current(eta_left, -bias)
            delta_odd = 0.5 * (fwd["Delta_I"] - bwd["Delta_I"])
            delta_even = 0.5 * (fwd["Delta_I"] + bwd["Delta_I"])
            rows.append({
                "eta_left": eta_left,
                "bias": bias,
                "Delta_I_fwd": fwd["Delta_I"],
                "Delta_I_bwd": bwd["Delta_I"],
                "Delta_I_odd": delta_odd,
                "Delta_I_even": delta_even,
                "A_fwd": fwd["A_current"],
                "A_bwd": bwd["A_current"],
            })
            print(
                f"eta_L={eta_left:.2f}  V={bias:.2f}  "
                f"dI_odd={delta_odd:+.3e}  dI_even={delta_even:+.3e}  "
                f"A(+V)={fwd['A_current']:+.3e}  A(-V)={bwd['A_current']:+.3e}",
                flush=True,
            )
    with (DATA / "electrostatic_bias_scan.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote electrostatic_bias_scan.csv")


if __name__ == "__main__":
    main()
