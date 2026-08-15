"""Gate 55: common two-time Meir--Wingreen charge/spin observable contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantum_transport import (  # noqa: E402
    equilibrium_one_body_density,
    two_time_greens,
    two_time_meir_wingreen_charge_spin_currents,
)


def _check(name: str, condition: bool, checks: dict[str, bool]) -> None:
    checks[name] = bool(condition)
    print(f"CHECK {name}: {'PASS' if condition else 'FAIL'}")


def run_gate() -> dict[str, object]:
    time = np.linspace(0.0, 0.6, 7)
    hamiltonian = np.array(
        [[0.2, 0.08 - 0.03j], [0.08 + 0.03j, -0.15]], dtype=complex
    )
    density = equilibrium_one_body_density(hamiltonian, mu=0.0, temperature=0.4)
    greens = two_time_greens(time, lambda _: hamiltonian, density)
    sigma_lesser = np.zeros_like(greens.retarded)
    sigma_lesser[:, :, 0, 0] = 0.03j
    sigma_lesser[:, :, 1, 1] = 0.02j
    sigma_lesser[:, :, 0, 1] = 0.01j
    sigma_lesser[:, :, 1, 0] = 0.01j
    sigma_advanced = np.zeros_like(greens.retarded)
    for index in range(time.size):
        sigma_advanced[index, index] = 0.1j * np.eye(2)
    channels = two_time_meir_wingreen_charge_spin_currents(
        time,
        green_retarded=greens.retarded,
        green_lesser=greens.lesser,
        lead_self_energy_lesser=sigma_lesser,
        lead_self_energy_advanced=sigma_advanced,
        spin_operators={
            "sx": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex) / 2.0,
            "sy": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex) / 2.0,
            "sz": np.diag([1.0, -1.0]).astype(complex) / 2.0,
        },
    )
    checks: dict[str, bool] = {}
    _check("charge_channel_is_present", "charge" in channels, checks)
    _check("three_spin_channels_are_present", {"sx", "sy", "sz"}.issubset(channels), checks)
    _check("all_channels_are_finite", all(np.all(np.isfinite(value)) for value in channels.values()), checks)
    _check("all_channels_share_time_grid", all(value.shape == time.shape for value in channels.values()), checks)
    _check("spin_channel_is_not_charge_alias", not np.allclose(channels["charge"], channels["sz"]), checks)
    report = {
        "gate": "GATE_55_CHARGE_SPIN_MEIR_WINGREEN",
        "checks": checks,
        "passed": all(checks.values()),
        "channel_maxima": {name: float(np.max(np.abs(value))) for name, value in channels.items()},
        "assessment": "PASS_SHARED_TWO_TIME_CHARGE_SPIN_CURRENT_CONTRACT",
        "claim_boundary": (
            "Charge and named spin observables now use the same two-time Keldysh "
            "contraction. Spin torque, reservoir partitioning, and any topological "
            "interpretation remain separate diagnostics."
        ),
    }
    return report


def main() -> None:
    report = run_gate()
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"VERDICT: {'PASS' if report['passed'] else 'FAIL'}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
