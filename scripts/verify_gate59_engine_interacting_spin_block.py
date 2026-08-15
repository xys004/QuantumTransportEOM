"""Gate 59: integrated ASTRA/ASTRUM audit of Gates54–55."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from verify_gate54_lesser_vertical_closure import run_gate as run_gate54
from verify_gate55_charge_spin_meir_wingreen import run_gate as run_gate55


def run_gate() -> dict[str, object]:
    gate54 = run_gate54()
    gate55 = run_gate55()
    checks = {
        "gate54_passes": bool(gate54["passed"]),
        "gate55_passes": bool(gate55["passed"]),
        "negative_lesser_closure_is_retained": float(gate54["residual_ratio"]) > 1.0,
        "spin_channels_are_resolved": float(gate55["channel_maxima"]["sz"]) > 0.0,
    }
    return {
        "gate": "GATE_59_ENGINE_INTERACTING_SPIN_BLOCK",
        "checks": checks,
        "passed": all(checks.values()),
        "gate54": gate54,
        "gate55": gate55,
        "assessment": "PASS_INTEGRATED_ENGINE_INTERACTING_SPIN_AUDIT",
        "claim_boundary": (
            "The engine block is reproducible and exposes both the negative lesser "
            "closure result and named spin currents. It does not claim a conserving "
            "contour solution or topological protection."
        ),
    }


def main() -> None:
    report = run_gate()
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"VERDICT: {'PASS' if report['passed'] else 'FAIL'}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
