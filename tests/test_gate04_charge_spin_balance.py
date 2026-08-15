from __future__ import annotations

from scripts.verify_gate04_charge_spin_balance import run_gate


def test_gate04_charge_spin_balance() -> None:
    report = run_gate()
    assert report["passed"], report
