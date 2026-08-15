from __future__ import annotations

from scripts.verify_gate01_capability_inventory import run_gate


def test_gate01_capability_inventory_passes() -> None:
    report = run_gate()
    assert report["passed"]
    assert len(report["checks"]) == 5
