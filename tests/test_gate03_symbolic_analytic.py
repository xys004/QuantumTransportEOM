from __future__ import annotations

from scripts.verify_gate03_symbolic_analytic import run_gate


def test_gate03_symbolic_analytic_passes() -> None:
    report = run_gate()
    assert report["passed"]
    assert len(report["checks"]) == 6
