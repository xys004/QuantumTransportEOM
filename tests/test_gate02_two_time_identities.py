from __future__ import annotations

from scripts.verify_gate02_two_time_identities import run_gate


def test_gate02_two_time_identities_pass() -> None:
    report = run_gate()
    assert report["passed"]
    assert len(report["checks"]) == 4
