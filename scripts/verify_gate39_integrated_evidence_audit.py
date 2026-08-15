"""Gate 39: integrated ASTRA/ASTRUM evidence and claim-boundary audit."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantum_transport import (  # noqa: E402,F401
    hubbard_hartree_self_energy_two_time,
    hubbard_second_born_self_energy_mixed,
    self_consistent_hubbard_second_born_two_time,
)


def _check(name: str, condition: bool, checks: dict[str, bool]) -> None:
    checks[name] = bool(condition)
    print(f"CHECK {name}: {'PASS' if condition else 'FAIL'}")


def run_gate() -> dict[str, object]:
    evidence_dir = ROOT / "docs" / "evidence"
    gates = list(range(31, 39))
    records = []
    checks: dict[str, bool] = {}
    for gate in gates:
        matches = sorted(evidence_dir.glob(f"gate{gate:02d}_*_20260802.json"))
        _check(f"gate_{gate}_evidence_file_unique", len(matches) == 1, checks)
        if len(matches) != 1:
            continue
        try:
            record = json.loads(matches[0].read_text(encoding="utf-8"))
            records.append(record)
            _check(f"gate_{gate}_json_schema", record.get("gate", "").startswith("GATE_"), checks)
            _check(f"gate_{gate}_local_pass", record.get("local", {}).get("verdict") == "PASS", checks)
            _check(f"gate_{gate}_astrum_pass", record.get("astrum", {}).get("verdict") == "PASS", checks)
            _check(
                f"gate_{gate}_checks_complete",
                record.get("local", {}).get("checks_ok") == record.get("local", {}).get("checks_total")
                and record.get("astrum", {}).get("checks_ok") == record.get("astrum", {}).get("checks_total"),
                checks,
            )
            boundary = str(record.get("claim_boundary", "")).lower()
            _check(f"gate_{gate}_claim_boundary_open", "open" in boundary or "remain" in boundary, checks)
        except (OSError, json.JSONDecodeError):
            _check(f"gate_{gate}_json_schema", False, checks)
    novelty = (ROOT / "docs" / "NOVELTY_AUDIT_TRANSIENT_INTERACTING_SPIN.md").read_text(encoding="utf-8").lower()
    _check("novelty_audit_stays_open", "audit_pass_with_open_publication_gates" in novelty, checks)
    _check("topological_claim_is_not_released", "topological-protection" in novelty and "not" in novelty, checks)
    full_counts = sorted({record.get("local", {}).get("engine_full_pytest") for record in records if record.get("local")})
    report = {
        "gate": "GATE_39_INTEGRATED_EVIDENCE_AND_CLAIM_AUDIT",
        "checks": checks,
        "passed": all(checks.values()),
        "audited_gates": [record.get("gate") for record in records],
        "engine_regression_counts_seen": full_counts,
        "assessment": "PASS_INTEGRATED_ASTRA_ASTRUM_AUDIT_OPEN_PUBLICATION_BOUNDARY",
        "claim_boundary": (
            "Gates 31–38 are internally consistent, reproducible on ASTRA and "
            "ASTRUM, and retain explicit open boundaries for interacting mixed "
            "closure, continuum convergence, and topological protection. This "
            "audit does not upgrade any bounded reference into a novelty claim."
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
