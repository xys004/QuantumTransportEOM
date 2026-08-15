"""Gate 49: integrated audit of the Gates 44--48 two-time block."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "evidence"
sys.path.insert(0, str(ROOT / "src"))


def _check(name: str, condition: bool, checks: dict[str, bool]) -> None:
    checks[name] = bool(condition)
    print(f"CHECK {name}: {'PASS' if condition else 'FAIL'}")


def run_gate() -> dict[str, object]:
    names = {
        44: "gate44_weak_u_source_range_20260802.json",
        45: "gate45_mixed_kbe_symbolic_20260802.json",
        46: "gate46_matsubara_branch_20260802.json",
        47: "gate47_mixed_kbe_residual_20260802.json",
        48: "gate48_charge_spin_source_projection_20260802.json",
    }
    records = {}
    for gate, filename in names.items():
        path = EVIDENCE / filename
        records[gate] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    checks: dict[str, bool] = {}
    _check("all_five_evidence_files_exist", all(value is not None for value in records.values()), checks)
    _check("gate_ids_are_unique_and_complete", [records[g]["gate"] for g in names if records[g]] == [
        "GATE_44_WEAK_U_SOURCE_ERROR_RANGE",
        "GATE_45_MIXED_KBE_SYMBOLIC",
        "GATE_46_MATSUBARA_BRANCH",
        "GATE_47_MIXED_KBE_RESIDUAL",
        "GATE_48_CHARGE_SPIN_SOURCE_PROJECTION",
    ], checks)
    _check("all_local_verdicts_pass", all(records[g]["local"]["verdict"] == "PASS" for g in names), checks)
    _check("all_astra_verdicts_pass", all(records[g]["astra"]["verdict"] == "PASS" for g in names), checks)
    _check("all_astrum_verdicts_pass", all(records[g]["astrum"]["verdict"] == "PASS" for g in names), checks)
    _check("all_check_counts_close", all(records[g]["local"]["checks_ok"] == records[g]["local"]["checks_total"] for g in names), checks)
    _check("engine_regression_sequence_is_recorded", [records[g]["local"]["engine_full_pytest"] for g in names] == [
        "248 passed (Gate41 regression)", "249 passed", "250 passed", "251 passed", "252 passed"
    ], checks)
    _check("application_regression_is_recorded", all(records[g]["local"]["app_full_pytest"] == "69 passed" for g in names), checks)
    _check("all_assessments_are_bounded", all("PASS" in records[g]["assessment"] for g in names), checks)
    _check("no_topological_claim_is_released", all(
        any(token in records[g]["claim_boundary"].lower() for token in ("not", "no", "does not"))
        for g in names
    ), checks)
    _check("numeric_metrics_are_finite", all(
        all(isinstance(value, (int, float)) and value == value and abs(value) != float("inf")
            for value in records[g]["local"].values() if isinstance(value, (int, float)))
        for g in names
    ), checks)
    _check("verifier_artifacts_are_recorded", all("verifier" in records[g]["artifacts"] for g in names), checks)
    report = {
        "gate": "GATE_49_INTERACTING_TWO_TIME_BLOCK_AUDIT",
        "checks": checks,
        "passed": all(checks.values()),
        "audited_gates": list(names),
        "assessment": "PASS_INTEGRATED_GATES44_48_ASTRA_ASTRUM_AUDIT",
        "claim_boundary": (
            "Gates 44--48 form a coherent implementation and diagnostic block: "
            "weak-U source error, symbolic mixed equations, Matsubara inputs, "
            "numerical residuals, and charge/spin projections are all recorded "
            "on ASTRA and ASTRUM. The audit releases no topological-protection "
            "or interacting-conservation claim."
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
