"""Gate 69: integrated ASTRA/ASTRUM evidence and API audit for Gates61--68."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _check(name: str, condition: bool, checks: dict[str, bool]) -> None:
    checks[name] = bool(condition)
    print(f"CHECK {name}: {'PASS' if condition else 'FAIL'}")


def _app_root() -> Path:
    candidates = [
        ROOT.parent / "project",
        ROOT.parent.parent / "physics" / "xene-ring-transport",
        Path(r"C:\Users\Nelson\Dev\physics\xene-ring-transport"),
    ]
    for candidate in candidates:
        if (candidate / "docs" / "evidence").is_dir():
            return candidate
    raise FileNotFoundError("xene-ring-transport app evidence directory was not found.")


def _load(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"evidence file {path} is not a JSON object.")
    return value


def run_gate() -> dict[str, object]:
    engine_evidence = ROOT / "docs" / "evidence"
    app_evidence = _app_root() / "docs" / "evidence"
    engine_files = [
        engine_evidence / "gate61_total_self_energy_balance_20260802.json",
        engine_evidence / "gate62_full_contour_lesser_reconstruction_20260802.json",
        engine_evidence / "gate63_full_contour_solver_option_20260802.json",
        engine_evidence / "gate64_exact_contact_charge_spin_20260802.json",
        engine_evidence / "gate65_mixed_grid_lead_size_convergence_20260802.json",
    ]
    app_files = [
        app_evidence / "gate66_hubbard_corbino_closed_branch_20260802.json",
        app_evidence / "gate67_closed_persistent_reservoir_20260802.json",
        app_evidence / "gate68_closed_topology_criteria_20260802.json",
    ]
    evidence = [_load(path) for path in engine_files + app_files]
    checks: dict[str, bool] = {}
    _check("all_gate_evidence_files_exist", all(path.is_file() for path in engine_files + app_files), checks)
    _check("all_gate_evidence_passed", all(item.get("passed") is True for item in evidence), checks)
    _check(
        "astra_and_astrum_are_recorded",
        all(
            (item.get("execution", {}).get("astra") == "PASS" and item.get("execution", {}).get("astrum") == "PASS")
            or (item.get("local", {}).get("verdict") == "PASS" and item.get("astrum", {}).get("verdict") == "PASS")
            for item in evidence
        ),
        checks,
    )
    _check("claim_boundaries_are_present", all(isinstance(item.get("claim_boundary"), str) and item["claim_boundary"] for item in evidence), checks)
    _check(
        "protection_verdict_is_explicitly_bounded",
        "NOT_CLAIMED" in str(evidence[-1].get("protection_verdict")) and "NOT_EVALUATED" in json.dumps(evidence[-1].get("criteria", {})),
        checks,
    )
    engine = importlib.import_module("quantum_transport")
    app_src = _app_root() / "src"
    sys.path.insert(0, str(app_src))
    app = importlib.import_module("xene_ring_transport")
    _check("engine_full_contour_api_is_importable", hasattr(engine, "kbe_lesser_contour_correction") and hasattr(engine, "self_consistent_hubbard_second_born_contour_two_time"), checks)
    _check("app_closed_corbino_api_is_importable", hasattr(app, "solve_hubbard_kane_mele_two_time") and hasattr(app, "HubbardKaneMeleTwoTimeResult"), checks)
    report = {
        "gate": "GATE_69_INTEGRATED_CLOSED_BRANCH_AUDIT",
        "checks": checks,
        "passed": all(checks.values()),
        "engine_evidence": [path.name for path in engine_files],
        "app_evidence": [path.name for path in app_files],
        "evidence_count": len(evidence),
        "assessment": "PASS_INTEGRATED_GATES61_68_ASTRA_ASTRUM_BOUNDARY_AUDIT",
        "claim_boundary": (
            "Gates61--68 form a reproducible ASTRA/ASTRUM block for total-self-energy accounting, three-term lesser reconstruction, "
            "exact charge/spin validation, grid/lead controls, closed Corbino channels, persistent separation, and explicit topology criteria. "
            "The block still carries an open interacting continuity criterion and does not claim topological protection."
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
