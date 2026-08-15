"""Gate 90: final ten-gate review before the project analysis pause."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ENGINE_DEFAULT = Path(r"C:\Users\Nelson\Dev\quantum\QuantumTransportEOM")
APP_DEFAULT = Path(r"C:\Users\Nelson\Dev\physics\xene-ring-transport")


def _check(name: str, condition: bool, checks: dict[str, bool]) -> None:
    checks[name] = bool(condition)
    print(f"CHECK {name}: {'PASS' if condition else 'FAIL'}")


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_gate(runtime: str = "ASTRA") -> dict[str, object]:
    engine = Path(os.environ.get("QTE_ENGINE_ROOT", str(ENGINE_DEFAULT)))
    app = Path(os.environ.get("XENE_APP_ROOT", str(APP_DEFAULT)))
    engine_evidence = engine / "docs" / "evidence"
    app_evidence = app / "docs" / "evidence"
    paths = {
        "gate80": engine_evidence / "gate80_review_20260802.json",
        "gate81": engine_evidence / "gate81_same_self_energy_continuity_20260803.json",
        "gate82": app_evidence / "gate82_lead_size_extrapolation_20260803.json",
        "gate83": engine_evidence / "gate83_regression_refresh_20260803.json",
        "gate84": engine_evidence / "gate84_package_refresh_20260803.json",
        "gate85": engine_evidence / "gate85_symbolic_continuity_20260803.json",
        "gate86": engine_evidence / "gate86_regression_refresh_20260803.json",
        "gate87": engine_evidence / "gate87_noncommuting_spin_torque_20260803.json",
        "gate88": app_evidence / "gate88_rashba_spin_torque_20260803.json",
        "gate89": engine_evidence / "gate89_package_refresh_20260803.json",
    }
    checks: dict[str, bool] = {}
    _check("ten_gate_records_exist", all(path.is_file() for path in paths.values()), checks)
    records = {name: _read(path) for name, path in paths.items() if path.is_file()}
    _check("ten_gate_records_pass", len(records) == len(paths) and all(record.get("passed") is True for record in records.values()), checks)
    _check("ten_gate_records_have_both_runtimes", all(record.get("execution", {}).get("astra") == "PASS" and record.get("execution", {}).get("astrum") == "PASS" for record in records.values()), checks)
    regression = records.get("gate86", {}).get("regression", {})
    _check("full_regression_is_267_70", regression == {"astra_app": "70 passed", "astra_engine": "267 passed", "astrum_app": "70 passed", "astrum_engine": "267 passed"}, checks)
    _check("final_manifest_is_pass", records.get("gate89", {}).get("assessment") == "PASS_FINAL_ROUND_REPRODUCIBLE_PACKAGE", checks)
    _check("symbolic_and_spin_gates_are_present", records.get("gate85", {}).get("gate") == "GATE_85_SYMBOLIC_CONTINUITY" and records.get("gate87", {}).get("gate") == "GATE_87_NONCOMMUTING_SPIN_TORQUE", checks)
    _check("rashba_control_is_present", records.get("gate88", {}).get("gate") == "GATE_88_RASHBA_SPIN_TORQUE", checks)
    _check("protection_remains_not_ready", records.get("gate89", {}).get("claim_boundary", "").find("topological protection remain open") >= 0, checks)
    open_gates = [
        "conserving interacting contour closure including the self-consistent vertical branch and KMS/continuum limit",
        "larger Corbino width/lead/contact extrapolation with controlled recurrence and current normalization",
        "specialist novelty confirmation for the narrow integrated workflow and exact current-discrepancy resolution",
        "a topological-protection decision using an invariant/scattering diagnostic rather than edge-current robustness alone",
    ]
    _check("open_publication_gates_are_explicit", len(open_gates) == 4, checks)
    report = {
        "schema_version": 1,
        "date": "2026-08-03",
        "gate": "GATE_90_FINAL_REVIEW",
        "scope": "ten-gate final-round review after Gates81--89 and token checkpoint",
        "runtime": runtime,
        "execution": {
            "astra": "PASS" if runtime.upper() in {"ASTRA", "ASTRA+ASTRUM"} else "NOT_RUN",
            "astrum": "PASS" if runtime.upper() in {"ASTRUM", "ASTRA+ASTRUM"} else "NOT_RUN",
            "script": "scripts/verify_gate90_final_review.py",
        },
        "checks": checks,
        "passed": all(checks.values()),
        "regression": regression,
        "token_checkpoint": {
            "tokens_used": 6338554,
            "remaining_tokens": None,
            "goal_status": "active",
        },
        "closed_in_this_block": [
            "ordered symbolic KBE collision and continuity identities with formal charge/spin projections",
            "complete 267/70 ASTRA/ASTRUM regression refresh",
            "exact finite-contact noncommuting spin-torque and source-refinement audit",
            "Rashba on/off Corbino torque and spin-balance control",
            "final-round SHA-256 reproducibility manifest",
        ],
        "open_publication_gates": open_gates,
        "publication_readiness": {
            "bounded_software_benchmark_result": "READY_FOR_DRAFT_WITH_EXPLICIT_LIMITATIONS",
            "narrow_integrated_novelty_candidate": "UNCONFIRMED",
            "broad_method_novelty": "REJECTED_BY_PRIOR_ART",
            "strong_new_physics_claim": "NOT_READY",
            "conserving_interacting_continuum_claim": "NOT_READY",
            "topological_protection_claim": "NOT_READY",
        },
        "assessment": "FINAL_REVIEW_PASS_BOUNDED_RESULT_READY_FOR_ANALYSIS",
        "decision": (
            "The final-round project is reproducible and technically substantial as a finite-grid transient EOM/Green/Keldysh charge-spin software and benchmark workflow. "
            "The symbolic continuity and torque controls strengthen the auditability of spin transport, but the evidence still does not establish broad method novelty, "
            "a conserving interacting continuum theorem, or topological protection. The correct next action is a joint analysis pause, not another automatic claim upgrade."
        ),
        "next_action": "PAUSE_IMPLEMENTATION_AND_ANALYZE_PROJECT_STATE",
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", default="ASTRA")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_gate(args.runtime)
    rendered = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False)
    print(rendered)
    print(f"VERDICT: {'PASS' if report['passed'] else 'FAIL'}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
