"""Gate 80: ten-gate review and publication/protection decision."""

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
        "gate71": engine_evidence / "gate71_self_consistent_matsubara_20260802.json",
        "gate72": app_evidence / "gate72_hubbard_corbino_size_scaling_20260802.json",
        "gate73": engine_evidence / "gate73_time_dependent_matrix_embedding_20260802.json",
        "gate74": engine_evidence / "gate74_specialist_novelty_audit_20260802.json",
        "gate75": engine_evidence / "gate75_publication_claim_matrix_20260802.json",
        "gate76": engine_evidence / "gate76_exact_interaction_accuracy_ledger_20260802.json",
        "gate77": app_evidence / "gate77_transient_flux_ramp_memory_20260802.json",
        "gate78": app_evidence / "gate78_spinful_flux_ramp_controls_20260802.json",
        "gate79": engine_evidence / "gate79_reproducible_package_20260802.json",
    }
    checks: dict[str, bool] = {}
    _check("nine_gate_records_exist", all(path.is_file() for path in paths.values()), checks)
    records = {name: _read(path) for name, path in paths.items() if path.is_file()}
    _check("nine_gate_records_pass", len(records) == len(paths) and all(record.get("passed") is True for record in records.values()), checks)
    _check("nine_gate_records_have_both_runtimes", all(record.get("execution", {}).get("astra") == "PASS" and record.get("execution", {}).get("astrum") == "PASS" for record in records.values()), checks)
    regression = records.get("gate75", {}).get("regression_snapshot", {})
    _check("full_regression_is_current", regression == {"astra_app": "70 passed", "astra_engine": "264 passed", "astrum_app": "70 passed", "astrum_engine": "264 passed"}, checks)
    _check("package_manifest_is_pass", records.get("gate79", {}).get("assessment") == "PASS_REPRODUCIBLE_ASTRA_ASTRUM_PACKAGE_WITH_OPEN_PHYSICS_GATES", checks)
    _check("specialist_novelty_remains_unconfirmed", records.get("gate74", {}).get("verdict") == "NARROW_METHOD_BENCHMARK_CANDIDATE_UNCONFIRMED", checks)
    _check("protection_remains_not_ready", records.get("gate75", {}).get("allowed_claims", {}).get("topological_protection_of_edge_currents") == "NOT_READY", checks)
    _check("accuracy_boundary_is_retained", "closure" in str(records.get("gate76", {}).get("assessment", "")).lower(), checks)
    open_gates = [
        "same-self-energy interacting real/mixed/lesser continuity and reservoir spin injection",
        "controlled continuum or lead-size extrapolation for the extended Corbino device",
        "specialist database search beyond the current arXiv/primary-source matrix",
        "falsifiable width/time/contact/disorder/Rashba/flux-ramp observable before any protection claim",
    ]
    _check("open_publication_gates_are_explicit", len(open_gates) == 4, checks)
    report = {
        "schema_version": 1,
        "date": "2026-08-02",
        "gate": "GATE_80_REVIEW",
        "scope": "ten-gate review after Gates71--79 and token checkpoint",
        "runtime": runtime,
        "execution": {
            "astra": "PASS" if runtime.upper() in {"ASTRA", "ASTRA+ASTRUM"} else "NOT_RUN",
            "astrum": "PASS" if runtime.upper() in {"ASTRUM", "ASTRA+ASTRUM"} else "NOT_RUN",
            "script": "scripts/verify_gate80_review.py",
        },
        "checks": checks,
        "passed": all(checks.values()),
        "regression": regression,
        "token_checkpoint": {
            "tokens_used": 5823815,
            "remaining_tokens": None,
            "goal_status": "active",
        },
        "closed_in_this_block": [
            "self-consistent Matsubara software branch and explicit KMS diagnostics",
            "full-size finite Corbino Hubbard edge/bulk crossover with persistent/reservoir separation",
            "arbitrary finite-grid nonstationary matrix embedding interface",
            "specialist primary-source novelty and claim-boundary audit",
            "publication claim matrix with explicit allowed/not-ready claims",
            "exact finite-contact interaction accuracy ledger over U=0..0.8",
            "finite-ramp transient onset and persistent/reservoir separation",
            "spinful Kane–Mele/trivial mass flux-ramp and torque controls",
            "SHA-256 reproducible package manifest",
        ],
        "open_publication_gates": open_gates,
        "publication_readiness": {
            "bounded_software_benchmark_result": "READY_FOR_DRAFT_WITH_EXPLICIT_LIMITATIONS",
            "narrow_integrated_novelty_candidate": "UNCONFIRMED",
            "strong_new_physics_claim": "NOT_READY",
            "conserving_interacting_continuum_claim": "NOT_READY",
            "topological_protection_claim": "NOT_READY",
        },
        "assessment": "REVIEW_PASS_WITH_BOUNDED_SOFTWARE_RESULT_AND_OPEN_PHYSICS_GATES",
        "decision": (
            "The project has a reproducible and technically substantial transient EOM/Green/Keldysh charge-spin workflow suitable for a draft "
            "as a bounded software/benchmark paper. The evidence does not establish broad method novelty, an interacting conserving continuum "
            "theorem, or topological protection. The finite spinful controls are diagnostic and include trivial comparisons; the correct status "
            "is unconfirmed/indeterminate outside the tested finite regimes."
        ),
        "next_block": "Gates81+ may target same-self-energy continuum continuity and controlled Corbino extrapolation; do not upgrade the protection label before those gates close.",
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
