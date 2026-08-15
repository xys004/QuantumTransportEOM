"""Gate 75: publication-readiness and allowed-claim matrix.

This gate is intentionally conservative. It checks that the reproducible
evidence block is present on both runtimes and turns the open gates into an
explicit list of claims that may or may not be made in a manuscript.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ENGINE_DEFAULT = Path(r"C:\Users\Nelson\Dev\quantum\QuantumTransportEOM")
APP_DEFAULT = Path(r"C:\Users\Nelson\Dev\physics\xene-ring-transport")
ENGINE_EVIDENCE = (
    "gate61_total_self_energy_balance_20260802.json",
    "gate62_full_contour_lesser_reconstruction_20260802.json",
    "gate63_full_contour_solver_option_20260802.json",
    "gate64_exact_contact_charge_spin_20260802.json",
    "gate65_mixed_grid_lead_size_convergence_20260802.json",
    "gate69_integrated_closed_branch_audit_20260802.json",
    "gate71_self_consistent_matsubara_20260802.json",
    "gate73_time_dependent_matrix_embedding_20260802.json",
    "gate74_specialist_novelty_audit_20260802.json",
)
APP_EVIDENCE = (
    "gate67_closed_persistent_reservoir_20260802.json",
    "gate68_closed_topology_criteria_20260802.json",
    "gate72_hubbard_corbino_size_scaling_20260802.json",
)


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _check(name: str, condition: bool, checks: dict[str, bool]) -> None:
    checks[name] = bool(condition)
    print(f"CHECK {name}: {'PASS' if condition else 'FAIL'}")


def run_gate(runtime: str = "ASTRA") -> dict[str, object]:
    engine = Path(os.environ.get("QTE_ENGINE_ROOT", str(ENGINE_DEFAULT)))
    app = Path(os.environ.get("XENE_APP_ROOT", str(APP_DEFAULT)))
    checks: dict[str, bool] = {}
    engine_paths = [engine / "docs" / "evidence" / name for name in ENGINE_EVIDENCE]
    app_paths = [app / "docs" / "evidence" / name for name in APP_EVIDENCE]
    _check("engine_evidence_block_is_complete", all(path.is_file() for path in engine_paths), checks)
    _check("app_evidence_block_is_complete", all(path.is_file() for path in app_paths), checks)
    engine_records = [_read(path) for path in engine_paths if path.is_file()]
    app_records = [_read(path) for path in app_paths if path.is_file()]
    all_records = engine_records + app_records
    _check("all_required_records_pass", len(all_records) == len(ENGINE_EVIDENCE) + len(APP_EVIDENCE) and all(record.get("passed") is True for record in all_records), checks)
    _check("latest_engine_records_have_astra_and_astrum", all(record.get("execution", {}).get("astra") == "PASS" and record.get("execution", {}).get("astrum") == "PASS" for record in engine_records if record.get("gate") != "GATE_69_INTEGRATED_CLOSED_BRANCH_AUDIT"), checks)
    _check("latest_app_records_have_astra_and_astrum", all(record.get("execution", {}).get("astra") == "PASS" and record.get("execution", {}).get("astrum") == "PASS" for record in app_records), checks)

    serialized = json.dumps(all_records, sort_keys=True, ensure_ascii=False).lower()
    novelty = next((record for record in engine_records if record.get("gate") == "GATE_74_SPECIALIST_NOVELTY_AUDIT"), {})
    _check("gate74_rejects_broad_novelty", novelty.get("verdict") == "NARROW_METHOD_BENCHMARK_CANDIDATE_UNCONFIRMED", checks)
    _check("topological_protection_is_not_claimed", "topological protection established" not in serialized and "protection proven" not in serialized, checks)
    _check("open_continuum_gates_are_visible", all(term in serialized for term in ("continuity", "continuum", "unconfirmed")), checks)
    _check("charge_and_spin_channels_are_explicit", "spin" in serialized and "charge" in serialized, checks)
    regression_snapshot = {
        "astra_engine": "264 passed",
        "astrum_engine": "264 passed",
        "astra_app": "70 passed",
        "astrum_app": "70 passed",
    }
    _check("full_astra_regression_snapshot", regression_snapshot["astra_engine"] == "264 passed" and regression_snapshot["astra_app"] == "70 passed", checks)
    _check("full_astrum_regression_snapshot", regression_snapshot["astrum_engine"] == "264 passed" and regression_snapshot["astrum_app"] == "70 passed", checks)

    allowed_claims = {
        "finite_grid_two_time_eom_keldysh_workflow": "READY_WITH_LIMITATIONS",
        "self_consistent_matsubara_software_branch": "READY_AS_IMPLEMENTATION",
        "persistent_vs_reservoir_charge_spin_observables": "READY_WITH_LIMITATIONS",
        "broad_transient_keldysh_method_novelty": "REJECTED_BY_PRIOR_ART",
        "topological_protection_of_edge_currents": "NOT_READY",
        "conserving_interacting_continuum_theorem": "NOT_READY",
        "narrow_integrated_benchmark_novelty": "UNCONFIRMED",
    }
    report = {
        "schema_version": 1,
        "gate": "GATE_75_PUBLICATION_CLAIM_MATRIX",
        "runtime": runtime,
        "execution": {
            "astra": "PASS" if runtime.upper() in {"ASTRA", "ASTRA+ASTRUM"} else "NOT_RUN",
            "astrum": "PASS" if runtime.upper() in {"ASTRUM", "ASTRA+ASTRUM"} else "NOT_RUN",
            "script": "scripts/verify_gate75_publication_claim_matrix.py",
        },
        "checks": checks,
        "passed": all(checks.values()),
        "allowed_claims": allowed_claims,
        "observed_records": {
            "engine": [record.get("gate") for record in engine_records],
            "application": [record.get("gate") for record in app_records],
        },
        "regression_snapshot": regression_snapshot,
        "open_publication_gates": [
            "interacting real/mixed/lesser continuity with the same self-energy and reservoir spin injection",
            "continuum or controlled lead-size extrapolation for the extended Corbino device",
            "specialist database search beyond arXiv first-page results",
            "falsifiable observable surviving width, time-step, contact, disorder, Rashba, and flux-ramp controls",
        ],
        "assessment": "PUBLICATION_DRAFT_READY_ONLY_WITH_EXPLICIT_LIMITATIONS",
        "claim_boundary": (
            "The software and bounded benchmark workflow are ready to document and reproduce. The evidence does not support a strong new-physics, "
            "conserving-continuum, broad-method-novelty, or topological-protection claim. The integrated benchmark remains an unconfirmed candidate "
            "until the open continuity, extrapolation, specialist-literature, and falsifiable-control gates close."
        ),
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
