"""Verify the focused transient Corbino/Xene novelty matrix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ROOT_DEFAULT = Path(r"C:\Users\Nelson\Dev\quantum\QuantumTransportEOM")


def _check(name: str, condition: bool, checks: dict[str, bool]) -> None:
    checks[name] = bool(condition)
    print(f"CHECK {name}: {'PASS' if condition else 'FAIL'}")


def run_audit(runtime: str = "ASTRA", matrix_path: Path | None = None) -> dict[str, object]:
    root = Path(os.environ.get("QTE_ENGINE_ROOT", str(ROOT_DEFAULT)))
    matrix = matrix_path or root / "docs" / "evidence" / "transient_novelty_matrix_20260803.json"
    payload = json.loads(matrix.read_text(encoding="utf-8"))
    sources = payload.get("sources", [])
    categories = {source.get("category") for source in sources}
    claim_status = {entry.get("claim"): entry.get("status") for entry in payload.get("claim_matrix", [])}
    checks: dict[str, bool] = {}
    _check("matrix_exists", matrix.is_file(), checks)
    _check("all_sources_have_direct_urls", bool(sources) and all(str(s.get("url", "")).startswith("http") for s in sources), checks)
    _check("stationary_corbino_prior_art_present", "stationary_xene_corbino" in categories, checks)
    _check("transient_ab_prior_art_present", "transient_ab_generic" in categories, checks)
    _check("transient_spin_soi_prior_art_present", "transient_ab_spin_soi" in categories, checks)
    _check("transient_qsh_prior_art_present", "transient_qsh_spin" in categories, checks)
    _check("two_time_method_prior_art_present", "two_time_keldysh_method" in categories, checks)
    _check("exact_match_not_found_is_explicit", payload.get("exact_combination_found") is False, checks)
    _check("integrated_candidate_is_unconfirmed", claim_status.get("Exact integrated persistent/reservoir + bulk/edge + torque observable in a finite Xene Corbino") == "UNCONFIRMED_CANDIDATE", checks)
    _check("broad_method_claim_rejected", claim_status.get("New Keldysh/EOM transient method") == "REJECTED_BY_PRIOR_ART", checks)
    _check("topology_is_secondary", claim_status.get("Topological protection of the transient current") == "NOT_A_PRIOR_NOVELTY_CLAIM; TEST_SECONDARY", checks)
    _check("specialist_followup_required", len(payload.get("required_followup", [])) >= 4, checks)
    report = {
        "schema_version": 1,
        "artifact": "TARGETED_TRANSIENT_CORBINO_XENE_NOVELTY_AUDIT",
        "runtime": runtime,
        "execution": {
            "astra": "PASS" if runtime.upper() in {"ASTRA", "ASTRA+ASTRUM"} else "NOT_RUN",
            "astrum": "PASS" if runtime.upper() in {"ASTRUM", "ASTRA+ASTRUM"} else "NOT_RUN",
            "script": "scripts/verify_transient_novelty_audit.py",
        },
        "checks": checks,
        "passed": all(checks.values()),
        "verdict": payload.get("verdict"),
        "allowed_claim": payload.get("allowed_claim"),
        "rejected_claims": payload.get("rejected_claims"),
        "source_count": len(sources),
        "categories": sorted(categories),
        "matrix": str(matrix),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", default="ASTRA")
    parser.add_argument("--matrix", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_audit(args.runtime, args.matrix)
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
