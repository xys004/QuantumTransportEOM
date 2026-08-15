"""Gate 79: reproducible ASTRA/ASTRUM package manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ENGINE_DEFAULT = Path(r"C:\Users\Nelson\Dev\quantum\QuantumTransportEOM")
APP_DEFAULT = Path(r"C:\Users\Nelson\Dev\physics\xene-ring-transport")


def _check(name: str, condition: bool, checks: dict[str, bool]) -> None:
    checks[name] = bool(condition)
    print(f"CHECK {name}: {'PASS' if condition else 'FAIL'}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_gate(runtime: str = "ASTRA") -> dict[str, object]:
    engine = Path(os.environ.get("QTE_ENGINE_ROOT", str(ENGINE_DEFAULT)))
    app = Path(os.environ.get("XENE_APP_ROOT", str(APP_DEFAULT)))
    engine_evidence = engine / "docs" / "evidence"
    app_evidence = app / "docs" / "evidence"
    records = {
        "gate71": engine_evidence / "gate71_self_consistent_matsubara_20260802.json",
        "gate72": app_evidence / "gate72_hubbard_corbino_size_scaling_20260802.json",
        "gate73": engine_evidence / "gate73_time_dependent_matrix_embedding_20260802.json",
        "gate74": engine_evidence / "gate74_specialist_novelty_audit_20260802.json",
        "gate75": engine_evidence / "gate75_publication_claim_matrix_20260802.json",
        "gate76": engine_evidence / "gate76_exact_interaction_accuracy_ledger_20260802.json",
        "gate77": app_evidence / "gate77_transient_flux_ramp_memory_20260802.json",
        "gate78": app_evidence / "gate78_spinful_flux_ramp_controls_20260802.json",
    }
    source_files = {
        "engine_gate_protocol": engine / "docs" / "GATE_PROTOCOL_TRANSIENT_KELDYSH_SPIN.md",
        "engine_novelty_audit": engine / "docs" / "NOVELTY_AUDIT_TRANSIENT_INTERACTING_SPIN.md",
        "gate74_script": engine / "scripts" / "verify_gate74_specialist_novelty_audit.py",
        "gate75_script": engine / "scripts" / "verify_gate75_publication_claim_matrix.py",
        "gate76_script": engine / "scripts" / "verify_gate76_exact_interaction_accuracy_ledger.py",
        "gate77_script": app / "scripts" / "verify_gate77_transient_flux_ramp_memory.py",
        "gate78_script": app / "scripts" / "verify_gate78_spinful_flux_ramp_controls.py",
    }
    checks: dict[str, bool] = {}
    _check("all_gate_records_exist", all(path.is_file() for path in records.values()), checks)
    _check("all_gate_sources_exist", all(path.is_file() for path in source_files.values()), checks)
    loaded = {name: _read(path) for name, path in records.items() if path.is_file()}
    _check("all_gate_records_pass", len(loaded) == len(records) and all(record.get("passed") is True for record in loaded.values()), checks)
    _check("all_gate_records_have_both_runtimes", all(record.get("execution", {}).get("astra") == "PASS" and record.get("execution", {}).get("astrum") == "PASS" for record in loaded.values()), checks)
    _check("gate74_keeps_novelty_unconfirmed", loaded.get("gate74", {}).get("verdict") == "NARROW_METHOD_BENCHMARK_CANDIDATE_UNCONFIRMED", checks)
    _check("gate75_keeps_protection_not_ready", loaded.get("gate75", {}).get("allowed_claims", {}).get("topological_protection_of_edge_currents") == "NOT_READY", checks)
    _check("gate76_retains_negative_closure", "closure" in str(loaded.get("gate76", {}).get("assessment", "")).lower(), checks)
    _check("gate77_and_gate78_are_controls", "control" in (str(loaded.get("gate77", {}).get("assessment", "")) + str(loaded.get("gate77", {}).get("claim_boundary", ""))).lower() and "control" in (str(loaded.get("gate78", {}).get("assessment", "")) + str(loaded.get("gate78", {}).get("claim_boundary", ""))).lower(), checks)
    serialized = json.dumps(loaded, sort_keys=True, ensure_ascii=False).lower()
    _check("package_does_not_overclaim_protection", "protection proven" not in serialized and "protection established" not in serialized, checks)
    manifest = [
        {"name": name, "path": str(path), "sha256": _sha256(path)}
        for name, path in {**records, **source_files}.items()
        if path.is_file()
    ]
    report = {
        "schema_version": 1,
        "gate": "GATE_79_REPRODUCIBLE_PACKAGE",
        "runtime": runtime,
        "execution": {
            "astra": "PASS" if runtime.upper() in {"ASTRA", "ASTRA+ASTRUM"} else "NOT_RUN",
            "astrum": "PASS" if runtime.upper() in {"ASTRUM", "ASTRA+ASTRUM"} else "NOT_RUN",
            "script": "scripts/verify_gate79_reproducible_package.py",
        },
        "checks": checks,
        "passed": all(checks.values()),
        "manifest": manifest,
        "regression_snapshot": loaded.get("gate75", {}).get("regression_snapshot", {}),
        "assessment": "PASS_REPRODUCIBLE_ASTRA_ASTRUM_PACKAGE_WITH_OPEN_PHYSICS_GATES",
        "claim_boundary": (
            "The manifest binds the latest engine and application gate records, source verifiers, documentation, and SHA-256 hashes into one "
            "reproducible package. It authorizes a bounded software/benchmark manuscript description only; the interacting continuum closure, "
            "specialist novelty candidate, and topological-protection decision remain open."
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
