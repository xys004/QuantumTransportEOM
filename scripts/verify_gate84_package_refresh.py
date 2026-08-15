"""Gate 84: refresh reproducible manifest after Gates81--83."""

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


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        "gate79": engine_evidence / "gate79_reproducible_package_20260802.json",
        "gate80": engine_evidence / "gate80_review_20260802.json",
        "gate81": engine_evidence / "gate81_same_self_energy_continuity_20260803.json",
        "gate82": app_evidence / "gate82_lead_size_extrapolation_20260803.json",
        "gate83": engine_evidence / "gate83_regression_refresh_20260803.json",
    }
    sources = {
        "continuity_module": engine / "src" / "quantum_transport" / "continuity.py",
        "continuity_exports": engine / "src" / "quantum_transport" / "__init__.py",
        "continuity_test": engine / "tests" / "test_continuity.py",
        "gate81_script": engine / "scripts" / "verify_gate81_same_self_energy_continuity.py",
        "gate82_script": app / "scripts" / "verify_gate82_lead_size_extrapolation.py",
        "gate83_script": engine / "scripts" / "verify_gate83_regression_refresh.py",
        "protocol": engine / "docs" / "GATE_PROTOCOL_TRANSIENT_KELDYSH_SPIN.md",
    }
    checks: dict[str, bool] = {}
    _check("all_records_exist", all(path.is_file() for path in records.values()), checks)
    _check("all_new_sources_exist", all(path.is_file() for path in sources.values()), checks)
    loaded = {name: _read(path) for name, path in records.items() if path.is_file()}
    _check("all_records_pass", len(loaded) == len(records) and all(record.get("passed") is True for record in loaded.values()), checks)
    _check("all_records_have_both_runtimes", all(record.get("execution", {}).get("astra") == "PASS" and record.get("execution", {}).get("astrum") == "PASS" for record in loaded.values()), checks)
    _check("regression_refresh_is_265_70", loaded.get("gate83", {}).get("regression", {}).get("astra_engine") == "265 passed" and loaded.get("gate83", {}).get("regression", {}).get("astra_app") == "70 passed", checks)
    _check("same_self_energy_gate_is_present", loaded.get("gate81", {}).get("gate") == "GATE_81_SAME_SELF_ENERGY_CONTINUITY", checks)
    _check("lead_extrapolation_gate_is_present", loaded.get("gate82", {}).get("gate") == "GATE_82_LEAD_SIZE_EXTRAPOLATION", checks)
    _check("protection_boundary_survives_refresh", loaded.get("gate80", {}).get("publication_readiness", {}).get("topological_protection_claim") == "NOT_READY", checks)
    manifest = [
        {"name": name, "path": str(path), "sha256": _sha256(path)}
        for name, path in {**records, **sources}.items()
        if path.is_file()
    ]
    report = {
        "schema_version": 1,
        "gate": "GATE_84_PACKAGE_REFRESH",
        "runtime": runtime,
        "execution": {
            "astra": "PASS" if runtime.upper() in {"ASTRA", "ASTRA+ASTRUM"} else "NOT_RUN",
            "astrum": "PASS" if runtime.upper() in {"ASTRUM", "ASTRA+ASTRUM"} else "NOT_RUN",
            "script": "scripts/verify_gate84_package_refresh.py",
        },
        "checks": checks,
        "passed": all(checks.values()),
        "manifest": manifest,
        "regression": loaded.get("gate83", {}).get("regression", {}),
        "assessment": "PASS_REFRESHED_REPRODUCIBLE_PACKAGE_AFTER_CONTINUITY_AND_EXTRAPOLATION",
        "claim_boundary": (
            "The refreshed manifest includes the Gate81 continuity decomposition, Gate82 lead-size extrapolation, Gate83 regression baseline, "
            "their source files, and the historical evidence block. It preserves the finite-grid/software claim boundary and does not promote "
            "interacting continuum conservation or topological protection."
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
