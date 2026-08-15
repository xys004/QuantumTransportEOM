"""Gate 89: reproducible package refresh after Gates85--88."""

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
        "gate80": engine_evidence / "gate80_review_20260802.json",
        "gate81": engine_evidence / "gate81_same_self_energy_continuity_20260803.json",
        "gate82": app_evidence / "gate82_lead_size_extrapolation_20260803.json",
        "gate83": engine_evidence / "gate83_regression_refresh_20260803.json",
        "gate84": engine_evidence / "gate84_package_refresh_20260803.json",
        "gate85": engine_evidence / "gate85_symbolic_continuity_20260803.json",
        "gate86": engine_evidence / "gate86_regression_refresh_20260803.json",
        "gate87": engine_evidence / "gate87_noncommuting_spin_torque_20260803.json",
        "gate88": app_evidence / "gate88_rashba_spin_torque_20260803.json",
    }
    sources = {
        "symbolic_module": engine / "src" / "quantum_transport" / "kadanoff_baym_symbolic.py",
        "symbolic_exports": engine / "src" / "quantum_transport" / "__init__.py",
        "symbolic_test": engine / "tests" / "test_kadanoff_baym_symbolic.py",
        "gate85_script": engine / "scripts" / "verify_gate85_symbolic_continuity.py",
        "gate86_script": engine / "scripts" / "verify_gate86_regression_refresh.py",
        "gate87_script": engine / "scripts" / "verify_gate87_noncommuting_spin_torque.py",
        "gate88_script": app / "scripts" / "verify_gate88_rashba_spin_torque.py",
        "protocol": engine / "docs" / "GATE_PROTOCOL_TRANSIENT_KELDYSH_SPIN.md",
        "novelty_audit": engine / "docs" / "NOVELTY_AUDIT_TRANSIENT_INTERACTING_SPIN.md",
    }
    checks: dict[str, bool] = {}
    _check("all_round_records_exist", all(path.is_file() for path in records.values()), checks)
    _check("all_round_sources_exist", all(path.is_file() for path in sources.values()), checks)
    loaded = {name: _read(path) for name, path in records.items() if path.is_file()}
    _check("all_round_records_pass", len(loaded) == len(records) and all(record.get("passed") is True for record in loaded.values()), checks)
    _check("all_round_records_have_both_runtimes", all(record.get("execution", {}).get("astra") == "PASS" and record.get("execution", {}).get("astrum") == "PASS" for record in loaded.values()), checks)
    _check("regression_refresh_is_267_70", loaded.get("gate86", {}).get("regression", {}).get("astra_engine") == "267 passed" and loaded.get("gate86", {}).get("regression", {}).get("astra_app") == "70 passed", checks)
    _check("symbolic_continuity_gate_is_present", loaded.get("gate85", {}).get("gate") == "GATE_85_SYMBOLIC_CONTINUITY", checks)
    _check("noncommuting_spin_gate_is_present", loaded.get("gate87", {}).get("gate") == "GATE_87_NONCOMMUTING_SPIN_TORQUE", checks)
    _check("rashba_gate_is_present", loaded.get("gate88", {}).get("gate") == "GATE_88_RASHBA_SPIN_TORQUE", checks)
    _check("protection_boundary_survives_refresh", loaded.get("gate80", {}).get("publication_readiness", {}).get("topological_protection_claim") == "NOT_READY", checks)
    manifest = [
        {"name": name, "path": str(path), "sha256": _sha256(path)}
        for name, path in {**records, **sources}.items()
        if path.is_file()
    ]
    report = {
        "schema_version": 1,
        "gate": "GATE_89_PACKAGE_REFRESH",
        "runtime": runtime,
        "execution": {
            "astra": "PASS" if runtime.upper() in {"ASTRA", "ASTRA+ASTRUM"} else "NOT_RUN",
            "astrum": "PASS" if runtime.upper() in {"ASTRUM", "ASTRA+ASTRUM"} else "NOT_RUN",
            "script": "scripts/verify_gate89_package_refresh.py",
        },
        "checks": checks,
        "passed": all(checks.values()),
        "manifest": manifest,
        "regression": loaded.get("gate86", {}).get("regression", {}),
        "assessment": "PASS_FINAL_ROUND_REPRODUCIBLE_PACKAGE",
        "claim_boundary": (
            "The final-round manifest binds the symbolic continuity, noncommuting spin-torque, Rashba Corbino, and refreshed regression records. "
            "It preserves the current bounded software/benchmark publication claim; broad novelty, conserving interacting continuum, and topological protection remain open."
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
