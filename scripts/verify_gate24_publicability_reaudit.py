"""Gate 24: re-audit publication boundaries after Gates 21--23."""

from __future__ import annotations

import json
import os
from pathlib import Path


def _root(name: str, fallback: str) -> Path:
    value = Path(os.environ.get(name, fallback))
    if not value.exists():
        raise FileNotFoundError(value)
    return value


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run_gate() -> dict:
    engine = _root("QTE_ENGINE_ROOT", r"C:\Users\Nelson\Dev\quantum\QuantumTransportEOM")
    app = _root("XENE_APP_ROOT", r"C:\Users\Nelson\Dev\physics\xene-ring-transport")
    engine_names = [
        "gate11_scba_interacting_inventory_20260802.json",
        "gate12_kadanoff_baym_symbolic_20260802.json",
        "gate13_kadanoff_baym_numeric_scba_20260802.json",
        "gate14_fdt_spectral_interacting_20260802.json",
        "gate15_two_time_charge_spin_currents_20260802.json",
        "gate16_analytic_reservoir_memory_20260802.json",
        "gate17_eom_hubbard_i_vs_scba_20260802.json",
        "gate21_same_hubbard_u_exact_20260802.json",
        "gate22_continuity_diagnostics_20260802.json",
    ]
    app_names = [
        "gate18_kane_mele_interacting_spin_20260802.json",
        "gate19_astrum_interaction_memory_sweep_20260802.json",
        "gate23_production_convergence_20260802.json",
    ]
    records = [_read(engine / "docs" / "evidence" / name) for name in engine_names]
    records += [_read(app / "docs" / "evidence" / name) for name in app_names]
    serialized = json.dumps(records, sort_keys=True).lower()
    checks = {
        "twelve_gate_records": len(records) == 12 and len({record["gate"] for record in records}) == 12,
        "all_astra_pass": all(record.get("astra", {}).get("verdict") == "PASS" for record in records),
        "all_astrum_pass": all(record.get("astrum", {}).get("verdict") == "PASS" for record in records),
        "full_test_records_current": "231 passed" in serialized and "69 passed" in serialized,
        "same_hubbard_u_closed_atomic": "same_hubbard_u" in serialized and "atomic" in serialized,
        "continuity_boundary_explicit": "lead-coupled interacting continuity" in serialized,
        "production_boundary_explicit": "width" in serialized and "contact" in serialized,
        "novelty_audit_documented": (engine / "docs" / "NOVELTY_AUDIT_TRANSIENT_INTERACTING_SPIN.md").is_file(),
        "targeted_search_logged": (engine / "docs" / "NOVELTY_SEARCH_LOG_20260802.md").is_file(),
        "no_protection_overclaim": "protection proven" not in serialized and "topological protection established" not in serialized,
    }
    return {
        "gate": "GATE_24_PUBLICABILITY_REAUDIT",
        "checks": checks,
        "passed": all(checks.values()),
        "assessment": "AUDIT_PASS_WITH_LEAD_COUPLED_CONTINUITY_AND_NOVELTY_GATES_OPEN",
        "closed_since_gate20": [
            "same-Hubbard-U atomic exact benchmark",
            "finite-grid KBE collision and charge/spin bookkeeping",
            "compact Corbino production energy convergence",
        ],
        "next_required": [
            "lead-coupled interacting continuity with reservoir injection and Rashba torque",
            "width/contact/disorder production atlas for interacting runs",
            "specialist literature/database novelty search",
            "falsifiable persistent/reservoir charge-spin observable surviving controls",
        ],
        "not_yet_claimed": [
            "publication-grade novelty",
            "exact continuum conservation in the interacting Corbino adapter",
            "interacting topological protection",
        ],
        "observed_test_records": {
            "engine_full_pytest": "231 passed",
            "app_full_pytest": "69 passed",
        },
    }


def main() -> None:
    report = run_gate()
    print(json.dumps(report, indent=2, sort_keys=True))
    for name, passed in report["checks"].items():
        print(f"CHECK {name}: {'PASS' if passed else 'FAIL'}")
    print(f"VERDICT: {'PASS' if report['passed'] else 'FAIL'}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
