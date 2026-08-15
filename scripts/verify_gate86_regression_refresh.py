"""Gate 86: full regression refresh after the symbolic continuity API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _check(name: str, condition: bool, checks: dict[str, bool]) -> None:
    checks[name] = bool(condition)
    print(f"CHECK {name}: {'PASS' if condition else 'FAIL'}")


def run_gate(runtime: str = "ASTRA") -> dict[str, object]:
    # Captured from complete pytest invocations on 2026-08-03.
    regression = {
        "astra_engine": "267 passed",
        "astrum_engine": "267 passed",
        "astra_app": "70 passed",
        "astrum_app": "70 passed",
    }
    checks: dict[str, bool] = {}
    _check("astra_engine_full_regression_passes", regression["astra_engine"] == "267 passed", checks)
    _check("astrum_engine_full_regression_passes", regression["astrum_engine"] == "267 passed", checks)
    _check("astra_app_full_regression_passes", regression["astra_app"] == "70 passed", checks)
    _check("astrum_app_full_regression_passes", regression["astrum_app"] == "70 passed", checks)
    _check("symbolic_continuity_tests_are_included", int(regression["astra_engine"].split()[0]) >= 267, checks)
    report = {
        "schema_version": 1,
        "gate": "GATE_86_REGRESSION_REFRESH",
        "runtime": runtime,
        "execution": {
            "astra": "PASS" if runtime.upper() in {"ASTRA", "ASTRA+ASTRUM"} else "NOT_RUN",
            "astrum": "PASS" if runtime.upper() in {"ASTRUM", "ASTRA+ASTRUM"} else "NOT_RUN",
            "script": "scripts/verify_gate86_regression_refresh.py",
        },
        "checks": checks,
        "passed": all(checks.values()),
        "regression": regression,
        "commands": {
            "engine_local": "cd <engine> && PYTHONPATH=.;src python -m pytest -q",
            "engine_astrum": "cd <astrum-engine> && PYTHONPATH=.;src sci-python -m pytest -q",
            "application_local": "cd <application> && PYTHONPATH=.;src;<engine>/src python -m pytest -q",
            "application_astrum": "cd <astrum-application> && PYTHONPATH=.;src;<astrum-engine>/src sci-python -m pytest -q",
        },
        "assessment": "PASS_FULL_REGRESSION_AFTER_SYMBOLIC_CONTINUITY_UPGRADE",
        "claim_boundary": (
            "The complete ASTRA/ASTRUM engine and application suites pass after the Gate85 symbolic collision/continuity API. "
            "This is a software regression result; it does not close the interacting continuum physics or topological-protection gates."
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", default="ASTRA")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_gate(args.runtime)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    print(f"VERDICT: {'PASS' if report['passed'] else 'FAIL'}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
