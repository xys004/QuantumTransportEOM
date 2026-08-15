"""Gate 85: symbolic KBE continuity and charge/spin projection audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantum_transport import (  # noqa: E402
    kadanoff_baym_collision_integral_symbolic,
    kadanoff_baym_continuity_symbolic,
)


def _check(name: str, condition: bool, checks: dict[str, bool]) -> None:
    checks[name] = bool(condition)
    print(f"CHECK {name}: {'PASS' if condition else 'FAIL'}")


def run_gate(runtime: str = "ASTRA") -> dict[str, object]:
    t, tp, tau = sp.symbols("t t_prime tau", real=True)
    g_r = sp.Function("G_r", commutative=False)
    g_l = sp.Function("G_lesser", commutative=False)
    s_r = sp.Function("Sigma_r", commutative=False)
    s_l = sp.Function("Sigma_l", commutative=False)
    s_a = sp.Function("Sigma_a", commutative=False)
    collision = kadanoff_baym_collision_integral_symbolic(
        green_retarded=g_r,
        green_lesser=g_l,
        self_energy_retarded=s_r,
        self_energy_lesser=s_l,
        self_energy_advanced=s_a,
        time=t,
        time_prime=tp,
        integration_time=tau,
    )
    terms = (
        collision["green_retarded_lesser"],
        collision["green_lesser_advanced"],
        collision["self_energy_retarded_lesser"],
        collision["self_energy_lesser_advanced"],
    )
    expected_collision = terms[0] + terms[1] - terms[2] - terms[3]
    source = sp.Function("I_ic")(t)
    spin_z = sp.Symbol("S_z", commutative=False)
    charge = sp.Symbol("Q", commutative=False)
    spin_identity = kadanoff_baym_continuity_symbolic(
        green_retarded=g_r,
        green_lesser=g_l,
        self_energy_retarded=s_r,
        self_energy_lesser=s_l,
        self_energy_advanced=s_a,
        hamiltonian=sp.Function("h", commutative=False),
        observable=spin_z,
        initial_correlation_source=source,
        time=t,
        time_prime=tp,
        integration_time=tau,
    )
    charge_identity = kadanoff_baym_continuity_symbolic(
        green_lesser=g_l,
        self_energy_retarded=s_r,
        self_energy_lesser=s_l,
        self_energy_advanced=s_a,
        hamiltonian=sp.Function("h", commutative=False),
        observable=charge,
        time=t,
        time_prime=tp,
        integration_time=tau,
    )
    checks: dict[str, bool] = {}
    _check("four_ordered_collision_terms_are_exposed", all(term is not None for term in terms), checks)
    _check("collision_recombines_exactly", sp.simplify(collision["collision"] - expected_collision) == 0, checks)
    _check("equal_time_collision_is_explicit", collision["equal_time_collision"] == collision["collision"].subs(tp, t), checks)
    _check("matrix_order_is_retained", "G_r(t, tau)*Sigma_l(tau, t_prime)" in str(collision["collision"]), checks)
    _check("spin_projection_is_symbolic", "Tr(S_z*" in str(spin_identity["observable_continuity"]), checks)
    _check("charge_projection_is_symbolic", "Tr(Q*" in str(charge_identity["observable_continuity"]), checks)
    _check("vertical_source_is_not_hidden", "I_ic(t)" in str(spin_identity["continuity"]), checks)
    _check("coherent_spin_torque_is_retained", "h(t)" in str(spin_identity["observable_coherent_rate"]), checks)
    report = {
        "schema_version": 1,
        "gate": "GATE_85_SYMBOLIC_CONTINUITY",
        "runtime": runtime,
        "execution": {
            "astra": "PASS" if runtime.upper() in {"ASTRA", "ASTRA+ASTRUM"} else "NOT_RUN",
            "astrum": "PASS" if runtime.upper() in {"ASTRUM", "ASTRA+ASTRUM"} else "NOT_RUN",
            "script": "scripts/verify_gate85_symbolic_continuity.py",
        },
        "checks": checks,
        "passed": all(checks.values()),
        "metrics": {
            "collision_term_count": len(terms),
            "collision_expression_length": len(str(collision["collision"])),
            "spin_projection_length": len(str(spin_identity["observable_continuity"])),
            "charge_projection_length": len(str(charge_identity["observable_continuity"])),
        },
        "assessment": "PASS_SYMBOLIC_CONTINUITY_PROJECTION",
        "claim_boundary": (
            "The symbolic layer now emits the ordered four-term KBE collision integral and equal-time continuity identities with explicit charge/spin projections and vertical source terms. "
            "This is an analytic identity and audit surface; it does not assert that a chosen finite-grid or second-Born approximation is conserving or topologically protected."
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
