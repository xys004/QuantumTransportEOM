"""Gate 45: explicit symbolic mixed Kadanoff--Baym equations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantum_transport import kadanoff_baym_mixed_equations  # noqa: E402


def _check(name: str, condition: bool, checks: dict[str, bool]) -> None:
    checks[name] = bool(condition)
    print(f"CHECK {name}: {'PASS' if condition else 'FAIL'}")


def run_gate() -> dict[str, object]:
    noncomm = lambda name: sp.Function(name, commutative=False)
    equations = kadanoff_baym_mixed_equations(
        self_energy_retarded=noncomm("Sigma_r"),
        self_energy_mixed=noncomm("Sigma_rceil"),
        self_energy_matsubara=noncomm("Sigma_M"),
        self_energy_advanced=noncomm("Sigma_a"),
        self_energy_lmixed=noncomm("Sigma_lceil"),
        green_mixed=noncomm("G_rceil"),
        green_lmixed=noncomm("G_lceil"),
        green_matsubara=noncomm("G_M"),
        one_body_hamiltonian=noncomm("h"),
    )
    rceil = str(equations["rceil"])
    lceil = str(equations["lceil"])
    checks: dict[str, bool] = {}
    _check("returns_both_mixed_branches", set(equations) == {"rceil", "lceil"}, checks)
    _check(
        "rceil_has_causal_real_convolution",
        "Integral(Sigma_r(t, t_prime)*G_rceil(t_prime, tau), (t_prime, -oo, t))" in rceil,
        checks,
    )
    _check(
        "rceil_has_vertical_source_measure",
        ("-I*Integral" in rceil or "- I*Integral" in rceil) and "Sigma_rceil(t, tau_prime)*G_M(tau_prime, tau)" in rceil and "(tau_prime, 0, beta)" in rceil,
        checks,
    )
    _check(
        "lceil_has_advanced_real_convolution",
        "Integral(G_lceil(tau, t_prime)*Sigma_a(t_prime, t), (t_prime, -oo, t))" in lceil,
        checks,
    )
    _check(
        "lceil_has_vertical_source_measure",
        "-I*Integral" in lceil and "G_M(tau, tau_prime)*Sigma_lceil(tau_prime, t)" in lceil and "(tau_prime, 0, beta)" in lceil,
        checks,
    )
    _check("real_and_vertical_time_symbols_are_explicit", "t_prime" in rceil and "tau_prime" in rceil, checks)
    report = {
        "gate": "GATE_45_MIXED_KBE_SYMBOLIC",
        "checks": checks,
        "passed": all(checks.values()),
        "equation_keys": sorted(equations),
        "assessment": "PASS_EXPLICIT_MIXED_KBE_BRANCH_EQUATIONS",
        "claim_boundary": (
            "The package now emits explicit real/vertical differential KBE equations "
            "with causal limits and the -i imaginary-branch measure. This is a "
            "symbolic closure contract, not evidence that a self-consistent "
            "interacting contour solver is already converged."
        ),
    }
    return report


def main() -> None:
    report = run_gate()
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"VERDICT: {'PASS' if report['passed'] else 'FAIL'}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
