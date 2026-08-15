"""Gate 12: symbolic two-time Langreth, KBE Dyson, and SCBA rules."""

from __future__ import annotations

import json

import sympy as sp

from quantum_transport import (
    electron_boson_scba_symbolic,
    kadanoff_baym_dyson_equations,
    langreth_two_time_convolution_symbolic,
    time_convolution_symbolic,
)


def run_gate() -> dict:
    t, tp = sp.symbols("t t_prime", real=True)
    ar, aa, al, ag = (sp.Function(name) for name in ("A_r", "A_a", "A_l", "A_g"))
    br, ba, bl, bg = (sp.Function(name) for name in ("B_r", "B_a", "B_l", "B_g"))
    convolution = time_convolution_symbolic(ar, br, t, tp)
    langreth = langreth_two_time_convolution_symbolic(
        {"r": ar, "a": aa, "<": al, ">": ag},
        {"r": br, "a": ba, "<": bl, ">": bg},
        t,
        tp,
    )
    equations = kadanoff_baym_dyson_equations(
        bare_retarded=sp.Function("g_r"),
        bare_lesser=sp.Function("g_l"),
        self_energy_retarded=sp.Function("S_r"),
        self_energy_lesser=sp.Function("S_l"),
        self_energy_advanced=sp.Function("S_a"),
    )
    omega, omega0, n, vertex = sp.symbols("omega omega0 n vertex")
    scba = electron_boson_scba_symbolic(
        energy=omega,
        boson_frequency=omega0,
        boson_occupation=n,
        coupling=vertex,
    )
    expected_lesser = vertex**2 * (
        n * sp.Function("G_lesser")(omega - omega0)
        + (n + 1) * sp.Function("G_lesser")(omega + omega0)
    )
    checks = [
        {
            "name": "explicit_time_convolution",
            "passed": "A_r(t, tau)*B_r(tau, t_prime)" in str(convolution),
            "details": {"expression": str(convolution)},
        },
        {
            "name": "two_time_langreth_lesser_rule",
            "passed": "A_r(t, tau)*B_l(tau, t_prime)" in str(langreth["<"])
            and "A_l(t, tau)*B_a(tau, t_prime)" in str(langreth["<"]),
            "details": {"lesser_expression": str(langreth["<"])},
        },
        {
            "name": "kadanoff_baym_initial_correlation_term",
            "passed": "g_l(t, t_prime)" in str(equations["lesser"])
            and "S_l(tau, nu)" in str(equations["lesser"]),
            "details": {"lesser_equation": str(equations["lesser"])},
        },
        {
            "name": "electron_boson_scba_shift_rule",
            "passed": sp.simplify(scba["lesser"] - expected_lesser) == 0,
            "details": {"lesser_formula": str(scba["lesser"])},
        },
    ]
    return {
        "gate": "GATE_12_KADANOFF_BAYM_SYMBOLIC",
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
        "claim_scope": "symbolic two-time convolutions, KBE Dyson equations, and Einstein-mode SCBA shifts",
        "not_yet_claimed": [
            "numerical time-dependent Kadanoff-Baym propagation",
            "interacting Kane–Mele/Corbino transient results",
        ],
    }


def main() -> None:
    report = run_gate()
    print(json.dumps(report, indent=2, sort_keys=True))
    for check in report["checks"]:
        print(f"CHECK {check['name']}: {'PASS' if check['passed'] else 'FAIL'}")
    print(f"VERDICT: {'PASS' if report['passed'] else 'FAIL'}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
