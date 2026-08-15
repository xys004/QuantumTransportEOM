"""Gate 3: symbolic EOM/Keldysh reductions and analytical transient oracles."""

from __future__ import annotations

import json

import numpy as np
import sympy as sp

from quantum_transport import (
    anderson_hubbard_i_green_function,
    dyson_lesser_stationary,
    langreth_convolution,
    langreth_double_convolution,
    latex_keldysh,
    resonant_level_spectral_amplitude,
)


def _langreth_rules() -> tuple[bool, dict]:
    ar, aa, al, ag, br, ba, bl, bg = sp.symbols(
        "ar aa al ag br ba bl bg"
    )
    result = langreth_convolution(
        {"r": ar, "a": aa, "<": al, ">": ag},
        {"r": br, "a": ba, "<": bl, ">": bg},
    )
    expected = {
        "r": ar * br,
        "a": aa * ba,
        "<": ar * bl + al * ba,
        ">": ar * bg + ag * ba,
    }
    errors = {
        key: sp.simplify(result[key] - value) for key, value in expected.items()
    }
    passed = all(value == 0 for value in errors.values())
    return passed, {"errors": {key: str(value) for key, value in errors.items()}}


def _double_langreth_rules() -> tuple[bool, dict]:
    wr, wa, wl, wg, xr, xa, xl, xg, yr, ya, yl, yg = sp.symbols(
        "wr wa wl wg xr xa xl xg yr ya yl yg"
    )
    result = langreth_double_convolution(
        {"r": wr, "a": wa, "<": wl, ">": wg},
        {"r": xr, "a": xa, "<": xl, ">": xg},
        {"r": yr, "a": ya, "<": yl, ">": yg},
    )
    expected_lesser = wr * xr * yl + wr * xl * ya + wl * xa * ya
    expected_greater = wr * xr * yg + wr * xg * ya + wg * xa * ya
    errors = {
        "retarded": sp.simplify(result["r"] - wr * xr * yr),
        "advanced": sp.simplify(result["a"] - wa * xa * ya),
        "lesser": sp.simplify(result["<"] - expected_lesser),
        "greater": sp.simplify(result[">"] - expected_greater),
    }
    passed = all(value == 0 for value in errors.values())
    return passed, {"errors": {key: str(value) for key, value in errors.items()}}


def _hubbard_i_oracle() -> tuple[bool, dict]:
    omega, eta, epsilon, u, occupation = sp.symbols(
        "omega eta epsilon U occupation"
    )
    result = anderson_hubbard_i_green_function(
        "up",
        omega,
        eta,
        epsilon,
        epsilon,
        u,
        occupations={"down": occupation},
    )
    z = omega + sp.I * eta
    expected = (1 - occupation) / (z - epsilon) + occupation / (z - epsilon - u)
    error = sp.simplify(result - expected)
    return error == 0, {"simplified_error": str(error)}


def _stationary_lesser_oracle() -> tuple[bool, dict]:
    gr, sigma_lesser, ga = sp.symbols("G_r Sigma_lesser G_a")
    result = dyson_lesser_stationary(gr, sigma_lesser, ga)
    error = sp.simplify(result - gr * sigma_lesser * ga)
    return error == 0, {"simplified_error": str(error)}


def _latex_oracle() -> tuple[bool, dict]:
    omega, xi, sigma = sp.symbols("omega xi Sigma")
    expression = 1 / (omega - xi - sigma)
    rendered = latex_keldysh(expression)
    passed = isinstance(rendered, str) and len(rendered) > 10
    return passed, {"latex_length": len(rendered), "contains_fraction": "frac" in rendered}


def _resonant_level_amplitude_oracle() -> tuple[bool, dict]:
    energy = np.linspace(-2.0, 2.0, 9)
    time = np.array([0.0, 1.0e-7])
    amplitude = resonant_level_spectral_amplitude(
        energy,
        time,
        level_energy=0.23,
        total_broadening=0.6,
    )
    initial_error = float(np.max(np.abs(amplitude[0])))
    derivative = amplitude[1] / time[1]
    derivative_error = float(np.max(np.abs(derivative + 1j)))
    return (
        initial_error < 1e-14 and derivative_error < 2e-7,
        {
            "initial_amplitude_error": initial_error,
            "short_time_derivative_error": derivative_error,
        },
    )


def run_gate() -> dict:
    checks = []
    for name, function in (
        ("single_langreth_rules", _langreth_rules),
        ("double_langreth_rules", _double_langreth_rules),
        ("hubbard_i_symbolic_oracle", _hubbard_i_oracle),
        ("stationary_lesser_dyson_oracle", _stationary_lesser_oracle),
        ("keldysh_latex_oracle", _latex_oracle),
        ("resonant_level_short_time_oracle", _resonant_level_amplitude_oracle),
    ):
        passed, details = function()
        checks.append({"name": name, "passed": passed, "details": details})
    return {
        "gate": "GATE_03_SYMBOLIC_ANALYTIC",
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
        "claim_scope": "symbolic stationary Keldysh and exact noninteracting transient oracles",
        "upgrade_targets": [
            "automatic symbolic lesser/greater solutions for arbitrary interacting EOM closures",
            "time-convolution self-energy algebra for arbitrary smooth protocols",
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
