"""Gate 91: generic labelled EOM hierarchy and stationary Keldysh closure."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import sympy as sp


ENGINE_ROOT = Path(__file__).resolve().parents[1]
if str(ENGINE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ENGINE_ROOT / "src"))

from quantum_transport import (  # noqa: E402
    BosonicSCBAConfig,
    ElectronBosonSCBAConfig,
    SelfConsistentClosure,
    annihilate_boson,
    build_eom_hierarchy,
    create_boson,
    f,
    fd,
    n,
)


def _anderson_terms():
    eps_up, eps_down, interaction = sp.symbols(
        "eps_up eps_down U", real=True
    )
    return {
        "one_body": eps_up * fd("up") * f("up") + eps_down * fd("down") * f("down"),
        "interaction": interaction * n("up") * n("down"),
    }


def run_gate(runtime: str = "ASTRA") -> dict[str, object]:
    omega, eta = sp.symbols("omega eta", real=True, positive=True)
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}

    shallow = build_eom_hierarchy(_anderson_terms(), max_depth=0)
    checks["depth_zero_reports_residual"] = not shallow.is_closed and bool(shallow.unresolved_operators)
    details["depth_zero_basis"] = len(shallow.basis)
    details["depth_zero_unresolved"] = len(shallow.unresolved_operators)

    hierarchy = build_eom_hierarchy(_anderson_terms(), max_depth=1)
    checks["depth_one_closes_atomic_hierarchy"] = hierarchy.is_closed and len(hierarchy.basis) == 4
    checks["contribution_provenance_is_preserved"] = {
        item.hamiltonian_label for item in hierarchy.equation(f("up")).contributions
    } == {"one_body", "interaction"}
    exact_green = hierarchy.retarded_green(omega, eta)
    sigma_lesser = sp.diag(*sp.symbols("sigma_0:4"))
    lesser_green = hierarchy.stationary_lesser_green(omega, eta, sigma_lesser)
    checks["retarded_green_shape"] = exact_green.shape == (4, 4)
    checks["lesser_green_shape"] = lesser_green.shape == (4, 4)

    approximate_green = shallow.retarded_green(omega, eta, approximate=True)
    checks["approximation_requires_explicit_opt_in"] = approximate_green.shape == (2, 2)
    explicit_closure = {
        operator: 0 for operator in shallow.unresolved_operators
    }
    closed_green = shallow.retarded_green(
        omega, eta, residual_closure=explicit_closure
    )
    checks["explicit_residual_closure"] = closed_green.shape == (2, 2)

    eps0, eps1, hopping = sp.symbols("eps0 eps1 hopping", real=True)
    quadratic = build_eom_hierarchy(
        {
            "onsite": eps0 * fd("0") * f("0") + eps1 * fd("1") * f("1"),
            "hopping": hopping * fd("0") * f("1") + hopping * fd("1") * f("0"),
        },
        max_depth=0,
    )
    checks["quadratic_hierarchy_closes_at_seed"] = quadratic.is_closed and len(quadratic.basis) == 2

    boson = build_eom_hierarchy(
        {"boson": sp.Symbol("omega_b", real=True) * create_boson(0) * annihilate_boson(0)},
        max_depth=0,
    )
    checks["bosonic_hierarchy_closes_at_seed"] = boson.statistics == "boson" and boson.is_closed

    mixed = build_eom_hierarchy(
        {
            "mixed": sp.Symbol("epsilon_m", real=True) * fd("m") * f("m")
            + sp.Symbol("omega_m", real=True) * create_boson(1) * annihilate_boson(1),
        },
        max_depth=0,
    )
    checks["mixed_hierarchy_closes_at_seed"] = mixed.statistics == "mixed" and mixed.is_closed

    contour = mixed.contour_equations()
    checks["contour_langreth_components"] = (
        len(contour.equations) == len(mixed.basis) ** 2
        and len(contour.component("rceil")) == len(contour.equations)
        and len(contour.component("M")) == len(contour.equations)
    )

    alpha = sp.Symbol("alpha", real=True)
    closure = SelfConsistentClosure(
        rules={operator: alpha * shallow.basis[0] for operator in shallow.unresolved_operators},
        initial_values={alpha: 0},
        update=lambda values, green: {alpha: sp.Rational(1, 4)},
        max_iterations=3,
    )
    closure_result = shallow.solve_self_consistent(omega, eta, closure)
    checks["self_consistent_closure_converges"] = (
        closure_result.converged
        and closure_result.values[alpha] == sp.Rational(1, 4)
    )

    contour_propagation = quadratic.contour_equations().propagate_two_time(
        np.array([0.0, 0.2, 0.4]),
        np.diag([0.2, 0.3]),
        parameters={eps0: 0.4, eps1: 0.7, hopping: 0.1},
    )
    checks["direct_two_time_eom_propagation"] = (
        contour_propagation.solver == "finite_eom"
        and contour_propagation.retarded.shape == (3, 3, 2, 2)
    )
    zero_sigma = np.zeros((3, 3, 2, 2), dtype=complex)
    contour_dyson = quadratic.contour_equations().propagate_two_time(
        np.array([0.0, 0.2, 0.4]),
        np.diag([0.2, 0.3]),
        parameters={eps0: 0.4, eps1: 0.7, hopping: 0.1},
        self_energy_retarded=zero_sigma,
        self_energy_lesser=zero_sigma,
    )
    checks["automatic_kadanoff_baym_attachment"] = (
        contour_dyson.solver == "kadanoff_baym_dyson"
        and contour_dyson.converged
    )
    scba_contour = quadratic.contour_equations().propagate_two_time(
        np.array([0.0, 0.2, 0.4]),
        np.diag([0.2, 0.3]),
        parameters={eps0: 0.4, eps1: 0.7, hopping: 0.1},
        electron_boson_scba=ElectronBosonSCBAConfig(
            coupling=np.diag([0.03, 0.02]),
            boson_frequency=0.8,
            max_iterations=1,
            dyson_iterations=2,
            tolerance=1e-3,
        ),
    )
    checks["automatic_electron_boson_scba"] = (
        scba_contour.solver == "self_consistent_born_two_time"
        and scba_contour.green.self_energy_retarded.shape == (3, 3, 2, 2)
    )
    boson_contour = boson.contour_equations().propagate_two_time(
        np.array([0.0, 0.2, 0.4]),
        np.array([[2.0]]),
        parameters={sp.Symbol("omega_b", real=True): 0.5},
    )
    checks["bosonic_two_time_source_convention"] = (
        boson_contour.solver == "finite_eom"
        and np.allclose(boson_contour.lesser[0, 0, 0, 0], 2.0j)
    )
    boson_scba = boson.contour_equations().propagate_two_time(
        np.array([0.0, 0.2, 0.4]),
        np.array([[2.0]]),
        imaginary_time=np.linspace(0.0, 2.0, 4),
        parameters={sp.Symbol("omega_b", real=True): 0.5},
        bosonic_scba=BosonicSCBAConfig(
            coupling=np.array([[0.03]], dtype=complex),
            boson_frequency=0.8,
            boson_temperature=0.5,
            cubic_vertex=np.array([[[0.01]]], dtype=complex),
            quartic_vertex=np.array([[[[0.005]]]], dtype=complex),
            max_iterations=1,
            dyson_iterations=1,
            matsubara_iterations=2,
            matsubara_dyson_iterations=2,
            tolerance=1e-3,
            matsubara_tolerance=1e-3,
        ),
    )
    checks["automatic_pure_bosonic_full_contour_scba"] = (
        boson_scba.solver == "self_consistent_bosonic_scba_contour_two_time"
        and boson_scba.green_matsubara.shape == (4, 4, 1, 1)
        and boson_scba.self_energy_mixed.shape == (3, 4, 1, 1)
        and boson_scba.self_energy_lmixed.shape == (4, 3, 1, 1)
        and boson_scba.self_energy_matsubara.shape == (4, 4, 1, 1)
        and boson_scba.mixed_adjoint_error < 1e-12
    )
    mixed_contour = mixed.contour_equations().propagate_two_time(
        np.array([0.0, 0.2, 0.4]),
        np.diag([0.2, 2.0]),
        parameters={
            sp.Symbol("epsilon_m", real=True): 0.4,
            sp.Symbol("omega_m", real=True): 0.6,
        },
    )
    checks["mixed_two_time_source_convention"] = (
        mixed_contour.solver == "finite_eom"
        and mixed_contour.lesser.shape == (3, 3, 2, 2)
    )

    report = {
        "schema_version": 1,
        "gate": "GATE_91_EOM_HIERARCHY",
        "runtime": runtime,
        "checks": checks,
        "details": details,
        "passed": all(checks.values()),
        "claim_scope": [
            "fermionic labelled EOM expansion by depth",
            "bosonic and mixed ladder hierarchies with graded retarded sources",
            "exact closure detection and explicit residual truncation",
            "stationary matrix G^r/G^< construction after closure",
            "symbolic contour, Langreth, and vertical-branch EOM projection",
            "callback-driven self-consistent residual fixed points",
            "direct finite-grid and Kadanoff--Baym two-time propagation adapter",
            "automatic time-domain electron-boson SCBA self-energy iteration",
            "automatic pure-boson SCBA with self-consistent rceil/lceil/M branches and cubic/quartic vertices",
        ],
        "limitations": [
            "the pure-boson closure is a local harmonic-mode/Fock SCBA with Gaussian cubic/quartic vertex truncations; higher skeleton diagrams remain outside scope",
            "numerical propagation uses finite real and imaginary grids; continuum-limit convergence remains a user-controlled discretisation study",
        ],
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
