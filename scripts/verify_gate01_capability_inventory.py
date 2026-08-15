"""Gate 1: verify the public EOM/Keldysh/two-time/spin capability contract."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import sympy as sp

import quantum_transport as qt


@dataclass(frozen=True)
class GateCheck:
    name: str
    passed: bool
    detail: str


def _check_public_api() -> GateCheck:
    required = (
        "retarded_green_from_eom",
        "source_anticommutator_matrix",
        "KeldyshSystem",
        "keldysh_system",
        "two_time_greens",
        "stationary_greens_two_time",
        "partition_free_wide_band_two_time_greens",
        "spin_current_density_omega",
        "spin_resolved_current_density_omega",
        "spin_axis_projector_numeric",
    )
    missing = [name for name in required if not hasattr(qt, name)]
    return GateCheck(
        "public_eom_keldysh_two_time_spin_api",
        not missing,
        "missing=" + repr(missing) if missing else "all required symbols exported",
    )


def _check_symbolic_eom() -> GateCheck:
    epsilon, omega, eta = sp.symbols("epsilon omega eta", real=True)
    closure = qt.EOMClosureResult(
        operators=[sp.Symbol("c")],
        eom_matrix=sp.Matrix([[epsilon]]),
        residuals=[sp.Integer(0)],
    )
    green = qt.retarded_green_from_eom(
        closure,
        sp.Matrix([[1]]),
        omega,
        eta,
    )[0, 0]
    expected = 1 / (omega + sp.I * eta - epsilon)
    passed = sp.simplify(green - expected) == 0
    return GateCheck("symbolic_retarded_eom_oracle", passed, str(green))


def _check_keldysh_symbolic() -> GateCheck:
    omega, xi = sp.symbols("omega xi", real=True)
    workspace = qt.keldysh_system(omega)
    sigma = workspace.self_energy("Sigma")
    expression = workspace.dyson_retarded(xi, sigma.retarded(omega)).doit()
    expected = 1 / (omega - xi - sigma.retarded(omega).doit())
    passed = sp.simplify(expression - expected) == 0
    return GateCheck("symbolic_keldysh_dyson_oracle", passed, str(expression))


def _check_finite_two_time() -> GateCheck:
    time = np.linspace(0.0, 0.7, 5)
    density = np.array([[0.37]], dtype=np.complex128)
    result = qt.two_time_greens(
        time,
        lambda _time: np.array([[0.21]], dtype=np.complex128),
        density,
    )
    expected = 1j * 0.37 * np.exp(
        -1j * 0.21 * (time[:, None] - time[None, :])
    )
    error = float(np.max(np.abs(result.lesser[:, :, 0, 0] - expected)))
    return GateCheck(
        "finite_two_time_equal_time_oracle",
        error < 1e-13,
        f"maximum_lesser_error={error:.3e}",
    )


def _check_spin_surface() -> GateCheck:
    basis = ["site_up", "site_down"]
    projector = qt.spin_axis_projector_numeric(basis, "z", "+")
    operator = qt.spin_axis_operator_numeric(basis, "z")
    passed = (
        projector.shape == (2, 2)
        and operator.shape == (2, 2)
        and np.allclose(projector, projector @ projector)
        and np.allclose(operator, operator.conj().T)
    )
    return GateCheck(
        "numeric_spin_projector_surface",
        bool(passed),
        "z-axis projectors/operators are Hermitian and idempotent",
    )


def run_gate() -> dict:
    checks = [
        _check_public_api(),
        _check_symbolic_eom(),
        _check_keldysh_symbolic(),
        _check_finite_two_time(),
        _check_spin_surface(),
    ]
    return {
        "gate": "GATE_01_CAPABILITY_INVENTORY",
        "checks": [check.__dict__ for check in checks],
        "passed": all(check.passed for check in checks),
        "implemented_layers": [
            "symbolic second quantization and finite EOM closure",
            "stationary retarded/advanced/lesser/greater Keldysh algebra",
            "exact finite quadratic two-time Green functions",
            "stationary continuum two-time transforms",
            "partition-free wide-band matrix step quenches",
            "charge and spin-resolved stationary transport observables",
        ],
        "upgrade_targets": [
            "automatic higher-order interacting EOM closure beyond finite basis/truncations",
            "arbitrary-time lead self-energies and smooth transient Keldysh memory kernels",
            "generic transient lead spin currents and spin-torque balance",
            "conserving interacting two-time approximations such as Kadanoff-Baym or SCBA",
            "open-system scattering invariant and novelty audit for the application",
        ],
    }


def main() -> None:
    report = run_gate()
    print(json.dumps(report, indent=2, sort_keys=True))
    for check in report["checks"]:
        print(
            f"CHECK {check['name']}: "
            f"{'PASS' if check['passed'] else 'FAIL'}"
        )
    print(f"VERDICT: {'PASS' if report['passed'] else 'FAIL'}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
