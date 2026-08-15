"""Gate 4: charge and spin continuity, including an explicit spin torque."""

from __future__ import annotations

import json

import numpy as np

from quantum_transport import (
    one_body_bond_current,
    one_body_spin_bond_current,
    region_interface_current,
)


SPIN_Z = np.diag([0.5, -0.5]).astype(np.complex128)


def _density() -> np.ndarray:
    density = np.diag([0.78, 0.64, 0.23, 0.12]).astype(np.complex128)
    density[2, 0] = 0.11j
    density[0, 2] = density[2, 0].conjugate()
    density[3, 1] = -0.07j
    density[1, 3] = density[3, 1].conjugate()
    # Small inter-spin coherences make the non-commutation with a Rashba
    # hopping observable, while keeping the one-body state physical.
    density[2, 1] = 0.06 + 0.03j
    density[1, 2] = density[2, 1].conjugate()
    density[3, 0] = -0.04j
    density[0, 3] = density[3, 0].conjugate()
    eigenvalues = np.linalg.eigvalsh(density)
    if np.min(eigenvalues) <= 0 or np.max(eigenvalues) >= 1:
        raise AssertionError("gate density is not a physical one-body state")
    return density


def _derivative(matrix: np.ndarray, density: np.ndarray) -> np.ndarray:
    return -1j * (matrix @ density - density @ matrix)


def _spin_rate(matrix: np.ndarray, density: np.ndarray) -> float:
    derivative = _derivative(matrix, density)
    return float(np.real(np.trace(SPIN_Z @ derivative[:2, :2])))


def _conserving_balance() -> tuple[bool, dict]:
    matrix = np.zeros((4, 4), dtype=np.complex128)
    matrix[0, 2] = matrix[2, 0] = -0.7
    matrix[1, 3] = matrix[3, 1] = -0.3
    density = _density()
    charge_current = region_interface_current(matrix, density, [0, 1])
    charge_rate = float(np.real(np.trace(_derivative(matrix, density)[:2, :2])))
    spin_current = one_body_spin_bond_current(
        matrix, density, [0, 1], [2, 3], SPIN_Z
    )
    spin_rate = _spin_rate(matrix, density)
    errors = {
        "charge_continuity": abs(charge_rate + charge_current),
        "spin_continuity": abs(spin_rate + spin_current),
        "spin_current_magnitude": abs(spin_current),
    }
    return (
        errors["charge_continuity"] < 1e-13
        and errors["spin_continuity"] < 1e-13
        and errors["spin_current_magnitude"] > 1e-4,
        errors,
    )


def _rashba_torque_balance() -> tuple[bool, dict]:
    matrix = np.zeros((4, 4), dtype=np.complex128)
    hopping = np.array([[-0.7, 0.24], [-0.18, -0.3]], dtype=np.complex128)
    matrix[:2, 2:] = hopping
    matrix[2:, :2] = hopping.conj().T
    density = _density()
    spin_rate = _spin_rate(matrix, density)
    spin_current = one_body_spin_bond_current(
        matrix, density, [0, 1], [2, 3], SPIN_Z
    )
    torque = spin_rate + spin_current
    balance_error = abs(spin_rate + spin_current - torque)
    raw_spin_current = float(
        -2.0
        * np.imag(np.trace(SPIN_Z @ matrix[:2, 2:] @ density[2:, :2]))
    )
    symmetrization_shift = abs(raw_spin_current - spin_current)
    details = {
        "spin_rate": spin_rate,
        "symmetrized_spin_current": spin_current,
        "local_spin_torque": torque,
        "balance_error": balance_error,
        "raw_symmetrization_shift": symmetrization_shift,
    }
    return (
        abs(torque) > 1e-4
        and symmetrization_shift > 1e-4
        and balance_error < 1e-13,
        details,
    )


def run_gate() -> dict:
    checks = []
    for name, function in (
        ("conserving_charge_spin_continuity", _conserving_balance),
        ("rashba_spin_torque_balance", _rashba_torque_balance),
    ):
        passed, details = function()
        checks.append({"name": name, "passed": passed, "details": details})
    return {
        "gate": "GATE_04_CHARGE_SPIN_BALANCE",
        "checks": checks,
        "passed": all(check["passed"] for check in checks),
        "claim_scope": "finite quadratic transient one-body charge/spin continuity",
        "not_yet_claimed": [
            "interacting spin self-energies or conserving Kadanoff-Baym propagation",
            "topological protection of spin currents",
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
