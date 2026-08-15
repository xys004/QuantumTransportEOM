"""Gate 61: audit continuity with embedding plus interacting self-energies."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantum_transport import (  # noqa: E402
    partition_free_finite_lead_two_time,
    self_consistent_hubbard_second_born_contour_two_time,
    two_time_kbe_continuity_balance,
)


def _check(name: str, condition: bool, checks: dict[str, bool]) -> None:
    checks[name] = bool(condition)
    print(f"CHECK {name}: {'PASS' if condition else 'FAIL'}")


def _balance(reference, result, time, *, total: bool):
    if total:
        sigma_r = reference.self_energy_retarded + result.self_energy_retarded
        sigma_l = reference.self_energy_lesser + result.self_energy_lesser
        sigma_a = reference.self_energy_advanced + result.self_energy_advanced
    else:
        sigma_r = reference.self_energy_retarded
        sigma_l = reference.self_energy_lesser
        sigma_a = reference.self_energy_advanced
    return two_time_kbe_continuity_balance(
        time,
        green_retarded=result.retarded,
        green_lesser=result.lesser,
        hamiltonian=reference.final_device_hamiltonian,
        self_energy_retarded=sigma_r,
        self_energy_lesser=sigma_l,
        self_energy_advanced=sigma_a,
    )


def run_gate() -> dict[str, object]:
    time = np.linspace(0.0, 0.5, 9)
    imaginary = np.linspace(0.0, 1.0 / 0.3, 21)
    reference = partition_free_finite_lead_two_time(
        time,
        imaginary,
        initial_device_hamiltonian=np.diag([-0.25, -0.18]).astype(complex),
        final_device_hamiltonian=np.diag([0.08, -0.02]).astype(complex),
        lead_hamiltonians=[np.diag([-0.8, -0.65]).astype(complex), np.diag([0.5, 0.62]).astype(complex)],
        coupling_matrices=[np.diag([0.25, 0.25]).astype(complex), np.diag([0.2, 0.2]).astype(complex)],
        lead_shifts=[0.15, -0.12],
        temperature=0.3,
    )
    common = dict(
        bare_retarded=reference.retarded,
        bare_lesser=reference.lesser,
        bare_mixed=reference.green_mixed.swapaxes(0, 1).conj().swapaxes(-1, -2),
        green_matsubara=reference.green_matsubara,
        hamiltonian=reference.final_device_hamiltonian,
        interaction_u=0.5,
        embedding_self_energy_retarded=reference.self_energy_retarded,
        embedding_self_energy_mixed=reference.self_energy_mixed,
        spin_pairs=((0, 1), (1, 0)),
        max_iterations=100,
        dyson_iterations=100,
        mixing=0.2,
        tolerance=1e-7,
        include_hartree=True,
    )
    bare = self_consistent_hubbard_second_born_contour_two_time(time, imaginary, **common)
    closed = self_consistent_hubbard_second_born_contour_two_time(
        time, imaginary, include_vertical_lesser=True, **common
    )
    lead_bare = _balance(reference, bare, time, total=False)
    total_bare = _balance(reference, bare, time, total=True)
    lead_closed = _balance(reference, closed, time, total=False)
    total_closed = _balance(reference, closed, time, total=True)
    checks: dict[str, bool] = {}
    _check("interacting_self_energy_is_nonzero", np.max(np.abs(bare.self_energy_retarded)) > 1e-4, checks)
    _check("total_balance_is_finite", np.all(np.isfinite(total_bare.residual)) and np.all(np.isfinite(total_closed.residual)), checks)
    _check("total_and_embedding_audits_differ", np.max(np.abs(total_bare.residual - lead_bare.residual)) > 1e-10, checks)
    _check("vertical_term_changes_total_balance", abs(total_closed.maximum_residual - total_bare.maximum_residual) > 1e-8, checks)
    _check("closed_branch_converges", closed.converged, checks)
    report = {
        "gate": "GATE_61_TOTAL_SELF_ENERGY_BALANCE",
        "checks": checks,
        "passed": all(checks.values()),
        "lead_only_bare_residual_max": lead_bare.maximum_residual,
        "total_bare_residual_max": total_bare.maximum_residual,
        "lead_only_closed_residual_max": lead_closed.maximum_residual,
        "total_closed_residual_max": total_closed.maximum_residual,
        "total_vertical_ratio": float(total_closed.maximum_residual / total_bare.maximum_residual),
        "interaction_sigma_max": float(np.max(np.abs(bare.self_energy_retarded))),
        "assessment": "PASS_TOTAL_SELF_ENERGY_ACCOUNTING_AUDIT",
        "claim_boundary": (
            "The continuity audit now distinguishes embedding-only from total "
            "embedding-plus-interaction collision terms. This fixes bookkeeping "
            "ambiguity but does not by itself establish a conserving contour closure."
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
