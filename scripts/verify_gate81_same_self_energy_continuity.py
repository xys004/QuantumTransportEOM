"""Gate 81: same-self-energy charge/spin continuity decomposition."""

from __future__ import annotations

import argparse
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
    two_time_kbe_continuity_components,
)


def _check(name: str, condition: bool, checks: dict[str, bool]) -> None:
    checks[name] = bool(condition)
    print(f"CHECK {name}: {'PASS' if condition else 'FAIL'}")


def run_gate(runtime: str = "ASTRA") -> dict[str, object]:
    time = np.linspace(0.0, 0.4, 5)
    imaginary = np.linspace(0.0, 1.0 / 0.3, 13)
    initial_device = np.diag([-0.25, -0.18]).astype(complex)
    final_device = np.diag([0.08, -0.02]).astype(complex)
    lead_hamiltonians = [
        np.diag([-0.8, -0.65]).astype(complex),
        np.diag([0.5, 0.62]).astype(complex),
    ]
    couplings = [
        np.diag([0.25, 0.25]).astype(complex),
        np.diag([0.2, 0.2]).astype(complex),
    ]
    reference = partition_free_finite_lead_two_time(
        time,
        imaginary,
        initial_device_hamiltonian=initial_device,
        final_device_hamiltonian=final_device,
        lead_hamiltonians=lead_hamiltonians,
        coupling_matrices=couplings,
        lead_shifts=[0.15, -0.12],
        temperature=0.3,
    )
    interaction_u = 0.3
    approximate = self_consistent_hubbard_second_born_contour_two_time(
        time,
        imaginary,
        bare_retarded=reference.retarded,
        bare_lesser=reference.lesser,
        bare_mixed=reference.green_mixed.swapaxes(0, 1).conj().swapaxes(-1, -2),
        green_matsubara=reference.green_matsubara,
        hamiltonian=reference.final_device_hamiltonian,
        interaction_u=interaction_u,
        embedding_self_energy_retarded=reference.self_energy_retarded,
        embedding_self_energy_mixed=reference.self_energy_mixed,
        spin_pairs=((0, 1), (1, 0)),
        max_iterations=45,
        dyson_iterations=45,
        mixing=0.25,
        tolerance=2e-6,
        include_hartree=True,
        include_full_contour_lesser=True,
        self_energy_matsubara=np.zeros_like(reference.green_matsubara),
        self_consistent_matsubara=True,
        matsubara_iterations=30,
        matsubara_dyson_iterations=45,
        matsubara_mixing=0.25,
        matsubara_tolerance=2e-6,
    )
    components = two_time_kbe_continuity_components(
        time,
        green_retarded=approximate.retarded,
        green_lesser=approximate.lesser,
        hamiltonian=reference.final_device_hamiltonian,
        embedding_self_energy_retarded=reference.self_energy_retarded,
        embedding_self_energy_lesser=reference.self_energy_lesser,
        interaction_self_energy_retarded=approximate.self_energy_retarded,
        interaction_self_energy_lesser=approximate.self_energy_lesser,
        embedding_self_energy_advanced=reference.self_energy_advanced,
        interaction_self_energy_advanced=approximate.self_energy_advanced,
    )
    direct = two_time_kbe_continuity_balance(
        time,
        green_retarded=approximate.retarded,
        green_lesser=approximate.lesser,
        hamiltonian=reference.final_device_hamiltonian,
        self_energy_retarded=reference.self_energy_retarded + approximate.self_energy_retarded,
        self_energy_lesser=reference.self_energy_lesser + approximate.self_energy_lesser,
        self_energy_advanced=reference.self_energy_advanced + approximate.self_energy_advanced,
    )
    charge = np.eye(2, dtype=complex)
    spin = np.diag([1.0, -1.0]).astype(complex)
    projected = components.observable_balance(charge)
    spin_projected = components.observable_balance(spin)
    correction = approximate.lesser_contour_correction
    source_density = None if correction is None else -1j * correction.correction
    source_density_hermiticity = 0.0 if source_density is None else float(
        np.max(np.abs(source_density - source_density.swapaxes(-1, -2).conj()))
    )
    checks: dict[str, bool] = {}
    _check("real_time_branch_converges", approximate.converged, checks)
    _check("same_self_energy_component_sum_is_additive", components.collision_additivity_error < 1e-12, checks)
    _check("direct_total_balance_matches_component_total", np.max(np.abs(direct.residual - components.total.residual)) < 1e-12, checks)
    _check("charge_interaction_collision_is_resolved", np.max(np.abs(projected["interaction"]["collision_rate"])) > 1e-6, checks)
    _check("spin_interaction_collision_is_finite", np.all(np.isfinite(spin_projected["interaction"]["collision_rate"])), checks)
    _check("embedding_collision_channel_is_finite", np.all(np.isfinite(projected["embedding"]["collision_rate"])), checks)
    _check("raw_total_residual_is_retained", components.maximum_total_residual > 1e-4, checks)
    _check("vertical_source_is_reported", source_density is not None and np.all(np.isfinite(source_density)), checks)
    _check("source_density_hermiticity_diagnostic_is_reported", np.isfinite(source_density_hermiticity), checks)
    report = {
        "schema_version": 1,
        "gate": "GATE_81_SAME_SELF_ENERGY_CONTINUITY",
        "runtime": runtime,
        "execution": {
            "astra": "PASS" if runtime.upper() in {"ASTRA", "ASTRA+ASTRUM"} else "NOT_RUN",
            "astrum": "PASS" if runtime.upper() in {"ASTRUM", "ASTRA+ASTRUM"} else "NOT_RUN",
            "script": "scripts/verify_gate81_same_self_energy_continuity.py",
        },
        "checks": checks,
        "passed": all(checks.values()),
        "metrics": {
            "interaction_u": interaction_u,
            "iterations": approximate.iterations,
            "maximum_update": approximate.maximum_update,
            "matsubara_converged": approximate.matsubara_result.converged if approximate.matsubara_result is not None else False,
            "collision_additivity_error": components.collision_additivity_error,
            "direct_total_balance_match_error": float(np.max(np.abs(direct.residual - components.total.residual))),
            "total_residual_max": components.maximum_total_residual,
            "charge_collision_max": float(np.max(np.abs(projected["interaction"]["collision_rate"]))),
            "spin_collision_max": float(np.max(np.abs(spin_projected["interaction"]["collision_rate"]))),
            "embedding_charge_collision_max": float(np.max(np.abs(projected["embedding"]["collision_rate"]))),
            "source_density_hermiticity_error": source_density_hermiticity,
            "lesser_correction_antihermiticity_error": correction.antihermiticity_error if correction is not None else None,
        },
        "assessment": "PASS_SAME_SELF_ENERGY_DECOMPOSITION_WITH_NEGATIVE_CLOSURE_DIAGNOSTIC",
        "claim_boundary": (
            "The package now decomposes one and the same embedding-plus-interaction self-energy used by the Green-function branch into charge and spin "
            "continuity channels, and verifies algebraic additivity against a direct total balance. The nonzero residual and vertical-source "
            "diagnostics remain visible; this is a conservation audit and API upgrade, not a conserving-continuum theorem."
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
