"""Gate 36: nested real/vertical grid convergence of the mixed source."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantum_transport import (  # noqa: E402
    hubbard_second_born_self_energy_mixed,
    kbe_initial_correlation_kernel,
    partition_free_finite_lead_two_time,
)


def _check(name: str, condition: bool, checks: dict[str, bool]) -> None:
    checks[name] = bool(condition)
    print(f"CHECK {name}: {'PASS' if condition else 'FAIL'}")


def _run(time_points: int, imaginary_points: int) -> dict[str, object]:
    time = np.linspace(0.0, 0.5, time_points)
    imaginary = np.linspace(0.0, 1.0 / 0.3, imaginary_points)
    finite = partition_free_finite_lead_two_time(
        time,
        imaginary,
        initial_device_hamiltonian=np.diag([-0.25, -0.18]).astype(complex),
        final_device_hamiltonian=np.diag([0.08, -0.02]).astype(complex),
        lead_hamiltonians=[np.diag([-0.8, -0.65]).astype(complex), np.diag([0.5, 0.62]).astype(complex)],
        coupling_matrices=[np.diag([0.25, 0.25]).astype(complex), np.diag([0.2, 0.2]).astype(complex)],
        lead_shifts=[0.15, -0.12],
        temperature=0.3,
    )
    sigma = hubbard_second_born_self_energy_mixed(
        time,
        imaginary,
        green_rceil=finite.green_mixed.swapaxes(0, 1),
        green_lceil=finite.green_mixed,
        interaction_u=0.5,
        spin_pairs=((0, 1), (1, 0)),
    )
    source = kbe_initial_correlation_kernel(
        time,
        imaginary,
        self_energy_mixed=sigma,
        green_mixed=finite.green_mixed,
    )
    return {
        "time_points": time_points,
        "imaginary_points": imaginary_points,
        "sigma": sigma,
        "source": source.density_source,
        "source_hermiticity_error": source.hermiticity_error,
        "source_max": float(np.max(np.abs(source.density_source))),
    }


def run_gate() -> dict[str, object]:
    grids = [_run(5, 21), _run(9, 41), _run(17, 81)]
    reference = grids[-1]["source"]
    errors: list[float] = []
    for coarse, stride in zip(grids[:-1], (4, 2)):
        source = coarse["source"]
        errors.append(float(np.max(np.abs(source - reference[::stride]))) )
    checks: dict[str, bool] = {}
    _check("all_mixed_sources_are_hermitian", all(item["source_hermiticity_error"] < 1e-14 for item in grids), checks)
    _check("coarse_to_medium_source_refines", errors[1] < errors[0], checks)
    _check("source_norm_is_stable", max(item["source_max"] for item in grids) - min(item["source_max"] for item in grids) < 5e-4, checks)
    report = {
        "gate": "GATE_36_MIXED_SOURCE_GRID_CONVERGENCE",
        "checks": checks,
        "passed": all(checks.values()),
        "grids": [{"time_points": item["time_points"], "imaginary_points": item["imaginary_points"], "source_max": item["source_max"], "source_hermiticity_error": item["source_hermiticity_error"]} for item in grids],
        "metrics": {
            "coarse_to_fine_source_errors": errors,
            "source_max_spread": float(max(item["source_max"] for item in grids) - min(item["source_max"] for item in grids)),
        },
        "assessment": "PASS_MIXED_SOURCE_NESTED_GRID_CONVERGENCE_REFERENCE",
        "claim_boundary": (
            "The explicit mixed second-Born source is stable under nested real "
            "and vertical grid refinement on the finite quadratic reference. "
            "This is a convergence result for a seeded reference kernel, not a "
            "self-consistent interacting continuum closure."
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
