"""Gate 35: explicit mixed second-Born source kernel."""

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


def run_gate() -> dict[str, object]:
    time = np.linspace(0.0, 0.5, 9)
    imaginary = np.linspace(0.0, 1.0 / 0.3, 41)
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
    green_rceil = finite.green_mixed.swapaxes(0, 1)
    sigma_zero = hubbard_second_born_self_energy_mixed(
        time,
        imaginary,
        green_rceil=green_rceil,
        green_lceil=finite.green_mixed,
        interaction_u=0.0,
        spin_pairs=((0, 1), (1, 0)),
    )
    sigma_quarter = hubbard_second_born_self_energy_mixed(
        time,
        imaginary,
        green_rceil=green_rceil,
        green_lceil=finite.green_mixed,
        interaction_u=0.25,
        spin_pairs=((0, 1), (1, 0)),
    )
    sigma_half = hubbard_second_born_self_energy_mixed(
        time,
        imaginary,
        green_rceil=green_rceil,
        green_lceil=finite.green_mixed,
        interaction_u=0.5,
        spin_pairs=((0, 1), (1, 0)),
    )
    source_quarter = kbe_initial_correlation_kernel(
        time,
        imaginary,
        self_energy_mixed=sigma_quarter,
        green_mixed=finite.green_mixed,
    )
    source_half = kbe_initial_correlation_kernel(
        time,
        imaginary,
        self_energy_mixed=sigma_half,
        green_mixed=finite.green_mixed,
    )
    checks: dict[str, bool] = {}
    _check("mixed_green_shapes_are_exposed", finite.green_mixed.shape == (imaginary.size, time.size, 2, 2), checks)
    _check("zero_u_mixed_self_energy_is_zero", np.max(np.abs(sigma_zero)) < 1e-15, checks)
    _check("mixed_self_energy_is_nonzero", np.max(np.abs(sigma_half)) > 1e-6, checks)
    _check("mixed_source_is_hermitian_at_quarter_u", source_quarter.hermiticity_error < 1e-14, checks)
    _check("mixed_source_is_hermitian_at_half_u", source_half.hermiticity_error < 1e-14, checks)
    _check("mixed_source_scales_as_u_squared", np.max(np.abs(source_half.density_source - 4.0 * source_quarter.density_source)) < 1e-12, checks)
    report = {
        "gate": "GATE_35_HUBBARD_SECOND_BORN_MIXED_SOURCE",
        "checks": checks,
        "passed": all(checks.values()),
        "model": {"device_spin_orbitals": 2, "lead_spin_orbitals": [2, 2], "temperature": 0.3, "time_points": 9, "imaginary_points": 41},
        "metrics": {
            "mixed_sigma_half_u_max": float(np.max(np.abs(sigma_half))),
            "mixed_source_quarter_u_max": float(np.max(np.abs(source_quarter.density_source))),
            "mixed_source_half_u_max": float(np.max(np.abs(source_half.density_source))),
            "mixed_source_quarter_hermiticity_error": source_quarter.hermiticity_error,
            "mixed_source_half_hermiticity_error": source_half.hermiticity_error,
            "u_squared_scaling_error": float(np.max(np.abs(source_half.density_source - 4.0 * source_quarter.density_source))),
        },
        "assessment": "PASS_MIXED_SECOND_BORN_SOURCE_KERNEL_EXPLICIT_SCALING_OPEN_CLOSURE",
        "claim_boundary": (
            "The second-Born mixed Sigma^rceil kernel is explicit, finite, "
            "Hermitian after the vertical contraction, and scales as U^2. "
            "The benchmark Green mixed branch is noninteracting; an interacting "
            "self-consistent mixed KBE solver and continuum source closure are "
            "still open and are not claimed here."
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
