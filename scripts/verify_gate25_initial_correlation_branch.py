"""Gate 25: vertical-branch Keldysh source and open lead-closure diagnostic.

This gate is intentionally a diagnostic gate.  It validates the new mixed
real--imaginary branch API and records the source required by the actual
partition-free Corbino run.  A required source inferred from a residual is not
treated as a microscopic closure; the latter remains open until mixed lead
kernels are supplied by the contour solver.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

if not hasattr(np, "trapezoid"):
    np.trapezoid = np.trapz

ROOT = Path(__file__).resolve().parents[1]
ENGINE = Path(os.environ.get("QTE_ENGINE_ROOT", ROOT))
APP = Path(os.environ.get("XENE_APP_ROOT", r"C:\Users\Nelson\Dev\physics\xene-ring-transport"))
sys.path.insert(0, str(ENGINE / "src"))
sys.path.insert(0, str(APP / "src"))
os.chdir(ROOT)

from quantum_transport import (  # noqa: E402
    continuity_residual_after_initial_correlation,
    kbe_initial_correlation_kernel,
    stationary_self_energy_two_time,
    two_time_kbe_continuity_balance,
    LeadSelfEnergy,
)
from xene_ring_transport import (  # noqa: E402
    ContinuousKaneMeleCorbinoParameters,
    build_honeycomb_annulus,
    corbino_contacts,
    corbino_flux_profiles,
    solve_partition_free_wide_band_kane_mele_corbino,
)


def _check(name: str, condition: bool, checks: dict[str, bool]) -> None:
    checks[name] = bool(condition)
    print(f"CHECK {name}: {'PASS' if condition else 'FAIL'}")


def _synthetic_vertical_branch() -> dict[str, object]:
    time = np.linspace(0.0, 1.0, 5)
    imaginary = np.linspace(0.0, 2.0, 9)
    sigma = np.zeros((time.size, imaginary.size, 2, 2), dtype=complex)
    green = np.zeros((imaginary.size, time.size, 2, 2), dtype=complex)
    sigma[..., 0, 0] = 0.4 + 0.2j
    sigma[..., 1, 1] = -0.3j
    green[..., 0, 0] = 0.7j
    green[..., 1, 1] = 0.2 - 0.1j
    result = kbe_initial_correlation_kernel(
        time,
        imaginary,
        self_energy_mixed=sigma,
        green_mixed=green,
    )
    corrected = continuity_residual_after_initial_correlation(
        result.density_source,
        result.density_source,
    )
    return {
        "hermiticity_error": result.hermiticity_error,
        "correction_error": float(np.max(np.abs(corrected))),
        "source_norm": float(np.max(np.abs(result.density_source))),
    }


def _partition_free_lead_diagnostic() -> dict[str, object]:
    geometry = build_honeycomb_annulus(inner_radius=1.4, outer_radius=4.0)
    contacts = corbino_contacts(geometry)
    profiles = corbino_flux_profiles(geometry)
    zero_flux = {name: 0.0 for name in profiles}
    parameters = ContinuousKaneMeleCorbinoParameters(
        geometry,
        contacts=contacts,
        intrinsic_soc=0.08,
        rashba=0.05,
        staggered_mass=0.18,
        chemical_potential=-0.2,
        temperature=0.12,
        total_gamma_inner=4.0,
        total_gamma_outer=4.0,
        inner_channels=len(contacts.left_sites),
        outer_channels=len(contacts.right_sites),
    )
    time = np.linspace(0.0, 0.6, 7)
    energy = np.linspace(-8.0, 8.0, 321)
    solution = solve_partition_free_wide_band_kane_mele_corbino(
        parameters,
        time,
        energy,
        initial_flux_values=zero_flux,
        final_flux_values={**zero_flux, "ab": 0.22},
        bias_shift=(0.2, -0.2),
        flux_profiles=profiles,
        two_time_indices=np.arange(time.size),
    )
    leads = [
        LeadSelfEnergy.wide_band(
            solution.gamma_inner, mu=0.2, temperature=0.12, name="inner"
        ),
        LeadSelfEnergy.wide_band(
            solution.gamma_outer, mu=-0.2, temperature=0.12, name="outer"
        ),
    ]
    kernels = [stationary_self_energy_two_time(lead, time, energy) for lead in leads]
    sigma_r = sum(kernel.retarded for kernel in kernels)
    sigma_a = sum(kernel.advanced for kernel in kernels)
    sigma_l = sum(kernel.lesser for kernel in kernels)
    balance = two_time_kbe_continuity_balance(
        time,
        green_retarded=solution.two_time_greens.retarded,
        green_lesser=solution.two_time_greens.lesser,
        hamiltonian=solution.final_hamiltonian,
        self_energy_retarded=sigma_r,
        self_energy_lesser=sigma_l,
        self_energy_advanced=sigma_a,
    )
    residual = balance.residual
    required_source = 0.5 * (residual + residual.swapaxes(-1, -2).conj())
    corrected = continuity_residual_after_initial_correlation(residual, required_source)
    return {
        "geometry_sites": geometry.n_sites,
        "energy_points": energy.size,
        "lead_kernel_consistency": [kernel.consistency_report().as_dict() for kernel in kernels],
        "maximum_residual": float(np.max(np.abs(residual))),
        "maximum_interior_residual": float(np.max(np.abs(residual[1:-1]))),
        "maximum_required_source": float(np.max(np.abs(required_source))),
        "required_source_hermiticity": float(
            np.max(np.abs(required_source - required_source.swapaxes(-1, -2).conj()))
        ),
        "residual_after_required_source": float(np.max(np.abs(corrected))),
        "time_residual_norms": np.linalg.norm(residual.reshape(time.size, -1), axis=1).tolist(),
    }


def run_gate() -> dict:
    synthetic = _synthetic_vertical_branch()
    lead = _partition_free_lead_diagnostic()
    checks: dict[str, bool] = {}
    _check("vertical_branch_source_hermitian", synthetic["hermiticity_error"] < 1e-14, checks)
    _check("vertical_branch_correction_identity", synthetic["correction_error"] < 1e-14, checks)
    _check("partition_free_lead_residual_is_finite", np.isfinite(lead["maximum_residual"]), checks)
    _check("required_initial_source_is_hermitian", lead["required_source_hermiticity"] < 1e-14, checks)
    _check("required_source_reconstructs_residual", lead["residual_after_required_source"] < 1e-14, checks)
    _check("lead_coupled_closure_remains_open", lead["maximum_interior_residual"] > 1e-2, checks)
    report = {
        "gate": "GATE_25_INITIAL_CORRELATION_BRANCH",
        "checks": checks,
        "passed": all(checks.values()),
        "synthetic_vertical_branch": synthetic,
        "partition_free_lead_diagnostic": lead,
        "assessment": "PASS_DIAGNOSTIC_WITH_MICROSCOPIC_LEAD_INITIAL_CORRELATION_OPEN",
        "claim_boundary": (
            "The mixed-branch API is validated, and the source required by the "
            "partition-free lead reconstruction is measured.  That inferred "
            "source is not a microscopic closure; the interacting lead-plus-" 
            "reservoir continuity claim remains open."
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
