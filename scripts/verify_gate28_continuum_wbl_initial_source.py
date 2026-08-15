"""Gate 28: continuum/WBL comparison with an explicit UV error boundary.

The analytic partition-free WBL solver is compared with finite midpoint star
quadratures of increasing bandwidth and mode density.  The finite star uses
the microscopic mixed Keldysh source; the WBL side remains an analytic
continuum reference.  The gate requires a monotone pre-recurrence comparison
and a finite, Hermitian mixed source, while recording that the residual UV
cutoff error is not yet zero.  This prevents a finite-band calculation from
being promoted silently to a continuum claim.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantum_transport import (  # noqa: E402
    flat_band_star_quadrature,
    partition_free_finite_lead_two_time,
    partition_free_wide_band_matrix_quench,
)


def _run_case(half_bandwidth: float, n_points: int) -> dict[str, float]:
    time = np.linspace(0.0, 0.25, 11)
    imaginary = np.linspace(0.0, 4.0, 41)
    temperature = 0.25
    initial_device = np.array(
        [[0.12, 0.035 - 0.02j], [0.035 + 0.02j, -0.08]], dtype=complex
    )
    final_device = np.array(
        [[0.05, 0.025 + 0.03j], [0.025 - 0.03j, -0.035]], dtype=complex
    )
    gamma_inner = np.diag([0.22, 0.0]).astype(complex)
    gamma_outer = np.diag([0.0, 0.16]).astype(complex)
    energy = np.linspace(-half_bandwidth, half_bandwidth, 801)
    reference = partition_free_wide_band_matrix_quench(
        time,
        energy,
        initial_hamiltonian=initial_device,
        final_hamiltonian=final_device,
        lead_broadenings=(gamma_inner, gamma_outer),
        bias_shift=(0.18, -0.14),
        initial_chemical_potential=0.0,
        temperature=temperature,
    )
    inner = flat_band_star_quadrature(
        gamma_inner, half_bandwidth=half_bandwidth, n_points=n_points
    )
    outer = flat_band_star_quadrature(
        gamma_outer, half_bandwidth=half_bandwidth, n_points=n_points
    )
    finite = partition_free_finite_lead_two_time(
        time,
        imaginary,
        initial_device_hamiltonian=initial_device,
        final_device_hamiltonian=final_device,
        lead_hamiltonians=(inner.lead_hamiltonian, outer.lead_hamiltonian),
        coupling_matrices=(inner.coupling_matrix, outer.coupling_matrix),
        lead_shifts=(0.18, -0.14),
        temperature=temperature,
    )
    density_error = float(np.max(np.abs(finite.density_matrices - reference.density_matrix)))
    rate = np.gradient(
        np.trace(reference.density_matrix, axis1=1, axis2=2).real,
        time,
        edge_order=2,
    )
    continuity_error = float(np.max(np.abs(rate - reference.particle_number_rate)))
    return {
        "half_bandwidth": half_bandwidth,
        "n_points": n_points,
        "density_error_vs_wbl": density_error,
        "wbl_continuity_error": continuity_error,
        "finite_spectral_identity_error": finite.spectral_identity_error,
        "mixed_source_max": float(np.max(np.abs(finite.initial_correlation.density_source))),
        "mixed_source_hermiticity_error": finite.initial_correlation.hermiticity_error,
        "finite_initial_density_trace": float(np.trace(finite.density_matrices[0]).real),
        "wbl_initial_density_trace": float(np.trace(reference.density_matrix[0]).real),
    }


def run_gate() -> dict[str, object]:
    cases = [_run_case(width, points) for width, points in ((8.0, 32), (12.0, 48), (16.0, 64))]
    errors = [case["density_error_vs_wbl"] for case in cases]
    checks = {
        "wbl_continuity_is_resolved": all(case["wbl_continuity_error"] < 5e-5 for case in cases),
        "finite_star_spectral_identities_close": all(
            case["finite_spectral_identity_error"] < 1e-12 for case in cases
        ),
        "mixed_source_is_finite_and_hermitian": all(
            np.isfinite(case["mixed_source_max"])
            and case["mixed_source_max"] > 1e-4
            and case["mixed_source_hermiticity_error"] < 1e-12
            for case in cases
        ),
        "star_wbl_error_decreases_with_bandwidth": errors[0] > errors[1] > errors[2],
        "uv_cutoff_error_remains_explicit": errors[-1] > 1e-4,
    }
    return {
        "gate": "GATE_28_CONTINUUM_WBL_INITIAL_SOURCE_BOUNDARY",
        "checks": checks,
        "passed": all(checks.values()),
        "cases": cases,
        "assessment": "PASS_WBL_REFERENCE_AND_STABLE_MIXED_SOURCE_UV_LIMIT_OPEN",
        "claim_boundary": (
            "The analytic partition-free WBL reference is continuous on the "
            "resolved grid, and the microscopic finite-star mixed source is "
            "stable after spectral KMS evaluation.  The star/WBL discrepancy "
            "decreases with bandwidth but remains finite, so the continuum "
            "initial-correlation limit is explicitly open."
        ),
    }


def main() -> None:
    report = run_gate()
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"VERDICT: {'PASS' if report['passed'] else 'FAIL'}")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
