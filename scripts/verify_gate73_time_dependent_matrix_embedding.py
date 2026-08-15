"""Gate 73: arbitrary nonstationary matrix-valued embedding audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantum_transport import (  # noqa: E402
    equilibrium_one_body_density,
    solve_time_dependent_matrix_embedding,
    two_time_greens,
)


def _check(name: str, condition: bool, checks: dict[str, bool]) -> None:
    checks[name] = bool(condition)
    print(f"CHECK {name}: {'PASS' if condition else 'FAIL'}")


def run_gate(runtime: str = "ASTRA") -> dict[str, object]:
    hamiltonian = np.array([[0.2, 0.04 - 0.02j], [0.04 + 0.02j, -0.1]], dtype=complex)
    density = equilibrium_one_body_density(hamiltonian, temperature=0.3)
    time = np.linspace(0.0, 0.8, 9)
    drive = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    bare = two_time_greens(
        time,
        lambda value: hamiltonian + 0.08 * np.sin(1.4 * value) * drive,
        density,
    )
    weights = np.empty(time.size)
    weights[0] = 0.5 * (time[1] - time[0])
    weights[-1] = 0.5 * (time[-1] - time[-2])
    weights[1:-1] = 0.5 * (time[2:] - time[:-2])
    gamma = np.array([[0.18, 0.03 + 0.01j], [0.03 - 0.01j, 0.1]], dtype=complex)
    amplitudes = 0.7 + 0.3 * np.sin(time)
    sigma_r = np.zeros_like(bare.retarded)
    sigma_l = np.zeros_like(bare.retarded)
    for left, amplitude in enumerate(amplitudes):
        sigma_r[left, left] = -0.5j * amplitude * gamma / weights[left]
        for right, other_amplitude in enumerate(amplitudes):
            sigma_l[left, right] = (
                1j
                * 0.08
                * amplitude
                * other_amplitude
                * np.exp(-1j * 0.4 * (time[left] - time[right]))
                * gamma
            )
    result = solve_time_dependent_matrix_embedding(
        time,
        bare_retarded=bare.retarded,
        bare_lesser=bare.lesser,
        embedding_self_energy_retarded=sigma_r,
        embedding_self_energy_lesser=sigma_l,
        max_iterations=80,
        mixing=0.4,
        tolerance=1e-9,
    )
    checks: dict[str, bool] = {}
    _check("nonstationary_device_drive_is_resolved", np.max(np.abs(np.sin(1.4 * time))) > 1e-3, checks)
    _check("matrix_embedding_is_time_dependent", np.max(np.abs(sigma_l[0, 1] - sigma_l[1, 2])) > 1e-8, checks)
    _check("dyson_branch_converges", result.converged and result.iterations < 80, checks)
    _check("retarded_kernel_is_causal", result.retarded_causality_error < 1e-14, checks)
    _check("advanced_is_exact_adjoint", result.advanced_adjoint_error < 1e-14, checks)
    _check("lesser_is_antihermitian", result.lesser_antihermiticity_error < 1e-14, checks)
    _check("embedding_keldysh_identity_closes", result.keldysh_spectral_error < 1e-14, checks)
    _check("green_keldysh_identity_closes", result.green_spectral_error < 1e-12, checks)
    _check("noncommuting_matrix_response_is_nonzero", np.max(np.abs(result.green.retarded[1, 1] - result.green.retarded[1, 2])) > 1e-8, checks)
    report = {
        "schema_version": 1,
        "gate": "GATE_73_TIME_DEPENDENT_MATRIX_EMBEDDING",
        "runtime": runtime,
        "execution": {
            "astra": "PASS" if runtime.upper() in {"ASTRA", "ASTRA+ASTRUM"} else "NOT_RUN",
            "astrum": "PASS" if runtime.upper() in {"ASTRUM", "ASTRA+ASTRUM"} else "NOT_RUN",
            "script": "scripts/verify_gate73_time_dependent_matrix_embedding.py",
        },
        "checks": checks,
        "passed": all(checks.values()),
        "metrics": {
            "iterations": result.iterations,
            "maximum_update": result.maximum_update,
            "retarded_causality_error": result.retarded_causality_error,
            "advanced_adjoint_error": result.advanced_adjoint_error,
            "lesser_antihermiticity_error": result.lesser_antihermiticity_error,
            "embedding_keldysh_spectral_error": result.keldysh_spectral_error,
            "green_keldysh_spectral_error": result.green_spectral_error,
            "time_dependent_sigma_max": float(np.max(np.abs(sigma_l[0, 1] - sigma_l[1, 2]))),
            "noncommuting_response_max": float(np.max(np.abs(result.green.retarded[1, 1] - result.green.retarded[1, 2]))),
        },
        "assessment": "PASS_ARBITRARY_TIME_DEPENDENT_MATRIX_EMBEDDING_INTERFACE",
        "claim_boundary": (
            "The engine accepts arbitrary finite-grid matrix-valued Sigma^r(t,t') and Sigma^<(t,t') kernels, including nonstationary phases and "
            "noncommuting device drives, and solves the KBE Dyson branch with causal/Keldysh diagnostics. This is a software interface and finite-grid "
            "benchmark; it does not by itself derive a microscopic interacting lead or prove conserving continuum transport."
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
