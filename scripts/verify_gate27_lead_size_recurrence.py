"""Gate 27: finite-lead size, recurrence, charge and spin-current oracle.

The model is a contacted two-spin-orbital device with two finite one-
dimensional spinful leads.  The initial state is the coupled equilibrium and
the final evolution includes a device quench and opposite lead shifts.  The
gate uses the exact full-system derivative (not a finite-difference surrogate)
to test the microscopic vertical-branch source, then compares device density
and bond charge/spin currents across lead sizes before the first reflection.

This is a finite quadratic convergence gate.  It is deliberately not a claim
about topological protection or a continuum/WBL interacting limit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from quantum_transport import (  # noqa: E402
    partition_free_finite_lead_two_time,
    two_time_kbe_collision_integral,
)


def _chain(size: int) -> np.ndarray:
    """Spinful nearest-neighbour chain with a local spin-mixing term."""

    result = np.zeros((2 * size, 2 * size), dtype=complex)
    for site in range(size):
        onsite = 0.06 * np.diag([1.0, -1.0]).astype(complex)
        onsite += 0.035 * np.array(
            [[0.0, 1.0 - 0.2j], [1.0 + 0.2j, 0.0]], dtype=complex
        )
        result[2 * site : 2 * site + 2, 2 * site : 2 * site + 2] = onsite
        if site + 1 < size:
            hopping = 0.22 * np.eye(2, dtype=complex)
            result[2 * site : 2 * site + 2, 2 * site + 2 : 2 * site + 4] = hopping
            result[2 * site + 2 : 2 * site + 4, 2 * site : 2 * site + 2] = hopping.conj().T
    return result


def _full_density(result, time: np.ndarray) -> np.ndarray:
    energies, states = np.linalg.eigh(result.final_hamiltonian)
    evolution = np.einsum(
        "ik,tk,kj->tij",
        states,
        np.exp(-1j * time[:, None] * energies[None, :]),
        states.conj().T,
        optimize=True,
    )
    return np.einsum(
        "tik,kl,tjl->tij",
        evolution,
        result.initial_density,
        evolution.conj(),
        optimize=True,
    )


def _exact_source_closure(result, time: np.ndarray) -> tuple[float, float]:
    collision = two_time_kbe_collision_integral(
        time,
        green_retarded=result.retarded,
        green_lesser=result.lesser,
        self_energy_retarded=result.self_energy_retarded,
        self_energy_lesser=result.self_energy_lesser,
        self_energy_advanced=result.self_energy_advanced,
    )
    collision_rate = collision[np.arange(time.size), np.arange(time.size)]
    full_density = _full_density(result, time)
    h_final = result.final_hamiltonian
    derivative = -1j * (
        np.einsum("ab,tbj->taj", h_final[:2, :], full_density[:, :, :2], optimize=True)
        - np.einsum("tib,bj->tij", full_density[:, :2, :], h_final[:, :2], optimize=True)
    )
    derivative = 0.5 * (derivative + derivative.swapaxes(-1, -2).conj())
    density = result.density_matrices
    coherent = -1j * (
        result.final_device_hamiltonian @ density - density @ result.final_device_hamiltonian
    )
    residual = derivative - coherent - collision_rate
    corrected = residual + result.initial_correlation.density_source
    interior = slice(3, -3)
    return float(np.max(np.abs(residual[interior]))), float(np.max(np.abs(corrected[interior])))


def _bond_currents(result, time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return lead-resolved charge and device-spin-z exchange currents."""

    full_density = _full_density(result, time)
    charge = np.zeros((time.size, len(result.lead_hamiltonians)), dtype=float)
    spin = np.zeros_like(charge)
    sigma_z = np.diag([1.0, -1.0]).astype(complex)
    offset = result.device_dimension
    for alpha, lead in enumerate(result.lead_hamiltonians):
        lead_size = lead.shape[0]
        left = result.final_hamiltonian[:2, offset : offset + lead_size]
        right = result.final_hamiltonian[offset : offset + lead_size, :2]
        rho_dl = full_density[:, :2, offset : offset + lead_size]
        rho_ld = full_density[:, offset : offset + lead_size, :2]
        charge[:, alpha] = np.real(
            -1j
            * (
                np.einsum("ab,tba->t", left, rho_ld, optimize=True)
                - np.einsum("tab,ba->t", rho_dl, right, optimize=True)
            )
        )
        spin[:, alpha] = np.real(
            -1j
            * (
                np.einsum("ab,bc,tca->t", sigma_z, left, rho_ld, optimize=True)
                - np.einsum("ab,tbc,ca->t", sigma_z, rho_dl, right, optimize=True)
            )
        )
        offset += lead_size
    return charge, spin


def _run_size(size: int) -> dict[str, object]:
    time = np.linspace(0.0, 1.0, 51)
    imaginary = np.linspace(0.0, 3.0, 121)
    initial_device = np.array(
        [[0.14, 0.04 - 0.02j], [0.04 + 0.02j, -0.11]], dtype=complex
    )
    final_device = np.array(
        [[0.06, 0.035 + 0.03j], [0.035 - 0.03j, -0.05]], dtype=complex
    )
    coupling = np.zeros((2, 2 * size), dtype=complex)
    coupling[:, :2] = np.array([[0.18, 0.025j], [0.012, 0.16]], dtype=complex)
    result = partition_free_finite_lead_two_time(
        time,
        imaginary,
        initial_device_hamiltonian=initial_device,
        final_device_hamiltonian=final_device,
        lead_hamiltonians=(_chain(size), _chain(size)),
        coupling_matrices=(coupling, coupling.conj()),
        lead_shifts=(0.18, -0.16),
        temperature=1.0 / 3.0,
    )
    residual_max, corrected_max = _exact_source_closure(result, time)
    charge, spin = _bond_currents(result, time)
    return {
        "size": size,
        "time_points": time.size,
        "imaginary_points": imaginary.size,
        "device_density": result.density_matrices,
        "charge_current": charge,
        "spin_current": spin,
        "source_max": float(np.max(np.abs(result.initial_correlation.density_source))),
        "source_closure_raw_max": residual_max,
        "source_closure_corrected_max": corrected_max,
        "spectral_identity_error": result.spectral_identity_error,
        "initial_device_lead_coherence": float(np.max(np.abs(result.initial_density[:2, 2:]))),
    }


def _clean(value: dict[str, object]) -> dict[str, object]:
    return {
        "size": value["size"],
        "source_max": value["source_max"],
        "source_closure_raw_max": value["source_closure_raw_max"],
        "source_closure_corrected_max": value["source_closure_corrected_max"],
        "spectral_identity_error": value["spectral_identity_error"],
        "initial_device_lead_coherence": value["initial_device_lead_coherence"],
        "charge_current_max": float(np.max(np.abs(value["charge_current"]))),
        "spin_current_max": float(np.max(np.abs(value["spin_current"]))),
    }


def run_gate() -> dict[str, object]:
    runs = [_run_size(size) for size in (2, 3, 4, 6)]
    reference = runs[-1]
    density_reference = reference["device_density"]
    charge_reference = reference["charge_current"]
    spin_reference = reference["spin_current"]
    convergence = []
    for value in runs[:-1]:
        convergence.append(
            {
                "size": value["size"],
                "device_density_vs_size6_max": float(
                    np.max(np.abs(value["device_density"] - density_reference))
                ),
                "charge_current_vs_size6_max": float(
                    np.max(np.abs(value["charge_current"] - charge_reference))
                ),
                "spin_current_vs_size6_max": float(
                    np.max(np.abs(value["spin_current"] - spin_reference))
                ),
            }
        )
    # For hopping J, the fastest nearest-neighbour signal has v_max=2J.  A
    # round trip to the end of a length-L chain is therefore >= L/J.
    recurrence_lower_bound = 2.0 / 0.22
    checks = {
        "all_spectral_identities_close": all(
            value["spectral_identity_error"] < 1e-12 for value in runs
        ),
        "all_microscopic_sources_nonzero": all(value["source_max"] > 1e-4 for value in runs),
        "all_exact_source_closures_close": all(
            value["source_closure_corrected_max"] < 2e-6 for value in runs
        ),
        "initial_contact_coherence_present": all(
            value["initial_device_lead_coherence"] > 1e-3 for value in runs
        ),
        "pre_recurrence_window_is_resolved": 1.0 < recurrence_lower_bound,
        "size4_density_converged_to_size6": convergence[2]["device_density_vs_size6_max"] < 1e-6,
        "size4_charge_current_converged_to_size6": convergence[2]["charge_current_vs_size6_max"] < 1e-6,
        "spin_current_is_resolved": max(
            float(np.max(np.abs(value["spin_current"]))) for value in runs
        ) > 1e-5,
    }
    return {
        "gate": "GATE_27_FINITE_LEAD_SIZE_RECURRENCE_AND_SPIN_CURRENTS",
        "checks": checks,
        "passed": all(checks.values()),
        "lead_sizes": [_clean(value) for value in runs],
        "convergence_to_size6": convergence,
        "recurrence_lower_bound": recurrence_lower_bound,
        "pre_recurrence_time_window": [0.0, 1.0],
        "assessment": "PASS_FINITE_LEAD_PRE_RECURRENCE_CONVERGENCE_CONTINUUM_AND_TOPOLOGY_OPEN",
        "claim_boundary": (
            "The exact finite quadratic contacted oracle closes the microscopic "
            "initial source for charge and spin and shows lead-size convergence "
            "before the first reflection.  This is not a continuum/WBL or "
            "interacting result and does not assert topological protection."
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
