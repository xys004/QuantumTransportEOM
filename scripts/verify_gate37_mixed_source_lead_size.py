"""Gate 37: finite-lead-size convergence of the seeded mixed source."""

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


def _chain(size: int) -> np.ndarray:
    result = np.zeros((2 * size, 2 * size), dtype=complex)
    for site in range(size):
        onsite = 0.06 * np.diag([1.0, -1.0]).astype(complex)
        onsite += 0.035 * np.array([[0.0, 1.0 - 0.2j], [1.0 + 0.2j, 0.0]], dtype=complex)
        result[2 * site : 2 * site + 2, 2 * site : 2 * site + 2] = onsite
        if site + 1 < size:
            hopping = 0.22 * np.eye(2, dtype=complex)
            result[2 * site : 2 * site + 2, 2 * site + 2 : 2 * site + 4] = hopping
            result[2 * site + 2 : 2 * site + 4, 2 * site : 2 * site + 2] = hopping
    return result


def _check(name: str, condition: bool, checks: dict[str, bool]) -> None:
    checks[name] = bool(condition)
    print(f"CHECK {name}: {'PASS' if condition else 'FAIL'}")


def _run(size: int) -> dict[str, object]:
    time = np.linspace(0.0, 0.5, 9)
    imaginary = np.linspace(0.0, 1.0 / 0.3, 41)
    initial_device = np.array([[0.14, 0.04 - 0.02j], [0.04 + 0.02j, -0.11]], dtype=complex)
    final_device = np.array([[0.06, 0.035 + 0.03j], [0.035 - 0.03j, -0.05]], dtype=complex)
    coupling = np.zeros((2, 2 * size), dtype=complex)
    coupling[:, :2] = np.array([[0.18, 0.025j], [0.012, 0.16]], dtype=complex)
    finite = partition_free_finite_lead_two_time(
        time,
        imaginary,
        initial_device_hamiltonian=initial_device,
        final_device_hamiltonian=final_device,
        lead_hamiltonians=[_chain(size), _chain(size)],
        coupling_matrices=[coupling, coupling.conj()],
        lead_shifts=[0.18, -0.16],
        temperature=1.0 / 3.0,
    )
    sigma = hubbard_second_born_self_energy_mixed(
        time,
        imaginary,
        green_rceil=finite.green_mixed.swapaxes(0, 1),
        green_lceil=finite.green_mixed,
        interaction_u=0.5,
        spin_pairs=((0, 1), (1, 0)),
    )
    source = kbe_initial_correlation_kernel(time, imaginary, self_energy_mixed=sigma, green_mixed=finite.green_mixed)
    return {"size": size, "source": source.density_source, "source_max": float(np.max(np.abs(source.density_source))), "hermiticity_error": source.hermiticity_error, "spectral_identity_error": finite.spectral_identity_error}


def run_gate() -> dict[str, object]:
    runs = [_run(size) for size in (2, 3, 4, 6)]
    reference = runs[-1]["source"]
    convergence = [{"size": item["size"], "source_vs_size6_max": float(np.max(np.abs(item["source"] - reference)))} for item in runs[:-1]]
    checks: dict[str, bool] = {}
    _check("all_sources_are_hermitian", all(item["hermiticity_error"] < 1e-14 for item in runs), checks)
    _check("all_finite_lead_oracles_are_spectral", all(item["spectral_identity_error"] < 1e-12 for item in runs), checks)
    _check("size4_source_converges_to_size6", convergence[2]["source_vs_size6_max"] < 1e-7, checks)
    _check("source_is_resolved", max(item["source_max"] for item in runs) > 1e-4, checks)
    report = {
        "gate": "GATE_37_MIXED_SOURCE_LEAD_SIZE_CONVERGENCE",
        "checks": checks,
        "passed": all(checks.values()),
        "runs": [{"size": item["size"], "source_max": item["source_max"], "hermiticity_error": item["hermiticity_error"], "spectral_identity_error": item["spectral_identity_error"]} for item in runs],
        "convergence": convergence,
        "assessment": "PASS_MIXED_SOURCE_FINITE_LEAD_SIZE_CONTROL",
        "claim_boundary": (
            "The seeded mixed second-Born source converges with finite lead size "
            "on the pre-recurrence window. This controls a quadratic/reference "
            "branch; it does not establish an interacting continuum limit or a "
            "topological invariant."
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
