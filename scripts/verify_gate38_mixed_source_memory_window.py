"""Gate 38: causal memory-window test for the seeded mixed source."""

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


def _run(tmax: float) -> dict[str, object]:
    time = np.linspace(0.0, tmax, int(round(tmax / 0.0625)) + 1)
    imaginary = np.linspace(0.0, 1.0 / 0.3, 41)
    initial_device = np.array([[0.14, 0.04 - 0.02j], [0.04 + 0.02j, -0.11]], dtype=complex)
    final_device = np.array([[0.06, 0.035 + 0.03j], [0.035 - 0.03j, -0.05]], dtype=complex)
    coupling = np.zeros((2, 12), dtype=complex)
    coupling[:, :2] = np.array([[0.18, 0.025j], [0.012, 0.16]], dtype=complex)
    finite = partition_free_finite_lead_two_time(
        time,
        imaginary,
        initial_device_hamiltonian=initial_device,
        final_device_hamiltonian=final_device,
        lead_hamiltonians=[_chain(6), _chain(6)],
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
    return {"tmax": tmax, "source": source.density_source, "source_max": float(np.max(np.abs(source.density_source))), "hermiticity_error": source.hermiticity_error}


def run_gate() -> dict[str, object]:
    short, medium, long = (_run(value) for value in (0.25, 0.5, 1.0))
    prefix_short_medium = float(np.max(np.abs(short["source"] - medium["source"][: short["source"].shape[0]])))
    prefix_medium_long = float(np.max(np.abs(medium["source"] - long["source"][: medium["source"].shape[0]])))
    checks: dict[str, bool] = {}
    _check("all_windows_are_hermitian", all(item["hermiticity_error"] < 1e-14 for item in (short, medium, long)), checks)
    _check("short_prefix_is_window_causal", prefix_short_medium < 1e-12, checks)
    _check("medium_prefix_is_window_causal", prefix_medium_long < 1e-12, checks)
    _check("extended_window_resolves_more_memory", long["source_max"] > medium["source_max"] > short["source_max"], checks)
    report = {
        "gate": "GATE_38_MIXED_SOURCE_MEMORY_WINDOW_CAUSALITY",
        "checks": checks,
        "passed": all(checks.values()),
        "windows": [{"tmax": item["tmax"], "source_max": item["source_max"], "hermiticity_error": item["hermiticity_error"]} for item in (short, medium, long)],
        "metrics": {"prefix_short_medium_max": prefix_short_medium, "prefix_medium_long_max": prefix_medium_long},
        "assessment": "PASS_MIXED_SOURCE_CAUSAL_MEMORY_WINDOW",
        "claim_boundary": (
            "The mixed source is causal under nested time-window extension and "
            "its norm grows as later times are included. This is a finite-lead "
            "memory-window control, not evidence for irreversible relaxation or "
            "an interacting continuum limit."
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
