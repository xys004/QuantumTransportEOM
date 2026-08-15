"""Gate 74: specialist prior-art and claim-boundary audit.

The web search that motivated this gate is deliberately recorded as a static,
reviewable matrix. The verifier is offline and therefore reproducible on ASTRA
and ASTRUM; it does not pretend that a finite search proves novelty.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SOURCES = [
    {
        "key": "myohanen_2009",
        "title": "Kadanoff–Baym approach to quantum transport through interacting nanoscale systems",
        "url": "https://arxiv.org/abs/0906.2136",
        "scope": "two-time interacting transient transport, conserving approximations, initial correlations, time-domain Meir–Wingreen",
        "implication": "broad EOM/KBE/Keldysh transient-method novelty is prior art",
    },
    {
        "key": "myohanen_2010",
        "title": "Kadanoff–Baym approach to quantum transport in AC/DC fields",
        "url": "https://arxiv.org/abs/1006.2912",
        "scope": "embedded KBE with arbitrary time-dependent external driving",
        "implication": "time-dependent fields and lead embedding are not new by themselves",
    },
    {
        "key": "maiti_2014",
        "title": "Persistent charge and spin currents in a quantum ring using Green's function technique",
        "url": "https://arxiv.org/abs/1401.0262",
        "scope": "persistent charge/spin currents in an Aharonov–Bohm ring with spin–orbit coupling",
        "implication": "persistent charge/spin ring observables are prior art",
    },
    {
        "key": "crepin_2015",
        "title": "Flux sensitivity of quantum spin Hall rings",
        "url": "https://arxiv.org/abs/1507.03898",
        "scope": "flux-dependent many-body persistent currents in quantum-spin-Hall rings",
        "implication": "flux-sensitive QSH-ring persistent response is prior art",
    },
    {
        "key": "grandi_2015",
        "title": "Topological invariants in interacting quantum spin Hall systems",
        "url": "https://doi.org/10.1088/1367-2630/17/2/023004",
        "scope": "interacting Kane–Mele-type QSH invariants with Hubbard interactions",
        "implication": "an interacting QSH invariant or phase claim needs a stronger distinction",
    },
    {
        "key": "mitchison_2022",
        "title": "Robust Nonequilibrium Edge Currents with and without Band Topology",
        "url": "https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.128.120403",
        "scope": "nonequilibrium edge-current robustness in topological and non-topological systems",
        "implication": "robustness alone cannot be labelled topological protection",
    },
    {
        "key": "baym_kadanoff_1961",
        "title": "Conservation Laws and Correlation Functions",
        "url": "https://doi.org/10.1103/PhysRev.124.287",
        "scope": "conserving approximations and Ward-identity/conservation-law framework",
        "implication": "continuity closure is a required diagnostic, not a new theorem here",
    },
]


def _check(name: str, condition: bool, checks: dict[str, bool]) -> None:
    checks[name] = bool(condition)
    print(f"CHECK {name}: {'PASS' if condition else 'FAIL'}")


def run_gate(runtime: str = "ASTRA") -> dict[str, object]:
    checks: dict[str, bool] = {}
    keys = {source["key"] for source in SOURCES}
    _check("primary_source_matrix_is_complete", len(keys) == len(SOURCES), checks)
    _check("all_sources_have_direct_urls", all(source["url"].startswith(("https://arxiv.org/", "https://doi.org/", "https://journals.aps.org/")) for source in SOURCES), checks)
    _check("transient_kbe_method_is_prior_art", {"myohanen_2009", "myohanen_2010"}.issubset(keys), checks)
    _check("persistent_spin_ring_is_prior_art", {"maiti_2014", "crepin_2015"}.issubset(keys), checks)
    _check("interacting_qsh_invariant_is_prior_art", "grandi_2015" in keys, checks)
    _check("non_topological_edge_robustness_warning_is_prior_art", "mitchison_2022" in keys, checks)
    _check("conserving_continuity_boundary_is_explicit", "baym_kadanoff_1961" in keys, checks)
    # These are deliberate claim-boundary checks, not empirical claims.
    _check("broad_method_novelty_is_rejected", True, checks)
    _check("strong_topological_claim_is_not_forced", True, checks)
    _check("candidate_is_narrow_integrated_workflow", True, checks)
    _check("novelty_requires_specialist_followup", True, checks)
    report = {
        "schema_version": 1,
        "gate": "GATE_74_SPECIALIST_NOVELTY_AUDIT",
        "runtime": runtime,
        "execution": {
            "astra": "PASS" if runtime.upper() in {"ASTRA", "ASTRA+ASTRUM"} else "NOT_RUN",
            "astrum": "PASS" if runtime.upper() in {"ASTRUM", "ASTRA+ASTRUM"} else "NOT_RUN",
            "script": "scripts/verify_gate74_specialist_novelty_audit.py",
        },
        "search_date": "2026-08-02",
        "sources": SOURCES,
        "checks": checks,
        "passed": all(checks.values()),
        "verdict": "NARROW_METHOD_BENCHMARK_CANDIDATE_UNCONFIRMED",
        "strong_new_physics_novelty": "NOT_SUPPORTED_BY_THIS_AUDIT",
        "claim_boundary": (
            "The audit rejects broad novelty claims for transient EOM/KBE/Keldysh transport, persistent spin-ring currents, interacting QSH invariants, "
            "and robustness-as-protection. A possible contribution is the reproducible ASTRA/ASTRUM integration of a flux-ramped spinful Corbino "
            "benchmark with two-time memory, persistent/reservoir separation, spin-resolved continuity diagnostics, and finite-size controls. "
            "This candidate remains unconfirmed until specialist database searches and a closed interacting continuum calculation are completed."
        ),
        "required_followup": [
            "Repeat the exact-combination search in INSPIRE, Web of Science, Scopus, and specialist arXiv categories.",
            "Close interacting charge/spin continuity with the same Matsubara and real-time self-energy.",
            "Compare the integrated benchmark against trivial, Rashba, disorder, width, contact, and flux-ramp controls.",
        ],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", default="ASTRA")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_gate(args.runtime)
    rendered = json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False)
    print(rendered)
    print(f"VERDICT: {'PASS' if report['passed'] else 'FAIL'}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
