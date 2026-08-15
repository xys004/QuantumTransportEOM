# Targeted novelty search log — 2026-08-02

This is a reproducible search record, not a proof of novelty. Queries were
issued against the web/arXiv search layer on 2026-08-02:

1. `"Kane-Mele" Corbino transient interacting Keldysh spin transport flux persistent current`
2. `"persistent current" "Kane-Mele" Corbino interacting Green function`
3. `"flux-ramped" Corbino spin transport Kadanoff Baym`
4. `"persistent" reservoir spin current quantum spin Hall ring nonequilibrium`

Closest directly relevant records returned:

- [Sun, Xie, and Wang, Persistent spin current in nano-devices](https://arxiv.org/abs/0806.2178): persistent spin current with spin–orbit coupling, its non-conservation, and the distinction from transport spin current.
- [Zegarra, Egues, and Chen, Persistent currents and spin torque caused by a percolated quantum spin Hall state](https://arxiv.org/abs/2001.01081): persistent charge/spin currents and spin torque near a QSH/ferromagnet interface.
- [Crépin and Trauzettel, Flux sensitivity of quantum spin Hall rings](https://arxiv.org/abs/1507.03898): flux-dependent many-body persistent currents in QSH rings.
- [Finsterhölzl, Katzer, and Carmele, Nonequilibrium non-Markovian steady states in open quantum many-body systems](https://doi.org/10.1103/PhysRevB.102.174309): structured-reservoir memory and persistent oscillations in an open spin chain.

No first-page result in these targeted searches combines all of the following
in one reproducible calculation: a flux-ramped Kane–Mele/Corbino annulus,
persistent and reservoir currents, interacting two-time KBE/SCBA memory, and
charge/spin/torque continuity. This is only a candidate gap: the search must be
repeated in INSPIRE, Web of Science, Scopus, and specialist arXiv categories
before a manuscript claims novelty.

## Gate74 specialist primary-source audit — 2026-08-02

The targeted search was expanded from the exact device wording to the method,
observable, and claim-boundary components. The following primary records set
the negative boundaries for the project:

- [Myöhänen et al., Kadanoff–Baym approach to quantum transport through interacting nanoscale systems](https://arxiv.org/abs/0906.2136) and [Kadanoff–Baym approach to quantum transport in AC/DC fields](https://arxiv.org/abs/1006.2912) cover two-time interacting transient transport, conserving approximations, embedding, and time-dependent driving.
- [Maiti, Dey, and Karmakar, Persistent charge and spin currents in a quantum ring using Green's function technique](https://arxiv.org/abs/1401.0262) and [Crépin and Trauzettel, Flux sensitivity of quantum spin Hall rings](https://arxiv.org/abs/1507.03898) cover persistent charge/spin ring observables and flux-sensitive QSH-ring response.
- [Grandi et al., Topological invariants in interacting quantum spin Hall systems](https://doi.org/10.1088/1367-2630/17/2/023004) covers interacting QSH invariants, so an interacting Kane–Mele invariant is not an unoccupied claim.
- [Mitchison, Rivas, and Martin-Delgado, Robust Nonequilibrium Edge Currents with and without Band Topology](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.128.120403) shows why a robust edge signal must be compared with a trivial control before it is called topological protection.
- [Baym and Kadanoff, Conservation Laws and Correlation Functions](https://doi.org/10.1103/PhysRev.124.287) fixes the conserving-approximation boundary: continuity closure is a criterion to verify, not a novelty claim by itself.

Gate74 therefore rejects broad method novelty and any inference from robustness
alone to topology. The only remaining candidate is a narrow, reproducible
ASTRA/ASTRUM benchmark integration: flux-ramped spinful Corbino geometry,
two-time memory, explicit persistent-versus-reservoir currents, spin-resolved
continuity diagnostics, and finite-size controls. This candidate is
**unconfirmed** until specialist database searches and a closed interacting
continuum calculation are complete. The machine-checkable matrix is in
`docs/evidence/gate74_specialist_novelty_audit_20260802.json`.

## Focused transient Corbino/Xene audit — 2026-08-03

The search was broadened to the exact neighboring literatures rather than only
the project wording: equilibrium graphene Corbino rings with Rashba, dynamic
spin/valley pumping in silicene, transient AB interferometers, transient
spin-dependent AB transport with SOI, transient QSH spin currents in
stanene/silicene-like systems, and two-time Kadanoff–Baym/Keldysh transport.

The result is a sharper boundary. Every individual block has prior art, so the
following claims are rejected: new Keldysh/EOM method, first Xene persistent
current, first Rashba transient spin current, and topological protection from
robustness or `Z2=1` alone. No exact primary-source match was found for the
integrated finite-Xene-Corbino protocol with a smooth AB ramp, reservoirs,
persistent/reservoir separation, inner/outer/bulk decomposition and explicit
Rashba torque. This is a **narrow candidate, not a priority claim**; the exact
search record and machine checks are in:

- `docs/NOVELTY_AUDIT_TRANSIENT_CORBINO_XENE_20260803.md`;
- `docs/evidence/transient_novelty_matrix_20260803.json`;
- `docs/evidence/transient_novelty_audit_20260803.json`.
