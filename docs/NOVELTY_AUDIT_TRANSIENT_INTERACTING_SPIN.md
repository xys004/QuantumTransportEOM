# Novelty audit after Gates 11–21

## Bottom line

The general ingredients are established in the literature: two-time Kadanoff–Baym transport through interacting open systems, conserving transient Meir–Wingreen currents, electron–phonon SCBA/KBE, persistent spin/charge currents in rings, and NEGF spin transport in Kane–Mele-like lattices. The project must therefore not claim novelty for “using Keldysh/KBE” or for persistent currents alone.

The defensible candidate is narrower:

> a reproducible finite-grid workflow that resolves the non-equilibrium interaction between an Aharonov–Bohm/persistent-current channel and reservoir-driven charge/spin channels in a spinful Kane–Mele Corbino annulus, with explicit two-time KBE/SCBA memory, smooth flux-gauge ramps, spin-torque bookkeeping, and an explicit test for—not an assumption of—topological protection.

This is a promising combination, not a demonstrated literature gap. The
targeted search record is in
[`NOVELTY_SEARCH_LOG_20260802.md`](NOVELTY_SEARCH_LOG_20260802.md). Gate21 now
provides a same-Hubbard-U exact benchmark in the atomic limit; a publication
claim still requires the lead-coupled extension, a specialist database search,
and production convergence before submission. Gate25 exposed the vertical
Keldysh branch, and Gate26 now closes that source microscopically for an exact
finite contacted benchmark. Gate27 adds finite-lead-size convergence before
recurrences and resolves a spin-current channel. The compact continuum
Corbino residual and the interacting lead closure remain open. Gate28 adds a
stable spectral KMS evaluation for a finite-star/WBL comparison, but its UV
error remains explicit. Gate29 shows that the compact Corbino residual has a
small but resolved spin component while the charge source remains large and
microscopic closure is open. These finite/reference benchmarks are validation
oracles, not new physical effects by themselves. Gate31 now adds an exact
lead-coupled finite-U many-body oracle and quantifies the Hubbard-I/EOM error.
Gate32 adds a reproducible symbolic and self-consistent numerical
Hubbard-second-Born two-time layer, but its exact finite-U density gap and raw
continuity residual are reported as approximation/source diagnostics; the
interacting continuum closure is still not established. Gate33 makes the
instantaneous Hartree term explicit and refinement-tested, while confirming
that this does not close the interacting vertical-branch source. Gate34 now
attaches the microscopic mixed source to the continuity API for the quadratic
finite-lead reference; this is not transferred to the interacting second-Born
run without an explicit mixed self-energy. Gate35 supplies that mixed
second-Born product and its (U^2) scaling on the quadratic branch, but the
mixed Green functions are not yet self-consistent in the interacting run.
Gate36 adds a nested real/vertical grid convergence study for that seeded
source; this is a numerical quadrature control, not an interacting-contour
claim. Gate37 adds finite-lead-size convergence for the seeded source before
recurrence, still without claiming an interacting continuum result. Gate38
checks causal memory-window extension; the growth of the source norm is kept as
a finite-memory diagnostic, not called irreversible relaxation.
Gate39 audits the entire Gates31–38 evidence block and keeps the publication
and topological-protection boundaries explicit.
Gate41 closes an important oracle gap by calculating the exact finite-U mixed
branch; it shows that the interacting source differs materially from the
quadratic seed, while leaving the self-consistent contour closure open.
Gate42 confirms that attaching this exact source to the approximate real-time
SCBA solution worsens continuity, establishing a reproducible criterion for
the joint contour solver still required. Gate43 quantifies the charge/spin
source error against the exact interacting oracle, providing a falsifiable
closure budget rather than a positive claim.
Gate44 scans that budget from weak to intermediate \(U\): the mixed second-Born
source is resolved at finite coupling, but its charge and spin discrepancies
remain measurable and grow by \(U=0.8\). The scan therefore supplies a
negative range-of-validity result; it does not establish a controlled
perturbative window or a topological effect.
Gate45 makes the mixed real/vertical Kadanoff--Baym differential equations
explicit in the public symbolic API, including the causal and imaginary
contour measures. This is an implementation upgrade and reproducibility
contract, not a claim of solver convergence or method novelty.
Gate46 adds a stable equilibrium Matsubara constructor and exposes (G^M) and
the embedding (Σ^M) in the finite reference. This closes a branch-input
gap for a joint contour implementation, while leaving interacting
self-consistency and the publication claim open.
Gate47 adds a finite-grid residual evaluator for both mixed KBE orientations.
The free control closes while the contacted benchmark retains a resolved
nonzero residual, so the evaluator strengthens falsifiability without
claiming a conserving interacting contour solve.
Gate48 exposes charge and spin projections of the microscopic vertical source
as a reusable API. The finite spin component is a resolved observable for the
benchmark, not a topological-protection or novelty claim.
Gate49 audits Gates44–48 as one evidence block, including the regression
sequence and claim boundaries. The block is reproducible on ASTRA/ASTRUM but
does not close the interacting contour or establish topological protection.
Gate51 adds a causal Volterra propagator for the mixed branch, controlled by a
free unitary benchmark and an explicit contacted residual. It is a solver
component, not evidence of a conserving interacting contour result.
Gate52 couples that propagator to a self-consistent Hubbard second-Born
iteration on one real/imaginary grid. The mixed branch responds at finite
interaction, but the supplied real lesser branch remains an explicit bare
input, so the result is a bounded research layer rather than a conserving
contour solve.
Gate53 exposes the resulting microscopic mixed source and its charge/spin
projections, then compares it to the source required by the real lesser
continuity residual. The mismatch is reproducible on ASTRA and ASTRUM and is
kept as a negative closure diagnostic. It does not establish conservation,
topological protection, or method novelty.
Gate54 propagates that source with the retarded branch and adds the explicit
anti-Hermitian lesser correction. The benchmark shows a residual ratio of
1.355 after the correction, so this is a falsifiable implementation upgrade,
not a conserving result. The negative outcome is retained as part of the
publication boundary.
Gate55 adds a named charge/spin Meir--Wingreen contract over the same two-time
 kernels. Gate56 maps it to inner/outer Kane--Mele Corbino reservoirs, and
 Gate57 keeps those reservoir channels separate from the flux-conjugate
 persistent response. Gate58 applies ideal, Rashba, and disorder controls;
 the responses are finite but perturbation-sensitive. Gate59 audits this block
 on ASTRA/ASTRUM with full regression counts. These gates establish a useful
reproducible observable workflow, not a new conservation theorem or a
topological-protection claim.
Gate60 reviews the ten-gate block. The regression and evidence records are
clean on ASTRA/ASTRUM, but the residual ratio 1.355, finite-lead scope, and
open prior-art search keep the publication verdict bounded. The correct next
step is closure/continuum validation, not a stronger protection claim.

## Prior-art boundaries

| Existing result | Consequence for our claim |
|---|---|
| Myöhänen, Stan, Stefanucci, and van Leeuwen, *Kadanoff–Baym approach to quantum transport through interacting nanoscale systems* ([arXiv:0906.2136](https://arxiv.org/abs/0906.2136)) | Two-time interacting transient transport, conserving approximations, and time-domain Meir–Wingreen are prior art. |
| Säkkinen, Peng, Appel, and van Leeuwen, *Many-body Green's function theory for electron-phonon interactions: the Kadanoff–Baym approach to spectral properties of the Holstein dimer* ([arXiv:1507.04726](https://arxiv.org/abs/1507.04726)) | Electron–phonon KBE/SCBA and initial-correlation issues are prior art. |
| Pournaghavi et al., *Quantum transport by spin-polarized edge states in graphene nanoribbons in the quantum spin Hall and quantum anomalous Hall regimes* ([arXiv:1805.02418](https://arxiv.org/abs/1805.02418)) | NEGF/Keldysh spin-resolved QSH edge transport is prior art. |
| Crépin and Trauzettel, *Flux sensitivity of quantum spin Hall rings* ([arXiv:1507.03898](https://arxiv.org/abs/1507.03898)) | Flux-dependent persistent currents in QSH rings are prior art. |
| Maiti, Dey, and Karmakar, *Persistent charge and spin currents in a quantum ring using Green's function technique* ([arXiv:1401.0262](https://arxiv.org/abs/1401.0262)) | Persistent charge/spin currents with spin–orbit coupling and Green functions are prior art. |

## Required publication gates still open

1. Replace the finite-grid fixed point with a documented contour/Volterra convergence study that includes lead memory and initial correlations in the same interacting run. Gates27–29 control finite quadratic and WBL/reference diagnostics, but not the interacting continuum closure.
2. Extend the Gate31 exact reference from the six-mode finite contact to a controlled lead-size/continuum extrapolation or trusted many-body solver for the spatially extended device.
3. Close the interacting charge and spin continuity equations, including reservoir injection and Rashba/spin-orbit torque, for the Kane–Mele device.
4. Perform a targeted literature search (INSPIRE, Web of Science, Scopus, arXiv) for the exact combination “flux-ramped Corbino/Kane–Mele + persistent current + interacting two-time spin transport”.
5. Demonstrate a falsifiable observable that is not just a generic transient: e.g. a memory-dependent crossover or a quantified persistent/reservoir current decomposition that survives width, time-step, contact, disorder, and topology controls.

## Current status

`AUDIT_PASS_WITH_OPEN_PUBLICATION_GATES`: the implementation is reproducible and technically substantial, including a bounded two-time second-Born layer, but no novelty or topological-protection claim is released yet.

## Gate74 specialist audit

Gate74 broadens the prior-art check to the method, observable, and protection
claims separately. Two-time interacting Kadanoff–Baym transport with embedding
and time-dependent fields is established by Myöhänen et al.; persistent
charge/spin ring currents by Green functions are established by Maiti et al.
and flux-sensitive QSH-ring currents by Crépin and Trauzettel; interacting QSH
invariants are established by Grandi et al.; and robust edge currents without
band topology are demonstrated by Mitchison et al. The conserving-approximation
boundary follows Baym–Kadanoff. Consequently, neither “Keldysh transient
transport”, “persistent spin current”, “interacting QSH invariant”, nor
“robust edge current” is a standalone novelty claim for this project.

The residual candidate is narrower: a reproducible ASTRA/ASTRUM integration of
a flux-ramped spinful Corbino benchmark that keeps two-time memory,
persistent/reservoir currents, spin-resolved continuity, and finite-size
controls in one auditable workflow. Gate74 labels this a candidate only; it
does not assert novelty, topology, or a conservation theorem. The source matrix
and exact claim boundary are recorded in
`docs/evidence/gate74_specialist_novelty_audit_20260802.json` and the search
log. The publication decision remains conditional on specialist database
searches and closure of the interacting continuum continuity equations.

## Gate75 publication boundary

The Gate75 matrix records 264 engine tests and 70 application tests passing in
both ASTRA and ASTRUM. The permitted manuscript claim is therefore a
reproducible finite-grid two-time EOM/Keldysh software and benchmark workflow,
with the self-consistent Matsubara branch documented as an implementation
upgrade. A conserving interacting continuum theorem, broad method novelty,
topological protection, and strong new physics remain explicitly not ready or
unconfirmed. The machine-readable matrix is
`docs/evidence/gate75_publication_claim_matrix_20260802.json`.

## Gate76 accuracy boundary

The exact finite-contact scan at (U=0,0.1,0.3,0.5,0.8) conserves charge and
the selected spin component to machine precision. The self-consistent
Matsubara plus real-time second-Born branch converges at weak coupling, while
the stronger-(U) rows expose a finite-grid Matsubara boundary and growing
lesser closure residual. This strengthens the falsifiability of the project
but does not convert the residual into a topological or publication-grade
conservation claim. The detailed rows are in
`docs/evidence/gate76_exact_interaction_accuracy_ledger_20260802.json`.

## Gate80 review decision

The ten-gate review (Gates71–79) passes in both runtimes with 264 engine tests
and 70 application tests. The defensible manuscript result is a reproducible
finite-grid two-time EOM/Green/Keldysh software and benchmark workflow with
charge, spin, memory, ramp, and persistent/reservoir diagnostics. The review
does not authorize broad method novelty, a conserving interacting-continuum
theorem, or topological protection; those remain `NOT_READY`, and the narrow
integrated novelty candidate remains `UNCONFIRMED`. See
`docs/evidence/gate80_review_20260802.json`.

## Gate81 continuity upgrade

The new component ledger uses the same embedding plus interaction self-energy
in the Green branch and in the charge/spin continuity audit. Its algebraic
additivity closes, but the finite-grid residual and vertical-source
diagnostics remain resolved. This strengthens falsifiability and prevents
post-hoc channel mixing; it does not establish a conserving continuum result
or alter the novelty/protection boundary.

## Gate85 symbolic continuity boundary

Gate85 closes a missing software-facing analytic layer: the package emits the
ordered four-term Kadanoff--Baym collision kernel, its equal-time limit, and a
continuity identity with optional vertical source. Formal charge and spin
operator traces expose coherent torque and collision channels while preserving
matrix order. This improves falsifiability and reproducibility, but it is an
identity generator rather than a new conservation theorem; the numerical
second-Born residual and the topological-protection question remain open. See
`docs/evidence/gate85_symbolic_continuity_20260803.json`.

The Gate86 regression refresh passes with 267 engine and 70 application tests
in both ASTRA and ASTRUM. This confirms that the analytic API is integrated
without regressing the existing transport layer; it does not change the
novelty or protection verdict.

## Gates87--89 final-round boundary

The exact finite-contact noncommuting-spin oracle and the finite Corbino Rashba
control both pass in ASTRA and ASTRUM. They establish that the package can
separate coherent spin torque, spin-current divergence, reservoir injection,
and charge continuity in transient runs. Gate89 binds these records with the
267-engine/70-application regression baseline. This is a stronger and more
falsifiable software/benchmark result, but it still does not supply a new
topological invariant, a conserving interacting continuum limit, or broad
method novelty. See `docs/evidence/gate89_package_refresh_20260803.json`.

## Gate90 final decision

The final ten-gate review passes in both runtimes with the 267/70 regression
baseline. The defensible result is a reproducible finite-grid transient
EOM/Green/Keldysh charge-spin workflow, with explicit symbolic continuity and
torque audits. It is ready to support a draft with limitations. The broad
method claim is rejected by prior art; narrow integrated novelty is
unconfirmed; conserving interacting-continuum and topological-protection
claims remain not ready. The implementation phase is therefore paused for a
joint project analysis. See `docs/evidence/gate90_final_review_20260803.json`.
