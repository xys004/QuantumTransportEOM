# Gate protocol: EOM, Keldysh, transients, and spin transport

## Scientific objective

QuantumTransportEOM is the central symbolic/numerical engine for the transport
project. The target is a reproducible transient calculation of charge and spin
observables from equations of motion, two-time Green functions, and Keldysh
self-energies. No topological or novelty claim is accepted merely because a
current is large or numerically robust.

The application layer xene-ring-transport supplies rings, flux profiles,
Corbino geometries, and physical observables. The engine supplies the
Hamiltonian/EOM/Keldysh propagation and analytical oracles.

## Regeneration note (2026-08-15)

Gate 31 was re-run and regenerated separately, after
`lead_coupled_hubbard_i_retarded` was changed to solve the Dyson equation
\(G=[g_{\rm at}^{-1}-\Sigma]^{-1}\) instead of inserting the embedding into
each atomic denominator.  Its finite-U mismatch moves from 6.50 to 8.07 and
the gate now reports both forms; see the Gate 31 section.  That change added
one regression test, so the Gate 31 record cites a 303-test engine suite while
the six records below cite 302 — they were verified before it.

Gates 47, 51, 54, 61, 62 and 63 were re-run and their evidence regenerated as
`*_20260815.json` after a correction to the causal quadrature in
`initial_correlations._prefix_trapezoid_weights`.  Retarded and advanced
kernels are stored with the `theta(0)=1/2` convention, so applying the plain
trapezoid endpoint weight on top of it halved the equal-time contribution
twice and degraded every causal convolution from second to first order
(measured against an analytic one-level reference: error ratio 2.0 per grid
halving instead of 4.0).  All six gates still pass on ASTRA and ASTRUM, which
agree to \(7.4\times10^{-14}\); the engine suite is 302 tests on both runtimes
and the application suite is 82.

The superseded `*_20260802.json` records are retained: they remain the valid
output of the pre-correction code, and the block audits G49, G59, G69 and G75
still reference them, since those audits pin a dated regression sequence
(248→252 engine tests, 69 application tests) that describes that block and not
the present state.  Raw ASTRA/ASTRUM verifier logs and the assembly script for
the regenerated records are deposited in `docs/evidence/logs_20260815/`.

One conclusion changed rather than merely shifting: the Gate 54 and Gate 61
closure ratio falls from 1.355 to 1.042.  The absolute residuals roughly
triple because the corrected quadrature no longer cancels part of the closure
error against a first-order discretisation error, but the vertical term is now
close to neutral instead of degrading the balance by a third.  The Gate 59 and
Gate 60 review sections below still quote 1.355 because they are dated
snapshots of that block; they are not restated here.

## Gate sequence

Each gate has four mandatory records:

1. a falsifiable statement;
2. a minimal numerical or symbolic oracle;
3. an ASTRA verifier with explicit CHECK lines;
4. an ASTRUM reproduction for the nontrivial scan.

After each block of ten gates, stop for a joint review of token budget,
evidence, scope, and publication novelty before starting another block.

| Gate | Claim to establish | Required evidence |
|---|---|---|
| G1 | The public engine contract is complete and importable | EOM, Keldysh, two-time, and spin API inventory |
| G2 | Two-time G has correct equal-time, adjoint, causal, and spectral identities | exact one-level and finite-matrix oracles |
| G3 | Symbolic EOM/Keldysh expressions can be generated and reduced analytically | closure, Langreth, Dyson, FDT, and LaTeX artefacts |
| G4 | Charge and spin currents have independent continuity/torque balances | lead, bond, spin, and torque residuals |
| G5 | Kane–Mele/Corbino is represented consistently in the engine | gauge, Rashba, contact, and spin-basis controls |
| G6 | Numerical transients reproduce analytic limits | resonant level, clean ring, WBL, and Landauer limits |
| G7 | The physical parameter atlas is converged | energy/time resolution and contact/cutoff controls |
| G8 | Robustness is tested rather than assumed | disorder seeds, Rashba, Zeeman, and matched trivial controls |
| G9 | Topological protection is tested as a separate hypothesis | bulk invariant, bulk–edge scaling, scattering invariant |
| G10 | The result is publication-ready or explicitly negative | ASTRA/ASTRUM audit, novelty matrix, limitations, token review |

## Gates 11–20: interacting two-time and publication audit block

The second block was run with the same ASTRA verifier convention and reproduced
on ASTRUM. A passing gate records an implemented, bounded capability; it does
not by itself establish a new physical effect or topological protection.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G11 | Baseline post-G10 contract and regression inventory are reproducible | PASS; engine/app regression records |
| G12 | Langreth and Kadanoff–Baym equations are generated symbolically | PASS; `gate12_kadanoff_baym_symbolic_20260802.json` |
| G13 | Finite-grid two-time KBE and electron–boson SCBA satisfy causal/adjoint identities | PASS; `gate13_kadanoff_baym_numeric_scba_20260802.json` |
| G14 | Interaction-memory diagnostics expose spectral and density consistency | PASS; covered by the G13 numerical verifier |
| G15 | Two-time Meir–Wingreen charge and arbitrary Hermitian spin currents are computed | PASS; `gate15_two_time_charge_spin_currents_20260802.json` |
| G16 | A finite-band Lorentzian reservoir memory has an analytic kernel and smooth gauge dressing | PASS; `gate16_analytic_reservoir_memory_20260802.json` |
| G17 | Hubbard-I EOM and electron–boson SCBA are compared in a controlled benchmark | PASS; `gate17_eom_hubbard_i_vs_scba_20260802.json` |
| G18 | Kane–Mele/Corbino exposes the interacting two-time charge/spin adapter | PASS; `gate18_kane_mele_interacting_spin_20260802.json` |
| G19 | Coupling/frequency memory sweeps reproduce on ASTRUM | PASS; `gate19_astrum_interaction_memory_sweep_20260802.json` |
| G20 | Publicability audit is reproducible and claim-bounded | PASS WITH OPEN PUBLICATION GATES; `gate20_publicability_audit_20260802.json` |

## Gate 21: same-Hubbard-U exact benchmark

Gate21 closes the first publication gate from the previous audit, but only in
the atomic diagonal limit. It is intentionally not promoted to a claim about
lead-coupled or spatially extended Hubbard dynamics.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G21 | Exact four-sector atomic Hubbard-U Green functions agree with EOM/Hubbard-I using identical parameters | PASS; `gate21_same_hubbard_u_exact_20260802.json` |

## Gates 22–23: continuity diagnostics and production resolution

| Gate | Claim established | Status and evidence |
|---|---|---|
| G22 | Finite-grid KBE collision integral closes charge and Hermitian spin projections, including a nonzero self-energy test | PASS; `gate22_continuity_diagnostics_20260802.json` |
| G23 | Compact Corbino production run passes density physicality, continuity, and energy-refinement checks on ASTRA/ASTRUM | PASS; application `gate23_production_convergence_20260802.json` |

## Gate 24: publicability re-audit

| Gate | Claim established | Status and evidence |
|---|---|---|
| G24 | The accumulated evidence and claim boundaries remain reproducible after Gates21–23 | PASS WITH LEAD-COUPLED CONTINUITY AND NOVELTY GATES OPEN; `gate24_publicability_reaudit_20260802.json` |

## Gate 25: vertical-branch initial-correlation diagnostic

Gate25 adds the missing mixed real--imaginary Keldysh branch to the engine.
The symbolic API writes the vertical-contour integral explicitly and the
numeric API evaluates it on a finite imaginary-time grid.  The gate also
reconstructs the partition-free lead kernels for the compact Corbino run.  The
required source is finite and Hermitian, but its magnitude is about
``6.85e-2`` throughout the transient, so the microscopic lead initial-
correlation closure is still open.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G25 | The vertical-branch API is internally consistent and the missing source is exposed without overclaiming closure | PASS DIAGNOSTIC WITH MICROSCOPIC LEAD INITIAL CORRELATION OPEN; `gate25_initial_correlation_branch_20260802.json` |

## Gate 26: microscopic finite-lead initial-contact closure

Gate26 replaces the residual-inferred source with a microscopic construction.
The complete contacted device-plus-finite-leads Hamiltonian is equilibrated
before the quench.  Its exact full propagation supplies the device Green
functions, lead embedding kernels, and the mixed real--imaginary kernels.  The
resulting source closes both charge and spin projections in the interior of a
refined time grid at the (10^{-7}) level.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G26 | Microscopic vertical-branch initial correlations close charge and spin continuity in an exact finite contacted benchmark | PASS; `gate26_finite_lead_initial_closure_20260802.json` |

## Gate 27: lead-size and recurrence control

Gate27 uses two finite one-dimensional spinful leads (sizes 2, 3, 4, and 6)
coupled to a two-orbital device. The full contacted quadratic system is
propagated exactly after a device quench and opposite lead shifts. The exact
full-system density derivative is compared with the finite-grid KBE collision
integral and the microscopic mixed-branch source; the corrected closure is
below \(8.0\times10^{-8}\) on every size. With hopping \(J=0.22\), the
fastest round-trip signal gives the conservative bound
\(t_{\rm rec}\ge L/J=9.09\) for the smallest lead, so the \([0,1]\) window is
pre-recurrence. At \(L=4\), the device density, charge current, and spin
current agree with \(L=6\) to \(6.7\times10^{-8}\), \(3.1\times10^{-8}\), and
\(1.6\times10^{-8}\), respectively. This is a finite quadratic oracle; it
does not establish a continuum/WBL interacting limit or topological
protection.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G27 | Finite spinful leads converge before recurrences, with exact microscopic source closure and resolved charge/spin currents | PASS; `gate27_lead_size_recurrence_20260802.json` |

The gate is deliberately bounded: finite-lead recurrences beyond this window,
the continuum WBL interacting Corbino limit, and any topological claim remain
separate gates.

## Gate 28: continuum/WBL initial-source boundary

Gate28 introduces a midpoint star quadrature for a positive-semidefinite flat
band broadening and compares its exact finite contacted propagation with the
analytic partition-free WBL matrix-quench reference.  The KMS products on the
vertical branch are evaluated directly in the spectral basis, avoiding the
overflow/cancellation that appears when \(e^{-h\tau}(1-f)\) and
\(f e^{h\tau}\) are formed separately.  For half-bandwidths 8, 12, and 16,
the star/WBL density errors decrease from \(3.98\times10^{-3}\) to
\(1.99\times10^{-3}\); the WBL particle-number continuity error is below
\(1.4\times10^{-5}\).  The remaining UV discrepancy is deliberately recorded
as open rather than absorbed into a residual-inferred source.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G28 | The analytic WBL reference and a stable finite-star mixed source are reproducible, with an explicit nonzero UV error boundary | PASS WITH CONTINUUM LIMIT OPEN; `gate28_continuum_wbl_initial_source_20260802.json` |

## Gate 29: Corbino mixed-source and spin diagnostic

Gate29 applies the residual-source audit API to the compact continuous-lead
Kane--Mele Corbino transient.  With the two-time embedding kernels rebuilt at
321 and 641 energies, the required Hermitian source changes by only
\(1.50\times10^{-4}\) in matrix norm and its spin projection is nonzero
(\(1.66\times10^{-4}\) and \(1.83\times10^{-4}\)).  The finite-energy
two-time consistency bound is \(1.11\times10^{-2}\), while the required charge
source remains \(1.15\), so the microscopic continuum initial-contact closure
is still open.  The source in this gate is explicitly residual-inferred and is
not treated as a microscopic or topological result.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G29 | Corbino charge/spin residual and the energy-refined required source are reproducible, with the missing microscopic charge closure exposed | PASS DIAGNOSTIC WITH CONTINUUM MICROSCOPIC CLOSURE OPEN; `gate29_corbino_mixed_source_spin_diagnostic_20260802.json` |

## Gate 31: exact interacting lead-coupled oracle

Gate31 adds a finite Fock-space reference for a two-spin-orbital Hubbard
device coupled to two finite spinful leads.  The complete contacted many-body
state is equilibrated, quenched, and propagated exactly; its two-time Green
functions obey the spectral identity, while charge and spin balances close at
below \(2\times10^{-18}\).  A lead-embedded Hubbard-I formula reproduces the
U=0 control at \(2.1\times10^{-13}\), but its finite-U discrepancy is 8.07 for
the chosen parameters.  That discrepancy is an auditable approximation
error, not a new physical signal or a topological claim.

The gate now records the discrepancy for both ways of inserting the embedding
into the atomic propagator, because the choice is not a detail.  Solving the
Dyson equation \(G=[g_{\rm at}^{-1}-\Sigma]^{-1}\), which is what lead-coupled
Hubbard-I conventionally means, gives 8.07; the two-pole ansatz that inserts
\(\Sigma\) into each atomic denominator separately gives 6.50, the value
recorded before 2026-08-15.  The two coincide exactly at \(n_o\in\{0,1\}\) and
at \(U=0\), so the non-interacting control in this gate cannot distinguish
them; roughly 1.6 of the reported mismatch is the insertion choice rather than
the Hubbard-I approximation itself.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G31 | A finite interacting contacted reference exists for two-time charge/spin transport, and the lead-coupled Hubbard-I/EOM error is quantified against it | PASS; `gate31_exact_interacting_lead_coupled_20260815.json` (supersedes `..._20260802.json`) |

## Gate 32: Hubbard second-Born two-time closure

Gate32 adds the first explicit interacting correlation self-energy to the
finite-grid KBE layer.  The symbolic API records the local density-density
closure

\[
\Sigma_s^{<}(t,t')=U^2G_s^{<}(t,t')G_{\bar s}^{<}(t,t')
G_{\bar s}^{>}(t',t),
\]

with the corresponding greater kernel and causal retarded discontinuity.  A
self-consistent Dyson/Volterra iteration is then run on the contacted
finite-lead partition-free benchmark.  It converges in 44 iterations at
\(8.4\times10^{-8}\), with Green spectral error
\(1.7\times10^{-18}\) and self-energy spectral error
\(2.8\times10^{-11}\) on the local run; ASTRUM reproduces the same values.
The exact six-mode Hubbard oracle gives a maximum density gap of
\(1.78\times10^{-1}\), which is retained as an approximation error rather
than interpreted as a physical signal.  The raw KBE continuity residuals are
\(2.23\times10^{-2}\) (charge) and \(9.29\times10^{-3}\) (spin); they remain
open because this correlation-only layer does not yet include the
instantaneous Hartree term and the full interacting vertical-branch source.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G32 | Symbolic and numerical Hubbard second-Born self-energy is reproducible in two time variables, converges on a contacted benchmark, and is quantitatively compared with an exact finite-U oracle | PASS WITH INTERACTING SOURCE CLOSURE OPEN; `gate32_hubbard_second_born_two_time_20260802.json` |

## Gate 33: explicit Hartree layer and source boundary

Gate33 separates the instantaneous Hubbard contribution from the memory
self-energy.  The symbolic API returns
\(\Sigma_{H,s}^{r,a}(t,t')=U n_{\bar s}(t)\delta(t-t')\), while the numerical
API collocates the delta on the trapezoidal grid and returns zero lesser and
greater Hartree kernels.  A static one-body quench gives errors
\(5.09\times10^{-3}\), \(2.55\times10^{-3}\), and
\(1.27\times10^{-3}\) for 9, 17, and 33 time points.  Enabling Hartree in
the contacted second-Born run converges in 58 iterations at
\(8.7\times10^{-8}\), with Green spectral error
\(1.4\times10^{-17}\).  The finite-U density gap remains
\(1.78\times10^{-1}\), so the layer is a controlled approximation component,
not an exact interacting closure.  Charge/spin residuals are retained as
source diagnostics (\(5.43\times10^{-2}\) and \(6.84\times10^{-3}\)); the
vertical interacting branch and endpoint delta treatment are still open.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G33 | Hartree is explicit, symbolically defined, and time-refinement tested; its inclusion in the second-Born KBE is reproducible while the vertical source remains explicit as an open boundary | PASS WITH SOURCE CLOSURE OPEN; `gate33_hartree_collocation_vertical_source_20260802.json` |

## Gate 34: microscopic vertical-source attachment

Gate34 connects the finite-lead mixed branch \(\Sigma^{\rceil}G^{\lceil}\) to
the continuity diagnostic without overwriting the raw real-time residual.  On
the same quadratic contacted benchmark, the raw matrix residual is
\(1.28\times10^{-2}\), while subtracting the microscopic source gives
\(1.15\times10^{-4}\); the charge and spin projections fall to
\(2.17\times10^{-4}\) and \(1.34\times10^{-5}\), respectively.  The API now
exposes both ``residual`` and ``source_corrected_residual``.  This is a
finite-grid quadratic closure control only: the interacting second-Born mixed
kernel is not inferred from the quadratic source.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G34 | A microscopic finite-lead vertical source can be attached and audited without hiding the raw continuity residual; charge and spin diagnostics improve at finite-grid accuracy | PASS QUADRATIC SOURCE REFERENCE; `gate34_vertical_source_continuity_attachment_20260802.json` |

## Gate 35: mixed Hubbard second-Born source kernel

Gate35 adds the mixed contour component
\[
\Sigma_s^{\rceil}(t,\tau)=U^2G_s^{\rceil}(t,\tau)
G_{\bar s}^{\rceil}(t,\tau)G_{\bar s}^{\lceil}(\tau,t),
\]
and exposes the finite-lead \(G^{\rceil}/G^{\lceil}\) arrays that seed it.  On
the quadratic branch the resulting source is Hermitian to machine precision;
its norm is (7.91\times10^{-4}) at \(U=0.25\) and
\(3.17\times10^{-3}\) at \(U=0.5\), with zero scaling residual against the
expected (U^2) law.  This is an explicit source kernel, not yet a
self-consistent interacting contour solution: the mixed Green functions used
in this gate are noninteracting reference kernels.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G35 | Mixed second-Born self-energy and its vertical contraction are explicit, Hermitian, and (U^2)-scaling tested on the reference branch | PASS WITH INTERACTING MIXED CLOSURE OPEN; `gate35_hubbard_second_born_mixed_source_20260802.json` |

## Gate 36: mixed-source grid convergence

Gate36 runs nested grids \((N_t,N_\tau)=(5,21),(9,41),(17,81)\) on the same
finite-lead reference.  The source norm changes only
\(5.25\times10^{-5}\) over the full range, while the coarse-to-fine nodewise
errors decrease from \(5.25\times10^{-5}\) to \(1.05\times10^{-5}\).  Every
source remains Hermitian to machine precision.  This certifies the numerical
quadrature of the seeded mixed product, not the self-consistency of an
interacting contour calculation.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G36 | The mixed source product refines on nested real/vertical grids with stable norm and Hermiticity | PASS SEEDED-REFERENCE CONVERGENCE; `gate36_mixed_source_grid_convergence_20260802.json` |

## Gate 37: mixed-source lead-size convergence

Gate37 repeats the seeded mixed source on spinful finite leads of lengths 2,
3, 4, and 6.  The source norm is stable near (2.27\times10^{-3}); the
size-4 versus size-6 discrepancy is (4.40\times10^{-9}), and size 3 versus
size 6 is (1.10\times10^{-7}).  All finite-lead spectral identities remain
below (2\times10^{-15}).  This is a pre-recurrence finite-reference
control, not an interacting continuum extrapolation.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G37 | Seeded mixed-source norms and matrices converge under finite lead-size refinement before recurrence | PASS FINITE-LEAD REFERENCE CONTROL; `gate37_mixed_source_lead_size_20260802.json` |

## Gate 38: mixed-source memory-window causality

Gate38 extends the same finite-lead run with a fixed \(\Delta t=0.0625\) and
windows \(t_{\max}=0.25,0.5,1.0\).  Extending the window leaves the already
computed prefix invariant to (3.4\times10^{-19}) and
(9.4\times10^{-18}), while the source norm grows from (1.35\times10^{-3})
to (3.78\times10^{-3}) as later times are included.  This establishes
Volterra causality and resolves memory in the finite reference; it is not an
irreversibility or continuum claim.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G38 | Mixed-source prefixes are causal under time-window extension and later memory is resolved | PASS MEMORY-WINDOW CONTROL; `gate38_mixed_source_memory_window_20260802.json` |

## Gate 39: integrated evidence and claim audit

Gate39 reads the evidence JSON for Gates31–38, verifies unique schemas,
complete local/ASTRUM check counts, the 247-engine/69-application regression
record, and the explicit open claim boundaries.  It also checks that the
novelty audit still carries `AUDIT_PASS_WITH_OPEN_PUBLICATION_GATES` and that
no topological-protection result has been released.  All 50 audit checks pass
on ASTRA; the same gate is executed on ASTRUM after synchronization.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G39 | Gates31–38 form a consistent, reproducible ASTRA/ASTRUM evidence block with bounded claims | PASS INTEGRATED AUDIT; `gate39_integrated_evidence_audit_20260802.json` |

## Gate 41: exact interacting mixed Keldysh branch

Gate41 extends the exact finite Fock-space oracle to calculate
\(G^{\lceil}(\tau,t)\) and \(G^{\rceil}(t,\tau)\) from the interacting
grand-canonical initial density.  At (U=0) the mixed branch matches the
finite-lead quadratic oracle at (9.1\times10^{-16}); at (U=0.5) it changes
by (1.86\times10^{-1}).  Feeding the exact interacting branch into the
second-Born mixed product gives a Hermitian source of (3.46\times10^{-2}),
which differs from the quadratic seed by (3.14\times10^{-2}).  The branch is
therefore no longer inferred from a residual, but the contour Dyson equation
is still not solved self-consistently.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G41 | Exact finite-U mixed Green functions are available and provide an interacting second-Born source seed with a validated U=0 control | PASS EXACT MIXED ORACLE; `gate41_exact_interacting_mixed_keldysh_branch_20260802.json` |

## Gate 42: joint-closure negative diagnostic

Gate42 pairs the exact interacting mixed source from Gate41 with the
real-time SCBA+Hartree solution.  The quadratic source/self-energy pairing
closes to (1.15\times10^{-4}), but the mismatched interacting pairing grows
the residual from (3.06\times10^{-2}) to (6.51\times10^{-2}), an amplification
of 2.13.  This is a useful negative result: a microscopic mixed source cannot
be appended after the fact to a different real-time approximation.  The two
branches must be solved jointly before claiming interacting conservation.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G42 | Mismatched exact-mixed/SCBA pairing fails reproducibly while the quadratic pairing closes; joint contour closure is required | PASS NEGATIVE DIAGNOSTIC; `gate42_interacting_source_pairing_diagnostic_20260802.json` |

## Gate 43: exact-interacting source error budget

Gate43 compares the source required by the exact finite-U real-time oracle
with the second-Born mixed source evaluated on the same exact
\(G^{\rceil}/G^{\lceil}\).  The exact oracle conserves charge and spin to
\(10^{-18}\), but the charge source mismatch is \(1.16\times10^{-1}\) and the
spin mismatch is \(7.04\times10^{-3}\).  The source-corrected matrix residual
reaches \(6.16\times10^{-2}\), so the missing contribution is resolved rather
than hidden in numerical noise.  This is a negative closure budget that
guides the next contour approximation.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G43 | Exact finite-U charge/spin conservation exposes a measurable second-Born mixed-source error budget | PASS SOURCE ERROR BUDGET; `gate43_exact_interacting_source_error_budget_20260802.json` |

## Gate 44: weak-to-intermediate interaction range

Gate44 scans the exact finite-Fock oracle over (U=0,0.1,0.2,0.3,0.5,0.8)
using the same contacted two-spin benchmark and compares the required
initial-correlation source with the mixed second-Born product.  The (U=0)
source vanishes as a control; at (U=0.1) the charge source is already
(1.08\times10^{-3}) and the spin error after subtracting the (U=0)
baseline is (2.33\times10^{-4}).  At (U=0.8), the charge error reaches
(1.49\times10^{-1}) and the spin error (2.53\times10^{-3}).  The exact
oracles remain spectral and every mixed source is Hermitian.  This is a
range-of-validity diagnostic, not a claim of a controlled perturbative window.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G44 | Weak-to-intermediate (U) resolves the onset and growth of second-Born charge/spin source error | PASS WEAK-(U) RANGE AUDIT; `gate44_weak_u_source_range_20260802.json` |

## Gate 45: explicit mixed Kadanoff--Baym equations

Gate45 adds `kadanoff_baym_mixed_equations` to the symbolic API.  It returns
the (G^{\rceil}) and (G^{\lceil}) differential equations with the causal
real-time convolution, the vertical convolution over (0\leq\tau\leq\beta),
the explicit (-i,d\tau) contour measure, and matrix multiplication order.
ASTRA and ASTRUM agree on all six structural checks, and the engine/app
regressions pass at 249/69 tests.  This removes an implicit equation from the
symbolic contract; it does not claim a converged interacting contour solver.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G45 | The mixed real/vertical KBE branch is explicit in the symbolic package | PASS EXPLICIT MIXED KBE; `gate45_mixed_kbe_symbolic_20260802.json` |

## Gate 46: Matsubara branch and embedding self-energy

Gate46 adds a numerically stable equilibrium (G^M(\tau,\tau')) constructor
and exposes both (G^M) and the finite-lead embedding (\Sigma^M) in
`FiniteLeadPartitionFreeResult`.  The mode-wise logarithmic evaluation avoids
overflow at the KMS endpoints.  ASTRA and ASTRUM reproduce the endpoint and
equal-time checks, and the 250-engine/69-application regressions pass.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G46 | The finite reference exposes the Matsubara Green and embedding branches on the same vertical grid | PASS EXPLICIT MATSUBARA BRANCH; `gate46_matsubara_branch_20260802.json` |

## Gate 47: numerical mixed-KBE residual diagnostic

Gate47 adds `mixed_kbe_residual`, which evaluates the two mixed differential
equations using causal real-time prefix quadrature and the full vertical
trapezoid rule. A homogeneous free control closes at
\(3.15\times10^{-4}\) on the finite-difference grid, unchanged by the
quadrature correction because that control carries no self-energy. The exact
finite contacted benchmark returns finite but nonzero residuals (0.1213 and
0.0853 for the two orientations); those values remain diagnostic evidence of
discretization/closure content and are not relabeled as a conserving
interacting solution.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G47 | Both mixed KBE branch residuals are numerically evaluable with explicit real/vertical quadrature | PASS MIXED-KBE RESIDUAL DIAGNOSTIC; `gate47_mixed_kbe_residual_20260815.json` (supersedes `..._20260802.json`) |

## Gate 48: charge/spin source projections

Gate48 adds public projections of the microscopic vertical source onto the
identity and an arbitrary spin operator.  In the finite contacted benchmark
the maximum charge projection is (1.665\times10^{-2}) and the (z)-spin
projection is (8.698\times10^{-3}), with zero source hermiticity error.
The API agrees with the direct trace and does not infer topological protection
from a nonzero spin channel.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G48 | Charge and spin components of the vertical source are directly exposed and tested | PASS CHARGE/SPIN SOURCE PROJECTION; `gate48_charge_spin_source_projection_20260802.json` |

## Gate 49: integrated two-time block audit

Gate49 reads the evidence for Gates44--48, checks unique gate IDs, complete
local/ASTRA/ASTRUM verdicts, the 248→252 engine regression sequence, the 69
application regression, finite metrics, and explicit claim boundaries. All
12 checks pass in both environments. The block is internally reproducible;
it still releases no topological-protection or interacting-conservation claim.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G49 | Gates44–48 form a coherent ASTRA/ASTRUM two-time implementation block with bounded claims | PASS INTEGRATED AUDIT; `gate49_interacting_two_time_block_audit_20260802.json` |

## Gate 51: mixed Volterra stepper

Gate51 adds `propagate_mixed_kbe_rceil`, a causal finite-grid propagator for
(G^\rceil(t,\tau)). It takes an explicit initial slice, retarded memory,
and vertical (Σ^\rceil G^M) source. The free unitary control has maximum
error (2.52\times10^{-4}), unchanged because the stepper's memory endpoint
lies strictly off the time diagonal and keeps the plain trapezoid weight; the
contacted finite reference gives a finite residual (8.84\times10^{-3}). This
is the first solver building block for a joint contour fixed point, not yet an
interacting conserving solution.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G51 | A causal mixed-branch Volterra propagator is implemented and controlled by a free benchmark | PASS MIXED VOLTERRA STEPPER; `gate51_mixed_volterra_stepper_20260815.json` (supersedes `..._20260802.json`) |

## Gate 52: joint Hubbard contour iteration

Gate52 couples the real-time Hubbard second-Born fixed point to the causal
mixed branch on the same finite real/imaginary grid through
`self_consistent_hubbard_second_born_contour_two_time`. The free control
converges in 52 iterations and the finite-(U=0.5) reference in 60
iterations. The mixed source reaches (6.96\times10^{-2}) and the mixed
branch changes by (2.23\times10^{-2}), so the interaction is visible in the
vertical branch. The real lesser branch still retains its supplied bare lesser
input; this is a joint-contour research layer, not a conserving closure.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G52 | Real and mixed Hubbard second-Born branches iterate self-consistently on one finite contour grid | PASS JOINT CONTOUR ITERATION; `gate52_joint_contour_iteration_20260802.json` |

## Gate 53: microscopic source attachment and negative closure test

Gate53 attaches the returned mixed self-energy/Green product to the
`InitialCorrelationResult` and projects it into charge and spin source
channels. The source is finite and Hermitian, but the required source inferred
from the real lesser continuity residual differs by (2.65\times10^{-2}).
The source-corrected residual is unchanged at the reported precision. This
pinpoints the missing vertical term in the real lesser equation rather than
hiding it in a post-hoc correction.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G53 | The joint result exposes a microscopic mixed source and a falsifiable charge/spin closure diagnostic | PASS ATTACHED SOURCE / NEGATIVE CLOSURE; `gate53_joint_contour_source_attachment_20260802.json` |

## Gate 54: propagated vertical term in the real lesser equation

Gate54 upgrades the lesser branch with the explicit finite-grid reconstruction

\[
I(t,t')=-i\int_0^\beta d\tau\,\Sigma^\rceil(t,\tau)G^\lceil(\tau,t'),
\qquad
C(t,t')=\int_{t_0}^{t}d\bar t\,G^R(t,\bar t)I(\bar t,t'),
\]

and adds \(C-C^\dagger\) to the real lesser Dyson update. The correction is
anti-Hermitian and changes the interacting branch (maximum
\(4.85\times10^{-2}\)). In the contacted finite benchmark the continuity
residual increases from \(7.32\times10^{-2}\) to \(7.63\times10^{-2}\)
(ratio 1.042). The result remains a controlled negative closure test — the
vertical term is necessary to expose the physics and does not by itself
conserve — but the corrected quadrature weakens it substantially: before the
correction the same comparison read \(2.65\times10^{-2}\) to
\(3.59\times10^{-2}\) with ratio 1.355, so what looked like a clear
degradation is closer to neutral once the first-order quadrature error is
removed from both sides.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G54 | The propagated mixed source is available in the real lesser equation and its conservation impact is measured | PASS EXPLICIT LESSER TERM / NEGATIVE CLOSURE; `gate54_lesser_vertical_initial_correlation_20260815.json` (supersedes `..._20260802.json`) |

## Gate 55: shared charge/spin Meir--Wingreen contract

Gate55 adds `two_time_meir_wingreen_charge_spin_currents`, which contracts the
same (G^{r,<}) and lead (Sigma^{a,<}) with the identity and caller-supplied
Hermitian spin operators. The synthetic two-orbital control resolves finite
charge, (S_x), (S_y), and (S_z) channels and keeps spin torque outside the
current definition.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G55 | Charge and named spin reservoir currents share one explicit two-time contraction | PASS CHARGE/SPIN CONTRACT; `gate55_charge_spin_meir_wingreen_20260802.json` |

## Gate 56: Kane--Mele Corbino reservoir mapping

Gate56 maps the engine contract to the spinful Corbino adapter. For each inner
and outer wide-band reservoir the application now constructs explicit
(Sigma^{r,a,<}(t,t')) and returns charge/(S_x)/(S_y)/(S_z) channels on the
selected two-time grid. The quench control resolves (S_z) at both reservoirs.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G56 | Two-time reservoir charge and spin channels are exposed by the Corbino application | PASS CORBINO MAPPING; `gate56_corbino_charge_spin_two_time_20260802.json` |

## Gate 57: persistent versus reservoir comparison

Gate57 adds `compare_corbino_two_time_with_persistent`. It returns the
flux-conjugate persistent response and the reservoir channels on a common grid
without adding them into a universal current. In the reference quench the
persistent response reaches (2.46\times10^{-2}), while the inner/outer charge
difference reaches (1.22\times10^{-1}) and the (S_z) difference
(7.77\times10^{-4}); these are distinct observables.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G57 | Persistent flux response and reservoir charge/spin currents are separately comparable | PASS SEPARATE CHANNELS; `gate57_persistent_reservoir_comparison_20260802.json` |

## Gate 58: spin robustness controls

Gate58 repeats the same protocol for ideal, weak-Rashba, and weak-disorder
controls. The inner (S_z) RMS changes from (6.259\times10^{-4}) (ideal) to
(6.244\times10^{-4}) (Rashba) and (6.234\times10^{-4}) (disorder), while
the persistent response changes slightly as well. The gate records these
sensitivities; it does not require a protection verdict.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G58 | Spin/reservoir and persistent responses are tested under explicit perturbation controls | PASS CONTROL ATLAS; `gate58_spin_robustness_controls_20260802.json` |

## Gate 59: integrated interacting-spin block

Gate59 runs the engine and application audits together and closes the current
implementation block. ASTRA and ASTRUM both report 257 engine tests and 69
application tests. The block is reproducible, but the Gate54 lesser residual
ratio (1.355) remains an explicit open closure error.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G59 | Gates54–58 form a coherent interacting/two-time/spin/Corbino audit with bounded claims | PASS INTEGRATED AUDIT; `gate59_interacting_spin_transport_block_20260802.json` |

## Gate 60: review and token checkpoint

The ten-gate review closes the current implementation block. ASTRA and ASTRUM
both pass 257 engine tests and 69 application tests; the evidence JSON records
parse cleanly. The token checkpoint is 4,625,513 used with no configured
remaining limit. The implementation is now a reproducible transient
EOM/Keldysh/spin workflow, but publication gates remain open: the propagated
lesser term increases the contacted residual by a factor 1.355, and the
interacting spinful Corbino problem still needs continuum/lead-size scaling and
a specialist prior-art audit. No protection verdict is forced.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G60 | Gates51–59 reviewed with regression and token checkpoints; open publication boundaries are explicit | PASS REVIEW WITH OPEN GATES; `gate60_review_20260802.json` |

## Gate 61: total self-energy continuity bookkeeping

Gate61 reruns the continuity balance with the full collision term,
\(\Sigma_{\rm emb}+\Sigma_{\rm int}\), rather than silently auditing the
embedding only. The interacting self-energy is nonzero and the total balance
is finite on ASTRA and ASTRUM. The closed vertical branch changes the total
residual from \(7.315\times10^{-2}\) to \(7.619\times10^{-2}\), a ratio of
1.042 (before the quadrature correction: \(2.649\times10^{-2}\) to
\(3.582\times10^{-2}\), ratio 1.352). This is an accounting correction and a
negative conserving-closure result, not a claim that the residual is solved.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G61 | Embedding-only and total interacting collision audits are distinguished | PASS TOTAL SELF-ENERGY ACCOUNTING; `gate61_total_self_energy_balance_20260815.json` (supersedes `..._20260802.json`) |

## Gate 62: full three-term lesser contour reconstruction

Gate62 adds the explicit finite-grid Langreth reconstruction of the vertical
lesser Dyson product. The API and symbolic layer expose separately
\(g^{\rceil}\star\Sigma^{\lceil}\cdot G^A\),
\(g^R\cdot\Sigma^{\rceil}\star G^{\lceil}\), and
\(g^{\rceil}\star\Sigma^M\star G^{\lceil}\). Both mixed-real terms are
resolved on the interacting finite-lead benchmark. The supplied Matsubara
self-energy is zero in this diagnostic, so the Matsubara term is finite but
zero; the measured antihermiticity error (\(3.806\times10^{-3}\)) remains an
explicit approximation diagnostic.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G62 | All three vertical Langreth terms are represented symbolically and numerically | PASS THREE-TERM RECONSTRUCTION; `gate62_full_contour_lesser_reconstruction_20260815.json` (supersedes `..._20260802.json`) |

## Gate 63: selectable full-contour lesser solver branch

Gate63 integrates the reconstruction into the joint real/mixed Hubbard
second-Born iteration behind an explicit option. The legacy propagated-source
branch remains available for comparison. On ASTRA and ASTRUM the full branch
converges in 60 iterations and changes the lesser kernel by
\(4.496\times10^{-3}\) relative to the legacy branch. A caller may supply a
total \(\Sigma^M\); if omitted, the zero-Matsubara approximation is retained
and reported rather than hidden.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G63 | Full three-term lesser correction is an explicit, selectable solver path | PASS OPTIONAL FULL-CONTOUR INTERFACE; `gate63_full_contour_solver_option_20260815.json` (supersedes `..._20260802.json`) |

## Gate 64: exact finite-contact charge and spin reference

Gate64 uses the exact finite many-body contacted oracle as a reference for the
same two-terminal quench. Its device charge and selected spin continuity
errors are below \(2\times10^{-11}\), and the net spin current is resolved.
The interacting EOM/Keldysh branch with the full three-term option converges
and returns finite charge/spin reservoir channels. Its charge and spin
deviations from the exact currents are retained as error metrics rather than
converted into a pass criterion.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G64 | Charge/spin observables have an exact finite-contact oracle and an explicit approximate comparison | PASS EXACT ORACLE + APPROXIMATE COMPARISON; `gate64_exact_contact_charge_spin_20260802.json` |

## Gate 65: mixed-grid and finite-lead convergence

Gate65 refines the full-contour lesser correction from 11 to 21 to 41
imaginary-time points. The successive differences decrease from
\(1.045\times10^{-4}\) to \(3.078\times10^{-5}\). Independently, the seeded
mixed source is evaluated with lead chains of sizes 2, 3, 4, and 6; the size-4
source is within \(2.6\times10^{-9}\) of size 6 on the audited window.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G65 | Mixed-grid and finite-lead controls are numerically resolved before continuum claims | PASS GRID + LEAD-SIZE CONTROLS; `gate65_mixed_grid_lead_size_convergence_20260802.json` |

## Gate 66: Corbino Hubbard closed-branch adapter

The application now exposes `solve_hubbard_kane_mele_two_time`. It requires a
caller-supplied \(G^{\rceil}\) and \(G^M\) seed, routes them through the
engine's three-term Hubbard contour branch, contracts inner/outer charge and
spin channels, and offers a persistent comparison object whose flux force is
kept separate. The compact spinful adapter converges on ASTRA and ASTRUM; the
zero persistent force in the minimal geometry is recorded rather than used as
a physical topology statement.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G66 | Corbino application can consume an explicit contour seed and return closed-branch charge/spin reservoir channels | PASS HUBBARD CORBINO ADAPTER; app `gate66_hubbard_corbino_closed_branch_20260802.json` |

## Gate 67: closed-branch persistent/reservoir comparison

Gate67 reruns the compact Corbino closed branch with a nonzero flux profile
and compares its persistent force to the inner/outer reservoir channels. The
persistent maximum is \(1.815\times10^{-1}\), while the charge channels are
about \(3.5\times10^{-2}\) and the resolved \(S_z\) channels are about
\(1.5\times10^{-5}\). These scales are kept in separate fields; no universal
current is constructed.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G67 | Persistent and reservoir responses remain distinct after the interacting adapter | PASS CLOSED-CHANNEL SEPARATION; app `gate67_closed_persistent_reservoir_20260802.json` |

## Gate 68: criteria for (not demanded) topological protection

Gate68 runs ideal, weak-Rashba, spin-independent-disorder, and large-mass
controls through the closed adapter. The spin response changes measurably,
but the compact geometry cannot evaluate bulk--edge separation and the
interacting continuity closure remains open (maximum lesser
antihermiticity diagnostic about \(5.12\times10^{-7}\)). The criteria matrix
therefore returns `NOT_CLAIMED_UNTIL_BULK_EDGE_AND_CONSERVATION_CRITERIA_CLOSE`.
This is the requested “determine whether protection exists” posture: no
protection is assumed, and no negative theorem is inferred from an unevaluable
geometry.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G68 | Robustness controls and explicit protection criteria are recorded without forcing a topology verdict | PASS CRITERIA AUDIT / VERDICT OPEN; app `gate68_closed_topology_criteria_20260802.json` |

## Gate 69: integrated closed-branch audit

Gate69 parses the five engine and three application evidence records from
Gates61–68, checks that ASTRA and ASTRUM are both recorded as passing, and
imports the public engine/application APIs. The full regressions are 259 engine
tests and 70 application tests in both environments; `git diff --check` is
clean in both repositories.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G69 | Gates61–68 form one reproducible, bounded evidence block | PASS INTEGRATED AUDIT; `gate69_integrated_closed_branch_audit_20260802.json` |

## Gate 70: review and token checkpoint

The ten-gate review closes Gates61–69 with 259 engine tests and 70
application tests on both ASTRA and ASTRUM. The block is now strong enough for
a bounded methodological/software draft, with explicit charge/spin and
persistent/reservoir separation. It is not yet ready for a strong new-physics
or topological-protection claim: the interacting Matsubara branch and
conserving continuity remain open, the compact Corbino gate has no bulk--edge
scaling, and exact-contact current discrepancies are recorded. The token
checkpoint is 5,016,546 used with no configured remaining limit.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G70 | Gates61–69 reviewed; readiness and open publication criteria are explicit | PASS REVIEW WITH OPEN GATES; `gate70_review_20260802.json` |

## Gate 71: self-consistent Matsubara interaction branch

Gate71 adds the missing symbolic and numerical Matsubara layer.  The local
Hubbard Hartree and second-Born terms are iterated with a finite-grid Dyson
equation, and the resulting (Sigma^M) and dressed (G^M) are fed into the
existing full three-term lesser contour reconstruction through an explicit
`self_consistent_matsubara=True` option.  The zero-interaction limit, the
(U^2) scaling, convergence, finite-grid KMS residuals, and contour attachment
are checked in `scripts/verify_gate71_self_consistent_matsubara.py`.  The
finite-grid KMS residual is reported rather than hidden: this gate closes the
software branch but not a continuum conserving theorem.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G71 | Self-consistent Matsubara Hartree/second-Born ΣM is symbolically exposed, numerically converged, and coupled to the full lesser contour branch | PASS IMPLEMENTED WITH FINITE-GRID KMS DIAGNOSTIC; `gate71_self_consistent_matsubara_20260802.json` |

## Gate 72: full-size Corbino closed-branch scaling

Gate72 moves the application adapter beyond the two-site geometry and runs the
interacting real/mixed closed branch on three honeycomb annuli (48, 42, and 72
sites).  It records inner/bulk/outer spin shells, inner/outer reservoir
charge/spin channels, the persistent AB force on a separate field, and the
lesser antihermiticity diagnostic.  The observed edge/bulk ratios are
20.28, 7.80, and 1.23: a resolved but non-monotone finite-size crossover,
not a forced topological trend.  To keep the matrix-size scan finite, the
vertical Matsubara seed is explicitly zero here; Gate71 is the self-consistent
Matsubara control.  Evidence is in the application repository at
`docs/evidence/gate72_hubbard_corbino_size_scaling_20260802.json`.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G72 | Increasing Corbino sizes resolve interacting spin shells and reservoir channels; finite-size edge/bulk crossover is measured without claiming protection | PASS FINITE-SIZE CROSSOVER WITH EXPLICIT VERTICAL BOUNDARY; app `gate72_hubbard_corbino_size_scaling_20260802.json` |

## Gate 73: arbitrary time-dependent matrix embeddings

Gate73 adds `solve_time_dependent_matrix_embedding`, a generic finite-grid
Kadanoff--Baym/Dyson interface for matrix-valued (Sigma^r(t,t')) and
(Sigma^<(t,t')).  The supplied retarded kernel is checked on the time axes
for causality, the advanced branch is adjointed explicitly, and lesser/greater
Keldysh residuals are returned with the converged Green functions.  A driven,
noncommuting two-level device and a phase-bearing time-dependent matrix bath
pass the ASTRA/ASTRUM checks in
`docs/evidence/gate73_time_dependent_matrix_embedding_20260802.json`.

| Gate | Claim established | Status and evidence |
|---|---|---|
| G73 | Arbitrary finite-grid nonstationary matrix embeddings are accepted and solved with causal/Keldysh diagnostics | PASS MATRIX EMBEDDING INTERFACE; `gate73_time_dependent_matrix_embedding_20260802.json` |

## Gate 74: specialist novelty audit

Gate74 records a specialist primary-source novelty audit in
`docs/evidence/gate74_specialist_novelty_audit_20260802.json` and the search
log. It rejects broad EOM/KBE/Keldysh, persistent-spin-ring, interacting-QSH
invariant, and robustness-as-protection novelty claims. The residual
contribution is only a narrow, unconfirmed ASTRA/ASTRUM benchmark workflow;
specialist database searches and interacting continuum closure remain open.

## Gate 75: publication claim matrix

Gate75 combines the current engine/app evidence block and the complete
regressions (264 engine tests and 70 application tests in both ASTRA and
ASTRUM). The machine-readable record is
`docs/evidence/gate75_publication_claim_matrix_20260802.json` and is generated
by `scripts/verify_gate75_publication_claim_matrix.py`. It permits only a
finite-grid two-time software/benchmark claim with explicit limitations,
marks the self-consistent Matsubara branch as an implementation result, and
keeps broad method novelty, a conserving interacting-continuum theorem,
topological protection, and strong new-physics novelty as `NOT_READY` or
`UNCONFIRMED`.

## Gate 76: exact interaction accuracy ledger

Gate76 compares the finite-contact exact Fock-space oracle with the
self-consistent Matsubara plus real-time second-Born branch at
(U=0,0.1,0.3,0.5,0.8). Both runtimes pass the oracle, convergence, and finite
charge/spin-channel checks. The result is intentionally negative for closure:
weak (U) converges, stronger (U) exposes a Matsubara convergence boundary,
finite charge/spin discrepancies, and a growing lesser anti-Hermiticity
residual. Evidence is in
`docs/evidence/gate76_exact_interaction_accuracy_ledger_20260802.json`; the
verifier is `scripts/verify_gate76_exact_interaction_accuracy_ledger.py`.

## Gate 77: finite-ramp transient control

The application-side Gate77 runs abrupt-but-resolved and slow AB flux ramps
with finite contacts. ASTRA/ASTRUM both close the local/global quadratic
continuity identities, recover zero initial terminal current, resolve delayed
sampled onset, and keep persistent flux force separate from reservoir current.
The ratios are 3.81 and 3.45 for the two ramp durations. This is a dynamic
memory control, not an interacting continuum or topological-protection claim;
evidence is in the application repository at
`docs/evidence/gate77_transient_flux_ramp_memory_20260802.json`.

## Gate 78: spinful flux-ramp controls

The application Gate78 repeats the finite-lead ramp for the Kane–Mele parent
and a trivial mass control. Both runtimes close charge continuity and the local
(s_z) balance with its explicit torque term, resolve spin-shell channels, and
keep persistent flux force separate from reservoir current. This is a finite
control—not a (Z_2) invariant or protection claim. Evidence is in
`docs/evidence/gate78_spinful_flux_ramp_controls_20260802.json` in the
application repository.

## Gate 79: reproducible package manifest

Gate79 binds the Gate71–78 evidence, source verifiers, protocol/audit
documents, and current regression snapshot into a SHA-256 manifest. It passes
in ASTRA and ASTRUM and retains the open physics gates: interacting continuum
continuity, specialist novelty confirmation, and the protection decision. The
record is `docs/evidence/gate79_reproducible_package_20260802.json`, generated
by `scripts/verify_gate79_reproducible_package.py`.

## Gate 80: ten-gate review

The review after Gates71–79 passes on ASTRA and ASTRUM: 264 engine tests and
70 application tests pass in each runtime, all nine gate records are present,
and the SHA-256 manifest is valid. The bounded software/benchmark result is
ready for a draft with explicit limitations. Broad method novelty, a
conserving interacting-continuum theorem, strong new physics, and topological
protection remain `NOT_READY`; the integrated novelty candidate remains
`UNCONFIRMED`. The review record is
`docs/evidence/gate80_review_20260802.json`; the next block must target
same-self-energy continuum continuity and controlled Corbino extrapolation.

## Gate 81: same-self-energy continuity decomposition

The engine now exposes `TwoTimeContinuityComponents` and
`two_time_kbe_continuity_components`, which evaluate one total balance and
separate embedding/interaction collision channels from the same kernels used
by the Green-function branch. ASTRA/ASTRUM give an additivity error of order
(10^{-17}) and a direct-total mismatch of zero, while the raw physical
residual remains (1.96\times10^{-2}) and the vertical-source diagnostic is
non-Hermitian at finite grid. Evidence is in
`docs/evidence/gate81_same_self_energy_continuity_20260803.json`; this is an
audit/API upgrade, not a conserving-continuum theorem.

## Gate 82: lead-size extrapolation

The application scans finite lead lengths (L=3,4,5,6) for the spinful
Kane–Mele Corbino ramp. ASTRA/ASTRUM keep the sampled time window before the
first recurrence, close charge and spin balances, and resolve a decreasing
inner-spin tail difference (0.00662 (ightarrow) 0.00487; ratio 0.735). The
result is a controlled finite-size extrapolation diagnostic, not an assumed
continuum limit or protection claim. Evidence is in the application repository
at `docs/evidence/gate82_lead_size_extrapolation_20260803.json`.

## Gate 83: regression refresh

After the Gate81 API and Gate82 verifier, complete suites pass with 265 engine
tests and 70 application tests in each ASTRA/ASTRUM runtime. The additional
engine test covers continuity-component additivity. Evidence is in
`docs/evidence/gate83_regression_refresh_20260803.json`; this refresh validates
software integrity only and leaves the physics closure gates open.

## Gate 84: refreshed package manifest

Gate84 refreshes the reproducibility manifest after Gates81–83, including the
new continuity module/test, lead-size verifier, regression baseline, and all
historical records. ASTRA/ASTRUM both pass; the regression snapshot is 265
engine and 70 application tests per runtime. The manifest is
`docs/evidence/gate84_package_refresh_20260803.json`.

## Gate 85: symbolic continuity and spin projection

The symbolic layer now exposes `kadanoff_baym_collision_integral_symbolic` and
`kadanoff_baym_continuity_symbolic`. The former retains all four ordered
two-time collision convolutions and their equal-time limit; the latter writes
`rho=-iG^<`, the coherent commutator, the collision term, and an optional
vertical-branch source as one continuity identity. Formal `Tr(Q ...)` and
`Tr(S_a ...)` projections make charge, spin, and coherent spin torque
auditable without assuming that the finite-grid closure is conserving.
ASTRA/ASTRUM pass all eight symbolic checks; evidence is in
`docs/evidence/gate85_symbolic_continuity_20260803.json` and the regression
test is included in the 265-test engine baseline.

## Gate 86: regression refresh after the symbolic upgrade

The complete suites now pass with 267 engine tests and 70 application tests in
both ASTRA and ASTRUM. The two additional engine tests cover the ordered
collision helper and the charge/spin continuity projection. Evidence is in
`docs/evidence/gate86_regression_refresh_20260803.json`; this gate validates
software integrity only and leaves the interacting-continuum and protection
questions open.

## Gate 87: noncommuting spin torque

An exact finite-contact two-time oracle with a spin-noncommuting device
Hamiltonian resolves coherent (S_x) and (S_z) torque while the charge
projection remains zero. The source-corrected continuity residual decreases
from (1.57\times10^{-3}) to (7.87\times10^{-4}) over the 5/7/9 time-point
refinement, with the spectral identity at machine precision. Evidence is in
`docs/evidence/gate87_noncommuting_spin_torque_20260803.json`.

## Gate 88: Rashba spin-torque control

The finite Kane--Mele Corbino adapter is run at Rashba (0) and (0.22). Both
charge and (s_z) balances close at (10^{-17}); the torque changes from
numerical zero to (6.34\times10^{-3}), and the spin-shell response changes
substantially. ASTRA/ASTRUM agree. This is an explicit torque-bookkeeping
control, not a protection claim. Evidence is in the application repository at
`docs/evidence/gate88_rashba_spin_torque_20260803.json`.

## Gate 89: final-round package refresh

The reproducibility manifest binds Gates80--88, the symbolic API and tests,
the exact noncommuting-spin oracle, the Rashba Corbino control, and the 267/70
regression snapshot. All records pass in ASTRA and ASTRUM; the protection
boundary remains `NOT_READY`. Evidence is
`docs/evidence/gate89_package_refresh_20260803.json`.

## Gate 90: final ten-gate review and pause

The review after Gates81--89 passes on ASTRA and ASTRUM. The current complete
baseline is 267 engine tests and 70 application tests in each runtime. The
bounded software/benchmark result is `READY_FOR_DRAFT_WITH_EXPLICIT_LIMITATIONS`;
broad method novelty is rejected by prior art, while the narrow integrated
novelty candidate remains unconfirmed. Conserving interacting continuum,
strong new physics, and topological protection remain `NOT_READY`. The token
checkpoint is 6,338,554 used; the prescribed next action is to pause
implementation and analyze the project state. Evidence is in
`docs/evidence/gate90_final_review_20260803.json`.

## Capability inventory at G1

Already implemented and tested:

- symbolic second quantization, commutators, normal ordering, finite EOM
  closure, and Hartree/Hubbard-I style reductions;
- symbolic contour objects, Langreth rules, stationary Dyson equations, FDT,
  Meir–Wingreen and wide-band current expressions;
- exact finite quadratic propagation and explicit two-time
  \(G^{r,a,<,>}(t,t')\);
- exact finite-interacting mixed real/vertical Green branches
  \(G^{\lceil},G^{\rceil}\) for source benchmarking;
- symbolic and self-consistent numerical Hubbard second-Born two-time
  self-energies with explicit spectral/causality diagnostics;
- symbolic and self-consistent Matsubara Hubbard Hartree/second-Born
  self-energy with KMS residuals and explicit contour-branch attachment;
- full-size Corbino closed-branch scaling adapter with inner/bulk/outer spin
  shells and persistent/reservoir separation;
- instantaneous Hubbard Hartree self-energy with finite-grid delta
  collocation and a static refinement control;
- explicit attachment of a microscopic vertical Keldysh source to the
  continuity balance, with raw/source-corrected residuals kept distinct;
- mixed Hubbard second-Born \(\Sigma^{\rceil}\) source kernel with exposed
  finite-lead mixed Green branches and (U^2) scaling diagnostics;
- nested real/vertical grid convergence for the seeded mixed source product;
- finite-lead-size convergence of the seeded mixed source before recurrences;
- causal time-window/memory sensitivity of the seeded mixed source;
- a negative joint-closure diagnostic rejecting post-hoc mixing of exact
  vertical and approximate real-time interacting branches;
- stationary continuum self-energy/Green-function transforms;
- partition-free wide-band matrix step quenches with \(\rho(t)\), lead
  currents, continuity, and selected two-time kernels;
- charge and stationary spin-resolved transport observables;
- finite-quadratic bond spin currents with explicit Rashba torque balance;
- a unitary wide-band Fisher–Lee scattering matrix and TRS eligibility test.

The original G1 inventory identified the following as upgrade targets. Gates
12–19 now cover bounded versions of the first four items:

- automated higher-order interacting EOM closure beyond finite bases and
  controlled truncations;
- arbitrary-time matrix lead self-energies and fully smooth transient memory
  kernels beyond the analytic Lorentzian/scalar-gauge oracle;
- generic transient lead spin currents and reservoir spin injection in the same
  conserving API as charge currents (the device/bond spin layer and two-time
  observable contraction are implemented);
- a production-grade conserving contour solver beyond the finite-grid KBE/SCBA
  research layer;
- a quantized open-system scattering invariant and a novelty audit tied to the
  application observable (the unitary scattering object is implemented, but no
  quantized invariant is claimed).

The current public transport result is therefore a quadratic/Keldysh transient
benchmark plus a bounded interacting two-time research layer. The audit does
not claim topological protection: that remains a hypothesis to test against
bulk, edge, disorder, Rashba, flux-ramp, and reservoir-coupling controls. The
candidate publication contribution is the combined, reproducible workflow;
the atomic same-Hubbard-U comparison is closed, while extension to lead-coupled
interacting continuity, production convergence, and a specialist literature
search remain open.

## Reproduction

From the engine repository:

    python scripts\verify_gate01_capability_inventory.py
    python scripts\verify_gate12_kadanoff_baym_symbolic.py
    python scripts\verify_gate13_kadanoff_baym_numeric.py
    python scripts\verify_gate15_two_time_charge_spin_currents.py
    python scripts\verify_gate16_analytic_reservoir_memory.py
    python scripts\verify_gate17_eom_hubbard_i_vs_scba.py
    python scripts\verify_gate32_hubbard_second_born_two_time.py
    python scripts\verify_gate33_hartree_vertical_source_layer.py
    python scripts\verify_gate34_vertical_source_continuity_attachment.py
    python scripts\verify_gate35_hubbard_second_born_mixed_source.py
    python scripts\verify_gate36_mixed_source_grid_convergence.py
    python scripts\verify_gate37_mixed_source_lead_size.py
    python scripts\verify_gate38_mixed_source_memory_window.py
    python scripts\verify_gate39_integrated_evidence_audit.py
    python scripts\verify_gate41_exact_interacting_mixed_branch.py
    python scripts\verify_gate42_interacting_source_pairing_diagnostic.py
    python scripts\verify_gate43_exact_source_error_budget.py
    python scripts\verify_gate51_mixed_volterra_stepper.py
    python scripts\verify_gate52_joint_hubbard_contour_iteration.py
    python scripts\verify_gate53_joint_contour_source_attachment.py
    python scripts\verify_gate54_lesser_vertical_closure.py
    python scripts\verify_gate55_charge_spin_meir_wingreen.py
    python scripts\verify_gate59_engine_interacting_spin_block.py
    python scripts\verify_gate71_self_consistent_matsubara.py
    python scripts\verify_gate73_time_dependent_matrix_embedding.py
    python scripts\verify_gate74_specialist_novelty_audit.py
    python scripts\verify_gate21_same_hubbard_u_exact.py
    python scripts\verify_gate20_publicability_audit.py
    python -m pytest

From the application repository, Gate72 is reproduced with:

    python scripts\verify_gate72_hubbard_corbino_size_scaling.py

The application repository must run its own gate verifier against the same
engine version. ASTRA checks the symbolic identities and guard conditions;
ASTRUM executes the larger parameter scans and timing-sensitive benchmarks.
