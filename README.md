# QuantumTransportEOM

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21265399.svg)](https://doi.org/10.5281/zenodo.21265399)

**Symbolic equation-of-motion Green functions and fast numerical quantum transport in one package.**

`quantum_transport` couples a SymPy-based second-quantization layer (commutators, EOM closure, mean-field truncations, Keldysh contour algebra) to a NumPy transport engine (Landauer, Meir–Wingreen, spin-resolved observables) and an exact finite-system real-time layer for \(G^{r,a,<,>}(t,t')\), with batched linear algebra, multi-core frequency sweeps, and optional CUDA GPU execution via CuPy.

Version `0.4.1` — Python ≥ 3.10, MIT license.

---

The transient EOM/Keldysh/spin work is tracked by the gate contract in
docs/GATE_PROTOCOL_TRANSIENT_KELDYSH_SPIN.md. Gates 1–32 have reproducible
ASTRA/ASTRUM evidence. The public API now includes symbolic Langreth and
Kadanoff–Baym equations, finite-grid two-time SCBA memory, analytic
finite-band Lorentzian reservoir kernels with smooth scalar gauge phases,
charge/spin two-time Meir–Wingreen currents, the generic EOM retarded solver,
finite-quadratic spin-bond current with Rashba torque accounting, and a
unitary wide-band Fisher–Lee scattering matrix, plus a bounded symbolic and
self-consistent Hubbard second-Born two-time layer. The publication audit
remains open: the exact finite-U density gap and raw continuity source are
quantified, while full interacting continuity closure, production convergence,
and a specialist novelty search are still required.

## Install

```bash
# from this folder
python -m pip install -e .            # core (numpy, sympy, matplotlib, threadpoolctl)
python -m pip install -e ".[test]"    # + pytest
python -m pip install -e ".[gpu]"     # + cupy-cuda12x (CUDA hosts only)
```

## Sixty-second tour

### 1. Any Hamiltonian, symbolically

```python
import sympy as sp
from quantum_transport import CustomModel, n

eps, U = sp.symbols("epsilon U", real=True)
dot = CustomModel(eps * (n("up") + n("down")) + U * n("up") * n("down"))

dot.model.eom()                       # exact EOM: [c_s, H] = M c + residual (cubic strings)
dot.model.eom(auto_expand_steps=1)    # atomic-limit hierarchy closes at 4x4
dot.model.eom(truncation="hartree")   # mean field: diag(eps + U <n_down>, eps + U <n_up>)

w, e = sp.symbols("omega eta", positive=True)
dot.gf("c_up").retarded(omega=w, eta=e, method="hartree")
# 1/(omega + I*eta - epsilon - U*n_down_avg)
```

Operators are built with `f`/`fd` (fermions) and `b`/`bd` (bosons); statistics, mode content, and the seed operator basis are detected automatically. Non-Hermitian Hamiltonians (usually a missing conjugate hopping) trigger a warning. In Jupyter, models, operators, and EOM systems render as LaTeX.

### 2. Open it into a device

Quadratic (non-interacting) models convert straight into a numeric two-terminal device — leads can be scalars, site-resolved dicts, coupling matrices, or `LeadSelfEnergy` objects (wide band, semi-infinite chain, ferromagnetic, sampled…):

```python
import numpy as np
from quantum_transport import CustomModel, f, fd, n

t = 1.0
chain = CustomModel(sum(t * (fd(i) * f(i + 1) + fd(i + 1) * f(i)) for i in range(2)))
view = chain.open({"0": 0.4}, {"2": 0.4})          # contacts on the chain ends only

grid = np.linspace(-3, 3, 801)
T = view.transmission_values(grid)                  # Landauer transmission curve
I = view.landauer_current(grid, mu_left=0.5, mu_right=-0.5)
```

### 3. Make it fast

Every `*_values` sweep and the interacting self-consistency loops accept `workers=` (CPU threads over frequency blocks) and `backend=` (`"numpy"` / `"cupy"` / `"auto"`):

```python
T = view.transmission_values(grid, workers=8)               # multi-core CPU
T = view.transmission_values(grid, backend="cupy")          # CUDA GPU (with .[gpu])
T = view.transmission_values(grid, backend="cupy", precision="single")  # GPU fast path

from quantum_transport import AndersonImpurity
imp = AndersonImpurity(eps=-0.5, U=2.0).open(0.3, 0.3, mu_left=0.25, mu_right=-0.25)
occ = imp.self_consistent_occupations(grid, eta=1e-3, workers=4)   # Hubbard-I SCF
```

Under the hood the per-frequency Python loop is replaced by stacked LAPACK/BLAS (or cuSOLVER on GPU) calls, ω-independent (wide-band) lead self-energies are evaluated once and broadcast, the grid is processed in memory-capped blocks, and small-matrix factorizations are pinned to one BLAS thread (multithreaded OpenBLAS LAPACK is up to 30–500× slower on small matrices on many-core machines). Measured results, 400-frequency transmission sweeps:

- 24-thread Windows laptop, 120 orbitals: ~30 s scalar loop → 2.4 s batched → **<1 s with `workers=8`**;
- Ryzen 9950X3D + RTX 3080 (Linux), 600 orbitals: 8.5 s batched CPU → **1.1 s on GPU with `precision="single"`** (~1e-5 accuracy; GeForce float64 is capped at 1/64 rate, so keep `precision="double"` on CPU or datacenter GPUs).

Whether `workers` beats the serial batched path depends on the machine's BLAS — benchmark both once (`examples/demo_parallel_gpu.py`); results are bit-identical either way and equivalence is enforced by the test suite.

### 4. Expand a labelled EOM hierarchy

For a new Hamiltonian, pass either one expression or labelled contributions.
The hierarchy detects fermionic, bosonic, and mixed ladder algebras, records
which part of the Hamiltonian generated each EOM term, expands the basis by a
requested depth, and refuses to call an open hierarchy exact unless the caller
explicitly requests a residual drop:

```python
import numpy as np
from quantum_transport import BosonicSCBAConfig, ElectronBosonSCBAConfig, SelfConsistentClosure, build_eom_hierarchy, f, fd, n

hierarchy = build_eom_hierarchy(
    {
        "one_body": eps * fd("up") * f("up"),
        "interaction": U * n("up") * n("down"),
    },
    max_depth=2,
    max_operators=128,
)

print(hierarchy.is_closed)
print(hierarchy.unresolved_operators)
G_r = hierarchy.retarded_green(omega, eta, approximate=True)
G_lesser = hierarchy.stationary_lesser_green(omega, eta, Sigma_lesser, approximate=True)

# Or supply a model-specific closure instead of dropping the residual:
# residual_closure = {n("down") * f("up"): n_down_avg * f("up")}
# G_r = hierarchy.retarded_green(omega, eta, residual_closure=residual_closure)

# Symbolic contour/Langreth projection, including r/a/</>, rceil/lceil, and M:
contour = hierarchy.contour_equations()
retarded_equations = contour.component("r")

# Direct numerical two-time propagation with statistics-aware source:
# two_time = contour.propagate_two_time(
#     time_grid, initial_density, parameters={eps: 0.2, U: 1.0}
# )
# Supplying Sigma^r and Sigma^< automatically selects the KBE/Dyson branch.

# Or generate Sigma self-consistently with time-domain electron-boson SCBA:
# scba = ElectronBosonSCBAConfig(
#     coupling=np.array([[0.05]]), boson_frequency=0.8,
#     max_iterations=30, dyson_iterations=80,
# )
# result = contour.propagate_two_time(
#     time_grid, initial_density, parameters={eps: 0.2},
#     electron_boson_scba=scba,
# )

# For a purely bosonic hierarchy, close all contour branches in one loop:
# boson_scba = BosonicSCBAConfig(
#     coupling=np.array([[0.03]]), boson_frequency=0.8,
#     boson_temperature=0.5,
#     cubic_vertex=np.array([[[0.01]]]),
#     quartic_vertex=np.array([[[[0.005]]]]),
# )
# result = contour.propagate_two_time(
#     time_grid, [[2.0]], imaginary_time=np.linspace(0, 2.0, 32),
#     parameters={omega_b: 0.8}, bosonic_scba=boson_scba,
# )
# result.self_energy_mixed       # Sigma^rceil(t,tau)
# result.self_energy_lmixed      # Sigma^lceil(tau,t)
# result.self_energy_matsubara   # Sigma^M(tau,tau')

# A self-consistent closure supplies its own observable update callback:
# closure = SelfConsistentClosure(
#     rules={residual_operator: n_down_avg * f("up")},
#     initial_values={n_down_avg: 0.5},
#     update=lambda values, G: {n_down_avg: occupation_from_lesser(G)},
# )
# result = hierarchy.solve_self_consistent(omega, eta, closure)
```

`max_depth` is the number of residual-expansion rounds, not the operator
degree. This keeps model-specific coefficients (such as Hely's spin-flip
parameters) outside the engine. The contour adapter preserves the vertical
Matsubara branch and leaves initial-correlation kernels explicit; numerical
propagation delegates automatically to the existing finite-grid/Kadanoff–Baym
solvers. See
`examples/demo_eom_hierarchy.py`.

## Capability map

| Layer | Highlights |
|---|---|
| Operator algebra | `commutator`, `anticommutator`, fermionic/bosonic wrappers (`SQObj`, `BQObj`), Wick + exact CAR normal ordering (`normal_order_fermionic`), `dagger_expression` |
| Symbolic models | Predefined (`AndersonImpurity`, `FermionicSingleLevel`, dimers, Holstein, Jaynes–Cummings…) and `CustomModel` for arbitrary Hamiltonians; EOM closure analysis, basis auto-expansion, Hartree / Hubbard-I truncations |
| Green functions | Retarded/advanced/lesser/greater, spectral functions, symbolic *and* numeric, equilibrium distributions |
| Keldysh | Contour objects, Langreth rules, Dyson equations, symbolic Meir–Wingreen, stationary self-energies, exact finite-system dynamics, and band-limited stationary continuum \(\Sigma^{r,a,<,>}(t,t')\), \(G^{r,a,<,>}(t,t')\) |
| Real time | Unitary midpoint propagation for finite embeddings; exact wide-band partition-free matrix step quenches with \(\rho(t)\), lead currents, and \(G^{r,a,<,>}(t,t')\); analytic resonant-level oracles |
| Full contour SCBA | Pure-boson Einstein-mode SCBA with simultaneous real-time, mixed \(rceil/lceil\), and periodic Matsubara Dyson updates; explicit KMS and mixed-adjoint diagnostics |
| Finite-lead spectral matching | `finite_lead_retarded_self_energy` and `finite_lead_spectral_density` evaluate the broadened microscopic embedding; `match_wide_band_broadening_from_finite_lead` returns an explicit \(f(1-f)\)-windowed constant-WBL calibration, while `LeadSelfEnergy.sampled` + `stationary_self_energy_two_time` retain the \(\Sigma(t,t')\) memory |
| Devices | `MatrixDevice`, `SpinfulSingleSite`, `SpinfulDimer`, `RashbaRingDevice`, Aharonov–Bohm rings; leads: wide-band, polarized, ferromagnetic, semi-infinite chain, sampled |
| Transport | Transmission, conductance (T = 0 and finite T), Landauer and Meir–Wingreen currents, spin-resolved everything (axis-resolved projectors), persistent currents, Drude weight |
| Interacting open dots | `AndersonImpurity.open(...)`: Hubbard-I / Hartree self-energies, self-consistent occupations, finite-bias currents |
| Acceleration | `numerics` module: array-backend resolution (`get_backend`), batched Green-function kernels, `blocked_over_grid`, `parallel_map`, BLAS thread management |

## Physics guarantees

`tests/test_physics_validation.py` pins the package to exact analytic results, independently of implementation details:

- resonant-level transmission is the exact Lorentzian `Γ_L Γ_R / ((ω−ε)² + (Γ/2)²)`, with `T(ε) = 1` for symmetric coupling;
- the spectral sum rule `∫ A(ω) dω/2π = 1` per orbital;
- the fluctuation–dissipation relation `G^< = −f(ω)(G^r − G^a)` at equilibrium and zero current at zero bias;
- steady-state current conservation `I_L = −I_R`, and Meir–Wingreen ≡ Landauer for non-interacting devices;
- single-channel unitarity `T ≤ 1`, with unit resonances in the weak-coupling limit.
- exact single-level two-time Green functions, the Keldysh spectral identity,
  unitary conservation, temporal-gauge covariance, and finite-embedding
  continuity.
- analytic wide-band resonant-level contact and partition-free bias quenches
  obey continuity and recover the zero-temperature Landauer steady state.
- stationary continuum two-time transforms preserve adjoint/anti-Hermiticity
  relations and the Keldysh spectral identity, reproduce the equal-time
  frequency integral, and approach the exact wide-band resonant-level kernel.
- the partition-free matrix quench reproduces the scalar analytic oracle,
  starts from contacted equilibrium with zero lead current, obeys charge
  continuity locally and globally, is covariant under fixed unitary basis
  changes, and relaxes to the final stationary NEGF solution. Per-lead orbital
  source terms expose site-resolved continuity without altering terminal
  currents; compact \(\Gamma=RR^\dagger\) factors accelerate low-rank contacts.

Run everything:

```bash
    python -m pytest            # 237 engine tests (the application has 69)
python examples/demo_custom_model.py
python examples/demo_parallel_gpu.py
```

## Examples

| Script | Shows |
|---|---|
| `examples/demo_custom_model.py` | Custom Hamiltonians end to end: EOM, Hartree, open transport, mixed fermion–boson |
| `examples/demo_parallel_gpu.py` | Batched vs loop benchmarks, `workers=`, GPU backend detection |
| `examples/demo_high_level_api.py` | QuTiP-style high-level API |
| `examples/demo_open_anderson.py` | Interacting dot with leads: Hubbard-I, SCF occupations, currents |
| `examples/keldysh_guide_01…05` | Progressive Keldysh tutorial (contour → Langreth → quantum dot → stationary → two-terminal) |
| `examples/demo_transient_keldysh.py` | Exact \(G^{r,a,<,>}(t,t')\), density dynamics, and finite-lead interface currents |
| `examples/demo_resonant_level_transient.py` | Analytic partitioned wide-band level occupation, lead currents, continuity, and steady-state limit |
| `examples/demo_partition_free_resonant_level.py` | Initially contacted equilibrium, sudden lead-bias quench, zero initial current, and Landauer relaxation |
| `examples/demo_continuum_two_time.py` | Matrix-valued stationary continuum self-energies and Green functions on an explicit \((t,t')\) grid |
| `examples/demo_partition_free_matrix_quench.py` | Initially contacted matrix device under simultaneous lead-bias and Hamiltonian step quenches; density, currents, continuity, and full two-time Green functions |
| `examples/demo_ab_ring.py`, `demo_models.py`, … | Rings, spin transport, symbolic model gallery |

PowerShell automation (runs demos, tests, and optionally builds the manual):

```powershell
.\run_all.ps1
.\run_all.ps1 -InstallPytest -BuildManual
```

## Documentation

- LaTeX manual: `docs/user_manual.tex` (PDF: `docs/user_manual.pdf`, build with `.\run_all.ps1 -BuildManual`).
- Every public function has a docstring; the high-level API is discoverable from `quantum_transport.__all__`.
- ASTRA/ASTRUM evidence for the continuum two-time update:
  `docs/evidence/continuum_two_time_update_20260802.json` and
  `docs/evidence/partition_free_matrix_transient_20260802.json`.
- ASTRA/ASTRUM evidence for the transient interacting/spin block:
  `docs/evidence/gate12_kadanoff_baym_symbolic_20260802.json`,
  `gate13_kadanoff_baym_numeric_scba_20260802.json`,
  `gate15_two_time_charge_spin_currents_20260802.json`,
  `gate16_analytic_reservoir_memory_20260802.json`,
  `gate17_eom_hubbard_i_vs_scba_20260802.json`, and
  `gate21_same_hubbard_u_exact_20260802.json`,
  `gate22_continuity_diagnostics_20260802.json`,
  `gate20_publicability_audit_20260802.json`, and
  `gate24_publicability_reaudit_20260802.json`.

  Gate25 adds the mixed real--imaginary Keldysh initial-correlation diagnostic;
  its evidence is `docs/evidence/gate25_initial_correlation_branch_20260802.json`.
  Gate26 closes the same source microscopically in a finite contacted benchmark;
  its evidence is `docs/evidence/gate26_finite_lead_initial_closure_20260802.json`.
  Gate27 shows finite spinful-lead convergence before recurrences, including
  charge and spin currents; its evidence is
  `docs/evidence/gate27_lead_size_recurrence_20260802.json`.
  Gate28 adds a stable spectral KMS evaluation for the finite-star/WBL
  comparison, with the UV error recorded explicitly; its evidence is
  `docs/evidence/gate28_continuum_wbl_initial_source_20260802.json`.
  Gate29 quantifies the residual-inferred Corbino charge/spin source at two
  energy resolutions; its evidence is
  `../physics/xene-ring-transport/docs/evidence/gate29_corbino_mixed_source_spin_diagnostic_20260802.json`.
  Gate31 adds an exact finite-U many-body contacted oracle and a lead-coupled
  Hubbard-I/EOM comparison; its evidence is
  `docs/evidence/gate31_exact_interacting_lead_coupled_20260802.json`.
  Gate32 adds the symbolic and self-consistent numerical Hubbard second-Born
  two-time self-energy, with exact finite-U comparison and explicit open
  Hartree/vertical-branch source diagnostics; its evidence is
  `docs/evidence/gate32_hubbard_second_born_two_time_20260802.json`.
  Gate33 makes the instantaneous Hartree term explicit and verifies its
  finite-grid refinement; the interacting vertical-branch source remains an
  open boundary. Its evidence is
  `docs/evidence/gate33_hartree_collocation_vertical_source_20260802.json`.
  Gate34 attaches the microscopic finite-lead vertical source to continuity
  diagnostics while preserving raw and source-corrected residuals separately;
  its evidence is
  `docs/evidence/gate34_vertical_source_continuity_attachment_20260802.json`.
  Gate35 adds the mixed Hubbard second-Born source product and exposes the
  finite-lead mixed Green branches, with an explicit (U^2) scaling check; its
  evidence is
  `docs/evidence/gate35_hubbard_second_born_mixed_source_20260802.json`.
  Gate36 verifies nested real/vertical grid convergence of the seeded mixed
  source product; its evidence is
  `docs/evidence/gate36_mixed_source_grid_convergence_20260802.json`.
  Gate37 verifies finite-lead-size convergence of the seeded mixed source
  before recurrences; its evidence is
  `docs/evidence/gate37_mixed_source_lead_size_20260802.json`.
  Gate38 checks causal memory-window extension of the seeded mixed source; its
  evidence is
  `docs/evidence/gate38_mixed_source_memory_window_20260802.json`.
  Gate39 audits the Gates31–38 evidence schemas, ASTRA/ASTRUM verdicts and
  claim boundaries; its evidence is
  `docs/evidence/gate39_integrated_evidence_audit_20260802.json`.
  Gate41 adds exact finite-U mixed Keldysh branches and a validated interacting
  second-Born source seed; its evidence is
  `docs/evidence/gate41_exact_interacting_mixed_keldysh_branch_20260802.json`.
  Gate42 records the negative joint-closure diagnostic: exact mixed and
  approximate real-time branches cannot be paired post hoc; its evidence is
  `docs/evidence/gate42_interacting_source_pairing_diagnostic_20260802.json`.
  Gate43 quantifies the exact finite-U charge/spin source error budget for
  second-Born; its evidence is
  `docs/evidence/gate43_exact_interacting_source_error_budget_20260802.json`.
  Gate44 scans the exact source error from weak to intermediate interaction,
  showing the onset of charge/spin mismatch without claiming a controlled
  perturbative window; its evidence is
  `docs/evidence/gate44_weak_u_source_range_20260802.json`.
  Gate45 exposes the mixed real/vertical Kadanoff--Baym differential equations
  symbolically, with explicit causal limits and the imaginary-branch measure;
  its evidence is `docs/evidence/gate45_mixed_kbe_symbolic_20260802.json`.
  Gate46 exposes stable equilibrium Matsubara (G^M) and embedding (Σ^M)
  branches on the finite vertical grid; its evidence is
  `docs/evidence/gate46_matsubara_branch_20260802.json`.
  Gate47 adds numerical residuals for both mixed KBE orientations using causal
  and vertical quadrature; its evidence is
  `docs/evidence/gate47_mixed_kbe_residual_20260802.json`.
  Gate48 exposes charge and spin projections of the microscopic vertical
  source; its evidence is
  `docs/evidence/gate48_charge_spin_source_projection_20260802.json`.
  Gate49 audits Gates44–48 as a coherent ASTRA/ASTRUM block; its evidence is
  `docs/evidence/gate49_interacting_two_time_block_audit_20260802.json`.
  Gate51 adds the causal mixed-branch Volterra propagator; its evidence is
  `docs/evidence/gate51_mixed_volterra_stepper_20260802.json`.
  Gate52 couples that propagator to the self-consistent Hubbard second-Born
  real/mixed iteration; its evidence is
  `docs/evidence/gate52_joint_contour_iteration_20260802.json`. Gate53
  attaches the microscopic mixed source and records the negative lesser
  closure diagnostic in
  `docs/evidence/gate53_joint_contour_source_attachment_20260802.json`.
  Gate54 propagates that mixed source into the real lesser equation; its
  controlled negative closure result is recorded in
  `docs/evidence/gate54_lesser_vertical_initial_correlation_20260802.json`.
  Gate55 exposes shared charge/spin Meir–Wingreen channels, Gate56 maps them
  to the Corbino reservoirs, Gate57 compares them with persistent flux
  response, and Gate58 runs Rashba/disorder controls. Gate59 audits the block;
  evidence is stored in `docs/evidence/gate55_charge_spin_meir_wingreen_20260802.json`,
  `docs/evidence/gate56_corbino_charge_spin_two_time_20260802.json`,
  `docs/evidence/gate57_persistent_reservoir_comparison_20260802.json`,
  `docs/evidence/gate58_spin_robustness_controls_20260802.json`, and
  `docs/evidence/gate59_interacting_spin_transport_block_20260802.json`.
  Gate60 records the ten-gate review, token checkpoint, and remaining
  publication gates in `docs/evidence/gate60_review_20260802.json`.
  Gate61 audits continuity with embedding plus interacting self-energies;
  Gate62 exposes the three vertical Langreth terms of the lesser contour
  reconstruction; and Gate63 integrates that reconstruction as a selectable
  solver branch. Their evidence is
  `docs/evidence/gate61_total_self_energy_balance_20260802.json`,
  `docs/evidence/gate62_full_contour_lesser_reconstruction_20260802.json`, and
  `docs/evidence/gate63_full_contour_solver_option_20260802.json`. These gates
  retain explicit Matsubara/antihermiticity diagnostics and do not claim a
  conserving closure when the Matsubara interaction is omitted.
  Gate64 adds an exact finite-contact many-body charge/spin oracle and records
  the interacting branch's deviations in
  `docs/evidence/gate64_exact_contact_charge_spin_20260802.json`.
  Gate65 records imaginary-grid and finite-lead-size convergence controls in
  `docs/evidence/gate65_mixed_grid_lead_size_convergence_20260802.json`.
  Gate66 is the application-side Corbino adapter, documented in
  `C:\Users\Nelson\Dev\physics\xene-ring-transport\docs\evidence\gate66_hubbard_corbino_closed_branch_20260802.json`.
  Gate67 records the separate persistent/reservoir comparison in the app
  evidence `gate67_closed_persistent_reservoir_20260802.json`.
  Gate68 records the closed-branch robustness controls and the explicit,
  non-forced protection criteria in the app evidence
  `gate68_closed_topology_criteria_20260802.json`.
  Gate69 integrates those eight records and the full 259/70 test regressions;
  its evidence is `docs/evidence/gate69_integrated_closed_branch_audit_20260802.json`.
  Gate70 is the ten-gate review and token checkpoint in
  `docs/evidence/gate70_review_20260802.json`; it classifies the bounded
  methodological result as draft-ready while keeping strong physics,
  continuum, novelty, and protection claims open.
  Gate71 adds the self-consistent Matsubara Hartree/second-Born branch and
  feeds its ΣM into the selectable full-contour lesser reconstruction. Its
  finite-grid KMS residual is retained as a diagnostic in
  `docs/evidence/gate71_self_consistent_matsubara_20260802.json`; this closes
  the software interface but does not yet establish a continuum conserving
  theorem.
  Gate72 extends the application closed branch to three larger Corbino annuli
  and measures the non-monotone edge/bulk spin crossover, with evidence in
  `C:\Users\Nelson\Dev\physics\xene-ring-transport\docs\evidence\gate72_hubbard_corbino_size_scaling_20260802.json`.
  Gate73 adds `solve_time_dependent_matrix_embedding` for arbitrary finite-grid
  matrix Σr(t,t')/Σ<(t,t') kernels, with explicit causal, adjoint and
  Keldysh diagnostics; its ASTRA/ASTRUM evidence is
  `docs/evidence/gate73_time_dependent_matrix_embedding_20260802.json`.
  Gate74 records the specialist prior-art and claim-boundary audit in
  `docs/evidence/gate74_specialist_novelty_audit_20260802.json`; broad method
  and protection novelty are rejected, leaving only an unconfirmed narrow
  benchmark-workflow candidate.
  Gate75 records the publication claim matrix in
  `docs/evidence/gate75_publication_claim_matrix_20260802.json`: 264 engine
  tests and 70 application tests pass in both ASTRA and ASTRUM, while the
  continuum-conserving, topological-protection, and broad-novelty claims stay
  explicitly open.
  Gate76 compares the self-consistent interacting branch with an exact finite
  contact over (U=0)--(0.8); the resulting Matsubara/conservation boundary
  is retained as a negative accuracy ledger in
  `docs/evidence/gate76_exact_interaction_accuracy_ledger_20260802.json`.
  Gate77 adds a finite-ramp AB transient control in the application: the
  sampled persistent response is delayed, continuity closes, and the
  persistent/reservoir channels remain separate; it is explicitly a quadratic
  memory control rather than a protection claim.
  Gate78 adds the finite-lead spinful Kane–Mele/trivial mass control with
  explicit charge, (s_z), torque, and reservoir-channel diagnostics; it also
  remains a control rather than a protection claim.
  Gate79 binds the latest engine/app gates and verifiers into a SHA-256
  reproducible manifest in
  `docs/evidence/gate79_reproducible_package_20260802.json`.
  Gate80 reviews Gates71–79: the bounded software/benchmark result is draft
  ready with explicit limitations, while continuum conservation, novelty and
  topological protection remain open. Its machine-readable decision is
  `docs/evidence/gate80_review_20260802.json`.
  Gate81 adds a same-self-energy continuity ledger that separates embedding
  and interaction charge/spin collision channels and verifies their algebraic
  sum; the nonzero closure residual remains explicit in
  `docs/evidence/gate81_same_self_energy_continuity_20260803.json`.
  Gate82 adds a lead-size extrapolation diagnostic for the spinful Corbino
  ramp; finite-size tails decrease across (L=3)--6 but remain explicitly
  finite in `gate82_lead_size_extrapolation_20260803.json`.
  Gate83 refreshes the complete regression baseline to 265 engine and 70
  application tests in both runtimes after the continuity API upgrade; see
  `docs/evidence/gate83_regression_refresh_20260803.json`.
  Gate84 refreshes the SHA-256 package manifest to include the continuity API,
  lead-size extrapolation and current 265/70 regression baseline in
  `docs/evidence/gate84_package_refresh_20260803.json`.
  Gate85 adds symbolic ordered KBE collision and continuity identities with
  optional vertical-branch source and formal charge/spin/torque projections;
  ASTRA/ASTRUM pass its eight checks in
  `docs/evidence/gate85_symbolic_continuity_20260803.json`. This is an
  analytic audit surface, not a conserving-continuum or protection theorem.
  Gate86 refreshes the complete regression baseline to 267 engine and 70
  application tests in both runtimes after that API upgrade; see
  `docs/evidence/gate86_regression_refresh_20260803.json`.
  Gate87 verifies noncommuting (S_x,S_z) torque against an exact finite-lead
  oracle and Gate88 verifies the Rashba torque switch in the Corbino app;
  their records are `gate87_noncommuting_spin_torque_20260803.json` and
  `gate88_rashba_spin_torque_20260803.json`. Gate89 binds this final round in
  `gate89_package_refresh_20260803.json`; all remain finite controls, not a
  topological-protection claim.
  Gate90 is the final ten-gate review: 267 engine and 70 application tests pass
  in ASTRA/ASTRUM, the bounded software result is draft-ready with limitations,
  and conservation/topological-protection claims remain open. Its evidence is
  `docs/evidence/gate90_final_review_20260803.json`; implementation is paused
  for project analysis.
- The novelty and publication boundary is recorded in
  `docs/NOVELTY_AUDIT_TRANSIENT_INTERACTING_SPIN.md`; the current verdict is
  an audit pass with open publication gates, not a claim of topological
  protection or a broad method novelty claim.

## Notes and conventions

- Units: `ħ = e = k_B = 1`; conductance prefactor `q²/2π` per spin channel.
- Retarded Green functions use `ω + iη` with `η > 0`; `advanced = retarded(η → −η)`.
- Frequency integrals use `numpy.trapezoid` on user-supplied grids — resolve sharp resonances (width ~Γ) with an adequate grid, especially at low temperature.
- The GPU extra targets CUDA 12 (`cupy-cuda12x`); use `cupy-cuda11x` for CUDA 11 hosts. `backend="auto"` falls back to NumPy when no usable GPU is present.
- Explicit two-time arrays scale as `n_time² × dim²` per component. The API
  refuses allocations above 512 MiB by default; use
  `propagate_density_matrix` when only equal-time observables are needed.
- The real-time layer is exact for finite quadratic Hamiltonians. The continuum
  API now transforms stationary matrix self-energies and open-device Green
  functions to explicit \((t,t')\) kernels, with a caller-controlled energy
  window and quadrature. Separate analytic oracles cover a suddenly coupled
  partitioned wide-band level and an initially contacted partition-free sudden
  bias quench. Constant post-quench wide-band matrix protocols are implemented
  by `partition_free_wide_band_matrix_quench` and
  `partition_free_wide_band_two_time_greens`. Interacting memory kernels now
  have a finite-grid Kadanoff–Baym/SCBA implementation, including the
  pure-boson full-contour branch, and an analytic Lorentzian reservoir-memory
  oracle. These are controlled research layers,
  not yet a production contour solver for arbitrary time-dependent matrix
  leads; full interacting continuity, KMS/continuum convergence, and reservoir
  spin-injection closure remain explicit publication gates.

## Citation

If this package contributes to a publication, cite it as:

> N. Bolívar, *QuantumTransportEOM: symbolic equation-of-motion Green functions and accelerated quantum transport*, v0.4.1 (2026).

Machine-readable metadata lives in `CITATION.cff` (GitHub renders a "Cite this
repository" button from it). Tagged releases are archived on Zenodo via the
GitHub integration: the concept DOI
[10.5281/zenodo.21265399](https://doi.org/10.5281/zenodo.21265399) always
resolves to the latest archived version; cite the version DOI of the release
you used:

| Release | Version DOI |
|---|---|
| v0.4.1 | minted on release |
| v0.4.0 | [10.5281/zenodo.21959411](https://doi.org/10.5281/zenodo.21959411) — superseded, cite v0.4.1 |
| v0.3.0 | [10.5281/zenodo.21265400](https://doi.org/10.5281/zenodo.21265400) |

