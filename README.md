# QuantumTransportEOM

**Symbolic equation-of-motion Green functions and fast numerical quantum transport in one package.**

`quantum_transport` couples a SymPy-based second-quantization layer (commutators, EOM closure, mean-field truncations, Keldysh contour algebra) to a NumPy transport engine (Landauer, Meir–Wingreen, spin-resolved observables) with batched linear algebra, multi-core frequency sweeps, and optional CUDA GPU execution via CuPy.

Version `0.2.0` — Python ≥ 3.10, MIT license.

---

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

## Capability map

| Layer | Highlights |
|---|---|
| Operator algebra | `commutator`, `anticommutator`, fermionic/bosonic wrappers (`SQObj`, `BQObj`), Wick + exact CAR normal ordering (`normal_order_fermionic`), `dagger_expression` |
| Symbolic models | Predefined (`AndersonImpurity`, `FermionicSingleLevel`, dimers, Holstein, Jaynes–Cummings…) and `CustomModel` for arbitrary Hamiltonians; EOM closure analysis, basis auto-expansion, Hartree / Hubbard-I truncations |
| Green functions | Retarded/advanced/lesser/greater, spectral functions, symbolic *and* numeric, equilibrium distributions |
| Keldysh | Contour objects, Langreth rules, Dyson equations, symbolic Meir–Wingreen, wide-band stationary results (`keldysh_symbolic`), numeric Keldysh self-energies |
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

Run everything:

```bash
python -m pytest            # 169 tests
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
| `examples/demo_ab_ring.py`, `demo_models.py`, … | Rings, spin transport, symbolic model gallery |

PowerShell automation (runs demos, tests, and optionally builds the manual):

```powershell
.\run_all.ps1
.\run_all.ps1 -InstallPytest -BuildManual
```

## Documentation

- LaTeX manual: `docs/user_manual.tex` (PDF: `docs/user_manual.pdf`, build with `.\run_all.ps1 -BuildManual`).
- Every public function has a docstring; the high-level API is discoverable from `quantum_transport.__all__`.

## Notes and conventions

- Units: `ħ = e = k_B = 1`; conductance prefactor `q²/2π` per spin channel.
- Retarded Green functions use `ω + iη` with `η > 0`; `advanced = retarded(η → −η)`.
- Frequency integrals use `numpy.trapezoid` on user-supplied grids — resolve sharp resonances (width ~Γ) with an adequate grid, especially at low temperature.
- The GPU extra targets CUDA 12 (`cupy-cuda12x`); use `cupy-cuda11x` for CUDA 11 hosts. `backend="auto"` falls back to NumPy when no usable GPU is present.

## Citation

If this package contributes to a publication, cite it as:

> N. Bolívar, *QuantumTransportEOM: symbolic equation-of-motion Green functions and accelerated quantum transport*, v0.2.0 (2026).
