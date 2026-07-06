"""Acceleration demo: batched frequency sweeps, CPU workers, and optional GPU.

Every ``*_values`` method on the numeric transport views accepts:

- ``backend=`` : "numpy" (default via "auto"), "cupy" for CUDA GPUs;
- ``workers=`` : number of CPU threads mapping frequency blocks in parallel.

The batched kernels replace the per-frequency Python loop with stacked
LAPACK/BLAS (or cuSOLVER/cuBLAS) calls, block the grid to cap peak memory,
and pin small-matrix factorizations to one BLAS thread so ``workers`` scales
across cores instead of fighting OpenBLAS's own pool.
"""

import time

import numpy as np

from quantum_transport import (
    AndersonImpurity,
    LeadSelfEnergy,
    RashbaRingDevice,
    available_backends,
)


def bench(label: str, fn) -> float:
    start = time.perf_counter()
    fn()
    elapsed = time.perf_counter() - start
    print(f"  {label:34s} {elapsed:8.3f} s")
    return elapsed


def ring_transmission_sweep() -> None:
    print("=== Rashba ring (dim = 120), transmission on 400 frequencies ===")
    device = RashbaRingDevice(n_sites=60, gamma=1.0, lambda_r=0.35, phi_over_phi0=0.2)
    dim = device.dim
    gamma_left = np.zeros((dim, dim))
    gamma_left[0, 0] = gamma_left[1, 1] = 0.5
    gamma_right = np.zeros((dim, dim))
    gamma_right[dim // 2, dim // 2] = gamma_right[dim // 2 + 1, dim // 2 + 1] = 0.5
    view = device.transport(LeadSelfEnergy.wide_band(gamma_left), LeadSelfEnergy.wide_band(gamma_right))
    grid = np.linspace(-3.0, 3.0, 400)

    bench("scalar loop (reference, 40 pts x10)", lambda: [view.transmission(float(w), eta=1e-6) for w in grid[:40]])
    bench("batched (backend='numpy')", lambda: view.transmission_values(grid, eta=1e-6))
    bench("batched + workers=8", lambda: view.transmission_values(grid, eta=1e-6, workers=8))

    if available_backends()["cupy"]:
        # warm-up compiles kernels / initializes the CUDA context
        view.transmission_values(grid[:8], eta=1e-6, backend="cupy")
        bench("batched on GPU (backend='cupy')", lambda: view.transmission_values(grid, eta=1e-6, backend="cupy"))
    else:
        print("  [GPU] cupy not available on this machine - install `pip install .[gpu]` on a CUDA host")
    print()


def anderson_scf() -> None:
    print("=== Open Anderson dot: self-consistent occupations (Hubbard-I) ===")
    impurity = AndersonImpurity(eps=-0.5, U=2.0)
    view = impurity.open(0.3, 0.3, mu_left=0.25, mu_right=-0.25)
    grid = np.linspace(-8.0, 8.0, 1200)

    def scf(workers=None):
        return view.self_consistent_occupations(grid, eta=1e-3, tol=1e-9, workers=workers)

    bench("SCF loop (batched serial)", scf)
    bench("SCF loop (workers=4)", lambda: scf(workers=4))
    result = scf()
    print(f"  converged={result.converged} in {result.iterations} iterations -> "
          f"n_up={result.occupations['up']:.4f}, n_down={result.occupations['down']:.4f}")
    print()


if __name__ == "__main__":
    print("available backends:", available_backends())
    print()
    ring_transmission_sweep()
    anderson_scf()
