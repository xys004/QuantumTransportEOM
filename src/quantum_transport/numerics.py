"""Numerical acceleration: pluggable array backends (NumPy/CuPy), batched Green-function kernels, and CPU-parallel sweeps.

The public entry points are:

- :func:`get_backend` / :func:`available_backends` — resolve the array module
  (``numpy`` on CPU, ``cupy`` on CUDA GPUs) used by the batched kernels.
- :func:`parallel_map` — thread-pool map for embarrassingly parallel sweeps
  (NumPy/SymPy-lambdified kernels release the GIL inside BLAS/LAPACK).
- :func:`batched_retarded_green`, :func:`batched_transmission`,
  :func:`batched_keldysh_component`, :func:`batched_current_spectral_density`
  — frequency-stacked linear algebra shared by the fast paths of
  :class:`~quantum_transport.devices.MatrixTransportView` and
  :class:`~quantum_transport.highlevel.OpenAndersonTransportView`.

All batched kernels accept and return arrays of shape ``(n_omega, dim, dim)``
(or ``(n_omega,)`` for scalars) and take an ``xp`` array module so the same
code path runs on CPU and GPU.
"""

from __future__ import annotations

import contextlib
import importlib
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, Sequence

import numpy as np

try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
except ImportError:  # pragma: no cover - threadpoolctl is a hard dependency
    _threadpool_limits = None

# Below this matrix dimension, multithreaded BLAS/LAPACK factorizations lose
# far more to thread synchronization than they gain (9-30x slowdowns measured
# with OpenBLAS on many-core machines, still ~1.2x at dim=1024); run them
# single-threaded and let callers parallelize across frequency blocks instead.
BLAS_SINGLE_THREAD_MAX_DIM = 1024


_BACKEND_ALIASES = {
    "cpu": "numpy",
    "np": "numpy",
    "gpu": "cupy",
    "cuda": "cupy",
    "cp": "cupy",
}


def _import_cupy():
    try:
        cupy = importlib.import_module("cupy")
    except ImportError as exc:
        raise ImportError(
            "CuPy is not installed. Install the GPU extra with "
            "`pip install quantum-transport-eom[gpu]` or pick the wheel matching "
            "your CUDA toolkit, e.g. `pip install cupy-cuda12x`."
        ) from exc
    try:
        device_count = cupy.cuda.runtime.getDeviceCount()
    except Exception as exc:  # pragma: no cover - depends on local driver state
        raise RuntimeError(f"CuPy is installed but CUDA is not usable: {exc}") from exc
    if device_count < 1:  # pragma: no cover - depends on local hardware
        raise RuntimeError("CuPy is installed but no CUDA device was detected.")
    return cupy


def available_backends() -> dict[str, bool]:
    """Report which array backends can be used on this machine."""
    cupy_ok = True
    try:
        _import_cupy()
    except Exception:
        cupy_ok = False
    return {"numpy": True, "cupy": cupy_ok}


def get_backend(name: str | Any = "auto"):
    """
    Resolve an array module for the batched kernels.

    Parameters
    ----------
    name:
        ``"numpy"``/``"cpu"``, ``"cupy"``/``"gpu"``/``"cuda"``, or ``"auto"``
        (CuPy when a CUDA device is available, NumPy otherwise). ``None`` means
        ``"auto"``. An already-imported array module is passed through.
    """
    if name is None:
        name = "auto"
    if not isinstance(name, str):
        return name
    key = _BACKEND_ALIASES.get(name.lower(), name.lower())
    if key == "auto":
        try:
            return _import_cupy()
        except Exception:
            return np
    if key == "numpy":
        return np
    if key == "cupy":
        return _import_cupy()
    raise ValueError(f"Unknown backend {name!r}; expected 'numpy', 'cupy', or 'auto'.")


def backend_name(xp: Any) -> str:
    """Short name of an array module returned by :func:`get_backend`."""
    return str(getattr(xp, "__name__", xp))


def to_numpy(array: Any) -> np.ndarray:
    """Move an array from any backend back to host NumPy."""
    if isinstance(array, np.ndarray):
        return array
    getter = getattr(array, "get", None)
    if callable(getter):
        return np.asarray(getter())
    return np.asarray(array)


def parallel_map(
    func: Callable[[Any], Any],
    values: Iterable[Any],
    *,
    workers: int | None = None,
) -> list[Any]:
    """
    Map ``func`` over ``values``, optionally with a thread pool.

    ``workers=None`` (or ``<= 1``) runs serially. Threads are the right pool
    here because the per-item work is NumPy linear algebra, which releases the
    GIL inside BLAS/LAPACK, and thread pools avoid pickling the closures that
    process pools would require.
    """
    items = list(values)
    if workers is None or int(workers) <= 1 or len(items) <= 1:
        return [func(value) for value in items]
    with ThreadPoolExecutor(max_workers=int(workers)) as pool:
        return list(pool.map(func, items))


@contextlib.contextmanager
def _small_matrix_blas_context(dim: int, xp: Any):
    """Single-thread BLAS for small-matrix LAPACK bursts on the NumPy backend."""
    if xp is np and _threadpool_limits is not None and dim <= BLAS_SINGLE_THREAD_MAX_DIM:
        with _threadpool_limits(limits=1, user_api="blas"):
            yield
    else:
        yield


def blocked_over_grid(
    fn: Callable[[np.ndarray], np.ndarray],
    omega_grid: np.ndarray,
    dim: int,
    *,
    workers: int | None = None,
    target_bytes: int = 64 * 2**20,
) -> np.ndarray:
    """
    Evaluate a stack-producing kernel over frequency blocks.

    Splits ``omega_grid`` so each block's ``(n_block, dim, dim)`` complex stack
    stays under ``target_bytes`` — shared across workers, since each worker
    holds its own intermediates — capping peak memory for large devices, and
    maps blocks with :func:`parallel_map`; combined with the single-threaded
    BLAS context inside the batched kernels, ``workers > 1`` gives real
    multi-core scaling.
    """
    grid = np.asarray(omega_grid, dtype=float)
    if grid.size == 0:
        return fn(grid)
    pool_size = int(workers) if workers is not None and int(workers) > 1 else 1
    block = max(1, int(target_bytes // pool_size // (16 * max(int(dim), 1) ** 2)))
    if pool_size > 1:
        per_worker = max(1, -(-grid.size // pool_size))
        block = min(block, per_worker)
    if block >= grid.size:
        return fn(grid)
    chunks = [grid[start : start + block] for start in range(0, grid.size, block)]
    return np.concatenate(parallel_map(fn, chunks, workers=workers), axis=0)


def sigma_stack(
    sigma_fn: Callable[[float], np.ndarray],
    omega_grid: np.ndarray,
    *,
    workers: int | None = None,
) -> np.ndarray:
    """Evaluate a per-frequency matrix callable into an ``(n, d, d)`` stack."""
    grid = np.asarray(omega_grid, dtype=float)
    values = parallel_map(
        lambda omega: np.asarray(sigma_fn(float(omega)), dtype=np.complex128),
        grid,
        workers=workers,
    )
    if not values:
        return np.zeros((0, 0, 0), dtype=np.complex128)
    return np.stack(values, axis=0)


def _as_stack(matrix: Any, n: int, xp: Any) -> Any:
    """Promote an ``(d, d)`` matrix to an ``(n, d, d)`` stack (no copy)."""
    arr = xp.asarray(matrix)
    if arr.ndim == 2:
        return xp.broadcast_to(arr, (n, *arr.shape))
    return arr


def dagger_stack(stack: Any, xp: Any = np) -> Any:
    """Hermitian conjugate of every matrix in an ``(n, d, d)`` stack."""
    return xp.conj(xp.swapaxes(xp.asarray(stack), -1, -2))


def gamma_from_sigma_stack(sigma_retarded: Any, xp: Any = np) -> Any:
    """Broadening Gamma = i (Sigma^r - Sigma^a) for a whole stack."""
    sig = xp.asarray(sigma_retarded)
    return 1j * (sig - dagger_stack(sig, xp))


def batched_retarded_green(
    hamiltonian: Any,
    sigma_retarded: Any,
    omega_grid: np.ndarray,
    *,
    eta: float = 0.0,
    xp: Any = np,
) -> Any:
    """
    Retarded Green function for every frequency at once.

    Computes ``G^r(omega) = [(omega + i eta) I - H - Sigma^r(omega)]^{-1}`` as a
    single batched inversion of shape ``(n, d, d)`` — one LAPACK/cuSOLVER call
    instead of a Python loop.
    """
    h = xp.asarray(hamiltonian)
    dim = h.shape[-1]
    grid = xp.asarray(np.asarray(omega_grid, dtype=float))
    n = grid.shape[0]
    sig = _as_stack(sigma_retarded, n, xp)
    z = (grid + 1j * float(eta))[:, None, None] * xp.eye(dim, dtype=np.complex128)
    with _small_matrix_blas_context(int(dim), xp):
        return xp.linalg.inv(z - h[None, :, :] - sig)


def batched_transmission(
    g_retarded: Any,
    gamma_left: Any,
    gamma_right: Any,
    *,
    xp: Any = np,
) -> Any:
    """Landauer transmission ``T(omega) = Re tr[Gamma_L G^r Gamma_R G^a]`` for a stack."""
    g_r = xp.asarray(g_retarded)
    n = g_r.shape[0]
    g_a = dagger_stack(g_r, xp)
    gl = _as_stack(gamma_left, n, xp)
    gr = _as_stack(gamma_right, n, xp)
    # Chain batched matmuls (BLAS) instead of one 4-operand einsum, whose
    # default non-optimized contraction path is O(n d^4).
    product = gl @ g_r @ gr @ g_a
    return xp.real(xp.einsum("nii->n", product))


def batched_keldysh_component(
    g_retarded: Any,
    sigma_component: Any,
    *,
    xp: Any = np,
) -> Any:
    """Keldysh component ``G^{<,>} = G^r Sigma^{<,>} G^a`` for a whole stack."""
    g_r = xp.asarray(g_retarded)
    n = g_r.shape[0]
    sig = _as_stack(sigma_component, n, xp)
    return g_r @ sig @ dagger_stack(g_r, xp)


def batched_current_spectral_density(
    g_lesser: Any,
    g_greater: Any,
    sigma_lesser_lead: Any,
    sigma_greater_lead: Any,
    *,
    charge: float = 1.0,
    xp: Any = np,
) -> Any:
    """Meir-Wingreen integrand ``(q/2pi) Re tr[Sigma^< G^> - Sigma^> G^<]`` for a stack."""
    g_l = xp.asarray(g_lesser)
    g_g = xp.asarray(g_greater)
    n = g_l.shape[0]
    sig_l = _as_stack(sigma_lesser_lead, n, xp)
    sig_g = _as_stack(sigma_greater_lead, n, xp)
    trace = xp.einsum("nij,nji->n", sig_l, g_g) - xp.einsum("nij,nji->n", sig_g, g_l)
    return (float(charge) / (2.0 * np.pi)) * xp.real(trace)
