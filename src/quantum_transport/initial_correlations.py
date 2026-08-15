"""Mixed real--imaginary Keldysh branch for initial correlations.

For a partition-free contour the real-time Kadanoff--Baym equations contain
an additional vertical-branch term.  With the convention used by this
package, the left mixed kernel is

``I(t,t') = -i integral_0^beta d tau Sigma^rceil(t,tau)
G^lceil(tau,t')``

and the equal-time density source is ``I(t,t) + I(t,t)^dagger``.  The
functions below deliberately require both mixed kernels as input: a source
inferred only from a density residual is a diagnostic, not a microscopic
initial-correlation calculation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _grid(value: Any, *, name: str, minimum: int = 2) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or result.size < minimum:
        raise ValueError(f"{name} must be a one-dimensional grid with at least {minimum} points.")
    if not np.all(np.isfinite(result)) or np.any(np.diff(result) <= 0.0):
        raise ValueError(f"{name} must be finite and strictly increasing.")
    return result


def _weights(grid: np.ndarray) -> np.ndarray:
    result = np.empty_like(grid)
    result[0] = 0.5 * (grid[1] - grid[0])
    result[-1] = 0.5 * (grid[-1] - grid[-2])
    if grid.size > 2:
        result[1:-1] = 0.5 * (grid[2:] - grid[:-2])
    return result


def _mixed_stack(value: Any, shape: tuple[int, ...], *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.complex128)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}.")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values.")
    return result


def equilibrium_matsubara_green(
    hamiltonian: Any,
    imaginary_time: Any,
    *,
    chemical_potential: float = 0.0,
    temperature: float | None = None,
) -> np.ndarray:
    r"""Evaluate a finite one-body equilibrium Matsubara Green kernel.

    The array has shape ``(n_imaginary, n_imaginary, dim, dim)`` and uses
    ``G^M(tau,tau') = -exp[-(h-mu)(tau-tau')] (1-f)`` for
    ``tau >= tau'`` and the antiperiodic continuation ``+exp[-(h-mu)
    (tau-tau')] f`` otherwise.  If ``temperature`` is omitted, the grid
    span is interpreted as beta.  Logarithmic mode factors keep the KMS
    products bounded at the endpoints.
    """
    grid = _grid(imaginary_time, name="imaginary_time")
    matrix = np.asarray(hamiltonian, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("hamiltonian must be square.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("hamiltonian must contain only finite values.")
    if not np.allclose(matrix, matrix.conj().T, atol=1e-12, rtol=1e-12):
        raise ValueError("hamiltonian must be Hermitian.")
    beta = float(grid[-1] - grid[0])
    if beta <= 0.0 or not np.isfinite(beta):
        raise ValueError("imaginary_time must span a positive beta.")
    temp = 1.0 / beta if temperature is None else float(temperature)
    if temp <= 0.0 or not np.isfinite(temp):
        raise ValueError("temperature must be positive and finite.")
    mu = float(chemical_potential)
    if not np.isfinite(mu):
        raise ValueError("chemical_potential must be finite.")
    energies, states = np.linalg.eigh(matrix)
    scaled = (energies - mu) / temp
    log_fermi = -np.logaddexp(0.0, scaled)
    log_complement = -np.logaddexp(0.0, -scaled)
    result = np.empty((grid.size, grid.size, matrix.shape[0], matrix.shape[0]), dtype=np.complex128)
    for left, left_time in enumerate(grid):
        for right, right_time in enumerate(grid):
            delta = float(left_time - right_time)
            if delta >= 0.0:
                factors = -np.exp(log_complement - scaled * temp * delta)
            else:
                factors = np.exp(log_fermi - scaled * temp * delta)
            result[left, right] = states @ np.diag(factors) @ states.conj().T
    return result


def equilibrium_bosonic_matsubara_green(
    hamiltonian: Any,
    imaginary_time: Any,
    *,
    chemical_potential: float = 0.0,
    temperature: float | None = None,
) -> np.ndarray:
    r"""Evaluate a finite one-body bosonic Matsubara Green kernel.

    The kernel uses the same matrix convention as the numerical Kadanoff--
    Baym layer, but with periodic rather than antiperiodic boundary
    conditions,

    ``D^M(tau,tau') = -(N+1) exp[-eps (tau-tau')]`` for ``tau >= tau'``
    and ``-N exp[-eps (tau-tau')]`` otherwise.

    Here ``N=(exp(beta*eps)-1)^(-1)``.  The result is useful as the bare
    vertical branch for a harmonic bosonic mode.  Zero-energy modes are
    intentionally rejected because their finite-temperature covariance is
    infrared divergent; callers can regularise such a mode explicitly.
    """

    grid = _grid(imaginary_time, name="imaginary_time")
    matrix = np.asarray(hamiltonian, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("hamiltonian must be square.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("hamiltonian must contain only finite values.")
    if not np.allclose(matrix, matrix.conj().T, atol=1e-12, rtol=1e-12):
        raise ValueError("hamiltonian must be Hermitian.")
    beta = float(grid[-1] - grid[0])
    if beta <= 0.0 or not np.isfinite(beta):
        raise ValueError("imaginary_time must span a positive beta.")
    temp = 1.0 / beta if temperature is None else float(temperature)
    if temp <= 0.0 or not np.isfinite(temp):
        raise ValueError("temperature must be positive and finite.")
    mu = float(chemical_potential)
    if not np.isfinite(mu):
        raise ValueError("chemical_potential must be finite.")

    energies, states = np.linalg.eigh(matrix)
    shifted = energies - mu
    if np.any(shifted <= 0.0):
        raise ValueError("bosonic one-body energies must be strictly positive.")
    # Evaluate the occupations logarithmically.  Forming ``N`` and
    # ``exp(-eps*tau)`` separately overflows for ``beta*eps`` beyond about 710
    # even though their product stays bounded, which silently returns NaN at
    # the vertical endpoints.  This mirrors the guard already used by the
    # fermionic kernel above.
    scaled = shifted / temp
    small = np.minimum(scaled, 1.0)
    large = np.maximum(scaled, 1.0)
    log_expm1 = np.where(
        scaled < 1.0,
        np.log(np.expm1(small)),
        large + np.log1p(-np.exp(-large)),
    )
    log_occupation = -log_expm1
    log_occupation_plus = scaled - log_expm1
    result = np.empty((grid.size, grid.size, matrix.shape[0], matrix.shape[0]), dtype=np.complex128)
    for left, left_time in enumerate(grid):
        for right, right_time in enumerate(grid):
            delta = float(left_time - right_time)
            log_weight = log_occupation_plus if delta >= 0.0 else log_occupation
            factors = -np.exp(log_weight - shifted * delta)
            result[left, right] = states @ np.diag(factors) @ states.conj().T
    return result


@dataclass(frozen=True)
class InitialCorrelationResult:
    """Vertical-branch kernel and its equal-time density source."""

    time: np.ndarray
    imaginary_time: np.ndarray
    kernel: np.ndarray
    density_source: np.ndarray

    @property
    def hermiticity_error(self) -> float:
        source = self.density_source
        return float(np.max(np.abs(source - source.swapaxes(-1, -2).conj())))


@dataclass(frozen=True)
class LesserInitialCorrelationResult:
    """Retarded-propagated vertical contribution to ``G^<``.

    ``source_kernel`` is the left Kadanoff--Baym source
    ``I(t,t') = -i integral Sigma^rceil(t,tau) G^lceil(tau,t')``.  The
    finite-grid reconstruction propagates it with the retarded branch and
    adds the right-acting adjoint,
    ``Delta G^< = C - C^dagger`` with ``C = G^R * I``.  This is the explicit
    two-time initial-correlation term used by the research Dyson layer; it is
    not a substitute for a production contour solver or a non-Gaussian initial
    vertex.
    """

    time: np.ndarray
    imaginary_time: np.ndarray
    source_kernel: np.ndarray
    propagated_source: np.ndarray
    correction: np.ndarray

    @property
    def antihermiticity_error(self) -> float:
        correction = self.correction
        return float(np.max(np.abs(correction + correction.swapaxes(0, 1).conj().swapaxes(-1, -2))))


@dataclass(frozen=True)
class LesserContourCorrectionResult:
    """Three vertical Langreth terms in a contour lesser Dyson update."""

    time: np.ndarray
    imaginary_time: np.ndarray
    mixed_advanced: np.ndarray
    propagated_mixed: np.ndarray
    matsubara: np.ndarray
    correction: np.ndarray

    @property
    def antihermiticity_error(self) -> float:
        correction = self.correction
        return float(np.max(np.abs(correction + correction.swapaxes(0, 1).conj().swapaxes(-1, -2))))


@dataclass(frozen=True)
class MixedKBEResidual:
    """Finite-grid residuals for the real/vertical KBE mixed equations."""

    time: np.ndarray
    imaginary_time: np.ndarray
    rceil: np.ndarray
    lceil: np.ndarray | None = None

    @property
    def maximum_rceil(self) -> float:
        return float(np.max(np.abs(self.rceil)))

    @property
    def maximum_lceil(self) -> float | None:
        return None if self.lceil is None else float(np.max(np.abs(self.lceil)))


def _prefix_trapezoid_weights(
    grid: np.ndarray,
    endpoint: int,
    *,
    equal_time_kernel_halved: bool = False,
) -> np.ndarray:
    r"""Trapezoid weights for a causal integral over ``[grid[0], grid[endpoint]]``.

    ``equal_time_kernel_halved`` selects the endpoint weight.  This package
    stores retarded/advanced kernels with the ``theta(0)=1/2`` convention, so
    the array entry at the integration endpoint already carries a factor
    one half.  Combining it with the plain trapezoid endpoint weight
    ``(t_e-t_{e-1})/2`` halves the equal-time contribution twice and degrades
    the causal convolution from second to first order.  Pass ``True`` when the
    convolved kernel is evaluated at equal time on the endpoint so that the
    stored one half is compensated; keep ``False`` for a kernel whose endpoint
    is strictly off the time diagonal or which stores the full one-sided limit.
    """

    weights = np.zeros_like(grid)
    if endpoint == 0:
        return weights
    weights[0] = 0.5 * (grid[1] - grid[0])
    if endpoint > 1:
        weights[1:endpoint] = 0.5 * (grid[2 : endpoint + 1] - grid[: endpoint - 1])
    endpoint_step = grid[endpoint] - grid[endpoint - 1]
    weights[endpoint] = endpoint_step if equal_time_kernel_halved else 0.5 * endpoint_step
    weights[endpoint + 1 :] = 0.0
    return weights


def mixed_kbe_residual(
    time: Any,
    imaginary_time: Any,
    *,
    green_mixed: Any,
    self_energy_retarded: Any,
    self_energy_mixed: Any,
    green_matsubara: Any,
    hamiltonian: Any,
    green_lmixed: Any | None = None,
    self_energy_advanced: Any | None = None,
    self_energy_lmixed: Any | None = None,
) -> MixedKBEResidual:
    r"""Evaluate finite-grid residuals of the mixed KBE equations.

    ``green_mixed`` is ``G^rceil(t,tau)`` with shape ``(nt,ni,d,d)``;
    ``green_matsubara`` is ``G^M(tau,tau')`` with shape ``(ni,ni,d,d)``.
    The retarded real-time convolution uses the causal prefix trapezoid rule,
    while the vertical convolution uses the full imaginary grid.  If
    ``green_lmixed`` and its two self-energy counterparts are supplied, the
    adjoint/right-acting ``G^lceil`` residual is returned as well.
    """
    real_grid = _grid(time, name="time")
    imaginary_grid = _grid(imaginary_time, name="imaginary_time")
    nt, ni = real_grid.size, imaginary_grid.size
    gm = np.asarray(green_mixed, dtype=np.complex128)
    sr = np.asarray(self_energy_retarded, dtype=np.complex128)
    sm = np.asarray(self_energy_mixed, dtype=np.complex128)
    gM = np.asarray(green_matsubara, dtype=np.complex128)
    if gm.ndim != 4 or gm.shape[:2] != (nt, ni) or gm.shape[-1] != gm.shape[-2]:
        raise ValueError("green_mixed must have shape (n_time, n_imaginary, dim, dim).")
    dim = gm.shape[-1]
    expected_rr = (nt, nt, dim, dim)
    expected_mm = (nt, ni, dim, dim)
    expected_MM = (ni, ni, dim, dim)
    if sr.shape != expected_rr or sm.shape != expected_mm or gM.shape != expected_MM:
        raise ValueError("mixed KBE arrays have incompatible grid or matrix shapes.")
    h = np.asarray(hamiltonian, dtype=np.complex128)
    if h.ndim == 2:
        if h.shape != (dim, dim):
            raise ValueError("hamiltonian has the wrong matrix dimension.")
        h_stack = np.broadcast_to(h, (nt, dim, dim))
    elif h.ndim == 3 and h.shape == (nt, dim, dim):
        h_stack = h
    else:
        raise ValueError("hamiltonian must have shape (dim, dim) or (n_time, dim, dim).")
    arrays = (gm, sr, sm, gM, h_stack)
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise ValueError("mixed KBE inputs must contain only finite values.")
    derivative = np.gradient(gm, real_grid, axis=0, edge_order=1)
    vertical_weights = _weights(imaginary_grid)
    real_term = np.zeros_like(gm)
    for left in range(nt):
        weights = _prefix_trapezoid_weights(real_grid, left, equal_time_kernel_halved=True)
        real_term[left] = np.einsum(
            "kab,kjbc,k->jac", sr[left], gm, weights, optimize=True
        )
    vertical_term = np.einsum("ilab,ljbc,l->ijac", sm, gM, vertical_weights, optimize=True)
    coherent = np.einsum("iab,ijbc->ijac", h_stack, gm, optimize=True)
    rceil_residual = 1j * derivative - coherent - real_term + 1j * vertical_term

    lceil_residual = None
    if green_lmixed is not None:
        gl = np.asarray(green_lmixed, dtype=np.complex128)
        sa = sr.conj().swapaxes(0, 1).swapaxes(-1, -2) if self_energy_advanced is None else np.asarray(self_energy_advanced, dtype=np.complex128)
        sl = sm.swapaxes(0, 1).conj().swapaxes(-1, -2) if self_energy_lmixed is None else np.asarray(self_energy_lmixed, dtype=np.complex128)
        if gl.shape != (ni, nt, dim, dim) or sa.shape != expected_rr or sl.shape != (ni, nt, dim, dim):
            raise ValueError("left mixed arrays have incompatible shapes.")
        if not all(np.all(np.isfinite(value)) for value in (gl, sa, sl)):
            raise ValueError("left mixed inputs must contain only finite values.")
        left_derivative = np.gradient(gl, real_grid, axis=1, edge_order=1)
        left_real = np.zeros_like(gl)
        for right in range(nt):
            weights = _prefix_trapezoid_weights(real_grid, right, equal_time_kernel_halved=True)
            left_real[:, right] = np.einsum(
                "ilab,lbc,l->iac", gl, sa[:, right], weights, optimize=True
            )
        left_vertical = np.einsum("ijab,jkbc,j->ikac", gM, sl, vertical_weights, optimize=True)
        right_coherent = np.einsum("ijab,jbc->ijac", gl, h_stack, optimize=True)
        lceil_residual = -1j * left_derivative - right_coherent - left_real + 1j * left_vertical
    return MixedKBEResidual(
        time=real_grid.copy(),
        imaginary_time=imaginary_grid.copy(),
        rceil=rceil_residual,
        lceil=lceil_residual,
    )


def propagate_mixed_kbe_rceil(
    time: Any,
    imaginary_time: Any,
    *,
    initial_green_mixed: Any,
    self_energy_retarded: Any,
    self_energy_mixed: Any,
    green_matsubara: Any,
    hamiltonian: Any,
) -> np.ndarray:
    r"""Propagate ``G^rceil(t,tau)`` with a causal Volterra stepper.

    The initial slice ``initial_green_mixed`` is ``G^rceil(t0,tau)``.  Each
    subsequent slice solves the differential mixed KBE using the retarded
    memory prefix and the explicit vertical source.  The endpoint of the
    retarded convolution is treated explicitly (the retarded kernel is zero
    on the open equal-time contour); this gives a stable first-order
    research stepper that can be refined in ``time`` and used inside a joint
    contour fixed point.
    """
    real_grid = _grid(time, name="time")
    imaginary_grid = _grid(imaginary_time, name="imaginary_time")
    nt, ni = real_grid.size, imaginary_grid.size
    initial = np.asarray(initial_green_mixed, dtype=np.complex128)
    sigma_r = np.asarray(self_energy_retarded, dtype=np.complex128)
    sigma_m = np.asarray(self_energy_mixed, dtype=np.complex128)
    green_M = np.asarray(green_matsubara, dtype=np.complex128)
    if initial.ndim != 3 or initial.shape[0] != ni or initial.shape[-1] != initial.shape[-2]:
        raise ValueError("initial_green_mixed must have shape (n_imaginary, dim, dim).")
    dim = initial.shape[-1]
    if sigma_r.shape != (nt, nt, dim, dim) or sigma_m.shape != (nt, ni, dim, dim):
        raise ValueError("mixed propagator self-energies have incompatible shapes.")
    if green_M.shape != (ni, ni, dim, dim):
        raise ValueError("green_matsubara has incompatible shape.")
    h = np.asarray(hamiltonian, dtype=np.complex128)
    if h.ndim == 2:
        if h.shape != (dim, dim):
            raise ValueError("hamiltonian has the wrong matrix dimension.")
        h_stack = np.broadcast_to(h, (nt, dim, dim))
    elif h.ndim == 3 and h.shape == (nt, dim, dim):
        h_stack = h
    else:
        raise ValueError("hamiltonian must have shape (dim, dim) or (n_time, dim, dim).")
    if not all(np.all(np.isfinite(value)) for value in (initial, sigma_r, sigma_m, green_M, h_stack)):
        raise ValueError("mixed propagator inputs must be finite.")
    vertical_weights = _weights(imaginary_grid)
    result = np.empty((nt, ni, dim, dim), dtype=np.complex128)
    result[0] = initial
    for left in range(1, nt):
        dt = float(real_grid[left] - real_grid[left - 1])
        # The memory endpoint is ``t_{left-1}``, strictly off the time diagonal
        # of ``sigma_r[left]``, so the plain trapezoid endpoint applies here.
        memory_weights = _prefix_trapezoid_weights(
            real_grid, left - 1, equal_time_kernel_halved=False
        )
        real_memory = np.einsum(
            "kab,kjbc,k->jac",
            sigma_r[left, :left],
            result[:left],
            memory_weights[:left],
            optimize=True,
        )
        vertical_source = np.einsum(
            "lab,ljbc,l->jac",
            sigma_m[left],
            green_M,
            vertical_weights,
            optimize=True,
        )
        rhs = (
            -1j * np.einsum("ab,jbc->jac", h_stack[left], result[left - 1], optimize=True)
            - 1j * real_memory
            - vertical_source
        )
        result[left] = result[left - 1] + dt * rhs
    return result


def kbe_initial_correlation_kernel(
    time: Any,
    imaginary_time: Any,
    *,
    self_energy_mixed: Any,
    green_mixed: Any,
) -> InitialCorrelationResult:
    r"""Evaluate the vertical Keldysh branch and its density source.

    ``self_energy_mixed`` is ``Sigma^rceil(t,tau)`` with shape
    ``(n_time, n_imaginary, dim, dim)``.  ``green_mixed`` is
    ``G^lceil(tau,t')`` with shape ``(n_imaginary, n_time, dim, dim)``.
    The imaginary grid is a positive, increasing parameter from ``0`` to
    ``beta``; its endpoint values are included in the trapezoidal rule.
    """

    real_grid = _grid(time, name="time")
    imaginary_grid = _grid(imaginary_time, name="imaginary_time")
    beta = float(imaginary_grid[-1] - imaginary_grid[0])
    if beta <= 0.0 or not np.isfinite(beta):
        raise ValueError("imaginary_time must span a positive inverse temperature.")
    sigma = np.asarray(self_energy_mixed, dtype=np.complex128)
    if sigma.ndim != 4 or sigma.shape[0:2] != (real_grid.size, imaginary_grid.size):
        raise ValueError(
            "self_energy_mixed must have shape (n_time, n_imaginary, dim, dim)."
        )
    if sigma.shape[2] != sigma.shape[3]:
        raise ValueError("self_energy_mixed must contain square matrices.")
    if not np.all(np.isfinite(sigma)):
        raise ValueError("self_energy_mixed must contain only finite values.")
    dim = sigma.shape[-1]
    green = _mixed_stack(
        green_mixed,
        (imaginary_grid.size, real_grid.size, dim, dim),
        name="green_mixed",
    )
    weights = _weights(imaginary_grid)
    kernel = -1j * np.einsum(
        "ikab,kjbc,k->ijac", sigma, green, weights, optimize=True
    )
    diagonal = kernel[np.arange(real_grid.size), np.arange(real_grid.size)]
    source = diagonal + diagonal.swapaxes(-1, -2).conj()
    source = 0.5 * (source + source.swapaxes(-1, -2).conj())
    return InitialCorrelationResult(
        time=real_grid.copy(),
        imaginary_time=imaginary_grid.copy(),
        kernel=kernel,
        density_source=source,
    )


def kbe_lesser_initial_correlation(
    time: Any,
    imaginary_time: Any,
    *,
    green_retarded: Any,
    self_energy_mixed: Any,
    green_lmixed: Any,
) -> LesserInitialCorrelationResult:
    r"""Build the two-time vertical initial-correlation correction.

    With ``I(t,t') = -i integral_0^beta d tau Sigma^rceil(t,tau)
    G^lceil(tau,t')``, this evaluates the causal real-time propagation

    ``C(t,t') = integral_t0^t d tbar G^R(t,tbar) I(tbar,t')``

    and returns the anti-Hermitian lesser contribution ``C-C^dagger``.  The
    causal and imaginary integrals use finite-grid trapezoidal weights.  The
    routine deliberately exposes both ``I`` and ``C`` so a continuity audit can
    distinguish the microscopic vertical source from the propagated lesser
    correction.

    ``green_retarded`` must follow the package ``theta(0)=1/2`` convention on
    the time diagonal, as produced by :func:`two_time_greens` and
    :func:`kadanoff_baym_dyson_two_time`.  The causal quadrature compensates
    that stored one half at the integration endpoint; supplying a kernel that
    stores the full one-sided limit instead would double the equal-time
    contribution.
    """

    real_grid = _grid(time, name="time")
    imaginary_grid = _grid(imaginary_time, name="imaginary_time")
    nt, ni = real_grid.size, imaginary_grid.size
    retarded = np.asarray(green_retarded, dtype=np.complex128)
    sigma = np.asarray(self_energy_mixed, dtype=np.complex128)
    if retarded.ndim != 4 or retarded.shape[0:2] != (nt, nt) or retarded.shape[-1] != retarded.shape[-2]:
        raise ValueError("green_retarded must have shape (n_time, n_time, dim, dim).")
    dim = retarded.shape[-1]
    if sigma.shape != (nt, ni, dim, dim):
        raise ValueError("self_energy_mixed must have shape (n_time, n_imaginary, dim, dim).")
    green = _mixed_stack(
        green_lmixed,
        (ni, nt, dim, dim),
        name="green_lmixed",
    )
    if not np.all(np.isfinite(retarded)) or not np.all(np.isfinite(sigma)):
        raise ValueError("lesser initial-correlation inputs must contain only finite values.")
    imaginary_weights = _weights(imaginary_grid)
    source = -1j * np.einsum(
        "ikab,kjbc,k->ijac", sigma, green, imaginary_weights, optimize=True
    )
    propagated = np.zeros_like(source)
    for left in range(nt):
        causal_weights = _prefix_trapezoid_weights(real_grid, left, equal_time_kernel_halved=True)
        propagated[left] = np.einsum(
            "kab,kjbc,k->jac", retarded[left], source, causal_weights, optimize=True
        )
    correction = propagated - propagated.swapaxes(0, 1).conj().swapaxes(-1, -2)
    return LesserInitialCorrelationResult(
        time=real_grid.copy(),
        imaginary_time=imaginary_grid.copy(),
        source_kernel=source,
        propagated_source=propagated,
        correction=correction,
    )


def kbe_lesser_contour_correction(
    time: Any,
    imaginary_time: Any,
    *,
    bare_retarded: Any,
    bare_mixed: Any,
    self_energy_mixed: Any,
    green_lmixed: Any,
    green_advanced: Any,
    self_energy_matsubara: Any | None = None,
    self_energy_lmixed: Any | None = None,
) -> LesserContourCorrectionResult:
    r"""Evaluate all three vertical terms of ``(g Sigma G)^<``.

    The finite-grid Langreth reconstruction is

    ``g^rceil star Sigma^lceil dot G^A``
    ``+ g^R dot Sigma^rceil star G^lceil``
    ``+ g^rceil star Sigma^M star G^lceil``.

    The imaginary contour measure is ``-i d tau``; hence the first two terms
    carry ``-i`` and the double-Matsubara term carries ``(-i)^2``.  The result
    deliberately exposes each contribution so a solver can test the missing
    branch separately instead of silently folding it into a residual source.
    """

    real_grid = _grid(time, name="time")
    imaginary_grid = _grid(imaginary_time, name="imaginary_time")
    nt, ni = real_grid.size, imaginary_grid.size
    bare_r = np.asarray(bare_retarded, dtype=np.complex128)
    bare_m = np.asarray(bare_mixed, dtype=np.complex128)
    sigma_m = np.asarray(self_energy_mixed, dtype=np.complex128)
    gl = np.asarray(green_lmixed, dtype=np.complex128)
    ga = np.asarray(green_advanced, dtype=np.complex128)
    if bare_r.ndim != 4 or bare_r.shape[0:2] != (nt, nt) or bare_r.shape[-1] != bare_r.shape[-2]:
        raise ValueError("bare_retarded must have shape (n_time, n_time, dim, dim).")
    dim = bare_r.shape[-1]
    expected_rm = (nt, ni, dim, dim)
    expected_mr = (ni, nt, dim, dim)
    expected_rr = (nt, nt, dim, dim)
    expected_mm = (ni, ni, dim, dim)
    if bare_m.shape != expected_rm or sigma_m.shape != expected_rm or gl.shape != expected_mr or ga.shape != expected_rr:
        raise ValueError("vertical contour arrays have incompatible shapes.")
    sigma_l = sigma_m.swapaxes(0, 1).conj().swapaxes(-1, -2) if self_energy_lmixed is None else np.asarray(self_energy_lmixed, dtype=np.complex128)
    sigma_M = np.zeros(expected_mm, dtype=np.complex128) if self_energy_matsubara is None else np.asarray(self_energy_matsubara, dtype=np.complex128)
    if sigma_l.shape != expected_mr or sigma_M.shape != expected_mm:
        raise ValueError("self_energy_lmixed or self_energy_matsubara has an incompatible shape.")
    if not all(np.all(np.isfinite(value)) for value in (bare_r, bare_m, sigma_m, sigma_l, sigma_M, gl, ga)):
        raise ValueError("vertical contour arrays must contain only finite values.")
    real_weights = _weights(real_grid)
    imaginary_weights = _weights(imaginary_grid)
    mixed_advanced = -1j * np.einsum(
        "ikab,kjbc,k->ijac", bare_m, sigma_l, imaginary_weights, optimize=True
    )
    mixed_advanced = np.einsum(
        "ikab,kjbc,k->ijac", mixed_advanced, ga, real_weights, optimize=True
    )
    source = -1j * np.einsum(
        "ikab,kjbc,k->ijac", sigma_m, gl, imaginary_weights, optimize=True
    )
    propagated = np.einsum(
        "ikab,kjbc,k->ijac", bare_r, source, real_weights, optimize=True
    )
    matsubara = -np.einsum(
        "ikab,klbc,ljcd,k,l->ijad",
        bare_m,
        sigma_M,
        gl,
        imaginary_weights,
        imaginary_weights,
        optimize=True,
    )
    correction = mixed_advanced + propagated + matsubara
    return LesserContourCorrectionResult(
        time=real_grid.copy(),
        imaginary_time=imaginary_grid.copy(),
        mixed_advanced=mixed_advanced,
        propagated_mixed=propagated,
        matsubara=matsubara,
        correction=correction,
    )


def continuity_residual_after_initial_correlation(
    residual: Any,
    density_source: Any,
) -> np.ndarray:
    """Subtract a microscopic vertical-branch source from a KBE residual."""

    residual_array = np.asarray(residual, dtype=np.complex128)
    source = np.asarray(density_source, dtype=np.complex128)
    if residual_array.shape != source.shape or residual_array.ndim != 3:
        raise ValueError("residual and density_source must have shape (n_time, dim, dim).")
    if not np.all(np.isfinite(residual_array)) or not np.all(np.isfinite(source)):
        raise ValueError("residual and density_source must be finite.")
    return residual_array - source


def project_initial_correlation_source(source: Any, observable: Any) -> np.ndarray:
    """Project a Hermitian initial-correlation source onto an observable."""
    source_array = np.asarray(source, dtype=np.complex128)
    operator = np.asarray(observable, dtype=np.complex128)
    if source_array.ndim != 3 or source_array.shape[-1] != source_array.shape[-2]:
        raise ValueError("source must have shape (n_time, dim, dim).")
    if operator.shape != source_array.shape[-2:]:
        raise ValueError("observable must have the source matrix dimension.")
    if not np.all(np.isfinite(source_array)) or not np.all(np.isfinite(operator)):
        raise ValueError("source and observable must be finite.")
    return np.real(np.einsum("ab,iba->i", operator, source_array, optimize=True))


def initial_correlation_charge_spin_source(
    source: Any,
    spin_operator: Any,
) -> dict[str, np.ndarray]:
    """Return charge and spin projections of a vertical-branch source."""
    source_array = np.asarray(source, dtype=np.complex128)
    identity = np.eye(source_array.shape[-1], dtype=np.complex128)
    return {
        "charge": project_initial_correlation_source(source_array, identity),
        "spin": project_initial_correlation_source(source_array, spin_operator),
    }


def required_initial_source_from_residual(residual: Any) -> np.ndarray:
    """Return the Hermitian source required by a continuity residual.

    This is an audit diagnostic for a continuum or interacting run.  It is
    deliberately named ``required`` rather than ``microscopic``: the result
    must not be used as a substitute for mixed kernels supplied by a contour
    solver.
    """

    residual_array = np.asarray(residual, dtype=np.complex128)
    if residual_array.ndim != 3 or residual_array.shape[-1] != residual_array.shape[-2]:
        raise ValueError("residual must have shape (n_time, dim, dim).")
    if not np.all(np.isfinite(residual_array)):
        raise ValueError("residual must be finite.")
    return 0.5 * (residual_array + residual_array.swapaxes(-1, -2).conj())


__all__ = [
    "InitialCorrelationResult",
    "LesserContourCorrectionResult",
    "LesserInitialCorrelationResult",
    "MixedKBEResidual",
    "continuity_residual_after_initial_correlation",
    "equilibrium_bosonic_matsubara_green",
    "equilibrium_matsubara_green",
    "initial_correlation_charge_spin_source",
    "kbe_initial_correlation_kernel",
    "kbe_lesser_initial_correlation",
    "kbe_lesser_contour_correction",
    "mixed_kbe_residual",
    "propagate_mixed_kbe_rceil",
    "project_initial_correlation_source",
    "required_initial_source_from_residual",
]
