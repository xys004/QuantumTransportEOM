"""Composable observables and standard transport expressions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import sympy as sp


ObservableLike = Any


def _unwrap(value: ObservableLike) -> Any:
    if isinstance(value, ObservableExpr):
        return value.expr
    if hasattr(value, "doit") and value.__class__.__name__ == "KeldyshExpression":
        return value.doit()
    return value


def _simplify_value(value: Any) -> Any:
    if isinstance(value, sp.MatrixBase):
        return value.applyfunc(sp.simplify)
    return sp.simplify(value)


def _is_matrix(value: Any) -> bool:
    return isinstance(value, sp.MatrixBase)


def _trace_value(value: Any) -> sp.Expr:
    inner = _unwrap(value)
    if _is_matrix(inner):
        return sp.trace(inner)
    return inner


def _integral_limits(limits: Sequence[Any]) -> tuple[Any, Any]:
    if len(limits) != 2:
        raise ValueError("limits must contain exactly two entries: (lower, upper)")
    return limits[0], limits[1]


def _fermi_symbolic(omega: sp.Expr, mu: sp.Expr, temperature: Any = 0) -> sp.Expr:
    if temperature is None:
        return sp.Function("f_FD")(omega)
    if temperature == 0:
        return sp.Heaviside(mu - omega)
    return sp.simplify(1 / (sp.exp((omega - mu) / temperature) + 1))


def _fermi_derivative_symbolic(omega: sp.Expr, mu: sp.Expr, temperature: Any = 0) -> sp.Expr:
    if temperature is None:
        return -sp.diff(sp.Function("f_FD")(omega), omega)
    if temperature == 0:
        return sp.DiracDelta(omega - mu)
    f = _fermi_symbolic(omega, mu, temperature)
    return sp.simplify(f * (1 - f) / temperature)


def _spin_mask(label: Any, spin: str) -> bool:
    spin_key = str(spin).lower()
    if isinstance(label, (tuple, list)) and label:
        return str(label[-1]).lower() == spin_key
    text = str(label).lower()
    if text == spin_key:
        return True
    return text.endswith(f"_{spin_key}") or text.endswith(f"({spin_key})") or text.endswith(spin_key)


@dataclass(frozen=True)
class ObservableExpr:
    expr: Any

    def doit(self) -> Any:
        return self.expr

    def simplify(self) -> "ObservableExpr":
        return ObservableExpr(_simplify_value(self.expr))

    def trace(self) -> "ObservableExpr":
        return ObservableExpr(sp.simplify(_trace_value(self.expr)))

    def integrate(self, variable: sp.Symbol, limits: Sequence[Any] = (-sp.oo, sp.oo), prefactor: Any = 1) -> "ObservableExpr":
        lower, upper = _integral_limits(limits)
        return ObservableExpr(sp.simplify(prefactor) * sp.Integral(_unwrap(self.expr), (variable, lower, upper)))

    def real(self) -> "ObservableExpr":
        return ObservableExpr(sp.re(_unwrap(self.expr)))

    def imag(self) -> "ObservableExpr":
        return ObservableExpr(sp.im(_unwrap(self.expr)))

    def subs(self, *args, **kwargs) -> "ObservableExpr":
        target = _unwrap(self.expr)
        return ObservableExpr(target.subs(*args, **kwargs))

    def latex(self) -> str:
        return sp.latex(_unwrap(self.expr))

    def __add__(self, other: ObservableLike) -> "ObservableExpr":
        return ObservableExpr(_unwrap(self.expr) + _unwrap(other))

    def __radd__(self, other: ObservableLike) -> "ObservableExpr":
        return ObservableExpr(_unwrap(other) + _unwrap(self.expr))

    def __sub__(self, other: ObservableLike) -> "ObservableExpr":
        return ObservableExpr(_unwrap(self.expr) - _unwrap(other))

    def __rsub__(self, other: ObservableLike) -> "ObservableExpr":
        return ObservableExpr(_unwrap(other) - _unwrap(self.expr))

    def __mul__(self, other: ObservableLike) -> "ObservableExpr":
        return ObservableExpr(_unwrap(self.expr) * _unwrap(other))

    def __rmul__(self, other: ObservableLike) -> "ObservableExpr":
        return ObservableExpr(_unwrap(other) * _unwrap(self.expr))

    def __matmul__(self, other: ObservableLike) -> "ObservableExpr":
        return ObservableExpr(_unwrap(self.expr) * _unwrap(other))

    def __rmatmul__(self, other: ObservableLike) -> "ObservableExpr":
        return ObservableExpr(_unwrap(other) * _unwrap(self.expr))

    def __truediv__(self, other: ObservableLike) -> "ObservableExpr":
        return ObservableExpr(_unwrap(self.expr) / _unwrap(other))

    def __neg__(self) -> "ObservableExpr":
        return ObservableExpr(-_unwrap(self.expr))

    def __repr__(self) -> str:
        return f"ObservableExpr({self.expr!r})"


def obs(value: ObservableLike) -> ObservableExpr:
    return value if isinstance(value, ObservableExpr) else ObservableExpr(value)


def trace_expr(value: ObservableLike) -> ObservableExpr:
    return obs(value).trace()


def omega_integral(value: ObservableLike, omega: sp.Symbol, limits: Sequence[Any] = (-sp.oo, sp.oo), prefactor: Any = 1) -> ObservableExpr:
    return obs(value).integrate(omega, limits=limits, prefactor=prefactor)


def real_part(value: ObservableLike) -> ObservableExpr:
    return obs(value).real()


def imag_part(value: ObservableLike) -> ObservableExpr:
    return obs(value).imag()


def contour_green_observable(g_greater: ObservableLike, g_lesser: ObservableLike, left_time: Any, right_time: Any) -> ObservableExpr:
    from .keldysh_symbolic import contour_green_from_lesser_greater

    return ObservableExpr(contour_green_from_lesser_greater(_unwrap(g_greater), _unwrap(g_lesser), left_time, right_time))


def langreth_observable(left: dict[str, ObservableLike], right: dict[str, ObservableLike]) -> dict[str, ObservableExpr]:
    from .keldysh_symbolic import langreth_convolution

    result = langreth_convolution(
        {key: _unwrap(value) for key, value in left.items()},
        {key: _unwrap(value) for key, value in right.items()},
    )
    return {key: ObservableExpr(value) for key, value in result.items()}


def langreth_double_observable(
    first: dict[str, ObservableLike],
    second: dict[str, ObservableLike],
    third: dict[str, ObservableLike],
) -> dict[str, ObservableExpr]:
    from .keldysh_symbolic import langreth_double_convolution

    result = langreth_double_convolution(
        {key: _unwrap(value) for key, value in first.items()},
        {key: _unwrap(value) for key, value in second.items()},
        {key: _unwrap(value) for key, value in third.items()},
    )
    return {key: ObservableExpr(value) for key, value in result.items()}


def keldysh_dyson_lesser(g_ret: ObservableLike, sigma_lesser: ObservableLike, g_adv: ObservableLike) -> ObservableExpr:
    from .keldysh_symbolic import dyson_lesser_stationary

    return ObservableExpr(dyson_lesser_stationary(_unwrap(g_ret), _unwrap(sigma_lesser), _unwrap(g_adv)))


def keldysh_population(g_lesser: ObservableLike, omega: sp.Symbol, limits: Sequence[Any] = (-sp.oo, sp.oo)) -> ObservableExpr:
    from .keldysh_symbolic import stationary_population

    return ObservableExpr(stationary_population(_unwrap(g_lesser), omega, limits=tuple(limits)))


def keldysh_meir_wingreen_current(
    sigma_retarded_eta: ObservableLike,
    g_lesser: ObservableLike,
    sigma_lesser_eta: ObservableLike,
    g_advanced: ObservableLike,
    omega: sp.Symbol,
    *,
    charge: Any = sp.Symbol("e"),
    hbar: Any = sp.Symbol("hbar"),
    limits: Sequence[Any] = (-sp.oo, sp.oo),
) -> ObservableExpr:
    from .keldysh_symbolic import meir_wingreen_current_symbolic

    return ObservableExpr(
        meir_wingreen_current_symbolic(
            _unwrap(sigma_retarded_eta),
            _unwrap(g_lesser),
            _unwrap(sigma_lesser_eta),
            _unwrap(g_advanced),
            omega,
            charge=charge,
            hbar=hbar,
            limits=tuple(limits),
        )
    )


def keldysh_two_terminal_wide_band_current(
    omega: sp.Symbol,
    density: ObservableLike,
    gamma_left: ObservableLike,
    gamma_right: ObservableLike,
    f_left: ObservableLike,
    f_right: ObservableLike,
    *,
    charge: Any = sp.Symbol("e"),
    h: Any = sp.Symbol("h"),
    limits: Sequence[Any] = (-sp.oo, sp.oo),
) -> ObservableExpr:
    from .keldysh_symbolic import two_terminal_wide_band_current_symbolic

    return ObservableExpr(
        two_terminal_wide_band_current_symbolic(
            omega,
            _unwrap(density),
            _unwrap(gamma_left),
            _unwrap(gamma_right),
            _unwrap(f_left),
            _unwrap(f_right),
            charge=charge,
            h=h,
            limits=tuple(limits),
        )
    )


def fermi_window(omega: sp.Expr, mu_left: sp.Expr, mu_right: sp.Expr, temperature: Any = 0) -> ObservableExpr:
    return ObservableExpr(_fermi_symbolic(omega, mu_left, temperature) - _fermi_symbolic(omega, mu_right, temperature))


def transmission(g_ret: ObservableLike, g_adv: ObservableLike, gamma_left: ObservableLike, gamma_right: ObservableLike) -> ObservableExpr:
    expr = _unwrap(gamma_left) * _unwrap(g_ret) * _unwrap(gamma_right) * _unwrap(g_adv)
    return ObservableExpr(sp.simplify(_trace_value(expr)))


def conductance(
    transmission_expr: ObservableLike,
    *,
    charge: Any = 1,
    omega: sp.Symbol | None = None,
    mu: sp.Expr | None = None,
    temperature: Any = 0,
    limits: Sequence[Any] = (-sp.oo, sp.oo),
) -> ObservableExpr:
    prefactor = sp.simplify(charge**2 / (2 * sp.pi))
    expr = _unwrap(transmission_expr)
    if omega is None or mu is None:
        return ObservableExpr(prefactor * expr)
    if temperature == 0:
        return ObservableExpr(sp.simplify(prefactor * expr.subs(omega, mu)))
    kernel = _fermi_derivative_symbolic(omega, mu, temperature)
    return ObservableExpr(prefactor * sp.Integral(kernel * expr, *_integral_limits(limits) and ((omega, *_integral_limits(limits)),)))


def spin_projector(spin: str, basis_labels: Sequence[Any]) -> sp.Matrix:
    mask = [1 if _spin_mask(label, spin) else 0 for label in basis_labels]
    return sp.diag(*mask)


def spin_resolved_coupling(gamma: ObservableLike, spin: str, basis_labels: Sequence[Any]) -> ObservableExpr:
    projector = spin_projector(spin, basis_labels)
    gamma_value = _unwrap(gamma)
    return ObservableExpr(sp.simplify(projector * gamma_value * projector))


def spin_transmission(
    g_ret: ObservableLike,
    g_adv: ObservableLike,
    gamma_left: ObservableLike,
    gamma_right: ObservableLike,
    left_spin: str,
    right_spin: str,
    basis_labels: Sequence[Any],
) -> ObservableExpr:
    gamma_l_spin = _unwrap(spin_resolved_coupling(gamma_left, left_spin, basis_labels))
    gamma_r_spin = _unwrap(spin_resolved_coupling(gamma_right, right_spin, basis_labels))
    return transmission(g_ret, g_adv, gamma_l_spin, gamma_r_spin)


def spin_transmission_channels(
    g_ret: ObservableLike,
    g_adv: ObservableLike,
    gamma_left: ObservableLike,
    gamma_right: ObservableLike,
    basis_labels: Sequence[Any],
    spins: Sequence[str] = ("up", "down"),
) -> dict[tuple[str, str], ObservableExpr]:
    out: dict[tuple[str, str], ObservableExpr] = {}
    for left_spin in spins:
        for right_spin in spins:
            out[(left_spin, right_spin)] = spin_transmission(
                g_ret,
                g_adv,
                gamma_left,
                gamma_right,
                left_spin,
                right_spin,
                basis_labels,
            )
    return out


def spin_conductance(
    g_ret: ObservableLike,
    g_adv: ObservableLike,
    gamma_left: ObservableLike,
    gamma_right: ObservableLike,
    basis_labels: Sequence[Any],
    *,
    charge: Any = 1,
    omega: sp.Symbol | None = None,
    mu: sp.Expr | None = None,
    temperature: Any = 0,
    limits: Sequence[Any] = (-sp.oo, sp.oo),
) -> ObservableExpr:
    channels = spin_transmission_channels(g_ret, g_adv, gamma_left, gamma_right, basis_labels)
    polarized = channels[("up", "up")] + channels[("up", "down")] - channels[("down", "up")] - channels[("down", "down")]
    return conductance(polarized, charge=charge, omega=omega, mu=mu, temperature=temperature, limits=limits)


def landauer_integrand(
    g_ret: ObservableLike,
    g_adv: ObservableLike,
    gamma_left: ObservableLike,
    gamma_right: ObservableLike,
    omega: sp.Expr,
    mu_left: sp.Expr,
    mu_right: sp.Expr,
    *,
    temperature: Any = 0,
    charge: Any = 1,
) -> ObservableExpr:
    prefactor = sp.simplify(charge / (2 * sp.pi))
    return ObservableExpr(prefactor * _unwrap(fermi_window(omega, mu_left, mu_right, temperature)) * _unwrap(transmission(g_ret, g_adv, gamma_left, gamma_right)))


def landauer_current(
    g_ret: ObservableLike,
    g_adv: ObservableLike,
    gamma_left: ObservableLike,
    gamma_right: ObservableLike,
    omega: sp.Symbol,
    mu_left: sp.Expr,
    mu_right: sp.Expr,
    *,
    temperature: Any = 0,
    charge: Any = 1,
    limits: Sequence[Any] = (-sp.oo, sp.oo),
) -> ObservableExpr:
    integrand = landauer_integrand(g_ret, g_adv, gamma_left, gamma_right, omega, mu_left, mu_right, temperature=temperature, charge=charge)
    return integrand.integrate(omega, limits=limits)


def evaluate_omega_integral(values: np.ndarray, omega_grid: np.ndarray, prefactor: complex | float = 1.0) -> complex:
    return prefactor * np.trapezoid(np.asarray(values), np.asarray(omega_grid))


def conductance_numeric(
    transmission_values: np.ndarray,
    omega_grid: np.ndarray,
    *,
    mu: float = 0.0,
    temperature: float = 0.0,
    charge: float = 1.0,
) -> float:
    transmission_values = np.asarray(transmission_values, dtype=np.complex128)
    omega_grid = np.asarray(omega_grid, dtype=float)
    prefactor = charge**2 / (2.0 * np.pi)
    if temperature <= 0.0:
        idx = int(np.argmin(np.abs(omega_grid - mu)))
        return float(np.real(prefactor * transmission_values[idx]))
    from .greens import fermi_dirac

    f = fermi_dirac(omega_grid, mu=mu, temperature=temperature)
    kernel = np.real(f * (1.0 - f) / temperature)
    return float(np.real(prefactor * np.trapezoid(kernel * transmission_values, omega_grid)))


def landauer_current_numeric(
    transmission_values: np.ndarray,
    omega_grid: np.ndarray,
    mu_left: float,
    mu_right: float,
    *,
    temperature: float = 0.0,
    charge: float = 1.0,
) -> float:
    from .greens import fermi_dirac

    omega_grid = np.asarray(omega_grid, dtype=float)
    transmission_values = np.asarray(transmission_values, dtype=np.complex128)
    window = fermi_dirac(omega_grid, mu=mu_left, temperature=temperature) - fermi_dirac(omega_grid, mu=mu_right, temperature=temperature)
    integrand = (charge / (2.0 * np.pi)) * window * transmission_values
    return float(np.real(np.trapezoid(integrand, omega_grid)))
