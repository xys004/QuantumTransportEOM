"""Symbolic Keldysh formalism helpers.

This module implements a small formal layer following the class-note
conventions used in the project: closed Keldysh contour branches, greater
and lesser components, retarded/advanced relations, Langreth rules, Dyson
equations, and stationary current/occupation expressions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import sympy as sp


FORWARD_BRANCH = "-"
BACKWARD_BRANCH = "+"


def _branch_key(branch: str) -> str:
    key = str(branch).strip().lower()
    aliases = {
        "-": FORWARD_BRANCH,
        "+": BACKWARD_BRANCH,
        "minus": FORWARD_BRANCH,
        "forward": FORWARD_BRANCH,
        "ck-": FORWARD_BRANCH,
        "c-": FORWARD_BRANCH,
        "plus": BACKWARD_BRANCH,
        "backward": BACKWARD_BRANCH,
        "ck+": BACKWARD_BRANCH,
        "c+": BACKWARD_BRANCH,
    }
    if key not in aliases:
        raise ValueError("branch must be '-'/'forward' or '+'/'backward'.")
    return aliases[key]


@dataclass(frozen=True)
class KeldyshTime:
    """Time point on the closed Keldysh contour."""

    time: Any
    branch: str = FORWARD_BRANCH

    def __post_init__(self) -> None:
        object.__setattr__(self, "branch", _branch_key(self.branch))


def kt(time: Any, branch: str = FORWARD_BRANCH) -> KeldyshTime:
    """Create a contour time point with a short, readable constructor."""

    return KeldyshTime(time, branch)


def keldysh_leq(left: KeldyshTime, right: KeldyshTime) -> sp.Expr:
    """Return the contour-order relation left <=_K right.

    For symbolic real times on the same branch, this returns a relational
    expression. Cross-branch ordering follows the closed contour convention:
    all forward-branch times precede all backward-branch times.
    """

    left_branch = _branch_key(left.branch)
    right_branch = _branch_key(right.branch)
    if left_branch == FORWARD_BRANCH and right_branch == FORWARD_BRANCH:
        return sp.Le(left.time, right.time)
    if left_branch == FORWARD_BRANCH and right_branch == BACKWARD_BRANCH:
        return sp.S.true
    if left_branch == BACKWARD_BRANCH and right_branch == FORWARD_BRANCH:
        return sp.S.false
    return sp.Ge(left.time, right.time)


def contour_heaviside(left: KeldyshTime, right: KeldyshTime) -> sp.Expr:
    """Two-argument Heaviside function on the Keldysh contour."""

    left_branch = _branch_key(left.branch)
    right_branch = _branch_key(right.branch)
    if left_branch == FORWARD_BRANCH and right_branch == FORWARD_BRANCH:
        return sp.Heaviside(left.time - right.time)
    if left_branch == FORWARD_BRANCH and right_branch == BACKWARD_BRANCH:
        return sp.Integer(0)
    if left_branch == BACKWARD_BRANCH and right_branch == FORWARD_BRANCH:
        return sp.Integer(1)
    return sp.Heaviside(right.time - left.time)


def contour_delta(left: KeldyshTime, right: KeldyshTime) -> sp.Expr:
    """Two-argument Dirac delta induced by the contour Heaviside."""

    left_branch = _branch_key(left.branch)
    right_branch = _branch_key(right.branch)
    if left_branch == FORWARD_BRANCH and right_branch == FORWARD_BRANCH:
        return sp.DiracDelta(left.time - right.time)
    if left_branch == BACKWARD_BRANCH and right_branch == BACKWARD_BRANCH:
        return -sp.DiracDelta(left.time - right.time)
    return sp.Integer(0)


@dataclass(frozen=True)
class KeldyshFunction:
    """Named symbolic Keldysh Green function or self-energy."""

    name: str

    def component(self, component: str, *args: Any) -> sp.Expr:
        key = _component_key(component)
        symbol = _component_label(self.name, key)
        return sp.Function(symbol)(*args)

    def greater(self, *args: Any) -> sp.Expr:
        return self.component("greater", *args)

    def lesser(self, *args: Any) -> sp.Expr:
        return self.component("lesser", *args)

    def retarded(self, *args: Any) -> sp.Expr:
        return self.component("retarded", *args)

    def advanced(self, *args: Any) -> sp.Expr:
        return self.component("advanced", *args)

    def keldysh(self, *args: Any) -> sp.Expr:
        return self.component("keldysh", *args)


def kgf(name: str) -> KeldyshFunction:
    """Create a symbolic Keldysh object."""

    return KeldyshFunction(name)


@dataclass(frozen=True)
class KeldyshExpression:
    """Small object wrapper for symbolic Keldysh expressions."""

    expr: Any

    def doit(self) -> sp.Expr:
        return sp.sympify(self.expr)

    def simplify(self) -> "KeldyshExpression":
        return KeldyshExpression(sp.simplify(self.doit()))

    def obs(self):
        from .observables import ObservableExpr

        return ObservableExpr(self.doit())

    def latex(self) -> str:
        return sp.latex(self.doit())

    def subs(self, *args: Any, **kwargs: Any) -> "KeldyshExpression":
        return KeldyshExpression(self.doit().subs(*args, **kwargs))

    def __add__(self, other: Any) -> "KeldyshExpression":
        return KeldyshExpression(self.doit() + _unwrap_kexpr(other))

    def __radd__(self, other: Any) -> "KeldyshExpression":
        return KeldyshExpression(_unwrap_kexpr(other) + self.doit())

    def __sub__(self, other: Any) -> "KeldyshExpression":
        return KeldyshExpression(self.doit() - _unwrap_kexpr(other))

    def __rsub__(self, other: Any) -> "KeldyshExpression":
        return KeldyshExpression(_unwrap_kexpr(other) - self.doit())

    def __mul__(self, other: Any) -> "KeldyshExpression":
        return KeldyshExpression(self.doit() * _unwrap_kexpr(other))

    def __rmul__(self, other: Any) -> "KeldyshExpression":
        return KeldyshExpression(_unwrap_kexpr(other) * self.doit())

    def __truediv__(self, other: Any) -> "KeldyshExpression":
        return KeldyshExpression(self.doit() / _unwrap_kexpr(other))


def _unwrap_kexpr(value: Any) -> sp.Expr:
    if isinstance(value, KeldyshExpression):
        return value.doit()
    if hasattr(value, "expr"):
        return sp.sympify(value.expr)
    return sp.sympify(value)


@dataclass(frozen=True)
class KeldyshObject:
    """QuTiP/SNEG-like facade for a named Keldysh object."""

    function: KeldyshFunction
    system: "KeldyshSystem | None" = None

    @property
    def name(self) -> str:
        return self.function.name

    def component(self, component: str, *args: Any) -> KeldyshExpression:
        return KeldyshExpression(self.function.component(component, *args))

    def greater(self, *args: Any) -> KeldyshExpression:
        return self.component(">", *args)

    def lesser(self, *args: Any) -> KeldyshExpression:
        return self.component("<", *args)

    def retarded(self, *args: Any) -> KeldyshExpression:
        return self.component("r", *args)

    def advanced(self, *args: Any) -> KeldyshExpression:
        return self.component("a", *args)

    def keldysh(self, *args: Any) -> KeldyshExpression:
        return self.component("k", *args)

    def contour(self, left: KeldyshTime, right: KeldyshTime) -> KeldyshExpression:
        return KeldyshExpression(contour_green_from_lesser_greater(self.function.greater(left.time, right.time), self.function.lesser(left.time, right.time), left, right))

    def retarded_from_pm(self, t: Any, tp: Any) -> KeldyshExpression:
        return KeldyshExpression(retarded_from_lesser_greater(self.function.greater(t, tp), self.function.lesser(t, tp), t, tp))

    def advanced_from_pm(self, t: Any, tp: Any) -> KeldyshExpression:
        return KeldyshExpression(advanced_from_lesser_greater(self.function.greater(t, tp), self.function.lesser(t, tp), t, tp))

    def dyson_retarded(self, omega: Any, xi: Any, sigma_retarded: Any) -> KeldyshExpression:
        return KeldyshExpression(dyson_retarded_from_level(omega, xi, _unwrap_kexpr(sigma_retarded)))

    def dyson_lesser(self, g_retarded: Any, sigma_lesser: Any, g_advanced: Any) -> KeldyshExpression:
        return KeldyshExpression(dyson_lesser_stationary(_unwrap_kexpr(g_retarded), _unwrap_kexpr(sigma_lesser), _unwrap_kexpr(g_advanced)))


@dataclass(frozen=True)
class KeldyshSystem:
    """High-level symbolic workspace for Keldysh calculations."""

    omega: Any = sp.Symbol("omega", real=True)

    def green(self, name: str = "G") -> KeldyshObject:
        return KeldyshObject(kgf(name), system=self)

    def self_energy(self, name: str = "Sigma") -> KeldyshObject:
        return KeldyshObject(kgf(name), system=self)

    def time(self, value: Any, branch: str = FORWARD_BRANCH) -> KeldyshTime:
        return kt(value, branch)

    def langreth(self, left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, KeldyshExpression]:
        result = langreth_convolution(
            {key: _unwrap_kexpr(value) for key, value in left.items()},
            {key: _unwrap_kexpr(value) for key, value in right.items()},
        )
        return {key: KeldyshExpression(value) for key, value in result.items()}

    def langreth3(self, first: Mapping[str, Any], second: Mapping[str, Any], third: Mapping[str, Any]) -> dict[str, KeldyshExpression]:
        result = langreth_double_convolution(
            {key: _unwrap_kexpr(value) for key, value in first.items()},
            {key: _unwrap_kexpr(value) for key, value in second.items()},
            {key: _unwrap_kexpr(value) for key, value in third.items()},
        )
        return {key: KeldyshExpression(value) for key, value in result.items()}

    def dyson_retarded(self, xi: Any, sigma_retarded: Any) -> KeldyshExpression:
        return KeldyshExpression(dyson_retarded_from_level(self.omega, xi, _unwrap_kexpr(sigma_retarded)))

    def dyson_lesser(self, g_retarded: Any, sigma_lesser: Any, g_advanced: Any) -> KeldyshExpression:
        return KeldyshExpression(dyson_lesser_stationary(_unwrap_kexpr(g_retarded), _unwrap_kexpr(sigma_lesser), _unwrap_kexpr(g_advanced)))

    def population(self, g_lesser: Any, limits: tuple[Any, Any] = (-sp.oo, sp.oo)):
        from .observables import ObservableExpr

        return ObservableExpr(stationary_population(_unwrap_kexpr(g_lesser), self.omega, limits=limits))

    def meir_wingreen_current(
        self,
        sigma_retarded_eta: Any,
        g_lesser: Any,
        sigma_lesser_eta: Any,
        g_advanced: Any,
        *,
        charge: Any = sp.Symbol("e"),
        hbar: Any = sp.Symbol("hbar"),
        limits: tuple[Any, Any] = (-sp.oo, sp.oo),
    ):
        from .observables import ObservableExpr

        return ObservableExpr(
            meir_wingreen_current_symbolic(
                _unwrap_kexpr(sigma_retarded_eta),
                _unwrap_kexpr(g_lesser),
                _unwrap_kexpr(sigma_lesser_eta),
                _unwrap_kexpr(g_advanced),
                self.omega,
                charge=charge,
                hbar=hbar,
                limits=limits,
            )
        )

    def wide_band_current(
        self,
        density: Any,
        gamma_left: Any,
        gamma_right: Any,
        f_left: Any,
        f_right: Any,
        *,
        charge: Any = sp.Symbol("e"),
        h: Any = sp.Symbol("h"),
        limits: tuple[Any, Any] = (-sp.oo, sp.oo),
    ):
        from .observables import ObservableExpr

        return ObservableExpr(
            two_terminal_wide_band_current_symbolic(
                self.omega,
                _unwrap_kexpr(density),
                _unwrap_kexpr(gamma_left),
                _unwrap_kexpr(gamma_right),
                _unwrap_kexpr(f_left),
                _unwrap_kexpr(f_right),
                charge=charge,
                h=h,
                limits=limits,
            )
        )


def keldysh_system(omega: Any = sp.Symbol("omega", real=True)) -> KeldyshSystem:
    return KeldyshSystem(omega=omega)


def _component_key(component: str) -> str:
    key = str(component).strip().lower()
    aliases = {
        ">": "greater",
        "greater": "greater",
        "gtr": "greater",
        "<": "lesser",
        "lesser": "lesser",
        "less": "lesser",
        "r": "retarded",
        "retarded": "retarded",
        "a": "advanced",
        "advanced": "advanced",
        "k": "keldysh",
        "keldysh": "keldysh",
    }
    if key not in aliases:
        raise ValueError(f"Unsupported Keldysh component: {component!r}")
    return aliases[key]


def _component_label(name: str, component: str) -> str:
    labels = {
        "greater": f"{name}^>",
        "lesser": f"{name}^<",
        "retarded": f"{name}^r",
        "advanced": f"{name}^a",
        "keldysh": f"{name}^K",
    }
    return labels[component]


def contour_green_from_lesser_greater(
    g_greater: Any,
    g_lesser: Any,
    left: KeldyshTime,
    right: KeldyshTime,
) -> sp.Expr:
    """Contour Green function G = Theta_C G^> + Theta_C' G^<."""

    if left == right:
        return sp.sympify(g_lesser)
    theta_lr = contour_heaviside(left, right)
    theta_rl = contour_heaviside(right, left)
    return sp.simplify(theta_lr * g_greater + theta_rl * g_lesser)


def retarded_from_lesser_greater(g_greater: Any, g_lesser: Any, t: Any, tp: Any) -> sp.Expr:
    return sp.simplify(sp.Heaviside(t - tp) * (sp.sympify(g_greater) - sp.sympify(g_lesser)))


def advanced_from_lesser_greater(g_greater: Any, g_lesser: Any, t: Any, tp: Any) -> sp.Expr:
    return sp.simplify(-sp.Heaviside(tp - t) * (sp.sympify(g_greater) - sp.sympify(g_lesser)))


def keldysh_component_from_lesser_greater(g_greater: Any, g_lesser: Any) -> sp.Expr:
    return sp.simplify(sp.sympify(g_greater) + sp.sympify(g_lesser))


def langreth_convolution(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, sp.Expr]:
    """Langreth rules for C = A * B in stationary/frequency notation."""

    a_r, a_a = sp.sympify(left["r"]), sp.sympify(left["a"])
    b_r, b_a = sp.sympify(right["r"]), sp.sympify(right["a"])
    a_l, a_g = sp.sympify(left["<"]), sp.sympify(left[">"])
    b_l, b_g = sp.sympify(right["<"]), sp.sympify(right[">"])
    return {
        "r": sp.simplify(a_r * b_r),
        "a": sp.simplify(a_a * b_a),
        "<": sp.simplify(a_r * b_l + a_l * b_a),
        ">": sp.simplify(a_r * b_g + a_g * b_a),
    }


def langreth_double_convolution(first: Mapping[str, Any], second: Mapping[str, Any], third: Mapping[str, Any]) -> dict[str, sp.Expr]:
    """Langreth rules for Z = W * X * Y."""

    w_r, w_a = sp.sympify(first["r"]), sp.sympify(first["a"])
    x_r, x_a = sp.sympify(second["r"]), sp.sympify(second["a"])
    y_r, y_a = sp.sympify(third["r"]), sp.sympify(third["a"])
    return {
        "r": sp.simplify(w_r * x_r * y_r),
        "a": sp.simplify(w_a * x_a * y_a),
        "<": sp.simplify(w_r * x_r * third["<"] + w_r * second["<"] * y_a + first["<"] * x_a * y_a),
        ">": sp.simplify(w_r * x_r * third[">"] + w_r * second[">"] * y_a + first[">"] * x_a * y_a),
    }


def dyson_retarded(g_retarded: Any, sigma_retarded: Any) -> sp.Expr:
    """Scalar stationary Dyson solution G^r = g^r/(1 - g^r Sigma^r)."""

    g_r = sp.sympify(g_retarded)
    sigma_r = sp.sympify(sigma_retarded)
    return sp.simplify(g_r / (1 - g_r * sigma_r))


def dyson_retarded_from_level(omega: Any, xi: Any, sigma_retarded: Any) -> sp.Expr:
    """Stationary level result G^r = 1/(omega - xi - Sigma^r)."""

    return sp.simplify(1 / (sp.sympify(omega) - sp.sympify(xi) - sp.sympify(sigma_retarded)))


def dyson_lesser_stationary(g_retarded: Any, sigma_lesser: Any, g_advanced: Any) -> sp.Expr:
    """Remote-time stationary result G^< = G^r Sigma^< G^a."""

    return sp.simplify(sp.sympify(g_retarded) * sp.sympify(sigma_lesser) * sp.sympify(g_advanced))


def spectral_density_from_retarded(g_retarded: Any) -> sp.Expr:
    """rho(omega) = -Im G^r / pi."""

    return sp.simplify(-sp.im(sp.sympify(g_retarded)) / sp.pi)


def lorentzian_density(omega: Any, xi: Any, gamma: Any, re_sigma: Any = 0) -> sp.Expr:
    """Wide-band style density used in the notes."""

    omega, xi, gamma, re_sigma = map(sp.sympify, (omega, xi, gamma, re_sigma))
    return sp.simplify((gamma / sp.pi) / ((omega - xi - re_sigma) ** 2 + gamma**2))


def stationary_population(g_lesser: Any, omega: Any, limits: tuple[Any, Any] = (-sp.oo, sp.oo)) -> sp.Expr:
    """<n> = int d omega G^< /(2 pi i)."""

    return sp.Integral(sp.sympify(g_lesser) / (2 * sp.pi * sp.I), (omega, limits[0], limits[1]))


def meir_wingreen_integrand(sigma_retarded_eta: Any, g_lesser: Any, sigma_lesser_eta: Any, g_advanced: Any) -> sp.Expr:
    """Frequency-domain integrand Re[Sigma_eta^r G^< + Sigma_eta^< G^a]."""

    return sp.re(sp.sympify(sigma_retarded_eta) * sp.sympify(g_lesser) + sp.sympify(sigma_lesser_eta) * sp.sympify(g_advanced))


def meir_wingreen_current_symbolic(
    sigma_retarded_eta: Any,
    g_lesser: Any,
    sigma_lesser_eta: Any,
    g_advanced: Any,
    omega: Any,
    *,
    charge: Any = sp.Symbol("e"),
    hbar: Any = sp.Symbol("hbar"),
    limits: tuple[Any, Any] = (-sp.oo, sp.oo),
) -> sp.Expr:
    """Symbolic current I_eta = 2e/hbar int d omega Re[...]."""

    integrand = meir_wingreen_integrand(sigma_retarded_eta, g_lesser, sigma_lesser_eta, g_advanced)
    return sp.Integral(2 * charge / hbar * integrand, (omega, limits[0], limits[1]))


def two_terminal_wide_band_current_symbolic(
    omega: Any,
    density: Any,
    gamma_left: Any,
    gamma_right: Any,
    f_left: Any,
    f_right: Any,
    *,
    charge: Any = sp.Symbol("e"),
    h: Any = sp.Symbol("h"),
    limits: tuple[Any, Any] = (-sp.oo, sp.oo),
) -> sp.Expr:
    """Two-reservoir wide-band current from the symmetric expression."""

    prefactor = 4 * sp.pi * charge / h
    gamma_l, gamma_r = sp.sympify(gamma_left), sp.sympify(gamma_right)
    gamma_total = gamma_l + gamma_r
    integrand = prefactor * gamma_l * gamma_r / gamma_total * sp.sympify(density) * (sp.sympify(f_left) - sp.sympify(f_right))
    return sp.Integral(sp.simplify(integrand), (omega, limits[0], limits[1]))


def latex_keldysh(expr: Any) -> str:
    return sp.latex(expr)
