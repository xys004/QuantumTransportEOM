"""High-level ergonomic API inspired by QuTiP-style usage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import sympy as sp

from .bosonic import destroy_b, num_b
from .devices import LeadSelfEnergy, MatrixDevice, MatrixTransportView, spin_axis_projector_numeric
from .greens import bose_einstein, fermi_dirac
from .keldysh import KeldyshSelfEnergy
from .models import (
    SymbolicModel,
    anderson_impurity_model,
    bosonic_harmonic_mode_model,
    custom_model,
    fermionic_single_level_model,
    single_particle_hamiltonian_matrix,
)
from .numerics import (
    batched_current_spectral_density,
    batched_keldysh_component,
    batched_retarded_green,
    batched_transmission,
    blocked_over_grid,
    gamma_from_sigma_stack,
    get_backend,
    sigma_stack,
    to_numpy,
)
from .observables import conductance as observable_conductance, landauer_current as observable_landauer_current, transmission as observable_transmission
from .secondquant import destroy, num


OperatorLike = Any


def f(index: Any) -> OperatorLike:
    """Fermionic annihilation operator."""
    return destroy(index)


def fd(index: Any) -> OperatorLike:
    """Fermionic creation operator."""
    return destroy(index).dag()


def b(index: Any) -> OperatorLike:
    """Bosonic annihilation operator."""
    return destroy_b(index)


def bd(index: Any) -> OperatorLike:
    """Bosonic creation operator."""
    return destroy_b(index).dag()


def n(index: Any, statistics: str = "fermion") -> OperatorLike:
    """Number operator with a short, QuTiP-like name."""
    key = statistics.lower()
    if key in {"fermion", "fermionic", "f"}:
        return num(index)
    if key in {"boson", "bosonic", "b"}:
        return num_b(index)
    raise ValueError(f"Unsupported statistics for n(...): {statistics}")


def _normalize_method(method: str | None) -> str | None:
    if method is None:
        return None
    key = method.lower().replace("-", "_")
    aliases = {
        "hf": "hartree",
        "hartree_fock": "hartree",
        "mean_field": "hartree",
        "collinear_hartree": "hartree",
        "hubbard1": "hubbard_i",
        "atomic": "hubbard_i",
    }
    return aliases.get(key, key)


def _normalize_substitutions(expr: sp.Expr, substitutions: Mapping[Any, Any] | None) -> dict[sp.Expr, Any]:
    if substitutions is None:
        return {}
    normalized: dict[sp.Expr, Any] = {}
    free_symbols = {symbol.name: symbol for symbol in expr.free_symbols}
    for key, value in substitutions.items():
        if isinstance(key, sp.Basic):
            normalized[key] = value
        elif isinstance(key, str):
            if key not in free_symbols:
                raise ValueError(f"Could not match substitution key '{key}' to a free symbol in expression {expr}.")
            normalized[free_symbols[key]] = value
        else:
            raise TypeError(f"Unsupported substitution key type: {type(key)!r}")
    return normalized


def _distribution_symbolic(
    statistics: str,
    omega: sp.Expr,
    *,
    distribution: sp.Expr | None = None,
    mu: sp.Expr | None = None,
    temperature: sp.Expr | None = None,
) -> sp.Expr:
    if distribution is not None:
        return distribution
    mu_expr = sp.Integer(0) if mu is None else mu
    if temperature is None:
        return sp.Function("f_FD" if statistics == "fermion" else "n_BE")(omega)
    if statistics == "fermion":
        return sp.simplify(1 / (sp.exp((omega - mu_expr) / temperature) + 1))
    return sp.simplify(1 / (sp.exp((omega - mu_expr) / temperature) - 1))


def _distribution_values(statistics: str, omega_grid: np.ndarray, *, mu: float = 0.0, temperature: float = 0.0) -> np.ndarray:
    if statistics == "fermion":
        return fermi_dirac(omega_grid, mu=mu, temperature=temperature)
    if statistics == "boson":
        return bose_einstein(omega_grid, mu=mu, temperature=temperature)
    raise ValueError(f"Unsupported statistics for distribution values: {statistics}")


@dataclass
class OccupationResult:
    occupations: dict[str, float]
    converged: bool
    iterations: int
    max_delta: float


def _to_numeric_parameter(value: Any, *, name: str) -> complex:
    try:
        return complex(value)
    except Exception as exc:  # pragma: no cover - defensive
        raise ValueError(f"{name} must be numeric for open-transport calculations; got {value!r}.") from exc


def _coerce_spinful_lead(
    lead: LeadSelfEnergy | Mapping[str, Any] | Any,
    *,
    mu: float = 0.0,
    temperature: float = 0.0,
    name: str,
) -> LeadSelfEnergy:
    if isinstance(lead, LeadSelfEnergy):
        if lead.dim != 2:
            raise ValueError(f"{name} must have dimension 2 for an Anderson impurity, got {lead.dim}.")
        return lead
    if isinstance(lead, Mapping):
        if "up" not in lead or "down" not in lead:
            raise ValueError(f"{name} mapping must contain 'up' and 'down' entries.")
        gamma = np.diag(
            [
                _to_numeric_parameter(lead["up"], name=f"{name}['up']"),
                _to_numeric_parameter(lead["down"], name=f"{name}['down']"),
            ]
        ).astype(np.complex128)
        return LeadSelfEnergy.wide_band(gamma, mu=mu, temperature=temperature, name=name)
    gamma_value = _to_numeric_parameter(lead, name=name)
    return LeadSelfEnergy.wide_band(np.diag([gamma_value, gamma_value]), mu=mu, temperature=temperature, name=name)


@dataclass
class EOMBasisView:
    model: SymbolicModel

    def expand(self, levels: int = 1) -> list[sp.Expr]:
        return self.model.expand_basis(max_steps=levels)

    def analyze(
        self,
        *,
        levels: int = 0,
        method: str | None = None,
        occupations: Mapping[str, Any] | None = None,
    ):
        normalized = _normalize_method(method)
        params = {"occupations": occupations} if occupations is not None else None
        return self.model.analyze_eom(
            auto_expand_steps=levels,
            truncation=normalized,
            truncation_params=params,
        )


@dataclass
class GreenFunctionView:
    model: SymbolicModel
    channel: Any = None

    def _statistics(self) -> str:
        return self.model.statistics

    def _resolve_pair(self) -> tuple[Any, Any]:
        operators = self.model.operators
        if self.model.name == "anderson_impurity":
            if self.channel not in {"up", "down"}:
                raise ValueError("AndersonImpurity.gf(...) expects 'up' or 'down'.")
            return operators[f"d_{self.channel}"], operators[f"d_{self.channel}_dag"]
        if self.model.name == "fermionic_single_level":
            return operators["d"], operators["d_dag"]
        if self.model.name == "bosonic_harmonic_mode":
            return operators["b"], operators["b_dag"]
        if self.channel in operators:
            key = str(self.channel)
            dag_key = f"{key}_dag"
            if dag_key in operators:
                return operators[key], operators[dag_key]
        raise ValueError(f"Could not resolve a Green-function channel for model '{self.model.name}'.")

    def retarded(
        self,
        *,
        omega: sp.Expr,
        eta: sp.Expr,
        method: str | None = None,
        occupations: Mapping[str, Any] | None = None,
        levels: int = 0,
    ) -> sp.Expr:
        left, right = self._resolve_pair()
        normalized = _normalize_method(method)
        params = {"occupations": occupations} if occupations is not None else None
        return self.model.retarded(
            left,
            right,
            omega=omega,
            eta=eta,
            auto_expand_steps=levels,
            truncation=normalized,
            truncation_params=params,
        )

    def _advanced_expression(
        self,
        *,
        omega: sp.Expr,
        eta: sp.Expr,
        method: str | None,
        occupations: Mapping[str, Any] | None,
        levels: int,
    ) -> sp.Expr:
        """Build ``G^a`` from the resolvent instead of substituting into ``G^r``.

        ``G^r.subs(eta, -eta)`` rewrites *every* occurrence of that value, so a
        Hamiltonian coefficient numerically equal to ``eta`` has its sign
        flipped as well and the result is silently wrong.  Evaluating the same
        resolvent at ``-eta`` touches only the regulator.
        """

        return self.retarded(
            omega=omega,
            eta=-eta,
            method=method,
            occupations=occupations,
            levels=levels,
        )

    def advanced(
        self,
        *,
        omega: sp.Expr,
        eta: sp.Expr,
        method: str | None = None,
        occupations: Mapping[str, Any] | None = None,
        levels: int = 0,
    ) -> sp.Expr:
        return sp.simplify(
            self._advanced_expression(
                omega=omega,
                eta=eta,
                method=method,
                occupations=occupations,
                levels=levels,
            )
        )

    def lesser(
        self,
        *,
        omega: sp.Expr,
        eta: sp.Expr,
        method: str | None = None,
        occupations: Mapping[str, Any] | None = None,
        levels: int = 0,
        distribution: sp.Expr | None = None,
        mu: sp.Expr | None = None,
        temperature: sp.Expr | None = None,
    ) -> sp.Expr:
        g_ret = self.retarded(
            omega=omega,
            eta=eta,
            method=method,
            occupations=occupations,
            levels=levels,
        )
        g_adv = sp.simplify(
            self._advanced_expression(
                omega=omega,
                eta=eta,
                method=method,
                occupations=occupations,
                levels=levels,
            )
        )
        dist = _distribution_symbolic(self._statistics(), omega, distribution=distribution, mu=mu, temperature=temperature)
        return sp.simplify(dist * (g_adv - g_ret))

    def greater(
        self,
        *,
        omega: sp.Expr,
        eta: sp.Expr,
        method: str | None = None,
        occupations: Mapping[str, Any] | None = None,
        levels: int = 0,
        distribution: sp.Expr | None = None,
        mu: sp.Expr | None = None,
        temperature: sp.Expr | None = None,
    ) -> sp.Expr:
        g_ret = self.retarded(
            omega=omega,
            eta=eta,
            method=method,
            occupations=occupations,
            levels=levels,
        )
        g_adv = sp.simplify(
            self._advanced_expression(
                omega=omega,
                eta=eta,
                method=method,
                occupations=occupations,
                levels=levels,
            )
        )
        dist = _distribution_symbolic(self._statistics(), omega, distribution=distribution, mu=mu, temperature=temperature)
        factor = dist - 1 if self._statistics() == "fermion" else dist + 1
        return sp.simplify(factor * (g_adv - g_ret))

    def spectral_function(
        self,
        *,
        omega: sp.Expr,
        eta: sp.Expr,
        method: str | None = None,
        occupations: Mapping[str, Any] | None = None,
        levels: int = 0,
    ) -> sp.Expr:
        g_ret = self.retarded(
            omega=omega,
            eta=eta,
            method=method,
            occupations=occupations,
            levels=levels,
        )
        g_adv = sp.simplify(
            self._advanced_expression(
                omega=omega,
                eta=eta,
                method=method,
                occupations=occupations,
                levels=levels,
            )
        )
        return sp.simplify(sp.I * (g_ret - g_adv))

    def spectral_density(
        self,
        *,
        omega: sp.Expr,
        eta: sp.Expr,
        method: str | None = None,
        occupations: Mapping[str, Any] | None = None,
        levels: int = 0,
    ) -> sp.Expr:
        return sp.simplify(
            self.spectral_function(
                omega=omega,
                eta=eta,
                method=method,
                occupations=occupations,
                levels=levels,
            )
            / (2 * sp.pi)
        )

    def _numeric_expression_values(
        self,
        expr: sp.Expr,
        *,
        omega_symbol: sp.Symbol,
        omega_grid: np.ndarray,
        parameters: Mapping[Any, Any] | None = None,
    ) -> np.ndarray:
        substitutions = {}
        substitutions.update(_normalize_substitutions(expr, parameters))
        expr_num = sp.simplify(expr.subs(substitutions))
        remaining = sorted(expr_num.free_symbols - {omega_symbol}, key=lambda symbol: symbol.sort_key())
        if remaining:
            raise ValueError(f"Expression still has free symbols after substitutions: {remaining}")
        func = sp.lambdify(omega_symbol, expr_num, "numpy")
        return np.asarray(func(omega_grid), dtype=np.complex128)

    def _numeric_retarded_values(
        self,
        *,
        omega_symbol: sp.Symbol,
        omega_grid: np.ndarray,
        eta: float,
        method: str | None = None,
        occupations: Mapping[str, Any] | None = None,
        levels: int = 0,
        parameters: Mapping[Any, Any] | None = None,
    ) -> np.ndarray:
        eta_symbol = sp.Symbol("eta", positive=True, real=True)
        expr = self.retarded(
            omega=omega_symbol,
            eta=eta_symbol,
            method=method,
            occupations=occupations,
            levels=levels,
        )
        substitutions = {eta_symbol: float(eta)}
        substitutions.update(_normalize_substitutions(expr, parameters))
        expr_num = sp.simplify(expr.subs(substitutions))
        remaining = sorted(expr_num.free_symbols - {omega_symbol}, key=lambda symbol: symbol.sort_key())
        if remaining:
            raise ValueError(f"Expression still has free symbols after substitutions: {remaining}")
        func = sp.lambdify(omega_symbol, expr_num, "numpy")
        values = np.asarray(func(omega_grid), dtype=np.complex128)
        return values

    def _numeric_advanced_values(
        self,
        *,
        omega_symbol: sp.Symbol,
        omega_grid: np.ndarray,
        eta: float,
        method: str | None = None,
        occupations: Mapping[str, Any] | None = None,
        levels: int = 0,
        parameters: Mapping[Any, Any] | None = None,
    ) -> np.ndarray:
        eta_symbol = sp.Symbol("eta", positive=True, real=True)
        expr = self.advanced(
            omega=omega_symbol,
            eta=eta_symbol,
            method=method,
            occupations=occupations,
            levels=levels,
        )
        substitutions = {eta_symbol: float(eta)}
        substitutions.update(_normalize_substitutions(expr, parameters))
        expr_num = sp.simplify(expr.subs(substitutions))
        remaining = sorted(expr_num.free_symbols - {omega_symbol}, key=lambda symbol: symbol.sort_key())
        if remaining:
            raise ValueError(f"Expression still has free symbols after substitutions: {remaining}")
        func = sp.lambdify(omega_symbol, expr_num, "numpy")
        values = np.asarray(func(omega_grid), dtype=np.complex128)
        return values

    def lesser_values(
        self,
        *,
        omega_symbol: sp.Symbol,
        omega_grid: np.ndarray,
        eta: float,
        method: str | None = None,
        occupations: Mapping[str, Any] | None = None,
        levels: int = 0,
        parameters: Mapping[Any, Any] | None = None,
        mu: float = 0.0,
        temperature: float = 0.0,
    ) -> np.ndarray:
        g_ret = self._numeric_retarded_values(
            omega_symbol=omega_symbol,
            omega_grid=omega_grid,
            eta=eta,
            method=method,
            occupations=occupations,
            levels=levels,
            parameters=parameters,
        )
        g_adv = self._numeric_advanced_values(
            omega_symbol=omega_symbol,
            omega_grid=omega_grid,
            eta=eta,
            method=method,
            occupations=occupations,
            levels=levels,
            parameters=parameters,
        )
        dist = _distribution_values(self._statistics(), omega_grid, mu=mu, temperature=temperature)
        return dist * (g_adv - g_ret)

    def greater_values(
        self,
        *,
        omega_symbol: sp.Symbol,
        omega_grid: np.ndarray,
        eta: float,
        method: str | None = None,
        occupations: Mapping[str, Any] | None = None,
        levels: int = 0,
        parameters: Mapping[Any, Any] | None = None,
        mu: float = 0.0,
        temperature: float = 0.0,
    ) -> np.ndarray:
        g_ret = self._numeric_retarded_values(
            omega_symbol=omega_symbol,
            omega_grid=omega_grid,
            eta=eta,
            method=method,
            occupations=occupations,
            levels=levels,
            parameters=parameters,
        )
        g_adv = self._numeric_advanced_values(
            omega_symbol=omega_symbol,
            omega_grid=omega_grid,
            eta=eta,
            method=method,
            occupations=occupations,
            levels=levels,
            parameters=parameters,
        )
        dist = _distribution_values(self._statistics(), omega_grid, mu=mu, temperature=temperature)
        factor = dist - 1 if self._statistics() == "fermion" else dist + 1
        return factor * (g_adv - g_ret)

    def spectral_density_values(
        self,
        *,
        omega_symbol: sp.Symbol,
        omega_grid: np.ndarray,
        eta: float,
        method: str | None = None,
        occupations: Mapping[str, Any] | None = None,
        levels: int = 0,
        parameters: Mapping[Any, Any] | None = None,
    ) -> np.ndarray:
        values = self._numeric_retarded_values(
            omega_symbol=omega_symbol,
            omega_grid=omega_grid,
            eta=eta,
            method=method,
            occupations=occupations,
            levels=levels,
            parameters=parameters,
        )
        return -np.imag(values) / np.pi

    def spectral_function_values(
        self,
        *,
        omega_symbol: sp.Symbol,
        omega_grid: np.ndarray,
        eta: float,
        method: str | None = None,
        occupations: Mapping[str, Any] | None = None,
        levels: int = 0,
        parameters: Mapping[Any, Any] | None = None,
    ) -> np.ndarray:
        return 2.0 * np.pi * self.spectral_density_values(
            omega_symbol=omega_symbol,
            omega_grid=omega_grid,
            eta=eta,
            method=method,
            occupations=occupations,
            levels=levels,
            parameters=parameters,
        )

    def spectral_values(
        self,
        *,
        omega_symbol: sp.Symbol,
        omega_grid: np.ndarray,
        eta: float,
        method: str | None = None,
        occupations: Mapping[str, Any] | None = None,
        levels: int = 0,
        parameters: Mapping[Any, Any] | None = None,
    ) -> np.ndarray:
        return self.spectral_density_values(
            omega_symbol=omega_symbol,
            omega_grid=omega_grid,
            eta=eta,
            method=method,
            occupations=occupations,
            levels=levels,
            parameters=parameters,
        )

    def occupation(
        self,
        *,
        omega_symbol: sp.Symbol,
        omega_grid: np.ndarray,
        eta: float,
        method: str | None = None,
        occupations: Mapping[str, Any] | None = None,
        levels: int = 0,
        parameters: Mapping[Any, Any] | None = None,
        mu: float = 0.0,
        temperature: float = 0.0,
    ) -> float:
        spectral = self.spectral_values(
            omega_symbol=omega_symbol,
            omega_grid=omega_grid,
            eta=eta,
            method=method,
            occupations=occupations,
            levels=levels,
            parameters=parameters,
        )
        filling = _distribution_values(self._statistics(), omega_grid, mu=mu, temperature=temperature)
        return float(np.trapezoid(spectral * filling, omega_grid).real)

    def latex(
        self,
        *,
        omega: sp.Expr,
        eta: sp.Expr,
        method: str | None = None,
        occupations: Mapping[str, Any] | None = None,
        levels: int = 0,
    ) -> str:
        return sp.latex(
            self.retarded(
                omega=omega,
                eta=eta,
                method=method,
                occupations=occupations,
                levels=levels,
            )
        )


@dataclass
class TransportView:
    model: SymbolicModel
    gamma_left: Any
    gamma_right: Any

    def _gamma_for_channel(self, gamma: Any, channel: Any) -> Any:
        if isinstance(gamma, Mapping):
            if channel not in gamma:
                raise ValueError(f"Missing coupling for channel {channel!r}.")
            return gamma[channel]
        return gamma

    def transmission(
        self,
        *,
        omega: sp.Expr,
        eta: sp.Expr,
        channel: Any = None,
        method: str | None = None,
        occupations: Mapping[str, Any] | None = None,
        levels: int = 0,
    ) -> sp.Expr:
        gf_view = GreenFunctionView(self.model, channel=channel)
        g_ret = gf_view.retarded(omega=omega, eta=eta, method=method, occupations=occupations, levels=levels)
        g_adv = gf_view.advanced(omega=omega, eta=eta, method=method, occupations=occupations, levels=levels)
        gamma_l = self._gamma_for_channel(self.gamma_left, channel)
        gamma_r = self._gamma_for_channel(self.gamma_right, channel)
        return observable_transmission(g_ret, g_adv, gamma_l, gamma_r).doit()

    def conductance(
        self,
        *,
        omega: sp.Symbol,
        eta: sp.Expr,
        channel: Any = None,
        method: str | None = None,
        occupations: Mapping[str, Any] | None = None,
        levels: int = 0,
        mu: sp.Expr | None = None,
        temperature: Any = 0,
        charge: Any = 1,
    ) -> sp.Expr:
        t_expr = self.transmission(omega=omega, eta=eta, channel=channel, method=method, occupations=occupations, levels=levels)
        return observable_conductance(t_expr, charge=charge, omega=omega, mu=mu, temperature=temperature).doit()

    def landauer_current(
        self,
        *,
        omega: sp.Symbol,
        eta: sp.Expr,
        mu_left: sp.Expr,
        mu_right: sp.Expr,
        channel: Any = None,
        method: str | None = None,
        occupations: Mapping[str, Any] | None = None,
        levels: int = 0,
        temperature: Any = 0,
        charge: Any = 1,
        limits: Sequence[Any] = (-sp.oo, sp.oo),
    ) -> sp.Expr:
        gf_view = GreenFunctionView(self.model, channel=channel)
        g_ret = gf_view.retarded(omega=omega, eta=eta, method=method, occupations=occupations, levels=levels)
        g_adv = gf_view.advanced(omega=omega, eta=eta, method=method, occupations=occupations, levels=levels)
        gamma_l = self._gamma_for_channel(self.gamma_left, channel)
        gamma_r = self._gamma_for_channel(self.gamma_right, channel)
        return observable_landauer_current(
            g_ret,
            g_adv,
            gamma_l,
            gamma_r,
            omega,
            mu_left,
            mu_right,
            temperature=temperature,
            charge=charge,
            limits=limits,
        ).doit()


@dataclass
class OpenAndersonGreenFunctionView:
    transport: "OpenAndersonTransportView"
    channel: str | None = None

    def _component(self, matrix: np.ndarray) -> complex | np.ndarray:
        if self.channel is None:
            return matrix
        index = self.transport._channel_index(self.channel)
        return complex(matrix[index, index])

    def retarded(self, omega: float, *, eta: float = 0.0, method: str = "hubbard_i", occupations: Mapping[str, float] | None = None):
        return self._component(self.transport.retarded(omega, eta=eta, method=method, occupations=occupations))

    def advanced(self, omega: float, *, eta: float = 0.0, method: str = "hubbard_i", occupations: Mapping[str, float] | None = None):
        return self._component(self.transport.advanced(omega, eta=eta, method=method, occupations=occupations))

    def lesser(self, omega: float, *, eta: float = 0.0, method: str = "hubbard_i", occupations: Mapping[str, float] | None = None):
        return self._component(self.transport.lesser(omega, eta=eta, method=method, occupations=occupations))

    def greater(self, omega: float, *, eta: float = 0.0, method: str = "hubbard_i", occupations: Mapping[str, float] | None = None):
        return self._component(self.transport.greater(omega, eta=eta, method=method, occupations=occupations))

    def spectral_function(self, omega: float, *, eta: float = 0.0, method: str = "hubbard_i", occupations: Mapping[str, float] | None = None):
        return self._component(self.transport.spectral_function(omega, eta=eta, method=method, occupations=occupations))

    def spectral_density(self, omega: float, *, eta: float = 0.0, method: str = "hubbard_i", occupations: Mapping[str, float] | None = None):
        return self._component(self.transport.spectral_density(omega, eta=eta, method=method, occupations=occupations))

    def occupation(
        self,
        omega_grid: np.ndarray,
        *,
        eta: float = 0.0,
        method: str = "hubbard_i",
        occupations: Mapping[str, float] | None = None,
    ) -> float:
        if self.channel is None:
            raise ValueError("Specify a spin channel to compute an occupation.")
        return self.transport.channel_occupation(
            self.channel,
            omega_grid,
            eta=eta,
            method=method,
            occupations=occupations,
        )


@dataclass
class OpenAndersonTransportView:
    model: "AndersonImpurity"
    left_lead: LeadSelfEnergy
    right_lead: LeadSelfEnergy
    extra_self_energies: tuple[KeldyshSelfEnergy, ...] = ()

    basis_labels: tuple[str, str] = ("up", "down")

    def __post_init__(self) -> None:
        if self.left_lead.dim != 2 or self.right_lead.dim != 2:
            raise ValueError("OpenAndersonTransportView requires two-dimensional leads.")
        for self_energy in self.extra_self_energies:
            if self_energy.dim != 2:
                raise ValueError("Extra Anderson self-energies must be two-dimensional.")

    @property
    def dim(self) -> int:
        return 2

    def gf(self, channel: str | None = None) -> OpenAndersonGreenFunctionView:
        return OpenAndersonGreenFunctionView(self, channel=channel)

    def with_self_energy(self, *self_energies: KeldyshSelfEnergy) -> "OpenAndersonTransportView":
        return OpenAndersonTransportView(
            self.model,
            self.left_lead,
            self.right_lead,
            extra_self_energies=self.extra_self_energies + tuple(self_energies),
            basis_labels=self.basis_labels,
        )

    def _channel_index(self, channel: str) -> int:
        key = str(channel).lower()
        if key == "up":
            return 0
        if key == "down":
            return 1
        raise ValueError("channel must be 'up' or 'down'.")

    def _lead(self, lead: str) -> LeadSelfEnergy:
        if lead == "left":
            return self.left_lead
        if lead == "right":
            return self.right_lead
        raise ValueError("lead must be 'left' or 'right'.")

    def _parameters(self) -> tuple[complex, complex, complex]:
        metadata = self.model.model.metadata
        return (
            _to_numeric_parameter(metadata["epsilon_up"], name="epsilon_up"),
            _to_numeric_parameter(metadata["epsilon_down"], name="epsilon_down"),
            _to_numeric_parameter(metadata["interaction_u"], name="interaction_u"),
        )

    def _spin_flip(self) -> complex:
        return _to_numeric_parameter(self.model.model.metadata.get("spin_flip", 0.0), name="spin_flip")

    def local_hamiltonian(self) -> np.ndarray:
        epsilon_up, epsilon_down, _interaction_u = self._parameters()
        spin_flip = self._spin_flip()
        return np.array(
            [
                [epsilon_up, spin_flip],
                [np.conjugate(spin_flip), epsilon_down],
            ],
            dtype=np.complex128,
        )

    def _normalized_method(self, method: str | None) -> str:
        normalized = _normalize_method(method)
        return "hubbard_i" if normalized is None else normalized

    def _occupation_map(self, occupations: Mapping[str, float] | None) -> dict[str, float]:
        if occupations is None:
            return {"up": 0.5, "down": 0.5}
        return {
            "up": float(occupations.get("up", 0.5)),
            "down": float(occupations.get("down", 0.5)),
        }

    def _sigma_extra(self, omega: float, component: str) -> np.ndarray:
        if not self.extra_self_energies:
            return np.zeros((2, 2), dtype=np.complex128)
        if component == "retarded":
            return sum((sigma.sigma_retarded(omega) for sigma in self.extra_self_energies), np.zeros((2, 2), dtype=np.complex128))
        if component == "lesser":
            return sum((sigma.sigma_lesser(omega) for sigma in self.extra_self_energies), np.zeros((2, 2), dtype=np.complex128))
        if component == "greater":
            return sum((sigma.sigma_greater(omega) for sigma in self.extra_self_energies), np.zeros((2, 2), dtype=np.complex128))
        raise ValueError("component must be 'retarded', 'lesser', or 'greater'.")

    def sigma_retarded(self, omega: float, lead: str | None = None) -> np.ndarray:
        if lead == "left":
            return self.left_lead.sigma_retarded(omega)
        if lead == "right":
            return self.right_lead.sigma_retarded(omega)
        if lead == "extra":
            return self._sigma_extra(omega, "retarded")
        if lead is None or lead == "total":
            return self.left_lead.sigma_retarded(omega) + self.right_lead.sigma_retarded(omega) + self._sigma_extra(omega, "retarded")
        raise ValueError("lead must be 'left', 'right', 'extra', or None/'total'.")

    def sigma_advanced(self, omega: float, lead: str | None = None) -> np.ndarray:
        return self.sigma_retarded(omega, lead=lead).conj().T

    def sigma_lesser(self, omega: float, lead: str | None = None) -> np.ndarray:
        if lead == "left":
            return self.left_lead.sigma_lesser(omega)
        if lead == "right":
            return self.right_lead.sigma_lesser(omega)
        if lead == "extra":
            return self._sigma_extra(omega, "lesser")
        if lead is None or lead == "total":
            return self.left_lead.sigma_lesser(omega) + self.right_lead.sigma_lesser(omega) + self._sigma_extra(omega, "lesser")
        raise ValueError("lead must be 'left', 'right', 'extra', or None/'total'.")

    def sigma_greater(self, omega: float, lead: str | None = None) -> np.ndarray:
        if lead == "left":
            return self.left_lead.sigma_greater(omega)
        if lead == "right":
            return self.right_lead.sigma_greater(omega)
        if lead == "extra":
            return self._sigma_extra(omega, "greater")
        if lead is None or lead == "total":
            return self.left_lead.sigma_greater(omega) + self.right_lead.sigma_greater(omega) + self._sigma_extra(omega, "greater")
        raise ValueError("lead must be 'left', 'right', 'extra', or None/'total'.")

    def gamma(self, omega: float, lead: str | None = None) -> np.ndarray:
        sigma_r = self.sigma_retarded(omega, lead=lead)
        return 1j * (sigma_r - sigma_r.conj().T)

    def interaction_self_energy(self, omega: float, *, eta: float = 0.0, method: str = "hubbard_i", occupations: Mapping[str, float] | None = None) -> np.ndarray:
        epsilon_up, epsilon_down, interaction_u = self._parameters()
        sigma_env = self.sigma_retarded(omega, lead="total")
        sigma_diag = np.diag(sigma_env)
        occ = self._occupation_map(occupations)
        z = complex(omega, eta)
        normalized = self._normalized_method(method)

        if normalized in {"noninteracting", "free"}:
            return np.zeros((2, 2), dtype=np.complex128)
        if normalized == "hartree":
            return np.diag(
                [
                    interaction_u * occ["down"],
                    interaction_u * occ["up"],
                ]
            ).astype(np.complex128)
        if normalized == "hubbard_i":
            n_down = occ["down"]
            n_up = occ["up"]
            sigma_up = interaction_u * n_down
            sigma_down = interaction_u * n_up
            sigma_up += interaction_u**2 * n_down * (1.0 - n_down) / (z - epsilon_up - sigma_diag[0] - interaction_u * (1.0 - n_down))
            sigma_down += interaction_u**2 * n_up * (1.0 - n_up) / (z - epsilon_down - sigma_diag[1] - interaction_u * (1.0 - n_up))
            return np.diag([sigma_up, sigma_down]).astype(np.complex128)
        raise ValueError(f"Unsupported open Anderson method: {method}")

    def retarded(self, omega: float, *, eta: float = 0.0, method: str = "hubbard_i", occupations: Mapping[str, float] | None = None) -> np.ndarray:
        z = complex(omega, eta)
        bare = z * np.eye(2, dtype=np.complex128) - self.local_hamiltonian()
        sigma_env = self.sigma_retarded(omega, lead="total")
        sigma_int = self.interaction_self_energy(omega, eta=eta, method=method, occupations=occupations)
        return np.linalg.inv(bare - sigma_env - sigma_int)

    def advanced(self, omega: float, *, eta: float = 0.0, method: str = "hubbard_i", occupations: Mapping[str, float] | None = None) -> np.ndarray:
        return self.retarded(omega, eta=eta, method=method, occupations=occupations).conj().T

    def lesser(self, omega: float, *, eta: float = 0.0, method: str = "hubbard_i", occupations: Mapping[str, float] | None = None) -> np.ndarray:
        g_r = self.retarded(omega, eta=eta, method=method, occupations=occupations)
        g_a = g_r.conj().T
        return g_r @ self.sigma_lesser(omega, lead="total") @ g_a

    def greater(self, omega: float, *, eta: float = 0.0, method: str = "hubbard_i", occupations: Mapping[str, float] | None = None) -> np.ndarray:
        g_r = self.retarded(omega, eta=eta, method=method, occupations=occupations)
        g_a = g_r.conj().T
        return g_r @ self.sigma_greater(omega, lead="total") @ g_a

    def keldysh(self, omega: float, *, eta: float = 0.0, method: str = "hubbard_i", occupations: Mapping[str, float] | None = None) -> np.ndarray:
        return self.lesser(omega, eta=eta, method=method, occupations=occupations) + self.greater(omega, eta=eta, method=method, occupations=occupations)

    def spectral_function(self, omega: float, *, eta: float = 0.0, method: str = "hubbard_i", occupations: Mapping[str, float] | None = None) -> np.ndarray:
        g_r = self.retarded(omega, eta=eta, method=method, occupations=occupations)
        g_a = g_r.conj().T
        return 1j * (g_r - g_a)

    def spectral_density(self, omega: float, *, eta: float = 0.0, method: str = "hubbard_i", occupations: Mapping[str, float] | None = None) -> np.ndarray:
        return self.spectral_function(omega, eta=eta, method=method, occupations=occupations) / (2.0 * np.pi)

    def _interaction_self_energy_diag_values(
        self,
        omega_grid: np.ndarray,
        sigma_env_diag: np.ndarray,
        *,
        eta: float,
        method: str,
        occupations: Mapping[str, float] | None,
    ) -> np.ndarray:
        """Vectorized interaction self-energy diagonal, shape (n, 2)."""
        epsilon_up, epsilon_down, interaction_u = self._parameters()
        occ = self._occupation_map(occupations)
        z = omega_grid + 1j * eta
        normalized = self._normalized_method(method)
        result = np.zeros((omega_grid.size, 2), dtype=np.complex128)
        if normalized in {"noninteracting", "free"}:
            return result
        if normalized == "hartree":
            result[:, 0] = interaction_u * occ["down"]
            result[:, 1] = interaction_u * occ["up"]
            return result
        if normalized == "hubbard_i":
            n_down = occ["down"]
            n_up = occ["up"]
            result[:, 0] = interaction_u * n_down + interaction_u**2 * n_down * (1.0 - n_down) / (
                z - epsilon_up - sigma_env_diag[:, 0] - interaction_u * (1.0 - n_down)
            )
            result[:, 1] = interaction_u * n_up + interaction_u**2 * n_up * (1.0 - n_up) / (
                z - epsilon_down - sigma_env_diag[:, 1] - interaction_u * (1.0 - n_up)
            )
            return result
        raise ValueError(f"Unsupported open Anderson method: {method}")

    def retarded_values(
        self,
        omega_grid: np.ndarray,
        *,
        eta: float = 0.0,
        method: str = "hubbard_i",
        occupations: Mapping[str, float] | None = None,
        backend: Any = None,
        workers: int | None = None,
    ) -> np.ndarray:
        """Retarded Green function on a frequency grid via batched inversions, shape (n, 2, 2)."""
        xp = get_backend(backend)

        def compute(subgrid: np.ndarray) -> np.ndarray:
            sigma_env = sigma_stack(lambda omega: self.sigma_retarded(omega, lead="total"), subgrid)
            sigma_int_diag = self._interaction_self_energy_diag_values(
                subgrid,
                sigma_env[:, (0, 1), (0, 1)],
                eta=eta,
                method=method,
                occupations=occupations,
            )
            sigma_total = sigma_env.copy()
            sigma_total[:, 0, 0] += sigma_int_diag[:, 0]
            sigma_total[:, 1, 1] += sigma_int_diag[:, 1]
            g_r = batched_retarded_green(self.local_hamiltonian(), xp.asarray(sigma_total), subgrid, eta=eta, xp=xp)
            return to_numpy(g_r)

        return blocked_over_grid(compute, omega_grid, self.dim, workers=workers)

    def lesser_values(
        self,
        omega_grid: np.ndarray,
        *,
        eta: float = 0.0,
        method: str = "hubbard_i",
        occupations: Mapping[str, float] | None = None,
        backend: Any = None,
        workers: int | None = None,
    ) -> np.ndarray:
        xp = get_backend(backend)

        def compute(subgrid: np.ndarray) -> np.ndarray:
            g_r = self.retarded_values(subgrid, eta=eta, method=method, occupations=occupations, backend=backend)
            sigma_less = sigma_stack(lambda omega: self.sigma_lesser(omega, lead="total"), subgrid)
            return to_numpy(batched_keldysh_component(xp.asarray(g_r), xp.asarray(sigma_less), xp=xp))

        return blocked_over_grid(compute, omega_grid, self.dim, workers=workers)

    def greater_values(
        self,
        omega_grid: np.ndarray,
        *,
        eta: float = 0.0,
        method: str = "hubbard_i",
        occupations: Mapping[str, float] | None = None,
        backend: Any = None,
        workers: int | None = None,
    ) -> np.ndarray:
        xp = get_backend(backend)

        def compute(subgrid: np.ndarray) -> np.ndarray:
            g_r = self.retarded_values(subgrid, eta=eta, method=method, occupations=occupations, backend=backend)
            sigma_great = sigma_stack(lambda omega: self.sigma_greater(omega, lead="total"), subgrid)
            return to_numpy(batched_keldysh_component(xp.asarray(g_r), xp.asarray(sigma_great), xp=xp))

        return blocked_over_grid(compute, omega_grid, self.dim, workers=workers)

    def transmission_values(
        self,
        omega_grid: np.ndarray,
        *,
        eta: float = 0.0,
        method: str = "hubbard_i",
        occupations: Mapping[str, float] | None = None,
        backend: Any = None,
        workers: int | None = None,
    ) -> np.ndarray:
        xp = get_backend(backend)

        def compute(subgrid: np.ndarray) -> np.ndarray:
            g_r = self.retarded_values(subgrid, eta=eta, method=method, occupations=occupations, backend=backend)
            gamma_l = gamma_from_sigma_stack(sigma_stack(self.left_lead.sigma_retarded, subgrid))
            gamma_r = gamma_from_sigma_stack(sigma_stack(self.right_lead.sigma_retarded, subgrid))
            values = batched_transmission(xp.asarray(g_r), xp.asarray(gamma_l), xp.asarray(gamma_r), xp=xp)
            return to_numpy(values).astype(float)

        return blocked_over_grid(compute, omega_grid, self.dim, workers=workers)

    def transmission(self, omega: float, *, eta: float = 0.0, method: str = "hubbard_i", occupations: Mapping[str, float] | None = None) -> float:
        g_r = self.retarded(omega, eta=eta, method=method, occupations=occupations)
        g_a = g_r.conj().T
        gamma_l = self.gamma(omega, lead="left")
        gamma_r = self.gamma(omega, lead="right")
        return float(np.real(np.trace(gamma_l @ g_r @ gamma_r @ g_a)))

    def spin_resolved_transmission(self, omega: float, component: str = "+", *, axis: str = "z", eta: float = 0.0, method: str = "hubbard_i", occupations: Mapping[str, float] | None = None) -> float:
        projector = spin_axis_projector_numeric(self.basis_labels, axis=axis, component=component)
        g_r = self.retarded(omega, eta=eta, method=method, occupations=occupations)
        g_a = g_r.conj().T
        gamma_l = self.gamma(omega, lead="left")
        gamma_r = projector @ self.gamma(omega, lead="right") @ projector
        return float(np.real(np.trace(gamma_l @ g_r @ gamma_r @ g_a)))

    def conductance(
        self,
        *,
        mu: float = 0.0,
        temperature: float = 0.0,
        charge: float = 1.0,
        eta: float = 0.0,
        omega_grid: np.ndarray | None = None,
        method: str = "hubbard_i",
        occupations: Mapping[str, float] | None = None,
    ) -> float:
        prefactor = charge**2 / (2.0 * np.pi)
        if temperature <= 0.0:
            return prefactor * self.transmission(mu, eta=eta, method=method, occupations=occupations)
        if omega_grid is None:
            raise ValueError("omega_grid is required for finite-temperature conductance.")
        omega_grid = np.asarray(omega_grid, dtype=float)
        f = fermi_dirac(omega_grid, mu=mu, temperature=temperature)
        kernel = np.real(f * (1.0 - f) / temperature)
        transmission_values = self.transmission_values(omega_grid, eta=eta, method=method, occupations=occupations)
        return float(prefactor * np.trapezoid(kernel * transmission_values, omega_grid))

    def meir_wingreen_current_density(
        self,
        omega: float,
        *,
        lead: str = "left",
        charge: float = 1.0,
        eta: float = 0.0,
        method: str = "hubbard_i",
        occupations: Mapping[str, float] | None = None,
    ) -> float:
        sigma_l = self.sigma_lesser(omega, lead=lead)
        sigma_g = self.sigma_greater(omega, lead=lead)
        g_l = self.lesser(omega, eta=eta, method=method, occupations=occupations)
        g_g = self.greater(omega, eta=eta, method=method, occupations=occupations)
        integrand = (charge / (2.0 * np.pi)) * np.trace(sigma_l @ g_g - sigma_g @ g_l)
        return float(np.real(integrand))

    def meir_wingreen_current(
        self,
        omega_grid: np.ndarray,
        *,
        lead: str = "left",
        charge: float = 1.0,
        eta: float = 0.0,
        method: str = "hubbard_i",
        occupations: Mapping[str, float] | None = None,
        backend: Any = None,
        workers: int | None = None,
    ) -> float:
        xp = get_backend(backend)
        grid = np.asarray(omega_grid, dtype=float)
        g_lesser = self.lesser_values(grid, eta=eta, method=method, occupations=occupations, backend=backend, workers=workers)
        g_greater = self.greater_values(grid, eta=eta, method=method, occupations=occupations, backend=backend, workers=workers)
        selected = self._lead(lead) if lead in {"left", "right"} else None
        if selected is not None:
            sigma_less = sigma_stack(selected.sigma_lesser, grid, workers=workers)
            sigma_great = sigma_stack(selected.sigma_greater, grid, workers=workers)
        else:
            sigma_less = sigma_stack(lambda omega: self.sigma_lesser(omega, lead=lead), grid, workers=workers)
            sigma_great = sigma_stack(lambda omega: self.sigma_greater(omega, lead=lead), grid, workers=workers)
        values = batched_current_spectral_density(
            xp.asarray(g_lesser),
            xp.asarray(g_greater),
            xp.asarray(sigma_less),
            xp.asarray(sigma_great),
            charge=charge,
            xp=xp,
        )
        return float(np.trapezoid(to_numpy(values), grid))

    def spin_resolved_meir_wingreen_current_density(
        self,
        omega: float,
        component: str = "+",
        *,
        lead: str = "left",
        axis: str = "z",
        charge: float = 1.0,
        eta: float = 0.0,
        method: str = "hubbard_i",
        occupations: Mapping[str, float] | None = None,
    ) -> float:
        projector = spin_axis_projector_numeric(self.basis_labels, axis=axis, component=component)
        sigma_l = projector @ self.sigma_lesser(omega, lead=lead) @ projector
        sigma_g = projector @ self.sigma_greater(omega, lead=lead) @ projector
        g_l = self.lesser(omega, eta=eta, method=method, occupations=occupations)
        g_g = self.greater(omega, eta=eta, method=method, occupations=occupations)
        integrand = (charge / (2.0 * np.pi)) * np.trace(sigma_l @ g_g - sigma_g @ g_l)
        return float(np.real(integrand))

    def spin_resolved_meir_wingreen_current(
        self,
        omega_grid: np.ndarray,
        component: str = "+",
        *,
        lead: str = "left",
        axis: str = "z",
        charge: float = 1.0,
        eta: float = 0.0,
        method: str = "hubbard_i",
        occupations: Mapping[str, float] | None = None,
    ) -> float:
        omega_grid = np.asarray(omega_grid, dtype=float)
        values = np.array(
            [
                self.spin_resolved_meir_wingreen_current_density(
                    float(omega),
                    component,
                    lead=lead,
                    axis=axis,
                    charge=charge,
                    eta=eta,
                    method=method,
                    occupations=occupations,
                )
                for omega in omega_grid
            ],
            dtype=float,
        )
        return float(np.trapezoid(values, omega_grid))

    def spin_meir_wingreen_current(
        self,
        omega_grid: np.ndarray,
        *,
        lead: str = "left",
        axis: str = "z",
        charge: float = 1.0,
        eta: float = 0.0,
        method: str = "hubbard_i",
        occupations: Mapping[str, float] | None = None,
    ) -> float:
        plus = self.spin_resolved_meir_wingreen_current(
            omega_grid,
            "+",
            lead=lead,
            axis=axis,
            charge=charge,
            eta=eta,
            method=method,
            occupations=occupations,
        )
        minus = self.spin_resolved_meir_wingreen_current(
            omega_grid,
            "-",
            lead=lead,
            axis=axis,
            charge=charge,
            eta=eta,
            method=method,
            occupations=occupations,
        )
        return float(plus - minus)

    def channel_occupation(
        self,
        channel: str,
        omega_grid: np.ndarray,
        *,
        eta: float = 0.0,
        method: str = "hubbard_i",
        occupations: Mapping[str, float] | None = None,
        backend: Any = None,
        workers: int | None = None,
    ) -> float:
        index = self._channel_index(channel)
        grid = np.asarray(omega_grid, dtype=float)
        lesser = self.lesser_values(grid, eta=eta, method=method, occupations=occupations, backend=backend, workers=workers)
        return float(np.trapezoid(lesser[:, index, index] / (2j * np.pi), grid).real)

    def self_consistent_occupations(
        self,
        omega_grid: np.ndarray,
        *,
        eta: float = 0.0,
        method: str = "hubbard_i",
        initial: Mapping[str, float] | None = None,
        mixing: float = 0.5,
        tol: float = 1e-8,
        max_iter: int = 200,
        backend: Any = None,
        workers: int | None = None,
    ) -> OccupationResult:
        current = {"up": 0.5, "down": 0.5}
        if initial is not None:
            current.update({key: float(value) for key, value in initial.items()})

        grid = np.asarray(omega_grid, dtype=float)
        max_delta = float("inf")
        for iteration in range(1, max_iter + 1):
            lesser = self.lesser_values(grid, eta=eta, method=method, occupations=current, backend=backend, workers=workers)
            raw_up = float(np.trapezoid(lesser[:, 0, 0] / (2j * np.pi), grid).real)
            raw_down = float(np.trapezoid(lesser[:, 1, 1] / (2j * np.pi), grid).real)
            updated = {
                "up": float((1.0 - mixing) * current["up"] + mixing * raw_up),
                "down": float((1.0 - mixing) * current["down"] + mixing * raw_down),
            }
            max_delta = max(abs(updated[spin] - current[spin]) for spin in current)
            current = updated
            if max_delta < tol:
                return OccupationResult(occupations=current, converged=True, iterations=iteration, max_delta=max_delta)
        return OccupationResult(occupations=current, converged=False, iterations=max_iter, max_delta=max_delta)


@dataclass
class SymbolicModelAPI:
    model: SymbolicModel

    @property
    def hamiltonian(self) -> Any:
        return self.model.hamiltonian

    @property
    def operators(self) -> dict[str, Any]:
        return self.model.operators

    def gf(self, channel: Any = None) -> GreenFunctionView:
        return GreenFunctionView(self.model, channel=channel)

    def eom_basis(self) -> EOMBasisView:
        return EOMBasisView(self.model)

    def transport(self, gamma_left: Any, gamma_right: Any) -> TransportView:
        return TransportView(self.model, gamma_left=gamma_left, gamma_right=gamma_right)

    def latex_hamiltonian(self) -> str:
        return self.model.latex_hamiltonian()

    def _repr_latex_(self) -> str:
        return f"$H = {self.model.latex_hamiltonian()}$"


class AndersonImpurity(SymbolicModelAPI):
    def __init__(
        self,
        *,
        eps: sp.Expr | None = None,
        U: sp.Expr,
        eps_up: sp.Expr | None = None,
        eps_down: sp.Expr | None = None,
        zeeman: sp.Expr = 0,
        spin_flip: sp.Expr = 0,
    ):
        epsilon_up = eps if eps_up is None else eps_up
        epsilon_down = eps if eps_down is None else eps_down
        if epsilon_up is None or epsilon_down is None:
            raise ValueError("Provide eps=... or both eps_up=... and eps_down=...")
        epsilon_up = sp.simplify(epsilon_up + zeeman / 2)
        epsilon_down = sp.simplify(epsilon_down - zeeman / 2)
        super().__init__(anderson_impurity_model(epsilon_up, epsilon_down, U, spin_flip=spin_flip))
        self.model.metadata["zeeman"] = zeeman
        self.model.metadata["spin_flip"] = spin_flip

    def open(
        self,
        left: LeadSelfEnergy | Mapping[str, Any] | Any,
        right: LeadSelfEnergy | Mapping[str, Any] | Any,
        *,
        mu_left: float = 0.0,
        mu_right: float = 0.0,
        temperature_left: float = 0.0,
        temperature_right: float = 0.0,
    ) -> OpenAndersonTransportView:
        left_lead = _coerce_spinful_lead(left, mu=mu_left, temperature=temperature_left, name="left")
        right_lead = _coerce_spinful_lead(right, mu=mu_right, temperature=temperature_right, name="right")
        return OpenAndersonTransportView(self, left_lead=left_lead, right_lead=right_lead)

    def open_system(
        self,
        left: LeadSelfEnergy | Mapping[str, Any] | Any,
        right: LeadSelfEnergy | Mapping[str, Any] | Any,
        *,
        mu_left: float = 0.0,
        mu_right: float = 0.0,
        temperature_left: float = 0.0,
        temperature_right: float = 0.0,
    ) -> OpenAndersonTransportView:
        return self.open(
            left,
            right,
            mu_left=mu_left,
            mu_right=mu_right,
            temperature_left=temperature_left,
            temperature_right=temperature_right,
        )

    def self_consistent_occupations(
        self,
        *,
        omega_symbol: sp.Symbol,
        omega_grid: np.ndarray,
        eta: float,
        method: str = "hubbard_i",
        parameters: Mapping[Any, Any] | None = None,
        mu: float = 0.0,
        temperature: float = 0.0,
        initial: Mapping[str, float] | None = None,
        mixing: float = 0.5,
        tol: float = 1e-8,
        max_iter: int = 200,
    ) -> OccupationResult:
        current = {"up": 0.5, "down": 0.5}
        if initial is not None:
            current.update({key: float(value) for key, value in initial.items()})

        max_delta = float("inf")
        for iteration in range(1, max_iter + 1):
            raw_up = self.gf("up").occupation(
                omega_symbol=omega_symbol,
                omega_grid=omega_grid,
                eta=eta,
                method=method,
                occupations=current,
                parameters=parameters,
                mu=mu,
                temperature=temperature,
            )
            raw_down = self.gf("down").occupation(
                omega_symbol=omega_symbol,
                omega_grid=omega_grid,
                eta=eta,
                method=method,
                occupations=current,
                parameters=parameters,
                mu=mu,
                temperature=temperature,
            )
            updated = {
                "up": float((1.0 - mixing) * current["up"] + mixing * raw_up),
                "down": float((1.0 - mixing) * current["down"] + mixing * raw_down),
            }
            max_delta = max(abs(updated[spin] - current[spin]) for spin in current)
            current = updated
            if max_delta < tol:
                return OccupationResult(occupations=current, converged=True, iterations=iteration, max_delta=max_delta)
        return OccupationResult(occupations=current, converged=False, iterations=max_iter, max_delta=max_delta)


class FermionicSingleLevel(SymbolicModelAPI):
    def __init__(self, *, eps: sp.Expr, index: Any = 0):
        super().__init__(fermionic_single_level_model(eps, index=index))


class BosonicHarmonicMode(SymbolicModelAPI):
    def __init__(self, *, omega0: sp.Expr, index: Any = 0):
        super().__init__(bosonic_harmonic_mode_model(omega0, index=index))


def _coerce_matrix_lead(
    lead: LeadSelfEnergy | Any,
    *,
    dim: int,
    mu: float = 0.0,
    temperature: float = 0.0,
    name: str,
    basis_labels: Sequence[str] | None = None,
) -> LeadSelfEnergy:
    if isinstance(lead, LeadSelfEnergy):
        if lead.dim != dim:
            raise ValueError(f"{name} must have dimension {dim}, got {lead.dim}.")
        return lead
    if isinstance(lead, Mapping):
        if basis_labels is None:
            raise ValueError(f"{name}: site-resolved couplings need known basis labels.")
        positions = {str(label): index for index, label in enumerate(basis_labels)}
        gamma = np.zeros((dim, dim), dtype=np.complex128)
        for site, coupling in lead.items():
            key = str(site)
            if key not in positions:
                raise ValueError(f"{name}: unknown site {site!r}; available sites: {sorted(positions)}.")
            gamma[positions[key], positions[key]] = _to_numeric_parameter(coupling, name=f"{name}[{site!r}]")
        return LeadSelfEnergy.wide_band(gamma, mu=mu, temperature=temperature, name=name)
    if np.isscalar(lead):
        gamma = np.eye(dim, dtype=np.complex128) * complex(lead)
    else:
        gamma = np.asarray(lead, dtype=np.complex128)
        if gamma.shape != (dim, dim):
            raise ValueError(f"{name} coupling matrix must have shape {(dim, dim)}, got {gamma.shape}.")
    return LeadSelfEnergy.wide_band(gamma, mu=mu, temperature=temperature, name=name)


class CustomModel(SymbolicModelAPI):
    """
    High-level wrapper for an arbitrary second-quantized Hamiltonian.

    Build the Hamiltonian with the operator constructors ``f``/``fd`` (fermions)
    and ``b``/``bd`` (bosons); statistics and the seed operator basis are
    detected automatically::

        eps, U = sp.symbols("epsilon U", real=True)
        model = CustomModel(eps * (n("up") + n("down")) + U * n("up") * n("down"))
        model.eom_basis().analyze(method="hartree")

    Quadratic (non-interacting) fermionic models can additionally be opened
    into a numeric two-terminal device with :meth:`open`.
    """

    def __init__(
        self,
        hamiltonian: Any,
        basis: Sequence[Any] | None = None,
        *,
        name: str = "custom",
        check_hermitian: bool = True,
    ):
        super().__init__(custom_model(hamiltonian, basis, name=name, check_hermitian=check_hermitian))

    def single_particle_matrix(self, parameters: Mapping[Any, Any] | None = None) -> tuple[sp.Matrix, list[sp.Expr]]:
        """Single-particle matrix ``h`` of a quadratic fermionic Hamiltonian, with mode labels."""
        matrix, modes = single_particle_hamiltonian_matrix(self.model.hamiltonian)
        if parameters:
            substitutions = _normalize_substitutions(sp.Matrix(matrix), parameters)
            matrix = matrix.subs(substitutions)
        return matrix, modes

    def matrix_device(self, parameters: Mapping[Any, Any] | None = None) -> MatrixDevice:
        """Numeric :class:`MatrixDevice` for a quadratic fermionic Hamiltonian."""
        matrix, modes = self.single_particle_matrix(parameters)
        remaining = sorted(sp.Matrix(matrix).free_symbols, key=lambda symbol: symbol.sort_key())
        if remaining:
            raise ValueError(f"Hamiltonian still has free symbols after substitutions: {remaining}; pass parameters={{...}}.")
        numeric = np.asarray(sp.Matrix(matrix), dtype=np.complex128)
        labels = [str(mode) for mode in modes]
        return MatrixDevice(hamiltonian=numeric, basis_labels=labels, name=self.model.name)

    def open(
        self,
        left: LeadSelfEnergy | Any,
        right: LeadSelfEnergy | Any,
        *,
        parameters: Mapping[Any, Any] | None = None,
        mu_left: float = 0.0,
        mu_right: float = 0.0,
        temperature_left: float = 0.0,
        temperature_right: float = 0.0,
    ) -> MatrixTransportView:
        """
        Open a quadratic fermionic model into a two-terminal transport view.

        ``left``/``right`` accept a :class:`LeadSelfEnergy`, a scalar wide-band
        coupling applied to every site, a full coupling matrix, or a mapping
        from mode labels to couplings for site-resolved contacts, e.g.
        ``open({"0": 0.5}, {"2": 0.5})`` to contact only the chain ends.
        Symbolic parameters must be fixed via ``parameters`` (by symbol or name).
        """
        device = self.matrix_device(parameters)
        left_lead = _coerce_matrix_lead(
            left, dim=device.dim, mu=mu_left, temperature=temperature_left, name="left", basis_labels=device.basis_labels
        )
        right_lead = _coerce_matrix_lead(
            right, dim=device.dim, mu=mu_right, temperature=temperature_right, name="right", basis_labels=device.basis_labels
        )
        return device.transport(left_lead, right_lead)
