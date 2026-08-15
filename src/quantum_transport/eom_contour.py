"""Contour and self-consistent closure adapters for EOM hierarchies.

The hierarchy itself is algebraic.  This module supplies the time structure:
the contour differential equation, its real-time Langreth components, the
vertical Matsubara branch, and an explicit fixed-point interface for closures
whose coefficients depend on the Green function being solved.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import sympy as sp

from .kadanoff_baym_symbolic import langreth_two_time_convolution_symbolic
from .kadanoff_baym import (
    self_consistent_bosonic_scba_contour_two_time,
    kadanoff_baym_dyson_two_time,
    self_consistent_born_two_time,
    two_time_greens_statistics,
)


def _expr(value: Any) -> sp.Expr:
    return sp.sympify(value.expr if hasattr(value, "expr") else value)


def _kernel_value(kernel: Any, left: Any, right: Any) -> sp.Expr:
    if callable(kernel):
        try:
            return sp.sympify(kernel(left, right))
        except TypeError:
            return sp.sympify(kernel)
    if isinstance(kernel, sp.FunctionClass):
        return kernel(left, right)
    return sp.sympify(kernel)


def _one_body_value(value: Any, argument: Any) -> sp.Expr:
    if callable(value):
        return sp.sympify(value(argument))
    if isinstance(value, sp.FunctionClass):
        return value(argument)
    return sp.sympify(value)


@dataclass(frozen=True)
class ContourEOMEquation:
    """One left/right Green-function equation and all its projections."""

    left_operator: sp.Expr
    right_operator: sp.Expr
    contour: sp.Equality
    components: Mapping[str, sp.Equality]
    source: sp.Expr
    residual: sp.Expr


@dataclass(frozen=True)
class ElectronBosonSCBAConfig:
    """Parameters for the automatic time-domain electron--boson SCBA loop."""

    coupling: Any
    boson_frequency: float
    boson_temperature: float = 0.0
    max_iterations: int = 30
    dyson_iterations: int = 80
    mixing: float = 0.5
    tolerance: float = 1e-9


@dataclass(frozen=True)
class BosonicSCBAConfig:
    """Parameters for the full-contour pure-boson SCBA loop."""

    coupling: Any | None = None
    boson_frequency: float = 1.0
    boson_temperature: float = 0.0
    cubic_vertex: Any | None = None
    quartic_vertex: Any | None = None
    max_iterations: int = 30
    dyson_iterations: int = 80
    mixing: float = 0.5
    tolerance: float = 1e-9
    matsubara_iterations: int = 60
    matsubara_dyson_iterations: int = 100
    matsubara_mixing: float = 0.25
    matsubara_tolerance: float = 1e-8


@dataclass
class EOMContourResult:
    """Symbolic contour/Langreth representation of an EOM hierarchy."""

    hierarchy: Any
    equations: tuple[ContourEOMEquation, ...]
    time: sp.Symbol
    time_prime: sp.Symbol
    imaginary_time: sp.Symbol
    imaginary_time_prime: sp.Symbol
    beta: Any

    def component(self, name: str) -> tuple[sp.Equality, ...]:
        """Return all equations for a Langreth component."""

        key = _component_key(name)
        return tuple(equation.components[key] for equation in self.equations)

    @property
    def contour_equations(self) -> tuple[sp.Equality, ...]:
        return tuple(equation.contour for equation in self.equations)

    @property
    def residuals(self) -> tuple[sp.Expr, ...]:
        return tuple(equation.residual for equation in self.equations)

    @property
    def source_matrix(self) -> sp.Matrix:
        """Return the graded equal-time source in hierarchy basis order."""

        dimension = len(self.hierarchy.basis)
        matrix = sp.zeros(dimension, dimension)
        if len(self.equations) != dimension * dimension:
            raise ValueError(
                "source_matrix requires a square hierarchy/right-operator grid."
            )
        for index, equation in enumerate(self.equations):
            row, column = divmod(index, dimension)
            matrix[row, column] = equation.source
        return matrix

    def propagate_two_time(
        self,
        time: Any,
        initial_density_matrix: Any,
        *,
        imaginary_time: Any | None = None,
        hamiltonian_matrix: Any | None = None,
        parameters: Mapping[Any, Any] | None = None,
        initial_lesser: Any | None = None,
        source_matrix: Any | None = None,
        self_energy_retarded: Any | None = None,
        self_energy_lesser: Any | None = None,
        self_energy_advanced: Any | None = None,
        initial_correlation_lesser: Any | None = None,
        electron_boson_scba: ElectronBosonSCBAConfig | Mapping[str, Any] | None = None,
        bosonic_scba: BosonicSCBAConfig | Mapping[str, Any] | None = None,
        bare_mixed: Any | None = None,
        bare_lmixed: Any | None = None,
        bare_matsubara: Any | None = None,
        max_memory_bytes: int = 512 * 1024**2,
        max_iterations: int = 100,
        mixing: float = 0.5,
        tolerance: float = 1e-10,
    ) -> "EOMTwoTimePropagationResult":
        """Propagate the EOM hierarchy directly on a numerical time grid.

        With no self-energy inputs, the EOM matrix is used as the one-body
        generator for :func:`two_time_greens`.  If retarded/lesser kernels are
        supplied, the same bare EOM Green functions are passed automatically
        to :func:`kadanoff_baym_dyson_two_time`; the caller does not rebuild the
        Dyson equations or manually assemble the bare kernels.

        ``hamiltonian_matrix`` may be a numeric matrix or ``t -> matrix``.  If
        omitted, the symbolic EOM matrix is evaluated with ``parameters``.
        This explicit numeric boundary prevents symbolic coefficients from
        being silently converted to invalid floating-point values.
        """

        grid = np.asarray(time, dtype=float)
        if grid.ndim != 1 or grid.size < 2 or not np.all(np.isfinite(grid)):
            raise ValueError("time must be a finite one-dimensional grid with at least two points.")
        if np.any(np.diff(grid) <= 0.0):
            raise ValueError("time must be strictly increasing.")
        dimension = len(self.hierarchy.basis)
        hamiltonian_at_time = _numeric_hamiltonian_function(
            self.hierarchy.eom_matrix,
            dimension=dimension,
            hamiltonian_matrix=hamiltonian_matrix,
            parameters=parameters,
        )
        numeric_source = _numeric_symbolic_matrix(
            self.source_matrix if source_matrix is None else source_matrix,
            dimension=dimension,
            parameters=parameters,
        )
        density = np.asarray(initial_density_matrix, dtype=np.complex128)
        if density.shape != (dimension, dimension):
            raise ValueError(
                f"initial_density_matrix must have shape {(dimension, dimension)}."
            )
        lesser_initial = (
            np.asarray(initial_lesser, dtype=np.complex128)
            if initial_lesser is not None
            else 1j * density
        )
        bare = two_time_greens_statistics(
            grid,
            hamiltonian_at_time,
            initial_lesser=lesser_initial,
            source_matrix=numeric_source,
            max_memory_bytes=max_memory_bytes,
        )
        if electron_boson_scba is not None and bosonic_scba is not None:
            raise ValueError("choose either electron_boson_scba or bosonic_scba.")
        if electron_boson_scba is not None and (
            self_energy_retarded is not None or self_energy_lesser is not None
        ):
            raise ValueError(
                "electron_boson_scba cannot be combined with explicit self-energy kernels."
            )
        requested_bosonic_scba = bosonic_scba
        if requested_bosonic_scba is None and electron_boson_scba is not None and self.hierarchy.statistics == "boson":
            requested_bosonic_scba = electron_boson_scba
        if requested_bosonic_scba is not None:
            if self.hierarchy.statistics != "boson":
                raise ValueError("bosonic_scba requires a purely bosonic EOM hierarchy.")
            config = (
                requested_bosonic_scba
                if isinstance(requested_bosonic_scba, BosonicSCBAConfig)
                else (
                    BosonicSCBAConfig(
                        coupling=requested_bosonic_scba.coupling,
                        boson_frequency=requested_bosonic_scba.boson_frequency,
                        boson_temperature=requested_bosonic_scba.boson_temperature,
                        cubic_vertex=None,
                        quartic_vertex=None,
                        max_iterations=requested_bosonic_scba.max_iterations,
                        dyson_iterations=requested_bosonic_scba.dyson_iterations,
                        mixing=requested_bosonic_scba.mixing,
                        tolerance=requested_bosonic_scba.tolerance,
                    )
                    if isinstance(requested_bosonic_scba, ElectronBosonSCBAConfig)
                    else BosonicSCBAConfig(**dict(requested_bosonic_scba))
                )
            )
            imaginary = (
                np.asarray(imaginary_time, dtype=float)
                if imaginary_time is not None
                else np.asarray([], dtype=float)
            )
            if imaginary.size == 0:
                raise ValueError(
                    "imaginary_time is required for bosonic_scba so the periodic Matsubara branch is explicit."
                )
            if imaginary.ndim != 1 or imaginary.size < 2 or not np.all(np.isfinite(imaginary)) or np.any(np.diff(imaginary) <= 0.0):
                raise ValueError("imaginary_time must be finite and strictly increasing.")
            from .initial_correlations import equilibrium_bosonic_matsubara_green
            from .transient import propagate_unitaries

            h_stack = np.stack([hamiltonian_at_time(value) for value in grid], axis=0)
            if bare_matsubara is None:
                green_M = equilibrium_bosonic_matsubara_green(
                    h_stack[0],
                    imaginary,
                    temperature=(config.boson_temperature if config.boson_temperature > 0.0 else None),
                )
            else:
                green_M = np.asarray(bare_matsubara, dtype=np.complex128)
            if green_M.shape != (imaginary.size, imaginary.size, dimension, dimension):
                raise ValueError("bare_matsubara must have shape (n_imaginary, n_imaginary, dim, dim).")
            evolution = propagate_unitaries(grid, hamiltonian_at_time)
            mixed = (
                np.einsum("tab,ibc->tiac", evolution, green_M[0], optimize=True)
                if bare_mixed is None
                else np.asarray(bare_mixed, dtype=np.complex128)
            )
            if mixed.shape != (grid.size, imaginary.size, dimension, dimension):
                raise ValueError("bare_mixed must have shape (n_time, n_imaginary, dim, dim).")
            lmixed = (
                mixed.swapaxes(0, 1).conj().swapaxes(-1, -2)
                if bare_lmixed is None
                else np.asarray(bare_lmixed, dtype=np.complex128)
            )
            dressed = self_consistent_bosonic_scba_contour_two_time(
                grid,
                imaginary,
                bare_retarded=bare.retarded,
                bare_lesser=bare.lesser,
                bare_mixed=mixed,
                bare_lmixed=lmixed,
                green_matsubara=green_M,
                hamiltonian=h_stack,
                coupling=config.coupling,
                boson_frequency=config.boson_frequency,
                boson_temperature=config.boson_temperature,
                cubic_vertex=config.cubic_vertex,
                quartic_vertex=config.quartic_vertex,
                max_iterations=config.max_iterations,
                dyson_iterations=config.dyson_iterations,
                mixing=config.mixing,
                tolerance=config.tolerance,
                matsubara_iterations=config.matsubara_iterations,
                matsubara_dyson_iterations=config.matsubara_dyson_iterations,
                matsubara_mixing=config.matsubara_mixing,
                matsubara_tolerance=config.matsubara_tolerance,
            )
            return EOMTwoTimePropagationResult(
                contour=self,
                green=dressed,
                solver="self_consistent_bosonic_scba_contour_two_time",
                used_self_energy=True,
            )
        if electron_boson_scba is not None:
            config = (
                electron_boson_scba
                if isinstance(electron_boson_scba, ElectronBosonSCBAConfig)
                else ElectronBosonSCBAConfig(**dict(electron_boson_scba))
            )
            _validate_electron_boson_scba_basis(self.hierarchy, config.coupling)
            dressed = self_consistent_born_two_time(
                grid,
                bare_retarded=bare.retarded,
                bare_lesser=bare.lesser,
                coupling=config.coupling,
                boson_frequency=config.boson_frequency,
                boson_temperature=config.boson_temperature,
                max_iterations=config.max_iterations,
                dyson_iterations=config.dyson_iterations,
                mixing=config.mixing,
                tolerance=config.tolerance,
            )
            return EOMTwoTimePropagationResult(
                contour=self,
                green=dressed,
                solver="self_consistent_born_two_time",
                used_self_energy=True,
            )
        if self_energy_retarded is None and self_energy_lesser is None:
            return EOMTwoTimePropagationResult(
                contour=self,
                green=bare,
                solver="finite_eom",
                used_self_energy=False,
            )
        if self_energy_retarded is None or self_energy_lesser is None:
            raise ValueError(
                "self_energy_retarded and self_energy_lesser must be supplied together."
            )
        dressed = kadanoff_baym_dyson_two_time(
            grid,
            bare_retarded=bare.retarded,
            bare_lesser=bare.lesser,
            self_energy_retarded=self_energy_retarded,
            self_energy_lesser=self_energy_lesser,
            self_energy_advanced=self_energy_advanced,
            initial_correlation_lesser=initial_correlation_lesser,
            max_iterations=max_iterations,
            mixing=mixing,
            tolerance=tolerance,
        )
        return EOMTwoTimePropagationResult(
            contour=self,
            green=dressed,
            solver="kadanoff_baym_dyson",
            used_self_energy=True,
        )

    @staticmethod
    def langreth_convolution(
        left: Mapping[str, Any],
        right: Mapping[str, Any],
        *,
        time: sp.Symbol | None = None,
        time_prime: sp.Symbol | None = None,
        imaginary_time: sp.Symbol | None = None,
        imaginary_time_prime: sp.Symbol | None = None,
        lower: Any = -sp.oo,
        upper: Any = sp.oo,
        beta: Any = sp.Symbol("beta", positive=True),
    ) -> dict[str, sp.Expr]:
        """Apply Langreth to real and vertical contour convolution components.

        The ``r/a/</>`` rules delegate to the package's two-time Langreth
        implementation.  The three vertical formulas retain the contour
        measure ``-i d tau`` explicitly, matching the Kadanoff--Baym helpers.
        """

        t = time or sp.Symbol("t", real=True)
        tp = time_prime or sp.Symbol("t_prime", real=True)
        tau = imaginary_time or sp.Symbol("tau", real=True)
        tau_prime = imaginary_time_prime or sp.Symbol("tau_prime", real=True)
        nu = sp.Symbol("nu", real=True)
        real_prime = sp.Symbol("t_bar", real=True)
        real = {
            key: value
            for key, value in left.items()
            if key in {"r", "a", "<", ">"}
        }
        right_real = {
            key: value
            for key, value in right.items()
            if key in {"r", "a", "<", ">"}
        }
        result = langreth_two_time_convolution_symbolic(
            real,
            right_real,
            t,
            tp,
            integration_time=real_prime,
            lower=lower,
            upper=upper,
        )

        def value(function: Any, left_time: Any, right_time: Any) -> sp.Expr:
            return _kernel_value(function, left_time, right_time)

        result["rceil"] = (
            sp.Integral(
                value(left["r"], t, real_prime)
                * value(right["rceil"], real_prime, tau),
                (real_prime, lower, upper),
            )
            - sp.I
            * sp.Integral(
                value(left["rceil"], t, tau_prime)
                * value(right["M"], tau_prime, tau),
                (tau_prime, 0, beta),
            )
        )
        result["lceil"] = (
            sp.Integral(
                value(left["lceil"], tau, real_prime)
                * value(right["a"], real_prime, tp),
                (real_prime, lower, upper),
            )
            - sp.I
            * sp.Integral(
                value(left["M"], tau, tau_prime)
                * value(right["lceil"], tau_prime, tp),
                (tau_prime, 0, beta),
            )
        )
        result["M"] = -sp.I * sp.Integral(
            value(left["M"], tau, nu) * value(right["M"], nu, tau_prime),
            (nu, 0, beta),
        )
        return {key: sp.sympify(value) for key, value in result.items()}


@dataclass(frozen=True)
class EOMTwoTimePropagationResult:
    """Common result facade for direct EOM and Kadanoff--Baym propagation."""

    contour: EOMContourResult
    green: Any
    solver: str
    used_self_energy: bool

    @property
    def time(self) -> np.ndarray:
        return np.asarray(self.green.time)

    @property
    def retarded(self) -> np.ndarray:
        return self.green.retarded

    @property
    def advanced(self) -> np.ndarray:
        return self.green.advanced

    @property
    def lesser(self) -> np.ndarray:
        return self.green.lesser

    @property
    def greater(self) -> np.ndarray:
        return self.green.greater

    @property
    def converged(self) -> bool:
        return bool(getattr(self.green, "converged", True))

    @property
    def iterations(self) -> int:
        return int(getattr(self.green, "iterations", 0))

    def __getattr__(self, name: str) -> Any:
        """Expose diagnostics of the underlying native solver result.

        Private and dunder names are refused rather than delegated.  Copy and
        pickle probe ``__deepcopy__``/``__getstate__``/``__reduce_ex__`` on a
        partially built instance, and forwarding those makes ``self.green``
        re-enter this method before the field exists, which recurses without
        bound instead of raising.
        """

        if name.startswith("_"):
            raise AttributeError(name)
        try:
            green = object.__getattribute__(self, "green")
        except AttributeError:
            raise AttributeError(name) from None
        return getattr(green, name)


def _numeric_symbolic_matrix(
    value: Any,
    *,
    dimension: int,
    parameters: Mapping[Any, Any] | None,
) -> np.ndarray:
    substitutions = {
        sp.sympify(key): sp.sympify(item)
        for key, item in (parameters or {}).items()
    }
    symbolic = sp.Matrix(value)
    resolved = symbolic.subs(substitutions)
    if resolved.shape != (dimension, dimension):
        raise ValueError(f"source_matrix must have shape {(dimension, dimension)}.")
    unresolved = set().union(*(entry.free_symbols for entry in resolved))
    if unresolved:
        names = ", ".join(sorted(str(item) for item in unresolved))
        raise ValueError(f"source_matrix contains unresolved symbolic coefficients: {names}.")
    numeric = np.asarray(resolved.evalf(), dtype=np.complex128)
    if not np.all(np.isfinite(numeric)):
        raise ValueError("source_matrix must contain finite values.")
    return numeric


def _validate_electron_boson_scba_basis(hierarchy: Any, coupling: Any) -> None:
    """Guard the electronic SCBA formula when a mixed basis is supplied."""

    if hierarchy.statistics == "boson":
        raise NotImplementedError(
            "Electron--boson SCBA needs an electronic Green-function block; "
            "a pure bosonic hierarchy requires a boson self-energy functional."
        )
    if hierarchy.statistics != "mixed":
        return
    from .eom_hierarchy import _fermion_parity

    matrix = np.asarray(coupling, dtype=np.complex128)
    dimension = len(hierarchy.basis)
    if matrix.shape != (dimension, dimension):
        raise ValueError(
            "For a mixed hierarchy, electron_boson_scba.coupling must match "
            f"the full basis shape {(dimension, dimension)}."
        )
    bosonic_rows = [
        index
        for index, operator in enumerate(hierarchy.basis)
        if _fermion_parity(operator) == 0
    ]
    if bosonic_rows and not np.allclose(matrix[bosonic_rows, :], 0.0):
        raise ValueError(
            "Electron--boson SCBA coupling must vanish on bosonic rows in a mixed basis."
        )
    if bosonic_rows and not np.allclose(matrix[:, bosonic_rows], 0.0):
        raise ValueError(
            "Electron--boson SCBA coupling must vanish on bosonic columns in a mixed basis."
        )


def _numeric_hamiltonian_function(
    symbolic_matrix: sp.Matrix,
    *,
    dimension: int,
    hamiltonian_matrix: Any | None,
    parameters: Mapping[Any, Any] | None,
) -> Callable[[float], np.ndarray]:
    if hamiltonian_matrix is None:
        substitutions = {
            sp.sympify(key): sp.sympify(value)
            for key, value in (parameters or {}).items()
        }
        resolved = symbolic_matrix.subs(substitutions)
        unresolved = set().union(*(entry.free_symbols for entry in resolved))
        if unresolved:
            names = ", ".join(sorted(str(item) for item in unresolved))
            raise ValueError(
                "EOM matrix contains unresolved symbolic coefficients: "
                f"{names}. Supply parameters or hamiltonian_matrix."
            )
        matrix = np.asarray(resolved.evalf(), dtype=np.complex128)
        return _constant_matrix_function(matrix, dimension)

    if callable(hamiltonian_matrix):
        def at_time(value: float) -> np.ndarray:
            return _validate_numeric_matrix(
                hamiltonian_matrix(value),
                dimension=dimension,
            )

        return at_time
    return _constant_matrix_function(
        _validate_numeric_matrix(hamiltonian_matrix, dimension=dimension),
        dimension,
    )


def _constant_matrix_function(matrix: np.ndarray, dimension: int) -> Callable[[float], np.ndarray]:
    checked = _validate_numeric_matrix(matrix, dimension=dimension)

    def at_time(value: float) -> np.ndarray:
        del value
        return checked.copy()

    return at_time


def _validate_numeric_matrix(value: Any, *, dimension: int) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.complex128)
    if matrix.shape != (dimension, dimension):
        raise ValueError(
            f"hamiltonian_matrix must have shape {(dimension, dimension)}."
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("hamiltonian_matrix must contain finite values.")
    if not np.allclose(matrix, matrix.conj().T, atol=1e-12, rtol=1e-12):
        raise ValueError("hamiltonian_matrix must be Hermitian for unitary propagation.")
    return matrix


def _component_key(component: str) -> str:
    aliases = {
        "r": "r",
        "retarded": "r",
        "a": "a",
        "advanced": "a",
        "<": "<",
        "lesser": "<",
        ">": ">",
        "greater": ">",
        "rceil": "rceil",
        "lceil": "lceil",
        "m": "M",
        "M": "M",
        "matsubara": "M",
    }
    key = str(component).strip()
    if key.lower() in {"m", "matsubara"}:
        return "M"
    if key.lower() not in aliases:
        raise ValueError(f"Unsupported contour component: {component!r}")
    return aliases[key.lower()]


def _component_kernel(
    supplied: Mapping[str, Any] | None,
    key: str,
    row: int,
    column: int,
    left: Any,
    right: Any,
    *,
    time: Any,
    time_prime: Any,
    imaginary_time: Any,
) -> sp.Expr:
    if supplied is not None and key in supplied:
        candidate = supplied[key]
        if callable(candidate):
            for arguments in (
                (row, column, left, right),
                (left, right),
            ):
                try:
                    return sp.sympify(candidate(*arguments))
                except TypeError:
                    continue
        return sp.sympify(candidate)
    label = f"R_{key}_{row}_{column}"
    if key == "M":
        return sp.Function(label)(imaginary_time, sp.Symbol("tau_prime", real=True))
    if key == "rceil":
        return sp.Function(label)(time, imaginary_time)
    if key == "lceil":
        return sp.Function(label)(imaginary_time, time_prime)
    return sp.Function(label)(time, time_prime)


def contour_eom_from_hierarchy(
    hierarchy: Any,
    *,
    time: sp.Symbol | None = None,
    time_prime: sp.Symbol | None = None,
    imaginary_time: sp.Symbol | None = None,
    imaginary_time_prime: sp.Symbol | None = None,
    right_operators: Sequence[Any] | None = None,
    residual_components: Mapping[str, Any] | None = None,
    beta: Any = sp.Symbol("beta", positive=True),
) -> EOMContourResult:
    """Build contour and Langreth equations for every hierarchy pair."""

    t = time or sp.Symbol("t", real=True)
    tp = time_prime or sp.Symbol("t_prime", real=True)
    tau = imaginary_time or sp.Symbol("tau", real=True)
    tau_prime = imaginary_time_prime or sp.Symbol("tau_prime", real=True)
    rights = (
        [_expr(item) for item in right_operators]
        if right_operators is not None
        else [_dagger(item) for item in hierarchy.basis]
    )
    if len(rights) != len(hierarchy.basis):
        raise ValueError("right_operators must have the same length as hierarchy.basis.")

    equations: list[ContourEOMEquation] = []
    from .models import _ladder_indices

    fermion_modes, boson_modes = _ladder_indices(hierarchy.hamiltonian)
    all_modes = [*fermion_modes, *boson_modes]
    from .eom_hierarchy import _retarded_source

    matrix = hierarchy.eom_matrix
    for row, left_operator in enumerate(hierarchy.basis):
        residual = hierarchy.equations[row].residual
        for right_index, right_operator in enumerate(rights):
            source = _retarded_source(left_operator, right_operator, all_modes)
            contour_value = sp.Function(
                f"G_C_{row}_{right_index}", commutative=False
            )(t, tp)
            left_side = sp.I * sp.Derivative(contour_value, t)
            for column in range(len(hierarchy.basis)):
                left_side -= matrix[row, column] * sp.Function(
                    f"G_C_{column}_{right_index}", commutative=False
                )(t, tp)
            contour_rhs = source * sp.DiracDelta(t - tp)
            if residual != 0:
                contour_rhs += _component_kernel(
                    residual_components,
                    "contour",
                    row,
                    right_index,
                    t,
                    tp,
                    time=t,
                    time_prime=tp,
                    imaginary_time=tau,
                )
            components: dict[str, sp.Equality] = {}
            green_names = {
                "r": f"G_r_{row}_{right_index}",
                "a": f"G_a_{row}_{right_index}",
                "<": f"G_lesser_{row}_{right_index}",
                ">": f"G_greater_{row}_{right_index}",
                "rceil": f"G_rceil_{row}_{right_index}",
                "lceil": f"G_lceil_{row}_{right_index}",
                "M": f"G_M_{row}_{right_index}",
            }
            residuals = {
                key: _component_kernel(
                    residual_components,
                    key,
                    row,
                    right_index,
                    t,
                    tp,
                    time=t,
                    time_prime=tp,
                    imaginary_time=tau,
                )
                if residual != 0
                else sp.Integer(0)
                for key in green_names
            }
            for key, name in green_names.items():
                if key == "M":
                    value = sp.Function(name, commutative=False)(tau, tau_prime)
                    left_m = -sp.Derivative(value, tau)
                    for column in range(len(hierarchy.basis)):
                        left_m -= matrix[row, column] * sp.Function(
                            f"G_M_{column}_{right_index}", commutative=False
                        )(tau, tau_prime)
                    rhs = source * sp.DiracDelta(tau - tau_prime) + residuals[key]
                elif key == "a":
                    value = sp.Function(name, commutative=False)(tp, t)
                    left_m = -sp.I * sp.Derivative(value, t)
                    for column in range(len(hierarchy.basis)):
                        # Right multiplication by the EOM matrix: the summed
                        # index is the second slot of G^a, so the free column
                        # of ``matrix`` is the right operator, not ``row``.
                        left_m -= sp.Function(
                            f"G_a_{row}_{column}", commutative=False
                        )(tp, t) * matrix[column, right_index]
                    rhs = source * sp.DiracDelta(t - tp) + residuals[key]
                elif key == "lceil":
                    value = sp.Function(name, commutative=False)(tau, t)
                    left_m = -sp.I * sp.Derivative(value, t)
                    for column in range(len(hierarchy.basis)):
                        left_m -= sp.Function(
                            f"G_lceil_{row}_{column}", commutative=False
                        )(tau, t) * matrix[column, right_index]
                    rhs = residuals[key]
                elif key == "rceil":
                    value = sp.Function(name, commutative=False)(t, tau)
                    left_m = sp.I * sp.Derivative(value, t)
                    for column in range(len(hierarchy.basis)):
                        left_m -= matrix[row, column] * sp.Function(
                            f"G_rceil_{column}_{right_index}", commutative=False
                        )(t, tau)
                    rhs = residuals[key]
                else:
                    value = sp.Function(name, commutative=False)(t, tp)
                    left_m = sp.I * sp.Derivative(value, t)
                    for column in range(len(hierarchy.basis)):
                        base = name.rsplit("_", 2)[0]
                        left_m -= matrix[row, column] * sp.Function(
                            f"{base}_{column}_{right_index}", commutative=False
                        )(t, tp)
                    rhs = (
                        source * sp.DiracDelta(t - tp)
                        if key == "r"
                        else sp.Integer(0)
                    ) + residuals[key]
                components[key] = sp.Eq(left_m, rhs)
            equations.append(
                ContourEOMEquation(
                    left_operator=left_operator,
                    right_operator=right_operator,
                    contour=sp.Eq(left_side, contour_rhs),
                    components=components,
                    source=source,
                    residual=residual,
                )
            )
    return EOMContourResult(
        hierarchy=hierarchy,
        equations=tuple(equations),
        time=t,
        time_prime=tp,
        imaginary_time=tau,
        imaginary_time_prime=tau_prime,
        beta=beta,
    )


def _dagger(value: Any) -> sp.Expr:
    from .models import dagger_expression

    return dagger_expression(value)


@dataclass(frozen=True)
class ClosureIteration:
    iteration: int
    values: Mapping[Any, sp.Expr]
    converged: bool
    max_change: Any


@dataclass(frozen=True)
class SelfConsistentClosureResult:
    values: Mapping[Any, sp.Expr]
    green: sp.Matrix
    converged: bool
    iterations: tuple[ClosureIteration, ...]


@dataclass
class SelfConsistentClosure:
    """Fixed-point closure for residual operator coefficients.

    ``rules`` maps unresolved operator strings to expressions such as
    ``n_down * f_up``.  ``initial_values`` supplies the scalar values of the
    closure variables, and ``update`` returns their next values from the
    current Green function.  This makes the closure policy explicit and
    auditable instead of silently replacing a residual by zero.
    """

    rules: Mapping[Any, Any]
    initial_values: Mapping[Any, Any]
    update: Callable[[Mapping[Any, sp.Expr], sp.Matrix], Mapping[Any, Any]]
    max_iterations: int = 50
    tolerance: float = 1e-10
    mixing: float = 1.0
    history: list[ClosureIteration] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive.")
        if not 0.0 < self.mixing <= 1.0:
            raise ValueError("mixing must be in (0, 1].")
        if self.tolerance < 0.0:
            raise ValueError("tolerance must be non-negative.")

    def _resolved_rules(self, values: Mapping[Any, Any]) -> dict[Any, sp.Expr]:
        substitutions = {sp.sympify(key): sp.sympify(value) for key, value in values.items()}
        resolved: dict[Any, sp.Expr] = {}
        for operator, rule in self.rules.items():
            expression = rule(values) if callable(rule) else sp.sympify(rule)
            resolved[_expr(operator)] = sp.sympify(expression).subs(substitutions)
        return resolved

    @staticmethod
    def _change(old: Mapping[Any, Any], new: Mapping[Any, Any]) -> Any:
        changes = []
        for key in set(old) | set(new):
            difference = sp.sympify(new.get(key, old.get(key, 0))) - sp.sympify(old.get(key, 0))
            if difference.is_number:
                changes.append(abs(complex(sp.N(difference))))
            elif sp.simplify(difference) != 0:
                return sp.oo
        return max(changes, default=0.0)

    def solve(
        self,
        hierarchy: Any,
        omega: sp.Expr,
        eta: sp.Expr,
        *,
        right_operators: Sequence[Any] | None = None,
    ) -> SelfConsistentClosureResult:
        values = {sp.sympify(key): sp.sympify(value) for key, value in self.initial_values.items()}
        iterations: list[ClosureIteration] = []
        green = sp.Matrix([])
        converged = False
        for index in range(1, self.max_iterations + 1):
            green = hierarchy.retarded_green(
                omega,
                eta,
                right_operators=right_operators,
                residual_closure=self._resolved_rules(values),
            )
            proposed = {
                sp.sympify(key): sp.sympify(value)
                for key, value in self.update(values, green).items()
            }
            mixing = sp.Rational(str(self.mixing))
            mixed = {
                key: sp.simplify(
                    mixing * proposed.get(key, value)
                    + (1 - mixing) * value
                )
                for key, value in values.items()
            }
            mixed.update({key: value for key, value in proposed.items() if key not in mixed})
            change = self._change(values, mixed)
            converged = bool(change != sp.oo and change <= self.tolerance)
            iterations.append(
                ClosureIteration(index, dict(mixed), converged, change)
            )
            values = mixed
            if converged:
                break
        self.history = iterations
        return SelfConsistentClosureResult(
            values=dict(values),
            green=green,
            converged=converged,
            iterations=tuple(iterations),
        )
