"""Traceable symbolic equation-of-motion hierarchies.

This module adds a model-independent workflow on top of the existing EOM
engine: labelled Hamiltonian contributions, depth-limited basis expansion,
residual provenance, explicit truncation, and Green-function solving.  The
same hierarchy now supports fermionic, bosonic, and mixed ladder algebras;
the contour/Langreth projection lives in :mod:`quantum_transport.eom_contour`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import sympy as sp
from sympy import KroneckerDelta
from sympy.physics.secondquant import (
    AnnihilateBoson,
    AnnihilateFermion,
    CreateBoson,
    CreateFermion,
)

from .models import (
    _decompose_canonical_ladder_terms,
    _exact_fermionic_rhs,
    _extract_operator_monomials,
    _ladder_factors,
    _ladder_indices,
    _unique_basis,
    collapse_orthogonal_mode_deltas,
    dagger_expression,
)
from .secondquant import normal_order_fermionic


ResidualClosure = Mapping[Any, Any]


def _expr(value: Any) -> sp.Expr:
    """Unwrap the lightweight operator wrappers used by the package."""

    return sp.sympify(value.expr if hasattr(value, "expr") else value)


def _key(value: Any) -> str:
    return sp.srepr(_expr(value))


def _operator_order(value: Any) -> int:
    """Return the number of ladder factors in a canonical operator string."""

    _, factors = _ladder_factors(_expr(value))
    return len(factors)


@dataclass(frozen=True)
class HamiltonianTerm:
    """A labelled contribution to a Hamiltonian."""

    label: str
    expression: sp.Expr
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "expression", _expr(self.expression))


@dataclass(frozen=True)
class EOMContribution:
    """One labelled contribution to one operator equation."""

    operator: sp.Expr
    hamiltonian_label: str
    rhs: sp.Expr
    projected: tuple[tuple[sp.Expr, sp.Expr], ...]
    residual: sp.Expr
    discovered_operators: tuple[sp.Expr, ...]

    @property
    def is_closed_on_basis(self) -> bool:
        return sp.simplify(self.residual) == 0

    @property
    def operator_order(self) -> int:
        return _operator_order(self.operator)


@dataclass(frozen=True)
class EOMEquation:
    """The assembled EOM for one basis operator."""

    operator: sp.Expr
    rhs: sp.Expr
    projected: tuple[tuple[sp.Expr, sp.Expr], ...]
    residual: sp.Expr
    contributions: tuple[EOMContribution, ...]
    depth: int

    @property
    def is_closed_on_basis(self) -> bool:
        return sp.simplify(self.residual) == 0

    @property
    def operator_order(self) -> int:
        return _operator_order(self.operator)


@dataclass
class EOMHierarchyResult:
    """Result of a depth-limited, provenance-preserving EOM expansion."""

    hamiltonian_terms: tuple[HamiltonianTerm, ...]
    basis: tuple[sp.Expr, ...]
    equations: tuple[EOMEquation, ...]
    depth_by_operator: Mapping[str, int]
    requested_depth: int
    reached_depth: int
    max_operators_reached: bool = False
    statistics: str = "fermion"

    @property
    def hamiltonian(self) -> sp.Expr:
        return sp.expand(sp.Add(*(term.expression for term in self.hamiltonian_terms)))

    @property
    def eom_matrix(self) -> sp.Matrix:
        positions = {_key(operator): index for index, operator in enumerate(self.basis)}
        rows = []
        for equation in self.equations:
            row = [sp.Integer(0)] * len(self.basis)
            for basis_operator, coefficient in equation.projected:
                row[positions[_key(basis_operator)]] = coefficient
            rows.append(row)
        return sp.Matrix(rows)

    @property
    def residuals(self) -> tuple[sp.Expr, ...]:
        return tuple(equation.residual for equation in self.equations)

    @property
    def is_closed(self) -> bool:
        return all(sp.simplify(residual) == 0 for residual in self.residuals)

    @property
    def unresolved_operators(self) -> tuple[sp.Expr, ...]:
        found: list[sp.Expr] = []
        seen: set[str] = set()
        for residual in self.residuals:
            for operator in _extract_operator_monomials(residual):
                key = _key(operator)
                if key not in seen:
                    seen.add(key)
                    found.append(operator)
        return tuple(found)

    def equation(self, operator: Any) -> EOMEquation:
        wanted = _key(operator)
        for equation in self.equations:
            if _key(equation.operator) == wanted:
                return equation
        raise KeyError(f"Operator is not in the hierarchy basis: {operator!r}")

    def retarded_green(
        self,
        omega: sp.Expr,
        eta: sp.Expr,
        *,
        right_operators: Sequence[Any] | None = None,
        approximate: bool = False,
        simplify: bool = False,
        residual_closure: ResidualClosure | None = None,
    ) -> sp.Matrix:
        """Solve G^r. Approximate mode explicitly drops unresolved residuals."""

        eom_matrix, residuals = self._matrix_with_closure(
            residual_closure,
            drop_residual=approximate,
        )
        if any(sp.simplify(residual) != 0 for residual in residuals) and not approximate:
            raise ValueError(
                "EOM hierarchy is not closed. Increase max_depth or pass "
                "a residual_closure or approximate=True to drop the unresolved "
                "residual explicitly."
            )
        right = (
            [_expr(item) for item in right_operators]
            if right_operators is not None
            else [dagger_expression(operator) for operator in self.basis]
        )
        fermion_modes, boson_modes = _ladder_indices(
            self.hamiltonian + sp.Add(*self.basis)
        )
        modes = [*fermion_modes, *boson_modes]
        source = sp.Matrix(
            [
                [
                    _retarded_source(left, right_op, modes)
                    for right_op in right
                ]
                for left in self.basis
            ]
        )
        lhs = (omega + sp.I * eta) * sp.eye(eom_matrix.shape[0]) - eom_matrix
        result = lhs.LUsolve(source)
        return result.applyfunc(sp.simplify) if simplify else result

    def contour_equations(
        self,
        *,
        time: sp.Symbol | None = None,
        time_prime: sp.Symbol | None = None,
        imaginary_time: sp.Symbol | None = None,
        imaginary_time_prime: sp.Symbol | None = None,
        right_operators: Sequence[Any] | None = None,
        residual_components: Mapping[str, Any] | None = None,
        beta: Any = sp.Symbol("beta", positive=True),
    ):
        """Project the hierarchy to contour and Langreth component equations.

        The returned object keeps the contour differential equation, the real
        time ``r/a/</>`` projections, and the vertical ``rceil/lceil/M``
        equations together.  Residuals are represented as explicit kernels;
        callers may provide their component expressions through
        ``residual_components`` when a self-energy or initial-correlation
        closure is available.
        """

        from .eom_contour import contour_eom_from_hierarchy

        return contour_eom_from_hierarchy(
            self,
            time=time,
            time_prime=time_prime,
            imaginary_time=imaginary_time,
            imaginary_time_prime=imaginary_time_prime,
            right_operators=right_operators,
            residual_components=residual_components,
            beta=beta,
        )

    def solve_self_consistent(
        self,
        omega: sp.Expr,
        eta: sp.Expr,
        closure: Any,
        *,
        right_operators: Sequence[Any] | None = None,
    ):
        """Iterate a configurable residual closure against the Green function."""

        return closure.solve(
            self,
            omega,
            eta,
            right_operators=right_operators,
        )

    def stationary_lesser_green(
        self,
        omega: sp.Expr,
        eta: sp.Expr,
        sigma_lesser: Any,
        *,
        right_operators: Sequence[Any] | None = None,
        approximate: bool = False,
        simplify: bool = False,
        residual_closure: ResidualClosure | None = None,
    ) -> sp.Matrix:
        """Return the stationary closure G^< = G^r Sigma^< G^a."""

        g_retarded = self.retarded_green(
            omega,
            eta,
            right_operators=right_operators,
            approximate=approximate,
            simplify=simplify,
            residual_closure=residual_closure,
        )
        # Solve the resolvent again at ``-eta`` rather than substituting into
        # ``g_retarded``.  A substitution rewrites *every* occurrence of that
        # value, so a Hamiltonian coefficient numerically equal to ``eta`` has
        # its sign flipped too and the lesser function is silently wrong.
        g_advanced = self.retarded_green(
            omega,
            -eta,
            right_operators=right_operators,
            approximate=approximate,
            simplify=simplify,
            residual_closure=residual_closure,
        )
        result = g_retarded * sp.sympify(sigma_lesser) * g_advanced
        return result.applyfunc(sp.simplify) if simplify else result

    def _matrix_with_closure(
        self,
        residual_closure: ResidualClosure | None,
        *,
        drop_residual: bool,
    ) -> tuple[sp.Matrix, tuple[sp.Expr, ...]]:
        """Project EOMs after an optional exact operator substitution."""

        rows = []
        residuals = []
        for equation in self.equations:
            rhs = (
                equation.rhs
                if residual_closure is None
                else _apply_residual_closure(equation.rhs, residual_closure)
            )
            projected, residual = _project(rhs, self.basis)
            positions = {_key(operator): index for index, operator in enumerate(self.basis)}
            row = [sp.Integer(0)] * len(self.basis)
            for operator, coefficient in projected:
                row[positions[_key(operator)]] = coefficient
            rows.append(row)
            residuals.append(sp.Integer(0) if drop_residual else residual)
        return sp.Matrix(rows), tuple(residuals)

    def latex_equations(self) -> str:
        """Render the assembled EOMs as a compact LaTeX block."""

        lines = []
        for equation in self.equations:
            lines.append(
                rf"[\hat{{O}}={sp.latex(equation.operator)},H]="
                rf"{sp.latex(equation.rhs)}"
            )
            if equation.residual != 0:
                lines.append(rf"\qquad R={sp.latex(equation.residual)}")
        return "\n".join(lines)


def _normalise_terms(hamiltonian: Any) -> tuple[HamiltonianTerm, ...]:
    if isinstance(hamiltonian, HamiltonianTerm):
        return (hamiltonian,)
    if isinstance(hamiltonian, Mapping):
        return tuple(HamiltonianTerm(str(label), expression) for label, expression in hamiltonian.items())
    if isinstance(hamiltonian, Sequence) and not isinstance(hamiltonian, (str, bytes, sp.Basic)):
        if all(isinstance(item, HamiltonianTerm) for item in hamiltonian):
            return tuple(hamiltonian)
        if all(isinstance(item, (tuple, list)) and len(item) == 2 for item in hamiltonian):
            return tuple(HamiltonianTerm(str(label), expression) for label, expression in hamiltonian)
    return (HamiltonianTerm("H", _expr(hamiltonian)),)


def _project(rhs: sp.Expr, basis: Sequence[sp.Expr]) -> tuple[tuple[tuple[sp.Expr, sp.Expr], ...], sp.Expr]:
    coefficients, residual = _decompose_canonical_ladder_terms(rhs, basis)
    projected = tuple(
        (operator, sp.simplify(coefficient))
        for operator, coefficient in zip(basis, coefficients)
        if coefficient != 0
    )
    return projected, sp.simplify(sp.expand(residual))


def _rhs_for(
    operator: sp.Expr,
    term: HamiltonianTerm,
    fermion_modes: Sequence[sp.Expr],
    boson_modes: Sequence[sp.Expr],
    statistics: str,
) -> sp.Expr:
    if statistics == "fermion":
        rhs = _exact_fermionic_rhs(operator, term.expression, fermion_modes)
    else:
        rhs = _normal_order_ladder_expression(
            sp.expand(operator * term.expression - term.expression * operator)
        )
    return sp.expand(
        collapse_orthogonal_mode_deltas(rhs, [*fermion_modes, *boson_modes])
    )


def _bosonic_delta(left: sp.Expr, right: sp.Expr) -> sp.Expr:
    if left == right:
        return sp.Integer(1)
    return KroneckerDelta(left, right)


def _normal_order_boson_factors(factors: Sequence[sp.Expr]) -> sp.Expr:
    """Normal-order bosons using ``b b^dag = b^dag b + delta``."""

    factors = list(factors)
    for position in range(len(factors) - 1):
        left, right = factors[position : position + 2]
        if not isinstance(left, AnnihilateBoson) or not isinstance(right, CreateBoson):
            continue
        prefix = factors[:position]
        suffix = factors[position + 2 :]
        swapped = prefix + [right, left] + suffix
        contracted = prefix + suffix
        return sp.expand(
            _normal_order_boson_factors(swapped)
            + _bosonic_delta(left.args[0], right.args[0])
            * _normal_order_boson_factors(contracted)
        )
    return sp.Mul(*factors) if factors else sp.Integer(1)


def _normal_order_ladder_expression(expr: sp.Expr) -> sp.Expr:
    """Canonical-order bosonic/mixed ladder strings while preserving scalars."""

    expanded = sp.expand(expr)
    terms = expanded.args if isinstance(expanded, sp.Add) else ((expanded,) if expanded != 0 else ())
    rebuilt: list[sp.Expr] = []
    for term in terms:
        coefficient, factors = _ladder_factors(term)
        if coefficient == 0:
            continue
        if not factors:
            rebuilt.append(term)
            continue
        expanded_factors: list[sp.Expr] = []
        for factor in factors:
            if (
                isinstance(factor, sp.Pow)
                and factor.exp.is_Integer
                and int(factor.exp) >= 1
                and isinstance(
                    factor.base,
                    (
                        AnnihilateFermion,
                        CreateFermion,
                        AnnihilateBoson,
                        CreateBoson,
                    ),
                )
            ):
                expanded_factors.extend([factor.base] * int(factor.exp))
            else:
                expanded_factors.append(factor)
        factors = expanded_factors
        fermions = [
            factor
            for factor in factors
            if isinstance(factor, (AnnihilateFermion, CreateFermion))
        ]
        bosons = [
            factor
            for factor in factors
            if isinstance(factor, (AnnihilateBoson, CreateBoson))
        ]
        unsupported = [factor for factor in factors if factor not in (*fermions, *bosons)]
        if unsupported:
            raise ValueError(
                "EOM hierarchy encountered unsupported noncommutative factors: "
                f"{unsupported!r}."
            )
        fermion_part = (
            normal_order_fermionic(sp.Mul(*fermions))
            if fermions
            else sp.Integer(1)
        )
        boson_part = _normal_order_boson_factors(bosons)
        rebuilt.append(coefficient * fermion_part * boson_part)
    return sp.expand(sp.Add(*rebuilt))


def _fermion_parity(expr: sp.Expr) -> int:
    count = len(expr.atoms(AnnihilateFermion, CreateFermion))
    return count % 2


def _retarded_source(
    left: sp.Expr,
    right: sp.Expr,
    modes: Sequence[sp.Expr],
) -> sp.Expr:
    """Return the graded retarded source for pure or mixed operators."""

    if _fermion_parity(left) != _fermion_parity(right):
        # Cross-statistics correlators are independent sectors in the mixed
        # hierarchy.  Keeping their source zero prevents an ordinary product
        # such as f*b^dag from being mistaken for a graded canonical delta.
        return sp.Integer(0)
    sign = 1 if _fermion_parity(left) else -1
    source = _normal_order_ladder_expression(
        sp.expand(left * right + sign * right * left)
    )
    return sp.expand(collapse_orthogonal_mode_deltas(source, modes))


def _apply_residual_closure(rhs: sp.Expr, residual_closure: ResidualClosure) -> sp.Expr:
    """Replace canonical operator monomials while preserving scalar factors."""

    substitutions = {_key(key): _expr(value) for key, value in residual_closure.items()}
    expanded = sp.expand(rhs)
    terms = expanded.args if isinstance(expanded, sp.Add) else ((expanded,) if expanded != 0 else ())
    replaced = []
    for term in terms:
        coefficient, factors = _ladder_factors(term)
        if not factors:
            replaced.append(term)
            continue
        operator = sp.Mul(*factors)
        operator_key = _key(operator)
        if operator_key in substitutions:
            replaced.append(coefficient * substitutions[operator_key])
        else:
            replaced.append(term)
    return sp.expand(sp.Add(*replaced))


def build_eom_hierarchy(
    hamiltonian: Any,
    *,
    basis: Sequence[Any] | None = None,
    max_depth: int = 1,
    max_operators: int | None = None,
    check_hermitian: bool = True,
) -> EOMHierarchyResult:
    """Build a labelled fermionic EOM hierarchy.

    hamiltonian may be a SymPy expression, a mapping label -> expression, a
    sequence of (label, expression) pairs, or HamiltonianTerm objects.

    max_depth counts residual-expansion rounds: depth zero keeps the seed
    basis, depth one adds operators found in its residuals, and so on.
    """

    if not isinstance(max_depth, int) or max_depth < 0:
        raise ValueError("max_depth must be a non-negative integer.")
    if max_operators is not None and (not isinstance(max_operators, int) or max_operators <= 0):
        raise ValueError("max_operators must be a positive integer or None.")

    terms = _normalise_terms(hamiltonian)
    if not terms:
        raise ValueError("At least one Hamiltonian term is required.")
    total = sp.expand(sp.Add(*(term.expression for term in terms)))

    fermion_modes, boson_modes = _ladder_indices(total)
    if not fermion_modes and not boson_modes:
        raise ValueError(
            "The Hamiltonian contains no fermionic or bosonic ladder operators."
        )
    if fermion_modes and boson_modes:
        statistics = "mixed"
    elif fermion_modes:
        statistics = "fermion"
    else:
        statistics = "boson"

    if check_hermitian:
        from .models import _warn_if_not_hermitian

        _warn_if_not_hermitian(
            total,
            statistics,
            "eom_hierarchy",
            [*fermion_modes, *boson_modes],
        )

    if basis is None:
        from .bosonic import destroy_b
        from .secondquant import destroy

        current_basis = [destroy(mode).doit() for mode in fermion_modes]
        current_basis.extend(destroy_b(mode).doit() for mode in boson_modes)
    else:
        current_basis = _unique_basis([_expr(operator) for operator in basis])
    if not current_basis:
        raise ValueError("The EOM seed basis cannot be empty.")

    depth_by_key: dict[str, int] = {_key(operator): 0 for operator in current_basis}
    frontier = list(current_basis)
    reached_depth = 0
    capped = False

    for depth in range(max_depth):
        discovered: list[sp.Expr] = []
        known = {_key(operator) for operator in current_basis}
        for operator in frontier:
            for term in terms:
                rhs = _rhs_for(
                    operator,
                    term,
                    fermion_modes,
                    boson_modes,
                    statistics,
                )
                _, residual = _project(rhs, current_basis)
                for candidate in _extract_operator_monomials(residual):
                    if _key(candidate) not in known:
                        discovered.append(candidate)
                        known.add(_key(candidate))
        if max_operators is not None:
            room = max_operators - len(current_basis)
            if room <= 0:
                capped = bool(discovered)
                break
            if len(discovered) > room:
                discovered = discovered[:room]
                capped = True
        if not discovered:
            break
        for operator in discovered:
            current_basis.append(operator)
            depth_by_key[_key(operator)] = depth + 1
        frontier = discovered
        reached_depth = depth + 1
        if capped:
            break

    equations: list[EOMEquation] = []
    for operator in current_basis:
        contributions: list[EOMContribution] = []
        rhs_terms: list[sp.Expr] = []
        for term in terms:
            rhs = _rhs_for(
                operator,
                term,
                fermion_modes,
                boson_modes,
                statistics,
            )
            projected, residual = _project(rhs, current_basis)
            discovered = tuple(_extract_operator_monomials(residual))
            contributions.append(
                EOMContribution(
                    operator=operator,
                    hamiltonian_label=term.label,
                    rhs=rhs,
                    projected=projected,
                    residual=residual,
                    discovered_operators=discovered,
                )
            )
            rhs_terms.append(rhs)
        total_rhs = sp.expand(sp.Add(*rhs_terms))
        projected, residual = _project(total_rhs, current_basis)
        equations.append(
            EOMEquation(
                operator=operator,
                rhs=total_rhs,
                projected=projected,
                residual=residual,
                contributions=tuple(contributions),
                depth=depth_by_key[_key(operator)],
            )
        )

    return EOMHierarchyResult(
        hamiltonian_terms=terms,
        basis=tuple(current_basis),
        equations=tuple(equations),
        depth_by_operator=dict(depth_by_key),
        requested_depth=max_depth,
        reached_depth=reached_depth,
        max_operators_reached=capped,
        statistics=statistics,
    )


__all__ = [
    "HamiltonianTerm",
    "EOMContribution",
    "EOMEquation",
    "EOMHierarchyResult",
    "build_eom_hierarchy",
]
