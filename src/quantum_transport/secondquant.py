"""Symbolic helpers built on top of sympy.physics.secondquant."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import sympy as sp
from sympy import KroneckerDelta
from sympy.physics.secondquant import AnnihilateFermion, CreateFermion, F, Fd, wicks

from .algebra import decompose_in_basis
from .symbolic_base import SymbolicOperator, eom_system_latex


@dataclass
class FermionicEOMResult:
    operators: Sequence[sp.Expr]
    eom_matrix: sp.Matrix
    residuals: Sequence[sp.Expr]

    @property
    def is_closed(self) -> bool:
        return all(sp.simplify(residual) == 0 for residual in self.residuals)

    def _repr_latex_(self) -> str:
        return eom_system_latex(self.operators, self.eom_matrix, self.residuals)


def annihilate(index: sp.Expr) -> sp.Expr:
    """Return the fermionic annihilation operator f(index)."""
    return F(index)


def create(index: sp.Expr) -> sp.Expr:
    """Return the fermionic creation operator f_dagger(index)."""
    return Fd(index)


def destroy(index: sp.Expr) -> "SQObj":
    """QuTiP-like constructor for a fermionic annihilation operator."""
    return SQObj(annihilate(index))


def num(index: sp.Expr) -> "SQObj":
    """QuTiP-like constructor for a number operator."""
    return SQObj(number_operator(index, simplify=False))


def sqobj(expr: sp.Expr | "SQObj") -> "SQObj":
    """Wrap a SymPy second-quantized expression in a small operator object."""
    return SQObj(_unwrap(expr))


def number_operator(index: sp.Expr, simplify: bool = True) -> sp.Expr:
    """Return n_index = f_dagger(index) f(index)."""
    expr = create(index) * annihilate(index)
    return simplify_secondquant(expr) if simplify else expr


def _unwrap(expr: sp.Expr | "SQObj") -> sp.Expr:
    # Unwrap any operator wrapper (SQObj, BQObj, ...) so mixed
    # fermion-boson products like fd(0) * b(0) build cleanly.
    return expr.expr if isinstance(expr, SymbolicOperator) else expr


def _wrap(expr: sp.Expr | "SQObj") -> "SQObj":
    return expr if isinstance(expr, SQObj) else SQObj(expr)


def _dagger_secondquant(expr: sp.Expr) -> sp.Expr:
    if isinstance(expr, AnnihilateFermion):
        return CreateFermion(expr.args[0])
    if isinstance(expr, CreateFermion):
        return AnnihilateFermion(expr.args[0])
    if isinstance(expr, sp.Add):
        return sp.Add(*[_dagger_secondquant(arg) for arg in expr.args])
    if isinstance(expr, sp.Mul):
        return sp.Mul(*[_dagger_secondquant(arg) for arg in reversed(expr.args)])
    if isinstance(expr, sp.Pow):
        return _dagger_secondquant(expr.base) ** expr.exp
    if expr.is_commutative:
        return sp.conjugate(expr)
    return expr


def _reduce_fermion_powers(expr: sp.Expr) -> sp.Expr:
    """Enforce nilpotency for identical fermionic creation/annihilation operators."""
    reduced = sp.expand(expr)

    def _replace(node: sp.Expr) -> sp.Expr:
        if isinstance(node, sp.Pow) and node.exp.is_integer and int(node.exp) >= 2:
            if isinstance(node.base, (AnnihilateFermion, CreateFermion)):
                return sp.Integer(0)
        return node

    return reduced.replace(lambda node: isinstance(node, sp.Pow), _replace)


def _is_partition_dummy(symbol: sp.Expr, *, above: bool) -> bool:
    if isinstance(symbol, sp.Dummy):
        assumptions = getattr(symbol, "assumptions0", {})
        if bool(assumptions.get("above_fermi" if above else "below_fermi", False)):
            return True
    if isinstance(symbol, sp.Symbol):
        name = str(symbol)
        if above and name == "_a":
            return True
        if (not above) and name == "_i":
            return True
    return False


def _extract_partition_delta(term: sp.Expr) -> tuple[tuple[sp.Expr, sp.Expr], str] | None:
    if not isinstance(term, sp.Mul):
        return None
    for factor in term.args:
        if not isinstance(factor, KroneckerDelta):
            continue
        left, right = factor.args
        if _is_partition_dummy(right, above=True):
            return (left, right), "above"
        if _is_partition_dummy(right, above=False):
            return (left, right), "below"
        if _is_partition_dummy(left, above=True):
            return (right, left), "above"
        if _is_partition_dummy(left, above=False):
            return (right, left), "below"
    return None


def _strip_delta(term: sp.Expr, delta_args: tuple[sp.Expr, sp.Expr]) -> sp.Expr:
    left, right = delta_args
    return term / KroneckerDelta(left, right)


def _collapse_partitioned_operator_terms(expr: sp.Expr) -> sp.Expr:
    if not isinstance(expr, sp.Add):
        return expr

    grouped: dict[sp.Expr, dict[str, tuple[sp.Expr, tuple[sp.Expr, sp.Expr]]]] = {}
    untouched: list[sp.Expr] = []

    for term in expr.args:
        partition = _extract_partition_delta(term)
        if partition is None:
            untouched.append(term)
            continue
        delta_args, kind = partition
        base_term = sp.simplify(_strip_delta(term, delta_args))
        key = sp.factor_terms(base_term)
        grouped.setdefault(key, {})[kind] = (term, delta_args)

    rebuilt = list(untouched)
    for base_term, entries in grouped.items():
        if "above" in entries and "below" in entries:
            rebuilt.append(base_term)
        else:
            rebuilt.extend(item[0] for item in entries.values())

    return sp.Add(*rebuilt) if rebuilt else sp.Integer(0)


def _collapse_partition_of_unity(expr: sp.Expr) -> sp.Expr:
    """Collapse Delta(x,_a) + Delta(x,_i) style structures to unity."""

    def _replace_add(node: sp.Expr) -> sp.Expr:
        if not isinstance(node, sp.Add) or len(node.args) != 2:
            return node
        first, second = node.args
        if not isinstance(first, KroneckerDelta) or not isinstance(second, KroneckerDelta):
            return node
        first_args = first.args
        second_args = second.args
        if first_args[0] != second_args[0]:
            return node
        left = first_args[1]
        right = second_args[1]
        if (_is_partition_dummy(left, above=True) and _is_partition_dummy(right, above=False)) or (
            _is_partition_dummy(left, above=False) and _is_partition_dummy(right, above=True)
        ):
            return sp.Integer(1)
        return node

    previous = None
    current = expr
    while previous != current:
        previous = current
        current = current.replace(lambda node: isinstance(node, sp.Add), _replace_add)
        current = _collapse_partitioned_operator_terms(current)
        current = sp.factor_terms(current)
    return current


def _normal_order_ladder_factors(factors: list[sp.Expr]) -> sp.Expr:
    """
    Normal-order a string of fermionic ladder operators with the CAR algebra.

    Rewrites adjacent ``F(i) Fd(j)`` pairs as ``KroneckerDelta(i, j) - Fd(j) F(i)``,
    kills adjacent identical operators (nilpotency), and sorts same-type runs
    canonically with anticommutation signs so equivalent strings cancel.
    Terminates because each rewrite strictly reduces a lexicographic measure
    (annihilator-before-creator inversions, then within-run disorder).
    """
    for position in range(len(factors) - 1):
        left, right = factors[position], factors[position + 1]
        if type(left) is type(right):
            if left.args[0] == right.args[0]:
                return sp.Integer(0)
            if sp.default_sort_key(left.args[0]) > sp.default_sort_key(right.args[0]):
                swapped = [*factors[:position], right, left, *factors[position + 2 :]]
                return -_normal_order_ladder_factors(swapped)
            continue
        if isinstance(left, AnnihilateFermion) and isinstance(right, CreateFermion):
            contracted = factors[:position] + factors[position + 2 :]
            swapped = [*factors[:position], right, left, *factors[position + 2 :]]
            return KroneckerDelta(left.args[0], right.args[0]) * _normal_order_ladder_factors(contracted) - _normal_order_ladder_factors(swapped)
    return sp.Mul(*factors) if factors else sp.Integer(1)


def normal_order_fermionic(expr: sp.Expr) -> sp.Expr:
    """
    Normal-order a polynomial in fermionic ladder operators without ``wicks``.

    This is the robust path for Hamiltonians with symbolic mode labels
    (e.g. ``"up"``/``"down"`` from ``custom_model``), where sympy's ``wicks``
    machinery fails while sorting repeated operators. Raises ``ValueError`` if
    the expression contains non-fermionic noncommutative factors.
    """
    expanded = sp.expand(_reduce_fermion_powers(_unwrap(expr)))
    if expanded == 0:
        return sp.Integer(0)
    terms = expanded.args if isinstance(expanded, sp.Add) else (expanded,)
    ordered_terms = []
    for term in terms:
        factors = term.args if isinstance(term, sp.Mul) else (term,)
        coefficient = sp.Integer(1)
        ladder: list[sp.Expr] = []
        for factor in factors:
            if factor.is_commutative:
                coefficient *= factor
            elif isinstance(factor, (AnnihilateFermion, CreateFermion)):
                ladder.append(factor)
            else:
                raise ValueError(f"normal_order_fermionic cannot handle factor {factor!r} in term {term}.")
        ordered_terms.append(coefficient * _normal_order_ladder_factors(ladder))
    return sp.expand(sp.Add(*ordered_terms))


def simplify_secondquant(
    expr: sp.Expr,
    *,
    expand: bool = True,
    simplify_kronecker_deltas: bool = True,
    simplify_dummies: bool = True,
) -> sp.Expr:
    """
    Simplify a second-quantized fermionic expression using Wick reordering.

    This is the main entry point users should call instead of interacting
    directly with ``wicks`` for routine operator algebra.
    """
    reduced = _reduce_fermion_powers(_unwrap(expr))
    if reduced == 0:
        return sp.Integer(0)
    try:
        simplified = wicks(
            reduced,
            expand=expand,
            simplify_kronecker_deltas=simplify_kronecker_deltas,
            simplify_dummies=simplify_dummies,
        )
    except AttributeError:
        # sympy's wicks machinery chokes on symbolic mode labels (its internal
        # sorting builds F(i)**2 powers it cannot handle); fall back to a
        # direct CAR normal-ordering that leaves explicit KroneckerDelta
        # factors for callers to collapse.
        simplified = normal_order_fermionic(reduced)
    simplified = sp.factor_terms(_reduce_fermion_powers(simplified))
    simplified = _collapse_partition_of_unity(simplified)
    return sp.expand(_reduce_fermion_powers(simplified))


def physical_simplify_fermionic(expr: sp.Expr | "SQObj") -> sp.Expr:
    """Presentation-oriented simplification for fermionic expressions."""
    simplified = simplify_secondquant(expr)

    def _combine_add_terms(node: sp.Expr) -> sp.Expr:
        if not isinstance(node, sp.Add):
            return node
        grouped: dict[tuple[sp.Expr, ...], list[sp.Expr]] = {}
        for term in node.args:
            if isinstance(term, sp.Mul):
                commutative = sp.Mul(*[arg for arg in term.args if arg.is_commutative])
                noncommutative = tuple(arg for arg in term.args if not arg.is_commutative)
            else:
                commutative = term if term.is_commutative else sp.Integer(1)
                noncommutative = tuple() if term.is_commutative else (term,)
            grouped.setdefault(noncommutative, []).append(commutative)

        rebuilt = []
        for noncommutative, coefficients in grouped.items():
            coefficient = _collapse_partition_of_unity(sp.factor_terms(sp.Add(*coefficients)))
            if coefficient == 0:
                continue
            op_part = sp.Mul(*noncommutative) if noncommutative else sp.Integer(1)
            rebuilt.append(sp.factor_terms(coefficient * op_part))
        return sp.Add(*rebuilt) if rebuilt else sp.Integer(0)

    previous = None
    current = simplified
    while previous != current:
        previous = current
        current = current.replace(lambda node: isinstance(node, sp.Add), _combine_add_terms)
        current = _collapse_partition_of_unity(sp.factor_terms(current))
    return current


def fermionic_commutator(a: sp.Expr | "SQObj", b: sp.Expr | "SQObj", simplify: bool = True) -> sp.Expr:
    """Return [a, b] with optional Wick simplification."""
    expr = sp.expand(_unwrap(a) * _unwrap(b) - _unwrap(b) * _unwrap(a))
    return simplify_secondquant(expr) if simplify else expr


def fermionic_anticommutator(a: sp.Expr | "SQObj", b: sp.Expr | "SQObj", simplify: bool = True) -> sp.Expr:
    """Return {a, b} with optional Wick simplification."""
    expr = sp.expand(_unwrap(a) * _unwrap(b) + _unwrap(b) * _unwrap(a))
    return simplify_secondquant(expr) if simplify else expr


def retarded_source(operator_a: sp.Expr | "SQObj", operator_b: sp.Expr | "SQObj") -> sp.Expr:
    """Return the EOM source term {A, B} used in fermionic retarded Green functions."""
    return fermionic_anticommutator(operator_a, operator_b, simplify=True)


def latex_expr(expr: sp.Expr | "SQObj") -> str:
    """Export a symbolic expression to LaTeX."""
    return sp.latex(_unwrap(expr))


def fermionic_eom_rhs(operator: sp.Expr | "SQObj", hamiltonian: sp.Expr | "SQObj") -> sp.Expr:
    """Return the simplified fermionic EOM right-hand side [operator, H]."""
    return fermionic_commutator(operator, hamiltonian, simplify=True)


def build_fermionic_eom_system(
    operators: Sequence[sp.Expr | "SQObj"],
    hamiltonian: sp.Expr | "SQObj",
) -> FermionicEOMResult:
    """
    Build the projected EOM system for a fermionic operator basis.

    The resulting matrix ``M`` satisfies:
    [O_i, H] = sum_j M_ij O_j + residual_i
    """
    operator_exprs = [_unwrap(operator) for operator in operators]
    hamiltonian_expr = _unwrap(hamiltonian)
    rows = []
    residuals = []
    for operator in operator_exprs:
        rhs = fermionic_eom_rhs(operator, hamiltonian_expr)
        coeffs, residual = decompose_in_basis(rhs, operator_exprs)
        rows.append(coeffs.T)
        residuals.append(sp.simplify(sp.expand(residual)))
    eom_matrix = sp.Matrix.vstack(*rows) if rows else sp.Matrix([])
    return FermionicEOMResult(operators=operator_exprs, eom_matrix=eom_matrix, residuals=residuals)


def retarded_green_from_fermionic_eom(
    eom_result: FermionicEOMResult,
    operators_left: Sequence[sp.Expr | "SQObj"],
    operators_right: Sequence[sp.Expr | "SQObj"],
    omega: sp.Expr,
    eta: sp.Expr,
) -> sp.Matrix:
    """
    Solve the closed fermionic EOM system in frequency space.

    ((omega + i eta) I - M) G^r = C
    with C_ij = {A_i, B_j}.
    """
    if not eom_result.is_closed:
        raise ValueError("Fermionic EOM basis is not closed; extend the basis or truncate it explicitly.")
    left_exprs = [_unwrap(operator) for operator in operators_left]
    right_exprs = [_unwrap(operator) for operator in operators_right]
    source = sp.Matrix(
        [[retarded_source(left, right) for right in right_exprs] for left in left_exprs]
    )
    identity = sp.eye(eom_result.eom_matrix.shape[0])
    lhs = (omega + sp.I * eta) * identity - eom_result.eom_matrix
    return lhs.LUsolve(source)


@dataclass(frozen=True)
class SQObj(SymbolicOperator):
    """Small QuTiP-inspired wrapper for symbolic second-quantized operators."""

    expr: sp.Expr
    statistics: str = field(init=False, default="fermion")

    @staticmethod
    def wrap(expr: sp.Expr | "SQObj") -> "SQObj":
        return _wrap(expr)

    @staticmethod
    def destroy(index: sp.Expr) -> "SQObj":
        return SQObj(annihilate(index))

    @staticmethod
    def create(index: sp.Expr) -> "SQObj":
        return SQObj(create(index))

    @staticmethod
    def num(index: sp.Expr) -> "SQObj":
        return SQObj(number_operator(index, simplify=False))

    def dag(self) -> "SQObj":
        return SQObj(_dagger_secondquant(self.expr))

    def comm(self, other: sp.Expr | "SQObj", simplify: bool = True) -> "SQObj":
        return SQObj(fermionic_commutator(self.expr, other, simplify=simplify))

    def anticomm(self, other: sp.Expr | "SQObj", simplify: bool = True) -> "SQObj":
        return SQObj(fermionic_anticommutator(self.expr, other, simplify=simplify))

    def eom_rhs(self, hamiltonian: sp.Expr | "SQObj") -> "SQObj":
        return SQObj(fermionic_eom_rhs(self.expr, hamiltonian))

    def eom(self, hamiltonian: sp.Expr | "SQObj", basis: Sequence[sp.Expr | "SQObj"] | None = None) -> FermionicEOMResult:
        basis_exprs = [self.expr] if basis is None else list(basis)
        return build_fermionic_eom_system(basis_exprs, hamiltonian)

    def retarded(
        self,
        right_operator: sp.Expr | "SQObj",
        hamiltonian: sp.Expr | "SQObj",
        omega: sp.Expr,
        eta: sp.Expr,
        basis: Sequence[sp.Expr | "SQObj"] | None = None,
    ) -> sp.Expr:
        basis_exprs = [self.expr] if basis is None else list(basis)
        eom_result = build_fermionic_eom_system(basis_exprs, hamiltonian)
        matrix = retarded_green_from_fermionic_eom(
            eom_result,
            basis_exprs,
            [_unwrap(right_operator)],
            omega=omega,
            eta=eta,
        )
        return matrix[0, 0]

    def simplify(self) -> "SQObj":
        return SQObj(simplify_secondquant(self.expr))

    def latex(self) -> str:
        return latex_expr(self.expr)

    def doit(self) -> sp.Expr:
        return self.expr

    def __add__(self, other: sp.Expr | "SQObj") -> "SQObj":
        return SQObj(self.expr + _unwrap(other))

    def __radd__(self, other: sp.Expr | "SQObj") -> "SQObj":
        return SQObj(_unwrap(other) + self.expr)

    def __sub__(self, other: sp.Expr | "SQObj") -> "SQObj":
        return SQObj(self.expr - _unwrap(other))

    def __rsub__(self, other: sp.Expr | "SQObj") -> "SQObj":
        return SQObj(_unwrap(other) - self.expr)

    def __mul__(self, other: sp.Expr | "SQObj") -> "SQObj":
        return SQObj(self.expr * _unwrap(other))

    def __rmul__(self, other: sp.Expr | "SQObj") -> "SQObj":
        return SQObj(_unwrap(other) * self.expr)

    def __truediv__(self, other: sp.Expr | "SQObj") -> "SQObj":
        return SQObj(self.expr / _unwrap(other))

    def __neg__(self) -> "SQObj":
        return SQObj(-self.expr)

    def __repr__(self) -> str:
        return f"SQObj({self.expr!r})"
