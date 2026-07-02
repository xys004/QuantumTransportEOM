"""Bosonic symbolic helpers with a QuTiP-like ergonomic layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import sympy as sp
from sympy import KroneckerDelta
from sympy.physics.secondquant import AnnihilateBoson, B, Bd, CreateBoson

from .algebra import decompose_in_basis
from .symbolic_base import SymbolicOperator


@dataclass
class BosonicEOMResult:
    operators: Sequence[sp.Expr]
    eom_matrix: sp.Matrix
    residuals: Sequence[sp.Expr]

    @property
    def is_closed(self) -> bool:
        return all(sp.simplify(residual) == 0 for residual in self.residuals)


def annihilate_boson(index: sp.Expr) -> sp.Expr:
    """Return the bosonic annihilation operator b(index)."""
    return B(index)


def create_boson(index: sp.Expr) -> sp.Expr:
    """Return the bosonic creation operator b_dagger(index)."""
    return Bd(index)


def number_operator_boson(index: sp.Expr) -> sp.Expr:
    """Return n_index = b_dagger(index) b(index)."""
    return create_boson(index) * annihilate_boson(index)


def create_b(index: sp.Expr) -> "BQObj":
    """QuTiP-like constructor for a bosonic creation operator."""
    return BQObj(create_boson(index))


def destroy_b(index: sp.Expr) -> "BQObj":
    """QuTiP-like constructor for a bosonic annihilation operator."""
    return BQObj(annihilate_boson(index))


def num_b(index: sp.Expr) -> "BQObj":
    """QuTiP-like constructor for a bosonic number operator."""
    return BQObj(number_operator_boson(index))


def bqobj(expr: sp.Expr | "BQObj") -> "BQObj":
    """Wrap a bosonic SymPy expression in a small operator object."""
    return expr if isinstance(expr, BQObj) else BQObj(expr)


def _unwrap(expr: sp.Expr | "BQObj") -> sp.Expr:
    return expr.expr if isinstance(expr, BQObj) else expr


def _dagger_boson(expr: sp.Expr) -> sp.Expr:
    if isinstance(expr, AnnihilateBoson):
        return CreateBoson(expr.args[0])
    if isinstance(expr, CreateBoson):
        return AnnihilateBoson(expr.args[0])
    if isinstance(expr, sp.Add):
        return sp.Add(*[_dagger_boson(arg) for arg in expr.args])
    if isinstance(expr, sp.Mul):
        return sp.Mul(*[_dagger_boson(arg) for arg in reversed(expr.args)])
    if isinstance(expr, sp.Pow):
        return _dagger_boson(expr.base) ** expr.exp
    if expr.is_commutative:
        return sp.conjugate(expr)
    return expr


def _bosonic_delta(left: sp.Expr, right: sp.Expr) -> sp.Expr:
    return sp.Integer(1) if left == right else KroneckerDelta(left, right)


def simplify_bosonic(expr: sp.Expr | "BQObj") -> sp.Expr:
    """Light simplification for bosonic second-quantized expressions."""
    return sp.expand(_unwrap(expr))


def _bosonic_commutator_expr(left: sp.Expr, right: sp.Expr) -> sp.Expr:
    left = simplify_bosonic(left)
    right = simplify_bosonic(right)

    if left.is_commutative or right.is_commutative:
        return sp.Integer(0)
    if left == right:
        return sp.Integer(0)
    if isinstance(left, sp.Add):
        return sp.Add(*[_bosonic_commutator_expr(term, right) for term in left.args])
    if isinstance(right, sp.Add):
        return sp.Add(*[_bosonic_commutator_expr(left, term) for term in right.args])
    if isinstance(left, sp.Mul):
        first = left.args[0]
        rest = sp.Mul(*left.args[1:]) if len(left.args) > 1 else sp.Integer(1)
        return simplify_bosonic(first * _bosonic_commutator_expr(rest, right) + _bosonic_commutator_expr(first, right) * rest)
    if isinstance(right, sp.Mul):
        first = right.args[0]
        rest = sp.Mul(*right.args[1:]) if len(right.args) > 1 else sp.Integer(1)
        return simplify_bosonic(_bosonic_commutator_expr(left, first) * rest + first * _bosonic_commutator_expr(left, rest))
    if isinstance(left, AnnihilateBoson) and isinstance(right, CreateBoson):
        return _bosonic_delta(left.args[0], right.args[0])
    if isinstance(left, CreateBoson) and isinstance(right, AnnihilateBoson):
        return -_bosonic_delta(left.args[0], right.args[0])
    if isinstance(left, (AnnihilateBoson, CreateBoson)) and isinstance(right, (AnnihilateBoson, CreateBoson)):
        return sp.Integer(0)
    return simplify_bosonic(left * right - right * left)


def bosonic_commutator(left: sp.Expr | "BQObj", right: sp.Expr | "BQObj", simplify: bool = True) -> sp.Expr:
    """Return [left, right] for bosonic operators."""
    expr = _bosonic_commutator_expr(_unwrap(left), _unwrap(right))
    return simplify_bosonic(expr) if simplify else expr


def bosonic_retarded_source(operator_a: sp.Expr | "BQObj", operator_b: sp.Expr | "BQObj") -> sp.Expr:
    """Return the bosonic retarded source term [A, B]."""
    return bosonic_commutator(operator_a, operator_b, simplify=True)


def bosonic_eom_rhs(operator: sp.Expr | "BQObj", hamiltonian: sp.Expr | "BQObj") -> sp.Expr:
    """Return the simplified bosonic EOM right-hand side [operator, H]."""
    return bosonic_commutator(operator, hamiltonian, simplify=True)


def build_bosonic_eom_system(
    operators: Sequence[sp.Expr | "BQObj"],
    hamiltonian: sp.Expr | "BQObj",
) -> BosonicEOMResult:
    """Build the projected bosonic EOM system for a chosen operator basis."""
    operator_exprs = [_unwrap(operator) for operator in operators]
    hamiltonian_expr = _unwrap(hamiltonian)
    rows = []
    residuals = []
    for operator in operator_exprs:
        rhs = bosonic_eom_rhs(operator, hamiltonian_expr)
        coeffs, residual = decompose_in_basis(rhs, operator_exprs)
        rows.append(coeffs.T)
        residuals.append(sp.simplify(sp.expand(residual)))
    eom_matrix = sp.Matrix.vstack(*rows) if rows else sp.Matrix([])
    return BosonicEOMResult(operators=operator_exprs, eom_matrix=eom_matrix, residuals=residuals)


def retarded_green_from_bosonic_eom(
    eom_result: BosonicEOMResult,
    operators_left: Sequence[sp.Expr | "BQObj"],
    operators_right: Sequence[sp.Expr | "BQObj"],
    omega: sp.Expr,
    eta: sp.Expr,
) -> sp.Matrix:
    """Solve the closed bosonic EOM system in frequency space."""
    if not eom_result.is_closed:
        raise ValueError("Bosonic EOM basis is not closed; extend the basis or truncate it explicitly.")
    left_exprs = [_unwrap(operator) for operator in operators_left]
    right_exprs = [_unwrap(operator) for operator in operators_right]
    source = sp.Matrix(
        [[bosonic_retarded_source(left, right) for right in right_exprs] for left in left_exprs]
    )
    identity = sp.eye(eom_result.eom_matrix.shape[0])
    lhs = (omega + sp.I * eta) * identity - eom_result.eom_matrix
    return lhs.LUsolve(source)


def latex_boson(expr: sp.Expr | "BQObj") -> str:
    return sp.latex(_unwrap(expr))


@dataclass(frozen=True)
class BQObj(SymbolicOperator):
    """Small QuTiP-inspired wrapper for symbolic bosonic operators."""

    expr: sp.Expr
    statistics: str = field(init=False, default="boson")

    @staticmethod
    def destroy(index: sp.Expr) -> "BQObj":
        return destroy_b(index)

    @staticmethod
    def create(index: sp.Expr) -> "BQObj":
        return create_b(index)

    @staticmethod
    def num(index: sp.Expr) -> "BQObj":
        return num_b(index)

    def dag(self) -> "BQObj":
        return BQObj(_dagger_boson(self.expr))

    def comm(self, other: sp.Expr | "BQObj", simplify: bool = True) -> "BQObj":
        return BQObj(bosonic_commutator(self.expr, other, simplify=simplify))

    def eom_rhs(self, hamiltonian: sp.Expr | "BQObj") -> "BQObj":
        return BQObj(bosonic_eom_rhs(self.expr, hamiltonian))

    def eom(self, hamiltonian: sp.Expr | "BQObj", basis: Sequence[sp.Expr | "BQObj"] | None = None) -> BosonicEOMResult:
        basis_exprs = [self.expr] if basis is None else list(basis)
        return build_bosonic_eom_system(basis_exprs, hamiltonian)

    def retarded(
        self,
        right_operator: sp.Expr | "BQObj",
        hamiltonian: sp.Expr | "BQObj",
        omega: sp.Expr,
        eta: sp.Expr,
        basis: Sequence[sp.Expr | "BQObj"] | None = None,
    ) -> sp.Expr:
        basis_exprs = [self.expr] if basis is None else list(basis)
        eom_result = build_bosonic_eom_system(basis_exprs, hamiltonian)
        matrix = retarded_green_from_bosonic_eom(
            eom_result,
            basis_exprs,
            [_unwrap(right_operator)],
            omega=omega,
            eta=eta,
        )
        return matrix[0, 0]

    def simplify(self) -> "BQObj":
        return BQObj(simplify_bosonic(self.expr))

    def latex(self) -> str:
        return latex_boson(self.expr)

    def doit(self) -> sp.Expr:
        return self.expr

    def __add__(self, other: sp.Expr | "BQObj") -> "BQObj":
        return BQObj(self.expr + _unwrap(other))

    def __radd__(self, other: sp.Expr | "BQObj") -> "BQObj":
        return BQObj(_unwrap(other) + self.expr)

    def __sub__(self, other: sp.Expr | "BQObj") -> "BQObj":
        return BQObj(self.expr - _unwrap(other))

    def __rsub__(self, other: sp.Expr | "BQObj") -> "BQObj":
        return BQObj(_unwrap(other) - self.expr)

    def __mul__(self, other: sp.Expr | "BQObj") -> "BQObj":
        return BQObj(self.expr * _unwrap(other))

    def __rmul__(self, other: sp.Expr | "BQObj") -> "BQObj":
        return BQObj(_unwrap(other) * self.expr)

    def __truediv__(self, other: sp.Expr | "BQObj") -> "BQObj":
        return BQObj(self.expr / _unwrap(other))

    def __neg__(self) -> "BQObj":
        return BQObj(-self.expr)

    def __repr__(self) -> str:
        return f"BQObj({self.expr!r})"
