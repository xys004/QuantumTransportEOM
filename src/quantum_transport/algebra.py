"""Symbolic operator algebra helpers."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import sympy as sp


def commutator(a: sp.Expr, b: sp.Expr, simplify: bool = True) -> sp.Expr:
    """Return [a, b] = a*b - b*a."""
    expr = sp.expand(a * b - b * a)
    return sp.simplify(expr) if simplify else expr


def anticommutator(a: sp.Expr, b: sp.Expr, simplify: bool = True) -> sp.Expr:
    """Return {a, b} = a*b + b*a."""
    expr = sp.expand(a * b + b * a)
    return sp.simplify(expr) if simplify else expr


def decompose_in_basis(
    expr: sp.Expr,
    basis: Sequence[sp.Expr],
    *,
    commutative_coefficients: bool = False,
) -> Tuple[sp.Matrix, sp.Expr]:
    """
    Decompose ``expr`` into a linear combination of noncommutative basis elements.

    Returns ``coeffs, residual`` such that:
    expr = sum_j coeffs[j] * basis[j] + residual

    With ``commutative_coefficients=True`` only scalar (commutative) prefactors
    are accepted as coefficients; operator-valued prefixes stay in the residual.
    Mixed fermion-boson systems keep the default, where bosonic operators may
    legitimately appear as coefficients of fermionic basis elements.
    """
    expanded = sp.expand(expr)
    coeffs: List[sp.Expr] = []
    residual = expanded
    for op in basis:
        coeff = expanded.coeff(op)
        if commutative_coefficients and coeff != 0:
            terms = coeff.args if isinstance(coeff, sp.Add) else (coeff,)
            coeff = sp.Add(*[term for term in terms if term.is_commutative])
        coeffs.append(coeff)
        residual -= coeff * op
    residual = sp.simplify(sp.expand(residual))
    return sp.Matrix(coeffs), residual


def operator_symbols(prefix: str, size: int) -> Tuple[sp.Symbol, ...]:
    """Create noncommutative symbols prefix0, prefix1, ..."""
    return sp.symbols(f"{prefix}0:{size}", commutative=False)
