"""Equation-of-motion (EOM) utilities for operator bases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import sympy as sp

from .algebra import anticommutator, commutator, decompose_in_basis


@dataclass
class EOMClosureResult:
    operators: Sequence[sp.Expr]
    eom_matrix: sp.Matrix
    residuals: Sequence[sp.Expr]

    @property
    def is_closed(self) -> bool:
        return all(r == 0 for r in self.residuals)


def check_eom_closure(operators: Sequence[sp.Expr], hamiltonian: sp.Expr) -> EOMClosureResult:
    """
    Build EOM matrix ``M`` for ``[O_i, H] = sum_j M_ij O_j + residual_i``.
    """
    rows = []
    residuals = []
    for op in operators:
        expr = commutator(op, hamiltonian)
        coeffs, residual = decompose_in_basis(expr, operators)
        rows.append(coeffs.T)
        residuals.append(residual)
    eom_matrix = sp.Matrix.vstack(*rows) if rows else sp.Matrix([])
    return EOMClosureResult(operators=operators, eom_matrix=eom_matrix, residuals=residuals)


def retarded_green_from_eom(
    closure: EOMClosureResult,
    source_matrix: sp.Matrix,
    omega: sp.Symbol | sp.Expr,
    eta: sp.Symbol | sp.Expr,
) -> sp.Matrix:
    """
    Solve the closed EOM system in frequency domain:
    ((omega + i*eta)I - M) G^r(omega) = C
    """
    if not closure.is_closed:
        raise ValueError("EOM basis is not closed; increase basis or use truncation.")
    m = closure.eom_matrix
    identity = sp.eye(m.shape[0])
    lhs = (omega + sp.I * eta) * identity - m
    return lhs.LUsolve(source_matrix)


def source_anticommutator_matrix(
    operators_left: Sequence[sp.Expr], operators_right: Sequence[sp.Expr]
) -> sp.Matrix:
    """Return C_ij = <{A_i, B_j}> as symbolic anticommutators."""
    return sp.Matrix(
        [[anticommutator(a, b, simplify=True) for b in operators_right] for a in operators_left]
    )
