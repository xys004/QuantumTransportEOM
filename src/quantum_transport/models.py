"""Predefined symbolic models for EOM workflows."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import sympy as sp
from sympy import KroneckerDelta
from sympy.physics.secondquant import AnnihilateBoson, AnnihilateFermion, CreateBoson, CreateFermion

from .algebra import decompose_in_basis
from .bosonic import build_bosonic_eom_system, bosonic_eom_rhs, retarded_green_from_bosonic_eom, destroy_b, num_b
from .secondquant import (
    FermionicEOMResult,
    build_fermionic_eom_system,
    destroy,
    fermionic_eom_rhs,
    normal_order_fermionic,
    num,
    physical_simplify_fermionic,
    retarded_green_from_fermionic_eom,
)
from .symbolic_base import eom_system_latex


def _unwrap_symbolic(obj: Any) -> sp.Expr:
    return obj.expr if hasattr(obj, "expr") else obj


def _basis_key(expr: sp.Expr) -> str:
    return sp.srepr(expr)


def _unique_basis(operators: Sequence[Any]) -> list[sp.Expr]:
    unique: list[sp.Expr] = []
    seen: set[str] = set()
    for operator in operators:
        expr = _unwrap_symbolic(operator)
        key = _basis_key(expr)
        if key in seen:
            continue
        seen.add(key)
        unique.append(expr)
    return unique


def _extract_operator_monomials(expr: sp.Expr) -> list[sp.Expr]:
    expanded = sp.expand(_unwrap_symbolic(expr))
    if expanded == 0:
        return []
    terms = expanded.args if isinstance(expanded, sp.Add) else (expanded,)
    extracted: list[sp.Expr] = []
    seen: set[str] = set()
    for term in terms:
        factors = term.args if isinstance(term, sp.Mul) else (term,)
        noncommutative = [factor for factor in factors if not factor.is_commutative]
        if not noncommutative:
            continue
        operator = sp.Mul(*noncommutative)
        key = _basis_key(operator)
        if key in seen:
            continue
        seen.add(key)
        extracted.append(operator)
    return extracted


def _project_known_rhs(operators: Sequence[sp.Expr], rhs_by_operator: Mapping[sp.Expr, Mapping[sp.Expr, sp.Expr]]) -> GenericEOMResult:
    operator_exprs = [_unwrap_symbolic(item) for item in operators]
    rows = []
    residuals = []
    for operator in operator_exprs:
        if operator not in rhs_by_operator:
            raise ValueError(f"Unsupported projected operator: {operator}")
        coefficient_map = rhs_by_operator[operator]
        coeffs = [sp.simplify(coefficient_map.get(target, 0)) for target in operator_exprs]
        represented = sp.Add(*[coeff * target for coeff, target in zip(coeffs, operator_exprs)])
        full_rhs = sp.Add(*[coeff * target for target, coeff in coefficient_map.items()])
        rows.append(sp.Matrix(coeffs).T)
        residuals.append(sp.simplify(sp.expand(full_rhs - represented)))
    eom_matrix = sp.Matrix.vstack(*rows) if rows else sp.Matrix([])
    return GenericEOMResult(operators=operator_exprs, eom_matrix=eom_matrix, residuals=residuals)


@dataclass
class GenericEOMResult:
    operators: Sequence[sp.Expr]
    eom_matrix: sp.Matrix
    residuals: Sequence[sp.Expr]

    @property
    def is_closed(self) -> bool:
        return all(sp.simplify(residual) == 0 for residual in self.residuals)

    def _repr_latex_(self) -> str:
        return eom_system_latex(self.operators, self.eom_matrix, self.residuals)


@dataclass
class ModelEOMAnalysis:
    success: bool
    result: Any | None = None
    error: Exception | None = None

    @property
    def is_closed(self) -> bool:
        return bool(self.success and self.result is not None and self.result.is_closed)


def build_mixed_eom_system(operators: Sequence[Any], hamiltonian: Any) -> GenericEOMResult:
    operator_exprs = [_unwrap_symbolic(item) for item in operators]
    hamiltonian_expr = _unwrap_symbolic(hamiltonian)
    rows = []
    residuals = []
    for operator in operator_exprs:
        rhs = sp.expand(operator * hamiltonian_expr - hamiltonian_expr * operator)
        coeffs, residual = decompose_in_basis(rhs, operator_exprs)
        rows.append(coeffs.T)
        residuals.append(sp.simplify(sp.expand(residual)))
    eom_matrix = sp.Matrix.vstack(*rows) if rows else sp.Matrix([])
    return GenericEOMResult(operators=operator_exprs, eom_matrix=eom_matrix, residuals=residuals)


def build_anderson_atomic_eom_system(
    operators: Sequence[Any],
    epsilon_up: sp.Expr,
    epsilon_down: sp.Expr,
    interaction_u: sp.Expr,
    indices: tuple[Any, Any] = ("up", "down"),
) -> GenericEOMResult:
    up, down = indices
    d_up = destroy(up).doit()
    d_down = destroy(down).doit()
    n_up = num(up).doit()
    n_down = num(down).doit()
    composite_up = n_down * d_up
    composite_down = n_up * d_down

    rhs_by_operator = {
        d_up: {d_up: epsilon_up, composite_up: interaction_u},
        d_down: {d_down: epsilon_down, composite_down: interaction_u},
        composite_up: {composite_up: epsilon_up + interaction_u},
        composite_down: {composite_down: epsilon_down + interaction_u},
    }
    return _project_known_rhs(operators, rhs_by_operator)


def build_anderson_hartree_fock_eom_system(
    operators: Sequence[Any],
    epsilon_up: sp.Expr,
    epsilon_down: sp.Expr,
    interaction_u: sp.Expr,
    occupations: Mapping[str, sp.Expr] | None = None,
    indices: tuple[Any, Any] = ("up", "down"),
) -> GenericEOMResult:
    up, down = indices
    d_up = destroy(up).doit()
    d_down = destroy(down).doit()
    occupations = {} if occupations is None else dict(occupations)
    n_up_avg = occupations.get("up", sp.Symbol("n_up_avg", real=True))
    n_down_avg = occupations.get("down", sp.Symbol("n_down_avg", real=True))

    rhs_by_operator = {
        d_up: {d_up: epsilon_up + interaction_u * n_down_avg},
        d_down: {d_down: epsilon_down + interaction_u * n_up_avg},
    }
    return _project_known_rhs(operators, rhs_by_operator)


def build_anderson_hubbard_i_eom_system(
    operators: Sequence[Any],
    epsilon_up: sp.Expr,
    epsilon_down: sp.Expr,
    interaction_u: sp.Expr,
    indices: tuple[Any, Any] = ("up", "down"),
) -> GenericEOMResult:
    up, down = indices
    d_up = destroy(up).doit()
    d_down = destroy(down).doit()
    n_up = num(up).doit()
    n_down = num(down).doit()
    composite_up = n_down * d_up
    composite_down = n_up * d_down

    rhs_by_operator = {
        d_up: {d_up: epsilon_up, composite_up: interaction_u},
        d_down: {d_down: epsilon_down, composite_down: interaction_u},
        composite_up: {composite_up: epsilon_up + interaction_u},
        composite_down: {composite_down: epsilon_down + interaction_u},
    }
    return _project_known_rhs(operators, rhs_by_operator)


def anderson_hubbard_i_green_function(
    spin: str,
    omega: sp.Expr,
    eta: sp.Expr,
    epsilon_up: sp.Expr,
    epsilon_down: sp.Expr,
    interaction_u: sp.Expr,
    occupations: Mapping[str, sp.Expr] | None = None,
) -> sp.Expr:
    occupations = {} if occupations is None else dict(occupations)
    n_up_avg = occupations.get("up", sp.Symbol("n_up_avg", real=True))
    n_down_avg = occupations.get("down", sp.Symbol("n_down_avg", real=True))

    if spin == "up":
        epsilon = epsilon_up
        opposite_occupation = n_down_avg
    elif spin == "down":
        epsilon = epsilon_down
        opposite_occupation = n_up_avg
    else:
        raise ValueError(f"Unsupported Anderson spin label for Hubbard-I: {spin}")

    z = omega + sp.I * eta
    return sp.simplify(
        (1 - opposite_occupation) / (z - epsilon)
        + opposite_occupation / (z - epsilon - interaction_u)
    )


@dataclass
class SymbolicModel:
    name: str
    statistics: str
    hamiltonian: Any
    basis: list[Any]
    operators: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    eom_builder: Callable[[Sequence[Any], Any], Any] | None = None

    def eom(
        self,
        basis: Sequence[Any] | None = None,
        *,
        auto_expand_steps: int = 0,
        truncation: str | None = None,
        truncation_params: Mapping[str, Any] | None = None,
    ):
        working_basis = _unique_basis(self.basis if basis is None else list(basis))
        if auto_expand_steps > 0:
            working_basis = self.expand_basis(basis=working_basis, max_steps=auto_expand_steps)
        if truncation is not None:
            return self._build_truncated_eom(working_basis, truncation=truncation, truncation_params=truncation_params)
        if self.eom_builder is not None:
            return self.eom_builder(working_basis, self.hamiltonian)
        if self.statistics == "fermion":
            return build_fermionic_eom_system(working_basis, self.hamiltonian)
        if self.statistics == "boson":
            return build_bosonic_eom_system(working_basis, self.hamiltonian)
        return build_mixed_eom_system(working_basis, self.hamiltonian)

    def analyze_eom(
        self,
        basis: Sequence[Any] | None = None,
        *,
        auto_expand_steps: int = 0,
        truncation: str | None = None,
        truncation_params: Mapping[str, Any] | None = None,
    ) -> ModelEOMAnalysis:
        try:
            return ModelEOMAnalysis(
                success=True,
                result=self.eom(
                    basis=basis,
                    auto_expand_steps=auto_expand_steps,
                    truncation=truncation,
                    truncation_params=truncation_params,
                ),
            )
        except Exception as exc:
            return ModelEOMAnalysis(success=False, error=exc)

    def expand_basis(self, basis: Sequence[Any] | None = None, max_steps: int = 1) -> list[sp.Expr]:
        current_basis = _unique_basis(self.basis if basis is None else list(basis))
        if self.name == "anderson_impurity":
            up, down = self.metadata["indices"]
            d_up = destroy(up).doit()
            d_down = destroy(down).doit()
            n_up = num(up).doit()
            n_down = num(down).doit()
            hierarchy = [n_down * d_up, n_up * d_down]
            return _unique_basis([*current_basis, *hierarchy]) if max_steps > 0 else current_basis
        for _ in range(max_steps):
            result = self.eom(basis=current_basis)
            discovered: list[sp.Expr] = []
            for residual in result.residuals:
                discovered.extend(_extract_operator_monomials(residual))
            next_basis = _unique_basis([*current_basis, *discovered])
            if len(next_basis) == len(current_basis):
                break
            current_basis = next_basis
        return current_basis

    def _build_truncated_eom(
        self,
        working_basis: Sequence[Any],
        *,
        truncation: str,
        truncation_params: Mapping[str, Any] | None = None,
    ):
        if self.name == "anderson_impurity" and sp.simplify(self.metadata.get("spin_flip", 0)) != 0:
            raise NotImplementedError(
                "Anderson truncations 'hartree' and 'hubbard_i' do not support spin_flip; "
                "use the untruncated EOM with an explicit basis."
            )
        if self.name == "anderson_impurity" and truncation in {"hartree", "hartree_fock", "mean_field", "collinear_hartree"}:
            parameters = self.metadata
            return build_anderson_hartree_fock_eom_system(
                working_basis,
                parameters["epsilon_up"],
                parameters["epsilon_down"],
                parameters["interaction_u"],
                occupations=(truncation_params or {}).get("occupations"),
                indices=parameters["indices"],
            )
        if self.name == "anderson_impurity" and truncation in {"hubbard_i", "hubbard-I", "hubbard1"}:
            parameters = self.metadata
            expanded_basis = self.expand_basis(basis=working_basis, max_steps=1)
            return build_anderson_hubbard_i_eom_system(
                expanded_basis,
                parameters["epsilon_up"],
                parameters["epsilon_down"],
                parameters["interaction_u"],
                indices=parameters["indices"],
            )
        if truncation in {"hartree", "hartree_fock", "mean_field", "collinear_hartree"} and self.statistics == "fermion":
            params = dict(truncation_params or {})
            return build_fermionic_hartree_eom_system(
                working_basis,
                self.hamiltonian,
                occupations=params.get("occupations"),
                include_fock=bool(params.get("include_fock", False)),
            )
        raise NotImplementedError(f"Truncation '{truncation}' is not implemented for model '{self.name}'.")

    def retarded(
        self,
        left: Any,
        right: Any,
        omega: sp.Expr,
        eta: sp.Expr,
        basis: Sequence[Any] | None = None,
        *,
        auto_expand_steps: int = 0,
        truncation: str | None = None,
        truncation_params: Mapping[str, Any] | None = None,
    ):
        working_basis = _unique_basis(self.basis if basis is None else list(basis))
        if auto_expand_steps > 0:
            working_basis = self.expand_basis(basis=working_basis, max_steps=auto_expand_steps)
        if self.statistics == "mixed":
            raise NotImplementedError("Mixed retarded Green functions need an explicit source convention and basis closure strategy.")
        left_expr = _unwrap_symbolic(left)
        right_expr = _unwrap_symbolic(right)
        if self.name == "anderson_impurity" and self.statistics == "fermion":
            parameters = self.metadata
            spin_flip = parameters.get("spin_flip", 0)
            interaction_u = parameters["interaction_u"]
            if sp.simplify(spin_flip) != 0 and sp.simplify(interaction_u) == 0 and truncation in {None, "noninteracting", "free"}:
                up, down = parameters["indices"]
                d_up = destroy(up).doit()
                d_down = destroy(down).doit()
                d_up_dag = destroy(up).dag().doit()
                d_down_dag = destroy(down).dag().doit()
                z = omega + sp.I * eta
                inverse = sp.Matrix(
                    [
                        [z - parameters["epsilon_up"], -spin_flip],
                        [-sp.conjugate(spin_flip), z - parameters["epsilon_down"]],
                    ]
                ).inv()
                entries = {
                    (d_up, d_up_dag): inverse[0, 0],
                    (d_up, d_down_dag): inverse[0, 1],
                    (d_down, d_up_dag): inverse[1, 0],
                    (d_down, d_down_dag): inverse[1, 1],
                }
                return sp.simplify(entries.get((left_expr, right_expr), 0))
        basis_exprs = [_unwrap_symbolic(item) for item in working_basis]
        if left_expr not in basis_exprs:
            working_basis = [*list(working_basis), left_expr]
            basis_exprs.append(left_expr)
        eom_result = self.eom(
            basis=working_basis,
            truncation=truncation,
            truncation_params=truncation_params,
        )
        left_index = basis_exprs.index(left_expr)
        if self.name == "anderson_impurity" and truncation in {"hartree", "hartree_fock", "mean_field", "collinear_hartree"} and self.statistics == "fermion":
            up, down = self.metadata["indices"]
            d_up = destroy(up).doit()
            d_down = destroy(down).doit()
            d_up_dag = destroy(up).dag().doit()
            d_down_dag = destroy(down).dag().doit()
            if left_expr == d_up and right_expr == d_up_dag:
                return sp.simplify(1 / ((omega + sp.I * eta) - eom_result.eom_matrix[left_index, left_index]))
            if left_expr == d_down and right_expr == d_down_dag:
                return sp.simplify(1 / ((omega + sp.I * eta) - eom_result.eom_matrix[left_index, left_index]))
            return sp.Integer(0)
        if self.name == "anderson_impurity" and truncation in {"hubbard_i", "hubbard-I", "hubbard1"} and self.statistics == "fermion":
            up, down = self.metadata["indices"]
            d_up = destroy(up).doit()
            d_down = destroy(down).doit()
            d_up_dag = destroy(up).dag().doit()
            d_down_dag = destroy(down).dag().doit()
            occupations = (truncation_params or {}).get("occupations")
            if left_expr == d_up and right_expr == d_up_dag:
                return anderson_hubbard_i_green_function(
                    "up",
                    omega,
                    eta,
                    self.metadata["epsilon_up"],
                    self.metadata["epsilon_down"],
                    self.metadata["interaction_u"],
                    occupations=occupations,
                )
            if left_expr == d_down and right_expr == d_down_dag:
                return anderson_hubbard_i_green_function(
                    "down",
                    omega,
                    eta,
                    self.metadata["epsilon_up"],
                    self.metadata["epsilon_down"],
                    self.metadata["interaction_u"],
                    occupations=occupations,
                )
            return sp.Integer(0)
        modes = [*self.metadata.get("fermion_indices", ()), *self.metadata.get("boson_indices", ())]
        if self.statistics == "fermion":
            if modes:
                # Custom models: exact CAR sources instead of the wicks path,
                # which litters symbolic labels with partition dummies.
                if not eom_result.is_closed:
                    raise ValueError("Fermionic EOM basis is not closed; extend the basis or truncate it explicitly.")
                source = sp.Matrix(
                    [
                        [
                            collapse_orthogonal_mode_deltas(
                                normal_order_fermionic(sp.expand(basis_op * right_expr + right_expr * basis_op)),
                                modes,
                            )
                        ]
                        for basis_op in basis_exprs
                    ]
                )
                lhs = (omega + sp.I * eta) * sp.eye(eom_result.eom_matrix.shape[0]) - eom_result.eom_matrix
                return sp.simplify(lhs.LUsolve(source)[left_index, 0])
            matrix = retarded_green_from_fermionic_eom(eom_result, working_basis, [right], omega=omega, eta=eta)
        else:
            matrix = retarded_green_from_bosonic_eom(eom_result, working_basis, [right], omega=omega, eta=eta)
        value = matrix[left_index, 0]
        if modes:
            value = collapse_orthogonal_mode_deltas(value, modes)
        return physical_simplify_fermionic(value) if self.statistics == "fermion" else value

    def latex_hamiltonian(self) -> str:
        expr = _unwrap_symbolic(self.hamiltonian)
        return sp.latex(expr)

    def _repr_latex_(self) -> str:
        return f"$H = {self.latex_hamiltonian()}$"


def fermionic_single_level_model(epsilon: sp.Expr, index: Any = 0) -> SymbolicModel:
    d = destroy(index)
    dd = d.dag()
    hamiltonian = epsilon * num(index)
    return SymbolicModel(
        name="fermionic_single_level",
        statistics="fermion",
        hamiltonian=hamiltonian,
        basis=[d],
        operators={"d": d, "d_dag": dd},
        metadata={"index": index},
    )


def fermionic_dimer_model(epsilon_left: sp.Expr, epsilon_right: sp.Expr, hopping: sp.Expr, indices: tuple[Any, Any] = (0, 1)) -> SymbolicModel:
    left, right = indices
    c0 = destroy(left)
    c1 = destroy(right)
    hamiltonian = (
        epsilon_left * num(left)
        + epsilon_right * num(right)
        + hopping * c0.dag() * c1
        + sp.conjugate(hopping) * c1.dag() * c0
    )
    return SymbolicModel(
        name="fermionic_dimer",
        statistics="fermion",
        hamiltonian=hamiltonian,
        basis=[c0, c1],
        operators={"c0": c0, "c1": c1, "c0_dag": c0.dag(), "c1_dag": c1.dag()},
        metadata={"indices": indices},
    )


def anderson_impurity_model(
    epsilon_up: sp.Expr,
    epsilon_down: sp.Expr,
    interaction_u: sp.Expr,
    indices: tuple[Any, Any] = ("up", "down"),
    spin_flip: sp.Expr = 0,
) -> SymbolicModel:
    up, down = indices
    d_up = destroy(up)
    d_down = destroy(down)
    hamiltonian = (
        epsilon_up * num(up)
        + epsilon_down * num(down)
        + interaction_u * num(up) * num(down)
        + spin_flip * d_up.dag() * d_down
        + sp.conjugate(spin_flip) * d_down.dag() * d_up
    )

    def _atomic_builder(operators: Sequence[Any], _hamiltonian: Any) -> GenericEOMResult:
        return build_anderson_atomic_eom_system(operators, epsilon_up, epsilon_down, interaction_u, indices=indices)

    spin_flip_is_zero = sp.simplify(spin_flip) == 0

    return SymbolicModel(
        name="anderson_impurity",
        statistics="fermion",
        hamiltonian=hamiltonian,
        basis=[d_up, d_down],
        operators={
            "d_up": d_up,
            "d_down": d_down,
            "d_up_dag": d_up.dag(),
            "d_down_dag": d_down.dag(),
        },
        metadata={
            "indices": indices,
            "interacting": True,
            "epsilon_up": epsilon_up,
            "epsilon_down": epsilon_down,
            "interaction_u": interaction_u,
            "spin_flip": spin_flip,
        },
        eom_builder=_atomic_builder if spin_flip_is_zero else None,
    )


def bosonic_harmonic_mode_model(omega0: sp.Expr, index: Any = 0) -> SymbolicModel:
    b = destroy_b(index)
    bd = b.dag()
    hamiltonian = omega0 * num_b(index)
    return SymbolicModel(
        name="bosonic_harmonic_mode",
        statistics="boson",
        hamiltonian=hamiltonian,
        basis=[b],
        operators={"b": b, "b_dag": bd},
        metadata={"index": index},
    )


def coupled_bosonic_dimer_model(omega_left: sp.Expr, omega_right: sp.Expr, coupling: sp.Expr, indices: tuple[Any, Any] = (0, 1)) -> SymbolicModel:
    left, right = indices
    b0 = destroy_b(left)
    b1 = destroy_b(right)
    hamiltonian = (
        omega_left * num_b(left)
        + omega_right * num_b(right)
        + coupling * b0.dag() * b1
        + sp.conjugate(coupling) * b1.dag() * b0
    )
    return SymbolicModel(
        name="coupled_bosonic_dimer",
        statistics="boson",
        hamiltonian=hamiltonian,
        basis=[b0, b1],
        operators={"b0": b0, "b1": b1, "b0_dag": b0.dag(), "b1_dag": b1.dag()},
        metadata={"indices": indices},
    )


def holstein_single_site_model(epsilon: sp.Expr, omega0: sp.Expr, coupling_g: sp.Expr, fermion_index: Any = 0, boson_index: Any = 0) -> SymbolicModel:
    d = destroy(fermion_index)
    b = destroy_b(boson_index)
    n_f = num(fermion_index).doit()
    n_b = num_b(boson_index).doit()
    hamiltonian = epsilon * n_f + omega0 * n_b + coupling_g * n_f * (b.doit() + b.dag().doit())
    return SymbolicModel(
        name="holstein_single_site",
        statistics="mixed",
        hamiltonian=hamiltonian,
        basis=[d, b],
        operators={"d": d, "d_dag": d.dag(), "b": b, "b_dag": b.dag()},
        metadata={"fermion_index": fermion_index, "boson_index": boson_index, "interacting": True, "mixed": True},
        eom_builder=build_mixed_eom_system,
    )


def collapse_orthogonal_mode_deltas(expr: Any, modes: Sequence[sp.Expr]) -> sp.Expr:
    """
    Evaluate KroneckerDelta factors between distinct concrete mode labels.

    ``custom_model`` enumerates each ladder index as an independent orthogonal
    mode, so ``KroneckerDelta(up, down)`` between two different labels is zero
    even though SymPy cannot decide it for generic symbols. Deltas whose
    arguments are not both in ``modes`` (e.g. genuine summation dummies) are
    left untouched.
    """
    expression = _unwrap_symbolic(expr)
    mode_set = {sp.sympify(mode) for mode in modes}

    def _replace(delta: sp.Expr) -> sp.Expr:
        left, right = delta.args
        if left in mode_set and right in mode_set:
            return sp.Integer(1) if left == right else sp.Integer(0)
        return delta

    return expression.replace(lambda node: isinstance(node, KroneckerDelta), _replace)


def _collapse_result_mode_deltas(result: Any, modes: Sequence[sp.Expr]) -> Any:
    """Collapse orthogonal-mode deltas inside an EOM result's matrix and residuals."""
    result.eom_matrix = result.eom_matrix.applyfunc(lambda entry: collapse_orthogonal_mode_deltas(entry, modes))
    result.residuals = [sp.expand(collapse_orthogonal_mode_deltas(residual, modes)) for residual in result.residuals]
    return result


def _exact_fermionic_rhs(operator: sp.Expr, hamiltonian: sp.Expr, modes: Sequence[sp.Expr]) -> sp.Expr:
    """EOM right-hand side [operator, H] via exact CAR normal ordering (no wicks)."""
    raw = sp.expand(operator * hamiltonian - hamiltonian * operator)
    return sp.expand(collapse_orthogonal_mode_deltas(normal_order_fermionic(raw), modes))


def _decompose_canonical_ladder_terms(
    rhs: sp.Expr,
    basis_exprs: Sequence[sp.Expr],
) -> tuple[list[sp.Expr], sp.Expr]:
    """
    Split a canonically normal-ordered rhs into basis coefficients and residual.

    Matches each term's ladder string structurally (srepr) against the basis,
    which is reliable where sympy's ``Expr.coeff`` mishandles noncommutative
    factors sitting in the middle of longer operator strings.
    """
    basis_keys = {_basis_key(expr): position for position, expr in enumerate(basis_exprs)}
    coeff_row: list[sp.Expr] = [sp.Integer(0)] * len(basis_exprs)
    residual_terms: list[sp.Expr] = []
    terms = rhs.args if isinstance(rhs, sp.Add) else ((rhs,) if rhs != 0 else ())
    for term in terms:
        coefficient, ladder = _ladder_factors(term)
        if not ladder:
            if coefficient != 0:
                residual_terms.append(term)
            continue
        position = basis_keys.get(_basis_key(sp.Mul(*ladder)))
        if position is None:
            residual_terms.append(term)
        else:
            coeff_row[position] += coefficient
    return coeff_row, sp.expand(sp.Add(*residual_terms))


def build_fermionic_eom_system_exact(
    operators: Sequence[Any],
    hamiltonian: Any,
    modes: Sequence[sp.Expr] | None = None,
) -> FermionicEOMResult:
    """
    Projected fermionic EOM built with exact CAR normal ordering.

    This is the engine used by :func:`custom_model`: it is deterministic for
    symbolic mode labels (where sympy's ``wicks`` machinery is fragile),
    collapses KroneckerDelta factors between distinct modes, and only accepts
    scalar coefficients in the EOM matrix — operator strings that do not
    close on the basis land in the residuals.
    """
    operator_exprs = [_unwrap_symbolic(item) for item in operators]
    hamiltonian_expr = sp.expand(_unwrap_symbolic(hamiltonian))
    if modes is None:
        fermion_modes, _ = _ladder_indices(hamiltonian_expr + sp.Add(*operator_exprs))
        modes = fermion_modes
    rows = []
    residuals = []
    for operator in operator_exprs:
        rhs = _exact_fermionic_rhs(operator, hamiltonian_expr, modes)
        coeff_row, residual = _decompose_canonical_ladder_terms(rhs, operator_exprs)
        rows.append(sp.Matrix([coeff_row]))
        residuals.append(residual)
    eom_matrix = sp.Matrix.vstack(*rows) if rows else sp.Matrix([])
    return FermionicEOMResult(operators=operator_exprs, eom_matrix=eom_matrix, residuals=residuals)


def dagger_expression(expr: Any) -> sp.Expr:
    """Hermitian conjugate of a second-quantized expression (fermions and bosons)."""
    expr = _unwrap_symbolic(expr)
    if isinstance(expr, AnnihilateFermion):
        return CreateFermion(expr.args[0])
    if isinstance(expr, CreateFermion):
        return AnnihilateFermion(expr.args[0])
    if isinstance(expr, AnnihilateBoson):
        return CreateBoson(expr.args[0])
    if isinstance(expr, CreateBoson):
        return AnnihilateBoson(expr.args[0])
    if isinstance(expr, sp.Add):
        return sp.Add(*[dagger_expression(arg) for arg in expr.args])
    if isinstance(expr, sp.Mul):
        return sp.Mul(*[dagger_expression(arg) for arg in reversed(expr.args)])
    if isinstance(expr, sp.Pow):
        return dagger_expression(expr.base) ** expr.exp
    if expr.is_commutative:
        return sp.conjugate(expr)
    return expr


def _warn_if_not_hermitian(hamiltonian: sp.Expr, statistics: str, name: str, modes: Sequence[sp.Expr]) -> None:
    """
    Warn when a user-supplied Hamiltonian is visibly non-Hermitian.

    Missing conjugate hopping terms and wrong daggers are the most common
    mistakes when building Hamiltonians by hand. For purely fermionic models
    the check is exact (both sides are canonically normal-ordered and deltas
    between distinct orthogonal modes are collapsed); for bosonic/mixed models
    only a structural comparison is attempted, and no warning is raised if it
    is inconclusive.
    """
    difference = sp.expand(hamiltonian - dagger_expression(hamiltonian))
    if difference == 0:
        return
    if statistics == "fermion":
        try:
            difference = collapse_orthogonal_mode_deltas(normal_order_fermionic(difference), modes)
        except ValueError:  # pragma: no cover - defensive
            return
        if sp.simplify(sp.expand(difference)) == 0:
            return
    else:
        return
    warnings.warn(
        f"custom_model '{name}': the Hamiltonian is not Hermitian (H - H^dagger = {difference}). "
        "Check for missing conjugate hopping terms; declare complex couplings with sympy symbols "
        "and add their sp.conjugate(...) partners, or pass check_hermitian=False if a "
        "non-Hermitian effective Hamiltonian is intentional.",
        UserWarning,
        stacklevel=3,
    )


def _ladder_indices(expr: sp.Expr) -> tuple[list[sp.Expr], list[sp.Expr]]:
    """Collect the distinct fermionic and bosonic mode indices appearing in ``expr``."""
    fermion: list[sp.Expr] = []
    boson: list[sp.Expr] = []
    for atom in expr.atoms(AnnihilateFermion, CreateFermion):
        index = atom.args[0]
        if index not in fermion:
            fermion.append(index)
    for atom in expr.atoms(AnnihilateBoson, CreateBoson):
        index = atom.args[0]
        if index not in boson:
            boson.append(index)
    return sorted(fermion, key=sp.default_sort_key), sorted(boson, key=sp.default_sort_key)


def custom_model(
    hamiltonian: Any,
    basis: Sequence[Any] | None = None,
    *,
    name: str = "custom",
    metadata: Mapping[str, Any] | None = None,
    check_hermitian: bool = True,
) -> SymbolicModel:
    """
    Build a :class:`SymbolicModel` from an arbitrary second-quantized Hamiltonian.

    The statistics (``"fermion"``, ``"boson"``, or ``"mixed"``) and the seed
    operator basis are detected automatically from the ladder operators present
    in ``hamiltonian``. Operators become available under ``model.operators`` as
    ``c_<index>`` / ``c_<index>_dag`` (fermions) and ``b_<index>`` /
    ``b_<index>_dag`` (bosons).

    Parameters
    ----------
    hamiltonian:
        A SymPy expression (or ``SQObj``/``BQObj``) built from ladder operators.
    basis:
        Optional explicit seed basis; defaults to one annihilation operator per
        detected mode.
    """
    hamiltonian_expr = sp.expand(_unwrap_symbolic(hamiltonian))
    fermion_indices, boson_indices = _ladder_indices(hamiltonian_expr)
    if not fermion_indices and not boson_indices:
        raise ValueError(
            "hamiltonian contains no fermionic or bosonic ladder operators; "
            "build it with f/fd/b/bd (quantum_transport.highlevel) or F/Fd/B/Bd (sympy)."
        )
    if fermion_indices and boson_indices:
        statistics = "mixed"
    elif fermion_indices:
        statistics = "fermion"
    else:
        statistics = "boson"

    if check_hermitian:
        _warn_if_not_hermitian(hamiltonian_expr, statistics, name, [*fermion_indices, *boson_indices])

    operators: dict[str, Any] = {}
    seed: list[Any] = []
    for index in fermion_indices:
        op = destroy(index)
        label = str(index)
        operators[f"c_{label}"] = op
        operators[f"c_{label}_dag"] = op.dag()
        seed.append(op)
    for index in boson_indices:
        op = destroy_b(index)
        label = str(index)
        operators[f"b_{label}"] = op
        operators[f"b_{label}_dag"] = op.dag()
        seed.append(op)

    model_metadata: dict[str, Any] = {
        "fermion_indices": tuple(fermion_indices),
        "boson_indices": tuple(boson_indices),
    }
    if metadata:
        model_metadata.update(metadata)

    modes = [*fermion_indices, *boson_indices]

    def _orthogonal_eom_builder(working_basis: Sequence[Any], working_hamiltonian: Any):
        if statistics == "fermion":
            return build_fermionic_eom_system_exact(working_basis, working_hamiltonian, modes=modes)
        if statistics == "boson":
            result = build_bosonic_eom_system(working_basis, working_hamiltonian)
        else:
            result = build_mixed_eom_system(working_basis, working_hamiltonian)
        return _collapse_result_mode_deltas(result, modes)

    return SymbolicModel(
        name=name,
        statistics=statistics,
        hamiltonian=hamiltonian_expr,
        basis=list(basis) if basis is not None else seed,
        operators=operators,
        metadata=model_metadata,
        eom_builder=_orthogonal_eom_builder,
    )


def single_particle_hamiltonian_matrix(
    hamiltonian: Any,
    modes: Sequence[Any] | None = None,
) -> tuple[sp.Matrix, list[sp.Expr]]:
    """
    Extract the single-particle matrix ``h`` from a quadratic fermionic Hamiltonian.

    For ``H = sum_ij h_ij c_i^dag c_j`` (normal-ordered, no bosons, no quartic
    terms) returns ``(h, modes)`` where ``h`` is a symbolic square matrix and
    ``modes`` the ordered list of mode indices labelling its rows/columns.
    Constant (operator-free) terms are ignored.

    Raises ``ValueError`` when the Hamiltonian is interacting (more than two
    ladder operators in a term), contains bosons, or is not normal-ordered.
    """
    expr = sp.expand(_unwrap_symbolic(hamiltonian))
    fermion_indices, boson_indices = _ladder_indices(expr)
    if boson_indices:
        raise ValueError("single_particle_hamiltonian_matrix supports fermionic Hamiltonians only.")
    if modes is None:
        mode_list = list(fermion_indices)
    else:
        mode_list = [sp.sympify(mode) for mode in modes]
    position = {mode: i for i, mode in enumerate(mode_list)}
    dim = len(mode_list)
    matrix = sp.zeros(dim, dim)

    terms = expr.args if isinstance(expr, sp.Add) else ((expr,) if expr != 0 else ())
    for term in terms:
        factors = term.args if isinstance(term, sp.Mul) else (term,)
        coefficient = sp.Integer(1)
        ladder: list[sp.Expr] = []
        for factor in factors:
            if factor.is_commutative:
                coefficient *= factor
            else:
                ladder.append(factor)
        if not ladder:
            continue
        if len(ladder) != 2:
            raise ValueError(
                f"Hamiltonian term {term} is not quadratic; interacting models cannot be reduced "
                "to a single-particle matrix (use the EOM layer with a truncation instead)."
            )
        creator, annihilator = ladder
        if not isinstance(creator, CreateFermion) or not isinstance(annihilator, AnnihilateFermion):
            raise ValueError(
                f"Hamiltonian term {term} is not normal-ordered (expected c_i^dag c_j); "
                "rewrite it with creation operators to the left."
            )
        i = position.get(creator.args[0])
        j = position.get(annihilator.args[0])
        if i is None or j is None:
            raise ValueError(f"Mode {creator.args[0]} or {annihilator.args[0]} missing from modes={mode_list}.")
        matrix[i, j] += coefficient
    return matrix, mode_list


def _ladder_factors(monomial: sp.Expr) -> tuple[sp.Expr, list[sp.Expr]]:
    """Split a monomial into (commutative coefficient, ordered ladder-operator factors)."""
    factors = monomial.args if isinstance(monomial, sp.Mul) else (monomial,)
    coefficient = sp.Integer(1)
    ladder: list[sp.Expr] = []
    for factor in factors:
        if factor.is_commutative:
            coefficient *= factor
            continue
        if isinstance(factor, sp.Pow) and factor.exp.is_Integer and int(factor.exp) >= 2 and isinstance(factor.base, (AnnihilateFermion, CreateFermion)):
            return sp.Integer(0), []
        ladder.append(factor)
    return coefficient, ladder


def _occupation_average(index: sp.Expr, occupations: Mapping[Any, Any]) -> sp.Expr:
    key = str(index)
    if key in occupations:
        return sp.sympify(occupations[key])
    if index in occupations:
        return sp.sympify(occupations[index])
    return sp.Symbol(f"n_{key}_avg", real=True)


def _coherence_average(create_index: sp.Expr, annihilate_index: sp.Expr, occupations: Mapping[Any, Any]) -> sp.Expr:
    key = (str(create_index), str(annihilate_index))
    if key in occupations:
        return sp.sympify(occupations[key])
    return sp.Symbol(f"avg_{create_index}_{annihilate_index}")


def _pair_average(x: sp.Expr, y: sp.Expr, occupations: Mapping[Any, Any], *, include_fock: bool) -> sp.Expr:
    """Mean-field contraction <x y> of two fermionic ladder operators."""
    x_create = isinstance(x, CreateFermion)
    y_create = isinstance(y, CreateFermion)
    if x_create and not y_create:
        i, j = x.args[0], y.args[0]
        if i == j:
            return _occupation_average(i, occupations)
        return _coherence_average(i, j, occupations) if include_fock else sp.Integer(0)
    if (not x_create) and y_create:
        i, j = x.args[0], y.args[0]
        if i == j:
            return 1 - _occupation_average(i, occupations)
        return -_coherence_average(j, i, occupations) if include_fock else sp.Integer(0)
    return sp.Integer(0)


def _decouple_cubic(coefficient: sp.Expr, ladder: Sequence[sp.Expr], occupations: Mapping[Any, Any], *, include_fock: bool) -> sp.Expr:
    """Wick-style mean-field factorization A B C -> <AB>C - <AC>B + <BC>A."""
    a, b, c = ladder
    return coefficient * (
        _pair_average(a, b, occupations, include_fock=include_fock) * c
        - _pair_average(a, c, occupations, include_fock=include_fock) * b
        + _pair_average(b, c, occupations, include_fock=include_fock) * a
    )


def build_fermionic_hartree_eom_system(
    operators: Sequence[Any],
    hamiltonian: Any,
    *,
    occupations: Mapping[Any, Any] | None = None,
    include_fock: bool = False,
) -> GenericEOMResult:
    """
    Projected fermionic EOM with automatic mean-field (Hartree) decoupling.

    Cubic operator strings generated by quartic interactions are factorized as
    ``A B C -> <AB>C - <AC>B + <BC>A``, keeping density contractions
    ``<c_i^dag c_i>`` (named ``n_<i>_avg`` unless supplied via ``occupations``).
    With ``include_fock=True`` off-diagonal coherences ``<c_i^dag c_j>`` are
    kept as ``avg_<i>_<j>`` symbols instead of being dropped. Anomalous
    contractions are always discarded. Works for any fermionic model, not just
    the Anderson impurity.
    """
    operator_exprs = [_unwrap_symbolic(item) for item in operators]
    hamiltonian_expr = _unwrap_symbolic(hamiltonian)
    fermion_modes, _boson_modes = _ladder_indices(
        sp.expand(hamiltonian_expr) + sp.Add(*operator_exprs)
    )
    occupation_map: Mapping[Any, Any] = {} if occupations is None else dict(occupations)
    rows = []
    residuals = []
    for operator in operator_exprs:
        rhs = _exact_fermionic_rhs(operator, hamiltonian_expr, fermion_modes)
        terms = rhs.args if isinstance(rhs, sp.Add) else ((rhs,) if rhs != 0 else ())
        decoupled_terms = []
        for term in terms:
            coefficient, ladder = _ladder_factors(term)
            if not ladder and coefficient == 0:
                continue
            if len(ladder) == 3:
                decoupled_terms.append(_decouple_cubic(coefficient, ladder, occupation_map, include_fock=include_fock))
            else:
                decoupled_terms.append(term)
        rhs_mf = sp.expand(sp.Add(*decoupled_terms))
        coeff_row, residual = _decompose_canonical_ladder_terms(rhs_mf, operator_exprs)
        rows.append(sp.Matrix([coeff_row]))
        residuals.append(residual)
    eom_matrix = sp.Matrix.vstack(*rows) if rows else sp.Matrix([])
    return GenericEOMResult(operators=operator_exprs, eom_matrix=eom_matrix, residuals=residuals)


def jaynes_cummings_like_model(epsilon: sp.Expr, omega0: sp.Expr, coupling_g: sp.Expr, fermion_index: Any = 0, boson_index: Any = 0) -> SymbolicModel:
    d = destroy(fermion_index)
    b = destroy_b(boson_index)
    n_f = num(fermion_index).doit()
    n_b = num_b(boson_index).doit()
    hamiltonian = epsilon * n_f + omega0 * n_b + coupling_g * (d.dag().doit() * b.doit() + b.dag().doit() * d.doit())
    return SymbolicModel(
        name="jaynes_cummings_like",
        statistics="mixed",
        hamiltonian=hamiltonian,
        basis=[d, b],
        operators={"d": d, "d_dag": d.dag(), "b": b, "b_dag": b.dag()},
        metadata={"fermion_index": fermion_index, "boson_index": boson_index, "mixed": True},
        eom_builder=build_mixed_eom_system,
    )
