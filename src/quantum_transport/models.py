"""Predefined symbolic models for EOM workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import sympy as sp

from .algebra import decompose_in_basis
from .bosonic import build_bosonic_eom_system, bosonic_eom_rhs, retarded_green_from_bosonic_eom, destroy_b, num_b
from .secondquant import build_fermionic_eom_system, fermionic_eom_rhs, physical_simplify_fermionic, retarded_green_from_fermionic_eom, destroy, num


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
        eom_result = self.eom(
            basis=working_basis,
            truncation=truncation,
            truncation_params=truncation_params,
        )
        basis_exprs = [_unwrap_symbolic(item) for item in working_basis]
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
        if self.statistics == "fermion":
            matrix = retarded_green_from_fermionic_eom(eom_result, working_basis, [right], omega=omega, eta=eta)
        else:
            matrix = retarded_green_from_bosonic_eom(eom_result, working_basis, [right], omega=omega, eta=eta)
        value = matrix[left_index, 0]
        return physical_simplify_fermionic(value) if self.statistics == "fermion" else value

    def latex_hamiltonian(self) -> str:
        expr = _unwrap_symbolic(self.hamiltonian)
        return sp.latex(expr)


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
