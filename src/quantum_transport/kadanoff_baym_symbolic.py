"""Symbolic two-time convolutions and Kadanoff--Baym Dyson equations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import sympy as sp


def _kernel_value(kernel: Any, left: sp.Symbol, right: sp.Symbol) -> sp.Expr:
    if callable(kernel):
        return sp.sympify(kernel(left, right))
    if isinstance(kernel, sp.FunctionClass):
        return kernel(left, right)
    return sp.sympify(kernel)


def _energy_value(kernel: Any, energy: sp.Expr) -> sp.Expr:
    if callable(kernel):
        return sp.sympify(kernel(energy))
    if isinstance(kernel, sp.FunctionClass):
        return kernel(energy)
    return sp.sympify(kernel)


def time_convolution_symbolic(
    left: Any,
    right: Any,
    time: sp.Symbol,
    time_prime: sp.Symbol,
    *,
    integration_time: sp.Symbol | None = None,
    lower: Any = -sp.oo,
    upper: Any = sp.oo,
) -> sp.Expr:
    r"""Return ``Integral(left(time,tau)*right(tau,time_prime),tau)``."""
    tau = integration_time or sp.Symbol("tau", real=True)
    integrand = _kernel_value(left, time, tau) * _kernel_value(right, tau, time_prime)
    return sp.Integral(integrand, (tau, lower, upper))


def one_body_correlation_symbolic(
    observable_left: Any,
    observable_right: Any,
    *,
    green_lesser: Any | None = None,
    green_greater: Any | None = None,
    time: sp.Symbol | None = None,
    time_prime: sp.Symbol | None = None,
) -> dict[str, sp.Expr]:
    r"""Write the connected charge/spin Wick bubble symbolically.

    The returned ``connected`` branch is

    ``Tr[A G<(t,t') B G>(t',t)]``.

    ``symmetrized`` adds the exchanged branch and divides by two.  The trace
    is represented by an abstract ``Tr`` function so the expression remains
    dimension-independent and can later be populated with explicit matrices.
    """

    t = time or sp.Symbol("t", real=True)
    tp = time_prime or sp.Symbol("t_prime", real=True)
    lesser = sp.Function("G_lesser") if green_lesser is None else green_lesser
    greater = sp.Function("G_greater") if green_greater is None else green_greater
    gl = _kernel_value(lesser, t, tp)
    gg = _kernel_value(greater, tp, t)
    gl_reverse = _kernel_value(lesser, tp, t)
    gg_reverse = _kernel_value(greater, t, tp)
    left = sp.sympify(observable_left)
    right = sp.sympify(observable_right)
    trace = sp.Function("Tr")
    connected = trace(left * gl * right * gg)
    exchanged = trace(right * gl_reverse * left * gg_reverse)
    return {
        "connected": sp.simplify(connected),
        "exchanged": sp.simplify(exchanged),
        "symmetrized": sp.simplify((connected + exchanged) / 2),
        "convention": sp.Symbol("connected_Wick_bubble"),
    }


def bethe_salpeter_ladder_symbolic(
    bubble: Any,
    interaction_kernel: Any,
) -> dict[str, sp.Expr | sp.MatrixBase]:
    r"""Return the algebraic particle--hole ladder used by the numeric helper.

    For a scalar channel the result is

    ``Gamma = (1 - chi0*K)^(-1) K`` and
    ``chi = chi0 + chi0*Gamma*chi0``.

    Matrix inputs are promoted to SymPy matrices and use the corresponding
    non-commuting channel ordering.  This is the symbolic declaration of the
    finite-grid local ladder; it does not imply an exact interacting current
    vertex or a Ward identity without a matching self-energy functional.
    """

    chi0 = sp.sympify(bubble)
    kernel = sp.sympify(interaction_kernel)
    if isinstance(chi0, sp.MatrixBase) or isinstance(kernel, sp.MatrixBase):
        chi0_matrix = chi0 if isinstance(chi0, sp.MatrixBase) else sp.Matrix([[chi0]])
        kernel_matrix = kernel if isinstance(kernel, sp.MatrixBase) else sp.Matrix([[kernel]])
        if chi0_matrix.shape != kernel_matrix.shape or chi0_matrix.rows != chi0_matrix.cols:
            raise ValueError("bubble and interaction_kernel matrices must be square with matching shape.")
        identity = sp.eye(chi0_matrix.rows)
        gamma = sp.simplify((identity - chi0_matrix * kernel_matrix).inv() * kernel_matrix)
        corrected = sp.simplify(chi0_matrix + chi0_matrix * gamma * chi0_matrix)
    else:
        denominator = sp.simplify(1 - chi0 * kernel)
        gamma = sp.simplify(kernel / denominator)
        corrected = sp.simplify(chi0 + chi0 * gamma * chi0)
    return {
        "gamma": gamma,
        "corrected": corrected,
        "bubble": chi0,
        "kernel": kernel,
        "claim_boundary": sp.Symbol("finite_grid_local_particle_hole_ladder"),
    }


def langreth_two_time_convolution_symbolic(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    time: sp.Symbol,
    time_prime: sp.Symbol,
    *,
    integration_time: sp.Symbol | None = None,
    lower: Any = -sp.oo,
    upper: Any = sp.oo,
) -> dict[str, sp.Expr]:
    """Apply Langreth rules while retaining explicit time convolutions."""
    convolution = lambda a, b: time_convolution_symbolic(
        left[a],
        right[b],
        time,
        time_prime,
        integration_time=integration_time,
        lower=lower,
        upper=upper,
    )
    return {
        "r": convolution("r", "r"),
        "a": convolution("a", "a"),
        "<": convolution("r", "<") + convolution("<", "a"),
        ">": convolution("r", ">") + convolution(">", "a"),
    }


def kadanoff_baym_dyson_equations(
    *,
    bare_retarded: Any,
    bare_lesser: Any,
    self_energy_retarded: Any,
    self_energy_lesser: Any,
    self_energy_advanced: Any,
    dressed_retarded: Any | None = None,
    dressed_lesser: Any | None = None,
    dressed_advanced: Any | None = None,
    time: sp.Symbol | None = None,
    time_prime: sp.Symbol | None = None,
    lower: Any = -sp.oo,
    upper: Any = sp.oo,
) -> dict[str, sp.Equality]:
    r"""Return the retarded and lesser two-time Dyson/KBE equations.

    The lesser equation is written in the causal Dyson expansion
    ``G< = g< + gr*Sr*G< + gr*S<*Ga + g<*Sa*Ga``.  Keeping the initial
    ``g<`` term explicit prevents a steady-state Keldysh ansatz from being
    silently substituted for a transient initial-correlation problem.
    """
    t = time or sp.Symbol("t", real=True)
    tp = time_prime or sp.Symbol("t_prime", real=True)
    gr = dressed_retarded or sp.Function("G_r")
    gl = dressed_lesser or sp.Function("G_lesser")
    ga = dressed_advanced or sp.Function("G_a")

    tau = sp.Symbol("tau", real=True)
    nu = sp.Symbol("nu", real=True)

    def double_convolution(first: Any, middle: Any, last: Any) -> sp.Expr:
        integrand = (
            _kernel_value(first, t, tau)
            * _kernel_value(middle, tau, nu)
            * _kernel_value(last, nu, tp)
        )
        return sp.Integral(
            integrand,
            (tau, lower, upper),
            (nu, lower, upper),
        )

    retarded_rhs = _kernel_value(bare_retarded, t, tp) + double_convolution(
        bare_retarded, self_energy_retarded, gr
    )
    lesser_rhs = _kernel_value(bare_lesser, t, tp)
    lesser_rhs += double_convolution(bare_retarded, self_energy_retarded, gl)
    lesser_rhs += double_convolution(bare_retarded, self_energy_lesser, ga)
    lesser_rhs += double_convolution(bare_lesser, self_energy_advanced, ga)
    return {
        "retarded": sp.Eq(_kernel_value(gr, t, tp), retarded_rhs),
        "lesser": sp.Eq(_kernel_value(gl, t, tp), lesser_rhs),
    }


def kadanoff_baym_mixed_equations(
    *,
    self_energy_retarded: Any,
    self_energy_mixed: Any,
    self_energy_matsubara: Any,
    self_energy_advanced: Any | None = None,
    self_energy_lmixed: Any | None = None,
    green_mixed: Any | None = None,
    green_lmixed: Any | None = None,
    green_matsubara: Any | None = None,
    one_body_hamiltonian: Any | None = None,
    time: sp.Symbol | None = None,
    imaginary_time: sp.Symbol | None = None,
    imaginary_time_prime: sp.Symbol | None = None,
    lower: Any = -sp.oo,
    beta: Any = sp.Symbol("beta", positive=True),
) -> dict[str, sp.Equality]:
    r"""Return the differential Kadanoff--Baym equations for mixed branches.

    The equations are the real/vertical components of the contour Dyson
    equation, with the contour measure ``d z = -i d tau`` on the imaginary
    branch.  For matrix kernels multiplication order is retained explicitly::

        (i d_t - h(t)) G^rceil(t,tau)
          = int_lower^t dt' Sigma^R(t,t') G^rceil(t',tau)
            - i int_0^beta dtau' Sigma^rceil(t,tau') G^M(tau',tau)

    The ``lceil`` equation is the adjoint/right-acting counterpart.  No
    steady-state ansatz is inserted: ``G^M`` and both mixed self-energies are
    required inputs, making the initial-correlation closure explicit.
    """
    t = time or sp.Symbol("t", real=True)
    tau = imaginary_time or sp.Symbol("tau", real=True)
    tau_prime = imaginary_time_prime or sp.Symbol("tau_prime", real=True)
    real_prime = sp.Symbol("t_prime", real=True)

    # Default kernels are non-commutative so the displayed products retain
    # matrix multiplication order.  Callers may still provide scalar
    # functions for compact model-specific reductions.
    rceil = sp.Function("G_rceil", commutative=False) if green_mixed is None else green_mixed
    lceil = sp.Function("G_lceil", commutative=False) if green_lmixed is None else green_lmixed
    matsubara = sp.Function("G_M", commutative=False) if green_matsubara is None else green_matsubara
    sigma_r = self_energy_retarded
    sigma_a = self_energy_advanced if self_energy_advanced is not None else sp.Function("Sigma_a", commutative=False)
    sigma_rceil = self_energy_mixed
    sigma_lceil = self_energy_lmixed if self_energy_lmixed is not None else sp.Function("Sigma_lceil", commutative=False)
    sigma_m = self_energy_matsubara
    h = one_body_hamiltonian if one_body_hamiltonian is not None else sp.Function("h", commutative=False)

    def value(function: Any, left: Any, right: Any) -> sp.Expr:
        return _kernel_value(function, left, right)

    def one_body(function: Any, argument: Any) -> sp.Expr:
        if callable(function):
            return sp.sympify(function(argument))
        if isinstance(function, sp.FunctionClass):
            return function(argument)
        return sp.sympify(function)

    rceil_value = value(rceil, t, tau)
    lceil_value = value(lceil, tau, t)
    right_real = sp.Integral(
        value(sigma_r, t, real_prime) * value(rceil, real_prime, tau),
        (real_prime, lower, t),
    )
    right_vertical = -sp.I * sp.Integral(
        value(sigma_rceil, t, tau_prime) * value(matsubara, tau_prime, tau),
        (tau_prime, 0, beta),
    )
    left_real = sp.Integral(
        value(lceil, tau, real_prime) * value(sigma_a, real_prime, t),
        (real_prime, lower, t),
    )
    left_vertical = -sp.I * sp.Integral(
        value(matsubara, tau, tau_prime) * value(sigma_lceil, tau_prime, t),
        (tau_prime, 0, beta),
    )
    return {
        "rceil": sp.Eq(
            sp.I * sp.Derivative(rceil_value, t) - one_body(h, t) * rceil_value,
            right_real + right_vertical,
        ),
        "lceil": sp.Eq(
            -sp.I * sp.Derivative(lceil_value, t) - lceil_value * one_body(h, t),
            left_real + left_vertical,
        ),
    }


def kadanoff_baym_initial_correlation_symbolic(
    *,
    self_energy_mixed: Any | None = None,
    green_mixed: Any | None = None,
    time: sp.Symbol | None = None,
    time_prime: sp.Symbol | None = None,
    imaginary_time: sp.Symbol | None = None,
    beta: Any = sp.Symbol("beta", positive=True),
) -> dict[str, sp.Expr]:
    r"""Return the vertical-branch Keldysh source symbolically.

    The mixed components are ``Sigma^rceil(t,tau)`` and
    ``G^lceil(tau,t_prime)``.  The density equation receives the equal-time
    combination ``I(t,t) + I(t,t)^dagger``; the explicit adjoint is kept as a
    symbolic function so a caller can impose the contour adjoint convention
    appropriate to its model.
    """

    t = time or sp.Symbol("t", real=True)
    tp = time_prime or sp.Symbol("t_prime", real=True)
    tau = imaginary_time or sp.Symbol("tau", real=True)
    sigma = (
        sp.Function("Sigma_rceil")
        if self_energy_mixed is None
        else self_energy_mixed
    )
    green = (
        sp.Function("G_lceil")
        if green_mixed is None
        else green_mixed
    )
    kernel = -sp.I * sp.Integral(
        _kernel_value(sigma, t, tau) * _kernel_value(green, tau, tp),
        (tau, 0, beta),
    )
    equal_time = kernel.subs(tp, t)
    adjoint = sp.Function("dagger")(equal_time)
    return {
        "mixed_kernel": kernel,
        "density_source": sp.simplify(equal_time + adjoint),
    }


def kadanoff_baym_lesser_initial_correlation_symbolic(
    *,
    green_retarded: Any | None = None,
    self_energy_mixed: Any | None = None,
    green_lmixed: Any | None = None,
    time: sp.Symbol | None = None,
    time_prime: sp.Symbol | None = None,
    imaginary_time: sp.Symbol | None = None,
    real_integration_time: sp.Symbol | None = None,
    beta: Any = sp.Symbol("beta", positive=True),
    lower: Any = -sp.oo,
) -> dict[str, sp.Expr]:
    r"""Return the propagated vertical contribution to the lesser Dyson term.

    The left KBE source and its causal retarded propagation are displayed as

    ``I(t,t') = -i int_0^beta d tau Sigma^rceil(t,tau) G^lceil(tau,t')``
    and ``C(t,t') = int_lower^t d tbar G^R(t,tbar) I(tbar,t')``.  The lesser
    correction is ``C(t,t') - C(t',t)^dagger``.  Keeping this term separate
    from the bare lesser input makes the initial-correlation approximation
    auditable and preserves matrix order for spin-orbit Hamiltonians.
    """

    t = time or sp.Symbol("t", real=True)
    tp = time_prime or sp.Symbol("t_prime", real=True)
    tau = imaginary_time or sp.Symbol("tau", real=True)
    tbar = real_integration_time or sp.Symbol("t_bar", real=True)
    sigma = (
        sp.Function("Sigma_rceil", commutative=False)
        if self_energy_mixed is None
        else self_energy_mixed
    )
    green_l = (
        sp.Function("G_lceil", commutative=False)
        if green_lmixed is None
        else green_lmixed
    )
    green_r = (
        sp.Function("G_r", commutative=False)
        if green_retarded is None
        else green_retarded
    )

    def value(function: Any, left: Any, right: Any) -> sp.Expr:
        return _kernel_value(function, left, right)

    source_at = lambda left, right: -sp.I * sp.Integral(
        value(sigma, left, tau) * value(green_l, tau, right),
        (tau, 0, beta),
    )
    source = source_at(t, tp)
    propagated = sp.Integral(
        value(green_r, t, tbar) * source_at(tbar, tp),
        (tbar, lower, t),
    )
    swapped = propagated.xreplace({t: tp, tp: t})
    adjoint = sp.Function("dagger")(swapped)
    return {
        "source_kernel": source,
        "propagated_source": propagated,
        "lesser_correction": sp.simplify(propagated - adjoint),
    }


def kadanoff_baym_contour_lesser_dyson_symbolic(
    *,
    bare_retarded: Any | None = None,
    bare_lesser: Any | None = None,
    bare_mixed: Any | None = None,
    self_energy_retarded: Any | None = None,
    self_energy_lesser: Any | None = None,
    self_energy_advanced: Any | None = None,
    self_energy_mixed: Any | None = None,
    self_energy_lmixed: Any | None = None,
    self_energy_matsubara: Any | None = None,
    dressed_lesser: Any | None = None,
    dressed_advanced: Any | None = None,
    dressed_lmixed: Any | None = None,
    time: sp.Symbol | None = None,
    time_prime: sp.Symbol | None = None,
    imaginary_time: sp.Symbol | None = None,
    imaginary_time_prime: sp.Symbol | None = None,
    real_integration_time: sp.Symbol | None = None,
    lower: Any = -sp.oo,
    upper: Any = sp.oo,
    beta: Any = sp.Symbol("beta", positive=True),
) -> dict[str, sp.Expr]:
    r"""Return the full vertical Langreth terms of the lesser Dyson equation.

    For ``G = g + g Sigma G`` the lesser expansion contains the usual three
    real-time terms plus

    ``g^rceil star Sigma^lceil dot G^A``
    ``+ g^R dot Sigma^rceil star G^lceil``
    ``+ g^rceil star Sigma^M star G^lceil``.

    The output keeps all matrix products ordered and leaves the contour measure
    ``-i d tau`` explicit.  This is a symbolic identity generator, not a
    claim that a finite-grid approximation is conserving.
    """

    t = time or sp.Symbol("t", real=True)
    tp = time_prime or sp.Symbol("t_prime", real=True)
    tau = imaginary_time or sp.Symbol("tau", real=True)
    tau_prime = imaginary_time_prime or sp.Symbol("tau_prime", real=True)
    bar = real_integration_time or sp.Symbol("t_bar", real=True)
    nu = sp.Symbol("nu", real=True)
    g_r = bare_retarded or sp.Function("g_r", commutative=False)
    g_l = bare_lesser or sp.Function("g_l", commutative=False)
    g_m = bare_mixed or sp.Function("g_rceil", commutative=False)
    s_r = self_energy_retarded or sp.Function("Sigma_r", commutative=False)
    s_l = self_energy_lesser or sp.Function("Sigma_l", commutative=False)
    s_a = self_energy_advanced or sp.Function("Sigma_a", commutative=False)
    s_m = self_energy_mixed or sp.Function("Sigma_rceil", commutative=False)
    s_lm = self_energy_lmixed or sp.Function("Sigma_lceil", commutative=False)
    s_M = self_energy_matsubara or sp.Function("Sigma_M", commutative=False)
    G_l = dressed_lesser or sp.Function("G_lesser", commutative=False)
    G_a = dressed_advanced or sp.Function("G_a", commutative=False)
    G_lm = dressed_lmixed or sp.Function("G_lceil", commutative=False)

    def value(function: Any, left: Any, right: Any) -> sp.Expr:
        return _kernel_value(function, left, right)

    def double_real(first: Any, middle: Any, last: Any) -> sp.Expr:
        return sp.Integral(
            value(first, t, tau) * value(middle, tau, nu) * value(last, nu, tp),
            (tau, lower, upper),
            (nu, lower, upper),
        )

    real_terms = (
        value(g_l, t, tp)
        + double_real(g_r, s_r, G_l)
        + double_real(g_r, s_l, G_a)
        + double_real(g_l, s_a, G_a)
    )
    mixed_advanced = sp.Integral(
        -sp.I
        * value(g_m, t, tau)
        * value(s_lm, tau, bar)
        * value(G_a, bar, tp),
        (tau, 0, beta),
        (bar, lower, upper),
    )
    propagated = sp.Integral(
        -sp.I
        * value(g_r, t, bar)
        * value(s_m, bar, tau)
        * value(G_lm, tau, tp),
        (bar, lower, upper),
        (tau, 0, beta),
    )
    matsubara = sp.Integral(
        (-sp.I) ** 2
        * value(g_m, t, tau)
        * value(s_M, tau, tau_prime)
        * value(G_lm, tau_prime, tp),
        (tau, 0, beta),
        (tau_prime, 0, beta),
    )
    return {
        "real_terms": real_terms,
        "mixed_advanced": mixed_advanced,
        "propagated_mixed": propagated,
        "matsubara": matsubara,
        "lesser": sp.Eq(
            value(G_l, t, tp),
            real_terms + mixed_advanced + propagated + matsubara,
        ),
    }


def kadanoff_baym_collision_integral_symbolic(
    *,
    green_retarded: Any | None = None,
    green_advanced: Any | None = None,
    green_lesser: Any | None = None,
    self_energy_retarded: Any | None = None,
    self_energy_lesser: Any | None = None,
    self_energy_advanced: Any | None = None,
    time: sp.Symbol | None = None,
    time_prime: sp.Symbol | None = None,
    integration_time: sp.Symbol | None = None,
    lower: Any = -sp.oo,
    upper: Any = sp.oo,
) -> dict[str, sp.Expr]:
    r"""Return the two-time KBE collision integral as ordered convolutions.

    With ``C = G^r*Sigma^< + G^<*Sigma^a - Sigma^r*G^< -
    Sigma^<*G^a`` this helper retains each of the four terms separately.
    The result is therefore suitable for auditing embedding and interaction
    self-energies before taking an equal-time or an observable projection.
    No steady-state Keldysh ansatz is inserted.
    """

    t = time or sp.Symbol("t", real=True)
    tp = time_prime or sp.Symbol("t_prime", real=True)
    g_r = green_retarded or sp.Function("G_r", commutative=False)
    g_a = green_advanced or sp.Function("G_a", commutative=False)
    g_l = green_lesser or sp.Function("G_lesser", commutative=False)
    s_r = self_energy_retarded or sp.Function("Sigma_r", commutative=False)
    s_l = self_energy_lesser or sp.Function("Sigma_l", commutative=False)
    s_a = self_energy_advanced or sp.Function("Sigma_a", commutative=False)
    tau = integration_time or sp.Symbol("tau", real=True)

    def convolution(left: Any, right: Any) -> sp.Expr:
        return time_convolution_symbolic(
            left,
            right,
            t,
            tp,
            integration_time=tau,
            lower=lower,
            upper=upper,
        )

    green_retarded_lesser = convolution(g_r, s_l)
    green_lesser_advanced = convolution(g_l, s_a)
    self_energy_retarded_lesser = convolution(s_r, g_l)
    # The fourth ordered term is ``Sigma^< * G^a``, matching the docstring and
    # the numerical kernel in :func:`two_time_kbe_collision_integral`.
    self_energy_lesser_advanced = convolution(s_l, g_a)
    collision = (
        green_retarded_lesser
        + green_lesser_advanced
        - self_energy_retarded_lesser
        - self_energy_lesser_advanced
    )
    return {
        "green_retarded_lesser": green_retarded_lesser,
        "green_lesser_advanced": green_lesser_advanced,
        "self_energy_retarded_lesser": self_energy_retarded_lesser,
        "self_energy_lesser_advanced": self_energy_lesser_advanced,
        "collision": collision,
        "equal_time_collision": collision.subs(tp, t),
    }


def kadanoff_baym_continuity_symbolic(
    *,
    green_retarded: Any | None = None,
    green_advanced: Any | None = None,
    green_lesser: Any | None = None,
    self_energy_retarded: Any | None = None,
    self_energy_lesser: Any | None = None,
    self_energy_advanced: Any | None = None,
    hamiltonian: Any | None = None,
    observable: Any | None = None,
    initial_correlation_source: Any | None = None,
    time: sp.Symbol | None = None,
    time_prime: sp.Symbol | None = None,
    integration_time: sp.Symbol | None = None,
    lower: Any = -sp.oo,
    upper: Any = sp.oo,
) -> dict[str, sp.Expr | sp.Equality]:
    r"""Generate the symbolic equal-time charge/spin continuity identity.

    The density convention is ``rho(t) = -i G^<(t,t)`` and the matrix balance
    is ``d rho/dt = -i[h,rho] + C(t,t) + I_ic(t)``.  ``observable`` may be a
    charge or spin operator; its projection is ``Tr[O (...) ]``.  An explicit
    ``initial_correlation_source`` keeps the vertical Keldysh branch visible
    instead of silently folding it into a numerical residual.  This is an
    identity generator only: a finite-grid or non-conserving approximation is
    not declared conserving by this routine.
    """

    t = time or sp.Symbol("t", real=True)
    tp = time_prime or sp.Symbol("t_prime", real=True)
    g_l = green_lesser or sp.Function("G_lesser", commutative=False)
    h = hamiltonian or sp.Function("h", commutative=False)
    rho = -sp.I * _kernel_value(g_l, t, t)
    if callable(h) and not isinstance(h, sp.FunctionClass):
        h_t = sp.sympify(h(t))
    elif isinstance(h, sp.FunctionClass):
        h_t = h(t)
    else:
        h_t = sp.sympify(h)
    coherent = -sp.I * (h_t * rho - rho * h_t)
    collision_data = kadanoff_baym_collision_integral_symbolic(
        green_retarded=green_retarded,
        green_advanced=green_advanced,
        green_lesser=g_l,
        self_energy_retarded=self_energy_retarded,
        self_energy_lesser=self_energy_lesser,
        self_energy_advanced=self_energy_advanced,
        time=t,
        time_prime=tp,
        integration_time=integration_time,
        lower=lower,
        upper=upper,
    )
    collision = collision_data["equal_time_collision"]
    source = sp.Integer(0) if initial_correlation_source is None else sp.sympify(initial_correlation_source)
    rhs = coherent + collision + source
    density_rate = sp.Derivative(rho, t)
    result: dict[str, sp.Expr | sp.Equality] = {
        "density": rho,
        "density_rate": density_rate,
        "coherent_rate": coherent,
        "collision_rate": collision,
        "initial_correlation_source": source,
        "continuity": sp.Eq(density_rate, rhs),
    }
    if observable is not None:
        operator = sp.sympify(observable)
        trace = sp.Function("Tr")
        result.update(
            {
                "observable": operator,
                "observable_coherent_rate": trace(operator * coherent),
                "observable_collision_rate": trace(operator * collision),
                "observable_source": trace(operator * source),
                "observable_continuity": sp.Eq(
                    trace(operator * density_rate),
                    trace(operator * rhs),
                ),
            }
        )
    return result


def electron_boson_scba_symbolic(
    *,
    energy: sp.Symbol | None = None,
    boson_frequency: Any = sp.Symbol("omega_0", positive=True),
    boson_occupation: Any = sp.Symbol("N_B", nonnegative=True),
    coupling: Any = sp.Symbol("V", commutative=True),
    lesser_green: Any | None = None,
    greater_green: Any | None = None,
) -> dict[str, sp.Expr]:
    """Return scalar Einstein-mode SCBA lesser/greater formulas symbolically."""
    omega = energy or sp.Symbol("omega", real=True)
    g_lesser = lesser_green or sp.Function("G_lesser")
    g_greater = greater_green or sp.Function("G_greater")
    n_bose = sp.sympify(boson_occupation)
    frequency = sp.sympify(boson_frequency)
    vertex = sp.sympify(coupling)
    sigma_lesser = vertex**2 * (
        n_bose * _energy_value(g_lesser, omega - frequency)
        + (n_bose + 1) * _energy_value(g_lesser, omega + frequency)
    )
    sigma_greater = vertex**2 * (
        n_bose * _energy_value(g_greater, omega + frequency)
        + (n_bose + 1) * _energy_value(g_greater, omega - frequency)
    )
    return {
        "lesser": sp.simplify(sigma_lesser),
        "greater": sp.simplify(sigma_greater),
        "retarded_discontinuity": sp.simplify(sigma_greater - sigma_lesser),
    }


def hubbard_second_born_self_energy_symbolic(
    *,
    interaction_u: Any = sp.Symbol("U", real=True),
    time: sp.Symbol | None = None,
    time_prime: sp.Symbol | None = None,
    spin: int = 0,
    opposite_spin: int = 1,
    green_lesser: Any | None = None,
    green_greater: Any | None = None,
) -> dict[str, sp.Expr]:
    r"""Return local Hubbard second-Born Keldysh components symbolically.

    For a density-density pair ``(spin, opposite_spin)`` the correlation-only
    second-Born closure is

    ``Sigma_s^< = U^2 G_s^< G_o^< G_o^>(t',t)`` and
    ``Sigma_s^> = U^2 G_s^> G_o^> G_o^<(t',t)``.

    The retarded component is the causal discontinuity.  Hartree terms are a
    separate instantaneous contribution and are intentionally not hidden in
    this memory kernel.
    """

    t = time or sp.Symbol("t", real=True)
    tp = time_prime or sp.Symbol("t_prime", real=True)
    u = sp.sympify(interaction_u)
    lesser = sp.Function("G_lesser") if green_lesser is None else green_lesser
    greater = sp.Function("G_greater") if green_greater is None else green_greater

    def component(function: Any, channel: int, left: Any, right: Any) -> sp.Expr:
        if callable(function):
            return sp.sympify(function(channel, left, right))
        if isinstance(function, sp.FunctionClass):
            return function(channel, left, right)
        return sp.sympify(function)

    sigma_lesser = u**2 * component(lesser, spin, t, tp) * component(lesser, opposite_spin, t, tp) * component(greater, opposite_spin, tp, t)
    sigma_greater = u**2 * component(greater, spin, t, tp) * component(greater, opposite_spin, t, tp) * component(lesser, opposite_spin, tp, t)
    causal = sp.Function("theta")(t - tp)
    return {
        "lesser": sp.simplify(sigma_lesser),
        "greater": sp.simplify(sigma_greater),
        "retarded": sp.simplify(causal * (sigma_greater - sigma_lesser)),
        "retarded_discontinuity": sp.simplify(sigma_greater - sigma_lesser),
    }


def hubbard_hartree_self_energy_symbolic(
    *,
    interaction_u: Any = sp.Symbol("U", real=True),
    time: sp.Symbol | None = None,
    time_prime: sp.Symbol | None = None,
    spin: int = 0,
    opposite_spin: int = 1,
    density: Any | None = None,
) -> dict[str, sp.Expr]:
    r"""Return the instantaneous density-density Hubbard Hartree layer.

    The finite-grid KBE implementation collocates

    ``Sigma_H,s^r(t,t') = U n_o(t) delta(t-t')``

    and keeps the lesser/greater correlation kernels zero.  Exchange is absent
    for opposite-spin local Hubbard orbitals; the second-Born correlation
    layer is supplied separately by :func:`hubbard_second_born_self_energy_symbolic`.
    """

    t = time or sp.Symbol("t", real=True)
    tp = time_prime or sp.Symbol("t_prime", real=True)
    u = sp.sympify(interaction_u)
    occupation = sp.Function("n") if density is None else density
    if callable(occupation):
        n_opposite = sp.sympify(occupation(opposite_spin, t))
    elif isinstance(occupation, sp.FunctionClass):
        n_opposite = occupation(opposite_spin, t)
    else:
        n_opposite = sp.sympify(occupation)
    potential = sp.simplify(u * n_opposite)
    delta = sp.DiracDelta(t - tp)
    return {
        "potential": potential,
        "retarded": potential * delta,
        "advanced": potential * delta,
        "lesser": sp.Integer(0),
        "greater": sp.Integer(0),
    }


def hubbard_second_born_self_energy_mixed_symbolic(
    *,
    interaction_u: Any = sp.Symbol("U", real=True),
    time: sp.Symbol | None = None,
    imaginary_time: sp.Symbol | None = None,
    spin: int = 0,
    opposite_spin: int = 1,
    green_rceil: Any | None = None,
    green_lceil: Any | None = None,
) -> dict[str, sp.Expr]:
    r"""Return the local Hubbard second-Born mixed contour component.

    For a real-time point ``t`` and a vertical-branch point ``tau`` the
    correlation self-energy is

    ``Sigma^rceil_s(t,tau) = U**2 G^rceil_s(t,tau)
    G^rceil_o(t,tau) G^lceil_o(tau,t)``.

    The mixed kernel is needed for the interacting initial-correlation source;
    it is not reconstructed from a real-time residual.
    """

    t = time or sp.Symbol("t", real=True)
    tau = imaginary_time or sp.Symbol("tau", real=True)
    u = sp.sympify(interaction_u)
    rceil = sp.Function("G_rceil") if green_rceil is None else green_rceil
    lceil = sp.Function("G_lceil") if green_lceil is None else green_lceil

    def value(function: Any, channel: int, left: Any, right: Any) -> sp.Expr:
        if callable(function):
            return sp.sympify(function(channel, left, right))
        if isinstance(function, sp.FunctionClass):
            return function(channel, left, right)
        return sp.sympify(function)

    mixed = u**2 * value(rceil, spin, t, tau) * value(rceil, opposite_spin, t, tau) * value(lceil, opposite_spin, tau, t)
    return {"mixed": sp.simplify(mixed), "u_scaling": sp.Integer(2)}


def hubbard_second_born_self_energy_matsubara_symbolic(
    *,
    interaction_u: Any = sp.Symbol("U", real=True),
    imaginary_time: sp.Symbol | None = None,
    imaginary_time_prime: sp.Symbol | None = None,
    spin: int = 0,
    opposite_spin: int = 1,
    green_matsubara: Any | None = None,
) -> dict[str, sp.Expr]:
    r"""Return the local Hubbard second-Born Matsubara closure symbolically.

    The fermionic imaginary-time convention gives

    ``Sigma^M_s(tau,tau') = -U**2 G^M_s(tau,tau')
    G^M_o(tau,tau') G^M_o(tau',tau)``.
    """

    tau = imaginary_time or sp.Symbol("tau", real=True)
    tau_prime = imaginary_time_prime or sp.Symbol("tau_prime", real=True)
    u = sp.sympify(interaction_u)
    green = sp.Function("G_M") if green_matsubara is None else green_matsubara

    def value(function: Any, channel: int, left: Any, right: Any) -> sp.Expr:
        if callable(function):
            return sp.sympify(function(channel, left, right))
        if isinstance(function, sp.FunctionClass):
            return function(channel, left, right)
        return sp.sympify(function)

    sigma = -u**2 * value(green, spin, tau, tau_prime) * value(green, opposite_spin, tau, tau_prime) * value(green, opposite_spin, tau_prime, tau)
    return {
        "matsubara": sp.simplify(sigma),
        "u_scaling": sp.Integer(2),
        "convention": "fermionic G^M with explicit minus sign",
    }
