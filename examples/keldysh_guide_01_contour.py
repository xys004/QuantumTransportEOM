from __future__ import annotations

import sympy as sp

from quantum_transport import (
    contour_delta,
    contour_green_observable,
    contour_heaviside,
    keldysh_leq,
    keldysh_system,
    kt,
)


def main() -> None:
    t, tp = sp.symbols("t tp", real=True)
    k = keldysh_system()
    G = k.green("G")

    t_minus = kt(t, "-")
    tp_minus = kt(tp, "-")
    t_plus = kt(t, "+")
    tp_plus = kt(tp, "+")

    print("Keldysh contour ordering")
    print("t_- <=_K tp_+:", keldysh_leq(t_minus, tp_plus))
    print("t_+ <=_K tp_-:", keldysh_leq(t_plus, tp_minus))

    print("\nContour Heaviside")
    print("Theta_C(t_-, tp_-):", contour_heaviside(t_minus, tp_minus))
    print("Theta_C(t_-, tp_+):", contour_heaviside(t_minus, tp_plus))
    print("Theta_C(t_+, tp_-):", contour_heaviside(t_plus, tp_minus))
    print("Theta_C(t_+, tp_+):", contour_heaviside(t_plus, tp_plus))

    print("\nContour delta")
    print("delta_C(t_-, tp_-):", contour_delta(t_minus, tp_minus))
    print("delta_C(t_+, tp_+):", contour_delta(t_plus, tp_plus))

    contour_g = contour_green_observable(
        G.greater(t, tp),
        G.lesser(t, tp),
        t_minus,
        tp_plus,
    )
    print("\nG_C(t_-, tp_+) =")
    sp.pprint(contour_g.doit())
    print("\nLaTeX:")
    print(contour_g.latex())


if __name__ == "__main__":
    main()
