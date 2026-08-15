"""Feed a symbolic EOM contour directly into the two-time numerical solvers."""

from __future__ import annotations

import numpy as np
import sympy as sp

from quantum_transport import ElectronBosonSCBAConfig, build_eom_hierarchy, f, fd


def main() -> None:
    eps0, eps1, hopping = sp.symbols("eps0 eps1 hopping", real=True)
    hierarchy = build_eom_hierarchy(
        {
            "onsite": eps0 * fd("0") * f("0") + eps1 * fd("1") * f("1"),
            "hopping": hopping * fd("0") * f("1") + hopping * fd("1") * f("0"),
        },
        max_depth=0,
    )
    contour = hierarchy.contour_equations()
    time = np.linspace(0.0, 1.0, 6)
    parameters = {eps0: 0.4, eps1: 0.7, hopping: 0.1}
    density = np.diag([0.2, 0.3])

    direct = contour.propagate_two_time(time, density, parameters=parameters)
    print("Direct solver:", direct.solver, direct.retarded.shape)

    zero_sigma = np.zeros((time.size, time.size, 2, 2), dtype=complex)
    kbe = contour.propagate_two_time(
        time,
        density,
        parameters=parameters,
        self_energy_retarded=zero_sigma,
        self_energy_lesser=zero_sigma,
    )
    print("KBE solver:", kbe.solver, "converged:", kbe.converged)

    scba = contour.propagate_two_time(
        time,
        density,
        parameters=parameters,
        electron_boson_scba=ElectronBosonSCBAConfig(
            coupling=np.diag([0.04, 0.03]),
            boson_frequency=0.8,
            max_iterations=2,
            dyson_iterations=3,
            tolerance=1e-4,
        ),
    )
    print("SCBA solver:", scba.solver, "iterations:", scba.iterations)


if __name__ == "__main__":
    main()
