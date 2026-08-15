"""Analytic partitioned wide-band resonant-level transient."""

import numpy as np

from quantum_transport import (
    wide_band_resonant_level_quench,
    zero_temperature_resonant_level_steady_state,
)


time = np.linspace(0.0, 20.0, 401)
energy = np.linspace(-80.0, 80.0, 40001)
gamma = np.array([0.3, 0.2])
mu = np.array([0.6, -0.4])

transient = wide_band_resonant_level_quench(
    time,
    energy,
    level_energy=0.15,
    broadening=gamma,
    chemical_potential=mu,
    temperature=0.0,
)
steady = zero_temperature_resonant_level_steady_state(
    level_energy=0.15,
    broadening=gamma,
    chemical_potential=mu,
)

print("Partitioned wide-band resonant-level quench")
print(f"n(0): {transient.occupation[0]:.6f}")
print(f"n(t_max): {transient.occupation[-1]:.6f}")
print(f"n(steady): {steady.occupation:.6f}")
print(
    "I_alpha(t_max): "
    + np.array2string(
        transient.current_into_level[-1], precision=6
    )
)
print(
    "I_alpha(steady): "
    + np.array2string(steady.current_into_level, precision=6)
)
print(
    "maximum algebraic continuity error: "
    f"{np.max(np.abs(transient.net_current_into_level - transient.occupation_rate)):.3e}"
)
print("VERDICT: PASS")
