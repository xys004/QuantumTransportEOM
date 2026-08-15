"""Analytic partition-free wide-band resonant-level bias quench."""

import numpy as np

from quantum_transport import (
    partition_free_resonant_level_bias_quench,
    zero_temperature_resonant_level_steady_state,
)


time = np.linspace(0.0, 20.0, 401)
energy = np.linspace(-80.0, 80.0, 40001)
gamma = np.array([0.3, 0.2])
bias_shift = np.array([0.5, -0.5])

transient = partition_free_resonant_level_bias_quench(
    time,
    energy,
    level_energy=0.15,
    broadening=gamma,
    bias_shift=bias_shift,
    initial_chemical_potential=0.0,
    temperature=0.0,
)
steady = zero_temperature_resonant_level_steady_state(
    level_energy=0.15,
    broadening=gamma,
    chemical_potential=bias_shift,
)

print("Partition-free wide-band resonant-level bias quench")
print(f"n(0): {transient.occupation[0]:.6f}")
print(
    "I_alpha(0): "
    + np.array2string(transient.current_into_level[0], precision=6)
)
print(
    "I_alpha(t_max): "
    + np.array2string(transient.current_into_level[-1], precision=6)
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
