from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from quantum_transport import physical_simplify_fermionic
from quantum_transport.models import bosonic_harmonic_mode_model, fermionic_single_level_model


def main() -> None:
    out_dir = Path(__file__).resolve().parent / "output"
    out_dir.mkdir(exist_ok=True)

    epsilon, omega0, omega, eta = sp.symbols("epsilon omega0 omega eta", real=True)

    fermion = fermionic_single_level_model(epsilon)
    boson = bosonic_harmonic_mode_model(omega0)

    g_f = physical_simplify_fermionic(
        fermion.retarded(fermion.operators["d"], fermion.operators["d_dag"], omega=omega, eta=eta)
    )
    g_b = boson.retarded(boson.operators["b"], boson.operators["b_dag"], omega=omega, eta=eta)

    g_f_num = sp.lambdify((omega, epsilon, eta), g_f, "numpy")
    g_b_num = sp.lambdify((omega, omega0, eta), g_b, "numpy")

    omega_grid = np.linspace(-4.0, 4.0, 1200)
    eta_value = 0.08
    epsilon_value = 1.0
    omega0_value = 1.5

    gf_vals = g_f_num(omega_grid, epsilon_value, eta_value)
    gb_vals = g_b_num(omega_grid, omega0_value, eta_value)

    spectral_f = -np.imag(gf_vals) / np.pi
    spectral_b = -np.imag(gb_vals) / np.pi

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    axes[0].plot(omega_grid, spectral_f, color="#0f766e", lw=2)
    axes[0].set_title("Fermionic Single-Level Spectral Function")
    axes[0].set_ylabel(r"$A_f(\omega)$")
    axes[0].grid(alpha=0.25)
    axes[0].text(0.02, 0.92, sp.latex(g_f), transform=axes[0].transAxes, fontsize=10)

    axes[1].plot(omega_grid, spectral_b, color="#b45309", lw=2)
    axes[1].set_title("Bosonic Harmonic-Mode Spectral Function")
    axes[1].set_xlabel(r"$\omega$")
    axes[1].set_ylabel(r"$A_b(\omega)$")
    axes[1].grid(alpha=0.25)
    axes[1].text(0.02, 0.92, sp.latex(g_b), transform=axes[1].transAxes, fontsize=10)

    fig.tight_layout()
    fig.savefig(out_dir / "symbolic_model_spectral_functions.png", dpi=160)
    plt.close(fig)

    latex_path = out_dir / "symbolic_model_formulas.tex"
    latex_path.write_text(
        "\\section*{Symbolic Model Formulas}\n"
        + "\\[ G_f^r(\\omega) = " + sp.latex(g_f) + " \\]\n"
        + "\\[ G_b^r(\\omega) = " + sp.latex(g_b) + " \\]\n"
    )

    print("Saved plot:", out_dir / "symbolic_model_spectral_functions.png")
    print("Saved LaTeX formulas:", latex_path)


if __name__ == "__main__":
    main()
