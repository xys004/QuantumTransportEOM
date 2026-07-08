from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).absolute().parents[2]
EXAMPLES = ROOT / "examples"
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))

from ciss_ladder_elastic_probe import integrate_with_elastic_probes
from ciss_rho_tau_sigma_ladder import (
    RhoTauSigmaParameters,
    asymmetry,
    build_rho_tau_sigma_ladder,
    ferromagnetic_edge_gamma,
    normal_edge_gamma,
)
from ciss_rho_tau_sigma_voltage_probe import integrate_voltage_probe_case


OUT = Path(__file__).absolute().parent
DATA = OUT / "data"
FIGURES = OUT / "figures"

MU_LEFT = 0.25
MU_RIGHT = -0.15
TEMPERATURE = 0.03
OMEGA_GRID = np.linspace(-4.0, 4.0, 201)
FINE_GRID = np.linspace(-4.0, 4.0, 501)


def ensure_dirs() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write for {path}.")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def voltage_pair(
    params: RhoTauSigmaParameters,
    *,
    gamma_probe: float,
    probe_kind: str,
    polarization: float = 0.65,
    omega_grid: np.ndarray = OMEGA_GRID,
) -> tuple[object, object, float]:
    plus = integrate_voltage_probe_case(
        params,
        gamma_probe=gamma_probe,
        probe_kind=probe_kind,
        polarization=polarization,
        magnetization_label="+z",
        theta=0.0,
        phi=0.0,
        omega_grid=omega_grid,
        mu_left=MU_LEFT,
        mu_right=MU_RIGHT,
        temperature=TEMPERATURE,
    )
    minus = integrate_voltage_probe_case(
        params,
        gamma_probe=gamma_probe,
        probe_kind=probe_kind,
        polarization=polarization,
        magnetization_label="-z",
        theta=np.pi,
        phi=0.0,
        omega_grid=omega_grid,
        mu_left=MU_LEFT,
        mu_right=MU_RIGHT,
        temperature=TEMPERATURE,
    )
    return plus, minus, asymmetry(plus.current, minus.current)


def coherent_pair(params: RhoTauSigmaParameters) -> tuple[float, float, float]:
    device = build_rho_tau_sigma_ladder(params)
    right = normal_edge_gamma(
        params.n_sites,
        params.n_sites - 1,
        2.0,
        chain_weights=(1.0, 0.25),
        channel_weights=(1.0, 0.40),
    )
    currents = []
    for theta in (0.0, np.pi):
        left = ferromagnetic_edge_gamma(params.n_sites, 0, 2.0, 0.65, theta=theta, phi=0.0)
        from quantum_transport import LeadSelfEnergy

        view = device.transport(
            LeadSelfEnergy.wide_band(left, mu=MU_LEFT, name="L"),
            LeadSelfEnergy.wide_band(right, mu=MU_RIGHT, name="R"),
        ).keldysh_view()
        currents.append(view.meir_wingreen_current(OMEGA_GRID, lead="left"))
    return currents[0], currents[1], asymmetry(currents[0], currents[1])


def elastic_pair(params: RhoTauSigmaParameters) -> tuple[float, float, float]:
    device = build_rho_tau_sigma_ladder(params)
    right = normal_edge_gamma(
        params.n_sites,
        params.n_sites - 1,
        2.0,
        chain_weights=(1.0, 0.25),
        channel_weights=(1.0, 0.40),
    )
    probes = [normal_edge_gamma(params.n_sites, site, 0.60) for site in range(1, params.n_sites - 1)]
    currents = []
    for theta in (0.0, np.pi):
        left = ferromagnetic_edge_gamma(params.n_sites, 0, 2.0, 0.65, theta=theta, phi=0.0)
        current, _spin, _residual = integrate_with_elastic_probes(
            device,
            gamma_left=left,
            gamma_right=right,
            probe_gammas=probes,
            omega_grid=OMEGA_GRID,
            mu_left=MU_LEFT,
            mu_right=MU_RIGHT,
        )
        currents.append(current)
    return currents[0], currents[1], asymmetry(currents[0], currents[1])


def generate_gamma_scan() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    gammas = [0.05, 0.10, 0.25, 0.50, 0.80, 1.20]
    for chirality in (+1, -1):
        params = RhoTauSigmaParameters(chirality=chirality, chain_detuning=0.50, channel_detuning=1.20)
        for probe_kind in ("all", "tau_plus", "rho_plus"):
            for gamma_probe in gammas:
                plus, minus, a_current = voltage_pair(params, gamma_probe=gamma_probe, probe_kind=probe_kind)
                rows.append(
                    {
                        "chirality": chirality,
                        "probe_kind": probe_kind,
                        "gamma_probe": gamma_probe,
                        "I_plus": plus.current,
                        "I_minus": minus.current,
                        "A_current": a_current,
                        "Iz_plus": plus.spin_current_z,
                        "Iz_minus": minus.spin_current_z,
                        "mu_probe_plus": plus.mu_probe,
                        "mu_probe_minus": minus.mu_probe,
                        "residual_plus": plus.residual,
                        "residual_minus": minus.residual,
                    }
                )
    write_csv(DATA / "gamma_probe_scan.csv", rows)
    return rows


def generate_detuning_map() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    chain_values = [0.0, 0.25, 0.50, 0.80, 1.20]
    channel_values = [0.0, 0.20, 0.50, 0.80, 1.20]
    for chain_detuning in chain_values:
        for channel_detuning in channel_values:
            params = RhoTauSigmaParameters(
                chirality=+1,
                chain_detuning=chain_detuning,
                channel_detuning=channel_detuning,
            )
            plus, minus, a_current = voltage_pair(params, gamma_probe=0.80, probe_kind="tau_plus")
            rows.append(
                {
                    "chain_detuning": chain_detuning,
                    "channel_detuning": channel_detuning,
                    "I_plus": plus.current,
                    "I_minus": minus.current,
                    "A_current": a_current,
                    "mu_probe_plus": plus.mu_probe,
                    "mu_probe_minus": minus.mu_probe,
                    "residual_plus": plus.residual,
                    "residual_minus": minus.residual,
                }
            )
    write_csv(DATA / "detuning_map_tau_plus_gamma_0p8.csv", rows)
    return rows


def generate_controls() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    params = RhoTauSigmaParameters(chirality=+1, chain_detuning=0.50, channel_detuning=1.20)

    for mechanism, fn in (
        ("coherent", coherent_pair),
        ("elastic_probe", elastic_pair),
    ):
        i_plus, i_minus, a_current = fn(params)
        rows.append(
            {
                "case": mechanism,
                "chirality": +1,
                "lambda_soc": params.lambda_soc,
                "gamma_hybrid": params.gamma_hybrid,
                "polarization": 0.65,
                "I_plus": i_plus,
                "I_minus": i_minus,
                "A_current": a_current,
            }
        )

    for case, control_params, polarization in (
        ("voltage_candidate", params, 0.65),
        ("voltage_chi_flip", RhoTauSigmaParameters(chirality=-1, chain_detuning=0.50, channel_detuning=1.20), 0.65),
        ("voltage_lambda0", RhoTauSigmaParameters(chirality=+1, chain_detuning=0.50, channel_detuning=1.20, lambda_soc=0.0), 0.65),
        ("voltage_pFM0", params, 0.0),
    ):
        plus, minus, a_current = voltage_pair(
            control_params,
            gamma_probe=0.80,
            probe_kind="tau_plus",
            polarization=polarization,
            omega_grid=FINE_GRID,
        )
        rows.append(
            {
                "case": case,
                "chirality": control_params.chirality,
                "lambda_soc": control_params.lambda_soc,
                "gamma_hybrid": control_params.gamma_hybrid,
                "polarization": polarization,
                "I_plus": plus.current,
                "I_minus": minus.current,
                "A_current": a_current,
            }
        )

    write_csv(DATA / "mechanism_controls.csv", rows)
    return rows


def load_numeric_column(rows: list[dict[str, object]], key: str) -> np.ndarray:
    return np.array([float(row[key]) for row in rows], dtype=float)


def plot_gamma_scan(rows: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), sharey=True)
    styles = {"all": "o-", "tau_plus": "s-", "rho_plus": "^-"}
    colors = {"all": "#3b82f6", "tau_plus": "#d97706", "rho_plus": "#059669"}
    for axis, chirality in zip(axes, (+1, -1)):
        subset = [row for row in rows if int(row["chirality"]) == chirality]
        for probe_kind in ("all", "tau_plus", "rho_plus"):
            probe_rows = [row for row in subset if row["probe_kind"] == probe_kind]
            probe_rows.sort(key=lambda row: float(row["gamma_probe"]))
            gamma = load_numeric_column(probe_rows, "gamma_probe")
            asym = load_numeric_column(probe_rows, "A_current")
            axis.plot(gamma, asym, styles[probe_kind], color=colors[probe_kind], label=probe_kind.replace("_", " "))
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(rf"$\chi={chirality:+d}$")
        axis.set_xlabel(r"$\Gamma_p$")
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel(r"$A_M$")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_gamma_probe_scan.pdf")
    fig.savefig(FIGURES / "fig_gamma_probe_scan.png", dpi=220)
    plt.close(fig)


def plot_detuning_map(rows: list[dict[str, object]]) -> None:
    chain_values = sorted({float(row["chain_detuning"]) for row in rows})
    channel_values = sorted({float(row["channel_detuning"]) for row in rows})
    grid = np.zeros((len(channel_values), len(chain_values)), dtype=float)
    for row in rows:
        x = chain_values.index(float(row["chain_detuning"]))
        y = channel_values.index(float(row["channel_detuning"]))
        grid[y, x] = float(row["A_current"])

    vmax = float(np.max(np.abs(grid)))
    fig, axis = plt.subplots(figsize=(4.5, 3.6))
    image = axis.imshow(
        grid,
        origin="lower",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        extent=[min(chain_values), max(chain_values), min(channel_values), max(channel_values)],
        aspect="auto",
    )
    axis.set_xlabel(r"$\Delta_\rho$")
    axis.set_ylabel(r"$\Delta_\tau$")
    axis.set_title(r"$A_M$ for $\tau_+$ voltage probe")
    fig.colorbar(image, ax=axis, label=r"$A_M$")
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_detuning_map.pdf")
    fig.savefig(FIGURES / "fig_detuning_map.png", dpi=220)
    plt.close(fig)


def plot_controls(rows: list[dict[str, object]]) -> None:
    labels = [str(row["case"]).replace("voltage_", "").replace("_", "\n") for row in rows]
    values = load_numeric_column(rows, "A_current")
    colors = ["#64748b", "#94a3b8", "#d97706", "#d97706", "#ef4444", "#ef4444"]
    fig, axis = plt.subplots(figsize=(7.2, 3.4))
    axis.bar(np.arange(len(rows)), values, color=colors[: len(rows)])
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(np.arange(len(rows)), labels, fontsize=8)
    axis.set_ylabel(r"$A_M$")
    axis.set_title("Mechanism and symmetry controls")
    axis.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_mechanism_controls.pdf")
    fig.savefig(FIGURES / "fig_mechanism_controls.png", dpi=220)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    gamma_rows = generate_gamma_scan()
    detuning_rows = generate_detuning_map()
    control_rows = generate_controls()
    plot_gamma_scan(gamma_rows)
    plot_detuning_map(detuning_rows)
    plot_controls(control_rows)
    print(f"Wrote data to {DATA}")
    print(f"Wrote figures to {FIGURES}")


if __name__ == "__main__":
    main()
