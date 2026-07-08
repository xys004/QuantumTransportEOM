"""Journal-quality figures for the CISS voltage-probe paper (PRB format).

Reads CSVs from data/ (produced by generate_data_and_figures.py and
extended_scans.py) and writes PDF+PNG figures to figures/.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).absolute().parent
DATA = OUT / "data"
FIGURES = OUT / "figures"
FIGURES.mkdir(exist_ok=True)

# --- style: PRB column widths, Okabe-Ito palette, STIX math ---
COL = 3.404  # inches, single column
DBL = 7.057  # inches, double column

mpl.rcParams.update({
    "font.size": 8.0,
    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "legend.fontsize": 7.0,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "lines.linewidth": 1.1,
    "lines.markersize": 3.6,
    "legend.frameon": False,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

# Okabe-Ito
C_BLUE = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN = "#009E73"
C_VERM = "#D55E00"
C_PURPLE = "#CC79A7"
C_SKY = "#56B4E9"
C_GRAY = "#7F7F7F"


def read(name: str) -> list[dict[str, str]]:
    with (DATA / name).open() as handle:
        return list(csv.DictReader(handle))


def col(rows, key) -> np.ndarray:
    return np.array([float(r[key]) for r in rows])


def save(fig, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.pdf")
    fig.savefig(FIGURES / f"{stem}.png")
    plt.close(fig)
    print("wrote", stem)


PANEL_KW = dict(fontsize=9, fontweight="bold", va="top", ha="left")


def label_panel(ax, text, x=0.03, y=0.96):
    ax.text(x, y, text, transform=ax.transAxes, **PANEL_KW)


# ---------------------------------------------------------------- Fig 2: controls
def fig_controls() -> None:
    rows = read("mechanism_controls.csv")
    order = ["coherent", "elastic_probe", "voltage_candidate", "voltage_chi_flip",
             "voltage_lambda0", "voltage_pFM0"]
    rows = sorted(rows, key=lambda r: order.index(r["case"]))
    values = col(rows, "A_current")
    labels = ["coherent", "elastic\nprobes", "voltage\n$\\chi=+1$",
              "voltage\n$\\chi=-1$", "voltage\n$\\lambda=0$", "voltage\n$p_{\\rm FM}=0$"]
    colors = [C_GRAY, C_SKY, C_BLUE, C_ORANGE, C_GREEN, C_PURPLE]

    fig, ax = plt.subplots(figsize=(COL, 2.4))
    x = np.arange(len(values))
    ax.bar(x, values * 1e3, color=colors, width=0.62, zorder=3)
    ax.axhline(0.0, color="black", linewidth=0.6)
    ax.set_xticks(x, labels, fontsize=6.6)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel(r"$A_M\ (\times 10^{-3})$")
    ax.set_ylim(-4.4, 4.4)
    ax.grid(True, axis="y", alpha=0.25, linewidth=0.4, zorder=0)
    for xi, v in zip(x, values):
        if abs(v) < 1e-10:
            ax.annotate("0 (exact)", (xi, 0.0), xytext=(0, 5), textcoords="offset points",
                        ha="center", fontsize=6.5, color="0.25")
        else:
            off = 5 if v > 0 else -11
            ax.annotate(f"{v*1e3:+.2f}", (xi, v * 1e3), xytext=(0, off),
                        textcoords="offset points", ha="center", fontsize=6.5, color="0.25")
    save(fig, "fig_mechanism_controls")


# ---------------------------------------------------------------- Fig 3: scaling laws
def fig_scaling() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(DBL, 2.15))

    # (a) bias
    rows = read("bias_scan.csv")
    bias = col(rows, "bias")
    a = col(rows, "A_current")
    di = col(rows, "Delta_I")
    order = np.argsort(bias)
    bias, a, di = bias[order], a[order], di[order]
    ax = axes[0]
    ax.plot(bias, a * 1e3, "o-", color=C_BLUE, label=r"$A_M$")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel(r"bias $eV/t$")
    ax.set_ylabel(r"$A_M\ (\times 10^{-3})$")
    # small-bias linear guide through origin, fitted on |V| <= 0.1
    mask = np.abs(bias) <= 0.101
    slope = np.sum(a[mask] * np.abs(bias[mask])) / np.sum(bias[mask] ** 2)
    vv = np.linspace(-0.42, 0.42, 100)
    ax.plot(vv, slope * np.abs(vv) * 1e3, "--", color=C_GRAY, linewidth=0.9,
            label=r"$\propto |V|$")
    ax.legend(loc="lower center")
    label_panel(ax, "(a)")
    axins = ax.inset_axes([0.60, 0.56, 0.37, 0.40])
    axins.plot(bias, di * 1e3, "s-", color=C_VERM, markersize=2.4, linewidth=0.8)
    axins.axhline(0, color="black", linewidth=0.4)
    axins.set_title(r"$\Delta I\ (\times 10^{-3})$", fontsize=6)
    axins.tick_params(labelsize=5.5)

    # (b) lambda
    rows = read("lambda_scan.csv")
    lam = col(rows, "lambda_soc")
    a = col(rows, "A_current")
    ax = axes[1]
    ax.plot(lam, a * 1e3, "o-", color=C_BLUE, label=r"$A_M$")
    coeff = a[5] / lam[5] ** 2  # anchor at lambda=0.12
    ll = np.linspace(0, 0.25, 100)
    ax.plot(ll, coeff * ll ** 2 * 1e3, "--", color=C_GRAY, linewidth=0.9,
            label=r"$\propto \lambda^2$")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel(r"$\lambda/t$")
    ax.legend(loc="lower left")
    label_panel(ax, "(b)", x=0.85)

    # (c) polarization
    rows = read("polarization_scan.csv")
    pol = col(rows, "polarization")
    a = col(rows, "A_current")
    ax = axes[2]
    ax.plot(pol, a * 1e3, "o-", color=C_BLUE, label=r"$A_M$")
    slope_p = a[2] / pol[2]
    pp = np.linspace(0, 1.0, 50)
    ax.plot(pp, slope_p * pp * 1e3, "--", color=C_GRAY, linewidth=0.9,
            label=r"$\propto p_{\rm FM}$")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel(r"$p_{\rm FM}$")
    ax.legend(loc="lower left")
    label_panel(ax, "(c)", x=0.85)

    fig.tight_layout(w_pad=1.4)
    save(fig, "fig_scaling")


# ---------------------------------------------------------------- Fig 4: probe engineering
def fig_probe_engineering() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(DBL, 2.5), width_ratios=[1.25, 1.0])

    # (a) gamma scan by probe kind, chi=+1 solid, chi=-1 open/dashed
    rows = read("gamma_probe_scan.csv")
    styles = {"tau_plus": (C_ORANGE, "s"), "all": (C_BLUE, "o"), "rho_plus": (C_GREEN, "^")}
    names = {"tau_plus": r"$\tau_+$ selective", "all": "uniform", "rho_plus": r"$\rho_+$ selective"}
    ax = axes[0]
    for kind, (color, marker) in styles.items():
        for chirality, ls, fill in ((+1, "-", color), (-1, "--", "white")):
            sub = [r for r in rows if r["probe_kind"] == kind and int(r["chirality"]) == chirality]
            sub.sort(key=lambda r: float(r["gamma_probe"]))
            g = col(sub, "gamma_probe")
            a = col(sub, "A_current")
            label = names[kind] if chirality == +1 else None
            ax.plot(g, a * 1e3, ls, marker=marker, color=color,
                    markerfacecolor=fill, markeredgecolor=color, markeredgewidth=0.7,
                    label=label)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel(r"$\Gamma_p/t$")
    ax.set_ylabel(r"$A_M\ (\times 10^{-3})$")
    handles, labels_ = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    handles += [Line2D([], [], color="0.3", ls="-", label=r"$\chi=+1$"),
                Line2D([], [], color="0.3", ls="--", label=r"$\chi=-1$")]
    ax.legend(handles=handles, loc="upper left", ncols=2, columnspacing=1.0,
              handlelength=1.6, bbox_to_anchor=(0.02, 1.0))
    label_panel(ax, "(a)", x=0.90, y=0.14)

    # (b) fine detuning map
    rows = read("detuning_map_fine.csv")
    chain = sorted({float(r["chain_detuning"]) for r in rows})
    channel = sorted({float(r["channel_detuning"]) for r in rows})
    grid = np.zeros((len(channel), len(chain)))
    for r in rows:
        grid[channel.index(float(r["channel_detuning"])), chain.index(float(r["chain_detuning"]))] = float(r["A_current"])
    vmax = np.max(np.abs(grid)) * 1e3
    ax = axes[1]
    im = ax.imshow(grid * 1e3, origin="lower", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   extent=[min(chain), max(chain), min(channel), max(channel)],
                   aspect="auto", interpolation="bicubic")
    ax.plot([0.5], [1.2], marker="*", color="black", markersize=8)
    ax.set_xlabel(r"$\Delta_\rho/t$")
    ax.set_ylabel(r"$\Delta_\tau/t$")
    cb = fig.colorbar(im, ax=ax, pad=0.02)
    cb.set_label(r"$A_M\ (\times 10^{-3})$")
    cb.ax.tick_params(labelsize=6.5)
    label_panel(ax, "(b)", x=0.04, y=0.96)

    fig.tight_layout(w_pad=1.6)
    save(fig, "fig_probe_engineering")


# ---------------------------------------------------------------- Fig 5: robustness (N, T)
def fig_robustness() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(COL, 3.4))

    rows = read("length_scan.csv")
    n = col(rows, "n_sites")
    a = col(rows, "A_current")
    ax = axes[0]
    markerline, stemlines, baseline = ax.stem(n, a * 1e3)
    plt.setp(markerline, color=C_BLUE, markersize=4)
    plt.setp(stemlines, color=C_BLUE, linewidth=1.0)
    plt.setp(baseline, color="black", linewidth=0.5)
    for nc, ha, dx in ((6, "left", 6), (12, "right", -6)):
        ax.annotate("commensurate", (nc, float(a[list(n).index(nc)]) * 1e3),
                    xytext=(dx, -1), textcoords="offset points", fontsize=6,
                    color="0.3", ha=ha)
    ax.set_xlabel(r"number of rungs $N$")
    ax.set_ylabel(r"$A_M\ (\times 10^{-3})$")
    ax.set_xticks([4, 5, 6, 7, 8, 10, 12])
    label_panel(ax, "(a)", x=0.04, y=0.20)

    rows = read("temperature_scan.csv")
    t = col(rows, "temperature")
    a = col(rows, "A_current")
    ax = axes[1]
    ax.plot(t, -a * 1e3, "o-", color=C_VERM)
    ax.set_xlabel(r"temperature $k_BT/t$")
    ax.set_ylabel(r"$-A_M\ (\times 10^{-3})$")
    ax.set_xscale("log")
    label_panel(ax, "(b)", x=0.04, y=0.20)

    fig.tight_layout(h_pad=1.2)
    save(fig, "fig_robustness")


if __name__ == "__main__":
    fig_controls()
    fig_scaling()
    fig_probe_engineering()
    fig_robustness()
