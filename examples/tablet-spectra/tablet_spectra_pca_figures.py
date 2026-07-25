"""PCA figures for the tablet-spectra example in the latent variable chapter.

Five committed PNGs, replacing the base-R output of ``spectral-data.R`` and
the Python 2 script ``pharma-spectra.py``:

- ``pharma-spectra.png``: the 460 raw spectra, 650 wavelengths each.
- ``spectral-data-R2-per-variable.png``: how much of each wavelength the
  model explains, after one, two and three components.
- ``spectral-data-SPE-per-tablet.png``: the residual distance of every
  tablet from the model plane, for the same three model sizes.
- ``spectral-data-t1-t2-scoreplot.png``: the score plot with the 95% and
  99% Hotelling's T-squared ellipses.
- ``spectral-data-T2-lineplot.png``: T-squared per tablet against those
  same two limits.

``pharma-spectra.py`` read a JCAMP-DX file from a path on a machine that no
longer exists, so the figure could not be regenerated at all. Everything
here reads ``tablet-spectra.csv`` from openmv.net, with the committed
``spectral-data.csv`` as a fallback, so a reader can reproduce all five.

The model is a PCA by singular value decomposition on the autoscaled data,
which is what ``prcomp(spectra, scale=TRUE)`` did. The T-squared limits are
computed from the F-distribution rather than pasted in as the constants
7.90771 and 11.5244 the R script carried.

Following the original, SPE here is the residual *distance* (the square
root of the sum of the squared residuals of a row), not the sum of squares.

Defects in the originals corrected here:

- The R² legend read "R²: 1st component", "2nd component", "3rd
  component", but the quantity plotted is cumulative: the black line is
  what one and two components explain together. The legend now says so.
- The two T-squared limits were hard-coded constants (7.90771 and
  11.5244) with no derivation, and would silently become wrong if the
  number of components or tablets changed. They are computed from the
  F-distribution, reproducing those two values, and printed when this
  script runs.
- The score plot's ellipses were drawn dotted in red and green, a pairing
  that is hard to tell apart for a colourblind reader; they are now
  distinguished by colour and dash pattern from a colourblind-safe pair.
- The three SPE panels had no limit drawn on them, so there was nothing to
  read a tablet's residual distance against, even though the neighbouring
  T-squared plot carried two limits. Each panel now shows its own 95%
  limit, computed from the SPE values for that number of components.

Usage
-----
    uv run --with numpy --with pandas --with scipy --with matplotlib python tablet_spectra_pca_figures.py [output_dir]
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
GREY = "#666666"
GRID = "#DDDDDD"

DPI = 300
HERE = pathlib.Path(__file__).parent
URL = "https://openmv.net/file/tablet-spectra.csv"
WAVELENGTHS = np.arange(600, 1900, 2)
COMPONENTS = (1, 2, 3)
COLOURS = {1: GREEN, 2: BLUE, 3: VERMILLION}

mpl.rcParams.update(
    {
        "font.size": 16,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
        "axes.axisbelow": True,
    }
)


def save(fig, outdir: pathlib.Path, name: str) -> None:
    fig.savefig(outdir / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outdir / name}")


def spectra() -> np.ndarray:
    """The 460 by 650 matrix of absorbances."""
    try:
        data = pd.read_csv(URL, header=None, index_col=0)
    except Exception:  # noqa: BLE001 - the committed copy is the fallback
        data = pd.read_csv(HERE / "spectral-data.csv", header=None)
    return data.to_numpy(dtype=float)


def autoscale(data: np.ndarray) -> np.ndarray:
    return (data - data.mean(axis=0)) / data.std(axis=0, ddof=1)


def principal_components(scaled: np.ndarray):
    left, singular_values, right = np.linalg.svd(scaled, full_matrices=False)
    scores = left * singular_values
    loadings = right.T
    explained = singular_values**2 / np.sum(singular_values**2)
    # prcomp reports the standard deviation of each score.
    spread = singular_values / np.sqrt(len(scaled) - 1)
    return scores, loadings, explained, spread


def raw_spectra_plot(data: np.ndarray, outdir: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.grid(color=GRID, linewidth=0.8)
    # 460 overlaid curves: a single translucent colour shows the shared
    # shape and the spread around it better than 460 distinct hues.
    ax.plot(WAVELENGTHS, data.T, color=BLUE, linewidth=0.5, alpha=0.25)
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("Absorbance")
    ax.set_xlim(WAVELENGTHS[0], WAVELENGTHS[-1])
    ax.annotate(f"{len(data)} tablets, {data.shape[1]} wavelengths each",
                (0.985, 0.95), xycoords="axes fraction", ha="right", va="top",
                fontsize=15, color=GREY)
    save(fig, outdir, "pharma-spectra.png")


def reconstruction(scaled: np.ndarray, scores: np.ndarray, loadings: np.ndarray, a: int):
    """Fitted values, residual distance per row, and cumulative R² per column."""
    fitted = scores[:, :a] @ loadings[:, :a].T
    residuals = scaled - fitted
    distance = np.sqrt((residuals**2).sum(axis=1))
    r2_per_column = (fitted**2).sum(axis=0) / (scaled**2).sum(axis=0)
    return distance, r2_per_column


def r2_per_wavelength(r2: dict, outdir: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.grid(color=GRID, linewidth=0.8)
    for a in COMPONENTS:
        ax.plot(WAVELENGTHS, r2[a], color=COLOURS[a], linewidth=1 + a,
                label=f"Cumulative $R^2$ after {a} component" + ("s" if a > 1 else ""))
    ax.set_ylim(0, 1.02)
    ax.set_xlim(WAVELENGTHS[0], WAVELENGTHS[-1])
    ax.set_xlabel("Wavelength [nm]")
    ax.set_ylabel("$R^2$ per wavelength")
    ax.legend(loc="lower left", frameon=False)
    save(fig, outdir, "spectral-data-R2-per-variable.png")


def spe_limit(distances: np.ndarray, confidence: float = 0.95) -> float:
    """Confidence limit on the residual distance, as a distance.

    The sum of squared residuals of a row is not chi-squared distributed,
    because the residuals are neither independent nor of equal variance.
    Nomikos and MacGregor (1995) match a scaled chi-squared, g * chi2(h),
    to the mean and variance of the observed sums of squares, and read the
    limit off that. This is the calculation ``process_improve`` uses in
    ``multivariate/_limits.py``; it is repeated here so the script needs
    only numpy and scipy.

    The input and the returned limit are distances (square roots), which is
    what the rest of this script plots, so the values are squared going in
    and the limit square-rooted coming out.
    """
    squared = np.asarray(distances, dtype=float) ** 2
    centre = float(squared.mean())
    variance = float(squared.var(ddof=1))
    g = variance / (2 * centre)
    h = 2 * centre**2 / variance
    return float(np.sqrt(scipy.stats.chi2.ppf(confidence, h) * g))


def spe_per_tablet(spe: dict, outdir: pathlib.Path) -> None:
    tablets = np.arange(1, len(spe[1]) + 1)
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    for ax, a in zip(axes, COMPONENTS):
        limit = spe_limit(spe[a])
        above = int((spe[a] > limit).sum())
        print(f"A = {a}: 95% SPE limit {limit:.2f}, {above} of {len(tablets)} tablets above it")
        ax.grid(color=GRID, linewidth=0.8)
        ax.plot(tablets, spe[a], color=COLOURS[a], linewidth=1.0)
        ax.axhline(limit, color=GREY, linestyle="--", linewidth=1.6)
        ax.annotate(f"95% limit = {limit:.1f}", (2, limit),
                    xytext=(0, 5), textcoords="offset points", ha="left",
                    va="bottom", color=GREY, fontsize=14)
        ax.set_ylabel(f"SPE: A = {a}")
        ax.set_ylim(0, spe[1].max() * 1.05)
    axes[-1].set_xlabel("Tablet number")
    axes[-1].set_xlim(0, len(tablets) + 1)
    fig.tight_layout()
    save(fig, outdir, "spectral-data-SPE-per-tablet.png")


def t2_limit(n_rows: int, n_components: int, confidence: float) -> float:
    """Hotelling's T-squared limit for observations used to build the model.

    Rows that the model was fitted on use A(N-1)/(N-A) times the F value;
    a future observation, not in the fit, uses A(N^2-1)/(N(N-A)) instead.
    These 460 tablets are the fitting set, so the first form applies.
    """
    return float(
        n_components * (n_rows - 1) / (n_rows - n_components)
        * scipy.stats.f.ppf(confidence, n_components, n_rows - n_components)
    )


def ellipse(spread_i: float, spread_j: float, limit: float) -> np.ndarray:
    angle = np.linspace(0, 2 * np.pi, 200)
    return np.column_stack([
        np.sqrt(limit) * spread_i * np.cos(angle),
        np.sqrt(limit) * spread_j * np.sin(angle),
    ])


def score_plot(scores: np.ndarray, spread: np.ndarray, limits: dict,
               outdir: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.grid(color=GRID, linewidth=0.8)
    ax.axhline(0, color="black", linewidth=1.0)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.plot(scores[:, 0], scores[:, 1], "o", markersize=6, markerfacecolor="none",
            markeredgecolor=BLUE, markeredgewidth=1.2)
    for confidence, style in ((0.95, "--"), (0.99, "-")):
        curve = ellipse(spread[0], spread[1], limits[confidence])
        ax.plot(curve[:, 0], curve[:, 1], style,
                color=ORANGE if confidence == 0.95 else VERMILLION,
                linewidth=2, label=f"{100 * confidence:.0f}% limit")
    ax.set_xlabel("$t_1$")
    ax.set_ylabel("$t_2$")
    ax.set_title("Score plot for tablet spectra")
    ax.legend(frameon=False, loc="upper right")
    save(fig, outdir, "spectral-data-t1-t2-scoreplot.png")


def t2_line_plot(t2: np.ndarray, limits: dict, outdir: pathlib.Path) -> None:
    tablets = np.arange(1, len(t2) + 1)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.grid(color=GRID, linewidth=0.8)
    ax.plot(tablets, t2, color=BLUE, linewidth=1.0)
    for confidence, colour, style in ((0.95, ORANGE, "--"), (0.99, VERMILLION, "-")):
        ax.axhline(limits[confidence], color=colour, linestyle=style, linewidth=2)
        ax.annotate(f"{100 * confidence:.0f}% limit", (2, limits[confidence]),
                    xytext=(0, 6), textcoords="offset points", color=colour,
                    fontsize=15, va="bottom")
    ax.set_xlabel("Tablet order")
    ax.set_ylabel("Hotelling's $T^2$")
    ax.set_xlim(0, len(t2) + 1)
    ax.set_ylim(0, max(t2.max(), limits[0.99]) * 1.08)
    save(fig, outdir, "spectral-data-T2-lineplot.png")


def main(outdir: pathlib.Path) -> None:
    data = spectra()
    scaled = autoscale(data)
    scores, loadings, explained, spread = principal_components(scaled)
    n_rows = len(data)

    print(f"{n_rows} tablets, {data.shape[1]} wavelengths")
    print("standard deviation per component: "
          + ", ".join(f"{v:.4f}" for v in spread[:4]))
    print("proportion of variance:           "
          + ", ".join(f"{v:.4f}" for v in explained[:4]))
    print("cumulative proportion:            "
          + ", ".join(f"{v:.4f}" for v in np.cumsum(explained[:4])))

    spe, r2 = {}, {}
    for a in COMPONENTS:
        spe[a], r2[a] = reconstruction(scaled, scores, loadings, a)
        print(f"A = {a}: largest SPE {spe[a].max():.2f}, "
              f"R² over all wavelengths {r2[a].mean():.4f}")

    limits = {c: t2_limit(n_rows, 3, c) for c in (0.95, 0.99)}
    print(f"T² limits with 3 components: 95% {limits[0.95]:.5f}, 99% {limits[0.99]:.5f}")
    t2 = ((scores[:, :3] / spread[:3]) ** 2).sum(axis=1)
    print(f"tablets above the 95% limit: {int((t2 > limits[0.95]).sum())}; "
          f"above the 99% limit: {int((t2 > limits[0.99]).sum())}")

    raw_spectra_plot(data, outdir)
    r2_per_wavelength(r2, outdir)
    spe_per_tablet(spe, outdir)
    score_plot(scores, spread, limits, outdir)
    t2_line_plot(t2, limits, outdir)


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE)
