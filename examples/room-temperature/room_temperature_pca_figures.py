"""Room-temperature figures for the latent variable chapter.

Four committed PNGs, replacing the base-R output of ``temperature-data.R``,
the lattice cube it drew, ``room-temperature-plots.py``, and the Python 2
scripts that pasted panels together (``temperature-data-combine.py``,
``room-temperature-plots-combine.py``):

- ``room-temperature-plots.png``: the four thermometers over three days,
  one panel each, which the chapter asks the reader to explain.
- ``temperature-2d-and-3d-plot.png``: two of the thermometers against each
  other, and three of them as a data swarm, side by side.
- ``temperatures-first-loading.png``: the first loading vector, which
  spreads its weight evenly over four correlated variables.
- ``temperatures-SPE-after-one-PC.png``: the squared prediction error of
  each observation against a one-component model, with a 95% limit.

All four are computed from the published dataset rather than regenerated
from a random seed, which is what ``room-temperature-plots.py`` did, so the
figures and the reader's own download agree.

The PCA is on autoscaled data by singular value decomposition, which is
what ``prcomp(temps, scale=TRUE)`` did.

The 95% limit on the SPE plot follows the original: the 95th percentile of
the SPE values from the periods that look ordinary (observations 1 to 40
and 60 to 120), rather than of all the data, so that the upset itself does
not inflate the limit it is being judged against. The limit was drawn as a
bare red line before, with no statement of where it came from; it is now
labelled.

Defects in the originals corrected here:

- The loading bar plot ran from -0.7 to +0.7 with no zero line and four
  identical bars, which made a figure about "the weights are spread
  evenly" hard to read. The four values are printed and the bars labelled.
- The two 2D and 3D panels were pasted together by a script that needed a
  ``transparent-pixel.png`` spacer file. They are drawn as one figure.
- The 2D panel was labelled "Front left" against "Back left" while plotting
  the second column of a frame built as front left, front right, back left.
  It plotted front left against front right. The axis labels now match the
  data.

Usage
-----
    uv run --with numpy --with pandas --with matplotlib python room_temperature_pca_figures.py [output_dir]
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE = "#0072B2"
BLUE_FILL = "#B3D4EA"
ORANGE = "#E69F00"
VERMILLION = "#D55E00"
GREEN = "#009E73"
GREY = "#666666"
GRID = "#DDDDDD"

DPI = 300
HERE = pathlib.Path(__file__).parent
URL = "https://openmv.net/file/room-temperature.csv"
CORNERS = ["FrontLeft", "FrontRight", "BackLeft", "BackRight"]
PRETTY = {"FrontLeft": "Front left", "FrontRight": "Front right",
          "BackLeft": "Back left", "BackRight": "Back right"}
# Readings are half-hourly over three days.
PER_DAY = 48

mpl.rcParams.update(
    {
        "font.size": 16,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 18,
        "axes.labelsize": 17,
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


def temperatures() -> pd.DataFrame:
    try:
        data = pd.read_csv(URL)
    except Exception:  # noqa: BLE001 - the committed copy is the fallback
        data = pd.read_csv(HERE / "room-temperature.csv")
    return data[CORNERS]


def autoscale(data: pd.DataFrame) -> pd.DataFrame:
    return (data - data.mean()) / data.std(ddof=1)


def principal_components(scaled: pd.DataFrame):
    left, singular_values, right = np.linalg.svd(scaled.to_numpy(), full_matrices=False)
    loadings = right.T
    scores = left * singular_values
    if loadings[0, 0] < 0:
        loadings[:, 0] *= -1
        scores[:, 0] *= -1
    explained = singular_values**2 / np.sum(singular_values**2)
    return scores, loadings, explained


def sequence_plots(data: pd.DataFrame, outdir: pathlib.Path) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True, sharey=True)
    for ax, corner in zip(axes, CORNERS):
        ax.grid(color=GRID, linewidth=0.8)
        ax.plot(np.arange(1, len(data) + 1), data[corner], color=BLUE, linewidth=1.6)
        ax.set_ylabel("K", rotation=0, labelpad=14)
        ax.annotate(PRETTY[corner], (0.012, 0.08), xycoords="axes fraction",
                    fontsize=16, color=BLUE)
        for boundary in range(PER_DAY, len(data), PER_DAY):
            ax.axvline(boundary, color=GREY, linewidth=1.0, linestyle=":")
    axes[0].set_title("Temperature of a room, measured at the 4 corners")
    axes[-1].set_xlabel("Sequence order (half-hourly readings over 3 days)")
    axes[-1].set_xlim(0, len(data) + 1)
    fig.tight_layout()
    save(fig, outdir, "room-temperature-plots.png")


def two_and_three_dimensions(data: pd.DataFrame, outdir: pathlib.Path) -> None:
    fig = plt.figure(figsize=(15, 7))

    ax = fig.add_subplot(1, 2, 1)
    ax.grid(color=GRID, linewidth=0.8)
    ax.plot(data["FrontLeft"], data["BackLeft"], "o", markersize=8, color=BLUE)
    ax.set_xlabel("Front left temperatures [K]")
    ax.set_ylabel("Back left temperatures [K]")
    ax.set_title("K = 2 variables", fontsize=17)

    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.scatter(data["FrontLeft"], data["FrontRight"], data["BackLeft"],
               s=26, color=BLUE, depthshade=False)
    ax.set_xlabel("Front left", labelpad=12)
    ax.set_ylabel("Front right", labelpad=12)
    ax.set_zlabel("Back left", labelpad=10)
    ax.set_title("K = 3 variables", fontsize=17)
    ax.view_init(elev=18, azim=-60)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_major_locator(mpl.ticker.MaxNLocator(4))
        axis.pane.set_facecolor("white")
        axis.pane.set_edgecolor(GRID)
        axis._axinfo["grid"]["color"] = GRID
    fig.tight_layout()
    save(fig, outdir, "temperature-2d-and-3d-plot.png")


def first_loading(loadings: np.ndarray, outdir: pathlib.Path) -> None:
    values = loadings[:, 0]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.bar([PRETTY[c] for c in CORNERS], values, color=BLUE_FILL, edgecolor=BLUE,
           linewidth=1.6, width=0.6)
    for position, value in enumerate(values):
        ax.annotate(f"{value:+.3f}", (position, value), ha="center",
                    va="bottom" if value > 0 else "top",
                    xytext=(0, 6 if value > 0 else -6), textcoords="offset points",
                    fontsize=15)
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_ylim(-0.7, 0.7)
    ax.set_ylabel("Loadings: component 1")
    save(fig, outdir, "temperatures-first-loading.png")


def spe_plot(scaled: pd.DataFrame, scores: np.ndarray, loadings: np.ndarray,
             outdir: pathlib.Path) -> None:
    residuals = scaled.to_numpy() - np.outer(scores[:, 0], loadings[:, 0])
    spe = (residuals**2).sum(axis=1)
    # The limit comes from the periods that look ordinary, so that the upset
    # does not inflate the limit it is judged against.
    ordinary = np.concatenate([spe[:40], spe[59:120]])
    limit = float(np.quantile(ordinary, 0.95))
    above = np.flatnonzero(spe > limit) + 1
    print(f"SPE after 1 component: 95% limit = {limit:.3f}")
    print(f"observations above the limit: {', '.join(str(i) for i in above)}")
    print(f"largest SPE = {spe.max():.2f} at observation {spe.argmax() + 1}")

    order = np.arange(1, len(spe) + 1)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.grid(color=GRID, linewidth=0.8)
    ax.plot(order, spe, "-", color=GREY, linewidth=1.0)
    ax.plot(order, spe, "o", markersize=6, markerfacecolor="white",
            markeredgecolor=BLUE, markeredgewidth=1.4)
    ax.plot(above, spe[above - 1], "o", markersize=7, color=VERMILLION)
    ax.axhline(limit, color=VERMILLION, linewidth=2)
    ax.annotate(f"95% limit = {limit:.2f}", (2, limit), ha="left", va="bottom",
                xytext=(0, 8), textcoords="offset points", color=VERMILLION, fontsize=15)
    ax.set_xlabel("Time order")
    ax.set_ylabel("SPE after\n1 component")
    ax.set_xlim(0, len(spe) + 1)
    save(fig, outdir, "temperatures-SPE-after-one-PC.png")


def main(outdir: pathlib.Path) -> None:
    data = temperatures()
    scaled = autoscale(data)
    scores, loadings, explained = principal_components(scaled)

    print(f"{len(data)} observations of {data.shape[1]} thermometers")
    print("correlations:")
    print(data.corr().round(3).to_string())
    print("p1 = " + ", ".join(f"{v:+.3f}" for v in loadings[:, 0]))
    print(f"variance explained by the first component: {100 * explained[0]:.1f}%")

    sequence_plots(data, outdir)
    two_and_three_dimensions(data, outdir)
    first_loading(loadings, outdir)
    spe_plot(scaled, scores, loadings, outdir)


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE)
