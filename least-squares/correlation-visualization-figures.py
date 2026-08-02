"""Scatterplot matrices and correlation heat maps for the least-squares chapter.

Six committed PNGs for the covariance-and-correlation section, which
previously described these displays without showing any of them:

- ``gas-cylinder-scatterplot-matrix.png``: the ten cylinder readings
  (temperature, pressure, humidity) that the section calculates
  covariance and correlation by hand for. The r values printed in the
  upper triangle are the same 0.997 and 0.380 quoted in the prose.
- ``flotation-cell-scatterplot-matrix.png`` and
  ``flotation-cell-correlation-heatmap.png``: five process tags on 2922
  samples. Small enough that the numbers still fit inside the heat map
  cells, so the two displays can be read against each other.
- ``distillation-tower-correlation-heatmap.png``: the same heat map for
  27 variables, the size at which printing the numbers stops working
  and the colour has to carry the message on its own. Rows and columns
  are reordered by hierarchical clustering, so that correlated
  variables gather into blocks along the diagonal instead of being
  scattered through the order the file happens to use.
- ``distillation-tower-scatterplot-matrix.png``: five of those 27
  columns, chosen to span the range of correlation with the outcome
  variable ``VapourPressure`` (strong negative through to strong
  positive), because a 27 by 27 scatterplot matrix is unreadable.
- ``unlimited-time-test-scatter.png``: time taken against grade
  achieved on an open-book exam with no time limit, r = -0.044. The
  counter-example for what no correlation looks like.

The diagonal of each scatterplot matrix carries a kernel density curve
over a rug (a short tick per recorded value), the upper triangle the
correlation coefficient sized and coloured by its magnitude, and the
lower triangle the scatter plot itself. This is the same layout used
for the cheddar-cheese and food-texture figures elsewhere in the book,
so the reader meets one display, not three.

Data are fetched from openmv.net; a local CSV of the same name next to
this script is used if the network is unavailable.

Usage
-----
    uv run --with numpy --with pandas --with matplotlib python correlation-visualization-figures.py [output_dir]
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREY = "#666666"

DPI = 300
HERE = pathlib.Path(__file__).parent

# Five of the 27 distillation columns, ordered from the strongest negative
# correlation with VapourPressure through to the strongest positive, so the
# scatterplot matrix shows the whole range rather than one corner of it.
DISTILL_SUBSET = ["Temp9", "Temp7", "OC1", "InvTemp3", "VapourPressure"]

mpl.rcParams.update(
    {
        "font.size": 15,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.axisbelow": True,
    }
)


def fetch(name: str, **kwargs) -> pd.DataFrame:
    """Read ``name``.csv from openmv.net, falling back to a local copy."""
    try:
        return pd.read_csv(f"https://openmv.net/file/{name}.csv", **kwargs)
    except Exception:
        return pd.read_csv(HERE / f"{name}.csv", **kwargs)


def kernel_density(values, cut: float = 3.0, points: int = 512):
    """Gaussian kernel density estimate, bandwidth from the nrd0 rule.

    0.9 * min(s, IQR/1.349) * n**(-1/5), the same rule used for the
    cheddar-cheese and food-texture scatterplot matrices, so the
    diagonals of all three figures are drawn the same way.
    """
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    spread = x.std(ddof=1)
    iqr = float(np.subtract(*np.percentile(x, [75, 25])))
    if iqr > 0:
        spread = min(spread, iqr / 1.349)
    bandwidth = 0.9 * spread * n ** (-0.2)
    grid = np.linspace(x.min() - cut * bandwidth, x.max() + cut * bandwidth, points)
    z = (grid[:, None] - x[None, :]) / bandwidth
    density = np.exp(-0.5 * z**2).sum(axis=1) / (n * bandwidth * np.sqrt(2 * np.pi))
    return grid, density


def diagonal_panel(ax, values, label: str, fontsize: int = 15) -> None:
    """A density curve over a rug, with the variable's name above both.

    The y-axis is scaled so the curve never reaches more than two thirds
    of the panel height, which leaves the top third clear for the name
    whatever shape the distribution takes. The rug is faded in proportion
    to the number of readings, so that a few thousand ticks still show
    where the values pile up instead of filling in as one solid bar.
    """
    x = np.asarray(values, dtype=float)
    grid, density = kernel_density(x)
    ax.plot(grid, density, color=BLUE, linewidth=1.8)
    rug_alpha = float(np.clip(30.0 / max(x.size, 1), 0.05, 1.0))
    ax.vlines(x, 0, 0.08 * density.max(), color=BLUE, linewidth=1.0, alpha=rug_alpha)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, density.max() * 1.5)
    ax.text(0.5, 0.97, label, ha="center", va="top", fontsize=fontsize,
            transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


def scatterplot_matrix(data: pd.DataFrame, outdir: pathlib.Path, name: str,
                       marker_size: float = 22, alpha: float = 1.0,
                       filled: bool = False) -> None:
    """Density-over-rug diagonal, r above it, scatter plots below it."""
    names = list(data.columns)
    n = len(names)
    correlation = data.corr()
    fig, axes = plt.subplots(n, n, figsize=(2.6 * n, 2.4 * n))
    for row in range(n):
        for column in range(n):
            ax = axes[row, column]
            ax.tick_params(labelsize=11)
            if row == column:
                diagonal_panel(ax, data[names[row]], names[row])
                continue
            if row < column:
                value = correlation.iloc[row, column]
                # Two decimals, except where that would round to a perfect
                # +/-1.00 for a pair that is not actually perfectly correlated.
                text = f"{value:+.2f}"
                if text in {"+1.00", "-1.00"} and abs(value) != 1.0:
                    text = f"{value:+.3f}"
                ax.text(0.5, 0.5, text, ha="center", va="center",
                        fontsize=14 + 9 * abs(value),
                        color=BLUE if value > 0 else VERMILLION,
                        transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                for side in ax.spines.values():
                    side.set_visible(False)
                continue
            if filled:
                ax.scatter(data[names[column]], data[names[row]], s=marker_size,
                           color=BLUE, alpha=alpha, linewidth=0)
            else:
                ax.scatter(data[names[column]], data[names[row]], s=marker_size,
                           facecolor="none", edgecolor=BLUE, linewidth=1.1,
                           alpha=alpha)
            ax.spines["top"].set_visible(True)
            ax.spines["right"].set_visible(True)
            if column != 0:
                ax.set_yticklabels([])
            if row != n - 1:
                ax.set_xticklabels([])
            if column == 0:
                ax.set_ylabel(names[row])
            if row == n - 1:
                ax.set_xlabel(names[column])
    fig.tight_layout()
    fig.savefig(outdir / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outdir / name}")


def cluster_order(correlation: pd.DataFrame) -> list[str]:
    """Variable names reordered so that correlated ones sit together.

    Hierarchical clustering on the distance :math:`1 - r`, which is 0 for a
    pair moving exactly together and 2 for a pair moving exactly opposite,
    with average linkage. The variables are then read off in the order the
    dendrogram's leaves fall, which is the ordering the D3 Les Miserables
    co-occurrence matrix offers under "by cluster". ``optimal_ordering``
    flips branches to put the most similar pair either side of each join,
    which the dendrogram is free to do without changing the clustering.
    """
    distance = 1 - correlation.to_numpy()
    np.fill_diagonal(distance, 0.0)
    # squareform wants the condensed upper triangle; the matrix is symmetric
    # up to floating-point noise, which checks=False lets through.
    tree = linkage(squareform(distance, checks=False), method="average",
                   optimal_ordering=True)
    return [correlation.columns[i] for i in leaves_list(tree)]


def correlation_heatmap(data: pd.DataFrame, outdir: pathlib.Path, name: str,
                        annotate: bool = False, size: float = 9.0,
                        label_size: int = 12, cluster: bool = False) -> None:
    """The correlation matrix as a diverging colour map, red high, blue low."""
    correlation = data.corr()
    if cluster:
        order = cluster_order(correlation)
        correlation = correlation.loc[order, order]
    names = list(correlation.columns)
    n = len(names)
    fig, ax = plt.subplots(figsize=(size * 1.15, size))
    image = ax.imshow(correlation.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n), names, rotation=90, fontsize=label_size)
    ax.set_yticks(range(n), names, fontsize=label_size)
    # White gridlines between the cells, so each one reads as a separate patch.
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    if annotate:
        for row in range(n):
            for column in range(n):
                value = correlation.iloc[row, column]
                ax.text(column, row, f"{value:+.2f}", ha="center", va="center",
                        fontsize=label_size,
                        color="white" if abs(value) > 0.6 else "black")
    bar = fig.colorbar(image, ax=ax, shrink=0.6, ticks=[-1, -0.5, 0, 0.5, 1])
    bar.ax.tick_params(labelsize=label_size)
    bar.outline.set_visible(False)
    bar.set_label("r(x, y)", fontsize=label_size + 2)
    fig.tight_layout()
    fig.savefig(outdir / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outdir / name}")


def near_zero_scatter(data: pd.DataFrame, outdir: pathlib.Path, name: str) -> None:
    """Time against grade: the counter-example, with no pattern to see."""
    r = data["Time"].corr(data["Grade"])
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(data["Time"], data["Grade"], s=45, facecolor="none",
               edgecolor=BLUE, linewidth=1.3)
    ax.set_xlabel("Time to finish the test [minutes]")
    ax.set_ylabel("Grade achieved [%]")
    ax.text(0.03, 0.04, f"r = {r:+.3f}", ha="left", va="bottom",
            fontsize=17, color=GREY, transform=ax.transAxes)
    ax.grid(color="#DDDDDD", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(outdir / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outdir / name}")


def main(outdir: pathlib.Path) -> None:
    # The cylinder readings the section works through by hand.
    cylinder = pd.DataFrame(
        {
            "Temperature": [273, 285, 297, 309, 321, 333, 345, 357, 369, 381],
            "Pressure": [1600, 1670, 1730, 1830, 1880, 1920, 2000, 2100, 2170, 2200],
            "Humidity": [42, 48, 45, 49, 41, 46, 48, 48, 45, 49],
        }
    )
    print("cylinder correlations:")
    print(cylinder.corr().round(4).to_string())
    scatterplot_matrix(cylinder, outdir, "gas-cylinder-scatterplot-matrix.png")

    flotation = fetch("flotation-cell", index_col=0)
    print(f"\nflotation cell: {flotation.shape[0]} samples, {flotation.shape[1]} variables")
    print(flotation.corr().round(3).to_string())
    scatterplot_matrix(flotation, outdir, "flotation-cell-scatterplot-matrix.png",
                       marker_size=6, alpha=0.15, filled=True)
    correlation_heatmap(flotation, outdir, "flotation-cell-correlation-heatmap.png",
                        annotate=True, size=7.0, label_size=13)

    distillation = fetch("distillation-tower", index_col=0)
    print(f"\ndistillation tower: {distillation.shape[0]} samples, "
          f"{distillation.shape[1]} variables")
    outcome = distillation.corr()["VapourPressure"]
    print("correlation with VapourPressure, strongest first:")
    print(outcome.drop("VapourPressure").reindex(
        outcome.drop("VapourPressure").abs().sort_values(ascending=False).index
    ).round(3).to_string())
    print("clustered column order: "
          + ", ".join(cluster_order(distillation.corr())))
    correlation_heatmap(distillation, outdir,
                        "distillation-tower-correlation-heatmap.png",
                        annotate=False, size=9.5, label_size=11, cluster=True)
    scatterplot_matrix(distillation[DISTILL_SUBSET], outdir,
                       "distillation-tower-scatterplot-matrix.png",
                       marker_size=16, alpha=0.55, filled=True)

    grades = fetch("unlimited-time-test")
    print(f"\nunlimited time test: {grades.shape[0]} students, "
          f"r = {grades['Time'].corr(grades['Grade']):+.4f}")
    near_zero_scatter(grades, outdir, "unlimited-time-test-scatter.png")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE)
