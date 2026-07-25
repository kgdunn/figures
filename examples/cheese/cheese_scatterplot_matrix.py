"""Scatterplot matrix for the cheddar-cheese exercise.

Writes three committed PNGs, replacing the base-R output of
``cheese-plots.R``:

- ``cheese-plots.png`` and ``cheese-plots-with-random.png``: acetic acid,
  H2S, lactic acid, taste, and a fifth column of random numbers.
- ``cheese-plots-no-random.png``: the same four measured variables without
  that fifth column.

The random column is deliberate, and it is the reason the figure earns its
place. ``cheese-plots.R`` creates it on its line 5, ``cheese$Random <-
rnorm(N, 1)``, and plots columns 2 to 6 for ``cheese-plots.png`` and
columns 2 to 5 for ``cheese-plots-no-random.png``. Having a variable that
is known to be unrelated to the rest gives the reader a reference for what
"no relationship" looks like in this display, against which the real
correlations can be read. The same column carries through the chapter into
the regression and the neural-net models further down that script, where
the point is that a variable of pure noise should earn no coefficient.

Its values are not regenerated here. The R script writes them out to
``cheese-with-random-data.csv`` in this directory, with the comment "to
double check calculations in other software packages", so the recorded
values are read from there and the figure reproduces the original exactly
rather than approximately.

The diagonal carries each variable's histogram and the upper triangle its
correlation with the other variable, which is the same layout used for the
food-texture example earlier in the chapter.

Usage
-----
    uv run --with numpy --with pandas --with matplotlib python cheese_scatterplot_matrix.py [output_dir]
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
VERMILLION = "#D55E00"
GRID = "#DDDDDD"

DPI = 300
HERE = pathlib.Path(__file__).parent
# The random column exists only in the committed copy: the dataset on
# openmv.net carries the four measured variables alone.
RECORDED = "cheese-with-random-data.csv"
MEASURED = ["Acetic", "H2S", "Lactic", "Taste"]
WITH_RANDOM = [*MEASURED, "Random"]

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


def cheese() -> pd.DataFrame:
    """The five columns, random one included, as the R script recorded them."""
    return pd.read_csv(HERE / RECORDED, index_col=0)[WITH_RANDOM]


def scatterplot_matrix(data: pd.DataFrame, outdir: pathlib.Path, name: str) -> None:
    names = list(data.columns)
    n = len(names)
    correlation = data.corr()
    fig, axes = plt.subplots(n, n, figsize=(2.6 * n, 2.4 * n))
    for row in range(n):
        for column in range(n):
            ax = axes[row, column]
            ax.tick_params(labelsize=11)
            if row == column:
                counts, _, _ = ax.hist(data[names[row]], bins=10, color=BLUE_FILL,
                                       edgecolor=BLUE, linewidth=1.0)
                ax.set_ylim(0, counts.max() * 1.45)
                ax.text(0.5, 0.97, names[row], ha="center", va="top", fontsize=15,
                        transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            if row < column:
                value = correlation.iloc[row, column]
                ax.text(0.5, 0.5, f"{value:+.2f}", ha="center", va="center",
                        fontsize=14 + 9 * abs(value),
                        color=BLUE if value > 0 else VERMILLION,
                        transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                for side in ax.spines.values():
                    side.set_visible(False)
                continue
            ax.scatter(data[names[column]], data[names[row]], s=22,
                       facecolor="none", edgecolor=BLUE, linewidth=1.1)
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


def main(outdir: pathlib.Path) -> None:
    data = cheese()
    print(f"{len(data)} cheeses, {data.shape[1]} columns")
    print(data.corr().round(3).to_string())
    print("correlation of the random column with taste: "
          f"{data['Random'].corr(data['Taste']):+.3f}")

    scatterplot_matrix(data, outdir, "cheese-plots.png")
    scatterplot_matrix(data, outdir, "cheese-plots-with-random.png")
    scatterplot_matrix(data[MEASURED], outdir, "cheese-plots-no-random.png")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE)
