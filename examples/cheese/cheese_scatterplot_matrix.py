"""Scatterplot matrix for the cheddar-cheese exercise.

Writes two committed PNGs, replacing the base-R output of
``cheese-plots.R``:

- ``cheese-plots.png``: the four measured variables (acetic acid, H2S,
  lactic acid and taste) plotted against each other, which is what the
  exercise's own code produces.
- ``cheese-plots-with-random.png``: the same four with a column of random
  numbers added, for showing what a variable unrelated to the others looks
  like in the same display.

The exercise prints ``scatterplotMatrix(cheese[, 2:5])``, four columns, but
the committed ``cheese-plots.png`` showed five, the fifth being a "Random"
variable that the exercise never introduces and its code never creates. A
reader running the printed code got a different figure from the one beside
it. The four-column version now carries that name; the five-column version
keeps its own.

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
URL = "https://openmv.net/file/cheddar-cheese.csv"
MEASURED = ["Acetic", "H2S", "Lactic", "Taste"]
SEED = 2

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
    try:
        data = pd.read_csv(URL)
    except Exception:  # noqa: BLE001
        data = pd.read_csv(HERE / "cheese-with-random-data.csv", index_col=0)
    return data[MEASURED]


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
    print(f"{len(data)} cheeses, {data.shape[1]} measured variables")
    print(data.corr().round(3).to_string())

    with_random = data.copy()
    with_random["Random"] = np.random.RandomState(SEED).normal(loc=1.0, size=len(data))
    print("correlation of the random column with taste: "
          f"{with_random['Random'].corr(with_random['Taste']):+.3f}")

    scatterplot_matrix(data, outdir, "cheese-plots.png")
    scatterplot_matrix(with_random, outdir, "cheese-plots-with-random.png")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE)
