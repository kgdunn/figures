"""Linking and brushing illustration for the latent variable chapter.

Writes ``brushing-illustration.png``: a scatterplot matrix of the four iris
measurements, with the three species marked by different symbols, showing
what linking looks like across every panel at once.

This replaces ``brushing-illustration.R``, which used lattice's ``splom``.
Two things about that script:

- it wrote ``brushing-illustration-colour.png``, while the chapter embeds
  ``brushing-illustration.png``, so the file the book uses had no script
  that produced it;
- the chapter source carries a note asking for the colour version to have
  a white background, which is what the lattice theme was fighting.

Both are settled here: the figure is written under the name the chapter
uses, on white, and each species differs in symbol *and* colour, so the
linking is legible in greyscale and in colour alike.

The data are Fisher's iris measurements, committed as ``iris.csv`` beside
this script so the figure does not depend on which statistical package
happens to be installed.

Usage
-----
    uv run --with pandas --with matplotlib python brushing_illustration.py [output_dir]
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
GRID = "#DDDDDD"

DPI = 300
HERE = pathlib.Path(__file__).parent
MEASUREMENTS = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"]
STYLE = {
    "Setosa": ("o", BLUE),
    "Versicolor": ("^", ORANGE),
    "Virginica": ("P", GREEN),
}

mpl.rcParams.update(
    {
        "font.size": 15,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 15,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.axisbelow": True,
    }
)


def iris() -> pd.DataFrame:
    return pd.read_csv(HERE / "iris.csv")


def main(outdir: pathlib.Path) -> None:
    data = iris()
    counts = data["Species"].value_counts()
    print(f"{len(data)} observations: "
          + ", ".join(f"{name} {counts[name]}" for name in STYLE))

    n = len(MEASUREMENTS)
    fig, axes = plt.subplots(n, n, figsize=(11, 11))
    for row in range(n):
        for column in range(n):
            ax = axes[row, column]
            ax.grid(color=GRID, linewidth=0.7)
            if row == column:
                ax.text(0.5, 0.5, MEASUREMENTS[row].replace(".", "\n"),
                        ha="center", va="center", fontsize=16, transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                for side in ax.spines.values():
                    side.set_visible(False)
                ax.grid(False)
                continue
            for species, (marker, colour) in STYLE.items():
                subset = data[data["Species"] == species]
                ax.plot(subset[MEASUREMENTS[column]], subset[MEASUREMENTS[row]],
                        marker, markersize=5, markerfacecolor="none",
                        markeredgecolor=colour, markeredgewidth=1.1, linestyle="none")
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(length=2)
            if column == 0:
                ax.set_ylabel(MEASUREMENTS[row])
            if row == n - 1:
                ax.set_xlabel(MEASUREMENTS[column])

    handles = [
        mpl.lines.Line2D([], [], marker=marker, color=colour, linestyle="none",
                         markerfacecolor="none", markeredgewidth=1.4, markersize=9,
                         label=species)
        for species, (marker, colour) in STYLE.items()
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.99), title="Three varieties of iris",
               title_fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(outdir / "brushing-illustration.png", dpi=DPI, facecolor="white",
                bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outdir / 'brushing-illustration.png'}")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE)
