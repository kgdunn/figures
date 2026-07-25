"""Coefficient plot for the LDPE PLS model, used in the PLS chapter.

Writes ``coefficient-plot-LDPE-A-is-6.png``: the regression coefficients
relating the fourteen X-variables to conversion, from a PLS model with six
components on the LDPE data.

This replaces ``coefficient-plot-LDPE.R``, which drew the same bars in base
R. The coefficients themselves are the output of a PLS fit reported in
``coefficient-plot-LDPE-A-is-6.csv``, committed alongside; they are read
from there rather than refitted, so the figure keeps matching the model
the surrounding text describes (A = 6, K = 14, M = 5).

Beyond redrawing, the figure now:

- sorts the variables by the size of their coefficient rather than by
  whatever order the file happened to be in, so the ones that matter are
  together at one end;
- colours each bar by sign, since the sign is what the reader is being
  asked to interpret;
- draws a zero line, which the original left to the eye.

Usage
-----
    uv run --with pandas --with matplotlib python ldpe_coefficient_plot.py [output_dir]
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

BLUE = "#0072B2"
BLUE_FILL = "#B3D4EA"
ORANGE = "#E69F00"
ORANGE_FILL = "#F5D28B"
GRID = "#DDDDDD"

DPI = 300
HERE = pathlib.Path(__file__).parent

mpl.rcParams.update(
    {
        "font.size": 16,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
        "axes.axisbelow": True,
    }
)


def coefficients() -> pd.DataFrame:
    data = pd.read_csv(HERE / "coefficient-plot-LDPE-A-is-6.csv")
    return data.sort_values("Value").reset_index(drop=True)


def main(outdir: pathlib.Path) -> None:
    data = coefficients()
    print(f"{len(data)} X-variables")
    for name, value in zip(data["Name"], data["Value"]):
        print(f"  {name:8s} {value:+.4f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    fills = [ORANGE_FILL if v > 0 else BLUE_FILL for v in data["Value"]]
    edges = [ORANGE if v > 0 else BLUE for v in data["Value"]]
    ax.barh(data["Name"], data["Value"], height=0.7, color=fills, edgecolor=edges,
            linewidth=1.5)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_xlabel("Coefficients related to $y$ = Conversion")
    ax.set_ylabel("X-variables")
    handles = [
        mpl.patches.Patch(facecolor=ORANGE_FILL, edgecolor=ORANGE,
                          label="Raises conversion"),
        mpl.patches.Patch(facecolor=BLUE_FILL, edgecolor=BLUE,
                          label="Lowers conversion"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False)
    fig.savefig(outdir / "coefficient-plot-LDPE-A-is-6.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outdir / 'coefficient-plot-LDPE-A-is-6.png'}")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE)
