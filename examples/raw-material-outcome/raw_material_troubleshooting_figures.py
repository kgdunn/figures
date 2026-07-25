"""Raw-material troubleshooting figures for the product-improvement chapter.

Writes ``process-troubleshooting.png``: the score plot and loading plot of a
PCA on six characteristic measurements taken on each lot of incoming raw
material, with the lots that gave a poor yield marked.

This replaces ``process-troubleshooting.R``, which read the dataset from
``stats4.eng.mcmaster.ca``, a host that no longer resolves. The same data
are published as ``raw-material-characterization.csv`` on openmv.net and
are read from there, with a committed copy as a fallback.

The model is a PCA by singular value decomposition on the autoscaled data,
which is what ``prcomp(X, scale=TRUE)`` did. Loading signs are anchored so
that the size measurements load negatively on the first component and
positively on the second, matching the orientation the chapter describes:
lots with a poor yield sit at low t1 and high t2.

Beyond redrawing, the figure now:

- marks each poor-yield lot with its number, so the batches the text works
  through (8 and 22) can be found;
- states the fraction of variance each component explains on its axes;
- uses a colourblind-safe pair of colours and distinct marker shapes, so
  the two groups are still distinguishable in greyscale.

The contribution terms quoted in the chapter for batches 8 and 22 are
printed when this script runs, so they can be checked against the text.

Usage
-----
    uv run --with numpy --with pandas --with matplotlib python raw_material_troubleshooting_figures.py [output_dir]
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
GREEN = "#009E73"
VERMILLION = "#D55E00"
GREY = "#666666"
GRID = "#DDDDDD"

DPI = 300
HERE = pathlib.Path(__file__).parent
URL = "https://openmv.net/file/raw-material-characterization.csv"
MEASUREMENTS = ["Size5", "Size10", "Size15", "TGA", "DSC", "TMA"]

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


def raw_materials() -> pd.DataFrame:
    try:
        data = pd.read_csv(URL)
    except Exception:  # noqa: BLE001 - the committed copy is the fallback
        data = pd.read_csv(HERE / "raw-material-characterization.csv")
    return data


def principal_components(scaled: np.ndarray):
    left, singular_values, right = np.linalg.svd(scaled, full_matrices=False)
    loadings = right.T
    scores = left * singular_values
    # Anchor the orientation the chapter describes: the size measurements
    # load negatively on the first component and positively on the second.
    for component, wanted in ((0, -1), (1, +1)):
        if np.sign(loadings[0, component]) != wanted:
            loadings[:, component] *= -1
            scores[:, component] *= -1
    explained = singular_values**2 / np.sum(singular_values**2)
    return scores, loadings, explained


def main(outdir: pathlib.Path) -> None:
    data = raw_materials()
    measurements = data[MEASUREMENTS]
    scaled = ((measurements - measurements.mean()) / measurements.std(ddof=1)).to_numpy()
    scores, loadings, explained = principal_components(scaled)
    poor = (data["Outcome"] == "Poor").to_numpy()
    numbers = np.arange(1, len(data) + 1)

    print(f"{len(data)} lots: {np.sum(~poor)} adequate, {poor.sum()} poor")
    print(f"variance explained: {100 * explained[0]:.1f}% and {100 * explained[1]:.1f}%")
    for component in (0, 1):
        print(f"p{component + 1} = "
              + ", ".join(f"{n} {v:+.3f}" for n, v in zip(MEASUREMENTS, loadings[:, component])))
    for lot in (8, 22):
        for component in (0, 1):
            terms = scaled[lot - 1] * loadings[:, component]
            print(f"t[{lot}, a={component + 1}] = "
                  + ", ".join(f"{n} {v:+.2f}" for n, v in zip(MEASUREMENTS, terms))
                  + f"; total {terms.sum():+.2f}")

    first, second = (100 * explained[:2]).round().astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))

    ax = axes[0]
    ax.grid(color=GRID, linewidth=0.8)
    ax.axhline(0, color="black", linewidth=1.0)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.plot(scores[~poor, 0], scores[~poor, 1], "^", markersize=11, markerfacecolor="none",
            markeredgecolor=GREEN, markeredgewidth=2, label="Adequate yield")
    ax.plot(scores[poor, 0], scores[poor, 1], "o", markersize=11, markerfacecolor="none",
            markeredgecolor=VERMILLION, markeredgewidth=2, label="Poor yield")
    for number, (x, y) in zip(numbers[poor], scores[poor, :2]):
        ax.annotate(str(number), (x, y), textcoords="offset points", xytext=(10, 6),
                    color=VERMILLION, fontsize=14)
    ax.set_xlabel(f"$t_1$ [{first}%]")
    ax.set_ylabel(f"$t_2$ [{second}%]")
    ax.set_title("Score plot")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    ax.grid(color=GRID, linewidth=0.8)
    ax.axhline(0, color="black", linewidth=1.0)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.plot(loadings[:, 0], loadings[:, 1], "o", markersize=11, color=BLUE)
    # TGA and TMA land on nearly the same point, so their labels are nudged
    # apart rather than printed on top of each other.
    offsets = {"TGA": (-14, 12), "TMA": (12, -20), "Size15": (12, -6)}
    for name, (x, y) in zip(MEASUREMENTS, loadings[:, :2]):
        ax.annotate(name, (x, y), textcoords="offset points",
                    xytext=offsets.get(name, (11, 5)), fontsize=16)
    span = 1.35 * np.abs(loadings[:, :2]).max()
    ax.set_xlim(-span, span)
    ax.set_ylim(-span, span)
    ax.set_xlabel(f"$p_1$ [{first}%]")
    ax.set_ylabel(f"$p_2$ [{second}%]")
    ax.set_title("Loadings plot")

    fig.tight_layout()
    fig.savefig(outdir / "process-troubleshooting.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outdir / 'process-troubleshooting.png'}")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE)
