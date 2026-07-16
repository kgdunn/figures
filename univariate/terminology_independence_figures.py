"""Figures for the terminology and independence sections of the univariate chapter.

Two committed PNGs, replacing images previously generated with older
tooling (``batch-yields.R``, and an early matplotlib version of
``simulate-independence.py``):

- ``batch-yields.png``: about five years of daily batch viscosity
  values (1825 batches), the example the "Sample" terminology entry
  uses for a sample so large it stands in for the population. The
  series is a seeded AR(1) simulation with the same character as the
  original figure: mean near 85 cP, values spanning roughly 70 to 95 cP.
- ``simulate-independence.png``: three sequences of 100 batch impurity
  values; sequence 1 is positively autocorrelated, sequence 2 is
  independent, and sequence 3 is negatively autocorrelated. The data
  generation (seed 13, phi = 0.8, 0, -0.5 and the same display scaling)
  is unchanged from the previous version of this script, so the reader
  sees the same series; only the styling is new.

Usage
-----
    uv run --with numpy --with matplotlib python terminology_independence_figures.py [output_dir]

Writes the two PNGs into ``output_dir`` (default: this script's own
directory), refreshing the committed images in place.
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

BLUE = "#0072B2"
GRID = "#DDDDDD"

mpl.rcParams.update(
    {
        "font.size": 12,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,
    }
)


def batch_yields(outdir: pathlib.Path) -> None:
    rng = np.random.default_rng(42)
    n = 1825  # about one batch per day, for 5 years
    phi = 0.7
    x = np.zeros(n)
    shocks = rng.normal(scale=2.3, size=n)
    for k in range(1, n):
        x[k] = phi * x[k - 1] + shocks[k]
    viscosity = 84.5 + x

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.plot(viscosity, color=BLUE, linewidth=0.6)
    ax.set_title("Batch viscosity for the past 5 years")
    ax.set_xlabel("Batch number (about one batch per day)")
    ax.set_ylabel("Motor oil viscosity [cP]")
    ax.set_xlim(0, n)
    ax.set_ylim(70, 95)
    fig.tight_layout()
    fig.savefig(outdir / "batch-yields.png", dpi=200)
    plt.close(fig)
    print(f"wrote {outdir / 'batch-yields.png'}")


def simulate_independence(outdir: pathlib.Path) -> None:
    N = 100
    targetY = 85
    stdevY = 10
    np.random.seed(13)  # kept from the previous version of this figure
    shocks = np.random.normal(loc=targetY, scale=stdevY, size=N)

    def walk(phi: float) -> np.ndarray:
        series = np.zeros(N)
        series[0] = targetY
        for k in range(1, N):
            series[k] = phi * series[k - 1] + (1 - phi) * shocks[k]
        return series

    # The display scaling (6x, 2x, 1x about the target) is unchanged
    # from the previous version, so the three sequences look the same.
    sequences = [
        ((walk(0.8) - targetY) * 6.0 + targetY, "Sequence 1"),
        ((walk(0.0) - targetY) * 2.0 + targetY, "Sequence 2"),
        (walk(-0.5), "Sequence 3"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(7.5, 3.75), sharex=True)
    for ax, (series, label) in zip(axes, sequences):
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.plot(series, ".-", color=BLUE, linewidth=0.8, markersize=3.5)
        ax.set_xlim(0, N)
        ax.set_ylim(40, 140)
        ax.set_yticks([50, 100])
        ax.set_ylabel(label, fontsize=9)
        ax.tick_params(labelsize=8)
    axes[-1].set_xlabel("Batch number", fontsize=9)
    fig.tight_layout()
    fig.savefig(outdir / "simulate-independence.png", dpi=200)
    plt.close(fig)
    print(f"wrote {outdir / 'simulate-independence.png'}")


if __name__ == "__main__":
    outdir = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).parent
    batch_yields(outdir)
    simulate_independence(outdir)
