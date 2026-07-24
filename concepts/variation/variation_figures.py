"""The four variability illustrations for the univariate-review chapter.

Four committed PNGs, shown in sequence in
``univariate-review/what-is-variability.rst`` as the section works
through the sources of variation in a process:

- ``variation-none.png``: a flat line at the 1680 mil target.
- ``variation-some.png``: the target plus measurement noise.
- ``variation-more.png``: the same noise with a slow drift added, of
  the kind a sensor produces between recalibrations.
- ``variation-spikes.png``: the same again, plus ten spikes and a
  stretch of missing data.

Replaces ``variability-illustration.py``, which loaded a ``.mat`` file
from a path on the author's laptop and so could not be run by anyone
else. The same quantity is now read from the public board-thickness
dataset (https://openmv.net/info/six-point-board-thickness): the mean
of the six thickness measurements on each board, first 500 boards.

Two changes from the original images, both to make the four read as a
sequence:

- All four now share one y-axis range, and both axes are labelled. The
  originals were each autoscaled, so the third panel was drawn on a
  40 mil range and the second on a 140 mil range, which made "a bit
  more variation" look calmer than "some variation".
- The third and fourth panels add the drift on top of the noise. The
  original passed the noise through a first-order filter instead, which
  removed the noise rather than adding to it.

The spike positions and sizes are drawn from a seeded generator, so the
committed images regenerate exactly.

Usage
-----
    uv run --with numpy --with pandas --with matplotlib python variation_figures.py [output_dir]

Writes the four PNGs into ``output_dir`` (default: this script's own
directory), refreshing the committed images in place.
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
GREY = "#666666"
GRID = "#DDDDDD"

DPI = 250
HERE = pathlib.Path(__file__).parent

TARGET = 1680.0
N = 500
YLIM = (1520, 1840)

# These strips are wide and short, so the labels are set a little
# smaller than in the taller chapter figures.
mpl.rcParams.update(
    {
        "font.size": 14,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 17,
        "axes.labelsize": 15,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "axes.axisbelow": True,
    }
)


def fetch(name: str) -> pd.DataFrame:
    try:
        return pd.read_csv(f"https://openmv.net/file/{name}")
    except Exception:
        return pd.read_csv(HERE / name)


def average_thickness() -> np.ndarray:
    boards = fetch("six-point-board-thickness.csv")
    positions = [f"Pos{i}" for i in range(1, 7)]
    return boards[positions].mean(axis=1).to_numpy(dtype=float)[:N]


def strip(values, title: str, outdir: pathlib.Path, name: str,
          show_target: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.grid(color=GRID, linewidth=0.8)
    # On the first panel the data sit exactly on the target, so drawing
    # the target as well would just make the line look dashed.
    if show_target:
        ax.axhline(TARGET, color=GREY, linewidth=1.2, linestyle="--")
    ax.plot(np.arange(1, N + 1), values, color=BLUE, linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Board number")
    ax.set_ylabel("Thickness [mils]")
    ax.set_xlim(0, N)
    ax.set_ylim(*YLIM)
    fig.tight_layout()
    fig.savefig(outdir / name, dpi=DPI)
    plt.close(fig)
    print(f"wrote {outdir / name}")


def main(outdir: pathlib.Path) -> None:
    rng = np.random.default_rng(7)
    noise = average_thickness()

    # A slow drift, built as a first-order random walk that is pulled
    # back towards zero, so it wanders without running away. The pull is
    # weak enough that the excursions last for tens of boards, which is
    # what makes them read as drift rather than as more noise.
    phi = 0.995
    drift = np.zeros(N)
    for k in range(1, N):
        drift[k] = phi * drift[k - 1] + rng.normal(scale=2.2)
    drifting = noise + drift

    spiky = drifting.copy()
    # Ten spikes, each large enough to stand clear of the noise band,
    # decaying over the next reading, as a jammed or newly recalibrated
    # sensor would give. Most read low, as a dropout does.
    for position in rng.choice(np.arange(10, N - 2), size=10, replace=False):
        direction = 1.0 if rng.random() < 0.3 else -1.0
        delta = direction * (0.035 + 0.015 * rng.random()) * TARGET
        spiky[position] += delta
        spiky[position + 1] += 0.2 * delta
    # A stretch where the sensor recorded nothing at all.
    gap = int(rng.integers(low=150, high=N - 100))
    spiky[gap:gap + 50] = np.nan
    print(
        f"drift range {drift.min():.0f} to {drift.max():.0f} mils; "
        f"missing data from board {gap} to {gap + 50}"
    )

    strip(np.full(N, TARGET), "No variability", outdir, "variation-none.png",
          show_target=False)
    strip(noise, "Some variation", outdir, "variation-some.png")
    strip(drifting, "A bit more variation", outdir, "variation-more.png")
    strip(
        spiky,
        "More variation, spikes and other real phenomena",
        outdir,
        "variation-spikes.png",
    )


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE)
