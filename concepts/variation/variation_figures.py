"""The four variability illustrations for the univariate-review chapter.

Four committed PNGs, shown in sequence in
``univariate-review/what-is-variability.rst`` as the section works
through the sources of variation in a process:

- ``variation-none.png``: a flat line at the 1680 mil target.
- ``variation-some.png``: the board-to-board thickness measurements.
- ``variation-more.png``: those measurements passed through a
  first-order filter, so the series wanders rather than jumps.
- ``variation-spikes.png``: the same filtered series, plus spikes and a
  stretch of missing data.

Replaces ``variability-illustration.py``, which loaded a ``.mat`` file
from a path on the author's laptop and so could not be run by anyone
else. The same quantity is now read from the public board-thickness
dataset (https://openmv.net/info/six-point-board-thickness): the mean
of the six thickness measurements on each board, first 500 boards. The
two datasets hold the same numbers, which is confirmed by the check in
:func:`recorded_spike_walk` below.

The series themselves are the originals, drawn the same way:

- ``variation-more.png`` applies the original first-order filter,
  :math:`w_k = \\varphi w_{k-1} + (1 - \\varphi) \\bar{x}_k` with
  :math:`\\varphi = 0.9`, started at the target. The filter is what
  gives the long, slowly recovering excursions.
- ``variation-spikes.png`` is read from ``spike_walk-numpy-python.csv``,
  written by the original script and committed beside it. Spikes at 8
  of the 10 drawn positions survive in the record (147, 244, 249, 261,
  313, 379, 446, 468 boards, from +60 to -69 mils); the other two fall
  inside the missing stretch, boards 175 to 224. Each spike is followed
  by the filter recovering towards the mean over the next tens of
  boards, which is the shape the panel is there to show, so the
  recorded realization is used rather than a fresh random one.

What differs from the original images: colour rather than black, a
grid, axis labels, and the target drawn as a dashed line on the panels
where the data move away from it. The y-axis stays autoscaled per
panel, as in the originals, since the filtered series covers a much
narrower range than the raw measurements.

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
PHI = 0.9

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


def filtered(thickness: np.ndarray) -> np.ndarray:
    """The original first-order filter, started at the target."""
    walk = np.zeros(N)
    walk[0] = TARGET
    for k in range(1, N):
        walk[k] = PHI * walk[k - 1] + (1 - PHI) * thickness[k]
    return walk


def recorded_spike_walk(thickness: np.ndarray) -> np.ndarray:
    """The spiked series as the original script wrote it out.

    Also checks that the record and the public dataset agree: up to the
    first spike, the recorded series must satisfy the same filter
    recursion as the data fetched here. A mismatch means the two
    datasets are not the same numbers, and the panel would then be
    telling a different story from the rest of the chapter.
    """
    walk = np.loadtxt(HERE / "spike_walk-numpy-python.csv")
    predicted = PHI * walk[:146] + (1 - PHI) * thickness[1:147]
    largest = np.max(np.abs(walk[1:147] - predicted))
    if largest > 1e-6:
        raise ValueError(
            f"recorded series and board data disagree by up to {largest:.3f} mils"
        )
    return walk


def strip(
    values: np.ndarray,
    title: str,
    outdir: pathlib.Path,
    name: str,
    ylim: tuple[float, float],
    show_target: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.grid(color=GRID, linewidth=0.8)
    # On the first panel the data sit exactly on the target, so drawing
    # the target as well would just make the line look dashed.
    if show_target:
        ax.axhline(TARGET, color=GREY, linewidth=1.2, linestyle="--")
    ax.plot(np.arange(N), values, color=BLUE, linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Board number")
    ax.set_ylabel("Thickness [mils]")
    ax.set_xlim(0, N)
    ax.set_ylim(*ylim)
    # The panels cover very different ranges, and the default choice
    # leaves the narrowest of them with two labelled gridlines.
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=6, steps=[1, 2, 2.5, 5, 10]))
    fig.tight_layout()
    fig.savefig(outdir / name, dpi=DPI)
    plt.close(fig)
    print(f"wrote {outdir / name}")


def main(outdir: pathlib.Path) -> None:
    thickness = average_thickness()
    walk = filtered(thickness)
    spiky = recorded_spike_walk(thickness)

    missing = np.where(~np.isfinite(spiky))[0]
    print(
        f"thickness {thickness.min():.0f} to {thickness.max():.0f} mils; "
        f"filtered {walk.min():.0f} to {walk.max():.0f} mils; "
        f"spiked {np.nanmin(spiky):.0f} to {np.nanmax(spiky):.0f} mils; "
        f"missing boards {missing.min()} to {missing.max()}"
    )

    strip(
        np.full(N, TARGET),
        "No variability",
        outdir,
        "variation-none.png",
        ylim=(1580, 1780),
        show_target=False,
    )
    strip(thickness, "Some variation", outdir, "variation-some.png", ylim=(1590, 1745))
    strip(walk, "A bit more variation", outdir, "variation-more.png", ylim=(1653, 1697))
    strip(
        spiky,
        "More variation, spikes and other real phenomena",
        outdir,
        "variation-spikes.png",
        ylim=(1595, 1745),
    )


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE)
