"""Generate the committed PNG showing how well the null-space direction is determined.

Fourth figure in the ``pls/`` set for the pid-book section "How well is that
direction determined?" in
``latent-variable-modelling/projection-to-latent-structures/pls-model-inversion-and-the-orthogonal-space.rst``.

The null-space direction rests on the second y-loading, which is the component
cross-validation did not keep. Refitting the model on bootstrap resamples of
the 26 calibration cheeses shows how much of that direction survives.

Left panel: one faint line per refit, each the null space that refit would have
returned. Overplotting turns the spread into a density, so the fan is the
uncertainty itself rather than a number describing it. The fan pinches near the
direct-inversion solution and spreads from there, which is the asymmetry the
chapter draws out: the refits agree on the point and disagree on the direction.

Right panel: the same spread as an angle from the direction the full
calibration set gives. Compared without sign, since a direction and its
negative describe the same line.

Requires the ``process_improve`` package (``pip install process-improve``) for
``PLS``. Runs roughly a minute for the default 1200 refits.

Usage::

    python pls/pls-null-space-bootstrap.py [output_dir]

``output_dir`` defaults to this script's own directory (``pls/``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from process_improve.multivariate import PLS

DATA_URL = "https://openmv.net/file/cheddar-cheese.csv"
X_COLUMNS = ["Acetic", "H2S", "Lactic"]
TARGET_TASTE = 20.9
N_BOOTSTRAP = 1200
SEED = 0

DARK_BLUE = "#1f3d7a"    # calibration cheeses
ORANGE = "#e6820a"       # the null space, faint for refits and solid for the point estimate
DEEP_ORANGE = "#a35a05"  # the reported direction, darker so it reads against the bars
MAROON = "#7b1d2b"       # the median of the resampled angles
GREY = "0.55"
BLACK = "#111111"

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 140,
    }
)


def build_figure(out_dir: Path) -> None:
    """Fit the model, refit it on resamples, and draw the spread two ways."""
    cheese = pd.read_csv(DATA_URL)
    train = cheese.iloc[4:].reset_index(drop=True)  # cheeses 5 to 30, as in the chapter
    pls = PLS(n_components=2).fit(train[X_COLUMNS], train[["Taste"]])

    result = pls.invert(TARGET_TASTE)
    tau_0 = result.scores.to_numpy()
    g_0 = result.null_space_basis.to_numpy().ravel()
    scores = pls.scores_.to_numpy()

    # The reported direction, mapped into the inputs so refits can be compared there.
    reference = g_0 @ pls.x_loadings_.to_numpy().T
    reference = reference / np.linalg.norm(reference)

    rng = np.random.default_rng(SEED)
    refits, angles = [], []
    for _ in range(N_BOOTSTRAP):
        sample = train.iloc[rng.integers(0, len(train), len(train))]
        boot = PLS(n_components=2).fit(sample[X_COLUMNS], sample[["Taste"]])
        boot_result = boot.invert(TARGET_TASTE)
        refits.append((boot_result.scores.to_numpy(), boot_result.null_space_basis.to_numpy().ravel()))
        direction = boot_result.null_space_basis.to_numpy().ravel() @ boot.x_loadings_.to_numpy().T
        direction = direction / np.linalg.norm(direction)
        # A direction and its negative are the same line, so compare without sign.
        angles.append(np.degrees(np.arccos(np.clip(abs(direction @ reference), 0, 1))))
    angles = np.array(angles)

    fig, (ax_fan, ax_hist) = plt.subplots(
        1, 2, figsize=(12.4, 5.9), gridspec_kw={"width_ratios": [1.15, 1.0]}
    )

    # --- left: one line per refit -------------------------------------------
    span = np.linspace(-9, 9, 2)  # long enough to cross the axes at any slope
    for tau_b, g_b in refits:
        segment = np.array([tau_b + s * g_b for s in span])
        ax_fan.plot(segment[:, 0], segment[:, 1], color=ORANGE, lw=0.7, alpha=0.030,
                    zorder=2, solid_capstyle="butt")

    segment_0 = np.array([tau_0 + s * g_0 for s in span])
    ax_fan.plot(segment_0[:, 0], segment_0[:, 1], color=ORANGE, lw=2.6, zorder=5,
                label="From all 26 cheeses")
    ax_fan.scatter(scores[:, 0], scores[:, 1], s=16, color=DARK_BLUE, alpha=0.55, lw=0,
                   zorder=4, label="Calibration cheeses")
    ax_fan.scatter(*tau_0, color=ORANGE, marker="s", s=95, edgecolors=BLACK, linewidths=0.9,
                   zorder=6, label="Direct-inversion solution")
    ax_fan.plot([], [], color=ORANGE, lw=2, alpha=0.55, label=f"{len(refits)} bootstrap refits")

    ax_fan.axhline(0, color="0.8", lw=0.8, zorder=0)
    ax_fan.axvline(0, color="0.8", lw=0.8, zorder=0)
    ax_fan.set_xlim(-4.0, 4.0)
    ax_fan.set_ylim(-4.0, 4.0)
    ax_fan.set_aspect("equal")
    ax_fan.set_xlabel("$t_1$")
    ax_fan.set_ylabel("$t_2$")
    ax_fan.set_title("Each refit gives its own null space", fontsize=11.5)
    ax_fan.legend(loc="upper left", fontsize=8.5, framealpha=0.93)

    # --- right: the same spread as an angle ---------------------------------
    ax_hist.hist(angles, bins=np.arange(0, 91, 3), color=ORANGE, alpha=0.85, ec="white", lw=0.6)
    median = float(np.median(angles))
    ax_hist.axvline(median, color=MAROON, lw=2.0, zorder=5)
    ax_hist.annotate(f"median {median:.0f}°", xy=(median, ax_hist.get_ylim()[1] * 0.93),
                     xytext=(10, 0), textcoords="offset points", color=MAROON,
                     fontsize=10, fontweight="bold")
    ax_hist.axvline(0, color=DEEP_ORANGE, lw=3.2, zorder=5)
    ax_hist.text(1.6, ax_hist.get_ylim()[1] * 0.50, "the direction reported in the text",
                 rotation=90, va="center", ha="left", color=BLACK, fontsize=9, zorder=6)

    beyond = float((angles > 45).mean()) * 100
    ax_hist.axvspan(45, 90, color=GREY, alpha=0.16, zorder=0)
    ax_hist.annotate(f"{beyond:.0f}% of refits land\nmore than 45° away", xy=(0.76, 0.62),
                     xycoords="axes fraction", ha="center", fontsize=9, color="0.35")
    ax_hist.set_xlim(0, 90)
    ax_hist.set_xlabel("Angle from the reported null-space direction (degrees)")
    ax_hist.set_ylabel("Number of bootstrap refits")
    ax_hist.set_title("The same spread, measured as an angle", fontsize=11.5)

    fig.tight_layout()
    fig.savefig(out_dir / "pls-null-space-bootstrap.png")
    plt.close(fig)


def main() -> None:
    """Entry point: render into the given directory (default: this script's own)."""
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
    build_figure(out_dir)


if __name__ == "__main__":
    main()
