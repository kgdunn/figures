"""Generate the committed PNG contrasting the two distance measures along the null space.

Third companion to ``pls/pls-model-inversion-null-space.py`` and
``pls/pls-null-space-geometry.py``, for the pid-book section "Where the
null-space line comes from" in
``latent-variable-modelling/projection-to-latent-structures/pls-model-inversion-and-the-orthogonal-space.rst``.

Walking along the null space from the direct-inversion solution leaves the
predicted taste unchanged, but not the distance from the centre of the model.
This plots both measures of that distance against the step size ``s``.

The two are deliberately drawn together because they are not the same curve.
The squared score norm is a plain sum of squares, so the step contributes
``s**2`` and the curve is least at ``s = 0``: that is the Pythagoras argument
the chapter gives for the direct-inversion solution being the minimum-norm one.
Hotelling's T-squared divides each score by its own standard deviation before
squaring, so it is a weighted sum of squares, and its least value sits slightly
away from zero.

Requires the ``process_improve`` package (``pip install process-improve``) for
``PLS``.

Usage::

    python pls/pls-null-space-distance.py [output_dir]

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

ORANGE = "#e6820a"     # the squared score norm, matching the null space elsewhere
DARK_BLUE = "#1f3d7a"  # Hotelling's T-squared
MAROON = "#7b1d2b"     # where T-squared is least
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
    """Fit the model, walk the null space, and draw both distance curves."""
    cheese = pd.read_csv(DATA_URL)
    train = cheese.iloc[4:]  # cheeses 5 to 30, as in the chapter
    pls = PLS(n_components=2).fit(train[X_COLUMNS], train[["Taste"]])

    result = pls.invert(y_desired=TARGET_TASTE)
    tau = result.scores.to_numpy().ravel()
    g = result.null_space_basis.to_numpy().ravel()
    sf = pls.scaling_factor_for_scores_.to_numpy()  # one standard deviation per score

    steps = np.linspace(-2.0, 2.0, 401)
    points = tau[None, :] + steps[:, None] * g[None, :]
    norm_sq = (points**2).sum(axis=1)
    t2 = ((points / sf) ** 2).sum(axis=1)

    # Where the weighted curve is least, found by setting its derivative to zero.
    s_t2 = -float((tau / sf**2) @ g) / float((g / sf**2) @ g)

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(steps, norm_sq, color=ORANGE, lw=2.4, label=r"Squared score norm $\|\tau\|^2$")
    ax.plot(steps, t2, color=DARK_BLUE, lw=2.4, label="Hotelling's $T^2$")

    ax.plot([0], [float(tau @ tau)], marker="o", ms=9, color=ORANGE, mec=BLACK, mew=0.9,
            zorder=5, label="Least norm, at $s = 0$")
    ax.plot([s_t2], [float((((tau + s_t2 * g) / sf) ** 2).sum())], marker="s", ms=9,
            color=MAROON, mec=BLACK, mew=0.9, zorder=5, label=f"Least $T^2$, at $s = {s_t2:.3f}$")
    ax.axvline(0, color=ORANGE, lw=1.0, ls=":", zorder=1)
    ax.axvline(s_t2, color=MAROON, lw=1.0, ls=":", zorder=1)

    # The three rows tabulated in the chapter, labelled clear of the curve and of
    # the two minimum markers, which sit close together near the origin.
    offsets = {-1.0: (-34, 12), 0.0: (14, 30), 1.0: (10, -20)}
    markers = {-1.0: "v", 0.0: "s", 1.0: "^"}
    for step, offset in offsets.items():
        value = float((((tau + step * g) / sf) ** 2).sum())
        ax.plot([step], [value], marker=markers[step], ms=8, color="none",
                mec=DARK_BLUE, mew=1.6, zorder=6)
        ax.annotate(
            f"$T^2 = {value:.2f}$", xy=(step, value), xytext=offset,
            textcoords="offset points", fontsize=9, color=DARK_BLUE,
            arrowprops={"arrowstyle": "-", "color": DARK_BLUE, "lw": 0.8,
                        "shrinkA": 1, "shrinkB": 4},
        )

    ax.set_xlabel("Step $s$ along the null space, from the direct-inversion solution")
    ax.set_ylabel("Squared distance from the model centre")
    ax.set_title("The predicted taste is flat along the null space; the distance is not")
    ax.set_xlim(-2.05, 2.05)
    ax.set_ylim(bottom=0, top=9.6)
    ax.legend(loc="upper center", fontsize=9, framealpha=0.93)
    fig.tight_layout()
    fig.savefig(out_dir / "pls-null-space-distance.png")
    plt.close(fig)


def main() -> None:
    """Entry point: render into the given directory (default: this script's own)."""
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
    build_figure(out_dir)


if __name__ == "__main__":
    main()
