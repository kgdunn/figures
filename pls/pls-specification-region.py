"""Generate the committed PNG contrasting a multivariate specification region with a box of ranges.

Sixth figure in the ``pls/`` set for the pid-book section "Turning the inversion
around: a specification region", in
``latent-variable-modelling/projection-to-latent-structures/pls-model-inversion-and-the-orthogonal-space.rst``.

A taste specification of 20 to 30 is one interval on one number. Inverted, it
becomes a flat, slanted region in the three inputs. Reporting that region as
three independent ranges, one per input, describes a box instead, and the box is
not the region: every one of its eight corners satisfies all three ranges while
failing the specification the ranges were derived from.

Left panel: how the region is built, in the score plot. Each acceptable taste has
its own null space, those null spaces are parallel, and sweeping the target from
20 to 30 sweeps the line across the plot. The Hotelling's T-squared limit closes
the region off in the other direction.

Right panel: the same region in the three inputs, with the enclosing box drawn
around it. The region is flat because every point on it is rebuilt from two
scores. The box is a solid. The corners are marked with the taste each one is
predicted to have.

Requires the ``process_improve`` package (``pip install process-improve``) for
``PLS``.

Usage::

    python pls/pls-specification-region.py [output_dir]

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
TASTE_LOW, TASTE_HIGH = 20.0, 30.0
CONFIDENCE = 0.95

DARK_BLUE = "#1f3d7a"  # calibration cheeses
ORANGE = "#e6820a"     # the specification region
MAROON = "#7b1d2b"     # the corners of the box, none of them acceptable
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
    """Fit the model, sweep the acceptable range, and draw the region against its box."""
    cheese = pd.read_csv(DATA_URL)
    train = cheese.iloc[4:]  # cheeses 5 to 30, as in the chapter
    x_block, y_block = train[X_COLUMNS], train[["Taste"]]
    pls = PLS(n_components=2).fit(x_block, y_block)

    t2_limit = float(pls.hotellings_t2_limit(CONFIDENCE))
    sf = pls.scaling_factor_for_scores_.to_numpy()
    q = pls.y_loadings_.to_numpy().ravel()
    loadings = pls.x_loadings_.to_numpy()
    weights = pls.direct_weights_.to_numpy()
    y_mean, y_std = float(y_block.mean().iloc[0]), float(y_block.std().iloc[0])
    x_mean, x_std = x_block.mean().to_numpy(), x_block.std().to_numpy()

    def taste_of(scores: np.ndarray) -> np.ndarray:
        """Predicted taste, in the original units, for scores given as rows."""
        return (scores @ q) * y_std + y_mean

    def t2_of(scores: np.ndarray) -> np.ndarray:
        """Hotelling's T-squared for scores given as rows."""
        return ((scores / sf) ** 2).sum(axis=1)

    # The region, swept out on a dense grid of the score plane.
    axis = np.linspace(-6, 6, 1500)
    grid = np.column_stack([a.ravel() for a in np.meshgrid(axis, axis)])
    inside = (t2_of(grid) <= t2_limit) & (taste_of(grid) >= TASTE_LOW) & (taste_of(grid) <= TASTE_HIGH)
    region_scores = grid[inside]
    region_inputs = region_scores @ loadings.T * x_std + x_mean

    low, high = region_inputs.min(axis=0), region_inputs.max(axis=0)

    # The eight corners of the box those three ranges describe.
    corners = np.array(np.meshgrid(*([lo, hi] for lo, hi in zip(low, high)))).reshape(3, -1).T
    corner_scores = (corners - x_mean) / x_std @ weights
    corner_taste = taste_of(corner_scores)

    fig = plt.figure(figsize=(12.8, 5.9))
    ax_scores = fig.add_subplot(1, 2, 1)
    ax_inputs = fig.add_subplot(1, 2, 2, projection="3d")

    # --- left: how the region is built, in the score plot ---------------------
    mesh_1, mesh_2 = np.meshgrid(axis, axis)
    mask = inside.reshape(mesh_1.shape).astype(float)
    ax_scores.contourf(mesh_1, mesh_2, mask, levels=[0.5, 1.5], colors=[ORANGE], alpha=0.28,
                       zorder=2)
    angle = np.linspace(0, 2 * np.pi, 400)
    ax_scores.plot(sf[0] * np.sqrt(t2_limit) * np.cos(angle),
                   sf[1] * np.sqrt(t2_limit) * np.sin(angle),
                   color=DARK_BLUE, lw=1.6, ls="--", zorder=4,
                   label=f"{CONFIDENCE:.0%} $T^2$ limit")
    span = np.linspace(-6, 6, 2)
    for target, style, nudge in ((TASTE_LOW, ":", -0.55), (TASTE_HIGH, "-", 0.55)):
        centre = pls.invert(target).scores.to_numpy()
        direction = pls.invert(target).null_space_basis.to_numpy().ravel()
        line = np.array([centre + s * direction for s in span])
        ax_scores.plot(line[:, 0], line[:, 1], color=ORANGE, lw=2.2, ls=style, zorder=5)
        # Label each boundary beside its own line, nudged along the gradient so the two
        # labels cannot be mistaken for one another.
        along = centre + 2.35 * direction + nudge * (q / np.linalg.norm(q))
        ax_scores.annotate(f"taste {target:.0f}", xy=tuple(along), fontsize=9.5, color=ORANGE,
                           fontweight="bold", ha="center", va="center",
                           bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8,
                                 "boxstyle": "round,pad=0.15"}, zorder=7)
    scores = pls.scores_.to_numpy()
    ax_scores.scatter(scores[:, 0], scores[:, 1], s=26, color=DARK_BLUE, alpha=0.65, lw=0,
                      zorder=6, label="Calibration cheeses")
    ax_scores.plot([], [], color=ORANGE, lw=6, alpha=0.35,
                   label=f"Acceptable: taste {TASTE_LOW:.0f} to {TASTE_HIGH:.0f}")
    ax_scores.axhline(0, color="0.8", lw=0.8, zorder=0)
    ax_scores.axvline(0, color="0.8", lw=0.8, zorder=0)
    ax_scores.set_xlim(-4.2, 4.2)
    ax_scores.set_ylim(-4.2, 4.2)
    ax_scores.set_aspect("equal")
    ax_scores.set_xlabel("$t_1$")
    ax_scores.set_ylabel("$t_2$")
    ax_scores.set_title("Sweeping the target sweeps its null space", fontsize=11.5)
    ax_scores.legend(loc="upper left", fontsize=8.5, framealpha=0.93)

    # --- right: the region against the box of three ranges --------------------
    thinned = region_inputs[:: max(1, len(region_inputs) // 4000)]
    ax_inputs.scatter(thinned[:, 0], thinned[:, 1], thinned[:, 2], s=4, color=ORANGE,
                      alpha=0.30, lw=0, depthshade=False)
    for start, end in (
        (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
        (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
    ):
        edge = np.vstack([corners[start], corners[end]])
        ax_inputs.plot(edge[:, 0], edge[:, 1], edge[:, 2], color=BLACK, lw=0.9, ls="--", alpha=0.65)
    ax_inputs.scatter(corners[:, 0], corners[:, 1], corners[:, 2], s=55, color=MAROON,
                      edgecolors="white", linewidths=0.8, depthshade=False, zorder=10)
    for corner, taste in zip(corners, corner_taste):
        ax_inputs.text(corner[0], corner[1], corner[2], f"  {taste:.0f}", fontsize=8,
                       color=MAROON, fontweight="bold")
    ax_inputs.set_xlabel("Acetic")
    ax_inputs.set_ylabel("H2S")
    ax_inputs.set_zlabel("Lactic")
    ax_inputs.set_title("The region is flat; the box of ranges is not", fontsize=11.5)
    ax_inputs.view_init(elev=20, azim=-58)
    ax_inputs.grid(visible=False)


    fig.tight_layout()
    # Placed at figure level, below both axes, so it cannot collide with the 3-D tick labels.
    fig.subplots_adjust(bottom=0.17)
    fig.text(
        0.74, 0.045,
        f"Each corner carries the taste it is predicted to have. All eight satisfy every one of the\n"
        f"three ranges, and none of them falls in the {TASTE_LOW:.0f} to {TASTE_HIGH:.0f} window "
        f"the ranges came from.",
        ha="center", va="bottom", fontsize=9.5, color=MAROON,
    )
    fig.savefig(out_dir / "pls-specification-region.png")
    plt.close(fig)

    print(f"box: {np.round(low, 2)} to {np.round(high, 2)}")
    print(f"corner tastes: {np.round(np.sort(corner_taste), 1)}")


def main() -> None:
    """Entry point: render into the given directory (default: this script's own)."""
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
    build_figure(out_dir)


if __name__ == "__main__":
    main()
