"""Generate the committed PNG contrasting a multivariate specification region with a box of ranges.

Sixth figure in the ``pls/`` set for the pid-book section "Turning the inversion
around: a specification region", in
``latent-variable-modelling/projection-to-latent-structures/pls-model-inversion-and-the-orthogonal-space.rst``.

A taste specification of 20 to 30 is one interval on one number. Inverted, it
becomes a flat, slanted region in the three inputs. Reporting that region as
three independent ranges, one per input, describes a box instead, and the box is
not the region: every one of its eight corners satisfies all three ranges while
failing the specification the ranges were derived from. They fail in two
different ways, which the panel distinguishes by colour. Six predict a taste
outside the window. The other two predict a perfectly acceptable taste but sit
far outside the Hotelling's T-squared limit, so they are extrapolations the data
do not support.

Left panel: how the region is built, in the score plot. Each acceptable taste has
its own null space, those null spaces are parallel, and sweeping the target from
20 to 30 sweeps the line across the plot. The Hotelling's T-squared limit closes
the region off in the other direction.

Right panel: the same region in the three inputs, with the enclosing box drawn
around it. The region is flat because every point on it is rebuilt from two
scores. The box is a solid.

Five points are marked in both panels with the same shape and colour, so a
location in the score plot can be followed to the recipe it stands for: the four
corners of the region, where the outer null spaces meet the T-squared limit, and
its centre, the direct-inversion solution for a taste of 25.

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
MAROON = "#7b1d2b"     # box corners rejected because the predicted taste is outside the window
DEEP_BLUE = "#123a6b"  # box corners whose taste is fine but which lie beyond the T-squared limit
GREY = "0.55"
BLACK = "#111111"

# The four corners of the region and its centre, marked identically in both panels.
# Colour carries the target taste, shape carries which end of that taste's null space,
# so the pairing can be read off without consulting the legend.
TASTE_20, TASTE_30 = "#6a3d9a", "#1b9e77"
LANDMARKS = (
    (TASTE_20, "v", "Taste 20, low end"),
    (TASTE_20, "^", "Taste 20, high end"),
    (TASTE_30, "v", "Taste 30, low end"),
    (TASTE_30, "^", "Taste 30, high end"),
    ("#111111", "*", "Region centre: taste 25"),
)

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
    y_mean, y_std = float(y_block.mean().iloc[0]), float(y_block.std().iloc[0])
    x_mean, x_std = x_block.mean().to_numpy(), x_block.std().to_numpy()

    def taste_of(scores: np.ndarray) -> np.ndarray:
        """Predicted taste, in the original units, for scores given as rows."""
        return (scores @ q) * y_std + y_mean

    def t2_of(scores: np.ndarray) -> np.ndarray:
        """Hotelling's T-squared for scores given as rows."""
        return ((scores / sf) ** 2).sum(axis=1)

    # The region, swept exactly as the chapter sweeps it: 11 acceptable tastes, and
    # 50 steps along each one's null space. The walk range covers the whole chord the
    # T-squared limit allows, which is about plus or minus 2.1.
    swept_scores, swept_inputs = [], []
    for target in np.linspace(TASTE_LOW, TASTE_HIGH, 11):
        for step in np.linspace(-2.5, 2.5, 50):
            candidate = pls.invert(target, null_space_coordinates=[step])
            if candidate.hotellings_t2 <= t2_limit:
                swept_scores.append(candidate.scores.to_numpy())
                swept_inputs.append(candidate.x_new.to_numpy())
    swept_scores = np.array(swept_scores)
    region_inputs = np.array(swept_inputs)
    low, high = region_inputs.min(axis=0), region_inputs.max(axis=0)

    def chord(target: float) -> np.ndarray:
        """The two steps at which this target's null space crosses the T-squared limit."""
        base = pls.invert(target)
        centre = base.scores.to_numpy()
        direction = base.null_space_basis.to_numpy().ravel()
        a = ((direction / sf) ** 2).sum()
        b = 2 * ((centre * direction) / sf**2).sum()
        c = ((centre / sf) ** 2).sum() - t2_limit
        root = np.sqrt(b**2 - 4 * a * c)
        return np.array([(-b - root) / (2 * a), (-b + root) / (2 * a)])

    # Four corners of the region, plus its centre, in both coordinate systems.
    marks = []
    for target in (TASTE_LOW, TASTE_HIGH):
        for step in chord(target):
            marks.append((target, float(step)))
    marks.append(((TASTE_LOW + TASTE_HIGH) / 2, 0.0))
    mark_scores = np.array([pls.invert(t, null_space_coordinates=[s]).scores.to_numpy()
                            for t, s in marks])
    mark_inputs = np.array([pls.invert(t, null_space_coordinates=[s]).x_new.to_numpy()
                            for t, s in marks])

    # The eight corners of the box those three ranges describe.
    corners = np.array(np.meshgrid(*([lo, hi] for lo, hi in zip(low, high)))).reshape(3, -1).T
    corner_frame = pd.DataFrame(corners, columns=X_COLUMNS)
    corner_taste = pls.predict(corner_frame).to_numpy().ravel()
    corner_t2 = pls.diagnose(corner_frame).hotellings_t2.to_numpy()
    taste_ok = (corner_taste >= TASTE_LOW) & (corner_taste <= TASTE_HIGH)

    fig = plt.figure(figsize=(13.2, 6.2))
    ax_scores = fig.add_subplot(1, 2, 1)
    ax_inputs = fig.add_subplot(1, 2, 2, projection="3d")

    # --- left: how the region is built, in the score plot ---------------------
    ax_scores.scatter(swept_scores[:, 0], swept_scores[:, 1], s=3, color=ORANGE, alpha=0.55,
                      lw=0, zorder=2, label=f"{len(swept_scores)} inverted designs")
    angle = np.linspace(0, 2 * np.pi, 400)
    ax_scores.plot(sf[0] * np.sqrt(t2_limit) * np.cos(angle),
                   sf[1] * np.sqrt(t2_limit) * np.sin(angle),
                   color=DARK_BLUE, lw=1.6, ls="--", zorder=4,
                   label=f"{CONFIDENCE:.0%} $T^2$ limit")
    span = np.linspace(-6, 6, 2)
    for target, style, nudge in ((TASTE_LOW, ":", -0.62), (TASTE_HIGH, "-", 0.62)):
        base = pls.invert(target)
        centre = base.scores.to_numpy()
        direction = base.null_space_basis.to_numpy().ravel()
        line = np.array([centre + s * direction for s in span])
        ax_scores.plot(line[:, 0], line[:, 1], color=ORANGE, lw=1.8, ls=style, zorder=3)
        along = centre + 2.9 * direction + nudge * (q / np.linalg.norm(q))
        ax_scores.annotate(f"taste {target:.0f}", xy=tuple(along), fontsize=9.5, color=ORANGE,
                           fontweight="bold", ha="center", va="center",
                           bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8,
                                 "boxstyle": "round,pad=0.15"}, zorder=7)
    scores = pls.scores_.to_numpy()
    ax_scores.scatter(scores[:, 0], scores[:, 1], s=24, color=DARK_BLUE, alpha=0.55, lw=0,
                      zorder=5, label="Calibration cheeses")
    for point, (colour, marker, label) in zip(mark_scores, LANDMARKS):
        ax_scores.scatter(*point, color=colour, marker=marker, s=150, edgecolors="white",
                          linewidths=1.1, zorder=9, label=label)
    ax_scores.axhline(0, color="0.8", lw=0.8, zorder=0)
    ax_scores.axvline(0, color="0.8", lw=0.8, zorder=0)
    ax_scores.set_xlim(-4.2, 4.2)
    ax_scores.set_ylim(-4.2, 4.2)
    ax_scores.set_aspect("equal")
    ax_scores.set_xlabel("$t_1$")
    ax_scores.set_ylabel("$t_2$")
    ax_scores.set_title("Sweeping the target sweeps its null space", fontsize=11.5)
    ax_scores.legend(loc="upper left", fontsize=7.4, framealpha=0.94, borderpad=0.5)

    # --- right: the region against the box of three ranges --------------------
    # Large enough for the swept points to read as a surface rather than a dusting.
    ax_inputs.scatter(region_inputs[:, 0], region_inputs[:, 1], region_inputs[:, 2], s=22,
                      color=ORANGE, alpha=0.5, lw=0, depthshade=False)
    for start, end in (
        (0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3),
        (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7),
    ):
        edge = np.vstack([corners[start], corners[end]])
        ax_inputs.plot(edge[:, 0], edge[:, 1], edge[:, 2], color=BLACK, lw=0.9, ls="--", alpha=0.6)
    for corner, taste, t2_value, ok in zip(corners, corner_taste, corner_t2, taste_ok):
        colour = DEEP_BLUE if ok else MAROON
        ax_inputs.scatter(*corner, s=48, color=colour, edgecolors="white", linewidths=0.8,
                          depthshade=False, zorder=8)
        note = f"  {taste:.0f}" if not ok else f"  {taste:.0f}, $T^2$={t2_value:.0f}"
        ax_inputs.text(corner[0], corner[1], corner[2], note, fontsize=8, color=colour,
                       fontweight="bold")
    for point, (colour, marker, _label) in zip(mark_inputs, LANDMARKS):
        ax_inputs.scatter(*point, color=colour, marker=marker, s=150, edgecolors="white",
                          linewidths=1.1, depthshade=False, zorder=10)
    ax_inputs.set_xlabel("Acetic", labelpad=10)
    ax_inputs.set_ylabel("H2S", labelpad=12)
    ax_inputs.set_zlabel("Lactic", labelpad=8)
    ax_inputs.tick_params(axis="both", pad=2, labelsize=9)
    ax_inputs.set_title("The region is flat; the box of ranges is not", fontsize=11.5)
    ax_inputs.view_init(elev=20, azim=-138)
    ax_inputs.grid(visible=False)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)
    fig.text(
        0.74, 0.045,
        "Every corner of the box satisfies all three ranges, and none is acceptable. Corners in\n"
        f"dark red predict a taste outside {TASTE_LOW:.0f} to {TASTE_HIGH:.0f}. The two in blue "
        f"predict an acceptable taste,\nbut sit beyond the $T^2$ limit of {t2_limit:.2f}, "
        "so the data do not support them.",
        ha="center", va="bottom", fontsize=9.5, color=BLACK,
    )
    fig.savefig(out_dir / "pls-specification-region.png")
    plt.close(fig)

    print(f"box: {np.round(low, 2)} to {np.round(high, 2)}, from {len(region_inputs)} kept designs")
    for taste, t2_value, ok in sorted(zip(corner_taste, corner_t2, taste_ok)):
        print(f"  corner taste {taste:6.2f}  T2 {t2_value:6.2f}  taste in window: {ok}")


def main() -> None:
    """Entry point: render into the given directory (default: this script's own)."""
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
    build_figure(out_dir)


if __name__ == "__main__":
    main()
