"""Generate the committed PNG explaining the geometry of the PLS null space.

Companion to ``pls/pls-model-inversion-null-space.py``. That figure shows the
null space; this one shows *why* it runs in the direction it does, for the
pid-book section "Where the null-space line comes from" in
``latent-variable-modelling/projection-to-latent-structures/pls-model-inversion-and-the-orthogonal-space.rst``.

The prediction of a two-component PLS model is ``yhat = q1*t1 + q2*t2``, so the
y-loading vector ``q`` is the gradient of the prediction in the score plot.
Fixing a target response gives one linear equation in two unknowns, whose
solutions form a line perpendicular to ``q``: a contour of the predicted
response. Different targets give parallel contours. The axes are drawn on a
common scale (``aspect="equal"``) so that the right angle between ``q`` and the
contours is visible rather than merely asserted. The two null-space steps from the
companion figure are marked so the same points can be found in both.

The right panel enlarges the boxed region of the left, because the argument for
the direct-inversion solution being the shortest one lives at the scale of a
single step while the contour family spans the whole score plot. The three
arrows there form a right triangle: from the origin to the perpendicular foot,
from the foot along the contour, and the hypotenuse back to the origin.
Pythagoras on that triangle is the argument.

Requires the ``process_improve`` package (``pip install process-improve``) for
``PLS``.

Usage::

    python pls/pls-null-space-geometry.py [output_dir]

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
from matplotlib.patches import Rectangle

from process_improve.multivariate import PLS

DATA_URL = "https://openmv.net/file/cheddar-cheese.csv"
X_COLUMNS = ["Acetic", "H2S", "Lactic"]
MAIN_TASTE = 20.9                       # the contour drawn in orange
OTHER_TASTES = (10.0, 30.0, 40.0)       # further contours, drawn in grey

DARK_BLUE = "#1f3d7a"  # calibration cheeses
ORANGE = "#e6820a"     # the null space for the main target
MAROON = "#7b1d2b"     # the y-loading vector q: the gradient, perpendicular to the contours
DARK_BLUE = "#1f3d7a"  # the step and the hypotenuse in the enlarged panel
GREY = "0.55"          # contours for the other targets
BLACK = "#111111"      # marker outlines and the geometric annotations
REFERENCE_AREA = 22.0            # marker area, in points^2, for a cheese of
TASTE_AT_REFERENCE_AREA = 20.0   # this taste; area scales proportionally from there
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 140,
    }
)


def build_figure(out_dir: Path) -> None:
    """Fit the model, then draw the gradient and the contours it induces."""
    cheese = pd.read_csv(DATA_URL)
    train = cheese.iloc[4:]  # cheeses 5 to 30, as in the chapter
    pls = PLS(n_components=2).fit(train[X_COLUMNS], train[["Taste"]])

    q = pls.y_loadings_.to_numpy().ravel()  # gradient of the prediction
    q_unit = q / np.linalg.norm(q)
    g = pls.invert(MAIN_TASTE).null_space_basis.to_numpy().ravel()  # perpendicular
    tau = pls.invert(MAIN_TASTE).scores.to_numpy()
    scores = pls.scores_.to_numpy()

    steps = np.linspace(-5.0, 5.0, 50)

    fig, (ax, ax_zoom) = plt.subplots(1, 2, figsize=(12.6, 6.2))

    # Fix the drawing area first, so every label can be placed inside it.
    x_limits, y_limits = (-3.6, 3.6), (-4.4, 4.4)
    label_y = y_limits[0] + 0.55       # contours are labelled near the bottom edge
    slope = -q[0] / q[1]

    def label_point(centre: np.ndarray) -> tuple[float, float]:
        """Where a contour through *centre* crosses the labelling height."""
        intercept = centre[1] - slope * centre[0]
        return ((label_y - intercept) / slope, label_y)

    # Contours for the other target tastes: parallel, and unlabelled in the legend.
    for taste in OTHER_TASTES:
        centre = pls.invert(taste).scores.to_numpy()
        line = np.array([centre + s * g for s in steps])
        ax.plot(line[:, 0], line[:, 1], color=GREY, lw=1.2, linestyle=":", zorder=1)
        ax.annotate(
            f"{taste:.0f}", xy=label_point(centre), xytext=(4, 0),
            textcoords="offset points", color=GREY, fontsize=9, va="center",
        )

    # The main contour: every score on it predicts a taste of MAIN_TASTE.
    line = np.array([tau + s * g for s in steps])
    ax.plot(
        line[:, 0], line[:, 1], color=ORANGE, lw=2.4,
        label=f"Null space: predicted taste = {MAIN_TASTE}", zorder=2,
    )
    ax.annotate(
        f"{MAIN_TASTE}", xy=label_point(tau), xytext=(4, 0),
        textcoords="offset points", color=ORANGE, fontsize=9, va="center", fontweight="bold",
    )

    # Marker area is proportional to the measured taste, as in the companion figure.
    taste = train["Taste"].to_numpy()
    cal = ax.scatter(scores[:, 0], scores[:, 1], color=DARK_BLUE, alpha=0.55, zorder=3,
                     s=REFERENCE_AREA * taste / TASTE_AT_REFERENCE_AREA,
                     label="Calibration cheeses")

    # The gradient q, drawn from the origin. The prediction rises fastest this way.
    arrow_length = 2.6
    ax.annotate(
        "", xy=tuple(q_unit * arrow_length), xytext=(0, 0),
        arrowprops={"arrowstyle": "-|>", "color": MAROON, "lw": 2.4, "mutation_scale": 18},
        zorder=5,
    )
    ax.annotate(
        r"$\mathbf{q}$", xy=tuple(q_unit * arrow_length), xytext=(10, -6),
        textcoords="offset points", color=MAROON, fontsize=13, fontweight="bold",
    )

    # The direct-inversion solution, and the perpendicular from the origin to it.
    ax.plot([0, tau[0]], [0, tau[1]], color=BLACK, lw=1.2, linestyle="--", zorder=4)
    ax.scatter(tau[0], tau[1], color=ORANGE, marker="s", s=90, zorder=6,
               edgecolors=BLACK, linewidths=0.8, label="Direct-inversion solution")

    # The -1 and +1 null-space steps, drawn as in the companion score plot so the
    # same two points can be recognised across both figures.
    g_step = g / np.linalg.norm(g)
    ax.scatter(*(tau - g_step), color=ORANGE, marker="v", s=110, zorder=6,
               edgecolors=BLACK, linewidths=0.8, label="Null-space step (-1)")
    ax.scatter(*(tau + g_step), color=ORANGE, marker="^", s=110, zorder=6,
               edgecolors=BLACK, linewidths=0.8, label="Null-space step (+1)")

    # A right-angle marker at the direct-inversion solution, between q and the contour.
    size = 0.32
    g_unit = g / np.linalg.norm(g)
    corner = np.array([
        tau + size * g_unit,
        tau + size * g_unit - size * q_unit,
        tau - size * q_unit,
    ])
    ax.plot(corner[:, 0], corner[:, 1], color=BLACK, lw=1.1, zorder=6)

    # The region the right panel enlarges.
    zoom_x, zoom_y = (-1.05, 0.85), (-0.60, 1.55)
    ax.add_patch(
        Rectangle((zoom_x[0], zoom_y[0]), zoom_x[1] - zoom_x[0], zoom_y[1] - zoom_y[0],
                  fill=False, edgecolor=BLACK, lw=1.0, linestyle="--", zorder=8)
    )
    ax.annotate("enlarged at right", xy=(sum(zoom_x) / 2, zoom_y[1]), xytext=(0, 4),
                textcoords="offset points", fontsize=8.5, color=BLACK, ha="center", va="bottom",
                zorder=9)

    ax.axhline(0, color="0.75", lw=0.8, zorder=0)
    ax.axvline(0, color="0.75", lw=0.8, zorder=0)
    ax.set_xlim(*x_limits)
    ax.set_ylim(*y_limits)
    ax.set_aspect("equal")  # so the right angle is a right angle on the page
    ax.set_xlabel("$t_1$")
    ax.set_ylabel("$t_2$")
    ax.set_title("Contours of predicted taste, perpendicular to the gradient", fontsize=11.5)
    legend = ax.legend(loc="upper left", framealpha=0.92, fontsize=8.5)
    # Show the cheeses at the reference size, so the legend doubles as a size key.
    handles = getattr(legend, "legend_handles", None) or legend.legendHandles
    for handle, text in zip(handles, legend.get_texts()):
        if text.get_text().startswith("Calibration"):
            handle.set_sizes([REFERENCE_AREA])
    # --- right panel: why the perpendicular foot is the shortest solution ----
    candidate = tau + g_step
    ax_zoom.plot(line[:, 0], line[:, 1], color=ORANGE, lw=2.6, zorder=3)
    ax_zoom.annotate("", xy=tuple(tau), xytext=(0, 0),
                     arrowprops={"arrowstyle": "-|>", "color": BLACK, "lw": 2.2,
                                 "mutation_scale": 15}, zorder=7)
    ax_zoom.annotate("", xy=tuple(candidate), xytext=tuple(tau),
                     arrowprops={"arrowstyle": "-|>", "color": DARK_BLUE, "lw": 2.5,
                                 "mutation_scale": 15}, zorder=7)
    ax_zoom.annotate("", xy=tuple(candidate), xytext=(0, 0),
                     arrowprops={"arrowstyle": "-|>", "color": DARK_BLUE, "lw": 1.4,
                                 "mutation_scale": 12, "ls": (0, (5, 3)), "alpha": 0.8}, zorder=6)
    ax_zoom.scatter(*tau, color=ORANGE, marker="s", s=100, edgecolors=BLACK, linewidths=0.9, zorder=9)
    ax_zoom.scatter(*candidate, color=ORANGE, marker="^", s=120, edgecolors=BLACK,
                    linewidths=0.9, zorder=9)
    ax_zoom.scatter([0], [0], color=BLACK, s=26, zorder=9)

    tau_unit = tau / np.linalg.norm(tau)
    small = 0.10
    corner_zoom = np.array([
        tau + small * g_step,
        tau + small * g_step - small * tau_unit,
        tau - small * tau_unit,
    ])
    ax_zoom.plot(corner_zoom[:, 0], corner_zoom[:, 1], color=BLACK, lw=1.2, zorder=10)

    ax_zoom.annotate("origin", xy=(0, 0), xytext=(9, 4), textcoords="offset points",
                     fontsize=9, color=GREY)
    ax_zoom.annotate("direct inversion\n" r"$\|\mathbf{\tau}_{DI}\| = 0.281$", xy=tuple(tau),
                     xytext=(-102, -36), textcoords="offset points", fontsize=9, color=BLACK,
                     arrowprops={"arrowstyle": "-", "color": BLACK, "lw": 0.8})
    ax_zoom.annotate("the +1 step: same\ntaste, " r"$\|\cdot\| = 1.039$", xy=tuple(candidate),
                     xytext=(-118, 4), textcoords="offset points", fontsize=9, color=DARK_BLUE,
                     arrowprops={"arrowstyle": "-", "color": DARK_BLUE, "lw": 0.8})
    ax_zoom.annotate("this leg adds length\nbut changes nothing", xy=tuple(tau + 0.55 * g_step),
                     xytext=(24, -20), textcoords="offset points", fontsize=9, color=DARK_BLUE,
                     arrowprops={"arrowstyle": "-", "color": DARK_BLUE, "lw": 0.8})
    ax_zoom.text(0.035, 0.035,
                 r"$\|\mathbf{\tau}_{DI} + s\mathbf{g}\|^2 = \|\mathbf{\tau}_{DI}\|^2 + s^2$",
                 transform=ax_zoom.transAxes, ha="left", va="bottom", fontsize=11,
                 bbox={"facecolor": "white", "edgecolor": "0.78", "boxstyle": "round,pad=0.4"})

    ax_zoom.axhline(0, color="0.75", lw=0.8, zorder=0)
    ax_zoom.axvline(0, color="0.75", lw=0.8, zorder=0)
    ax_zoom.set_xlim(*zoom_x)
    ax_zoom.set_ylim(*zoom_y)
    ax_zoom.set_aspect("equal")
    ax_zoom.set_xlabel("$t_1$")
    ax_zoom.set_title("The step is at right angles, so it only adds length", fontsize=11.5)

    fig.tight_layout()
    fig.savefig(out_dir / "pls-null-space-geometry.png")
    plt.close(fig)


def main() -> None:
    """Entry point: render into the given directory (default: this script's own)."""
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
    build_figure(out_dir)


if __name__ == "__main__":
    main()
