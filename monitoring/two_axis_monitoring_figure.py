"""The two-variable monitoring figure for the multivariate monitoring chapter.

Writes ``two-axis-monitoring-plot.png``: two negatively correlated variables,
shown with their joint scatter plot and the Hotelling's T-squared ellipse.
Two observations are marked, one for each way the univariate and joint
views can disagree: observation 11 is inside both 3-sigma charts and
outside the ellipse, and observation 30 is outside one chart and inside the
ellipse. Neither view alone is sufficient, which is the point the section
makes.

The layout keeps the original arrangement: the joint scatter plot in the
corner, with each variable's time series running away from it along the
axis it belongs to, so a point can be followed from one panel to the other.

This replaces ``two-axis-monitoring-plot.py``, which was Python 2 and used
several matplotlib arguments that have since been removed
(``spines.iteritems()``, ``annotate(s=...)``, ``papertype``,
``orientation``). The data are the same: fifty observations from the same
seeded generator, with one point moved off the correlation line, and one
added along it.

Beyond redrawing, the figure now:

- draws the flagged observations in one colour rather than cycling through
  six, so "outlier" reads as one category instead of six;
- marks, in all three panels and in its own colour, each of the two
  observations on which the univariate and joint readings disagree, which
  is the comparison being made;
- labels both charts, signs the limits so the reader can tell the upper
  from the lower without reading the words, and puts each pair of limit
  labels at the far end of its panel, away from the joint plot;
- states the confidence level of the ellipse.

Usage
-----
    uv run --with numpy --with scipy --with matplotlib python two_axis_monitoring_figure.py [output_dir]
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

BLUE = "#0072B2"
VERMILLION = "#D55E00"
ORANGE = "#E69F00"
GREEN = "#009E73"
GREY = "#666666"
GRID = "#DDDDDD"

DPI = 300
HERE = pathlib.Path(__file__).parent

N = 50
K = 2
SIGMA = 3
SEED = 16
# The second variable is built to move against the first one.
SLOPE = -2.4
NOISE = 2.0


def simulated_data() -> np.ndarray:
    """Fifty observations of two negatively correlated variables."""
    generator = np.random.RandomState(SEED)
    data = generator.standard_normal((N, K))
    data[:, 1] = SLOPE * data[:, 0] + NOISE * data[:, 1]
    # Move observation 11 (index 10) off the correlation line, without
    # taking either variable outside its own limits: a joint outlier that
    # neither univariate chart sees.
    data[10, 0] = -2.0 + generator.standard_normal(1)[0] * 0.01
    data[10, 1] = SLOPE * data[10, 0] + NOISE * data[10, 1] - 12
    # Observation 30 (index 29) is the converse: low x1 with high x2, the
    # direction the two variables move in together, which is where the
    # ellipse reaches furthest. x1 lands past its own 3-sigma limit (3.81
    # against 3.45) while x2 is unremarkable at 45% of its limit, and the
    # pair sits at 74% of the joint limit, inside the ellipse.
    data[29] = np.array([-3.9, 4.0])
    return data - data.mean(axis=0)


def hotelling(data: np.ndarray, confidence: float):
    """T-squared per observation, its limit, and the ellipse in data units."""
    scale = data.std(axis=0, ddof=1)
    scaled = (data - data.mean(axis=0)) / scale
    _, singular_values, right = np.linalg.svd(scaled.T @ scaled)
    spread = np.sqrt(singular_values / len(data))
    loadings = right.T
    scores = scaled @ loadings
    t2 = ((scores / spread) ** 2).sum(axis=1)
    limit = (K * (len(data) - 1) * (len(data) + 1) / (len(data) * (len(data) - K))
             * scipy.stats.f.ppf(confidence, K, len(data) - K))

    angle = np.linspace(0, 2 * np.pi, 200)
    ellipse = np.column_stack([
        np.sqrt(limit) * spread[0] * np.cos(angle),
        np.sqrt(limit) * spread[1] * np.sin(angle),
    ])
    return t2, limit, ellipse @ loadings * scale + data.mean(axis=0)


def main(outdir: pathlib.Path) -> None:
    data = simulated_data()
    # The ellipse carries the same nominal confidence as the charts beside
    # it: norm.cdf(3) = 99.87% is the one-sided normal tail at 3 sigma. Tying
    # the two together is what makes the comparison in this figure fair, since
    # neither method can then be said to have been given the easier limit.
    confidence = float(scipy.stats.norm.cdf(SIGMA))
    upper = data.mean(axis=0) + SIGMA * data.std(axis=0, ddof=1)
    lower = data.mean(axis=0) - SIGMA * data.std(axis=0, ddof=1)

    t2, limit, ellipse = hotelling(data, confidence)
    univariate = np.any((data < lower) | (data > upper), axis=1)
    joint = t2 > limit
    print(f"correlation between the two variables: {np.corrcoef(data.T)[0, 1]:.3f}")
    print(f"{SIGMA}-sigma limits: x1 in [{lower[0]:.2f}, {upper[0]:.2f}], "
          f"x2 in [{lower[1]:.2f}, {upper[1]:.2f}]")
    print(f"Hotelling's T2 limit at {100 * confidence:.2f}%: {limit:.2f}")
    print("flagged by a univariate chart: "
          + (", ".join(str(i + 1) for i in np.flatnonzero(univariate)) or "none"))
    print("outside the ellipse: "
          + (", ".join(str(i + 1) for i in np.flatnonzero(joint)) or "none"))
    print("only the ellipse catches: "
          + (", ".join(str(i + 1) for i in np.flatnonzero(joint & ~univariate)) or "none"))
    print("only a univariate chart catches: "
          + (", ".join(str(i + 1) for i in np.flatnonzero(univariate & ~joint)) or "none"))

    span = np.max(np.abs(np.vstack([data, ellipse])), axis=0) * 1.12
    time = np.arange(1, N + 1)
    only_joint = joint & ~univariate            # the ellipse alone flags these
    only_univariate = univariate & ~joint       # a Shewhart chart alone flags these

    # The scatter plot sits in the top left; x1 runs down the left-hand
    # panel and x2 runs across the top panel, so each point can be traced
    # from the joint plot into the chart for its own variable.
    #
    # Two constraints fix the geometry, and the figure is square because of
    # them. The scatter panel is square, so that a standard deviation of x1
    # and one of x2 occupy the same distance: the axis limits are a common
    # multiple of each variable's own sigma, so the ellipse then shows the
    # correlation rather than a shape the panel imposed on it. And the
    # sequence axis is the same length in both marginal charts, the width of
    # the top panel matching the height of the left one, so the fifty
    # observations are drawn at one scale instead of spread across the top
    # and crowded down the side.
    panel = 0.335   # scatter: square, and the width and height the marginals share
    run = 0.475     # the sequence axis, the same length across and down
    left, bottom, gap = 0.085, 0.075, 0.045

    fig = plt.figure(figsize=(11.5, 11.5))
    scatter = fig.add_axes([left, bottom + run + gap, panel, panel])
    across = fig.add_axes([left + panel + gap, bottom + run + gap, run, panel])
    down = fig.add_axes([left, bottom, panel, run])

    scatter.grid(color=GRID, linewidth=0.8)
    scatter.axhline(0, color="black", linewidth=1.0)
    scatter.axvline(0, color="black", linewidth=1.0)
    scatter.plot(data[:, 0], data[:, 1], "o", markersize=7, color=BLUE)
    scatter.plot(data[joint, 0], data[joint, 1], "o", markersize=10, color=VERMILLION)
    scatter.plot(data[only_univariate, 0], data[only_univariate, 1], "o",
                 markersize=10, color=GREEN)
    scatter.plot(ellipse[:, 0], ellipse[:, 1], color=ORANGE, linewidth=2.5)
    for value in (lower[0], upper[0]):
        scatter.axvline(value, color=VERMILLION, linestyle="--", linewidth=1.4)
    for value in (lower[1], upper[1]):
        scatter.axhline(value, color=VERMILLION, linestyle="--", linewidth=1.4)
    scatter.set_xlim(-span[0], span[0])
    scatter.set_ylim(-span[1], span[1])
    scatter.set_xlabel("$x_1$")
    scatter.set_ylabel("$x_2$")
    scatter.annotate(f"{100 * confidence:.1f}% $T^2$ limit",
                     (ellipse[:, 0].min(), ellipse[:, 1].max()), color=ORANGE,
                     fontsize=14, ha="left", va="bottom")

    # x2 runs across the top panel, sharing the scatter plot's vertical
    # axis; x1 runs down the left panel, sharing its horizontal axis. Each
    # chart is therefore aligned with the variable it belongs to.
    across.grid(color=GRID, linewidth=0.8)
    across.plot(time, data[:, 1], "-o", color=BLUE, markersize=6, linewidth=1.2)
    across.plot(time[joint], data[joint, 1], "o", color=VERMILLION, markersize=9)
    across.plot(time[only_univariate], data[only_univariate, 1], "o", color=GREEN, markersize=9)
    across.axhline(0, color="black", linewidth=1.0)
    for value in (lower[1], upper[1]):
        across.axhline(value, color=VERMILLION, linestyle="--", linewidth=1.6)
    across.annotate(f"$+{SIGMA}\\sigma$ UCL", (N + 0.5, upper[1]), color=VERMILLION,
                    fontsize=14, ha="right", va="bottom",
                    textcoords="offset points", xytext=(0, 5))
    across.annotate(f"$-{SIGMA}\\sigma$ LCL", (N + 0.5, lower[1]), color=VERMILLION,
                    fontsize=14, ha="right", va="top",
                    textcoords="offset points", xytext=(0, -5))
    across.set_xlim(0, N + 1)
    across.set_ylim(-span[1], span[1])
    across.set_xlabel("Sequence order")
    across.set_ylabel("$x_2$")
    # Its y label would otherwise sit against the scatter plot.
    across.yaxis.set_label_position("right")
    across.spines["right"].set_visible(True)
    across.spines["left"].set_visible(False)
    across.yaxis.tick_right()

    down.grid(color=GRID, linewidth=0.8)
    down.plot(data[:, 0], time, "-o", color=BLUE, markersize=6, linewidth=1.2)
    down.plot(data[joint, 0], time[joint], "o", color=VERMILLION, markersize=9)
    down.plot(data[only_univariate, 0], time[only_univariate], "o", color=GREEN, markersize=9)
    down.axvline(0, color="black", linewidth=1.0)
    for value in (lower[0], upper[0]):
        down.axvline(value, color=VERMILLION, linestyle="--", linewidth=1.6)
    down.annotate(f"$-{SIGMA}\\sigma$ LCL", (lower[0], N * 0.88), color=VERMILLION,
                  fontsize=14, ha="right", va="center", rotation=90,
                  textcoords="offset points", xytext=(-5, 0))
    down.annotate(f"$+{SIGMA}\\sigma$ UCL", (upper[0], N * 0.88), color=VERMILLION,
                  fontsize=14, ha="left", va="center", rotation=90,
                  textcoords="offset points", xytext=(5, 0))
    down.set_xlim(-span[0], span[0])
    down.set_ylim(N + 1, 0)
    down.set_xlabel("$x_1$")
    down.set_ylabel("Sequence order")

    for marked, colour in ((only_joint, VERMILLION), (only_univariate, GREEN)):
        for index in np.flatnonzero(marked):
            scatter.annotate(f"{index + 1}", (data[index, 0], data[index, 1]),
                             textcoords="offset points", xytext=(10, -4),
                             color=colour, fontsize=15)
            across.annotate(f"{index + 1}", (index + 1, data[index, 1]),
                            textcoords="offset points", xytext=(8, 0),
                            color=colour, fontsize=15)
            down.annotate(f"{index + 1}", (data[index, 0], index + 1),
                          textcoords="offset points", xytext=(6, 12),
                          color=colour, fontsize=15)

    # The quarter opposite the scatter plot is the one place no axis can
    # reach, so the reading of the figure goes there. Each paragraph is in
    # the colour of the observation it describes.
    middle = left + panel + gap + run / 2
    fig.text(middle, bottom + run * 0.74,
             "Observation 11 lies inside the $3\\sigma$ limits of\n"
             "both charts, and outside the ellipse. It breaks\n"
             "the correlation between the two variables, and\n"
             "only the joint plot shows it.",
             fontsize=17, ha="center", va="center", color=VERMILLION)
    fig.text(middle, bottom + run * 0.44,
             "Observation 30 is the other way round. It is\n"
             "outside the $x_1$ chart, yet inside the ellipse. A low\n"
             "$x_1$ with a high $x_2$ is the direction these two\n"
             "variables move in together, and the joint limit\n"
             "reaches furthest that way, so the pair is ordinary\n"
             "even though one measurement on its own is not.",
             fontsize=17, ha="center", va="center", color=GREEN)
    fig.text(middle, bottom + run * 0.14,
             "Neither chart alone is enough.",
             fontsize=17, ha="center", va="center", color=GREY)

    fig.savefig(outdir / "two-axis-monitoring-plot.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outdir / 'two-axis-monitoring-plot.png'}")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE)
