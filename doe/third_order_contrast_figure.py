"""How a central composite design measures each factor's slope twice.

One PNG, ``third-order-contrasts.png``, for the third-order contrasts in the
response-surface section of the designed-experiments chapter. Four panels:

- (a) Where the two slopes of temperature come from in the bioreactor's
  composite design: the two axial runs on the temperature axis, at
  x_S = 0, and the four factorial runs, at x_S = -1 and +1.
- (b) The bioreactor's own runs plotted against coded temperature. The
  chord through the two axial runs and the chord through the two averages
  of the factorial runs are nearly parallel: 12.73 against 13.25 $/day per
  coded unit, a third-order contrast of -0.52.
- (c) The same fitted quadratic with a T^3 term added. The chord between
  two points at +h and -h has slope b_T + b_TTT h^2, so the wider axial
  pair (h = sqrt(2)) tilts more than the factorial pair (h = 1), and the
  contrast is (alpha^2 - 1) b_TTT.
- (d) The quadratic with a T x S^2 term added instead. That term changes
  the slope of temperature only where x_S is not zero, so it tilts the
  factorial chord and not the axial one, and the contrast is -b_TSS.

Under any second-order model the two chords are parallel, because a chord
between two points placed symmetrically about the centre has the slope of
the tangent at the centre, whatever the curvature. Only a third-order term
can make them differ, which is what the contrast detects.

The quadratic in panels (b) to (d) is the least-squares fit to the nine
bioreactor runs printed in the chapter, coded about T = 335 K and
S = 1.97 g/L: 688 + 12.99 x_T - 39.07 x_S - 2.25 x_T x_S - 4.19 x_T^2
- 12.19 x_S^2. The cubic coefficients in panels (c) and (d) are
illustrative, chosen large enough to see; the chapter reports only the
bioreactor's contrasts, which are close to zero.

Usage
-----
    uv run --with numpy --with matplotlib python third_order_contrast_figure.py [output_dir]

Writes the PNG into ``output_dir`` (default: this script's own directory).
"""

from __future__ import annotations

import pathlib
import sys

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

BLUE = "#0072B2"
VERMILLION = "#D55E00"
GREY = "#666666"
LIGHT_GREY = "#BBBBBB"
GRID = "#DDDDDD"

DPI = 300
HERE = pathlib.Path(__file__).parent
ALPHA = np.sqrt(2.0)

mpl.rcParams.update(
    {
        "font.size": 15,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 16,
        "axes.labelsize": 15,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 13,
        "axes.axisbelow": True,
    }
)

# The nine bioreactor runs, coded: run number, x_T, x_S, profit [$/day].
RUNS = [
    (8, -1, -1, 694),
    (9, +1, -1, 725),
    (10, -1, +1, 620),
    (11, +1, +1, 642),
    (6, 0, 0, 688),
    (13, 0, -ALPHA, 720),
    (14, +ALPHA, 0, 699),
    (15, 0, +ALPHA, 610),
    (16, -ALPHA, 0, 663),
]


def fit_quadratic() -> np.ndarray:
    """Least-squares fit of the full quadratic to the nine runs."""
    xt = np.array([r[1] for r in RUNS], dtype=float)
    xs = np.array([r[2] for r in RUNS], dtype=float)
    y = np.array([r[3] for r in RUNS], dtype=float)
    X = np.column_stack([np.ones_like(xt), xt, xs, xt * xs, xt**2, xs**2])
    return np.linalg.lstsq(X, y, rcond=None)[0]


B0, BT, BS, BTS, BTT, BSS = fit_quadratic()


def surface(xt, xs, b_ttt: float = 0.0, b_tss: float = 0.0):
    """The fitted quadratic, plus an optional T^3 or T x S^2 term."""
    quadratic = B0 + BT * xt + BS * xs + BTS * xt * xs + BTT * xt**2 + BSS * xs**2
    return quadratic + b_ttt * xt**3 + b_tss * xt * xs**2


def slopes(response) -> tuple[float, float, tuple, tuple]:
    """Axial and factorial slopes of temperature for a response function y(x_T, x_S).

    Returns the two slopes and the two pairs of points that define the
    chords: the axial runs at x_T = -alpha and +alpha (with x_S = 0), and the
    averages of the factorial runs at x_T = -1 and +1 (over x_S = -1 and +1).
    """
    axial = (response(-ALPHA, 0.0), response(ALPHA, 0.0))
    factorial = (
        (response(-1.0, -1.0) + response(-1.0, 1.0)) / 2,
        (response(1.0, -1.0) + response(1.0, 1.0)) / 2,
    )
    return (axial[1] - axial[0]) / (2 * ALPHA), (factorial[1] - factorial[0]) / 2, axial, factorial


def draw_design(ax) -> None:
    """Panel (a): the composite design, with the runs each slope uses."""
    ax.set_aspect("equal")
    ax.add_patch(plt.Circle((0, 0), ALPHA, fill=False, color=GRID, linewidth=1.2))
    ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], color=GRID, linewidth=1.2)
    for run, xt, xs, _ in RUNS:
        if xs == 0 and xt != 0:
            ax.plot(xt, xs, "o", color=VERMILLION, markersize=15, zorder=3)
        elif abs(xt) == 1:
            ax.plot(xt, xs, "s", color=BLUE, markersize=14, zorder=3)
        else:
            ax.plot(xt, xs, "o", markersize=12, markerfacecolor="white", markeredgecolor=GREY,
                    markeredgewidth=1.6, zorder=3)
        dx = 0.17 if xt >= 0 else -0.17
        ax.annotate(str(run), (xt, xs), xytext=(xt + dx, xs + 0.15), fontsize=11, color=GREY,
                    ha="left" if xt >= 0 else "right")

    # The axial pair: one comparison, 2 alpha apart, at x_S = 0.
    ax.annotate("", xy=(ALPHA - 0.14, 0), xytext=(-ALPHA + 0.14, 0),
                arrowprops={"arrowstyle": "<->", "color": VERMILLION, "linewidth": 2}, zorder=2)
    ax.text(0, -0.16, "axial slope:\nruns 16 to 14", color=VERMILLION, ha="center", va="top",
            fontsize=13)
    # The factorial runs: two comparisons, 2 units apart, at x_S = -1 and +1, averaged.
    for xs in (-1, 1):
        ax.annotate("", xy=(0.86, xs), xytext=(-0.86, xs),
                    arrowprops={"arrowstyle": "<->", "color": BLUE, "linewidth": 2})
    ax.text(0, 1.72, "factorial slope: average of\nruns 10 to 11 and 8 to 9", color=BLUE,
            ha="center", va="bottom", fontsize=13)
    ax.text(0, -1.72, "open circles: runs 6, 13 and 15,\nnot used for the slope of $T$", color=GREY,
            ha="center", va="top", fontsize=12)

    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.45, 2.45)
    ax.set_xticks([-ALPHA, -1, 0, 1, ALPHA], ["$-\\sqrt{2}$", "$-1$", "0", "$+1$", "$+\\sqrt{2}$"])
    ax.set_yticks([-ALPHA, -1, 0, 1, ALPHA], ["$-\\sqrt{2}$", "$-1$", "0", "$+1$", "$+\\sqrt{2}$"])
    ax.set_xlabel("Coded temperature, $x_T$")
    ax.set_ylabel("Coded substrate, $x_S$")
    ax.set_title("(a) Two measurements of the slope of $T$", loc="left")


def draw_profile(ax, response, title: str, note: str, show_runs: bool = False) -> None:
    """A panel of profit against coded temperature, with the two chords."""
    s_axial, s_factorial, axial, factorial = slopes(response)
    grid = np.linspace(-1.65, 1.65, 200)

    # The surface along the line the axial runs lie on (x_S = 0), and its
    # average over the two lines the factorial runs lie on (x_S = -1, +1).
    ax.plot(grid, response(grid, 0.0), color=VERMILLION, alpha=0.30, linewidth=2)
    ax.plot(grid, (response(grid, -1.0) + response(grid, 1.0)) / 2, color=BLUE, alpha=0.30, linewidth=2)

    if show_runs:
        for _, xt, xs, y in RUNS:
            if abs(xt) == 1:
                ax.plot(xt, y, "s", markersize=8, markerfacecolor="white", markeredgecolor=BLUE,
                        markeredgewidth=1.4, zorder=3)
        # The chords through the measured runs, not through the fitted surface.
        observed = {(xt, xs): y for _, xt, xs, y in RUNS}
        s_axial, s_factorial, axial, factorial = slopes(lambda t, s: observed[(t, s)])

    ax.plot([-ALPHA, ALPHA], axial, color=VERMILLION, linewidth=3, zorder=2,
            label=f"axial chord, slope {s_axial:.2f}")
    ax.plot([-ALPHA, ALPHA], axial, "o", color=VERMILLION, markersize=11, zorder=4)
    ax.plot([-1, 1], factorial, color=BLUE, linewidth=3, zorder=2,
            label=f"factorial chord, slope {s_factorial:.2f}")
    ax.plot([-1, 1], factorial, "s", color=BLUE, markersize=10, zorder=4)

    contrast = s_axial - s_factorial
    ax.text(0.03, 0.97, f"contrast = {contrast:+.2f}\n{note}", transform=ax.transAxes,
            ha="left", va="top", fontsize=13,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": LIGHT_GREY})
    ax.legend(loc="lower right", frameon=False)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_xticks([-ALPHA, -1, 0, 1, ALPHA], ["$-\\sqrt{2}$", "$-1$", "0", "$+1$", "$+\\sqrt{2}$"])
    ax.set_xlim(-1.75, 1.75)
    ax.set_xlabel("Coded temperature, $x_T$")
    ax.set_ylabel("Profit [$/day]")
    ax.set_title(title, loc="left")


def main(outdir: pathlib.Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 13))
    draw_design(axes[0, 0])
    draw_profile(axes[0, 1], surface, "(b) Bioreactor runs: the two slopes agree",
                 "the chords are close to parallel,\nas any quadratic surface makes them", show_runs=True)

    b_ttt = 6.0
    draw_profile(axes[1, 0], lambda t, s: surface(t, s, b_ttt=b_ttt),
                 f"(c) Add $b_{{TTT}}\\,x_T^3$, $b_{{TTT}} = {b_ttt:g}$",
                 "$= (\\alpha^2 - 1)\\,b_{TTT}$: the wider\naxial pair tilts more")
    b_tss = 8.0
    draw_profile(axes[1, 1], lambda t, s: surface(t, s, b_tss=b_tss),
                 f"(d) Add $b_{{TSS}}\\,x_T x_S^2$, $b_{{TSS}} = {b_tss:g}$",
                 "$= -\\,b_{TSS}$: only the factorial\nruns, at $x_S = \\pm 1$, tilt")

    fig.tight_layout(h_pad=2.5, w_pad=2.5)
    fig.savefig(outdir / "third-order-contrasts.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outdir / 'third-order-contrasts.png'}")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE)
