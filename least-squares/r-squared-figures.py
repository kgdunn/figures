"""Two figures for the R-squared subsections of the least-squares chapter.

- ``r-squared-symmetry.png``: the 11-point example used throughout the
  chapter, with both least-squares lines drawn on the same axes. The
  blue line predicts y from x and minimises the vertical distances; the
  vermillion line predicts x from y and minimises the horizontal ones.
  The two lines have different slopes (0.500 and 0.750 when the second
  is rewritten in these axes) yet report the same R-squared of 0.667.

- ``r-squared-versus-standard-error.png``: four simulated data sets from
  the same model, y = 5 + x + e, drawn on common axes. The rows differ
  in the size of the error, S_E = 1 kg and S_E = 2 kg; the columns
  differ in how widely x was sampled, over the full range or over the
  middle half of it. Each row therefore holds the prediction error
  fixed while R-squared moves, and the two panels on the anti-diagonal
  report the same R-squared of 0.90 with a two-fold difference in
  prediction error. The errors are rescaled after they are drawn so
  that the realised S_E is exactly 1.000 kg and 2.000 kg rather than
  approximately so: least squares residuals are linear in the errors,
  so scaling the errors by a constant scales S_E by the same constant.

Both figures use a fixed seed, so the numbers annotated on the panels
are the same ones the plotly code in the chapter produces.

Usage
-----
    uv run --with numpy --with matplotlib python r-squared-figures.py [output_dir]
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

DPI = 300
SEED = 225
HERE = pathlib.Path(__file__).parent

# The 11-point example carried through the least-squares chapter.
X_EXAMPLE = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5], dtype=float)
Y_EXAMPLE = np.array(
    [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
)

mpl.rcParams.update(
    {
        "font.size": 15,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.axisbelow": True,
    }
)


def fit(x, y):
    """Least squares intercept, slope and R-squared for y on x."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    slope = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
    intercept = y.mean() - slope * x.mean()
    residuals = y - (intercept + slope * x)
    r2 = 1 - np.sum(residuals**2) / np.sum((y - y.mean()) ** 2)
    standard_error = np.sqrt(np.sum(residuals**2) / (len(x) - 2))
    return intercept, slope, r2, standard_error


def symmetry_figure(outdir: pathlib.Path, filename: str) -> None:
    """Both regression lines on one set of axes, with the same R-squared."""
    x, y = X_EXAMPLE, Y_EXAMPLE
    b0, b1, r2, _ = fit(x, y)
    # Predicting x from y gives a line in these same axes once it is
    # rearranged from x = a0 + a1*y into y = -a0/a1 + (1/a1)*x.
    a0, a1, r2_reverse, _ = fit(y, x)
    c0, c1 = -a0 / a1, 1.0 / a1

    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    grid = np.linspace(2.5, 15.5, 200)

    # Residuals each line minimises: vertical for one, horizontal for the other.
    for xi, yi in zip(x, y):
        ax.plot([xi, xi], [yi, b0 + b1 * xi], color=BLUE, linewidth=1.8, alpha=0.55)
        ax.plot(
            [xi, a0 + a1 * yi], [yi, yi], color=VERMILLION, linewidth=1.8, alpha=0.55
        )

    ax.plot(grid, b0 + b1 * grid, color=BLUE, linewidth=3.6,
            label=f"Predicting y from x: slope {b1:.3f}")
    ax.plot(grid, c0 + c1 * grid, color=VERMILLION, linewidth=3.6, linestyle="--",
            label=f"Predicting x from y: slope {c1:.3f} in these axes")
    ax.plot(x, y, "o", color="black", markersize=11, zorder=5)
    ax.plot(x.mean(), y.mean(), "P", color=GREY, markersize=18, zorder=6,
            label="Mean of the data")

    ax.text(
        0.03, 0.97,
        f"Both models: $R^2$ = {r2:.4f}",
        transform=ax.transAxes, va="top", ha="left", fontsize=17,
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(2.5, 15.5)
    ax.set_ylim(3.0, 11.5)
    ax.grid(True, color="#DDDDDD")
    ax.legend(loc="lower right", fontsize=13, framealpha=1.0)
    fig.tight_layout()
    fig.savefig(outdir / filename, dpi=DPI)
    plt.close(fig)
    print(f"{filename}: R2 forward {r2:.4f}, R2 reverse {r2_reverse:.4f}, "
          f"slopes {b1:.3f} and {c1:.3f}")


def r2_versus_se_figure(outdir: pathlib.Path, filename: str) -> None:
    """Four data sets from one model: R-squared moves, S_E stays or not."""
    rng = np.random.default_rng(SEED)
    n = 40
    # Sampled over the full range of x, or over its middle half: the same
    # underlying relationship, but half the spread in x.
    spreads = {"x sampled over 0 to 20": (0.0, 20.0),
               "x sampled over 5 to 15": (5.0, 15.0)}
    targets = [1.0, 2.0]

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.5), sharex=True, sharey=True)
    for row, target in enumerate(targets):
        for col, (title, (low, high)) in enumerate(spreads.items()):
            ax = axes[row][col]
            x = rng.uniform(low, high, n)
            errors = rng.normal(0.0, 1.0, n)

            # Rescale the errors so the realised standard error is exactly
            # the target. Least squares residuals are linear in the errors,
            # so scaling the errors scales S_E by the same factor.
            se_draw = fit(x, 5.0 + 1.0 * x + errors)[3]
            y = 5.0 + 1.0 * x + errors * (target / se_draw)
            b0, b1, r2, se = fit(x, y)

            grid = np.linspace(low, high, 100)
            ax.plot(x, y, "o", color=BLUE, markersize=6, alpha=0.85)
            ax.plot(grid, b0 + b1 * grid, color=VERMILLION, linewidth=2.2)
            ax.text(
                0.04, 0.96,
                f"$R^2$ = {r2:.2f}\n$S_E$ = {se:.1f} kg\nslope = {b1:.2f} kg/unit",
                transform=ax.transAxes, va="top", ha="left", fontsize=13,
            )
            # Shade the two panels that report the same R-squared, so the
            # comparison the figure is making is the one the eye makes.
            same_r2 = (row, col) in {(0, 1), (1, 0)}
            ax.set_facecolor("#EFEFEF" if same_r2 else "white")
            ax.grid(True, color="#CCCCCC" if same_r2 else "#DDDDDD")
            if row == 0:
                ax.set_title(title, fontsize=14)
            if row == 1:
                ax.set_xlabel("x")
            if col == 0:
                ax.set_ylabel("y [kg]")
            print(f"{filename}: target S_E {target}, {title}: "
                  f"R2 {r2:.3f}, S_E {se:.4f}, slope {b1:.3f}")

    axes[0][0].set_xlim(-1.0, 21.0)
    axes[0][0].set_ylim(-1.0, 31.0)
    fig.tight_layout()
    fig.savefig(outdir / filename, dpi=DPI)
    plt.close(fig)


def main(outdir: pathlib.Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    symmetry_figure(outdir, "r-squared-symmetry.png")
    r2_versus_se_figure(outdir, "r-squared-versus-standard-error.png")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE)
