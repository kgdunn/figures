"""PCA figures for the food-texture example in the latent variable chapter.

Seven committed PNGs, replacing the base-R output of
``pca-on-food-texture-data.R`` and its companion
``pca-on-food-texture-combine-plots.py``:

- ``pca-on-food-texture-scatterplot-matrix.png``: the five quality
  attributes plotted against each other, the starting point of the example.
- ``pca-on-food-texture-centering-and-scaling.png``: box plots of the raw
  data, after centering, and after centering and scaling.
- ``pca-on-food-texture-pc1-loadings.png`` and
  ``pca-on-food-texture-pc2-loadings.png``: the first two loading vectors
  as bar plots.
- ``pca-on-food-texture-pc1-scores.png``: the first component's scores in
  sequence order.
- ``pca-on-food-texture-score-t1-contribution-for-obs-33.png``: what each
  variable contributes to the score of the most extreme pastry.
- ``pca-on-food-texture-scores-and-loadings.png``: the score plot and the
  loading plot side by side, which the chapter reads as a pair.

The model is a PCA on the autoscaled data (each column centered to zero
mean and scaled to unit variance, using the sample standard deviation),
computed by singular value decomposition. That is what ``prcomp(food,
scale=TRUE)`` did in the R. Loading signs are fixed so that oil loads
positively on the first component, matching the vectors printed in the
chapter; the sign of a loading vector is otherwise arbitrary.

Defects in the originals corrected here:

- The score plot marked sample 33 and sample 36 with nothing to
  distinguish them, although the text singles both out. They are marked.
- The contribution plot ran its y axis to -1.4 with no zero line labelled,
  and gave no indication of which variables push the score down. The bars
  are now coloured by sign.
- Points in the score plot overlapped their own numbers. Labels are placed
  to the side and the axes carry the percentage of variance explained.
- The scatterplot matrix had no axis labels along its diagonal beyond the
  variable name, and drew all 25 panels including the redundant upper
  triangle; the redundant half now carries the correlation instead.

Every number the chapter quotes is printed when this script runs.

Usage
-----
    uv run --with numpy --with pandas --with matplotlib python food_texture_pca_figures.py [output_dir]
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
BLUE_FILL = "#B3D4EA"
ORANGE = "#E69F00"
ORANGE_FILL = "#F5D28B"
VERMILLION = "#D55E00"
GREY = "#666666"
GRID = "#DDDDDD"

DPI = 300
HERE = pathlib.Path(__file__).parent
URL = "https://openmv.net/file/food-texture.csv"
VARIABLES = ["Oil", "Density", "Crispy", "Fracture", "Hardness"]

mpl.rcParams.update(
    {
        "font.size": 16,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 15,
        "axes.axisbelow": True,
    }
)


def save(fig, outdir: pathlib.Path, name: str) -> None:
    fig.savefig(outdir / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outdir / name}")


def food_texture() -> pd.DataFrame:
    """The 50 pastry samples, five quality attributes each."""
    try:
        data = pd.read_csv(URL)
    except Exception:  # noqa: BLE001 - the committed copy is the fallback
        data = pd.read_csv(HERE / "food-texture.csv")
    return data[VARIABLES] if set(VARIABLES) <= set(data.columns) else data.iloc[:, -5:]


def autoscale(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Center each column to zero mean and scale it to unit variance."""
    mean = data.mean()
    standard_deviation = data.std(ddof=1)
    return (data - mean) / standard_deviation, mean, standard_deviation


def principal_components(scaled: pd.DataFrame, n_components: int = 2):
    """Scores, loadings and the fraction of variance each component explains."""
    left, singular_values, right = np.linalg.svd(scaled.to_numpy(), full_matrices=False)
    loadings = right.T
    scores = left * singular_values
    # The sign of a component is arbitrary: anchor it so the first variable
    # loads positively on the first component, as the chapter prints it.
    for component in range(loadings.shape[1]):
        if loadings[0, component] < 0 and component == 0:
            loadings[:, component] *= -1
            scores[:, component] *= -1
    explained = singular_values**2 / np.sum(singular_values**2)
    return scores[:, :n_components], loadings[:, :n_components], explained[:n_components]


def scatterplot_matrix(data: pd.DataFrame, outdir: pathlib.Path) -> None:
    n = len(VARIABLES)
    fig, axes = plt.subplots(n, n, figsize=(13, 11))
    correlation = data.corr()
    for row in range(n):
        for column in range(n):
            ax = axes[row, column]
            ax.tick_params(labelsize=11)
            if row == column:
                ax.text(0.5, 0.5, VARIABLES[row], ha="center", va="center",
                        fontsize=17, transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                for side in ax.spines.values():
                    side.set_visible(False)
                continue
            if row < column:
                # The upper triangle repeats the lower one, so use it for the
                # correlation instead of drawing the same cloud twice.
                value = correlation.iloc[row, column]
                ax.text(0.5, 0.5, f"{value:+.2f}", ha="center", va="center",
                        fontsize=15 + 10 * abs(value),
                        color=BLUE if value > 0 else VERMILLION,
                        transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                for side in ax.spines.values():
                    side.set_visible(False)
                continue
            ax.scatter(data[VARIABLES[column]], data[VARIABLES[row]], s=18,
                       facecolor="none", edgecolor=BLUE, linewidth=1.1)
            ax.spines["top"].set_visible(True)
            ax.spines["right"].set_visible(True)
            if column != 0:
                ax.set_yticklabels([])
            if row != n - 1:
                ax.set_xticklabels([])
            if column == 0:
                ax.set_ylabel(VARIABLES[row], fontsize=14)
            if row == n - 1:
                ax.set_xlabel(VARIABLES[column], fontsize=14)
    fig.tight_layout()
    save(fig, outdir, "pca-on-food-texture-scatterplot-matrix.png")


def centering_and_scaling(data: pd.DataFrame, outdir: pathlib.Path) -> None:
    centered = data - data.mean()
    scaled, _, _ = autoscale(data)
    panels = [
        (data, "Raw data"),
        (centered, "After mean centering"),
        (scaled, "Centered and scaled to unit variance"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    for ax, (frame, title) in zip(axes, panels):
        boxes = ax.boxplot(frame.to_numpy(), tick_labels=VARIABLES, patch_artist=True,
                           medianprops={"color": VERMILLION, "linewidth": 2})
        for box in boxes["boxes"]:
            box.set_facecolor(BLUE_FILL)
            box.set_edgecolor(BLUE)
        ax.set_title(title, fontsize=16)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.tick_params(axis="x", labelrotation=90)
    fig.tight_layout()
    save(fig, outdir, "pca-on-food-texture-centering-and-scaling.png")


def loading_bars(loading: np.ndarray, component: int, limit: float,
                 outdir: pathlib.Path, name: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.bar(VARIABLES, loading,
           color=[BLUE_FILL if v > 0 else ORANGE_FILL for v in loading],
           edgecolor=[BLUE if v > 0 else ORANGE for v in loading], linewidth=1.6)
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_ylim(-limit, limit)
    ax.set_ylabel(f"{'1st' if component == 0 else '2nd'} component loadings")
    save(fig, outdir, name)


def score_sequence(scores: np.ndarray, outdir: pathlib.Path) -> None:
    order = np.arange(1, len(scores) + 1)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.grid(color=GRID, linewidth=0.8)
    ax.plot(order, scores[:, 0], "o", markersize=8, markerfacecolor="none",
            markeredgecolor=BLUE, markeredgewidth=1.6)
    ax.axhline(0, color="black", linewidth=1.0)
    for sample in (33, 36):
        value = scores[sample - 1, 0]
        ax.plot(sample, value, "o", markersize=9, color=VERMILLION)
        ax.annotate(f"{sample}", (sample, value), textcoords="offset points",
                    xytext=(8, -4), color=VERMILLION, fontsize=15)
    ax.set_ylim(-4.6, 4.6)
    ax.set_xlabel("Sequence order")
    ax.set_ylabel("1st component scores")
    save(fig, outdir, "pca-on-food-texture-pc1-scores.png")


def contribution_plot(scaled: pd.DataFrame, loadings: np.ndarray,
                      outdir: pathlib.Path, sample: int = 33) -> None:
    contributions = scaled.iloc[sample - 1].to_numpy() * loadings[:, 0]
    print(f"observation {sample} autoscaled: "
          + ", ".join(f"{v:+.3f}" for v in scaled.iloc[sample - 1]))
    print(f"observation {sample} contributions to t1: "
          + ", ".join(f"{n} {v:+.3f}" for n, v in zip(VARIABLES, contributions))
          + f"; sum {contributions.sum():+.3f}")

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.bar(VARIABLES, contributions,
           color=[BLUE_FILL if v > 0 else ORANGE_FILL for v in contributions],
           edgecolor=[BLUE if v > 0 else ORANGE for v in contributions], linewidth=1.6)
    ax.axhline(0, color="black", linewidth=1.0)
    ax.set_ylabel(f"Contribution to $t_1$,\nobservation {sample}")
    save(fig, outdir, "pca-on-food-texture-score-t1-contribution-for-obs-33.png")


def scores_and_loadings(scores: np.ndarray, loadings: np.ndarray,
                        explained: np.ndarray, outdir: pathlib.Path) -> None:
    first, second = (100 * explained[:2]).round().astype(int)
    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    ax = axes[0]
    ax.grid(color=GRID, linewidth=0.8)
    ax.axhline(0, color="black", linewidth=1.0)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.plot(scores[:, 0], scores[:, 1], "o", markersize=9, color=BLUE)
    for index, (x, y) in enumerate(scores[:, :2], start=1):
        ax.annotate(str(index), (x, y), textcoords="offset points", xytext=(7, 3),
                    fontsize=12, color=GREY)
    ax.set_xlabel(f"1st component scores [{first}%]")
    ax.set_ylabel(f"2nd component scores [{second}%]")

    ax = axes[1]
    ax.grid(color=GRID, linewidth=0.8)
    ax.axhline(0, color="black", linewidth=1.0)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.plot(loadings[:, 0], loadings[:, 1], "o", markersize=11, color=VERMILLION)
    for name, (x, y) in zip(VARIABLES, loadings[:, :2]):
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(10, 4), fontsize=16)
    ax.set_xlim(-0.75, 0.75)
    ax.set_ylim(-0.55, 1.0)
    ax.set_xlabel(f"1st component loadings [{first}%]")
    ax.set_ylabel(f"2nd component loadings [{second}%]")

    fig.tight_layout()
    save(fig, outdir, "pca-on-food-texture-scores-and-loadings.png")


def main(outdir: pathlib.Path) -> None:
    data = food_texture()
    scaled, mean, standard_deviation = autoscale(data)
    scores, loadings, explained = principal_components(scaled)

    print(f"{len(data)} observations of {data.shape[1]} variables")
    for name in VARIABLES:
        print(f"  {name:9s} mean {mean[name]:8.2f}  standard deviation {standard_deviation[name]:7.2f}")
    print("p1 = " + ", ".join(f"{v:+.2f}" for v in loadings[:, 0]))
    print("p2 = " + ", ".join(f"{v:+.2f}" for v in loadings[:, 1]))
    print(f"variance explained: {100 * explained[0]:.1f}% and {100 * explained[1]:.1f}%")
    print(f"t1 for sample 33 = {scores[32, 0]:.2f}; t1 for sample 36 = {scores[35, 0]:.2f}")

    scatterplot_matrix(data, outdir)
    centering_and_scaling(data, outdir)
    loading_bars(loadings[:, 0], 0, 0.7, outdir, "pca-on-food-texture-pc1-loadings.png")
    loading_bars(loadings[:, 1], 1, 1.0, outdir, "pca-on-food-texture-pc2-loadings.png")
    score_sequence(scores, outdir)
    contribution_plot(scaled, loadings, outdir)
    scores_and_loadings(scores, loadings, explained, outdir)


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE)
