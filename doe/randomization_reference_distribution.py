"""Fisher's randomization reference distribution for the two-reactor example.

Writes ``single-experiment-randomization.png``, the figure used in the
designed-experiments chapter to show why experiments are run in random
order. Eight brittleness values from reactor TK104 (case A) and nine from
TK107 (case B) are pooled, and every one of the

    (8 + 9)! / (8! 9!) = 24310

ways of splitting those seventeen values into a group of eight and a group
of nine is enumerated. For each split the difference of the group averages
is recorded, which gives the reference distribution the actual experiment
is judged against.

This replaces ``single-experiment-randomization.py``, which was Python 2
(``xrange``, ``print`` statements, ``scipy.misc.factorial``), wrote a file
under a different name than the chapter used, and produced a plot with no
axis labels. Two things are added here, both of them noted as wanted in the
chapter source:

- the axes are labelled, and the sign convention is stated: the plotted
  difference is ybar_B - ybar_A, matching the vertical line;
- the t-distribution the classical test uses is scaled onto the histogram,
  so the reader can see how closely the two agree.

Every number the chapter quotes is printed when this script runs.

Usage
-----
    uv run --with numpy --with scipy --with matplotlib python randomization_reference_distribution.py [output_dir]
"""

from __future__ import annotations

import itertools
import math
import pathlib
import sys

import matplotlib as mpl

mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as t_distribution

BLUE = "#0072B2"
BLUE_FILL = "#B3D4EA"
VERMILLION = "#D55E00"
GREY = "#666666"
GRID = "#DDDDDD"

DPI = 300
HERE = pathlib.Path(__file__).parent

CASE_A = np.array([254, 440, 501, 368, 697, 476, 188, 525])
CASE_B = np.array([338, 470, 558, 426, 733, 539, 240, 628, 517])

mpl.rcParams.update(
    {
        "font.size": 16,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 18,
        "axes.labelsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 14,
        "axes.axisbelow": True,
    }
)


def reference_distribution(case_a: np.ndarray, case_b: np.ndarray) -> np.ndarray:
    """Every possible value of ybar_B - ybar_A under random assignment."""
    outcomes = np.concatenate((case_a, case_b))
    n_a, total = len(case_a), len(outcomes)
    grand_total = outcomes.sum()
    differences = []
    for subset in itertools.combinations(range(total), n_a):
        sum_a = outcomes[list(subset)].sum()
        differences.append((grand_total - sum_a) / (total - n_a) - sum_a / n_a)
    return np.array(differences)


def pooled_standard_error(case_a: np.ndarray, case_b: np.ndarray) -> float:
    """Standard error of the difference in averages, pooling the two variances."""
    n_a, n_b = len(case_a), len(case_b)
    pooled_variance = (
        (n_a - 1) * case_a.var(ddof=1) + (n_b - 1) * case_b.var(ddof=1)
    ) / (n_a + n_b - 2)
    return float(np.sqrt(pooled_variance * (1 / n_a + 1 / n_b)))


def main(outdir: pathlib.Path) -> None:
    differences = reference_distribution(CASE_A, CASE_B)
    observed = CASE_B.mean() - CASE_A.mean()
    n_a, n_b = len(CASE_A), len(CASE_B)
    n_splits = math.comb(n_a + n_b, n_a)
    greater = int((differences > observed).sum())

    standard_error = pooled_standard_error(CASE_A, CASE_B)
    z_value = observed / standard_error
    degrees_of_freedom = n_a + n_b - 2
    classical = t_distribution.cdf(z_value, degrees_of_freedom)

    print(f"splits enumerated: {len(differences)} (expected {n_splits})")
    print(f"average of case A = {CASE_A.mean():.3f}, case B = {CASE_B.mean():.3f}")
    print(f"observed difference (B - A) = {observed:.4f}")
    print(f"splits with a larger difference: {greater}")
    print(f"fraction of splits at or below the observed difference: "
          f"{100 * (1 - greater / len(differences)):.1f}%")
    print(f"pooled standard error = {standard_error:.4f}, z = {z_value:.4f}")
    print(f"t-distribution with {degrees_of_freedom} degrees of freedom: {100 * classical:.1f}%")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    counts, edges, _ = ax.hist(differences, bins=50, color=BLUE_FILL, edgecolor=BLUE,
                               linewidth=0.8)
    width = edges[1] - edges[0]

    grid = np.linspace(edges[0], edges[-1], 400)
    density = t_distribution.pdf(grid / standard_error, degrees_of_freedom) / standard_error
    ax.plot(grid, density * len(differences) * width, color=GREY, linewidth=2,
            label=f"$t$-distribution, {degrees_of_freedom} degrees of freedom")
    ax.axvline(observed, color=VERMILLION, linewidth=2.5,
               label=f"Observed difference = {observed:.1f}")

    ax.set_xlabel(r"Difference in averages, $\overline{y}_B - \overline{y}_A$")
    ax.set_ylabel(f"Number of the {len(differences):,} splits")
    ax.legend(frameon=False, loc="upper left")
    ax.annotate(
        f"{greater} splits ({100 * greater / len(differences):.1f}%)\nexceed the observed value",
        xy=(observed + 4, counts.max() * 0.42), color=VERMILLION, fontsize=15, va="center",
    )
    fig.tight_layout()
    fig.savefig(outdir / "single-experiment-randomization.png", dpi=DPI)
    plt.close(fig)
    print(f"wrote {outdir / 'single-experiment-randomization.png'}")


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE)
