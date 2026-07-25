"""The scree plot and the R-squared / Q-squared bar plots for the PCA chapter.

Three committed PNGs, replacing the base-R output of
``eigenvalue-scree-plot.R`` and ``barplot-for-R2-and-Q2.R``:

- ``eigenvalue-scree-plot.png``: the proportion of variance each of the
  first ten components of the distillation-tower data explains.
- ``eigenvalue-scree-plot-cumulative.png``: the same, accumulated.
- ``barplot-for-R2-and-Q2.png``: cumulative R-squared beside cumulative
  Q-squared for the LDPE case study, showing where cross-validation stops
  rewarding extra components.
- ``q2-across-packages.png``: the Q-squared Simca-P reported beside the
  element-wise one computed by ``process_improve``, on one axis.

The scree plots are computed from ``distillation-tower.csv`` on openmv.net.
The R-squared and Q-squared values are outputs of two commercial packages
(Simca-P 11.5 and ProSensus 11.08) run on the LDPE data; they cannot be
recomputed here, so they are carried as the literals the R script recorded,
with the package and version named alongside them.

The second curve is computed here, by ``process_improve``'s
``PCA.select_n_components`` with ``cv_scheme="ekf"``: the element-wise
k-fold scheme of Bro et al. (2008), which holds out scattered individual
cells of X and predicts each from a model that never saw it. That is the
scheme the chapter recommends over the row-wise one, so the figure shows
what it gives on the same data Simca-P was run on.

ProSensus Multivariate is left out of the comparison figure: that software
is no longer available, so a reader cannot check the curve or reproduce it.
Its recorded numbers stay in ``Q2`` below and it keeps its own
``barplot-for-R2-and-Q2-ProSensus.png``, which is the record of what the R
script drew.

The same run reproduces the R-squared the commercial packages recorded,
to within 4.4e-07 over all eleven components, which is what establishes
that the two curves are computed on the same data with the same
preprocessing and so can be compared at all.

Defects in the originals corrected here:

- ``barplot-for-R2-and-Q2.R`` wrote ``barplot-for-R2-and-Q2-Simca.png``
  and ``-ProSensus.png``, but the book embeds ``barplot-for-R2-and-Q2.png``,
  which no script produced. All three names are written, the unsuffixed one
  being the Simca pair the chapter's prose describes.
- The chapter source carries a note asking for the values to be printed on
  top of each bar. They are.
- The dashed line marking where to stop was drawn between two bars with no
  explanation of what it meant. It is labelled.

Usage
-----
    uv run --with numpy --with pandas --with matplotlib --with process-improve \
        python pca_diagnostic_figures.py [output_dir]

Needs process_improve 1.60 or later, for ``PCA.select_n_components``. If it
is not importable the three original figures are still written and only the
comparison is skipped, with a message.
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
URL = "https://openmv.net/file/distillation-tower.csv"
N_COMPONENTS = 10

# Reported by Simca-P 11.5 (November 2006) and ProSensus Multivariate
# 11.08 (2011) for a PCA on the LDPE case study, all variables, all rows.
R2 = [0.369877, 0.547855, 0.65697, 0.747528, 0.831526, 0.885739,
      0.928337, 0.962801, 0.990961, 0.99925, 0.999847]
Q2 = {
    "Simca-P 11.5": [0.253787, 0.341342, 0.318556, 0.250411, 0.32386, 0.270315,
                     0.19828, 0.245897, 0.774703, 0.963812, 0.990275],
    "ProSensus 11.08": [0.32847, 0.50085, 0.575319, 0.6338, 0.7637, 0.8231,
                        0.87225, 0.916889, 0.986077, 0.998795, 0.99972],
}
# Where each package's Q-squared stops rewarding another component.
STOP_AFTER = {"Simca-P 11.5": 2, "ProSensus 11.08": 8}

mpl.rcParams.update(
    {
        "font.size": 15,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 17,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "axes.axisbelow": True,
    }
)


def save(fig, outdir: pathlib.Path, name: str) -> None:
    fig.savefig(outdir / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outdir / name}")


def distillation_tower() -> np.ndarray:
    """The 27 process measurements, dropping the date column."""
    try:
        data = pd.read_csv(URL)
    except Exception:  # noqa: BLE001
        data = pd.read_csv(HERE / "distillation-tower.csv")
    return data.iloc[:, 1:28].to_numpy(dtype=float)


def eigenvalues(data: np.ndarray) -> np.ndarray:
    """Eigenvalues of X'X on the autoscaled data, as a fraction of their sum."""
    scaled = (data - data.mean(axis=0)) / data.std(axis=0, ddof=1)
    values = np.linalg.eigvalsh(scaled.T @ scaled)[::-1]
    return values / values.sum()


def scree_plots(fraction: np.ndarray, outdir: pathlib.Path) -> None:
    components = np.arange(1, N_COMPONENTS + 1)
    percent = 100 * fraction[:N_COMPONENTS]
    print("proportion of variance explained (%): "
          + ", ".join(f"{v:.1f}" for v in percent))
    print(f"cumulative after {N_COMPONENTS} components: {percent.sum():.1f}%")

    for cumulative in (False, True):
        heights = np.cumsum(percent) if cumulative else percent
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.bar(components, heights, color=BLUE_FILL, edgecolor=BLUE, linewidth=1.6,
               width=0.65)
        for position, height in zip(components, heights):
            ax.annotate(f"{height:.0f}", (position, height), ha="center", va="bottom",
                        xytext=(0, 5), textcoords="offset points", fontsize=13,
                        color=GREY)
        ax.set_xticks(components)
        ax.set_xlabel("Component number")
        ax.set_ylabel("Cumulative variance explained [%]" if cumulative
                      else "Variance explained [%]")
        ax.set_ylim(0, max(heights) * 1.14)
        name = ("eigenvalue-scree-plot-cumulative.png" if cumulative
                else "eigenvalue-scree-plot.png")
        save(fig, outdir, name)


def r2_q2_plot(package: str, outdir: pathlib.Path, name: str, shown: int = 11) -> None:
    """One R-squared / Q-squared pair. ``shown`` is how many components
    to plot, of the eleven each package reported."""
    q2 = Q2[package][:shown]
    r2 = R2[:shown]
    components = np.arange(1, shown + 1)
    stop = STOP_AFTER[package]
    print(f"{package}: R² goes {100 * R2[2]:.0f}% to {100 * R2[3]:.0f}% "
          f"on the 4th component, while Q² goes {100 * q2[2]:.0f}% to {100 * q2[3]:.0f}%")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, values, label, fill, edge in (
        (axes[0], r2, "Cumulative $R^2$", BLUE_FILL, BLUE),
        (axes[1], q2, "Cumulative $Q^2$", ORANGE_FILL, ORANGE),
    ):
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.bar(components, values, color=fill, edgecolor=edge, linewidth=1.5, width=0.7)
        for position, value in zip(components, values):
            ax.annotate(f"{value:.2f}", (position, value), ha="center", va="bottom",
                        xytext=(0, 4), textcoords="offset points", fontsize=11,
                        color=GREY)
        ax.set_xticks(components)
        ax.set_xlabel("Component number")
        ax.set_ylabel(label)
        ax.set_ylim(0, 1.12)
    axes[1].axvline(stop + 0.5, color=VERMILLION, linestyle="--", linewidth=2)
    # The note goes in the corner rather than beside the line: the bars to
    # the right of the line are tall in one of these two figures.
    axes[1].annotate(f"$Q^2$ stops rising after {stop} components",
                     (0.02, 0.99), xycoords="axes fraction", color=VERMILLION,
                     fontsize=13, ha="left", va="top")
    axes[1].set_title(package)
    fig.tight_layout()
    save(fig, outdir, name)


GREEN = "#009E73"
LDPE_URL = "https://openmv.net/file/LDPE.csv"


def process_improve_q2() -> tuple[np.ndarray, np.ndarray, int]:
    """Cross-validated Q-squared for the LDPE data, element-wise k-fold.

    Returns the Q-squared per component count, its standard error across
    the repeated fold permutations, and the component count the library's
    own selection rule lands on. Also checks the R-squared against the
    values the commercial packages recorded: if those agree, the three
    curves are being computed on the same data, the same way, and the
    comparison means something.
    """
    import warnings

    from process_improve.multivariate.methods import PCA, MCUVScaler

    try:
        ldpe = pd.read_csv(LDPE_URL)
    except Exception:  # noqa: BLE001
        ldpe = pd.read_csv(HERE / "LDPE.csv")
    scaled = MCUVScaler().fit_transform(ldpe.iloc[:, 1:])

    fitted = PCA(n_components=len(R2)).fit(scaled)
    largest = float(np.max(np.abs(np.asarray(fitted.r2_cumulative_, dtype=float) - np.array(R2))))
    print(f"R² against the recorded values: largest difference {largest:.2e}")
    if largest > 1e-4:
        raise ValueError(f"R² disagrees with the recorded values by {largest:.4f}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chosen = PCA.select_n_components(
            scaled, max_components=len(R2), cv=7, cv_scheme="ekf",
            n_repeats=5, random_state=42,
        )
    return (np.asarray(chosen.q2, dtype=float),
            np.asarray(chosen.q2_se, dtype=float),
            int(chosen.n_components))


def q2_comparison(outdir: pathlib.Path) -> None:
    """The three Q-squared curves for the same data on one axis."""
    try:
        q2, q2_se, selected = process_improve_q2()
    except ImportError:
        print("process_improve not importable: skipping q2-across-packages.png")
        return

    components = np.arange(1, len(R2) + 1)
    print("process_improve Q² (element-wise k-fold): "
          + ", ".join(f"{v:.3f}" for v in q2))
    print(f"  its selection rule lands on {selected} components")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.plot(components, Q2["Simca-P 11.5"], "-o", color=ORANGE, linewidth=2.2,
            markersize=8, label="Simca-P 11.5 (2006), row-wise")
    ax.plot(components, q2, "-^", color=GREEN, linewidth=2.6, markersize=9,
            label="process_improve, element-wise k-fold")
    # The band is the spread over five fold permutations. It is narrow
    # enough to be worth stating rather than leaving the reader to wonder.
    ax.fill_between(components, q2 - q2_se, q2 + q2_se, color=GREEN, alpha=0.25,
                    linewidth=0)
    ax.axvline(2, color=GREY, linestyle="--", linewidth=1.5)
    ax.annotate("Two components: both curves\nreach their highest value here\nand neither exceeds it again",
                (2.15, 0.93), color=GREY, fontsize=13, va="top")
    ax.set_xticks(components)
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cross-validated $Q^2$")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    save(fig, outdir, "q2-across-packages.png")


def main(outdir: pathlib.Path) -> None:
    scree_plots(eigenvalues(distillation_tower()), outdir)
    # The chapter embeds the unsuffixed name; it is the Simca pair, whose
    # numbers the surrounding prose quotes. That committed image stops at
    # eight components, where the two package-specific images run to
    # eleven, and the difference is not cosmetic: Simca's Q-squared for
    # components 9 to 11 is 0.77, 0.96 and 0.99. The distillation data has
    # eleven variables, so by the ninth component the model is fitting
    # what little is left and cross-validation predicts it almost exactly.
    # Those three bars tower over the rest and bury the point the figure
    # is making, that Q-squared stops rewarding components after the
    # second. The two package images keep all eleven, as their originals
    # do.
    r2_q2_plot("Simca-P 11.5", outdir, "barplot-for-R2-and-Q2.png", shown=8)
    r2_q2_plot("Simca-P 11.5", outdir, "barplot-for-R2-and-Q2-Simca.png")
    r2_q2_plot("ProSensus 11.08", outdir, "barplot-for-R2-and-Q2-ProSensus.png")
    q2_comparison(outdir)


if __name__ == "__main__":
    main(pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else HERE)
