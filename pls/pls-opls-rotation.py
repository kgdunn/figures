"""Generate the committed PNG showing the same solution line in both coordinate systems.

Fifth figure in the ``pls/`` set for the pid-book section on model inversion and
the orthogonal space, in
``latent-variable-modelling/projection-to-latent-structures/pls-model-inversion-and-the-orthogonal-space.rst``.

PLS spreads the response across both components, so the set of scores predicting
a chosen taste is a diagonal line, and reaching it means solving one equation in
two unknowns. O-PLS puts the whole response on its predictive component, so the
same set of scores becomes a line at a fixed predictive score, parallel to the
orthogonal axis. Reaching it is then a division.

Both panels show the same 26 cheeses and the same set of solutions. Only the axes
have moved: the second panel is a rotation of the first. Drawing the pair says
what several paragraphs of prose otherwise have to.

Requires the ``process_improve`` package (``pip install process-improve``) for
``PLS`` and ``OPLS``.

Usage::

    python pls/pls-opls-rotation.py [output_dir]

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

from process_improve.multivariate import OPLS, PLS

DATA_URL = "https://openmv.net/file/cheddar-cheese.csv"
X_COLUMNS = ["Acetic", "H2S", "Lactic"]
TARGET_TASTE = 20.9

DARK_BLUE = "#1f3d7a"  # calibration cheeses
ORANGE = "#e6820a"     # the set of solutions, and the design on it
MAROON = "#7b1d2b"     # the gradient of the prediction
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
    """Fit both models and draw the same solution line in each coordinate system."""
    cheese = pd.read_csv(DATA_URL)
    train = cheese.iloc[4:]  # cheeses 5 to 30, as in the chapter
    x_block, y_block = train[X_COLUMNS], train[["Taste"]]

    pls = PLS(n_components=2).fit(x_block, y_block)
    opls = OPLS(n_orthogonal_components=1).fit(x_block, y_block)

    result = pls.invert(y_desired=TARGET_TASTE)
    tau = result.scores.to_numpy()
    g = result.null_space_basis.to_numpy().ravel()
    q = pls.y_loadings_.to_numpy().ravel()

    t_p = opls.predictive_scores_.to_numpy().ravel()
    t_o = opls.orthogonal_scores_.to_numpy().ravel()
    q_p = float(opls.y_loadings_)
    t_p_target = opls.invert(y_desired=TARGET_TASTE).predictive_score

    fig, (ax_pls, ax_opls) = plt.subplots(1, 2, figsize=(12.2, 5.8))
    limits = (-3.4, 3.4)
    span = np.linspace(-6, 6, 2)
    arrow = 2.3

    # --- left: PLS coordinates, the solutions run diagonally ------------------
    scores = pls.scores_.to_numpy()
    line = np.array([tau + s * g for s in span])
    ax_pls.plot(line[:, 0], line[:, 1], color=ORANGE, lw=2.6, zorder=3,
                label=f"Predicted taste = {TARGET_TASTE}")
    ax_pls.scatter(scores[:, 0], scores[:, 1], s=26, color=DARK_BLUE, alpha=0.6, lw=0,
                   zorder=4, label="Calibration cheeses")
    ax_pls.scatter(*tau, color=ORANGE, marker="s", s=95, edgecolors=BLACK, linewidths=0.9,
                   zorder=6, label="Direct-inversion solution")
    unit_q = q / np.linalg.norm(q)
    ax_pls.annotate("", xy=tuple(unit_q * arrow), xytext=(0, 0), zorder=5,
                    arrowprops={"arrowstyle": "-|>", "color": MAROON, "lw": 2.4,
                                "mutation_scale": 18})
    ax_pls.annotate(r"$\mathbf{q}$", xy=tuple(unit_q * arrow), xytext=(9, -4),
                    textcoords="offset points", color=MAROON, fontsize=13, fontweight="bold")
    ax_pls.set_xlabel("$t_1$")
    ax_pls.set_ylabel("$t_2$")
    ax_pls.set_title("PLS: the response is spread over both components", fontsize=11.5)
    ax_pls.annotate("one equation,\ntwo unknowns", xy=(0.03, 0.03), xycoords="axes fraction",
                    fontsize=9.5, color=GREY, ha="left", va="bottom")

    # --- right: O-PLS coordinates, the same solutions along an axis -----------
    ax_opls.axvline(t_p_target, color=ORANGE, lw=2.6, zorder=3,
                    label=f"Predicted taste = {TARGET_TASTE}")
    ax_opls.scatter(t_p, t_o, s=26, color=DARK_BLUE, alpha=0.6, lw=0, zorder=4,
                    label="Calibration cheeses")
    ax_opls.scatter(t_p_target, 0.0, color=ORANGE, marker="s", s=95, edgecolors=BLACK,
                    linewidths=0.9, zorder=6, label="O-PLS solution")
    ax_opls.annotate("", xy=(arrow, 0), xytext=(0, 0), zorder=5,
                     arrowprops={"arrowstyle": "-|>", "color": MAROON, "lw": 2.4,
                                 "mutation_scale": 18})
    ax_opls.annotate(rf"$q_\mathrm{{p}} = {q_p:.3f}$", xy=(arrow * 0.5, 0), xytext=(0, -22),
                     textcoords="offset points", color=MAROON, fontsize=11,
                     fontweight="bold", ha="center", va="top",
                     bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85,
                           "boxstyle": "round,pad=0.15"})
    ax_opls.set_xlabel(r"$t_\mathrm{p}$, the predictive score")
    ax_opls.set_ylabel(r"$t_\mathrm{o}$, the orthogonal score")
    ax_opls.set_title("O-PLS: the response is all on one component", fontsize=11.5)
    ax_opls.annotate(f"one unknown:\n$t_p = {t_p_target:.3f}$", xy=(0.03, 0.03),
                     xycoords="axes fraction", fontsize=9.5, color=GREY,
                     ha="left", va="bottom")

    for ax in (ax_pls, ax_opls):
        ax.axhline(0, color="0.8", lw=0.8, zorder=0)
        ax.axvline(0, color="0.8", lw=0.8, zorder=0)
        ax.set_xlim(*limits)
        ax.set_ylim(*limits)
        ax.set_aspect("equal")
        ax.legend(loc="upper left", fontsize=8.5, framealpha=0.93)

    fig.tight_layout()
    fig.savefig(out_dir / "pls-opls-rotation.png")
    plt.close(fig)


def main() -> None:
    """Entry point: render into the given directory (default: this script's own)."""
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
    build_figure(out_dir)


if __name__ == "__main__":
    main()
