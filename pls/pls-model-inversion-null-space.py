"""Generate the committed PNG for the PLS model-inversion / null-space figure.

Mirrors the worked example in the pid-book page
``latent-variable-modelling/projection-to-latent-structures/pls-model-inversion-and-the-orthogonal-space.rst``:
a two-component PLS model of the cheddar-cheese data (trained on cheeses 5 to
30), inverted toward a target taste of 20.9. The score plot shows the
calibration cheeses, the direct-inversion solution, the null space (the line of
scores that all predict the target taste), and the O-PLS orthogonal space
projected into the PLS score space. The null space and the projected orthogonal
space coincide, which is the point of the page and of Garcia-Carrion et al.
(2025).

The chapter shows the equivalent Plotly code; this committed figure is the
matplotlib rendering.

Requires the ``process_improve`` package (``pip install process-improve``) for
``PLS`` and ``OPLS``.

Usage::

    python pls/pls-model-inversion-null-space.py [output_dir]

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
PURPLE = "#6a3d9a"     # the null space (PLS)
GREEN = "#2e6f3e"      # the orthogonal space (O-PLS), projected into PLS space
BLACK = "#111111"      # the direct-inversion solution
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "figure.dpi": 140,
    }
)


def build_figure(out_dir: Path) -> None:
    """Fit the models, invert, and render the score plot."""
    cheese = pd.read_csv(DATA_URL)
    train = cheese.iloc[4:]  # cheeses 5 to 30
    x_train = train[X_COLUMNS]
    y_train = train[["Taste"]]

    pls = PLS(n_components=2).fit(x_train, y_train)
    opls = OPLS(n_orthogonal_components=1).fit(x_train, y_train)

    result = pls.invert(y_desired=TARGET_TASTE)
    tau = result.scores.to_numpy()
    g_ns = result.null_space_basis.to_numpy().ravel()

    # Null-space line in the PLS score space.
    steps = np.linspace(-4.0, 4.0, 60)
    ns_line = np.array([tau + s * g_ns for s in steps])

    # O-PLS orthogonal space: walk along the orthogonal loading in the input
    # space (scaled), starting from the O-PLS design, then project into the PLS
    # score space with the direct weights. These points should land on the NS.
    opls_result = opls.invert(y_desired=TARGET_TASTE)
    x_scaler = pls._x_scaler  # noqa: SLF001 - reuse the fitted scaling for projection
    x_design_scaled = x_scaler.transform(opls_result.x_new.to_frame().T).to_numpy().ravel()
    os_direction = opls_result.orthogonal_space_basis.to_numpy().ravel()
    os_direction = os_direction / np.linalg.norm(os_direction)
    direct_weights = pls.direct_weights_.to_numpy()  # W* maps scaled X -> scores
    os_points = np.array(
        [(x_design_scaled + s * os_direction) @ direct_weights for s in np.linspace(-3.0, 3.0, 25)]
    )

    scores = pls.scores_.to_numpy()

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    ax.scatter(scores[:, 0], scores[:, 1], color=DARK_BLUE, s=30, label="Calibration cheeses")
    ax.plot(ns_line[:, 0], ns_line[:, 1], color=PURPLE, lw=2, label="Null space (PLS inversion)")
    ax.scatter(
        os_points[:, 0],
        os_points[:, 1],
        facecolors="none",
        edgecolors=GREEN,
        s=42,
        label="Orthogonal space (O-PLS), projected",
    )
    ax.scatter(
        tau[0], tau[1], color=BLACK, marker="s", s=70, zorder=5, label="Direct-inversion solution"
    )
    ax.axhline(0, color="0.7", lw=0.8)
    ax.axvline(0, color="0.7", lw=0.8)
    ax.set_xlabel("$t_1$")
    ax.set_ylabel("$t_2$")
    ax.set_title(f"Cheddar cheese: designing toward a taste of {TARGET_TASTE}")
    ax.legend(loc="best", framealpha=0.9, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "pls-model-inversion-null-space.png")
    plt.close(fig)


def main() -> None:
    """Entry point: render into the given directory (default: this script's own)."""
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
    build_figure(out_dir)


if __name__ == "__main__":
    main()
