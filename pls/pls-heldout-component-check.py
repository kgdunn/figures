"""Generate the two committed PNGs that check the second cheese component on testing data.

Companion to the pid-book section "Why a component that points the right way can
still be dropped" in
``latent-variable-modelling/projection-to-latent-structures/variability-explained-with-each-component.rst``.

Both figures come from one seven-fold cross-validation of a PLS model of Taste on
Acetic, H2S and Lactic (all 30 cheeses), with the folds that
``compare_cv_criteria(X, Y, random_state=0)`` uses: shuffled ``KFold(7)`` with seed 0.

* ``pls-heldout-slope-ratio.png``: for every test cheese (the testing data of its
  fold), the correction a component makes to its predicted taste, against the
  prediction error before that correction (actual minus predicted taste). The
  least-squares slope through the origin is the slope ratio ``s_a``; the prediction
  error falls only when ``s_a > 1/2``, because
  ``PRESS[a-1] - PRESS[a] = (2 s_a - 1) * sum(correction**2)``.
* ``pls-fold-weights.png``: the weights of components 1 and 2 in the model fitted
  to all 30 cheeses (bars), and in each of the seven fold models (dots), with the
  arbitrary sign of each fold's component aligned to the full model.

Requires the ``process_improve`` package (``pip install process-improve``) for ``PLS``.

Usage::

    python pls/pls-heldout-component-check.py [output_dir]

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
from sklearn.model_selection import KFold

from process_improve.multivariate import PLS

DATA_URL = "https://openmv.net/file/cheddar-cheese.csv"
X_COLUMNS = ["Acetic", "H2S", "Lactic"]

DARK_BLUE = "#1f3d7a"  # test cheeses, and the slope on the testing data
ORANGE = "#e6820a"     # the break-even slope of one half
BLACK = "#111111"      # a correction that is exactly right, slope one
GREY = "#c9c9c9"       # bars for the model fitted to all cheeses

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 140})


def cross_validate(X: pd.DataFrame, Y: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Correction and prediction error before it, per test cheese and component, and each fold's weights."""
    correction = np.zeros((len(X), 2))
    error_before = np.zeros((len(X), 2))
    fold_weights = []
    for train, test in KFold(n_splits=7, shuffle=True, random_state=0).split(X):
        before = np.full(len(test), Y.iloc[train, 0].mean())
        for a in (1, 2):
            fold_model = PLS(n_components=a).fit(X.iloc[train], Y.iloc[train])
            after = fold_model.predict(X.iloc[test]).to_numpy().ravel()
            correction[test, a - 1] = after - before
            error_before[test, a - 1] = Y.iloc[test, 0].to_numpy() - before
            before = after
        fold_weights.append(fold_model.x_weights_.to_numpy())
    return correction, error_before, fold_weights


def slope_figure(correction: np.ndarray, error_before: np.ndarray, out_dir: Path) -> None:
    """One panel per component: test cheeses, slope one, slope one half, and the slope on the testing data."""
    slopes = (correction * error_before).sum(axis=0) / (correction**2).sum(axis=0)
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 5.0))
    for a, ax in enumerate(axes):
        g, r = correction[:, a], error_before[:, a]
        reach = 1.08 * np.abs(g).max()
        ends = np.array([-reach, reach])
        ax.axhline(0, color=BLACK, lw=0.6, zorder=1)
        ax.axvline(0, color=BLACK, lw=0.6, zorder=1)
        ax.plot(ends, ends, color=BLACK, lw=1.8, label="Correction exactly right (slope 1)")
        ax.plot(ends, 0.5 * ends, color=ORANGE, lw=1.8, ls=":", label="Break-even (slope 1/2)")
        ax.plot(ends, slopes[a] * ends, color=DARK_BLUE, lw=1.8, ls="--", label="Slope on the testing data, $s_a$")
        ax.plot(g, r, "o", ms=6, color=DARK_BLUE, mec="white", mew=0.8, zorder=4, label="Test cheese")
        ax.set_xlim(ends)
        ax.set_title(f"Component {a + 1}: $s_{a + 1} = {slopes[a]:.2f}$")
        ax.set_xlabel(f"Correction by component {a + 1}\n(change in predicted taste)")
    axes[0].set_ylabel("Prediction error before the component\n(actual \u2212 predicted taste)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9.5, frameon=False)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out_dir / "pls-heldout-slope-ratio.png")
    plt.close(fig)


def weights_figure(full_weights: np.ndarray, fold_weights: list[np.ndarray], out_dir: Path) -> None:
    """Bars for the full model's weights, dots for the seven sign-aligned fold models."""
    positions = np.arange(len(X_COLUMNS))
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0), sharey=True)
    for a, ax in enumerate(axes):
        ax.bar(positions, full_weights[:, a], width=0.55, color=GREY, edgecolor=BLACK, lw=0.8,
               label="All 30 cheeses", zorder=2)
        for k, weights in enumerate(fold_weights):
            aligned = weights[:, a] * np.sign(weights[:, a] @ full_weights[:, a])
            jitter = (k - 3) * 0.045
            ax.plot(positions + jitter, aligned, "o", ms=6, color="white", mec=DARK_BLUE, mew=1.4, zorder=4,
                    label="Each of the 7 fold models" if k == 0 else None)
        ax.axhline(0, color=BLACK, lw=0.8, zorder=3)
        ax.set_xticks(positions, X_COLUMNS)
        ax.set_title(f"Weights $w_{a + 1}$ of component {a + 1}")
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Weight")
    axes[1].legend(loc="upper right", fontsize=9, framealpha=0.93)
    fig.tight_layout()
    fig.savefig(out_dir / "pls-fold-weights.png")
    plt.close(fig)


def main() -> None:
    """Entry point: render both figures into the given directory (default: this script's own)."""
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent
    cheese = pd.read_csv(DATA_URL)
    X, Y = cheese[X_COLUMNS], cheese[["Taste"]]
    correction, error_before, fold_weights = cross_validate(X, Y)
    full_weights = PLS(n_components=2).fit(X, Y).x_weights_.to_numpy()
    slope_figure(correction, error_before, out_dir)
    weights_figure(full_weights, fold_weights, out_dir)


if __name__ == "__main__":
    main()
