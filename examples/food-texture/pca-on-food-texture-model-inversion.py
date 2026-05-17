"""
Static figures for the "PCA model inversion" worked example in pid-book
(Latent Variable Modelling -> Applications chapter).

Builds a 2-component PCA model of the food-texture dataset
(`food-texture.csv`, 50 pastries x 5 measurements) after mean-centring and
scaling to unit variance. The worked example then inverts the model, going
from a target score back to a pastry recipe. The unconstrained inversion is
shown interactively in a downloadable notebook; the figures generated here
are:

  1. SPE per pastry, with the 95% limit line.
  2. Hotelling's T^2 per pastry, with the 95% limit line.
  3. Score plot (t_1 vs t_2, with the 95% T^2 ellipse) and loadings plot
     (p_1 vs p_2), side by side.
  4. Constrained inversion: the target score projected onto the Hardness
     constraint lines in the (t_1, t_2) score plane.

Headline numbers: R^2 cumulative = [0.606, 0.865]; SPE 95% limit = 1.383;
T^2 95% limit = 6.645. Constrained inversion: Hardness loadings
p_h = [0.153, 0.805]; target t* = (2, -1); Hardness 110 projects to
(1.980, -1.103) and Hardness 150 projects to (2.274, 0.438).

Generates four PNG figures next to this script, referenced by
`latent-variable-modelling/applications-of-latent-variable-models.rst`.

Run:

    python examples/food-texture/pca-on-food-texture-model-inversion.py

Requires:
    process_improve, matplotlib, numpy, pandas
"""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from process_improve.multivariate import PCA, MCUVScaler

FIGURES_DIR = pathlib.Path(__file__).resolve().parent
DATA_FILE = FIGURES_DIR / "food-texture.csv"
CONF_LEVEL = 0.95


def figure_spe(spe: pd.Series, spe_limit: float, out: pathlib.Path) -> None:
    """SPE per pastry, with the 95% limit line."""
    pastry = spe.index + 1
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(pastry, spe.values, color="#4c72b0", width=0.7)
    ax.axhline(spe_limit, color="red", linestyle="--", linewidth=1,
               label=f"95% limit ({spe_limit:.3f})")
    ax.set_xlabel("Pastry number")
    ax.set_ylabel("SPE")
    ax.set_title("Squared prediction error per pastry")
    ax.set_xlim(0.3, len(spe) + 0.7)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def figure_t2(t2: pd.Series, t2_limit: float, out: pathlib.Path) -> None:
    """Hotelling's T^2 per pastry, with the 95% limit line."""
    pastry = t2.index + 1
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(pastry, t2.values, color="#4c72b0", width=0.7)
    ax.axhline(t2_limit, color="red", linestyle="--", linewidth=1,
               label=f"95% limit ({t2_limit:.3f})")
    ax.set_xlabel("Pastry number")
    ax.set_ylabel("Hotelling's $T^2$")
    ax.set_title("Hotelling's $T^2$ per pastry")
    ax.set_xlim(0.3, len(t2) + 0.7)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def figure_scores_and_loadings(model: PCA, out: pathlib.Path) -> None:
    """Score plot (with 95% T^2 ellipse) and loadings plot, side by side."""
    scores = model.scores_
    loadings = model.loadings_
    ci_x, ci_y = model.ellipse_coordinates(score_horiz=1, score_vert=2,
                                           conf_level=CONF_LEVEL)
    r2 = model.r2_cumulative_.values
    pct1 = 100 * r2[0]
    pct2 = 100 * (r2[1] - r2[0])

    fig, (ax_s, ax_l) = plt.subplots(1, 2, figsize=(13, 6))

    # Scores: t_1 vs t_2 with the 95% T^2 ellipse.
    ax_s.plot(scores.iloc[:, 0], scores.iloc[:, 1], "k.", markersize=7,
              label="Pastry scores")
    ax_s.plot(ci_x, ci_y, color="palevioletred", linewidth=1.8,
              label="95% $T^2$ limit")
    for pastry, (t1, t2) in enumerate(zip(scores.iloc[:, 0], scores.iloc[:, 1]), start=1):
        ax_s.annotate(str(pastry), (t1, t2), textcoords="offset points",
                      xytext=(4, 2), fontsize=7, color="grey")
    ax_s.axhline(0, color="grey", linewidth=0.5)
    ax_s.axvline(0, color="grey", linewidth=0.5)
    ax_s.set_xlabel(f"$t_1$ [{pct1:.1f}%]")
    ax_s.set_ylabel(f"$t_2$ [{pct2:.1f}%]")
    ax_s.set_title("Scores")
    ax_s.set_aspect("equal", adjustable="datalim")
    ax_s.legend(loc="upper right", fontsize=9)
    ax_s.grid(True, alpha=0.3)

    # Loadings: p_1 vs p_2 with "x" markers, one per measured variable.
    ax_l.plot(loadings.iloc[:, 0], loadings.iloc[:, 1], "x", color="#4c72b0",
              markersize=10, markeredgewidth=2)
    for name, (p1, p2) in loadings.iterrows():
        ax_l.annotate(name, (p1, p2), textcoords="offset points",
                      xytext=(6, 4), fontsize=10)
    ax_l.axhline(0, color="grey", linewidth=0.5)
    ax_l.axvline(0, color="grey", linewidth=0.5)
    ax_l.set_xlabel(f"$p_1$ [{pct1:.1f}%]")
    ax_l.set_ylabel(f"$p_2$ [{pct2:.1f}%]")
    ax_l.set_title("Loadings")
    ax_l.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def figure_constraint_projection(model: PCA, scaler: MCUVScaler,
                                 out: pathlib.Path) -> None:
    """Constrained inversion in the (t_1, t_2) score plane.

    Fixing Hardness restricts the score to a line; the constrained score is
    the target t* projected onto that line. Two constraint lines are drawn,
    for Hardness = 110 (close to t*) and Hardness = 150 (far from t*).
    """
    scores = model.scores_
    t1 = scores.iloc[:, 0].to_numpy()
    t2 = scores.iloc[:, 1].to_numpy()
    ci_x, ci_y = model.ellipse_coordinates(score_horiz=1, score_vert=2,
                                           conf_level=CONF_LEVEL)
    p_h = model.loadings_.loc["Hardness"].to_numpy()
    t_star = np.array([2.0, -1.0])

    def project_to_constraint(hardness: float) -> tuple[float, np.ndarray]:
        """Score meeting the Hardness target, closest to t*."""
        z_h = (hardness - scaler.center_["Hardness"]) / scaler.scale_["Hardness"]
        lam = (p_h @ t_star - z_h) / (p_h @ p_h)
        return z_h, t_star - lam * p_h

    constraints = {110: ("#4c72b0", project_to_constraint(110)),
                   150: ("#dd8452", project_to_constraint(150))}

    fig, ax = plt.subplots(figsize=(8, 7.5))

    # Observed pastry scores (context) and the 95% T^2 ellipse.
    ax.plot(t1, t2, ".", color="#bdbdbd", markersize=6,
            label="Pastry scores")
    ax.plot(ci_x, ci_y, color="palevioletred", linewidth=1.3, alpha=0.7,
            label="95% $T^2$ limit")
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)

    # Axis range, wide enough to hold the data, t*, and both projections.
    proj_x = [t_proj[0] for _, (_, t_proj) in constraints.values()]
    proj_y = [t_proj[1] for _, (_, t_proj) in constraints.values()]
    x_lo = min(t1.min(), ci_x.min(), t_star[0], *proj_x)
    x_hi = max(t1.max(), ci_x.max(), t_star[0], *proj_x)
    y_lo = min(t2.min(), ci_y.min(), t_star[1], *proj_y)
    y_hi = max(t2.max(), ci_y.max(), t_star[1], *proj_y)
    x_pad = 0.10 * (x_hi - x_lo)
    y_pad = 0.10 * (y_hi - y_lo)
    ax.set_xlim(x_lo - x_pad, x_hi + x_pad)
    ax.set_ylim(y_lo - y_pad, y_hi + y_pad)

    # Constraint lines: { (t1, t2) : p_h[0]*t1 + p_h[1]*t2 = z_h }.
    line_x = np.array([x_lo - x_pad, x_hi + x_pad])
    for hardness, (color, (z_h, t_proj)) in constraints.items():
        line_y = (z_h - p_h[0] * line_x) / p_h[1]
        ax.plot(line_x, line_y, color=color, linewidth=1.8)
        # Label the line on its open left end.
        ax.annotate(f"Hardness = {hardness}",
                    (line_x[0], line_y[0]), textcoords="offset points",
                    xytext=(8, 6), ha="left", fontsize=9, color=color)
        # Projection arrow from t* to the constrained score.
        ax.annotate("", xy=tuple(t_proj), xytext=tuple(t_star),
                    arrowprops=dict(arrowstyle="->", color=color,
                                    linewidth=1.6, shrinkA=0, shrinkB=0))
        ax.plot(*t_proj, "o", color=color, markersize=10,
                markeredgecolor="white", markeredgewidth=1.0, zorder=5)
        ax.annotate(f"  ({t_proj[0]:.2f}, {t_proj[1]:.2f})", tuple(t_proj),
                    textcoords="offset points", xytext=(8, -4),
                    fontsize=8, color=color)

    # The target score t*.
    ax.plot(*t_star, "*", color="black", markersize=15, zorder=6,
            label="Target $t^*$")
    ax.annotate("$t^* = (2,\\ -1)$", tuple(t_star), textcoords="offset points",
                xytext=(10, 8), fontsize=10)

    ax.set_xlabel("$t_1$")
    ax.set_ylabel("$t_2$")
    ax.set_title("Constrained inversion: target projected onto Hardness lines")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> None:
    food = pd.read_csv(DATA_FILE)
    print(f"food-texture data: {food.shape[0]} pastries x {food.shape[1]} measurements")

    scaler = MCUVScaler().fit(food)
    model = PCA(n_components=2).fit(scaler.transform(food))
    print(f"R^2 cumulative: {model.r2_cumulative_.values.round(3)}")

    spe = model.spe_.iloc[:, -1]
    spe_limit = float(model.spe_limit(conf_level=CONF_LEVEL))
    print(f"95% SPE limit: {spe_limit:.3f}")
    figure_spe(spe, spe_limit,
               FIGURES_DIR / "pca-on-food-texture-model-inversion-spe.png")
    print("[1/3] SPE figure done")

    t2 = model.hotellings_t2_.iloc[:, -1]
    t2_limit = float(model.hotellings_t2_limit(conf_level=CONF_LEVEL))
    print(f"95% T^2 limit: {t2_limit:.3f}")
    figure_t2(t2, t2_limit,
              FIGURES_DIR / "pca-on-food-texture-model-inversion-t2.png")
    print("[2/3] T^2 figure done")

    figure_scores_and_loadings(
        model, FIGURES_DIR / "pca-on-food-texture-model-inversion-scores-and-loadings.png")
    print("[3/4] scores-and-loadings figure done")

    figure_constraint_projection(
        model, scaler,
        FIGURES_DIR / "pca-on-food-texture-model-inversion-constraint-projection.png")
    print("[4/4] constraint-projection figure done")

    print("Wrote 4 PNG figures to:", FIGURES_DIR)


if __name__ == "__main__":
    main()
