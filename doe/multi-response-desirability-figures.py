"""Figures for the multi-response optimization subchapter of the DOE chapter.

Continues the bioreactor example from the response-surface-methods section. That
section climbed to T = 343 K, S = 1.60 g/L using profit as the single response,
having folded the competing outcomes into one number by hand. Here a fresh
central composite design is run centred on that point, and a second response is
measured alongside profit: the purity of the product stream.

Profit rises with temperature; purity falls with it, because the product
degrades thermally. Their individual optima therefore sit at opposite ends of
the temperature range, which is what makes the trade-off real rather than
decorative.

Writes, into this directory:

    multi-response-desirability-functions.png
    multi-response-two-contours.png
    multi-response-sweet-spot.png
    multi-response-composite-desirability.png

Run from this directory with process_improve installed:

    python multi-response-desirability-figures.py
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from process_improve.experiments import optimize_responses
from process_improve.experiments._desirability import (
    composite_desirability,
    desirability_maximize,
    desirability_target,
    individual_desirability,
)

# ---------------------------------------------------------------------------
# The two true surfaces.
#
# The profit surface is the one already used by RSM-base-case.py, unchanged, so
# the numbers in this subchapter join up with the ones earlier in the chapter.
# The purity surface is new.
# ---------------------------------------------------------------------------

PROFIT_COLOUR = "#1f5fa8"
PURITY_COLOUR = "#c0392b"
SWEET_COLOUR = "#059669"

# Centre of the new design, and the half-range of each factor.
T_CENTRE, S_CENTRE = 343.0, 1.60
T_HALF, S_HALF = 4.0, 0.2

# Specification limits agreed with the plant, and the desirability ramps.
# The ramp starts at the specification limit, so a setting that misses the
# specification scores zero, and ends at the best value the region can deliver.
PROFIT_MIN, PROFIT_BEST = 725.0, 740.0
PURITY_MIN, PURITY_BEST = 90.0, 96.0

PROFIT_RAMP = (PROFIT_MIN, PROFIT_BEST)
PURITY_RAMP = (PURITY_MIN, PURITY_BEST)


def true_profit(temperature, substrate):
    """Profit in dollars per day. Same surface as RSM-base-case.py."""
    u = (np.asarray(temperature) - 320.0) / 20.0
    v = (np.asarray(substrate) - 1.5) / 1.0
    return (18 * u + 10 * v - 5 * u * v - 7 * u * u - 24 * v * v + 50) * 12 + 2 * np.sin(
        temperature
    ) + 2 * np.cos(substrate)


def true_purity(temperature, substrate):
    """Purity of the product stream, in percent.

    Falls with temperature (thermal degradation) and rises with substrate
    concentration up to a point, so it opposes profit on both axes.
    """
    u = (np.asarray(temperature) - 320.0) / 20.0
    v = (np.asarray(substrate) - 1.5) / 1.0
    return (
        107.5
        - 12 * u
        - 4 * u * u
        + 20 * v
        - 30 * v * v
        - 0.8 * u * v
        + 0.35 * np.sin(temperature)
        - 0.25 * np.cos(3 * substrate)
    )


def build_design():
    """Return the nine-run central composite design and both measured responses."""
    alpha = 1.41  # rotatable axial distance 2**(2/4), rounded as in the book's code
    coded = [
        (-1.0, -1.0),
        (+1.0, -1.0),
        (-1.0, +1.0),
        (+1.0, +1.0),
        (0.0, 0.0),
        (0.0, -alpha),
        (+alpha, 0.0),
        (0.0, +alpha),
        (-alpha, 0.0),
    ]
    x_t = np.array([c[0] for c in coded])
    x_s = np.array([c[1] for c in coded])
    temperature = T_CENTRE + T_HALF * x_t
    substrate = S_CENTRE + S_HALF * x_s
    # Rounded as they would be recorded: profit to the dollar, purity to 0.1%.
    profit = np.round(true_profit(temperature, substrate))
    purity = np.round(true_purity(temperature, substrate), 1)
    return x_t, x_s, temperature, substrate, profit, purity


def fit_quadratic(x_t, x_s, y):
    """Least-squares fit of the full quadratic in two coded factors."""
    design_matrix = np.column_stack(
        [np.ones_like(x_t), x_t, x_s, x_t * x_s, x_t * x_t, x_s * x_s]
    )
    beta, *_ = np.linalg.lstsq(design_matrix, y, rcond=None)
    residual = y - design_matrix @ beta
    r_squared = 1.0 - (residual**2).sum() / ((y - y.mean()) ** 2).sum()
    return beta, r_squared


def evaluate(beta, x_t, x_s):
    """Predicted response from a fitted quadratic at coded settings."""
    return (
        beta[0]
        + beta[1] * x_t
        + beta[2] * x_s
        + beta[3] * x_t * x_s
        + beta[4] * x_t * x_t
        + beta[5] * x_s * x_s
    )


def coefficient_dicts(beta):
    """Coefficients in the form optimize_responses expects."""
    terms = ["Intercept", "T", "S", "T:S", "I(T ** 2)", "I(S ** 2)"]
    return [{"term": t, "coefficient": float(b)} for t, b in zip(terms, beta)]


def save(fig, filename):
    fig.savefig(
        filename,
        dpi=300,
        facecolor="w",
        edgecolor="w",
        orientation="portrait",
        format=None,
        transparent=True,
    )
    print(f"saved {filename}")


# ---------------------------------------------------------------------------
# Figure 1: the shape of an individual desirability function
# ---------------------------------------------------------------------------


def figure_desirability_functions():
    """Show the one-sided and two-sided ramps, and what the weight does."""
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)

    y = np.linspace(86.0, 99.0, 400)
    low, high = PURITY_RAMP

    ax = axes[0]
    ax.plot(y, [desirability_maximize(v, low, high) for v in y], color=PURITY_COLOUR, lw=2)
    ax.set_title("Maximize", fontsize=11)
    ax.set_xlabel("Purity [%]")
    ax.set_ylabel("Desirability, $d$")

    ax = axes[1]
    ax.plot(
        y,
        [desirability_target(v, low, 93.0, high) for v in y],
        color=PURITY_COLOUR,
        lw=2,
    )
    ax.set_title("Target, at 93%", fontsize=11)
    ax.set_xlabel("Purity [%]")

    ax = axes[2]
    for weight, style in ((0.3, ":"), (1.0, "-"), (3.0, "--")):
        ax.plot(
            y,
            [desirability_maximize(v, low, high, weight) for v in y],
            color=PURITY_COLOUR,
            lw=2,
            ls=style,
            label=f"weight = {weight:g}",
        )
    ax.set_title("Effect of the weight", fontsize=11)
    ax.set_xlabel("Purity [%]")
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    for ax in axes:
        ax.axvline(low, color="0.6", lw=0.8, ls=":")
        ax.axvline(high, color="0.6", lw=0.8, ls=":")
        ax.set_ylim(-0.05, 1.08)
        ax.grid(alpha=0.25)

    fig.tight_layout()
    save(fig, "multi-response-desirability-functions.png")


# ---------------------------------------------------------------------------
# Figure 2: the two response surfaces, side by side
# ---------------------------------------------------------------------------


def figure_two_contours(beta_profit, beta_purity, temperature, substrate):
    """Two competing surfaces over the same factor space."""
    # Widened past the cube so the axial runs of the design fall inside the
    # contoured area rather than floating outside it.
    grid_t, grid_s, mesh_t, mesh_s = coded_grid(span=1.45)
    actual_t = T_CENTRE + T_HALF * mesh_t
    actual_s = S_CENTRE + S_HALF * mesh_s

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.4), sharey=True)

    for ax, beta, colour, title, fmt in (
        (axes[0], beta_profit, PROFIT_COLOUR, "Profit [dollars per day]", "%1.0f"),
        (axes[1], beta_purity, PURITY_COLOUR, "Purity [%]", "%1.1f"),
    ):
        z = evaluate(beta, mesh_t, mesh_s)
        contours = ax.contour(actual_t, actual_s, z, colors=colour, linewidths=1.0)
        ax.clabel(contours, inline=1, fontsize=8, fmt=fmt)
        ax.plot(temperature, substrate, "k.", ms=9)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Temperature [K]")
        ax.grid(alpha=0.2)

    axes[0].set_ylabel("Substrate concentration [g/L]")
    fig.tight_layout()
    save(fig, "multi-response-two-contours.png")


# ---------------------------------------------------------------------------
# Figure 3: the sweet spot
# ---------------------------------------------------------------------------


def figure_sweet_spot(beta_profit, beta_purity, optimum_actual):
    """Overlay both surfaces and shade where both specifications are met."""
    grid_t, grid_s, mesh_t, mesh_s = coded_grid()
    actual_t = T_CENTRE + T_HALF * mesh_t
    actual_s = S_CENTRE + S_HALF * mesh_s

    z_profit = evaluate(beta_profit, mesh_t, mesh_s)
    z_purity = evaluate(beta_purity, mesh_t, mesh_s)
    feasible = (z_profit >= PROFIT_MIN) & (z_purity >= PURITY_MIN)

    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    ax.contourf(
        actual_t,
        actual_s,
        feasible.astype(float),
        levels=[0.5, 1.5],
        colors=[SWEET_COLOUR],
        alpha=0.22,
    )

    # Background contours give the shape of each surface, kept faint so the two
    # specification limits stay the most visible lines on the plot.
    ax.contour(actual_t, actual_s, z_profit, levels=8, colors=PROFIT_COLOUR, linewidths=0.5, alpha=0.35)
    ax.contour(actual_t, actual_s, z_purity, levels=8, colors=PURITY_COLOUR, linewidths=0.5, alpha=0.35)

    ax.contour(actual_t, actual_s, z_profit, levels=[PROFIT_MIN], colors=PROFIT_COLOUR, linewidths=2.4)
    ax.contour(actual_t, actual_s, z_purity, levels=[PURITY_MIN], colors=PURITY_COLOUR, linewidths=2.4)

    ax.plot(optimum_actual["T"], optimum_actual["S"], "k*", ms=17, zorder=5)
    ax.annotate(
        "highest overall\ndesirability",
        xy=(optimum_actual["T"], optimum_actual["S"]),
        xytext=(optimum_actual["T"] + 0.35, optimum_actual["S"] + 0.055),
        fontsize=9,
        ha="left",
        va="bottom",
    )

    handles = [
        Line2D([], [], color=PROFIT_COLOUR, lw=2.4, label=f"profit = {PROFIT_MIN:.0f} dollars per day"),
        Line2D([], [], color=PURITY_COLOUR, lw=2.4, label=f"purity = {PURITY_MIN:.0f}%"),
        Patch(facecolor=SWEET_COLOUR, alpha=0.22, label="both specifications met"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.95)

    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Substrate concentration [g/L]")
    ax.set_title("The sweet spot: where both specifications are met at once", fontsize=11)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    save(fig, "multi-response-sweet-spot.png")


# ---------------------------------------------------------------------------
# Figure 4: the composite desirability surface
# ---------------------------------------------------------------------------


def figure_composite(beta_profit, beta_purity, goals, optimum_actual):
    """The single surface the optimizer actually climbs."""
    grid_t, grid_s, mesh_t, mesh_s = coded_grid()
    actual_t = T_CENTRE + T_HALF * mesh_t
    actual_s = S_CENTRE + S_HALF * mesh_s

    z_profit = evaluate(beta_profit, mesh_t, mesh_s)
    z_purity = evaluate(beta_purity, mesh_t, mesh_s)

    overall = np.zeros_like(z_profit)
    for i in range(overall.shape[0]):
        for j in range(overall.shape[1]):
            d_values = [
                individual_desirability(float(z_profit[i, j]), goals[0]),
                individual_desirability(float(z_purity[i, j]), goals[1]),
            ]
            overall[i, j] = composite_desirability(d_values)

    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    filled = ax.contourf(actual_t, actual_s, overall, levels=12, cmap="YlGn")
    lines = ax.contour(actual_t, actual_s, overall, levels=8, colors="0.35", linewidths=0.6)
    ax.clabel(lines, inline=1, fontsize=8, fmt="%1.2f")
    fig.colorbar(filled, ax=ax, label="Overall desirability, $D$")

    ax.plot(optimum_actual["T"], optimum_actual["S"], "k*", ms=16)
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Substrate concentration [g/L]")
    ax.set_title("Overall desirability, and the setting that maximizes it", fontsize=11)
    fig.tight_layout()
    save(fig, "multi-response-composite-desirability.png")


def coded_grid(n=241, span=1.0):
    """Square grid over the coded region, and its meshed form."""
    grid_t = np.linspace(-span, span, n)
    grid_s = np.linspace(-span, span, n)
    mesh_t, mesh_s = np.meshgrid(grid_t, grid_s)
    return grid_t, grid_s, mesh_t, mesh_s


def main():
    x_t, x_s, temperature, substrate, profit, purity = build_design()

    beta_profit, r2_profit = fit_quadratic(x_t, x_s, profit)
    beta_purity, r2_purity = fit_quadratic(x_t, x_s, purity)

    print("Design and measured responses")
    print(f"{'x_T':>7} {'x_S':>7} {'T [K]':>8} {'S [g/L]':>9} {'profit':>8} {'purity':>8}")
    for row in zip(x_t, x_s, temperature, substrate, profit, purity):
        print(f"{row[0]:>7.2f} {row[1]:>7.2f} {row[2]:>8.1f} {row[3]:>9.2f} {row[4]:>8.0f} {row[5]:>8.1f}")

    print(f"\nprofit: R2 = {r2_profit:.4f}, b = {np.round(beta_profit, 2)}")
    print(f"purity: R2 = {r2_purity:.4f}, b = {np.round(beta_purity, 2)}")

    goals = [
        {
            "response": "profit",
            "goal": "maximize",
            "low": PROFIT_RAMP[0],
            "high": PROFIT_RAMP[1],
        },
        {
            "response": "purity",
            "goal": "maximize",
            "low": PURITY_RAMP[0],
            "high": PURITY_RAMP[1],
        },
    ]
    models = [
        {
            "response_name": "profit",
            "coefficients": coefficient_dicts(beta_profit),
            "factor_names": ["T", "S"],
        },
        {
            "response_name": "purity",
            "coefficients": coefficient_dicts(beta_purity),
            "factor_names": ["T", "S"],
        },
    ]
    factor_ranges = {
        "T": {"low": T_CENTRE - T_HALF, "high": T_CENTRE + T_HALF},
        "S": {"low": S_CENTRE - S_HALF, "high": S_CENTRE + S_HALF},
    }
    result = optimize_responses(
        models, goals=goals, method="desirability", factor_ranges=factor_ranges
    )["desirability"]
    optimum_actual = result["optimal_actual"]

    print("\nOverall desirability optimum")
    print(f"  coded : {  {k: round(v, 3) for k, v in result['optimal_coded'].items()} }")
    print(f"  actual: T = {optimum_actual['T']:.1f} K, S = {optimum_actual['S']:.2f} g/L")
    print(f"  predicted: {  {k: round(v, 1) for k, v in result['predicted_responses'].items()} }")
    print(f"  D = {result['composite_desirability']:.3f}")

    figure_desirability_functions()
    figure_two_contours(beta_profit, beta_purity, temperature, substrate)
    figure_sweet_spot(beta_profit, beta_purity, optimum_actual)
    figure_composite(beta_profit, beta_purity, goals, optimum_actual)


if __name__ == "__main__":
    main()
