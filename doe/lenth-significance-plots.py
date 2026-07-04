"""Effect-significance figures for Chapter 5 of "Process Improvement using Data".

Generates three committed PNGs, all with a perceptually uniform, colourblind-safe
palette (Okabe-Ito):

  * pareto-plot-full-fraction.png : Pareto of the fifteen effects of the 2^4 example,
    with Lenth's margin-of-error (ME) and simultaneous-margin-of-error (SME) lines.
  * half-normal-full-fraction.png : half-normal plot of the same fifteen effects.
  * pareto-plot-pid.png           : Pareto of the seven effects of the saturated
    2^(7-4) screening design, with the Lenth ME line.

The book shows Plotly code; these committed images are the matplotlib versions
(per the chapter-rework playbook). Run with:
    uv run --with numpy --with scipy --with matplotlib python lenth-significance-plots.py
"""

import itertools

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy import stats

# Okabe-Ito colourblind-safe palette. Bars/points are coloured by the SIGN of
# the effect (positive = orange, negative = blue); the Lenth margin-of-error
# and simultaneous-margin-of-error lines, not the colour, mark significance.
BLUE, ORANGE = "#0072B2", "#E69F00"
ME_COLOUR, SME_COLOUR = "#000000", "#666666"


def lenth(coeffs, alpha=0.05):
    """Return (PSE, ME, SME) on the scale of the supplied coefficients."""
    a = np.abs(coeffs)
    m = len(a)
    s0 = 1.5 * np.median(a)
    pse = 1.5 * np.median(a[a < 2.5 * s0])
    me = stats.t.ppf(1 - alpha / 2, m / 3) * pse
    sme = stats.t.ppf(1 - (alpha / 2) / m, m / 3) * pse
    return pse, me, sme


def effects_from(y, k):
    """Least-squares effects (coefficients, excl. intercept) of a full 2^k factorial."""
    levels = [-1, 1]
    design = np.array(list(itertools.product(*([levels] * k))))[:, ::-1]
    names = [chr(ord("A") + i) for i in range(k)]
    factor = {n: design[:, i] for i, n in enumerate(names)}
    terms = {}
    for order in range(1, k + 1):
        for combo in itertools.combinations(names, order):
            col = np.ones(len(design))
            for f in combo:
                col = col * factor[f]
            terms["".join(combo)] = col
    X = np.column_stack([np.ones(len(design))] + list(terms.values()))
    b = np.linalg.solve(X.T @ X, X.T @ y)
    return dict(zip(terms.keys(), b[1:]))


def _sign_legend(me, sme=None):
    """Legend handles: sign colours plus the ME/SME cutoff lines."""
    handles = [Patch(color=ORANGE, label="positive effect"),
               Patch(color=BLUE, label="negative effect"),
               Line2D([0], [0], color=ME_COLOUR, linestyle="--", label=f"ME = {me:.2f}")]
    if sme is not None:
        handles.append(Line2D([0], [0], color=SME_COLOUR, linestyle=":", label=f"SME = {sme:.2f}"))
    return handles


def pareto(effects, me, sme, filename, title):
    ordered = sorted(effects, key=lambda k: abs(effects[k]))
    vals = [abs(effects[k]) for k in ordered]
    # Colour each bar by the SIGN of its effect, not by significance.
    colours = [ORANGE if effects[k] > 0 else BLUE for k in ordered]
    fig, ax = plt.subplots(figsize=(6.5, max(3.5, 0.32 * len(ordered) + 1)))
    ax.barh(ordered, vals, color=colours)
    ax.axvline(me, color=ME_COLOUR, linestyle="--", linewidth=1.6)
    if sme is not None:
        ax.axvline(sme, color=SME_COLOUR, linestyle=":", linewidth=1.6)
    ax.set_xlabel("|effect|  (coefficient scale)")
    ax.set_ylabel("Term")
    ax.set_title(title)
    ax.legend(handles=_sign_legend(me, sme), loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)


def half_normal(effects, me, filename, title):
    ordered = sorted(effects, key=lambda k: abs(effects[k]))
    vals = np.array([abs(effects[k]) for k in ordered])
    positive = np.array([effects[k] > 0 for k in ordered])
    m = len(vals)
    q = stats.halfnorm.ppf((np.arange(1, m + 1) - 0.5) / m)
    active = vals > me
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    # Colour points by sign; significance is read from the ME line and the labels.
    ax.scatter(q[positive], vals[positive], color=ORANGE, zorder=3)
    ax.scatter(q[~positive], vals[~positive], color=BLUE, zorder=3)
    # reference line through the origin fitted to the inactive (noise) effects
    if active.sum() < m:
        slope = np.sum(q[~active] * vals[~active]) / np.sum(q[~active] ** 2)
        xs = np.linspace(0, q.max() * 1.05, 50)
        ax.plot(xs, slope * xs, color="#999999", linewidth=1.0, zorder=1)
    ax.axhline(me, color=ME_COLOUR, linestyle="--", linewidth=1.2, zorder=2)
    for xi, yi, name in zip(q, vals, ordered):
        if yi > me:
            ax.annotate(name, (xi, yi), textcoords="offset points", xytext=(-10, 3), fontsize=9)
    ax.set_xlabel("Half-normal quantile")
    ax.set_ylabel("|effect|  (coefficient scale)")
    ax.set_title(title)
    ax.legend(handles=_sign_legend(me), loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)


# ---- 2^4 example -----------------------------------------------------------
y16 = np.array([45, 71, 48, 65, 68, 60, 80, 65, 43, 100, 45, 104, 75, 86, 70, 96])
eff16 = effects_from(y16, 4)
pse, me, sme = lenth(np.array(list(eff16.values())))
print(f"2^4:  PSE={pse:.2f} ME={me:.2f} SME={sme:.2f}")
pareto(eff16, me, sme, "pareto-plot-full-fraction.png",
       "Pareto plot with Lenth cutoffs (2$^4$ factorial)")
half_normal(eff16, me, "half-normal-full-fraction.png",
            "Half-normal plot of effects (2$^4$ factorial)")

# ---- saturated 2^(7-4) screening design ------------------------------------
levels = [-1, 1]
d3 = np.array(list(itertools.product(levels, levels, levels)))[:, ::-1]
A, B, C = d3.T
cols = {"A": A, "B": B, "C": C, "D": A * B, "E": A * C, "F": B * C, "G": A * B * C}
y8 = np.array([77.1, 68.9, 75.5, 72.5, 67.9, 68.5, 71.5, 63.7])
X8 = np.column_stack([np.ones(8)] + list(cols.values()))
b8 = np.linalg.solve(X8.T @ X8, X8.T @ y8)
eff8 = dict(zip(cols.keys(), b8[1:]))
pse7, me7, _ = lenth(np.array(list(eff8.values())))
print(f"2^(7-4):  PSE={pse7:.2f} ME={me7:.2f}")
pareto(eff8, me7, None, "pareto-plot-pid.png",
       "Pareto plot with Lenth cutoff (saturated 2$^{7-4}$ design)")
