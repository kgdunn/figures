"""The estimability frontier for foldover OMARS designs.

Illustrates the "OMARS designs" section of the Design and Analysis of Experiments
chapter. A foldover design is built as [H; -H; 0], so every second-order term, being an
even function, takes identical values in H and in -H. The second-order columns therefore
see at most h + 1 distinct rows, and the full second-order model in k factors only becomes
estimable at N = k^2 + k + 1 runs.

That frontier sits above the parameter count 1 + 2k + k(k-1)/2 by exactly k(k-1)/2 runs,
the number of two-factor interactions. The shaded band is the region where a design has
more runs than the model has parameters and still cannot fit it.

Reproducible; run from this directory to write the PNG alongside it.
"""
import matplotlib.pyplot as plt
import numpy as np

k = np.arange(3, 8)
saturated = 2 * k + 1                        # the DSD size: estimable, but no error df
parameters = 1 + 2 * k + k * (k - 1) // 2    # terms in the full second-order model
frontier = k**2 + k + 1                      # smallest foldover that can estimate them

# Okabe-Ito colourblind-safe palette.
BLUE, ORANGE, VERMILLION, GREY = "#0072B2", "#E69F00", "#D55E00", "#666666"

fig, ax = plt.subplots(figsize=(8.4, 5.6))

# The band between "enough parameters" and "actually estimable" is the whole point.
ax.fill_between(k, parameters, frontier, color=VERMILLION, alpha=0.13, zorder=1)
ax.annotate(
    "Shaded band: more runs than\nparameters, yet still not estimable\n"
    "(the band is $k(k-1)/2$ runs deep)",
    xy=(6.3, 20.0), fontsize=12.5, color=VERMILLION, ha="center", va="center", zorder=6,
)

ax.plot(k, frontier, marker="o", ms=8, lw=2.4, color=VERMILLION, zorder=5,
        label="Estimability frontier, $k^2 + k + 1$")
ax.plot(k, parameters, marker="s", ms=7, lw=2.2, color=BLUE, zorder=5,
        label="Parameters in the full second-order model, $1 + 2k + k(k-1)/2$")
ax.plot(k, saturated, marker="^", ms=7, lw=2.2, color=ORANGE, zorder=5,
        label="Definitive screening design, $2k + 1$ runs")

# Label each frontier value; these are the default run sizes generate_omars picks.
for kk, ff in zip(k, frontier):
    ax.annotate(f"{ff}", xy=(kk, ff), xytext=(0, 9), textcoords="offset points",
                ha="center", fontsize=12, color=VERMILLION, fontweight="bold")
for kk, pp in zip(k, parameters):
    ax.annotate(f"{pp}", xy=(kk, pp), xytext=(0, -17), textcoords="offset points",
                ha="center", fontsize=11.5, color=BLUE)

# The four-factor case, worked in the text: 19 runs, 15 parameters, rank 14. The note sits
# above the frontier and to its left, where the axes are empty, rather than below the line
# among the definitive screening design markers. Its leader stops short of the frontier and
# points towards the marked cross, rather than crossing the line to touch it.
ax.plot([4], [19], marker="x", ms=12, mew=2.6, color=GREY, zorder=6)
ax.annotate("19 runs, 15 parameters,\nmodel matrix rank 14",
            xy=(3.88, 21.4), xytext=(3.05, 33.0), ha="left", va="center", fontsize=12,
            color=GREY, zorder=7,
            arrowprops=dict(arrowstyle="->", color=GREY, lw=2.2, shrinkB=3, zorder=7,
                            connectionstyle="arc3,rad=0.3"))

ax.set_xticks(k)
ax.set_xlabel("Number of factors, $k$", fontsize=13)
ax.set_ylabel("Number of runs, $N$", fontsize=13)
ax.set_xlim(2.8, 7.35)
ax.set_ylim(0, 63)
ax.grid(axis="y", color="0.9", lw=0.9)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(loc="upper left", fontsize=11.5, frameon=False)

fig.tight_layout()
fig.savefig("omars-estimability-frontier.png", dpi=300, facecolor="w", edgecolor="w",
            format=None, transparent=True)
print("saved figure")
