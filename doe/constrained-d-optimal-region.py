"""Constrained D-optimal design for a two-factor quadratic model.

Illustrates section 5 of the "Optimal designs and OMARS designs" chapter. The region is the
coded square [-1, +1]^2 with the corner x1 + x2 > 1 forbidden. The ten-run D-optimal design
(found by coordinate-exchange, maximising det(X'X)) uses the eight feasible grid points, with
two of them run twice. Reproducible; run from this directory to write the PNG alongside it.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# The 10-run D-optimal design: (x1, x2): replication.
runs = {(-1, -1): 2, (-1, 0): 1, (-1, 1): 2, (0, -1): 1,
        (0, 0): 1, (0, 1): 1, (1, -1): 1, (1, 0): 1}

fig, ax = plt.subplots(figsize=(5.6, 5.6))

# Design region boundary and the infeasible corner x1 + x2 > 1.
ax.add_patch(Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)], closed=True,
                     fill=False, edgecolor="0.5", lw=1.2))
ax.add_patch(Polygon([(1, 0), (1, 1), (0, 1)], closed=True,
                     facecolor="0.85", edgecolor="none", zorder=0))
ax.plot([0, 1], [1, 0], color="0.55", lw=1.2, ls="--")
ax.text(0.72, 0.72, "infeasible\n$x_1 + x_2 > 1$", color="0.4", fontsize=9,
        ha="center", va="center")

# The forbidden corner.
ax.plot(1, 1, marker="x", color="#c0392b", ms=10, mew=2.0, zorder=4)
ax.text(1.0, 1.08, "forbidden", color="#c0392b", fontsize=8, ha="center")

# The chosen runs, marker area proportional to replication.
for (x1, x2), rep in runs.items():
    ax.scatter(x1, x2, s=90 * rep, color="#1f5fa8", zorder=5, edgecolor="white", lw=0.8)
    if rep == 2:
        ax.annotate(r"$\times 2$", (x1, x2), textcoords="offset points",
                    xytext=(11, 9), fontsize=9, color="#1f5fa8")

ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_xlim(-1.35, 1.35)
ax.set_ylim(-1.35, 1.35)
ax.set_aspect("equal")
ax.set_xticks([-1, 0, 1])
ax.set_yticks([-1, 0, 1])
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("constrained-d-optimal-region.png", dpi=300, facecolor="w", edgecolor="w",
            orientation="portrait", format=None, transparent=True)
print("saved figure")
