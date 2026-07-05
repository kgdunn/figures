"""The Cox-effect direction for a mixture coordinate exchange.

Illustrates the mixture-experiments section of the "Design and Analysis of Experiments"
chapter. A single ingredient proportion cannot be changed on its own, because the
proportions must keep summing to one. Moving ingredient 1 from 0.30 to 0.50 forces the
other two to shrink while keeping their ratio x2 : x3 fixed at 0.75. The move runs along
the Cox-effect direction: the straight line from the x1 vertex through the current point to
the opposite edge. Reproducible; run from this directory to write the PNG alongside it.
"""
import numpy as np
import matplotlib.pyplot as plt

H = np.sqrt(3.0) / 2.0


def bary(a1, a2, a3):
    """Barycentric (proportions summing to 1) to 2-D: x1 top, x2 lower-left, x3 lower-right."""
    return (0.5 * a1 + a3, H * a1)


fig, ax = plt.subplots(figsize=(5.6, 5.2))

# Simplex boundary.
verts = [bary(1, 0, 0), bary(0, 1, 0), bary(0, 0, 1)]
ax.add_patch(plt.Polygon(verts, closed=True, fill=False, edgecolor="0.5", lw=1.2))

# The Cox-effect direction for ingredient 1 is the line from the x1 vertex, through the
# current point, to the opposite edge (a1 = 0) at the same x2 : x3 ratio.
P = (0.30, 0.30, 0.40)          # current design point
M = (0.50, 0.214, 0.286)        # after increasing x1 by 0.20
ratio = P[1] / (P[1] + P[2])    # x2 share of the remaining 1 - x1
edge = (0.0, ratio, 1 - ratio)  # foot of the line on the a1 = 0 edge
ax.plot(*zip(bary(*(1, 0, 0)), bary(*edge)), color="0.55", lw=1.1, ls="--", zorder=2)

# Current and moved points, with an arrow along the Cox direction.
px, py = bary(*P)
mx, my = bary(*M)
ax.annotate("", xy=(mx, my), xytext=(px, py),
            arrowprops=dict(arrowstyle="-|>", color="#c0392b", lw=1.8))
ax.scatter([px], [py], s=90, color="#1f5fa8", zorder=5, edgecolor="white", lw=0.8)
ax.scatter([mx], [my], s=90, color="#c0392b", zorder=5, edgecolor="white", lw=0.8)
ax.annotate("(0.30, 0.30, 0.40)", (px, py), textcoords="offset points", xytext=(10, -12),
            fontsize=8, color="#1f5fa8")
ax.annotate("(0.50, 0.214, 0.286)", (mx, my), textcoords="offset points", xytext=(8, 6),
            fontsize=8, color="#c0392b")

# Vertex labels (pure components).
ax.annotate(r"$x_1 = 1$", bary(1, 0, 0), textcoords="offset points", xytext=(0, 8),
            ha="center", fontsize=10)
ax.annotate(r"$x_2 = 1$", bary(0, 1, 0), textcoords="offset points", xytext=(-6, -12),
            ha="center", fontsize=10)
ax.annotate(r"$x_3 = 1$", bary(0, 0, 1), textcoords="offset points", xytext=(6, -12),
            ha="center", fontsize=10)

ax.set_xlim(-0.18, 1.18)
ax.set_ylim(-0.22, H + 0.16)
ax.set_aspect("equal")
ax.axis("off")
fig.tight_layout()
fig.savefig("mixture-cox-direction.png", dpi=300, facecolor="w", edgecolor="w",
            transparent=True)
print("saved mixture-cox-direction.png")
