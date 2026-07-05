"""Optimal design for a second-order Scheffe model in three mixture ingredients.

Illustrates the mixture-experiments section of the "Design and Analysis of Experiments"
chapter. The experimental region is the simplex x1 + x2 + x3 = 1. A second-order Scheffe
model has six terms (three linear, three binary blends), and its natural support is the
three pure-component vertices plus the three binary-blend edge midpoints. The overall
centroid is shown as an open marker: a common extra run used to check the fit. Reproducible;
run from this directory to write the PNG alongside it.
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
tri = plt.Polygon(verts, closed=True, fill=False, edgecolor="0.5", lw=1.2)
ax.add_patch(tri)

# Support points of the second-order Scheffe model.
vertices = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
midpoints = [(0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]
for a in vertices + midpoints:
    x, y = bary(*a)
    ax.scatter(x, y, s=110, color="#1f5fa8", zorder=5, edgecolor="white", lw=0.8)

# Overall centroid: an optional check run, drawn as an open marker.
cx, cy = bary(1 / 3, 1 / 3, 1 / 3)
ax.scatter(cx, cy, s=110, facecolor="none", edgecolor="#1f5fa8", lw=1.6, zorder=5)
ax.annotate("centroid\n(check run)", (cx, cy), textcoords="offset points",
            xytext=(0, -28), ha="center", fontsize=8, color="#1f5fa8")

# Vertex labels (pure components) and one edge label.
ax.annotate(r"$x_1 = 1$", bary(1, 0, 0), textcoords="offset points", xytext=(0, 8),
            ha="center", fontsize=10)
ax.annotate(r"$x_2 = 1$", bary(0, 1, 0), textcoords="offset points", xytext=(-6, -12),
            ha="center", fontsize=10)
ax.annotate(r"$x_3 = 1$", bary(0, 0, 1), textcoords="offset points", xytext=(6, -12),
            ha="center", fontsize=10)
mx, my = bary(0.5, 0.5, 0)
ax.annotate("binary\nblend", (mx, my), textcoords="offset points", xytext=(-20, 4),
            ha="center", fontsize=8, color="0.4")

ax.set_xlim(-0.18, 1.18)
ax.set_ylim(-0.22, H + 0.16)
ax.set_aspect("equal")
ax.axis("off")
fig.tight_layout()
fig.savefig("mixture-scheffe-design.png", dpi=300, facecolor="w", edgecolor="w",
            transparent=True)
print("saved mixture-scheffe-design.png")
