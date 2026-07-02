"""Predicted conversion contours for the three-factor chemical-reaction
exercise in Chapter 5 of "Process Improvement using Data".

This regenerates ``chemical-reaction-contours.png``. It reproduces the
model from the earlier MATLAB script ``chemical_reaction_contours.m``,
but draws the contour surface with a perceptually uniform, colourblind-safe
colormap (viridis) instead of the rainbow ``hsv`` map.

Run with, for example:
    uv run --with numpy --with matplotlib python chemical_reaction_contours.py
"""

import matplotlib.pyplot as plt
import numpy as np

# Coded 2^3 design (factors A, B, C) written out with all interactions, then
# the full least-squares fit, exactly as in the exercise.
design = np.array(
    [
        [1, -1, -1, -1],
        [1, +1, -1, -1],
        [1, -1, +1, -1],
        [1, +1, +1, -1],
        [1, -1, -1, +1],
        [1, +1, -1, +1],
        [1, -1, +1, +1],
        [1, +1, +1, +1],
    ],
    dtype=float,
)
A_col, B_col, C_col = design[:, 1], design[:, 2], design[:, 3]
X = np.column_stack(
    [design, A_col * B_col, A_col * C_col, B_col * C_col, A_col * B_col * C_col]
)
y = np.array([72, 73, 66, 87, 70, 73, 67, 87], dtype=float)
b = np.linalg.solve(X.T @ X, X.T @ y)  # X is orthogonal

# Predicted-conversion surface in the two active factors A and B.
A, B = np.meshgrid(np.arange(-2, 2.01, 0.05), np.arange(-2, 2.01, 0.05))
y_hat = b[0] + b[1] * A + b[2] * B + b[4] * A * B

fig, ax = plt.subplots(figsize=(6.5, 5.5))
cs = ax.contourf(A, B, y_hat, levels=12, cmap="viridis")
lines = ax.contour(A, B, y_hat, levels=12, colors="white", linewidths=0.6, alpha=0.6)
ax.clabel(lines, inline=True, fontsize=8, fmt="%d")
fig.colorbar(cs, ax=ax, label="Predicted conversion, y")

# The four factorial corners, and a candidate next run at (1.5, 1.5).
ax.plot([-1, 1, -1, 1], [-1, -1, 1, 1], "ko", markersize=9, markerfacecolor="none",
        markeredgewidth=2)
ax.plot(1.5, 1.5, "k*", markersize=13)

ax.set_xlabel("A  [temperature, coded]", fontsize=12)
ax.set_ylabel("B  [pH, coded]", fontsize=12)
ax.set_title("Predicted contours of conversion in factors A and B", fontsize=12)
ax.set_aspect("equal")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig("chemical-reaction-contours.png", dpi=200)
