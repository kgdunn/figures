"""Geometry of the central composite variants and their neighbours at k = 2.

A row of coded-space pictures mirroring the diagram in the central composite design post:
the three CCD variants (circumscribed/rotatable, face-centred, inscribed) plus the Doehlert
(uniform shell) design, all in two factors so the point layout is visible. Factorial (cube)
runs are squares, axial (star) runs are stars, the centre run is a filled circle; the dashed
square marks the +/-1 coded region.

This is the visual companion to the FDS figures: circumscribed pushes the stars outside
+/-1, inscribed pulls the whole design inside it, face-centred puts the stars on the faces,
and Doehlert spreads seven points evenly on a hexagon.

Reproducible; run from this directory: writes ``ccd-variants-geometry.png``.
"""

import matplotlib.pyplot as plt
import numpy as np

from ccd_variants_designs import ccd_matrix, doehlert_matrix

ALPHA_ROT_2D = float((2**2) ** 0.25)  # rotatable alpha for k = 2 is sqrt(2)


def split_ccd(mat: np.ndarray):
    """Return (factorial rows, axial rows, centre rows) of a k = 2 CCD matrix."""
    factorial = mat[np.all(np.abs(np.abs(mat) - np.abs(mat).max(axis=1, keepdims=True)) < 1e-9, axis=1) & (np.count_nonzero(mat, axis=1) == 2)]
    centre = mat[np.count_nonzero(mat, axis=1) == 0]
    axial = mat[np.count_nonzero(mat, axis=1) == 1]
    return factorial, axial, centre


panels = [
    ("CCD circumscribed\n(rotatable, $\\alpha$ = 1.414)", ccd_matrix(ALPHA_ROT_2D, k=2), "ccd"),
    ("CCD face-centred\n($\\alpha$ = 1)", ccd_matrix(1.0, k=2), "ccd"),
    ("CCD inscribed\n(stars on $\\pm$1)", ccd_matrix(ALPHA_ROT_2D, k=2) / ALPHA_ROT_2D, "ccd"),
    ("Doehlert\n(uniform shell)", doehlert_matrix(k=2), "shell"),
]

fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.6))
for ax, (title, mat, kind) in zip(axes, panels):
    ax.axhline(0, color="#cccccc", lw=0.8, zorder=0)
    ax.axvline(0, color="#cccccc", lw=0.8, zorder=0)
    ax.plot([-1, 1, 1, -1, -1], [-1, -1, 1, 1, -1], ls="--", color="#999999", lw=1.0, zorder=1)
    if kind == "ccd":
        fac, ax_pts, cen = split_ccd(mat)
        ax.scatter(fac[:, 0], fac[:, 1], marker="s", s=90, facecolor="#2e8b57", edgecolor="k", zorder=3, label="factorial")
        ax.scatter(ax_pts[:, 0], ax_pts[:, 1], marker="*", s=230, facecolor="#d35400", edgecolor="k", zorder=3, label="axial (star)")
        ax.scatter(cen[:, 0], cen[:, 1], marker="o", s=70, facecolor="#1f5fa8", edgecolor="k", zorder=4, label="centre")
    else:
        shell = mat[np.count_nonzero(mat, axis=1) > 0]
        cen = mat[np.count_nonzero(mat, axis=1) == 0]
        ax.scatter(shell[:, 0], shell[:, 1], marker="h", s=110, facecolor="#c0392b", edgecolor="k", zorder=3, label="shell")
        ax.scatter(cen[:, 0], cen[:, 1], marker="o", s=70, facecolor="#1f5fa8", edgecolor="k", zorder=4, label="centre")
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-1.75, 1.75)
    ax.set_xticks([-1, 0, 1])
    ax.set_yticks([-1, 0, 1])
    ax.legend(loc="upper right", fontsize=6.5, frameon=True, framealpha=0.9)

fig.suptitle("Central composite variants and the Doehlert design in two factors (coded units)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig("ccd-variants-geometry.png", dpi=300, facecolor="w", edgecolor="w", transparent=True)
print("saved ccd-variants-geometry.png")
