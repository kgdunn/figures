"""Run matrix of a six-factor definitive screening design (DSD).

Illustrates the "Definitive screening designs" section of the Design and Analysis of
Experiments chapter. A six-factor DSD has 2k + 1 = 13 runs. The design is generated with
process_improve and shown in construction order [C; -C; 0]: a conference-matrix block C,
its sign-flipped mirror -C, and a single centre run of all zeros. Drawn this way the
foldover structure and the three coded levels (-1, 0, +1) of every factor are visible.

Reproducible; run from this directory to write the PNG alongside it.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from process_improve.experiments import Factor, generate_design

# Generate the six-factor DSD (13 runs; the centre run is embedded, so center_points is
# not used here).  generate_design returns the runs in randomized execution order.
factors = [Factor(name=c, low=-1, high=1) for c in "ABCDEF"]
dsd = generate_design(factors, design_type="dsd", random_seed=42)
levels = dsd.design[dsd.factor_names].to_numpy(dtype=float)

# Reorder the runs into the construction pattern [C; -C; 0] with a transparent mirror
# pairing: the rows whose first non-zero entry is +1 form C, their negations form -C,
# and the all-zero row is the centre run.
top = [i for i, r in enumerate(levels) if r.any() and r[r != 0][0] > 0]
mirror = [next(j for j, s in enumerate(levels) if np.array_equal(s, -levels[i])) for i in top]
centre = [i for i, r in enumerate(levels) if not r.any()]
order = top + mirror + centre
matrix = levels[order]

# Three discrete colours for the coded levels -1 / 0 / +1.
cmap = ListedColormap(["#1f5fa8", "#f2f2f2", "#c0392b"])
norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], cmap.N)

fig, ax = plt.subplots(figsize=(5.6, 6.4))
ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

# Light grid between the cells.
ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
ax.grid(which="minor", color="white", linewidth=1.5)
ax.tick_params(which="minor", length=0)

ax.set_xticks(range(matrix.shape[1]))
ax.set_xticklabels(dsd.factor_names)
ax.set_yticks(range(matrix.shape[0]))
ax.set_yticklabels(range(1, matrix.shape[0] + 1))
ax.set_xlabel("Factor")
ax.set_ylabel("Run (construction order)")

# Bracket the three blocks on the right-hand side.
k = len(top)
spans = [(0, k - 1, "$\\mathbf{C}$"),
         (k, 2 * k - 1, "$-\\mathbf{C}$"),
         (2 * k, 2 * k, "centre")]
x_b = matrix.shape[1] - 0.4
for lo, hi, label in spans:
    ax.annotate("", xy=(x_b, lo - 0.35), xytext=(x_b, hi + 0.35),
                xycoords="data", textcoords="data",
                arrowprops=dict(arrowstyle="-", color="0.35", lw=1.2),
                annotation_clip=False)
    ax.text(x_b + 0.35, (lo + hi) / 2, label, color="0.25", fontsize=10,
            ha="left", va="center", rotation=0)

ax.set_xlim(-0.5, matrix.shape[1] + 1.1)

# Legend mapping the three colours to the coded factor levels.
handles = [Patch(facecolor="#1f5fa8", edgecolor="0.6", label="$-1$"),
           Patch(facecolor="#f2f2f2", edgecolor="0.6", label="$0$"),
           Patch(facecolor="#c0392b", edgecolor="0.6", label="$+1$")]
ax.legend(handles=handles, title="Coded level", loc="upper center",
          bbox_to_anchor=(0.5, -0.08), ncol=3, frameon=False)

fig.tight_layout()
fig.savefig("dsd-run-matrix.png", dpi=300, facecolor="w", edgecolor="w",
            orientation="portrait", format=None, transparent=True)
print("saved figure")
