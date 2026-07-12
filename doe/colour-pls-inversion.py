"""Model-inversion result: the continuous-factor settings that compensate each chromogen.

Model inversion finds, for each candidate chromogen, the minimum-adjustment continuous-factor
settings whose score matches the reference goal (chromogen A at the centre point). This plots those
settings in coded units, one row per factor, one marker per chromogen. The band between the dashed
lines at -1 and +1 is the studied range; a marker outside it is a setting the experiment never
explored. B, C and D can be compensated within the ranges; E and F require a pH (and, for F, a
concentration) beyond the studied window, so they cannot be made to match the reference there.
Regenerates ``colour-pls-inversion.png``.
"""

import contextlib
import io

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from colour_case_study import CONT, build_design, goal_projection, interaction_matrix, invert_to_factors, simulate_curves
from process_improve.multivariate.methods import PLS

design = build_design("i_optimal", budget=60)
curves = simulate_curves(design)
X_int, design_info = interaction_matrix(design)
with contextlib.redirect_stderr(io.StringIO()):
    pls = PLS(n_components=3, scale=True).fit(X_int, curves)

goal = goal_projection(pls, design_info)
table = invert_to_factors(pls, design_info, goal["score"])
candidates = ["B", "C", "D", "E", "F"]
factors = list(CONT)                              # concentration, co_solvent, pH, temperature
labels = {"concentration": "concentration", "co_solvent": "co-solvent", "pH": "pH",
          "temperature": "temperature"}
palette = {"B": "#c0392b", "C": "#2e8b57", "D": "#8e44ad", "E": "#d68910", "F": "#17a2b8"}
_JIT = {"B": -0.18, "C": -0.09, "D": 0.0, "E": 0.09, "F": 0.18}   # spread markers within a row

fig, ax = plt.subplots(figsize=(8.0, 4.6))
yrows = {f: len(factors) - 1 - i for i, f in enumerate(factors)}   # first factor on top

# Studied range band and the nominal (reference A) line.
ax.axvspan(-1, 1, color="#e8eef5", zorder=0)
ax.axvline(0, color="0.75", lw=0.9, zorder=1)
for xline in (-1, 1):
    ax.axvline(xline, color="0.55", ls="--", lw=1.0, zorder=1)

# Light dashed separators between the factor rows, so each factor reads as its own lane.
for ysep in np.arange(len(factors) - 1) + 0.5:
    ax.axhline(ysep, color="0.8", ls="--", lw=0.7, zorder=1)

for f in factors:
    y = yrows[f]
    for c in candidates:
        val = float(table.loc[c, f"{f}_coded"])
        inside = abs(val) <= 1.0 + 1e-9
        ax.scatter([val], [y + _JIT[c]], s=58, color=palette[c],
                   edgecolor="w" if inside else "k", linewidth=0.6 if inside else 1.3, zorder=4)
        if not inside:
            ax.annotate(f"{c}", (val, y + _JIT[c]), fontsize=7.5, color="k",
                        xytext=(5, 0), textcoords="offset points", va="center", zorder=5)

ax.set_yticks([yrows[f] for f in factors])
ax.set_yticklabels([labels[f] for f in factors])
ax.set_ylim(-0.6, len(factors) - 0.4)
ax.set_xlim(-1.6, 2.0)
ax.set_xlabel("coded factor setting to match the reference goal (0 = nominal centre)")
ax.set_title("Compensation to match the reference: inverted factor settings per chromogen",
             fontsize=10, loc="left")
ax.text(0, -0.5, "studied range", color="0.45", fontsize=8, ha="center", va="bottom")

legend = [Line2D([], [], marker="o", ls="", color=palette[c], markeredgecolor="w", label=c)
          for c in candidates]
legend.append(Line2D([], [], marker="o", ls="", color="0.5", markeredgecolor="k", markeredgewidth=1.3,
                     label="outside studied range"))
ax.legend(handles=legend, frameon=False, fontsize=8, ncol=6, loc="lower center",
          bbox_to_anchor=(0.5, -0.30), columnspacing=1.1, handletextpad=0.3)
ax.grid(axis="x", alpha=0.2)
fig.tight_layout()
fig.savefig("colour-pls-inversion.png", dpi=300, facecolor="w", edgecolor="w", transparent=True)
print("saved colour-pls-inversion.png")
