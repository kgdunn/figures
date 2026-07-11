"""Hotelling's T2 versus SPE for the interaction PLS model (3 components).

Each of the sixty runs is placed by its Hotelling's T2 (distance inside the model plane) and its
SPE (distance to the model plane). The vertical and horizontal dashed lines are the 95% limits from
``hotellings_t2_limit`` and ``spe_limit``; a run in the lower-left rectangle is within both. The
reference goal (chromogen A at the centre point, projected onto the model) is drawn as an asterisk
and sits well inside both limits, so the model's prediction can be trusted at that point.
Regenerates ``colour-pls-t2-spe.png``.
"""

import contextlib
import io

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from colour_case_study import COMPOUND_LEVELS, build_design, goal_projection, interaction_matrix, simulate_curves
from process_improve.multivariate.methods import PLS

design = build_design("i_optimal", budget=60)
curves = simulate_curves(design)
X_int, design_info = interaction_matrix(design)
with contextlib.redirect_stderr(io.StringIO()):
    pls = PLS(n_components=3, scale=True).fit(X_int, curves)

t2 = pls.hotellings_t2_.iloc[:, -1].to_numpy()      # per-run T2 at 3 components (correct)
spe = pls.spe_.iloc[:, -1].to_numpy()               # per-run SPE at 3 components (correct)
t2_lim = float(pls.hotellings_t2_limit(0.95))
spe_lim = float(pls.spe_limit(0.95))
goal = goal_projection(pls, design_info)

palette = ["#1f5fa8", "#c0392b", "#2e8b57", "#8e44ad", "#d68910", "#17a2b8"]
compound = design.design["compound"].to_numpy()

fig, ax = plt.subplots(figsize=(7.6, 5.4))
for c, colour in zip(COMPOUND_LEVELS, palette):
    m = compound == c
    ax.scatter(t2[m], spe[m], s=42, color=colour, edgecolor="w", linewidth=0.5,
               label="A (ref)" if c == "A" else c, zorder=3)

# 95% limits: vertical for T2, horizontal for SPE.
ax.axvline(t2_lim, color="0.4", ls="--", lw=1.1)
ax.axhline(spe_lim, color="0.4", ls="--", lw=1.1)
ax.text(t2_lim, ax.get_ylim()[1], f" T2 95% = {t2_lim:.1f}", color="0.35", fontsize=8,
        va="top", ha="left")
ax.text(ax.get_xlim()[1], spe_lim, f"SPE 95% = {spe_lim:.1f} ", color="0.35", fontsize=8,
        va="bottom", ha="right")

# Reference goal: chromogen A at the centre point, projected onto the model.
ax.scatter([goal["t2"]], [goal["spe"]], s=190, color="k", marker="*", zorder=5,
           label="goal: A at centre")

ax.set_xlabel("Hotelling's $T^2$ (3 components)")
ax.set_ylabel("SPE (3 components)")
ax.set_title("Model diagnostics: $T^2$ vs SPE, with 95% limits", fontsize=10, loc="left")
handles, labels = ax.get_legend_handles_labels()
handles.append(Line2D([], [], color="0.4", ls="--", lw=1.1, label="95% limits"))
ax.legend(handles=handles, frameon=False, fontsize=8, ncol=2, loc="upper right")
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig("colour-pls-t2-spe.png", dpi=300, facecolor="w", edgecolor="w", transparent=True)
print("saved colour-pls-t2-spe.png")
print(f"goal: T2={goal['t2']:.3f} SPE={goal['spe']:.3f} | limits T2={t2_lim:.2f} SPE={spe_lim:.2f}")
print(f"runs over T2: {int((t2 > t2_lim).sum())} | over SPE: {int((spe > spe_lim).sum())}")
