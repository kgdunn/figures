"""Hotelling's T2 versus SPE for the interaction PLS model (3 components).

Each of the sixty runs is placed by its Hotelling's T2 (distance inside the model plane) and its
SPE (distance to the model plane). The vertical and horizontal dashed lines are the 95% limits from
``hotellings_t2_limit`` and ``spe_limit``; a run in the lower-left rectangle is within both. The
reference goal (chromogen A at the centre point, projected onto the model) is drawn as an asterisk
and sits well inside both limits, so the model's prediction can be trusted at that point.

Each run carries the same four-way encoding as the interaction score plot
(``colour-pls-interaction-scores-loadings.py``), so the same run can be tracked across the two
figures: colour = chromogen, marker shape = pH (down triangle low, circle high), marker size grows
with concentration, and the marker fill marks the co-solvent (open low, filled high).
Regenerates ``colour-pls-t2-spe.png``.
"""

import contextlib
import io

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from colour_case_study import (
    CONT, COMPOUND_LEVELS, build_design, goal_projection, interaction_matrix, simulate_curves,
)
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
adf = design.design[["compound"] + list(CONT)].copy()
compound = adf["compound"].astype(str).to_numpy()
pH = adf["pH"].to_numpy(float)
conc = adf["concentration"].to_numpy(float)
cosolv = adf["co_solvent"].to_numpy(float)
size = 22 + 78 * (conc + 1) / 2                     # coded concentration in [-1, 1] -> marker area

fig, ax = plt.subplots(figsize=(7.8, 5.6))
# Four encodings on one point, matching the interaction score plot: colour = compound; shape = pH
# (down triangle low, circle high); marker area grows with concentration; co-solvent is an open
# (outline-only) marker at the low setting and a filled marker at the high setting.
for c, colour in zip(COMPOUND_LEVELS, palette):
    for lo_pH, marker in ((True, "v"), (False, "o")):
        lo_s = (compound == c) & ((pH < 0) == lo_pH) & (cosolv < 0)
        hi_s = (compound == c) & ((pH < 0) == lo_pH) & (cosolv >= 0)
        if lo_s.any():                              # low co-solvent: open marker, coloured outline
            ax.scatter(t2[lo_s], spe[lo_s], s=size[lo_s], facecolors="none", edgecolors=colour,
                       marker=marker, linewidth=1.3, zorder=3)
        if hi_s.any():                              # high co-solvent: filled marker
            ax.scatter(t2[hi_s], spe[hi_s], s=size[hi_s], color=colour, marker=marker,
                       edgecolor="w", linewidth=0.5, zorder=3)

# 95% limits: vertical for T2, horizontal for SPE.
ax.axvline(t2_lim, color="0.4", ls="--", lw=1.1)
ax.axhline(spe_lim, color="0.4", ls="--", lw=1.1)
ax.text(t2_lim, ax.get_ylim()[1], f" T2 95% = {t2_lim:.1f}", color="0.35", fontsize=8,
        va="top", ha="left")
ax.text(ax.get_xlim()[1], spe_lim, f"SPE 95% = {spe_lim:.1f} ", color="0.35", fontsize=8,
        va="bottom", ha="right")

# Reference goal: chromogen A at the centre point, projected onto the model.
ax.scatter([goal["t2"]], [goal["spe"]], s=190, color="k", marker="*", zorder=5)

ax.set_xlabel("Hotelling's $T^2$ (3 components)")
ax.set_ylabel("SPE (3 components)")
ax.set_title("Model diagnostics: $T^2$ vs SPE, with 95% limits", fontsize=10, loc="left")

colour_handles = [Line2D([], [], marker="o", ls="", color=colour, markeredgecolor="w",
                         label="A (ref)" if c == "A" else c)
                  for c, colour in zip(COMPOUND_LEVELS, palette)]
enc_handles = [
    Line2D([], [], marker="v", ls="", color="0.35", markeredgecolor="w", label="low pH"),
    Line2D([], [], marker="o", ls="", color="0.35", markeredgecolor="w", label="high pH"),
    Line2D([], [], marker="None", ls="", label=r"size $\propto$ concentration"),
    Line2D([], [], marker="None", ls="", label="open = low co-solvent, filled = high"),
    Line2D([], [], marker="*", ls="", color="k", markersize=11, label="goal: A at centre"),
    Line2D([], [], color="0.4", ls="--", lw=1.1, label="95% limits"),
]
leg1 = ax.legend(handles=colour_handles, frameon=False, fontsize=8, ncol=2, loc="upper right",
                 title="chromogen")
ax.add_artist(leg1)
ax.legend(handles=enc_handles, frameon=False, fontsize=8, loc="center right", handletextpad=0.6)
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig("colour-pls-t2-spe.png", dpi=300, facecolor="w", edgecolor="w", transparent=True)
print("saved colour-pls-t2-spe.png")
print(f"goal: T2={goal['t2']:.3f} SPE={goal['spe']:.3f} | limits T2={t2_lim:.2f} SPE={spe_lim:.2f}")
print(f"runs over T2: {int((t2 > t2_lim).sum())} | over SPE: {int((spe > spe_lim).sum())}")
