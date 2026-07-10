"""PLS score plot and W*-C loadings plot for the profile case study.

Left: the first two PLS score vectors (one point per run), coloured by chromogen, showing how the
runs separate in the latent space. Right: the W* (direct X-weights, the process factors) and C
(Y-weights, the ten development time points) on the same axes, so the relationship between the
factors and the response is read directly. Regenerates ``colour-pls-scores-loadings.png``.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from colour_case_study import COMPOUND_LEVELS, build_design, fit_profile_pls, model_matrix, simulate_curves

design = build_design("i_optimal", budget=60)
curves = simulate_curves(design)
pls = fit_profile_pls(design, curves, n_components=5)

scores = np.asarray(pls.scores_)
wstar = pls.direct_weights_          # X-space weights (factors), rows indexed by factor name
cweights = pls.y_weights_            # Y-space weights (time points), rows indexed by t0..t9
r2 = np.asarray(pls.r2_cumulative_).ravel()
r2_1 = r2[0]
r2_2 = r2[1] - r2[0]

palette = ["#1f5fa8", "#c0392b", "#2e8b57", "#8e44ad", "#d68910", "#17a2b8"]
compound = design.design["compound"].to_numpy()

fig, (axS, axL) = plt.subplots(1, 2, figsize=(11.4, 5.2))

# --- Panel A: PLS scores, coloured by compound ---
for c, colour in zip(COMPOUND_LEVELS, palette):
    m = compound == c
    label = "A (ref)" if c == "A" else c
    axS.scatter(scores[m, 0], scores[m, 1], s=42, color=colour, edgecolor="w", linewidth=0.5,
                label=label)
axS.axhline(0, color="0.7", lw=0.7)
axS.axvline(0, color="0.7", lw=0.7)
axS.set_xlabel(f"PLS score t1 (R2Y cumulative = {r2_1:.2f})")
axS.set_ylabel(f"PLS score t2 (+{r2_2:.2f})")
axS.set_title("(a) Score plot: runs in the latent space", fontsize=10, loc="left")
axS.legend(frameon=False, fontsize=8, ncol=2, title="chromogen")
axS.grid(alpha=0.2)

# --- Panel B: W* (factors) and C (time points) loadings ---
# Factors: label continuous ones by name and compound indicators by their letter.
def short(name):
    return name.replace("cmp_", "") if name.startswith("cmp_") else name

axL.scatter(wstar.iloc[:, 0], wstar.iloc[:, 1], s=55, color="#1f5fa8", marker="o",
            edgecolor="w", linewidth=0.5, zorder=3)
for name, (a, b) in zip(wstar.index, wstar.iloc[:, :2].to_numpy()):
    axL.annotate(short(name), (a, b), fontsize=8, color="#12406e",
                 xytext=(4, 3), textcoords="offset points")

# Time points: a trajectory t0 -> t9, coloured light-to-dark, showing early vs late response.
tvals = cweights.iloc[:, :2].to_numpy()
axL.plot(tvals[:, 0], tvals[:, 1], color="#c0392b", lw=1.0, alpha=0.6, zorder=2)
axL.scatter(tvals[:, 0], tvals[:, 1], s=40, color="#c0392b", marker="s", edgecolor="w",
            linewidth=0.5, zorder=3)
for lbl in ("t0", "t4", "t9"):
    i = list(cweights.index).index(lbl)
    axL.annotate(lbl, tvals[i], fontsize=8, color="#7a2318", xytext=(4, -9),
                 textcoords="offset points")

axL.axhline(0, color="0.7", lw=0.7)
axL.axvline(0, color="0.7", lw=0.7)
axL.set_xlabel("weight on component 1")
axL.set_ylabel("weight on component 2")
axL.set_title("(b) Loadings: W* factors and C response points", fontsize=10, loc="left")
axL.legend(handles=[Line2D([], [], marker="o", ls="", color="#1f5fa8", label="factor (W*)"),
                    Line2D([], [], marker="s", ls="", color="#c0392b", label="time point (C)")],
           frameon=False, fontsize=8, loc="best")
axL.grid(alpha=0.2)

fig.tight_layout()
fig.savefig("colour-pls-scores-loadings.png", dpi=300, facecolor="w", edgecolor="w",
            transparent=True)
print("saved colour-pls-scores-loadings.png")
