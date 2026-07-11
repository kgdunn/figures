"""PLS score and W*-C loadings plot for the interaction model on the full profile.

This is the expanded model: the compound-by-factor interaction terms (sum-to-zero coding), fitted
to the whole ten-point colour-development curve rather than to the peak. It differs from
``colour-pls-scores-loadings.py`` (the main-effects model) in the factor block, and from the
coefficient comparison (``colour-coefficient-comparison.py``) in the response, which there was the
single peak.

Left: the first two PLS scores, one point per run, encoded three ways so a single panel carries the
run's compound, its pH, and its concentration. Colour is the chromogen; marker shape is the pH level
(circle low, up triangle high); marker size is the concentration level (small low, large high).
Right: the W* factor weights (24 model terms) and the C response-point weights (ten time points) on
the same axes. Each compound term (main effect or interaction) is coloured like its compound in the
score plot; the other factor terms are black; the time points are red. Regenerates
``colour-pls-interaction-scores-loadings.png``.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from patsy import dmatrix

from colour_case_study import CONT, COMPOUND_LEVELS, build_design, simulate_curves
from process_improve.multivariate.methods import PLS

design = build_design("i_optimal", budget=60)
curves = simulate_curves(design)

adf = design.design[["compound"] + list(CONT)].copy()
adf["compound"] = adf["compound"].astype(str)
rhs = ("C(compound, Sum)*co_solvent + C(compound, Sum)*pH "
       "+ C(compound, Sum)*temperature + concentration")
X_int = dmatrix(rhs, adf, return_type="dataframe").drop(columns=["Intercept"])

pls = PLS(n_components=3, scale=True).fit(X_int, curves)
scores = np.asarray(pls.scores_)
wstar = pls.direct_weights_          # X-space weights (24 model terms)
cweights = pls.y_weights_            # Y-space weights (ten time points t0..t9)
r2 = np.asarray(pls.r2_cumulative_).ravel()
r2_1 = r2[0]
r2_2 = r2[1] - r2[0]

palette = ["#1f5fa8", "#c0392b", "#2e8b57", "#8e44ad", "#d68910", "#17a2b8"]
compound_colour = dict(zip(COMPOUND_LEVELS, palette))
compound = adf["compound"].to_numpy()
pH = adf["pH"].to_numpy(float)
conc = adf["concentration"].to_numpy(float)

fig, (axS, axL) = plt.subplots(1, 2, figsize=(11.6, 5.4))

# --- Panel A: scores, colour = compound, shape = pH level, size = concentration level ---
# Circle for low pH (acidic), up triangle for high pH; small marker at low concentration, large
# marker at high concentration. Colour stays the chromogen so all three read at once.
for c, colour in zip(COMPOUND_LEVELS, palette):
    for lo_pH, marker in ((True, "o"), (False, "^")):
        for lo_conc, size in ((True, 34), (False, 96)):
            m = (compound == c) & ((pH < 0) == lo_pH) & ((conc < 0) == lo_conc)
            if m.any():
                axS.scatter(scores[m, 0], scores[m, 1], s=size, color=colour, marker=marker,
                            edgecolor="w", linewidth=0.5)
axS.axhline(0, color="0.7", lw=0.7)
axS.axvline(0, color="0.7", lw=0.7)
axS.set_xlabel(f"PLS score t1 (R2Y cumulative = {r2_1:.2f})")
axS.set_ylabel(f"PLS score t2 (+{r2_2:.2f})")
axS.set_title("(a) Score plot: colour = chromogen, shape = pH, size = concentration",
              fontsize=9.5, loc="left")

colour_handles = [Line2D([], [], marker="o", ls="", color=colour, markeredgecolor="w",
                         label="A (ref)" if c == "A" else c)
                  for c, colour in zip(COMPOUND_LEVELS, palette)]
enc_handles = [
    Line2D([], [], marker="o", ls="", color="0.35", markeredgecolor="w", label="low pH"),
    Line2D([], [], marker="^", ls="", color="0.35", markeredgecolor="w", label="high pH"),
    Line2D([], [], marker="o", ls="", color="0.35", markeredgecolor="w", markersize=5,
           label="low concentration"),
    Line2D([], [], marker="o", ls="", color="0.35", markeredgecolor="w", markersize=10,
           label="high concentration"),
]
leg1 = axS.legend(handles=colour_handles, frameon=False, fontsize=8, ncol=2, loc="lower left",
                  title="chromogen")
axS.add_artist(leg1)
axS.legend(handles=enc_handles, frameon=False, fontsize=8, loc="lower right")
axS.grid(alpha=0.2)


# --- Panel B: W* (24 model terms) and C (ten time points) loadings ---
def short(name):
    return name.replace("C(compound, Sum)[S.", "cmp").replace("]", "").replace("co_solvent", "cosolv")


def term_compound(name):
    """Compound letter if this is a compound main-effect or interaction term, else None."""
    return name.split("[S.")[1][0] if "[S." in name else None

wcolours = [compound_colour[term_compound(n)] if term_compound(n) else "black" for n in wstar.index]
axL.scatter(wstar.iloc[:, 0], wstar.iloc[:, 1], s=44, c=wcolours, marker="o",
            edgecolor="w", linewidth=0.5, zorder=3)
for name, (a, b) in zip(wstar.index, wstar.iloc[:, :2].to_numpy()):
    axL.annotate(short(name), (a, b), fontsize=6.5, color="0.2",
                 xytext=(3, 2), textcoords="offset points")

# Time points t0 -> t9: red circles, joined by a faint trajectory line.
tvals = cweights.iloc[:, :2].to_numpy()
axL.plot(tvals[:, 0], tvals[:, 1], color="#c0392b", lw=1.0, alpha=0.5, zorder=2)
axL.scatter(tvals[:, 0], tvals[:, 1], s=40, color="#c0392b", marker="o", edgecolor="w",
            linewidth=0.5, zorder=3)
for lbl in ("t0", "t4", "t9"):
    i = list(cweights.index).index(lbl)
    axL.annotate(lbl, tvals[i], fontsize=8, color="#7a2318", xytext=(4, -9),
                 textcoords="offset points")

axL.axhline(0, color="0.7", lw=0.7)
axL.axvline(0, color="0.7", lw=0.7)
axL.set_xlabel("weight on component 1")
axL.set_ylabel("weight on component 2")
axL.set_title("(b) Loadings: W* model terms and C response points", fontsize=9.5, loc="left")
axL.legend(handles=[
    Line2D([], [], marker="o", ls="", color="black", label="process factor (W*)"),
    Line2D([], [], marker="o", ls="", color="0.6", label="compound term, coloured by chromogen"),
    Line2D([], [], marker="o", ls="", color="#c0392b", label="time point (C)")],
    frameon=False, fontsize=8, loc="best")
axL.grid(alpha=0.2)

fig.tight_layout()
fig.savefig("colour-pls-interaction-scores-loadings.png", dpi=300, facecolor="w", edgecolor="w",
            transparent=True)
print("saved colour-pls-interaction-scores-loadings.png")
print("r2_cumulative (3 comp):", np.round(r2, 3))
