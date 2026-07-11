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

# --- Panel A: scores, four encodings ---
# colour = compound; shape = pH (down triangle low, circle high); marker area grows with the coded
# concentration; co-solvent is open (outline-only) markers at the low setting and filled at the
# high setting. The size and co-solvent keys are stated as text in the lower-right legend.
cosolv = adf["co_solvent"].to_numpy(float)
size = 22 + 78 * (conc + 1) / 2          # coded concentration in [-1, 1] -> marker area
for c, colour in zip(COMPOUND_LEVELS, palette):
    for lo_pH, marker in ((True, "v"), (False, "o")):
        lo_s = (compound == c) & ((pH < 0) == lo_pH) & (cosolv < 0)
        hi_s = (compound == c) & ((pH < 0) == lo_pH) & (cosolv >= 0)
        if lo_s.any():                     # low co-solvent: open marker, coloured outline
            axS.scatter(scores[lo_s, 0], scores[lo_s, 1], s=size[lo_s], facecolors="none",
                        edgecolors=colour, marker=marker, linewidth=1.3)
        if hi_s.any():                     # high co-solvent: filled marker
            axS.scatter(scores[hi_s, 0], scores[hi_s, 1], s=size[hi_s], color=colour, marker=marker,
                        edgecolor="w", linewidth=0.5)
axS.axhline(0, color="0.7", lw=0.7)
axS.axvline(0, color="0.7", lw=0.7)
axS.set_xlabel(f"PLS score t1 (R2Y cumulative = {r2_1:.2f})")
axS.set_ylabel(f"PLS score t2 (+{r2_2:.2f})")
axS.set_title("(a) Score plot: colour = chromogen, shape = pH", fontsize=9.5, loc="left")

colour_handles = [Line2D([], [], marker="o", ls="", color=colour, markeredgecolor="w",
                         label="A (ref)" if c == "A" else c)
                  for c, colour in zip(COMPOUND_LEVELS, palette)]
enc_handles = [
    Line2D([], [], marker="v", ls="", color="0.35", markeredgecolor="w", label="low pH"),
    Line2D([], [], marker="o", ls="", color="0.35", markeredgecolor="w", label="high pH"),
    Line2D([], [], marker="None", ls="", label=r"size $\propto$ concentration"),
    Line2D([], [], marker="None", ls="", label="open marker = low co-solvent"),
    Line2D([], [], marker="None", ls="", label="filled marker = high co-solvent"),
]
leg1 = axS.legend(handles=colour_handles, frameon=False, fontsize=8, ncol=2, loc="lower left",
                  title="chromogen")
axS.add_artist(leg1)
axS.legend(handles=enc_handles, frameon=False, fontsize=8, loc="lower right", handletextpad=0.6)
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

# Fix the axis extent (over both point clouds, with padding) before the label repulsion, so the
# transforms stay stable while labels move and none is pushed off the panel.
allxy = np.vstack([wstar.iloc[:, :2].to_numpy(), cweights.iloc[:, :2].to_numpy()])
padx = 0.12 * (allxy[:, 0].max() - allxy[:, 0].min())
pady = 0.10 * (allxy[:, 1].max() - allxy[:, 1].min())
axL.set_xlim(allxy[:, 0].min() - padx, allxy[:, 0].max() + padx)
axL.set_ylim(allxy[:, 1].min() - pady, allxy[:, 1].max() + pady)

# The 24 term labels overlap badly where the interaction terms cluster. Place each label with a
# leader line and push the labels apart with a small deterministic repulsion pass (a light-weight
# stand-in for adjustText: no randomness, so the layout is reproducible).
# Start each label a little up and to the right of its marker (not centred on it), so even an
# isolated label with nothing to repel it still sits clear of the point.
_xr, _yr = axL.get_xlim(), axL.get_ylim()
_ox, _oy = 0.03 * (_xr[1] - _xr[0]), 0.035 * (_yr[1] - _yr[0])
anns = [axL.annotate(short(name), xy=(a, b), xytext=(a + _ox, b + _oy), textcoords="data",
                     fontsize=6.3, color="0.15", zorder=4,
                     arrowprops=dict(arrowstyle="-", lw=0.4, color="0.55", shrinkA=0, shrinkB=3))
        for name, (a, b) in zip(wstar.index, wstar.iloc[:, :2].to_numpy())]


def repel_labels(ax, annotations, anchor_disp, iterations=600, step=1.3, spring=0.02):
    """Separate overlapping labels by repelling them, with a spring back to each anchor so the
    leader lines stay short, and a clamp keeping every label inside the axes. Deterministic."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    axbb = ax.get_window_extent(rend)
    for _ in range(iterations):
        boxes = [a.get_window_extent(rend) for a in annotations]
        any_move = False
        for i, a in enumerate(annotations):
            bi = boxes[i]
            cix, ciy = (bi.x0 + bi.x1) / 2, (bi.y0 + bi.y1) / 2
            hw, hh = (bi.x1 - bi.x0) / 2, (bi.y1 - bi.y0) / 2
            rx = ry = 0.0
            for j, bj in enumerate(boxes):
                if i != j and bi.overlaps(bj):
                    ox, oy = cix - (bj.x0 + bj.x1) / 2, ciy - (bj.y0 + bj.y1) / 2
                    norm = (ox * ox + oy * oy) ** 0.5 or 1.0
                    rx += ox / norm
                    ry += oy / norm
            ax_, ay_ = anchor_disp[i]
            for px, py in anchor_disp:                       # keep labels off the markers
                ox, oy = cix - px, ciy - py
                d2 = ox * ox + oy * oy
                if d2 < 18 ** 2:
                    norm = d2 ** 0.5 or 1.0
                    rx += 0.6 * ox / norm
                    ry += 0.6 * oy / norm
            rnorm = (rx * rx + ry * ry) ** 0.5
            mx = (step * rx / rnorm if rnorm else 0.0) + spring * (ax_ - cix)
            my = (step * ry / rnorm if rnorm else 0.0) + spring * (ay_ - ciy)
            if abs(mx) > 0.05 or abs(my) > 0.05:
                nx = min(max(cix + mx, axbb.x0 + hw), axbb.x1 - hw)
                ny = min(max(ciy + my, axbb.y0 + hh), axbb.y1 - hh)
                px0, py0 = ax.transData.transform(a.get_position())
                a.set_position(inv.transform((px0 + (nx - cix), py0 + (ny - ciy))))
                any_move = True
        if not any_move:
            break


repel_labels(axL, anns, axL.transData.transform(wstar.iloc[:, :2].to_numpy()))

# Time points t0 -> t9: red circles (no connecting line; it clashed with the label leader lines).
tvals = cweights.iloc[:, :2].to_numpy()
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
