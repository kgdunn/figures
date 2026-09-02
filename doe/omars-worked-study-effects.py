"""The full second-order model fitted in one step, for the worked OMARS study.

At thirty runs the fifteen-term model can be fitted directly: an intercept, four main effects,
four quadratics and six two-factor interactions, on log titer with the cassette shift removed.
Each coefficient is drawn with its 95% confidence interval on the fifteen residual degrees of
freedom. The three terms the staged analysis selects are filled; the rest are hollow. The
script checks that exactly those three intervals exclude zero, which is what the chapter says.

Every number comes from omars_worked_study_common.py, which reproduces the chapter's study and
checks it against the values the chapter prints.

Reproducible; run from this directory to write the PNG alongside it.
"""
import itertools

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from omars_worked_study_common import BLUE, LABELS, NAMES, SPINE, VERMILION, model_matrix, study

S = study()
plan, C, result = S["plan"], S["C"], S["result"]

terms = [("m", j) for j in range(4)] + [("q", j) for j in range(4)]
terms += [("i", pair) for pair in itertools.combinations(range(4), 2)]
def lower(label):
    return label if label == "pH" else label.lower()


labels = [LABELS[NAMES[j]] if k == "m" else f"{LABELS[NAMES[j]]}²" if k == "q"
          else f"{LABELS[NAMES[j[0]]]} × {lower(LABELS[NAMES[j[1]]])}" for k, j in terms]
tags = [NAMES[j] if k == "m" else f"{NAMES[j]}^2" if k == "q" else f"{NAMES[j[0]]}:{NAMES[j[1]]}"
        for k, j in terms]

X = model_matrix(terms, C)
y = plan["log_titer_adj"].to_numpy()
beta = np.linalg.lstsq(X, y, rcond=None)[0]
df = len(y) - X.shape[1]
sigma2 = ((y - X @ beta) @ (y - X @ beta)) / df
se = np.sqrt(sigma2 * np.diag(np.linalg.inv(X.T @ X)))
half = stats.t.ppf(0.975, df) * se
coef, half, se = beta[1:], half[1:], se[1:]          # drop the intercept

selected = set(result.active_main_effects) | set(result.active_quadratics) | set(result.active_interactions)
significant = {t for t, c, h in zip(tags, coef, half) if abs(c) > h}
print(f"{df} residual df; significant at 5%: {sorted(significant)}; staged analysis: {sorted(selected)}")
if significant != selected:
    msg = "the one-step fit and the staged analysis disagree; the chapter says they agree"
    raise AssertionError(msg)

fig, ax = plt.subplots(figsize=(8.6, 5.0))
ypos = np.array([15, 14, 13, 12, 10, 9, 8, 7, 5, 4, 3, 2, 1, 0], float)   # gaps between groups

ax.axvline(0, color="0.6", lw=1.0, zorder=1)
for y_i, c, h, t in zip(ypos, coef, half, tags):
    filled = t in selected
    ax.plot([c - h, c + h], [y_i, y_i], color=BLUE if filled else "0.55", lw=2.0 if filled else 1.4, zorder=2)
    ax.plot([c], [y_i], "o", ms=8, color=BLUE if filled else "white", mec=BLUE if filled else "0.45",
            mew=1.5, zorder=3)
for y_i, label in zip(ypos, labels):
    ax.text(-0.005, y_i, label, ha="right", va="center", fontsize=10.5, color="0.2",
            transform=ax.get_yaxis_transform())
for y_gap, title in ((16.0, "Main effects"), (11.0, "Quadratics"), (6.0, "Two-factor interactions")):
    ax.text(0.0, y_gap, title, ha="left", va="center", fontsize=9.5, color="0.45",
            transform=ax.get_yaxis_transform(), style="italic")

ax.set_yticks([])
ax.set_ylim(-0.8, 16.5)
ax.set_xlabel(f"Coefficient on log titer, coded units, with a 95% interval on {df} df", fontsize=11.5)
ax.plot([], [], "o", color=BLUE, ms=8, label="selected by the staged analysis")
ax.plot([], [], "o", color="white", mec="0.45", mew=1.5, ms=8, label="not selected")
ax.legend(loc="upper right", fontsize=9.5, frameon=False)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.spines["bottom"].set_color(SPINE)
ax.tick_params(colors="0.25", labelsize=10)

fig.tight_layout()
fig.savefig("omars-worked-study-effects.png", dpi=300, facecolor="w", edgecolor="w",
            format=None, transparent=True)
print("saved figure")
