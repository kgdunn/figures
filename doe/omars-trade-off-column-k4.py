"""The four-factor column of the OMARS trade-off table, drawn on its own.

Illustrates "Choosing the run count" in the worked OMARS study. The capability staircase
(omars-capability-staircase.py) shows the whole table; this is one column of it, laid along
the run-count axis so it reads left to right as the budget grows, at every odd run count from
the definitive screening design to 31 runs. Each cell carries the largest model that run count
makes estimable and the error degrees of freedom left to test it. The cells past the
Box-Behnken design are blank, as in the staircase: every one of them repeats Full on more runs.

Every value is read from the library, so the figure cannot drift away from it.

Reproducible; run from this directory to write the PNG alongside it.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from process_improve.experiments import get_omars_trade_off_table_entry
from process_improve.experiments.omars_trade_off import (
    box_behnken_runs, definitive_screening_runs, omars_anchor_entry,
)

from omars_worked_study_common import SPINE, VERMILION

# Fills and inks as in omars-capability-staircase.py, so the two figures read the same.
FILLS = {"full": "#0072B2", "quad": "#56B4E9", "satd": "#E69F00", "none": "#F4F4F4", "bbd": "#009E73"}
INKS = {"full": "white", "quad": "#10334A", "satd": "#4A2F00", "bbd": "white"}

K = 4
dsd, bbd = definitive_screening_runs(K), box_behnken_runs(K)
runs = list(range(dsd, 33, 2))
frontier = K**2 + K + 1

fig, ax = plt.subplots(figsize=(8.6, 3.0))

for j, n in enumerate(runs):
    if n > bbd:
        ax.add_patch(Rectangle((j, 0), 1, 1, facecolor=FILLS["none"], edgecolor="white", lw=2.0))
        continue
    if n == bbd:
        entry, key, mark = omars_anchor_entry("bbd", K), "bbd", "BBD"
    else:
        entry = get_omars_trade_off_table_entry(n, K, display=False)
        key, mark = entry.capability, ("DSD" if n == dsd else "")
    ax.add_patch(Rectangle((j, 0), 1, 1, facecolor=FILLS[key], edgecolor="white", lw=2.0))
    ax.text(j + 0.5, 0.34, entry.tag, ha="center", va="center", fontsize=12, fontweight="bold",
            color=INKS[key])
    ax.text(j + 0.5, 0.70, f"df = {entry.error_df}", ha="center", va="center", fontsize=9.5,
            color=INKS[key])
    if mark:
        ax.text(j + 0.5, 0.90, mark, ha="center", va="center", fontsize=8.5, color=INKS[key])

# The estimability frontier, N = k^2 + k + 1, outlined as in the staircase.
j_front = runs.index(frontier)
ax.add_patch(Rectangle((j_front + 0.03, 0.03), 0.94, 0.94, fill=False, edgecolor=VERMILION,
                       lw=2.6, zorder=4))

# The blank cells past the Box-Behnken design, explained once across them.
j_bbd = runs.index(bbd)
ax.text((j_bbd + 1 + len(runs)) / 2, 0.5, "Full on\nmore runs", ha="center", va="center",
        fontsize=9.5, color="0.45")

# Where this study sits.
ax.annotate("", xy=(j_bbd + 0.5, 1.06), xytext=(j_bbd + 0.5, 1.36),
            arrowprops={"arrowstyle": "-|>", "color": "0.25", "lw": 1.3})
ax.text(j_bbd + 0.5, 1.42, "this study: 27 runs, plus 3 centre runs, is 30 batches",
        ha="center", va="top", fontsize=10, color="0.25")

ax.set_xlim(0, len(runs))
ax.set_ylim(1.9, 0)
ax.set_xticks([j + 0.5 for j in range(len(runs))])
ax.set_xticklabels([str(n) for n in runs], fontsize=11)
ax.set_yticks([])
ax.set_xlabel("Number of runs, $N$, with one centre run  (four factors)", fontsize=11.5)
ax.tick_params(length=0, colors="0.25")
for side in ("top", "right", "bottom", "left"):
    ax.spines[side].set_visible(False)

legend = [
    Patch(facecolor=FILLS["satd"], label="Satd: saturated, no error df"),
    Patch(facecolor=FILLS["quad"], label="Quad: main effects and quadratics"),
    Patch(facecolor=FILLS["full"], label="Full: adds all two-factor interactions"),
    Patch(facecolor=FILLS["bbd"], label="BBD: the Box-Behnken design"),
    Patch(facecolor="white", edgecolor=VERMILION, lw=2.2, label="Outlined: the estimability frontier"),
]
ax.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncols=3, fontsize=9.5,
          frameon=False, handlelength=1.3, columnspacing=1.4)

fig.tight_layout()
fig.savefig("omars-trade-off-column-k4.png", dpi=300, facecolor="w", edgecolor="w",
            format=None, transparent=True)
print("saved figure")
