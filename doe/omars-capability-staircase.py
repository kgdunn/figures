"""The OMARS trade-off table drawn as a capability staircase.

Illustrates the "OMARS designs" section of the Design and Analysis of Experiments
chapter, and is the OMARS counterpart of the two-level trade-off table. A two-level
fractional factorial trades run count against resolution; an OMARS design has clean main
effects at every size, so resolution cannot be the currency. What varies instead is which
model the run budget makes estimable:

    Full  main effects, quadratics and all two-factor interactions jointly estimable
    Quad  main effects and pure quadratics, with error degrees of freedom to test them
    Satd  saturated (the definitive screening design size): no error degrees of freedom

The number in each cell is the error degrees of freedom left over. Blank cells are budgets
that are not a foldover design at all.

The first and last rows are anchors rather than budgets: the definitive screening design is
the smallest member of the OMARS family and the Box-Behnken design is among the largest, so
between them they show the span the family covers. A named design has its own run count in
each column, unlike a budget row where one count is read straight across, so the anchor cells
carry their run count with them.

Every value is read from the library, get_omars_trade_off_table_entry for the budgets and
_reference_entry for the anchors, so the figure cannot drift away from it.

Reproducible; run from this directory to write the PNG alongside it.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from process_improve.experiments import get_omars_trade_off_table_entry
from process_improve.experiments.omars_trade_off import DEFAULT_FACTORS, DEFAULT_RUNS, _reference_entry

# Okabe-Ito colourblind-safe palette, ordered by decreasing capability.
FILLS = {"full": "#0072B2", "quad": "#56B4E9", "satd": "#E69F00", "none": "#F4F4F4"}
INKS = {"full": "white", "quad": "#10334A", "satd": "#4A2F00", "none": "#F4F4F4"}

runs, factors = DEFAULT_RUNS, DEFAULT_FACTORS
cells = [[get_omars_trade_off_table_entry(n, k, display=False) for k in factors] for n in runs]

# Anchor rows. The definitive screening design is the smallest member of the OMARS family and
# the Box-Behnken design is among the largest, so they bracket the budgets and show the span the
# family covers. Their run counts change from column to column, so those go inside the cell.
anchors = [("DSD", [_reference_entry("dsd", k) for k in factors]),
           ("BBD", [_reference_entry("bbd", k) for k in factors])]
rows = [anchors[0], *((str(n), row) for n, row in zip(runs, cells)), anchors[1]]

fig, ax = plt.subplots(figsize=(8.2, 7.8))

for i, (_, row) in enumerate(rows):
    anchor = i in (0, len(rows) - 1)
    for j, cell in enumerate(row):
        capability = "none" if cell is None else cell.capability
        ax.add_patch(Rectangle((j, i), 1, 1, facecolor=FILLS[capability],
                               edgecolor="white", lw=2.0))
        if cell is None or not cell.exists:
            continue
        ink = INKS[capability]
        if anchor:
            # The run count belongs in the cell here, because an anchor row is a named design
            # whose size changes column by column rather than a single budget read across.
            ax.text(j + 0.5, i + 0.36, cell.tag, ha="center", va="center",
                    fontsize=13.5, fontweight="bold", color=ink)
            ax.text(j + 0.5, i + 0.72, f"{cell.n_runs} runs, df = {cell.error_df}",
                    ha="center", va="center", fontsize=10, color=ink)
        else:
            ax.text(j + 0.5, i + 0.36, cell.tag, ha="center", va="center",
                    fontsize=13.5, fontweight="bold", color=ink)
            ax.text(j + 0.5, i + 0.72, f"df = {cell.error_df}", ha="center", va="center",
                    fontsize=11, color=ink)

# Separate the anchor rows from the budget rows: the budgets are a single run count read across
# every column, the anchors are a named design whose size changes column by column.
for edge in (1, len(rows) - 1):
    ax.plot([0, len(factors)], [edge, edge], color="#4A5560", lw=1.8, zorder=5)

# Outline the first Full cell in each column: that is the estimability frontier,
# N = k^2 + k + 1, the same staircase plotted in omars-estimability-frontier.py. The offset of
# one skips the DSD row, which sits above the budgets.
for j, k in enumerate(factors):
    first = next((i for i, n in enumerate(runs) if n >= k**2 + k + 1), None)
    if first is not None:
        ax.add_patch(Rectangle((j + 0.03, first + 1.03), 0.94, 0.94, fill=False,
                               edgecolor="#D55E00", lw=2.6, zorder=4))

ax.set_xlim(0, len(factors))
ax.set_ylim(len(rows), 0)
ax.set_xticks([j + 0.5 for j in range(len(factors))])
ax.set_xticklabels([f"$k$ = {k}" for k in factors], fontsize=12.5)
ax.set_yticks([i + 0.5 for i in range(len(rows))])
ax.set_yticklabels([label for label, _ in rows], fontsize=12.5)
ax.xaxis.set_ticks_position("top")
ax.xaxis.set_label_position("top")
ax.set_xlabel("Number of factors", fontsize=13, labelpad=10)
ax.set_ylabel("Number of runs", fontsize=13)
ax.tick_params(length=0)
for side in ("top", "right", "bottom", "left"):
    ax.spines[side].set_visible(False)

legend = [
    Patch(facecolor=FILLS["full"], label="Full: main effects, quadratics and all\n"
                                         "two-factor interactions, jointly estimable"),
    Patch(facecolor=FILLS["quad"], label="Quad: main effects and pure quadratics,\n"
                                         "with error degrees of freedom to test them"),
    Patch(facecolor=FILLS["satd"], label="Satd: saturated, so estimates but no inference"),
    Patch(facecolor=FILLS["none"], edgecolor="0.8",
          label="Not a foldover design at this run count"),
    Patch(facecolor="white", edgecolor="#D55E00", lw=2.2,
          label="Outlined: the estimability frontier, $N = k^2 + k + 1$"),
    Line2D([0], [0], color="#4A5560", lw=1.8,
           label="Ruled off: the DSD and Box-Behnken rows, named designs\n"
                 "whose run count changes from column to column"),
]
ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(0.0, -0.02),
          fontsize=11, frameon=False, handlelength=1.4, labelspacing=0.9)

fig.tight_layout()
fig.savefig("omars-capability-staircase.png", dpi=300, facecolor="w", edgecolor="w",
            format=None, transparent=True)
print("saved figure")
