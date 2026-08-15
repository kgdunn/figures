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

Two standard designs are marked in place, on the row of their own run count: the definitive
screening design, the smallest member of the OMARS family, and the Box-Behnken design, among
the largest. Between them they show the span a column covers. The Box-Behnken cell closes its
column: every row below it would repeat Full on more runs, so those are left blank.

Every value is read from the library, get_omars_trade_off_table_entry for the budgets and
omars_anchor_entry for the Box-Behnken cells, so the figure cannot drift away from it.

Reproducible; run from this directory to write the PNG alongside it.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from process_improve.experiments import get_omars_trade_off_table_entry
from process_improve.experiments.omars_trade_off import (
    DEFAULT_FACTORS, DEFAULT_RUNS, box_behnken_runs, definitive_screening_runs, omars_anchor_entry,
)

# Okabe-Ito colourblind-safe palette, ordered by decreasing capability.
FILLS = {"full": "#0072B2", "quad": "#56B4E9", "satd": "#E69F00", "none": "#F4F4F4",
         "bbd": "#009E73"}
INKS = {"full": "white", "quad": "#10334A", "satd": "#4A2F00", "none": "#F4F4F4",
        "bbd": "white"}

factors = DEFAULT_FACTORS
dsd_runs = {k: definitive_screening_runs(k) for k in factors}
bbd_runs = {k: box_behnken_runs(k) for k in factors}

# A Box-Behnken design sits at a run count that is not one of the budgets, so the table carries
# a row for it. Sorting keeps every row in run-count order.
runs = sorted(set(DEFAULT_RUNS) | {n for n in bbd_runs.values() if n is not None})

cells = []
for n in runs:
    row = []
    for k in factors:
        bbd = bbd_runs[k]
        if bbd is not None and n > bbd:
            row.append(None)                       # the column has closed
        elif n == bbd:
            row.append(("bbd", omars_anchor_entry("bbd", k)))
        else:
            entry = get_omars_trade_off_table_entry(n, k, display=False)
            row.append((("dsd" if n == dsd_runs[k] else None), entry) if entry.exists else None)
    cells.append(row)

fig, ax = plt.subplots(figsize=(8.6, 11.0))

for i, row in enumerate(cells):
    for j, cell in enumerate(row):
        if cell is None:
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=FILLS["none"], edgecolor="white", lw=2.0))
            continue
        marker, entry = cell
        fill = FILLS["bbd"] if marker == "bbd" else FILLS[entry.capability]
        ink = INKS["bbd"] if marker == "bbd" else INKS[entry.capability]
        ax.add_patch(Rectangle((j, i), 1, 1, facecolor=fill, edgecolor="white", lw=2.0))
        ax.text(j + 0.5, i + 0.36, entry.tag, ha="center", va="center",
                fontsize=13, fontweight="bold", color=ink)
        below = f"df = {entry.error_df}" + (f"  |  {marker.upper()}" if marker else "")
        ax.text(j + 0.5, i + 0.72, below, ha="center", va="center", fontsize=10.5, color=ink)

# Outline the first Full cell in each column: that is the estimability frontier,
# N = k^2 + k + 1, the same staircase plotted in omars-estimability-frontier.py.
for j, k in enumerate(factors):
    first = next((i for i, n in enumerate(runs) if n >= k**2 + k + 1), None)
    if first is not None:
        ax.add_patch(Rectangle((j + 0.03, first + 0.03), 0.94, 0.94, fill=False,
                               edgecolor="#D55E00", lw=2.6, zorder=4))

ax.set_xlim(0, len(factors))
ax.set_ylim(len(runs), 0)
ax.set_xticks([j + 0.5 for j in range(len(factors))])
ax.set_xticklabels([f"$k$ = {k}" for k in factors], fontsize=12.5)
ax.set_yticks([i + 0.5 for i in range(len(runs))])
ax.set_yticklabels([str(n) for n in runs], fontsize=12.5)
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
    Patch(facecolor=FILLS["bbd"],
          label="BBD: the Box-Behnken design, which closes its column;\n"
                "every row below it repeats Full on more runs"),
    Patch(facecolor="white", edgecolor="white",
          label="DSD: marks the definitive screening design, the\n"
                "smallest member of the family"),
]
ax.legend(handles=legend, loc="upper left", bbox_to_anchor=(0.0, -0.02),
          fontsize=11, frameon=False, handlelength=1.4, labelspacing=0.9)

fig.tight_layout()
fig.savefig("omars-capability-staircase.png", dpi=300, facecolor="w", edgecolor="w",
            format=None, transparent=True)
print("saved figure")
