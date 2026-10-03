"""Where each candidate's inverted recipe falls relative to the studied factor ranges.

For each candidate, the inversion asks: what settings of the four continuous factors make it develop
colour like the reference goal (chromogen A at the centre point)? Three readings match the goal in
the 3-component *score* space, one per categorical coding; each can return a different recipe,
because a truncated PLS score is coding-dependent. The fourth reading matches the predicted *ten-point
curve* at full rank, which is coding-invariant (identical to ~1e-13 under any coding), and is
over-determined, so it returns the least-squares closest curve.

Rather than a hard reachable/unreachable verdict (a modest step outside the coded box is exactly the
extrapolation a designed experiment supports), each cell states the factors whose setting falls
outside the studied range, with the required value in real units and the bound it crosses in
brackets, so the reader sees how many factors leave the window and by how much.
Regenerates ``colour-coding-inversion.png``.
"""

# check-scripts: requires pyoptex -- the I-optimal colour design comes from pyoptex
import contextlib
import io

import matplotlib.pyplot as plt

from colour_case_study import (
    build_design,
    curve_match_inversion,
    fit_coding,
    goal_projection,
    invert_to_factors,
    simulate_curves,
)

design = build_design("i_optimal", budget=60)
curves = simulate_curves(design)
candidates = ["B", "C", "D", "E", "F"]
factors = ["concentration", "co_solvent", "pH", "temperature"]
RANGES = {"concentration": (2, 8, "umol/L"), "co_solvent": (5, 25, "%"),
          "pH": (4.0, 7.0, ""), "temperature": (15, 35, "degC")}
SHORT = {"concentration": "conc", "co_solvent": "co-solv", "pH": "pH", "temperature": "temp"}


def score_table(coding):
    with contextlib.redirect_stderr(io.StringIO()):
        pls, di = fit_coding(design, curves, coding)
    return invert_to_factors(pls, di, goal_projection(pls, di)["score"])


def cell_text(row):
    """Lines naming each out-of-range factor: 'name real > bound unit'. Empty -> within range."""
    lines = []
    for f in factors:
        coded = float(row[f"{f}_coded"])
        real = float(row[f])
        lo, hi, unit = RANGES[f]
        tail = f" {unit}" if unit else ""
        if coded > 1.0 + 1e-9:
            lines.append(f"{SHORT[f]} {real:.1f} > {hi:g}{tail}")
        elif coded < -1.0 - 1e-9:
            lines.append(f"{SHORT[f]} {real:.1f} < {lo:g}{tail}")
    return lines


curve_tbl = curve_match_inversion(design, curves, "sum")   # coding-invariant; "sum" == any coding
readings = [
    ("Score match\nsum coding", score_table("sum")),
    ("Score match\ntreatment coding", score_table("treatment")),
    ("Score match\ncell-means coding", score_table("cell_means")),
    ("Curve match\n(coding-invariant)", curve_tbl),
]

fig, ax = plt.subplots(figsize=(9.6, 5.6))
for j, (_, table) in enumerate(readings):
    for i, c in enumerate(candidates):
        y = len(candidates) - 1 - i
        ax.add_patch(plt.Rectangle((j - 0.47, y - 0.47), 0.94, 0.94, facecolor="none",
                                   edgecolor="0.8", linewidth=1.0))
        lines = cell_text(table.loc[c])
        if not lines:
            ax.text(j, y, "within range", ha="center", va="center", fontsize=8.5,
                    color="0.55", style="italic")
        else:
            ax.text(j, y, "\n".join(lines), ha="center", va="center", fontsize=7.3, color="0.15")

ax.set_xticks(range(len(readings)))
ax.set_xticklabels([r[0] for r in readings], fontsize=9)
ax.set_yticks([len(candidates) - 1 - i for i in range(len(candidates))])
ax.set_yticklabels([f"chromogen {c}" for c in candidates], fontsize=9)
ax.set_xlim(-0.6, len(readings) - 0.4)
ax.set_ylim(-0.6, len(candidates) - 0.4)
ax.set_title("Inverted recipes against the studied ranges, by how the inversion is posed",
             fontsize=11, loc="left")
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(length=0)
fig.tight_layout()
fig.savefig("colour-coding-inversion.png", dpi=300, facecolor="w", edgecolor="w", transparent=True)
print("saved colour-coding-inversion.png")
for label, table in readings:
    print(label.replace(chr(10), " "), "->",
          {c: (cell_text(table.loc[c]) or "within range") for c in candidates})
